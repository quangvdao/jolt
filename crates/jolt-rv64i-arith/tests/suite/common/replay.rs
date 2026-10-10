//! Reconstructs execution from committed bits, without invoking the interpreter
//! or the committed-bits generator.

use std::collections::BTreeMap;

use jolt_rv64i_arith::{
    BaseWords, BitsRow, Bytecode, BytecodeError, BytecodeRow, Layout, RowFailure, RowSystem,
    WitnessRow,
};

/// Sparse RAM omits zero words; each snapshot's PC is the next instruction's PC.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct State {
    pub pc: u64,
    pub registers: [u64; 32],
    pub ram: BTreeMap<u64, u64>,
}

impl State {
    /// Zero registers and RAM at the specified entry PC.
    pub fn new(pc: u64) -> Self {
        Self {
            pc,
            ..Self::default()
        }
    }

    /// Stores nonzero words and removes zero words, preserving sparse equality.
    pub fn set_ram_word(&mut self, index: u64, value: u64) {
        if value == 0 {
            let _ = self.ram.remove(&index);
        } else {
            let _ = self.ram.insert(index, value);
        }
    }
}

/// Post-cycle snapshots and the final state (the initial state for an empty trace).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trace {
    pub cycles: Vec<State>,
    pub final_state: State,
}

/// A failed execution obligation or a failed local constraint row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    LayoutMismatch,
    BytecodeIndex { cycle: usize, index: u64 },
    InvalidRow { cycle: usize, index: u64 },
    EntryPcMismatch { entry: u64, fetched: u64 },
    NonzeroX0 { value: u64 },
    RamIndexOutOfRange { index: u64 },
    FinalPc(BytecodeError),
    Rows { cycle: usize, error: RowFailure },
}

/// Reconstructs register reads from the pre-state and reads the RAM word at the
/// committed index. `RdWriteValue = old_rd XOR Inc` off stores and `old_rd` on
/// stores. The caller supplies an authenticated row and a valid successor PC;
/// this helper leaves those two obligations to the enclosing enumeration.
pub fn base_words(
    layout: &Layout,
    row: &BytecodeRow,
    bits: &BitsRow,
    registers: &[u64; 32],
    ram: &BTreeMap<u64, u64>,
    next_pc: u64,
) -> BaseWords {
    let read_register = |register| registers.get(usize::from(register)).copied().unwrap_or(0);
    let store = row.variant.is_some_and(|variant| variant.is_store());
    BaseWords {
        rs1_value: read_register(row.rs1),
        rs2_value: read_register(row.rs2),
        rd_write_value: read_register(row.rd) ^ if store { 0 } else { layout.inc(bits) },
        ram_read_value: ram.get(&layout.ram_index(bits)).copied().unwrap_or(0),
        next_pc,
    }
}

fn fetch<'a>(
    layout: &Layout,
    bytecode: &'a Bytecode,
    bits: &BitsRow,
    cycle: usize,
) -> Result<&'a BytecodeRow, ReplayError> {
    let index = layout.bytecode_index(bits);
    let row = usize::try_from(index)
        .ok()
        .and_then(|index| bytecode.rows().get(index))
        .ok_or(ReplayError::BytecodeIndex { cycle, index })?;
    if row.variant.is_none() {
        return Err(ReplayError::InvalidRow { cycle, index });
    }
    Ok(row)
}

/// Discharges bytecode selection by fetching each committed index; successors
/// come from the next fetched row or the checked `final_pc`. Register and RAM
/// writes use the two XOR identities on this replay's own state. Each transition
/// then checks a canonical witness. Neither oracle outcomes nor `CycleFacts`
/// participate in state reconstruction.
pub fn replay(
    layout: &Layout,
    bytecode: &Bytecode,
    bits: &[BitsRow],
    initial: &State,
    final_pc: u64,
) -> Result<Trace, ReplayError> {
    if layout.log_K_bytecode() != bytecode.log_K()
        || layout.lowest_address() != bytecode.lowest_address()
    {
        return Err(ReplayError::LayoutMismatch);
    }
    let x0 = initial.registers.first().copied().unwrap_or(0);
    if x0 != 0 {
        return Err(ReplayError::NonzeroX0 { value: x0 });
    }
    for &index in initial.ram.keys() {
        if index >= 1_u64 << layout.log_K_ram() {
            return Err(ReplayError::RamIndexOutOfRange { index });
        }
    }
    let _ = bytecode
        .final_pc_index(final_pc)
        .map_err(ReplayError::FinalPc)?;
    let system = RowSystem::new(layout);
    let mut state = initial.clone();
    state.ram.retain(|_, value| *value != 0);
    let mut cycles = Vec::with_capacity(bits.len());
    for (cycle, committed) in bits.iter().enumerate() {
        let row = fetch(layout, bytecode, committed, cycle)?;
        if cycle == 0 && row.pc != initial.pc {
            return Err(ReplayError::EntryPcMismatch {
                entry: initial.pc,
                fetched: row.pc,
            });
        }
        let next_pc = if let Some(next) = bits.get(cycle + 1) {
            fetch(layout, bytecode, next, cycle + 1)?.pc
        } else {
            final_pc
        };
        let base = base_words(
            layout,
            row,
            committed,
            &state.registers,
            &state.ram,
            next_pc,
        );
        let witness = WitnessRow::compute(layout, row, &base, committed);
        system
            .check(&witness)
            .map_err(|error| ReplayError::Rows { cycle, error })?;
        if row.variant.is_some_and(|variant| variant.is_store()) {
            let index = layout.ram_index(committed);
            state.set_ram_word(index, base.ram_read_value ^ layout.inc(committed));
        } else if row.rd != 0 {
            if let Some(destination) = state.registers.get_mut(usize::from(row.rd)) {
                *destination = base.rd_write_value;
            }
        }
        state.pc = next_pc;
        cycles.push(state.clone());
    }
    if bits.is_empty() && initial.pc != final_pc {
        return Err(ReplayError::EntryPcMismatch {
            entry: initial.pc,
            fetched: final_pc,
        });
    }
    Ok(Trace {
        cycles,
        final_state: state,
    })
}
