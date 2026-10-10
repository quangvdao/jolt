//! Replayed witness data: XOR increments start from zero registers and canonical
//! initial RAM. `from_bits` and `from_facts` establish the pre-state reads and
//! fetched-row successors; direct construction owes that same contract.

use crate::error::Rv64iProverError;
#[cfg(feature = "test-utils")]
use common::{constants::RAM_START_ADDRESS, jolt_device::MemoryLayout};
use jolt_field::JoltField;
use jolt_kernels::WitnessPlane;
use jolt_rv64i_arith::{BaseWords, BitsBuilder, BitsRow, Bytecode, CycleFacts, Layout};
#[cfg(feature = "test-utils")]
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CycleWords {
    pub rs1_value: u64,
    pub rs2_value: u64,
    pub rd_pre_value: u64,
    pub ram_read_value: u64,
    pub next_pc: u64,
}
impl CycleWords {
    /// Forms the destination post-value from the replay's pre-value and increment.
    pub fn base_words(&self, store: bool, inc: u64) -> BaseWords {
        BaseWords {
            rs1_value: self.rs1_value,
            rs2_value: self.rs2_value,
            rd_write_value: self.rd_pre_value ^ if store { 0 } else { inc },
            ram_read_value: self.ram_read_value,
            next_pc: self.next_pc,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Rv64iWitness {
    pub layout: Layout,
    pub bytecode: Arc<Bytecode>,
    pub bits: Arc<[BitsRow]>,
    pub words: Arc<[CycleWords]>,
    pub initial_ram: Vec<(u64, u64)>,
    pub final_pc: u64,
}
impl Rv64iWitness {
    pub fn from_bits(
        layout: Layout,
        bytecode: Arc<Bytecode>,
        bits: Arc<[BitsRow]>,
        initial_ram: Vec<(u64, u64)>,
        final_pc: u64,
    ) -> Result<Self, Rv64iProverError> {
        if !bits.len().is_power_of_two() {
            return Err(Rv64iProverError::RowCount { rows: bits.len() });
        }
        let _ = BitsBuilder::new(&layout, &bytecode)?;
        let _ = bytecode.final_pc_index(final_pc)?;
        let mut previous = None;
        for &(index, value) in &initial_ram {
            if value == 0
                || index >= 1_u64 << layout.log_K_ram()
                || previous.is_some_and(|p| p >= index)
            {
                return Err(Rv64iProverError::InitialRam { index });
            }
            previous = Some(index);
        }
        let mut ram: BTreeMap<_, _> = initial_ram.iter().copied().collect();
        let mut registers = [0_u64; 32];
        let mut words: Vec<CycleWords> = Vec::with_capacity(bits.len());
        for (cycle, committed) in bits.iter().enumerate() {
            for chunk in layout.chunks() {
                if chunk.stored(committed).count_ones() > 1 {
                    return Err(Rv64iProverError::MultipleIndicators {
                        cycle,
                        start: chunk.start(),
                    });
                }
            }
            let index = layout.bytecode_index(committed);
            let row = usize::try_from(index)
                .ok()
                .and_then(|i| bytecode.rows().get(i))
                .filter(|r| r.variant.is_some())
                .ok_or(Rv64iProverError::InvalidBytecode { cycle, index })?;
            if let Some(previous) = words.last_mut() {
                previous.next_pc = row.pc;
            }
            let read = |register: u8| {
                registers
                    .get(usize::from(register))
                    .copied()
                    .ok_or(Rv64iProverError::Register { register })
            };
            let ram_index = layout.ram_index(committed);
            words.push(CycleWords {
                rs1_value: read(row.rs1)?,
                rs2_value: read(row.rs2)?,
                rd_pre_value: read(row.rd)?,
                ram_read_value: ram.get(&ram_index).copied().unwrap_or(0),
                next_pc: final_pc,
            });
            let inc = layout.inc(committed);
            if row.variant.is_some_and(|v| v.is_store()) {
                let value = ram.get(&ram_index).copied().unwrap_or(0) ^ inc;
                if value == 0 {
                    let _ = ram.remove(&ram_index);
                } else {
                    let _ = ram.insert(ram_index, value);
                }
            } else {
                *registers
                    .get_mut(usize::from(row.rd))
                    .ok_or(Rv64iProverError::Register { register: row.rd })? ^= inc;
            }
        }
        Ok(Self {
            layout,
            bytecode,
            bits,
            words: words.into(),
            initial_ram,
            final_pc,
        })
    }
    /// Derives all value words by replay rather than copying reads from facts.
    pub fn from_facts(
        layout: Layout,
        bytecode: Arc<Bytecode>,
        facts: &[CycleFacts],
        initial_ram: Vec<(u64, u64)>,
    ) -> Result<Self, Rv64iProverError> {
        let final_pc = facts
            .last()
            .ok_or(Rv64iProverError::RowCount { rows: 0 })?
            .next_pc;
        let mut bits = vec![[0; 4]; facts.len()];
        BitsBuilder::new(&layout, &bytecode)?.fill(facts, &mut bits)?;
        Self::from_bits(layout, bytecode, bits.into(), initial_ram, final_pc)
    }
    #[cfg(feature = "test-utils")]
    pub fn synthetic(
        seed: u64,
        layout: Layout,
        bytecode: Arc<Bytecode>,
        memory_layout: &MemoryLayout,
        initial_ram: Vec<(u64, u64)>,
        log_T: usize,
    ) -> Result<Self, Rv64iProverError> {
        let count = 1_usize
            .checked_shl(u32::try_from(log_T).unwrap_or(u32::MAX))
            .ok_or(Rv64iProverError::TraceDimension { log_T })?;
        let valid: Vec<_> = bytecode
            .rows()
            .iter()
            .enumerate()
            .filter(|(_, r)| r.variant.is_some())
            .collect();
        let store = valid
            .iter()
            .find(|(_, r)| r.variant.is_some_and(|v| v.is_store()))
            .ok_or(Rv64iProverError::MissingStore)?
            .0;
        let termination = memory_layout.remapped_word_address(memory_layout.termination)?;
        let io_start = memory_layout.remapped_word_address(memory_layout.input_start)?;
        let io_end = memory_layout.remapped_word_address(RAM_START_ADDRESS)?;
        let output_start = memory_layout.remapped_word_address(memory_layout.output_start)?;
        let output_end = memory_layout.remapped_word_address(memory_layout.output_end)?;
        let limit = 1_u64 << layout.log_K_ram();
        let ordinary = (0..io_start.min(limit))
            .chain(output_start..output_end.min(limit))
            .chain(io_end..limit)
            .next()
            .ok_or(Rv64iProverError::StoreAddress)?;
        let termination_value = initial_ram
            .iter()
            .find(|(i, _)| *i == termination)
            .map_or(0, |(_, v)| *v);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut bits = Vec::with_capacity(count);
        for cycle in 0..count {
            let (index, row) = if cycle + 1 == count {
                (
                    store,
                    bytecode
                        .rows()
                        .get(store)
                        .ok_or(Rv64iProverError::MissingStore)?,
                )
            } else {
                *valid
                    .get(rng.gen_range(0..valid.len()))
                    .ok_or(Rv64iProverError::MissingStore)?
            };
            let mut committed = rng.gen::<BitsRow>();
            layout.write_bytecode_index(&mut committed, index as u64)?;
            let address = if cycle + 1 == count {
                termination
            } else if row.variant.is_some_and(|v| v.is_store()) {
                ordinary
            } else if row.variant.is_some_and(|v| v.access().is_some()) {
                rng.gen_range(0..limit)
            } else {
                0
            };
            layout.write_ram_index(&mut committed, address)?;
            layout.write_pos(&mut committed, rng.gen_range(0..64))?;
            if cycle + 1 == count {
                committed[0] = termination_value ^ 1;
            }
            bits.push(committed);
        }
        let final_pc = valid
            .get(rng.gen_range(0..valid.len()))
            .ok_or(Rv64iProverError::MissingStore)?
            .1
            .pc;
        Self::from_bits(layout, bytecode, bits.into(), initial_ram, final_pc)
    }
}
pub struct Rv64iPlane;
impl<F: JoltField> WitnessPlane<F> for Rv64iPlane {
    type Ref<'w>
        = &'w Rv64iWitness
    where
        F: 'w;
}
