//! Dense test-oracle views in low-variable-first table order. State tables
//! replay the witness's XOR update contract, except final RAM, which reads its retained state.

use crate::{error::Rv64iProverError, plane::Rv64iWitness};
use jolt_field::{One, Zero, F128};
use jolt_poly::Polynomial;
use jolt_rv64i_arith::{BytecodeColumn, BytecodeRow, Chunk, BITS_COLUMNS};
use jolt_rv64i_verifier::points::{chunk as evaluate_chunk, eq_index, lift};
use std::collections::BTreeMap;

/// One of the five replayed cycle words, extended over its 64 bit indices.
#[derive(Clone, Copy, Debug)]
pub enum BaseWord {
    /// First source-register value before the cycle.
    Rs1Value,
    /// Second source-register value before the cycle.
    Rs2Value,
    /// Destination-register value before the cycle.
    RdPreValue,
    /// RAM word selected by the committed address, including word zero without an access.
    RamReadValue,
    /// Successor PC, including the supplied final PC on the last cycle.
    NextPC,
}
/// A fetched instruction's one-hot kind selector; absent kinds contribute zero.
#[derive(Clone, Copy, Debug)]
pub enum Selector {
    /// Instruction variant in the 64-entry domain.
    Variant,
    /// Shift kind in the eight-entry domain.
    ShiftKind,
    /// Memory access kind in the sixteen-entry domain.
    AccessKind,
    /// Comparison key kind in the eight-entry domain.
    KeyKind,
}
/// A fetched row's register selector in the 32-entry domain; absent operands select zero.
#[derive(Clone, Copy, Debug)]
pub enum RegisterSelector {
    /// First source-register selector.
    Rs1,
    /// Second source-register selector.
    Rs2,
    /// Destination-register selector.
    Rd,
}

/// Extends one committed column in low-variable-first cycle order; columns outside the packed row are zero.
pub fn bits_column(witness: &Rv64iWitness, column: usize) -> Polynomial<F128> {
    Polynomial::new(
        witness
            .bits
            .iter()
            .map(|r| {
                if r.get(column / 64)
                    .is_some_and(|w| (w >> (column % 64)) & 1 != 0)
                {
                    F128::one()
                } else {
                    F128::zero()
                }
            })
            .collect(),
    )
}
/// Extends the increment at a six-coordinate low-variable-first bit point over cycle variables.
/// Returns a point-dimension error unless the bit point has six coordinates.
pub fn inc(witness: &Rv64iWitness, point: &[F128]) -> Result<Polynomial<F128>, Rv64iProverError> {
    Ok(Polynomial::new(
        witness
            .bits
            .iter()
            .map(|r| lift(witness.layout.inc(r), point))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}
/// Extends a chunk at its low-variable-first digit point over cycle variables.
/// Returns a point-dimension or missing-column error from the canonical chunk evaluator.
pub fn chunk(
    witness: &Rv64iWitness,
    chunk: Chunk,
    point: &[F128],
) -> Result<Polynomial<F128>, Rv64iProverError> {
    let mut columns = [F128::zero(); BITS_COLUMNS];
    Ok(Polynomial::new(
        witness
            .bits
            .iter()
            .map(|r| {
                for (y, value) in columns.iter_mut().enumerate() {
                    *value = if r.get(y / 64).is_some_and(|w| (w >> (y % 64)) & 1 != 0) {
                        F128::one()
                    } else {
                        F128::zero()
                    };
                }
                evaluate_chunk(chunk, point, &columns)
            })
            .collect::<Result<Vec<_>, _>>()?,
    ))
}
/// Extends a replayed word at a six-coordinate low-variable-first bit point over cycle variables.
/// Returns a point-dimension error unless the bit point has six coordinates.
pub fn base_word(
    witness: &Rv64iWitness,
    word: BaseWord,
    point: &[F128],
) -> Result<Polynomial<F128>, Rv64iProverError> {
    Ok(Polynomial::new(
        witness
            .words
            .iter()
            .map(|w| {
                lift(
                    match word {
                        BaseWord::Rs1Value => w.rs1_value,
                        BaseWord::Rs2Value => w.rs2_value,
                        BaseWord::RdPreValue => w.rd_pre_value,
                        BaseWord::RamReadValue => w.ram_read_value,
                        BaseWord::NextPC => w.next_pc,
                    },
                    point,
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    ))
}
fn fetched(witness: &Rv64iWitness, cycle: usize) -> Result<&BytecodeRow, Rv64iProverError> {
    let bits = witness.bits.get(cycle).ok_or(Rv64iProverError::RowCount {
        rows: witness.bits.len(),
    })?;
    let index = witness.layout.bytecode_index(bits);
    usize::try_from(index)
        .ok()
        .and_then(|i| witness.bytecode.rows().get(i))
        .filter(|r| r.variant.is_some())
        .ok_or(Rv64iProverError::InvalidBytecode { cycle, index })
}
/// Extends a fetched row's 64-bit column at a six-coordinate low-variable-first bit point over cycles.
/// The caller selects one of its four word columns; invalid fetches and bit dimensions return errors.
pub fn bytecode_word(
    witness: &Rv64iWitness,
    column: BytecodeColumn,
    point: &[F128],
) -> Result<Polynomial<F128>, Rv64iProverError> {
    Ok(Polynomial::new(
        (0..witness.bits.len())
            .map(|j| Ok(lift(fetched(witness, j)?.column(column), point)?))
            .collect::<Result<Vec<_>, Rv64iProverError>>()?,
    ))
}
/// Extends a kind selector at a low-variable-first kind point over cycle variables.
/// Returns an error for an invalid fetch or an index outside the point's domain.
pub fn selector(
    witness: &Rv64iWitness,
    selector: Selector,
    point: &[F128],
) -> Result<Polynomial<F128>, Rv64iProverError> {
    Ok(Polynomial::new(
        (0..witness.bits.len())
            .map(|j| {
                let row = fetched(witness, j)?;
                let value = row.variant.and_then(|v| match selector {
                    Selector::Variant => Some(v.index()),
                    Selector::ShiftKind => v.shift().map(|s| s.kind as usize),
                    Selector::AccessKind => v.access().and_then(|a| a.kind).map(|k| k as usize),
                    Selector::KeyKind => v.key_kind().map(|k| k as usize),
                });
                Ok(match value {
                    Some(index) => eq_index(point, index)?,
                    None => F128::zero(),
                })
            })
            .collect::<Result<Vec<_>, Rv64iProverError>>()?,
    ))
}
/// Extends the fetched instruction's branch flag over low-variable-first cycle variables.
/// Returns an error when a cycle fetches an invalid bytecode row.
pub fn branch(witness: &Rv64iWitness) -> Result<Polynomial<F128>, Rv64iProverError> {
    Ok(Polynomial::new(
        (0..witness.bits.len())
            .map(|j| {
                Ok(
                    if fetched(witness, j)?
                        .variant
                        .is_some_and(|v| v.branch().is_some())
                    {
                        F128::one()
                    } else {
                        F128::zero()
                    },
                )
            })
            .collect::<Result<Vec<_>, Rv64iProverError>>()?,
    ))
}
/// Extends the fetched instruction's store flag over low-variable-first cycle variables.
/// Returns an error when a cycle fetches an invalid bytecode row.
pub fn store(witness: &Rv64iWitness) -> Result<Polynomial<F128>, Rv64iProverError> {
    Ok(Polynomial::new(
        (0..witness.bits.len())
            .map(|j| {
                Ok(
                    if fetched(witness, j)?.variant.is_some_and(|v| v.is_store()) {
                        F128::one()
                    } else {
                        F128::zero()
                    },
                )
            })
            .collect::<Result<Vec<_>, Rv64iProverError>>()?,
    ))
}
fn register(row: &BytecodeRow, selector: RegisterSelector) -> u8 {
    match selector {
        RegisterSelector::Rs1 => row.rs1,
        RegisterSelector::Rs2 => row.rs2,
        RegisterSelector::Rd => row.rd,
    }
}
/// Builds the register selector table with five low-variable-first address variables before cycle variables.
/// Returns an error when a cycle fetches an invalid bytecode row.
pub fn register_selector(
    witness: &Rv64iWitness,
    selector: RegisterSelector,
) -> Result<Polynomial<F128>, Rv64iProverError> {
    let mut values = Vec::with_capacity(32 * witness.bits.len());
    for j in 0..witness.bits.len() {
        let selected = usize::from(register(fetched(witness, j)?, selector));
        values.extend((0..32).map(|k| {
            if k == selected {
                F128::one()
            } else {
                F128::zero()
            }
        }));
    }
    Ok(Polynomial::new(values))
}
/// Fixes a register selector's low-variable-first address point and leaves its cycle variables.
/// Returns an error for an invalid fetch or an index outside the point's domain.
pub fn register_selector_at(
    witness: &Rv64iWitness,
    selector: RegisterSelector,
    point: &[F128],
) -> Result<Polynomial<F128>, Rv64iProverError> {
    Ok(Polynomial::new(
        (0..witness.bits.len())
            .map(|j| {
                Ok(eq_index(
                    point,
                    usize::from(register(fetched(witness, j)?, selector)),
                )?)
            })
            .collect::<Result<Vec<_>, Rv64iProverError>>()?,
    ))
}
/// Builds the RAM selector table with low-variable-first address variables before cycle variables.
/// Returns an error if the RAM domain cannot be represented on this host.
pub fn ram_ra(witness: &Rv64iWitness) -> Result<Polynomial<F128>, Rv64iProverError> {
    let count = domain(witness)?;
    let mut values = Vec::with_capacity(count * witness.bits.len());
    for bits in witness.bits.iter() {
        let index = witness.layout.ram_index(bits);
        values.extend((0..count).map(|k| {
            if k as u64 == index {
                F128::one()
            } else {
                F128::zero()
            }
        }));
    }
    Ok(Polynomial::new(values))
}
/// Fixes a RAM selector's low-variable-first address point and leaves its cycle variables.
/// Returns an error for an address outside the point's domain.
pub fn ram_ra_at(
    witness: &Rv64iWitness,
    point: &[F128],
) -> Result<Polynomial<F128>, Rv64iProverError> {
    Ok(Polynomial::new(
        witness
            .bits
            .iter()
            .map(|r| eq_index(point, witness.layout.ram_index(r) as usize))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}
fn domain(witness: &Rv64iWitness) -> Result<usize, Rv64iProverError> {
    Rv64iWitness::ram_words(&witness.layout)
}
/// Builds the replayed register pre-state at a fixed six-coordinate low-variable-first bit point.
/// Address variables precede cycles; invalid fetches, register selectors or bit dimensions return errors.
pub fn registers_val(
    witness: &Rv64iWitness,
    point: &[F128],
) -> Result<Polynomial<F128>, Rv64iProverError> {
    let mut registers = [0_u64; 32];
    let mut values = Vec::with_capacity(32 * witness.bits.len());
    for (j, bits) in witness.bits.iter().enumerate() {
        for word in registers {
            values.push(lift(word, point)?);
        }
        let row = fetched(witness, j)?;
        if !row.variant.is_some_and(|v| v.is_store()) {
            *registers
                .get_mut(usize::from(row.rd))
                .ok_or(Rv64iProverError::Register { register: row.rd })? ^=
                witness.layout.inc(bits);
        }
    }
    Ok(Polynomial::new(values))
}
/// Builds the replayed RAM pre-state at a fixed six-coordinate low-variable-first bit point.
/// Address variables precede cycles; invalid fetches, RAM dimensions or bit dimensions return errors.
pub fn ram_val(
    witness: &Rv64iWitness,
    point: &[F128],
) -> Result<Polynomial<F128>, Rv64iProverError> {
    let count = domain(witness)?;
    let mut ram: BTreeMap<_, _> = witness.initial_ram.iter().copied().collect();
    let mut values = Vec::with_capacity(count * witness.bits.len());
    for (j, bits) in witness.bits.iter().enumerate() {
        for k in 0..count {
            values.push(lift(ram.get(&(k as u64)).copied().unwrap_or(0), point)?);
        }
        if fetched(witness, j)?.variant.is_some_and(|v| v.is_store()) {
            let index = witness.layout.ram_index(bits);
            *ram.entry(index).or_default() ^= witness.layout.inc(bits);
        }
    }
    Ok(Polynomial::new(values))
}
/// Extends retained final RAM over low-variable-first address variables at a fixed bit point.
/// Returns an error for a final-RAM length inconsistent with the layout or a bit point without six coordinates.
pub fn ram_val_final(
    witness: &Rv64iWitness,
    point: &[F128],
) -> Result<Polynomial<F128>, Rv64iProverError> {
    let expected = domain(witness)?;
    if witness.final_ram.len() != expected {
        return Err(Rv64iProverError::FinalRamLength {
            expected,
            found: witness.final_ram.len(),
        });
    }
    Ok(Polynomial::new(
        witness
            .final_ram
            .iter()
            .map(|word| lift(*word, point))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}
