//! Dense test-oracle views in low-variable-first table order. State tables
//! replay the same XOR update contract as the witness and never read fact values.

use crate::{error::Rv64iProverError, plane::Rv64iWitness};
use jolt_field::{Ring, Zero, F128};
use jolt_poly::Polynomial;
use jolt_rv64i_arith::{BytecodeColumn, BytecodeRow, Chunk};
use jolt_rv64i_verifier::points::{chunk as evaluate_chunk, eq_index, lift};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub enum BaseWord {
    Rs1Value,
    Rs2Value,
    RdPreValue,
    RamReadValue,
    NextPC,
}
#[derive(Clone, Copy, Debug)]
pub enum Selector {
    Variant,
    ShiftKind,
    AccessKind,
    KeyKind,
}
#[derive(Clone, Copy, Debug)]
pub enum RegisterSelector {
    Rs1,
    Rs2,
    Rd,
}

/// A column is extended only in the cycle variables.
pub fn bits_column(witness: &Rv64iWitness, column: usize) -> Polynomial<F128> {
    Polynomial::new(
        witness
            .bits
            .iter()
            .map(|r| {
                F128::from_u64(u64::from(
                    r.get(column / 64)
                        .is_some_and(|w| (w >> (column % 64)) & 1 != 0),
                ))
            })
            .collect(),
    )
}
pub fn inc(witness: &Rv64iWitness, point: &[F128]) -> Result<Polynomial<F128>, Rv64iProverError> {
    Ok(Polynomial::new(
        witness
            .bits
            .iter()
            .map(|r| lift(witness.layout.inc(r), point))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}
pub fn chunk(
    witness: &Rv64iWitness,
    chunk: Chunk,
    point: &[F128],
) -> Result<Polynomial<F128>, Rv64iProverError> {
    Ok(Polynomial::new(
        witness
            .bits
            .iter()
            .map(|r| {
                let columns: Vec<_> = (0..256)
                    .map(|y| {
                        F128::from_u64(u64::from(
                            r.get(y / 64).is_some_and(|w| (w >> (y % 64)) & 1 != 0),
                        ))
                    })
                    .collect();
                evaluate_chunk(chunk, point, &columns)
            })
            .collect::<Result<Vec<_>, _>>()?,
    ))
}
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
/// The column must be one of the fetched row's four 64-bit words.
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
pub fn branch(witness: &Rv64iWitness) -> Result<Polynomial<F128>, Rv64iProverError> {
    Ok(Polynomial::new(
        (0..witness.bits.len())
            .map(|j| {
                Ok(F128::from_u64(u64::from(
                    fetched(witness, j)?
                        .variant
                        .is_some_and(|v| v.branch().is_some()),
                )))
            })
            .collect::<Result<Vec<_>, Rv64iProverError>>()?,
    ))
}
pub fn store(witness: &Rv64iWitness) -> Result<Polynomial<F128>, Rv64iProverError> {
    Ok(Polynomial::new(
        (0..witness.bits.len())
            .map(|j| {
                Ok(F128::from_u64(u64::from(
                    fetched(witness, j)?.variant.is_some_and(|v| v.is_store()),
                )))
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
pub fn register_selector(
    witness: &Rv64iWitness,
    selector: RegisterSelector,
) -> Result<Polynomial<F128>, Rv64iProverError> {
    let mut values = Vec::with_capacity(32 * witness.bits.len());
    for j in 0..witness.bits.len() {
        let selected = usize::from(register(fetched(witness, j)?, selector));
        values.extend((0..32).map(|k| F128::from_u64(u64::from(k == selected))));
    }
    Ok(Polynomial::new(values))
}
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
pub fn ram_ra(witness: &Rv64iWitness) -> Result<Polynomial<F128>, Rv64iProverError> {
    let count = domain(witness)?;
    let mut values = Vec::with_capacity(count * witness.bits.len());
    for bits in witness.bits.iter() {
        let index = witness.layout.ram_index(bits);
        values.extend((0..count).map(|k| F128::from_u64(u64::from(k as u64 == index))));
    }
    Ok(Polynomial::new(values))
}
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
    1_usize
        .checked_shl(u32::try_from(witness.layout.log_K_ram()).unwrap_or(u32::MAX))
        .ok_or(Rv64iProverError::TraceDimension {
            log_T: witness.layout.log_K_ram(),
        })
}
/// Address variables precede the cycle variables; the bit point is fixed.
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
pub fn ram_val_final(
    witness: &Rv64iWitness,
    point: &[F128],
) -> Result<Polynomial<F128>, Rv64iProverError> {
    let count = domain(witness)?;
    let mut ram: BTreeMap<_, _> = witness.initial_ram.iter().copied().collect();
    for (j, bits) in witness.bits.iter().enumerate() {
        if fetched(witness, j)?.variant.is_some_and(|v| v.is_store()) {
            let index = witness.layout.ram_index(bits);
            *ram.entry(index).or_default() ^= witness.layout.inc(bits);
        }
    }
    Ok(Polynomial::new(
        (0..count)
            .map(|k| lift(ram.get(&(k as u64)).copied().unwrap_or(0), point))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}
