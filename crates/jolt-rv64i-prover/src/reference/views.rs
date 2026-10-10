//! Dense test-oracle views in low-variable-first table order. State tables
//! replay the witness's XOR update contract, except final RAM, which reads its retained state.

use crate::{error::Rv64iProverError, plane::Rv64iWitness};
use jolt_field::{JoltField, One, Zero, F128};
use jolt_poly::Polynomial;
use jolt_rv64i_arith::{BytecodeColumn, BytecodeRow, Chunk};
use jolt_rv64i_verifier::points::{eq_index, equality_table, ChunkWeights, PointsError, WordLift};

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
    Ok(inc_with_lift(witness, &WordLift::new(point)?))
}
/// Extends every cycle increment using weights prepared once for its bit point.
pub fn inc_with_lift(witness: &Rv64iWitness, lift: &WordLift<F128>) -> Polynomial<F128> {
    let mut values = Vec::with_capacity(witness.bits.len());
    for bits in witness.bits.iter() {
        values.push(lift.evaluate(witness.layout.inc(bits)));
    }
    Polynomial::new(values)
}
/// Extends a chunk at its low-variable-first digit point over cycle variables.
/// Returns a point-dimension or missing-column error from the canonical chunk evaluator.
pub fn chunk(
    witness: &Rv64iWitness,
    chunk: Chunk,
    point: &[F128],
) -> Result<Polynomial<F128>, Rv64iProverError> {
    chunk_in_field(witness, chunk, point)
}
/// Extends stored chunk indicators directly in the requested field, with one prepared digit table.
pub fn chunk_in_field<F: JoltField>(
    witness: &Rv64iWitness,
    chunk: Chunk,
    point: &[F],
) -> Result<Polynomial<F>, Rv64iProverError> {
    let weights = ChunkWeights::new(chunk, point)?;
    let mut values = Vec::with_capacity(witness.bits.len());
    for bits in witness.bits.iter() {
        values.push(weights.evaluate_packed(bits));
    }
    Ok(Polynomial::new(values))
}
/// Extends a replayed word using weights prepared once for its bit point.
pub fn base_word_with_lift(
    witness: &Rv64iWitness,
    word: BaseWord,
    lift: &WordLift<F128>,
) -> Polynomial<F128> {
    let mut values = Vec::with_capacity(witness.words.len());
    for words in witness.words.iter() {
        values.push(lift.evaluate(match word {
            BaseWord::Rs1Value => words.rs1_value,
            BaseWord::Rs2Value => words.rs2_value,
            BaseWord::RdPreValue => words.rd_pre_value,
            BaseWord::RamReadValue => words.ram_read_value,
            BaseWord::NextPC => words.next_pc,
        }));
    }
    Polynomial::new(values)
}
fn fetched(witness: &Rv64iWitness, cycle: usize) -> Result<&BytecodeRow, Rv64iProverError> {
    let bits = witness
        .bits
        .get(cycle)
        .ok_or(Rv64iProverError::CycleIndex {
            cycle,
            rows: witness.bits.len(),
        })?;
    let index = witness.layout.bytecode_index(bits);
    usize::try_from(index)
        .ok()
        .and_then(|i| witness.bytecode.rows().get(i))
        .filter(|r| r.variant.is_some())
        .ok_or(Rv64iProverError::InvalidBytecode { cycle, index })
}
/// Extends a fetched word using prepared bit weights; invalid fetches return an error.
pub fn bytecode_word_with_lift(
    witness: &Rv64iWitness,
    column: BytecodeColumn,
    lift: &WordLift<F128>,
) -> Result<Polynomial<F128>, Rv64iProverError> {
    let mut values = Vec::with_capacity(witness.bits.len());
    for cycle in 0..witness.bits.len() {
        values.push(lift.evaluate(fetched(witness, cycle)?.column(column)));
    }
    Ok(Polynomial::new(values))
}
/// Extends a kind selector at a low-variable-first kind point over cycle variables.
/// Returns an error for an invalid fetch or an index outside the point's domain.
pub fn selector(
    witness: &Rv64iWitness,
    selector: Selector,
    point: &[F128],
) -> Result<Polynomial<F128>, Rv64iProverError> {
    let weights = equality_table(point)?;
    let mut values = Vec::with_capacity(witness.bits.len());
    for cycle in 0..witness.bits.len() {
        let row = fetched(witness, cycle)?;
        let value = row.variant.and_then(|variant| match selector {
            Selector::Variant => Some(variant.index()),
            Selector::ShiftKind => variant.shift().map(|shift| shift.kind as usize),
            Selector::AccessKind => variant
                .access()
                .and_then(|access| access.kind)
                .map(|kind| kind as usize),
            Selector::KeyKind => variant.key_kind().map(|kind| kind as usize),
        });
        values.push(match value {
            Some(index) => *weights.get(index).ok_or(PointsError::Index {
                index,
                variables: point.len(),
            })?,
            None => F128::zero(),
        });
    }
    Ok(Polynomial::new(values))
}
/// Extends the fetched instruction's branch flag over low-variable-first cycle variables.
/// Returns an error when a cycle fetches an invalid bytecode row.
pub fn branch(witness: &Rv64iWitness) -> Result<Polynomial<F128>, Rv64iProverError> {
    let mut values = Vec::with_capacity(witness.bits.len());
    for cycle in 0..witness.bits.len() {
        values.push(
            if fetched(witness, cycle)?
                .variant
                .is_some_and(|variant| variant.branch().is_some())
            {
                F128::one()
            } else {
                F128::zero()
            },
        );
    }
    Ok(Polynomial::new(values))
}
/// Extends the fetched instruction's store flag over low-variable-first cycle variables.
/// Returns an error when a cycle fetches an invalid bytecode row.
pub fn store(witness: &Rv64iWitness) -> Result<Polynomial<F128>, Rv64iProverError> {
    let mut values = Vec::with_capacity(witness.bits.len());
    for cycle in 0..witness.bits.len() {
        values.push(
            if fetched(witness, cycle)?
                .variant
                .is_some_and(|variant| variant.is_store())
            {
                F128::one()
            } else {
                F128::zero()
            },
        );
    }
    Ok(Polynomial::new(values))
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
    let weights = equality_table(point)?;
    let mut values = Vec::with_capacity(witness.bits.len());
    for cycle in 0..witness.bits.len() {
        let index = usize::from(register(fetched(witness, cycle)?, selector));
        values.push(*weights.get(index).ok_or(PointsError::Index {
            index,
            variables: point.len(),
        })?);
    }
    Ok(Polynomial::new(values))
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
    let mut values = Vec::with_capacity(witness.bits.len());
    for bits in witness.bits.iter() {
        values.push(eq_index(point, witness.layout.ram_index(bits) as usize)?);
    }
    Ok(Polynomial::new(values))
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
    registers_val_with_lift(witness, &WordLift::new(point)?)
}
/// Replays register pre-state using prepared bit weights; invalid fetches or register selectors return errors.
pub fn registers_val_with_lift(
    witness: &Rv64iWitness,
    lift: &WordLift<F128>,
) -> Result<Polynomial<F128>, Rv64iProverError> {
    let mut registers = [0_u64; 32];
    let mut values = Vec::with_capacity(32 * witness.bits.len());
    for (j, bits) in witness.bits.iter().enumerate() {
        for word in registers {
            values.push(lift.evaluate(word));
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
    ram_val_with_lift(witness, &WordLift::new(point)?)
}
/// Replays RAM pre-state using prepared bit weights and one dense state buffer.
/// Invalid addresses, fetches or RAM allocations return errors.
pub fn ram_val_with_lift(
    witness: &Rv64iWitness,
    lift: &WordLift<F128>,
) -> Result<Polynomial<F128>, Rv64iProverError> {
    let count = domain(witness)?;
    let mut ram = Vec::new();
    ram.try_reserve_exact(count)
        .map_err(|source| Rv64iProverError::RamAllocation {
            log_K_ram: witness.layout.log_K_ram(),
            source,
        })?;
    ram.resize(count, 0_u64);
    for &(index, value) in &witness.initial_ram {
        let address = usize::try_from(index).map_err(|_| Rv64iProverError::InitialRam { index })?;
        *ram.get_mut(address)
            .ok_or(Rv64iProverError::InitialRam { index })? = value;
    }
    let cycles = witness.bits.len();
    let size_error = || Rv64iProverError::RamViewSize {
        words: count,
        cycles,
    };
    let elements = count.checked_mul(cycles).ok_or_else(size_error)?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<F128>())
        .ok_or_else(size_error)?;
    let maximum_bytes = usize::try_from(isize::MAX).map_err(|_| size_error())?;
    if bytes > maximum_bytes {
        return Err(size_error());
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|source| Rv64iProverError::RamViewAllocation {
            words: count,
            cycles,
            source,
        })?;
    for (j, bits) in witness.bits.iter().enumerate() {
        for &word in &ram {
            values.push(lift.evaluate(word));
        }
        if fetched(witness, j)?.variant.is_some_and(|v| v.is_store()) {
            let index = witness.layout.ram_index(bits);
            let address = usize::try_from(index)
                .map_err(|_| Rv64iProverError::StoreRam { cycle: j, index })?;
            *ram.get_mut(address)
                .ok_or(Rv64iProverError::StoreRam { cycle: j, index })? ^= witness.layout.inc(bits);
        }
    }
    Ok(Polynomial::new(values))
}
/// Extends retained final RAM using prepared bit weights; inconsistent final-RAM lengths return an error.
pub fn ram_val_final_with_lift(
    witness: &Rv64iWitness,
    lift: &WordLift<F128>,
) -> Result<Polynomial<F128>, Rv64iProverError> {
    let expected = domain(witness)?;
    if witness.final_ram.len() != expected {
        return Err(Rv64iProverError::FinalRamLength {
            expected,
            found: witness.final_ram.len(),
        });
    }
    let mut values = Vec::with_capacity(expected);
    for &word in &witness.final_ram {
        values.push(lift.evaluate(word));
    }
    Ok(Polynomial::new(values))
}
