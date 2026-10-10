//! Memory views for the experiment in RV64I hash-based Jolt over binary fields.
//! Groups must belong to one prepared trace: construction checks their geometry,
//! but cannot check that association or the protocol meaning of a lifted word.

use crate::packed::lift::WordLift;
use crate::packed::scatter::{ScatterError, ScatterPlan};
use crate::par::{CycleChunks, ParError};
use crate::round::eq::eq_table;
use crate::source::{CycleSource, OptionalGroup, PresentGroup, ValidatedTrace};
use jolt_field::{Accumulator, F128Accumulator, F128};
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;
use std::sync::Arc;
use thiserror::Error;

/// Largest supported RAM address dimension.
pub const MAX_ADDRESS_BITS: usize = 32;

/// Malformed memory geometry or ownership, and the address capability bound.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MemoryError {
    #[error("RAM address has {bits} bits; at least one is required")]
    AddressBits { bits: usize },
    #[error("RAM address has {bits} bits, supported maximum is {supported}")]
    UnsupportedAddressBits { bits: usize, supported: usize },
    #[error("register group has {columns} columns, expected three")]
    RegisterColumns { columns: usize },
    #[error("register column {column} has width {bits}, expected five")]
    RegisterWidth { column: usize, bits: usize },
    #[error("store group has {columns} columns, expected one")]
    StoreColumns { columns: usize },
    #[error("store column {column} has width {bits}, expected zero")]
    StoreWidth { column: usize, bits: usize },
    #[error("{group} group has {actual} cycles, expected {expected}")]
    Cycles {
        group: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("RAM group is still in use by {handles} other handles")]
    InUse { handles: usize },
    #[error("word {word} is outside {words} trace words")]
    Word { word: usize, words: usize },
    #[error("{words} words cannot be folded at {point} coordinates")]
    Words { words: usize, point: usize },
    #[error("point has {actual} coordinates, expected {expected}")]
    PointLength { expected: usize, actual: usize },
    #[error(transparent)]
    Geometry(#[from] ParError),
    #[error(transparent)]
    Scatter(#[from] ScatterError),
}

/// The groups retained independently by the value-evaluation core.
#[derive(Debug)]
pub(crate) struct RegisterStore {
    registers: PresentGroup,
    store: OptionalGroup,
}

impl RegisterStore {
    #[inline]
    pub(crate) fn registers(&self, cycle: usize) -> [usize; 3] {
        let bytes = &self.registers.bytes()[3 * cycle..3 * cycle + 3];
        [
            usize::from(bytes[0]),
            usize::from(bytes[1]),
            usize::from(bytes[2]),
        ]
    }
    #[inline]
    pub(crate) fn store(&self, cycle: usize) -> bool {
        self.store.bytes()[cycle] != 0
    }
}

/// Owned RAM, register and store byte groups, without a second digit gather.
/// Missing operands and accesses must already be present digits of value zero.
#[derive(Debug)]
pub struct MemoryTrace {
    ram: PresentGroup,
    pub(crate) register_store: Arc<RegisterStore>,
    address_bits: usize,
}

impl MemoryTrace {
    /// Checks column counts, widths and matching cycle counts without a cycle pass.
    pub fn new(
        ram: PresentGroup,
        registers: PresentGroup,
        store: OptionalGroup,
    ) -> Result<Self, MemoryError> {
        let address_bits = ram.widths().iter().sum();
        if address_bits == 0 {
            return Err(MemoryError::AddressBits { bits: address_bits });
        }
        if address_bits > MAX_ADDRESS_BITS {
            return Err(MemoryError::UnsupportedAddressBits {
                bits: address_bits,
                supported: MAX_ADDRESS_BITS,
            });
        }
        if registers.columns().len() != 3 {
            return Err(MemoryError::RegisterColumns {
                columns: registers.columns().len(),
            });
        }
        for (&column, &bits) in registers.columns().iter().zip(registers.widths()) {
            if bits != 5 {
                return Err(MemoryError::RegisterWidth { column, bits });
            }
        }
        if store.columns().len() != 1 {
            return Err(MemoryError::StoreColumns {
                columns: store.columns().len(),
            });
        }
        if store.widths()[0] != 0 {
            return Err(MemoryError::StoreWidth {
                column: store.columns()[0],
                bits: store.widths()[0],
            });
        }
        for (group, actual) in [("registers", registers.cycles()), ("store", store.cycles())] {
            if actual != ram.cycles() {
                return Err(MemoryError::Cycles {
                    group,
                    expected: ram.cycles(),
                    actual,
                });
            }
        }
        Ok(Self {
            ram,
            register_store: Arc::new(RegisterStore { registers, store }),
            address_bits,
        })
    }

    pub fn cycles(&self) -> usize {
        self.ram.cycles()
    }
    pub fn address_bits(&self) -> usize {
        self.address_bits
    }

    /// Moves the original RAM allocation out if no other trace handle exists.
    /// Separate register/store handles may remain alive.
    pub fn into_ram_chunks(this: Arc<Self>) -> Result<PresentGroup, MemoryError> {
        Arc::try_unwrap(this)
            .map(|memory| memory.ram)
            .map_err(|memory| MemoryError::InUse {
                handles: Arc::strong_count(&memory) - 1,
            })
    }

    /// The caller reads only cycles below `cycles()`.
    #[inline]
    pub(crate) fn index(&self, cycle: usize) -> usize {
        let count = self.ram.columns().len();
        let mut index = 0;
        let mut shift = 0;
        for (&digit, &width) in self.ram.bytes()[cycle * count..(cycle + 1) * count]
            .iter()
            .zip(self.ram.widths())
        {
            index |= usize::from(digit) << shift;
            shift += width;
        }
        index
    }

    #[inline]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "crate-visible contract for the subsequent memory-core items"
        )
    )]
    pub(crate) fn registers(&self, cycle: usize) -> [usize; 3] {
        self.register_store.registers(cycle)
    }
    #[inline]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "crate-visible contract for the subsequent memory-core items"
        )
    )]
    pub(crate) fn store(&self, cycle: usize) -> bool {
        self.register_store.store(cycle)
    }
}

/// Lift the chosen trace word once per cycle, without reading source digits.
/// Its protocol meaning and the choice of bit weights are the caller's.
pub fn inc_lift<S: CycleSource>(
    trace: &ValidatedTrace<S>,
    word: usize,
    lift: &WordLift,
) -> Result<Vec<F128>, MemoryError> {
    let source = trace.source();
    if word >= source.trace_words() {
        return Err(MemoryError::Word {
            word,
            words: source.trace_words(),
        });
    }
    let chunks = CycleChunks::new(source.cycles().ilog2() as usize, 0)?;
    let mut output = unsafe_allocate_zero_vec(source.cycles());
    output
        .par_chunks_mut(chunks.chunk_len())
        .enumerate()
        .for_each(|(chunk, values)| {
            for (offset, value) in values.iter_mut().enumerate() {
                *value = lift.lift(source.trace_word(word, chunk * chunks.chunk_len() + offset));
            }
        });
    Ok(output)
}

/// Fold `2^m` packed words at an `n <= m` coordinate point, low bit first:
/// `out[c] = sum_low eq(point, low) * lift(words[c * 2^n + low])`.
/// An empty point lifts each word separately.
pub fn fold_words(
    words: &[u64],
    lift: &WordLift,
    point: &[F128],
) -> Result<Vec<F128>, MemoryError> {
    if !words.len().is_power_of_two() || point.len() > words.len().ilog2() as usize {
        return Err(MemoryError::Words {
            words: words.len(),
            point: point.len(),
        });
    }
    let chunks = CycleChunks::new(point.len(), 0)?;
    let (low, high) = chunks.split_point(point)?;
    let low = eq_table(low, None);
    let high = eq_table(high, None);
    let mut output = unsafe_allocate_zero_vec(words.len() / chunks.len());
    output
        .par_iter_mut()
        .zip(words.par_chunks(chunks.len()))
        .for_each(|(value, words)| {
            let block = |(index, words): (usize, &[u64])| {
                let mut sum = F128Accumulator::default();
                for (&word, &weight) in words.iter().zip(&low) {
                    sum.fmadd(lift.lift(word), weight);
                }
                sum.reduce() * high[index]
            };
            *value = if chunks.len() > chunks.chunk_len() {
                words
                    .par_chunks(low.len())
                    .enumerate()
                    .map(block)
                    .reduce(|| F128::from_raw(0), |a, b| a + b)
            } else {
                words
                    .chunks(low.len())
                    .enumerate()
                    .map(block)
                    .fold(F128::from_raw(0), |a, b| a + b)
            };
        });
    Ok(output)
}

/// Return `eq(point, addr_j)` per cycle, using equality pieces of at most 11 bits.
pub fn address_column(memory: &MemoryTrace, point: &[F128]) -> Result<Vec<F128>, MemoryError> {
    if point.len() != memory.address_bits() {
        return Err(MemoryError::PointLength {
            expected: memory.address_bits(),
            actual: point.len(),
        });
    }
    let tables: Vec<_> = point
        .chunks(11)
        .map(|point| eq_table(point, None))
        .collect();
    let chunks = CycleChunks::new(memory.cycles().ilog2() as usize, 0)?;
    let mut output = unsafe_allocate_zero_vec(memory.cycles());
    output
        .par_chunks_mut(chunks.chunk_len())
        .enumerate()
        .for_each(|(chunk, values)| {
            for (offset, value) in values.iter_mut().enumerate() {
                let mut address = memory.index(chunk * chunks.chunk_len() + offset);
                let mut result = tables[0][address & (tables[0].len() - 1)];
                address >>= 11;
                for table in &tables[1..] {
                    result *= table[address & (table.len() - 1)];
                    address >>= 11;
                }
                *value = result;
            }
        });
    Ok(output)
}

/// Cycle equality weights, or equality at the predecessor (zero at cycle 0).
#[derive(Debug, Clone, Copy)]
pub enum RowWeight<'a> {
    Eq(&'a [F128]),
    Next(&'a [F128]),
}

/// Sum cycle weights by the plan's cached bytecode row, using two half tables.
pub fn row_weights<S: CycleSource>(
    plan: &ScatterPlan<S>,
    weight: RowWeight<'_>,
) -> Result<Vec<F128>, MemoryError> {
    let (point, next) = match weight {
        RowWeight::Eq(point) => (point, false),
        RowWeight::Next(point) => (point, true),
    };
    let expected = plan.cycles().ilog2() as usize;
    if point.len() != expected {
        return Err(MemoryError::PointLength {
            expected,
            actual: point.len(),
        });
    }
    let chunks = CycleChunks::new(expected, 0)?;
    let (low, high) = chunks.split_point(point)?;
    let low = eq_table(low, None);
    let high = eq_table(high, None);
    Ok(plan.scatter(|cycle| {
        if next && cycle == 0 {
            return F128::from_raw(0);
        }
        let index = cycle - usize::from(next);
        low[index & (low.len() - 1)] * high[index >> chunks.low_bits()]
    })?)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "invalid fixtures and contract failures fail the test"
)]
mod tests {
    use super::MemoryTrace;
    use crate::source::{CycleSource, PrepareRequest, ValidatedTrace};
    use std::sync::Arc;

    const ADDRESSES: [usize; 8] = [0, 1, 0, 1, 3, 0x13, 0x31, 0x10203];
    const REGISTERS: [[usize; 3]; 8] = [
        [1, 2, 3],
        [3, 4, 0],
        [0, 5, 6],
        [6, 7, 8],
        [8, 9, 10],
        [10, 11, 12],
        [12, 13, 14],
        [14, 15, 16],
    ];
    const STORES: [bool; 8] = [true, false, true, true, false, false, false, false];

    struct Fixture {
        widths: Vec<usize>,
        cycles: usize,
    }

    impl CycleSource for Fixture {
        fn cycles(&self) -> usize {
            self.cycles
        }
        fn trace_words(&self) -> usize {
            0
        }
        fn trace_word(&self, _: usize, _: usize) -> u64 {
            0
        }
        fn bytecode_rows(&self) -> usize {
            8
        }
        fn bytecode_words(&self) -> usize {
            0
        }
        fn bytecode_word(&self, _: usize, _: usize) -> u64 {
            0
        }
        fn bytecode_index(&self, cycle: usize) -> usize {
            cycle % 8
        }
        fn digit_columns(&self) -> usize {
            self.widths.len() + 4
        }
        fn bits(&self, column: usize) -> usize {
            self.widths.get(column).copied().unwrap_or_else(|| {
                if column < self.widths.len() + 3 {
                    5
                } else {
                    0
                }
            })
        }
        fn by_row(&self, _: usize) -> bool {
            true
        }
        fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
            self.row_digit(column, cycle % 8)
        }
        fn row_digit(&self, column: usize, row: usize) -> Option<usize> {
            if column < self.widths.len() {
                Some(
                    (ADDRESSES[row] >> self.widths[..column].iter().sum::<usize>())
                        & ((1 << self.widths[column]) - 1),
                )
            } else if column < self.widths.len() + 3 {
                Some(REGISTERS[row][column - self.widths.len()])
            } else {
                STORES[row].then_some(0)
            }
        }
    }

    #[test]
    fn memory_reads_and_separate_register_store_ownership() {
        for cycles in [8, 256] {
            for widths in [vec![1], vec![4, 4], vec![8, 4], vec![4; 5]] {
                let bits = widths.iter().sum::<usize>();
                let columns = widths.len();
                let source = Arc::new(Fixture { widths, cycles });
                let (_, mut groups) = ValidatedTrace::prepare(
                    source,
                    PrepareRequest {
                        present: vec![(0..columns).collect(), (columns..columns + 3).collect()],
                        optional: vec![vec![columns + 3]],
                    },
                )
                .unwrap();
                let registers = groups.present.pop().unwrap();
                let ram = groups.present.pop().unwrap();
                let store = groups.optional.pop().unwrap();
                let pointer = ram.bytes().as_ptr();
                let memory = Arc::new(MemoryTrace::new(ram, registers, store).unwrap());
                for cycle in 0..cycles {
                    assert_eq!(
                        memory.index(cycle),
                        ADDRESSES[cycle % 8] & ((1 << bits) - 1)
                    );
                    assert_eq!(memory.registers(cycle), REGISTERS[cycle % 8]);
                    assert_eq!(memory.store(cycle), STORES[cycle % 8]);
                }
                let retained = Arc::clone(&memory.register_store);
                let ram = MemoryTrace::into_ram_chunks(memory).unwrap();
                assert_eq!(ram.bytes().as_ptr(), pointer);
                assert_eq!(retained.registers(1), [3, 4, 0]);
                assert!(retained.store(0));
            }
        }
    }
}
