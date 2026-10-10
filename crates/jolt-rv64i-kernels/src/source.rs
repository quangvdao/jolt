//! Shared packed inputs for sum-check kernels. Digits are absent or smaller than
//! `2^bits(column)`; a row-based column is a function of the bytecode row alone.

use jolt_kernels::optimized::lazy_ra::ChunkIndexSource;
use std::mem::size_of;
use std::num::NonZeroUsize;
use std::sync::Arc;
use thiserror::Error;

/// Two groups of packed A, B and C lanes and the two-row tail.
/// Tail bits 0–1, 2–3 and 4–5 hold A, B and C respectively.
/// Implementations return zero for an out-of-range cycle.
pub trait LaneSource: Send + Sync + 'static {
    fn cycles(&self) -> usize;
    fn lanes(&self, cycle: usize) -> [[u64; 3]; 2];
    fn tail(&self, cycle: usize) -> u8;
}

/// Trace words, bytecode words and optional one-hot digits.
/// A `by_row(column)` column satisfies
/// `digit(column, cycle) == row_digit(column, bytecode_index(cycle))`.
/// Invalid word, column, cycle and row indices return zero or absence.
/// All dimensions and values must remain immutable while the source is shared;
/// checked adapters validate this data once and subsequently read it directly.
pub trait CycleSource: Send + Sync + 'static {
    fn cycles(&self) -> usize;
    fn trace_words(&self) -> usize;
    fn trace_word(&self, word: usize, cycle: usize) -> u64;
    fn bytecode_rows(&self) -> usize;
    fn bytecode_words(&self) -> usize;
    fn bytecode_word(&self, word: usize, row: usize) -> u64;
    fn bytecode_index(&self, cycle: usize) -> usize;
    fn digit_columns(&self) -> usize;
    fn bits(&self, column: usize) -> usize;
    fn by_row(&self, column: usize) -> bool;
    fn digit(&self, column: usize, cycle: usize) -> Option<usize>;
    fn row_digit(&self, column: usize, row: usize) -> Option<usize>;
}

/// Invalid source dimensions, column selection or source digits.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SourceError {
    #[error("cycle count {cycles} is not a power of two")]
    CycleCount { cycles: usize },
    #[error("bytecode row count {rows} is not a power of two")]
    BytecodeRows { rows: usize },
    #[error("column {column} is outside {columns} digit columns")]
    Column { column: usize, columns: usize },
    #[error("column {column} width {bits} cannot be represented")]
    Width { column: usize, bits: usize },
    #[error("cycle {cycle} selects bytecode row {row} outside {rows} rows")]
    BytecodeIndex {
        cycle: usize,
        row: usize,
        rows: usize,
    },
    #[error("column {column} at cycle {cycle} has digit {digit} outside bound {bound}")]
    Digit {
        column: usize,
        cycle: usize,
        digit: usize,
        bound: usize,
    },
    #[error("column {column} at bytecode row {row} has digit {digit} outside bound {bound}")]
    RowDigitRange {
        column: usize,
        row: usize,
        digit: usize,
        bound: usize,
    },
    #[error("row-based column {column} disagrees with row {row} at cycle {cycle}")]
    RowDigit {
        column: usize,
        cycle: usize,
        row: usize,
    },
    #[error("validation scratch {name} with {len} elements of {element_size} bytes cannot be represented")]
    ValidationScratchSize {
        name: &'static str,
        len: usize,
        element_size: usize,
    },
}

fn check_validation_size<T>(len: usize, name: &'static str) -> Result<(), SourceError> {
    let element_size = size_of::<T>();
    if len
        .checked_mul(element_size)
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(SourceError::ValidationScratchSize {
            name,
            len,
            element_size,
        });
    }
    Ok(())
}

/// Shared source whose dimensions, indices and every digit column are checked.
/// Construction reads every row-based digit once and every cycle digit once.
/// Temporary row caches are dropped before returning; the source owns its data.
/// The guarantee relies on the immutability contract of [`CycleSource`].
#[derive(Debug, Clone)]
pub struct ValidatedTrace<S: CycleSource> {
    source: Arc<S>,
    cycles: usize,
    widths: Vec<usize>,
}

impl<S: CycleSource> ValidatedTrace<S> {
    pub fn new(source: Arc<S>) -> Result<Self, SourceError> {
        let cycles = source.cycles();
        if !cycles.is_power_of_two() {
            return Err(SourceError::CycleCount { cycles });
        }
        let rows = source.bytecode_rows();
        if !rows.is_power_of_two() {
            return Err(SourceError::BytecodeRows { rows });
        }
        let columns = source.digit_columns();
        check_validation_size::<usize>(columns, "digit widths")?;
        check_validation_size::<Option<Vec<Option<NonZeroUsize>>>>(
            columns,
            "row digit cache metadata",
        )?;
        let mut widths = Vec::with_capacity(columns);
        let mut row_digits = Vec::with_capacity(columns);
        for column in 0..columns {
            let bits = source.bits(column);
            if bits >= usize::BITS as usize {
                return Err(SourceError::Width { column, bits });
            }
            widths.push(bits);
            let cache = if source.by_row(column) {
                check_validation_size::<Option<NonZeroUsize>>(rows, "row digit cache")?;
                Some(vec![None; rows])
            } else {
                None
            };
            row_digits.push(cache);
        }
        for row in 0..rows {
            for (column, cache) in row_digits.iter_mut().enumerate() {
                if let Some(cache) = cache {
                    let bound = 1 << widths[column];
                    let digit = source.row_digit(column, row);
                    if let Some(digit) = digit {
                        if digit >= bound {
                            return Err(SourceError::RowDigitRange {
                                column,
                                row,
                                digit,
                                bound,
                            });
                        }
                    }
                    cache[row] = digit.and_then(|value| NonZeroUsize::new(value + 1));
                }
            }
        }
        for cycle in 0..cycles {
            let row = source.bytecode_index(cycle);
            if row >= rows {
                return Err(SourceError::BytecodeIndex { cycle, row, rows });
            }
            for (column, &bits) in widths.iter().enumerate() {
                let bound = 1 << bits;
                let digit = source.digit(column, cycle);
                if let Some(digit) = digit {
                    if digit >= bound {
                        return Err(SourceError::Digit {
                            column,
                            cycle,
                            digit,
                            bound,
                        });
                    }
                }
                if let Some(cache) = &row_digits[column] {
                    if digit != cache[row].map(|value| value.get() - 1) {
                        return Err(SourceError::RowDigit { column, cycle, row });
                    }
                }
            }
        }
        Ok(Self {
            source,
            cycles,
            widths,
        })
    }

    pub fn source(&self) -> &Arc<S> {
        &self.source
    }
}

/// A checked column selection retaining shared ownership of its source.
/// Multiple selections share a [`ValidatedTrace`] without repeating its scan.
#[derive(Debug, Clone)]
pub struct DigitColumns<S: CycleSource> {
    trace: Arc<ValidatedTrace<S>>,
    columns: Vec<usize>,
}

impl<S: CycleSource> DigitColumns<S> {
    /// Checks the entire arbitrary source, including columns outside the selection.
    pub fn new(source: Arc<S>, columns: Vec<usize>) -> Result<Self, SourceError> {
        Self::from_validated(Arc::new(ValidatedTrace::new(source)?), columns)
    }

    /// Checks only the column list against an already validated source.
    pub fn from_validated(
        trace: Arc<ValidatedTrace<S>>,
        columns: Vec<usize>,
    ) -> Result<Self, SourceError> {
        for &column in &columns {
            if column >= trace.widths.len() {
                return Err(SourceError::Column {
                    column,
                    columns: trace.widths.len(),
                });
            }
        }
        Ok(Self { trace, columns })
    }

    pub fn source(&self) -> &Arc<S> {
        self.trace.source()
    }
    pub fn columns(&self) -> &[usize] {
        &self.columns
    }
    pub fn cycles(&self) -> usize {
        self.trace.cycles
    }
    pub fn num_polys(&self) -> usize {
        self.columns.len()
    }
    #[inline]
    pub fn index(&self, column: usize, cycle: usize) -> Option<usize> {
        self.columns
            .get(column)
            .and_then(|&c| self.trace.source.digit(c, cycle))
    }
    #[inline]
    pub fn index_bound(&self, column: usize) -> Option<usize> {
        self.columns.get(column).map(|&c| 1 << self.trace.widths[c])
    }
}

impl<S: CycleSource> ChunkIndexSource for DigitColumns<S> {
    fn num_polys(&self) -> usize {
        self.num_polys()
    }

    fn cycles(&self) -> usize {
        self.cycles()
    }

    fn index(&self, column: usize, cycle: usize) -> Option<usize> {
        self.index(column, cycle)
    }

    fn index_bound(&self, column: usize) -> Option<usize> {
        self.index_bound(column)
    }
}
