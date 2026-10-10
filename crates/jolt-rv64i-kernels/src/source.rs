//! Shared packed inputs for sum-check kernels. Digits are absent or smaller than
//! `2^bits(column)`; a row-based column is a function of the bytecode row alone.

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
}

/// A checked column selection retaining shared ownership of its source.
#[derive(Debug, Clone)]
pub struct DigitColumns<S: CycleSource> {
    source: Arc<S>,
    columns: Vec<usize>,
}

impl<S: CycleSource> DigitColumns<S> {
    pub fn new(source: Arc<S>, columns: Vec<usize>) -> Result<Self, SourceError> {
        if !source.cycles().is_power_of_two() {
            return Err(SourceError::CycleCount {
                cycles: source.cycles(),
            });
        }
        if !source.bytecode_rows().is_power_of_two() {
            return Err(SourceError::BytecodeRows {
                rows: source.bytecode_rows(),
            });
        }
        for &column in &columns {
            if column >= source.digit_columns() {
                return Err(SourceError::Column {
                    column,
                    columns: source.digit_columns(),
                });
            }
            if source.bits(column) >= usize::BITS as usize {
                return Err(SourceError::Width {
                    column,
                    bits: source.bits(column),
                });
            }
        }
        for &column in &columns {
            if source.by_row(column) {
                let bound = 1 << source.bits(column);
                for row in 0..source.bytecode_rows() {
                    if let Some(digit) = source.row_digit(column, row) {
                        if digit >= bound {
                            return Err(SourceError::RowDigitRange {
                                column,
                                row,
                                digit,
                                bound,
                            });
                        }
                    }
                }
            }
        }
        for cycle in 0..source.cycles() {
            let row = source.bytecode_index(cycle);
            if row >= source.bytecode_rows() {
                return Err(SourceError::BytecodeIndex {
                    cycle,
                    row,
                    rows: source.bytecode_rows(),
                });
            }
            for &column in &columns {
                let bound = 1 << source.bits(column);
                if let Some(digit) = source.digit(column, cycle) {
                    if digit >= bound {
                        return Err(SourceError::Digit {
                            column,
                            cycle,
                            digit,
                            bound,
                        });
                    }
                }
                if source.by_row(column)
                    && source.digit(column, cycle) != source.row_digit(column, row)
                {
                    return Err(SourceError::RowDigit { column, cycle, row });
                }
            }
        }
        Ok(Self { source, columns })
    }

    pub fn source(&self) -> &Arc<S> {
        &self.source
    }
    pub fn columns(&self) -> &[usize] {
        &self.columns
    }
    pub fn cycles(&self) -> usize {
        self.source.cycles()
    }
    pub fn num_polys(&self) -> usize {
        self.columns.len()
    }
    #[inline]
    pub fn index(&self, column: usize, cycle: usize) -> Option<usize> {
        self.columns
            .get(column)
            .and_then(|&c| self.source.digit(c, cycle))
    }
    #[inline]
    pub fn index_bound(&self, column: usize) -> Option<usize> {
        self.columns.get(column).map(|&c| 1 << self.source.bits(c))
    }
}
