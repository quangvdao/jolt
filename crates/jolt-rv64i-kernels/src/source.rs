//! Shared packed inputs for sum-check kernels. Digits are absent or smaller than
//! `2^bits(column)`; a row-based column is a function of the bytecode row alone.

use crate::par::CycleChunks;
use jolt_kernels::optimized::lazy_ra::ChunkIndexSource;
use rayon::prelude::*;
use std::mem::size_of;
use std::num::{NonZeroU8, NonZeroUsize};
use std::ops::Range;
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

const ROW_CHUNK: usize = 4096;

enum RowDigits {
    Byte(Vec<Option<NonZeroU8>>),
    Word(Vec<Option<NonZeroUsize>>),
}

struct RowColumn {
    column: usize,
    bound: usize,
    digits: RowDigits,
}

enum RowChunk<'a> {
    Byte {
        column: usize,
        bound: usize,
        digits: &'a mut [Option<NonZeroU8>],
    },
    Word {
        column: usize,
        bound: usize,
        digits: &'a mut [Option<NonZeroUsize>],
    },
}

struct RowFault {
    column: usize,
    row: usize,
    digit: usize,
    bound: usize,
}

impl RowChunk<'_> {
    fn validate(&mut self, source: &impl CycleSource, rows: Range<usize>) -> Option<RowFault> {
        match self {
            Self::Byte {
                column,
                bound,
                digits,
            } => cache_rows(source, *column, *bound, rows, digits, |value| {
                NonZeroU8::new((value + 1) as u8)
            }),
            Self::Word {
                column,
                bound,
                digits,
            } => cache_rows(source, *column, *bound, rows, digits, |value| {
                NonZeroUsize::new(value + 1)
            }),
        }
    }
}

fn cache_rows<N>(
    source: &impl CycleSource,
    column: usize,
    bound: usize,
    rows: Range<usize>,
    cache: &mut [Option<N>],
    encode: impl Fn(usize) -> Option<N>,
) -> Option<RowFault> {
    for (row, slot) in rows.zip(cache) {
        let digit = source.row_digit(column, row);
        if let Some(digit) = digit {
            if digit >= bound {
                return Some(RowFault {
                    column,
                    row,
                    digit,
                    bound,
                });
            }
        }
        *slot = digit.and_then(&encode);
    }
    None
}

struct CycleValidation<'a> {
    per_cycle: Vec<(usize, usize)>,
    row_bytes: Vec<(usize, usize, &'a [Option<NonZeroU8>])>,
    row_words: Vec<(usize, usize, &'a [Option<NonZeroUsize>])>,
}

impl CycleValidation<'_> {
    fn validate_chunk(
        &self,
        source: &impl CycleSource,
        cycles: Range<usize>,
        rows: usize,
    ) -> Result<(), SourceError> {
        let mut fault = cycles.clone().find_map(|cycle| {
            let row = source.bytecode_index(cycle);
            (row >= rows).then_some((cycle, 0, SourceError::BytecodeIndex { cycle, row, rows }))
        });
        let end = fault.as_ref().map_or(cycles.end, |(cycle, _, _)| *cycle);
        macro_rules! scan {
            ($column:expr, $bound:expr, $matches:expr) => {{
                let column = $column;
                let bound = $bound;
                for cycle in cycles.start..end {
                    let digit = source.digit(column, cycle);
                    let error = if let Some(digit) = digit.filter(|&digit| digit >= bound) {
                        Some(SourceError::Digit {
                            column,
                            cycle,
                            digit,
                            bound,
                        })
                    } else {
                        $matches(digit, cycle)
                    };
                    if let Some(error) = error {
                        let rank = column + 1;
                        if fault.as_ref().is_none_or(|(previous, previous_rank, _)| {
                            (cycle, rank) < (*previous, *previous_rank)
                        }) {
                            fault = Some((cycle, rank, error));
                        }
                        break;
                    }
                }
            }};
        }
        for &(column, bound) in &self.per_cycle {
            scan!(column, bound, |_, _| None);
        }
        for &(column, bound, cache) in &self.row_bytes {
            scan!(column, bound, |digit, cycle| {
                let row = source.bytecode_index(cycle);
                (digit != cache[row].map(|value| usize::from(value.get()) - 1))
                    .then_some(SourceError::RowDigit { column, cycle, row })
            });
        }
        for &(column, bound, cache) in &self.row_words {
            scan!(column, bound, |digit, cycle| {
                let row = source.bytecode_index(cycle);
                (digit != cache[row].map(|value| value.get() - 1))
                    .then_some(SourceError::RowDigit { column, cycle, row })
            });
        }
        if let Some((_, _, error)) = fault {
            return Err(error);
        }
        Ok(())
    }
}

/// Shared source whose dimensions, indices and every digit column are checked.
/// Construction reads every row-based digit once and every cycle digit once.
/// Temporary row caches are dropped before returning; the source owns its data.
/// Digit faults follow row order before cycle order, then column order. A cycle
/// index fault precedes its digit faults; a digit range fault precedes row
/// disagreement in the same column. Parallel completion cannot change this order.
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
        check_validation_size::<RowColumn>(columns, "row digit cache columns")?;
        let mut widths = Vec::with_capacity(columns);
        let mut row_digits = Vec::new();
        let mut per_cycle = Vec::new();
        for column in 0..columns {
            let bits = source.bits(column);
            if bits >= usize::BITS as usize {
                return Err(SourceError::Width { column, bits });
            }
            widths.push(bits);
            let bound = 1 << bits;
            if source.by_row(column) {
                check_validation_size::<Option<NonZeroUsize>>(rows, "row digit cache")?;
                let digits = if bits < u8::BITS as usize {
                    RowDigits::Byte(vec![None; rows])
                } else {
                    RowDigits::Word(vec![None; rows])
                };
                row_digits.push(RowColumn {
                    column,
                    bound,
                    digits,
                });
            } else {
                per_cycle.push((column, bound));
            }
        }
        if !row_digits.is_empty() {
            let chunk_count = rows.div_ceil(ROW_CHUNK);
            let descriptors = chunk_count.checked_mul(row_digits.len()).ok_or(
                SourceError::ValidationScratchSize {
                    name: "row chunk metadata",
                    len: chunk_count,
                    element_size: size_of::<RowChunk<'_>>(),
                },
            )?;
            check_validation_size::<RowChunk<'_>>(descriptors, "row chunk metadata")?;
            let column_count = row_digits.len();
            let mut chunks = Vec::with_capacity(descriptors);
            for RowColumn {
                column,
                bound,
                digits,
            } in &mut row_digits
            {
                match digits {
                    RowDigits::Byte(digits) => {
                        for (chunk, digits) in digits.chunks_mut(ROW_CHUNK).enumerate() {
                            chunks.push((
                                chunk,
                                RowChunk::Byte {
                                    column: *column,
                                    bound: *bound,
                                    digits,
                                },
                            ));
                        }
                    }
                    RowDigits::Word(digits) => {
                        for (chunk, digits) in digits.chunks_mut(ROW_CHUNK).enumerate() {
                            chunks.push((
                                chunk,
                                RowChunk::Word {
                                    column: *column,
                                    bound: *bound,
                                    digits,
                                },
                            ));
                        }
                    }
                }
            }
            chunks.sort_unstable_by_key(|(chunk, _)| *chunk);
            if let Some(Some(fault)) = chunks
                .par_chunks_mut(column_count)
                .enumerate()
                .map(|(chunk, columns)| {
                    let rows = chunk * ROW_CHUNK..((chunk + 1) * ROW_CHUNK).min(rows);
                    columns
                        .iter_mut()
                        .filter_map(|(_, column)| column.validate(source.as_ref(), rows.clone()))
                        .min_by_key(|fault| (fault.row, fault.column))
                })
                .find_first(Option::is_some)
            {
                return Err(SourceError::RowDigitRange {
                    column: fault.column,
                    row: fault.row,
                    digit: fault.digit,
                    bound: fault.bound,
                });
            }
        }
        let mut validation = CycleValidation {
            per_cycle,
            row_bytes: Vec::new(),
            row_words: Vec::new(),
        };
        for RowColumn {
            column,
            bound,
            digits,
        } in &row_digits
        {
            match digits {
                RowDigits::Byte(digits) => validation.row_bytes.push((*column, *bound, digits)),
                RowDigits::Word(digits) => validation.row_words.push((*column, *bound, digits)),
            }
        }
        let geometry = CycleChunks::new(cycles.ilog2() as usize, 0)
            .map_err(|_| SourceError::CycleCount { cycles })?;
        if let Some(Err(error)) = (0..cycles / geometry.chunk_len())
            .into_par_iter()
            .map(|chunk| {
                let start = chunk * geometry.chunk_len();
                validation.validate_chunk(
                    source.as_ref(),
                    start..start + geometry.chunk_len(),
                    rows,
                )
            })
            .find_first(Result::is_err)
        {
            return Err(error);
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
