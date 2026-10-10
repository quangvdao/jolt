//! Shared packed inputs for sum-check kernels. Digits are absent or smaller than
//! `2^bits(column)`; a row-based column is a function of the bytecode row alone.

use crate::par::CycleChunks;
use jolt_kernels::optimized::lazy_ra::ChunkIndexSource;
use jolt_utils::unsafe_allocate_zero_vec;
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
    /// Writes encoded digits in cycle-major, column-list order: slot
    /// `(j - cycles.start) * digit_columns() + c` is zero for absence and
    /// `min(d.saturating_add(1), u16::MAX)` for `digit(c, j) == Some(d)`.
    /// Saturation preserves range rejection even for an over-wide malformed digit.
    /// Every output slot is written. If the output length differs from
    /// `cycles.len() * digit_columns()` (including overflow), all slots are zeroed
    /// and no digit is read. Empty/reversed ranges therefore require empty output.
    /// An override decodes the same storage with the same field description as
    /// `digit`: these are two views of one immutable digit function.
    /// Preparation checks agreement only in tiles it rejects; agreement in
    /// accepted tiles is the source's obligation. No unsafe operation in this
    /// crate may rely on that obligation; later source reads use safe indexing
    /// or masking.
    fn digits(&self, cycles: Range<usize>, out: &mut [u16]) {
        let columns = self.digit_columns();
        if cycles.len().checked_mul(columns) != Some(out.len()) || columns == 0 {
            out.fill(0);
            return;
        }
        for (cycle, row) in cycles.zip(out.chunks_exact_mut(columns)) {
            for (column, slot) in row.iter_mut().enumerate() {
                *slot = self.digit(column, cycle).map_or(0, |digit| {
                    digit.saturating_add(1).min(u16::MAX as usize) as u16
                });
            }
        }
    }
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
    #[error("column {column} width {bits} exceeds group limit {max_bits}")]
    GroupWidth {
        column: usize,
        bits: usize,
        max_bits: usize,
    },
    #[error("column {column} has no digit at cycle {cycle}")]
    MissingDigit { column: usize, cycle: usize },
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

/// Column lists in output order; a column may occur in several groups or twice
/// in one list. Empty groups are valid. Requests are checked before digit reads.
#[derive(Debug, Default)]
pub struct PrepareRequest {
    pub present: Vec<Vec<usize>>,
    pub optional: Vec<Vec<usize>>,
}

/// Owned groups, in the order of each request list.
#[derive(Debug)]
pub struct PreparedGroups {
    pub present: Vec<PresentGroup>,
    pub optional: Vec<OptionalGroup>,
}

#[derive(Debug, Clone)]
struct GroupData {
    columns: Vec<usize>,
    widths: Vec<usize>,
    cycles: usize,
    bytes: Vec<u8>,
}

/// Validated, present digits of at most eight bits, owned independently of the
/// source. Only [`ValidatedTrace::prepare`] builds this type. Bytes of a cycle
/// are adjacent in column-list order, with no presence encoding.
#[derive(Debug, Clone)]
pub struct PresentGroup(GroupData);

/// Validated optional digits of at most seven bits. Only
/// [`ValidatedTrace::prepare`] builds this type. Each byte is the digit plus
/// one, or zero for absence, in cycle-major, column-list order.
#[derive(Debug, Clone)]
pub struct OptionalGroup(GroupData);

macro_rules! group_accessors {
    ($group:ident) => {
        impl $group {
            pub fn columns(&self) -> &[usize] {
                &self.0.columns
            }
            pub fn widths(&self) -> &[usize] {
                &self.0.widths
            }
            pub fn cycles(&self) -> usize {
                self.0.cycles
            }
            pub fn bytes(&self) -> &[u8] {
                &self.0.bytes
            }
        }
    };
}
group_accessors!(PresentGroup);
group_accessors!(OptionalGroup);

impl ChunkIndexSource for PresentGroup {
    fn num_polys(&self) -> usize {
        self.0.columns.len()
    }
    fn cycles(&self) -> usize {
        self.0.cycles
    }
    #[inline]
    fn index(&self, column: usize, cycle: usize) -> Option<usize> {
        Some(usize::from(self.0.bytes[cycle * self.num_polys() + column]))
    }
    fn index_bound(&self, column: usize) -> Option<usize> {
        Some(1 << self.0.widths[column])
    }
}
impl ChunkIndexSource for OptionalGroup {
    fn num_polys(&self) -> usize {
        self.0.columns.len()
    }
    fn cycles(&self) -> usize {
        self.0.cycles
    }
    #[inline]
    fn index(&self, column: usize, cycle: usize) -> Option<usize> {
        let byte = self.0.bytes[cycle * self.num_polys() + column];
        (byte != 0).then(|| usize::from(byte - 1))
    }
    fn index_bound(&self, column: usize) -> Option<usize> {
        Some(1 << self.0.widths[column])
    }
}

impl GroupData {
    fn allocate(
        columns: Vec<usize>,
        widths: &[usize],
        cycles: usize,
        max_bits: usize,
    ) -> Result<Self, SourceError> {
        let mut selected_widths = Vec::with_capacity(columns.len());
        for &column in &columns {
            let &bits = widths.get(column).ok_or(SourceError::Column {
                column,
                columns: widths.len(),
            })?;
            if bits > max_bits {
                return Err(SourceError::GroupWidth {
                    column,
                    bits,
                    max_bits,
                });
            }
            selected_widths.push(bits);
        }
        let len = cycles
            .checked_mul(columns.len())
            .ok_or(SourceError::ValidationScratchSize {
                name: "group bytes",
                len: cycles,
                element_size: columns.len(),
            })?;
        check_validation_size::<u8>(len, "group bytes")?;
        Ok(Self {
            columns,
            widths: selected_widths,
            cycles,
            bytes: unsafe_allocate_zero_vec(len),
        })
    }
}

#[derive(Clone, Copy)]
struct GroupTarget {
    group: usize,
    position: usize,
    optional: bool,
}
struct GroupChunk<'a> {
    bytes: &'a mut [u8],
    stride: usize,
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
    targets: Vec<Vec<GroupTarget>>,
}

impl CycleValidation<'_> {
    fn validate_chunk(
        &self,
        source: &impl CycleSource,
        cycles: Range<usize>,
        rows: usize,
        output: &mut [(usize, GroupChunk<'_>)],
    ) -> Result<(), SourceError> {
        let mut fault = cycles.clone().find_map(|cycle| {
            let row = source.bytecode_index(cycle);
            (row >= rows).then_some((cycle, 0, SourceError::BytecodeIndex { cycle, row, rows }))
        });
        let end = fault
            .as_ref()
            .map_or(cycles.end, |(cycle, _, _)| *cycle)
            .min(source.cycles());
        let source_columns = source.digit_columns();
        macro_rules! scan {
            ($column:expr, $bound:expr, $matches:expr, $slots:expr, $store:expr, $present:expr) => {{
                let column = $column;
                let bound = $bound;
                for (cycle, slot) in (cycles.start..end).zip($slots) {
                    let digit = source.digit(column, cycle);
                    let error = if let Some(digit) = digit.filter(|&digit| digit >= bound) {
                        Some(SourceError::Digit {
                            column,
                            cycle,
                            digit,
                            bound,
                        })
                    } else if let Some(error) = $matches(digit, cycle) {
                        Some(error)
                    } else if $present && digit.is_none() {
                        Some(SourceError::MissingDigit { column, cycle })
                    } else {
                        $store(slot, digit, cycle);
                        None
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
        macro_rules! column {
            ($column:expr, $bound:expr, $matches:expr) => {{
                if $column >= source_columns {
                    return Err(SourceError::Column {
                        column: $column,
                        columns: source_columns,
                    });
                }
                let targets = &self.targets[$column];
                match targets.as_slice() {
                    [] => scan!(
                        $column,
                        $bound,
                        $matches,
                        std::iter::repeat(()),
                        |_, _, _| {},
                        false
                    ),
                    [target] => {
                        let GroupChunk { bytes, stride } = &mut output[target.group].1;
                        let slots = bytes.iter_mut().skip(target.position).step_by(*stride);
                        if target.optional {
                            scan!(
                                $column,
                                $bound,
                                $matches,
                                slots,
                                |slot: &mut u8, digit: Option<usize>, _| *slot =
                                    digit.map_or(0, |d| (d + 1) as u8),
                                false
                            );
                        } else {
                            scan!(
                                $column,
                                $bound,
                                $matches,
                                slots,
                                |slot: &mut u8, digit: Option<usize>, _| *slot =
                                    digit.unwrap_or(0) as u8,
                                true
                            );
                        }
                    }
                    _ => {
                        let present = targets.iter().any(|target| !target.optional);
                        scan!(
                            $column,
                            $bound,
                            $matches,
                            std::iter::repeat(()),
                            |_, digit: Option<usize>, cycle: usize| {
                                for target in targets {
                                    let group = &mut output[target.group].1;
                                    let byte = if target.optional {
                                        digit.map_or(0, |d| (d + 1) as u8)
                                    } else {
                                        digit.unwrap_or(0) as u8
                                    };
                                    group.bytes
                                        [(cycle - cycles.start) * group.stride + target.position] =
                                        byte;
                                }
                            },
                            present
                        );
                    }
                }
            }};
        }
        for &(column, bound) in &self.per_cycle {
            column!(column, bound, |_, _| None);
        }
        for &(column, bound, cache) in &self.row_bytes {
            column!(column, bound, |digit: Option<usize>, cycle| {
                let row = source.bytecode_index(cycle);
                (digit.map_or(0, |value| value + 1)
                    != cache[row].map_or(0, |value| usize::from(value.get())))
                .then_some(SourceError::RowDigit { column, cycle, row })
            });
        }
        for &(column, bound, cache) in &self.row_words {
            column!(column, bound, |digit: Option<usize>, cycle| {
                let row = source.bytecode_index(cycle);
                (digit.map_or(0, |value| value + 1) != cache[row].map_or(0, NonZeroUsize::get))
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
/// After dimension and request checks, fault order is: all rows before all cycles; rows ascending, then columns
/// ascending; cycles ascending, then `BytecodeIndex`, then columns ascending,
/// and within a column `Digit`, `RowDigit`, `MissingDigit`. Each column scan
/// stops at its first fault; each chunk chooses the least cycle and rank, and
/// preparation returns the fault of the first offending chunk. A row-based
/// column reads no cache entry at an invalid bytecode index. Parallel
/// completion cannot change this order.
/// The guarantee relies on the immutability contract of [`CycleSource`].
#[derive(Debug, Clone)]
pub struct ValidatedTrace<S: CycleSource> {
    source: Arc<S>,
}

impl<S: CycleSource> ValidatedTrace<S> {
    pub fn new(source: Arc<S>) -> Result<Self, SourceError> {
        Self::prepare(source, PrepareRequest::default()).map(|(trace, _)| trace)
    }

    /// Validates once and writes requested byte groups in that same parallel
    /// walk. Request columns and widths are rejected before any digit read.
    /// Group buffers have exactly `cycles * columns` bytes; row caches and
    /// chunk descriptors are temporary, with no allocation per chunk.
    pub fn prepare(
        source: Arc<S>,
        request: PrepareRequest,
    ) -> Result<(Self, PreparedGroups), SourceError> {
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
        let mut groups = PreparedGroups {
            present: request
                .present
                .into_iter()
                .map(|columns| GroupData::allocate(columns, &widths, cycles, 8).map(PresentGroup))
                .collect::<Result<_, _>>()?,
            optional: request
                .optional
                .into_iter()
                .map(|columns| GroupData::allocate(columns, &widths, cycles, 7).map(OptionalGroup))
                .collect::<Result<_, _>>()?,
        };
        let mut targets = vec![Vec::new(); columns];
        let mut group_count = 0;
        for (group, optional) in groups
            .present
            .iter()
            .map(|group| (&group.0, false))
            .chain(groups.optional.iter().map(|group| (&group.0, true)))
        {
            if group.columns.is_empty() {
                continue;
            }
            for (position, &column) in group.columns.iter().enumerate() {
                targets[column].push(GroupTarget {
                    group: group_count,
                    position,
                    optional,
                });
            }
            group_count += 1;
        }
        if !row_digits.is_empty() {
            let chunk_count = rows.div_ceil(ROW_CHUNK);
            let descriptors = chunk_count.checked_mul(row_digits.len()).ok_or(
                SourceError::ValidationScratchSize {
                    name: "row chunk metadata",
                    len: chunk_count,
                    element_size: size_of::<(usize, RowChunk<'_>)>(),
                },
            )?;
            check_validation_size::<(usize, RowChunk<'_>)>(descriptors, "row chunk metadata")?;
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
            targets,
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
        let chunk_count = cycles / geometry.chunk_len();
        let validate = |chunk: usize, output: &mut [(usize, GroupChunk<'_>)]| {
            let start = chunk * geometry.chunk_len();
            validation.validate_chunk(
                source.as_ref(),
                start..start + geometry.chunk_len(),
                rows,
                output,
            )
        };
        let error = if group_count == 0 {
            (0..chunk_count)
                .into_par_iter()
                .map(|chunk| validate(chunk, &mut []))
                .find_first(Result::is_err)
        } else {
            let len =
                chunk_count
                    .checked_mul(group_count)
                    .ok_or(SourceError::ValidationScratchSize {
                        name: "group chunk metadata",
                        len: chunk_count,
                        element_size: size_of::<GroupChunk<'_>>(),
                    })?;
            check_validation_size::<(usize, GroupChunk<'_>)>(len, "group chunk metadata")?;
            let mut chunks = Vec::with_capacity(len);
            let mut group_index = 0;
            for group in groups
                .present
                .iter_mut()
                .map(|group| &mut group.0)
                .chain(groups.optional.iter_mut().map(|group| &mut group.0))
            {
                let stride = group.columns.len();
                if stride == 0 {
                    continue;
                }
                for (chunk, bytes) in group
                    .bytes
                    .chunks_mut(geometry.chunk_len() * stride)
                    .enumerate()
                {
                    chunks.push((
                        chunk * group_count + group_index,
                        GroupChunk { bytes, stride },
                    ));
                }
                group_index += 1;
            }
            chunks.sort_unstable_by_key(|(chunk, _)| *chunk);
            chunks
                .par_chunks_mut(group_count)
                .enumerate()
                .map(|(chunk, output)| validate(chunk, output))
                .find_first(Result::is_err)
        };
        if let Some(Err(error)) = error {
            return Err(error);
        }
        Ok((Self { source }, groups))
    }

    pub fn source(&self) -> &Arc<S> {
        &self.source
    }
}
