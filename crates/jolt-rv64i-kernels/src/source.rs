//! Shared packed inputs for sum-check kernels. Digits are absent or smaller than
//! `2^bits(column)`; a row-based column is a function of the bytecode row alone.

use crate::par::CycleChunks;
use jolt_kernels::optimized::lazy_ra::ChunkIndexSource;
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;
use std::mem::size_of;
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
                *slot = encode_digit(self.digit(column, cycle));
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
    #[error("column {column} width {bits} exceeds the 15-bit source limit")]
    Width { column: usize, bits: usize },
    #[error("{columns} digit columns exceed the stack tile limit {max_columns}")]
    ColumnCapacity { columns: usize, max_columns: usize },
    #[error("bulk digit {encoded} disagrees with {digit:?} at column {column}, cycle {cycle}")]
    DigitView {
        column: usize,
        cycle: usize,
        encoded: u16,
        digit: Option<usize>,
    },
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
enum GroupColumns<'a> {
    RunFive(usize),
    RunEight(usize),
    Five([usize; 5]),
    Eight([usize; 8]),
    General(&'a [usize]),
}

impl<'a> GroupColumns<'a> {
    #[expect(
        clippy::expect_used,
        reason = "the matching length fixes the array width"
    )]
    fn new(columns: &'a [usize]) -> Self {
        let contiguous = columns.windows(2).all(|pair| pair[0] + 1 == pair[1]);
        match columns.len() {
            5 if contiguous => Self::RunFive(columns[0]),
            8 if contiguous => Self::RunEight(columns[0]),
            5 => Self::Five(columns.try_into().expect("five columns")),
            8 => Self::Eight(columns.try_into().expect("eight columns")),
            _ => Self::General(columns),
        }
    }
}

struct GroupChunk<'a> {
    bytes: &'a mut [u8],
    columns: &'a GroupColumns<'a>,
    bias: u16,
}

const ROW_CHUNK: usize = 4096;
const TILE_ENTRIES: usize = 4096;
const MAX_COLUMNS: usize = 128;

struct RowColumn {
    column: usize,
    bound: usize,
}

struct RowRun {
    columns: Range<usize>,
    cache: Range<usize>,
    compare: fn(&[u16], &[u16]) -> bool,
}

fn compare_general(values: &[u16], expected: &[u16]) -> bool {
    values
        .iter()
        .zip(expected)
        .fold(0_u16, |bad, (a, b)| bad | (a ^ b))
        != 0
}

#[expect(
    clippy::expect_used,
    reason = "the row plan selects this function from the checked run width"
)]
fn compare_run<const N: usize>(values: &[u16], expected: &[u16]) -> bool {
    let values: &[u16; N] = values.try_into().expect("compiled row run");
    let expected: &[u16; N] = expected.try_into().expect("compiled cache run");
    values
        .iter()
        .zip(expected)
        .fold(0_u16, |bad, (a, b)| bad | (a ^ b))
        != 0
}

trait Writer {
    type Plan: Copy;
    fn plan(&self) -> Self::Plan;
    fn check(&self, plan: &Self::Plan, columns: usize, cycles: usize);
    fn write(&mut self, digits: &[u16], offset: usize, plan: &Self::Plan);
}
struct EmptyWriter;
impl Writer for EmptyWriter {
    type Plan = ();
    fn plan(&self) {}
    fn check(&self, &(): &(), _: usize, _: usize) {}
    fn write(&mut self, _: &[u16], _: usize, &(): &()) {}
}
struct FixedWriter<'a, const N: usize> {
    indices: [usize; N],
    bytes: &'a mut [[u8; N]],
    bias: u16,
}
impl<'a, const N: usize> FixedWriter<'a, N> {
    #[expect(
        clippy::expect_used,
        reason = "chunk descriptors cover all cycles and tile ranges partition each chunk"
    )]
    fn new(
        indices: [usize; N],
        bytes: &'a mut [u8],
        bias: u16,
        offset: usize,
        cycles: usize,
    ) -> Self {
        Self {
            indices,
            bytes: bytes
                .as_chunks_mut::<N>()
                .0
                .get_mut(offset..offset + cycles)
                .expect("compiled output extent"),
            bias,
        }
    }
}
impl<const N: usize> Writer for FixedWriter<'_, N> {
    type Plan = ([usize; N], u16);
    #[inline(always)]
    fn plan(&self) -> Self::Plan {
        (self.indices, self.bias)
    }
    #[inline(always)]
    fn check(&self, plan: &Self::Plan, columns: usize, cycles: usize) {
        assert!(!plan.0.iter().any(|&index| index >= columns));
        assert!(self.bytes.len() >= cycles);
    }
    #[inline(always)]
    fn write(&mut self, digits: &[u16], offset: usize, plan: &Self::Plan) {
        for (slot, &column) in self.bytes[offset].iter_mut().zip(&plan.0) {
            *slot = digits[column].wrapping_sub(plan.1) as u8;
        }
    }
}
struct RunWriter<'a, const N: usize> {
    start: usize,
    bytes: &'a mut [[u8; N]],
    bias: u16,
}
impl<'a, const N: usize> RunWriter<'a, N> {
    fn new(start: usize, bytes: &'a mut [u8], bias: u16, offset: usize, cycles: usize) -> Self {
        let writer = FixedWriter::<N>::new([0; N], bytes, bias, offset, cycles);
        Self {
            start,
            bytes: writer.bytes,
            bias,
        }
    }
}
impl<const N: usize> Writer for RunWriter<'_, N> {
    type Plan = (usize, u16);
    #[inline(always)]
    fn plan(&self) -> Self::Plan {
        (self.start, self.bias)
    }
    #[inline(always)]
    fn check(&self, plan: &Self::Plan, columns: usize, cycles: usize) {
        assert!(plan.0.checked_add(N).is_some_and(|end| end <= columns));
        assert!(self.bytes.len() >= cycles);
    }
    #[inline(always)]
    #[expect(
        clippy::expect_used,
        reason = "run width is fixed and its extent is checked before the cycle loop"
    )]
    fn write(&mut self, digits: &[u16], offset: usize, plan: &Self::Plan) {
        let values: &[u16; N] = digits[plan.0..plan.0 + N]
            .try_into()
            .expect("compiled contiguous run");
        self.bytes[offset] = (*values).map(|digit| digit.wrapping_sub(plan.1) as u8);
    }
}

struct GeneralWriter<'a> {
    columns: &'a [usize],
    bytes: &'a mut [u8],
    bias: u16,
}
impl<'a> GeneralWriter<'a> {
    fn new(
        columns: &'a [usize],
        bytes: &'a mut [u8],
        bias: u16,
        offset: usize,
        cycles: usize,
    ) -> Self {
        let width = columns.len();
        Self {
            columns,
            bytes: &mut bytes[offset * width..(offset + cycles) * width],
            bias,
        }
    }
}
impl<'a> Writer for GeneralWriter<'a> {
    type Plan = (&'a [usize], u16);
    fn plan(&self) -> Self::Plan {
        (self.columns, self.bias)
    }
    fn check(&self, plan: &Self::Plan, columns: usize, _: usize) {
        assert!(!plan.0.iter().any(|&index| index >= columns));
    }
    #[inline]
    fn write(&mut self, digits: &[u16], offset: usize, plan: &Self::Plan) {
        let width = plan.0.len();
        for (slot, &column) in self.bytes[offset * width..(offset + 1) * width]
            .iter_mut()
            .zip(plan.0)
        {
            *slot = digits[column].wrapping_sub(plan.1) as u8;
        }
    }
}
struct ManyWriter<'a, 'b> {
    output: &'a mut [Option<GroupChunk<'b>>],
    offset: usize,
}
impl Writer for ManyWriter<'_, '_> {
    type Plan = ();
    fn plan(&self) {}
    fn check(&self, &(): &(), _: usize, _: usize) {}
    fn write(&mut self, digits: &[u16], offset: usize, &(): &()) {
        let offset = self.offset + offset;
        for group in self.output.iter_mut().flatten() {
            match *group.columns {
                GroupColumns::RunFive(start) => {
                    for (slot, &digit) in group.bytes[offset * 5..(offset + 1) * 5]
                        .iter_mut()
                        .zip(&digits[start..start + 5])
                    {
                        *slot = digit.wrapping_sub(group.bias) as u8;
                    }
                }
                GroupColumns::RunEight(start) => {
                    for (slot, &digit) in group.bytes[offset * 8..(offset + 1) * 8]
                        .iter_mut()
                        .zip(&digits[start..start + 8])
                    {
                        *slot = digit.wrapping_sub(group.bias) as u8;
                    }
                }
                GroupColumns::Five(columns) => {
                    for (slot, column) in group.bytes[offset * 5..(offset + 1) * 5]
                        .iter_mut()
                        .zip(columns)
                    {
                        *slot = digits[column].wrapping_sub(group.bias) as u8;
                    }
                }
                GroupColumns::Eight(columns) => {
                    for (slot, column) in group.bytes[offset * 8..(offset + 1) * 8]
                        .iter_mut()
                        .zip(columns)
                    {
                        *slot = digits[column].wrapping_sub(group.bias) as u8;
                    }
                }
                GroupColumns::General(columns) => {
                    let width = columns.len();
                    for (slot, &column) in group.bytes[offset * width..(offset + 1) * width]
                        .iter_mut()
                        .zip(columns)
                    {
                        *slot = digits[column].wrapping_sub(group.bias) as u8;
                    }
                }
            }
        }
    }
}

trait RowChecker {
    fn check(&self, columns: usize, rows: usize, cache: &[u16]);
    fn compare(&self, row: usize, digits: &[u16], cache: &[u16]) -> bool;
}

struct FixedRows<const A: usize, const B: usize, const WIDTH: usize> {
    first: usize,
    second: usize,
}
impl<const A: usize, const B: usize, const WIDTH: usize> RowChecker for FixedRows<A, B, WIDTH> {
    #[inline(always)]
    fn check(&self, columns: usize, rows: usize, cache: &[u16]) {
        if let Some(cached_rows) = cache.len().checked_div(WIDTH) {
            assert!(cached_rows >= rows);
            assert!(self.first.checked_add(A).is_some_and(|end| end <= columns));
            if B != 0 {
                assert!(self.second.checked_add(B).is_some_and(|end| end <= columns));
            }
        }
    }
    #[inline(always)]
    fn compare(&self, row: usize, digits: &[u16], cache: &[u16]) -> bool {
        if WIDTH == 0 {
            return false;
        }
        let expected = &cache.as_chunks::<WIDTH>().0[row];
        let mut difference = if A == 1 {
            digits[self.first] ^ expected[0]
        } else {
            digits[self.first..self.first + A]
                .iter()
                .zip(&expected[..A])
                .fold(0_u16, |difference, (a, b)| difference | (a ^ b))
        };
        if B != 0 {
            difference |= if B == 1 {
                digits[self.second] ^ expected[A]
            } else {
                digits[self.second..self.second + B]
                    .iter()
                    .zip(&expected[A..A + B])
                    .fold(0_u16, |difference, (a, b)| difference | (a ^ b))
            };
        }
        difference != 0
    }
}

struct GeneralRows<'a> {
    runs: &'a [RowRun],
    width: usize,
}
impl RowChecker for GeneralRows<'_> {
    fn check(&self, _: usize, _: usize, _: &[u16]) {}
    #[inline]
    fn compare(&self, row: usize, digits: &[u16], cache: &[u16]) -> bool {
        let cache = &cache[row * self.width..(row + 1) * self.width];
        self.runs.iter().fold(false, |bad, run| {
            let difference = (run.compare)(&digits[run.columns.clone()], &cache[run.cache.clone()]);
            bad || difference
        })
    }
}

struct TilePass<'a> {
    source: &'a dyn CycleSource,
    start: usize,
    rows: usize,
    tile: &'a [u16],
    maxima: &'a mut [u16; MAX_COLUMNS],
    minima: &'a mut [u16; MAX_COLUMNS],
    rejected: &'a mut bool,
}

struct CycleValidation<'a> {
    bounds: &'a [u16],
    present: &'a [u16],
    row_columns: &'a [RowColumn],
    row_runs: &'a [RowRun],
    row_cache: &'a [u16],
}

impl CycleValidation<'_> {
    fn validate_chunk(
        &self,
        source: &impl CycleSource,
        cycles: Range<usize>,
        rows: usize,
        output: &mut [Option<GroupChunk<'_>>],
    ) -> Result<(), SourceError> {
        let columns = self.bounds.len();
        let tile_cycles = TILE_ENTRIES / columns.max(1);
        let mut buffer = [0_u16; TILE_ENTRIES];
        for start in (cycles.start..cycles.end).step_by(tile_cycles) {
            let end = (start + tile_cycles).min(cycles.end);
            let tile = &mut buffer[..(end - start) * columns];
            source.digits(start..end, tile);
            let mut maxima = [0_u16; MAX_COLUMNS];
            let mut minima = [u16::MAX; MAX_COLUMNS];
            let mut rejected = false;
            if columns == 0 {
                rejected = (start..end).any(|cycle| source.bytecode_index(cycle) >= rows);
            } else {
                self.fast_source_tile(
                    source,
                    TilePass {
                        source,
                        start,
                        rows,
                        tile,
                        maxima: &mut maxima,
                        minima: &mut minima,
                        rejected: &mut rejected,
                    },
                    output,
                    start - cycles.start,
                );
            }
            rejected |= maxima[..columns]
                .iter()
                .zip(self.bounds)
                .any(|(&maximum, &bound)| maximum > bound);
            rejected |= minima[..columns]
                .iter()
                .zip(self.present)
                .any(|(&minimum, &present)| minimum < present);
            if rejected {
                self.scalar_ladder(source, start..end, rows, tile)?;
            }
        }
        Ok(())
    }

    fn fast_source_tile<S: CycleSource>(
        &self,
        source: &S,
        pass: TilePass<'_>,
        output: &mut [Option<GroupChunk<'_>>],
        offset: usize,
    ) {
        match self.row_runs {
            [run] if run.columns.len() == 1 => self.fast_tile_rows(
                pass,
                output,
                offset,
                FixedRows::<1, 0, 1> {
                    first: run.columns.start,
                    second: 0,
                },
                source,
            ),
            [first, second] if first.columns.len() == 5 && second.columns.len() == 6 => self
                .fast_tile_rows(
                    pass,
                    output,
                    offset,
                    FixedRows::<5, 6, { 5 + 6 }> {
                        first: first.columns.start,
                        second: second.columns.start,
                    },
                    source,
                ),
            _ => self.fast_tile(pass, output, offset),
        }
    }

    fn fast_tile(&self, pass: TilePass<'_>, output: &mut [Option<GroupChunk<'_>>], offset: usize) {
        let source = pass.source;
        macro_rules! dispatch {
            ($checker:expr) => {
                self.fast_tile_rows(pass, output, offset, $checker, source)
            };
        }
        macro_rules! pair {
            ($first:expr, $second:expr; $($a:literal),*) => {
                match $first.columns.len() {
                    $($a => match $second.columns.len() {
                        1 => dispatch!(FixedRows::<$a, 1, {$a + 1}> { first: $first.columns.start, second: $second.columns.start }),
                        5 => dispatch!(FixedRows::<$a, 5, {$a + 5}> { first: $first.columns.start, second: $second.columns.start }),
                        6 => dispatch!(FixedRows::<$a, 6, {$a + 6}> { first: $first.columns.start, second: $second.columns.start }),
                        _ => dispatch!(GeneralRows { runs: self.row_runs, width: self.row_columns.len() }),
                    },)*
                    _ => dispatch!(GeneralRows { runs: self.row_runs, width: self.row_columns.len() }),
                }
            };
        }
        match self.row_runs {
            [] => dispatch!(FixedRows::<0, 0, 0> {
                first: 0,
                second: 0
            }),
            [run] => match run.columns.len() {
                1 => dispatch!(FixedRows::<1, 0, 1> {
                    first: run.columns.start,
                    second: 0
                }),
                5 => dispatch!(FixedRows::<5, 0, 5> {
                    first: run.columns.start,
                    second: 0
                }),
                6 => dispatch!(FixedRows::<6, 0, 6> {
                    first: run.columns.start,
                    second: 0
                }),
                _ => dispatch!(GeneralRows {
                    runs: self.row_runs,
                    width: self.row_columns.len()
                }),
            },
            [first, second] => {
                pair!(first, second; 1, 5, 6);
            }
            _ => dispatch!(GeneralRows {
                runs: self.row_runs,
                width: self.row_columns.len()
            }),
        }
    }

    fn fast_tile_rows<R: RowChecker, S: CycleSource + ?Sized>(
        &self,
        pass: TilePass<'_>,
        output: &mut [Option<GroupChunk<'_>>],
        offset: usize,
        checker: R,
        source: &S,
    ) {
        macro_rules! tails {
            ($a:expr, $b:expr, $c:expr; $($tail:literal),*) => {
                match self.bounds.len() % 16 {
                    $($tail => self.healthy::<$tail, _, _, _, _, _>(source, pass.start, pass.rows, pass.tile, pass.maxima, pass.minima, pass.rejected, $a, $b, $c, checker),)*
                    _ => unreachable!("remainder is below sixteen"),
                }
            };
            ($a:expr, $b:expr, $c:expr) => {
                tails!($a, $b, $c; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
            };
        }
        macro_rules! writer {
            ($group:expr, $name:ident, $body:block) => {{
                let group = $group;
                match group.map(|g| (*g.columns, &mut *g.bytes, g.bias)) {
                    None => {
                        let $name = EmptyWriter;
                        $body
                    }
                    Some((GroupColumns::RunFive(start), bytes, bias)) => {
                        let $name = RunWriter::<5>::new(
                            start,
                            bytes,
                            bias,
                            offset,
                            pass.tile.len() / self.bounds.len(),
                        );
                        $body
                    }
                    Some((GroupColumns::RunEight(start), bytes, bias)) => {
                        let $name = RunWriter::<8>::new(
                            start,
                            bytes,
                            bias,
                            offset,
                            pass.tile.len() / self.bounds.len(),
                        );
                        $body
                    }
                    Some((GroupColumns::Five(indices), bytes, bias)) => {
                        let $name = FixedWriter::<5>::new(
                            indices,
                            bytes,
                            bias,
                            offset,
                            pass.tile.len() / self.bounds.len(),
                        );
                        $body
                    }
                    Some((GroupColumns::Eight(indices), bytes, bias)) => {
                        let $name = FixedWriter::<8>::new(
                            indices,
                            bytes,
                            bias,
                            offset,
                            pass.tile.len() / self.bounds.len(),
                        );
                        $body
                    }
                    Some((GroupColumns::General(columns), bytes, bias)) => {
                        let $name = GeneralWriter::new(
                            columns,
                            bytes,
                            bias,
                            offset,
                            pass.tile.len() / self.bounds.len(),
                        );
                        $body
                    }
                }
            }};
        }
        if output.len() <= 3 {
            let mut groups = output.iter_mut();
            writer!(groups.next().and_then(Option::as_mut), a, {
                writer!(groups.next().and_then(Option::as_mut), b, {
                    writer!(groups.next().and_then(Option::as_mut), c, {
                        tails!(a, b, c);
                    });
                });
            });
        } else {
            let a = ManyWriter { output, offset };
            tails!(a, EmptyWriter, EmptyWriter);
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "tail dispatch fixes the conversion widths before the cycle loop"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "separate slice references preserve LLVM noalias for the vectorized accumulator loop"
    )]
    fn healthy<
        const TAIL: usize,
        A: Writer,
        B: Writer,
        C: Writer,
        R: RowChecker,
        S: CycleSource + ?Sized,
    >(
        &self,
        source: &S,
        start: usize,
        rows: usize,
        tile: &[u16],
        maxima: &mut [u16; MAX_COLUMNS],
        minima: &mut [u16; MAX_COLUMNS],
        rejected: &mut bool,
        mut a: A,
        mut b: B,
        mut c: C,
        checker: R,
    ) {
        let columns = self.bounds.len();
        let prefix = columns - TAIL;
        let (max_prefix, max_tail) = maxima[..columns].split_at_mut(prefix);
        let (min_prefix, min_tail) = minima[..columns].split_at_mut(prefix);
        let max_tail: &mut [u16; TAIL] = max_tail.try_into().expect("compiled tail width");
        let min_tail: &mut [u16; TAIL] = min_tail.try_into().expect("compiled tail width");
        let max_prefix = max_prefix.as_chunks_mut::<16>().0;
        let min_prefix = min_prefix.as_chunks_mut::<16>().0;
        let tile_cycles = tile.len() / columns;
        // Local copies keep source calls and output writes from invalidating
        // the index checks made before this loop.
        let a_plan = a.plan();
        let b_plan = b.plan();
        let c_plan = c.plan();
        a.check(&a_plan, columns, tile_cycles);
        b.check(&b_plan, columns, tile_cycles);
        c.check(&c_plan, columns, tile_cycles);
        checker.check(columns, rows, self.row_cache);
        for (offset, digits) in (0..tile_cycles).zip(tile.chunks_exact(columns)) {
            let row = source.bytecode_index(start + offset);
            if row >= rows {
                *rejected = true;
            } else {
                *rejected |= checker.compare(row, digits, self.row_cache);
            }
            let (values, tail) = digits.split_at(prefix);
            for ((maximum, minimum), values) in max_prefix
                .iter_mut()
                .zip(min_prefix.iter_mut())
                .zip(values.as_chunks::<16>().0)
            {
                for ((maximum, minimum), &value) in
                    maximum.iter_mut().zip(minimum.iter_mut()).zip(values)
                {
                    *maximum = (*maximum).max(value);
                    *minimum = (*minimum).min(value);
                }
            }
            let tail: &[u16; TAIL] = tail.try_into().expect("compiled tail width");
            for ((maximum, minimum), &value) in
                max_tail.iter_mut().zip(min_tail.iter_mut()).zip(tail)
            {
                *maximum = (*maximum).max(value);
                *minimum = (*minimum).min(value);
            }
            a.write(digits, offset, &a_plan);
            b.write(digits, offset, &b_plan);
            c.write(digits, offset, &c_plan);
        }
    }

    #[cold]
    #[inline(never)]
    fn scalar_ladder(
        &self,
        source: &impl CycleSource,
        cycles: Range<usize>,
        rows: usize,
        tile: &[u16],
    ) -> Result<(), SourceError> {
        for cycle in cycles.clone() {
            let row = source.bytecode_index(cycle);
            if row >= rows {
                return Err(SourceError::BytecodeIndex { cycle, row, rows });
            }
            for (column, &bound) in self.bounds.iter().enumerate() {
                let digit = source.digit(column, cycle);
                let encoded = tile[(cycle - cycles.start) * self.bounds.len() + column];
                if encoded != encode_digit(digit) {
                    return Err(SourceError::DigitView {
                        column,
                        cycle,
                        encoded,
                        digit,
                    });
                }
                if let Some(digit) = digit.filter(|&digit| digit >= usize::from(bound)) {
                    return Err(SourceError::Digit {
                        column,
                        cycle,
                        digit,
                        bound: usize::from(bound),
                    });
                }
                if source.by_row(column) {
                    let position = self.row_columns.iter().position(|row| row.column == column);
                    let cache_matches = position.is_none_or(|position| {
                        self.row_cache[row * self.row_columns.len() + position] == encoded
                    });
                    if !cache_matches || digit != source.row_digit(column, row) {
                        return Err(SourceError::RowDigit { column, cycle, row });
                    }
                }
                if self.present[column] != 0 && digit.is_none() {
                    return Err(SourceError::MissingDigit { column, cycle });
                }
            }
        }
        // Every fast predicate forces a ladder fault on an immutable source:
        // bad index, encoded range/presence, or disagreement with the row cache.
        // A source that changes between reads violates the trait contract; this
        // total recheck does not introduce an impossible arm or an unsafe premise.
        Ok(())
    }
}

#[inline]
fn encode_digit(digit: Option<usize>) -> u16 {
    digit.map_or(0, |digit| {
        digit.saturating_add(1).min(u16::MAX as usize) as u16
    })
}

/// Shared source whose dimensions, indices and every digit column are checked.
/// Successful construction reads each row-based digit once and each tile once.
/// Rejected tiles are reread through the scalar accessors to locate their fault.
/// Temporary row caches are dropped before returning; the source owns its data.
/// After dimension and request checks, fault order is: all rows before all cycles; rows ascending, then columns
/// ascending; cycles ascending, then `BytecodeIndex`, then columns ascending,
/// and within a column `DigitView`, `Digit`, `RowDigit`, `MissingDigit`.
/// Tiles are visited in ascending cycle order; rejected tiles use the scalar
/// ladder in that order. Preparation returns the fault of the first offending chunk. A row-based
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
    /// chunk descriptors are temporary, with no allocation per chunk. Group
    /// indices and row runs are compiled before the cycle walk; fixed-width
    /// writers and comparisons are selected outside its cycle loop.
    /// At most 128 digit columns of at most 15 bits are accepted. Each active
    /// worker uses an 8 KiB stack tile and two 256-byte column accumulators.
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
        if columns > MAX_COLUMNS {
            return Err(SourceError::ColumnCapacity {
                columns,
                max_columns: MAX_COLUMNS,
            });
        }
        check_validation_size::<usize>(columns, "digit widths")?;
        check_validation_size::<RowColumn>(columns, "row digit cache columns")?;
        let mut widths = Vec::with_capacity(columns);
        let mut row_columns = Vec::new();
        let mut bounds = Vec::with_capacity(columns);
        for column in 0..columns {
            let bits = source.bits(column);
            if bits > 15 {
                return Err(SourceError::Width { column, bits });
            }
            widths.push(bits);
            let bound = 1 << bits;
            bounds.push(bound as u16);
            if source.by_row(column) {
                row_columns.push(RowColumn { column, bound });
            }
        }
        let cache_len =
            rows.checked_mul(row_columns.len())
                .ok_or(SourceError::ValidationScratchSize {
                    name: "row digit cache",
                    len: rows,
                    element_size: row_columns.len(),
                })?;
        check_validation_size::<u16>(cache_len, "row digit cache")?;
        let mut row_cache = unsafe_allocate_zero_vec::<u16>(cache_len);
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
        let mut present = vec![0_u16; columns];
        for group in &groups.present {
            for &column in &group.0.columns {
                present[column] = 1;
            }
        }
        let group_count = groups
            .present
            .iter()
            .filter(|g| !g.0.columns.is_empty())
            .count()
            + groups
                .optional
                .iter()
                .filter(|g| !g.0.columns.is_empty())
                .count();
        if !row_columns.is_empty() {
            if let Some(Some(error)) = row_cache
                .par_chunks_mut(ROW_CHUNK * row_columns.len())
                .enumerate()
                .map(|(chunk, cache)| {
                    for (offset, output) in cache.chunks_exact_mut(row_columns.len()).enumerate() {
                        let row = chunk * ROW_CHUNK + offset;
                        for (slot, RowColumn { column, bound }) in
                            output.iter_mut().zip(&row_columns)
                        {
                            let digit = source.row_digit(*column, row);
                            if let Some(digit) = digit.filter(|digit| digit >= bound) {
                                return Some(SourceError::RowDigitRange {
                                    column: *column,
                                    row,
                                    digit,
                                    bound: *bound,
                                });
                            }
                            *slot = digit.map_or(0, |digit| (digit + 1) as u16);
                        }
                    }
                    None
                })
                .find_first(Option::is_some)
            {
                return Err(error);
            }
        }
        let mut row_runs: Vec<RowRun> = Vec::new();
        for (position, row) in row_columns.iter().enumerate() {
            if let Some(run) = row_runs
                .last_mut()
                .filter(|run| run.columns.end == row.column)
            {
                run.columns.end += 1;
                run.cache.end += 1;
            } else {
                row_runs.push(RowRun {
                    columns: row.column..row.column + 1,
                    cache: position..position + 1,
                    compare: compare_general,
                });
            }
        }
        for run in &mut row_runs {
            run.compare = match run.columns.len() {
                1 => compare_run::<1>,
                5 => compare_run::<5>,
                6 => compare_run::<6>,
                _ => compare_general,
            };
        }
        let validation = CycleValidation {
            bounds: &bounds,
            present: &present,
            row_columns: &row_columns,
            row_runs: &row_runs,
            row_cache: &row_cache,
        };
        let geometry = CycleChunks::new(cycles.ilog2() as usize, 0)
            .map_err(|_| SourceError::CycleCount { cycles })?;
        let chunk_count = cycles / geometry.chunk_len();
        let validate = |chunk: usize, output: &mut [Option<GroupChunk<'_>>]| {
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
            check_validation_size::<Option<GroupChunk<'_>>>(len, "group chunk metadata")?;
            let mut chunks: Vec<Option<GroupChunk<'_>>> =
                std::iter::repeat_with(|| None).take(len).collect();
            let views: Vec<_> = groups
                .present
                .iter_mut()
                .map(|group| (&group.0.columns[..], &mut group.0.bytes[..], false))
                .chain(
                    groups
                        .optional
                        .iter_mut()
                        .map(|group| (&group.0.columns[..], &mut group.0.bytes[..], true)),
                )
                .filter(|(columns, _, _)| !columns.is_empty())
                .collect();
            check_validation_size::<GroupColumns<'_>>(views.len(), "group plans")?;
            let plans: Vec<_> = views
                .iter()
                .map(|(columns, _, _)| GroupColumns::new(columns))
                .collect();
            for (group_index, ((columns, bytes, optional), plan)) in
                views.into_iter().zip(&plans).enumerate()
            {
                let stride = columns.len();
                for (slot, bytes) in chunks
                    .iter_mut()
                    .skip(group_index)
                    .step_by(group_count)
                    .zip(bytes.chunks_mut(geometry.chunk_len() * stride))
                {
                    *slot = Some(GroupChunk {
                        bytes,
                        columns: plan,
                        bias: u16::from(!optional),
                    });
                }
            }
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
