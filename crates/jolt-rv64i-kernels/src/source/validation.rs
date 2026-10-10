use super::{encode_digit, CycleSource, SourceError};
use std::ops::Range;

pub(super) const ROW_CHUNK: usize = 4096;
pub(super) const MAX_COLUMNS: usize = 128;
const TILE_ENTRIES: usize = 4096;

#[derive(Clone, Copy)]
pub(super) enum GroupColumns<'a> {
    RunFive(usize),
    Eight([usize; 8]),
    General(&'a [usize]),
}

impl<'a> GroupColumns<'a> {
    #[expect(
        clippy::expect_used,
        reason = "the matching length fixes the array width"
    )]
    pub(super) fn new(columns: &'a [usize]) -> Self {
        match columns.len() {
            5 if columns.windows(2).all(|pair| pair[0] + 1 == pair[1]) => Self::RunFive(columns[0]),
            8 => Self::Eight(columns.try_into().expect("eight columns")),
            _ => Self::General(columns),
        }
    }
}

pub(super) struct GroupChunk<'a> {
    pub(super) bytes: &'a mut [u8],
    pub(super) columns: &'a GroupColumns<'a>,
    pub(super) bias: u16,
}

#[inline]
fn group_byte(value: u16, bias: u16) -> u8 {
    value.wrapping_sub(bias) as u8
}

#[expect(
    clippy::expect_used,
    reason = "the validated run and fixed width establish the slice extent"
)]
fn narrow_run<const N: usize>(digits: &[u16], start: usize, bias: u16) -> [u8; N] {
    let values: &[u16; N] = digits[start..start + N].try_into().expect("compiled run");
    values.map(|value| group_byte(value, bias))
}

fn gather_group<const N: usize>(digits: &[u16], indices: [usize; N], bias: u16) -> [u8; N] {
    indices.map(|column| group_byte(digits[column], bias))
}

impl GroupChunk<'_> {
    fn write_cycle(&mut self, digits: &[u16], offset: usize) {
        match *self.columns {
            GroupColumns::RunFive(start) => {
                self.bytes.as_chunks_mut::<5>().0[offset] = narrow_run(digits, start, self.bias);
            }
            GroupColumns::Eight(indices) => {
                self.bytes.as_chunks_mut::<8>().0[offset] =
                    gather_group(digits, indices, self.bias);
            }
            GroupColumns::General(columns) => {
                let width = columns.len();
                for (slot, &column) in self.bytes[offset * width..(offset + 1) * width]
                    .iter_mut()
                    .zip(columns)
                {
                    *slot = group_byte(digits[column], self.bias);
                }
            }
        }
    }
}

pub(super) struct RowColumn {
    pub(super) column: usize,
    pub(super) bound: usize,
}

impl RowColumn {
    pub(super) fn pair(columns: &[Self]) -> Option<[usize; 2]> {
        if columns.len() != 5 + 6 {
            return None;
        }
        let first = columns[0].column;
        let second = columns[5].column;
        (columns[..5]
            .windows(2)
            .all(|pair| pair[0].column + 1 == pair[1].column)
            && columns[5..]
                .windows(2)
                .all(|pair| pair[0].column + 1 == pair[1].column)
            && second > first + 5)
            .then_some([first, second])
    }
}

trait Outputs {
    type Plan: Copy;
    fn plan(&self) -> Self::Plan;
    fn check(&self, plan: &Self::Plan, columns: usize, cycles: usize);
    fn write(&mut self, digits: &[u16], offset: usize, plan: &Self::Plan);
}

struct EmptyOutputs;
impl Outputs for EmptyOutputs {
    type Plan = ();
    fn plan(&self) {}
    fn check(&self, &(): &(), _: usize, _: usize) {}
    fn write(&mut self, _: &[u16], _: usize, &(): &()) {}
}

struct TailOutputs<'a> {
    starts: [usize; 2],
    first: &'a mut [[u8; 5]],
    second: &'a mut [[u8; 5]],
}

impl<'a> TailOutputs<'a> {
    #[expect(
        clippy::expect_used,
        reason = "chunk descriptors and tile ranges cover the checked output extent"
    )]
    fn new(
        first: &'a mut GroupChunk<'_>,
        second: &'a mut GroupChunk<'_>,
        starts: [usize; 2],
        offset: usize,
        cycles: usize,
    ) -> Self {
        Self {
            starts,
            first: first
                .bytes
                .as_chunks_mut::<5>()
                .0
                .get_mut(offset..offset + cycles)
                .expect("first output extent"),
            second: second
                .bytes
                .as_chunks_mut::<5>()
                .0
                .get_mut(offset..offset + cycles)
                .expect("second output extent"),
        }
    }
}

impl Outputs for TailOutputs<'_> {
    type Plan = [usize; 2];
    fn plan(&self) -> Self::Plan {
        self.starts
    }
    #[inline(always)]
    fn check(&self, starts: &Self::Plan, columns: usize, cycles: usize) {
        assert!(starts
            .iter()
            .all(|start| start.checked_add(5).is_some_and(|end| end <= columns)));
        assert!(self.first.len() >= cycles && self.second.len() >= cycles);
    }
    #[inline(always)]
    fn write(&mut self, digits: &[u16], offset: usize, starts: &Self::Plan) {
        self.first[offset] = narrow_run(digits, starts[0], 1);
        self.second[offset] = narrow_run(digits, starts[1], 1);
    }
}

struct RouterOutputs<'a> {
    tail: TailOutputs<'a>,
    indices: [usize; 8],
    optional: &'a mut [[u8; 8]],
}

impl Outputs for RouterOutputs<'_> {
    type Plan = ([usize; 2], [usize; 8]);
    fn plan(&self) -> Self::Plan {
        (self.tail.plan(), self.indices)
    }
    #[inline(always)]
    fn check(&self, plan: &Self::Plan, columns: usize, cycles: usize) {
        self.tail.check(&plan.0, columns, cycles);
        assert!(plan.1.iter().all(|&column| column < columns));
        assert!(self.optional.len() >= cycles);
    }
    #[inline(always)]
    fn write(&mut self, digits: &[u16], offset: usize, plan: &Self::Plan) {
        self.tail.write(digits, offset, &plan.0);
        self.optional[offset] = gather_group(digits, plan.1, 0);
    }
}

struct GeneralOutputs<'a, 'b> {
    groups: &'a mut [Option<GroupChunk<'b>>],
    offset: usize,
}
impl Outputs for GeneralOutputs<'_, '_> {
    type Plan = ();
    fn plan(&self) {}
    fn check(&self, &(): &(), _: usize, _: usize) {}
    fn write(&mut self, digits: &[u16], offset: usize, &(): &()) {
        for group in self.groups.iter_mut().flatten() {
            group.write_cycle(digits, self.offset + offset);
        }
    }
}

trait RowChecker {
    fn check(&self, columns: usize, rows: usize, cache: &[u16]);
    fn compare(&self, row: usize, digits: &[u16], cache: &[u16]) -> bool;
}

struct PairedRows {
    first: usize,
    second: usize,
}
impl RowChecker for PairedRows {
    fn check(&self, columns: usize, rows: usize, cache: &[u16]) {
        assert!(cache.as_chunks::<{ 5 + 6 }>().0.len() >= rows);
        assert!(self.first.checked_add(5).is_some_and(|end| end <= columns));
        assert!(self.second.checked_add(6).is_some_and(|end| end <= columns));
    }
    #[inline(always)]
    fn compare(&self, row: usize, digits: &[u16], cache: &[u16]) -> bool {
        let expected = &cache.as_chunks::<{ 5 + 6 }>().0[row];
        let first = digits[self.first..self.first + 5]
            .iter()
            .zip(&expected[..5])
            .fold(0_u16, |bad, (a, b)| bad | (a ^ b));
        let second = digits[self.second..self.second + 6]
            .iter()
            .zip(&expected[5..])
            .fold(0_u16, |bad, (a, b)| bad | (a ^ b));
        first | second != 0
    }
}

struct GeneralRows<'a> {
    columns: &'a [RowColumn],
}
impl RowChecker for GeneralRows<'_> {
    fn check(&self, _: usize, _: usize, _: &[u16]) {}
    fn compare(&self, row: usize, digits: &[u16], cache: &[u16]) -> bool {
        let width = self.columns.len();
        let expected = &cache[row * width..(row + 1) * width];
        self.columns
            .iter()
            .zip(expected)
            .fold(0_u16, |bad, (column, &value)| {
                bad | (digits[column.column] ^ value)
            })
            != 0
    }
}

struct TilePass<'a> {
    start: usize,
    rows: usize,
    tile: &'a [u16],
    maxima: &'a mut [u16; MAX_COLUMNS],
    minima: &'a mut [u16; MAX_COLUMNS],
    rejected: &'a mut bool,
}

pub(super) struct CycleValidation<'a> {
    pub(super) bounds: &'a [u16],
    pub(super) present: &'a [u16],
    pub(super) row_columns: &'a [RowColumn],
    pub(super) row_pair: Option<[usize; 2]>,
    pub(super) row_cache: &'a [u16],
}

impl CycleValidation<'_> {
    pub(super) fn validate_chunk(
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
                self.select_tile(
                    source,
                    TilePass {
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

    #[expect(
        clippy::expect_used,
        reason = "chunk descriptors cover the selected optional output extent"
    )]
    fn select_tile<S: CycleSource>(
        &self,
        source: &S,
        pass: TilePass<'_>,
        output: &mut [Option<GroupChunk<'_>>],
        offset: usize,
    ) {
        if self.bounds.len() % 16 == 5 {
            if let Some([first, second]) = self.row_pair {
                let checker = PairedRows { first, second };
                let cycles = pass.tile.len() / self.bounds.len();
                macro_rules! run {
                    ($outputs:expr) => {
                        self.validate_accumulate::<true, _, _, _>(
                            source,
                            pass.start,
                            pass.rows,
                            pass.tile,
                            pass.maxima,
                            pass.minima,
                            pass.rejected,
                            $outputs,
                            checker,
                        )
                    };
                }
                match output {
                    [] => return run!(EmptyOutputs),
                    [Some(a), Some(b)] if a.bias == 1 && b.bias == 1 => {
                        if let (GroupColumns::RunFive(x), GroupColumns::RunFive(y)) =
                            (*a.columns, *b.columns)
                        {
                            return run!(TailOutputs::new(a, b, [x, y], offset, cycles));
                        }
                    }
                    [Some(a), Some(b), Some(c)] if a.bias == 1 && b.bias == 1 && c.bias == 0 => {
                        if let (
                            GroupColumns::RunFive(x),
                            GroupColumns::RunFive(y),
                            GroupColumns::Eight(indices),
                        ) = (*a.columns, *b.columns, *c.columns)
                        {
                            let tail = TailOutputs::new(a, b, [x, y], offset, cycles);
                            let optional = c
                                .bytes
                                .as_chunks_mut::<8>()
                                .0
                                .get_mut(offset..offset + cycles)
                                .expect("optional output extent");
                            return run!(RouterOutputs {
                                tail,
                                indices,
                                optional
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        let source: &dyn CycleSource = source;
        self.validate_accumulate::<false, _, _, _>(
            source,
            pass.start,
            pass.rows,
            pass.tile,
            pass.maxima,
            pass.minima,
            pass.rejected,
            GeneralOutputs {
                groups: output,
                offset,
            },
            GeneralRows {
                columns: self.row_columns,
            },
        );
    }

    #[expect(
        clippy::expect_used,
        reason = "the selected complete shape fixes its five-column accumulator tail"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "separate slice references preserve noalias in the accumulator loop"
    )]
    fn validate_accumulate<
        const FIXED: bool,
        O: Outputs,
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
        mut output: O,
        checker: R,
    ) {
        let columns = self.bounds.len();
        let prefix = if FIXED { columns - 5 } else { columns };
        let (max_prefix, max_tail) = maxima[..columns].split_at_mut(prefix);
        let (min_prefix, min_tail) = minima[..columns].split_at_mut(prefix);
        let cycles = tile.len() / columns;
        let plan = output.plan();
        output.check(&plan, columns, cycles);
        checker.check(columns, rows, self.row_cache);
        for (offset, digits) in (0..cycles).zip(tile.chunks_exact(columns)) {
            let row = source.bytecode_index(start + offset);
            if row >= rows {
                *rejected = true;
            } else {
                *rejected |= checker.compare(row, digits, self.row_cache);
            }
            let (values, tail) = digits.split_at(prefix);
            if FIXED {
                for ((maximum, minimum), values) in max_prefix
                    .as_chunks_mut::<16>()
                    .0
                    .iter_mut()
                    .zip(min_prefix.as_chunks_mut::<16>().0)
                    .zip(values.as_chunks::<16>().0)
                {
                    for ((maximum, minimum), &value) in maximum.iter_mut().zip(minimum).zip(values)
                    {
                        *maximum = (*maximum).max(value);
                        *minimum = (*minimum).min(value);
                    }
                }
                let max_tail: &mut [u16; 5] =
                    (&mut *max_tail).try_into().expect("fixed maximum tail");
                let min_tail: &mut [u16; 5] =
                    (&mut *min_tail).try_into().expect("fixed minimum tail");
                let tail: &[u16; 5] = tail.try_into().expect("fixed digit tail");
                for ((maximum, minimum), &value) in max_tail.iter_mut().zip(min_tail).zip(tail) {
                    *maximum = (*maximum).max(value);
                    *minimum = (*minimum).min(value);
                }
            } else {
                for ((maximum, minimum), &value) in
                    max_prefix.iter_mut().zip(min_prefix.iter_mut()).zip(values)
                {
                    *maximum = (*maximum).max(value);
                    *minimum = (*minimum).min(value);
                }
            }
            output.write(digits, offset, &plan);
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
