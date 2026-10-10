//! Complete source-selector folds; routing support is deliberately absent from this pass.

use super::shape::{table_len, BitEntry, RouterError, RouterShape, WordSlot};
mod readout;
use crate::packed::buckets::{BucketPlacement, ByteBuckets, DigitHistogram, NibbleBuckets};
use crate::packed::pool::ScratchPool;
use crate::packed::scatter::ScatterPlan;
use crate::par::CycleChunks;
use crate::round::eq::eq_table;
use crate::source::{CycleSource, ValidatedTrace};
use jolt_field::F128;
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;
use readout::{BankStorage, ReadBit, ReadoutShape};
use std::cmp::Reverse;

#[cfg(feature = "test-utils")]
mod calibration;
#[cfg(feature = "test-utils")]
pub use calibration::FoldCalibration;

const ZERO: F128 = F128::from_raw(0);

#[derive(Debug, Clone)]
struct ShapeLayout {
    words: Vec<(usize, WordSlot)>,
    row_words: Vec<(usize, usize)>,
    digits: Vec<(usize, usize)>,
    flags: Vec<usize>,
    selectors: usize,
    byte_values: Vec<usize>,
    byte_flags: Vec<bool>,
    bases: Vec<usize>,
    row_bases: Vec<usize>,
    metadata: usize,
    totals: Option<usize>,
    meta_len: usize,
}

/// Bucket sizing and byte-bucket selection, independent of routing support.
/// `new` checks one selector set per shape. Each set may be empty or contain
/// every selector. Word, digit and flag ranges have one owner here.
#[derive(Debug, Clone)]
pub struct FoldLayout {
    shapes: Vec<ShapeLayout>,
    entries: usize,
    row_entries: usize,
    readout_shapes: Vec<ReadoutShape>,
}

fn storage_add(total: &mut usize, added: usize) -> Result<(), RouterError> {
    *total = total
        .checked_add(added)
        .filter(|&n| n <= isize::MAX as usize / 16)
        .ok_or(RouterError::Dimension {
            variables: usize::BITS as usize,
        })?;
    Ok(())
}

const fn word_sets(selectors: usize, words: usize, bytes: usize) -> usize {
    words
        * (bytes * BucketPlacement::Byte.word_entries()
            + (selectors - bytes) * BucketPlacement::Nibble.word_entries())
}

impl FoldLayout {
    /// Performance defaults to eight selector values, saving bucket XORs with limited extra scratch.
    pub const DEFAULT_BYTE_BUCKET_LIMIT: usize = 8;

    /// Choose up to `limit` selector values from `selector_counts`, ordered by
    /// descending count. Ties put the lower value first, making the layout a
    /// deterministic function of the trace. A larger limit selects every value.
    pub fn byte_bucket_values(counts: &[usize], limit: usize) -> Vec<usize> {
        let mut values: Vec<_> = (0..counts.len()).collect();
        values.sort_unstable_by_key(|&value| (Reverse(counts[value]), value));
        values.truncate(limit);
        values
    }

    /// Construct ranges for validated shapes and chosen byte-bucket selector
    /// values. Checks source indices and selector-set lengths and membership.
    pub fn new<S: CycleSource>(
        source: &ValidatedTrace<S>,
        shapes: &[RouterShape],
        byte_values: &[Vec<usize>],
    ) -> Result<Self, RouterError> {
        if byte_values.len() != shapes.len() {
            return Err(RouterError::TableLength {
                table: "layout shapes",
                expected: shapes.len(),
                actual: byte_values.len(),
            });
        }
        let source = source.source();
        let mut entries = 0;
        let mut row_entries = 0;
        let mut layouts = Vec::with_capacity(shapes.len());
        for (index, (shape, values)) in shapes.iter().zip(byte_values).enumerate() {
            shape.check_source(source.as_ref())?;
            let mut sorted = values.clone();
            sorted.sort_unstable();
            for (i, &selector) in sorted.iter().enumerate() {
                if selector >= shape.selectors() || (i > 0 && selector == sorted[i - 1]) {
                    return Err(RouterError::Layout {
                        shape: index,
                        selector,
                        bound: shape.selectors(),
                    });
                }
            }
            let by_row = shape.factors().iter().all(|f| source.by_row(f.column));
            let mut words = Vec::new();
            let mut row_words = Vec::new();
            let mut digits = Vec::new();
            let mut flags = Vec::new();
            for (slot, word) in shape.bank().iter().enumerate() {
                match word {
                    WordSlot::Trace(_) => words.push((slot, word.clone())),
                    WordSlot::Bytecode(word) if by_row => row_words.push((slot, *word)),
                    WordSlot::Bytecode(_) => words.push((slot, word.clone())),
                    WordSlot::Bits(bits) => {
                        for bit in bits {
                            let column = match *bit {
                                BitEntry::Indicator { column, .. }
                                | BitEntry::DigitBit { column, .. } => column,
                                BitEntry::One | BitEntry::Zero => continue,
                            };
                            if source.bits(column) == 0 {
                                if !flags.contains(&column) {
                                    flags.push(column);
                                }
                            } else if !digits.iter().any(|&(c, _)| c == column) {
                                let _ = table_len(source.bits(column))?;
                                digits.push((
                                    column,
                                    (1 << source.bits(column))
                                        .max(NibbleBuckets::ENTRIES_PER_POSITION),
                                ));
                            }
                        }
                    }
                    WordSlot::Zero => {}
                }
            }
            let selectors = shape.selectors();
            let mut byte_flags = vec![false; selectors];
            for &h in &sorted {
                byte_flags[h] = true;
            }
            let mut meta_len = flags.len().div_ceil(NibbleBuckets::BITS_PER_POSITION)
                * NibbleBuckets::ENTRIES_PER_POSITION;
            for &(_, bound) in &digits {
                storage_add(&mut meta_len, bound)?;
            }
            let metadata_entries =
                selectors
                    .checked_mul(meta_len)
                    .ok_or(RouterError::Dimension {
                        variables: usize::BITS as usize,
                    })?;
            let start = entries;
            let row_start = row_entries;
            let mut bases = Vec::with_capacity(selectors);
            let mut row_bases = Vec::with_capacity(selectors);
            for h in 0..selectors {
                let bytes = sorted.binary_search(&h).is_ok();
                bases.push(entries);
                row_bases.push(row_entries);
                storage_add(
                    &mut entries,
                    words
                        .len()
                        .checked_mul(BucketPlacement::from_bytes(bytes).word_entries())
                        .ok_or(RouterError::Dimension {
                            variables: usize::BITS as usize,
                        })?,
                )?;
                storage_add(
                    &mut row_entries,
                    row_words
                        .len()
                        .checked_mul(BucketPlacement::from_bytes(bytes).word_entries())
                        .ok_or(RouterError::Dimension {
                            variables: usize::BITS as usize,
                        })?,
                )?;
            }
            debug_assert_eq!(
                entries - start,
                word_sets(selectors, words.len(), sorted.len())
            );
            debug_assert_eq!(
                row_entries - row_start,
                word_sets(selectors, row_words.len(), sorted.len())
            );
            let metadata = entries;
            storage_add(&mut entries, metadata_entries)?;
            let totals = if words.is_empty() && row_words.is_empty() {
                let start = entries;
                storage_add(&mut entries, selectors)?;
                Some(start)
            } else {
                None
            };
            layouts.push(ShapeLayout {
                words,
                row_words,
                digits,
                flags,
                selectors,
                byte_values: sorted,
                byte_flags,
                bases,
                row_bases,
                metadata,
                totals,
                meta_len,
            });
        }
        if entries
            .checked_add(row_entries)
            .is_none_or(|n| n > isize::MAX as usize / 16)
        {
            return Err(RouterError::Dimension {
                variables: usize::BITS as usize,
            });
        }
        let readout_shapes = shapes
            .iter()
            .zip(&layouts)
            .map(|(shape, layout)| ReadoutShape::new(shape, layout, source.as_ref()))
            .collect();
        Ok(Self {
            readout_shapes,
            shapes: layouts,
            entries: entries.max(row_entries),
            row_entries,
        })
    }

    fn validate<S: CycleSource>(
        &self,
        trace: &ValidatedTrace<S>,
        shapes: &[RouterShape],
        r_cycle: &[F128],
        plan: &ScatterPlan<S>,
    ) -> Result<usize, RouterError> {
        let source = trace.source().as_ref();
        let log_t = source.cycles().ilog2() as usize;
        if r_cycle.len() != log_t {
            return Err(RouterError::PointLength {
                expected: log_t,
                actual: r_cycle.len(),
            });
        }
        for (table, expected, actual) in [
            ("layout shapes", shapes.len(), self.shapes.len()),
            ("scatter cycles", source.cycles(), plan.cycles()),
            ("scatter rows", source.bytecode_rows(), plan.bytecode_rows()),
        ] {
            if expected != actual {
                return Err(RouterError::TableLength {
                    table,
                    expected,
                    actual,
                });
            }
        }
        self.check_identity(trace, shapes)?;
        Ok(log_t)
    }

    fn check_identity<S: CycleSource>(
        &self,
        trace: &ValidatedTrace<S>,
        shapes: &[RouterShape],
    ) -> Result<(), RouterError> {
        let source = trace.source();
        for shape in shapes {
            shape.check_source(source.as_ref())?;
        }
        if self
            .readout_shapes
            .iter()
            .zip(shapes)
            .all(|(geometry, shape)| geometry.matches(shape, source.as_ref()))
        {
            return Ok(());
        }
        let values: Vec<_> = self
            .shapes
            .iter()
            .map(|shape| shape.byte_values.clone())
            .collect();
        let checked = Self::new(trace, shapes, &values)?;
        Err(RouterError::TableLength {
            table: "layout geometry",
            expected: checked.entries,
            actual: self.entries,
        })
    }

    /// Cycle bucket elements per worker; includes metadata and any constant totals.
    pub const fn entries(&self) -> usize {
        self.entries
    }
    /// Row bucket elements per worker, used after the cycle scratch is merged.
    pub const fn row_entries(&self) -> usize {
        self.row_entries
    }
}

impl ShapeLayout {
    fn bytes(&self, h: usize) -> bool {
        self.byte_flags[h]
    }
    fn meta_len(&self) -> usize {
        self.meta_len
    }
    fn digit_base(&self, h: usize, column: usize) -> Option<(usize, usize)> {
        let mut base = self.metadata + h * self.meta_len();
        for &(c, bound) in &self.digits {
            if c == column {
                return Some((base, bound));
            }
            base += bound;
        }
        None
    }
    fn flag_base(&self, h: usize, column: usize) -> Option<(usize, usize)> {
        let index = self.flags.iter().position(|&c| c == column)?;
        Some((
            self.metadata
                + h * self.meta_len()
                + self.digits.iter().map(|&(_, n)| n).sum::<usize>()
                + index / NibbleBuckets::BITS_PER_POSITION * NibbleBuckets::ENTRIES_PER_POSITION,
            index % NibbleBuckets::BITS_PER_POSITION,
        ))
    }
}

/// Outputs of the complete fold. Tables follow each shape's occupied-slot order;
/// histogram order follows the requested column list, with one entry per digit.
#[derive(Debug)]
pub struct FoldOutput {
    pub folds: Vec<Vec<F128>>,
    pub ra_fold: Vec<F128>,
    pub histograms: Vec<Vec<F128>>,
}

#[expect(
    clippy::expect_used,
    reason = "private word slices have the checked layout dimensions"
)]
#[inline]
fn bucket_word(storage: &mut [F128], mut word: u64, weight: F128, bytes: bool) {
    if bytes {
        for position in 0..ByteBuckets::POSITIONS_PER_WORD {
            let base = BucketPlacement::Byte.position_offset(position);
            storage[base + (word & (ByteBuckets::ENTRIES_PER_POSITION - 1) as u64) as usize] +=
                weight;
            word >>= ByteBuckets::BITS_PER_POSITION;
        }
    } else {
        let mut buckets = NibbleBuckets::new(&mut storage[..NibbleBuckets::ELEMENTS_PER_WORD])
            .expect("whole nibble positions");
        for position in buckets.positions_mut() {
            position[(word & (NibbleBuckets::ENTRIES_PER_POSITION - 1) as u64) as usize] += weight;
            word >>= NibbleBuckets::BITS_PER_POSITION;
        }
    }
}

#[derive(Clone, Copy)]
enum HistogramPlan {
    Digit { shape: usize },
    Flag { shape: usize },
    Row,
    Cycle,
}

/// Build every `Fold[s,h] = Σ_j eq(r_cycle,j) Source[s,j] Select[h,j]`,
/// `ra_fold` and the requested digit histograms. The cycle pass emits each
/// equality weight through the caller's scatter plan, without recomputing it.
/// `layout` must be made by `FoldLayout::new` for these shapes and this source;
/// this is checked against its read-out identity. The plan must refer to the
/// same immutable source: required of the caller, not checked, detected by the
/// verifier through the resulting claims. `route` is never read.
pub fn fold_pass<S: CycleSource>(
    trace: &ValidatedTrace<S>,
    shapes: &[RouterShape],
    r_cycle: &[F128],
    plan: &ScatterPlan<S>,
    layout: &FoldLayout,
    histogram_columns: &[usize],
) -> Result<FoldOutput, RouterError> {
    fold_impl(
        trace,
        shapes,
        r_cycle,
        plan,
        layout,
        histogram_columns,
        &mut NoPhases,
    )
}

struct Histograms<'a, S> {
    source: &'a S,
    columns: &'a [usize],
    plans: Vec<HistogramPlan>,
    offsets: Vec<usize>,
    fallbacks: Vec<Vec<(usize, usize)>>,
    direct: Vec<(usize, usize)>,
    rows: Vec<(usize, usize)>,
    cycle_entries: usize,
    row_entries: usize,
}

impl<'a, S: CycleSource> Histograms<'a, S> {
    fn new(
        source: &'a S,
        layout: &FoldLayout,
        histogram_columns: &'a [usize],
    ) -> Result<Self, RouterError> {
        let mut hist_plans = Vec::with_capacity(histogram_columns.len());
        let mut hist_offsets = Vec::with_capacity(histogram_columns.len());
        let mut scratch_len = layout.entries;
        for &column in histogram_columns {
            if column >= source.digit_columns() {
                return Err(RouterError::Column {
                    column,
                    columns: source.digit_columns(),
                });
            }
            let bound = table_len(source.bits(column))?;
            hist_offsets.push(scratch_len);
            storage_add(&mut scratch_len, bound)?;
            let marginal = layout.shapes.iter().enumerate().find_map(|(shape, s)| {
                if s.digits.iter().any(|&(c, _)| c == column) {
                    Some(HistogramPlan::Digit { shape })
                } else if s.flags.contains(&column) {
                    Some(HistogramPlan::Flag { shape })
                } else {
                    None
                }
            });
            hist_plans.push(if source.by_row(column) {
                HistogramPlan::Row
            } else {
                marginal.unwrap_or(HistogramPlan::Cycle)
            });
        }
        let mut fallbacks = vec![Vec::new(); layout.shapes.len()];
        let mut direct_histograms = Vec::new();
        let mut row_histograms = Vec::new();
        let mut row_hist_base = layout.row_entries;
        for ((&column, &hp), &base) in histogram_columns.iter().zip(&hist_plans).zip(&hist_offsets)
        {
            match hp {
                HistogramPlan::Digit { shape } | HistogramPlan::Flag { shape } => {
                    fallbacks[shape].push((column, base));
                }
                HistogramPlan::Cycle => direct_histograms.push((column, base)),
                HistogramPlan::Row => row_histograms.push((column, row_hist_base)),
            }
            row_hist_base += 1 << source.bits(column);
        }
        let mut row_entries = layout.row_entries;
        for &column in histogram_columns {
            storage_add(&mut row_entries, table_len(source.bits(column))?)?;
        }
        Ok(Self {
            source,
            columns: histogram_columns,
            plans: hist_plans,
            offsets: hist_offsets,
            fallbacks,
            direct: direct_histograms,
            rows: row_histograms,
            cycle_entries: scratch_len,
            row_entries,
        })
    }
    #[expect(
        clippy::expect_used,
        reason = "the plan names checked marginal storage"
    )]
    fn read_cycles(&self, layout: &FoldLayout, buckets: &mut [F128]) -> Vec<Vec<F128>> {
        let mut histograms = Vec::with_capacity(self.columns.len());
        for ((&column, &hp), &base) in self.columns.iter().zip(&self.plans).zip(&self.offsets) {
            let bound = 1 << self.source.bits(column);
            match hp {
                HistogramPlan::Row => {}
                HistogramPlan::Digit { shape } => {
                    for h in 0..layout.shapes[shape].selectors {
                        let (offset, _) = layout.shapes[shape]
                            .digit_base(h, column)
                            .expect("selected digit marginal");
                        for digit in 0..bound {
                            let e = buckets[offset + digit];
                            buckets[base + digit] += e;
                        }
                    }
                }
                HistogramPlan::Flag { shape } => {
                    for h in 0..layout.shapes[shape].selectors {
                        let (offset, bit) = layout.shapes[shape]
                            .flag_base(h, column)
                            .expect("selected flag marginal");
                        let sum = NibbleBuckets::position_bit(
                            buckets[offset..offset + NibbleBuckets::ENTRIES_PER_POSITION]
                                .try_into()
                                .expect("flag position"),
                            bit,
                        )
                        .expect("checked flag bit");
                        buckets[base] += sum;
                    }
                }
                HistogramPlan::Cycle => {}
            }
            histograms.push(buckets[base..base + bound].to_vec());
        }
        histograms
    }
    fn read_rows(&self, layout: &FoldLayout, row_buckets: &[F128], histograms: &mut [Vec<F128>]) {
        let mut row_base = layout.row_entries;
        for ((&column, &hp), histogram) in self.columns.iter().zip(&self.plans).zip(histograms) {
            let bound = 1 << self.source.bits(column);
            if matches!(hp, HistogramPlan::Row) {
                histogram.copy_from_slice(&row_buckets[row_base..row_base + bound]);
            }
            row_base += bound;
        }
    }
}

struct CyclePass<'a, S: CycleSource> {
    source: &'a S,
    layout: &'a FoldLayout,
    shapes: &'a [RouterShape],
    plan: &'a ScatterPlan<S>,
    pool: &'a ScratchPool,
    histograms: &'a Histograms<'a, S>,
    low: &'a [F128],
    high: &'a [F128],
    weights: &'a mut [F128],
}

impl<S: CycleSource> CyclePass<'_, S> {
    #[expect(
        clippy::expect_used,
        reason = "validated chunk geometry and exclusive scratch loans"
    )]
    fn run(self) {
        let Self {
            source,
            layout,
            shapes,
            plan,
            pool,
            histograms,
            low,
            high,
            weights,
        } = self;
        plan.emission_chunks(weights)
            .expect("freshly sized weight buffer")
            .for_each(|(interval, slots, weights)| {
                let Some(last) = weights.len().checked_sub(1) else {
                    return;
                };
                let mut scratch = pool.take().expect("sequential chunk body");
                for (block, slots) in slots.chunks_exact(low.len()).enumerate() {
                    let start = interval.start + block * low.len();
                    let hi = high[start / low.len()];
                    for (offset, (&lo, &slot)) in low.iter().zip(slots).enumerate() {
                        let cycle = start + offset;
                        let e = hi * lo;
                        // Scatter slots are relative to the whole chunk, not its equality block.
                        weights[usize::from(slot).min(last)] = e;
                        for (index, (shape, sl)) in shapes.iter().zip(&layout.shapes).enumerate() {
                            if let Some(h) = shape.selector(source, cycle, false) {
                                let bytes = sl.bytes(h);
                                let size = BucketPlacement::from_bytes(bytes).word_entries();
                                let base = sl.bases[h];
                                for ((_, word), storage) in sl.words.iter().zip(
                                    scratch[base..base + sl.words.len() * size]
                                        .chunks_exact_mut(size),
                                ) {
                                    let word = match *word {
                                        WordSlot::Trace(index) => source.trace_word(index, cycle),
                                        WordSlot::Bytecode(index) => source
                                            .bytecode_word(index, source.bytecode_index(cycle)),
                                        WordSlot::Bits(_) | WordSlot::Zero => 0,
                                    };
                                    bucket_word(storage, word, e, bytes);
                                }
                                let mut base = sl.metadata + h * sl.meta_len();
                                for &(column, bound) in &sl.digits {
                                    if let Some(digit) = source.digit(column, cycle) {
                                        let storage = &mut scratch[base..base + bound];
                                        if bound == NibbleBuckets::ENTRIES_PER_POSITION {
                                            let mut buckets = NibbleBuckets::new(storage)
                                                .expect("one padded digit position");
                                            buckets.positions_mut()[0][digit
                                                & (NibbleBuckets::ENTRIES_PER_POSITION - 1)] += e;
                                        } else if let Some(last) = storage.len().checked_sub(1) {
                                            storage[digit.min(last)] += e;
                                        }
                                    }
                                    base += bound;
                                }
                                for flags in sl.flags.chunks(NibbleBuckets::BITS_PER_POSITION) {
                                    let mut value = 0;
                                    for (bit, &column) in flags.iter().enumerate() {
                                        value |= usize::from(source.digit(column, cycle).is_some())
                                            << bit;
                                    }
                                    let mut buckets = NibbleBuckets::new(
                                        &mut scratch
                                            [base..base + NibbleBuckets::ENTRIES_PER_POSITION],
                                    )
                                    .expect("one packed flag position");
                                    buckets.positions_mut()[0]
                                        [value & (NibbleBuckets::ENTRIES_PER_POSITION - 1)] += e;
                                    base += NibbleBuckets::ENTRIES_PER_POSITION;
                                }
                                if let Some(base) = sl.totals {
                                    scratch[base + h] += e;
                                }
                            } else {
                                for &(column, base) in &histograms.fallbacks[index] {
                                    if let Some(digit) = source.digit(column, cycle) {
                                        scratch[base + digit] += e;
                                    }
                                }
                            }
                        }
                        for &(column, base) in &histograms.direct {
                            if let Some(digit) = source.digit(column, cycle) {
                                scratch[base + digit] += e;
                            }
                        }
                    }
                }
            });
    }
}

struct RowPass<'a, S> {
    source: &'a S,
    layout: &'a FoldLayout,
    shapes: &'a [RouterShape],
    pool: &'a ScratchPool,
    histograms: &'a [(usize, usize)],
    weights: &'a [F128],
}

impl<S: CycleSource> RowPass<'_, S> {
    #[expect(
        clippy::expect_used,
        reason = "row chunks hold exclusive scratch loans"
    )]
    fn run(self, chunk_len: usize) {
        let Self {
            source,
            layout,
            shapes,
            pool: row_pool,
            histograms,
            weights,
        } = self;
        let row_shapes: Vec<_> = shapes
            .iter()
            .zip(&layout.shapes)
            .filter(|(_, sl)| !sl.row_words.is_empty())
            .collect();
        weights
            .par_chunks(chunk_len)
            .enumerate()
            .for_each(|(chunk, rows)| {
                let mut scratch = row_pool.take().expect("sequential row body");
                for (offset, &e) in rows.iter().enumerate() {
                    if e == ZERO {
                        continue;
                    }
                    let row = chunk * chunk_len + offset;
                    for &(shape, sl) in &row_shapes {
                        if let Some(h) = shape.selector(source, row, true) {
                            let bytes = sl.bytes(h);
                            let size = BucketPlacement::from_bytes(bytes).word_entries();
                            let base = sl.row_bases[h];
                            for (&(_, word), storage) in sl.row_words.iter().zip(
                                scratch[base..base + sl.row_words.len() * size]
                                    .chunks_exact_mut(size),
                            ) {
                                bucket_word(storage, source.bytecode_word(word, row), e, bytes);
                            }
                        }
                    }
                    for &(column, base) in histograms {
                        if let Some(digit) = source.row_digit(column, row) {
                            scratch[base + digit] += e;
                        }
                    }
                }
            });
    }
}

trait PhaseHook {
    fn finish_phase(&mut self, phase: usize);
}

struct NoPhases;

impl PhaseHook for NoPhases {
    #[inline]
    fn finish_phase(&mut self, _: usize) {}
}

#[expect(
    clippy::expect_used,
    reason = "validated geometry, private scratch ownership and freshly sized scatter buffers cannot fail"
)]
fn fold_impl<S: CycleSource, H: PhaseHook>(
    trace: &ValidatedTrace<S>,
    shapes: &[RouterShape],
    r_cycle: &[F128],
    plan: &ScatterPlan<S>,
    layout: &FoldLayout,
    histogram_columns: &[usize],
    phases: &mut H,
) -> Result<FoldOutput, RouterError> {
    let source = trace.source().as_ref();
    let log_t = layout.validate(trace, shapes, r_cycle, plan)?;
    let histograms = Histograms::new(source, layout, histogram_columns)?;
    let geometry = CycleChunks::new(log_t, 0).expect("validated cycles");
    let (low, high) = geometry.split_point(r_cycle).expect("checked cycle point");
    let low = eq_table(low, None);
    let high = eq_table(high, None);
    let pool = ScratchPool::new(histograms.cycle_entries).map_err(|_| RouterError::Dimension {
        variables: usize::BITS as usize,
    })?;
    let mut weights = unsafe_allocate_zero_vec(source.cycles());
    phases.finish_phase(3);
    CyclePass {
        source,
        layout,
        shapes,
        plan,
        pool: &pool,
        histograms: &histograms,
        low: &low,
        high: &high,
        weights: &mut weights,
    }
    .run();
    phases.finish_phase(0);
    drop(low);
    drop(high);
    let mut ra_fold = unsafe_allocate_zero_vec(source.bytecode_rows());
    plan.apply_buffer(&weights, &mut ra_fold)
        .expect("freshly sized scatter buffers");
    drop(weights);
    phases.finish_phase(1);
    let mut buckets = pool.merge().expect("all chunk loans returned");
    let mut folds: Vec<_> = shapes
        .iter()
        .map(|shape| unsafe_allocate_zero_vec(shape.fold_len()))
        .collect();
    readout(layout, FoldStorage::Cycles(&buckets), &mut folds);
    let mut histogram_outputs = histograms.read_cycles(layout, &mut buckets);
    drop(buckets);
    let row_pool =
        ScratchPool::new(histograms.row_entries).map_err(|_| RouterError::Dimension {
            variables: usize::BITS as usize,
        })?;
    phases.finish_phase(3);
    RowPass {
        source,
        layout,
        shapes,
        pool: &row_pool,
        histograms: &histograms.rows,
        weights: &ra_fold,
    }
    .run(geometry.chunk_len());
    phases.finish_phase(2);
    let row_buckets = row_pool.merge().expect("all row loans returned");
    readout(layout, FoldStorage::Rows(&row_buckets), &mut folds);
    histograms.read_rows(layout, &row_buckets, &mut histogram_outputs);
    phases.finish_phase(3);
    Ok(FoldOutput {
        folds,
        ra_fold,
        histograms: histogram_outputs,
    })
}

enum FoldStorage<'a> {
    Cycles(&'a [F128]),
    Rows(&'a [F128]),
}

#[expect(
    clippy::expect_used,
    reason = "checked metadata has checked power-of-two domains and valid bit indices"
)]
fn readout(layout: &FoldLayout, storage: FoldStorage<'_>, folds: &mut [Vec<F128>]) {
    let (buckets, rows) = match storage {
        FoldStorage::Cycles(buckets) => (buckets, false),
        FoldStorage::Rows(buckets) => (buckets, true),
    };
    for ((sl, geometry), fold) in layout.shapes.iter().zip(&layout.readout_shapes).zip(folds) {
        if rows && sl.row_words.is_empty() {
            continue;
        }
        for h in 0..sl.selectors {
            let bytes = sl.bytes(h);
            let size = BucketPlacement::from_bytes(bytes).word_entries();
            let total = if rows {
                word_total(buckets, sl.row_bases[h], bytes)
            } else if let Some(base) = sl.totals {
                buckets[base + h]
            } else if !sl.words.is_empty() {
                word_total(buckets, sl.bases[h], bytes)
            } else {
                ZERO
            };
            let metadata = sl.metadata + h * sl.meta_len;
            for (word, bank) in geometry.bank_storage.iter().enumerate() {
                let destination = geometry.destinations[h * geometry.bank_storage.len() + word];
                match bank {
                    BankStorage::Cycle(index) if !rows => read_word(
                        buckets,
                        sl.bases[h] + index * size,
                        bytes,
                        &mut fold[destination..destination + 64],
                    ),
                    BankStorage::Row(index) if rows => read_word(
                        buckets,
                        sl.row_bases[h] + index * size,
                        bytes,
                        &mut fold[destination..destination + 64],
                    ),
                    BankStorage::Bits(entries) => {
                        for (bit, entry) in entries.iter().enumerate() {
                            if rows
                                != (matches!(entry, ReadBit::One)
                                    && sl.words.is_empty()
                                    && sl.totals.is_none())
                            {
                                continue;
                            }
                            fold[destination + bit] = match *entry {
                                ReadBit::One => total,
                                ReadBit::Value { offset, value } => {
                                    buckets[metadata + offset + value]
                                }
                                ReadBit::Bit { offset, bound, bit } => DigitHistogram::sum_bit(
                                    &buckets[metadata + offset..metadata + offset + bound],
                                    bound.ilog2() as usize,
                                    bit,
                                )
                                .expect("checked bit domain"),
                                ReadBit::Zero => ZERO,
                            };
                        }
                    }
                    BankStorage::Cycle(_) | BankStorage::Row(_) | BankStorage::Zero => {}
                }
            }
        }
    }
}

#[expect(
    clippy::expect_used,
    reason = "checked word storage contains each complete position"
)]
fn read_word(buckets: &[F128], base: usize, bytes: bool, output: &mut [F128]) {
    if bytes {
        for (position, output) in output
            .as_chunks_mut::<{ ByteBuckets::BITS_PER_POSITION }>()
            .0
            .iter_mut()
            .enumerate()
        {
            let offset = base + BucketPlacement::Byte.position_offset(position);
            *output = ByteBuckets::position_bits(
                buckets[offset..offset + ByteBuckets::ENTRIES_PER_POSITION]
                    .try_into()
                    .expect("byte position"),
            );
        }
    } else {
        for (position, output) in buckets[base..base + NibbleBuckets::ELEMENTS_PER_WORD]
            .as_chunks::<{ NibbleBuckets::ENTRIES_PER_POSITION }>()
            .0
            .iter()
            .zip(
                output
                    .as_chunks_mut::<{ NibbleBuckets::BITS_PER_POSITION }>()
                    .0,
            )
        {
            *output = NibbleBuckets::position_bits(position);
        }
    }
}

#[expect(
    clippy::expect_used,
    reason = "checked word storage contains its complete first position"
)]
fn word_total(buckets: &[F128], base: usize, bytes: bool) -> F128 {
    if bytes {
        ByteBuckets::position_total(
            buckets[base..base + ByteBuckets::ENTRIES_PER_POSITION]
                .try_into()
                .expect("byte position"),
        )
    } else {
        NibbleBuckets::position_total(
            buckets[base..base + NibbleBuckets::ENTRIES_PER_POSITION]
                .try_into()
                .expect("nibble position"),
        )
    }
}
