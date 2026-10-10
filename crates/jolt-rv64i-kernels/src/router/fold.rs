//! Complete source-selector folds; routing support is deliberately absent from this pass.

use super::shape::{table_len, BitEntry, RouterError, RouterShape, WordSlot};
mod compiled;
use crate::packed::buckets::NibbleBuckets;
use crate::packed::pool::ScratchPool;
use crate::packed::scatter::ScatterPlan;
use crate::par::CycleChunks;
use crate::round::eq::eq_table;
use crate::source::{CycleSource, ValidatedTrace};
use compiled::{BankStorage, ChunkBuckets, CompiledShape, ReadBit};
use jolt_field::F128;
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;
use std::time::{Duration, Instant};

#[cfg(feature = "test-utils")]
mod calibration;
#[cfg(feature = "test-utils")]
pub use calibration::FoldCalibration;

const ZERO: F128 = F128::from_raw(0);
const NIBBLE_WORD: usize = 256;
const BYTE_WORD: usize = 2048;

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
    compiled: Vec<CompiledShape>,
}

const fn word_entries(bytes: bool) -> usize {
    if bytes {
        BYTE_WORD
    } else {
        NIBBLE_WORD
    }
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
    words * (bytes * word_entries(true) + (selectors - bytes) * word_entries(false))
}

impl FoldLayout {
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
                                digits.push((column, (1 << source.bits(column)).max(16)));
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
            let mut meta_len = flags.len().div_ceil(4) * 16;
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
                        .checked_mul(word_entries(bytes))
                        .ok_or(RouterError::Dimension {
                            variables: usize::BITS as usize,
                        })?,
                )?;
                storage_add(
                    &mut row_entries,
                    row_words.len().checked_mul(word_entries(bytes)).ok_or(
                        RouterError::Dimension {
                            variables: usize::BITS as usize,
                        },
                    )?,
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
        let compiled = shapes
            .iter()
            .zip(&layouts)
            .map(|(shape, layout)| CompiledShape::new(shape, layout, source.as_ref()))
            .collect();
        Ok(Self {
            compiled,
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
            .compiled
            .iter()
            .zip(shapes)
            .all(|(compiled, shape)| compiled.matches(shape, source.as_ref()))
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

    /// Benchmark-only observation of the identical fold implementation. Durations
    /// cover fused equality/buckets/emission, scatter application, visited-row
    /// buckets, and preparation/merges/read-out, respectively. Lazy scratch
    /// zero-fill is included in its bucket phase; the first phase cannot isolate
    /// its fused multiplication and XORs without instrumenting every cycle.
    #[cfg(feature = "test-utils")]
    pub fn measure<S: CycleSource>(
        &self,
        source: &ValidatedTrace<S>,
        shapes: &[RouterShape],
        point: &[F128],
        plan: &ScatterPlan<S>,
        histogram_columns: &[usize],
    ) -> Result<(FoldOutput, [Duration; 4]), RouterError> {
        fold_impl::<S, true, true>(source, shapes, point, plan, self, histogram_columns)
    }

    /// Benchmark observation of the cycle-major compiled fold pass. The arguments
    /// and checked layout contract are the same as `measure`.
    #[cfg(feature = "test-utils")]
    pub fn measure_cycle_major<S: CycleSource>(
        &self,
        source: &ValidatedTrace<S>,
        shapes: &[RouterShape],
        point: &[F128],
        plan: &ScatterPlan<S>,
        histogram_columns: &[usize],
    ) -> Result<(FoldOutput, [Duration; 4]), RouterError> {
        fold_impl::<S, true, false>(source, shapes, point, plan, self, histogram_columns)
    }

    /// Benchmark observation of the shape-major compiled fold pass. The arguments
    /// and checked layout contract are the same as `measure`.
    #[cfg(feature = "test-utils")]
    pub fn measure_shape_major<S: CycleSource>(
        &self,
        source: &ValidatedTrace<S>,
        shapes: &[RouterShape],
        point: &[F128],
        plan: &ScatterPlan<S>,
        histogram_columns: &[usize],
    ) -> Result<(FoldOutput, [Duration; 4]), RouterError> {
        fold_impl::<S, true, true>(source, shapes, point, plan, self, histogram_columns)
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
                + index / 4 * 16,
            index % 4,
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
/// this is checked against its compiled identity. The plan must refer to the
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
    fold_impl::<S, false, true>(trace, shapes, r_cycle, plan, layout, histogram_columns)
        .map(|(output, _)| output)
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
                        let sum = (0..16)
                            .filter(|value| value & (1 << bit) != 0)
                            .fold(ZERO, |sum, value| sum + buckets[offset + value]);
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
    fn run<const SHAPE_MAJOR: bool>(self, chunk_len: usize) {
        let Self {
            source,
            layout,
            plan,
            pool,
            histograms,
            low,
            high,
            weights,
        } = self;
        let weight_pool = if SHAPE_MAJOR {
            Some(ScratchPool::new(chunk_len).expect("validated chunk scratch"))
        } else {
            None
        };
        plan.emission_chunks(weights)
            .expect("freshly sized weight buffer")
            .for_each(|(interval, slots, weights)| {
                let Some(last) = weights.len().checked_sub(1) else {
                    return;
                };
                let mut scratch = pool.take().expect("sequential chunk body");
                let mut buckets = ChunkBuckets::new(&mut scratch);
                let mut local_weights = weight_pool
                    .as_ref()
                    .map(|pool| pool.take().expect("sequential chunk weights"));
                for (block, slots) in slots.chunks_exact(low.len()).enumerate() {
                    let start = interval.start + block * low.len();
                    let hi = high[start / low.len()];
                    for (offset, (&lo, &slot)) in low.iter().zip(slots).enumerate() {
                        let cycle = start + offset;
                        let e = hi * lo;
                        // Scatter slots are relative to the whole chunk, not its equality block.
                        weights[usize::from(slot).min(last)] = e;
                        if SHAPE_MAJOR {
                            let local = local_weights.as_mut().expect("shape-major weight loan");
                            let end = local.len() - 1;
                            local[(cycle - interval.start).min(end)] = e;
                        } else {
                            for (index, (sl, compiled)) in
                                layout.shapes.iter().zip(&layout.compiled).enumerate()
                            {
                                if !compiled.cycle(source, cycle, e, &mut buckets, sl.totals) {
                                    for &(column, base) in &histograms.fallbacks[index] {
                                        if let Some(digit) = source.digit(column, cycle) {
                                            buckets.xor(base + digit, e);
                                        }
                                    }
                                }
                            }
                        }
                        for &(column, base) in &histograms.direct {
                            if let Some(digit) = source.digit(column, cycle) {
                                buckets.xor(base + digit, e);
                            }
                        }
                    }
                }
                if SHAPE_MAJOR {
                    let local = local_weights.as_ref().expect("shape-major weight loan");
                    for (index, (sl, compiled)) in
                        layout.shapes.iter().zip(&layout.compiled).enumerate()
                    {
                        compiled.cycles(
                            source,
                            interval.start,
                            &local[..interval.len()],
                            &mut buckets,
                            sl.totals,
                            &histograms.fallbacks[index],
                        );
                    }
                }
            });
        drop(weight_pool);
    }
}

struct RowPass<'a, S> {
    source: &'a S,
    layout: &'a FoldLayout,
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
            pool: row_pool,
            histograms,
            weights,
        } = self;
        weights
            .par_chunks(chunk_len)
            .enumerate()
            .for_each(|(chunk, rows)| {
                let start = chunk * chunk_len;
                let mut scratch = row_pool.take().expect("sequential row body");
                let mut buckets = ChunkBuckets::new(&mut scratch);
                for (sl, compiled) in layout.shapes.iter().zip(&layout.compiled) {
                    if !sl.row_words.is_empty() {
                        compiled.rows(source, start, rows, &sl.row_words, &mut buckets);
                    }
                }
                for (offset, &e) in rows.iter().enumerate() {
                    if e == ZERO {
                        continue;
                    }
                    for &(column, base) in histograms {
                        if let Some(digit) = source.row_digit(column, start + offset) {
                            buckets.xor(base + digit, e);
                        }
                    }
                }
            });
    }
}

#[expect(
    clippy::expect_used,
    reason = "validated geometry, private scratch ownership and freshly sized scatter buffers cannot fail"
)]
fn fold_impl<S: CycleSource, const MEASURE: bool, const SHAPE_MAJOR: bool>(
    trace: &ValidatedTrace<S>,
    shapes: &[RouterShape],
    r_cycle: &[F128],
    plan: &ScatterPlan<S>,
    layout: &FoldLayout,
    histogram_columns: &[usize],
) -> Result<(FoldOutput, [Duration; 4]), RouterError> {
    let mut times = [Duration::ZERO; 4];
    let mut start = MEASURE.then(Instant::now);
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
    if let Some(clock) = start {
        times[3] += clock.elapsed();
        start = Some(Instant::now());
    }
    CyclePass {
        source,
        layout,
        plan,
        pool: &pool,
        histograms: &histograms,
        low: &low,
        high: &high,
        weights: &mut weights,
    }
    .run::<SHAPE_MAJOR>(geometry.chunk_len());
    if let Some(clock) = start {
        times[0] = clock.elapsed();
        start = Some(Instant::now());
    }
    drop(low);
    drop(high);
    let mut ra_fold = unsafe_allocate_zero_vec(source.bytecode_rows());
    plan.apply_buffer(&weights, &mut ra_fold)
        .expect("freshly sized scatter buffers");
    drop(weights);
    if let Some(clock) = start {
        times[1] = clock.elapsed();
        start = Some(Instant::now());
    }
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
    if let Some(clock) = start {
        times[3] += clock.elapsed();
        start = Some(Instant::now());
    }
    RowPass {
        source,
        layout,
        pool: &row_pool,
        histograms: &histograms.rows,
        weights: &ra_fold,
    }
    .run(geometry.chunk_len());
    if let Some(clock) = start {
        times[2] = clock.elapsed();
        start = Some(Instant::now());
    }
    let row_buckets = row_pool.merge().expect("all row loans returned");
    readout(layout, FoldStorage::Rows(&row_buckets), &mut folds);
    histograms.read_rows(layout, &row_buckets, &mut histogram_outputs);
    if let Some(clock) = start {
        times[3] += clock.elapsed();
    }
    Ok((
        FoldOutput {
            folds,
            ra_fold,
            histograms: histogram_outputs,
        },
        times,
    ))
}

enum FoldStorage<'a> {
    Cycles(&'a [F128]),
    Rows(&'a [F128]),
}

fn readout(layout: &FoldLayout, storage: FoldStorage<'_>, folds: &mut [Vec<F128>]) {
    let (buckets, rows) = match storage {
        FoldStorage::Cycles(buckets) => (buckets, false),
        FoldStorage::Rows(buckets) => (buckets, true),
    };
    for ((sl, compiled), fold) in layout.shapes.iter().zip(&layout.compiled).zip(folds) {
        if rows && sl.row_words.is_empty() {
            continue;
        }
        for h in 0..sl.selectors {
            let bytes = sl.bytes(h);
            let size = word_entries(bytes);
            let bits = if bytes { 8 } else { 4 };
            let width = 1 << bits;
            let total = if rows {
                buckets[sl.row_bases[h]..sl.row_bases[h] + width]
                    .iter()
                    .copied()
                    .sum()
            } else if let Some(base) = sl.totals {
                buckets[base + h]
            } else if !sl.words.is_empty() {
                buckets[sl.bases[h]..sl.bases[h] + width]
                    .iter()
                    .copied()
                    .sum()
            } else {
                ZERO
            };
            let metadata = sl.metadata + h * sl.meta_len;
            for (word, bank) in compiled.bank_storage.iter().enumerate() {
                let destination = compiled.destinations[h * compiled.bank_storage.len() + word];
                match bank {
                    BankStorage::Cycle(index) if !rows => read_word(
                        buckets,
                        sl.bases[h] + index * size,
                        bits,
                        width,
                        &mut fold[destination..destination + 64],
                    ),
                    BankStorage::Row(index) if rows => read_word(
                        buckets,
                        sl.row_bases[h] + index * size,
                        bits,
                        width,
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
                                ReadBit::Bit { offset, bound, bit } => bit_sum(
                                    &buckets[metadata + offset..metadata + offset + bound],
                                    bit,
                                ),
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

fn read_word(buckets: &[F128], base: usize, bits: usize, width: usize, output: &mut [F128]) {
    for (position, output) in buckets[base..base + 64 / bits * width]
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(bits))
    {
        for (bit, output) in output.iter_mut().enumerate() {
            *output = bit_sum(position, bit);
        }
    }
}

fn bit_sum(position: &[F128], bit: usize) -> F128 {
    let half = 1 << bit;
    position
        .chunks_exact(half * 2)
        .flat_map(|period| &period[half..])
        .copied()
        .sum()
}
