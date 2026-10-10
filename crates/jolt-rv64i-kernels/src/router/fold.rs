//! Complete source-selector folds; routing support is deliberately absent from this pass.

use super::shape::{table_len, BitEntry, RouterError, RouterShape, SlotVariable, WordSlot};
use crate::packed::buckets::{ByteBuckets, NibbleBuckets};
use crate::packed::pool::ScratchPool;
use crate::packed::scatter::ScatterPlan;
use crate::par::CycleChunks;
use crate::round::eq::eq_table;
use crate::source::{CycleSource, ValidatedTrace};
use jolt_field::F128;
use rayon::prelude::*;
use std::time::{Duration, Instant};

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
    calibration_bytes: usize,
    offsets: [usize; 5],
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
    /// Selector domains of the five experiment shapes in protocol order.
    pub const CALIBRATION_SELECTORS: [usize; 5] = [64, 512, 128, 512, 1];
    /// Cycle-bucketed words of the five experiment shapes.
    pub const CALIBRATION_WORD_SETS: [usize; 5] = [5, 1, 2, 3, 2];
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
        Ok(Self {
            shapes: layouts,
            entries: entries.max(row_entries),
            row_entries,
            calibration_bytes: 0,
            offsets: [0; 5],
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
        fold_impl::<S, true>(source, shapes, point, plan, self, histogram_columns)
    }

    /// Calibration geometry of the experiment, with the first `byte_selectors`
    /// Variant values bucketed by byte. Values above 64 are clamped to 64.
    /// This sizes the machinery/probe layouts without borrowing a trace.
    pub const fn calibration(byte_selectors: usize) -> Self {
        let bytes = if byte_selectors > 64 {
            64
        } else {
            byte_selectors
        };
        let metadata = word_sets(64, 5, bytes);
        let shift = metadata + 64 * 8 * 16;
        let memory = shift + word_sets(512, 1, 0);
        let compare = memory + word_sets(128, 2, 0);
        let branch = compare + word_sets(512, 3, 0);
        Self {
            shapes: Vec::new(),
            entries: branch + word_sets(1, 2, 0),
            row_entries: word_sets(64, 4, bytes),
            calibration_bytes: bytes,
            offsets: [metadata, shift, memory, compare, branch],
        }
    }
    /// Cycle bucket elements per worker; includes metadata and any constant totals.
    pub const fn entries(&self) -> usize {
        self.entries
    }
    /// Row bucket elements per worker, used after the cycle scratch is merged.
    pub const fn row_entries(&self) -> usize {
        self.row_entries
    }
    /// Calibration metadata and subsequent shape starts, in field elements.
    pub const fn offsets(&self) -> [usize; 5] {
        self.offsets
    }
    /// Calibration Variant word-set start. Caller supplies a value below 64.
    pub const fn variant_base(&self, selector: usize) -> usize {
        let bytes = self.calibration_bytes;
        if selector < bytes {
            selector * 5 * BYTE_WORD
        } else {
            bytes * 5 * BYTE_WORD + (selector - bytes) * 5 * NIBBLE_WORD
        }
    }
    /// Calibration shape start. Caller supplies shape 0..5 and a valid selector.
    pub const fn shape_base(&self, shape: usize, selector: usize) -> usize {
        let words = [5, 1, 2, 3, 2];
        if shape == 0 {
            self.variant_base(selector)
        } else {
            self.offsets[shape] + selector * words[shape] * NIBBLE_WORD
        }
    }
    /// Calibration Variant digit/flag start. Caller supplies a value below 64.
    pub const fn variant_metadata_base(&self, selector: usize) -> usize {
        self.offsets[0] + selector * 8 * 16
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

#[expect(
    clippy::expect_used,
    reason = "private bucket slices are whole words, sized by FoldLayout"
)]
#[inline]
fn bucket_word(storage: &mut [F128], mut word: u64, weight: F128, bytes: bool) {
    if bytes {
        let mut buckets = ByteBuckets::new(storage).expect("whole byte positions");
        for position in buckets.positions_mut() {
            position[(word & 255) as usize] += weight;
            word >>= 8;
        }
    } else {
        let mut buckets = NibbleBuckets::new(storage).expect("whole nibble positions");
        for position in buckets.positions_mut() {
            position[(word & 15) as usize] += weight;
            word >>= 4;
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
/// this is checked by rebuilding its small geometry. The plan must refer to the
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
    fold_impl::<S, false>(trace, shapes, r_cycle, plan, layout, histogram_columns)
        .map(|(output, _)| output)
}

#[expect(
    clippy::expect_used,
    reason = "validated geometry, private scratch ownership and freshly sized scatter buffers cannot fail"
)]
fn fold_impl<S: CycleSource, const MEASURE: bool>(
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
    let log_t = source.cycles().ilog2() as usize;
    if r_cycle.len() != log_t {
        return Err(RouterError::PointLength {
            expected: log_t,
            actual: r_cycle.len(),
        });
    }
    for (table, expected, actual) in [
        ("layout shapes", shapes.len(), layout.shapes.len()),
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
    let values: Vec<_> = layout
        .shapes
        .iter()
        .map(|s| s.byte_values.clone())
        .collect();
    let checked = FoldLayout::new(trace, shapes, &values)?;
    if checked.entries != layout.entries
        || checked.row_entries != layout.row_entries
        || checked.shapes.iter().zip(&layout.shapes).any(|(a, b)| {
            a.words != b.words
                || a.row_words != b.row_words
                || a.digits != b.digits
                || a.flags != b.flags
                || a.selectors != b.selectors
        })
    {
        return Err(RouterError::TableLength {
            table: "layout geometry",
            expected: checked.entries,
            actual: layout.entries,
        });
    }
    drop(checked);
    drop(values);
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
    let mut fallbacks = vec![Vec::new(); shapes.len()];
    let mut direct_histograms = Vec::new();
    let mut row_histograms = Vec::new();
    let mut row_hist_base = layout.row_entries;
    for ((&column, &hp), &base) in histogram_columns.iter().zip(&hist_plans).zip(&hist_offsets) {
        match hp {
            HistogramPlan::Digit { shape } | HistogramPlan::Flag { shape } => {
                fallbacks[shape].push((column, base));
            }
            HistogramPlan::Cycle => direct_histograms.push((column, base)),
            HistogramPlan::Row => row_histograms.push((column, row_hist_base)),
        }
        row_hist_base += 1 << source.bits(column);
    }
    let geometry = CycleChunks::new(log_t, 0).expect("validated cycles");
    let (low, high) = geometry.split_point(r_cycle).expect("checked cycle point");
    let low = eq_table(low, None);
    let high = eq_table(high, None);
    let pool = ScratchPool::new(scratch_len).map_err(|_| RouterError::Dimension {
        variables: usize::BITS as usize,
    })?;
    let mut weights = vec![ZERO; source.cycles()];
    if let Some(clock) = start {
        times[3] += clock.elapsed();
        start = Some(Instant::now());
    }
    plan.emission_chunks(&mut weights)
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
                            let size = word_entries(bytes);
                            let base = sl.bases[h];
                            for ((_, word), storage) in sl.words.iter().zip(
                                scratch[base..base + sl.words.len() * size].chunks_exact_mut(size),
                            ) {
                                let word = match *word {
                                    WordSlot::Trace(index) => source.trace_word(index, cycle),
                                    WordSlot::Bytecode(index) => {
                                        source.bytecode_word(index, source.bytecode_index(cycle))
                                    }
                                    WordSlot::Bits(_) | WordSlot::Zero => 0,
                                };
                                bucket_word(storage, word, e, bytes);
                            }
                            let mut base = sl.metadata + h * sl.meta_len();
                            for &(column, bound) in &sl.digits {
                                if let Some(digit) = source.digit(column, cycle) {
                                    let storage = &mut scratch[base..base + bound];
                                    if bound == 16 {
                                        let mut buckets = NibbleBuckets::new(storage)
                                            .expect("one padded digit position");
                                        buckets.positions_mut()[0][digit & 15] += e;
                                    } else if let Some(last) = storage.len().checked_sub(1) {
                                        storage[digit.min(last)] += e;
                                    }
                                }
                                base += bound;
                            }
                            for flags in sl.flags.chunks(4) {
                                let mut value = 0;
                                for (bit, &column) in flags.iter().enumerate() {
                                    value |=
                                        usize::from(source.digit(column, cycle).is_some()) << bit;
                                }
                                let mut buckets = NibbleBuckets::new(&mut scratch[base..base + 16])
                                    .expect("one packed flag position");
                                buckets.positions_mut()[0][value & 15] += e;
                                base += 16;
                            }
                            if let Some(base) = sl.totals {
                                scratch[base + h] += e;
                            }
                        } else {
                            for &(column, base) in &fallbacks[index] {
                                if let Some(digit) = source.digit(column, cycle) {
                                    scratch[base + digit] += e;
                                }
                            }
                        }
                    }
                    for &(column, base) in &direct_histograms {
                        if let Some(digit) = source.digit(column, cycle) {
                            scratch[base + digit] += e;
                        }
                    }
                }
            }
        });
    if let Some(clock) = start {
        times[0] = clock.elapsed();
        start = Some(Instant::now());
    }
    drop(low);
    drop(high);
    drop(fallbacks);
    drop(direct_histograms);
    let mut ra_fold = vec![ZERO; source.bytecode_rows()];
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
        .map(|shape| vec![ZERO; shape.fold_len()])
        .collect();
    readout(shapes, layout, &buckets, &[], false, &mut folds);
    let mut histograms = Vec::with_capacity(histogram_columns.len());
    for ((&column, &hp), &base) in histogram_columns.iter().zip(&hist_plans).zip(&hist_offsets) {
        let bound = 1 << source.bits(column);
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
    drop(buckets);
    let mut row_scratch_len = layout.row_entries;
    for &column in histogram_columns {
        storage_add(&mut row_scratch_len, 1 << source.bits(column))?;
    }
    let row_pool = ScratchPool::new(row_scratch_len).map_err(|_| RouterError::Dimension {
        variables: usize::BITS as usize,
    })?;
    if let Some(clock) = start {
        times[3] += clock.elapsed();
        start = Some(Instant::now());
    }
    let row_shapes: Vec<_> = shapes
        .iter()
        .zip(&layout.shapes)
        .filter(|(_, sl)| !sl.row_words.is_empty())
        .collect();
    ra_fold
        .par_chunks(geometry.chunk_len())
        .enumerate()
        .for_each(|(chunk, rows)| {
            let mut scratch = row_pool.take().expect("sequential row body");
            for (offset, &e) in rows.iter().enumerate() {
                if e == ZERO {
                    continue;
                }
                let row = chunk * geometry.chunk_len() + offset;
                for &(shape, sl) in &row_shapes {
                    if let Some(h) = shape.selector(source, row, true) {
                        let bytes = sl.bytes(h);
                        let size = word_entries(bytes);
                        let base = sl.row_bases[h];
                        for (&(_, word), storage) in sl.row_words.iter().zip(
                            scratch[base..base + sl.row_words.len() * size].chunks_exact_mut(size),
                        ) {
                            bucket_word(storage, source.bytecode_word(word, row), e, bytes);
                        }
                    }
                }
                for &(column, base) in &row_histograms {
                    if let Some(digit) = source.row_digit(column, row) {
                        scratch[base + digit] += e;
                    }
                }
            }
        });
    if let Some(clock) = start {
        times[2] = clock.elapsed();
        start = Some(Instant::now());
    }
    drop(row_shapes);
    drop(row_histograms);
    let row_buckets = row_pool.merge().expect("all row loans returned");
    readout(shapes, layout, &[], &row_buckets, true, &mut folds);
    let mut row_base = layout.row_entries;
    for ((&column, &hp), histogram) in histogram_columns
        .iter()
        .zip(&hist_plans)
        .zip(&mut histograms)
    {
        let bound = 1 << source.bits(column);
        if matches!(hp, HistogramPlan::Row) {
            histogram.copy_from_slice(&row_buckets[row_base..row_base + bound]);
        }
        row_base += bound;
    }
    if let Some(clock) = start {
        times[3] += clock.elapsed();
    }
    Ok((
        FoldOutput {
            folds,
            ra_fold,
            histograms,
        },
        times,
    ))
}

#[expect(
    clippy::expect_used,
    reason = "FoldLayout checked every source slot and bit entry"
)]
fn readout(
    shapes: &[RouterShape],
    layout: &FoldLayout,
    buckets: &[F128],
    row_buckets: &[F128],
    row_mode: bool,
    folds: &mut [Vec<F128>],
) {
    shapes
        .iter()
        .zip(&layout.shapes)
        .zip(folds)
        .for_each(|((shape, sl), fold)| {
            if row_mode && sl.row_words.is_empty() {
                return;
            }
            for h in 0..sl.selectors {
                let bytes = sl.bytes(h);
                let size = word_entries(bytes);
                let bits = if bytes { 8 } else { 4 };
                let width = 1 << bits;
                let total = if row_mode {
                    row_buckets[sl.row_bases[h]..sl.row_bases[h] + width]
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
                let mut selector_shift = [0; 3];
                for index in 1..shape.factors().len() {
                    selector_shift[index] =
                        selector_shift[index - 1] + shape.factors()[index - 1].slots.len();
                }
                for (word_slot, word) in shape.bank().iter().enumerate() {
                    let row_word = sl.row_words.iter().any(|&(slot, _)| slot == word_slot);
                    if matches!(word, WordSlot::Trace(_) | WordSlot::Bytecode(_))
                        && row_word != row_mode
                    {
                        continue;
                    }
                    if row_mode
                        && matches!(word, WordSlot::Bits(_))
                        && (!sl.words.is_empty() || sl.totals.is_some())
                    {
                        continue;
                    }
                    let entries = match word {
                        WordSlot::Zero => continue,
                        WordSlot::Bits(entries) => entries.len(),
                        _ => 64,
                    };
                    let mut destination = 0;
                    for (position, &(_, variable)) in shape.slot_map().iter().enumerate().skip(6) {
                        let value = match variable {
                            SlotVariable::Bit(_) => 0,
                            SlotVariable::Word(bit) => word_slot >> bit,
                            SlotVariable::Selector { factor, bit } => {
                                h >> (selector_shift[factor] + bit)
                            }
                        };
                        destination |= (value & 1) << position;
                    }
                    let word_storage = sl
                        .words
                        .iter()
                        .position(|(slot, _)| *slot == word_slot)
                        .map(|index| (&buckets, sl.bases[h] + index * size))
                        .or_else(|| {
                            sl.row_words
                                .iter()
                                .position(|&(slot, _)| slot == word_slot)
                                .map(|index| (&row_buckets, sl.row_bases[h] + index * size))
                        });
                    for bit in 0..entries {
                        if let WordSlot::Bits(entries) = word {
                            let one = matches!(entries[bit], BitEntry::One);
                            if row_mode != (one && sl.words.is_empty() && sl.totals.is_none()) {
                                continue;
                            }
                        }
                        let value = match word {
                            WordSlot::Trace(_) | WordSlot::Bytecode(_) => {
                                let (storage, base) = word_storage.expect("checked word slot");
                                let position = &storage
                                    [base + (bit / bits) * width..base + (bit / bits + 1) * width];
                                position
                                    .iter()
                                    .enumerate()
                                    .filter(|(value, _)| value & (1 << (bit % bits)) != 0)
                                    .map(|(_, &value)| value)
                                    .sum()
                            }
                            WordSlot::Bits(entries) => match entries.get(bit) {
                                Some(BitEntry::One) => total,
                                Some(BitEntry::Indicator { column, value }) => {
                                    if let Some((base, _)) = sl.digit_base(h, *column) {
                                        buckets[base + value]
                                    } else {
                                        let (base, flag) =
                                            sl.flag_base(h, *column).expect("checked flag");
                                        (0..16)
                                            .filter(|value| value & (1 << flag) != 0)
                                            .map(|value| buckets[base + value])
                                            .sum()
                                    }
                                }
                                Some(BitEntry::DigitBit { column, bit }) => {
                                    let (base, bound) =
                                        sl.digit_base(h, *column).expect("checked digit bit");
                                    (0..bound)
                                        .filter(|value| value & (1 << bit) != 0)
                                        .map(|value| buckets[base + value])
                                        .sum()
                                }
                                Some(BitEntry::Zero) | None => ZERO,
                            },
                            WordSlot::Zero => ZERO,
                        };
                        // new pins the six bit variables to the first six occupied slots.
                        fold[destination + bit] = value;
                    }
                }
            }
        });
}
