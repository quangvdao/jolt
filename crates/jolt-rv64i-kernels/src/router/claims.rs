//! Claim extraction sums retained word lifts and scatters cycle equality weights.

use jolt_field::{Accumulator, F128Accumulator, Zero, F128};
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;
use std::ops::Add;

use super::lift::RetainedWordLifts;
use super::shape::RouterError;
use crate::packed::scatter::ScatterPlan;
use crate::par::CycleChunks;
use crate::round::eq::eq_table;
use crate::source::{CycleSource, ValidatedTrace};

const ZERO: F128 = F128::from_raw(0);
const WORDS_PER_TILE: usize = 8;

#[repr(transparent)]
#[derive(Clone, Copy, Default)]
struct Partial(F128Accumulator);
impl Add for Partial {
    type Output = Self;
    fn add(mut self, other: Self) -> Self {
        self.0.merge(other.0);
        self
    }
}
impl Zero for Partial {
    fn zero() -> Self {
        Self::default()
    }
    fn is_zero(&self) -> bool {
        self.0.reduce() == ZERO
    }
}

/// Listed trace-word claims, all bytecode-word claims, and the equality scatter.
/// Trace claims follow the requested order; bytecode claims follow source order.
/// Each word is evaluated at the retained bit point and the supplied cycle point.
pub struct ClaimsOutput {
    pub trace_words: Vec<F128>,
    pub bytecode_words: Vec<F128>,
    pub row_weights: Vec<F128>,
}

struct Pass<'a> {
    geometry: CycleChunks,
    low: Vec<F128>,
    high: Vec<F128>,
    tables: Vec<&'a [F128]>,
}

impl Pass<'_> {
    #[expect(
        clippy::expect_used,
        reason = "caller checks exact word count and full cycle tables before fixed-width dispatch"
    )]
    fn fixed<const N: usize, S: CycleSource>(
        &self,
        plan: &ScatterPlan<S>,
        weights: &mut [F128],
    ) -> [F128Accumulator; N] {
        let tables: [&[F128]; N] = self.tables.as_slice().try_into().expect("fixed word count");
        let blocks = self.geometry.chunk_len() / self.geometry.block_len();
        plan.emission_chunks(weights)
            .expect("checked plan and newly sized weights")
            .zip(self.high.par_chunks(blocks))
            .map(|((interval, slots, segment), high)| {
                let tables = tables.map(|table| &table[interval.clone()]);
                let mut sums = [F128Accumulator::default(); N];
                for (block, ((slots, &hi), lo)) in slots
                    .chunks_exact(self.low.len())
                    .zip(high)
                    .zip(std::iter::repeat(&self.low))
                    .enumerate()
                {
                    let base = block * lo.len();
                    for (offset, (&lo, &slot)) in lo.iter().zip(slots).enumerate() {
                        let index = base + offset;
                        let e = lo * hi;
                        segment[usize::from(slot)] = e;
                        for (sum, table) in sums.iter_mut().zip(&tables) {
                            sum.fmadd(e, table[index]);
                        }
                    }
                }
                sums
            })
            .reduce(
                || [F128Accumulator::default(); N],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        left.merge(right);
                    }
                    left
                },
            )
    }

    #[expect(
        clippy::expect_used,
        reason = "caller checks exact table and scatter dimensions before emission"
    )]
    fn dynamic<S: CycleSource>(&self, plan: &ScatterPlan<S>, weights: &mut [F128]) -> Vec<F128> {
        const TILE: usize = 64;
        let words = self.tables.len();
        let chunks = self.geometry.len() / self.geometry.chunk_len();
        let mut partials: Vec<Partial> = unsafe_allocate_zero_vec(chunks * words);
        plan.emission_chunks(weights)
            .expect("checked plan and newly sized weights")
            .zip(partials.par_chunks_mut(words))
            .for_each(|((interval, slots, segment), sums)| {
                let block_len = self.low.len();
                for (block, slots) in slots.chunks_exact(block_len).enumerate() {
                    let hi = self.high[interval.start / block_len + block];
                    let base = interval.start + block * block_len;
                    for (tile, (lo, slots)) in
                        self.low.chunks(TILE).zip(slots.chunks(TILE)).enumerate()
                    {
                        let mut equality = [ZERO; TILE];
                        for ((e, &lo), &slot) in equality.iter_mut().zip(lo).zip(slots) {
                            *e = lo * hi;
                            segment[usize::from(slot)] = *e;
                        }
                        for (total, table) in sums.iter_mut().zip(&self.tables) {
                            let start = base + tile * TILE;
                            let mut sum = F128Accumulator::default();
                            for (&e, &value) in equality.iter().zip(&table[start..start + lo.len()])
                            {
                                sum.fmadd(e, value);
                            }
                            total.0.merge(sum);
                        }
                    }
                }
            });
        let mut totals = vec![F128Accumulator::default(); words];
        for chunk in partials.chunks_exact(words) {
            for (total, &partial) in totals.iter_mut().zip(chunk) {
                total.merge(partial.0);
            }
        }
        totals.into_iter().map(Accumulator::reduce).collect()
    }

    fn trace<S: CycleSource>(&self, plan: &ScatterPlan<S>, weights: &mut [F128]) -> Vec<F128> {
        macro_rules! counts {
            ($($count:literal),*) => {
                match self.tables.len() {
                    $($count => self.fixed::<$count, _>(plan, weights).map(Accumulator::reduce).to_vec(),)*
                    _ => self.dynamic(plan, weights),
                }
            };
        }
        counts!(0, 1, 2, 3, 4, 5, 6, 7, 8)
    }

    fn bytecode<S: CycleSource>(
        &self,
        source: &S,
        lifts: &RetainedWordLifts,
        rows: &[F128],
    ) -> Vec<F128> {
        let words = source.bytecode_words();
        let mut values = vec![ZERO; words];
        for base in (0..words).step_by(WORDS_PER_TILE) {
            let count = WORDS_PER_TILE.min(words - base);
            let sums = rows
                .par_chunks(self.geometry.chunk_len())
                .enumerate()
                .map(|(index, rows)| {
                    let mut sums = [F128Accumulator::default(); WORDS_PER_TILE];
                    let start = index * self.geometry.chunk_len();
                    for (offset, &weight) in rows.iter().enumerate() {
                        if weight != ZERO {
                            for (word, sum) in sums[..count].iter_mut().enumerate() {
                                sum.fmadd(
                                    weight,
                                    lifts
                                        .lift
                                        .lift(source.bytecode_word(base + word, start + offset)),
                                );
                            }
                        }
                    }
                    sums
                })
                .reduce(
                    || [F128Accumulator::default(); WORDS_PER_TILE],
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            left.merge(right);
                        }
                        left
                    },
                );
            for (value, sum) in values[base..base + count].iter_mut().zip(sums) {
                *value = sum.reduce();
            }
        }
        values
    }
}

/// Sum each listed retained trace word at the cycle point `r_prime`, scatter
/// `eq(r_prime,j)` through `plan`, and sum every bytecode word against the scatter.
/// Checks cycle-point length, listed trace indices, retained table lengths, and
/// the plan's cycle and row dimensions. Each cycle equality is formed once and
/// feeds the trace accumulators and scatter in the same parallel pass.
/// Agreement of the plan, retained lifts and immutable source is required of
/// the caller, not checked, and incorrect word claims are detected by the verifier.
pub fn claims_pass<S: CycleSource>(
    trace: &ValidatedTrace<S>,
    lifts: &RetainedWordLifts,
    words: &[usize],
    plan: &ScatterPlan<S>,
    r_prime: &[F128],
) -> Result<ClaimsOutput, RouterError> {
    let source = trace.source().as_ref();
    let log_t = source.cycles().ilog2() as usize;
    if r_prime.len() != log_t {
        return Err(RouterError::PointLength {
            expected: log_t,
            actual: r_prime.len(),
        });
    }
    for (table, actual, expected) in [
        ("scatter cycles", plan.cycles(), source.cycles()),
        ("scatter rows", plan.bytecode_rows(), source.bytecode_rows()),
    ] {
        if actual != expected {
            return Err(RouterError::TableLength {
                table,
                expected,
                actual,
            });
        }
    }
    let mut tables = Vec::with_capacity(words.len());
    for &word in words {
        if word >= source.trace_words() {
            return Err(RouterError::WordIndex {
                bank: "trace",
                index: word,
                words: source.trace_words(),
            });
        }
        let index = lifts
            .words
            .binary_search(&word)
            .map_err(|_| RouterError::MissingRetainedWord { word })?;
        let table = &lifts.tables[index];
        if table.len() != source.cycles() {
            return Err(RouterError::TableLength {
                table: "word lift",
                expected: source.cycles(),
                actual: table.len(),
            });
        }
        tables.push(table.as_slice());
    }
    let geometry = CycleChunks::new(log_t, 0)?;
    let (low, high) = geometry.split_point(r_prime)?;
    let pass = Pass {
        geometry,
        low: eq_table(low, None),
        high: eq_table(high, None),
        tables,
    };
    let mut weights = unsafe_allocate_zero_vec(source.cycles());
    let trace_words = pass.trace(plan, &mut weights);
    let mut row_weights = unsafe_allocate_zero_vec(source.bytecode_rows());
    plan.apply_buffer(&weights, &mut row_weights)?;
    let bytecode_words = pass.bytecode(source, lifts, &row_weights);
    Ok(ClaimsOutput {
        trace_words,
        bytecode_words,
        row_weights,
    })
}
