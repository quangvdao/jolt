//! Equality-weighted cycle products with one shared transition per batch round.

use super::shape::{RouterError, RouterShape, SelectorFactor};
use crate::par::CycleChunks;
use crate::round::eq::{eq_table, split_eq};
use crate::round::{coefficients_from_nodes, linear_at_nodes, quadratic, quadratic_at_nodes};
use crate::source::{CycleSource, ValidatedTrace};
use jolt_field::{Accumulator, F128Accumulator, Zero, F128};
use jolt_kernels::optimized::lazy_ra::{ChunkIndexSource, LazyFoldedRa};
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;
use std::alloc::Layout;
use std::num::NonZeroU8;
use std::ops::{Add, BitXor};
use std::sync::{Arc, Mutex};
#[cfg(feature = "test-utils")]
use std::time::{Duration, Instant};

const ZERO: F128 = F128::from_raw(0);

// Option<NonZeroU8> has the all-zero representation None, the zero allocator's
// required additive identity. The addition is only its Zero trait requirement.
#[repr(transparent)]
#[derive(Clone, Copy, Default)]
struct CompactDigit(Option<NonZeroU8>);

impl Add for CompactDigit {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(NonZeroU8::new(
            self.0
                .map_or(0, NonZeroU8::get)
                .bitxor(other.0.map_or(0, NonZeroU8::get)),
        ))
    }
}
impl Zero for CompactDigit {
    fn zero() -> Self {
        Self(None)
    }
    fn is_zero(&self) -> bool {
        self.0.is_none()
    }
}

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

struct CompactColumns {
    digits: Arc<Vec<CompactDigit>>,
    widths: Vec<usize>,
    cycles: usize,
}
impl ChunkIndexSource for CompactColumns {
    fn num_polys(&self) -> usize {
        self.widths.len()
    }
    fn cycles(&self) -> usize {
        self.cycles
    }
    #[inline]
    fn index(&self, column: usize, cycle: usize) -> Option<usize> {
        self.digits[cycle * self.widths.len() + column]
            .0
            .map(|value| usize::from(value.get() - 1))
    }
    fn index_bound(&self, column: usize) -> Option<usize> {
        Some(1 << self.widths[column])
    }
}

struct SourceTable {
    table: Vec<F128>,
    scratch: Vec<F128>,
}
struct Single {
    shape: usize,
    column: usize,
}
struct Double {
    shape: usize,
    columns: [usize; 2],
}
struct Triple {
    shape: usize,
    column: usize,
}
struct TripleGroup {
    columns: [usize; 2],
    members: Vec<Triple>,
}
const SHAPES_PER_BATCH: usize = 8;

#[derive(Default)]
struct BatchRecipe {
    singles: Vec<Single>,
    doubles: Vec<Double>,
    triples: Vec<TripleGroup>,
    degrees: Vec<usize>,
}
struct Recipe {
    columns: Vec<Vec<usize>>,
    degrees: Vec<usize>,
    batches: Vec<BatchRecipe>,
}
struct Job<'a> {
    chunk: usize,
    shape: usize,
    input: &'a [F128],
    output: &'a mut [F128],
}

#[derive(Clone, Copy)]
struct RoundSums {
    nodes: [F128; 4],
    endpoint: Option<F128>,
}

#[derive(Clone, Copy)]
struct ComputedRound {
    round: usize,
    bind: Option<F128>,
}

struct Shared {
    sources: Vec<SourceTable>,
    columns: LazyFoldedRa<F128, CompactColumns>,
    eq: GruenSplitEqPolynomial<F128>,
    recipe: Recipe,
    log_t: usize,
    computed: Option<ComputedRound>,
    sums: Vec<RoundSums>,
    partials: Vec<Partial>,
    finished: Option<F128>,
    failed: bool,
    support: Option<Arc<Vec<CompactDigit>>>,
    final_sources: Vec<F128>,
    #[cfg(feature = "test-utils")]
    times: [Duration; 2],
}

/// Shared degree `2 + factors` cycle kernels for checked router shapes.
/// Source tables must be `Source_ρ(x|src, ·)`: this is required of the caller,
/// not checked, and detected by the verifier's final evaluation check.
/// Points and table indices have their low variable first.
pub struct RoutersCycleCore {
    shared: Arc<Mutex<Shared>>,
    log_t: usize,
    shapes: usize,
}

impl RoutersCycleCore {
    /// Takes one `T`-element source table per shape and compiles all factor
    /// references into one byte per cycle and distinct `(column, slots)`.
    /// Checks point lengths, source references, table lengths and compact widths.
    pub fn new<S: CycleSource>(
        source: &ValidatedTrace<S>,
        shapes: &[RouterShape],
        r_cycle: &[F128],
        x: &[F128],
        source_tables: Vec<Vec<F128>>,
    ) -> Result<Self, RouterError> {
        let source = source.source();
        let cycles = source.cycles();
        let log_t = cycles.ilog2() as usize;
        if r_cycle.len() != log_t {
            return Err(RouterError::PointLength {
                expected: log_t,
                actual: r_cycle.len(),
            });
        }
        if shapes.is_empty() {
            return Err(RouterError::EmptyShapes);
        }
        for shape in shapes {
            shape.check_source(source.as_ref())?;
            if x.len() != shape.slots() {
                return Err(RouterError::PointLength {
                    expected: shape.slots(),
                    actual: x.len(),
                });
            }
        }
        if source_tables.len() != shapes.len() {
            return Err(RouterError::TableLength {
                table: "router source tables",
                expected: shapes.len(),
                actual: source_tables.len(),
            });
        }
        for table in &source_tables {
            if table.len() != cycles {
                return Err(RouterError::TableLength {
                    table: "router source",
                    expected: cycles,
                    actual: table.len(),
                });
            }
        }
        let geometry = CycleChunks::new(log_t, 0)?;
        let eq = split_eq(r_cycle, None)?;
        let mut factors: Vec<SelectorFactor> = Vec::new();
        let mut recipe = Recipe {
            columns: Vec::new(),
            degrees: Vec::new(),
            batches: Vec::new(),
        };
        for (shape, value) in shapes.iter().enumerate() {
            let mut columns = Vec::new();
            for factor in value.factors() {
                let bits = source.bits(factor.column);
                if bits >= 8 {
                    return Err(RouterError::FactorCapacity {
                        column: factor.column,
                        bound: 7,
                        width: bits,
                    });
                }
                let index = factors
                    .iter()
                    .position(|other| other == factor)
                    .unwrap_or_else(|| {
                        factors.push(factor.clone());
                        factors.len() - 1
                    });
                columns.push(index);
            }
            if shape.is_multiple_of(SHAPES_PER_BATCH) {
                recipe.batches.push(BatchRecipe::default());
            }
            let batch = &mut recipe.batches[shape / SHAPES_PER_BATCH];
            let local_shape = shape % SHAPES_PER_BATCH;
            match *columns.as_slice() {
                [column] => batch.singles.push(Single {
                    shape: local_shape,
                    column,
                }),
                [a, b] => batch.doubles.push(Double {
                    shape: local_shape,
                    columns: [a, b],
                }),
                [a, b, c] => {
                    let group = batch
                        .triples
                        .iter()
                        .position(|group| group.columns == [a, b])
                        .unwrap_or_else(|| {
                            batch.triples.push(TripleGroup {
                                columns: [a, b],
                                members: Vec::new(),
                            });
                            batch.triples.len() - 1
                        });
                    batch.triples[group].members.push(Triple {
                        shape: local_shape,
                        column: c,
                    });
                }
                _ => {
                    return Err(RouterError::Factors {
                        count: columns.len(),
                    })
                }
            }
            let degree = columns.len() + 1;
            batch.degrees.push(degree);
            recipe.degrees.push(degree);
            recipe.columns.push(columns);
        }
        let width = factors.len();
        let digit_len = checked_len::<CompactDigit>(cycles.checked_mul(width), log_t)?;
        let mut digits: Vec<CompactDigit> = unsafe_allocate_zero_vec(digit_len);
        digits
            .par_chunks_mut(geometry.chunk_len() * width)
            .enumerate()
            .for_each(|(chunk, output)| {
                let start = chunk * geometry.chunk_len();
                for (offset, row) in output.chunks_exact_mut(width).enumerate() {
                    for (digit, factor) in row.iter_mut().zip(&factors) {
                        digit.0 = source
                            .digit(factor.column, start + offset)
                            .and_then(|value| NonZeroU8::new((value + 1) as u8));
                    }
                }
            });
        let tables = factors
            .iter()
            .map(|factor| {
                eq_table(
                    &factor.slots.iter().map(|&slot| x[slot]).collect::<Vec<_>>(),
                    None,
                )
            })
            .collect();
        let digits = Arc::new(digits);
        let support = Some(Arc::clone(&digits));
        let compact = CompactColumns {
            digits,
            widths: factors.iter().map(|factor| factor.slots.len()).collect(),
            cycles,
        };
        let columns = LazyFoldedRa::try_new(tables, compact)?;
        let sources = source_tables
            .into_iter()
            .map(|table| SourceTable {
                table,
                scratch: unsafe_allocate_zero_vec(cycles / 2),
            })
            .collect();
        let pairs_geometry = CycleChunks::new(log_t, 1)?;
        let partial_len = checked_len::<Partial>(
            (pairs_geometry.len() / pairs_geometry.chunk_len())
                .checked_mul(shapes.len())
                .and_then(|n| n.checked_mul(8)),
            log_t,
        )?;
        let shared = Shared {
            sources,
            columns,
            eq,
            recipe,
            log_t,
            computed: None,
            sums: vec![
                RoundSums {
                    nodes: [ZERO; 4],
                    endpoint: None
                };
                shapes.len()
            ],
            partials: unsafe_allocate_zero_vec(partial_len),
            finished: None,
            failed: false,
            support,
            final_sources: Vec::with_capacity(shapes.len()),
            #[cfg(feature = "test-utils")]
            times: [Duration::ZERO; 2],
        };
        Ok(Self {
            shared: Arc::new(Mutex::new(shared)),
            log_t,
            shapes: shapes.len(),
        })
    }

    /// One handle per shape, in shape order. Handles share source and selector
    /// transitions; the batch may visit them in any order each round.
    pub fn members(&self) -> Vec<RouterCycleMember> {
        (0..self.shapes)
            .map(|shape| RouterCycleMember {
                shared: Arc::clone(&self.shared),
                shape,
                log_t: self.log_t,
                next: 0,
                finished: false,
            })
            .collect()
    }

    /// Accumulated whole-pass time `[products and messages, selector binds]`.
    /// Available only for the benchmark fixture feature; neither clock is read
    /// in a cycle or pair loop.
    #[cfg(feature = "test-utils")]
    pub fn phase_times(&self) -> [Duration; 2] {
        self.shared
            .lock()
            .map_or([Duration::ZERO; 2], |shared| shared.times)
    }
}

/// One equality-weighted shape member of a shared router cycle sumcheck.
/// Honest input claims are required of the caller, not checked by division
/// recovery, and detected by the verifier's final evaluation check.
pub struct RouterCycleMember {
    shared: Arc<Mutex<Shared>>,
    shape: usize,
    log_t: usize,
    next: usize,
    finished: bool,
}
impl RouterCycleMember {
    /// Returns source and factors at the cycle challenge point, in factor order.
    /// This member must have finished; earlier reads return an error. Complete
    /// every member's `finish_rounds` before reading the batch's final values.
    pub fn final_values(&self) -> Result<(F128, Vec<F128>), RouterError> {
        let shared = self.shared.lock().map_err(|_| RouterError::Poisoned)?;
        if shared.finished.is_none() || !self.finished {
            return Err(RouterError::Unfinished);
        }
        Ok((
            shared.final_sources[self.shape],
            shared.recipe.columns[self.shape]
                .iter()
                .map(|&column| shared.columns.value(column, 0))
                .collect(),
        ))
    }
}

fn checked_len<T>(len: Option<usize>, variables: usize) -> Result<usize, RouterError> {
    let len = len.ok_or(RouterError::Dimension { variables })?;
    let _layout = Layout::array::<T>(len).map_err(|_| RouterError::Dimension { variables })?;
    Ok(len)
}
fn missing(kind: &'static str) -> SumcheckError<F128> {
    SumcheckError::MissingEvaluationSource { kind }
}

impl Shared {
    fn transition(&mut self, bind: Option<F128>, round: usize) -> Result<(), SumcheckError<F128>> {
        if self.failed || self.finished.is_some() {
            return Err(missing("router active round"));
        }
        if let Some(computed) = self.computed {
            if computed.round == round {
                return if computed.bind == bind {
                    Ok(())
                } else {
                    Err(missing("router previous challenge disagreement"))
                };
            }
        }
        let expected = self.computed.map_or(0, |previous| previous.round + 1);
        if round != expected || (round == 0) != bind.is_none() {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected,
                got: round,
            });
        }
        if let Some(challenge) = bind {
            #[cfg(feature = "test-utils")]
            let start = Instant::now();
            self.columns.bind(challenge);
            if round == 4 {
                self.support = None;
            }
            #[cfg(feature = "test-utils")]
            {
                self.times[1] += start.elapsed();
            }
            self.eq.bind(challenge);
        }
        #[cfg(feature = "test-utils")]
        let start = Instant::now();
        if self.columns.num_polys() <= 8 {
            if bind.is_some() {
                self.accumulate::<true, true>(bind, round)?;
            } else {
                self.accumulate::<false, true>(bind, round)?;
            }
        } else if bind.is_some() {
            self.accumulate::<true, false>(bind, round)?;
        } else {
            self.accumulate::<false, false>(bind, round)?;
        }
        #[cfg(feature = "test-utils")]
        {
            self.times[0] += start.elapsed();
        }
        self.computed = Some(ComputedRound { round, bind });
        Ok(())
    }

    fn accumulate<const BIND: bool, const CACHE: bool>(
        &mut self,
        bind: Option<F128>,
        round: usize,
    ) -> Result<(), SumcheckError<F128>> {
        // With a zero Gruen endpoint, division cannot recover q(1). Its
        // exceptional node reuses the same pair products in this pass.
        if self.eq.current_linear_evals().1 == ZERO {
            self.accumulate_endpoint::<BIND, CACHE, true>(bind, round)
        } else {
            self.accumulate_endpoint::<BIND, CACHE, false>(bind, round)
        }
    }

    fn accumulate_endpoint<const BIND: bool, const CACHE: bool, const SINGULAR: bool>(
        &mut self,
        bind: Option<F128>,
        round: usize,
    ) -> Result<(), SumcheckError<F128>> {
        let geometry = CycleChunks::new(self.log_t, round + 1)
            .map_err(|_| missing("router cycle geometry"))?;
        let pairs = geometry.len();
        let chunk_pairs = geometry.chunk_len();
        let count = self.sources.len();
        let block_len = self.eq.e_in_current_len();
        let mut jobs = Vec::with_capacity(count * pairs / chunk_pairs);
        for (shape, source) in self.sources.iter_mut().enumerate() {
            if BIND {
                source.scratch.truncate(2 * pairs);
            }
            let input_stride = if BIND { 4 } else { 2 };
            for (chunk, (input, output)) in source
                .table
                .chunks_exact(chunk_pairs * input_stride)
                .zip(
                    source
                        .scratch
                        .chunks_mut(chunk_pairs * if BIND { 2 } else { 1 }),
                )
                .enumerate()
            {
                jobs.push(Job {
                    chunk,
                    shape,
                    input,
                    output,
                });
            }
        }
        jobs.sort_unstable_by_key(|job| (job.chunk, job.shape));
        let partial_len = count * (pairs / chunk_pairs) * 8;
        let partials = &mut self.partials[..partial_len];
        let columns = &self.columns;
        let recipe = &self.recipe;
        let support = self.support.as_deref();
        let column_count = columns.num_polys();
        let support_width = 2 << round;
        let inner_weights = self.eq.e_in_current();
        let outer_weights = self.eq.e_out_current();
        let challenge = bind.unwrap_or(ZERO);
        jobs.par_chunks_mut(count)
            .zip(partials.par_chunks_mut(count * 8))
            .enumerate()
            .for_each(|(chunk, (jobs, partials))| {
                partials.fill(Partial::default());
                // Bounded stack batches reread selectors above eight shapes, but
                // each source table is bound and streamed once in this chunk.
                for ((jobs, partials), recipe) in jobs
                    .chunks_mut(SHAPES_PER_BATCH)
                    .zip(partials.chunks_mut(SHAPES_PER_BATCH * 8))
                    .zip(&recipe.batches)
                {
                    let mut stack_totals = [[Partial::default(); 5]; SHAPES_PER_BATCH];
                    let mut stack_inner = [[Partial::default(); 5]; SHAPES_PER_BATCH];
                    for block in (0..chunk_pairs).step_by(block_len) {
                        for sums in stack_inner.iter_mut().take(jobs.len()) {
                            sums.fill(Partial::default());
                        }
                        for (offset, &weight) in inner_weights.iter().enumerate() {
                            let local_pair = block + offset;
                            let pair = chunk * chunk_pairs + local_pair;
                            let mut cached = [(ZERO, ZERO); 8];
                            if CACHE {
                                columns.lo_hi_all(pair, &mut cached[..columns.num_polys()]);
                            }
                            let support_at = |column: usize| {
                                support.is_none_or(|digits| {
                                    digits[pair * support_width * column_count
                                        ..(pair + 1) * support_width * column_count]
                                        .chunks_exact(column_count)
                                        .any(|row| row[column].0.is_some())
                                })
                            };
                            let mut cached_support = [true; 8];
                            if CACHE {
                                for (column, value) in
                                    cached_support[..column_count].iter_mut().enumerate()
                                {
                                    *value = support_at(column);
                                }
                            }
                            let supported = |column: usize| {
                                if CACHE {
                                    cached_support[column]
                                } else {
                                    support_at(column)
                                }
                            };
                            let factor = |column| {
                                let (lo, hi) = if CACHE {
                                    cached[column]
                                } else {
                                    columns.lo_hi(column, pair)
                                };
                                [lo, lo + hi]
                            };
                            let mut cached_sources = [[ZERO; 2]; SHAPES_PER_BATCH];
                            if BIND {
                                for (shape, job) in jobs.iter_mut().enumerate() {
                                    let values = &job.input[4 * local_pair..4 * local_pair + 4];
                                    let lo = values[0] + challenge * (values[0] + values[1]);
                                    let hi = values[2] + challenge * (values[2] + values[3]);
                                    job.output[2 * local_pair] = lo;
                                    job.output[2 * local_pair + 1] = hi;
                                    cached_sources[shape] = [lo, lo + hi];
                                }
                            } else {
                                for (shape, job) in jobs.iter().enumerate() {
                                    let values = &job.input[2 * local_pair..2 * local_pair + 2];
                                    cached_sources[shape] = [values[0], values[0] + values[1]];
                                }
                            }
                            let source = |shape: usize| cached_sources[shape];
                            for member in &recipe.singles {
                                if !supported(member.column) {
                                    continue;
                                }
                                let left = source(member.shape);
                                let right = factor(member.column);
                                let sums = &mut stack_inner[member.shape];
                                sums[0].0.fmadd(left[0] * right[0], weight);
                                sums[1].0.fmadd(left[1] * right[1], weight);
                                if SINGULAR {
                                    sums[4]
                                        .0
                                        .fmadd((left[0] + left[1]) * (right[0] + right[1]), weight);
                                }
                            }
                            for member in &recipe.doubles {
                                if !supported(member.columns[0]) || !supported(member.columns[1]) {
                                    continue;
                                }
                                let left =
                                    quadratic(source(member.shape), factor(member.columns[1]));
                                let right = factor(member.columns[0]);
                                let node = quadratic_at_nodes(left)[0] * linear_at_nodes(right)[0];
                                let values = [left[0] * right[0], left[2] * right[1], node];
                                let sums = &mut stack_inner[member.shape];
                                for (sum, value) in sums.iter_mut().zip(values) {
                                    sum.0.fmadd(value, weight);
                                }
                                if SINGULAR {
                                    sums[4].0.fmadd(
                                        (left[0] + left[1] + left[2]) * (right[0] + right[1]),
                                        weight,
                                    );
                                }
                            }
                            for group in &recipe.triples {
                                if !supported(group.columns[0]) || !supported(group.columns[1]) {
                                    continue;
                                }
                                let left =
                                    quadratic(factor(group.columns[0]), factor(group.columns[1]));
                                let left_nodes = quadratic_at_nodes(left);
                                for member in &group.members {
                                    if !supported(member.column) {
                                        continue;
                                    }
                                    let right =
                                        quadratic(source(member.shape), factor(member.column));
                                    let right_nodes = quadratic_at_nodes(right);
                                    let values = [
                                        left[0] * right[0],
                                        left[2] * right[2],
                                        left_nodes[0] * right_nodes[0],
                                        left_nodes[1] * right_nodes[1],
                                    ];
                                    let sums = &mut stack_inner[member.shape];
                                    for (sum, value) in sums.iter_mut().zip(values) {
                                        sum.0.fmadd(value, weight);
                                    }
                                    if SINGULAR {
                                        sums[4].0.fmadd(
                                            (left[0] + left[1] + left[2])
                                                * (right[0] + right[1] + right[2]),
                                            weight,
                                        );
                                    }
                                }
                            }
                        }
                        let outer = outer_weights[(chunk * chunk_pairs + block) / block_len];
                        for ((totals, inner), &points) in stack_totals
                            .iter_mut()
                            .zip(&stack_inner)
                            .zip(&recipe.degrees)
                        {
                            for (total, inner) in totals[..points].iter_mut().zip(&inner[..points])
                            {
                                total.0.fmadd(inner.0.reduce(), outer);
                            }
                            if SINGULAR {
                                totals[4].0.fmadd(inner[4].0.reduce(), outer);
                            }
                        }
                    }
                    for (output, totals) in partials.chunks_exact_mut(8).zip(stack_totals) {
                        output[..if SINGULAR { 5 } else { 4 }]
                            .copy_from_slice(&totals[..if SINGULAR { 5 } else { 4 }]);
                    }
                }
            });
        for shape in 0..count {
            let mut sums = [F128Accumulator::default(); 5];
            for chunk in partials.chunks_exact(count * 8) {
                for (sum, partial) in sums[..if SINGULAR { 5 } else { 4 }]
                    .iter_mut()
                    .zip(&chunk[shape * 8..shape * 8 + if SINGULAR { 5 } else { 4 }])
                {
                    sum.merge(partial.0);
                }
            }
            self.sums[shape] = RoundSums {
                nodes: std::array::from_fn(|node| sums[node].reduce()),
                endpoint: if SINGULAR {
                    Some(sums[4].reduce())
                } else {
                    None
                },
            };
        }
        drop(jobs);
        if BIND {
            for source in &mut self.sources {
                std::mem::swap(&mut source.table, &mut source.scratch);
            }
        }
        Ok(())
    }

    fn finish(&mut self, challenge: F128) -> Result<(), SumcheckError<F128>> {
        if let Some(previous) = self.finished {
            return if previous == challenge {
                Ok(())
            } else {
                Err(missing("router final challenge disagreement"))
            };
        }
        if self.failed || self.computed.map(|computed| computed.round) != Some(self.log_t - 1) {
            return Err(missing("router final transition"));
        }
        #[cfg(feature = "test-utils")]
        let start = Instant::now();
        self.columns.bind(challenge);
        #[cfg(feature = "test-utils")]
        {
            self.times[1] += start.elapsed();
        }
        self.eq.bind(challenge);
        #[cfg(feature = "test-utils")]
        let start = Instant::now();
        for source in &mut self.sources {
            let value = source.table[0] + challenge * (source.table[0] + source.table[1]);
            drop(std::mem::take(&mut source.table));
            drop(std::mem::take(&mut source.scratch));
            self.final_sources.push(value);
        }
        self.sources.clear();
        self.support = None;
        drop(std::mem::take(&mut self.partials));
        self.finished = Some(challenge);
        #[cfg(feature = "test-utils")]
        {
            self.times[0] += start.elapsed();
        }
        Ok(())
    }
}

impl ProveRounds<F128> for RouterCycleMember {
    fn num_rounds(&self) -> usize {
        self.log_t
    }
    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        if self.finished || round != self.next || round >= self.log_t {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: self.next,
                got: round,
            });
        }
        if (round == 0) != bind.is_none() {
            return Err(missing("router previous challenge"));
        }
        let mut shared = self
            .shared
            .lock()
            .map_err(|_| missing("router shared lock"))?;
        shared.transition(bind, round)?;
        #[cfg(feature = "test-utils")]
        let start = Instant::now();
        let sums = shared.sums[self.shape];
        let endpoint = if shared.eq.current_linear_evals().1 == ZERO {
            sums.endpoint
                .ok_or_else(|| missing("router singular endpoint"))?
        } else {
            ZERO
        };
        let degree = shared.recipe.degrees[self.shape];
        let at_one = shared
            .eq
            .recover_q_one(sums.nodes[0], claim, || endpoint)
            .map_err(|actual| {
                shared.failed = true;
                SumcheckError::RoundCheckFailed {
                    round,
                    expected: claim,
                    actual,
                }
            })?;
        let coefficients = coefficients_from_nodes(
            degree,
            sums.nodes[0],
            sums.nodes[1],
            at_one,
            &sums.nodes[2..degree],
        )
        .map_err(|_| missing("router interpolation"))?;
        let message = shared.eq.round_poly_from_q_coeffs(&coefficients);
        #[cfg(feature = "test-utils")]
        {
            shared.times[0] += start.elapsed();
        }
        self.next += 1;
        Ok(message)
    }
    fn finish_rounds(&mut self, challenge: F128) -> Result<(), SumcheckError<F128>> {
        if self.finished || self.next != self.log_t {
            return Err(missing("router member final transition"));
        }
        let mut shared = self
            .shared
            .lock()
            .map_err(|_| missing("router shared lock"))?;
        shared.finish(challenge)?;
        self.finished = true;
        Ok(())
    }
}
