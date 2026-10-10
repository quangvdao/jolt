//! Run with `cargo bench -p jolt-rv64i-kernels --features test-utils --bench
//! chunk_product -- --log-t 20,22 --threads 1,12 --samples 3`.

pub mod support;

use std::hint::black_box;
use std::mem::size_of;
use std::sync::Arc;
use std::time::Duration;

use jolt_field::{Accumulator, F128Accumulator, F128};
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_rv64i_kernels::chunk_product::{
    combined_weight, ChunkProductCore, ChunkProductError, ChunkWeight, ChunkWeightTerm, EqTerm,
};
use jolt_rv64i_kernels::par::{CycleChunks, ParError};
use jolt_rv64i_kernels::round::eq::eq_table;
use jolt_rv64i_kernels::source::{CycleSource, DigitColumns, SourceError, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthError, SynthProfile, SyntheticTrace};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use thiserror::Error;

use support::allocator::{
    AllocationAllowance, AllocationMeasurement, AllocationStats, RAYON_WORKER_ALLOWANCE,
};
use support::gather::run_gathers;
use support::{core_rounds, run_cases, seeded_challenges, Case, Clock, RunnerError};

#[derive(Debug, Error)]
enum BenchError {
    #[error(transparent)]
    Chunk(#[from] ChunkProductError),
    #[error(transparent)]
    Source(#[from] SourceError),
    #[error(transparent)]
    Par(#[from] ParError),
    #[error(transparent)]
    Synth(#[from] SynthError),
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError<F128>),
    #[error("core {quantity} was {actual} at log_t={log_t}, above limit {limit}")]
    Allocation {
        log_t: usize,
        quantity: &'static str,
        actual: usize,
        limit: usize,
    },
}

#[derive(Clone, Copy)]
enum Variant {
    Dense,
    DenseTwo,
    EqTerms,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::DenseTwo => "dense_two",
            Self::EqTerms => "eq_terms",
        }
    }
}

struct Prepared {
    trace: Arc<ValidatedTrace<SyntheticTrace>>,
    dense_claim: F128,
    equality_claims: [F128; 2],
}

struct TimedCore {
    inner: ChunkProductCore,
    combined: Duration,
    rounds: [Duration; 2],
}

impl ProveRounds<F128> for TimedCore {
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }
    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        let start = Clock::start();
        let result = self.inner.prove_round(bind, round, claim);
        self.rounds[usize::from(round >= 4)] += start.elapsed();
        result
    }
    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        self.inner.finish_rounds(bind)
    }
}

fn digit_points() -> Vec<Vec<F128>> {
    (0..5)
        .map(|column| {
            (0..4)
                .map(|bit| F128::from_raw(0x1325 + 31 * column + 7 * bit))
                .collect()
        })
        .collect()
}

fn equality_terms<const M: usize>(log_t: usize) -> [EqTerm; M] {
    std::array::from_fn(|term| EqTerm {
        coefficient: F128::from_raw(0x831 + term as u128),
        point: (0..log_t)
            .map(|bit| F128::from_raw(0x241 + (term * 37 + bit * 11) as u128))
            .collect(),
        claim: F128::from_raw(0),
    })
}

fn terms(log_t: usize, variant: Variant) -> Vec<ChunkWeightTerm> {
    let mut terms = equality_terms::<5>(log_t);
    if matches!(variant, Variant::Dense) {
        terms[4].point = (0..log_t)
            .map(|bit| F128::from_raw((bit % 2) as u128))
            .collect();
    }
    let count = if matches!(variant, Variant::Dense) {
        5
    } else {
        2
    };
    terms
        .into_iter()
        .take(count)
        .enumerate()
        .map(|(index, term)| {
            if matches!(variant, Variant::Dense) && index == 3 {
                ChunkWeightTerm::Next {
                    coefficient: term.coefficient,
                    point: term.point,
                }
            } else {
                ChunkWeightTerm::Eq {
                    coefficient: term.coefficient,
                    point: term.point,
                }
            }
        })
        .collect()
}

fn column_product(
    columns: &DigitColumns<SyntheticTrace>,
    tables: &[Vec<F128>],
    cycle: usize,
) -> F128 {
    let first = columns
        .index(0, cycle)
        .map_or(F128::from_raw(0), |digit| tables[0][digit]);
    tables
        .iter()
        .enumerate()
        .skip(1)
        .fold(first, |product, (column, table)| {
            product
                * columns
                    .index(column, cycle)
                    .map_or(F128::from_raw(0), |digit| table[digit])
        })
}

fn equality_claims<const M: usize>(
    columns: &DigitColumns<SyntheticTrace>,
    tables: &[Vec<F128>],
    terms: &[EqTerm; M],
    geometry: CycleChunks,
) -> Result<[F128; M], BenchError> {
    let halves: Vec<_> = terms
        .iter()
        .map(|term| {
            let (low, high) = geometry.split_point(&term.point)?;
            Ok::<_, BenchError>((eq_table(low, None), eq_table(high, Some(term.coefficient))))
        })
        .collect::<Result<_, _>>()?;
    Ok(halves[0]
        .1
        .par_chunks(geometry.chunk_len() / geometry.block_len())
        .enumerate()
        .map(|(chunk, highs)| {
            let mut outer = [F128Accumulator::default(); M];
            for offset in 0..highs.len() {
                let block = chunk * (geometry.chunk_len() / geometry.block_len()) + offset;
                let mut inner = [F128Accumulator::default(); M];
                for index in 0..geometry.block_len() {
                    let product =
                        column_product(columns, tables, block * geometry.block_len() + index);
                    for (sum, (low, _high)) in inner.iter_mut().zip(&halves) {
                        sum.fmadd(low[index], product);
                    }
                }
                for ((sum, inner), (_low, high)) in outer.iter_mut().zip(inner).zip(&halves) {
                    sum.fmadd(high[block], inner.reduce());
                }
            }
            outer
        })
        .reduce(
            || [F128Accumulator::default(); M],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    left.merge(right);
                }
                left
            },
        )
        .map(Accumulator::reduce))
}

impl Prepared {
    fn new(source: Arc<SyntheticTrace>) -> Result<Self, BenchError> {
        let log_t = source.cycles().ilog2() as usize;
        let trace = Arc::new(ValidatedTrace::new(source)?);
        let columns = DigitColumns::from_validated(Arc::clone(&trace), (0..5).collect())?;
        let tables: Vec<_> = digit_points()
            .iter()
            .map(|point| eq_table(point, None))
            .collect();
        let geometry = CycleChunks::new(log_t, 0)?;
        let dense = combined_weight(log_t, &terms(log_t, Variant::Dense))?;
        let dense_claim = dense
            .par_chunks(geometry.chunk_len())
            .enumerate()
            .map(|(chunk, weights)| {
                let mut sum = F128Accumulator::default();
                for (offset, &weight) in weights.iter().enumerate() {
                    sum.fmadd(
                        weight,
                        column_product(&columns, &tables, chunk * geometry.chunk_len() + offset),
                    );
                }
                sum
            })
            .reduce(F128Accumulator::default, |mut left, right| {
                left.merge(right);
                left
            })
            .reduce();
        let equality_claims =
            equality_claims(&columns, &tables, &equality_terms::<2>(log_t), geometry)?;
        Ok(Self {
            trace,
            dense_claim,
            equality_claims,
        })
    }
}

impl TimedCore {
    fn new(prepared: &Prepared, variant: Variant) -> Result<(Self, F128), BenchError> {
        let log_t = prepared.trace.source().cycles().ilog2() as usize;
        let columns = DigitColumns::from_validated(Arc::clone(&prepared.trace), (0..5).collect())?;
        let points = digit_points();
        let _geometry = CycleChunks::new(log_t, 0)?;
        let (weight, claim, combined) = match variant {
            Variant::Dense => {
                let terms = terms(log_t, variant);
                let start = Clock::start();
                let dense = combined_weight(log_t, &terms)?;
                (
                    ChunkWeight::Dense(dense),
                    prepared.dense_claim,
                    start.elapsed(),
                )
            }
            Variant::DenseTwo | Variant::EqTerms => {
                let mut eq_terms = equality_terms::<2>(log_t);
                let sums = prepared.equality_claims;
                for (term, claim) in eq_terms.iter_mut().zip(sums) {
                    term.claim = claim;
                }
                let claim = sums[0] + sums[1];
                let (weight, combined) = if matches!(variant, Variant::DenseTwo) {
                    let terms: Vec<_> = eq_terms
                        .into_iter()
                        .map(|term| ChunkWeightTerm::Eq {
                            coefficient: term.coefficient,
                            point: term.point,
                        })
                        .collect();
                    let start = Clock::start();
                    let dense = combined_weight(log_t, &terms)?;
                    (ChunkWeight::Dense(dense), start.elapsed())
                } else {
                    (
                        ChunkWeight::EqTerms(eq_terms.into_iter().collect()),
                        Duration::ZERO,
                    )
                };
                (weight, claim, combined)
            }
        };
        let inner = ChunkProductCore::new(columns, points, weight)?;
        Ok((
            Self {
                inner,
                combined,
                rounds: [Duration::ZERO; 2],
            },
            claim,
        ))
    }
}

fn nine_term_allocation_core(
    source: Arc<SyntheticTrace>,
) -> Result<(ChunkProductCore, F128), BenchError> {
    let log_t = source.cycles().ilog2() as usize;
    let columns = DigitColumns::new(source, (0..5).collect())?;
    let points = digit_points();
    let tables: Vec<_> = points.iter().map(|point| eq_table(point, None)).collect();
    let geometry = CycleChunks::new(log_t, 0)?;
    let mut terms = equality_terms::<9>(log_t);
    let sums = equality_claims(&columns, &tables, &terms, geometry)?;
    for (term, claim) in terms.iter_mut().zip(sums) {
        term.claim = claim;
    }
    let claim = sums
        .iter()
        .copied()
        .fold(F128::from_raw(0), |sum, claim| sum + claim);
    let core = ChunkProductCore::new(
        columns,
        points,
        ChunkWeight::EqTerms(terms.into_iter().collect()),
    )?;
    Ok((core, claim))
}

fn allocation_bound(log_t: usize, terms: usize) -> AllocationAllowance {
    let columns = 5;
    let column_bits = 4;
    let cycles = 1 << log_t;
    // Capacity bounds cover the shared dense bind's current/next buffers,
    // overlapping eight/sixteen-branch generations, and live round vectors.
    let column_bytes = 3 * columns * cycles / 2;
    let branch_bytes = 24 * columns * (1 << column_bits) * size_of::<F128>();
    let round_values = 8 * terms + (2 * terms + 2) * (columns + 2);
    let metadata_bytes = 4 * columns * (size_of::<Vec<F128>>() + size_of::<Polynomial<F128>>());
    AllocationAllowance {
        allocs: if terms > 7 {
            2 * (terms + 1) * log_t + 64
        } else {
            16 * log_t + 64
        },
        bytes: column_bytes + branch_bytes + round_values * size_of::<F128>() + metadata_bytes,
    }
}

const EQ_TERMS_NINE_ALLOCATIONS_PER_EXTRA_ROUND: usize = 25;

fn measure_round_allocations(
    log_t: usize,
    term_count: Option<usize>,
) -> Result<AllocationStats, BenchError> {
    let source = Arc::new(SyntheticTrace::new(
        SynthProfile::UniformDigits,
        log_t,
        16,
        0x5eed,
    )?);
    let (mut core, mut claim) = if term_count == Some(9) {
        nine_term_allocation_core(source)?
    } else {
        let variant = if term_count.is_some() {
            Variant::EqTerms
        } else {
            Variant::Dense
        };
        let prepared = Prepared::new(source)?;
        let (core, claim) = TimedCore::new(&prepared, variant)?;
        (core.inner, claim)
    };
    let challenges = seeded_challenges();
    let measurement = AllocationMeasurement::begin();
    let mut bind = None;
    for (round, &challenge) in challenges.iter().take(log_t).enumerate() {
        let message = core.prove_round(bind, round, claim)?;
        claim = message.evaluate(challenge);
        bind = Some(challenge);
    }
    core.finish_rounds(challenges[log_t - 1])?;
    Ok(measurement.finish())
}

fn round_chunk_visits(log_t: usize) -> Result<usize, ParError> {
    (1..=log_t)
        .map(|round| CycleChunks::new(log_t, round).map(|geometry| geometry.ranges().len()))
        .sum()
}

fn fit_allocation_growth(rounds: [usize; 3], visits: [usize; 3], counts: [usize; 3]) -> [f64; 3] {
    let rounds = rounds.map(|value| value as f64);
    let visits = visits.map(|value| value as f64);
    let counts = counts.map(|value| value as f64);
    let dr1 = rounds[1] - rounds[0];
    let dr2 = rounds[2] - rounds[0];
    let dv1 = visits[1] - visits[0];
    let dv2 = visits[2] - visits[0];
    let dc1 = counts[1] - counts[0];
    let dc2 = counts[2] - counts[0];
    let determinant = dr1 * dv2 - dr2 * dv1;
    let b = (dc1 * dv2 - dc2 * dv1) / determinant;
    let c = (dr1 * dc2 - dr2 * dc1) / determinant;
    [counts[0] - b * rounds[0] - c * visits[0], b, c]
}

#[expect(
    clippy::print_stdout,
    reason = "allocation growth acceptance reports measured counts and chunk visits"
)]
fn check_round_allocations_grow_with_rounds_not_chunk_visits(
    pool: &ThreadPool,
) -> Result<(), RunnerError> {
    let sizes = [14, 18];
    let visits = sizes.map(round_chunk_visits);
    let [small_visits, large_visits] = visits;
    let small_visits = small_visits.map_err(|error| RunnerError::Core {
        message: error.to_string(),
    })?;
    let large_visits = large_visits.map_err(|error| RunnerError::Core {
        message: error.to_string(),
    })?;
    if large_visits < 4 * small_visits {
        return Err(RunnerError::Core {
            message: "allocation growth cases must differ by at least four times the chunk visits"
                .to_owned(),
        });
    }
    let mut failure = None;
    for (name, term_count, per_round) in [
        ("dense", None, 6),
        ("eq_terms_2", Some(2), 11),
        (
            "eq_terms_9",
            Some(9),
            EQ_TERMS_NINE_ALLOCATIONS_PER_EXTRA_ROUND,
        ),
    ] {
        let limit = per_round * (sizes[1] - sizes[0])
            + RAYON_WORKER_ALLOWANCE.allocs * pool.current_num_threads();
        let mut minimum = [usize::MAX; 2];
        let mut maximum = [0; 2];
        let mut growth = 0;
        for sample in 0..30 {
            let mut counts = [0; 2];
            let order = if sample % 2 == 0 { [0, 1] } else { [1, 0] };
            for index in order {
                counts[index] = pool
                    .install(|| measure_round_allocations(sizes[index], term_count))
                    .map_err(|error| RunnerError::Core {
                        message: error.to_string(),
                    })?
                    .allocs;
                minimum[index] = minimum[index].min(counts[index]);
                maximum[index] = maximum[index].max(counts[index]);
            }
            growth = growth.max(counts[1].saturating_sub(counts[0]));
        }
        println!("chunk_product_allocation_growth/{name}/1 samples=30 log_t_small={} log_t_large={} visits_small={small_visits} visits_large={large_visits} count_small_min={} count_small_max={} count_large_min={} count_large_max={} growth={growth} limit={limit}", sizes[0], sizes[1], minimum[0], maximum[0], minimum[1], maximum[1]);
        if term_count == Some(9) {
            let middle_size = 16;
            let middle_visits =
                round_chunk_visits(middle_size).map_err(|error| RunnerError::Core {
                    message: error.to_string(),
                })?;
            let mut middle_min = usize::MAX;
            let mut middle_max = 0;
            for _ in 0..30 {
                let count = pool
                    .install(|| measure_round_allocations(middle_size, term_count))
                    .map_err(|error| RunnerError::Core {
                        message: error.to_string(),
                    })?
                    .allocs;
                middle_min = middle_min.min(count);
                middle_max = middle_max.max(count);
            }
            let [a, b, c] = fit_allocation_growth(
                [sizes[0], middle_size, sizes[1]],
                [small_visits, middle_visits, large_visits],
                [minimum[0], middle_min, minimum[1]],
            );
            println!("chunk_product_allocation_fit/{name}/1 samples=30 log_t_middle={middle_size} visits_middle={middle_visits} count_middle_min={middle_min} count_middle_max={middle_max} a={a} b={b} c={c}");
        }
        if growth > limit && failure.is_none() {
            failure = Some(RunnerError::Core {
                message: BenchError::Allocation {
                    log_t: sizes[1],
                    quantity: "allocation growth with chunk visits",
                    actual: growth,
                    limit,
                }
                .to_string(),
            });
        }
    }
    failure.map_or(Ok(()), Err)
}

#[expect(
    clippy::print_stdout,
    reason = "allocation acceptance reports its measured counts"
)]
fn check_allocations() -> Result<(), RunnerError> {
    let pool = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .map_err(|error| RunnerError::ThreadPool {
            message: error.to_string(),
        })?;
    let _ = pool.broadcast(|_| black_box(()));
    for log_t in [8, 14] {
        for (name, term_count) in [
            ("dense", None),
            ("eq_terms_2", Some(2)),
            ("eq_terms_9", Some(9)),
        ] {
            let kernel = allocation_bound(log_t, term_count.unwrap_or_default());
            let workers = pool.current_num_threads();
            let limit = AllocationAllowance {
                allocs: kernel.allocs + RAYON_WORKER_ALLOWANCE.allocs * workers,
                bytes: kernel.bytes + RAYON_WORKER_ALLOWANCE.bytes * workers,
            };
            let stats = pool
                .install(|| -> Result<_, BenchError> {
                    let stats = measure_round_allocations(log_t, term_count)?;
                    for (quantity, actual, limit) in [
                        ("allocation count", stats.allocs, limit.allocs),
                        ("peak bytes", stats.peak_bytes, limit.bytes),
                        ("live-after bytes", stats.final_bytes, limit.bytes),
                    ] {
                        if actual > limit {
                            return Err(BenchError::Allocation {
                                log_t,
                                quantity,
                                actual,
                                limit,
                            });
                        }
                    }
                    Ok(stats)
                })
                .map_err(|error| RunnerError::Core {
                    message: error.to_string(),
                })?;
            println!("chunk_product_allocation/{name}/{log_t}/{workers} allocs={} limit={} peak_bytes={} final_bytes={} byte_limit={} kernel_alloc_limit={} kernel_byte_limit={} runtime_alloc_allowance={} runtime_byte_allowance={} PASS", stats.allocs, limit.allocs, stats.peak_bytes, stats.final_bytes, limit.bytes, kernel.allocs, kernel.bytes, RAYON_WORKER_ALLOWANCE.allocs * workers, RAYON_WORKER_ALLOWANCE.bytes * workers);
        }
    }
    check_round_allocations_grow_with_rounds_not_chunk_visits(&pool)
}

#[expect(
    clippy::print_stdout,
    reason = "model and option records are benchmark output"
)]
fn main() -> Result<(), RunnerError> {
    check_allocations()?;
    let cases: Vec<_> = [
        ("chunk_product", Variant::Dense),
        ("chunk_product_dense_two", Variant::DenseTwo),
        ("chunk_product_eq_terms", Variant::EqTerms),
    ]
    .into_iter()
    .map(|(name, variant)| {
        let mut case = Case::core(
            name,
            variant,
            &["combined_weight", "first_four_rounds", "later_rounds"],
        );
        if matches!(variant, Variant::Dense) {
            case.threshold = Some(|_, threads| if threads == 12 { 69.0 / 9.6 } else { 69.0 });
        }
        case
    })
    .collect();
    let records = run_cases(
        &[SynthProfile::UniformDigits],
        &cases,
        |source| {
            Prepared::new(source).map_err(|error| RunnerError::Core {
                message: error.to_string(),
            })
        },
        |prepared, &variant, challenges, times| {
            let start = Clock::start();
            let (mut core, claim) =
                TimedCore::new(prepared, variant).map_err(|error| RunnerError::Core {
                    message: error.to_string(),
                })?;
            times.set(0, start.elapsed());
            let batch = core_rounds(&mut core, claim, challenges)?;
            times.set(1, batch.rounds);
            times.set(2, batch.finish);
            let start = Clock::start();
            let values = core
                .inner
                .final_values()
                .map_err(|error| RunnerError::Core {
                    message: error.to_string(),
                })?;
            let _ = black_box(&values);
            times.set(3, start.elapsed());
            Ok::<_, RunnerError>((core, values))
        },
        |(core, _), times| {
            times.set(4, core.combined);
            times.set(5, core.rounds[0]);
            times.set(6, core.rounds[1]);
        },
        |record, _, variant| {
            let log_t = record.log_t;
            let threads = record.threads;
            let name = variant.name();
            println!("chunk_product_split/{name}/{log_t}/{threads} validation_ns=outside_samples claim_preparation_ns=outside_samples combined_weight_ns={:.6} first_four_rounds_ns={:.6} later_rounds_ns={:.6} samples={} loaded_machine=true statistic=median parallel_evidence=false", record.phases[4].median, record.phases[5].median, record.phases[6].median, record.samples);
            if matches!(variant, Variant::Dense) {
                println!("chunk_product_model_split/{log_t}/{threads} combined_weight_ns={:.6} column_gathers_ns=separate_measurement rounds_minus_independent_gather_diagnostic_ns=unavailable difference_is_independent_phase=false model_combined_ns=5.1 model_gathers_ns=10.0 model_rounds_ns=40.1 loaded_machine=true statistic=median parallel_evidence=false", record.phases[4].median);
            }
        },
    )?;
    let mut wins = [None; 2];
    for (index, log_t) in [20, 22].into_iter().enumerate() {
        let dense = records.iter().find(|record| {
            record.log_t == log_t
                && record.threads == 1
                && record.id.starts_with("chunk_product_dense_two/")
        });
        let eq = records.iter().find(|record| {
            record.log_t == log_t
                && record.threads == 1
                && record.id.starts_with("chunk_product_eq_terms/")
        });
        if let (Some(dense), Some(eq)) = (dense, eq) {
            eq.print_comparison(&format!("chunk_product_options/{log_t}/1"), dense);
            wins[index] = Some(eq.improves_on(dense));
        }
    }
    let complete = wins.iter().all(Option::is_some);
    let winner = if complete && wins.iter().all(|win| *win == Some(true)) {
        "eq_terms"
    } else {
        "dense"
    };
    println!("chunk_product_two_term_default recommendation={winner} compared_both_sizes={complete} criterion=median_gain_exceeds_sum_spreads_at_both_sizes loaded_machine=true");
    println!("chunk_product_gather_note standalone_gathers_and_lazy_binds=separate_measurement_not_a_share_of_fused_rounds loaded_machine=true");
    run_gathers(
        &[SynthProfile::UniformDigits],
        &[(
            "chunk_product/standalone_gathers_and_lazy_binds",
            (0..5).collect(),
        )],
        |_| digit_points(),
    )
}
