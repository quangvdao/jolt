//! Run with `cargo bench -p jolt-rv64i-kernels --features test-utils --bench
//! chunk_product -- --log-t 20,22 --threads 1,12 --samples 3`.

pub mod support;

use std::hint::black_box;
use std::mem::size_of;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use jolt_field::{Accumulator, F128Accumulator, Field, F128};
use jolt_kernels::optimized::lazy_ra::{ChunkIndexSource, LazyFoldedRa, LazyRaError};
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_rv64i_kernels::chunk_product::{
    combined_weight, ChunkProductCore, ChunkProductError, ChunkWeight, ChunkWeightTerm, EqTerm,
};
use jolt_rv64i_kernels::par::{CycleChunks, ParError};
use jolt_rv64i_kernels::round::eq::eq_table;
use jolt_rv64i_kernels::source::{CycleSource, DigitColumns, SourceError, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthError, SynthProfile, SyntheticTrace};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use thiserror::Error;

use support::allocator::{AllocationAllowance, AllocationMeasurement, RAYON_WORKER_ALLOWANCE};
use support::{run_core, RunnerError};

#[derive(Debug, Error)]
enum BenchError {
    #[error(transparent)]
    Chunk(#[from] ChunkProductError),
    #[error(transparent)]
    Source(#[from] SourceError),
    #[error(transparent)]
    Par(#[from] ParError),
    #[error(transparent)]
    Lazy(#[from] LazyRaError),
    #[error(transparent)]
    Synth(#[from] SynthError),
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError<F128>),
    #[error("benchmark timing lock was poisoned")]
    TimingLock,
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

#[derive(Clone, Copy)]
enum PreparedClaims {
    Dense(F128),
    Equality([F128; 2]),
}

struct Prepared {
    trace: Arc<ValidatedTrace<SyntheticTrace>>,
    claims: PreparedClaims,
}

struct TimedCore {
    inner: ChunkProductCore,
    log_t: usize,
    setup_ns: [u128; 3],
    round_ns: [u128; 2],
    construct_ns: u128,
    finish_ns: u128,
    source_setup: bool,
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
        let start = Instant::now();
        let result = self.inner.prove_round(bind, round, claim);
        self.round_ns[usize::from(round >= 4)] += start.elapsed().as_nanos();
        result
    }

    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        let start = Instant::now();
        let result = self.inner.finish_rounds(bind);
        self.finish_ns += start.elapsed().as_nanos();
        result
    }
}

struct Record {
    variant: &'static str,
    log_t: usize,
    threads: usize,
    setup_ns: [u128; 3],
    round_ns: [u128; 2],
    construct_ns: u128,
    finish_ns: u128,
    source_setup: bool,
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

impl TimedCore {
    fn new(
        source: Arc<SyntheticTrace>,
        variant: Variant,
        cache: &Mutex<Option<Prepared>>,
    ) -> Result<(Self, F128), BenchError> {
        let construction = Instant::now();
        let log_t = source.cycles().ilog2() as usize;
        let mut cache = cache.lock().map_err(|_| BenchError::TimingLock)?;
        let start = Instant::now();
        let prepared = cache.as_ref().filter(|prepared| {
            Arc::ptr_eq(prepared.trace.source(), &source)
                && matches!(
                    (prepared.claims, variant),
                    (PreparedClaims::Dense(_), Variant::Dense)
                        | (
                            PreparedClaims::Equality(_),
                            Variant::DenseTwo | Variant::EqTerms
                        )
                )
        });
        let source_setup = prepared.is_none();
        let trace = if let Some(prepared) = prepared {
            Arc::clone(&prepared.trace)
        } else {
            Arc::new(ValidatedTrace::new(source)?)
        };
        let cached_claims = prepared.map(|prepared| prepared.claims);
        let validation_ns = if source_setup {
            start.elapsed().as_nanos()
        } else {
            0
        };
        let columns = DigitColumns::from_validated(Arc::clone(&trace), (0..5).collect())?;
        let points = digit_points();
        let tables: Vec<_> = if source_setup {
            points.iter().map(|point| eq_table(point, None)).collect()
        } else {
            Vec::new()
        };
        let geometry = CycleChunks::new(log_t, 0)?;
        let (weight, claim, claims, combined_ns, claim_ns) = match variant {
            Variant::Dense => {
                let terms = terms(log_t, variant);
                let start = Instant::now();
                let dense = combined_weight(log_t, &terms)?;
                let combined_ns = start.elapsed().as_nanos();
                let start = Instant::now();
                let claim = if let Some(PreparedClaims::Dense(claim)) = cached_claims {
                    claim
                } else {
                    dense
                        .par_chunks(geometry.chunk_len())
                        .enumerate()
                        .map(|(chunk, weights)| {
                            let mut sum = F128Accumulator::default();
                            for (offset, &weight) in weights.iter().enumerate() {
                                sum.fmadd(
                                    weight,
                                    column_product(
                                        &columns,
                                        &tables,
                                        chunk * geometry.chunk_len() + offset,
                                    ),
                                );
                            }
                            sum
                        })
                        .reduce(F128Accumulator::default, |mut left, right| {
                            left.merge(right);
                            left
                        })
                        .reduce()
                };
                let claim_ns = if source_setup {
                    start.elapsed().as_nanos()
                } else {
                    0
                };
                (
                    ChunkWeight::Dense(dense),
                    claim,
                    PreparedClaims::Dense(claim),
                    combined_ns,
                    claim_ns,
                )
            }
            Variant::DenseTwo | Variant::EqTerms => {
                let mut eq_terms = equality_terms::<2>(log_t);
                let start = Instant::now();
                let sums = if let Some(PreparedClaims::Equality(claims)) = cached_claims {
                    claims
                } else {
                    equality_claims(&columns, &tables, &eq_terms, geometry)?
                };
                let claim_ns = if source_setup {
                    start.elapsed().as_nanos()
                } else {
                    0
                };
                for (term, claim) in eq_terms.iter_mut().zip(sums) {
                    term.claim = claim;
                }
                let claim = sums[0] + sums[1];
                let (weight, combined_ns) = if matches!(variant, Variant::DenseTwo) {
                    let terms: Vec<_> = eq_terms
                        .into_iter()
                        .map(|term| ChunkWeightTerm::Eq {
                            coefficient: term.coefficient,
                            point: term.point,
                        })
                        .collect();
                    let start = Instant::now();
                    let dense = combined_weight(log_t, &terms)?;
                    (ChunkWeight::Dense(dense), start.elapsed().as_nanos())
                } else {
                    (ChunkWeight::EqTerms(eq_terms.into_iter().collect()), 0)
                };
                (
                    weight,
                    claim,
                    PreparedClaims::Equality(sums),
                    combined_ns,
                    claim_ns,
                )
            }
        };
        if source_setup {
            *cache = Some(Prepared { trace, claims });
        }
        drop(cache);
        let inner = ChunkProductCore::new(columns, points, weight)?;
        let construct_ns = construction.elapsed().as_nanos();
        Ok((
            Self {
                inner,
                log_t,
                setup_ns: [validation_ns, combined_ns, claim_ns],
                round_ns: [0; 2],
                construct_ns,
                finish_ns: 0,
                source_setup,
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

#[derive(Clone)]
struct CompactDigits {
    digits: Vec<u8>,
    bits: [usize; 7],
    d: usize,
    cycles: usize,
}

impl CompactDigits {
    fn new(columns: &DigitColumns<SyntheticTrace>) -> Result<Self, BenchError> {
        let cycles = columns.cycles();
        let d = columns.num_polys();
        let bits = std::array::from_fn(|column| {
            if column < d {
                columns.source().bits(columns.columns()[column])
            } else {
                0
            }
        });
        let geometry = CycleChunks::new(cycles.ilog2() as usize, 0)?;
        let mut digits = jolt_utils::unsafe_allocate_zero_vec(cycles * d);
        let missing = digits
            .par_chunks_mut(geometry.chunk_len() * d)
            .enumerate()
            .find_map_first(|(chunk, digits)| {
                let start = chunk * geometry.chunk_len();
                for (offset, row) in digits.chunks_exact_mut(d).enumerate() {
                    for (column, digit) in row.iter_mut().enumerate() {
                        let cycle = start + offset;
                        match columns.index(column, cycle) {
                            Some(index) => *digit = index as u8,
                            None => return Some(ChunkProductError::MissingDigit { column, cycle }),
                        }
                    }
                }
                None
            });
        if let Some(error) = missing {
            return Err(error.into());
        }
        Ok(Self {
            digits,
            bits,
            d,
            cycles,
        })
    }
}

impl ChunkIndexSource for CompactDigits {
    fn num_polys(&self) -> usize {
        self.d
    }
    fn cycles(&self) -> usize {
        self.cycles
    }
    #[inline]
    fn index(&self, column: usize, cycle: usize) -> Option<usize> {
        Some(usize::from(self.digits[cycle * self.d + column]))
    }
    fn index_bound(&self, column: usize) -> Option<usize> {
        Some(1 << self.bits[column])
    }
}

fn seeded_challenges() -> [F128; 32] {
    let mut rng = ChaCha20Rng::seed_from_u64(0x726f_756e_6473);
    std::array::from_fn(|_| F128::random(&mut rng))
}

fn measure_gathers(columns: CompactDigits, log_t: usize) -> Result<u128, BenchError> {
    let tables = digit_points()
        .iter()
        .map(|point| eq_table(point, None))
        .collect();
    let mut columns = LazyFoldedRa::try_new(tables, columns)?;
    let challenges = seeded_challenges();
    let mut elapsed = 0;
    for (round, &challenge) in challenges.iter().take(log_t.min(4)).enumerate() {
        let geometry = CycleChunks::new(log_t, round)?;
        let start = Instant::now();
        let checksum = (0..geometry.len() / geometry.chunk_len())
            .into_par_iter()
            .map(|chunk| {
                let start = chunk * geometry.chunk_len() / 2;
                let end = start + geometry.chunk_len() / 2;
                let mut values = [(F128::from_raw(0), F128::from_raw(0)); 5];
                let mut sum = F128::from_raw(0);
                for pair in start..end {
                    columns.lo_hi_all(pair, &mut values);
                    for &(low, high) in &values {
                        sum += low + high;
                    }
                }
                sum
            })
            .reduce(|| F128::from_raw(0), |left, right| left + right);
        let _ = black_box(checksum);
        columns.bind(challenge);
        elapsed += start.elapsed().as_nanos();
    }
    Ok(elapsed)
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        samples[middle - 1].midpoint(samples[middle])
    } else {
        samples[middle]
    }
}

#[expect(clippy::print_stdout, reason = "phase splits are benchmark output")]
fn report(records: &[Record]) -> Result<(), RunnerError> {
    for (index, record) in records.iter().enumerate() {
        if records[..index].iter().any(|prior| {
            prior.variant == record.variant
                && prior.log_t == record.log_t
                && prior.threads == record.threads
        }) {
            continue;
        }
        let samples: Vec<_> = records
            .iter()
            .filter(|sample| {
                sample.variant == record.variant
                    && sample.log_t == record.log_t
                    && sample.threads == record.threads
            })
            .collect();
        let cycles = (1_usize << record.log_t) as f64;
        let statistic = if record.threads == 1 {
            "minimum"
        } else {
            "median"
        };
        let summarize = |values: Vec<f64>| {
            if record.threads == 1 {
                values.into_iter().fold(f64::INFINITY, f64::min)
            } else {
                median(values)
            }
        };
        let setup: [f64; 3] = std::array::from_fn(|phase| {
            summarize(
                samples
                    .iter()
                    .map(|sample| sample.setup_ns[phase] as f64 / cycles)
                    .collect(),
            )
        });
        let rounds: [f64; 2] = std::array::from_fn(|phase| {
            summarize(
                samples
                    .iter()
                    .map(|sample| sample.round_ns[phase] as f64 / cycles)
                    .collect(),
            )
        });
        let cold = samples.iter().find(|sample| sample.source_setup);
        let cold_validation = cold.map_or(0.0, |sample| sample.setup_ns[0] as f64 / cycles);
        let cold_claim = cold.map_or(0.0, |sample| sample.setup_ns[2] as f64 / cycles);
        let cold_total = cold.map_or(0.0, |sample| {
            (sample.construct_ns + sample.round_ns.iter().sum::<u128>() + sample.finish_ns) as f64
                / cycles
        });
        println!("chunk_product_source_setup/{}/{}/{} source_setup_cache=true first_use_validation_ns={cold_validation:.6} first_use_claim_preparation_ns={cold_claim:.6} cold_total_without_extraction_ns={cold_total:.6} loaded_machine=true statistic={statistic} parallel_evidence=false", record.variant, record.log_t, record.threads);
        println!("chunk_product_split/{}/{}/{} validation_ns={:.6} combined_weight_ns={:.6} claim_preparation_ns={:.6} first_four_rounds_ns={:.6} later_rounds_ns={:.6} loaded_machine=true statistic={statistic} parallel_evidence=false", record.variant, record.log_t, record.threads, setup[0], setup[1], setup[2], rounds[0], rounds[1]);
        if record.variant != "dense" {
            continue;
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(record.threads)
            .build()
            .map_err(|error| RunnerError::ThreadPool {
                message: error.to_string(),
            })?;
        let _ = pool.broadcast(|_| black_box(()));
        let gathers = pool
            .install(|| -> Result<_, BenchError> {
                let source = Arc::new(SyntheticTrace::new(
                    SynthProfile::UniformDigits,
                    record.log_t,
                    1 << 20,
                    0x5eed,
                )?);
                let trace = Arc::new(ValidatedTrace::new(source)?);
                let columns = DigitColumns::from_validated(trace, (0..5).collect())?;
                let compact = CompactDigits::new(&columns)?;
                let mut measured = Vec::with_capacity(samples.len());
                for _ in 0..samples.len() {
                    measured.push(measure_gathers(compact.clone(), record.log_t)? as f64 / cycles);
                }
                Ok(summarize(measured))
            })
            .map_err(|error| RunnerError::Core {
                message: error.to_string(),
            })?;
        println!("chunk_product_model_split/{}/{} combined_weight_ns={:.6} column_gathers_ns={gathers:.6} rounds_minus_independent_gather_diagnostic_ns={:.6} difference_is_independent_phase=false model_combined_ns=5.1 model_gathers_ns=10.0 model_rounds_ns=40.1 threshold_ns={:.6} loaded_machine=true statistic={statistic} parallel_evidence=false", record.log_t, record.threads, setup[1], rounds.iter().sum::<f64>() - gathers, if record.threads == 12 { 69.0 / 9.6 } else { 69.0 });
    }
    Ok(())
}

struct OptionStats {
    min: f64,
    median: f64,
    max: f64,
}

impl OptionStats {
    fn new(samples: &[f64]) -> Self {
        Self {
            min: samples.iter().copied().reduce(f64::min).unwrap_or_default(),
            median: median(samples.to_vec()),
            max: samples.iter().copied().reduce(f64::max).unwrap_or_default(),
        }
    }

    fn spread(&self) -> f64 {
        self.max - self.min
    }
}

fn option_sample(
    source: &Arc<SyntheticTrace>,
    variant: Variant,
    cache: &Mutex<Option<Prepared>>,
    challenges: &[F128],
) -> Result<f64, BenchError> {
    let start = Instant::now();
    let (mut core, mut claim) = TimedCore::new(Arc::clone(source), variant, cache)?;
    let mut bind = None;
    for (round, &challenge) in challenges.iter().enumerate() {
        let message = core.prove_round(bind, round, claim)?;
        claim = message.evaluate(challenge);
        bind = Some(challenge);
        let _ = black_box(message);
    }
    if let Some(challenge) = bind {
        core.finish_rounds(challenge)?;
    }
    let values = core.inner.final_values()?;
    let _ = black_box((claim, values));
    Ok(start.elapsed().as_nanos() as f64 / source.cycles() as f64)
}

#[expect(
    clippy::print_stdout,
    reason = "matched option statistics and default recommendation are benchmark output"
)]
fn compare_options(records: &[Record]) -> Result<(), RunnerError> {
    let pool = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .map_err(|error| RunnerError::ThreadPool {
            message: error.to_string(),
        })?;
    let _ = pool.broadcast(|_| black_box(()));
    let mut wins = [None; 2];
    for (index, log_t) in [20, 22].into_iter().enumerate() {
        let Some(record) = records
            .iter()
            .find(|record| record.log_t == log_t && record.threads == 1)
        else {
            continue;
        };
        let samples = records
            .iter()
            .filter(|sample| {
                sample.log_t == log_t && sample.threads == 1 && sample.variant == record.variant
            })
            .count()
            .max(5);
        let (dense, eq) = pool
            .install(|| -> Result<_, BenchError> {
                let source = Arc::new(SyntheticTrace::new(
                    SynthProfile::UniformDigits,
                    log_t,
                    1 << 20,
                    0x5eed,
                )?);
                let cache = Mutex::new(None);
                let challenges = seeded_challenges();
                let challenges = &challenges[..log_t];
                let _ = option_sample(&source, Variant::DenseTwo, &cache, challenges)?;
                let _ = option_sample(&source, Variant::EqTerms, &cache, challenges)?;
                let mut dense = Vec::with_capacity(samples);
                let mut eq = Vec::with_capacity(samples);
                for sample in 0..samples {
                    let order = if sample.is_multiple_of(2) {
                        [Variant::DenseTwo, Variant::EqTerms]
                    } else {
                        [Variant::EqTerms, Variant::DenseTwo]
                    };
                    for variant in order {
                        let timing = option_sample(&source, variant, &cache, challenges)?;
                        if matches!(variant, Variant::DenseTwo) {
                            dense.push(timing);
                        } else {
                            eq.push(timing);
                        }
                    }
                }
                Ok((OptionStats::new(&dense), OptionStats::new(&eq)))
            })
            .map_err(|error| RunnerError::Core {
                message: error.to_string(),
            })?;
        let gain = dense.median - eq.median;
        let spread = dense.spread() + eq.spread();
        let wins_here = gain > spread;
        wins[index] = Some(wins_here);
        println!("chunk_product_options/{log_t}/1 samples={samples} alternating=true same_input=true source_validation_timed=false claim_preparation_timed=false dense_weight_construction_timed=true core_construction_timed=true dense_two_min_ns={:.6} dense_two_median_ns={:.6} dense_two_max_ns={:.6} eq_terms_two_min_ns={:.6} eq_terms_two_median_ns={:.6} eq_terms_two_max_ns={:.6} median_gain_ns={gain:.6} sum_spreads_ns={spread:.6} eq_terms_wins={wins_here} loaded_machine=true", dense.min, dense.median, dense.max, eq.min, eq.median, eq.max);
    }
    let complete = wins.iter().all(Option::is_some);
    let winner = if complete && wins.iter().all(|win| *win == Some(true)) {
        "eq_terms"
    } else {
        "dense"
    };
    println!("chunk_product_two_term_default recommendation={winner} compared_both_sizes={complete} criterion=median_gain_exceeds_sum_spreads_at_both_sizes loaded_machine=true");
    Ok(())
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
                        let (core, claim) = TimedCore::new(source, variant, &Mutex::new(None))?;
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
                    let stats = measurement.finish();
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
    Ok(())
}

fn main() -> Result<(), RunnerError> {
    check_allocations()?;
    let records = Mutex::new(Vec::new());
    for variant in [Variant::Dense, Variant::EqTerms] {
        let name = match variant {
            Variant::Dense => "chunk_product",
            Variant::DenseTwo => "chunk_product_dense_two",
            Variant::EqTerms => "chunk_product_eq_terms",
        };
        let cache = Mutex::new(None);
        run_core(
            name,
            &[SynthProfile::UniformDigits],
            |source| TimedCore::new(source, variant, &cache),
            |core, _point| {
                let values = core.inner.final_values()?;
                records
                    .lock()
                    .map_err(|_| BenchError::TimingLock)?
                    .push(Record {
                        variant: variant.name(),
                        log_t: core.log_t,
                        threads: rayon::current_num_threads(),
                        setup_ns: core.setup_ns,
                        round_ns: core.round_ns,
                        construct_ns: core.construct_ns,
                        finish_ns: core.finish_ns,
                        source_setup: core.source_setup,
                    });
                Ok::<_, BenchError>(values)
            },
        )?;
    }
    let records = records.into_inner().map_err(|_| RunnerError::Core {
        message: BenchError::TimingLock.to_string(),
    })?;
    report(&records)?;
    compare_options(&records)
}
