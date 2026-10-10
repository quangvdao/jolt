//! Run with `cargo bench -p jolt-rv64i-kernels --features test-utils --bench
//! chunk_product -- --log-t 20,22 --threads 1,12 --samples 3`.

pub mod support;

use std::hint::black_box;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use jolt_field::{Accumulator, F128Accumulator, Field, F128};
use jolt_kernels::optimized::lazy_ra::{LazyFoldedRa, LazyRaError};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_kernels::chunk_product::{
    combined_weight, ChunkProductCore, ChunkProductError, ChunkWeight,
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

use support::allocator::AllocationMeasurement;
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
    #[error("core allocated {actual} times at log_t={log_t}, above limit {limit}")]
    Allocation {
        log_t: usize,
        actual: usize,
        limit: usize,
    },
}

#[derive(Clone, Copy)]
enum Variant {
    Dense,
    EqTerms,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::EqTerms => "eq_terms",
        }
    }
}

struct Prepared {
    trace: Arc<ValidatedTrace<SyntheticTrace>>,
    claims: Vec<F128>,
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

fn terms(log_t: usize, variant: Variant) -> Vec<ChunkWeight> {
    let count = match variant {
        Variant::Dense => 5,
        Variant::EqTerms => 2,
    };
    (0..count)
        .map(|term| {
            let coefficient = F128::from_raw(0x831 + term as u128);
            let point = (0..log_t)
                .map(|bit| {
                    if term == 4 {
                        F128::from_raw((bit % 2) as u128)
                    } else {
                        F128::from_raw(0x241 + (term * 37 + bit * 11) as u128)
                    }
                })
                .collect();
            if term == 3 {
                ChunkWeight::Next { coefficient, point }
            } else {
                ChunkWeight::Eq { coefficient, point }
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
        let prepared = cache
            .as_ref()
            .filter(|prepared| Arc::ptr_eq(prepared.trace.source(), &source));
        let source_setup = prepared.is_none();
        let trace = if let Some(prepared) = prepared {
            Arc::clone(&prepared.trace)
        } else {
            Arc::new(ValidatedTrace::new(source)?)
        };
        let cached_claims = prepared.map(|prepared| prepared.claims.clone());
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
        let terms = terms(log_t, variant);
        let start = Instant::now();
        let dense = match variant {
            Variant::Dense => Some(combined_weight(log_t, &terms)?),
            Variant::EqTerms => None,
        };
        let combined_ns = start.elapsed().as_nanos();
        let start = Instant::now();
        let geometry = CycleChunks::new(log_t, 0)?;
        let (weight, claim, claims) = if let Some(dense) = dense {
            let claim = if let Some(claims) = &cached_claims {
                claims[0]
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
            (ChunkWeight::Dense(dense), claim, vec![claim])
        } else {
            let mut eq_terms = Vec::with_capacity(2);
            let mut halves = Vec::with_capacity(2);
            for term in terms {
                if let ChunkWeight::Eq { coefficient, point } = term {
                    if source_setup {
                        let (low, high) = geometry.split_point(&point)?;
                        halves.push((eq_table(low, None), eq_table(high, Some(coefficient))));
                    }
                    eq_terms.push((coefficient, point, F128::from_raw(0)));
                }
            }
            let sums = if let Some(claims) = &cached_claims {
                [claims[0], claims[1]]
            } else {
                halves[0]
                    .1
                    .par_chunks(geometry.chunk_len() / geometry.block_len())
                    .enumerate()
                    .map(|(chunk, highs)| {
                        let mut outer = [F128Accumulator::default(); 2];
                        for offset in 0..highs.len() {
                            let block =
                                chunk * (geometry.chunk_len() / geometry.block_len()) + offset;
                            let mut inner = [F128Accumulator::default(); 2];
                            for index in 0..geometry.block_len() {
                                let product = column_product(
                                    &columns,
                                    &tables,
                                    block * geometry.block_len() + index,
                                );
                                for (sum, (low, _high)) in inner.iter_mut().zip(&halves) {
                                    sum.fmadd(low[index], product);
                                }
                            }
                            for ((sum, inner), (_low, high)) in
                                outer.iter_mut().zip(inner).zip(&halves)
                            {
                                sum.fmadd(high[block], inner.reduce());
                            }
                        }
                        outer
                    })
                    .reduce(
                        || [F128Accumulator::default(); 2],
                        |mut left, right| {
                            for (left, right) in left.iter_mut().zip(right) {
                                left.merge(right);
                            }
                            left
                        },
                    )
                    .map(Accumulator::reduce)
            };
            for (term, sum) in eq_terms.iter_mut().zip(sums) {
                term.2 = sum;
            }
            let claim = sums[0] + sums[1];
            (ChunkWeight::EqTerms(eq_terms), claim, sums.to_vec())
        };
        let claim_ns = if source_setup {
            start.elapsed().as_nanos()
        } else {
            0
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
    let mut terms = Vec::with_capacity(9);
    let mut halves = Vec::with_capacity(9);
    for term in 0..9 {
        let coefficient = F128::from_raw(0x831 + term as u128);
        let point: Vec<_> = (0..log_t)
            .map(|bit| F128::from_raw(0x241 + (term * 37 + bit * 11) as u128))
            .collect();
        let (low, high) = geometry.split_point(&point)?;
        halves.push((eq_table(low, None), eq_table(high, Some(coefficient))));
        terms.push((coefficient, point, F128::from_raw(0)));
    }
    let sums = halves[0]
        .1
        .par_chunks(geometry.chunk_len() / geometry.block_len())
        .enumerate()
        .map(|(chunk, highs)| {
            let mut outer = [F128Accumulator::default(); 9];
            for offset in 0..highs.len() {
                let block = chunk * (geometry.chunk_len() / geometry.block_len()) + offset;
                let mut inner = [F128Accumulator::default(); 9];
                for index in 0..geometry.block_len() {
                    let product =
                        column_product(&columns, &tables, block * geometry.block_len() + index);
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
            || [F128Accumulator::default(); 9],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    left.merge(right);
                }
                left
            },
        )
        .map(Accumulator::reduce);
    for (term, claim) in terms.iter_mut().zip(sums) {
        term.2 = claim;
    }
    let claim = sums
        .iter()
        .copied()
        .fold(F128::from_raw(0), |sum, claim| sum + claim);
    let core = ChunkProductCore::new(columns, points, ChunkWeight::EqTerms(terms))?;
    Ok((core, claim))
}

fn seeded_challenges() -> [F128; 32] {
    let mut rng = ChaCha20Rng::seed_from_u64(0x726f_756e_6473);
    std::array::from_fn(|_| F128::random(&mut rng))
}

fn measure_gathers(
    columns: DigitColumns<SyntheticTrace>,
    log_t: usize,
) -> Result<u128, BenchError> {
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
                let mut measured = Vec::with_capacity(samples.len());
                for _ in 0..samples.len() {
                    let columns =
                        DigitColumns::from_validated(Arc::clone(&trace), (0..5).collect())?;
                    measured.push(measure_gathers(columns, record.log_t)? as f64 / cycles);
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
            let limit = match term_count {
                Some(terms) if terms > 7 => 2 * (terms + 1) * log_t + 64,
                _ => 16 * log_t + 64,
            };
            let count = pool
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
                    let count = measurement.finish().allocs;
                    if count > limit {
                        return Err(BenchError::Allocation {
                            log_t,
                            actual: count,
                            limit,
                        });
                    }
                    Ok(count)
                })
                .map_err(|error| RunnerError::Core {
                    message: error.to_string(),
                })?;
            println!("chunk_product_allocation/{name}/{log_t}/1 allocs={count} limit={limit} PASS");
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
    report(&records.into_inner().map_err(|_| RunnerError::Core {
        message: BenchError::TimingLock.to_string(),
    })?)
}
