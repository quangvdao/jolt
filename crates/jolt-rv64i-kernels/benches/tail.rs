//! Run `cargo bench -p jolt-rv64i-kernels --features test-utils --bench tail
//! -- --log-t 20,22 --threads 1,12 --samples 3`. Records are loaded-machine evidence.

pub mod support;

use jolt_field::{Accumulator, F128Accumulator, F128};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_kernels::chunk_product::{
    combined_weight, ChunkProductCore, ChunkProductError, ChunkWeight, ChunkWeightTerm,
};
use jolt_rv64i_kernels::column_pass::{column_pass, ColumnPassError};
use jolt_rv64i_kernels::par::{CycleChunks, ParError};
use jolt_rv64i_kernels::reduction::{
    g_pass_digits, ColumnMap, ReductionCore, ReductionError, ReductionLeg,
};
use jolt_rv64i_kernels::round::eq::eq_table;
use jolt_rv64i_kernels::source::{CycleSource, DigitColumns, SourceError, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, ClearSumcheckRecorder, MemberFinish, MemberRound,
    ProveRounds, ProvedBatch, RoundScheduler, SumcheckError,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rayon::prelude::*;
use std::hint::black_box;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use support::{run_core, RunnerError};
use thiserror::Error;

const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);

#[derive(Debug, Error)]
enum BenchError {
    #[error(transparent)]
    Source(#[from] SourceError),
    #[error(transparent)]
    Chunk(#[from] ChunkProductError),
    #[error(transparent)]
    Reduction(#[from] ReductionError),
    #[error(transparent)]
    Column(#[from] ColumnPassError),
    #[error(transparent)]
    Par(#[from] ParError),
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError<F128>),
    #[error("benchmark fixture or telemetry lock was poisoned")]
    Lock,
    #[error("batch must complete before extraction")]
    Unfinished,
}

fn weights() -> Vec<Vec<F128>> {
    (0..3)
        .map(|support| {
            (0..256)
                .map(|y| {
                    let active = match support {
                        0 => (64..=228).contains(&y),
                        1 => y < 64 || (139..=230).contains(&y),
                        _ => y < 64,
                    };
                    if active {
                        F128::from_raw(((y as u128 + 1) << 64) | (support as u128 + 7))
                    } else {
                        ZERO
                    }
                })
                .collect()
        })
        .collect()
}

fn points(member: usize) -> Vec<Vec<F128>> {
    (0..5)
        .map(|column| {
            (0..4)
                .map(|bit| F128::from_raw(0x1325 + (member * 157 + column * 31 + bit * 7) as u128))
                .collect()
        })
        .collect()
}

fn terms(log_t: usize, member: usize) -> Vec<ChunkWeightTerm> {
    (0..if member == 0 { 5 } else { 2 })
        .map(|term| {
            let coefficient = F128::from_raw(0x831 + (member * 17 + term) as u128);
            let point = (0..log_t)
                .map(|bit| {
                    if member == 0 && term == 0 {
                        F128::from_raw((bit & 1) as u128)
                    } else {
                        F128::from_raw(0x241 + (member * 71 + term * 37 + bit * 11) as u128)
                    }
                })
                .collect();
            if member == 0 && term == 4 {
                ChunkWeightTerm::Next { coefficient, point }
            } else {
                ChunkWeightTerm::Eq { coefficient, point }
            }
        })
        .collect()
}

fn chunk_claim(
    source: &SyntheticTrace,
    member: usize,
    weight: &[F128],
    geometry: CycleChunks,
) -> F128 {
    let tables: Vec<_> = points(member).iter().map(|a| eq_table(a, None)).collect();
    weight
        .par_chunks(geometry.chunk_len())
        .enumerate()
        .map(|(chunk, weights)| {
            let mut sum = F128Accumulator::default();
            for (offset, &w) in weights.iter().enumerate() {
                let cycle = chunk * geometry.chunk_len() + offset;
                let product = tables
                    .iter()
                    .enumerate()
                    .map(|(c, table)| {
                        source
                            .digit(5 * member + c, cycle)
                            .map_or(ZERO, |digit| table[digit])
                    })
                    .product();
                sum.fmadd(w, product);
            }
            sum
        })
        .reduce(F128Accumulator::default, |mut left, right| {
            left.merge(right);
            left
        })
        .reduce()
}

struct Prepared {
    trace: Arc<ValidatedTrace<SyntheticTrace>>,
    claims: [F128; 2],
}

#[derive(Default)]
struct TimingScheduler {
    members: [u128; 3],
    finish: u128,
}

impl RoundScheduler<F128> for TimingScheduler {
    fn batch_prove_round(
        &mut self,
        work: &mut [MemberRound<'_, F128>],
    ) -> Result<(), SumcheckError<F128>> {
        for item in work {
            let start = Instant::now();
            item.run()?;
            self.members[item.index] += start.elapsed().as_nanos();
        }
        Ok(())
    }
    fn batch_finish_rounds(
        &mut self,
        work: &mut [MemberFinish<'_, F128>],
    ) -> Result<(), SumcheckError<F128>> {
        let start = Instant::now();
        for item in work {
            item.run()?;
        }
        self.finish = start.elapsed().as_nanos();
        Ok(())
    }
}

struct Record {
    log_t: usize,
    threads: usize,
    phases: [u128; 4],
    setup: [u128; 6],
    members: [u128; 3],
    fixture_setup: u128,
    column: u128,
}

struct Tail {
    source: Arc<SyntheticTrace>,
    chunks: [ChunkProductCore; 2],
    reduction: ReductionCore,
    prelude: BatchPrelude<F128>,
    proved: Option<ProvedBatch<F128>>,
    scheduler: TimingScheduler,
    construction: u128,
    batch: u128,
    setup: [u128; 6],
    fixture_setup: u128,
}

impl Tail {
    fn new(
        source: Arc<SyntheticTrace>,
        cache: &Mutex<Option<Prepared>>,
        map: &[ColumnMap],
        weights: &[Vec<F128>],
    ) -> Result<Self, BenchError> {
        let construction = Instant::now();
        let log_t = source.cycles().ilog2() as usize;
        let geometry = CycleChunks::new(log_t, 0)?;
        let start = Instant::now();
        let mut cache = cache.lock().map_err(|_| BenchError::Lock)?;
        let cached = cache
            .as_ref()
            .filter(|prepared| Arc::ptr_eq(prepared.trace.source(), &source));
        let trace = if let Some(prepared) = cached {
            Arc::clone(&prepared.trace)
        } else {
            Arc::new(ValidatedTrace::new(Arc::clone(&source))?)
        };
        let cached_claims = cached.map(|prepared| prepared.claims);
        let mut fixture_setup = start.elapsed().as_nanos();
        let mut setup = [0; 6];
        let start = Instant::now();
        let tables = g_pass_digits(&trace, map, weights)?;
        setup[0] = start.elapsed().as_nanos();
        let start = Instant::now();
        let first = combined_weight(log_t, &terms(log_t, 0))?;
        setup[1] = start.elapsed().as_nanos();
        let start = Instant::now();
        let second = combined_weight(log_t, &terms(log_t, 1))?;
        setup[2] = start.elapsed().as_nanos();
        let start = Instant::now();
        let claims = cached_claims.unwrap_or_else(|| {
            [
                chunk_claim(&source, 0, &first, geometry),
                chunk_claim(&source, 1, &second, geometry),
            ]
        });
        fixture_setup += start.elapsed().as_nanos();
        *cache = Some(Prepared {
            trace: Arc::clone(&trace),
            claims,
        });
        drop(cache);
        let mut reduction_claim = ZERO;
        let legs = (0..3)
            .map(|table| {
                // Boolean input points select an existing claim without a preparation pass.
                let vertex = ((table + 1) * 0x39a5) & (source.cycles() - 1);
                let point = (0..log_t)
                    .map(|bit| if vertex & (1 << bit) == 0 { ZERO } else { ONE })
                    .collect();
                let coefficient = F128::from_raw(table as u128 + 1);
                let claim = tables[table][vertex];
                reduction_claim += coefficient * claim;
                ReductionLeg {
                    table,
                    point,
                    coefficient,
                    claim,
                }
            })
            .collect();
        let start = Instant::now();
        let a = ChunkProductCore::new(
            DigitColumns::from_validated(Arc::clone(&trace), (0..5).collect())?,
            points(0),
            ChunkWeight::Dense(first),
        )?;
        setup[3] = start.elapsed().as_nanos();
        let start = Instant::now();
        let b = ChunkProductCore::new(
            DigitColumns::from_validated(trace, (5..10).collect())?,
            points(1),
            ChunkWeight::Dense(second),
        )?;
        setup[4] = start.elapsed().as_nanos();
        let start = Instant::now();
        let reduction = ReductionCore::new(tables, legs)?;
        setup[5] = start.elapsed().as_nanos();
        let prelude = BatchPrelude::try_new(
            claims
                .into_iter()
                .chain([reduction_claim])
                .enumerate()
                .map(|(member, input_claim)| BatchMember {
                    input_claim,
                    coefficient: F128::from_raw(71 + member as u128 * 13),
                    rounds: log_t,
                    offset: 0,
                })
                .collect(),
            log_t,
            6,
        )?;
        Ok(Self {
            source,
            chunks: [a, b],
            reduction,
            prelude,
            proved: None,
            scheduler: TimingScheduler::default(),
            construction: construction.elapsed().as_nanos(),
            batch: 0,
            setup,
            fixture_setup,
        })
    }

    fn extract(&self, records: &Mutex<Vec<Record>>) -> Result<[F128; 256], BenchError> {
        let proved = self.proved.as_ref().ok_or(BenchError::Unfinished)?;
        let start = Instant::now();
        let _ = black_box(self.chunks[0].final_values()?);
        let _ = black_box(self.chunks[1].final_values()?);
        let _ = black_box(self.reduction.final_values()?);
        let column_start = Instant::now();
        let columns = column_pass(self.source.rows(), &proved.challenges)?;
        let column = column_start.elapsed().as_nanos();
        let extraction = start.elapsed().as_nanos();
        records.lock().map_err(|_| BenchError::Lock)?.push(Record {
            log_t: self.prelude.max_num_vars,
            threads: rayon::current_num_threads(),
            phases: [
                self.construction,
                self.batch.saturating_sub(self.scheduler.finish),
                self.scheduler.finish,
                extraction,
            ],
            setup: self.setup,
            members: self.scheduler.members,
            fixture_setup: self.fixture_setup,
            column,
        });
        Ok(columns)
    }
}

// run_core's single timed invocation carries the complete batch. Its raw rounds
// include terminal binds; tail_phases separates them using TimingScheduler.
impl ProveRounds<F128> for Tail {
    fn num_rounds(&self) -> usize {
        1
    }
    fn prove_round(
        &mut self,
        _: Option<F128>,
        _: usize,
        _: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        let [a, b] = &mut self.chunks;
        let mut transcript = Blake2bTranscript::new(b"rv64i-tail-benchmark-seed");
        let mut recorder = ClearSumcheckRecorder::<F128>::new();
        let start = Instant::now();
        self.proved = Some(prove_batch(
            &self.prelude,
            &mut [a, b, &mut self.reduction],
            &mut self.scheduler,
            &mut recorder,
            &mut transcript,
        )?);
        self.batch = start.elapsed().as_nanos();
        Ok(UnivariatePoly::new(vec![ZERO, ZERO]))
    }
    fn finish_rounds(&mut self, _: F128) -> Result<(), SumcheckError<F128>> {
        Ok(())
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        values[middle - 1].midpoint(values[middle])
    } else {
        values[middle]
    }
}

#[expect(
    clippy::print_stdout,
    reason = "phase and model records are benchmark output"
)]
fn report(records: &[Record]) {
    for (index, record) in records.iter().enumerate() {
        if records[..index]
            .iter()
            .any(|r| r.log_t == record.log_t && r.threads == record.threads)
        {
            continue;
        }
        let samples: Vec<_> = records
            .iter()
            .filter(|r| r.log_t == record.log_t && r.threads == record.threads)
            .collect();
        let cycles = (1usize << record.log_t) as f64;
        let ns = |select: fn(&Record) -> u128| {
            median(samples.iter().map(|r| select(r) as f64 / cycles).collect())
        };
        let phases: [f64; 4] = std::array::from_fn(|i| {
            median(
                samples
                    .iter()
                    .map(|r| r.phases[i] as f64 / cycles)
                    .collect(),
            )
        });
        let setup: [f64; 6] = std::array::from_fn(|i| {
            median(samples.iter().map(|r| r.setup[i] as f64 / cycles).collect())
        });
        let members: [f64; 3] = std::array::from_fn(|i| {
            median(
                samples
                    .iter()
                    .map(|r| r.members[i] as f64 / cycles)
                    .collect(),
            )
        });
        let scale = record.threads as f64;
        println!("tail_phases/local/{}/{} construct_ns={:.6} rounds_ns={:.6} finish_ns={:.6} extract_ns={:.6} fixture_setup_ns={:.6} threshold_ns={:.6} model_construct_ns={:.6} model_rounds_ns={:.6} model_finish_ns=0 model_extract_ns={:.6} samples={} loaded_machine=true", record.log_t, record.threads, phases[0], phases[1], phases[2], phases[3], ns(|r| r.fixture_setup), if record.threads == 1 { 192.0 } else { 20.0 }, 23.3 / scale, 109.0 / scale, 21.0 / scale, samples.len());
        for (name, measured, model) in [
            ("g_pass_digits", setup[0], 15.3),
            ("combined_weight_five", setup[1], 5.1),
            ("combined_weight_two", setup[2], 2.9),
            ("chunk_zero_gathers_and_rounds", members[0], 50.1),
            ("chunk_one_gathers_and_rounds", members[1], 50.1),
            ("reduction_rounds", members[2], 8.8),
            ("column_pass", ns(|r| r.column), 21.0),
        ] {
            println!("tail_part/{name}/local/{}/{} ns={measured:.6} model_ns={:.6} over_25_percent={} loaded_machine=true", record.log_t, record.threads, model / scale, measured > 1.25 * model / scale);
        }
        println!("tail_gather_boundary/local/{}/{} column_gathers_ns=unavailable model_gathers_ns={:.6} model_product_rounds_ns={:.6} reason=fused_library_pair_loop compact_digit_construction_zero_ns={:.6} compact_digit_construction_one_ns={:.6} reduction_construction_ns={:.6}", record.log_t, record.threads, 10.0 / scale, 40.1 / scale, setup[3], setup[4], setup[5]);
    }
}

#[expect(
    clippy::print_stdout,
    reason = "the runner bridge timing boundary is part of benchmark output"
)]
fn main() -> Result<(), RunnerError> {
    println!("tail_driver_note raw_runner_rounds_include_batch_finish=true raw_runner_finish_is_noop=true actual_four_phases=tail_phases challenge_source=seeded_blake2b_batch column_gathers=fused_with_products loaded_machine=true");
    let cache = Mutex::new(None);
    let records = Mutex::new(Vec::with_capacity(4096));
    let map = SyntheticTrace::column_map();
    let weights = weights();
    run_core(
        "tail",
        &[SynthProfile::Local],
        |source| Ok::<_, BenchError>((Tail::new(source, &cache, &map, &weights)?, ZERO)),
        |core, _runner_point| core.extract(&records),
    )?;
    report(&records.lock().map_err(|_| RunnerError::Core {
        message: BenchError::Lock.to_string(),
    })?);
    Ok(())
}
