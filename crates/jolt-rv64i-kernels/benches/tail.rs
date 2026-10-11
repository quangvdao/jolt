//! Run `cargo bench -p jolt-rv64i-kernels --features test-utils --bench tail
//! -- --log-t 20,22 --threads 1,12 --samples 3`. Records are loaded-machine evidence.

pub mod support;

use jolt_field::{Accumulator, F128Accumulator, F128};
use jolt_rv64i_kernels::chunk_product::{
    combined_weight, ChunkProductCore, ChunkProductError, ChunkWeight, ChunkWeightTerm,
};
use jolt_rv64i_kernels::column_pass::{column_pass, ColumnPassError};
use jolt_rv64i_kernels::par::{CycleChunks, ParError};
use jolt_rv64i_kernels::reduction::{
    g_pass_digits, ColumnMap, ReductionCore, ReductionError, ReductionLeg,
};
use jolt_rv64i_kernels::round::eq::eq_table;
use jolt_rv64i_kernels::source::{
    CycleSource, PrepareRequest, PreparedGroups, SourceError, ValidatedTrace,
};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, ClearSumcheckRecorder, MemberFinish, MemberRound,
    RoundScheduler, SumcheckError,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rayon::prelude::*;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;
use support::gather::run_gathers;
use support::{run_batch, BatchRun, Case, Clock, PhaseTimes, RunnerError};
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
    members: [Duration; 3],
    finish: Duration,
}

impl RoundScheduler<F128> for TimingScheduler {
    fn batch_prove_round(
        &mut self,
        work: &mut [MemberRound<'_, F128>],
    ) -> Result<(), SumcheckError<F128>> {
        for item in work {
            let start = Clock::start();
            item.run()?;
            self.members[item.index] += start.elapsed();
        }
        Ok(())
    }
    fn batch_finish_rounds(
        &mut self,
        work: &mut [MemberFinish<'_, F128>],
    ) -> Result<(), SumcheckError<F128>> {
        let start = Clock::start();
        for item in work {
            item.run()?;
        }
        self.finish = start.elapsed();
        Ok(())
    }
}

struct Tail {
    source: Arc<SyntheticTrace>,
    chunks: [ChunkProductCore; 2],
    reduction: ReductionCore,
    prelude: BatchPrelude<F128>,
}

impl Tail {
    fn new(
        prepared: &Prepared,
        mut groups: PreparedGroups,
        map: &[ColumnMap],
        weights: &[Vec<F128>],
        times: &mut PhaseTimes,
    ) -> Result<Self, BenchError> {
        let source = Arc::clone(prepared.trace.source());
        let trace = Arc::clone(&prepared.trace);
        let log_t = source.cycles().ilog2() as usize;
        let start = Clock::start();
        let tables = g_pass_digits(&trace, map, weights)?;
        times.set(4, start.elapsed());
        let start = Clock::start();
        let first = combined_weight(log_t, &terms(log_t, 0))?;
        times.set(5, start.elapsed());
        let start = Clock::start();
        let second = combined_weight(log_t, &terms(log_t, 1))?;
        times.set(6, start.elapsed());
        let claims = prepared.claims;
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
        let start = Clock::start();
        let a = ChunkProductCore::new(
            groups.present.remove(0),
            points(0),
            ChunkWeight::Dense(first),
        )?;
        times.set(7, start.elapsed());
        let start = Clock::start();
        let b = ChunkProductCore::new(
            groups.present.remove(0),
            points(1),
            ChunkWeight::Dense(second),
        )?;
        times.set(8, start.elapsed());
        let start = Clock::start();
        let reduction = ReductionCore::new(tables, legs)?;
        times.set(9, start.elapsed());
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
        })
    }

    fn prove(&mut self, times: &mut PhaseTimes) -> Result<BatchRun, BenchError> {
        let [a, b] = &mut self.chunks;
        let mut transcript = Blake2bTranscript::new(b"rv64i-tail-benchmark-seed");
        let mut recorder = ClearSumcheckRecorder::<F128>::new();
        let mut scheduler = TimingScheduler::default();
        let start = Clock::start();
        let proved = prove_batch(
            &self.prelude,
            &mut [a, b, &mut self.reduction],
            &mut scheduler,
            &mut recorder,
            &mut transcript,
        )?;
        let batch = start.elapsed();
        for (member, duration) in scheduler.members.into_iter().enumerate() {
            times.set(10 + member, duration);
        }
        Ok(BatchRun {
            proved,
            rounds: batch.saturating_sub(scheduler.finish),
            finish: scheduler.finish,
        })
    }

    fn extract(&self, point: &[F128], times: &mut PhaseTimes) -> Result<[F128; 256], BenchError> {
        let _ = black_box(self.chunks[0].final_values()?);
        let _ = black_box(self.chunks[1].final_values()?);
        let _ = black_box(self.reduction.final_values()?);
        let start = Clock::start();
        let columns = column_pass(self.source.rows(), point)?;
        times.set(13, start.elapsed());
        Ok(columns)
    }
}

#[expect(
    clippy::print_stdout,
    reason = "phase and model records are benchmark output"
)]
fn main() -> Result<(), RunnerError> {
    println!("tail_driver_note challenge_source=seeded_blake2b_batch column_gathers=fused_with_products standalone_gathers_and_lazy_binds=separate_measurement_not_a_share_of_fused_rounds loaded_machine=true");
    let map = SyntheticTrace::column_map();
    let weights = weights();
    let mut case = Case::core(
        "tail",
        (),
        &[
            "g_pass_digits",
            "combined_weight_five",
            "combined_weight_two",
            "compact_digit_construction_zero",
            "compact_digit_construction_one",
            "reduction_construction",
            "chunk_zero_gathers_and_rounds",
            "chunk_one_gathers_and_rounds",
            "reduction_rounds",
            "column_pass",
        ],
    );
    case.threshold = Some(|_, threads| if threads == 1 { 192.0 } else { 20.0 });
    run_batch(
        &[SynthProfile::Local],
        case,
        (
            |source| {
                let log_t = source.cycles().ilog2() as usize;
                let geometry = CycleChunks::new(log_t, 0)?;
                let trace = Arc::new(ValidatedTrace::new(Arc::clone(&source))?);
                let first = combined_weight(log_t, &terms(log_t, 0))?;
                let second = combined_weight(log_t, &terms(log_t, 1))?;
                Ok::<_, BenchError>(Prepared {
                    trace,
                    claims: [
                        chunk_claim(&source, 0, &first, geometry),
                        chunk_claim(&source, 1, &second, geometry),
                    ],
                })
            },
            |prepared| {
                ValidatedTrace::prepare(
                    Arc::clone(prepared.trace.source()),
                    PrepareRequest {
                        present: vec![(0..5).collect(), (5..10).collect()],
                        optional: vec![],
                    },
                )
                .map(|(_, groups)| groups)
                .map_err(BenchError::from)
            },
        ),
        |prepared, groups, times| Tail::new(prepared, groups, &map, &weights, times),
        Tail::prove,
        Tail::extract,
        |record, _| {
            let log_t = record.log_t;
            let threads = record.threads;
            let names = ["construct", "rounds", "finish", "extract"].map(str::to_owned);
            record.print(
                &format!("tail_phases/local/{log_t}/{threads}"),
                &names,
                Some(if threads == 1 { 192.0 } else { 20.0 }),
            );
            let scale = threads as f64;
            for (name, phase, model) in [
                ("g_pass_digits", 4, 15.3),
                ("combined_weight_five", 5, 5.1),
                ("combined_weight_two", 6, 2.9),
                ("chunk_zero_gathers_and_rounds", 10, 50.1),
                ("chunk_one_gathers_and_rounds", 11, 50.1),
                ("reduction_rounds", 12, 8.8),
                ("column_pass", 13, 21.0),
            ] {
                record.print_phase(
                    &format!("tail_part/{name}/local/{log_t}/{threads}"),
                    phase,
                    Some(model / scale),
                );
            }
            println!("tail_gather_boundary/local/{log_t}/{threads} column_gathers_ns=separate_measurement model_gathers_ns={:.6} model_product_rounds_ns={:.6} reason=fused_library_pair_loop compact_digit_construction_zero_ns={:.6} compact_digit_construction_one_ns={:.6} reduction_construction_ns={:.6} loaded_machine=true", 10.0 / scale, 40.1 / scale, record.phases[7].median, record.phases[8].median, record.phases[9].median);
        },
    )?;
    run_gathers(
        &[SynthProfile::Local],
        &[
            (
                "tail_part/chunk_zero/standalone_gathers_and_lazy_binds",
                (0..5).collect(),
            ),
            (
                "tail_part/chunk_one/standalone_gathers_and_lazy_binds",
                (5..10).collect(),
            ),
        ],
        points,
    )
}
