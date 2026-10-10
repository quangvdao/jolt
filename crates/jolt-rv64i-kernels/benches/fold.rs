//! Complete folds on all_rows. The runner reports preparation in construction
//! and the entire fold in extraction; this pass has no sum-check rounds.

pub mod support;

use jolt_field::{Field, F128};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_kernels::packed::scatter::ScatterError;
use jolt_rv64i_kernels::packed::scatter::ScatterPlan;
use jolt_rv64i_kernels::par::CycleChunks;
use jolt_rv64i_kernels::par::ParError;
use jolt_rv64i_kernels::router::fold::{FoldCalibration, FoldLayout, FoldOutput};
use jolt_rv64i_kernels::router::shape::RouterError;
use jolt_rv64i_kernels::router::shape::{selector_counts, synthetic_router_shapes, RouterShape};
use jolt_rv64i_kernels::source::SourceError;
use jolt_rv64i_kernels::source::{CycleSource, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use thiserror::Error;

#[derive(Debug, Error)]
enum FoldBenchError {
    #[error(transparent)]
    Router(#[from] RouterError),
    #[error(transparent)]
    Source(#[from] SourceError),
    #[error(transparent)]
    Scatter(#[from] ScatterError),
    #[error(transparent)]
    Par(#[from] ParError),
}
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Duration;
use support::{run_core_variants, CycleScale, RunnerError};

const HISTOGRAM_COLUMNS: [usize; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20];

struct FoldBench {
    trace: Arc<ValidatedTrace<SyntheticTrace>>,
    shapes: Vec<RouterShape>,
    point: Vec<F128>,
    plan: ScatterPlan<SyntheticTrace>,
    layout: FoldLayout,
    diagnostics: Diagnostics,
}

struct Diagnostics {
    byte_count: usize,
    log_t: usize,
    threads: usize,
    cycle_bucket_bytes: usize,
    row_bucket_bytes: usize,
    scatter_bytes: usize,
    plan_bytes: usize,
    updates: f64,
    row_model_ns: f64,
    merge_model_ns: f64,
}

struct MeasuredFold {
    output: FoldOutput,
    phases: [Duration; 4],
}

impl ProveRounds<F128> for FoldBench {
    fn num_rounds(&self) -> usize {
        0
    }
    fn prove_round(
        &mut self,
        _: Option<F128>,
        round: usize,
        _: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        Err(SumcheckError::WrongNumberOfRounds {
            expected: 0,
            got: round,
        })
    }
    fn finish_rounds(&mut self, _: F128) -> Result<(), SumcheckError<F128>> {
        Err(SumcheckError::MissingEvaluationSource {
            kind: "fold pass has no rounds",
        })
    }
}

impl FoldBench {
    fn new(source: Arc<SyntheticTrace>, byte_count: usize) -> Result<(Self, F128), FoldBenchError> {
        let log_t = source.cycles().ilog2() as usize;
        let trace = Arc::new(ValidatedTrace::new(source)?);
        let shapes = synthetic_router_shapes()?;
        let counts = selector_counts(&trace, &shapes[0])?;
        let mut values = vec![vec![]; shapes.len()];
        values[0] = FoldLayout::byte_bucket_values(&counts, byte_count);
        let calibration = FoldCalibration::new(0)?;
        let word_sets = calibration.word_sets();
        let variant_xors: usize = counts
            .iter()
            .enumerate()
            .map(|(h, count)| {
                count * (word_sets[0] * if values[0].contains(&h) { 8 } else { 16 } + 8)
            })
            .sum();
        let geometry = CycleChunks::new(log_t, 0)?;
        let other_xors: usize = geometry
            .ranges()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|range| {
                range
                    .clone()
                    .map(|cycle| {
                        shapes
                            .iter()
                            .enumerate()
                            .skip(1)
                            .filter(|(_, shape)| {
                                shape.factors().iter().all(|factor| {
                                    trace.source().digit(factor.column, cycle).is_some()
                                })
                            })
                            .map(|(index, _)| word_sets[index] * 16)
                            .sum::<usize>()
                    })
                    .sum::<usize>()
            })
            .sum();
        let bucket_xors = variant_xors + other_xors;
        let layout = FoldLayout::new(&trace, &shapes, &values)?;
        let plan = ScatterPlan::new(Arc::clone(&trace))?;
        let mut rng = ChaCha20Rng::seed_from_u64(0x666f_6c64);
        let point = (0..log_t).map(|_| F128::random(&mut rng)).collect();
        let cycles = trace.source().cycles();
        let chunks = CycleChunks::new(log_t, 0)?.ranges().len();
        let plan_bytes =
            4 * cycles + chunks * 256 * 8 + std::mem::size_of::<ScatterPlan<SyntheticTrace>>();
        let histogram_entries: usize = HISTOGRAM_COLUMNS
            .iter()
            .map(|&column| 1 << trace.source().bits(column))
            .sum();
        let rho = trace.source().bytecode_rows() as f64 / cycles as f64;
        let diagnostics = Diagnostics {
            byte_count,
            log_t,
            threads: rayon::current_num_threads(),
            cycle_bucket_bytes: (layout.entries() + histogram_entries) * 16,
            row_bucket_bytes: (layout.row_entries() + histogram_entries) * 16,
            scatter_bytes: cycles * 16,
            plan_bytes,
            updates: bucket_xors as f64 / cycles as f64,
            row_model_ns: rho * (64.0 * 0.6 + 5.0 * 0.3),
            merge_model_ns: 2.2e6 * 0.3 / cycles as f64,
        };
        Ok((
            Self {
                trace,
                shapes,
                point,
                plan,
                layout,
                diagnostics,
            },
            F128::from_raw(0),
        ))
    }
    fn extract(&self) -> Result<MeasuredFold, FoldBenchError> {
        let (output, phases) = self.layout.measure(
            &self.trace,
            &self.shapes,
            &self.point,
            &self.plan,
            &HISTOGRAM_COLUMNS,
        )?;
        Ok(MeasuredFold { output, phases })
    }

    #[expect(
        clippy::print_stdout,
        reason = "typed benchmark diagnostics are printed outside measurements"
    )]
    fn report(&self, measured: &MeasuredFold, scale: CycleScale) {
        let d = &self.diagnostics;
        let times = scale.durations(measured.phases);
        let _ = std::hint::black_box(&measured.output);
        println!("fold/scratch/{}/{}/{}/cycle cycle_bucket_bytes_per_worker={} row_bucket_bytes_per_worker={} chunk_weight_scratch_bytes_per_worker=0 scatter_buffer_bytes={} plan_bytes={} cycle_bucket_xors={:.6} model_cycle_bucket_xors=111.5 model_multiplication_ns=1.83 model_cycle_bucket_ns={:.6} model_scatter_ns=1.4 model_rows_ns={:.6} model_zero_merge_readout_ns={:.6} loaded_machine=true", d.byte_count, d.log_t, d.threads, d.cycle_bucket_bytes, d.row_bucket_bytes, d.scatter_bytes, d.plan_bytes, d.updates, d.updates*0.6, d.row_model_ns, d.merge_model_ns);
        println!("fold/pass_phases/{}/{}/{}/cycle fused_cycle_ns={:.6} scatter_ns={:.6} rows_ns={:.6} setup_merge_readout_ns={:.6}", d.byte_count, d.log_t, d.threads, times[0], times[1], times[2], times[3]);
    }
}

fn main() -> Result<(), RunnerError> {
    let variants = [
        ("none", 0),
        ("default", FoldLayout::DEFAULT_BYTE_BUCKET_LIMIT),
        ("all", 64),
    ];
    run_core_variants(
        "fold",
        &[SynthProfile::AllRows],
        &variants,
        |source, &bytes| FoldBench::new(source, bytes),
        |core, _| core.extract(),
        |core, measured, scale| core.report(measured, scale),
    )
}
