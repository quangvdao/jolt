//! Complete folds on all_rows and small_values. The runner reports preparation
//! in construction and the entire fold in extraction; this pass has no sum-check rounds.

pub mod support;

use jolt_field::{Field, F128};
use jolt_rv64i_kernels::packed::scatter::ScatterError;
use jolt_rv64i_kernels::packed::scatter::ScatterPlan;
use jolt_rv64i_kernels::par::CycleChunks;
use jolt_rv64i_kernels::par::ParError;
use jolt_rv64i_kernels::router::fold::{FoldCalibration, FoldLayout};
use jolt_rv64i_kernels::router::shape::RouterError;
use jolt_rv64i_kernels::router::shape::{selector_counts, synthetic_router_shapes, RouterShape};
use jolt_rv64i_kernels::source::SourceError;
use jolt_rv64i_kernels::source::{CycleSource, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
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
use support::{run_cases, Case, Clock, RunnerError};

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

impl FoldBench {
    fn new(source: Arc<SyntheticTrace>, byte_count: usize) -> Result<Self, FoldBenchError> {
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
        Ok(Self {
            trace,
            shapes,
            point,
            plan,
            layout,
            diagnostics,
        })
    }
    #[expect(
        clippy::print_stdout,
        reason = "typed benchmark diagnostics are printed outside measurements"
    )]
    fn report_diagnostics(&self) {
        let d = &self.diagnostics;
        let profile = if self.trace.source().profile() == SynthProfile::AllRows {
            String::new()
        } else {
            format!("{}/", self.trace.source().profile().name())
        };
        println!("fold/scratch/{profile}{}/{}/{}/cycle cycle_bucket_bytes_per_worker={} row_bucket_bytes_per_worker={} chunk_weight_scratch_bytes_per_worker=0 scatter_buffer_bytes={} plan_bytes={} cycle_bucket_xors={:.6} model_cycle_bucket_xors=111.5 model_multiplication_ns=1.83 model_cycle_bucket_ns={:.6} model_scatter_ns=1.4 model_rows_ns={:.6} model_zero_merge_readout_ns={:.6} loaded_machine=true", d.byte_count, d.log_t, d.threads, d.cycle_bucket_bytes, d.row_bucket_bytes, d.scatter_bytes, d.plan_bytes, d.updates, d.updates*0.6, d.row_model_ns, d.merge_model_ns);
    }
}

#[expect(
    clippy::print_stdout,
    reason = "fold phase distributions are benchmark output"
)]
fn main() -> Result<(), RunnerError> {
    let cases = [
        Case::core(
            "fold/none",
            0,
            &["fused_cycle", "scatter", "rows", "setup_merge_readout"],
        ),
        Case::core(
            "fold/default",
            FoldLayout::DEFAULT_BYTE_BUCKET_LIMIT,
            &["fused_cycle", "scatter", "rows", "setup_merge_readout"],
        ),
        Case::core(
            "fold/all",
            64,
            &["fused_cycle", "scatter", "rows", "setup_merge_readout"],
        ),
    ];
    let _ = run_cases(
        &[SynthProfile::AllRows, SynthProfile::SmallValues],
        &cases,
        Ok::<_, RunnerError>,
        |source, &bytes, _, times| {
            let clock = Clock::start();
            let core =
                FoldBench::new(Arc::clone(source), bytes).map_err(|error| RunnerError::Core {
                    message: error.to_string(),
                })?;
            times.set(0, clock.elapsed());
            let clock = Clock::start();
            let (output, phases) = core
                .layout
                .measure(
                    &core.trace,
                    &core.shapes,
                    &core.point,
                    &core.plan,
                    &HISTOGRAM_COLUMNS,
                )
                .map_err(|error| RunnerError::Core {
                    message: error.to_string(),
                })?;
            let _ = std::hint::black_box(&output);
            times.set(3, clock.elapsed());
            for (index, phase) in phases.into_iter().enumerate() {
                times.set(4 + index, phase);
            }
            Ok((core, output))
        },
        |(core, _), _| core.report_diagnostics(),
        |record, _, &bytes| {
            let all_rows = record.id.contains("/all_rows/");
            if all_rows && bytes == FoldLayout::DEFAULT_BYTE_BUCKET_LIMIT && record.threads == 1 {
                record.print_requirement(
                    &format!("fold_pass/all_rows/{}/1", record.log_t),
                    3,
                    if record.log_t == 22 { 100.0 } else { 138.0 },
                );
            }
            let profile = if all_rows { "" } else { "small_values/" };
            println!("fold/pass_phases/{profile}{}/{}/{}/cycle fused_cycle_ns={:.6} scatter_ns={:.6} rows_ns={:.6} setup_merge_readout_ns={:.6} loaded_machine=true", bytes, record.log_t, record.threads, record.phases[4].median, record.phases[5].median, record.phases[6].median, record.phases[7].median);
        },
    )?;
    Ok(())
}
