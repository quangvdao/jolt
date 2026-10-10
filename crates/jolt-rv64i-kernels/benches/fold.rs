//! Complete folds on all_rows. The runner reports preparation in construction
//! and the entire fold in extraction; this pass has no sum-check rounds.

pub mod support;

use jolt_field::{Field, F128};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_kernels::packed::scatter::ScatterError;
use jolt_rv64i_kernels::packed::scatter::ScatterPlan;
use jolt_rv64i_kernels::par::CycleChunks;
use jolt_rv64i_kernels::par::ParError;
use jolt_rv64i_kernels::router::fold::{FoldLayout, FoldOutput};
use jolt_rv64i_kernels::router::shape::RouterError;
use jolt_rv64i_kernels::router::shape::{selector_counts, synthetic_router_shapes, RouterShape};
use jolt_rv64i_kernels::source::SourceError;
use jolt_rv64i_kernels::source::{CycleSource, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::cmp::Reverse;
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
use std::sync::Arc;
use support::{run_core, RunnerError};

const HISTOGRAM_COLUMNS: [usize; 11] = [5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20];

struct FoldBench {
    trace: Arc<ValidatedTrace<SyntheticTrace>>,
    shapes: Vec<RouterShape>,
    point: Vec<F128>,
    plan: ScatterPlan<SyntheticTrace>,
    layout: FoldLayout,
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
    #[expect(
        clippy::print_stdout,
        reason = "scratch and operation counts are required benchmark records"
    )]
    fn new(source: Arc<SyntheticTrace>, byte_count: usize) -> Result<(Self, F128), FoldBenchError> {
        let log_t = source.cycles().ilog2() as usize;
        let trace = Arc::new(ValidatedTrace::new(source)?);
        let shapes = synthetic_router_shapes()?;
        let mut values = vec![vec![]; shapes.len()];
        let mut bucket_xors = 0;
        for (index, shape) in shapes.iter().enumerate() {
            let counts = selector_counts(&trace, shape)?;
            if index == 0 {
                let mut selectors: Vec<_> = (0..counts.len()).collect();
                selectors.sort_unstable_by_key(|&h| (Reverse(counts[h]), h));
                values[0] = selectors.into_iter().take(byte_count).collect();
            }
            let words = FoldLayout::CALIBRATION_WORD_SETS[index];
            for (h, count) in counts.into_iter().enumerate() {
                let positions = if values[index].contains(&h) { 8 } else { 16 };
                bucket_xors += count * (words * positions + if index == 0 { 8 } else { 0 });
            }
        }
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
        println!("fold/scratch/{byte_count}/{log_t}/{} cycle_bucket_bytes_per_worker={} row_bucket_bytes_per_worker={} scatter_buffer_bytes={} plan_bytes={plan_bytes} cycle_bucket_xors={:.6} model_cycle_bucket_xors=111.5 model_multiplication_ns=1.83 model_cycle_bucket_ns={:.6} model_scatter_ns=1.4 model_rows_ns={:.6} model_zero_merge_readout_ns={:.6} loaded_machine=true", rayon::current_num_threads(), (layout.entries()+histogram_entries)*16, (layout.row_entries()+histogram_entries)*16, cycles*16, bucket_xors as f64/cycles as f64, bucket_xors as f64/cycles as f64*0.6, rho*(64.0*0.6+5.0*0.3), 2.2e6*0.3/cycles as f64);
        Ok((
            Self {
                trace,
                shapes,
                point,
                plan,
                layout,
            },
            F128::from_raw(0),
        ))
    }
    #[expect(
        clippy::print_stdout,
        reason = "pass subphase timing is a required benchmark record"
    )]
    fn extract(&self) -> Result<FoldOutput, FoldBenchError> {
        let (output, phases) = self.layout.measure(
            &self.trace,
            &self.shapes,
            &self.point,
            &self.plan,
            &HISTOGRAM_COLUMNS,
        )?;
        let times = phases.map(|time| time.as_nanos() as f64 / self.trace.source().cycles() as f64);
        println!("fold/pass_phases fused_cycle_ns={:.6} scatter_ns={:.6} rows_ns={:.6} setup_merge_readout_ns={:.6}", times[0], times[1], times[2], times[3]);
        Ok(output)
    }
}

fn main() -> Result<(), RunnerError> {
    for (name, bytes) in [("fold/none", 0), ("fold/default", 8), ("fold/all", 64)] {
        run_core(
            name,
            &[SynthProfile::AllRows],
            |source| FoldBench::new(source, bytes),
            |core, _| core.extract(),
        )?;
    }
    Ok(())
}
