//! Loaded-machine router pipeline, measured through the four-phase core runner.

pub mod support;

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use jolt_field::F128;
use jolt_poly::UnivariatePoly;
use jolt_rv64i_kernels::packed::scatter::{ScatterError, ScatterPlan};
use jolt_rv64i_kernels::router::claims::claims_pass;
use jolt_rv64i_kernels::router::cycle::RoutersCycleCore;
use jolt_rv64i_kernels::router::fold::FoldLayout;
use jolt_rv64i_kernels::router::lift::{source_lift_timed, RetainedWordLifts};
use jolt_rv64i_kernels::router::shape::{
    selector_counts, synthetic_router_shapes, RouterError, RouterShape,
};
use jolt_rv64i_kernels::router::short::RouterShortCore;
use jolt_rv64i_kernels::source::{CycleSource, SourceError, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, ClearSumcheckRecorder, ProveRounds, SequentialRounds,
    SumcheckError,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use thiserror::Error;

use support::{core_rounds, run_cases, Case, Clock, Record, RunnerError};

const HISTOGRAM_COLUMNS: [usize; 11] = [5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20];

#[derive(Debug, Error)]
enum BenchError {
    #[error(transparent)]
    Router(#[from] RouterError),
    #[error(transparent)]
    Source(#[from] SourceError),
    #[error(transparent)]
    Scatter(#[from] ScatterError),
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError<F128>),
    #[error(transparent)]
    Runner(#[from] RunnerError),
    #[error("router lifts are unavailable before pipeline completion")]
    Lock,
}

struct Prepared {
    trace: Arc<ValidatedTrace<SyntheticTrace>>,
    shapes: Vec<RouterShape>,
    plan: ScatterPlan<SyntheticTrace>,
    layout: FoldLayout,
    r_cycle: Vec<F128>,
}

struct RouterBench {
    prepared: Arc<Prepared>,
    short: RouterShortCore,
    lifts: Option<RetainedWordLifts>,
    cycle_point: Vec<F128>,
    short_point: Vec<F128>,
    times: [Duration; 7],
    cycle_setup: Duration,
    short_finish: Duration,
}

impl RouterBench {
    fn new(prepared: Arc<Prepared>) -> Result<(Self, F128), BenchError> {
        let start = Clock::start();
        let (fold, _) = prepared.layout.measure(
            &prepared.trace,
            &prepared.shapes,
            &prepared.r_cycle,
            &prepared.plan,
            &HISTOGRAM_COLUMNS,
        )?;
        let fold_ns = start.elapsed();
        drop(fold.ra_fold);
        drop(fold.histograms);
        let start = Clock::start();
        let w: Vec<_> = (0..prepared.shapes[0].log_outputs())
            .map(|i| F128::from_raw(0x457 + 29 * i as u128))
            .collect();
        let short = RouterShortCore::new(&prepared.shapes, &w, fold.folds)?;
        let mut times = [Duration::ZERO; 7];
        times[0] = fold_ns;
        times[1] = start.elapsed();
        Ok((
            Self {
                prepared,
                short,
                lifts: None,
                cycle_point: Vec::new(),
                short_point: Vec::with_capacity(17),
                times,
                cycle_setup: Duration::ZERO,
                short_finish: Duration::ZERO,
            },
            F128::from_raw(0),
        ))
    }

    fn extract(&mut self) -> Result<(), BenchError> {
        let lifts = self.lifts.as_ref().ok_or(BenchError::Lock)?;
        let start = Clock::start();
        let claims = claims_pass(
            &self.prepared.trace,
            lifts,
            &[0, 1, 2, 3, 4],
            &self.prepared.plan,
            &self.cycle_point,
        )?;
        let _ = black_box(claims);
        self.times[6] = start.elapsed();
        Ok(())
    }
}

impl ProveRounds<F128> for RouterBench {
    fn num_rounds(&self) -> usize {
        self.short.num_rounds()
    }
    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        let start = Clock::start();
        let message = self.short.prove_round(bind, round, claim)?;
        let elapsed = start.elapsed();
        self.times[1] += elapsed;
        if let Some(challenge) = bind {
            self.short_point.push(challenge);
        }
        Ok(message)
    }
    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        let start = Clock::start();
        self.short.finish_rounds(bind)?;
        self.short_finish = start.elapsed();
        self.times[1] += self.short_finish;
        self.short_point.push(bind);
        let x = &self.short_point;
        let prepared = &self.prepared;
        let (lifted, [rows_time, cycles_time]) =
            source_lift_timed(&prepared.trace, &prepared.shapes, x)
                .map_err(|_| missing("router source lift"))?;
        self.times[2] = cycles_time;
        self.times[3] = rows_time;
        let claims = self
            .short
            .final_values()
            .map_err(|_| missing("short final values"))?;
        let start = Clock::start();
        let cycle = RoutersCycleCore::new(
            &prepared.trace,
            &prepared.shapes,
            &prepared.r_cycle,
            x,
            lifted.source_tables,
        )
        .map_err(|_| missing("router cycle construction"))?;
        self.cycle_setup = start.elapsed();
        let mut members = cycle.members();
        let log_t = prepared.r_cycle.len();
        let prelude = BatchPrelude::try_new(
            claims
                .iter()
                .map(|&(input_claim, _)| BatchMember {
                    input_claim,
                    coefficient: F128::from_raw(1),
                    rounds: log_t,
                    offset: 0,
                })
                .collect(),
            log_t,
            5,
        )?;
        let mut recorder = ClearSumcheckRecorder::<F128>::new();
        let mut transcript = Blake2bTranscript::<F128>::new(b"router-benchmark-cycle");
        let mut handles: Vec<&mut dyn ProveRounds<F128>> = members
            .iter_mut()
            .map(|member| member as &mut dyn ProveRounds<F128>)
            .collect();
        let proved = prove_batch(
            &prelude,
            &mut handles,
            &mut SequentialRounds,
            &mut recorder,
            &mut transcript,
        )?;
        let equality = prepared
            .r_cycle
            .iter()
            .zip(&proved.challenges)
            .fold(F128::from_raw(1), |value, (&coordinate, &challenge)| {
                value * (F128::from_raw(1) + coordinate + challenge)
            });
        for (member, &expected) in members.iter().zip(&proved.member_claims) {
            let (source, selectors) = member
                .final_values()
                .map_err(|_| missing("router final values"))?;
            let actual = selectors
                .into_iter()
                .fold(equality * source, |value, selector| value * selector);
            if actual != expected {
                return Err(SumcheckError::RoundCheckFailed {
                    round: log_t,
                    expected,
                    actual,
                });
            }
        }
        let _ = black_box(proved.final_claim);
        self.cycle_point = proved.challenges;
        let [products, selectors] = cycle.phase_times();
        self.times[4] = products;
        self.times[5] = selectors;
        self.lifts = Some(lifted.lifts);
        Ok(())
    }
}

fn missing(kind: &'static str) -> SumcheckError<F128> {
    SumcheckError::MissingEvaluationSource { kind }
}

#[expect(clippy::print_stdout, reason = "model records are benchmark output")]
fn report(record: &Record, bytes: usize) {
    let log_t = record.log_t;
    let threads = record.threads;
    let cycles = (1_usize << log_t) as f64;
    let rho = (1_usize << 20) as f64 / cycles;
    let model = [
        70.13 + rho * 39.9 + 660_000.0 / cycles,
        (245_888.0 * (3.0 * 1.83 + 2.0 * 1.10 + 2.3)) / cycles,
        35.6,
        rho * 22.6,
        87.1,
        16.9,
        8.73 + rho * 17.2,
    ];
    let scaling = if threads == 12 { 9.6 } else { 1.0 };
    for (phase, name) in [
        "fold_pass",
        "short",
        "source_cycles",
        "source_rows",
        "cycle_products",
        "cycle_selectors",
        "claims_pass",
    ]
    .into_iter()
    .enumerate()
    {
        record.print_phase(
            &format!("routers_phase/{bytes}/{log_t}/{threads} phase={name}"),
            4 + phase,
            Some(model[phase] / scaling),
        );
    }
    record.print_phase(
        &format!("routers_setup/{bytes}/{log_t}/{threads} phase=short_finish"),
        12,
        None,
    );
    record.print(
        &format!("routers_model/{bytes}/{log_t}/{threads}"),
        &["construct", "rounds", "finish", "extract"].map(str::to_owned),
        Some(threshold(log_t, threads)),
    );
    println!("routers_model_counts/{bytes}/{log_t}/{threads} model_construct_ns={:.6} model_rounds_ns={:.6} model_finish_ns={:.6} model_extract_ns={:.6} model_total_ns={:.6} model_threads=1 target_model_total_ns={:.6} cycle_setup_ns={:.6} loaded_machine=true", model[0], model[1], model[2..6].iter().sum::<f64>(), model[6], model.iter().sum::<f64>(), model.iter().sum::<f64>() / scaling, record.phases[11].median);
}

fn threshold(log_t: usize, threads: usize) -> f64 {
    if threads == 12 && log_t == 22 {
        31.0
    } else {
        (if log_t == 22 { 299.0 } else { 376.0 }) / if threads == 12 { 9.6 } else { 1.0 }
    }
}

fn main() -> Result<(), RunnerError> {
    let variants = [
        ("routers/none", 0),
        ("routers/default", FoldLayout::DEFAULT_BYTE_BUCKET_LIMIT),
        ("routers/all", 64),
    ];
    let cases: Vec<_> = variants
        .iter()
        .enumerate()
        .map(|(index, (name, _))| {
            let mut case = Case::core(
                name,
                index,
                &[
                    "fold_pass",
                    "short",
                    "source_cycles",
                    "source_rows",
                    "cycle_products",
                    "cycle_selectors",
                    "claims_pass",
                    "cycle_setup",
                    "short_finish",
                ],
            );
            case.threshold = Some(threshold);
            case
        })
        .collect();
    let records = run_cases(
        &[SynthProfile::AllRows],
        &cases,
        |source| {
            let trace = Arc::new(ValidatedTrace::new(source)?);
            let shapes = synthetic_router_shapes()?;
            let counts = selector_counts(&trace, &shapes[0])?;
            variants
                .iter()
                .map(|(_, bytes)| {
                    let mut byte_values = vec![vec![]; shapes.len()];
                    byte_values[0] = FoldLayout::byte_bucket_values(&counts, *bytes);
                    let layout = FoldLayout::new(&trace, &shapes, &byte_values)?;
                    let plan = ScatterPlan::new(Arc::clone(&trace))?;
                    let log_t = trace.source().cycles().ilog2() as usize;
                    let r_cycle = (0..log_t)
                        .map(|i| F128::from_raw(0x987 + 53 * i as u128))
                        .collect();
                    Ok::<_, BenchError>(Arc::new(Prepared {
                        trace: Arc::clone(&trace),
                        shapes: shapes.clone(),
                        plan,
                        layout,
                        r_cycle,
                    }))
                })
                .collect::<Result<Vec<_>, _>>()
        },
        |prepared, &index, point, times| {
            let start = Clock::start();
            let (mut core, claim) =
                RouterBench::new(Arc::clone(&prepared[index])).map_err(|error| {
                    RunnerError::Core {
                        message: error.to_string(),
                    }
                })?;
            times.set(0, start.elapsed());
            let batch = core_rounds(&mut core, claim, point)?;
            times.set(1, batch.rounds);
            times.set(2, batch.finish);
            let start = Clock::start();
            core.extract().map_err(|error| RunnerError::Core {
                message: error.to_string(),
            })?;
            times.set(3, start.elapsed());
            Ok::<_, BenchError>(core)
        },
        |core, times| {
            for (phase, time) in core.times.iter().enumerate() {
                times.set(4 + phase, *time);
            }
            times.set(11, core.cycle_setup);
            times.set(12, core.short_finish);
        },
        |record, _, &index| report(record, variants[index].1),
    )?;
    for record in &records {
        if record.threads != 1 {
            continue;
        }
        if let Some(default) = records.iter().find(|other| {
            other.log_t == record.log_t
                && other.threads == 1
                && other.id.starts_with("routers/default/")
        }) {
            record.print_comparison(
                &format!(
                    "routers_layout_comparison/{}/1 layout={}",
                    record.log_t,
                    variants[cases
                        .iter()
                        .position(|case| record.id.starts_with(&format!("{}/", case.name)))
                        .unwrap_or(0)]
                    .1
                ),
                default,
            );
        }
    }
    Ok(())
}
