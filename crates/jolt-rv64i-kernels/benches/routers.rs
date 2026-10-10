//! Loaded-machine router pipeline, measured through the four-phase core runner.

pub mod support;

use std::cmp::Reverse;
use std::hint::black_box;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use jolt_field::{Field, F128};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_kernels::packed::scatter::{ScatterError, ScatterPlan};
use jolt_rv64i_kernels::router::claims::claims_pass;
use jolt_rv64i_kernels::router::cycle::RoutersCycleCore;
use jolt_rv64i_kernels::router::fold::FoldLayout;
use jolt_rv64i_kernels::router::lift::{source_lift, RetainedWordLifts};
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
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::ThreadPoolBuilder;
use thiserror::Error;

use support::{run_core, RunnerError};

const HISTOGRAM_COLUMNS: [usize; 11] = [5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20];

#[derive(Debug, Error)]
enum BenchError {
    #[error(transparent)]
    Router(#[from] RouterError),
    #[error(transparent)]
    Source(#[from] SourceError),
    #[error(transparent)]
    Scatter(#[from] ScatterError),
    #[error("synthetic trace failed: {0}")]
    Synth(String),
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError<F128>),
    #[error("benchmark preparation lock was poisoned")]
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
    runner_times: [u128; 4],
    times: [u128; 7],
    setup_ns: u128,
    cycle_setup_ns: u128,
    short_finish_ns: u128,
}

struct Record {
    runner_times: [u128; 4],
    layout: usize,
    log_t: usize,
    threads: usize,
    times: [u128; 7],
    setup_ns: u128,
    cycle_setup_ns: u128,
    short_finish_ns: u128,
}

impl RouterBench {
    fn new(
        source: Arc<SyntheticTrace>,
        bytes: usize,
        cache: &Mutex<Option<Arc<Prepared>>>,
    ) -> Result<(Self, F128), BenchError> {
        let construction = Instant::now();
        let start = Instant::now();
        let mut cached = cache.lock().map_err(|_| BenchError::Lock)?;
        let prepared = if let Some(prepared) = cached
            .as_ref()
            .filter(|prepared| Arc::ptr_eq(prepared.trace.source(), &source))
        {
            Arc::clone(prepared)
        } else {
            let trace = Arc::new(ValidatedTrace::new(source)?);
            let shapes = synthetic_router_shapes()?;
            let mut byte_values = vec![vec![]; shapes.len()];
            let counts = selector_counts(&trace, &shapes[0])?;
            let mut values: Vec<_> = (0..counts.len()).collect();
            values.sort_unstable_by_key(|&h| (Reverse(counts[h]), h));
            byte_values[0] = values.into_iter().take(bytes).collect();
            let layout = FoldLayout::new(&trace, &shapes, &byte_values)?;
            let plan = ScatterPlan::new(Arc::clone(&trace))?;
            let log_t = trace.source().cycles().ilog2() as usize;
            let r_cycle = (0..log_t)
                .map(|i| F128::from_raw(0x987 + 53 * i as u128))
                .collect();
            let prepared = Arc::new(Prepared {
                trace,
                shapes,
                plan,
                layout,
                r_cycle,
            });
            *cached = Some(Arc::clone(&prepared));
            prepared
        };
        drop(cached);
        let setup_ns = start.elapsed().as_nanos();
        let start = Instant::now();
        let (fold, _) = prepared.layout.measure(
            &prepared.trace,
            &prepared.shapes,
            &prepared.r_cycle,
            &prepared.plan,
            &HISTOGRAM_COLUMNS,
        )?;
        let fold_ns = start.elapsed().as_nanos();
        drop(fold.ra_fold);
        drop(fold.histograms);
        let start = Instant::now();
        let w: Vec<_> = (0..prepared.shapes[0].log_outputs())
            .map(|i| F128::from_raw(0x457 + 29 * i as u128))
            .collect();
        let short = RouterShortCore::new(&prepared.shapes, &w, fold.folds)?;
        let mut times = [0; 7];
        times[0] = fold_ns;
        times[1] = start.elapsed().as_nanos();
        Ok((
            Self {
                prepared,
                short,
                lifts: None,
                cycle_point: Vec::new(),
                short_point: Vec::with_capacity(17),
                runner_times: [construction.elapsed().as_nanos(), 0, 0, 0],
                times,
                setup_ns,
                cycle_setup_ns: 0,
                short_finish_ns: 0,
            },
            F128::from_raw(0),
        ))
    }

    fn extract(&self, layout: usize, records: &Mutex<Vec<Record>>) -> Result<(), BenchError> {
        let lifts = self.lifts.as_ref().ok_or(BenchError::Lock)?;
        let start = Instant::now();
        let claims = claims_pass(
            &self.prepared.trace,
            lifts,
            &[0, 1, 2, 3, 4],
            &self.prepared.plan,
            &self.cycle_point,
        )?;
        let _ = black_box(claims);
        let mut times = self.times;
        times[6] = start.elapsed().as_nanos();
        let mut runner_times = self.runner_times;
        runner_times[3] = times[6];
        records.lock().map_err(|_| BenchError::Lock)?.push(Record {
            runner_times,
            layout,
            log_t: self.prepared.r_cycle.len(),
            threads: rayon::current_num_threads(),
            times,
            setup_ns: self.setup_ns,
            cycle_setup_ns: self.cycle_setup_ns,
            short_finish_ns: self.short_finish_ns,
        });
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
        let start = Instant::now();
        let message = self.short.prove_round(bind, round, claim)?;
        let elapsed = start.elapsed().as_nanos();
        self.times[1] += elapsed;
        self.runner_times[1] += elapsed;
        if let Some(challenge) = bind {
            self.short_point.push(challenge);
        }
        Ok(message)
    }
    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        let finish = Instant::now();
        let start = Instant::now();
        self.short.finish_rounds(bind)?;
        self.short_finish_ns = start.elapsed().as_nanos();
        self.times[1] += self.short_finish_ns;
        self.short_point.push(bind);
        let x = &self.short_point;
        let prepared = &self.prepared;
        let lifted = source_lift(&prepared.trace, &prepared.shapes, x)
            .map_err(|_| missing("router source lift"))?;
        self.times[2] = lifted.cycles_time.as_nanos();
        self.times[3] = lifted.row_tables_time.as_nanos();
        let claims = self
            .short
            .final_values()
            .map_err(|_| missing("short final values"))?;
        let start = Instant::now();
        let cycle = RoutersCycleCore::new(
            &prepared.trace,
            &prepared.shapes,
            &prepared.r_cycle,
            x,
            lifted.source_tables,
        )
        .map_err(|_| missing("router cycle construction"))?;
        self.cycle_setup_ns = start.elapsed().as_nanos();
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
        self.times[4] = products.as_nanos();
        self.times[5] = selectors.as_nanos();
        self.lifts = Some(lifted.lifts);
        self.runner_times[2] = finish.elapsed().as_nanos();
        Ok(())
    }
}

fn missing(kind: &'static str) -> SumcheckError<F128> {
    SumcheckError::MissingEvaluationSource { kind }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(0.0)
}

#[expect(
    clippy::print_stdout,
    reason = "model comparisons are required benchmark output"
)]
fn report(records: &[Record]) {
    for (index, record) in records.iter().enumerate() {
        if records[..index].iter().any(|prior| {
            (prior.layout, prior.log_t, prior.threads)
                == (record.layout, record.log_t, record.threads)
        }) {
            continue;
        }
        let samples: Vec<_> = records
            .iter()
            .filter(|sample| {
                (sample.layout, sample.log_t, sample.threads)
                    == (record.layout, record.log_t, record.threads)
            })
            .collect();
        let cycles = (1_usize << record.log_t) as f64;
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
        let mut measured = [0.0; 7];
        let scaling = if record.threads == 12 { 9.6 } else { 1.0 };
        for phase in 0..7 {
            measured[phase] = median(
                samples
                    .iter()
                    .map(|sample| sample.times[phase] as f64 / cycles)
                    .collect(),
            );
            println!("routers_phase/{}/{}/{} phase={} median_ns={:.6} min_ns={:.6} model_ns={:.6} model_threads=1 target_model_ns={:.6} over_model_25pct={} over_target_25pct={} samples={} loaded_machine=true", record.layout, record.log_t, record.threads, ["fold_pass", "short", "source_cycles", "source_rows", "cycle_products", "cycle_selectors", "claims_pass"][phase], measured[phase], samples.iter().map(|sample| sample.times[phase] as f64 / cycles).fold(f64::INFINITY, f64::min), model[phase], model[phase] / scaling, measured[phase] > 1.25 * model[phase], measured[phase] > 1.25 * model[phase] / scaling, samples.len());
        }
        let total = median(
            samples
                .iter()
                .map(|sample| sample.runner_times.iter().sum::<u128>() as f64 / cycles)
                .collect(),
        );
        let threshold = if record.log_t == 22 { 299.0 } else { 376.0 } / scaling;
        println!(
            "routers_setup/{}/{}/{} first_use_ns={:.6} short_finish_ns={:.6} loaded_machine=true",
            record.layout,
            record.log_t,
            record.threads,
            samples
                .iter()
                .map(|sample| sample.setup_ns as f64 / cycles)
                .fold(0.0, f64::max),
            median(
                samples
                    .iter()
                    .map(|sample| sample.short_finish_ns as f64 / cycles)
                    .collect()
            )
        );
        let [constructor, rounds, finish, extract] = std::array::from_fn(|phase| {
            median(
                samples
                    .iter()
                    .map(|sample| sample.runner_times[phase] as f64 / cycles)
                    .collect(),
            )
        });
        println!("routers_model/{}/{}/{} construct_ns={constructor:.6} rounds_ns={rounds:.6} finish_ns={finish:.6} extract_ns={extract:.6} total_ns={total:.6} model_construct_ns={:.6} model_rounds_ns={:.6} model_finish_ns={:.6} model_extract_ns={:.6} model_total_ns={:.6} model_threads=1 target_model_total_ns={:.6} threshold_ns={threshold:.6} meets_threshold={} cycle_setup_ns={:.6} loaded_machine=true", record.layout, record.log_t, record.threads, model[0], model[1], model[2..6].iter().sum::<f64>(), model[6], model.iter().sum::<f64>(), model.iter().sum::<f64>() / scaling, total <= threshold, median(samples.iter().map(|sample| sample.cycle_setup_ns as f64 / cycles).collect()));
    }
}

#[expect(
    clippy::print_stdout,
    reason = "five alternating layout samples are required comparison evidence"
)]
fn compare_layouts(records: &[Record]) -> Result<(), RunnerError> {
    let pool = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .map_err(|error| RunnerError::ThreadPool {
            message: error.to_string(),
        })?;
    let _ = pool.broadcast(|_| black_box(()));
    for log_t in [20, 22] {
        if !records
            .iter()
            .any(|record| record.log_t == log_t && record.threads == 1)
        {
            continue;
        }
        let measurements = pool
            .install(|| -> Result<_, BenchError> {
                let source = Arc::new(
                    SyntheticTrace::new(SynthProfile::AllRows, log_t, 1 << 20, 0x5eed)
                        .map_err(|error| BenchError::Synth(error.to_string()))?,
                );
                let caches: [Mutex<Option<Arc<Prepared>>>; 3] =
                    std::array::from_fn(|_| Mutex::new(None));
                let mut rng = ChaCha20Rng::seed_from_u64(0x726f_756e_6473);
                let x: [F128; 17] = std::array::from_fn(|_| F128::random(&mut rng));
                let mut values: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(5));
                for sample in 0..6 {
                    let order = if sample % 2 == 0 {
                        [0, 1, 2]
                    } else {
                        [2, 1, 0]
                    };
                    for variant in order {
                        let start = Instant::now();
                        let (mut core, mut claim) = RouterBench::new(
                            Arc::clone(&source),
                            [0, 8, 64][variant],
                            &caches[variant],
                        )?;
                        for (round, &challenge) in x.iter().enumerate() {
                            let message = core.prove_round(
                                round.checked_sub(1).map(|previous| x[previous]),
                                round,
                                claim,
                            )?;
                            claim = message.evaluate(challenge);
                        }
                        core.finish_rounds(x[16])?;
                        let temporary = Mutex::new(Vec::new());
                        core.extract([0, 8, 64][variant], &temporary)?;
                        let _ = black_box(claim);
                        if sample > 0 {
                            values[variant]
                                .push(start.elapsed().as_nanos() as f64 / source.cycles() as f64);
                        }
                    }
                }
                Ok(values)
            })
            .map_err(|error| RunnerError::Core {
                message: error.to_string(),
            })?;
        for (variant, samples) in measurements.iter().enumerate() {
            let spread = |samples: &[f64]| {
                samples.iter().copied().fold(0.0, f64::max)
                    - samples.iter().copied().fold(f64::INFINITY, f64::min)
            };
            let gain = median(measurements[1].clone()) - median(samples.clone());
            let combined_spread = spread(&measurements[1]) + spread(samples);
            println!("routers_layout_comparison/{log_t}/1 layout={} samples=5 alternating=true same_source=true min_ns={:.6} median_ns={:.6} max_ns={:.6} gain_over_default_ns={gain:.6} combined_spread_ns={combined_spread:.6} gain_exceeds_spread={} loaded_machine=true retained_default=8", [0, 8, 64][variant], samples.iter().copied().fold(f64::INFINITY, f64::min), median(samples.clone()), samples.iter().copied().fold(0.0, f64::max), gain > combined_spread);
        }
    }
    Ok(())
}

fn main() -> Result<(), RunnerError> {
    let records = Mutex::new(Vec::new());
    for (name, bytes) in [
        ("routers/none", 0),
        ("routers/default", 8),
        ("routers/all", 64),
    ] {
        let cache = Mutex::new(None);
        run_core(
            name,
            &[SynthProfile::AllRows],
            |source| RouterBench::new(source, bytes, &cache),
            |core, _| core.extract(bytes, &records),
        )?;
    }
    let records = records.into_inner().map_err(|_| RunnerError::Core {
        message: BenchError::Lock.to_string(),
    })?;
    report(&records);
    compare_layouts(&records)
}
