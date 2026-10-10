//! Local-profile outer sum-check timings, including the option matrix and the
//! median wall time of each round. Use `--samples 3` or more on a loaded machine.
//! For paired samples use `--compare-monomial --log-t 20,22 --threads 1
//! --samples 5`, or `--variants monomial_rounds2,monomial_rounds3`. The paired
//! mode alternates execution order on one resident source and reports total
//! construction, rounds and finish time; extraction remains a separate phase.

pub mod support;

use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use jolt_field::{Field, F128};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_kernels::outer_f2::{OuterF2Core, OuterF2Options};
use jolt_rv64i_kernels::source::CycleSource;
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::ThreadPoolBuilder;

use support::{run_core, RunnerError};

struct TimedCore {
    core: OuterF2Core<SyntheticTrace>,
    cycles: usize,
    threads: usize,
    rounds: [u128; 40],
    finish: u128,
}

impl ProveRounds<F128> for TimedCore {
    fn num_rounds(&self) -> usize {
        self.core.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        previous_claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        let start = Instant::now();
        let message = self.core.prove_round(bind, round, previous_claim)?;
        if let Some(time) = self.rounds.get_mut(round) {
            *time = start.elapsed().as_nanos();
        }
        Ok(message)
    }

    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        let start = Instant::now();
        self.core.finish_rounds(bind)?;
        self.finish = start.elapsed().as_nanos();
        Ok(())
    }
}

struct PhaseSample {
    times: [u128; 40],
    finish: u128,
    phases: Option<[u128; 4]>,
}

#[derive(Default)]
struct Reports {
    samples: BTreeMap<(usize, usize), Vec<PhaseSample>>,
}

impl Reports {
    fn record(&mut self, core: &TimedCore) {
        self.samples
            .entry((core.cycles.ilog2() as usize, core.threads))
            .or_default()
            .push(PhaseSample {
                times: core.rounds,
                finish: core.finish,
                phases: None,
            });
    }

    fn print(&self, variant: &str, options: OuterF2Options) {
        for (&(log_t, threads), samples) in &self.samples {
            let divisor = (1_usize << log_t) as f64;
            let prefix = format!("{variant}/local/{log_t}/{threads}");
            let models = [
                6.9,
                15.1,
                if options.monomial_rounds == 2 {
                    74.0
                } else {
                    36.6
                },
                40.8,
                32.0,
                27.6,
            ];
            for round in 0..8 + log_t {
                report(
                    &format!("{prefix}/round{}", round + 1),
                    samples.iter().map(|sample| sample.times[round]).collect(),
                    divisor,
                    models.get(round).copied(),
                );
            }
            report(
                &format!("{prefix}/materialisation_groups"),
                samples
                    .iter()
                    .map(|sample| sample.times[6] + sample.times[7])
                    .collect(),
                divisor,
                Some(37.21),
            );
            report(
                &format!("{prefix}/cycles"),
                samples
                    .iter()
                    .map(|sample| sample.times[8..8 + log_t].iter().sum::<u128>() + sample.finish)
                    .collect(),
                divisor,
                Some(18.09),
            );
            for (phase, name) in ["construct", "rounds", "finish", "extract"]
                .into_iter()
                .enumerate()
            {
                let values = samples
                    .iter()
                    .filter_map(|sample| sample.phases.map(|times| times[phase]))
                    .collect::<Vec<_>>();
                if !values.is_empty() {
                    report(&format!("{prefix}/{name}"), values, divisor, None);
                }
            }
            let totals = samples
                .iter()
                .filter_map(|sample| sample.phases.map(|times| times[..3].iter().sum()))
                .collect::<Vec<_>>();
            if !totals.is_empty() {
                report(&format!("{prefix}/total_core"), totals, divisor, None);
            }
        }
    }
}

#[expect(
    clippy::print_stdout,
    reason = "sample distributions are benchmark output"
)]
fn report(label: &str, values: Vec<u128>, divisor: f64, model: Option<f64>) {
    let samples = values
        .iter()
        .map(|value| format!("{:.6}", *value as f64 / divisor))
        .collect::<Vec<_>>()
        .join(",");
    let mut sorted = values;
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] as f64).midpoint(sorted[middle] as f64) / divisor
    } else {
        sorted[middle] as f64 / divisor
    };
    let min = sorted.first().copied().unwrap_or(0) as f64 / divisor;
    let max = sorted.last().copied().unwrap_or(0) as f64 / divisor;
    print!("{label} ns={median:.6} min_ns={min:.6} max_ns={max:.6} samples={} sample_ns={samples} loaded_machine=true", sorted.len());
    if let Some(model) = model {
        print!(" model_point1_ns={model:.2}");
    }
    println!();
}

#[expect(
    clippy::print_stdout,
    reason = "thresholds and phase attribution are benchmark output"
)]
fn main() -> Result<(), RunnerError> {
    if std::env::args().any(|argument| argument == "--compare-monomial" || argument == "--variants")
    {
        return Comparison::parse()?.run();
    }
    println!(
        "outer_f2/local loaded_machine=true threshold_1_thread_ns=268 threshold_12_threads_log22_ns=28 model_point1_ns=214.3 model_point2_ns=173.9 tail_histogram_in_round1=true materialisation_in_round7=true"
    );
    let default = OuterF2Options::default();
    let variants = [
        ("outer_f2", default),
        (
            "outer_f2/nibble_round_2",
            OuterF2Options {
                nibble_round_2: true,
                ..default
            },
        ),
        (
            "outer_f2/unfolded_group_weights",
            OuterF2Options {
                folded_group_weights: false,
                ..default
            },
        ),
        (
            "outer_f2/monomial_rounds4",
            OuterF2Options {
                monomial_rounds: 4,
                ..default
            },
        ),
        (
            "outer_f2/monomial_rounds5",
            OuterF2Options {
                monomial_rounds: 5,
                ..default
            },
        ),
        (
            "outer_f2/monomial_rounds6",
            OuterF2Options {
                monomial_rounds: 6,
                ..default
            },
        ),
    ];
    for (variant, options) in variants {
        let reports = Mutex::new(Reports::default());
        run_core(
            variant,
            &[SynthProfile::Local],
            |source| {
                let cycles = source.cycles();
                let mut rng = ChaCha20Rng::seed_from_u64(0x7461_755f_6f75_7465);
                let tau = (0..8 + cycles.ilog2() as usize)
                    .map(|_| F128::random(&mut rng))
                    .collect::<Vec<_>>();
                let core =
                    OuterF2Core::new(Arc::clone(&source), &tau, options).map_err(|error| {
                        RunnerError::Core {
                            message: error.to_string(),
                        }
                    })?;
                Ok::<_, RunnerError>((
                    TimedCore {
                        core,
                        cycles,
                        threads: rayon::current_num_threads(),
                        rounds: [0; 40],
                        finish: 0,
                    },
                    F128::from_raw(0),
                ))
            },
            |core, _point| {
                reports
                    .lock()
                    .map_err(|error| RunnerError::Core {
                        message: error.to_string(),
                    })?
                    .record(core);
                Ok::<_, RunnerError>(core.core.final_values())
            },
        )?;
        reports
            .into_inner()
            .map_err(|error| RunnerError::Core {
                message: error.to_string(),
            })?
            .print(variant, options);
    }
    Ok(())
}

struct Comparison {
    log_t: Vec<usize>,
    threads: Vec<usize>,
    samples: usize,
    variants: Vec<usize>,
}

impl Comparison {
    fn parse() -> Result<Self, RunnerError> {
        let mut result = Self {
            log_t: vec![20, 22],
            threads: vec![1],
            samples: 5,
            variants: vec![2, 3],
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            if ["--bench", "--compare-monomial"].contains(&argument.as_str()) {
                continue;
            }
            if !["--log-t", "--threads", "--samples", "--variants"].contains(&argument.as_str()) {
                return Err(RunnerError::UnknownArgument { argument });
            }
            let value = arguments.next().ok_or_else(|| RunnerError::MissingValue {
                option: argument.clone(),
            })?;
            let invalid = || RunnerError::InvalidValue {
                option: argument.clone(),
                value: value.clone(),
            };
            if argument == "--variants" {
                result.variants = value
                    .split(',')
                    .map(|variant| match variant {
                        "monomial_rounds2" => Ok(2),
                        "monomial_rounds3" => Ok(3),
                        _ => Err(invalid()),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if result.variants.len() > 2
                    || result.variants.windows(2).any(|pair| pair[0] == pair[1])
                {
                    return Err(invalid());
                }
                continue;
            }
            let parsed = value
                .split(',')
                .map(|entry| entry.parse::<usize>().map_err(|_| invalid()))
                .collect::<Result<Vec<_>, _>>()?;
            match argument.as_str() {
                "--log-t" if parsed.iter().all(|log_t| (1..=32).contains(log_t)) => {
                    result.log_t = parsed;
                }
                "--threads" if parsed.iter().all(|threads| *threads > 0) => {
                    result.threads = parsed;
                }
                "--samples" if parsed.len() == 1 && parsed[0] >= 5 => {
                    result.samples = parsed[0];
                }
                _ => return Err(invalid()),
            }
        }
        Ok(result)
    }

    #[expect(
        clippy::print_stdout,
        reason = "paired order and phase attribution are benchmark output"
    )]
    fn run(&self) -> Result<(), RunnerError> {
        println!("outer_f2/compare_monomial loaded_machine=true folded_group_weights=true nibble_round_2=false alternating_sample_order=true samples={} histogram_in_round1=true materialisation_in_round7=true cycle_model_includes_final_row_bind=true", self.samples);
        let pools = self
            .threads
            .iter()
            .map(|&threads| {
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map(|pool| (threads, pool))
                    .map_err(|error| RunnerError::ThreadPool {
                        message: error.to_string(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (threads, pool) in &pools {
            let _ = pool.broadcast(|_| black_box(()));
            for &log_t in &self.log_t {
                let source = Arc::new(
                    pool.install(|| {
                        SyntheticTrace::new(SynthProfile::Local, log_t, 1 << 20, 0x5eed)
                    })
                    .map_err(|error| RunnerError::Trace {
                        message: error.to_string(),
                    })?,
                );
                let mut rng = ChaCha20Rng::seed_from_u64(0x7461_755f_6f75_7465);
                let tau = (0..8 + log_t)
                    .map(|_| F128::random(&mut rng))
                    .collect::<Vec<_>>();
                let mut rng = ChaCha20Rng::seed_from_u64(0x726f_756e_6473);
                let challenges: [F128; 256] = std::array::from_fn(|_| F128::random(&mut rng));
                let mut reports: [Reports; 2] = std::array::from_fn(|_| Reports::default());
                for sample in 0..self.samples {
                    for offset in 0..self.variants.len() {
                        let index = if sample.is_multiple_of(2) {
                            offset
                        } else {
                            self.variants.len() - 1 - offset
                        };
                        let monomial_rounds = self.variants[index];
                        let options = OuterF2Options {
                            monomial_rounds,
                            nibble_round_2: false,
                            folded_group_weights: true,
                        };
                        let recorded = pool.install(|| {
                            let start = Instant::now();
                            let core = OuterF2Core::new(Arc::clone(&source), &tau, options)
                                .map_err(|error| RunnerError::Core {
                                    message: error.to_string(),
                                })?;
                            let mut core = TimedCore {
                                core,
                                cycles: source.cycles(),
                                threads: *threads,
                                rounds: [0; 40],
                                finish: 0,
                            };
                            let construct = start.elapsed().as_nanos();
                            let start = Instant::now();
                            let mut claim = F128::from_raw(0);
                            let mut bind = None;
                            for (round, &challenge) in
                                challenges[..core.num_rounds()].iter().enumerate()
                            {
                                let message = core.prove_round(bind, round, claim)?;
                                claim = message.evaluate(challenge);
                                bind = Some(challenge);
                                let _ = black_box(&message);
                            }
                            let rounds = start.elapsed().as_nanos();
                            let start = Instant::now();
                            if let Some(challenge) = bind {
                                core.finish_rounds(challenge)?;
                            }
                            let finish = start.elapsed().as_nanos();
                            let start = Instant::now();
                            let _ = black_box(core.core.final_values());
                            let extract = start.elapsed().as_nanos();
                            Ok::<_, RunnerError>(PhaseSample {
                                times: core.rounds,
                                finish: core.finish,
                                phases: Some([construct, rounds, finish, extract]),
                            })
                        })?;
                        reports[monomial_rounds - 2]
                            .samples
                            .entry((log_t, *threads))
                            .or_default()
                            .push(recorded);
                    }
                }
                for &monomial_rounds in &self.variants {
                    reports[monomial_rounds - 2].print(
                        &format!("outer_f2/monomial_rounds{monomial_rounds}"),
                        OuterF2Options {
                            monomial_rounds,
                            nibble_round_2: false,
                            folded_group_weights: true,
                        },
                    );
                }
            }
        }
        Ok(())
    }
}
