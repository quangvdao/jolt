//! Local-profile outer sum-check timings, including the option matrix and each
//! round. Paired monomial cases use `--compare-monomial --samples 5` or
//! `--variants monomial_rounds2,monomial_rounds3`; the runner interleaves cases.

pub mod support;

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use jolt_field::{Field, F128};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_kernels::outer_f2::{OuterF2Core, OuterF2Options};
use jolt_rv64i_kernels::source::CycleSource;
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

use support::{core_rounds, run_cases, Case, Clock, RunnerError};

// Columns follow the spec's M, A, R, L, Bk, w. The last two rows charge
// the row binds to the passes that execute them, rather than to their messages.
const DEFAULT_MODEL_COUNTS: [[f64; 6]; 8] = [
    [0.0, 1.0, 0.0, 8.0, 1.0, 40.0],
    [0.0, 2.0, 0.0, 16.0, 0.0, 130.0],
    [0.0, 2.0, 0.0, 56.0, 0.0, 240.0],
    [0.0, 18.0, 2.0, 40.0, 0.0, 0.0],
    [0.0, 10.0, 2.0, 40.0, 0.0, 0.0],
    [0.0, 6.0, 2.0, 40.0, 0.0, 0.0],
    [7.0, 4.0, 0.0, 50.0, 0.0, 0.0],
    [8.0, 2.0, 0.0, 3.0, 0.0, 0.0],
];
const POINT1_PRICES: [f64; 6] = [1.83, 1.10, 0.7, 0.4, 0.6, 0.05];
const POINT2_PRICES: [f64; 6] = [0.9, 0.7, 0.2, 0.4, 0.6, 0.05];

fn model_price(counts: &[f64; 6], prices: &[f64; 6]) -> f64 {
    counts
        .iter()
        .zip(prices)
        .map(|(count, price)| count * price)
        .sum()
}

struct TimedCore {
    core: OuterF2Core<SyntheticTrace>,
    rounds: [Duration; 40],
    finish: Duration,
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
        let clock = Clock::start();
        let message = self.core.prove_round(bind, round, previous_claim)?;
        if let Some(time) = self.rounds.get_mut(round) {
            *time = clock.elapsed();
        }
        Ok(message)
    }

    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        let clock = Clock::start();
        self.core.finish_rounds(bind)?;
        self.finish = clock.elapsed();
        Ok(())
    }
}

struct Fixture {
    source: Arc<SyntheticTrace>,
    comparison_tau: Option<Vec<F128>>,
}

impl Fixture {
    fn tau(source: &SyntheticTrace) -> Vec<F128> {
        let mut rng = ChaCha20Rng::seed_from_u64(0x7461_755f_6f75_7465);
        (0..8 + source.cycles().ilog2() as usize)
            .map(|_| F128::random(&mut rng))
            .collect()
    }
}

fn variants(args: &[String]) -> Result<(bool, Vec<(String, OuterF2Options)>), RunnerError> {
    let comparison = args
        .iter()
        .any(|arg| arg == "--compare-monomial" || arg == "--variants");
    let default = OuterF2Options::default();
    if comparison {
        let rounds = if let Some(index) = args.iter().position(|arg| arg == "--variants") {
            let value = args
                .get(index + 1)
                .ok_or_else(|| RunnerError::MissingValue {
                    option: "--variants".to_owned(),
                })?;
            value
                .split(',')
                .map(|name| match name {
                    "monomial_rounds2" => Ok(2),
                    "monomial_rounds3" => Ok(3),
                    _ => Err(RunnerError::InvalidValue {
                        option: "--variants".to_owned(),
                        value: value.clone(),
                    }),
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![2, 3]
        };
        return Ok((
            true,
            rounds
                .into_iter()
                .map(|monomial_rounds| {
                    (
                        format!("outer_f2/monomial_rounds{monomial_rounds}"),
                        OuterF2Options {
                            monomial_rounds,
                            nibble_round_2: false,
                            folded_group_weights: true,
                        },
                    )
                })
                .collect(),
        ));
    }
    let other_rounds = if default.monomial_rounds == 3 { 2 } else { 3 };
    let mut result = vec![
        ("outer_f2".to_owned(), default),
        (
            format!("outer_f2/monomial_rounds{other_rounds}"),
            OuterF2Options {
                monomial_rounds: other_rounds,
                ..default
            },
        ),
        (
            "outer_f2/nibble_round_2".to_owned(),
            OuterF2Options {
                nibble_round_2: true,
                ..default
            },
        ),
        (
            "outer_f2/unfolded_group_weights".to_owned(),
            OuterF2Options {
                folded_group_weights: false,
                ..default
            },
        ),
    ];
    result.extend((4..=6).map(|monomial_rounds| {
        (
            format!("outer_f2/monomial_rounds{monomial_rounds}"),
            OuterF2Options {
                monomial_rounds,
                ..default
            },
        )
    }));
    Ok((false, result))
}

#[expect(
    clippy::print_stdout,
    reason = "thresholds and phase attribution are benchmark output"
)]
fn main() -> Result<(), RunnerError> {
    let (comparison, variants) = variants(&std::env::args().collect::<Vec<_>>())?;
    let model_point1 = DEFAULT_MODEL_COUNTS
        .iter()
        .map(|counts| model_price(counts, &POINT1_PRICES))
        .sum::<f64>();
    let model_point2 = DEFAULT_MODEL_COUNTS
        .iter()
        .map(|counts| model_price(counts, &POINT2_PRICES))
        .sum::<f64>();
    println!("outer_f2/local loaded_machine=true threshold_1_thread_ns=268 threshold_12_threads_log22_ns=28 spec_threshold_1_thread_ns=254 spec_threshold_12_threads_log22_ns=26 model_point1_ns={model_point1:.1} model_point2_ns={model_point2:.1} tail_histogram_in_round1=true materialisation_in_round7=true materialisation_row7_bind=rounds7_8 row8_bind_cycles=round9_through_finish timing_includes_allocation_first_touch_parallel_overhead=true");
    if comparison {
        println!("outer_f2/compare_monomial loaded_machine=true folded_group_weights=true nibble_round_2=false alternating_sample_order=true histogram_in_round1=true materialisation_in_round7=true cycle_model_includes_final_row_bind=true materialisation_row7_bind=rounds7_8 row8_bind_cycles=round9_through_finish timing_includes_allocation_first_touch_parallel_overhead=true");
    }
    let phase_names = (1..=40)
        .map(|round| format!("round{round}"))
        .chain([
            "materialisation_row7_bind".to_owned(),
            "row8_bind_cycles".to_owned(),
            "total_core".to_owned(),
        ])
        .collect::<Vec<_>>();
    let extra = phase_names.iter().map(String::as_str).collect::<Vec<_>>();
    let cases = variants
        .into_iter()
        .map(|(name, options)| {
            let mut case = Case::core(&name, options, &extra);
            if name == "outer_f2" {
                case.threshold = Some(|log_t, threads| {
                    if threads == 12 && log_t == 22 {
                        26.0
                    } else {
                        254.0
                    }
                });
            }
            case
        })
        .collect::<Vec<_>>();
    let _ = run_cases(
        &[SynthProfile::Local],
        &cases,
        |source| {
            let comparison_tau = comparison.then(|| Fixture::tau(&source));
            Ok::<_, RunnerError>(Fixture {
                source,
                comparison_tau,
            })
        },
        |fixture, &options, challenges, times| {
            let clock = Clock::start();
            let core = if let Some(tau) = &fixture.comparison_tau {
                OuterF2Core::new(Arc::clone(&fixture.source), tau, options)
            } else {
                let tau = Fixture::tau(&fixture.source);
                OuterF2Core::new(Arc::clone(&fixture.source), &tau, options)
            }
            .map_err(|error| RunnerError::Core {
                message: error.to_string(),
            })?;
            let mut core = TimedCore {
                core,
                rounds: [Duration::ZERO; 40],
                finish: Duration::ZERO,
            };
            let construct = clock.elapsed();
            times.set(0, construct);
            let rounds = core_rounds(&mut core, F128::from_raw(0), challenges)?;
            times.set(1, rounds.rounds);
            times.set(2, rounds.finish);
            times.set(46, construct + rounds.rounds + rounds.finish);
            let clock = Clock::start();
            let values = core.core.final_values();
            let _ = black_box(&values);
            times.set(3, clock.elapsed());
            Ok((core, values))
        },
        |(core, _), times, _| {
            for (round, &duration) in core.rounds.iter().enumerate() {
                times.set(4 + round, duration);
            }
            times.set(44, core.rounds[6] + core.rounds[7]);
            times.set(
                45,
                core.rounds[8..core.num_rounds()].iter().sum::<Duration>() + core.finish,
            );
        },
        |record, _, &options| {
            let default = OuterF2Options::default();
            let is_default = options.monomial_rounds == default.monomial_rounds
                && options.nibble_round_2 == default.nibble_round_2
                && options.folded_group_weights == default.folded_group_weights;
            let models = DEFAULT_MODEL_COUNTS.map(|counts| model_price(&counts, &POINT1_PRICES));
            for round in 0..8 + record.log_t {
                record.print_phase(
                    &format!("{}/round{}", record.id, round + 1),
                    4 + round,
                    if is_default && round < 6 {
                        models.get(round).copied()
                    } else {
                        None
                    },
                );
            }
            record.print_phase(
                &format!("{}/materialisation_row7_bind", record.id),
                44,
                is_default.then_some(models[6]),
            );
            record.print_phase(
                &format!("{}/row8_bind_cycles", record.id),
                45,
                is_default.then_some(models[7]),
            );
            if comparison {
                for (phase, name) in ["construct", "rounds", "finish", "extract"]
                    .into_iter()
                    .enumerate()
                {
                    record.print_phase(&format!("{}/{name}", record.id), phase, None);
                }
                record.print_phase(&format!("{}/total_core", record.id), 46, None);
            }
        },
    )?;
    Ok(())
}
