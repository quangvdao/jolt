//! Local-profile outer sum-check timings, including the option matrix and the
//! median wall time of each round. Use `--samples 3` or more on a loaded machine.

pub mod support;

use std::collections::BTreeMap;
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
            });
    }

    #[expect(
        clippy::print_stdout,
        reason = "per-round medians and model comparisons are benchmark output"
    )]
    fn print(&self, variant: &str) {
        for (&(log_t, threads), samples) in &self.samples {
            let divisor = (1_usize << log_t) as f64;
            let median = |values: Vec<u128>| {
                let mut values = values;
                values.sort_unstable();
                let middle = values.len() / 2;
                if values.len().is_multiple_of(2) {
                    (values[middle - 1] as f64).midpoint(values[middle] as f64) / divisor
                } else {
                    values[middle] as f64 / divisor
                }
            };
            let models = [6.9, 15.1, 36.6, 40.8, 32.0, 27.6];
            for (round, model) in models.into_iter().enumerate() {
                let time = median(samples.iter().map(|sample| sample.times[round]).collect());
                println!(
                    "{variant}/local/{log_t}/{threads}/round{} ns={time:.6} model_point1_ns={model:.1} samples={} loaded_machine=true histogram_in_round1=true",
                    round + 1,
                    samples.len(),
                );
            }
            for round in 6..8 + log_t {
                let time = median(samples.iter().map(|sample| sample.times[round]).collect());
                println!(
                    "{variant}/local/{log_t}/{threads}/round{} ns={time:.6} samples={} loaded_machine=true materialisation_in_round7=true",
                    round + 1,
                    samples.len(),
                );
            }
            let materialisation_groups = median(
                samples
                    .iter()
                    .map(|sample| sample.times[6] + sample.times[7])
                    .collect(),
            );
            let cycles = median(
                samples
                    .iter()
                    .map(|sample| sample.times[8..8 + log_t].iter().sum::<u128>() + sample.finish)
                    .collect(),
            );
            let finish = median(samples.iter().map(|sample| sample.finish).collect());
            println!(
                "{variant}/local/{log_t}/{threads}/materialisation_groups ns={materialisation_groups:.6} model_point1_ns=43.9 histogram_measured_in_round1=true samples={} loaded_machine=true",
                samples.len(),
            );
            println!(
                "{variant}/local/{log_t}/{threads}/cycles ns={cycles:.6} model_point1_ns=11.4 finish_ns={finish:.6} samples={} loaded_machine=true",
                samples.len(),
            );
        }
    }
}

#[expect(
    clippy::print_stdout,
    reason = "thresholds and phase attribution are benchmark output"
)]
fn main() -> Result<(), RunnerError> {
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
            "outer_f2/folded_group_weights",
            OuterF2Options {
                folded_group_weights: true,
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
            .print(variant);
    }
    Ok(())
}
