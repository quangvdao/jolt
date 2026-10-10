pub mod support;

use jolt_field::{Field, F128};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_rv64i_kernels::{packed, par, round};
use std::hint::black_box;
use std::mem::size_of;
use std::sync::Arc;
use support::{run_cases, Case, Clock, PhaseTimes, RunnerError};

// Including the canonical implementation gives this bench access to its private
// phase boundaries without adding a timing API to the library.
mod measured {
    include!("../src/column_pass.rs");

    use super::{Arc, Clock, Field, PhaseTimes, SyntheticTrace};
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    pub(super) struct Pass {
        prepared: Prepared,
        source: Arc<SyntheticTrace>,
    }

    impl Pass {
        pub(super) fn new(source: Arc<SyntheticTrace>) -> Result<Self, ColumnPassError> {
            let mut rng = ChaCha20Rng::seed_from_u64(0xc011_5eed);
            let point: Vec<_> = (0..source.rows().len().ilog2())
                .map(|_| F128::random(&mut rng))
                .collect();
            let prepared = Prepared::new(source.rows(), &point)?;
            Ok(Self { prepared, source })
        }

        pub(super) fn extract(&self, times: &mut PhaseTimes) -> [F128; 256] {
            let clock = Clock::start();
            self.prepared.pass(self.source.rows());
            times.set(4, clock.elapsed());
            let clock = Clock::start();
            let mut sums = self.prepared.merge();
            times.set(5, clock.elapsed());
            let clock = Clock::start();
            let columns = read_columns(&mut sums);
            times.set(6, clock.elapsed());
            columns
        }
    }

    pub(super) fn warm_up() -> Result<[F128; 256], ColumnPassError> {
        column_pass(&[[0; 4]], &[])
    }
}

use measured::Pass;

fn threshold(_: usize, threads: usize) -> f64 {
    if threads == 1 {
        26.0
    } else {
        26.0 / 9.6
    }
}

#[expect(
    clippy::print_stdout,
    reason = "phase diagnostics are benchmark output"
)]
fn main() -> Result<(), RunnerError> {
    let _ = measured::warm_up().map_err(|error| RunnerError::Core {
        message: error.to_string(),
    })?;
    let mut case = Case::core("column_pass", (), &["pass", "merge", "readout"]);
    case.threshold = Some(threshold);
    let _ = run_cases(
        &[SynthProfile::Local],
        &[case],
        Ok::<_, RunnerError>,
        |source, (), _, times| {
            let clock = Clock::start();
            let pass = Pass::new(Arc::clone(source)).map_err(|error| RunnerError::Core {
                message: error.to_string(),
            })?;
            times.set(0, clock.elapsed());
            let clock = Clock::start();
            let columns = pass.extract(times);
            let _ = black_box(&columns);
            times.set(3, clock.elapsed());
            Ok((pass, columns))
        },
        |_, _| {},
        |record, _, ()| {
            println!("column_pass_parts/local/{}/{} pass_ns={:.6} merge_ns={:.6} readout_ns={:.6} bucket_bytes_per_worker={} samples={} model_pass_ns={} threshold_ns={:.6} loaded_machine=true", record.log_t, record.threads,
                record.phases[4].median, record.phases[5].median, record.phases[6].median,
                8192 * size_of::<F128>(), record.samples, 21.0 / record.threads as f64,
                threshold(record.log_t, record.threads));
        },
    )?;
    Ok(())
}
