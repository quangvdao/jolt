pub mod support;

use jolt_field::{Field, F128};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_rv64i_kernels::{packed, par, round};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use std::mem::size_of;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use support::{run_core, RunnerError};

// Including the canonical implementation gives this bench access to its private
// phase boundaries without adding a timing API to the library.
mod measured {
    include!("../src/column_pass.rs");

    use super::{Arc, Field, Instant, Record, SyntheticTrace, RECORDS};
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

        #[expect(
            clippy::expect_used,
            reason = "a poisoned benchmark telemetry lock means an earlier sample panicked"
        )]
        pub(super) fn extract(&self) -> [F128; 256] {
            let start = Instant::now();
            self.prepared.pass(self.source.rows());
            let pass = start.elapsed().as_nanos() as f64;
            let start = Instant::now();
            let mut sums = self.prepared.merge();
            let merge = start.elapsed().as_nanos() as f64;
            let start = Instant::now();
            let columns = read_columns(&mut sums);
            let read = start.elapsed().as_nanos() as f64;
            RECORDS
                .lock()
                .expect("benchmark telemetry lock")
                .push(Record {
                    log_t: self.source.rows().len().ilog2() as usize,
                    threads: rayon::current_num_threads(),
                    times: [pass, merge, read],
                });
            columns
        }
    }

    pub(super) fn warm_up() -> Result<[F128; 256], ColumnPassError> {
        column_pass(&[[0; 4]], &[])
    }
}

use measured::{ColumnPassError, Pass};

struct Record {
    log_t: usize,
    threads: usize,
    times: [f64; 3],
}

static RECORDS: Mutex<Vec<Record>> = Mutex::new(Vec::new());

impl ProveRounds<F128> for Pass {
    fn num_rounds(&self) -> usize {
        0
    }

    fn prove_round(
        &mut self,
        _bind: Option<F128>,
        round: usize,
        _previous_claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        Err(SumcheckError::WrongNumberOfRounds {
            expected: 0,
            got: round,
        })
    }

    fn finish_rounds(&mut self, _bind: F128) -> Result<(), SumcheckError<F128>> {
        Ok(())
    }
}

fn phase_median(records: &[Record], phase: usize) -> f64 {
    let mut values: Vec<_> = records.iter().map(|record| record.times[phase]).collect();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        values[middle - 1].midpoint(values[middle])
    } else {
        values[middle]
    }
}

#[expect(
    clippy::print_stdout,
    clippy::expect_used,
    reason = "benchmark output reports phase medians; telemetry poisoning identifies a failed sample"
)]
fn main() -> Result<(), RunnerError> {
    let _ = measured::warm_up().map_err(|error| RunnerError::Core {
        message: error.to_string(),
    })?;
    RECORDS
        .lock()
        .expect("benchmark telemetry lock")
        .reserve(4096);
    run_core(
        "column_pass",
        &[SynthProfile::Local],
        |source| {
            let core = Pass::new(source)?;
            Ok::<_, ColumnPassError>((core, F128::from_raw(0)))
        },
        |core, _point| Ok::<_, ColumnPassError>(core.extract()),
    )?;
    let mut records = RECORDS.lock().expect("benchmark telemetry lock");
    records.sort_unstable_by_key(|record| (record.threads, record.log_t));
    let mut remaining = records.as_slice();
    while let Some(first) = remaining.first() {
        let count = remaining.partition_point(|record| {
            record.log_t == first.log_t && record.threads == first.threads
        });
        let (group, rest) = remaining.split_at(count);
        let cycles = (1_usize << first.log_t) as f64;
        let phases = std::array::from_fn::<_, 3, _>(|phase| phase_median(group, phase) / cycles);
        let threshold = if first.threads == 1 { 26.0 } else { 26.0 / 9.6 };
        println!(
            "column_pass_parts/local/{}/{} pass_ns={:.6} merge_ns={:.6} readout_ns={:.6} bucket_bytes_per_worker={} samples={} model_pass_ns={} threshold_ns={threshold:.6} loaded_machine=true",
            first.log_t, first.threads, phases[0], phases[1], phases[2],
            8192 * size_of::<F128>(), group.len(), 21.0 / first.threads as f64,
        );
        remaining = rest;
    }
    Ok(())
}
