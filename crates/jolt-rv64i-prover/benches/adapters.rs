//! Run with `RUSTFLAGS='-C target-cpu=native' cargo bench -p jolt-rv64i-prover
//! --features test-utils --bench adapters -- adapters/session/20/1`.
//! With no id filter, sizes 20,22 and pools 1,12 run in sample rotation.
//! Options: `--log-t 20,22 --threads 1,12 --samples 3`.
//! Kernel hooks time preparation, rounds, terminal binds, validation/extraction,
//! and parking. Batch bookkeeping (transcript, point derivation, final-claim
//! comparison) cannot be separated through the driver; it enters extract and
//! is additionally reported as driver_ns. That allocation phase is driver.
//! All figures are loaded-machine evidence, including missed thresholds.

pub mod support;

use std::error::Error;
use support::runner::Options;

fn main() -> Result<(), Box<dyn Error>> {
    support::runner::run(Options::parse()?)
}
