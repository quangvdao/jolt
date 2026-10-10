//! Trace-to-witness measurements on the executed adapter fixture.
//! Run: `cargo bench -p jolt-rv64i-prover --features test-utils
//! --bench witness_pipeline -- --log-t 20,22 --threads 1,12 --samples 5`.
//! `inventory` lists requests of at least T bytes and fails on recorder overflow.
//! Validation-only is diagnostic: preparation validates again while gathering
//! groups. The production total excludes validation-only. Scatter is lazy in
//! the session and timed separately, with preparation-plus-scatter also reported.
//! Setup, statement admission, warmed pools, and destruction are outside timers.

pub mod support;

use support::{pipelines::BenchResult, runner::Options};

fn main() -> BenchResult<()> {
    support::runner::run_witness(Options::parse_witness()?)
}
