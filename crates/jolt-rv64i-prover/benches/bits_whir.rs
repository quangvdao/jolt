//! WHIR's Performance phase table and allocation lifetimes (§9).
//! Run: `RUSTFLAGS='-C target-cpu=native' cargo bench -p jolt-rv64i-prover
//! --features test-utils,jolt-rv64i-pcs/arch --bench bits_whir --
//! --log-t 20,22 --threads 1,12 --samples 5 --proof-samples 1000`.
//! Cases are `bits_whir/commit/<log-t>/<threads>` and `bits_whir/open/...`.
//! `--proof-samples 0` runs phase measurements only, without size acceptance.
//! Seeded rows and warmed pools precede the combined commit/open allocation
//! interval. Domain constants and bridge setup remain inside the timed phases.
//! Proof-size samples are independent honest proofs, each verified, rather
//! than simulated query paths. Load is recorded around every measured sample.

pub mod support;

use std::error::Error;
use support::whir_runner::Options;

fn main() -> Result<(), Box<dyn Error>> {
    support::whir_runner::run(Options::parse()?)
}
