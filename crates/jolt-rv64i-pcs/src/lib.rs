//! Prover algorithms for the binary-field RV64I bit-table commitment scheme.
//! Shared protocol geometry and validation belong to `jolt_rv64i_verifier::whir`.

#![deny(unsafe_code)]

#[cfg(feature = "arch")]
#[expect(
    unsafe_code,
    reason = "architecture intrinsics are confined to this module"
)]
mod arch;
#[forbid(unsafe_code)]
pub mod merkle;
