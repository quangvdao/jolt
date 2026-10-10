//! An experiment in RV64I hash-based Jolt over binary fields.
//! The plane owns packed witnesses; reference kernels form a test oracle,
//! optimized kernels attach to the same plane, and stages own kernel registries.
//! Commitment and prover modules orchestrate proving. Points and indices are
//! least-significant-bit first; sum-check binds low to high, with columns before
//! cycles in the Bits table.

#![forbid(unsafe_code)]
#![expect(
    non_snake_case,
    reason = "protocol dimensions retain their mathematical names"
)]

pub mod backend;
pub mod commitment;
pub mod error;
pub mod optimized;
pub mod plane;
pub mod prover;
pub mod reference;
pub mod stages;
