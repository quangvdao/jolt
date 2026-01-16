//! Grumpkin MSM inline implementation module.
//!
//! NOTE: This is currently implemented as a **trace-only precompile**: during tracing,
//! the host computes the result(s) and injects them via `VirtualAdvice` instructions,
//! then the inline sequence stores the result into guest memory.
//!
//! This is useful for cycle profiling and plumbing. It is **NOT** suitable for proving
//! unless additional constraints are added to verify the relations.

#![cfg_attr(not(feature = "host"), no_std)]

pub const INLINE_OPCODE: u32 = 0x0B;

/// funct7 namespace: 0x00..0x04 are taken by sha2/keccak/blake2/blake3/bigint.
pub const GRUMPKIN_FUNCT7: u32 = 0x05;

/// MSM(2048) oracle precompile (trace-only).
pub const GRUMPKIN_MSM2048_FUNCT3: u32 = 0x00;
pub const GRUMPKIN_MSM2048_NAME: &str = "GRUMPKIN_MSM2048_INLINE";

/// Jacobian point doubling: `out = 2 * in` (trace-only oracle for now).
pub const GRUMPKIN_DOUBLE_JAC_FUNCT3: u32 = 0x01;
pub const GRUMPKIN_DOUBLE_JAC_NAME: &str = "GRUMPKIN_DOUBLE_JAC_INLINE";

/// Mixed addition: `out = jac + aff` (trace-only oracle for now).
pub const GRUMPKIN_ADD_MIXED_FUNCT3: u32 = 0x02;
pub const GRUMPKIN_ADD_MIXED_NAME: &str = "GRUMPKIN_ADD_MIXED_INLINE";

pub mod types;

pub mod sdk;
pub use sdk::*;

#[cfg(feature = "host")]
pub mod sequence_builder;

#[cfg(feature = "host")]
mod host;
#[cfg(feature = "host")]
pub use host::*;

#[cfg(all(test, feature = "host"))]
mod curve_ops_tests;

