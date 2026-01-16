//! BN254 Dory GT exponentiation inline implementation module (skeleton).
//!
//! This crate provides:
//! - a guest-side SDK wrapper that emits the inline instruction
//! - a host-side registration hook that wires the inline into the tracer
//! - a (placeholder) sequence builder that will be replaced by a real GT_EXP inline

#![cfg_attr(not(feature = "host"), no_std)]

pub const INLINE_OPCODE: u32 = 0x0B;

/// BN254-family inlines (reserved for this workstream).
pub const BN254_FUNCT7: u32 = 0x06;

/// BN254_GT_EXP: exponentiation in GT (Fq12).
pub const BN254_GT_EXP_FUNCT3: u32 = 0x02;
pub const BN254_GT_EXP_FUNCT7: u32 = BN254_FUNCT7;
pub const BN254_GT_EXP_NAME: &str = "BN254_GT_EXP_INLINE";

/// Limb sizes for the ABI contract.
pub const GT_LIMBS_U64: usize = 48; // Fq12 = 12 * Fq; Fq = 4 u64 limbs => 48 u64 limbs
pub const FR_LIMBS_U64: usize = 4;  // Fr = 256-bit => 4 u64 limbs

pub mod sdk;
pub use sdk::*;

#[cfg(feature = "host")]
pub mod sequence_builder;

#[cfg(feature = "host")]
mod host;
#[cfg(feature = "host")]
pub use host::*;
