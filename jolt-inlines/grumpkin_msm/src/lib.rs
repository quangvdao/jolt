//! Provable (sound) Grumpkin MSM implementation for Jolt's recursion verifier hot-path (Hyrax).
//!
//! This crate provides:
//! - stable `#[repr(C)]` limb layouts for points/scalars (`types`)
//! - a pure-Rust MSM implementation over Grumpkin (`provable`)

#![no_std]

extern crate alloc;

/// Fixed MSM size used by Hyrax in the recursion verifier.
pub const MSM_N: usize = 2048;

pub mod provable;
pub mod types;

pub use provable::grumpkin_msm_2048;
