//! Generic MSM (multi-scalar multiplication) algorithms for zkVM guest code.
//!
//! This crate provides curve-agnostic MSM implementations that work with any
//! elliptic curve group implementing the [`MsmGroup`] trait.
//!
//! # Algorithms
//!
//! - **Pippenger MSM**: Variable-base multi-scalar multiplication using the
//!   bucket method with configurable window sizes.
//! - **GLV-accelerated MSM**: For curves with an efficient endomorphism, decomposes
//!   scalars to halve their bit-width before running Pippenger.
//! - **Fixed-base MSM**: Uses precomputed tables for repeated scalar multiplications
//!   with the same base point.
//!
//! # Usage
//!
//! Curves implement the traits in this crate, then use the generic MSM functions:
//!
//! ```ignore
//! use jolt_inlines_msm::{MsmGroup, GlvCapable, msm_pippenger, msm_glv};
//!
//! // For any curve implementing MsmGroup:
//! let result = msm_pippenger(&scalars, &points, window_bits);
//!
//! // For curves with GLV endomorphism (faster):
//! let result = msm_glv(&scalars, &points, window_bits);
//! ```
//!
//! # no_std Support
//!
//! This crate is `no_std` compatible and only requires `alloc`.

#![no_std]

extern crate alloc;

mod fixed_base;
mod glv;
mod pippenger;
mod scalar;
mod traits;

// Re-export all public items
pub use fixed_base::{msm_fixed_base, FixedBaseTable};
pub use glv::{msm_glv, msm_glv_const, msm_glv_with_scratch_const};
pub use pippenger::{msm_pippenger, msm_pippenger_const};
pub use traits::{GlvCapable, MsmGroup, WindowedScalar};
