//! Grumpkin MSM utilities (GLV + Pippenger + Fixed-base) for use in the zkVM guest.
//!
//! This module is a direct integration of the MSM code that previously lived in
//! `examples/msm/guest`, made reusable for Hyrax verification and recursion.

pub mod fixed_base;
pub mod glv;
pub mod pippenger;
pub mod scalar;
pub mod traits;

pub use fixed_base::*;
pub use glv::*;
pub use pippenger::*;
pub use traits::*;

use crate::{GrumpkinFr, GrumpkinPoint};

// ============================================================
// Curve-Specific Constants for Grumpkin MSM
// ============================================================

/// Number of bits in a full Grumpkin scalar.
pub const SCALAR_BITS: usize = 256;

/// Number of bits in a GLV half-scalar (after decomposition).
pub const GLV_SCALAR_BITS: usize = 128;

/// Pippenger window size for baseline (256-bit scalars).
pub const BASELINE_WINDOW: usize = 12;

/// Number of buckets for baseline Pippenger.
pub const BASELINE_BUCKETS: usize = 1 << BASELINE_WINDOW;

/// Number of windows for baseline Pippenger.
pub const BASELINE_WINDOWS: usize = SCALAR_BITS.div_ceil(BASELINE_WINDOW);

/// Pippenger window size for GLV (128-bit scalars).
pub const GLV_WINDOW: usize = 8;

/// Number of buckets for GLV Pippenger.
pub const GLV_BUCKETS: usize = 1 << GLV_WINDOW;

/// Number of windows for GLV Pippenger.
pub const GLV_WINDOWS: usize = GLV_SCALAR_BITS.div_ceil(GLV_WINDOW);

/// Fixed-base window size (256-bit scalars).
pub const FIXED_BASE_WINDOW: usize = 14;

/// Number of buckets for fixed-base multiplication.
pub const FIXED_BASE_BUCKETS: usize = 1 << FIXED_BASE_WINDOW;

/// Number of windows for fixed-base multiplication.
pub const FIXED_BASE_WINDOWS: usize = SCALAR_BITS.div_ceil(FIXED_BASE_WINDOW);

/// Default compile-time window for GLV-accelerated MSM.
///
/// This matches the benchmarked setting in `examples/msm/guest`.
pub const DEFAULT_GLV_WINDOW_BITS: usize = GLV_WINDOW;

/// Type alias for a fixed-base table for Grumpkin generator multiplication.
pub type GrumpkinFixedBaseTable = FixedBaseTable<GrumpkinPoint, FIXED_BASE_WINDOWS, FIXED_BASE_BUCKETS>;

impl traits::MsmGroup for GrumpkinPoint {
    #[inline(always)]
    fn identity() -> Self {
        GrumpkinPoint::infinity()
    }

    #[inline(always)]
    fn is_identity(&self) -> bool {
        self.is_infinity()
    }

    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        GrumpkinPoint::add(self, other)
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        GrumpkinPoint::neg(self)
    }

    #[inline(always)]
    fn double(&self) -> Self {
        GrumpkinPoint::double(self)
    }

    #[inline(always)]
    fn double_and_add(&self, other: &Self) -> Self {
        GrumpkinPoint::double_and_add(self, other)
    }
}

impl traits::GlvCapable for GrumpkinPoint {
    type HalfScalar = u128;
    type FullScalar = GrumpkinFr;

    #[inline(always)]
    fn endomorphism(&self) -> Self {
        GrumpkinPoint::endomorphism(self)
    }

    #[inline(always)]
    fn decompose_scalar(k: &Self::FullScalar) -> [(bool, Self::HalfScalar); 2] {
        GrumpkinPoint::decompose_scalar(k)
    }
}
