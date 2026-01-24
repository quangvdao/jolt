//! Grumpkin MSM utilities for use in the zkVM guest.
//!
//! This module provides Grumpkin-specific MSM functionality by:
//! 1. Re-exporting generic MSM algorithms from `jolt-inlines-msm`
//! 2. Implementing the MSM traits for `GrumpkinPoint`
//! 3. Providing Grumpkin-specific constants and type aliases

// Re-export all generic MSM functionality from jolt-inlines-msm
pub use jolt_inlines_msm::{
    msm_fixed_base, msm_glv, msm_glv_const, msm_glv_with_scratch_const, msm_pippenger,
    msm_pippenger_const, FixedBaseTable, GlvCapable, MsmGroup, WindowedScalar,
};

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
pub type GrumpkinFixedBaseTable =
    FixedBaseTable<GrumpkinPoint, FIXED_BASE_WINDOWS, FIXED_BASE_BUCKETS>;

// ============================================================
// Trait Implementations for GrumpkinPoint
// ============================================================

impl MsmGroup for GrumpkinPoint {
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

impl GlvCapable for GrumpkinPoint {
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
