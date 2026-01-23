//! Grumpkin MSM utilities (GLV + Pippenger) for use in the zkVM guest.
//!
//! This module is a direct integration of the MSM code that previously lived in
//! `examples/msm/guest`, made reusable for Hyrax verification.

pub mod glv;
pub mod pippenger;
pub mod scalar;
pub mod traits;

pub use glv::*;
pub use pippenger::*;
pub use traits::*;

use crate::{GrumpkinFr, GrumpkinPoint};

/// Default compile-time window for GLV-accelerated MSM.
///
/// This matches the benchmarked setting in `examples/msm/guest`.
pub const DEFAULT_GLV_WINDOW_BITS: usize = 8;

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
