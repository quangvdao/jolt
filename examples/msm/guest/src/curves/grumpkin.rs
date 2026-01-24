//! Grumpkin curve-specific benchmark helpers.
//!
//! The core MSM functionality (traits, algorithms, constants) is now in
//! `jolt-inlines-grumpkin::msm`. This module only contains benchmark-specific
//! helpers for generating test data.

use jolt_inlines_grumpkin::msm::GrumpkinFixedBaseTable;
use jolt_inlines_grumpkin::{GrumpkinFr, GrumpkinPoint, UnwrapOrSpoilProof};

// Re-export constants and types from jolt-inlines for convenience.
pub use jolt_inlines_grumpkin::msm::{
    BASELINE_BUCKETS, BASELINE_WINDOW, BASELINE_WINDOWS, FIXED_BASE_BUCKETS, FIXED_BASE_WINDOW,
    FIXED_BASE_WINDOWS, GLV_BUCKETS, GLV_SCALAR_BITS, GLV_WINDOW, GLV_WINDOWS, SCALAR_BITS,
};

/// Type alias for convenience - re-export from jolt-inlines.
pub type FixedBaseTable = GrumpkinFixedBaseTable;

// ============================================================
// Benchmark Helpers
// ============================================================

/// Grumpkin Fr modulus limbs for scalar reduction.
const FR_MODULUS_LIMBS: [u64; 4] = [
    4332616871279656263,
    10917124144477883021,
    13281191951274694749,
    3486998266802970665,
];

#[inline(always)]
fn is_ge_modulus(x: &[u64; 4]) -> bool {
    let m = FR_MODULUS_LIMBS;
    if x[3] > m[3] {
        return true;
    }
    if x[3] < m[3] {
        return false;
    }
    if x[2] > m[2] {
        return true;
    }
    if x[2] < m[2] {
        return false;
    }
    if x[1] > m[1] {
        return true;
    }
    if x[1] < m[1] {
        return false;
    }
    x[0] >= m[0]
}

#[inline(always)]
fn sub_modulus(x: [u64; 4]) -> [u64; 4] {
    let m = FR_MODULUS_LIMBS;
    let mut out = [0u64; 4];
    let mut borrow = 0u128;
    let mut i = 0;
    while i < 4 {
        let xi = x[i] as u128;
        let mi = m[i] as u128 + borrow;
        if xi >= mi {
            out[i] = (xi - mi) as u64;
            borrow = 0;
        } else {
            out[i] = ((1u128 << 64) + xi - mi) as u64;
            borrow = 1;
        }
        i += 1;
    }
    out
}

/// Reduce scalar modulo Fr.
#[inline(always)]
pub fn reduce_scalar(mut scalar: [u64; 4]) -> [u64; 4] {
    while is_ge_modulus(&scalar) {
        scalar = sub_modulus(scalar);
    }
    scalar
}

/// Generate deterministic test scalars using a simple LCG.
#[inline(always)]
pub fn generate_scalars<const N: usize>(seed: u64) -> [[u64; 4]; N] {
    let mut scalars = [[0u64; 4]; N];
    let (a, c) = (6364136223846793005u64, 1442695040888963407u64);
    let mut state = seed;

    for scalar in scalars.iter_mut() {
        for limb in scalar.iter_mut() {
            state = state.wrapping_mul(a).wrapping_add(c);
            *limb = state;
        }
        *scalar = reduce_scalar(*scalar);
    }
    scalars
}

/// Generate deterministic test points by fixed-base scalar multiplication of generator.
#[inline(always)]
pub fn generate_points_fixed_base<const N: usize>(
    seed: u64,
    table_g: &FixedBaseTable,
) -> [GrumpkinPoint; N] {
    let mut points: [GrumpkinPoint; N] = core::array::from_fn(|_| GrumpkinPoint::infinity());
    let (a, c) = (6364136223846793005u64, 1442695040888963407u64);
    let mut state = seed;

    for point in points.iter_mut() {
        state = state.wrapping_mul(a).wrapping_add(c);
        let small_scalar = [state, 0, 0, 0];
        *point = table_g.scalar_mul(&small_scalar);
    }
    points
}

/// Convert [u64; 4] to GrumpkinFr.
#[inline(always)]
pub fn scalar_to_fr(scalar: &[u64; 4]) -> GrumpkinFr {
    GrumpkinFr::from_u64_arr(scalar).unwrap_or_spoil_proof()
}
