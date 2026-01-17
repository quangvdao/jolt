//! Provable Grumpkin MSM on stable limb layouts.

use crate::types::{AffinePoint, FrLimbs, JacobianPoint};
use crate::MSM_N;

use super::curve::{add_jac, add_mixed, double_jac, AffineMont, JacobianMont};

use alloc::vec::Vec;

// Grumpkin scalar field is `ark_bn254::Fq`, which is 254-bit.
const SCALAR_BITS: usize = 254;
const WINDOW_SIZE: usize = 8;
const NUM_BUCKETS: usize = 1 << WINDOW_SIZE; // 256
const NUM_WINDOWS: usize = (SCALAR_BITS + WINDOW_SIZE - 1) / WINDOW_SIZE; // 32

/// Computes a fixed-size Grumpkin MSM for `MSM_N == 2048`.
///
/// - `bases`: `MSM_N` affine points (stable limb encoding)
/// - `scalars`: `MSM_N` scalars as 4×64-bit limbs (LSB-first)
///
/// Output is a Jacobian point (stable limb encoding).
pub fn grumpkin_msm_2048(bases: &[AffinePoint], scalars: &[FrLimbs]) -> JacobianPoint {
    assert!(
        bases.len() >= MSM_N && scalars.len() >= MSM_N,
        "grumpkin_msm_2048 expects at least {MSM_N} bases and scalars"
    );

    // Pre-convert bases into Montgomery form once (critical for performance).
    let mut bases_mont: Vec<AffineMont> = Vec::with_capacity(MSM_N);
    for i in 0..MSM_N {
        bases_mont.push(AffineMont::from_affine_point(&bases[i]));
    }

    let mut result = JacobianMont::infinity();

    for window in (0..NUM_WINDOWS).rev() {
        // Shift accumulated result by WINDOW_SIZE bits: multiply by 2^w.
        for _ in 0..WINDOW_SIZE {
            result = double_jac(result);
        }

        // Buckets for this window.
        let mut buckets: [JacobianMont; NUM_BUCKETS] =
            core::array::from_fn(|_| JacobianMont::infinity());

        let start_bit = window * WINDOW_SIZE;
        for i in 0..MSM_N {
            let digit = scalars[i].window(start_bit, WINDOW_SIZE) as usize;
            if digit != 0 {
                buckets[digit] = add_mixed(buckets[digit], &bases_mont[i]);
            }
        }

        // Accumulate buckets via running sum trick:
        //   sum = Σ_{j=1..(2^w-1)} j * bucket[j]
        let mut running = JacobianMont::infinity();
        let mut window_sum = JacobianMont::infinity();
        for b in (1..NUM_BUCKETS).rev() {
            running = add_jac(running, buckets[b]);
            window_sum = add_jac(window_sum, running);
        }

        result = add_jac(result, window_sum);
    }

    result.to_canonical()
}
