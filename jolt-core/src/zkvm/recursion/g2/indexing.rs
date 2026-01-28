//! Helpers for G2-local constraint indexing (split-k aware).
//!
//! This mirrors the GT indexing scheme in `gt/indexing.rs` and the G1 scheme in `g1/indexing.rs`,
//! but for the *G2 group*.
//!
//! Goals:
//! - Keep **committed/prefix-packed** fused polynomials **family-local**:
//!   - G2ScalarMul fused rows use `k_smul`.
//!   - G2Add fused rows use `k_add`.
//! - Allow a **single** fused wiring sumcheck to use a common suffix length
//!   `k_g2 = max(k_smul, k_add)` and consume family-local openings via:
//!   - dummy-low-bits embedding, and
//!   - β(dummy) normalization (marginalizing dummy bits).
//!
//! Definitions (constraint_types order):
//! - `c_smul` is the rank of a `G2ScalarMul` constraint among all G2ScalarMul constraints.
//! - `c_add`  is the rank of a `G2Add` constraint among all G2Add constraints.
//! - `k_smul = log2(next_pow2(num_g2_smul).max(1))`
//! - `k_add  = log2(next_pow2(num_g2_add ).max(1))`
//! - `k_g2   = max(k_smul, k_add)` (common wiring suffix length)
//!
//! Dummy-bit convention (uniform across fused families):
//! - The *dummy* bits are the **low** bits of the common `c` domain.
//! - Family-local index bits are a **suffix** of the common `c` vector.
//! - `embed(idx) = idx << dummy` where `dummy = k_common - k_family`.

use crate::zkvm::recursion::constraints::system::ConstraintType;

/// Return the global constraint indices that are G2ScalarMul constraints, in global order.
pub fn g2_smul_constraint_indices(constraint_types: &[ConstraintType]) -> Vec<usize> {
    constraint_types
        .iter()
        .enumerate()
        .filter_map(|(i, ct)| matches!(ct, ConstraintType::G2ScalarMul { .. }).then_some(i))
        .collect()
}

/// Return the global constraint indices that are G2Add constraints, in global order.
pub fn g2_add_constraint_indices(constraint_types: &[ConstraintType]) -> Vec<usize> {
    constraint_types
        .iter()
        .enumerate()
        .filter_map(|(i, ct)| matches!(ct, ConstraintType::G2Add).then_some(i))
        .collect()
}

/// Number of G2ScalarMul constraints.
pub fn num_g2_smul_constraints(constraint_types: &[ConstraintType]) -> usize {
    constraint_types
        .iter()
        .filter(|ct| matches!(ct, ConstraintType::G2ScalarMul { .. }))
        .count()
}

/// Number of G2Add constraints.
pub fn num_g2_add_constraints(constraint_types: &[ConstraintType]) -> usize {
    constraint_types
        .iter()
        .filter(|ct| matches!(ct, ConstraintType::G2Add))
        .count()
}

/// Padded G2ScalarMul constraint count (power of two, min 1).
pub fn num_g2_smul_constraints_padded(constraint_types: &[ConstraintType]) -> usize {
    num_g2_smul_constraints(constraint_types)
        .max(1)
        .next_power_of_two()
}

/// Padded G2Add constraint count (power of two, min 1).
pub fn num_g2_add_constraints_padded(constraint_types: &[ConstraintType]) -> usize {
    num_g2_add_constraints(constraint_types)
        .max(1)
        .next_power_of_two()
}

/// Shared padded constraint count used by fused G2 wiring (max of the family paddings).
pub fn num_g2_constraints_padded(constraint_types: &[ConstraintType]) -> usize {
    core::cmp::max(
        num_g2_smul_constraints_padded(constraint_types),
        num_g2_add_constraints_padded(constraint_types),
    )
}

/// `k_smul = log2(next_pow2(num_g2_smul).max(1))`.
pub fn k_smul(constraint_types: &[ConstraintType]) -> usize {
    num_g2_smul_constraints_padded(constraint_types).trailing_zeros() as usize
}

/// `k_add = log2(next_pow2(num_g2_add).max(1))`.
pub fn k_add(constraint_types: &[ConstraintType]) -> usize {
    num_g2_add_constraints_padded(constraint_types).trailing_zeros() as usize
}

/// `k_g2 = log2(num_g2_constraints_padded) = max(k_smul, k_add)`.
pub fn k_g2(constraint_types: &[ConstraintType]) -> usize {
    num_g2_constraints_padded(constraint_types).trailing_zeros() as usize
}

/// Number of dummy low bits when embedding a family-local domain into a common domain.
#[inline]
pub fn dummy_bits(k_common: usize, k_family: usize) -> usize {
    k_common.saturating_sub(k_family)
}

/// Embed a family-local index into a common-domain index by shifting left by `dummy` low bits.
#[inline]
pub fn embed_index(idx_family: usize, k_common: usize, k_family: usize) -> usize {
    idx_family << dummy_bits(k_common, k_family)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_index_uses_dummy_low_bits() {
        // k_common=6, k_family=4 -> dummy=2; embed shifts left by 2.
        assert_eq!(embed_index(3, 6, 4), 12);
        assert_eq!(embed_index(0, 6, 4), 0);
        assert_eq!(dummy_bits(6, 4), 2);
    }
}
