//! Helpers for G1-local constraint indexing (split-k aware).
//!
//! This mirrors the GT indexing scheme in `gt/indexing.rs`, but for the *G1 group*.
//!
//! Goals:
//! - Keep **committed/prefix-packed** fused polynomials **family-local**:
//!   - G1ScalarMul fused rows use `k_smul`.
//!   - G1Add fused rows use `k_add`.
//! - Allow a **single** fused wiring sumcheck to use a common suffix length
//!   `k_g1 = max(k_smul, k_add)` and consume family-local openings via:
//!   - dummy-low-bits embedding, and
//!   - β(dummy) normalization (marginalizing dummy bits).
//!
//! Definitions (constraint_types order):
//! - `c_smul` is the rank of a `G1ScalarMul` constraint among all G1ScalarMul constraints.
//! - `c_add`  is the rank of a `G1Add` constraint among all G1Add constraints.
//! - `k_smul = log2(next_pow2(num_g1_smul).max(1))`
//! - `k_add  = log2(next_pow2(num_g1_add ).max(1))`
//! - `k_g1   = max(k_smul, k_add)` (common wiring suffix length)
//!
//! Dummy-bit convention (same as fused GT wiring):
//! - The *dummy* bits are the **low** bits of the common `c` domain.
//! - Family-local index bits are a **suffix** of the common `c` vector.
//! - `embed(idx) = idx << dummy` where `dummy = k_common - k_family`.

use crate::zkvm::recursion::constraints::system::ConstraintType;

/// Return the global constraint indices that are G1ScalarMul constraints, in global order.
pub fn g1_smul_constraint_indices(constraint_types: &[ConstraintType]) -> Vec<usize> {
    constraint_types
        .iter()
        .enumerate()
        .filter_map(|(i, ct)| matches!(ct, ConstraintType::G1ScalarMul { .. }).then_some(i))
        .collect()
}

/// Return the global constraint indices that are G1Add constraints, in global order.
pub fn g1_add_constraint_indices(constraint_types: &[ConstraintType]) -> Vec<usize> {
    constraint_types
        .iter()
        .enumerate()
        .filter_map(|(i, ct)| matches!(ct, ConstraintType::G1Add).then_some(i))
        .collect()
}

/// Number of G1ScalarMul constraints.
pub fn num_g1_smul_constraints(constraint_types: &[ConstraintType]) -> usize {
    constraint_types
        .iter()
        .filter(|ct| matches!(ct, ConstraintType::G1ScalarMul { .. }))
        .count()
}

/// Number of G1Add constraints.
pub fn num_g1_add_constraints(constraint_types: &[ConstraintType]) -> usize {
    constraint_types
        .iter()
        .filter(|ct| matches!(ct, ConstraintType::G1Add))
        .count()
}

/// Padded G1ScalarMul constraint count (power of two, min 1).
pub fn num_g1_smul_constraints_padded(constraint_types: &[ConstraintType]) -> usize {
    num_g1_smul_constraints(constraint_types)
        .max(1)
        .next_power_of_two()
}

/// Padded G1Add constraint count (power of two, min 1).
pub fn num_g1_add_constraints_padded(constraint_types: &[ConstraintType]) -> usize {
    num_g1_add_constraints(constraint_types).max(1).next_power_of_two()
}

/// Shared padded constraint count used by fused G1 wiring (max of the family paddings).
pub fn num_g1_constraints_padded(constraint_types: &[ConstraintType]) -> usize {
    core::cmp::max(
        num_g1_smul_constraints_padded(constraint_types),
        num_g1_add_constraints_padded(constraint_types),
    )
}

/// `k_smul = log2(next_pow2(num_g1_smul).max(1))`.
pub fn k_smul(constraint_types: &[ConstraintType]) -> usize {
    num_g1_smul_constraints_padded(constraint_types).trailing_zeros() as usize
}

/// `k_add = log2(next_pow2(num_g1_add).max(1))`.
pub fn k_add(constraint_types: &[ConstraintType]) -> usize {
    num_g1_add_constraints_padded(constraint_types).trailing_zeros() as usize
}

/// `k_g1 = log2(num_g1_constraints_padded) = max(k_smul, k_add)`.
pub fn k_g1(constraint_types: &[ConstraintType]) -> usize {
    num_g1_constraints_padded(constraint_types).trailing_zeros() as usize
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
        // k_common=5, k_family=3 -> dummy=2; embed shifts left by 2.
        assert_eq!(embed_index(3, 5, 3), 12);
        assert_eq!(embed_index(0, 5, 3), 0);
        assert_eq!(dummy_bits(5, 3), 2);
    }
}

