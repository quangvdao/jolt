//! Helpers for GT-local constraint indexing.
//!
//! We define **family-local** index spaces for GTExp and GTMul constraints appearing in the
//! global `constraint_types` list.
//!
//! Rationale:
//! - Using a single union GT index space over `{GtExp,GtMul}` forces padding to
//!   `next_pow2(num_gt_exp + num_gt_mul)` and introduces "type-mismatch zeros" in fused rows.
//! - We instead pad each family separately and then take the max padded size so all GT-fused
//!   instances can share the same Stage-2 index suffix length `k`.
//!
//! Definitions:
//! - `c_exp` is the rank of a `GtExp` constraint among GTExp constraints (in global order).
//! - `c_mul` is the rank of a `GtMul` constraint among GTMul constraints (in global order).
//! - `k_exp = log2(next_pow2(num_gt_exp).max(1))`
//! - `k_mul = log2(next_pow2(num_gt_mul).max(1))`
//! - `k_gt = max(k_exp, k_mul)` (shared Stage-2 suffix length for GT-fused mode)
//! - `num_gt_constraints_padded = 2^k_gt`

use crate::zkvm::recursion::constraints::system::ConstraintType;
use core::ops::Range;

/// Return the global constraint indices that are GTExp constraints, in global order.
pub fn gt_exp_constraint_indices(constraint_types: &[ConstraintType]) -> Vec<usize> {
    constraint_types
        .iter()
        .enumerate()
        .filter_map(|(i, ct)| matches!(ct, ConstraintType::GtExp).then_some(i))
        .collect()
}

/// Return the global constraint indices that are GTMul constraints, in global order.
pub fn gt_mul_constraint_indices(constraint_types: &[ConstraintType]) -> Vec<usize> {
    constraint_types
        .iter()
        .enumerate()
        .filter_map(|(i, ct)| matches!(ct, ConstraintType::GtMul).then_some(i))
        .collect()
}

/// Number of GTExp constraints.
pub fn num_gt_exp_constraints(constraint_types: &[ConstraintType]) -> usize {
    constraint_types
        .iter()
        .filter(|ct| matches!(ct, ConstraintType::GtExp))
        .count()
}

/// Number of GTMul constraints.
pub fn num_gt_mul_constraints(constraint_types: &[ConstraintType]) -> usize {
    constraint_types
        .iter()
        .filter(|ct| matches!(ct, ConstraintType::GtMul))
        .count()
}

/// Number of GT constraints total (GtExp + GtMul).
pub fn num_gt_constraints(constraint_types: &[ConstraintType]) -> usize {
    num_gt_exp_constraints(constraint_types) + num_gt_mul_constraints(constraint_types)
}

/// Return the global constraint indices that are GT constraints (GtExp or GtMul), in global order.
///
/// This is a convenience helper for scans; it does **not** define the fused `c` indexing scheme.
pub fn gt_constraint_indices(constraint_types: &[ConstraintType]) -> Vec<usize> {
    constraint_types
        .iter()
        .enumerate()
        .filter_map(|(i, ct)| {
            matches!(ct, ConstraintType::GtExp | ConstraintType::GtMul).then_some(i)
        })
        .collect()
}

/// Padded GTExp constraint count (power of two, min 1).
pub fn num_gt_exp_constraints_padded(constraint_types: &[ConstraintType]) -> usize {
    num_gt_exp_constraints(constraint_types)
        .max(1)
        .next_power_of_two()
}

/// Padded GTMul constraint count (power of two, min 1).
pub fn num_gt_mul_constraints_padded(constraint_types: &[ConstraintType]) -> usize {
    num_gt_mul_constraints(constraint_types)
        .max(1)
        .next_power_of_two()
}

/// Shared padded GT constraint count used by GT-fused mode.
///
/// This is `max(next_pow2(num_gt_exp), next_pow2(num_gt_mul))` (min 1), so both families can
/// share the same `k` without paying the union padding factor.
pub fn num_gt_constraints_padded(constraint_types: &[ConstraintType]) -> usize {
    core::cmp::max(
        num_gt_exp_constraints_padded(constraint_types),
        num_gt_mul_constraints_padded(constraint_types),
    )
}

/// `k_exp = log2(next_pow2(num_gt_exp).max(1))`.
pub fn k_exp(constraint_types: &[ConstraintType]) -> usize {
    num_gt_exp_constraints_padded(constraint_types).trailing_zeros() as usize
}

/// `k_mul = log2(next_pow2(num_gt_mul).max(1))`.
pub fn k_mul(constraint_types: &[ConstraintType]) -> usize {
    num_gt_mul_constraints_padded(constraint_types).trailing_zeros() as usize
}

/// `k_gt = log2(num_gt_constraints_padded)`.
pub fn k_gt(constraint_types: &[ConstraintType]) -> usize {
    num_gt_constraints_padded(constraint_types).trailing_zeros() as usize
}

/// Range of `c_common` (all k bits) within a GTExp-style `(s,u,c_common)` slice.
#[inline]
pub fn gt_exp_c_common_range(k_common: usize) -> Range<usize> {
    let x_vars = 11usize;
    x_vars..(x_vars + k_common)
}

/// Range of `c_tail` (last `k_family` bits) within a GTExp-style `(s,u,c_common)` slice.
#[inline]
pub fn gt_exp_c_tail_range(k_common: usize, k_family: usize) -> Range<usize> {
    let x_vars = 11usize;
    let d = k_common.saturating_sub(k_family);
    (x_vars + d)..(x_vars + k_common)
}

/// Range of `c_common` (all k bits) within a GTMul-style `(u,c_common)` slice.
#[inline]
pub fn gt_mul_c_common_range(k_common: usize) -> Range<usize> {
    let u_vars = 4usize;
    u_vars..(u_vars + k_common)
}

/// Range of `c_tail` (last `k_family` bits) within a GTMul-style `(u,c_common)` slice.
#[inline]
pub fn gt_mul_c_tail_range(k_common: usize, k_family: usize) -> Range<usize> {
    let u_vars = 4usize;
    let d = k_common.saturating_sub(k_family);
    (u_vars + d)..(u_vars + k_common)
}

/// Map each global constraint index to its GTExp-local rank (c_exp), if it is a GTExp constraint.
pub fn global_to_c_exp(constraint_types: &[ConstraintType]) -> Vec<Option<usize>> {
    let mut out = vec![None; constraint_types.len()];
    let mut rank = 0usize;
    for (i, ct) in constraint_types.iter().enumerate() {
        if matches!(ct, ConstraintType::GtExp) {
            out[i] = Some(rank);
            rank += 1;
        }
    }
    out
}

/// Map each global constraint index to its GTMul-local rank (c_mul), if it is a GTMul constraint.
pub fn global_to_c_mul(constraint_types: &[ConstraintType]) -> Vec<Option<usize>> {
    let mut out = vec![None; constraint_types.len()];
    let mut rank = 0usize;
    for (i, ct) in constraint_types.iter().enumerate() {
        if matches!(ct, ConstraintType::GtMul) {
            out[i] = Some(rank);
            rank += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::recursion::constraints::system::ConstraintType;

    #[test]
    fn split_k_values_are_family_local_and_k_gt_is_max() {
        let mut constraint_types = Vec::new();
        // 5 exp, 300 mul -> k_exp=3 (8), k_mul=9 (512), k_gt=9
        constraint_types.extend(std::iter::repeat_n(ConstraintType::GtExp, 5));
        constraint_types.extend(std::iter::repeat_n(ConstraintType::GtMul, 300));

        assert_eq!(num_gt_exp_constraints(&constraint_types), 5);
        assert_eq!(num_gt_mul_constraints(&constraint_types), 300);
        assert_eq!(num_gt_exp_constraints_padded(&constraint_types), 8);
        assert_eq!(num_gt_mul_constraints_padded(&constraint_types), 512);
        assert_eq!(k_exp(&constraint_types), 3);
        assert_eq!(k_mul(&constraint_types), 9);
        assert_eq!(k_gt(&constraint_types), 9);
    }

    #[test]
    fn tail_ranges_target_last_bits() {
        // For GTExp slice: len = 11 + k_common.
        let k_common = 9usize;
        let k_family = 3usize;
        let r = gt_exp_c_tail_range(k_common, k_family);
        assert_eq!(r.start, 11 + (k_common - k_family));
        assert_eq!(r.end, 11 + k_common);

        // For GTMul slice: len = 4 + k_common.
        let r = gt_mul_c_tail_range(k_common, k_family);
        assert_eq!(r.start, 4 + (k_common - k_family));
        assert_eq!(r.end, 4 + k_common);
    }
}
