//! Helpers for GT-local constraint indexing.
//!
//! We define a GT-local index space over only the `{GtExp,GtMul}` constraints appearing
//! in the global `constraint_types` list.
//!
//! - `c_gt` is the rank of a GT constraint among GT constraints (in global order).
//! - `k_gt = log2(num_gt_constraints_padded)` where `num_gt_constraints_padded` is the next power
//!   of two (min 1).

use crate::zkvm::recursion::constraints::system::ConstraintType;

/// Return the global constraint indices that are GT constraints, in global order.
pub fn gt_constraint_indices(constraint_types: &[ConstraintType]) -> Vec<usize> {
    constraint_types
        .iter()
        .enumerate()
        .filter_map(|(i, ct)| matches!(ct, ConstraintType::GtExp | ConstraintType::GtMul).then_some(i))
        .collect()
}

/// Number of GT constraints (GtExp + GtMul).
pub fn num_gt_constraints(constraint_types: &[ConstraintType]) -> usize {
    constraint_types
        .iter()
        .filter(|ct| matches!(ct, ConstraintType::GtExp | ConstraintType::GtMul))
        .count()
}

/// Padded GT constraint count (power of two, min 1).
pub fn num_gt_constraints_padded(constraint_types: &[ConstraintType]) -> usize {
    num_gt_constraints(constraint_types).max(1).next_power_of_two()
}

/// `k_gt = log2(num_gt_constraints_padded)`.
pub fn k_gt(constraint_types: &[ConstraintType]) -> usize {
    num_gt_constraints_padded(constraint_types).trailing_zeros() as usize
}

/// Map each global constraint index to its GT-local rank (c_gt), if it is a GT constraint.
pub fn global_to_c_gt(constraint_types: &[ConstraintType]) -> Vec<Option<usize>> {
    let mut out = vec![None; constraint_types.len()];
    let mut rank = 0usize;
    for (i, ct) in constraint_types.iter().enumerate() {
        if matches!(ct, ConstraintType::GtExp | ConstraintType::GtMul) {
            out[i] = Some(rank);
            rank += 1;
        }
    }
    out
}

