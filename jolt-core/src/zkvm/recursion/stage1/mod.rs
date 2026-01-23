//! Stage 1: Constraint Sumchecks
//!
//! This module contains the sumcheck protocols for proving
//! different types of constraints in the recursion SNARK:
//!
//! - `gt_exp`: GT exponentiation constraints (packed 11-var, base-4 digits)
//! - `gt_mul`: GT multiplication constraints
//! - `g1_add`: G1 point addition constraints
//! - `g1_scalar_mul`: G1 scalar multiplication constraints
//! - `g2_add`: G2 point addition constraints
//! - `g2_scalar_mul`: G2 scalar multiplication constraints
//! - `shift_rho`: Shift sumcheck for verifying rho_next claims (Stage 1b)
//! - `packed_gt_exp_reduction`: Reduces PackedGtExp claims to a single point (Stage 1b)
//! - `constraint_list_sumcheck`: Generic wrapper for batched constraint sumchecks

pub mod constraint_list_sumcheck;
pub mod g1_add;
pub mod g1_scalar_mul;
pub mod g2_add;
pub mod g2_scalar_mul;
pub mod gt_exp;
pub mod gt_mul;
pub mod packed_gt_exp_reduction;
pub mod shift_rho;
