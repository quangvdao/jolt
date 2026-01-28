//! GT group operations sumchecks
//!
//! This module contains sumcheck protocols for GT (target group) operations:
//! - Exponentiation: Packed GT exponentiation with 2-phase protocol (base-4 digits)
//! - Multiplication: GT multiplication constraints

pub mod fused_exponentiation;
pub mod fused_multiplication;
pub mod fused_shift;
pub mod fused_stage2_openings;
pub mod fused_wiring;
pub mod indexing;
pub mod types;

pub use fused_exponentiation::{FusedGtExpParams, FusedGtExpProver, FusedGtExpVerifier};
pub use fused_multiplication::{FusedGtMulParams, FusedGtMulProver, FusedGtMulVerifier};
pub use fused_shift::{FusedGtShiftParams, FusedGtShiftProver, FusedGtShiftVerifier};
pub use fused_stage2_openings::{FusedGtExpStage2OpeningsProver, FusedGtExpStage2OpeningsVerifier};
pub use fused_wiring::{FusedWiringGtProver, FusedWiringGtVerifier};
pub use indexing::{gt_constraint_indices, k_gt, num_gt_constraints, num_gt_constraints_padded};
pub use types::{
    eq_lsb_evals, eq_lsb_mle, eq_plus_one_lsb_evals, eq_plus_one_lsb_mle, GtExpPublicInputs,
    GtExpWitness, GtMulConstraintPolynomials,
};
