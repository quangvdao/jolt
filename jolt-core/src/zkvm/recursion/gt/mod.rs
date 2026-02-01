//! GT group operations sumchecks
//!
//! This module contains sumcheck protocols for GT (target group) operations:
//! - Exponentiation: Packed GT exponentiation with 2-phase protocol (base-4 digits)
//! - Multiplication: GT multiplication constraints

pub mod base_power;
pub mod exponentiation;
pub mod indexing;
pub mod multiplication;
pub mod shift;
pub mod stage1_base_openings;
pub mod stage2_base_openings;
pub mod stage2_openings;
pub mod types;
pub mod wiring;

pub use base_power::{GtExpBasePowProver, GtExpBasePowVerifier};
pub use exponentiation::{GtExpParams, GtExpProver, GtExpVerifier};
pub use indexing::{gt_constraint_indices, k_gt, num_gt_constraints, num_gt_constraints_padded};
pub use multiplication::{GtMulParams, GtMulProver, GtMulVerifier};
pub use shift::{GtShiftParams, GtShiftProver, GtShiftVerifier};
pub use stage1_base_openings::{GtExpBaseStage1OpeningsProver, GtExpBaseStage1OpeningsVerifier};
pub use stage2_base_openings::{GtExpBaseStage2OpeningsProver, GtExpBaseStage2OpeningsVerifier};
pub use stage2_openings::{GtExpStage2OpeningsProver, GtExpStage2OpeningsVerifier};
pub use types::{
    eq_lsb_evals, eq_lsb_mle, eq_plus_one_lsb_evals, eq_plus_one_lsb_mle, GtExpPublicInputs,
    GtExpWitness, GtMulConstraintPolynomials,
};
pub use wiring::{WiringGtProver, WiringGtVerifier};
