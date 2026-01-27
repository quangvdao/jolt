//! GT group operations sumchecks
//!
//! This module contains sumcheck protocols for GT (target group) operations:
//! - Exponentiation: Packed GT exponentiation with 2-phase protocol (base-4 digits)
//! - Multiplication: GT multiplication constraints
//! - Claim reduction: Reduces GtExp claims via eq-weighted sumcheck
//! - Shift rho: Shift sumcheck for verifying rho_next claims in packed GT exp

pub mod claim_reduction;
pub mod exponentiation;
pub mod fused_exponentiation;
pub mod fused_multiplication;
pub mod fused_shift;
pub mod fused_stage2_openings;
pub mod indexing;
pub mod multiplication;
pub mod shift;
pub mod wiring;
pub mod wiring_binding;

pub use claim_reduction::{
    GtExpClaimReductionParams, GtExpClaimReductionProver, GtExpClaimReductionVerifier,
};
pub use exponentiation::{GtExpProver, GtExpPublicInputs, GtExpVerifier};
pub use fused_exponentiation::{FusedGtExpParams, FusedGtExpProver, FusedGtExpVerifier};
pub use fused_multiplication::{FusedGtMulParams, FusedGtMulProver, FusedGtMulVerifier};
pub use fused_shift::{FusedGtShiftParams, FusedGtShiftProver, FusedGtShiftVerifier};
pub use fused_stage2_openings::{FusedGtExpStage2OpeningsProver, FusedGtExpStage2OpeningsVerifier};
pub use indexing::{gt_constraint_indices, k_gt, num_gt_constraints_padded};
pub use multiplication::{GtMulProver, GtMulVerifier};
pub use shift::{GtShiftClaim, GtShiftParams, GtShiftProver, GtShiftVerifier};
pub use wiring::{WiringGtProver, WiringGtVerifier};
pub use wiring_binding::{GtWiringBinding, GtWiringBindingProver, GtWiringBindingVerifier};
