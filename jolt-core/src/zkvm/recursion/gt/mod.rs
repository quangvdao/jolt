//! GT group operations sumchecks
//!
//! This module contains sumcheck protocols for GT (target group) operations:
//! - Exponentiation: Packed GT exponentiation with 2-phase protocol (base-4 digits)
//! - Multiplication: GT multiplication constraints
//! - Claim reduction: Reduces PackedGtExp claims via eq-weighted sumcheck
//! - Shift rho: Shift sumcheck for verifying rho_next claims in packed GT exp

pub mod claim_reduction;
pub mod exponentiation;
pub mod multiplication;
pub mod shift_rho;

pub use claim_reduction::{
    PackedGtExpClaimReductionParams, PackedGtExpClaimReductionProver,
    PackedGtExpClaimReductionVerifier,
};
pub use exponentiation::{PackedGtExpProver, PackedGtExpPublicInputs, PackedGtExpVerifier};
pub use multiplication::{GtMulProver, GtMulVerifier};
pub use shift_rho::{ShiftClaim, ShiftRhoParams, ShiftRhoProver, ShiftRhoVerifier};
