//! G1 curve operations sumchecks
//!
//! This module contains sumcheck protocols for G1 group operations:
//! - Addition: Proves G1 point addition constraints
//! - Scalar multiplication: Proves G1 scalar multiplication constraints

pub mod fused_addition;
pub mod fused_scalar_multiplication;
pub mod fused_wiring;
pub mod indexing;
pub mod types;

pub use fused_addition::{FusedG1AddParams, FusedG1AddProver, FusedG1AddVerifier};
pub use fused_scalar_multiplication::{
    FusedG1ScalarMulProver, FusedG1ScalarMulVerifier, FusedShiftG1ScalarMulProver,
    FusedShiftG1ScalarMulVerifier,
};
pub use fused_wiring::{FusedWiringG1Prover, FusedWiringG1Verifier};
pub use types::{G1AddValues, G1ScalarMulPublicInputs};
