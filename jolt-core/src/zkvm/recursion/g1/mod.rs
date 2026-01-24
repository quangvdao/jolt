//! G1 curve operations sumchecks
//!
//! This module contains sumcheck protocols for G1 group operations:
//! - Addition: Proves G1 point addition constraints
//! - Scalar multiplication: Proves G1 scalar multiplication constraints
//! - Shift scalar multiplication: Shift sumchecks for scalar multiplication traces

pub mod addition;
pub mod scalar_multiplication;
pub mod shift_scalar_multiplication;

pub use addition::{G1AddProver, G1AddVerifier};
pub use scalar_multiplication::{G1ScalarMulProver, G1ScalarMulPublicInputs, G1ScalarMulVerifier};
pub use shift_scalar_multiplication::{
    g1_shift_params, g2_shift_params, ShiftG1ScalarMulProver, ShiftG1ScalarMulVerifier,
    ShiftG2ScalarMulProver, ShiftG2ScalarMulVerifier, ShiftScalarMulParams,
};
