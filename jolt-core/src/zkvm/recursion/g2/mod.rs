//! G2 curve operations sumchecks
//!
//! This module contains sumcheck protocols for G2 group operations:
//! - Addition: Proves G2 point addition constraints
//! - Scalar multiplication: Proves G2 scalar multiplication constraints

pub mod addition;
pub mod scalar_multiplication;

pub use addition::{G2AddProver, G2AddVerifier};
pub use scalar_multiplication::{G2ScalarMulProver, G2ScalarMulPublicInputs, G2ScalarMulVerifier};
