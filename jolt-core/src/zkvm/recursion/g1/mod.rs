//! G1 curve operations sumchecks
//!
//! This module contains sumcheck protocols for G1 group operations:
//! - Addition: Proves G1 point addition constraints
//! - Scalar multiplication: Proves G1 scalar multiplication constraints

pub mod addition;
pub mod indexing;
pub mod scalar_multiplication;
pub mod types;
pub mod wiring;

pub use addition::{G1AddParams, G1AddProver, G1AddVerifier};
pub use scalar_multiplication::{
    G1ScalarMulProver, G1ScalarMulVerifier, ShiftG1ScalarMulProver, ShiftG1ScalarMulVerifier,
};
pub use types::{G1AddValues, G1ScalarMulPublicInputs};
pub use wiring::{WiringG1Prover, WiringG1Verifier};
