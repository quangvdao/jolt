//! Recursion verifier metadata derived by the prover.
//!
//! This is **not** part of the recursion proof payload. It is an internal artifact produced by
//! the recursion prover (`poly_commit`) and can be deterministically re-derived by the verifier
//! from the AST instance plan when needed.

use super::{
    constraints::system::ConstraintType, g1::scalar_multiplication::G1ScalarMulPublicInputs,
    g2::scalar_multiplication::G2ScalarMulPublicInputs, gt::exponentiation::GtExpPublicInputs,
};

/// Constraint metadata for the recursion verifier (internal; not serialized in the proof).
#[derive(Clone, Debug)]
pub struct RecursionConstraintMetadata {
    pub constraint_types: Vec<ConstraintType>,
    /// Number of variables of the packed dense polynomial committed via Hyrax.
    pub dense_num_vars: usize,
    /// Public inputs for packed GT exp (base Fq12 and scalar bits for each GT exp)
    pub gt_exp_public_inputs: Vec<GtExpPublicInputs>,
    /// Public inputs for G1 scalar multiplication (scalar per G1ScalarMul constraint)
    pub g1_scalar_mul_public_inputs: Vec<G1ScalarMulPublicInputs>,
    /// Public inputs for G2 scalar multiplication (scalar per G2ScalarMul constraint)
    pub g2_scalar_mul_public_inputs: Vec<G2ScalarMulPublicInputs>,
}
