//! Witness types for the recursion SNARK
//!
//! This module defines the witness data structures used in the recursion protocol.

use crate::poly::dense_mlpoly::DensePolynomial;
use ark_bn254::{Fq, Fq12, Fr, G1Affine};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// Aggregated witness data for GT exponentiation constraints (used by DoryRecursionWitness).
///
/// This structure aggregates data from multiple GT exponentiation operations
/// for the recursion prover's constraint system.
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct GTExpWitness {
    /// The g polynomial (irreducible polynomial for Fq12)
    pub g_poly: DensePolynomial<Fq>,
    /// The g values
    pub g_values: Vec<Fq>,
    /// The scalar exponent
    pub scalar: Fr,
    /// Binary representation of the scalar
    pub bits: Vec<bool>,
    /// Base values for each constraint
    pub base_values: Vec<Fq>,
    /// Rho values (accumulated results)
    pub rho_values: Vec<Fq>,
    /// Quotient values
    pub quotient_values: Vec<Fq>,
}

/// Aggregated witness data for GT multiplication constraints (used by DoryRecursionWitness).
///
/// This structure aggregates data from multiple GT multiplication operations
/// for the recursion prover's constraint system.
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct GTMulWitness {
    /// Left operand values
    pub lhs_values: Vec<Fq>,
    /// Right operand values
    pub rhs_values: Vec<Fq>,
    /// Result values
    pub result_values: Vec<Fq>,
    /// Quotient values
    pub quotient_values: Vec<Fq>,
}

/// Per-operation witness for a single GT exponentiation (used by GTCombineWitness).
///
/// Captures the intermediate values for one iterative GT exponentiation.
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct GTExpOpWitness {
    /// Base GT element
    pub base: Fq12,
    /// Scalar exponent
    pub exponent: Fr,
    /// Result of exponentiation
    pub result: Fq12,
    /// Rho MLEs (accumulated results at each bit)
    pub rho_mles: Vec<Vec<Fq>>,
    /// Quotient MLEs for constraint verification
    pub quotient_mles: Vec<Vec<Fq>>,
    /// Binary representation of the scalar
    pub bits: Vec<bool>,
}

/// Per-operation witness for a single GT multiplication (used by GTCombineWitness).
///
/// Captures the intermediate values for one GT field multiplication.
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct GTMulOpWitness {
    /// Left operand
    pub lhs: Fq12,
    /// Right operand
    pub rhs: Fq12,
    /// Result of multiplication
    pub result: Fq12,
    /// Quotient MLE for constraint verification
    pub quotient_mle: Vec<Fq>,
}

/// Witness data for G1 scalar multiplication constraints
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct G1ScalarMulWitness {
    /// Base points for scalar multiplication
    pub base_points: Vec<G1Affine>,
    /// Scalars for multiplication
    pub scalars: Vec<Fr>,
    /// x-coordinate MLEs of accumulator point A
    pub x_a_mles: Vec<Vec<Fq>>,
    /// y-coordinate MLEs of accumulator point A
    pub y_a_mles: Vec<Vec<Fq>>,
    /// x-coordinate MLEs of temporary point T
    pub x_t_mles: Vec<Vec<Fq>>,
    /// y-coordinate MLEs of temporary point T
    pub y_t_mles: Vec<Vec<Fq>>,
    /// x-coordinate MLEs of next accumulator point A'
    pub x_a_next_mles: Vec<Vec<Fq>>,
    /// y-coordinate MLEs of next accumulator point A'
    pub y_a_next_mles: Vec<Vec<Fq>>,
    /// Infinity indicator MLEs for T
    pub t_is_infinity_mles: Vec<Vec<Fq>>,
}

impl G1ScalarMulWitness {
    /// Create a new G1 scalar multiplication witness
    pub fn new(base_points: Vec<G1Affine>, scalars: Vec<Fr>) -> Self {
        Self {
            base_points,
            scalars,
            x_a_mles: Vec::new(),
            y_a_mles: Vec::new(),
            x_t_mles: Vec::new(),
            y_t_mles: Vec::new(),
            x_a_next_mles: Vec::new(),
            y_a_next_mles: Vec::new(),
            t_is_infinity_mles: Vec::new(),
        }
    }

    /// Compute witness values (placeholder - actual implementation in Dory)
    pub fn compute_witness(&mut self) {
        // This would normally be populated by Dory's witness generation
    }
}

/// Witness data for G1 addition constraints (aggregated)
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct G1AddWitness {
    pub x_p_mles: Vec<Vec<Fq>>,
    pub y_p_mles: Vec<Vec<Fq>>,
    pub ind_p_mles: Vec<Vec<Fq>>,
    pub x_q_mles: Vec<Vec<Fq>>,
    pub y_q_mles: Vec<Vec<Fq>>,
    pub ind_q_mles: Vec<Vec<Fq>>,
    pub x_r_mles: Vec<Vec<Fq>>,
    pub y_r_mles: Vec<Vec<Fq>>,
    pub ind_r_mles: Vec<Vec<Fq>>,
    pub lambda_mles: Vec<Vec<Fq>>,
    pub inv_dx_mles: Vec<Vec<Fq>>,
    pub is_double_mles: Vec<Vec<Fq>>,
    pub is_inverse_mles: Vec<Vec<Fq>>,
}

/// Witness data for a single G1 addition instance (used by DoryMatrixBuilder)
#[derive(Clone, Debug)]
pub struct G1AddInstanceWitness {
    pub x_p: Vec<Fq>,
    pub y_p: Vec<Fq>,
    pub ind_p: Vec<Fq>,
    pub x_q: Vec<Fq>,
    pub y_q: Vec<Fq>,
    pub ind_q: Vec<Fq>,
    pub x_r: Vec<Fq>,
    pub y_r: Vec<Fq>,
    pub ind_r: Vec<Fq>,
    pub lambda: Vec<Fq>,
    pub inv_dx: Vec<Fq>,
    pub is_double: Vec<Fq>,
    pub is_inverse: Vec<Fq>,
}

/// Witness data for G2 addition constraints (aggregated)
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct G2AddWitness {
    pub x_p_c0_mles: Vec<Vec<Fq>>,
    pub x_p_c1_mles: Vec<Vec<Fq>>,
    pub y_p_c0_mles: Vec<Vec<Fq>>,
    pub y_p_c1_mles: Vec<Vec<Fq>>,
    pub ind_p_mles: Vec<Vec<Fq>>,
    pub x_q_c0_mles: Vec<Vec<Fq>>,
    pub x_q_c1_mles: Vec<Vec<Fq>>,
    pub y_q_c0_mles: Vec<Vec<Fq>>,
    pub y_q_c1_mles: Vec<Vec<Fq>>,
    pub ind_q_mles: Vec<Vec<Fq>>,
    pub x_r_c0_mles: Vec<Vec<Fq>>,
    pub x_r_c1_mles: Vec<Vec<Fq>>,
    pub y_r_c0_mles: Vec<Vec<Fq>>,
    pub y_r_c1_mles: Vec<Vec<Fq>>,
    pub ind_r_mles: Vec<Vec<Fq>>,
    pub lambda_c0_mles: Vec<Vec<Fq>>,
    pub lambda_c1_mles: Vec<Vec<Fq>>,
    pub inv_dx_c0_mles: Vec<Vec<Fq>>,
    pub inv_dx_c1_mles: Vec<Vec<Fq>>,
    pub is_double_mles: Vec<Vec<Fq>>,
    pub is_inverse_mles: Vec<Vec<Fq>>,
}

/// Witness data for a single G2 addition instance (used by DoryMatrixBuilder)
#[derive(Clone, Debug)]
pub struct G2AddInstanceWitness {
    pub x_p_c0: Vec<Fq>,
    pub x_p_c1: Vec<Fq>,
    pub y_p_c0: Vec<Fq>,
    pub y_p_c1: Vec<Fq>,
    pub ind_p: Vec<Fq>,
    pub x_q_c0: Vec<Fq>,
    pub x_q_c1: Vec<Fq>,
    pub y_q_c0: Vec<Fq>,
    pub y_q_c1: Vec<Fq>,
    pub ind_q: Vec<Fq>,
    pub x_r_c0: Vec<Fq>,
    pub x_r_c1: Vec<Fq>,
    pub y_r_c0: Vec<Fq>,
    pub y_r_c1: Vec<Fq>,
    pub ind_r: Vec<Fq>,
    pub lambda_c0: Vec<Fq>,
    pub lambda_c1: Vec<Fq>,
    pub inv_dx_c0: Vec<Fq>,
    pub inv_dx_c1: Vec<Fq>,
    pub is_double: Vec<Fq>,
    pub is_inverse: Vec<Fq>,
}

/// Witness for homomorphic combination of GT commitments.
///
/// Captures the intermediate witnesses for computing:
/// `result = sum_i(coeff_i * commitment_i)`
///
/// Uses a linear fold: first compute all scaled commitments via GT exponentiation,
/// then accumulate via sequential GT multiplications.
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct GTCombineWitness {
    /// Exponentiation witnesses: `scaled[i] = coeff[i] * commitment[i]`
    pub exp_witnesses: Vec<GTExpOpWitness>,
    /// Multiplication witnesses for linear fold: `acc[i] = acc[i-1] * scaled[i]`
    pub mul_witnesses: Vec<GTMulOpWitness>,
}

/// Combined witness data for all recursion constraints
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryRecursionWitness {
    /// GT exponentiation witness
    pub gt_exp_witness: GTExpWitness,
    /// GT multiplication witness
    pub gt_mul_witness: GTMulWitness,
    /// G1 scalar multiplication witness
    pub g1_scalar_mul_witness: G1ScalarMulWitness,
    /// G1 addition witness
    pub g1_add_witness: G1AddWitness,
    /// G2 addition witness
    pub g2_add_witness: G2AddWitness,
    /// Witness for combine_commitments offloading
    pub combine_witness: Option<GTCombineWitness>,
}

/// Extended witness data including recursion witnesses
#[derive(Clone, Debug, Default)]
pub struct WitnessData {
    /// Recursion witness data (optional)
    pub recursion: Option<DoryRecursionWitness>,
    // Other witness fields would go here
}
