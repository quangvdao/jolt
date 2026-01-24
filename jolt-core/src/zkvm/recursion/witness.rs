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

/// Witness data for multi-Miller loop constraints (aggregated)
#[cfg(feature = "experimental-pairing-recursion")]
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct MultiMillerLoopWitness {
    /// Accumulator f(s, x) - packed 11-var MLE
    pub f_packed: Vec<Vec<Fq>>,
    /// Quotient Q(s, x) - packed 11-var MLE
    pub quotient_packed: Vec<Vec<Fq>>,
    /// G2 state x-coordinate c0 (packed)
    pub t_x_c0_packed: Vec<Vec<Fq>>,
    /// G2 state x-coordinate c1 (packed)
    pub t_x_c1_packed: Vec<Vec<Fq>>,
    /// G2 state y-coordinate c0 (packed)
    pub t_y_c0_packed: Vec<Vec<Fq>>,
    /// G2 state y-coordinate c1 (packed)
    pub t_y_c1_packed: Vec<Vec<Fq>>,
    /// Slope lambda c0 (packed)
    pub lambda_c0_packed: Vec<Vec<Fq>>,
    /// Slope lambda c1 (packed)
    pub lambda_c1_packed: Vec<Vec<Fq>>,
    /// Inverse dx c0 (packed)
    pub inv_dx_c0_packed: Vec<Vec<Fq>>,
    /// Inverse dx c1 (packed)
    pub inv_dx_c1_packed: Vec<Vec<Fq>>,
    /// Line coeff c0 c0 (packed) - c0 of the first coefficient
    pub l_c0_c0_packed: Vec<Vec<Fq>>,
    /// Line coeff c0 c1 (packed)
    pub l_c0_c1_packed: Vec<Vec<Fq>>,
    /// Line coeff c1 c0 (packed)
    pub l_c1_c0_packed: Vec<Vec<Fq>>,
    /// Line coeff c1 c1 (packed)
    pub l_c1_c1_packed: Vec<Vec<Fq>>,
    /// Number of steps in the Miller loop
    pub num_steps: usize,
}

/// Witness data for a single Multi-Miller loop instance (used by DoryMatrixBuilder)
#[cfg(feature = "experimental-pairing-recursion")]
#[derive(Clone, Debug)]
pub struct MultiMillerLoopInstanceWitness {
    pub f_packed: Vec<Fq>,
    pub quotient_packed: Vec<Fq>,
    pub t_x_c0_packed: Vec<Fq>,
    pub t_x_c1_packed: Vec<Fq>,
    pub t_y_c0_packed: Vec<Fq>,
    pub t_y_c1_packed: Vec<Fq>,
    pub lambda_c0_packed: Vec<Fq>,
    pub lambda_c1_packed: Vec<Fq>,
    pub inv_dx_c0_packed: Vec<Fq>,
    pub inv_dx_c1_packed: Vec<Fq>,
    pub l_c0_c0_packed: Vec<Fq>,
    pub l_c0_c1_packed: Vec<Fq>,
    pub l_c1_c0_packed: Vec<Fq>,
    pub l_c1_c1_packed: Vec<Fq>,
    pub num_steps: usize,
}

/// Witness for homomorphic combination of GT commitments.
///
/// Captures the intermediate witnesses for computing:
/// `result = sum_i(coeff_i * commitment_i)`
///
/// Uses a balanced fold: first compute all scaled commitments via GT exponentiation,
/// then accumulate via a deterministic balanced binary-tree of GT multiplications.
///
/// The tree shape is **fully deterministic** given `exp_witnesses.len()`:
/// - Level 0 inputs are `exp_witnesses[i].result` for i=0..n-1
/// - Each level pairs adjacent elements left-to-right; if the level has an odd
///   number of nodes, the final node is carried forward unchanged.
/// - `mul_layers[level][j]` multiplies the pair (2j, 2j+1) from the previous level.
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct GTCombineWitness {
    /// Exponentiation witnesses: `scaled[i] = coeff[i] * commitment[i]`
    pub exp_witnesses: Vec<GTExpOpWitness>,
    /// Multiplication witnesses, grouped by fold level (left-to-right per level).
    pub mul_layers: Vec<Vec<GTMulOpWitness>>,
}

impl GTCombineWitness {
    /// Validate the deterministic balanced-tree wiring.
    ///
    /// This checks:
    /// - Each `mul_layers[level][j]` multiplies the expected children from the previous level
    ///   (adjacent pairing, left-to-right; odd tail carried forward).
    /// - Each mul witness's `result` matches `lhs * rhs`.
    ///
    /// This is intended as a cheap structural invariant check to support future
    /// explicit wiring/equality constraints.
    pub fn validate_tree_wiring(&self) -> Result<(), String> {
        if self.exp_witnesses.is_empty() {
            return Err("GTCombineWitness.exp_witnesses is empty".to_owned());
        }

        let mut prev: Vec<Fq12> = self.exp_witnesses.iter().map(|w| w.result).collect();

        for (level, layer_wits) in self.mul_layers.iter().enumerate() {
            let expected_pairs = prev.len() / 2;
            if layer_wits.len() != expected_pairs {
                return Err(format!(
                    "GTCombineWitness.mul_layers[{level}] wrong length: got {}, expected {} (prev_len={})",
                    layer_wits.len(),
                    expected_pairs,
                    prev.len()
                ));
            }

            let mut next = Vec::with_capacity((prev.len() + 1) / 2);
            for j in 0..expected_pairs {
                let expected_lhs = prev[2 * j];
                let expected_rhs = prev[2 * j + 1];
                let wit = &layer_wits[j];

                if wit.lhs != expected_lhs {
                    return Err(format!(
                        "mul_layers[{level}][{j}].lhs mismatch: got {:?}, expected {:?}",
                        wit.lhs, expected_lhs
                    ));
                }
                if wit.rhs != expected_rhs {
                    return Err(format!(
                        "mul_layers[{level}][{j}].rhs mismatch: got {:?}, expected {:?}",
                        wit.rhs, expected_rhs
                    ));
                }

                let expected_result = expected_lhs * expected_rhs;
                if wit.result != expected_result {
                    return Err(format!(
                        "mul_layers[{level}][{j}].result mismatch: got {:?}, expected {:?}",
                        wit.result, expected_result
                    ));
                }

                next.push(wit.result);
            }

            // Carry forward odd tail, if any.
            if prev.len() % 2 == 1 {
                next.push(*prev.last().unwrap());
            }
            prev = next;
        }

        // After applying all layers, we should have reduced to a single accumulator.
        if prev.len() != 1 {
            return Err(format!(
                "GTCombineWitness did not reduce to a single value: final_len={}",
                prev.len()
            ));
        }
        Ok(())
    }
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
    /// Multi-Miller loop witness (experimental; gated behind `experimental-pairing-recursion`)
    #[cfg(feature = "experimental-pairing-recursion")]
    pub multi_miller_loop_witness: MultiMillerLoopWitness,
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
