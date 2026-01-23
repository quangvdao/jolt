//! Jolt implementation of Dory's recursion backend

use ark_bn254::{Fq, Fq12, Fr, G1Affine, G2Affine};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Read, SerializationError, Write};
use dory::{
    backends::arkworks::{ArkG1, ArkG2, ArkGT, BN254},
    primitives::arithmetic::{Group, PairingCurve},
    primitives::serialization::{DoryDeserialize, DorySerialize},
    recursion::{
        ast::AstGraph, HintMap, TraceContext, WitnessBackend, WitnessGenerator, WitnessResult,
    },
    verify_recursive,
};
use std::{marker::PhantomData, rc::Rc};

/// Wrapper for `HintMap` that implements ark's serialization traits.
///
/// This bridges dory's `DorySerialize`/`DoryDeserialize` to ark's
/// `CanonicalSerialize`/`CanonicalDeserialize` for proof transport.
#[derive(Clone)]
pub struct JoltHintMap(pub HintMap<BN254>);

impl CanonicalSerialize for JoltHintMap {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        use dory::primitives::serialization::Compress as DoryCompress;

        // First serialize to a buffer to get the length
        let mut buffer = Vec::new();
        self.0
            .serialize_with_mode(&mut buffer, DoryCompress::Yes)
            .map_err(|_| SerializationError::InvalidData)?;

        // Write length prefix + data
        let len = buffer.len() as u64;
        <u64 as CanonicalSerialize>::serialize_compressed(&len, &mut writer)?;
        writer
            .write_all(&buffer)
            .map_err(SerializationError::from)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        use dory::primitives::serialization::Compress as DoryCompress;
        // Size = 8 bytes for length + serialized data
        8 + self.0.serialized_size(DoryCompress::Yes)
    }
}

impl ark_serialize::Valid for JoltHintMap {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for JoltHintMap {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        use dory::primitives::serialization::{Compress as DoryCompress, Validate as DoryValidate};

        // Read length prefix using ark's deserialize
        let len = <u64 as CanonicalDeserialize>::deserialize_compressed(&mut reader)? as usize;
        // Read data
        let mut bytes = vec![0u8; len];
        reader
            .read_exact(&mut bytes)
            .map_err(SerializationError::from)?;
        // Deserialize using dory's format
        let hint_map =
            HintMap::deserialize_with_mode(&bytes[..], DoryCompress::Yes, DoryValidate::Yes)
                .map_err(|_| SerializationError::InvalidData)?;
        Ok(JoltHintMap(hint_map))
    }
}

use super::{
    commitment_scheme::DoryCommitmentScheme,
    g1_scalar_mul_witness::ScalarMultiplicationSteps,
    g2_scalar_mul_witness::G2ScalarMultiplicationSteps,
    gt_exp_witness::Base4ExponentiationSteps,
    gt_mul_witness::MultiplicationSteps,
    jolt_dory_routines::{JoltG1Routines, JoltG2Routines},
    multi_miller_loop_witness::MultiMillerLoopSteps,
    wrappers::{
        ark_to_jolt, jolt_to_ark, ArkDoryProof, ArkFr, ArkworksVerifierSetup, JoltToDoryTranscript,
    },
};
use crate::poly::commitment::commitment_scheme::RecursionExt;
use crate::utils::errors::ProofVerifyError;
use crate::zkvm::recursion::witness::{GTCombineWitness, GTExpOpWitness, GTMulOpWitness};

/// Jolt witness backend implementation for dory recursion
#[derive(Debug, Clone)]
pub struct JoltWitness;

/// GTExp witness following the ExponentiationSteps pattern
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltGtExpWitness {
    pub base: Fq12,
    pub exponent: Fr,
    pub result: Fq12,
    pub rho_mles: Vec<Vec<Fq>>,
    pub quotient_mles: Vec<Vec<Fq>>,
    pub bits: Vec<bool>,
    ark_result: ArkGT,
}

impl WitnessResult<ArkGT> for JoltGtExpWitness {
    fn result(&self) -> Option<&ArkGT> {
        Some(&self.ark_result)
    }
}

/// GTMul witness following the MultiplicationSteps pattern
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltGtMulWitness {
    pub lhs: Fq12,             // Left operand (a)
    pub rhs: Fq12,             // Right operand (b)
    pub result: Fq12,          // Product (c = a × b)
    pub quotient_mle: Vec<Fq>, // Quotient polynomial Q(x)
    ark_result: ArkGT,         // For WitnessResult trait
}

impl WitnessResult<ArkGT> for JoltGtMulWitness {
    fn result(&self) -> Option<&ArkGT> {
        Some(&self.ark_result)
    }
}

/// G1 scalar multiplication witness following the ScalarMultiplicationSteps pattern
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltG1ScalarMulWitness {
    pub point_base: G1Affine,
    pub scalar: Fr,
    pub result: G1Affine,
    pub x_a_mles: Vec<Vec<Fq>>,           // x-coords of A_0, A_1, ..., A_n
    pub y_a_mles: Vec<Vec<Fq>>,           // y-coords of A_0, A_1, ..., A_n
    pub x_t_mles: Vec<Vec<Fq>>,           // x-coords of T_0, T_1, ..., T_{n-1}
    pub y_t_mles: Vec<Vec<Fq>>,           // y-coords of T_0, T_1, ..., T_{n-1}
    pub x_a_next_mles: Vec<Vec<Fq>>,      // x-coords of A_{i+1} (shifted by 1)
    pub y_a_next_mles: Vec<Vec<Fq>>,      // y-coords of A_{i+1} (shifted by 1)
    pub t_is_infinity_mles: Vec<Vec<Fq>>, // 1 if T_i = O, 0 otherwise
    pub a_is_infinity_mles: Vec<Vec<Fq>>, // 1 if A_i = O, 0 otherwise
    pub bit_mles: Vec<Vec<Fq>>,           // Scalar bit b_i (CRITICAL for soundness)
    pub bits: Vec<bool>,
    ark_result: ArkG1,
}

impl WitnessResult<ArkG1> for JoltG1ScalarMulWitness {
    fn result(&self) -> Option<&ArkG1> {
        Some(&self.ark_result)
    }
}

/// G2 scalar multiplication witness following the ScalarMultiplicationSteps pattern,
/// with Fq2 coordinates split into (c0, c1) components in Fq.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltG2ScalarMulWitness {
    pub point_base: G2Affine,
    pub scalar: Fr,
    pub result: G2Affine,

    pub x_a_c0_mles: Vec<Vec<Fq>>,
    pub x_a_c1_mles: Vec<Vec<Fq>>,
    pub y_a_c0_mles: Vec<Vec<Fq>>,
    pub y_a_c1_mles: Vec<Vec<Fq>>,

    pub x_t_c0_mles: Vec<Vec<Fq>>,
    pub x_t_c1_mles: Vec<Vec<Fq>>,
    pub y_t_c0_mles: Vec<Vec<Fq>>,
    pub y_t_c1_mles: Vec<Vec<Fq>>,

    pub x_a_next_c0_mles: Vec<Vec<Fq>>,
    pub x_a_next_c1_mles: Vec<Vec<Fq>>,
    pub y_a_next_c0_mles: Vec<Vec<Fq>>,
    pub y_a_next_c1_mles: Vec<Vec<Fq>>,

    pub t_is_infinity_mles: Vec<Vec<Fq>>,
    pub a_is_infinity_mles: Vec<Vec<Fq>>,
    pub bit_mles: Vec<Vec<Fq>>,
    pub bits: Vec<bool>,
    ark_result: ArkG2,
}

impl WitnessResult<ArkG2> for JoltG2ScalarMulWitness {
    fn result(&self) -> Option<&ArkG2> {
        Some(&self.ark_result)
    }
}

/// G1 addition witness for Dory recursion.
///
/// For a single point addition P + Q = R, we store:
/// - The coordinates and infinity indicators for P, Q, R
/// - Auxiliary values: lambda (slope), inv_delta_x, is_double, is_inverse
///
/// Since G1Add is a single operation (not a trace), the MLE is constant:
/// all 2^11 entries have the same value.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltG1AddWitness {
    // Input P
    pub x_p: Fq,
    pub y_p: Fq,
    pub ind_p: Fq, // 1 if P = O (infinity), 0 otherwise

    // Input Q
    pub x_q: Fq,
    pub y_q: Fq,
    pub ind_q: Fq,

    // Output R = P + Q
    pub x_r: Fq,
    pub y_r: Fq,
    pub ind_r: Fq,

    // Auxiliary witness values
    pub lambda: Fq,      // slope
    pub inv_delta_x: Fq, // 1/(x_q - x_p) in add case
    pub is_double: Fq,   // 1 if P == Q
    pub is_inverse: Fq,  // 1 if P == -Q (result is O)

    // For WitnessResult trait
    ark_result: ArkG1,
}

impl WitnessResult<ArkG1> for JoltG1AddWitness {
    fn result(&self) -> Option<&ArkG1> {
        Some(&self.ark_result)
    }
}

/// G2 addition witness for Dory recursion.
///
/// Same as G1 but with Fq2 coordinates split into (c0, c1) components.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltG2AddWitness {
    // Input P (Fq2 coordinates)
    pub x_p_c0: Fq,
    pub x_p_c1: Fq,
    pub y_p_c0: Fq,
    pub y_p_c1: Fq,
    pub ind_p: Fq,

    // Input Q
    pub x_q_c0: Fq,
    pub x_q_c1: Fq,
    pub y_q_c0: Fq,
    pub y_q_c1: Fq,
    pub ind_q: Fq,

    // Output R = P + Q
    pub x_r_c0: Fq,
    pub x_r_c1: Fq,
    pub y_r_c0: Fq,
    pub y_r_c1: Fq,
    pub ind_r: Fq,

    // Auxiliary witness values (Fq2)
    pub lambda_c0: Fq,
    pub lambda_c1: Fq,
    pub inv_delta_x_c0: Fq,
    pub inv_delta_x_c1: Fq,
    pub is_double: Fq,
    pub is_inverse: Fq,

    // For WitnessResult trait
    ark_result: ArkG2,
}

impl WitnessResult<ArkG2> for JoltG2AddWitness {
    fn result(&self) -> Option<&ArkG2> {
        Some(&self.ark_result)
    }
}

/// Multi-Miller loop witness for Dory recursion.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltMultiMillerLoopWitness {
    pub f_packed_mles: Vec<Vec<Fq>>,
    pub quotient_packed_mles: Vec<Vec<Fq>>,
    pub t_x_c0_packed_mles: Vec<Vec<Fq>>,
    pub t_x_c1_packed_mles: Vec<Vec<Fq>>,
    pub t_y_c0_packed_mles: Vec<Vec<Fq>>,
    pub t_y_c1_packed_mles: Vec<Vec<Fq>>,
    pub lambda_c0_packed_mles: Vec<Vec<Fq>>,
    pub lambda_c1_packed_mles: Vec<Vec<Fq>>,
    pub inv_dx_c0_packed_mles: Vec<Vec<Fq>>,
    pub inv_dx_c1_packed_mles: Vec<Vec<Fq>>,
    pub l_c0_c0_packed_mles: Vec<Vec<Fq>>,
    pub l_c0_c1_packed_mles: Vec<Vec<Fq>>,
    pub l_c1_c0_packed_mles: Vec<Vec<Fq>>,
    pub l_c1_c1_packed_mles: Vec<Vec<Fq>>,
    pub num_steps: usize,
    ark_result: ArkGT,
}

impl WitnessResult<ArkGT> for JoltMultiMillerLoopWitness {
    fn result(&self) -> Option<&ArkGT> {
        Some(&self.ark_result)
    }
}

/// Witness type for unimplemented operations that panics when used
#[derive(Clone, Debug)]
pub struct UnimplementedWitness<T> {
    _operation: &'static str,
    _marker: PhantomData<T>,
}

impl<T> UnimplementedWitness<T> {
    fn new(operation: &'static str) -> Self {
        Self {
            _operation: operation,
            _marker: PhantomData,
        }
    }
}

impl WitnessResult<ArkG1> for UnimplementedWitness<ArkG1> {
    fn result(&self) -> Option<&ArkG1> {
        None
    }
}

impl WitnessResult<ArkG2> for UnimplementedWitness<ArkG2> {
    fn result(&self) -> Option<&ArkG2> {
        None
    }
}

impl WitnessResult<ArkGT> for UnimplementedWitness<ArkGT> {
    fn result(&self) -> Option<&ArkGT> {
        None
    }
}

impl WitnessBackend for JoltWitness {
    // G1 operations
    type G1AddWitness = JoltG1AddWitness;
    type G1ScalarMulWitness = JoltG1ScalarMulWitness;
    type MsmG1Witness = UnimplementedWitness<ArkG1>;

    // G2 operations
    type G2AddWitness = JoltG2AddWitness;
    type G2ScalarMulWitness = JoltG2ScalarMulWitness;
    type MsmG2Witness = UnimplementedWitness<ArkG2>;

    // GT operations
    type GtMulWitness = JoltGtMulWitness;
    type GtExpWitness = JoltGtExpWitness;

    // Pairing operations
    type PairingWitness = JoltMultiMillerLoopWitness;
    type MultiPairingWitness = JoltMultiMillerLoopWitness;
}

pub struct JoltWitnessGenerator;

impl WitnessGenerator<JoltWitness, BN254> for JoltWitnessGenerator {
    fn generate_g1_add(
        a: &<BN254 as PairingCurve>::G1,
        b: &<BN254 as PairingCurve>::G1,
        result: &<BN254 as PairingCurve>::G1,
    ) -> JoltG1AddWitness {
        use ark_ec::AffineRepr;
        use ark_ff::Field;

        let p: G1Affine = a.0.into();
        let q: G1Affine = b.0.into();
        let r: G1Affine = result.0.into();

        let zero = Fq::from(0u64);
        let one = Fq::from(1u64);

        // Extract coordinates (0 for infinity points)
        let (x_p, y_p, ind_p) = if p.is_zero() {
            (zero, zero, one)
        } else {
            (p.x, p.y, zero)
        };

        let (x_q, y_q, ind_q) = if q.is_zero() {
            (zero, zero, one)
        } else {
            (q.x, q.y, zero)
        };

        let (x_r, y_r, ind_r) = if r.is_zero() {
            (zero, zero, one)
        } else {
            (r.x, r.y, zero)
        };

        // Compute auxiliary witness values
        let (lambda, inv_delta_x, is_double, is_inverse) = if ind_p == one || ind_q == one {
            // At least one input is infinity - no slope needed
            (zero, zero, zero, zero)
        } else {
            let dx = x_q - x_p;
            let dy = y_q - y_p;

            if dx == zero {
                if dy == zero {
                    // P == Q: doubling case
                    // lambda = 3*x_p^2 / (2*y_p)
                    let two = Fq::from(2u64);
                    let three = Fq::from(3u64);
                    let numerator = three * x_p * x_p;
                    let denominator = two * y_p;
                    let lam = if denominator == zero {
                        zero // Edge case: y_p = 0 means P = -P, result is O
                    } else {
                        numerator * denominator.inverse().unwrap()
                    };
                    (lam, zero, one, zero)
                } else {
                    // P == -Q: inverse case, result is infinity
                    (zero, zero, zero, one)
                }
            } else {
                // General add case: lambda = dy/dx
                let inv_dx = dx.inverse().unwrap();
                let lam = dy * inv_dx;
                (lam, inv_dx, zero, zero)
            }
        };

        JoltG1AddWitness {
            x_p,
            y_p,
            ind_p,
            x_q,
            y_q,
            ind_q,
            x_r,
            y_r,
            ind_r,
            lambda,
            inv_delta_x,
            is_double,
            is_inverse,
            ark_result: *result,
        }
    }

    fn generate_g2_add(
        a: &<BN254 as PairingCurve>::G2,
        b: &<BN254 as PairingCurve>::G2,
        result: &<BN254 as PairingCurve>::G2,
    ) -> JoltG2AddWitness {
        use ark_bn254::Fq2;
        use ark_ec::AffineRepr;
        use ark_ff::Field;

        let p: G2Affine = a.0.into();
        let q: G2Affine = b.0.into();
        let r: G2Affine = result.0.into();

        let zero = Fq::from(0u64);
        let one = Fq::from(1u64);
        let fq2_zero = Fq2::from(0u64);

        // Extract Fq2 coordinates split into (c0, c1)
        let (x_p, y_p, ind_p) = if p.is_zero() {
            (fq2_zero, fq2_zero, one)
        } else {
            (p.x, p.y, zero)
        };

        let (x_q, y_q, ind_q) = if q.is_zero() {
            (fq2_zero, fq2_zero, one)
        } else {
            (q.x, q.y, zero)
        };

        let (x_r, y_r, ind_r) = if r.is_zero() {
            (fq2_zero, fq2_zero, one)
        } else {
            (r.x, r.y, zero)
        };

        // Compute auxiliary witness values in Fq2
        let (lambda, inv_delta_x, is_double, is_inverse) = if ind_p == one || ind_q == one {
            (fq2_zero, fq2_zero, zero, zero)
        } else {
            let dx = x_q - x_p;
            let dy = y_q - y_p;

            if dx == fq2_zero {
                if dy == fq2_zero {
                    // Doubling case
                    let two = Fq2::from(2u64);
                    let three = Fq2::from(3u64);
                    let numerator = three * x_p * x_p;
                    let denominator = two * y_p;
                    let lam = if denominator == fq2_zero {
                        fq2_zero
                    } else {
                        numerator * denominator.inverse().unwrap()
                    };
                    (lam, fq2_zero, one, zero)
                } else {
                    // Inverse case
                    (fq2_zero, fq2_zero, zero, one)
                }
            } else {
                // General add
                let inv_dx = dx.inverse().unwrap();
                let lam = dy * inv_dx;
                (lam, inv_dx, zero, zero)
            }
        };

        JoltG2AddWitness {
            x_p_c0: x_p.c0,
            x_p_c1: x_p.c1,
            y_p_c0: y_p.c0,
            y_p_c1: y_p.c1,
            ind_p,
            x_q_c0: x_q.c0,
            x_q_c1: x_q.c1,
            y_q_c0: y_q.c0,
            y_q_c1: y_q.c1,
            ind_q,
            x_r_c0: x_r.c0,
            x_r_c1: x_r.c1,
            y_r_c0: y_r.c0,
            y_r_c1: y_r.c1,
            ind_r,
            lambda_c0: lambda.c0,
            lambda_c1: lambda.c1,
            inv_delta_x_c0: inv_delta_x.c0,
            inv_delta_x_c1: inv_delta_x.c1,
            is_double,
            is_inverse,
            ark_result: *result,
        }
    }

    fn generate_gt_exp(
        base: &<BN254 as PairingCurve>::GT,
        scalar: &<<BN254 as PairingCurve>::G1 as Group>::Scalar,
        result: &<BN254 as PairingCurve>::GT,
    ) -> JoltGtExpWitness {
        let base_fq12 = base.0;
        let scalar_fr = ark_to_jolt(scalar);

        let exp_steps = Base4ExponentiationSteps::new(base_fq12, scalar_fr);

        debug_assert_eq!(
            exp_steps.result, result.0,
            "ExponentiationSteps result doesn't match expected result"
        );

        JoltGtExpWitness {
            base: exp_steps.base,
            exponent: exp_steps.exponent,
            result: exp_steps.result,
            rho_mles: exp_steps.rho_mles,
            quotient_mles: exp_steps.quotient_mles,
            bits: exp_steps.bits,
            ark_result: *result,
        }
    }

    fn generate_g1_scalar_mul(
        point: &<BN254 as PairingCurve>::G1,
        scalar: &<<BN254 as PairingCurve>::G1 as Group>::Scalar,
        result: &<BN254 as PairingCurve>::G1,
    ) -> JoltG1ScalarMulWitness {
        let point_affine: G1Affine = point.0.into();
        let scalar_fr = ark_to_jolt(scalar);

        let scalar_mul_steps = ScalarMultiplicationSteps::new(point_affine, scalar_fr);

        let result_affine: G1Affine = result.0.into();
        debug_assert_eq!(
            scalar_mul_steps.result, result_affine,
            "ScalarMultiplicationSteps result doesn't match expected result"
        );

        JoltG1ScalarMulWitness {
            point_base: scalar_mul_steps.point_base,
            scalar: scalar_mul_steps.scalar,
            result: scalar_mul_steps.result,
            x_a_mles: scalar_mul_steps.x_a_mles,
            y_a_mles: scalar_mul_steps.y_a_mles,
            x_t_mles: scalar_mul_steps.x_t_mles,
            y_t_mles: scalar_mul_steps.y_t_mles,
            x_a_next_mles: scalar_mul_steps.x_a_next_mles,
            y_a_next_mles: scalar_mul_steps.y_a_next_mles,
            t_is_infinity_mles: scalar_mul_steps.t_is_infinity_mles,
            a_is_infinity_mles: scalar_mul_steps.a_is_infinity_mles,
            bit_mles: scalar_mul_steps.bit_mles,
            bits: scalar_mul_steps.bits,
            ark_result: *result,
        }
    }

    fn generate_g2_scalar_mul(
        point: &<BN254 as PairingCurve>::G2,
        scalar: &<<BN254 as PairingCurve>::G1 as Group>::Scalar,
        result: &<BN254 as PairingCurve>::G2,
    ) -> JoltG2ScalarMulWitness {
        let point_affine: G2Affine = point.0.into();
        let scalar_fr = ark_to_jolt(scalar);

        let scalar_mul_steps = G2ScalarMultiplicationSteps::new(point_affine, scalar_fr);

        let result_affine: G2Affine = result.0.into();
        debug_assert_eq!(
            scalar_mul_steps.result, result_affine,
            "G2ScalarMultiplicationSteps result doesn't match expected result"
        );

        JoltG2ScalarMulWitness {
            point_base: scalar_mul_steps.point_base,
            scalar: scalar_mul_steps.scalar,
            result: scalar_mul_steps.result,
            x_a_c0_mles: scalar_mul_steps.x_a_c0_mles,
            x_a_c1_mles: scalar_mul_steps.x_a_c1_mles,
            y_a_c0_mles: scalar_mul_steps.y_a_c0_mles,
            y_a_c1_mles: scalar_mul_steps.y_a_c1_mles,
            x_t_c0_mles: scalar_mul_steps.x_t_c0_mles,
            x_t_c1_mles: scalar_mul_steps.x_t_c1_mles,
            y_t_c0_mles: scalar_mul_steps.y_t_c0_mles,
            y_t_c1_mles: scalar_mul_steps.y_t_c1_mles,
            x_a_next_c0_mles: scalar_mul_steps.x_a_next_c0_mles,
            x_a_next_c1_mles: scalar_mul_steps.x_a_next_c1_mles,
            y_a_next_c0_mles: scalar_mul_steps.y_a_next_c0_mles,
            y_a_next_c1_mles: scalar_mul_steps.y_a_next_c1_mles,
            t_is_infinity_mles: scalar_mul_steps.t_is_infinity_mles,
            a_is_infinity_mles: scalar_mul_steps.a_is_infinity_mles,
            bit_mles: scalar_mul_steps.bit_mles,
            bits: scalar_mul_steps.bits,
            ark_result: *result,
        }
    }

    fn generate_gt_mul(
        lhs: &<BN254 as PairingCurve>::GT,
        rhs: &<BN254 as PairingCurve>::GT,
        result: &<BN254 as PairingCurve>::GT,
    ) -> JoltGtMulWitness {
        let lhs_fq12 = lhs.0;
        let rhs_fq12 = rhs.0;

        let mul_steps = MultiplicationSteps::new(lhs_fq12, rhs_fq12);

        debug_assert_eq!(
            mul_steps.result, result.0,
            "MultiplicationSteps result doesn't match expected result"
        );

        JoltGtMulWitness {
            lhs: mul_steps.lhs,
            rhs: mul_steps.rhs,
            result: mul_steps.result,
            quotient_mle: mul_steps.quotient_mle,
            ark_result: *result,
        }
    }

    fn generate_pairing(
        g1: &<BN254 as PairingCurve>::G1,
        g2: &<BN254 as PairingCurve>::G2,
        result: &<BN254 as PairingCurve>::GT,
    ) -> JoltMultiMillerLoopWitness {
        let g1_affine: G1Affine = g1.0.into();
        let g2_affine: G2Affine = g2.0.into();

        let steps = MultiMillerLoopSteps::new(&[g1_affine], &[g2_affine]);

        // TODO: Verify result matches steps.result (after final exp)
        // For now we just store the result

        JoltMultiMillerLoopWitness {
            f_packed_mles: steps.f_packed_mles,
            quotient_packed_mles: steps.quotient_packed_mles,
            t_x_c0_packed_mles: steps.t_x_c0_packed_mles,
            t_x_c1_packed_mles: steps.t_x_c1_packed_mles,
            t_y_c0_packed_mles: steps.t_y_c0_packed_mles,
            t_y_c1_packed_mles: steps.t_y_c1_packed_mles,
            lambda_c0_packed_mles: steps.lambda_c0_packed_mles,
            lambda_c1_packed_mles: steps.lambda_c1_packed_mles,
            inv_dx_c0_packed_mles: steps.inv_dx_c0_packed_mles,
            inv_dx_c1_packed_mles: steps.inv_dx_c1_packed_mles,
            l_c0_c0_packed_mles: steps.l_c0_c0_packed_mles,
            l_c0_c1_packed_mles: steps.l_c0_c1_packed_mles,
            l_c1_c0_packed_mles: steps.l_c1_c0_packed_mles,
            l_c1_c1_packed_mles: steps.l_c1_c1_packed_mles,
            num_steps: steps.num_steps,
            ark_result: *result,
        }
    }

    fn generate_multi_pairing(
        g1s: &[<BN254 as PairingCurve>::G1],
        g2s: &[<BN254 as PairingCurve>::G2],
        result: &<BN254 as PairingCurve>::GT,
    ) -> JoltMultiMillerLoopWitness {
        let g1_affines: Vec<G1Affine> = g1s.iter().map(|p| p.0.into()).collect();
        let g2_affines: Vec<G2Affine> = g2s.iter().map(|p| p.0.into()).collect();

        let steps = MultiMillerLoopSteps::new(&g1_affines, &g2_affines);

        JoltMultiMillerLoopWitness {
            f_packed_mles: steps.f_packed_mles,
            quotient_packed_mles: steps.quotient_packed_mles,
            t_x_c0_packed_mles: steps.t_x_c0_packed_mles,
            t_x_c1_packed_mles: steps.t_x_c1_packed_mles,
            t_y_c0_packed_mles: steps.t_y_c0_packed_mles,
            t_y_c1_packed_mles: steps.t_y_c1_packed_mles,
            lambda_c0_packed_mles: steps.lambda_c0_packed_mles,
            lambda_c1_packed_mles: steps.lambda_c1_packed_mles,
            inv_dx_c0_packed_mles: steps.inv_dx_c0_packed_mles,
            inv_dx_c1_packed_mles: steps.inv_dx_c1_packed_mles,
            l_c0_c0_packed_mles: steps.l_c0_c0_packed_mles,
            l_c0_c1_packed_mles: steps.l_c0_c1_packed_mles,
            l_c1_c0_packed_mles: steps.l_c1_c0_packed_mles,
            l_c1_c1_packed_mles: steps.l_c1_c1_packed_mles,
            num_steps: steps.num_steps,
            ark_result: *result,
        }
    }

    fn generate_msm_g1(
        _bases: &[<BN254 as PairingCurve>::G1],
        _scalars: &[<<BN254 as PairingCurve>::G1 as Group>::Scalar],
        _result: &<BN254 as PairingCurve>::G1,
    ) -> UnimplementedWitness<ArkG1> {
        UnimplementedWitness::new("G1 MSM")
    }

    fn generate_msm_g2(
        _bases: &[<BN254 as PairingCurve>::G2],
        _scalars: &[<<BN254 as PairingCurve>::G1 as Group>::Scalar],
        _result: &<BN254 as PairingCurve>::G2,
    ) -> UnimplementedWitness<ArkG2> {
        UnimplementedWitness::new("G2 MSM")
    }
}

impl RecursionExt<Fr> for DoryCommitmentScheme {
    type Witness = dory::recursion::WitnessCollection<JoltWitness>;
    type Hint = JoltHintMap;
    type CombineHint = ArkGT;

    fn witness_gen<ProofTranscript: crate::transcripts::Transcript>(
        proof: &ArkDoryProof,
        setup: &ArkworksVerifierSetup,
        transcript: &mut ProofTranscript,
        point: &[<Fr as crate::field::JoltField>::Challenge],
        evaluation: &Fr,
        commitment: &ArkGT,
    ) -> Result<(Self::Witness, Self::Hint), ProofVerifyError> {
        // Convert Jolt types to dory types
        let reordered_point =
            crate::poly::commitment::dory::commitment_scheme::reorder_opening_point_for_layout::<Fr>(
                point,
            );
        let ark_point: Vec<ArkFr> = reordered_point
            .iter()
            .rev() // Reverse for dory endianness
            .map(|c| {
                let f_val: Fr = (*c).into();
                jolt_to_ark(&f_val)
            })
            .collect();
        let ark_evaluation = jolt_to_ark(evaluation);

        // Create witness generation context
        let ctx =
            Rc::new(TraceContext::<JoltWitness, BN254, JoltWitnessGenerator>::for_witness_gen());

        // Wrap transcript for dory compatibility
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        // Call verify_recursive to collect witnesses
        verify_recursive::<_, BN254, JoltG1Routines, JoltG2Routines, _, _, _>(
            *commitment,
            ark_evaluation,
            &ark_point,
            proof,
            setup.clone().into(),
            &mut dory_transcript,
            ctx.clone(),
        )
        .map_err(|_e| ProofVerifyError::default())?;

        // Extract witness collection
        let witnesses = Rc::try_unwrap(ctx)
            .ok()
            .expect("Should have sole ownership")
            .finalize()
            .ok_or(ProofVerifyError::default())?;

        // Convert witnesses to hints and wrap in JoltHintMap
        let hints = JoltHintMap(witnesses.to_hints::<BN254>());

        // Return both witnesses and hints
        Ok((witnesses, hints))
    }

    fn verify_with_hint<ProofTranscript: crate::transcripts::Transcript>(
        proof: &ArkDoryProof,
        setup: &ArkworksVerifierSetup,
        transcript: &mut ProofTranscript,
        point: &[<Fr as crate::field::JoltField>::Challenge],
        evaluation: &Fr,
        commitment: &ArkGT,
        hint: &Self::Hint,
    ) -> Result<(), ProofVerifyError> {
        // Convert point for dory
        let reordered_point =
            crate::poly::commitment::dory::commitment_scheme::reorder_opening_point_for_layout::<Fr>(
                point,
            );
        let ark_point: Vec<ArkFr> = reordered_point
            .iter()
            .rev() // Reverse for dory endianness
            .map(|c| {
                let f_val: Fr = (*c).into();
                jolt_to_ark(&f_val)
            })
            .collect();
        let ark_evaluation = jolt_to_ark(evaluation);

        // Create hint-based verification context (unwrap JoltHintMap)
        let ctx = Rc::new(
            TraceContext::<JoltWitness, BN254, JoltWitnessGenerator>::for_hints(hint.0.clone()),
        );

        // Wrap transcript for dory compatibility
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        // Verify using hints
        verify_recursive::<_, BN254, JoltG1Routines, JoltG2Routines, _, _, _>(
            *commitment,
            ark_evaluation,
            &ark_point,
            proof,
            setup.clone().into(),
            &mut dory_transcript,
            ctx,
        )
        .map_err(|_| ProofVerifyError::default())?;

        Ok(())
    }

    fn generate_combine_witness<C: std::borrow::Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Fr],
    ) -> (GTCombineWitness, Self::CombineHint) {
        assert!(!commitments.is_empty(), "commitments cannot be empty");
        assert_eq!(
            commitments.len(),
            coeffs.len(),
            "commitments and coeffs must have same length"
        );

        // Step 1: Generate exponentiation witnesses for each coeff * commitment
        let exp_witnesses: Vec<GTExpOpWitness> = commitments
            .iter()
            .zip(coeffs.iter())
            .map(|(comm, coeff)| {
                let comm_fq12 = comm.borrow().0;
                let exp_steps = Base4ExponentiationSteps::new(comm_fq12, *coeff);

                GTExpOpWitness {
                    base: exp_steps.base,
                    exponent: exp_steps.exponent,
                    result: exp_steps.result,
                    rho_mles: exp_steps.rho_mles,
                    quotient_mles: exp_steps.quotient_mles,
                    bits: exp_steps.bits,
                }
            })
            .collect();

        // Step 2: Linear fold with multiplication witnesses
        let mut mul_witnesses = Vec::with_capacity(exp_witnesses.len().saturating_sub(1));
        let mut accumulator = exp_witnesses[0].result;

        for exp_wit in &exp_witnesses[1..] {
            let mul_steps = MultiplicationSteps::new(accumulator, exp_wit.result);

            mul_witnesses.push(GTMulOpWitness {
                lhs: mul_steps.lhs,
                rhs: mul_steps.rhs,
                result: mul_steps.result,
                quotient_mle: mul_steps.quotient_mle,
            });

            accumulator = mul_steps.result;
        }

        let witness = GTCombineWitness {
            exp_witnesses,
            mul_witnesses,
        };

        // The hint is the final accumulated result
        let hint = ArkGT(accumulator);

        (witness, hint)
    }

    fn combine_with_hint(hint: &Self::CombineHint) -> Self::Commitment {
        *hint
    }

    fn combine_hint_to_fq12(hint: &Self::CombineHint) -> Fq12 {
        hint.0
    }

    fn combine_with_hint_fq12(hint: &Fq12) -> Self::Commitment {
        ArkGT(*hint)
    }
}

// ============================================================================
// AST-enabled witness generation
// ============================================================================

/// Result of witness generation with AST tracing enabled.
///
/// Contains everything needed for AST-ordered constraint building and wiring.
pub struct WitnessWithAst {
    /// Collected witnesses from all operations
    pub witnesses: dory::recursion::WitnessCollection<JoltWitness>,
    /// AST graph capturing the computation DAG
    pub ast: AstGraph<BN254>,
    /// Hints for efficient verification
    pub hints: JoltHintMap,
}

/// Generate witnesses with AST tracing enabled.
///
/// This is the entry point for recursion provers that need the AST for wiring constraints.
/// The AST captures the full computation DAG, enabling:
/// - Deterministic constraint ordering (by topological sort of AST nodes)
/// - Wiring constraint derivation (from AST edges)
/// - Boundary constraint identification (from AST roots and leaves)
pub fn witness_gen_with_ast<ProofTranscript: crate::transcripts::Transcript>(
    proof: &ArkDoryProof,
    setup: &ArkworksVerifierSetup,
    transcript: &mut ProofTranscript,
    point: &[<Fr as crate::field::JoltField>::Challenge],
    evaluation: &Fr,
    commitment: &ArkGT,
) -> Result<WitnessWithAst, ProofVerifyError> {
    // Convert Jolt types to dory types
    let reordered_point =
        crate::poly::commitment::dory::commitment_scheme::reorder_opening_point_for_layout::<Fr>(
            point,
        );
    let ark_point: Vec<ArkFr> = reordered_point
        .iter()
        .rev() // Reverse for dory endianness
        .map(|c| {
            let f_val: Fr = (*c).into();
            jolt_to_ark(&f_val)
        })
        .collect();
    let ark_evaluation = jolt_to_ark(evaluation);

    // Create witness generation context WITH AST tracing
    let ctx = Rc::new(
        TraceContext::<JoltWitness, BN254, JoltWitnessGenerator>::for_witness_gen_with_ast(),
    );

    // Wrap transcript for dory compatibility
    let mut dory_transcript = JoltToDoryTranscript::new(transcript);

    // Call verify_recursive to collect witnesses and build AST
    verify_recursive::<_, BN254, JoltG1Routines, JoltG2Routines, _, _, _>(
        *commitment,
        ark_evaluation,
        &ark_point,
        proof,
        setup.clone().into(),
        &mut dory_transcript,
        ctx.clone(),
    )
    .map_err(|_e| ProofVerifyError::default())?;

    // Extract both witnesses and AST
    let (witnesses_opt, ast_opt) = Rc::try_unwrap(ctx)
        .ok()
        .expect("Should have sole ownership")
        .finalize_with_ast();

    let witnesses = witnesses_opt.ok_or(ProofVerifyError::default())?;
    let ast = ast_opt.ok_or(ProofVerifyError::default())?;

    // Convert witnesses to hints
    let hints = JoltHintMap(witnesses.to_hints::<BN254>());

    Ok(WitnessWithAst {
        witnesses,
        ast,
        hints,
    })
}

/// Reconstruct AST from public inputs (verifier-side).
///
/// This re-runs verification with AST tracing but uses hints instead of
/// computing the expensive operations. The resulting AST is deterministic
/// and matches the prover's AST.
pub fn reconstruct_ast<ProofTranscript: crate::transcripts::Transcript>(
    proof: &ArkDoryProof,
    setup: &ArkworksVerifierSetup,
    transcript: &mut ProofTranscript,
    point: &[<Fr as crate::field::JoltField>::Challenge],
    evaluation: &Fr,
    commitment: &ArkGT,
    hints: &JoltHintMap,
) -> Result<AstGraph<BN254>, ProofVerifyError> {
    // Convert Jolt types to dory types
    let reordered_point =
        crate::poly::commitment::dory::commitment_scheme::reorder_opening_point_for_layout::<Fr>(
            point,
        );
    let ark_point: Vec<ArkFr> = reordered_point
        .iter()
        .rev()
        .map(|c| {
            let f_val: Fr = (*c).into();
            jolt_to_ark(&f_val)
        })
        .collect();
    let ark_evaluation = jolt_to_ark(evaluation);

    // Create hint-based context WITH AST tracing
    let ctx = Rc::new(
        TraceContext::<JoltWitness, BN254, JoltWitnessGenerator>::for_hints(hints.0.clone())
            .with_ast(),
    );

    // Wrap transcript for dory compatibility
    let mut dory_transcript = JoltToDoryTranscript::new(transcript);

    // Run verification with hints (fast) while building AST
    verify_recursive::<_, BN254, JoltG1Routines, JoltG2Routines, _, _, _>(
        *commitment,
        ark_evaluation,
        &ark_point,
        proof,
        setup.clone().into(),
        &mut dory_transcript,
        ctx.clone(),
    )
    .map_err(|_| ProofVerifyError::default())?;

    // Extract only the AST (witnesses are None in hint mode)
    let (_, ast_opt) = Rc::try_unwrap(ctx)
        .ok()
        .expect("Should have sole ownership")
        .finalize_with_ast();

    ast_opt.ok_or(ProofVerifyError::default())
}
