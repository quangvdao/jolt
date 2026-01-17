//! Jolt implementation of Dory's recursion backend

use ark_bn254::{Fq, Fq12, Fr, G1Affine};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid,
};
use dory::{
    backends::arkworks::{ArkG1, ArkG2, ArkGT, BN254},
    primitives::arithmetic::{Group, PairingCurve},
    recursion::{
        HintMap, OpId, OpType, TraceContext, WitnessBackend, WitnessGenerator, WitnessResult,
    },
    verify_recursive,
};
use jolt_optimizations::witness_gen::ExponentiationSteps;
use std::{marker::PhantomData, rc::Rc};

use super::{
    commitment_scheme::DoryCommitmentScheme,
    g1_scalar_mul_witness::ScalarMultiplicationSteps,
    gt_mul_witness::MultiplicationSteps,
    jolt_dory_routines::{JoltG1Routines, JoltG2Routines},
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

/// Canonically-serializable, deterministic hint format for Dory recursion verification.
///
/// `dory::recursion::HintMap` is backed by a `HashMap` and (in the current dory fork) only
/// implements Dory's custom serialization, which is not deterministic due to hash iteration
/// order. Jolt needs a deterministic encoding because hints are embedded into proofs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoryHintValue {
    G1(ArkG1),
    G2(ArkG2),
    GT(ArkGT),
}

impl CanonicalSerialize for DoryHintValue {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            DoryHintValue::G1(g1) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                g1.serialize_with_mode(&mut writer, compress)?;
            }
            DoryHintValue::G2(g2) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                g2.serialize_with_mode(&mut writer, compress)?;
            }
            DoryHintValue::GT(gt) => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                gt.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            DoryHintValue::G1(g1) => g1.serialized_size(compress),
            DoryHintValue::G2(g2) => g2.serialized_size(compress),
            DoryHintValue::GT(gt) => gt.serialized_size(compress),
        }
    }
}

impl CanonicalDeserialize for DoryHintValue {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        compress: Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match tag {
            0 => Ok(DoryHintValue::G1(ArkG1::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            1 => Ok(DoryHintValue::G2(ArkG2::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            2 => Ok(DoryHintValue::GT(ArkGT::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl Valid for DoryHintValue {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            DoryHintValue::G1(g1) => g1.check(),
            DoryHintValue::G2(g2) => g2.check(),
            DoryHintValue::GT(gt) => gt.check(),
        }
    }
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryHintEntry {
    pub round: u16,
    pub op_type: u8,
    pub index: u16,
    pub value: DoryHintValue,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryHintMap {
    pub num_rounds: u64,
    pub entries: Vec<DoryHintEntry>, // sorted by (round, op_type, index)
}

impl DoryHintMap {
    pub fn from_dory_hint_map(hints: HintMap<BN254>) -> Self {
        let mut entries = Vec::with_capacity(hints.len());
        for (id, result) in hints.iter() {
            let value = if let Some(g1) = result.as_g1() {
                DoryHintValue::G1(*g1)
            } else if let Some(g2) = result.as_g2() {
                DoryHintValue::G2(*g2)
            } else if let Some(gt) = result.as_gt() {
                DoryHintValue::GT(*gt)
            } else {
                continue;
            };
            entries.push(DoryHintEntry {
                round: id.round,
                op_type: id.op_type as u8,
                index: id.index,
                value,
            });
        }

        entries.sort_by_key(|e| (e.round, e.op_type, e.index));

        Self {
            num_rounds: hints.num_rounds as u64,
            entries,
        }
    }

    pub fn to_dory_hint_map(&self) -> HintMap<BN254> {
        let mut hints = HintMap::<BN254>::new(self.num_rounds as usize);
        for e in &self.entries {
            let op_type = match e.op_type {
                0 => OpType::GtExp,
                1 => OpType::G1ScalarMul,
                2 => OpType::G2ScalarMul,
                3 => OpType::GtMul,
                4 => OpType::Pairing,
                5 => OpType::MultiPairing,
                6 => OpType::MsmG1,
                7 => OpType::MsmG2,
                _ => continue, // unknown op type; ignore
            };
            let id = OpId::new(e.round, op_type, e.index);
            match e.value {
                DoryHintValue::G1(g1) => hints.insert_g1(id, g1),
                DoryHintValue::G2(g2) => hints.insert_g2(id, g2),
                DoryHintValue::GT(gt) => hints.insert_gt(id, gt),
            }
        }
        hints
    }
}

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
    pub x_a_mles: Vec<Vec<Fq>>,      // x-coords of A_0, A_1, ..., A_n
    pub y_a_mles: Vec<Vec<Fq>>,      // y-coords of A_0, A_1, ..., A_n
    pub x_t_mles: Vec<Vec<Fq>>,      // x-coords of T_0, T_1, ..., T_{n-1}
    pub y_t_mles: Vec<Vec<Fq>>,      // y-coords of T_0, T_1, ..., T_{n-1}
    pub x_a_next_mles: Vec<Vec<Fq>>, // x-coords of A_{i+1} (shifted by 1)
    pub y_a_next_mles: Vec<Vec<Fq>>, // y-coords of A_{i+1} (shifted by 1)
    pub bits: Vec<bool>,
    ark_result: ArkG1,
}

impl WitnessResult<ArkG1> for JoltG1ScalarMulWitness {
    fn result(&self) -> Option<&ArkG1> {
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
    type GtExpWitness = JoltGtExpWitness;
    type G1ScalarMulWitness = JoltG1ScalarMulWitness;
    type G2ScalarMulWitness = UnimplementedWitness<ArkG2>;
    type GtMulWitness = JoltGtMulWitness;
    type PairingWitness = UnimplementedWitness<ArkGT>;
    type MultiPairingWitness = UnimplementedWitness<ArkGT>;
    type MsmG1Witness = UnimplementedWitness<ArkG1>;
    type MsmG2Witness = UnimplementedWitness<ArkG2>;
}

pub struct JoltWitnessGenerator;

impl WitnessGenerator<JoltWitness, BN254> for JoltWitnessGenerator {
    fn generate_gt_exp(
        base: &<BN254 as PairingCurve>::GT,
        scalar: &<<BN254 as PairingCurve>::G1 as Group>::Scalar,
        result: &<BN254 as PairingCurve>::GT,
    ) -> JoltGtExpWitness {
        let base_fq12 = base.0;
        let scalar_fr = ark_to_jolt(scalar);

        let exp_steps = ExponentiationSteps::new(base_fq12, scalar_fr);

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
            bits: scalar_mul_steps.bits,
            ark_result: *result,
        }
    }

    fn generate_g2_scalar_mul(
        _point: &<BN254 as PairingCurve>::G2,
        _scalar: &<<BN254 as PairingCurve>::G1 as Group>::Scalar,
        _result: &<BN254 as PairingCurve>::G2,
    ) -> UnimplementedWitness<ArkG2> {
        UnimplementedWitness::new("G2 scalar multiplication")
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
        _g1: &<BN254 as PairingCurve>::G1,
        _g2: &<BN254 as PairingCurve>::G2,
        _result: &<BN254 as PairingCurve>::GT,
    ) -> UnimplementedWitness<ArkGT> {
        UnimplementedWitness::new("Pairing")
    }

    fn generate_multi_pairing(
        _g1s: &[<BN254 as PairingCurve>::G1],
        _g2s: &[<BN254 as PairingCurve>::G2],
        _result: &<BN254 as PairingCurve>::GT,
    ) -> UnimplementedWitness<ArkGT> {
        UnimplementedWitness::new("Multi-pairing")
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
    type Hint = DoryHintMap;
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
        let ark_point: Vec<ArkFr> = point
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

        // Convert witnesses to deterministic, canonically-serializable hints.
        let hints = DoryHintMap::from_dory_hint_map(witnesses.to_hints::<BN254>());

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
        let ark_point: Vec<ArkFr> = point
            .iter()
            .rev() // Reverse for dory endianness
            .map(|c| {
                let f_val: Fr = (*c).into();
                jolt_to_ark(&f_val)
            })
            .collect();
        let ark_evaluation = jolt_to_ark(evaluation);

        // Create hint-based verification context
        let ctx = Rc::new(
            TraceContext::<JoltWitness, BN254, JoltWitnessGenerator>::for_hints(
                hint.to_dory_hint_map(),
            ),
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
                let exp_steps = ExponentiationSteps::new(comm_fq12, *coeff);

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
