//! Unified verifier for the recursion SNARK protocol
//!
//! This module provides a high-level verifier that verifies:
//! - Stage 1: Packed GT exp sumcheck
//! - Stage 2: Batched constraint sumchecks (shift + claim reduction + remaining constraints)
//! - Stage 3: Prefix packing reduction to a single dense polynomial opening
//!
//! The verifier returns an opening accumulator for PCS verification.

use ark_bn254::{Fq, Fq12, G1Affine, G2Affine};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::Zero;
use jolt_platform::{end_cycle_tracking, start_cycle_tracking};
use std::io::{Read, Write};

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::{OpeningAccumulator, SumcheckId, VerifierOpeningAccumulator};
use crate::subprotocols::sumcheck::{BatchedSumcheck, SumcheckInstanceProof};
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::Transcript;
use crate::zkvm::proof_serialization::PairingBoundary;
use crate::zkvm::witness::{
    CommittedPolynomial, G1AddTerm, G1ScalarMulTerm, G2AddTerm, G2ScalarMulTerm, GtExpTerm,
    GtMulTerm, RecursionPoly, VirtualPolynomial,
};

use super::constraints::system::{ConstraintType, PolyType};
use super::curve::{Bn254Recursion, RecursionCurve};
use super::g1::addition::{G1AddParams, G1AddVerifier};
use super::g1::indexing::k_g1;
use super::g1::scalar_multiplication::{G1ScalarMulVerifier, ShiftG1ScalarMulVerifier};
use super::g1::types::G1ScalarMulPublicInputs;
use super::g1::wiring::WiringG1Verifier;
use super::g2::addition::{G2AddParams, G2AddVerifier};
use super::g2::indexing::k_g2;
use super::g2::scalar_multiplication::{G2ScalarMulVerifier, ShiftG2ScalarMulVerifier};
use super::g2::types::G2ScalarMulPublicInputs;
use super::g2::wiring::WiringG2Verifier;
use super::gt::base_power::GtExpBasePowVerifier;
use super::gt::exponentiation::{GtExpParams, GtExpVerifier};
use super::gt::indexing::{k_gt, num_gt_mul_constraints_padded};
use super::gt::multiplication::{GtMulParams, GtMulVerifier};
use super::gt::shift::{GtShiftParams, GtShiftVerifier};
use super::gt::stage1_base_openings::GtExpBaseStage1OpeningsVerifier;
use super::gt::stage2_base_openings::GtExpBaseStage2OpeningsVerifier;
use super::gt::stage2_openings::GtExpStage2OpeningsVerifier;
use super::gt::types::GtExpPublicInputs;
use super::gt::wiring::WiringGtVerifier;
use super::prefix_packing::{packed_eval_from_claims, PrefixPackingLayout};
use super::prover::RecursionProof;
use super::WiringPlan;

// Cycle-marker labels must be static strings: the tracer keys markers by the guest string pointer.
const CYCLE_RECURSION_STAGE1: &str = "jolt_recursion_stage1";
const CYCLE_RECURSION_STAGE2: &str = "jolt_recursion_stage2";
const CYCLE_RECURSION_STAGE3: &str = "jolt_recursion_stage3";
const CYCLE_RECURSION_PCS_OPENING: &str = "jolt_recursion_pcs_opening";

struct CycleMarkerGuard(&'static str);
impl CycleMarkerGuard {
    #[inline(always)]
    fn new(label: &'static str) -> Self {
        start_cycle_tracking(label);
        Self(label)
    }
}
impl Drop for CycleMarkerGuard {
    #[inline(always)]
    fn drop(&mut self) {
        end_cycle_tracking(self.0);
    }
}

/// Input required by the verifier
#[derive(Clone, Debug)]
pub struct RecursionVerifierInput {
    /// Constraint types to verify
    pub constraint_types: Vec<ConstraintType>,
    /// Number of variables in the constraint system
    pub num_vars: usize,
    /// Number of constraint variables (x variables) in the matrix
    pub num_constraint_vars: usize,
    /// Number of s-variables (log2 of matrix row domain)
    pub num_s_vars: usize,
    /// Total number of constraints
    pub num_constraints: usize,
    /// Padded number of constraints
    pub num_constraints_padded: usize,
    /// Public inputs for packed GT exp (scalar bits).
    pub gt_exp_public_inputs: Vec<GtExpPublicInputs>,
    /// Boundary GT bases for GTExp instances whose base is an AST input (or a combine-leaf input).
    ///
    /// - Length must match `gt_exp_public_inputs.len()`.
    /// - For non-input bases (bound via wiring), this entry should be `None` so the verifier does
    ///   not need to materialize the base value.
    pub gt_exp_base_inputs: Vec<Option<Fq12>>,
    /// Boundary GT inputs keyed by AST `ValueId` index.
    ///
    /// These are GT-valued `AstOp::Input`s that feed directly into GT ports (e.g. GTMul lhs/rhs)
    /// but are not represented as GTExp base boundaries.
    ///
    /// Each entry is `(value_id, value)`, where `value_id` is the underlying `ValueId.0`.
    pub gt_inputs: Vec<(u32, Fq12)>,
    /// Boundary G1 inputs keyed by AST `ValueId` index.
    ///
    /// These are G1-valued `AstOp::Input`s that feed directly into G1 ports (e.g. G1Add inputs)
    /// and must be treated as verifier-derived boundary constants.
    pub g1_inputs: Vec<(u32, G1Affine)>,
    /// Boundary G2 inputs keyed by AST `ValueId` index.
    ///
    /// These are G2-valued `AstOp::Input`s that feed directly into G2 ports (e.g. G2Add inputs)
    /// and must be treated as verifier-derived boundary constants.
    pub g2_inputs: Vec<(u32, G2Affine)>,
    /// Public inputs for G1 scalar multiplication (scalar per G1ScalarMul constraint)
    pub g1_scalar_mul_public_inputs: Vec<G1ScalarMulPublicInputs>,
    /// Public inputs for G2 scalar multiplication (scalar per G2ScalarMul constraint)
    pub g2_scalar_mul_public_inputs: Vec<G2ScalarMulPublicInputs>,

    /// Verifier-derived wiring plan (explicit edges), used by Stage-2 wiring sumchecks.
    pub wiring: WiringPlan,
    /// Boundary outputs for the external pairing check, bound by wiring/boundary constraints.
    pub pairing_boundary: PairingBoundary,
    /// Stage-8 joint commitment (GT), bound to combine-commitments GT ops by wiring constraints.
    pub joint_commitment: Fq12,
}

impl CanonicalSerialize for RecursionVerifierInput {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.constraint_types
            .serialize_with_mode(&mut writer, compress)?;
        self.num_vars.serialize_with_mode(&mut writer, compress)?;
        self.num_constraint_vars
            .serialize_with_mode(&mut writer, compress)?;
        self.num_s_vars.serialize_with_mode(&mut writer, compress)?;
        self.num_constraints
            .serialize_with_mode(&mut writer, compress)?;
        self.num_constraints_padded
            .serialize_with_mode(&mut writer, compress)?;
        self.gt_exp_public_inputs
            .serialize_with_mode(&mut writer, compress)?;
        self.gt_exp_base_inputs
            .serialize_with_mode(&mut writer, compress)?;
        self.gt_inputs.serialize_with_mode(&mut writer, compress)?;
        self.g1_inputs.serialize_with_mode(&mut writer, compress)?;
        self.g2_inputs.serialize_with_mode(&mut writer, compress)?;
        self.g1_scalar_mul_public_inputs
            .serialize_with_mode(&mut writer, compress)?;
        self.g2_scalar_mul_public_inputs
            .serialize_with_mode(&mut writer, compress)?;
        self.wiring.serialize_with_mode(&mut writer, compress)?;
        self.pairing_boundary
            .serialize_with_mode(&mut writer, compress)?;
        self.joint_commitment
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.constraint_types.serialized_size(compress)
            + self.num_vars.serialized_size(compress)
            + self.num_constraint_vars.serialized_size(compress)
            + self.num_s_vars.serialized_size(compress)
            + self.num_constraints.serialized_size(compress)
            + self.num_constraints_padded.serialized_size(compress)
            + self.gt_exp_public_inputs.serialized_size(compress)
            + self.gt_exp_base_inputs.serialized_size(compress)
            + self.gt_inputs.serialized_size(compress)
            + self.g1_inputs.serialized_size(compress)
            + self.g2_inputs.serialized_size(compress)
            + self.g1_scalar_mul_public_inputs.serialized_size(compress)
            + self.g2_scalar_mul_public_inputs.serialized_size(compress)
            + self.wiring.serialized_size(compress)
            + self.pairing_boundary.serialized_size(compress)
            + self.joint_commitment.serialized_size(compress)
    }
}

impl Valid for RecursionVerifierInput {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for RecursionVerifierInput {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            constraint_types: Vec::deserialize_with_mode(&mut reader, compress, validate)?,
            num_vars: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            num_constraint_vars: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            num_s_vars: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            num_constraints: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            num_constraints_padded: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            gt_exp_public_inputs: Vec::deserialize_with_mode(&mut reader, compress, validate)?,
            gt_exp_base_inputs: Vec::deserialize_with_mode(&mut reader, compress, validate)?,
            gt_inputs: Vec::deserialize_with_mode(&mut reader, compress, validate)?,
            g1_inputs: Vec::deserialize_with_mode(&mut reader, compress, validate)?,
            g2_inputs: Vec::deserialize_with_mode(&mut reader, compress, validate)?,
            g1_scalar_mul_public_inputs: Vec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            g2_scalar_mul_public_inputs: Vec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            wiring: WiringPlan::deserialize_with_mode(&mut reader, compress, validate)?,
            pairing_boundary: PairingBoundary::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            joint_commitment: Fq12::deserialize_with_mode(&mut reader, compress, validate)?,
        })
    }
}

/// Unified verifier for the recursion SNARK
pub struct RecursionVerifier<F: JoltField = Fq> {
    /// Input parameters for verification
    input: RecursionVerifierInput,
    /// Phantom data for the field type
    _phantom: std::marker::PhantomData<F>,
}

impl RecursionVerifier<Fq> {
    /// Create a new recursion verifier
    pub fn new(input: RecursionVerifierInput) -> Self {
        Self {
            input,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Verify the full two-stage recursion proof and PCS opening
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify")]
    pub fn verify<T: Transcript, PCS: CommitmentScheme<Field = Fq>>(
        &self,
        proof: &RecursionProof<Fq, T, PCS>,
        transcript: &mut T,
        matrix_commitment: &PCS::Commitment,
        verifier_setup: &PCS::VerifierSetup,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        if self.input.gt_exp_base_inputs.len() != self.input.gt_exp_public_inputs.len() {
            return Err(format!(
                "RecursionVerifierInput.gt_exp_base_inputs length {} must match gt_exp_public_inputs length {}",
                self.input.gt_exp_base_inputs.len(),
                self.input.gt_exp_public_inputs.len(),
            )
            .into());
        }

        // Bind the Hyrax dense commitment into the transcript.
        //
        // Prover order: commit dense polynomial (Hyrax) → append commitment → run recursion sumchecks.
        {
            let _span = tracing::info_span!("transcript_append_commitment").entered();
            transcript.append_serializable(matrix_commitment);
        }

        // Initialize opening accumulator and populate with opening claims from proof
        let mut accumulator = {
            let _span = tracing::info_span!("accumulator_init").entered();
            let mut acc = VerifierOpeningAccumulator::<Fq>::new(self.input.num_vars);
            for (key, value) in &proof.opening_claims {
                acc.openings.insert(*key, value.clone());
            }
            acc
        };

        let _cycle_stage1 = CycleMarkerGuard::new(CYCLE_RECURSION_STAGE1);
        let _r_stage1_packed = tracing::info_span!("verify_recursion_stage1")
            .in_scope(|| {
                tracing::info!("Verifying Stage 1: Packed GT exp sumcheck");
                self.verify_stage1(&proof.stage1_proof, transcript, &mut accumulator)
            })
            .map_err(|e| format!("Stage 1 failed: {e}"))?;
        drop(_cycle_stage1);

        let _cycle_stage2 = CycleMarkerGuard::new(CYCLE_RECURSION_STAGE2);
        let r_stage2 = tracing::info_span!("verify_recursion_stage2")
            .in_scope(|| {
                tracing::info!("Verifying Stage 2: Batched constraint sumchecks");
                self.verify_stage2(&proof.stage2_proof, transcript, &mut accumulator)
            })
            .map_err(|e| format!("Stage 2 failed: {e}"))?;
        drop(_cycle_stage2);

        // Stage 2 challenges layout:
        // - `r_stage2` ends with `r_x` (length = num_constraint_vars)
        // - the prefix may include family-index variables (e.g. `c_gt`) depending on which
        //   constraint families are present
        //
        // NOTE: Recursion constraint sumchecks are **suffix-aligned** in the batched sumcheck
        // (`round_offset = max_num_rounds - num_rounds`), so shorter points are suffixes of longer
        // ones in the batched challenge order. We therefore interpret `r_x` as the **suffix** of
        // the Stage-2 challenge vector.
        let num_constraint_vars = self.input.num_constraint_vars;
        if r_stage2.len() < num_constraint_vars {
            return Err(format!(
                "Stage 2 returned {} challenges, expected at least {} (num_constraint_vars)",
                r_stage2.len(),
                num_constraint_vars,
            )
            .into());
        }
        // Interpret `r_x` as the suffix of length `num_constraint_vars`.
        let r_x_start = r_stage2.len() - num_constraint_vars;
        let _r_x = &r_stage2[r_x_start..];
        // If the Stage-2 point contains the GT-local `c_gt` suffix (length k_gt), it is the slice
        // immediately before `r_x`. Other family suffixes may exist, but are not interpreted here.
        let k = k_gt(&self.input.constraint_types);
        let _r_c_gt = if r_x_start >= k {
            &r_stage2[r_x_start - k..r_x_start]
        } else {
            &[][..]
        };

        // Debug hook: allow stopping after Stage 2 to isolate failures.
        #[cfg(test)]
        if std::env::var("JOLT_RECURSION_STOP_AFTER_STAGE2").is_ok() {
            return Ok(true);
        }

        let _cycle_stage3 = CycleMarkerGuard::new(CYCLE_RECURSION_STAGE3);
        tracing::info_span!("verify_recursion_stage3")
            .in_scope(|| {
                tracing::info!("Verifying Stage 3: Prefix packing reduction");
                self.verify_stage3_prefix_packing(
                    transcript,
                    &mut accumulator,
                    &r_stage2,
                    proof.stage3_packed_eval,
                )
            })
            .map_err(|e| format!("Stage 3 failed: {e}"))?;
        drop(_cycle_stage3);

        let _cycle_pcs = CycleMarkerGuard::new(CYCLE_RECURSION_PCS_OPENING);
        tracing::info_span!("verify_recursion_pcs_opening")
            .in_scope(|| {
                tracing::info!("Verifying PCS opening proof");
                // Verify opening proof using PCS
                accumulator.verify_single::<T, PCS>(
                    &proof.opening_proof,
                    matrix_commitment.clone(),
                    verifier_setup,
                    transcript,
                )
            })
            .map_err(|e| format!("PCS opening failed: {e}"))?;
        drop(_cycle_pcs);

        Ok(true)
    }

    /// Verify Stage 1: Packed GT exp sumcheck
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage1")]
    fn verify_stage1<T: Transcript>(
        &self,
        proof: &SumcheckInstanceProof<Fq, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
    ) -> Result<Vec<<Fq as JoltField>::Challenge>, Box<dyn std::error::Error>> {
        if self.input.gt_exp_public_inputs.is_empty() {
            return Err("No GtExp constraints to verify in Stage 1".into());
        }

        // Use packed GT exp path
        let params = GtExpParams::from_constraint_types(&self.input.constraint_types);
        let base_verifier = GtExpBaseStage1OpeningsVerifier::new(&self.input.constraint_types);
        let verifier = GtExpVerifier::new(
            params,
            &self.input.constraint_types,
            self.input.gt_exp_public_inputs.clone(),
            transcript,
        );
        let r_stage1 = BatchedSumcheck::verify(
            proof,
            vec![&base_verifier, &verifier],
            accumulator,
            transcript,
        )?;
        Ok(r_stage1)
    }

    /// Verify Stage 2: Batched constraint sumchecks.
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage2")]
    fn verify_stage2<T: Transcript>(
        &self,
        proof: &SumcheckInstanceProof<Fq, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
    ) -> Result<Vec<<Fq as JoltField>::Challenge>, Box<dyn std::error::Error>> {
        let mut verifiers: Vec<Box<dyn SumcheckInstanceVerifier<Fq, T>>> = Vec::new();

        // Count constraints by type and collect per-type sequential indices (matching the extractor).
        let mut num_gt_exp = 0usize;
        let mut num_gt_mul = 0usize;
        let mut num_g1_scalar_mul = 0usize;
        let mut num_g2_scalar_mul = 0usize;
        let mut num_g1_add = 0usize;
        let mut num_g2_add = 0usize;

        for constraint in self.input.constraint_types.iter() {
            match constraint {
                ConstraintType::GtExp => {
                    num_gt_exp += 1;
                }
                ConstraintType::GtMul => {
                    num_gt_mul += 1;
                }
                ConstraintType::G1ScalarMul { base_point: _ } => {
                    num_g1_scalar_mul += 1;
                }
                ConstraintType::G2ScalarMul { base_point: _ } => {
                    num_g2_scalar_mul += 1;
                }
                ConstraintType::G1Add => {
                    num_g1_add += 1;
                }
                ConstraintType::G2Add => {
                    num_g2_add += 1;
                }
            }
        }

        // Packed GT exp auxiliary subprotocols (shift rho + claim reduction).
        if num_gt_exp > 0 {
            // Ordering matters: shift expects the GTExp rho to already exist at the Stage-2 point
            // (emitted by the claim-reduction/openings instance).
            verifiers.push(Box::new(GtExpStage2OpeningsVerifier::new(
                &self.input.constraint_types,
            )));
            verifiers.push(Box::new(GtExpBaseStage2OpeningsVerifier::new(
                &self.input.constraint_types,
            )));

            verifiers.push(Box::new(GtExpBasePowVerifier::new(
                &self.input.constraint_types,
                <Bn254Recursion as RecursionCurve>::g_mle(),
                transcript,
            )));

            let params = GtShiftParams::from_constraint_types(&self.input.constraint_types);
            verifiers.push(Box::new(GtShiftVerifier::new(params)));
        }

        // GT mul
        if num_gt_mul > 0 {
            let num_gt_constraints = num_gt_mul;
            let k_common = k_gt(&self.input.constraint_types);
            let num_gt_constraints_padded =
                num_gt_mul_constraints_padded(&self.input.constraint_types);
            let params = GtMulParams::new(num_gt_constraints, num_gt_constraints_padded, k_common);
            let verifier = GtMulVerifier::new(
                params,
                &self.input.constraint_types,
                <Bn254Recursion as RecursionCurve>::g_mle(),
                transcript,
            );
            verifiers.push(Box::new(verifier));
        }

        // G1 scalar mul
        if num_g1_scalar_mul > 0 {
            let k_common = k_g1(&self.input.constraint_types);
            if self.input.g1_scalar_mul_public_inputs.len() != num_g1_scalar_mul {
                return Err(format!(
                    "RecursionVerifierInput.g1_scalar_mul_public_inputs length {} must match number of G1ScalarMul constraints {}",
                    self.input.g1_scalar_mul_public_inputs.len(),
                    num_g1_scalar_mul,
                )
                .into());
            }
            verifiers.push(Box::new(G1ScalarMulVerifier::new_with_k_common(
                num_g1_scalar_mul,
                k_common,
                self.input.g1_scalar_mul_public_inputs.clone(),
                transcript,
            )));

            // Shift check (no additional openings; reuses scalar-mul cached openings).
            verifiers.push(Box::new(ShiftG1ScalarMulVerifier::new_with_k_common(
                num_g1_scalar_mul,
                k_common,
                transcript,
            )));
        }

        // G2 scalar mul
        if num_g2_scalar_mul > 0 {
            let k_common = k_g2(&self.input.constraint_types);

            if self.input.g2_scalar_mul_public_inputs.len() != num_g2_scalar_mul {
                return Err(format!(
                    "RecursionVerifierInput.g2_scalar_mul_public_inputs length {} must match number of G2ScalarMul constraints {}",
                    self.input.g2_scalar_mul_public_inputs.len(),
                    num_g2_scalar_mul,
                )
                .into());
            }
            verifiers.push(Box::new(G2ScalarMulVerifier::new_with_k_common(
                num_g2_scalar_mul,
                k_common,
                self.input.g2_scalar_mul_public_inputs.clone(),
                transcript,
            )));

            // Shift check (no additional openings; reuses scalar-mul cached openings).
            verifiers.push(Box::new(ShiftG2ScalarMulVerifier::new_with_k_common(
                num_g2_scalar_mul,
                k_common,
                transcript,
            )));
        }

        // G1 add
        if num_g1_add > 0 {
            let params = G1AddParams::new(num_g1_add);
            let verifier = G1AddVerifier::new(params, transcript);
            verifiers.push(Box::new(verifier));
        }

        // G2 add
        if num_g2_add > 0 {
            let params = G2AddParams::new(num_g2_add);
            let verifier = G2AddVerifier::new(params, transcript);
            verifiers.push(Box::new(verifier));
        }

        // Wiring/boundary constraints (AST-driven), appended LAST in Stage 2
        if !self.input.wiring.gt.is_empty() {
            verifiers.push(Box::new(WiringGtVerifier::new(&self.input, transcript)));
        }
        if !self.input.wiring.g1.is_empty() {
            verifiers.push(Box::new(WiringG1Verifier::new(&self.input, transcript)));
        }
        if !self.input.wiring.g2.is_empty() {
            verifiers.push(Box::new(WiringG2Verifier::new(&self.input, transcript)));
        }

        if verifiers.is_empty() {
            return Err("No constraints to verify in Stage 2".into());
        }

        let verifier_refs: Vec<&dyn SumcheckInstanceVerifier<Fq, T>> =
            verifiers.iter().map(|v| &**v).collect();
        let r_stage2 = BatchedSumcheck::verify(proof, verifier_refs, accumulator, transcript)?;
        Ok(r_stage2)
    }

    /// Verify Stage 3: Prefix packing reduction.
    ///
    /// This mirrors `RecursionProver::prove_stage3_prefix_packing`.
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage3_prefix_packing")]
    fn verify_stage3_prefix_packing<T: Transcript>(
        &self,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        r_stage2: &[<Fq as JoltField>::Challenge],
        stage3_packed_eval: Fq,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Derive the public packing layout.
        let layout = PrefixPackingLayout::from_constraint_types(&self.input.constraint_types);
        let max_native_vars = layout.entries.iter().map(|e| e.num_vars).max().unwrap_or(0);
        if r_stage2.len() < max_native_vars {
            return Err(format!(
                "Stage 2 produced r_stage2 of length {}, but prefix packing needs at least {} bits",
                r_stage2.len(),
                max_native_vars
            )
            .into());
        }

        // Low variables: Stage-2 point suffix (suffix-aligned Stage 2), **reversed**.
        //
        // See `RecursionProver::prove_stage3_prefix_packing` for rationale.
        let mut r_native_fq: Vec<Fq> = r_stage2[r_stage2.len() - max_native_vars..]
            .iter()
            .map(|c| (*c).into())
            .collect();
        r_native_fq.reverse();

        // High variables: fresh packing challenges (Fiat–Shamir).
        let pack_len = layout.num_dense_vars.saturating_sub(max_native_vars);
        let r_pack: Vec<Fq> = (0..pack_len)
            .map(|_| transcript.challenge_scalar::<Fq>())
            .collect();

        // Full packed opening point in little-endian (low-to-high) variable order.
        let mut r_full_lsb: Vec<Fq> = Vec::with_capacity(layout.num_dense_vars);
        r_full_lsb.extend_from_slice(&r_native_fq);
        r_full_lsb.extend_from_slice(&r_pack);

        // Compute the expected packed evaluation.
        let expected = packed_eval_from_claims(&layout, &r_full_lsb, |entry| {
            if entry.is_gt {
                let (sumcheck, vp) = match entry.poly_type {
                    PolyType::RhoPrev => (
                        SumcheckId::GtExpClaimReduction,
                        VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                            term: GtExpTerm::Rho,
                        }),
                    ),
                    PolyType::Quotient => (
                        SumcheckId::GtExpClaimReduction,
                        VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                            term: GtExpTerm::Quotient,
                        }),
                    ),
                    PolyType::GtExpBase => (
                        SumcheckId::GtExpBaseClaimReduction,
                        VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                            term: GtExpTerm::Base,
                        }),
                    ),
                    PolyType::GtExpBase2 => (
                        SumcheckId::GtExpBaseClaimReduction,
                        VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                            term: GtExpTerm::Base2,
                        }),
                    ),
                    PolyType::GtExpBase3 => (
                        SumcheckId::GtExpBaseClaimReduction,
                        VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                            term: GtExpTerm::Base3,
                        }),
                    ),
                    PolyType::GtExpBaseSquareQuotient => (
                        SumcheckId::GtExpBaseClaimReduction,
                        VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                            term: GtExpTerm::BaseSquareQuotient,
                        }),
                    ),
                    PolyType::GtExpBaseCubeQuotient => (
                        SumcheckId::GtExpBaseClaimReduction,
                        VirtualPolynomial::Recursion(RecursionPoly::GtExp {
                            term: GtExpTerm::BaseCubeQuotient,
                        }),
                    ),
                    PolyType::MulLhs => (
                        SumcheckId::GtMul,
                        VirtualPolynomial::Recursion(RecursionPoly::GtMul {
                            term: GtMulTerm::Lhs,
                        }),
                    ),
                    PolyType::MulRhs => (
                        SumcheckId::GtMul,
                        VirtualPolynomial::Recursion(RecursionPoly::GtMul {
                            term: GtMulTerm::Rhs,
                        }),
                    ),
                    PolyType::MulResult => (
                        SumcheckId::GtMul,
                        VirtualPolynomial::Recursion(RecursionPoly::GtMul {
                            term: GtMulTerm::Result,
                        }),
                    ),
                    PolyType::MulQuotient => (
                        SumcheckId::GtMul,
                        VirtualPolynomial::Recursion(RecursionPoly::GtMul {
                            term: GtMulTerm::Quotient,
                        }),
                    ),
                    _ => return Fq::zero(),
                };
                accumulator.get_virtual_polynomial_claim(vp, sumcheck)
            } else if entry.is_g1_scalar_mul {
                let vp = match entry.poly_type {
                    PolyType::G1ScalarMulXA => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                            term: G1ScalarMulTerm::XA,
                        })
                    }
                    PolyType::G1ScalarMulYA => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                            term: G1ScalarMulTerm::YA,
                        })
                    }
                    PolyType::G1ScalarMulXT => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                            term: G1ScalarMulTerm::XT,
                        })
                    }
                    PolyType::G1ScalarMulYT => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                            term: G1ScalarMulTerm::YT,
                        })
                    }
                    PolyType::G1ScalarMulXANext => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                            term: G1ScalarMulTerm::XANext,
                        })
                    }
                    PolyType::G1ScalarMulYANext => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                            term: G1ScalarMulTerm::YANext,
                        })
                    }
                    PolyType::G1ScalarMulTIndicator => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                            term: G1ScalarMulTerm::TIndicator,
                        })
                    }
                    PolyType::G1ScalarMulAIndicator => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                            term: G1ScalarMulTerm::AIndicator,
                        })
                    }
                    _ => return Fq::zero(),
                };
                accumulator.get_virtual_polynomial_claim(vp, SumcheckId::G1ScalarMul)
            } else if entry.is_g1_scalar_mul_base {
                let vp = match entry.poly_type {
                    PolyType::G1ScalarMulXP => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                            term: G1ScalarMulTerm::XP,
                        })
                    }
                    PolyType::G1ScalarMulYP => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                            term: G1ScalarMulTerm::YP,
                        })
                    }
                    _ => return Fq::zero(),
                };
                accumulator.get_virtual_polynomial_claim(vp, SumcheckId::G1ScalarMul)
            } else if entry.is_g1_add {
                let vp = match entry.poly_type {
                    PolyType::G1AddXP => VirtualPolynomial::Recursion(RecursionPoly::G1Add {
                        term: G1AddTerm::XP,
                    }),
                    PolyType::G1AddYP => VirtualPolynomial::Recursion(RecursionPoly::G1Add {
                        term: G1AddTerm::YP,
                    }),
                    PolyType::G1AddPIndicator => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1Add {
                            term: G1AddTerm::PIndicator,
                        })
                    }
                    PolyType::G1AddXQ => VirtualPolynomial::Recursion(RecursionPoly::G1Add {
                        term: G1AddTerm::XQ,
                    }),
                    PolyType::G1AddYQ => VirtualPolynomial::Recursion(RecursionPoly::G1Add {
                        term: G1AddTerm::YQ,
                    }),
                    PolyType::G1AddQIndicator => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1Add {
                            term: G1AddTerm::QIndicator,
                        })
                    }
                    PolyType::G1AddXR => VirtualPolynomial::Recursion(RecursionPoly::G1Add {
                        term: G1AddTerm::XR,
                    }),
                    PolyType::G1AddYR => VirtualPolynomial::Recursion(RecursionPoly::G1Add {
                        term: G1AddTerm::YR,
                    }),
                    PolyType::G1AddRIndicator => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1Add {
                            term: G1AddTerm::RIndicator,
                        })
                    }
                    PolyType::G1AddLambda => VirtualPolynomial::Recursion(RecursionPoly::G1Add {
                        term: G1AddTerm::Lambda,
                    }),
                    PolyType::G1AddInvDeltaX => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1Add {
                            term: G1AddTerm::InvDeltaX,
                        })
                    }
                    PolyType::G1AddIsDouble => VirtualPolynomial::Recursion(RecursionPoly::G1Add {
                        term: G1AddTerm::IsDouble,
                    }),
                    PolyType::G1AddIsInverse => {
                        VirtualPolynomial::Recursion(RecursionPoly::G1Add {
                            term: G1AddTerm::IsInverse,
                        })
                    }
                    _ => return Fq::zero(),
                };
                accumulator.get_virtual_polynomial_claim(vp, SumcheckId::G1Add)
            } else if entry.is_g2_scalar_mul {
                let vp = match entry.poly_type {
                    PolyType::G2ScalarMulXAC0 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::XAC0,
                        })
                    }
                    PolyType::G2ScalarMulXAC1 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::XAC1,
                        })
                    }
                    PolyType::G2ScalarMulYAC0 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::YAC0,
                        })
                    }
                    PolyType::G2ScalarMulYAC1 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::YAC1,
                        })
                    }
                    PolyType::G2ScalarMulXTC0 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::XTC0,
                        })
                    }
                    PolyType::G2ScalarMulXTC1 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::XTC1,
                        })
                    }
                    PolyType::G2ScalarMulYTC0 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::YTC0,
                        })
                    }
                    PolyType::G2ScalarMulYTC1 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::YTC1,
                        })
                    }
                    PolyType::G2ScalarMulXANextC0 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::XANextC0,
                        })
                    }
                    PolyType::G2ScalarMulXANextC1 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::XANextC1,
                        })
                    }
                    PolyType::G2ScalarMulYANextC0 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::YANextC0,
                        })
                    }
                    PolyType::G2ScalarMulYANextC1 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::YANextC1,
                        })
                    }
                    PolyType::G2ScalarMulTIndicator => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::TIndicator,
                        })
                    }
                    PolyType::G2ScalarMulAIndicator => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::AIndicator,
                        })
                    }
                    PolyType::G2ScalarMulXPC0 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::XPC0,
                        })
                    }
                    PolyType::G2ScalarMulXPC1 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::XPC1,
                        })
                    }
                    PolyType::G2ScalarMulYPC0 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::YPC0,
                        })
                    }
                    PolyType::G2ScalarMulYPC1 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::YPC1,
                        })
                    }
                    _ => return Fq::zero(),
                };
                accumulator.get_virtual_polynomial_claim(vp, SumcheckId::G2ScalarMul)
            } else if entry.is_g2_scalar_mul_base {
                let vp = match entry.poly_type {
                    PolyType::G2ScalarMulXPC0 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::XPC0,
                        })
                    }
                    PolyType::G2ScalarMulXPC1 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::XPC1,
                        })
                    }
                    PolyType::G2ScalarMulYPC0 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::YPC0,
                        })
                    }
                    PolyType::G2ScalarMulYPC1 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
                            term: G2ScalarMulTerm::YPC1,
                        })
                    }
                    _ => return Fq::zero(),
                };
                accumulator.get_virtual_polynomial_claim(vp, SumcheckId::G2ScalarMul)
            } else if entry.is_g2_add {
                let vp = match entry.poly_type {
                    PolyType::G2AddXPC0 => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::XPC0,
                    }),
                    PolyType::G2AddXPC1 => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::XPC1,
                    }),
                    PolyType::G2AddYPC0 => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::YPC0,
                    }),
                    PolyType::G2AddYPC1 => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::YPC1,
                    }),
                    PolyType::G2AddPIndicator => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                            term: G2AddTerm::PIndicator,
                        })
                    }
                    PolyType::G2AddXQC0 => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::XQC0,
                    }),
                    PolyType::G2AddXQC1 => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::XQC1,
                    }),
                    PolyType::G2AddYQC0 => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::YQC0,
                    }),
                    PolyType::G2AddYQC1 => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::YQC1,
                    }),
                    PolyType::G2AddQIndicator => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                            term: G2AddTerm::QIndicator,
                        })
                    }
                    PolyType::G2AddXRC0 => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::XRC0,
                    }),
                    PolyType::G2AddXRC1 => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::XRC1,
                    }),
                    PolyType::G2AddYRC0 => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::YRC0,
                    }),
                    PolyType::G2AddYRC1 => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::YRC1,
                    }),
                    PolyType::G2AddRIndicator => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                            term: G2AddTerm::RIndicator,
                        })
                    }
                    PolyType::G2AddLambdaC0 => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::LambdaC0,
                    }),
                    PolyType::G2AddLambdaC1 => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::LambdaC1,
                    }),
                    PolyType::G2AddInvDeltaXC0 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                            term: G2AddTerm::InvDeltaXC0,
                        })
                    }
                    PolyType::G2AddInvDeltaXC1 => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                            term: G2AddTerm::InvDeltaXC1,
                        })
                    }
                    PolyType::G2AddIsDouble => VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                        term: G2AddTerm::IsDouble,
                    }),
                    PolyType::G2AddIsInverse => {
                        VirtualPolynomial::Recursion(RecursionPoly::G2Add {
                            term: G2AddTerm::IsInverse,
                        })
                    }
                    _ => return Fq::zero(),
                };
                accumulator.get_virtual_polynomial_claim(vp, SumcheckId::G2Add)
            } else {
                panic!("unexpected prefix-packing entry without a family tag: {entry:?}")
            }
        });

        if expected != stage3_packed_eval {
            return Err(format!(
                "Stage 3 packed eval mismatch: expected {expected} but proof has {stage3_packed_eval}"
            )
            .into());
        }

        // Append as a Fiat–Shamir message (matches prover transcript pattern).
        transcript.append_scalar(&stage3_packed_eval);

        // Register the committed dense opening (this appends the claimed eval again).
        let opening_point: Vec<<Fq as JoltField>::Challenge> =
            r_full_lsb.into_iter().rev().map(|f| f.into()).collect();
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::DoryDenseMatrix,
            SumcheckId::RecursionPacked,
            opening_point,
        );

        Ok(())
    }
}
