//! Unified verifier for the recursion SNARK protocol
//!
//! This module provides a high-level verifier that verifies:
//! - Stage 1: Packed GT exp sumcheck
//! - Stage 2: Batched constraint sumchecks (shift + claim reduction + remaining constraints)
//! - Stage 3: Prefix packing reduction to a single dense polynomial opening
//!
//! The verifier returns an opening accumulator for PCS verification.

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::{OpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    },
    transcripts::Transcript,
    zkvm::witness::{CommittedPolynomial, G2AddTerm, VirtualPolynomial},
};
use ark_bn254::{Fq, Fq12};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::Zero;
use std::io::{Read, Write};

use crate::subprotocols::sumcheck::{BatchedSumcheck, SumcheckInstanceProof};
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};
use crate::zkvm::recursion::prefix_packing::{packed_eval_from_claims, PrefixPackingLayout};

#[cfg(feature = "experimental-pairing-recursion")]
use super::pairing::{
    multi_miller_loop::{
        MultiMillerLoopParams, MultiMillerLoopVerifier, MultiMillerLoopVerifierSpec,
    },
    shift::{ShiftMultiMillerLoopParams, ShiftMultiMillerLoopVerifier},
};
use super::{
    constraints::system::{ConstraintType, PolyType},
    curve::{Bn254Recursion, RecursionCurve},
    g1::{
        addition::{G1AddParams, G1AddVerifier, G1AddVerifierSpec},
        fused_addition::{FusedG1AddParams, FusedG1AddVerifier},
        fused_scalar_multiplication::{FusedG1ScalarMulVerifier, FusedShiftG1ScalarMulVerifier},
        fused_wiring::FusedWiringG1Verifier,
        indexing::k_g1,
        scalar_multiplication::{
            G1ScalarMulParams, G1ScalarMulPublicInputs, G1ScalarMulVerifier,
            G1ScalarMulVerifierSpec,
        },
        shift::{
            g1_shift_params, g2_shift_params, ShiftG1ScalarMulVerifier, ShiftG2ScalarMulVerifier,
        },
        WiringG1Verifier,
    },
    g2::{
        addition::{G2AddParams, G2AddVerifier, G2AddVerifierSpec},
        fused_addition::{FusedG2AddParams, FusedG2AddVerifier},
        fused_scalar_multiplication::{FusedG2ScalarMulVerifier, FusedShiftG2ScalarMulVerifier},
        fused_wiring::FusedWiringG2Verifier,
        indexing::k_g2,
        scalar_multiplication::{
            G2ScalarMulParams, G2ScalarMulPublicInputs, G2ScalarMulVerifier,
            G2ScalarMulVerifierSpec,
        },
        WiringG2Verifier,
    },
    gt::{
        claim_reduction::{GtExpClaimReductionParams, GtExpClaimReductionVerifier},
        exponentiation::{GtExpParams, GtExpPublicInputs, GtExpVerifier},
        fused_exponentiation::{FusedGtExpParams, FusedGtExpVerifier},
        fused_multiplication::{FusedGtMulParams, FusedGtMulVerifier},
        fused_shift::{FusedGtShiftParams, FusedGtShiftVerifier},
        fused_stage2_openings::FusedGtExpStage2OpeningsVerifier,
        fused_wiring::FusedWiringGtVerifier,
        indexing::{k_gt, num_gt_mul_constraints_padded},
        multiplication::{GtMulParams, GtMulVerifier, GtMulVerifierSpec},
        shift::{GtShiftParams, GtShiftVerifier},
        wiring_binding::GtWiringBindingVerifier,
        WiringGtVerifier,
    },
    prover::RecursionProof,
    virtualization::extract_virtual_claims_from_accumulator,
    WiringPlan,
};

use crate::zkvm::proof_serialization::PairingBoundary;
use jolt_platform::{end_cycle_tracking, start_cycle_tracking};

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
    /// Whether to use the fused-GT end-to-end recursion protocol.
    ///
    /// This affects Stage 1/2/3 instance selection and packing layout; it must match the
    /// recursion artifact/proof being verified (in-guest there is no reliable env-var channel).
    pub enable_gt_fused_end_to_end: bool,
    /// Whether to use the fused-G1-scalar-mul end-to-end recursion protocol.
    ///
    /// This affects Stage 2 instance selection and packing layout; it must match the
    /// recursion artifact/proof being verified.
    pub enable_g1_scalar_mul_fused_end_to_end: bool,
    /// Whether to use the **fully fused G1 wiring** end-to-end recursion protocol.
    ///
    /// When enabled, Stage 2 uses:
    /// - fused G1ScalarMul (aligned over `k_g1`),
    /// - fused G1Add, and
    /// - fused G1 wiring backend,
    /// and Stage 3 prefix packing uses fused G1Add rows (family-local `k_add`).
    pub enable_g1_fused_wiring_end_to_end: bool,
    /// Whether to use the fused-G2-scalar-mul end-to-end recursion protocol.
    ///
    /// This affects Stage 2 instance selection and packing layout; it must match the
    /// recursion artifact/proof being verified.
    pub enable_g2_scalar_mul_fused_end_to_end: bool,
    /// Whether to use the **fully fused G2 wiring** end-to-end recursion protocol.
    ///
    /// When enabled, Stage 2 uses:
    /// - fused G2ScalarMul (aligned over `k_g2`),
    /// - fused G2Add, and
    /// - fused G2 wiring backend.
    pub enable_g2_fused_wiring_end_to_end: bool,
    /// Number of variables in the constraint system
    pub num_vars: usize,
    /// Number of constraint variables (x variables) in the matrix
    pub num_constraint_vars: usize,
    /// Number of s-variables for virtualization
    pub num_s_vars: usize,
    /// Total number of constraints
    pub num_constraints: usize,
    /// Padded number of constraints
    pub num_constraints_padded: usize,
    /// Public inputs for packed GT exp (base Fq12 and scalar bits)
    pub gt_exp_public_inputs: Vec<GtExpPublicInputs>,
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
        self.enable_gt_fused_end_to_end
            .serialize_with_mode(&mut writer, compress)?;
        self.enable_g1_scalar_mul_fused_end_to_end
            .serialize_with_mode(&mut writer, compress)?;
        self.enable_g1_fused_wiring_end_to_end
            .serialize_with_mode(&mut writer, compress)?;
        self.enable_g2_scalar_mul_fused_end_to_end
            .serialize_with_mode(&mut writer, compress)?;
        self.enable_g2_fused_wiring_end_to_end
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
            + self.enable_gt_fused_end_to_end.serialized_size(compress)
            + self
                .enable_g1_scalar_mul_fused_end_to_end
                .serialized_size(compress)
            + self
                .enable_g1_fused_wiring_end_to_end
                .serialized_size(compress)
            + self
                .enable_g2_scalar_mul_fused_end_to_end
                .serialized_size(compress)
            + self
                .enable_g2_fused_wiring_end_to_end
                .serialized_size(compress)
            + self.num_vars.serialized_size(compress)
            + self.num_constraint_vars.serialized_size(compress)
            + self.num_s_vars.serialized_size(compress)
            + self.num_constraints.serialized_size(compress)
            + self.num_constraints_padded.serialized_size(compress)
            + self.gt_exp_public_inputs.serialized_size(compress)
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
            enable_gt_fused_end_to_end: bool::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            enable_g1_scalar_mul_fused_end_to_end: bool::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            enable_g1_fused_wiring_end_to_end: bool::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            enable_g2_scalar_mul_fused_end_to_end: bool::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            enable_g2_fused_wiring_end_to_end: bool::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            num_vars: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            num_constraint_vars: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            num_s_vars: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            num_constraints: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            num_constraints_padded: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            gt_exp_public_inputs: Vec::deserialize_with_mode(&mut reader, compress, validate)?,
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
            pairing_boundary:
                crate::zkvm::proof_serialization::PairingBoundary::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?,
            joint_commitment: Fq12::deserialize_with_mode(&mut reader, compress, validate)?,
        })
    }
}

impl GuestSerialize for RecursionVerifierInput {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.constraint_types.guest_serialize(w)?;
        self.enable_gt_fused_end_to_end.guest_serialize(w)?;
        self.enable_g1_scalar_mul_fused_end_to_end
            .guest_serialize(w)?;
        self.enable_g1_fused_wiring_end_to_end.guest_serialize(w)?;
        self.enable_g2_scalar_mul_fused_end_to_end
            .guest_serialize(w)?;
        self.enable_g2_fused_wiring_end_to_end.guest_serialize(w)?;
        self.num_vars.guest_serialize(w)?;
        self.num_constraint_vars.guest_serialize(w)?;
        self.num_s_vars.guest_serialize(w)?;
        self.num_constraints.guest_serialize(w)?;
        self.num_constraints_padded.guest_serialize(w)?;
        self.gt_exp_public_inputs.guest_serialize(w)?;
        self.g1_scalar_mul_public_inputs.guest_serialize(w)?;
        self.g2_scalar_mul_public_inputs.guest_serialize(w)?;
        self.wiring.guest_serialize(w)?;
        self.pairing_boundary.guest_serialize(w)?;
        self.joint_commitment.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for RecursionVerifierInput {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            constraint_types: Vec::guest_deserialize(r)?,
            enable_gt_fused_end_to_end: bool::guest_deserialize(r)?,
            enable_g1_scalar_mul_fused_end_to_end: bool::guest_deserialize(r)?,
            enable_g1_fused_wiring_end_to_end: bool::guest_deserialize(r)?,
            enable_g2_scalar_mul_fused_end_to_end: bool::guest_deserialize(r)?,
            enable_g2_fused_wiring_end_to_end: bool::guest_deserialize(r)?,
            num_vars: usize::guest_deserialize(r)?,
            num_constraint_vars: usize::guest_deserialize(r)?,
            num_s_vars: usize::guest_deserialize(r)?,
            num_constraints: usize::guest_deserialize(r)?,
            num_constraints_padded: usize::guest_deserialize(r)?,
            gt_exp_public_inputs: Vec::guest_deserialize(r)?,
            g1_scalar_mul_public_inputs: Vec::guest_deserialize(r)?,
            g2_scalar_mul_public_inputs: Vec::guest_deserialize(r)?,
            wiring: WiringPlan::guest_deserialize(r)?,
            pairing_boundary: crate::zkvm::proof_serialization::PairingBoundary::guest_deserialize(
                r,
            )?,
            joint_commitment: Fq12::guest_deserialize(r)?,
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
        // - legacy: r_stage2 = r_x (length = num_constraint_vars)
        // - fused:  r_stage2 includes extra index variables `r_c` (length >= num_constraint_vars + k)
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
        // immediately before `r_x`. Other fused-family suffixes may exist, but are not interpreted here.
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

        let enable_gt_fused_end_to_end = self.input.enable_gt_fused_end_to_end;

        let r_stage1 = if enable_gt_fused_end_to_end {
            let params = FusedGtExpParams::from_constraint_types(&self.input.constraint_types);
            let verifier = FusedGtExpVerifier::new(
                params,
                &self.input.constraint_types,
                self.input.gt_exp_public_inputs.clone(),
                transcript,
            );
            BatchedSumcheck::verify(proof, vec![&verifier], accumulator, transcript)?
        } else {
            let params = GtExpParams::new();
            let verifier =
                GtExpVerifier::new(params, self.input.gt_exp_public_inputs.clone(), transcript);
            BatchedSumcheck::verify(proof, vec![&verifier], accumulator, transcript)?
        };
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
        let env_flag_default = |name: &str, default: bool| -> bool {
            std::env::var(name)
                .ok()
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(default)
        };
        let enable_shift_rho = env_flag_default("JOLT_RECURSION_ENABLE_SHIFT_RHO", true);
        let enable_shift_g1_scalar_mul =
            env_flag_default("JOLT_RECURSION_ENABLE_SHIFT_G1_SCALAR_MUL", true);
        let enable_shift_g2_scalar_mul =
            env_flag_default("JOLT_RECURSION_ENABLE_SHIFT_G2_SCALAR_MUL", true);
        let enable_claim_reduction = env_flag_default("JOLT_RECURSION_ENABLE_PGX_REDUCTION", true);
        let enable_gt_fused_end_to_end = self.input.enable_gt_fused_end_to_end;
        let enable_g1_fused_wiring_end_to_end = self.input.enable_g1_fused_wiring_end_to_end;
        let enable_g1_scalar_mul_fused_end_to_end =
            self.input.enable_g1_scalar_mul_fused_end_to_end || enable_g1_fused_wiring_end_to_end;
        let enable_g1_add_fused_end_to_end = enable_g1_fused_wiring_end_to_end;
        let enable_g2_fused_wiring_end_to_end = self.input.enable_g2_fused_wiring_end_to_end;
        let enable_g2_scalar_mul_fused_end_to_end =
            self.input.enable_g2_scalar_mul_fused_end_to_end || enable_g2_fused_wiring_end_to_end;
        let enable_g2_add_fused_end_to_end = enable_g2_fused_wiring_end_to_end;
        let enable_wiring = env_flag_default("JOLT_RECURSION_ENABLE_WIRING", true);
        let enable_wiring_gt = env_flag_default("JOLT_RECURSION_ENABLE_WIRING_GT", true);
        let enable_wiring_g1 = env_flag_default("JOLT_RECURSION_ENABLE_WIRING_G1", true);
        let enable_wiring_g2 = env_flag_default("JOLT_RECURSION_ENABLE_WIRING_G2", true);
        #[cfg(feature = "experimental-pairing-recursion")]
        let enable_shift_multi_miller_loop =
            env_flag_default("JOLT_RECURSION_ENABLE_SHIFT_MULTI_MILLER_LOOP", true);

        let mut verifiers: Vec<Box<dyn SumcheckInstanceVerifier<Fq, T>>> = Vec::new();

        // Count constraints by type and collect per-type sequential indices (matching the extractor).
        let mut num_gt_exp = 0usize;
        let mut num_gt_mul = 0usize;
        let mut num_g1_scalar_mul = 0usize;
        let mut num_g2_scalar_mul = 0usize;
        let mut num_g1_add = 0usize;
        let mut num_g2_add = 0usize;
        #[cfg(feature = "experimental-pairing-recursion")]
        let mut num_multi_miller_loop = 0usize;

        let mut gt_mul_indices = Vec::new();
        let mut g1_scalar_mul_base_points = Vec::new();
        let mut g1_scalar_mul_indices = Vec::new();
        let mut g2_scalar_mul_base_points = Vec::new();
        let mut g2_scalar_mul_indices = Vec::new();
        let mut g1_add_indices = Vec::new();
        let mut g2_add_indices = Vec::new();
        #[cfg(feature = "experimental-pairing-recursion")]
        let mut multi_miller_loop_indices = Vec::new();

        for constraint in self.input.constraint_types.iter() {
            match constraint {
                ConstraintType::GtExp => {
                    num_gt_exp += 1;
                }
                ConstraintType::GtMul => {
                    gt_mul_indices.push(num_gt_mul);
                    num_gt_mul += 1;
                }
                ConstraintType::G1ScalarMul { base_point } => {
                    g1_scalar_mul_base_points.push(*base_point);
                    g1_scalar_mul_indices.push(num_g1_scalar_mul);
                    num_g1_scalar_mul += 1;
                }
                ConstraintType::G2ScalarMul { base_point } => {
                    g2_scalar_mul_base_points.push(*base_point);
                    g2_scalar_mul_indices.push(num_g2_scalar_mul);
                    num_g2_scalar_mul += 1;
                }
                ConstraintType::G1Add => {
                    g1_add_indices.push(num_g1_add);
                    num_g1_add += 1;
                }
                ConstraintType::G2Add => {
                    g2_add_indices.push(num_g2_add);
                    num_g2_add += 1;
                }
                #[cfg(feature = "experimental-pairing-recursion")]
                ConstraintType::MultiMillerLoop => {
                    multi_miller_loop_indices.push(num_multi_miller_loop);
                    num_multi_miller_loop += 1;
                }
            }
        }

        // Packed GT exp auxiliary subprotocols (shift rho + claim reduction).
        if num_gt_exp > 0 {
            let claim_indices: Vec<usize> = (0..num_gt_exp).collect();
            if enable_gt_fused_end_to_end {
                // Ordering matters: fused shift expects fused GTExp rho to already exist at the
                // Stage-2 point (emitted by the fused claim-reduction/openings instance).
                if enable_claim_reduction {
                    verifiers.push(Box::new(FusedGtExpStage2OpeningsVerifier::new(
                        &self.input.constraint_types,
                    )));
                }

                if enable_shift_rho {
                    let params =
                        FusedGtShiftParams::from_constraint_types(&self.input.constraint_types);
                    verifiers.push(Box::new(FusedGtShiftVerifier::new(params)));
                }
            } else {
                if enable_shift_rho {
                    let shift_verifier = GtShiftVerifier::<Fq>::new(
                        GtShiftParams::new(num_gt_exp),
                        claim_indices.clone(),
                        transcript,
                    );
                    verifiers.push(Box::new(shift_verifier));
                }

                if enable_claim_reduction {
                    let reduction_verifier = GtExpClaimReductionVerifier::<Fq>::new(
                        GtExpClaimReductionParams::new(2 * num_gt_exp),
                        claim_indices,
                        transcript,
                    );
                    verifiers.push(Box::new(reduction_verifier));
                }
            }
        }

        // GT mul
        if num_gt_mul > 0 {
            if enable_gt_fused_end_to_end {
                let num_gt_constraints = num_gt_mul;
                let k_common = k_gt(&self.input.constraint_types);
                let num_gt_constraints_padded =
                    num_gt_mul_constraints_padded(&self.input.constraint_types);
                let params =
                    FusedGtMulParams::new(num_gt_constraints, num_gt_constraints_padded, k_common);
                let verifier = FusedGtMulVerifier::new(
                    params,
                    &self.input.constraint_types,
                    <Bn254Recursion as RecursionCurve>::g_mle(),
                    transcript,
                );
                verifiers.push(Box::new(verifier));
            } else {
                let params = GtMulParams::new(num_gt_mul);
                let spec = GtMulVerifierSpec::<Bn254Recursion>::new(params);
                let verifier =
                    GtMulVerifier::<Bn254Recursion>::from_spec(spec, gt_mul_indices, transcript);
                verifiers.push(Box::new(verifier));
            }
        }

        if num_g1_scalar_mul > 0 {
            if enable_g1_scalar_mul_fused_end_to_end {
                let k_common = if enable_g1_fused_wiring_end_to_end {
                    k_g1(&self.input.constraint_types)
                } else {
                    super::g1::indexing::k_smul(&self.input.constraint_types)
                };
                debug_assert_eq!(
                    self.input.g1_scalar_mul_public_inputs.len(),
                    num_g1_scalar_mul,
                    "RecursionVerifierInput.g1_scalar_mul_public_inputs must match number of G1ScalarMul constraints"
                );
                verifiers.push(Box::new(FusedG1ScalarMulVerifier::new_with_k_common(
                    num_g1_scalar_mul,
                    k_common,
                    self.input.g1_scalar_mul_public_inputs.clone(),
                    g1_scalar_mul_base_points,
                    transcript,
                )));

                // Fused shift check (no additional openings; reuses scalar-mul cached openings).
                if enable_shift_g1_scalar_mul {
                    verifiers.push(Box::new(FusedShiftG1ScalarMulVerifier::new_with_k_common(
                        num_g1_scalar_mul,
                        k_common,
                        transcript,
                    )));
                }
            } else {
                // G1 scalar mul shift: link A and A_next (x/y)
                if enable_shift_g1_scalar_mul {
                    let mut pairs: Vec<(VirtualPolynomial, VirtualPolynomial)> =
                        Vec::with_capacity(num_g1_scalar_mul * 2);
                    for i in 0..num_g1_scalar_mul {
                        pairs.push((
                            VirtualPolynomial::g1_scalar_mul_xa(i),
                            VirtualPolynomial::g1_scalar_mul_xa_next(i),
                        ));
                        pairs.push((
                            VirtualPolynomial::g1_scalar_mul_ya(i),
                            VirtualPolynomial::g1_scalar_mul_ya_next(i),
                        ));
                    }
                    verifiers.push(Box::new(ShiftG1ScalarMulVerifier::<Fq>::new(
                        g1_shift_params(pairs.len()),
                        pairs,
                        transcript,
                    )));
                }

                // G1 scalar mul
                let params = G1ScalarMulParams::new(num_g1_scalar_mul);
                debug_assert_eq!(
                    self.input.g1_scalar_mul_public_inputs.len(),
                    num_g1_scalar_mul,
                    "RecursionVerifierInput.g1_scalar_mul_public_inputs must match number of G1ScalarMul constraints"
                );
                let spec = G1ScalarMulVerifierSpec::new(
                    params,
                    self.input.g1_scalar_mul_public_inputs.clone(),
                    g1_scalar_mul_base_points,
                );
                let verifier =
                    G1ScalarMulVerifier::from_spec(spec, g1_scalar_mul_indices, transcript);
                verifiers.push(Box::new(verifier));
            }
        }

        // G2 scalar mul
        if num_g2_scalar_mul > 0 {
            if enable_g2_scalar_mul_fused_end_to_end {
                let k_common = if enable_g2_fused_wiring_end_to_end {
                    k_g2(&self.input.constraint_types)
                } else {
                    super::g2::indexing::k_smul(&self.input.constraint_types)
                };

                debug_assert_eq!(
                    self.input.g2_scalar_mul_public_inputs.len(),
                    num_g2_scalar_mul,
                    "RecursionVerifierInput.g2_scalar_mul_public_inputs must match number of G2ScalarMul constraints"
                );
                verifiers.push(Box::new(FusedG2ScalarMulVerifier::new_with_k_common(
                    num_g2_scalar_mul,
                    k_common,
                    self.input.g2_scalar_mul_public_inputs.clone(),
                    g2_scalar_mul_base_points,
                    transcript,
                )));

                // Fused shift check (no additional openings; reuses scalar-mul cached openings).
                if enable_shift_g2_scalar_mul {
                    verifiers.push(Box::new(FusedShiftG2ScalarMulVerifier::new_with_k_common(
                        num_g2_scalar_mul,
                        k_common,
                        transcript,
                    )));
                }
            } else {
                // G2 scalar mul shift: link A and A_next (x/y, c0/c1)
                if enable_shift_g2_scalar_mul {
                    let mut pairs: Vec<(VirtualPolynomial, VirtualPolynomial)> =
                        Vec::with_capacity(num_g2_scalar_mul * 4);
                    for i in 0..num_g2_scalar_mul {
                        pairs.push((
                            VirtualPolynomial::g2_scalar_mul_xa_c0(i),
                            VirtualPolynomial::g2_scalar_mul_xa_next_c0(i),
                        ));
                        pairs.push((
                            VirtualPolynomial::g2_scalar_mul_xa_c1(i),
                            VirtualPolynomial::g2_scalar_mul_xa_next_c1(i),
                        ));
                        pairs.push((
                            VirtualPolynomial::g2_scalar_mul_ya_c0(i),
                            VirtualPolynomial::g2_scalar_mul_ya_next_c0(i),
                        ));
                        pairs.push((
                            VirtualPolynomial::g2_scalar_mul_ya_c1(i),
                            VirtualPolynomial::g2_scalar_mul_ya_next_c1(i),
                        ));
                    }
                    verifiers.push(Box::new(ShiftG2ScalarMulVerifier::<Fq>::new(
                        g2_shift_params(pairs.len()),
                        pairs,
                        transcript,
                    )));
                }

                let params = G2ScalarMulParams::new(num_g2_scalar_mul);
                debug_assert_eq!(
                    self.input.g2_scalar_mul_public_inputs.len(),
                    num_g2_scalar_mul,
                    "RecursionVerifierInput.g2_scalar_mul_public_inputs must match number of G2ScalarMul constraints"
                );
                let spec = G2ScalarMulVerifierSpec::new(
                    params,
                    self.input.g2_scalar_mul_public_inputs.clone(),
                    g2_scalar_mul_base_points,
                );
                let verifier =
                    G2ScalarMulVerifier::from_spec(spec, g2_scalar_mul_indices, transcript);
                verifiers.push(Box::new(verifier));
            }
        }

        // G1 add
        if num_g1_add > 0 {
            if enable_g1_add_fused_end_to_end {
                let params = FusedG1AddParams::new(num_g1_add);
                let verifier = FusedG1AddVerifier::new(params, transcript);
                verifiers.push(Box::new(verifier));
            } else {
                let params = G1AddParams::new(num_g1_add);
                let spec = G1AddVerifierSpec::new(params);
                let verifier = G1AddVerifier::from_spec(spec, g1_add_indices, transcript);
                verifiers.push(Box::new(verifier));
            }
        }

        // G2 add
        if num_g2_add > 0 {
            if enable_g2_add_fused_end_to_end {
                let params = FusedG2AddParams::new(num_g2_add);
                let verifier = FusedG2AddVerifier::new(params, transcript);
                verifiers.push(Box::new(verifier));
            } else {
                let params = G2AddParams::new(num_g2_add);
                let spec = G2AddVerifierSpec::new(params);
                let verifier = G2AddVerifier::from_spec(spec, g2_add_indices, transcript);
                verifiers.push(Box::new(verifier));
            }
        }

        // Multi-Miller loop shift: link f/t and *_next columns (experimental)
        #[cfg(feature = "experimental-pairing-recursion")]
        if enable_shift_multi_miller_loop && num_multi_miller_loop > 0 {
            let mut pairs: Vec<(VirtualPolynomial, VirtualPolynomial)> =
                Vec::with_capacity(num_multi_miller_loop * 5);
            for i in 0..num_multi_miller_loop {
                pairs.push((
                    VirtualPolynomial::multi_miller_loop_f(i),
                    VirtualPolynomial::multi_miller_loop_f_next(i),
                ));
                pairs.push((
                    VirtualPolynomial::multi_miller_loop_t_x_c0(i),
                    VirtualPolynomial::multi_miller_loop_t_x_c0_next(i),
                ));
                pairs.push((
                    VirtualPolynomial::multi_miller_loop_t_x_c1(i),
                    VirtualPolynomial::multi_miller_loop_t_x_c1_next(i),
                ));
                pairs.push((
                    VirtualPolynomial::multi_miller_loop_t_y_c0(i),
                    VirtualPolynomial::multi_miller_loop_t_y_c0_next(i),
                ));
                pairs.push((
                    VirtualPolynomial::multi_miller_loop_t_y_c1(i),
                    VirtualPolynomial::multi_miller_loop_t_y_c1_next(i),
                ));
            }

            verifiers.push(Box::new(ShiftMultiMillerLoopVerifier::<Fq>::new(
                ShiftMultiMillerLoopParams::new(pairs.len()),
                pairs,
                transcript,
            )));
        }

        // Multi-Miller loop (experimental)
        #[cfg(feature = "experimental-pairing-recursion")]
        if num_multi_miller_loop > 0 {
            let params = MultiMillerLoopParams::new(num_multi_miller_loop);
            let spec = MultiMillerLoopVerifierSpec::new(params);
            let verifier =
                MultiMillerLoopVerifier::from_spec(spec, multi_miller_loop_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // Wiring/boundary constraints (AST-driven), appended LAST in Stage 2
        if enable_wiring {
            if enable_wiring_gt && !self.input.wiring.gt.is_empty() {
                if enable_gt_fused_end_to_end {
                    // Fully fused GT wiring backend (no aux sums / no legacy binding check).
                    verifiers.push(Box::new(FusedWiringGtVerifier::new(
                        &self.input,
                        transcript,
                    )));
                } else {
                    // Legacy backend (aux sums + binding safety net).
                    let wiring_gt = WiringGtVerifier::new(&self.input, transcript);
                    // Legacy safety net: bind prover-emitted wiring sums to the actual per-edge
                    // wiring expression.
                    verifiers.push(Box::new(GtWiringBindingVerifier::new(wiring_gt.binding())));
                    verifiers.push(Box::new(wiring_gt));
                }
            }
            if enable_wiring_g1 && !self.input.wiring.g1.is_empty() {
                if enable_g1_fused_wiring_end_to_end {
                    verifiers.push(Box::new(FusedWiringG1Verifier::new(
                        &self.input,
                        transcript,
                    )));
                } else {
                    verifiers.push(Box::new(WiringG1Verifier::new(&self.input, transcript)));
                }
            }
            if enable_wiring_g2 && !self.input.wiring.g2.is_empty() {
                if enable_g2_fused_wiring_end_to_end {
                    verifiers.push(Box::new(FusedWiringG2Verifier::new(
                        &self.input,
                        transcript,
                    )));
                } else {
                    verifiers.push(Box::new(WiringG2Verifier::new(&self.input, transcript)));
                }
            }
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
        let enable_gt_fused_end_to_end = self.input.enable_gt_fused_end_to_end;
        let enable_g1_fused_wiring_end_to_end = self.input.enable_g1_fused_wiring_end_to_end;
        let enable_g1_scalar_mul_fused_end_to_end =
            self.input.enable_g1_scalar_mul_fused_end_to_end || enable_g1_fused_wiring_end_to_end;
        let enable_g1_add_fused_end_to_end = enable_g1_fused_wiring_end_to_end;
        let enable_g2_fused_wiring_end_to_end = self.input.enable_g2_fused_wiring_end_to_end;
        let enable_g2_scalar_mul_fused_end_to_end =
            self.input.enable_g2_scalar_mul_fused_end_to_end || enable_g2_fused_wiring_end_to_end;
        let enable_g2_add_fused_end_to_end = enable_g2_fused_wiring_end_to_end;
        let layout = if enable_gt_fused_end_to_end
            || enable_g1_scalar_mul_fused_end_to_end
            || enable_g1_add_fused_end_to_end
            || enable_g2_scalar_mul_fused_end_to_end
            || enable_g2_add_fused_end_to_end
        {
            PrefixPackingLayout::from_constraint_types_fused(
                &self.input.constraint_types,
                enable_gt_fused_end_to_end,
                enable_g1_scalar_mul_fused_end_to_end,
                enable_g1_add_fused_end_to_end,
                enable_g2_scalar_mul_fused_end_to_end,
                enable_g2_add_fused_end_to_end,
            )
        } else {
            PrefixPackingLayout::from_constraint_types(&self.input.constraint_types)
        };
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

        // Extract Stage 2 virtual claims in the standard [constraint-major, poly-type-minor] layout.
        let virtual_claims = extract_virtual_claims_from_accumulator(
            accumulator,
            &self.input.constraint_types,
            &self.input.gt_exp_public_inputs,
            enable_gt_fused_end_to_end,
            enable_g1_scalar_mul_fused_end_to_end,
            enable_g1_add_fused_end_to_end,
            enable_g2_scalar_mul_fused_end_to_end,
            enable_g2_add_fused_end_to_end,
        );
        let num_poly_types = PolyType::NUM_TYPES;

        // Compute the expected packed evaluation.
        let expected = packed_eval_from_claims(&layout, &r_full_lsb, |entry| {
            if enable_gt_fused_end_to_end && entry.is_gt_fused {
                let (sumcheck, vp) = match entry.poly_type {
                    PolyType::RhoPrev => (
                        SumcheckId::GtExpClaimReduction,
                        VirtualPolynomial::gt_exp_rho_fused(),
                    ),
                    PolyType::Quotient => (
                        SumcheckId::GtExpClaimReduction,
                        VirtualPolynomial::gt_exp_quotient_fused(),
                    ),
                    PolyType::MulLhs => (SumcheckId::GtMul, VirtualPolynomial::gt_mul_lhs_fused()),
                    PolyType::MulRhs => (SumcheckId::GtMul, VirtualPolynomial::gt_mul_rhs_fused()),
                    PolyType::MulResult => {
                        (SumcheckId::GtMul, VirtualPolynomial::gt_mul_result_fused())
                    }
                    PolyType::MulQuotient => (
                        SumcheckId::GtMul,
                        VirtualPolynomial::gt_mul_quotient_fused(),
                    ),
                    _ => return Fq::zero(),
                };
                let (_, claim) = accumulator.get_virtual_polynomial_opening(vp, sumcheck);
                claim
            } else if enable_g1_scalar_mul_fused_end_to_end && entry.is_g1_scalar_mul_fused {
                let vp = match entry.poly_type {
                    PolyType::G1ScalarMulXA => VirtualPolynomial::g1_scalar_mul_xa_fused(),
                    PolyType::G1ScalarMulYA => VirtualPolynomial::g1_scalar_mul_ya_fused(),
                    PolyType::G1ScalarMulXT => VirtualPolynomial::g1_scalar_mul_xt_fused(),
                    PolyType::G1ScalarMulYT => VirtualPolynomial::g1_scalar_mul_yt_fused(),
                    PolyType::G1ScalarMulXANext => VirtualPolynomial::g1_scalar_mul_xa_next_fused(),
                    PolyType::G1ScalarMulYANext => VirtualPolynomial::g1_scalar_mul_ya_next_fused(),
                    PolyType::G1ScalarMulTIndicator => {
                        VirtualPolynomial::g1_scalar_mul_t_indicator_fused()
                    }
                    PolyType::G1ScalarMulAIndicator => {
                        VirtualPolynomial::g1_scalar_mul_a_indicator_fused()
                    }
                    _ => return Fq::zero(),
                };
                let (_, claim) =
                    accumulator.get_virtual_polynomial_opening(vp, SumcheckId::G1ScalarMul);
                claim
            } else if enable_g1_add_fused_end_to_end && entry.is_g1_add_fused {
                let vp = match entry.poly_type {
                    PolyType::G1AddXP => VirtualPolynomial::g1_add_xp_fused(),
                    PolyType::G1AddYP => VirtualPolynomial::g1_add_yp_fused(),
                    PolyType::G1AddPIndicator => VirtualPolynomial::g1_add_p_indicator_fused(),
                    PolyType::G1AddXQ => VirtualPolynomial::g1_add_xq_fused(),
                    PolyType::G1AddYQ => VirtualPolynomial::g1_add_yq_fused(),
                    PolyType::G1AddQIndicator => VirtualPolynomial::g1_add_q_indicator_fused(),
                    PolyType::G1AddXR => VirtualPolynomial::g1_add_xr_fused(),
                    PolyType::G1AddYR => VirtualPolynomial::g1_add_yr_fused(),
                    PolyType::G1AddRIndicator => VirtualPolynomial::g1_add_r_indicator_fused(),
                    PolyType::G1AddLambda => {
                        VirtualPolynomial::g1_add_fused(crate::zkvm::witness::G1AddTerm::Lambda)
                    }
                    PolyType::G1AddInvDeltaX => {
                        VirtualPolynomial::g1_add_fused(crate::zkvm::witness::G1AddTerm::InvDeltaX)
                    }
                    PolyType::G1AddIsDouble => {
                        VirtualPolynomial::g1_add_fused(crate::zkvm::witness::G1AddTerm::IsDouble)
                    }
                    PolyType::G1AddIsInverse => {
                        VirtualPolynomial::g1_add_fused(crate::zkvm::witness::G1AddTerm::IsInverse)
                    }
                    _ => return Fq::zero(),
                };
                let (_, claim) = accumulator.get_virtual_polynomial_opening(vp, SumcheckId::G1Add);
                claim
            } else if enable_g2_scalar_mul_fused_end_to_end && entry.is_g2_scalar_mul_fused {
                let vp = match entry.poly_type {
                    PolyType::G2ScalarMulXAC0 => VirtualPolynomial::g2_scalar_mul_xa_c0_fused(),
                    PolyType::G2ScalarMulXAC1 => VirtualPolynomial::g2_scalar_mul_xa_c1_fused(),
                    PolyType::G2ScalarMulYAC0 => VirtualPolynomial::g2_scalar_mul_ya_c0_fused(),
                    PolyType::G2ScalarMulYAC1 => VirtualPolynomial::g2_scalar_mul_ya_c1_fused(),
                    PolyType::G2ScalarMulXTC0 => VirtualPolynomial::g2_scalar_mul_xt_c0_fused(),
                    PolyType::G2ScalarMulXTC1 => VirtualPolynomial::g2_scalar_mul_xt_c1_fused(),
                    PolyType::G2ScalarMulYTC0 => VirtualPolynomial::g2_scalar_mul_yt_c0_fused(),
                    PolyType::G2ScalarMulYTC1 => VirtualPolynomial::g2_scalar_mul_yt_c1_fused(),
                    PolyType::G2ScalarMulXANextC0 => {
                        VirtualPolynomial::g2_scalar_mul_xa_next_c0_fused()
                    }
                    PolyType::G2ScalarMulXANextC1 => {
                        VirtualPolynomial::g2_scalar_mul_xa_next_c1_fused()
                    }
                    PolyType::G2ScalarMulYANextC0 => {
                        VirtualPolynomial::g2_scalar_mul_ya_next_c0_fused()
                    }
                    PolyType::G2ScalarMulYANextC1 => {
                        VirtualPolynomial::g2_scalar_mul_ya_next_c1_fused()
                    }
                    PolyType::G2ScalarMulTIndicator => {
                        VirtualPolynomial::g2_scalar_mul_t_indicator_fused()
                    }
                    PolyType::G2ScalarMulAIndicator => {
                        VirtualPolynomial::g2_scalar_mul_a_indicator_fused()
                    }
                    _ => return Fq::zero(),
                };
                let (_, claim) =
                    accumulator.get_virtual_polynomial_opening(vp, SumcheckId::G2ScalarMul);
                claim
            } else if enable_g2_add_fused_end_to_end && entry.is_g2_add_fused {
                let vp = match entry.poly_type {
                    PolyType::G2AddXPC0 => VirtualPolynomial::g2_add_fused(G2AddTerm::XPC0),
                    PolyType::G2AddXPC1 => VirtualPolynomial::g2_add_fused(G2AddTerm::XPC1),
                    PolyType::G2AddYPC0 => VirtualPolynomial::g2_add_fused(G2AddTerm::YPC0),
                    PolyType::G2AddYPC1 => VirtualPolynomial::g2_add_fused(G2AddTerm::YPC1),
                    PolyType::G2AddPIndicator => {
                        VirtualPolynomial::g2_add_fused(G2AddTerm::PIndicator)
                    }
                    PolyType::G2AddXQC0 => VirtualPolynomial::g2_add_fused(G2AddTerm::XQC0),
                    PolyType::G2AddXQC1 => VirtualPolynomial::g2_add_fused(G2AddTerm::XQC1),
                    PolyType::G2AddYQC0 => VirtualPolynomial::g2_add_fused(G2AddTerm::YQC0),
                    PolyType::G2AddYQC1 => VirtualPolynomial::g2_add_fused(G2AddTerm::YQC1),
                    PolyType::G2AddQIndicator => {
                        VirtualPolynomial::g2_add_fused(G2AddTerm::QIndicator)
                    }
                    PolyType::G2AddXRC0 => VirtualPolynomial::g2_add_fused(G2AddTerm::XRC0),
                    PolyType::G2AddXRC1 => VirtualPolynomial::g2_add_fused(G2AddTerm::XRC1),
                    PolyType::G2AddYRC0 => VirtualPolynomial::g2_add_fused(G2AddTerm::YRC0),
                    PolyType::G2AddYRC1 => VirtualPolynomial::g2_add_fused(G2AddTerm::YRC1),
                    PolyType::G2AddRIndicator => {
                        VirtualPolynomial::g2_add_fused(G2AddTerm::RIndicator)
                    }
                    PolyType::G2AddLambdaC0 => VirtualPolynomial::g2_add_fused(G2AddTerm::LambdaC0),
                    PolyType::G2AddLambdaC1 => VirtualPolynomial::g2_add_fused(G2AddTerm::LambdaC1),
                    PolyType::G2AddInvDeltaXC0 => {
                        VirtualPolynomial::g2_add_fused(G2AddTerm::InvDeltaXC0)
                    }
                    PolyType::G2AddInvDeltaXC1 => {
                        VirtualPolynomial::g2_add_fused(G2AddTerm::InvDeltaXC1)
                    }
                    PolyType::G2AddIsDouble => VirtualPolynomial::g2_add_fused(G2AddTerm::IsDouble),
                    PolyType::G2AddIsInverse => {
                        VirtualPolynomial::g2_add_fused(G2AddTerm::IsInverse)
                    }
                    _ => return Fq::zero(),
                };
                let (_, claim) = accumulator.get_virtual_polynomial_opening(vp, SumcheckId::G2Add);
                claim
            } else {
                let claim_idx = entry.constraint_idx * num_poly_types + (entry.poly_type as usize);
                virtual_claims
                    .get(claim_idx)
                    .copied()
                    .unwrap_or_else(Fq::zero)
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
