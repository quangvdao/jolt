//! High-level API for generating and verifying recursion proofs for base Jolt proofs.
//!
//! This is a *wrapper layer* around:
//! - the base verifier stages (1–7) in `crate::zkvm::verifier::JoltVerifier`
//! - the recursion SNARK prover/verifier in `crate::zkvm::recursion::{prover, verifier}`
//!
//! The goal is to keep the base Jolt proof format recursion-free, while still allowing a
//! standalone recursion artifact to be produced and verified (including inside a guest).

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use ark_ec::AffineRepr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::primitives::arithmetic::PairingCurve;

use crate::transcripts::Transcript;
use crate::zkvm::config::ProgramMode;
use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};
use crate::zkvm::proof_serialization::{JoltProof, NonInputBaseHints, PairingBoundary};
use crate::zkvm::verifier::{JoltVerifier, JoltVerifierPreprocessing};
use crate::zkvm::witness::all_committed_polynomials;

use crate::poly::commitment::commitment_scheme::{CommitmentScheme, RecursionExt};
use crate::poly::commitment::dory::{ArkG1, ArkG2, ArkGT, DoryContext, DoryGlobals, BN254};
use crate::poly::rlc_utils::compute_rlc_coefficients;
use crate::zkvm::recursion::prover::{DoryOpeningSnapshot, RecursionInput, RecursionProver};
use crate::zkvm::recursion::verifier::RecursionVerifier;
use crate::zkvm::recursion::MAX_RECURSION_DENSE_NUM_VARS;
use crate::zkvm::witness::CommittedPolynomial;
use jolt_platform::{end_cycle_tracking, start_cycle_tracking};

use ark_bn254::{Fq, Fq12, Fr};

type DoryPCS = crate::poly::commitment::dory::DoryCommitmentScheme;
type HyraxPCS = crate::zkvm::recursion::prover::HyraxPCS;

/// Standalone recursion artifact for a base Jolt proof.
///
/// This is separate from the base `JoltProof`. The verifier reconstructs transcript state by
/// re-verifying base stages 1–7, then verifies this recursion SNARK proof.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct RecursionArtifact<FS: Transcript> {
    /// Whether this recursion artifact uses the fused-GT end-to-end recursion protocol.
    ///
    /// This must be carried explicitly because in-guest verification does not have a reliable
    /// environment-variable channel.
    pub enable_gt_fused_end_to_end: bool,
    /// Whether this recursion artifact uses the fused-G1-scalar-mul end-to-end recursion protocol.
    ///
    /// This must be carried explicitly because in-guest verification does not have a reliable
    /// environment-variable channel.
    pub enable_g1_scalar_mul_fused_end_to_end: bool,
    /// Whether this recursion artifact uses the **fully fused G1 wiring** end-to-end recursion protocol.
    ///
    /// This must be carried explicitly because in-guest verification does not have a reliable
    /// environment-variable channel.
    pub enable_g1_fused_wiring_end_to_end: bool,
    /// Hint for Stage 8 combine_commitments offloading (the combined GT element).
    ///
    /// This is **required** by the verifier: if absent, verification rejects.
    pub stage8_combine_hint: Option<Fq12>,
    /// Boundary outputs for the external pairing check (treated as a hint; guest recomputes).
    pub pairing_boundary: PairingBoundary,
    /// Minimal hints for Dory instance-plan derivation (guest recomputes without trusting).
    pub non_input_base_hints: NonInputBaseHints,
    /// The recursion SNARK proof itself (Hyrax + sumchecks).
    pub proof: crate::zkvm::recursion::prover::RecursionProof<Fq, FS, HyraxPCS>,
}

impl<FS: Transcript> GuestSerialize for RecursionArtifact<FS> {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.enable_gt_fused_end_to_end.guest_serialize(w)?;
        self.enable_g1_scalar_mul_fused_end_to_end
            .guest_serialize(w)?;
        self.enable_g1_fused_wiring_end_to_end
            .guest_serialize(w)?;
        self.stage8_combine_hint.guest_serialize(w)?;
        self.pairing_boundary.guest_serialize(w)?;
        self.non_input_base_hints.guest_serialize(w)?;
        self.proof.guest_serialize(w)?;
        Ok(())
    }
}

impl<FS: Transcript> GuestDeserialize for RecursionArtifact<FS> {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            enable_gt_fused_end_to_end: bool::guest_deserialize(r)?,
            enable_g1_scalar_mul_fused_end_to_end: bool::guest_deserialize(r)?,
            enable_g1_fused_wiring_end_to_end: bool::guest_deserialize(r)?,
            stage8_combine_hint: Option::<Fq12>::guest_deserialize(r)?,
            pairing_boundary: PairingBoundary::guest_deserialize(r)?,
            non_input_base_hints: NonInputBaseHints::guest_deserialize(r)?,
            proof: crate::zkvm::recursion::prover::RecursionProof::<Fq, FS, HyraxPCS>::guest_deserialize(
                r,
            )?,
        })
    }
}

/// Generate a recursion artifact for a base Jolt proof.
///
/// This runs base verification Stages 1–7 to reconstruct transcript state, then proves the
/// recursion SNARK.
pub fn prove_recursion<FS: Transcript>(
    preprocessing: &JoltVerifierPreprocessing<Fr, DoryPCS>,
    program_io: tracer::JoltDevice,
    trusted_advice_commitment: Option<<DoryPCS as CommitmentScheme>::Commitment>,
    base_proof: &JoltProof<Fr, DoryPCS, FS>,
) -> Result<RecursionArtifact<FS>> {
    type F = Fr;

    // We need an owned proof to build a `JoltVerifier`. A plain clone is much cheaper than the
    // previous serialize→deserialize roundtrip and is equivalent for verification.
    let base_proof = base_proof.clone();
    let mut v = JoltVerifier::new(
        preprocessing,
        base_proof,
        program_io,
        trusted_advice_commitment,
        None,
    )
    .map_err(|e| anyhow!("failed to construct base verifier: {e:?}"))?;

    v.initialize_transcript_preamble()?;

    // Base stages 1..7 (no Stage 8 PCS verification).
    v.verify_stages_1_to_7()?;

    let enable_gt_fused_end_to_end = std::env::var("JOLT_RECURSION_ENABLE_GT_FUSED_END_TO_END")
        .ok()
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(false);
    let enable_g1_fused_wiring_end_to_end =
        std::env::var("JOLT_RECURSION_ENABLE_G1_FUSED_WIRING_END_TO_END")
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false);
    let enable_g1_scalar_mul_fused_end_to_end =
        std::env::var("JOLT_RECURSION_ENABLE_G1_SCALAR_MUL_FUSED_END_TO_END")
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
            || enable_g1_fused_wiring_end_to_end;

    // Ensure Dory globals match the proof layout before any Stage 8 replay / witness generation.
    let _dory_globals_guard = if v.proof.program_mode == ProgramMode::Committed {
        let committed = v.preprocessing.program.as_committed()?;
        DoryGlobals::initialize_main_context_with_num_columns(
            1 << v.one_hot_params.log_k_chunk,
            v.proof.trace_length.next_power_of_two(),
            committed.bytecode_num_columns,
            Some(v.proof.dory_layout),
        )
    } else {
        DoryGlobals::initialize_context(
            1 << v.one_hot_params.log_k_chunk,
            v.proof.trace_length.next_power_of_two(),
            DoryContext::Main,
            Some(v.proof.dory_layout),
        )
    };

    let (dory_snap, pre_opening_proof_transcript, gamma_powers, joint_claim) =
        v.build_stage8_recursion_prep()?;

    let stage8_snapshot = DoryOpeningSnapshot::<F, FS> {
        pre_opening_proof_transcript: pre_opening_proof_transcript.clone(),
        opening_point: dory_snap.opening_point.r.clone(),
        polynomial_claims: dory_snap.polynomial_claims.clone(),
        gamma_powers: gamma_powers.clone(),
        joint_claim,
    };

    // Advance the main transcript to the post-Stage8 state expected by recursion SNARK.
    v.transcript = pre_opening_proof_transcript;
    <DoryPCS as RecursionExt<F>>::replay_opening_proof_transcript(
        &v.proof.joint_opening_proof,
        &mut v.transcript,
    )
    .map_err(|e| anyhow!("Stage 8 PCS FS replay failed: {e:?}"))?;

    // Build the commitments map needed by Stage 8 offloading + instance-plan derivation.
    let mut commitments_map =
        HashMap::<CommittedPolynomial, <DoryPCS as CommitmentScheme>::Commitment>::new();
    let all_polys = all_committed_polynomials(&v.one_hot_params);
    if all_polys.len() != v.proof.commitments.len() {
        return Err(anyhow!(
            "commitment vector length mismatch: expected {}, got {}",
            all_polys.len(),
            v.proof.commitments.len()
        ));
    }
    for (poly, commitment) in all_polys.into_iter().zip(v.proof.commitments.iter()) {
        commitments_map.insert(poly, commitment.clone());
    }

    // Advice commitments (only if they were folded into the Stage 8 batch).
    let needs_trusted_advice = stage8_snapshot
        .polynomial_claims
        .iter()
        .any(|(p, _)| *p == CommittedPolynomial::TrustedAdvice);
    let needs_untrusted_advice = stage8_snapshot
        .polynomial_claims
        .iter()
        .any(|(p, _)| *p == CommittedPolynomial::UntrustedAdvice);
    if needs_trusted_advice {
        if let Some(ref commitment) = v.trusted_advice_commitment {
            commitments_map.insert(CommittedPolynomial::TrustedAdvice, commitment.clone());
        }
    }
    if needs_untrusted_advice {
        if let Some(ref commitment) = v.proof.untrusted_advice_commitment {
            commitments_map.insert(CommittedPolynomial::UntrustedAdvice, commitment.clone());
        }
    }

    // Program commitments (BytecodeChunk and ProgramImageInit) in committed mode.
    if v.proof.program_mode == ProgramMode::Committed {
        let committed = v.preprocessing.program.as_committed()?;
        for (idx, commitment) in committed.bytecode_commitments.iter().enumerate() {
            commitments_map
                .entry(CommittedPolynomial::BytecodeChunk(idx))
                .or_insert_with(|| commitment.clone());
        }
        if stage8_snapshot
            .polynomial_claims
            .iter()
            .any(|(p, _)| *p == CommittedPolynomial::ProgramImageInit)
        {
            commitments_map.insert(
                CommittedPolynomial::ProgramImageInit,
                committed.program_image_commitment.clone(),
            );
        }
    }

    // Hyrax setup for recursion proving (dense commitment/opening).
    //
    // Hyrax uses the same type for prover+verifier setup (`PedersenGenerators`), and the base
    // preprocessing already includes a cached recursion-sized setup.
    let hyrax_prover_setup = &preprocessing.hyrax_recursion_setup;

    let (
        recursion_snark_proof,
        _constraint_metadata,
        pairing_boundary,
        stage8_combine_hint,
        non_input_base_hints,
    ) = RecursionProver::<Fq>::prove::<F, DoryPCS, FS>(
        &mut v.transcript,
        hyrax_prover_setup,
        RecursionInput {
            joint_opening_proof: &v.proof.joint_opening_proof,
            stage8_snapshot,
            verifier_setup: &v.preprocessing.generators,
            commitments: &commitments_map,
        },
    )
    .map_err(|e| anyhow!("failed to generate recursion proof: {e:?}"))?;

    if stage8_combine_hint.is_none() {
        return Err(anyhow!(
            "missing required Stage 8 combine hint (stage8_combine_hint)"
        ));
    }

    Ok(RecursionArtifact {
        enable_gt_fused_end_to_end,
        enable_g1_scalar_mul_fused_end_to_end,
        enable_g1_fused_wiring_end_to_end,
        stage8_combine_hint,
        pairing_boundary,
        non_input_base_hints,
        proof: recursion_snark_proof,
    })
}

/// Verify a recursion artifact for a base Jolt proof.
pub fn verify_recursion<FS: Transcript>(
    preprocessing: &JoltVerifierPreprocessing<Fr, DoryPCS>,
    program_io: tracer::JoltDevice,
    trusted_advice_commitment: Option<<DoryPCS as CommitmentScheme>::Commitment>,
    base_proof: &JoltProof<Fr, DoryPCS, FS>,
    recursion: &RecursionArtifact<FS>,
) -> Result<()> {
    type F = Fr;

    start_cycle_tracking("verify_recursion_total");
    // We need an owned proof to build a `JoltVerifier`. A plain clone is much cheaper than the
    // previous serialize→deserialize roundtrip and is equivalent for verification.
    let base_proof = base_proof.clone();
    let mut v = JoltVerifier::new(
        preprocessing,
        base_proof,
        program_io,
        trusted_advice_commitment,
        None,
    )
    .map_err(|e| anyhow!("failed to construct base verifier: {e:?}"))?;

    v.initialize_transcript_preamble()?;

    start_cycle_tracking("verify_recursion_base_stages_1_to_7_total");
    v.verify_stages_1_to_7()?;
    end_cycle_tracking("verify_recursion_base_stages_1_to_7_total");

    // Ensure Dory globals match the proof layout before any Stage 8 replay / AST derivation.
    let _dory_globals_guard = if v.proof.program_mode == ProgramMode::Committed {
        let committed = v.preprocessing.program.as_committed()?;
        DoryGlobals::initialize_main_context_with_num_columns(
            1 << v.one_hot_params.log_k_chunk,
            v.proof.trace_length.next_power_of_two(),
            committed.bytecode_num_columns,
            Some(v.proof.dory_layout),
        )
    } else {
        DoryGlobals::initialize_context(
            1 << v.one_hot_params.log_k_chunk,
            v.proof.trace_length.next_power_of_two(),
            DoryContext::Main,
            Some(v.proof.dory_layout),
        )
    };

    start_cycle_tracking("verify_recursion_stage8_prep_total");
    let (dory_snap, pre_opening_proof_transcript, gamma_powers, joint_claim) =
        v.build_stage8_recursion_prep()?;

    // Build commitments map (must match Stage 8 native verifier path).
    let mut commitments_map =
        HashMap::<CommittedPolynomial, <DoryPCS as CommitmentScheme>::Commitment>::new();
    for (poly, commitment) in all_committed_polynomials(&v.one_hot_params)
        .into_iter()
        .zip(v.proof.commitments.iter())
    {
        commitments_map.insert(poly, commitment.clone());
    }
    if let Some(ref commitment) = v.trusted_advice_commitment {
        if dory_snap
            .polynomial_claims
            .iter()
            .any(|(p, _)| *p == CommittedPolynomial::TrustedAdvice)
        {
            commitments_map.insert(CommittedPolynomial::TrustedAdvice, commitment.clone());
        }
    }
    if let Some(ref commitment) = v.proof.untrusted_advice_commitment {
        if dory_snap
            .polynomial_claims
            .iter()
            .any(|(p, _)| *p == CommittedPolynomial::UntrustedAdvice)
        {
            commitments_map.insert(CommittedPolynomial::UntrustedAdvice, commitment.clone());
        }
    }
    if v.proof.program_mode == ProgramMode::Committed {
        let committed = v.preprocessing.program.as_committed()?;
        for (idx, commitment) in committed.bytecode_commitments.iter().enumerate() {
            commitments_map
                .entry(CommittedPolynomial::BytecodeChunk(idx))
                .or_insert_with(|| commitment.clone());
        }
        if dory_snap
            .polynomial_claims
            .iter()
            .any(|(p, _)| *p == CommittedPolynomial::ProgramImageInit)
        {
            commitments_map.insert(
                CommittedPolynomial::ProgramImageInit,
                committed.program_image_commitment.clone(),
            );
        }
    }

    // Deterministic combine plan (must match the prover's ordering).
    let rlc_map = compute_rlc_coefficients(&gamma_powers, dory_snap.polynomial_claims.clone());
    let (combine_coeffs, combine_commitments): (
        Vec<F>,
        Vec<<DoryPCS as CommitmentScheme>::Commitment>,
    ) = rlc_map
        .into_iter()
        .map(|(poly, coeff)| {
            (
                coeff,
                commitments_map
                    .get(&poly)
                    .expect("missing commitment for polynomial in batch")
                    .clone(),
            )
        })
        .unzip();

    // Get Stage 8 combine hint
    let hint_fq12 = recursion
        .stage8_combine_hint
        .as_ref()
        .ok_or_else(|| anyhow!("missing required Stage 8 combine hint (stage8_combine_hint)"))?;
    let joint_commitment: <DoryPCS as CommitmentScheme>::Commitment =
        <DoryPCS as RecursionExt<F>>::combine_with_hint_fq12(hint_fq12);

    // Build symbolic AST on a transcript clone at the pre-Stage8-proof state.
    let mut ast_transcript = pre_opening_proof_transcript.clone();
    let ast = <DoryPCS as RecursionExt<F>>::build_symbolic_ast(
        &v.proof.joint_opening_proof,
        &v.preprocessing.generators,
        &mut ast_transcript,
        &dory_snap.opening_point.r,
        &joint_claim,
        &joint_commitment,
    )
    .map_err(|e| anyhow!("Stage 8 symbolic AST build failed: {e:?}"))?;

    // Advance the main transcript to the post-Stage8 state expected by recursion SNARK.
    v.transcript = pre_opening_proof_transcript;
    <DoryPCS as RecursionExt<F>>::replay_opening_proof_transcript(
        &v.proof.joint_opening_proof,
        &mut v.transcript,
    )
    .map_err(|e| anyhow!("Stage 8 PCS FS replay failed: {e:?}"))?;

    // Derive recursion verifier input using hint-based plan derivation (NO expensive group ops).
    //
    // Rationale: the recursion SNARK now enforces AST-driven wiring/boundary constraints (Stage 2),
    // including binding non-input bases/points and the pairing boundary. Re-evaluating the Dory AST
    // (pairings / scalar muls / GT exp) inside the verifier is therefore unnecessary and extremely
    // expensive in the zkVM / cycle-tracking path.
    let combine_coeffs_fr: Vec<Fr> = combine_coeffs;
    let joint_commitment_dory: ArkGT = joint_commitment;
    let combine_commitments_dory: Vec<ArkGT> = combine_commitments;
    let plan = crate::poly::commitment::dory::derive_plan_with_hints(
        &ast,
        &v.proof.joint_opening_proof,
        &v.preprocessing.generators,
        joint_commitment_dory,
        &combine_commitments_dory,
        &combine_coeffs_fr,
        &recursion.non_input_base_hints,
        recursion.pairing_boundary.clone(),
        *hint_fq12,
    )
    .map_err(|e| anyhow!("AST->recursion-plan derivation (with hints) failed: {e:?}"))?;
    end_cycle_tracking("verify_recursion_stage8_prep_total");

    if plan.dense_num_vars > MAX_RECURSION_DENSE_NUM_VARS {
        return Err(anyhow!(
            "dense_num_vars {} exceeds max {}",
            plan.dense_num_vars,
            MAX_RECURSION_DENSE_NUM_VARS
        ));
    }

    // Verify recursion SNARK (use cached Hyrax setup from preprocessing).
    let hyrax_verifier_setup = &preprocessing.hyrax_recursion_setup;

    let mut verifier_input = plan.verifier_input;
    verifier_input.enable_gt_fused_end_to_end = recursion.enable_gt_fused_end_to_end;
    verifier_input.enable_g1_scalar_mul_fused_end_to_end =
        recursion.enable_g1_scalar_mul_fused_end_to_end;
    verifier_input.enable_g1_fused_wiring_end_to_end =
        recursion.enable_g1_fused_wiring_end_to_end;
    let recursion_verifier = RecursionVerifier::<Fq>::new(verifier_input);
    start_cycle_tracking("verify_recursion_snark_verify_total");
    let ok = recursion_verifier
        .verify::<FS, HyraxPCS>(
            &recursion.proof,
            &mut v.transcript,
            &recursion.proof.dense_commitment,
            hyrax_verifier_setup,
        )
        .map_err(|e| anyhow!("Recursion verification failed: {e:?}"))?;
    end_cycle_tracking("verify_recursion_snark_verify_total");
    if !ok {
        return Err(anyhow!("Recursion proof verification failed"));
    }

    // External pairing check using the boundary value that is bound by wiring constraints.
    let got = &recursion.pairing_boundary;
    start_cycle_tracking("jolt_external_pairing_check");
    let lhs = {
        let g1s = [
            ArkG1(got.p1_g1.into_group()),
            ArkG1(got.p2_g1.into_group()),
            ArkG1(got.p3_g1.into_group()),
        ];
        let g2s = [
            ArkG2(got.p1_g2.into_group()),
            ArkG2(got.p2_g2.into_group()),
            ArkG2(got.p3_g2.into_group()),
        ];
        BN254::multi_pair(&g1s, &g2s)
    };
    end_cycle_tracking("jolt_external_pairing_check");
    if lhs.0 != got.rhs {
        return Err(anyhow!("external pairing check failed"));
    }

    end_cycle_tracking("verify_recursion_total");
    Ok(())
}
