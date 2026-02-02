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
use ark_bn254::{Fq, Fq12, Fr};
use ark_ec::AffineRepr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::primitives::arithmetic::PairingCurve;
use jolt_platform::{end_cycle_tracking, start_cycle_tracking};

use crate::poly::commitment::commitment_scheme::{CommitmentScheme, RecursionExt};
use crate::poly::commitment::dory::instance_plan::derive_plan_with_hints;
use crate::poly::commitment::dory::{ArkG1, ArkG2, DoryCommitmentScheme, DoryContext, DoryGlobals, BN254};
use crate::poly::rlc_utils::compute_rlc_coefficients;
use crate::transcripts::Transcript;
use crate::zkvm::config::ProgramMode;
use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};
use crate::zkvm::proof_serialization::{JoltProof, PairingBoundary};
use crate::zkvm::verifier::{JoltVerifier, JoltVerifierPreprocessing};
use crate::zkvm::witness::{all_committed_polynomials, CommittedPolynomial};

use super::prover::{
    DoryOpeningSnapshot, HyraxPCS, RecursionInput, RecursionProof, RecursionProver,
};
use super::verifier::RecursionVerifier;
use super::MAX_RECURSION_DENSE_NUM_VARS;

type DoryPCS = DoryCommitmentScheme;

/// Standalone recursion artifact for a base Jolt proof.
///
/// This is separate from the base `JoltProof`. The verifier reconstructs transcript state by
/// re-verifying base stages 1–7, then verifies this recursion SNARK proof.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct RecursionArtifact<FS: Transcript> {
    /// Hint for Stage 8 combine_commitments offloading (the combined GT element).
    ///
    /// This is **required** by the verifier: if absent, verification rejects.
    pub stage8_combine_hint: Option<Fq12>,
    /// Boundary outputs for the external pairing check (treated as a hint; guest recomputes).
    pub pairing_boundary: PairingBoundary,
    /// The recursion SNARK proof itself (Hyrax + sumchecks).
    pub proof: RecursionProof<Fq, FS, HyraxPCS>,
}

impl<FS: Transcript> GuestSerialize for RecursionArtifact<FS> {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.stage8_combine_hint.guest_serialize(w)?;
        self.pairing_boundary.guest_serialize(w)?;
        self.proof.guest_serialize(w)?;
        Ok(())
    }
}

impl<FS: Transcript> GuestDeserialize for RecursionArtifact<FS> {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            stage8_combine_hint: Option::<Fq12>::guest_deserialize(r)?,
            pairing_boundary: PairingBoundary::guest_deserialize(r)?,
            proof: RecursionProof::<Fq, FS, HyraxPCS>::guest_deserialize(r)?,
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
        commitments_map.insert(poly, *commitment);
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
            commitments_map.insert(CommittedPolynomial::TrustedAdvice, *commitment);
        }
    }
    if needs_untrusted_advice {
        if let Some(ref commitment) = v.proof.untrusted_advice_commitment {
            commitments_map.insert(CommittedPolynomial::UntrustedAdvice, *commitment);
        }
    }

    // Program commitments (BytecodeChunk and ProgramImageInit) in committed mode.
    if v.proof.program_mode == ProgramMode::Committed {
        let committed = v.preprocessing.program.as_committed()?;
        for (idx, commitment) in committed.bytecode_commitments.iter().enumerate() {
            commitments_map
                .entry(CommittedPolynomial::BytecodeChunk(idx))
                .or_insert(*commitment);
        }
        if stage8_snapshot
            .polynomial_claims
            .iter()
            .any(|(p, _)| *p == CommittedPolynomial::ProgramImageInit)
        {
            commitments_map.insert(
                CommittedPolynomial::ProgramImageInit,
                committed.program_image_commitment,
            );
        }
    }

    // Hyrax setup for recursion proving (dense commitment/opening).
    //
    // Hyrax uses the same type for prover+verifier setup (`PedersenGenerators`), and the base
    // preprocessing already includes a cached recursion-sized setup.
    let hyrax_prover_setup = &preprocessing.hyrax_recursion_setup;

    let (recursion_snark_proof, _constraint_metadata, pairing_boundary, stage8_combine_hint) =
        RecursionProver::<Fq>::prove::<F, DoryPCS, FS>(
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
        stage8_combine_hint,
        pairing_boundary,
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
    let _stage8_prep_commitments_cycle = "verify_recursion_stage8_prep_commitments_map_total";
    start_cycle_tracking(_stage8_prep_commitments_cycle);

    // Single scan over polynomial_claims for optional commitments.
    let mut needs_trusted_advice = false;
    let mut needs_untrusted_advice = false;
    let mut needs_program_image_init = false;
    for (p, _claim) in dory_snap.polynomial_claims.iter() {
        match *p {
            CommittedPolynomial::TrustedAdvice => needs_trusted_advice = true,
            CommittedPolynomial::UntrustedAdvice => needs_untrusted_advice = true,
            CommittedPolynomial::ProgramImageInit => needs_program_image_init = true,
            _ => {}
        }
    }

    let mut commitments_map = HashMap::<
        CommittedPolynomial,
        <DoryPCS as CommitmentScheme>::Commitment,
    >::with_capacity(v.proof.commitments.len() + 8);
    for (poly, commitment) in all_committed_polynomials(&v.one_hot_params)
        .into_iter()
        .zip(v.proof.commitments.iter())
    {
        commitments_map.insert(poly, *commitment);
    }
    if let Some(ref commitment) = v.trusted_advice_commitment {
        if needs_trusted_advice {
            commitments_map.insert(CommittedPolynomial::TrustedAdvice, *commitment);
        }
    }
    if let Some(ref commitment) = v.proof.untrusted_advice_commitment {
        if needs_untrusted_advice {
            commitments_map.insert(CommittedPolynomial::UntrustedAdvice, *commitment);
        }
    }
    if v.proof.program_mode == ProgramMode::Committed {
        let committed = v.preprocessing.program.as_committed()?;
        for (idx, commitment) in committed.bytecode_commitments.iter().enumerate() {
            commitments_map
                .entry(CommittedPolynomial::BytecodeChunk(idx))
                .or_insert(*commitment);
        }
        if needs_program_image_init {
            commitments_map.insert(
                CommittedPolynomial::ProgramImageInit,
                committed.program_image_commitment,
            );
        }
    }
    end_cycle_tracking(_stage8_prep_commitments_cycle);

    // Deterministic combine plan (must match the prover's ordering).
    let _stage8_prep_rlc_cycle = "verify_recursion_stage8_prep_rlc_total";
    start_cycle_tracking(_stage8_prep_rlc_cycle);
    let rlc_map = compute_rlc_coefficients(&gamma_powers, dory_snap.polynomial_claims.iter().copied());
    let (combine_coeffs, combine_commitments): (
        Vec<F>,
        Vec<<DoryPCS as CommitmentScheme>::Commitment>,
    ) = rlc_map
        .into_iter()
        .map(|(poly, coeff)| {
            (
                coeff,
                *commitments_map
                    .get(&poly)
                    .expect("missing commitment for polynomial in batch"),
            )
        })
        .unzip();
    end_cycle_tracking(_stage8_prep_rlc_cycle);

    // Get Stage 8 combine hint
    let hint_fq12 = recursion
        .stage8_combine_hint
        .as_ref()
        .ok_or_else(|| anyhow!("missing required Stage 8 combine hint (stage8_combine_hint)"))?;
    let joint_commitment: <DoryPCS as CommitmentScheme>::Commitment =
        <DoryPCS as RecursionExt<F>>::combine_with_hint_fq12(hint_fq12);

    // Build symbolic AST at the pre-Stage8-proof transcript state.
    //
    // Important: `build_symbolic_ast()` runs the Dory verifier in symbolic mode and therefore
    // mutates the transcript to the same post-Stage8 state expected by the recursion SNARK.
    // We can reuse that transcript directly and skip the (cheaper but still nontrivial)
    // `replay_opening_proof_transcript()` pass.
    let _stage8_prep_ast_cycle = "verify_recursion_stage8_prep_symbolic_ast_total";
    start_cycle_tracking(_stage8_prep_ast_cycle);
    #[cfg(debug_assertions)]
    let pre_opening_proof_transcript_dbg = pre_opening_proof_transcript.clone();
    let mut ast_transcript = pre_opening_proof_transcript;
    let ast = <DoryPCS as RecursionExt<F>>::build_symbolic_ast(
        &v.proof.joint_opening_proof,
        &v.preprocessing.generators,
        &mut ast_transcript,
        &dory_snap.opening_point.r,
        &joint_claim,
        &joint_commitment,
    )
    .map_err(|e| anyhow!("Stage 8 symbolic AST build failed: {e:?}"))?;
    end_cycle_tracking(_stage8_prep_ast_cycle);

    // Use the post-Stage8 transcript state expected by recursion SNARK.
    //
    // - In release builds (including guest cycle-tracking), we reuse the transcript advanced by
    //   `build_symbolic_ast()` and skip the additional replay pass for performance.
    // - In debug builds, we conservatively run the replay pass and use that transcript.
    //   (We also have a dedicated regression test that checks symbolic-vs-replay equality for
    //   the concrete transcript used in recursion benchmarking.)
    #[cfg(debug_assertions)]
    {
        let _stage8_prep_replay_cycle = "verify_recursion_stage8_prep_replay_total";
        start_cycle_tracking(_stage8_prep_replay_cycle);
        let mut replay_transcript = pre_opening_proof_transcript_dbg;
        <DoryPCS as RecursionExt<F>>::replay_opening_proof_transcript(
            &v.proof.joint_opening_proof,
            &mut replay_transcript,
        )
        .map_err(|e| anyhow!("Stage 8 PCS FS replay failed: {e:?}"))?;
        end_cycle_tracking(_stage8_prep_replay_cycle);
        v.transcript = replay_transcript;
    }
    #[cfg(not(debug_assertions))]
    {
        v.transcript = ast_transcript;
    }

    // Derive recursion verifier input (NO expensive group ops).
    //
    // Rationale: the recursion SNARK now enforces AST-driven wiring/boundary constraints (Stage 2),
    // including binding non-input bases/points and the pairing boundary. Re-evaluating the Dory AST
    // (pairings / scalar muls / GT exp) inside the verifier is therefore unnecessary and extremely
    // expensive in the zkVM / cycle-tracking path.
    let _stage8_prep_plan_cycle = "verify_recursion_stage8_prep_plan_total";
    start_cycle_tracking(_stage8_prep_plan_cycle);
    let plan = derive_plan_with_hints(
        &ast,
        &v.proof.joint_opening_proof,
        &v.preprocessing.generators,
        joint_commitment,
        &combine_commitments,
        &combine_coeffs,
        recursion.pairing_boundary.clone(),
        *hint_fq12,
    )
    .map_err(|e| anyhow!("AST->recursion-plan derivation failed: {e:?}"))?;
    end_cycle_tracking(_stage8_prep_plan_cycle);
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

    let recursion_verifier = RecursionVerifier::<Fq>::new(plan.verifier_input);
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
