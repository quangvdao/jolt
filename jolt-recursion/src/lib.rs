//! Jolt recursion prover/verifier.
//!
//! This crate contains the recursion SNARK (and Dory recursion backend glue) extracted from
//! `jolt-core`. The base `jolt-core` proof is recursion-free; recursion is an optional, separate
//! artifact built on top of `jolt-core`'s public stage verification APIs.

use anyhow::Result;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

// Used by macros in extracted recursion code (expects `$crate::paste`).
pub use paste;
// Used by macros in extracted recursion code (expects `$crate::allocative`).
pub use allocative;

// -----------------------------------------------------------------------------
// Compatibility re-exports
// -----------------------------------------------------------------------------
//
// The extracted recursion code was originally written inside `jolt-core` and uses many
// `crate::...` paths. To keep the move mechanical, we provide a thin `crate::field`, `crate::poly`,
// `crate::zkvm`, etc layer that re-exports most of `jolt-core` and adds only recursion-specific
// pieces (notably `RecursionExt` and Dory recursion helpers).

pub mod field {
    pub use jolt_core::field::*;
}

pub mod subprotocols {
    pub use jolt_core::subprotocols::*;
}

pub mod transcripts {
    pub use jolt_core::transcripts::*;
}

pub mod utils {
    pub use jolt_core::utils::*;
}

pub mod poly {
    pub use jolt_core::poly::{
        compact_polynomial, dense_mlpoly, eq_plus_one_poly, eq_poly, identity_poly, lagrange_poly,
        lt_poly, multilinear_polynomial, multiquadratic_poly, one_hot_polynomial, opening_proof,
        prefix_suffix, ra_poly, range_mask_polynomial, rlc_polynomial, rlc_utils, shared_ra_polys,
        split_eq_poly, unipoly,
    };

    pub mod commitment {
        pub use jolt_core::poly::commitment::{hyrax, hyrax::*};

        pub mod commitment_scheme {
            pub use jolt_core::poly::commitment::commitment_scheme::{
                CommitmentScheme, StreamingCommitmentScheme,
            };

            use crate::field::JoltField;
            use crate::transcripts::Transcript;
            use crate::utils::errors::ProofVerifyError;
            use crate::zkvm::proof_serialization::PairingBoundary;
            use crate::zkvm::recursion::witness::GTCombineWitness;

            /// Extension trait for commitment schemes that adds recursion support.
            ///
            /// This trait enables a commitment scheme to be used with Jolt's recursion SNARK.
            /// Currently only Dory implements this trait.
            pub trait RecursionExt<F: JoltField>: CommitmentScheme<Field = F> {
                /// Witness collection type for recursion constraint building.
                type Witness;

                /// Commitment-scheme specific AST type used to derive recursion obligations.
                type Ast;

                /// Hint for combine_commitments offloading (the final combined commitment).
                type CombineHint;

                fn witness_gen_with_ast<ProofTranscript: Transcript>(
                    proof: &Self::Proof,
                    setup: &Self::VerifierSetup,
                    transcript: &mut ProofTranscript,
                    point: &[<F as JoltField>::Challenge],
                    evaluation: &F,
                    commitment: &Self::Commitment,
                ) -> Result<(Self::Witness, Self::Ast), ProofVerifyError>;

                fn build_symbolic_ast<ProofTranscript: Transcript>(
                    proof: &Self::Proof,
                    setup: &Self::VerifierSetup,
                    transcript: &mut ProofTranscript,
                    point: &[<F as JoltField>::Challenge],
                    evaluation: &F,
                    commitment: &Self::Commitment,
                ) -> Result<Self::Ast, ProofVerifyError>;

                fn derive_pairing_boundary_from_ast(
                    ast: &Self::Ast,
                    proof: &Self::Proof,
                    setup: &Self::VerifierSetup,
                    joint_commitment: Self::Commitment,
                    combine_commitments: &[Self::Commitment],
                    combine_coeffs: &[F],
                ) -> Result<PairingBoundary, ProofVerifyError>;

                fn replay_opening_proof_transcript<ProofTranscript: Transcript>(
                    proof: &Self::Proof,
                    transcript: &mut ProofTranscript,
                ) -> Result<(), ProofVerifyError>;

                fn generate_combine_witness<C: std::borrow::Borrow<Self::Commitment>>(
                    commitments: &[C],
                    coeffs: &[F],
                ) -> (GTCombineWitness, Self::CombineHint);

                fn combine_with_hint(hint: &Self::CombineHint) -> Self::Commitment;

                fn combine_hint_to_fq12(hint: &Self::CombineHint) -> ark_bn254::Fq12;

                fn combine_with_hint_fq12(hint: &ark_bn254::Fq12) -> Self::Commitment;
            }
        }

        pub mod dory {
            pub use jolt_core::poly::commitment::dory::{
                deserialize_ark_dory_proof_marked, ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT,
                ArkworksProverSetup, ArkworksVerifierSetup, DoryCommitmentScheme, DoryContext,
                DoryGlobals, DoryLayout, JoltFieldWrapper, JoltG1Routines, JoltG2Routines, BN254,
            };

            // Re-export wrappers as a module so `super::wrappers::...` works unchanged.
            pub use jolt_core::poly::commitment::dory::wrappers;

            // Provide the expected `super::jolt_dory_routines::{...}` module path.
            pub mod jolt_dory_routines {
                pub use super::{JoltG1Routines, JoltG2Routines};
            }

            pub mod commitment_scheme {
                use crate::field::JoltField;
                pub use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
                use jolt_core::poly::commitment::dory::{DoryGlobals, DoryLayout};
                use jolt_core::utils::math::Math;

                /// Reorders `opening_point` for AddressMajor layout.
                ///
                /// For AddressMajor layout, reorders opening_point from `[r_address, r_cycle]` to
                /// `[r_cycle, r_address]`. For CycleMajor, returns the point unchanged.
                pub(crate) fn reorder_opening_point_for_layout<F: JoltField>(
                    opening_point: &[F::Challenge],
                ) -> Vec<F::Challenge> {
                    if DoryGlobals::get_layout() == DoryLayout::AddressMajor {
                        let log_t = DoryGlobals::get_T().log_2();
                        let log_k = opening_point.len().saturating_sub(log_t);
                        let (r_address, r_cycle) = opening_point.split_at(log_k);
                        [r_cycle, r_address].concat()
                    } else {
                        opening_point.to_vec()
                    }
                }
            }

            // These recursion-only modules were removed from `jolt-core`; we compile them here.
            #[path = "../../../../../jolt-core/src/poly/commitment/dory/instance_plan.rs"]
            pub mod instance_plan;
            #[path = "../../../../../jolt-core/src/poly/commitment/dory/recursion.rs"]
            pub mod recursion;
            #[path = "../../../../../jolt-core/src/poly/commitment/dory/witness/mod.rs"]
            pub mod witness;

            pub use instance_plan::{
                derive_from_dory_ast, derive_plan_with_hints, DerivedRecursionInput,
                DerivedRecursionPlan,
            };
        }
    }
}

pub mod zkvm {
    pub use jolt_core::zkvm::guest_serde;
    pub use jolt_core::zkvm::witness;

    pub mod proof_serialization {
        pub use crate::{NonInputBaseHints, PairingBoundary};
    }

    // The recursion SNARK implementation (extracted from `jolt-core`).
    #[path = "../../../jolt-core/src/zkvm/recursion/mod.rs"]
    pub mod recursion;
}

/// Boundary outputs for the final external pairing check in recursion mode.
///
/// The verifier must **not trust** prover-supplied values here; it should re-derive the expected
/// boundary from public inputs (proof + setup) and compare.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PairingBoundary {
    pub p1_g1: ark_bn254::G1Affine,
    pub p1_g2: ark_bn254::G2Affine,
    pub p2_g1: ark_bn254::G1Affine,
    pub p2_g2: ark_bn254::G2Affine,
    pub p3_g1: ark_bn254::G1Affine,
    pub p3_g2: ark_bn254::G2Affine,
    pub rhs: ark_bn254::Fq12,
}

/// Hints for recursion instance-plan derivation when an op's base/point is not an `AstOp::Input`.
///
/// These are used to avoid requiring the verifier to evaluate the full Dory verification DAG just
/// to recover bases/points for public inputs in the recursion verifier input.
///
/// **Security note**: without wiring/boundary constraints, these hints are not bound to the Dory
/// verification computation. They are intended for performance/profiling until wiring is added.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NonInputBaseHints {
    /// One entry per Dory-traced `GTExp` op, in OpId-sorted order.
    /// `None` means the base was an `AstOp::Input` and can be resolved by the verifier.
    pub gt_exp_base_hints: Vec<Option<ark_bn254::Fq12>>,
    /// One entry per Dory-traced `G1ScalarMul` op, in OpId-sorted order.
    pub g1_scalar_mul_base_hints: Vec<Option<ark_bn254::G1Affine>>,
    /// One entry per Dory-traced `G2ScalarMul` op, in OpId-sorted order.
    pub g2_scalar_mul_base_hints: Vec<Option<ark_bn254::G2Affine>>,
}

/// Recursion proof artifact.
///
/// This is separate from the base `jolt-core` proof. The structure is *flat*: it contains both the
/// Stage 8 auxiliary data and the recursion SNARK proof pieces.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct RecursionProof<FS: jolt_core::transcripts::Transcript> {
    /// Hint for combine_commitments offloading (the combined GT element).
    pub stage8_combine_hint: Option<ark_bn254::Fq12>,
    /// Boundary outputs for the external pairing check.
    pub pairing_boundary: PairingBoundary,
    /// Minimal hints for Dory instance-plan derivation.
    pub non_input_base_hints: NonInputBaseHints,

    /// Stage 1: recursion sumcheck instance proof.
    pub stage1_proof: jolt_core::subprotocols::sumcheck::SumcheckInstanceProof<ark_bn254::Fq, FS>,
    /// Stage 2: recursion sumcheck instance proof.
    pub stage2_proof: jolt_core::subprotocols::sumcheck::SumcheckInstanceProof<ark_bn254::Fq, FS>,
    /// Stage 3: packed evaluation.
    pub stage3_packed_eval: ark_bn254::Fq,
    /// Hyrax opening proof.
    pub opening_proof:
        <jolt_core::poly::commitment::hyrax::Hyrax<1, ark_grumpkin::Projective> as jolt_core::poly::commitment::commitment_scheme::CommitmentScheme>::Proof,
    /// Opening claims for virtual polynomials.
    pub opening_claims: jolt_core::poly::opening_proof::Openings<ark_bn254::Fq>,
    /// Dense polynomial commitment.
    pub dense_commitment:
        <jolt_core::poly::commitment::hyrax::Hyrax<1, ark_grumpkin::Projective> as jolt_core::poly::commitment::commitment_scheme::CommitmentScheme>::Commitment,
}

fn clone_base_proof_via_guest_serde<FS: transcripts::Transcript>(
    proof: &jolt_core::zkvm::proof_serialization::JoltProof<
        ark_bn254::Fr,
        poly::commitment::dory::DoryCommitmentScheme,
        FS,
    >,
) -> Result<
    jolt_core::zkvm::proof_serialization::JoltProof<
        ark_bn254::Fr,
        poly::commitment::dory::DoryCommitmentScheme,
        FS,
    >,
> {
    use anyhow::anyhow;
    use zkvm::guest_serde::{GuestDeserialize, GuestSerialize};

    let mut buf = Vec::new();
    proof
        .guest_serialize(&mut buf)
        .map_err(|e| anyhow!("failed to serialize base proof: {e:?}"))?;

    let mut cursor = std::io::Cursor::new(buf);
    let cloned = <jolt_core::zkvm::proof_serialization::JoltProof<
        ark_bn254::Fr,
        poly::commitment::dory::DoryCommitmentScheme,
        FS,
    > as GuestDeserialize>::guest_deserialize(&mut cursor)
    .map_err(|e| anyhow!("failed to deserialize base proof: {e:?}"))?;
    Ok(cloned)
}

/// Generate a recursion proof for a base Jolt proof.
///
/// This runs base verification Stages 1–7 to reconstruct transcript state, then derives Stage 8
/// artifacts and proves the recursion SNARK.
pub fn prove_recursion<FS: transcripts::Transcript>(
    preprocessing: &jolt_core::zkvm::verifier::JoltVerifierPreprocessing<
        ark_bn254::Fr,
        poly::commitment::dory::DoryCommitmentScheme,
    >,
    program_io: tracer::JoltDevice,
    trusted_advice_commitment: Option<
        <poly::commitment::dory::DoryCommitmentScheme as poly::commitment::commitment_scheme::CommitmentScheme>::Commitment,
    >,
    base_proof: &jolt_core::zkvm::proof_serialization::JoltProof<
        ark_bn254::Fr,
        poly::commitment::dory::DoryCommitmentScheme,
        FS,
    >,
) -> Result<RecursionProof<FS>> {
    use anyhow::{anyhow, Context};
    use poly::commitment::commitment_scheme::CommitmentScheme;
    use poly::commitment::commitment_scheme::RecursionExt;
    use poly::commitment::dory::{DoryContext, DoryGlobals};
    use zkvm::witness::CommittedPolynomial;

    type F = ark_bn254::Fr;
    type DoryPCS = poly::commitment::dory::DoryCommitmentScheme;
    type HyraxPCS = poly::commitment::hyrax::Hyrax<1, ark_grumpkin::Projective>;

    let base_proof = clone_base_proof_via_guest_serde(base_proof)?;
    let mut v = jolt_core::zkvm::verifier::JoltVerifier::new(
        preprocessing,
        base_proof,
        program_io,
        trusted_advice_commitment,
        None,
    )
    .map_err(|e| anyhow!("failed to construct base verifier: {e:?}"))?;

    // Stage-0 transcript preamble + commitments (mirrors `jolt-core` verifier).
    jolt_core::zkvm::fiat_shamir_preamble(
        &v.program_io,
        v.proof.ram_K,
        v.proof.trace_length,
        &mut v.transcript,
    );
    for commitment in &v.proof.commitments {
        v.transcript.append_serializable(commitment);
    }
    if let Some(ref untrusted_advice_commitment) = v.proof.untrusted_advice_commitment {
        v.transcript
            .append_serializable(untrusted_advice_commitment);
    }
    if let Some(ref trusted_advice_commitment) = v.trusted_advice_commitment {
        v.transcript.append_serializable(trusted_advice_commitment);
    }
    if v.proof.program_mode == jolt_core::zkvm::config::ProgramMode::Committed {
        let committed = v
            .preprocessing
            .program
            .as_committed()
            .context("committed program mode requires committed preprocessing")?;
        for commitment in &committed.bytecode_commitments {
            v.transcript.append_serializable(commitment);
        }
        v.transcript
            .append_serializable(&committed.program_image_commitment);
    }

    // Base stages 1..7 (no Stage 8 PCS verification).
    v.verify_stage1()?;
    v.verify_stage2()?;
    v.verify_stage3()?;
    v.verify_stage4()?;
    v.verify_stage5()?;
    v.verify_stage6a()?;
    v.verify_stage6b()?;
    v.verify_stage7()?;

    // Ensure Dory globals match the proof layout before any Stage 8 replay / witness generation.
    let _dory_globals_guard =
        if v.proof.program_mode == jolt_core::zkvm::config::ProgramMode::Committed {
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

    // Stage 8 snapshot (claims + ordering) from the verifier state.
    let dory_snap = v.build_dory_verify_snapshot()?;

    // Fork transcript right after gamma sampling (before PCS opening proof transcript mutations).
    let mut pre_opening_proof_transcript = v.transcript.clone();
    pre_opening_proof_transcript.append_scalars(&dory_snap.claims);
    let gamma_powers: Vec<F> =
        pre_opening_proof_transcript.challenge_scalar_powers(dory_snap.claims.len());
    let joint_claim: F = gamma_powers
        .iter()
        .zip(dory_snap.claims.iter())
        .map(|(g, c)| *g * c)
        .sum();

    let stage8_snapshot = zkvm::recursion::prover::DoryOpeningSnapshot::<F, FS> {
        pre_opening_proof_transcript: pre_opening_proof_transcript.clone(),
        opening_point: dory_snap.opening_point.r.clone(),
        polynomial_claims: dory_snap.polynomial_claims.clone(),
        gamma_powers: gamma_powers.clone(),
        joint_claim,
    };

    // Advance the main transcript to the post-Stage8 state expected by recursion SNARK.
    v.transcript = pre_opening_proof_transcript;
    <DoryPCS as RecursionExt<F>>::replay_opening_proof_transcript(
        &v.proof.stage8_opening_proof,
        &mut v.transcript,
    )
    .map_err(|e| anyhow!("Stage 8 PCS FS replay failed: {e:?}"))?;

    // Build the commitments map needed by Stage 8 offloading + instance-plan derivation.
    let mut commitments_map = std::collections::HashMap::<
        CommittedPolynomial,
        <DoryPCS as CommitmentScheme>::Commitment,
    >::new();
    let all_polys = jolt_core::zkvm::witness::all_committed_polynomials(&v.one_hot_params);
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
    if v.proof.program_mode == jolt_core::zkvm::config::ProgramMode::Committed {
        let committed = v.preprocessing.program.as_committed()?;
        for (idx, commitment) in committed.bytecode_commitments.iter().enumerate() {
            commitments_map
                .entry(CommittedPolynomial::BytecodeChunk(idx))
                .or_insert_with(|| commitment.clone());
        }
        if needs_trusted_advice || needs_untrusted_advice {
            // (no-op; keeps intent explicit)
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
    let hyrax_prover_setup =
        <HyraxPCS as CommitmentScheme>::setup_prover(zkvm::recursion::MAX_RECURSION_DENSE_NUM_VARS);

    let (
        recursion_snark_proof,
        _constraint_metadata,
        pairing_boundary,
        stage8_combine_hint,
        non_input_base_hints,
    ) = zkvm::recursion::prover::RecursionProver::<ark_bn254::Fq>::prove::<F, DoryPCS, FS>(
        &mut v.transcript,
        &hyrax_prover_setup,
        zkvm::recursion::prover::RecursionInput {
            stage8_opening_proof: &v.proof.stage8_opening_proof,
            stage8_snapshot,
            verifier_setup: &v.preprocessing.generators,
            commitments: &commitments_map,
        },
    )
    .map_err(|e| anyhow!("failed to generate recursion proof: {e:?}"))?;

    Ok(RecursionProof {
        stage8_combine_hint,
        pairing_boundary,
        non_input_base_hints,
        stage1_proof: recursion_snark_proof.stage1_proof,
        stage2_proof: recursion_snark_proof.stage2_proof,
        stage3_packed_eval: recursion_snark_proof.stage3_packed_eval,
        opening_proof: recursion_snark_proof.opening_proof,
        opening_claims: recursion_snark_proof.opening_claims,
        dense_commitment: recursion_snark_proof.dense_commitment,
    })
}

/// Verify a recursion proof for a base Jolt proof.
pub fn verify_recursion<FS: transcripts::Transcript>(
    preprocessing: &jolt_core::zkvm::verifier::JoltVerifierPreprocessing<
        ark_bn254::Fr,
        poly::commitment::dory::DoryCommitmentScheme,
    >,
    program_io: tracer::JoltDevice,
    trusted_advice_commitment: Option<
        <poly::commitment::dory::DoryCommitmentScheme as poly::commitment::commitment_scheme::CommitmentScheme>::Commitment,
    >,
    base_proof: &jolt_core::zkvm::proof_serialization::JoltProof<
        ark_bn254::Fr,
        poly::commitment::dory::DoryCommitmentScheme,
        FS,
    >,
    recursion_proof: &RecursionProof<FS>,
) -> Result<()> {
    use anyhow::{anyhow, Context};
    use ark_ec::AffineRepr;
    use dory::primitives::arithmetic::PairingCurve;
    use poly::commitment::commitment_scheme::CommitmentScheme;
    use poly::commitment::commitment_scheme::RecursionExt;
    use poly::commitment::dory::{ArkG1, ArkG2, ArkGT, DoryContext, DoryGlobals, BN254};
    use zkvm::witness::CommittedPolynomial;

    type F = ark_bn254::Fr;
    type DoryPCS = poly::commitment::dory::DoryCommitmentScheme;
    type HyraxPCS = poly::commitment::hyrax::Hyrax<1, ark_grumpkin::Projective>;

    let base_proof = clone_base_proof_via_guest_serde(base_proof)?;
    let mut v = jolt_core::zkvm::verifier::JoltVerifier::new(
        preprocessing,
        base_proof,
        program_io,
        trusted_advice_commitment,
        None,
    )
    .map_err(|e| anyhow!("failed to construct base verifier: {e:?}"))?;

    // Stage-0 transcript preamble + commitments (mirrors `jolt-core` verifier).
    jolt_core::zkvm::fiat_shamir_preamble(
        &v.program_io,
        v.proof.ram_K,
        v.proof.trace_length,
        &mut v.transcript,
    );
    for commitment in &v.proof.commitments {
        v.transcript.append_serializable(commitment);
    }
    if let Some(ref untrusted_advice_commitment) = v.proof.untrusted_advice_commitment {
        v.transcript
            .append_serializable(untrusted_advice_commitment);
    }
    if let Some(ref trusted_advice_commitment) = v.trusted_advice_commitment {
        v.transcript.append_serializable(trusted_advice_commitment);
    }
    if v.proof.program_mode == jolt_core::zkvm::config::ProgramMode::Committed {
        let committed = v
            .preprocessing
            .program
            .as_committed()
            .context("committed program mode requires committed preprocessing")?;
        for commitment in &committed.bytecode_commitments {
            v.transcript.append_serializable(commitment);
        }
        v.transcript
            .append_serializable(&committed.program_image_commitment);
    }

    // Base stages 1..7.
    v.verify_stage1()?;
    v.verify_stage2()?;
    v.verify_stage3()?;
    v.verify_stage4()?;
    v.verify_stage5()?;
    v.verify_stage6a()?;
    v.verify_stage6b()?;
    v.verify_stage7()?;

    // Ensure Dory globals match the proof layout before any Stage 8 replay / AST derivation.
    let _dory_globals_guard =
        if v.proof.program_mode == jolt_core::zkvm::config::ProgramMode::Committed {
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

    // Stage 8 snapshot (claims + ordering) from the verifier state.
    let dory_snap = v.build_dory_verify_snapshot()?;

    // Transcript state right after gamma sampling (before PCS proof transcript mutations).
    let mut pre_opening_proof_transcript = v.transcript.clone();
    pre_opening_proof_transcript.append_scalars(&dory_snap.claims);
    let gamma_powers: Vec<F> =
        pre_opening_proof_transcript.challenge_scalar_powers(dory_snap.claims.len());
    let joint_claim: F = gamma_powers
        .iter()
        .zip(dory_snap.claims.iter())
        .map(|(g, c)| *g * c)
        .sum();

    // Build commitments map (must match Stage 8 native verifier path).
    let mut commitments_map = std::collections::HashMap::<
        CommittedPolynomial,
        <DoryPCS as CommitmentScheme>::Commitment,
    >::new();
    for (poly, commitment) in jolt_core::zkvm::witness::all_committed_polynomials(&v.one_hot_params)
        .into_iter()
        .zip(v.proof.commitments.iter())
    {
        commitments_map.insert(poly, commitment.clone());
    }

    // Add advice commitments if they're part of the Stage 8 batch.
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
    if v.proof.program_mode == jolt_core::zkvm::config::ProgramMode::Committed {
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
    let rlc_map = poly::rlc_utils::compute_rlc_coefficients(
        &gamma_powers,
        dory_snap.polynomial_claims.clone(),
    );
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

    // Use the prover-supplied Stage 8 combine hint when provided (fast path).
    let joint_commitment: <DoryPCS as CommitmentScheme>::Commitment =
        match recursion_proof.stage8_combine_hint.as_ref() {
            Some(hint_fq12) => <DoryPCS as RecursionExt<F>>::combine_with_hint_fq12(hint_fq12),
            None => <DoryPCS as CommitmentScheme>::combine_commitments(
                &combine_commitments,
                &combine_coeffs,
            ),
        };

    // Build symbolic AST on a transcript clone at the pre-Stage8-proof state.
    let mut ast_transcript = pre_opening_proof_transcript.clone();
    let ast = <DoryPCS as RecursionExt<F>>::build_symbolic_ast(
        &v.proof.stage8_opening_proof,
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
        &v.proof.stage8_opening_proof,
        &mut v.transcript,
    )
    .map_err(|e| anyhow!("Stage 8 PCS FS replay failed: {e:?}"))?;

    // Dory-specific instance plan derivation.
    let combine_coeffs_fr: Vec<ark_bn254::Fr> = combine_coeffs;
    let joint_commitment_dory: ArkGT = joint_commitment;
    let combine_commitments_dory: Vec<ArkGT> = combine_commitments;

    let derived_plan = poly::commitment::dory::derive_plan_with_hints(
        &ast,
        &v.proof.stage8_opening_proof,
        &v.preprocessing.generators,
        joint_commitment_dory,
        &combine_commitments_dory,
        &combine_coeffs_fr,
        &recursion_proof.non_input_base_hints,
        recursion_proof.pairing_boundary.clone(),
        joint_commitment_dory.0,
    )
    .map_err(|e| anyhow!("AST->instance-plan derivation failed: {e:?}"))?;

    if derived_plan.dense_num_vars > zkvm::recursion::MAX_RECURSION_DENSE_NUM_VARS {
        return Err(anyhow!(
            "dense_num_vars {} exceeds max {}",
            derived_plan.dense_num_vars,
            zkvm::recursion::MAX_RECURSION_DENSE_NUM_VARS
        ));
    }

    // Verify recursion SNARK.
    let hyrax_prover_setup =
        <HyraxPCS as CommitmentScheme>::setup_prover(zkvm::recursion::MAX_RECURSION_DENSE_NUM_VARS);
    let hyrax_verifier_setup = <HyraxPCS as CommitmentScheme>::setup_verifier(&hyrax_prover_setup);

    let inner = zkvm::recursion::prover::RecursionProof::<ark_bn254::Fq, FS, HyraxPCS> {
        stage1_proof: recursion_proof.stage1_proof.clone(),
        stage2_proof: recursion_proof.stage2_proof.clone(),
        stage3_packed_eval: recursion_proof.stage3_packed_eval,
        opening_proof: recursion_proof.opening_proof.clone(),
        opening_claims: recursion_proof.opening_claims.clone(),
        dense_commitment: recursion_proof.dense_commitment.clone(),
    };

    let recursion_verifier = zkvm::recursion::verifier::RecursionVerifier::<ark_bn254::Fq>::new(
        derived_plan.verifier_input,
    );
    let ok = recursion_verifier
        .verify::<FS, HyraxPCS>(
            &inner,
            &mut v.transcript,
            &inner.dense_commitment,
            &hyrax_verifier_setup,
        )
        .map_err(|e| anyhow!("Recursion verification failed: {e:?}"))?;
    if !ok {
        return Err(anyhow!("Recursion proof verification failed"));
    }

    // External pairing check (pairing boundary treated as prover hint for now).
    let got = &recursion_proof.pairing_boundary;
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
    if lhs.0 != got.rhs {
        return Err(anyhow!("external pairing check failed"));
    }

    Ok(())
}
