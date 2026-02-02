use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::{CommitmentScheme, RecursionExt},
            dory::{
                instance_plan::derive_plan_with_hints, DoryCommitmentScheme, DoryContext,
                DoryGlobals,
            },
            hyrax::Hyrax,
        },
        dense_mlpoly::DensePolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        rlc_utils::compute_rlc_coefficients,
    },
    transcripts::{Blake2bTranscript, Transcript},
    zkvm::{
        recursion::{
            prover::{DoryOpeningSnapshot, RecursionInput, RecursionProof, RecursionProver},
            verifier::RecursionVerifier,
            RecursionVerifierInput, MAX_RECURSION_DENSE_NUM_VARS,
        },
        witness::CommittedPolynomial,
    },
};

use ark_bn254::{Fq, Fq12, Fr, G1Affine, G2Affine};
use ark_ff::{One, UniformRand};
use ark_grumpkin::Projective as GrumpkinProjective;
use ark_std::test_rng;
use serial_test::serial;
use std::collections::HashMap;

use crate::poly::commitment::dory::wrappers::ArkDoryProof;
use crate::zkvm::proof_serialization::PairingBoundary;

type HyraxPCS = Hyrax<1, GrumpkinProjective>;

struct RecursionFixture {
    // Stage-8 artifacts
    dory_proof: <DoryCommitmentScheme as CommitmentScheme>::Proof,
    ark_dory_proof: ArkDoryProof,
    verifier_setup: <DoryCommitmentScheme as CommitmentScheme>::VerifierSetup,
    stage8_pre_transcript: Blake2bTranscript,
    opening_point: Vec<<Fr as JoltField>::Challenge>,
    joint_claim: Fr,
    combine_commitments: Vec<<DoryCommitmentScheme as CommitmentScheme>::Commitment>,
    combine_coeffs: Vec<Fr>,
    joint_commitment: <DoryCommitmentScheme as CommitmentScheme>::Commitment,
    joint_commitment_fq12: Fq12,
    // Recursion outputs
    recursion_proof: RecursionProof<Fq, Blake2bTranscript, HyraxPCS>,
    pairing_boundary: PairingBoundary,
    hyrax_prover_setup: <HyraxPCS as CommitmentScheme>::ProverSetup,
}

fn build_fixture() -> RecursionFixture {
    // Small deterministic Dory context.
    DoryGlobals::reset();
    DoryGlobals::initialize_context(1 << 2, 1 << 2, DoryContext::Main, None);

    let mut rng = test_rng();

    // Small test polynomial over Fr.
    let num_vars = 4;
    let coeffs: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs));

    let prover_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover(num_vars);
    let verifier_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_verifier(&prover_setup);
    let (commitment, hint) =
        <DoryCommitmentScheme as CommitmentScheme>::commit(&poly, &prover_setup);

    // Opening point (big-endian challenges for Dory).
    let mut point_transcript: Blake2bTranscript = Transcript::new(b"pt");
    let opening_point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
        .map(|_| point_transcript.challenge_scalar_optimized::<Fr>())
        .collect();
    let eval = PolynomialEvaluation::evaluate(&poly, &opening_point);

    // Stage 8 transcript prefix: append claims and sample gamma powers.
    let mut stage8_transcript: Blake2bTranscript = Transcript::new(b"stage8");
    let polynomial_claims = vec![(CommittedPolynomial::RdInc, eval)];
    stage8_transcript.append_scalars(&[eval]);
    let gamma_powers: Vec<Fr> = stage8_transcript.challenge_scalar_powers(1);
    let joint_claim: Fr = gamma_powers[0] * eval;
    let stage8_pre_transcript = stage8_transcript.clone();

    // Stage 8 proof (mutates transcript after the fork point).
    let dory_proof = <DoryCommitmentScheme as CommitmentScheme>::prove(
        &prover_setup,
        &poly,
        &opening_point,
        Some(hint),
        &mut stage8_transcript,
    );
    let ark_dory_proof = ArkDoryProof::from(dory_proof.clone());

    // Commitments map for recursion (Stage8 snapshot → combine).
    let mut commitments: HashMap<
        CommittedPolynomial,
        <DoryCommitmentScheme as CommitmentScheme>::Commitment,
    > = HashMap::new();
    commitments.insert(CommittedPolynomial::RdInc, commitment);

    // Minimal Stage8 snapshot.
    let stage8_snapshot = DoryOpeningSnapshot {
        pre_opening_proof_transcript: stage8_pre_transcript.clone(),
        opening_point: opening_point.clone(),
        polynomial_claims: polynomial_claims.clone(),
        gamma_powers: gamma_powers.clone(),
        joint_claim,
    };

    // Hyrax setup (upper bound is fine).
    let hyrax_prover_setup =
        <HyraxPCS as CommitmentScheme>::setup_prover(MAX_RECURSION_DENSE_NUM_VARS);

    // Produce recursion proof from Stage8 artifacts.
    let mut recursion_transcript: Blake2bTranscript = Transcript::new(b"recursion");
    let input = RecursionInput::<Fr, DoryCommitmentScheme, Blake2bTranscript> {
        joint_opening_proof: &dory_proof,
        stage8_snapshot,
        verifier_setup: &verifier_setup,
        commitments: &commitments,
    };

    let (recursion_proof, _metadata, pairing_boundary, stage8_combine_hint_fq12_opt) =
        RecursionProver::<Fq>::prove::<Fr, DoryCommitmentScheme, Blake2bTranscript>(
            &mut recursion_transcript,
            &hyrax_prover_setup,
            input,
        )
        .expect("recursion proving must succeed");

    let stage8_combine_hint_fq12 =
        stage8_combine_hint_fq12_opt.expect("recursion prover should emit stage8 combine hint");
    let joint_commitment = <DoryCommitmentScheme as RecursionExt<Fr>>::combine_with_hint_fq12(
        &stage8_combine_hint_fq12,
    );

    // Deterministic combine ordering (BTreeMap order).
    let rlc_map = compute_rlc_coefficients(&gamma_powers, polynomial_claims);
    let mut combine_coeffs = Vec::with_capacity(rlc_map.len());
    let mut combine_commitments = Vec::with_capacity(rlc_map.len());
    for (poly_id, coeff) in rlc_map.into_iter() {
        combine_coeffs.push(coeff);
        combine_commitments.push(*commitments.get(&poly_id).expect("missing commitment"));
    }

    RecursionFixture {
        dory_proof,
        ark_dory_proof,
        verifier_setup,
        stage8_pre_transcript,
        opening_point,
        joint_claim,
        combine_commitments,
        combine_coeffs,
        joint_commitment,
        joint_commitment_fq12: stage8_combine_hint_fq12,
        recursion_proof,
        pairing_boundary,
        hyrax_prover_setup,
    }
}

fn verify_with_input(fixture: &RecursionFixture, verifier_input: RecursionVerifierInput) -> bool {
    let verifier = RecursionVerifier::<Fq>::new(verifier_input);
    let mut transcript: Blake2bTranscript = Transcript::new(b"recursion");
    let hyrax_verifier_setup =
        <HyraxPCS as CommitmentScheme>::setup_verifier(&fixture.hyrax_prover_setup);
    verifier
        .verify::<Blake2bTranscript, HyraxPCS>(
            &fixture.recursion_proof,
            &mut transcript,
            &fixture.recursion_proof.dense_commitment,
            &hyrax_verifier_setup,
        )
        .unwrap_or_default()
}

#[test]
#[serial]
fn wiring_rejects_tampered_pairing_boundary_rhs() {
    let fixture = build_fixture();

    // Build symbolic AST (verifier side).
    let mut sym_transcript = fixture.stage8_pre_transcript.clone();
    let ast = <DoryCommitmentScheme as RecursionExt<Fr>>::build_symbolic_ast(
        &fixture.dory_proof,
        &fixture.verifier_setup,
        &mut sym_transcript,
        &fixture.opening_point,
        &fixture.joint_claim,
        &fixture.joint_commitment,
    )
    .expect("symbolic AST reconstruction must succeed");

    // Regression check: symbolic verification transcript must match replay transcript.
    //
    // This ensures the recursion verifier can reuse the symbolic transcript state and skip
    // `replay_opening_proof_transcript()` in release builds.
    {
        let mut replay = fixture.stage8_pre_transcript.clone();
        <DoryCommitmentScheme as RecursionExt<Fr>>::replay_opening_proof_transcript(
            &fixture.dory_proof,
            &mut replay,
        )
        .expect("Stage 8 transcript replay must succeed");
        assert_eq!(
            sym_transcript.state, replay.state,
            "symbolic transcript state must match replay transcript state"
        );
    }

    let derived = derive_plan_with_hints(
        &ast,
        &fixture.ark_dory_proof,
        &fixture.verifier_setup,
        fixture.joint_commitment,
        &fixture.combine_commitments,
        &fixture.combine_coeffs,
        fixture.pairing_boundary.clone(),
        fixture.joint_commitment_fq12,
    )
    .expect("instance plan derivation must succeed");

    // Sanity: the untampered proof verifies (show error on failure).
    {
        let verifier = RecursionVerifier::<Fq>::new(derived.verifier_input.clone());
        let mut transcript: Blake2bTranscript = Transcript::new(b"recursion");
        let hyrax_verifier_setup =
            <HyraxPCS as CommitmentScheme>::setup_verifier(&fixture.hyrax_prover_setup);
        let res = verifier.verify::<Blake2bTranscript, HyraxPCS>(
            &fixture.recursion_proof,
            &mut transcript,
            &fixture.recursion_proof.dense_commitment,
            &hyrax_verifier_setup,
        );
        assert!(matches!(res, Ok(true)), "untampered verify failed: {res:?}");
    }

    // Tamper with the externally visible pairing RHS (payload-style attack).
    let mut bad_input = derived.verifier_input;
    bad_input.pairing_boundary.rhs += Fq12::one();

    assert!(
        !verify_with_input(&fixture, bad_input),
        "wiring/boundary constraints must bind pairing boundary RHS"
    );
}

#[test]
#[serial]
fn wiring_rejects_tampered_pairing_boundary_points() {
    let fixture = build_fixture();

    // Build symbolic AST (verifier side).
    let mut sym_transcript = fixture.stage8_pre_transcript.clone();
    let ast = <DoryCommitmentScheme as RecursionExt<Fr>>::build_symbolic_ast(
        &fixture.dory_proof,
        &fixture.verifier_setup,
        &mut sym_transcript,
        &fixture.opening_point,
        &fixture.joint_claim,
        &fixture.joint_commitment,
    )
    .expect("symbolic AST reconstruction must succeed");

    let derived = derive_plan_with_hints(
        &ast,
        &fixture.ark_dory_proof,
        &fixture.verifier_setup,
        fixture.joint_commitment,
        &fixture.combine_commitments,
        &fixture.combine_coeffs,
        fixture.pairing_boundary.clone(),
        fixture.joint_commitment_fq12,
    )
    .expect("instance plan derivation must succeed");

    // Tamper with pairing boundary points (must be rejected by wiring/boundary constraints).
    let mut bad_input = derived.verifier_input;
    {
        let orig = bad_input.pairing_boundary.p1_g1;
        let id = G1Affine::identity();
        bad_input.pairing_boundary.p1_g1 = if orig == id {
            bad_input.pairing_boundary.p2_g1
        } else {
            id
        };
        assert_ne!(
            bad_input.pairing_boundary.p1_g1, orig,
            "test must change pairing_boundary.p1_g1"
        );
    }
    {
        let orig = bad_input.pairing_boundary.p1_g2;
        let id = G2Affine::identity();
        bad_input.pairing_boundary.p1_g2 = if orig == id {
            bad_input.pairing_boundary.p2_g2
        } else {
            id
        };
        assert_ne!(
            bad_input.pairing_boundary.p1_g2, orig,
            "test must change pairing_boundary.p1_g2"
        );
    }

    assert!(
        !verify_with_input(&fixture, bad_input),
        "tampered pairing boundary points must not verify"
    );
}

#[test]
#[serial]
fn wiring_rejects_tampered_gt_exp_base_input() {
    let fixture = build_fixture();

    // Build symbolic AST (verifier side).
    let mut sym_transcript = fixture.stage8_pre_transcript.clone();
    let ast = <DoryCommitmentScheme as RecursionExt<Fr>>::build_symbolic_ast(
        &fixture.dory_proof,
        &fixture.verifier_setup,
        &mut sym_transcript,
        &fixture.opening_point,
        &fixture.joint_claim,
        &fixture.joint_commitment,
    )
    .expect("symbolic AST reconstruction must succeed");

    let derived = derive_plan_with_hints(
        &ast,
        &fixture.ark_dory_proof,
        &fixture.verifier_setup,
        fixture.joint_commitment,
        &fixture.combine_commitments,
        &fixture.combine_coeffs,
        fixture.pairing_boundary.clone(),
        fixture.joint_commitment_fq12,
    )
    .expect("instance plan derivation must succeed");

    // Tamper with a GTExp base boundary input (must be rejected by wiring/base binding).
    let mut bad_input = derived.verifier_input;
    let mut tampered = false;
    for b in bad_input.gt_exp_base_inputs.iter_mut() {
        if let Some(v) = b.as_mut() {
            *v += Fq12::one();
            tampered = true;
            break;
        }
    }
    assert!(
        tampered,
        "test requires at least one GTExp base boundary input (e.g., combine commitments)"
    );

    assert!(
        !verify_with_input(&fixture, bad_input),
        "tampered GTExp base inputs must not verify"
    );
}
