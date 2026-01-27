//! End-to-end tests for the recursion SNARK using the unified API

use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{wrappers::ArkDoryProof, DoryCommitmentScheme, DoryContext, DoryGlobals},
            hyrax::Hyrax,
        },
        dense_mlpoly::DensePolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::{Blake2bTranscript, Transcript},
    zkvm::proof_serialization::PairingBoundary,
    zkvm::recursion::{
        ConstraintType, RecursionProof, RecursionProver, RecursionVerifier, RecursionVerifierInput,
        WiringPlan,
    },
};
use ark_bn254::{Fq, Fq12, Fr, G1Affine, G2Affine};
use ark_ff::{UniformRand, Zero};
use ark_grumpkin::Projective as GrumpkinProjective;
use ark_std::test_rng;
use serial_test::serial;

#[test]
#[serial]
fn test_recursion_snark_e2e_with_dory() {
    // Initialize test environment
    DoryGlobals::reset();
    DoryGlobals::initialize_context(1 << 2, 1 << 2, DoryContext::Main, None);

    let mut rng = test_rng();

    // ============ CREATE A DORY PROOF TO VERIFY ============
    // Create test polynomial
    let num_vars = 4;
    let poly_coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly_coefficients));

    // Setup Dory
    let prover_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover(num_vars);
    let verifier_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_verifier(&prover_setup);

    // Commit to polynomial
    let (commitment, hint) =
        <DoryCommitmentScheme as CommitmentScheme>::commit(&poly, &prover_setup);

    // Create evaluation point using transcript challenges
    let mut point_transcript: Blake2bTranscript = Transcript::new(b"test_point");
    let point_challenges: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
        .map(|_| point_transcript.challenge_scalar_optimized::<Fr>())
        .collect();

    // Compute evaluation
    let evaluation = PolynomialEvaluation::evaluate(&poly, &point_challenges);

    // Generate Dory proof
    let mut prover_transcript: Blake2bTranscript = Transcript::new(b"dory_test_proof");
    let opening_proof = <DoryCommitmentScheme as CommitmentScheme>::prove(
        &prover_setup,
        &poly,
        &point_challenges,
        Some(hint),
        &mut prover_transcript,
    );

    // Convert to ArkDoryProof
    let ark_proof = ArkDoryProof::from(opening_proof);
    let ark_commitment = commitment;

    let mut prover_transcript = Blake2bTranscript::new(b"recursion_snark");

    // Create prover using Dory witness generation
    let mut witness_transcript: Blake2bTranscript = Transcript::new(b"dory_test_proof");

    let mut prover = RecursionProver::<Fq>::new_from_dory_proof(
        &ark_proof,
        &verifier_setup,
        &mut witness_transcript,
        &point_challenges,
        &evaluation,
        &ark_commitment,
    )
    .expect("Failed to create recursion prover");

    // Extract constraint information before moving prover
    let num_constraints = prover.constraint_system.num_constraints();
    let num_vars = prover.constraint_system.num_vars();
    let num_s_vars = prover.constraint_system.num_s_vars();
    let num_constraint_vars = prover.constraint_system.num_constraint_vars();
    let num_constraints_padded = prover.constraint_system.num_constraints_padded();

    // Extract constraint types for verification
    let constraint_types: Vec<ConstraintType> = prover.constraint_system.constraint_types.clone();

    // Determine dense_num_vars for Hyrax setup without extracting dense evals.
    let enable_gt_fused_end_to_end = std::env::var("JOLT_RECURSION_ENABLE_GT_FUSED_END_TO_END")
        .ok()
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(false);
    let dense_num_vars = if enable_gt_fused_end_to_end {
        crate::zkvm::recursion::prefix_packing::PrefixPackingLayout::from_constraint_types_gt_fused(
            &constraint_types,
        )
        .num_dense_vars
    } else {
        crate::zkvm::recursion::prefix_packing::PrefixPackingLayout::from_constraint_types(
            &constraint_types,
        )
        .num_dense_vars
    };

    let num_g1_add = constraint_types
        .iter()
        .filter(|t| matches!(t, ConstraintType::G1Add))
        .count();
    let num_g2_add = constraint_types
        .iter()
        .filter(|t| matches!(t, ConstraintType::G2Add))
        .count();

    // Allow selectively disabling constraint families when debugging recursion composition.
    // By default, we expect these constraint families to be present.
    let env_flag_default = |name: &str, default: bool| -> bool {
        std::env::var(name)
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(default)
    };
    let enable_g1_add = env_flag_default("JOLT_RECURSION_ENABLE_G1_ADD", true);
    let enable_g2_add = env_flag_default("JOLT_RECURSION_ENABLE_G2_ADD", true);

    if enable_g1_add {
        assert!(
            num_g1_add > 0,
            "Expected at least one G1Add constraint in recursion constraint system"
        );
    }
    if enable_g2_add {
        assert!(
            num_g2_add > 0,
            "Expected at least one G2Add constraint in recursion constraint system"
        );
    }

    // (Verifier input will be constructed from prover-produced metadata later.)

    // Setup Hyrax PCS (which works with Fq)
    const RATIO: usize = 1;
    type HyraxPCS = Hyrax<RATIO, GrumpkinProjective>;

    let hyrax_prover_setup = <HyraxPCS as CommitmentScheme>::setup_prover(dense_num_vars);

    // Run the prover phases explicitly (commit → sumchecks → opening).
    let poly_commit = prover
        .poly_commit::<Blake2bTranscript>(&mut prover_transcript, &hyrax_prover_setup)
        .expect("poly_commit failed");
    let sumchecks = prover
        .prove_sumchecks::<Blake2bTranscript>(&mut prover_transcript, &poly_commit.metadata)
        .expect("prove_sumchecks failed");
    let (opening_proof, opening_claims) =
        RecursionProver::<Fq>::poly_opening::<Blake2bTranscript, HyraxPCS>(
            &mut prover_transcript,
            &hyrax_prover_setup,
            sumchecks.accumulator,
            poly_commit.dense_mlpoly,
        )
        .expect("poly_opening failed");

    let dense_commitment = poly_commit.dense_commitment.clone();
    let recursion_constraint_metadata = poly_commit.metadata.clone();
    let recursion_proof = RecursionProof::<Fq, Blake2bTranscript, HyraxPCS> {
        stage1_proof: sumchecks.stage1_proof,
        stage2_proof: sumchecks.stage2_proof,
        stage3_packed_eval: sumchecks.stage3_packed_eval,
        opening_proof,
        opening_claims,
        dense_commitment: dense_commitment.clone(),
    };

    // ============ VERIFY THE RECURSION PROOF ============

    // Create verifier input
    // Prefer the metadata produced by `poly_commit`, to ensure perfect alignment.
    let verifier_input = RecursionVerifierInput {
        constraint_types: recursion_constraint_metadata.constraint_types,
        num_vars,
        num_constraint_vars,
        num_s_vars,
        num_constraints,
        num_constraints_padded,
        gt_exp_public_inputs: recursion_constraint_metadata.gt_exp_public_inputs,
        g1_scalar_mul_public_inputs: recursion_constraint_metadata.g1_scalar_mul_public_inputs,
        g2_scalar_mul_public_inputs: recursion_constraint_metadata.g2_scalar_mul_public_inputs,
        wiring: WiringPlan::default(),
        pairing_boundary: PairingBoundary {
            p1_g1: G1Affine::identity(),
            p1_g2: G2Affine::identity(),
            p2_g1: G1Affine::identity(),
            p2_g2: G2Affine::identity(),
            p3_g1: G1Affine::identity(),
            p3_g2: G2Affine::identity(),
            rhs: Fq12::zero(),
        },
        joint_commitment: Fq12::zero(),
    };

    // Create verifier
    let verifier = RecursionVerifier::<Fq>::new(verifier_input);

    // Create transcript for verification
    let mut verifier_transcript = Blake2bTranscript::new(b"recursion_snark");

    // Setup Hyrax verifier
    let hyrax_verifier_setup = <HyraxPCS as CommitmentScheme>::setup_verifier(&hyrax_prover_setup);

    // Verify the proof
    let verification_result = verifier
        .verify::<Blake2bTranscript, HyraxPCS>(
            &recursion_proof,
            &mut verifier_transcript,
            &dense_commitment,
            &hyrax_verifier_setup,
        )
        .expect("Verification should not fail");

    assert!(verification_result, "Recursion proof verification failed!");
}
