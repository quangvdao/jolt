use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::{CommitmentScheme, RecursionExt},
            dory::{DoryCommitmentScheme, DoryContext, DoryGlobals},
        },
        dense_mlpoly::DensePolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::{Blake2bTranscript, Transcript},
};
use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_std::test_rng;
use serial_test::serial;

/// Stage 8 transcript invariant regression test.
///
/// In Stage 8 we must fork the transcript **after** sampling gamma (from claims) but **before**
/// `PCS::prove` mutates the main transcript. The fork is what recursion uses for `PCS::witness_gen`,
/// and `witness_gen` must bring that fork to the same final transcript state as the prover’s
/// transcript after `prove`.
#[test]
#[serial]
fn test_stage8_transcript_fork_matches_prover_after_witness_gen() {
    // Initialize Dory globals for this test (small context is sufficient).
    DoryGlobals::reset();
    DoryGlobals::initialize_context(1 << 2, 1 << 2, DoryContext::Main, None);

    let mut rng = test_rng();

    // Small test polynomial.
    let num_vars = 4;
    let coeffs: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs));

    // Setup + commit.
    let prover_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover(num_vars);
    let verifier_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_verifier(&prover_setup);
    let (commitment, hint) = <DoryCommitmentScheme as CommitmentScheme>::commit(&poly, &prover_setup);

    // Opening point + evaluation.
    let mut point_transcript: Blake2bTranscript = Transcript::new(b"pt");
    let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
        .map(|_| point_transcript.challenge_scalar_optimized::<Fr>())
        .collect();
    let evaluation = PolynomialEvaluation::evaluate(&poly, &point);

    // Stage 8 transcript prefix: append claims and sample gamma powers.
    let mut prover_transcript: Blake2bTranscript = Transcript::new(b"stage8");
    let claims = vec![evaluation];
    prover_transcript.append_scalars(&claims);
    let _gamma_powers: Vec<Fr> = prover_transcript.challenge_scalar_powers(claims.len());

    // Fork transcript after gamma sampling, before `prove`.
    let mut witness_gen_transcript = prover_transcript.clone();

    // `prove` mutates the main transcript.
    let proof = <DoryCommitmentScheme as CommitmentScheme>::prove(
        &prover_setup,
        &poly,
        &point,
        Some(hint),
        &mut prover_transcript,
    );
    let prover_final_state = prover_transcript.state;

    // `witness_gen` should bring the forked transcript to the same final state.
    let _ = <DoryCommitmentScheme as RecursionExt<Fr>>::witness_gen(
        &proof,
        &verifier_setup,
        &mut witness_gen_transcript,
        &point,
        &evaluation,
        &commitment,
    )
    .expect("witness_gen must succeed on the Stage 8 forked transcript");

    assert_eq!(
        witness_gen_transcript.state, prover_final_state,
        "witness_gen transcript state must match prover transcript after prove"
    );
}

