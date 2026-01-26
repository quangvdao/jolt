use ark_bn254::Fr;
use ark_ff::UniformRand;
use jolt_recursion::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_recursion::poly::commitment::commitment_scheme::RecursionExt;
use jolt_recursion::poly::commitment::dory::{DoryCommitmentScheme, DoryContext, DoryGlobals};
use jolt_recursion::poly::dense_mlpoly::DensePolynomial;
use jolt_recursion::poly::multilinear_polynomial::MultilinearPolynomial;
use rand::thread_rng;
use serial_test::serial;

#[test]
#[serial]
fn combine_witness_matches_direct() {
    DoryGlobals::reset();

    let num_vars = 8;
    let num_coeffs = 1 << num_vars;
    let num_polys = 5;

    let _guard = DoryGlobals::initialize_context(1, num_coeffs, DoryContext::Main, None);

    let mut rng = thread_rng();

    // Generate random polynomials
    let polys: Vec<MultilinearPolynomial<Fr>> = (0..num_polys)
        .map(|_| {
            let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs))
        })
        .collect();

    let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);

    // Commit to each polynomial
    let commitments: Vec<_> = polys
        .iter()
        .map(|poly| {
            let (commitment, _) = DoryCommitmentScheme::commit(poly, &prover_setup);
            commitment
        })
        .collect();

    // Generate random coefficients
    let coeffs: Vec<Fr> = (0..num_polys).map(|_| Fr::rand(&mut rng)).collect();

    // Compute via direct method
    let direct_result = DoryCommitmentScheme::combine_commitments(&commitments, &coeffs);

    // Compute via witness generation
    let (witness, hint) = DoryCommitmentScheme::generate_combine_witness(&commitments, &coeffs);
    let witness_result = DoryCommitmentScheme::combine_with_hint(&hint);

    // Results should match
    assert_eq!(
        direct_result, witness_result,
        "combine_with_hint result should match combine_commitments"
    );

    // Verify witness structure: n exp witnesses, n-1 mul witnesses
    assert_eq!(
        witness.exp_witnesses.len(),
        num_polys,
        "should have n exponentiation witnesses"
    );
    assert_eq!(
        witness.mul_layers.iter().map(|l| l.len()).sum::<usize>(),
        num_polys - 1,
        "should have n-1 multiplication witnesses"
    );
}

#[test]
#[serial]
fn combine_single_commitment() {
    DoryGlobals::reset();

    let num_vars = 8;
    let num_coeffs = 1 << num_vars;

    let _guard = DoryGlobals::initialize_context(1, num_coeffs, DoryContext::Main, None);

    let mut rng = thread_rng();

    // Single polynomial
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
        (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect(),
    ));

    let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
    let (commitment, _) = DoryCommitmentScheme::commit(&poly, &prover_setup);

    let coeff = Fr::rand(&mut rng);

    // Direct computation
    let direct_result = DoryCommitmentScheme::combine_commitments(&[commitment], &[coeff]);

    // Via witness
    let (witness, hint) = DoryCommitmentScheme::generate_combine_witness(&[commitment], &[coeff]);
    let witness_result = DoryCommitmentScheme::combine_with_hint(&hint);

    assert_eq!(direct_result, witness_result);

    // Single commitment: 1 exp witness, 0 mul witnesses
    assert_eq!(witness.exp_witnesses.len(), 1);
    assert_eq!(witness.mul_layers.iter().map(|l| l.len()).sum::<usize>(), 0);

    // Result should equal the exp result directly
    assert_eq!(witness.exp_witnesses[0].result, hint.0);
}
