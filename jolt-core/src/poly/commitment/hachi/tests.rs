use super::commitment_scheme::JoltHachiCommitmentScheme;
use super::wrappers::{jolt_to_hachi, ArkBridge, Fp128};
use crate::field::fp128::JoltFp128;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::transcripts::{Blake2bTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;
use hachi_pcs::protocol::commitment::CommitmentConfig;
use hachi_pcs::protocol::proof::{HachiProof, PackedDigits};
use hachi_pcs::protocol::SmallTestCommitmentConfig;
use hachi_pcs::FromSmallInt;

type Cfg = SmallTestCommitmentConfig;
type Scheme = JoltHachiCommitmentScheme<{ Cfg::D }, Cfg>;

#[test]
fn polynomial_adapter_preserves_coefficients() {
    let evals: Vec<JoltFp128> = (0..16).map(|i| JoltFp128::from_u64(i as u64)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(evals.clone()));

    let ring_coeffs = super::commitment_scheme::poly_to_ring_coeffs::<{ Cfg::D }>(&poly);

    assert_eq!(ring_coeffs.len() * Cfg::D, 16);
    for (i, (&jolt_val, hachi_val)) in evals
        .iter()
        .zip(
            ring_coeffs
                .iter()
                .flat_map(|r| r.coefficients().iter())
                .cloned()
                .collect::<Vec<_>>()
                .iter(),
        )
        .enumerate()
    {
        assert_eq!(
            jolt_to_hachi(&jolt_val),
            *hachi_val,
            "coefficient mismatch at index {i}"
        );
    }
}

#[test]
fn polynomial_adapter_compact_scalars() {
    use crate::poly::compact_polynomial::CompactPolynomial;

    let u8_coeffs: Vec<u8> = (0..8).collect();
    let poly = MultilinearPolynomial::U8Scalars(CompactPolynomial::from_coeffs(u8_coeffs.clone()));

    let ring_coeffs = super::commitment_scheme::poly_to_ring_coeffs::<{ Cfg::D }>(&poly);

    let field_coeffs: Vec<Fp128> = ring_coeffs
        .iter()
        .flat_map(|r| r.coefficients().iter())
        .cloned()
        .collect();
    for (i, &coeff) in u8_coeffs.iter().enumerate() {
        let expected = Fp128::from_u64(coeff as u64);
        assert_eq!(field_coeffs[i], expected);
    }
}

#[test]
fn commit_roundtrip() {
    // compute_advice_layout hardcodes 64-bit commit bounds (ceil(64/3) = 22
    // digits), which exceeds SmallTestCommitmentConfig's fixed B matrix width
    // for large polynomials. Use alpha+1 variables so reduced_vars=1, which
    // triggers the early-return path and produces a small enough layout.
    let alpha = Cfg::D.trailing_zeros() as usize;
    let num_vars = alpha + 1;
    let len = 1usize << num_vars;

    let evals: Vec<JoltFp128> = (0..len).map(|i| JoltFp128::from_u64(i as u64)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(evals));

    let setup = Scheme::setup_prover(num_vars.max(10));
    let pcs = Scheme::default();

    let (commitment1, _hint1) = pcs.commit(&poly, &setup);
    let (commitment2, _hint2) = pcs.commit(&poly, &setup);

    assert_eq!(commitment1, commitment2, "deterministic commitment");
}

#[test]
fn hachi_batch_verify_rejects_truncated_individual_commitments() {
    let setup = Scheme::setup_prover(16);
    let verifier_setup = Scheme::setup_verifier(&setup);
    let pcs = Scheme::default();
    let opening_point = vec![JoltFp128::from_u64(3), JoltFp128::from_u64(5)];

    let packed_poly_proof = ArkBridge(HachiProof {
        levels: vec![],
        final_w: PackedDigits::from_i8_digits(&[], 1),
    });
    let indiv_proof = ArkBridge(HachiProof {
        levels: vec![],
        final_w: PackedDigits::from_i8_digits(&[], 1),
    });
    let proof = super::commitment_scheme::HachiBatchedProof {
        packed_poly_proof,
        num_packed_polys: 2,
        log_k: 1,
        individual_proofs: vec![indiv_proof],
    };

    let commitment = <Scheme as CommitmentScheme>::Commitment::default();
    // 2 packed + 1 individual = 3 claims, but only 1 commitment (should be 2)
    let claims = vec![
        JoltFp128::from_u64(1),
        JoltFp128::from_u64(2),
        JoltFp128::from_u64(3),
    ];
    let verify_commitments = vec![&commitment];

    let mut verify_transcript = Blake2bTranscript::new(b"hachi_batch_truncated");
    let result = pcs.batch_verify(
        &proof,
        &verifier_setup,
        &mut verify_transcript,
        &opening_point,
        &verify_commitments,
        &claims,
        &[],
    );
    assert!(
        matches!(result, Err(ProofVerifyError::InvalidInputLength(_, _))),
        "expected InvalidInputLength for truncated commitments, got: {result:?}"
    );
}

#[test]
fn hachi_batch_verify_rejects_invalid_num_packed() {
    let setup = Scheme::setup_prover(16);
    let verifier_setup = Scheme::setup_verifier(&setup);
    let pcs = Scheme::default();
    let opening_point = vec![JoltFp128::from_u64(3), JoltFp128::from_u64(5)];

    let packed_poly_proof = ArkBridge(HachiProof {
        levels: vec![],
        final_w: PackedDigits::from_i8_digits(&[], 1),
    });
    let proof = super::commitment_scheme::HachiBatchedProof {
        packed_poly_proof,
        num_packed_polys: 0,
        log_k: 1,
        individual_proofs: vec![],
    };

    let commitment = <Scheme as CommitmentScheme>::Commitment::default();
    let claims = vec![JoltFp128::from_u64(1), JoltFp128::from_u64(2)];
    let commitment_refs = vec![&commitment];

    let mut verify_transcript = Blake2bTranscript::new(b"hachi_batch_bad_num_packed");
    let result = pcs.batch_verify(
        &proof,
        &verifier_setup,
        &mut verify_transcript,
        &opening_point,
        &commitment_refs,
        &claims,
        &[],
    );
    assert!(
        matches!(result, Err(ProofVerifyError::InvalidInputLength(_, _))),
        "expected InvalidInputLength for invalid num_packed_polys, got: {result:?}"
    );
}

#[test]
fn hachi_batch_verify_rejects_invalid_log_k() {
    let setup = Scheme::setup_prover(16);
    let verifier_setup = Scheme::setup_verifier(&setup);
    let pcs = Scheme::default();
    let opening_point = vec![JoltFp128::from_u64(3), JoltFp128::from_u64(5)];
    let claim = JoltFp128::from_u64(7);

    let packed_poly_proof = ArkBridge(HachiProof {
        levels: vec![],
        final_w: PackedDigits::from_i8_digits(&[], 1),
    });
    let proof = super::commitment_scheme::HachiBatchedProof {
        packed_poly_proof,
        num_packed_polys: 1,
        log_k: (opening_point.len() + 1) as u32,
        individual_proofs: vec![],
    };
    let commitment = <Scheme as CommitmentScheme>::Commitment::default();
    let claims = vec![claim];
    let commitment_refs = vec![&commitment];

    let mut verify_transcript = Blake2bTranscript::new(b"hachi_single_bad_log_k");
    let result = pcs.batch_verify(
        &proof,
        &verifier_setup,
        &mut verify_transcript,
        &opening_point,
        &commitment_refs,
        &claims,
        &[],
    );
    assert!(
        matches!(
            result,
            Err(ProofVerifyError::InvalidInputLength(expected, actual))
                if expected == opening_point.len() && actual == opening_point.len() + 1
        ),
        "expected InvalidInputLength for invalid log_k, got: {result:?}"
    );
}
