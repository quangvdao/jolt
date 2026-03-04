use super::commitment_scheme::JoltHachiCommitmentScheme;
use super::wrappers::{jolt_to_hachi, Fp128};
use crate::field::fp128::JoltFp128;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use crate::poly::one_hot_polynomial::OneHotPolynomial;
use crate::poly::opening_proof::BatchPolynomialSource;
use crate::transcripts::{Blake2bTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;
use hachi_pcs::protocol::commitment::CommitmentConfig;
use hachi_pcs::protocol::SmallTestCommitmentConfig;
use hachi_pcs::FromSmallInt;

type Cfg = SmallTestCommitmentConfig;
type Scheme = JoltHachiCommitmentScheme<{ Cfg::D }, Cfg>;

struct UnusedSource;

impl BatchPolynomialSource<JoltFp128> for UnusedSource {
    fn build_joint_polynomial(&self, _coeffs: &[JoltFp128]) -> MultilinearPolynomial<JoltFp128> {
        panic!("unused batch polynomial source in Hachi tests")
    }
}

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
    let layout = Cfg::commitment_layout(16).unwrap();
    let alpha = Cfg::D.trailing_zeros() as usize;
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let len = 1usize << num_vars;

    let evals: Vec<JoltFp128> = (0..len).map(|i| JoltFp128::from_u64(i as u64)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(evals));

    let setup = Scheme::setup_prover(num_vars);
    let pcs = Scheme::default();

    let (commitment1, _hint1) = pcs.commit(&poly, &setup);
    let (commitment2, _hint2) = pcs.commit(&poly, &setup);

    assert_eq!(commitment1, commitment2, "deterministic commitment");
}

#[test]
fn hachi_packed_poly_batch_roundtrip() {
    let num_polys: usize = 2;
    let selector_bits = num_polys.next_power_of_two().trailing_zeros() as usize;
    let packed_num_vars = 16 + selector_bits;
    let layout = Cfg::commitment_layout(packed_num_vars).unwrap();
    let alpha = Cfg::D.trailing_zeros() as usize;
    let individual_num_vars = layout.m_vars + layout.r_vars + alpha - selector_bits;
    let len = 1usize << individual_num_vars;

    let poly1 = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
        (0..len).map(|i| JoltFp128::from_u64(i as u64)).collect(),
    ));
    let poly2 = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
        (0..len)
            .map(|i| JoltFp128::from_u64((i * 3 + 7) as u64))
            .collect(),
    ));

    let setup = Scheme::setup_prover(packed_num_vars);
    let pcs = Scheme::default();

    let polys: Vec<&MultilinearPolynomial<JoltFp128>> = vec![&poly1, &poly2];
    let (commitments, _hint) = pcs.batch_commit(&polys, &setup);
    assert_eq!(
        commitments.len(),
        1,
        "packed poly produces single commitment"
    );
}

#[test]
fn hachi_batch_verify_rejects_truncated_individual_commitments() {
    let num_polys: usize = 2;
    let selector_bits = num_polys.next_power_of_two().trailing_zeros() as usize;
    let packed_num_vars = 16 + selector_bits;
    let layout = Cfg::commitment_layout(packed_num_vars).unwrap();
    let alpha = Cfg::D.trailing_zeros() as usize;
    let individual_num_vars = layout.m_vars + layout.r_vars + alpha - selector_bits;
    let len = 1usize << individual_num_vars;

    let dense_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
        (0..len)
            .map(|i| JoltFp128::from_u64((i * 5 + 3) as u64))
            .collect(),
    ));
    let onehot_k = 2usize;
    let onehot_t = len / onehot_k;
    let onehot_indices: Vec<Option<u8>> = (0..onehot_t)
        .map(|i| {
            if i % 2 == 0 {
                Some((i % onehot_k) as u8)
            } else {
                None
            }
        })
        .collect();
    let onehot_poly = MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
        onehot_indices,
        onehot_k,
        onehot_t,
    ));

    let advice_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
        (0..len)
            .map(|i| JoltFp128::from_u64((i * 9 + 1) as u64))
            .collect(),
    ));

    let setup = Scheme::setup_prover(packed_num_vars);
    let verifier_setup = Scheme::setup_verifier(&setup);
    let pcs = Scheme::default();
    let main_polys: Vec<&MultilinearPolynomial<JoltFp128>> = vec![&dense_poly, &onehot_poly];
    let (commitments, batch_hint) = pcs.batch_commit(&main_polys, &setup);
    let (advice_commitment, advice_hint) = pcs.commit(&advice_poly, &setup);

    let opening_point: Vec<JoltFp128> = (0..individual_num_vars)
        .map(|i| JoltFp128::from_u64((i as u64) + 7))
        .collect();
    let dense_claim = dense_poly.evaluate(&opening_point);
    let onehot_claim = onehot_poly.evaluate(&opening_point);
    let advice_claim = advice_poly.evaluate(&opening_point);
    let claims = vec![dense_claim, onehot_claim, advice_claim];

    let prove_commitments = vec![&commitments[0], &advice_commitment];
    let mut prove_transcript = Blake2bTranscript::new(b"hachi_batch_truncated");
    let proof = pcs.batch_prove(
        &setup,
        &UnusedSource,
        batch_hint,
        vec![advice_hint],
        &prove_commitments,
        &opening_point,
        &claims,
        &[],
        &mut prove_transcript,
    );

    let verify_commitments = vec![&commitments[0]];
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
    let num_polys: usize = 2;
    let selector_bits = num_polys.next_power_of_two().trailing_zeros() as usize;
    let packed_num_vars = 16 + selector_bits;
    let layout = Cfg::commitment_layout(packed_num_vars).unwrap();
    let alpha = Cfg::D.trailing_zeros() as usize;
    let individual_num_vars = layout.m_vars + layout.r_vars + alpha - selector_bits;
    let len = 1usize << individual_num_vars;

    let poly1 = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
        (0..len).map(|i| JoltFp128::from_u64(i as u64)).collect(),
    ));
    let poly2 = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
        (0..len)
            .map(|i| JoltFp128::from_u64((i * 3 + 7) as u64))
            .collect(),
    ));

    let setup = Scheme::setup_prover(packed_num_vars);
    let verifier_setup = Scheme::setup_verifier(&setup);
    let pcs = Scheme::default();
    let main_polys: Vec<&MultilinearPolynomial<JoltFp128>> = vec![&poly1, &poly2];
    let (commitments, batch_hint) = pcs.batch_commit(&main_polys, &setup);

    let opening_point: Vec<JoltFp128> = (0..individual_num_vars)
        .map(|i| JoltFp128::from_u64((i as u64) + 5))
        .collect();
    let claims = vec![
        poly1.evaluate(&opening_point),
        poly2.evaluate(&opening_point),
    ];
    let commitment_refs = vec![&commitments[0]];

    let mut prove_transcript = Blake2bTranscript::new(b"hachi_batch_bad_num_packed");
    let mut proof = pcs.batch_prove(
        &setup,
        &UnusedSource,
        batch_hint,
        vec![],
        &commitment_refs,
        &opening_point,
        &claims,
        &[],
        &mut prove_transcript,
    );
    proof.num_packed_polys = 0;

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
