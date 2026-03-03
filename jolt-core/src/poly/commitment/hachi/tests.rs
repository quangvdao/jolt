use super::commitment_scheme::JoltHachiCommitmentScheme;
use super::wrappers::{jolt_to_hachi, Fp128};
use crate::field::fp128::JoltFp128;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use hachi_pcs::protocol::commitment::CommitmentConfig;
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

    let (commitments, _hint) = pcs.batch_commit(&[&poly1, &poly2], &setup);
    assert_eq!(
        commitments.len(),
        1,
        "packed poly produces single commitment"
    );
}
