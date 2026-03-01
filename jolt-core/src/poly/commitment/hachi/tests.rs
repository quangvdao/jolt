use super::commitment_scheme::JoltHachiCommitmentScheme;
use super::wrappers::{jolt_to_hachi, Fp128};
use crate::field::fp128::JoltFp128;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use hachi_pcs::protocol::commitment::CommitmentConfig;
use hachi_pcs::protocol::SmallTestCommitmentConfig;
use hachi_pcs::{FromSmallInt, Polynomial};

type Cfg = SmallTestCommitmentConfig;
type Scheme = JoltHachiCommitmentScheme<{ Cfg::D }, Cfg>;

#[test]
fn polynomial_adapter_preserves_coefficients() {
    let evals: Vec<JoltFp128> = (0..16).map(|i| JoltFp128::from_u64(i as u64)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(evals.clone()));

    let hachi_poly = super::commitment_scheme::to_hachi_poly(&poly);

    assert_eq!(hachi_poly.num_vars(), 4);

    let hachi_coeffs = hachi_poly.coeffs();
    assert_eq!(hachi_coeffs.len(), 16);
    for (i, (&jolt_val, hachi_val)) in evals.iter().zip(hachi_coeffs.iter()).enumerate() {
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

    let hachi_poly = super::commitment_scheme::to_hachi_poly(&poly);
    assert_eq!(hachi_poly.num_vars(), 3);

    for (i, &coeff) in u8_coeffs.iter().enumerate() {
        let expected = Fp128::from_u64(coeff as u64);
        assert_eq!(hachi_poly.coeffs()[i], expected);
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

// NOTE: prove/verify tests are disabled because Hachi's `balanced_decompose_pow2` asserts
// `levels * log_basis <= 128` in the ring-switch path, but `r_decomp_levels` returns 33
// for Fp128 fields (33 * 4 = 132 > 128). This is a known Hachi limitation that needs to
// be fixed upstream before end-to-end prove/verify works with 128-bit primes.
