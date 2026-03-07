use super::commitment_scheme::JoltHachiCommitmentScheme;
use super::packed_layout::choose_packed_bit_layout;
use super::wrappers::{jolt_to_hachi, ArkBridge, Fp128};
use crate::field::fp128::JoltFp128;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{CommitmentScheme, PolynomialBatchSource};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::one_hot_polynomial::OneHotPolynomial;
use crate::poly::opening_proof::BatchPolynomialSource;
use crate::transcripts::{Blake2bTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;
use hachi_pcs::protocol::commitment::{optimal_m_r_split, CommitmentConfig};
use hachi_pcs::protocol::proof::{HachiProof, PackedDigits};
use hachi_pcs::protocol::SmallTestCommitmentConfig;
use hachi_pcs::FromSmallInt;

type Cfg = SmallTestCommitmentConfig;
type Scheme = JoltHachiCommitmentScheme<{ Cfg::D }, Cfg>;

#[derive(Clone)]
struct TestPackedSource {
    per_poly: Vec<Vec<Option<u8>>>,
    onehot_k: usize,
}

impl TestPackedSource {
    fn new(per_poly: Vec<Vec<Option<u8>>>, onehot_k: usize) -> Self {
        Self { per_poly, onehot_k }
    }

    fn num_cycles(&self) -> usize {
        self.per_poly[0].len()
    }
}

impl PolynomialBatchSource<JoltFp128> for TestPackedSource {
    fn num_polys(&self) -> usize {
        self.per_poly.len()
    }

    fn onehot_index(&self, cycle_idx: usize, poly_idx: usize) -> Option<u8> {
        self.per_poly[poly_idx][cycle_idx]
    }

    fn num_cycles(&self) -> Option<usize> {
        Some(self.num_cycles())
    }

    fn onehot_k(&self) -> Option<usize> {
        Some(self.onehot_k)
    }
}

impl BatchPolynomialSource<JoltFp128> for TestPackedSource {
    fn build_joint_polynomial(&self, _coeffs: &[JoltFp128]) -> MultilinearPolynomial<JoltFp128> {
        panic!("TestPackedSource does not build joint polynomials")
    }

    fn onehot_index(&self, cycle_idx: usize, poly_idx: usize) -> Option<u8> {
        self.per_poly[poly_idx][cycle_idx]
    }

    fn num_cycles(&self) -> Option<usize> {
        Some(self.num_cycles())
    }

    fn onehot_k(&self) -> Option<usize> {
        Some(self.onehot_k)
    }

    fn num_polys(&self) -> Option<usize> {
        Some(self.per_poly.len())
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
fn packed_layout_reorders_poly_bits_into_inner_prefix() {
    let log_k = Cfg::D.trailing_zeros() as usize;
    let layout = choose_packed_bit_layout::<{ Cfg::D }, Cfg>(log_k, 16, 6);
    let cycle_bits: Vec<usize> = (0..16).collect();
    let addr_bits: Vec<usize> = (100..100 + log_k).collect();
    let poly_bits: Vec<usize> = (200..206).collect();

    let reordered = layout.reorder_packed_point(&cycle_bits, &addr_bits, &poly_bits);
    let expected: Vec<usize> = addr_bits
        .iter()
        .copied()
        .chain(cycle_bits[..layout.cycle_inner_bits()].iter().copied())
        .chain(poly_bits[..layout.poly_inner_bits()].iter().copied())
        .chain(cycle_bits[layout.cycle_inner_bits()..].iter().copied())
        .chain(poly_bits[layout.poly_inner_bits()..].iter().copied())
        .collect();

    assert_eq!(reordered, expected, "packed point permutation mismatch");
    assert!(
        layout.poly_inner_bits() > 0,
        "test expects the packed policy to tile across at least one poly bit"
    );
    assert!(
        layout.cycle_inner_bits() < 16,
        "test expects the packed policy to split cycle bits across inner/outer"
    );
}

#[test]
fn packed_layout_preserves_hachi_optimal_total_split() {
    let alpha = Cfg::D.trailing_zeros() as usize;
    let log_k = alpha;

    for (log_t, log_packed) in [(8usize, 3usize), (16, 6), (21, 6)] {
        let layout = choose_packed_bit_layout::<{ Cfg::D }, Cfg>(log_k, log_t, log_packed);
        let reduced_vars = (log_k - alpha) + log_t + log_packed;
        let (expected_m_vars, expected_r_vars) = optimal_m_r_split::<Cfg>(reduced_vars);

        assert_eq!(
            layout.m_vars(),
            expected_m_vars,
            "packed layout changed total m_vars for log_t={log_t}, log_packed={log_packed}"
        );
        assert_eq!(
            layout.r_vars(),
            expected_r_vars,
            "packed layout changed total r_vars for log_t={log_t}, log_packed={log_packed}"
        );
    }
}

#[test]
fn packed_layout_block_ranges_match_positions() {
    let log_k = Cfg::D.trailing_zeros() as usize;
    let layout = choose_packed_bit_layout::<{ Cfg::D }, Cfg>(log_k, 8, 3);
    let num_cycles = 1usize << 8;
    let num_polys = 5usize;

    for cycle_idx in 0..num_cycles {
        for poly_idx in 0..num_polys {
            let hot_index = (cycle_idx + poly_idx) % Cfg::D;
            let packed = layout.locate(cycle_idx, poly_idx, hot_index);
            let range = layout.block_range(packed.block_idx, num_cycles, num_polys);
            assert!(
                (range.cycle_start..range.cycle_end).contains(&cycle_idx),
                "cycle {cycle_idx} missing from block range {range:?}"
            );
            assert!(
                (range.poly_start..range.poly_end).contains(&poly_idx),
                "poly {poly_idx} missing from block range {range:?}"
            );
            assert!(
                packed.pos_in_block < layout.block_len(),
                "pos_in_block {} exceeds block_len {block_len}",
                packed.pos_in_block,
                block_len = layout.block_len()
            );
        }
    }
}

#[test]
fn hachi_batch_roundtrip_with_packed_layout() {
    let log_k = Cfg::D.trailing_zeros() as usize;
    let num_cycles = 1usize << 3;
    let source = TestPackedSource::new(
        vec![
            (0u8..8).map(Some).collect(),
            (0u8..8).rev().map(Some).collect(),
            vec![1, 3, 5, 7, 1, 3, 5, 7].into_iter().map(Some).collect(),
        ],
        Cfg::D,
    );
    let log_packed = source.per_poly.len().next_power_of_two().trailing_zeros() as usize;
    let setup = Scheme::setup_prover_from_shape(
        num_cycles.trailing_zeros() as usize,
        log_k,
        Some(log_packed),
    );
    let verifier_setup = Scheme::setup_verifier(&setup);
    let pcs = Scheme::default();

    let (commitments, batch_hint) = pcs.batch_commit(&source, &setup);
    assert_eq!(
        commitments.len(),
        1,
        "packed Hachi should produce one commitment"
    );

    let opening_point: Vec<JoltFp128> = (0..(log_k + num_cycles.trailing_zeros() as usize))
        .map(|i| JoltFp128::from_u64((i + 2) as u64))
        .collect();

    let claims: Vec<JoltFp128> = source
        .per_poly
        .iter()
        .map(|indices| {
            OneHotPolynomial::<JoltFp128>::from_indices(indices.clone(), Cfg::D, num_cycles)
                .evaluate(&opening_point)
        })
        .collect();
    let commitment_refs = vec![&commitments[0]];

    let mut prove_transcript = Blake2bTranscript::new(b"hachi_batch_roundtrip");
    let proof = pcs.batch_prove(
        &setup,
        &source,
        batch_hint,
        vec![],
        &commitment_refs,
        &opening_point,
        &claims,
        &[],
        &mut prove_transcript,
    );

    let mut verify_transcript = Blake2bTranscript::new(b"hachi_batch_roundtrip");
    pcs.batch_verify(
        &proof,
        &verifier_setup,
        &mut verify_transcript,
        &opening_point,
        &commitment_refs,
        &claims,
        &[],
    )
    .expect("packed Hachi batch verify should succeed");
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
