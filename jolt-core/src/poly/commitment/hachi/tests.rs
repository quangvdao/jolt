use std::{array::from_fn, collections::HashSet, time::Instant};

use super::commitment_scheme::{
    build_packed_poly, poly_to_ring_coeffs, summarize_block_occupancy, Fp128OneHot64Config,
    HachiBatchedProof, JoltHachiCommitmentScheme, JoltPackedPoly,
};
use super::packed_layout::{choose_packed_bit_layout, PackedBitLayout};
use super::wrappers::{jolt_to_hachi, ArkBridge, Fp128};
use crate::field::fp128::JoltFp128;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{CommitmentScheme, PolynomialBatchSource};
use crate::poly::compact_polynomial::CompactPolynomial;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::one_hot_polynomial::OneHotPolynomial;
use crate::poly::opening_proof::BatchPolynomialSource;
use crate::transcripts::{Blake2bTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;
use hachi_pcs::algebra::ring::sparse_challenge::SparseChallenge;
use hachi_pcs::algebra::ring::CyclotomicRing;
use hachi_pcs::protocol::commitment::utils::flat_matrix::FlatMatrix;
use hachi_pcs::protocol::commitment::{compute_num_digits, optimal_m_r_split, CommitmentConfig};
use hachi_pcs::protocol::proof::{HachiProof, HachiProofTail, PackedDigits};
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

fn build_test_packed_poly<const D: usize, C: CommitmentConfig>(
    source: &TestPackedSource,
) -> (JoltPackedPoly<D>, PackedBitLayout) {
    let log_k = source.onehot_k.trailing_zeros() as usize;
    let log_t = source.num_cycles().trailing_zeros() as usize;
    let log_packed = source.per_poly.len().next_power_of_two().trailing_zeros() as usize;
    let packed_layout = choose_packed_bit_layout::<D, C>(log_k, log_t, log_packed);
    let index_fn = |cycle_idx: usize, poly_idx: usize| {
        PolynomialBatchSource::onehot_index(source, cycle_idx, poly_idx)
    };
    let packed_poly = build_packed_poly::<D, _>(
        index_fn,
        source.num_cycles(),
        source.per_poly.len(),
        packed_layout,
        None,
    );
    (packed_poly, packed_layout)
}

fn make_test_a_matrix<const D: usize>(
    n_a: usize,
    block_len: usize,
    num_digits_commit: usize,
) -> FlatMatrix<Fp128> {
    let rows: Vec<Vec<CyclotomicRing<Fp128, D>>> = (0..n_a)
        .map(|row_idx| {
            (0..block_len * num_digits_commit)
                .map(|col_idx| {
                    CyclotomicRing::from_coefficients(from_fn(|coeff_idx| {
                        let value = (row_idx as u64 + 1) * 1_000_000
                            + (col_idx as u64 + 1) * 1_003
                            + coeff_idx as u64;
                        Fp128::from_u64(value)
                    }))
                })
                .collect()
        })
        .collect();
    FlatMatrix::from_ring_matrix(&rows)
}

fn make_test_challenges<const D: usize>(num_blocks: usize) -> Vec<SparseChallenge> {
    (0..num_blocks)
        .map(|block_idx| {
            if block_idx % 5 == 0 {
                SparseChallenge::zero()
            } else {
                SparseChallenge {
                    positions: vec![
                        (block_idx % D) as u32,
                        ((block_idx + 3) % D) as u32,
                        ((block_idx + 7) % D) as u32,
                    ],
                    coeffs: if block_idx % 2 == 0 {
                        vec![1, -1, 2]
                    } else {
                        vec![-1, 2, -1]
                    },
                }
            }
        })
        .collect()
}

#[test]
fn polynomial_adapter_preserves_coefficients() {
    let evals: Vec<JoltFp128> = (0..16).map(|i| JoltFp128::from_u64(i as u64)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(evals.clone()));

    let ring_coeffs = poly_to_ring_coeffs::<{ Cfg::D }>(&poly);

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
    let u8_coeffs: Vec<u8> = (0..8).collect();
    let poly = MultilinearPolynomial::U8Scalars(CompactPolynomial::from_coeffs(u8_coeffs.clone()));

    let ring_coeffs = poly_to_ring_coeffs::<{ Cfg::D }>(&poly);

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
fn packed_layout_reorders_lifted_coeff_bits_for_k16() {
    type FastCfg = Fp128OneHot64Config;

    let log_k = 4usize;
    let layout = choose_packed_bit_layout::<{ FastCfg::D }, FastCfg>(log_k, 8, 3);
    let cycle_bits: Vec<usize> = (0..8).collect();
    let addr_bits: Vec<usize> = (100..100 + log_k).collect();
    let poly_bits: Vec<usize> = (200..203).collect();

    let reordered = layout.reorder_packed_point(&cycle_bits, &addr_bits, &poly_bits);
    let expected: Vec<usize> = addr_bits
        .iter()
        .copied()
        .chain(cycle_bits[..layout.cycle_coeff_bits()].iter().copied())
        .chain(poly_bits[..layout.poly_coeff_bits()].iter().copied())
        .chain(
            cycle_bits
                [layout.cycle_coeff_bits()..layout.cycle_coeff_bits() + layout.cycle_inner_bits()]
                .iter()
                .copied(),
        )
        .chain(
            poly_bits
                [layout.poly_coeff_bits()..layout.poly_coeff_bits() + layout.poly_inner_bits()]
                .iter()
                .copied(),
        )
        .chain(
            cycle_bits[layout.cycle_coeff_bits() + layout.cycle_inner_bits()..]
                .iter()
                .copied(),
        )
        .chain(
            poly_bits[layout.poly_coeff_bits() + layout.poly_inner_bits()..]
                .iter()
                .copied(),
        )
        .collect();

    assert_eq!(
        reordered, expected,
        "packed point permutation mismatch for K=16"
    );
    assert_eq!(
        layout.cycle_coeff_bits() + layout.poly_coeff_bits(),
        FastCfg::D.trailing_zeros() as usize - log_k,
        "K=16 should lift the missing coefficient bits from cycle/poly axes"
    );
    assert_eq!(
        layout.max_coeffs_per_entry(),
        1usize << (FastCfg::D.trailing_zeros() as usize - log_k),
        "K=16 should allow D/K coefficients per packed entry"
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
fn packed_layout_live_entries_are_injective() {
    let source = TestPackedSource::new(
        vec![
            vec![
                Some(0),
                Some(15),
                None,
                Some(1),
                Some(14),
                Some(2),
                None,
                Some(13),
            ],
            vec![
                Some(15),
                None,
                Some(0),
                Some(12),
                Some(3),
                None,
                Some(11),
                Some(4),
            ],
            vec![
                Some(7),
                Some(8),
                Some(9),
                None,
                Some(10),
                Some(6),
                Some(5),
                Some(0),
            ],
            vec![
                None,
                Some(1),
                Some(2),
                Some(3),
                None,
                Some(4),
                Some(5),
                Some(6),
            ],
            vec![
                Some(15),
                Some(14),
                Some(13),
                Some(12),
                Some(11),
                Some(10),
                Some(9),
                Some(8),
            ],
        ],
        Cfg::D,
    );
    let log_k = Cfg::D.trailing_zeros() as usize;
    let log_packed = source.per_poly.len().next_power_of_two().trailing_zeros() as usize;
    let layout = choose_packed_bit_layout::<{ Cfg::D }, Cfg>(log_k, 3, log_packed);
    let mut seen = HashSet::new();

    for (poly_idx, indices) in source.per_poly.iter().enumerate() {
        for (cycle_idx, &maybe_hot_index) in indices.iter().enumerate() {
            let Some(hot_index) = maybe_hot_index else {
                continue;
            };
            let packed = layout.locate(cycle_idx, poly_idx, hot_index as usize);
            assert!(
                seen.insert((packed.block_idx, packed.pos_in_block)),
                "packed slot collision at cycle={cycle_idx}, poly={poly_idx}, hot_index={hot_index}"
            );
        }
    }
}

#[test]
fn packed_layout_k16_uses_multi_coeff_entries() {
    type FastCfg = Fp128OneHot64Config;

    let source = TestPackedSource::new(
        vec![
            (0u8..16).map(Some).collect(),
            (0u8..16).rev().map(Some).collect(),
        ],
        16,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ FastCfg::D }, FastCfg>(&source);
    let coeff_counts = packed_poly.entry_coeff_counts_for_test();
    let live_coeff_counts = packed_poly.block_live_coeff_counts_for_test();

    assert_eq!(
        packed_poly.max_coeffs_per_entry_for_test(),
        FastCfg::D / source.onehot_k,
        "K=16 should pack D/K logical chunks per ring entry"
    );
    assert!(
        coeff_counts.iter().any(|&count| count > 1),
        "K=16 packed layout should store multiple coefficients in at least one entry"
    );
    assert!(
        live_coeff_counts
            .iter()
            .any(|&count| count > packed_layout.block_len()),
        "K=16 packed blocks should contain more live coefficients than occupied entries"
    );
}

#[test]
fn packed_occupancy_summary_reports_percentiles_and_buckets() {
    let summary = summarize_block_occupancy(&[0u32, 16, 17, 32, 33, 48, 49, 64], 64);

    assert_eq!(summary.nonempty_blocks(), 7);
    assert_eq!(summary.bucket_counts(), [1, 1, 2, 2, 2]);
    assert!((summary.p50_ratio(64) - 0.5).abs() < f64::EPSILON);
    assert!((summary.p95_ratio(64) - 1.0).abs() < f64::EPSILON);
    assert!((summary.max_ratio(64) - 1.0).abs() < f64::EPSILON);
}

#[test]
fn packed_commit_inner_fast_matches_generic_and_reference_helpers() {
    let source = TestPackedSource::new(
        vec![
            (0u8..16).map(Some).collect(),
            (0u8..16).rev().map(Some).collect(),
            vec![1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15]
                .into_iter()
                .map(Some)
                .collect(),
            vec![
                Some(0),
                None,
                Some(2),
                Some(4),
                None,
                Some(6),
                Some(8),
                Some(10),
                None,
                Some(12),
                Some(14),
                Some(0),
                Some(1),
                None,
                Some(3),
                Some(5),
            ],
            vec![Some(15); 16],
        ],
        Cfg::D,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ Cfg::D }, Cfg>(&source);
    let block_len = packed_layout.block_len();
    let log_basis = Cfg::decomposition().log_basis;
    let num_digits_open = compute_num_digits(128, log_basis);
    let a_flat = make_test_a_matrix::<{ Cfg::D }>(1, block_len, 1);
    let a_view = a_flat.view::<{ Cfg::D }>();

    let generic = packed_poly.commit_inner_generic(&a_view, 1, 1, num_digits_open, log_basis);
    let fast = packed_poly.commit_inner_fast_singleton(&a_view, num_digits_open, log_basis);
    let small = packed_poly.commit_inner_small(&a_view, 1, 1, num_digits_open, log_basis);
    let tiled = packed_poly.commit_inner_tiled(&a_view, 1, 1, num_digits_open, log_basis);

    assert_eq!(fast, generic, "fast commit_inner must match generic");
    assert_eq!(small, generic, "small commit_inner must match generic");
    assert_eq!(tiled, generic, "tiled commit_inner must match generic");
}

#[test]
fn packed_commit_inner_generic_matches_reference_with_multiple_rows() {
    let source = TestPackedSource::new(
        vec![
            (0u8..8).map(Some).collect(),
            vec![1, 3, 5, 7, 1, 3, 5, 7].into_iter().map(Some).collect(),
            vec![
                Some(7),
                Some(6),
                None,
                Some(5),
                Some(4),
                None,
                Some(3),
                Some(2),
            ],
        ],
        Cfg::D,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ Cfg::D }, Cfg>(&source);
    let block_len = packed_layout.block_len();
    let log_basis = Cfg::decomposition().log_basis;
    let num_digits_open = compute_num_digits(128, log_basis);
    let a_flat = make_test_a_matrix::<{ Cfg::D }>(2, block_len, 2);
    let a_view = a_flat.view::<{ Cfg::D }>();

    let generic = packed_poly.commit_inner_generic(&a_view, 2, 2, num_digits_open, log_basis);
    let small = packed_poly.commit_inner_small(&a_view, 2, 2, num_digits_open, log_basis);
    let tiled = packed_poly.commit_inner_tiled(&a_view, 2, 2, num_digits_open, log_basis);

    assert_eq!(small, generic, "small commit_inner must match generic");
    assert_eq!(tiled, generic, "tiled commit_inner must match generic");
}

#[test]
fn packed_commit_inner_column_sweep_matches_generic_in_k256_singleton_regime() {
    type FastCfg = Fp128OneHot64Config;

    let source = TestPackedSource::new(
        vec![
            vec![
                Some(0),
                Some(255),
                Some(1),
                Some(254),
                Some(2),
                Some(253),
                Some(3),
                Some(252),
            ],
            vec![
                Some(255),
                Some(0),
                Some(254),
                Some(1),
                Some(253),
                Some(2),
                Some(252),
                Some(3),
            ],
            vec![
                Some(127),
                Some(128),
                Some(64),
                Some(192),
                Some(32),
                Some(224),
                Some(16),
                Some(240),
            ],
            vec![
                Some(5),
                None,
                Some(10),
                Some(15),
                None,
                Some(20),
                Some(25),
                Some(30),
            ],
            vec![
                Some(250),
                Some(200),
                Some(150),
                Some(100),
                Some(50),
                Some(0),
                Some(25),
                Some(75),
            ],
        ],
        256,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ FastCfg::D }, FastCfg>(&source);
    let log_basis = FastCfg::decomposition().log_basis;
    let num_digits_open = compute_num_digits(128, log_basis);
    let a_flat = make_test_a_matrix::<{ FastCfg::D }>(1, packed_layout.block_len(), 1);
    let a_view = a_flat.view::<{ FastCfg::D }>();

    let generic = packed_poly.commit_inner_generic(&a_view, 1, 1, num_digits_open, log_basis);
    let sweep = packed_poly.commit_inner_column_sweep(&a_view, 1, 1, num_digits_open, log_basis);

    assert_eq!(
        sweep, generic,
        "column-sweep commit_inner must match generic for K=256"
    );
}

#[test]
fn packed_commit_inner_column_sweep_matches_generic_in_d64_k16_regime() {
    type FastCfg = Fp128OneHot64Config;

    let source = TestPackedSource::new(
        vec![
            (0u8..16).map(Some).collect(),
            (0u8..16).rev().map(Some).collect(),
            vec![1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15]
                .into_iter()
                .map(Some)
                .collect(),
            vec![
                Some(0),
                None,
                Some(2),
                Some(4),
                None,
                Some(6),
                Some(8),
                Some(10),
                None,
                Some(12),
                Some(14),
                Some(0),
                Some(1),
                None,
                Some(3),
                Some(5),
            ],
        ],
        16,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ FastCfg::D }, FastCfg>(&source);
    let log_basis = FastCfg::decomposition().log_basis;
    let num_digits_open = compute_num_digits(128, log_basis);
    let a_flat = make_test_a_matrix::<{ FastCfg::D }>(2, packed_layout.block_len(), 2);
    let a_view = a_flat.view::<{ FastCfg::D }>();

    let generic = packed_poly.commit_inner_generic(&a_view, 2, 2, num_digits_open, log_basis);
    let sweep = packed_poly.commit_inner_column_sweep(&a_view, 2, 2, num_digits_open, log_basis);

    assert_eq!(
        sweep, generic,
        "column-sweep commit_inner must match generic for K=16"
    );
}

#[test]
fn packed_decompose_fold_fast_matches_generic_and_reference_helpers() {
    let source = TestPackedSource::new(
        vec![
            (0u8..16).map(Some).collect(),
            (0u8..16).rev().map(Some).collect(),
            vec![1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15]
                .into_iter()
                .map(Some)
                .collect(),
            vec![
                Some(0),
                None,
                Some(2),
                Some(4),
                None,
                Some(6),
                Some(8),
                Some(10),
                None,
                Some(12),
                Some(14),
                Some(0),
                Some(1),
                None,
                Some(3),
                Some(5),
            ],
            vec![Some(15); 16],
        ],
        Cfg::D,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ Cfg::D }, Cfg>(&source);
    let challenges = make_test_challenges::<{ Cfg::D }>(packed_layout.num_blocks());

    let generic = packed_poly.decompose_fold_generic(&challenges, packed_layout.block_len(), 1);
    let striped = packed_poly.decompose_fold_striped_delta1(&challenges, packed_layout.block_len());
    let fast = packed_poly.decompose_fold_fast_singleton(&challenges, packed_layout.block_len(), 1);
    let small = packed_poly.decompose_fold_small(&challenges, packed_layout.block_len(), 1);
    let large = packed_poly.decompose_fold_large(&challenges, packed_layout.block_len(), 1);

    assert_eq!(
        striped, generic,
        "striped decompose_fold must match generic"
    );
    assert_eq!(fast, generic, "fast decompose_fold must match generic");
    assert_eq!(small, generic, "small decompose_fold must match generic");
    assert_eq!(large, generic, "large decompose_fold must match generic");
}

#[test]
fn packed_fast_paths_match_generic_in_d64_regime() {
    type FastCfg = Fp128OneHot64Config;

    let source = TestPackedSource::new(
        vec![
            vec![
                Some(0),
                Some(63),
                Some(1),
                Some(62),
                Some(2),
                Some(61),
                Some(3),
                Some(60),
            ],
            vec![
                Some(63),
                Some(0),
                Some(62),
                Some(1),
                Some(61),
                Some(2),
                Some(60),
                Some(3),
            ],
            vec![
                Some(31),
                Some(32),
                Some(16),
                Some(48),
                Some(8),
                Some(56),
                Some(4),
                Some(60),
            ],
            vec![
                Some(5),
                None,
                Some(10),
                Some(15),
                None,
                Some(20),
                Some(25),
                Some(30),
            ],
            vec![
                Some(58),
                Some(48),
                Some(38),
                Some(28),
                Some(18),
                Some(8),
                Some(12),
                Some(22),
            ],
        ],
        FastCfg::D,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ FastCfg::D }, FastCfg>(&source);
    let log_basis = FastCfg::decomposition().log_basis;
    let num_digits_open = compute_num_digits(128, log_basis);
    let a_flat = make_test_a_matrix::<{ FastCfg::D }>(1, packed_layout.block_len(), 1);
    let a_view = a_flat.view::<{ FastCfg::D }>();
    let generic_commit =
        packed_poly.commit_inner_generic(&a_view, 1, 1, num_digits_open, log_basis);
    let fast_commit = packed_poly.commit_inner_fast_singleton(&a_view, num_digits_open, log_basis);
    let challenges = make_test_challenges::<{ FastCfg::D }>(packed_layout.num_blocks());
    let generic_fold =
        packed_poly.decompose_fold_generic(&challenges, packed_layout.block_len(), 1);
    let striped_fold =
        packed_poly.decompose_fold_striped_delta1(&challenges, packed_layout.block_len());
    let fast_fold =
        packed_poly.decompose_fold_fast_singleton(&challenges, packed_layout.block_len(), 1);

    assert_eq!(
        fast_commit, generic_commit,
        "D=64 fast commit_inner must match generic"
    );
    assert_eq!(
        striped_fold, generic_fold,
        "D=64 striped decompose_fold must match generic"
    );
    assert_eq!(
        fast_fold, generic_fold,
        "D=64 fast decompose_fold must match generic"
    );
}

#[test]
fn packed_fast_paths_match_generic_in_k256_singleton_regime() {
    type FastCfg = Fp128OneHot64Config;

    let source = TestPackedSource::new(
        vec![
            vec![
                Some(0),
                Some(255),
                Some(1),
                Some(254),
                Some(2),
                Some(253),
                Some(3),
                Some(252),
            ],
            vec![
                Some(255),
                Some(0),
                Some(254),
                Some(1),
                Some(253),
                Some(2),
                Some(252),
                Some(3),
            ],
            vec![
                Some(127),
                Some(128),
                Some(64),
                Some(192),
                Some(32),
                Some(224),
                Some(16),
                Some(240),
            ],
            vec![
                Some(5),
                None,
                Some(10),
                Some(15),
                None,
                Some(20),
                Some(25),
                Some(30),
            ],
            vec![
                Some(250),
                Some(200),
                Some(150),
                Some(100),
                Some(50),
                Some(0),
                Some(25),
                Some(75),
            ],
        ],
        256,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ FastCfg::D }, FastCfg>(&source);
    let log_basis = FastCfg::decomposition().log_basis;
    let num_digits_open = compute_num_digits(128, log_basis);
    let a_flat = make_test_a_matrix::<{ FastCfg::D }>(1, packed_layout.block_len(), 1);
    let a_view = a_flat.view::<{ FastCfg::D }>();
    let generic_commit =
        packed_poly.commit_inner_generic(&a_view, 1, 1, num_digits_open, log_basis);
    let fast_commit = packed_poly.commit_inner_fast_singleton(&a_view, num_digits_open, log_basis);
    let challenges = make_test_challenges::<{ FastCfg::D }>(packed_layout.num_blocks());
    let generic_fold =
        packed_poly.decompose_fold_generic(&challenges, packed_layout.block_len(), 1);
    let striped_fold =
        packed_poly.decompose_fold_striped_delta1(&challenges, packed_layout.block_len());
    let fast_fold =
        packed_poly.decompose_fold_fast_singleton(&challenges, packed_layout.block_len(), 1);

    assert_eq!(
        fast_commit, generic_commit,
        "K=256 fast commit_inner must match generic"
    );
    assert_eq!(
        striped_fold, generic_fold,
        "K=256 striped decompose_fold must match generic"
    );
    assert_eq!(
        fast_fold, generic_fold,
        "K=256 fast decompose_fold must match generic"
    );
}

#[test]
fn packed_commit_inner_generic_matches_multi_coeff_helpers_in_d64_k16_regime() {
    type FastCfg = Fp128OneHot64Config;

    let source = TestPackedSource::new(
        vec![
            (0u8..16).map(Some).collect(),
            (0u8..16).rev().map(Some).collect(),
            vec![1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15]
                .into_iter()
                .map(Some)
                .collect(),
            vec![
                Some(0),
                None,
                Some(2),
                Some(4),
                None,
                Some(6),
                Some(8),
                Some(10),
                None,
                Some(12),
                Some(14),
                Some(0),
                Some(1),
                None,
                Some(3),
                Some(5),
            ],
        ],
        16,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ FastCfg::D }, FastCfg>(&source);
    let log_basis = FastCfg::decomposition().log_basis;
    let num_digits_open = compute_num_digits(128, log_basis);
    let a_flat = make_test_a_matrix::<{ FastCfg::D }>(2, packed_layout.block_len(), 2);
    let a_view = a_flat.view::<{ FastCfg::D }>();

    let generic = packed_poly.commit_inner_generic(&a_view, 2, 2, num_digits_open, log_basis);
    let small = packed_poly.commit_inner_small(&a_view, 2, 2, num_digits_open, log_basis);
    let tiled = packed_poly.commit_inner_tiled(&a_view, 2, 2, num_digits_open, log_basis);

    assert_eq!(
        small, generic,
        "small commit_inner must match generic for K=16"
    );
    assert_eq!(
        tiled, generic,
        "tiled commit_inner must match generic for K=16"
    );
}

#[test]
fn packed_decompose_fold_generic_matches_multi_coeff_helpers_in_d64_k16_regime() {
    type FastCfg = Fp128OneHot64Config;

    let source = TestPackedSource::new(
        vec![
            (0u8..16).map(Some).collect(),
            (0u8..16).rev().map(Some).collect(),
            vec![1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15]
                .into_iter()
                .map(Some)
                .collect(),
            vec![
                Some(0),
                None,
                Some(2),
                Some(4),
                None,
                Some(6),
                Some(8),
                Some(10),
                None,
                Some(12),
                Some(14),
                Some(0),
                Some(1),
                None,
                Some(3),
                Some(5),
            ],
        ],
        16,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ FastCfg::D }, FastCfg>(&source);
    let challenges = make_test_challenges::<{ FastCfg::D }>(packed_layout.num_blocks());

    let generic = packed_poly.decompose_fold_generic(&challenges, packed_layout.block_len(), 1);
    let small = packed_poly.decompose_fold_small(&challenges, packed_layout.block_len(), 1);
    let large = packed_poly.decompose_fold_large(&challenges, packed_layout.block_len(), 1);

    assert_eq!(
        small, generic,
        "small decompose_fold must match generic for K=16"
    );
    assert_eq!(
        large, generic,
        "large decompose_fold must match generic for K=16"
    );
}

#[test]
#[ignore = "profiling helper for packed fast paths"]
fn packed_fast_path_benchmark_smoke() {
    type FastCfg = Fp128OneHot64Config;

    let source = TestPackedSource::new(
        (0..16)
            .map(|poly_idx| {
                (0..32)
                    .map(|cycle_idx| Some(((poly_idx * 17 + cycle_idx * 9) % FastCfg::D) as u8))
                    .collect()
            })
            .collect(),
        FastCfg::D,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ FastCfg::D }, FastCfg>(&source);
    let log_basis = FastCfg::decomposition().log_basis;
    let num_digits_open = compute_num_digits(128, log_basis);
    let a_flat = make_test_a_matrix::<{ FastCfg::D }>(1, packed_layout.block_len(), 1);
    let a_view = a_flat.view::<{ FastCfg::D }>();
    let challenges = make_test_challenges::<{ FastCfg::D }>(packed_layout.num_blocks());

    let generic_commit_start = Instant::now();
    let _ = packed_poly.commit_inner_generic(&a_view, 1, 1, num_digits_open, log_basis);
    let generic_commit = generic_commit_start.elapsed();

    let fast_commit_start = Instant::now();
    let _ = packed_poly.commit_inner_fast_singleton(&a_view, num_digits_open, log_basis);
    let fast_commit = fast_commit_start.elapsed();

    let generic_fold_start = Instant::now();
    let _ = packed_poly.decompose_fold_generic(&challenges, packed_layout.block_len(), 1);
    let generic_fold = generic_fold_start.elapsed();

    let fast_fold_start = Instant::now();
    let _ = packed_poly.decompose_fold_fast_singleton(&challenges, packed_layout.block_len(), 1);
    let fast_fold = fast_fold_start.elapsed();

    eprintln!(
        "packed bench: commit generic={generic_commit:?}, commit fast={fast_commit:?}, fold generic={generic_fold:?}, fold fast={fast_fold:?}"
    );
}

#[test]
#[ignore = "profiling helper for K=16 packed commit_inner"]
fn packed_k16_commit_inner_benchmark_smoke() {
    type FastCfg = Fp128OneHot64Config;

    let source = TestPackedSource::new(
        (0..32)
            .map(|poly_idx| {
                (0..256)
                    .map(|cycle_idx| {
                        if (poly_idx + cycle_idx) % 11 == 0 {
                            None
                        } else {
                            Some(((poly_idx * 7 + cycle_idx * 5) % 16) as u8)
                        }
                    })
                    .collect()
            })
            .collect(),
        16,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ FastCfg::D }, FastCfg>(&source);
    let log_basis = FastCfg::decomposition().log_basis;
    let num_digits_open = compute_num_digits(128, log_basis);
    let a_flat = make_test_a_matrix::<{ FastCfg::D }>(2, packed_layout.block_len(), 2);
    let a_view = a_flat.view::<{ FastCfg::D }>();

    let generic_start = Instant::now();
    let _ = packed_poly.commit_inner_generic(&a_view, 2, 2, num_digits_open, log_basis);
    let generic = generic_start.elapsed();

    let sweep_start = Instant::now();
    let _ = packed_poly.commit_inner_column_sweep(&a_view, 2, 2, num_digits_open, log_basis);
    let sweep = sweep_start.elapsed();

    eprintln!("packed K=16 commit_inner: generic={generic:?}, sweep={sweep:?}");
}

#[test]
#[ignore = "profiling helper for K=256 packed commit_inner"]
fn packed_k256_commit_inner_benchmark_smoke() {
    type FastCfg = Fp128OneHot64Config;

    let source = TestPackedSource::new(
        (0..32)
            .map(|poly_idx| {
                (0..256)
                    .map(|cycle_idx| {
                        if (poly_idx * 3 + cycle_idx) % 13 == 0 {
                            None
                        } else {
                            Some(((poly_idx * 37 + cycle_idx * 29) % 256) as u8)
                        }
                    })
                    .collect()
            })
            .collect(),
        256,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ FastCfg::D }, FastCfg>(&source);
    let log_basis = FastCfg::decomposition().log_basis;
    let num_digits_open = compute_num_digits(128, log_basis);
    let a_flat = make_test_a_matrix::<{ FastCfg::D }>(1, packed_layout.block_len(), 1);
    let a_view = a_flat.view::<{ FastCfg::D }>();

    let fast_start = Instant::now();
    let _ = packed_poly.commit_inner_fast_singleton(&a_view, num_digits_open, log_basis);
    let fast = fast_start.elapsed();

    let sweep_start = Instant::now();
    let _ = packed_poly.commit_inner_column_sweep(&a_view, 1, 1, num_digits_open, log_basis);
    let sweep = sweep_start.elapsed();

    eprintln!("packed K=256 commit_inner: fast={fast:?}, sweep={sweep:?}");
}

#[test]
#[ignore = "profiling helper for K=16 packed decompose_fold"]
fn packed_k16_decompose_fold_benchmark_smoke() {
    type FastCfg = Fp128OneHot64Config;

    let source = TestPackedSource::new(
        (0..32)
            .map(|poly_idx| {
                (0..256)
                    .map(|cycle_idx| {
                        if (poly_idx + cycle_idx) % 11 == 0 {
                            None
                        } else {
                            Some(((poly_idx * 7 + cycle_idx * 5) % 16) as u8)
                        }
                    })
                    .collect()
            })
            .collect(),
        16,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ FastCfg::D }, FastCfg>(&source);
    let challenges = make_test_challenges::<{ FastCfg::D }>(packed_layout.num_blocks());

    let generic_start = Instant::now();
    let _ = packed_poly.decompose_fold_generic(&challenges, packed_layout.block_len(), 1);
    let generic = generic_start.elapsed();

    let large_start = Instant::now();
    let _ = packed_poly.decompose_fold_large(&challenges, packed_layout.block_len(), 1);
    let large = large_start.elapsed();

    eprintln!("packed K=16 decompose_fold: generic={generic:?}, large={large:?}");
}

#[test]
#[ignore = "profiling helper for K=256 packed decompose_fold"]
fn packed_k256_decompose_fold_benchmark_smoke() {
    type FastCfg = Fp128OneHot64Config;

    let source = TestPackedSource::new(
        (0..32)
            .map(|poly_idx| {
                (0..256)
                    .map(|cycle_idx| {
                        if (poly_idx * 3 + cycle_idx) % 13 == 0 {
                            None
                        } else {
                            Some(((poly_idx * 37 + cycle_idx * 29) % 256) as u8)
                        }
                    })
                    .collect()
            })
            .collect(),
        256,
    );
    let (packed_poly, packed_layout) = build_test_packed_poly::<{ FastCfg::D }, FastCfg>(&source);
    let challenges = make_test_challenges::<{ FastCfg::D }>(packed_layout.num_blocks());

    let striped_start = Instant::now();
    let _ = packed_poly.decompose_fold_striped_delta1(&challenges, packed_layout.block_len());
    let striped = striped_start.elapsed();

    let fast_start = Instant::now();
    let _ = packed_poly.decompose_fold_fast_singleton(&challenges, packed_layout.block_len(), 1);
    let fast = fast_start.elapsed();

    eprintln!("packed K=256 decompose_fold: striped={striped:?}, fast={fast:?}");
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
fn hachi_batch_roundtrip_with_packed_layout_k16() {
    type FastCfg = Fp128OneHot64Config;
    type FastScheme = JoltHachiCommitmentScheme<{ FastCfg::D }, FastCfg>;

    let log_k = 4usize;
    let num_cycles = 1usize << 4;
    let source = TestPackedSource::new(
        vec![
            (0u8..16).map(Some).collect(),
            (0u8..16).rev().map(Some).collect(),
            vec![1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15]
                .into_iter()
                .map(Some)
                .collect(),
            vec![
                Some(0),
                None,
                Some(2),
                Some(4),
                None,
                Some(6),
                Some(8),
                Some(10),
                None,
                Some(12),
                Some(14),
                Some(0),
                Some(1),
                None,
                Some(3),
                Some(5),
            ],
        ],
        16,
    );
    let log_packed = source.per_poly.len().next_power_of_two().trailing_zeros() as usize;
    let setup = FastScheme::setup_prover_from_shape(
        num_cycles.trailing_zeros() as usize,
        log_k,
        Some(log_packed),
    );
    let verifier_setup = FastScheme::setup_verifier(&setup);
    let pcs = FastScheme::default();

    let (commitments, batch_hint) = pcs.batch_commit(&source, &setup);
    assert_eq!(
        commitments.len(),
        1,
        "packed Hachi should produce one commitment for K=16"
    );

    let opening_point: Vec<JoltFp128> = (0..(log_k + num_cycles.trailing_zeros() as usize))
        .map(|i| JoltFp128::from_u64((i + 2) as u64))
        .collect();

    let claims: Vec<JoltFp128> = source
        .per_poly
        .iter()
        .map(|indices| {
            OneHotPolynomial::<JoltFp128>::from_indices(indices.clone(), 16, num_cycles)
                .evaluate(&opening_point)
        })
        .collect();
    let commitment_refs = vec![&commitments[0]];

    let mut prove_transcript = Blake2bTranscript::new(b"hachi_batch_roundtrip_k16");
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

    let mut verify_transcript = Blake2bTranscript::new(b"hachi_batch_roundtrip_k16");
    pcs.batch_verify(
        &proof,
        &verifier_setup,
        &mut verify_transcript,
        &opening_point,
        &commitment_refs,
        &claims,
        &[],
    )
    .expect("packed Hachi batch verify should succeed for K=16");
}

#[test]
fn hachi_k256_setup_envelope_supports_k16_roundtrip() {
    type FastCfg = Fp128OneHot64Config;
    type FastScheme = JoltHachiCommitmentScheme<{ FastCfg::D }, FastCfg>;

    let num_cycles = 1usize << 4;
    let source = TestPackedSource::new(
        vec![
            (0u8..16).map(Some).collect(),
            (0u8..16).rev().map(Some).collect(),
            vec![1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15]
                .into_iter()
                .map(Some)
                .collect(),
            vec![
                Some(0),
                None,
                Some(2),
                Some(4),
                None,
                Some(6),
                Some(8),
                Some(10),
                None,
                Some(12),
                Some(14),
                Some(0),
                Some(1),
                None,
                Some(3),
                Some(5),
            ],
        ],
        16,
    );
    let log_packed = source.per_poly.len().next_power_of_two().trailing_zeros() as usize;
    let setup = FastScheme::setup_prover_from_shape(
        num_cycles.trailing_zeros() as usize,
        8,
        Some(log_packed),
    );
    let verifier_setup = FastScheme::setup_verifier(&setup);
    let pcs = FastScheme::default();

    let (commitments, batch_hint) = pcs.batch_commit(&source, &setup);
    let opening_point: Vec<JoltFp128> = (0..(4 + num_cycles.trailing_zeros() as usize))
        .map(|i| JoltFp128::from_u64((i + 7) as u64))
        .collect();
    let claims: Vec<JoltFp128> = source
        .per_poly
        .iter()
        .map(|indices| {
            OneHotPolynomial::<JoltFp128>::from_indices(indices.clone(), 16, num_cycles)
                .evaluate(&opening_point)
        })
        .collect();
    let commitment_refs = vec![&commitments[0]];

    let mut prove_transcript = Blake2bTranscript::new(b"hachi_setup_envelope_k16");
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

    let mut verify_transcript = Blake2bTranscript::new(b"hachi_setup_envelope_k16");
    pcs.batch_verify(
        &proof,
        &verifier_setup,
        &mut verify_transcript,
        &opening_point,
        &commitment_refs,
        &claims,
        &[],
    )
    .expect("K=256 setup envelope should still verify K=16 packed proofs");
}

#[test]
fn hachi_batch_verify_rejects_truncated_individual_commitments() {
    let setup = Scheme::setup_prover(16);
    let verifier_setup = Scheme::setup_verifier(&setup);
    let pcs = Scheme::default();
    let opening_point = vec![JoltFp128::from_u64(3), JoltFp128::from_u64(5)];

    let packed_poly_proof = ArkBridge(HachiProof {
        levels: vec![],
        tail: HachiProofTail::Direct(PackedDigits::from_i8_digits(&[], 1)),
    });
    let indiv_proof = ArkBridge(HachiProof {
        levels: vec![],
        tail: HachiProofTail::Direct(PackedDigits::from_i8_digits(&[], 1)),
    });
    let proof = HachiBatchedProof {
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
        tail: HachiProofTail::Direct(PackedDigits::from_i8_digits(&[], 1)),
    });
    let proof = HachiBatchedProof {
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
        tail: HachiProofTail::Direct(PackedDigits::from_i8_digits(&[], 1)),
    });
    let proof = HachiBatchedProof {
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
