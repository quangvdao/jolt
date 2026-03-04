use std::marker::PhantomData;

use super::wrappers::{jolt_to_hachi, ArkBridge, Fp128, JoltToHachiTranscript};
use crate::field::fp128::JoltFp128;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{
    CommitmentScheme, PolynomialBatchSource, StreamingCommitmentScheme,
};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::BatchPolynomialSource;
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;
use crate::utils::small_scalar::SmallScalar;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use hachi_pcs::algebra::fields::wide::Fp128x8i32;
use hachi_pcs::algebra::ring::sparse_challenge::SparseChallenge;
use hachi_pcs::algebra::ring::{CyclotomicRing, WideCyclotomicRing};
use hachi_pcs::protocol::commitment::utils::crt_ntt::NttSlotCache;
use hachi_pcs::protocol::commitment::utils::flat_matrix::FlatMatrix;
use hachi_pcs::protocol::commitment::utils::linear::decompose_rows_i8;
use hachi_pcs::protocol::commitment::{
    compute_num_digits, compute_num_digits_fold, optimal_m_r_split, CommitmentConfig,
    Fp128BoundedCommitmentConfig, HachiCommitmentLayout, RingCommitment,
};
use hachi_pcs::protocol::opening_point::BasisMode;
use hachi_pcs::protocol::proof::{HachiCommitmentHint, HachiProof};
use hachi_pcs::protocol::{HachiCommitmentScheme, HachiProverSetup, HachiVerifierSetup};
use hachi_pcs::CommitmentScheme as HachiCommitmentSchemeTrait;
use hachi_pcs::{
    CanonicalField, DensePoly, FieldCore, FromSmallInt, HachiPolyOps, OneHotIndex, OneHotPoly,
};
use rayon::prelude::*;

#[derive(Clone, Default)]
pub struct JoltHachiCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HachiBatchedProof<const D: usize> {
    pub packed_poly_proof: ArkBridge<HachiProof<Fp128>>,
    pub num_packed_polys: u32,
    pub log_k: u32,
    pub individual_proofs: Vec<ArkBridge<HachiProof<Fp128>>>,
}

#[derive(Clone, Debug)]
pub struct JoltHachiBatchHint<const D: usize> {
    hachi_hint: HachiCommitmentHint<Fp128, D>,
    block_len: usize,
    num_packed_polys: usize,
    log_k: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JoltHachiOpeningHint<const D: usize> {
    hachi_hint: HachiCommitmentHint<Fp128, D>,
    ring_coeffs: Vec<CyclotomicRing<Fp128, D>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HachiChunkState<const D: usize> {
    Dense(Vec<CyclotomicRing<Fp128, D>>),
    OneHot {
        onehot_k: usize,
        indices: Vec<Option<u8>>,
    },
}

/// Streaming view over the packed one-hot trace. Uses T-major layout
/// (field_pos = c * K + k) so cycle-contiguous entries map to adjacent
/// ring elements, enabling per-block parallelism and sequential A-matrix
/// access.
///
/// Generic over `IndexFn` — a closure `(cycle, poly) -> Option<u8>` that
/// the compiler monomorphizes and inlines into every hot loop, avoiding
/// the overhead of an erased function pointer.
#[derive(Clone)]
struct JoltPackedPoly<const D: usize, IndexFn> {
    index_of: IndexFn,
    num_cycles: usize,
    num_polys: usize,
    onehot_k: usize,
    block_len: usize,
    blocks_per_poly: usize,
    num_padded: usize,
}

impl<const D: usize, IndexFn> JoltPackedPoly<D, IndexFn> {
    /// T-major layout: field_pos = c * K + k. Addr varies fast, cycle
    /// varies slow. Opening-point convention: [r_addr_LE, r_cycle_LE].
    #[inline]
    fn decompose(&self, c: usize, k: u8) -> (usize, usize, usize) {
        let field_pos = c * self.onehot_k + (k as usize);
        let ring_idx = field_pos / D;
        let coeff_idx = field_pos % D;
        let block_within = ring_idx / self.block_len;
        let pos_in_block = ring_idx % self.block_len;
        (block_within, pos_in_block, coeff_idx)
    }

    /// Cycle range that may contribute to within-poly block `bw`.
    /// Slight over-approximation; callers verify ring_idx bounds.
    #[inline]
    fn block_cycle_range(&self, bw: usize) -> (usize, usize) {
        let ring_start = bw * self.block_len;
        let ring_end = ring_start + self.block_len;
        let c_start = (ring_start * D) / self.onehot_k;
        let c_end = (ring_end * D).div_ceil(self.onehot_k).min(self.num_cycles);
        (c_start, c_end)
    }
}

impl<const D: usize, IndexFn> HachiPolyOps<Fp128, D> for JoltPackedPoly<D, IndexFn>
where
    IndexFn: Fn(usize, usize) -> Option<u8> + Clone + Send + Sync,
{
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        (self.num_padded * self.blocks_per_poly) * self.block_len
    }

    #[tracing::instrument(skip_all, name = "JoltPackedPoly::evaluate_ring")]
    fn evaluate_ring(&self, scalars: &[Fp128]) -> CyclotomicRing<Fp128, D> {
        let bl = self.block_len;
        let bpp = self.blocks_per_poly;
        let chunk_size = 512;
        let num_chunks = self.num_cycles.div_ceil(chunk_size);

        (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let c_start = chunk_idx * chunk_size;
                let c_end = (c_start + chunk_size).min(self.num_cycles);
                let mut acc = [Fp128::zero(); D];
                for c in c_start..c_end {
                    for p in 0..self.num_polys {
                        if let Some(k) = (self.index_of)(c, p) {
                            let (block_within, pos_in_block, ci) = self.decompose(c, k);
                            let global_ring = (p * bpp + block_within) * bl + pos_in_block;
                            if global_ring < scalars.len() {
                                acc[ci] += scalars[global_ring];
                            }
                        }
                    }
                }
                CyclotomicRing::from_coefficients(acc)
            })
            .reduce(CyclotomicRing::zero, |a, b| a + b)
    }

    #[tracing::instrument(skip_all, name = "JoltPackedPoly::fold_blocks")]
    fn fold_blocks(&self, scalars: &[Fp128], _block_len: usize) -> Vec<CyclotomicRing<Fp128, D>> {
        let bpp = self.blocks_per_poly;
        let total_blocks = self.num_padded * bpp;
        let real_blocks = self.num_polys * bpp;

        let mut result: Vec<CyclotomicRing<Fp128, D>> = (0..real_blocks)
            .into_par_iter()
            .map(|b| {
                let p = b / bpp;
                let bw = b % bpp;
                let (c_start, c_end) = self.block_cycle_range(bw);
                let ring_base = bw * self.block_len;
                let mut acc = [Fp128::zero(); D];
                for c in c_start..c_end {
                    if let Some(k) = (self.index_of)(c, p) {
                        let field_pos = c * self.onehot_k + (k as usize);
                        let ring_idx = field_pos / D;
                        let ci = field_pos % D;
                        let pos = ring_idx.wrapping_sub(ring_base);
                        if pos < self.block_len && pos < scalars.len() {
                            acc[ci] += scalars[pos];
                        }
                    }
                }
                CyclotomicRing::from_coefficients(acc)
            })
            .collect();
        result.resize(total_blocks, CyclotomicRing::zero());
        result
    }

    #[tracing::instrument(skip_all, name = "JoltPackedPoly::evaluate_and_fold")]
    fn evaluate_and_fold(
        &self,
        eval_scalars: &[Fp128],
        fold_scalars: &[Fp128],
        _block_len: usize,
    ) -> (CyclotomicRing<Fp128, D>, Vec<CyclotomicRing<Fp128, D>>) {
        if eval_scalars.is_empty() {
            return (CyclotomicRing::zero(), self.fold_blocks(fold_scalars, 0));
        }
        if fold_scalars.is_empty() {
            return (self.evaluate_ring(eval_scalars), vec![]);
        }

        let bl = self.block_len;
        let bpp = self.blocks_per_poly;
        let total_blocks = self.num_padded * bpp;
        let real_blocks = self.num_polys * bpp;

        let mut fold_result = vec![CyclotomicRing::zero(); total_blocks];
        let eval_total = fold_result[..real_blocks]
            .par_iter_mut()
            .enumerate()
            .fold(
                || [Fp128::zero(); D],
                |mut eval_local, (b, fold_slot)| {
                    let p = b / bpp;
                    let bw = b % bpp;
                    let (c_start, c_end) = self.block_cycle_range(bw);
                    let ring_base = bw * bl;
                    let mut eval_acc = [Fp128::zero(); D];
                    let mut fold_acc = [Fp128::zero(); D];

                    for c in c_start..c_end {
                        if let Some(k) = (self.index_of)(c, p) {
                            let field_pos = c * self.onehot_k + (k as usize);
                            let ring_idx = field_pos / D;
                            let ci = field_pos % D;
                            let pos = ring_idx.wrapping_sub(ring_base);
                            if pos < bl {
                                let global_ring = b * bl + pos;
                                if global_ring < eval_scalars.len() {
                                    eval_acc[ci] += eval_scalars[global_ring];
                                }
                                if pos < fold_scalars.len() {
                                    fold_acc[ci] += fold_scalars[pos];
                                }
                            }
                        }
                    }

                    *fold_slot = CyclotomicRing::from_coefficients(fold_acc);
                    for i in 0..D {
                        eval_local[i] += eval_acc[i];
                    }
                    eval_local
                },
            )
            .reduce(
                || [Fp128::zero(); D],
                |mut a, b| {
                    for i in 0..D {
                        a[i] += b[i];
                    }
                    a
                },
            );

        (CyclotomicRing::from_coefficients(eval_total), fold_result)
    }

    #[tracing::instrument(skip_all, name = "JoltPackedPoly::decompose_fold")]
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        delta: usize,
        _log_basis: u32,
    ) -> Vec<CyclotomicRing<Fp128, D>> {
        let inner_width = block_len * delta;
        let bpp = self.blocks_per_poly;
        let chunk_size = 512;
        let num_chunks = self.num_cycles.div_ceil(chunk_size);

        // Per-thread accumulator is inner_width * D * 4 bytes. When this
        // exceeds ~4MB the buffer no longer fits in L3 cache, causing
        // massive slowdowns on large traces. Tile the inner_width dimension
        // so each pass keeps the working set cache-resident.
        const TILE_THRESHOLD_ELEMS: usize = 4096;
        let tile_size = if inner_width <= TILE_THRESHOLD_ELEMS {
            inner_width
        } else {
            TILE_THRESHOLD_ELEMS
        };
        let num_tiles = inner_width.div_ceil(tile_size);

        let mut z_i32 = vec![[0i32; D]; inner_width];

        for tile_idx in 0..num_tiles {
            let tile_start = tile_idx * tile_size;
            let tile_end = (tile_start + tile_size).min(inner_width);
            let tile_len = tile_end - tile_start;

            let tile: Vec<[i32; D]> = (0..num_chunks)
                .into_par_iter()
                .fold(
                    || vec![[0i32; D]; tile_len],
                    |mut z, chunk_idx| {
                        let c_start = chunk_idx * chunk_size;
                        let c_end = (c_start + chunk_size).min(self.num_cycles);
                        for c in c_start..c_end {
                            for p in 0..self.num_polys {
                                if let Some(k) = (self.index_of)(c, p) {
                                    let (block_within, pos_in_block, ci) = self.decompose(c, k);
                                    let packed_block = p * bpp + block_within;
                                    if packed_block >= challenges.len() {
                                        continue;
                                    }
                                    let base_j = pos_in_block * delta;
                                    if base_j < tile_start || base_j >= tile_end {
                                        continue;
                                    }
                                    let local_j = base_j - tile_start;
                                    let c_i = &challenges[packed_block];
                                    for (&pos, &challenge_coeff) in
                                        c_i.positions.iter().zip(c_i.coeffs.iter())
                                    {
                                        let target = ci + pos as usize;
                                        let (idx, sign) = if target < D {
                                            (target, 1i32)
                                        } else {
                                            (target - D, -1i32)
                                        };
                                        z[local_j][idx] += sign * challenge_coeff as i32;
                                    }
                                }
                            }
                        }
                        z
                    },
                )
                .reduce(
                    || vec![[0i32; D]; tile_len],
                    |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b.iter()) {
                            for (a_coeff, b_coeff) in ai.iter_mut().zip(bi.iter()) {
                                *a_coeff += b_coeff;
                            }
                        }
                        a
                    },
                );

            z_i32[tile_start..tile_end].copy_from_slice(&tile);
        }

        let q = (-Fp128::one()).to_canonical_u128() + 1;
        z_i32
            .into_iter()
            .map(|arr| {
                let coeffs = std::array::from_fn(|k| {
                    let v = arr[k];
                    if v >= 0 {
                        Fp128::from_canonical_u128_reduced(v as u128)
                    } else {
                        Fp128::from_canonical_u128_reduced(q - ((-v) as u128))
                    }
                });
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect()
    }

    #[allow(non_snake_case)]
    #[tracing::instrument(skip_all, name = "JoltPackedPoly::commit_inner")]
    fn commit_inner(
        &self,
        a_matrix: &FlatMatrix<Fp128>,
        _ntt_a: &NttSlotCache<D>,
        _block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<[i8; D]>>, hachi_pcs::HachiError> {
        let a_view = a_matrix.view::<D>();
        let n_a = a_view.num_rows();
        let t_hat_len = n_a.checked_mul(num_digits_open).unwrap();
        let bl = self.block_len;
        let bpp = self.blocks_per_poly;
        let total_blocks = self.num_padded * bpp;
        let real_blocks = self.num_polys * bpp;

        let a_wide_flat: Vec<WideCyclotomicRing<Fp128x8i32, D>> = (0..bl)
            .into_par_iter()
            .flat_map_iter(|pos| {
                let col = pos * num_digits_commit;
                (0..n_a).map(move |a| WideCyclotomicRing::from_ring(&a_view.row(a)[col]))
            })
            .collect();

        let mut results: Vec<Vec<[i8; D]>> = (0..real_blocks)
            .into_par_iter()
            .map_init(
                || vec![WideCyclotomicRing::<Fp128x8i32, D>::zero(); n_a],
                |t_wide, b| {
                    for tw in t_wide.iter_mut() {
                        *tw = WideCyclotomicRing::zero();
                    }

                    let p = b / bpp;
                    let bw = b % bpp;
                    let (c_start, c_end) = self.block_cycle_range(bw);
                    let ring_base = bw * bl;

                    for c in c_start..c_end {
                        if let Some(k) = (self.index_of)(c, p) {
                            let field_pos = c * self.onehot_k + (k as usize);
                            let ring_idx = field_pos / D;
                            let ci = field_pos % D;
                            let pos = ring_idx.wrapping_sub(ring_base);
                            if pos < bl {
                                let a_base = pos * n_a;
                                for a in 0..n_a {
                                    a_wide_flat[a_base + a]
                                        .mul_by_monomial_sum_into(&mut t_wide[a], &[ci]);
                                }
                            }
                        }
                    }

                    let t: Vec<CyclotomicRing<Fp128, D>> =
                        t_wide.iter().map(|w| w.reduce()).collect();
                    decompose_rows_i8(&t, num_digits_open, log_basis)
                },
            )
            .collect();
        results.resize_with(total_blocks, || vec![[0i8; D]; t_hat_len]);

        Ok(results)
    }
}

fn to_hachi_opening_point<const D: usize>(point: &[JoltFp128]) -> Vec<Fp128> {
    point.iter().rev().map(jolt_to_hachi).collect()
}

fn to_hachi_packed_opening_point<const D: usize>(
    opening_point: &[JoltFp128],
    rho: &[JoltFp128],
    log_k: usize,
) -> Vec<Fp128> {
    // Jolt opening point (BE): [addr_BE, cycle_BE]
    // After rev: [cycle_LE, addr_LE]
    let reversed: Vec<Fp128> = opening_point.iter().rev().map(jolt_to_hachi).collect();
    assert!(
        log_k <= reversed.len(),
        "packed opening point expects log_k <= num_vars (log_k={}, num_vars={})",
        log_k,
        reversed.len()
    );
    let log_t = reversed.len() - log_k;
    // T-major LE variable order: [addr_LE, cycle_LE, poly_LE]
    let mut out = Vec::with_capacity(reversed.len() + rho.len());
    out.extend_from_slice(&reversed[log_t..]);
    out.extend_from_slice(&reversed[..log_t]);
    out.extend(rho.iter().rev().map(jolt_to_hachi));
    out
}

fn onehot_commit_layout<Cfg: CommitmentConfig>(
    m_vars: usize,
    r_vars: usize,
) -> HachiCommitmentLayout {
    let log_basis = Cfg::decomposition().log_basis;
    HachiCommitmentLayout::new_with_decomp(
        m_vars,
        r_vars,
        Cfg::N_A,
        1,
        compute_num_digits(128, log_basis),
        compute_num_digits_fold(r_vars, Cfg::CHALLENGE_WEIGHT, log_basis),
        log_basis,
    )
    .unwrap()
}

fn advice_commit_layout<Cfg: CommitmentConfig>(
    m_vars: usize,
    r_vars: usize,
) -> HachiCommitmentLayout {
    let log_basis = Cfg::decomposition().log_basis;
    HachiCommitmentLayout::new_with_decomp(
        m_vars,
        r_vars,
        Cfg::N_A,
        compute_num_digits(64, log_basis),
        compute_num_digits(128, log_basis),
        compute_num_digits_fold(r_vars, Cfg::CHALLENGE_WEIGHT, log_basis),
        log_basis,
    )
    .unwrap()
}

/// Compute the advice commit layout using the polynomial's own optimal m/r split
/// rather than inheriting from the setup layout.
///
/// The setup may use a very different m/r split (optimized for onehot with
/// num_digits_commit=1), so advice polynomials (with num_digits_commit=22)
/// need their own split to keep inner_width within the setup's bounds.
fn compute_advice_layout<const D: usize, Cfg: CommitmentConfig>(
    poly_num_vars: usize,
) -> HachiCommitmentLayout {
    let alpha = D.trailing_zeros() as usize;
    let reduced_vars = poly_num_vars.saturating_sub(alpha);
    if reduced_vars <= 1 {
        return advice_commit_layout::<Cfg>(reduced_vars.max(1), 0);
    }
    // Advice polynomials have log_commit_bound=64, so use that config for
    // the optimal m/r split. All Fp128BoundedCommitmentConfig variants
    // share the same N_A, CHALLENGE_WEIGHT, log_basis.
    let (m_vars, r_vars) = optimal_m_r_split::<Fp128BoundedCommitmentConfig<64>>(reduced_vars);
    advice_commit_layout::<Cfg>(m_vars, r_vars)
}

fn hachi_commit_dense<const D: usize, Cfg: CommitmentConfig>(
    ring_coeffs: Vec<CyclotomicRing<Fp128, D>>,
    setup: &HachiProverSetup<Fp128, D>,
    layout: &HachiCommitmentLayout,
) -> (RingCommitment<Fp128, D>, JoltHachiOpeningHint<D>) {
    let mut dense_poly = DensePoly::from_ring_coeffs(ring_coeffs);
    let (commitment, hachi_hint) = <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<
        Fp128,
        D,
    >>::commit(&dense_poly, setup, layout)
    .expect("Hachi commit failed");
    let ring_coeffs = std::mem::take(&mut dense_poly.coeffs);
    (
        commitment,
        JoltHachiOpeningHint {
            hachi_hint,
            ring_coeffs,
        },
    )
}

fn hachi_commit_onehot<const D: usize, Cfg: CommitmentConfig, I: OneHotIndex>(
    onehot_k: usize,
    indices: Vec<Option<I>>,
    setup: &HachiProverSetup<Fp128, D>,
    layout: &HachiCommitmentLayout,
) -> (RingCommitment<Fp128, D>, JoltHachiOpeningHint<D>) {
    let onehot_poly =
        OneHotPoly::<Fp128, D, I>::new(onehot_k, indices, layout.r_vars, layout.m_vars)
            .expect("OneHotPoly construction failed");
    let (commitment, hachi_hint) = <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<
        Fp128,
        D,
    >>::commit(&onehot_poly, setup, layout)
    .expect("Hachi commit_onehot failed");
    (
        commitment,
        JoltHachiOpeningHint {
            hachi_hint,
            ring_coeffs: vec![],
        },
    )
}

fn build_packed_poly<const D: usize, IndexFn: Fn(usize, usize) -> Option<u8>>(
    index_of: IndexFn,
    num_cycles: usize,
    num_polys: usize,
    onehot_k: usize,
    block_len: usize,
) -> JoltPackedPoly<D, IndexFn> {
    let poly_field_len = num_cycles * onehot_k;
    let poly_ring_len = poly_field_len.div_ceil(D);
    let blocks_per_poly = poly_ring_len.div_ceil(block_len);
    let num_padded = num_polys.next_power_of_two();

    JoltPackedPoly {
        index_of,
        num_cycles,
        num_polys,
        onehot_k,
        block_len,
        blocks_per_poly,
        num_padded,
    }
}

impl<const D: usize, Cfg> CommitmentScheme for JoltHachiCommitmentScheme<D, Cfg>
where
    Cfg: CommitmentConfig + Default,
{
    type Field = JoltFp128;
    type Config = ();
    type ProverSetup = ArkBridge<HachiProverSetup<Fp128, D>>;
    type VerifierSetup = ArkBridge<HachiVerifierSetup<Fp128>>;
    type Commitment = ArkBridge<RingCommitment<Fp128, D>>;
    type Proof = ArkBridge<HachiProof<Fp128>>;
    type BatchedProof = HachiBatchedProof<D>;
    type OpeningProofHint = JoltHachiOpeningHint<D>;
    type BatchOpeningHint = JoltHachiBatchHint<D>;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        ArkBridge(
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::setup_prover(
                max_num_vars,
            ),
        )
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        ArkBridge(
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::setup_verifier(
                &setup.0,
            ),
        )
    }

    fn from_proof(_proof: &Self::BatchedProof) -> Self {
        Self::default()
    }

    fn config(&self) -> &() {
        &()
    }

    fn commit(
        &self,
        poly: &MultilinearPolynomial<JoltFp128>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let poly_num_vars = poly.len().trailing_zeros() as usize;
        let layout = compute_advice_layout::<D, Cfg>(poly_num_vars);

        let (commitment, hint) = if let MultilinearPolynomial::OneHot(onehot) = poly {
            let indices: Vec<Option<u8>> = onehot.nonzero_indices.as_ref().clone();
            hachi_commit_onehot::<D, Cfg, u8>(onehot.K, indices, &setup.0, &layout)
        } else {
            let ring_coeffs = poly_to_ring_coeffs::<D>(poly);
            hachi_commit_dense::<D, Cfg>(ring_coeffs, &setup.0, &layout)
        };
        (ArkBridge(commitment), hint)
    }

    fn batch_commit<S: PolynomialBatchSource<JoltFp128>>(
        &self,
        source: &S,
        setup: &Self::ProverSetup,
    ) -> (Vec<Self::Commitment>, Self::BatchOpeningHint) {
        assert!(source.num_polys() > 0);
        let layout = setup.0.layout();
        let block_len = layout.block_len;

        let num_cycles = source
            .num_cycles()
            .expect("batch_commit requires lazy source");
        let onehot_k = source.onehot_k().unwrap();
        let num_polys = source.num_polys();

        let index_fn = |c: usize, p: usize| source.onehot_index(c, p);
        let packed_poly =
            build_packed_poly::<D, _>(index_fn, num_cycles, num_polys, onehot_k, block_len);
        let total_packed_blocks = packed_poly.num_padded * packed_poly.blocks_per_poly;
        let packed_r_vars = (total_packed_blocks as u32).trailing_zeros() as usize;
        let batch_layout = onehot_commit_layout::<Cfg>(layout.m_vars, packed_r_vars);

        let (commitment, hachi_hint) =
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::commit(
                &packed_poly,
                &setup.0,
                &batch_layout,
            )
            .expect("Hachi packed poly commit failed");

        let log_k = onehot_k.trailing_zeros() as usize;

        let hint = JoltHachiBatchHint {
            hachi_hint,
            block_len,
            num_packed_polys: num_polys,
            log_k,
        };
        (vec![ArkBridge(commitment)], hint)
    }

    fn prove<ProofTranscript: Transcript>(
        &self,
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<JoltFp128>,
        opening_point: &[JoltFp128],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
        commitment: &Self::Commitment,
    ) -> Self::Proof {
        let hint = hint.expect("prove() requires a hint");
        let hachi_point = to_hachi_opening_point::<D>(opening_point);
        let mut adapter = JoltToHachiTranscript::new(transcript);

        let layout = compute_advice_layout::<D, Cfg>(opening_point.len());

        let proof = if let MultilinearPolynomial::OneHot(onehot) = poly {
            let indices: Vec<Option<u8>> = onehot.nonzero_indices.as_ref().clone();
            let onehot_poly =
                OneHotPoly::<Fp128, D, u8>::new(onehot.K, indices, layout.r_vars, layout.m_vars)
                    .expect("OneHotPoly construction failed");
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::prove(
                &setup.0,
                &onehot_poly,
                &hachi_point,
                hint.hachi_hint,
                &mut adapter,
                &commitment.0,
                BasisMode::Lagrange,
                &layout,
            )
        } else {
            let dense_poly = if hint.ring_coeffs.is_empty() {
                let ring_coeffs = poly_to_ring_coeffs::<D>(poly);
                DensePoly::from_ring_coeffs(ring_coeffs)
            } else {
                DensePoly::from_ring_coeffs(hint.ring_coeffs)
            };
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::prove(
                &setup.0,
                &dense_poly,
                &hachi_point,
                hint.hachi_hint,
                &mut adapter,
                &commitment.0,
                BasisMode::Lagrange,
                &layout,
            )
        }
        .expect("Hachi prove failed");
        ArkBridge(proof)
    }

    fn verify<ProofTranscript: Transcript>(
        &self,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[JoltFp128],
        opening: &JoltFp128,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let hachi_point = to_hachi_opening_point::<D>(opening_point);
        let hachi_opening = jolt_to_hachi(opening);
        let mut adapter = JoltToHachiTranscript::new(transcript);

        let layout = compute_advice_layout::<D, Cfg>(opening_point.len());

        <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::verify(
            &proof.0,
            &setup.0,
            &mut adapter,
            &hachi_point,
            &hachi_opening,
            &commitment.0,
            BasisMode::Lagrange,
            &layout,
        )
        .map_err(|_| ProofVerifyError::InternalError)
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_prove<ProofTranscript: Transcript, PolySource: BatchPolynomialSource<JoltFp128>>(
        &self,
        setup: &Self::ProverSetup,
        poly_source: &PolySource,
        batch_hint: Self::BatchOpeningHint,
        individual_hints: Vec<Self::OpeningProofHint>,
        commitments: &[&Self::Commitment],
        opening_point: &[JoltFp128],
        claims: &[JoltFp128],
        _coeffs: &[JoltFp128],
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        let num_individual = individual_hints.len();
        let num_packed = batch_hint.num_packed_polys;
        assert!(
            num_packed > 0,
            "batch_prove requires at least one packed claim"
        );
        assert_eq!(
            claims.len(),
            num_packed + num_individual,
            "batch_prove claims must be [packed_claims, individual_claims]"
        );
        assert_eq!(
            commitments.len(),
            1 + num_individual,
            "batch_prove commitments must be [packed_commitment, individual_commitments...]"
        );

        let packed_claims = &claims[..num_packed];
        let num_padded = num_packed.next_power_of_two();
        let r_vars = (num_padded as u32).trailing_zeros() as usize;

        transcript.append_bytes(b"hachi_packed_num", &(num_packed as u64).to_le_bytes());
        let rho: Vec<JoltFp128> = transcript.challenge_vector(r_vars);

        let eq_table = EqPolynomial::<JoltFp128>::evals(&rho);
        let combined_claim: JoltFp128 = packed_claims
            .iter()
            .zip(eq_table.iter())
            .map(|(&v, &eq)| v * eq)
            .sum();

        let setup_layout = &setup.0.expanded.seed.layout;
        let log_k = batch_hint.log_k;
        assert!(
            log_k <= opening_point.len(),
            "batch_prove log_k exceeds opening point length (log_k={}, point_len={})",
            log_k,
            opening_point.len()
        );
        let packed_point = to_hachi_packed_opening_point::<D>(opening_point, &rho, log_k);

        let alpha = D.trailing_zeros() as usize;
        let packed_r_vars = packed_point
            .len()
            .saturating_sub(alpha + setup_layout.m_vars);
        let packed_layout = onehot_commit_layout::<Cfg>(setup_layout.m_vars, packed_r_vars);

        let hachi_combined = jolt_to_hachi(&combined_claim);
        let packed_commitment = commitments[0];

        transcript.append_bytes(
            b"hachi_packed_claim",
            &hachi_combined.to_canonical_u128().to_le_bytes(),
        );

        let num_cycles = poly_source
            .num_cycles()
            .expect("batch_prove requires lazy source");
        let onehot_k = poly_source.onehot_k().unwrap();
        let num_polys = poly_source.num_polys().unwrap();
        let block_len = batch_hint.block_len;

        let index_fn = |c: usize, p: usize| poly_source.onehot_index(c, p);
        let packed_poly =
            build_packed_poly::<D, _>(index_fn, num_cycles, num_polys, onehot_k, block_len);

        let mut adapter = JoltToHachiTranscript::new(transcript);

        let packed_proof =
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::prove(
                &setup.0,
                &packed_poly,
                &packed_point,
                batch_hint.hachi_hint,
                &mut adapter,
                &packed_commitment.0,
                BasisMode::Lagrange,
                &packed_layout,
            )
            .expect("Hachi packed poly prove failed");

        let hachi_point = to_hachi_opening_point::<D>(opening_point);
        let indiv_layout = compute_advice_layout::<D, Cfg>(opening_point.len());

        let individual_commitments = &commitments[1..];
        let individual_proofs: Vec<ArkBridge<HachiProof<Fp128>>> = individual_hints
            .into_iter()
            .zip(individual_commitments.iter())
            .enumerate()
            .map(|(i, (hint, commitment))| {
                transcript.append_bytes(b"hachi_individual_item", &(i as u64).to_le_bytes());
                let individual_poly = DensePoly::from_ring_coeffs(hint.ring_coeffs);
                let mut adapter = JoltToHachiTranscript::new(transcript);
                let proof =
                    <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::prove(
                        &setup.0,
                        &individual_poly,
                        &hachi_point,
                        hint.hachi_hint,
                        &mut adapter,
                        &commitment.0,
                        BasisMode::Lagrange,
                        &indiv_layout,
                    )
                    .expect("Hachi individual prove failed");
                ArkBridge(proof)
            })
            .collect();

        HachiBatchedProof {
            packed_poly_proof: ArkBridge(packed_proof),
            num_packed_polys: num_packed as u32,
            log_k: log_k as u32,
            individual_proofs,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_verify<ProofTranscript: Transcript>(
        &self,
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[JoltFp128],
        commitments: &[&Self::Commitment],
        claims: &[JoltFp128],
        _coeffs: &[JoltFp128],
    ) -> Result<(), ProofVerifyError> {
        let num_packed = proof.num_packed_polys as usize;
        if num_packed == 0 {
            return Err(ProofVerifyError::InvalidInputLength(1, num_packed));
        }
        if num_packed > claims.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                num_packed,
                claims.len(),
            ));
        }

        let packed_claims = &claims[..num_packed];
        let individual_claims = &claims[num_packed..];
        if commitments.len() != 1 + individual_claims.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                1 + individual_claims.len(),
                commitments.len(),
            ));
        }
        if proof.individual_proofs.len() != individual_claims.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                individual_claims.len(),
                proof.individual_proofs.len(),
            ));
        }

        let num_padded = num_packed.next_power_of_two();
        let selector_vars = (num_padded as u32).trailing_zeros() as usize;

        transcript.append_bytes(b"hachi_packed_num", &(num_packed as u64).to_le_bytes());
        let rho: Vec<JoltFp128> = transcript.challenge_vector(selector_vars);

        let eq_table = EqPolynomial::<JoltFp128>::evals(&rho);
        let combined_claim: JoltFp128 = packed_claims
            .iter()
            .zip(eq_table.iter())
            .map(|(&v, &eq)| v * eq)
            .sum();

        let setup_layout = &setup.0.expanded.seed.layout;
        let log_k = proof.log_k as usize;
        if log_k > opening_point.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                opening_point.len(),
                log_k,
            ));
        }
        let packed_point = to_hachi_packed_opening_point::<D>(opening_point, &rho, log_k);

        let alpha = D.trailing_zeros() as usize;
        let packed_r_vars = packed_point
            .len()
            .saturating_sub(alpha + setup_layout.m_vars);
        let packed_layout = onehot_commit_layout::<Cfg>(setup_layout.m_vars, packed_r_vars);

        let hachi_combined = jolt_to_hachi(&combined_claim);
        let packed_commitment = commitments[0];

        transcript.append_bytes(
            b"hachi_packed_claim",
            &hachi_combined.to_canonical_u128().to_le_bytes(),
        );
        let mut adapter = JoltToHachiTranscript::new(transcript);
        <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::verify(
            &proof.packed_poly_proof.0,
            &setup.0,
            &mut adapter,
            &packed_point,
            &hachi_combined,
            &packed_commitment.0,
            BasisMode::Lagrange,
            &packed_layout,
        )
        .map_err(|_| ProofVerifyError::InternalError)?;

        let hachi_point = to_hachi_opening_point::<D>(opening_point);
        let indiv_layout = compute_advice_layout::<D, Cfg>(opening_point.len());

        let individual_commitments = &commitments[1..];
        for i in 0..individual_claims.len() {
            transcript.append_bytes(b"hachi_individual_item", &(i as u64).to_le_bytes());
            let hachi_claim = jolt_to_hachi(&individual_claims[i]);
            let mut adapter = JoltToHachiTranscript::new(transcript);
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::verify(
                &proof.individual_proofs[i].0,
                &setup.0,
                &mut adapter,
                &hachi_point,
                &hachi_claim,
                &individual_commitments[i].0,
                BasisMode::Lagrange,
                &indiv_layout,
            )
            .map_err(|_| ProofVerifyError::InternalError)?;
        }

        Ok(())
    }

    fn split_batch_hint(_batch_hint: &Self::BatchOpeningHint) -> Vec<Self::OpeningProofHint> {
        vec![]
    }

    fn protocol_name() -> &'static [u8] {
        b"Hachi"
    }

    fn packed_main_commitment_arity() -> Option<usize> {
        Some(1)
    }

    fn uses_onehot_inc() -> bool {
        true
    }
}

impl<const D: usize, Cfg> StreamingCommitmentScheme for JoltHachiCommitmentScheme<D, Cfg>
where
    Cfg: CommitmentConfig + Default,
{
    type ChunkState = HachiChunkState<D>;

    #[allow(non_snake_case)]
    fn streaming_chunk_size(&self, _K: usize, _T: usize) -> Option<usize> {
        None
    }

    fn process_chunk<T: SmallScalar>(
        &self,
        _setup: &Self::ProverSetup,
        _chunk: &[T],
    ) -> Self::ChunkState {
        unreachable!("Hachi uses batch_commit via PolynomialBatchSource, not streaming")
    }

    fn process_chunk_onehot(
        &self,
        _setup: &Self::ProverSetup,
        _onehot_k: usize,
        _chunk: &[Option<usize>],
    ) -> Self::ChunkState {
        unreachable!("Hachi uses batch_commit via PolynomialBatchSource, not streaming")
    }

    fn aggregate_chunks(
        &self,
        _setup: &Self::ProverSetup,
        _onehot_k: Option<usize>,
        _tier1_commitments: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        unreachable!("Hachi uses batch_commit via PolynomialBatchSource, not streaming")
    }

    fn aggregate_streaming_batch(
        &self,
        _setup: &Self::ProverSetup,
        _onehot_ks: &[Option<usize>],
        _tier1_per_poly: &[Vec<Self::ChunkState>],
    ) -> Option<(Vec<Self::Commitment>, Self::BatchOpeningHint)> {
        None
    }

    fn streaming_batch_hint(_hints: Vec<Self::OpeningProofHint>) -> Self::BatchOpeningHint {
        panic!("Hachi uses batch_commit via PolynomialBatchSource, not streaming")
    }
}

pub(super) fn poly_to_ring_coeffs<const D: usize>(
    poly: &MultilinearPolynomial<JoltFp128>,
) -> Vec<CyclotomicRing<Fp128, D>> {
    match poly {
        MultilinearPolynomial::LargeScalars(p) => {
            // SAFETY: JoltFp128 is repr(transparent) over Fp128.
            let field_coeffs: &[Fp128] =
                unsafe { std::slice::from_raw_parts(p.Z.as_ptr() as *const Fp128, p.Z.len()) };
            pack_field_to_ring::<D>(field_coeffs)
        }
        MultilinearPolynomial::BoolScalars(p) => {
            pack_scalars::<D, _, _>(&p.coeffs, |&b| if b { Fp128::one() } else { Fp128::zero() })
        }
        MultilinearPolynomial::U8Scalars(p) => {
            pack_scalars::<D, _, _>(&p.coeffs, |&v| Fp128::from_u64(v as u64))
        }
        MultilinearPolynomial::U16Scalars(p) => {
            pack_scalars::<D, _, _>(&p.coeffs, |&v| Fp128::from_u64(v as u64))
        }
        MultilinearPolynomial::U32Scalars(p) => {
            pack_scalars::<D, _, _>(&p.coeffs, |&v| Fp128::from_u64(v as u64))
        }
        MultilinearPolynomial::U64Scalars(p) => {
            pack_scalars::<D, _, _>(&p.coeffs, |&v| Fp128::from_u64(v))
        }
        MultilinearPolynomial::U128Scalars(p) => pack_scalars::<D, _, _>(&p.coeffs, |&v| {
            <Fp128 as CanonicalField>::from_canonical_u128_reduced(v)
        }),
        MultilinearPolynomial::I64Scalars(p) => pack_scalars::<D, _, _>(&p.coeffs, |&v| {
            if v >= 0 {
                Fp128::from_u64(v as u64)
            } else {
                -Fp128::from_u64(v.unsigned_abs())
            }
        }),
        MultilinearPolynomial::I128Scalars(p) => pack_scalars::<D, _, _>(&p.coeffs, |&v| {
            let jolt = JoltFp128::from_i128(v);
            jolt_to_hachi(&jolt)
        }),
        MultilinearPolynomial::S128Scalars(p) => pack_scalars::<D, _, _>(&p.coeffs, |v| {
            if let Some(i) = v.to_i128() {
                let jolt = JoltFp128::from_i128(i);
                jolt_to_hachi(&jolt)
            } else {
                let mag = v.magnitude_as_u128();
                let f = <Fp128 as CanonicalField>::from_canonical_u128_reduced(mag);
                if v.is_positive {
                    f
                } else {
                    -f
                }
            }
        }),
        MultilinearPolynomial::OneHot(_) | MultilinearPolynomial::RLC(_) => {
            panic!("OneHot and RLC polynomials cannot be materialized for Hachi commit")
        }
    }
}

fn pack_scalars<const D: usize, T: Sync, F: Fn(&T) -> Fp128 + Sync + Send>(
    scalars: &[T],
    convert: F,
) -> Vec<CyclotomicRing<Fp128, D>> {
    let par_grain = D * 256;
    scalars
        .par_chunks(par_grain)
        .flat_map_iter(|big_chunk| {
            big_chunk.chunks(D).map(|chunk| {
                let mut coeffs = [Fp128::zero(); D];
                for (i, scalar) in chunk.iter().enumerate() {
                    coeffs[i] = convert(scalar);
                }
                CyclotomicRing::from_coefficients(coeffs)
            })
        })
        .collect()
}

fn pack_field_to_ring<const D: usize>(field_coeffs: &[Fp128]) -> Vec<CyclotomicRing<Fp128, D>> {
    field_coeffs
        .par_chunks(D)
        .map(|chunk| {
            let mut coeffs = [Fp128::zero(); D];
            coeffs[..chunk.len()].copy_from_slice(chunk);
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect()
}
