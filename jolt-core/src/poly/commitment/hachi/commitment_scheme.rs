use std::{fmt, marker::PhantomData, sync::Arc, time::Instant};

use super::packed_layout::{choose_packed_bit_layout, PackedBitLayout, PackedBlockRange};
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
use hachi_pcs::protocol::commitment::utils::flat_matrix::{FlatMatrix, RingMatrixView};
use hachi_pcs::protocol::commitment::utils::linear::decompose_rows_i8;
use hachi_pcs::protocol::commitment::{
    compute_num_digits, compute_num_digits_fold, optimal_m_r_split, CommitmentConfig,
    DecompositionParams, Fp128BoundedCommitmentConfig, HachiCommitmentCore, HachiCommitmentLayout,
    RingCommitment,
};
use hachi_pcs::protocol::opening_point::BasisMode;
use hachi_pcs::protocol::proof::{HachiCommitmentHint, HachiProof};
use hachi_pcs::protocol::{HachiCommitmentScheme, HachiProverSetup, HachiVerifierSetup};
use hachi_pcs::CommitmentScheme as HachiCommitmentSchemeTrait;
use hachi_pcs::HachiError;
use hachi_pcs::{
    CanonicalField, DensePoly, FieldCore, FromSmallInt, HachiPolyOps, OneHotIndex, OneHotPoly,
};
use rayon::prelude::*;

/// Tile size for commit_inner A-matrix tiling. Each tile occupies
/// tile_size * n_a * sizeof(WideCyclotomicRing) ≈ tile_size * 16 KB.
/// Must fit in the shared last-level cache (Apple Silicon P-cluster L2 = 16 MB,
/// typical x86 L3 ≈ 32 MB). 1024 × 16 KB = 16 MB.
const COMMIT_TILE_SIZE: usize = 1024;

/// Maximum WideCyclotomicRing<Fp128x8i32> additions before i32 limb overflow.
/// Each limb starts ≤ 65535; i32 max ≈ 2.1B → safe up to ~32K additions.
const WIDE_RING_REDUCE_INTERVAL: usize = 32_000;

/// decompose_fold z_chunk fits in cache when inner_width * D * 4 ≤ ~16 MB.
/// With D=512: inner_width ≤ 8192.
const DECOMPOSE_FOLD_CACHE_THRESHOLD: usize = 8_192;

/// `Fp128BoundedCommitmentConfig` with `D = 256` instead of `512`.
/// All other parameters (N_A, N_B, N_D, decomposition) are
/// identical to the upstream `Fp128BoundedCommitmentConfig<LOG_COMMIT_BOUND>`.
/// CHALLENGE_WEIGHT is 23 (vs 19 at D=512) to maintain ≥128-bit challenge entropy.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp128Bounded256Config<const LOG_COMMIT_BOUND: u32>;

impl<const LOG_COMMIT_BOUND: u32> CommitmentConfig for Fp128Bounded256Config<LOG_COMMIT_BOUND> {
    const D: usize = 256;
    const N_A: usize = 1;
    const N_B: usize = 1;
    const N_D: usize = 1;
    const CHALLENGE_WEIGHT: usize = 23;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: LOG_COMMIT_BOUND,
            log_open_bound: if LOG_COMMIT_BOUND < 128 {
                Some(128)
            } else {
                None
            },
        }
    }

    fn commitment_layout(max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        let alpha = Self::D.trailing_zeros() as usize;
        let reduced_vars = max_num_vars.checked_sub(alpha).ok_or_else(|| {
            HachiError::InvalidSetup("max_num_vars is smaller than alpha".to_string())
        })?;
        if reduced_vars == 0 {
            return Err(HachiError::InvalidSetup(
                "max_num_vars must leave at least one outer variable".to_string(),
            ));
        }
        let (m_vars, r_vars) = optimal_m_r_split::<Self>(reduced_vars);
        HachiCommitmentLayout::new::<Self>(m_vars, r_vars, &Self::decomposition())
    }
}

pub type Fp128OneHot256Config = Fp128Bounded256Config<1>;

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
    packed_layout: PackedBitLayout,
    packed_block_cache: Arc<PackedBlockCache>,
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

#[derive(Clone)]
struct PackedBlockCache {
    coeffs: Arc<[u8]>,
    present_words: Arc<[u64]>,
    block_len: usize,
    present_words_per_block: usize,
    num_blocks: usize,
}

impl fmt::Debug for PackedBlockCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PackedBlockCache")
            .field("block_len", &self.block_len)
            .field("present_words_per_block", &self.present_words_per_block)
            .field("num_blocks", &self.num_blocks)
            .field("total_slots", &self.coeffs.len())
            .finish()
    }
}

impl PackedBlockCache {
    #[inline]
    fn coeffs_for_block(&self, block_idx: usize) -> &[u8] {
        let start = block_idx * self.block_len;
        &self.coeffs[start..start + self.block_len]
    }

    #[inline]
    fn present_words_for_block(&self, block_idx: usize) -> &[u64] {
        let start = block_idx * self.present_words_per_block;
        &self.present_words[start..start + self.present_words_per_block]
    }
}

/// Streaming view over the packed one-hot trace using a packed-specific
/// cycle/poly tile layout.
#[derive(Clone)]
struct JoltPackedPoly<const D: usize> {
    packed_layout: PackedBitLayout,
    packed_block_cache: Arc<PackedBlockCache>,
}

impl<const D: usize> JoltPackedPoly<D> {
    #[inline]
    fn num_blocks(&self) -> usize {
        self.packed_block_cache.num_blocks
    }

    #[inline]
    fn for_each_nonzero_in_block(&self, block_idx: usize, mut f: impl FnMut(usize, usize)) {
        let coeffs = self.packed_block_cache.coeffs_for_block(block_idx);
        let present_words = self.packed_block_cache.present_words_for_block(block_idx);
        for (word_idx, &word) in present_words.iter().enumerate() {
            let mut mask = word;
            while mask != 0 {
                let bit_idx = mask.trailing_zeros() as usize;
                let pos_in_block = word_idx * u64::BITS as usize + bit_idx;
                debug_assert!(pos_in_block < coeffs.len());
                f(pos_in_block, coeffs[pos_in_block] as usize);
                mask &= mask - 1;
            }
        }
    }
}

impl<const D: usize> HachiPolyOps<Fp128, D> for JoltPackedPoly<D> {
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        self.num_blocks() * self.packed_layout.block_len()
    }

    #[tracing::instrument(skip_all, name = "JoltPackedPoly::evaluate_ring")]
    fn evaluate_ring(&self, scalars: &[Fp128]) -> CyclotomicRing<Fp128, D> {
        let block_len = self.packed_layout.block_len();
        let total = (0..self.num_blocks())
            .into_par_iter()
            .map(|block_idx| {
                let mut acc = [Fp128::zero(); D];

                self.for_each_nonzero_in_block(block_idx, |pos_in_block, coeff_idx| {
                    let global_ring = block_idx * block_len + pos_in_block;
                    if global_ring < scalars.len() {
                        acc[coeff_idx] += scalars[global_ring];
                    }
                });
                acc
            })
            .reduce(
                || [Fp128::zero(); D],
                |mut a, b| {
                    for i in 0..D {
                        a[i] += b[i];
                    }
                    a
                },
            );

        CyclotomicRing::from_coefficients(total)
    }

    #[tracing::instrument(skip_all, name = "JoltPackedPoly::fold_blocks")]
    fn fold_blocks(&self, scalars: &[Fp128], _block_len: usize) -> Vec<CyclotomicRing<Fp128, D>> {
        let block_len = self.packed_layout.block_len();
        (0..self.num_blocks())
            .into_par_iter()
            .map(|block_idx| {
                let mut fold_acc = [Fp128::zero(); D];
                self.for_each_nonzero_in_block(block_idx, |pos_in_block, coeff_idx| {
                    if pos_in_block < block_len && pos_in_block < scalars.len() {
                        fold_acc[coeff_idx] += scalars[pos_in_block];
                    }
                });
                CyclotomicRing::from_coefficients(fold_acc)
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "JoltPackedPoly::evaluate_and_fold")]
    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[Fp128],
        fold_scalars: &[Fp128],
        _block_len: usize,
    ) -> (CyclotomicRing<Fp128, D>, Vec<CyclotomicRing<Fp128, D>>) {
        let folded = self.fold_blocks(fold_scalars, 0);
        let eval = folded
            .iter()
            .zip(eval_outer_scalars.iter())
            .fold(CyclotomicRing::<Fp128, D>::zero(), |acc, (f_i, s_i)| {
                acc + f_i.scale(s_i)
            });
        (eval, folded)
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

        let total_z = if inner_width <= DECOMPOSE_FOLD_CACHE_THRESHOLD {
            self.decompose_fold_small(challenges, block_len, delta)
        } else {
            self.decompose_fold_large(challenges, block_len, delta)
        };

        let q = (-Fp128::one()).to_canonical_u128() + 1;
        total_z
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
    ) -> Result<Vec<Vec<[i8; D]>>, HachiError> {
        let a_view = a_matrix.view::<D>();
        let n_a = a_view.num_rows();
        let block_len = self.packed_layout.block_len();
        let t_commit_inner = Instant::now();
        let (branch, results_flat): (&str, Vec<Vec<[i8; D]>>) = if block_len <= COMMIT_TILE_SIZE {
            (
                "small",
                self.commit_inner_small(
                    &a_view,
                    n_a,
                    num_digits_commit,
                    num_digits_open,
                    log_basis,
                ),
            )
        } else {
            (
                "tiled",
                self.commit_inner_tiled(
                    &a_view,
                    n_a,
                    num_digits_commit,
                    num_digits_open,
                    log_basis,
                ),
            )
        };
        eprintln!(
            "    [packed poly] commit_inner: {:.2}s (branch={branch}, block_len={block_len}, num_blocks={})",
            t_commit_inner.elapsed().as_secs_f64(),
            self.num_blocks(),
        );

        Ok(results_flat)
    }
}

impl<const D: usize> JoltPackedPoly<D> {
    #[inline]
    fn reduce_wide_accumulators(t_wide: &mut [WideCyclotomicRing<Fp128x8i32, D>]) {
        for tw in t_wide.iter_mut() {
            let reduced: CyclotomicRing<Fp128, D> = tw.reduce();
            *tw = WideCyclotomicRing::from_ring(&reduced);
        }
    }

    /// Fast path: a_wide_flat fits in L3. Precompute once, all tasks read from cache.
    fn commit_inner_small(
        &self,
        a_view: &RingMatrixView<'_, Fp128, D>,
        n_a: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Vec<Vec<[i8; D]>> {
        let block_len = self.packed_layout.block_len();

        let a_wide_flat: Vec<WideCyclotomicRing<Fp128x8i32, D>> = (0..block_len)
            .into_par_iter()
            .flat_map_iter(|pos| {
                let col = pos * num_digits_commit;
                (0..n_a).map(move |a| WideCyclotomicRing::from_ring(&a_view.row(a)[col]))
            })
            .collect();

        (0..self.num_blocks())
            .into_par_iter()
            .map(|block_idx| {
                let mut t_wide = vec![WideCyclotomicRing::<Fp128x8i32, D>::zero(); n_a];

                self.for_each_nonzero_in_block(block_idx, |pos_in_block, coeff_idx| {
                    let a_base = pos_in_block * n_a;
                    let a_slice = &a_wide_flat[a_base..a_base + n_a];
                    for a in 0..n_a {
                        a_slice[a].mul_by_monomial_sum_into(&mut t_wide[a], &[coeff_idx]);
                    }
                });

                let t: Vec<CyclotomicRing<Fp128, D>> =
                    t_wide.iter_mut().map(|w| w.reduce()).collect();
                decompose_rows_i8(&t, num_digits_open, log_basis)
            })
            .collect()
    }

    /// Tiled path for large block_len. Precomputes A-matrix tiles (shared
    /// read-only, total = block_len * n_a * sizeof(WideRing)), then processes
    /// blocks via par_iter so only O(num_threads) accumulators exist at a time.
    fn commit_inner_tiled(
        &self,
        a_view: &RingMatrixView<'_, Fp128, D>,
        n_a: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Vec<Vec<[i8; D]>> {
        let block_len = self.packed_layout.block_len();
        let tile_size = COMMIT_TILE_SIZE;
        let num_tiles = block_len.div_ceil(tile_size);

        let a_tiles: Vec<Vec<WideCyclotomicRing<Fp128x8i32, D>>> = (0..num_tiles)
            .map(|tile| {
                let tile_start = tile * tile_size;
                let tile_end = (tile_start + tile_size).min(block_len);
                let tile_len = tile_end - tile_start;
                (0..tile_len)
                    .into_par_iter()
                    .flat_map_iter(|local_pos| {
                        let pos = tile_start + local_pos;
                        let col = pos * num_digits_commit;
                        (0..n_a).map(move |a| WideCyclotomicRing::from_ring(&a_view.row(a)[col]))
                    })
                    .collect()
            })
            .collect();

        (0..self.num_blocks())
            .into_par_iter()
            .map(|block_idx| {
                let mut t_wide = vec![WideCyclotomicRing::<Fp128x8i32, D>::zero(); n_a];
                let mut entries_since_reduce = 0usize;

                self.for_each_nonzero_in_block(block_idx, |pos_in_block, coeff_idx| {
                    let tile_idx = pos_in_block / tile_size;
                    let tile_start = tile_idx * tile_size;
                    let local_pos = pos_in_block - tile_start;
                    let a_tile = &a_tiles[tile_idx];
                    let a_base = local_pos * n_a;
                    let a_slice = &a_tile[a_base..a_base + n_a];
                    for a in 0..n_a {
                        a_slice[a].mul_by_monomial_sum_into(&mut t_wide[a], &[coeff_idx]);
                    }

                    entries_since_reduce += 1;
                    if entries_since_reduce >= WIDE_RING_REDUCE_INTERVAL {
                        Self::reduce_wide_accumulators(&mut t_wide);
                        entries_since_reduce = 0;
                    }
                });

                let t: Vec<CyclotomicRing<Fp128, D>> =
                    t_wide.iter_mut().map(|w| w.reduce()).collect();
                decompose_rows_i8(&t, num_digits_open, log_basis)
            })
            .collect()
    }

    /// Small path: per-task z_chunk fits in L3, tree-reduce is cheap.
    fn decompose_fold_small(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        delta: usize,
    ) -> Vec<[i32; D]> {
        let inner_width = block_len * delta;

        (0..self.num_blocks())
            .into_par_iter()
            .map(|block_idx| {
                let mut z_chunk = vec![[0i32; D]; inner_width];

                if block_idx < challenges.len() {
                    let c_i = &challenges[block_idx];
                    if !c_i.positions.is_empty() {
                        Self::scatter_challenge(
                            self,
                            block_idx,
                            block_len,
                            delta,
                            c_i,
                            &mut z_chunk,
                        );
                    }
                }
                z_chunk
            })
            .reduce(
                || vec![[0i32; D]; inner_width],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        for (a_coeff, b_coeff) in ai.iter_mut().zip(bi.iter()) {
                            *a_coeff += b_coeff;
                        }
                    }
                    a
                },
            )
    }

    /// Large path: use thread-local accumulation (fold → reduce) to avoid
    /// allocating one huge buffer per task. Each Rayon thread reuses a single
    /// buffer across all its assigned tasks, then we merge only ~2×num_threads
    /// buffers instead of ~131K.
    fn decompose_fold_large(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        delta: usize,
    ) -> Vec<[i32; D]> {
        let inner_width = block_len * delta;

        (0..self.num_blocks())
            .into_par_iter()
            .fold(
                || vec![[0i32; D]; inner_width],
                |mut z_accum, block_idx| {
                    if block_idx < challenges.len() {
                        let c_i = &challenges[block_idx];
                        if !c_i.positions.is_empty() {
                            Self::scatter_challenge(
                                self,
                                block_idx,
                                block_len,
                                delta,
                                c_i,
                                &mut z_accum,
                            );
                        }
                    }
                    z_accum
                },
            )
            .reduce(
                || vec![[0i32; D]; inner_width],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        for (a_coeff, b_coeff) in ai.iter_mut().zip(bi.iter()) {
                            *a_coeff += b_coeff;
                        }
                    }
                    a
                },
            )
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn scatter_challenge(
        packed_poly: &Self,
        block_idx: usize,
        block_len: usize,
        delta: usize,
        c_i: &SparseChallenge,
        z_chunk: &mut [[i32; D]],
    ) {
        packed_poly.for_each_nonzero_in_block(block_idx, |pos_in_block, coeff_idx| {
            if pos_in_block < block_len {
                let base_j = pos_in_block * delta;
                for (&pos, &challenge_coeff) in c_i.positions.iter().zip(c_i.coeffs.iter()) {
                    let target = coeff_idx + pos as usize;
                    let (idx, sign) = if target < D {
                        (target, 1i32)
                    } else {
                        (target - D, -1i32)
                    };
                    z_chunk[base_j][idx] += sign * challenge_coeff as i32;
                }
            }
        });
    }
}

fn to_hachi_opening_point<const D: usize>(point: &[JoltFp128]) -> Vec<Fp128> {
    point.iter().rev().map(jolt_to_hachi).collect()
}

fn to_hachi_packed_opening_point<const D: usize>(
    opening_point: &[JoltFp128],
    rho: &[JoltFp128],
    packed_layout: PackedBitLayout,
) -> Vec<Fp128> {
    let reversed: Vec<Fp128> = opening_point.iter().rev().map(jolt_to_hachi).collect();
    let log_k = packed_layout.log_k();
    assert!(
        log_k <= reversed.len(),
        "packed opening point expects log_k <= num_vars (log_k={log_k}, num_vars={})",
        reversed.len()
    );
    let log_t = reversed.len() - log_k;
    let rho_le: Vec<Fp128> = rho.iter().rev().map(jolt_to_hachi).collect();
    packed_layout.reorder_packed_point(&reversed[..log_t], &reversed[log_t..], &rho_le)
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
/// rather than inheriting from the setup envelope.
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

fn choose_packed_layout_for_shape<const D: usize, Cfg: CommitmentConfig>(
    log_k: usize,
    log_t: usize,
    log_packed: usize,
) -> (PackedBitLayout, HachiCommitmentLayout) {
    let packed_layout = choose_packed_bit_layout::<D, Cfg>(log_k, log_t, log_packed);
    let hachi_layout = packed_layout.into_hachi_layout::<Cfg>();
    (packed_layout, hachi_layout)
}

fn choose_packed_layout_for_dims<const D: usize, Cfg: CommitmentConfig>(
    num_cycles: usize,
    num_polys: usize,
    onehot_k: usize,
) -> (PackedBitLayout, HachiCommitmentLayout) {
    assert!(
        num_cycles.is_power_of_two(),
        "packed Hachi layout expects num_cycles to be a power of two (got {num_cycles})"
    );
    assert!(
        onehot_k.is_power_of_two(),
        "packed Hachi layout expects onehot_k to be a power of two (got {onehot_k})"
    );
    let log_k = onehot_k.trailing_zeros() as usize;
    let log_t = num_cycles.trailing_zeros() as usize;
    let log_packed = num_polys.next_power_of_two().trailing_zeros() as usize;
    choose_packed_layout_for_shape::<D, Cfg>(log_k, log_t, log_packed)
}

fn compute_packed_setup_layouts<const D: usize, Cfg: CommitmentConfig>(
    max_log_t: usize,
    max_log_k: usize,
    log_packed: usize,
) -> [HachiCommitmentLayout; 2] {
    let advice_num_vars = max_log_k + max_log_t;
    let advice_layout = compute_advice_layout::<D, Cfg>(advice_num_vars);
    let (_, packed_layout) =
        choose_packed_layout_for_shape::<D, Cfg>(max_log_k, max_log_t, log_packed);
    if std::env::var_os("HACHI_SETUP_DIAGNOSTICS").is_some() {
        eprintln!(
            "[jolt hachi setup] max_log_t={max_log_t}, max_log_k={max_log_k}, log_packed={log_packed}"
        );
        eprintln!("  advice_layout={advice_layout:?}");
        eprintln!("  packed_layout={packed_layout:?}");
    }
    [advice_layout, packed_layout]
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

fn build_packed_block_cache<const D: usize, IndexFn>(
    index_of: &IndexFn,
    num_cycles: usize,
    num_polys: usize,
    packed_layout: PackedBitLayout,
) -> Arc<PackedBlockCache>
where
    IndexFn: Fn(usize, usize) -> Option<u8> + Sync,
{
    let t_cache = Instant::now();
    let block_len = packed_layout.block_len();
    let num_blocks = packed_layout.num_blocks();
    let present_words_per_block = block_len.div_ceil(u64::BITS as usize);
    let total_slots = num_blocks
        .checked_mul(block_len)
        .expect("packed block cache size overflow");
    let total_present_words = num_blocks
        .checked_mul(present_words_per_block)
        .expect("packed block presence size overflow");
    let mut coeffs = vec![0u8; total_slots];
    let mut present_words = vec![0u64; total_present_words];

    coeffs
        .par_chunks_mut(block_len)
        .zip(present_words.par_chunks_mut(present_words_per_block))
        .enumerate()
        .for_each(|(block_idx, (coeffs_block, present_words_block))| {
            let PackedBlockRange {
                cycle_start,
                cycle_end,
                poly_start,
                poly_end,
            } = packed_layout.block_range(block_idx, num_cycles, num_polys);
            for c in cycle_start..cycle_end {
                for p in poly_start..poly_end {
                    if let Some(k) = index_of(c, p) {
                        let packed = packed_layout.locate(c, p, k as usize);
                        debug_assert_eq!(packed.block_idx, block_idx);
                        coeffs_block[packed.pos_in_block] = packed.coeff_idx as u8;
                        let word_idx = packed.pos_in_block / u64::BITS as usize;
                        let bit_idx = packed.pos_in_block % u64::BITS as usize;
                        present_words_block[word_idx] |= 1u64 << bit_idx;
                    }
                }
            }
        });

    eprintln!(
        "    [packed poly] build_block_cache: {:.2}s (block_len={block_len}, num_blocks={num_blocks})",
        t_cache.elapsed().as_secs_f64(),
    );

    Arc::new(PackedBlockCache {
        coeffs: Arc::from(coeffs.into_boxed_slice()),
        present_words: Arc::from(present_words.into_boxed_slice()),
        block_len,
        present_words_per_block,
        num_blocks,
    })
}

fn build_packed_poly<const D: usize, IndexFn>(
    index_of: IndexFn,
    num_cycles: usize,
    num_polys: usize,
    packed_layout: PackedBitLayout,
    packed_block_cache: Option<Arc<PackedBlockCache>>,
) -> JoltPackedPoly<D>
where
    IndexFn: Fn(usize, usize) -> Option<u8> + Sync,
{
    let num_padded = num_polys.next_power_of_two();
    assert_eq!(
        num_padded,
        packed_layout.num_padded_polys(),
        "packed layout padding mismatch (expected {}, got {num_padded})",
        packed_layout.num_padded_polys()
    );
    let packed_block_cache = packed_block_cache.unwrap_or_else(|| {
        build_packed_block_cache::<D, _>(&index_of, num_cycles, num_polys, packed_layout)
    });
    assert_eq!(
        packed_block_cache.block_len,
        packed_layout.block_len(),
        "packed block cache block_len mismatch"
    );
    assert_eq!(
        packed_block_cache.num_blocks,
        packed_layout.num_blocks(),
        "packed block cache num_blocks mismatch"
    );

    JoltPackedPoly {
        packed_layout,
        packed_block_cache,
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

    fn setup_prover_from_shape(
        max_log_t: usize,
        max_log_k: usize,
        log_packed: Option<usize>,
    ) -> Self::ProverSetup {
        let setup_layouts =
            compute_packed_setup_layouts::<D, Cfg>(max_log_t, max_log_k, log_packed.unwrap_or(0));
        let (setup, _) = HachiCommitmentCore::setup_with_layouts::<Fp128, D, Cfg>(&setup_layouts)
            .expect("Hachi packed setup failed");
        ArkBridge(setup)
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
        let num_cycles = source
            .num_cycles()
            .expect("batch_commit requires lazy source");
        let onehot_k = source.onehot_k().unwrap();
        let num_polys = source.num_polys();
        let (packed_layout, batch_layout) =
            choose_packed_layout_for_dims::<D, Cfg>(num_cycles, num_polys, onehot_k);

        let index_fn = |c: usize, p: usize| source.onehot_index(c, p);
        let packed_poly =
            build_packed_poly::<D, _>(index_fn, num_cycles, num_polys, packed_layout, None);

        let (commitment, hachi_hint) =
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::commit(
                &packed_poly,
                &setup.0,
                &batch_layout,
            )
            .expect("Hachi packed poly commit failed");

        let hint = JoltHachiBatchHint {
            hachi_hint,
            packed_layout,
            packed_block_cache: packed_poly.packed_block_cache.clone(),
            num_packed_polys: num_polys,
            log_k: packed_layout.log_k(),
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

        let log_k = batch_hint.log_k;
        assert!(
            log_k <= opening_point.len(),
            "batch_prove log_k exceeds opening point length (log_k={log_k}, point_len={})",
            opening_point.len()
        );
        let packed_layout = batch_hint.packed_layout;
        let packed_hachi_layout = packed_layout.into_hachi_layout::<Cfg>();
        let packed_point = to_hachi_packed_opening_point::<D>(opening_point, &rho, packed_layout);

        let hachi_combined = jolt_to_hachi(&combined_claim);
        let packed_commitment = commitments[0];

        transcript.append_bytes(
            b"hachi_packed_claim",
            &hachi_combined.to_canonical_u128().to_le_bytes(),
        );

        let num_cycles = poly_source
            .num_cycles()
            .expect("batch_prove requires lazy source");
        let num_polys = poly_source.num_polys().unwrap();
        assert_eq!(
            num_polys, num_packed,
            "batch_prove packed poly count mismatch (commit={num_packed}, prove={num_polys})"
        );

        let index_fn = |c: usize, p: usize| poly_source.onehot_index(c, p);
        let packed_poly = build_packed_poly::<D, _>(
            index_fn,
            num_cycles,
            num_polys,
            packed_layout,
            Some(batch_hint.packed_block_cache.clone()),
        );

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
                &packed_hachi_layout,
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

        let log_k = proof.log_k as usize;
        if log_k > opening_point.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                opening_point.len(),
                log_k,
            ));
        }
        let log_t = opening_point.len() - log_k;
        let (packed_layout, packed_hachi_layout) =
            choose_packed_layout_for_shape::<D, Cfg>(log_k, log_t, selector_vars);
        let packed_point = to_hachi_packed_opening_point::<D>(opening_point, &rho, packed_layout);

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
            &packed_hachi_layout,
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
