use std::{array::from_fn, mem::size_of, slice::from_ref};

use super::packed_layout::{PackedBitLayout, SingletonLocateParams};
use super::wrappers::Fp128;
use hachi_pcs::algebra::fields::wide::Fp128x8i32;
use hachi_pcs::algebra::ring::sparse_challenge::SparseChallenge;
use hachi_pcs::algebra::ring::{CyclotomicRing, WideCyclotomicRing};
use hachi_pcs::protocol::commitment::utils::crt_ntt::NttSlotCache;
use hachi_pcs::protocol::commitment::utils::flat_matrix::{FlatMatrix, RingMatrixView};
use hachi_pcs::protocol::commitment::utils::linear::decompose_rows_i8;
use hachi_pcs::protocol::hachi_poly_ops::{CommitInnerWitness, DecomposeFoldWitness};
use hachi_pcs::HachiError;
use hachi_pcs::{CanonicalField, FieldCore, HachiPolyOps};
use rayon::prelude::*;

/// Tile size for commit_inner A-matrix tiling. Each tile occupies
/// tile_size * n_a * sizeof(WideCyclotomicRing) ≈ tile_size * 16 KB.
/// Must fit in the shared last-level cache (Apple Silicon P-cluster L2 = 16 MB,
/// typical x86 L3 ≈ 32 MB). 1024 × 16 KB = 16 MB.
#[allow(dead_code)]
const COMMIT_TILE_SIZE: usize = 1024;

/// Cache the full widened singleton A row only for smaller shapes. At
/// `sha2-chain` scale, a 512 MiB row cache causes enough memory pressure to
/// outweigh the saved bitmap scans.
const COMMIT_WIDE_ROW_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;

/// Cap the concurrently live widened block accumulators in the tiled singleton
/// path. Large traces otherwise keep one wide ring per block, which spills far
/// beyond LLC and turns the block sweep into a DRAM-heavy pass.
const COMMIT_WIDE_BATCH_MAX_BYTES: usize = 64 * 1024 * 1024;
const COMMIT_COLUMN_SWEEP_TILE_BYTES: usize = 1usize << 21;
const COMMIT_COLUMN_SWEEP_THRESHOLD: usize = 128;

/// Maximum WideCyclotomicRing<Fp128x8i32> additions before i32 limb overflow.
/// Each limb starts ≤ 65535; i32 max ≈ 2.1B → safe up to ~32K additions.
const WIDE_RING_REDUCE_INTERVAL: usize = 32_000;

/// Per-worker stripe size for the large-shape decompose_fold path. At D=64,
/// 4096 rows is 1 MiB of i32 output, which keeps merge working sets cacheable.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PackedOccupancySummary {
    nonempty_blocks: usize,
    min_nonzero: usize,
    p50_nonzero: usize,
    p95_nonzero: usize,
    max_nonzero: usize,
    buckets: [usize; 5],
}

#[cfg(test)]
impl PackedOccupancySummary {
    #[inline]
    fn ratio_for(nonzero: usize, block_len: usize) -> f64 {
        if block_len == 0 {
            0.0
        } else {
            nonzero as f64 / block_len as f64
        }
    }

    #[inline]
    pub(super) fn nonempty_blocks(&self) -> usize {
        self.nonempty_blocks
    }

    #[inline]
    pub(super) fn bucket_counts(&self) -> [usize; 5] {
        self.buckets
    }

    #[inline]
    pub(super) fn p50_ratio(&self, block_len: usize) -> f64 {
        Self::ratio_for(self.p50_nonzero, block_len)
    }

    #[inline]
    pub(super) fn p95_ratio(&self, block_len: usize) -> f64 {
        Self::ratio_for(self.p95_nonzero, block_len)
    }

    #[inline]
    pub(super) fn max_ratio(&self, block_len: usize) -> f64 {
        Self::ratio_for(self.max_nonzero, block_len)
    }
}

#[cfg(test)]
#[inline]
fn nearest_rank_count(
    sorted_counts: &[usize],
    percentile_numerator: usize,
    percentile_denominator: usize,
) -> usize {
    if sorted_counts.is_empty() {
        return 0;
    }
    let rank = (sorted_counts.len() * percentile_numerator).div_ceil(percentile_denominator);
    sorted_counts[rank.saturating_sub(1)]
}

#[cfg(test)]
pub(super) fn summarize_block_occupancy(
    nonzero_counts: &[u32],
    block_len: usize,
) -> PackedOccupancySummary {
    let mut sorted_counts = nonzero_counts
        .iter()
        .map(|&count| count as usize)
        .collect::<Vec<_>>();
    sorted_counts.sort_unstable();

    let mut buckets = [0usize; 5];
    let mut nonempty_blocks = 0usize;
    for &count in &sorted_counts {
        if count == 0 || block_len == 0 {
            buckets[0] += 1;
        } else if count * 4 <= block_len {
            buckets[1] += 1;
        } else if count * 2 <= block_len {
            buckets[2] += 1;
        } else if count * 4 <= block_len * 3 {
            buckets[3] += 1;
        } else {
            buckets[4] += 1;
        }
        if count != 0 {
            nonempty_blocks += 1;
        }
    }

    PackedOccupancySummary {
        nonempty_blocks,
        min_nonzero: sorted_counts.first().copied().unwrap_or(0),
        p50_nonzero: nearest_rank_count(&sorted_counts, 50, 100),
        p95_nonzero: nearest_rank_count(&sorted_counts, 95, 100),
        max_nonzero: sorted_counts.last().copied().unwrap_or(0),
        buckets,
    }
}

struct PackedCommitInnerOutput<const D: usize> {
    t_hat: Vec<Vec<[i8; D]>>,
    t: Vec<Vec<CyclotomicRing<Fp128, D>>>,
}

/// Streaming view over the packed one-hot trace. Instead of materializing
/// a block cache, holds the original index function and re-reads entries
/// on the fly via `packed_layout.locate_singleton()` / `locate()`.
///
/// `index_fn(cycle, poly) -> Option<u8>` returns the one-hot index for a
/// single (cycle, poly) pair.
///
/// `batch_fn(cycle, poly_start, buf)` fills `buf[0..len]` with the one-hot
/// indices for polys `poly_start..poly_start+len` at the given cycle,
/// amortizing per-cycle work (e.g. loading the trace entry once).
pub(super) struct JoltPackedPoly<F, B, const D: usize> {
    pub(super) packed_layout: PackedBitLayout,
    pub(super) index_fn: F,
    pub(super) batch_fn: B,
    pub(super) num_cycles: usize,
    pub(super) num_polys: usize,
}

impl<F: Clone, B: Clone, const D: usize> Clone for JoltPackedPoly<F, B, D> {
    fn clone(&self) -> Self {
        Self {
            packed_layout: self.packed_layout,
            index_fn: self.index_fn.clone(),
            batch_fn: self.batch_fn.clone(),
            num_cycles: self.num_cycles,
            num_polys: self.num_polys,
        }
    }
}

impl<
        F: Fn(usize, usize) -> Option<u8> + Sync,
        B: Fn(usize, usize, &mut [Option<u8>]) + Sync,
        const D: usize,
    > JoltPackedPoly<F, B, D>
{
    #[inline]
    fn num_blocks(&self) -> usize {
        self.packed_layout.num_blocks()
    }

    #[inline]
    fn is_singleton(&self) -> bool {
        self.packed_layout.lifted_coeff_bits() == 0
    }

    #[inline]
    fn can_use_commit_inner_fast_path(&self, n_a: usize, num_digits_commit: usize) -> bool {
        self.is_singleton() && n_a == 1 && num_digits_commit == 1
    }

    #[inline]
    fn can_use_commit_inner_column_sweep(&self) -> bool {
        let num_blocks = self.num_blocks();
        if num_blocks == 0 {
            return false;
        }
        let num_threads = rayon::current_num_threads().min(num_blocks).max(1);
        let blocks_per_thread = num_blocks.div_ceil(num_threads);
        blocks_per_thread > COMMIT_COLUMN_SWEEP_THRESHOLD
    }

    fn commit_inner_output(
        &self,
        a_view: &RingMatrixView<'_, Fp128, D>,
        n_a: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> PackedCommitInnerOutput<D> {
        if self.can_use_commit_inner_column_sweep() {
            self.commit_inner_column_sweep_output(
                a_view,
                n_a,
                num_digits_commit,
                num_digits_open,
                log_basis,
            )
        } else if self.can_use_commit_inner_fast_path(n_a, num_digits_commit) {
            self.commit_inner_fast_singleton_output(a_view, num_digits_open, log_basis)
        } else {
            self.commit_inner_generic_output(
                a_view,
                n_a,
                num_digits_commit,
                num_digits_open,
                log_basis,
            )
        }
    }

    #[inline]
    fn can_cache_commit_wide_row(&self) -> bool {
        self.packed_layout
            .block_len()
            .saturating_mul(size_of::<WideCyclotomicRing<Fp128x8i32, D>>())
            <= COMMIT_WIDE_ROW_CACHE_MAX_BYTES
    }

    #[inline]
    fn for_each_entry_in_block(&self, block_idx: usize, mut f: impl FnMut(usize, &[u8])) {
        let range = self
            .packed_layout
            .block_range(block_idx, self.num_cycles, self.num_polys);
        if self.is_singleton() {
            for c in range.cycle_start..range.cycle_end {
                for p in range.poly_start..range.poly_end {
                    if let Some(k) = (self.index_fn)(c, p) {
                        let (pos, coeff_idx) =
                            self.packed_layout.locate_singleton(c, p, k as usize);
                        f(pos, &[coeff_idx as u8]);
                    }
                }
            }
        } else {
            for c in range.cycle_start..range.cycle_end {
                for p in range.poly_start..range.poly_end {
                    if let Some(k) = (self.index_fn)(c, p) {
                        let pos = self.packed_layout.locate(c, p, k as usize);
                        debug_assert_eq!(pos.block_idx, block_idx);
                        f(pos.pos_in_block, &[pos.coeff_idx as u8]);
                    }
                }
            }
        }
    }

    #[inline]
    fn for_each_entry_in_block_range(
        &self,
        block_idx: usize,
        pos_start: usize,
        pos_end: usize,
        mut f: impl FnMut(usize, &[u8]),
    ) {
        let block_len = self.packed_layout.block_len();
        let pos_start = pos_start.min(block_len);
        let pos_end = pos_end.min(block_len);
        if pos_start >= pos_end {
            return;
        }
        let range = self
            .packed_layout
            .block_range(block_idx, self.num_cycles, self.num_polys);
        if self.is_singleton() {
            for c in range.cycle_start..range.cycle_end {
                for p in range.poly_start..range.poly_end {
                    if let Some(k) = (self.index_fn)(c, p) {
                        let (pos, coeff_idx) =
                            self.packed_layout.locate_singleton(c, p, k as usize);
                        if pos >= pos_start && pos < pos_end {
                            f(pos, &[coeff_idx as u8]);
                        }
                    }
                }
            }
        } else {
            for c in range.cycle_start..range.cycle_end {
                for p in range.poly_start..range.poly_end {
                    if let Some(k) = (self.index_fn)(c, p) {
                        let pos = self.packed_layout.locate(c, p, k as usize);
                        debug_assert_eq!(pos.block_idx, block_idx);
                        if pos.pos_in_block >= pos_start && pos.pos_in_block < pos_end {
                            f(pos.pos_in_block, &[pos.coeff_idx as u8]);
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn for_each_nonzero_in_block(&self, block_idx: usize, mut f: impl FnMut(usize, usize)) {
        self.for_each_entry_in_block(block_idx, |pos_in_block, coeffs| {
            for &coeff_idx in coeffs {
                f(pos_in_block, coeff_idx as usize);
            }
        });
    }
}

impl<
        F: Fn(usize, usize) -> Option<u8> + Clone + Send + Sync,
        B: Fn(usize, usize, &mut [Option<u8>]) + Clone + Send + Sync,
        const D: usize,
    > HachiPolyOps<Fp128, D> for JoltPackedPoly<F, B, D>
{
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

                self.for_each_entry_in_block(block_idx, |pos_in_block, coeffs| {
                    let global_ring = block_idx * block_len + pos_in_block;
                    if global_ring < scalars.len() {
                        let scalar = scalars[global_ring];
                        for &coeff_idx in coeffs {
                            acc[coeff_idx as usize] += scalar;
                        }
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
        if self.is_singleton() {
            self.fold_blocks_fast_singleton(scalars)
        } else {
            self.fold_blocks_generic(scalars)
        }
    }

    #[tracing::instrument(skip_all, name = "JoltPackedPoly::evaluate_and_fold")]
    fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[Fp128],
        fold_scalars: &[Fp128],
        _block_len: usize,
    ) -> (CyclotomicRing<Fp128, D>, Vec<CyclotomicRing<Fp128, D>>) {
        if self.is_singleton() {
            self.evaluate_and_fold_fast_singleton(eval_outer_scalars, fold_scalars)
        } else {
            self.evaluate_and_fold_generic(eval_outer_scalars, fold_scalars)
        }
    }

    #[tracing::instrument(skip_all, name = "JoltPackedPoly::decompose_fold")]
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        delta: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<Fp128, D> {
        let total_z = if self.is_singleton() {
            self.decompose_fold_fast_singleton(challenges, block_len, delta)
        } else {
            self.decompose_fold_generic(challenges, block_len, delta)
        };

        let q = (-Fp128::one()).to_canonical_u128() + 1;
        let z_pre = total_z
            .par_iter()
            .map(|arr| {
                let coeffs = from_fn(|k| {
                    let v = arr[k];
                    if v >= 0 {
                        Fp128::from_canonical_u128_reduced(v as u128)
                    } else {
                        Fp128::from_canonical_u128_reduced(q - ((-v) as u128))
                    }
                });
                CyclotomicRing::from_coefficients(coeffs)
            })
            .collect();
        let centered_inf_norm = total_z
            .iter()
            .flat_map(|row| row.iter())
            .map(|coeff| coeff.unsigned_abs())
            .max()
            .unwrap_or(0);
        DecomposeFoldWitness {
            z_pre,
            centered_coeffs: total_z,
            centered_inf_norm,
        }
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
        let output =
            self.commit_inner_output(&a_view, n_a, num_digits_commit, num_digits_open, log_basis);
        Ok(output.t_hat)
    }

    #[allow(non_snake_case)]
    #[tracing::instrument(skip_all, name = "JoltPackedPoly::commit_inner")]
    fn commit_inner_witness(
        &self,
        a_matrix: &FlatMatrix<Fp128>,
        _ntt_a: &NttSlotCache<D>,
        _block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<CommitInnerWitness<Fp128, D>, HachiError> {
        let a_view = a_matrix.view::<D>();
        let n_a = a_view.num_rows();
        let output =
            self.commit_inner_output(&a_view, n_a, num_digits_commit, num_digits_open, log_basis);
        Ok(CommitInnerWitness {
            t_hat: output.t_hat,
            t: output.t,
        })
    }
}

impl<
        F: Fn(usize, usize) -> Option<u8> + Sync,
        B: Fn(usize, usize, &mut [Option<u8>]) + Sync,
        const D: usize,
    > JoltPackedPoly<F, B, D>
{
    #[inline(always)]
    fn accumulate_rotated_i32(dst: &mut [i32; D], rotated: &[i32; D]) {
        for (dst_coeff, &rotated_coeff) in dst.iter_mut().zip(rotated.iter()) {
            *dst_coeff += rotated_coeff;
        }
    }

    #[inline(always)]
    fn fill_rotated_challenge_i32(table: &mut [[i32; D]], challenge: &SparseChallenge) {
        debug_assert!(D.is_power_of_two());
        debug_assert!(table.len() >= D);

        let mut dense = [0i32; D];
        for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
            dense[pos as usize] = coeff as i32;
        }

        for (coeff_idx, row) in table.iter_mut().enumerate().take(D) {
            let split = D - coeff_idx;
            row[coeff_idx..D].copy_from_slice(&dense[..split]);
            for (dst, src) in row[..coeff_idx].iter_mut().zip(dense[split..].iter()) {
                *dst = -*src;
            }
        }
    }

    #[inline]
    fn reduce_wide_accumulators(t_wide: &mut [WideCyclotomicRing<Fp128x8i32, D>]) {
        for tw in t_wide.iter_mut() {
            let reduced: CyclotomicRing<Fp128, D> = tw.reduce();
            *tw = WideCyclotomicRing::from_ring(&reduced);
        }
    }

    #[inline]
    fn reduce_wide_accumulator(t_wide: &mut WideCyclotomicRing<Fp128x8i32, D>) {
        let reduced: CyclotomicRing<Fp128, D> = t_wide.reduce();
        *t_wide = WideCyclotomicRing::from_ring(&reduced);
    }

    #[inline(always)]
    fn wide_ring_coeffs(ring: &WideCyclotomicRing<Fp128x8i32, D>) -> &[Fp128x8i32; D] {
        // SAFETY: `WideCyclotomicRing` is `#[repr(transparent)]` over `[W; D]`,
        // and here `W = Fp128x8i32`, so the layouts are identical.
        unsafe { &*(ring as *const WideCyclotomicRing<Fp128x8i32, D> as *const [Fp128x8i32; D]) }
    }

    #[inline(always)]
    fn wide_ring_coeffs_mut(ring: &mut WideCyclotomicRing<Fp128x8i32, D>) -> &mut [Fp128x8i32; D] {
        // SAFETY: `WideCyclotomicRing` is `#[repr(transparent)]` over `[W; D]`,
        // and here `W = Fp128x8i32`, so the layouts are identical.
        unsafe { &mut *(ring as *mut WideCyclotomicRing<Fp128x8i32, D> as *mut [Fp128x8i32; D]) }
    }

    #[inline(always)]
    fn shift_accumulate_into_fast(
        src: &WideCyclotomicRing<Fp128x8i32, D>,
        dst: &mut WideCyclotomicRing<Fp128x8i32, D>,
        k: usize,
    ) {
        if k >= D {
            src.shift_accumulate_into(dst, k);
            return;
        }

        let src_coeffs = Self::wide_ring_coeffs(src);
        let dst_coeffs = Self::wide_ring_coeffs_mut(dst);
        if k == 0 {
            for (dst_coeff, src_coeff) in dst_coeffs.iter_mut().zip(src_coeffs.iter()) {
                *dst_coeff += *src_coeff;
            }
            return;
        }

        let split = D - k;
        let (src_lo, src_hi) = src_coeffs.split_at(split);
        let (dst_lo, dst_hi) = dst_coeffs.split_at_mut(k);
        for (dst_coeff, src_coeff) in dst_hi.iter_mut().zip(src_lo.iter()) {
            *dst_coeff += *src_coeff;
        }
        for (dst_coeff, src_coeff) in dst_lo.iter_mut().zip(src_hi.iter()) {
            *dst_coeff -= *src_coeff;
        }
    }

    fn compute_fold_block_generic(
        &self,
        block_idx: usize,
        scalars: &[Fp128],
        idx_buf: &mut Vec<Option<u8>>,
    ) -> CyclotomicRing<Fp128, D> {
        let range = self
            .packed_layout
            .block_range(block_idx, self.num_cycles, self.num_polys);
        if range.poly_start >= range.poly_end || range.cycle_start >= range.cycle_end {
            return CyclotomicRing::zero();
        }

        let poly_len = range.poly_end - range.poly_start;
        if idx_buf.len() < poly_len {
            idx_buf.resize(poly_len, None);
        }
        let idx_buf = &mut idx_buf[..poly_len];
        let mut fold_acc = [Fp128::zero(); D];

        for cycle_idx in range.cycle_start..range.cycle_end {
            (self.batch_fn)(cycle_idx, range.poly_start, idx_buf);
            for (local_poly_idx, slot) in idx_buf.iter().enumerate() {
                if let Some(k) = *slot {
                    let poly_idx = range.poly_start + local_poly_idx;
                    let pos = self.packed_layout.locate(cycle_idx, poly_idx, k as usize);
                    debug_assert_eq!(pos.block_idx, block_idx);
                    if pos.pos_in_block < scalars.len() {
                        fold_acc[pos.coeff_idx] += scalars[pos.pos_in_block];
                    }
                }
            }
        }

        CyclotomicRing::from_coefficients(fold_acc)
    }

    fn compute_fold_block_fast_singleton(
        &self,
        block_idx: usize,
        scalars: &[Fp128],
        params: SingletonLocateParams,
        poly_shifted: &[usize],
        idx_buf: &mut Vec<Option<u8>>,
    ) -> CyclotomicRing<Fp128, D> {
        let range = self
            .packed_layout
            .block_range(block_idx, self.num_cycles, self.num_polys);
        if range.poly_start >= range.poly_end || range.cycle_start >= range.cycle_end {
            return CyclotomicRing::zero();
        }

        let poly_offsets = &poly_shifted[range.poly_start..range.poly_end];
        if idx_buf.len() < poly_offsets.len() {
            idx_buf.resize(poly_offsets.len(), None);
        }
        let idx_buf = &mut idx_buf[..poly_offsets.len()];
        let mut fold_acc = [Fp128::zero(); D];

        for cycle_idx in range.cycle_start..range.cycle_end {
            (self.batch_fn)(cycle_idx, range.poly_start, idx_buf);
            let cycle_shifted = params.cycle_shifted(cycle_idx);
            for (slot, &poly_offset) in idx_buf.iter().zip(poly_offsets.iter()) {
                if let Some(k) = *slot {
                    let k = k as usize;
                    let coeff_idx = k & params.addr_coeff_mask;
                    let addr_inner = k >> params.addr_coeff_bits;
                    let pos_in_block = addr_inner | cycle_shifted | poly_offset;
                    if pos_in_block < scalars.len() {
                        fold_acc[coeff_idx] += scalars[pos_in_block];
                    }
                }
            }
        }

        CyclotomicRing::from_coefficients(fold_acc)
    }

    pub(super) fn fold_blocks_generic(&self, scalars: &[Fp128]) -> Vec<CyclotomicRing<Fp128, D>> {
        (0..self.num_blocks())
            .into_par_iter()
            .map_init(Vec::<Option<u8>>::new, |idx_buf, block_idx| {
                self.compute_fold_block_generic(block_idx, scalars, idx_buf)
            })
            .collect()
    }

    pub(super) fn fold_blocks_fast_singleton(
        &self,
        scalars: &[Fp128],
    ) -> Vec<CyclotomicRing<Fp128, D>> {
        let params = self.packed_layout.singleton_params();
        let poly_shifted: Vec<usize> = (0..self.num_polys)
            .map(|poly_idx| params.poly_shifted(poly_idx))
            .collect();

        (0..self.num_blocks())
            .into_par_iter()
            .map_init(Vec::<Option<u8>>::new, |idx_buf, block_idx| {
                self.compute_fold_block_fast_singleton(
                    block_idx,
                    scalars,
                    params,
                    &poly_shifted,
                    idx_buf,
                )
            })
            .collect()
    }

    pub(super) fn evaluate_and_fold_generic(
        &self,
        eval_outer_scalars: &[Fp128],
        fold_scalars: &[Fp128],
    ) -> (CyclotomicRing<Fp128, D>, Vec<CyclotomicRing<Fp128, D>>) {
        let mut folded = vec![CyclotomicRing::<Fp128, D>::zero(); self.num_blocks()];
        let eval = folded
            .par_iter_mut()
            .enumerate()
            .map_init(
                Vec::<Option<u8>>::new,
                |idx_buf, (block_idx, folded_slot)| {
                    let folded_block =
                        self.compute_fold_block_generic(block_idx, fold_scalars, idx_buf);
                    let eval_contrib = if block_idx < eval_outer_scalars.len() {
                        folded_block.scale(&eval_outer_scalars[block_idx])
                    } else {
                        CyclotomicRing::zero()
                    };
                    *folded_slot = folded_block;
                    eval_contrib
                },
            )
            .reduce(CyclotomicRing::<Fp128, D>::zero, |acc, contrib| {
                acc + contrib
            });
        (eval, folded)
    }

    pub(super) fn evaluate_and_fold_fast_singleton(
        &self,
        eval_outer_scalars: &[Fp128],
        fold_scalars: &[Fp128],
    ) -> (CyclotomicRing<Fp128, D>, Vec<CyclotomicRing<Fp128, D>>) {
        let params = self.packed_layout.singleton_params();
        let poly_shifted: Vec<usize> = (0..self.num_polys)
            .map(|poly_idx| params.poly_shifted(poly_idx))
            .collect();
        let mut folded = vec![CyclotomicRing::<Fp128, D>::zero(); self.num_blocks()];
        let eval = folded
            .par_iter_mut()
            .enumerate()
            .map_init(
                Vec::<Option<u8>>::new,
                |idx_buf, (block_idx, folded_slot)| {
                    let folded_block = self.compute_fold_block_fast_singleton(
                        block_idx,
                        fold_scalars,
                        params,
                        &poly_shifted,
                        idx_buf,
                    );
                    let eval_contrib = if block_idx < eval_outer_scalars.len() {
                        folded_block.scale(&eval_outer_scalars[block_idx])
                    } else {
                        CyclotomicRing::zero()
                    };
                    *folded_slot = folded_block;
                    eval_contrib
                },
            )
            .reduce(CyclotomicRing::<Fp128, D>::zero, |acc, contrib| {
                acc + contrib
            });
        (eval, folded)
    }

    fn commit_inner_generic_output(
        &self,
        a_view: &RingMatrixView<'_, Fp128, D>,
        n_a: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> PackedCommitInnerOutput<D> {
        let block_len = self.packed_layout.block_len();
        let a_wide_flat: Vec<WideCyclotomicRing<Fp128x8i32, D>> = (0..block_len)
            .into_par_iter()
            .flat_map_iter(|pos| {
                let col = pos * num_digits_commit;
                (0..n_a).map(move |a| WideCyclotomicRing::from_ring(&a_view.row(a)[col]))
            })
            .collect();

        let per_block = (0..self.num_blocks())
            .into_par_iter()
            .map(|block_idx| {
                let mut t_wide = vec![WideCyclotomicRing::<Fp128x8i32, D>::zero(); n_a];
                let mut entries_since_reduce = 0usize;

                self.for_each_entry_in_block(block_idx, |pos_in_block, coeffs| {
                    let a_base = pos_in_block * n_a;
                    let a_slice = &a_wide_flat[a_base..a_base + n_a];
                    for a in 0..n_a {
                        for &coeff_idx in coeffs {
                            Self::shift_accumulate_into_fast(
                                &a_slice[a],
                                &mut t_wide[a],
                                coeff_idx as usize,
                            );
                        }
                    }

                    entries_since_reduce += coeffs.len();
                    if entries_since_reduce >= WIDE_RING_REDUCE_INTERVAL {
                        Self::reduce_wide_accumulators(&mut t_wide);
                        entries_since_reduce = 0;
                    }
                });

                let t: Vec<CyclotomicRing<Fp128, D>> =
                    t_wide.iter_mut().map(|w| w.reduce()).collect();
                let t_hat = decompose_rows_i8(&t, num_digits_open, log_basis);
                (t, t_hat)
            })
            .collect::<Vec<_>>();
        let (t, t_hat): (Vec<_>, Vec<_>) = per_block.into_iter().unzip();

        PackedCommitInnerOutput { t_hat, t }
    }

    fn commit_inner_column_sweep_output(
        &self,
        a_view: &RingMatrixView<'_, Fp128, D>,
        n_a: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> PackedCommitInnerOutput<D> {
        let num_blocks = self.num_blocks();
        let block_len = self.packed_layout.block_len();
        let wide_ring_bytes = size_of::<WideCyclotomicRing<Fp128x8i32, D>>().max(1);
        let accum_bytes = n_a.saturating_mul(wide_ring_bytes);
        let block_tile = if accum_bytes == 0 {
            num_blocks.max(1)
        } else {
            (COMMIT_COLUMN_SWEEP_TILE_BYTES / accum_bytes).max(1)
        };
        let num_threads = rayon::current_num_threads().min(num_blocks).max(1);
        let blocks_per_thread = num_blocks.div_ceil(num_threads);

        let thread_results = (0..num_threads)
            .into_par_iter()
            .map(|thread_idx| {
                let block_start = thread_idx * blocks_per_thread;
                let block_end = (block_start + blocks_per_thread).min(num_blocks);
                if block_start >= block_end {
                    return (
                        block_start,
                        Vec::<Vec<CyclotomicRing<Fp128, D>>>::new(),
                        Vec::<Vec<[i8; D]>>::new(),
                    );
                }

                let my_count = block_end - block_start;
                let zero_t = vec![CyclotomicRing::<Fp128, D>::zero(); n_a];
                let zero_t_hat = decompose_rows_i8(&zero_t, num_digits_open, log_basis);
                let mut t = vec![zero_t.clone(); my_count];
                let mut t_hat = vec![zero_t_hat.clone(); my_count];

                for tile_start in (0..my_count).step_by(block_tile) {
                    let tile_end = (tile_start + block_tile).min(my_count);
                    let tile_len = tile_end - tile_start;
                    let mut position_entries: Vec<Vec<(u32, u8)>> = vec![Vec::new(); block_len];
                    let mut active_positions = Vec::new();

                    for local_block in 0..tile_len {
                        let block_idx = block_start + tile_start + local_block;
                        self.for_each_entry_in_block(block_idx, |pos_in_block, coeffs| {
                            let bucket = &mut position_entries[pos_in_block];
                            if bucket.is_empty() {
                                active_positions.push(pos_in_block);
                            }
                            for &coeff_idx in coeffs {
                                bucket.push((local_block as u32, coeff_idx));
                            }
                        });
                    }

                    if active_positions.is_empty() {
                        continue;
                    }
                    active_positions.sort_unstable();

                    let mut accumulators =
                        vec![WideCyclotomicRing::<Fp128x8i32, D>::zero(); tile_len * n_a];
                    let mut adds_since_reduce = vec![0usize; tile_len];

                    for &pos_in_block in active_positions.iter() {
                        let entries = &position_entries[pos_in_block];
                        let col = pos_in_block * num_digits_commit;
                        for a_idx in 0..n_a {
                            let a_wide = WideCyclotomicRing::from_ring(&a_view.row(a_idx)[col]);
                            for &(local_block, coeff_idx) in entries.iter() {
                                let accum_idx = local_block as usize * n_a + a_idx;
                                Self::shift_accumulate_into_fast(
                                    &a_wide,
                                    &mut accumulators[accum_idx],
                                    coeff_idx as usize,
                                );
                            }
                        }

                        for &(local_block, _) in entries.iter() {
                            let local_block = local_block as usize;
                            adds_since_reduce[local_block] += 1;
                            if adds_since_reduce[local_block] >= WIDE_RING_REDUCE_INTERVAL {
                                let accum_start = local_block * n_a;
                                Self::reduce_wide_accumulators(
                                    &mut accumulators[accum_start..accum_start + n_a],
                                );
                                adds_since_reduce[local_block] = 0;
                            }
                        }
                    }

                    let t_chunk: Vec<CyclotomicRing<Fp128, D>> = accumulators
                        .iter_mut()
                        .map(|accumulator| accumulator.reduce())
                        .collect();
                    let t_hat_chunk = decompose_rows_i8(&t_chunk, num_digits_open, log_basis);

                    for local_block in 0..tile_len {
                        let block_offset = tile_start + local_block;
                        let ring_start = local_block * n_a;
                        let ring_end = ring_start + n_a;
                        let digit_start = ring_start * num_digits_open;
                        let digit_end = ring_end * num_digits_open;
                        t[block_offset] = t_chunk[ring_start..ring_end].to_vec();
                        t_hat[block_offset] = t_hat_chunk[digit_start..digit_end].to_vec();
                    }
                }

                (block_start, t, t_hat)
            })
            .collect::<Vec<_>>();

        let mut t = vec![Vec::new(); num_blocks];
        let mut t_hat = vec![Vec::new(); num_blocks];
        for (block_start, t_chunk, t_hat_chunk) in thread_results {
            for (offset, (t_block, t_hat_block)) in
                t_chunk.into_iter().zip(t_hat_chunk.into_iter()).enumerate()
            {
                let block_idx = block_start + offset;
                t[block_idx] = t_block;
                t_hat[block_idx] = t_hat_block;
            }
        }

        PackedCommitInnerOutput { t_hat, t }
    }

    #[allow(dead_code)]
    pub(super) fn commit_inner_generic(
        &self,
        a_view: &RingMatrixView<'_, Fp128, D>,
        n_a: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Vec<Vec<[i8; D]>> {
        let output = self.commit_inner_generic_output(
            a_view,
            n_a,
            num_digits_commit,
            num_digits_open,
            log_basis,
        );
        output.t_hat
    }

    #[allow(dead_code)]
    pub(super) fn commit_inner_column_sweep(
        &self,
        a_view: &RingMatrixView<'_, Fp128, D>,
        n_a: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Vec<Vec<[i8; D]>> {
        let output = self.commit_inner_column_sweep_output(
            a_view,
            n_a,
            num_digits_commit,
            num_digits_open,
            log_basis,
        );
        output.t_hat
    }

    #[allow(dead_code)]
    pub(super) fn commit_inner_fast_singleton(
        &self,
        a_view: &RingMatrixView<'_, Fp128, D>,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Vec<Vec<[i8; D]>> {
        let output = self.commit_inner_fast_singleton_output(a_view, num_digits_open, log_basis);
        output.t_hat
    }

    fn commit_inner_fast_singleton_output(
        &self,
        a_view: &RingMatrixView<'_, Fp128, D>,
        num_digits_open: usize,
        log_basis: u32,
    ) -> PackedCommitInnerOutput<D> {
        if self.can_cache_commit_wide_row() {
            self.commit_inner_fast_singleton_cached_row_output(a_view, num_digits_open, log_basis)
        } else {
            self.commit_inner_fast_singleton_tiled_output(a_view, num_digits_open, log_basis)
        }
    }

    fn commit_inner_fast_singleton_cached_row_output(
        &self,
        a_view: &RingMatrixView<'_, Fp128, D>,
        num_digits_open: usize,
        log_basis: u32,
    ) -> PackedCommitInnerOutput<D> {
        let block_len = self.packed_layout.block_len();
        let a_row = a_view.row(0);
        let a_wide_row: Vec<WideCyclotomicRing<Fp128x8i32, D>> = (0..block_len)
            .into_par_iter()
            .map(|pos| WideCyclotomicRing::from_ring(&a_row[pos]))
            .collect();

        let per_block = (0..self.num_blocks())
            .into_par_iter()
            .map(|block_idx| {
                let mut t_wide = WideCyclotomicRing::<Fp128x8i32, D>::zero();
                let mut entries_since_reduce = 0usize;

                self.for_each_nonzero_in_block(block_idx, |pos_in_block, coeff_idx| {
                    Self::shift_accumulate_into_fast(
                        &a_wide_row[pos_in_block],
                        &mut t_wide,
                        coeff_idx,
                    );
                    entries_since_reduce += 1;
                    if entries_since_reduce >= WIDE_RING_REDUCE_INTERVAL {
                        Self::reduce_wide_accumulator(&mut t_wide);
                        entries_since_reduce = 0;
                    }
                });

                let t: CyclotomicRing<Fp128, D> = t_wide.reduce();
                let t_hat = decompose_rows_i8(from_ref(&t), num_digits_open, log_basis);
                (vec![t], t_hat)
            })
            .collect::<Vec<_>>();
        let (t, t_hat): (Vec<_>, Vec<_>) = per_block.into_iter().unzip();

        PackedCommitInnerOutput { t_hat, t }
    }

    fn commit_inner_fast_singleton_tiled_output(
        &self,
        a_view: &RingMatrixView<'_, Fp128, D>,
        num_digits_open: usize,
        log_basis: u32,
    ) -> PackedCommitInnerOutput<D> {
        let block_len = self.packed_layout.block_len();
        let tile_size = COMMIT_TILE_SIZE.min(block_len.max(1));
        let num_tiles = block_len.div_ceil(tile_size);
        let a_row = a_view.row(0);
        let wide_ring_bytes = size_of::<WideCyclotomicRing<Fp128x8i32, D>>().max(1);
        let chunk_parallelism = rayon::current_num_threads()
            .max(1)
            .min(self.num_blocks().max(1));
        let target_active_blocks = (COMMIT_WIDE_BATCH_MAX_BYTES / wide_ring_bytes)
            .max(chunk_parallelism)
            .min(self.num_blocks().max(1));
        let blocks_per_chunk = target_active_blocks.div_ceil(chunk_parallelism).max(1);
        let blocks_per_batch = (blocks_per_chunk * chunk_parallelism).min(self.num_blocks().max(1));
        let mut t = vec![Vec::new(); self.num_blocks()];
        let mut t_hat = vec![Vec::new(); self.num_blocks()];

        debug_assert!(
            tile_size <= WIDE_RING_REDUCE_INTERVAL,
            "tiled singleton commit assumes each tile fits within one unreduced window"
        );

        for batch_block_start in (0..self.num_blocks()).step_by(blocks_per_batch) {
            let batch_block_end = (batch_block_start + blocks_per_batch).min(self.num_blocks());
            let batch_chunk_starts: Vec<usize> = (batch_block_start..batch_block_end)
                .step_by(blocks_per_chunk)
                .collect();
            let mut batch_t_wide: Vec<Vec<WideCyclotomicRing<Fp128x8i32, D>>> = batch_chunk_starts
                .iter()
                .map(|&chunk_start| {
                    let chunk_end = (chunk_start + blocks_per_chunk).min(batch_block_end);
                    vec![WideCyclotomicRing::<Fp128x8i32, D>::zero(); chunk_end - chunk_start]
                })
                .collect();
            let mut batch_entries_since_reduce: Vec<Vec<usize>> = batch_chunk_starts
                .iter()
                .map(|&chunk_start| {
                    let chunk_end = (chunk_start + blocks_per_chunk).min(batch_block_end);
                    vec![0usize; chunk_end - chunk_start]
                })
                .collect();

            for tile_idx in 0..num_tiles {
                let tile_start = tile_idx * tile_size;
                let tile_end = (tile_start + tile_size).min(block_len);
                let a_tile: Vec<WideCyclotomicRing<Fp128x8i32, D>> = (tile_start..tile_end)
                    .into_par_iter()
                    .map(|pos| WideCyclotomicRing::from_ring(&a_row[pos]))
                    .collect();

                batch_chunk_starts
                    .par_iter()
                    .zip(batch_t_wide.par_iter_mut())
                    .zip(batch_entries_since_reduce.par_iter_mut())
                    .for_each(|((&chunk_start, t_wide_chunk), entries_chunk)| {
                        for (chunk_offset, (t_wide, entries_since_reduce)) in t_wide_chunk
                            .iter_mut()
                            .zip(entries_chunk.iter_mut())
                            .enumerate()
                        {
                            let block_idx = chunk_start + chunk_offset;
                            self.for_each_entry_in_block_range(
                                block_idx,
                                tile_start,
                                tile_end,
                                |pos_in_block, coeffs| {
                                    let local_pos = pos_in_block - tile_start;
                                    for &coeff_idx in coeffs {
                                        Self::shift_accumulate_into_fast(
                                            &a_tile[local_pos],
                                            t_wide,
                                            coeff_idx as usize,
                                        );
                                        *entries_since_reduce += 1;
                                        if *entries_since_reduce >= WIDE_RING_REDUCE_INTERVAL {
                                            Self::reduce_wide_accumulator(t_wide);
                                            *entries_since_reduce = 0;
                                        }
                                    }
                                },
                            );
                        }
                    });
            }

            let batch_outputs = batch_chunk_starts
                .into_par_iter()
                .zip(batch_t_wide.into_par_iter())
                .map(|(chunk_start, mut t_wide_chunk)| {
                    let t_chunk: Vec<CyclotomicRing<Fp128, D>> = t_wide_chunk
                        .iter_mut()
                        .map(|t_wide| t_wide.reduce())
                        .collect();
                    let t_hat_flat = decompose_rows_i8(&t_chunk, num_digits_open, log_basis);
                    let t_hat_chunk: Vec<Vec<[i8; D]>> = t_hat_flat
                        .chunks(num_digits_open)
                        .map(|digits| digits.to_vec())
                        .collect();
                    (chunk_start, t_chunk, t_hat_chunk)
                })
                .collect::<Vec<_>>();

            for (chunk_start, t_chunk, t_hat_chunk) in batch_outputs {
                for (offset, (t_i, t_hat_i)) in t_chunk.into_iter().zip(t_hat_chunk).enumerate() {
                    let block_idx = chunk_start + offset;
                    t[block_idx] = vec![t_i];
                    t_hat[block_idx] = t_hat_i;
                }
            }
        }

        PackedCommitInnerOutput { t_hat, t }
    }

    pub(super) fn decompose_fold_generic(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        delta: usize,
    ) -> Vec<[i32; D]> {
        if block_len == 0 {
            return Vec::new();
        }

        let block_limit = challenges.len().min(self.num_blocks());
        let num_threads = rayon::current_num_threads().max(1);
        let blocks_per_chunk = block_limit.div_ceil(num_threads).max(1);
        let num_chunks = block_limit.div_ceil(blocks_per_chunk);

        let rows = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let chunk_start = chunk_idx * blocks_per_chunk;
                let chunk_end = (chunk_start + blocks_per_chunk).min(block_limit);
                let mut local_rows = vec![[0i32; D]; block_len];
                let mut rotated = [[0i32; D]; D];
                let mut idx_buf = vec![None::<u8>; self.num_polys];
                for block_idx in chunk_start..chunk_end {
                    let challenge = &challenges[block_idx];
                    if challenge.positions.is_empty() {
                        continue;
                    }
                    Self::fill_rotated_challenge_i32(&mut rotated, challenge);
                    let range =
                        self.packed_layout
                            .block_range(block_idx, self.num_cycles, self.num_polys);
                    if range.poly_start >= range.poly_end || range.cycle_start >= range.cycle_end {
                        continue;
                    }
                    let poly_len = range.poly_end - range.poly_start;
                    let buf = &mut idx_buf[..poly_len];
                    for c in range.cycle_start..range.cycle_end {
                        (self.batch_fn)(c, range.poly_start, buf);
                        for (i, p) in (range.poly_start..range.poly_end).enumerate() {
                            if let Some(k) = buf[i] {
                                let pos = self.packed_layout.locate(c, p, k as usize);
                                debug_assert_eq!(pos.block_idx, block_idx);
                                Self::accumulate_rotated_i32(
                                    &mut local_rows[pos.pos_in_block],
                                    &rotated[pos.coeff_idx],
                                );
                            }
                        }
                    }
                }
                local_rows
            })
            .reduce(
                || vec![[0i32; D]; block_len],
                |mut a, b| {
                    for (a_row, b_row) in a.iter_mut().zip(b.iter()) {
                        for d in 0..D {
                            a_row[d] += b_row[d];
                        }
                    }
                    a
                },
            );

        Self::expand_decompose_fold_rows(rows, delta)
    }

    pub(super) fn decompose_fold_fast_singleton(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        delta: usize,
    ) -> Vec<[i32; D]> {
        if block_len == 0 {
            return Vec::new();
        }

        let block_limit = challenges.len().min(self.num_blocks());
        let num_threads = rayon::current_num_threads().max(1);
        let params = self.packed_layout.singleton_params();
        let cycle_inner_span = params.cycle_inner_mask + 1;
        let cycle_outer_bits = self.packed_layout.cycle_outer_bits();
        let cycle_outer_count = self.num_cycles.div_ceil(cycle_inner_span);
        let poly_group_span = params.poly_inner_mask + 1;
        let poly_shifted: Vec<usize> = (0..self.num_polys)
            .map(|p| params.poly_shifted(p))
            .collect();
        let cycle_shifted: Vec<usize> = (0..cycle_inner_span)
            .map(|c| params.cycle_shifted(c))
            .collect();
        let poly_groups: Vec<(usize, usize, usize)> = (0..self.num_polys)
            .step_by(poly_group_span.max(1))
            .map(|group_start| {
                let group_end = (group_start + poly_group_span).min(self.num_polys);
                let poly_outer = group_start / poly_group_span.max(1);
                let block_base = poly_outer << cycle_outer_bits;
                (group_start, group_end, block_base)
            })
            .collect();

        let addr_inner_bits = self.packed_layout.addr_inner_bits();
        let poly_inner_bits = self.packed_layout.poly_inner_bits();
        let positions_per_ci = 1usize << (addr_inner_bits + poly_inner_bits);
        let ci_per_thread = cycle_inner_span.div_ceil(num_threads).max(1);
        let chunk_positions = ci_per_thread * positions_per_ci;

        let mut rows = vec![[0i32; D]; block_len];

        rows.par_chunks_mut(chunk_positions)
            .enumerate()
            .for_each(|(thread_idx, output_chunk)| {
                let ci_start = thread_idx * ci_per_thread;
                let ci_end = (ci_start + ci_per_thread).min(cycle_inner_span);
                let pos_base = ci_start * positions_per_ci;

                let mut idx_buf = vec![None::<u8>; self.num_polys];
                let mut group_block_indices = vec![usize::MAX; poly_groups.len()];
                let mut rotated_tables = vec![[[0i32; D]; D]; poly_groups.len()];

                for cycle_outer in 0..cycle_outer_count {
                    for (group_idx, ((_, _, block_base), block_idx_slot)) in poly_groups
                        .iter()
                        .zip(group_block_indices.iter_mut())
                        .enumerate()
                    {
                        let block_idx = cycle_outer | *block_base;
                        *block_idx_slot = if block_idx < block_limit
                            && !challenges[block_idx].positions.is_empty()
                        {
                            Self::fill_rotated_challenge_i32(
                                unsafe { rotated_tables.get_unchecked_mut(group_idx) },
                                unsafe { challenges.get_unchecked(block_idx) },
                            );
                            block_idx
                        } else {
                            usize::MAX
                        };
                    }

                    let cycle_base = cycle_outer * cycle_inner_span;
                    let c_start = cycle_base + ci_start;
                    let c_end = (cycle_base + ci_end).min(self.num_cycles);
                    for c in c_start..c_end {
                        (self.batch_fn)(c, 0, &mut idx_buf);
                        let cs =
                            unsafe { *cycle_shifted.get_unchecked(c & params.cycle_inner_mask) };
                        for (group_idx, ((group_start, group_end, _), &block_idx)) in poly_groups
                            .iter()
                            .zip(group_block_indices.iter())
                            .enumerate()
                        {
                            if block_idx == usize::MAX {
                                continue;
                            }
                            let buf = &idx_buf[*group_start..*group_end];
                            let poly_offsets = &poly_shifted[*group_start..*group_end];
                            for (slot, &poly_offset) in buf.iter().zip(poly_offsets.iter()) {
                                if let Some(k) = *slot {
                                    let k = k as usize;
                                    let coeff_idx = k & params.addr_coeff_mask;
                                    let addr_inner = k >> params.addr_coeff_bits;
                                    let pos = addr_inner | cs | poly_offset;
                                    unsafe {
                                        let rotated = rotated_tables
                                            .get_unchecked(group_idx)
                                            .get_unchecked(coeff_idx);
                                        Self::accumulate_rotated_i32(
                                            output_chunk.get_unchecked_mut(pos - pos_base),
                                            rotated,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            });

        Self::expand_decompose_fold_rows(rows, delta)
    }

    #[allow(dead_code)]
    pub(super) fn commit_inner_small(
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

                self.for_each_entry_in_block(block_idx, |pos_in_block, coeffs| {
                    let a_base = pos_in_block * n_a;
                    let a_slice = &a_wide_flat[a_base..a_base + n_a];
                    for a in 0..n_a {
                        for &coeff_idx in coeffs {
                            Self::shift_accumulate_into_fast(
                                &a_slice[a],
                                &mut t_wide[a],
                                coeff_idx as usize,
                            );
                        }
                    }
                });

                let t: Vec<CyclotomicRing<Fp128, D>> =
                    t_wide.iter_mut().map(|w| w.reduce()).collect();
                decompose_rows_i8(&t, num_digits_open, log_basis)
            })
            .collect()
    }

    #[allow(dead_code)]
    pub(super) fn commit_inner_tiled(
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

                self.for_each_entry_in_block(block_idx, |pos_in_block, coeffs| {
                    let tile_idx = pos_in_block / tile_size;
                    let tile_start = tile_idx * tile_size;
                    let local_pos = pos_in_block - tile_start;
                    let a_tile = &a_tiles[tile_idx];
                    let a_base = local_pos * n_a;
                    let a_slice = &a_tile[a_base..a_base + n_a];
                    for a in 0..n_a {
                        for &coeff_idx in coeffs {
                            Self::shift_accumulate_into_fast(
                                &a_slice[a],
                                &mut t_wide[a],
                                coeff_idx as usize,
                            );
                        }
                    }

                    entries_since_reduce += coeffs.len();
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

    #[allow(dead_code)]
    pub(super) fn decompose_fold_small(
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

    #[allow(dead_code)]
    pub(super) fn decompose_fold_large(
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

    #[inline]
    fn expand_decompose_fold_rows(rows: Vec<[i32; D]>, delta: usize) -> Vec<[i32; D]> {
        if delta == 1 {
            return rows;
        }

        let mut total_z = vec![[0i32; D]; rows.len() * delta];
        for (pos_in_block, row) in rows.into_iter().enumerate() {
            total_z[pos_in_block * delta] = row;
        }
        total_z
    }
}

pub(super) fn build_packed_poly<F, B, const D: usize>(
    index_fn: F,
    batch_fn: B,
    num_cycles: usize,
    num_polys: usize,
    packed_layout: PackedBitLayout,
) -> JoltPackedPoly<F, B, D>
where
    F: Fn(usize, usize) -> Option<u8> + Sync,
    B: Fn(usize, usize, &mut [Option<u8>]) + Sync,
{
    let num_padded = num_polys.next_power_of_two();
    assert_eq!(
        num_padded,
        packed_layout.num_padded_polys(),
        "packed layout padding mismatch (expected {}, got {num_padded})",
        packed_layout.num_padded_polys()
    );

    JoltPackedPoly {
        packed_layout,
        index_fn,
        batch_fn,
        num_cycles,
        num_polys,
    }
}
