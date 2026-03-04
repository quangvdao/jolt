use std::collections::HashMap;
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
use hachi_pcs::protocol::commitment::utils::linear::decompose_rows_i8;
use hachi_pcs::protocol::commitment::{
    compute_num_digits, compute_num_digits_fold, CommitmentConfig, HachiCommitmentLayout,
    RingCommitment, SparseBlockEntry,
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
    pub packed_poly_proof: ArkBridge<HachiProof<Fp128, D>>,
    pub num_packed_polys: u32,
    pub log_k: u32,
    pub individual_proofs: Vec<ArkBridge<HachiProof<Fp128, D>>>,
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

/// Thin view over the trace: all HachiPolyOps methods iterate the trace
/// on-the-fly via the type-erased `index_of` callback.  Zero bytes of
/// polynomial data are materialized.
#[derive(Clone, Copy)]
struct JoltPackedPoly<const D: usize> {
    index_of: ErasedIndexFn,
    num_cycles: usize,
    num_polys: usize,
    block_len: usize,
    blocks_per_poly: usize,
    num_padded: usize,
}

impl<const D: usize> JoltPackedPoly<D> {
    fn entries_for_block(&self, packed_block_idx: usize) -> Vec<SparseBlockEntry> {
        let poly_idx = packed_block_idx / self.blocks_per_poly;
        let block_idx = packed_block_idx % self.blocks_per_poly;

        if poly_idx >= self.num_polys {
            return vec![];
        }

        let mut entries_map: HashMap<usize, usize> = HashMap::new();
        let mut entries: Vec<SparseBlockEntry> = Vec::new();

        // K-major layout: field_pos = k * T + c (address varies slowly, cycle
        // varies fast).  This matches the old streaming_chunks_to_blocks and
        // the opening-point convention [r_cycle_LE, r_addr_LE] produced by
        // reversing [r_addr_BE, r_cycle_BE].
        for c in 0..self.num_cycles {
            if let Some(k) = self.index_of.call(c, poly_idx) {
                let field_pos = (k as usize) * self.num_cycles + c;
                let ring_idx = field_pos / D;
                let coeff_idx = field_pos % D;
                let block_of_ring = ring_idx / self.block_len;
                if block_of_ring != block_idx {
                    continue;
                }
                let pos_in_block = ring_idx % self.block_len;
                if let Some(&entry_idx) = entries_map.get(&pos_in_block) {
                    entries[entry_idx].nonzero_coeffs.push(coeff_idx);
                } else {
                    entries_map.insert(pos_in_block, entries.len());
                    entries.push(SparseBlockEntry {
                        pos_in_block,
                        nonzero_coeffs: vec![coeff_idx],
                    });
                }
            }
        }
        entries
    }
}

impl<const D: usize> HachiPolyOps<Fp128, D> for JoltPackedPoly<D> {
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        (self.num_padded * self.blocks_per_poly) * self.block_len
    }

    #[tracing::instrument(skip_all, name = "evaluate_ring")]
    fn evaluate_ring(&self, scalars: &[Fp128]) -> CyclotomicRing<Fp128, D> {
        let (eval, _) = self.evaluate_and_fold(scalars, &[], 0);
        eval
    }

    #[tracing::instrument(skip_all, name = "fold_blocks")]
    fn fold_blocks(&self, scalars: &[Fp128], block_len: usize) -> Vec<CyclotomicRing<Fp128, D>> {
        let (_, folded) = self.evaluate_and_fold(&[], scalars, block_len);
        folded
    }

    #[tracing::instrument(skip_all, name = "evaluate_and_fold")]
    fn evaluate_and_fold(
        &self,
        eval_scalars: &[Fp128],
        fold_scalars: &[Fp128],
        _block_len: usize,
    ) -> (CyclotomicRing<Fp128, D>, Vec<CyclotomicRing<Fp128, D>>) {
        let total_blocks = self.num_padded * self.blocks_per_poly;
        let bl = self.block_len;
        let do_eval = !eval_scalars.is_empty();
        let do_fold = !fold_scalars.is_empty();

        let per_block: Vec<(CyclotomicRing<Fp128, D>, CyclotomicRing<Fp128, D>)> = (0
            ..total_blocks)
            .into_par_iter()
            .map(|bi| {
                let entries = self.entries_for_block(bi);
                let block_start = bi * bl;

                let mut eval_acc = [Fp128::zero(); D];
                let mut fold_acc = [Fp128::zero(); D];

                for entry in &entries {
                    if do_eval {
                        let ring_idx = block_start + entry.pos_in_block;
                        if ring_idx < eval_scalars.len() {
                            let s = eval_scalars[ring_idx];
                            for &ci in &entry.nonzero_coeffs {
                                eval_acc[ci] += s;
                            }
                        }
                    }
                    if do_fold && entry.pos_in_block < fold_scalars.len() {
                        let s = fold_scalars[entry.pos_in_block];
                        for &ci in &entry.nonzero_coeffs {
                            fold_acc[ci] += s;
                        }
                    }
                }
                (
                    CyclotomicRing::from_coefficients(eval_acc),
                    CyclotomicRing::from_coefficients(fold_acc),
                )
            })
            .collect();

        let eval_result = per_block
            .iter()
            .map(|(e, _)| *e)
            .reduce(|a, b| a + b)
            .unwrap_or_else(CyclotomicRing::zero);
        let fold_result: Vec<_> = per_block.into_iter().map(|(_, f)| f).collect();
        (eval_result, fold_result)
    }

    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        delta: usize,
        _log_basis: u32,
    ) -> Vec<CyclotomicRing<Fp128, D>> {
        let inner_width = block_len * delta;
        let total_blocks = self.num_padded * self.blocks_per_poly;

        let z_i32: Vec<[i32; D]> = (0..total_blocks)
            .into_par_iter()
            .fold(
                || vec![[0i32; D]; inner_width],
                |mut z, bi| {
                    if bi >= challenges.len() {
                        return z;
                    }
                    let c_i = &challenges[bi];
                    let entries = self.entries_for_block(bi);
                    for entry in &entries {
                        assert!(
                            entry.pos_in_block < block_len,
                            "onehot entry position {} exceeds block_len {}",
                            entry.pos_in_block,
                            block_len,
                        );
                        let base_j = entry.pos_in_block * delta;
                        for &ci in &entry.nonzero_coeffs {
                            for (&pos, &challenge_coeff) in
                                c_i.positions.iter().zip(c_i.coeffs.iter())
                            {
                                let target = ci + pos as usize;
                                let (idx, sign) = if target < D {
                                    (target, 1i32)
                                } else {
                                    (target - D, -1i32)
                                };
                                z[base_j][idx] += sign * challenge_coeff as i32;
                            }
                        }
                    }
                    z
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
            );

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
    #[tracing::instrument(skip_all, name = "commit_inner")]
    fn commit_inner(
        &self,
        a_matrix: &[Vec<CyclotomicRing<Fp128, D>>],
        _ntt_a: &NttSlotCache<D>,
        _block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<[i8; D]>>, hachi_pcs::HachiError> {
        let n_a = a_matrix.len();
        let t_hat_len = n_a.checked_mul(num_digits_open).unwrap();
        let total_blocks = self.num_padded * self.blocks_per_poly;

        let all_entries: Vec<Vec<SparseBlockEntry>> = (0..total_blocks)
            .into_par_iter()
            .map(|bi| self.entries_for_block(bi))
            .collect();

        let unique_cols: std::collections::HashSet<usize> = all_entries
            .iter()
            .flat_map(|entries| entries.iter().map(|e| e.pos_in_block * num_digits_commit))
            .collect();
        let a_wide_cols: HashMap<usize, Vec<WideCyclotomicRing<Fp128x8i32, D>>> = unique_cols
            .into_par_iter()
            .map(|col| {
                let wide_col: Vec<_> = (0..n_a)
                    .map(|a| WideCyclotomicRing::from_ring(&a_matrix[a][col]))
                    .collect();
                (col, wide_col)
            })
            .collect();

        let results: Vec<Vec<[i8; D]>> = all_entries
            .into_par_iter()
            .map(|entries| {
                if entries.is_empty() {
                    return vec![[0i8; D]; t_hat_len];
                }
                let mut t_wide = vec![WideCyclotomicRing::<Fp128x8i32, D>::zero(); n_a];
                for entry in &entries {
                    let col = entry.pos_in_block * num_digits_commit;
                    let a_wide = &a_wide_cols[&col];
                    for a in 0..n_a {
                        a_wide[a].mul_by_monomial_sum_into(&mut t_wide[a], &entry.nonzero_coeffs);
                    }
                }
                let t: Vec<CyclotomicRing<Fp128, D>> =
                    t_wide.into_iter().map(|w| w.reduce()).collect();
                decompose_rows_i8(&t, num_digits_open, log_basis)
            })
            .collect();

        Ok(results)
    }
}

fn to_hachi_opening_point<const D: usize>(point: &[JoltFp128]) -> Vec<Fp128> {
    point.iter().rev().map(jolt_to_hachi).collect()
}

fn to_hachi_packed_opening_point<const D: usize>(
    opening_point: &[JoltFp128],
    rho: &[JoltFp128],
    _log_k: usize,
) -> Vec<Fp128> {
    let mut out = to_hachi_opening_point::<D>(opening_point);
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

/// Type-erased index callback.  Stores a raw data pointer and a monomorphized
/// trampoline that knows how to call the concrete source type.
///
/// SAFETY: the caller **must** ensure the pointee outlives this struct.
#[derive(Clone, Copy)]
struct ErasedIndexFn {
    ptr: usize,
    call: unsafe fn(usize, usize, usize) -> Option<u8>,
}

// SAFETY: we only read through the pointer, and the source is Sync.
unsafe impl Send for ErasedIndexFn {}
unsafe impl Sync for ErasedIndexFn {}

impl ErasedIndexFn {
    #[inline]
    fn call(&self, cycle: usize, poly: usize) -> Option<u8> {
        // SAFETY: the pointer is valid for the lifetime of the packed poly.
        unsafe { (self.call)(self.ptr, cycle, poly) }
    }
}

fn erase_batch_source<S: PolynomialBatchSource<JoltFp128>>(source: &S) -> ErasedIndexFn {
    unsafe fn trampoline<S: PolynomialBatchSource<JoltFp128>>(
        ptr: usize,
        c: usize,
        p: usize,
    ) -> Option<u8> {
        (&*(ptr as *const S)).onehot_index(c, p)
    }
    ErasedIndexFn {
        ptr: source as *const S as usize,
        call: trampoline::<S>,
    }
}

fn erase_prove_source<S: BatchPolynomialSource<JoltFp128>>(source: &S) -> ErasedIndexFn {
    unsafe fn trampoline<S: BatchPolynomialSource<JoltFp128>>(
        ptr: usize,
        c: usize,
        p: usize,
    ) -> Option<u8> {
        (&*(ptr as *const S)).onehot_index(c, p)
    }
    ErasedIndexFn {
        ptr: source as *const S as usize,
        call: trampoline::<S>,
    }
}

fn build_packed_poly_from_fn<const D: usize>(
    index_of: ErasedIndexFn,
    num_cycles: usize,
    num_polys: usize,
    onehot_k: usize,
    block_len: usize,
) -> JoltPackedPoly<D> {
    let poly_field_len = num_cycles * onehot_k;
    let poly_ring_len = poly_field_len.div_ceil(D);
    let blocks_per_poly = poly_ring_len.div_ceil(block_len);
    let num_padded = num_polys.next_power_of_two();

    JoltPackedPoly {
        index_of,
        num_cycles,
        num_polys,
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
    type VerifierSetup = ArkBridge<HachiVerifierSetup<Fp128, D>>;
    type Commitment = ArkBridge<RingCommitment<Fp128, D>>;
    type Proof = ArkBridge<HachiProof<Fp128, D>>;
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
        let setup_layout = setup.0.layout();
        let poly_num_vars = poly.len().trailing_zeros() as usize;
        let alpha = D.trailing_zeros() as usize;
        let r_vars = poly_num_vars.saturating_sub(alpha + setup_layout.m_vars);
        let layout = advice_commit_layout::<Cfg>(setup_layout.m_vars, r_vars);

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

        let index_fn = erase_batch_source(source);
        let packed_poly =
            build_packed_poly_from_fn::<D>(index_fn, num_cycles, num_polys, onehot_k, block_len);
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

        let setup_layout = setup.0.layout();
        let alpha = D.trailing_zeros() as usize;
        let r_vars = opening_point
            .len()
            .saturating_sub(alpha + setup_layout.m_vars);
        let layout = advice_commit_layout::<Cfg>(setup_layout.m_vars, r_vars);

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

        let setup_layout = &setup.0.expanded.seed.layout;
        let alpha = D.trailing_zeros() as usize;
        let r_vars = opening_point
            .len()
            .saturating_sub(alpha + setup_layout.m_vars);
        let layout = advice_commit_layout::<Cfg>(setup_layout.m_vars, r_vars);

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

        let poly_field_len = num_cycles * onehot_k;
        let poly_ring_len = poly_field_len.div_ceil(D);
        let blocks_per_poly = poly_ring_len.div_ceil(block_len);

        let index_fn = erase_prove_source(poly_source);

        let packed_poly = JoltPackedPoly::<D> {
            index_of: index_fn,
            num_cycles,
            num_polys,
            block_len,
            blocks_per_poly,
            num_padded: num_polys.next_power_of_two(),
        };

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
        let indiv_r_vars = opening_point
            .len()
            .saturating_sub(alpha + setup_layout.m_vars);
        let indiv_layout = advice_commit_layout::<Cfg>(setup_layout.m_vars, indiv_r_vars);

        let individual_commitments = &commitments[1..];
        let individual_proofs: Vec<ArkBridge<HachiProof<Fp128, D>>> = individual_hints
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
        let packed_point =
            to_hachi_packed_opening_point::<D>(opening_point, &rho, proof.log_k as usize);

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
        let indiv_r_vars = opening_point
            .len()
            .saturating_sub(alpha + setup_layout.m_vars);
        let indiv_layout = advice_commit_layout::<Cfg>(setup_layout.m_vars, indiv_r_vars);

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
    scalars
        .par_chunks(D)
        .map(|chunk| {
            let mut coeffs = [Fp128::zero(); D];
            for (i, scalar) in chunk.iter().enumerate() {
                coeffs[i] = convert(scalar);
            }
            CyclotomicRing::from_coefficients(coeffs)
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
