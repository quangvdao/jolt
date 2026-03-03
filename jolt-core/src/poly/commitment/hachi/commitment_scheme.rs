use std::borrow::Borrow;
use std::collections::HashMap;
use std::marker::PhantomData;

use super::wrappers::{jolt_to_hachi, ArkBridge, Fp128, JoltToHachiTranscript};
use crate::field::fp128::JoltFp128;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{CommitmentScheme, StreamingCommitmentScheme};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::one_hot_polynomial::OneHotPolynomial;
use crate::poly::opening_proof::BatchPolynomialSource;
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;
use crate::utils::small_scalar::SmallScalar;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use hachi_pcs::algebra::ring::CyclotomicRing;
use hachi_pcs::protocol::commitment::utils::crt_ntt::NttMatrixCache;
use hachi_pcs::protocol::commitment::utils::linear::{
    decompose_block, decompose_block_i8, decompose_rows_i8, mat_vec_mul_ntt_cached_i8, MatrixSlot,
};
use hachi_pcs::protocol::commitment::{CommitmentConfig, RingCommitment, SparseBlockEntry};
use hachi_pcs::protocol::opening_point::BasisMode;
use hachi_pcs::protocol::proof::{HachiCommitmentHint, HachiProof};
use hachi_pcs::protocol::{HachiCommitmentScheme, HachiProverSetup, HachiVerifierSetup};
use hachi_pcs::CommitmentScheme as HachiCommitmentSchemeTrait;
use hachi_pcs::{CanonicalField, DensePoly, FieldCore, FromSmallInt, HachiPolyOps, OneHotPoly};
use rayon::prelude::*;

#[derive(Clone, Default)]
pub struct JoltHachiCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HachiBatchedProof<const D: usize> {
    pub packed_poly_proof: ArkBridge<HachiProof<Fp128, D>>,
    pub num_packed_polys: u32,
    pub individual_proofs: Vec<ArkBridge<HachiProof<Fp128, D>>>,
}

/// Batch hint: carries block-structured polynomial data for packed proving.
/// Preserves one-hot vs dense block classification so the prove path can
/// dispatch sparse operations.
#[derive(Clone, Debug)]
pub struct JoltHachiBatchHint<const D: usize> {
    hachi_hint: HachiCommitmentHint<Fp128, D>,
    blocks: Vec<PackedBlock<D>>,
    block_len: usize,
}

/// Individual hint: carries dense ring coefficients for single-poly proving
/// (used for advice polynomials committed individually).
#[derive(Clone, Debug, PartialEq)]
pub struct JoltHachiOpeningHint<const D: usize> {
    hachi_hint: HachiCommitmentHint<Fp128, D>,
    ring_coeffs: Vec<CyclotomicRing<Fp128, D>>,
}

#[derive(Clone, Debug)]
enum PackedBlock<const D: usize> {
    Dense(Vec<CyclotomicRing<Fp128, D>>),
    OneHot(Vec<SparseBlockEntry>),
    Zero,
}

/// Packed polynomial combining multiple sub-polynomials with per-block
/// dispatch: dense blocks use decompose + NTT matvec, one-hot blocks use
/// sparse coefficient accumulation, zero blocks are skipped entirely.
#[derive(Clone)]
struct JoltPackedPoly<const D: usize> {
    blocks: Vec<PackedBlock<D>>,
    block_len: usize,
}

fn to_hachi_opening_point<const D: usize>(point: &[JoltFp128]) -> Vec<Fp128> {
    point.iter().rev().map(jolt_to_hachi).collect()
}

fn to_hachi_packed_opening_point<const D: usize>(
    opening_point: &[JoltFp128],
    rho: &[JoltFp128],
    _m_vars: usize,
) -> Vec<Fp128> {
    let mut out = to_hachi_opening_point::<D>(opening_point);
    out.extend(rho.iter().rev().map(jolt_to_hachi));
    out
}

impl<const D: usize> HachiPolyOps<Fp128, D> for JoltPackedPoly<D> {
    fn num_ring_elems(&self) -> usize {
        self.blocks.len() * self.block_len
    }

    fn evaluate_ring(&self, scalars: &[Fp128]) -> CyclotomicRing<Fp128, D> {
        let bl = self.block_len;
        self.blocks
            .par_iter()
            .enumerate()
            .map(|(block_idx, block)| {
                let block_start = block_idx * bl;
                match block {
                    PackedBlock::Dense(coeffs) => {
                        let end = (block_start + coeffs.len()).min(scalars.len());
                        if block_start >= scalars.len() {
                            return CyclotomicRing::zero();
                        }
                        let mut acc = CyclotomicRing::<Fp128, D>::zero();
                        for (f_i, &w_i) in coeffs.iter().zip(scalars[block_start..end].iter()) {
                            acc += f_i.scale(&w_i);
                        }
                        acc
                    }
                    PackedBlock::OneHot(entries) => {
                        let mut coeffs_arr = [Fp128::zero(); D];
                        for entry in entries {
                            let ring_idx = block_start + entry.pos_in_block;
                            if ring_idx < scalars.len() {
                                let s = scalars[ring_idx];
                                for &ci in &entry.nonzero_coeffs {
                                    coeffs_arr[ci] = coeffs_arr[ci] + s;
                                }
                            }
                        }
                        CyclotomicRing::from_coefficients(coeffs_arr)
                    }
                    PackedBlock::Zero => CyclotomicRing::zero(),
                }
            })
            .reduce(CyclotomicRing::<Fp128, D>::zero, |a, b| a + b)
    }

    fn fold_blocks(&self, scalars: &[Fp128], block_len: usize) -> Vec<CyclotomicRing<Fp128, D>> {
        self.blocks
            .par_iter()
            .map(|block| match block {
                PackedBlock::Dense(coeffs) => {
                    let mut acc = CyclotomicRing::<Fp128, D>::zero();
                    for (b_j, &a_j) in coeffs.iter().zip(scalars.iter()) {
                        acc += b_j.scale(&a_j);
                    }
                    acc
                }
                PackedBlock::OneHot(entries) => {
                    let mut coeffs_arr = [Fp128::zero(); D];
                    for entry in entries {
                        if entry.pos_in_block < scalars.len() && entry.pos_in_block < block_len {
                            let s = scalars[entry.pos_in_block];
                            for &ci in &entry.nonzero_coeffs {
                                coeffs_arr[ci] = coeffs_arr[ci] + s;
                            }
                        }
                    }
                    CyclotomicRing::from_coefficients(coeffs_arr)
                }
                PackedBlock::Zero => CyclotomicRing::zero(),
            })
            .collect()
    }

    fn decompose_fold(
        &self,
        challenges: &[hachi_pcs::algebra::ring::sparse_challenge::SparseChallenge],
        block_len: usize,
        delta: usize,
        log_basis: u32,
    ) -> Vec<CyclotomicRing<Fp128, D>> {
        let inner_width = block_len * delta;
        let mut z = vec![CyclotomicRing::<Fp128, D>::zero(); inner_width];

        for (i, c_i) in challenges.iter().enumerate() {
            if i >= self.blocks.len() {
                continue;
            }
            match &self.blocks[i] {
                PackedBlock::Dense(coeffs) => {
                    let s_i = decompose_block(coeffs, delta, log_basis);
                    for (j, z_j) in z.iter_mut().enumerate() {
                        *z_j += s_i[j].mul_by_sparse(c_i);
                    }
                }
                PackedBlock::OneHot(entries) => {
                    // One-hot coefficients are {0,1}: balanced decomposition is
                    // nonzero only in digit plane 0, where it equals the original.
                    for entry in entries {
                        let j = entry.pos_in_block * delta;
                        let mut one_hot_coeffs = [Fp128::zero(); D];
                        for &ci in &entry.nonzero_coeffs {
                            one_hot_coeffs[ci] = Fp128::one();
                        }
                        let one_hot = CyclotomicRing::from_coefficients(one_hot_coeffs);
                        z[j] += one_hot.mul_by_sparse(c_i);
                    }
                }
                PackedBlock::Zero => {}
            }
        }

        z
    }

    #[allow(non_snake_case)]
    fn commit_inner(
        &self,
        a_matrix: &[Vec<CyclotomicRing<Fp128, D>>],
        cache: &NttMatrixCache<D>,
        _block_len: usize,
        delta: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<[i8; D]>>, hachi_pcs::HachiError> {
        let n_a = a_matrix.len();
        let t_hat_len = n_a.checked_mul(delta).unwrap();

        let mut dense_indices: Vec<(usize, &[CyclotomicRing<Fp128, D>])> = Vec::new();
        let mut onehot_blocks: Vec<(usize, &[SparseBlockEntry])> = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            match block {
                PackedBlock::Dense(coeffs) => dense_indices.push((i, coeffs)),
                PackedBlock::OneHot(entries) if !entries.is_empty() => {
                    onehot_blocks.push((i, entries));
                }
                _ => {}
            }
        }

        let dense_results: Vec<(usize, Vec<[i8; D]>)> = dense_indices
            .into_par_iter()
            .map(|(i, coeffs)| {
                let s_i = decompose_block_i8(coeffs, delta, log_basis);
                let t_i: Vec<CyclotomicRing<Fp128, D>> =
                    mat_vec_mul_ntt_cached_i8(cache, MatrixSlot::A, &s_i);
                (i, decompose_rows_i8(&t_i, delta, log_basis))
            })
            .collect();

        let onehot_results: Vec<(usize, Vec<[i8; D]>)> = onehot_blocks
            .into_par_iter()
            .map(|(i, sparse_entries)| {
                let mut t = vec![CyclotomicRing::<Fp128, D>::zero(); n_a];
                for entry in sparse_entries {
                    let col = entry.pos_in_block * delta;
                    for a in 0..n_a {
                        t[a] += a_matrix[a][col].mul_by_monomial_sum(&entry.nonzero_coeffs);
                    }
                }
                (i, decompose_rows_i8(&t, delta, log_basis))
            })
            .collect();

        let mut results: Vec<Vec<[i8; D]>> =
            (0..self.blocks.len()).map(|_| Vec::new()).collect();
        for (block_i, t_hat) in dense_results {
            results[block_i] = t_hat;
        }
        for (block_i, t_hat) in onehot_results {
            results[block_i] = t_hat;
        }
        for slot in &mut results {
            if slot.is_empty() {
                *slot = vec![[0i8; D]; t_hat_len];
            }
        }

        Ok(results)
    }
}

fn hachi_commit_dense<const D: usize, Cfg: CommitmentConfig>(
    ring_coeffs: Vec<CyclotomicRing<Fp128, D>>,
    setup: &HachiProverSetup<Fp128, D>,
) -> (RingCommitment<Fp128, D>, JoltHachiOpeningHint<D>) {
    let mut dense_poly = DensePoly::from_ring_coeffs(ring_coeffs);
    let (commitment, hachi_hint) = <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<
        Fp128,
        D,
    >>::commit(&dense_poly, setup)
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

fn hachi_commit_onehot<const D: usize, Cfg: CommitmentConfig>(
    onehot: &OneHotPolynomial<JoltFp128>,
    setup: &HachiProverSetup<Fp128, D>,
) -> (RingCommitment<Fp128, D>, JoltHachiOpeningHint<D>) {
    let indices: Vec<Option<usize>> = onehot
        .nonzero_indices
        .iter()
        .map(|opt| opt.map(|v| v as usize))
        .collect();
    let layout = setup.layout();
    let onehot_poly = OneHotPoly::<Fp128, D>::new(onehot.K, indices, layout.r_vars, layout.m_vars)
        .expect("OneHotPoly construction failed");
    let (commitment, hachi_hint) = <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<
        Fp128,
        D,
    >>::commit(&onehot_poly, setup)
    .expect("Hachi commit_onehot failed");
    (
        commitment,
        JoltHachiOpeningHint {
            hachi_hint,
            ring_coeffs: vec![],
        },
    )
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
        let (commitment, hint) = if let MultilinearPolynomial::OneHot(onehot) = poly {
            hachi_commit_onehot::<D, Cfg>(onehot, &setup.0)
        } else {
            let ring_coeffs = poly_to_ring_coeffs::<D>(poly);
            hachi_commit_dense::<D, Cfg>(ring_coeffs, &setup.0)
        };
        (ArkBridge(commitment), hint)
    }

    fn batch_commit<U>(
        &self,
        polys: &[U],
        setup: &Self::ProverSetup,
    ) -> (Vec<Self::Commitment>, Self::BatchOpeningHint)
    where
        U: Borrow<MultilinearPolynomial<JoltFp128>> + Sync,
    {
        assert!(!polys.is_empty());

        let layout = setup.0.layout();
        let block_len = layout.block_len;

        let poly_field_len = polys[0].borrow().len();
        let poly_ring_len = poly_field_len.div_ceil(D);
        let blocks_per_poly = poly_ring_len.div_ceil(block_len);

        let n_padded = polys.len().next_power_of_two();
        let total_packed_blocks = n_padded * blocks_per_poly;

        let poly_ring_data: Vec<PolyRingData<D>> = polys
            .par_iter()
            .map(|p| poly_to_ring_data(p.borrow(), block_len, blocks_per_poly))
            .collect();

        let mut blocks: Vec<PackedBlock<D>> = Vec::with_capacity(total_packed_blocks);

        for data in poly_ring_data {
            match data {
                PolyRingData::Dense(all_rings) => {
                    for chunk in all_rings.chunks(block_len) {
                        let mut block = vec![CyclotomicRing::<Fp128, D>::zero(); block_len];
                        block[..chunk.len()].copy_from_slice(chunk);
                        blocks.push(PackedBlock::Dense(block));
                    }
                    let filled = all_rings.len().div_ceil(block_len);
                    for _ in filled..blocks_per_poly {
                        blocks.push(PackedBlock::Zero);
                    }
                }
                PolyRingData::OneHot(per_block_entries) => {
                    for blk_entries in per_block_entries {
                        if blk_entries.is_empty() {
                            blocks.push(PackedBlock::Zero);
                        } else {
                            blocks.push(PackedBlock::OneHot(blk_entries));
                        }
                    }
                }
            }
        }

        for _ in polys.len()..n_padded {
            for _ in 0..blocks_per_poly {
                blocks.push(PackedBlock::Zero);
            }
        }

        let packed_poly = JoltPackedPoly { blocks, block_len };
        let (commitment, hachi_hint) =
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::commit(
                &packed_poly,
                &setup.0,
            )
            .expect("Hachi packed poly commit failed");

        let hint = JoltHachiBatchHint {
            hachi_hint,
            blocks: packed_poly.blocks,
            block_len,
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

        let proof = if let MultilinearPolynomial::OneHot(onehot) = poly {
            let indices: Vec<Option<usize>> = onehot
                .nonzero_indices
                .iter()
                .map(|opt| opt.map(|v| v as usize))
                .collect();
            let layout = setup.0.layout();
            let onehot_poly =
                OneHotPoly::<Fp128, D>::new(onehot.K, indices, layout.r_vars, layout.m_vars)
                    .expect("OneHotPoly construction failed");
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::prove(
                &setup.0,
                &onehot_poly,
                &hachi_point,
                hint.hachi_hint,
                &mut adapter,
                &commitment.0,
                BasisMode::Lagrange,
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

        <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::verify(
            &proof.0,
            &setup.0,
            &mut adapter,
            &hachi_point,
            &hachi_opening,
            &commitment.0,
            BasisMode::Lagrange,
        )
        .map_err(|_| ProofVerifyError::InternalError)
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_prove<ProofTranscript: Transcript, S: BatchPolynomialSource<JoltFp128>>(
        &self,
        setup: &Self::ProverSetup,
        _poly_source: &S,
        batch_hint: Self::BatchOpeningHint,
        individual_hints: Vec<Self::OpeningProofHint>,
        commitments: &[&Self::Commitment],
        opening_point: &[JoltFp128],
        claims: &[JoltFp128],
        _coeffs: &[JoltFp128],
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        let num_individual = individual_hints.len();
        let num_packed = claims.len() - num_individual;
        assert!(
            num_packed > 0,
            "batch_prove requires at least one packed claim"
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

        let packed_point = to_hachi_packed_opening_point::<D>(
            opening_point,
            &rho,
            setup.0.expanded.seed.layout.m_vars,
        );

        let packed_poly = JoltPackedPoly {
            blocks: batch_hint.blocks,
            block_len: batch_hint.block_len,
        };

        let hachi_combined = jolt_to_hachi(&combined_claim);
        let packed_commitment = commitments[0];

        transcript.append_bytes(
            b"hachi_packed_claim",
            &hachi_combined.to_canonical_u128().to_le_bytes(),
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
            )
            .expect("Hachi packed poly prove failed");

        let hachi_point = to_hachi_opening_point::<D>(opening_point);
        let individual_commitments = &commitments[num_packed..];
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
                    )
                    .expect("Hachi individual prove failed");
                ArkBridge(proof)
            })
            .collect();

        HachiBatchedProof {
            packed_poly_proof: ArkBridge(packed_proof),
            num_packed_polys: num_packed as u32,
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
        if num_packed > claims.len() || num_packed > commitments.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                claims.len(),
                num_packed,
            ));
        }

        let packed_claims = &claims[..num_packed];
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

        let packed_point = to_hachi_packed_opening_point::<D>(
            opening_point,
            &rho,
            setup.0.expanded.seed.layout.m_vars,
        );

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
        )
        .map_err(|_| ProofVerifyError::InternalError)?;

        let individual_claims = &claims[num_packed..];
        let individual_commitments = &commitments[num_packed..];
        if proof.individual_proofs.len() != individual_claims.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                individual_claims.len(),
                proof.individual_proofs.len(),
            ));
        }

        let hachi_point = to_hachi_opening_point::<D>(opening_point);
        for (i, ((proof_i, claim_i), commitment_i)) in proof
            .individual_proofs
            .iter()
            .zip(individual_claims.iter())
            .zip(individual_commitments.iter())
            .enumerate()
        {
            transcript.append_bytes(b"hachi_individual_item", &(i as u64).to_le_bytes());
            let hachi_claim = jolt_to_hachi(claim_i);
            let mut adapter = JoltToHachiTranscript::new(transcript);
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::verify(
                &proof_i.0,
                &setup.0,
                &mut adapter,
                &hachi_point,
                &hachi_claim,
                &commitment_i.0,
                BasisMode::Lagrange,
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

    fn uses_onehot_inc() -> bool {
        true
    }
}

impl<const D: usize, Cfg> StreamingCommitmentScheme for JoltHachiCommitmentScheme<D, Cfg>
where
    Cfg: CommitmentConfig + Default,
{
    type ChunkState = ();

    #[allow(non_snake_case)]
    fn streaming_chunk_size(&self, _K: usize, _T: usize) -> Option<usize> {
        None
    }

    fn process_chunk<T: SmallScalar>(
        &self,
        _setup: &Self::ProverSetup,
        _chunk: &[T],
    ) -> Self::ChunkState {
        unreachable!("streaming_chunk_size returns None")
    }

    fn process_chunk_onehot(
        &self,
        _setup: &Self::ProverSetup,
        _onehot_k: usize,
        _chunk: &[Option<usize>],
    ) -> Self::ChunkState {
        unreachable!("streaming_chunk_size returns None")
    }

    fn aggregate_chunks(
        &self,
        _setup: &Self::ProverSetup,
        _onehot_k: Option<usize>,
        _tier1_commitments: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        unreachable!("streaming_chunk_size returns None")
    }

    fn streaming_batch_hint(_hints: Vec<Self::OpeningProofHint>) -> Self::BatchOpeningHint {
        unreachable!("streaming_chunk_size returns None")
    }
}

enum PolyRingData<const D: usize> {
    Dense(Vec<CyclotomicRing<Fp128, D>>),
    OneHot(Vec<Vec<SparseBlockEntry>>),
}

fn poly_to_ring_data<const D: usize>(
    poly: &MultilinearPolynomial<JoltFp128>,
    block_len: usize,
    blocks_per_poly: usize,
) -> PolyRingData<D> {
    match poly {
        MultilinearPolynomial::OneHot(onehot) => {
            let mut per_block: Vec<Vec<SparseBlockEntry>> = vec![Vec::new(); blocks_per_poly];
            let mut block_maps: Vec<HashMap<usize, usize>> = vec![HashMap::new(); blocks_per_poly];
            let t = onehot.nonzero_indices.len();
            for (c, opt) in onehot.nonzero_indices.iter().enumerate() {
                if let Some(idx) = opt {
                    let k = *idx as usize;
                    let field_pos = k * t + c;
                    let ring_idx = field_pos / D;
                    let coeff_idx = field_pos % D;
                    let block_idx = ring_idx / block_len;
                    let pos_in_block = ring_idx % block_len;
                    if block_idx < blocks_per_poly {
                        if let Some(&entry_idx) = block_maps[block_idx].get(&pos_in_block) {
                            per_block[block_idx][entry_idx]
                                .nonzero_coeffs
                                .push(coeff_idx);
                        } else {
                            block_maps[block_idx].insert(pos_in_block, per_block[block_idx].len());
                            per_block[block_idx].push(SparseBlockEntry {
                                pos_in_block,
                                nonzero_coeffs: vec![coeff_idx],
                            });
                        }
                    }
                }
            }
            PolyRingData::OneHot(per_block)
        }
        _ => PolyRingData::Dense(poly_to_ring_coeffs(poly)),
    }
}

pub(super) fn poly_to_ring_coeffs<const D: usize>(
    poly: &MultilinearPolynomial<JoltFp128>,
) -> Vec<CyclotomicRing<Fp128, D>> {
    match poly {
        MultilinearPolynomial::LargeScalars(p) => {
            // SAFETY: JoltFp128 is repr(transparent) over Fp128.
            // Reinterpret the slice directly — avoids cloning the full coefficient vector.
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
