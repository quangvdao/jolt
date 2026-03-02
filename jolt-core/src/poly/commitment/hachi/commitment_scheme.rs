use std::borrow::Borrow;
use std::marker::PhantomData;
use std::sync::Arc;

use super::wrappers::{jolt_to_hachi, ArkBridge, Fp128, JoltToHachiTranscript};
use crate::field::fp128::JoltFp128;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{CommitmentScheme, StreamingCommitmentScheme};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::BatchPolynomialSource;
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;
use crate::utils::small_scalar::SmallScalar;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use hachi_pcs::algebra::ring::CyclotomicRing;
use hachi_pcs::primitives::multilinear_evals::DenseMultilinearEvals;
use hachi_pcs::protocol::commitment::{
    CommitmentConfig, HachiCommitmentCore, HachiCommitmentLayout, MegaPolyBlock, RingCommitment,
    SparseBlockEntry,
};
use hachi_pcs::protocol::proof::{HachiCommitmentHint, HachiProof};
use hachi_pcs::protocol::HachiCommitmentScheme;
use hachi_pcs::protocol::HachiProverSetup;
use hachi_pcs::protocol::HachiVerifierSetup;
use hachi_pcs::CommitmentScheme as HachiCommitmentSchemeTrait;
use hachi_pcs::{CanonicalField, FieldCore, FromSmallInt, Polynomial};
use rayon::prelude::*;

#[derive(Clone, Default)]
pub struct JoltHachiCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HachiBatchedProof<const D: usize> {
    pub mega_poly_proof: ArkBridge<HachiProof<Fp128, D>>,
    pub num_mega_polys: u32,
    pub individual_proofs: Vec<ArkBridge<HachiProof<Fp128, D>>>,
}

#[derive(Clone)]
struct HintOnlyPolynomial {
    num_vars: usize,
}

impl Polynomial<Fp128> for HintOnlyPolynomial {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, _point: &[Fp128]) -> Fp128 {
        panic!("HintOnlyPolynomial::evaluate should never be called")
    }

    fn coeffs(&self) -> Vec<Fp128> {
        panic!("HintOnlyPolynomial::coeffs should never be called")
    }
}

/// Shared data for the mega-polynomial commitment.
///
/// Created once during `batch_commit` and shared (via Arc) across all
/// per-polynomial hints. Contains the commitment witness AND the setup
/// that was used, since the mega-poly may require a larger setup than
/// the per-polynomial setup.
#[derive(Clone, Debug, PartialEq)]
pub struct MegaPolyData<const D: usize> {
    pub hint: HachiCommitmentHint<Fp128, D>,
    pub prover_setup: HachiProverSetup<Fp128, D>,
}

/// Wrapper hint type distinguishing individual and mega-poly (shared) hints.
#[derive(Clone, Debug)]
pub enum HachiOpeningHint<const D: usize> {
    Individual(HachiCommitmentHint<Fp128, D>),
    MegaPoly(Arc<MegaPolyData<D>>),
}

impl<const D: usize> PartialEq for HachiOpeningHint<D> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Individual(a), Self::Individual(b)) => a == b,
            (Self::MegaPoly(a), Self::MegaPoly(b)) => Arc::ptr_eq(a, b) || *a == *b,
            _ => false,
        }
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
    type OpeningProofHint = HachiOpeningHint<D>;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        ArkBridge(
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128>>::setup_prover(
                max_num_vars,
            ),
        )
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        ArkBridge(
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128>>::setup_verifier(
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
            let indices: Vec<Option<usize>> = onehot
                .nonzero_indices
                .iter()
                .map(|opt| opt.map(|v| v as usize))
                .collect();
            hachi_pcs::protocol::commitment_scheme::commit_onehot::<Fp128, D, Cfg>(
                onehot.K,
                &indices,
                &setup.0,
            )
            .expect("Hachi commit_onehot failed")
        } else {
            let hachi_poly = to_hachi_poly(poly);
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128>>::commit(
                &hachi_poly,
                &setup.0,
            )
            .expect("Hachi commit failed")
        };
        (ArkBridge(commitment), HachiOpeningHint::Individual(hint))
    }

    fn batch_commit<U>(
        &self,
        polys: &[U],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: Borrow<MultilinearPolynomial<JoltFp128>> + Sync,
    {
        if polys.is_empty() {
            return vec![];
        }

        let layout = setup.0.layout();
        let block_len = layout.block_len;
        let blocks_per_poly = layout.num_blocks;
        let n_padded = polys.len().next_power_of_two();
        let total_mega_blocks = n_padded * blocks_per_poly;
        let r_vars_mega = (total_mega_blocks as u32).trailing_zeros() as usize;

        // Convert each polynomial into ring elements (dense) or sparse entries.
        let poly_ring_data: Vec<PolyRingData<D>> = polys
            .par_iter()
            .map(|p| poly_to_ring_data(p.borrow(), block_len, blocks_per_poly))
            .collect();

        // Assemble MegaPolyBlock descriptors per block.
        let mut owned_blocks: Vec<OwnedBlockData<D>> = Vec::with_capacity(total_mega_blocks);
        let mut ring_coeffs: Vec<CyclotomicRing<Fp128, D>> =
            Vec::with_capacity(total_mega_blocks * block_len);

        for data in &poly_ring_data {
            match data {
                PolyRingData::Dense(all_rings) => {
                    for chunk in all_rings.chunks(block_len) {
                        let mut block = vec![CyclotomicRing::<Fp128, D>::zero(); block_len];
                        block[..chunk.len()].copy_from_slice(chunk);
                        ring_coeffs.extend_from_slice(&block);
                        owned_blocks.push(OwnedBlockData::Dense(block));
                    }
                    let filled = all_rings.len().div_ceil(block_len);
                    for _ in filled..blocks_per_poly {
                        ring_coeffs
                            .extend(std::iter::repeat_n(CyclotomicRing::zero(), block_len));
                        owned_blocks.push(OwnedBlockData::Zero);
                    }
                }
                PolyRingData::OneHot(per_block_entries) => {
                    for blk_entries in per_block_entries {
                        if blk_entries.is_empty() {
                            ring_coeffs.extend(
                                std::iter::repeat_n(CyclotomicRing::zero(), block_len),
                            );
                            owned_blocks.push(OwnedBlockData::Zero);
                        } else {
                            let mut block_ring =
                                vec![CyclotomicRing::<Fp128, D>::zero(); block_len];
                            for entry in blk_entries {
                                let mut arr = [Fp128::zero(); D];
                                for &ci in &entry.nonzero_coeffs {
                                    arr[ci] = Fp128::one();
                                }
                                block_ring[entry.pos_in_block] =
                                    CyclotomicRing::from_coefficients(arr);
                            }
                            ring_coeffs.extend_from_slice(&block_ring);
                            owned_blocks.push(OwnedBlockData::OneHot(blk_entries.clone()));
                        }
                    }
                }
            }
        }

        // Pad remaining slots with zero blocks.
        for _ in polys.len()..n_padded {
            for _ in 0..blocks_per_poly {
                ring_coeffs.extend(std::iter::repeat_n(CyclotomicRing::zero(), block_len));
                owned_blocks.push(OwnedBlockData::Zero);
            }
        }

        // Build borrowed MegaPolyBlock slice from owned data.
        let mega_blocks: Vec<MegaPolyBlock<'_, Fp128, D>> = owned_blocks
            .iter()
            .map(|b| match b {
                OwnedBlockData::Dense(v) => MegaPolyBlock::Dense(v),
                OwnedBlockData::OneHot(e) => MegaPolyBlock::OneHot(e),
                OwnedBlockData::Zero => MegaPolyBlock::Zero,
            })
            .collect();

        // Create a setup for the mega-poly: same m_vars, but r_vars includes
        // the selector variables for choosing which sub-polynomial.
        let mega_layout = HachiCommitmentLayout::new::<Cfg>(layout.m_vars, r_vars_mega)
            .expect("Hachi mega-poly layout failed");
        let (mega_prover_setup, _) =
            HachiCommitmentCore::setup_with_layout::<Fp128, D, Cfg>(mega_layout)
                .expect("Hachi mega-poly setup failed");

        let witness =
            HachiCommitmentCore::commit_mixed::<Fp128, D, Cfg>(&mega_blocks, &mega_prover_setup)
                .expect("Hachi mega-poly commit_mixed failed");

        let commitment = ArkBridge(witness.commitment.clone());
        let mega_data = Arc::new(MegaPolyData {
            hint: HachiCommitmentHint {
                t_hat: witness.t_hat,
                ring_coeffs,
            },
            prover_setup: mega_prover_setup,
        });

        polys
            .iter()
            .map(|_| (commitment.clone(), HachiOpeningHint::MegaPoly(mega_data.clone())))
            .collect()
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
        let hint = match hint {
            Some(HachiOpeningHint::Individual(h)) => h,
            Some(HachiOpeningHint::MegaPoly(_)) => {
                panic!("prove() called with MegaPoly hint — use batch_prove() instead")
            }
            None => panic!("prove() requires a hint"),
        };
        let hachi_point: Vec<Fp128> = opening_point.iter().map(jolt_to_hachi).collect();
        let mut adapter = JoltToHachiTranscript::new(transcript);

        let proof = match poly {
            MultilinearPolynomial::OneHot(_) | MultilinearPolynomial::RLC(_) => {
                let stub = HintOnlyPolynomial {
                    num_vars: poly.get_num_vars(),
                };
                <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128>>::prove(
                    &setup.0,
                    &stub,
                    &hachi_point,
                    Some(hint),
                    &mut adapter,
                    &commitment.0,
                )
            }
            _ => {
                let hachi_poly = to_hachi_poly(poly);
                <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128>>::prove(
                    &setup.0,
                    &hachi_poly,
                    &hachi_point,
                    Some(hint),
                    &mut adapter,
                    &commitment.0,
                )
            }
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
        let hachi_point: Vec<Fp128> = opening_point.iter().map(jolt_to_hachi).collect();
        let hachi_opening = jolt_to_hachi(opening);
        let mut adapter = JoltToHachiTranscript::new(transcript);

        <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128>>::verify(
            &proof.0,
            &setup.0,
            &mut adapter,
            &hachi_point,
            &hachi_opening,
            &commitment.0,
        )
        .map_err(|_| ProofVerifyError::InternalError)
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_prove<ProofTranscript: Transcript, S: BatchPolynomialSource<JoltFp128>>(
        &self,
        setup: &Self::ProverSetup,
        _poly_source: &S,
        hints: Vec<Self::OpeningProofHint>,
        commitments: &[&Self::Commitment],
        opening_point: &[JoltFp128],
        claims: &[JoltFp128],
        _coeffs: &[JoltFp128],
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        // Partition into mega-poly and individual hints.
        let mut mega_data: Option<Arc<MegaPolyData<D>>> = None;
        let mut mega_claims: Vec<JoltFp128> = Vec::new();
        let mut mega_commitment: Option<&Self::Commitment> = None;
        let mut individual: Vec<(HachiCommitmentHint<Fp128, D>, &Self::Commitment, JoltFp128)> =
            Vec::new();

        for (i, hint) in hints.into_iter().enumerate() {
            match hint {
                HachiOpeningHint::MegaPoly(arc) => {
                    if mega_data.is_none() {
                        mega_data = Some(arc);
                        mega_commitment = Some(commitments[i]);
                    }
                    mega_claims.push(claims[i]);
                }
                HachiOpeningHint::Individual(h) => {
                    individual.push((h, commitments[i], claims[i]));
                }
            }
        }

        // Mega-poly proof: selector challenge + single opening.
        let mega_proof = if let Some(mega_data_arc) = mega_data {
            let num_mega = mega_claims.len().next_power_of_two();
            let r_vars = (num_mega as u32).trailing_zeros() as usize;

            // Sample selector challenge ρ from transcript.
            transcript.append_bytes(b"hachi_mega_num", &(mega_claims.len() as u64).to_le_bytes());
            let rho: Vec<JoltFp128> = transcript.challenge_scalar_powers(r_vars);

            // Compute combined claim: v* = Σ_i eq(ρ, i) · v_i
            let eq_table = EqPolynomial::<JoltFp128>::evals(&rho);
            let combined_claim: JoltFp128 = mega_claims
                .iter()
                .zip(eq_table.iter())
                .map(|(&v, &eq)| v * eq)
                .sum();

            // Build mega opening point: (opening_point || ρ)
            let mut mega_point: Vec<Fp128> = opening_point.iter().map(jolt_to_hachi).collect();
            mega_point.extend(rho.iter().map(jolt_to_hachi));

            let mega_num_vars = mega_point.len();
            let poly_stub = HintOnlyPolynomial {
                num_vars: mega_num_vars,
            };

            // Extract hint data and setup from the shared Arc.
            let data = Arc::try_unwrap(mega_data_arc).unwrap_or_else(|arc| (*arc).clone());

            let hachi_combined = jolt_to_hachi(&combined_claim);
            let mega_commitment = mega_commitment.unwrap();

            transcript.append_bytes(
                b"hachi_mega_claim",
                &hachi_combined.to_canonical_u128().to_le_bytes(),
            );
            let mut adapter = JoltToHachiTranscript::new(transcript);

            let proof =
                <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128>>::prove(
                    &data.prover_setup,
                    &poly_stub,
                    &mega_point,
                    Some(data.hint),
                    &mut adapter,
                    &mega_commitment.0,
                )
                .expect("Hachi mega-poly prove failed");
            (ArkBridge(proof), mega_claims.len() as u32)
        } else {
            panic!("batch_prove called without any mega-poly hints")
        };

        // Individual proofs (e.g. advice polynomials).
        let hachi_point: Vec<Fp128> = opening_point.iter().map(jolt_to_hachi).collect();
        let individual_proofs: Vec<ArkBridge<HachiProof<Fp128, D>>> = individual
            .into_iter()
            .enumerate()
            .map(|(i, (hint, commitment, _claim))| {
                transcript.append_bytes(
                    b"hachi_individual_item",
                    &(i as u64).to_le_bytes(),
                );
                let poly_stub = HintOnlyPolynomial {
                    num_vars: opening_point.len(),
                };
                let mut adapter = JoltToHachiTranscript::new(transcript);
                let proof =
                    <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128>>::prove(
                        &setup.0,
                        &poly_stub,
                        &hachi_point,
                        Some(hint),
                        &mut adapter,
                        &commitment.0,
                    )
                    .expect("Hachi individual prove failed");
                ArkBridge(proof)
            })
            .collect();

        HachiBatchedProof {
            mega_poly_proof: mega_proof.0,
            num_mega_polys: mega_proof.1,
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
        let num_mega = proof.num_mega_polys as usize;
        if num_mega > claims.len() || num_mega > commitments.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                claims.len(),
                num_mega,
            ));
        }

        let mega_claims = &claims[..num_mega];
        let num_padded = num_mega.next_power_of_two();
        let selector_vars = (num_padded as u32).trailing_zeros() as usize;

        // Re-derive selector challenge ρ from transcript.
        transcript.append_bytes(b"hachi_mega_num", &(num_mega as u64).to_le_bytes());
        let rho: Vec<JoltFp128> = transcript.challenge_scalar_powers(selector_vars);

        // Recompute combined claim.
        let eq_table = EqPolynomial::<JoltFp128>::evals(&rho);
        let combined_claim: JoltFp128 = mega_claims
            .iter()
            .zip(eq_table.iter())
            .map(|(&v, &eq)| v * eq)
            .sum();

        // Build mega opening point.
        let mut mega_point: Vec<Fp128> = opening_point.iter().map(jolt_to_hachi).collect();
        mega_point.extend(rho.iter().map(jolt_to_hachi));

        // Recreate mega-poly setup with matching layout (deterministic: same
        // seed + same layout → same matrices). The mega layout uses the same
        // m_vars as the individual setup, with r_vars extended by the selector.
        let individual_layout = setup.0.expanded.seed.layout;
        let r_vars_mega = individual_layout.r_vars + selector_vars;
        let mega_layout = HachiCommitmentLayout::new::<Cfg>(individual_layout.m_vars, r_vars_mega)
            .map_err(|_| ProofVerifyError::InternalError)?;
        let (mega_prover_setup, _) =
            HachiCommitmentCore::setup_with_layout::<Fp128, D, Cfg>(mega_layout)
                .map_err(|_| ProofVerifyError::InternalError)?;
        let mega_verifier_setup = HachiVerifierSetup {
            expanded: mega_prover_setup.expanded,
        };

        let hachi_combined = jolt_to_hachi(&combined_claim);
        let mega_commitment = commitments[0];

        transcript.append_bytes(
            b"hachi_mega_claim",
            &hachi_combined.to_canonical_u128().to_le_bytes(),
        );
        let mut adapter = JoltToHachiTranscript::new(transcript);

        <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128>>::verify(
            &proof.mega_poly_proof.0,
            &mega_verifier_setup,
            &mut adapter,
            &mega_point,
            &hachi_combined,
            &mega_commitment.0,
        )
        .map_err(|_| ProofVerifyError::InternalError)?;

        // Verify individual proofs (advice etc.)
        let individual_claims = &claims[num_mega..];
        let individual_commitments = &commitments[num_mega..];
        if proof.individual_proofs.len() != individual_claims.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                individual_claims.len(),
                proof.individual_proofs.len(),
            ));
        }

        let hachi_point: Vec<Fp128> = opening_point.iter().map(jolt_to_hachi).collect();
        for (i, ((proof_i, claim_i), commitment_i)) in proof
            .individual_proofs
            .iter()
            .zip(individual_claims.iter())
            .zip(individual_commitments.iter())
            .enumerate()
        {
            transcript.append_bytes(
                b"hachi_individual_item",
                &(i as u64).to_le_bytes(),
            );
            let hachi_claim = jolt_to_hachi(claim_i);
            let mut adapter = JoltToHachiTranscript::new(transcript);
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128>>::verify(
                &proof_i.0,
                &setup.0,
                &mut adapter,
                &hachi_point,
                &hachi_claim,
                &commitment_i.0,
            )
            .map_err(|_| ProofVerifyError::InternalError)?;
        }

        Ok(())
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
}

enum OwnedBlockData<const D: usize> {
    Dense(Vec<CyclotomicRing<Fp128, D>>),
    OneHot(Vec<SparseBlockEntry>),
    Zero,
}

enum PolyRingData<const D: usize> {
    Dense(Vec<CyclotomicRing<Fp128, D>>),
    /// Per-block sparse entries: `per_block_entries[block_idx]` lists the
    /// nonzero ring-element positions within that block.
    OneHot(Vec<Vec<SparseBlockEntry>>),
}

/// Convert a polynomial into ring data split across `blocks_per_poly` blocks
/// of `block_len` ring elements each.
fn poly_to_ring_data<const D: usize>(
    poly: &MultilinearPolynomial<JoltFp128>,
    block_len: usize,
    blocks_per_poly: usize,
) -> PolyRingData<D> {
    match poly {
        MultilinearPolynomial::OneHot(onehot) => {
            let mut per_block: Vec<Vec<SparseBlockEntry>> = vec![Vec::new(); blocks_per_poly];
            for (c, opt) in onehot.nonzero_indices.iter().enumerate() {
                if let Some(idx) = opt {
                    let field_pos = c * onehot.K + *idx as usize;
                    let ring_idx = field_pos / D;
                    let coeff_idx = field_pos % D;
                    let block_idx = ring_idx / block_len;
                    let pos_in_block = ring_idx % block_len;
                    if block_idx < blocks_per_poly {
                        let entry = per_block[block_idx]
                            .iter_mut()
                            .find(|e| e.pos_in_block == pos_in_block);
                        if let Some(existing) = entry {
                            existing.nonzero_coeffs.push(coeff_idx);
                        } else {
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
        _ => {
            let field_coeffs = materialize_coeffs(poly);
            let ring_coeffs = pack_field_to_ring::<D>(&field_coeffs);
            PolyRingData::Dense(ring_coeffs)
        }
    }
}

/// Pack a slice of field elements into ring elements (D field elements per ring).
fn pack_field_to_ring<const D: usize>(field_coeffs: &[Fp128]) -> Vec<CyclotomicRing<Fp128, D>> {
    let num_rings = field_coeffs.len().div_ceil(D);
    let mut rings = Vec::with_capacity(num_rings);
    for chunk in field_coeffs.chunks(D) {
        let mut coeffs = [Fp128::zero(); D];
        coeffs[..chunk.len()].copy_from_slice(chunk);
        rings.push(CyclotomicRing::from_coefficients(coeffs));
    }
    rings
}

pub(super) fn to_hachi_poly(
    poly: &MultilinearPolynomial<JoltFp128>,
) -> DenseMultilinearEvals<Fp128> {
    let evals = materialize_coeffs(poly);
    DenseMultilinearEvals::new_padded(evals)
}

fn materialize_coeffs(poly: &MultilinearPolynomial<JoltFp128>) -> Vec<Fp128> {
    match poly {
        MultilinearPolynomial::LargeScalars(p) => {
            // SAFETY: JoltFp128 is repr(transparent) over Fp128
            #[allow(clippy::missing_transmute_annotations)]
            unsafe {
                std::mem::transmute(p.Z.clone())
            }
        }
        MultilinearPolynomial::BoolScalars(p) => p
            .coeffs
            .iter()
            .map(|&b| if b { Fp128::one() } else { Fp128::zero() })
            .collect(),
        MultilinearPolynomial::U8Scalars(p) => p
            .coeffs
            .iter()
            .map(|&v| Fp128::from_u64(v as u64))
            .collect(),
        MultilinearPolynomial::U16Scalars(p) => p
            .coeffs
            .iter()
            .map(|&v| Fp128::from_u64(v as u64))
            .collect(),
        MultilinearPolynomial::U32Scalars(p) => p
            .coeffs
            .iter()
            .map(|&v| Fp128::from_u64(v as u64))
            .collect(),
        MultilinearPolynomial::U64Scalars(p) => {
            p.coeffs.iter().map(|&v| Fp128::from_u64(v)).collect()
        }
        MultilinearPolynomial::U128Scalars(p) => p
            .coeffs
            .iter()
            .map(|&v| <Fp128 as CanonicalField>::from_canonical_u128_reduced(v))
            .collect(),
        MultilinearPolynomial::I64Scalars(p) => p
            .coeffs
            .iter()
            .map(|&v| {
                if v >= 0 {
                    Fp128::from_u64(v as u64)
                } else {
                    -Fp128::from_u64(v.unsigned_abs())
                }
            })
            .collect(),
        MultilinearPolynomial::I128Scalars(p) => p
            .coeffs
            .iter()
            .map(|&v| {
                let jolt = JoltFp128::from_i128(v);
                jolt_to_hachi(&jolt)
            })
            .collect(),
        MultilinearPolynomial::S128Scalars(p) => p
            .coeffs
            .iter()
            .map(|v| {
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
            })
            .collect(),
        MultilinearPolynomial::OneHot(_) | MultilinearPolynomial::RLC(_) => {
            panic!("OneHot and RLC polynomials cannot be materialized for Hachi commit")
        }
    }
}
