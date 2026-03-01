use std::borrow::Borrow;
use std::marker::PhantomData;

use super::wrappers::{jolt_to_hachi, ArkBridge, Fp128, JoltToHachiTranscript};
use crate::field::fp128::JoltFp128;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{CommitmentScheme, StreamingCommitmentScheme};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::BatchPolynomialSource;
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;
use crate::utils::small_scalar::SmallScalar;
use hachi_pcs::primitives::multilinear_evals::DenseMultilinearEvals;
use hachi_pcs::protocol::commitment::CommitmentConfig;
use hachi_pcs::protocol::commitment::RingCommitment;
use hachi_pcs::protocol::proof::{HachiCommitmentHint, HachiProof};
use hachi_pcs::protocol::HachiCommitmentScheme;
use hachi_pcs::protocol::HachiProverSetup;
use hachi_pcs::protocol::HachiVerifierSetup;
use hachi_pcs::CommitmentScheme as HachiCommitmentSchemeTrait;
use hachi_pcs::{CanonicalField, FieldCore, FromSmallInt};
use rayon::prelude::*;

#[derive(Clone, Default)]
pub struct JoltHachiCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
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
    type BatchedProof = ArkBridge<HachiProof<Fp128, D>>;
    type OpeningProofHint = HachiCommitmentHint<Fp128, D>;

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
        let hachi_poly = to_hachi_poly(poly);
        let (commitment, hint) = <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<
            Fp128,
        >>::commit(&hachi_poly, &setup.0)
        .expect("Hachi commit failed");
        (ArkBridge(commitment), hint)
    }

    fn batch_commit<U>(
        &self,
        polys: &[U],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: Borrow<MultilinearPolynomial<JoltFp128>> + Sync,
    {
        polys
            .par_iter()
            .map(|p| self.commit(p.borrow(), setup))
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
        let hachi_poly = to_hachi_poly(poly);
        let hachi_point: Vec<Fp128> = opening_point.iter().map(jolt_to_hachi).collect();
        let mut adapter = JoltToHachiTranscript::new(transcript);

        let proof = <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128>>::prove(
            &setup.0,
            &hachi_poly,
            &hachi_point,
            hint,
            &mut adapter,
            &commitment.0,
        )
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

    fn batch_prove<ProofTranscript: Transcript, S: BatchPolynomialSource<JoltFp128>>(
        &self,
        setup: &Self::ProverSetup,
        poly_source: &S,
        _hints: Vec<Self::OpeningProofHint>,
        _commitments: &[&Self::Commitment],
        opening_point: &[JoltFp128],
        claims: &[JoltFp128],
        coeffs: &[JoltFp128],
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        // Non-homomorphic fallback: build the RLC polynomial and commit/prove from scratch.
        // The mega-polynomial design (Phase 4) will replace this.
        let joint_poly = poly_source.build_joint_polynomial(coeffs);
        let (joint_commitment, joint_hint) = self.commit(&joint_poly, setup);

        let _joint_claim: JoltFp128 = coeffs.iter().zip(claims).map(|(c, v)| *c * *v).sum();

        self.prove(
            setup,
            &joint_poly,
            opening_point,
            Some(joint_hint),
            transcript,
            &joint_commitment,
        )
    }

    fn batch_verify<ProofTranscript: Transcript>(
        &self,
        _proof: &Self::BatchedProof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut ProofTranscript,
        _opening_point: &[JoltFp128],
        _commitments: &[&Self::Commitment],
        _claims: &[JoltFp128],
        _coeffs: &[JoltFp128],
    ) -> Result<(), ProofVerifyError> {
        // TODO(hachi): batch_verify needs the joint commitment from batch_prove.
        // The mega-polynomial design (Phase 4) will replace this path entirely.
        Err(ProofVerifyError::InternalError)
    }

    fn protocol_name() -> &'static [u8] {
        b"Hachi"
    }
}

impl<const D: usize, Cfg> StreamingCommitmentScheme for JoltHachiCommitmentScheme<D, Cfg>
where
    Cfg: CommitmentConfig + Default,
{
    type ChunkState = ();

    #[allow(non_snake_case)]
    fn streaming_chunk_size(&self, _K: usize, _T: usize) -> Option<usize> {
        // Hachi streaming requires chunk alignment between dense (T-element)
        // and one-hot (K*T-element) polynomial layouts. Returning None forces
        // the non-streaming path (materialize full polynomial, then commit).
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

/// Convert a Jolt `MultilinearPolynomial<JoltFp128>` to Hachi's `DenseMultilinearEvals<Fp128>`.
///
/// Materializes all coefficients as field elements regardless of the underlying
/// compact/sparse representation. The conversion from `JoltFp128` to `Fp128` is
/// zero-cost due to repr(transparent).
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
