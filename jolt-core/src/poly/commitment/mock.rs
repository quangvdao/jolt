use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    poly::opening_proof::BatchPolynomialSource,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, small_scalar::SmallScalar},
};

use super::commitment_scheme::{CommitmentScheme, PolynomialBatchSource};

#[derive(Clone)]
pub struct MockCommitScheme<F: JoltField> {
    _marker: PhantomData<F>,
}

impl<F: JoltField> Default for MockCommitScheme<F> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

#[derive(Default, Debug, PartialEq, Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct MockCommitment<F: JoltField> {
    _field: PhantomData<F>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MockProof<F: JoltField> {
    opening_point: Vec<F::Challenge>,
}

impl<F> CommitmentScheme for MockCommitScheme<F>
where
    F: JoltField,
{
    type Field = F;
    type Config = ();
    type ProverSetup = ();
    type VerifierSetup = ();
    type Commitment = MockCommitment<F>;
    type Proof = MockProof<F>;
    type BatchedProof = MockProof<F>;
    type OpeningProofHint = ();
    type BatchOpeningHint = ();

    fn setup_prover(_num_vars: usize) -> Self::ProverSetup {}

    fn setup_verifier(_setup: &Self::ProverSetup) -> Self::VerifierSetup {}

    fn from_proof(_proof: &Self::BatchedProof) -> Self {
        Self::default()
    }

    fn config(&self) -> &() {
        &()
    }

    fn commit(
        &self,
        _poly: &MultilinearPolynomial<Self::Field>,
        _setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        (MockCommitment::default(), ())
    }

    fn batch_commit<S: PolynomialBatchSource<Self::Field>>(
        &self,
        source: &S,
        gens: &Self::ProverSetup,
    ) -> (Vec<Self::Commitment>, Self::BatchOpeningHint) {
        let commitments = (0..source.num_polys())
            .map(|i| self.commit(source.get_poly(i).unwrap(), gens).0)
            .collect();
        (commitments, ())
    }

    fn prove<ProofTranscript: Transcript>(
        &self,
        _setup: &Self::ProverSetup,
        _poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        _hint: Option<Self::OpeningProofHint>,
        _transcript: &mut ProofTranscript,
        _commitment: &Self::Commitment,
    ) -> (Self::Proof, Option<Self::Field>) {
        (
            MockProof {
                opening_point: opening_point.to_owned(),
            },
            None,
        )
    }

    fn verify<ProofTranscript: Transcript>(
        &self,
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        _opening: &Self::Field,
        _commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        assert_eq!(proof.opening_point, opening_point);
        Ok(())
    }

    fn batch_prove<ProofTranscript: Transcript, S: BatchPolynomialSource<Self::Field>>(
        &self,
        _setup: &Self::ProverSetup,
        _poly_source: &S,
        _batch_hint: Self::BatchOpeningHint,
        _individual_hints: Vec<Self::OpeningProofHint>,
        _commitments: &[&Self::Commitment],
        opening_point: &[<Self::Field as JoltField>::Challenge],
        _claims: &[Self::Field],
        _coeffs: &[Self::Field],
        _transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        MockProof {
            opening_point: opening_point.to_owned(),
        }
    }

    fn batch_verify<ProofTranscript: Transcript>(
        &self,
        proof: &Self::BatchedProof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        _commitments: &[&Self::Commitment],
        _claims: &[Self::Field],
        _coeffs: &[Self::Field],
    ) -> Result<(), ProofVerifyError> {
        assert_eq!(proof.opening_point, opening_point);
        Ok(())
    }

    fn split_batch_hint(_batch_hint: &Self::BatchOpeningHint) -> Vec<Self::OpeningProofHint> {
        vec![]
    }

    fn protocol_name() -> &'static [u8] {
        b"mock_commit"
    }
}

impl<F> super::commitment_scheme::StreamingCommitmentScheme for MockCommitScheme<F>
where
    F: JoltField,
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
    }

    fn process_chunk_onehot(
        &self,
        _setup: &Self::ProverSetup,
        _onehot_k: usize,
        _chunk: &[Option<usize>],
    ) -> Self::ChunkState {
    }

    fn aggregate_chunks(
        &self,
        _setup: &Self::ProverSetup,
        _onehot_k: Option<usize>,
        _tier1_commitments: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        (MockCommitment::default(), ())
    }

    fn streaming_batch_hint(_hints: Vec<Self::OpeningProofHint>) -> Self::BatchOpeningHint {}
}
