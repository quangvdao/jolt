use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::borrow::Borrow;
use std::fmt::Debug;

use crate::poly::opening_proof::BatchPolynomialSource;
use crate::transcripts::Transcript;
use crate::{
    field::JoltField,
    poly::commitment::dory::DoryLayout,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, small_scalar::SmallScalar},
};

pub trait CommitmentScheme: Clone + Sync + Send + Default + 'static {
    type Field: JoltField + Sized;
    /// PCS-specific configuration carried by the instance. Opaque to generic code.
    type Config: Clone + Sync + Send + CanonicalSerialize + CanonicalDeserialize;
    type ProverSetup: Clone + Sync + Send + Debug + CanonicalSerialize + CanonicalDeserialize;
    type VerifierSetup: Clone + Sync + Send + Debug + CanonicalSerialize + CanonicalDeserialize;
    type Commitment: Default
        + Debug
        + Sync
        + Send
        + PartialEq
        + CanonicalSerialize
        + CanonicalDeserialize
        + Clone;
    type Proof: Sync + Send + CanonicalSerialize + CanonicalDeserialize + Clone + Debug;
    type BatchedProof: Sync + Send + CanonicalSerialize + CanonicalDeserialize;
    /// Per-polynomial hint from individual `commit()` calls (e.g. Dory row commitments).
    type OpeningProofHint: Sync + Send + Clone + Debug + PartialEq;
    /// Hint produced by `batch_commit()` for the entire batch (e.g. Hachi packed commitment witness).
    type BatchOpeningHint: Sync + Send + Clone + Debug;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup;

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup;

    /// Reconstruct a PCS instance from a batched proof (e.g. for the verifier to
    /// recover PCS-specific configuration serialized during proving).
    fn from_proof(proof: &Self::BatchedProof) -> Self;

    fn config(&self) -> &Self::Config;

    /// Exposes Dory matrix layout when this PCS uses Dory contexts.
    /// Non-Dory schemes return `None`.
    fn dory_layout(_config: &Self::Config) -> Option<DoryLayout> {
        None
    }

    fn commit(
        &self,
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint);

    fn batch_commit<U>(
        &self,
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> (Vec<Self::Commitment>, Self::BatchOpeningHint)
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync;

    fn prove<ProofTranscript: Transcript>(
        &self,
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
        commitment: &Self::Commitment,
    ) -> Self::Proof;

    fn verify<ProofTranscript: Transcript>(
        &self,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError>;

    #[allow(clippy::too_many_arguments)]
    fn batch_prove<ProofTranscript: Transcript, S: BatchPolynomialSource<Self::Field>>(
        &self,
        setup: &Self::ProverSetup,
        poly_source: &S,
        batch_hint: Self::BatchOpeningHint,
        individual_hints: Vec<Self::OpeningProofHint>,
        commitments: &[&Self::Commitment],
        opening_point: &[<Self::Field as JoltField>::Challenge],
        claims: &[Self::Field],
        coeffs: &[Self::Field],
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof;

    #[allow(clippy::too_many_arguments)]
    fn batch_verify<ProofTranscript: Transcript>(
        &self,
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        commitments: &[&Self::Commitment],
        claims: &[Self::Field],
        coeffs: &[Self::Field],
    ) -> Result<(), ProofVerifyError>;

    /// Extract per-polynomial hints from a batch hint.
    /// For PCS where `BatchOpeningHint` contains per-polynomial data (e.g. Dory),
    /// returns a clone of the per-polynomial hints. For PCS where the batch hint
    /// is a single aggregate (e.g. Hachi), returns an empty Vec.
    fn split_batch_hint(batch_hint: &Self::BatchOpeningHint) -> Vec<Self::OpeningProofHint>;

    fn protocol_name() -> &'static [u8];

    /// Number of commitments used to represent all main witness polynomials.
    /// `None` means one commitment per main polynomial.
    /// `Some(1)` means all main polynomials share one packed commitment.
    fn packed_main_commitment_arity() -> Option<usize> {
        None
    }

    /// Whether this PCS uses one-hot decomposition for increment polynomials.
    /// When true, increments are committed as:
    /// - `RdIncRa` + `RdIncMsb`
    /// - `RamIncRa` + `RamIncMsb`
    ///
    /// When false (default), increments use dense `RdInc` / `RamInc`.
    fn uses_onehot_inc() -> bool {
        false
    }
}

pub trait StreamingCommitmentScheme: CommitmentScheme {
    /// The type representing chunk state (tier 1 commitments)
    type ChunkState: Send + Sync + Clone + PartialEq + Debug;

    /// Chunk size in field elements for streaming commit.
    ///
    /// Returns `Some(size)` to use the streaming path where the trace is
    /// split into chunks of `size` cycles, or `None` to use the non-streaming
    /// path (materialize full polynomial, then commit).
    #[allow(non_snake_case)]
    fn streaming_chunk_size(&self, K: usize, T: usize) -> Option<usize>;

    fn process_chunk<T: SmallScalar>(
        &self,
        setup: &Self::ProverSetup,
        chunk: &[T],
    ) -> Self::ChunkState;

    fn process_chunk_onehot(
        &self,
        setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState;

    fn aggregate_chunks(
        &self,
        setup: &Self::ProverSetup,
        onehot_k: Option<usize>,
        tier1_commitments: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint);

    /// Optional PCS-specific streaming batch finalization hook.
    /// If implemented, receives all per-polynomial chunk states and returns the
    /// final commitments + batch hint directly.
    fn aggregate_streaming_batch(
        &self,
        _setup: &Self::ProverSetup,
        _onehot_ks: &[Option<usize>],
        _tier1_per_poly: &[Vec<Self::ChunkState>],
    ) -> Option<(Vec<Self::Commitment>, Self::BatchOpeningHint)> {
        None
    }

    /// Convert per-polynomial hints from the streaming path into a batch hint.
    /// Only called when `streaming_chunk_size` returns `Some`.
    fn streaming_batch_hint(hints: Vec<Self::OpeningProofHint>) -> Self::BatchOpeningHint;
}
