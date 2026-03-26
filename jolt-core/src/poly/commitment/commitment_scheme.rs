use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T;
use std::borrow::Borrow;
use std::fmt::Debug;

use crate::poly::opening_proof::BatchPolynomialSource;
use crate::transcripts::Transcript;
use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::commitment::dory::DoryLayout,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, small_scalar::SmallScalar},
};

pub trait PolynomialBatchSource<F: JoltField>: Sync {
    fn num_polys(&self) -> usize;
    fn get_poly(&self, _idx: usize) -> Option<&MultilinearPolynomial<F>> {
        None
    }
    fn onehot_index(&self, _cycle_idx: usize, _poly_idx: usize) -> Option<u8> {
        None
    }
    /// Fill `buf[0..len]` with `onehot_index(cycle_idx, poly_start + i)` for each `i`.
    /// Implementations should override this to amortize per-cycle work (e.g. loading
    /// the trace entry once across all polynomials).
    fn batch_onehot_indices(&self, cycle_idx: usize, poly_start: usize, buf: &mut [Option<u8>]) {
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = self.onehot_index(cycle_idx, poly_start + i);
        }
    }
    fn num_cycles(&self) -> Option<usize> {
        None
    }
    fn onehot_k(&self) -> Option<usize> {
        None
    }
}

impl<F: JoltField, U: Borrow<MultilinearPolynomial<F>> + Sync> PolynomialBatchSource<F> for [U] {
    fn num_polys(&self) -> usize {
        self.len()
    }
    fn get_poly(&self, idx: usize) -> Option<&MultilinearPolynomial<F>> {
        Some(self[idx].borrow())
    }
}

impl<F: JoltField, U: Borrow<MultilinearPolynomial<F>> + Sync> PolynomialBatchSource<F> for Vec<U> {
    fn num_polys(&self) -> usize {
        self.len()
    }
    fn get_poly(&self, idx: usize) -> Option<&MultilinearPolynomial<F>> {
        Some(self[idx].borrow())
    }
}

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

    /// Build prover setup using the main-trace shape information when the PCS
    /// needs more than a flat `max_num_vars` bound to size its matrices.
    fn setup_prover_from_shape(
        max_log_t: usize,
        max_log_k: usize,
        log_packed: Option<usize>,
    ) -> Self::ProverSetup {
        let max_num_vars = max_log_t
            .checked_add(max_log_k)
            .and_then(|n| n.checked_add(log_packed.unwrap_or(0)))
            .expect("setup_prover_from_shape max_num_vars overflow");
        Self::setup_prover(max_num_vars)
    }

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

    fn batch_commit<S: PolynomialBatchSource<Self::Field>>(
        &self,
        source: &S,
        gens: &Self::ProverSetup,
    ) -> (Vec<Self::Commitment>, Self::BatchOpeningHint);

    /// Homomorphically combines multiple commitments into a single commitment, computed as a
    /// linear combination with the given coefficients.
    fn combine_commitments<C: Borrow<Self::Commitment>>(
        _commitments: &[C],
        _coeffs: &[Self::Field],
    ) -> Self::Commitment {
        todo!("`combine_commitments` should be on a separate `AdditivelyHomomorphic` trait")
    }

    /// Homomorphically combines multiple opening proof hints into a single hint, computed as a
    /// linear combination with the given coefficients.
    fn combine_hints(
        _hints: Vec<Self::OpeningProofHint>,
        _coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        unimplemented!()
    }

    fn prove<ProofTranscript: Transcript>(
        &self,
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
        commitment: &Self::Commitment,
    ) -> (Self::Proof, Option<Self::Field>);

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

    /// Returns the log₂ chunk size for one-hot encoding given the trace length.
    ///
    /// Each PCS can implement its own threshold-based switching policy.
    /// Must be monotonically non-decreasing in `log_T` so that
    /// `setup_prover_from_shape(max_log_T, ...)` sizes the setup correctly.
    fn log_k_chunk_for_trace(log_T: usize) -> usize {
        if log_T >= ONEHOT_CHUNK_THRESHOLD_LOG_T {
            8
        } else {
            4
        }
    }

    /// Returns the one-hot chunk sizes this PCS may use when preprocessing is
    /// sized for `max_log_k`. PCS that switch chunk sizes at runtime should
    /// override this to include every supported mode up to that bound.
    fn supported_log_k_chunks(max_log_k: usize) -> Vec<usize> {
        vec![max_log_k]
    }

    /// Optional PCS-specific consistency check between a batched proof and the
    /// one-hot chunk size validated from `OneHotConfig`.
    fn validate_batch_proof_shape(
        _proof: &Self::BatchedProof,
        _one_hot_log_k_chunk: usize,
    ) -> Result<(), ProofVerifyError> {
        Ok(())
    }
}

pub trait ZkEvalCommitment<C: JoltCurve>: CommitmentScheme {
    /// Returns the evaluation commitment (e.g. y_com) if present in the proof.
    fn eval_commitment(proof: &Self::BatchedProof) -> Option<C::G1>;

    /// Returns the generators used for evaluation commitments in the prover setup.
    fn eval_commitment_gens(setup: &Self::ProverSetup) -> Option<(C::G1, C::G1)>;

    /// Returns the generators used for evaluation commitments in the verifier setup.
    fn eval_commitment_gens_verifier(setup: &Self::VerifierSetup) -> Option<(C::G1, C::G1)>;

    /// Extracts the y_blinding scalar from a batched proof (used in ZK mode).
    #[cfg(feature = "zk")]
    fn eval_blinding(proof: &Self::BatchedProof) -> Option<Self::Field> {
        let _ = proof;
        None
    }

    /// Extracts G1 generators and blinding generator from the prover setup for Pedersen commitments.
    /// Returns None for PCS that don't support ZK Pedersen commitments.
    #[cfg(feature = "zk")]
    fn zk_generators(_setup: &Self::ProverSetup, _count: usize) -> Option<(Vec<C::G1>, C::G1)> {
        None
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
