use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_openings::{BatchCommitmentSource, OneHotEntries, SourceRow};
use rayon::prelude::*;
use std::borrow::Borrow;
use std::fmt::Debug;

use crate::transcripts::Transcript;
use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, small_scalar::SmallScalar},
};

pub trait CommitmentScheme: Clone + Sync + Send + 'static {
    type Field: JoltField + Sized;
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
    /// A hint that helps the prover compute an opening proof. Typically some byproduct of
    /// the commitment computation, e.g. for Dory the Pedersen commitments to the rows can be
    /// used as a hint for the opening proof.
    type OpeningProofHint: Sync + Send + Clone + Debug + PartialEq;

    /// Generates the prover setup for this PCS. `max_num_vars` is the maximum number of
    /// variables of any polynomial that will be committed using this setup.
    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup;

    /// Generates the verifier setup from the prover setup.
    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup;

    /// Commits to a multilinear polynomial using the provided setup.
    ///
    /// # Arguments
    /// * `poly` - The multilinear polynomial to commit to
    /// * `setup` - The prover setup for the commitment scheme
    ///
    /// # Returns
    /// A tuple containing the commitment to the polynomial and a hint that can be used
    /// to optimize opening proof generation
    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint);

    /// Commits to multiple multilinear polynomials in batch.
    ///
    /// # Arguments
    /// * `polys` - A slice of multilinear polynomials to commit to
    /// * `gens` - The prover setup for the commitment scheme
    ///
    /// # Returns
    /// A vector of commitments, one for each input polynomial
    fn batch_commit<U>(
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync;

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

    /// Generates a proof of evaluation for a polynomial at a specific point.
    ///
    /// # Arguments
    /// * `setup` - The prover setup for the commitment scheme
    /// * `poly` - The multilinear polynomial being proved
    /// * `opening_point` - The point at which the polynomial is evaluated
    /// * `hint` - An optional hint that helps optimize the proof generation.
    ///   When `None`, implementations should compute the hint internally if needed.
    /// * `transcript` - The transcript for Fiat-Shamir transformation
    ///
    /// # Returns
    /// A tuple containing:
    /// - The proof of the polynomial evaluation at the specified point
    /// - An optional ZK blinding factor (y_blinding) for use in BlindFold; None for non-ZK schemes
    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> (Self::Proof, Option<Self::Field>);

    /// Verifies a proof of polynomial evaluation at a specific point.
    ///
    /// # Arguments
    /// * `proof` - The proof to be verified
    /// * `setup` - The verifier setup for the commitment scheme
    /// * `transcript` - The transcript for Fiat-Shamir transformation
    /// * `opening_point` - The point at which the polynomial is evaluated
    /// * `opening` - The claimed evaluation value of the polynomial at the opening point
    /// * `commitment` - The commitment to the polynomial
    ///
    /// # Returns
    /// Ok(()) if the proof is valid, otherwise a ProofVerifyError
    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError>;

    fn protocol_name() -> &'static [u8];
}

pub trait ZkEvalCommitment<C: JoltCurve>: CommitmentScheme {
    /// Returns the evaluation commitment (e.g. y_com) if present in the proof.
    fn eval_commitment(proof: &Self::Proof) -> Option<C::G1>;

    /// Returns the generators used for evaluation commitments in the prover setup.
    fn eval_commitment_gens(setup: &Self::ProverSetup) -> Option<(C::G1, C::G1)>;

    /// Returns the generators used for evaluation commitments in the verifier setup.
    fn eval_commitment_gens_verifier(setup: &Self::VerifierSetup) -> Option<(C::G1, C::G1)>;

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

    /// Compute tier 1 commitment for a chunk of small scalar values
    fn process_chunk<T: SmallScalar>(setup: &Self::ProverSetup, chunk: &[T]) -> Self::ChunkState;

    /// Compute tier 1 commitment for a chunk of one-hot values
    fn process_chunk_onehot(
        setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState;

    /// Compute tier 2 commitment from accumulated tier 1 commitments
    fn aggregate_chunks(
        setup: &Self::ProverSetup,
        onehot_k: Option<usize>,
        tier1_commitments: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint);

    /// Compute a tier 1 commitment from a source row.
    ///
    /// This keeps the CycleMajor prover call site source-based while the rest
    /// of `jolt-core` is still on the in-core PCS trait family. Once
    /// `jolt-core` is fully parameterized by `jolt-openings`, this logic moves
    /// behind the concrete PCS backend's `commit_batch` implementation.
    fn process_source_row(
        setup: &Self::ProverSetup,
        row: SourceRow<'_, jolt_field::Fr>,
    ) -> (Self::ChunkState, Option<usize>) {
        match row {
            SourceRow::I128(values) => (Self::process_chunk(setup, values), None),
            SourceRow::OneHot(row) => {
                let onehot_k = 1usize << row.log_domain_size;
                let indices: Vec<Option<usize>> = match row.entries {
                    OneHotEntries::OnePerColumn(indices) => {
                        indices.iter().map(|index| Some(index.get())).collect()
                    }
                    OneHotEntries::MaybeZero(indices) => {
                        indices.iter().map(|index| index.map(|i| i.get())).collect()
                    }
                };
                (
                    Self::process_chunk_onehot(setup, onehot_k, &indices),
                    Some(onehot_k),
                )
            }
            SourceRow::FieldElements(_) => {
                panic!("source-batch streaming expects compact CycleMajor rows")
            }
        }
    }

    /// Commits to a batch of sources through the source-row traversal API.
    fn commit_batch<B: BatchCommitmentSource<jolt_field::Fr>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)> {
        if ids.is_empty() {
            return Vec::new();
        }

        let max_num_vars = ids
            .iter()
            .map(|&id| batch.num_vars(id))
            .max()
            .expect("ids is non-empty");
        let sigma = max_num_vars.div_ceil(2);

        let row_commitments =
            batch.map_rows(sigma, ids, |_, row| Self::process_source_row(setup, row));

        (0..ids.len())
            .into_par_iter()
            .map(|source_index| {
                let tier1_with_kind: Vec<_> = row_commitments
                    .iter()
                    .flat_map(|row| row.get(source_index).cloned())
                    .collect();
                let onehot_k = tier1_with_kind.iter().find_map(|(_, onehot_k)| *onehot_k);
                debug_assert!(
                    tier1_with_kind
                        .iter()
                        .all(|(_, row_onehot_k)| *row_onehot_k == onehot_k),
                    "source changed row encoding within one batch commitment",
                );
                let tier1_commitments: Vec<_> = tier1_with_kind
                    .into_iter()
                    .map(|(chunk, _)| chunk)
                    .collect();
                Self::aggregate_chunks(setup, onehot_k, &tier1_commitments)
            })
            .collect()
    }

    /// Commits to a batch of sources in the build's ZK commitment mode.
    #[cfg(feature = "zk")]
    fn commit_batch_zk<B: BatchCommitmentSource<jolt_field::Fr>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)> {
        Self::commit_batch(batch, ids, setup)
    }
}
