use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_openings::BatchCommitmentSource;
use std::borrow::Borrow;
use std::fmt::Debug;

use crate::transcripts::Transcript;
use crate::{
    curve::JoltCurve, field::JoltField, poly::multilinear_polynomial::MultilinearPolynomial,
    utils::errors::ProofVerifyError,
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

/// ZK opening data needed by Stage 8 and BlindFold while `jolt-core` still uses
/// the in-core PCS trait family.
///
/// This is a temporary compatibility trait. The canonical cutover should move
/// these capabilities onto `jolt-openings` / backend-owned extension traits.
pub trait ZkOpeningSupport<C: JoltCurve>: CommitmentScheme {
    /// Returns the hiding commitment to the opened evaluation from a batch
    /// proof, if the proof was produced in ZK mode.
    fn batch_eval_commitment(_proof: &Self::BatchedProof) -> Option<C::G1> {
        None
    }

    /// Returns the generators used for Dory-style evaluation commitments in
    /// the prover setup.
    fn eval_commitment_gens(setup: &Self::ProverSetup) -> Option<(C::G1, C::G1)>;

    /// Returns the generators used for Dory-style evaluation commitments in
    /// the verifier setup.
    fn eval_commitment_gens_verifier(setup: &Self::VerifierSetup) -> Option<(C::G1, C::G1)>;

    /// Extracts G1 generators and the blinding generator from the prover setup
    /// for BlindFold Pedersen commitments.
    ///
    /// Returns `None` for PCS backends that do not support ZK Pedersen
    /// commitments.
    #[cfg(feature = "zk")]
    fn zk_generators(_setup: &Self::ProverSetup, _count: usize) -> Option<(Vec<C::G1>, C::G1)> {
        None
    }
}

/// Source-batch commitment support for the old in-core PCS trait family.
///
/// This is a temporary compatibility boundary while `jolt-core` still stores
/// proofs, hints, and setup parameters in the pre-`jolt-openings` associated
/// types. A PCS only implements this trait when it has an implementation
/// matching the source row shapes it accepts.
///
/// After the full PCS cutover, callers should use
/// `jolt_openings::CommitmentScheme::commit_batch` and
/// `jolt_openings::ZkOpeningScheme::commit_batch_zk` directly.
pub trait SourceBatchCommitmentScheme: CommitmentScheme {
    /// Commits to a batch of sources through the source-row traversal API.
    fn commit_batch<B: BatchCommitmentSource<jolt_field::Fr>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>;

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

/// Batch-opening support for the old in-core PCS trait family.
///
/// This compatibility trait exposes Stage 8 through a batch-shaped proof while
/// `jolt-core` still stores old PCS associated types. The current Stage 8
/// implementation builds one homomorphically RLC-combined polynomial before
/// entering the PCS, so the compatibility method receives a single combined
/// opening group and returns `BatchedProof`.
pub trait BatchOpeningScheme: CommitmentScheme {
    /// Proves the already-combined Stage 8 opening group.
    fn prove_batch<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> (Self::BatchedProof, Option<Self::Field>);

    /// Verifies the already-combined Stage 8 opening group.
    fn verify_batch<ProofTranscript: Transcript>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError>;
}
