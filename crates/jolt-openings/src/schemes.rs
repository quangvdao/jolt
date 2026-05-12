//! Polynomial commitment scheme (PCS) trait hierarchy.
//!
//! The base verifier/prover traits expose single openings, fused batch
//! openings, and source-based commitment. Homomorphic and ZK traits only add
//! the operations that are genuinely extra for those schemes.

use std::fmt::Debug;

use jolt_crypto::{Commitment, HomomorphicCommitment};
use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{de::DeserializeOwned, Serialize};

use crate::claims::{OpeningClaim, ProverClaim};
use crate::error::OpeningsError;
use crate::sources::{BatchCommitmentSource, CommitmentSource};

/// Verifier-side interface for a polynomial commitment scheme.
pub trait CommitmentSchemeVerifier: Commitment + Clone + Send + Sync + 'static {
    type Field: Field;
    type Proof: Clone + Send + Sync + Serialize + DeserializeOwned;
    type BatchProof: Clone + Send + Sync + Serialize + DeserializeOwned;
    type VerifierSetup: Clone + Send + Sync + Serialize + DeserializeOwned;

    /// Verifies one opening proof.
    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;

    /// Verifies a fused batch-opening proof.
    fn verify_batch(
        claims: Vec<OpeningClaim<Self::Field, Self>>,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;

    /// Binds one transparent opening input to the Fiat-Shamir transcript.
    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    );
}

/// Verifier setup derivable from public parameters without prover setup.
///
/// Transparent schemes such as Dory can build verifier setup from a size
/// parameter alone. Structured-reference-string schemes such as KZG generally
/// cannot: their verifier setup contains trapdoor-derived elements generated
/// during setup, so verifier-only code should receive the verifier setup as an
/// input rather than pretending it can derive it from public generators.
pub trait PublicVerifierSetup: CommitmentSchemeVerifier {
    type PublicParams;

    /// Builds verifier setup directly from public parameters.
    fn verifier_setup(params: Self::PublicParams) -> Self::VerifierSetup;
}

/// Prover-side interface for a polynomial commitment scheme.
pub trait CommitmentScheme: CommitmentSchemeVerifier {
    type ProverSetup: Clone + Send + Sync;
    type Polynomial: CommitmentSource<Self::Field> + From<Vec<Self::Field>>;
    type OpeningHint: Clone + Send + Sync + Default;
    type SetupParams;

    /// Builds prover and verifier setup.
    fn setup(params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup);

    /// Projects prover setup down to verifier setup.
    fn project_verifier_setup(prover_setup: &Self::ProverSetup) -> Self::VerifierSetup;

    /// Commits to one source.
    fn commit<S: CommitmentSource<Self::Field> + ?Sized>(
        source: &S,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint);

    /// Commits to a batch of sources.
    fn commit_batch<B: BatchCommitmentSource<Self::Field>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Output, Self::OpeningHint)> {
        ids.iter()
            .map(|&id| {
                let source = batch.source(id);
                Self::commit(&source, setup)
            })
            .collect()
    }

    /// Proves one opening.
    fn open(
        polynomial: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof;

    /// Proves a fused batch opening.
    fn prove_batch(
        claims: Vec<ProverClaim<Self::Field, Self::Polynomial>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::BatchProof;
}

/// Verifier-side additive combination of commitments.
pub trait AdditivelyHomomorphicVerifier: CommitmentSchemeVerifier
where
    Self::Output: HomomorphicCommitment<Self::Field>,
{
    /// Computes `Σ scalars[i] * commitments[i]`.
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output;
}

/// Prover-side additive combination of commitment hints.
pub trait AdditivelyHomomorphic: CommitmentScheme + AdditivelyHomomorphicVerifier
where
    Self::Output: HomomorphicCommitment<Self::Field>,
{
    /// Computes the hint corresponding to the same linear combination as
    /// [`AdditivelyHomomorphicVerifier::combine`].
    fn combine_hints(
        _hints: Vec<Self::OpeningHint>,
        _scalars: &[Self::Field],
    ) -> Self::OpeningHint {
        Self::OpeningHint::default()
    }
}

/// Verifier-side interface for openings that hide evaluations.
pub trait ZkOpeningSchemeVerifier: CommitmentSchemeVerifier {
    type HidingCommitment: Clone
        + Debug
        + Eq
        + Send
        + Sync
        + 'static
        + Serialize
        + DeserializeOwned
        + AppendToTranscript;

    fn verify_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError>;
}

/// Prover-side interface for openings that hide evaluations.
pub trait ZkOpeningScheme: CommitmentScheme + ZkOpeningSchemeVerifier {
    type Blind: Clone + Send + Sync;

    /// Commits in the scheme's ZK/hiding mode.
    fn commit_zk<S: CommitmentSource<Self::Field> + ?Sized>(
        source: &S,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint);

    /// Commits to a batch of sources in the scheme's ZK/hiding mode.
    fn commit_batch_zk<B: BatchCommitmentSource<Self::Field>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Output, Self::OpeningHint)> {
        ids.iter()
            .map(|&id| {
                let source = batch.source(id);
                Self::commit_zk(&source, setup)
            })
            .collect()
    }

    /// Opens a ZK/hiding commitment using the hint returned by
    /// [`commit_zk`](Self::commit_zk).
    fn open_zk(
        polynomial: &Self::Polynomial,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind);
}
