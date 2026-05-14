//! Stub `CommitmentScheme` for symbolic transpilation (stages 1-7).
//!
//! # Purpose
//!
//! Jolt's verifier is generic over `PCS: CommitmentScheme`. To run the verifier with
//! symbolic types (`MleAst` instead of `Fr`), we need a `CommitmentScheme` that works
//! with `MleAst`. This module provides that stub.
//!
//! # Current Usage (Stages 1-7)
//!
//! The transpiler runs the Jolt verifier symbolically to record all field operations:
//!
//! ```ignore
//! // In main.rs - the verifier is instantiated with symbolic types
//! let verifier = TranspilableVerifier::<
//!     MleAst,                    // Symbolic field (records operations)
//!     AstCommitmentScheme,       // This stub (satisfies trait bounds)
//!     PoseidonAstTranscript,     // Symbolic transcript
//!     AstOpeningAccumulator,     // Collects opening claims
//! >::new(...);
//!
//! verifier.verify(&proof, ...);  // Runs stages 1-7, records AST
//! ```
//!
//! Stages 1-7 are **sumcheck-based** and don't call PCS methods. This stub satisfies
//! the `CommitmentScheme` trait bound without doing any work. Methods that would be
//! called during proving (`commit`, `prove`) panic since we only run verification.
//!
//! # Future: Stage 8 (PCS Verification)
//!
//! Stage 8 verifies polynomial commitment openings. Currently NOT transpiled because:
//! - **Dory**: Uses pairings (very expensive in-circuit, not practical)
//! - **Hyrax**: Technically transpilable (MSM becomes sumchecks + scalar muls), but
//!   the current `TranspilableVerifier` uses Dory, not Hyrax with recursion support
//!
//! With proper recursion setup (Hyrax over Grumpkin, sumcheck-based MSM verification),
//! stage 8 could be transpiled.
//! The `todo!()` methods (`combine_commitments`, `verify`) mark where this would plug in.
//!
//! # Why Vec<MleAst> Instead of Unit Types?
//!
//! Types like `AstProof(Vec<MleAst>)` use vectors instead of `()` for future
//! extensibility. If Hyrax-over-Grumpkin is ever transpiled, these types could
//! hold symbolic curve points.

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_crypto::Commitment;
use jolt_openings::{
    AdditivelyHomomorphic, AdditivelyHomomorphicVerifier, BatchOpeningProverResult,
    BatchOpeningPublic, CommitmentScheme, CommitmentSchemeVerifier, CommitmentSource,
    EvaluationCommitmentProver, EvaluationCommitmentScheme, LinearCombinationOpeningSource,
    LinearOpeningScheme, LinearOpeningSchemeVerifier, OpeningClaim, OpeningsError,
    ProverBatchOpeningTerm, ProverClaim, PublicVerifierSetup, SourceId, VerifierBatchOpeningTerm,
    ZkBatchOpeningProverResult, ZkLinearOpeningScheme, ZkLinearOpeningSchemeVerifier,
    ZkOpeningScheme, ZkOpeningSchemeVerifier,
};
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};
use zklean_extractor::{mle_ast::MleAst, AstCommitment};

use super::ast_curve::AstGroupElement;

// =============================================================================
// Type Definitions
// =============================================================================

/// Symbolic commitment scheme for MleAst transpilation.
///
/// This is used to instantiate `TranspilableVerifier<MleAst, AstCommitmentScheme, ...>`
/// for symbolic execution that generates Gnark circuits.
#[derive(Clone, Debug)]
pub struct AstCommitmentScheme;

/// Verifier setup - empty for symbolic execution
#[derive(
    Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize, Serialize, Deserialize,
)]
pub struct AstVerifierSetup;

/// Prover setup - empty, never used in verification
#[derive(
    Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize, Serialize, Deserialize,
)]
pub struct AstProverSetup;

/// Opening proof - vector of MleAst for future PCS extensibility
#[derive(
    Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize, Serialize, Deserialize,
)]
pub struct AstProof(pub Vec<MleAst>);

/// Batched opening proof - vector of MleAst for future PCS extensibility
#[derive(
    Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize, Serialize, Deserialize,
)]
pub struct AstBatchedProof(pub Vec<MleAst>);

/// Opening proof hint - not used in verification
#[derive(
    Clone,
    Debug,
    Default,
    PartialEq,
    CanonicalSerialize,
    CanonicalDeserialize,
    Serialize,
    Deserialize,
)]
pub struct AstOpeningHint;

// =============================================================================
// Canonical commitment scheme implementation
// =============================================================================

impl Commitment for AstCommitmentScheme {
    type Output = AstCommitment;
}

impl CommitmentSchemeVerifier for AstCommitmentScheme {
    type Field = MleAst;
    type Proof = AstProof;
    type BatchProof = AstBatchedProof;
    type VerifierSetup = AstVerifierSetup;

    fn verify(
        _commitment: &Self::Output,
        _point: &[Self::Field],
        _eval: Self::Field,
        _proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        unimplemented!("AstCommitmentScheme::verify is not needed for stages 1-7 transpilation")
    }

    fn verify_batch(
        _claims: Vec<OpeningClaim<Self::Field, Self>>,
        _proof: &Self::BatchProof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        unimplemented!(
            "AstCommitmentScheme::verify_batch is not needed for stages 1-7 transpilation"
        )
    }

    fn bind_opening_inputs(
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
        _point: &[Self::Field],
        _eval: &Self::Field,
    ) {
    }
}

impl PublicVerifierSetup for AstCommitmentScheme {
    type PublicParams = ();

    fn verifier_setup(_params: Self::PublicParams) -> Self::VerifierSetup {
        AstVerifierSetup
    }
}

impl CommitmentScheme for AstCommitmentScheme {
    type ProverSetup = AstProverSetup;
    type OpeningHint = AstOpeningHint;
    type SetupParams = usize;

    fn setup(_params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
        (AstProverSetup, AstVerifierSetup)
    }

    fn project_verifier_setup(_prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
        AstVerifierSetup
    }

    fn commit<S: CommitmentSource<Self::Field> + ?Sized>(
        _source: &S,
        _setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        panic!("AstCommitmentScheme::commit should never be called during verification")
    }

    fn open<S>(
        _polynomial: &S,
        _point: &[Self::Field],
        _eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Option<Self::OpeningHint>,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof
    where
        S: CommitmentSource<Self::Field> + ?Sized,
    {
        panic!("AstCommitmentScheme::open should never be called during verification")
    }

    fn prove_batch<S>(
        _claims: Vec<ProverClaim<Self::Field, S>>,
        _hints: Vec<Self::OpeningHint>,
        _setup: &Self::ProverSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::BatchProof
    where
        S: CommitmentSource<Self::Field>,
    {
        panic!("AstCommitmentScheme::prove_batch should never be called during verification")
    }
}

impl LinearOpeningSchemeVerifier for AstCommitmentScheme {
    fn verify_batch_opening<ClaimId, SourceIdT>(
        _terms: Vec<VerifierBatchOpeningTerm<Self::Field, Self, ClaimId, SourceIdT>>,
        _proof: &Self::BatchProof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<BatchOpeningPublic<Self::Field, (), ClaimId>, OpeningsError>
    where
        Self: Sized,
        SourceIdT: SourceId,
    {
        unimplemented!(
            "AstCommitmentScheme::verify_batch_opening is not needed for stages 1-7 transpilation"
        )
    }
}

impl LinearOpeningScheme for AstCommitmentScheme {
    fn prove_batch_opening<B, ClaimId>(
        _terms: Vec<ProverBatchOpeningTerm<Self::Field, ClaimId, B::Id>>,
        _source_batch: &mut B,
        _setup: &Self::ProverSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> BatchOpeningProverResult<Self, ClaimId>
    where
        Self: Sized,
        B: LinearCombinationOpeningSource<Self::Field, Self::OpeningHint>,
    {
        panic!(
            "AstCommitmentScheme::prove_batch_opening should never be called during verification"
        )
    }
}

impl AdditivelyHomomorphicVerifier for AstCommitmentScheme {
    fn combine(_commitments: &[Self::Output], _scalars: &[Self::Field]) -> Self::Output {
        unimplemented!("AstCommitmentScheme::combine is not needed for stages 1-7 transpilation")
    }
}

impl AdditivelyHomomorphic for AstCommitmentScheme {
    fn combine_hints(
        _hints: Vec<Self::OpeningHint>,
        _scalars: &[Self::Field],
    ) -> Self::OpeningHint {
        AstOpeningHint
    }
}

impl ZkOpeningSchemeVerifier for AstCommitmentScheme {
    type HidingCommitment = AstGroupElement;

    fn verify_zk(
        _commitment: &Self::Output,
        _point: &[Self::Field],
        _proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        unimplemented!("AstCommitmentScheme::verify_zk is not needed for stages 1-7 transpilation")
    }

    fn verify_batch_zk(
        _claims: Vec<OpeningClaim<Self::Field, Self>>,
        _proof: &Self::BatchProof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        unimplemented!(
            "AstCommitmentScheme::verify_batch_zk is not needed for stages 1-7 transpilation"
        )
    }

    fn bind_zk_opening_inputs(
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
        _point: &[Self::Field],
        _hiding_commitment: &Self::HidingCommitment,
    ) {
    }
}

impl ZkOpeningScheme for AstCommitmentScheme {
    type Blind = MleAst;

    fn commit_zk<S: CommitmentSource<Self::Field> + ?Sized>(
        _source: &S,
        _setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        panic!("AstCommitmentScheme::commit_zk should never be called during verification")
    }

    fn open_zk<S>(
        _polynomial: &S,
        _point: &[Self::Field],
        _eval: Self::Field,
        _setup: &Self::ProverSetup,
        _hint: Self::OpeningHint,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind)
    where
        S: CommitmentSource<Self::Field> + ?Sized,
    {
        panic!("AstCommitmentScheme::open_zk should never be called during verification")
    }

    fn prove_batch_zk<S>(
        _claims: Vec<ProverClaim<Self::Field, S>>,
        _hints: Vec<Self::OpeningHint>,
        _setup: &Self::ProverSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::BatchProof, Self::HidingCommitment, Self::Blind)
    where
        S: CommitmentSource<Self::Field>,
    {
        panic!("AstCommitmentScheme::prove_batch_zk should never be called during verification")
    }
}

impl ZkLinearOpeningSchemeVerifier for AstCommitmentScheme {
    fn verify_batch_opening_zk<ClaimId, SourceIdT>(
        _terms: Vec<VerifierBatchOpeningTerm<Self::Field, Self, ClaimId, SourceIdT>>,
        _proof: &Self::BatchProof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<BatchOpeningPublic<Self::Field, Self::HidingCommitment, ClaimId>, OpeningsError>
    where
        Self: Sized,
        SourceIdT: SourceId,
    {
        unimplemented!(
            "AstCommitmentScheme::verify_batch_opening_zk is not needed for stages 1-7 transpilation"
        )
    }
}

impl ZkLinearOpeningScheme for AstCommitmentScheme {
    fn prove_batch_opening_zk<B, ClaimId>(
        _terms: Vec<ProverBatchOpeningTerm<Self::Field, ClaimId, B::Id>>,
        _source_batch: &mut B,
        _setup: &Self::ProverSetup,
        _transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> ZkBatchOpeningProverResult<Self, ClaimId>
    where
        Self: Sized,
        B: LinearCombinationOpeningSource<Self::Field, Self::OpeningHint>,
    {
        panic!(
            "AstCommitmentScheme::prove_batch_opening_zk should never be called during verification"
        )
    }
}

impl EvaluationCommitmentScheme<AstGroupElement> for AstCommitmentScheme {
    fn batch_eval_commitment(_proof: &Self::BatchProof) -> Option<AstGroupElement> {
        None
    }

    fn eval_commitment_gens_verifier(
        _setup: &Self::VerifierSetup,
    ) -> Option<(AstGroupElement, AstGroupElement)> {
        None
    }
}

impl EvaluationCommitmentProver<AstGroupElement> for AstCommitmentScheme {
    fn eval_commitment_gens(
        _setup: &Self::ProverSetup,
    ) -> Option<(AstGroupElement, AstGroupElement)> {
        None
    }

    fn zk_generators(
        _setup: &Self::ProverSetup,
        _count: usize,
    ) -> Option<(Vec<AstGroupElement>, AstGroupElement)> {
        None
    }
}
