//! PCS traits and homomorphic batch openings for the Jolt zkVM.
//!
//! The crate owns the abstract polynomial-commitment boundary. Protocol code
//! asks to commit and open sources; PCS backends decide whether to materialize,
//! stream, parallelize by row, or use a backend-specific schedule.
//!
//! # Trait Hierarchy
//!
//! ```text
//! CommitmentSchemeVerifier        verify / verify_batch
//!          │
//! CommitmentScheme                commit / commit_batch / open / prove_batch
//! ShapedCommitmentScheme          commit_with_shape
//!
//! PublicVerifierSetup             verifier_setup from public params
//!
//! AdditivelyHomomorphicVerifier   combine commitments
//!          │
//! AdditivelyHomomorphic           combine opening hints
//!
//! ZkOpeningSchemeVerifier         verify_zk
//!          │
//! ZkOpeningScheme                 commit_zk / commit_batch_zk / open_zk
//! ShapedZkOpeningScheme           commit_zk_with_shape
//! ```

mod claims;
mod error;
mod homomorphic;
#[cfg(any(test, feature = "test-utils"))]
pub mod mock;
mod schemes;
mod sources;

pub use claims::{OpeningClaim, ProverClaim};
pub use error::OpeningsError;
pub use homomorphic::{
    homomorphic_prove_batch, homomorphic_verify_batch, rlc_combine, rlc_combine_scalars,
};
pub use schemes::{
    AdditivelyHomomorphic, AdditivelyHomomorphicVerifier, CommitmentScheme,
    CommitmentSchemeVerifier, EvaluationCommitmentProver, EvaluationCommitmentScheme,
    PublicVerifierSetup, ShapedCommitmentScheme, ShapedZkOpeningScheme, ZkOpeningScheme,
    ZkOpeningSchemeVerifier,
};
pub use sources::{
    BatchCommitmentSource, CommitmentSource, OneHotEntries, OneHotIndex, OneHotRow, SourceId,
    SourceRow,
};
