//! Prover calls on the bit-table commitment seam share packed rows by `Arc`.
use jolt_field::F128;
use jolt_rv64i_arith::BitsRow;
use jolt_rv64i_verifier::commitment::{BitsCommitmentScheme, BitsGeometry, BitsOpening};
use jolt_transcript::Transcript;
use std::sync::Arc;
#[cfg(feature = "test-utils")]
pub mod transparent;

/// Prover half of the bit-table scheme, with commit absorption matching `verify_commit` and opening after column absorption.
/// A scheme retaining packed rows shares their allocation; neither phase copies a trace-sized buffer.
pub trait BitsCommitmentProver: BitsCommitmentScheme {
    /// Scheme setup borrowed by both prover phases.
    type ProverSetup;
    /// State retained after commit and consumed by one opening operation.
    type ProverState;
    /// Commits exactly `2^log_T` packed rows and absorbs all scheme messages after the preamble, before front-end challenges.
    /// Returns the scheme's typed error for unsupported geometry or inconsistent rows, retaining shared rows for the opening phase.
    fn commit<T: Transcript<Challenge = F128>>(
        setup: &Self::ProverSetup,
        geometry: BitsGeometry,
        bits: &Arc<[BitsRow]>,
        transcript: &mut T,
    ) -> Result<(Self::Commitment, Self::ProverState), Self::Error>;
    /// Consumes retained state to open at low-variable-first points, after all 256 columns are absorbed and the column point is drawn.
    /// Returns the scheme's typed error for inconsistent geometry, invalid request dimensions or an unsupported opening.
    fn open<T: Transcript<Challenge = F128>>(
        setup: &Self::ProverSetup,
        state: Self::ProverState,
        opening: &BitsOpening<'_>,
        transcript: &mut T,
    ) -> Result<Self::OpeningProof, Self::Error>;
}
