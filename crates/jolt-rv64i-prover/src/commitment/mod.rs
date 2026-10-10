//! Prover calls on the bit-table commitment seam share packed rows by `Arc`.
use jolt_field::F128;
use jolt_rv64i_arith::BitsRow;
use jolt_rv64i_verifier::commitment::{BitsCommitmentScheme, BitsGeometry, BitsOpening};
use jolt_transcript::Transcript;
use std::sync::Arc;
#[cfg(feature = "test-utils")]
pub mod transparent;

pub trait BitsCommitmentProver: BitsCommitmentScheme {
    type ProverSetup;
    type ProverState;
    fn commit<T: Transcript<Challenge = F128>>(
        setup: &Self::ProverSetup,
        geometry: BitsGeometry,
        bits: &Arc<[BitsRow]>,
        transcript: &mut T,
    ) -> Result<(Self::Commitment, Self::ProverState), Self::Error>;
    fn open<T: Transcript<Challenge = F128>>(
        setup: &Self::ProverSetup,
        state: Self::ProverState,
        opening: &BitsOpening<'_>,
        transcript: &mut T,
    ) -> Result<Self::OpeningProof, Self::Error>;
}
