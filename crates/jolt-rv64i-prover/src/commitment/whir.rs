//! Production bit-table commitment adapter using the shared WHIR prover.

use super::BitsCommitmentProver;
use jolt_field::F128;
use jolt_rv64i_arith::BitsRow;
use jolt_rv64i_pcs::{
    commit::{self, ProverState},
    open,
};
use jolt_rv64i_verifier::{
    commitment::{BitsGeometry, BitsOpening},
    whir::{
        error::WhirError,
        wire::{WhirCommitment, WhirOpeningProof},
        WhirBits,
    },
};
use jolt_transcript::Transcript;
use std::sync::Arc;

impl BitsCommitmentProver for WhirBits {
    type ProverSetup = ();
    type ProverState = ProverState;

    fn commit<T: Transcript<Challenge = F128>>(
        (): &(),
        geometry: BitsGeometry,
        bits: &Arc<[BitsRow]>,
        transcript: &mut T,
    ) -> Result<(WhirCommitment, ProverState), WhirError> {
        commit::commit(geometry, bits, transcript)
    }

    fn open<T: Transcript<Challenge = F128>>(
        (): &(),
        state: ProverState,
        opening: &BitsOpening<'_>,
        transcript: &mut T,
    ) -> Result<WhirOpeningProof, WhirError> {
        open::open(state, opening, transcript)
    }
}
