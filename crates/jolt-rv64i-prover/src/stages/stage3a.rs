//! Prover driver for the shared seventeen short router rounds.

use crate::error::Rv64iProverError;
use crate::plane::{Rv64iPlane, Rv64iWitness};
use crate::reference::routers::RouterShortPrepare;
use jolt_crypto::NoCommitment;
use jolt_field::{JoltField, F128};
use jolt_kernels::ProofSession;
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::driver::StageProver;
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::commitment::BitsCommitmentScheme;
use jolt_rv64i_verifier::error::Rv64iVerifierError;
use jolt_rv64i_verifier::proof::{BatchProof, RouterFoldValues};
use jolt_rv64i_verifier::stages::stage2::Output as Stage2Output;
use jolt_rv64i_verifier::stages::stage3a;
use jolt_rv64i_verifier::stages::stage3a::{verify::Inputs, Output};
use jolt_rv64i_verifier::stages::stage3a::{
    RouterShort, Stage3aChallenges, Stage3aInputClaims, Stage3aInputPoints, Stage3aOutputClaims,
    Stage3aOutputPoints, Stage3aSumchecks as VerifierStage3aSumchecks,
};
use jolt_rv64i_verifier::statement::CheckedInputs;
use jolt_sumcheck::{ClearSumcheckRecorder, SequentialRounds};
use jolt_transcript::Transcript;
use std::ops::Deref;

/// Local wrapper sharing the verifier batch and its low-variable-first geometry.
pub struct Stage3aSumchecks<F: JoltField>(pub VerifierStage3aSumchecks<F>);
impl<F: JoltField> Deref for Stage3aSumchecks<F> {
    type Target = VerifierStage3aSumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(KernelSlots)]
pub struct Stage3aKernels<F: JoltField> {
    pub router_short: Box<dyn PrepareKernel<F, RouterShort<F>, Rv64iPlane>>,
}
impl Default for Stage3aKernels<F128> {
    fn default() -> Self {
        Self {
            router_short: Box::<RouterShortPrepare>::default(),
        }
    }
}

jolt_rv64i_verifier::stage3a_sumchecks_members!(impl_stage_prover plane=Rv64iPlane,);

/// Proves this batch from checked dimensions and verified upstream cells in low-variable-first order.
/// Concrete input conversion is shared with verification; invalid points or equations return a typed error.
pub fn prove<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    witness: &Rv64iWitness,
    kernels: &Stage3aKernels<F128>,
    session: &mut ProofSession,
    transcript: &mut T,
    stage2: &Stage2Output,
) -> Result<(BatchProof<RouterFoldValues>, Output), Rv64iProverError> {
    let Inputs {
        batch,
        claims: inputs,
        points,
    } = stage3a::verify::from_upstream(checked, stage2)?;
    let batch = Stage3aSumchecks(batch);
    let challenges = batch
        .draw_challenges(transcript)
        .map_err(Rv64iVerifierError::from)?;
    let proved = batch.prove(
        kernels,
        session,
        &mut SequentialRounds,
        witness,
        &inputs,
        &points,
        &challenges,
        ClearSumcheckRecorder::<F128, NoCommitment>::new(),
        transcript,
    )?;
    let values = stage3a::verify::values(&proved.output_claims);
    let wire = BatchProof {
        rounds: proved.recorded.proof,
        values,
    };
    let output = Output::new(proved.output_claims, proved.output_points)?;
    Ok((wire, output))
}
