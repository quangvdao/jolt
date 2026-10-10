//! Local stage6a prover driver and reference-kernel registry.
use crate::error::Rv64iProverError;
use crate::plane::{Rv64iPlane, Rv64iWitness};
use crate::reference::bytecode::BytecodeReadAddressPrepare;
use jolt_crypto::NoCommitment;
use jolt_field::{JoltField, F128};
use jolt_kernels::{KernelSlots, PrepareKernel, ProofSession};
use jolt_prover::{driver::StageProver, impl_stage_prover};
use jolt_rv64i_verifier::error::Rv64iVerifierError;
use jolt_rv64i_verifier::stages::stage6a::{
    BytecodeReadAddress, Stage6aChallenges, Stage6aInputClaims, Stage6aInputPoints,
    Stage6aOutputClaims, Stage6aOutputPoints, Stage6aSumchecks as VerifierStage6aSumchecks,
};
use std::ops::Deref;

pub struct Stage6aSumchecks<F: JoltField>(pub VerifierStage6aSumchecks<F>);
impl<F: JoltField> Deref for Stage6aSumchecks<F> {
    type Target = VerifierStage6aSumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[derive(KernelSlots)]
pub struct Stage6aKernels<F: JoltField> {
    pub bytecode_read_address: Box<dyn PrepareKernel<F, BytecodeReadAddress<F>, Rv64iPlane>>,
}
impl<F: JoltField> Default for Stage6aKernels<F> {
    fn default() -> Self {
        Self {
            bytecode_read_address: Box::<BytecodeReadAddressPrepare>::default(),
        }
    }
}
jolt_rv64i_verifier::stage6a_sumchecks_members!(impl_stage_prover plane=Rv64iPlane,);

use jolt_rv64i_verifier::commitment::BitsCommitmentScheme;
use jolt_rv64i_verifier::proof::{BatchProof, BytecodeAddressValue};
use jolt_rv64i_verifier::stages::stage3a::Output as Stage3aOutput;
use jolt_rv64i_verifier::stages::stage3b::Output as Stage3bOutput;
use jolt_rv64i_verifier::stages::stage4::Output as Stage4Output;
use jolt_rv64i_verifier::stages::stage5::Output as Stage5Output;
use jolt_rv64i_verifier::stages::stage6a::verify::{self, Inputs, Output};
use jolt_rv64i_verifier::statement::CheckedInputs;
use jolt_sumcheck::{ClearSumcheckRecorder, SequentialRounds};
use jolt_transcript::Transcript;

/// Proves batch 6a from verified earlier outputs at low-variable-first points.
/// Checked entry and final PCs enter the same member construction used by the verifier.
#[expect(
    clippy::too_many_arguments,
    reason = "the stage driver receives its context and four upstream outputs"
)]
pub fn prove<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    witness: &Rv64iWitness,
    kernels: &Stage6aKernels<F128>,
    session: &mut ProofSession,
    transcript: &mut T,
    stage3a: &Stage3aOutput,
    stage3b: &Stage3bOutput,
    stage4: &Stage4Output,
    stage5: &Stage5Output,
) -> Result<(BatchProof<BytecodeAddressValue>, Output), Rv64iProverError> {
    let Inputs {
        batch,
        claims,
        points,
    } = verify::from_upstream(checked, stage3a, stage3b, stage4, stage5)?;
    let batch = Stage6aSumchecks(batch);
    let challenges = batch
        .draw_challenges(transcript)
        .map_err(Rv64iVerifierError::from)?;
    let proved = batch.prove(
        kernels,
        session,
        &mut SequentialRounds,
        witness,
        &claims,
        &points,
        &challenges,
        ClearSumcheckRecorder::<F128, NoCommitment>::new(),
        transcript,
    )?;
    let values = BytecodeAddressValue {
        address_claim: proved.output_claims.bytecode_read_address.address_claim,
    };
    let output = verify::finish(
        checked,
        &batch,
        proved.output_claims,
        proved.output_points,
        &challenges,
    )?;
    Ok((
        BatchProof {
            rounds: proved.recorded.proof,
            values,
        },
        output,
    ))
}
