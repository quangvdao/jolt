//! Local prover driver and reference registry for batch 4.

use crate::commitment::BitsCommitmentProver;
use crate::error::Rv64iProverError;
use crate::plane::{Rv64iPlane, Rv64iWitness};
use crate::reference::read_checking::{
    RamOutputCheckPrepare, RamReadCheckingPrepare, RegistersReadCheckingPrepare,
};
use jolt_crypto::NoCommitment;
use jolt_field::{JoltField, F128};
use jolt_kernels::ProofSession;
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::driver::StageProver;
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::error::Rv64iVerifierError;
use jolt_rv64i_verifier::proof::{BatchProof, ReadCheckingValues};
use jolt_rv64i_verifier::stages::stage3a::Output as Stage3aOutput;
use jolt_rv64i_verifier::stages::stage3b::Output as Stage3bOutput;
use jolt_rv64i_verifier::stages::stage4::verify::{self, Output};
use jolt_rv64i_verifier::stages::stage4::{
    ram_output_check::RamOutputCheck,
    ram_read_checking::RamReadChecking,
    registers_read_checking::RegistersReadChecking,
    verify::{
        Stage4Challenges, Stage4InputClaims, Stage4InputPoints, Stage4OutputClaims,
        Stage4OutputPoints, Stage4Sumchecks as VerifierStage4Sumchecks,
    },
};
use jolt_rv64i_verifier::statement::CheckedInputs;
use jolt_sumcheck::{ClearSumcheckRecorder, SequentialRounds};
use jolt_transcript::Transcript;
use std::ops::Deref;

pub struct Stage4Sumchecks<F: JoltField>(pub VerifierStage4Sumchecks<F>);
impl<F: JoltField> Deref for Stage4Sumchecks<F> {
    type Target = VerifierStage4Sumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[derive(KernelSlots)]
pub struct Stage4Kernels<F: JoltField> {
    pub registers_read_checking: Box<dyn PrepareKernel<F, RegistersReadChecking<F>, Rv64iPlane>>,
    pub ram_read_checking: Box<dyn PrepareKernel<F, RamReadChecking<F>, Rv64iPlane>>,
    pub ram_output_check: Box<dyn PrepareKernel<F, RamOutputCheck<F>, Rv64iPlane>>,
}
impl Default for Stage4Kernels<F128> {
    fn default() -> Self {
        Self {
            registers_read_checking: Box::<RegistersReadCheckingPrepare>::default(),
            ram_read_checking: Box::<RamReadCheckingPrepare>::default(),
            ram_output_check: Box::<RamOutputCheckPrepare>::default(),
        }
    }
}
jolt_rv64i_verifier::stage4_sumchecks_members!(impl_stage_prover plane = Rv64iPlane,);

/// Proves read and public output checks from batches 3a and 3b at their low-variable-first points.
/// Uses the verifier's conversion and shares the checked public I/O allocation.
pub fn prove<S: BitsCommitmentProver, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    witness: &Rv64iWitness,
    kernels: &Stage4Kernels<F128>,
    session: &mut ProofSession,
    transcript: &mut T,
    stage3a: &Stage3aOutput,
    stage3b: &Stage3bOutput,
) -> Result<(BatchProof<ReadCheckingValues>, Output), Rv64iProverError> {
    let inputs = verify::from_upstream(checked, transcript, stage3a, stage3b)
        .map_err(Rv64iVerifierError::from)?;
    let batch = Stage4Sumchecks(inputs.batch);
    let challenges = batch
        .draw_challenges(transcript)
        .map_err(Rv64iVerifierError::from)?;
    let proved = batch.prove(
        kernels,
        session,
        &mut SequentialRounds,
        witness,
        &inputs.claims,
        &inputs.points,
        &challenges,
        ClearSumcheckRecorder::<F128, NoCommitment>::new(),
        transcript,
    )?;
    let values = proved.output_claims.into_wire();
    let output = Output {
        claims: Stage4OutputClaims::from_wire(&values),
        points: proved.output_points,
    };
    Ok((
        BatchProof {
            rounds: proved.recorded.proof,
            values,
        },
        output,
    ))
}
