//! The local stage-5 driver wraps the verifier-owned batch and member order.

use crate::commitment::BitsCommitmentProver;
use crate::error::Rv64iProverError;
use crate::plane::{Rv64iPlane, Rv64iWitness};
use crate::reference::val_evaluation::{RamValEvaluationPrepare, RegistersValEvaluationPrepare};
use jolt_crypto::NoCommitment;
use jolt_field::{JoltField, F128};
use jolt_kernels::ProofSession;
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::driver::StageProver;
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::error::Rv64iVerifierError;
use jolt_rv64i_verifier::proof::{BatchProof, ValEvaluationValues};
use jolt_rv64i_verifier::stages::stage3a::Output as Stage3aOutput;
use jolt_rv64i_verifier::stages::stage4::verify::Output as Stage4Output;
use jolt_rv64i_verifier::stages::stage5::val_evaluation::{
    RamValEvaluation, RegistersValEvaluation,
};
use jolt_rv64i_verifier::stages::stage5::verify::{self, Output};
use jolt_rv64i_verifier::stages::stage5::{
    Stage5Challenges, Stage5InputClaims, Stage5InputPoints, Stage5OutputClaims, Stage5OutputPoints,
    Stage5Sumchecks as VerifierStage5Sumchecks,
};
use jolt_rv64i_verifier::statement::CheckedInputs;
use jolt_sumcheck::{ClearSumcheckRecorder, SequentialRounds};
use jolt_transcript::Transcript;
use std::ops::Deref;

pub struct Stage5Sumchecks<F: JoltField>(pub VerifierStage5Sumchecks<F>);
impl<F: JoltField> Deref for Stage5Sumchecks<F> {
    type Target = VerifierStage5Sumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[derive(KernelSlots)]
pub struct Stage5Kernels<F: JoltField> {
    pub registers_val_evaluation: Box<dyn PrepareKernel<F, RegistersValEvaluation<F>, Rv64iPlane>>,
    pub ram_val_evaluation: Box<dyn PrepareKernel<F, RamValEvaluation<F>, Rv64iPlane>>,
}
impl Default for Stage5Kernels<F128> {
    fn default() -> Self {
        Self {
            registers_val_evaluation: Box::<RegistersValEvaluationPrepare>::default(),
            ram_val_evaluation: Box::<RamValEvaluationPrepare>::default(),
        }
    }
}
jolt_rv64i_verifier::stage5_sumchecks_members!(impl_stage_prover plane=Rv64iPlane,);

/// Proves update reductions from verified batches 3a and 4 at low-variable-first points.
/// Uses the verifier's conversion and returns the absorbed wire values with their downstream output.
pub fn prove<S: BitsCommitmentProver, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    witness: &Rv64iWitness,
    kernels: &Stage5Kernels<F128>,
    session: &mut ProofSession,
    transcript: &mut T,
    stage3a: &Stage3aOutput,
    stage4: &Stage4Output,
) -> Result<(BatchProof<ValEvaluationValues>, Output), Rv64iProverError> {
    let inputs =
        verify::from_upstream(checked, stage3a, stage4).map_err(Rv64iVerifierError::from)?;
    let batch = Stage5Sumchecks(inputs.batch);
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
        claims: values.expand(),
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
