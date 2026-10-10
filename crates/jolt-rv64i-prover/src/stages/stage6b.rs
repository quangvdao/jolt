//! Local stage6b prover driver and reference-kernel registry.
use crate::error::Rv64iProverError;
use crate::plane::{Rv64iPlane, Rv64iWitness};
use crate::reference::bits_reduction::BitsReductionPrepare;
use crate::reference::bytecode::BytecodeReadCyclePrepare;
use crate::reference::ra_product::RamRaProductPrepare;
use jolt_crypto::NoCommitment;
use jolt_field::{JoltField, F128};
use jolt_kernels::{KernelSlots, PrepareKernel, ProofSession};
use jolt_prover::{driver::StageProver, impl_stage_prover};
use jolt_rv64i_verifier::error::Rv64iVerifierError;
use jolt_rv64i_verifier::stages::stage6b::{
    BitsReduction, BytecodeReadCycle, RamRaProduct, Stage6bChallenges, Stage6bInputClaims,
    Stage6bInputPoints, Stage6bOutputClaims, Stage6bOutputPoints,
    Stage6bSumchecks as VerifierStage6bSumchecks,
};
use std::ops::Deref;

pub struct Stage6bSumchecks<F: JoltField>(pub VerifierStage6bSumchecks<F>);
impl<F: JoltField> Deref for Stage6bSumchecks<F> {
    type Target = VerifierStage6bSumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[derive(KernelSlots)]
pub struct Stage6bKernels<F: JoltField> {
    pub bytecode_read_cycle: Box<dyn PrepareKernel<F, BytecodeReadCycle<F>, Rv64iPlane>>,
    pub ram_ra_product: Box<dyn PrepareKernel<F, RamRaProduct<F>, Rv64iPlane>>,
    pub bits_reduction: Box<dyn PrepareKernel<F, BitsReduction<F>, Rv64iPlane>>,
}
impl<F: JoltField> Default for Stage6bKernels<F> {
    fn default() -> Self {
        Self {
            bytecode_read_cycle: Box::<BytecodeReadCyclePrepare>::default(),
            ram_ra_product: Box::<RamRaProductPrepare>::default(),
            bits_reduction: Box::<BitsReductionPrepare>::default(),
        }
    }
}
jolt_rv64i_verifier::stage6b_sumchecks_members!(impl_stage_prover plane=Rv64iPlane, curate=|_batch,claims,_points| { Ok(claims.bits_reduction.columns.clone()) },);

use jolt_rv64i_verifier::commitment::BitsCommitmentScheme;
use jolt_rv64i_verifier::proof::{BatchProof, BitsColumns};
use jolt_rv64i_verifier::stages::stage1::Output as Stage1Output;
use jolt_rv64i_verifier::stages::stage2::Output as Stage2Output;
use jolt_rv64i_verifier::stages::stage3a::Output as Stage3aOutput;
use jolt_rv64i_verifier::stages::stage3b::Output as Stage3bOutput;
use jolt_rv64i_verifier::stages::stage4::Output as Stage4Output;
use jolt_rv64i_verifier::stages::stage5::Output as Stage5Output;
use jolt_rv64i_verifier::stages::stage6a::Output as Stage6aOutput;
use jolt_rv64i_verifier::stages::stage6b::verify::{self, Inputs, Output};
use jolt_rv64i_verifier::statement::CheckedInputs;
use jolt_sumcheck::{ClearSumcheckRecorder, SequentialRounds};
use jolt_transcript::Transcript;
use jolt_verifier::VerifierError;

/// Proves the terminal batch from all seven earlier outputs and absorbs its 256 columns.
/// The caller draws `rho` only after return and opens at `(rho, output.point)`, low variables first.
#[expect(
    clippy::too_many_arguments,
    reason = "the terminal driver receives its context and seven upstream outputs"
)]
pub fn prove<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    witness: &Rv64iWitness,
    kernels: &Stage6bKernels<F128>,
    session: &mut ProofSession,
    transcript: &mut T,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    stage3a: &Stage3aOutput,
    stage3b: &Stage3bOutput,
    stage4: &Stage4Output,
    stage5: &Stage5Output,
    stage6a: &Stage6aOutput,
) -> Result<(BatchProof<BitsColumns>, Output), Rv64iProverError> {
    let Inputs {
        batch,
        claims,
        points,
    } = verify::from_upstream(
        checked, stage1, stage2, stage3a, stage3b, stage4, stage5, stage6a,
    )?;
    let batch = Stage6bSumchecks(batch);
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
    let point = proved
        .output_points
        .bits_reduction
        .columns
        .into_iter()
        .next()
        .ok_or(VerifierError::StageClaimOutputMismatch { stage: 6 })
        .map_err(Rv64iVerifierError::from)?;
    let columns = proved.output_claims.bits_reduction.columns;
    Ok((
        BatchProof {
            rounds: proved.recorded.proof,
            values: BitsColumns(columns),
        },
        Output { point },
    ))
}
