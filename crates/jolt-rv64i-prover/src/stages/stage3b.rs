//! Prover driver for the five router cycle reductions.

use crate::error::Rv64iProverError;
use crate::plane::{Rv64iPlane, Rv64iWitness};
use crate::reference::routers::{
    RouterCycleBranchPrepare, RouterCycleComparePrepare, RouterCycleMemoryPrepare,
    RouterCycleShiftPrepare, RouterCycleVariantPrepare,
};
use jolt_crypto::NoCommitment;
use jolt_field::{JoltField, F128};
use jolt_kernels::ProofSession;
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::driver::StageProver;
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::commitment::BitsCommitmentScheme;
use jolt_rv64i_verifier::error::Rv64iVerifierError;
use jolt_rv64i_verifier::proof::{BatchProof, RouterCycleValues};
use jolt_rv64i_verifier::stages::stage1::Output as Stage1Output;
use jolt_rv64i_verifier::stages::stage3a::Output as Stage3aOutput;
use jolt_rv64i_verifier::stages::stage3b;
use jolt_rv64i_verifier::stages::stage3b::{verify::Inputs, Output};
use jolt_rv64i_verifier::stages::stage3b::{
    RouterCycleBranch, RouterCycleCompare, RouterCycleMemory, RouterCycleShift, RouterCycleVariant,
    Stage3bChallenges, Stage3bInputClaims, Stage3bInputPoints, Stage3bOutputClaims,
    Stage3bOutputPoints, Stage3bSumchecks as VerifierStage3bSumchecks,
};
use jolt_rv64i_verifier::statement::CheckedInputs;
use jolt_sumcheck::{ClearSumcheckRecorder, SequentialRounds};
use jolt_transcript::Transcript;
use std::ops::Deref;

/// Local wrapper sharing the verifier batch and its low-variable-first geometry.
pub struct Stage3bSumchecks<F: JoltField>(pub VerifierStage3bSumchecks<F>);
impl<F: JoltField> Deref for Stage3bSumchecks<F> {
    type Target = VerifierStage3bSumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(KernelSlots)]
pub struct Stage3bKernels<F: JoltField> {
    pub variant: Box<dyn PrepareKernel<F, RouterCycleVariant<F>, Rv64iPlane>>,
    pub shift: Box<dyn PrepareKernel<F, RouterCycleShift<F>, Rv64iPlane>>,
    pub memory: Box<dyn PrepareKernel<F, RouterCycleMemory<F>, Rv64iPlane>>,
    pub compare: Box<dyn PrepareKernel<F, RouterCycleCompare<F>, Rv64iPlane>>,
    pub branch: Box<dyn PrepareKernel<F, RouterCycleBranch<F>, Rv64iPlane>>,
}
impl Default for Stage3bKernels<F128> {
    fn default() -> Self {
        Self {
            variant: Box::<RouterCycleVariantPrepare>::default(),
            shift: Box::<RouterCycleShiftPrepare>::default(),
            memory: Box::<RouterCycleMemoryPrepare>::default(),
            compare: Box::<RouterCycleComparePrepare>::default(),
            branch: Box::<RouterCycleBranchPrepare>::default(),
        }
    }
}

jolt_rv64i_verifier::stage3b_sumchecks_members!(impl_stage_prover plane=Rv64iPlane,);

/// Proves this batch from checked dimensions and verified upstream cells in low-variable-first order.
/// Concrete input conversion is shared with verification; invalid points or equations return a typed error.
pub fn prove<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    witness: &Rv64iWitness,
    kernels: &Stage3bKernels<F128>,
    session: &mut ProofSession,
    transcript: &mut T,
    stage1: &Stage1Output,
    stage3a: &Stage3aOutput,
) -> Result<(BatchProof<RouterCycleValues>, Output), Rv64iProverError> {
    let Inputs {
        batch,
        claims: inputs,
        points,
    } = stage3b::verify::from_upstream(checked, stage1, stage3a)?;
    let batch = Stage3bSumchecks(batch);
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
    let values = stage3b::verify::values(&proved.output_claims);
    let wire = BatchProof {
        rounds: proved.recorded.proof,
        values,
    };
    let output = Output {
        claims: proved.output_claims,
        points: proved.output_points,
    };
    Ok((wire, output))
}
