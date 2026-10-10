//! Local prover driver and reference registry for batch 2.
use crate::error::Rv64iProverError;
use crate::plane::{Rv64iPlane, Rv64iWitness};
use crate::reference::spartan::SpartanInnerPrepare;
use jolt_crypto::NoCommitment;
use jolt_field::{JoltField, F128};
use jolt_kernels::ProofSession;
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::driver::StageProver;
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::commitment::BitsCommitmentScheme;
use jolt_rv64i_verifier::error::Rv64iVerifierError;
use jolt_rv64i_verifier::proof::{BatchProof, InnerValues};
use jolt_rv64i_verifier::public::matrices::RowMatrices;
use jolt_rv64i_verifier::stages::stage1::Output as Stage1Output;
use jolt_rv64i_verifier::stages::stage2;
use jolt_rv64i_verifier::stages::stage2::{verify::Inputs, Output};
use jolt_rv64i_verifier::stages::stage2::{
    SpartanInner, Stage2Challenges, Stage2InputClaims, Stage2InputPoints, Stage2OutputClaims,
    Stage2OutputPoints, Stage2Sumchecks as VerifierStage2Sumchecks,
};
use jolt_rv64i_verifier::statement::CheckedInputs;
use jolt_sumcheck::{ClearSumcheckRecorder, SequentialRounds};
use jolt_transcript::Transcript;
use std::ops::Deref;
use std::sync::Arc;

/// Local wrapper sharing the verifier batch and its low-variable-first geometry.
pub struct Stage2Sumchecks<F: JoltField>(pub VerifierStage2Sumchecks<F>);
impl<F: JoltField> Deref for Stage2Sumchecks<F> {
    type Target = VerifierStage2Sumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[derive(KernelSlots)]
pub struct Stage2Kernels<F: JoltField> {
    pub spartan_inner: Box<dyn PrepareKernel<F, SpartanInner<F>, Rv64iPlane>>,
}
impl Default for Stage2Kernels<F128> {
    fn default() -> Self {
        Self {
            spartan_inner: Box::<SpartanInnerPrepare>::default(),
        }
    }
}
jolt_rv64i_verifier::stage2_sumchecks_members!(impl_stage_prover plane=Rv64iPlane,);

/// Proves this batch from checked dimensions and verified upstream cells in low-variable-first order.
/// Concrete input conversion is shared with verification; invalid points or equations return a typed error.
pub fn prove<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    witness: &Rv64iWitness,
    kernels: &Stage2Kernels<F128>,
    session: &mut ProofSession,
    transcript: &mut T,
    stage1: &Stage1Output,
) -> Result<(BatchProof<InnerValues>, Output), Rv64iProverError> {
    let Inputs {
        batch,
        claims: inputs,
        points,
    } = stage2::verify::from_upstream(Arc::new(RowMatrices::new(checked.layout())), stage1)?;
    let batch = Stage2Sumchecks(batch);
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
    let claims = &proved.output_claims.spartan_inner;
    let values = InnerValues {
        witness_routed: claims.witness_routed,
        direct_columns: claims.direct_columns,
    };
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
