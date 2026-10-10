//! Local prover driver and reference registry for batch 1.
use crate::error::Rv64iProverError;
use crate::plane::{Rv64iPlane, Rv64iWitness};
use crate::reference::spartan::{SpartanOuterF128Prepare, SpartanOuterF2Prepare};
use jolt_crypto::NoCommitment;
use jolt_field::{JoltField, F128};
use jolt_kernels::ProofSession;
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::driver::StageProver;
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::commitment::BitsCommitmentScheme;
use jolt_rv64i_verifier::error::Rv64iVerifierError;
use jolt_rv64i_verifier::proof::{BatchProof, OuterValues};
use jolt_rv64i_verifier::stages::stage1::{self, Output};
use jolt_rv64i_verifier::stages::stage1::{
    SpartanOuterF128, SpartanOuterF2, Stage1Challenges, Stage1InputClaims, Stage1InputPoints,
    Stage1OutputClaims, Stage1OutputPoints, Stage1Sumchecks as VerifierStage1Sumchecks,
};
use jolt_rv64i_verifier::statement::CheckedInputs;
use jolt_sumcheck::{ClearSumcheckRecorder, SequentialRounds};
use jolt_transcript::Transcript;
use std::ops::Deref;

/// Local wrapper sharing the verifier batch and its low-variable-first geometry.
pub struct Stage1Sumchecks<F: JoltField>(pub VerifierStage1Sumchecks<F>);
impl<F: JoltField> Deref for Stage1Sumchecks<F> {
    type Target = VerifierStage1Sumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[derive(KernelSlots)]
pub struct Stage1Kernels<F: JoltField> {
    pub spartan_outer_f2: Box<dyn PrepareKernel<F, SpartanOuterF2<F>, Rv64iPlane>>,
    pub spartan_outer_f128: Box<dyn PrepareKernel<F, SpartanOuterF128<F>, Rv64iPlane>>,
}
impl Default for Stage1Kernels<F128> {
    fn default() -> Self {
        Self {
            spartan_outer_f2: Box::<SpartanOuterF2Prepare>::default(),
            spartan_outer_f128: Box::<SpartanOuterF128Prepare>::default(),
        }
    }
}
jolt_rv64i_verifier::stage1_sumchecks_members!(impl_stage_prover plane=Rv64iPlane,);

/// Proves this batch from checked dimensions and verified upstream cells in low-variable-first order.
/// Concrete input conversion is shared with verification; invalid points or equations return a typed error.
pub fn prove<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    witness: &Rv64iWitness,
    kernels: &Stage1Kernels<F128>,
    session: &mut ProofSession,
    transcript: &mut T,
) -> Result<(BatchProof<OuterValues>, Output), Rv64iProverError> {
    let batch = Stage1Sumchecks(stage1::verify::from_checked(checked, transcript)?);
    let inputs = Stage1InputClaims {
        spartan_outer_f2: Default::default(),
        spartan_outer_f128: Default::default(),
    };
    let points = batch.empty_input_points();
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
    let f2 = &proved.output_claims.spartan_outer_f2;
    let f128 = &proved.output_claims.spartan_outer_f128;
    let values = OuterValues {
        az_f2: f2.az,
        bz_f2: f2.bz,
        cz_f2: f2.cz,
        az_f128: f128.az,
        bz_f128: f128.bz,
        cz_f128: f128.cz,
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
