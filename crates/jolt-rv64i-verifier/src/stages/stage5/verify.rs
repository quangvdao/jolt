//! Batch 5 binds the stage-4 values to canonical initial RAM.

use super::{
    RamValEvaluationInputClaims, RegistersValEvaluationInputClaims, Stage5InputClaims,
    Stage5InputPoints, Stage5OutputClaims, Stage5OutputPoints, Stage5Sumchecks,
};
use crate::commitment::BitsCommitmentScheme;
use crate::error::Rv64iVerifierError;
use crate::points::PointsError;
use crate::proof::{BatchProof, ValEvaluationValues};
use crate::stages::stage3a::verify::Output as Stage3aOutput;
use crate::stages::stage4::verify::Output as Stage4Output;
use crate::statement::CheckedInputs;
use jolt_field::F128;
use jolt_transcript::Transcript;
use jolt_verifier::VerifierError;

/// Update cells with low-variable-first points, consumed by batches 6a and 6b.
pub struct Output {
    /// Update values read by batches 6a and 6b.
    pub claims: Stage5OutputClaims<F128>,
    /// Cycle points, prefixed by register/address or bit coordinates where appropriate, read by batches 6a and 6b.
    pub points: Stage5OutputPoints<F128>,
}

impl Output {
    /// The low-variable-first cycle point consumed by batches 6a and 6b.
    pub fn r_5(&self) -> &[F128] {
        &self.points.registers_val_evaluation.store
    }
}

/// The batch and consumed cells established by `from_upstream` from verified stage-4 points.
pub struct Inputs {
    pub batch: Stage5Sumchecks<F128>,
    pub claims: Stage5InputClaims<F128>,
    pub points: Stage5InputPoints<F128>,
}

/// Uses batches 3a and 4 at their verified low-variable-first points.
/// Rejects inconsistent shared bit/address/cycle geometry before deriving the initial RAM term.
pub fn from_upstream<S: BitsCommitmentScheme>(
    checked: &CheckedInputs<'_, S>,
    stage3a: &Stage3aOutput,
    stage4: &Stage4Output,
) -> Result<Inputs, VerifierError> {
    let source = &stage4.points;
    let short_bit = stage3a.x.get(..6).ok_or_else(|| {
        term_error(PointsError::Dimension {
            expected: 6,
            actual: stage3a.x.len(),
        })
    })?;
    let final_bit = source
        .ram_output_check
        .ram_val_final
        .get(checked.log_K_ram()..);
    if final_bit != Some(short_bit) {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: "Stage5".to_owned(),
            reason: "value points differ from the verified router bit point".to_owned(),
        });
    }
    let batch = Stage5Sumchecks::from_points(
        checked,
        &source.registers_read_checking.registers_val,
        &source.ram_read_checking.ram_val,
        &source.ram_output_check.ram_val_final,
    )?;
    let points = batch.input_points();
    let claims = Stage5InputClaims {
        registers_val_evaluation: RegistersValEvaluationInputClaims {
            registers_val: stage4.claims.registers_read_checking.registers_val,
        },
        ram_val_evaluation: RamValEvaluationInputClaims {
            ram_val: stage4.claims.ram_read_checking.ram_val,
            ram_val_final: stage4.claims.ram_output_check.ram_val_final,
        },
    };
    Ok(Inputs {
        batch,
        claims,
        points,
    })
}

/// Verifies update reductions from batches 3a and 4, preserving their low-variable-first points.
pub fn verify<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    proof: &BatchProof<ValEvaluationValues>,
    transcript: &mut T,
    stage3a: &Stage3aOutput,
    stage4: &Stage4Output,
) -> Result<Output, Rv64iVerifierError> {
    let inputs = from_upstream(checked, stage3a, stage4)?;
    verify_inputs(
        &inputs.batch,
        proof,
        transcript,
        &inputs.claims,
        &inputs.points,
    )
}

#[cfg(any(test, feature = "test-utils"))]
pub use converted::verify_inputs;
#[cfg(not(any(test, feature = "test-utils")))]
pub(crate) use converted::verify_inputs;

mod converted {
    use super::*;

    /// Verifies the same batch from already converted low-variable-first inputs.
    /// Batch-local callers establish the point bounds with `Stage5Sumchecks::new`.
    pub fn verify_inputs<T: Transcript<Challenge = F128>>(
        batch: &Stage5Sumchecks<F128>,
        proof: &BatchProof<ValEvaluationValues>,
        transcript: &mut T,
        inputs: &Stage5InputClaims<F128>,
        input_points: &Stage5InputPoints<F128>,
    ) -> Result<Output, Rv64iVerifierError> {
        let challenges = batch.draw_challenges(transcript)?;
        let claims = proof.values.expand();
        let points = batch.verify_clear(
            inputs,
            input_points,
            &challenges,
            &claims,
            &proof.rounds,
            transcript,
            5,
        )?;
        batch.append_output_claims(transcript, &claims);
        Ok(Output { claims, points })
    }
}

fn term_error(error: PointsError) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: "Stage5".to_owned(),
        reason: error.to_string(),
    }
}
