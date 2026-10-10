//! Verifies the address batch and computes the five public folds for batch 6b.

use jolt_field::F128;
use jolt_transcript::Transcript;
use jolt_verifier::VerifierError;

use super::{
    BytecodeReadAddress, BytecodeReadAddressOutputClaims, BytecodeReadPoints, Stage6aChallenges,
    Stage6aInputClaims, Stage6aInputPoints, Stage6aOutputClaims, Stage6aOutputPoints,
    Stage6aSumchecks,
};
use crate::commitment::BitsCommitmentScheme;
use crate::error::Rv64iVerifierError;
use crate::points::PointsError;
use crate::proof::{BatchProof, BytecodeAddressValue};
use crate::public::bytecode::BytecodeWeights;
use crate::statement::CheckedInputs;

pub struct Stage6aOutput {
    pub output_values: Stage6aOutputClaims<F128>,
    pub output_points: Stage6aOutputPoints<F128>,
    pub challenges: Stage6aChallenges<F128>,
    pub bytecode_folds: [F128; 5],
}

/// The fifteen inputs and their earlier points are carried by the caller. The
/// entry PC, final PC and public bytecode come only from checked inputs.
pub fn verify<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    points: BytecodeReadPoints<F128>,
    inputs: &Stage6aInputClaims<F128>,
    proof: &BatchProof<BytecodeAddressValue>,
    transcript: &mut T,
) -> Result<Stage6aOutput, Rv64iVerifierError> {
    let term_error = |error: PointsError| VerifierError::StageClaimSumcheckFailed {
        stage: "BytecodeReadAddress".to_owned(),
        reason: error.to_string(),
    };
    if points.r_3.len() != checked.log_T() {
        return Err(term_error(PointsError::Dimension {
            expected: checked.log_T(),
            actual: points.r_3.len(),
        })
        .into());
    }
    let batch = Stage6aSumchecks {
        bytecode_read_address: BytecodeReadAddress::new(
            checked.log_K_bytecode(),
            points,
            checked.statement().entry_pc,
            checked.final_pc(),
        )
        .map_err(term_error)?,
    };
    let input_points = Stage6aInputPoints {
        bytecode_read_address: batch.bytecode_read_address.input_points(),
    };
    let challenges = batch.draw_challenges(transcript)?;
    let output_values = Stage6aOutputClaims {
        bytecode_read_address: BytecodeReadAddressOutputClaims {
            address_claim: proof.values.address_claim,
        },
    };
    let output_points = batch.verify_clear(
        inputs,
        &input_points,
        &challenges,
        &output_values,
        &proof.rounds,
        transcript,
        6,
    )?;
    batch.append_output_claims(transcript, &output_values);
    let bytecode_folds = BytecodeWeights::new(
        batch.bytecode_read_address.public_points(),
        &challenges.bytecode_read_address,
    )
    .and_then(|weights| {
        weights.evaluate(
            checked.preprocessing().bytecode(),
            &output_points.bytecode_read_address.address_claim,
        )
    })
    .map_err(term_error)?;
    Ok(Stage6aOutput {
        output_values,
        output_points,
        challenges,
        bytecode_folds,
    })
}
