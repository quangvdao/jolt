//! Batch-local wiring binds the three stage-4 values to canonical initial RAM.

use super::{Stage5OutputClaims, Stage5OutputPoints, Stage5Source, Stage5Sumchecks};
use crate::commitment::BitsCommitmentScheme;
use crate::error::Rv64iVerifierError;
use crate::proof::{BatchProof, ValEvaluationValues};
use crate::statement::CheckedInputs;
use jolt_field::F128;
use jolt_transcript::Transcript;

pub struct Output {
    pub r_5: Vec<F128>,
    pub output_values: Stage5OutputClaims<F128>,
    pub output_points: Stage5OutputPoints<F128>,
}
pub fn verify<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    proof: &BatchProof<ValEvaluationValues>,
    transcript: &mut T,
    source: &Stage5Source,
) -> Result<Output, Rv64iVerifierError> {
    let batch = Stage5Sumchecks::new(checked, source)?;
    let inputs = &source.values;
    let input_points = &source.points;
    let challenges = batch.draw_challenges(transcript)?;
    let output_values = proof.values.expand();
    let output_points = batch.verify_clear(
        inputs,
        input_points,
        &challenges,
        &output_values,
        &proof.rounds,
        transcript,
        5,
    )?;
    batch.append_output_claims(transcript, &output_values);
    Ok(Output {
        r_5: output_points.registers_val_evaluation.store.clone(),
        output_values,
        output_points,
    })
}
