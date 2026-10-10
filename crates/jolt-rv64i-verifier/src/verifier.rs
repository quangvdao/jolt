//! Ordered verification of all eight reduction batches and the bit-table opening.
use crate::{
    commitment::{BitsCommitmentScheme, BitsGeometry},
    error::Rv64iVerifierError,
    preprocessing::VerifierPreprocessing,
    proof::Rv64iProof,
    stages::{stage1, stage2, stage3a, stage3b, stage4, stage5, stage6a, stage6b},
    statement::{CheckedInputs, Statement},
    transcript::preamble,
};
use jolt_field::F128;
use jolt_transcript::Transcript;

/// Checks the statement and proof shape before starting the protocol transcript.
/// Returns its final state on acceptance, or the first statement, batch or commitment error.
pub fn verify<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    preprocessing: &VerifierPreprocessing<S>,
    statement: &Statement,
    proof: &Rv64iProof<S>,
) -> Result<T, Rv64iVerifierError> {
    let checked = CheckedInputs::new(preprocessing, statement, proof)?;
    let mut transcript = preamble::<S, T>(&checked);
    let state = S::verify_commit(
        preprocessing.scheme(),
        BitsGeometry {
            log_T: checked.log_T(),
        },
        &proof.bits_commitment,
        &mut transcript,
    )
    .map_err(|error| Rv64iVerifierError::CommitPhase(Box::new(error)))?;
    let s1 = stage1::verify::verify(&checked, &proof.stage1, &mut transcript)?;
    let s2 = stage2::verify::verify(&checked, &proof.stage2, &mut transcript, &s1)?;
    let s3a = stage3a::verify::verify(&checked, &proof.stage3a, &mut transcript, &s2)?;
    let s3b = stage3b::verify::verify(&checked, &proof.stage3b, &mut transcript, &s1, &s3a)?;
    let s4 = stage4::verify::verify(&checked, &proof.stage4, &mut transcript, &s3a, &s3b)?;
    let s5 = stage5::verify::verify(&checked, &proof.stage5, &mut transcript, &s3a, &s4)?;
    let s6a = stage6a::verify::verify(
        &checked,
        &proof.stage6a,
        &mut transcript,
        &s3a,
        &s3b,
        &s4,
        &s5,
    )?;
    let _s6b = stage6b::verify::verify(
        &checked,
        &proof.stage6b,
        &mut transcript,
        &s1,
        &s2,
        &s3a,
        &s3b,
        &s4,
        &s5,
        &s6a,
        state,
        &proof.opening,
    )?;
    Ok(transcript)
}
