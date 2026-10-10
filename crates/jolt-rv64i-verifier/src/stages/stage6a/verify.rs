//! Verifies the address batch and computes the five public folds for batch 6b.

use jolt_field::F128;
use jolt_transcript::Transcript;
use jolt_verifier::VerifierError;

use super::{
    BytecodeReadAddress, BytecodeReadAddressInputClaims, BytecodeReadAddressOutputClaims,
    BytecodeReadPoints, Stage6aChallenges, Stage6aInputClaims, Stage6aInputPoints,
    Stage6aOutputClaims, Stage6aOutputPoints, Stage6aSumchecks,
};
use crate::commitment::BitsCommitmentScheme;
use crate::error::Rv64iVerifierError;
use crate::points::PointsError;
use crate::proof::{BatchProof, BytecodeAddressValue};
use crate::stages::stage3a::Output as Stage3aOutput;
use crate::stages::stage3b::Output as Stage3bOutput;
use crate::stages::stage4::Output as Stage4Output;
use crate::stages::stage5::Output as Stage5Output;
use crate::statement::CheckedInputs;

/// Verified address cells consumed by batch 6b at a low-variable-first bytecode point.
pub struct Output {
    /// The address claim consumed by batch 6b.
    pub claims: Stage6aOutputClaims<F128>,
    /// The bytecode address point consumed by batch 6b.
    pub points: Stage6aOutputPoints<F128>,
    /// Public bytecode folds in cycle-weight order, consumed by batch 6b.
    pub bytecode_folds: [F128; 5],
}

/// Concrete address member and consumed cells at their earlier low-variable-first points.
pub struct Inputs {
    pub batch: Stage6aSumchecks<F128>,
    pub claims: Stage6aInputClaims<F128>,
    pub points: Stage6aInputPoints<F128>,
}

/// Converts the verified outputs of batches 3a, 3b, 4 and 5 into the fifteen address claims.
/// The member constructor checks bit, kind, register and cycle dimensions.
pub fn from_upstream<S: BitsCommitmentScheme>(
    checked: &CheckedInputs<'_, S>,
    stage3a: &Stage3aOutput,
    stage3b: &Stage3bOutput,
    stage4: &Stage4Output,
    stage5: &Stage5Output,
) -> Result<Inputs, Rv64iVerifierError> {
    let points = BytecodeReadPoints::new(
        &stage3a.x,
        stage4.a_reg().map_err(term_error)?.to_vec(),
        stage3b.r_3().to_vec(),
        stage4.r_4().map_err(term_error)?.to_vec(),
        stage5.r_5().to_vec(),
    )
    .map_err(term_error)?;
    let variant = &stage3b.claims.variant;
    let read = &stage4.claims.registers_read_checking;
    let update = &stage5.claims.registers_val_evaluation;
    let claims = Stage6aInputClaims {
        bytecode_read_address: BytecodeReadAddressInputClaims {
            imm: variant.imm,
            fall_through_pc: variant.fall_through_pc,
            pc_plus_imm: variant.pc_plus_imm,
            pc: variant.pc,
            next_pc: variant.next_pc,
            variant: variant.variant,
            shift_kind: stage3b.claims.shift.shift_kind,
            access_kind: stage3b.claims.memory.access_kind,
            key_kind: stage3b.claims.compare.key_kind,
            branch: stage3b.claims.branch.branch,
            rs1_ra: read.rs1_ra,
            rs2_ra: read.rs2_ra,
            rd_wa_read: read.rd_wa,
            rd_wa_write: update.rd_wa,
            store: update.store,
        },
    };
    from_points(checked, points, claims)
}

/// Constructs the same address member for independently evaluated batch-local inputs.
/// Points use bit, kind or address variables before cycle variables; malformed dimensions fail.
pub fn from_points<S: BitsCommitmentScheme>(
    checked: &CheckedInputs<'_, S>,
    points: BytecodeReadPoints<F128>,
    claims: Stage6aInputClaims<F128>,
) -> Result<Inputs, Rv64iVerifierError> {
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
    Ok(Inputs {
        batch,
        claims,
        points: input_points,
    })
}

/// Verifies batch 6a from its four upstream outputs and checked entry and final PCs.
/// Returns the address claim and five public folds at its low-variable-first address point.
pub fn verify<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    proof: &BatchProof<BytecodeAddressValue>,
    transcript: &mut T,
    stage3a: &Stage3aOutput,
    stage3b: &Stage3bOutput,
    stage4: &Stage4Output,
    stage5: &Stage5Output,
) -> Result<Output, Rv64iVerifierError> {
    verify_inputs(
        checked,
        proof,
        transcript,
        from_upstream(checked, stage3a, stage3b, stage4, stage5)?,
    )
}

#[cfg(any(test, feature = "test-utils"))]
pub use converted::verify_inputs;
#[cfg(not(any(test, feature = "test-utils")))]
pub(crate) use converted::verify_inputs;

mod converted {
    use super::*;

    /// Verifies the constructed member and consumed cells; this is the batch-local entry depth.
    pub fn verify_inputs<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
        checked: &CheckedInputs<'_, S>,
        proof: &BatchProof<BytecodeAddressValue>,
        transcript: &mut T,
        inputs: Inputs,
    ) -> Result<Output, Rv64iVerifierError> {
        let Inputs {
            batch,
            claims: inputs,
            points: input_points,
        } = inputs;
        let challenges = batch.draw_challenges(transcript)?;
        let output_values = Stage6aOutputClaims {
            bytecode_read_address: BytecodeReadAddressOutputClaims {
                address_claim: proof.values.address_claim,
            },
        };
        let output_points = batch.verify_clear(
            &inputs,
            &input_points,
            &challenges,
            &output_values,
            &proof.rounds,
            transcript,
            6,
        )?;
        batch.append_output_claims(transcript, &output_values);
        finish(checked, &batch, output_values, output_points, &challenges)
    }
}

/// Computes the public folds once after an accepted address batch, for both stage drivers.
pub fn finish<S: BitsCommitmentScheme>(
    checked: &CheckedInputs<'_, S>,
    batch: &Stage6aSumchecks<F128>,
    claims: Stage6aOutputClaims<F128>,
    points: Stage6aOutputPoints<F128>,
    challenges: &Stage6aChallenges<F128>,
) -> Result<Output, Rv64iVerifierError> {
    let bytecode_folds = batch
        .bytecode_read_address
        .public_weights(&challenges.bytecode_read_address)
        .and_then(|weights| {
            weights.evaluate(
                checked.preprocessing().bytecode(),
                &points.bytecode_read_address.address_claim,
            )
        })
        .map_err(term_error)?;
    Ok(Output {
        claims,
        points,
        bytecode_folds,
    })
}

fn term_error(error: PointsError) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: "BytecodeReadAddress".to_owned(),
        reason: error.to_string(),
    }
}
