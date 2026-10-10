//! Batch 4 wiring: read checking followed by agreement with public I/O.

use crate::ids::{ChallengeId, DerivedId, FamilyExpr, OpeningId};
use jolt_claims::{OutputClaims, SumcheckChallenges};
use jolt_field::{JoltField, F128};
use jolt_transcript::Transcript;
use jolt_verifier::VerifierError;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::ram_output_check::RamOutputCheckOutputClaims;
use super::ram_read_checking::RamReadCheckingInputClaims;
use super::ram_read_checking::RamReadCheckingOutputClaims;
use super::registers_read_checking::{
    RegistersReadCheckingInputClaims, RegistersReadCheckingOutputClaims,
};
pub use super::{
    Stage4Challenges, Stage4InputClaims, Stage4InputPoints, Stage4OutputClaims, Stage4OutputPoints,
    Stage4Sumchecks,
};
use crate::commitment::BitsCommitmentScheme;
use crate::error::Rv64iVerifierError;
use crate::points::PointsError;
use crate::proof::{BatchProof, ReadCheckingValues};
use crate::stages::stage3a::Output as Stage3aOutput;
use crate::stages::stage3b::Output as Stage3bOutput;
use crate::statement::{CheckedInputs, LOG_T_MAX};

/// Read-checking cells and their low-variable-first points, consumed by batches 5, 6a and 6b.
pub struct Output {
    /// Register selectors and state values read by batches 5 and 6a; RAM selector read by 6b.
    pub claims: Stage4OutputClaims<F128>,
    /// Address-first points, followed by bit and/or cycle coordinates, read by batches 5, 6a and 6b.
    pub points: Stage4OutputPoints<F128>,
}
impl Output {
    /// The low-variable-first RAM address point consumed by batches 5, 6a and 6b.
    pub fn a_ram(&self) -> Result<&[F128], PointsError> {
        let point = &self.points.ram_output_check.ram_val_final;
        let a = point.len().checked_sub(6).ok_or(PointsError::Dimension {
            expected: 6,
            actual: point.len(),
        })?;
        point.get(..a).ok_or(PointsError::Dimension {
            expected: a,
            actual: point.len(),
        })
    }
    /// The final five RAM address coordinates, consumed by batch 6a as the register address point.
    pub fn a_reg(&self) -> Result<&[F128], PointsError> {
        let address = self.a_ram()?;
        let start = address.len().checked_sub(5).ok_or(PointsError::Dimension {
            expected: 5,
            actual: address.len(),
        })?;
        address.get(start..).ok_or(PointsError::Dimension {
            expected: 5,
            actual: address.len(),
        })
    }
    /// The low-variable-first cycle point consumed by batches 5, 6a and 6b.
    pub fn r_4(&self) -> Result<&[F128], PointsError> {
        let a = self.a_ram()?.len();
        let point = &self.points.ram_read_checking.ram_ra;
        point.get(a..).ok_or(PointsError::Dimension {
            expected: a,
            actual: point.len(),
        })
    }
}

/// The batch and consumed cells established by `from_upstream` from verified router outputs.
pub struct Inputs {
    pub batch: Stage4Sumchecks<F128>,
    pub claims: Stage4InputClaims<F128>,
    pub points: Stage4InputPoints<F128>,
}

/// Converts batches 3a and 3b at their verified bit-first, cycle-second points.
/// Rejects inconsistent read points before drawing the public output-check challenge.
pub fn from_upstream<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    transcript: &mut T,
    stage3a: &Stage3aOutput,
    stage3b: &Stage3bOutput,
) -> Result<Inputs, VerifierError> {
    let bit = stage3a.x.get(..6).ok_or_else(|| {
        term_error(PointsError::Dimension {
            expected: 6,
            actual: stage3a.x.len(),
        })
    })?;
    let point = &stage3b.points.memory.ram_read_value;
    let expected = 6 + checked.log_T();
    if point.len() != expected {
        return Err(term_error(PointsError::Dimension {
            expected,
            actual: point.len(),
        }));
    }
    for other in [
        &stage3b.points.variant.rs1_value,
        &stage3b.points.variant.rs2_value,
        &stage3b.points.variant.rd_pre_value,
    ] {
        if other != point {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage: "Stage4".to_owned(),
                reason: "read claims do not share a bit and cycle point".to_owned(),
            });
        }
    }
    if point.get(..6) != Some(bit) {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: "Stage4".to_owned(),
            reason: "read points differ from the verified router bit point".to_owned(),
        });
    }
    let (_, cycle) = point.split_at(6);
    let batch = Stage4Sumchecks::new(
        checked.layout(),
        bit.to_vec(),
        cycle.to_vec(),
        Arc::clone(checked.shared_io()),
        transcript,
    )
    .map_err(term_error)?;
    let points = Stage4InputPoints {
        registers_read_checking: batch.registers_read_checking.input_points(),
        ram_read_checking: batch.ram_read_checking.input_points(),
        ram_output_check: Default::default(),
    };
    let claims = Stage4InputClaims {
        registers_read_checking: RegistersReadCheckingInputClaims {
            rs1_value: stage3b.claims.variant.rs1_value,
            rs2_value: stage3b.claims.variant.rs2_value,
            rd_pre_value: stage3b.claims.variant.rd_pre_value,
        },
        ram_read_checking: RamReadCheckingInputClaims {
            ram_read_value: stage3b.claims.memory.ram_read_value,
        },
        ram_output_check: Default::default(),
    };
    Ok(Inputs {
        batch,
        claims,
        points,
    })
}

/// Verifies the read and public output checks from batches 3a and 3b at their verified points.
pub fn verify<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    proof: &BatchProof<ReadCheckingValues>,
    transcript: &mut T,
    stage3a: &Stage3aOutput,
    stage3b: &Stage3bOutput,
) -> Result<Output, Rv64iVerifierError> {
    let inputs = from_upstream(checked, transcript, stage3a, stage3b)?;
    verify_inputs(
        &inputs.batch,
        proof,
        transcript,
        &inputs.claims,
        &inputs.points,
    )
}

/// Runs the same verification from converted low-variable-first inputs.
/// Batch-local callers establish bounds through `Stage4Sumchecks::new` before drawing member challenges.
pub fn verify_inputs<T: Transcript<Challenge = F128>>(
    batch: &Stage4Sumchecks<F128>,
    proof: &BatchProof<ReadCheckingValues>,
    transcript: &mut T,
    inputs: &Stage4InputClaims<F128>,
    input_points: &Stage4InputPoints<F128>,
) -> Result<Output, Rv64iVerifierError> {
    let challenges = batch.draw_challenges(transcript)?;
    let points = batch.verify(inputs, input_points, &challenges, proof, transcript)?;
    Ok(Output {
        claims: Stage4OutputClaims::from_wire(&proof.values),
        points,
    })
}

impl Stage4OutputClaims<F128> {
    pub fn into_wire(self) -> ReadCheckingValues {
        ReadCheckingValues {
            rs1_ra: self.registers_read_checking.rs1_ra,
            rs2_ra: self.registers_read_checking.rs2_ra,
            rd_wa: self.registers_read_checking.rd_wa,
            registers_val: self.registers_read_checking.registers_val,
            ram_ra: self.ram_read_checking.ram_ra,
            ram_val: self.ram_read_checking.ram_val,
            ram_val_final: self.ram_output_check.ram_val_final,
        }
    }
    pub fn from_wire(values: &ReadCheckingValues) -> Self {
        Self {
            registers_read_checking: RegistersReadCheckingOutputClaims {
                rs1_ra: values.rs1_ra,
                rs2_ra: values.rs2_ra,
                rd_wa: values.rd_wa,
                registers_val: values.registers_val,
            },
            ram_read_checking: RamReadCheckingOutputClaims {
                ram_ra: values.ram_ra,
                ram_val: values.ram_val,
            },
            ram_output_check: RamOutputCheckOutputClaims {
                ram_val_final: values.ram_val_final,
            },
        }
    }
}

impl Stage4Sumchecks<F128> {
    /// Verifies the batch and absorbs its seven hand-off values in member order.
    pub fn verify<T: Transcript<Challenge = F128>>(
        &self,
        inputs: &Stage4InputClaims<F128>,
        input_points: &Stage4InputPoints<F128>,
        challenges: &Stage4Challenges<F128>,
        proof: &BatchProof<ReadCheckingValues>,
        transcript: &mut T,
    ) -> Result<Stage4OutputPoints<F128>, VerifierError> {
        let outputs = Stage4OutputClaims::from_wire(&proof.values);
        let points = self.verify_clear(
            inputs,
            input_points,
            challenges,
            &outputs,
            &proof.rounds,
            transcript,
            4,
        )?;
        self.append_output_claims(transcript, &outputs);
        Ok(points)
    }
}

pub(super) fn term_error(error: PointsError) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: "Stage4".to_owned(),
        reason: error.to_string(),
    }
}

pub(super) fn check_read_points<F>(r_bit: &[F], r_3: &[F]) -> Result<(), PointsError> {
    if r_bit.len() != 6 {
        return Err(PointsError::Dimension {
            expected: 6,
            actual: r_bit.len(),
        });
    }
    if r_3.is_empty() || r_3.len() > usize::from(LOG_T_MAX) {
        return Err(PointsError::Dimension {
            expected: usize::from(LOG_T_MAX),
            actual: r_3.len(),
        });
    }
    Ok(())
}

pub(super) fn evaluate_output<F: JoltField>(
    expression: FamilyExpr<F>,
    values: &impl OutputClaims<F, OpeningId>,
    challenges: &impl SumcheckChallenges<F, ChallengeId>,
    mut derive: impl FnMut(&DerivedId) -> Result<F, VerifierError>,
) -> Result<F, VerifierError> {
    let mut terms = BTreeMap::new();
    expression.try_evaluate(
        |id| {
            values
                .resolve_output(id)
                .ok_or(VerifierError::MissingOpeningClaim { id: (*id).into() })
        },
        |id| {
            challenges
                .resolve_challenge(id)
                .ok_or(VerifierError::MissingStageClaimChallenge { id: (*id).into() })
        },
        |id| {
            if let Some(term) = terms.get(id) {
                return Ok(*term);
            }
            let value = derive(id)?;
            let _ = terms.insert(*id, value);
            Ok(value)
        },
    )
}
