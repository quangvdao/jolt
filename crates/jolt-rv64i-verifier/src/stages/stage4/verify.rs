//! Batch 4 wiring: read checking followed by agreement with public I/O.

use crate::ids::{ChallengeId, DerivedId, FamilyExpr, OpeningId};
use jolt_claims::{OutputClaims, SumcheckChallenges};
use jolt_field::{JoltField, F128};
use jolt_program::preprocess::PublicIoMemory;
use jolt_rv64i_arith::Layout;
use jolt_transcript::Transcript;
use jolt_verifier::{stages::relations::SumcheckBatch, VerifierError};
use std::collections::BTreeMap;

use super::ram_output_check::{RamOutputCheck, RamOutputCheckOutputClaims};
use super::ram_read_checking::{RamReadChecking, RamReadCheckingOutputClaims};
use super::registers_read_checking::{RegistersReadChecking, RegistersReadCheckingOutputClaims};
use crate::commitment::BitsCommitmentScheme;
use crate::error::Rv64iVerifierError;
use crate::points::PointsError;
use crate::proof::{BatchProof, ReadCheckingValues};
use crate::statement::{CheckedInputs, LOG_T_MAX};

#[derive(SumcheckBatch)]
pub struct Stage4Sumchecks<F: JoltField> {
    pub registers_read_checking: RegistersReadChecking<F>,
    pub ram_read_checking: RamReadChecking<F>,
    pub ram_output_check: RamOutputCheck<F>,
}

impl<F: JoltField> Stage4Sumchecks<F> {
    /// Draws the address vector before the three register-read coefficients.
    /// The caller draws the member coefficients with `draw_challenges` next.
    pub fn new<T: Transcript<Challenge = F>>(
        layout: &Layout,
        r_bit: Vec<F>,
        r_3: Vec<F>,
        io: PublicIoMemory,
        transcript: &mut T,
    ) -> Result<Self, PointsError> {
        let registers_read_checking = RegistersReadChecking::new(r_bit.clone(), r_3.clone())?;
        let ram_read_checking = RamReadChecking::new(layout, r_bit.clone(), r_3)?;
        let tau = transcript.challenge_vector(layout.log_K_ram());
        let ram_output_check = RamOutputCheck::new(tau, r_bit, io)?;
        Ok(Self {
            registers_read_checking,
            ram_read_checking,
            ram_output_check,
        })
    }
}

/// Batch 4's point, opening cells and points, and its address challenge.
pub struct Output {
    pub batch_point: Vec<F128>,
    pub claims: Stage4OutputClaims<F128>,
    pub points: Stage4OutputPoints<F128>,
    pub tau: Vec<F128>,
}

/// Runs batch 4 from checked public data and the four earlier read claims.
/// The integration caller assembles these generated input aggregates from the
/// outputs of batches 3a and 3b; their points must share `r_bit ++ r_3`.
pub fn verify<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    proof: &BatchProof<ReadCheckingValues>,
    transcript: &mut T,
    inputs: &Stage4InputClaims<F128>,
    input_points: &Stage4InputPoints<F128>,
) -> Result<Output, Rv64iVerifierError> {
    let point = &input_points.ram_read_checking.ram_read_value;
    let expected = 6 + checked.log_T();
    if point.len() != expected {
        return Err(term_error(PointsError::Dimension {
            expected,
            actual: point.len(),
        })
        .into());
    }
    for other in [
        &input_points.registers_read_checking.rs1_value,
        &input_points.registers_read_checking.rs2_value,
        &input_points.registers_read_checking.rd_pre_value,
    ] {
        if other != point {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage: "Stage4".to_owned(),
                reason: "read claims do not share a bit and cycle point".to_owned(),
            }
            .into());
        }
    }
    let (bit, cycle) = point.split_at(6);
    let batch = Stage4Sumchecks::new(
        checked.layout(),
        bit.to_vec(),
        cycle.to_vec(),
        checked.io().clone(),
        transcript,
    )
    .map_err(term_error)?;
    let challenges = batch.draw_challenges(transcript)?;
    let points = batch.verify(inputs, input_points, &challenges, proof, transcript)?;
    Ok(Output {
        batch_point: points.ram_read_checking.ram_ra.clone(),
        claims: Stage4OutputClaims::from_wire(&proof.values),
        points,
        tau: batch.ram_output_check.tau().to_vec(),
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
