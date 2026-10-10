//! Batch 5 reduces register and RAM values to shared cycle update cells.

pub mod val_evaluation;
pub mod verify;

use crate::commitment::BitsCommitmentScheme;
use crate::points::PointsError;
use crate::proof::{ReadCheckingValues, ValEvaluationValues};
use crate::public::ram_init;
use crate::statement::CheckedInputs;
use jolt_field::{JoltField, F128};
use jolt_verifier::{stages::relations::SumcheckBatch, VerifierError};
pub use val_evaluation::{
    RamValEvaluation, RamValEvaluationChallenges, RamValEvaluationInputClaims,
    RamValEvaluationOutputClaims, RegistersValEvaluation, RegistersValEvaluationInputClaims,
    RegistersValEvaluationOutputClaims,
};

#[derive(SumcheckBatch)]
pub struct Stage5Sumchecks<F: JoltField> {
    pub registers_val_evaluation: RegistersValEvaluation<F>,
    pub ram_val_evaluation: RamValEvaluation<F>,
}

/// Consumed stage-4 cells retain their earlier opening points. Construction of
/// `Stage5Sumchecks` checks the shared address, bit and cycle geometry.
pub struct Stage5Source {
    pub values: Stage5InputClaims<F128>,
    pub points: Stage5InputPoints<F128>,
}
impl Stage5Sumchecks<F128> {
    pub fn new<S: BitsCommitmentScheme>(
        checked: &CheckedInputs<'_, S>,
        source: &Stage5Source,
    ) -> Result<Self, VerifierError> {
        let failed = |error: PointsError| VerifierError::StageClaimSumcheckFailed {
            stage: "Stage5".to_owned(),
            reason: error.to_string(),
        };
        let final_point = &source.points.ram_val_evaluation.ram_val_final;
        let cycle_point = &source.points.ram_val_evaluation.ram_val;
        let register_point = &source.points.registers_val_evaluation.registers_val;
        let a = checked.log_K_ram();
        let t = checked.log_T();
        for (point, expected) in [
            (final_point, a + 6),
            (cycle_point, a + 6 + t),
            (register_point, 5 + 6 + t),
        ] {
            if point.len() != expected {
                return Err(failed(PointsError::Dimension {
                    expected,
                    actual: point.len(),
                }));
            }
        }
        let (a_ram, r_bit) = final_point.split_at(a);
        let r_4 = cycle_point.get(a + 6..).ok_or_else(|| {
            failed(PointsError::Dimension {
                expected: a + 6 + t,
                actual: cycle_point.len(),
            })
        })?;
        let register_start = a.checked_sub(5).ok_or_else(|| {
            failed(PointsError::Dimension {
                expected: 5,
                actual: a,
            })
        })?;
        let a_reg = a_ram.get(register_start..).ok_or_else(|| {
            failed(PointsError::Dimension {
                expected: 5,
                actual: a,
            })
        })?;
        if cycle_point.get(..a + 6) != Some(final_point.as_slice())
            || cycle_point.get(register_start..) != Some(register_point.as_slice())
        {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage: "Stage5".to_owned(),
                reason: "consumed value claims have inconsistent address, bit or cycle points"
                    .to_owned(),
            });
        }
        let init = ram_init::evaluate(checked, a_ram, r_bit).map_err(failed)?;
        let batch = Self {
            registers_val_evaluation: RegistersValEvaluation::new(
                a_reg.to_vec(),
                r_bit.to_vec(),
                r_4.to_vec(),
            )
            .map_err(failed)?,
            ram_val_evaluation: RamValEvaluation::new(
                a_ram.to_vec(),
                r_bit.to_vec(),
                r_4.to_vec(),
                init,
            )
            .map_err(failed)?,
        };
        Ok(batch)
    }
    pub fn input_points(&self) -> Stage5InputPoints<F128> {
        Stage5InputPoints {
            registers_val_evaluation: self.registers_val_evaluation.input_points(),
            ram_val_evaluation: self.ram_val_evaluation.input_points(),
        }
    }
}
impl Stage5InputClaims<F128> {
    pub fn from_stage4(values: &ReadCheckingValues) -> Self {
        Self {
            registers_val_evaluation: RegistersValEvaluationInputClaims {
                registers_val: values.registers_val,
            },
            ram_val_evaluation: RamValEvaluationInputClaims {
                ram_val: values.ram_val,
                ram_val_final: values.ram_val_final,
            },
        }
    }
}
impl ValEvaluationValues {
    pub fn expand(&self) -> Stage5OutputClaims<F128> {
        Stage5OutputClaims {
            registers_val_evaluation: RegistersValEvaluationOutputClaims {
                rd_wa: self.rd_wa,
                store: self.store,
                inc: self.inc,
            },
            ram_val_evaluation: RamValEvaluationOutputClaims {
                ram_ra: self.ram_ra,
                store: self.store,
                inc: self.inc,
            },
        }
    }
}
impl Stage5OutputClaims<F128> {
    pub fn into_wire(self) -> ValEvaluationValues {
        ValEvaluationValues {
            rd_wa: self.registers_val_evaluation.rd_wa,
            store: self.registers_val_evaluation.store,
            inc: self.registers_val_evaluation.inc,
            ram_ra: self.ram_val_evaluation.ram_ra,
        }
    }
}
