//! Batch 5 reduces register and RAM values to shared cycle update cells.

pub mod val_evaluation;
pub mod verify;

use crate::commitment::BitsCommitmentScheme;
use crate::points::PointsError;
use crate::proof::batch_geometry;
use crate::proof::{ReadCheckingValues, ValEvaluationValues};
use crate::public::ram_init;
use crate::statement::{CheckedInputs, LOG_T_MAX};
use jolt_field::{JoltField, Zero, F128};
use jolt_rv64i_arith::Layout;
use jolt_verifier::{stages::relations::SumcheckBatch, VerifierError};
pub use val_evaluation::{
    RamValEvaluation, RamValEvaluationChallenges, RamValEvaluationInputClaims,
    RamValEvaluationOutputClaims, RegistersValEvaluation, RegistersValEvaluationInputClaims,
    RegistersValEvaluationOutputClaims,
};

/// Batch 5 reduces values at shared address/bit points over low-variable-first cycles.
#[derive(SumcheckBatch)]
pub struct Stage5Sumchecks<F: JoltField> {
    pub registers_val_evaluation: RegistersValEvaluation<F>,
    pub ram_val_evaluation: RamValEvaluation<F>,
}

impl Stage5Sumchecks<F128> {
    /// Establishes shared address, bit and cycle geometry from the low-variable-first stage-4 value points.
    /// Rejects inconsistent points and dimensions before evaluating the canonical initial RAM.
    pub fn new<S: BitsCommitmentScheme>(
        checked: &CheckedInputs<'_, S>,
        points: &Stage5InputPoints<F128>,
    ) -> Result<Self, VerifierError> {
        Self::from_points(
            checked,
            &points.registers_val_evaluation.registers_val,
            &points.ram_val_evaluation.ram_val,
            &points.ram_val_evaluation.ram_val_final,
        )
    }

    pub(super) fn from_points<S: BitsCommitmentScheme>(
        checked: &CheckedInputs<'_, S>,
        register_point: &[F128],
        cycle_point: &[F128],
        final_point: &[F128],
    ) -> Result<Self, VerifierError> {
        let failed = |error: PointsError| VerifierError::StageClaimSumcheckFailed {
            stage: "Stage5".to_owned(),
            reason: error.to_string(),
        };
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
        if cycle_point.get(..a + 6) != Some(final_point)
            || cycle_point.get(register_start..) != Some(register_point)
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
    /// Value opening points in register/address, six bit, then cycle order.
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

pub use verify::Output;

impl Stage5Sumchecks<F128> {
    /// Constructs geometry instances with bounded address/bit points and no witness or public-memory table.
    /// Returns `PointsError` for unsupported trace widths before allocation; the generated schedule reads the concrete members.
    pub fn for_geometry(log_T: usize, layout: &Layout) -> Result<Self, PointsError> {
        if !(1..=usize::from(LOG_T_MAX)).contains(&log_T) {
            return Err(PointsError::Dimension {
                expected: usize::from(LOG_T_MAX),
                actual: log_T,
            });
        }
        Ok(Self {
            registers_val_evaluation: RegistersValEvaluation::new(
                vec![F128::zero(); 5],
                vec![F128::zero(); 6],
                vec![F128::zero(); log_T],
            )?,
            ram_val_evaluation: RamValEvaluation::new(
                vec![F128::zero(); layout.log_K_ram()],
                vec![F128::zero(); 6],
                vec![F128::zero(); log_T],
                F128::zero(),
            )?,
        })
    }
}
stage5_sumchecks_members!(batch_geometry);
