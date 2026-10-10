//! Stage 2 consumes the outer cells and reduces them over witness columns.

use super::spartan_inner::{SpartanInnerInputClaims, SpartanInnerOutputClaims};
use super::{
    Stage2InputClaims, Stage2InputPoints, Stage2OutputClaims, Stage2OutputPoints, Stage2Sumchecks,
};
use crate::commitment::BitsCommitmentScheme;
use crate::error::Rv64iVerifierError;
use crate::points::PointsError;
use crate::proof::{BatchProof, InnerValues};
use crate::public::matrices::RowMatrices;
use crate::stages::stage1::verify::Output as Stage1Output;
use crate::statement::CheckedInputs;
use jolt_field::F128;
use jolt_transcript::Transcript;
use jolt_verifier::VerifierError;
use std::sync::Arc;

/// Verified witness-column cells at `w ++ r_1`, consumed by batches 3a and 6b.
pub struct Output {
    /// Routed and direct-column values consumed by batches 3a and 6b respectively.
    pub claims: Stage2OutputClaims<F128>,
    /// Low-variable-first column and cycle points consumed by batches 3a and 6b.
    pub points: Stage2OutputPoints<F128>,
}

impl Output {
    /// Borrows the ten low witness-column coordinates; malformed direct construction returns `PointsError`.
    pub fn w(&self) -> Result<&[F128], PointsError> {
        let point = &self.points.spartan_inner.witness_routed;
        point.get(..10).ok_or(PointsError::Dimension {
            expected: 10,
            actual: point.len(),
        })
    }
    /// Borrows the cycle suffix after the ten witness-column coordinates.
    /// Malformed direct construction returns `PointsError`.
    pub fn r_1(&self) -> Result<&[F128], PointsError> {
        let point = &self.points.spartan_inner.witness_routed;
        point.get(10..).ok_or(PointsError::Dimension {
            expected: 10,
            actual: point.len(),
        })
    }
}

/// Expands routed and direct witness-column wire values into their generated cells.
pub fn expand(values: &InnerValues) -> Stage2OutputClaims<F128> {
    Stage2OutputClaims {
        spartan_inner: SpartanInnerOutputClaims {
            witness_routed: values.witness_routed,
            direct_columns: values.direct_columns,
        },
    }
}

/// The batch instance and consumed cells share the verified outer points.
pub struct Inputs {
    pub batch: Stage2Sumchecks<F128>,
    pub claims: Stage2InputClaims<F128>,
    pub points: Stage2InputPoints<F128>,
}

/// Converts the verified stage-1 row-first, cycle-last cells into batch-2 inputs.
/// The concrete constructor checks row widths and the shared cycle dimension.
pub fn from_upstream(
    matrices: Arc<RowMatrices>,
    stage1: &Stage1Output,
) -> Result<Inputs, PointsError> {
    let f2 = &stage1.points.spartan_outer_f2;
    let f128 = &stage1.points.spartan_outer_f128;
    let rho_f2 = f2
        .az
        .get(..8)
        .ok_or(PointsError::Dimension {
            expected: 8,
            actual: f2.az.len(),
        })?
        .to_vec();
    let r_1 = f2
        .az
        .get(8..)
        .ok_or(PointsError::Dimension {
            expected: 8,
            actual: f2.az.len(),
        })?
        .to_vec();
    let m_F = matrices.f128_row_variables();
    let rho_f128 = f128
        .az
        .get(..m_F)
        .ok_or(PointsError::Dimension {
            expected: m_F,
            actual: f128.az.len(),
        })?
        .to_vec();
    let batch = Stage2Sumchecks::new(matrices, rho_f2, rho_f128, r_1)?;
    let a = &stage1.claims.spartan_outer_f2;
    let b = &stage1.claims.spartan_outer_f128;
    Ok(Inputs {
        batch,
        claims: Stage2InputClaims {
            spartan_inner: SpartanInnerInputClaims {
                az_f2: a.az,
                bz_f2: a.bz,
                cz_f2: a.cz,
                az_f128: b.az,
                bz_f128: b.bz,
                cz_f128: b.cz,
            },
        },
        points: Stage2InputPoints {
            spartan_inner: SpartanInnerInputClaims {
                az_f2: f2.az.clone(),
                bz_f2: f2.bz.clone(),
                cz_f2: f2.cz.clone(),
                az_f128: f128.az.clone(),
                bz_f128: f128.bz.clone(),
                cz_f128: f128.cz.clone(),
            },
        },
    })
}

/// Verifies batch 2 using the stage-1 outer cells and their low-variable-first points.
/// Invalid points or a failed terminal equation return a stage-2 verifier error.
pub fn verify<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    proof: &BatchProof<InnerValues>,
    transcript: &mut T,
    stage1: &Stage1Output,
) -> Result<Output, Rv64iVerifierError> {
    let matrices = Arc::new(RowMatrices::new(checked.layout()));
    let Inputs {
        batch,
        claims: inputs,
        points: input_points,
    } = from_upstream(matrices, stage1).map_err(|error| {
        VerifierError::StageClaimSumcheckFailed {
            stage: "Stage2".to_owned(),
            reason: error.to_string(),
        }
    })?;
    let challenges = batch.draw_challenges(transcript)?;
    let claims = expand(&proof.values);
    let points = batch.verify_clear(
        &inputs,
        &input_points,
        &challenges,
        &claims,
        &proof.rounds,
        transcript,
        2,
    )?;
    batch.append_output_claims(transcript, &claims);
    Ok(Output { claims, points })
}
