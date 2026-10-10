//! Expands and verifies the five short router fold values.
use super::router_short::{RouterShortInputClaims, RouterShortOutputClaims};
use super::{
    Stage3aInputClaims, Stage3aInputPoints, Stage3aOutputClaims, Stage3aOutputPoints,
    Stage3aSumchecks,
};
use crate::commitment::BitsCommitmentScheme;
use crate::error::Rv64iVerifierError;
use crate::points::PointsError;
use crate::proof::{BatchProof, RouterFoldValues};
use crate::public::routes::{short_point, RouteTensors};
use crate::stages::stage2::Output as Stage2Output;
use crate::statement::CheckedInputs;
use jolt_field::F128;
use jolt_transcript::Transcript;
use jolt_verifier::VerifierError;
use std::sync::Arc;

/// Verified router folds and their low-variable-first short slots, consumed by batches 3b through 6b.
pub struct Output {
    /// Seventeen short-slot coordinates consumed by batches 3b, 4, 5, 6a and 6b.
    pub x: Vec<F128>,
    /// Five router folds consumed by batch 3b.
    pub claims: Stage3aOutputClaims<F128>,
}

impl Output {
    /// Retains verified short cells and recovers the shared low-variable-first slot point.
    /// Invalid compare-router point dimensions return `PointsError`.
    pub fn new(
        claims: Stage3aOutputClaims<F128>,
        points: Stage3aOutputPoints<F128>,
    ) -> Result<Self, PointsError> {
        let x = short_point(&points.router_short.compare)?;
        Ok(Self { x, claims })
    }
}

/// Concrete short relation and its consumed stage-2 cell at `w ++ r_1`.
pub struct Inputs {
    pub batch: Stage3aSumchecks<F128>,
    pub claims: Stage3aInputClaims<F128>,
    pub points: Stage3aInputPoints<F128>,
}

/// Converts the stage-2 routed cell into the short router relation.
/// Points are low-variable-first; malformed column or cycle dimensions return `PointsError`.
pub fn from_upstream<S: BitsCommitmentScheme>(
    checked: &CheckedInputs<'_, S>,
    stage2: &Stage2Output,
) -> Result<Inputs, PointsError> {
    let w = stage2.w()?;
    let r_1 = stage2.r_1()?;
    if r_1.len() != checked.log_T() {
        return Err(PointsError::Dimension {
            expected: checked.log_T(),
            actual: r_1.len(),
        });
    }
    let routes = Arc::new(RouteTensors::new(checked.layout())?);
    let batch = Stage3aSumchecks::new(w.to_vec(), r_1.to_vec(), routes)?;
    let claims = Stage3aInputClaims {
        router_short: RouterShortInputClaims {
            witness_routed: stage2.claims.spartan_inner.witness_routed,
        },
    };
    let points = batch.input_points();
    Ok(Inputs {
        batch,
        claims,
        points,
    })
}

/// Expands the five wire folds into canonical router cells.
pub fn expand(values: &RouterFoldValues) -> Stage3aOutputClaims<F128> {
    Stage3aOutputClaims {
        router_short: RouterShortOutputClaims {
            variant: values.variant,
            shift: values.shift,
            memory: values.memory,
            compare: values.compare,
            branch: values.branch,
        },
    }
}
/// Extracts the five scalar folds in wire order.
pub fn values(claims: &Stage3aOutputClaims<F128>) -> RouterFoldValues {
    let c = &claims.router_short;
    RouterFoldValues {
        variant: c.variant,
        shift: c.shift,
        memory: c.memory,
        compare: c.compare,
        branch: c.branch,
    }
}
/// Verifies batch 3a from the stage-2 routed value at its column-first, cycle-last point.
/// Invalid points or a failed terminal equation return a stage-3a verifier error.
pub fn verify<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    proof: &BatchProof<RouterFoldValues>,
    transcript: &mut T,
    stage2: &Stage2Output,
) -> Result<Output, Rv64iVerifierError> {
    let inputs = from_upstream(checked, stage2).map_err(|error| {
        VerifierError::StageClaimSumcheckFailed {
            stage: "Stage3a".to_owned(),
            reason: error.to_string(),
        }
    })?;
    verify_converted(proof, transcript, inputs)
}

/// Verifies the same short batch from already converted low-variable-first inputs.
/// The caller must derive them from the stage-2 cell; failed equations return a stage-3a error.
pub fn verify_converted<T: Transcript<Challenge = F128>>(
    proof: &BatchProof<RouterFoldValues>,
    transcript: &mut T,
    inputs: Inputs,
) -> Result<Output, Rv64iVerifierError> {
    let Inputs {
        batch,
        claims: inputs,
        points: input_points,
    } = inputs;
    let challenges = batch.draw_challenges(transcript)?;
    let claims = expand(&proof.values);
    batch.validate_output_claims(&claims)?;
    let points = batch.verify_clear(
        &inputs,
        &input_points,
        &challenges,
        &claims,
        &proof.rounds,
        transcript,
        3,
    )?;
    batch.append_output_claims(transcript, &claims);
    Output::new(claims, points).map_err(|error| {
        VerifierError::StageClaimSumcheckFailed {
            stage: "Stage3a".to_owned(),
            reason: error.to_string(),
        }
        .into()
    })
}
