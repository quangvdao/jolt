//! Expands and verifies the five short router fold values.
use super::router_short::{RouterShortInputClaims, RouterShortOutputClaims};
use super::{Stage3aInputClaims, Stage3aOutputClaims, Stage3aOutputPoints, Stage3aSumchecks};
use crate::commitment::BitsCommitmentScheme;
use crate::error::Rv64iVerifierError;
use crate::points::PointsError;
use crate::proof::{BatchProof, RouterFoldValues};
use crate::public::routes::{short_point, RouteTensors};
use crate::statement::CheckedInputs;
use jolt_field::F128;
use jolt_transcript::Transcript;
use jolt_verifier::VerifierError;
use std::sync::Arc;

pub struct Output {
    pub x: Vec<F128>,
    pub r_1: Vec<F128>,
    pub values: RouterFoldValues,
    pub claims: Stage3aOutputClaims<F128>,
    pub points: Stage3aOutputPoints<F128>,
}
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
pub fn verify<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    proof: &BatchProof<RouterFoldValues>,
    transcript: &mut T,
    w: &[F128],
    r_1: &[F128],
    witness_routed: F128,
) -> Result<Output, Rv64iVerifierError> {
    let error = |error: PointsError| VerifierError::StageClaimSumcheckFailed {
        stage: "Stage3a".to_owned(),
        reason: error.to_string(),
    };
    if r_1.len() != checked.log_T() {
        return Err(error(PointsError::Dimension {
            expected: checked.log_T(),
            actual: r_1.len(),
        })
        .into());
    }
    let routes = Arc::new(RouteTensors::new(checked.layout()).map_err(error)?);
    let batch = Stage3aSumchecks::new(w.to_vec(), r_1.to_vec(), routes).map_err(error)?;
    let inputs = Stage3aInputClaims {
        router_short: RouterShortInputClaims { witness_routed },
    };
    let input_points = batch.input_points();
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
        2,
    )?;
    batch.append_output_claims(transcript, &claims);
    let x = short_point(&points.router_short.compare).map_err(error)?;
    Ok(Output {
        x,
        r_1: r_1.to_vec(),
        values: proof.values.clone(),
        claims,
        points,
    })
}
