//! Stage 1 draws the two row weights and verifies the outer batch.

use super::spartan_outer::SpartanOuterF128OutputClaims;
use super::spartan_outer::SpartanOuterF2OutputClaims;
use super::{Stage1InputClaims, Stage1OutputClaims, Stage1OutputPoints, Stage1Sumchecks};
use crate::commitment::BitsCommitmentScheme;
use crate::error::Rv64iVerifierError;
use crate::points::PointsError;
use crate::proof::{BatchProof, OuterValues};
use crate::public::matrices::RowMatrices;
use crate::statement::CheckedInputs;
use jolt_field::F128;
use jolt_transcript::Transcript;
use jolt_verifier::VerifierError;

/// Verified outer cells at row-first, cycle-last points, consumed by batches 2, 3b and 6b.
pub struct Output {
    /// Outer zero-check values consumed by batch 2.
    pub claims: Stage1OutputClaims<F128>,
    /// Row and cycle points consumed by batches 2, 3b and 6b.
    pub points: Stage1OutputPoints<F128>,
}
impl Output {
    /// Borrows the shared cycle suffix after the eight low row variables.
    /// A directly constructed output with fewer row coordinates returns `PointsError`.
    pub fn r_1(&self) -> Result<&[F128], PointsError> {
        let point = &self.points.spartan_outer_f2.az;
        point.get(8..).ok_or(PointsError::Dimension {
            expected: 8,
            actual: point.len(),
        })
    }
}

/// Draws the row-first, cycle-last outer weights on the protocol transcript.
/// The batch constructor checks their dimensions against the checked trace and layout.
pub fn from_checked<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    transcript: &mut T,
) -> Result<Stage1Sumchecks<F128>, PointsError> {
    let m_F = RowMatrices::f128_row_variables_for(checked.layout());
    let tau_f2 = transcript.challenge_vector(8 + checked.log_T());
    let tau_f128 = transcript.challenge_vector(m_F + checked.log_T());
    Stage1Sumchecks::new(checked.log_T(), m_F, tau_f2, tau_f128)
}

/// Expands the six wire values in canonical outer-cell order.
pub fn expand(values: &OuterValues) -> Stage1OutputClaims<F128> {
    Stage1OutputClaims {
        spartan_outer_f2: SpartanOuterF2OutputClaims {
            az: values.az_f2,
            bz: values.bz_f2,
            cz: values.cz_f2,
        },
        spartan_outer_f128: SpartanOuterF128OutputClaims {
            az: values.az_f128,
            bz: values.bz_f128,
            cz: values.cz_f128,
        },
    }
}

/// Verifies batch 1 against checked dimensions and draws both row-first, cycle-last weights.
/// A failed round or terminal equation returns a stage-1 verifier error.
pub fn verify<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    proof: &BatchProof<OuterValues>,
    transcript: &mut T,
) -> Result<Output, Rv64iVerifierError> {
    let batch = from_checked(checked, transcript).map_err(|error| {
        VerifierError::StageClaimSumcheckFailed {
            stage: "Stage1".to_owned(),
            reason: error.to_string(),
        }
    })?;
    let challenges = batch.draw_challenges(transcript)?;
    let inputs = Stage1InputClaims {
        spartan_outer_f2: Default::default(),
        spartan_outer_f128: Default::default(),
    };
    let input_points = batch.empty_input_points();
    let claims = expand(&proof.values);
    let points = batch.verify_clear(
        &inputs,
        &input_points,
        &challenges,
        &claims,
        &proof.rounds,
        transcript,
        1,
    )?;
    batch.append_output_claims(transcript, &claims);
    Ok(Output { claims, points })
}
