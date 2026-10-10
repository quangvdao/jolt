//! Terminal verification: project columns, check the batch, absorb columns, and open.
use super::{
    BitsReduction, BytecodeReadCycle, BytecodeReadCycleInputClaims, RamRaProduct,
    Stage6bInputClaims, Stage6bInputPoints, Stage6bOutputClaims, Stage6bOutputPoints,
    Stage6bSumchecks,
};
use crate::commitment::{BitsCommitmentScheme, BitsGeometry, BitsOpening};
use crate::error::{ProofDecodeError, Rv64iVerifierError};
use crate::points::PointsError;
use crate::proof::{BatchProof, BitsColumns};
use crate::stages::stage6a::verify::Stage6aOutput;
use crate::statement::CheckedInputs;
use jolt_field::F128;
use jolt_transcript::Transcript;

/// Earlier batch points. Address geometry and bytecode folds come from checked inputs and batch 6a.
pub struct Stage6bPoints {
    pub r_1: Vec<F128>,
    pub r_3: Vec<F128>,
    pub r_4: Vec<F128>,
    pub r_5: Vec<F128>,
    pub w: Vec<F128>,
    pub x: Vec<F128>,
    pub a_ram: Vec<F128>,
}
pub struct Output {
    pub point: Vec<F128>,
    pub claims: Stage6bOutputClaims<F128>,
    pub points: Stage6bOutputPoints<F128>,
}
#[expect(
    clippy::too_many_arguments,
    reason = "the terminal batch threads its scheme state and earlier aggregates once"
)]
pub fn verify<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    proof: &BatchProof<BitsColumns>,
    transcript: &mut T,
    previous: &Stage6aOutput,
    earlier: Stage6bPoints,
    inputs: &Stage6bInputClaims<F128>,
    state: S::VerifierState,
    opening_proof: &S::OpeningProof,
) -> Result<Output, Rv64iVerifierError> {
    let fail = |error: PointsError| BytecodeReadCycle::<F128>::term_error(error);
    for point in [&earlier.r_1, &earlier.r_3, &earlier.r_4, &earlier.r_5] {
        if point.len() != checked.log_T() {
            return Err(fail(PointsError::Dimension {
                expected: checked.log_T(),
                actual: point.len(),
            })
            .into());
        }
    }
    let bytecode_read_cycle = BytecodeReadCycle::new(
        checked.layout(),
        previous.bytecode_folds,
        previous
            .output_points
            .bytecode_read_address
            .address_claim
            .clone(),
        earlier.r_3.clone(),
        earlier.r_4.clone(),
        earlier.r_5.clone(),
    )
    .map_err(fail)?;
    let ram_ra_product = RamRaProduct::new(
        checked.layout(),
        earlier.a_ram,
        earlier.r_4,
        earlier.r_5.clone(),
    )
    .map_err(fail)?;
    let bits_reduction = BitsReduction::new(
        checked.layout(),
        earlier.r_1,
        earlier.r_3,
        earlier.r_5,
        earlier.w,
        earlier.x,
    )
    .map_err(fail)?;
    let batch = Stage6bSumchecks {
        bytecode_read_cycle,
        ram_ra_product,
        bits_reduction,
    };
    if inputs.bytecode_read_cycle.address_claim
        != previous.output_values.bytecode_read_address.address_claim
    {
        return Err(Rv64iVerifierError::ProofShape(ProofDecodeError::ProofShape));
    }
    let input_points = Stage6bInputPoints {
        bytecode_read_cycle: BytecodeReadCycleInputClaims {
            address_claim: previous
                .output_points
                .bytecode_read_address
                .address_claim
                .clone(),
        },
        ram_ra_product: batch.ram_ra_product.input_points(),
        bits_reduction: batch.bits_reduction.input_points(),
    };
    let claims = batch.expand(&proof.values.0)?;
    let challenges = batch.draw_challenges(transcript)?;
    let points = batch.verify_clear(
        inputs,
        &input_points,
        &challenges,
        &claims,
        &proof.rounds,
        transcript,
        6,
    )?;
    let point = points
        .bits_reduction
        .columns
        .first()
        .cloned()
        .ok_or(Rv64iVerifierError::ProofShape(ProofDecodeError::ProofShape))?;
    batch.append_output_claims(transcript, &claims);
    let rho = transcript.challenge_vector(8);
    S::verify_opening(
        checked.preprocessing().scheme(),
        state,
        &BitsOpening {
            geometry: BitsGeometry {
                log_T: checked.log_T(),
            },
            column_point: &rho,
            cycle_point: &point,
            columns: &proof.values.0,
        },
        opening_proof,
        transcript,
    )
    .map_err(|error| Rv64iVerifierError::Opening(Box::new(error)))?;
    Ok(Output {
        point,
        claims,
        points,
    })
}
