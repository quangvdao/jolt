//! Terminal verification: project columns, check the batch, absorb columns, and open.
use super::{
    BitsReduction, BitsReductionInputClaims, BytecodeReadCycle, BytecodeReadCycleInputClaims,
    RamRaProduct, RamRaProductInputClaims, Stage6bInputClaims, Stage6bInputPoints,
    Stage6bSumchecks,
};
use crate::commitment::{BitsCommitmentScheme, BitsGeometry, BitsOpening};
use crate::error::{ProofDecodeError, Rv64iVerifierError};
use crate::proof::{BatchProof, BitsColumns};
use crate::stages::stage1::Output as Stage1Output;
use crate::stages::stage2::Output as Stage2Output;
use crate::stages::stage3a::Output as Stage3aOutput;
use crate::stages::stage3b::Output as Stage3bOutput;
use crate::stages::stage4::Output as Stage4Output;
use crate::stages::stage5::Output as Stage5Output;
use crate::stages::stage6a::Output as Stage6aOutput;
use crate::statement::CheckedInputs;
use jolt_field::F128;
use jolt_rv64i_arith::BITS_COLUMNS;
use jolt_transcript::Transcript;

/// The terminal cycle point consumed by the shared commitment opening.
pub struct Output {
    /// Low-variable-first cycle coordinates shared by all 256 column evaluations.
    pub point: Vec<F128>,
}

/// Terminal members and the nine consumed cells at their earlier points.
pub struct Inputs {
    pub batch: Stage6bSumchecks<F128>,
    pub claims: Stage6bInputClaims<F128>,
    pub points: Stage6bInputPoints<F128>,
}

/// Converts all seven earlier outputs to the terminal relations and their consumed cells.
/// Member constructors establish matching cycle dimensions and bounded chunk and short points.
#[expect(
    clippy::too_many_arguments,
    reason = "the terminal batch consumes all seven earlier outputs"
)]
pub fn from_upstream<S: BitsCommitmentScheme>(
    checked: &CheckedInputs<'_, S>,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    stage3a: &Stage3aOutput,
    stage3b: &Stage3bOutput,
    stage4: &Stage4Output,
    stage5: &Stage5Output,
    stage6a: &Stage6aOutput,
) -> Result<Inputs, Rv64iVerifierError> {
    let fail = BytecodeReadCycle::<F128>::term_error;
    let bytecode_read_cycle = BytecodeReadCycle::new(
        checked.layout(),
        stage6a.bytecode_folds,
        stage6a.points.bytecode_read_address.address_claim.clone(),
        stage3b.r_3().to_vec(),
        stage4.r_4().map_err(fail)?.to_vec(),
        stage5.r_5().to_vec(),
    )
    .map_err(fail)?;
    let ram_ra_product = RamRaProduct::new(
        checked.layout(),
        stage4.a_ram().map_err(fail)?.to_vec(),
        stage4.r_4().map_err(fail)?.to_vec(),
        stage5.r_5().to_vec(),
    )
    .map_err(fail)?;
    let bits_reduction = BitsReduction::new(
        checked.layout(),
        stage1.r_1().map_err(fail)?.to_vec(),
        stage3b.r_3().to_vec(),
        stage5.r_5().to_vec(),
        stage2.w().map_err(fail)?.to_vec(),
        stage3a.x.clone(),
    )
    .map_err(fail)?;
    let points = Stage6bInputPoints {
        bytecode_read_cycle: bytecode_read_cycle.input_points(),
        ram_ra_product: ram_ra_product.input_points(),
        bits_reduction: bits_reduction.input_points(),
    };
    let claims = Stage6bInputClaims {
        bytecode_read_cycle: BytecodeReadCycleInputClaims {
            address_claim: stage6a.claims.bytecode_read_address.address_claim,
        },
        ram_ra_product: RamRaProductInputClaims {
            ram_ra_read: stage4.claims.ram_read_checking.ram_ra,
            ram_ra_val: stage5.claims.ram_val_evaluation.ram_ra,
        },
        bits_reduction: BitsReductionInputClaims {
            direct_columns: stage2.claims.spartan_inner.direct_columns,
            variant_bits: stage3b.claims.variant.variant_bits,
            pos_ra_0: stage3b.claims.shift.pos_ra_0,
            pos_ra_1: stage3b.claims.shift.pos_ra_1,
            should_branch: stage3b.claims.branch.should_branch,
            inc: stage5.claims.registers_val_evaluation.inc,
        },
    };
    Ok(Inputs {
        batch: Stage6bSumchecks {
            bytecode_read_cycle,
            ram_ra_product,
            bits_reduction,
        },
        claims,
        points,
    })
}

/// Verifies the terminal batch from all seven upstream outputs, absorbs columns, then opens.
/// Scheme state comes from the completed preamble; the opening point is low-variable-first `(rho,r_6)`.
#[expect(
    clippy::too_many_arguments,
    reason = "the terminal batch consumes all earlier outputs and the retained commitment state"
)]
pub fn verify<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    proof: &BatchProof<BitsColumns>,
    transcript: &mut T,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    stage3a: &Stage3aOutput,
    stage3b: &Stage3bOutput,
    stage4: &Stage4Output,
    stage5: &Stage5Output,
    stage6a: &Stage6aOutput,
    state: S::VerifierState,
    opening_proof: &S::OpeningProof,
) -> Result<Output, Rv64iVerifierError> {
    let inputs = from_upstream(
        checked, stage1, stage2, stage3a, stage3b, stage4, stage5, stage6a,
    )?;
    verify_inputs(checked, proof, transcript, inputs, state, opening_proof)
}

/// Verifies the same constructed terminal members for batch-local independently evaluated inputs.
pub fn verify_inputs<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    proof: &BatchProof<BitsColumns>,
    transcript: &mut T,
    inputs: Inputs,
    state: S::VerifierState,
    opening_proof: &S::OpeningProof,
) -> Result<Output, Rv64iVerifierError> {
    if proof.values.0.len() != BITS_COLUMNS {
        return Err(Rv64iVerifierError::ProofShape(ProofDecodeError::ProofShape));
    }
    let Inputs {
        batch,
        claims: inputs,
        points: input_points,
    } = inputs;
    let claims = batch.expand(&proof.values.0)?;
    let challenges = batch.draw_challenges(transcript)?;
    let points = batch.verify_clear(
        &inputs,
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
        .into_iter()
        .next()
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
    Ok(Output { point })
}
