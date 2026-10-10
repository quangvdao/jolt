//! Expands the eighteen router values and checks the five cycle reductions.

use jolt_field::F128;
use jolt_transcript::Transcript;
use jolt_verifier::VerifierError;

use super::router_cycle::{
    RouterCycleBranchInputClaims, RouterCycleBranchOutputClaims, RouterCycleCompareInputClaims,
    RouterCycleCompareOutputClaims, RouterCycleMemoryInputClaims, RouterCycleMemoryOutputClaims,
    RouterCycleShiftInputClaims, RouterCycleShiftOutputClaims, RouterCycleVariantInputClaims,
    RouterCycleVariantOutputClaims,
};
use super::{Stage3bInputClaims, Stage3bOutputClaims, Stage3bOutputPoints, Stage3bSumchecks};
use crate::commitment::BitsCommitmentScheme;
use crate::error::Rv64iVerifierError;
use crate::points::PointsError;
use crate::proof::{BatchProof, RouterCycleValues, RouterFoldValues};
use crate::stages::stage3a::verify::Output as ShortOutput;
use crate::statement::CheckedInputs;

pub struct Output {
    pub r_3: Vec<F128>,
    pub claims: Stage3bOutputClaims<F128>,
    pub points: Stage3bOutputPoints<F128>,
}

pub fn input_values(values: &RouterFoldValues) -> Stage3bInputClaims<F128> {
    Stage3bInputClaims {
        variant: RouterCycleVariantInputClaims {
            fold: values.variant,
        },
        shift: RouterCycleShiftInputClaims { fold: values.shift },
        memory: RouterCycleMemoryInputClaims {
            fold: values.memory,
        },
        compare: RouterCycleCompareInputClaims {
            fold: values.compare,
        },
        branch: RouterCycleBranchInputClaims {
            fold: values.branch,
        },
    }
}

/// Copies aliases from their canonical wire cells before generated validation.
pub fn expand(values: &RouterCycleValues) -> Stage3bOutputClaims<F128> {
    Stage3bOutputClaims {
        variant: RouterCycleVariantOutputClaims {
            rs1_value: values.rs1_value,
            rs2_value: values.rs2_value,
            rd_pre_value: values.rd_pre_value,
            imm: values.imm,
            fall_through_pc: values.fall_through_pc,
            pc_plus_imm: values.pc_plus_imm,
            pc: values.pc,
            next_pc: values.next_pc,
            variant_bits: values.variant_bits,
            variant: values.variant,
        },
        shift: RouterCycleShiftOutputClaims {
            rs1_value: values.rs1_value,
            shift_kind: values.shift_kind,
            pos_ra_0: values.pos_ra_0,
            pos_ra_1: values.pos_ra_1,
        },
        memory: RouterCycleMemoryOutputClaims {
            ram_read_value: values.ram_read_value,
            rs2_value: values.rs2_value,
            access_kind: values.access_kind,
            pos_ra_0: values.pos_ra_0,
        },
        compare: RouterCycleCompareOutputClaims {
            rs1_value: values.rs1_value,
            rs2_value: values.rs2_value,
            imm: values.imm,
            key_kind: values.key_kind,
            pos_ra_0: values.pos_ra_0,
            pos_ra_1: values.pos_ra_1,
        },
        branch: RouterCycleBranchOutputClaims {
            fall_through_pc: values.fall_through_pc,
            pc_plus_imm: values.pc_plus_imm,
            branch: values.branch,
            should_branch: values.should_branch,
        },
    }
}

pub fn values(claims: &Stage3bOutputClaims<F128>) -> RouterCycleValues {
    RouterCycleValues {
        rs1_value: claims.variant.rs1_value,
        rs2_value: claims.variant.rs2_value,
        rd_pre_value: claims.variant.rd_pre_value,
        imm: claims.variant.imm,
        fall_through_pc: claims.variant.fall_through_pc,
        pc_plus_imm: claims.variant.pc_plus_imm,
        pc: claims.variant.pc,
        next_pc: claims.variant.next_pc,
        variant_bits: claims.variant.variant_bits,
        variant: claims.variant.variant,
        shift_kind: claims.shift.shift_kind,
        pos_ra_0: claims.shift.pos_ra_0,
        pos_ra_1: claims.shift.pos_ra_1,
        ram_read_value: claims.memory.ram_read_value,
        access_kind: claims.memory.access_kind,
        key_kind: claims.compare.key_kind,
        branch: claims.branch.branch,
        should_branch: claims.branch.should_branch,
    }
}

pub fn verify<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    proof: &BatchProof<RouterCycleValues>,
    transcript: &mut T,
    stage3a: &ShortOutput,
) -> Result<Output, Rv64iVerifierError> {
    if stage3a.r_1.len() != checked.log_T() {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: "Stage3b".to_owned(),
            reason: "router cycle point differs from the checked trace dimension".to_owned(),
        }
        .into());
    }
    let error = |error: PointsError| VerifierError::StageClaimSumcheckFailed {
        stage: "Stage3b".to_owned(),
        reason: error.to_string(),
    };
    let sumchecks = Stage3bSumchecks::new(checked.layout(), stage3a.r_1.clone(), stage3a.x.clone())
        .map_err(error)?;
    let inputs = input_values(&stage3a.values);
    let input_points = sumchecks.input_points().map_err(error)?;
    let challenges = sumchecks.draw_challenges(transcript)?;
    let claims = expand(&proof.values);
    sumchecks.validate_output_claims(&claims)?;
    let points = sumchecks.verify_clear(
        &inputs,
        &input_points,
        &challenges,
        &claims,
        &proof.rounds,
        transcript,
        3,
    )?;
    sumchecks.append_output_claims(transcript, &claims);
    Ok(Output {
        r_3: points.branch.branch.clone(),
        claims,
        points,
    })
}
