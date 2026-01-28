//! Streaming recursion witness generation (plan + emit).
//!
//! This module implements the two-pass pipeline:
//! - **PlanPass**: derive the public constraint ordering + build native witness stores.
//! - **EmitPass**: derive the prefix-packing layout and emit the packed dense evaluation table.

use crate::poly::dense_mlpoly::DensePolynomial;
use crate::zkvm::recursion::constraints::system::{
    ConstraintLocator, ConstraintSystem, ConstraintType, G1AddNative, G1ScalarMulNative,
    G2AddNative, G2ScalarMulNative, GtMulNativeRows, PolyType, RecursionMatrixShape,
};
use crate::zkvm::recursion::g1::scalar_multiplication::G1ScalarMulPublicInputs;
use crate::zkvm::recursion::g2::scalar_multiplication::G2ScalarMulPublicInputs;
use crate::zkvm::recursion::gt::exponentiation::{GtExpPublicInputs, GtExpWitness};
use crate::zkvm::recursion::gt::indexing::{k_exp, k_mul};
use crate::zkvm::recursion::prefix_packing::{PrefixPackedEntry, PrefixPackingLayout};
use crate::zkvm::recursion::witness::{GTCombineWitness, GTExpOpWitness, GTMulOpWitness};

use ark_bn254::Fq;
use ark_ff::Zero;
use dory::recursion::WitnessCollection;
use jolt_optimizations::fq12_to_multilinear_evals;
use rayon::prelude::*;

use crate::poly::commitment::dory::recursion::JoltWitness;

fn env_flag_default(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(default)
}

#[tracing::instrument(skip_all, fields(num_constraints))]
fn compute_shape(num_constraints: usize) -> RecursionMatrixShape {
    let num_constraints_padded = num_constraints.next_power_of_two();
    let num_rows_unpadded = PolyType::NUM_TYPES * num_constraints_padded;
    let num_s_vars = (num_rows_unpadded as f64).log2().ceil() as usize;
    let num_constraint_vars = 11usize;
    let num_vars = num_s_vars + num_constraint_vars;
    RecursionMatrixShape {
        num_constraints,
        num_constraints_padded,
        num_constraint_vars,
        num_s_vars,
        num_vars,
    }
}

fn pack_gt_exp_op_witness(exp_wit: &GTExpOpWitness) -> GtExpWitness<Fq> {
    // Matches the legacy logic in `DoryMatrixBuilder::add_combine_witness`.
    if exp_wit.bits.is_empty() {
        let base_mle = fq12_to_multilinear_evals(&exp_wit.base);
        let base2_mle = fq12_to_multilinear_evals(&(exp_wit.base * exp_wit.base));
        let base3_mle = fq12_to_multilinear_evals(&(exp_wit.base * exp_wit.base * exp_wit.base));

        let rho_mles = if exp_wit.rho_mles.is_empty() {
            vec![fq12_to_multilinear_evals(&exp_wit.result)]
        } else {
            exp_wit.rho_mles.clone()
        };
        let quotient_mles = exp_wit.quotient_mles.clone();

        return GtExpWitness::from_steps(
            &rho_mles,
            &quotient_mles,
            &exp_wit.bits,
            &base_mle,
            &base2_mle,
            &base3_mle,
        );
    }

    let base_mle = fq12_to_multilinear_evals(&exp_wit.base);
    let base2_mle = fq12_to_multilinear_evals(&(exp_wit.base * exp_wit.base));
    let base3_mle = fq12_to_multilinear_evals(&(exp_wit.base * exp_wit.base * exp_wit.base));

    let num_steps = exp_wit.bits.len().div_ceil(2);
    let (rho_mles, quotient_mles) = if exp_wit.quotient_mles.len() != num_steps {
        let mut fixed_quotients = exp_wit.quotient_mles.clone();
        fixed_quotients.resize(num_steps, vec![Fq::zero(); 16]);

        let mut fixed_rhos = exp_wit.rho_mles.clone();
        if fixed_rhos.len() < num_steps + 1 {
            let result_mle = fq12_to_multilinear_evals(&exp_wit.result);
            while fixed_rhos.len() < num_steps + 1 {
                fixed_rhos.push(result_mle.clone());
            }
        }
        (fixed_rhos, fixed_quotients)
    } else {
        (exp_wit.rho_mles.clone(), exp_wit.quotient_mles.clone())
    };

    GtExpWitness::from_steps(
        &rho_mles,
        &quotient_mles,
        &exp_wit.bits,
        &base_mle,
        &base2_mle,
        &base3_mle,
    )
}

fn gt_mul_rows_from_op_witness(w: &GTMulOpWitness) -> Option<GtMulNativeRows> {
    // Matches the legacy guard in `DoryMatrixBuilder::add_gt_mul_op_witness`.
    if w.quotient_mle.is_empty() {
        return None;
    }
    let lhs = fq12_to_multilinear_evals(&w.lhs);
    let rhs = fq12_to_multilinear_evals(&w.rhs);
    let result = fq12_to_multilinear_evals(&w.result);
    let quotient = w.quotient_mle.clone();
    debug_assert_eq!(lhs.len(), 16);
    debug_assert_eq!(rhs.len(), 16);
    debug_assert_eq!(result.len(), 16);
    debug_assert_eq!(quotient.len(), 16);
    Some(GtMulNativeRows {
        lhs,
        rhs,
        result,
        quotient,
    })
}

#[tracing::instrument(
    skip_all,
    name = "recursion.witness_gen.plan.gt_exp_direct",
    fields(num_ops = witness_collection.gt_exp.len())
)]
fn plan_gt_exp_direct(
    witness_collection: &WitnessCollection<JoltWitness>,
) -> Vec<(GtExpWitness<Fq>, GtExpPublicInputs)> {
    let mut gt_exp_items: Vec<_> = witness_collection.gt_exp.iter().collect();
    gt_exp_items.sort_by_key(|(op_id, _)| *op_id);

    gt_exp_items
        .par_iter()
        .map(|(_op_id, witness)| {
            let base_mle = fq12_to_multilinear_evals(&witness.base);
            let base2 = witness.base * witness.base;
            let base2_mle = fq12_to_multilinear_evals(&base2);
            let base3 = base2 * witness.base;
            let base3_mle = fq12_to_multilinear_evals(&base3);

            let packed = GtExpWitness::from_steps(
                &witness.rho_mles,
                &witness.quotient_mles,
                &witness.bits,
                &base_mle,
                &base2_mle,
                &base3_mle,
            );
            let public_input = GtExpPublicInputs::new(witness.base, witness.bits.clone());
            (packed, public_input)
        })
        .collect()
}

#[tracing::instrument(
    skip_all,
    name = "recursion.witness_gen.plan.gt_mul_direct",
    fields(num_ops = witness_collection.gt_mul.len())
)]
fn plan_gt_mul_direct(witness_collection: &WitnessCollection<JoltWitness>) -> Vec<GtMulNativeRows> {
    let mut gt_mul_items: Vec<_> = witness_collection.gt_mul.iter().collect();
    gt_mul_items.sort_by_key(|(op_id, _)| *op_id);

    gt_mul_items
        .par_iter()
        .map(|(_op_id, witness)| {
            let lhs = fq12_to_multilinear_evals(&witness.lhs);
            let rhs = fq12_to_multilinear_evals(&witness.rhs);
            let result = fq12_to_multilinear_evals(&witness.result);
            let quotient = witness.quotient_mle.clone();
            debug_assert_eq!(lhs.len(), 16);
            debug_assert_eq!(rhs.len(), 16);
            debug_assert_eq!(result.len(), 16);
            debug_assert_eq!(quotient.len(), 16);
            GtMulNativeRows {
                lhs,
                rhs,
                result,
                quotient,
            }
        })
        .collect()
}

#[tracing::instrument(
    skip_all,
    name = "recursion.witness_gen.plan.g1_scalar_mul_direct",
    fields(num_ops = witness_collection.g1_scalar_mul.len())
)]
fn plan_g1_scalar_mul_direct(
    witness_collection: &WitnessCollection<JoltWitness>,
) -> Vec<(G1ScalarMulNative, G1ScalarMulPublicInputs)> {
    let mut g1_items: Vec<_> = witness_collection.g1_scalar_mul.iter().collect();
    g1_items.sort_by_key(|(op_id, _)| *op_id);

    g1_items
        .into_iter()
        .map(|(_op_id, witness)| {
            debug_assert_eq!(witness.x_a_mles.len(), 1);
            debug_assert_eq!(witness.y_a_mles.len(), 1);
            debug_assert_eq!(witness.x_t_mles.len(), 1);
            debug_assert_eq!(witness.y_t_mles.len(), 1);
            debug_assert_eq!(witness.x_a_next_mles.len(), 1);
            debug_assert_eq!(witness.y_a_next_mles.len(), 1);
            debug_assert_eq!(witness.t_is_infinity_mles.len(), 1);
            debug_assert_eq!(witness.a_is_infinity_mles.len(), 1);

            let base_point = (witness.point_base.x, witness.point_base.y);
            let rows = G1ScalarMulNative {
                base_point,
                x_a: witness.x_a_mles[0].clone(),
                y_a: witness.y_a_mles[0].clone(),
                x_t: witness.x_t_mles[0].clone(),
                y_t: witness.y_t_mles[0].clone(),
                x_a_next: witness.x_a_next_mles[0].clone(),
                y_a_next: witness.y_a_next_mles[0].clone(),
                t_indicator: witness.t_is_infinity_mles[0].clone(),
                a_indicator: witness.a_is_infinity_mles[0].clone(),
            };
            let public_input = G1ScalarMulPublicInputs::new(witness.scalar);
            (rows, public_input)
        })
        .collect()
}

#[tracing::instrument(
    skip_all,
    name = "recursion.witness_gen.plan.g2_scalar_mul_direct",
    fields(num_ops = witness_collection.g2_scalar_mul.len())
)]
fn plan_g2_scalar_mul_direct(
    witness_collection: &WitnessCollection<JoltWitness>,
) -> Vec<(G2ScalarMulNative, G2ScalarMulPublicInputs)> {
    let mut g2_items: Vec<_> = witness_collection.g2_scalar_mul.iter().collect();
    g2_items.sort_by_key(|(op_id, _)| *op_id);

    g2_items
        .into_iter()
        .map(|(_op_id, witness)| {
            debug_assert_eq!(witness.x_a_c0_mles.len(), 1);
            debug_assert_eq!(witness.x_a_c1_mles.len(), 1);
            debug_assert_eq!(witness.y_a_c0_mles.len(), 1);
            debug_assert_eq!(witness.y_a_c1_mles.len(), 1);
            debug_assert_eq!(witness.x_t_c0_mles.len(), 1);
            debug_assert_eq!(witness.x_t_c1_mles.len(), 1);
            debug_assert_eq!(witness.y_t_c0_mles.len(), 1);
            debug_assert_eq!(witness.y_t_c1_mles.len(), 1);
            debug_assert_eq!(witness.x_a_next_c0_mles.len(), 1);
            debug_assert_eq!(witness.x_a_next_c1_mles.len(), 1);
            debug_assert_eq!(witness.y_a_next_c0_mles.len(), 1);
            debug_assert_eq!(witness.y_a_next_c1_mles.len(), 1);
            debug_assert_eq!(witness.t_is_infinity_mles.len(), 1);
            debug_assert_eq!(witness.a_is_infinity_mles.len(), 1);

            let base_point = (witness.point_base.x, witness.point_base.y);
            let rows = G2ScalarMulNative {
                base_point,
                x_a_c0: witness.x_a_c0_mles[0].clone(),
                x_a_c1: witness.x_a_c1_mles[0].clone(),
                y_a_c0: witness.y_a_c0_mles[0].clone(),
                y_a_c1: witness.y_a_c1_mles[0].clone(),
                x_t_c0: witness.x_t_c0_mles[0].clone(),
                x_t_c1: witness.x_t_c1_mles[0].clone(),
                y_t_c0: witness.y_t_c0_mles[0].clone(),
                y_t_c1: witness.y_t_c1_mles[0].clone(),
                x_a_next_c0: witness.x_a_next_c0_mles[0].clone(),
                x_a_next_c1: witness.x_a_next_c1_mles[0].clone(),
                y_a_next_c0: witness.y_a_next_c0_mles[0].clone(),
                y_a_next_c1: witness.y_a_next_c1_mles[0].clone(),
                t_indicator: witness.t_is_infinity_mles[0].clone(),
                a_indicator: witness.a_is_infinity_mles[0].clone(),
            };

            let public_input = G2ScalarMulPublicInputs::new(witness.scalar);
            (rows, public_input)
        })
        .collect()
}

#[tracing::instrument(
    skip_all,
    name = "recursion.witness_gen.plan.g1_add_direct",
    fields(num_ops = witness_collection.g1_add.len())
)]
fn plan_g1_add_direct(witness_collection: &WitnessCollection<JoltWitness>) -> Vec<G1AddNative> {
    let mut g1_add_items: Vec<_> = witness_collection.g1_add.iter().collect();
    g1_add_items.sort_by_key(|(op_id, _)| *op_id);

    g1_add_items
        .into_iter()
        .map(|(_op_id, witness)| G1AddNative {
            x_p: witness.x_p,
            y_p: witness.y_p,
            ind_p: witness.ind_p,
            x_q: witness.x_q,
            y_q: witness.y_q,
            ind_q: witness.ind_q,
            x_r: witness.x_r,
            y_r: witness.y_r,
            ind_r: witness.ind_r,
            lambda: witness.lambda,
            inv_delta_x: witness.inv_delta_x,
            is_double: witness.is_double,
            is_inverse: witness.is_inverse,
        })
        .collect()
}

#[tracing::instrument(
    skip_all,
    name = "recursion.witness_gen.plan.g2_add_direct",
    fields(num_ops = witness_collection.g2_add.len())
)]
fn plan_g2_add_direct(witness_collection: &WitnessCollection<JoltWitness>) -> Vec<G2AddNative> {
    let mut g2_add_items: Vec<_> = witness_collection.g2_add.iter().collect();
    g2_add_items.sort_by_key(|(op_id, _)| *op_id);

    g2_add_items
        .into_iter()
        .map(|(_op_id, witness)| G2AddNative {
            x_p_c0: witness.x_p_c0,
            x_p_c1: witness.x_p_c1,
            y_p_c0: witness.y_p_c0,
            y_p_c1: witness.y_p_c1,
            ind_p: witness.ind_p,
            x_q_c0: witness.x_q_c0,
            x_q_c1: witness.x_q_c1,
            y_q_c0: witness.y_q_c0,
            y_q_c1: witness.y_q_c1,
            ind_q: witness.ind_q,
            x_r_c0: witness.x_r_c0,
            x_r_c1: witness.x_r_c1,
            y_r_c0: witness.y_r_c0,
            y_r_c1: witness.y_r_c1,
            ind_r: witness.ind_r,
            lambda_c0: witness.lambda_c0,
            lambda_c1: witness.lambda_c1,
            inv_delta_x_c0: witness.inv_delta_x_c0,
            inv_delta_x_c1: witness.inv_delta_x_c1,
            is_double: witness.is_double,
            is_inverse: witness.is_inverse,
        })
        .collect()
}

#[tracing::instrument(
    skip_all,
    name = "recursion.witness_gen.plan.combine_witness",
    fields(num_exp = cw.exp_witnesses.len(), num_mul_layers = cw.mul_layers.len())
)]
fn plan_combine_witness(
    cw: &GTCombineWitness,
    constraint_types: &mut Vec<ConstraintType>,
    locator_by_constraint: &mut Vec<ConstraintLocator>,
    gt_exp_witnesses: &mut Vec<GtExpWitness<Fq>>,
    gt_exp_public_inputs: &mut Vec<GtExpPublicInputs>,
    gt_mul_rows: &mut Vec<GtMulNativeRows>,
) {
    // Append GT exp constraints for combine terms.
    for exp_wit in &cw.exp_witnesses {
        let packed = pack_gt_exp_op_witness(exp_wit);
        let public_input = GtExpPublicInputs::new(exp_wit.base, exp_wit.bits.clone());
        let local = gt_exp_witnesses.len();
        gt_exp_witnesses.push(packed);
        gt_exp_public_inputs.push(public_input);
        constraint_types.push(ConstraintType::GtExp);
        locator_by_constraint.push(ConstraintLocator::GtExp { local });
    }

    // Append GT mul constraints for combine reduction tree.
    for layer in &cw.mul_layers {
        for mul_wit in layer {
            if let Some(rows) = gt_mul_rows_from_op_witness(mul_wit) {
                let local = gt_mul_rows.len();
                gt_mul_rows.push(rows);
                constraint_types.push(ConstraintType::GtMul);
                locator_by_constraint.push(ConstraintLocator::GtMul { local });
            }
        }
    }
}

/// PlanPass: build constraint ordering + native witness stores + `RecursionMatrixShape`.
#[tracing::instrument(
    skip_all,
    name = "recursion.witness_gen.plan_constraint_system",
    fields(
        direct_gt_exp = witness_collection.gt_exp.len(),
        direct_gt_mul = witness_collection.gt_mul.len(),
        direct_g1_scalar_mul = witness_collection.g1_scalar_mul.len(),
        direct_g2_scalar_mul = witness_collection.g2_scalar_mul.len(),
        direct_g1_add = witness_collection.g1_add.len(),
        direct_g2_add = witness_collection.g2_add.len(),
        has_combine = combine_witness.is_some()
    )
)]
pub fn plan_constraint_system(
    witness_collection: &WitnessCollection<JoltWitness>,
    combine_witness: Option<&GTCombineWitness>,
    g_poly_4var: DensePolynomial<Fq>,
) -> Result<ConstraintSystem, Box<dyn std::error::Error>> {
    // Constraint-family toggles (useful to isolate failures).
    let enable_gt_mul = env_flag_default("JOLT_RECURSION_ENABLE_GT_MUL", true);
    let enable_g1_scalar_mul = env_flag_default("JOLT_RECURSION_ENABLE_G1_SCALAR_MUL", true);
    let enable_g2_scalar_mul = env_flag_default("JOLT_RECURSION_ENABLE_G2_SCALAR_MUL", true);
    let enable_g1_add = env_flag_default("JOLT_RECURSION_ENABLE_G1_ADD", true);
    let enable_g2_add = env_flag_default("JOLT_RECURSION_ENABLE_G2_ADD", true);
    #[cfg(feature = "experimental-pairing-recursion")]
    let enable_pairing = env_flag_default("JOLT_RECURSION_ENABLE_PAIRING", true);

    // Outputs
    let mut constraint_types: Vec<ConstraintType> = Vec::new();
    let mut locator_by_constraint: Vec<ConstraintLocator> = Vec::new();

    let mut gt_exp_witnesses: Vec<GtExpWitness<Fq>> = Vec::new();
    let mut gt_exp_public_inputs: Vec<GtExpPublicInputs> = Vec::new();

    let mut gt_mul_rows: Vec<GtMulNativeRows> = Vec::new();
    let mut g1_scalar_mul_rows: Vec<G1ScalarMulNative> = Vec::new();
    let mut g2_scalar_mul_rows: Vec<G2ScalarMulNative> = Vec::new();
    let mut g1_add_rows: Vec<G1AddNative> = Vec::new();
    let mut g2_add_rows: Vec<G2AddNative> = Vec::new();

    let mut g1_scalar_mul_public_inputs: Vec<G1ScalarMulPublicInputs> = Vec::new();
    let mut g2_scalar_mul_public_inputs: Vec<G2ScalarMulPublicInputs> = Vec::new();

    // ---- GT exp (direct) ----
    let prepared_gt_exp = plan_gt_exp_direct(witness_collection);

    for (packed, public_input) in prepared_gt_exp {
        let local = gt_exp_witnesses.len();
        gt_exp_witnesses.push(packed);
        gt_exp_public_inputs.push(public_input);
        constraint_types.push(ConstraintType::GtExp);
        locator_by_constraint.push(ConstraintLocator::GtExp { local });
    }

    // ---- GT mul (direct) ----
    if enable_gt_mul {
        let prepared_gt_mul = plan_gt_mul_direct(witness_collection);
        for rows in prepared_gt_mul {
            let local = gt_mul_rows.len();
            gt_mul_rows.push(rows);
            constraint_types.push(ConstraintType::GtMul);
            locator_by_constraint.push(ConstraintLocator::GtMul { local });
        }
    }

    // ---- G1 scalar mul ----
    if enable_g1_scalar_mul {
        for (rows, public_input) in plan_g1_scalar_mul_direct(witness_collection) {
            let base_point = rows.base_point;
            let local = g1_scalar_mul_rows.len();
            g1_scalar_mul_rows.push(rows);
            g1_scalar_mul_public_inputs.push(public_input);
            constraint_types.push(ConstraintType::G1ScalarMul { base_point });
            locator_by_constraint.push(ConstraintLocator::G1ScalarMul { local });
        }
    }

    // ---- G2 scalar mul ----
    if enable_g2_scalar_mul {
        for (rows, public_input) in plan_g2_scalar_mul_direct(witness_collection) {
            let base_point = rows.base_point;
            let local = g2_scalar_mul_rows.len();
            g2_scalar_mul_rows.push(rows);
            g2_scalar_mul_public_inputs.push(public_input);
            constraint_types.push(ConstraintType::G2ScalarMul { base_point });
            locator_by_constraint.push(ConstraintLocator::G2ScalarMul { local });
        }
    }

    // ---- G1 add ----
    if enable_g1_add {
        for rows in plan_g1_add_direct(witness_collection) {
            let local = g1_add_rows.len();
            g1_add_rows.push(rows);
            constraint_types.push(ConstraintType::G1Add);
            locator_by_constraint.push(ConstraintLocator::G1Add { local });
        }
    }

    // ---- G2 add ----
    if enable_g2_add {
        for rows in plan_g2_add_direct(witness_collection) {
            let local = g2_add_rows.len();
            g2_add_rows.push(rows);
            constraint_types.push(ConstraintType::G2Add);
            locator_by_constraint.push(ConstraintLocator::G2Add { local });
        }
    }

    // ---- Pairing (experimental) ----
    #[cfg(feature = "experimental-pairing-recursion")]
    if enable_pairing {
        // NOTE: Streaming pairing recursion stores are wired up later in the plan.
        // For now, keep behavior consistent by not claiming support here.
        // (The legacy matrix path still supports this feature.)
    }

    // ---- Combine witness (homomorphic combine offloading) ----
    if let Some(cw) = combine_witness {
        plan_combine_witness(
            cw,
            &mut constraint_types,
            &mut locator_by_constraint,
            &mut gt_exp_witnesses,
            &mut gt_exp_public_inputs,
            &mut gt_mul_rows,
        );
    }

    debug_assert_eq!(
        constraint_types.len(),
        locator_by_constraint.len(),
        "locator length must match constraint_types length"
    );

    let shape = compute_shape(constraint_types.len());

    Ok(ConstraintSystem {
        shape,
        constraint_types,
        locator_by_constraint,
        g_poly: g_poly_4var,
        gt_exp_witnesses,
        gt_exp_public_inputs,
        gt_mul_rows,
        g1_scalar_mul_rows,
        g2_scalar_mul_rows,
        g1_add_rows,
        g2_add_rows,
        g1_scalar_mul_public_inputs,
        g2_scalar_mul_public_inputs,
    })
}

/// EmitPass: build `PrefixPackingLayout` and emit the prefix-packed dense polynomial.
///
/// Implemented in the next todo.
#[tracing::instrument(
    skip_all,
    name = "recursion.witness_gen.emit_dense",
    fields(num_constraints = cs.constraint_types.len())
)]
pub fn emit_dense(cs: &ConstraintSystem) -> (DensePolynomial<Fq>, PrefixPackingLayout) {
    #[inline]
    fn bit_reverse(mut x: usize, bits: usize) -> usize {
        let mut y = 0usize;
        for _ in 0..bits {
            y = (y << 1) | (x & 1);
            x >>= 1;
        }
        y
    }

    #[inline]
    fn fill_block(dst: &mut [Fq], src: &[Fq], num_vars: usize) {
        let native_size = 1usize << num_vars;
        debug_assert_eq!(dst.len(), native_size);
        debug_assert!(src.len() >= native_size);
        for t in 0..native_size {
            dst[t] = src[bit_reverse(t, num_vars)];
        }
    }

    fn fill_entry(dst: &mut [Fq], cs: &ConstraintSystem, entry: &PrefixPackedEntry) {
        if entry.is_gt_fused {
            // Option B: commit exp/mul fused rows at their family-local padded sizes.
            let num_vars_gt_exp = 11usize + k_exp(&cs.constraint_types);
            let num_vars_gt_mul = 4usize + k_mul(&cs.constraint_types);

            // IMPORTANT (no-padding GTMul + c-suffix):
            // - for GTExp fused rows, variables are (x11 low bits, c_gt high bits), size 2^(11+k)
            // - for GTMul fused rows, variables are (u low bits, c_gt high bits), size 2^(4+k)
            //
            // This avoids the old 4→11 replication for GTMul.
            let mut fused_src = vec![Fq::zero(); 1usize << entry.num_vars];

            match entry.poly_type {
                PolyType::RhoPrev | PolyType::Quotient => {
                    if entry.num_vars != num_vars_gt_exp {
                        panic!(
                            "GT-fused GTExp entry has num_vars={}, expected {} (11 + k_exp)",
                            entry.num_vars, num_vars_gt_exp
                        );
                    }
                    let row_size = 1usize << 11;
                    for global_idx in 0..cs.constraint_types.len() {
                        let ConstraintLocator::GtExp { local } =
                            cs.locator_by_constraint[global_idx]
                        else {
                            continue;
                        };
                        let c_exp = local;
                        let src = match entry.poly_type {
                            PolyType::RhoPrev => &cs.gt_exp_witnesses[local].rho_packed,
                            PolyType::Quotient => &cs.gt_exp_witnesses[local].quotient_packed,
                            _ => unreachable!(),
                        };
                        debug_assert_eq!(src.len(), row_size);
                        let off = c_exp << 11;
                        fused_src[off..off + row_size].copy_from_slice(src);
                    }
                }
                PolyType::MulLhs
                | PolyType::MulRhs
                | PolyType::MulResult
                | PolyType::MulQuotient => {
                    if entry.num_vars != num_vars_gt_mul {
                        panic!(
                            "GT-fused GTMul entry has num_vars={}, expected {} (4 + k_mul)",
                            entry.num_vars, num_vars_gt_mul
                        );
                    }
                    let row_size = 1usize << 4;
                    for global_idx in 0..cs.constraint_types.len() {
                        let ConstraintLocator::GtMul { local } =
                            cs.locator_by_constraint[global_idx]
                        else {
                            continue;
                        };
                        let c_mul = local;
                        let src4 = match entry.poly_type {
                            PolyType::MulLhs => &cs.gt_mul_rows[local].lhs,
                            PolyType::MulRhs => &cs.gt_mul_rows[local].rhs,
                            PolyType::MulResult => &cs.gt_mul_rows[local].result,
                            PolyType::MulQuotient => &cs.gt_mul_rows[local].quotient,
                            _ => unreachable!(),
                        };
                        debug_assert_eq!(src4.len(), row_size);
                        let off = c_mul << 4;
                        fused_src[off..off + row_size].copy_from_slice(src4);
                    }
                }
                _ => {
                    // Other poly types are not GT-fused.
                }
            }

            fill_block(dst, &fused_src, entry.num_vars);
            return;
        }

        if entry.is_g1_scalar_mul_fused {
            // Fused G1 scalar-mul rows are native 8-var step traces plus a family-local padded `c`.
            let num_g1 = cs.g1_scalar_mul_rows.len();
            let padded = num_g1.max(1).next_power_of_two();
            let k = padded.trailing_zeros() as usize;
            let expected_num_vars = 8usize + k;
            if entry.num_vars != expected_num_vars {
                panic!(
                    "G1-scalar-mul-fused entry has num_vars={}, expected {} (8 + k_g1)",
                    entry.num_vars, expected_num_vars
                );
            }

            let row_size = 1usize << 8;
            let mut fused_src = vec![Fq::zero(); 1usize << entry.num_vars];

            for global_idx in 0..cs.constraint_types.len() {
                let ConstraintLocator::G1ScalarMul { local } = cs.locator_by_constraint[global_idx]
                else {
                    continue;
                };
                let c = local;
                let src8 = match entry.poly_type {
                    PolyType::G1ScalarMulXA => &cs.g1_scalar_mul_rows[local].x_a,
                    PolyType::G1ScalarMulYA => &cs.g1_scalar_mul_rows[local].y_a,
                    PolyType::G1ScalarMulXT => &cs.g1_scalar_mul_rows[local].x_t,
                    PolyType::G1ScalarMulYT => &cs.g1_scalar_mul_rows[local].y_t,
                    PolyType::G1ScalarMulXANext => &cs.g1_scalar_mul_rows[local].x_a_next,
                    PolyType::G1ScalarMulYANext => &cs.g1_scalar_mul_rows[local].y_a_next,
                    PolyType::G1ScalarMulTIndicator => &cs.g1_scalar_mul_rows[local].t_indicator,
                    PolyType::G1ScalarMulAIndicator => &cs.g1_scalar_mul_rows[local].a_indicator,
                    _ => continue,
                };
                debug_assert_eq!(src8.len(), row_size);
                let off = c << 8;
                fused_src[off..off + row_size].copy_from_slice(src8);
            }

            fill_block(dst, &fused_src, entry.num_vars);
            return;
        }

        if entry.is_g1_add_fused {
            // Fused G1 add rows are c-only over a family-local padded `c_add`.
            let num_g1 = cs.g1_add_rows.len();
            let padded = num_g1.max(1).next_power_of_two();
            let k = padded.trailing_zeros() as usize;
            let expected_num_vars = k;
            if entry.num_vars != expected_num_vars {
                panic!(
                    "G1-add-fused entry has num_vars={}, expected {} (k_add)",
                    entry.num_vars, expected_num_vars
                );
            }

            let mut fused_src = vec![Fq::zero(); 1usize << entry.num_vars];
            for global_idx in 0..cs.constraint_types.len() {
                let ConstraintLocator::G1Add { local } = cs.locator_by_constraint[global_idx]
                else {
                    continue;
                };
                let c = local;
                let v = match entry.poly_type {
                    PolyType::G1AddXP => cs.g1_add_rows[local].x_p,
                    PolyType::G1AddYP => cs.g1_add_rows[local].y_p,
                    PolyType::G1AddPIndicator => cs.g1_add_rows[local].ind_p,
                    PolyType::G1AddXQ => cs.g1_add_rows[local].x_q,
                    PolyType::G1AddYQ => cs.g1_add_rows[local].y_q,
                    PolyType::G1AddQIndicator => cs.g1_add_rows[local].ind_q,
                    PolyType::G1AddXR => cs.g1_add_rows[local].x_r,
                    PolyType::G1AddYR => cs.g1_add_rows[local].y_r,
                    PolyType::G1AddRIndicator => cs.g1_add_rows[local].ind_r,
                    PolyType::G1AddLambda => cs.g1_add_rows[local].lambda,
                    PolyType::G1AddInvDeltaX => cs.g1_add_rows[local].inv_delta_x,
                    PolyType::G1AddIsDouble => cs.g1_add_rows[local].is_double,
                    PolyType::G1AddIsInverse => cs.g1_add_rows[local].is_inverse,
                    _ => continue,
                };
                fused_src[c] = v;
            }

            fill_block(dst, &fused_src, entry.num_vars);
            return;
        }

        if entry.is_g2_scalar_mul_fused {
            // Fused G2 scalar-mul rows are native 8-var step traces plus a family-local padded `c`.
            let num_g2 = cs.g2_scalar_mul_rows.len();
            let padded = num_g2.max(1).next_power_of_two();
            let k = padded.trailing_zeros() as usize;
            let expected_num_vars = 8usize + k;
            if entry.num_vars != expected_num_vars {
                panic!(
                    "G2-scalar-mul-fused entry has num_vars={}, expected {} (8 + k_g2)",
                    entry.num_vars, expected_num_vars
                );
            }

            let row_size = 1usize << 8;
            let mut fused_src = vec![Fq::zero(); 1usize << entry.num_vars];

            for global_idx in 0..cs.constraint_types.len() {
                let ConstraintLocator::G2ScalarMul { local } = cs.locator_by_constraint[global_idx]
                else {
                    continue;
                };
                let c = local;
                let src8 = match entry.poly_type {
                    PolyType::G2ScalarMulXAC0 => &cs.g2_scalar_mul_rows[local].x_a_c0,
                    PolyType::G2ScalarMulXAC1 => &cs.g2_scalar_mul_rows[local].x_a_c1,
                    PolyType::G2ScalarMulYAC0 => &cs.g2_scalar_mul_rows[local].y_a_c0,
                    PolyType::G2ScalarMulYAC1 => &cs.g2_scalar_mul_rows[local].y_a_c1,
                    PolyType::G2ScalarMulXTC0 => &cs.g2_scalar_mul_rows[local].x_t_c0,
                    PolyType::G2ScalarMulXTC1 => &cs.g2_scalar_mul_rows[local].x_t_c1,
                    PolyType::G2ScalarMulYTC0 => &cs.g2_scalar_mul_rows[local].y_t_c0,
                    PolyType::G2ScalarMulYTC1 => &cs.g2_scalar_mul_rows[local].y_t_c1,
                    PolyType::G2ScalarMulXANextC0 => &cs.g2_scalar_mul_rows[local].x_a_next_c0,
                    PolyType::G2ScalarMulXANextC1 => &cs.g2_scalar_mul_rows[local].x_a_next_c1,
                    PolyType::G2ScalarMulYANextC0 => &cs.g2_scalar_mul_rows[local].y_a_next_c0,
                    PolyType::G2ScalarMulYANextC1 => &cs.g2_scalar_mul_rows[local].y_a_next_c1,
                    PolyType::G2ScalarMulTIndicator => &cs.g2_scalar_mul_rows[local].t_indicator,
                    PolyType::G2ScalarMulAIndicator => &cs.g2_scalar_mul_rows[local].a_indicator,
                    _ => continue,
                };
                debug_assert_eq!(src8.len(), row_size);
                let off = c << 8;
                fused_src[off..off + row_size].copy_from_slice(src8);
            }

            fill_block(dst, &fused_src, entry.num_vars);
            return;
        }

        if entry.is_g2_add_fused {
            // Fused G2 add rows are c-only over a family-local padded `c_add`.
            let num_g2 = cs.g2_add_rows.len();
            let padded = num_g2.max(1).next_power_of_two();
            let k = padded.trailing_zeros() as usize;
            let expected_num_vars = k;
            if entry.num_vars != expected_num_vars {
                panic!(
                    "G2-add-fused entry has num_vars={}, expected {} (k_add)",
                    entry.num_vars, expected_num_vars
                );
            }

            let mut fused_src = vec![Fq::zero(); 1usize << entry.num_vars];
            for global_idx in 0..cs.constraint_types.len() {
                let ConstraintLocator::G2Add { local } = cs.locator_by_constraint[global_idx]
                else {
                    continue;
                };
                let c = local;
                let v = match entry.poly_type {
                    PolyType::G2AddXPC0 => cs.g2_add_rows[local].x_p_c0,
                    PolyType::G2AddXPC1 => cs.g2_add_rows[local].x_p_c1,
                    PolyType::G2AddYPC0 => cs.g2_add_rows[local].y_p_c0,
                    PolyType::G2AddYPC1 => cs.g2_add_rows[local].y_p_c1,
                    PolyType::G2AddPIndicator => cs.g2_add_rows[local].ind_p,
                    PolyType::G2AddXQC0 => cs.g2_add_rows[local].x_q_c0,
                    PolyType::G2AddXQC1 => cs.g2_add_rows[local].x_q_c1,
                    PolyType::G2AddYQC0 => cs.g2_add_rows[local].y_q_c0,
                    PolyType::G2AddYQC1 => cs.g2_add_rows[local].y_q_c1,
                    PolyType::G2AddQIndicator => cs.g2_add_rows[local].ind_q,
                    PolyType::G2AddXRC0 => cs.g2_add_rows[local].x_r_c0,
                    PolyType::G2AddXRC1 => cs.g2_add_rows[local].x_r_c1,
                    PolyType::G2AddYRC0 => cs.g2_add_rows[local].y_r_c0,
                    PolyType::G2AddYRC1 => cs.g2_add_rows[local].y_r_c1,
                    PolyType::G2AddRIndicator => cs.g2_add_rows[local].ind_r,
                    PolyType::G2AddLambdaC0 => cs.g2_add_rows[local].lambda_c0,
                    PolyType::G2AddLambdaC1 => cs.g2_add_rows[local].lambda_c1,
                    PolyType::G2AddInvDeltaXC0 => cs.g2_add_rows[local].inv_delta_x_c0,
                    PolyType::G2AddInvDeltaXC1 => cs.g2_add_rows[local].inv_delta_x_c1,
                    PolyType::G2AddIsDouble => cs.g2_add_rows[local].is_double,
                    PolyType::G2AddIsInverse => cs.g2_add_rows[local].is_inverse,
                    _ => continue,
                };
                fused_src[c] = v;
            }

            fill_block(dst, &fused_src, entry.num_vars);
            return;
        }

        let loc = cs.locator_by_constraint[entry.constraint_idx];
        match entry.poly_type {
            // Packed GT exp (11-var)
            PolyType::RhoPrev => {
                let ConstraintLocator::GtExp { local } = loc else {
                    panic!("RhoPrev entry with non-GtExp locator: {loc:?}");
                };
                fill_block(dst, &cs.gt_exp_witnesses[local].rho_packed, entry.num_vars);
            }
            PolyType::Quotient => {
                let ConstraintLocator::GtExp { local } = loc else {
                    panic!("Quotient entry with non-GtExp locator: {loc:?}");
                };
                fill_block(
                    dst,
                    &cs.gt_exp_witnesses[local].quotient_packed,
                    entry.num_vars,
                );
            }

            // GT mul (4-var)
            PolyType::MulLhs => {
                let ConstraintLocator::GtMul { local } = loc else {
                    panic!("MulLhs entry with non-GtMul locator: {loc:?}");
                };
                fill_block(dst, &cs.gt_mul_rows[local].lhs, entry.num_vars);
            }
            PolyType::MulRhs => {
                let ConstraintLocator::GtMul { local } = loc else {
                    panic!("MulRhs entry with non-GtMul locator: {loc:?}");
                };
                fill_block(dst, &cs.gt_mul_rows[local].rhs, entry.num_vars);
            }
            PolyType::MulResult => {
                let ConstraintLocator::GtMul { local } = loc else {
                    panic!("MulResult entry with non-GtMul locator: {loc:?}");
                };
                fill_block(dst, &cs.gt_mul_rows[local].result, entry.num_vars);
            }
            PolyType::MulQuotient => {
                let ConstraintLocator::GtMul { local } = loc else {
                    panic!("MulQuotient entry with non-GtMul locator: {loc:?}");
                };
                fill_block(dst, &cs.gt_mul_rows[local].quotient, entry.num_vars);
            }

            // G1 scalar mul (8-var)
            PolyType::G1ScalarMulXA => {
                let ConstraintLocator::G1ScalarMul { local } = loc else {
                    panic!("G1ScalarMulXA entry with non-G1ScalarMul locator: {loc:?}");
                };
                fill_block(dst, &cs.g1_scalar_mul_rows[local].x_a, entry.num_vars);
            }
            PolyType::G1ScalarMulYA => {
                let ConstraintLocator::G1ScalarMul { local } = loc else {
                    panic!("G1ScalarMulYA entry with non-G1ScalarMul locator: {loc:?}");
                };
                fill_block(dst, &cs.g1_scalar_mul_rows[local].y_a, entry.num_vars);
            }
            PolyType::G1ScalarMulXT => {
                let ConstraintLocator::G1ScalarMul { local } = loc else {
                    panic!("G1ScalarMulXT entry with non-G1ScalarMul locator: {loc:?}");
                };
                fill_block(dst, &cs.g1_scalar_mul_rows[local].x_t, entry.num_vars);
            }
            PolyType::G1ScalarMulYT => {
                let ConstraintLocator::G1ScalarMul { local } = loc else {
                    panic!("G1ScalarMulYT entry with non-G1ScalarMul locator: {loc:?}");
                };
                fill_block(dst, &cs.g1_scalar_mul_rows[local].y_t, entry.num_vars);
            }
            PolyType::G1ScalarMulXANext => {
                let ConstraintLocator::G1ScalarMul { local } = loc else {
                    panic!("G1ScalarMulXANext entry with non-G1ScalarMul locator: {loc:?}");
                };
                fill_block(dst, &cs.g1_scalar_mul_rows[local].x_a_next, entry.num_vars);
            }
            PolyType::G1ScalarMulYANext => {
                let ConstraintLocator::G1ScalarMul { local } = loc else {
                    panic!("G1ScalarMulYANext entry with non-G1ScalarMul locator: {loc:?}");
                };
                fill_block(dst, &cs.g1_scalar_mul_rows[local].y_a_next, entry.num_vars);
            }
            PolyType::G1ScalarMulTIndicator => {
                let ConstraintLocator::G1ScalarMul { local } = loc else {
                    panic!("G1ScalarMulTIndicator entry with non-G1ScalarMul locator: {loc:?}");
                };
                fill_block(
                    dst,
                    &cs.g1_scalar_mul_rows[local].t_indicator,
                    entry.num_vars,
                );
            }
            PolyType::G1ScalarMulAIndicator => {
                let ConstraintLocator::G1ScalarMul { local } = loc else {
                    panic!("G1ScalarMulAIndicator entry with non-G1ScalarMul locator: {loc:?}");
                };
                fill_block(
                    dst,
                    &cs.g1_scalar_mul_rows[local].a_indicator,
                    entry.num_vars,
                );
            }

            // G2 scalar mul (8-var)
            PolyType::G2ScalarMulXAC0 => {
                let ConstraintLocator::G2ScalarMul { local } = loc else {
                    panic!("G2ScalarMulXAC0 entry with non-G2ScalarMul locator: {loc:?}");
                };
                fill_block(dst, &cs.g2_scalar_mul_rows[local].x_a_c0, entry.num_vars);
            }
            PolyType::G2ScalarMulXAC1 => {
                let ConstraintLocator::G2ScalarMul { local } = loc else {
                    panic!("G2ScalarMulXAC1 entry with non-G2ScalarMul locator: {loc:?}");
                };
                fill_block(dst, &cs.g2_scalar_mul_rows[local].x_a_c1, entry.num_vars);
            }
            PolyType::G2ScalarMulYAC0 => {
                let ConstraintLocator::G2ScalarMul { local } = loc else {
                    panic!("G2ScalarMulYAC0 entry with non-G2ScalarMul locator: {loc:?}");
                };
                fill_block(dst, &cs.g2_scalar_mul_rows[local].y_a_c0, entry.num_vars);
            }
            PolyType::G2ScalarMulYAC1 => {
                let ConstraintLocator::G2ScalarMul { local } = loc else {
                    panic!("G2ScalarMulYAC1 entry with non-G2ScalarMul locator: {loc:?}");
                };
                fill_block(dst, &cs.g2_scalar_mul_rows[local].y_a_c1, entry.num_vars);
            }
            PolyType::G2ScalarMulXTC0 => {
                let ConstraintLocator::G2ScalarMul { local } = loc else {
                    panic!("G2ScalarMulXTC0 entry with non-G2ScalarMul locator: {loc:?}");
                };
                fill_block(dst, &cs.g2_scalar_mul_rows[local].x_t_c0, entry.num_vars);
            }
            PolyType::G2ScalarMulXTC1 => {
                let ConstraintLocator::G2ScalarMul { local } = loc else {
                    panic!("G2ScalarMulXTC1 entry with non-G2ScalarMul locator: {loc:?}");
                };
                fill_block(dst, &cs.g2_scalar_mul_rows[local].x_t_c1, entry.num_vars);
            }
            PolyType::G2ScalarMulYTC0 => {
                let ConstraintLocator::G2ScalarMul { local } = loc else {
                    panic!("G2ScalarMulYTC0 entry with non-G2ScalarMul locator: {loc:?}");
                };
                fill_block(dst, &cs.g2_scalar_mul_rows[local].y_t_c0, entry.num_vars);
            }
            PolyType::G2ScalarMulYTC1 => {
                let ConstraintLocator::G2ScalarMul { local } = loc else {
                    panic!("G2ScalarMulYTC1 entry with non-G2ScalarMul locator: {loc:?}");
                };
                fill_block(dst, &cs.g2_scalar_mul_rows[local].y_t_c1, entry.num_vars);
            }
            PolyType::G2ScalarMulXANextC0 => {
                let ConstraintLocator::G2ScalarMul { local } = loc else {
                    panic!("G2ScalarMulXANextC0 entry with non-G2ScalarMul locator: {loc:?}");
                };
                fill_block(
                    dst,
                    &cs.g2_scalar_mul_rows[local].x_a_next_c0,
                    entry.num_vars,
                );
            }
            PolyType::G2ScalarMulXANextC1 => {
                let ConstraintLocator::G2ScalarMul { local } = loc else {
                    panic!("G2ScalarMulXANextC1 entry with non-G2ScalarMul locator: {loc:?}");
                };
                fill_block(
                    dst,
                    &cs.g2_scalar_mul_rows[local].x_a_next_c1,
                    entry.num_vars,
                );
            }
            PolyType::G2ScalarMulYANextC0 => {
                let ConstraintLocator::G2ScalarMul { local } = loc else {
                    panic!("G2ScalarMulYANextC0 entry with non-G2ScalarMul locator: {loc:?}");
                };
                fill_block(
                    dst,
                    &cs.g2_scalar_mul_rows[local].y_a_next_c0,
                    entry.num_vars,
                );
            }
            PolyType::G2ScalarMulYANextC1 => {
                let ConstraintLocator::G2ScalarMul { local } = loc else {
                    panic!("G2ScalarMulYANextC1 entry with non-G2ScalarMul locator: {loc:?}");
                };
                fill_block(
                    dst,
                    &cs.g2_scalar_mul_rows[local].y_a_next_c1,
                    entry.num_vars,
                );
            }
            PolyType::G2ScalarMulTIndicator => {
                let ConstraintLocator::G2ScalarMul { local } = loc else {
                    panic!("G2ScalarMulTIndicator entry with non-G2ScalarMul locator: {loc:?}");
                };
                fill_block(
                    dst,
                    &cs.g2_scalar_mul_rows[local].t_indicator,
                    entry.num_vars,
                );
            }
            PolyType::G2ScalarMulAIndicator => {
                let ConstraintLocator::G2ScalarMul { local } = loc else {
                    panic!("G2ScalarMulAIndicator entry with non-G2ScalarMul locator: {loc:?}");
                };
                fill_block(
                    dst,
                    &cs.g2_scalar_mul_rows[local].a_indicator,
                    entry.num_vars,
                );
            }

            // G1 add (0-var)
            PolyType::G1AddXP => {
                let ConstraintLocator::G1Add { local } = loc else {
                    panic!("G1AddXP entry with non-G1Add locator: {loc:?}");
                };
                dst[0] = cs.g1_add_rows[local].x_p;
            }
            PolyType::G1AddYP => {
                let ConstraintLocator::G1Add { local } = loc else {
                    panic!("G1AddYP entry with non-G1Add locator: {loc:?}");
                };
                dst[0] = cs.g1_add_rows[local].y_p;
            }
            PolyType::G1AddPIndicator => {
                let ConstraintLocator::G1Add { local } = loc else {
                    panic!("G1AddPIndicator entry with non-G1Add locator: {loc:?}");
                };
                dst[0] = cs.g1_add_rows[local].ind_p;
            }
            PolyType::G1AddXQ => {
                let ConstraintLocator::G1Add { local } = loc else {
                    panic!("G1AddXQ entry with non-G1Add locator: {loc:?}");
                };
                dst[0] = cs.g1_add_rows[local].x_q;
            }
            PolyType::G1AddYQ => {
                let ConstraintLocator::G1Add { local } = loc else {
                    panic!("G1AddYQ entry with non-G1Add locator: {loc:?}");
                };
                dst[0] = cs.g1_add_rows[local].y_q;
            }
            PolyType::G1AddQIndicator => {
                let ConstraintLocator::G1Add { local } = loc else {
                    panic!("G1AddQIndicator entry with non-G1Add locator: {loc:?}");
                };
                dst[0] = cs.g1_add_rows[local].ind_q;
            }
            PolyType::G1AddXR => {
                let ConstraintLocator::G1Add { local } = loc else {
                    panic!("G1AddXR entry with non-G1Add locator: {loc:?}");
                };
                dst[0] = cs.g1_add_rows[local].x_r;
            }
            PolyType::G1AddYR => {
                let ConstraintLocator::G1Add { local } = loc else {
                    panic!("G1AddYR entry with non-G1Add locator: {loc:?}");
                };
                dst[0] = cs.g1_add_rows[local].y_r;
            }
            PolyType::G1AddRIndicator => {
                let ConstraintLocator::G1Add { local } = loc else {
                    panic!("G1AddRIndicator entry with non-G1Add locator: {loc:?}");
                };
                dst[0] = cs.g1_add_rows[local].ind_r;
            }
            PolyType::G1AddLambda => {
                let ConstraintLocator::G1Add { local } = loc else {
                    panic!("G1AddLambda entry with non-G1Add locator: {loc:?}");
                };
                dst[0] = cs.g1_add_rows[local].lambda;
            }
            PolyType::G1AddInvDeltaX => {
                let ConstraintLocator::G1Add { local } = loc else {
                    panic!("G1AddInvDeltaX entry with non-G1Add locator: {loc:?}");
                };
                dst[0] = cs.g1_add_rows[local].inv_delta_x;
            }
            PolyType::G1AddIsDouble => {
                let ConstraintLocator::G1Add { local } = loc else {
                    panic!("G1AddIsDouble entry with non-G1Add locator: {loc:?}");
                };
                dst[0] = cs.g1_add_rows[local].is_double;
            }
            PolyType::G1AddIsInverse => {
                let ConstraintLocator::G1Add { local } = loc else {
                    panic!("G1AddIsInverse entry with non-G1Add locator: {loc:?}");
                };
                dst[0] = cs.g1_add_rows[local].is_inverse;
            }

            // G2 add (0-var)
            PolyType::G2AddXPC0 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddXPC0 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].x_p_c0;
            }
            PolyType::G2AddXPC1 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddXPC1 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].x_p_c1;
            }
            PolyType::G2AddYPC0 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddYPC0 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].y_p_c0;
            }
            PolyType::G2AddYPC1 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddYPC1 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].y_p_c1;
            }
            PolyType::G2AddPIndicator => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddPIndicator entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].ind_p;
            }
            PolyType::G2AddXQC0 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddXQC0 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].x_q_c0;
            }
            PolyType::G2AddXQC1 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddXQC1 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].x_q_c1;
            }
            PolyType::G2AddYQC0 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddYQC0 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].y_q_c0;
            }
            PolyType::G2AddYQC1 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddYQC1 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].y_q_c1;
            }
            PolyType::G2AddQIndicator => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddQIndicator entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].ind_q;
            }
            PolyType::G2AddXRC0 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddXRC0 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].x_r_c0;
            }
            PolyType::G2AddXRC1 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddXRC1 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].x_r_c1;
            }
            PolyType::G2AddYRC0 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddYRC0 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].y_r_c0;
            }
            PolyType::G2AddYRC1 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddYRC1 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].y_r_c1;
            }
            PolyType::G2AddRIndicator => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddRIndicator entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].ind_r;
            }
            PolyType::G2AddLambdaC0 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddLambdaC0 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].lambda_c0;
            }
            PolyType::G2AddLambdaC1 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddLambdaC1 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].lambda_c1;
            }
            PolyType::G2AddInvDeltaXC0 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddInvDeltaXC0 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].inv_delta_x_c0;
            }
            PolyType::G2AddInvDeltaXC1 => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddInvDeltaXC1 entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].inv_delta_x_c1;
            }
            PolyType::G2AddIsDouble => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddIsDouble entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].is_double;
            }
            PolyType::G2AddIsInverse => {
                let ConstraintLocator::G2Add { local } = loc else {
                    panic!("G2AddIsInverse entry with non-G2Add locator: {loc:?}");
                };
                dst[0] = cs.g2_add_rows[local].is_inverse;
            }

            // Not part of any `ConstraintType::committed_poly_specs()` today.
            _ => panic!(
                "Unexpected prefix-packing PolyType entry: {:?}",
                entry.poly_type
            ),
        }
    }

    #[tracing::instrument(
        skip_all,
        name = "recursion.witness_gen.emit.build_layout",
        fields(num_constraints = cs.constraint_types.len())
    )]
    fn build_layout(cs: &ConstraintSystem) -> PrefixPackingLayout {
        let enable_gt_fused_end_to_end = std::env::var("JOLT_RECURSION_ENABLE_GT_FUSED_END_TO_END")
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false);
        let enable_g1_fused_wiring_end_to_end =
            std::env::var("JOLT_RECURSION_ENABLE_G1_FUSED_WIRING_END_TO_END")
                .ok()
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(false);
        let enable_g1_scalar_mul_fused_end_to_end =
            std::env::var("JOLT_RECURSION_ENABLE_G1_SCALAR_MUL_FUSED_END_TO_END")
                .ok()
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(false)
                || enable_g1_fused_wiring_end_to_end;
        // Full fused G1 wiring requires fused G1Add packing as well.
        let enable_g1_add_fused_end_to_end = enable_g1_fused_wiring_end_to_end;

        let enable_g2_fused_wiring_end_to_end =
            std::env::var("JOLT_RECURSION_ENABLE_G2_FUSED_WIRING_END_TO_END")
                .ok()
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(false);
        let enable_g2_scalar_mul_fused_end_to_end =
            std::env::var("JOLT_RECURSION_ENABLE_G2_SCALAR_MUL_FUSED_END_TO_END")
                .ok()
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(false)
                || enable_g2_fused_wiring_end_to_end;
        // Full fused G2 wiring requires fused G2Add packing as well.
        let enable_g2_add_fused_end_to_end = enable_g2_fused_wiring_end_to_end;

        if enable_gt_fused_end_to_end
            || enable_g1_scalar_mul_fused_end_to_end
            || enable_g1_add_fused_end_to_end
            || enable_g2_scalar_mul_fused_end_to_end
            || enable_g2_add_fused_end_to_end
        {
            PrefixPackingLayout::from_constraint_types_fused(
                &cs.constraint_types,
                enable_gt_fused_end_to_end,
                enable_g1_scalar_mul_fused_end_to_end,
                enable_g1_add_fused_end_to_end,
                enable_g2_scalar_mul_fused_end_to_end,
                enable_g2_add_fused_end_to_end,
            )
        } else {
            PrefixPackingLayout::from_constraint_types(&cs.constraint_types)
        }
    }

    #[tracing::instrument(
        skip_all,
        name = "recursion.witness_gen.emit.fill_entries",
        fields(num_entries = layout.entries.len(), packed_size = layout.packed_size())
    )]
    fn fill_entries(
        cs: &ConstraintSystem,
        layout: &PrefixPackingLayout,
        fill_entry: fn(&mut [Fq], &ConstraintSystem, &PrefixPackedEntry),
    ) -> Vec<Fq> {
        let mut packed = vec![Fq::zero(); layout.packed_size()];
        for entry in &layout.entries {
            let native_size = 1usize << entry.num_vars;
            let dst = &mut packed[entry.offset..entry.offset + native_size];
            fill_entry(dst, cs, entry);
        }
        packed
    }

    let layout = build_layout(cs);
    let packed = fill_entries(cs, &layout, fill_entry);
    (DensePolynomial::new(packed), layout)
}
