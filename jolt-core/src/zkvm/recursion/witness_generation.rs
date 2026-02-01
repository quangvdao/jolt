//! Streaming recursion witness generation (plan + emit).
//!
//! This module implements the two-pass pipeline:
//! - **PlanPass**: derive the public constraint ordering + build native witness stores.
//! - **EmitPass**: derive the prefix-packing layout and emit the packed dense evaluation table.

use ark_bn254::{Fq, Fq12};
use ark_ff::{Field, Zero};
use dory::recursion::{OpId, WitnessCollection};
use jolt_optimizations::fq12_to_multilinear_evals;
use rayon::prelude::*;

use crate::poly::commitment::dory::recursion::{JoltGtExpWitness, JoltGtMulWitness, JoltWitness};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::zkvm::recursion::constraints::system::{
    ConstraintLocator, ConstraintSystem, ConstraintType, G1AddNative, G1ScalarMulNative,
    G2AddNative, G2ScalarMulNative, GtMulNativeRows, PolyType, RecursionMatrixShape,
};
use crate::zkvm::recursion::g1::types::G1ScalarMulPublicInputs;
use crate::zkvm::recursion::g2::types::G2ScalarMulPublicInputs;
use crate::zkvm::recursion::gt::indexing::{k_exp, k_mul};
use crate::zkvm::recursion::gt::types::{GtExpPublicInputs, GtExpWitness};
use crate::zkvm::recursion::prefix_packing::{PrefixPackedEntry, PrefixPackingLayout};
use crate::zkvm::recursion::witness::{GTCombineWitness, GTExpOpWitness, GTMulOpWitness};

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

fn pack_gt_exp_op_witness(exp_wit: GTExpOpWitness) -> (GtExpWitness<Fq>, GtExpPublicInputs, Fq12) {
    // Matches the `DoryMatrixBuilder::add_combine_witness` packing logic.
    let GTExpOpWitness {
        base,
        result,
        rho_mles,
        quotient_mles,
        bits,
        ..
    } = exp_wit;

    // Move scalar bits exactly once: keep them in the public inputs, and borrow for witness packing.
    let public_input = GtExpPublicInputs::new(bits);

    let base_mle = fq12_to_multilinear_evals(&base);
    let base2 = base * base;
    let base2_mle = fq12_to_multilinear_evals(&base2);
    let base3 = base2 * base;
    let base3_mle = fq12_to_multilinear_evals(&base3);

    let num_steps = public_input.scalar_bits.len().div_ceil(2);
    if public_input.scalar_bits.is_empty() {
        // Degenerate exponentiation: treat as one rho row (the result) and no quotient rows.
        let rho_mles = if rho_mles.is_empty() {
            vec![fq12_to_multilinear_evals(&result)]
        } else {
            rho_mles
        };
        let packed = GtExpWitness::from_steps(
            &rho_mles,
            &[],
            &public_input.scalar_bits,
            &base_mle,
            &base2_mle,
            &base3_mle,
        );
        return (packed, public_input, base);
    }

    let mut fixed_quotients = quotient_mles;
    if fixed_quotients.len() != num_steps {
        fixed_quotients.resize(num_steps, vec![Fq::zero(); 16]);
    }

    let mut fixed_rhos = rho_mles;
    if fixed_rhos.len() < num_steps + 1 {
        let result_mle = fq12_to_multilinear_evals(&result);
        while fixed_rhos.len() < num_steps + 1 {
            fixed_rhos.push(result_mle.clone());
        }
    }

    let packed = GtExpWitness::from_steps(
        &fixed_rhos,
        &fixed_quotients,
        &public_input.scalar_bits,
        &base_mle,
        &base2_mle,
        &base3_mle,
    );
    (packed, public_input, base)
}

fn gt_mul_rows_from_op_witness(w: GTMulOpWitness) -> Option<GtMulNativeRows> {
    // Matches the `DoryMatrixBuilder::add_gt_mul_op_witness` guard.
    if w.quotient_mle.is_empty() {
        return None;
    }
    let lhs = fq12_to_multilinear_evals(&w.lhs);
    let rhs = fq12_to_multilinear_evals(&w.rhs);
    let result = fq12_to_multilinear_evals(&w.result);
    let quotient = w.quotient_mle;
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
    name = "recursion.witness_gen.plan.gt_exp",
    fields(num_ops = gt_exp_items.len())
)]
fn plan_gt_exp(
    mut gt_exp_items: Vec<(OpId, JoltGtExpWitness)>,
) -> Vec<(GtExpWitness<Fq>, GtExpPublicInputs, Fq12)> {
    gt_exp_items.sort_by_key(|(op_id, _)| *op_id);

    gt_exp_items
        .into_par_iter()
        .map(|(_op_id, witness)| {
            let JoltGtExpWitness {
                base,
                bits,
                rho_mles,
                quotient_mles,
                ..
            } = witness;

            let public_input = GtExpPublicInputs::new(bits);

            let base_mle = fq12_to_multilinear_evals(&base);
            let base2 = base * base;
            let base2_mle = fq12_to_multilinear_evals(&base2);
            let base3 = base2 * base;
            let base3_mle = fq12_to_multilinear_evals(&base3);

            let packed = GtExpWitness::from_steps(
                &rho_mles,
                &quotient_mles,
                &public_input.scalar_bits,
                &base_mle,
                &base2_mle,
                &base3_mle,
            );
            (packed, public_input, base)
        })
        .collect()
}

#[tracing::instrument(
    skip_all,
    name = "recursion.witness_gen.plan.gt_mul",
    fields(num_ops = gt_mul_items.len())
)]
fn plan_gt_mul(mut gt_mul_items: Vec<(OpId, JoltGtMulWitness)>) -> Vec<GtMulNativeRows> {
    gt_mul_items.sort_by_key(|(op_id, _)| *op_id);

    gt_mul_items
        .into_par_iter()
        .map(|(_op_id, witness)| {
            let lhs = fq12_to_multilinear_evals(&witness.lhs);
            let rhs = fq12_to_multilinear_evals(&witness.rhs);
            let result = fq12_to_multilinear_evals(&witness.result);
            let quotient = witness.quotient_mle;
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
    name = "recursion.witness_gen.plan.combine_witness",
    fields(num_exp = cw.exp_witnesses.len(), num_mul_layers = cw.mul_layers.len())
)]
fn plan_combine_witness(
    cw: GTCombineWitness,
    constraint_types: &mut Vec<ConstraintType>,
    locator_by_constraint: &mut Vec<ConstraintLocator>,
    gt_exp_witnesses: &mut Vec<GtExpWitness<Fq>>,
    gt_exp_public_inputs: &mut Vec<GtExpPublicInputs>,
    gt_exp_base_inputs: &mut Vec<Fq12>,
    gt_mul_rows: &mut Vec<GtMulNativeRows>,
) {
    // Append GT exp constraints for combine terms.
    for exp_wit in cw.exp_witnesses {
        let (packed, public_input, base) = pack_gt_exp_op_witness(exp_wit);
        let local = gt_exp_witnesses.len();
        gt_exp_witnesses.push(packed);
        gt_exp_public_inputs.push(public_input);
        gt_exp_base_inputs.push(base);
        constraint_types.push(ConstraintType::GtExp);
        locator_by_constraint.push(ConstraintLocator::GtExp { local });
    }

    // Append GT mul constraints for combine reduction tree.
    for layer in cw.mul_layers {
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
    witness_collection: WitnessCollection<JoltWitness>,
    combine_witness: Option<GTCombineWitness>,
    g_poly_4var: DensePolynomial<Fq>,
) -> Result<ConstraintSystem, Box<dyn std::error::Error>> {
    // Always include all constraint families present in the witness collection (plus combine
    // constraints when provided).

    // Outputs
    let mut constraint_types: Vec<ConstraintType> = Vec::new();
    let mut locator_by_constraint: Vec<ConstraintLocator> = Vec::new();

    let mut gt_exp_witnesses: Vec<GtExpWitness<Fq>> = Vec::new();
    let mut gt_exp_public_inputs: Vec<GtExpPublicInputs> = Vec::new();
    let mut gt_exp_base_inputs: Vec<Fq12> = Vec::new();

    let mut gt_mul_rows: Vec<GtMulNativeRows> = Vec::new();
    let mut g1_scalar_mul_rows: Vec<G1ScalarMulNative> = Vec::new();
    let mut g2_scalar_mul_rows: Vec<G2ScalarMulNative> = Vec::new();
    let mut g1_add_rows: Vec<G1AddNative> = Vec::new();
    let mut g2_add_rows: Vec<G2AddNative> = Vec::new();

    let mut g1_scalar_mul_public_inputs: Vec<G1ScalarMulPublicInputs> = Vec::new();
    let mut g2_scalar_mul_public_inputs: Vec<G2ScalarMulPublicInputs> = Vec::new();

    // Move out the per-family witness maps (we’ll sort by OpId for determinism and move large
    // evaluation tables out without cloning).
    let WitnessCollection {
        gt_exp,
        gt_mul,
        g1_scalar_mul,
        g2_scalar_mul,
        g1_add,
        g2_add,
        ..
    } = witness_collection;

    // ---- GT exp ----
    let prepared_gt_exp = plan_gt_exp(gt_exp.into_iter().collect());

    for (packed, public_input, base) in prepared_gt_exp {
        let local = gt_exp_witnesses.len();
        gt_exp_witnesses.push(packed);
        gt_exp_public_inputs.push(public_input);
        gt_exp_base_inputs.push(base);
        constraint_types.push(ConstraintType::GtExp);
        locator_by_constraint.push(ConstraintLocator::GtExp { local });
    }

    // ---- GT mul ----
    {
        let prepared_gt_mul = plan_gt_mul(gt_mul.into_iter().collect());
        for rows in prepared_gt_mul {
            let local = gt_mul_rows.len();
            gt_mul_rows.push(rows);
            constraint_types.push(ConstraintType::GtMul);
            locator_by_constraint.push(ConstraintLocator::GtMul { local });
        }
    }

    // ---- G1 scalar mul ----
    {
        let mut g1_items: Vec<_> = g1_scalar_mul.into_iter().collect();
        g1_items.sort_by_key(|(op_id, _)| *op_id);
        for (_op_id, mut witness) in g1_items {
            let base_point = (witness.point_base.x, witness.point_base.y);
            let rows = G1ScalarMulNative {
                base_point,
                x_a: witness.x_a_mles.pop().expect("missing x_a MLE"),
                y_a: witness.y_a_mles.pop().expect("missing y_a MLE"),
                x_t: witness.x_t_mles.pop().expect("missing x_t MLE"),
                y_t: witness.y_t_mles.pop().expect("missing y_t MLE"),
                x_a_next: witness.x_a_next_mles.pop().expect("missing x_a_next MLE"),
                y_a_next: witness.y_a_next_mles.pop().expect("missing y_a_next MLE"),
                t_indicator: witness
                    .t_is_infinity_mles
                    .pop()
                    .expect("missing t_is_infinity MLE"),
                a_indicator: witness
                    .a_is_infinity_mles
                    .pop()
                    .expect("missing a_is_infinity MLE"),
            };
            debug_assert!(witness.x_a_mles.is_empty());
            debug_assert!(witness.y_a_mles.is_empty());
            debug_assert!(witness.x_t_mles.is_empty());
            debug_assert!(witness.y_t_mles.is_empty());
            debug_assert!(witness.x_a_next_mles.is_empty());
            debug_assert!(witness.y_a_next_mles.is_empty());
            debug_assert!(witness.t_is_infinity_mles.is_empty());
            debug_assert!(witness.a_is_infinity_mles.is_empty());

            let public_input = G1ScalarMulPublicInputs::new(witness.scalar);
            let local = g1_scalar_mul_rows.len();
            g1_scalar_mul_rows.push(rows);
            g1_scalar_mul_public_inputs.push(public_input);
            constraint_types.push(ConstraintType::G1ScalarMul { base_point });
            locator_by_constraint.push(ConstraintLocator::G1ScalarMul { local });
        }
    }

    // ---- G2 scalar mul ----
    {
        let mut g2_items: Vec<_> = g2_scalar_mul.into_iter().collect();
        g2_items.sort_by_key(|(op_id, _)| *op_id);
        for (_op_id, mut witness) in g2_items {
            let base_point = (witness.point_base.x, witness.point_base.y);
            let rows = G2ScalarMulNative {
                base_point,
                x_a_c0: witness.x_a_c0_mles.pop().expect("missing x_a_c0 MLE"),
                x_a_c1: witness.x_a_c1_mles.pop().expect("missing x_a_c1 MLE"),
                y_a_c0: witness.y_a_c0_mles.pop().expect("missing y_a_c0 MLE"),
                y_a_c1: witness.y_a_c1_mles.pop().expect("missing y_a_c1 MLE"),
                x_t_c0: witness.x_t_c0_mles.pop().expect("missing x_t_c0 MLE"),
                x_t_c1: witness.x_t_c1_mles.pop().expect("missing x_t_c1 MLE"),
                y_t_c0: witness.y_t_c0_mles.pop().expect("missing y_t_c0 MLE"),
                y_t_c1: witness.y_t_c1_mles.pop().expect("missing y_t_c1 MLE"),
                x_a_next_c0: witness
                    .x_a_next_c0_mles
                    .pop()
                    .expect("missing x_a_next_c0 MLE"),
                x_a_next_c1: witness
                    .x_a_next_c1_mles
                    .pop()
                    .expect("missing x_a_next_c1 MLE"),
                y_a_next_c0: witness
                    .y_a_next_c0_mles
                    .pop()
                    .expect("missing y_a_next_c0 MLE"),
                y_a_next_c1: witness
                    .y_a_next_c1_mles
                    .pop()
                    .expect("missing y_a_next_c1 MLE"),
                t_indicator: witness
                    .t_is_infinity_mles
                    .pop()
                    .expect("missing t_is_infinity MLE"),
                a_indicator: witness
                    .a_is_infinity_mles
                    .pop()
                    .expect("missing a_is_infinity MLE"),
            };
            debug_assert!(witness.x_a_c0_mles.is_empty());
            debug_assert!(witness.x_a_c1_mles.is_empty());
            debug_assert!(witness.y_a_c0_mles.is_empty());
            debug_assert!(witness.y_a_c1_mles.is_empty());
            debug_assert!(witness.x_t_c0_mles.is_empty());
            debug_assert!(witness.x_t_c1_mles.is_empty());
            debug_assert!(witness.y_t_c0_mles.is_empty());
            debug_assert!(witness.y_t_c1_mles.is_empty());
            debug_assert!(witness.x_a_next_c0_mles.is_empty());
            debug_assert!(witness.x_a_next_c1_mles.is_empty());
            debug_assert!(witness.y_a_next_c0_mles.is_empty());
            debug_assert!(witness.y_a_next_c1_mles.is_empty());
            debug_assert!(witness.t_is_infinity_mles.is_empty());
            debug_assert!(witness.a_is_infinity_mles.is_empty());

            let public_input = G2ScalarMulPublicInputs::new(witness.scalar);
            let local = g2_scalar_mul_rows.len();
            g2_scalar_mul_rows.push(rows);
            g2_scalar_mul_public_inputs.push(public_input);
            constraint_types.push(ConstraintType::G2ScalarMul { base_point });
            locator_by_constraint.push(ConstraintLocator::G2ScalarMul { local });
        }
    }

    // ---- G1 add ----
    {
        let mut g1_add_items: Vec<_> = g1_add.into_iter().collect();
        g1_add_items.sort_by_key(|(op_id, _)| *op_id);
        for (_op_id, witness) in g1_add_items {
            let rows = G1AddNative {
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
            };
            let local = g1_add_rows.len();
            g1_add_rows.push(rows);
            constraint_types.push(ConstraintType::G1Add);
            locator_by_constraint.push(ConstraintLocator::G1Add { local });
        }
    }

    // ---- G2 add ----
    {
        let mut g2_add_items: Vec<_> = g2_add.into_iter().collect();
        g2_add_items.sort_by_key(|(op_id, _)| *op_id);
        for (_op_id, witness) in g2_add_items {
            let rows = G2AddNative {
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
            };
            let local = g2_add_rows.len();
            g2_add_rows.push(rows);
            constraint_types.push(ConstraintType::G2Add);
            locator_by_constraint.push(ConstraintLocator::G2Add { local });
        }
    }

    // ---- Combine witness (homomorphic combine offloading) ----
    if let Some(cw) = combine_witness {
        plan_combine_witness(
            cw,
            &mut constraint_types,
            &mut locator_by_constraint,
            &mut gt_exp_witnesses,
            &mut gt_exp_public_inputs,
            &mut gt_exp_base_inputs,
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
        gt_exp_base_inputs,
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

    #[inline]
    fn fill_block_bit_reversed(
        dst: &mut [Fq],
        src: &[Fq],
        num_vars: usize,
        c: usize,
        row_bits: usize,
    ) {
        let row_size = 1usize << row_bits;
        debug_assert_eq!(src.len(), row_size);
        debug_assert_eq!(dst.len(), 1usize << num_vars);
        let off = c << row_bits;
        for u in 0..row_size {
            let src_idx = off + u;
            dst[bit_reverse(src_idx, num_vars)] = src[u];
        }
    }

    #[inline]
    fn fill_scalar_bit_reversed(dst: &mut [Fq], num_vars: usize, c: usize, v: Fq) {
        debug_assert_eq!(dst.len(), 1usize << num_vars);
        dst[bit_reverse(c, num_vars)] = v;
    }

    fn fill_entry(dst: &mut [Fq], cs: &ConstraintSystem, entry: &PrefixPackedEntry) {
        if entry.is_gt {
            // Commit exp/mul rows at family-local padded sizes.
            let num_vars_gt_exp = 11usize + k_exp(&cs.constraint_types);
            let num_vars_gt_exp_base = 4usize + k_exp(&cs.constraint_types);
            let num_vars_gt_mul = 4usize + k_mul(&cs.constraint_types);

            // IMPORTANT (no-padding GTMul + c-suffix):
            // - for GTExp rows, variables are (x11 low bits, c_gt high bits), size 2^(11+k)
            // - for GTMul rows, variables are (u low bits, c_gt high bits), size 2^(4+k)
            //
            // This avoids the old 4→11 replication for GTMul.
            match entry.poly_type {
                PolyType::RhoPrev | PolyType::Quotient => {
                    if entry.num_vars != num_vars_gt_exp {
                        panic!(
                            "GTExp entry has num_vars={}, expected {} (11 + k_exp)",
                            entry.num_vars, num_vars_gt_exp
                        );
                    }
                    for c_exp in 0..cs.gt_exp_witnesses.len() {
                        let src = match entry.poly_type {
                            PolyType::RhoPrev => &cs.gt_exp_witnesses[c_exp].rho_packed,
                            PolyType::Quotient => &cs.gt_exp_witnesses[c_exp].quotient_packed,
                            _ => unreachable!(),
                        };
                        fill_block_bit_reversed(dst, src, entry.num_vars, c_exp, 11);
                    }
                }
                PolyType::GtExpBase
                | PolyType::GtExpBase2
                | PolyType::GtExpBase3
                | PolyType::GtExpBaseSquareQuotient
                | PolyType::GtExpBaseCubeQuotient => {
                    if entry.num_vars != num_vars_gt_exp_base {
                        panic!(
                            "GTExp base entry has num_vars={}, expected {} (4 + k_exp)",
                            entry.num_vars, num_vars_gt_exp_base
                        );
                    }
                    // Extract the native 4-var table from the 11-var packed witness by
                    // taking the s=0 slice (base is replicated across s).
                    const STEP_STRIDE: usize = 1usize << 7; // 2^STEP_VARS (STEP_VARS = 7)
                    debug_assert_eq!(
                        cs.g_poly.Z.len(),
                        16,
                        "expected GT g polynomial to have 16 evaluations"
                    );
                    for c_exp in 0..cs.gt_exp_witnesses.len() {
                        // Always extract base/base2/base3 on the native 4-var u-domain.
                        let b11 = &cs.gt_exp_witnesses[c_exp].base_packed;
                        let b211 = &cs.gt_exp_witnesses[c_exp].base2_packed;
                        let b311 = &cs.gt_exp_witnesses[c_exp].base3_packed;
                        debug_assert_eq!(b11.len(), 1usize << 11);
                        debug_assert_eq!(b211.len(), 1usize << 11);
                        debug_assert_eq!(b311.len(), 1usize << 11);

                        let mut base4 = [Fq::zero(); 16];
                        let mut base24 = [Fq::zero(); 16];
                        let mut base34 = [Fq::zero(); 16];
                        for u in 0..16 {
                            base4[u] = b11[u * STEP_STRIDE];
                            base24[u] = b211[u * STEP_STRIDE];
                            base34[u] = b311[u * STEP_STRIDE];
                        }

                        let src4: [Fq; 16] = match entry.poly_type {
                            PolyType::GtExpBase => base4,
                            PolyType::GtExpBase2 => base24,
                            PolyType::GtExpBase3 => base34,
                            PolyType::GtExpBaseSquareQuotient => {
                                let mut q2 = [Fq::zero(); 16];
                                for u in 0..16 {
                                    let g = cs.g_poly.Z[u];
                                    if g.is_zero() {
                                        // At roots of g we require base^2 == base2; quotient is irrelevant.
                                        debug_assert!((base4[u] * base4[u] - base24[u]).is_zero());
                                        q2[u] = Fq::zero();
                                    } else {
                                        q2[u] = (base4[u] * base4[u] - base24[u])
                                            * g.inverse().unwrap();
                                    }
                                }
                                q2
                            }
                            PolyType::GtExpBaseCubeQuotient => {
                                let mut q3 = [Fq::zero(); 16];
                                for u in 0..16 {
                                    let g = cs.g_poly.Z[u];
                                    if g.is_zero() {
                                        debug_assert!((base24[u] * base4[u] - base34[u]).is_zero());
                                        q3[u] = Fq::zero();
                                    } else {
                                        q3[u] = (base24[u] * base4[u] - base34[u])
                                            * g.inverse().unwrap();
                                    }
                                }
                                q3
                            }
                            _ => unreachable!(),
                        };

                        fill_block_bit_reversed(dst, &src4, entry.num_vars, c_exp, 4);
                    }
                }
                PolyType::MulLhs
                | PolyType::MulRhs
                | PolyType::MulResult
                | PolyType::MulQuotient => {
                    if entry.num_vars != num_vars_gt_mul {
                        panic!(
                            "GTMul entry has num_vars={}, expected {} (4 + k_mul)",
                            entry.num_vars, num_vars_gt_mul
                        );
                    }
                    for c_mul in 0..cs.gt_mul_rows.len() {
                        let src4 = match entry.poly_type {
                            PolyType::MulLhs => &cs.gt_mul_rows[c_mul].lhs,
                            PolyType::MulRhs => &cs.gt_mul_rows[c_mul].rhs,
                            PolyType::MulResult => &cs.gt_mul_rows[c_mul].result,
                            PolyType::MulQuotient => &cs.gt_mul_rows[c_mul].quotient,
                            _ => unreachable!(),
                        };
                        fill_block_bit_reversed(dst, src4, entry.num_vars, c_mul, 4);
                    }
                }
                _ => {
                    // Other poly types are not GT.
                }
            }
            return;
        }

        if entry.is_g1_scalar_mul {
            // G1 scalar-mul rows are native 8-var step traces plus a family-local padded `c`.
            let num_g1 = cs.g1_scalar_mul_rows.len();
            let padded = num_g1.max(1).next_power_of_two();
            let k = padded.trailing_zeros() as usize;
            let expected_num_vars = 8usize + k;
            if entry.num_vars != expected_num_vars {
                panic!(
                    "G1 scalar-mul entry has num_vars={}, expected {} (8 + k_g1)",
                    entry.num_vars, expected_num_vars
                );
            }
            for c in 0..cs.g1_scalar_mul_rows.len() {
                let src8 = match entry.poly_type {
                    PolyType::G1ScalarMulXA => &cs.g1_scalar_mul_rows[c].x_a,
                    PolyType::G1ScalarMulYA => &cs.g1_scalar_mul_rows[c].y_a,
                    PolyType::G1ScalarMulXT => &cs.g1_scalar_mul_rows[c].x_t,
                    PolyType::G1ScalarMulYT => &cs.g1_scalar_mul_rows[c].y_t,
                    PolyType::G1ScalarMulXANext => &cs.g1_scalar_mul_rows[c].x_a_next,
                    PolyType::G1ScalarMulYANext => &cs.g1_scalar_mul_rows[c].y_a_next,
                    PolyType::G1ScalarMulTIndicator => &cs.g1_scalar_mul_rows[c].t_indicator,
                    PolyType::G1ScalarMulAIndicator => &cs.g1_scalar_mul_rows[c].a_indicator,
                    _ => continue,
                };
                fill_block_bit_reversed(dst, src8, entry.num_vars, c, 8);
            }
            return;
        }

        if entry.is_g1_scalar_mul_base {
            // G1 scalar-mul base rows are **c-only** over a family-local padded `c`.
            let num_g1 = cs.g1_scalar_mul_rows.len();
            let padded = num_g1.max(1).next_power_of_two();
            let k = padded.trailing_zeros() as usize;
            let expected_num_vars = k;
            if entry.num_vars != expected_num_vars {
                panic!(
                    "G1 scalar-mul base entry has num_vars={}, expected {} (k_g1)",
                    entry.num_vars, expected_num_vars
                );
            }
            for c in 0..cs.g1_scalar_mul_rows.len() {
                let (x, y) = cs.g1_scalar_mul_rows[c].base_point;
                let v = match entry.poly_type {
                    PolyType::G1ScalarMulXP => x,
                    PolyType::G1ScalarMulYP => y,
                    _ => continue,
                };
                fill_scalar_bit_reversed(dst, entry.num_vars, c, v);
            }
            return;
        }

        if entry.is_g1_add {
            // G1 add rows are c-only over a family-local padded `c_add`.
            let num_g1 = cs.g1_add_rows.len();
            let padded = num_g1.max(1).next_power_of_two();
            let k = padded.trailing_zeros() as usize;
            let expected_num_vars = k;
            if entry.num_vars != expected_num_vars {
                panic!(
                    "G1 add entry has num_vars={}, expected {} (k_add)",
                    entry.num_vars, expected_num_vars
                );
            }

            for c in 0..cs.g1_add_rows.len() {
                let v = match entry.poly_type {
                    PolyType::G1AddXP => cs.g1_add_rows[c].x_p,
                    PolyType::G1AddYP => cs.g1_add_rows[c].y_p,
                    PolyType::G1AddPIndicator => cs.g1_add_rows[c].ind_p,
                    PolyType::G1AddXQ => cs.g1_add_rows[c].x_q,
                    PolyType::G1AddYQ => cs.g1_add_rows[c].y_q,
                    PolyType::G1AddQIndicator => cs.g1_add_rows[c].ind_q,
                    PolyType::G1AddXR => cs.g1_add_rows[c].x_r,
                    PolyType::G1AddYR => cs.g1_add_rows[c].y_r,
                    PolyType::G1AddRIndicator => cs.g1_add_rows[c].ind_r,
                    PolyType::G1AddLambda => cs.g1_add_rows[c].lambda,
                    PolyType::G1AddInvDeltaX => cs.g1_add_rows[c].inv_delta_x,
                    PolyType::G1AddIsDouble => cs.g1_add_rows[c].is_double,
                    PolyType::G1AddIsInverse => cs.g1_add_rows[c].is_inverse,
                    _ => continue,
                };
                fill_scalar_bit_reversed(dst, entry.num_vars, c, v);
            }
            return;
        }

        if entry.is_g2_scalar_mul {
            // G2 scalar-mul rows are native 8-var step traces plus a family-local padded `c`.
            let num_g2 = cs.g2_scalar_mul_rows.len();
            let padded = num_g2.max(1).next_power_of_two();
            let k = padded.trailing_zeros() as usize;
            let expected_num_vars = 8usize + k;
            if entry.num_vars != expected_num_vars {
                panic!(
                    "G2 scalar-mul entry has num_vars={}, expected {} (8 + k_g2)",
                    entry.num_vars, expected_num_vars
                );
            }

            for c in 0..cs.g2_scalar_mul_rows.len() {
                match entry.poly_type {
                    PolyType::G2ScalarMulXAC0 => fill_block_bit_reversed(
                        dst,
                        &cs.g2_scalar_mul_rows[c].x_a_c0,
                        entry.num_vars,
                        c,
                        8,
                    ),
                    PolyType::G2ScalarMulXAC1 => fill_block_bit_reversed(
                        dst,
                        &cs.g2_scalar_mul_rows[c].x_a_c1,
                        entry.num_vars,
                        c,
                        8,
                    ),
                    PolyType::G2ScalarMulYAC0 => fill_block_bit_reversed(
                        dst,
                        &cs.g2_scalar_mul_rows[c].y_a_c0,
                        entry.num_vars,
                        c,
                        8,
                    ),
                    PolyType::G2ScalarMulYAC1 => fill_block_bit_reversed(
                        dst,
                        &cs.g2_scalar_mul_rows[c].y_a_c1,
                        entry.num_vars,
                        c,
                        8,
                    ),
                    PolyType::G2ScalarMulXTC0 => fill_block_bit_reversed(
                        dst,
                        &cs.g2_scalar_mul_rows[c].x_t_c0,
                        entry.num_vars,
                        c,
                        8,
                    ),
                    PolyType::G2ScalarMulXTC1 => fill_block_bit_reversed(
                        dst,
                        &cs.g2_scalar_mul_rows[c].x_t_c1,
                        entry.num_vars,
                        c,
                        8,
                    ),
                    PolyType::G2ScalarMulYTC0 => fill_block_bit_reversed(
                        dst,
                        &cs.g2_scalar_mul_rows[c].y_t_c0,
                        entry.num_vars,
                        c,
                        8,
                    ),
                    PolyType::G2ScalarMulYTC1 => fill_block_bit_reversed(
                        dst,
                        &cs.g2_scalar_mul_rows[c].y_t_c1,
                        entry.num_vars,
                        c,
                        8,
                    ),
                    PolyType::G2ScalarMulXANextC0 => fill_block_bit_reversed(
                        dst,
                        &cs.g2_scalar_mul_rows[c].x_a_next_c0,
                        entry.num_vars,
                        c,
                        8,
                    ),
                    PolyType::G2ScalarMulXANextC1 => fill_block_bit_reversed(
                        dst,
                        &cs.g2_scalar_mul_rows[c].x_a_next_c1,
                        entry.num_vars,
                        c,
                        8,
                    ),
                    PolyType::G2ScalarMulYANextC0 => fill_block_bit_reversed(
                        dst,
                        &cs.g2_scalar_mul_rows[c].y_a_next_c0,
                        entry.num_vars,
                        c,
                        8,
                    ),
                    PolyType::G2ScalarMulYANextC1 => fill_block_bit_reversed(
                        dst,
                        &cs.g2_scalar_mul_rows[c].y_a_next_c1,
                        entry.num_vars,
                        c,
                        8,
                    ),
                    PolyType::G2ScalarMulTIndicator => fill_block_bit_reversed(
                        dst,
                        &cs.g2_scalar_mul_rows[c].t_indicator,
                        entry.num_vars,
                        c,
                        8,
                    ),
                    PolyType::G2ScalarMulAIndicator => fill_block_bit_reversed(
                        dst,
                        &cs.g2_scalar_mul_rows[c].a_indicator,
                        entry.num_vars,
                        c,
                        8,
                    ),
                    _ => continue,
                };
            }
            return;
        }

        if entry.is_g2_scalar_mul_base {
            // G2 scalar-mul base rows are **c-only** over a family-local padded `c`.
            let num_g2 = cs.g2_scalar_mul_rows.len();
            let padded = num_g2.max(1).next_power_of_two();
            let k = padded.trailing_zeros() as usize;
            let expected_num_vars = k;
            if entry.num_vars != expected_num_vars {
                panic!(
                    "G2 scalar-mul base entry has num_vars={}, expected {} (k_g2)",
                    entry.num_vars, expected_num_vars
                );
            }

            for c in 0..cs.g2_scalar_mul_rows.len() {
                let (x, y) = cs.g2_scalar_mul_rows[c].base_point;
                let v = match entry.poly_type {
                    PolyType::G2ScalarMulXPC0 => x.c0,
                    PolyType::G2ScalarMulXPC1 => x.c1,
                    PolyType::G2ScalarMulYPC0 => y.c0,
                    PolyType::G2ScalarMulYPC1 => y.c1,
                    _ => continue,
                };
                fill_scalar_bit_reversed(dst, entry.num_vars, c, v);
            }
            return;
        }

        if entry.is_g2_add {
            // G2 add rows are c-only over a family-local padded `c_add`.
            let num_g2 = cs.g2_add_rows.len();
            let padded = num_g2.max(1).next_power_of_two();
            let k = padded.trailing_zeros() as usize;
            let expected_num_vars = k;
            if entry.num_vars != expected_num_vars {
                panic!(
                    "G2 add entry has num_vars={}, expected {} (k_add)",
                    entry.num_vars, expected_num_vars
                );
            }

            for c in 0..cs.g2_add_rows.len() {
                let v = match entry.poly_type {
                    PolyType::G2AddXPC0 => cs.g2_add_rows[c].x_p_c0,
                    PolyType::G2AddXPC1 => cs.g2_add_rows[c].x_p_c1,
                    PolyType::G2AddYPC0 => cs.g2_add_rows[c].y_p_c0,
                    PolyType::G2AddYPC1 => cs.g2_add_rows[c].y_p_c1,
                    PolyType::G2AddPIndicator => cs.g2_add_rows[c].ind_p,
                    PolyType::G2AddXQC0 => cs.g2_add_rows[c].x_q_c0,
                    PolyType::G2AddXQC1 => cs.g2_add_rows[c].x_q_c1,
                    PolyType::G2AddYQC0 => cs.g2_add_rows[c].y_q_c0,
                    PolyType::G2AddYQC1 => cs.g2_add_rows[c].y_q_c1,
                    PolyType::G2AddQIndicator => cs.g2_add_rows[c].ind_q,
                    PolyType::G2AddXRC0 => cs.g2_add_rows[c].x_r_c0,
                    PolyType::G2AddXRC1 => cs.g2_add_rows[c].x_r_c1,
                    PolyType::G2AddYRC0 => cs.g2_add_rows[c].y_r_c0,
                    PolyType::G2AddYRC1 => cs.g2_add_rows[c].y_r_c1,
                    PolyType::G2AddRIndicator => cs.g2_add_rows[c].ind_r,
                    PolyType::G2AddLambdaC0 => cs.g2_add_rows[c].lambda_c0,
                    PolyType::G2AddLambdaC1 => cs.g2_add_rows[c].lambda_c1,
                    PolyType::G2AddInvDeltaXC0 => cs.g2_add_rows[c].inv_delta_x_c0,
                    PolyType::G2AddInvDeltaXC1 => cs.g2_add_rows[c].inv_delta_x_c1,
                    PolyType::G2AddIsDouble => cs.g2_add_rows[c].is_double,
                    PolyType::G2AddIsInverse => cs.g2_add_rows[c].is_inverse,
                    _ => continue,
                };
                fill_scalar_bit_reversed(dst, entry.num_vars, c, v);
            }
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
        PrefixPackingLayout::from_constraint_types(&cs.constraint_types)
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
