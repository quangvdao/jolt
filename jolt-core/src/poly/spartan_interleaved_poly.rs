use super::{
    eq_poly::EqPolynomial,
    multilinear_polynomial::{BindingOrder, MultilinearPolynomial},
    sparse_interleaved_poly::SparseCoefficient,
    split_eq_poly::{NewSplitEqPolynomial, SplitEqPolynomial},
    unipoly::{CompressedUniPoly, UniPoly},
};
// #[cfg(test)]
use crate::poly::dense_mlpoly::DensePolynomial;
#[cfg(test)]
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::subprotocols::sumcheck::process_eq_sumcheck_round;
use crate::{
    field::{JoltField, OptimizedMul},
    r1cs::builder::{eval_offset_lc, eval_offset_lc_i64, Constraint, OffsetEqConstraint},
    utils::{
        math::Math,
        transcript::{AppendToTranscript, Transcript},
    },
};
use ark_ff::Zero;
use rayon::prelude::*;

use std::collections::BTreeMap;

// Helper Enum for SVO evaluation points
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)] // Added Ord for BTreeMap key
pub enum SVOEvalPoint {
    Zero,
    One,
    Infinity,
}

// TODO: Implement these helper functions

// Evaluates Az, Bz for a given R1CS row with fully binary SVO prefix.
fn evaluate_Az_Bz_for_r1cs_row_binary<F: JoltField>(
    uniform_constraints: &[Constraint],
    cross_step_constraints: &[OffsetEqConstraint],
    flattened_polynomials: &[&MultilinearPolynomial<F>],
    current_step_idx: usize,
    constraint_idx_within_step: usize, // Full constraint index including SVO bits (all binary)
    num_uniform_constraints: usize,
    num_total_constraints_padded: usize, // For calculating global_r1cs_idx if needed by caller
    num_total_steps: usize,
) -> (i64, i64) {
    // Returns (az_coeff_i64, bz_coeff_i64)
    let mut az_i64 = 0i64;
    let mut bz_i64 = 0i64;

    if constraint_idx_within_step < num_uniform_constraints {
        let constraint = &uniform_constraints[constraint_idx_within_step];
        if !constraint.a.terms().is_empty() {
            az_i64 = constraint
                .a
                .evaluate_row_i64(flattened_polynomials, current_step_idx);
        }
        if !constraint.b.terms().is_empty() {
            bz_i64 = constraint
                .b
                .evaluate_row_i64(flattened_polynomials, current_step_idx);
        }
    } else if constraint_idx_within_step < num_uniform_constraints + cross_step_constraints.len() {
        let cross_step_constraint_idx = constraint_idx_within_step - num_uniform_constraints;
        let constraint = &cross_step_constraints[cross_step_constraint_idx];
        let next_step_index_opt = if current_step_idx + 1 < num_total_steps {
            Some(current_step_idx + 1)
        } else {
            None
        };

        let eq_a_eval = eval_offset_lc_i64(
            &constraint.a,
            flattened_polynomials,
            current_step_idx,
            next_step_index_opt,
        );
        let eq_b_eval = eval_offset_lc_i64(
            &constraint.b,
            flattened_polynomials,
            current_step_idx,
            next_step_index_opt,
        );
        let temp_az = eq_a_eval - eq_b_eval;

        if !temp_az.is_zero() {
            az_i64 = temp_az;
            // Optional: Assert cond is zero:
            // let cond_eval = eval_offset_lc_i64(&constraint.cond, flattened_polynomials, current_step_idx, next_step_index_opt);
            // assert_eq!(cond_eval, 0, "Cross-step constraint violated");
        } else {
            bz_i64 = eval_offset_lc_i64(
                &constraint.cond,
                flattened_polynomials,
                current_step_idx,
                next_step_index_opt,
            );
        }
    }
    (az_i64, bz_i64)
}

// Computes product P(Z_curr, Y_ext) = Az_ext * Bz_ext (no Cz for SVO infinity paths)
// from all binary Az/Bz values at Z_curr. Updates temp_tA.
fn TODO_compute_extended_products_and_update_tA<F: JoltField>(
    binary_evals_at_Z_current: &[(i64, i64)], // Indexed by Y_binary_prefix_val (0 to 2^num_svo_rounds - 1)
    num_svo_rounds: usize,
    E_in_val: F, // Factor from Teq over x_in variables (using tau parts for Z_in)
    temp_tA: &mut BTreeMap<Vec<SVOEvalPoint>, F>, // Key: Y_extended (length num_svo_rounds)
) {
    // Placeholder: Actual implementation is complex.
    // Iterate all 3^num_svo_rounds Y_ext (Vec<SVOEvalPoint>).
    // For each Y_ext:
    //   Calculate Az_ext(Z_curr, Y_ext) and Bz_ext(Z_curr, Y_ext) using multilinearity
    //   from the provided `binary_evals_at_Z_current`.
    //   product_P = Az_ext * Bz_ext.
    //   *temp_tA.entry(Y_ext.clone()).or_insert(F::zero()) += E_in_val * product_P;
}

// Computes Teq factor for E_out_s
// This function now uses the precomputed E_out_s evaluations from eq_poly.E1.
#[inline]
fn get_E_out_s_val<F: JoltField>(
    E_out_s_evals: &[F],              // Precomputed evaluations from eq_poly.E1[svo_round_s]
    num_y_suffix_vars: usize,         // Number of variables in y_suffix
    num_x_out_vars: usize,    // Number of variables in x_out
    y_suffix_as_int: usize,           // Integer representation of y_suffix (LSB)
    x_out_val: usize,           // Integer assignment for x_out variables (MSB)
) -> F {
    // Sanity check for evaluation table length
    let expected_len = 1 << (num_y_suffix_vars + num_x_out_vars);
    assert_eq!(E_out_s_evals.len(), expected_len, "E_out_s_evals has unexpected length {}. Expected {}. num_y_suffix_vars = {}, num_x_out_vars = {}", E_out_s_evals.len(), expected_len, num_y_suffix_vars, num_x_out_vars);

    // y_suffix_as_int is LSB, x_out_val is MSB.
    let combined_idx = (x_out_val << num_y_suffix_vars) | y_suffix_as_int;
    
    assert!(combined_idx < E_out_s_evals.len(), "combined_idx out of bounds for E_out_s_evals");
    E_out_s_evals[combined_idx]
}

// Implements Definition 4 ("idx4") from the paper to map an extended SVO prefix
// to (round_s, v_config, u_eval) tuples. y_suffix is handled by the caller.
fn idx_mapping(
    svo_prefix_extended: &[SVOEvalPoint], // k_eff == num_svo_rounds length
    num_svo_rounds: usize,
) -> Vec<(
    usize,             // SVO round index s (0 to num_svo_rounds-1)
    Vec<SVOEvalPoint>, // v_config (v_0, ..., v_{s-1}), length s
    SVOEvalPoint,      // u_eval (for y_s)
)> {
    let mut result = Vec::new();
    if num_svo_rounds == 0 {
        return result;
    }
    assert_eq!(svo_prefix_extended.len(), num_svo_rounds);

    for s in 0..num_svo_rounds {
        let v_config: Vec<SVOEvalPoint> = svo_prefix_extended[0..s].to_vec();
        let u_eval = svo_prefix_extended[s];

        // Condition: u_eval must be Zero or Infinity for Spartan Az*Bz-Cz context with SVO.
        if !(u_eval == SVOEvalPoint::Zero || u_eval == SVOEvalPoint::Infinity) {
            continue;
        }

        let y_suffix_components = &svo_prefix_extended[s + 1..num_svo_rounds];
        let mut y_suffix_is_binary = true;
        for &point in y_suffix_components.iter() {
            if point == SVOEvalPoint::Infinity {
                y_suffix_is_binary = false;
                break;
            }
        }

        // Condition: y_suffix must be binary.
        // This condition from idx4 ensures that the Y_ext can correspond to a P(v,u,y,...) where y is binary.
        if !y_suffix_is_binary {
            continue;
        }

        result.push((s, v_config, u_eval));
    }
    result
}

// Maps a v-configuration (Vec<SVOEvalPoint> of length s) to a unique usize index (0 to 3^s - 1).
fn map_v_config_to_idx(v_config: &[SVOEvalPoint], _round_s: usize) -> usize {
    // Example: Treat SVOEvalPoint::{Zero, One, Infinity} as digits 0, 1, 2 in base 3.
    let mut index = 0;
    let base: usize = 3;
    for (i, point) in v_config.iter().enumerate() {
        let digit = match point {
            SVOEvalPoint::Zero => 0,
            SVOEvalPoint::One => 1,
            SVOEvalPoint::Infinity => 2,
        };
        index += digit * base.pow(i as u32);
    }
    index
}

// Result of a single parallel task in the fold-reduce pattern for the small value optimization (SVO)
struct PartialSmallValueContrib<F: JoltField> {
    // Outer Vec indexed by y_svo_e2_binary_bucket.
    // Inner Vec contains SparseCoefficients for that bucket from this task,
    // generated in sorted order by global_r1cs_idx.
    ab_coeffs_buckets_contribution: Vec<SparseCoefficient<i64>>,

    // Direct contribution to the final accumulator structure.
    // Outer Vec: SVO round s (0 to num_svo_rounds-1)
    // Tuple: (Vec of vals_for_u_eq_0, Vec of vals_for_u_eq_infty), each inner Vec indexed by v-config index.
    svo_accumulators_contribution: Vec<(Vec<F>, Vec<F>)>,
}

pub struct NewSpartanInterleavedPolynomial<F: JoltField> {
    /// A sparse vector representing the (interleaved) coefficients for the Az, Bz polynomials
    /// (note: **no** Cz coefficients are stored here, since they are not needed for small value
    /// precomputation, and can be computed on the fly in streaming round)
    pub(crate) ab_unbound_coeffs: Vec<SparseCoefficient<i64>>,

    pub(crate) az_bound_coeffs: DensePolynomial<F>,
    pub(crate) bz_bound_coeffs: DensePolynomial<F>,
    pub(crate) cz_bound_coeffs: DensePolynomial<F>,

    pub(crate) dense_len: usize,
}

impl<F: JoltField> NewSpartanInterleavedPolynomial<F> {
    /// Compute the unbound coefficients for the Az and Bz polynomials (no Cz coefficients are
    /// needed), along with the accumulators for the small value optimization (SVO) rounds.
    ///
    /// Recall that the accumulators are of the form:
    /// accum_i[v_0, ..., v_{i-1}, u] = \sum_{y_rest} \sum_{x_out} E_out(x_out || y_rest) * 
    /// \sum_{x_in} E_in(x_in) * P(x_out, x_in, y_rest, u, v_0, ..., v_{i-1}),
    ///
    /// for all i < num_svo_rounds, v_0,..., v_{i-1} \in {0,1,infty}, u \in {0,infty}, and 
    /// P(X) = Az(X) * Bz(X) - Cz(X).
    /// 
    /// Note that we have reverse the order of variables from the paper, since in this codebase
    /// the indexing is MSB to LSB (as we go from 0 to N-1, i.e. left to right).
    /// 
    /// Note that only the accumulators with at least one infinity among v_j and u are non-zero, so
    /// the fully binary ones do not need to be computed. Plus, the ones with at least one infinity
    /// will NOT have any Cz contributions.
    ///
    /// This is why we do not need to compute the Cz terms in the unbound coefficients.
    ///
    /// The output of the accumulators is Vec<(Vec<F>, Vec<F>)>, where the outer Vec is indexed by SVO
    /// round `i` (0 to `num_svo_rounds`-1). The tuple contains two Vec<F>: the first for evals at
    /// u = 0, and the second for u = infty. Each of these inner Vecs is indexed by the v-config
    /// (v_0, ..., v_{i-1}).
    pub fn new_with_precompute(
        padded_num_constraints: usize,
        uniform_constraints: &[Constraint],
        cross_step_constraints: &[OffsetEqConstraint],
        flattened_polynomials: &[&MultilinearPolynomial<F>],
        tau: &[F],
        num_svo_rounds: usize,
    ) -> (Vec<(Vec<F>, Vec<F>)>, Self) {
        let eq_poly = NewSplitEqPolynomial::new_for_small_value(tau, num_svo_rounds);

        // E_in is the unique vector in E1 (E_in)
        let E_in_evals = eq_poly.E_in_current();
        // E_out_vec contains the evaluation tables for E_out_s, for s = 0 to num_svo_rounds-1
        // E_out_vec[s] corresponds to challenges for
        // (X_{out}, Y_{num_step_vars-num_svo_rounds}..Y_{num_step_vars-s-1})
        let E_out_vec = &eq_poly.E_out_vec; // eq_poly.E_out_vec is Vec<Vec<F>>

        assert_eq!(E_out_vec.len(), num_svo_rounds);

        // Here's the full layout of the variables:
        // 0 ... (N/2 - l) ... (n_s) ... (N - l) ... (N - i - 1) ... (N - 1)
        // where n_s = num_step_vars, n_c = num_constraint_vars, N = n_s + n_c, l = num_svo_rounds
        // and i is an iterator over 0..l (for the SVO rounds)

        // Within this layout, we have the partition:
        // - 0 ... (N/2 - l) is x_out
        // - (N/2 - l) ... (n_s) is x_in_step
        // - (n_s) ... (N - l) is x_in_constraint (i.e. non_svo_constraint)
        // - (N/2 - l) ... (N - l) in total is x_in
        // - (N - l) ... (N - i - 1) is y_suffix_svo
        // - (N - i - 1) ... (N - 1) is u || v_config

        let num_steps = flattened_polynomials[0].len();
        let num_step_vars = if num_steps > 0 { num_steps.log_2() } else { 0 }; // Handle num_steps = 0 or 1
        let num_constraint_vars = if padded_num_constraints > 0 {
            padded_num_constraints.log_2()
        } else {
            0
        };
        let total_num_vars = num_step_vars + num_constraint_vars;

        assert_eq!(
            tau.len(),
            total_num_vars,
            "tau length ({}) mismatch with R1CS variable count (step_vars {} + constraint_vars {})",
            tau.len(),
            num_step_vars,
            num_constraint_vars
        );
        assert!(
            num_svo_rounds <= num_constraint_vars,
            "num_svo_rounds ({}) cannot exceed total constraint variables ({})",
            num_svo_rounds,
            num_constraint_vars
        );

        // This is the number of non-SVO bits within the constraint index part
        let num_non_svo_constraint_vars = num_constraint_vars.saturating_sub(num_svo_rounds);

        // // Defining x_in and x_out variables (my writing, commented out, do not delete)
        // let num_x_out_vars = total_num_vars / 2 - num_svo_rounds;
        // let num_x_in_vars = eq_poly.E_in_current_len().log_2();
        // let num_x_in_step_vars = num_x_in_vars - num_non_svo_constraint_vars;
        // let num_x_in_constraint_vars = num_non_svo_constraint_vars;

        // Defining x_in and x_out variables based on tau partitioning for E_in and E_out
        // These are counts of *all* variables (SVO or non-SVO) in these conceptual tau segments.
        let num_vars_for_E_out_prefix_segment = total_num_vars / 2 - num_svo_rounds;
        let num_vars_for_E_in_segment = eq_poly.E_in_current_len().log_2(); // num_vars in tau[N/2-l .. N-l-1]

        // Now, determine how many of *these specific counts* are to be iterated as 
        // non-SVO Z variables (x_out_val, x_in_val) in the main loops.
        // This interpretation assumes x_out_val iterates assignments for non-SVO Z vars
        // that fall into the E_out prefix segment, and x_in_val for those in E_in segment.

        // num_x_out_vars: Non-SVO Z variables corresponding to tau[0 ... num_vars_for_E_out_prefix_segment-1]
        // These are purely step_vars if num_vars_for_E_out_prefix_segment <= num_step_vars
        let iter_num_x_out_vars = std::cmp::min(num_step_vars, num_vars_for_E_out_prefix_segment);
        
        // num_x_in_vars: Non-SVO Z variables corresponding to tau[num_vars_for_E_out_prefix_segment ... num_vars_for_E_out_prefix_segment + num_vars_for_E_in_segment - 1]
        let start_tau_idx_for_E_in_iter_vars = num_vars_for_E_out_prefix_segment;
        let end_tau_idx_for_E_in_iter_vars = num_vars_for_E_out_prefix_segment + num_vars_for_E_in_segment;

        let iter_num_x_in_vars;
        let iter_num_x_in_step_vars;
        let iter_num_x_in_constraint_vars;

        if start_tau_idx_for_E_in_iter_vars >= num_step_vars + num_non_svo_constraint_vars { // E_in segment is entirely outside non-SVO Z vars
            iter_num_x_in_vars = 0;
            iter_num_x_in_step_vars = 0;
            iter_num_x_in_constraint_vars = 0;
        } else {
            let Z_vars_start_in_E_in_segment = std::cmp::max(start_tau_idx_for_E_in_iter_vars, 0); // Should be >= 0
            let Z_vars_end_in_E_in_segment = std::cmp::min(num_step_vars + num_non_svo_constraint_vars, end_tau_idx_for_E_in_iter_vars);
            iter_num_x_in_vars = Z_vars_end_in_E_in_segment.saturating_sub(Z_vars_start_in_E_in_segment);

            // Split iter_num_x_in_vars into step and constraint parts
            let Z_step_vars_end_in_E_in_segment = std::cmp::min(num_step_vars, Z_vars_end_in_E_in_segment);
            iter_num_x_in_step_vars = Z_step_vars_end_in_E_in_segment.saturating_sub(std::cmp::max(start_tau_idx_for_E_in_iter_vars, std::cmp::min(num_step_vars, Z_vars_start_in_E_in_segment)));
            
            iter_num_x_in_constraint_vars = iter_num_x_in_vars - iter_num_x_in_step_vars;
        }


        let num_x_out_vals = 1 << iter_num_x_out_vars; 
        let num_x_in_vals = 1 << iter_num_x_in_vars; 

        assert_eq!(E_in_evals.len(), 1 << num_vars_for_E_in_segment, "E_in_evals length mismatch with variables in E_in tau segment");
        for i in 0..num_svo_rounds {
            // num_y_suffix_vars for E_out_vec[i] is num_svo_rounds - 1 - i
            // num_x_out_vars for E_out_vec[i] (from tau perspective) is num_vars_for_E_out_prefix_segment
            let expected_E_out_len = 1 << (num_vars_for_E_out_prefix_segment + (num_svo_rounds - 1 - i));
            assert_eq!(E_out_vec[i].len(), expected_E_out_len, "E_out_vec[{}] length mismatch", i);
        }

        // --- Parallel Fold-Reduce over x_out_val ---
        let reduction_identity = || PartialSmallValueContrib {
            ab_coeffs_buckets_contribution: vec![],
            svo_accumulators_contribution: (0..num_svo_rounds)
                .map(|s| {
                    let v_config_count = 3_usize.pow(s as u32);
                    (vec![F::zero(); v_config_count], vec![F::zero(); v_config_count])
                })
                .collect(),
        };

        let fold_result: PartialSmallValueContrib<F> = (0..num_x_out_vals)
            .into_par_iter() // Make it parallel
            .map(|x_out_val| {
                let mut task_ab_coeffs_buckets: Vec<SparseCoefficient<i64>> = vec![];
                let mut task_svo_acc_contrib: Vec<(Vec<F>, Vec<F>)> = (0..num_svo_rounds)
                    .map(|s| {
                        let v_config_count = 3_usize.pow(s as u32);
                        (vec![F::zero(); v_config_count], vec![F::zero(); v_config_count])
                    })
                    .collect();

                // Temporary accumulators tA for this x_out_val.
                // Key: Full SVO prefix (Vec<SVOEvalPoint>) of length num_svo_rounds.
                // Value: Sum over x_in of (E_in_val * Az_extended * Bz_extended).
                let mut temp_tA: BTreeMap<Vec<SVOEvalPoint>, F> = BTreeMap::new();

                for x_in_val in 0..num_x_in_vals {
                    // Reconstruct current_step_idx and current_lower_bits_val from x_out_val and x_in_val
                    // x_out_val is assignment for the first iter_num_x_out_vars (which are step_vars)
                    // x_in_val is assignment for iter_num_x_in_step_vars of step_vars and iter_num_x_in_constraint_vars of constraint_vars
                    
                    let mut temp_step_idx = x_out_val; // These are the most significant bits of step_idx
                    
                    // Add step bits from x_in_val
                    let x_in_step_part = x_in_val >> iter_num_x_in_constraint_vars; // Higher bits of x_in_val
                    temp_step_idx = (temp_step_idx << iter_num_x_in_step_vars) | x_in_step_part;
                    
                    let current_step_idx = temp_step_idx;
                    
                    // Lower bits of x_in_val are for constraints
                    let constraint_mask = (1 << iter_num_x_in_constraint_vars) - 1;
                    let current_lower_bits_val = x_in_val & constraint_mask;


                    // This stores (Az_bin, Bz_bin) for current Z and all binary Y
                    let mut binary_evals_for_current_Z: Vec<(i64, i64)> =
                        vec![(0, 0); 1 << num_svo_rounds];

                    for y_svo_binary_prefix_val in 0..(1 << num_svo_rounds) {
                        let constraint_idx_within_step =
                            (y_svo_binary_prefix_val << num_non_svo_constraint_vars) + current_lower_bits_val;

                        if constraint_idx_within_step < padded_num_constraints {
                            let (az_i64, bz_i64) = evaluate_Az_Bz_for_r1cs_row_binary::<F>(
                                uniform_constraints,
                                cross_step_constraints,
                                flattened_polynomials,
                                current_step_idx,
                                constraint_idx_within_step,
                                uniform_constraints.len(),
                                padded_num_constraints,
                                num_steps,
                            );
                            binary_evals_for_current_Z[y_svo_binary_prefix_val] = (az_i64, bz_i64);

                            if num_svo_rounds > 0 { // Only bucket if SVO is active
                                let global_r1cs_idx =
                                    current_step_idx * padded_num_constraints + constraint_idx_within_step;
                                if az_i64 != 0 {
                                    task_ab_coeffs_buckets.push((global_r1cs_idx * 2, az_i64).into());
                                }
                                if bz_i64 != 0 {
                                    task_ab_coeffs_buckets.push((global_r1cs_idx * 2 + 1, bz_i64).into());
                                }
                            } else { // No SVO, all coeffs go to a single conceptual bucket (bucket 0)
                                 let global_r1cs_idx =
                                    current_step_idx * padded_num_constraints + constraint_idx_within_step;
                                if az_i64 != 0 {
                                    task_ab_coeffs_buckets.push((global_r1cs_idx * 2, az_i64).into());
                                }
                                if bz_i64 != 0 {
                                    task_ab_coeffs_buckets.push((global_r1cs_idx * 2 + 1, bz_i64).into());
                                }
                            }
                        }
                    } // End loop over binary SVO prefixes

                    if num_svo_rounds > 0 {
                        let E_in_val = E_in_evals[x_in_val];
                        TODO_compute_extended_products_and_update_tA::<F>(
                            &binary_evals_for_current_Z,
                            num_svo_rounds,
                            E_in_val,
                            &mut temp_tA,
                        );
                    }
                } // End loop over x_in_val

                // --- Distribute temp_tA to task_svo_acc_contrib ---
                if num_svo_rounds > 0 {
                    for (svo_prefix_extended, tA_val) in temp_tA {
                        if tA_val.is_zero() { continue; }

                        // idx_mapping gives (s, v_config, u_eval) for which this svo_prefix_extended is relevant.
                        // The y_suffix part of svo_prefix_extended must be binary for it to be in the result of idx_mapping.
                        for (round_s, v_config_vec, u_eval_point) in
                            idx_mapping(&svo_prefix_extended, num_svo_rounds)
                        {
                            // The specific y_suffix from the svo_prefix_extended that led to this tA_val.
                            // This is the y_suffix for which P( (v_config, u_eval, y_suffix_from_Yext), Z_current) was computed in tA_val.
                            let y_suffix_components_from_Yext = &svo_prefix_extended[round_s + 1..num_svo_rounds];
                            let mut y_suffix_as_int_from_Yext = 0;
                            for (bit_idx, &point) in y_suffix_components_from_Yext.iter().enumerate() {
                                if point == SVOEvalPoint::One {
                                    y_suffix_as_int_from_Yext |= 1 << bit_idx; // LSB-first conversion
                                }
                            }
                            let num_y_suffix_vars_for_this_Yext = y_suffix_components_from_Yext.len();

                            // The accumulator definition sums over all binary y_suffix for a given (round_s, v_config, u_eval).
                            // Here, tA_val is for a specific Y_ext, which includes a specific y_suffix (y_suffix_as_int_from_Yext).
                            // So, we only contribute this tA_val to the sum for that specific y_suffix.
                            // The E_out_s_val will be specific to y_suffix_as_int_from_Yext.

                            let E_out_s_val = get_E_out_s_val::<F>(
                                &E_out_vec[round_s],
                                num_y_suffix_vars_for_this_Yext, 
                                num_vars_for_E_out_prefix_segment, // Num X_out vars for E_out table
                                y_suffix_as_int_from_Yext,       
                                x_out_val, // x_out_val is assignment for the X_out vars of E_out table
                            );
                            let v_config_idx = map_v_config_to_idx(&v_config_vec, round_s);

                            match u_eval_point {
                                SVOEvalPoint::Zero => {
                                    task_svo_acc_contrib[round_s].0[v_config_idx] += E_out_s_val * tA_val;
                                }
                                SVOEvalPoint::Infinity => {
                                    task_svo_acc_contrib[round_s].1[v_config_idx] += E_out_s_val * tA_val;
                                }
                                SVOEvalPoint::One => {
                                     panic!("SVOEvalPoint::One should not be used in the accumulator");
                                    // As per current SVO logic, P(1) is not directly stored in these primary accums.
                                    // It's derived later: P(1) = P(0) + slope_at_0.
                                }
                            }
                        }
                    }
                }
                PartialSmallValueContrib {
                    ab_coeffs_buckets_contribution: task_ab_coeffs_buckets,
                    svo_accumulators_contribution: task_svo_acc_contrib,
                }
            })
            .reduce(reduction_identity, |mut acc_res, task_res| {
                acc_res.ab_coeffs_buckets_contribution.extend(
                    task_res.ab_coeffs_buckets_contribution
                        .iter()
                        .cloned(),
                );
                if num_svo_rounds > 0 {
                    for s in 0..num_svo_rounds {
                        for v_idx in 0..3_usize.pow(s as u32) {
                            acc_res.svo_accumulators_contribution[s].0[v_idx] +=
                                task_res.svo_accumulators_contribution[s].0[v_idx];
                            acc_res.svo_accumulators_contribution[s].1[v_idx] +=
                                task_res.svo_accumulators_contribution[s].1[v_idx];
                        }
                    }
                }
                acc_res
            });

        // If the parallel iteration and reduction strategy preserves the global sorted order
        // (e.g., by processing x_out_val in order and extending slices),
        // the final sort per bucket for ab_unbound_coeffs might not be needed.
        // However, a general Rayon reduce does not guarantee order of combination of task results.
        // To be safe and ensure sorted buckets if the reduce was arbitrary, or if tasks finished out of order:
        // We comment this out for now, and will revisit if the order is not sorted by fold-reduce.
        // let mut final_ab_unbound_coeffs = fold_result.ab_coeffs_buckets_contribution;
        // if num_y_svo_e2_buckets > 0 {
        //     // only sort if there are buckets
        //     for bucket in final_ab_unbound_coeffs.iter_mut() {
        //         bucket.sort_by_key(|sc| sc.index);
        //     }
        // }
        
        let mut final_ab_unbound_coeffs = fold_result.ab_coeffs_buckets_contribution;
        final_ab_unbound_coeffs.sort_by_key(|sc| sc.index); // Sort the flat list


        #[cfg(test)]
        {
            if num_svo_rounds > 0 && !final_ab_unbound_coeffs.is_empty() {
                let mut prev_index = final_ab_unbound_coeffs[0].index;
                for coeff in final_ab_unbound_coeffs.iter().skip(1) { // Iterate the flat list directly
                    assert!(
                        coeff.index > prev_index,
                        "Indices not monotonically increasing in final_ab_unbound_coeffs: prev {}, current {}",
                        prev_index, coeff.index
                    );
                    prev_index = coeff.index;
                }
            }
        }

        (
            fold_result.svo_accumulators_contribution,
            Self {
                ab_unbound_coeffs: final_ab_unbound_coeffs,
                az_bound_coeffs: DensePolynomial::new(vec![]),
                bz_bound_coeffs: DensePolynomial::new(vec![]),
                cz_bound_coeffs: DensePolynomial::new(vec![]),
                dense_len: num_steps * padded_num_constraints,
            },
        )
    }

    /// This function uses the streaming algorithm to compute the sum-check polynomial for the round
    /// right after the small value precomputed rounds.
    ///
    /// At this point, we have the `unbound_coeffs` generated from `new_with_precompute`. We will
    /// use these to compute the evals {Az/Bz/Cz}(r, u, x') needed for later linear-time sumcheck
    /// rounds (storing them in `{az/bz/cz}_bound_coeffs`), and compute the polynomial for this
    /// round at the same time.
    ///
    /// Recall that we need to compute
    ///
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] * (unbound_coeffs_a(r,0,x_in, x_out) *
    /// unbound_coeffs_b(r,0, x_in, x_out) - unbound_coeffs_c(r,0, x_in, x_out))`
    ///
    /// and `t_i(infty) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// (unbound_coeffs_a(r,infty,x_in, x_out) * unbound_coeffs_b(r,infty, x_in, x_out))`
    ///
    /// Here the "_a,b,c" subscript indicates the coefficients of `unbound_coeffs` corresponding
    /// to Az, Bz, Cz respectively. Importantly, since the eval at `r` is not cached, we will need
    /// to recompute it via another sum
    ///
    /// `unbound_coeffs_{a,b,c}(r, {0,infty}, x_in, x_out) = \sum_{y} eq(r, y) *
    /// unbound_coeffs_{a,b,c}(y, {0,infty}, x_in, x_out)`
    ///
    /// (and the eval at infty is computed as (eval at 1) - (eval at 0))
    ///
    /// Since `unbound_coeffs` are in sparse format, we will need to be more careful with
    /// indexing; see the old implementation for details.
    ///
    /// Finally, as we compute each `unbound_coeffs_{a,b,c}(r, {0,infty}, x_in, x_out)`, we will
    /// need to store them in `{az/bz/cz}_bound_coeffs`. (the eval at 1 will be eval at 0 + eval at
    /// infty). We then derive the next challenge from the transcript, and bind these bound coeffs
    /// for the next round.
    pub fn streaming_sumcheck_round<ProofTranscript: Transcript>(
        &mut self,
        eq_poly: &mut NewSplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r_challenges: &mut Vec<F>,
        round_polys: &mut Vec<CompressedUniPoly<F>>,
        claim: &mut F,
    ) {
        // --- 1. Initial Setup --- 
        let num_rounds_done = r_challenges.len();
        let total_vars = self.dense_len.log_2();
        let remaining_vars = total_vars - num_rounds_done;
        let current_var_index_k = remaining_vars - 1;
        let N_k = 1 << current_var_index_k;
        let N_high = 1 << num_rounds_done;
        let eq_r = EqPolynomial::evals(r_challenges);

        let num_x_out_vars = eq_poly.E_out_current_len().log_2();
        let num_x_in_vars = eq_poly.E_in_current_len().log_2();
        assert_eq!(num_x_out_vars + num_x_in_vars, current_var_index_k, "Mismatch in variable split for x_prime");
        let num_x_out_points = 1 << num_x_out_vars;
        let num_x_in_points = if num_x_in_vars == 0 { 1 } else { 1 << num_x_in_vars };

        // Pre-calculate ranges in self.ab_unbound_coeffs for each x_out_idx
        let mut ranges_for_x_out: Vec<std::ops::Range<usize>> = Vec::with_capacity(num_x_out_points);
        if !self.ab_unbound_coeffs.is_empty() {
            for x_out_idx_val in 0..num_x_out_points {
                let min_r1cs_row_idx_for_x_out = x_out_idx_val << num_x_in_vars; // Min y_high=0, min x_in=0
                let max_r1cs_row_idx_for_x_out = 
                    ((N_high -1) << (current_var_index_k +1)) | // Max y_high
                    (x_out_idx_val << num_x_in_vars) |          // Current x_out
                    ((1 << num_x_in_vars) -1);                   // Max x_in

                let min_sparse_idx = min_r1cs_row_idx_for_x_out * 2;
                let max_sparse_idx = max_r1cs_row_idx_for_x_out * 2 + 1;

                let start = self.ab_unbound_coeffs.partition_point(|sc| sc.index < min_sparse_idx);
                let end = self.ab_unbound_coeffs.partition_point(|sc| sc.index <= max_sparse_idx);
                ranges_for_x_out.push(start..end);
            }
        }

        #[derive(Debug)]
        struct StreamingRoundAccumulator<F: JoltField> {
            az0_evals_all: Vec<F>,
            az1_evals_all: Vec<F>,
            bz0_evals_all: Vec<F>,
            bz1_evals_all: Vec<F>,
            cz0_evals_all: Vec<F>,
            cz1_evals_all: Vec<F>,
            total_quadratic_eval_at_0: F,
            total_quadratic_eval_at_infty: F,
        }

        impl<F: JoltField> StreamingRoundAccumulator<F> {
            #[inline]
            fn new() -> Self {
                Self {
                    az0_evals_all: Vec::new(), az1_evals_all: Vec::new(),
                    bz0_evals_all: Vec::new(), bz1_evals_all: Vec::new(),
                    cz0_evals_all: Vec::new(), cz1_evals_all: Vec::new(),
                    total_quadratic_eval_at_0: F::zero(), total_quadratic_eval_at_infty: F::zero(),
                }
            }
        }

        let final_results: StreamingRoundAccumulator<F> = (0..num_x_out_points)
            .into_par_iter()
            .fold(
                || StreamingRoundAccumulator::new(),
                |mut thread_acc, x_out_idx| {
                    let e_out_val = eq_poly.E_out_current()[x_out_idx];
                    
                    let relevant_coeffs_slice = if !self.ab_unbound_coeffs.is_empty() && !ranges_for_x_out.is_empty() {
                        &self.ab_unbound_coeffs[ranges_for_x_out[x_out_idx].clone()] 
                    } else {
                        &[]
                    };
                    
                    let mut inner_sum_0_for_this_x_out = F::zero();
                    let mut inner_sum_infty_for_this_x_out = F::zero();

                    let mut az0_for_current_x_out: Vec<F> = vec![F::zero(); num_x_in_points];
                    let mut az1_for_current_x_out: Vec<F> = vec![F::zero(); num_x_in_points];
                    let mut bz0_for_current_x_out: Vec<F> = vec![F::zero(); num_x_in_points];
                    let mut bz1_for_current_x_out: Vec<F> = vec![F::zero(); num_x_in_points];
                    let mut cz0_for_current_x_out: Vec<F> = vec![F::zero(); num_x_in_points];
                    let mut cz1_for_current_x_out: Vec<F> = vec![F::zero(); num_x_in_points];

                    for x_in_idx in 0..num_x_in_points { 
                        let mut az0_at_x_prime = F::zero();
                        let mut bz0_at_x_prime = F::zero();
                        let mut cz0_at_x_prime = F::zero();
                        let mut az1_at_x_prime = F::zero();
                        let mut bz1_at_x_prime = F::zero();
                        let mut cz1_at_x_prime = F::zero();

                        let x_prime_val = (x_out_idx << num_x_in_vars) | x_in_idx;
                        
                        let mut current_coeffs_iter = relevant_coeffs_slice.iter().peekable();

                        for y_high_idx in 0..N_high {                            
                            let eq_r_val = eq_r[y_high_idx];

                            let r1cs_row_idx_if_Xk_0 = (y_high_idx << (current_var_index_k + 1)) | x_prime_val;
                            let r1cs_row_idx_if_Xk_1 = r1cs_row_idx_if_Xk_0 | (1 << current_var_index_k);
                                                    
                            let target_sparse_idx_Az_Xk_0 = r1cs_row_idx_if_Xk_0 * 2;
                            let target_sparse_idx_Bz_Xk_0 = r1cs_row_idx_if_Xk_0 * 2 + 1;
                            let target_sparse_idx_Az_Xk_1 = r1cs_row_idx_if_Xk_1 * 2;
                            let target_sparse_idx_Bz_Xk_1 = r1cs_row_idx_if_Xk_1 * 2 + 1;
                            
                            let mut loc_az0: i64 = 0;
                            let mut loc_bz0: i64 = 0;
                            let mut loc_az1: i64 = 0;
                            let mut loc_bz1: i64 = 0;
                            
                            // Peekable iterator allows advancing without consuming if condition not met.
                            // Iterator state persists across these while loops for a single (y_high_idx, x_prime_val)
                            while let Some(&coeff) = current_coeffs_iter.peek() { 
                                if coeff.index < target_sparse_idx_Az_Xk_0 { current_coeffs_iter.next(); continue; } 
                                if coeff.index == target_sparse_idx_Az_Xk_0 { loc_az0 = coeff.value; current_coeffs_iter.next(); } 
                                break; 
                            }
                            while let Some(&coeff) = current_coeffs_iter.peek() { 
                                if coeff.index < target_sparse_idx_Bz_Xk_0 { current_coeffs_iter.next(); continue; } 
                                if coeff.index == target_sparse_idx_Bz_Xk_0 { loc_bz0 = coeff.value; current_coeffs_iter.next(); } 
                                break;  
                            }
                            while let Some(&coeff) = current_coeffs_iter.peek() {
                                if coeff.index < target_sparse_idx_Az_Xk_1 { current_coeffs_iter.next(); continue; }
                                if coeff.index == target_sparse_idx_Az_Xk_1 { loc_az1 = coeff.value; current_coeffs_iter.next(); }
                                break; 
                            }
                            while let Some(&coeff) = current_coeffs_iter.peek() {
                                if coeff.index < target_sparse_idx_Bz_Xk_1 { current_coeffs_iter.next(); continue; }
                                if coeff.index == target_sparse_idx_Bz_Xk_1 { loc_bz1 = coeff.value; current_coeffs_iter.next(); }
                                break; 
                            }

                            if loc_az0 != 0 { az0_at_x_prime += eq_r_val.mul_i64(loc_az0); }
                            if loc_bz0 != 0 { bz0_at_x_prime += eq_r_val.mul_i64(loc_bz0); }
                            if loc_az0 != 0 && loc_bz0 != 0 { cz0_at_x_prime += eq_r_val.mul_i128(loc_az0 as i128 * loc_bz0 as i128); }
                            
                            if loc_az1 != 0 { az1_at_x_prime += eq_r_val.mul_i64(loc_az1); }
                            if loc_bz1 != 0 { bz1_at_x_prime += eq_r_val.mul_i64(loc_bz1); }
                            if loc_az1 != 0 && loc_bz1 != 0 { cz1_at_x_prime += eq_r_val.mul_i128(loc_az1 as i128 * loc_bz1 as i128); }
                        } 

                        az0_for_current_x_out[x_in_idx] = az0_at_x_prime;
                        az1_for_current_x_out[x_in_idx] = az1_at_x_prime;
                        bz0_for_current_x_out[x_in_idx] = bz0_at_x_prime;
                        bz1_for_current_x_out[x_in_idx] = bz1_at_x_prime;
                        cz0_for_current_x_out[x_in_idx] = cz0_at_x_prime;
                        cz1_for_current_x_out[x_in_idx] = cz1_at_x_prime;

                        let e_in_val = eq_poly.E_in_current()[x_in_idx];

                        let term_eval_at_0 = az0_at_x_prime * bz0_at_x_prime - cz0_at_x_prime;
                        let az_m = az1_at_x_prime - az0_at_x_prime;
                        let bz_m = bz1_at_x_prime - bz0_at_x_prime;
                        let term_eval_at_infty = az_m * bz_m;

                        inner_sum_0_for_this_x_out += e_in_val * term_eval_at_0;
                        inner_sum_infty_for_this_x_out += e_in_val * term_eval_at_infty;
                    } 

                    // Append the computed Vec<F> for this x_out_idx to the thread-local accumulator.
                    thread_acc.az0_evals_all.extend(az0_for_current_x_out);
                    thread_acc.az1_evals_all.extend(az1_for_current_x_out);
                    thread_acc.bz0_evals_all.extend(bz0_for_current_x_out);
                    thread_acc.bz1_evals_all.extend(bz1_for_current_x_out);
                    thread_acc.cz0_evals_all.extend(cz0_for_current_x_out);
                    thread_acc.cz1_evals_all.extend(cz1_for_current_x_out);

                    thread_acc.total_quadratic_eval_at_0 += e_out_val * inner_sum_0_for_this_x_out;
                    thread_acc.total_quadratic_eval_at_infty += e_out_val * inner_sum_infty_for_this_x_out;
                    
                    thread_acc
                },
            )
            .reduce(
                || StreamingRoundAccumulator::new(), 
                |mut acc_a, acc_b| {
                    // Concatenate the Vec<F> evaluations. Order is preserved by fold + extend.
                    acc_a.az0_evals_all.extend(acc_b.az0_evals_all);
                    acc_a.az1_evals_all.extend(acc_b.az1_evals_all);
                    acc_a.bz0_evals_all.extend(acc_b.bz0_evals_all);
                    acc_a.bz1_evals_all.extend(acc_b.bz1_evals_all);
                    acc_a.cz0_evals_all.extend(acc_b.cz0_evals_all);
                    acc_a.cz1_evals_all.extend(acc_b.cz1_evals_all);
                    
                    acc_a.total_quadratic_eval_at_0 += acc_b.total_quadratic_eval_at_0;
                    acc_a.total_quadratic_eval_at_infty += acc_b.total_quadratic_eval_at_infty;
                    acc_a
                }
            );

        // --- 3. Extract Final Results and Process Round ---
        // At this point, final_results.az0_evals_all (etc.) should have N_k elements in correct order.
        assert_eq!(
            final_results.az0_evals_all.len(),
            N_k,
            "Length mismatch in final Az0 evals"
        );

        let quadratic_eval_at_0 = final_results.total_quadratic_eval_at_0;
        let quadratic_eval_at_infty = final_results.total_quadratic_eval_at_infty;

        let r_i = process_eq_sumcheck_round(
            (quadratic_eval_at_0, quadratic_eval_at_infty),
            eq_poly,
            round_polys,
            r_challenges,
            claim,
            transcript,
        );

        // --- 4. Bind Dense Polynomials for Next Round ---
        let mut az_bound_next = vec![F::zero(); N_k];
        let mut bz_bound_next = vec![F::zero(); N_k];
        let mut cz_bound_next = vec![F::zero(); N_k];

        let final_az0 = &final_results.az0_evals_all;
        let final_az1 = &final_results.az1_evals_all;
        let final_bz0 = &final_results.bz0_evals_all;
        let final_bz1 = &final_results.bz1_evals_all;
        let final_cz0 = &final_results.cz0_evals_all;
        let final_cz1 = &final_results.cz1_evals_all;

        az_bound_next
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, val)| {
                *val = final_az0[idx] + r_i * (final_az1[idx] - final_az0[idx]);
            });
        bz_bound_next
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, val)| {
                *val = final_bz0[idx] + r_i * (final_bz1[idx] - final_bz0[idx]);
            });
        cz_bound_next
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, val)| {
                *val = final_cz0[idx] + r_i * (final_cz1[idx] - final_cz0[idx]);
            });

        self.ab_unbound_coeffs = Vec::new();
        self.az_bound_coeffs = DensePolynomial::new(az_bound_next);
        self.bz_bound_coeffs = DensePolynomial::new(bz_bound_next);
        self.cz_bound_coeffs = DensePolynomial::new(cz_bound_next);
        self.dense_len = N_k;
    }

    /// This function computes the polynomial for each of the remaining rounds, using the
    /// linear-time algorithm with split-eq optimizations
    ///
    /// At this point, we have computed the `{az/bz/cz}_bound_coeffs` for the current round.
    /// We need to compute:
    /// 
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// (az_bound[x_out, x_in, 0] * bz_bound[x_out, x_in, 0] - cz_bound[x_out, x_in, 0])`
    /// 
    /// and
    /// 
    /// `t_i(infty) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// az_bound[x_out, x_in, infty] * bz_bound[x_out, x_in, infty]`
    ///
    /// (ordering of indices is MSB to LSB, so x_out is the MSB and x_in is the LSB)
    /// 
    /// We can compute this via a single pass over the coefficients, similar to the old method below
    /// (but simpler since we work directly with dense instead of sparse coefficients)
    /// The closest analogue is the `dense_interleaved_poly` implementation.
    ///
    /// We then process this to form `s_i(X) = l_i(X) * t_i(X)`, append `s_i.compress()` to the transcript,
    /// derive next challenge `r_i`, then bind both `eq_poly` and `{az/bz/cz}_bound_coeffs` with `r_i`.
    pub fn remaining_sumcheck_round<ProofTranscript: Transcript>(
        &mut self,
        eq_poly: &mut NewSplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r_challenges: &mut Vec<F>,
        round_polys: &mut Vec<CompressedUniPoly<F>>,
        current_claim: &mut F,
    ) {
        // num_x_rest_evals is the number of evaluation points for the *remaining* variables (X_rest),
        // after the current sumcheck variable (X_current_MSB) is fixed to 0 or 1.
        let num_x_rest_evals = self.az_bound_coeffs.len() / 2;

        // Get references to the coefficient vectors' current state
        let az_coeffs = self.az_bound_coeffs.evals_ref();
        let bz_coeffs = self.bz_bound_coeffs.evals_ref();
        let cz_coeffs = self.cz_bound_coeffs.evals_ref();

        // Split each coefficient vector into low (X_MSB = 0) and high (X_MSB = 1) halves
        let (az_low, az_high) = az_coeffs.split_at(num_x_rest_evals);
        let (bz_low, bz_high) = bz_coeffs.split_at(num_x_rest_evals);
        let (cz_low, _cz_high) = cz_coeffs.split_at(num_x_rest_evals); // cz_high unused for t_i(infty)

        // Compute t_i(0) and t_i(infty)
        let (quadratic_eval_at_0, quadratic_eval_at_infty) = if eq_poly.E_in_current_len() == 1 {
            // E_in part is fully bound. E_out covers all remaining eq_poly variables.
            debug_assert_eq!(
                eq_poly.E_out_current_len(),
                num_x_rest_evals,
                "E_out_current_len should match num_x_rest_evals when E_in_current_len is 1"
            );

            // Combine iterators over the halves and E_out
            az_low
                .par_iter()
                .zip(az_high.par_iter())
                .zip(bz_low.par_iter())
                .zip(bz_high.par_iter())
                .zip(cz_low.par_iter())
                .zip(eq_poly.E_out_current().par_iter())
                .map(
                    |(((((az0_ref, az1_ref), bz0_ref), bz1_ref), cz0_ref), e_out_val_ref)| {
                        let az0 = *az0_ref;
                        let az1 = *az1_ref;
                        let bz0 = *bz0_ref;
                        let bz1 = *bz1_ref;
                        let cz0 = *cz0_ref;
                        let e_out_val = *e_out_val_ref;

                        let az_m = az1 - az0; // Coeff of X_current_MSB in Az
                        let bz_m = bz1 - bz0; // Coeff of X_current_MSB in Bz

                        let term_eval_at_0 = az0 * bz0 - cz0;
                        let term_eval_at_infty = az_m * bz_m; // X^2 coeff of (Az*Bz)

                        (e_out_val * term_eval_at_0, e_out_val * term_eval_at_infty)
                    },
                )
                .reduce(
                    || (F::zero(), F::zero()),
                    |(acc_0, acc_infty), (val_0, val_infty)| (acc_0 + val_0, acc_infty + val_infty),
                )
        } else {
            // Nested sum structure: sum over E_out (outer), then sum over E_in (inner)
            let num_e_out_points = eq_poly.E_out_current_len();
            let num_e_in_points = eq_poly.E_in_current_len();
            debug_assert_eq!(
                num_x_rest_evals,
                num_e_out_points * num_e_in_points,
                "num_x_rest_evals should be product of E_out_current_len and E_in_current_len"
            );

            eq_poly
                .E_out_current()
                .par_iter()
                .enumerate()
                .map(|(e_out_idx, e_out_val)| {
                    let mut inner_sum_eval_at_0 = F::zero();
                    let mut inner_sum_eval_at_infty = F::zero();

                    let start_idx_rest = e_out_idx * num_e_in_points;

                    // Iterate over the indices corresponding to this E_out chunk
                    for e_in_idx in 0..num_e_in_points {
                        let idx_rest = start_idx_rest + e_in_idx;
                        let e_in_val = eq_poly.E_in_current()[e_in_idx];

                        // Access halves using the calculated index
                        let az0 = az_low[idx_rest];
                        let az1 = az_high[idx_rest];
                        let az_m = az1 - az0;

                        let bz0 = bz_low[idx_rest];
                        let bz1 = bz_high[idx_rest];
                        let bz_m = bz1 - bz0;

                        let cz0 = cz_low[idx_rest];

                        let term_eval_at_0 = az0 * bz0 - cz0;
                        let term_eval_at_infty = az_m * bz_m;

                        // Inner sum part: sum_{x_in} E_in[x_in] * P(x_out, x_in, {0,infty}, r_{high})
                        inner_sum_eval_at_0 += e_in_val * term_eval_at_0;
                        inner_sum_eval_at_infty += e_in_val * term_eval_at_infty;
                    }
                    // Outer sum part: E_out[x_out] * (inner sum)
                    (
                        *e_out_val * inner_sum_eval_at_0,
                        *e_out_val * inner_sum_eval_at_infty,
                    )
                })
                .reduce(
                    || (F::zero(), F::zero()),
                    |(acc_0, acc_infty), (val_0, val_infty)| (acc_0 + val_0, acc_infty + val_infty),
                )
        };

        // Use the helper function to process the rest of the sumcheck round
        let r_i = process_eq_sumcheck_round(
            (quadratic_eval_at_0, quadratic_eval_at_infty), // (t_i(0), t_i(infty))
            eq_poly,                                        // Helper will bind this
            round_polys,
            r_challenges,
            current_claim,
            transcript,
        );

        // Bind Az, Bz, Cz polynomials for the next round using the challenge r_i
        // self.bind uses BindingOrder::HighToLow internally now
        self.bind(r_i);
    }

    /// Binds the current `{az/bz/cz}_bound_coeffs` with the challenge `r`
    pub fn bind(&mut self, r: F) {
        self.az_bound_coeffs.bind(r, BindingOrder::HighToLow);
        self.bz_bound_coeffs.bind(r, BindingOrder::HighToLow);
        self.cz_bound_coeffs.bind(r, BindingOrder::HighToLow);
    }

    pub fn final_sumcheck_evals(&self) -> [F; 3] {
        // Simply returns the final evals of Az, Bz, Cz
        // At this point `DensePolynomial` has been fully bound, and hence has length 1
        let az = self.az_bound_coeffs[0];
        let bz = self.bz_bound_coeffs[0];
        let cz = self.cz_bound_coeffs[0];
        [az, bz, cz]
    }
}

#[derive(Default, Debug, Clone)]
pub struct SpartanInterleavedPolynomial<F: JoltField> {
    /// A sparse vector representing the (interleaved) coefficients in the Az, Bz, Cz
    /// polynomials used in the first Spartan sumcheck. Before the polynomial is bound
    /// the first time, all the coefficients can be represented by `i128`s.
    pub(crate) unbound_coeffs: Vec<SparseCoefficient<i128>>,
    /// A sparse vector representing the (interleaved) coefficients in the Az, Bz, Cz
    /// polynomials used in the first Spartan sumcheck. Once the polynomial has been
    /// bound, we switch to using `bound_coeffs` instead of `unbound_coeffs`, because
    /// coefficients will be full-width field elements rather than `i128`s.
    pub(crate) bound_coeffs: Vec<SparseCoefficient<F>>,
    /// A reused buffer where bound values are written to during `bind`.
    /// With every bind, `coeffs` and `binding_scratch_space` are swapped.
    binding_scratch_space: Vec<SparseCoefficient<F>>,
    /// The length of one of the Az, Bz, or Cz polynomials if it were represented by
    /// a single dense vector.
    dense_len: usize,
}

impl<F: JoltField> SpartanInterleavedPolynomial<F> {
    /// Computes the matrix-vector products Az, Bz, and Cz as a single interleaved sparse vector
    #[tracing::instrument(skip_all, name = "SpartanInterleavedPolynomial::new")]
    pub fn new(
        uniform_constraints: &[Constraint],
        cross_step_constraints: &[OffsetEqConstraint],
        flattened_polynomials: &[&MultilinearPolynomial<F>], // N variables of (S steps)
        padded_num_constraints: usize,
    ) -> Self {
        let num_steps = flattened_polynomials[0].len();

        let num_chunks = rayon::current_num_threads().next_power_of_two() * 4;
        let chunk_size = num_steps.div_ceil(num_chunks);

        let unbound_coeffs: Vec<SparseCoefficient<i128>> = (0..num_chunks)
            .into_par_iter()
            .flat_map_iter(|chunk_index| {
                let mut coeffs = Vec::with_capacity(chunk_size * padded_num_constraints * 3);
                for step_index in chunk_size * chunk_index..chunk_size * (chunk_index + 1) {
                    // Uniform constraints
                    for (constraint_index, constraint) in uniform_constraints.iter().enumerate() {
                        let global_index =
                            3 * (step_index * padded_num_constraints + constraint_index);

                        // Az
                        let mut az_coeff = 0;
                        if !constraint.a.terms().is_empty() {
                            az_coeff = constraint
                                .a
                                .evaluate_row(flattened_polynomials, step_index);
                            if !az_coeff.is_zero() {
                                coeffs.push((global_index, az_coeff).into());
                            }
                        }
                        // Bz
                        let mut bz_coeff = 0;
                        if !constraint.b.terms().is_empty() {
                            bz_coeff = constraint
                                .b
                                .evaluate_row(flattened_polynomials, step_index);
                            if !bz_coeff.is_zero() {
                                coeffs.push((global_index + 1, bz_coeff).into());
                            }
                        }
                        // Cz = Az ⊙ Cz
                        if !az_coeff.is_zero() && !bz_coeff.is_zero() {
                            let cz_coeff = az_coeff * bz_coeff;
                            #[cfg(test)]
                            {
                                if cz_coeff != constraint
                                    .c
                                    .evaluate_row(flattened_polynomials, step_index) {
                                        let mut constraint_string = String::new();
                                        let _ = constraint
                                            .pretty_fmt::<4, JoltR1CSInputs, F>(
                                                &mut constraint_string,
                                                flattened_polynomials,
                                                step_index,
                                            );
                                        println!("{constraint_string}");
                                        panic!(
                                            "Uniform constraint {constraint_index} violated at step {step_index}",
                                        );
                                    }
                            }
                            coeffs.push((global_index + 2, cz_coeff).into());
                        }
                    }

                    // For the final step we will not compute the offset terms, and will assume the condition to be set to 0
                    let next_step_index = if step_index + 1 < num_steps {
                        Some(step_index + 1)
                    } else {
                        None
                    };

                    // Cross-step constraints
                    for (constraint_index, constraint) in cross_step_constraints.iter().enumerate()
                    {
                        let global_index = 3
                            * (step_index * padded_num_constraints
                                + uniform_constraints.len()
                                + constraint_index);

                        // Az
                        let eq_a_eval = eval_offset_lc(
                            &constraint.a,
                            flattened_polynomials,
                            step_index,
                            next_step_index,
                        );
                        let eq_b_eval = eval_offset_lc(
                            &constraint.b,
                            flattened_polynomials,
                            step_index,
                            next_step_index,
                        );
                        let az_coeff = eq_a_eval - eq_b_eval;
                        if !az_coeff.is_zero() {
                            coeffs.push((global_index, az_coeff).into());
                            // If Az != 0, then the condition must be false (i.e. Bz = 0)
                            #[cfg(test)]
                            {
                                let bz_coeff = eval_offset_lc(
                                    &constraint.cond,
                                    flattened_polynomials,
                                    step_index,
                                    next_step_index,
                                );
                                assert_eq!(bz_coeff, 0, "Cross-step constraint {constraint_index} violated at step {step_index}");
                            }
                        } else {
                            // Bz
                            let bz_coeff = eval_offset_lc(
                                &constraint.cond,
                                flattened_polynomials,
                                step_index,
                                next_step_index,
                            );
                            if !bz_coeff.is_zero() {
                                coeffs.push((global_index + 1, bz_coeff).into());
                            }
                        }
                        // Cz is always 0 for cross-step constraints
                    }
                }

                coeffs
            })
            .collect();

        #[cfg(test)]
        {
            // Check that indices are monotonically increasing
            let mut prev_index = unbound_coeffs[0].index;
            for coeff in unbound_coeffs[1..].iter() {
                assert!(coeff.index > prev_index);
                prev_index = coeff.index;
            }
        }

        Self {
            unbound_coeffs,
            bound_coeffs: vec![],
            binding_scratch_space: vec![],
            dense_len: num_steps * padded_num_constraints,
        }
    }

    #[cfg(test)]
    fn uninterleave(&self) -> (DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>) {
        let mut az = vec![F::zero(); self.dense_len];
        let mut bz = vec![F::zero(); self.dense_len];
        let mut cz = vec![F::zero(); self.dense_len];

        if !self.is_bound() {
            for coeff in &self.unbound_coeffs {
                match coeff.index % 3 {
                    0 => az[coeff.index / 3] = F::from_i128(coeff.value),
                    1 => bz[coeff.index / 3] = F::from_i128(coeff.value),
                    2 => cz[coeff.index / 3] = F::from_i128(coeff.value),
                    _ => unreachable!(),
                }
            }
        } else {
            for coeff in &self.bound_coeffs {
                match coeff.index % 3 {
                    0 => az[coeff.index / 3] = coeff.value,
                    1 => bz[coeff.index / 3] = coeff.value,
                    2 => cz[coeff.index / 3] = coeff.value,
                    _ => unreachable!(),
                }
            }
        }
        (
            DensePolynomial::new(az),
            DensePolynomial::new(bz),
            DensePolynomial::new(cz),
        )
    }

    pub fn is_bound(&self) -> bool {
        !self.bound_coeffs.is_empty()
    }

    /// The first round of the first Spartan sumcheck. Since the polynomials
    /// are still unbound at the beginning of this round, we can replace some
    /// of the field arithmetic with `i128` arithmetic.
    pub fn first_sumcheck_round<ProofTranscript: Transcript>(
        &mut self,
        eq_poly: &mut SplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r: &mut Vec<F>,
        polys: &mut Vec<CompressedUniPoly<F>>,
        claim: &mut F,
    ) {
        assert!(!self.is_bound());

        // In order to parallelize, we do a first pass over the coefficients to
        // determine how to divide it into chunks that can be processed independently.
        // In particular, coefficients whose indices are the same modulo 6 cannot
        // be processed independently.
        let block_size = self
            .unbound_coeffs
            .len()
            .div_ceil(rayon::current_num_threads())
            .next_multiple_of(6);
        let chunks: Vec<_> = self
            .unbound_coeffs
            .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
            .collect();

        // We start by computing the E1 evals:
        // (1 - j) * E1[0, x1] + j * E1[1, x1]
        let E1_evals: Vec<_> = eq_poly.E1[..eq_poly.E1_len]
            .par_chunks(2)
            .map(|E1_chunk| {
                let eval_point_0 = E1_chunk[0];
                let m_eq = E1_chunk[1] - E1_chunk[0];
                let eval_point_2 = E1_chunk[1] + m_eq;
                let eval_point_3 = eval_point_2 + m_eq;
                (eval_point_0, eval_point_2, eval_point_3)
            })
            .collect();

        let num_x1_bits = eq_poly.E1_len.log_2() - 1;
        let x1_bitmask = (1 << num_x1_bits) - 1;

        let evals: (F, F, F) = chunks
            .par_iter()
            .map(|chunk| {
                let mut eval_point_0 = F::zero();
                let mut eval_point_2 = F::zero();
                let mut eval_point_3 = F::zero();

                let mut inner_sums = (F::zero(), F::zero(), F::zero());
                let mut prev_x2 = 0;

                for sparse_block in chunk.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                    let block_index = sparse_block[0].index / 6;
                    let x1 = block_index & x1_bitmask;
                    let E1_evals = E1_evals[x1];
                    let x2 = block_index >> num_x1_bits;

                    if x2 != prev_x2 {
                        eval_point_0 += eq_poly.E2[prev_x2] * inner_sums.0;
                        eval_point_2 += eq_poly.E2[prev_x2] * inner_sums.1;
                        eval_point_3 += eq_poly.E2[prev_x2] * inner_sums.2;

                        inner_sums = (F::zero(), F::zero(), F::zero());
                        prev_x2 = x2;
                    }

                    let mut block = [0; 6];
                    for coeff in sparse_block {
                        block[coeff.index % 6] = coeff.value;
                    }

                    let az = (block[0], block[3]);
                    let bz = (block[1], block[4]);
                    let cz = (block[2], block[5]);

                    let m_az = az.1 - az.0;
                    let m_bz = bz.1 - bz.0;
                    let m_cz = cz.1 - cz.0;

                    let az_eval_2 = az.1 + m_az;
                    let az_eval_3 = az_eval_2 + m_az;

                    let bz_eval_2 = bz.1 + m_bz;
                    let bz_eval_3 = bz_eval_2 + m_bz;

                    let cz_eval_2 = cz.1 + m_cz;
                    let cz_eval_3 = cz_eval_2 + m_cz;

                    // TODO(moodlezoup): optimize
                    inner_sums.0 += E1_evals.0.mul_i128(az.0 * bz.0 - cz.0);
                    inner_sums.1 += E1_evals.1.mul_i128(az_eval_2 * bz_eval_2 - cz_eval_2);
                    inner_sums.2 += E1_evals.2.mul_i128(az_eval_3 * bz_eval_3 - cz_eval_3);
                }

                eval_point_0 += eq_poly.E2[prev_x2] * inner_sums.0;
                eval_point_2 += eq_poly.E2[prev_x2] * inner_sums.1;
                eval_point_3 += eq_poly.E2[prev_x2] * inner_sums.2;

                (eval_point_0, eval_point_2, eval_point_3)
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
            );

        // Note: s_1(0) = 0 since the R1CS relation is supposed to hold
        assert_eq!(evals.0, F::zero());

        let cubic_evals = [evals.0, /* 0 */ -evals.0, evals.1, evals.2];
        let cubic_poly = UniPoly::from_evals(&cubic_evals);

        let compressed_poly = cubic_poly.compress();

        // append the prover's message to the transcript
        compressed_poly.append_to_transcript(transcript);

        // derive the verifier's challenge for the next round
        let r_i = transcript.challenge_scalar();
        r.push(r_i);
        polys.push(compressed_poly);

        // Set up next round
        *claim = cubic_poly.evaluate(&r_i);

        // Bind polynomials
        eq_poly.bind(r_i);

        #[cfg(test)]
        let (mut az, mut bz, mut cz) = self.uninterleave();

        // Compute the number of non-zero bound coefficients that will be produced
        // per chunk.
        let output_sizes: Vec<_> = chunks
            .par_iter()
            .map(|chunk| Self::binding_output_length(chunk))
            .collect();

        let total_output_len = output_sizes.iter().sum();
        self.bound_coeffs = Vec::with_capacity(total_output_len);
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.bound_coeffs.set_len(total_output_len);
        }
        let mut output_slices: Vec<&mut [SparseCoefficient<F>]> = Vec::with_capacity(chunks.len());
        let mut remainder = self.bound_coeffs.as_mut_slice();
        for slice_len in output_sizes {
            let (first, second) = remainder.split_at_mut(slice_len);
            output_slices.push(first);
            remainder = second;
        }
        debug_assert_eq!(remainder.len(), 0);

        chunks
            .par_iter()
            .zip_eq(output_slices.into_par_iter())
            .for_each(|(unbound_coeffs, output_slice)| {
                let mut output_index = 0;
                for block in unbound_coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                    let block_index = block[0].index / 6;

                    let mut az_coeff: (Option<i128>, Option<i128>) = (None, None);
                    let mut bz_coeff: (Option<i128>, Option<i128>) = (None, None);
                    let mut cz_coeff: (Option<i128>, Option<i128>) = (None, None);

                    for coeff in block {
                        match coeff.index % 6 {
                            0 => az_coeff.0 = Some(coeff.value),
                            1 => bz_coeff.0 = Some(coeff.value),
                            2 => cz_coeff.0 = Some(coeff.value),
                            3 => az_coeff.1 = Some(coeff.value),
                            4 => bz_coeff.1 = Some(coeff.value),
                            5 => cz_coeff.1 = Some(coeff.value),
                            _ => unreachable!(),
                        }
                    }
                    if az_coeff != (None, None) {
                        let (low, high) = (az_coeff.0.unwrap_or(0), az_coeff.1.unwrap_or(0));
                        output_slice[output_index] = (
                            3 * block_index,
                            F::from_i128(low) + r_i.mul_i128(high - low),
                        )
                            .into();
                        output_index += 1;
                    }
                    if bz_coeff != (None, None) {
                        let (low, high) = (bz_coeff.0.unwrap_or(0), bz_coeff.1.unwrap_or(0));
                        output_slice[output_index] = (
                            3 * block_index + 1,
                            F::from_i128(low) + r_i.mul_i128(high - low),
                        )
                            .into();
                        output_index += 1;
                    }
                    if cz_coeff != (None, None) {
                        let (low, high) = (cz_coeff.0.unwrap_or(0), cz_coeff.1.unwrap_or(0));
                        output_slice[output_index] = (
                            3 * block_index + 2,
                            F::from_i128(low) + r_i.mul_i128(high - low),
                        )
                            .into();
                        output_index += 1;
                    }
                }
                debug_assert_eq!(output_index, output_slice.len())
            });
        self.dense_len /= 2;

        #[cfg(test)]
        {
            // Check that the binding is consistent with binding
            // Az, Bz, Cz individually
            let (az_bound, bz_bound, cz_bound) = self.uninterleave();
            az.bound_poly_var_bot(&r_i);
            bz.bound_poly_var_bot(&r_i);
            cz.bound_poly_var_bot(&r_i);
            for i in 0..az.len() {
                if az_bound[i] != az[i] {
                    println!("{i} {} != {}", az_bound[i], az[i]);
                }
            }
            assert!(az_bound.Z[..az_bound.len()] == az.Z[..az.len()]);
            assert!(bz_bound.Z[..bz_bound.len()] == bz.Z[..bz.len()]);
            assert!(cz_bound.Z[..cz_bound.len()] == cz.Z[..cz.len()]);
        }
    }

    /// All subsequent rounds of the first Spartan sumcheck.
    pub fn subsequent_sumcheck_round<ProofTranscript: Transcript>(
        &mut self,
        eq_poly: &mut SplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r: &mut Vec<F>,
        polys: &mut Vec<CompressedUniPoly<F>>,
        claim: &mut F,
    ) {
        assert!(self.is_bound());

        // In order to parallelize, we do a first pass over the coefficients to
        // determine how to divide it into chunks that can be processed independently.
        // In particular, coefficients whose indices are the same modulo 6 cannot
        // be processed independently.
        let block_size = self
            .bound_coeffs
            .len()
            .div_ceil(rayon::current_num_threads())
            .next_multiple_of(6);
        let chunks: Vec<_> = self
            .bound_coeffs
            .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
            .collect();

        let cubic_poly = if eq_poly.E1_len == 1 {
            let eq_evals: Vec<(F, F, F)> = eq_poly.E2[..eq_poly.E2_len]
                .par_chunks(2)
                .map(|eq_chunk| {
                    let eval_point_0 = eq_chunk[0];
                    let m_eq = eq_chunk[1] - eq_chunk[0];
                    let eval_point_2 = eq_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();

            let evals: (F, F, F) = chunks
                .par_iter()
                .flat_map_iter(|chunk| {
                    chunk
                        .chunk_by(|x, y| x.index / 6 == y.index / 6)
                        .map(|sparse_block| {
                            let block_index = sparse_block[0].index / 6;
                            let mut block = [F::zero(); 6];
                            for coeff in sparse_block {
                                block[coeff.index % 6] = coeff.value;
                            }

                            let az = (block[0], block[3]);
                            let bz = (block[1], block[4]);
                            let cz = (block[2], block[5]);

                            let m_az = az.1 - az.0;
                            let m_bz = bz.1 - bz.0;
                            let m_cz = cz.1 - cz.0;

                            let az_eval_2 = az.1 + m_az;
                            let az_eval_3 = az_eval_2 + m_az;

                            let bz_eval_2 = bz.1 + m_bz;
                            let bz_eval_3 = bz_eval_2 + m_bz;

                            let cz_eval_2 = cz.1 + m_cz;
                            let cz_eval_3 = cz_eval_2 + m_cz;

                            let eq_evals = eq_evals[block_index];

                            (
                                eq_evals
                                    .0
                                    .mul_0_optimized(az.0.mul_0_optimized(bz.0) - cz.0),
                                eq_evals.1.mul_0_optimized(
                                    az_eval_2.mul_0_optimized(bz_eval_2) - cz_eval_2,
                                ),
                                eq_evals.2.mul_0_optimized(
                                    az_eval_3.mul_0_optimized(bz_eval_3) - cz_eval_3,
                                ),
                            )
                        })
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );

            let cubic_evals = [evals.0, *claim - evals.0, evals.1, evals.2];
            UniPoly::from_evals(&cubic_evals)
        } else {
            // We start by computing the E1 evals:
            // (1 - j) * E1[0, x1] + j * E1[1, x1]
            let E1_evals: Vec<_> = eq_poly.E1[..eq_poly.E1_len]
                .par_chunks(2)
                .map(|E1_chunk| {
                    let eval_point_0 = E1_chunk[0];
                    let m_eq = E1_chunk[1] - E1_chunk[0];
                    let eval_point_2 = E1_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();

            let num_x1_bits = eq_poly.E1_len.log_2() - 1;
            let x1_bitmask = (1 << num_x1_bits) - 1;

            let evals: (F, F, F) = chunks
                .par_iter()
                .map(|chunk| {
                    let mut eval_point_0 = F::zero();
                    let mut eval_point_2 = F::zero();
                    let mut eval_point_3 = F::zero();

                    let mut inner_sums = (F::zero(), F::zero(), F::zero());
                    let mut prev_x2 = 0;

                    for sparse_block in chunk.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                        let block_index = sparse_block[0].index / 6;
                        let x1 = block_index & x1_bitmask;
                        let E1_evals = E1_evals[x1];
                        let x2 = block_index >> num_x1_bits;

                        if x2 != prev_x2 {
                            eval_point_0 += eq_poly.E2[prev_x2] * inner_sums.0;
                            eval_point_2 += eq_poly.E2[prev_x2] * inner_sums.1;
                            eval_point_3 += eq_poly.E2[prev_x2] * inner_sums.2;

                            inner_sums = (F::zero(), F::zero(), F::zero());
                            prev_x2 = x2;
                        }

                        let mut block = [F::zero(); 6];
                        for coeff in sparse_block {
                            block[coeff.index % 6] = coeff.value;
                        }

                        let az = (block[0], block[3]);
                        let bz = (block[1], block[4]);
                        let cz = (block[2], block[5]);

                        let m_az = az.1 - az.0;
                        let m_bz = bz.1 - bz.0;
                        let m_cz = cz.1 - cz.0;

                        let az_eval_2 = az.1 + m_az;
                        let az_eval_3 = az_eval_2 + m_az;

                        let bz_eval_2 = bz.1 + m_bz;
                        let bz_eval_3 = bz_eval_2 + m_bz;

                        let cz_eval_2 = cz.1 + m_cz;
                        let cz_eval_3 = cz_eval_2 + m_cz;

                        inner_sums.0 += E1_evals
                            .0
                            .mul_0_optimized(az.0.mul_0_optimized(bz.0) - cz.0);
                        inner_sums.1 += E1_evals
                            .1
                            .mul_0_optimized(az_eval_2.mul_0_optimized(bz_eval_2) - cz_eval_2);
                        inner_sums.2 += E1_evals
                            .2
                            .mul_0_optimized(az_eval_3.mul_0_optimized(bz_eval_3) - cz_eval_3);
                    }

                    eval_point_0 += eq_poly.E2[prev_x2] * inner_sums.0;
                    eval_point_2 += eq_poly.E2[prev_x2] * inner_sums.1;
                    eval_point_3 += eq_poly.E2[prev_x2] * inner_sums.2;

                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );

            let cubic_evals = [evals.0, *claim - evals.0, evals.1, evals.2];
            UniPoly::from_evals(&cubic_evals)
        };

        let compressed_poly = cubic_poly.compress();

        // append the prover's message to the transcript
        compressed_poly.append_to_transcript(transcript);

        // derive the verifier's challenge for the next round
        let r_i = transcript.challenge_scalar();
        r.push(r_i);
        polys.push(compressed_poly);

        // Set up next round
        *claim = cubic_poly.evaluate(&r_i);

        // Bind polynomials
        eq_poly.bind(r_i);

        #[cfg(test)]
        let (mut az, mut bz, mut cz) = self.uninterleave();

        let output_sizes: Vec<_> = chunks
            .par_iter()
            .map(|chunk| Self::binding_output_length(chunk))
            .collect();

        let total_output_len = output_sizes.iter().sum();
        if self.binding_scratch_space.is_empty() {
            self.binding_scratch_space = Vec::with_capacity(total_output_len);
        }
        unsafe {
            self.binding_scratch_space.set_len(total_output_len);
        }

        let mut output_slices: Vec<&mut [SparseCoefficient<F>]> = Vec::with_capacity(chunks.len());
        let mut remainder = self.binding_scratch_space.as_mut_slice();
        for slice_len in output_sizes {
            let (first, second) = remainder.split_at_mut(slice_len);
            output_slices.push(first);
            remainder = second;
        }
        debug_assert_eq!(remainder.len(), 0);

        chunks
            .par_iter()
            .zip_eq(output_slices.into_par_iter())
            .for_each(|(coeffs, output_slice)| {
                let mut output_index = 0;
                for block in coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                    let block_index = block[0].index / 6;

                    let mut az_coeff: (Option<F>, Option<F>) = (None, None);
                    let mut bz_coeff: (Option<F>, Option<F>) = (None, None);
                    let mut cz_coeff: (Option<F>, Option<F>) = (None, None);

                    for coeff in block {
                        match coeff.index % 6 {
                            0 => az_coeff.0 = Some(coeff.value),
                            1 => bz_coeff.0 = Some(coeff.value),
                            2 => cz_coeff.0 = Some(coeff.value),
                            3 => az_coeff.1 = Some(coeff.value),
                            4 => bz_coeff.1 = Some(coeff.value),
                            5 => cz_coeff.1 = Some(coeff.value),
                            _ => unreachable!(),
                        }
                    }
                    if az_coeff != (None, None) {
                        let (low, high) = (
                            az_coeff.0.unwrap_or(F::zero()),
                            az_coeff.1.unwrap_or(F::zero()),
                        );
                        output_slice[output_index] =
                            (3 * block_index, low + r_i * (high - low)).into();
                        output_index += 1;
                    }
                    if bz_coeff != (None, None) {
                        let (low, high) = (
                            bz_coeff.0.unwrap_or(F::zero()),
                            bz_coeff.1.unwrap_or(F::zero()),
                        );
                        output_slice[output_index] =
                            (3 * block_index + 1, low + r_i * (high - low)).into();
                        output_index += 1;
                    }
                    if cz_coeff != (None, None) {
                        let (low, high) = (
                            cz_coeff.0.unwrap_or(F::zero()),
                            cz_coeff.1.unwrap_or(F::zero()),
                        );
                        output_slice[output_index] =
                            (3 * block_index + 2, low + r_i * (high - low)).into();
                        output_index += 1;
                    }
                }
                debug_assert_eq!(output_index, output_slice.len())
            });

        std::mem::swap(&mut self.bound_coeffs, &mut self.binding_scratch_space);
        self.dense_len /= 2;

        #[cfg(test)]
        {
            // Check that the binding is consistent with binding
            // Az, Bz, Cz individually
            let (az_bound, bz_bound, cz_bound) = self.uninterleave();
            az.bound_poly_var_bot(&r_i);
            bz.bound_poly_var_bot(&r_i);
            cz.bound_poly_var_bot(&r_i);
            assert!(az_bound.Z[..az_bound.len()] == az.Z[..az.len()]);
            assert!(bz_bound.Z[..bz_bound.len()] == bz.Z[..bz.len()]);
            assert!(cz_bound.Z[..cz_bound.len()] == cz.Z[..cz.len()]);
        }
    }

    /// Computes the number of non-zero coefficients that would result from
    /// binding the given slice of coefficients.
    fn binding_output_length<T>(coeffs: &[SparseCoefficient<T>]) -> usize {
        let mut output_size = 0;
        for block in coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
            let mut Az_coeff_found = false;
            let mut Bz_coeff_found = false;
            let mut Cz_coeff_found = false;
            for coeff in block {
                match coeff.index % 3 {
                    0 => {
                        if !Az_coeff_found {
                            Az_coeff_found = true;
                            output_size += 1;
                        }
                    }
                    1 => {
                        if !Bz_coeff_found {
                            Bz_coeff_found = true;
                            output_size += 1;
                        }
                    }
                    2 => {
                        if !Cz_coeff_found {
                            Cz_coeff_found = true;
                            output_size += 1;
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }

        output_size
    }

    pub fn final_sumcheck_evals(&self) -> [F; 3] {
        let mut final_az_eval = F::zero();
        let mut final_bz_eval = F::zero();
        let mut final_cz_eval = F::zero();
        for i in 0..3 {
            if let Some(coeff) = self.bound_coeffs.get(i) {
                match coeff.index {
                    0 => final_az_eval = coeff.value,
                    1 => final_bz_eval = coeff.value,
                    2 => final_cz_eval = coeff.value,
                    _ => {}
                }
            }
        }

        [final_az_eval, final_bz_eval, final_cz_eval]
    }
}
