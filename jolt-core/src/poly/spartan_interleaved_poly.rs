use super::{
    eq_poly::EqPolynomial,
    multilinear_polynomial::MultilinearPolynomial,
    sparse_interleaved_poly::SparseCoefficient,
    split_eq_poly::{NewSplitEqPolynomial, SplitEqPolynomial},
    unipoly::{CompressedUniPoly, UniPoly},
};
#[cfg(test)]
use crate::poly::dense_mlpoly::DensePolynomial;
#[cfg(test)]
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::subprotocols::sumcheck::process_eq_sumcheck_round;
use crate::{
    field::{JoltField, OptimizedMul},
    r1cs::builder::{eval_offset_lc, Constraint, OffsetEqConstraint},
    utils::{
        math::Math,
        small_value::svo_helpers,
        transcript::{AppendToTranscript, Transcript},
    },
};
use ark_ff::Zero;
use rayon::prelude::*;

use crate::r1cs::spartan::small_value_optimization::NUM_SVO_ROUNDS;

const fn num_non_trivial_ternary_points(num_svo_rounds: usize) -> usize {
    // Returns 3^num_svo_rounds - 2^num_svo_rounds
    let pow_3 = 3_usize
        .checked_pow(num_svo_rounds as u32)
        .expect("Number of ternary points overflowed");
    let pow_2 = 2_usize
        .checked_pow(num_svo_rounds as u32)
        .expect("Number of ternary points overflowed");
    pow_3 - pow_2
}

const fn total_num_accums(num_svo_rounds: usize) -> usize {
    // Compute the sum \sum_{i=1}^{num_svo_rounds} (3^i - 2^i)
    let mut sum = 0;
    let mut i = 1;
    while i <= num_svo_rounds {
        let pow_3 = 3_usize
            .checked_pow(i as u32)
            .expect("Number of ternary points overflowed");
        let pow_2 = 2_usize
            .checked_pow(i as u32)
            .expect("Number of ternary points overflowed");
        sum += pow_3 - pow_2;
        i += 1;
    }
    sum
}

const fn num_accums_eval_zero(num_svo_rounds: usize) -> usize {
    // Returns \sum_{i=0}^{num_svo_rounds - 1} (3^i - 2^i)
    let mut sum = 0;
    let mut i = 0;
    while i < num_svo_rounds {
        let pow_3 = 3_usize
            .checked_pow(i as u32)
            .expect("Number of ternary points overflowed");
        let pow_2 = 2_usize
            .checked_pow(i as u32)
            .expect("Number of binary points overflowed");
        sum += pow_3 - pow_2;
        i += 1;
    }
    sum
}

const fn num_accums_eval_infty(num_svo_rounds: usize) -> usize {
    // Returns \sum_{i=0}^{num_svo_rounds - 1} 3^i
    let mut sum = 0;
    let mut i = 0;
    while i < num_svo_rounds {
        let pow_3 = 3_usize
            .checked_pow(i as u32)
            .expect("Number of ternary points overflowed");
        sum += pow_3;
        i += 1;
    }
    sum
}

pub const TOTAL_NUM_ACCUMS: usize = total_num_accums(NUM_SVO_ROUNDS);
pub const NUM_NONTRIVIAL_TERNARY_POINTS: usize = num_non_trivial_ternary_points(NUM_SVO_ROUNDS);
pub const NUM_ACCUMS_EVAL_ZERO: usize = num_accums_eval_zero(NUM_SVO_ROUNDS);
pub const NUM_ACCUMS_EVAL_INFTY: usize = num_accums_eval_infty(NUM_SVO_ROUNDS);

pub struct NewSpartanInterleavedPolynomial<const NUM_SVO_ROUNDS: usize, F: JoltField> {
    /// A sparse vector representing the (interleaved) coefficients for the Az, Bz polynomials
    /// Generated from binary evaluations. Sorted by index.
    ///
    /// (note: **no** Cz coefficients are stored here, since they are not needed for small value
    /// precomputation, and can be computed on the fly in streaming round)
    pub(crate) ab_unbound_coeffs: Vec<SparseCoefficient<i128>>,

    // pub(crate) az_bound_coeffs: DensePolynomial<F>,
    // pub(crate) bz_bound_coeffs: DensePolynomial<F>,
    // pub(crate) cz_bound_coeffs: DensePolynomial<F>,

    // What if we do sparse coeffs throughout like old method...
    // Really need best performance for the first few rounds
    // and the old methods are already tested and optimized
    bound_coeffs: Vec<SparseCoefficient<F>>,

    binding_scratch_space: Vec<SparseCoefficient<F>>,

    pub(crate) dense_len: usize,
}

impl<const NUM_SVO_ROUNDS: usize, F: JoltField> NewSpartanInterleavedPolynomial<NUM_SVO_ROUNDS, F> {
    /// Compute the unbound coefficients for the Az and Bz polynomials (no Cz coefficients are
    /// needed), along with the accumulators for the small value optimization (SVO) rounds.
    ///
    /// Recall that the accumulators are of the form: accum_i[v_0, ..., v_{i-1}, u] = \sum_{y_rest}
    /// \sum_{x_out} E_out(x_out || y_rest) * \sum_{x_in} E_in(x_in) * P(x_out, x_in, y_rest, u,
    /// v_0, ..., v_{i-1}),
    ///
    /// for all i < NUM_SVO_ROUNDS, v_0,..., v_{i-1} \in {0,1,∞}, u \in {0,∞}, and P(X) = Az(X) *
    /// Bz(X) - Cz(X).
    ///
    /// Note that we have reverse the order of variables from the paper, since in this codebase the
    /// indexing is MSB to LSB (as we go from 0 to N-1, i.e. left to right).
    ///
    /// Note that only the accumulators with at least one infinity among v_j and u are non-zero, so
    /// the fully binary ones do not need to be computed. Plus, the ones with at least one infinity
    /// will NOT have any Cz contributions.
    ///
    /// This is why we do not need to compute the Cz terms in the unbound coefficients.
    ///
    /// The output of the accumulators is ([F; NUM_ACCUMS_EVAL_ZERO]; [F; NUM_ACCUMS_EVAL_INFTY]),
    /// where the outer array is for evals at u = 0 and u = ∞. The inner array contains all non-zero
    /// accumulators across all rounds, concatenated in order.
    ///
    /// For 1 round of small value optimization, this is:
    /// - Eval at zero: empty
    /// - Eval at infty: acc_1(infty)
    ///
    /// For 2 rounds of small value optimization, this is same as 1 round, with addition of:
    /// (recall: we do MSB => LSB, so 0/infty refers to the leftmost variable)
    /// - Eval at zero: acc_2(0, infty)
    /// - Eval at infty: acc_2(infty,0), acc_2(infty,1), acc_2(infty, infty)
    ///
    /// Total = 5 accumulators
    ///
    /// For 3 rounds of small value optimization, this is same as 2 rounds, with addition of:
    /// - Eval at zero: acc_3(0, 0, infty), acc_3(0, 1, infty),
    ///   acc_3(0, infty, 0), acc_3(0, infty, 1), acc_3(0, infty, infty)
    /// - Eval at infty: acc_3(infty, v_1, v_2), where v_1, v_2 \in {0, 1, infty}
    ///
    /// Total = 19 accumulators
    #[tracing::instrument(
        skip_all,
        name = "NewSpartanInterleavedPolynomial::new_with_precompute"
    )]
    pub fn new_with_precompute(
        padded_num_constraints: usize,
        uniform_constraints: &[Constraint],
        cross_step_constraints: &[OffsetEqConstraint],
        flattened_polynomials: &[&MultilinearPolynomial<F>],
        tau: &[F], // Challenges for ALL N_total R1CS variables
    ) -> ([F; NUM_ACCUMS_EVAL_ZERO], [F; NUM_ACCUMS_EVAL_INFTY], Self) {
        let func_span = tracing::info_span!("new_with_precompute_body");
        let _func_guard = func_span.enter();

        let var_setup_span = tracing::debug_span!("variable_setup_and_assertions");
        let _var_setup_guard = var_setup_span.enter();
        // Here's the full layout of the variables:
        // 0 ... (N/2 - l) ... (n_s) ... (N - l) ... (N - i - 1) ... (N - 1)
        // where n_s = num_step_vars, n_c = num_constraint_vars, N = n_s + n_c, l = NUM_SVO_ROUNDS
        // and i is an iterator over 0..l (for the SVO rounds)

        // Within this layout, we have the partition:
        // - 0 ... (N/2 - l) is x_out
        // - (N/2 - l) ... (n_s) is x_in_step
        // - (n_s) ... (N - l) is x_in_constraint (i.e. non_svo_constraint)
        // - (N/2 - l) ... (N - l) in total is x_in
        // - (N - l) ... (N - i - 1) is y_suffix_svo
        // - (N - i - 1) ... (N - 1) is u || v_config

        assert!(
            NUM_SVO_ROUNDS <= 3,
            "NUM_SVO_ROUNDS ({}) must be <= 3",
            NUM_SVO_ROUNDS
        );
        assert!(
            NUM_SVO_ROUNDS > 0,
            "NUM_SVO_ROUNDS ({}) must be > 0",
            NUM_SVO_ROUNDS
        );

        // --- Variable Definitions ---
        let num_steps = flattened_polynomials[0].len();
        let num_step_vars = if num_steps > 0 { num_steps.log_2() } else { 0 };
        let num_constraint_vars = if padded_num_constraints > 0 {
            padded_num_constraints.log_2()
        } else {
            0
        };
        let total_num_vars = num_step_vars + num_constraint_vars; // N_total (or l in Algo 6)

        assert_eq!(
            tau.len(),
            total_num_vars,
            "tau length ({}) mismatch with R1CS variable count (step_vars {} + constraint_vars {})",
            tau.len(),
            num_step_vars,
            num_constraint_vars
        );
        assert!(
            NUM_SVO_ROUNDS <= num_constraint_vars,
            "NUM_SVO_ROUNDS ({}) cannot exceed total constraint variables ({})",
            NUM_SVO_ROUNDS,
            num_constraint_vars
        );

        // Number of constraint variables that are NOT part of the SVO prefix Y.
        let num_non_svo_constraint_vars = num_constraint_vars.saturating_sub(NUM_SVO_ROUNDS);
        let num_non_svo_z_vars = num_step_vars + num_non_svo_constraint_vars;
        assert_eq!(
            num_non_svo_z_vars,
            total_num_vars - NUM_SVO_ROUNDS,
            "num_non_svo_z_vars ({}) + NUM_SVO_ROUNDS ({}) must be == total_num_vars ({})",
            num_non_svo_z_vars,
            NUM_SVO_ROUNDS,
            total_num_vars
        );

        // --- Define Iteration Spaces for Non-SVO Z variables (x_out_val, x_in_val) ---
        let potential_x_out_vars = total_num_vars / 2 - NUM_SVO_ROUNDS;
        let iter_num_x_out_vars = std::cmp::min(potential_x_out_vars, num_step_vars);
        let iter_num_x_in_vars = num_non_svo_z_vars - iter_num_x_out_vars;
        let iter_num_x_in_step_vars = num_step_vars - iter_num_x_out_vars;
        let iter_num_x_in_constraint_vars = num_non_svo_constraint_vars;
        assert_eq!(
            iter_num_x_in_vars,
            iter_num_x_in_step_vars + iter_num_x_in_constraint_vars
        );
        assert_eq!(num_non_svo_z_vars, iter_num_x_out_vars + iter_num_x_in_vars);

        drop(_var_setup_guard);

        let eq_setup_span = tracing::debug_span!("eq_poly_setup");
        let _eq_setup_guard = eq_setup_span.enter();
        // --- Setup: E_in and E_out tables ---
        // Call NewSplitEqPolynomial::new_for_small_value with the determined variable splits.
        let eq_poly = NewSplitEqPolynomial::new_for_small_value(
            tau,
            iter_num_x_out_vars,
            iter_num_x_in_vars,
            NUM_SVO_ROUNDS,
        );
        let E_in_evals = eq_poly.E_in_current();
        let E_out_vec = &eq_poly.E_out_vec;

        assert_eq!(E_out_vec.len(), NUM_SVO_ROUNDS);

        let num_x_out_vals = 1usize << iter_num_x_out_vars;
        let num_x_in_vals = 1 << iter_num_x_in_vars;

        assert_eq!(
            num_x_in_vals,
            E_in_evals.len(),
            "num_x_in_vals ({}) != E_in_evals.len ({})",
            num_x_in_vals,
            E_in_evals.len()
        );

        drop(_eq_setup_guard);

        let main_fold_span = tracing::info_span!("parallel_fold_reduce_x_out");
        let _main_fold_guard = main_fold_span.enter();
        // --- Parallel Fold-Reduce over x_out_val ---
        // Corresponds to Algo 6, Line 7: Outer loop over x_out.

        // Define the structure returned by each parallel map task
        struct PrecomputeTaskOutput<F: JoltField> {
            ab_coeffs_local: Vec<SparseCoefficient<i128>>,
            svo_accums_zero_local: [F; NUM_ACCUMS_EVAL_ZERO],
            svo_accums_infty_local: [F; NUM_ACCUMS_EVAL_INFTY],
        }

        let num_parallel_chunks = rayon::current_num_threads().next_power_of_two() * 2;
        let x_out_chunk_size = num_x_out_vals.div_ceil(num_parallel_chunks);

        let collected_chunk_outputs: Vec<PrecomputeTaskOutput<F>> = (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| { 
                let x_out_task_span = tracing::debug_span!("chunk_task", chunk_idx);
                let _x_out_task_guard = x_out_task_span.enter();

                let mut chunk_ab_coeffs = Vec::new(); 
                let mut chunk_svo_accums_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
                let mut chunk_svo_accums_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];

                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = std::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);

                for x_out_val in x_out_start..x_out_end {
                    // Accumulator for SUM_{x_in} E_in * P_ext for this specific x_out_val.
                    let mut task_tA_accumulator_vec = [F::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];
                    let mut current_x_out_svo_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
                    let mut current_x_out_svo_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];

                    // --- Inner Loop over x_in_val ---
                    for x_in_val in 0..num_x_in_vals {
                        let x_in_step_part = x_in_val >> iter_num_x_in_constraint_vars;
                        let current_step_idx = (x_out_val << iter_num_x_in_step_vars) | x_in_step_part;
                        let constraint_mask = (1 << iter_num_x_in_constraint_vars) - 1;
                        let current_lower_bits_val = x_in_val & constraint_mask;
                        let mut binary_az_evals: Vec<i128> = vec![0i128; 1 << NUM_SVO_ROUNDS];
                        let mut binary_bz_evals: Vec<i128> = vec![0i128; 1 << NUM_SVO_ROUNDS];

                        for y_svo_binary_prefix_val in 0..(1 << NUM_SVO_ROUNDS) {
                            let constraint_idx_within_step =
                                (current_lower_bits_val << NUM_SVO_ROUNDS) + y_svo_binary_prefix_val;
                            let global_r1cs_idx =
                                2 * (current_step_idx * padded_num_constraints + constraint_idx_within_step);

                            assert!(constraint_idx_within_step < padded_num_constraints,
                                "Constraint index must be less than the padded number of constraints!");

                            if constraint_idx_within_step < uniform_constraints.len() {
                                let constraint = &uniform_constraints[constraint_idx_within_step];
                                if !constraint.a.terms().is_empty() {
                                    let az_i128 = constraint
                                        .a
                                        .evaluate_row(flattened_polynomials, current_step_idx);
                                    if !az_i128.is_zero() {
                                        binary_az_evals[y_svo_binary_prefix_val] = az_i128;
                                        chunk_ab_coeffs.push((global_r1cs_idx, az_i128).into());
                                    }
                                }
                                if !constraint.b.terms().is_empty() {
                                    let bz_i128 = constraint
                                        .b
                                        .evaluate_row(flattened_polynomials, current_step_idx);
                                    if !bz_i128.is_zero() {
                                        binary_bz_evals[y_svo_binary_prefix_val] = bz_i128;
                                        chunk_ab_coeffs.push((global_r1cs_idx + 1, bz_i128).into());
                                    }
                                }
                            } else if constraint_idx_within_step < uniform_constraints.len() + cross_step_constraints.len() {
                                let cross_step_constraint_idx = constraint_idx_within_step - uniform_constraints.len();
                                let constraint = &cross_step_constraints[cross_step_constraint_idx];
                                let next_step_index_opt = if current_step_idx + 1 < num_steps {
                                    Some(current_step_idx + 1)
                                } else {
                                    None
                                };
                                let eq_a_eval = eval_offset_lc(
                                    &constraint.a,
                                    flattened_polynomials,
                                    current_step_idx,
                                    next_step_index_opt,
                                );
                                let eq_b_eval = eval_offset_lc(
                                    &constraint.b,
                                    flattened_polynomials,
                                    current_step_idx,
                                    next_step_index_opt,
                                );
                                let az_i128 = eq_a_eval - eq_b_eval;
                                if !az_i128.is_zero() {
                                    binary_az_evals[y_svo_binary_prefix_val] = az_i128;
                                    chunk_ab_coeffs.push((global_r1cs_idx, az_i128).into());
                                } else {
                                    let bz_i128 = eval_offset_lc(
                                        &constraint.cond,
                                        flattened_polynomials,
                                        current_step_idx,
                                        next_step_index_opt,
                                    );
                                    if !bz_i128.is_zero() {
                                        binary_bz_evals[y_svo_binary_prefix_val] = bz_i128;
                                        chunk_ab_coeffs.push((global_r1cs_idx + 1, bz_i128).into());
                                    }
                                }
                            }
                        }
                        let E_in_val_for_current_x_in = &E_in_evals[x_in_val];
                        svo_helpers::compute_and_update_tA_inplace_generic::<NUM_SVO_ROUNDS, F>(
                            &binary_az_evals,
                            &binary_bz_evals,
                            E_in_val_for_current_x_in,
                            &mut task_tA_accumulator_vec,
                        );
                    } // End inner loop over x_in_val

                    svo_helpers::distribute_tA_to_svo_accumulators_generic::<NUM_SVO_ROUNDS, F>(
                        &task_tA_accumulator_vec,
                        x_out_val, // Use the specific x_out_val for this iteration
                        E_out_vec,
                        &mut current_x_out_svo_zero, 
                        &mut current_x_out_svo_infty,
                    );

                    // Accumulate SVO contributions for this x_out_val into chunk accumulators
                    for i in 0..NUM_ACCUMS_EVAL_ZERO {
                        chunk_svo_accums_zero[i] += current_x_out_svo_zero[i];
                    }
                    for i in 0..NUM_ACCUMS_EVAL_INFTY {
                        chunk_svo_accums_infty[i] += current_x_out_svo_infty[i];
                    }
                } // End loop over x_out_val in chunk

                drop(_x_out_task_guard);
                PrecomputeTaskOutput {
                    ab_coeffs_local: chunk_ab_coeffs,
                    svo_accums_zero_local: chunk_svo_accums_zero,
                    svo_accums_infty_local: chunk_svo_accums_infty,
                }
            }) // End .map() over chunks
            .collect(); // Collect all chunk outputs
        
        drop(_main_fold_guard);

        // --- Finalization ---
        let finalization_span = tracing::info_span!("finalization");
        let _finalization_guard = finalization_span.enter();

        let total_ab_coeffs_len = collected_chunk_outputs
            .iter()
            .map(|output| output.ab_coeffs_local.len())
            .sum();
        
        let mut final_ab_unbound_coeffs = Vec::with_capacity(total_ab_coeffs_len);
        let mut final_svo_accums_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
        let mut final_svo_accums_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];

        for task_output in collected_chunk_outputs {
            final_ab_unbound_coeffs.extend(task_output.ab_coeffs_local);
            for idx in 0..NUM_ACCUMS_EVAL_ZERO {
                final_svo_accums_zero[idx] += task_output.svo_accums_zero_local[idx];
            }
            for idx in 0..NUM_ACCUMS_EVAL_INFTY {
                final_svo_accums_infty[idx] += task_output.svo_accums_infty_local[idx];
            }
        }
        
        // final_ab_unbound_coeffs is now fully populated and SVO accumulators are summed.

        // Try commenting this for now
        // Get final flat list of sparse Az/Bz coefficients from reduction result.
        // Sort the combined list globally by R1CS index.
        // final_ab_unbound_coeffs.sort_by_key(|sc| sc.index);

        // If this is not sorted, revert to `flat_map_iter` as `new()`, which gives a vector
        // of acc_res, which we then reduce (in parallel)

        // Debug check for sortedness
        #[cfg(test)]
        {
            if NUM_SVO_ROUNDS > 0 && !final_ab_unbound_coeffs.is_empty() {
                let mut prev_index = final_ab_unbound_coeffs[0].index;
                for coeff in final_ab_unbound_coeffs.iter().skip(1) {
                    assert!(
                        coeff.index > prev_index,
                        "Indices not monotonically increasing in final_ab_unbound_coeffs: prev {}, current {}",
                        prev_index, coeff.index
                    );
                    prev_index = coeff.index;
                }
            }
            println!("Sortedness check passed!");
        }

        #[cfg(test)]
        {
            let old_new_result = SpartanInterleavedPolynomial::new(
                uniform_constraints,
                cross_step_constraints,
                flattened_polynomials,
                padded_num_constraints,
            );
            // Check that the Az Bz coeffs are the same (note that the old result also contains Cz coeffs)
        }

        drop(_finalization_guard);

        // Return final SVO accumulators and Self struct.
        // Corresponds to Algo 6, Line 15: Return {A_i(v,u)}.
        (
            final_svo_accums_zero,
            final_svo_accums_infty,
            Self {
                ab_unbound_coeffs: final_ab_unbound_coeffs,
                bound_coeffs: vec![],
                binding_scratch_space: vec![],
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
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] * (unbound_coeffs_a(x_out, x_in,
    /// 0, r) * unbound_coeffs_b(x_out, x_in, 0, r) - unbound_coeffs_c(x_out, x_in, 0, r))`
    ///
    /// and `t_i(∞) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] * (unbound_coeffs_a(x_out,
    /// x_in, ∞, r) * unbound_coeffs_b(x_out, x_in, ∞, r))`
    ///
    /// Here the "_a,b,c" subscript indicates the coefficients of `unbound_coeffs` corresponding to
    /// Az, Bz, Cz respectively. Note that we index with x_out being the MSB here.
    ///
    /// Importantly, since the eval at `r` is not cached, we will need to recompute it via another
    /// sum
    ///
    /// `unbound_coeffs_{a,b,c}(x_out, x_in, {0,∞}, r) = \sum_{binary y} eq(r, y) *
    /// unbound_coeffs_{a,b,c}(x_out, x_in, {0,∞}, y)`
    ///
    /// (and the eval at ∞ is computed as (eval at 1) - (eval at 0))
    ///
    /// Since `unbound_coeffs` are in sparse format, we will need to be more careful with indexing;
    /// see the old implementation for details.
    ///
    /// Finally, as we compute each `unbound_coeffs_{a,b,c}(x_out, x_in, {0,∞}, r)`, we will
    /// store them in `bound_coeffs`. which is still in sparse format (the eval at 1 will be eval
    /// at 0 + eval at ∞). We then derive the next challenge from the transcript, and bind these
    /// bound coeffs for the next round.
    #[tracing::instrument(
        skip_all,
        name = "NewSpartanInterleavedPolynomial::streaming_sumcheck_round"
    )]
    pub fn streaming_sumcheck_round<ProofTranscript: Transcript>(
        &mut self,
        eq_poly: &mut NewSplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r_challenges: &mut Vec<F>,
        round_polys: &mut Vec<CompressedUniPoly<F>>,
        claim: &mut F,
    ) {
        let top_level_span = tracing::span!(tracing::Level::INFO, "streaming_sumcheck_round_body");
        let _top_level_guard = top_level_span.enter();

        let setup_span = tracing::span!(tracing::Level::DEBUG, "streaming_round_setup");
        let _setup_guard = setup_span.enter();

        // --- 1. Initial Setup ---
        let num_y_svo_vars = r_challenges.len();
        let N_high = 1 << num_y_svo_vars;
        let eq_r = EqPolynomial::evals(r_challenges);

        let num_x_out_vars = eq_poly.E_out_current_len().log_2();
        let num_x_in_vars = eq_poly.E_in_current_len().log_2();
        let num_x_prime_vars = num_x_out_vars + num_x_in_vars;
        let num_non_x_out_vars = num_x_in_vars + num_y_svo_vars;

        let num_x_out_points = 1 << num_x_out_vars;
        let num_x_in_points = if num_x_in_vars == 0 {
            1
        } else {
            1 << num_x_in_vars
        };
        let num_non_x_out_points = 1 << num_non_x_out_vars;

        println!("num_x_out_vars {}", num_x_out_vars);
        println!("num_x_in_vars {}", num_x_in_vars);
        println!("num_x_prime_vars {}", num_x_prime_vars);
        println!("num_y_svo_vars {}", num_y_svo_vars);
        println!("num_non_x_out_vars {}", num_non_x_out_vars);
        println!("dense_len {}", self.dense_len);
        println!("unbound_coeffs_len {}", self.ab_unbound_coeffs.len());

        let N_k_dense_len_for_x_prime = 1 << num_x_prime_vars;
        drop(_setup_guard);

        let ranges_calc_span = tracing::span!(
            tracing::Level::DEBUG,
            "streaming_round_ranges_calc",
            num_x_out_points
        );
        let _ranges_calc_guard = ranges_calc_span.enter();
        let mut ranges_for_x_out: Vec<std::ops::Range<usize>> =
            Vec::with_capacity(num_x_out_points);
        let mut output_sizes: Vec<usize> = Vec::with_capacity(num_x_out_points);

        if !self.ab_unbound_coeffs.is_empty() {
            for x_out_task_val in 0..num_x_out_points {
                let min_r1cs_row = (x_out_task_val << (num_x_in_vars + num_y_svo_vars))
                    | (0 << num_y_svo_vars)
                    | 0;
                let max_x_in_val = (1 << num_x_in_vars) - 1;
                let max_y_svo_val = (1 << num_y_svo_vars) - 1;
                let max_r1cs_row = (x_out_task_val << (num_x_in_vars + num_y_svo_vars))
                    | (max_x_in_val << num_y_svo_vars)
                    | max_y_svo_val;
                let start_sparse_idx = min_r1cs_row * 2;
                let end_sparse_idx = max_r1cs_row * 2 + 1;
                let start = self
                    .ab_unbound_coeffs
                    .partition_point(|sc| sc.index < start_sparse_idx);
                let end = self
                    .ab_unbound_coeffs
                    .partition_point(|sc| sc.index <= end_sparse_idx);
                ranges_for_x_out.push(start..end);
                output_sizes.push(end - start);
            }
        }
        drop(_ranges_calc_guard);

        // In order to parallelize, we do a first pass over the coefficients to
        // determine how to divide it into chunks that can be processed independently.
        // In particular, coefficients whose indices are the same modulo (2 ^ (NUM_SVO_ROUNDS + 1))
        // cannot be processed independently.

        // Simplifying assumption: group by `x_out`
        // let block_size = self
        //     .ab_unbound_coeffs
        //     .len()
        //     .div_ceil(rayon::current_num_threads())
        //     .next_multiple_of(2 * (1 << NUM_SVO_ROUNDS));

        // Parallel chunking based on the same highest `x_out` bits
        let chunks: Vec<_> = self
            .ab_unbound_coeffs
            .par_chunk_by(|x, y| x.index / num_non_x_out_points == y.index / num_non_x_out_points)
            .collect();

        // assert_eq!(chunks.len(), num_x_out_points, "Length of chunks should be number of x_out points!");

        // Initialize the bound coeffs

        // TODO: refine (can use binding scratch space)
        let mut bound_coeffs: Vec<SparseCoefficient<F>> = Vec::with_capacity(self.dense_len);

        // Partition `bound_coeffs` into slices for iteration
        let mut output_slices: Vec<&mut [SparseCoefficient<F>]> = vec![];

        // Inside the loop, we will set elements of the output_slices
        // TODO: how to make Rust not complain about parallel threads operating on slices
        // that we have partitioned beforehand
        // The loop returns a bunch of partial quadratic evals which are then summed together
        // to form the final evals (at 0 and infty)

        #[derive(Debug)]
        struct StreamingRoundAccumulator<F: JoltField> {
            total_quadratic_eval_at_0: F,
            total_quadratic_eval_at_infty: F,
        }

        impl<F: JoltField> StreamingRoundAccumulator<F> {
            #[inline]
            fn new() -> Self {
                Self {
                    total_quadratic_eval_at_0: F::zero(),
                    total_quadratic_eval_at_infty: F::zero(),
                }
            }
        }

        let parallel_fold_span =
            tracing::span!(tracing::Level::INFO, "streaming_round_parallel_fold");
        let _parallel_fold_guard = parallel_fold_span.enter();

        // LOOP NEEDS TO BE SIGNIFICANTLY OVERHAULED!!!
        let final_results: StreamingRoundAccumulator<F> = (0..num_x_out_points)
            .into_par_iter()
            .fold(
                || StreamingRoundAccumulator::new(),
                |mut thread_acc, x_out_task_val| {
                    let fold_task_span =
                        tracing::span!(tracing::Level::DEBUG, "fold_task", x_out = x_out_task_val);
                    let _fold_task_guard = fold_task_span.enter();

                    let e_out_val = eq_poly.E_out_current()[x_out_task_val];

                    let relevant_coeffs_slice = if !self.ab_unbound_coeffs.is_empty()
                        && x_out_task_val < ranges_for_x_out.len()
                    {
                        &self.ab_unbound_coeffs[ranges_for_x_out[x_out_task_val].clone()]
                    } else {
                        &[]
                    };

                    let mut current_group_y_val: Option<usize> = None;
                    let mut current_group_x_in_val: Option<usize> = None;

                    let mut loc_raw_az0: i128 = 0;
                    let mut loc_raw_bz0: i128 = 0;
                    let mut loc_raw_az1: i128 = 0;
                    let mut loc_raw_bz1: i128 = 0;

                    let coeff_scan_span = tracing::span!(
                        tracing::Level::DEBUG,
                        "coeff_scan_loop",
                        slice_len = relevant_coeffs_slice.len()
                    );
                    let _coeff_scan_guard = coeff_scan_span.enter();
                    for coeff_sparse in relevant_coeffs_slice.iter() {
                        let r1cs_row_idx = coeff_sparse.index / 2;
                        let type_is_B = (coeff_sparse.index % 2) == 1;

                        let y_val_of_coeff = r1cs_row_idx & ((1 << num_y_svo_vars) - 1);
                        let x_part_of_coeff = r1cs_row_idx >> num_y_svo_vars;
                        let x_in_val_of_coeff = x_part_of_coeff & ((1 << num_x_in_vars) - 1);
                        let x_out_val_of_coeff = x_part_of_coeff >> num_x_in_vars;

                        if x_out_val_of_coeff != x_out_task_val {
                            continue;
                        }

                        let xk_bit_of_coeff = if num_x_out_vars > 0 {
                            (x_out_val_of_coeff >> (num_x_out_vars - 1)) & 1
                        } else if num_x_in_vars > 0 {
                            (x_in_val_of_coeff >> (num_x_in_vars - 1)) & 1
                        } else {
                            0
                        };

                        if current_group_y_val.map_or(true, |prev_y| prev_y != y_val_of_coeff)
                            || current_group_x_in_val
                                .map_or(true, |prev_x_in| prev_x_in != x_in_val_of_coeff)
                        {
                            if let (Some(prev_y), Some(prev_x_in)) =
                                (current_group_y_val, current_group_x_in_val)
                            {
                                if prev_y < eq_r.len() {
                                    let eq_r_val = eq_r[prev_y];
                                    // if loc_raw_az0 != 0 {
                                    //     az0_summed_for_x_out[prev_x_in] += eq_r_val.mul_i128(loc_raw_az0); }
                                    // if loc_raw_bz0 != 0 {
                                    //     bz0_summed_for_x_out[prev_x_in] += eq_r_val.mul_i128(loc_raw_bz0); }
                                    // if loc_raw_az0 != 0 && loc_raw_bz0 != 0 {
                                    //     cz0_summed_for_x_out[prev_x_in] += eq_r_val.mul_i128(loc_raw_az0.checked_mul(loc_raw_bz0).expect("Cz0 product overflow"));
                                    // }
                                    // if loc_raw_az1 != 0 {
                                    //     az1_summed_for_x_out[prev_x_in] += eq_r_val.mul_i128(loc_raw_az1); }
                                    // if loc_raw_bz1 != 0 {
                                    //     bz1_summed_for_x_out[prev_x_in] += eq_r_val.mul_i128(loc_raw_bz1); }
                                    // if loc_raw_az1 != 0 && loc_raw_bz1 != 0 {
                                    //     cz1_summed_for_x_out[prev_x_in] += eq_r_val.mul_i128(loc_raw_az1.checked_mul(loc_raw_bz1).expect("Cz1 product overflow"));
                                    // }
                                }
                            }
                            current_group_y_val = Some(y_val_of_coeff);
                            current_group_x_in_val = Some(x_in_val_of_coeff);
                            loc_raw_az0 = 0;
                            loc_raw_bz0 = 0;
                            loc_raw_az1 = 0;
                            loc_raw_bz1 = 0;
                        }

                        let val_i128 = coeff_sparse.value;
                        if !type_is_B {
                            if xk_bit_of_coeff == 0 {
                                loc_raw_az0 = val_i128;
                            } else {
                                loc_raw_az1 = val_i128;
                            }
                        } else {
                            if xk_bit_of_coeff == 0 {
                                loc_raw_bz0 = val_i128;
                            } else {
                                loc_raw_bz1 = val_i128;
                            }
                        }
                    }
                    drop(_coeff_scan_guard);

                    if let (Some(prev_y), Some(prev_x_in)) =
                        (current_group_y_val, current_group_x_in_val)
                    {
                        if prev_y < eq_r.len() {
                            let eq_r_val = eq_r[prev_y];
                            // if loc_raw_az0 != 0 { az0_summed_for_x_out[prev_x_in] += eq_r_val.mul_i128(loc_raw_az0); }
                            // if loc_raw_bz0 != 0 { bz0_summed_for_x_out[prev_x_in] += eq_r_val.mul_i128(loc_raw_bz0); }
                            // if loc_raw_az0 != 0 && loc_raw_bz0 != 0 {
                            //     cz0_summed_for_x_out[prev_x_in] += eq_r_val.mul_i128(loc_raw_az0.checked_mul(loc_raw_bz0).expect("Cz0 product overflow"));
                            // }
                            // if loc_raw_az1 != 0 { az1_summed_for_x_out[prev_x_in] += eq_r_val.mul_i128(loc_raw_az1); }
                            // if loc_raw_bz1 != 0 { bz1_summed_for_x_out[prev_x_in] += eq_r_val.mul_i128(loc_raw_bz1); }
                            // if loc_raw_az1 != 0 && loc_raw_bz1 != 0 {
                            //     cz1_summed_for_x_out[prev_x_in] += eq_r_val.mul_i128(loc_raw_az1.checked_mul(loc_raw_bz1).expect("Cz1 product overflow"));
                            // }
                        }
                    }

                    let inner_sum_calc_span = tracing::span!(
                        tracing::Level::DEBUG,
                        "inner_sum_calculation",
                        num_x_in_points
                    );
                    let _inner_sum_calc_guard = inner_sum_calc_span.enter();
                    let mut inner_sum_0_for_this_x_out = F::zero();
                    let mut inner_sum_infty_for_this_x_out = F::zero();

                    for x_in_idx_loop in 0..num_x_in_points {
                        // let az0 = az0_summed_for_x_out[x_in_idx_loop];
                        // let bz0 = bz0_summed_for_x_out[x_in_idx_loop];
                        // let cz0 = cz0_summed_for_x_out[x_in_idx_loop];
                        // let az1 = az1_summed_for_x_out[x_in_idx_loop];
                        // let bz1 = bz1_summed_for_x_out[x_in_idx_loop];

                        let az0 = F::zero();
                        let az1 = F::zero();
                        let bz0 = F::zero();
                        let bz1 = F::zero();
                        let cz0 = F::zero();

                        let e_in_val = eq_poly.E_in_current()[x_in_idx_loop];
                        let term_eval_at_0 = az0 * bz0 - cz0;
                        let term_eval_at_infty = (az1 - az0) * (bz1 - bz0);

                        inner_sum_0_for_this_x_out += e_in_val * term_eval_at_0;
                        inner_sum_infty_for_this_x_out += e_in_val * term_eval_at_infty;
                    }
                    drop(_inner_sum_calc_guard);

                    let extend_span = tracing::span!(tracing::Level::DEBUG, "extend_thread_acc");
                    let _extend_guard = extend_span.enter();
                    // thread_acc.az0_evals_all.extend(az0_summed_for_x_out);
                    // thread_acc.bz0_evals_all.extend(bz0_summed_for_x_out);
                    // thread_acc.cz0_evals_all.extend(cz0_summed_for_x_out);
                    // thread_acc.az1_evals_all.extend(az1_summed_for_x_out);
                    // thread_acc.bz1_evals_all.extend(bz1_summed_for_x_out);
                    // thread_acc.cz1_evals_all.extend(cz1_summed_for_x_out);
                    drop(_extend_guard);

                    thread_acc.total_quadratic_eval_at_0 += e_out_val * inner_sum_0_for_this_x_out;
                    thread_acc.total_quadratic_eval_at_infty +=
                        e_out_val * inner_sum_infty_for_this_x_out;

                    drop(_fold_task_guard);
                    thread_acc
                },
            )
            .reduce(
                || StreamingRoundAccumulator::new(),
                |mut acc_a, acc_b| {
                    // let reduce_span = tracing::span!(tracing::Level::DEBUG, "fold_reduce",
                    //     acc_a_az0_len = acc_a.az0_evals_all.len(), acc_b_az0_len = acc_b.az0_evals_all.len());
                    // let _reduce_guard = reduce_span.enter();

                    // let extend_az0_span = tracing::span!(tracing::Level::TRACE, "reduce_extend_az0");
                    // let _extend_az0_guard = extend_az0_span.enter();
                    // acc_a.az0_evals_all.extend(acc_b.az0_evals_all);
                    // drop(_extend_az0_guard);

                    // let extend_az1_span = tracing::span!(tracing::Level::TRACE, "reduce_extend_az1");
                    // let _extend_az1_guard = extend_az1_span.enter();
                    // acc_a.az1_evals_all.extend(acc_b.az1_evals_all);
                    // drop(_extend_az1_guard);

                    // let extend_bz0_span = tracing::span!(tracing::Level::TRACE, "reduce_extend_bz0");
                    // let _extend_bz0_guard = extend_bz0_span.enter();
                    // acc_a.bz0_evals_all.extend(acc_b.bz0_evals_all);
                    // drop(_extend_bz0_guard);

                    // let extend_bz1_span = tracing::span!(tracing::Level::TRACE, "reduce_extend_bz1");
                    // let _extend_bz1_guard = extend_bz1_span.enter();
                    // acc_a.bz1_evals_all.extend(acc_b.bz1_evals_all);
                    // drop(_extend_bz1_guard);

                    // let extend_cz0_span = tracing::span!(tracing::Level::TRACE, "reduce_extend_cz0");
                    // let _extend_cz0_guard = extend_cz0_span.enter();
                    // acc_a.cz0_evals_all.extend(acc_b.cz0_evals_all);
                    // drop(_extend_cz0_guard);

                    // let extend_cz1_span = tracing::span!(tracing::Level::TRACE, "reduce_extend_cz1");
                    // let _extend_cz1_guard = extend_cz1_span.enter();
                    // acc_a.cz1_evals_all.extend(acc_b.cz1_evals_all);
                    // drop(_extend_cz1_guard);

                    let add_totals_span =
                        tracing::span!(tracing::Level::TRACE, "reduce_add_totals");
                    let _add_totals_guard = add_totals_span.enter();
                    acc_a.total_quadratic_eval_at_0 += acc_b.total_quadratic_eval_at_0;
                    acc_a.total_quadratic_eval_at_infty += acc_b.total_quadratic_eval_at_infty;
                    drop(_add_totals_guard);

                    // drop(_reduce_guard);
                    acc_a
                },
            );
        drop(_parallel_fold_guard);

        let process_round_span =
            tracing::span!(tracing::Level::INFO, "streaming_round_process_eq_sumcheck");
        let _process_round_guard = process_round_span.enter();

        // No longer needed
        self.ab_unbound_coeffs = Vec::new();

        // assert_eq!(
        //     final_results.az0_evals_all.len(),
        //     N_k_dense_len_for_x_prime,
        //     "Length mismatch in final Az0 evals"
        // );
        // Similar assertions for bz0, cz0, az1, bz1, cz1 if needed.

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
        drop(_process_round_guard);

        let bind_dense_span = tracing::span!(tracing::Level::INFO, "streaming_round_bind_dense");
        let _bind_dense_guard = bind_dense_span.enter();

        // Bind `bound_coeffs` similar to `subsequent_sumcheck_round`

        self.bound_coeffs = bound_coeffs;
        self.dense_len = N_k_dense_len_for_x_prime;
        drop(_bind_dense_guard);
        drop(_top_level_guard);
    }

    /// This function computes the polynomial for each of the remaining rounds, using the
    /// linear-time algorithm with split-eq optimizations
    ///
    /// At this point, we have computed the `bound_coeffs` for the current round.
    /// We need to compute:
    ///
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// (az_bound[x_out, x_in, 0] * bz_bound[x_out, x_in, 0] - cz_bound[x_out, x_in, 0])`
    ///
    /// and
    ///
    /// `t_i(∞) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// az_bound[x_out, x_in, ∞] * bz_bound[x_out, x_in, ∞]`
    ///
    /// (ordering of indices is MSB to LSB, so x_out is the MSB and x_in is the LSB)
    ///
    /// We then process this to form `s_i(X) = l_i(X) * t_i(X)`, append `s_i.compress()` to the transcript,
    /// derive next challenge `r_i`, then bind both `eq_poly` and `bound_coeffs` with `r_i`.
    ///
    /// NOTE: this is now basically identical to `subsequent_sumcheck_round`, modulo extra Gruen's optimization, and modulo tests (we can add tests back later)
    #[tracing::instrument(
        skip_all,
        name = "NewSpartanInterleavedPolynomial::remaining_sumcheck_round"
    )]
    pub fn remaining_sumcheck_round<ProofTranscript: Transcript>(
        &mut self,
        eq_poly: &mut NewSplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r_challenges: &mut Vec<F>,
        round_polys: &mut Vec<CompressedUniPoly<F>>,
        current_claim: &mut F,
    ) {
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

        // If `E_in` is fully bound, then we simply sum over `E_out`
        let quadratic_evals = if eq_poly.E_in_current_len() == 1 {
            let evals: (F, F) = chunks
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
                            let cz0 = block[2];

                            let az_eval_infty = az.1 - az.0;
                            let bz_eval_infty = bz.1 - bz.0;

                            let eq_evals = eq_poly.E_out_current()[block_index];

                            (
                                eq_evals.mul_0_optimized(az.0.mul_0_optimized(bz.0) - cz0),
                                eq_evals
                                    .mul_0_optimized(az_eval_infty.mul_0_optimized(bz_eval_infty)),
                            )
                        })
                })
                .reduce(
                    || (F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                );
            evals
        } else {
            // If `E_in` is not fully bound, then we have to collect the sum over `E_out` as well
            let num_x1_bits = eq_poly.E_in_current_len().log_2() - 1;
            let x1_bitmask = (1 << num_x1_bits) - 1;

            let evals: (F, F) = chunks
                .par_iter()
                .map(|chunk| {
                    let mut eval_point_0 = F::zero();
                    let mut eval_point_infty = F::zero();

                    let mut inner_sums = (F::zero(), F::zero());
                    let mut prev_x2 = 0;

                    for sparse_block in chunk.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                        let block_index = sparse_block[0].index / 6;
                        let x1 = block_index & x1_bitmask;
                        let E_in_evals = eq_poly.E_in_current()[x1];
                        let x2 = block_index >> num_x1_bits;

                        if x2 != prev_x2 {
                            eval_point_0 += eq_poly.E_out_current()[prev_x2] * inner_sums.0;
                            eval_point_infty += eq_poly.E_out_current()[prev_x2] * inner_sums.1;

                            inner_sums = (F::zero(), F::zero());
                            prev_x2 = x2;
                        }

                        let mut block = [F::zero(); 6];
                        for coeff in sparse_block {
                            block[coeff.index % 6] = coeff.value;
                        }

                        let az = (block[0], block[3]);
                        let bz = (block[1], block[4]);
                        let cz0 = block[2];

                        let az_eval_infty = az.1 - az.0;
                        let bz_eval_infty = bz.1 - bz.0;

                        inner_sums.0 +=
                            E_in_evals.mul_0_optimized(az.0.mul_0_optimized(bz.0) - cz0);
                        inner_sums.1 += E_in_evals
                            .mul_0_optimized(az_eval_infty.mul_0_optimized(bz_eval_infty));
                    }

                    eval_point_0 += eq_poly.E_out_current()[prev_x2] * inner_sums.0;
                    eval_point_infty += eq_poly.E_out_current()[prev_x2] * inner_sums.1;

                    (eval_point_0, eval_point_infty)
                })
                .reduce(
                    || (F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                );
            evals
        };

        // Use the helper function to process the rest of the sumcheck round
        let r_i = process_eq_sumcheck_round(
            quadratic_evals, // (t_i(0), t_i(infty))
            eq_poly,         // Helper will bind this
            round_polys,
            r_challenges,
            current_claim,
            transcript,
        );

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

        // Commented out dense representation for now
        // // num_x_rest_evals is the number of evaluation points for the *remaining* variables (X_rest),
        // // after the current sumcheck variable (X_current_MSB) is fixed to 0 or 1.
        // let num_x_rest_evals = self.az_bound_coeffs.len() / 2;

        // // Get references to the coefficient vectors' current state
        // let az_coeffs = self.az_bound_coeffs.evals_ref();
        // let bz_coeffs = self.bz_bound_coeffs.evals_ref();
        // let cz_coeffs = self.cz_bound_coeffs.evals_ref();

        // // Split each coefficient vector into low (X_MSB = 0) and high (X_MSB = 1) halves
        // // TODO: should be low to high, not high to low, i.e. iterate over az_coeffs by blocks of two at a time
        // let (az_low, az_high) = az_coeffs.split_at(num_x_rest_evals);
        // let (bz_low, bz_high) = bz_coeffs.split_at(num_x_rest_evals);
        // let (cz_low, _cz_high) = cz_coeffs.split_at(num_x_rest_evals); // cz_high unused for t_i(infty)

        // // Compute t_i(0) and t_i(infty)
        // let (quadratic_eval_at_0, quadratic_eval_at_infty) = if eq_poly.E_in_current_len() == 1 {
        //     // E_in part is fully bound. E_out covers all remaining eq_poly variables.
        //     debug_assert_eq!(
        //         eq_poly.E_out_current_len(),
        //         num_x_rest_evals,
        //         "E_out_current_len should match num_x_rest_evals when E_in_current_len is 1"
        //     );

        //     // Combine iterators over the halves and E_out
        //     az_low
        //         .par_iter()
        //         .zip(az_high.par_iter())
        //         .zip(bz_low.par_iter())
        //         .zip(bz_high.par_iter())
        //         .zip(cz_low.par_iter())
        //         .zip(eq_poly.E_out_current().par_iter())
        //         .map(
        //             |(((((az0_ref, az1_ref), bz0_ref), bz1_ref), cz0_ref), e_out_val_ref)| {
        //                 let e_out_val = *e_out_val_ref;

        //                 let az_infty = *az1_ref - *az0_ref; // Coeff of X_current_MSB in Az
        //                 let bz_infty = *bz1_ref - *bz0_ref; // Coeff of X_current_MSB in Bz

        //                 let term_eval_at_0 = *az0_ref * *bz0_ref - *cz0_ref;
        //                 let term_eval_at_infty = az_infty * bz_infty; // X^2 coeff of (Az*Bz)

        //                 (e_out_val * term_eval_at_0, e_out_val * term_eval_at_infty)
        //             },
        //         )
        //         .reduce(
        //             || (F::zero(), F::zero()),
        //             |(acc_0, acc_infty), (val_0, val_infty)| (acc_0 + val_0, acc_infty + val_infty),
        //         )
        // } else {
        //     // Nested sum structure: sum over E_out (outer), then sum over E_in (inner)
        //     let num_e_out_points = eq_poly.E_out_current_len();
        //     let num_e_in_points = eq_poly.E_in_current_len();
        //     debug_assert_eq!(
        //         num_x_rest_evals,
        //         num_e_out_points * num_e_in_points,
        //         "num_x_rest_evals should be product of E_out_current_len and E_in_current_len"
        //     );

        //     eq_poly
        //         .E_out_current()
        //         .par_iter()
        //         .enumerate()
        //         .map(|(e_out_idx, e_out_val)| {
        //             let mut inner_sum_eval_at_0 = F::zero();
        //             let mut inner_sum_eval_at_infty = F::zero();

        //             let start_idx_rest = e_out_idx * num_e_in_points;

        //             // Iterate over the indices corresponding to this E_out chunk
        //             for e_in_idx in 0..num_e_in_points {
        //                 let idx_rest = start_idx_rest + e_in_idx;
        //                 let e_in_val = eq_poly.E_in_current()[e_in_idx];

        //                 // Access halves using the calculated index
        //                 let az0 = az_low[idx_rest];
        //                 let az1 = az_high[idx_rest];
        //                 let az_m = az1 - az0;

        //                 let bz0 = bz_low[idx_rest];
        //                 let bz1 = bz_high[idx_rest];
        //                 let bz_m = bz1 - bz0;

        //                 let cz0 = cz_low[idx_rest];

        //                 let term_eval_at_0 = az0 * bz0 - cz0;
        //                 let term_eval_at_infty = az_m * bz_m;

        //                 // Inner sum part: sum_{x_in} E_in[x_in] * P(x_out, x_in, {0,∞}, r_{high})
        //                 inner_sum_eval_at_0 += e_in_val * term_eval_at_0;
        //                 inner_sum_eval_at_infty += e_in_val * term_eval_at_infty;
        //             }
        //             // Outer sum part: E_out[x_out] * (inner sum)
        //             (
        //                 *e_out_val * inner_sum_eval_at_0,
        //                 *e_out_val * inner_sum_eval_at_infty,
        //             )
        //         })
        //         .reduce(
        //             || (F::zero(), F::zero()),
        //             |(acc_0, acc_infty), (val_0, val_infty)| (acc_0 + val_0, acc_infty + val_infty),
        //         )
        // };

        // // Use the helper function to process the rest of the sumcheck round
        // let r_i = process_eq_sumcheck_round(
        //     (quadratic_eval_at_0, quadratic_eval_at_infty), // (t_i(0), t_i(infty))
        //     eq_poly,                                        // Helper will bind this
        //     round_polys,
        //     r_challenges,
        //     current_claim,
        //     transcript,
        // );

        // // Bind Az, Bz, Cz polynomials for the next round using the challenge r_i
        // // self.bind uses BindingOrder::LowToHigh internally now
        // self.bind(r_i);
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

    // /// Binds the current `{az/bz/cz}_bound_coeffs` with the challenge `r`
    // pub fn bind(&mut self, r: F) {
    //     self.az_bound_coeffs.bind(r, BindingOrder::LowToHigh);
    //     self.bz_bound_coeffs.bind(r, BindingOrder::LowToHigh);
    //     self.cz_bound_coeffs.bind(r, BindingOrder::LowToHigh);
    // }

    // No longer applicable with revert to sparse coefficients
    // pub fn final_sumcheck_evals(&self) -> [F; 3] {
    //     // Simply returns the final evals of Az, Bz, Cz
    //     // At this point `DensePolynomial` has been fully bound, and hence has length 1
    //     let az = self.az_bound_coeffs[0];
    //     let bz = self.bz_bound_coeffs[0];
    //     let cz = self.cz_bound_coeffs[0];
    //     [az, bz, cz]
    // }

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
    #[tracing::instrument(skip_all, name = "SpartanInterleavedPolynomial::first_sumcheck_round")]
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
    #[tracing::instrument(
        skip_all,
        name = "SpartanInterleavedPolynomial::subsequent_sumcheck_round"
    )]
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
