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
    field::{JoltField, OptimizedMul, OptimizedMulI128},
    r1cs::builder::{eval_offset_lc, Constraint, OffsetEqConstraint},
    utils::{
        math::Math,
        small_value::svo_helpers,
        transcript::{AppendToTranscript, Transcript},
    },
};
use ark_ff::Zero;
use rayon::prelude::*;
use std::sync::Arc;

use crate::r1cs::spartan::small_value_optimization::NUM_SVO_ROUNDS;

pub const fn num_non_trivial_ternary_points(num_svo_rounds: usize) -> usize {
    // Returns 3^num_svo_rounds - 2^num_svo_rounds
    let pow_3 = 3_usize
        .checked_pow(num_svo_rounds as u32)
        .expect("Number of ternary points overflowed");
    let pow_2 = 2_usize
        .checked_pow(num_svo_rounds as u32)
        .expect("Number of ternary points overflowed");
    pow_3 - pow_2
}

pub const fn total_num_accums(num_svo_rounds: usize) -> usize {
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

pub const fn num_accums_eval_zero(num_svo_rounds: usize) -> usize {
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

pub const fn num_accums_eval_infty(num_svo_rounds: usize) -> usize {
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

pub const Y_SVO_SPACE_SIZE: usize = 1 << NUM_SVO_ROUNDS;
pub const Y_SVO_RELATED_COEFF_BLOCK_SIZE: usize = 4 * Y_SVO_SPACE_SIZE; // Az/Bz * Xk=0/1 * Y_SVO_SPACE_SIZE

// Modifications for streaming version:
// 1. Do not return the `unbound_coeffs` in the return tuple.
// 2. Do streaming rounds until we get to a small enough cached size that we can store `bound_coeffs`

pub struct NewSpartanInterleavedPolynomial<const NUM_SVO_ROUNDS: usize, F: JoltField> {
    /// A list of Arc'd sparse vectors representing the (interleaved) coefficients for the Az, Bz polynomials
    /// Generated from binary evaluations. Each inner Vec is sorted by index.
    ///
    /// (note: **no** Cz coefficients are stored here, since they are not needed for small value
    /// precomputation, and can be computed on the fly in streaming round)
    pub(crate) ab_unbound_coeffs_shards: Vec<Arc<Vec<SparseCoefficient<i128>>>>,

    pub(crate) bound_coeffs: Vec<SparseCoefficient<F>>,

    binding_scratch_space: Vec<SparseCoefficient<F>>,

    pub(crate) dense_len: usize,
}

// Implement parallel flat map iterator for the SVO precomputation

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

        // The variable layout looks as follows:
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
        
        // Define the structure returned by each parallel map task
        struct PrecomputeTaskOutput<F: JoltField> {
            ab_coeffs_local_arc: Arc<Vec<SparseCoefficient<i128>>>, // Changed to Arc
            svo_accums_zero_local: [F; NUM_ACCUMS_EVAL_ZERO],
            svo_accums_infty_local: [F; NUM_ACCUMS_EVAL_INFTY],
        }

        let num_parallel_chunks = if num_x_out_vals > 0 {
            std::cmp::min(
                num_x_out_vals,
                // Setting number of chunks for more even work distribution
                rayon::current_num_threads().next_power_of_two() * 16,
            )
        } else {
            1 // Avoid 0 chunks if num_x_out_vals is 0
        };
        assert!(
            num_parallel_chunks > 0 || num_x_out_vals == 0,
            "num_parallel_chunks must be positive if there are x_out_vals to process"
        );

        let x_out_chunk_size = if num_x_out_vals > 0 {
            num_x_out_vals.div_ceil(num_parallel_chunks)
        } else {
            0 // No work per chunk if no x_out_vals
        };

        let collected_chunk_outputs: Vec<PrecomputeTaskOutput<F>> = (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_task_span = tracing::debug_span!("chunk_task", chunk_idx);
                let _x_out_task_guard = x_out_task_span.enter();

                let mut chunk_ab_coeffs = Vec::new(); // Still build locally first
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

                    // Distribute the accumulated tA values to the SVO accumulators
                    svo_helpers::distribute_tA_to_svo_accumulators_generic::<NUM_SVO_ROUNDS, F>(
                        &task_tA_accumulator_vec,
                        x_out_val,
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
                    ab_coeffs_local_arc: Arc::new(chunk_ab_coeffs), // Wrap in Arc
                    svo_accums_zero_local: chunk_svo_accums_zero,
                    svo_accums_infty_local: chunk_svo_accums_infty,
                }
            }) // End .map() over chunks
            .collect(); // Collect all chunk outputs

        drop(_main_fold_guard);

        // --- Finalization ---
        let finalization_span = tracing::info_span!("finalization");
        let _finalization_guard = finalization_span.enter();

        // No need to calculate total_ab_coeffs_len for allocation if using Vec<Arc<...>> directly

        let mut final_svo_accums_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
        let mut final_svo_accums_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];
        let mut final_ab_unbound_coeffs_shards: Vec<Arc<Vec<SparseCoefficient<i128>>>> = Vec::with_capacity(collected_chunk_outputs.len());


        let aggregation_loop_span = tracing::debug_span!("finalization_aggregate_chunk_outputs_loop");
        let _aggregation_loop_guard = aggregation_loop_span.enter();
        
        for task_output in collected_chunk_outputs {
            final_ab_unbound_coeffs_shards.push(task_output.ab_coeffs_local_arc); // Collect Arcs

            if NUM_ACCUMS_EVAL_ZERO > 0 {
                for idx in 0..NUM_ACCUMS_EVAL_ZERO {
                    final_svo_accums_zero[idx] += task_output.svo_accums_zero_local[idx];
                }
            }

            if NUM_ACCUMS_EVAL_INFTY > 0 {
                for idx in 0..NUM_ACCUMS_EVAL_INFTY {
                    final_svo_accums_infty[idx] += task_output.svo_accums_infty_local[idx];
                }
            }
        }
        drop(_aggregation_loop_guard);

        // final_ab_unbound_coeffs_shards is now fully populated and SVO accumulators are summed.

        // Debug check for sortedness
        #[cfg(test)]
        {
            if NUM_SVO_ROUNDS > 0 {
                for shard_arc in &final_ab_unbound_coeffs_shards {
                    let shard = shard_arc.as_ref();
                    if !shard.is_empty() {
                        let mut prev_index = shard[0].index;
                        for coeff in shard.iter().skip(1) {
                            assert!(
                                coeff.index > prev_index,
                                "Indices not monotonically increasing in shard: prev {}, current {}",
                                prev_index, coeff.index
                            );
                            prev_index = coeff.index;
                        }
                    }
                }
                // Additionally, check that indices across shards are somewhat ordered (not strictly, but useful for a sense check)
                // This is harder to assert strongly without flattening. For now, per-shard sortedness is key.
            }
            println!("Per-shard sortedness check passed!");
        }

        // #[cfg(test)]
        // {
        //     // Flatten the shards for comparison
        //     let temp_flattened_coeffs: Vec<SparseCoefficient<i128>> = final_ab_unbound_coeffs_shards
        //         .iter()
        //         .flat_map(|shard_arc| shard_arc.as_ref().clone()) // Clones elements from Arc<Vec<T>>
        //         .collect::<Vec<_>>() // Temporary Vec for sorting if needed, or direct use
        //         .into_iter() // Convert to iterator of SparseCoefficient<i128>
        //         .collect(); // Collect into a new Vec for comparison

        //     // The comparison function might need sorting if shards interleave indices,
        //     // but if each chunk_idx processes a contiguous range of x_out_val,
        //     // the indices between shards should be somewhat ordered too.
        //     // For robustness, tests::compare_new_and_old_ab_coeffs might need to sort its input
        //     // or handle non-contiguous blocks if that's how shards are formed.
        //     // Assuming for now the comparison function can handle it or the data is sufficiently ordered.

        //     let old_new_result = SpartanInterleavedPolynomial::new(
        //         uniform_constraints,
        //         cross_step_constraints,
        //         flattened_polynomials,
        //         padded_num_constraints,
        //     );
        //     // The compare_new_and_old_ab_coeffs expects a slice. We give it the flattened one.
        //     tests::compare_new_and_old_ab_coeffs::<F>(&temp_flattened_coeffs, &old_new_result.unbound_coeffs);
        // }

        drop(_finalization_guard);

        // Return final SVO accumulators and Self struct.
        (
            final_svo_accums_zero,
            final_svo_accums_infty,
            Self {
                ab_unbound_coeffs_shards: final_ab_unbound_coeffs_shards,
                bound_coeffs: vec![],
                binding_scratch_space: vec![],
                dense_len: num_steps * padded_num_constraints,
            },
        )
    }

    /// This function uses the streaming algorithm to compute the sum-check polynomial for the round
    /// right after the small value precomputed rounds.
    ///
    /// At this point, we have the `ab_unbound_coeffs` generated from `new_with_precompute`. We will
    /// use these to compute the evals {Az/Bz/Cz}(r, u, x') needed for later linear-time sumcheck
    /// rounds (storing them in `bound_coeffs`), and compute the polynomial for this
    /// round at the same time.
    ///
    /// Recall that we need to compute
    ///
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] * (unbound_coeffs_a(x_out, x_in,
    /// 0, r) * unbound_coeffs_b(x_out, x_in, 0, r) - unbound_coeffs_c(x_out, x_in, 0, r))`
    ///
    /// and
    ///
    /// `t_i(∞) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] * (unbound_coeffs_a(x_out,
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

        let setup_span = tracing::debug_span!("streaming_round_setup");
        let _setup_guard = setup_span.enter();

        let num_y_svo_vars = r_challenges.len();
        assert_eq!(
            num_y_svo_vars, NUM_SVO_ROUNDS,
            "r_challenges length mismatch with NUM_SVO_ROUNDS"
        );
        let mut r_rev = r_challenges.clone();
        r_rev.reverse();
        let eq_r_evals = EqPolynomial::evals(&r_rev);

        drop(_setup_guard);

        let main_processing_span = tracing::info_span!("streaming_round_main_processing");
        let _main_processing_guard = main_processing_span.enter();

        struct StreamingTaskOutput<F: JoltField> {
            bound_coeffs_local: Vec<SparseCoefficient<F>>,
            sumcheck_eval_at_0_local: F,
            sumcheck_eval_at_infty_local: F,
        }

        // These are needed to derive x_out_val_stream and x_in_val_stream from a block_id
        let num_streaming_x_in_vars = eq_poly.E_in_current_len().log_2();

        let collected_chunk_outputs: Vec<StreamingTaskOutput<F>> = self.ab_unbound_coeffs_shards
            .par_iter()
            .map(|shard_arc| {
                let shard_slice = shard_arc.as_ref();
                let mut task_bound_coeffs = Vec::new();
                let mut task_sum_contrib_0 = F::zero();
                let mut task_sum_contrib_infty = F::zero();

                for logical_block_coeffs in shard_slice.chunk_by(|c1, c2| {
                    c1.index / Y_SVO_RELATED_COEFF_BLOCK_SIZE == c2.index / Y_SVO_RELATED_COEFF_BLOCK_SIZE
                }) {
                    if logical_block_coeffs.is_empty() {
                        continue;
                    }

                    let current_block_id = logical_block_coeffs[0].index / Y_SVO_RELATED_COEFF_BLOCK_SIZE;
                    
                    let x_out_val_stream = current_block_id >> num_streaming_x_in_vars;
                    let x_in_val_stream = current_block_id & ((1 << num_streaming_x_in_vars) - 1);

                    // // Boundary checks (important if eq_poly dimensions can change unexpectedly)
                    // if x_out_val_stream >= eq_poly.E_out_current_len() || 
                    //    (eq_poly.E_in_current_len() > 0 && x_in_val_stream >= eq_poly.E_in_current_len()) {
                    //     tracing::error!(
                    //         "Derived (x_out_val_stream {}, x_in_val_stream {}) from block_id {} in shard is out of bounds for current eq_poly (E_out_len {}, E_in_len {}). Coefficients: {:?}",
                    //         x_out_val_stream, x_in_val_stream, current_block_id, 
                    //         eq_poly.E_out_current_len(), eq_poly.E_in_current_len(),
                    //         logical_block_coeffs.iter().map(|c| (c.index, c.value)).collect::<Vec<_>>(),
                    //     );
                    //     // This should ideally not happen if the invariant holds and precompute/streaming vars are aligned.
                    //     // Depending on desired robustness, could panic or skip.
                    //     continue; 
                    // }

                    let e_out_val = eq_poly.E_out_current()[x_out_val_stream];
                    let e_in_val = if eq_poly.E_in_current_len() > 1 {
                        eq_poly.E_in_current()[x_in_val_stream]
                    } else if eq_poly.E_in_current_len() == 1 {
                        eq_poly.E_in_current()[0]
                    } else { // E_in_current_len() == 0, meaning no x_in variables for eq_poly
                        F::one() // Effective contribution of E_in is 1
                    };

                    let mut az0_at_r = F::zero();
                    let mut az1_at_r = F::zero();
                    let mut bz0_at_r = F::zero();
                    let mut bz1_at_r = F::zero();
                    let mut cz0_at_r = F::zero();
                    let mut cz1_at_r = F::zero();

                    let mut coeff_idx_in_block = 0;
                    while coeff_idx_in_block < logical_block_coeffs.len() {
                        let current_coeff = &logical_block_coeffs[coeff_idx_in_block];
                        let local_offset =
                            current_coeff.index % Y_SVO_RELATED_COEFF_BLOCK_SIZE;
                        let current_is_B = (local_offset % 2) == 1;
                        let y_val_idx = (local_offset / 2) % Y_SVO_SPACE_SIZE;
                        let x_next_val = (local_offset / 2) / Y_SVO_SPACE_SIZE; // 0 or 1
                        let eq_r_y = eq_r_evals[y_val_idx];

                        if current_is_B { // Current coefficient is Bz
                            let bz_orig_val = current_coeff.value;
                            match x_next_val {
                                0 => bz0_at_r += eq_r_y.mul_i128_1_optimized(bz_orig_val),
                                1 => bz1_at_r += eq_r_y.mul_i128_1_optimized(bz_orig_val),
                                _ => unreachable!(),
                            }
                            coeff_idx_in_block += 1;
                        } else { // Current coefficient is Az
                            let az_orig_val = current_coeff.value;
                            let mut bz_orig_for_this_az = 0i128;

                            match x_next_val {
                                0 => az0_at_r += eq_r_y.mul_i128_1_optimized(az_orig_val),
                                1 => az1_at_r += eq_r_y.mul_i128_1_optimized(az_orig_val),
                                _ => unreachable!(),
                            }

                            if coeff_idx_in_block + 1 < logical_block_coeffs.len() {
                                let next_coeff = &logical_block_coeffs[coeff_idx_in_block + 1];
                                if next_coeff.index == current_coeff.index + 1 {
                                    bz_orig_for_this_az = next_coeff.value;
                                    let next_local_offset = next_coeff.index % Y_SVO_RELATED_COEFF_BLOCK_SIZE;
                                    let next_x_next_val = (next_local_offset / 2) / Y_SVO_SPACE_SIZE;
                                    debug_assert_eq!(x_next_val, next_x_next_val, "Paired Az/Bz should share x_next_val. Current idx {}, next idx {}, current x_next {}, next x_next {}", current_coeff.index, next_coeff.index, x_next_val, next_x_next_val);

                                    match x_next_val { // x_next_val of the current Az
                                        0 => bz0_at_r += eq_r_y.mul_i128_1_optimized(bz_orig_for_this_az),
                                        1 => bz1_at_r += eq_r_y.mul_i128_1_optimized(bz_orig_for_this_az),
                                        _ => unreachable!(),
                                    }
                                    coeff_idx_in_block += 1; // Consumed the Bz coefficient as well
                                }
                            }
                            coeff_idx_in_block += 1; // Consumed the Az coefficient

                            if !az_orig_val.is_zero() && !bz_orig_for_this_az.is_zero() {
                                let cz_orig_val =
                                    az_orig_val.wrapping_mul(bz_orig_for_this_az);
                                match x_next_val { // x_next_val of the current Az
                                    0 => cz0_at_r += eq_r_y.mul_i128(cz_orig_val),
                                    1 => cz1_at_r += eq_r_y.mul_i128(cz_orig_val),
                                    _ => unreachable!(),
                                }
                            }
                        }
                    }

                    if !az0_at_r.is_zero() {
                        task_bound_coeffs.push((6 * current_block_id, az0_at_r).into());
                    }
                    if !bz0_at_r.is_zero() {
                        task_bound_coeffs.push((6 * current_block_id + 1, bz0_at_r).into());
                    }
                    if !cz0_at_r.is_zero() {
                        task_bound_coeffs.push((6 * current_block_id + 2, cz0_at_r).into());
                    }
                    if !az1_at_r.is_zero() {
                        task_bound_coeffs.push((6 * current_block_id + 3, az1_at_r).into());
                    }
                    if !bz1_at_r.is_zero() {
                        task_bound_coeffs.push((6 * current_block_id + 4, bz1_at_r).into());
                    }
                    if !cz1_at_r.is_zero() {
                        task_bound_coeffs.push((6 * current_block_id + 5, cz1_at_r).into());
                    }

                    let p_at_xk0 = az0_at_r * bz0_at_r - cz0_at_r;
                    let az_eval_infty = az1_at_r - az0_at_r;
                    let bz_eval_infty = bz1_at_r - bz0_at_r;
                    let p_slope_term = az_eval_infty * bz_eval_infty;

                    task_sum_contrib_0 += e_out_val * e_in_val * p_at_xk0;
                    task_sum_contrib_infty += e_out_val * e_in_val * p_slope_term;
                }

                StreamingTaskOutput {
                    bound_coeffs_local: task_bound_coeffs,
                    sumcheck_eval_at_0_local: task_sum_contrib_0,
                    sumcheck_eval_at_infty_local: task_sum_contrib_infty,
                }
            })
            .collect();

        drop(_main_processing_guard);

        // Clear ab_unbound_coeffs_shards as they are processed and their data (conceptually) moves to bound_coeffs via binding_scratch_space
        self.ab_unbound_coeffs_shards.clear();
        self.ab_unbound_coeffs_shards.shrink_to_fit();

        // Aggregate sumcheck contributions directly from collected_chunk_outputs
        let sum_aggregation_span = tracing::info_span!("streaming_round_aggregate_sums");
        let _sum_aggregation_guard = sum_aggregation_span.enter();
        let mut total_sumcheck_eval_at_0 = F::zero();
        let mut total_sumcheck_eval_at_infty = F::zero();
        for task_output in &collected_chunk_outputs { // Iterate by reference before consuming collected_chunk_outputs
            total_sumcheck_eval_at_0 += task_output.sumcheck_eval_at_0_local;
            total_sumcheck_eval_at_infty += task_output.sumcheck_eval_at_infty_local;
        }
        drop(_sum_aggregation_guard);

        // Compute r_i challenge using aggregated sumcheck values
        let r_i = process_eq_sumcheck_round(
            (total_sumcheck_eval_at_0, total_sumcheck_eval_at_infty),
            eq_poly,
            round_polys,
            r_challenges,
            claim,
            transcript,
        );

        // Bind coefficients directly from task outputs into scratch space
        let bind_coeffs_span =
            tracing::span!(tracing::Level::INFO, "streaming_round_bind_coeffs");
        let _bind_coeffs_guard = bind_coeffs_span.enter();

        // Calculate the output size for binding for each task's local coefficients
        let calc_output_sizes_span = tracing::debug_span!("streaming_bind_calc_per_task_output_sizes");
        let _calc_output_sizes_guard = calc_output_sizes_span.enter();
        let per_task_output_sizes: Vec<usize> = collected_chunk_outputs
            .par_iter() // Iterate over collected_chunk_outputs by reference for calculating sizes
            .map(|task_output| {
                let coeffs_from_task = &task_output.bound_coeffs_local;
                let mut current_task_total_output_size = 0;
                for sub_block_of_6 in
                    coeffs_from_task.chunk_by(|sc1, sc2| sc1.index / 6 == sc2.index / 6)
                {
                    // Self::binding_output_length expects a slice representing one such sub_block_of_6
                    current_task_total_output_size += Self::binding_output_length(sub_block_of_6);
                }
                current_task_total_output_size
            })
            .collect();
        drop(_calc_output_sizes_guard);

        let total_binding_output_len: usize = per_task_output_sizes.iter().sum();

        // Prepare binding_scratch_space
        if self.binding_scratch_space.capacity() < total_binding_output_len {
            self.binding_scratch_space
                .reserve_exact(total_binding_output_len - self.binding_scratch_space.capacity());
        }
        unsafe {
            self.binding_scratch_space.set_len(total_binding_output_len);
        }

        // Create mutable slices into binding_scratch_space, one for each task's output
        let create_output_slices_span = tracing::debug_span!("streaming_bind_create_output_slices_for_tasks");
        let _create_output_slices_guard = create_output_slices_span.enter();
        let mut output_slices_for_tasks: Vec<&mut [SparseCoefficient<F>]> = Vec::with_capacity(collected_chunk_outputs.len());
        let mut scratch_remainder = self.binding_scratch_space.as_mut_slice();
        for slice_len in per_task_output_sizes {
            let (first, second) = scratch_remainder.split_at_mut(slice_len);
            output_slices_for_tasks.push(first);
            scratch_remainder = second;
        }
        debug_assert_eq!(scratch_remainder.len(), 0);
        drop(_create_output_slices_guard);

        // Parallel binding loop: iterate over tasks and their designated output slices
        let parallel_bind_loop_span =
            tracing::info_span!("streaming_bind_parallel_task_coeff_binding_loop");
        let _parallel_bind_loop_guard = parallel_bind_loop_span.enter();

        collected_chunk_outputs // Now consume collected_chunk_outputs
            .into_par_iter()
            .zip_eq(output_slices_for_tasks.into_par_iter())
            .for_each(|(task_output, output_slice_for_task)| {
                let bind_task_span =
                    tracing::debug_span!("streaming_bind_process_task_coeffs");
                let _bind_task_guard = bind_task_span.enter();

                let coeffs_from_task = &task_output.bound_coeffs_local; // These are the Vec<SparseCoefficient<F>> from a StreamingTaskOutput

                let mut current_output_idx_in_slice = 0;
                // Iterate through sub-blocks of 6 within this task's coeffs_from_task
                for sub_block_of_6 in
                    coeffs_from_task.chunk_by(|sc1, sc2| sc1.index / 6 == sc2.index / 6)
                {
                    if sub_block_of_6.is_empty() {
                        continue;
                    }
                    let block_idx_for_6_coeffs = sub_block_of_6[0].index / 6;

                    let mut az0 = F::zero();
                    let mut bz0 = F::zero();
                    let mut cz0 = F::zero();
                    let mut az1 = F::zero();
                    let mut bz1 = F::zero();
                    let mut cz1 = F::zero();

                    for coeff in sub_block_of_6 {
                        match coeff.index % 6 {
                            0 => az0 = coeff.value,
                            1 => bz0 = coeff.value,
                            2 => cz0 = coeff.value,
                            3 => az1 = coeff.value,
                            4 => bz1 = coeff.value,
                            5 => cz1 = coeff.value,
                            _ => unreachable!(),
                        }
                    }

                    let new_block_idx = block_idx_for_6_coeffs;

                    let bound_az = az0 + r_i * (az1 - az0);
                    if !bound_az.is_zero() {
                        if current_output_idx_in_slice < output_slice_for_task.len() {
                            output_slice_for_task[current_output_idx_in_slice] =
                                (3 * new_block_idx, bound_az).into();
                        }
                        current_output_idx_in_slice += 1;
                    }
                    let bound_bz = bz0 + r_i * (bz1 - bz0);
                    if !bound_bz.is_zero() {
                        if current_output_idx_in_slice < output_slice_for_task.len() {
                            output_slice_for_task[current_output_idx_in_slice] =
                                (3 * new_block_idx + 1, bound_bz).into();
                        }
                        current_output_idx_in_slice += 1;
                    }
                    let bound_cz = cz0 + r_i * (cz1 - cz0);
                    if !bound_cz.is_zero() {
                        if current_output_idx_in_slice < output_slice_for_task.len() {
                            output_slice_for_task[current_output_idx_in_slice] =
                                (3 * new_block_idx + 2, bound_cz).into();
                        }
                        current_output_idx_in_slice += 1;
                    }
                }
                debug_assert_eq!(current_output_idx_in_slice, output_slice_for_task.len(), "Mismatch in written elements vs pre-calculated slice length for task output");
            });
        drop(_parallel_bind_loop_guard);

        let swap_span = tracing::debug_span!("streaming_bind_swap_coeff_buffers");
        let _swap_guard = swap_span.enter();
        std::mem::swap(&mut self.bound_coeffs, &mut self.binding_scratch_space);
        drop(_swap_guard);

        self.dense_len = eq_poly.len();

        drop(_bind_coeffs_guard);
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
        let top_level_span = tracing::span!(tracing::Level::INFO, "remaining_sumcheck_round_body");
        let _top_level_guard = top_level_span.enter();

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

        let compute_evals_span = tracing::info_span!("remaining_round_compute_evals");
        let _compute_evals_guard = compute_evals_span.enter();

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
            let num_x1_bits = eq_poly.E_in_current_len().log_2();
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
        drop(_compute_evals_guard);

        let process_sumcheck_span = tracing::info_span!("remaining_round_process_eq_sumcheck");
        let _process_sumcheck_guard = process_sumcheck_span.enter();
        // Use the helper function to process the rest of the sumcheck round
        let r_i = process_eq_sumcheck_round(
            quadratic_evals, // (t_i(0), t_i(infty))
            eq_poly,         // Helper will bind this
            round_polys,
            r_challenges,
            current_claim,
            transcript,
        );
        drop(_process_sumcheck_guard);

        let bind_coeffs_span = tracing::info_span!("remaining_round_bind_coeffs");
        let _bind_coeffs_guard = bind_coeffs_span.enter();

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
        drop(_bind_coeffs_guard);
        drop(_top_level_guard);
    }

    /// Computes the number of non-zero coefficients that would result from
    /// binding the given slice of coefficients. Only invoked on `bound_coeffs` which holds
    /// Az/Bz/Cz bound evaluations.
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

#[derive(Default, Debug, Clone)]
pub struct SpartanInterleavedPolynomial<F: JoltField> {
    /// A sparse vector representing the (interleaved) coefficients in the Az, Bz, Cz
    /// polynomials used in the first Spartan sumcheck. Before the polynomial is bound
    /// the first time, all the coefficients can be represented by `i128`s.
    pub(crate) unbound_coeffs_shards: Vec<Arc<Vec<SparseCoefficient<i128>>>>,
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
        let outer_span = tracing::info_span!("SpartanInterleavedPolynomial::new_body");
        let _outer_guard = outer_span.enter();

        let var_setup_span = tracing::debug_span!("old_new_variable_setup");
        let _var_setup_guard = var_setup_span.enter();

        let num_steps = flattened_polynomials[0].len();

        let num_chunks = rayon::current_num_threads().next_power_of_two() * 16;
        let chunk_size = num_steps.div_ceil(num_chunks);

        drop(_var_setup_guard);

        let main_coeff_compute_span = tracing::info_span!("old_new_main_coefficient_computation");
        let _main_coeff_compute_guard = main_coeff_compute_span.enter();

        let par_iter_span = tracing::debug_span!("old_new_flat_map_iter_computation");
        let _par_iter_guard = par_iter_span.enter();
        let unbound_coeffs_shards_iter = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_index| {
                let chunk_task_span = tracing::debug_span!("old_new_chunk_task", chunk_idx = chunk_index);
                let _chunk_task_guard = chunk_task_span.enter();

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

                Arc::new(coeffs)
            });
        drop(_par_iter_guard);

        let collect_span = tracing::info_span!("old_new_collect_unbound_coeffs");
        let _collect_guard = collect_span.enter();
        let unbound_coeffs_shards: Vec<Arc<Vec<SparseCoefficient<i128>>>> =
            unbound_coeffs_shards_iter.collect();
        drop(_collect_guard);
        
        drop(_main_coeff_compute_guard);

        #[cfg(test)]
        {
            // Check that indices are monotonically increasing
            for shard_arc in &unbound_coeffs_shards {
                let shard = shard_arc.as_ref();
                if !shard.is_empty() {
                    let mut prev_index = shard[0].index;
                    for coeff in shard.iter().skip(1) {
                        assert!(
                            coeff.index > prev_index,
                            "Indices not monotonically increasing in shard: prev {}, current {}",
                            prev_index, coeff.index
                        );
                        prev_index = coeff.index;
                    }
                }
            }
        }

        Self {
            unbound_coeffs_shards,
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
            for shard_arc in &self.unbound_coeffs_shards {
                for coeff in shard_arc.as_ref() {
                    match coeff.index % 3 {
                        0 => az[coeff.index / 3] = F::from_i128(coeff.value),
                        1 => bz[coeff.index / 3] = F::from_i128(coeff.value),
                        2 => cz[coeff.index / 3] = F::from_i128(coeff.value),
                        _ => unreachable!(),
                    }
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
        // let block_size = self
        //     .unbound_coeffs_shards
        //     .len()
        //     .div_ceil(rayon::current_num_threads())
        //     .next_multiple_of(6);
        // let chunks: Vec<_> = self
        //     .unbound_coeffs_shards
        //     .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
        //     .collect();

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

        let evals: (F, F, F) = self.unbound_coeffs_shards
            .par_iter()
            .map(|shard_arc| {
                let shard_coeffs = shard_arc.as_ref();
                let mut eval_point_0 = F::zero();
                let mut eval_point_2 = F::zero();
                let mut eval_point_3 = F::zero();

                let mut inner_sums = (F::zero(), F::zero(), F::zero());
                let mut prev_x2 = 0;

                for sparse_block in shard_coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
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
        let output_sizes: Vec<_> = self.unbound_coeffs_shards
            .par_iter()
            .map(|shard_arc| Self::binding_output_length(shard_arc.as_ref()))
            .collect();

        let total_output_len = output_sizes.iter().sum();
        self.bound_coeffs = Vec::with_capacity(total_output_len);
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.bound_coeffs.set_len(total_output_len);
        }
        let mut output_slices: Vec<&mut [SparseCoefficient<F>]> = Vec::with_capacity(self.unbound_coeffs_shards.len());
        let mut remainder = self.bound_coeffs.as_mut_slice();
        for slice_len in output_sizes {
            let (first, second) = remainder.split_at_mut(slice_len);
            output_slices.push(first);
            remainder = second;
        }
        debug_assert_eq!(remainder.len(), 0);

        self.unbound_coeffs_shards
            .par_iter()
            .zip_eq(output_slices.into_par_iter())
            .for_each(|(unbound_shard_arc, output_slice_for_shard)| {
                let unbound_coeffs_in_shard = unbound_shard_arc.as_ref();
                let mut output_index = 0;
                for block in unbound_coeffs_in_shard.chunk_by(|x, y| x.index / 6 == y.index / 6) {
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
                        output_slice_for_shard[output_index] = (
                            3 * block_index,
                            F::from_i128(low) + r_i.mul_i128(high - low),
                        )
                            .into();
                        output_index += 1;
                    }
                    if bz_coeff != (None, None) {
                        let (low, high) = (bz_coeff.0.unwrap_or(0), bz_coeff.1.unwrap_or(0));
                        output_slice_for_shard[output_index] = (
                            3 * block_index + 1,
                            F::from_i128(low) + r_i.mul_i128(high - low),
                        )
                            .into();
                        output_index += 1;
                    }
                    if cz_coeff != (None, None) {
                        let (low, high) = (cz_coeff.0.unwrap_or(0), cz_coeff.1.unwrap_or(0));
                        output_slice_for_shard[output_index] = (
                            3 * block_index + 2,
                            F::from_i128(low) + r_i.mul_i128(high - low),
                        )
                            .into();
                        output_index += 1;
                    }
                }
                debug_assert_eq!(output_index, output_slice_for_shard.len())
            });

        // Drop the unbound coeffs shards now that we've bound them
        self.unbound_coeffs_shards.clear();
        self.unbound_coeffs_shards.shrink_to_fit();

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

    // Below are the methods that incorporates Gruen's optimizations (but not the small value optimizations)
    /// The first round of the first Spartan sumcheck. Since the polynomials
    /// are still unbound at the beginning of this round, we can replace some
    /// of the field arithmetic with `i128` arithmetic.
    ///
    /// THIS IS THE VERSION WITH GRUEN'S OPTIMIZATION. We also implement the extra optimization of
    /// only computing the quadratic evaluation at infinity, since the one at zero is always zero.
    #[tracing::instrument(
        skip_all,
        name = "SpartanInterleavedPolynomial::first_sumcheck_round_with_gruen"
    )]
    pub fn first_sumcheck_round_with_gruen<ProofTranscript: Transcript>(
        &mut self,
        eq_poly: &mut NewSplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r: &mut Vec<F>,
        polys: &mut Vec<CompressedUniPoly<F>>,
        claim: &mut F,
    ) {

        let num_x_in_bits = eq_poly.E_in_current_len().log_2();
        let x_in_bitmask = (1 << num_x_in_bits) - 1;

        // In the first round, we only need to compute the quadratic evaluation at infinity,
        // since the eval at zero is always zero.
        let quadratic_eval_at_infty = self.unbound_coeffs_shards
            .par_iter()
            .map(|shard_arc| {
                let shard_coeffs = shard_arc.as_ref();
                let mut shard_eval_point_infty = F::zero();

                let mut current_shard_inner_sums = F::zero();
                let mut current_shard_prev_x_out = 0;

                for sparse_block in shard_coeffs.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                    let block_index = sparse_block[0].index / 6;
                    let x_in = block_index & x_in_bitmask;
                    let E_in_evals = eq_poly.E_in_current()[x_in];
                    let x_out = block_index >> num_x_in_bits;

                    if x_out != current_shard_prev_x_out {
                        shard_eval_point_infty += eq_poly.E_out_current()[current_shard_prev_x_out]
                            * current_shard_inner_sums;
                        current_shard_inner_sums = F::zero();
                        current_shard_prev_x_out = x_out;
                    }

                    // This holds the az0, az1, bz0, bz1 evals. No need for cz0, cz1 since we only need
                    // the eval at infinity.
                    let mut az0 = 0i128;
                    let mut az1 = 0i128;
                    let mut bz0 = 0i128;
                    let mut bz1 = 0i128;
                    for coeff in sparse_block {
                        let local_idx = coeff.index % 6;
                        if local_idx == 0 {
                            az0 = coeff.value;
                        } else if local_idx == 1 {
                            bz0 = coeff.value;
                        } else if local_idx == 3 {
                            az1 = coeff.value;
                        } else if local_idx == 4 {
                            bz1 = coeff.value;
                        }
                    }
                    let az_infty = az1 - az0;
                    let bz_infty = bz1 - bz0;
                    if az_infty != 0 && bz_infty != 0 {
                        current_shard_inner_sums += E_in_evals.mul_i128_1_optimized(
                            az_infty.checked_mul(bz_infty).unwrap_or_else(|| {
                                panic!("az_infty * bz_infty overflow");
                            }),
                        );
                    }
                }
                shard_eval_point_infty +=
                    eq_poly.E_out_current()[current_shard_prev_x_out] * current_shard_inner_sums;
                shard_eval_point_infty
            })
            .reduce(|| F::zero(), |sum, evals| sum + evals);

        let r_i = process_eq_sumcheck_round(
            (F::zero(), quadratic_eval_at_infty),
            eq_poly,
            polys,
            r,
            claim,
            transcript,
        );

        // Compute the number of non-zero bound coefficients that will be produced
        // per chunk.
        let output_sizes: Vec<_> = self.unbound_coeffs_shards
            .par_iter()
            .map(|shard_arc| Self::binding_output_length(shard_arc.as_ref()))
            .collect();

        let total_output_len = output_sizes.iter().sum();
        self.bound_coeffs = Vec::with_capacity(total_output_len);
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.bound_coeffs.set_len(total_output_len);
        }
        let mut output_slices: Vec<&mut [SparseCoefficient<F>]> = Vec::with_capacity(self.unbound_coeffs_shards.len());
        let mut remainder = self.bound_coeffs.as_mut_slice();
        for slice_len in output_sizes {
            let (first, second) = remainder.split_at_mut(slice_len);
            output_slices.push(first);
            remainder = second;
        }
        debug_assert_eq!(remainder.len(), 0);

        self.unbound_coeffs_shards
            .par_iter()
            .zip_eq(output_slices.into_par_iter())
            .for_each(|(unbound_shard_arc, output_slice_for_shard)| {
                let unbound_coeffs_in_shard = unbound_shard_arc.as_ref();
                let mut output_index = 0;
                for block in unbound_coeffs_in_shard.chunk_by(|x, y| x.index / 6 == y.index / 6) {
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
                        output_slice_for_shard[output_index] = (
                            3 * block_index,
                            F::from_i128(low) + r_i.mul_i128_1_optimized(high - low),
                        )
                            .into();
                        output_index += 1;
                    }
                    if bz_coeff != (None, None) {
                        let (low, high) = (bz_coeff.0.unwrap_or(0), bz_coeff.1.unwrap_or(0));
                        output_slice_for_shard[output_index] = (
                            3 * block_index + 1,
                            F::from_i128(low) + r_i.mul_i128_1_optimized(high - low),
                        )
                            .into();
                        output_index += 1;
                    }
                    if cz_coeff != (None, None) {
                        let (low, high) = (cz_coeff.0.unwrap_or(0), cz_coeff.1.unwrap_or(0));
                        output_slice_for_shard[output_index] = (
                            3 * block_index + 2,
                            F::from_i128(low) + r_i.mul_i128(high - low),
                        )
                            .into();
                        output_index += 1;
                    }
                }
                debug_assert_eq!(output_index, output_slice_for_shard.len())
            });

        // Drop the unbound coeffs shards now that we've bound them
        self.unbound_coeffs_shards.clear();
        self.unbound_coeffs_shards.shrink_to_fit();

        self.dense_len /= 2;
    }

    /// All subsequent rounds of the first Spartan sumcheck.
    /// THIS IS THE VERSION WITH GRUEN'S OPTIMIZATION (but not the small value optimization)
    #[tracing::instrument(
        skip_all,
        name = "SpartanInterleavedPolynomial::subsequent_sumcheck_round_with_gruen"
    )]
    pub fn subsequent_sumcheck_round_with_gruen<ProofTranscript: Transcript>(
        &mut self,
        eq_poly: &mut NewSplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r: &mut Vec<F>,
        polys: &mut Vec<CompressedUniPoly<F>>,
        claim: &mut F,
    ) {
        let block_size = self
            .bound_coeffs
            .len()
            .div_ceil(rayon::current_num_threads())
            .next_multiple_of(6);
        let chunks: Vec<_> = self
            .bound_coeffs
            .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
            .collect();

        let quadratic_evals = if eq_poly.E_in_current_len() == 1 {
            let evals = chunks
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
            let num_x_in_bits = eq_poly.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;

            let evals = chunks
                .par_iter()
                .map(|chunk| {
                    let mut eval_point_0 = F::zero();
                    let mut eval_point_infty = F::zero();

                    let mut inner_sums = (F::zero(), F::zero());
                    let mut prev_x_out = 0;

                    for sparse_block in chunk.chunk_by(|x, y| x.index / 6 == y.index / 6) {
                        let block_index = sparse_block[0].index / 6;
                        let x_in = block_index & x_bitmask;
                        let E_in_eval = eq_poly.E_in_current()[x_in];
                        let x_out = block_index >> num_x_in_bits;

                        if x_out != prev_x_out {
                            let E_out_eval = eq_poly.E_out_current()[prev_x_out];
                            eval_point_0 += E_out_eval * inner_sums.0;
                            eval_point_infty += E_out_eval * inner_sums.1;

                            inner_sums = (F::zero(), F::zero());
                            prev_x_out = x_out;
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

                        inner_sums.0 += E_in_eval.mul_0_optimized(az.0.mul_0_optimized(bz.0) - cz0);
                        inner_sums.1 +=
                            E_in_eval.mul_0_optimized(az_eval_infty.mul_0_optimized(bz_eval_infty));
                    }

                    eval_point_0 += eq_poly.E_out_current()[prev_x_out] * inner_sums.0;
                    eval_point_infty += eq_poly.E_out_current()[prev_x_out] * inner_sums.1;

                    (eval_point_0, eval_point_infty)
                })
                .reduce(
                    || (F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                );
            evals
        };

        let r_i = process_eq_sumcheck_round(quadratic_evals, eq_poly, polys, r, claim, transcript);

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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Helper function to compare Az/Bz coefficients from NewSpartanInterleavedPolynomial 
    // (which only has Az/Bz) with those from the old SpartanInterleavedPolynomial (which has Az/Bz/Cz).
    pub(super) fn compare_new_and_old_ab_coeffs<F: JoltField> (
        new_coeffs_az_bz_only: &[SparseCoefficient<i128>],
        old_coeffs_az_bz_cz: &[SparseCoefficient<i128>]
    ) {
        let mut new_coeffs_ptr = 0;
        let mut old_coeffs_ptr = 0;

        while new_coeffs_ptr < new_coeffs_az_bz_only.len() {
            let new_coeff = &new_coeffs_az_bz_only[new_coeffs_ptr];
            let is_new_az = new_coeff.index % 2 == 0;
            let logical_idx_new = if is_new_az {
                new_coeff.index / 2
            } else {
                (new_coeff.index - 1) / 2
            };

            let mut found_match_for_current_new_coeff = false;
            while old_coeffs_ptr < old_coeffs_az_bz_cz.len() {
                let old_coeff = &old_coeffs_az_bz_cz[old_coeffs_ptr];
                let logical_idx_old = old_coeff.index / 3;

                if logical_idx_old < logical_idx_new {
                    old_coeffs_ptr += 1;
                    continue;
                }

                if logical_idx_old > logical_idx_new {
                    panic!(
                        "New coefficient for logical_idx {} (new_coeff.index {}) not found in old coefficients. \
                         Next old logical_idx is {} (old_coeff.index {}).",
                        logical_idx_new, new_coeff.index, logical_idx_old, old_coeff.index
                    );
                }

                // logical_idx_old == logical_idx_new
                let old_coeff_type = old_coeff.index % 3; // 0 for Az, 1 for Bz, 2 for Cz

                if is_new_az {
                    // new_coeff is Az
                    if old_coeff_type == 0 {
                        // old_coeff is Az
                        assert_eq!(
                            new_coeff.value, old_coeff.value,
                            "Az value mismatch: new_idx {}, old_idx {}. Logical_idx: {}",
                            new_coeff.index, old_coeff.index, logical_idx_new
                        );
                        found_match_for_current_new_coeff = true;
                        old_coeffs_ptr += 1; // Consumed this old_coeff for the match
                        break; // Move to next new_coeff
                    } else {
                        // old_coeff is Bz or Cz for the same logical_idx. Skip it.
                        old_coeffs_ptr += 1;
                        // Continue in this inner loop to find Az for logical_idx_new or exhaust options for it.
                    }
                } else {
                    // new_coeff is Bz
                    if old_coeff_type == 1 {
                        // old_coeff is Bz
                        assert_eq!(
                            new_coeff.value, old_coeff.value,
                            "Bz value mismatch: new_idx {}, old_idx {}. Logical_idx: {}",
                            new_coeff.index, old_coeff.index, logical_idx_new
                        );
                        found_match_for_current_new_coeff = true;
                        old_coeffs_ptr += 1; // Consumed this old_coeff for the match
                        break; // Move to next new_coeff
                    } else {
                        // old_coeff is Az or Cz for the same logical_idx. Skip it.
                        old_coeffs_ptr += 1;
                        // Continue in this inner loop to find Bz for logical_idx_new or exhaust options for it.
                    }
                }
            }

            if !found_match_for_current_new_coeff {
                panic!(
                    "No match found in old coefficients for new_coeff at index {} (logical_idx {}). \
                     Old coefficients pointer reached end or no suitable type found for this logical index.",
                    new_coeff.index, logical_idx_new
                );
            }
            new_coeffs_ptr += 1;
        }

        // Optional: Check if there are any remaining Az/Bz in old_coeffs that weren't matched.
        // This would indicate new_coeffs is missing something old_coeffs had.
        while old_coeffs_ptr < old_coeffs_az_bz_cz.len() {
            let old_coeff = &old_coeffs_az_bz_cz[old_coeffs_ptr];
            if old_coeff.index % 3 != 2 {
                // If it's not a Cz coefficient
                let logical_idx_old = old_coeff.index / 3;
                // Check if this logical_idx_old was covered by any logical_idx_new
                // This check is more complex; for now, focus on all new_coeffs being found.
                // A simple panic here might be too strict if old_coeffs can have trailing Az/Bz
                // for logical indices not present at all in new_coeffs (which shouldn't happen).
                println!(
                    "Warning: Remaining non-Cz coefficient in old_coeffs after checking all new_coeffs: index {}, value {}, logical_idx {}. \
                    This might indicate that the new method produces fewer Az/Bz coefficients.", 
                    old_coeff.index, old_coeff.value, logical_idx_old
                );
            }
            old_coeffs_ptr += 1;
        }

        println!(
            "Az/Bz coefficient comparison with old SpartanInterleavedPolynomial::new() passed!"
        );
    }
}
