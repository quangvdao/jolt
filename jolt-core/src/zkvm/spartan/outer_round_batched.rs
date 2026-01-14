#![allow(clippy::too_many_arguments)]
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::{
    dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, multilinear_polynomial::BindingOrder,
    split_eq_poly::GruenSplitEqPolynomial, unipoly::UniPoly,
};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::zkvm::r1cs::evaluation::R1CSEval;
use crate::zkvm::r1cs::inputs::ALL_R1CS_INPUTS;
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::spartan::outer_baseline::SparseCoefficient;
use crate::zkvm::witness::VirtualPolynomial;
use crate::{
    field::JoltField,
    transcripts::Transcript,
    utils::small_value::accum::{SignedUnreducedAccum, UnreducedProduct},
    utils::{math::Math, small_value::svo_helpers},
    zkvm::bytecode::BytecodePreprocessing,
    zkvm::r1cs::{
        constraints::R1CS_CONSTRAINTS, evaluation::eval_az_bz_batch_from_row,
        inputs::R1CSCycleInputs,
    },
};
use allocative::Allocative;
use ark_ff::biginteger::{S160, S96 as I8OrI96};
use num_traits::Zero;
use rayon::prelude::*;
use std::sync::Arc;
use tracer::instruction::Cycle;

/// Number of rounds to use for small value optimization.
/// Testing & estimation shows that 3 rounds is the best tradeoff
pub const NUM_SVO_ROUNDS: usize = 3;

pub const NUM_NONTRIVIAL_TERNARY_POINTS: usize =
    svo_helpers::num_non_trivial_ternary_points(NUM_SVO_ROUNDS);
pub const NUM_ACCUMS_EVAL_ZERO: usize = svo_helpers::num_accums_eval_zero(NUM_SVO_ROUNDS);
pub const NUM_ACCUMS_EVAL_INFTY: usize = svo_helpers::num_accums_eval_infty(NUM_SVO_ROUNDS);

/// Number of Y-assignments per SVO block. Equal to 2^NUM_SVO_ROUNDS.
/// This is the size of the subspace over the prefix Y used in small-value optimization.
pub const Y_SVO_SPACE_SIZE: usize = 1 << NUM_SVO_ROUNDS;

/// Number of interleaved coefficients per logical block for a fixed (x_out, x_in):
///  - 2 polynomials (Az, Bz)
///  - 2 evaluations at x_next ∈ {0, 1}
///  - Y_SVO_SPACE_SIZE assignments of Y
///    So total = 4 * Y_SVO_SPACE_SIZE.
pub const Y_SVO_RELATED_COEFF_BLOCK_SIZE: usize = 4 * Y_SVO_SPACE_SIZE;

/// Bit-width of a logical block. Computed as log2(4) + NUM_SVO_ROUNDS = 2 + NUM_SVO_ROUNDS.
/// Use this for fast block id calculation via shifts
pub const Y_SVO_RELATED_COEFF_BLOCK_SIZE_SHIFT: usize = 2 + NUM_SVO_ROUNDS;

/// Bitmask for the local offset within a block. Equals (1 << SHIFT) - 1.
/// Use this to extract local offsets via bit-and, instead of modulo.
pub const Y_SVO_RELATED_COEFF_BLOCK_SIZE_MASK: usize =
    (1usize << Y_SVO_RELATED_COEFF_BLOCK_SIZE_SHIFT) - 1;

#[derive(Clone, Debug, Allocative)]
pub struct RoundBatchedSpartanInterleavedPolynomial<const NUM_SVO_ROUNDS: usize, F: JoltField> {
    /// The bound coefficients for the Az and Bz polynomials.
    /// Will be populated in the streaming round (after SVO rounds)
    pub(crate) bound_coeffs: Vec<SparseCoefficient<F>>,

    binding_scratch_space: Vec<SparseCoefficient<F>>,

    padded_num_constraints: usize,
}

impl<const NUM_SVO_ROUNDS: usize, F: JoltField>
    RoundBatchedSpartanInterleavedPolynomial<NUM_SVO_ROUNDS, F>
{
    /// Compute the unbound coefficients for the Az and Bz polynomials (no Cz coefficients are
    /// needed), along with the accumulators for the small value optimization (SVO) rounds.
    ///
    /// Recall that the accumulators are of the form: accum_i[v_0, ..., v_{i-1}, u] = \sum_{y_rest}
    /// \sum_{x_out} E_out(x_out || y_rest) * \sum_{x_in} E_in(x_in) * P(x_out, x_in, y_rest, u,
    /// v_0, ..., v_{i-1}),
    ///
    /// for all i < NUM_SVO_ROUNDS, v_0,..., v_{i-1} \in {0,1,∞}, u \in {0,∞}, and P(X) = Az(X) *
    /// Bz(X).
    ///
    /// Note that we have reverse the order of variables from the paper, since in this codebase the
    /// indexing is MSB to LSB (as we go from 0 to N-1, i.e. left to right).
    ///
    /// Note that only the accumulators with at least one infinity among v_j and u are non-zero, so
    /// the fully binary ones do not need to be computed. Plus, the ones with at least one infinity
    /// has no Cz contribution in this setting (Cz is always zero).
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
        name = "RoundBatchedSpartanInterleavedPolynomial::svo_sumcheck_round"
    )]
    pub fn svo_sumcheck_round(
        preprocess: &BytecodePreprocessing,
        trace: &[Cycle],
        tau: &[F::Challenge],
    ) -> ([F; NUM_ACCUMS_EVAL_ZERO], [F; NUM_ACCUMS_EVAL_INFTY], Self) {
        // Variable layout and binding order (MSB -> LSB):
        // 0 ... (N/2 - l) ... (n_s) ... (N - l) ... (N - i - 1) ... (N - 1)
        // where:
        //   n_s = number of step vars (log2(steps))
        //   N   = total num vars (step + constraint)
        //   l   = NUM_SVO_ROUNDS (number of Y-prefix variables used by SVO)
        // Partition:
        //   - 0 .. (N/2 - l)              => x_out (MSB block shared with y_rest)
        //   - (N/2 - l) .. (n_s)          => x_in_step
        //   - (n_s) .. (N - l)            => x_in_constraint (non-SVO constraint vars)
        //   - (N - l) .. (N - i - 1)      => y_rest (suffix of Y for current round)
        //   - (N - i - 1) .. (N - 1)      => (u || v_0..v_{i-1}) SVO variables
        // We use MSB->LSB indexing throughout the codebase.

        let padded_num_constraints = R1CS_CONSTRAINTS.len().next_power_of_two();
        let num_steps = trace.len();
        let num_step_vars = if num_steps > 0 { num_steps.log_2() } else { 0 };
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
            NUM_SVO_ROUNDS <= num_constraint_vars,
            "NUM_SVO_ROUNDS ({NUM_SVO_ROUNDS}) cannot exceed total constraint variables ({num_constraint_vars})"
        );

        let num_non_svo_constraint_vars = num_constraint_vars.saturating_sub(NUM_SVO_ROUNDS);
        let num_non_svo_z_vars = num_step_vars + num_non_svo_constraint_vars;
        assert_eq!(
            num_non_svo_z_vars,
            total_num_vars - NUM_SVO_ROUNDS,
            "num_non_svo_z_vars ({num_non_svo_z_vars}) + NUM_SVO_ROUNDS ({NUM_SVO_ROUNDS}) must be == total_num_vars ({total_num_vars})"
        );

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

        let num_uniform_r1cs_constraints = R1CS_CONSTRAINTS.len();
        let rem_num_uniform_r1cs_constraints = num_uniform_r1cs_constraints % Y_SVO_SPACE_SIZE;

        // Build split-eq helper and precompute E_in (over x_in) and E_out (over x_out)
        let eq_poly = GruenSplitEqPolynomial::new_for_small_value(
            tau,
            iter_num_x_out_vars,
            iter_num_x_in_vars,
            NUM_SVO_ROUNDS,
            // Scale E_in by R^2 so typed accumulators reduce to field semantics
            Some(F::MONTGOMERY_R_SQUARE),
        );
        let E_in_evals = eq_poly.E_in_current();
        let E_out_vec = &eq_poly.E_out_vec;

        assert_eq!(E_out_vec.len(), NUM_SVO_ROUNDS);

        let num_x_out_vals = 1usize << iter_num_x_out_vars;
        let num_x_in_step_vals = 1usize << iter_num_x_in_step_vars;

        struct PrecomputeTaskOutput<F: JoltField> {
            svo_accums_zero_local: [F; NUM_ACCUMS_EVAL_ZERO],
            svo_accums_infty_local: [F; NUM_ACCUMS_EVAL_INFTY],
        }

        let num_parallel_chunks = if num_x_out_vals > 0 {
            std::cmp::min(
                num_x_out_vals,
                rayon::current_num_threads().next_power_of_two() * 8,
            )
        } else {
            1
        };
        assert!(
            num_parallel_chunks > 0 || num_x_out_vals == 0,
            "num_parallel_chunks must be positive if there are x_out_vals to process"
        );

        let x_out_chunk_size = if num_x_out_vals > 0 {
            std::cmp::max(1, num_x_out_vals.div_ceil(num_parallel_chunks))
        } else {
            0
        };

        // Parallel over chunks of x_out values. For each (x_out, x_in_step):
        //   - Evaluate each constraint-row's A and B LC at step index to obtain Az/Bz blocks
        //   - Fold Az/Bz with E_in(x_in) into tA contributions
        //   - Distribute tA to SVO accumulators via E_out(x_out)
        // We also collect sparse AB coefficients interleaved by (x_next ∈ {0,1}).
        let collected_chunk_outputs: Vec<PrecomputeTaskOutput<F>> = (0..num_parallel_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let x_out_start = chunk_idx * x_out_chunk_size;
                let x_out_end = std::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);

                let mut chunk_svo_accums_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
                let mut chunk_svo_accums_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];

                for x_out_val in x_out_start..x_out_end {
                    let mut tA_pos_acc_for_current_x_out =
                        [UnreducedProduct::<F>::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];
                    let mut tA_neg_acc_for_current_x_out =
                        [UnreducedProduct::<F>::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];
                    let mut current_x_out_svo_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
                    let mut current_x_out_svo_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];

                    for x_in_step_val in 0..num_x_in_step_vals {
                        let current_step_idx =
                            (x_out_val << iter_num_x_in_step_vars) | x_in_step_val;
                        let mut current_x_in_constraint_val = 0;
                        // Materialize row once for this step; reused for all chunks
                        let row_inputs =
                            R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);

                        let mut binary_az_block = [I8OrI96::zero(); Y_SVO_SPACE_SIZE];
                        let mut binary_bz_block = [S160::zero(); Y_SVO_SPACE_SIZE];

                        // Iterate constraints in Y_SVO_SPACE_SIZE blocks so we can call the
                        // small-value kernels on full Az/Bz blocks when available.
                        for uniform_svo_chunk in R1CS_CONSTRAINTS.chunks(Y_SVO_SPACE_SIZE) {
                            let chunk_size = uniform_svo_chunk.len();
                            // Fill Az/Bz values for this chunk using materialized row
                            eval_az_bz_batch_from_row::<F>(
                                uniform_svo_chunk,
                                &row_inputs,
                                &mut binary_az_block[..chunk_size],
                                &mut binary_bz_block[..chunk_size],
                            );

                            // If this is a full block, compute and update tA, then reset Az, Bz blocks
                            // (the last block may not be full, in which case we need to delay
                            // computation of tA until after processing all constraints in the block)
                            if uniform_svo_chunk.len() == Y_SVO_SPACE_SIZE {
                                let x_in_val = (x_in_step_val << iter_num_x_in_constraint_vars)
                                    | current_x_in_constraint_val;
                                let E_in_val = &E_in_evals[x_in_val];

                                // New typed path
                                svo_helpers::compute_and_update_tA_inplace::<NUM_SVO_ROUNDS, F>(
                                    &binary_az_block,
                                    &binary_bz_block,
                                    E_in_val,
                                    &mut tA_pos_acc_for_current_x_out,
                                    &mut tA_neg_acc_for_current_x_out,
                                );

                                current_x_in_constraint_val += 1;
                                // Reset local blocks for the next iteration
                                binary_az_block = [I8OrI96::zero(); Y_SVO_SPACE_SIZE];
                                binary_bz_block = [S160::zero(); Y_SVO_SPACE_SIZE];
                            }
                        }

                        // Process final partial block, if any
                        if rem_num_uniform_r1cs_constraints > 0 {
                            let x_in_val_last = (x_in_step_val << iter_num_x_in_constraint_vars)
                                | current_x_in_constraint_val;
                            let E_in_val_last = &E_in_evals[x_in_val_last];

                            // New typed path on partial block currently in binary_* blocks
                            svo_helpers::compute_and_update_tA_inplace::<NUM_SVO_ROUNDS, F>(
                                &binary_az_block,
                                &binary_bz_block,
                                E_in_val_last,
                                &mut tA_pos_acc_for_current_x_out,
                                &mut tA_neg_acc_for_current_x_out,
                            );
                        }
                    }

                    // finalize: reduce unreduced accumulators and combine pos/neg into field
                    let mut tA_sum_for_current_x_out = [F::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];
                    for i in 0..NUM_NONTRIVIAL_TERNARY_POINTS {
                        let pos_f = F::from_montgomery_reduce::<8>(tA_pos_acc_for_current_x_out[i]);
                        let neg_f = F::from_montgomery_reduce::<8>(tA_neg_acc_for_current_x_out[i]);
                        // E_in was pre-scaled by R^2, so reduction already matches field semantics
                        tA_sum_for_current_x_out[i] = pos_f - neg_f;
                    }

                    // Distribute accumulated tA for this x_out into the SVO accumulators
                    // (both zero and infty evaluations) using precomputed E_out tables.
                    svo_helpers::distribute_tA_to_svo_accumulators_generic::<NUM_SVO_ROUNDS, F>(
                        &tA_sum_for_current_x_out,
                        x_out_val,
                        E_out_vec,
                        &mut current_x_out_svo_zero,
                        &mut current_x_out_svo_infty,
                    );

                    for i in 0..NUM_ACCUMS_EVAL_ZERO {
                        chunk_svo_accums_zero[i] += current_x_out_svo_zero[i];
                    }
                    for i in 0..NUM_ACCUMS_EVAL_INFTY {
                        chunk_svo_accums_infty[i] += current_x_out_svo_infty[i];
                    }
                }

                PrecomputeTaskOutput {
                    svo_accums_zero_local: chunk_svo_accums_zero,
                    svo_accums_infty_local: chunk_svo_accums_infty,
                }
            })
            .collect();

        let mut final_svo_accums_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
        let mut final_svo_accums_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];

        for task_output in collected_chunk_outputs {
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

        (
            final_svo_accums_zero,  // Use new baseline results
            final_svo_accums_infty, // Use new baseline results
            Self {
                bound_coeffs: vec![],
                binding_scratch_space: vec![],
                padded_num_constraints,
            },
        )
    }

    /// Computes the number of non-zero coefficients that would result from
    /// binding the given slice of coefficients. Only invoked on `bound_coeffs` which holds
    /// Az/Bz bound evaluations.
    fn binding_output_length<T>(coeffs: &[SparseCoefficient<T>]) -> usize {
        let mut output_size = 0;
        for block in coeffs.chunk_by(|x, y| x.index / 4 == y.index / 4) {
            let mut Az_coeff_found = false;
            let mut Bz_coeff_found = false;
            for coeff in block {
                match coeff.index % 2 {
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
                    _ => unreachable!(),
                }
            }
        }
        output_size
    }

    pub fn final_sumcheck_evals(&self) -> [F; 2] {
        let mut final_az_eval = F::zero();
        let mut final_bz_eval = F::zero();
        for coeff in &self.bound_coeffs {
            match coeff.index {
                0 => final_az_eval = coeff.value,
                1 => final_bz_eval = coeff.value,
                _ => {}
            }
        }
        [final_az_eval, final_bz_eval]
    }
}

/// Compute-only helper for the streaming round (right after the small value precomputed rounds).
///
/// Recall we need to compute:
///   t_i(0)  = Σ_{x_out} E_out[x_out] Σ_{x_in} E_in[x_in] · (Az(x_out,x_in,0,r) · Bz(x_out,x_in,0,r))
///   t_i(∞)  = Σ_{x_out} E_out[x_out] Σ_{x_in} E_in[x_in] · (Az(x_out,x_in,∞,r) · Bz(x_out,x_in,∞,r))
/// where the eval at r over y is recomputed via eq(r, y). We also collect sparse
/// [az0,bz0,az1,bz1] per logical block (x_next ∈ {0,1}) at y=r for subsequent binding.
///
/// Returns ((t(0), t(∞)), block4_at_r).
#[inline]
pub fn compute_streaming_endpoints_and_block4<F: JoltField>(
    preprocess: &BytecodePreprocessing,
    trace: &[Cycle],
    eq_poly: &GruenSplitEqPolynomial<F>,
    r_challenges: &[F::Challenge],
) -> ((F, F), Vec<SparseCoefficient<F>>) {
    let mut r_rev = r_challenges.to_vec();
    r_rev.reverse();
    let eq_r_evals = EqPolynomial::<F>::evals_with_scaling(&r_rev, Some(F::MONTGOMERY_R_SQUARE));

    let num_x_out_vals = eq_poly.E_out_current_len();
    let iter_num_x_out_vars = if num_x_out_vals > 0 {
        num_x_out_vals.log_2()
    } else {
        0
    };
    let num_steps = trace.len();
    let num_step_vars = if num_steps > 0 { num_steps.log_2() } else { 0 };
    debug_assert!(iter_num_x_out_vars <= num_step_vars);
    let iter_num_x_in_step_vars = num_step_vars - iter_num_x_out_vars;
    let num_x_in_step_vals = if iter_num_x_in_step_vars > 0 {
        1usize << iter_num_x_in_step_vars
    } else {
        1
    };

    let num_uniform_r1cs_constraints = R1CS_CONSTRAINTS.len();
    let y_blocks_in_constraints = if num_uniform_r1cs_constraints > 0 {
        num_uniform_r1cs_constraints.div_ceil(Y_SVO_SPACE_SIZE)
    } else {
        0
    };
    let num_block_pairs_per_step =
        (R1CS_CONSTRAINTS.len().next_power_of_two()) >> (NUM_SVO_ROUNDS + 1);

    let mut sum0 = F::zero();
    let mut sumInf = F::zero();
    let mut out: Vec<SparseCoefficient<F>> = Vec::new();

    for x_out_val in 0..num_x_out_vals {
        let mut inner_sum0 = F::Unreduced::<9>::zero();
        let mut inner_sumInf = F::Unreduced::<9>::zero();
        for x_in_step_val in 0..num_x_in_step_vals {
            let current_step_idx = (x_out_val << iter_num_x_in_step_vars) | x_in_step_val;
            let row_inputs = R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);

            for block_pair_idx in 0..num_block_pairs_per_step {
                let mut az_acc = [
                    SignedUnreducedAccum::<F>::new(),
                    SignedUnreducedAccum::<F>::new(),
                ];
                let mut bz_acc = [
                    SignedUnreducedAccum::<F>::new(),
                    SignedUnreducedAccum::<F>::new(),
                ];

                for k in 0..2 {
                    let chunk_index = (block_pair_idx << 1) | k;
                    if chunk_index >= y_blocks_in_constraints {
                        continue;
                    }
                    let start = chunk_index * Y_SVO_SPACE_SIZE;
                    let end =
                        core::cmp::min(start + Y_SVO_SPACE_SIZE, num_uniform_r1cs_constraints);
                    let uniform_svo_chunk = &R1CS_CONSTRAINTS[start..end];
                    let chunk_size = uniform_svo_chunk.len();

                    let mut binary_az_block = [I8OrI96::zero(); Y_SVO_SPACE_SIZE];
                    let mut binary_bz_block = [S160::zero(); Y_SVO_SPACE_SIZE];
                    eval_az_bz_batch_from_row::<F>(
                        uniform_svo_chunk,
                        &row_inputs,
                        &mut binary_az_block[..chunk_size],
                        &mut binary_bz_block[..chunk_size],
                    );

                    let x_next_val = k;
                    for idx_in_svo_block in 0..chunk_size {
                        let eq = eq_r_evals[idx_in_svo_block];
                        let az = binary_az_block[idx_in_svo_block];
                        let bz = binary_bz_block[idx_in_svo_block];
                        az_acc[x_next_val].fmadd_az(&eq, az);
                        bz_acc[x_next_val].fmadd_bz(&eq, bz);
                    }
                }

                let az0 = az_acc[0].reduce_to_field();
                let bz0 = bz_acc[0].reduce_to_field();
                let az1 = az_acc[1].reduce_to_field();
                let bz1 = bz_acc[1].reduce_to_field();

                let p0 = az0 * bz0;
                let slope = (az1 - az0) * (bz1 - bz0);

                let current_block_id = current_step_idx * num_block_pairs_per_step + block_pair_idx;
                let num_streaming_x_in_vars = eq_poly.E_in_current_len().log_2();
                let x_out_idx = current_block_id >> num_streaming_x_in_vars;
                let x_in_idx = current_block_id & ((1 << num_streaming_x_in_vars) - 1);

                let e_out = if x_out_idx < eq_poly.E_out_current_len() {
                    eq_poly.E_out_current()[x_out_idx]
                } else {
                    F::zero()
                };
                let e_in = if eq_poly.E_in_current_len() == 0 {
                    F::one()
                } else if eq_poly.E_in_current_len() == 1 {
                    eq_poly.E_in_current()[0]
                } else if x_in_idx < eq_poly.E_in_current_len() {
                    eq_poly.E_in_current()[x_in_idx]
                } else {
                    F::zero()
                };

                inner_sum0 += e_in.mul_unreduced::<9>(p0);
                inner_sumInf += e_in.mul_unreduced::<9>(slope);

                let block_id = current_block_id;
                if !az0.is_zero() {
                    out.push((4 * block_id, az0).into());
                }
                if !bz0.is_zero() {
                    out.push((4 * block_id + 1, bz0).into());
                }
                if !az1.is_zero() {
                    out.push((4 * block_id + 2, az1).into());
                }
                if !bz1.is_zero() {
                    out.push((4 * block_id + 3, bz1).into());
                }

                let red0 = F::from_montgomery_reduce::<9>(inner_sum0);
                let redi = F::from_montgomery_reduce::<9>(inner_sumInf);
                sum0 += e_out * red0;
                sumInf += e_out * redi;
                inner_sum0 = F::Unreduced::zero();
                inner_sumInf = F::Unreduced::zero();
            }
        }
    }

    ((sum0, sumInf), out)
}

/// Bind [az0,bz0,az1,bz1] blocks at r to [az(r), bz(r)] pairs
#[inline]
pub fn bind_block4_at_r<F: JoltField>(
    block4_at_r: &[SparseCoefficient<F>],
    r_i: F::Challenge,
) -> Vec<SparseCoefficient<F>> {
    let mut out: Vec<SparseCoefficient<F>> = Vec::with_capacity(block4_at_r.len() / 2);
    let mut sorted = block4_at_r.to_vec();
    sorted.sort_by_key(|c| c.index / 4);
    let mut i = 0usize;
    while i < sorted.len() {
        let blk = sorted[i].index / 4;
        let mut az0 = F::zero();
        let mut bz0 = F::zero();
        let mut az1 = F::zero();
        let mut bz1 = F::zero();
        while i < sorted.len() && sorted[i].index / 4 == blk {
            match sorted[i].index % 4 {
                0 => az0 = sorted[i].value,
                1 => bz0 = sorted[i].value,
                2 => az1 = sorted[i].value,
                3 => bz1 = sorted[i].value,
                _ => {}
            }
            i += 1;
        }
        let azb = az0 + r_i * (az1 - az0);
        if !azb.is_zero() {
            out.push((2 * blk, azb).into());
        }
        let bzb = bz0 + r_i * (bz1 - bz0);
        if !bzb.is_zero() {
            out.push((2 * blk + 1, bzb).into());
        }
    }
    out
}

/// Compute-only helper for remaining rounds endpoints from bound sparse coefficients.
///
/// Given current `eq_poly` state and the interleaved bound coefficients [az_lo, bz_lo, az_hi, bz_hi]
/// per block, compute the quadratic endpoints used to build the cubic in each remaining round:
///   t_i(0)  = Σ_{x_out} E_out[x_out] Σ_{x_in} E_in[x_in] · (Az_lo · Bz_lo)
///   t_i(∞)  = Σ_{x_out} E_out[x_out] Σ_{x_in} E_in[x_in] · ((Az_hi−Az_lo) · (Bz_hi−Bz_lo))
/// The ordering of indices is MSB→LSB (x_out is MSB group, x_in is LSB group).
#[inline]
pub fn compute_remaining_endpoints_from_bound_coeffs<F: JoltField>(
    eq_poly: &GruenSplitEqPolynomial<F>,
    bound_coeffs: &[SparseCoefficient<F>],
) -> (F, F) {
    if eq_poly.E_in_current_len() == 1 {
        // groups are pairs (0,1) per block
        let groups = bound_coeffs.len().div_ceil(4);
        let (t0_unr, tinf_unr) = (0..groups)
            .into_par_iter()
            .map(|g| {
                let base = 4 * g;
                let mut az0 = F::zero();
                let mut az1 = F::zero();
                let mut bz0 = F::zero();
                let mut bz1 = F::zero();
                for j in 0..4 {
                    if let Some(c) = bound_coeffs.get(base + j) {
                        match c.index % 2 {
                            0 => {
                                if c.index % 4 == 0 {
                                    az0 = c.value
                                } else {
                                    az1 = c.value
                                }
                            }
                            1 => {
                                if c.index % 4 == 1 {
                                    bz0 = c.value
                                } else {
                                    bz1 = c.value
                                }
                            }
                            _ => {}
                        }
                    }
                }
                let eq = eq_poly.E_out_current()[g];
                let p0 = az0 * bz0;
                let slope = (az1 - az0) * (bz1 - bz0);
                (eq.mul_unreduced::<9>(p0), eq.mul_unreduced::<9>(slope))
            })
            .reduce(
                || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                |a, b| (a.0 + b.0, a.1 + b.1),
            );
        (
            F::from_montgomery_reduce::<9>(t0_unr),
            F::from_montgomery_reduce::<9>(tinf_unr),
        )
    } else {
        let num_x1_bits = eq_poly.E_in_current_len().log_2();
        let x1_len = eq_poly.E_in_current_len();
        let x2_len = eq_poly.E_out_current_len();
        let (sum0_unr, suminf_unr) = (0..x2_len)
            .into_par_iter()
            .map(|x2| {
                let mut inner0_unr = F::Unreduced::<9>::zero();
                let mut inner_inf_unr = F::Unreduced::<9>::zero();
                for x1 in 0..x1_len {
                    let g = (x2 << num_x1_bits) | x1;
                    // indices for az0,bz0,az1,bz1
                    let az0 = bound_coeffs
                        .iter()
                        .find(|c| c.index == 2 * g)
                        .map(|c| c.value)
                        .unwrap_or(F::zero());
                    let bz0 = bound_coeffs
                        .iter()
                        .find(|c| c.index == 2 * g + 1)
                        .map(|c| c.value)
                        .unwrap_or(F::zero());
                    let az1 = bound_coeffs
                        .iter()
                        .find(|c| c.index == 2 * g + 2)
                        .map(|c| c.value)
                        .unwrap_or(F::zero());
                    let bz1 = bound_coeffs
                        .iter()
                        .find(|c| c.index == 2 * g + 3)
                        .map(|c| c.value)
                        .unwrap_or(F::zero());
                    let e_in = eq_poly.E_in_current()[x1];
                    let p0 = az0 * bz0;
                    let slope = (az1 - az0) * (bz1 - bz0);
                    inner0_unr += e_in.mul_unreduced::<9>(p0);
                    inner_inf_unr += e_in.mul_unreduced::<9>(slope);
                }
                let e_out = eq_poly.E_out_current()[x2];
                let inner0_red = F::from_montgomery_reduce::<9>(inner0_unr);
                let inner_inf_red = F::from_montgomery_reduce::<9>(inner_inf_unr);
                (
                    e_out.mul_unreduced::<9>(inner0_red),
                    e_out.mul_unreduced::<9>(inner_inf_red),
                )
            })
            .reduce(
                || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                |a, b| (a.0 + b.0, a.1 + b.1),
            );
        (
            F::from_montgomery_reduce::<9>(sum0_unr),
            F::from_montgomery_reduce::<9>(suminf_unr),
        )
    }
}

// =======================
// SumcheckInstance (Prover) for outer_round_batched (no uni-skip)
// =======================

#[derive(Allocative)]
pub struct OuterRoundBatchedSumcheckProver<F: JoltField> {
    #[allocative(skip)]
    preprocess: BytecodePreprocessing,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    /// Split-eq helper tracking binding state for the main outer Eq_τ polynomial
    eq_poly: GruenSplitEqPolynomial<F>,
    /// Static split-eq helper specialized for small-value optimization (no binding)
    eq_small_value: GruenSplitEqPolynomial<F>,
    /// Interleaved Az/Bz holder and binding workspace
    spartan_poly: RoundBatchedSpartanInterleavedPolynomial<NUM_SVO_ROUNDS, F>,
    /// Precomputed SVO accumulators across rounds
    svo_accums_zero: [F; NUM_ACCUMS_EVAL_ZERO],
    svo_accums_infty: [F; NUM_ACCUMS_EVAL_INFTY],
    /// Cached r for SVO rounds (in natural order of rounds)
    r_svo: Vec<F::Challenge>,
    /// Lagrange coefficients over ternary for current SVO prefix (size 3^i at round i)
    lagrange_coeffs: Vec<F>,
    /// Streaming cache: [az0,bz0,az1,bz1] at y=r_svo for each logical block
    block4_at_r: Vec<SparseCoefficient<F>>,
    /// Total rounds = step_vars + constraint_vars
    total_rounds: usize,
    /// Number of cycle bits (step vars)
    num_cycle_bits: usize,
    /// Dense Az,Bz used to compute remaining-round endpoints with the baseline implementation.
    baseline_az: DensePolynomial<F>,
    baseline_bz: DensePolynomial<F>,
}

impl<F: JoltField> OuterRoundBatchedSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "OuterRoundBatchedSumcheckProver::gen")]
    pub fn gen<T: Transcript>(
        trace: Arc<Vec<Cycle>>,
        bytecode_preprocessing: &BytecodePreprocessing,
        transcript: &mut T,
    ) -> Self {
        // Determine step and constraint vars
        let num_steps = trace.len();
        let num_step_vars = if num_steps > 0 { num_steps.log_2() } else { 0 };
        let padded_num_constraints = R1CS_CONSTRAINTS.len().next_power_of_two();
        let num_constraint_vars = if padded_num_constraints > 0 {
            padded_num_constraints.log_2()
        } else {
            0
        };
        let total_num_vars = num_step_vars + num_constraint_vars;

        // Sample tau for entire outer sumcheck (no uni-skip)
        let tau: Vec<F::Challenge> = transcript.challenge_vector_optimized::<F>(total_num_vars);

        // Run SVO preprocessing round to collect accumulators and initialize polynomial holder
        let (svo_zero, svo_infty, spartan_poly) = RoundBatchedSpartanInterleavedPolynomial::<
            NUM_SVO_ROUNDS,
            F,
        >::svo_sumcheck_round(
            bytecode_preprocessing, trace.as_ref(), &tau
        );

        // Eq helper for the sumcheck itself: use standard split-eq semantics over all variables.
        // Small-value optimization only changes how we _evaluate_ Az·Bz in early rounds, not the
        // outer Eq_τ(·) polynomial, so we keep the regular layout here.
        let eq_poly = GruenSplitEqPolynomial::new(&tau, BindingOrder::LowToHigh);

        // Static split-eq structure matching the partition used in `svo_sumcheck_round`.
        // This is used only for weighting blocks in the streaming + remaining rounds; it
        // is never mutated or bound.
        let potential_x_out_vars = total_num_vars / 2 - NUM_SVO_ROUNDS;
        let iter_num_x_out_vars = core::cmp::min(potential_x_out_vars, num_step_vars);
        let iter_num_x_in_vars = (num_step_vars + num_constraint_vars - NUM_SVO_ROUNDS)
            .saturating_sub(iter_num_x_out_vars);
        let eq_small_value = GruenSplitEqPolynomial::new_for_small_value(
            &tau,
            iter_num_x_out_vars,
            iter_num_x_in_vars,
            NUM_SVO_ROUNDS,
            Some(F::MONTGOMERY_R_SQUARE),
        );

        let (baseline_az, baseline_bz) = {
            use crate::zkvm::r1cs::constraints::R1CSConstraint;
            use crate::zkvm::r1cs::evaluation::BaselineConstraintEval;
            use crate::zkvm::r1cs::inputs::R1CSCycleInputs;

            let uniform_constraints: Vec<R1CSConstraint> =
                R1CS_CONSTRAINTS.iter().map(|c| c.cons).collect();

            // This mirrors OuterBaselineSumcheckProver::build_dense_polynomials.
            let num_steps = trace.len();
            let num_cycles_padded = if num_step_vars == 0 {
                1usize
            } else {
                1usize << num_step_vars
            };

            let domain_size = num_cycles_padded
                .checked_mul(padded_num_constraints)
                .expect("overflow computing baseline outer domain size (debug)");

            debug_assert!(
                num_steps <= num_cycles_padded,
                "trace length ({num_steps}) must be <= padded cycles ({num_cycles_padded})"
            );

            let total_vars_dbg = num_step_vars + num_constraint_vars;
            debug_assert_eq!(
                domain_size,
                1usize << total_vars_dbg,
                "baseline outer debug: domain_size != 2^{total_vars_dbg}"
            );

            let mut az_vals = vec![F::zero(); domain_size];
            let mut bz_vals = vec![F::zero(); domain_size];

            az_vals
                .par_iter_mut()
                .zip(bz_vals.par_iter_mut())
                .enumerate()
                .for_each(|(d, (az_ref, bz_ref))| {
                    let step_idx = d / padded_num_constraints;
                    let constraint_idx = d % padded_num_constraints;

                    if step_idx < num_steps && constraint_idx < uniform_constraints.len() {
                        let row = R1CSCycleInputs::from_trace::<F>(
                            bytecode_preprocessing,
                            trace.as_ref(),
                            step_idx,
                        );
                        let cons = &uniform_constraints[constraint_idx];
                        *az_ref = BaselineConstraintEval::eval_az(cons, &row);
                        *bz_ref = BaselineConstraintEval::eval_bz(cons, &row);
                    } else {
                        *az_ref = F::zero();
                        *bz_ref = F::zero();
                    }
                });

            (DensePolynomial::new(az_vals), DensePolynomial::new(bz_vals))
        };

        // Bind nothing yet; all rounds (SVO + streaming + remaining) will bind via
        // `SumcheckInstanceProver::bind`. Prepare instance:
        Self {
            preprocess: bytecode_preprocessing.clone(),
            trace,
            eq_poly,
            eq_small_value,
            spartan_poly,
            svo_accums_zero: svo_zero,
            svo_accums_infty: svo_infty,
            r_svo: Vec::with_capacity(NUM_SVO_ROUNDS),
            lagrange_coeffs: vec![F::one()],
            block4_at_r: Vec::new(),
            total_rounds: total_num_vars,
            num_cycle_bits: num_step_vars,
            baseline_az,
            baseline_bz,
        }
    }

    #[inline]
    fn compute_svo_quadratic_evals(&self, round: usize) -> (F, F) {
        debug_assert!(round < NUM_SVO_ROUNDS);
        // Offsets within flat accum arrays
        let (offsets_infty, offsets_zero) =
            svo_helpers::precompute_accumulator_offsets::<NUM_SVO_ROUNDS>();
        let start_inf = offsets_infty[round];
        let len_inf = svo_helpers::pow(3, round);
        let start_zero = offsets_zero[round];
        let len_zero = if round == 0 {
            0
        } else {
            svo_helpers::pow(3, round) - svo_helpers::pow(2, round)
        };

        let mut eval_inf = F::zero();
        for i in 0..len_inf {
            eval_inf += self.svo_accums_infty[start_inf + i] * self.lagrange_coeffs[i];
        }

        let mut eval_zero = F::zero();
        if len_zero > 0 {
            // Iterate global k in [0, 3^round), pick only non-binary positions in order
            let mut non_binary_idx = 0usize;
            let pow3 = svo_helpers::pow(3, round);
            for k in 0..pow3 {
                // detect any ternary digit == 2 among 'round' digits
                let mut is_non_binary = false;
                let mut tmp = k;
                for _ in 0..round {
                    if tmp % 3 == 2 {
                        is_non_binary = true;
                        break;
                    }
                    tmp /= 3;
                }
                if is_non_binary {
                    eval_zero +=
                        self.svo_accums_zero[start_zero + non_binary_idx] * self.lagrange_coeffs[k];
                    non_binary_idx += 1;
                }
            }
        }
        (eval_zero, eval_inf)
    }

    /// Compute endpoints (t0, t_inf) using the baseline dense Az,Bz
    /// implementation, sharing the same `eq_poly` state as the round-batched prover.
    fn baseline_compute_endpoints(&self) -> (F, F) {
        let eq_poly = &self.eq_poly;

        let n = self.baseline_az.len();
        debug_assert_eq!(n, self.baseline_bz.len());

        if eq_poly.E_in_current_len() == 1 {
            // groups are pairs (0,1)
            let groups = n / 2;
            (0..groups)
                .into_par_iter()
                .map(|g| {
                    let az0 = self.baseline_az[2 * g];
                    let az1 = self.baseline_az[2 * g + 1];
                    let bz0 = self.baseline_bz[2 * g];
                    let bz1 = self.baseline_bz[2 * g + 1];
                    let eq = eq_poly.E_out_current()[g];

                    let p0 = az0 * bz0;
                    let slope = (az1 - az0) * (bz1 - bz0);
                    (eq * p0, eq * slope)
                })
                .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
        } else {
            let num_x1_bits = eq_poly.E_in_current_len().log_2();
            let x1_len = eq_poly.E_in_current_len();
            let x2_len = eq_poly.E_out_current_len();

            (0..x2_len)
                .into_par_iter()
                .map(|x2| {
                    let mut inner0 = F::zero();
                    let mut inner_inf = F::zero();
                    for x1 in 0..x1_len {
                        let g = (x2 << num_x1_bits) | x1;
                        let az0 = self.baseline_az[2 * g];
                        let az1 = self.baseline_az[2 * g + 1];
                        let bz0 = self.baseline_bz[2 * g];
                        let bz1 = self.baseline_bz[2 * g + 1];
                        let e_in = eq_poly.E_in_current()[x1];
                        let p0 = az0 * bz0;
                        let slope = (az1 - az0) * (bz1 - bz0);
                        inner0 += e_in * p0;
                        inner_inf += e_in * slope;
                    }
                    let e_out = eq_poly.E_out_current()[x2];
                    (e_out * inner0, e_out * inner_inf)
                })
                .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for OuterRoundBatchedSumcheckProver<F>
{
    fn degree(&self) -> usize {
        3
    }
    fn num_rounds(&self) -> usize {
        self.total_rounds
    }
    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "OuterRoundBatchedSumcheckProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let (t0, tinf) = if round < NUM_SVO_ROUNDS {
            self.compute_svo_quadratic_evals(round)
        } else if round == NUM_SVO_ROUNDS {
            let ((a, b), block4) = compute_streaming_endpoints_and_block4::<F>(
                &self.preprocess,
                &self.trace,
                &self.eq_poly,
                &self.r_svo,
            );
            self.block4_at_r = block4;
            (a, b)
        } else {
            // Remaining rounds: for now, defer to the dense baseline endpoints for correctness.
            // The sparse bound-coeff path is kept for testing/comparison only.
            #[cfg(test)]
            {
                let (_dbg_t0, _dbg_tinf) = compute_remaining_endpoints_from_bound_coeffs::<F>(
                    &self.eq_poly,
                    &self.spartan_poly.bound_coeffs,
                );
                // If needed, we can compare `_dbg_t0/_dbg_tinf` against the baseline endpoints.
            }
            self.baseline_compute_endpoints()
        };

        #[cfg(test)]
        {
            let (t0_base, tinf_base) = self.baseline_compute_endpoints();
            if t0 != t0_base || tinf != tinf_base {
                println!(
                    "RoundBatched endpoints mismatch at round {round}: t0_rb={t0} t0_base={t0_base}, tinf_rb={tinf} tinf_base={tinf_base}",
                );
            }
        }

        self.eq_poly.gruen_poly_deg_3(t0, tinf, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "OuterRoundBatchedSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if round < NUM_SVO_ROUNDS {
            // Update SVO prefix state
            self.r_svo.push(r_j);
            if round + 1 < NUM_SVO_ROUNDS {
                // Update lagrange coefficients for next SVO round: expand by [1-r, r, r(r-1)]
                let lag = [F::one() - r_j, r_j.into(), r_j * (r_j - F::one())];
                let prev = core::mem::take(&mut self.lagrange_coeffs);
                let mut next = Vec::with_capacity(prev.len() * 3);
                for c in lag.iter() {
                    for p in prev.iter() {
                        next.push(*c * *p);
                    }
                }
                self.lagrange_coeffs = next;
            }
            self.eq_poly.bind(r_j);
            // Keep the baseline dense polynomials bound in lockstep with `eq_poly`,
            // since `compute_message` defers to `baseline_compute_endpoints` for rounds
            // after the streaming phase.
            rayon::join(
                || self.baseline_az.bind_parallel(r_j, BindingOrder::LowToHigh),
                || self.baseline_bz.bind_parallel(r_j, BindingOrder::LowToHigh),
            );
            return;
        }

        if round == NUM_SVO_ROUNDS {
            // Bind cached 4-at-r coefficients into Az/Bz at this r_j
            let bound = bind_block4_at_r::<F>(&self.block4_at_r, r_j);
            self.spartan_poly.bound_coeffs = bound;
            self.block4_at_r.clear();
            self.eq_poly.bind(r_j);
            rayon::join(
                || self.baseline_az.bind_parallel(r_j, BindingOrder::LowToHigh),
                || self.baseline_bz.bind_parallel(r_j, BindingOrder::LowToHigh),
            );
            return;
        }

        // Remaining rounds: bind Az/Bz bound coeffs one bit
        // Compute output size and allocate scratch
        let block_size = self
            .spartan_poly
            .bound_coeffs
            .len()
            .div_ceil(rayon::current_num_threads())
            .next_multiple_of(4);
        let chunks: Vec<_> = self
            .spartan_poly
            .bound_coeffs
            .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
            .collect();

        let output_sizes: Vec<_> = chunks
            .par_iter()
            .map(|chunk| RoundBatchedSpartanInterleavedPolynomial::<NUM_SVO_ROUNDS, F>::binding_output_length(chunk))
            .collect();
        let total_output_len: usize = output_sizes.iter().sum();
        if self.spartan_poly.binding_scratch_space.capacity() < total_output_len {
            self.spartan_poly.binding_scratch_space.reserve_exact(
                total_output_len - self.spartan_poly.binding_scratch_space.capacity(),
            );
        }
        unsafe {
            self.spartan_poly
                .binding_scratch_space
                .set_len(total_output_len);
        }

        let mut output_slices: Vec<&mut [SparseCoefficient<F>]> = Vec::with_capacity(chunks.len());
        let mut remainder = self.spartan_poly.binding_scratch_space.as_mut_slice();
        for slice_len in output_sizes {
            let (a, b) = remainder.split_at_mut(slice_len);
            output_slices.push(a);
            remainder = b;
        }

        chunks
            .par_iter()
            .zip_eq(output_slices.into_par_iter())
            .for_each(|(coeffs, output_slice)| {
                let mut output_index = 0usize;
                for block in coeffs.chunk_by(|x, y| x.index / 4 == y.index / 4) {
                    let block_index = block[0].index / 4;
                    let mut az_coeff: (Option<F>, Option<F>) = (None, None);
                    let mut bz_coeff: (Option<F>, Option<F>) = (None, None);
                    for coeff in block {
                        match coeff.index % 4 {
                            0 => az_coeff.0 = Some(coeff.value),
                            1 => bz_coeff.0 = Some(coeff.value),
                            2 => az_coeff.1 = Some(coeff.value),
                            3 => bz_coeff.1 = Some(coeff.value),
                            _ => {}
                        }
                    }
                    if az_coeff != (None, None) {
                        let (low, high) = (
                            az_coeff.0.unwrap_or(F::zero()),
                            az_coeff.1.unwrap_or(F::zero()),
                        );
                        output_slice[output_index] =
                            (2 * block_index, low + r_j * (high - low)).into();
                        output_index += 1;
                    }
                    if bz_coeff != (None, None) {
                        let (low, high) = (
                            bz_coeff.0.unwrap_or(F::zero()),
                            bz_coeff.1.unwrap_or(F::zero()),
                        );
                        output_slice[output_index] =
                            (2 * block_index + 1, low + r_j * (high - low)).into();
                        output_index += 1;
                    }
                }
                debug_assert_eq!(output_index, output_slice.len());
            });

        core::mem::swap(
            &mut self.spartan_poly.bound_coeffs,
            &mut self.spartan_poly.binding_scratch_space,
        );
        self.eq_poly.bind(r_j);
        rayon::join(
            || self.baseline_az.bind_parallel(r_j, BindingOrder::LowToHigh),
            || self.baseline_bz.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Opening point uses the sumcheck challenges; endianness matched by OpeningPoint impl
        let opening_point: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();

        // Witness openings at r_cycle
        let (r_cycle, _rx_var) = opening_point.r.split_at(self.num_cycle_bits);
        let r_cycle_point =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(r_cycle.to_vec()).match_endianness();
        let claimed_witness_evals =
            R1CSEval::compute_claimed_inputs(&self.preprocess, &self.trace, &r_cycle_point);
        for (i, input) in crate::zkvm::r1cs::inputs::ALL_R1CS_INPUTS
            .iter()
            .enumerate()
        {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                r_cycle_point.clone(),
                claimed_witness_evals[i],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

// =======================
// SumcheckInstance (Verifier) for outer_round_batched (no uni-skip)
// =======================
pub struct OuterRoundBatchedSumcheckVerifier<F: JoltField> {
    num_cycle_bits: usize,
    total_rounds: usize,
    tau: Vec<F::Challenge>,
    key: UniformSpartanKey<F>,
    _phantom: core::marker::PhantomData<F>,
}

impl<F: JoltField> OuterRoundBatchedSumcheckVerifier<F> {
    pub fn new(
        num_cycle_bits: usize,
        num_constraint_bits: usize,
        tau: Vec<F::Challenge>,
        key: UniformSpartanKey<F>,
    ) -> Self {
        Self {
            num_cycle_bits,
            total_rounds: num_cycle_bits + num_constraint_bits,
            tau,
            key,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for OuterRoundBatchedSumcheckVerifier<F>
{
    fn degree(&self) -> usize {
        3
    }
    fn num_rounds(&self) -> usize {
        self.total_rounds
    }
    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        F::zero()
    }
    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Recover all z_i(r_cycle) openings registered for Spartan outer.
        let r1cs_input_evals = ALL_R1CS_INPUTS.map(|input| {
            accumulator
                .get_virtual_polynomial_opening((&input).into(), SumcheckId::SpartanOuter)
                .1
        });

        // Inner sum-product over R1CS rows at a row-binding point derived from the
        // first two sumcheck challenges (streaming bit followed by a row bit).
        debug_assert!(
            sumcheck_challenges.len() >= 2,
            "round-batched outer: expected at least two challenges for row binding"
        );
        let rx_constr = &[sumcheck_challenges[0], sumcheck_challenges[1]];
        let inner_sum_prod = self
            .key
            .evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals);

        // Eq kernel over all remaining outer variables (cycle + constraint bits),
        // using the same τ layout as the round-batched prover.
        let r_rev: Vec<F::Challenge> = sumcheck_challenges.iter().rev().copied().collect();
        let eq_tau_r = EqPolynomial::<F>::mle(&self.tau, &r_rev);

        eq_tau_r * inner_sum_prod
    }
    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();

        // Witness openings at r_cycle
        let (r_cycle, _rx_var) = opening_point.r.split_at(self.num_cycle_bits);
        ALL_R1CS_INPUTS.iter().for_each(|input| {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                OpeningPoint::new(r_cycle.to_vec()),
            );
        });
    }
}
