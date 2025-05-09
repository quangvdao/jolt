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
    pub fn new_with_precompute(
        padded_num_constraints: usize,
        uniform_constraints: &[Constraint],
        cross_step_constraints: &[OffsetEqConstraint],
        flattened_polynomials: &[&MultilinearPolynomial<F>],
        tau: &[F],
    ) -> (Vec<(Vec<F>, Vec<F>)>, Self) {
        // Initialize special split eq polynomial for small value precompute.

        // Instead of having a clean split around the half-point, we will have a delayed +
        // wrap-around split around the point l + n/2, where l is the number of small value rounds

        // 0 ..... l ..... (l + n/2) ..... n
        //          <--- E_in --->
        // E_out ->                <-- E_out
        let _eq_poly = NewSplitEqPolynomial::new(tau);

        // TODO: add small value precompute logic

        let num_steps = flattened_polynomials[0].len();

        let num_chunks = rayon::current_num_threads().next_power_of_two() * 4;
        let chunk_size = num_steps.div_ceil(num_chunks);

        // The parallel computation below generates coefficients for each chunk. Each chunk's output
        // (`coeffs_chunk`) is internally sorted by index due to sequential processing of step_index
        // and constraint_index. `flat_map_iter`, when used with an IndexedParallelIterator (like
        // the range `0..num_chunks`), processes these chunks in parallel **but** then concatenates
        // their resulting iterators (which are `coeffs_chunk.into_iter()`) in the order of the
        // original indexed items.

        // This ensures that the final `ab_unbound_coeffs` vector is sorted by `index` without
        // requiring a separate sort step.
        let ab_unbound_coeffs: Vec<SparseCoefficient<i64>> = (0..num_chunks)
            .into_par_iter()
            .flat_map_iter(|chunk_index| {
                let mut coeffs_chunk = Vec::with_capacity(chunk_size * padded_num_constraints * 2); // Max 2 coeffs (A,B) per constraint
                for step_index in chunk_size * chunk_index..chunk_size * (chunk_index + 1).min(num_steps) {
                    // Uniform constraints
                    for (constraint_index, constraint) in uniform_constraints.iter().enumerate() {
                        let global_index = step_index * padded_num_constraints + constraint_index;

                        // Az
                        if !constraint.a.terms().is_empty() {
                            let az_coeff = constraint
                                .a
                                .evaluate_row_i64(flattened_polynomials, step_index);
                            if !az_coeff.is_zero() {
                                coeffs_chunk.push((global_index * 2, az_coeff).into());
                            }
                        }
                        // Bz
                        if !constraint.b.terms().is_empty() {
                            let bz_coeff = constraint
                                .b
                                .evaluate_row_i64(flattened_polynomials, step_index);
                            if !bz_coeff.is_zero() {
                                coeffs_chunk.push((global_index * 2 + 1, bz_coeff).into());
                            }
                        }
                    }

                    let next_step_index = if step_index + 1 < num_steps {
                        Some(step_index + 1)
                    } else {
                        None
                    };

                    // Cross-step constraints
                    for (constraint_index, constraint) in cross_step_constraints.iter().enumerate()
                    {
                        let global_index =
                                step_index * padded_num_constraints
                                + uniform_constraints.len()
                                + constraint_index;

                        // Az
                        let eq_a_eval = eval_offset_lc_i64(
                            &constraint.a,
                            flattened_polynomials,
                            step_index,
                            next_step_index,
                        );
                        let eq_b_eval = eval_offset_lc_i64(
                            &constraint.b,
                            flattened_polynomials,
                            step_index,
                            next_step_index,
                        );
                        let az_coeff = eq_a_eval - eq_b_eval;
                        if !az_coeff.is_zero() {
                            coeffs_chunk.push((global_index * 2, az_coeff).into());
                            #[cfg(test)]
                            {
                                let bz_coeff_cond_check = eval_offset_lc_i64(
                                    &constraint.cond,
                                    flattened_polynomials,
                                    step_index,
                                    next_step_index,
                                );
                                assert_eq!(bz_coeff_cond_check, 0, "Cross-step constraint {constraint_index} violated at step {step_index} for Az term");
                            }
                        } else {
                            // Bz
                            let bz_coeff = eval_offset_lc_i64(
                                &constraint.cond,
                                flattened_polynomials,
                                step_index,
                                next_step_index,
                            );
                            if !bz_coeff.is_zero() {
                                coeffs_chunk.push((global_index * 2 + 1, bz_coeff).into());
                            }
                        }
                    }
                }
                coeffs_chunk
            })
            .collect();

        #[cfg(test)]
        {
            // Check that indices are monotonically increasing
            if !ab_unbound_coeffs.is_empty() {
                let mut prev_index = ab_unbound_coeffs[0].index;
                for coeff in ab_unbound_coeffs[1..].iter() {
                    assert!(
                        coeff.index > prev_index,
                        "Indices not monotonically increasing: prev {}, current {}",
                        prev_index,
                        coeff.index
                    );
                    prev_index = coeff.index;
                }
            }
        }

        (
            vec![],
            Self {
                ab_unbound_coeffs,
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
    /// `t_i(0) = \sum_{x_out} E2[x_out] \sum_{x_in} E1[x_in] * (unbound_coeffs_a(r,0,x_in, x_out) *
    /// unbound_coeffs_b(r,0, x_in, x_out) - unbound_coeffs_c(r,0, x_in, x_out))`
    ///
    /// and `t_i(infty) = \sum_{x_out} E2[x_out] \sum_{x_in} E1[x_in] *
    /// (unbound_coeffs_a(r,infty,x_in, x_out) * unbound_coeffs_b(r,infty, x_in, x_out))`
    ///
    /// Here the "_{a,b,c}" subscript indicates the coefficients of `unbound_coeffs` corresponding
    /// to Az, Bz, Cz respectively. Importantly, since the eval at `r` is not cached, we will need
    /// to recompute it via another sum
    ///
    /// `unbound_coeffs_{a,b,c}(r, {0,infty}, x_in, x_out) = \sum_{y} eq(r, y) *
    /// unbound_coeffs_{a,b,c}(y, {0,infty}, x_in, x_out)`
    ///
    /// (and the eval at infty is computed as (eval at 1) - (eval at 0))
    ///
    /// Note that since `unbound_coeffs` are in sparse format, we will need to be more careful with
    /// indexing; see the old implementation for details.
    ///
    /// Finally, as we compute each `unbound_coeffs_{a,b,c}(r, {0,infty}, x_in, x_out)`, we will
    /// need to store them in `{az/bz/cz}_bound_coeffs`. (the eval at 1 will be eval at 0 + eval at
    /// infty)
    pub fn streaming_sumcheck_round<ProofTranscript: Transcript>(
        &mut self,
        eq_poly: &mut NewSplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
        r_challenges: &mut Vec<F>,
        round_polys: &mut Vec<CompressedUniPoly<F>>,
        claim: &mut F,
    ) {
        let num_rounds_done = r_challenges.len();
        // Assuming initial dense_len corresponds to the total number of variables ell
        let total_vars = self.dense_len.log_2();
        let remaining_vars = total_vars - num_rounds_done;

        // Index k of the *current* variable being bound (using 0..ell-1 index, LSB=0)
        // Sumcheck binds vars ell-1, ell-2, ..., 0.
        // Round i=0 binds var ell-1. Round i=ell-1 binds var 0.
        // k = total_vars - 1 - num_rounds_done
        let current_var_index_k = remaining_vars - 1;

        // Size of the evaluation domain for variables *lower* than k (indices 0..k-1)
        let N_k = 1 << current_var_index_k;
        // Size of the evaluation domain for variables *higher* than k (indices k+1..ell-1) -> these are the bound variables
        let N_high = 1 << num_rounds_done;

        // 1. Compute eq(r_{[<i]}, y_{high}) table
        // r_challenges contains r_0, ..., r_{i-1}
        let eq_r = EqPolynomial::evals(r_challenges);
        assert_eq!(eq_r.len(), N_high);

        // 2. Compute dense bound evaluations P(r_{[<i]}, u, x') for u=0 and u=1
        // These will be populated by iterating through ab_unbound_coeffs
        let mut az_evals_0: Vec<F> = vec![F::zero(); N_k];
        let mut az_evals_1: Vec<F> = vec![F::zero(); N_k];
        let mut bz_evals_0: Vec<F> = vec![F::zero(); N_k];
        let mut bz_evals_1: Vec<F> = vec![F::zero(); N_k];
        // We need to populate cz_evals_0/1 **during** the below loop, not after it.
        let mut cz_evals_0: Vec<F> = vec![F::zero(); N_k];
        let mut cz_evals_1: Vec<F> = vec![F::zero(); N_k];

        // Create iterators for parallel processing if possible, or manage indices carefully.
        // The core idea: for each (y_high, x_prime_idx) pair, find relevant sparse coeffs.
        // Since ab_unbound_coeffs is sorted, we can advance a single iterator.

        // Simplified sequential version for clarity, can be parallelized over x_prime_idx later
        // by giving each thread/task its own iterator or a sub-slice of ab_unbound_coeffs.
        let mut ab_iter = self.ab_unbound_coeffs.iter().peekable();

        // THIS NEEDS TO BE MERGED WITH THE BELOW LOOP!
        for y_high_idx in 0..N_high {
            // Iterate over assignments y_high (already bound variables)
            let eq_r_value = eq_r[y_high_idx];
            if eq_r_value.is_zero() {
                // If eq_r_value is zero, this y_high_idx contributes nothing to any sum, so we can skip processing its x_prime_idx range.
                // However, the iterator ab_iter must still be advanced past all coefficients related to this y_high_idx.
                // The simplest way is to calculate the max dense_idx for this y_high_idx and advance iterator past it.
                let max_r1cs_row_idx_for_y_high = (y_high_idx << (current_var_index_k + 1))
                    | ((1 << (current_var_index_k + 1)) - 1); // Max index if all lower bits are 1
                let max_sparse_idx_for_y_high = max_r1cs_row_idx_for_y_high * 2 + 1;
                while let Some(coeff) = ab_iter.peek() {
                    if coeff.index <= max_sparse_idx_for_y_high {
                        ab_iter.next();
                    } else {
                        break;
                    }
                }
                continue;
            }

            for x_prime_idx in 0..N_k {
                // Iterate over assignments to x' (variables lower than k)

                // Temporary variables to store original i64 coefficients for the current (y_high_idx, x_prime_idx) contribution
                let mut az_orig_val_at_y_xk0: i64 = 0;
                let mut bz_orig_val_at_y_xk0: i64 = 0;
                let mut az_orig_val_at_y_xk1: i64 = 0;
                let mut bz_orig_val_at_y_xk1: i64 = 0;

                // Target R1CS row index for (y_high, X_k=0, x_prime) and (y_high, X_k=1, x_prime)
                let r1cs_row_idx_if_Xk_0 = (y_high_idx << (current_var_index_k + 1))
                    | (0 << current_var_index_k)
                    | x_prime_idx;
                let r1cs_row_idx_if_Xk_1 = (y_high_idx << (current_var_index_k + 1))
                    | (1 << current_var_index_k)
                    | x_prime_idx;

                // Corresponding target sparse indices
                let target_sparse_idx_Az_Xk_0 = r1cs_row_idx_if_Xk_0 * 2;
                let target_sparse_idx_Bz_Xk_0 = r1cs_row_idx_if_Xk_0 * 2 + 1;
                let target_sparse_idx_Az_Xk_1 = r1cs_row_idx_if_Xk_1 * 2;
                let target_sparse_idx_Bz_Xk_1 = r1cs_row_idx_if_Xk_1 * 2 + 1;

                // --- Process Az for Xk=0 ---
                while let Some(coeff) = ab_iter.peek() {
                    if coeff.index < target_sparse_idx_Az_Xk_0 {
                        ab_iter.next();
                    } else {
                        if coeff.index == target_sparse_idx_Az_Xk_0 {
                            // Found Az(y_high, 0, x')
                            az_evals_0[x_prime_idx] += eq_r_value.mul_i64(coeff.value);
                            az_orig_val_at_y_xk0 = coeff.value; // Store for Cz calculation
                        }
                        break;
                    }
                }
                // --- Process Bz for Xk=0 ---
                // Iterator is now at or after Az_Xk_0
                while let Some(coeff) = ab_iter.peek() {
                    if coeff.index < target_sparse_idx_Bz_Xk_0 {
                        ab_iter.next();
                    } else {
                        if coeff.index == target_sparse_idx_Bz_Xk_0 {
                            // Found Bz(y_high, 0, x')
                            bz_evals_0[x_prime_idx] += eq_r_value.mul_i64(coeff.value);
                            bz_orig_val_at_y_xk0 = coeff.value; // Store for Cz calculation
                        }
                        break;
                    }
                }
                // --- Process Az for Xk=1 ---
                while let Some(coeff) = ab_iter.peek() {
                    if coeff.index < target_sparse_idx_Az_Xk_1 {
                        ab_iter.next();
                    } else {
                        if coeff.index == target_sparse_idx_Az_Xk_1 {
                            // Found Az(y_high, 1, x')
                            az_evals_1[x_prime_idx] += eq_r_value.mul_i64(coeff.value);
                            az_orig_val_at_y_xk1 = coeff.value; // Store for Cz calculation
                        }
                        break;
                    }
                }
                // --- Process Bz for Xk=1 ---
                while let Some(coeff) = ab_iter.peek() {
                    if coeff.index < target_sparse_idx_Bz_Xk_1 {
                        ab_iter.next();
                    } else {
                        if coeff.index == target_sparse_idx_Bz_Xk_1 {
                            // Found Bz(y_high, 1, x')
                            bz_evals_1[x_prime_idx] += eq_r_value.mul_i64(coeff.value);
                            bz_orig_val_at_y_xk1 = coeff.value; // Store for Cz calculation
                        }
                        break;
                    }
                }

                // Accumulate Cz contributions for the current y_high_idx
                // Cz_orig(y,0,x') = Az_orig(y,0,x') * Bz_orig(y,0,x')
                cz_evals_0[x_prime_idx] += eq_r_value.mul_i128(az_orig_val_at_y_xk0 as i128 * bz_orig_val_at_y_xk0 as i128);
                // Cz_orig(y,1,x') = Az_orig(y,1,x') * Bz_orig(y,1,x')
                cz_evals_1[x_prime_idx] += eq_r_value.mul_i128(az_orig_val_at_y_xk1 as i128 * bz_orig_val_at_y_xk1 as i128);
            } // End x_prime_idx loop
              // After processing all x_prime_idx for a given y_high_idx, the iterator ab_iter
              // should ideally be positioned at the start of the next y_high_idx block or past the
              // current one. The current structure with nested loops and a single shared iterator
              // is complex for parallelization and correctness. The iterator must be managed
              // carefully so it's correctly positioned for the *next* y_high_idx, x_prime_idx
              // combination.
        } // End y_high_idx loop

        // 3. Compute t_i(0) and t_i(infty) using the dense evals and eq_poly logic
        let (quadratic_eval_at_0, quadratic_eval_at_infty) = if eq_poly.E1_len() == 1 {
            // E1 is bound, E2 covers all remaining variables x' (size N_k)
            debug_assert_eq!(
                eq_poly.E2_len(),
                N_k,
                "E2_len should match N_k when E1_len is 1"
            );
            // Sum over all N_k points, weighted only by E2
            (0..N_k)
                .into_par_iter()
                .map(|idx_rest| {
                    // idx_rest corresponds to x'
                    let e2_val = eq_poly.E2_current()[idx_rest];
                    // P(r_{high}, u, x') values have been computed into *_evals_0/1
                    let az0 = az_evals_0[idx_rest];
                    let az1 = az_evals_1[idx_rest];
                    let bz0 = bz_evals_0[idx_rest];
                    let bz1 = bz_evals_1[idx_rest];
                    let cz0 = cz_evals_0[idx_rest];

                    let az_m = az1 - az0; // Az_m(r_{high}, x')
                    let bz_m = bz1 - bz0; // Bz_m(r_{high}, x')

                    let term_eval_at_0 = az0 * bz0 - cz0; // P(r_{high}, 0, x')
                    let term_eval_at_infty = az_m * bz_m; // P(r_{high}, infty, x')

                    (e2_val * term_eval_at_0, e2_val * term_eval_at_infty)
                })
                .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
        } else {
            // Nested sum: E2 corresponds to outer variables x_out, E1 to inner variables x_in
            // where x' = (x_out, x_in)
            let num_e2_points = eq_poly.E2_len(); // = 2^{num_vars_in_E2}
            let num_e1_points = eq_poly.E1_len(); // = 2^{num_vars_in_E1}
            debug_assert_eq!(
                N_k,
                num_e2_points * num_e1_points,
                "N_k should be product of E1_len and E2_len"
            );

            eq_poly
                .E2_current()
                .par_iter()
                .enumerate()
                .map(|(e2_idx, e2_val)| {
                    // e2_idx is index over x_out
                    let mut inner_sum_eval_at_0 = F::zero();
                    let mut inner_sum_eval_at_infty = F::zero();
                    let start_idx_rest = e2_idx * num_e1_points;

                    for e1_idx in 0..num_e1_points {
                        // e1_idx is index over x_in
                        let idx_rest = start_idx_rest + e1_idx; // Index for x' = (x_out, x_in)
                        let e1_val = eq_poly.E1_current()[e1_idx];

                        let az0 = az_evals_0[idx_rest];
                        let az1 = az_evals_1[idx_rest];
                        let bz0 = bz_evals_0[idx_rest];
                        let bz1 = bz_evals_1[idx_rest];
                        let cz0 = cz_evals_0[idx_rest];

                        let az_m = az1 - az0;
                        let bz_m = bz1 - bz0;

                        let term_eval_at_0 = az0 * bz0 - cz0;
                        let term_eval_at_infty = az_m * bz_m;

                        // Inner sum part: sum_{x_in} E1(x_in) * P(r_{high}, {0,infty}, x_out, x_in)
                        inner_sum_eval_at_0 += e1_val * term_eval_at_0;
                        inner_sum_eval_at_infty += e1_val * term_eval_at_infty;
                    }
                    // Outer sum part: E2(x_out) * (inner sum)
                    (
                        *e2_val * inner_sum_eval_at_0,
                        *e2_val * inner_sum_eval_at_infty,
                    )
                })
                .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
        };

        // 4. Process Round & Get Challenge r_i
        let r_i = process_eq_sumcheck_round(
            (quadratic_eval_at_0, quadratic_eval_at_infty),
            eq_poly, // Helper binds this according to r_i
            round_polys,
            r_challenges, // Helper pushes r_i into this
            claim,        // Helper updates this to s_i(r_i)
            transcript,   // Helper appends poly and gets challenge
        );

        // 5. Compute and Store Next Bound Coefficients
        // Resulting vectors represent P(r_{high}, r_i, x')
        let mut az_bound_next = vec![F::zero(); N_k];
        let mut bz_bound_next = vec![F::zero(); N_k];
        let mut cz_bound_next = vec![F::zero(); N_k];

        // This binding implements P'(x') = P(0, x') + r_i * (P(1, x') - P(0, x'))
        // where P' is the poly for the next round (bound in var k), and P is the poly for this round.
        az_bound_next
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, val)| {
                *val = az_evals_0[idx] + r_i * (az_evals_1[idx] - az_evals_0[idx]);
            });
        bz_bound_next
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, val)| {
                *val = bz_evals_0[idx] + r_i * (bz_evals_1[idx] - bz_evals_0[idx]);
            });
        cz_bound_next
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, val)| {
                *val = cz_evals_0[idx] + r_i * (cz_evals_1[idx] - cz_evals_0[idx]);
            });

        // Replace sparse unbound with dense bound coefficients for subsequent rounds
        // Note: This assumes streaming_sumcheck_round is called exactly once.
        self.ab_unbound_coeffs = Vec::new(); // Free memory
        self.az_bound_coeffs = DensePolynomial::new(az_bound_next);
        self.bz_bound_coeffs = DensePolynomial::new(bz_bound_next);
        self.cz_bound_coeffs = DensePolynomial::new(cz_bound_next);
        // Update dense_len to reflect the size of the *newly bound* polynomials
        // The internal DensePolynomials now represent functions over k variables.
        self.dense_len = N_k;
    }

    /// This function computes the polynomial for each of the remaining rounds, using the
    /// linear-time algorithm with split-eq optimizations
    ///
    /// At this point, we have computed the `{az/bz/cz}_bound_coeffs` for the current round.
    /// We need to compute:
    /// `t_i(0) = \sum_{x_out} E2[x_out] \sum_{x_in} E1[x_in] *
    /// (az_bound[0,x_in, x_out] * bz_bound[0, x_in, x_out] - cz_bound[0, x_in, x_out])`
    /// and
    /// `t_i(infty) = \sum_{x_out} E2[x_out] \sum_{x_in} E1[x_in] *
    /// az_bound[infty, x_in, x_out] * bz_bound[infty, x_in, x_out]`
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
        let (quadratic_eval_at_0, quadratic_eval_at_infty) = if eq_poly.E1_len() == 1 {
            // E1 part is fully bound. E2 covers all remaining eq_poly variables.
            debug_assert_eq!(
                eq_poly.E2_len(),
                num_x_rest_evals,
                "E2_len should match num_x_rest_evals when E1_len is 1"
            );

            // Combine iterators over the halves and E2
            az_low
                .par_iter()
                .zip(az_high.par_iter())
                .zip(bz_low.par_iter())
                .zip(bz_high.par_iter())
                .zip(cz_low.par_iter())
                .zip(eq_poly.E2_current().par_iter())
                .map(
                    |(((((az0_ref, az1_ref), bz0_ref), bz1_ref), cz0_ref), e2_val_ref)| {
                        let az0 = *az0_ref;
                        let az1 = *az1_ref;
                        let bz0 = *bz0_ref;
                        let bz1 = *bz1_ref;
                        let cz0 = *cz0_ref;
                        let e2_val = *e2_val_ref;

                        let az_m = az1 - az0; // Coeff of X_current_MSB in Az
                        let bz_m = bz1 - bz0; // Coeff of X_current_MSB in Bz

                        let term_eval_at_0 = az0 * bz0 - cz0;
                        let term_eval_at_infty = az_m * bz_m; // X^2 coeff of (Az*Bz)

                        (e2_val * term_eval_at_0, e2_val * term_eval_at_infty)
                    },
                )
                .reduce(
                    || (F::zero(), F::zero()),
                    |(acc_0, acc_infty), (val_0, val_infty)| (acc_0 + val_0, acc_infty + val_infty),
                )
        } else {
            // Nested sum structure: sum over E2 (outer), then sum over E1 (inner)
            let num_e2_points = eq_poly.E2_len();
            let num_e1_points = eq_poly.E1_len();
            debug_assert_eq!(
                num_x_rest_evals,
                num_e2_points * num_e1_points,
                "num_x_rest_evals should be product of E1_len and E2_len"
            );

            eq_poly
                .E2_current()
                .par_iter()
                .enumerate()
                .map(|(e2_idx, e2_val)| {
                    let mut inner_sum_eval_at_0 = F::zero();
                    let mut inner_sum_eval_at_infty = F::zero();

                    let start_idx_rest = e2_idx * num_e1_points;

                    // Iterate over the indices corresponding to this E2 chunk
                    for e1_idx in 0..num_e1_points {
                        let idx_rest = start_idx_rest + e1_idx;
                        let e1_val = eq_poly.E1_current()[e1_idx];

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

                        inner_sum_eval_at_0 += e1_val * term_eval_at_0;
                        inner_sum_eval_at_infty += e1_val * term_eval_at_infty;
                    }
                    (
                        *e2_val * inner_sum_eval_at_0,
                        *e2_val * inner_sum_eval_at_infty,
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
                    inner_sums.0 += E1_evals.0 * F::from_i128(az.0 * bz.0 - cz.0);
                    inner_sums.1 += E1_evals.1 * F::from_i128(az_eval_2 * bz_eval_2 - cz_eval_2);
                    inner_sums.2 += E1_evals.2 * F::from_i128(az_eval_3 * bz_eval_3 - cz_eval_3);
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
                            F::from_i128(low) + r_i * F::from_i128(high - low),
                        )
                            .into();
                        output_index += 1;
                    }
                    if bz_coeff != (None, None) {
                        let (low, high) = (bz_coeff.0.unwrap_or(0), bz_coeff.1.unwrap_or(0));
                        output_slice[output_index] = (
                            3 * block_index + 1,
                            F::from_i128(low) + r_i * F::from_i128(high - low),
                        )
                            .into();
                        output_index += 1;
                    }
                    if cz_coeff != (None, None) {
                        let (low, high) = (cz_coeff.0.unwrap_or(0), cz_coeff.1.unwrap_or(0));
                        output_slice[output_index] = (
                            3 * block_index + 2,
                            F::from_i128(low) + r_i * F::from_i128(high - low),
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
