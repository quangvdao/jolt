#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use crate::poly::spartan_interleaved_poly::{
    NewSpartanInterleavedPolynomial, SpartanInterleavedPolynomial,
};
use crate::poly::split_eq_poly::{OldSplitEqPolynomial, SplitEqPolynomial};
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::r1cs::constraints::LOG_ONE_FOURTH_NUM_CONSTRAINTS_PADDED;
use crate::utils::errors::ProofVerifyError;
use crate::utils::mul_0_optimized;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use common::rv_trace::MAX_ACTIVE_CIRCUIT_FLAGS;
use ark_serialize::*;
use rayon::prelude::*;
use std::marker::PhantomData;

pub trait Bindable<F: JoltField>: Sync {
    fn bind(&mut self, r: F);
}

/// Batched cubic sumcheck used in grand products
pub trait BatchedCubicSumcheck<F, ProofTranscript>: Bindable<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    fn compute_cubic(&self, eq_poly: &SplitEqPolynomial<F>, previous_round_claim: F) -> UniPoly<F>;
    fn final_claims(&self) -> (F, F);

    #[cfg(test)]
    fn sumcheck_sanity_check(&self, eq_poly: &SplitEqPolynomial<F>, round_claim: F);

    #[tracing::instrument(skip_all, name = "BatchedCubicSumcheck::prove_sumcheck")]
    fn prove_sumcheck(
        &mut self,
        claim: &F,
        eq_poly: &mut SplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, (F, F)) {
        let num_rounds = eq_poly.get_num_vars();

        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _ in 0..num_rounds {
            #[cfg(test)]
            self.sumcheck_sanity_check(eq_poly, previous_claim);

            let cubic_poly = self.compute_cubic(eq_poly, previous_claim);

            let compressed_poly = cubic_poly.compress();
            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            // derive the verifier's challenge for the next round
            let r_j = transcript.challenge_scalar();

            r.push(r_j);
            // bind polynomials to verifier's challenge
            self.bind(r_j);

            eq_poly.bind(r_j);

            previous_claim = cubic_poly.evaluate(&r_j);
            cubic_polys.push(compressed_poly);
        }

        #[cfg(test)]
        self.sumcheck_sanity_check(eq_poly, previous_claim);

        debug_assert_eq!(eq_poly.len(), 1);

        (
            SumcheckInstanceProof::new(cubic_polys),
            r,
            self.final_claims(),
        )
    }
}

impl<F: JoltField, ProofTranscript: Transcript> SumcheckInstanceProof<F, ProofTranscript> {
    /// Create a sumcheck proof for polynomial(s) of arbitrary degree.
    ///
    /// Params
    /// - `claim`: Claimed sumcheck evaluation (note: currently unused)
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `polys`: Dense polynomials to combine and sumcheck
    /// - `comb_func`: Function used to combine each polynomial evaluation
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (SumcheckInstanceProof, r_eval_point, final_evals)
    /// - `r_eval_point`: Final random point of evaluation
    /// - `final_evals`: Each of the polys evaluated at `r_eval_point`
    #[tracing::instrument(skip_all, name = "Sumcheck.prove")]
    pub fn prove_arbitrary<Func>(
        claim: &F,
        num_rounds: usize,
        polys: &mut Vec<MultilinearPolynomial<F>>,
        comb_func: Func,
        combined_degree: usize,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>)
    where
        Func: Fn(&[F]) -> F + std::marker::Sync,
    {
        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        #[cfg(test)]
        {
            let total_evals = 1 << num_rounds;
            let mut sum = F::zero();
            for i in 0..total_evals {
                let params: Vec<F> = polys.iter().map(|poly| poly.get_coeff(i)).collect();
                sum += comb_func(&params);
            }
            assert_eq!(&sum, claim, "Sumcheck claim is wrong");
        }

        for _round in 0..num_rounds {
            // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
            // for points {0, ..., |g(x)|}
            let mut eval_points = vec![F::zero(); combined_degree];

            let mle_half = polys[0].len() / 2;

            let accum: Vec<Vec<F>> = (0..mle_half)
                .into_par_iter()
                .map(|poly_term_i| {
                    let mut accum = vec![F::zero(); combined_degree];
                    // TODO(moodlezoup): Optimize
                    let evals: Vec<_> = polys
                        .iter()
                        .map(|poly| {
                            poly.sumcheck_evals(
                                poly_term_i,
                                combined_degree,
                                BindingOrder::HighToLow,
                            )
                        })
                        .collect();
                    for j in 0..combined_degree {
                        let evals_j: Vec<_> = evals.iter().map(|x| x[j]).collect();
                        accum[j] += comb_func(&evals_j);
                    }

                    accum
                })
                .collect();

            eval_points
                .par_iter_mut()
                .enumerate()
                .for_each(|(poly_i, eval_point)| {
                    *eval_point = accum
                        .par_iter()
                        .take(mle_half)
                        .map(|mle| mle[poly_i])
                        .sum::<F>();
                });

            eval_points.insert(1, previous_claim - eval_points[0]);
            let univariate_poly = UniPoly::from_evals(&eval_points);
            let compressed_poly = univariate_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // bound all tables to the verifier's challenge
            polys
                .par_iter_mut()
                .for_each(|poly| poly.bind(r_j, BindingOrder::HighToLow));
            previous_claim = univariate_poly.evaluate(&r_j);
            compressed_polys.push(compressed_poly);
        }

        let final_evals = polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim())
            .collect();

        (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
    }

    /// Using the small-value optimization, precompute the necessary data for the first few rounds of sumcheck for 
    /// the Spartan polynomial `eq(w, X) * (Az(X) * Bz(X) - Cz(X))`.
    /// This is the **generic** version of the small-value optimization.
    /// We have special handling for binary constraints and (very) sparse constraints (such as the single product constraint)
    pub fn precompute_small_value_spartan_generic(
        num_small_value_rounds: usize,
        eq_poly: &SplitEqPolynomial<F>,
        az_evals: &Vec<F>,
        bz_evals: &Vec<F>,
        cz_evals: &Vec<F>,
    ) -> Vec<(Vec<F>, Vec<F>)> {
        // Sanity check that the input polynomials are consistent
        #[cfg(test)]
        {
            assert_eq!(az_evals.len(), bz_evals.len());
            assert_eq!(az_evals.len(), cz_evals.len());
            (0..az_evals.len()).for_each(|i| {
                assert_eq!(az_evals[i] * bz_evals[i], cz_evals[i]);
            });
        }

        let len = az_evals.len();
        let E2_len = eq_poly.E2_len();
        let num_E2_chunks = len / E2_len;
        let E1_len = num_E2_chunks / (1 << num_small_value_rounds);
        
        assert!(len % E2_len == 0);
        assert!(num_E2_chunks % E1_len == 0);

        // Accumulator of the form `A[i][(v_0, ..., v_i , x_R)]`, where `v_0, ..., v_{i-1} \in \{0, 1, \infty\}`, `v_i \in \{0, \infty\}`
        // and `x_R \in {0,1}^{E2_len}`
        let mut accum: Vec<Vec<F>> = Vec::new();

        (0..num_small_value_rounds).for_each(|i| {
            accum.push(vec![F::zero(); 2 * (3 ^ i)]);
        });

        // Recall that we need to compute the sum
        // \sum_{x_R} eq(w_R, x_R) * \sum_{x_L} (az(v, u, x_L, x_R) * bz(v, u, x_L, x_R) - cz(v, u, x_L, x_R))

        // We can do this by first iterating over x_R (i.e. the E2 chunks), and then iterating over x_L (i.e. the E1 chunks),
        // and then summing over the values of `v` and `u`.
        // We do this map-reduce style for parallelization.
        let accum_flat = eq_poly.E2_current().par_iter()
            .zip(az_evals.par_chunks(num_E2_chunks))
            .zip(bz_evals.par_chunks(num_E2_chunks))
            .zip(cz_evals.par_chunks(num_E2_chunks))
            // .enumerate()
            .map(|(((E2_eval, az_E2_chunk), bz_E2_chunk), cz_E2_chunk)| {
                az_E2_chunk.par_chunks(E1_len)
                    .zip(bz_E2_chunk.par_chunks(E1_len))
                    .zip(cz_E2_chunk.par_chunks(E1_len))
                    .enumerate()
                    .map(|(i, ((az_E1_chunk, bz_E1_chunk), cz_E1_chunk))| {
                        todo!()
                });
                return vec![F::zero(); 3 ^ num_small_value_rounds - 1];
            }).reduce(|| vec![F::zero(); 3usize.pow(num_small_value_rounds as u32) - 1], |a, b| {
                a.par_iter().zip(b.par_iter()).map(|(a_val, b_val)| *a_val + *b_val).collect()
        });

        (0..num_small_value_rounds)
            .scan(&accum_flat[..], |remaining_slice, i| {
                let level_size = 2 * (3usize.pow(i as u32));
                if remaining_slice.len() >= level_size {
                    let (current_level_data, next_remaining_slice) = remaining_slice.split_at(level_size);
                    *remaining_slice = next_remaining_slice; // Update state (remaining slice)
                    // Split the current level data in half
                    let mid = level_size / 2;
                    let (first_half, second_half) = current_level_data.split_at(mid);
                    Some((first_half.to_vec(), second_half.to_vec())) // Output the tuple of halves
                } else {
                    // This case should be caught by the size check above, but included for robustness.
                    panic!("Slice length mismatch during scan-based unflattening");
                }
        }).collect()
    }

    pub fn precompute_small_value_binary_constraints(
        num_small_value_rounds: usize,
        max_num_ones : usize,
        eq_poly: &SplitEqPolynomial<F>,
        one_indices: &Vec<Vec<u8>>,
    ) -> Vec<(Vec<F>, Vec<F>)> {
        todo!()
    }

    pub fn merge_binary_and_other_constraints(
        binary_constraints: Vec<Vec<F>>,
        other_constraints: Vec<Vec<F>>,
    ) -> Vec<Vec<F>> {
        todo!()
    }

    // Computing the evaluations for linear-time sumcheck, after the small value rounds
    // pub fn prove_spartan_small_space_one_round(
    //     eq_poly: &mut SplitEqPolynomial<F>,
    // )

    #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::prove_spartan_cubic_new")]
    pub fn prove_spartan_cubic_new(
        num_rounds: usize,
        eq_poly: &mut SplitEqPolynomial<F>,
        az_bz_cz_poly: &mut NewSpartanInterleavedPolynomial<F>,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, [F; 3]) {

        let mut r: Vec<F> = Vec::new();
        let mut polys: Vec<CompressedUniPoly<F>> = Vec::new();
        let mut claim = F::zero();

        // TODO: Combine results from specialized sumchecks to get the claim for the first round
        // and potentially update the state of az_bz_cz_poly or polys/r.
        // The current implementation below assumes a structure similar to the old one,
        // which needs to be replaced.

        let instruction_accums = Self::precompute_small_value_binary_constraints(
            LOG_ONE_FOURTH_NUM_CONSTRAINTS_PADDED,
            1,
            eq_poly,
            &az_bz_cz_poly.instruction_indices,
        );

        let circuit_accums = Self::precompute_small_value_binary_constraints(
            LOG_ONE_FOURTH_NUM_CONSTRAINTS_PADDED,
            MAX_ACTIVE_CIRCUIT_FLAGS,
            eq_poly,
            &az_bz_cz_poly.circuit_indices,
        );

        // Two rounds is enough... can even do special handling (6 values)
        const REDUCED_NUM_SMALL_VALUE_ROUNDS: usize = 2;

        let other_accums = Self::precompute_small_value_spartan_generic(
            REDUCED_NUM_SMALL_VALUE_ROUNDS,
            eq_poly,
            &az_bz_cz_poly.bound_coeffs.0,
            &az_bz_cz_poly.bound_coeffs.1,
            &az_bz_cz_poly.bound_coeffs.2,
        );

        let mut lagrange_coeffs = vec![F::one(); 1];

        // Subsequent rounds (using the stubbed method)
        for round in 0..num_rounds {
            if round < LOG_ONE_FOURTH_NUM_CONSTRAINTS_PADDED {
                let quadratic_evals: (F, F) = {
                    let binary_constraints_eval_0 = lagrange_coeffs.iter()
                        .zip(instruction_accums[round].0.iter())
                        .zip(circuit_accums[round].0.iter())
                        .map(|((lagrange_coeff, instr_val), circ_val)| *lagrange_coeff * (*instr_val + *circ_val))
                        .fold(F::zero(), |a, b| a + b);

                    let binary_constraints_eval_infty = lagrange_coeffs.iter()
                        .zip(instruction_accums[round].1.iter())
                        .zip(circuit_accums[round].1.iter())
                        .map(|((lagrange_coeff, instr_val), circ_val)| *lagrange_coeff * (*instr_val + *circ_val))
                        .fold(F::zero(), |a, b| a + b);

                    if round < REDUCED_NUM_SMALL_VALUE_ROUNDS {
                        let other_constraints_eval_0 = lagrange_coeffs.iter()
                            .zip(other_accums[round].0.iter())
                            .map(|(lagrange_coeff, val)| *lagrange_coeff * *val)
                            .fold(F::zero(), |a, b| a + b);

                        let other_constraints_eval_infty = lagrange_coeffs.iter()
                            .zip(other_accums[round].1.iter())
                            .map(|(lagrange_coeff, val)| *lagrange_coeff * *val)
                            .fold(F::zero(), |a, b| a + b);

                        (binary_constraints_eval_0 + other_constraints_eval_0, binary_constraints_eval_infty + other_constraints_eval_infty)
                    } else {
                        todo!()
                }};

                let scalar_times_w_i = eq_poly.current_scalar * eq_poly.w[eq_poly.current_index - 1];

                let cubic_poly = UniPoly::from_linear_times_quadratic_with_hint(
                    // The coefficients of `eq(w[(n - i)..], r[..i]) * eq(w[n - i - 1], X)`
                    [
                        eq_poly.current_scalar - scalar_times_w_i,
                        scalar_times_w_i + scalar_times_w_i - eq_poly.current_scalar,
                    ],
                    quadratic_evals.0,
                    quadratic_evals.1,
                    claim,
                );

                let compressed_poly = cubic_poly.compress();
                compressed_poly.append_to_transcript(transcript);

                let r_i: F = transcript.challenge_scalar();
                r.push(r_i);
                polys.push(compressed_poly);
                
                claim = cubic_poly.evaluate(&r_i);
                eq_poly.bind(r_i);
                
                // Update Lagrange coefficients: L_{i+1} = L_i \otimes [r_i.square(), (F::one() - r_i).square(), r_i * (F::one() - r_i)]
                let lagrange_coeffs_r_i = [r_i.square(), (F::one() - r_i).square(), r_i * (F::one() - r_i)];
                lagrange_coeffs = lagrange_coeffs.iter()
                    .flat_map(|lagrange_coeff| {
                        lagrange_coeffs_r_i.iter().map(move |coeff| *lagrange_coeff * *coeff)
                    })
                    .collect();
                
            } else if round == LOG_ONE_FOURTH_NUM_CONSTRAINTS_PADDED {
                // Do the merging & compute the cached arrays for linear-time sumcheck, which may be delicate...
                todo!()
            } else {
                az_bz_cz_poly
                    .linear_time_sumcheck_round(eq_poly, transcript, &mut r, &mut polys, &mut claim);
            }
        }

        (
            SumcheckInstanceProof::new(polys),
            r,
            az_bz_cz_poly.final_sumcheck_evals(),
        )
    }

    #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::prove_spartan_cubic")]
    pub fn prove_spartan_cubic(
        num_rounds: usize,
        eq_poly: &mut OldSplitEqPolynomial<F>,
        az_bz_cz_poly: &mut SpartanInterleavedPolynomial<F>,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, [F; 3]) {
        let mut r: Vec<F> = Vec::new();
        let mut polys: Vec<CompressedUniPoly<F>> = Vec::new();
        let mut claim = F::zero();

        for round in 0..num_rounds {
            if round == 0 {
                az_bz_cz_poly
                    .first_sumcheck_round(eq_poly, transcript, &mut r, &mut polys, &mut claim);
            } else {
                az_bz_cz_poly
                    .subsequent_sumcheck_round(eq_poly, transcript, &mut r, &mut polys, &mut claim);
            }
        }

        (
            SumcheckInstanceProof::new(polys),
            r,
            az_bz_cz_poly.final_sumcheck_evals(),
        )
    }

    #[tracing::instrument(skip_all)]
    // A specialized sumcheck implementation with the 0th round unrolled from the rest of the
    // `for` loop. This allows us to pass in `witness_polynomials` by reference instead of
    // passing them in as a single `DensePolynomial`, which would require an expensive
    // concatenation. We defer the actual instantiation of a `DensePolynomial` to the end of the
    // 0th round.
    pub fn prove_spartan_quadratic(
        claim: &F,
        num_rounds: usize,
        poly_A: &mut DensePolynomial<F>,
        witness_polynomials: &[&MultilinearPolynomial<F>],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>) {
        let mut r: Vec<F> = Vec::with_capacity(num_rounds);
        let mut polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
        let mut claim_per_round = *claim;

        /*          Round 0 START         */

        let len = poly_A.len() / 2;
        let trace_len = witness_polynomials[0].len();
        // witness_polynomials
        //     .iter()
        //     .for_each(|poly| debug_assert_eq!(poly.len(), trace_len));

        // We don't materialize the full, flattened witness vector, but this closure
        // simulates it
        let witness_value = |index: usize| {
            if (index / trace_len) >= witness_polynomials.len() {
                F::zero()
            } else {
                witness_polynomials[index / trace_len].get_coeff(index % trace_len)
            }
        };

        let poly = {
            // eval_point_0 = \sum_i A[i] * B[i]
            // where B[i] = witness_value(i) for i in 0..len
            let eval_point_0: F = (0..len)
                .into_par_iter()
                .map(|i| {
                    if poly_A[i].is_zero() || witness_value(i).is_zero() {
                        F::zero()
                    } else {
                        poly_A[i] * witness_value(i)
                    }
                })
                .sum();
            // eval_point_2 = \sum_i (2 * A[len + i] - A[i]) * (2 * B[len + i] - B[i])
            // where B[i] = witness_value(i) for i in 0..len, B[len] = 1, and B[i] = 0 for i > len
            let mut eval_point_2: F = (1..len)
                .into_par_iter()
                .map(|i| {
                    if witness_value(i).is_zero() {
                        F::zero()
                    } else {
                        let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
                        let poly_B_bound_point = -witness_value(i);
                        mul_0_optimized(&poly_A_bound_point, &poly_B_bound_point)
                    }
                })
                .sum();
            eval_point_2 += mul_0_optimized(
                &(poly_A[len] + poly_A[len] - poly_A[0]),
                &(F::from_u8(2) - witness_value(0)),
            );

            let evals = [eval_point_0, claim_per_round - eval_point_0, eval_point_2];
            UniPoly::from_evals(&evals)
        };

        let compressed_poly = poly.compress();
        // append the prover's message to the transcript
        compressed_poly.append_to_transcript(transcript);

        //derive the verifier's challenge for the next round
        let r_i: F = transcript.challenge_scalar();
        r.push(r_i);
        polys.push(compressed_poly);

        // Set up next round
        claim_per_round = poly.evaluate(&r_i);

        // bound all tables to the verifier's challenge
        let (_, mut poly_B) = rayon::join(
            || poly_A.bound_poly_var_top_zero_optimized(&r_i),
            || {
                // Simulates `poly_B.bound_poly_var_top(&r_i)` by
                // iterating over `witness_polynomials`
                // We need to do this because we don't actually have
                // a `DensePolynomial` instance for `poly_B` yet.
                let zero = F::zero();
                let one = [F::one()];
                let W_iter = (0..len).into_par_iter().map(witness_value);
                let Z_iter = W_iter
                    .chain(one.into_par_iter())
                    .chain(rayon::iter::repeatn(zero, len));
                let left_iter = Z_iter.clone().take(len);
                let right_iter = Z_iter.skip(len).take(len);
                let B = left_iter
                    .zip(right_iter)
                    .map(|(a, b)| if a == b { a } else { a + r_i * (b - a) })
                    .collect();
                DensePolynomial::new(B)
            },
        );

        /*          Round 0 END          */

        for _i in 1..num_rounds {
            let poly = {
                let (eval_point_0, eval_point_2) =
                    Self::compute_eval_points_spartan_quadratic(poly_A, &poly_B);

                let evals = [eval_point_0, claim_per_round - eval_point_0, eval_point_2];
                UniPoly::from_evals(&evals)
            };

            let compressed_poly = poly.compress();
            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i: F = transcript.challenge_scalar();

            r.push(r_i);
            polys.push(compressed_poly);

            // Set up next round
            claim_per_round = poly.evaluate(&r_i);

            // bound all tables to the verifier's challenge
            rayon::join(
                || poly_A.bound_poly_var_top_zero_optimized(&r_i),
                || poly_B.bound_poly_var_top_zero_optimized(&r_i),
            );
        }

        let evals = vec![poly_A[0], poly_B[0]];
        drop_in_background_thread(poly_B);

        (SumcheckInstanceProof::new(polys), r, evals)
    }

    #[inline]
    #[tracing::instrument(skip_all, name = "Sumcheck::compute_eval_points_spartan_quadratic")]
    pub fn compute_eval_points_spartan_quadratic(
        poly_A: &DensePolynomial<F>,
        poly_B: &DensePolynomial<F>,
    ) -> (F, F) {
        let len = poly_A.len() / 2;
        (0..len)
            .into_par_iter()
            .map(|i| {
                // eval 0: bound_func is A(low)
                let eval_point_0 = if poly_B[i].is_zero() || poly_A[i].is_zero() {
                    F::zero()
                } else {
                    poly_A[i] * poly_B[i]
                };

                // eval 2: bound_func is -A(low) + 2*A(high)
                let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
                let eval_point_2 = if poly_B_bound_point.is_zero() {
                    F::zero()
                } else {
                    let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
                    mul_0_optimized(&poly_A_bound_point, &poly_B_bound_point)
                };

                (eval_point_0, eval_point_2)
            })
            .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct SumcheckInstanceProof<F: JoltField, ProofTranscript: Transcript> {
    pub compressed_polys: Vec<CompressedUniPoly<F>>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> SumcheckInstanceProof<F, ProofTranscript> {
    pub fn new(
        compressed_polys: Vec<CompressedUniPoly<F>>,
    ) -> SumcheckInstanceProof<F, ProofTranscript> {
        SumcheckInstanceProof {
            compressed_polys,
            _marker: PhantomData,
        }
    }

    /// Verify this sumcheck proof.
    /// Note: Verification does not execute the final check of sumcheck protocol: g_v(r_v) = oracle_g(r),
    /// as the oracle is not passed in. Expected that the caller will implement.
    ///
    /// Params
    /// - `claim`: Claimed evaluation
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `degree_bound`: Maximum allowed degree of the combined univariate polynomial
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, r)
    /// - `e`: Claimed evaluation at random point
    /// - `r`: Evaluation point
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F>), ProofVerifyError> {
        let mut e = claim;
        let mut r: Vec<F> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..self.compressed_polys.len() {
            // verify degree bound
            if self.compressed_polys[i].degree() != degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(
                    degree_bound,
                    self.compressed_polys[i].degree(),
                ));
            }

            // append the prover's message to the transcript
            self.compressed_polys[i].append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i = transcript.challenge_scalar();
            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i using the hint
            e = self.compressed_polys[i].eval_from_hint(&e, &r_i);
        }

        Ok((e, r))
    }
}
