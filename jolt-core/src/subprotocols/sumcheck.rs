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
use crate::poly::split_eq_poly::{NewSplitEqPolynomial, SplitEqPolynomial};
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::r1cs::builder::{Constraint, OffsetEqConstraint};
use crate::r1cs::spartan::small_value_optimization::USES_SMALL_VALUE_OPTIMIZATION;
use crate::utils::errors::ProofVerifyError;
use crate::utils::mul_0_optimized;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, Transcript};
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

    #[tracing::instrument(skip_all, name = "Spartan2::prove_spartan_small_value")]
    pub fn prove_spartan_small_value<const NUM_SVO_ROUNDS: usize>(
        num_rounds: usize,
        padded_num_constraints: usize,
        uniform_constraints: &[Constraint],
        cross_step_constraints: &[OffsetEqConstraint],
        flattened_polys: &[&MultilinearPolynomial<F>],
        tau: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, [F; 3]) {
        let mut r = Vec::new();
        let mut polys = Vec::new();
        let mut claim = F::zero();

        // Clone the transcript at this point so that we could also test with non-svo sumcheck
        // #[cfg(test)]
        let mut old_transcript = transcript.clone();

        // First, precompute the accumulators and also the `NewSpartanInterleavedPolynomial`
        let (accums_zero, accums_infty, mut az_bz_cz_poly) =
            NewSpartanInterleavedPolynomial::<NUM_SVO_ROUNDS, F>::new_with_precompute(
                padded_num_constraints,
                uniform_constraints,
                cross_step_constraints,
                &flattened_polys,
                tau,
            );

        let mut lagrange_coeffs: Vec<F> = vec![F::one()];

        let mut eq_poly = NewSplitEqPolynomial::new(&tau);

        // Then, do the sumcheck logic
        for i in 0..NUM_SVO_ROUNDS {
            let mut quadratic_eval_0 = F::zero();
            let mut quadratic_eval_infty = F::zero();

            // Hard-coding for up to 3 svo rounds right now
            if USES_SMALL_VALUE_OPTIMIZATION {
                match NUM_SVO_ROUNDS {
                    1 => {
                        assert!(i == 0);
                        quadratic_eval_infty = accums_infty[0];
                    }
                    2 => match i {
                        0 => {
                            // A_0(I)
                            quadratic_eval_infty = accums_infty[0];
                        }
                        1 => {
                            // A_1(0, I) * L_1(I)
                            quadratic_eval_0 = accums_zero[0] * lagrange_coeffs[2];
                            // A_1(I, {0/1/I}) * L_1({0/1/I})
                            quadratic_eval_infty = accums_infty[1] * lagrange_coeffs[0]
                                + accums_infty[2] * lagrange_coeffs[1]
                                + accums_infty[3] * lagrange_coeffs[2];
                        }
                        _ => {
                            unreachable!("i must be less than NUM_SVO_ROUNDS!")
                        }
                    },
                    3 => {
                        match i {
                            0 => {
                                quadratic_eval_infty = accums_infty[0];
                            }
                            1 => {
                                quadratic_eval_0 = accums_zero[0] * lagrange_coeffs[2];
                                quadratic_eval_infty = accums_infty[1] * lagrange_coeffs[0]
                                    + accums_infty[2] * lagrange_coeffs[1]
                                    + accums_infty[3] * lagrange_coeffs[2];
                            }
                            2 => {
                                // We have accums_zero[1..6] corresponding to
                                // (0, 0, infty),(0, 1, infty),(0, infty 0),(0, infty, 1),(0, infty, infty)
                                // => matches with indices 2, 5, 6, 7, 8 of lagrange_coeffs respectively
                                // (recall the order MSB => LSB, 0 is MSB)

                                // We have accums_infty[4..] corresponding to
                                // (infty, v_1, v_2), where v_1, v_2 \in {0, 1, infty}
                                // Do full inner product over lagrange_coeffs
                                quadratic_eval_0 = accums_zero[1] * lagrange_coeffs[2]
                                    + accums_zero[2] * lagrange_coeffs[5]
                                    + accums_zero[3] * lagrange_coeffs[6]
                                    + accums_zero[4] * lagrange_coeffs[7]
                                    + accums_zero[5] * lagrange_coeffs[8];
                                quadratic_eval_infty = accums_infty[4] * lagrange_coeffs[0]
                                    + accums_infty[5] * lagrange_coeffs[1]
                                    + accums_infty[6] * lagrange_coeffs[2]
                                    + accums_infty[7] * lagrange_coeffs[3]
                                    + accums_infty[8] * lagrange_coeffs[4]
                                    + accums_infty[9] * lagrange_coeffs[5]
                                    + accums_infty[10] * lagrange_coeffs[6]
                                    + accums_infty[11] * lagrange_coeffs[7]
                                    + accums_infty[12] * lagrange_coeffs[8];
                            }
                            _ => {
                                unreachable!("i must be less than NUM_SVO_ROUNDS!")
                            }
                        }
                    }
                    _ => {
                        unreachable!("Hard-coding up to three small value rounds for now!")
                    }
                }
            }

            let r_i = process_eq_sumcheck_round(
                (quadratic_eval_0, quadratic_eval_infty),
                &mut eq_poly,
                &mut polys,
                &mut r,
                &mut claim,
                transcript,
            );

            // Lagrange coefficients for 0, 1, and infty, respectively
            let lagrange_coeffs_r_i = [F::one() - r_i, r_i, r_i * (r_i - F::one())];

            // Update Lagrange coefficients (so that indices for `r_i` is in the most significant digit):
            // L_{i+1} = lagrange_coeffs_r_i \otimes L_i
            // Update only needed for round < NUM_SVO_ROUNDS - 1
            if i < NUM_SVO_ROUNDS.saturating_sub(1) {
                lagrange_coeffs = lagrange_coeffs_r_i
                    .iter()
                    .flat_map(|lagrange_coeff| {
                        lagrange_coeffs
                            .iter()
                            .map(move |coeff| *lagrange_coeff * *coeff)
                    })
                    .collect();
            }
        }
        // Round NUM_SVO_ROUNDS : do the streaming sumcheck to compute cached values
        az_bz_cz_poly.streaming_sumcheck_round(
            &mut eq_poly,
            transcript,
            &mut r,
            &mut polys,
            &mut claim,
        );

        // Round (NUM_SVO_ROUNDS + 1)..num_rounds : do the linear time sumcheck
        for _ in (NUM_SVO_ROUNDS + 1)..num_rounds {
            az_bz_cz_poly.remaining_sumcheck_round(
                &mut eq_poly,
                transcript,
                &mut r,
                &mut polys,
                &mut claim,
            );
        }

        // #[cfg(test)]
        {
            let mut old_az_bz_cz_poly = SpartanInterleavedPolynomial::new(
                uniform_constraints,
                cross_step_constraints,
                flattened_polys,
                padded_num_constraints,
            );

            let mut old_r: Vec<F> = Vec::new();
            let mut old_polys: Vec<CompressedUniPoly<F>> = Vec::new();
            let mut old_claim = F::zero();
            let mut old_eq_poly = SplitEqPolynomial::new(tau);

            old_az_bz_cz_poly.first_sumcheck_round(
                &mut old_eq_poly,
                &mut old_transcript,
                &mut old_r,
                &mut old_polys,
                &mut old_claim,
            );

            for _ in 1..num_rounds {
                old_az_bz_cz_poly.subsequent_sumcheck_round(
                    &mut old_eq_poly,
                    &mut old_transcript,
                    &mut old_r,
                    &mut old_polys,
                    &mut old_claim,
                );
            }

            // Assert that the sumcheck polys at each steps are the same
            // (and hence the `r` challenges and the round claims are also the same)
            for round in 0..num_rounds {
                assert_eq!(
                    old_polys[round].coeffs_except_linear_term,
                    polys[round].coeffs_except_linear_term,
                    "The old and new method do not yield the same results for polynomial coeffs (round {})!", round
                );
                // If challenges are derived correctly from independent but identically starting transcripts:
                assert_eq!(
                    old_r[round], r[round],
                    "The old and new method do not yield the same challenges (round {})!",
                    round
                );
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
        eq_poly: &mut SplitEqPolynomial<F>,
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

    /// Version of `prove_spartan_cubic` that uses Gruen's optimization (but no small value optimization)
    #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::prove_spartan_cubic_with_gruen")]
    pub fn prove_spartan_cubic_with_gruen(
        num_rounds: usize,
        eq_poly: &mut NewSplitEqPolynomial<F>,
        az_bz_cz_poly: &mut SpartanInterleavedPolynomial<F>,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, [F; 3]) {
        let mut r: Vec<F> = Vec::new();
        let mut polys: Vec<CompressedUniPoly<F>> = Vec::new();
        let mut claim = F::zero();

        for round in 0..num_rounds {
            if round == 0 {
                az_bz_cz_poly.first_sumcheck_round_with_gruen(
                    eq_poly, transcript, &mut r, &mut polys, &mut claim,
                );
            } else {
                az_bz_cz_poly.subsequent_sumcheck_round_with_gruen(
                    eq_poly, transcript, &mut r, &mut polys, &mut claim,
                );
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

/// Helper function to encapsulate the common subroutine for sumcheck with eq poly factor:
/// - Compute the linear factor E_i(X) from the current eq-poly
/// - Reconstruct the cubic polynomial s_i(X) = E_i(X) * t_i(X) for the i-th round
/// - Compress the cubic polynomial
/// - Append the compressed polynomial to the transcript
/// - Derive the challenge for the next round
/// - Bind the cubic polynomial to the challenge
/// - Update the claim as the evaluation of the cubic polynomial at the challenge
///
/// Returns the derived challenge
pub fn process_eq_sumcheck_round<F: JoltField, ProofTranscript: Transcript>(
    quadratic_evals: (F, F), // (t_i(0), t_i(infty))
    eq_poly: &mut NewSplitEqPolynomial<F>,
    polys: &mut Vec<CompressedUniPoly<F>>,
    r: &mut Vec<F>,
    claim: &mut F,
    transcript: &mut ProofTranscript,
) -> F {
    let scalar_times_w_i = eq_poly.current_scalar * eq_poly.w[eq_poly.current_index - 1];

    let cubic_poly = UniPoly::from_linear_times_quadratic_with_hint(
        // The coefficients of `eq(w[(n - i)..], r[..i]) * eq(w[n - i - 1], X)`
        [
            eq_poly.current_scalar - scalar_times_w_i,
            scalar_times_w_i + scalar_times_w_i - eq_poly.current_scalar,
        ],
        quadratic_evals.0,
        quadratic_evals.1,
        *claim,
    );

    // Compress and add to transcript
    let compressed_poly = cubic_poly.compress();
    compressed_poly.append_to_transcript(transcript);

    // Derive challenge
    let r_i: F = transcript.challenge_scalar();
    r.push(r_i);
    polys.push(compressed_poly);

    // Evaluate for next round's claim
    *claim = cubic_poly.evaluate(&r_i);

    // Bind eq_poly for next round
    eq_poly.bind(r_i);

    r_i
}

#[cfg(test)]
mod tests {
    /// Display the ordering of the Lagrange coefficients in the small value sumcheck
    #[test]
    fn test_lagrange_coeffs_ordering() {
        // Challenges for 3 rounds
        let r_challenges = [
            2_i32, // r_0
            3_i32, // r_1
            4_i32, // r_2
        ];

        let mut lagrange_coeffs: Vec<i32> = vec![1_i32];

        for &r_i in &r_challenges {
            // Lagrange coefficients for 0, 1, and infty, respectively for the current r_i
            // [1 - r_i, r_i, r_i * (r_i - 1)]
            let lagrange_coeffs_r_i = [1_i32 - r_i, r_i, r_i * (r_i - 1_i32)];

            // Update Lagrange coefficients: L_{i+1} = lagrange_coeffs_r_i \otimes L_i
            lagrange_coeffs = lagrange_coeffs_r_i
                .iter()
                .flat_map(|&lagrange_coeff_from_r_i| {
                    lagrange_coeffs.iter().map(move |&coeff_from_prev_round| {
                        lagrange_coeff_from_r_i * coeff_from_prev_round
                    })
                })
                .collect();

            // Optional: Print intermediate states to trace
            // println!("After round with r_i = {:?}, lagrange_coeffs = {:?}", r_i, lagrange_coeffs);
        }

        // Expected values from the worked out example:
        // Initial: lagrange_coeffs = [1]
        //
        // Round 0 (r_0 = 2):
        //   lagrange_coeffs_r_0 = [1-2, 2, 2*(2-1)] = [-1, 2, 2]
        //   lagrange_coeffs = flat_map([-1, 2, 2] with [1])
        //                   = [-1*1, 2*1, 2*1]
        //                   = [-1, 2, 2]
        //
        // Round 1 (r_1 = 3):
        //   lagrange_coeffs_r_1 = [1-3, 3, 3*(3-1)] = [-2, 3, 6]
        //   lagrange_coeffs = flat_map([-2, 3, 6] with [-1, 2, 2])
        //                   = concat(
        //                       [-2*-1, -2*2, -2*2],       // from -2
        //                       [3*-1,  3*2,  3*2],        // from  3
        //                       [6*-1,  6*2,  6*2]         // from  6
        //                     )
        //                   = [2, -4, -4, -3, 6, 6, -6, 12, 12]
        //
        // Round 2 (r_2 = 4):
        //   lagrange_coeffs_r_2 = [1-4, 4, 4*(4-1)] = [-3, 4, 12]
        //   lagrange_coeffs = flat_map([-3, 4, 12] with [2, -4, -4, -3, 6, 6, -6, 12, 12])
        //                   = concat(
        //                       [-3*2, -3*-4, -3*-4, -3*-3, -3*6, -3*6, -3*-6, -3*12, -3*12], // from -3
        //                       [4*2,  4*-4,  4*-4,  4*-3,  4*6,  4*6,  4*-6,  4*12,  4*12],  // from  4
        //                       [12*2, 12*-4, 12*-4, 12*-3, 12*6, 12*6, 12*-6, 12*12, 12*12] // from 12
        //                     )
        //                   = [-6, 12, 12, 9, -18, -18, 18, -36, -36,
        //                      8, -16, -16, -12, 24, 24, -24, 48, 48,
        //                      24, -48, -48, -36, 72, 72, -72, 144, 144]
        let expected_coeffs: Vec<i32> = vec![
            -6, 12, 12, 9, -18, -18, 18, -36, -36, 8, -16, -16, -12, 24, 24, -24, 48, 48, 24, -48,
            -48, -36, 72, 72, -72, 144, 144,
        ];

        assert_eq!(lagrange_coeffs.len(), 27);
        assert_eq!(lagrange_coeffs, expected_coeffs);
    }
}
