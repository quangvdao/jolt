#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;

use ark_serialize::*;
use std::marker::PhantomData;

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>) {
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        // Append input claims to transcript
        sumcheck_instances.iter().for_each(|sumcheck| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            transcript.append_scalar(b"sumcheck_claim", &input_claim);
        });

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // To see why we may need to scale by a power of two, consider a batch of
        // two sumchecks:
        //   claim_a = \sum_x P(x)             where x \in {0, 1}^M
        //   claim_b = \sum_{x, y} Q(x, y)     where x \in {0, 1}^M, y \in {0, 1}^N
        // Then the batched sumcheck is:
        //   \sum_{x, y} A * P(x) + B * Q(x, y)  where A and B are batching coefficients
        //   = A * \sum_y \sum_x P(x) + B * \sum_{x, y} Q(x, y)
        //   = A * \sum_y claim_a + B * claim_b
        //   = A * 2^N * claim_a + B * claim_b
        let mut individual_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                let num_rounds = sumcheck.num_rounds();
                let input_claim = sumcheck.input_claim(opening_accumulator);
                input_claim.mul_pow_2(max_num_rounds - num_rounds)
            })
            .collect();

        #[cfg(test)]
        let mut batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);
        let two_inv = F::from_u64(2).inverse().unwrap();

        for round in 0..max_num_rounds {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let label = format!("Sumcheck round {round}");
                print_current_memory_usage(label.as_str());
            }

            let r_prev = r_sumcheck.last().copied();

            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .iter_mut()
                .zip(individual_claims.iter())
                .map(|(sumcheck, previous_claim)| {
                    let num_rounds = sumcheck.num_rounds();
                    let offset = sumcheck.round_offset(max_num_rounds);
                    let active = round >= offset && round < offset + num_rounds;
                    if !active {
                        UniPoly::from_coeff(vec![*previous_claim * two_inv])
                    } else {
                        let local_round = round - offset;
                        match (local_round, r_prev) {
                            (0, _) | (_, None) => {
                                sumcheck.compute_message(local_round, *previous_claim)
                            }
                            (_, Some(r)) => {
                                sumcheck.fused_bind_eval(r, local_round, *previous_claim)
                            }
                        }
                    }
                })
                .collect();

            let max_degree = univariate_polys
                .iter()
                .map(|p| p.coeffs.len())
                .max()
                .unwrap_or(0);
            let mut batched_coeffs = vec![F::zero(); max_degree];
            for (poly, &coeff) in univariate_polys.iter().zip(&batching_coeffs) {
                for (bc, pc) in batched_coeffs.iter_mut().zip(&poly.coeffs) {
                    *bc += *pc * coeff;
                }
            }
            let batched_univariate_poly = UniPoly::from_coeff(batched_coeffs);

            let compressed_poly = batched_univariate_poly.compress();

            // append the prover's message to the transcript
            transcript.append_scalars(b"sumcheck_poly", &compressed_poly.coeffs_except_linear_term);
            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);

            individual_claims
                .iter_mut()
                .zip(univariate_polys.into_iter())
                .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));

            #[cfg(test)]
            {
                let h0 = batched_univariate_poly.evaluate::<F>(&F::zero());
                let h1 = batched_univariate_poly.evaluate::<F>(&F::one());
                assert_eq!(
                    h0 + h1,
                    batched_claim,
                    "round {round}: H(0) + H(1) = {h0} + {h1} != {batched_claim}"
                );
                batched_claim = batched_univariate_poly.evaluate(&r_j);
            }

            compressed_polys.push(compressed_poly);
        }

        // Apply the final bind for each instance: the challenge from its last active round
        // was never applied inside the loop (the fused approach defers binding by one round).
        for sumcheck in sumcheck_instances.iter_mut() {
            let num_rounds = sumcheck.num_rounds();
            let offset = sumcheck.round_offset(max_num_rounds);
            let last_active_global = offset + num_rounds - 1;
            let r_final = r_sumcheck[last_active_global];
            sumcheck.ingest_challenge(r_final, num_rounds - 1);
        }

        // Allow each sumcheck instance to perform any end-of-protocol work (e.g. flushing
        // delayed bindings) after the final challenge has been ingested and before we cache
        // openings.
        for sumcheck in sumcheck_instances.iter_mut() {
            sumcheck.finalize();
        }

        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        for sumcheck in sumcheck_instances.iter() {
            // Instance-local slice can start at a custom global offset.
            let offset = sumcheck.round_offset(max_num_rounds);
            let r_slice = &r_sumcheck[offset..offset + sumcheck.num_rounds()];

            // Cache polynomial opening claims, to be proven using either an
            // opening proof or sumcheck (in the case of virtual polynomials).
            sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
        }

        (SumcheckInstanceProof::new(compressed_polys), r_sumcheck)
    }

    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        sumcheck_instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let max_degree = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.degree())
            .max()
            .unwrap();
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        // Append input claims to transcript
        sumcheck_instances.iter().for_each(|sumcheck| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            transcript.append_scalar(b"sumcheck_claim", &input_claim);
        });

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // To see why we may need to scale by a power of two, consider a batch of
        // two sumchecks:
        //   claim_a = \sum_x P(x)             where x \in {0, 1}^M
        //   claim_b = \sum_{x, y} Q(x, y)     where x \in {0, 1}^M, y \in {0, 1}^N
        // Then the batched sumcheck is:
        //   \sum_{x, y} A * P(x) + B * Q(x, y)  where A and B are batching coefficients
        //   = A * \sum_y \sum_x P(x) + B * \sum_{x, y} Q(x, y)
        //   = A * \sum_y claim_a + B * claim_b
        //   = A * 2^N * claim_a + B * claim_b
        let claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let num_rounds = sumcheck.num_rounds();
                let input_claim = sumcheck.input_claim(opening_accumulator);
                input_claim.mul_pow_2(max_num_rounds - num_rounds) * coeff
            })
            .sum();

        let (output_claim, r_sumcheck) =
            proof.verify(claim, max_num_rounds, max_degree, transcript)?;

        let mut expected_output_claim = F::zero();
        for (sumcheck, coeff) in sumcheck_instances.iter().zip(batching_coeffs.iter()) {
            let offset = sumcheck.round_offset(max_num_rounds);
            let r_slice = &r_sumcheck[offset..offset + sumcheck.num_rounds()];

            sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
            let claim = sumcheck.expected_output_claim(opening_accumulator, r_slice);
            expected_output_claim += claim * coeff;
        }

        if output_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(r_sumcheck)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
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
    ) -> Result<(F, Vec<F::Challenge>), ProofVerifyError> {
        let mut e = claim;
        let mut r: Vec<F::Challenge> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..self.compressed_polys.len() {
            // verify degree bound
            if self.compressed_polys[i].degree() > degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(
                    degree_bound,
                    self.compressed_polys[i].degree(),
                ));
            }

            // append the prover's message to the transcript
            transcript.append_scalars(
                b"sumcheck_poly",
                &self.compressed_polys[i].coeffs_except_linear_term,
            );

            //derive the verifier's challenge for the next round
            let r_i: F::Challenge = transcript.challenge_scalar_optimized::<F>();
            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i using the hint
            e = self.compressed_polys[i].eval_from_hint(&e, &r_i);
        }

        Ok((e, r))
    }
}
