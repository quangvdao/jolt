#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::{AppendToTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;
use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};

use ark_serialize::*;
use jolt_platform::{end_cycle_tracking, start_cycle_tracking};
use std::marker::PhantomData;

// Cycle-marker labels must be static strings: the tracer keys markers by the guest string pointer.
const CYCLE_BATCHED_SUMCHECK_VERIFY_TOTAL: &str = "batched_sumcheck_verify_total";
const CYCLE_BATCHED_SUMCHECK_VERIFY_PROOF_VERIFY: &str = "batched_sumcheck_verify_proof_verify";
const CYCLE_BATCHED_SUMCHECK_VERIFY_INPUT_CLAIMS: &str = "batched_sumcheck_verify_input_claims";
const CYCLE_BATCHED_SUMCHECK_VERIFY_CACHE_OPENINGS: &str = "batched_sumcheck_verify_cache_openings";
const CYCLE_BATCHED_SUMCHECK_VERIFY_EXPECTED_OUTPUT: &str =
    "batched_sumcheck_verify_expected_output_claim";

struct CycleMarkerGuard(&'static str);
impl CycleMarkerGuard {
    #[inline(always)]
    fn new(label: &'static str) -> Self {
        start_cycle_tracking(label);
        Self(label)
    }
}
impl Drop for CycleMarkerGuard {
    #[inline(always)]
    fn drop(&mut self) {
        end_cycle_tracking(self.0);
    }
}

#[inline]
fn maybe_override_batching_coeffs_for_recursion_stage2<F: JoltField>(
    batching_coeffs: &mut [F],
    max_num_rounds: usize,
) {
    // Debug helper: isolate which recursion Stage-2 sumcheck instance is inconsistent.
    //
    // We intentionally scope this override very narrowly to avoid perturbing other
    // batched sumchecks in the system.
    //
    // Current recursion Stage-2 shapes (depending on whether MML is included):
    // - instances == 14 or 16
    // - max_num_rounds == 19
    if (batching_coeffs.len() != 14 && batching_coeffs.len() != 16) || max_num_rounds != 19 {
        return;
    }
    let Ok(s) = std::env::var("JOLT_DEBUG_RECURSION_STAGE2_SINGLE_INDEX") else {
        return;
    };
    let Ok(idx) = s.parse::<usize>() else {
        return;
    };
    if idx >= batching_coeffs.len() {
        return;
    }
    for c in batching_coeffs.iter_mut() {
        *c = F::zero();
    }
    batching_coeffs[idx] = F::one();
}

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

        // Compute input claims once (used for transcript binding and batching).
        let input_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.input_claim(opening_accumulator))
            .collect();

        // Append input claims to transcript
        input_claims
            .iter()
            .for_each(|input_claim| transcript.append_scalar(input_claim));

        let mut batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());
        maybe_override_batching_coeffs_for_recursion_stage2(&mut batching_coeffs, max_num_rounds);
        if std::env::var_os("JOLT_DEBUG_SUMCHECK_INPUTS").is_some() {
            tracing::info!(
                "BatchedSumcheck::prove inputs: instances={} max_num_rounds={}",
                sumcheck_instances.len(),
                max_num_rounds
            );
            for (i, (sumcheck, input_claim)) in sumcheck_instances.iter().zip(input_claims.iter()).enumerate() {
                let num_rounds = sumcheck.num_rounds();
                let offset = sumcheck.round_offset(max_num_rounds);
                let scaled = input_claim.mul_pow_2(max_num_rounds - num_rounds);
                tracing::info!(
                    "  inst[{i}] rounds={num_rounds} offset={offset} degree={} input_claim={input_claim:?} scaled_input_claim={scaled:?}",
                    sumcheck.degree(),
                );
            }
        }

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
            .zip(input_claims.iter())
            .map(|(sumcheck, input_claim)| {
                let num_rounds = sumcheck.num_rounds();
                input_claim.mul_pow_2(max_num_rounds - num_rounds)
            })
            .collect();

        // For a shorter instance inside a longer batched sumcheck, there are `max_num_rounds - num_rounds`
        // dummy variables total, but they may be split between "dummy-before" and "dummy-after" depending
        // on the instance's `round_offset`.
        //
        // The initial scaling above accounts for *all* dummy variables, but if a shorter instance has
        // dummy-after rounds, the built-in dummy-round behavior (constant H with H(0)=H(1)=claim/2)
        // would divide out those dummy-after dimensions unless we compensate.
        //
        // We do this by keeping each such instance's claim scaled by 2^{dummy_after} throughout its
        // active window:
        // - feed the inner instance an unscaled `previous_claim / 2^{dummy_after}`
        // - scale the returned univariate by 2^{dummy_after}
        //
        // The subsequent dummy-after rounds then divide by 2 each time, cancelling the scale and
        // yielding the correct final claim at the end of the batched protocol.
        let batching_info: Vec<(usize, usize, usize, F, F)> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                let num_rounds = sumcheck.num_rounds();
                let offset = sumcheck.round_offset(max_num_rounds);
                let end = offset + num_rounds;
                assert!(
                    end <= max_num_rounds,
                    "sumcheck instance has invalid round_offset: offset({offset}) + num_rounds({num_rounds}) > max_num_rounds({max_num_rounds})"
                );
                let dummy_after = max_num_rounds - end;
                if dummy_after == 0 {
                    return (offset, num_rounds, dummy_after, F::one(), F::one());
                }
                let scale = F::one().mul_pow_2(dummy_after);
                let inv_scale = scale
                    .inverse()
                    .expect("2^dummy_after has an inverse in the field");
                (offset, num_rounds, dummy_after, scale, inv_scale)
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

            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .iter_mut()
                .zip(individual_claims.iter())
                .enumerate()
                .map(|(i, (sumcheck, previous_claim))| {
                    let (offset, num_rounds, dummy_after, scale, inv_scale) = batching_info[i];
                    let active = round >= offset && round < offset + num_rounds;
                    if active {
                        if dummy_after == 0 {
                            return sumcheck.compute_message(round - offset, *previous_claim);
                        }
                        // Keep the batched claim scaled by 2^{dummy_after}, while feeding the
                        // inner instance an unscaled claim and returning a scaled univariate.
                        let prev_unscaled = *previous_claim * inv_scale;
                        let poly_unscaled = sumcheck.compute_message(round - offset, prev_unscaled);
                        return poly_unscaled * scale;
                    }
                    // Variable is "dummy" for this instance: polynomial is independent of it,
                    // so the round univariate is constant with H(0)=H(1)=previous_claim/2.
                    UniPoly::from_coeff(vec![*previous_claim * two_inv])
                })
                .collect();

            // Linear combination of individual univariate polynomials
            let batched_univariate_poly: UniPoly<F> =
                univariate_polys.iter().zip(&batching_coeffs).fold(
                    UniPoly::from_coeff(vec![]),
                    |mut batched_poly, (poly, &coeff)| {
                        batched_poly += &(poly * coeff);
                        batched_poly
                    },
                );

            let compressed_poly = batched_univariate_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);

            // Cache individual claims for this round
            individual_claims
                .iter_mut()
                .zip(univariate_polys.into_iter())
                .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));

            #[cfg(test)]
            {
                // Sanity check
                let h0 = batched_univariate_poly.evaluate::<F>(&F::zero());
                let h1 = batched_univariate_poly.evaluate::<F>(&F::one());
                assert_eq!(
                    h0 + h1,
                    batched_claim,
                    "round {round}: H(0) + H(1) = {h0} + {h1} != {batched_claim}"
                );
                batched_claim = batched_univariate_poly.evaluate(&r_j);
            }

            for sumcheck in sumcheck_instances.iter_mut() {
                let num_rounds = sumcheck.num_rounds();
                let offset = sumcheck.round_offset(max_num_rounds);
                let active = round >= offset && round < offset + num_rounds;
                if active {
                    sumcheck.ingest_challenge(r_j, round - offset);
                }
            }

            compressed_polys.push(compressed_poly);
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
        let _verify_total = CycleMarkerGuard::new(CYCLE_BATCHED_SUMCHECK_VERIFY_TOTAL);
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

        // Compute input claims once (used for transcript binding and batching).
        let input_claims: Vec<F> = {
            let _input_claims = CycleMarkerGuard::new(CYCLE_BATCHED_SUMCHECK_VERIFY_INPUT_CLAIMS);
            sumcheck_instances
                .iter()
                .map(|sumcheck| sumcheck.input_claim(opening_accumulator))
                .collect()
        };

        // Append input claims to transcript
        input_claims
            .iter()
            .for_each(|input_claim| transcript.append_scalar(input_claim));

        let mut batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());
        maybe_override_batching_coeffs_for_recursion_stage2(&mut batching_coeffs, max_num_rounds);
        if std::env::var_os("JOLT_DEBUG_SUMCHECK_INPUTS").is_some() {
            tracing::info!(
                "BatchedSumcheck::verify inputs: instances={} max_num_rounds={}",
                sumcheck_instances.len(),
                max_num_rounds
            );
            for (i, (sumcheck, input_claim)) in sumcheck_instances.iter().zip(input_claims.iter()).enumerate() {
                let num_rounds = sumcheck.num_rounds();
                let offset = sumcheck.round_offset(max_num_rounds);
                let scaled = input_claim.mul_pow_2(max_num_rounds - num_rounds);
                tracing::info!(
                    "  inst[{i}] rounds={num_rounds} offset={offset} degree={} input_claim={input_claim:?} scaled_input_claim={scaled:?}",
                    sumcheck.degree(),
                );
            }
        }

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
            .zip(input_claims.iter())
            .map(|((sumcheck, coeff), input_claim)| {
                let num_rounds = sumcheck.num_rounds();
                input_claim.mul_pow_2(max_num_rounds - num_rounds) * coeff
            })
            .sum();

        let (output_claim, r_sumcheck) = {
            let _proof_verify = CycleMarkerGuard::new(CYCLE_BATCHED_SUMCHECK_VERIFY_PROOF_VERIFY);
            proof.verify(claim, max_num_rounds, max_degree, transcript)?
        };

        let mut expected_output_claim = F::zero();
        for (sumcheck, coeff) in sumcheck_instances.iter().zip(batching_coeffs.iter()) {
            let offset = sumcheck.round_offset(max_num_rounds);
            let r_slice = &r_sumcheck[offset..offset + sumcheck.num_rounds()];

            if let Some(label) = sumcheck.cycle_tracking_label() {
                let _instance_cycle = CycleMarkerGuard::new(label);
                // Cache polynomial opening claims, to be proven using either an
                // opening proof or sumcheck (in the case of virtual polynomials).
                {
                    let _cache =
                        CycleMarkerGuard::new(CYCLE_BATCHED_SUMCHECK_VERIFY_CACHE_OPENINGS);
                    sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
                }
                let claim = {
                    let _expected =
                        CycleMarkerGuard::new(CYCLE_BATCHED_SUMCHECK_VERIFY_EXPECTED_OUTPUT);
                    sumcheck.expected_output_claim(opening_accumulator, r_slice)
                };
                expected_output_claim += claim * coeff;
            } else {
                // Cache polynomial opening claims, to be proven using either an
                // opening proof or sumcheck (in the case of virtual polynomials).
                {
                    let _cache =
                        CycleMarkerGuard::new(CYCLE_BATCHED_SUMCHECK_VERIFY_CACHE_OPENINGS);
                    sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
                }
                let claim = {
                    let _expected =
                        CycleMarkerGuard::new(CYCLE_BATCHED_SUMCHECK_VERIFY_EXPECTED_OUTPUT);
                    sumcheck.expected_output_claim(opening_accumulator, r_slice)
                };
                expected_output_claim += claim * coeff;
            }
        }

        if output_claim != expected_output_claim {
            // Debug aid: when requested, emit per-instance contributions to help localize which
            // sumcheck instance has an inconsistent `expected_output_claim` implementation.
            if std::env::var_os("JOLT_DEBUG_SUMCHECK_VERBOSE").is_some() {
                tracing::error!(
                    "Batched sumcheck mismatch: output_claim={output_claim:?} expected_output_claim={expected_output_claim:?} (instances={}, max_num_rounds={}, max_degree={})",
                    sumcheck_instances.len(),
                    max_num_rounds,
                    max_degree
                );
                for (i, (sumcheck, coeff)) in
                    sumcheck_instances.iter().zip(batching_coeffs.iter()).enumerate()
                {
                    let offset = sumcheck.round_offset(max_num_rounds);
                    let r_slice = &r_sumcheck[offset..offset + sumcheck.num_rounds()];
                    let label = sumcheck.cycle_tracking_label().unwrap_or("<no label>");
                    let inst_expected = sumcheck.expected_output_claim(opening_accumulator, r_slice);
                    tracing::error!(
                        "  inst[{i}] label={label} rounds={} degree={} coeff={coeff:?} inst_expected={inst_expected:?}",
                        sumcheck.num_rounds(),
                        sumcheck.degree(),
                    );
                }
            }
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

impl<F, ProofTranscript> GuestSerialize for SumcheckInstanceProof<F, ProofTranscript>
where
    F: JoltField + GuestSerialize,
    ProofTranscript: Transcript,
{
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.compressed_polys.guest_serialize(w)
    }
}

impl<F, ProofTranscript> GuestDeserialize for SumcheckInstanceProof<F, ProofTranscript>
where
    F: JoltField + GuestDeserialize,
    ProofTranscript: Transcript,
{
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            compressed_polys: Vec::guest_deserialize(r)?,
            _marker: PhantomData,
        })
    }
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
            self.compressed_polys[i].append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i: F::Challenge = transcript.challenge_scalar_optimized::<F>();
            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i using the hint
            e = self.compressed_polys[i].eval_from_hint(&e, &r_i);
        }

        Ok((e, r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcripts::Blake2bTranscript;
    use ark_bn254::Fr;
    use ark_ff::{One, Zero};

    #[derive(Clone)]
    struct ToyMlSumcheck {
        /// Number of variables in the multilinear polynomial.
        num_rounds: usize,
        /// Global offset inside a batched sumcheck of length `max_num_rounds`.
        offset: usize,
        /// Full evaluation table on {0,1}^n, LSB-first variable indexing.
        evals: Vec<Fr>,
        /// Current evaluation table after binding the first `round` variables.
        cur: Vec<Fr>,
        /// How many local rounds have been bound so far.
        round: usize,
    }

    impl ToyMlSumcheck {
        fn new_affine(num_rounds: usize, offset: usize, constant: Fr, coeffs: &[Fr]) -> Self {
            assert_eq!(coeffs.len(), num_rounds);
            let size = 1usize << num_rounds;
            let mut evals = vec![Fr::zero(); size];
            for idx in 0..size {
                let mut v = constant;
                for j in 0..num_rounds {
                    let bit = ((idx >> j) & 1) as u64;
                    if bit == 1 {
                        v += coeffs[j];
                    }
                }
                evals[idx] = v;
            }
            Self {
                num_rounds,
                offset,
                cur: evals.clone(),
                evals,
                round: 0,
            }
        }

        fn expected_eval_at(&self, r: &[<Fr as JoltField>::Challenge]) -> Fr {
            assert_eq!(r.len(), self.num_rounds);
            let mut cur = self.evals.clone();
            for (j, &r_j) in r.iter().enumerate() {
                let rj: Fr = r_j.into();
                let one_minus = Fr::one() - rj;
                let mut next = vec![Fr::zero(); cur.len() / 2];
                for i in 0..next.len() {
                    next[i] = cur[2 * i] * one_minus + cur[2 * i + 1] * rj;
                }
                cur = next;
                assert_eq!(cur.len(), 1usize << (self.num_rounds - (j + 1)));
            }
            assert_eq!(cur.len(), 1);
            cur[0]
        }
    }

    impl SumcheckInstanceProver<Fr, Blake2bTranscript> for ToyMlSumcheck {
        fn degree(&self) -> usize {
            1
        }

        fn num_rounds(&self) -> usize {
            self.num_rounds
        }

        fn round_offset(&self, _max_num_rounds: usize) -> usize {
            self.offset
        }

        fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<Fr>) -> Fr {
            self.evals.iter().copied().sum()
        }

        fn compute_message(&mut self, round: usize, previous_claim: Fr) -> UniPoly<Fr> {
            assert_eq!(round, self.round);
            let claim_from_state: Fr = self.cur.iter().copied().sum();
            assert_eq!(
                previous_claim, claim_from_state,
                "previous_claim mismatch at local round {} (offset={}, num_rounds={})",
                round, self.offset, self.num_rounds
            );

            let mut h0 = Fr::zero();
            let mut h1 = Fr::zero();
            for i in 0..(self.cur.len() / 2) {
                h0 += self.cur[2 * i];
                h1 += self.cur[2 * i + 1];
            }
            assert_eq!(h0 + h1, previous_claim);
            UniPoly::from_coeff(vec![h0, h1 - h0])
        }

        fn ingest_challenge(&mut self, r_j: <Fr as JoltField>::Challenge, round: usize) {
            assert_eq!(round, self.round);
            let rj: Fr = r_j.into();
            let one_minus = Fr::one() - rj;
            let mut next = vec![Fr::zero(); self.cur.len() / 2];
            for i in 0..next.len() {
                next[i] = self.cur[2 * i] * one_minus + self.cur[2 * i + 1] * rj;
            }
            self.cur = next;
            self.round += 1;
        }

        fn cache_openings(
            &self,
            _accumulator: &mut ProverOpeningAccumulator<Fr>,
            _transcript: &mut Blake2bTranscript,
            _sumcheck_challenges: &[<Fr as JoltField>::Challenge],
        ) {
        }
    }

    impl SumcheckInstanceVerifier<Fr, Blake2bTranscript> for ToyMlSumcheck {
        fn degree(&self) -> usize {
            1
        }

        fn num_rounds(&self) -> usize {
            self.num_rounds
        }

        fn round_offset(&self, _max_num_rounds: usize) -> usize {
            self.offset
        }

        fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<Fr>) -> Fr {
            self.evals.iter().copied().sum()
        }

        fn expected_output_claim(
            &self,
            _accumulator: &VerifierOpeningAccumulator<Fr>,
            sumcheck_challenges: &[<Fr as JoltField>::Challenge],
        ) -> Fr {
            self.expected_eval_at(sumcheck_challenges)
        }

        fn cache_openings(
            &self,
            _accumulator: &mut VerifierOpeningAccumulator<Fr>,
            _transcript: &mut Blake2bTranscript,
            _sumcheck_challenges: &[<Fr as JoltField>::Challenge],
        ) {
        }
    }

    #[test]
    fn batched_sumcheck_supports_dummy_after_offset_0() {
        let mut prover_transcript = Blake2bTranscript::new(b"batched_sumcheck_dummy_after");
        let mut acc_p = ProverOpeningAccumulator::<Fr>::new(0);

        // Two instances: max rounds = 5, plus a shorter 3-round instance that is prefix-aligned (offset=0),
        // which introduces dummy-after rounds.
        let constant = Fr::from(7u64);
        let coeffs5 = [
            Fr::from(3u64),
            Fr::from(5u64),
            Fr::from(11u64),
            Fr::from(13u64),
            Fr::from(17u64),
        ];
        let coeffs3 = [Fr::from(19u64), Fr::from(23u64), Fr::from(29u64)];

        let mut inst_max = ToyMlSumcheck::new_affine(5, /*offset=*/ 0, constant, &coeffs5);
        let mut inst_short_prefix = ToyMlSumcheck::new_affine(3, /*offset=*/ 0, constant, &coeffs3);

        let mut provers: Vec<&mut dyn SumcheckInstanceProver<Fr, Blake2bTranscript>> = Vec::new();
        provers.push(&mut inst_max);
        provers.push(&mut inst_short_prefix);
        let (proof, r_sumcheck) =
            BatchedSumcheck::prove::<Fr, Blake2bTranscript>(provers, &mut acc_p, &mut prover_transcript);

        // Verify with transcript synchronization.
        let mut verifier_transcript = Blake2bTranscript::new(b"batched_sumcheck_dummy_after");
        verifier_transcript.compare_to(prover_transcript.clone());

        let mut acc_v = VerifierOpeningAccumulator::<Fr>::new(0);
        let inst_max_v = inst_max.clone();
        let inst_short_prefix_v = inst_short_prefix.clone();
        let mut verifiers: Vec<&dyn SumcheckInstanceVerifier<Fr, Blake2bTranscript>> = Vec::new();
        verifiers.push(&inst_max_v);
        verifiers.push(&inst_short_prefix_v);
        let r2 = BatchedSumcheck::verify::<Fr, Blake2bTranscript>(
            &proof,
            verifiers,
            &mut acc_v,
            &mut verifier_transcript,
        )
        .expect("batched sumcheck verify should succeed");

        assert_eq!(r2, r_sumcheck);
    }
}
