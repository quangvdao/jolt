#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use crate::poly::spartan_interleaved_poly::{
    NewSpartanInterleavedPolynomial, SpartanInterleavedPolynomial,
};
use crate::poly::split_eq_poly::{OldSplitEqPolynomial, SplitEqPolynomial};
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::r1cs::constraints::LOG_ONE_HALF_NUM_CONSTRAINTS_PADDED;
use crate::utils::errors::ProofVerifyError;
use crate::utils::mul_0_optimized;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use crate::utils::{get_vec_by_fixed_bit, get_vec_by_fixed_bit_pair};
use ark_serialize::*;
use rayon::prelude::*;
use std::marker::PhantomData;

pub trait Bindable<F: JoltField>: Sync {
    fn bind(&mut self, r: F);
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
    eq_poly: &mut SplitEqPolynomial<F>,
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
        claim.clone(),
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

// Represents a point y in {0,1,inf}^ell
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TernaryVec(Vec<u8>);

impl TernaryVec {
    fn from_index(idx: usize, ell: usize) -> Self {
        let mut y_vec = vec![0u8; ell];
        let mut temp_idx = idx;
        for k in 0..ell {
            let digit = (temp_idx % 3) as u8;
            y_vec[ell - 1 - k] = digit;
            temp_idx /= 3;
        }
        TernaryVec(y_vec)
    }

    fn local_index(&self) -> usize {
        let mut index = 0;
        let mut power_of_3 = 1;
        for &val in self.0.iter().rev() {
            debug_assert!(val <= 2);
            index += (val as usize) * power_of_3;
            power_of_3 *= 3;
        }
        index
    }

    // Map (i, v', u') to 0..3^ell - 2, where v' in {0,1,inf}^{i}, u' in {0,inf}
    // Order: i (0..ell-1), u' (0=0, inf=2), v' (lexicographic base-3)
    fn flat_index(&self, i: usize) -> usize {
        debug_assert_eq!(self.0.len(), i + 1);
        let u_prime = self.0[i];
            debug_assert!(u_prime == 0 || u_prime == 2);

        let base_offset: usize = if i == 0 { 0 } else { 3usize.pow(i as u32) - 1 }; // Sum_{k=0}^{i-1} 2*3^k
        let u_offset = if u_prime == 0 { 0 } else { 3usize.pow(i as u32) }; // Size of the v' space for u'=0
        
        // Calculate index of v' within its {0,1,inf}^i space
        let mut v_prime_index = 0;
        let mut power_of_3 = 1;
        for k in 0..i {
                v_prime_index += (self.0[i - 1 - k] as usize) * power_of_3;
                power_of_3 *= 3;
        }
        
        base_offset + u_offset + v_prime_index
    }
    
    fn has_inf(&self) -> bool {
        self.0.iter().any(|&bit| bit == 2)
    }

    fn _is_valid(&self) -> bool {
        self.0.iter().all(|&ternary_val| ternary_val <= 2)
    }
}

// Stub for DP computation
// Input: flags = indices y in {0,1}^ell where Az(y)=1 for this (x_L, x_R)
// Output: Vec<i8> of size 3^ell storing Az(y) for y in {0,1,inf}^ell
// Side effect: updates max_k_abs encountered during calculation.
pub fn compute_all_az_dp(flags: &[u8], ell: usize, max_k_abs: &mut usize) -> Vec<i8> {
    // TODO: Implement the DP logic here
    let size = 3usize.pow(ell as u32);
    let mut az_values = vec![0i8; size];
    if ell == 0 { // Base case for recursion/DP if needed, though ell > 0 typically
        if !flags.is_empty() && flags[0] == 0 { // Check if Az() = 1
            // Decide how to represent Az() for ell=0 if applicable
        }
        return az_values; 
    }

    // Initialize for y in {0,1}^ell based on flags
    let binary_domain_size = 1 << ell;
    for y_bin_idx in 0..binary_domain_size {
        // Convert y_bin_idx to a binary vec to check against flags
        let mut is_flag = false;
        for &flag_val in flags {
            if flag_val as usize == y_bin_idx {
                is_flag = true;
                break;
            }
        }

        if is_flag {
            // Map binary y_bin_idx to its corresponding ternary index (all 0s and 1s)
            let mut y_ternary_vec = vec![0u8; ell];
            let mut temp_y_bin = y_bin_idx;
            for k in 0..ell {
                y_ternary_vec[ell - 1 - k] = (temp_y_bin % 2) as u8;
                temp_y_bin /= 2;
            }
            let ternary_idx = TernaryVec::from_index(y_bin_idx, ell).local_index();
        az_values[ternary_idx] = 1;
        }
    }

    // DP recurrence: compute Az for y containing infinity (2)
    let mut power_of_3_k = 1;
    for k in 0..ell { // Iterate through variable index k
        let power_of_3_kp1 = power_of_3_k * 3;
        for suffix_idx in 0..(size / power_of_3_kp1) {
            for prefix_idx in 0..power_of_3_k {
                let base_idx = suffix_idx * power_of_3_kp1 + prefix_idx;
                let idx_0 = base_idx + 0 * power_of_3_k; // y_k = 0
                let idx_1 = base_idx + 1 * power_of_3_k; // y_k = 1
                let idx_inf = base_idx + 2 * power_of_3_k; // y_k = inf
                
                az_values[idx_inf] = az_values[idx_1] - az_values[idx_0];
                
                let k_abs_val = az_values[idx_inf].abs() as usize;
                if k_abs_val > *max_k_abs {
                    *max_k_abs = k_abs_val;
                }
            }
        }
        power_of_3_k = power_of_3_kp1;
    }

    az_values
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
    /// the binary constraints: `eq(w, X) * Az(X) * Bz(X)`, where `Bz(X) = 1 - Az(X)` and
    /// `Az(y_{0-5}, x_L, x_R)` is given by the sparse representation `flag_indices`.
    /// (more precisely, it is `Az(y_{0-5}, 0, x_L, x_R)` within the larger R1CS constraint system,
    /// but we will ignore it here)
    /// 
    /// What this means is that we need to compute the accumulators
    /// 
    /// `accum[i][vec(v)][u] = Σ_{b \in {0,1}^{\ell - i}} Σ_{x_L \in {0,1}^{n/2 - \ell}} eq(w_L, x_L) 
    ///      * (Σ_{x_R \in {0,1}^{n/2}} eq(w_R, x_R) * Az(vec(v), u, b, x_L, x_R) * Bz(vec(v), u, b, x_L, x_R))`
    /// 
    /// for all `i = 0..5`, `(v_0, ..., v_{i-1}) ∈ {0, 1, ∞}^{i-1}`, and `u ∈ {0, ∞}`. (`\ell = num_small_value_rounds`)
    /// There are `2 + 2 * 3 + 2 * 3^2 + ... + 2 * 3^5 = 3^6 - 1 = 728` such accumulators.
    /// 
    /// We make several important observations:
    /// 
    /// (0) In order for the above sum to have good memory locality, we need to store Az evals via 
    /// chunks with the same left bits (vec(u), u, b, x_L part), and iterating over the right bits x_R.
    /// We do this when building `flag_indices`.
    /// 
    /// (1) Let vec* = (vec(v), u). If `S := {j : vec*(j) = ∞} \subset [0..5]`, then
    /// `Az(vec*, 0, x_L, x_R) = Σ_{T \subset S} (-1)^|T| * Az(vec*[T], 0, x_L, x_R)`,
    /// where `vec*[T]` is the vector vec* with the elements of T set to 0, and of `S \ T` set to 1.
    /// 
    /// Concretely, `Az(infty, 1, infty, 0..0, x_L, x_R) = Az(1, 1, 1, 0..0, x_L, x_R) - Az(1, 1, 0, 0..0, x_L, x_R) - Az(0, 0, 1, 0..0, x_L, x_R) + Az(0, 0, 0, 0..0, x_L, x_R)`.
    /// 
    /// (2) Since Bz = 1 - Az and Az is binary, we can show that:
    /// - `Bz(vec*, 0, x_L, x_R) = 1 - Az(vec*, 0, x_L, x_R)` if `vec*` has no \infty elements (and thus `Az(.) * Bz(.) = 0`)
    /// - `Bz(vec*, 0, x_L, x_R) = - Az(vec*, 0, x_L, x_R)` if `vec*` has one or more \infty elements (and thus `Az(.) * Bz(.) = - Az(.)^2`)
    /// 
    /// (3) Since `Az(y, 0, x_L, x_R)` is very sparse (at most `max_num_ones` ones, given by `flag_indices[x_L, x_R]`),
    /// `Az(vec*, 0, x_L, x_R)` is between `-max_num_ones` and `max_num_ones`.
    /// 
    /// Meaning that `Az(.) * Bz(.) \in {0, -1, -4, ..., -max_num_ones^2}`. And so we can pre-compute `eq(w_L, (0,x_L)) * (-i^2)` 
    /// for `i = 0..max_num_ones`, and use a lookup to get the contribution `eq(w_L, (0,x_L)) * Az(.) * Bz(.)` for any `vec*`.
    /// 
    /// (4) We can compute `Az(vec*, 0, x_L, x_R)` iteratively using the sparse representation `flag_indices[x_L][x_R]`, storing the results once again in a sparse representation.
    pub fn precompute_small_value_binary_constraints(
        num_small_value_rounds: usize,
        max_num_ones : usize,
        eq_poly: &SplitEqPolynomial<F>,
        flag_indices: &Vec<Vec<u8>>,
    ) -> Vec<(Vec<F>, Vec<F>)> {
        let ell = num_small_value_rounds;
        let n_vars = eq_poly.get_num_vars();
        let n_half = n_vars / 2;
        assert!(ell <= n_half, "num_small_value_rounds must be <= n/2");

        let E2_len = 1 << n_half;
        let E1_base_len = 1 << (n_half - ell); // Size of the x_L space
        let total_expected_flags_len = E1_base_len * E2_len; 
        // This assumption might be wrong if flag_indices is structured differently
        assert_eq!(flag_indices.len(), total_expected_flags_len, "flag_indices length mismatch");

        // --- Precomputations ---
        // Stores [-0^2, -1^2, -2^2, ..., -max_num_ones^2]
        let neg_sq: Vec<F> = (0..=max_num_ones)
            .map(|k| F::from_u64((k * k) as u64).neg())
            .collect();

        let E2_evals = eq_poly.E2_current();
        assert_eq!(E2_evals.len(), E2_len);

        // precomputed_E2_times_neg_k_sq[k-1][idx_R] = E2_evals[idx_R] * neg_sq[k]
        // Stores E2_evals * (-k^2) for k = 1 to max_num_ones
        let precomputed_E2_times_neg_k_sq: Vec<Vec<F>> = (1..=max_num_ones).map(|k| {
            E2_evals.iter().map(|E2_eval| *E2_eval * neg_sq[k]).collect()
        }).collect();

        // --- Mappings & Helper ---
        let num_accumulators = 3usize.pow(ell as u32) - 1; // Total number of outputs ((v', u'), i)
        let num_local_states = 3usize.pow(ell as u32); // Number of states y in {0,1,inf}^ell

        // Accumulator of the form `A[i][(v_0, ..., v_{i-1}, u)]`, where `v_0, ..., v_{i-1} ∈ {0, 1, ∞}`, `u ∈ {0, ∞}`
        // This holds a total of `2 + 2 * 3 + 2 * 3^2 + ... + 2 * 3^{5} = 3^6 - 1 = 728` elements
        let mut accum: Vec<Vec<F>> = Vec::new();

        (0..num_small_value_rounds).for_each(|i| {
            accum.push(vec![F::zero(); 2 * (3 ^ i)]);
        });

        // Recall that the accumulator formula is:
        // accum[i][(vec(v), u)] = Σ_{b \in {0,1}^{\ell - i}} Σ_{x_L \in {0,1}^{n/2 - \ell}} eq(w_L, (b, x_L))
        //          * (Σ_{x_R} eq(w_R, x_R) * (-a(vec(v), u, b, x_L, x_R)^2))

        // We want the following performance characteristics of our algorithm:
        // 1. It computes each a(vec(v), u, b, x_L, x_R) only once
        // 2. It re-uses the precomputed `eq(w_L, (0,x_L)) * (-i^2)` for each `i`
        // 3. It doesn't cost too much memory (3^5 * 2 * 2^(n/2 - 6) \approx 7.6 * 2^(n/2) is acceptable)

        // To satisfy these constraints, we will compute these accumulators via a `filling in` approach:
        // (1) Parallel iterate over `x_L \in {0,1}^{n/2 - 6}`,
        // (2) Iterate over `v \in {0, 1, ∞}^{5}`, `u \in {0, ∞}` (there are 3^5 * 2 = 486 such tuple (v, u)).
        // Initialize an `E2_sum(v,u)` for each `(v, u)` (and `x_L`), costing `3^5 * 2 * 2^(n/2 - 6)` memory.
        // (3) Iterate over `x_R \in {0,1}^{n/2}`,
        // (4)* Compute `Az(v, u, x_L, x_R)` over all `v, u` using `flag_indices[x_L][x_R]`, representing the result as a sparse vector,
        // (5) Use a lookup to get the (non-zero) contribution `eq(w_R, x_R) * Az(.) * Bz(.)` for the corresponding `(v,u)`,
        // (6) Add this contribution to the intermediate `E2_sum(v,u)` for the current E2 chunk.
        // (7) Look at all tuple `(i, v', u')` that is a _valid_ prefix of `(v, u)` (meaning that `(v', u')` is a prefix of `(v, u)` and that the rest of `(v, u)` does not contain any `∞`, i.e. is some `b \in {0,1}^{\ell - i}`).
        // (8) For each such tuple (i, v', u', b), add to the accumulator `accum[i][(v, u)]` the product
        // `eq_poly.E1[i][(b, x_L)] * E2_sum(v, u)`.

        // To compute step (4), we can use an iterative approach, computing in each variable y_0 to y_5
        // i.e. if \ell=2 (instead of 5), we compute:
        // `a(infty, 0) = a(1, 0) - a(0, 0)`, 
        // `a(0, infty) = a(0, 1) - a(0, 0)`, 
        // `a(1, infty) = a(1, 1) - a(1, 0)`,
        // `a(infty, infty) = a(1, 1) - a(1, 0) - a(0, 1) + a(0, 0) = a(infty, 1) - a(infty, 0)`
        // and so on...
        // This will be its own method

        // --- Main Computation ---
        let final_accum_flat: Vec<F> = (0..E1_base_len) // Parallelize over x_L
        .into_par_iter()
        .map(|idx_L| {
            // Per x_L accumulator, stores contributions to the final flat_accum vector
            let mut local_accum = vec![F::zero(); num_accumulators];
            // Stores Σ_{x_R} eq(w_R, x_R) * Term(y, x_L, x_R) for y in {0,1,inf}^ell
            let mut E2_sum = vec![F::zero(); num_local_states];
            let mut max_k_abs_local = 0usize; // Track max |Az(y)| for this x_L

            // Inner loop over x_R
            for idx_R in 0..E2_len {
                // Calculate index into flag_indices based on idx_L and idx_R.
                // The exact structure depends on how flag_indices was built.
                // Assuming flags are grouped by x_L first, then x_R:
                let global_flag_index = idx_L * E2_len + idx_R;
                let flags = &flag_indices[global_flag_index];

                // No need to compute if flags is empty, Az will be 0.
                if flags.is_empty() { continue; }

                let all_az_k_values: Vec<i8> = compute_all_az_dp(flags, ell, &mut max_k_abs_local);

                // Accumulate E2_sum using precomputed values
                for idx_y in 0..num_local_states {
                    let y = TernaryVec::from_index(idx_y, ell);
                    if !y.has_inf() { continue; } // Term is 0 if no infinity

                    let k = all_az_k_values[idx_y];
                    let k_abs = k.abs() as usize;

                    if k_abs > 0 && k_abs <= max_num_ones {
                        // Use precomputed E2_evals[idx_R] * neg_sq[k_abs]
                        let contrib = precomputed_E2_times_neg_k_sq[k_abs - 1][idx_R];
                        E2_sum[idx_y] += contrib;
                    }
                }
            }
            // Sanity check after processing all x_R for this x_L
            assert!(max_k_abs_local <= max_num_ones, "Max k encountered ({}) exceeds max_num_ones ({})", max_k_abs_local, max_num_ones);


            // Prefix Accumulation Step
            for idx_y in 0..num_local_states {
                let current_E2_sum = E2_sum[idx_y];
                if current_E2_sum.is_zero() { continue; }

                let y = TernaryVec::from_index(idx_y, ell);

                // Check all possible valid prefixes (i, v', u')
                for i in 0..ell { // Length of v' is i
                        let prefix_len = i + 1; // Length of (v', u')
                        let v_prime_u_prime = TernaryVec(y.0[0..prefix_len].to_vec());
                        let u_prime = v_prime_u_prime.0[i];
                        
                        // Prefix must end in 0 or inf
                        if u_prime == 1 { continue; } 
                        
                        let b_suffix = &y.0[prefix_len..ell];

                        // Suffix `b` must be binary (contain only 0s and 1s)
                        if b_suffix.iter().all(|&bit| bit == 0 || bit == 1) {
                            // Calculate index for b (binary interpretation)
                            let mut b_idx: usize = 0;
                            let mut power_of_2 = 1;
                            for &bit in b_suffix.iter().rev() {
                                b_idx += (bit as usize) * power_of_2;
                                power_of_2 *= 2;
                            }
                            let ell_minus_i = ell - prefix_len;
                            debug_assert_eq!(b_suffix.len(), ell_minus_i);

                        // TODO: Calculate eq_L_eval = eq(w_{L,i}, (b, x_L))
                            // This requires accessing eq evaluations for variables i..ell-1 (from b)
                            // combined with variables ell..n/2-1 (from x_L).
                            // Accessing eq_poly.E1 might be complex here.
                            // Placeholder:
                            let eq_L_eval = F::one();

                            let final_contrib = eq_L_eval * current_E2_sum;
                            let target_idx = v_prime_u_prime.flat_index(i);
                            local_accum[target_idx] += final_contrib;
                        }
                }
            }
            local_accum // Return contributions for this idx_L
        })
        .reduce(|| vec![F::zero(); num_accumulators], |mut a, b| {
            // Combine accumulators from different idx_L threads
            for (a_val, b_val) in a.iter_mut().zip(b.iter()) {
                *a_val += *b_val;
            }
            a
        });

        // --- Reshape final_accum_flat -> Vec<(Vec<F>, Vec<F>)> ---
        let mut result = Vec::with_capacity(ell);
        let mut current_idx = 0;
        for i in 0..ell {
            let level_size = 3usize.pow(i as u32); // Size of v' space for fixed i
            let vec0 = final_accum_flat[current_idx..(current_idx + level_size)].to_vec();
            current_idx += level_size;
            let vecinf = final_accum_flat[current_idx..(current_idx + level_size)].to_vec();
            current_idx += level_size;

            result.push((vec0, vecinf));
        }
        assert_eq!(current_idx, num_accumulators);

        result

        // // For each `i`, split the accumulator down in exact halves, and return the pair of halves
        // accum.iter().enumerate().map(|(i, accum_i)| {
        //     // assert!(accum_i.len() == 3usize.pow(i as u32) * 2);
        //     let mid = 3usize.pow(i as u32);
        //     (accum_i[..mid].to_vec(), accum_i[mid..].to_vec())
        // }).collect()
    }

    /// This function computes, for each `x_L ∈ {0,1}^{eq_poly.E1_len()}` and `x_R ∈ {0,1}^{eq_poly.E2_len()}`,
    /// Az[r, 0, x_L, x_R] = Σ_{y ∈ {0, 1}^{r.len()}} eq(r, y) * Az[y, 0, x_L, x_R]
    /// and 
    /// Bz[r, 0, x_L, x_R] = Σ_{y ∈ {0, 1}^{r.len()}} eq(r, y) * Bz[y, 0, x_L, x_R]
    /// 
    /// Note that since Bz[y, 0, x_L, x_R] = 1 - Az[y, 0, x_L, x_R], we have that
    /// Bz[r, 0, x_L, x_R] = Σ_{y} eq(r, y) - Az[r, 0, x_L, x_R]
    /// 
    /// Also, since Az[y, 0, x_L, x_R] is given via the sparse representation `flag_indices`,
    /// we can compute Az[r, 0, x_L, x_R] very quickly.
    /// 
    /// We can then derive the evaluation of
    /// t_i(0) = Σ_{x_R} E2[x_R] * Σ_{x_L} E1[x_L] * Az[r, 0, x_L, x_R] * Bz[r, 0, x_L, x_R]
    pub fn compute_evals_binary_constraints_after_small_value_rounds(
        eq_poly: &SplitEqPolynomial<F>,
        r: Vec<F>,
        flag_indices: &Vec<Vec<u8>>,
    ) -> (Vec<F>, Vec<F>, F) {
        let E1_len = eq_poly.E1_len(); // 2^{num_rounds / 2 - 7}
        let E2_len = eq_poly.E2_len(); // 2^{num_rounds / 2}
        let eq_poly_r = EqPolynomial::evals(&r);

        let sum_eq_poly_r = eq_poly_r.iter().sum::<F>();

        // Pre-allocate vectors with the correct size
        let total_size = E1_len * E2_len;
        assert_eq!(flag_indices.len(), total_size);
        let mut Az_evals_r = vec![F::zero(); total_size];
        let mut Bz_evals_r = vec![F::zero(); total_size];

        let E2_current_evals = eq_poly.E2_current(); // Get E2 evaluations once
        
        // Use par_chunks_mut to get mutable slices of the output vectors.
        // Zip them with the corresponding E2 evaluations.
        // Calculate values for each chunk, write them directly via the mutable slices,
        // and sum the quadratic contributions in the end.
        let quadratic_eval_0 = Az_evals_r
            .par_chunks_mut(E1_len) // Mutable slice of Az_evals_r for this E2 chunk
            .zip(Bz_evals_r.par_chunks_mut(E1_len)) // Mutable slice of Bz_evals_r for this E2 chunk
            .zip(E2_current_evals.par_iter()) // Corresponding E2 evaluation
            .enumerate() // Get the E2 chunk index `i` for flag_indices access
            .map(|(i, ((az_chunk, bz_chunk), E2_eval))| {
                let mut E2_sum = F::zero();
                // Compute Az and Bz evals for the current E1 chunk associated with this E2 chunk
                let (Az_evals_r_E2, Bz_evals_r_E2): (Vec<F>, Vec<F>) = eq_poly
                    .E1_current() // Evals for E1 associated with *this* E2 chunk
                    .iter()
                    .enumerate()
                    .map(|(j, E1_eval)| {
                        // Calculate indices based on E2 chunk index `i` and E1 index `j`
                        let global_index = i * E1_len + j;
                        let indices = &flag_indices[global_index];
                        let mut Az_eval_r = F::zero();
                        for index in indices {
                            Az_eval_r += eq_poly_r[*index as usize];
                        }
                        let Bz_eval_r = sum_eq_poly_r - Az_eval_r;
                        E2_sum += *E1_eval * Az_eval_r * Bz_eval_r;
                        (Az_eval_r, Bz_eval_r)
                    })
                    .unzip();

                // assert_eq!(az_chunk.len(), Az_evals_r_E2.len());
                // assert_eq!(bz_chunk.len(), Bz_evals_r_E2.len());
                az_chunk.copy_from_slice(&Az_evals_r_E2);
                bz_chunk.copy_from_slice(&Bz_evals_r_E2);

                // Return the quadratic contribution for this E2 chunk
                *E2_eval * E2_sum
            })
            .sum::<F>(); // Sum up the quadratic contributions from all E2 chunks

        (Az_evals_r, Bz_evals_r, quadratic_eval_0)
    }

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

        let flag_accums = Self::precompute_small_value_binary_constraints(
            LOG_ONE_HALF_NUM_CONSTRAINTS_PADDED, // 64 binary constraints
            6, // at most 6 flags (1 instruction flag + 5 circuit flags) can be active at once
            eq_poly,
            &az_bz_cz_poly.flag_indices,
        );

        let mut lagrange_coeffs = vec![F::one(); 1];

        // Recall that the indices are as follows:
        // (y_0, y_1, y_2, y_3, y_4, y_5, z, x_L, x_R)
        // where:
        // - `x_L \in {0,1}^{num_rounds / 2 - 7}`
        // - `x_R \in {0,1}^{num_rounds / 2}`
        // - If `z = 0`, then `y_{0-5}` corresponds to the 64 binary constraints
        // - If `z = 1` and `y_5 = 0`, then `y_{0-4}` corresponds to empty padded constraints
        // - If `z = 1` and `y_5 = 1`, then `y_{0-4}` corresponds to the 32 other (non-binary) constraints

        // We need to derive the right slices of E1_poly for the binary & other constraints
        // 1. For binary constraints: in round i = 0..4, we have:
        // - eq_poly.E1_current() ranges over (y_{i+1}, ..., y_5, z, x_L)
        // We need to only take the slices where `z = 0` for the binary constraints
        // 2. For other constraints: in round i = 0..3, we have:
        // - eq_poly.E1_current() ranges over (y_{i+1}, ..., y_5, z, x_L)
        // We need to only take the slices where `z = 1` and `y_5 = 1` for the other constraints

        // Rounds 0, 1, 2, 3, 4, 5
        for round in 0..LOG_ONE_HALF_NUM_CONSTRAINTS_PADDED {

            // Compute the contributions from the binary constraints to the evaluations t_i(0), t_i(infty)
            let binary_constraints_eval_0 = lagrange_coeffs.iter()
            .zip(flag_accums[round].0.iter())
            .map(|(lagrange_coeff, val)| *lagrange_coeff * *val)
            .fold(F::zero(), |a, b| a + b);

            let binary_constraints_eval_infty = lagrange_coeffs.iter()
                .zip(flag_accums[round].1.iter())
                .map(|(lagrange_coeff, val)| *lagrange_coeff * *val)
                .fold(F::zero(), |a, b| a + b);

            let E1_evals_for_other_constraints = get_vec_by_fixed_bit_pair(eq_poly.E1_current(), 4 - round, true, 5 - round, true);

            // For round 0, compute the contribution from the other constraints using specialized method
            // with handling of `i128` instead of full `F`
            if round == 0 {
                let az_bz_cz_chunks = az_bz_cz_poly.chunks_first_round();

                let (other_evals_0, other_evals_infty) = az_bz_cz_poly
                    .compute_first_round_other_evals(az_bz_cz_chunks, &E1_evals_for_other_constraints, eq_poly.E2_current());

                let quadratic_evals: (F, F) = (binary_constraints_eval_0 + other_evals_0, binary_constraints_eval_infty + other_evals_infty);

                let r_i = process_eq_sumcheck_round(quadratic_evals, eq_poly, &mut polys, &mut r, &mut claim, transcript);

                // Lagrange coefficients for 0, 1, and infty, respectively
                let lagrange_coeffs_r_i = [F::one() - r_i, r_i, r_i * (r_i - F::one())];

                // Update Lagrange coefficients (so that indices for `r_i` is in the most significant digit):
                // L_{i+1} = lagrange_coeffs_r_i \otimes L_i
                lagrange_coeffs = lagrange_coeffs_r_i.iter()
                    .flat_map(|lagrange_coeff| {
                        lagrange_coeffs.iter().map(move |coeff| *lagrange_coeff * *coeff)
                    })
                    .collect();

                // TODO: can't do mutable borrow here, may need to revert to having sumcheck logic inside az_bz_cz_poly
                // az_bz_cz_poly.bind_first_round_other_coeffs(az_bz_cz_chunks, r_i);
            } else {
                let az_bz_cz_chunks = az_bz_cz_poly.chunks_subsequent_round();

                let (other_evals_0, other_evals_infty) = az_bz_cz_poly
                    .compute_subsequent_round_other_evals(az_bz_cz_chunks, &E1_evals_for_other_constraints, eq_poly.E2_current());

                // Combine evaluations
                let quadratic_evals: (F, F) = (binary_constraints_eval_0 + other_evals_0, binary_constraints_eval_infty + other_evals_infty);

                let r_i = process_eq_sumcheck_round(quadratic_evals, eq_poly, &mut polys, &mut r, &mut claim, transcript);

                // Lagrange coefficients for 0, 1, and infty, respectively
                let lagrange_coeffs_r_i = [F::one() - r_i, r_i, r_i * (r_i - F::one())];

                // Update Lagrange coefficients (so that indices for `r_i` is in the most significant digit):
                // L_{i+1} = lagrange_coeffs_r_i \otimes L_i
                lagrange_coeffs = lagrange_coeffs_r_i.iter()
                    .flat_map(|lagrange_coeff| {
                        lagrange_coeffs.iter().map(move |coeff| *lagrange_coeff * *coeff)
                    })
                    .collect();
                
                // az_bz_cz_poly.bind_subsequent_round_other_coeffs(az_bz_cz_chunks, r_i);
            }
        }
        // Round 6
        // For this round, we will compute the arrays
        // P[r_0, ..., r_5, (z ∈ {0, 1}), x'] for P ∈ {Az, Bz, Cz}
        // and use those to compute the evaluations t_6(0), t_6(infty)

        // If `z = 0`, then we are computing the evaluations of the binary constraints
        // This can be computed quickly in a subroutine
        let (Az_evals_0, Bz_evals_0, quadratic_eval_0) =
            Self::compute_evals_binary_constraints_after_small_value_rounds(eq_poly, r.clone(), &az_bz_cz_poly.flag_indices);
            
        // If `z = 1`, then we compute the evaluations of the other constraints
        // This corresponds to coalescing / uninterleaving the other constraints
        let (Az_other_evals_1, Bz_other_evals_1, Cz_other_evals_1) =
            (vec![F::zero(); 32], vec![F::zero(); 32], vec![F::zero(); 32]);
        // az_bz_cz_poly.uninterleave(eq_poly.E2_current(), &E1_evals_for_other_constraints);

        // Round 7 onwards
        // We just follow the linear-time algorithm for `az_bz_cz_poly`
        for round in (LOG_ONE_HALF_NUM_CONSTRAINTS_PADDED+1)..num_rounds {
            az_bz_cz_poly.linear_time_sumcheck_round(eq_poly, transcript, &mut r, &mut polys, &mut claim);
            todo!()
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
