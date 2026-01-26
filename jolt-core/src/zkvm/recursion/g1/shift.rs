//! Shift sumchecks for scalar-multiplication traces (G1/G2).
//!
//! These sumchecks prevent the prover from choosing `*_next` polynomials arbitrarily.
//! Concretely, for each scalar-mul instance and each relevant coordinate polynomial, we enforce
//! that `A_next` is a **one-step shift** of `A` along the 8-bit step index.
//!
//! ## The precise per-step relation (what we *want*)
//! Let \(n = 8\) be the number of step variables, and interpret \(s \in \{0,1\}^n\) as an
//! integer \(i \in \{0,\dots,2^n-1\}\) in **LSB-first** order (consistent with
//! `BindingOrder::LowToHigh` in sumcheck).
//!
//! For each scalar-mul instance and each coordinate polynomial \(A\) (and its witness-shift
//! \(A_{\text{next}}\)), we want:
//!
//! \[
//!   \forall i \in \{0,\dots,254\}:\quad A_{\text{next}}(i) = A(i+1)
//! \]
//!
//! - For G1 we do this for \(x_A\) and \(y_A\).
//! - For G2 we do this component-wise for \(x_A.c0, x_A.c1, y_A.c0, y_A.c1\).
//!
//! Note: the boundary values \(A(0)\) (the initial accumulator) and \(A_{\text{next}}(255)\)
//! (the final result after 256 steps) are **not** part of the “shift wiring” relation above.
//! Those are boundary conditions and must be constrained elsewhere if the protocol needs them.
//!
//! The constraints are enforced *as MLEs* via a random linear check using Eq/EqPlusOne
//! over the (low) 8 step variables. We intentionally do **not** constrain the last `A_next[255]`
//! (the final result) and we do **not** constrain `A[0]` (the initial accumulator).
//!
//! ## The randomized identity we prove (what the sumcheck checks)
//! Sample a random \(r^* \in \mathbb{F}^n\) (via Fiat–Shamir). Define:
//! - \(Eq(r^*, s)\) as the standard multilinear equality indicator on \(\{0,1\}^n\)
//! - \(Eq(r^*, s-1)\) as the same indicator applied to the predecessor index (treating “\(-1\)”
//!   as out-of-range, i.e. contributing 0)
//! - \(notLast(s) = 1 - Eq(s, 2^n-1)\), i.e. a mask that is 0 only at the last step.
//!
//! Then we prove the following identity equals 0:
//!
//! \[
//!   \sum_{s \in \{0,1\}^n} Eq(r^*, s)\cdot notLast(s) \cdot A_{\text{next}}(s)
//!   \;=\;
//!   \sum_{s \in \{0,1\}^n} Eq(r^*, s-1)\cdot A(s)
//! \]
//!
//! Expanding both sides on the Boolean hypercube, this is exactly the per-step relation
//! \(A_{\text{next}}(i) = A(i+1)\) for all \(i \in \{0,\dots,254\}\), while:
//! - excluding the last entry \(A_{\text{next}}(255)\) via \(notLast\), and
//! - excluding \(A(0)\) because the predecessor of 0 is out-of-range.

use crate::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::witness::VirtualPolynomial,
};
use rayon::prelude::*;

use crate::zkvm::recursion::gt::shift::{
    eq_lsb_evals, eq_lsb_mle, eq_plus_one_lsb_evals, eq_plus_one_lsb_mle,
};

/// Number of step variables (256 steps).
///
/// NOTE: This sumcheck is defined over the native 8-var scalar-mul trace polynomials.
/// In the batched sumcheck, it is **suffix-aligned** via `round_offset`.
const STEP_VARS: usize = 8;

#[derive(Clone, Debug)]
struct ShiftPair<F: JoltField> {
    a_poly: MultilinearPolynomial<F>,
    a_next_poly: MultilinearPolynomial<F>,
    a_id: VirtualPolynomial,
    a_next_id: VirtualPolynomial,
}

/// not_last(s) = 1 for all s != 2^n-1, 0 at the last index.
fn not_last_poly<F: JoltField>() -> MultilinearPolynomial<F> {
    let mut evals_8 = vec![F::one(); 1 << STEP_VARS];
    evals_8[(1 << STEP_VARS) - 1] = F::zero();
    MultilinearPolynomial::from(evals_8)
}

fn not_last_lsb_mle<F: JoltField>(y_step: &[F::Challenge]) -> F {
    // not_last(y) = 1 - ∏_i y_i  (since last index is all-ones)
    debug_assert_eq!(y_step.len(), STEP_VARS);
    let mut prod = F::one();
    let one = F::one();
    for &y_i in y_step {
        let y_i_f: F = y_i.into();
        prod *= y_i_f;
    }
    one - prod
}

#[derive(Clone)]
pub struct ShiftScalarMulParams {
    pub num_vars: usize,
    pub num_pairs: usize,
    pub sumcheck_id: SumcheckId,
}

impl ShiftScalarMulParams {
    pub fn new(sumcheck_id: SumcheckId, num_pairs: usize) -> Self {
        Self {
            num_vars: STEP_VARS,
            num_pairs,
            sumcheck_id,
        }
    }
}

pub struct ShiftScalarMulProver<F: JoltField, T: Transcript> {
    pub params: ShiftScalarMulParams,
    eq_step_poly: MultilinearPolynomial<F>,
    /// Note: `eq_plus_one_lsb_*` in `shift_rho` actually corresponds to Eq(r, s-1),
    /// i.e. a "minus one" selector in the step index (LSB-first).
    eq_minus_one_step_poly: MultilinearPolynomial<F>,
    not_last_poly: MultilinearPolynomial<F>,
    pairs: Vec<ShiftPair<F>>,
    gamma: F,
    round: usize,
    pub _marker: core::marker::PhantomData<T>,
}

impl<F: JoltField, T: Transcript> ShiftScalarMulProver<F, T> {
    pub fn new(
        params: ShiftScalarMulParams,
        pairs: Vec<(VirtualPolynomial, Vec<F>, VirtualPolynomial, Vec<F>)>,
        transcript: &mut T,
    ) -> Self {
        // Sample reference point r* (8 challenges) and batching gamma.
        let step_ref: Vec<F::Challenge> = (0..STEP_VARS)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        // Build weight polynomials (native 8-var).
        let eq_step_8 = eq_lsb_evals::<F>(&step_ref);
        let eq_minus_one_8 = eq_plus_one_lsb_evals::<F>(&step_ref);
        let eq_step_poly = MultilinearPolynomial::from(eq_step_8);
        let eq_minus_one_step_poly = MultilinearPolynomial::from(eq_minus_one_8);
        let not_last_poly = not_last_poly::<F>();

        let pairs = pairs
            .into_iter()
            .map(|(a_id, a, a_next_id, a_next)| ShiftPair {
                a_poly: MultilinearPolynomial::from(a),
                a_next_poly: MultilinearPolynomial::from(a_next),
                a_id,
                a_next_id,
            })
            .collect::<Vec<_>>();

        debug_assert_eq!(params.num_pairs, pairs.len());

        Self {
            params,
            eq_step_poly,
            eq_minus_one_step_poly,
            not_last_poly,
            pairs,
            gamma,
            round: 0,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ShiftScalarMulProver<F, T> {
    fn degree(&self) -> usize {
        // max product: Eq * not_last * A_next  (3 multilinear factors)
        3
    }

    fn num_rounds(&self) -> usize {
        self.params.num_vars
    }

    /// See `BatchedSumcheck`: shorter instances are suffix-aligned.
    fn round_offset(&self, max_num_rounds: usize) -> usize {
        max_num_rounds - self.params.num_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        // We prove the (batched) shift-consistency sum equals 0.
        F::zero()
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 3;

        if self.pairs.is_empty() {
            return UniPoly::from_evals_and_hint(previous_claim, &[F::zero(); DEGREE]);
        }

        let half = self.pairs[0].a_poly.len() / 2;
        let gamma = self.gamma;

        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                let eq_evals = self
                    .eq_step_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let eqm1_evals = self
                    .eq_minus_one_step_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let not_last_evals = self
                    .not_last_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                let mut term_evals = [F::zero(); DEGREE];
                let mut gamma_power = F::one();

                for pair in &self.pairs {
                    let a_evals = pair
                        .a_poly
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    let an_evals = pair
                        .a_next_poly
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    for t in 0..DEGREE {
                        term_evals[t] += gamma_power
                            * (eq_evals[t] * not_last_evals[t] * an_evals[t]
                                - eqm1_evals[t] * a_evals[t]);
                    }
                    gamma_power *= gamma;
                }

                term_evals
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut acc, evals| {
                    for (a, e) in acc.iter_mut().zip(evals.iter()) {
                        *a += *e;
                    }
                    acc
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        for pair in &mut self.pairs {
            pair.a_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            pair.a_next_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        self.eq_step_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_minus_one_step_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.not_last_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.round = round + 1;
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<{ BIG_ENDIAN }, F>::new(sumcheck_challenges.to_vec());
        for pair in &self.pairs {
            let a_eval = pair.a_poly.get_bound_coeff(0);
            let a_next_eval = pair.a_next_poly.get_bound_coeff(0);
            accumulator.append_virtual(
                transcript,
                pair.a_id,
                self.params.sumcheck_id,
                opening_point.clone(),
                a_eval,
            );
            accumulator.append_virtual(
                transcript,
                pair.a_next_id,
                self.params.sumcheck_id,
                opening_point.clone(),
                a_next_eval,
            );
        }
    }
}

pub struct ShiftScalarMulVerifier<F: JoltField> {
    pub params: ShiftScalarMulParams,
    step_ref: Vec<F::Challenge>,
    gamma: F,
    // Same ordering as prover (for gamma batching + opening ids)
    pairs: Vec<(VirtualPolynomial, VirtualPolynomial)>,
}

impl<F: JoltField> ShiftScalarMulVerifier<F> {
    pub fn new<T: Transcript>(
        params: ShiftScalarMulParams,
        pairs: Vec<(VirtualPolynomial, VirtualPolynomial)>,
        transcript: &mut T,
    ) -> Self {
        let step_ref: Vec<F::Challenge> = (0..STEP_VARS)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        debug_assert_eq!(params.num_pairs, pairs.len());

        Self {
            params,
            step_ref,
            gamma,
            pairs,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ShiftScalarMulVerifier<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.params.num_vars
    }

    /// See `ShiftScalarMulProver::round_offset`.
    fn round_offset(&self, max_num_rounds: usize) -> usize {
        max_num_rounds - self.params.num_vars
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        debug_assert_eq!(sumcheck_challenges.len(), STEP_VARS);
        let y_step = sumcheck_challenges;

        let eq = eq_lsb_mle::<F>(&self.step_ref, y_step);
        let eqm1 = eq_plus_one_lsb_mle::<F>(&self.step_ref, y_step);
        let not_last = not_last_lsb_mle::<F>(y_step);

        let mut sum = F::zero();
        let mut gamma_power = F::one();

        for (a_id, a_next_id) in &self.pairs {
            let (_, a_eval) =
                accumulator.get_virtual_polynomial_opening(*a_id, self.params.sumcheck_id);
            let (_, a_next_eval) =
                accumulator.get_virtual_polynomial_opening(*a_next_id, self.params.sumcheck_id);
            sum += gamma_power * (eq * not_last * a_next_eval - eqm1 * a_eval);
            gamma_power *= self.gamma;
        }

        sum
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<{ BIG_ENDIAN }, F>::new(sumcheck_challenges.to_vec());
        for (a_id, a_next_id) in &self.pairs {
            accumulator.append_virtual(
                transcript,
                *a_id,
                self.params.sumcheck_id,
                opening_point.clone(),
            );
            accumulator.append_virtual(
                transcript,
                *a_next_id,
                self.params.sumcheck_id,
                opening_point.clone(),
            );
        }
    }
}

// Convenience aliases for callers (different SumcheckId values via params).
pub type ShiftG1ScalarMulProver<F, T> = ShiftScalarMulProver<F, T>;
pub type ShiftG1ScalarMulVerifier<F> = ShiftScalarMulVerifier<F>;
pub type ShiftG2ScalarMulProver<F, T> = ShiftScalarMulProver<F, T>;
pub type ShiftG2ScalarMulVerifier<F> = ShiftScalarMulVerifier<F>;

pub fn g1_shift_params(num_pairs: usize) -> ShiftScalarMulParams {
    ShiftScalarMulParams::new(SumcheckId::ShiftG1ScalarMul, num_pairs)
}

pub fn g2_shift_params(num_pairs: usize) -> ShiftScalarMulParams {
    ShiftScalarMulParams::new(SumcheckId::ShiftG2ScalarMul, num_pairs)
}
