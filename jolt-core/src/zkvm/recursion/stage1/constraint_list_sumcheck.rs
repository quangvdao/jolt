//! Generic Stage-1 "sumcheck over a list of constraints" helper.
//!
//! Many Stage-1 recursion sumchecks share the same structure:
//! - sample an `eq_point` for `eq(eq_point, x)`
//! - batch multiple constraint instances with `instance_batch_coeff`
//! - (optionally) batch multiple constraint terms within an instance with `term_batch_coeff`
//! - bind all polynomials round-by-round
//! - cache virtual polynomial openings at the final `eval_point`
//!
//! This module provides reusable prover/verifier cores for that pattern. Op-specific logic
//! is provided via small "spec" traits.
//!
//! IMPORTANT: Transcript ordering is consensus-critical. In particular, the order of
//! `accumulator.append_virtual(transcript, ...)` calls in `cache_openings` must match the
//! legacy per-op implementations.

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
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

// ============================================================================
// OpeningSpec: Declarative specification for virtual polynomial openings
// ============================================================================

/// Specifies one virtual opening to emit per instance.
///
/// - `kind` indexes into the prover's `polys_by_kind()[kind][instance]`.
/// - `make_poly` is a function that constructs the `VirtualPolynomial` from an instance index.
#[derive(Clone, Copy)]
pub struct OpeningSpec {
    pub kind: usize,
    pub make_poly: fn(usize) -> VirtualPolynomial,
}

impl std::fmt::Debug for OpeningSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpeningSpec")
            .field("kind", &self.kind)
            .field("make_poly", &"<fn>")
            .finish()
    }
}

impl OpeningSpec {
    pub const fn new(kind: usize, make_poly: fn(usize) -> VirtualPolynomial) -> Self {
        Self { kind, make_poly }
    }

    /// Construct the VirtualPolynomial for the given instance index.
    pub fn with_index(&self, instance: usize) -> VirtualPolynomial {
        (self.make_poly)(instance)
    }
}

// ============================================================================
// Spec Traits: Base trait + Prover/Verifier extensions
// ============================================================================

/// Common metadata shared by prover and verifier specs.
pub trait ConstraintListSpec: Send + Sync {
    fn sumcheck_id(&self) -> SumcheckId;
    fn num_rounds(&self) -> usize;
    fn num_instances(&self) -> usize;

    /// Whether this op batches multiple constraint terms within each instance.
    /// If true, `term_batch_coeff` will be sampled and passed to constraint evaluation.
    fn uses_term_batching(&self) -> bool {
        false
    }

    /// Virtual openings to emit (ordered). Order is consensus-critical.
    fn opening_specs(&self) -> &'static [OpeningSpec];
}

/// Prover-specific: owns polynomial data, evaluates constraint at univariate point t.
pub trait ConstraintListProverSpec<F: JoltField, const DEGREE: usize>: ConstraintListSpec {
    /// Polynomials grouped by kind: `polys_by_kind[kind][instance]`.
    fn polys_by_kind(&self) -> &[Vec<MultilinearPolynomial<F>>];
    fn polys_by_kind_mut(&mut self) -> &mut [Vec<MultilinearPolynomial<F>>];

    /// Shared polynomials (e.g. a global `g(x)`), bound each round.
    fn shared_polys(&self) -> &[MultilinearPolynomial<F>];
    fn shared_polys_mut(&mut self) -> &mut [MultilinearPolynomial<F>];

    /// Initial sumcheck claim (default: 0).
    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        F::zero()
    }

    /// Evaluate the instance constraint at a univariate evaluation index.
    ///
    /// During each sumcheck round, we evaluate the constraint polynomial at multiple points
    /// to construct the univariate round polynomial. `eval_index` indexes into the evaluations:
    /// - `poly_evals[kind][eval_index]` and `shared_evals[j][eval_index]`
    /// - Points are `{0, 2, 3, ..., DEGREE}` (skipping 1, as produced by `sumcheck_evals_array`)
    ///
    /// `term_batch_coeff` is the random coefficient for batching multiple constraint terms
    /// within this instance (only present if `uses_term_batching()` returns true).
    fn eval_constraint(
        &self,
        instance: usize,
        eval_index: usize,
        poly_evals: &[[F; DEGREE]],
        shared_evals: &[[F; DEGREE]],
        term_batch_coeff: Option<F>,
    ) -> F;
}

/// Verifier-specific: computes shared scalars once, evaluates constraint at the final point.
pub trait ConstraintListVerifierSpec<F: JoltField, const DEGREE: usize>:
    ConstraintListSpec
{
    /// Initial sumcheck claim (default: 0).
    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        F::zero()
    }

    /// Compute shared scalar values (e.g., g(eval_point)) ONCE before the instance loop.
    /// These are passed to `eval_constraint_at_point` for each instance.
    fn compute_shared_scalars(&self, _eval_point: &[F]) -> Vec<F> {
        vec![]
    }

    /// Evaluate the instance constraint at the final evaluation point (in little-endian order),
    /// using the opened claims (in `opening_specs()` order) and pre-computed shared scalars.
    ///
    /// `term_batch_coeff` is the random coefficient for batching multiple constraint terms
    /// within this instance (only present if `uses_term_batching()` returns true).
    fn eval_constraint_at_point(
        &self,
        instance: usize,
        opened_claims: &[F],
        shared_scalars: &[F],
        term_batch_coeff: Option<F>,
    ) -> F;
}

// ============================================================================
// ConstraintListProver: Generic prover core
// ============================================================================

/// Generic prover core for a "list of constraints" sumcheck.
pub struct ConstraintListProver<F: JoltField, Spec, const DEGREE: usize> {
    pub spec: Spec,
    pub eq_poly: MultilinearPolynomial<F>,
    /// Random point sampled for the eq polynomial: eq(eq_point, x)
    pub eq_point: Vec<F::Challenge>,
    /// Random coefficient for batching constraint instances: Σ_i (instance_batch_coeff)^i * C_i
    pub instance_batch_coeff: F,
    /// Random coefficient for batching constraint terms within an instance (if used)
    pub term_batch_coeff: Option<F>,
}

impl<F: JoltField, Spec, const DEGREE: usize> ConstraintListProver<F, Spec, DEGREE>
where
    Spec: ConstraintListProverSpec<F, DEGREE>,
{
    /// Create a prover from a spec. Op-specific files typically wrap this in a `new` method.
    pub fn from_spec<T: Transcript>(spec: Spec, transcript: &mut T) -> Self {
        let num_rounds = spec.num_rounds();
        let uses_term_batching = spec.uses_term_batching();

        let eq_point: Vec<F::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();
        let instance_batch_coeff: F = transcript.challenge_scalar_optimized::<F>().into();
        let term_batch_coeff: Option<F> =
            uses_term_batching.then(|| transcript.challenge_scalar_optimized::<F>().into());

        let eq_poly = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&eq_point));

        Self {
            spec,
            eq_poly,
            eq_point,
            instance_batch_coeff,
            term_batch_coeff,
        }
    }
}

impl<F: JoltField, T: Transcript, Spec, const DEGREE: usize> SumcheckInstanceProver<F, T>
    for ConstraintListProver<F, Spec, DEGREE>
where
    Spec: ConstraintListProverSpec<F, DEGREE>,
{
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.spec.num_rounds()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.spec.input_claim(accumulator)
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let num_x_remaining = self.eq_poly.get_num_vars();
        let x_half = 1 << (num_x_remaining - 1);

        let polys_by_kind = self.spec.polys_by_kind();
        let shared_polys = self.spec.shared_polys();
        let num_instances = self.spec.num_instances();
        let num_kinds = polys_by_kind.len();
        let num_shared = shared_polys.len();

        let instance_batch_coeff = self.instance_batch_coeff;
        let term_batch_coeff = self.term_batch_coeff;

        let total_evals = (0..x_half)
            .into_par_iter()
            .map(|x_idx| {
                let eq_evals = self
                    .eq_poly
                    .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                let mut shared_evals = vec![[F::zero(); DEGREE]; num_shared];
                for (j, poly) in shared_polys.iter().enumerate() {
                    shared_evals[j] =
                        poly.sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                }

                let mut x_evals = [F::zero(); DEGREE];
                let mut batch_power = instance_batch_coeff;

                for i in 0..num_instances {
                    let mut poly_evals = vec![[F::zero(); DEGREE]; num_kinds];
                    for (k, kind_polys) in polys_by_kind.iter().enumerate() {
                        poly_evals[k] = kind_polys[i]
                            .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    }

                    for eval_index in 0..DEGREE {
                        let constraint_eval = self.spec.eval_constraint(
                            i,
                            eval_index,
                            &poly_evals,
                            &shared_evals,
                            term_batch_coeff,
                        );
                        x_evals[eval_index] += eq_evals[eval_index] * batch_power * constraint_eval;
                    }

                    batch_power *= instance_batch_coeff;
                }

                x_evals
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

        UniPoly::from_evals_and_hint(previous_claim, &total_evals)
    }

    fn ingest_challenge(&mut self, challenge: F::Challenge, _round: usize) {
        self.eq_poly
            .bind_parallel(challenge, BindingOrder::LowToHigh);

        for poly in self.spec.shared_polys_mut().iter_mut() {
            poly.bind_parallel(challenge, BindingOrder::LowToHigh);
        }
        for kind in self.spec.polys_by_kind_mut().iter_mut() {
            for poly in kind.iter_mut() {
                poly.bind_parallel(challenge, BindingOrder::LowToHigh);
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(sumcheck_challenges.to_vec());
        let sumcheck_id = self.spec.sumcheck_id();
        let opening_specs = self.spec.opening_specs();
        let polys_by_kind = self.spec.polys_by_kind();

        // Order is instance-major, then per-instance opening order.
        for i in 0..self.spec.num_instances() {
            for spec in opening_specs {
                let claim = polys_by_kind[spec.kind][i].get_bound_coeff(0);
                accumulator.append_virtual(
                    transcript,
                    spec.with_index(i),
                    sumcheck_id,
                    opening_point.clone(),
                    claim,
                );
            }
        }
    }
}

// ============================================================================
// ConstraintListVerifier: Generic verifier core
// ============================================================================

/// Generic verifier core for a "list of constraints" sumcheck.
pub struct ConstraintListVerifier<F: JoltField, Spec, const DEGREE: usize> {
    pub spec: Spec,
    /// Random point sampled for the eq polynomial: eq(eq_point, x)
    pub eq_point: Vec<F::Challenge>,
    /// Random coefficient for batching constraint instances: Σ_i (instance_batch_coeff)^i * C_i
    pub instance_batch_coeff: F,
    /// Random coefficient for batching constraint terms within an instance (if used)
    pub term_batch_coeff: Option<F>,
}

impl<F: JoltField, Spec, const DEGREE: usize> ConstraintListVerifier<F, Spec, DEGREE>
where
    Spec: ConstraintListVerifierSpec<F, DEGREE>,
{
    /// Create a verifier from a spec. Op-specific files typically wrap this in a `new` method.
    pub fn from_spec<T: Transcript>(spec: Spec, transcript: &mut T) -> Self {
        let num_rounds = spec.num_rounds();
        let uses_term_batching = spec.uses_term_batching();

        let eq_point: Vec<F::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();
        let instance_batch_coeff: F = transcript.challenge_scalar_optimized::<F>().into();
        let term_batch_coeff: Option<F> =
            uses_term_batching.then(|| transcript.challenge_scalar_optimized::<F>().into());

        Self {
            spec,
            eq_point,
            instance_batch_coeff,
            term_batch_coeff,
        }
    }
}

impl<F: JoltField, T: Transcript, Spec, const DEGREE: usize> SumcheckInstanceVerifier<F, T>
    for ConstraintListVerifier<F, Spec, DEGREE>
where
    Spec: ConstraintListVerifierSpec<F, DEGREE>,
{
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.spec.num_rounds()
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.spec.input_claim(accumulator)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let eq_point_f: Vec<F> = self.eq_point.iter().map(|c| (*c).into()).collect();
        // Convert sumcheck challenges to evaluation point (reversed for little-endian)
        let eval_point: Vec<F> = sumcheck_challenges
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();
        let eq_eval = EqPolynomial::mle(&eq_point_f, &eval_point);

        let sumcheck_id = self.spec.sumcheck_id();
        let opening_specs = self.spec.opening_specs();

        // Compute shared scalars ONCE (e.g., g(eval_point) for GT ops)
        let shared_scalars = self.spec.compute_shared_scalars(&eval_point);

        let mut total = F::zero();
        let mut batch_power = self.instance_batch_coeff;

        for i in 0..self.spec.num_instances() {
            let mut claims = Vec::with_capacity(opening_specs.len());
            for spec in opening_specs {
                let (_, claim) =
                    accumulator.get_virtual_polynomial_opening(spec.with_index(i), sumcheck_id);
                claims.push(claim);
            }

            let constraint_value = self.spec.eval_constraint_at_point(
                i,
                &claims,
                &shared_scalars,
                self.term_batch_coeff,
            );
            total += batch_power * constraint_value;
            batch_power *= self.instance_batch_coeff;
        }

        eq_eval * total
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(sumcheck_challenges.to_vec());
        let sumcheck_id = self.spec.sumcheck_id();
        let opening_specs = self.spec.opening_specs();

        // Order is instance-major, then per-instance opening order.
        for i in 0..self.spec.num_instances() {
            for spec in opening_specs {
                accumulator.append_virtual(
                    transcript,
                    spec.with_index(i),
                    sumcheck_id,
                    opening_point.clone(),
                );
            }
        }
    }
}
