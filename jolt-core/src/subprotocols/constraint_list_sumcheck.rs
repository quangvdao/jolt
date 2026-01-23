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
use allocative::Allocative;
use rayon::prelude::*;

// ============================================================================
// OpeningSpec: Data-driven specification for virtual polynomial openings
// ============================================================================

/// Specifies one virtual opening to emit per instance.
///
/// - `kind` indexes into the prover's `polys_by_kind()[kind][instance]`.
/// - `term_index` identifies which term within the constraint (used by `build_virtual_poly`).
///
/// This is pure data - the actual VirtualPolynomial construction is delegated to the spec trait.
#[derive(Clone, Copy, Debug, Allocative)]
pub struct OpeningSpec {
    pub kind: usize,
    pub term_index: usize,
}

impl OpeningSpec {
    pub const fn new(kind: usize, term_index: usize) -> Self {
        Self { kind, term_index }
    }
}

/// Generate sequential opening specs where kind == term_index for all N terms.
/// This is the common case when polynomial kinds are 1:1 with terms.
pub const fn sequential_opening_specs<const N: usize>() -> [OpeningSpec; N] {
    let mut specs = [OpeningSpec {
        kind: 0,
        term_index: 0,
    }; N];
    let mut i = 0;
    while i < N {
        specs[i] = OpeningSpec {
            kind: i,
            term_index: i,
        };
        i += 1;
    }
    specs
}

// ============================================================================
// Spec Traits: Base trait + Prover/Verifier extensions
// ============================================================================

/// Common metadata shared by prover and verifier specs.
pub trait ConstraintListSpec: Send + Sync + Allocative {
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

    /// Build the VirtualPolynomial for the given term_index and instance.
    /// This is called by the wrapper to construct polynomial identifiers for openings.
    fn build_virtual_poly(&self, term_index: usize, instance: usize) -> VirtualPolynomial;
}

/// Prover-specific: owns polynomial data, evaluates constraint at univariate point t.
pub trait ConstraintListProverSpec<F: JoltField, const DEGREE: usize>: ConstraintListSpec {
    /// Polynomials grouped by kind: `polys_by_kind[kind][instance]`.
    /// These are committed polynomials that will be opened at the end of sumcheck.
    fn polys_by_kind(&self) -> &[Vec<MultilinearPolynomial<F>>];
    fn polys_by_kind_mut(&mut self) -> &mut [Vec<MultilinearPolynomial<F>>];

    /// Public polynomials: participate in sumcheck binding but are NOT opened.
    /// The verifier computes their evaluations directly from public inputs.
    /// Layout: `public_polys[kind][instance]`.
    fn public_polys(&self) -> &[Vec<MultilinearPolynomial<F>>] {
        &[]
    }
    fn public_polys_mut(&mut self) -> &mut [Vec<MultilinearPolynomial<F>>] {
        &mut []
    }

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
    /// - `poly_evals[kind][eval_index]`: committed polynomial evaluations
    /// - `public_evals[kind][eval_index]`: public polynomial evaluations (not opened)
    /// - `shared_evals[j][eval_index]`: shared polynomial evaluations
    /// - Points are `{0, 2, 3, ..., DEGREE}` (skipping 1, as produced by `sumcheck_evals_array`)
    ///
    /// `term_batch_coeff` is the random coefficient for batching multiple constraint terms
    /// within this instance (only present if `uses_term_batching()` returns true).
    fn eval_constraint(
        &self,
        instance: usize,
        eval_index: usize,
        poly_evals: &[[F; DEGREE]],
        public_evals: &[[F; DEGREE]],
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
    /// `eval_point` is provided so the verifier can compute public polynomial evaluations
    /// directly from public inputs (e.g., scalar bit MLE evaluation).
    ///
    /// `term_batch_coeff` is the random coefficient for batching multiple constraint terms
    /// within this instance (only present if `uses_term_batching()` returns true).
    fn eval_constraint_at_point(
        &self,
        instance: usize,
        opened_claims: &[F],
        shared_scalars: &[F],
        eval_point: &[F],
        term_batch_coeff: Option<F>,
    ) -> F;
}

// ============================================================================
// ConstraintListProver: Generic prover core
// ============================================================================

/// Generic prover core for a "list of constraints" sumcheck.
#[derive(Allocative)]
pub struct ConstraintListProver<F: JoltField, Spec, const DEGREE: usize> {
    pub spec: Spec,
    pub eq_poly: MultilinearPolynomial<F>,
    /// Random point sampled for the eq polynomial: eq(eq_point, x)
    pub eq_point: Vec<F::Challenge>,
    /// Random coefficient for batching constraint instances: Σ_i (instance_batch_coeff)^i * C_i
    pub instance_batch_coeff: F,
    /// Random coefficient for batching constraint terms within an instance (if used)
    pub term_batch_coeff: Option<F>,
    /// Maps local instance index to global constraint index (for VirtualPolynomial identifiers)
    pub constraint_indices: Vec<usize>,
}

impl<F: JoltField, Spec, const DEGREE: usize> ConstraintListProver<F, Spec, DEGREE>
where
    Spec: ConstraintListProverSpec<F, DEGREE>,
{
    /// Create a prover from a spec and constraint indices.
    /// `constraint_indices` maps local instance i to global constraint index.
    pub fn from_spec<T: Transcript>(
        spec: Spec,
        constraint_indices: Vec<usize>,
        transcript: &mut T,
    ) -> Self {
        debug_assert_eq!(
            spec.num_instances(),
            constraint_indices.len(),
            "constraint_indices length must match num_instances"
        );

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
            constraint_indices,
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
        let public_polys = self.spec.public_polys();
        let shared_polys = self.spec.shared_polys();
        let num_instances = self.spec.num_instances();
        let num_kinds = polys_by_kind.len();
        let num_public = public_polys.len();
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
                    // Committed polynomial evaluations (will be opened)
                    let mut poly_evals = vec![[F::zero(); DEGREE]; num_kinds];
                    for (k, kind_polys) in polys_by_kind.iter().enumerate() {
                        poly_evals[k] = kind_polys[i]
                            .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    }

                    // Public polynomial evaluations (not opened, verifier computes from public inputs)
                    let mut public_evals = vec![[F::zero(); DEGREE]; num_public];
                    for (k, kind_polys) in public_polys.iter().enumerate() {
                        public_evals[k] = kind_polys[i]
                            .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    }

                    for eval_index in 0..DEGREE {
                        let constraint_eval = self.spec.eval_constraint(
                            i,
                            eval_index,
                            &poly_evals,
                            &public_evals,
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
        // Bind public polynomials (participate in sumcheck but not opened)
        for kind in self.spec.public_polys_mut().iter_mut() {
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
            let global_idx = self.constraint_indices[i];
            for spec in opening_specs {
                let claim = polys_by_kind[spec.kind][i].get_bound_coeff(0);
                let poly_id = self.spec.build_virtual_poly(spec.term_index, global_idx);
                accumulator.append_virtual(
                    transcript,
                    poly_id,
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
#[derive(Allocative)]
pub struct ConstraintListVerifier<F: JoltField, Spec, const DEGREE: usize> {
    pub spec: Spec,
    /// Random point sampled for the eq polynomial: eq(eq_point, x)
    pub eq_point: Vec<F::Challenge>,
    /// Random coefficient for batching constraint instances: Σ_i (instance_batch_coeff)^i * C_i
    pub instance_batch_coeff: F,
    /// Random coefficient for batching constraint terms within an instance (if used)
    pub term_batch_coeff: Option<F>,
    /// Maps local instance index to global constraint index (for VirtualPolynomial identifiers)
    pub constraint_indices: Vec<usize>,
}

impl<F: JoltField, Spec, const DEGREE: usize> ConstraintListVerifier<F, Spec, DEGREE>
where
    Spec: ConstraintListVerifierSpec<F, DEGREE>,
{
    /// Create a verifier from a spec and constraint indices.
    /// `constraint_indices` maps local instance i to global constraint index.
    pub fn from_spec<T: Transcript>(
        spec: Spec,
        constraint_indices: Vec<usize>,
        transcript: &mut T,
    ) -> Self {
        debug_assert_eq!(
            spec.num_instances(),
            constraint_indices.len(),
            "constraint_indices length must match num_instances"
        );

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
            constraint_indices,
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
            let global_idx = self.constraint_indices[i];
            let mut claims = Vec::with_capacity(opening_specs.len());
            for spec in opening_specs {
                let poly_id = self.spec.build_virtual_poly(spec.term_index, global_idx);
                let (_, claim) = accumulator.get_virtual_polynomial_opening(poly_id, sumcheck_id);
                claims.push(claim);
            }

            let constraint_value = self.spec.eval_constraint_at_point(
                i,
                &claims,
                &shared_scalars,
                &eval_point,
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
            let global_idx = self.constraint_indices[i];
            for spec in opening_specs {
                let poly_id = self.spec.build_virtual_poly(spec.term_index, global_idx);
                accumulator.append_virtual(transcript, poly_id, sumcheck_id, opening_point.clone());
            }
        }
    }
}

// ============================================================================
// Macro: Full constraint type definition
// ============================================================================

/// Macro to define a complete constraint type with all boilerplate auto-generated.
///
/// 1. `{Name}ConstraintPolynomials<F>` - generic over JoltField, holds `Vec<F>` for each field
/// 2. `{Name}Values<F>` - single-point evaluations with `from_poly_evals`/`from_claims`
/// 3. `{Name}Params` - sumcheck parameters
/// 4. `{Name}ProverSpec<F>` - prover specification with all trait impls
/// 5. `{Name}VerifierSpec<F>` - verifier specification with all trait impls
/// 6. `{Name}Prover<F>` / `{Name}Verifier<F>` - type aliases
/// 7. Opening specs (auto-generated sequential)
/// 8. `build_virtual_poly` implementations
///
/// The ONLY custom code needed after invoking this macro is:
/// ```ignore
/// impl<F: JoltField> {Name}Values<F> {
///     pub fn eval_constraint(&self, delta: F) -> F {
///         // THE ACTUAL CONSTRAINT LOGIC
///     }
/// }
/// ```
///
/// # Example
/// ```ignore
/// define_constraint!(
///     name: G1Add,
///     sumcheck_id: SumcheckId::G1Add,
///     num_vars: 11,
///     degree: 6,
///     uses_term_batching: true,
///     term_enum: G1AddTerm,
///     recursion_poly_variant: G1Add,
///     fields: [x_p, y_p, ind_p, x_q, y_q, ind_q, x_r, y_r, ind_r, lambda, inv_delta_x, is_double, is_inverse]
/// );
/// ```
#[macro_export]
macro_rules! define_constraint {
    (
        name: $name:ident,
        sumcheck_id: $sumcheck_id:expr,
        num_vars: $num_vars:expr,
        degree: $degree:expr,
        uses_term_batching: $uses_term_batching:expr,
        term_enum: $term_enum:ty,
        recursion_poly_variant: $recursion_variant:ident,
        fields: [$($field:ident),* $(,)?]
        $(,)?
    ) => {
        $crate::paste::paste! {
            // ================================================================
            // Constraint Polynomials
            // ================================================================

            #[derive(Clone, Debug, $crate::allocative::Allocative)]
            pub struct [<$name ConstraintPolynomials>]<F> {
                $(pub $field: Vec<F>,)*
                pub constraint_index: usize,
            }

            impl<F: $crate::field::JoltField + $crate::allocative::Allocative> [<$name ConstraintPolynomials>]<F> {
                #[allow(unused_assignments)]
                pub fn unpack_all(
                    polys: Vec<Self>,
                ) -> (
                    Vec<Vec<$crate::poly::multilinear_polynomial::MultilinearPolynomial<F>>>,
                    Vec<usize>,
                ) {
                    let num_instances = polys.len();
                    let num_fields = [<$name Values>]::<F>::COUNT;

                    let mut polys_by_kind: Vec<Vec<$crate::poly::multilinear_polynomial::MultilinearPolynomial<F>>> =
                        (0..num_fields).map(|_| Vec::with_capacity(num_instances)).collect();
                    let mut constraint_indices = Vec::with_capacity(num_instances);

                    for poly in polys {
                        constraint_indices.push(poly.constraint_index);
                        let mut kind_idx = 0usize;
                        $(
                            polys_by_kind[kind_idx].push(
                                $crate::poly::multilinear_polynomial::MultilinearPolynomial::LargeScalars(
                                    $crate::poly::dense_mlpoly::DensePolynomial::new(poly.$field)
                                )
                            );
                            kind_idx += 1;
                        )*
                    }

                    (polys_by_kind, constraint_indices)
                }
            }

            pub type [<$name Witness>]<F> = [<$name ConstraintPolynomials>]<F>;

            // ================================================================
            // Values (single-point evaluations)
            // ================================================================

            #[derive(Clone, Copy, Debug, $crate::allocative::Allocative)]
            pub struct [<$name Values>]<F> {
                $(pub $field: F,)*
            }

            impl<F: Copy> [<$name Values>]<F> {
                pub const COUNT: usize = {
                    #[allow(unused_mut)]
                    let mut n = 0usize;
                    $(let _ = stringify!($field); n += 1;)*
                    n
                };

                #[inline]
                #[allow(unused_assignments)]
                pub fn from_poly_evals<const DEGREE: usize>(poly_evals: &[[F; DEGREE]], eval_index: usize) -> Self {
                    let mut idx = 0usize;
                    Self { $($field: { let v = poly_evals[idx][eval_index]; idx += 1; v },)* }
                }

                #[inline]
                #[allow(unused_assignments)]
                pub fn from_claims(claims: &[F]) -> Self {
                    let mut idx = 0usize;
                    Self { $($field: { let v = claims[idx]; idx += 1; v },)* }
                }
            }

            // ================================================================
            // Parameters
            // ================================================================

            #[derive(Clone, $crate::allocative::Allocative)]
            pub struct [<$name Params>] {
                pub num_constraint_vars: usize,
                pub num_constraints: usize,
                pub sumcheck_id: $crate::poly::opening_proof::SumcheckId,
            }

            impl [<$name Params>] {
                pub fn new(num_constraints: usize) -> Self {
                    Self {
                        num_constraint_vars: $num_vars,
                        num_constraints,
                        sumcheck_id: $sumcheck_id,
                    }
                }
            }

            // ================================================================
            // Constants
            // ================================================================

            const [<$name:upper _OPENING_SPECS>]: [$crate::subprotocols::constraint_list_sumcheck::OpeningSpec; [<$name Values>]::<()>::COUNT] =
                $crate::subprotocols::constraint_list_sumcheck::sequential_opening_specs::<{ [<$name Values>]::<()>::COUNT }>();

            const [<$name:upper _DEGREE>]: usize = $degree;

            // ================================================================
            // Prover Spec
            // ================================================================

            #[derive(Clone, $crate::allocative::Allocative)]
            pub struct [<$name ProverSpec>]<F: $crate::field::JoltField> {
                params: [<$name Params>],
                polys_by_kind: Vec<Vec<$crate::poly::multilinear_polynomial::MultilinearPolynomial<F>>>,
            }

            impl<F: $crate::field::JoltField + $crate::allocative::Allocative> [<$name ProverSpec>]<F> {
                pub fn new(
                    params: [<$name Params>],
                    constraint_polys: Vec<[<$name ConstraintPolynomials>]<F>>,
                ) -> (Self, Vec<usize>) {
                    debug_assert_eq!(constraint_polys.len(), params.num_constraints);
                    let (polys_by_kind, constraint_indices) = [<$name ConstraintPolynomials>]::unpack_all(constraint_polys);
                    (Self { params, polys_by_kind }, constraint_indices)
                }
            }

            impl<F: $crate::field::JoltField + $crate::allocative::Allocative> $crate::subprotocols::constraint_list_sumcheck::ConstraintListSpec for [<$name ProverSpec>]<F> {
                fn sumcheck_id(&self) -> $crate::poly::opening_proof::SumcheckId { self.params.sumcheck_id }
                fn num_rounds(&self) -> usize { self.params.num_constraint_vars }
                fn num_instances(&self) -> usize { self.params.num_constraints }
                fn uses_term_batching(&self) -> bool { $uses_term_batching }
                fn opening_specs(&self) -> &'static [$crate::subprotocols::constraint_list_sumcheck::OpeningSpec] { &[<$name:upper _OPENING_SPECS>] }

                fn build_virtual_poly(&self, term_index: usize, instance: usize) -> $crate::zkvm::witness::VirtualPolynomial {
                    use $crate::zkvm::witness::TermEnum;
                    $crate::zkvm::witness::VirtualPolynomial::Recursion(
                        $crate::zkvm::witness::RecursionPoly::$recursion_variant {
                            term: <$term_enum>::from_index(term_index).expect("invalid term index"),
                            instance,
                        }
                    )
                }
            }

            impl<F: $crate::field::JoltField + $crate::allocative::Allocative> $crate::subprotocols::constraint_list_sumcheck::ConstraintListProverSpec<F, [<$name:upper _DEGREE>]> for [<$name ProverSpec>]<F> {
                fn polys_by_kind(&self) -> &[Vec<$crate::poly::multilinear_polynomial::MultilinearPolynomial<F>>] { &self.polys_by_kind }
                fn polys_by_kind_mut(&mut self) -> &mut [Vec<$crate::poly::multilinear_polynomial::MultilinearPolynomial<F>>] { &mut self.polys_by_kind }
                fn shared_polys(&self) -> &[$crate::poly::multilinear_polynomial::MultilinearPolynomial<F>] { &[] }
                fn shared_polys_mut(&mut self) -> &mut [$crate::poly::multilinear_polynomial::MultilinearPolynomial<F>] { &mut [] }

                fn eval_constraint(
                    &self,
                    _instance: usize,
                    eval_index: usize,
                    poly_evals: &[[F; [<$name:upper _DEGREE>]]],
                    _public_evals: &[[F; [<$name:upper _DEGREE>]]],
                    _shared_evals: &[[F; [<$name:upper _DEGREE>]]],
                    term_batch_coeff: Option<F>,
                ) -> F {
                    let vals = [<$name Values>]::from_poly_evals(poly_evals, eval_index);
                    if $uses_term_batching {
                        vals.eval_constraint(term_batch_coeff.expect(concat!(stringify!($name), " requires term_batch_coeff")))
                    } else {
                        vals.eval_constraint_no_batching()
                    }
                }
            }

            // ================================================================
            // Verifier Spec
            // ================================================================

            #[derive(Clone, $crate::allocative::Allocative)]
            pub struct [<$name VerifierSpec>]<F: $crate::field::JoltField> {
                params: [<$name Params>],
                _marker: std::marker::PhantomData<F>,
            }

            impl<F: $crate::field::JoltField + $crate::allocative::Allocative> [<$name VerifierSpec>]<F> {
                pub fn new(params: [<$name Params>]) -> Self {
                    Self { params, _marker: std::marker::PhantomData }
                }
            }

            impl<F: $crate::field::JoltField + $crate::allocative::Allocative> $crate::subprotocols::constraint_list_sumcheck::ConstraintListSpec for [<$name VerifierSpec>]<F> {
                fn sumcheck_id(&self) -> $crate::poly::opening_proof::SumcheckId { self.params.sumcheck_id }
                fn num_rounds(&self) -> usize { self.params.num_constraint_vars }
                fn num_instances(&self) -> usize { self.params.num_constraints }
                fn uses_term_batching(&self) -> bool { $uses_term_batching }
                fn opening_specs(&self) -> &'static [$crate::subprotocols::constraint_list_sumcheck::OpeningSpec] { &[<$name:upper _OPENING_SPECS>] }

                fn build_virtual_poly(&self, term_index: usize, instance: usize) -> $crate::zkvm::witness::VirtualPolynomial {
                    use $crate::zkvm::witness::TermEnum;
                    $crate::zkvm::witness::VirtualPolynomial::Recursion(
                        $crate::zkvm::witness::RecursionPoly::$recursion_variant {
                            term: <$term_enum>::from_index(term_index).expect("invalid term index"),
                            instance,
                        }
                    )
                }
            }

            impl<F: $crate::field::JoltField + $crate::allocative::Allocative> $crate::subprotocols::constraint_list_sumcheck::ConstraintListVerifierSpec<F, [<$name:upper _DEGREE>]> for [<$name VerifierSpec>]<F> {
                fn compute_shared_scalars(&self, _eval_point: &[F]) -> Vec<F> { vec![] }

                fn eval_constraint_at_point(
                    &self,
                    _instance: usize,
                    opened_claims: &[F],
                    _shared_scalars: &[F],
                    _eval_point: &[F],
                    term_batch_coeff: Option<F>,
                ) -> F {
                    let vals = [<$name Values>]::from_claims(opened_claims);
                    if $uses_term_batching {
                        vals.eval_constraint(term_batch_coeff.expect(concat!(stringify!($name), " requires term_batch_coeff")))
                    } else {
                        vals.eval_constraint_no_batching()
                    }
                }
            }

            // ================================================================
            // Type Aliases
            // ================================================================

            pub type [<$name Prover>]<F> = $crate::subprotocols::constraint_list_sumcheck::ConstraintListProver<F, [<$name ProverSpec>]<F>, [<$name:upper _DEGREE>]>;
            pub type [<$name Verifier>]<F> = $crate::subprotocols::constraint_list_sumcheck::ConstraintListVerifier<F, [<$name VerifierSpec>]<F>, [<$name:upper _DEGREE>]>;
        }
    };
}
