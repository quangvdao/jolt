//! Shift sumcheck for verifying `rho_next` claims in packed GT exponentiation.
//!
//! ## What objects this relates
//! For each packed GT exponentiation constraint instance \(i\), there is a committed packed
//! polynomial \(\rho_i(s,x)\) over:
//! - step index \(s \in \{0,1\}^7\) (128 steps), and
//! - element index \(x \in \{0,1\}^4\) (16 limb/element positions),
//!
//! giving an 11-var MLE overall.
//!
//! In the packed GT exp constraint system, `rho_next` is a **virtual** polynomial: it is not
//! committed/opened directly by the PCS. Instead, Stage 1 (`GtExp`) produces *claimed*
//! evaluations of `rho_next` at a random point, and this sumcheck verifies those claims are
//! consistent with the committed \(\rho_i\) after a one-step shift in \(s\).
//!
//! ## The precise relation being proved
//! Let \(r_i = (r^s_i, r^x_i) \in \mathbb{F}^{7} \times \mathbb{F}^{4}\) be the (shared) opening
//! point used by the `GtExp` sumcheck for instance \(i\). Let:
//! - \(v_i := \rho^{next}_i(r_i)\) be the claimed `rho_next` evaluation pulled from the opening
//!   accumulator via `VirtualPolynomial::gt_exp_rho_next(i)` under `SumcheckId::GtExp`.
//! - \(\gamma \in \mathbb{F}\) be the batching coefficient sampled by this sumcheck.
//!
//! This sumcheck proves the following batched identity:
//!
//! \[
//!   \sum_{i=0}^{m-1} \gamma^i \, v_i
//!   \;=\;
//!   \sum_{s \in \{0,1\}^7} \sum_{x \in \{0,1\}^4}
//!     EqPlusOne(r^s_i, s)\cdot Eq(r^x_i, x)\cdot \left(\sum_{i=0}^{m-1}\gamma^i \rho_i(s,x)\right)
//! \]
//!
//! where `EqPlusOne(r^s_i, s)` is the multilinear selector corresponding to shifting the step
//! index by one (with the out-of-range boundary contributing 0). Concretely, this enforces that
//! each claimed `rho_next` evaluation equals the committed `rho` evaluated at the shifted step
//! index (at the same element coordinates), at the shared random point \(r_i\).
//!
//! ## Boundary behavior
//! As with any shift, one boundary slice is intentionally excluded via the `EqPlusOne` selector
//! (the “out of range” step has weight 0). Any required boundary initialization/finalization
//! conditions are enforced by other constraints in the packed GT exp gadget, not here.

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
    virtual_claims,
    zkvm::{
        recursion::{constraints::config::CONFIG, utils::virtual_polynomial_utils::*},
        witness::VirtualPolynomial,
    },
};
use rayon::prelude::*;

/// A claim to verify in the shift sumcheck
#[derive(Clone, Debug)]
pub struct GtShiftClaim {
    /// Index of the constraint this claim belongs to
    pub constraint_idx: usize,
}

/// Parameters for shift rho sumcheck
#[derive(Clone)]
pub struct GtShiftParams {
    /// Number of variables (11 = 7 step + 4 element)
    pub num_vars: usize,
    /// Number of claims to verify
    pub num_claims: usize,
    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,
}

impl GtShiftParams {
    pub fn new(num_claims: usize) -> Self {
        Self {
            num_vars: CONFIG.packed_vars,
            num_claims,
            sumcheck_id: SumcheckId::GtShift,
        }
    }
}

/// Prover for shift rho sumcheck
pub struct GtShiftProver<F: JoltField, T: Transcript> {
    /// Parameters
    pub params: GtShiftParams,

    /// Rho polynomials (11-var, one per claim)
    pub rho_polys: Vec<MultilinearPolynomial<F>>,

    /// EqPlusOne polynomial for step variables (7-var, shared by all claims)
    pub eq_plus_one_poly: MultilinearPolynomial<F>,

    /// Eq polynomial for element variables (4-var, shared by all claims)
    pub eq_x_poly: MultilinearPolynomial<F>,

    /// Gamma for batching
    pub gamma: F,

    /// Current round
    pub round: usize,

    /// Claimed values for each rho_next
    pub claimed_values: Vec<F>,

    pub _marker: std::marker::PhantomData<T>,
}

#[cfg(feature = "allocative")]
impl<F: JoltField, T: Transcript> allocative::Allocative for GtShiftProver<F, T> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<F: JoltField, T: Transcript> GtShiftProver<F, T> {
    pub fn new(
        params: GtShiftParams,
        rho_polys: Vec<MultilinearPolynomial<F>>,
        claim_indices: Vec<usize>,
        accumulator: &ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        assert_eq!(params.num_claims, claim_indices.len());
        assert_eq!(params.num_claims, rho_polys.len());

        // Sample batching coefficient
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        struct ShiftEntry<F: JoltField> {
            constraint_idx: usize,
            rho_poly: MultilinearPolynomial<F>,
            point: OpeningPoint<BIG_ENDIAN, F>,
            claimed_value: F,
        }

        let mut entries: Vec<ShiftEntry<F>> = claim_indices
            .into_iter()
            .zip(rho_polys)
            .map(|(claim, rho_poly)| {
                let (point, claimed_value) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::gt_exp_rho_next(claim),
                    SumcheckId::GtExp,
                );
                ShiftEntry {
                    constraint_idx: claim,
                    rho_poly,
                    point,
                    claimed_value,
                }
            })
            .collect();

        entries.sort_by_key(|entry| entry.constraint_idx);

        // All claims should share the same evaluation point from GtExp sumcheck.
        let point = entries
            .first()
            .expect("shift rho requires at least one claim")
            .point
            .clone();
        for entry in &entries[1..] {
            debug_assert_eq!(
                entry.point.r, point.r,
                "GtShift claims must share the same opening point"
            );
        }

        // Extract rho polynomials and claimed values in the sorted order.
        let claimed_values: Vec<F> = entries.iter().map(|entry| entry.claimed_value).collect();
        let rho_polys: Vec<MultilinearPolynomial<F>> =
            entries.into_iter().map(|entry| entry.rho_poly).collect();

        // Eq eval tables expect MSB-first ordering; sumcheck challenges are LSB-first.
        let r_s: Vec<_> = point.r[..7].to_vec();
        let r_x: Vec<_> = point.r[7..].to_vec();

        // Create single EqPlusOne polynomial for step variables (7-var).
        // point.r is in sumcheck round order (LSB first). EqPlusOnePolynomial::evals expects
        // big-endian and will interpret this as MSB-first, effectively matching LSB variable order.
        let eq_plus_one_evals = eq_plus_one_lsb_evals::<F>(&r_s);
        let eq_plus_one_poly = MultilinearPolynomial::from(eq_plus_one_evals);

        // Create single Eq polynomial for element variables (4-var).
        // Same endianness convention as above.
        let eq_x_evals = eq_lsb_evals::<F>(&r_x);
        let eq_x_poly = MultilinearPolynomial::from(eq_x_evals);

        Self {
            params,
            rho_polys,
            eq_plus_one_poly,
            eq_x_poly,
            gamma,
            round: 0,
            claimed_values,
            _marker: std::marker::PhantomData,
        }
    }

    /// Check if we're in Phase 1 (step variable rounds 0-6)
    /// Data layout: index = x * 128 + s (s in low 7 bits)
    /// With LowToHigh binding: rounds 0-6 bind s (step), rounds 7-10 bind x (element)
    fn in_step_phase(&self) -> bool {
        self.round < 7 // First 7 rounds are step phase
    }
}

/// Compute Eq evaluations in LSB-first order
pub(crate) fn eq_lsb_evals<F: JoltField>(r: &[F::Challenge]) -> Vec<F> {
    let n = r.len();
    let mut evals = vec![F::zero(); 1 << n];
    for idx in 0..(1 << n) {
        let mut prod = F::one();
        for i in 0..n {
            let bit = ((idx >> i) & 1) == 1;
            let r_i: F = r[i].into();
            let y_i = if bit { F::one() } else { F::zero() };
            prod *= r_i * y_i + (F::one() - r_i) * (F::one() - y_i);
        }
        evals[idx] = prod;
    }
    evals
}

/// Compute Eq MLE in LSB-first order
pub(crate) fn eq_lsb_mle<F: JoltField>(r: &[F::Challenge], y: &[F::Challenge]) -> F {
    let mut prod = F::one();
    for i in 0..r.len() {
        let r_i: F = r[i].into();
        let y_i: F = y[i].into();
        prod *= r_i * y_i + (F::one() - r_i) * (F::one() - y_i);
    }
    prod
}

/// Compute EqPlusOne MLE in LSB-first order
/// EqPlusOne(r, y) = Eq(r, y+1) where y+1 is binary increment
pub(crate) fn eq_plus_one_lsb_mle<F: JoltField>(r: &[F::Challenge], y: &[F::Challenge]) -> F {
    let n = r.len();
    let one = F::one();
    let mut sum = F::zero();
    for k in 0..n {
        let mut lower = F::one();
        for i in 0..k {
            let r_i: F = r[i].into();
            let y_i: F = y[i].into();
            lower *= r_i * (one - y_i);
        }

        let r_k: F = r[k].into();
        let y_k: F = y[k].into();
        let kth = (one - r_k) * y_k;

        let mut higher = F::one();
        for i in (k + 1)..n {
            let r_i: F = r[i].into();
            let y_i: F = y[i].into();
            higher *= r_i * y_i + (one - r_i) * (one - y_i);
        }

        sum += lower * kth * higher;
    }
    sum
}

/// Compute EqPlusOne evaluations in LSB-first order
pub(crate) fn eq_plus_one_lsb_evals<F: JoltField>(r: &[F::Challenge]) -> Vec<F> {
    let n = r.len();
    let mut evals = vec![F::zero(); 1 << n];
    for idx in 0..(1 << n) {
        let mut y = vec![F::Challenge::from(0u128); n];
        for i in 0..n {
            if ((idx >> i) & 1) == 1 {
                y[i] = F::Challenge::from(1u128);
            }
        }
        evals[idx] = eq_plus_one_lsb_mle::<F>(r, &y);
    }
    evals
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for GtShiftProver<F, T> {
    fn degree(&self) -> usize {
        3 // EqPlusOne * Eq * rho (each degree 1)
    }

    fn num_rounds(&self) -> usize {
        self.params.num_vars
    }

    /// See `BatchedSumcheck`: shorter instances are suffix-aligned.
    fn round_offset(&self, max_num_rounds: usize) -> usize {
        max_num_rounds - self.params.num_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        // The sum should equal the batched claimed values
        let mut sum = F::zero();
        let mut gamma_power = F::one();

        for claimed_value in &self.claimed_values {
            sum += gamma_power * claimed_value;
            gamma_power *= self.gamma;
        }

        sum
    }

    #[tracing::instrument(skip_all, name = "GtShift::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 3;

        let half = if !self.rho_polys.is_empty() {
            self.rho_polys[0].len() / 2
        } else {
            return UniPoly::from_evals_and_hint(previous_claim, &[F::zero(); DEGREE]);
        };

        let gamma = self.gamma;
        let in_step_phase = self.in_step_phase();

        // Phase-aware sizes
        let eq_plus_one_half = if in_step_phase {
            self.eq_plus_one_poly.len() / 2
        } else {
            1 // Fully bound in element phase
        };
        let eq_x_len = self.eq_x_poly.len();

        // Compute evaluations in parallel
        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                // Compute eq contributions based on phase (shared across all claims)
                let (eq_plus_one_evals, eq_x_evals) = if in_step_phase {
                    // Phase 1 (rounds 0-6): sumcheck over s, eq_x is constant per x-block
                    // Index i maps to: s_pair_idx = i % eq_plus_one_half, x_idx = i / eq_plus_one_half
                    let s_pair_idx = i % eq_plus_one_half;
                    let x_idx = i / eq_plus_one_half;

                    let eq_plus_one_arr = self
                        .eq_plus_one_poly
                        .sumcheck_evals_array::<DEGREE>(s_pair_idx, BindingOrder::LowToHigh);

                    // eq_x[x_idx] is constant for this s-block
                    let eq_x_val = if x_idx < eq_x_len {
                        self.eq_x_poly.get_bound_coeff(x_idx)
                    } else {
                        F::zero()
                    };
                    let eq_x_arr = [eq_x_val; DEGREE];

                    (eq_plus_one_arr, eq_x_arr)
                } else {
                    // Phase 2 (rounds 7-10): eq_plus_one is fully bound (constant), sumcheck over x
                    let eq_plus_one_val = self.eq_plus_one_poly.get_bound_coeff(0);
                    let eq_plus_one_arr = [eq_plus_one_val; DEGREE];

                    let eq_x_arr = self
                        .eq_x_poly
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    (eq_plus_one_arr, eq_x_arr)
                };

                let mut term_evals = [F::zero(); DEGREE];
                let mut gamma_power = F::one();

                for claim_idx in 0..self.params.num_claims {
                    let rho_evals = self.rho_polys[claim_idx]
                        .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    for t in 0..DEGREE {
                        term_evals[t] +=
                            eq_plus_one_evals[t] * eq_x_evals[t] * gamma_power * rho_evals[t];
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
        // Always bind rho polynomials (11-var)
        for rho in &mut self.rho_polys {
            rho.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        if self.in_step_phase() {
            // Phase 1: Bind eq_plus_one polynomial (7-var)
            self.eq_plus_one_poly
                .bind_parallel(r_j, BindingOrder::LowToHigh);
            // eq_x polynomial remains unbound in this phase
        } else {
            // Phase 2: Bind eq_x polynomial (4-var)
            // eq_plus_one polynomial is already fully bound
            self.eq_x_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        self.round = round + 1;
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<{ BIG_ENDIAN }, F>::new(sumcheck_challenges.to_vec());

        for idx in 0..self.params.num_claims {
            // Get the final rho evaluation after all rounds
            let rho_eval = self.rho_polys[idx].get_bound_coeff(0);

            // Cache the rho evaluation at the shift sumcheck challenge point
            let claims = virtual_claims![
                VirtualPolynomial::gt_exp_rho(idx) => rho_eval,
            ];
            append_virtual_claims(
                accumulator,
                transcript,
                SumcheckId::GtShift,
                &opening_point,
                &claims,
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Verifier for shift rho sumcheck
pub struct GtShiftVerifier<F: JoltField> {
    pub params: GtShiftParams,
    pub claim_indices: Vec<usize>,
    pub gamma: F,
}

impl<F: JoltField> GtShiftVerifier<F> {
    pub fn new<T: Transcript>(
        params: GtShiftParams,
        claim_indices: Vec<usize>,
        transcript: &mut T,
    ) -> Self {
        assert_eq!(params.num_claims, claim_indices.len());

        // Sample same batching coefficient
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        Self {
            params,
            claim_indices,
            gamma,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for GtShiftVerifier<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.params.num_vars
    }

    /// See `GtShiftProver::round_offset`.
    fn round_offset(&self, max_num_rounds: usize) -> usize {
        max_num_rounds - self.params.num_vars
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        // The sum should equal the batched claimed values fetched from accumulator
        let mut sum = F::zero();
        let mut gamma_power = F::one();

        for claim in &self.claim_indices {
            let (_, value) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::gt_exp_rho_next(*claim),
                SumcheckId::GtExp,
            );

            sum += gamma_power * value;
            gamma_power *= self.gamma;
        }

        sum
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Get the shared point from the first claim (all claims share the same point)
        let (rho_next_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_exp_rho_next(self.claim_indices[0]),
            SumcheckId::GtExp,
        );

        // Eq eval tables expect MSB-first ordering; sumcheck challenges are LSB-first.
        let r_s: Vec<_> = rho_next_point.r[..7].to_vec();
        let r_x: Vec<_> = rho_next_point.r[7..].to_vec();

        let s_challenges: Vec<_> = sumcheck_challenges[..7].to_vec();
        let x_challenges: Vec<_> = sumcheck_challenges[7..].to_vec();

        // Compute eq polynomials once (shared across all claims).
        // Use LSB-first ordering for both the opening point and sumcheck challenges.
        let eq_plus_one = eq_plus_one_lsb_mle::<F>(&r_s, &s_challenges);
        let eq_x = eq_lsb_mle::<F>(&r_x, &x_challenges);
        let eq_product = eq_plus_one * eq_x;

        // Compute batched rho sum
        let mut rho_sum = F::zero();
        let mut gamma_power = F::one();

        for claim in &self.claim_indices {
            // Get rho evaluation at the shift sumcheck challenge point
            let (_, rho_eval) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::gt_exp_rho(*claim),
                SumcheckId::GtShift, // GtShift sumcheck ID!
            );

            rho_sum += gamma_power * rho_eval;
            gamma_power *= self.gamma;
        }

        eq_product * rho_sum
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<{ BIG_ENDIAN }, F>::new(sumcheck_challenges.to_vec());

        for claim in &self.claim_indices {
            let polynomials = vec![VirtualPolynomial::gt_exp_rho(*claim)];
            append_virtual_openings(
                accumulator,
                transcript,
                SumcheckId::GtShift,
                &opening_point,
                &polynomials,
            );
        }
    }
}
