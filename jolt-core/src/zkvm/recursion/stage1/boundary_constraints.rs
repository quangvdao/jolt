//! Boundary sumcheck for verifying initial and final states of recursion witnesses
//!
//! Proves:
//! 1. Packed GT Exp: rho(0, r_x) = 1
//! 2. G1 Scalar Mul: A(0) = O
//! 3. G2 Scalar Mul: A(0) = O
//!
//! (Final state checks are currently placeholders as results are not public inputs)
//!
//! Protocol:
//! 1. Verifier sends r_x (4 challenges) to bind element variables for GT Exp.
//! 2. Sumcheck over s (8 rounds) proves boundary conditions.
//! 3. Output claims are added to Stage 2 accumulator.

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
    virtual_claims,
    zkvm::{
        recursion::{
            stage1::{
                g1_scalar_mul::G1ScalarMulPublicInputs, g2_scalar_mul::G2ScalarMulPublicInputs,
                packed_gt_exp::PackedGtExpPublicInputs,
            },
            utils::virtual_polynomial_utils::*,
        },
        witness::VirtualPolynomial,
    },
};
use rayon::prelude::*;

/// Parameters for boundary sumcheck
#[derive(Clone)]
pub struct BoundarySumcheckParams {
    /// Number of step variables (8 for G1/G2, 7 for GT Exp)
    pub num_rounds: usize,
    /// Number of Packed GT Exp instances
    pub num_gt_exp: usize,
    /// Number of G1 Scalar Mul instances
    pub num_g1: usize,
    /// Number of G2 Scalar Mul instances
    pub num_g2: usize,
    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,
}

impl BoundarySumcheckParams {
    pub fn new(num_gt_exp: usize, num_g1: usize, num_g2: usize) -> Self {
        Self {
            num_rounds: 8, // Max of 7 (GT) and 8 (G1/G2)
            num_gt_exp,
            num_g1,
            num_g2,
            sumcheck_id: SumcheckId::BoundaryConstraints,
        }
    }
}

/// Prover for boundary sumcheck
pub struct BoundarySumcheckProver<F: JoltField, T: Transcript> {
    params: BoundarySumcheckParams,

    // GT Exp polynomials (bound to r_x)
    rho_polys: Vec<MultilinearPolynomial<F>>,

    // G1 Scalar Mul polynomials
    g1_x_a_polys: Vec<MultilinearPolynomial<F>>,
    g1_y_a_polys: Vec<MultilinearPolynomial<F>>,
    g1_a_is_infinity_polys: Vec<MultilinearPolynomial<F>>,

    // G2 Scalar Mul polynomials (c0, c1 components)
    g2_x_a_c0_polys: Vec<MultilinearPolynomial<F>>,
    g2_x_a_c1_polys: Vec<MultilinearPolynomial<F>>,
    g2_y_a_c0_polys: Vec<MultilinearPolynomial<F>>,
    g2_y_a_c1_polys: Vec<MultilinearPolynomial<F>>,
    g2_a_is_infinity_polys: Vec<MultilinearPolynomial<F>>,

    // Eq polynomials for boundary checks
    eq_0: MultilinearPolynomial<F>,

    // Challenges
    r_x: Vec<F::Challenge>,
    gamma: F,

    round: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<F: JoltField, T: Transcript> BoundarySumcheckProver<F, T> {
    pub fn new(
        params: BoundarySumcheckParams,
        // GT Exp inputs
        gt_rho_polys: Vec<MultilinearPolynomial<F>>,
        _gt_public_inputs: &[PackedGtExpPublicInputs],
        // G1 inputs
        g1_polys: (
            Vec<MultilinearPolynomial<F>>, // x_a
            Vec<MultilinearPolynomial<F>>, // y_a
            Vec<MultilinearPolynomial<F>>, // a_is_infinity
        ),
        _g1_public_inputs: &[G1ScalarMulPublicInputs],
        // G2 inputs
        g2_polys: (
            Vec<MultilinearPolynomial<F>>, // x_a_c0
            Vec<MultilinearPolynomial<F>>, // x_a_c1
            Vec<MultilinearPolynomial<F>>, // y_a_c0
            Vec<MultilinearPolynomial<F>>, // y_a_c1
            Vec<MultilinearPolynomial<F>>, // a_is_infinity
        ),
        _g2_public_inputs: &[G2ScalarMulPublicInputs],
        transcript: &mut T,
    ) -> Self {
        // 1. Sample r_x for GT Exp element binding (4 variables)
        let r_x: Vec<F::Challenge> = (0..4)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        // 2. Bind GT Exp polynomials to r_x
        // rho is 11 vars: x (high 4) + s (low 7). We bind x (high) first.
        // HighToLow binding binds from the highest variable index.
        let mut rho_bound = gt_rho_polys;
        for poly in &mut rho_bound {
            for &r in r_x.iter().rev() {
                poly.bind_parallel(r, BindingOrder::HighToLow);
            }
        }

        let gamma = transcript.challenge_scalar_optimized::<F>().into();

        Self {
            params,
            rho_polys: rho_bound,
            g1_x_a_polys: g1_polys.0,
            g1_y_a_polys: g1_polys.1,
            g1_a_is_infinity_polys: g1_polys.2,
            g2_x_a_c0_polys: g2_polys.0,
            g2_x_a_c1_polys: g2_polys.1,
            g2_y_a_c0_polys: g2_polys.2,
            g2_y_a_c1_polys: g2_polys.3,
            g2_a_is_infinity_polys: g2_polys.4,
            eq_0: MultilinearPolynomial::default(), // Initialized in compute_message
            r_x,
            gamma,
            round: 0,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for BoundarySumcheckProver<F, T> {
    fn degree(&self) -> usize {
        1 // Linear in variables
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        F::zero() // Proving sum is 0
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let num_vars = self.params.num_rounds - round;
        let half = 1 << (num_vars - 1);

        if round == 0 {
            // Initialize eq polys
            let size = 1 << self.params.num_rounds;
            let mut eq_0_vals = vec![F::zero(); size];
            eq_0_vals[0] = F::one();
            self.eq_0 = MultilinearPolynomial::from(eq_0_vals);
        }

        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                let mut eval_0 = F::zero();
                let mut eval_1 = F::zero();

                // Get eq values for this round
                let eq_0_0 = self.eq_0.get_bound_coeff(i); // Value at x_round = 0
                let eq_0_1 = self.eq_0.get_bound_coeff(i + half); // Value at x_round = 1

                let mut gamma_pow = self.gamma; // Start with gamma^1

                // 1. Packed GT Exp
                for w in 0..self.params.num_gt_exp {
                    // rho(0) = 1
                    let rho_0 = self.rho_polys[w].get_bound_coeff(i);
                    let rho_1 = self.rho_polys[w].get_bound_coeff(i + half);

                    // Term: eq(0, s) * (rho - 1)
                    eval_0 += gamma_pow * eq_0_0 * (rho_0 - F::one());
                    eval_1 += gamma_pow * eq_0_1 * (rho_1 - F::one());

                    gamma_pow *= self.gamma;
                }

                // 2. G1 Scalar Mul
                for w in 0..self.params.num_g1 {
                    let x_a_0 = self.g1_x_a_polys[w].get_bound_coeff(i);
                    let x_a_1 = self.g1_x_a_polys[w].get_bound_coeff(i + half);
                    let y_a_0 = self.g1_y_a_polys[w].get_bound_coeff(i);
                    let y_a_1 = self.g1_y_a_polys[w].get_bound_coeff(i + half);
                    let ind_a_0 = self.g1_a_is_infinity_polys[w].get_bound_coeff(i);
                    let ind_a_1 = self.g1_a_is_infinity_polys[w].get_bound_coeff(i + half);

                    // Initial: A(0) = O => x=0, y=0, ind=1
                    eval_0 += gamma_pow * eq_0_0 * x_a_0;
                    eval_1 += gamma_pow * eq_0_1 * x_a_1;

                    eval_0 += gamma_pow * eq_0_0 * y_a_0;
                    eval_1 += gamma_pow * eq_0_1 * y_a_1;

                    eval_0 += gamma_pow * eq_0_0 * (ind_a_0 - F::one());
                    eval_1 += gamma_pow * eq_0_1 * (ind_a_1 - F::one());

                    gamma_pow *= self.gamma;
                }

                // 3. G2 Scalar Mul
                for w in 0..self.params.num_g2 {
                    let x_c0_0 = self.g2_x_a_c0_polys[w].get_bound_coeff(i);
                    let x_c0_1 = self.g2_x_a_c0_polys[w].get_bound_coeff(i + half);
                    let x_c1_0 = self.g2_x_a_c1_polys[w].get_bound_coeff(i);
                    let x_c1_1 = self.g2_x_a_c1_polys[w].get_bound_coeff(i + half);

                    let y_c0_0 = self.g2_y_a_c0_polys[w].get_bound_coeff(i);
                    let y_c0_1 = self.g2_y_a_c0_polys[w].get_bound_coeff(i + half);
                    let y_c1_0 = self.g2_y_a_c1_polys[w].get_bound_coeff(i);
                    let y_c1_1 = self.g2_y_a_c1_polys[w].get_bound_coeff(i + half);

                    let ind_a_0 = self.g2_a_is_infinity_polys[w].get_bound_coeff(i);
                    let ind_a_1 = self.g2_a_is_infinity_polys[w].get_bound_coeff(i + half);

                    // Initial: A(0) = O
                    eval_0 += gamma_pow * eq_0_0 * x_c0_0;
                    eval_1 += gamma_pow * eq_0_1 * x_c0_1;
                    eval_0 += gamma_pow * eq_0_0 * x_c1_0;
                    eval_1 += gamma_pow * eq_0_1 * x_c1_1;

                    eval_0 += gamma_pow * eq_0_0 * y_c0_0;
                    eval_1 += gamma_pow * eq_0_1 * y_c0_1;
                    eval_0 += gamma_pow * eq_0_0 * y_c1_0;
                    eval_1 += gamma_pow * eq_0_1 * y_c1_1;

                    eval_0 += gamma_pow * eq_0_0 * (ind_a_0 - F::one());
                    eval_1 += gamma_pow * eq_0_1 * (ind_a_1 - F::one());

                    gamma_pow *= self.gamma;
                }

                [eval_0, eval_1]
            })
            .reduce(|| [F::zero(), F::zero()], |a, b| [a[0] + b[0], a[1] + b[1]]);

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        // Bind all polynomials
        self.eq_0.bind_parallel(r_j, BindingOrder::LowToHigh);

        for p in &mut self.rho_polys {
            p.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        for p in &mut self.g1_x_a_polys {
            p.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for p in &mut self.g1_y_a_polys {
            p.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for p in &mut self.g1_a_is_infinity_polys {
            p.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        for p in &mut self.g2_x_a_c0_polys {
            p.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for p in &mut self.g2_x_a_c1_polys {
            p.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for p in &mut self.g2_y_a_c0_polys {
            p.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for p in &mut self.g2_y_a_c1_polys {
            p.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for p in &mut self.g2_a_is_infinity_polys {
            p.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        self.round = round + 1;
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_s = sumcheck_challenges.to_vec();

        // Construct opening point for GT Exp: (r_s || r_x)
        let mut full_point_gt = r_s.clone();
        for r in &self.r_x {
            full_point_gt.push(*r);
        }
        let opening_point_gt = OpeningPoint::<BIG_ENDIAN, F>::new(full_point_gt);

        // Opening point for G1/G2: (r_s)
        let opening_point_scalar = OpeningPoint::<BIG_ENDIAN, F>::new(r_s);

        // 1. GT Exp Claims
        for w in 0..self.params.num_gt_exp {
            let val = self.rho_polys[w].get_bound_coeff(0);
            let claims = virtual_claims![
                VirtualPolynomial::PackedGtExpRho(w) => val
            ];
            append_virtual_claims(
                accumulator,
                transcript,
                self.params.sumcheck_id,
                &opening_point_gt,
                &claims,
            );
        }

        // 2. G1 Claims
        for w in 0..self.params.num_g1 {
            let claims = virtual_claims![
                VirtualPolynomial::RecursionG1ScalarMulXA(w) => self.g1_x_a_polys[w].get_bound_coeff(0),
                VirtualPolynomial::RecursionG1ScalarMulYA(w) => self.g1_y_a_polys[w].get_bound_coeff(0),
                VirtualPolynomial::RecursionG1ScalarMulAIndicator(w) => self.g1_a_is_infinity_polys[w].get_bound_coeff(0)
            ];
            append_virtual_claims(
                accumulator,
                transcript,
                self.params.sumcheck_id,
                &opening_point_scalar,
                &claims,
            );
        }

        // 3. G2 Claims
        for w in 0..self.params.num_g2 {
            let claims = virtual_claims![
                VirtualPolynomial::RecursionG2ScalarMulXAC0(w) => self.g2_x_a_c0_polys[w].get_bound_coeff(0),
                VirtualPolynomial::RecursionG2ScalarMulXAC1(w) => self.g2_x_a_c1_polys[w].get_bound_coeff(0),
                VirtualPolynomial::RecursionG2ScalarMulYAC0(w) => self.g2_y_a_c0_polys[w].get_bound_coeff(0),
                VirtualPolynomial::RecursionG2ScalarMulYAC1(w) => self.g2_y_a_c1_polys[w].get_bound_coeff(0),
                VirtualPolynomial::RecursionG2ScalarMulAIndicator(w) => self.g2_a_is_infinity_polys[w].get_bound_coeff(0)
            ];
            append_virtual_claims(
                accumulator,
                transcript,
                self.params.sumcheck_id,
                &opening_point_scalar,
                &claims,
            );
        }
    }
}

/// Verifier for boundary sumcheck
pub struct BoundarySumcheckVerifier<F: JoltField> {
    params: BoundarySumcheckParams,
    r_x: Vec<F::Challenge>,
    gamma: F,
}

impl<F: JoltField> BoundarySumcheckVerifier<F> {
    pub fn new<T: Transcript>(
        params: BoundarySumcheckParams,
        _gt_public_inputs: &[PackedGtExpPublicInputs],
        _g1_public_inputs: &[G1ScalarMulPublicInputs],
        _g2_public_inputs: &[G2ScalarMulPublicInputs],
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<F::Challenge> = (0..4)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        let gamma = transcript.challenge_scalar_optimized::<F>().into();

        Self { params, r_x, gamma }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for BoundarySumcheckVerifier<F> {
    fn degree(&self) -> usize {
        1
    }
    fn num_rounds(&self) -> usize {
        self.params.num_rounds
    }
    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_s: Vec<F> = sumcheck_challenges.iter().map(|c| (*c).into()).collect();

        // Evaluate eq(0, r_s)
        let eq_0 = EqPolynomial::<F>::evals(&r_s)[0];

        let mut total = F::zero();
        let mut gamma_pow = self.gamma;

        // 1. GT Exp
        for w in 0..self.params.num_gt_exp {
            // Get rho(r_s, r_x)
            let (_, rho_val) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpRho(w),
                self.params.sumcheck_id,
            );

            // Term: eq(0, s) * (rho - 1)
            total += gamma_pow * eq_0 * (rho_val - F::one());

            gamma_pow *= self.gamma;
        }

        // 2. G1
        for w in 0..self.params.num_g1 {
            let (_, x_a) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionG1ScalarMulXA(w),
                self.params.sumcheck_id,
            );
            let (_, y_a) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionG1ScalarMulYA(w),
                self.params.sumcheck_id,
            );
            let (_, ind_a) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionG1ScalarMulAIndicator(w),
                self.params.sumcheck_id,
            );

            total += gamma_pow * eq_0 * x_a;
            total += gamma_pow * eq_0 * y_a;
            total += gamma_pow * eq_0 * (ind_a - F::one());

            gamma_pow *= self.gamma;
        }

        // 3. G2
        for w in 0..self.params.num_g2 {
            let (_, x_c0) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionG2ScalarMulXAC0(w),
                self.params.sumcheck_id,
            );
            let (_, x_c1) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionG2ScalarMulXAC1(w),
                self.params.sumcheck_id,
            );
            let (_, y_c0) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionG2ScalarMulYAC0(w),
                self.params.sumcheck_id,
            );
            let (_, y_c1) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionG2ScalarMulYAC1(w),
                self.params.sumcheck_id,
            );
            let (_, ind_a) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RecursionG2ScalarMulAIndicator(w),
                self.params.sumcheck_id,
            );

            total += gamma_pow * eq_0 * x_c0;
            total += gamma_pow * eq_0 * x_c1;
            total += gamma_pow * eq_0 * y_c0;
            total += gamma_pow * eq_0 * y_c1;
            total += gamma_pow * eq_0 * (ind_a - F::one());

            gamma_pow *= self.gamma;
        }

        total
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_s = sumcheck_challenges.to_vec();

        let mut full_point_gt = r_s.clone();
        for r in &self.r_x {
            full_point_gt.push(*r);
        }
        let opening_point_gt = OpeningPoint::<BIG_ENDIAN, F>::new(full_point_gt);

        let opening_point_scalar = OpeningPoint::<BIG_ENDIAN, F>::new(r_s);

        // 1. GT Exp Claims
        for w in 0..self.params.num_gt_exp {
            let claims = vec![VirtualPolynomial::PackedGtExpRho(w)];
            append_virtual_openings(
                accumulator,
                transcript,
                self.params.sumcheck_id,
                &opening_point_gt,
                &claims,
            );
        }

        // 2. G1 Claims
        for w in 0..self.params.num_g1 {
            let claims = vec![
                VirtualPolynomial::RecursionG1ScalarMulXA(w),
                VirtualPolynomial::RecursionG1ScalarMulYA(w),
                VirtualPolynomial::RecursionG1ScalarMulAIndicator(w),
            ];
            append_virtual_openings(
                accumulator,
                transcript,
                self.params.sumcheck_id,
                &opening_point_scalar,
                &claims,
            );
        }

        // 3. G2 Claims
        for w in 0..self.params.num_g2 {
            let claims = vec![
                VirtualPolynomial::RecursionG2ScalarMulXAC0(w),
                VirtualPolynomial::RecursionG2ScalarMulXAC1(w),
                VirtualPolynomial::RecursionG2ScalarMulYAC0(w),
                VirtualPolynomial::RecursionG2ScalarMulYAC1(w),
                VirtualPolynomial::RecursionG2ScalarMulAIndicator(w),
            ];
            append_virtual_openings(
                accumulator,
                transcript,
                self.params.sumcheck_id,
                &opening_point_scalar,
                &claims,
            );
        }
    }
}
