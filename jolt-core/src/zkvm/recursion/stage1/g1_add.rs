//! G1 addition sumcheck for proving G1 group addition constraints.
//!
//! Proves: 0 = Σ_x eq(r_x, x) * Σ_i γ^i * (Σ_j δ^j * C_{i,j}(x))
//! where C_{i,j} are the per-instance addition constraints.
//!
//! This protocol is intended to be used inside the recursion SNARK Stage 1,
//! alongside GT exp/mul and G1/G2 scalar mul sumchecks.

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    virtual_claims,
    zkvm::{recursion::utils::virtual_polynomial_utils::*, witness::VirtualPolynomial},
};
use ark_bn254::Fq;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

/// Public inputs for a single G1 addition.
///
/// There are no public inputs for this sumcheck: all operands/results are witness polynomials.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G1AddPublicInputs {}

/// Witness polynomials for a G1 addition constraint, proving R = P + Q.
///
/// Points are encoded as affine coordinates (x,y) plus an "infinity indicator" bit:
/// - `ind = 1` => point is infinity and (x,y) must be (0,0)
/// - `ind = 0` => point is finite
#[derive(Clone, Debug)]
pub struct G1AddWitness {
    /// Index of this constraint in the constraint system
    pub constraint_index: usize,
    /// P = (x_P, y_P), ind_P
    pub x_p: Vec<Fq>,
    pub y_p: Vec<Fq>,
    pub ind_p: Vec<Fq>,
    /// Q = (x_Q, y_Q), ind_Q
    pub x_q: Vec<Fq>,
    pub y_q: Vec<Fq>,
    pub ind_q: Vec<Fq>,
    /// R = (x_R, y_R), ind_R
    pub x_r: Vec<Fq>,
    pub y_r: Vec<Fq>,
    pub ind_r: Vec<Fq>,
    /// Slope λ (affine add/double), witness
    pub lambda: Vec<Fq>,
    /// inv_delta_x = (x_Q - x_P)^{-1} in the generic-add case, else arbitrary
    pub inv_delta_x: Vec<Fq>,
    /// is_double = 1 iff the operation is doubling (P == Q) in the finite case
    pub is_double: Vec<Fq>,
    /// is_inverse = 1 iff the operation is inverse (P == -Q) in the finite case
    pub is_inverse: Vec<Fq>,
}

/// Helper to append all virtual claims for a G1 add constraint
#[allow(clippy::too_many_arguments)]
fn append_g1_add_virtual_claims<F: JoltField, T: Transcript>(
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
    x_p: F,
    y_p: F,
    ind_p: F,
    x_q: F,
    y_q: F,
    ind_q: F,
    x_r: F,
    y_r: F,
    ind_r: F,
    lambda: F,
    inv_delta_x: F,
    is_double: F,
    is_inverse: F,
) {
    let claims = virtual_claims![
        VirtualPolynomial::RecursionG1AddXP(constraint_idx) => x_p,
        VirtualPolynomial::RecursionG1AddYP(constraint_idx) => y_p,
        VirtualPolynomial::RecursionG1AddPIndicator(constraint_idx) => ind_p,
        VirtualPolynomial::RecursionG1AddXQ(constraint_idx) => x_q,
        VirtualPolynomial::RecursionG1AddYQ(constraint_idx) => y_q,
        VirtualPolynomial::RecursionG1AddQIndicator(constraint_idx) => ind_q,
        VirtualPolynomial::RecursionG1AddXR(constraint_idx) => x_r,
        VirtualPolynomial::RecursionG1AddYR(constraint_idx) => y_r,
        VirtualPolynomial::RecursionG1AddRIndicator(constraint_idx) => ind_r,
        VirtualPolynomial::RecursionG1AddLambda(constraint_idx) => lambda,
        VirtualPolynomial::RecursionG1AddInvDeltaX(constraint_idx) => inv_delta_x,
        VirtualPolynomial::RecursionG1AddIsDouble(constraint_idx) => is_double,
        VirtualPolynomial::RecursionG1AddIsInverse(constraint_idx) => is_inverse,
    ];
    append_virtual_claims(accumulator, transcript, sumcheck_id, opening_point, &claims);
}

/// Helper to retrieve all virtual claims for a G1 add constraint
#[allow(clippy::type_complexity)]
fn get_g1_add_virtual_claims<F: JoltField>(
    accumulator: &VerifierOpeningAccumulator<F>,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
) -> (F, F, F, F, F, F, F, F, F, F, F, F, F) {
    let polynomials = vec![
        VirtualPolynomial::RecursionG1AddXP(constraint_idx),
        VirtualPolynomial::RecursionG1AddYP(constraint_idx),
        VirtualPolynomial::RecursionG1AddPIndicator(constraint_idx),
        VirtualPolynomial::RecursionG1AddXQ(constraint_idx),
        VirtualPolynomial::RecursionG1AddYQ(constraint_idx),
        VirtualPolynomial::RecursionG1AddQIndicator(constraint_idx),
        VirtualPolynomial::RecursionG1AddXR(constraint_idx),
        VirtualPolynomial::RecursionG1AddYR(constraint_idx),
        VirtualPolynomial::RecursionG1AddRIndicator(constraint_idx),
        VirtualPolynomial::RecursionG1AddLambda(constraint_idx),
        VirtualPolynomial::RecursionG1AddInvDeltaX(constraint_idx),
        VirtualPolynomial::RecursionG1AddIsDouble(constraint_idx),
        VirtualPolynomial::RecursionG1AddIsInverse(constraint_idx),
    ];
    let claims = get_virtual_claims(accumulator, sumcheck_id, &polynomials);
    (
        claims[0], claims[1], claims[2], claims[3], claims[4], claims[5], claims[6], claims[7],
        claims[8], claims[9], claims[10], claims[11], claims[12],
    )
}

/// Helper to append virtual opening points for a G1 add constraint (verifier side)
fn append_g1_add_virtual_openings<F: JoltField, T: Transcript>(
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
) {
    let polynomials = vec![
        VirtualPolynomial::RecursionG1AddXP(constraint_idx),
        VirtualPolynomial::RecursionG1AddYP(constraint_idx),
        VirtualPolynomial::RecursionG1AddPIndicator(constraint_idx),
        VirtualPolynomial::RecursionG1AddXQ(constraint_idx),
        VirtualPolynomial::RecursionG1AddYQ(constraint_idx),
        VirtualPolynomial::RecursionG1AddQIndicator(constraint_idx),
        VirtualPolynomial::RecursionG1AddXR(constraint_idx),
        VirtualPolynomial::RecursionG1AddYR(constraint_idx),
        VirtualPolynomial::RecursionG1AddRIndicator(constraint_idx),
        VirtualPolynomial::RecursionG1AddLambda(constraint_idx),
        VirtualPolynomial::RecursionG1AddInvDeltaX(constraint_idx),
        VirtualPolynomial::RecursionG1AddIsDouble(constraint_idx),
        VirtualPolynomial::RecursionG1AddIsInverse(constraint_idx),
    ];
    append_virtual_openings(
        accumulator,
        transcript,
        sumcheck_id,
        opening_point,
        &polynomials,
    );
}

// =============================================================================
// CONSTRAINT EVALUATION (batched with δ)
// =============================================================================

/// Evaluate the batched G1 add constraint polynomial at a single point (all values are scalars in F).
#[allow(clippy::too_many_arguments)]
fn eval_g1_add_constraint<F: JoltField>(
    x_p: F,
    y_p: F,
    ind_p: F,
    x_q: F,
    y_q: F,
    ind_q: F,
    x_r: F,
    y_r: F,
    ind_r: F,
    lambda: F,
    inv_dx: F,
    is_double: F,
    is_inverse: F,
    delta: F,
) -> F {
    let one = F::one();
    let two = F::from_u64(2);
    let three = F::from_u64(3);

    let dx = x_q - x_p;
    let dy = y_q - y_p;
    let s_finite = (one - ind_p) * (one - ind_q);

    // Term ordering matters only for consistency between prover/verifier.
    // We batch all terms with powers of δ:
    //   Σ_j δ^j * term_j
    let mut acc = F::zero();
    let mut delta_pow = F::one();

    // (0) ind_P boolean
    acc += delta_pow * (ind_p * (one - ind_p));
    delta_pow *= delta;
    // (1) ind_Q boolean
    acc += delta_pow * (ind_q * (one - ind_q));
    delta_pow *= delta;
    // (2) ind_R boolean
    acc += delta_pow * (ind_r * (one - ind_r));
    delta_pow *= delta;

    // (3..8) infinity encoding: ind * x = 0, ind * y = 0
    acc += delta_pow * (ind_p * x_p);
    delta_pow *= delta;
    acc += delta_pow * (ind_p * y_p);
    delta_pow *= delta;
    acc += delta_pow * (ind_q * x_q);
    delta_pow *= delta;
    acc += delta_pow * (ind_q * y_q);
    delta_pow *= delta;
    acc += delta_pow * (ind_r * x_r);
    delta_pow *= delta;
    acc += delta_pow * (ind_r * y_r);
    delta_pow *= delta;

    // (9..11) if P = O then R = Q
    acc += delta_pow * (ind_p * (x_r - x_q));
    delta_pow *= delta;
    acc += delta_pow * (ind_p * (y_r - y_q));
    delta_pow *= delta;
    acc += delta_pow * (ind_p * (ind_r - ind_q));
    delta_pow *= delta;

    // (12..14) if Q = O and P != O then R = P
    let q_inf = ind_q * (one - ind_p);
    acc += delta_pow * (q_inf * (x_r - x_p));
    delta_pow *= delta;
    acc += delta_pow * (q_inf * (y_r - y_p));
    delta_pow *= delta;
    acc += delta_pow * (q_inf * (ind_r - ind_p));
    delta_pow *= delta;

    // (15..16) booleanity of branch bits in finite case
    acc += delta_pow * (s_finite * is_double * (one - is_double));
    delta_pow *= delta;
    acc += delta_pow * (s_finite * is_inverse * (one - is_inverse));
    delta_pow *= delta;

    // (17) branch selection: if x_Q = x_P then must be in (double or inverse),
    // else inv_dx must be the inverse of dx (so inv_dx * dx = 1).
    acc += delta_pow * (s_finite * (one - is_double - is_inverse) * (one - inv_dx * dx));
    delta_pow *= delta;

    // (18..19) if doubling, enforce P == Q
    acc += delta_pow * (s_finite * is_double * dx);
    delta_pow *= delta;
    acc += delta_pow * (s_finite * is_double * (y_q - y_p));
    delta_pow *= delta;

    // (20..21) if inverse, enforce P == -Q
    acc += delta_pow * (s_finite * is_inverse * dx);
    delta_pow *= delta;
    acc += delta_pow * (s_finite * is_inverse * (y_q + y_p));
    delta_pow *= delta;

    // (22) slope equation (add or double). Inverse case is ungated (vanishes).
    let add_branch = (one - is_double - is_inverse) * (dx * lambda - dy);
    let dbl_branch = is_double * (two * y_p * lambda - three * x_p * x_p);
    acc += delta_pow * (s_finite * (add_branch + dbl_branch));
    delta_pow *= delta;

    // (23) inverse => ind_R = 1
    acc += delta_pow * (s_finite * is_inverse * (one - ind_r));
    delta_pow *= delta;
    // (24) non-inverse => ind_R = 0
    acc += delta_pow * (s_finite * (one - is_inverse) * ind_r);
    delta_pow *= delta;

    // (25) x_R formula for non-inverse
    acc += delta_pow * (s_finite * (one - is_inverse) * (x_r - (lambda * lambda - x_p - x_q)));
    delta_pow *= delta;
    // (26) y_R formula for non-inverse
    acc += delta_pow * (s_finite * (one - is_inverse) * (y_r - (lambda * (x_p - x_r) - y_p)));

    acc
}

// =============================================================================
// SUMCHECK TYPES
// =============================================================================

#[derive(Clone)]
pub struct G1AddConstraintPolynomials {
    pub x_p: Vec<Fq>,
    pub y_p: Vec<Fq>,
    pub ind_p: Vec<Fq>,
    pub x_q: Vec<Fq>,
    pub y_q: Vec<Fq>,
    pub ind_q: Vec<Fq>,
    pub x_r: Vec<Fq>,
    pub y_r: Vec<Fq>,
    pub ind_r: Vec<Fq>,
    pub lambda: Vec<Fq>,
    pub inv_delta_x: Vec<Fq>,
    pub is_double: Vec<Fq>,
    pub is_inverse: Vec<Fq>,
    pub constraint_index: usize,
}

#[derive(Clone)]
pub struct G1AddParams {
    pub num_constraint_vars: usize,
    pub num_constraints: usize,
    pub sumcheck_id: SumcheckId,
}

impl G1AddParams {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            num_constraint_vars: 11, // uniform recursion matrix
            num_constraints,
            sumcheck_id: SumcheckId::G1Add,
        }
    }
}

/// Prover for the G1 addition sumcheck.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct G1AddProver<F: JoltField, T: Transcript> {
    pub params: G1AddParams,
    pub constraint_indices: Vec<usize>,

    pub eq_x: MultilinearPolynomial<F>,
    pub r_x: Vec<F::Challenge>,
    pub gamma: F,
    pub delta: F,

    pub x_p_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_p_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub ind_p_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub x_q_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_q_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub ind_q_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub x_r_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_r_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub ind_r_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub lambda_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub inv_dx_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub is_double_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub is_inverse_mlpoly: Vec<MultilinearPolynomial<F>>,

    pub x_p_claims: Vec<F>,
    pub y_p_claims: Vec<F>,
    pub ind_p_claims: Vec<F>,
    pub x_q_claims: Vec<F>,
    pub y_q_claims: Vec<F>,
    pub ind_q_claims: Vec<F>,
    pub x_r_claims: Vec<F>,
    pub y_r_claims: Vec<F>,
    pub ind_r_claims: Vec<F>,
    pub lambda_claims: Vec<F>,
    pub inv_dx_claims: Vec<F>,
    pub is_double_claims: Vec<F>,
    pub is_inverse_claims: Vec<F>,

    pub round: usize,
    pub _marker: std::marker::PhantomData<T>,
}

impl<F: JoltField, T: Transcript> G1AddProver<F, T> {
    pub fn new(
        params: G1AddParams,
        constraint_polys: Vec<G1AddConstraintPolynomials>,
        transcript: &mut T,
    ) -> Self {
        use std::any::TypeId;
        // Runtime check that F = Fq for recursion SNARK
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("G1 add requires F = Fq for recursion SNARK");
        }

        let r_x: Vec<F::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();
        let gamma = transcript.challenge_scalar_optimized::<F>();
        let delta = transcript.challenge_scalar_optimized::<F>();

        let eq_x = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_x));

        let mut constraint_indices = Vec::with_capacity(constraint_polys.len());
        let mut x_p_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut y_p_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut ind_p_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut x_q_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut y_q_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut ind_q_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut x_r_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut y_r_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut ind_r_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut lambda_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut inv_dx_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut is_double_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut is_inverse_mlpoly = Vec::with_capacity(constraint_polys.len());

        for poly in constraint_polys {
            constraint_indices.push(poly.constraint_index);
            // SAFETY: We checked F = Fq above, so these transmutes are safe
            let x_p_f: Vec<F> = unsafe { std::mem::transmute(poly.x_p) };
            let y_p_f: Vec<F> = unsafe { std::mem::transmute(poly.y_p) };
            let ind_p_f: Vec<F> = unsafe { std::mem::transmute(poly.ind_p) };
            let x_q_f: Vec<F> = unsafe { std::mem::transmute(poly.x_q) };
            let y_q_f: Vec<F> = unsafe { std::mem::transmute(poly.y_q) };
            let ind_q_f: Vec<F> = unsafe { std::mem::transmute(poly.ind_q) };
            let x_r_f: Vec<F> = unsafe { std::mem::transmute(poly.x_r) };
            let y_r_f: Vec<F> = unsafe { std::mem::transmute(poly.y_r) };
            let ind_r_f: Vec<F> = unsafe { std::mem::transmute(poly.ind_r) };
            let lambda_f: Vec<F> = unsafe { std::mem::transmute(poly.lambda) };
            let inv_dx_f: Vec<F> = unsafe { std::mem::transmute(poly.inv_delta_x) };
            let is_double_f: Vec<F> = unsafe { std::mem::transmute(poly.is_double) };
            let is_inverse_f: Vec<F> = unsafe { std::mem::transmute(poly.is_inverse) };

            x_p_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                x_p_f,
            )));
            y_p_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                y_p_f,
            )));
            ind_p_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                ind_p_f,
            )));
            x_q_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                x_q_f,
            )));
            y_q_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                y_q_f,
            )));
            ind_q_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                ind_q_f,
            )));
            x_r_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                x_r_f,
            )));
            y_r_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                y_r_f,
            )));
            ind_r_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                ind_r_f,
            )));
            lambda_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                lambda_f,
            )));
            inv_dx_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                inv_dx_f,
            )));
            is_double_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                is_double_f,
            )));
            is_inverse_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                is_inverse_f,
            )));
        }

        Self {
            params,
            constraint_indices,
            eq_x,
            r_x,
            gamma: gamma.into(),
            delta: delta.into(),
            x_p_mlpoly,
            y_p_mlpoly,
            ind_p_mlpoly,
            x_q_mlpoly,
            y_q_mlpoly,
            ind_q_mlpoly,
            x_r_mlpoly,
            y_r_mlpoly,
            ind_r_mlpoly,
            lambda_mlpoly,
            inv_dx_mlpoly,
            is_double_mlpoly,
            is_inverse_mlpoly,
            x_p_claims: vec![],
            y_p_claims: vec![],
            ind_p_claims: vec![],
            x_q_claims: vec![],
            y_q_claims: vec![],
            ind_q_claims: vec![],
            x_r_claims: vec![],
            y_r_claims: vec![],
            ind_r_claims: vec![],
            lambda_claims: vec![],
            inv_dx_claims: vec![],
            is_double_claims: vec![],
            is_inverse_claims: vec![],
            round: 0,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for G1AddProver<F, T> {
    fn degree(&self) -> usize {
        // Max constraint degree is 5; multiplied by eq_x (degree 1) => 6.
        6
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 6;
        let num_x_remaining = self.eq_x.get_num_vars();
        let x_half = 1 << (num_x_remaining - 1);

        let total_evals = (0..x_half)
            .into_par_iter()
            .map(|x_idx| {
                let eq_x_evals = self
                    .eq_x
                    .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                let mut x_evals = [F::zero(); DEGREE];
                let mut gamma_power = self.gamma;

                for i in 0..self.params.num_constraints {
                    let x_p_evals = self.x_p_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_p_evals = self.y_p_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let ind_p_evals = self.ind_p_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let x_q_evals = self.x_q_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_q_evals = self.y_q_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let ind_q_evals = self.ind_q_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let x_r_evals = self.x_r_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_r_evals = self.y_r_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let ind_r_evals = self.ind_r_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let lambda_evals = self.lambda_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let inv_dx_evals = self.inv_dx_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let is_double_evals = self.is_double_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let is_inverse_evals = self.is_inverse_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                    for t in 0..DEGREE {
                        let constraint_val = eval_g1_add_constraint(
                            x_p_evals[t],
                            y_p_evals[t],
                            ind_p_evals[t],
                            x_q_evals[t],
                            y_q_evals[t],
                            ind_q_evals[t],
                            x_r_evals[t],
                            y_r_evals[t],
                            ind_r_evals[t],
                            lambda_evals[t],
                            inv_dx_evals[t],
                            is_double_evals[t],
                            is_inverse_evals[t],
                            self.delta,
                        );
                        x_evals[t] += eq_x_evals[t] * gamma_power * constraint_val;
                    }

                    gamma_power *= self.gamma;
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

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.eq_x.bind_parallel(r_j, BindingOrder::LowToHigh);

        for poly in &mut self.x_p_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_p_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.ind_p_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.x_q_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_q_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.ind_q_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.x_r_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_r_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.ind_r_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.lambda_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.inv_dx_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.is_double_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.is_inverse_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        self.round = round + 1;
        if self.round == self.params.num_constraint_vars {
            self.x_p_claims.clear();
            self.y_p_claims.clear();
            self.ind_p_claims.clear();
            self.x_q_claims.clear();
            self.y_q_claims.clear();
            self.ind_q_claims.clear();
            self.x_r_claims.clear();
            self.y_r_claims.clear();
            self.ind_r_claims.clear();
            self.lambda_claims.clear();
            self.inv_dx_claims.clear();
            self.is_double_claims.clear();
            self.is_inverse_claims.clear();

            for i in 0..self.params.num_constraints {
                self.x_p_claims.push(self.x_p_mlpoly[i].get_bound_coeff(0));
                self.y_p_claims.push(self.y_p_mlpoly[i].get_bound_coeff(0));
                self.ind_p_claims
                    .push(self.ind_p_mlpoly[i].get_bound_coeff(0));
                self.x_q_claims.push(self.x_q_mlpoly[i].get_bound_coeff(0));
                self.y_q_claims.push(self.y_q_mlpoly[i].get_bound_coeff(0));
                self.ind_q_claims
                    .push(self.ind_q_mlpoly[i].get_bound_coeff(0));
                self.x_r_claims.push(self.x_r_mlpoly[i].get_bound_coeff(0));
                self.y_r_claims.push(self.y_r_mlpoly[i].get_bound_coeff(0));
                self.ind_r_claims
                    .push(self.ind_r_mlpoly[i].get_bound_coeff(0));
                self.lambda_claims
                    .push(self.lambda_mlpoly[i].get_bound_coeff(0));
                self.inv_dx_claims
                    .push(self.inv_dx_mlpoly[i].get_bound_coeff(0));
                self.is_double_claims
                    .push(self.is_double_mlpoly[i].get_bound_coeff(0));
                self.is_inverse_claims
                    .push(self.is_inverse_mlpoly[i].get_bound_coeff(0));
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
        for i in 0..self.params.num_constraints {
            append_g1_add_virtual_claims(
                accumulator,
                transcript,
                i,
                self.params.sumcheck_id,
                &opening_point,
                self.x_p_claims[i],
                self.y_p_claims[i],
                self.ind_p_claims[i],
                self.x_q_claims[i],
                self.y_q_claims[i],
                self.ind_q_claims[i],
                self.x_r_claims[i],
                self.y_r_claims[i],
                self.ind_r_claims[i],
                self.lambda_claims[i],
                self.inv_dx_claims[i],
                self.is_double_claims[i],
                self.is_inverse_claims[i],
            );
        }
    }
}

/// Verifier for the G1 addition sumcheck.
pub struct G1AddVerifier<F: JoltField> {
    pub params: G1AddParams,
    pub r_x: Vec<F::Challenge>,
    pub gamma: F,
    pub delta: F,
    pub num_constraints: usize,
    pub constraint_indices: Vec<usize>,
}

impl<F: JoltField> G1AddVerifier<F> {
    pub fn new<T: Transcript>(
        params: G1AddParams,
        constraint_indices: Vec<usize>,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<F::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();
        let gamma = transcript.challenge_scalar_optimized::<F>();
        let delta = transcript.challenge_scalar_optimized::<F>();
        let num_constraints = params.num_constraints;
        Self {
            params,
            r_x,
            gamma: gamma.into(),
            delta: delta.into(),
            num_constraints,
            constraint_indices,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for G1AddVerifier<F> {
    fn degree(&self) -> usize {
        6
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        use crate::poly::eq_poly::EqPolynomial;

        let r_x_f: Vec<F> = self.r_x.iter().map(|c| (*c).into()).collect();
        let r_star_f: Vec<F> = sumcheck_challenges
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();
        let eq_eval = EqPolynomial::mle(&r_x_f, &r_star_f);

        let mut total = F::zero();
        let mut gamma_power = self.gamma;

        for i in 0..self.num_constraints {
            let global_idx = self.constraint_indices[i];
            let (
                x_p,
                y_p,
                ind_p,
                x_q,
                y_q,
                ind_q,
                x_r,
                y_r,
                ind_r,
                lambda,
                inv_dx,
                is_double,
                is_inverse,
            ) = get_g1_add_virtual_claims(accumulator, global_idx, self.params.sumcheck_id);

            let constraint_val = eval_g1_add_constraint(
                x_p, y_p, ind_p, x_q, y_q, ind_q, x_r, y_r, ind_r, lambda, inv_dx, is_double,
                is_inverse, self.delta,
            );

            total += gamma_power * constraint_val;
            gamma_power *= self.gamma;
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
        for i in 0..self.num_constraints {
            append_g1_add_virtual_openings(
                accumulator,
                transcript,
                self.constraint_indices[i],
                self.params.sumcheck_id,
                &opening_point,
            );
        }
    }
}
