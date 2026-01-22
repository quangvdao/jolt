//! G2 addition sumcheck for proving G2 group addition constraints.
//!
//! This is the G2 analogue of `g1_add.rs`, but for points over Fq2. Since the recursion SNARK
//! runs over the base field Fq, we split each Fq2 coordinate into (c0,c1) components in Fq and
//! enforce all constraints component-wise.

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
use rayon::prelude::*;

/// Witness polynomials for a G2 addition constraint, proving R = P + Q.
///
/// Coordinates are in Fq2, represented as (c0,c1) in Fq.
#[derive(Clone, Debug)]
pub struct G2AddWitness {
    pub constraint_index: usize,
    // P
    pub x_p_c0: Vec<Fq>,
    pub x_p_c1: Vec<Fq>,
    pub y_p_c0: Vec<Fq>,
    pub y_p_c1: Vec<Fq>,
    pub ind_p: Vec<Fq>,
    // Q
    pub x_q_c0: Vec<Fq>,
    pub x_q_c1: Vec<Fq>,
    pub y_q_c0: Vec<Fq>,
    pub y_q_c1: Vec<Fq>,
    pub ind_q: Vec<Fq>,
    // R
    pub x_r_c0: Vec<Fq>,
    pub x_r_c1: Vec<Fq>,
    pub y_r_c0: Vec<Fq>,
    pub y_r_c1: Vec<Fq>,
    pub ind_r: Vec<Fq>,
    // λ and inv_dx in Fq2
    pub lambda_c0: Vec<Fq>,
    pub lambda_c1: Vec<Fq>,
    pub inv_delta_x_c0: Vec<Fq>,
    pub inv_delta_x_c1: Vec<Fq>,
    // branch bits (in Fq)
    pub is_double: Vec<Fq>,
    pub is_inverse: Vec<Fq>,
}

/// Helper to append all virtual claims for a G2 add constraint
#[allow(clippy::too_many_arguments)]
fn append_g2_add_virtual_claims<F: JoltField, T: Transcript>(
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
    x_p_c0: F,
    x_p_c1: F,
    y_p_c0: F,
    y_p_c1: F,
    ind_p: F,
    x_q_c0: F,
    x_q_c1: F,
    y_q_c0: F,
    y_q_c1: F,
    ind_q: F,
    x_r_c0: F,
    x_r_c1: F,
    y_r_c0: F,
    y_r_c1: F,
    ind_r: F,
    lambda_c0: F,
    lambda_c1: F,
    inv_dx_c0: F,
    inv_dx_c1: F,
    is_double: F,
    is_inverse: F,
) {
    let claims = virtual_claims![
        VirtualPolynomial::RecursionG2AddXPC0(constraint_idx) => x_p_c0,
        VirtualPolynomial::RecursionG2AddXPC1(constraint_idx) => x_p_c1,
        VirtualPolynomial::RecursionG2AddYPC0(constraint_idx) => y_p_c0,
        VirtualPolynomial::RecursionG2AddYPC1(constraint_idx) => y_p_c1,
        VirtualPolynomial::RecursionG2AddPIndicator(constraint_idx) => ind_p,
        VirtualPolynomial::RecursionG2AddXQC0(constraint_idx) => x_q_c0,
        VirtualPolynomial::RecursionG2AddXQC1(constraint_idx) => x_q_c1,
        VirtualPolynomial::RecursionG2AddYQC0(constraint_idx) => y_q_c0,
        VirtualPolynomial::RecursionG2AddYQC1(constraint_idx) => y_q_c1,
        VirtualPolynomial::RecursionG2AddQIndicator(constraint_idx) => ind_q,
        VirtualPolynomial::RecursionG2AddXRC0(constraint_idx) => x_r_c0,
        VirtualPolynomial::RecursionG2AddXRC1(constraint_idx) => x_r_c1,
        VirtualPolynomial::RecursionG2AddYRC0(constraint_idx) => y_r_c0,
        VirtualPolynomial::RecursionG2AddYRC1(constraint_idx) => y_r_c1,
        VirtualPolynomial::RecursionG2AddRIndicator(constraint_idx) => ind_r,
        VirtualPolynomial::RecursionG2AddLambdaC0(constraint_idx) => lambda_c0,
        VirtualPolynomial::RecursionG2AddLambdaC1(constraint_idx) => lambda_c1,
        VirtualPolynomial::RecursionG2AddInvDeltaXC0(constraint_idx) => inv_dx_c0,
        VirtualPolynomial::RecursionG2AddInvDeltaXC1(constraint_idx) => inv_dx_c1,
        VirtualPolynomial::RecursionG2AddIsDouble(constraint_idx) => is_double,
        VirtualPolynomial::RecursionG2AddIsInverse(constraint_idx) => is_inverse,
    ];
    append_virtual_claims(accumulator, transcript, sumcheck_id, opening_point, &claims);
}

#[allow(clippy::type_complexity)]
fn get_g2_add_virtual_claims<F: JoltField>(
    accumulator: &VerifierOpeningAccumulator<F>,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
) -> (
    F,
    F,
    F,
    F,
    F,
    F,
    F,
    F,
    F,
    F,
    F,
    F,
    F,
    F,
    F,
    F,
    F,
    F,
    F,
    F,
    F,
) {
    let polynomials = vec![
        VirtualPolynomial::RecursionG2AddXPC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddXPC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddYPC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddYPC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddPIndicator(constraint_idx),
        VirtualPolynomial::RecursionG2AddXQC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddXQC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddYQC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddYQC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddQIndicator(constraint_idx),
        VirtualPolynomial::RecursionG2AddXRC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddXRC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddYRC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddYRC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddRIndicator(constraint_idx),
        VirtualPolynomial::RecursionG2AddLambdaC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddLambdaC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddInvDeltaXC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddInvDeltaXC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddIsDouble(constraint_idx),
        VirtualPolynomial::RecursionG2AddIsInverse(constraint_idx),
    ];
    let claims = get_virtual_claims(accumulator, sumcheck_id, &polynomials);
    (
        claims[0], claims[1], claims[2], claims[3], claims[4], claims[5], claims[6], claims[7],
        claims[8], claims[9], claims[10], claims[11], claims[12], claims[13], claims[14],
        claims[15], claims[16], claims[17], claims[18], claims[19], claims[20],
    )
}

fn append_g2_add_virtual_openings<F: JoltField, T: Transcript>(
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
) {
    let polynomials = vec![
        VirtualPolynomial::RecursionG2AddXPC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddXPC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddYPC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddYPC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddPIndicator(constraint_idx),
        VirtualPolynomial::RecursionG2AddXQC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddXQC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddYQC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddYQC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddQIndicator(constraint_idx),
        VirtualPolynomial::RecursionG2AddXRC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddXRC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddYRC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddYRC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddRIndicator(constraint_idx),
        VirtualPolynomial::RecursionG2AddLambdaC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddLambdaC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddInvDeltaXC0(constraint_idx),
        VirtualPolynomial::RecursionG2AddInvDeltaXC1(constraint_idx),
        VirtualPolynomial::RecursionG2AddIsDouble(constraint_idx),
        VirtualPolynomial::RecursionG2AddIsInverse(constraint_idx),
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
// Fq2 helpers over the base field
// =============================================================================

#[inline]
fn fq2_add<F: JoltField>(a0: F, a1: F, b0: F, b1: F) -> (F, F) {
    (a0 + b0, a1 + b1)
}

#[inline]
fn fq2_sub<F: JoltField>(a0: F, a1: F, b0: F, b1: F) -> (F, F) {
    (a0 - b0, a1 - b1)
}

#[inline]
fn fq2_mul<F: JoltField>(a0: F, a1: F, b0: F, b1: F) -> (F, F) {
    // (a0 + a1 u)(b0 + b1 u) with u^2 = -1
    let c0 = a0 * b0 - a1 * b1;
    let c1 = a0 * b1 + a1 * b0;
    (c0, c1)
}

// =============================================================================
// Constraint evaluation (batched with δ)
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn eval_g2_add_constraint<F: JoltField>(
    x_p0: F,
    x_p1: F,
    y_p0: F,
    y_p1: F,
    ind_p: F,
    x_q0: F,
    x_q1: F,
    y_q0: F,
    y_q1: F,
    ind_q: F,
    x_r0: F,
    x_r1: F,
    y_r0: F,
    y_r1: F,
    ind_r: F,
    lambda0: F,
    lambda1: F,
    inv_dx0: F,
    inv_dx1: F,
    is_double: F,
    is_inverse: F,
    delta: F,
) -> F {
    let one = F::one();
    let two = F::from_u64(2);
    let three = F::from_u64(3);

    let (dx0, dx1) = fq2_sub(x_q0, x_q1, x_p0, x_p1);
    let (dy0, dy1) = fq2_sub(y_q0, y_q1, y_p0, y_p1);
    let s_finite = (one - ind_p) * (one - ind_q);

    let mut acc = F::zero();
    let mut delta_pow = F::one();

    // (0..2) indicator booleanity
    acc += delta_pow * (ind_p * (one - ind_p));
    delta_pow *= delta;
    acc += delta_pow * (ind_q * (one - ind_q));
    delta_pow *= delta;
    acc += delta_pow * (ind_r * (one - ind_r));
    delta_pow *= delta;

    // (3..14) infinity encoding for P/Q/R coords (ind * coord = 0)
    acc += delta_pow * (ind_p * x_p0);
    delta_pow *= delta;
    acc += delta_pow * (ind_p * x_p1);
    delta_pow *= delta;
    acc += delta_pow * (ind_p * y_p0);
    delta_pow *= delta;
    acc += delta_pow * (ind_p * y_p1);
    delta_pow *= delta;

    acc += delta_pow * (ind_q * x_q0);
    delta_pow *= delta;
    acc += delta_pow * (ind_q * x_q1);
    delta_pow *= delta;
    acc += delta_pow * (ind_q * y_q0);
    delta_pow *= delta;
    acc += delta_pow * (ind_q * y_q1);
    delta_pow *= delta;

    acc += delta_pow * (ind_r * x_r0);
    delta_pow *= delta;
    acc += delta_pow * (ind_r * x_r1);
    delta_pow *= delta;
    acc += delta_pow * (ind_r * y_r0);
    delta_pow *= delta;
    acc += delta_pow * (ind_r * y_r1);
    delta_pow *= delta;

    // (15..19) if P = O then R = Q
    acc += delta_pow * (ind_p * (x_r0 - x_q0));
    delta_pow *= delta;
    acc += delta_pow * (ind_p * (x_r1 - x_q1));
    delta_pow *= delta;
    acc += delta_pow * (ind_p * (y_r0 - y_q0));
    delta_pow *= delta;
    acc += delta_pow * (ind_p * (y_r1 - y_q1));
    delta_pow *= delta;
    acc += delta_pow * (ind_p * (ind_r - ind_q));
    delta_pow *= delta;

    // (20..24) if Q = O and P != O then R = P
    let q_inf = ind_q * (one - ind_p);
    acc += delta_pow * (q_inf * (x_r0 - x_p0));
    delta_pow *= delta;
    acc += delta_pow * (q_inf * (x_r1 - x_p1));
    delta_pow *= delta;
    acc += delta_pow * (q_inf * (y_r0 - y_p0));
    delta_pow *= delta;
    acc += delta_pow * (q_inf * (y_r1 - y_p1));
    delta_pow *= delta;
    acc += delta_pow * (q_inf * (ind_r - ind_p));
    delta_pow *= delta;

    // (25..26) booleanity of branch bits in finite case
    acc += delta_pow * (s_finite * is_double * (one - is_double));
    delta_pow *= delta;
    acc += delta_pow * (s_finite * is_inverse * (one - is_inverse));
    delta_pow *= delta;

    // (27..28) branch selection (Fq2 components)
    let sel = s_finite * (one - is_double - is_inverse);
    let (inv_dx_dx0, inv_dx_dx1) = fq2_mul(inv_dx0, inv_dx1, dx0, dx1);
    let one_minus0 = one - inv_dx_dx0;
    let one_minus1 = F::zero() - inv_dx_dx1;
    acc += delta_pow * (sel * one_minus0);
    delta_pow *= delta;
    acc += delta_pow * (sel * one_minus1);
    delta_pow *= delta;

    // (29..32) doubling implies dx = 0 and dy = 0 (components)
    let dbl = s_finite * is_double;
    acc += delta_pow * (dbl * dx0);
    delta_pow *= delta;
    acc += delta_pow * (dbl * dx1);
    delta_pow *= delta;
    acc += delta_pow * (dbl * dy0);
    delta_pow *= delta;
    acc += delta_pow * (dbl * dy1);
    delta_pow *= delta;

    // (33..36) inverse implies dx = 0 and (y_q + y_p) = 0
    let inv = s_finite * is_inverse;
    acc += delta_pow * (inv * dx0);
    delta_pow *= delta;
    acc += delta_pow * (inv * dx1);
    delta_pow *= delta;
    let (y_sum0, y_sum1) = fq2_add(y_q0, y_q1, y_p0, y_p1);
    acc += delta_pow * (inv * y_sum0);
    delta_pow *= delta;
    acc += delta_pow * (inv * y_sum1);
    delta_pow *= delta;

    // (37..38) slope equation (Fq2 components)
    // add branch: (1 - is_double - is_inverse) * (dx*lambda - dy)
    let add_sel = one - is_double - is_inverse;
    let (dx_l0, dx_l1) = fq2_mul(dx0, dx1, lambda0, lambda1);
    let add0 = add_sel * (dx_l0 - dy0);
    let add1 = add_sel * (dx_l1 - dy1);
    // double branch: is_double * (2*y_p*lambda - 3*x_p^2)
    let (ypl0, ypl1) = fq2_mul(y_p0, y_p1, lambda0, lambda1);
    let (xp2_0, xp2_1) = fq2_mul(x_p0, x_p1, x_p0, x_p1);
    let dbl0 = is_double * (two * ypl0 - three * xp2_0);
    let dbl1 = is_double * (two * ypl1 - three * xp2_1);
    acc += delta_pow * (s_finite * (add0 + dbl0));
    delta_pow *= delta;
    acc += delta_pow * (s_finite * (add1 + dbl1));
    delta_pow *= delta;

    // (39) inverse => ind_R = 1
    acc += delta_pow * (s_finite * is_inverse * (one - ind_r));
    delta_pow *= delta;
    // (40) non-inverse => ind_R = 0
    acc += delta_pow * (s_finite * (one - is_inverse) * ind_r);
    delta_pow *= delta;

    // (41..42) x_R formula (Fq2 components) for non-inverse
    let noninv = s_finite * (one - is_inverse);
    let (lambda2_0, lambda2_1) = fq2_mul(lambda0, lambda1, lambda0, lambda1);
    // rhs = lambda^2 - x_p - x_q
    let (rhs_x0, rhs_x1) = fq2_sub(lambda2_0, lambda2_1, x_p0 + x_q0, x_p1 + x_q1);
    acc += delta_pow * (noninv * (x_r0 - rhs_x0));
    delta_pow *= delta;
    acc += delta_pow * (noninv * (x_r1 - rhs_x1));
    delta_pow *= delta;

    // (43..44) y_R formula (Fq2 components) for non-inverse
    // y_r = lambda*(x_p - x_r) - y_p
    let (xpr0, xpr1) = fq2_sub(x_p0, x_p1, x_r0, x_r1);
    let (l_xpr0, l_xpr1) = fq2_mul(lambda0, lambda1, xpr0, xpr1);
    let (rhs_y0, rhs_y1) = fq2_sub(l_xpr0, l_xpr1, y_p0, y_p1);
    acc += delta_pow * (noninv * (y_r0 - rhs_y0));
    delta_pow *= delta;
    acc += delta_pow * (noninv * (y_r1 - rhs_y1));

    acc
}

// =============================================================================
// Sumcheck types
// =============================================================================

#[derive(Clone)]
pub struct G2AddConstraintPolynomials {
    pub x_p_c0: Vec<Fq>,
    pub x_p_c1: Vec<Fq>,
    pub y_p_c0: Vec<Fq>,
    pub y_p_c1: Vec<Fq>,
    pub ind_p: Vec<Fq>,
    pub x_q_c0: Vec<Fq>,
    pub x_q_c1: Vec<Fq>,
    pub y_q_c0: Vec<Fq>,
    pub y_q_c1: Vec<Fq>,
    pub ind_q: Vec<Fq>,
    pub x_r_c0: Vec<Fq>,
    pub x_r_c1: Vec<Fq>,
    pub y_r_c0: Vec<Fq>,
    pub y_r_c1: Vec<Fq>,
    pub ind_r: Vec<Fq>,
    pub lambda_c0: Vec<Fq>,
    pub lambda_c1: Vec<Fq>,
    pub inv_delta_x_c0: Vec<Fq>,
    pub inv_delta_x_c1: Vec<Fq>,
    pub is_double: Vec<Fq>,
    pub is_inverse: Vec<Fq>,
    pub constraint_index: usize,
}

#[derive(Clone)]
pub struct G2AddParams {
    pub num_constraint_vars: usize,
    pub num_constraints: usize,
    pub sumcheck_id: SumcheckId,
}

impl G2AddParams {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            num_constraint_vars: 11,
            num_constraints,
            sumcheck_id: SumcheckId::G2Add,
        }
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct G2AddProver<F: JoltField, T: Transcript> {
    pub params: G2AddParams,

    pub eq_x: MultilinearPolynomial<F>,
    pub r_x: Vec<F::Challenge>,
    pub gamma: F,
    pub delta: F,

    pub x_p_c0_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub x_p_c1_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_p_c0_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_p_c1_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub ind_p_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub x_q_c0_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub x_q_c1_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_q_c0_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_q_c1_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub ind_q_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub x_r_c0_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub x_r_c1_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_r_c0_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_r_c1_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub ind_r_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub lambda_c0_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub lambda_c1_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub inv_dx_c0_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub inv_dx_c1_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub is_double_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub is_inverse_mlpoly: Vec<MultilinearPolynomial<F>>,

    pub x_p_c0_claims: Vec<F>,
    pub x_p_c1_claims: Vec<F>,
    pub y_p_c0_claims: Vec<F>,
    pub y_p_c1_claims: Vec<F>,
    pub ind_p_claims: Vec<F>,
    pub x_q_c0_claims: Vec<F>,
    pub x_q_c1_claims: Vec<F>,
    pub y_q_c0_claims: Vec<F>,
    pub y_q_c1_claims: Vec<F>,
    pub ind_q_claims: Vec<F>,
    pub x_r_c0_claims: Vec<F>,
    pub x_r_c1_claims: Vec<F>,
    pub y_r_c0_claims: Vec<F>,
    pub y_r_c1_claims: Vec<F>,
    pub ind_r_claims: Vec<F>,
    pub lambda_c0_claims: Vec<F>,
    pub lambda_c1_claims: Vec<F>,
    pub inv_dx_c0_claims: Vec<F>,
    pub inv_dx_c1_claims: Vec<F>,
    pub is_double_claims: Vec<F>,
    pub is_inverse_claims: Vec<F>,

    pub round: usize,
    pub _marker: std::marker::PhantomData<T>,
}

impl<F: JoltField, T: Transcript> G2AddProver<F, T> {
    pub fn new(params: G2AddParams, constraint_polys: Vec<G2AddConstraintPolynomials>, transcript: &mut T) -> Self {
        use std::any::TypeId;
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("G2 add requires F = Fq for recursion SNARK");
        }

        let r_x: Vec<F::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();
        let gamma = transcript.challenge_scalar_optimized::<F>();
        let delta = transcript.challenge_scalar_optimized::<F>();
        let eq_x = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_x));

        let mut x_p_c0_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut x_p_c1_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut y_p_c0_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut y_p_c1_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut ind_p_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut x_q_c0_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut x_q_c1_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut y_q_c0_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut y_q_c1_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut ind_q_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut x_r_c0_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut x_r_c1_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut y_r_c0_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut y_r_c1_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut ind_r_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut lambda_c0_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut lambda_c1_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut inv_dx_c0_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut inv_dx_c1_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut is_double_mlpoly = Vec::with_capacity(constraint_polys.len());
        let mut is_inverse_mlpoly = Vec::with_capacity(constraint_polys.len());

        for poly in constraint_polys {
            // SAFETY: F == Fq
            let x_p_c0_f: Vec<F> = unsafe { std::mem::transmute(poly.x_p_c0) };
            let x_p_c1_f: Vec<F> = unsafe { std::mem::transmute(poly.x_p_c1) };
            let y_p_c0_f: Vec<F> = unsafe { std::mem::transmute(poly.y_p_c0) };
            let y_p_c1_f: Vec<F> = unsafe { std::mem::transmute(poly.y_p_c1) };
            let ind_p_f: Vec<F> = unsafe { std::mem::transmute(poly.ind_p) };
            let x_q_c0_f: Vec<F> = unsafe { std::mem::transmute(poly.x_q_c0) };
            let x_q_c1_f: Vec<F> = unsafe { std::mem::transmute(poly.x_q_c1) };
            let y_q_c0_f: Vec<F> = unsafe { std::mem::transmute(poly.y_q_c0) };
            let y_q_c1_f: Vec<F> = unsafe { std::mem::transmute(poly.y_q_c1) };
            let ind_q_f: Vec<F> = unsafe { std::mem::transmute(poly.ind_q) };
            let x_r_c0_f: Vec<F> = unsafe { std::mem::transmute(poly.x_r_c0) };
            let x_r_c1_f: Vec<F> = unsafe { std::mem::transmute(poly.x_r_c1) };
            let y_r_c0_f: Vec<F> = unsafe { std::mem::transmute(poly.y_r_c0) };
            let y_r_c1_f: Vec<F> = unsafe { std::mem::transmute(poly.y_r_c1) };
            let ind_r_f: Vec<F> = unsafe { std::mem::transmute(poly.ind_r) };
            let lambda_c0_f: Vec<F> = unsafe { std::mem::transmute(poly.lambda_c0) };
            let lambda_c1_f: Vec<F> = unsafe { std::mem::transmute(poly.lambda_c1) };
            let inv_dx_c0_f: Vec<F> = unsafe { std::mem::transmute(poly.inv_delta_x_c0) };
            let inv_dx_c1_f: Vec<F> = unsafe { std::mem::transmute(poly.inv_delta_x_c1) };
            let is_double_f: Vec<F> = unsafe { std::mem::transmute(poly.is_double) };
            let is_inverse_f: Vec<F> = unsafe { std::mem::transmute(poly.is_inverse) };

            x_p_c0_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(x_p_c0_f)));
            x_p_c1_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(x_p_c1_f)));
            y_p_c0_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(y_p_c0_f)));
            y_p_c1_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(y_p_c1_f)));
            ind_p_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(ind_p_f)));
            x_q_c0_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(x_q_c0_f)));
            x_q_c1_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(x_q_c1_f)));
            y_q_c0_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(y_q_c0_f)));
            y_q_c1_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(y_q_c1_f)));
            ind_q_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(ind_q_f)));
            x_r_c0_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(x_r_c0_f)));
            x_r_c1_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(x_r_c1_f)));
            y_r_c0_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(y_r_c0_f)));
            y_r_c1_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(y_r_c1_f)));
            ind_r_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(ind_r_f)));
            lambda_c0_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(lambda_c0_f)));
            lambda_c1_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(lambda_c1_f)));
            inv_dx_c0_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(inv_dx_c0_f)));
            inv_dx_c1_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(inv_dx_c1_f)));
            is_double_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(is_double_f)));
            is_inverse_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(is_inverse_f)));
        }

        Self {
            params,
            eq_x,
            r_x,
            gamma: gamma.into(),
            delta: delta.into(),
            x_p_c0_mlpoly,
            x_p_c1_mlpoly,
            y_p_c0_mlpoly,
            y_p_c1_mlpoly,
            ind_p_mlpoly,
            x_q_c0_mlpoly,
            x_q_c1_mlpoly,
            y_q_c0_mlpoly,
            y_q_c1_mlpoly,
            ind_q_mlpoly,
            x_r_c0_mlpoly,
            x_r_c1_mlpoly,
            y_r_c0_mlpoly,
            y_r_c1_mlpoly,
            ind_r_mlpoly,
            lambda_c0_mlpoly,
            lambda_c1_mlpoly,
            inv_dx_c0_mlpoly,
            inv_dx_c1_mlpoly,
            is_double_mlpoly,
            is_inverse_mlpoly,
            x_p_c0_claims: vec![],
            x_p_c1_claims: vec![],
            y_p_c0_claims: vec![],
            y_p_c1_claims: vec![],
            ind_p_claims: vec![],
            x_q_c0_claims: vec![],
            x_q_c1_claims: vec![],
            y_q_c0_claims: vec![],
            y_q_c1_claims: vec![],
            ind_q_claims: vec![],
            x_r_c0_claims: vec![],
            x_r_c1_claims: vec![],
            y_r_c0_claims: vec![],
            y_r_c1_claims: vec![],
            ind_r_claims: vec![],
            lambda_c0_claims: vec![],
            lambda_c1_claims: vec![],
            inv_dx_c0_claims: vec![],
            inv_dx_c1_claims: vec![],
            is_double_claims: vec![],
            is_inverse_claims: vec![],
            round: 0,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for G2AddProver<F, T> {
    fn degree(&self) -> usize {
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
                let eq_x_evals = self.eq_x.sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                let mut x_evals = [F::zero(); DEGREE];
                let mut gamma_power = self.gamma;

                for i in 0..self.params.num_constraints {
                    let x_p0 = self.x_p_c0_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let x_p1 = self.x_p_c1_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_p0 = self.y_p_c0_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_p1 = self.y_p_c1_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let ind_p = self.ind_p_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                    let x_q0 = self.x_q_c0_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let x_q1 = self.x_q_c1_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_q0 = self.y_q_c0_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_q1 = self.y_q_c1_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let ind_q = self.ind_q_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                    let x_r0 = self.x_r_c0_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let x_r1 = self.x_r_c1_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_r0 = self.y_r_c0_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_r1 = self.y_r_c1_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let ind_r = self.ind_r_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                    let lambda0 = self.lambda_c0_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let lambda1 = self.lambda_c1_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let inv0 = self.inv_dx_c0_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let inv1 = self.inv_dx_c1_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let is_double = self.is_double_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let is_inverse = self.is_inverse_mlpoly[i].sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                    for t in 0..DEGREE {
                        let constraint_val = eval_g2_add_constraint(
                            x_p0[t], x_p1[t], y_p0[t], y_p1[t], ind_p[t],
                            x_q0[t], x_q1[t], y_q0[t], y_q1[t], ind_q[t],
                            x_r0[t], x_r1[t], y_r0[t], y_r1[t], ind_r[t],
                            lambda0[t], lambda1[t],
                            inv0[t], inv1[t],
                            is_double[t], is_inverse[t],
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

        for poly in &mut self.x_p_c0_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.x_p_c1_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.y_p_c0_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.y_p_c1_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.ind_p_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }

        for poly in &mut self.x_q_c0_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.x_q_c1_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.y_q_c0_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.y_q_c1_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.ind_q_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }

        for poly in &mut self.x_r_c0_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.x_r_c1_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.y_r_c0_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.y_r_c1_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.ind_r_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }

        for poly in &mut self.lambda_c0_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.lambda_c1_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.inv_dx_c0_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.inv_dx_c1_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.is_double_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }
        for poly in &mut self.is_inverse_mlpoly { poly.bind_parallel(r_j, BindingOrder::LowToHigh); }

        self.round = round + 1;
        if self.round == self.params.num_constraint_vars {
            self.x_p_c0_claims.clear();
            self.x_p_c1_claims.clear();
            self.y_p_c0_claims.clear();
            self.y_p_c1_claims.clear();
            self.ind_p_claims.clear();
            self.x_q_c0_claims.clear();
            self.x_q_c1_claims.clear();
            self.y_q_c0_claims.clear();
            self.y_q_c1_claims.clear();
            self.ind_q_claims.clear();
            self.x_r_c0_claims.clear();
            self.x_r_c1_claims.clear();
            self.y_r_c0_claims.clear();
            self.y_r_c1_claims.clear();
            self.ind_r_claims.clear();
            self.lambda_c0_claims.clear();
            self.lambda_c1_claims.clear();
            self.inv_dx_c0_claims.clear();
            self.inv_dx_c1_claims.clear();
            self.is_double_claims.clear();
            self.is_inverse_claims.clear();

            for i in 0..self.params.num_constraints {
                self.x_p_c0_claims.push(self.x_p_c0_mlpoly[i].get_bound_coeff(0));
                self.x_p_c1_claims.push(self.x_p_c1_mlpoly[i].get_bound_coeff(0));
                self.y_p_c0_claims.push(self.y_p_c0_mlpoly[i].get_bound_coeff(0));
                self.y_p_c1_claims.push(self.y_p_c1_mlpoly[i].get_bound_coeff(0));
                self.ind_p_claims.push(self.ind_p_mlpoly[i].get_bound_coeff(0));
                self.x_q_c0_claims.push(self.x_q_c0_mlpoly[i].get_bound_coeff(0));
                self.x_q_c1_claims.push(self.x_q_c1_mlpoly[i].get_bound_coeff(0));
                self.y_q_c0_claims.push(self.y_q_c0_mlpoly[i].get_bound_coeff(0));
                self.y_q_c1_claims.push(self.y_q_c1_mlpoly[i].get_bound_coeff(0));
                self.ind_q_claims.push(self.ind_q_mlpoly[i].get_bound_coeff(0));
                self.x_r_c0_claims.push(self.x_r_c0_mlpoly[i].get_bound_coeff(0));
                self.x_r_c1_claims.push(self.x_r_c1_mlpoly[i].get_bound_coeff(0));
                self.y_r_c0_claims.push(self.y_r_c0_mlpoly[i].get_bound_coeff(0));
                self.y_r_c1_claims.push(self.y_r_c1_mlpoly[i].get_bound_coeff(0));
                self.ind_r_claims.push(self.ind_r_mlpoly[i].get_bound_coeff(0));
                self.lambda_c0_claims.push(self.lambda_c0_mlpoly[i].get_bound_coeff(0));
                self.lambda_c1_claims.push(self.lambda_c1_mlpoly[i].get_bound_coeff(0));
                self.inv_dx_c0_claims.push(self.inv_dx_c0_mlpoly[i].get_bound_coeff(0));
                self.inv_dx_c1_claims.push(self.inv_dx_c1_mlpoly[i].get_bound_coeff(0));
                self.is_double_claims.push(self.is_double_mlpoly[i].get_bound_coeff(0));
                self.is_inverse_claims.push(self.is_inverse_mlpoly[i].get_bound_coeff(0));
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
            append_g2_add_virtual_claims(
                accumulator,
                transcript,
                i,
                self.params.sumcheck_id,
                &opening_point,
                self.x_p_c0_claims[i],
                self.x_p_c1_claims[i],
                self.y_p_c0_claims[i],
                self.y_p_c1_claims[i],
                self.ind_p_claims[i],
                self.x_q_c0_claims[i],
                self.x_q_c1_claims[i],
                self.y_q_c0_claims[i],
                self.y_q_c1_claims[i],
                self.ind_q_claims[i],
                self.x_r_c0_claims[i],
                self.x_r_c1_claims[i],
                self.y_r_c0_claims[i],
                self.y_r_c1_claims[i],
                self.ind_r_claims[i],
                self.lambda_c0_claims[i],
                self.lambda_c1_claims[i],
                self.inv_dx_c0_claims[i],
                self.inv_dx_c1_claims[i],
                self.is_double_claims[i],
                self.is_inverse_claims[i],
            );
        }
    }
}

pub struct G2AddVerifier<F: JoltField> {
    pub params: G2AddParams,
    pub r_x: Vec<F::Challenge>,
    pub gamma: F,
    pub delta: F,
    pub num_constraints: usize,
}

impl<F: JoltField> G2AddVerifier<F> {
    pub fn new<T: Transcript>(params: G2AddParams, transcript: &mut T) -> Self {
        let r_x: Vec<F::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();
        let gamma = transcript.challenge_scalar_optimized::<F>();
        let delta = transcript.challenge_scalar_optimized::<F>();
        let num_constraints = params.num_constraints;
        Self { params, r_x, gamma: gamma.into(), delta: delta.into(), num_constraints }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for G2AddVerifier<F> {
    fn degree(&self) -> usize { 6 }
    fn num_rounds(&self) -> usize { self.params.num_constraint_vars }
    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F { F::zero() }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        use crate::poly::eq_poly::EqPolynomial;

        let r_x_f: Vec<F> = self.r_x.iter().map(|c| (*c).into()).collect();
        let r_star_f: Vec<F> = sumcheck_challenges.iter().rev().map(|c| (*c).into()).collect();
        let eq_eval = EqPolynomial::mle(&r_x_f, &r_star_f);

        let mut total = F::zero();
        let mut gamma_power = self.gamma;

        for i in 0..self.num_constraints {
            let (
                x_p0, x_p1, y_p0, y_p1, ind_p,
                x_q0, x_q1, y_q0, y_q1, ind_q,
                x_r0, x_r1, y_r0, y_r1, ind_r,
                lambda0, lambda1, inv0, inv1, is_double, is_inverse,
            ) = get_g2_add_virtual_claims(accumulator, i, self.params.sumcheck_id);

            let constraint_val = eval_g2_add_constraint(
                x_p0, x_p1, y_p0, y_p1, ind_p,
                x_q0, x_q1, y_q0, y_q1, ind_q,
                x_r0, x_r1, y_r0, y_r1, ind_r,
                lambda0, lambda1,
                inv0, inv1,
                is_double, is_inverse,
                self.delta,
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
            append_g2_add_virtual_openings(
                accumulator,
                transcript,
                i,
                self.params.sumcheck_id,
                &opening_point,
            );
        }
    }
}

