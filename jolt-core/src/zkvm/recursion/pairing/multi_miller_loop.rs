//! Multi-Miller loop sumcheck for proving pairing constraints.
//!
//! This sumcheck verifies the Multi-Miller loop computation for BN254 pairings.
//! It combines:
//! 1. G2 point updates (affine coordinates with witnessed inverses)
//! 2. Line evaluations (ell-coefficients)
//! 3. Accumulator updates (Fq12 multiplication via ring-switching)
//!
//! The constraints are batched using `delta` for term batching.

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::SumcheckId,
    },
    zkvm::recursion::constraints::sumcheck::{
        sequential_opening_specs, ConstraintListProver, ConstraintListProverSpec,
        ConstraintListSpec, ConstraintListVerifier, ConstraintListVerifierSpec, OpeningSpec,
    },
    zkvm::witness::{MultiMillerLoopTerm, RecursionPoly, TermEnum, VirtualPolynomial},
};

use allocative::Allocative;
use ark_bn254::{Fq, Fq12, Fq2, Fq6};
use ark_ff::{One, Zero};
use jolt_optimizations::{fq12_to_multilinear_evals, get_g_mle};

// =============================================================================
// Constants
// =============================================================================

const NUM_VARS: usize = 11;
const STEP_VARS: usize = 7;
const ELEM_VARS: usize = 4;
const STEP_SIZE: usize = 1 << STEP_VARS; // 128
const ELEM_SIZE: usize = 1 << ELEM_VARS; // 16

const NUM_COMMITTED_KINDS: usize = 24; // Must match MultiMillerLoopTerm::COUNT
const DEGREE: usize = 6;
const OPENING_SPECS: [OpeningSpec; NUM_COMMITTED_KINDS] = sequential_opening_specs();

// =============================================================================
// Witness / Params
// =============================================================================

/// Constraint polynomials for one MultiMillerLoop instance (one traced (G1,G2) pair).
#[derive(Clone, Debug, Allocative)]
pub struct MultiMillerLoopWitness<F> {
    pub f: Vec<F>,
    pub f_next: Vec<F>,
    pub quotient: Vec<F>,

    pub t_x_c0: Vec<F>,
    pub t_x_c1: Vec<F>,
    pub t_y_c0: Vec<F>,
    pub t_y_c1: Vec<F>,
    pub t_x_c0_next: Vec<F>,
    pub t_x_c1_next: Vec<F>,
    pub t_y_c0_next: Vec<F>,
    pub t_y_c1_next: Vec<F>,

    pub lambda_c0: Vec<F>,
    pub lambda_c1: Vec<F>,
    pub inv_delta_x_c0: Vec<F>,
    pub inv_delta_x_c1: Vec<F>,

    pub x_p: Vec<F>,
    pub y_p: Vec<F>,

    pub x_q_c0: Vec<F>,
    pub x_q_c1: Vec<F>,
    pub y_q_c0: Vec<F>,
    pub y_q_c1: Vec<F>,

    pub is_double: Vec<F>,
    pub is_add: Vec<F>,

    /// Line evaluation as an Fq12 element, expanded to 16 MLE evals per step and packed.
    pub l_val: Vec<F>,

    pub constraint_index: usize,
}

/// Parameters for MultiMillerLoop sumcheck.
#[derive(Clone, Allocative)]
pub struct MultiMillerLoopParams {
    pub num_constraint_vars: usize,
    pub num_constraints: usize,
    pub sumcheck_id: SumcheckId,
}

impl MultiMillerLoopParams {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            num_constraint_vars: NUM_VARS,
            num_constraints,
            sumcheck_id: SumcheckId::MultiMillerLoop,
        }
    }
}

// =============================================================================
// Shared constants (g + selectors)
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct SharedScalars<F> {
    g: F,
    sel0: F,
    sel1: F,
    sel2: F,
    sel3: F,
    sel4: F,
    sel5: F,
}

fn fq12_sparse_034(c0: Fq2, c3: Fq2, c4: Fq2) -> Fq12 {
    // Fq12 = c0 + c1 * w where c0,c1 ∈ Fq6
    // Fq6 = a0 + a1 * v + a2 * v^2 where ai ∈ Fq2
    //
    // `mul_by_034(c0,c3,c4)` corresponds to multiplying by:
    //   (c0, 0, 0) + (c3, c4, 0) * w
    Fq12::new(
        Fq6::new(c0, Fq2::zero(), Fq2::zero()),
        Fq6::new(c3, c4, Fq2::zero()),
    )
}

fn expand_elem_4_to_11(evals_4: &[Fq]) -> Vec<Fq> {
    debug_assert_eq!(evals_4.len(), ELEM_SIZE);
    let mut evals_11 = vec![Fq::zero(); 1 << NUM_VARS];
    for x in 0..ELEM_SIZE {
        let base = x * STEP_SIZE;
        let v = evals_4[x];
        for s in 0..STEP_SIZE {
            evals_11[base + s] = v;
        }
    }
    evals_11
}

fn build_shared_polys_11() -> Vec<MultilinearPolynomial<Fq>> {
    // g(x)
    let g_4 = get_g_mle();
    debug_assert_eq!(g_4.len(), ELEM_SIZE);

    // Selector basis: coefficients at sparse-034 slots (0,3,4) in Fq2, split into (c0,c1).
    let basis0_c0 = fq12_sparse_034(Fq2::new(Fq::one(), Fq::zero()), Fq2::zero(), Fq2::zero());
    let basis0_c1 = fq12_sparse_034(Fq2::new(Fq::zero(), Fq::one()), Fq2::zero(), Fq2::zero());
    let basis3_c0 = fq12_sparse_034(Fq2::zero(), Fq2::new(Fq::one(), Fq::zero()), Fq2::zero());
    let basis3_c1 = fq12_sparse_034(Fq2::zero(), Fq2::new(Fq::zero(), Fq::one()), Fq2::zero());
    let basis4_c0 = fq12_sparse_034(Fq2::zero(), Fq2::zero(), Fq2::new(Fq::one(), Fq::zero()));
    let basis4_c1 = fq12_sparse_034(Fq2::zero(), Fq2::zero(), Fq2::new(Fq::zero(), Fq::one()));

    let sel0_4 = fq12_to_multilinear_evals(&basis0_c0);
    let sel1_4 = fq12_to_multilinear_evals(&basis0_c1);
    let sel2_4 = fq12_to_multilinear_evals(&basis3_c0);
    let sel3_4 = fq12_to_multilinear_evals(&basis3_c1);
    let sel4_4 = fq12_to_multilinear_evals(&basis4_c0);
    let sel5_4 = fq12_to_multilinear_evals(&basis4_c1);

    let mut out = Vec::with_capacity(7);
    for evals_4 in [g_4, sel0_4, sel1_4, sel2_4, sel3_4, sel4_4, sel5_4] {
        let evals_11 = expand_elem_4_to_11(&evals_4);
        out.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(evals_11)));
    }
    out
}

#[inline]
fn eval_mle_lsb_first_in_place(mut evals: Vec<Fq>, r: &[Fq]) -> Fq {
    debug_assert_eq!(evals.len(), 1usize << r.len());
    let mut len = evals.len();
    for r_i in r {
        let half = len / 2;
        let one_minus = Fq::one() - *r_i;
        for j in 0..half {
            let a = evals[2 * j];
            let b = evals[2 * j + 1];
            evals[j] = a * one_minus + b * *r_i;
        }
        len = half;
    }
    evals[0]
}

fn compute_shared_scalars(eval_point: &[Fq]) -> SharedScalars<Fq> {
    debug_assert_eq!(eval_point.len(), NUM_VARS);
    let r_elem = &eval_point[STEP_VARS..];
    debug_assert_eq!(r_elem.len(), ELEM_VARS);

    let g = eval_mle_lsb_first_in_place(get_g_mle(), r_elem);

    let basis0_c0 = fq12_sparse_034(Fq2::new(Fq::one(), Fq::zero()), Fq2::zero(), Fq2::zero());
    let basis0_c1 = fq12_sparse_034(Fq2::new(Fq::zero(), Fq::one()), Fq2::zero(), Fq2::zero());
    let basis3_c0 = fq12_sparse_034(Fq2::zero(), Fq2::new(Fq::one(), Fq::zero()), Fq2::zero());
    let basis3_c1 = fq12_sparse_034(Fq2::zero(), Fq2::new(Fq::zero(), Fq::one()), Fq2::zero());
    let basis4_c0 = fq12_sparse_034(Fq2::zero(), Fq2::zero(), Fq2::new(Fq::one(), Fq::zero()));
    let basis4_c1 = fq12_sparse_034(Fq2::zero(), Fq2::zero(), Fq2::new(Fq::zero(), Fq::one()));

    let sel0 = eval_mle_lsb_first_in_place(fq12_to_multilinear_evals(&basis0_c0), r_elem);
    let sel1 = eval_mle_lsb_first_in_place(fq12_to_multilinear_evals(&basis0_c1), r_elem);
    let sel2 = eval_mle_lsb_first_in_place(fq12_to_multilinear_evals(&basis3_c0), r_elem);
    let sel3 = eval_mle_lsb_first_in_place(fq12_to_multilinear_evals(&basis3_c1), r_elem);
    let sel4 = eval_mle_lsb_first_in_place(fq12_to_multilinear_evals(&basis4_c0), r_elem);
    let sel5 = eval_mle_lsb_first_in_place(fq12_to_multilinear_evals(&basis4_c1), r_elem);

    SharedScalars {
        g,
        sel0,
        sel1,
        sel2,
        sel3,
        sel4,
        sel5,
    }
}

// =============================================================================
// Values + constraint logic
// =============================================================================

// ============================================================================
// Logic Implementation
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct MultiMillerLoopValues<F: JoltField> {
    pub f: F,
    pub f_next: F,
    pub quotient: F,
    pub t_x_c0: F,
    pub t_x_c1: F,
    pub t_y_c0: F,
    pub t_y_c1: F,
    pub t_x_c0_next: F,
    pub t_x_c1_next: F,
    pub t_y_c0_next: F,
    pub t_y_c1_next: F,
    pub lambda_c0: F,
    pub lambda_c1: F,
    pub inv_delta_x_c0: F,
    pub inv_delta_x_c1: F,
    pub x_p: F,
    pub y_p: F,
    pub x_q_c0: F,
    pub x_q_c1: F,
    pub y_q_c0: F,
    pub y_q_c1: F,
    pub is_double: F,
    pub is_add: F,
    pub l_val: F,
}

impl<F: JoltField> MultiMillerLoopValues<F> {
    /// Evaluate the batched Multi-Miller loop constraint polynomial at this point.
    ///
    /// Uses `delta` to batch the constraint terms.
    ///
    /// Constraints:
    /// 1. G2 Arithmetic (Affine)
    /// 2. Line Evaluation (Ell-Coefficients)
    /// 3. Accumulator Update (Ring-Switching)
    pub fn eval_constraint(&self, delta: F, shared: SharedScalars<F>) -> F {
        let one = F::one();
        let two = F::from_u64(2);
        let three = F::from_u64(3);

        // Fq2 helper functions (inline)
        // mul: (a0,a1) * (b0,b1) = (a0*b0 - a1*b1, a0*b1 + a1*b0)
        let mul_c0 = |a0: F, a1: F, b0: F, b1: F| a0 * b0 - a1 * b1;
        // sq: (a0,a1)^2 = (a0^2 - a1^2, 2*a0*a1)
        let sq_c0 = |a0: F, a1: F| a0 * a0 - a1 * a1;
        let sq_c1 = |a0: F, a1: F| two * a0 * a1;

        let mut acc = F::zero();
        let mut delta_pow = F::one();

        // 1. Booleanity of selectors
        acc += delta_pow * (self.is_double * (one - self.is_double));
        delta_pow *= delta;
        acc += delta_pow * (self.is_add * (one - self.is_add));
        delta_pow *= delta;
        // Mutually exclusive
        acc += delta_pow * (self.is_double * self.is_add);
        delta_pow *= delta;

        let is_active = self.is_double + self.is_add; // 1 if active step

        // 2. G2 Arithmetic
        // Current point T
        let tx0 = self.t_x_c0;
        let tx1 = self.t_x_c1;
        let ty0 = self.t_y_c0;
        let ty1 = self.t_y_c1;

        // Next point T_next
        let tx_next0 = self.t_x_c0_next;
        let tx_next1 = self.t_x_c1_next;
        let ty_next0 = self.t_y_c0_next;
        let ty_next1 = self.t_y_c1_next;

        // Operand point (T for double, Q for add)
        let op_x0 = self.is_double * tx0 + self.is_add * self.x_q_c0;
        let op_x1 = self.is_double * tx1 + self.is_add * self.x_q_c1;
        let _op_y0 = self.is_double * ty0 + self.is_add * self.y_q_c0;
        let _op_y1 = self.is_double * ty1 + self.is_add * self.y_q_c1;

        // Slope constraints
        // Double case: 2 * y * lambda = 3 * x^2
        let two_y_lam0 = two * mul_c0(ty0, ty1, self.lambda_c0, self.lambda_c1);
        let two_y_lam1 = two * (ty0 * self.lambda_c1 + ty1 * self.lambda_c0);
        let three_x_sq0 = three * sq_c0(tx0, tx1);
        let three_x_sq1 = three * sq_c1(tx0, tx1);

        acc += delta_pow * (self.is_double * (two_y_lam0 - three_x_sq0));
        delta_pow *= delta;
        acc += delta_pow * (self.is_double * (two_y_lam1 - three_x_sq1));
        delta_pow *= delta;

        // Add case: lambda * (x_q - x) = y_q - y
        let dx0 = self.x_q_c0 - tx0;
        let dx1 = self.x_q_c1 - tx1;
        let dy0 = self.y_q_c0 - ty0;
        let dy1 = self.y_q_c1 - ty1;

        let lam_dx0 = mul_c0(self.lambda_c0, self.lambda_c1, dx0, dx1);
        let lam_dx1 = self.lambda_c0 * dx1 + self.lambda_c1 * dx0;

        acc += delta_pow * (self.is_add * (lam_dx0 - dy0));
        delta_pow *= delta;
        acc += delta_pow * (self.is_add * (lam_dx1 - dy1));
        delta_pow *= delta;

        // Inverse constraint for add case: inv_dx * dx = 1
        let inv_dx_dx0 = mul_c0(self.inv_delta_x_c0, self.inv_delta_x_c1, dx0, dx1);
        let inv_dx_dx1 = self.inv_delta_x_c0 * dx1 + self.inv_delta_x_c1 * dx0;

        acc += delta_pow * (self.is_add * (inv_dx_dx0 - one));
        delta_pow *= delta;
        acc += delta_pow * (self.is_add * inv_dx_dx1);
        delta_pow *= delta;

        // Point update constraints
        // x_next = lambda^2 - x - x_op
        let lam_sq0 = sq_c0(self.lambda_c0, self.lambda_c1);
        let lam_sq1 = sq_c1(self.lambda_c0, self.lambda_c1);

        acc += delta_pow * (is_active * (tx_next0 - (lam_sq0 - tx0 - op_x0)));
        delta_pow *= delta;
        acc += delta_pow * (is_active * (tx_next1 - (lam_sq1 - tx1 - op_x1)));
        delta_pow *= delta;

        // y_next = lambda * (x - x_next) - y
        let dx_next0 = tx0 - tx_next0;
        let dx_next1 = tx1 - tx_next1;
        let lam_dx_next0 = mul_c0(self.lambda_c0, self.lambda_c1, dx_next0, dx_next1);
        let lam_dx_next1 = self.lambda_c0 * dx_next1 + self.lambda_c1 * dx_next0;

        acc += delta_pow * (is_active * (ty_next0 - (lam_dx_next0 - ty0)));
        delta_pow *= delta;
        acc += delta_pow * (is_active * (ty_next1 - (lam_dx_next1 - ty1)));
        delta_pow *= delta;

        // 3. Line Evaluation Coefficients (BN254 / TwistType::D convention)
        //
        // We use unscaled line coefficients (c0,c1,c2) ∈ Fq2 for the line:
        //   c0 * y + c1 * x + c2 = 0
        // then the pairing line contribution is embedded via:
        //   c0' = c0 * y_P,  c1' = c1 * x_P,  c2' = c2
        // and mapped into the Fq12 MLE using the selector polynomials.
        //
        // Double (tangent) at T:
        //   c0 = -2y_T
        //   c1 = 3x_T^2
        //   c2 = 2y_T^2 - 3x_T^3
        //
        // Add (chord) through T and Q:
        //   c0 = x_T - x_Q
        //   c1 = y_Q - y_T
        //   c2 = x_Q*y_T - x_T*y_Q

        // --- Double coefficients ---
        let dbl_c0_0 = -(two * ty0);
        let dbl_c0_1 = -(two * ty1);
        let x_sq0 = sq_c0(tx0, tx1);
        let x_sq1 = sq_c1(tx0, tx1);
        let dbl_c1_0 = three * x_sq0;
        let dbl_c1_1 = three * x_sq1;
        let y_sq0 = sq_c0(ty0, ty1);
        let y_sq1 = sq_c1(ty0, ty1);
        let x_cub0 = mul_c0(x_sq0, x_sq1, tx0, tx1);
        let x_cub1 = x_sq0 * tx1 + x_sq1 * tx0;
        let dbl_c2_0 = two * y_sq0 - three * x_cub0;
        let dbl_c2_1 = two * y_sq1 - three * x_cub1;

        // --- Add coefficients ---
        let add_c0_0 = tx0 - self.x_q_c0;
        let add_c0_1 = tx1 - self.x_q_c1;
        let add_c1_0 = self.y_q_c0 - ty0;
        let add_c1_1 = self.y_q_c1 - ty1;
        let xq_yt_0 = mul_c0(self.x_q_c0, self.x_q_c1, ty0, ty1);
        let xq_yt_1 = self.x_q_c0 * ty1 + self.x_q_c1 * ty0;
        let xt_yq_0 = mul_c0(tx0, tx1, self.y_q_c0, self.y_q_c1);
        let xt_yq_1 = tx0 * self.y_q_c1 + tx1 * self.y_q_c0;
        let add_c2_0 = xq_yt_0 - xt_yq_0;
        let add_c2_1 = xq_yt_1 - xt_yq_1;

        // Select coefficients based on branch.
        let c0_0 = self.is_double * dbl_c0_0 + self.is_add * add_c0_0;
        let c0_1 = self.is_double * dbl_c0_1 + self.is_add * add_c0_1;
        let c1_0 = self.is_double * dbl_c1_0 + self.is_add * add_c1_0;
        let c1_1 = self.is_double * dbl_c1_1 + self.is_add * add_c1_1;
        let c2_0 = self.is_double * dbl_c2_0 + self.is_add * add_c2_0;
        let c2_1 = self.is_double * dbl_c2_1 + self.is_add * add_c2_1;

        // 4. Line Evaluation Value (sparse 034 embedding)
        //
        // We embed the Fq2 coefficients into an Fq12 element with nonzero coefficients
        // at positions (0,3,4) and then use selector polynomials to evaluate its MLE at x:
        //   coeff0 = c0 * y_P   (Fq2)
        //   coeff3 = c1 * x_P   (Fq2)
        //   coeff4 = c2         (Fq2)
        let coeff0_c0 = c0_0 * self.y_p;
        let coeff0_c1 = c0_1 * self.y_p;
        let coeff3_c0 = c1_0 * self.x_p;
        let coeff3_c1 = c1_1 * self.x_p;
        let coeff4_c0 = c2_0;
        let coeff4_c1 = c2_1;

        let calc_l_val = shared.sel0 * coeff0_c0
            + shared.sel1 * coeff0_c1
            + shared.sel2 * coeff3_c0
            + shared.sel3 * coeff3_c1
            + shared.sel4 * coeff4_c0
            + shared.sel5 * coeff4_c1;

        acc += delta_pow * (is_active * (self.l_val - calc_l_val));
        delta_pow *= delta;

        // 5. Accumulator Update
        // f_next = f^2 * l_val (if double)
        // f_next = f * l_val (if add)
        // Ring switching: A * B - C - Q * g = 0

        let a = self.is_double * self.f * self.f + self.is_add * self.f;
        let b = self.l_val;
        let c = self.f_next;

        acc += delta_pow * (is_active * (a * b - c - self.quotient * shared.g));
        // delta_pow *= delta; // Last term

        acc
    }
}

// =============================================================================
// Prover / Verifier specs (ConstraintList)
// =============================================================================

#[derive(Clone, Allocative)]
pub struct MultiMillerLoopProverSpec<F: JoltField + Allocative> {
    params: MultiMillerLoopParams,
    polys_by_kind: Vec<Vec<MultilinearPolynomial<F>>>,
    shared_polys: Vec<MultilinearPolynomial<F>>,
}

impl MultiMillerLoopProverSpec<Fq> {
    pub fn new(
        params: MultiMillerLoopParams,
        constraint_polys: Vec<MultiMillerLoopWitness<Fq>>,
    ) -> (Self, Vec<usize>) {
        debug_assert_eq!(constraint_polys.len(), params.num_constraints);

        let num_instances = constraint_polys.len();
        let mut polys_by_kind: Vec<Vec<MultilinearPolynomial<Fq>>> =
            (0..NUM_COMMITTED_KINDS).map(|_| Vec::with_capacity(num_instances)).collect();
        let mut constraint_indices = Vec::with_capacity(num_instances);

        for poly in constraint_polys {
            constraint_indices.push(poly.constraint_index);
            let mut k = 0usize;
            let push = |dst: &mut Vec<Vec<MultilinearPolynomial<Fq>>>,
                        k: &mut usize,
                        v: Vec<Fq>| {
                dst[*k].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(v)));
                *k += 1;
            };

            push(&mut polys_by_kind, &mut k, poly.f);
            push(&mut polys_by_kind, &mut k, poly.f_next);
            push(&mut polys_by_kind, &mut k, poly.quotient);
            push(&mut polys_by_kind, &mut k, poly.t_x_c0);
            push(&mut polys_by_kind, &mut k, poly.t_x_c1);
            push(&mut polys_by_kind, &mut k, poly.t_y_c0);
            push(&mut polys_by_kind, &mut k, poly.t_y_c1);
            push(&mut polys_by_kind, &mut k, poly.t_x_c0_next);
            push(&mut polys_by_kind, &mut k, poly.t_x_c1_next);
            push(&mut polys_by_kind, &mut k, poly.t_y_c0_next);
            push(&mut polys_by_kind, &mut k, poly.t_y_c1_next);
            push(&mut polys_by_kind, &mut k, poly.lambda_c0);
            push(&mut polys_by_kind, &mut k, poly.lambda_c1);
            push(&mut polys_by_kind, &mut k, poly.inv_delta_x_c0);
            push(&mut polys_by_kind, &mut k, poly.inv_delta_x_c1);
            push(&mut polys_by_kind, &mut k, poly.x_p);
            push(&mut polys_by_kind, &mut k, poly.y_p);
            push(&mut polys_by_kind, &mut k, poly.x_q_c0);
            push(&mut polys_by_kind, &mut k, poly.x_q_c1);
            push(&mut polys_by_kind, &mut k, poly.y_q_c0);
            push(&mut polys_by_kind, &mut k, poly.y_q_c1);
            push(&mut polys_by_kind, &mut k, poly.is_double);
            push(&mut polys_by_kind, &mut k, poly.is_add);
            push(&mut polys_by_kind, &mut k, poly.l_val);

            debug_assert_eq!(k, NUM_COMMITTED_KINDS);
        }

        let shared_polys = build_shared_polys_11();
        debug_assert_eq!(shared_polys.len(), 7);

        (
            Self {
                params,
                polys_by_kind,
                shared_polys,
            },
            constraint_indices,
        )
    }
}

impl<F: JoltField + Allocative> ConstraintListSpec for MultiMillerLoopProverSpec<F> {
    fn sumcheck_id(&self) -> SumcheckId {
        self.params.sumcheck_id
    }
    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }
    fn num_instances(&self) -> usize {
        self.params.num_constraints
    }
    fn uses_term_batching(&self) -> bool {
        true
    }
    fn opening_specs(&self) -> &'static [OpeningSpec] {
        &OPENING_SPECS
    }
    fn build_virtual_poly(&self, term_index: usize, instance: usize) -> VirtualPolynomial {
        VirtualPolynomial::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::from_index(term_index).expect("invalid term index"),
            instance,
        })
    }
}

impl ConstraintListProverSpec<Fq, DEGREE> for MultiMillerLoopProverSpec<Fq> {
    fn polys_by_kind(&self) -> &[Vec<MultilinearPolynomial<Fq>>] {
        &self.polys_by_kind
    }
    fn polys_by_kind_mut(&mut self) -> &mut [Vec<MultilinearPolynomial<Fq>>] {
        &mut self.polys_by_kind
    }

    fn shared_polys(&self) -> &[MultilinearPolynomial<Fq>] {
        &self.shared_polys
    }
    fn shared_polys_mut(&mut self) -> &mut [MultilinearPolynomial<Fq>] {
        &mut self.shared_polys
    }

    fn eval_constraint(
        &self,
        _instance: usize,
        eval_index: usize,
        poly_evals: &[[Fq; DEGREE]],
        _public_evals: &[[Fq; DEGREE]],
        shared_evals: &[[Fq; DEGREE]],
        term_batch_coeff: Option<Fq>,
    ) -> Fq {
        let delta = term_batch_coeff.expect("MultiMillerLoop requires term_batch_coeff");
        debug_assert_eq!(shared_evals.len(), 7);

        let vals = MultiMillerLoopValues::<Fq> {
            f: poly_evals[0][eval_index],
            f_next: poly_evals[1][eval_index],
            quotient: poly_evals[2][eval_index],
            t_x_c0: poly_evals[3][eval_index],
            t_x_c1: poly_evals[4][eval_index],
            t_y_c0: poly_evals[5][eval_index],
            t_y_c1: poly_evals[6][eval_index],
            t_x_c0_next: poly_evals[7][eval_index],
            t_x_c1_next: poly_evals[8][eval_index],
            t_y_c0_next: poly_evals[9][eval_index],
            t_y_c1_next: poly_evals[10][eval_index],
            lambda_c0: poly_evals[11][eval_index],
            lambda_c1: poly_evals[12][eval_index],
            inv_delta_x_c0: poly_evals[13][eval_index],
            inv_delta_x_c1: poly_evals[14][eval_index],
            x_p: poly_evals[15][eval_index],
            y_p: poly_evals[16][eval_index],
            x_q_c0: poly_evals[17][eval_index],
            x_q_c1: poly_evals[18][eval_index],
            y_q_c0: poly_evals[19][eval_index],
            y_q_c1: poly_evals[20][eval_index],
            is_double: poly_evals[21][eval_index],
            is_add: poly_evals[22][eval_index],
            l_val: poly_evals[23][eval_index],
        };

        let shared = SharedScalars::<Fq> {
            g: shared_evals[0][eval_index],
            sel0: shared_evals[1][eval_index],
            sel1: shared_evals[2][eval_index],
            sel2: shared_evals[3][eval_index],
            sel3: shared_evals[4][eval_index],
            sel4: shared_evals[5][eval_index],
            sel5: shared_evals[6][eval_index],
        };

        vals.eval_constraint(delta, shared)
    }
}

#[derive(Clone, Allocative)]
pub struct MultiMillerLoopVerifierSpec<F: JoltField + Allocative> {
    params: MultiMillerLoopParams,
    _marker: core::marker::PhantomData<F>,
}

impl MultiMillerLoopVerifierSpec<Fq> {
    pub fn new(params: MultiMillerLoopParams) -> Self {
        Self {
            params,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F: JoltField + Allocative> ConstraintListSpec for MultiMillerLoopVerifierSpec<F> {
    fn sumcheck_id(&self) -> SumcheckId {
        self.params.sumcheck_id
    }
    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }
    fn num_instances(&self) -> usize {
        self.params.num_constraints
    }
    fn uses_term_batching(&self) -> bool {
        true
    }
    fn opening_specs(&self) -> &'static [OpeningSpec] {
        &OPENING_SPECS
    }
    fn build_virtual_poly(&self, term_index: usize, instance: usize) -> VirtualPolynomial {
        VirtualPolynomial::Recursion(RecursionPoly::MultiMillerLoop {
            term: MultiMillerLoopTerm::from_index(term_index).expect("invalid term index"),
            instance,
        })
    }
}

impl ConstraintListVerifierSpec<Fq, DEGREE> for MultiMillerLoopVerifierSpec<Fq> {
    fn compute_shared_scalars(&self, eval_point: &[Fq]) -> Vec<Fq> {
        let s = compute_shared_scalars(eval_point);
        vec![s.g, s.sel0, s.sel1, s.sel2, s.sel3, s.sel4, s.sel5]
    }

    fn eval_constraint_at_point(
        &self,
        _instance: usize,
        opened_claims: &[Fq],
        shared_scalars: &[Fq],
        _eval_point: &[Fq],
        term_batch_coeff: Option<Fq>,
    ) -> Fq {
        let delta = term_batch_coeff.expect("MultiMillerLoop requires term_batch_coeff");
        debug_assert_eq!(opened_claims.len(), NUM_COMMITTED_KINDS);
        debug_assert_eq!(shared_scalars.len(), 7);

        let vals = MultiMillerLoopValues::<Fq> {
            f: opened_claims[0],
            f_next: opened_claims[1],
            quotient: opened_claims[2],
            t_x_c0: opened_claims[3],
            t_x_c1: opened_claims[4],
            t_y_c0: opened_claims[5],
            t_y_c1: opened_claims[6],
            t_x_c0_next: opened_claims[7],
            t_x_c1_next: opened_claims[8],
            t_y_c0_next: opened_claims[9],
            t_y_c1_next: opened_claims[10],
            lambda_c0: opened_claims[11],
            lambda_c1: opened_claims[12],
            inv_delta_x_c0: opened_claims[13],
            inv_delta_x_c1: opened_claims[14],
            x_p: opened_claims[15],
            y_p: opened_claims[16],
            x_q_c0: opened_claims[17],
            x_q_c1: opened_claims[18],
            y_q_c0: opened_claims[19],
            y_q_c1: opened_claims[20],
            is_double: opened_claims[21],
            is_add: opened_claims[22],
            l_val: opened_claims[23],
        };

        let shared = SharedScalars::<Fq> {
            g: shared_scalars[0],
            sel0: shared_scalars[1],
            sel1: shared_scalars[2],
            sel2: shared_scalars[3],
            sel3: shared_scalars[4],
            sel4: shared_scalars[5],
            sel5: shared_scalars[6],
        };

        vals.eval_constraint(delta, shared)
    }
}

pub type MultiMillerLoopProver<F> = ConstraintListProver<F, MultiMillerLoopProverSpec<F>, DEGREE>;
pub type MultiMillerLoopVerifier<F> =
    ConstraintListVerifier<F, MultiMillerLoopVerifierSpec<F>, DEGREE>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::commitment::dory::witness::multi_miller_loop::MultiMillerLoopSteps;
    use ark_bn254::{Fq, G1Affine, G2Affine};
    use ark_ec::AffineRepr;
    use ark_ff::Zero;

    #[test]
    fn test_constraint_zero_on_boolean_hypercube_single_pair() {
        let p = G1Affine::generator();
        let q = G2Affine::generator();
        let steps = MultiMillerLoopSteps::new(&[p], &[q]);

        let pair = 0usize;
        let f = &steps.f_packed_mles[pair];
        let f_next = &steps.f_next_packed_mles[pair];
        let quotient = &steps.quotient_packed_mles[pair];
        let t_x_c0 = &steps.t_x_c0_packed_mles[pair];
        let t_x_c1 = &steps.t_x_c1_packed_mles[pair];
        let t_y_c0 = &steps.t_y_c0_packed_mles[pair];
        let t_y_c1 = &steps.t_y_c1_packed_mles[pair];
        let t_x_c0_next = &steps.t_x_c0_next_packed_mles[pair];
        let t_x_c1_next = &steps.t_x_c1_next_packed_mles[pair];
        let t_y_c0_next = &steps.t_y_c0_next_packed_mles[pair];
        let t_y_c1_next = &steps.t_y_c1_next_packed_mles[pair];
        let lambda_c0 = &steps.lambda_c0_packed_mles[pair];
        let lambda_c1 = &steps.lambda_c1_packed_mles[pair];
        let inv_dx_c0 = &steps.inv_dx_c0_packed_mles[pair];
        let inv_dx_c1 = &steps.inv_dx_c1_packed_mles[pair];
        let x_p = &steps.x_p_packed_mles[pair];
        let y_p = &steps.y_p_packed_mles[pair];
        let x_q_c0 = &steps.x_q_c0_packed_mles[pair];
        let x_q_c1 = &steps.x_q_c1_packed_mles[pair];
        let y_q_c0 = &steps.y_q_c0_packed_mles[pair];
        let y_q_c1 = &steps.y_q_c1_packed_mles[pair];
        let is_double = &steps.is_double_packed_mles[pair];
        let is_add = &steps.is_add_packed_mles[pair];
        let l_val = &steps.l_val_packed_mles[pair];
        let shared_polys = build_shared_polys_11();
        fn dense<'a>(p: &'a MultilinearPolynomial<Fq>) -> &'a [Fq] {
            match p {
                MultilinearPolynomial::LargeScalars(dp) => &dp.Z,
                _ => panic!("expected dense polynomial for shared MultiMillerLoop constants"),
            }
        }
        let g = dense(&shared_polys[0]);
        let selector_0 = dense(&shared_polys[1]);
        let selector_1 = dense(&shared_polys[2]);
        let selector_2 = dense(&shared_polys[3]);
        let selector_3 = dense(&shared_polys[4]);
        let selector_4 = dense(&shared_polys[5]);
        let selector_5 = dense(&shared_polys[6]);
        let delta1 = Fq::from(7u64);
        let delta2 = Fq::from(13u64);

        let step_size = 1usize << 7; // 128
        let elem_size = 1usize << 4; // 16

        for s in 0..step_size {
            for x in 0..elem_size {
                let idx = x * step_size + s;

                let vals = MultiMillerLoopValues::<Fq> {
                    f: f[idx],
                    f_next: f_next[idx],
                    quotient: quotient[idx],
                    t_x_c0: t_x_c0[idx],
                    t_x_c1: t_x_c1[idx],
                    t_y_c0: t_y_c0[idx],
                    t_y_c1: t_y_c1[idx],
                    t_x_c0_next: t_x_c0_next[idx],
                    t_x_c1_next: t_x_c1_next[idx],
                    t_y_c0_next: t_y_c0_next[idx],
                    t_y_c1_next: t_y_c1_next[idx],
                    lambda_c0: lambda_c0[idx],
                    lambda_c1: lambda_c1[idx],
                    inv_delta_x_c0: inv_dx_c0[idx],
                    inv_delta_x_c1: inv_dx_c1[idx],
                    x_p: x_p[idx],
                    y_p: y_p[idx],
                    x_q_c0: x_q_c0[idx],
                    x_q_c1: x_q_c1[idx],
                    y_q_c0: y_q_c0[idx],
                    y_q_c1: y_q_c1[idx],
                    is_double: is_double[idx],
                    is_add: is_add[idx],
                    l_val: l_val[idx],
                };

                let shared = SharedScalars::<Fq> {
                    g: g[idx],
                    sel0: selector_0[idx],
                    sel1: selector_1[idx],
                    sel2: selector_2[idx],
                    sel3: selector_3[idx],
                    sel4: selector_4[idx],
                    sel5: selector_5[idx],
                };

                let c1 = vals.eval_constraint(delta1, shared);
                let c2 = vals.eval_constraint(delta2, shared);
                if !c1.is_zero() || !c2.is_zero() {
                    panic!(
                        "constraint nonzero at (s={s}, x={x}): c1={c1:?}, c2={c2:?}, is_double={}, is_add={}",
                        vals.is_double, vals.is_add
                    );
                }
            }
        }
    }
}
