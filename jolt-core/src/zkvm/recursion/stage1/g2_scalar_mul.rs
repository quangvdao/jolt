//! G2 scalar multiplication sumcheck for proving G2 scalar multiplication constraints
//!
//! Analogue of `g1_scalar_mul.rs`, but for points in G2 (over Fq2). Since the recursion
//! SNARK runs over the base field Fq, we "relinearize" Fq2 elements by splitting each
//! coordinate into its (c0, c1) components in Fq and enforcing constraints component-wise.
//!
//! Double-and-add trace (256 steps, MSB-first):
//! - A_0 = O
//! - T_i = [2]A_i
//! - A_{i+1} = T_i + b_i·P
//!
//! Constraints are the standard short-Weierstrass affine laws (denominator-free), applied in Fq2:
//! - C1/C2: doubling formulas
//! - C3/C4: conditional add formulas (bit-dependent), with a special case when T = O
//! - C5: if A = O then T = O (infinity preserved)
//! - C6: if ind_T = 1 then (x_T, y_T) = (0,0) in Fq2 (implemented as 4 Fq constraints)

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
use ark_bn254::{Fq, Fq2, Fr};
use ark_ff::{BigInteger, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

/// Public inputs for a single G2 scalar multiplication (the scalar).
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G2ScalarMulPublicInputs {
    pub scalar: Fr,
}

/// Witness polynomials for a G2 scalar multiplication constraint.
///
/// Represents the double-and-add trace with 256 steps (MSB-first) over Fq2.
/// Since the recursion SNARK runs over Fq, each Fq2 element is split into
/// its (c0, c1) components.
///
/// - `A_i`: accumulator point at step `i` (in G2)
/// - `T_i = [2]A_i`: doubled point at step `i`
/// - `A_{i+1} = T_i + b_i·P`: next accumulator
#[derive(Clone, Debug)]
pub struct G2ScalarMulWitness {
    /// Index of this constraint in the constraint system
    pub constraint_index: usize,
    /// Base point P = (x, y) in G2 being multiplied (Fq2 coordinates)
    pub base_point: (Fq2, Fq2),
    /// Accumulator x-coordinate c0 component: x_A.c0(s)
    pub x_a_c0: Vec<Fq>,
    /// Accumulator x-coordinate c1 component: x_A.c1(s)
    pub x_a_c1: Vec<Fq>,
    /// Accumulator y-coordinate c0 component: y_A.c0(s)
    pub y_a_c0: Vec<Fq>,
    /// Accumulator y-coordinate c1 component: y_A.c1(s)
    pub y_a_c1: Vec<Fq>,
    /// Doubled point x-coordinate c0 component: x_T.c0(s)
    pub x_t_c0: Vec<Fq>,
    /// Doubled point x-coordinate c1 component: x_T.c1(s)
    pub x_t_c1: Vec<Fq>,
    /// Doubled point y-coordinate c0 component: y_T.c0(s)
    pub y_t_c0: Vec<Fq>,
    /// Doubled point y-coordinate c1 component: y_T.c1(s)
    pub y_t_c1: Vec<Fq>,
    /// Next accumulator x-coordinate c0 component: x_A_next.c0(s)
    pub x_a_next_c0: Vec<Fq>,
    /// Next accumulator x-coordinate c1 component: x_A_next.c1(s)
    pub x_a_next_c1: Vec<Fq>,
    /// Next accumulator y-coordinate c0 component: y_A_next.c0(s)
    pub y_a_next_c0: Vec<Fq>,
    /// Next accumulator y-coordinate c1 component: y_A_next.c1(s)
    pub y_a_next_c1: Vec<Fq>,
    /// Indicator for T being at infinity: 1 if T_s = O, else 0
    pub t_indicator: Vec<Fq>,
    /// Indicator for A being at infinity: 1 if A_s = O, else 0
    pub a_indicator: Vec<Fq>,
}

impl G2ScalarMulPublicInputs {
    pub fn new(scalar: Fr) -> Self {
        Self { scalar }
    }

    /// Scalar bits MSB-first, length 256 (matches witness generation).
    pub fn bits_msb(&self) -> Vec<bool> {
        let scalar_bits_le = self.scalar.into_bigint().to_bits_le();
        (0..256).rev().map(|i| scalar_bits_le[i]).collect()
    }

    /// Evaluate the (padded) bit MLE at the sumcheck challenge point r* (11 vars).
    ///
    /// Padding convention matches `DoryMatrixBuilder::pad_8var_to_11var_zero_padding`:
    /// only the first 256 entries are populated (bits), remaining 2048-256 are 0.
    pub fn evaluate_bit_mle<F: JoltField>(&self, r_star: &[F]) -> F {
        assert_eq!(r_star.len(), 11);
        let bits = self.bits_msb();

        // First 3 variables select the "prefix = 0" block (since bits live in indices 0..256).
        let pad_factor = EqPolynomial::<F>::zero_selector(&r_star[..3]);

        // Remaining 8 variables index the 256 step positions.
        let eq_step = EqPolynomial::<F>::evals(&r_star[3..]);
        debug_assert_eq!(eq_step.len(), 256);

        let mut acc = F::zero();
        for (i, eq) in eq_step.iter().enumerate() {
            if bits[i] {
                acc += *eq;
            }
        }

        pad_factor * acc
    }
}

// =============================================================================
// VIRTUAL CLAIM HELPERS
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn append_g2_scalar_mul_virtual_claims<F: JoltField, T: Transcript>(
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
    x_a_c0_claim: F,
    x_a_c1_claim: F,
    y_a_c0_claim: F,
    y_a_c1_claim: F,
    x_t_c0_claim: F,
    x_t_c1_claim: F,
    y_t_c0_claim: F,
    y_t_c1_claim: F,
    x_a_next_c0_claim: F,
    x_a_next_c1_claim: F,
    y_a_next_c0_claim: F,
    y_a_next_c1_claim: F,
    t_is_infinity_claim: F,
    a_is_infinity_claim: F,
) {
    let claims = virtual_claims![
        VirtualPolynomial::RecursionG2ScalarMulXAC0(constraint_idx) => x_a_c0_claim,
        VirtualPolynomial::RecursionG2ScalarMulXAC1(constraint_idx) => x_a_c1_claim,
        VirtualPolynomial::RecursionG2ScalarMulYAC0(constraint_idx) => y_a_c0_claim,
        VirtualPolynomial::RecursionG2ScalarMulYAC1(constraint_idx) => y_a_c1_claim,
        VirtualPolynomial::RecursionG2ScalarMulXTC0(constraint_idx) => x_t_c0_claim,
        VirtualPolynomial::RecursionG2ScalarMulXTC1(constraint_idx) => x_t_c1_claim,
        VirtualPolynomial::RecursionG2ScalarMulYTC0(constraint_idx) => y_t_c0_claim,
        VirtualPolynomial::RecursionG2ScalarMulYTC1(constraint_idx) => y_t_c1_claim,
        VirtualPolynomial::RecursionG2ScalarMulXANextC0(constraint_idx) => x_a_next_c0_claim,
        VirtualPolynomial::RecursionG2ScalarMulXANextC1(constraint_idx) => x_a_next_c1_claim,
        VirtualPolynomial::RecursionG2ScalarMulYANextC0(constraint_idx) => y_a_next_c0_claim,
        VirtualPolynomial::RecursionG2ScalarMulYANextC1(constraint_idx) => y_a_next_c1_claim,
        VirtualPolynomial::RecursionG2ScalarMulTIndicator(constraint_idx) => t_is_infinity_claim,
        VirtualPolynomial::RecursionG2ScalarMulAIndicator(constraint_idx) => a_is_infinity_claim,
    ];
    append_virtual_claims(accumulator, transcript, sumcheck_id, opening_point, &claims);
}

#[allow(clippy::type_complexity)]
fn get_g2_scalar_mul_virtual_claims<F: JoltField>(
    accumulator: &VerifierOpeningAccumulator<F>,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
) -> (F, F, F, F, F, F, F, F, F, F, F, F, F, F) {
    let polynomials = vec![
        VirtualPolynomial::RecursionG2ScalarMulXAC0(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulXAC1(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulYAC0(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulYAC1(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulXTC0(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulXTC1(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulYTC0(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulYTC1(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulXANextC0(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulXANextC1(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulYANextC0(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulYANextC1(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulTIndicator(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulAIndicator(constraint_idx),
    ];
    let claims = get_virtual_claims(accumulator, sumcheck_id, &polynomials);
    (
        claims[0], claims[1], claims[2], claims[3], claims[4], claims[5], claims[6], claims[7],
        claims[8], claims[9], claims[10], claims[11], claims[12], claims[13],
    )
}

fn append_g2_scalar_mul_virtual_openings<F: JoltField, T: Transcript>(
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
) {
    let polynomials = vec![
        VirtualPolynomial::RecursionG2ScalarMulXAC0(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulXAC1(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulYAC0(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulYAC1(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulXTC0(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulXTC1(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulYTC0(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulYTC1(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulXANextC0(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulXANextC1(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulYANextC0(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulYANextC1(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulTIndicator(constraint_idx),
        VirtualPolynomial::RecursionG2ScalarMulAIndicator(constraint_idx),
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
// CONSTRAINT FUNCTIONS (IN Fq2)
// =============================================================================

fn fq2_from_fq(x: Fq) -> Fq2 {
    Fq2::new(x, Fq::zero())
}

/// C1: Doubling x-coordinate constraint over Fq2
fn compute_c1(x_a: Fq2, y_a: Fq2, x_t: Fq2) -> Fq2 {
    let four = fq2_from_fq(Fq::from(4u64));
    let two = fq2_from_fq(Fq::from(2u64));
    let nine = fq2_from_fq(Fq::from(9u64));

    let y_a_sq = y_a * y_a;
    let x_a_sq = x_a * x_a;
    let x_a_fourth = x_a_sq * x_a_sq;

    four * y_a_sq * (x_t + two * x_a) - nine * x_a_fourth
}

/// C2: Doubling y-coordinate constraint over Fq2
fn compute_c2(x_a: Fq2, y_a: Fq2, x_t: Fq2, y_t: Fq2) -> Fq2 {
    let three = fq2_from_fq(Fq::from(3u64));
    let two = fq2_from_fq(Fq::from(2u64));

    let x_a_sq = x_a * x_a;
    three * x_a_sq * (x_t - x_a) + two * y_a * (y_t + y_a)
}

/// C3: Conditional addition x-coordinate constraint over Fq2 (bit-dependent)
#[allow(clippy::too_many_arguments)]
fn compute_c3(bit: Fq, ind_t: Fq, x_a_next: Fq2, x_t: Fq2, y_t: Fq2, x_p: Fq2, y_p: Fq2) -> Fq2 {
    let one = Fq2::one();
    let bit2 = fq2_from_fq(bit);
    let ind_t2 = fq2_from_fq(ind_t);

    let c3_skip = (one - bit2) * (x_a_next - x_t);
    let c3_infinity = bit2 * ind_t2 * (x_a_next - x_p);

    let x_diff = x_p - x_t;
    let y_diff = y_p - y_t;
    let chord_x = (x_a_next + x_t + x_p) * x_diff * x_diff - y_diff * y_diff;
    let c3_add = bit2 * (one - ind_t2) * chord_x;

    c3_skip + c3_infinity + c3_add
}

/// C4: Conditional addition y-coordinate constraint over Fq2 (bit-dependent)
#[allow(clippy::too_many_arguments)]
fn compute_c4(
    bit: Fq,
    ind_t: Fq,
    x_a_next: Fq2,
    y_a_next: Fq2,
    x_t: Fq2,
    y_t: Fq2,
    x_p: Fq2,
    y_p: Fq2,
) -> Fq2 {
    let one = Fq2::one();
    let bit2 = fq2_from_fq(bit);
    let ind_t2 = fq2_from_fq(ind_t);

    let c4_skip = (one - bit2) * (y_a_next - y_t);
    let c4_infinity = bit2 * ind_t2 * (y_a_next - y_p);

    let x_diff = x_p - x_t;
    let y_diff = y_p - y_t;
    let chord_y = (y_a_next + y_t) * x_diff - y_diff * (x_t - x_a_next);
    let c4_add = bit2 * (one - ind_t2) * chord_y;

    c4_skip + c4_infinity + c4_add
}

/// C5: Doubling preserves infinity (in Fq)
fn compute_c5(ind_a: Fq, ind_t: Fq) -> Fq {
    ind_a * (Fq::one() - ind_t)
}

// =============================================================================
// DATA TYPES
// =============================================================================

#[derive(Clone)]
pub struct G2ScalarMulConstraintPolynomials {
    pub x_a_c0: Vec<Fq>,
    pub x_a_c1: Vec<Fq>,
    pub y_a_c0: Vec<Fq>,
    pub y_a_c1: Vec<Fq>,
    pub x_t_c0: Vec<Fq>,
    pub x_t_c1: Vec<Fq>,
    pub y_t_c0: Vec<Fq>,
    pub y_t_c1: Vec<Fq>,
    pub x_a_next_c0: Vec<Fq>,
    pub x_a_next_c1: Vec<Fq>,
    pub y_a_next_c0: Vec<Fq>,
    pub y_a_next_c1: Vec<Fq>,
    pub t_is_infinity: Vec<Fq>,
    pub a_is_infinity: Vec<Fq>,
    pub base_point: (Fq2, Fq2),
    pub constraint_index: usize,
}

#[derive(Clone)]
pub struct G2ScalarMulParams {
    pub num_constraint_vars: usize,
    pub num_constraints: usize,
    pub sumcheck_id: SumcheckId,
}

impl G2ScalarMulParams {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            num_constraint_vars: 11, // uniform matrix (8-var MLEs zero-padded to 11-var)
            num_constraints,
            sumcheck_id: SumcheckId::G2ScalarMul,
        }
    }
}

// =============================================================================
// PROVER
// =============================================================================

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct G2ScalarMulProver<F: JoltField, T: Transcript> {
    pub params: G2ScalarMulParams,
    pub base_points: Vec<(Fq2, Fq2)>,
    pub constraint_indices: Vec<usize>,

    pub eq_x: MultilinearPolynomial<F>,
    pub r_x: Vec<F::Challenge>,
    pub gamma: F,
    pub delta: F,

    pub x_a_c0_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub x_a_c1_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_a_c0_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_a_c1_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub x_t_c0_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub x_t_c1_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_t_c0_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_t_c1_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub x_a_next_c0_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub x_a_next_c1_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_a_next_c0_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub y_a_next_c1_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub t_is_infinity_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub a_is_infinity_mlpoly: Vec<MultilinearPolynomial<F>>,
    pub bit_public_mlpoly: Vec<MultilinearPolynomial<F>>,

    pub x_a_c0_claims: Vec<F>,
    pub x_a_c1_claims: Vec<F>,
    pub y_a_c0_claims: Vec<F>,
    pub y_a_c1_claims: Vec<F>,
    pub x_t_c0_claims: Vec<F>,
    pub x_t_c1_claims: Vec<F>,
    pub y_t_c0_claims: Vec<F>,
    pub y_t_c1_claims: Vec<F>,
    pub x_a_next_c0_claims: Vec<F>,
    pub x_a_next_c1_claims: Vec<F>,
    pub y_a_next_c0_claims: Vec<F>,
    pub y_a_next_c1_claims: Vec<F>,
    pub t_is_infinity_claims: Vec<F>,
    pub a_is_infinity_claims: Vec<F>,

    pub round: usize,
    pub _marker: std::marker::PhantomData<T>,
}

impl<F: JoltField, T: Transcript> G2ScalarMulProver<F, T> {
    pub fn new(
        params: G2ScalarMulParams,
        constraint_polys: Vec<G2ScalarMulConstraintPolynomials>,
        public_inputs: Vec<G2ScalarMulPublicInputs>,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<F::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        let gamma = transcript.challenge_scalar_optimized::<F>();
        let delta = transcript.challenge_scalar_optimized::<F>();

        // Runtime check that F = Fq (recursion SNARK base field)
        use std::any::TypeId;
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("G2 scalar multiplication requires F = Fq for recursion SNARK");
        }

        let eq_x = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_x));

        let mut base_points = Vec::new();
        let mut constraint_indices = Vec::new();

        let mut x_a_c0_mlpoly = Vec::new();
        let mut x_a_c1_mlpoly = Vec::new();
        let mut y_a_c0_mlpoly = Vec::new();
        let mut y_a_c1_mlpoly = Vec::new();
        let mut x_t_c0_mlpoly = Vec::new();
        let mut x_t_c1_mlpoly = Vec::new();
        let mut y_t_c0_mlpoly = Vec::new();
        let mut y_t_c1_mlpoly = Vec::new();
        let mut x_a_next_c0_mlpoly = Vec::new();
        let mut x_a_next_c1_mlpoly = Vec::new();
        let mut y_a_next_c0_mlpoly = Vec::new();
        let mut y_a_next_c1_mlpoly = Vec::new();
        let mut t_is_infinity_mlpoly = Vec::new();
        let mut a_is_infinity_mlpoly = Vec::new();
        let mut bit_public_mlpoly = Vec::new();

        assert_eq!(
            constraint_polys.len(),
            public_inputs.len(),
            "G2ScalarMulProver: constraint_polys and public_inputs must have same length"
        );

        for (poly, pub_in) in constraint_polys.into_iter().zip(public_inputs.iter()) {
            base_points.push(poly.base_point);
            constraint_indices.push(poly.constraint_index);

            // SAFETY: We checked F = Fq above, so these transmutes are safe
            let to_f = |v: Vec<Fq>| -> Vec<F> { unsafe { std::mem::transmute(v) } };

            x_a_c0_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                to_f(poly.x_a_c0),
            )));
            x_a_c1_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                to_f(poly.x_a_c1),
            )));
            y_a_c0_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                to_f(poly.y_a_c0),
            )));
            y_a_c1_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                to_f(poly.y_a_c1),
            )));
            x_t_c0_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                to_f(poly.x_t_c0),
            )));
            x_t_c1_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                to_f(poly.x_t_c1),
            )));
            y_t_c0_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                to_f(poly.y_t_c0),
            )));
            y_t_c1_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                to_f(poly.y_t_c1),
            )));
            x_a_next_c0_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                to_f(poly.x_a_next_c0),
            )));
            x_a_next_c1_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                to_f(poly.x_a_next_c1),
            )));
            y_a_next_c0_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                to_f(poly.y_a_next_c0),
            )));
            y_a_next_c1_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                to_f(poly.y_a_next_c1),
            )));
            t_is_infinity_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                to_f(poly.t_is_infinity),
            )));
            a_is_infinity_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                to_f(poly.a_is_infinity),
            )));

            // Build the (padded) bit polynomial from public scalar bits.
            let bits = pub_in.bits_msb();
            let mut bit_evals = vec![F::zero(); 1 << params.num_constraint_vars];
            for i in 0..256 {
                bit_evals[i] = if bits[i] { F::one() } else { F::zero() };
            }
            bit_public_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                bit_evals,
            )));
        }

        Self {
            params,
            base_points,
            constraint_indices,
            eq_x,
            r_x,
            gamma: gamma.into(),
            delta: delta.into(),
            x_a_c0_mlpoly,
            x_a_c1_mlpoly,
            y_a_c0_mlpoly,
            y_a_c1_mlpoly,
            x_t_c0_mlpoly,
            x_t_c1_mlpoly,
            y_t_c0_mlpoly,
            y_t_c1_mlpoly,
            x_a_next_c0_mlpoly,
            x_a_next_c1_mlpoly,
            y_a_next_c0_mlpoly,
            y_a_next_c1_mlpoly,
            t_is_infinity_mlpoly,
            a_is_infinity_mlpoly,
            bit_public_mlpoly,
            x_a_c0_claims: vec![],
            x_a_c1_claims: vec![],
            y_a_c0_claims: vec![],
            y_a_c1_claims: vec![],
            x_t_c0_claims: vec![],
            x_t_c1_claims: vec![],
            y_t_c0_claims: vec![],
            y_t_c1_claims: vec![],
            x_a_next_c0_claims: vec![],
            x_a_next_c1_claims: vec![],
            y_a_next_c0_claims: vec![],
            y_a_next_c1_claims: vec![],
            t_is_infinity_claims: vec![],
            a_is_infinity_claims: vec![],
            round: 0,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for G2ScalarMulProver<F, T> {
    fn degree(&self) -> usize {
        // Max per-variable degree comes from eq_x (degree 1) times the highest-degree constraint
        // term (degree 5 from C3/C4 chord formulas), so total degree bound is 6.
        6
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "G2ScalarMul::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 6;
        const NUM_CONSTRAINT_TERMS: usize = 13;

        let num_x_remaining = self.eq_x.get_num_vars();
        let x_half = 1 << (num_x_remaining - 1);

        // Precompute delta powers for batching the 13 constraint terms within an instance.
        let mut delta_pows = [F::zero(); NUM_CONSTRAINT_TERMS];
        delta_pows[0] = F::one();
        for i in 1..NUM_CONSTRAINT_TERMS {
            delta_pows[i] = delta_pows[i - 1] * self.delta;
        }

        let total_evals = (0..x_half)
            .into_par_iter()
            .map(|x_idx| {
                let eq_x_evals = self
                    .eq_x
                    .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                let mut x_evals = [F::zero(); DEGREE];
                let mut gamma_power = self.gamma;

                for i in 0..self.params.num_constraints {
                    let x_a_c0_evals = self.x_a_c0_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let x_a_c1_evals = self.x_a_c1_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_a_c0_evals = self.y_a_c0_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_a_c1_evals = self.y_a_c1_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let x_t_c0_evals = self.x_t_c0_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let x_t_c1_evals = self.x_t_c1_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_t_c0_evals = self.y_t_c0_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_t_c1_evals = self.y_t_c1_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let x_a_next_c0_evals = self.x_a_next_c0_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let x_a_next_c1_evals = self.x_a_next_c1_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_a_next_c0_evals = self.y_a_next_c0_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_a_next_c1_evals = self.y_a_next_c1_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let ind_t_evals = self.t_is_infinity_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let ind_a_evals = self.a_is_infinity_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let bit_evals = self.bit_public_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                    let (x_p, y_p) = self.base_points[i];

                    for t in 0..DEGREE {
                        // SAFETY: We checked F = Fq in new(), so these transmutes are safe.
                        let x_a_c0_fq: Fq = unsafe { std::mem::transmute_copy(&x_a_c0_evals[t]) };
                        let x_a_c1_fq: Fq = unsafe { std::mem::transmute_copy(&x_a_c1_evals[t]) };
                        let y_a_c0_fq: Fq = unsafe { std::mem::transmute_copy(&y_a_c0_evals[t]) };
                        let y_a_c1_fq: Fq = unsafe { std::mem::transmute_copy(&y_a_c1_evals[t]) };
                        let x_t_c0_fq: Fq = unsafe { std::mem::transmute_copy(&x_t_c0_evals[t]) };
                        let x_t_c1_fq: Fq = unsafe { std::mem::transmute_copy(&x_t_c1_evals[t]) };
                        let y_t_c0_fq: Fq = unsafe { std::mem::transmute_copy(&y_t_c0_evals[t]) };
                        let y_t_c1_fq: Fq = unsafe { std::mem::transmute_copy(&y_t_c1_evals[t]) };
                        let x_a_next_c0_fq: Fq =
                            unsafe { std::mem::transmute_copy(&x_a_next_c0_evals[t]) };
                        let x_a_next_c1_fq: Fq =
                            unsafe { std::mem::transmute_copy(&x_a_next_c1_evals[t]) };
                        let y_a_next_c0_fq: Fq =
                            unsafe { std::mem::transmute_copy(&y_a_next_c0_evals[t]) };
                        let y_a_next_c1_fq: Fq =
                            unsafe { std::mem::transmute_copy(&y_a_next_c1_evals[t]) };
                        let ind_t_fq: Fq = unsafe { std::mem::transmute_copy(&ind_t_evals[t]) };
                        let ind_a_fq: Fq = unsafe { std::mem::transmute_copy(&ind_a_evals[t]) };
                        let bit_fq: Fq = unsafe { std::mem::transmute_copy(&bit_evals[t]) };

                        let x_a = Fq2::new(x_a_c0_fq, x_a_c1_fq);
                        let y_a = Fq2::new(y_a_c0_fq, y_a_c1_fq);
                        let x_t = Fq2::new(x_t_c0_fq, x_t_c1_fq);
                        let y_t = Fq2::new(y_t_c0_fq, y_t_c1_fq);
                        let x_a_next = Fq2::new(x_a_next_c0_fq, x_a_next_c1_fq);
                        let y_a_next = Fq2::new(y_a_next_c0_fq, y_a_next_c1_fq);

                        // Compute constraints (Fq2 for curve-law constraints; Fq for bit/indicator constraints)
                        let c1 = compute_c1(x_a, y_a, x_t);
                        let c2 = compute_c2(x_a, y_a, x_t, y_t);
                        let c3 = compute_c3(bit_fq, ind_t_fq, x_a_next, x_t, y_t, x_p, y_p);
                        let c4 =
                            compute_c4(bit_fq, ind_t_fq, x_a_next, y_a_next, x_t, y_t, x_p, y_p);
                        let c5_fq = compute_c5(ind_a_fq, ind_t_fq);

                        // C6: if ind_T = 1 then x_T = 0 and y_T = 0 in Fq2
                        // Implemented as 4 base-field constraints on (c0, c1) components.
                        let c6_xt_c0_fq = ind_t_fq * x_t.c0;
                        let c6_xt_c1_fq = ind_t_fq * x_t.c1;
                        let c6_yt_c0_fq = ind_t_fq * y_t.c0;
                        let c6_yt_c1_fq = ind_t_fq * y_t.c1;

                        // Convert to F (Fq)
                        let c1_c0: F = unsafe { std::mem::transmute_copy(&c1.c0) };
                        let c1_c1: F = unsafe { std::mem::transmute_copy(&c1.c1) };
                        let c2_c0: F = unsafe { std::mem::transmute_copy(&c2.c0) };
                        let c2_c1: F = unsafe { std::mem::transmute_copy(&c2.c1) };
                        let c3_c0: F = unsafe { std::mem::transmute_copy(&c3.c0) };
                        let c3_c1: F = unsafe { std::mem::transmute_copy(&c3.c1) };
                        let c4_c0: F = unsafe { std::mem::transmute_copy(&c4.c0) };
                        let c4_c1: F = unsafe { std::mem::transmute_copy(&c4.c1) };
                        let c5: F = unsafe { std::mem::transmute_copy(&c5_fq) };
                        let c6_xt_c0: F = unsafe { std::mem::transmute_copy(&c6_xt_c0_fq) };
                        let c6_xt_c1: F = unsafe { std::mem::transmute_copy(&c6_xt_c1_fq) };
                        let c6_yt_c0: F = unsafe { std::mem::transmute_copy(&c6_yt_c0_fq) };
                        let c6_yt_c1: F = unsafe { std::mem::transmute_copy(&c6_yt_c1_fq) };

                        let constraint_val = delta_pows[0] * c1_c0
                            + delta_pows[1] * c1_c1
                            + delta_pows[2] * c2_c0
                            + delta_pows[3] * c2_c1
                            + delta_pows[4] * c3_c0
                            + delta_pows[5] * c3_c1
                            + delta_pows[6] * c4_c0
                            + delta_pows[7] * c4_c1
                            + delta_pows[8] * c5
                            + delta_pows[9] * c6_xt_c0
                            + delta_pows[10] * c6_xt_c1
                            + delta_pows[11] * c6_yt_c0
                            + delta_pows[12] * c6_yt_c1;

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

    #[tracing::instrument(skip_all, name = "G2ScalarMul::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.eq_x.bind_parallel(r_j, BindingOrder::LowToHigh);

        for poly in &mut self.x_a_c0_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.x_a_c1_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_a_c0_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_a_c1_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.x_t_c0_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.x_t_c1_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_t_c0_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_t_c1_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.x_a_next_c0_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.x_a_next_c1_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_a_next_c0_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_a_next_c1_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.t_is_infinity_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.a_is_infinity_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.bit_public_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        self.round = round + 1;

        if self.round == self.params.num_constraint_vars {
            self.x_a_c0_claims.clear();
            self.x_a_c1_claims.clear();
            self.y_a_c0_claims.clear();
            self.y_a_c1_claims.clear();
            self.x_t_c0_claims.clear();
            self.x_t_c1_claims.clear();
            self.y_t_c0_claims.clear();
            self.y_t_c1_claims.clear();
            self.x_a_next_c0_claims.clear();
            self.x_a_next_c1_claims.clear();
            self.y_a_next_c0_claims.clear();
            self.y_a_next_c1_claims.clear();
            self.t_is_infinity_claims.clear();
            self.a_is_infinity_claims.clear();

            for i in 0..self.params.num_constraints {
                self.x_a_c0_claims
                    .push(self.x_a_c0_mlpoly[i].get_bound_coeff(0));
                self.x_a_c1_claims
                    .push(self.x_a_c1_mlpoly[i].get_bound_coeff(0));
                self.y_a_c0_claims
                    .push(self.y_a_c0_mlpoly[i].get_bound_coeff(0));
                self.y_a_c1_claims
                    .push(self.y_a_c1_mlpoly[i].get_bound_coeff(0));
                self.x_t_c0_claims
                    .push(self.x_t_c0_mlpoly[i].get_bound_coeff(0));
                self.x_t_c1_claims
                    .push(self.x_t_c1_mlpoly[i].get_bound_coeff(0));
                self.y_t_c0_claims
                    .push(self.y_t_c0_mlpoly[i].get_bound_coeff(0));
                self.y_t_c1_claims
                    .push(self.y_t_c1_mlpoly[i].get_bound_coeff(0));
                self.x_a_next_c0_claims
                    .push(self.x_a_next_c0_mlpoly[i].get_bound_coeff(0));
                self.x_a_next_c1_claims
                    .push(self.x_a_next_c1_mlpoly[i].get_bound_coeff(0));
                self.y_a_next_c0_claims
                    .push(self.y_a_next_c0_mlpoly[i].get_bound_coeff(0));
                self.y_a_next_c1_claims
                    .push(self.y_a_next_c1_mlpoly[i].get_bound_coeff(0));
                self.t_is_infinity_claims
                    .push(self.t_is_infinity_mlpoly[i].get_bound_coeff(0));
                self.a_is_infinity_claims
                    .push(self.a_is_infinity_mlpoly[i].get_bound_coeff(0));
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
            append_g2_scalar_mul_virtual_claims(
                accumulator,
                transcript,
                i,
                self.params.sumcheck_id,
                &opening_point,
                self.x_a_c0_claims[i],
                self.x_a_c1_claims[i],
                self.y_a_c0_claims[i],
                self.y_a_c1_claims[i],
                self.x_t_c0_claims[i],
                self.x_t_c1_claims[i],
                self.y_t_c0_claims[i],
                self.y_t_c1_claims[i],
                self.x_a_next_c0_claims[i],
                self.x_a_next_c1_claims[i],
                self.y_a_next_c0_claims[i],
                self.y_a_next_c1_claims[i],
                self.t_is_infinity_claims[i],
                self.a_is_infinity_claims[i],
            );
        }
    }
}

// =============================================================================
// VERIFIER
// =============================================================================

pub struct G2ScalarMulVerifier<F: JoltField> {
    pub params: G2ScalarMulParams,
    pub r_x: Vec<F::Challenge>,
    pub gamma: F,
    pub delta: F,
    pub num_constraints: usize,
    pub base_points: Vec<(Fq2, Fq2)>,
    pub constraint_indices: Vec<usize>,
    pub public_inputs: Vec<G2ScalarMulPublicInputs>,
}

impl<F: JoltField> G2ScalarMulVerifier<F> {
    pub fn new<T: Transcript>(
        params: G2ScalarMulParams,
        base_points: Vec<(Fq2, Fq2)>,
        constraint_indices: Vec<usize>,
        public_inputs: Vec<G2ScalarMulPublicInputs>,
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
            base_points,
            constraint_indices,
            public_inputs,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for G2ScalarMulVerifier<F> {
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
        use std::any::TypeId;

        // Runtime check that F = Fq
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("G2 scalar multiplication requires F = Fq for recursion SNARK");
        }

        let r_x_f: Vec<F> = self.r_x.iter().map(|c| (*c).into()).collect();
        let r_star_f: Vec<F> = sumcheck_challenges
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();
        let eq_eval = EqPolynomial::mle(&r_x_f, &r_star_f);

        let mut total = F::zero();
        let mut gamma_power = self.gamma;

        // Precompute delta powers for the 13 batched terms
        const NUM_CONSTRAINT_TERMS: usize = 13;
        let mut delta_pows = [F::zero(); NUM_CONSTRAINT_TERMS];
        delta_pows[0] = F::one();
        for i in 1..NUM_CONSTRAINT_TERMS {
            delta_pows[i] = delta_pows[i - 1] * self.delta;
        }

        for i in 0..self.num_constraints {
            let (
                x_a_c0_claim,
                x_a_c1_claim,
                y_a_c0_claim,
                y_a_c1_claim,
                x_t_c0_claim,
                x_t_c1_claim,
                y_t_c0_claim,
                y_t_c1_claim,
                x_a_next_c0_claim,
                x_a_next_c1_claim,
                y_a_next_c0_claim,
                y_a_next_c1_claim,
                t_is_infinity_claim,
                a_is_infinity_claim,
            ) = get_g2_scalar_mul_virtual_claims(accumulator, i, self.params.sumcheck_id);

            // SAFETY: We checked F = Fq above, so these transmutes are safe.
            let x_a_c0_fq: Fq = unsafe { std::mem::transmute_copy(&x_a_c0_claim) };
            let x_a_c1_fq: Fq = unsafe { std::mem::transmute_copy(&x_a_c1_claim) };
            let y_a_c0_fq: Fq = unsafe { std::mem::transmute_copy(&y_a_c0_claim) };
            let y_a_c1_fq: Fq = unsafe { std::mem::transmute_copy(&y_a_c1_claim) };
            let x_t_c0_fq: Fq = unsafe { std::mem::transmute_copy(&x_t_c0_claim) };
            let x_t_c1_fq: Fq = unsafe { std::mem::transmute_copy(&x_t_c1_claim) };
            let y_t_c0_fq: Fq = unsafe { std::mem::transmute_copy(&y_t_c0_claim) };
            let y_t_c1_fq: Fq = unsafe { std::mem::transmute_copy(&y_t_c1_claim) };
            let x_a_next_c0_fq: Fq = unsafe { std::mem::transmute_copy(&x_a_next_c0_claim) };
            let x_a_next_c1_fq: Fq = unsafe { std::mem::transmute_copy(&x_a_next_c1_claim) };
            let y_a_next_c0_fq: Fq = unsafe { std::mem::transmute_copy(&y_a_next_c0_claim) };
            let y_a_next_c1_fq: Fq = unsafe { std::mem::transmute_copy(&y_a_next_c1_claim) };
            let ind_t_fq: Fq = unsafe { std::mem::transmute_copy(&t_is_infinity_claim) };
            let ind_a_fq: Fq = unsafe { std::mem::transmute_copy(&a_is_infinity_claim) };
            let bit_eval: F = self.public_inputs[i].evaluate_bit_mle(&r_star_f);
            let bit_fq: Fq = unsafe { std::mem::transmute_copy(&bit_eval) };

            let x_a = Fq2::new(x_a_c0_fq, x_a_c1_fq);
            let y_a = Fq2::new(y_a_c0_fq, y_a_c1_fq);
            let x_t = Fq2::new(x_t_c0_fq, x_t_c1_fq);
            let y_t = Fq2::new(y_t_c0_fq, y_t_c1_fq);
            let x_a_next = Fq2::new(x_a_next_c0_fq, x_a_next_c1_fq);
            let y_a_next = Fq2::new(y_a_next_c0_fq, y_a_next_c1_fq);

            let (x_p, y_p) = self.base_points[i];

            let c1 = compute_c1(x_a, y_a, x_t);
            let c2 = compute_c2(x_a, y_a, x_t, y_t);
            let c3 = compute_c3(bit_fq, ind_t_fq, x_a_next, x_t, y_t, x_p, y_p);
            let c4 = compute_c4(bit_fq, ind_t_fq, x_a_next, y_a_next, x_t, y_t, x_p, y_p);
            let c5_fq = compute_c5(ind_a_fq, ind_t_fq);

            let c6_xt_c0_fq = ind_t_fq * x_t.c0;
            let c6_xt_c1_fq = ind_t_fq * x_t.c1;
            let c6_yt_c0_fq = ind_t_fq * y_t.c0;
            let c6_yt_c1_fq = ind_t_fq * y_t.c1;

            let c1_c0: F = unsafe { std::mem::transmute_copy(&c1.c0) };
            let c1_c1: F = unsafe { std::mem::transmute_copy(&c1.c1) };
            let c2_c0: F = unsafe { std::mem::transmute_copy(&c2.c0) };
            let c2_c1: F = unsafe { std::mem::transmute_copy(&c2.c1) };
            let c3_c0: F = unsafe { std::mem::transmute_copy(&c3.c0) };
            let c3_c1: F = unsafe { std::mem::transmute_copy(&c3.c1) };
            let c4_c0: F = unsafe { std::mem::transmute_copy(&c4.c0) };
            let c4_c1: F = unsafe { std::mem::transmute_copy(&c4.c1) };
            let c5: F = unsafe { std::mem::transmute_copy(&c5_fq) };
            let c6_xt_c0: F = unsafe { std::mem::transmute_copy(&c6_xt_c0_fq) };
            let c6_xt_c1: F = unsafe { std::mem::transmute_copy(&c6_xt_c1_fq) };
            let c6_yt_c0: F = unsafe { std::mem::transmute_copy(&c6_yt_c0_fq) };
            let c6_yt_c1: F = unsafe { std::mem::transmute_copy(&c6_yt_c1_fq) };

            let constraint_value = delta_pows[0] * c1_c0
                + delta_pows[1] * c1_c1
                + delta_pows[2] * c2_c0
                + delta_pows[3] * c2_c1
                + delta_pows[4] * c3_c0
                + delta_pows[5] * c3_c1
                + delta_pows[6] * c4_c0
                + delta_pows[7] * c4_c1
                + delta_pows[8] * c5
                + delta_pows[9] * c6_xt_c0
                + delta_pows[10] * c6_xt_c1
                + delta_pows[11] * c6_yt_c0
                + delta_pows[12] * c6_yt_c1;

            total += gamma_power * constraint_value;
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
            append_g2_scalar_mul_virtual_openings(
                accumulator,
                transcript,
                i,
                self.params.sumcheck_id,
                &opening_point,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::dense_mlpoly::DensePolynomial;
    use crate::subprotocols::sumcheck::BatchedSumcheck;
    use crate::transcripts::Blake2bTranscript;
    use crate::zkvm::recursion::constraints_sys::DoryMatrixBuilder;
    use ark_bn254::{Fr, G2Affine};
    use ark_std::UniformRand;

    #[test]
    fn test_g2_scalar_mul_sumcheck_roundtrip_single_instance() {
        let mut rng = ark_std::test_rng();

        // Build a single G2 scalar mul witness (8-var MLEs)
        let point = G2Affine::rand(&mut rng);
        let scalar = Fr::rand(&mut rng);
        let steps =
            crate::poly::commitment::dory::g2_scalar_mul_witness::G2ScalarMultiplicationSteps::new(
                point, scalar,
            );

        // Pad 8-var MLEs to 11-var (zero padding) to match recursion matrix convention.
        let pad11 = |mle_8: &Vec<Fq>| -> Vec<Fq> {
            DoryMatrixBuilder::pad_8var_to_11var_zero_padding(mle_8)
        };

        let constraint = G2ScalarMulConstraintPolynomials {
            x_a_c0: pad11(&steps.x_a_c0_mles[0]),
            x_a_c1: pad11(&steps.x_a_c1_mles[0]),
            y_a_c0: pad11(&steps.y_a_c0_mles[0]),
            y_a_c1: pad11(&steps.y_a_c1_mles[0]),
            x_t_c0: pad11(&steps.x_t_c0_mles[0]),
            x_t_c1: pad11(&steps.x_t_c1_mles[0]),
            y_t_c0: pad11(&steps.y_t_c0_mles[0]),
            y_t_c1: pad11(&steps.y_t_c1_mles[0]),
            x_a_next_c0: pad11(&steps.x_a_next_c0_mles[0]),
            x_a_next_c1: pad11(&steps.x_a_next_c1_mles[0]),
            y_a_next_c0: pad11(&steps.y_a_next_c0_mles[0]),
            y_a_next_c1: pad11(&steps.y_a_next_c1_mles[0]),
            t_is_infinity: pad11(&steps.t_is_infinity_mles[0]),
            a_is_infinity: pad11(&steps.a_is_infinity_mles[0]),
            base_point: (steps.point_base.x, steps.point_base.y),
            constraint_index: 0,
        };

        let params = G2ScalarMulParams::new(1);
        let public_inputs = vec![G2ScalarMulPublicInputs::new(scalar)];

        // Prove
        let mut prover_transcript = Blake2bTranscript::new(b"g2_scalar_mul_sumcheck_test");
        let mut prover = G2ScalarMulProver::<Fq, Blake2bTranscript>::new(
            params.clone(),
            vec![constraint],
            public_inputs.clone(),
            &mut prover_transcript,
        );

        // Debug: check the boolean-cube sum (the claimed input for the sumcheck) is actually 0.
        // If this fails, either the witness/constraints are inconsistent, or padding/gating is wrong.
        {
            const NUM_CONSTRAINT_TERMS: usize = 13;
            let mut delta_pows = [Fq::zero(); NUM_CONSTRAINT_TERMS];
            delta_pows[0] = Fq::one();
            for i in 1..NUM_CONSTRAINT_TERMS {
                delta_pows[i] = delta_pows[i - 1] * prover.delta;
            }

            let eq_x: &DensePolynomial<Fq> = (&prover.eq_x).try_into().unwrap();
            let x_a_c0: &DensePolynomial<Fq> = (&prover.x_a_c0_mlpoly[0]).try_into().unwrap();
            let x_a_c1: &DensePolynomial<Fq> = (&prover.x_a_c1_mlpoly[0]).try_into().unwrap();
            let y_a_c0: &DensePolynomial<Fq> = (&prover.y_a_c0_mlpoly[0]).try_into().unwrap();
            let y_a_c1: &DensePolynomial<Fq> = (&prover.y_a_c1_mlpoly[0]).try_into().unwrap();
            let x_t_c0: &DensePolynomial<Fq> = (&prover.x_t_c0_mlpoly[0]).try_into().unwrap();
            let x_t_c1: &DensePolynomial<Fq> = (&prover.x_t_c1_mlpoly[0]).try_into().unwrap();
            let y_t_c0: &DensePolynomial<Fq> = (&prover.y_t_c0_mlpoly[0]).try_into().unwrap();
            let y_t_c1: &DensePolynomial<Fq> = (&prover.y_t_c1_mlpoly[0]).try_into().unwrap();
            let x_a_next_c0: &DensePolynomial<Fq> =
                (&prover.x_a_next_c0_mlpoly[0]).try_into().unwrap();
            let x_a_next_c1: &DensePolynomial<Fq> =
                (&prover.x_a_next_c1_mlpoly[0]).try_into().unwrap();
            let y_a_next_c0: &DensePolynomial<Fq> =
                (&prover.y_a_next_c0_mlpoly[0]).try_into().unwrap();
            let y_a_next_c1: &DensePolynomial<Fq> =
                (&prover.y_a_next_c1_mlpoly[0]).try_into().unwrap();
            let ind_t: &DensePolynomial<Fq> = (&prover.t_is_infinity_mlpoly[0]).try_into().unwrap();
            let ind_a: &DensePolynomial<Fq> = (&prover.a_is_infinity_mlpoly[0]).try_into().unwrap();
            let bit: &DensePolynomial<Fq> = (&prover.bit_public_mlpoly[0]).try_into().unwrap();

            let (x_p, y_p) = prover.base_points[0];

            let mut sum = Fq::zero();
            let gamma_power = prover.gamma; // single constraint → starts at γ¹

            for idx in 0..(1 << prover.params.num_constraint_vars) {
                let eq = eq_x.Z[idx];

                let x_a = Fq2::new(x_a_c0.Z[idx], x_a_c1.Z[idx]);
                let y_a = Fq2::new(y_a_c0.Z[idx], y_a_c1.Z[idx]);
                let x_t = Fq2::new(x_t_c0.Z[idx], x_t_c1.Z[idx]);
                let y_t = Fq2::new(y_t_c0.Z[idx], y_t_c1.Z[idx]);
                let x_a_next = Fq2::new(x_a_next_c0.Z[idx], x_a_next_c1.Z[idx]);
                let y_a_next = Fq2::new(y_a_next_c0.Z[idx], y_a_next_c1.Z[idx]);

                let ind_t_fq = ind_t.Z[idx];
                let ind_a_fq = ind_a.Z[idx];
                let bit_fq = bit.Z[idx];

                let c1 = compute_c1(x_a, y_a, x_t);
                let c2 = compute_c2(x_a, y_a, x_t, y_t);
                let c3 = compute_c3(bit_fq, ind_t_fq, x_a_next, x_t, y_t, x_p, y_p);
                let c4 = compute_c4(bit_fq, ind_t_fq, x_a_next, y_a_next, x_t, y_t, x_p, y_p);
                let c5_fq = compute_c5(ind_a_fq, ind_t_fq);

                let c6_xt_c0_fq = ind_t_fq * x_t.c0;
                let c6_xt_c1_fq = ind_t_fq * x_t.c1;
                let c6_yt_c0_fq = ind_t_fq * y_t.c0;
                let c6_yt_c1_fq = ind_t_fq * y_t.c1;

                let constraint_val = delta_pows[0] * c1.c0
                    + delta_pows[1] * c1.c1
                    + delta_pows[2] * c2.c0
                    + delta_pows[3] * c2.c1
                    + delta_pows[4] * c3.c0
                    + delta_pows[5] * c3.c1
                    + delta_pows[6] * c4.c0
                    + delta_pows[7] * c4.c1
                    + delta_pows[8] * c5_fq
                    + delta_pows[9] * c6_xt_c0_fq
                    + delta_pows[10] * c6_xt_c1_fq
                    + delta_pows[11] * c6_yt_c0_fq
                    + delta_pows[12] * c6_yt_c1_fq;

                sum += eq * gamma_power * constraint_val;
            }

            assert_eq!(
                sum,
                Fq::zero(),
                "boolean-cube sum is not 0; input_claim=0 is incorrect for this witness/constraint"
            );
        }
        let mut prover_acc = ProverOpeningAccumulator::<Fq>::new(params.num_constraint_vars);
        let (proof, r_prove) =
            BatchedSumcheck::prove(vec![&mut prover], &mut prover_acc, &mut prover_transcript);

        // Debug: sanity-check endianness assumptions for eq_x and the public bit polynomial.
        {
            use crate::poly::eq_poly::EqPolynomial;
            let r_star_f: Vec<Fq> = r_prove.iter().rev().map(|c| (*c).into()).collect();
            let r_x_f: Vec<Fq> = prover.r_x.iter().map(|c| (*c).into()).collect();

            let eq_eval_bound = prover.eq_x.get_bound_coeff(0);
            let eq_eval_mle = EqPolynomial::<Fq>::mle(&r_x_f, &r_star_f);
            assert_eq!(
                eq_eval_bound, eq_eval_mle,
                "eq_x(r*) mismatch (endianness/order?)"
            );

            let bit_eval_bound = prover.bit_public_mlpoly[0].get_bound_coeff(0);
            let bit_eval_pub = public_inputs[0].evaluate_bit_mle(&r_star_f);
            assert_eq!(
                bit_eval_bound, bit_eval_pub,
                "public bit polynomial eval mismatch (padding/order?)"
            );
        }

        // Verify (replay with fresh transcript and prover claims copied into verifier accumulator)
        let mut verifier_transcript = Blake2bTranscript::new(b"g2_scalar_mul_sumcheck_test");
        let verifier = G2ScalarMulVerifier::<Fq>::new(
            params,
            vec![(steps.point_base.x, steps.point_base.y)],
            vec![0],
            public_inputs,
            &mut verifier_transcript,
        );
        let mut verifier_acc = VerifierOpeningAccumulator::<Fq>::new(11);
        // Copy prover openings (claims) so verifier can compute expected_output_claim
        verifier_acc.openings = prover_acc.openings.clone();

        let r = BatchedSumcheck::verify(
            &proof,
            vec![&verifier],
            &mut verifier_acc,
            &mut verifier_transcript,
        )
        .expect("sumcheck should verify");
        assert_eq!(r.len(), 11);
    }
}
