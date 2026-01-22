//! G1 scalar multiplication sumcheck for proving G1 scalar multiplication constraints
//!
//! Proves: 0 = Σ_x eq(r_x, x) * Σ_i γ^i * (Σ_j δ^j * C_{i,j}(x))
//! Where C_{i,j} are the scalar-mul constraints for each instance.
//!
//! See `spec.md` for the full specification and soundness proof.
//!
//! ## Constraints
//! - C1: Doubling x-coordinate: 4y_A²(x_T + 2x_A) - 9x_A⁴ = 0
//! - C2: Doubling y-coordinate: 3x_A²(x_T - x_A) + 2y_A(y_T + y_A) = 0
//! - C3: Conditional addition x-coord (bit-dependent)
//! - C4: Conditional addition y-coord (bit-dependent)
//! - C5: If A = O then T = O (infinity preserved)
//! - C6: If ind_T = 1 then (x_T, y_T) = (0,0)
//!
//! ## Batching
//! - Delta (δ) batches constraints within each scalar multiplication
//! - Gamma (γ) batches multiple scalar multiplication instances
//!
//! ## Public inputs
//! The scalar bits are treated as **public inputs** (derived from the scalar),
//! so we do NOT emit openings for the bit polynomial and we do NOT enforce a
//! separate bit-booleanity constraint.

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
use ark_bn254::{Fq, Fr};
use ark_ff::{BigInteger, One, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

/// Public inputs for a single G1 scalar multiplication (the scalar).
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G1ScalarMulPublicInputs {
    pub scalar: Fr,
}

/// Witness polynomials for a G1 scalar multiplication constraint.
///
/// Represents the double-and-add trace with 256 steps (MSB-first):
/// - `A_i`: accumulator point at step `i`
/// - `T_i = [2]A_i`: doubled point at step `i`
/// - `A_{i+1} = T_i + b_i·P`: next accumulator
#[derive(Clone, Debug)]
pub struct G1ScalarMulWitness {
    /// Index of this constraint in the constraint system
    pub constraint_index: usize,
    /// Base point P = (x, y) being multiplied
    pub base_point: (Fq, Fq),
    /// Accumulator x-coordinate: x_A(s) for each step s
    pub x_a: Vec<Fq>,
    /// Accumulator y-coordinate: y_A(s) for each step s
    pub y_a: Vec<Fq>,
    /// Doubled point x-coordinate: x_T(s) = x([2]A_s)
    pub x_t: Vec<Fq>,
    /// Doubled point y-coordinate: y_T(s) = y([2]A_s)
    pub y_t: Vec<Fq>,
    /// Next accumulator x-coordinate: x_A(s+1)
    pub x_a_next: Vec<Fq>,
    /// Next accumulator y-coordinate: y_A(s+1)
    pub y_a_next: Vec<Fq>,
    /// Indicator for T being at infinity: 1 if T_s = O, else 0
    pub t_indicator: Vec<Fq>,
    /// Indicator for A being at infinity: 1 if A_s = O, else 0
    pub a_indicator: Vec<Fq>,
}

impl G1ScalarMulPublicInputs {
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

/// Helper to append all virtual claims for a G1 scalar mul constraint
#[allow(clippy::too_many_arguments)]
fn append_g1_scalar_mul_virtual_claims<F: JoltField, T: Transcript>(
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
    x_a_claim: F,
    y_a_claim: F,
    x_t_claim: F,
    y_t_claim: F,
    x_a_next_claim: F,
    y_a_next_claim: F,
    t_is_infinity_claim: F,
    a_is_infinity_claim: F,
) {
    let claims = virtual_claims![
        VirtualPolynomial::g1_scalar_mul_xa(constraint_idx) => x_a_claim,
        VirtualPolynomial::g1_scalar_mul_ya(constraint_idx) => y_a_claim,
        VirtualPolynomial::g1_scalar_mul_xt(constraint_idx) => x_t_claim,
        VirtualPolynomial::g1_scalar_mul_yt(constraint_idx) => y_t_claim,
        VirtualPolynomial::g1_scalar_mul_xa_next(constraint_idx) => x_a_next_claim,
        VirtualPolynomial::g1_scalar_mul_ya_next(constraint_idx) => y_a_next_claim,
        VirtualPolynomial::g1_scalar_mul_t_indicator(constraint_idx) => t_is_infinity_claim,
        VirtualPolynomial::g1_scalar_mul_a_indicator(constraint_idx) => a_is_infinity_claim,
    ];
    append_virtual_claims(accumulator, transcript, sumcheck_id, opening_point, &claims);
}

/// Helper to retrieve all virtual claims for a G1 scalar mul constraint
/// Returns: (x_a, y_a, x_t, y_t, x_a_next, y_a_next, t_is_infinity, a_is_infinity)
fn get_g1_scalar_mul_virtual_claims<F: JoltField>(
    accumulator: &VerifierOpeningAccumulator<F>,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
) -> (F, F, F, F, F, F, F, F) {
    let polynomials = vec![
        VirtualPolynomial::g1_scalar_mul_xa(constraint_idx),
        VirtualPolynomial::g1_scalar_mul_ya(constraint_idx),
        VirtualPolynomial::g1_scalar_mul_xt(constraint_idx),
        VirtualPolynomial::g1_scalar_mul_yt(constraint_idx),
        VirtualPolynomial::g1_scalar_mul_xa_next(constraint_idx),
        VirtualPolynomial::g1_scalar_mul_ya_next(constraint_idx),
        VirtualPolynomial::g1_scalar_mul_t_indicator(constraint_idx),
        VirtualPolynomial::g1_scalar_mul_a_indicator(constraint_idx),
    ];
    let claims = get_virtual_claims(accumulator, sumcheck_id, &polynomials);
    (
        claims[0], claims[1], claims[2], claims[3], claims[4], claims[5], claims[6], claims[7],
    )
}

/// Helper to append virtual opening points for a G1 scalar mul constraint (verifier side)
fn append_g1_scalar_mul_virtual_openings<F: JoltField, T: Transcript>(
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
    constraint_idx: usize,
    sumcheck_id: SumcheckId,
    opening_point: &OpeningPoint<BIG_ENDIAN, F>,
) {
    let polynomials = vec![
        VirtualPolynomial::g1_scalar_mul_xa(constraint_idx),
        VirtualPolynomial::g1_scalar_mul_ya(constraint_idx),
        VirtualPolynomial::g1_scalar_mul_xt(constraint_idx),
        VirtualPolynomial::g1_scalar_mul_yt(constraint_idx),
        VirtualPolynomial::g1_scalar_mul_xa_next(constraint_idx),
        VirtualPolynomial::g1_scalar_mul_ya_next(constraint_idx),
        VirtualPolynomial::g1_scalar_mul_t_indicator(constraint_idx),
        VirtualPolynomial::g1_scalar_mul_a_indicator(constraint_idx),
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
// CONSTRAINT FUNCTIONS
// See g1_scalar_mul_spec.md for derivations and soundness proof
// =============================================================================

/// C1: Doubling x-coordinate constraint
/// Derived from tangent formula: λ = 3x_A² / 2y_A, x_T = λ² - 2x_A
/// Eliminating denominators: 4y_A²(x_T + 2x_A) - 9x_A⁴ = 0
///
/// Note: When A = O (infinity), we have x_A = y_A = 0, so C1 = 0 trivially.
/// The doubling of O is handled by C6 which ensures T = O when A = O.
fn compute_c1(x_a: Fq, y_a: Fq, x_t: Fq) -> Fq {
    let four = Fq::from(4u64);
    let two = Fq::from(2u64);
    let nine = Fq::from(9u64);

    let y_a_sq = y_a * y_a;
    let x_a_sq = x_a * x_a;
    let x_a_fourth = x_a_sq * x_a_sq;

    four * y_a_sq * (x_t + two * x_a) - nine * x_a_fourth
}

/// C2: Doubling y-coordinate constraint
/// Derived from: y_T = λ(x_A - x_T) - y_A
/// Eliminating denominators: 3x_A²(x_T - x_A) + 2y_A(y_T + y_A) = 0
fn compute_c2(x_a: Fq, y_a: Fq, x_t: Fq, y_t: Fq) -> Fq {
    let three = Fq::from(3u64);
    let two = Fq::from(2u64);

    let x_a_sq = x_a * x_a;
    three * x_a_sq * (x_t - x_a) + two * y_a * (y_t + y_a)
}

/// C3: Conditional addition x-coordinate constraint (BIT-DEPENDENT)
///
/// This is the CRITICAL constraint that binds the scalar bit to the operation:
/// - If b = 0: A_{i+1} = T_i (skip addition) → x_A' = x_T
/// - If b = 1: A_{i+1} = T_i + P (add base point) → chord formula
///
/// Additionally handles T = O case:
/// - If b = 1 and T = O: A_{i+1} = P → x_A' = x_P
///
/// Combined formula:
/// C3 = (1 - b) * (x_A' - x_T)
///    + b * ind_T * (x_A' - x_P)
///    + b * (1 - ind_T) * [(x_A' + x_T + x_P)(x_P - x_T)² - (y_P - y_T)²]
#[allow(clippy::too_many_arguments)]
fn compute_c3(bit: Fq, ind_t: Fq, x_a_next: Fq, x_t: Fq, y_t: Fq, x_p: Fq, y_p: Fq) -> Fq {
    let one = Fq::one();

    // Case b = 0: skip addition, must have x_A' = x_T
    let c3_skip = (one - bit) * (x_a_next - x_t);

    // Case b = 1, T = O: adding to infinity gives P, must have x_A' = x_P
    let c3_infinity = bit * ind_t * (x_a_next - x_p);

    // Case b = 1, T ≠ O: chord addition formula
    let x_diff = x_p - x_t;
    let y_diff = y_p - y_t;
    let chord_x = (x_a_next + x_t + x_p) * x_diff * x_diff - y_diff * y_diff;
    let c3_add = bit * (one - ind_t) * chord_x;

    c3_skip + c3_infinity + c3_add
}

/// C4: Conditional addition y-coordinate constraint (BIT-DEPENDENT)
///
/// Mirrors C3 for the y-coordinate:
/// - If b = 0: y_A' = y_T
/// - If b = 1 and T = O: y_A' = y_P
/// - If b = 1 and T ≠ O: chord formula for y
///
/// Chord y formula (denominator-free):
/// (y_A' + y_T)(x_P - x_T) - (y_P - y_T)(x_T - x_A') = 0
#[allow(clippy::too_many_arguments)]
fn compute_c4(
    bit: Fq,
    ind_t: Fq,
    x_a_next: Fq,
    y_a_next: Fq,
    x_t: Fq,
    y_t: Fq,
    x_p: Fq,
    y_p: Fq,
) -> Fq {
    let one = Fq::one();

    // Case b = 0: skip addition, must have y_A' = y_T
    let c4_skip = (one - bit) * (y_a_next - y_t);

    // Case b = 1, T = O: adding to infinity gives P, must have y_A' = y_P
    let c4_infinity = bit * ind_t * (y_a_next - y_p);

    // Case b = 1, T ≠ O: chord addition formula
    let x_diff = x_p - x_t;
    let y_diff = y_p - y_t;
    let chord_y = (y_a_next + y_t) * x_diff - y_diff * (x_t - x_a_next);
    let c4_add = bit * (one - ind_t) * chord_y;

    c4_skip + c4_infinity + c4_add
}

/// C5: Doubling preserves infinity
/// If A = O (ind_A = 1), then T = O (ind_T = 1)
/// Constraint: ind_A * (1 - ind_T) = 0
fn compute_c5(ind_a: Fq, ind_t: Fq) -> Fq {
    ind_a * (Fq::one() - ind_t)
}

/// C6: Infinity encoding check for T (field-independent)
///
/// We encode infinity as affine coordinates (0,0) plus an indicator bit `ind_T`.
/// The robust way to enforce `ind_T = 1 => (x_T, y_T) = (0,0)` over *any* field is:
///   - ind_T * x_T = 0
///   - ind_T * y_T = 0
fn compute_c6_x(ind_t: Fq, x_t: Fq) -> Fq {
    ind_t * x_t
}

fn compute_c6_y(ind_t: Fq, y_t: Fq) -> Fq {
    ind_t * y_t
}

/// Individual polynomial data for a single G1 scalar multiplication constraint
/// Note: This struct uses Fq for polynomial evaluations because G1 operations
/// produce values in the base field Fq
#[derive(Clone)]
pub struct G1ScalarMulConstraintPolynomials {
    pub x_a: Vec<Fq>,            // x-coords of accumulator A_i (all 256 steps)
    pub y_a: Vec<Fq>,            // y-coords of accumulator A_i (all 256 steps)
    pub x_t: Vec<Fq>,            // x-coords of doubled point T_i (all 256 steps)
    pub y_t: Vec<Fq>,            // y-coords of doubled point T_i (all 256 steps)
    pub x_a_next: Vec<Fq>,       // x-coords of A_{i+1} (shifted by 1)
    pub y_a_next: Vec<Fq>,       // y-coords of A_{i+1} (shifted by 1)
    pub t_is_infinity: Vec<Fq>,  // Indicator: 1 if T_i = O, 0 otherwise
    pub a_is_infinity: Vec<Fq>,  // Indicator: 1 if A_i = O, 0 otherwise
    pub base_point: (Fq, Fq),    // Base point P coordinates (public)
    pub constraint_index: usize, // Global constraint index
}

/// Parameters for G1 scalar multiplication sumcheck
#[derive(Clone)]
pub struct G1ScalarMulParams {
    /// Number of constraint variables (x) - 8 for 256-bit scalars
    pub num_constraint_vars: usize,

    /// Number of G1 scalar multiplication instances
    pub num_constraints: usize,

    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,
}

impl G1ScalarMulParams {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            num_constraint_vars: 11, // 11 vars for uniform matrix (8 scalar bits padded to 11)
            num_constraints,
            sumcheck_id: SumcheckId::G1ScalarMul,
        }
    }
}

/// Prover for G1 scalar multiplication sumcheck
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct G1ScalarMulProver<F: JoltField, T: Transcript> {
    /// Parameters
    pub params: G1ScalarMulParams,

    /// Base points for each scalar multiplication instance (must be Fq as G1 points)
    pub base_points: Vec<(Fq, Fq)>,

    /// Global constraint indices for each constraint
    pub constraint_indices: Vec<usize>,

    /// Equality polynomial for constraint variables x
    pub eq_x: MultilinearPolynomial<F>,

    /// Random challenge for eq(r_x, x)
    pub r_x: Vec<F::Challenge>,

    /// Gamma coefficient for batching scalar multiplication instances
    pub gamma: F,

    /// Delta coefficient for batching 7 constraints within each instance
    pub delta: F,

    /// x_a polynomials as multilinear (one per instance, contains all steps)
    pub x_a_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// y_a polynomials as multilinear (one per instance, contains all steps)
    pub y_a_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// x_t polynomials as multilinear (one per instance, contains all steps)
    pub x_t_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// y_t polynomials as multilinear (one per instance, contains all steps)
    pub y_t_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// x_a_next polynomials as multilinear (shifted A_{i+1} values)
    pub x_a_next_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// y_a_next polynomials as multilinear (shifted A_{i+1} values)
    pub y_a_next_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// Infinity indicator for T (1 if T = O, 0 otherwise)
    pub t_is_infinity_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// Infinity indicator for A (1 if A = O, 0 otherwise)
    pub a_is_infinity_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// Scalar bit polynomials derived from public inputs (not opened/claimed)
    pub bit_public_mlpoly: Vec<MultilinearPolynomial<F>>,

    /// Individual claims for each constraint (not batched)
    pub x_a_claims: Vec<F>,
    pub y_a_claims: Vec<F>,
    pub x_t_claims: Vec<F>,
    pub y_t_claims: Vec<F>,
    pub x_a_next_claims: Vec<F>,
    pub y_a_next_claims: Vec<F>,
    pub t_is_infinity_claims: Vec<F>,
    pub a_is_infinity_claims: Vec<F>,

    /// Current round
    pub round: usize,

    pub _marker: std::marker::PhantomData<T>,
}

impl<F: JoltField, T: Transcript> G1ScalarMulProver<F, T> {
    pub fn new(
        params: G1ScalarMulParams,
        constraint_polys: Vec<G1ScalarMulConstraintPolynomials>,
        public_inputs: Vec<G1ScalarMulPublicInputs>,
        transcript: &mut T,
    ) -> Self {
        let r_x: Vec<F::Challenge> = (0..params.num_constraint_vars)
            .map(|_| transcript.challenge_scalar_optimized::<F>())
            .collect();

        let gamma = transcript.challenge_scalar_optimized::<F>();
        let delta = transcript.challenge_scalar_optimized::<F>();

        // Runtime check that F = Fq for G1 scalar multiplication
        use std::any::TypeId;
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("G1 scalar multiplication requires F = Fq for recursion SNARK");
        }

        let eq_x = MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_x));
        let mut base_points = Vec::new();
        let mut constraint_indices = Vec::new();
        let mut x_a_mlpoly = Vec::new();
        let mut y_a_mlpoly = Vec::new();
        let mut x_t_mlpoly = Vec::new();
        let mut y_t_mlpoly = Vec::new();
        let mut x_a_next_mlpoly = Vec::new();
        let mut y_a_next_mlpoly = Vec::new();
        let mut t_is_infinity_mlpoly = Vec::new();
        let mut a_is_infinity_mlpoly = Vec::new();
        let mut bit_public_mlpoly = Vec::new();

        assert_eq!(
            constraint_polys.len(),
            public_inputs.len(),
            "G1ScalarMulProver: constraint_polys and public_inputs must have same length"
        );

        for (poly, pub_in) in constraint_polys.into_iter().zip(public_inputs.iter()) {
            base_points.push(poly.base_point);
            constraint_indices.push(poly.constraint_index);
            // SAFETY: We checked F = Fq above, so this transmute is safe
            let x_a_f: Vec<F> = unsafe { std::mem::transmute(poly.x_a) };
            x_a_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                x_a_f,
            )));
            let y_a_f: Vec<F> = unsafe { std::mem::transmute(poly.y_a) };
            y_a_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                y_a_f,
            )));
            let x_t_f: Vec<F> = unsafe { std::mem::transmute(poly.x_t) };
            x_t_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                x_t_f,
            )));
            let y_t_f: Vec<F> = unsafe { std::mem::transmute(poly.y_t) };
            y_t_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                y_t_f,
            )));
            let x_a_next_f: Vec<F> = unsafe { std::mem::transmute(poly.x_a_next) };
            x_a_next_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                x_a_next_f,
            )));
            let y_a_next_f: Vec<F> = unsafe { std::mem::transmute(poly.y_a_next) };
            y_a_next_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                y_a_next_f,
            )));
            let t_is_infinity_f: Vec<F> = unsafe { std::mem::transmute(poly.t_is_infinity) };
            t_is_infinity_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                t_is_infinity_f,
            )));
            let a_is_infinity_f: Vec<F> = unsafe { std::mem::transmute(poly.a_is_infinity) };
            a_is_infinity_mlpoly.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                a_is_infinity_f,
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
            x_a_mlpoly,
            y_a_mlpoly,
            x_t_mlpoly,
            y_t_mlpoly,
            x_a_next_mlpoly,
            y_a_next_mlpoly,
            t_is_infinity_mlpoly,
            a_is_infinity_mlpoly,
            bit_public_mlpoly,
            x_a_claims: vec![],
            y_a_claims: vec![],
            x_t_claims: vec![],
            y_t_claims: vec![],
            x_a_next_claims: vec![],
            y_a_next_claims: vec![],
            t_is_infinity_claims: vec![],
            a_is_infinity_claims: vec![],
            round: 0,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for G1ScalarMulProver<F, T> {
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

    #[tracing::instrument(skip_all, name = "G1ScalarMul::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 6;
        const NUM_CONSTRAINT_TERMS: usize = 7; // C1,C2,C3,C4,C5,C6_x,C6_y
        let num_x_remaining = self.eq_x.get_num_vars();
        let x_half = 1 << (num_x_remaining - 1);

        // Precompute delta powers for batching within an instance.
        let mut delta_pows = [F::zero(); NUM_CONSTRAINT_TERMS];
        delta_pows[0] = F::one();
        for j in 1..NUM_CONSTRAINT_TERMS {
            delta_pows[j] = delta_pows[j - 1] * self.delta;
        }

        let total_evals = (0..x_half)
            .into_par_iter()
            .map(|x_idx| {
                let eq_x_evals = self
                    .eq_x
                    .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                let mut x_evals = [F::zero(); DEGREE];
                let mut gamma_power = self.gamma;

                // For each G1 scalar multiplication instance
                for i in 0..self.params.num_constraints {
                    let x_a_evals = self.x_a_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_a_evals = self.y_a_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let x_t_evals = self.x_t_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_t_evals = self.y_t_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let x_a_next_evals = self.x_a_next_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let y_a_next_evals = self.y_a_next_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let ind_t_evals = self.t_is_infinity_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let ind_a_evals = self.a_is_infinity_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);
                    let bit_evals = self.bit_public_mlpoly[i]
                        .sumcheck_evals_array::<DEGREE>(x_idx, BindingOrder::LowToHigh);

                    let (x_p, y_p) = self.base_points[i];

                    for t in 0..DEGREE {
                        // SAFETY: We checked F = Fq in new(), so these transmutes are safe
                        let x_a_fq: Fq = unsafe { std::mem::transmute_copy(&x_a_evals[t]) };
                        let y_a_fq: Fq = unsafe { std::mem::transmute_copy(&y_a_evals[t]) };
                        let x_t_fq: Fq = unsafe { std::mem::transmute_copy(&x_t_evals[t]) };
                        let y_t_fq: Fq = unsafe { std::mem::transmute_copy(&y_t_evals[t]) };
                        let x_a_next_fq: Fq =
                            unsafe { std::mem::transmute_copy(&x_a_next_evals[t]) };
                        let y_a_next_fq: Fq =
                            unsafe { std::mem::transmute_copy(&y_a_next_evals[t]) };
                        let ind_t_fq: Fq = unsafe { std::mem::transmute_copy(&ind_t_evals[t]) };
                        let ind_a_fq: Fq = unsafe { std::mem::transmute_copy(&ind_a_evals[t]) };
                        let bit_fq: Fq = unsafe { std::mem::transmute_copy(&bit_evals[t]) };

                        // Compute constraints
                        let c1_fq = compute_c1(x_a_fq, y_a_fq, x_t_fq);
                        let c2_fq = compute_c2(x_a_fq, y_a_fq, x_t_fq, y_t_fq);
                        let c3_fq =
                            compute_c3(bit_fq, ind_t_fq, x_a_next_fq, x_t_fq, y_t_fq, x_p, y_p);
                        let c4_fq = compute_c4(
                            bit_fq,
                            ind_t_fq,
                            x_a_next_fq,
                            y_a_next_fq,
                            x_t_fq,
                            y_t_fq,
                            x_p,
                            y_p,
                        );
                        let c5_fq = compute_c5(ind_a_fq, ind_t_fq);
                        let c6_x_fq = compute_c6_x(ind_t_fq, x_t_fq);
                        let c6_y_fq = compute_c6_y(ind_t_fq, y_t_fq);

                        // Convert results back to F
                        let c1: F = unsafe { std::mem::transmute_copy(&c1_fq) };
                        let c2: F = unsafe { std::mem::transmute_copy(&c2_fq) };
                        let c3: F = unsafe { std::mem::transmute_copy(&c3_fq) };
                        let c4: F = unsafe { std::mem::transmute_copy(&c4_fq) };
                        let c5: F = unsafe { std::mem::transmute_copy(&c5_fq) };
                        let c6_x: F = unsafe { std::mem::transmute_copy(&c6_x_fq) };
                        let c6_y: F = unsafe { std::mem::transmute_copy(&c6_y_fq) };

                        // Batch constraints with powers of delta
                        let constraint_val = delta_pows[0] * c1
                            + delta_pows[1] * c2
                            + delta_pows[2] * c3
                            + delta_pows[3] * c4
                            + delta_pows[4] * c5
                            + delta_pows[5] * c6_x
                            + delta_pows[6] * c6_y;

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

    #[tracing::instrument(skip_all, name = "G1ScalarMul::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.eq_x.bind_parallel(r_j, BindingOrder::LowToHigh);

        for poly in &mut self.x_a_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_a_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.x_t_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_t_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.x_a_next_mlpoly {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        for poly in &mut self.y_a_next_mlpoly {
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
            self.x_a_claims.clear();
            self.y_a_claims.clear();
            self.x_t_claims.clear();
            self.y_t_claims.clear();
            self.x_a_next_claims.clear();
            self.y_a_next_claims.clear();
            self.t_is_infinity_claims.clear();
            self.a_is_infinity_claims.clear();

            for i in 0..self.params.num_constraints {
                self.x_a_claims.push(self.x_a_mlpoly[i].get_bound_coeff(0));
                self.y_a_claims.push(self.y_a_mlpoly[i].get_bound_coeff(0));
                self.x_t_claims.push(self.x_t_mlpoly[i].get_bound_coeff(0));
                self.y_t_claims.push(self.y_t_mlpoly[i].get_bound_coeff(0));
                self.x_a_next_claims
                    .push(self.x_a_next_mlpoly[i].get_bound_coeff(0));
                self.y_a_next_claims
                    .push(self.y_a_next_mlpoly[i].get_bound_coeff(0));
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
            append_g1_scalar_mul_virtual_claims(
                accumulator,
                transcript,
                i,
                self.params.sumcheck_id,
                &opening_point,
                self.x_a_claims[i],
                self.y_a_claims[i],
                self.x_t_claims[i],
                self.y_t_claims[i],
                self.x_a_next_claims[i],
                self.y_a_next_claims[i],
                self.t_is_infinity_claims[i],
                self.a_is_infinity_claims[i],
            );
        }
    }
}

/// Verifier for G1 scalar multiplication sumcheck
pub struct G1ScalarMulVerifier<F: JoltField> {
    pub params: G1ScalarMulParams,
    pub r_x: Vec<F::Challenge>,
    pub gamma: F,
    pub delta: F,
    pub num_constraints: usize,
    pub base_points: Vec<(Fq, Fq)>, // Base points must be Fq as G1 points
    pub constraint_indices: Vec<usize>,
    pub public_inputs: Vec<G1ScalarMulPublicInputs>,
}

impl<F: JoltField> G1ScalarMulVerifier<F> {
    pub fn new<T: Transcript>(
        params: G1ScalarMulParams,
        base_points: Vec<(Fq, Fq)>,
        constraint_indices: Vec<usize>,
        public_inputs: Vec<G1ScalarMulPublicInputs>,
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

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for G1ScalarMulVerifier<F> {
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
            let (
                x_a_claim,
                y_a_claim,
                x_t_claim,
                y_t_claim,
                x_a_next_claim,
                y_a_next_claim,
                t_is_infinity_claim,
                a_is_infinity_claim,
            ) = get_g1_scalar_mul_virtual_claims(accumulator, i, self.params.sumcheck_id);

            let (x_p, y_p) = self.base_points[i];

            // Runtime check that F = Fq for G1 scalar multiplication
            use std::any::TypeId;
            if TypeId::of::<F>() != TypeId::of::<Fq>() {
                panic!("G1 scalar multiplication requires F = Fq for recursion SNARK");
            }

            // SAFETY: We checked F = Fq above, so these transmutes are safe
            let x_a_fq: Fq = unsafe { std::mem::transmute_copy(&x_a_claim) };
            let y_a_fq: Fq = unsafe { std::mem::transmute_copy(&y_a_claim) };
            let x_t_fq: Fq = unsafe { std::mem::transmute_copy(&x_t_claim) };
            let y_t_fq: Fq = unsafe { std::mem::transmute_copy(&y_t_claim) };
            let x_a_next_fq: Fq = unsafe { std::mem::transmute_copy(&x_a_next_claim) };
            let y_a_next_fq: Fq = unsafe { std::mem::transmute_copy(&y_a_next_claim) };
            let ind_t_fq: Fq = unsafe { std::mem::transmute_copy(&t_is_infinity_claim) };
            let ind_a_fq: Fq = unsafe { std::mem::transmute_copy(&a_is_infinity_claim) };
            let bit_eval: F = self.public_inputs[i].evaluate_bit_mle(&r_star_f);
            let bit_fq: Fq = unsafe { std::mem::transmute_copy(&bit_eval) };

            // Compute constraints
            let c1_fq = compute_c1(x_a_fq, y_a_fq, x_t_fq);
            let c2_fq = compute_c2(x_a_fq, y_a_fq, x_t_fq, y_t_fq);
            let c3_fq = compute_c3(bit_fq, ind_t_fq, x_a_next_fq, x_t_fq, y_t_fq, x_p, y_p);
            let c4_fq = compute_c4(
                bit_fq,
                ind_t_fq,
                x_a_next_fq,
                y_a_next_fq,
                x_t_fq,
                y_t_fq,
                x_p,
                y_p,
            );
            let c5_fq = compute_c5(ind_a_fq, ind_t_fq);
            let c6_x_fq = compute_c6_x(ind_t_fq, x_t_fq);
            let c6_y_fq = compute_c6_y(ind_t_fq, y_t_fq);

            // Convert results back to F
            let c1: F = unsafe { std::mem::transmute_copy(&c1_fq) };
            let c2: F = unsafe { std::mem::transmute_copy(&c2_fq) };
            let c3: F = unsafe { std::mem::transmute_copy(&c3_fq) };
            let c4: F = unsafe { std::mem::transmute_copy(&c4_fq) };
            let c5: F = unsafe { std::mem::transmute_copy(&c5_fq) };
            let c6_x: F = unsafe { std::mem::transmute_copy(&c6_x_fq) };
            let c6_y: F = unsafe { std::mem::transmute_copy(&c6_y_fq) };

            // Batch constraints with powers of delta
            let delta_2 = self.delta * self.delta;
            let delta_3 = delta_2 * self.delta;
            let delta_4 = delta_3 * self.delta;
            let delta_5 = delta_4 * self.delta;
            let delta_6 = delta_5 * self.delta;

            let constraint_value = c1
                + self.delta * c2
                + delta_2 * c3
                + delta_3 * c4
                + delta_4 * c5
                + delta_5 * c6_x
                + delta_6 * c6_y;

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
            append_g1_scalar_mul_virtual_openings(
                accumulator,
                transcript,
                i, // Use local index, not global constraint index
                self.params.sumcheck_id,
                &opening_point,
            );
        }
    }
}
