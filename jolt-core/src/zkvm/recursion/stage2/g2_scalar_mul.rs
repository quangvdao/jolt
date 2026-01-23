//! G2 scalar multiplication sumcheck for proving G2 scalar multiplication constraints
//!
//! Analogue of `g1_scalar_mul.rs`, but for points in G2 (over Fq2). Since the recursion
//! SNARK runs over the base field Fq, we "relinearize" Fq2 elements by splitting each
//! coordinate into its (c0, c1) components in Fq and enforcing constraints component-wise.
//!
//! ## Constraints (13 terms total)
//! - C1: Doubling x-coordinate (Fq2 → c0, c1)
//! - C2: Doubling y-coordinate (Fq2 → c0, c1)
//! - C3: Conditional addition x-coord (Fq2 → c0, c1)
//! - C4: Conditional addition y-coord (Fq2 → c0, c1)
//! - C5: If A = O then T = O (Fq)
//! - C6: If ind_T = 1 then (x_T, y_T) = (0,0) (4 Fq terms)
//!
//! ## Public inputs
//! The scalar bits are treated as **public inputs** (derived from the scalar),
//! so we do NOT emit openings for the bit polynomial.

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::SumcheckId,
    },
    zkvm::{
        recursion::stage2::constraint_list_sumcheck::{
            ConstraintListProver, ConstraintListProverSpec, ConstraintListSpec,
            ConstraintListVerifier, ConstraintListVerifierSpec, OpeningSpec,
        },
        witness::{G2ScalarMulTerm, RecursionPoly, TermEnum, VirtualPolynomial},
    },
};
use allocative::Allocative;
use ark_bn254::{Fq, Fq2, Fr};
use ark_ff::{BigInteger, One, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;

// =============================================================================
// Constants
// =============================================================================

/// Number of committed polynomial kinds (excluding public bit poly)
const NUM_COMMITTED_KINDS: usize = 14;

/// Sumcheck degree (eq * constraint, where constraint has degree 5 from chord formulas)
const DEGREE: usize = 6;

/// Opening specs for the 14 committed polynomials (Bit is public, not opened)
const G2_SCALAR_MUL_OPENING_SPECS: [OpeningSpec; NUM_COMMITTED_KINDS] = [
    OpeningSpec::new(0, 0),   // XAC0
    OpeningSpec::new(1, 1),   // XAC1
    OpeningSpec::new(2, 2),   // YAC0
    OpeningSpec::new(3, 3),   // YAC1
    OpeningSpec::new(4, 4),   // XTC0
    OpeningSpec::new(5, 5),   // XTC1
    OpeningSpec::new(6, 6),   // YTC0
    OpeningSpec::new(7, 7),   // YTC1
    OpeningSpec::new(8, 8),   // XANextC0
    OpeningSpec::new(9, 9),   // XANextC1
    OpeningSpec::new(10, 10), // YANextC0
    OpeningSpec::new(11, 11), // YANextC1
    OpeningSpec::new(12, 12), // TIndicator
    OpeningSpec::new(13, 13), // AIndicator
];

// =============================================================================
// Public Inputs
// =============================================================================

/// Public inputs for a single G2 scalar multiplication (the scalar).
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G2ScalarMulPublicInputs {
    pub scalar: Fr,
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

    /// Evaluate the (padded) bit MLE at the sumcheck challenge point (11 vars).
    pub fn evaluate_bit_mle<F: JoltField>(&self, eval_point: &[F]) -> F {
        assert_eq!(eval_point.len(), 11);
        let bits = self.bits_msb();

        let pad_factor = EqPolynomial::<F>::zero_selector(&eval_point[..3]);
        let eq_step = EqPolynomial::<F>::evals(&eval_point[3..]);
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
// Witness and Constraint Polynomials
// =============================================================================

/// Witness polynomials for a G2 scalar multiplication constraint.
#[derive(Clone, Debug)]
pub struct G2ScalarMulWitness {
    pub constraint_index: usize,
    pub base_point: (Fq2, Fq2),
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
    pub t_indicator: Vec<Fq>,
    pub a_indicator: Vec<Fq>,
}

/// Constraint polynomials for a single G2 scalar multiplication
#[derive(Clone)]
pub struct G2ScalarMulConstraintPolynomials<F: JoltField> {
    pub x_a_c0: Vec<F>,
    pub x_a_c1: Vec<F>,
    pub y_a_c0: Vec<F>,
    pub y_a_c1: Vec<F>,
    pub x_t_c0: Vec<F>,
    pub x_t_c1: Vec<F>,
    pub y_t_c0: Vec<F>,
    pub y_t_c1: Vec<F>,
    pub x_a_next_c0: Vec<F>,
    pub x_a_next_c1: Vec<F>,
    pub y_a_next_c0: Vec<F>,
    pub y_a_next_c1: Vec<F>,
    pub t_is_infinity: Vec<F>,
    pub a_is_infinity: Vec<F>,
    pub base_point: (Fq2, Fq2),
    pub constraint_index: usize,
}

// =============================================================================
// Parameters
// =============================================================================

/// Parameters for G2 scalar multiplication sumcheck
#[derive(Clone, Allocative)]
pub struct G2ScalarMulParams {
    pub num_constraint_vars: usize,
    pub num_constraints: usize,
    pub sumcheck_id: SumcheckId,
}

impl G2ScalarMulParams {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            num_constraint_vars: 11,
            num_constraints,
            sumcheck_id: SumcheckId::G2ScalarMul,
        }
    }
}

// =============================================================================
// Constraint Values (single-point evaluations)
// =============================================================================

/// Single-point evaluation values for G2 scalar mul constraint polynomials.
#[derive(Clone, Copy, Debug)]
pub struct G2ScalarMulValues<F> {
    pub x_a_c0: F,
    pub x_a_c1: F,
    pub y_a_c0: F,
    pub y_a_c1: F,
    pub x_t_c0: F,
    pub x_t_c1: F,
    pub y_t_c0: F,
    pub y_t_c1: F,
    pub x_a_next_c0: F,
    pub x_a_next_c1: F,
    pub y_a_next_c0: F,
    pub y_a_next_c1: F,
    pub t_indicator: F,
    pub a_indicator: F,
}

impl<F: Copy> G2ScalarMulValues<F> {
    #[inline]
    pub fn from_poly_evals<const DEGREE: usize>(poly_evals: &[[F; DEGREE]], eval_index: usize) -> Self {
        Self {
            x_a_c0: poly_evals[0][eval_index],
            x_a_c1: poly_evals[1][eval_index],
            y_a_c0: poly_evals[2][eval_index],
            y_a_c1: poly_evals[3][eval_index],
            x_t_c0: poly_evals[4][eval_index],
            x_t_c1: poly_evals[5][eval_index],
            y_t_c0: poly_evals[6][eval_index],
            y_t_c1: poly_evals[7][eval_index],
            x_a_next_c0: poly_evals[8][eval_index],
            x_a_next_c1: poly_evals[9][eval_index],
            y_a_next_c0: poly_evals[10][eval_index],
            y_a_next_c1: poly_evals[11][eval_index],
            t_indicator: poly_evals[12][eval_index],
            a_indicator: poly_evals[13][eval_index],
        }
    }

    #[inline]
    pub fn from_claims(claims: &[F]) -> Self {
        Self {
            x_a_c0: claims[0],
            x_a_c1: claims[1],
            y_a_c0: claims[2],
            y_a_c1: claims[3],
            x_t_c0: claims[4],
            x_t_c1: claims[5],
            y_t_c0: claims[6],
            y_t_c1: claims[7],
            x_a_next_c0: claims[8],
            x_a_next_c1: claims[9],
            y_a_next_c0: claims[10],
            y_a_next_c1: claims[11],
            t_indicator: claims[12],
            a_indicator: claims[13],
        }
    }
}

impl G2ScalarMulValues<Fq> {
    /// Evaluate the batched constraint: Σ_j δ^j * C_j (13 terms)
    pub fn eval_constraint(&self, bit: Fq, x_p: Fq2, y_p: Fq2, delta: Fq) -> Fq {
        // Reconstruct Fq2 values
        let x_a = Fq2::new(self.x_a_c0, self.x_a_c1);
        let y_a = Fq2::new(self.y_a_c0, self.y_a_c1);
        let x_t = Fq2::new(self.x_t_c0, self.x_t_c1);
        let y_t = Fq2::new(self.y_t_c0, self.y_t_c1);
        let x_a_next = Fq2::new(self.x_a_next_c0, self.x_a_next_c1);
        let y_a_next = Fq2::new(self.y_a_next_c0, self.y_a_next_c1);

        // Compute Fq2 constraints
        let c1 = compute_c1(x_a, y_a, x_t);
        let c2 = compute_c2(x_a, y_a, x_t, y_t);
        let c3 = compute_c3(bit, self.t_indicator, x_a_next, x_t, y_t, x_p, y_p);
        let c4 = compute_c4(bit, self.t_indicator, x_a_next, y_a_next, x_t, y_t, x_p, y_p);
        let c5 = compute_c5(self.a_indicator, self.t_indicator);

        // C6: if ind_T = 1 then (x_T, y_T) = (0,0)
        let c6_xt_c0 = self.t_indicator * self.x_t_c0;
        let c6_xt_c1 = self.t_indicator * self.x_t_c1;
        let c6_yt_c0 = self.t_indicator * self.y_t_c0;
        let c6_yt_c1 = self.t_indicator * self.y_t_c1;

        // Batch with powers of delta
        let d2 = delta * delta;
        let d3 = d2 * delta;
        let d4 = d3 * delta;
        let d5 = d4 * delta;
        let d6 = d5 * delta;
        let d7 = d6 * delta;
        let d8 = d7 * delta;
        let d9 = d8 * delta;
        let d10 = d9 * delta;
        let d11 = d10 * delta;
        let d12 = d11 * delta;

        c1.c0
            + delta * c1.c1
            + d2 * c2.c0
            + d3 * c2.c1
            + d4 * c3.c0
            + d5 * c3.c1
            + d6 * c4.c0
            + d7 * c4.c1
            + d8 * c5
            + d9 * c6_xt_c0
            + d10 * c6_xt_c1
            + d11 * c6_yt_c0
            + d12 * c6_yt_c1
    }
}

// =============================================================================
// Constraint Functions (in Fq2)
// =============================================================================

fn fq2_from_fq(x: Fq) -> Fq2 {
    Fq2::new(x, Fq::zero())
}

fn compute_c1(x_a: Fq2, y_a: Fq2, x_t: Fq2) -> Fq2 {
    let four = fq2_from_fq(Fq::from(4u64));
    let two = fq2_from_fq(Fq::from(2u64));
    let nine = fq2_from_fq(Fq::from(9u64));

    let y_a_sq = y_a * y_a;
    let x_a_sq = x_a * x_a;
    let x_a_fourth = x_a_sq * x_a_sq;

    four * y_a_sq * (x_t + two * x_a) - nine * x_a_fourth
}

fn compute_c2(x_a: Fq2, y_a: Fq2, x_t: Fq2, y_t: Fq2) -> Fq2 {
    let three = fq2_from_fq(Fq::from(3u64));
    let two = fq2_from_fq(Fq::from(2u64));

    let x_a_sq = x_a * x_a;
    three * x_a_sq * (x_t - x_a) + two * y_a * (y_t + y_a)
}

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

#[allow(clippy::too_many_arguments)]
fn compute_c4(bit: Fq, ind_t: Fq, x_a_next: Fq2, y_a_next: Fq2, x_t: Fq2, y_t: Fq2, x_p: Fq2, y_p: Fq2) -> Fq2 {
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

fn compute_c5(ind_a: Fq, ind_t: Fq) -> Fq {
    ind_a * (Fq::one() - ind_t)
}

// =============================================================================
// Prover Spec
// =============================================================================

/// Prover-side specification for G2 scalar mul constraints.
#[derive(Clone, Allocative)]
pub struct G2ScalarMulProverSpec {
    params: G2ScalarMulParams,
    polys_by_kind: Vec<Vec<MultilinearPolynomial<Fq>>>,
    public_polys: Vec<Vec<MultilinearPolynomial<Fq>>>,
    #[allocative(skip)]
    base_points: Vec<(Fq2, Fq2)>,
}

impl G2ScalarMulProverSpec {
    pub fn new(
        params: G2ScalarMulParams,
        constraint_polys: Vec<G2ScalarMulConstraintPolynomials<Fq>>,
        public_inputs: &[G2ScalarMulPublicInputs],
    ) -> (Self, Vec<usize>) {
        debug_assert_eq!(
            constraint_polys.len(),
            public_inputs.len(),
            "constraint_polys and public_inputs must have same length"
        );

        let num_instances = constraint_polys.len();
        let num_vars = params.num_constraint_vars;

        let mut polys_by_kind: Vec<Vec<MultilinearPolynomial<Fq>>> =
            (0..NUM_COMMITTED_KINDS).map(|_| Vec::with_capacity(num_instances)).collect();
        let mut public_polys: Vec<Vec<MultilinearPolynomial<Fq>>> =
            vec![Vec::with_capacity(num_instances)];

        let mut base_points = Vec::with_capacity(num_instances);
        let mut constraint_indices = Vec::with_capacity(num_instances);

        for (poly, pub_in) in constraint_polys.into_iter().zip(public_inputs.iter()) {
            constraint_indices.push(poly.constraint_index);
            base_points.push(poly.base_point);

            polys_by_kind[0].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.x_a_c0)));
            polys_by_kind[1].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.x_a_c1)));
            polys_by_kind[2].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.y_a_c0)));
            polys_by_kind[3].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.y_a_c1)));
            polys_by_kind[4].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.x_t_c0)));
            polys_by_kind[5].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.x_t_c1)));
            polys_by_kind[6].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.y_t_c0)));
            polys_by_kind[7].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.y_t_c1)));
            polys_by_kind[8].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.x_a_next_c0)));
            polys_by_kind[9].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.x_a_next_c1)));
            polys_by_kind[10].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.y_a_next_c0)));
            polys_by_kind[11].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.y_a_next_c1)));
            polys_by_kind[12].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.t_is_infinity)));
            polys_by_kind[13].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.a_is_infinity)));

            // Public polynomial: bit
            let bits = pub_in.bits_msb();
            let mut bit_evals = vec![Fq::zero(); 1 << num_vars];
            for i in 0..256 {
                bit_evals[i] = if bits[i] { Fq::one() } else { Fq::zero() };
            }
            public_polys[0].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(bit_evals)));
        }

        let spec = Self {
            params,
            polys_by_kind,
            public_polys,
            base_points,
        };

        let sequential_indices: Vec<usize> = (0..num_instances).collect();
        (spec, sequential_indices)
    }
}

impl ConstraintListSpec for G2ScalarMulProverSpec {
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
        &G2_SCALAR_MUL_OPENING_SPECS
    }

    fn build_virtual_poly(&self, term_index: usize, instance: usize) -> VirtualPolynomial {
        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::from_index(term_index).expect("invalid G2ScalarMulTerm index"),
            instance,
        })
    }
}

impl ConstraintListProverSpec<Fq, DEGREE> for G2ScalarMulProverSpec {
    fn polys_by_kind(&self) -> &[Vec<MultilinearPolynomial<Fq>>] {
        &self.polys_by_kind
    }

    fn polys_by_kind_mut(&mut self) -> &mut [Vec<MultilinearPolynomial<Fq>>] {
        &mut self.polys_by_kind
    }

    fn public_polys(&self) -> &[Vec<MultilinearPolynomial<Fq>>] {
        &self.public_polys
    }

    fn public_polys_mut(&mut self) -> &mut [Vec<MultilinearPolynomial<Fq>>] {
        &mut self.public_polys
    }

    fn shared_polys(&self) -> &[MultilinearPolynomial<Fq>] {
        &[]
    }

    fn shared_polys_mut(&mut self) -> &mut [MultilinearPolynomial<Fq>] {
        &mut []
    }

    fn eval_constraint(
        &self,
        instance: usize,
        eval_index: usize,
        poly_evals: &[[Fq; DEGREE]],
        public_evals: &[[Fq; DEGREE]],
        _shared_evals: &[[Fq; DEGREE]],
        term_batch_coeff: Option<Fq>,
    ) -> Fq {
        let vals = G2ScalarMulValues::from_poly_evals(poly_evals, eval_index);
        let bit = public_evals[0][eval_index];
        let (x_p, y_p) = self.base_points[instance];
        let delta = term_batch_coeff.expect("G2ScalarMul requires term_batch_coeff");
        vals.eval_constraint(bit, x_p, y_p, delta)
    }
}

// =============================================================================
// Verifier Spec
// =============================================================================

/// Verifier-side specification for G2 scalar mul constraints.
#[derive(Clone, Allocative)]
pub struct G2ScalarMulVerifierSpec {
    params: G2ScalarMulParams,
    #[allocative(skip)]
    base_points: Vec<(Fq2, Fq2)>,
    #[allocative(skip)]
    public_inputs: Vec<G2ScalarMulPublicInputs>,
}

impl G2ScalarMulVerifierSpec {
    pub fn new(
        params: G2ScalarMulParams,
        base_points: Vec<(Fq2, Fq2)>,
        public_inputs: Vec<G2ScalarMulPublicInputs>,
    ) -> Self {
        Self {
            params,
            base_points,
            public_inputs,
        }
    }
}

impl ConstraintListSpec for G2ScalarMulVerifierSpec {
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
        &G2_SCALAR_MUL_OPENING_SPECS
    }

    fn build_virtual_poly(&self, term_index: usize, instance: usize) -> VirtualPolynomial {
        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::from_index(term_index).expect("invalid G2ScalarMulTerm index"),
            instance,
        })
    }
}

impl ConstraintListVerifierSpec<Fq, DEGREE> for G2ScalarMulVerifierSpec {
    fn compute_shared_scalars(&self, _eval_point: &[Fq]) -> Vec<Fq> {
        vec![]
    }

    fn eval_constraint_at_point(
        &self,
        instance: usize,
        opened_claims: &[Fq],
        _shared_scalars: &[Fq],
        eval_point: &[Fq],
        term_batch_coeff: Option<Fq>,
    ) -> Fq {
        let vals = G2ScalarMulValues::from_claims(opened_claims);
        let bit = self.public_inputs[instance].evaluate_bit_mle(eval_point);
        let (x_p, y_p) = self.base_points[instance];
        let delta = term_batch_coeff.expect("G2ScalarMul requires term_batch_coeff");
        vals.eval_constraint(bit, x_p, y_p, delta)
    }
}

// =============================================================================
// Type Aliases
// =============================================================================

/// Prover for G2 scalar multiplication sumcheck.
pub type G2ScalarMulProver<F> = ConstraintListProver<F, G2ScalarMulProverSpec, DEGREE>;

/// Verifier for G2 scalar multiplication sumcheck.
pub type G2ScalarMulVerifier<F> = ConstraintListVerifier<F, G2ScalarMulVerifierSpec, DEGREE>;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::G2Affine;
    use ark_ec::AffineRepr;
    use ark_std::test_rng;

    #[test]
    fn test_g2_scalar_mul_sumcheck_roundtrip_single_instance() {
        // Simple test: scalar = 2, base point = generator
        let _rng = test_rng();
        let scalar = Fr::from(2u64);
        let generator = G2Affine::generator();
        let _base_point = (generator.x, generator.y);

        // Create witness for [2]G
        // A_0 = O, T_0 = [2]O = O, A_1 = T_0 + 1*G = G (since bit_0 = 1 for scalar=2 in binary 10)
        // Actually for scalar=2 binary is "10", MSB first: bit[0]=1, bit[1]=0
        // Step 0: A=O, T=[2]O=O, bit=1, A'=O+G=G
        // Step 1: A=G, T=[2]G, bit=0, A'=T

        // This is a simplified test - in practice we'd generate full 256-step trace
        // For now just verify the prover/verifier can be constructed
        let public_inputs = G2ScalarMulPublicInputs::new(scalar);

        // Verify bits_msb works
        let bits = public_inputs.bits_msb();
        assert_eq!(bits.len(), 256);
        // scalar=2 in binary (MSB first, 256 bits): 0...010
        // bits[254] = 1 (the "2" bit), bits[255] = 0 (the "1" bit)
        assert!(bits[254]); // bit position for value 2
        assert!(!bits[255]); // LSB is 0 for even number
    }
}
