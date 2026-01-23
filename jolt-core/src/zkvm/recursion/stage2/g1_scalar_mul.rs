//! G1 scalar multiplication sumcheck for proving G1 scalar multiplication constraints.
//!
//! ## Constraints (7 terms, batched with delta)
//! - C1: Doubling x-coordinate: 4y_A²(x_T + 2x_A) - 9x_A⁴ = 0
//! - C2: Doubling y-coordinate: 3x_A²(x_T - x_A) + 2y_A(y_T + y_A) = 0
//! - C3: Conditional addition x-coord (bit-dependent)
//! - C4: Conditional addition y-coord (bit-dependent)
//! - C5: If A = O then T = O (infinity preserved)
//! - C6: If ind_T = 1 then (x_T, y_T) = (0,0) [2 terms]
//!
//! ## Public inputs
//! The scalar bits are treated as **public inputs** (derived from the scalar),
//! so we do NOT emit openings for the bit polynomial.

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial, opening_proof::SumcheckId,
    },
    zkvm::{
        recursion::stage2::constraint_list_sumcheck::{
            sequential_opening_specs, ConstraintListProver, ConstraintListProverSpec,
            ConstraintListSpec, ConstraintListVerifier, ConstraintListVerifierSpec, OpeningSpec,
        },
        witness::{G1ScalarMulTerm, RecursionPoly, TermEnum, VirtualPolynomial},
    },
};
use allocative::Allocative;
use ark_bn254::{Fq, Fr};
use ark_ff::{BigInteger, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

// =============================================================================
// Constants
// =============================================================================

const NUM_COMMITTED_KINDS: usize = 8;
const DEGREE: usize = 6;
const OPENING_SPECS: [OpeningSpec; NUM_COMMITTED_KINDS] = sequential_opening_specs();

// =============================================================================
// Data Types
// =============================================================================

/// Public inputs for a single G1 scalar multiplication (the scalar).
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G1ScalarMulPublicInputs {
    pub scalar: Fr,
}

impl G1ScalarMulPublicInputs {
    pub fn new(scalar: Fr) -> Self {
        Self { scalar }
    }

    pub fn bits_msb(&self) -> Vec<bool> {
        let scalar_bits_le = self.scalar.into_bigint().to_bits_le();
        (0..256).rev().map(|i| scalar_bits_le[i]).collect()
    }

    pub fn evaluate_bit_mle<F: JoltField>(&self, eval_point: &[F]) -> F {
        assert_eq!(eval_point.len(), 11);
        let bits = self.bits_msb();
        let pad_factor = EqPolynomial::<F>::zero_selector(&eval_point[..3]);
        let eq_step = EqPolynomial::<F>::evals(&eval_point[3..]);
        let mut acc = F::zero();
        for (i, eq) in eq_step.iter().enumerate() {
            if bits[i] {
                acc += *eq;
            }
        }
        pad_factor * acc
    }

    fn build_bit_poly(&self, num_vars: usize) -> Vec<Fq> {
        let bits = self.bits_msb();
        let mut evals = vec![Fq::zero(); 1 << num_vars];
        for i in 0..256 {
            evals[i] = if bits[i] { Fq::one() } else { Fq::zero() };
        }
        evals
    }
}

/// Constraint polynomials for one G1 scalar mul instance.
#[derive(Clone, Debug, Allocative)]
pub struct G1ScalarMulConstraintPolynomials<F> {
    pub x_a: Vec<F>,
    pub y_a: Vec<F>,
    pub x_t: Vec<F>,
    pub y_t: Vec<F>,
    pub x_a_next: Vec<F>,
    pub y_a_next: Vec<F>,
    pub t_indicator: Vec<F>,
    pub a_indicator: Vec<F>,
    pub constraint_index: usize,
}

/// Parameters for G1 scalar multiplication sumcheck.
#[derive(Clone, Allocative)]
pub struct G1ScalarMulParams {
    pub num_constraint_vars: usize,
    pub num_constraints: usize,
    pub sumcheck_id: SumcheckId,
}

impl G1ScalarMulParams {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            num_constraint_vars: 11,
            num_constraints,
            sumcheck_id: SumcheckId::G1ScalarMul,
        }
    }
}

/// Single-point evaluation values for G1 scalar mul polynomials.
#[derive(Clone, Copy, Debug)]
struct G1ScalarMulValues {
    x_a: Fq,
    y_a: Fq,
    x_t: Fq,
    y_t: Fq,
    x_a_next: Fq,
    y_a_next: Fq,
    t_indicator: Fq,
    a_indicator: Fq,
}

impl G1ScalarMulValues {
    #[inline]
    fn from_poly_evals<const D: usize>(poly_evals: &[[Fq; D]], idx: usize) -> Self {
        Self {
            x_a: poly_evals[0][idx],
            y_a: poly_evals[1][idx],
            x_t: poly_evals[2][idx],
            y_t: poly_evals[3][idx],
            x_a_next: poly_evals[4][idx],
            y_a_next: poly_evals[5][idx],
            t_indicator: poly_evals[6][idx],
            a_indicator: poly_evals[7][idx],
        }
    }

    #[inline]
    fn from_claims(claims: &[Fq]) -> Self {
        Self {
            x_a: claims[0],
            y_a: claims[1],
            x_t: claims[2],
            y_t: claims[3],
            x_a_next: claims[4],
            y_a_next: claims[5],
            t_indicator: claims[6],
            a_indicator: claims[7],
        }
    }

    /// Evaluate batched constraint: Σ_j δ^j * C_j
    fn eval_constraint(&self, bit: Fq, x_p: Fq, y_p: Fq, delta: Fq) -> Fq {
        let one = Fq::one();
        let two = Fq::from(2u64);
        let three = Fq::from(3u64);
        let four = Fq::from(4u64);
        let nine = Fq::from(9u64);

        // C1: 4y_A²(x_T + 2x_A) - 9x_A⁴
        let y_a_sq = self.y_a * self.y_a;
        let x_a_sq = self.x_a * self.x_a;
        let c1 = four * y_a_sq * (self.x_t + two * self.x_a) - nine * x_a_sq * x_a_sq;

        // C2: 3x_A²(x_T - x_A) + 2y_A(y_T + y_A)
        let c2 = three * x_a_sq * (self.x_t - self.x_a) + two * self.y_a * (self.y_t + self.y_a);

        // C3: Conditional addition x-coord
        let c3_skip = (one - bit) * (self.x_a_next - self.x_t);
        let c3_infinity = bit * self.t_indicator * (self.x_a_next - x_p);
        let x_diff = x_p - self.x_t;
        let y_diff = y_p - self.y_t;
        let chord_x = (self.x_a_next + self.x_t + x_p) * x_diff * x_diff - y_diff * y_diff;
        let c3_add = bit * (one - self.t_indicator) * chord_x;
        let c3 = c3_skip + c3_infinity + c3_add;

        // C4: Conditional addition y-coord
        let c4_skip = (one - bit) * (self.y_a_next - self.y_t);
        let c4_infinity = bit * self.t_indicator * (self.y_a_next - y_p);
        let chord_y = (self.y_a_next + self.y_t) * x_diff - y_diff * (self.x_t - self.x_a_next);
        let c4_add = bit * (one - self.t_indicator) * chord_y;
        let c4 = c4_skip + c4_infinity + c4_add;

        // C5: ind_A * (1 - ind_T)
        let c5 = self.a_indicator * (one - self.t_indicator);

        // C6: ind_T * x_T, ind_T * y_T
        let c6_x = self.t_indicator * self.x_t;
        let c6_y = self.t_indicator * self.y_t;

        // Batch with powers of delta
        let d2 = delta * delta;
        let d3 = d2 * delta;
        let d4 = d3 * delta;
        let d5 = d4 * delta;
        let d6 = d5 * delta;

        c1 + delta * c2 + d2 * c3 + d3 * c4 + d4 * c5 + d5 * c6_x + d6 * c6_y
    }
}

// =============================================================================
// Prover Spec
// =============================================================================

#[derive(Clone, Allocative)]
pub struct G1ScalarMulProverSpec {
    params: G1ScalarMulParams,
    polys_by_kind: Vec<Vec<MultilinearPolynomial<Fq>>>,
    public_polys: Vec<Vec<MultilinearPolynomial<Fq>>>,
    #[allocative(skip)]
    base_points: Vec<(Fq, Fq)>,
}

impl G1ScalarMulProverSpec {
    pub fn new(
        params: G1ScalarMulParams,
        constraint_polys: Vec<G1ScalarMulConstraintPolynomials<Fq>>,
        public_inputs: &[G1ScalarMulPublicInputs],
        base_points: Vec<(Fq, Fq)>,
    ) -> (Self, Vec<usize>) {
        let num_instances = constraint_polys.len();
        let num_vars = params.num_constraint_vars;

        let mut polys_by_kind: Vec<Vec<MultilinearPolynomial<Fq>>> =
            (0..NUM_COMMITTED_KINDS).map(|_| Vec::with_capacity(num_instances)).collect();
        let mut public_polys = vec![Vec::with_capacity(num_instances)];
        let mut constraint_indices = Vec::with_capacity(num_instances);

        for (poly, pub_in) in constraint_polys.into_iter().zip(public_inputs.iter()) {
            constraint_indices.push(poly.constraint_index);

            polys_by_kind[0].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.x_a)));
            polys_by_kind[1].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.y_a)));
            polys_by_kind[2].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.x_t)));
            polys_by_kind[3].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.y_t)));
            polys_by_kind[4].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.x_a_next)));
            polys_by_kind[5].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.y_a_next)));
            polys_by_kind[6].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.t_indicator)));
            polys_by_kind[7].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.a_indicator)));

            public_polys[0].push(MultilinearPolynomial::LargeScalars(
                DensePolynomial::new(pub_in.build_bit_poly(num_vars))
            ));
        }

        let sequential_indices: Vec<usize> = (0..num_instances).collect();
        (Self { params, polys_by_kind, public_polys, base_points }, sequential_indices)
    }
}

impl ConstraintListSpec for G1ScalarMulProverSpec {
    fn sumcheck_id(&self) -> SumcheckId { self.params.sumcheck_id }
    fn num_rounds(&self) -> usize { self.params.num_constraint_vars }
    fn num_instances(&self) -> usize { self.params.num_constraints }
    fn uses_term_batching(&self) -> bool { true }
    fn opening_specs(&self) -> &'static [OpeningSpec] { &OPENING_SPECS }

    fn build_virtual_poly(&self, term_index: usize, instance: usize) -> VirtualPolynomial {
        VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
            term: G1ScalarMulTerm::from_index(term_index).expect("invalid term index"),
            instance,
        })
    }
}

impl ConstraintListProverSpec<Fq, DEGREE> for G1ScalarMulProverSpec {
    fn polys_by_kind(&self) -> &[Vec<MultilinearPolynomial<Fq>>] { &self.polys_by_kind }
    fn polys_by_kind_mut(&mut self) -> &mut [Vec<MultilinearPolynomial<Fq>>] { &mut self.polys_by_kind }
    fn public_polys(&self) -> &[Vec<MultilinearPolynomial<Fq>>] { &self.public_polys }
    fn public_polys_mut(&mut self) -> &mut [Vec<MultilinearPolynomial<Fq>>] { &mut self.public_polys }
    fn shared_polys(&self) -> &[MultilinearPolynomial<Fq>] { &[] }
    fn shared_polys_mut(&mut self) -> &mut [MultilinearPolynomial<Fq>] { &mut [] }

    fn eval_constraint(
        &self,
        instance: usize,
        eval_index: usize,
        poly_evals: &[[Fq; DEGREE]],
        public_evals: &[[Fq; DEGREE]],
        _shared_evals: &[[Fq; DEGREE]],
        term_batch_coeff: Option<Fq>,
    ) -> Fq {
        let vals = G1ScalarMulValues::from_poly_evals(poly_evals, eval_index);
        let bit = public_evals[0][eval_index];
        let (x_p, y_p) = self.base_points[instance];
        let delta = term_batch_coeff.expect("requires term_batch_coeff");
        vals.eval_constraint(bit, x_p, y_p, delta)
    }
}

// =============================================================================
// Verifier Spec
// =============================================================================

#[derive(Clone, Allocative)]
pub struct G1ScalarMulVerifierSpec {
    params: G1ScalarMulParams,
    #[allocative(skip)]
    public_inputs: Vec<G1ScalarMulPublicInputs>,
    #[allocative(skip)]
    base_points: Vec<(Fq, Fq)>,
}

impl G1ScalarMulVerifierSpec {
    pub fn new(
        params: G1ScalarMulParams,
        public_inputs: Vec<G1ScalarMulPublicInputs>,
        base_points: Vec<(Fq, Fq)>,
    ) -> Self {
        Self { params, public_inputs, base_points }
    }
}

impl ConstraintListSpec for G1ScalarMulVerifierSpec {
    fn sumcheck_id(&self) -> SumcheckId { self.params.sumcheck_id }
    fn num_rounds(&self) -> usize { self.params.num_constraint_vars }
    fn num_instances(&self) -> usize { self.params.num_constraints }
    fn uses_term_batching(&self) -> bool { true }
    fn opening_specs(&self) -> &'static [OpeningSpec] { &OPENING_SPECS }

    fn build_virtual_poly(&self, term_index: usize, instance: usize) -> VirtualPolynomial {
        VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
            term: G1ScalarMulTerm::from_index(term_index).expect("invalid term index"),
            instance,
        })
    }
}

impl ConstraintListVerifierSpec<Fq, DEGREE> for G1ScalarMulVerifierSpec {
    fn compute_shared_scalars(&self, _eval_point: &[Fq]) -> Vec<Fq> { vec![] }

    fn eval_constraint_at_point(
        &self,
        instance: usize,
        opened_claims: &[Fq],
        _shared_scalars: &[Fq],
        eval_point: &[Fq],
        term_batch_coeff: Option<Fq>,
    ) -> Fq {
        let vals = G1ScalarMulValues::from_claims(opened_claims);
        let bit = self.public_inputs[instance].evaluate_bit_mle(eval_point);
        let (x_p, y_p) = self.base_points[instance];
        let delta = term_batch_coeff.expect("requires term_batch_coeff");
        vals.eval_constraint(bit, x_p, y_p, delta)
    }
}

// =============================================================================
// Type Aliases
// =============================================================================

pub type G1ScalarMulProver<F> = ConstraintListProver<F, G1ScalarMulProverSpec, DEGREE>;
pub type G1ScalarMulVerifier<F> = ConstraintListVerifier<F, G1ScalarMulVerifierSpec, DEGREE>;
