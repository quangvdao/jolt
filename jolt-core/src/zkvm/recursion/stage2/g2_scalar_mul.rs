//! G2 scalar multiplication sumcheck for proving G2 scalar multiplication constraints.
//!
//! Analogue of `g1_scalar_mul.rs`, but for points in G2 (over Fq2). Since the recursion
//! SNARK runs over the base field Fq, we split each Fq2 coordinate into (c0, c1) components.
//!
//! ## Constraints (13 terms, batched with delta)
//! - C1: Doubling x-coordinate (Fq2 → c0, c1)
//! - C2: Doubling y-coordinate (Fq2 → c0, c1)
//! - C3: Conditional addition x-coord (Fq2 → c0, c1)
//! - C4: Conditional addition y-coord (Fq2 → c0, c1)
//! - C5: If A = O then T = O (Fq)
//! - C6: If ind_T = 1 then (x_T, y_T) = (0,0) (4 Fq terms)
//!
//! ## Public inputs
//! The scalar bits are treated as **public inputs**.

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
        witness::{G2ScalarMulTerm, RecursionPoly, TermEnum, VirtualPolynomial},
    },
};
use allocative::Allocative;
use ark_bn254::{Fq, Fq2, Fr};
use ark_ff::{BigInteger, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

// =============================================================================
// Constants
// =============================================================================

const NUM_COMMITTED_KINDS: usize = 14;
const DEGREE: usize = 6;
const OPENING_SPECS: [OpeningSpec; NUM_COMMITTED_KINDS] = sequential_opening_specs();

// =============================================================================
// Data Types
// =============================================================================

/// Public inputs for a single G2 scalar multiplication.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G2ScalarMulPublicInputs {
    pub scalar: Fr,
}

impl G2ScalarMulPublicInputs {
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

/// Constraint polynomials for one G2 scalar mul instance.
#[derive(Clone, Debug, Allocative)]
pub struct G2ScalarMulConstraintPolynomials<F> {
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
    pub t_indicator: Vec<F>,
    pub a_indicator: Vec<F>,
    pub constraint_index: usize,
}

/// Parameters for G2 scalar multiplication sumcheck.
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

/// Single-point evaluation values for G2 scalar mul polynomials.
#[derive(Clone, Copy, Debug)]
struct G2ScalarMulValues {
    x_a_c0: Fq, x_a_c1: Fq,
    y_a_c0: Fq, y_a_c1: Fq,
    x_t_c0: Fq, x_t_c1: Fq,
    y_t_c0: Fq, y_t_c1: Fq,
    x_a_next_c0: Fq, x_a_next_c1: Fq,
    y_a_next_c0: Fq, y_a_next_c1: Fq,
    t_indicator: Fq,
    a_indicator: Fq,
}

impl G2ScalarMulValues {
    #[inline]
    fn from_poly_evals<const D: usize>(poly_evals: &[[Fq; D]], idx: usize) -> Self {
        Self {
            x_a_c0: poly_evals[0][idx], x_a_c1: poly_evals[1][idx],
            y_a_c0: poly_evals[2][idx], y_a_c1: poly_evals[3][idx],
            x_t_c0: poly_evals[4][idx], x_t_c1: poly_evals[5][idx],
            y_t_c0: poly_evals[6][idx], y_t_c1: poly_evals[7][idx],
            x_a_next_c0: poly_evals[8][idx], x_a_next_c1: poly_evals[9][idx],
            y_a_next_c0: poly_evals[10][idx], y_a_next_c1: poly_evals[11][idx],
            t_indicator: poly_evals[12][idx],
            a_indicator: poly_evals[13][idx],
        }
    }

    #[inline]
    fn from_claims(claims: &[Fq]) -> Self {
        Self {
            x_a_c0: claims[0], x_a_c1: claims[1],
            y_a_c0: claims[2], y_a_c1: claims[3],
            x_t_c0: claims[4], x_t_c1: claims[5],
            y_t_c0: claims[6], y_t_c1: claims[7],
            x_a_next_c0: claims[8], x_a_next_c1: claims[9],
            y_a_next_c0: claims[10], y_a_next_c1: claims[11],
            t_indicator: claims[12],
            a_indicator: claims[13],
        }
    }

    /// Evaluate batched constraint: Σ_j δ^j * C_j (13 terms)
    fn eval_constraint(&self, bit: Fq, x_p: Fq2, y_p: Fq2, delta: Fq) -> Fq {
        // Reconstruct Fq2 values
        let x_a = Fq2::new(self.x_a_c0, self.x_a_c1);
        let y_a = Fq2::new(self.y_a_c0, self.y_a_c1);
        let x_t = Fq2::new(self.x_t_c0, self.x_t_c1);
        let y_t = Fq2::new(self.y_t_c0, self.y_t_c1);
        let x_a_next = Fq2::new(self.x_a_next_c0, self.x_a_next_c1);
        let y_a_next = Fq2::new(self.y_a_next_c0, self.y_a_next_c1);

        // Fq2 constants
        let one2 = Fq2::one();
        let two2 = Fq2::new(Fq::from(2u64), Fq::zero());
        let three2 = Fq2::new(Fq::from(3u64), Fq::zero());
        let four2 = Fq2::new(Fq::from(4u64), Fq::zero());
        let nine2 = Fq2::new(Fq::from(9u64), Fq::zero());
        let bit2 = Fq2::new(bit, Fq::zero());
        let ind_t2 = Fq2::new(self.t_indicator, Fq::zero());

        // C1: 4y_A²(x_T + 2x_A) - 9x_A⁴
        let y_a_sq = y_a * y_a;
        let x_a_sq = x_a * x_a;
        let c1 = four2 * y_a_sq * (x_t + two2 * x_a) - nine2 * x_a_sq * x_a_sq;

        // C2: 3x_A²(x_T - x_A) + 2y_A(y_T + y_A)
        let c2 = three2 * x_a_sq * (x_t - x_a) + two2 * y_a * (y_t + y_a);

        // C3: Conditional addition x-coord
        let c3_skip = (one2 - bit2) * (x_a_next - x_t);
        let c3_infinity = bit2 * ind_t2 * (x_a_next - x_p);
        let x_diff = x_p - x_t;
        let y_diff = y_p - y_t;
        let chord_x = (x_a_next + x_t + x_p) * x_diff * x_diff - y_diff * y_diff;
        let c3_add = bit2 * (one2 - ind_t2) * chord_x;
        let c3 = c3_skip + c3_infinity + c3_add;

        // C4: Conditional addition y-coord
        let c4_skip = (one2 - bit2) * (y_a_next - y_t);
        let c4_infinity = bit2 * ind_t2 * (y_a_next - y_p);
        let chord_y = (y_a_next + y_t) * x_diff - y_diff * (x_t - x_a_next);
        let c4_add = bit2 * (one2 - ind_t2) * chord_y;
        let c4 = c4_skip + c4_infinity + c4_add;

        // C5: ind_A * (1 - ind_T)
        let one = Fq::one();
        let c5 = self.a_indicator * (one - self.t_indicator);

        // C6: ind_T * x_T (c0,c1), ind_T * y_T (c0,c1)
        let c6_xt_c0 = self.t_indicator * self.x_t_c0;
        let c6_xt_c1 = self.t_indicator * self.x_t_c1;
        let c6_yt_c0 = self.t_indicator * self.y_t_c0;
        let c6_yt_c1 = self.t_indicator * self.y_t_c1;

        // Batch with powers of delta (13 terms)
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

        c1.c0 + delta * c1.c1
            + d2 * c2.c0 + d3 * c2.c1
            + d4 * c3.c0 + d5 * c3.c1
            + d6 * c4.c0 + d7 * c4.c1
            + d8 * c5
            + d9 * c6_xt_c0 + d10 * c6_xt_c1
            + d11 * c6_yt_c0 + d12 * c6_yt_c1
    }
}

// =============================================================================
// Prover Spec
// =============================================================================

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
        base_points: Vec<(Fq2, Fq2)>,
    ) -> (Self, Vec<usize>) {
        let num_instances = constraint_polys.len();
        let num_vars = params.num_constraint_vars;

        let mut polys_by_kind: Vec<Vec<MultilinearPolynomial<Fq>>> =
            (0..NUM_COMMITTED_KINDS).map(|_| Vec::with_capacity(num_instances)).collect();
        let mut public_polys = vec![Vec::with_capacity(num_instances)];
        let mut constraint_indices = Vec::with_capacity(num_instances);

        for (poly, pub_in) in constraint_polys.into_iter().zip(public_inputs.iter()) {
            constraint_indices.push(poly.constraint_index);

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
            polys_by_kind[12].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.t_indicator)));
            polys_by_kind[13].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.a_indicator)));

            public_polys[0].push(MultilinearPolynomial::LargeScalars(
                DensePolynomial::new(pub_in.build_bit_poly(num_vars))
            ));
        }

        let sequential_indices: Vec<usize> = (0..num_instances).collect();
        (Self { params, polys_by_kind, public_polys, base_points }, sequential_indices)
    }
}

impl ConstraintListSpec for G2ScalarMulProverSpec {
    fn sumcheck_id(&self) -> SumcheckId { self.params.sumcheck_id }
    fn num_rounds(&self) -> usize { self.params.num_constraint_vars }
    fn num_instances(&self) -> usize { self.params.num_constraints }
    fn uses_term_batching(&self) -> bool { true }
    fn opening_specs(&self) -> &'static [OpeningSpec] { &OPENING_SPECS }

    fn build_virtual_poly(&self, term_index: usize, instance: usize) -> VirtualPolynomial {
        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::from_index(term_index).expect("invalid term index"),
            instance,
        })
    }
}

impl ConstraintListProverSpec<Fq, DEGREE> for G2ScalarMulProverSpec {
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
        let vals = G2ScalarMulValues::from_poly_evals(poly_evals, eval_index);
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
pub struct G2ScalarMulVerifierSpec {
    params: G2ScalarMulParams,
    #[allocative(skip)]
    public_inputs: Vec<G2ScalarMulPublicInputs>,
    #[allocative(skip)]
    base_points: Vec<(Fq2, Fq2)>,
}

impl G2ScalarMulVerifierSpec {
    pub fn new(
        params: G2ScalarMulParams,
        public_inputs: Vec<G2ScalarMulPublicInputs>,
        base_points: Vec<(Fq2, Fq2)>,
    ) -> Self {
        Self { params, public_inputs, base_points }
    }
}

impl ConstraintListSpec for G2ScalarMulVerifierSpec {
    fn sumcheck_id(&self) -> SumcheckId { self.params.sumcheck_id }
    fn num_rounds(&self) -> usize { self.params.num_constraint_vars }
    fn num_instances(&self) -> usize { self.params.num_constraints }
    fn uses_term_batching(&self) -> bool { true }
    fn opening_specs(&self) -> &'static [OpeningSpec] { &OPENING_SPECS }

    fn build_virtual_poly(&self, term_index: usize, instance: usize) -> VirtualPolynomial {
        VirtualPolynomial::Recursion(RecursionPoly::G2ScalarMul {
            term: G2ScalarMulTerm::from_index(term_index).expect("invalid term index"),
            instance,
        })
    }
}

impl ConstraintListVerifierSpec<Fq, DEGREE> for G2ScalarMulVerifierSpec {
    fn compute_shared_scalars(&self, _eval_point: &[Fq]) -> Vec<Fq> { vec![] }

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
        let delta = term_batch_coeff.expect("requires term_batch_coeff");
        vals.eval_constraint(bit, x_p, y_p, delta)
    }
}

// =============================================================================
// Type Aliases
// =============================================================================

pub type G2ScalarMulProver<F> = ConstraintListProver<F, G2ScalarMulProverSpec, DEGREE>;
pub type G2ScalarMulVerifier<F> = ConstraintListVerifier<F, G2ScalarMulVerifierSpec, DEGREE>;
