//! G1 scalar multiplication sumcheck for proving G1 scalar multiplication constraints
//!
//! Proves: 0 = Σ_x eq(r_x, x) * Σ_i γ^i * (Σ_j δ^j * C_{i,j}(x))
//! Where C_{i,j} are the scalar-mul constraints for each instance.
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
//! - Delta (term_batch_coeff) batches constraints within each scalar multiplication
//! - Gamma (instance_batch_coeff) batches multiple scalar multiplication instances
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
        recursion::stage1::constraint_list_sumcheck::{
            ConstraintListProver, ConstraintListProverSpec, ConstraintListSpec,
            ConstraintListVerifier, ConstraintListVerifierSpec, OpeningSpec,
        },
        witness::{G1ScalarMulTerm, RecursionPoly, TermEnum, VirtualPolynomial},
    },
};
use allocative::Allocative;
use ark_bn254::{Fq, Fr};
use ark_ff::{BigInteger, One, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;

// =============================================================================
// Constants
// =============================================================================

/// Number of committed polynomial kinds (excluding public bit poly)
const NUM_COMMITTED_KINDS: usize = 8;

/// Sumcheck degree (eq * constraint, where constraint has degree 5 from chord formulas)
const DEGREE: usize = 6;

/// Opening specs for the 8 committed polynomials (Bit is public, not opened)
const G1_SCALAR_MUL_OPENING_SPECS: [OpeningSpec; NUM_COMMITTED_KINDS] = [
    OpeningSpec::new(0, 0), // XA
    OpeningSpec::new(1, 1), // YA
    OpeningSpec::new(2, 2), // XT
    OpeningSpec::new(3, 3), // YT
    OpeningSpec::new(4, 4), // XANext
    OpeningSpec::new(5, 5), // YANext
    OpeningSpec::new(6, 6), // TIndicator
    OpeningSpec::new(7, 7), // AIndicator
];

// =============================================================================
// Public Inputs
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

    /// Scalar bits MSB-first, length 256 (matches witness generation).
    pub fn bits_msb(&self) -> Vec<bool> {
        let scalar_bits_le = self.scalar.into_bigint().to_bits_le();
        (0..256).rev().map(|i| scalar_bits_le[i]).collect()
    }

    /// Evaluate the (padded) bit MLE at the sumcheck challenge point (11 vars).
    ///
    /// Padding convention: only the first 256 entries are populated (bits),
    /// remaining 2048-256 are 0.
    pub fn evaluate_bit_mle<F: JoltField>(&self, eval_point: &[F]) -> F {
        assert_eq!(eval_point.len(), 11);
        let bits = self.bits_msb();

        // First 3 variables select the "prefix = 0" block (since bits live in indices 0..256).
        let pad_factor = EqPolynomial::<F>::zero_selector(&eval_point[..3]);

        // Remaining 8 variables index the 256 step positions.
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

/// Witness polynomials for a G1 scalar multiplication constraint.
#[derive(Clone, Debug)]
pub struct G1ScalarMulWitness {
    pub constraint_index: usize,
    pub base_point: (Fq, Fq),
    pub x_a: Vec<Fq>,
    pub y_a: Vec<Fq>,
    pub x_t: Vec<Fq>,
    pub y_t: Vec<Fq>,
    pub x_a_next: Vec<Fq>,
    pub y_a_next: Vec<Fq>,
    pub t_indicator: Vec<Fq>,
    pub a_indicator: Vec<Fq>,
}

/// Constraint polynomials for a single G1 scalar multiplication
#[derive(Clone)]
pub struct G1ScalarMulConstraintPolynomials {
    pub x_a: Vec<Fq>,
    pub y_a: Vec<Fq>,
    pub x_t: Vec<Fq>,
    pub y_t: Vec<Fq>,
    pub x_a_next: Vec<Fq>,
    pub y_a_next: Vec<Fq>,
    pub t_is_infinity: Vec<Fq>,
    pub a_is_infinity: Vec<Fq>,
    pub base_point: (Fq, Fq),
    pub constraint_index: usize,
}

// =============================================================================
// Parameters
// =============================================================================

/// Parameters for G1 scalar multiplication sumcheck
#[derive(Clone, Allocative)]
pub struct G1ScalarMulParams {
    pub num_constraint_vars: usize,
    pub num_constraints: usize,
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

// =============================================================================
// Constraint Values (single-point evaluations)
// =============================================================================

/// Single-point evaluation values for G1 scalar mul constraint polynomials.
#[derive(Clone, Copy, Debug)]
pub struct G1ScalarMulValues<F> {
    pub x_a: F,
    pub y_a: F,
    pub x_t: F,
    pub y_t: F,
    pub x_a_next: F,
    pub y_a_next: F,
    pub t_indicator: F,
    pub a_indicator: F,
}

impl<F: Copy> G1ScalarMulValues<F> {
    /// Extract values from prover's per-round polynomial evaluations.
    #[inline]
    pub fn from_poly_evals<const DEGREE: usize>(poly_evals: &[[F; DEGREE]], eval_index: usize) -> Self {
        Self {
            x_a: poly_evals[0][eval_index],
            y_a: poly_evals[1][eval_index],
            x_t: poly_evals[2][eval_index],
            y_t: poly_evals[3][eval_index],
            x_a_next: poly_evals[4][eval_index],
            y_a_next: poly_evals[5][eval_index],
            t_indicator: poly_evals[6][eval_index],
            a_indicator: poly_evals[7][eval_index],
        }
    }

    /// Extract values from verifier's opened claims.
    #[inline]
    pub fn from_claims(claims: &[F]) -> Self {
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
}

impl G1ScalarMulValues<Fq> {
    /// Evaluate the batched constraint: Σ_j δ^j * C_j
    ///
    /// 7 constraints: C1, C2, C3, C4, C5, C6_x, C6_y
    #[allow(clippy::too_many_arguments)]
    pub fn eval_constraint(&self, bit: Fq, x_p: Fq, y_p: Fq, delta: Fq) -> Fq {
        let c1 = compute_c1(self.x_a, self.y_a, self.x_t);
        let c2 = compute_c2(self.x_a, self.y_a, self.x_t, self.y_t);
        let c3 = compute_c3(bit, self.t_indicator, self.x_a_next, self.x_t, self.y_t, x_p, y_p);
        let c4 = compute_c4(
            bit,
            self.t_indicator,
            self.x_a_next,
            self.y_a_next,
            self.x_t,
            self.y_t,
            x_p,
            y_p,
        );
        let c5 = compute_c5(self.a_indicator, self.t_indicator);
        let c6_x = compute_c6_x(self.t_indicator, self.x_t);
        let c6_y = compute_c6_y(self.t_indicator, self.y_t);

        // Batch with powers of delta: c1 + δ*c2 + δ²*c3 + δ³*c4 + δ⁴*c5 + δ⁵*c6_x + δ⁶*c6_y
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta3 * delta;
        let delta5 = delta4 * delta;
        let delta6 = delta5 * delta;

        c1 + delta * c2 + delta2 * c3 + delta3 * c4 + delta4 * c5 + delta5 * c6_x + delta6 * c6_y
    }
}

// =============================================================================
// Constraint Functions
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

    let c3_skip = (one - bit) * (x_a_next - x_t);
    let c3_infinity = bit * ind_t * (x_a_next - x_p);

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
fn compute_c4(bit: Fq, ind_t: Fq, x_a_next: Fq, y_a_next: Fq, x_t: Fq, y_t: Fq, x_p: Fq, y_p: Fq) -> Fq {
    let one = Fq::one();

    let c4_skip = (one - bit) * (y_a_next - y_t);
    let c4_infinity = bit * ind_t * (y_a_next - y_p);

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

/// C6: Infinity encoding check for T (y component)
fn compute_c6_y(ind_t: Fq, y_t: Fq) -> Fq {
    ind_t * y_t
}

// =============================================================================
// Prover Spec
// =============================================================================

/// Prover-side specification for G1 scalar mul constraints.
#[derive(Clone, Allocative)]
pub struct G1ScalarMulProverSpec {
    params: G1ScalarMulParams,
    /// Committed polynomials: polys_by_kind[kind][instance]
    polys_by_kind: Vec<Vec<MultilinearPolynomial<Fq>>>,
    /// Public polynomials (bit): public_polys[0][instance]
    public_polys: Vec<Vec<MultilinearPolynomial<Fq>>>,
    /// Base points per instance
    #[allocative(skip)]
    base_points: Vec<(Fq, Fq)>,
}

impl G1ScalarMulProverSpec {
    /// Create a new prover spec from constraint polynomials and public inputs.
    ///
    /// Returns `(spec, constraint_indices)` for use with `ConstraintListProver::from_spec`.
    pub fn new(
        params: G1ScalarMulParams,
        constraint_polys: Vec<G1ScalarMulConstraintPolynomials>,
        public_inputs: &[G1ScalarMulPublicInputs],
    ) -> (Self, Vec<usize>) {
        debug_assert_eq!(
            constraint_polys.len(),
            public_inputs.len(),
            "constraint_polys and public_inputs must have same length"
        );

        let num_instances = constraint_polys.len();
        let num_vars = params.num_constraint_vars;

        // Initialize polys_by_kind for 8 committed polynomial types
        let mut polys_by_kind: Vec<Vec<MultilinearPolynomial<Fq>>> =
            (0..NUM_COMMITTED_KINDS).map(|_| Vec::with_capacity(num_instances)).collect();

        // Initialize public_polys for 1 public polynomial type (bit)
        let mut public_polys: Vec<Vec<MultilinearPolynomial<Fq>>> =
            vec![Vec::with_capacity(num_instances)];

        let mut base_points = Vec::with_capacity(num_instances);
        let mut constraint_indices = Vec::with_capacity(num_instances);

        for (poly, pub_in) in constraint_polys.into_iter().zip(public_inputs.iter()) {
            constraint_indices.push(poly.constraint_index);
            base_points.push(poly.base_point);

            // Committed polynomials
            polys_by_kind[0].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.x_a)));
            polys_by_kind[1].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.y_a)));
            polys_by_kind[2].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.x_t)));
            polys_by_kind[3].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.y_t)));
            polys_by_kind[4].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.x_a_next)));
            polys_by_kind[5].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.y_a_next)));
            polys_by_kind[6].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.t_is_infinity)));
            polys_by_kind[7].push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly.a_is_infinity)));

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

        // Use sequential indices to match Stage 2's expectation
        let sequential_indices: Vec<usize> = (0..num_instances).collect();
        (spec, sequential_indices)
    }
}

impl ConstraintListSpec for G1ScalarMulProverSpec {
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
        &G1_SCALAR_MUL_OPENING_SPECS
    }

    fn build_virtual_poly(&self, term_index: usize, instance: usize) -> VirtualPolynomial {
        VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
            term: G1ScalarMulTerm::from_index(term_index).expect("invalid G1ScalarMulTerm index"),
            instance,
        })
    }
}

impl ConstraintListProverSpec<Fq, DEGREE> for G1ScalarMulProverSpec {
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
        let vals = G1ScalarMulValues::from_poly_evals(poly_evals, eval_index);
        let bit = public_evals[0][eval_index];
        let (x_p, y_p) = self.base_points[instance];
        let delta = term_batch_coeff.expect("G1ScalarMul requires term_batch_coeff");
        vals.eval_constraint(bit, x_p, y_p, delta)
    }
}

// =============================================================================
// Verifier Spec
// =============================================================================

/// Verifier-side specification for G1 scalar mul constraints.
#[derive(Clone, Allocative)]
pub struct G1ScalarMulVerifierSpec {
    params: G1ScalarMulParams,
    #[allocative(skip)]
    base_points: Vec<(Fq, Fq)>,
    #[allocative(skip)]
    public_inputs: Vec<G1ScalarMulPublicInputs>,
}

impl G1ScalarMulVerifierSpec {
    pub fn new(
        params: G1ScalarMulParams,
        base_points: Vec<(Fq, Fq)>,
        public_inputs: Vec<G1ScalarMulPublicInputs>,
    ) -> Self {
        Self {
            params,
            base_points,
            public_inputs,
        }
    }
}

impl ConstraintListSpec for G1ScalarMulVerifierSpec {
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
        &G1_SCALAR_MUL_OPENING_SPECS
    }

    fn build_virtual_poly(&self, term_index: usize, instance: usize) -> VirtualPolynomial {
        VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
            term: G1ScalarMulTerm::from_index(term_index).expect("invalid G1ScalarMulTerm index"),
            instance,
        })
    }
}

impl ConstraintListVerifierSpec<Fq, DEGREE> for G1ScalarMulVerifierSpec {
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
        let vals = G1ScalarMulValues::from_claims(opened_claims);
        // Compute bit evaluation from public inputs
        let bit = self.public_inputs[instance].evaluate_bit_mle(eval_point);
        let (x_p, y_p) = self.base_points[instance];
        let delta = term_batch_coeff.expect("G1ScalarMul requires term_batch_coeff");
        vals.eval_constraint(bit, x_p, y_p, delta)
    }
}

// =============================================================================
// Type Aliases
// =============================================================================

/// Prover for G1 scalar multiplication sumcheck.
pub type G1ScalarMulProver<F> = ConstraintListProver<F, G1ScalarMulProverSpec, DEGREE>;

/// Verifier for G1 scalar multiplication sumcheck.
pub type G1ScalarMulVerifier<F> = ConstraintListVerifier<F, G1ScalarMulVerifierSpec, DEGREE>;
