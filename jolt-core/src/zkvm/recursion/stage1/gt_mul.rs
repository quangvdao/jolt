//! GT multiplication sumcheck for proving GT multiplication constraints
//! Proves: 0 = Σ_x eq(eq_point, x) * Σ_i (instance_batch_coeff)^i * C_i(x)
//! Where C_i(x) = a_i(x) × b_i(x) - c_i(x) - Q_i(x) × g(x)
//!
//! This is a separate sumcheck protocol for GT multiplication constraints.
//! Output: Virtual polynomial claims for each polynomial in each constraint

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial, multilinear_polynomial::MultilinearPolynomial,
        opening_proof::SumcheckId,
    },
};

use crate::zkvm::witness::{GtMulTerm, RecursionPoly, VirtualPolynomial};

use super::constraint_list_sumcheck::{
    ConstraintListProver, ConstraintListProverSpec, ConstraintListSpec, ConstraintListVerifier,
    ConstraintListVerifierSpec, OpeningSpec,
};

// ============================================================================
// Opening Specs
// ============================================================================

const GT_MUL_OPENING_SPECS: [OpeningSpec; 4] = [
    OpeningSpec::new(0, |i| {
        VirtualPolynomial::Recursion(RecursionPoly::GtMul {
            term: GtMulTerm::Lhs,
            instance: i,
        })
    }),
    OpeningSpec::new(1, |i| {
        VirtualPolynomial::Recursion(RecursionPoly::GtMul {
            term: GtMulTerm::Rhs,
            instance: i,
        })
    }),
    OpeningSpec::new(2, |i| {
        VirtualPolynomial::Recursion(RecursionPoly::GtMul {
            term: GtMulTerm::Result,
            instance: i,
        })
    }),
    OpeningSpec::new(3, |i| {
        VirtualPolynomial::Recursion(RecursionPoly::GtMul {
            term: GtMulTerm::Quotient,
            instance: i,
        })
    }),
];

// ============================================================================
// Parameters and Witness Types
// ============================================================================

/// Individual polynomial data for a single GT mul constraint
#[derive(Clone)]
pub struct GtMulConstraintPolynomials<F: JoltField> {
    pub lhs: Vec<F>,
    pub rhs: Vec<F>,
    pub result: Vec<F>,
    pub quotient: Vec<F>,
    pub constraint_index: usize,
}

/// Parameters for GT mul sumcheck
#[derive(Clone)]
pub struct GtMulParams {
    /// Number of constraint variables (x) - fixed at 11 for uniform matrix
    pub num_constraint_vars: usize,

    /// Number of constraints
    pub num_constraints: usize,

    /// Sumcheck instance identifier
    pub sumcheck_id: SumcheckId,
}

impl GtMulParams {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            num_constraint_vars: 11, // 11 vars for uniform matrix (4 element + 7 padding)
            num_constraints,
            sumcheck_id: SumcheckId::GtMul,
        }
    }
}

// ============================================================================
// Prover Spec
// ============================================================================

/// Prover-side specification for GT mul constraints.
#[derive(Clone)]
pub struct GtMulProverSpec<F: JoltField> {
    params: GtMulParams,
    polys_by_kind: Vec<Vec<MultilinearPolynomial<F>>>,
    shared_polys: Vec<MultilinearPolynomial<F>>,
}

impl<F: JoltField> GtMulProverSpec<F> {
    /// Create a new prover spec from constraint polynomials and the g polynomial.
    pub fn new(
        params: GtMulParams,
        constraint_polys: Vec<GtMulConstraintPolynomials<F>>,
        g_poly: DensePolynomial<F>,
    ) -> Self {
        let num_instances = constraint_polys.len();
        debug_assert_eq!(
            num_instances, params.num_constraints,
            "GtMulProverSpec: params.num_constraints must match constraint_polys length"
        );

        let mut lhs = Vec::with_capacity(num_instances);
        let mut rhs = Vec::with_capacity(num_instances);
        let mut result = Vec::with_capacity(num_instances);
        let mut quotient = Vec::with_capacity(num_instances);

        for poly in constraint_polys {
            lhs.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.lhs,
            )));
            rhs.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.rhs,
            )));
            result.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.result,
            )));
            quotient.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                poly.quotient,
            )));
        }

        Self {
            params,
            polys_by_kind: vec![lhs, rhs, result, quotient],
            shared_polys: vec![MultilinearPolynomial::LargeScalars(g_poly)],
        }
    }
}

impl<F: JoltField> ConstraintListSpec for GtMulProverSpec<F> {
    fn sumcheck_id(&self) -> SumcheckId {
        self.params.sumcheck_id
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn num_instances(&self) -> usize {
        self.params.num_constraints
    }

    fn opening_specs(&self) -> &'static [OpeningSpec] {
        &GT_MUL_OPENING_SPECS
    }
}

impl<F: JoltField> ConstraintListProverSpec<F, 3> for GtMulProverSpec<F> {
    fn polys_by_kind(&self) -> &[Vec<MultilinearPolynomial<F>>] {
        &self.polys_by_kind
    }

    fn polys_by_kind_mut(&mut self) -> &mut [Vec<MultilinearPolynomial<F>>] {
        &mut self.polys_by_kind
    }

    fn shared_polys(&self) -> &[MultilinearPolynomial<F>] {
        &self.shared_polys
    }

    fn shared_polys_mut(&mut self) -> &mut [MultilinearPolynomial<F>] {
        &mut self.shared_polys
    }

    fn eval_constraint(
        &self,
        _instance: usize,
        eval_index: usize,
        poly_evals: &[[F; 3]],
        shared_evals: &[[F; 3]],
        _term_batch_coeff: Option<F>,
    ) -> F {
        let lhs = poly_evals[0][eval_index];
        let rhs = poly_evals[1][eval_index];
        let result = poly_evals[2][eval_index];
        let quotient = poly_evals[3][eval_index];
        let g = shared_evals[0][eval_index];
        lhs * rhs - result - quotient * g
    }
}

// ============================================================================
// Verifier Spec
// ============================================================================

/// Verifier-side specification for GT mul constraints.
#[derive(Clone)]
pub struct GtMulVerifierSpec {
    params: GtMulParams,
}

impl GtMulVerifierSpec {
    pub fn new(params: GtMulParams) -> Self {
        Self { params }
    }
}

impl ConstraintListSpec for GtMulVerifierSpec {
    fn sumcheck_id(&self) -> SumcheckId {
        self.params.sumcheck_id
    }

    fn num_rounds(&self) -> usize {
        self.params.num_constraint_vars
    }

    fn num_instances(&self) -> usize {
        self.params.num_constraints
    }

    fn opening_specs(&self) -> &'static [OpeningSpec] {
        &GT_MUL_OPENING_SPECS
    }
}

impl<F: JoltField> ConstraintListVerifierSpec<F, 3> for GtMulVerifierSpec {
    fn compute_shared_scalars(&self, eval_point: &[F]) -> Vec<F> {
        // Compute g(eval_point) once from the public g MLE.
        // The g polynomial is the MLE of the irreducible polynomial p(X) for Fq12.
        use crate::zkvm::recursion::constraints_sys::DoryMatrixBuilder;
        use ark_bn254::Fq;
        use jolt_optimizations::get_g_mle;
        use std::any::TypeId;

        // Runtime check that F = Fq (recursion SNARK is always over Fq)
        if TypeId::of::<F>() != TypeId::of::<Fq>() {
            panic!("g polynomial evaluation requires F = Fq for recursion SNARK");
        }

        let g_mle_4var = get_g_mle();
        let g_mle_padded = if eval_point.len() == 11 {
            DoryMatrixBuilder::pad_4var_to_11var_zero_padding(&g_mle_4var)
        } else if eval_point.len() == 8 {
            DoryMatrixBuilder::pad_4var_to_8var_zero_padding(&g_mle_4var)
        } else {
            g_mle_4var
        };

        // Evaluate g polynomial at eval_point
        // SAFETY: F = Fq verified above, and slice references have same layout
        let g_poly_fq =
            MultilinearPolynomial::<Fq>::LargeScalars(DensePolynomial::new(g_mle_padded));
        let eval_point_fq: &[Fq] = unsafe {
            std::slice::from_raw_parts(eval_point.as_ptr() as *const Fq, eval_point.len())
        };
        let g_eval_fq = g_poly_fq.evaluate_dot_product(eval_point_fq);
        let g_eval: F = unsafe { std::mem::transmute_copy(&g_eval_fq) };

        vec![g_eval]
    }

    fn eval_constraint_at_point(
        &self,
        _instance: usize,
        opened_claims: &[F],
        shared_scalars: &[F],
        _term_batch_coeff: Option<F>,
    ) -> F {
        let lhs = opened_claims[0];
        let rhs = opened_claims[1];
        let result = opened_claims[2];
        let quotient = opened_claims[3];
        let g_eval = shared_scalars[0];

        lhs * rhs - result - quotient * g_eval
    }
}

// ============================================================================
// Type Aliases (no wrapper structs needed!)
// ============================================================================

/// Prover for GT mul sumcheck.
///
/// This is a type alias - no manual trait delegation required.
pub type GtMulProver<F> = ConstraintListProver<F, GtMulProverSpec<F>, 3>;

/// Verifier for GT mul sumcheck.
///
/// This is a type alias - no manual trait delegation required.
pub type GtMulVerifier<F> = ConstraintListVerifier<F, GtMulVerifierSpec, 3>;
