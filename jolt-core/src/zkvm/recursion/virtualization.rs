//! Stage 2/3 helpers for the recursion SNARK.
//!
//! - **Stage 2 (sumchecks)** produce “virtual claims” \(v_i = f_i(r_x)\) at a shared point `r_x`.
//! - **Stage 3 (prefix packing)** reduces all \(v_i\) into one packed evaluation.
//!   To do that deterministically, we need a canonical ordering / indexing of the claims, plus
//!   a helper to extract them from the opening accumulator.
//!
//! This file contains:
//! - `extract_virtual_claims_from_accumulator`: used by Stage 3 prefix packing (prover + verifier).
//! - `virtual_claim_index` / `matrix_s_index`: used by tests and by legacy code paths.
//! - `DirectEvaluation{Prover,Verifier}`: a legacy “direct evaluation” helper that is not part of
//!   the current prefix-packing recursion pipeline, but is kept for now as a debugging/prototyping
//!   tool.
//!
//! ## Mathematical Foundation
//!
//! The matrix M is defined such that M(i, r_x) = v_i for all i, where v_i are
//! the virtual claims from Stage 2. The direct evaluation protocol uses the fact
//! that for the MLE of M:
//!
//! M(r_s, r_x) = Σ_i eq(r_s, i) · M(i, r_x) = Σ_i eq(r_s, i) · v_i
//!
//! ## Data Layout
//!
//! ### Virtual Claims Layout
//! Virtual claims from Stage 2 are organized by constraint then polynomial type:
//! [c0_p0, c0_p1, ..., c0_p14, c1_p0, c1_p1, ..., c1_p14, ...]
//!
//! ### Matrix S Layout
//! The matrix S rows are indexed differently for mathematical efficiency:
//! Row index = poly_type * num_constraints_padded + constraint_idx
//!
//! This transposed layout ensures proper alignment for the virtualization sumcheck.
//!
//! ## Protocol Flow
//!
//! 1. **Sampling**: Sample r_s directly from the Fiat-Shamir transcript
//! 2. **Prover Evaluation**: Prover evaluates M(r_s, r_x) where r_x comes from Stage 2
//! 3. **Verifier Computation**: Verifier computes Σ_i eq(r_s, i) · v_i independently
//! 4. **Verification**: Check that prover's evaluation matches verifier's computation

use thiserror::Error;

/// Errors that can occur in Stage 3 direct evaluation protocol
#[derive(Debug, Error)]
pub enum Stage2Error {
    #[error("Direct evaluation mismatch: expected {expected}, got {actual}")]
    EvaluationMismatch { expected: String, actual: String },

    #[error("Invalid accumulator state: {0}")]
    InvalidAccumulator(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

#[cfg(feature = "experimental-pairing-recursion")]
use crate::zkvm::witness::{MultiMillerLoopTerm, RecursionPoly};
use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    transcripts::Transcript,
    zkvm::{
        recursion::{
            constraints::system::{ConstraintType, PolyType},
            gt::exponentiation::GtExpPublicInputs,
        },
        witness::VirtualPolynomial,
    },
};
use ark_bn254::Fq;
use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// Number of polynomial types in the constraint system
const NUM_POLY_TYPES: usize = PolyType::NUM_TYPES;

/// Helper function to compute the index in the virtual claims array
///
/// Virtual claims are laid out as:
/// [constraint_0_poly_0, constraint_0_poly_1, ..., constraint_0_poly_13,
///  constraint_1_poly_0, constraint_1_poly_1, ..., constraint_1_poly_13, ...]
///
/// So for constraint i and polynomial type j, the index is: i * NUM_POLY_TYPES + j
#[inline]
pub fn virtual_claim_index(constraint_idx: usize, poly_idx: usize) -> usize {
    constraint_idx * NUM_POLY_TYPES + poly_idx
}

/// Helper function to compute the index in the matrix S evaluations
///
/// The matrix S is laid out with a different pattern than virtual claims:
/// - Rows are indexed by polynomial type first, then constraint
/// - This layout is: poly_type * num_constraints_padded + constraint_idx
///
/// This is the transpose of how virtual claims are laid out, which is important
/// for the mathematical properties of the virtualization protocol.
#[inline]
pub fn matrix_s_index(
    poly_idx: usize,
    constraint_idx: usize,
    num_constraints_padded: usize,
) -> usize {
    poly_idx * num_constraints_padded + constraint_idx
}

/// Parameters for the direct evaluation protocol
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DirectEvaluationParams {
    /// Number of s-variables (log of matrix rows)
    pub num_s_vars: usize,
    /// Number of constraints
    pub num_constraints: usize,
    /// Padded number of constraints (next power of 2)
    pub num_constraints_padded: usize,
    /// Number of constraint variables (x variables)
    pub num_constraint_vars: usize,
}

impl DirectEvaluationParams {
    pub fn new(
        num_s_vars: usize,
        num_constraints: usize,
        num_constraints_padded: usize,
        num_constraint_vars: usize,
    ) -> Self {
        Self {
            num_s_vars,
            num_constraints,
            num_constraints_padded,
            num_constraint_vars,
        }
    }
}

/// Prover for the direct evaluation protocol
pub struct DirectEvaluationProver {
    /// Protocol parameters
    pub params: DirectEvaluationParams,
    /// The row-evaluation vector \(i \mapsto M(i, r_x)\) materialized as an MLE over `s`.
    ///
    /// Row index layout matches [`matrix_s_index`]: poly-type major then constraint index.
    pub row_values_mle: MultilinearPolynomial<Fq>,
    /// Virtual claims from Stage 2
    pub virtual_claims: Vec<Fq>,
    /// The r_x point from Stage 2
    pub r_x: Vec<Fq>,
}

impl DirectEvaluationProver {
    /// Create a new prover
    pub fn new(params: DirectEvaluationParams, virtual_claims: Vec<Fq>, r_x: Vec<Fq>) -> Self {
        // Build the row-evaluation table `M(i, r_x)` from Stage 2 virtual claims:
        //
        // For each row i (poly_idx, constraint_idx), Stage 2 provides v_i = M(i, r_x).
        // We place v_i at `row = matrix_s_index(poly_idx, constraint_idx, num_constraints_padded)`.
        let rows = params.num_constraints_padded * NUM_POLY_TYPES;
        debug_assert_eq!(
            1usize << params.num_s_vars,
            rows,
            "Expected 2^num_s_vars rows (got rows={rows}, num_s_vars={})",
            params.num_s_vars
        );

        let mut row_values = vec![Fq::zero(); rows];
        for constraint_idx in 0..params.num_constraints {
            for poly_idx in 0..NUM_POLY_TYPES {
                let claim_idx = virtual_claim_index(constraint_idx, poly_idx);
                let row_idx =
                    matrix_s_index(poly_idx, constraint_idx, params.num_constraints_padded);
                row_values[row_idx] = virtual_claims[claim_idx];
            }
        }

        let row_values_mle = MultilinearPolynomial::LargeScalars(DensePolynomial::new(row_values));

        Self {
            params,
            row_values_mle,
            virtual_claims,
            r_x,
        }
    }

    /// Run the prover protocol
    pub fn prove<T: Transcript>(
        &self,
        transcript: &mut T,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
    ) -> (Vec<Fq>, Fq) {
        // Sample r_s from the transcript
        let r_s: Vec<Fq> = (0..self.params.num_s_vars)
            .map(|_| transcript.challenge_scalar::<Fq>())
            .collect();

        // Evaluate M(r_s, r_x) = Σ_i eq(r_s, i) · v_i via MLE evaluation of row values.
        let m_eval = PolynomialEvaluation::evaluate(&self.row_values_mle, &r_s);

        // Note: m_eval is passed in the proof structure, but we still append
        // to transcript to maintain Fiat-Shamir soundness
        transcript.append_scalar(&m_eval);

        // Store the opening in the accumulator for Stage 3
        // Note: We reverse r_s and r_x because OpeningPoint expects big-endian ordering
        // while our polynomials use little-endian variable ordering internally.
        // The matrix has variables ordered as [x_vars, s_vars] in little-endian.
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(
            r_s.iter()
                .rev()
                .chain(self.r_x.iter().rev())
                .cloned()
                .map(|f| f.into())
                .collect(),
        );

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::DorySparseConstraintMatrix,
            SumcheckId::RecursionVirtualization,
            opening_point,
            m_eval,
        );

        (r_s, m_eval)
    }

    /// Run the prover protocol, using a fixed prefix `r_c` and sampling only the remaining
    /// `r_t` coordinates in Stage 3.
    ///
    /// This is used by the fused-virtual-polynomials rollout: Stage 2 fixes `r_c`, Stage 3
    /// samples only the remaining `s`-coordinates.
    pub fn prove_with_r_c<T: Transcript>(
        &self,
        r_c: &[Fq],
        transcript: &mut T,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
    ) -> (Vec<Fq>, Fq) {
        assert!(
            r_c.len() <= self.params.num_s_vars,
            "r_c length must be <= num_s_vars"
        );
        let num_remaining = self.params.num_s_vars - r_c.len();
        let r_t: Vec<Fq> = (0..num_remaining)
            .map(|_| transcript.challenge_scalar::<Fq>())
            .collect();

        let mut r_s: Vec<Fq> = Vec::with_capacity(self.params.num_s_vars);
        r_s.extend_from_slice(r_c);
        r_s.extend_from_slice(&r_t);

        // Evaluate M(r_s, r_x) = Σ_i eq(r_s, i) · v_i via MLE evaluation of row values.
        let m_eval = PolynomialEvaluation::evaluate(&self.row_values_mle, &r_s);
        transcript.append_scalar(&m_eval);

        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(
            r_s.iter()
                .rev()
                .chain(self.r_x.iter().rev())
                .cloned()
                .map(|f| f.into())
                .collect(),
        );

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::DorySparseConstraintMatrix,
            SumcheckId::RecursionVirtualization,
            opening_point,
            m_eval,
        );

        (r_s, m_eval)
    }
}

/// Verifier for the direct evaluation protocol
pub struct DirectEvaluationVerifier {
    /// Protocol parameters
    pub params: DirectEvaluationParams,
    /// Virtual claims from Stage 2
    pub virtual_claims: Vec<Fq>,
    /// The r_x point from Stage 2
    pub r_x: Vec<Fq>,
}

impl DirectEvaluationVerifier {
    /// Create a new verifier
    pub fn new(params: DirectEvaluationParams, virtual_claims: Vec<Fq>, r_x: Vec<Fq>) -> Self {
        Self {
            params,
            virtual_claims,
            r_x,
        }
    }

    /// Run the verifier protocol
    pub fn verify<T: Transcript>(
        &self,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        m_eval_claimed: Fq,
    ) -> Result<Vec<Fq>, Stage2Error> {
        // Sample the same r_s as the prover
        let r_s: Vec<Fq> = (0..self.params.num_s_vars)
            .map(|_| transcript.challenge_scalar::<Fq>())
            .collect();

        // Compute the expected value: Σ_i eq(r_s, i) · v_i
        let eq_evals = EqPolynomial::<Fq>::evals(&r_s);
        let m_eval_expected = self.compute_expected_evaluation(&eq_evals);

        // Verify the claim
        if m_eval_claimed != m_eval_expected {
            return Err(Stage2Error::EvaluationMismatch {
                expected: format!("{m_eval_expected:?}"),
                actual: format!("{m_eval_claimed:?}"),
            });
        }

        // Append to transcript to maintain Fiat-Shamir soundness
        transcript.append_scalar(&m_eval_claimed);

        // Store the opening in the accumulator for Stage 3
        // Note: We reverse r_s and r_x because OpeningPoint expects big-endian ordering
        // while our polynomials use little-endian variable ordering internally.
        // The matrix has variables ordered as [x_vars, s_vars] in little-endian.
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(
            r_s.iter()
                .rev()
                .chain(self.r_x.iter().rev())
                .cloned()
                .map(|f| f.into())
                .collect(),
        );

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::DorySparseConstraintMatrix,
            SumcheckId::RecursionVirtualization,
            opening_point,
        );

        Ok(r_s)
    }

    /// Run the verifier protocol, using a fixed prefix `r_c` and sampling only the remaining
    /// `r_t` coordinates in Stage 3.
    pub fn verify_with_r_c<T: Transcript>(
        &self,
        r_c: &[Fq],
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        m_eval_claimed: Fq,
    ) -> Result<Vec<Fq>, Stage2Error> {
        assert!(
            r_c.len() <= self.params.num_s_vars,
            "r_c length must be <= num_s_vars"
        );
        let num_remaining = self.params.num_s_vars - r_c.len();
        let r_t: Vec<Fq> = (0..num_remaining)
            .map(|_| transcript.challenge_scalar::<Fq>())
            .collect();
        let mut r_s: Vec<Fq> = Vec::with_capacity(self.params.num_s_vars);
        r_s.extend_from_slice(r_c);
        r_s.extend_from_slice(&r_t);

        let eq_evals = EqPolynomial::<Fq>::evals(&r_s);
        let m_eval_expected = self.compute_expected_evaluation(&eq_evals);

        if m_eval_claimed != m_eval_expected {
            return Err(Stage2Error::EvaluationMismatch {
                expected: format!("{m_eval_expected:?}"),
                actual: format!("{m_eval_claimed:?}"),
            });
        }

        transcript.append_scalar(&m_eval_claimed);

        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(
            r_s.iter()
                .rev()
                .chain(self.r_x.iter().rev())
                .cloned()
                .map(|f| f.into())
                .collect(),
        );

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::DorySparseConstraintMatrix,
            SumcheckId::RecursionVirtualization,
            opening_point,
        );

        Ok(r_s)
    }

    /// Compute Σ_i eq(r_s, i) · v_i
    fn compute_expected_evaluation(&self, eq_evals: &[Fq]) -> Fq {
        let mut result = Fq::zero();

        // The virtual claims are laid out as:
        // [constraint_0_poly_0, constraint_1_poly_0, ..., constraint_0_poly_1, ...]
        // We need to match this with the eq evaluations

        for constraint_idx in 0..self.params.num_constraints {
            for poly_idx in 0..NUM_POLY_TYPES {
                let claim_idx = virtual_claim_index(constraint_idx, poly_idx);
                let s_idx =
                    matrix_s_index(poly_idx, constraint_idx, self.params.num_constraints_padded);

                if claim_idx < self.virtual_claims.len() && s_idx < eq_evals.len() {
                    result += eq_evals[s_idx] * self.virtual_claims[claim_idx];
                }
            }
        }

        result
    }
}

/// Extract virtual claims from any accumulator (Prover or Verifier) in the correct order
///
/// This function extracts the virtual polynomial claims from Stage 2 accumulators
/// and organizes them in the standard layout expected by Stage 3:
/// [constraint_0_poly_0, constraint_0_poly_1, ..., constraint_0_poly_12,
///  constraint_1_poly_0, constraint_1_poly_1, ..., constraint_1_poly_12, ...]
///
/// For GtExp constraints, base and bit evaluations are computed directly from
/// public inputs rather than being extracted from the accumulator.
///
/// # Type Parameters
/// - `F`: The field type
/// - `A`: The accumulator type (ProverOpeningAccumulator or VerifierOpeningAccumulator)
///
/// # Arguments
/// - `accumulator`: The Stage 2 opening accumulator
/// - `constraint_types`: The types of constraints in order
/// - `gt_exp_public_inputs`: Public inputs for each packed GT exp (base, scalar_bits)
///
/// # Returns
/// A vector of virtual claims organized by constraint then polynomial type
pub fn extract_virtual_claims_from_accumulator<F: JoltField, A: OpeningAccumulator<F>>(
    accumulator: &A,
    constraint_types: &[ConstraintType],
    _gt_exp_public_inputs: &[GtExpPublicInputs],
    enable_gt_fused_end_to_end: bool,
) -> Vec<F> {
    let mut claims = Vec::new();

    // Track separate indices for each constraint type. These indices correspond to the
    // `instance` field in `VirtualPolynomial::Recursion(...)` for that constraint family.
    let mut gt_exp_idx = 0usize;
    let mut gt_mul_idx = 0usize;
    let mut g1_scalar_mul_idx = 0usize;
    let mut g2_scalar_mul_idx = 0usize;
    let mut g1_add_idx = 0usize;
    let mut g2_add_idx = 0usize;
    #[cfg(feature = "experimental-pairing-recursion")]
    let mut multi_miller_loop_idx = 0usize;

    // Process each constraint
    for (idx, constraint_type) in constraint_types.iter().enumerate() {
        // For each constraint, extract claims for all polynomial types
        // in the correct order matching the PolyType enum.

        let mut constraint_claims = vec![F::zero(); NUM_POLY_TYPES];

        match constraint_type {
            ConstraintType::GtExp => {
                if enable_gt_fused_end_to_end {
                    // In end-to-end GT-fused mode, per-instance GT rows are not part of the packed
                    // dense matrix, so Stage 3 does not need these claims.
                    gt_exp_idx += 1;
                    claims.extend_from_slice(&constraint_claims);
                    continue;
                }
                // Packed GT Exp uses matrix polynomials: RhoPrev + Quotient.
                // Base/digits/rho_next are public inputs or separately-verified, not in the matrix.
                tracing::debug!(
                    "[extract_constraint_claims] Getting GtExp({}) openings for constraint {}",
                    gt_exp_idx,
                    idx,
                );

                // Packed GT exp claims are *claim-reduced* in Stage 2 to the shared r_x.
                let (_, rho_prev) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::gt_exp_rho(gt_exp_idx),
                    SumcheckId::GtExpClaimReduction,
                );
                let (_, quotient) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::gt_exp_quotient(gt_exp_idx),
                    SumcheckId::GtExpClaimReduction,
                );

                constraint_claims[PolyType::RhoPrev as usize] = rho_prev;
                constraint_claims[PolyType::Quotient as usize] = quotient;
                gt_exp_idx += 1;
            }
            ConstraintType::GtMul => {
                if enable_gt_fused_end_to_end {
                    // In end-to-end GT-fused mode, per-instance GT rows are not part of the packed
                    // dense matrix, so Stage 3 does not need these claims.
                    gt_mul_idx += 1;
                    claims.extend_from_slice(&constraint_claims);
                    continue;
                }
                let (_, lhs) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::gt_mul_lhs(gt_mul_idx),
                    SumcheckId::GtMul,
                );
                let (_, rhs) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::gt_mul_rhs(gt_mul_idx),
                    SumcheckId::GtMul,
                );
                let (_, result) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::gt_mul_result(gt_mul_idx),
                    SumcheckId::GtMul,
                );
                let (_, quotient) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::gt_mul_quotient(gt_mul_idx),
                    SumcheckId::GtMul,
                );

                constraint_claims[PolyType::MulLhs as usize] = lhs;
                constraint_claims[PolyType::MulRhs as usize] = rhs;
                constraint_claims[PolyType::MulResult as usize] = result;
                constraint_claims[PolyType::MulQuotient as usize] = quotient;
                gt_mul_idx += 1;
            }
            ConstraintType::G1ScalarMul { .. } => {
                let (_, x_a) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_scalar_mul_xa(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, y_a) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_scalar_mul_ya(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, x_t) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_scalar_mul_xt(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, y_t) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_scalar_mul_yt(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, x_a_next) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_scalar_mul_xa_next(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, y_a_next) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_scalar_mul_ya_next(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, t_indicator) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_scalar_mul_t_indicator(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );
                let (_, a_indicator) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_scalar_mul_a_indicator(g1_scalar_mul_idx),
                    SumcheckId::G1ScalarMul,
                );

                constraint_claims[PolyType::G1ScalarMulXA as usize] = x_a;
                constraint_claims[PolyType::G1ScalarMulYA as usize] = y_a;
                constraint_claims[PolyType::G1ScalarMulXT as usize] = x_t;
                constraint_claims[PolyType::G1ScalarMulYT as usize] = y_t;
                constraint_claims[PolyType::G1ScalarMulXANext as usize] = x_a_next;
                constraint_claims[PolyType::G1ScalarMulYANext as usize] = y_a_next;
                constraint_claims[PolyType::G1ScalarMulTIndicator as usize] = t_indicator;
                constraint_claims[PolyType::G1ScalarMulAIndicator as usize] = a_indicator;
                g1_scalar_mul_idx += 1;
            }
            ConstraintType::G2ScalarMul { .. } => {
                let (_, x_a_c0) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_xa_c0(g2_scalar_mul_idx),
                    SumcheckId::G2ScalarMul,
                );
                let (_, x_a_c1) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_xa_c1(g2_scalar_mul_idx),
                    SumcheckId::G2ScalarMul,
                );
                let (_, y_a_c0) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_ya_c0(g2_scalar_mul_idx),
                    SumcheckId::G2ScalarMul,
                );
                let (_, y_a_c1) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_ya_c1(g2_scalar_mul_idx),
                    SumcheckId::G2ScalarMul,
                );
                let (_, x_t_c0) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_xt_c0(g2_scalar_mul_idx),
                    SumcheckId::G2ScalarMul,
                );
                let (_, x_t_c1) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_xt_c1(g2_scalar_mul_idx),
                    SumcheckId::G2ScalarMul,
                );
                let (_, y_t_c0) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_yt_c0(g2_scalar_mul_idx),
                    SumcheckId::G2ScalarMul,
                );
                let (_, y_t_c1) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_yt_c1(g2_scalar_mul_idx),
                    SumcheckId::G2ScalarMul,
                );
                let (_, x_a_next_c0) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_xa_next_c0(g2_scalar_mul_idx),
                    SumcheckId::G2ScalarMul,
                );
                let (_, x_a_next_c1) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_xa_next_c1(g2_scalar_mul_idx),
                    SumcheckId::G2ScalarMul,
                );
                let (_, y_a_next_c0) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_ya_next_c0(g2_scalar_mul_idx),
                    SumcheckId::G2ScalarMul,
                );
                let (_, y_a_next_c1) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_ya_next_c1(g2_scalar_mul_idx),
                    SumcheckId::G2ScalarMul,
                );
                let (_, t_indicator) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_t_indicator(g2_scalar_mul_idx),
                    SumcheckId::G2ScalarMul,
                );
                let (_, a_indicator) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_scalar_mul_a_indicator(g2_scalar_mul_idx),
                    SumcheckId::G2ScalarMul,
                );

                constraint_claims[PolyType::G2ScalarMulXAC0 as usize] = x_a_c0;
                constraint_claims[PolyType::G2ScalarMulXAC1 as usize] = x_a_c1;
                constraint_claims[PolyType::G2ScalarMulYAC0 as usize] = y_a_c0;
                constraint_claims[PolyType::G2ScalarMulYAC1 as usize] = y_a_c1;
                constraint_claims[PolyType::G2ScalarMulXTC0 as usize] = x_t_c0;
                constraint_claims[PolyType::G2ScalarMulXTC1 as usize] = x_t_c1;
                constraint_claims[PolyType::G2ScalarMulYTC0 as usize] = y_t_c0;
                constraint_claims[PolyType::G2ScalarMulYTC1 as usize] = y_t_c1;
                constraint_claims[PolyType::G2ScalarMulXANextC0 as usize] = x_a_next_c0;
                constraint_claims[PolyType::G2ScalarMulXANextC1 as usize] = x_a_next_c1;
                constraint_claims[PolyType::G2ScalarMulYANextC0 as usize] = y_a_next_c0;
                constraint_claims[PolyType::G2ScalarMulYANextC1 as usize] = y_a_next_c1;
                constraint_claims[PolyType::G2ScalarMulTIndicator as usize] = t_indicator;
                constraint_claims[PolyType::G2ScalarMulAIndicator as usize] = a_indicator;
                g2_scalar_mul_idx += 1;
            }
            ConstraintType::G1Add => {
                let (_, x_p) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_xp(g1_add_idx),
                    SumcheckId::G1Add,
                );
                let (_, y_p) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_yp(g1_add_idx),
                    SumcheckId::G1Add,
                );
                let (_, ind_p) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_p_indicator(g1_add_idx),
                    SumcheckId::G1Add,
                );
                let (_, x_q) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_xq(g1_add_idx),
                    SumcheckId::G1Add,
                );
                let (_, y_q) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_yq(g1_add_idx),
                    SumcheckId::G1Add,
                );
                let (_, ind_q) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_q_indicator(g1_add_idx),
                    SumcheckId::G1Add,
                );
                let (_, x_r) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_xr(g1_add_idx),
                    SumcheckId::G1Add,
                );
                let (_, y_r) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_yr(g1_add_idx),
                    SumcheckId::G1Add,
                );
                let (_, ind_r) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_r_indicator(g1_add_idx),
                    SumcheckId::G1Add,
                );
                let (_, lambda) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_lambda(g1_add_idx),
                    SumcheckId::G1Add,
                );
                let (_, inv_delta_x) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_inv_delta_x(g1_add_idx),
                    SumcheckId::G1Add,
                );
                let (_, is_double) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_is_double(g1_add_idx),
                    SumcheckId::G1Add,
                );
                let (_, is_inverse) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g1_add_is_inverse(g1_add_idx),
                    SumcheckId::G1Add,
                );

                constraint_claims[PolyType::G1AddXP as usize] = x_p;
                constraint_claims[PolyType::G1AddYP as usize] = y_p;
                constraint_claims[PolyType::G1AddPIndicator as usize] = ind_p;
                constraint_claims[PolyType::G1AddXQ as usize] = x_q;
                constraint_claims[PolyType::G1AddYQ as usize] = y_q;
                constraint_claims[PolyType::G1AddQIndicator as usize] = ind_q;
                constraint_claims[PolyType::G1AddXR as usize] = x_r;
                constraint_claims[PolyType::G1AddYR as usize] = y_r;
                constraint_claims[PolyType::G1AddRIndicator as usize] = ind_r;
                constraint_claims[PolyType::G1AddLambda as usize] = lambda;
                constraint_claims[PolyType::G1AddInvDeltaX as usize] = inv_delta_x;
                constraint_claims[PolyType::G1AddIsDouble as usize] = is_double;
                constraint_claims[PolyType::G1AddIsInverse as usize] = is_inverse;
                g1_add_idx += 1;
            }
            ConstraintType::G2Add => {
                let (_, x_p_c0) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_xp_c0(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, x_p_c1) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_xp_c1(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, y_p_c0) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_yp_c0(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, y_p_c1) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_yp_c1(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, ind_p) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_p_indicator(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, x_q_c0) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_xq_c0(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, x_q_c1) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_xq_c1(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, y_q_c0) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_yq_c0(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, y_q_c1) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_yq_c1(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, ind_q) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_q_indicator(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, x_r_c0) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_xr_c0(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, x_r_c1) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_xr_c1(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, y_r_c0) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_yr_c0(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, y_r_c1) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_yr_c1(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, ind_r) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_r_indicator(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, lambda_c0) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_lambda_c0(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, lambda_c1) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_lambda_c1(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, inv_delta_x_c0) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_inv_delta_x_c0(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, inv_delta_x_c1) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_inv_delta_x_c1(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, is_double) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_is_double(g2_add_idx),
                    SumcheckId::G2Add,
                );
                let (_, is_inverse) = accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::g2_add_is_inverse(g2_add_idx),
                    SumcheckId::G2Add,
                );

                constraint_claims[PolyType::G2AddXPC0 as usize] = x_p_c0;
                constraint_claims[PolyType::G2AddXPC1 as usize] = x_p_c1;
                constraint_claims[PolyType::G2AddYPC0 as usize] = y_p_c0;
                constraint_claims[PolyType::G2AddYPC1 as usize] = y_p_c1;
                constraint_claims[PolyType::G2AddPIndicator as usize] = ind_p;

                constraint_claims[PolyType::G2AddXQC0 as usize] = x_q_c0;
                constraint_claims[PolyType::G2AddXQC1 as usize] = x_q_c1;
                constraint_claims[PolyType::G2AddYQC0 as usize] = y_q_c0;
                constraint_claims[PolyType::G2AddYQC1 as usize] = y_q_c1;
                constraint_claims[PolyType::G2AddQIndicator as usize] = ind_q;

                constraint_claims[PolyType::G2AddXRC0 as usize] = x_r_c0;
                constraint_claims[PolyType::G2AddXRC1 as usize] = x_r_c1;
                constraint_claims[PolyType::G2AddYRC0 as usize] = y_r_c0;
                constraint_claims[PolyType::G2AddYRC1 as usize] = y_r_c1;
                constraint_claims[PolyType::G2AddRIndicator as usize] = ind_r;

                constraint_claims[PolyType::G2AddLambdaC0 as usize] = lambda_c0;
                constraint_claims[PolyType::G2AddLambdaC1 as usize] = lambda_c1;
                constraint_claims[PolyType::G2AddInvDeltaXC0 as usize] = inv_delta_x_c0;
                constraint_claims[PolyType::G2AddInvDeltaXC1 as usize] = inv_delta_x_c1;
                constraint_claims[PolyType::G2AddIsDouble as usize] = is_double;
                constraint_claims[PolyType::G2AddIsInverse as usize] = is_inverse;
                g2_add_idx += 1;
            }
            #[cfg(feature = "experimental-pairing-recursion")]
            ConstraintType::MultiMillerLoop => {
                let vp = |term: MultiMillerLoopTerm| -> VirtualPolynomial {
                    VirtualPolynomial::Recursion(RecursionPoly::MultiMillerLoop {
                        term,
                        instance: multi_miller_loop_idx,
                    })
                };

                let (_, f) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::F),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, f_next) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::FNext),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, quotient) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::Quotient),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, t_x_c0) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::TXC0),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, t_x_c1) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::TXC1),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, t_y_c0) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::TYC0),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, t_y_c1) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::TYC1),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, t_x_c0_next) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::TXC0Next),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, t_x_c1_next) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::TXC1Next),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, t_y_c0_next) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::TYC0Next),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, t_y_c1_next) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::TYC1Next),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, lambda_c0) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::LambdaC0),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, lambda_c1) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::LambdaC1),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, inv_dx_c0) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::InvDeltaXC0),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, inv_dx_c1) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::InvDeltaXC1),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, x_p) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::XP),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, y_p) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::YP),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, x_q_c0) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::XQC0),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, x_q_c1) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::XQC1),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, y_q_c0) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::YQC0),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, y_q_c1) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::YQC1),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, is_double) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::IsDouble),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, is_add) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::IsAdd),
                    SumcheckId::MultiMillerLoop,
                );
                let (_, l_val) = accumulator.get_virtual_polynomial_opening(
                    vp(MultiMillerLoopTerm::LVal),
                    SumcheckId::MultiMillerLoop,
                );

                constraint_claims[PolyType::MultiMillerLoopF as usize] = f;
                constraint_claims[PolyType::MultiMillerLoopFNext as usize] = f_next;
                constraint_claims[PolyType::MultiMillerLoopQuotient as usize] = quotient;
                constraint_claims[PolyType::MultiMillerLoopTXC0 as usize] = t_x_c0;
                constraint_claims[PolyType::MultiMillerLoopTXC1 as usize] = t_x_c1;
                constraint_claims[PolyType::MultiMillerLoopTYC0 as usize] = t_y_c0;
                constraint_claims[PolyType::MultiMillerLoopTYC1 as usize] = t_y_c1;
                constraint_claims[PolyType::MultiMillerLoopTXC0Next as usize] = t_x_c0_next;
                constraint_claims[PolyType::MultiMillerLoopTXC1Next as usize] = t_x_c1_next;
                constraint_claims[PolyType::MultiMillerLoopTYC0Next as usize] = t_y_c0_next;
                constraint_claims[PolyType::MultiMillerLoopTYC1Next as usize] = t_y_c1_next;
                constraint_claims[PolyType::MultiMillerLoopLambdaC0 as usize] = lambda_c0;
                constraint_claims[PolyType::MultiMillerLoopLambdaC1 as usize] = lambda_c1;
                constraint_claims[PolyType::MultiMillerLoopInvDeltaXC0 as usize] = inv_dx_c0;
                constraint_claims[PolyType::MultiMillerLoopInvDeltaXC1 as usize] = inv_dx_c1;
                constraint_claims[PolyType::MultiMillerLoopXP as usize] = x_p;
                constraint_claims[PolyType::MultiMillerLoopYP as usize] = y_p;
                constraint_claims[PolyType::MultiMillerLoopXQC0 as usize] = x_q_c0;
                constraint_claims[PolyType::MultiMillerLoopXQC1 as usize] = x_q_c1;
                constraint_claims[PolyType::MultiMillerLoopYQC0 as usize] = y_q_c0;
                constraint_claims[PolyType::MultiMillerLoopYQC1 as usize] = y_q_c1;
                constraint_claims[PolyType::MultiMillerLoopIsDouble as usize] = is_double;
                constraint_claims[PolyType::MultiMillerLoopIsAdd as usize] = is_add;
                constraint_claims[PolyType::MultiMillerLoopLVal as usize] = l_val;

                multi_miller_loop_idx += 1;
            }
        }

        claims.extend(constraint_claims);
    }

    claims
}
