//! GTExp base-power correctness sumcheck (Stage 2).
//!
//! This instance proves that the committed GTExp base-power rows satisfy the same pointwise
//! multiplication relation used by GTMul, but over the **GTExp-local** index space `c_exp`.
//!
//! Concretely, over `(u, c_common) ∈ {0,1}^{4 + k_common}`, where `c_common` has `k_common = k_gt`
//! bits and `c_tail` is the last `k_exp` bits (family-local), we prove:
//!
//! ```text
//! Σ_{u,c_common} eq(r, (u,c_common)) · I_gtexp(c_common) ·
//!   [ (B·B - B2 - Q2·g) + β·(B2·B - B3 - Q3·g) ](u, c_tail) = 0
//! ```
//!
//! where:
//! - `B, B2, B3` are the committed GTExp base-power rows over `(u, c_exp)`,
//! - `Q2, Q3` are committed quotient rows over `(u, c_exp)`,
//! - `g(u)` is the public 4-var `g` polynomial,
//! - `I_gtexp` gates padding rows (`c_exp >= num_gt_exp`),
//! - `β` is a transcript-derived scalar batching the square/cube checks.
//!
//! This lets the verifier avoid recomputing `base^2/base^3` from public inputs in the guest.

use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    zkvm::recursion::constraints::system::eq_lsb_index,
    zkvm::recursion::constraints::system::{ConstraintLocator, ConstraintType},
    zkvm::recursion::gt::indexing::{gt_mul_c_tail_range, k_exp, k_gt, num_gt_exp_constraints, num_gt_exp_constraints_padded},
    zkvm::recursion::gt::types::GtExpWitness,
    zkvm::witness::{GtExpTerm, RecursionPoly, VirtualPolynomial},
};

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use rayon::prelude::*;

const U_VARS: usize = 4;
const DEGREE: usize = 4;
const STEP_STRIDE: usize = 1usize << 7; // 2^STEP_VARS (STEP_VARS = 7)

#[derive(Clone, Allocative)]
pub struct GtExpBasePowParams {
    pub k_common: usize,
    pub k_exp: usize,
    pub num_gt_exp: usize,
    pub num_gt_exp_padded: usize,
}

impl GtExpBasePowParams {
    pub fn from_constraint_types(constraint_types: &[ConstraintType]) -> Self {
        let k_common = k_gt(constraint_types);
        let k_exp = k_exp(constraint_types);
        let num_gt_exp = num_gt_exp_constraints(constraint_types);
        let num_gt_exp_padded = num_gt_exp_constraints_padded(constraint_types);
        Self {
            k_common,
            k_exp,
            num_gt_exp,
            num_gt_exp_padded,
        }
    }

    #[inline]
    pub fn dummy_c_bits(&self) -> usize {
        self.k_common - self.k_exp
    }
}

impl SumcheckInstanceParams<Fq> for GtExpBasePowParams {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        U_VARS + self.k_common
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<Fq>) -> Fq {
        // Prove the (Eq-weighted) sum is 0.
        Fq::zero()
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<Fq as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, Fq> {
        // Opening point must match committed row arity: (u, c_exp_tail).
        debug_assert_eq!(challenges.len(), U_VARS + self.k_common);
        let mut r = Vec::with_capacity(U_VARS + self.k_exp);
        r.extend_from_slice(&challenges[..U_VARS]);
        let tail = gt_mul_c_tail_range(self.k_common, self.k_exp);
        r.extend_from_slice(&challenges[tail]);
        OpeningPoint::<BIG_ENDIAN, Fq>::new(r)
    }
}

#[derive(Allocative)]
pub struct GtExpBasePowProver {
    params: GtExpBasePowParams,
    eq_poly: MultilinearPolynomial<Fq>,
    indicator_poly: MultilinearPolynomial<Fq>,
    base: MultilinearPolynomial<Fq>,
    base2: MultilinearPolynomial<Fq>,
    base3: MultilinearPolynomial<Fq>,
    q2: MultilinearPolynomial<Fq>,
    q3: MultilinearPolynomial<Fq>,
    g_poly: MultilinearPolynomial<Fq>,
    beta: Fq,
}

impl GtExpBasePowProver {
    pub fn new<T: Transcript>(
        constraint_types: &[ConstraintType],
        locator_by_constraint: &[ConstraintLocator],
        gt_exp_witnesses: &[GtExpWitness<Fq>],
        g_poly_4var: &DensePolynomial<Fq>,
        transcript: &mut T,
    ) -> Self {
        let params = GtExpBasePowParams::from_constraint_types(constraint_types);
        let num_rounds = params.num_rounds();
        let row_size = 1usize << U_VARS; // 16

        // Sample eq_point for the (u, c_common) domain.
        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let eq_poly = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&eq_point));

        // Batch square/cube checks with a transcript scalar.
        let beta: Fq = transcript.challenge_scalar_optimized::<Fq>().into();

        // Indicator table in [u_low, c_high] layout (c is the high bits).
        let mut ind_uc = vec![Fq::zero(); params.num_gt_exp_padded * row_size];
        for c in 0..params.num_gt_exp_padded {
            if c < params.num_gt_exp {
                let off = c * row_size;
                for u in 0..row_size {
                    ind_uc[off + u] = Fq::one();
                }
            }
        }
        let indicator_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(ind_uc));

        // Build a full (u, c_exp) g polynomial table (independent of c_exp).
        debug_assert_eq!(g_poly_4var.Z.len(), row_size);
        let g4 = &g_poly_4var.Z;
        let mut g_uc = vec![Fq::zero(); params.num_gt_exp_padded * row_size];
        for c in 0..params.num_gt_exp_padded {
            let off = c * row_size;
            g_uc[off..off + row_size].copy_from_slice(g4);
        }
        let g_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(g_uc));

        // Helper to build a GTExp base-power term table (native 4 vars), stacked over c_exp.
        let build_term_from_packed = |get_src11: fn(&GtExpWitness<Fq>) -> &Vec<Fq>| {
            let mut term_uc = vec![Fq::zero(); params.num_gt_exp_padded * row_size];
            for global_idx in 0..constraint_types.len() {
                if let ConstraintLocator::GtExp { local } = locator_by_constraint[global_idx] {
                    let src11 = get_src11(&gt_exp_witnesses[local]);
                    debug_assert_eq!(src11.len(), 1usize << 11);
                    let off = local * row_size;
                    for u in 0..row_size {
                        term_uc[off + u] = src11[u * STEP_STRIDE];
                    }
                }
            }
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(term_uc))
        };

        let base = build_term_from_packed(|w| &w.base_packed);
        let base2 = build_term_from_packed(|w| &w.base2_packed);
        let base3 = build_term_from_packed(|w| &w.base3_packed);

        // Quotients are committed rows as well; derive them deterministically from (base,base2,base3,g).
        // NOTE: This mirrors `stage2_base_openings` and `emit_dense` behavior.
        let MultilinearPolynomial::LargeScalars(base_dense) = &base else {
            unreachable!("expected LargeScalars base polynomial")
        };
        let MultilinearPolynomial::LargeScalars(base2_dense) = &base2 else {
            unreachable!("expected LargeScalars base2 polynomial")
        };
        let MultilinearPolynomial::LargeScalars(base3_dense) = &base3 else {
            unreachable!("expected LargeScalars base3 polynomial")
        };
        let MultilinearPolynomial::LargeScalars(g_dense) = &g_poly else {
            unreachable!("expected LargeScalars g polynomial")
        };
        let mut q2_uc = vec![Fq::zero(); params.num_gt_exp_padded * row_size];
        let mut q3_uc = vec![Fq::zero(); params.num_gt_exp_padded * row_size];
        for i in 0..q2_uc.len() {
            let g = g_dense.Z[i];
            let b = base_dense.Z[i];
            let b2 = base2_dense.Z[i];
            let b3 = base3_dense.Z[i];
            if g.is_zero() {
                q2_uc[i] = Fq::zero();
                q3_uc[i] = Fq::zero();
            } else {
                let inv_g = g.inverse().unwrap();
                q2_uc[i] = (b * b - b2) * inv_g;
                q3_uc[i] = (b2 * b - b3) * inv_g;
            }
        }
        let q2 = MultilinearPolynomial::LargeScalars(DensePolynomial::new(q2_uc));
        let q3 = MultilinearPolynomial::LargeScalars(DensePolynomial::new(q3_uc));

        Self {
            params,
            eq_poly,
            indicator_poly,
            base,
            base2,
            base3,
            q2,
            q3,
            g_poly,
            beta,
        }
    }
}

impl<FqT: Transcript> SumcheckInstanceProver<Fq, FqT> for GtExpBasePowProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn compute_message(&mut self, round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        // Dummy c rounds (the first k_common-k_exp bits of the c-suffix) are treated as
        // variables the polynomial does not depend on, so we emit a constant univariate.
        let dummy_start = U_VARS;
        let dummy_end = U_VARS + self.params.dummy_c_bits();
        if (dummy_start..dummy_end).contains(&round) {
            let two_inv = Fq::from_u64(2).inverse().unwrap();
            return UniPoly::from_coeff(vec![previous_claim * two_inv]);
        }

        let num_remaining = self.eq_poly.get_num_vars();
        debug_assert!(num_remaining > 0, "base_pow should have at least one round");
        let half = 1usize << (num_remaining - 1);

        let beta = self.beta;
        let total_evals: [Fq; DEGREE] = (0..half)
            .into_par_iter()
            .map(|idx| {
                let eq_evals = self
                    .eq_poly
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let ind_evals = self
                    .indicator_poly
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let b_e = self
                    .base
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let b2_e = self
                    .base2
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let b3_e = self
                    .base3
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let q2_e = self
                    .q2
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let q3_e = self
                    .q3
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let g_e = self
                    .g_poly
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);

                let mut out = [Fq::zero(); DEGREE];
                for eval_index in 0..DEGREE {
                    let sq = b_e[eval_index] * b_e[eval_index]
                        - b2_e[eval_index]
                        - q2_e[eval_index] * g_e[eval_index];
                    let cu = b2_e[eval_index] * b_e[eval_index]
                        - b3_e[eval_index]
                        - q3_e[eval_index] * g_e[eval_index];
                    let c_val = sq + beta * cu;
                    out[eval_index] = eq_evals[eval_index] * ind_evals[eval_index] * c_val;
                }
                out
            })
            .reduce(
                || [Fq::zero(); DEGREE],
                |mut acc, arr| {
                    for i in 0..DEGREE {
                        acc[i] += arr[i];
                    }
                    acc
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &total_evals)
    }

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, round: usize) {
        // Skip bindings on dummy c rounds (same logic as compute_message).
        let dummy_start = U_VARS;
        let dummy_end = U_VARS + self.params.dummy_c_bits();
        if (dummy_start..dummy_end).contains(&round) {
            return;
        }

        self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.indicator_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        for p in [
            &mut self.base,
            &mut self.base2,
            &mut self.base3,
            &mut self.q2,
            &mut self.q3,
            &mut self.g_poly,
        ] {
            p.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<Fq>,
        _transcript: &mut FqT,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No new openings: we reuse cached base/base2/base3/q2/q3 from `GtExpBaseStage2Openings`.
    }
}

#[derive(Allocative)]
pub struct GtExpBasePowVerifier {
    params: GtExpBasePowParams,
    eq_point: Vec<<Fq as JoltField>::Challenge>,
    beta: Fq,
    gtexp_c_indices: Vec<usize>,
    g_mle_4var: Vec<Fq>,
}

impl GtExpBasePowVerifier {
    pub fn new<T: Transcript>(
        constraint_types: &[ConstraintType],
        g_mle_4var: Vec<Fq>,
        transcript: &mut T,
    ) -> Self {
        let params = GtExpBasePowParams::from_constraint_types(constraint_types);
        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..params.num_rounds())
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let beta: Fq = transcript.challenge_scalar_optimized::<Fq>().into();

        let gtexp_c_indices: Vec<usize> = (0..params.num_gt_exp).collect();
        Self {
            params,
            eq_point,
            beta,
            gtexp_c_indices,
            g_mle_4var,
        }
    }

    fn eval_g_at_u(&self, r_u: &[Fq]) -> Fq {
        debug_assert_eq!(r_u.len(), U_VARS);
        let mut evals = self.g_mle_4var.clone();
        let mut len = evals.len();
        for &r_i in r_u {
            let half = len / 2;
            for j in 0..half {
                let a = evals[2 * j];
                let b = evals[2 * j + 1];
                evals[j] = a + r_i * (b - a);
            }
            len = half;
        }
        debug_assert_eq!(len, 1);
        evals[0]
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for GtExpBasePowVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        let k_common = self.params.k_common;
        let k_exp = self.params.k_exp;
        debug_assert_eq!(sumcheck_challenges.len(), U_VARS + k_common);

        // Effective (u, c_tail) slice for eq evaluation (drop dummy c bits).
        let dummy = k_common - k_exp;
        let mut eff: Vec<<Fq as JoltField>::Challenge> = Vec::with_capacity(U_VARS + k_exp);
        eff.extend_from_slice(&sumcheck_challenges[..U_VARS]);
        eff.extend_from_slice(&sumcheck_challenges[U_VARS + dummy..]);

        let eval_point: Vec<Fq> = eff.iter().rev().map(|c| (*c).into()).collect();
        let mut eq_point_eff: Vec<<Fq as JoltField>::Challenge> = Vec::with_capacity(U_VARS + k_exp);
        eq_point_eff.extend_from_slice(&self.eq_point[..U_VARS]);
        eq_point_eff.extend_from_slice(&self.eq_point[U_VARS + dummy..]);
        let eq_point_f: Vec<Fq> = eq_point_eff.iter().map(|c| (*c).into()).collect();
        let eq_eval = EqPolynomial::mle(&eq_point_f, &eval_point);

        // Indicator I_gtexp(r_c) as Σ_{c in [0..num_gt_exp)} Eq(r_c, c) (over c_tail bits).
        let r_c: Vec<Fq> = sumcheck_challenges[U_VARS + dummy..]
            .iter()
            .map(|c| (*c).into())
            .collect();
        let mut ind_eval = Fq::zero();
        for &c in &self.gtexp_c_indices {
            ind_eval += eq_lsb_index(&r_c, c);
        }

        // Evaluate g(u) at r_u.
        let r_u: Vec<Fq> = sumcheck_challenges[..U_VARS]
            .iter()
            .map(|c| (*c).into())
            .collect();
        let g_eval = self.eval_g_at_u(&r_u);

        // Fetch opened claims for base polynomials (cached by `GtExpBaseStage2Openings`).
        let get = |term: GtExpTerm| {
            accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::GtExp { term }),
                SumcheckId::GtExpBaseClaimReduction,
            )
        };
        let b = get(GtExpTerm::Base);
        let b2 = get(GtExpTerm::Base2);
        let b3 = get(GtExpTerm::Base3);
        let q2 = get(GtExpTerm::BaseSquareQuotient);
        let q3 = get(GtExpTerm::BaseCubeQuotient);

        let sq = b * b - b2 - q2 * g_eval;
        let cu = b2 * b - b3 - q3 * g_eval;
        let constraint_value = sq + self.beta * cu;

        eq_eval * ind_eval * constraint_value
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No new openings: we reuse cached base/base2/base3/q2/q3 from `GtExpBaseStage2Openings`.
    }
}

