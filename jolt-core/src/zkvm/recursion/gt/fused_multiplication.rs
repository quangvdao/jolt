//! Fused GT multiplication sumcheck (over GT-local constraint index + u variables).
//!
//! End-to-end GT fusion packs GT mul witness polynomials as fused MLEs over `(u, c_gt)`,
//! where `c_gt` ranges over only `{GtExp,GtMul}` constraints (in global order), and `u` is
//! the native 4-var GT element domain (NO replication over step bits).
//!
//! Variable order for this sumcheck instance (round order, `BindingOrder::LowToHigh`):
//! - first 4 rounds bind the element variables `u` (LSB first)
//! - last `k_gt` rounds bind the GT-local constraint index `c_gt` (LSB first, as a suffix in Stage 2)

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
    zkvm::recursion::constraints::system::{index_to_binary, ConstraintType},
    zkvm::recursion::gt::indexing::gt_mul_c_tail_range,
    zkvm::recursion::gt::types::GtMulConstraintPolynomials,
    zkvm::witness::VirtualPolynomial,
};

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use rayon::prelude::*;

/// Degree bound for GTMul constraint polynomial: eq * ind * (lhs*rhs - result - quotient*g)
/// Each term: eq (deg 1) * ind (deg 1) * constraint (deg 2) = deg 4.
const DEGREE: usize = 4;

#[derive(Clone, Allocative)]
pub struct FusedGtMulParams {
    /// Number of c-index variables in Stage 2 (k_common).
    ///
    /// This is the shared suffix length used for GT in Stage 2 batching.
    pub num_constraint_index_vars_common: usize,
    /// Number of c-index variables actually used by the committed GTMul fused rows (k_mul).
    pub num_constraint_index_vars_family: usize,
    pub num_constraint_vars: usize, // 4 (u-vars)
    /// Number of GTMul constraints (family-local).
    pub num_gt_constraints: usize,
    pub num_gt_constraints_padded: usize,
}

impl FusedGtMulParams {
    pub fn new(
        num_gt_constraints: usize,
        num_gt_constraints_padded: usize,
        k_common: usize,
    ) -> Self {
        debug_assert!(num_gt_constraints_padded.is_power_of_two());
        let num_constraint_index_vars_family = num_gt_constraints_padded.trailing_zeros() as usize;
        debug_assert!(
            k_common >= num_constraint_index_vars_family,
            "k_common must be >= k_mul"
        );
        Self {
            num_constraint_index_vars_common: k_common,
            num_constraint_index_vars_family,
            num_constraint_vars: 4,
            num_gt_constraints,
            num_gt_constraints_padded,
        }
    }

    #[inline]
    pub fn num_rounds(&self) -> usize {
        self.num_constraint_index_vars_common + self.num_constraint_vars
    }

    #[inline]
    pub fn dummy_c_bits(&self) -> usize {
        self.num_constraint_index_vars_common - self.num_constraint_index_vars_family
    }
}

impl SumcheckInstanceParams<Fq> for FusedGtMulParams {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<Fq>) -> Fq {
        // Prove the (Eq-weighted) sum is 0.
        Fq::zero()
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<Fq as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, Fq> {
        // Opening point must match committed fused row arity: (u, c_mul_tail).
        debug_assert_eq!(challenges.len(), self.num_rounds());
        let u_vars = self.num_constraint_vars; // 4
        let mut r = Vec::with_capacity(u_vars + self.num_constraint_index_vars_family);
        r.extend_from_slice(&challenges[..u_vars]);
        let tail = gt_mul_c_tail_range(
            self.num_constraint_index_vars_common,
            self.num_constraint_index_vars_family,
        );
        r.extend_from_slice(&challenges[tail]);
        OpeningPoint::<BIG_ENDIAN, Fq>::new(r)
    }
}

#[derive(Allocative)]
pub struct FusedGtMulProver {
    params: FusedGtMulParams,
    eq_poly: MultilinearPolynomial<Fq>,
    indicator_poly: MultilinearPolynomial<Fq>,
    lhs: MultilinearPolynomial<Fq>,
    rhs: MultilinearPolynomial<Fq>,
    result: MultilinearPolynomial<Fq>,
    quotient: MultilinearPolynomial<Fq>,
    g_poly: MultilinearPolynomial<Fq>,
}

impl FusedGtMulProver {
    pub fn new<T: Transcript>(
        params: FusedGtMulParams,
        constraint_types: &[ConstraintType],
        gt_mul_rows: &[GtMulConstraintPolynomials<Fq>],
        g_poly_4var: &DensePolynomial<Fq>,
        transcript: &mut T,
    ) -> Self {
        let num_rounds = params.num_rounds();
        let row_size = 1usize << params.num_constraint_vars; // 16

        // Sample eq_point for the fused (u, c_gt) domain.
        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let eq_poly = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&eq_point));

        // Family-local indexing: c indexes GTMul instances only, in GTMul-local order.
        // The constraint_types arg is unused here (kept for signature stability).
        let _ = constraint_types;
        debug_assert_eq!(
            params.num_gt_constraints,
            gt_mul_rows.len(),
            "FusedGtMulParams.num_gt_constraints must match gt_mul_rows.len()"
        );

        // Indicator table in [u_low, c_high] layout (c is the high bits).
        let mut ind_uc = vec![Fq::zero(); params.num_gt_constraints_padded * row_size];
        for c in 0..params.num_gt_constraints_padded {
            if c < params.num_gt_constraints {
                let off = c * row_size;
                for u in 0..row_size {
                    ind_uc[off + u] = Fq::one();
                }
            }
        }
        let indicator_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(ind_uc));

        // Build a full (u, c_gt) g polynomial table (independent of c_gt).
        let g4 = &g_poly_4var.Z;
        debug_assert_eq!(g4.len(), 1usize << 4);
        let mut g_uc = vec![Fq::zero(); params.num_gt_constraints_padded * row_size];
        for c in 0..params.num_gt_constraints_padded {
            let off = c * row_size;
            g_uc[off..off + row_size].copy_from_slice(g4);
        }
        let g_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(g_uc));

        // Helper to build a fused GTMul term table (native 4 vars).
        let build_term = |get_term4: fn(&GtMulConstraintPolynomials<Fq>) -> &Vec<Fq>| {
            let mut term_uc = vec![Fq::zero(); params.num_gt_constraints_padded * row_size];
            for c in 0..params.num_gt_constraints_padded {
                if c >= params.num_gt_constraints {
                    continue;
                }
                let local = c;
                let src4 = get_term4(&gt_mul_rows[local]);
                debug_assert_eq!(src4.len(), 1usize << 4);
                let off = c * row_size;
                term_uc[off..off + row_size].copy_from_slice(src4);
            }
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(term_uc))
        };

        let lhs = build_term(|r| &r.lhs);
        let rhs = build_term(|r| &r.rhs);
        let result = build_term(|r| &r.result);
        let quotient = build_term(|r| &r.quotient);

        Self {
            params,
            eq_poly,
            indicator_poly,
            lhs,
            rhs,
            result,
            quotient,
            g_poly,
        }
    }
}

impl<FqT: Transcript> SumcheckInstanceProver<Fq, FqT> for FusedGtMulProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn compute_message(&mut self, round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        // Dummy c rounds (the first k_common-k_mul bits of the c-suffix) are treated as
        // variables the polynomial does not depend on, so we emit a constant univariate.
        let u_vars = self.params.num_constraint_vars; // 4
        let dummy_start = u_vars;
        let dummy_end = u_vars + self.params.dummy_c_bits();
        if (dummy_start..dummy_end).contains(&round) {
            let two_inv = Fq::from_u64(2).inverse().unwrap();
            return UniPoly::from_coeff(vec![previous_claim * two_inv]);
        }

        let num_remaining = self.eq_poly.get_num_vars();
        debug_assert!(
            num_remaining > 0,
            "fused gtmul should have at least one round"
        );
        let half = 1usize << (num_remaining - 1);

        let total_evals: [Fq; DEGREE] = (0..half)
            .into_par_iter()
            .map(|idx| {
                let eq_evals = self
                    .eq_poly
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let ind_evals = self
                    .indicator_poly
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let lhs_e = self
                    .lhs
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let rhs_e = self
                    .rhs
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let res_e = self
                    .result
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let quo_e = self
                    .quotient
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let g_e = self
                    .g_poly
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);

                let mut out = [Fq::zero(); DEGREE];
                for eval_index in 0..DEGREE {
                    let c_val = lhs_e[eval_index] * rhs_e[eval_index]
                        - res_e[eval_index]
                        - quo_e[eval_index] * g_e[eval_index];
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

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, _round: usize) {
        // Skip bindings on dummy c rounds (same logic as compute_message).
        let u_vars = self.params.num_constraint_vars; // 4
        let dummy_start = u_vars;
        let dummy_end = u_vars + self.params.dummy_c_bits();
        if (dummy_start..dummy_end).contains(&_round) {
            return;
        }

        self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.indicator_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.lhs.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.rhs.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.result.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.quotient.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.g_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut FqT,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::gt_mul_lhs_fused(),
            SumcheckId::GtMul,
            opening_point.clone(),
            self.lhs.get_bound_coeff(0),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::gt_mul_rhs_fused(),
            SumcheckId::GtMul,
            opening_point.clone(),
            self.rhs.get_bound_coeff(0),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::gt_mul_result_fused(),
            SumcheckId::GtMul,
            opening_point.clone(),
            self.result.get_bound_coeff(0),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::gt_mul_quotient_fused(),
            SumcheckId::GtMul,
            opening_point,
            self.quotient.get_bound_coeff(0),
        );
    }
}

#[derive(Allocative)]
pub struct FusedGtMulVerifier {
    params: FusedGtMulParams,
    eq_point: Vec<<Fq as JoltField>::Challenge>,
    /// GT-local indices `c_gt` where the constraint is `GtMul`.
    gtmul_c_indices: Vec<usize>,
    g_mle_4var: Vec<Fq>,
}

impl FusedGtMulVerifier {
    pub fn new<T: Transcript>(
        params: FusedGtMulParams,
        constraint_types: &[ConstraintType],
        g_mle_4var: Vec<Fq>,
        transcript: &mut T,
    ) -> Self {
        let num_rounds = params.num_rounds();
        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        // Family-local indexing: c indexes GTMul instances only, in GTMul-local order.
        let num_gt_mul = constraint_types
            .iter()
            .filter(|ct| matches!(ct, ConstraintType::GtMul))
            .count();
        let gtmul_c_indices: Vec<usize> = (0..num_gt_mul).collect();
        Self {
            params,
            eq_point,
            gtmul_c_indices,
            g_mle_4var,
        }
    }

    fn eval_g_at_u(&self, r_u: &[Fq]) -> Fq {
        debug_assert_eq!(r_u.len(), 4);
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

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for FusedGtMulVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        let k_common = self.params.num_constraint_index_vars_common;
        let k_mul = self.params.num_constraint_index_vars_family;
        debug_assert_eq!(sumcheck_challenges.len(), 4 + k_common);

        // Effective (u, c_tail) slice for eq evaluation.
        let dummy = k_common - k_mul;
        let mut eff: Vec<<Fq as JoltField>::Challenge> = Vec::with_capacity(4 + k_mul);
        eff.extend_from_slice(&sumcheck_challenges[..4]);
        eff.extend_from_slice(&sumcheck_challenges[4 + dummy..]);

        let eval_point: Vec<Fq> = eff.iter().rev().map(|c| (*c).into()).collect();
        let mut eq_point_eff: Vec<<Fq as JoltField>::Challenge> = Vec::with_capacity(4 + k_mul);
        eq_point_eff.extend_from_slice(&self.eq_point[..4]);
        eq_point_eff.extend_from_slice(&self.eq_point[4 + dummy..]);
        let eq_point_f: Vec<Fq> = eq_point_eff.iter().map(|c| (*c).into()).collect();
        let eq_eval = EqPolynomial::mle(&eq_point_f, &eval_point);

        // Indicator I_gtmul(r_c) as Σ_{c in gtmul_c_indices} Eq(r_c, c).
        //
        // Variable order (LSB-first rounds) is (u, c_common), but the committed rows use only
        // the tail `k_mul` bits.
        //
        // So:
        // - u is the first 4 challenges
        // - c_tail is the last k_mul challenges of the c suffix
        let r_c: Vec<Fq> = sumcheck_challenges[4 + dummy..]
            .iter()
            .map(|c| (*c).into())
            .collect();
        let mut ind_eval = Fq::zero();
        for &c in &self.gtmul_c_indices {
            let bits = index_to_binary::<Fq>(c, k_mul);
            ind_eval += EqPolynomial::mle(&r_c, &bits);
        }

        // Extract u (first 4 bits) and evaluate g(u).
        let r_u: Vec<Fq> = sumcheck_challenges[..4]
            .iter()
            .map(|c| (*c).into())
            .collect();
        let g_eval = self.eval_g_at_u(&r_u);

        // Fetch fused opened claims.
        let (_, lhs) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_mul_lhs_fused(),
            SumcheckId::GtMul,
        );
        let (_, rhs) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_mul_rhs_fused(),
            SumcheckId::GtMul,
        );
        let (_, result) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_mul_result_fused(),
            SumcheckId::GtMul,
        );
        let (_, quotient) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_mul_quotient_fused(),
            SumcheckId::GtMul,
        );

        let constraint_value = lhs * rhs - result - quotient * g_eval;
        eq_eval * ind_eval * constraint_value
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        for vp in [
            VirtualPolynomial::gt_mul_lhs_fused(),
            VirtualPolynomial::gt_mul_rhs_fused(),
            VirtualPolynomial::gt_mul_result_fused(),
            VirtualPolynomial::gt_mul_quotient_fused(),
        ] {
            accumulator.append_virtual(transcript, vp, SumcheckId::GtMul, opening_point.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_opening_point_drops_dummy_c_bits() {
        // k_common = 5, k_mul = 3 => dummy = 2
        let k_common = 5usize;
        let num_gt_constraints = 3usize;
        let num_gt_constraints_padded = 1usize << 3; // k_mul = 3
        let params = FusedGtMulParams::new(num_gt_constraints, num_gt_constraints_padded, k_common);

        // challenges = [u0,u1,u2,u3, c0,c1, c2,c3,c4] where c0,c1 are dummy
        let challenges: Vec<<Fq as JoltField>::Challenge> = (0..(4 + k_common))
            .map(|i| Fq::from_u64(i as u64).into())
            .collect();

        let p = params.normalize_opening_point(&challenges);
        // Expected: [u0..u3, c2,c3,c4]
        assert_eq!(p.r.len(), 4 + 3);
        for i in 0..4 {
            assert_eq!(p.r[i], challenges[i]);
        }
        assert_eq!(p.r[4], challenges[6]);
        assert_eq!(p.r[5], challenges[7]);
        assert_eq!(p.r[6], challenges[8]);
    }
}
