//! Fused shift sumcheck for verifying `rho_next_fused` consistency.
//!
//! GT-fused analogue of `gt/shift.rs`:
//! - Stage 1 (fused GTExp) emits `gt_exp_rho_next_fused()` at point `r1 = (r_s, r_u, r_c_exp)`.
//! - This sumcheck proves that `rho_next_fused(r1)` is consistent with the fused committed `rho`
//!   table via a one-step shift on the step variables.
//!
//! In end-to-end fusion we want **no per-instance GTExp openings**, so this protocol:
//! - takes the single Stage-1 fused `rho_next` claim as its input claim
//! - uses the fused rho table over `(x11, c_exp)` as its witness
//! - emits **no** additional openings (it is purely a check)
//!
//! Variable order (round order, `BindingOrder::LowToHigh`):
//! - first 7 rounds: step variables `s` (LSB first)
//! - next 4 rounds: element variables `u` (LSB first)
//! - last `k_gt` rounds: GT-local constraint index `c_gt` (LSB first, as a suffix)
use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
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
    zkvm::recursion::constraints::config::CONFIG,
    zkvm::recursion::constraints::system::{ConstraintLocator, ConstraintType},
    zkvm::recursion::gt::indexing::{k_gt, num_gt_constraints_padded},
    zkvm::recursion::gt::shift::{
        eq_lsb_evals, eq_lsb_mle, eq_plus_one_lsb_evals, eq_plus_one_lsb_mle,
    },
    zkvm::witness::VirtualPolynomial,
};

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::Zero;
use rayon::prelude::*;

/// Degree bound: we take a conservative bound (EqPlusOne * Eq * Eq * rho).
const DEGREE: usize = 4;

#[inline]
fn evals_skip_one<const D: usize>(a0: Fq, a1: Fq) -> [Fq; D] {
    let m = a1 - a0;
    let mut out = [Fq::zero(); D];
    out[0] = a0;
    for j in 1..D {
        let t = Fq::from_u64((j + 1) as u64); // 2..=D
        out[j] = a0 + t * m;
    }
    out
}

#[derive(Clone, Allocative)]
pub struct FusedGtShiftParams {
    pub num_c_vars: usize,
    pub num_step_vars: usize,
    pub num_elem_vars: usize,
    pub num_x_vars: usize,
    pub num_gt_constraints_padded: usize,
}

impl FusedGtShiftParams {
    pub fn from_constraint_types(constraint_types: &[ConstraintType]) -> Self {
        let num_gt_constraints_padded = num_gt_constraints_padded(constraint_types);
        Self {
            num_c_vars: k_gt(constraint_types),
            num_step_vars: CONFIG.step_vars,
            num_elem_vars: CONFIG.element_vars,
            num_x_vars: CONFIG.packed_vars,
            num_gt_constraints_padded,
        }
    }

    #[inline]
    pub fn num_rounds(&self) -> usize {
        self.num_c_vars + self.num_x_vars
    }
}

impl SumcheckInstanceParams<Fq> for FusedGtShiftParams {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<Fq>) -> Fq {
        // Provided by prover/verifier implementations (needs Stage-1 opening).
        Fq::zero()
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<Fq as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, Fq> {
        OpeningPoint::<BIG_ENDIAN, Fq>::new(challenges.to_vec())
    }
}

#[derive(Allocative)]
pub struct FusedGtShiftProver {
    params: FusedGtShiftParams,
    /// P(y) = Eq(r_c, c) * EqPlusOne(r_s, s) * Eq(r_u, u)  (as a fused (s,u,c) table)
    eq_prod: MultilinearPolynomial<Fq>,
    /// Fused rho(c,s,x) table.
    rho: MultilinearPolynomial<Fq>,
    /// Input claim = rho_next_fused(r1).
    input_claim: Fq,
}

impl FusedGtShiftProver {
    pub fn new(
        params: FusedGtShiftParams,
        constraint_types: &[ConstraintType],
        locator_by_constraint: &[ConstraintLocator],
        gt_exp_witnesses: &[crate::zkvm::recursion::gt::exponentiation::GtExpWitness<Fq>],
        accumulator: &ProverOpeningAccumulator<Fq>,
    ) -> Self {
        // Read Stage-1 fused rho_next claim and its opening point r1.
        let (r1_point, input_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_exp_rho_next_fused(),
            SumcheckId::GtExp,
        );
        let r1 = r1_point.r;
        debug_assert_eq!(r1.len(), params.num_rounds());

        let k = params.num_c_vars;
        let s_len = params.num_step_vars;
        let u_len = params.num_elem_vars;
        let r1_s = &r1[..s_len];
        let r1_x = &r1[s_len..s_len + u_len];
        let r1_c = &r1[s_len + u_len..];
        debug_assert_eq!(r1_c.len(), k);

        // Build fused rho(c_gt,x11) from GTExp witnesses (zero elsewhere).
        //
        // Split-k convention (shared with GT wiring): when `k_gt > k_exp`, the GTExp family
        // occupies only the tail `k_exp` bits of `c_gt`; the first `dummy = k_gt-k_exp` low bits
        // are dummy and rho is replicated across them.
        let row_size = 1usize << params.num_x_vars;
        let mut rho_xc = vec![Fq::zero(); params.num_gt_constraints_padded * row_size];
        let k_gt = params.num_c_vars;
        let k_exp = crate::zkvm::recursion::gt::indexing::k_exp(constraint_types);
        let dummy = k_gt.saturating_sub(k_exp);
        for global_idx in 0..constraint_types.len() {
            if let ConstraintLocator::GtExp { local } = locator_by_constraint[global_idx] {
                let src = &gt_exp_witnesses[local].rho_packed;
                debug_assert_eq!(src.len(), row_size);
                for d in 0..(1usize << dummy) {
                    let c = d + (local << dummy);
                    let off = c * row_size;
                    rho_xc[off..off + row_size].copy_from_slice(src);
                }
            }
        }
        // Store in [x11 low bits, c_gt high bits] order so `c_gt` is a suffix in Stage 2.
        let rho = MultilinearPolynomial::LargeScalars(DensePolynomial::new(rho_xc));

        // Build eq product table over (s,u,c) with LSB-first indexing for each component.
        let eq_c = eq_lsb_evals::<Fq>(r1_c);
        let eq_plus_one = eq_plus_one_lsb_evals::<Fq>(r1_s);
        let eq_x = eq_lsb_evals::<Fq>(r1_x);

        let mut eq_xc = vec![Fq::zero(); params.num_gt_constraints_padded * row_size];
        for c in 0..params.num_gt_constraints_padded {
            let eqc = eq_c.get(c).copied().unwrap_or_else(Fq::zero);
            let off = c * row_size;
            for x11 in 0..row_size {
                let s_idx = x11 & ((1usize << params.num_step_vars) - 1);
                let x_idx = x11 >> params.num_step_vars;
                let w = eqc * eq_plus_one[s_idx] * eq_x[x_idx];
                eq_xc[off + x11] = w;
            }
        }
        let eq_prod = MultilinearPolynomial::LargeScalars(DensePolynomial::new(eq_xc));

        Self {
            params,
            eq_prod,
            rho,
            input_claim,
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for FusedGtShiftProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<Fq>) -> Fq {
        self.input_claim
    }

    fn compute_message(&mut self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        let half = self.rho.len() / 2;
        if half == 0 {
            return UniPoly::from_evals_and_hint(previous_claim, &[Fq::zero(); DEGREE]);
        }

        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                let w = evals_skip_one::<DEGREE>(
                    self.eq_prod.get_bound_coeff(2 * i),
                    self.eq_prod.get_bound_coeff(2 * i + 1),
                );
                let rho = evals_skip_one::<DEGREE>(
                    self.rho.get_bound_coeff(2 * i),
                    self.rho.get_bound_coeff(2 * i + 1),
                );
                let mut out = [Fq::zero(); DEGREE];
                for t in 0..DEGREE {
                    out[t] = w[t] * rho[t];
                }
                out
            })
            .reduce(
                || [Fq::zero(); DEGREE],
                |mut acc, e| {
                    for t in 0..DEGREE {
                        acc[t] += e[t];
                    }
                    acc
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, _round: usize) {
        self.eq_prod.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.rho.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No-op: this check should not emit duplicate rho claims.
    }
}

#[derive(Allocative)]
pub struct FusedGtShiftVerifier {
    params: FusedGtShiftParams,
}

impl FusedGtShiftVerifier {
    pub fn new(params: FusedGtShiftParams) -> Self {
        Self { params }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for FusedGtShiftVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn input_claim(&self, acc: &VerifierOpeningAccumulator<Fq>) -> Fq {
        let (_pt, claim) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_exp_rho_next_fused(),
            SumcheckId::GtExp,
        );
        claim
    }

    fn expected_output_claim(
        &self,
        acc: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        debug_assert_eq!(sumcheck_challenges.len(), self.params.num_rounds());

        // Stage-1 point r1 comes from the fused rho_next opening under SumcheckId::GtExp.
        let (rho_next_point, _rho_next_claim) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_exp_rho_next_fused(),
            SumcheckId::GtExp,
        );
        let r1 = rho_next_point.r;
        debug_assert_eq!(r1.len(), self.params.num_rounds());

        let k = self.params.num_c_vars;
        let s_len = self.params.num_step_vars;
        let u_len = self.params.num_elem_vars;
        let r1_s = &r1[..s_len];
        let r1_x = &r1[s_len..s_len + u_len];
        let r1_c = &r1[s_len + u_len..];
        debug_assert_eq!(r1_c.len(), k);

        let r2_s = &sumcheck_challenges[..s_len];
        let r2_x = &sumcheck_challenges[s_len..s_len + u_len];
        let r2_c = &sumcheck_challenges[s_len + u_len..];

        let eq_c = eq_lsb_mle::<Fq>(r1_c, r2_c);
        let eq_plus_one = eq_plus_one_lsb_mle::<Fq>(r1_s, r2_s);
        let eq_x = eq_lsb_mle::<Fq>(r1_x, r2_x);
        let eq_prod = eq_c * eq_plus_one * eq_x;

        // Consume the fused rho opening at the Stage-2 point (emitted elsewhere).
        let (_pt, rho_eval) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_exp_rho_fused(),
            SumcheckId::GtExpClaimReduction,
        );

        eq_prod * rho_eval
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No-op.
    }
}
