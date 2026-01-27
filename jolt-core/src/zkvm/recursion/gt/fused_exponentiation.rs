//! Fused packed GT exponentiation sumcheck (over GT-local constraint index + packed (s,x) vars).
//!
//! Goal (end-to-end GT fusion): eliminate per-instance GTExp virtual openings from the recursion
//! proof payload by treating packed GTExp as a single fused polynomial over:
//! - `c_gt` : GT-local constraint index (only `{GtExp,GtMul}` in global order)
//! - `x11`  : packed GT exp domain (7 step bits + 4 element bits), layout `idx = x * 128 + s`
//!
//! The fused witness polynomials are formed by stacking per-instance packed tables into the
//! `c_gt` dimension, with zeros on non-GtExp `c_gt` and on padding rows.
//!
//! This mirrors the existing (per-instance) `GtExp` protocol but changes:\n//! - number of rounds: `k_gt + 11`\n//! - openings emitted: only `gt_exp_{rho,rho_next,quotient}_fused()` under `SumcheckId::GtExp`.

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
    zkvm::recursion::constraints::config::CONFIG,
    zkvm::recursion::constraints::system::{ConstraintLocator, ConstraintType},
    zkvm::recursion::curve::{Bn254Recursion, RecursionCurve},
    zkvm::recursion::gt::exponentiation::{GtExpPublicInputs, GtExpWitness},
    zkvm::recursion::gt::indexing::{gt_constraint_indices, k_gt, num_gt_constraints_padded},
    zkvm::witness::VirtualPolynomial,
};

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use rayon::prelude::*;

/// Degree bound matches existing packed GT exp (see `gt/exponentiation.rs`).
/// This bound is used by the batching sumcheck interface and by UniPoly compression.
const DEGREE: usize = 8; // Was 7, but eq*C has degree 1+7=8

#[inline]
fn transpose_xc_to_cx(input: &[Fq], num_constraints_padded: usize, row_size: usize) -> Vec<Fq> {
    debug_assert_eq!(input.len(), num_constraints_padded * row_size);
    let mut out = vec![Fq::zero(); input.len()];
    for c in 0..num_constraints_padded {
        let row_off = c * row_size;
        for x in 0..row_size {
            out[x * num_constraints_padded + c] = input[row_off + x];
        }
    }
    out
}

#[derive(Clone, Allocative)]
pub struct FusedGtExpParams {
    pub num_c_vars: usize,    // k_gt
    pub num_x_vars: usize,    // 11
    pub num_step_vars: usize, // 7
    pub num_elem_vars: usize, // 4
    pub num_gt_constraints: usize,
    pub num_gt_constraints_padded: usize,
}

impl FusedGtExpParams {
    pub fn from_constraint_types(constraint_types: &[ConstraintType]) -> Self {
        let num_gt_constraints_padded = num_gt_constraints_padded(constraint_types);
        let num_c_vars = k_gt(constraint_types);
        Self {
            num_c_vars,
            num_x_vars: CONFIG.packed_vars,
            num_step_vars: CONFIG.step_vars,
            num_elem_vars: CONFIG.element_vars,
            num_gt_constraints: constraint_types
                .iter()
                .filter(|ct| matches!(ct, ConstraintType::GtExp | ConstraintType::GtMul))
                .count(),
            num_gt_constraints_padded,
        }
    }

    #[inline]
    pub fn num_rounds(&self) -> usize {
        self.num_c_vars + self.num_x_vars
    }
}

impl SumcheckInstanceParams<Fq> for FusedGtExpParams {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<Fq>) -> Fq {
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
pub struct FusedGtExpProver {
    params: FusedGtExpParams,
    /// Eq polynomial evaluations for the sampled point over (c_gt,x11) domain.
    eq_poly: MultilinearPolynomial<Fq>,
    rho: MultilinearPolynomial<Fq>,
    rho_next: MultilinearPolynomial<Fq>,
    quotient: MultilinearPolynomial<Fq>,
    digit_lo: MultilinearPolynomial<Fq>,
    digit_hi: MultilinearPolynomial<Fq>,
    base: MultilinearPolynomial<Fq>,
    base2: MultilinearPolynomial<Fq>,
    base3: MultilinearPolynomial<Fq>,
    g: MultilinearPolynomial<Fq>,
}

impl FusedGtExpProver {
    pub fn new<T: Transcript>(
        params: FusedGtExpParams,
        constraint_types: &[ConstraintType],
        locator_by_constraint: &[ConstraintLocator],
        witnesses: &[GtExpWitness<Fq>],
        g_poly_11var: DensePolynomial<Fq>,
        transcript: &mut T,
    ) -> Self {
        let num_rounds = params.num_rounds();
        let row_size = 1usize << params.num_x_vars;

        // Sample eq_point for fused (c_gt, x11) domain (same convention as fused G1Add).
        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let eq_poly = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&eq_point));

        // Build GT-local constraint list in global order, and fill packed rows for GtExp only.
        let gt_globals = gt_constraint_indices(constraint_types);
        debug_assert_eq!(gt_globals.len(), params.num_gt_constraints);

        let build_from_witness = |get: fn(&GtExpWitness<Fq>) -> &Vec<Fq>| {
            let mut xc = vec![Fq::zero(); params.num_gt_constraints_padded * row_size];
            for (c_gt, &global_idx) in gt_globals.iter().enumerate() {
                if let ConstraintLocator::GtExp { local } = locator_by_constraint[global_idx] {
                    let src = get(&witnesses[local]);
                    debug_assert_eq!(src.len(), row_size);
                    let off = c_gt * row_size;
                    xc[off..off + row_size].copy_from_slice(src);
                }
            }
            let cx = transpose_xc_to_cx(&xc, params.num_gt_constraints_padded, row_size);
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(cx))
        };

        let rho = build_from_witness(|w| &w.rho_packed);
        let rho_next = build_from_witness(|w| &w.rho_next_packed);
        let quotient = build_from_witness(|w| &w.quotient_packed);
        let digit_lo = build_from_witness(|w| &w.digit_lo_packed);
        let digit_hi = build_from_witness(|w| &w.digit_hi_packed);
        let base = build_from_witness(|w| &w.base_packed);
        let base2 = build_from_witness(|w| &w.base2_packed);
        let base3 = build_from_witness(|w| &w.base3_packed);

        // Pad g(x11) across c_gt.
        let g_11 = g_poly_11var.Z;
        debug_assert_eq!(g_11.len(), row_size);
        let mut g_xc = vec![Fq::zero(); params.num_gt_constraints_padded * row_size];
        for c in 0..params.num_gt_constraints_padded {
            let off = c * row_size;
            g_xc[off..off + row_size].copy_from_slice(&g_11);
        }
        let g_cx = transpose_xc_to_cx(&g_xc, params.num_gt_constraints_padded, row_size);
        let g = MultilinearPolynomial::LargeScalars(DensePolynomial::new(g_cx));

        Self {
            params,
            eq_poly,
            rho,
            rho_next,
            quotient,
            digit_lo,
            digit_hi,
            base,
            base2,
            base3,
            g,
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for FusedGtExpProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        let half = self.rho.len() / 2;
        if half == 0 {
            return UniPoly::from_evals_and_hint(previous_claim, &[Fq::zero(); DEGREE]);
        }

        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                // Use sumcheck_evals_array like FusedG1Add does
                let eq = self
                    .eq_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let rho = self
                    .rho
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let rho_next = self
                    .rho_next
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let quotient = self
                    .quotient
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let digit_lo = self
                    .digit_lo
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let digit_hi = self
                    .digit_hi
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let base = self
                    .base
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let base2 = self
                    .base2
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let base3 = self
                    .base3
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let g = self
                    .g
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                let mut out = [Fq::zero(); DEGREE];
                for t in 0..DEGREE {
                    let u = digit_lo[t];
                    let v = digit_hi[t];
                    let w0 = (Fq::one() - u) * (Fq::one() - v);
                    let w1 = u * (Fq::one() - v);
                    let w2 = (Fq::one() - u) * v;
                    let w3 = u * v;
                    let base_power = w0 + w1 * base[t] + w2 * base2[t] + w3 * base3[t];

                    let rho2 = rho[t] * rho[t];
                    let rho4 = rho2 * rho2;

                    let constraint = rho_next[t] - rho4 * base_power - quotient[t] * g[t];
                    out[t] = eq[t] * constraint;
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
        self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.rho.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.rho_next.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.quotient.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.digit_lo.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.digit_hi.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.base.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.base2.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.base3.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.g.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(sumcheck_challenges.to_vec());

        let rho_val = self.rho.get_bound_coeff(0);
        let rho_next_val = self.rho_next.get_bound_coeff(0);
        let quotient_val = self.quotient.get_bound_coeff(0);

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::gt_exp_rho_fused(),
            SumcheckId::GtExp,
            opening_point.clone(),
            rho_val,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::gt_exp_rho_next_fused(),
            SumcheckId::GtExp,
            opening_point.clone(),
            rho_next_val,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::gt_exp_quotient_fused(),
            SumcheckId::GtExp,
            opening_point,
            quotient_val,
        );
    }
}

pub struct FusedGtExpVerifier {
    params: FusedGtExpParams,
    eq_point: Vec<<Fq as JoltField>::Challenge>,
    /// Map GTExp local witness index -> GT-local c index.
    gtexp_c_indices: Vec<usize>,
    public_inputs: Vec<GtExpPublicInputs>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for FusedGtExpVerifier {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl FusedGtExpVerifier {
    pub fn new<T: Transcript>(
        params: FusedGtExpParams,
        constraint_types: &[ConstraintType],
        public_inputs: Vec<GtExpPublicInputs>,
        transcript: &mut T,
    ) -> Self {
        // Sample the eq point for the fused domain (must match prover).
        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..params.num_rounds())
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();

        // Build GTExp -> c_gt mapping by scanning global order and counting GT constraints.
        let mut gtexp_c_indices = Vec::new();
        let mut c_gt = 0usize;
        for ct in constraint_types {
            if matches!(ct, ConstraintType::GtExp | ConstraintType::GtMul) {
                if matches!(ct, ConstraintType::GtExp) {
                    gtexp_c_indices.push(c_gt);
                }
                c_gt += 1;
            }
        }

        debug_assert_eq!(
            gtexp_c_indices.len(),
            public_inputs.len(),
            "GtExp public_inputs length must match #GtExp constraints"
        );

        Self {
            params,
            eq_point,
            gtexp_c_indices,
            public_inputs,
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for FusedGtExpVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        debug_assert_eq!(sumcheck_challenges.len(), self.params.num_rounds());

        // Eq polynomial convention (same as fused G1Add): reverse challenges to match the
        // big-endian ordering used by `EqPolynomial::evals`.
        let eval_point: Vec<Fq> = sumcheck_challenges
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();
        let eq_point_f: Vec<Fq> = self.eq_point.iter().map(|c| (*c).into()).collect();
        let eq_eval = EqPolynomial::mle(&eq_point_f, &eval_point);

        // Parse (c,s,x) portions from the sumcheck point.
        let k = self.params.num_c_vars;
        let s0 = k;
        let x0 = k + self.params.num_step_vars;
        let r_c_lsb: Vec<Fq> = sumcheck_challenges[..k]
            .iter()
            .map(|c| (*c).into())
            .collect();

        // For digit/base evaluation helpers, match the existing packed GT exp verifier convention:
        // reverse within each chunk (big-endian).
        let r_s_star: Vec<Fq> = sumcheck_challenges[s0..x0]
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();
        let r_x_star: Vec<Fq> = sumcheck_challenges[x0..]
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();

        // Fetch fused opened claims at the sumcheck point.
        let (_, rho) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_exp_rho_fused(),
            SumcheckId::GtExp,
        );
        let (_, rho_next) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_exp_rho_next_fused(),
            SumcheckId::GtExp,
        );
        let (_, quotient) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::gt_exp_quotient_fused(),
            SumcheckId::GtExp,
        );

        // Compute g(r_x_star) using the public 4-var g MLE.
        let g_eval: Fq = {
            let g_mle_4var = <Bn254Recursion as RecursionCurve>::g_mle();
            let g_poly =
                MultilinearPolynomial::<Fq>::LargeScalars(DensePolynomial::new(g_mle_4var));
            g_poly.evaluate_dot_product::<Fq>(&r_x_star)
        };

        // Compute mixed digit/base values at this c-point.
        let eq_evals_s = EqPolynomial::<Fq>::evals(&r_s_star);
        let eq_evals_x = EqPolynomial::<Fq>::evals(&r_x_star);

        let mut digit_lo = Fq::zero();
        let mut digit_hi = Fq::zero();
        let mut base = Fq::zero();
        let mut base2 = Fq::zero();
        let mut base3 = Fq::zero();

        for (w, &c_idx) in self.gtexp_c_indices.iter().enumerate() {
            // Eq(r_c, c_idx) using little-endian bits for indices (round order is LSB-first).
            let bits = crate::zkvm::recursion::constraints::system::index_to_binary::<Fq>(c_idx, k);
            let w_c = EqPolynomial::mle(&r_c_lsb, &bits);

            let (u, v) = self.public_inputs[w].evaluate_digit_mles(&eq_evals_s);
            let (b1, b2, b3v) = self.public_inputs[w].evaluate_base_powers_mle(&eq_evals_x);

            digit_lo += w_c * u;
            digit_hi += w_c * v;
            base += w_c * b1;
            base2 += w_c * b2;
            base3 += w_c * b3v;
        }

        let u = digit_lo;
        let v = digit_hi;
        let w0 = (Fq::one() - u) * (Fq::one() - v);
        let w1 = u * (Fq::one() - v);
        let w2 = (Fq::one() - u) * v;
        let w3 = u * v;
        let base_power = w0 + w1 * base + w2 * base2 + w3 * base3;

        let rho2 = rho * rho;
        let rho4 = rho2 * rho2;
        let constraint_eval = rho_next - rho4 * base_power - quotient * g_eval;

        eq_eval * constraint_eval
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(sumcheck_challenges.to_vec());
        for vp in [
            VirtualPolynomial::gt_exp_rho_fused(),
            VirtualPolynomial::gt_exp_rho_next_fused(),
            VirtualPolynomial::gt_exp_quotient_fused(),
        ] {
            accumulator.append_virtual(transcript, vp, SumcheckId::GtExp, opening_point.clone());
        }
    }
}
