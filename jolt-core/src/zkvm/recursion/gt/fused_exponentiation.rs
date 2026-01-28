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
    zkvm::recursion::constraints::system::{index_to_binary, ConstraintLocator, ConstraintType},
    zkvm::recursion::curve::{Bn254Recursion, RecursionCurve},
    zkvm::recursion::gt::indexing::{k_exp, k_gt, num_gt_constraints_padded},
    zkvm::recursion::gt::types::{GtExpPublicInputs, GtExpWitness},
    zkvm::witness::VirtualPolynomial,
};

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use rayon::prelude::*;

/// Degree bound matches existing packed GT exp (see `gt/exponentiation.rs`).
/// This bound is used by the batching sumcheck interface and by UniPoly compression.
const DEGREE: usize = 8; // Was 7, but eq*C has degree 1+7=8

#[derive(Clone, Allocative)]
pub struct FusedGtExpParams {
    /// Number of c-index variables for Stage 1 (k_gt).
    ///
    /// Stage 1 uses the GT-local c domain so that Stage-2 subprotocols (fused shift + wiring)
    /// can consume Stage-1 openings at a point whose `(s,u)` challenges align with Stage 2.
    pub num_c_vars: usize,
    /// Family-local GTExp suffix length (k_exp).
    pub k_exp: usize,
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
        let k_exp = k_exp(constraint_types);
        Self {
            num_c_vars,
            k_exp,
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

        let build_from_witness = |get: fn(&GtExpWitness<Fq>) -> &Vec<Fq>| {
            let mut xc = vec![Fq::zero(); params.num_gt_constraints_padded * row_size];
            let dummy = params.num_c_vars.saturating_sub(params.k_exp);
            for global_idx in 0..constraint_types.len() {
                if let ConstraintLocator::GtExp { local } = locator_by_constraint[global_idx] {
                    let src = get(&witnesses[local]);
                    debug_assert_eq!(src.len(), row_size);
                    // Split-k convention: the GTExp family index lives in the **high** bits of c_gt,
                    // and the low `dummy` bits are replicated.
                    for d in 0..(1usize << dummy) {
                        let c = d + (local << dummy);
                        let off = c * row_size;
                        xc[off..off + row_size].copy_from_slice(src);
                    }
                }
            }
            // Store in [x11 low bits, c_gt high bits] order so `c_gt` is a suffix in Stage 2.
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(xc))
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
        let g = MultilinearPolynomial::LargeScalars(DensePolynomial::new(g_xc));

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

        // Build GTExp -> c index mapping (family-local: 0..num_gt_exp in global order).
        let mut gtexp_c_indices = Vec::new();
        let mut c_exp = 0usize;
        for ct in constraint_types {
            if matches!(ct, ConstraintType::GtExp) {
                gtexp_c_indices.push(c_exp);
                c_exp += 1;
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

        // Parse (s,u,c) portions from the sumcheck point.
        //
        // Variable order (LSB-first rounds) is:
        // - step bits s (7), then elem bits u (4), then c_gt bits (k) as a suffix.
        let k_gt = self.params.num_c_vars;
        let k_exp = self.params.k_exp;
        let dummy = k_gt.saturating_sub(k_exp);
        let s0 = 0usize;
        let x0 = self.params.num_step_vars; // 7
                                            // Split-k convention: dummy bits are the first `dummy` *low* bits of the c suffix.
                                            // The GTExp family index lives in the remaining `k_exp` high bits.
        let r_c_tail_lsb: Vec<Fq> = sumcheck_challenges[CONFIG.packed_vars + dummy..]
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
        let r_x_star: Vec<Fq> = sumcheck_challenges[x0..CONFIG.packed_vars]
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
            let bits = index_to_binary::<Fq>(c_idx, k_exp);
            let w_c = EqPolynomial::mle(&r_c_tail_lsb, &bits);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::dense_mlpoly::DensePolynomial;
    use crate::transcripts::Blake2bTranscript;
    use crate::zkvm::recursion::constraints::system::ConstraintLocator;

    #[test]
    fn stage1_fused_gtexp_replicates_across_dummy_c_bits() {
        // Split-k scenario:
        // - 3 GTExp => padded 4 => k_exp = 2
        // - 16 GTMul => padded 16 => k_mul = 4
        // => k_gt = 4, dummy = 2
        let mut constraint_types = Vec::new();
        constraint_types.extend(std::iter::repeat_n(ConstraintType::GtExp, 3));
        constraint_types.extend(std::iter::repeat_n(ConstraintType::GtMul, 16));

        let params = FusedGtExpParams::from_constraint_types(&constraint_types);
        assert_eq!(params.num_c_vars, k_gt(&constraint_types));
        assert_eq!(params.k_exp, k_exp(&constraint_types));
        let dummy = params.num_c_vars - params.k_exp;
        assert_eq!(dummy, 2);

        // Build locator_by_constraint with family-local ranks.
        let mut locator_by_constraint = Vec::with_capacity(constraint_types.len());
        let mut exp_rank = 0usize;
        let mut mul_rank = 0usize;
        for ct in &constraint_types {
            match ct {
                ConstraintType::GtExp => {
                    locator_by_constraint.push(ConstraintLocator::GtExp { local: exp_rank });
                    exp_rank += 1;
                }
                ConstraintType::GtMul => {
                    locator_by_constraint.push(ConstraintLocator::GtMul { local: mul_rank });
                    mul_rank += 1;
                }
                _ => unreachable!("test only uses GTExp/GTMul"),
            }
        }
        assert_eq!(exp_rank, 3);
        assert_eq!(mul_rank, 16);

        // Fake witnesses: each row is constant per instance so replication is easy to observe.
        let row_size = 1usize << CONFIG.packed_vars; // 2048
        let mk = |v: u64| vec![Fq::from_u64(v); row_size];
        let witnesses: Vec<GtExpWitness<Fq>> = (0..3)
            .map(|i| GtExpWitness::<Fq> {
                rho_packed: mk(10 + i as u64),
                rho_next_packed: mk(20 + i as u64),
                quotient_packed: mk(30 + i as u64),
                digit_lo_packed: mk(40 + i as u64),
                digit_hi_packed: mk(50 + i as u64),
                base_packed: mk(60 + i as u64),
                base2_packed: mk(70 + i as u64),
                base3_packed: mk(80 + i as u64),
                num_steps: 1,
            })
            .collect();

        let g_poly_11var = DensePolynomial::new(vec![Fq::one(); row_size]);
        let mut transcript = Blake2bTranscript::new(b"test_fused_gtexp_replication");
        let prover = FusedGtExpProver::new(
            params.clone(),
            &constraint_types,
            &locator_by_constraint,
            &witnesses,
            g_poly_11var,
            &mut transcript,
        );

        // Inspect the backing table for rho(c_gt, x11) in [x11_low, c_gt_high] layout.
        let MultilinearPolynomial::LargeScalars(rho_dense) = &prover.rho else {
            panic!("expected LargeScalars rho polynomial");
        };
        let z = &rho_dense.Z;

        // For each GTExp local index, rho must be replicated across dummy low bits.
        // Embed: c = d + (local << dummy), for d in [0..2^dummy).
        for local in 0..3usize {
            let c0 = local << dummy;
            let c1 = 1 + (local << dummy);
            let c2 = 2 + (local << dummy);
            let c3 = 3 + (local << dummy);
            for x in 0..row_size {
                let v0 = z[c0 * row_size + x];
                assert_eq!(v0, z[c1 * row_size + x]);
                assert_eq!(v0, z[c2 * row_size + x]);
                assert_eq!(v0, z[c3 * row_size + x]);
            }
        }

        // Padding GTExp local index = 3 (since padded to 4) should remain all-zero across its
        // replicated block.
        let pad_local = 3usize;
        for d in 0..(1usize << dummy) {
            let c = d + (pad_local << dummy);
            for x in 0..row_size {
                assert_eq!(z[c * row_size + x], Fq::zero());
            }
        }
    }
}
