//! Fused GT multiplication sumcheck (over GT-local constraint index + x variables).
//!
//! End-to-end GT fusion packs GT mul witness polynomials as fused MLEs over `(c_gt, x)`,
//! where `c_gt` ranges over only `{GtExp,GtMul}` constraints (in global order), and `x` is
//! the standard 11-var packed GT exp domain.
//!
//! Variable order for this sumcheck instance (round order, `BindingOrder::LowToHigh`):
//! - first `k_gt` rounds bind the GT-local constraint index `c_gt` (LSB first)
//! - last 11 rounds bind the packed x-variables `x` (LSB first)
//!
//! For GTMul, we embed the native 4-var Fq12 element witness into 11 vars by replicating
//! over the 7 step bits: `x = step (7 bits) || u (4 bits)` with `step` as low bits.

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
    zkvm::witness::VirtualPolynomial,
};

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use rayon::prelude::*;

/// Degree bound for GTMul constraint polynomial: eq * ind * (lhs*rhs - result - quotient*g)
/// Each term: eq (deg 1) * ind (deg 1) * constraint (deg 2) = deg 4.
const DEGREE: usize = 4;

/// Transpose a `[x_low, c_high]` table into `[c_low, x_high]`.
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
pub struct FusedGtMulParams {
    pub num_constraint_index_vars: usize, // k_gt
    pub num_constraint_vars: usize,       // 11
    pub num_gt_constraints: usize,
    pub num_gt_constraints_padded: usize,
}

impl FusedGtMulParams {
    pub fn new(num_gt_constraints: usize, num_gt_constraints_padded: usize) -> Self {
        debug_assert!(num_gt_constraints_padded.is_power_of_two());
        let num_constraint_index_vars = num_gt_constraints_padded.trailing_zeros() as usize;
        Self {
            num_constraint_index_vars,
            num_constraint_vars: 11,
            num_gt_constraints,
            num_gt_constraints_padded,
        }
    }

    #[inline]
    pub fn num_rounds(&self) -> usize {
        self.num_constraint_index_vars + self.num_constraint_vars
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
        OpeningPoint::<BIG_ENDIAN, Fq>::new(challenges.to_vec())
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
        gt_mul_rows: &[crate::zkvm::recursion::gt::multiplication::GtMulConstraintPolynomials<Fq>],
        g_poly_4var: &DensePolynomial<Fq>,
        transcript: &mut T,
    ) -> Self {
        let num_rounds = params.num_rounds();
        let row_size = 1usize << params.num_constraint_vars;

        // Sample eq_point for the fused (c_gt, x) domain.
        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let eq_poly = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&eq_point));

        // Build GT-local ordering and map each c_gt to an optional GTMul local index.
        let mut gt_kinds: Vec<ConstraintType> = Vec::new();
        let mut c_gt_to_gt_mul_local: Vec<Option<usize>> = Vec::new();
        let mut gt_mul_local = 0usize;
        for ct in constraint_types {
            if matches!(ct, ConstraintType::GtExp | ConstraintType::GtMul) {
                gt_kinds.push(ct.clone());
                if matches!(ct, ConstraintType::GtMul) {
                    c_gt_to_gt_mul_local.push(Some(gt_mul_local));
                    gt_mul_local += 1;
                } else {
                    c_gt_to_gt_mul_local.push(None);
                }
            }
        }
        debug_assert_eq!(gt_kinds.len(), params.num_gt_constraints);
        debug_assert_eq!(c_gt_to_gt_mul_local.len(), params.num_gt_constraints);
        debug_assert_eq!(
            gt_mul_local,
            gt_mul_rows.len(),
            "expected GTMul local index count to match gt_mul_rows"
        );

        // Indicator table in [x_low, c_high] layout, then transpose to [c_low, x_high].
        let mut ind_xc = vec![Fq::zero(); params.num_gt_constraints_padded * row_size];
        for c in 0..params.num_gt_constraints_padded {
            let is_gtmul = c < params.num_gt_constraints && matches!(gt_kinds[c], ConstraintType::GtMul);
            if is_gtmul {
                let off = c * row_size;
                for x in 0..row_size {
                    ind_xc[off + x] = Fq::one();
                }
            }
        }
        let ind_cx = transpose_xc_to_cx(&ind_xc, params.num_gt_constraints_padded, row_size);
        let indicator_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(ind_cx));

        // Pad g(u) to 11 vars by replicating over the 7 step bits.
        let g4 = &g_poly_4var.Z;
        debug_assert_eq!(g4.len(), 1usize << 4);
        let mut g_padded = vec![Fq::zero(); row_size];
        for x_idx in 0..row_size {
            let u_idx = x_idx >> 7;
            g_padded[x_idx] = g4[u_idx];
        }

        // Build a full (c_gt, x) g polynomial table (independent of c_gt).
        let mut g_xc = vec![Fq::zero(); params.num_gt_constraints_padded * row_size];
        for c in 0..params.num_gt_constraints_padded {
            let off = c * row_size;
            g_xc[off..off + row_size].copy_from_slice(&g_padded);
        }
        let g_cx = transpose_xc_to_cx(&g_xc, params.num_gt_constraints_padded, row_size);
        let g_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(g_cx));

        // Helper to build a fused GTMul term table (padded to 11 vars).
        let build_term = |get_term4: fn(&crate::zkvm::recursion::gt::multiplication::GtMulConstraintPolynomials<Fq>) -> &Vec<Fq>| {
            let mut term_xc = vec![Fq::zero(); params.num_gt_constraints_padded * row_size];
            for c in 0..params.num_gt_constraints_padded {
                let Some(local) = (c < params.num_gt_constraints)
                    .then(|| c_gt_to_gt_mul_local[c])
                    .flatten()
                else {
                    continue;
                };
                let src4 = get_term4(&gt_mul_rows[local]);
                debug_assert_eq!(src4.len(), 1usize << 4);
                let off = c * row_size;
                for x_idx in 0..row_size {
                    let u_idx = x_idx >> 7;
                    term_xc[off + x_idx] = src4[u_idx];
                }
            }
            let term_cx = transpose_xc_to_cx(&term_xc, params.num_gt_constraints_padded, row_size);
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(term_cx))
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

    fn compute_message(&mut self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        let num_remaining = self.eq_poly.get_num_vars();
        debug_assert!(num_remaining > 0, "fused gtmul should have at least one round");
        let half = 1usize << (num_remaining - 1);

        let total_evals: [Fq; DEGREE] = (0..half)
            .into_par_iter()
            .map(|idx| {
                let eq_evals = self.eq_poly.sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let ind_evals =
                    self.indicator_poly
                        .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let lhs_e = self.lhs.sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let rhs_e = self.rhs.sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let res_e =
                    self.result
                        .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let quo_e = self
                    .quotient
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let g_e = self.g_poly.sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);

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

        // Build GT-local ordering and collect GTMul indices in that order.
        let mut gtmul_c_indices = Vec::new();
        let mut c_gt = 0usize;
        for ct in constraint_types {
            if matches!(ct, ConstraintType::GtExp | ConstraintType::GtMul) {
                if matches!(ct, ConstraintType::GtMul) {
                    gtmul_c_indices.push(c_gt);
                }
                c_gt += 1;
            }
        }
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
        // Reverse challenges to form the evaluation point used by EqPolynomial::mle.
        let eval_point: Vec<Fq> = sumcheck_challenges.iter().rev().map(|c| (*c).into()).collect();
        let eq_point_f: Vec<Fq> = self.eq_point.iter().map(|c| (*c).into()).collect();
        let eq_eval = EqPolynomial::mle(&eq_point_f, &eval_point);

        // Indicator I_gtmul(r_c) as Σ_{c in gtmul_c_indices} Eq(r_c, c).
        let k = self.params.num_constraint_index_vars;
        let r_c: Vec<Fq> = sumcheck_challenges.iter().take(k).map(|c| (*c).into()).collect();
        let mut ind_eval = Fq::zero();
        for &c in &self.gtmul_c_indices {
            let bits = index_to_binary::<Fq>(c, k);
            ind_eval += EqPolynomial::mle(&r_c, &bits);
        }

        // Extract u (last 4 x-bits) and evaluate g(u).
        let x_start = k;
        let r_x: Vec<Fq> = sumcheck_challenges[x_start..].iter().map(|c| (*c).into()).collect();
        debug_assert_eq!(r_x.len(), 11);
        let r_u = &r_x[7..11];
        let g_eval = self.eval_g_at_u(r_u);

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

