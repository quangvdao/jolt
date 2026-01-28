//! Fused G1 scalar multiplication sumchecks (family-local, Option B).
//!
//! This implements a fused variant of the existing per-instance `G1ScalarMul` constraint list:
//! - We fuse over a **family-local** constraint index `c_g1smul`.
//! - The native step trace domain remains 8 variables (256 steps).
//! - All committed witness polynomials are treated as MLEs over `(step, c_g1smul)` with
//!   `step` in the low bits and `c` as a suffix (LSB-first binding order).
//! - Padding rows are gated by a public indicator `I_g1smul(c)`.
//!
//! This is the intended Option-B fusion style for scalar multiplication, analogous to fused GT.

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
    zkvm::recursion::constraints::system::{index_to_binary, G1ScalarMulNative},
    zkvm::recursion::g1::types::G1ScalarMulPublicInputs,
    zkvm::recursion::gt::types::{
        eq_lsb_evals, eq_lsb_mle, eq_plus_one_lsb_evals, eq_plus_one_lsb_mle,
    },
    zkvm::witness::VirtualPolynomial,
};

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use rayon::prelude::*;

const STEP_VARS: usize = 8; // 256 steps
                            // Degree bound:
                            // - constraint_value has max degree 5 (matches existing per-instance analysis),
                            // - we multiply by Eq(step,c) (deg 1) and I(c) (deg 1),
                            // so total per-variable degree is 7.
const DEGREE: usize = 7;

#[inline]
fn k_from_num_constraints(num_constraints: usize) -> usize {
    num_constraints.max(1).next_power_of_two().trailing_zeros() as usize
}

#[inline]
fn row_size() -> usize {
    1usize << STEP_VARS
}

#[inline]
fn build_bit_poly_8(public_input: &G1ScalarMulPublicInputs) -> Vec<Fq> {
    let bits = public_input.bits_msb();
    debug_assert_eq!(bits.len(), 256);
    let mut evals = vec![Fq::zero(); 1usize << STEP_VARS];
    for i in 0..256 {
        evals[i] = if bits[i] { Fq::one() } else { Fq::zero() };
    }
    evals
}

#[derive(Clone, Allocative)]
pub struct FusedG1ScalarMulParams {
    /// Step variable count (always 8 for the 256-step scalar-mul trace).
    pub num_step_vars: usize,
    /// Family-local constraint-index var count (`k_smul`).
    pub k_smul: usize,
    /// Common suffix length used for Stage-2 alignment (`k_common >= k_smul`).
    ///
    /// In fully-fused wiring mode, G1 wiring uses `k_g1 = max(k_smul, k_add)`. We keep the
    /// committed polynomials family-local (k_smul), but allow the sumcheck to run over k_common
    /// by replicating across dummy low bits and caching openings at the family-local point.
    pub k_common: usize,
    pub num_constraints: usize,
    /// Padded family-local constraint count (power of two, min 1): `2^{k_smul}`.
    pub num_constraints_padded: usize,
    /// Padded common constraint count (power of two, min 1): `2^{k_common}`.
    pub num_constraints_padded_common: usize,
}

impl FusedG1ScalarMulParams {
    pub fn new(num_constraints: usize) -> Self {
        let num_constraints_padded = num_constraints.max(1).next_power_of_two();
        let k_smul = num_constraints_padded.trailing_zeros() as usize;
        Self::new_with_k_common(num_constraints, k_smul)
    }

    pub fn new_with_k_common(num_constraints: usize, k_common: usize) -> Self {
        let num_constraints_padded = num_constraints.max(1).next_power_of_two();
        let k_smul = num_constraints_padded.trailing_zeros() as usize;
        let k_common = core::cmp::max(k_common, k_smul);
        let num_constraints_padded_common = 1usize << k_common;
        Self {
            num_step_vars: STEP_VARS,
            k_smul,
            k_common,
            num_constraints,
            num_constraints_padded,
            num_constraints_padded_common,
        }
    }

    #[inline]
    pub fn num_rounds(&self) -> usize {
        self.num_step_vars + self.k_common
    }

    #[inline]
    pub fn num_opening_vars(&self) -> usize {
        // Family-local opening point length.
        self.num_step_vars + self.k_smul
    }

    #[inline]
    pub fn dummy_bits(&self) -> usize {
        self.k_common.saturating_sub(self.k_smul)
    }
}

impl SumcheckInstanceParams<Fq> for FusedG1ScalarMulParams {
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
        // Stage-2 alignment: this sumcheck can run over `k_common` c-bits, but committed witness
        // polynomials are family-local over `k_smul` bits.
        //
        // Dummy-bit convention: dummy bits are the *low* bits of the common `c` segment, so the
        // family-local bits are the *suffix* of the common `c`.
        let expected = self.num_rounds();
        assert_eq!(
            challenges.len(),
            expected,
            "expected {} sumcheck challenges, got {}",
            expected,
            challenges.len()
        );
        let dummy = self.dummy_bits();
        let step_end = self.num_step_vars;
        let c_common_start = step_end;
        let c_tail_start = c_common_start + dummy;
        let mut opening: Vec<<Fq as JoltField>::Challenge> =
            Vec::with_capacity(self.num_opening_vars());
        opening.extend_from_slice(&challenges[..step_end]);
        opening.extend_from_slice(&challenges[c_tail_start..]);
        debug_assert_eq!(opening.len(), self.num_opening_vars());
        OpeningPoint::<BIG_ENDIAN, Fq>::new(opening)
    }
}

#[derive(Clone, Copy, Debug)]
struct FusedG1ScalarMulValues {
    x_a: Fq,
    y_a: Fq,
    x_t: Fq,
    y_t: Fq,
    x_a_next: Fq,
    y_a_next: Fq,
    t_indicator: Fq,
    a_indicator: Fq,
}

impl FusedG1ScalarMulValues {
    #[inline]
    fn from_claims(claims: &[Fq]) -> Self {
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

    #[inline]
    fn from_poly_evals<const D: usize>(poly_evals: &[[Fq; D]], idx: usize) -> Self {
        Self {
            x_a: poly_evals[0][idx],
            y_a: poly_evals[1][idx],
            x_t: poly_evals[2][idx],
            y_t: poly_evals[3][idx],
            x_a_next: poly_evals[4][idx],
            y_a_next: poly_evals[5][idx],
            t_indicator: poly_evals[6][idx],
            a_indicator: poly_evals[7][idx],
        }
    }

    /// Evaluate the batched scalar-mul constraint polynomial at a point.
    ///
    /// Mirrors `g1/scalar_multiplication.rs` (per-instance) constraint semantics.
    fn eval_constraint(&self, bit: Fq, x_p: Fq, y_p: Fq, delta: Fq) -> Fq {
        let one = Fq::one();
        let two = Fq::from(2u64);
        let three = Fq::from(3u64);
        let four = Fq::from(4u64);
        let nine = Fq::from(9u64);

        // C1: 4y_A²(x_T + 2x_A) - 9x_A⁴
        let y_a_sq = self.y_a * self.y_a;
        let x_a_sq = self.x_a * self.x_a;
        let c1 = four * y_a_sq * (self.x_t + two * self.x_a) - nine * x_a_sq * x_a_sq;

        // C2: 3x_A²(x_T - x_A) + 2y_A(y_T + y_A)
        let c2 = three * x_a_sq * (self.x_t - self.x_a) + two * self.y_a * (self.y_t + self.y_a);

        // C3: Conditional addition x-coord
        let c3_skip = (one - bit) * (self.x_a_next - self.x_t);
        let c3_infinity = bit * self.t_indicator * (self.x_a_next - x_p);
        let x_diff = x_p - self.x_t;
        let y_diff = y_p - self.y_t;
        let chord_x = (self.x_a_next + self.x_t + x_p) * x_diff * x_diff - y_diff * y_diff;
        let c3_add = bit * (one - self.t_indicator) * chord_x;
        let c3 = c3_skip + c3_infinity + c3_add;

        // C4: Conditional addition y-coord
        let c4_skip = (one - bit) * (self.y_a_next - self.y_t);
        let c4_infinity = bit * self.t_indicator * (self.y_a_next - y_p);
        let chord_y = (self.y_a_next + self.y_t) * x_diff - y_diff * (self.x_t - self.x_a_next);
        let c4_add = bit * (one - self.t_indicator) * chord_y;
        let c4 = c4_skip + c4_infinity + c4_add;

        // C5: ind_A * (1 - ind_T)
        let c5 = self.a_indicator * (one - self.t_indicator);

        // C6: ind_T * x_T, ind_T * y_T
        let c6_x = self.t_indicator * self.x_t;
        let c6_y = self.t_indicator * self.y_t;

        // Batch with powers of delta
        let d2 = delta * delta;
        let d3 = d2 * delta;
        let d4 = d3 * delta;
        let d5 = d4 * delta;
        let d6 = d5 * delta;

        c1 + delta * c2 + d2 * c3 + d3 * c4 + d4 * c5 + d5 * c6_x + d6 * c6_y
    }
}

#[derive(Allocative)]
pub struct FusedG1ScalarMulProver {
    params: FusedG1ScalarMulParams,
    eq_poly: MultilinearPolynomial<Fq>,
    indicator_poly: MultilinearPolynomial<Fq>,
    // committed witness polys over (step,c)
    x_a: MultilinearPolynomial<Fq>,
    y_a: MultilinearPolynomial<Fq>,
    x_t: MultilinearPolynomial<Fq>,
    y_t: MultilinearPolynomial<Fq>,
    x_a_next: MultilinearPolynomial<Fq>,
    y_a_next: MultilinearPolynomial<Fq>,
    t_indicator: MultilinearPolynomial<Fq>,
    a_indicator: MultilinearPolynomial<Fq>,
    // public polys over (step,c)
    bit: MultilinearPolynomial<Fq>,
    x_p: MultilinearPolynomial<Fq>,
    y_p: MultilinearPolynomial<Fq>,
    term_batch_coeff: Fq,
}

impl FusedG1ScalarMulProver {
    pub fn new<T: Transcript>(
        rows: &[G1ScalarMulNative],
        public_inputs: &[G1ScalarMulPublicInputs],
        transcript: &mut T,
    ) -> Self {
        debug_assert_eq!(
            rows.len(),
            public_inputs.len(),
            "G1ScalarMul rows must match public_inputs length"
        );
        let params = FusedG1ScalarMulParams::new(rows.len());
        Self::new_with_params(rows, public_inputs, params, transcript)
    }

    pub fn new_with_k_common<T: Transcript>(
        rows: &[G1ScalarMulNative],
        public_inputs: &[G1ScalarMulPublicInputs],
        k_common: usize,
        transcript: &mut T,
    ) -> Self {
        debug_assert_eq!(rows.len(), public_inputs.len());
        let params = FusedG1ScalarMulParams::new_with_k_common(rows.len(), k_common);
        Self::new_with_params(rows, public_inputs, params, transcript)
    }

    fn new_with_params<T: Transcript>(
        rows: &[G1ScalarMulNative],
        public_inputs: &[G1ScalarMulPublicInputs],
        params: FusedG1ScalarMulParams,
        transcript: &mut T,
    ) -> Self {
        let num_rounds = params.num_rounds();

        // Sample eq point for the fused (step,c) domain.
        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let eq_poly = MultilinearPolynomial::from(EqPolynomial::<Fq>::evals(&eq_point));

        // Sample δ for term batching (matches per-instance protocol behavior).
        let term_batch_coeff: Fq = transcript.challenge_scalar_optimized::<Fq>().into();

        let rs = row_size();
        let blocks = params.num_constraints_padded_common;
        let total_len = blocks * rs;

        let mut ind_sc = vec![Fq::zero(); total_len];
        let dummy = params.dummy_bits();
        for c_common in 0..blocks {
            let c_smul = c_common >> dummy;
            if c_smul < params.num_constraints {
                let off = c_common * rs;
                for s in 0..rs {
                    ind_sc[off + s] = Fq::one();
                }
            }
        }
        let indicator_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(ind_sc));

        let build_term = |get: fn(&G1ScalarMulNative) -> &Vec<Fq>| -> MultilinearPolynomial<Fq> {
            let mut v = vec![Fq::zero(); total_len];
            for c_smul in 0..params.num_constraints {
                let src = get(&rows[c_smul]);
                debug_assert_eq!(src.len(), rs);
                for d in 0..(1usize << dummy) {
                    let c_common = d + (c_smul << dummy);
                    let off = c_common * rs;
                    v[off..off + rs].copy_from_slice(src);
                }
            }
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(v))
        };

        let x_a = build_term(|r| &r.x_a);
        let y_a = build_term(|r| &r.y_a);
        let x_t = build_term(|r| &r.x_t);
        let y_t = build_term(|r| &r.y_t);
        let x_a_next = build_term(|r| &r.x_a_next);
        let y_a_next = build_term(|r| &r.y_a_next);
        let t_indicator = build_term(|r| &r.t_indicator);
        let a_indicator = build_term(|r| &r.a_indicator);

        // Public bit polynomial per instance: 8-var bit table, fused over c.
        let mut bit_sc = vec![Fq::zero(); total_len];
        for c_smul in 0..params.num_constraints {
            let src = build_bit_poly_8(&public_inputs[c_smul]);
            debug_assert_eq!(src.len(), rs);
            for d in 0..(1usize << dummy) {
                let c_common = d + (c_smul << dummy);
                let off = c_common * rs;
                bit_sc[off..off + rs].copy_from_slice(&src);
            }
        }
        let bit = MultilinearPolynomial::LargeScalars(DensePolynomial::new(bit_sc));

        // Public base point polynomials (constant over step, fused over c).
        let mut x_p_sc = vec![Fq::zero(); total_len];
        let mut y_p_sc = vec![Fq::zero(); total_len];
        for c_smul in 0..params.num_constraints {
            let (x_p, y_p) = rows[c_smul].base_point;
            for d in 0..(1usize << dummy) {
                let c_common = d + (c_smul << dummy);
                let off = c_common * rs;
                for s in 0..rs {
                    x_p_sc[off + s] = x_p;
                    y_p_sc[off + s] = y_p;
                }
            }
        }
        let x_p = MultilinearPolynomial::LargeScalars(DensePolynomial::new(x_p_sc));
        let y_p = MultilinearPolynomial::LargeScalars(DensePolynomial::new(y_p_sc));

        Self {
            params,
            eq_poly,
            indicator_poly,
            x_a,
            y_a,
            x_t,
            y_t,
            x_a_next,
            y_a_next,
            t_indicator,
            a_indicator,
            bit,
            x_p,
            y_p,
            term_batch_coeff,
        }
    }
}

impl<FqT: Transcript> SumcheckInstanceProver<Fq, FqT> for FusedG1ScalarMulProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        let num_remaining = self.eq_poly.get_num_vars();
        debug_assert!(
            num_remaining > 0,
            "fused g1 scalar mul should have at least one round"
        );
        let half = 1usize << (num_remaining - 1);

        let delta = self.term_batch_coeff;

        let total_evals: [Fq; DEGREE] = (0..half)
            .into_par_iter()
            .map(|idx| {
                let eq_e = self
                    .eq_poly
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let ind_e = self
                    .indicator_poly
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);

                let polys = [
                    &self.x_a,
                    &self.y_a,
                    &self.x_t,
                    &self.y_t,
                    &self.x_a_next,
                    &self.y_a_next,
                    &self.t_indicator,
                    &self.a_indicator,
                ];
                let mut poly_evals = vec![[Fq::zero(); DEGREE]; polys.len()];
                for (t, p) in polys.iter().enumerate() {
                    poly_evals[t] = p.sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                }

                let bit_e = self
                    .bit
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let x_p_e = self
                    .x_p
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let y_p_e = self
                    .y_p
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);

                let mut out = [Fq::zero(); DEGREE];
                for t in 0..DEGREE {
                    let vals = FusedG1ScalarMulValues::from_poly_evals(&poly_evals, t);
                    let c_val = vals.eval_constraint(bit_e[t], x_p_e[t], y_p_e[t], delta);
                    out[t] = eq_e[t] * ind_e[t] * c_val;
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
        for p in [
            &mut self.eq_poly,
            &mut self.indicator_poly,
            &mut self.x_a,
            &mut self.y_a,
            &mut self.x_t,
            &mut self.y_t,
            &mut self.x_a_next,
            &mut self.y_a_next,
            &mut self.t_indicator,
            &mut self.a_indicator,
            &mut self.bit,
            &mut self.x_p,
            &mut self.y_p,
        ] {
            p.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<Fq>,
        transcript: &mut FqT,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);

        for (vp, claim) in [
            (
                VirtualPolynomial::g1_scalar_mul_xa_fused(),
                self.x_a.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g1_scalar_mul_ya_fused(),
                self.y_a.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g1_scalar_mul_xt_fused(),
                self.x_t.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g1_scalar_mul_yt_fused(),
                self.y_t.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g1_scalar_mul_xa_next_fused(),
                self.x_a_next.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g1_scalar_mul_ya_next_fused(),
                self.y_a_next.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g1_scalar_mul_t_indicator_fused(),
                self.t_indicator.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g1_scalar_mul_a_indicator_fused(),
                self.a_indicator.get_bound_coeff(0),
            ),
        ] {
            accumulator.append_virtual(
                transcript,
                vp,
                SumcheckId::G1ScalarMul,
                opening_point.clone(),
                claim,
            );
        }
    }
}

pub struct FusedG1ScalarMulVerifier {
    params: FusedG1ScalarMulParams,
    eq_point: Vec<<Fq as JoltField>::Challenge>,
    term_batch_coeff: Fq,
    num_constraints: usize,
    public_inputs: Vec<G1ScalarMulPublicInputs>,
    base_points: Vec<(Fq, Fq)>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for FusedG1ScalarMulVerifier {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl FusedG1ScalarMulVerifier {
    pub fn new<T: Transcript>(
        num_constraints: usize,
        public_inputs: Vec<G1ScalarMulPublicInputs>,
        base_points: Vec<(Fq, Fq)>,
        transcript: &mut T,
    ) -> Self {
        debug_assert_eq!(public_inputs.len(), num_constraints);
        debug_assert_eq!(base_points.len(), num_constraints);
        let params = FusedG1ScalarMulParams::new(num_constraints);
        Self::new_with_params(
            params,
            num_constraints,
            public_inputs,
            base_points,
            transcript,
        )
    }

    pub fn new_with_k_common<T: Transcript>(
        num_constraints: usize,
        k_common: usize,
        public_inputs: Vec<G1ScalarMulPublicInputs>,
        base_points: Vec<(Fq, Fq)>,
        transcript: &mut T,
    ) -> Self {
        debug_assert_eq!(public_inputs.len(), num_constraints);
        debug_assert_eq!(base_points.len(), num_constraints);
        let params = FusedG1ScalarMulParams::new_with_k_common(num_constraints, k_common);
        Self::new_with_params(
            params,
            num_constraints,
            public_inputs,
            base_points,
            transcript,
        )
    }

    fn new_with_params<T: Transcript>(
        params: FusedG1ScalarMulParams,
        num_constraints: usize,
        public_inputs: Vec<G1ScalarMulPublicInputs>,
        base_points: Vec<(Fq, Fq)>,
        transcript: &mut T,
    ) -> Self {
        let num_rounds = params.num_rounds();
        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let term_batch_coeff: Fq = transcript.challenge_scalar_optimized::<Fq>().into();
        Self {
            params,
            eq_point,
            term_batch_coeff,
            num_constraints,
            public_inputs,
            base_points,
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for FusedG1ScalarMulVerifier {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        debug_assert_eq!(sumcheck_challenges.len(), self.params.num_rounds());

        // Eq polynomial convention: reverse challenges to match big-endian ordering in EqPolynomial::evals.
        let eval_point: Vec<Fq> = sumcheck_challenges
            .iter()
            .rev()
            .map(|c| (*c).into())
            .collect();
        let eq_point_f: Vec<Fq> = self.eq_point.iter().map(|c| (*c).into()).collect();
        let eq_eval = EqPolynomial::mle(&eq_point_f, &eval_point);

        // Split step/c portions from the sumcheck point (round order is LSB-first).
        let r_step = &sumcheck_challenges[..STEP_VARS];
        let r_c_chal = &sumcheck_challenges[STEP_VARS..];
        debug_assert_eq!(r_c_chal.len(), self.params.k_common);
        let dummy = self.params.dummy_bits();
        let k = self.params.k_smul;
        let r_c_tail = &r_c_chal[dummy..];
        debug_assert_eq!(r_c_tail.len(), k);

        let r_c: Vec<Fq> = r_c_tail.iter().map(|c| (*c).into()).collect();

        // Indicator I(c): sum_{c < num_constraints} Eq(r_c, c).
        let mut ind_eval = Fq::zero();
        for c in 0..self.num_constraints {
            let bits = index_to_binary::<Fq>(c, k);
            ind_eval += EqPolynomial::mle(&r_c, &bits);
        }

        // Fused public inputs at this point:
        // - base points are c-only (replicated across step)
        // - bit is step-only per instance, then fused across c by Eq(r_c,c)
        let r_step_fq: Vec<Fq> = r_step.iter().map(|c| (*c).into()).collect();
        let mut x_p = Fq::zero();
        let mut y_p = Fq::zero();
        let mut bit = Fq::zero();
        for c in 0..self.num_constraints {
            let bits_c = index_to_binary::<Fq>(c, k);
            let w_c = EqPolynomial::mle(&r_c, &bits_c);
            let (xp_i, yp_i) = self.base_points[c];
            x_p += w_c * xp_i;
            y_p += w_c * yp_i;
            bit += w_c * self.public_inputs[c].evaluate_bit_mle(&r_step_fq);
        }

        // Fetch fused opened claims (8 committed witness polynomials).
        let mut claims = Vec::with_capacity(8);
        for vp in [
            VirtualPolynomial::g1_scalar_mul_xa_fused(),
            VirtualPolynomial::g1_scalar_mul_ya_fused(),
            VirtualPolynomial::g1_scalar_mul_xt_fused(),
            VirtualPolynomial::g1_scalar_mul_yt_fused(),
            VirtualPolynomial::g1_scalar_mul_xa_next_fused(),
            VirtualPolynomial::g1_scalar_mul_ya_next_fused(),
            VirtualPolynomial::g1_scalar_mul_t_indicator_fused(),
            VirtualPolynomial::g1_scalar_mul_a_indicator_fused(),
        ] {
            let (_, claim) =
                accumulator.get_virtual_polynomial_opening(vp, SumcheckId::G1ScalarMul);
            claims.push(claim);
        }

        let vals = FusedG1ScalarMulValues::from_claims(&claims);
        let constraint_value = vals.eval_constraint(bit, x_p, y_p, self.term_batch_coeff);

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
            VirtualPolynomial::g1_scalar_mul_xa_fused(),
            VirtualPolynomial::g1_scalar_mul_ya_fused(),
            VirtualPolynomial::g1_scalar_mul_xt_fused(),
            VirtualPolynomial::g1_scalar_mul_yt_fused(),
            VirtualPolynomial::g1_scalar_mul_xa_next_fused(),
            VirtualPolynomial::g1_scalar_mul_ya_next_fused(),
            VirtualPolynomial::g1_scalar_mul_t_indicator_fused(),
            VirtualPolynomial::g1_scalar_mul_a_indicator_fused(),
        ] {
            accumulator.append_virtual(
                transcript,
                vp,
                SumcheckId::G1ScalarMul,
                opening_point.clone(),
            );
        }
    }
}

#[derive(Clone, Debug, Allocative)]
pub struct FusedShiftG1ScalarMulParams {
    pub num_step_vars: usize, // 8
    pub k_smul: usize,
    pub k_common: usize,
}

impl FusedShiftG1ScalarMulParams {
    pub fn new(num_constraints: usize) -> Self {
        let k_smul = k_from_num_constraints(num_constraints);
        Self::new_with_k_common(num_constraints, k_smul)
    }

    pub fn new_with_k_common(num_constraints: usize, k_common: usize) -> Self {
        let k_smul = k_from_num_constraints(num_constraints);
        Self {
            num_step_vars: STEP_VARS,
            k_smul,
            k_common: core::cmp::max(k_common, k_smul),
        }
    }

    #[inline]
    pub fn num_rounds(&self) -> usize {
        self.num_step_vars + self.k_common
    }

    #[inline]
    pub fn num_opening_vars(&self) -> usize {
        self.num_step_vars + self.k_smul
    }

    #[inline]
    pub fn dummy_bits(&self) -> usize {
        self.k_common.saturating_sub(self.k_smul)
    }

    #[inline]
    pub fn normalize_opening_point(
        &self,
        challenges: &[<Fq as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, Fq> {
        let expected = self.num_rounds();
        assert_eq!(challenges.len(), expected);
        let dummy = self.dummy_bits();
        let step_end = self.num_step_vars;
        let c_tail_start = step_end + dummy;
        let mut opening: Vec<<Fq as JoltField>::Challenge> =
            Vec::with_capacity(self.num_opening_vars());
        opening.extend_from_slice(&challenges[..step_end]);
        opening.extend_from_slice(&challenges[c_tail_start..]);
        debug_assert_eq!(opening.len(), self.num_opening_vars());
        OpeningPoint::<BIG_ENDIAN, Fq>::new(opening)
    }
}

#[derive(Allocative)]
pub struct FusedShiftG1ScalarMulProver {
    params: FusedShiftG1ScalarMulParams,
    // full-domain weight polynomials over (step,c)
    eq_step_poly: MultilinearPolynomial<Fq>,
    eq_minus_one_step_poly: MultilinearPolynomial<Fq>,
    not_last_poly: MultilinearPolynomial<Fq>,
    eq_c_poly: MultilinearPolynomial<Fq>,
    // fused witness polys (step,c)
    x_a: MultilinearPolynomial<Fq>,
    x_a_next: MultilinearPolynomial<Fq>,
    y_a: MultilinearPolynomial<Fq>,
    y_a_next: MultilinearPolynomial<Fq>,
    gamma: Fq,
}

impl FusedShiftG1ScalarMulProver {
    pub fn new<T: Transcript>(rows: &[G1ScalarMulNative], transcript: &mut T) -> Self {
        let params = FusedShiftG1ScalarMulParams::new(rows.len());
        Self::new_with_params(rows, params, transcript)
    }

    pub fn new_with_k_common<T: Transcript>(
        rows: &[G1ScalarMulNative],
        k_common: usize,
        transcript: &mut T,
    ) -> Self {
        let params = FusedShiftG1ScalarMulParams::new_with_k_common(rows.len(), k_common);
        Self::new_with_params(rows, params, transcript)
    }

    fn new_with_params<T: Transcript>(
        rows: &[G1ScalarMulNative],
        params: FusedShiftG1ScalarMulParams,
        transcript: &mut T,
    ) -> Self {
        let k_common = params.k_common;
        let rs = row_size();
        let blocks = 1usize << k_common;
        let total_len = blocks * rs;

        // Sample reference points (step*, c*) and batching gamma.
        let step_ref: Vec<<Fq as JoltField>::Challenge> = (0..STEP_VARS)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let c_ref: Vec<<Fq as JoltField>::Challenge> = (0..k_common)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let gamma: Fq = transcript.challenge_scalar_optimized::<Fq>().into();

        // Build 8-var step weights.
        let eq_step_8 = eq_lsb_evals::<Fq>(&step_ref);
        let eq_minus_one_8 = eq_plus_one_lsb_evals::<Fq>(&step_ref);
        let mut not_last_8 = vec![Fq::one(); rs];
        not_last_8[rs - 1] = Fq::zero();

        // Build k_common-var Eq(c_ref, c) table.
        let eq_c_k = eq_lsb_evals::<Fq>(&c_ref);
        debug_assert_eq!(eq_c_k.len(), 1usize << k_common);

        // Embed weights into full (step,c) domain.
        let mut eq_step_sc = vec![Fq::zero(); total_len];
        let mut eqm1_step_sc = vec![Fq::zero(); total_len];
        let mut not_last_sc = vec![Fq::zero(); total_len];
        let mut eq_c_sc = vec![Fq::zero(); total_len];
        for c in 0..blocks {
            let off = c * rs;
            // step-only weights replicated across c
            eq_step_sc[off..off + rs].copy_from_slice(&eq_step_8);
            eqm1_step_sc[off..off + rs].copy_from_slice(&eq_minus_one_8);
            not_last_sc[off..off + rs].copy_from_slice(&not_last_8);
            // c-only weight replicated across step
            let eqc = eq_c_k[c];
            for s in 0..rs {
                eq_c_sc[off + s] = eqc;
            }
        }

        let eq_step_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(eq_step_sc));
        let eq_minus_one_step_poly =
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(eqm1_step_sc));
        let not_last_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(not_last_sc));
        let eq_c_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(eq_c_sc));

        let dummy = params.dummy_bits();
        let build_term = |get: fn(&G1ScalarMulNative) -> &Vec<Fq>| -> MultilinearPolynomial<Fq> {
            let mut v = vec![Fq::zero(); total_len];
            for c_smul in 0..rows.len() {
                let src = get(&rows[c_smul]);
                debug_assert_eq!(src.len(), rs);
                for d in 0..(1usize << dummy) {
                    let c_common = d + (c_smul << dummy);
                    let off = c_common * rs;
                    v[off..off + rs].copy_from_slice(src);
                }
            }
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(v))
        };

        let x_a = build_term(|r| &r.x_a);
        let x_a_next = build_term(|r| &r.x_a_next);
        let y_a = build_term(|r| &r.y_a);
        let y_a_next = build_term(|r| &r.y_a_next);

        Self {
            params,
            eq_step_poly,
            eq_minus_one_step_poly,
            not_last_poly,
            eq_c_poly,
            x_a,
            x_a_next,
            y_a,
            y_a_next,
            gamma,
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for FusedShiftG1ScalarMulProver {
    fn degree(&self) -> usize {
        // Eq(c)*Eq(step)*not_last*A_next - Eq(c)*EqMinusOne*A
        4
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _acc: &ProverOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn compute_message(&mut self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        const D: usize = 4;
        let half = self.x_a.len() / 2;
        if half == 0 {
            return UniPoly::from_evals_and_hint(previous_claim, &[Fq::zero(); D]);
        }
        let gamma = self.gamma;

        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                let eqs = self
                    .eq_step_poly
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);
                let eqm1 = self
                    .eq_minus_one_step_poly
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);
                let not_last = self
                    .not_last_poly
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);
                let eqc = self
                    .eq_c_poly
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);

                let xa = self
                    .x_a
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);
                let xan = self
                    .x_a_next
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);
                let ya = self
                    .y_a
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);
                let yan = self
                    .y_a_next
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);

                let mut out = [Fq::zero(); D];
                for t in 0..D {
                    let x_term = eqc[t] * (eqs[t] * not_last[t] * xan[t] - eqm1[t] * xa[t]);
                    let y_term = eqc[t] * (eqs[t] * not_last[t] * yan[t] - eqm1[t] * ya[t]);
                    out[t] = x_term + gamma * y_term;
                }
                out
            })
            .reduce(
                || [Fq::zero(); D],
                |mut acc, arr| {
                    for t in 0..D {
                        acc[t] += arr[t];
                    }
                    acc
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: <Fq as JoltField>::Challenge, _round: usize) {
        for p in [
            &mut self.eq_step_poly,
            &mut self.eq_minus_one_step_poly,
            &mut self.not_last_poly,
            &mut self.eq_c_poly,
            &mut self.x_a,
            &mut self.x_a_next,
            &mut self.y_a,
            &mut self.y_a_next,
        ] {
            p.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No-op: reuse openings cached by `FusedG1ScalarMul*` (same polynomials, same point).
    }
}

#[derive(Allocative)]
pub struct FusedShiftG1ScalarMulVerifier {
    params: FusedShiftG1ScalarMulParams,
    step_ref: Vec<<Fq as JoltField>::Challenge>,
    c_ref: Vec<<Fq as JoltField>::Challenge>,
    gamma: Fq,
}

impl FusedShiftG1ScalarMulVerifier {
    pub fn new<T: Transcript>(num_constraints: usize, transcript: &mut T) -> Self {
        let params = FusedShiftG1ScalarMulParams::new(num_constraints);
        Self::new_with_params(params, transcript)
    }

    pub fn new_with_k_common<T: Transcript>(
        num_constraints: usize,
        k_common: usize,
        transcript: &mut T,
    ) -> Self {
        let params = FusedShiftG1ScalarMulParams::new_with_k_common(num_constraints, k_common);
        Self::new_with_params(params, transcript)
    }

    fn new_with_params<T: Transcript>(
        params: FusedShiftG1ScalarMulParams,
        transcript: &mut T,
    ) -> Self {
        let step_ref: Vec<<Fq as JoltField>::Challenge> = (0..STEP_VARS)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let c_ref: Vec<<Fq as JoltField>::Challenge> = (0..params.k_common)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let gamma: Fq = transcript.challenge_scalar_optimized::<Fq>().into();
        Self {
            params,
            step_ref,
            c_ref,
            gamma,
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for FusedShiftG1ScalarMulVerifier {
    fn degree(&self) -> usize {
        4
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _acc: &VerifierOpeningAccumulator<Fq>) -> Fq {
        Fq::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<Fq>,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) -> Fq {
        debug_assert_eq!(sumcheck_challenges.len(), self.params.num_rounds());
        let y_step = &sumcheck_challenges[..STEP_VARS];
        let y_c = &sumcheck_challenges[STEP_VARS..];

        let eq = eq_lsb_mle::<Fq>(&self.step_ref, y_step);
        let eqm1 = eq_plus_one_lsb_mle::<Fq>(&self.step_ref, y_step);
        // not_last(y) = 1 - ∏ y_i  (last index is all-ones)
        let mut prod = Fq::one();
        for &y in y_step {
            let y_f: Fq = y.into();
            prod *= y_f;
        }
        let not_last = Fq::one() - prod;

        let eqc = eq_lsb_mle::<Fq>(&self.c_ref, y_c);

        let (_, xa) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::g1_scalar_mul_xa_fused(),
            SumcheckId::G1ScalarMul,
        );
        let (_, xan) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::g1_scalar_mul_xa_next_fused(),
            SumcheckId::G1ScalarMul,
        );
        let (_, ya) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::g1_scalar_mul_ya_fused(),
            SumcheckId::G1ScalarMul,
        );
        let (_, yan) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::g1_scalar_mul_ya_next_fused(),
            SumcheckId::G1ScalarMul,
        );

        let x_term = eqc * (eq * not_last * xan - eqm1 * xa);
        let y_term = eqc * (eq * not_last * yan - eqm1 * ya);
        x_term + self.gamma * y_term
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No-op: reuse openings cached by `FusedG1ScalarMul*` (same polynomials, same point).
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcripts::Blake2bTranscript;

    #[test]
    fn fused_g1_scalar_mul_params_has_expected_rounds() {
        let p = FusedG1ScalarMulParams::new(3);
        assert_eq!(p.num_step_vars, 8);
        assert_eq!(p.k_smul, 2); // padded to 4
        assert_eq!(p.k_common, 2);
        assert_eq!(p.num_rounds(), 10);
    }

    #[test]
    fn fused_shift_params_has_expected_rounds() {
        let p = FusedShiftG1ScalarMulParams::new(3);
        assert_eq!(p.num_rounds(), 10);
    }

    #[test]
    fn fused_g1_scalar_mul_builds_tables_with_step_low_c_high_layout() {
        let rs = row_size();
        let rows = vec![
            G1ScalarMulNative {
                base_point: (Fq::from_u64(5), Fq::from_u64(7)),
                x_a: vec![Fq::from_u64(1); rs],
                y_a: vec![Fq::from_u64(2); rs],
                x_t: vec![Fq::from_u64(3); rs],
                y_t: vec![Fq::from_u64(4); rs],
                x_a_next: vec![Fq::from_u64(6); rs],
                y_a_next: vec![Fq::from_u64(8); rs],
                t_indicator: vec![Fq::zero(); rs],
                a_indicator: vec![Fq::zero(); rs],
            },
            G1ScalarMulNative {
                base_point: (Fq::from_u64(9), Fq::from_u64(11)),
                x_a: vec![Fq::from_u64(10); rs],
                y_a: vec![Fq::from_u64(20); rs],
                x_t: vec![Fq::from_u64(30); rs],
                y_t: vec![Fq::from_u64(40); rs],
                x_a_next: vec![Fq::from_u64(60); rs],
                y_a_next: vec![Fq::from_u64(80); rs],
                t_indicator: vec![Fq::zero(); rs],
                a_indicator: vec![Fq::zero(); rs],
            },
        ];
        let public_inputs = vec![
            G1ScalarMulPublicInputs::new(ark_bn254::Fr::from(0u64)),
            G1ScalarMulPublicInputs::new(ark_bn254::Fr::from(1u64)),
        ];
        let mut transcript = Blake2bTranscript::new(b"test_fused_g1_scalar_mul_layout");
        let prover = FusedG1ScalarMulProver::new(&rows, &public_inputs, &mut transcript);
        let MultilinearPolynomial::LargeScalars(xa) = &prover.x_a else {
            panic!("expected LargeScalars");
        };
        // total_len = padded(2)->2 * 256
        assert_eq!(xa.Z.len(), 2 * rs);
        // c=0 block then c=1 block
        assert_eq!(xa.Z[0], Fq::from_u64(1));
        assert_eq!(xa.Z[rs], Fq::from_u64(10));
    }
}
