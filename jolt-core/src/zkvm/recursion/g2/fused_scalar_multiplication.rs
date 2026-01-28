//! Fused G2 scalar multiplication sumchecks (family-local, Option B).
//!
//! This mirrors `g1/fused_scalar_multiplication.rs`, but for G2 points over Fq2 (split into
//! (c0,c1) components over Fq).
//!
//! Key conventions:
//! - Native step domain is 8 variables (256 steps).
//! - We fuse over a **family-local** constraint index `c_g2smul` (k_smul bits).
//! - For Stage-2 alignment we optionally run over `k_common >= k_smul` c-bits by replicating
//!   across dummy **low** bits (dummy-low-bits convention).
//! - The committed witness polynomials are MLEs over `(step, c_g2smul)` (step low bits, c suffix).

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
    zkvm::recursion::constraints::system::{index_to_binary, G2ScalarMulNative},
    zkvm::recursion::g2::scalar_multiplication::G2ScalarMulPublicInputs,
    zkvm::recursion::gt::shift::{
        eq_lsb_evals, eq_lsb_mle, eq_plus_one_lsb_evals, eq_plus_one_lsb_mle,
    },
    zkvm::witness::VirtualPolynomial,
};

use allocative::Allocative;
use ark_bn254::{Fq, Fq2};
use ark_ff::{One, Zero};
use rayon::prelude::*;

const STEP_VARS: usize = 8; // 256 steps

// Conservative degree bound:
// - per-instance G2 scalar-mul constraint list uses degree 6,
// - we multiply by Eq(step,c) (deg 1) and I(c) (deg 1),
// so we use 8.
const DEGREE: usize = 8;

#[inline]
fn k_from_num_constraints(num_constraints: usize) -> usize {
    num_constraints.max(1).next_power_of_two().trailing_zeros() as usize
}

#[inline]
fn row_size() -> usize {
    1usize << STEP_VARS
}

#[inline]
fn build_bit_poly_8(public_input: &G2ScalarMulPublicInputs) -> Vec<Fq> {
    let bits = public_input.bits_msb();
    debug_assert_eq!(bits.len(), 256);
    let mut evals = vec![Fq::zero(); 1usize << STEP_VARS];
    for i in 0..256 {
        evals[i] = if bits[i] { Fq::one() } else { Fq::zero() };
    }
    evals
}

#[derive(Clone, Allocative)]
pub struct FusedG2ScalarMulParams {
    /// Step variable count (always 8 for the 256-step scalar-mul trace).
    pub num_step_vars: usize,
    /// Family-local constraint-index var count (`k_smul`).
    pub k_smul: usize,
    /// Common suffix length used for Stage-2 alignment (`k_common >= k_smul`).
    pub k_common: usize,
    pub num_constraints: usize,
    /// Padded family-local constraint count (power of two, min 1): `2^{k_smul}`.
    pub num_constraints_padded: usize,
    /// Padded common constraint count (power of two, min 1): `2^{k_common}`.
    pub num_constraints_padded_common: usize,
}

impl FusedG2ScalarMulParams {
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

impl SumcheckInstanceParams<Fq> for FusedG2ScalarMulParams {
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
        // Dummy-bit convention: dummy bits are the *low* bits of the common c segment, so the
        // family-local bits are the *suffix* of the common c.
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

#[derive(Clone, Copy, Debug)]
struct FusedG2ScalarMulValues {
    x_a_c0: Fq,
    x_a_c1: Fq,
    y_a_c0: Fq,
    y_a_c1: Fq,
    x_t_c0: Fq,
    x_t_c1: Fq,
    y_t_c0: Fq,
    y_t_c1: Fq,
    x_a_next_c0: Fq,
    x_a_next_c1: Fq,
    y_a_next_c0: Fq,
    y_a_next_c1: Fq,
    t_indicator: Fq,
    a_indicator: Fq,
}

impl FusedG2ScalarMulValues {
    #[inline]
    fn from_claims(claims: &[Fq]) -> Self {
        Self {
            x_a_c0: claims[0],
            x_a_c1: claims[1],
            y_a_c0: claims[2],
            y_a_c1: claims[3],
            x_t_c0: claims[4],
            x_t_c1: claims[5],
            y_t_c0: claims[6],
            y_t_c1: claims[7],
            x_a_next_c0: claims[8],
            x_a_next_c1: claims[9],
            y_a_next_c0: claims[10],
            y_a_next_c1: claims[11],
            t_indicator: claims[12],
            a_indicator: claims[13],
        }
    }

    #[inline]
    fn from_poly_evals<const D: usize>(poly_evals: &[[Fq; D]], idx: usize) -> Self {
        Self {
            x_a_c0: poly_evals[0][idx],
            x_a_c1: poly_evals[1][idx],
            y_a_c0: poly_evals[2][idx],
            y_a_c1: poly_evals[3][idx],
            x_t_c0: poly_evals[4][idx],
            x_t_c1: poly_evals[5][idx],
            y_t_c0: poly_evals[6][idx],
            y_t_c1: poly_evals[7][idx],
            x_a_next_c0: poly_evals[8][idx],
            x_a_next_c1: poly_evals[9][idx],
            y_a_next_c0: poly_evals[10][idx],
            y_a_next_c1: poly_evals[11][idx],
            t_indicator: poly_evals[12][idx],
            a_indicator: poly_evals[13][idx],
        }
    }

    /// Evaluate the batched G2 scalar-mul constraint polynomial at a point.
    ///
    /// Mirrors `g2/scalar_multiplication.rs` (per-instance) constraint semantics.
    fn eval_constraint(&self, bit: Fq, x_p: Fq2, y_p: Fq2, delta: Fq) -> Fq {
        // Reconstruct Fq2 values
        let x_a = Fq2::new(self.x_a_c0, self.x_a_c1);
        let y_a = Fq2::new(self.y_a_c0, self.y_a_c1);
        let x_t = Fq2::new(self.x_t_c0, self.x_t_c1);
        let y_t = Fq2::new(self.y_t_c0, self.y_t_c1);
        let x_a_next = Fq2::new(self.x_a_next_c0, self.x_a_next_c1);
        let y_a_next = Fq2::new(self.y_a_next_c0, self.y_a_next_c1);

        // Fq2 constants
        let one2 = Fq2::one();
        let two2 = Fq2::new(Fq::from(2u64), Fq::zero());
        let three2 = Fq2::new(Fq::from(3u64), Fq::zero());
        let four2 = Fq2::new(Fq::from(4u64), Fq::zero());
        let nine2 = Fq2::new(Fq::from(9u64), Fq::zero());
        let bit2 = Fq2::new(bit, Fq::zero());
        let ind_t2 = Fq2::new(self.t_indicator, Fq::zero());

        // C1: 4y_A²(x_T + 2x_A) - 9x_A⁴
        let y_a_sq = y_a * y_a;
        let x_a_sq = x_a * x_a;
        let c1 = four2 * y_a_sq * (x_t + two2 * x_a) - nine2 * x_a_sq * x_a_sq;

        // C2: 3x_A²(x_T - x_A) + 2y_A(y_T + y_A)
        let c2 = three2 * x_a_sq * (x_t - x_a) + two2 * y_a * (y_t + y_a);

        // C3: Conditional addition x-coord
        let c3_skip = (one2 - bit2) * (x_a_next - x_t);
        let c3_infinity = bit2 * ind_t2 * (x_a_next - x_p);
        let x_diff = x_p - x_t;
        let y_diff = y_p - y_t;
        let chord_x = (x_a_next + x_t + x_p) * x_diff * x_diff - y_diff * y_diff;
        let c3_add = bit2 * (one2 - ind_t2) * chord_x;
        let c3 = c3_skip + c3_infinity + c3_add;

        // C4: Conditional addition y-coord
        let c4_skip = (one2 - bit2) * (y_a_next - y_t);
        let c4_infinity = bit2 * ind_t2 * (y_a_next - y_p);
        let chord_y = (y_a_next + y_t) * x_diff - y_diff * (x_t - x_a_next);
        let c4_add = bit2 * (one2 - ind_t2) * chord_y;
        let c4 = c4_skip + c4_infinity + c4_add;

        // C5: ind_A * (1 - ind_T)
        let one = Fq::one();
        let c5 = self.a_indicator * (one - self.t_indicator);

        // C6: ind_T * x_T (c0,c1), ind_T * y_T (c0,c1)
        let c6_xt_c0 = self.t_indicator * self.x_t_c0;
        let c6_xt_c1 = self.t_indicator * self.x_t_c1;
        let c6_yt_c0 = self.t_indicator * self.y_t_c0;
        let c6_yt_c1 = self.t_indicator * self.y_t_c1;

        // Batch with powers of delta (13 terms)
        let d2 = delta * delta;
        let d3 = d2 * delta;
        let d4 = d3 * delta;
        let d5 = d4 * delta;
        let d6 = d5 * delta;
        let d7 = d6 * delta;
        let d8 = d7 * delta;
        let d9 = d8 * delta;
        let d10 = d9 * delta;
        let d11 = d10 * delta;
        let d12 = d11 * delta;

        c1.c0
            + delta * c1.c1
            + d2 * c2.c0
            + d3 * c2.c1
            + d4 * c3.c0
            + d5 * c3.c1
            + d6 * c4.c0
            + d7 * c4.c1
            + d8 * c5
            + d9 * c6_xt_c0
            + d10 * c6_xt_c1
            + d11 * c6_yt_c0
            + d12 * c6_yt_c1
    }
}

#[derive(Allocative)]
pub struct FusedG2ScalarMulProver {
    params: FusedG2ScalarMulParams,
    eq_poly: MultilinearPolynomial<Fq>,
    indicator_poly: MultilinearPolynomial<Fq>,
    // committed witness polys over (step,c)
    x_a_c0: MultilinearPolynomial<Fq>,
    x_a_c1: MultilinearPolynomial<Fq>,
    y_a_c0: MultilinearPolynomial<Fq>,
    y_a_c1: MultilinearPolynomial<Fq>,
    x_t_c0: MultilinearPolynomial<Fq>,
    x_t_c1: MultilinearPolynomial<Fq>,
    y_t_c0: MultilinearPolynomial<Fq>,
    y_t_c1: MultilinearPolynomial<Fq>,
    x_a_next_c0: MultilinearPolynomial<Fq>,
    x_a_next_c1: MultilinearPolynomial<Fq>,
    y_a_next_c0: MultilinearPolynomial<Fq>,
    y_a_next_c1: MultilinearPolynomial<Fq>,
    t_indicator: MultilinearPolynomial<Fq>,
    a_indicator: MultilinearPolynomial<Fq>,
    // public polys over (step,c)
    bit: MultilinearPolynomial<Fq>,
    x_p_c0: MultilinearPolynomial<Fq>,
    x_p_c1: MultilinearPolynomial<Fq>,
    y_p_c0: MultilinearPolynomial<Fq>,
    y_p_c1: MultilinearPolynomial<Fq>,
    term_batch_coeff: Fq,
}

impl FusedG2ScalarMulProver {
    pub fn new<T: Transcript>(
        rows: &[G2ScalarMulNative],
        public_inputs: &[G2ScalarMulPublicInputs],
        transcript: &mut T,
    ) -> Self {
        debug_assert_eq!(rows.len(), public_inputs.len());
        let params = FusedG2ScalarMulParams::new(rows.len());
        Self::new_with_params(rows, public_inputs, params, transcript)
    }

    pub fn new_with_k_common<T: Transcript>(
        rows: &[G2ScalarMulNative],
        public_inputs: &[G2ScalarMulPublicInputs],
        k_common: usize,
        transcript: &mut T,
    ) -> Self {
        debug_assert_eq!(rows.len(), public_inputs.len());
        let params = FusedG2ScalarMulParams::new_with_k_common(rows.len(), k_common);
        Self::new_with_params(rows, public_inputs, params, transcript)
    }

    fn new_with_params<T: Transcript>(
        rows: &[G2ScalarMulNative],
        public_inputs: &[G2ScalarMulPublicInputs],
        params: FusedG2ScalarMulParams,
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

        // Indicator I(c_common): 1 on real constraints (replicated across step and dummy bits), else 0.
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

        let build_term = |get: fn(&G2ScalarMulNative) -> &Vec<Fq>| -> MultilinearPolynomial<Fq> {
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

        let x_a_c0 = build_term(|r| &r.x_a_c0);
        let x_a_c1 = build_term(|r| &r.x_a_c1);
        let y_a_c0 = build_term(|r| &r.y_a_c0);
        let y_a_c1 = build_term(|r| &r.y_a_c1);
        let x_t_c0 = build_term(|r| &r.x_t_c0);
        let x_t_c1 = build_term(|r| &r.x_t_c1);
        let y_t_c0 = build_term(|r| &r.y_t_c0);
        let y_t_c1 = build_term(|r| &r.y_t_c1);
        let x_a_next_c0 = build_term(|r| &r.x_a_next_c0);
        let x_a_next_c1 = build_term(|r| &r.x_a_next_c1);
        let y_a_next_c0 = build_term(|r| &r.y_a_next_c0);
        let y_a_next_c1 = build_term(|r| &r.y_a_next_c1);
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
        let mut x_p_c0_sc = vec![Fq::zero(); total_len];
        let mut x_p_c1_sc = vec![Fq::zero(); total_len];
        let mut y_p_c0_sc = vec![Fq::zero(); total_len];
        let mut y_p_c1_sc = vec![Fq::zero(); total_len];
        for c_smul in 0..params.num_constraints {
            let (x_p, y_p) = rows[c_smul].base_point;
            for d in 0..(1usize << dummy) {
                let c_common = d + (c_smul << dummy);
                let off = c_common * rs;
                for s in 0..rs {
                    x_p_c0_sc[off + s] = x_p.c0;
                    x_p_c1_sc[off + s] = x_p.c1;
                    y_p_c0_sc[off + s] = y_p.c0;
                    y_p_c1_sc[off + s] = y_p.c1;
                }
            }
        }
        let x_p_c0 = MultilinearPolynomial::LargeScalars(DensePolynomial::new(x_p_c0_sc));
        let x_p_c1 = MultilinearPolynomial::LargeScalars(DensePolynomial::new(x_p_c1_sc));
        let y_p_c0 = MultilinearPolynomial::LargeScalars(DensePolynomial::new(y_p_c0_sc));
        let y_p_c1 = MultilinearPolynomial::LargeScalars(DensePolynomial::new(y_p_c1_sc));

        Self {
            params,
            eq_poly,
            indicator_poly,
            x_a_c0,
            x_a_c1,
            y_a_c0,
            y_a_c1,
            x_t_c0,
            x_t_c1,
            y_t_c0,
            y_t_c1,
            x_a_next_c0,
            x_a_next_c1,
            y_a_next_c0,
            y_a_next_c1,
            t_indicator,
            a_indicator,
            bit,
            x_p_c0,
            x_p_c1,
            y_p_c0,
            y_p_c1,
            term_batch_coeff,
        }
    }
}

impl<FqT: Transcript> SumcheckInstanceProver<Fq, FqT> for FusedG2ScalarMulProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        let num_remaining = self.eq_poly.get_num_vars();
        debug_assert!(num_remaining > 0);
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
                    &self.x_a_c0,
                    &self.x_a_c1,
                    &self.y_a_c0,
                    &self.y_a_c1,
                    &self.x_t_c0,
                    &self.x_t_c1,
                    &self.y_t_c0,
                    &self.y_t_c1,
                    &self.x_a_next_c0,
                    &self.x_a_next_c1,
                    &self.y_a_next_c0,
                    &self.y_a_next_c1,
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
                let x_p_c0_e = self
                    .x_p_c0
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let x_p_c1_e = self
                    .x_p_c1
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let y_p_c0_e = self
                    .y_p_c0
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);
                let y_p_c1_e = self
                    .y_p_c1
                    .sumcheck_evals_array::<DEGREE>(idx, BindingOrder::LowToHigh);

                let mut out = [Fq::zero(); DEGREE];
                for t in 0..DEGREE {
                    let vals = FusedG2ScalarMulValues::from_poly_evals(&poly_evals, t);
                    let x_p = Fq2::new(x_p_c0_e[t], x_p_c1_e[t]);
                    let y_p = Fq2::new(y_p_c0_e[t], y_p_c1_e[t]);
                    let c_val = vals.eval_constraint(bit_e[t], x_p, y_p, delta);
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
            &mut self.x_a_c0,
            &mut self.x_a_c1,
            &mut self.y_a_c0,
            &mut self.y_a_c1,
            &mut self.x_t_c0,
            &mut self.x_t_c1,
            &mut self.y_t_c0,
            &mut self.y_t_c1,
            &mut self.x_a_next_c0,
            &mut self.x_a_next_c1,
            &mut self.y_a_next_c0,
            &mut self.y_a_next_c1,
            &mut self.t_indicator,
            &mut self.a_indicator,
            &mut self.bit,
            &mut self.x_p_c0,
            &mut self.x_p_c1,
            &mut self.y_p_c0,
            &mut self.y_p_c1,
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
                VirtualPolynomial::g2_scalar_mul_xa_c0_fused(),
                self.x_a_c0.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g2_scalar_mul_xa_c1_fused(),
                self.x_a_c1.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g2_scalar_mul_ya_c0_fused(),
                self.y_a_c0.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g2_scalar_mul_ya_c1_fused(),
                self.y_a_c1.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g2_scalar_mul_xt_c0_fused(),
                self.x_t_c0.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g2_scalar_mul_xt_c1_fused(),
                self.x_t_c1.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g2_scalar_mul_yt_c0_fused(),
                self.y_t_c0.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g2_scalar_mul_yt_c1_fused(),
                self.y_t_c1.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g2_scalar_mul_xa_next_c0_fused(),
                self.x_a_next_c0.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g2_scalar_mul_xa_next_c1_fused(),
                self.x_a_next_c1.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g2_scalar_mul_ya_next_c0_fused(),
                self.y_a_next_c0.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g2_scalar_mul_ya_next_c1_fused(),
                self.y_a_next_c1.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g2_scalar_mul_t_indicator_fused(),
                self.t_indicator.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::g2_scalar_mul_a_indicator_fused(),
                self.a_indicator.get_bound_coeff(0),
            ),
        ] {
            accumulator.append_virtual(
                transcript,
                vp,
                SumcheckId::G2ScalarMul,
                opening_point.clone(),
                claim,
            );
        }
    }
}

pub struct FusedG2ScalarMulVerifier {
    params: FusedG2ScalarMulParams,
    eq_point: Vec<<Fq as JoltField>::Challenge>,
    term_batch_coeff: Fq,
    num_constraints: usize,
    public_inputs: Vec<G2ScalarMulPublicInputs>,
    base_points: Vec<(Fq2, Fq2)>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for FusedG2ScalarMulVerifier {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl FusedG2ScalarMulVerifier {
    pub fn new<T: Transcript>(
        num_constraints: usize,
        public_inputs: Vec<G2ScalarMulPublicInputs>,
        base_points: Vec<(Fq2, Fq2)>,
        transcript: &mut T,
    ) -> Self {
        debug_assert_eq!(public_inputs.len(), num_constraints);
        debug_assert_eq!(base_points.len(), num_constraints);
        let params = FusedG2ScalarMulParams::new(num_constraints);
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
        public_inputs: Vec<G2ScalarMulPublicInputs>,
        base_points: Vec<(Fq2, Fq2)>,
        transcript: &mut T,
    ) -> Self {
        debug_assert_eq!(public_inputs.len(), num_constraints);
        debug_assert_eq!(base_points.len(), num_constraints);
        let params = FusedG2ScalarMulParams::new_with_k_common(num_constraints, k_common);
        Self::new_with_params(
            params,
            num_constraints,
            public_inputs,
            base_points,
            transcript,
        )
    }

    fn new_with_params<T: Transcript>(
        params: FusedG2ScalarMulParams,
        num_constraints: usize,
        public_inputs: Vec<G2ScalarMulPublicInputs>,
        base_points: Vec<(Fq2, Fq2)>,
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

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for FusedG2ScalarMulVerifier {
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
        let mut x_p = Fq2::zero();
        let mut y_p = Fq2::zero();
        let mut bit = Fq::zero();
        for c in 0..self.num_constraints {
            let bits_c = index_to_binary::<Fq>(c, k);
            let w_c = EqPolynomial::mle(&r_c, &bits_c);
            let (xp_i, yp_i) = self.base_points[c];
            let w_c_fq2 = Fq2::new(w_c, Fq::zero());
            x_p += xp_i * w_c_fq2;
            y_p += yp_i * w_c_fq2;
            bit += w_c * self.public_inputs[c].evaluate_bit_mle(&r_step_fq);
        }

        // Fetch fused opened claims (14 committed witness polynomials).
        let mut claims = Vec::with_capacity(14);
        for vp in [
            VirtualPolynomial::g2_scalar_mul_xa_c0_fused(),
            VirtualPolynomial::g2_scalar_mul_xa_c1_fused(),
            VirtualPolynomial::g2_scalar_mul_ya_c0_fused(),
            VirtualPolynomial::g2_scalar_mul_ya_c1_fused(),
            VirtualPolynomial::g2_scalar_mul_xt_c0_fused(),
            VirtualPolynomial::g2_scalar_mul_xt_c1_fused(),
            VirtualPolynomial::g2_scalar_mul_yt_c0_fused(),
            VirtualPolynomial::g2_scalar_mul_yt_c1_fused(),
            VirtualPolynomial::g2_scalar_mul_xa_next_c0_fused(),
            VirtualPolynomial::g2_scalar_mul_xa_next_c1_fused(),
            VirtualPolynomial::g2_scalar_mul_ya_next_c0_fused(),
            VirtualPolynomial::g2_scalar_mul_ya_next_c1_fused(),
            VirtualPolynomial::g2_scalar_mul_t_indicator_fused(),
            VirtualPolynomial::g2_scalar_mul_a_indicator_fused(),
        ] {
            let (_, claim) =
                accumulator.get_virtual_polynomial_opening(vp, SumcheckId::G2ScalarMul);
            claims.push(claim);
        }

        let vals = FusedG2ScalarMulValues::from_claims(&claims);
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
            VirtualPolynomial::g2_scalar_mul_xa_c0_fused(),
            VirtualPolynomial::g2_scalar_mul_xa_c1_fused(),
            VirtualPolynomial::g2_scalar_mul_ya_c0_fused(),
            VirtualPolynomial::g2_scalar_mul_ya_c1_fused(),
            VirtualPolynomial::g2_scalar_mul_xt_c0_fused(),
            VirtualPolynomial::g2_scalar_mul_xt_c1_fused(),
            VirtualPolynomial::g2_scalar_mul_yt_c0_fused(),
            VirtualPolynomial::g2_scalar_mul_yt_c1_fused(),
            VirtualPolynomial::g2_scalar_mul_xa_next_c0_fused(),
            VirtualPolynomial::g2_scalar_mul_xa_next_c1_fused(),
            VirtualPolynomial::g2_scalar_mul_ya_next_c0_fused(),
            VirtualPolynomial::g2_scalar_mul_ya_next_c1_fused(),
            VirtualPolynomial::g2_scalar_mul_t_indicator_fused(),
            VirtualPolynomial::g2_scalar_mul_a_indicator_fused(),
        ] {
            accumulator.append_virtual(
                transcript,
                vp,
                SumcheckId::G2ScalarMul,
                opening_point.clone(),
            );
        }
    }
}

#[derive(Clone, Debug, Allocative)]
pub struct FusedShiftG2ScalarMulParams {
    pub num_step_vars: usize, // 8
    pub k_smul: usize,
    pub k_common: usize,
}

impl FusedShiftG2ScalarMulParams {
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
pub struct FusedShiftG2ScalarMulProver {
    params: FusedShiftG2ScalarMulParams,
    // full-domain weight polynomials over (step,c)
    eq_step_poly: MultilinearPolynomial<Fq>,
    eq_minus_one_step_poly: MultilinearPolynomial<Fq>,
    not_last_poly: MultilinearPolynomial<Fq>,
    eq_c_poly: MultilinearPolynomial<Fq>,
    // fused witness polys (step,c)
    x_a_c0: MultilinearPolynomial<Fq>,
    x_a_c1: MultilinearPolynomial<Fq>,
    y_a_c0: MultilinearPolynomial<Fq>,
    y_a_c1: MultilinearPolynomial<Fq>,
    x_a_next_c0: MultilinearPolynomial<Fq>,
    x_a_next_c1: MultilinearPolynomial<Fq>,
    y_a_next_c0: MultilinearPolynomial<Fq>,
    y_a_next_c1: MultilinearPolynomial<Fq>,
    gamma: Fq,
}

impl FusedShiftG2ScalarMulProver {
    pub fn new<T: Transcript>(rows: &[G2ScalarMulNative], transcript: &mut T) -> Self {
        let params = FusedShiftG2ScalarMulParams::new(rows.len());
        Self::new_with_params(rows, params, transcript)
    }

    pub fn new_with_k_common<T: Transcript>(
        rows: &[G2ScalarMulNative],
        k_common: usize,
        transcript: &mut T,
    ) -> Self {
        let params = FusedShiftG2ScalarMulParams::new_with_k_common(rows.len(), k_common);
        Self::new_with_params(rows, params, transcript)
    }

    fn new_with_params<T: Transcript>(
        rows: &[G2ScalarMulNative],
        params: FusedShiftG2ScalarMulParams,
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

        // Build 8-var step weights (LSB-first domain).
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
        let build_term = |get: fn(&G2ScalarMulNative) -> &Vec<Fq>| -> MultilinearPolynomial<Fq> {
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

        let x_a_c0 = build_term(|r| &r.x_a_c0);
        let x_a_c1 = build_term(|r| &r.x_a_c1);
        let y_a_c0 = build_term(|r| &r.y_a_c0);
        let y_a_c1 = build_term(|r| &r.y_a_c1);
        let x_a_next_c0 = build_term(|r| &r.x_a_next_c0);
        let x_a_next_c1 = build_term(|r| &r.x_a_next_c1);
        let y_a_next_c0 = build_term(|r| &r.y_a_next_c0);
        let y_a_next_c1 = build_term(|r| &r.y_a_next_c1);

        Self {
            params,
            eq_step_poly,
            eq_minus_one_step_poly,
            not_last_poly,
            eq_c_poly,
            x_a_c0,
            x_a_c1,
            y_a_c0,
            y_a_c1,
            x_a_next_c0,
            x_a_next_c1,
            y_a_next_c0,
            y_a_next_c1,
            gamma,
        }
    }
}

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for FusedShiftG2ScalarMulProver {
    fn degree(&self) -> usize {
        // Eq(c)*Eq(step)*not_last*A_next - Eq(c)*EqMinusOne*A (batched across components)
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
        let half = self.x_a_c0.len() / 2;
        if half == 0 {
            return UniPoly::from_evals_and_hint(previous_claim, &[Fq::zero(); D]);
        }

        let gamma = self.gamma;
        let gamma2 = gamma * gamma;
        let gamma3 = gamma2 * gamma;

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

                let xa0 = self
                    .x_a_c0
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);
                let xa1 = self
                    .x_a_c1
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);
                let ya0 = self
                    .y_a_c0
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);
                let ya1 = self
                    .y_a_c1
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);
                let xan0 = self
                    .x_a_next_c0
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);
                let xan1 = self
                    .x_a_next_c1
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);
                let yan0 = self
                    .y_a_next_c0
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);
                let yan1 = self
                    .y_a_next_c1
                    .sumcheck_evals_array::<D>(i, BindingOrder::LowToHigh);

                let mut out = [Fq::zero(); D];
                for t in 0..D {
                    let rel_x0 = eqc[t] * (eqs[t] * not_last[t] * xan0[t] - eqm1[t] * xa0[t]);
                    let rel_x1 = eqc[t] * (eqs[t] * not_last[t] * xan1[t] - eqm1[t] * xa1[t]);
                    let rel_y0 = eqc[t] * (eqs[t] * not_last[t] * yan0[t] - eqm1[t] * ya0[t]);
                    let rel_y1 = eqc[t] * (eqs[t] * not_last[t] * yan1[t] - eqm1[t] * ya1[t]);
                    out[t] = rel_x0 + gamma * rel_x1 + gamma2 * rel_y0 + gamma3 * rel_y1;
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
            &mut self.x_a_c0,
            &mut self.x_a_c1,
            &mut self.y_a_c0,
            &mut self.y_a_c1,
            &mut self.x_a_next_c0,
            &mut self.x_a_next_c1,
            &mut self.y_a_next_c0,
            &mut self.y_a_next_c1,
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
        // No-op: reuse openings cached by `FusedG2ScalarMul*` (same polynomials, same point).
    }
}

#[derive(Allocative)]
pub struct FusedShiftG2ScalarMulVerifier {
    params: FusedShiftG2ScalarMulParams,
    step_ref: Vec<<Fq as JoltField>::Challenge>,
    c_ref: Vec<<Fq as JoltField>::Challenge>,
    gamma: Fq,
}

impl FusedShiftG2ScalarMulVerifier {
    pub fn new<T: Transcript>(num_constraints: usize, transcript: &mut T) -> Self {
        let params = FusedShiftG2ScalarMulParams::new(num_constraints);
        Self::new_with_params(params, transcript)
    }

    pub fn new_with_k_common<T: Transcript>(
        num_constraints: usize,
        k_common: usize,
        transcript: &mut T,
    ) -> Self {
        let params = FusedShiftG2ScalarMulParams::new_with_k_common(num_constraints, k_common);
        Self::new_with_params(params, transcript)
    }

    fn new_with_params<T: Transcript>(
        params: FusedShiftG2ScalarMulParams,
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

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for FusedShiftG2ScalarMulVerifier {
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

        let (_, xa0) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::g2_scalar_mul_xa_c0_fused(),
            SumcheckId::G2ScalarMul,
        );
        let (_, xa1) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::g2_scalar_mul_xa_c1_fused(),
            SumcheckId::G2ScalarMul,
        );
        let (_, ya0) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::g2_scalar_mul_ya_c0_fused(),
            SumcheckId::G2ScalarMul,
        );
        let (_, ya1) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::g2_scalar_mul_ya_c1_fused(),
            SumcheckId::G2ScalarMul,
        );
        let (_, xan0) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::g2_scalar_mul_xa_next_c0_fused(),
            SumcheckId::G2ScalarMul,
        );
        let (_, xan1) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::g2_scalar_mul_xa_next_c1_fused(),
            SumcheckId::G2ScalarMul,
        );
        let (_, yan0) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::g2_scalar_mul_ya_next_c0_fused(),
            SumcheckId::G2ScalarMul,
        );
        let (_, yan1) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::g2_scalar_mul_ya_next_c1_fused(),
            SumcheckId::G2ScalarMul,
        );

        let gamma = self.gamma;
        let gamma2 = gamma * gamma;
        let gamma3 = gamma2 * gamma;

        let rel_x0 = eqc * (eq * not_last * xan0 - eqm1 * xa0);
        let rel_x1 = eqc * (eq * not_last * xan1 - eqm1 * xa1);
        let rel_y0 = eqc * (eq * not_last * yan0 - eqm1 * ya0);
        let rel_y1 = eqc * (eq * not_last * yan1 - eqm1 * ya1);
        rel_x0 + gamma * rel_x1 + gamma2 * rel_y0 + gamma3 * rel_y1
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<Fq>,
        _transcript: &mut T,
        _sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // No-op: reuse openings cached by `FusedG2ScalarMul*` (same polynomials, same point).
    }
}
