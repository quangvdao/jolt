//! G1 scalar multiplication sumcheck
//!
//! - We batch over a **family-local** constraint index `c_g1smul`.
//! - The native step trace domain remains 8 variables (256 steps).
//! - All committed witness polynomials are treated as MLEs over `(step, c_g1smul)` with
//!   `step` in the low bits and `c` as a suffix (LSB-first binding order).
//! - Padding rows are gated by a public indicator `I_g1smul(c)`.
//!
//! ## Sumcheck relation: `G1ScalarMul`
//!
//! **Input claim:** `0`.
//!
//! Let:
//! - `s ∈ {0,1}^8` be the step variables (LSB-first),
//! - `c_common ∈ {0,1}^{k_common}` be the Stage-2-aligned common constraint-index domain,
//! - `dummy = k_common - k_smul` (dummy **low** bits),
//! - `c_tail = c_common[dummy..] ∈ {0,1}^{k_smul}` (family-local suffix),
//! - `r = (r_s, r_c_common)` be the verifier-sampled `eq_point`,
//! - `eq(r, (s,c_common)) = eq(r_s, s) · eq(r_c_common, c_common)` be the multilinear equality
//!   polynomial (LSB-first indexing).
//!
//! This sumcheck proves the following identity over `(s,c_common) ∈ {0,1}^{8+k_common}`:
//!
//! ```text
//! Σ_{s,c_common} eq(r, (s,c_common)) · I_g1smul(c_common) · C_g1smul(s, c_tail; δ, public_inputs) = 0
//! ```
//!
//! where:
//! - `I_g1smul(c_common) ∈ {0,1}` gates padding rows,
//! - `δ` is the transcript-sampled term-batching coefficient (`term_batch_coeff`),
//! - `C_g1smul(...)` is the (batched) G1 scalar-mul constraint polynomial evaluated in
//!   `G1ScalarMulValues::eval_constraint(...)`, using public scalar bits/basepoints.
//!
//! **Dummy-bit convention:** when `k_common > k_smul`, witness polynomials are replicated across the
//! `dummy` low bits of `c_common`, but openings are cached at the family-local point
//! `(r_s, r_c_tail)` via `normalize_opening_point(...)`.
//!
//! ## Sumcheck relation: `ShiftG1ScalarMul`
//!
//! **Input claim:** `0`.
//!
//! This is a *shift-consistency* check (no new openings) that enforces, at a random reference point,
//! that the committed “next” columns correspond to a one-step shift in the step variable.
//!
//! Let `step_ref ∈ Fq^8`, `c_ref ∈ Fq^{k_common}` and `γ ∈ Fq` be transcript-sampled, and define:
//! - `Eq(step_ref, s)` as `eq_lsb_evals(step_ref)[s]`,
//! - `EqPlusOne(step_ref, s)` as `eq_plus_one_lsb_evals(step_ref)[s]` (i.e. `Eq(step_ref+1, s)`),
//! - `not_last(s) ∈ {0,1}` as 1 except at the final step (`s = 255`), where it is 0.
//!
//! The sumcheck proves:
//!
//! ```text
//! Σ_{s,c_common} Eq(c_ref, c_common) · (
//!     (Eq(step_ref, s) · not_last(s) · x_a_next(s,c_common) - EqPlusOne(step_ref, s) · x_a(s,c_common))
//!   + γ · (Eq(step_ref, s) · not_last(s) · y_a_next(s,c_common) - EqPlusOne(step_ref, s) · y_a(s,c_common))
//! ) = 0
//! ```
//!
//! which implies (informally) `x_a_next(s,·)=x_a(s+1,·)` and `y_a_next(s,·)=y_a(s+1,·)` for non-last
//! steps, at the randomly sampled point.

use allocative::Allocative;
use ark_bn254::Fq;
use ark_ff::{One, PrimeField, Zero};
use core::cmp::max;
use rayon::prelude::*;

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
    zkvm::recursion::constraints::system::G1ScalarMulNative,
    zkvm::recursion::g1::types::G1ScalarMulPublicInputs,
    zkvm::recursion::gt::types::{
        eq_lsb_evals, eq_lsb_mle, eq_plus_one_lsb_evals, eq_plus_one_lsb_mle,
    },
    zkvm::witness::{G1ScalarMulTerm, RecursionPoly, VirtualPolynomial},
};

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
pub struct G1ScalarMulParams {
    /// Step variable count (always 8 for the 256-step scalar-mul trace).
    pub num_step_vars: usize,
    /// Family-local constraint-index var count (`k_smul`).
    pub k_smul: usize,
    /// Common suffix length used for Stage-2 alignment (`k_common >= k_smul`).
    ///
    /// G1 wiring uses `k_g1 = max(k_smul, k_add)`. We keep the
    /// committed polynomials family-local (k_smul), but allow the sumcheck to run over k_common
    /// by replicating across dummy low bits and caching openings at the family-local point.
    pub k_common: usize,
    pub num_constraints: usize,
    /// Padded family-local constraint count (power of two, min 1): `2^{k_smul}`.
    pub num_constraints_padded: usize,
    /// Padded common constraint count (power of two, min 1): `2^{k_common}`.
    pub num_constraints_padded_common: usize,
}

impl G1ScalarMulParams {
    pub fn new(num_constraints: usize) -> Self {
        let num_constraints_padded = num_constraints.max(1).next_power_of_two();
        let k_smul = num_constraints_padded.trailing_zeros() as usize;
        Self::new_with_k_common(num_constraints, k_smul)
    }

    pub fn new_with_k_common(num_constraints: usize, k_common: usize) -> Self {
        let num_constraints_padded = num_constraints.max(1).next_power_of_two();
        let k_smul = num_constraints_padded.trailing_zeros() as usize;
        let k_common = max(k_common, k_smul);
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

impl SumcheckInstanceParams<Fq> for G1ScalarMulParams {
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
struct G1ScalarMulValues {
    x_a: Fq,
    y_a: Fq,
    x_t: Fq,
    y_t: Fq,
    x_a_next: Fq,
    y_a_next: Fq,
    t_indicator: Fq,
    a_indicator: Fq,
}

impl G1ScalarMulValues {
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
pub struct G1ScalarMulProver {
    params: G1ScalarMulParams,
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

impl G1ScalarMulProver {
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
        let params = G1ScalarMulParams::new(rows.len());
        Self::new_with_params(rows, public_inputs, params, transcript)
    }

    pub fn new_with_k_common<T: Transcript>(
        rows: &[G1ScalarMulNative],
        public_inputs: &[G1ScalarMulPublicInputs],
        k_common: usize,
        transcript: &mut T,
    ) -> Self {
        debug_assert_eq!(rows.len(), public_inputs.len());
        let params = G1ScalarMulParams::new_with_k_common(rows.len(), k_common);
        Self::new_with_params(rows, public_inputs, params, transcript)
    }

    fn new_with_params<T: Transcript>(
        rows: &[G1ScalarMulNative],
        public_inputs: &[G1ScalarMulPublicInputs],
        params: G1ScalarMulParams,
        transcript: &mut T,
    ) -> Self {
        let num_rounds = params.num_rounds();

        // Sample eq point for the (step,c) domain.
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

        // Public bit polynomial per instance: 8-var bit table, batched over c.
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

        // Public base point polynomials (constant over step, batched over c).
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

impl<FqT: Transcript> SumcheckInstanceProver<Fq, FqT> for G1ScalarMulProver {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<Fq> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: Fq) -> UniPoly<Fq> {
        let num_remaining = self.eq_poly.get_num_vars();
        debug_assert!(
            num_remaining > 0,
            "g1 scalar mul should have at least one round"
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
                    let vals = G1ScalarMulValues::from_poly_evals(&poly_evals, t);
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
        // Default opening point for step-trace polynomials: (r_step, r_c_tail).
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        // Base point polynomials are **c-only** (no 8-var step domain): open them at r_c_tail only.
        let r_c_chal = &sumcheck_challenges[STEP_VARS..];
        let dummy = self.params.dummy_bits();
        let r_c_tail: Vec<<Fq as JoltField>::Challenge> = r_c_chal[dummy..].to_vec();
        let base_opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(r_c_tail);

        for (vp, claim) in [
            (
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::XA,
                }),
                self.x_a.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::YA,
                }),
                self.y_a.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::XT,
                }),
                self.x_t.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::YT,
                }),
                self.y_t.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::XANext,
                }),
                self.x_a_next.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::YANext,
                }),
                self.y_a_next.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::TIndicator,
                }),
                self.t_indicator.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::AIndicator,
                }),
                self.a_indicator.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::XP,
                }),
                self.x_p.get_bound_coeff(0),
            ),
            (
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::YP,
                }),
                self.y_p.get_bound_coeff(0),
            ),
        ] {
            let op = if matches!(
                vp,
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::XP | G1ScalarMulTerm::YP
                })
            ) {
                base_opening_point.clone()
            } else {
                opening_point.clone()
            };
            accumulator.append_virtual(transcript, vp, SumcheckId::G1ScalarMul, op, claim);
        }
    }
}

pub struct G1ScalarMulVerifier {
    params: G1ScalarMulParams,
    eq_point: Vec<<Fq as JoltField>::Challenge>,
    term_batch_coeff: Fq,
    num_constraints: usize,
    /// For each scalar, the step indices `s ∈ [0,256)` where bit(s) = 1, where
    /// `bit(s)` corresponds to the MSB-first scalar bit at position `255 - s`.
    ///
    /// This lets us evaluate the step-bit MLE at `r_step` as:
    /// `Σ_{s ∈ set_bits} Eq(r_step, s)` without allocating 256-element tables.
    scalar_set_bits: Vec<Vec<u16>>,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for G1ScalarMulVerifier {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl G1ScalarMulVerifier {
    pub fn new<T: Transcript>(
        num_constraints: usize,
        public_inputs: Vec<G1ScalarMulPublicInputs>,
        transcript: &mut T,
    ) -> Self {
        debug_assert_eq!(public_inputs.len(), num_constraints);
        let params = G1ScalarMulParams::new(num_constraints);
        Self::new_with_params(params, num_constraints, public_inputs, transcript)
    }

    pub fn new_with_k_common<T: Transcript>(
        num_constraints: usize,
        k_common: usize,
        public_inputs: Vec<G1ScalarMulPublicInputs>,
        transcript: &mut T,
    ) -> Self {
        debug_assert_eq!(public_inputs.len(), num_constraints);
        let params = G1ScalarMulParams::new_with_k_common(num_constraints, k_common);
        Self::new_with_params(params, num_constraints, public_inputs, transcript)
    }

    fn new_with_params<T: Transcript>(
        params: G1ScalarMulParams,
        num_constraints: usize,
        public_inputs: Vec<G1ScalarMulPublicInputs>,
        transcript: &mut T,
    ) -> Self {
        let num_rounds = params.num_rounds();
        let eq_point: Vec<<Fq as JoltField>::Challenge> = (0..num_rounds)
            .map(|_| transcript.challenge_scalar_optimized::<Fq>())
            .collect();
        let term_batch_coeff: Fq = transcript.challenge_scalar_optimized::<Fq>().into();

        // Precompute set-bit indices for each scalar (not counted in `expected_output_claim`).
        let scalar_set_bits: Vec<Vec<u16>> = public_inputs
            .iter()
            .map(|pi| {
                let bigint = pi.scalar.into_bigint();
                let limbs = bigint.as_ref();
                let mut out = Vec::new();
                // Step index `s` corresponds to scalar bit position `255 - s` (MSB-first).
                for s in 0..256usize {
                    let bit_pos = 255usize - s;
                    let limb = limbs[bit_pos / 64];
                    let bit = ((limb >> (bit_pos % 64)) & 1) == 1;
                    if bit {
                        out.push(s as u16);
                    }
                }
                out
            })
            .collect();

        Self {
            params,
            eq_point,
            term_batch_coeff,
            num_constraints,
            scalar_set_bits,
        }
    }
}

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for G1ScalarMulVerifier {
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

        // Precompute Eq(r_step, s) once for all instances, and Eq(r_c, c) once for all constraints.
        let eq_step = eq_lsb_evals::<Fq>(r_step);
        debug_assert_eq!(eq_step.len(), 1 << STEP_VARS);
        let eq_c = eq_lsb_evals::<Fq>(r_c_tail);

        // Public inputs at this point:
        // - base point is committed as (x_p, y_p) over (step,c), but opened as **c-only**
        //   (no 8-var step padding)
        // - bit is step-only per instance, then batched across c by Eq(r_c,c)
        // Combine indicator + public input batching in one pass (avoid recomputing Eq(r_c, c)).
        let mut ind_eval = Fq::zero();
        let mut bit = Fq::zero();
        for c in 0..self.num_constraints {
            let w_c = eq_c[c];
            ind_eval += w_c;
            // bit_c(r_step) = Σ_{s ∈ set_bits_c} Eq(r_step, s)
            let mut bit_c = Fq::zero();
            for &s in &self.scalar_set_bits[c] {
                bit_c += eq_step[s as usize];
            }
            bit += w_c * bit_c;
        }

        let x_p = accumulator.get_virtual_polynomial_claim(
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::XP,
            }),
            SumcheckId::G1ScalarMul,
        );
        let y_p = accumulator.get_virtual_polynomial_claim(
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::YP,
            }),
            SumcheckId::G1ScalarMul,
        );

        // Fetch opened claims (8 committed witness polynomials), without heap allocation.
        let vals = G1ScalarMulValues {
            x_a: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::XA,
                }),
                SumcheckId::G1ScalarMul,
            ),
            y_a: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::YA,
                }),
                SumcheckId::G1ScalarMul,
            ),
            x_t: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::XT,
                }),
                SumcheckId::G1ScalarMul,
            ),
            y_t: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::YT,
                }),
                SumcheckId::G1ScalarMul,
            ),
            x_a_next: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::XANext,
                }),
                SumcheckId::G1ScalarMul,
            ),
            y_a_next: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::YANext,
                }),
                SumcheckId::G1ScalarMul,
            ),
            t_indicator: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::TIndicator,
                }),
                SumcheckId::G1ScalarMul,
            ),
            a_indicator: accumulator.get_virtual_polynomial_claim(
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::AIndicator,
                }),
                SumcheckId::G1ScalarMul,
            ),
        };
        let constraint_value = vals.eval_constraint(bit, x_p, y_p, self.term_batch_coeff);

        eq_eval * ind_eval * constraint_value
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        transcript: &mut T,
        sumcheck_challenges: &[<Fq as JoltField>::Challenge],
    ) {
        // Default opening point for step-trace polynomials: (r_step, r_c_tail).
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        // Base point polynomials are **c-only** (no 8-var step domain): open them at r_c_tail only.
        let r_c_chal = &sumcheck_challenges[STEP_VARS..];
        let dummy = self.params.dummy_bits();
        let r_c_tail: Vec<<Fq as JoltField>::Challenge> = r_c_chal[dummy..].to_vec();
        let base_opening_point = OpeningPoint::<BIG_ENDIAN, Fq>::new(r_c_tail);
        for vp in [
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::XA,
            }),
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::YA,
            }),
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::XT,
            }),
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::YT,
            }),
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::XANext,
            }),
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::YANext,
            }),
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::TIndicator,
            }),
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::AIndicator,
            }),
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::XP,
            }),
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::YP,
            }),
        ] {
            let op = if matches!(
                vp,
                VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                    term: G1ScalarMulTerm::XP | G1ScalarMulTerm::YP
                })
            ) {
                base_opening_point.clone()
            } else {
                opening_point.clone()
            };
            accumulator.append_virtual(transcript, vp, SumcheckId::G1ScalarMul, op);
        }
    }
}

#[derive(Clone, Debug, Allocative)]
pub struct ShiftG1ScalarMulParams {
    pub num_step_vars: usize, // 8
    pub k_smul: usize,
    pub k_common: usize,
}

impl ShiftG1ScalarMulParams {
    pub fn new(num_constraints: usize) -> Self {
        let k_smul = k_from_num_constraints(num_constraints);
        Self::new_with_k_common(num_constraints, k_smul)
    }

    pub fn new_with_k_common(num_constraints: usize, k_common: usize) -> Self {
        let k_smul = k_from_num_constraints(num_constraints);
        Self {
            num_step_vars: STEP_VARS,
            k_smul,
            k_common: max(k_common, k_smul),
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
pub struct ShiftG1ScalarMulProver {
    params: ShiftG1ScalarMulParams,
    // full-domain weight polynomials over (step,c)
    eq_step_poly: MultilinearPolynomial<Fq>,
    eq_minus_one_step_poly: MultilinearPolynomial<Fq>,
    not_last_poly: MultilinearPolynomial<Fq>,
    eq_c_poly: MultilinearPolynomial<Fq>,
    // witness polys (step,c)
    x_a: MultilinearPolynomial<Fq>,
    x_a_next: MultilinearPolynomial<Fq>,
    y_a: MultilinearPolynomial<Fq>,
    y_a_next: MultilinearPolynomial<Fq>,
    gamma: Fq,
}

impl ShiftG1ScalarMulProver {
    pub fn new<T: Transcript>(rows: &[G1ScalarMulNative], transcript: &mut T) -> Self {
        let params = ShiftG1ScalarMulParams::new(rows.len());
        Self::new_with_params(rows, params, transcript)
    }

    pub fn new_with_k_common<T: Transcript>(
        rows: &[G1ScalarMulNative],
        k_common: usize,
        transcript: &mut T,
    ) -> Self {
        let params = ShiftG1ScalarMulParams::new_with_k_common(rows.len(), k_common);
        Self::new_with_params(rows, params, transcript)
    }

    fn new_with_params<T: Transcript>(
        rows: &[G1ScalarMulNative],
        params: ShiftG1ScalarMulParams,
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

impl<T: Transcript> SumcheckInstanceProver<Fq, T> for ShiftG1ScalarMulProver {
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
        // No-op: reuse openings cached by `G1ScalarMul*` (same polynomials, same point).
    }
}

#[derive(Allocative)]
pub struct ShiftG1ScalarMulVerifier {
    params: ShiftG1ScalarMulParams,
    step_ref: Vec<<Fq as JoltField>::Challenge>,
    c_ref: Vec<<Fq as JoltField>::Challenge>,
    gamma: Fq,
}

impl ShiftG1ScalarMulVerifier {
    pub fn new<T: Transcript>(num_constraints: usize, transcript: &mut T) -> Self {
        let params = ShiftG1ScalarMulParams::new(num_constraints);
        Self::new_with_params(params, transcript)
    }

    pub fn new_with_k_common<T: Transcript>(
        num_constraints: usize,
        k_common: usize,
        transcript: &mut T,
    ) -> Self {
        let params = ShiftG1ScalarMulParams::new_with_k_common(num_constraints, k_common);
        Self::new_with_params(params, transcript)
    }

    fn new_with_params<T: Transcript>(params: ShiftG1ScalarMulParams, transcript: &mut T) -> Self {
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

impl<T: Transcript> SumcheckInstanceVerifier<Fq, T> for ShiftG1ScalarMulVerifier {
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

        let xa = accumulator.get_virtual_polynomial_claim(
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::XA,
            }),
            SumcheckId::G1ScalarMul,
        );
        let xan = accumulator.get_virtual_polynomial_claim(
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::XANext,
            }),
            SumcheckId::G1ScalarMul,
        );
        let ya = accumulator.get_virtual_polynomial_claim(
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::YA,
            }),
            SumcheckId::G1ScalarMul,
        );
        let yan = accumulator.get_virtual_polynomial_claim(
            VirtualPolynomial::Recursion(RecursionPoly::G1ScalarMul {
                term: G1ScalarMulTerm::YANext,
            }),
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
        // No-op: reuse openings cached by `G1ScalarMul*` (same polynomials, same point).
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcripts::Blake2bTranscript;

    #[test]
    fn g1_scalar_mul_params_has_expected_rounds() {
        let p = G1ScalarMulParams::new(3);
        assert_eq!(p.num_step_vars, 8);
        assert_eq!(p.k_smul, 2); // padded to 4
        assert_eq!(p.k_common, 2);
        assert_eq!(p.num_rounds(), 10);
    }

    #[test]
    fn shift_params_has_expected_rounds() {
        let p = ShiftG1ScalarMulParams::new(3);
        assert_eq!(p.num_rounds(), 10);
    }

    #[test]
    fn g1_scalar_mul_builds_tables_with_step_low_c_high_layout() {
        let rs = row_size();
        let rows = vec![
            G1ScalarMulNative {
                base_point: (Fq::from(5u64), Fq::from(7u64)),
                x_a: vec![Fq::from(1u64); rs],
                y_a: vec![Fq::from(2u64); rs],
                x_t: vec![Fq::from(3u64); rs],
                y_t: vec![Fq::from(4u64); rs],
                x_a_next: vec![Fq::from(6u64); rs],
                y_a_next: vec![Fq::from(8u64); rs],
                t_indicator: vec![Fq::zero(); rs],
                a_indicator: vec![Fq::zero(); rs],
            },
            G1ScalarMulNative {
                base_point: (Fq::from(9u64), Fq::from(11u64)),
                x_a: vec![Fq::from(10u64); rs],
                y_a: vec![Fq::from(20u64); rs],
                x_t: vec![Fq::from(30u64); rs],
                y_t: vec![Fq::from(40u64); rs],
                x_a_next: vec![Fq::from(60u64); rs],
                y_a_next: vec![Fq::from(80u64); rs],
                t_indicator: vec![Fq::zero(); rs],
                a_indicator: vec![Fq::zero(); rs],
            },
        ];
        let public_inputs = vec![
            G1ScalarMulPublicInputs::new(ark_bn254::Fr::from(0u64)),
            G1ScalarMulPublicInputs::new(ark_bn254::Fr::from(1u64)),
        ];
        let mut transcript = Blake2bTranscript::new(b"test_g1_scalar_mul_layout");
        let prover = G1ScalarMulProver::new(&rows, &public_inputs, &mut transcript);
        let MultilinearPolynomial::LargeScalars(xa) = &prover.x_a else {
            panic!("expected LargeScalars");
        };
        // total_len = padded(2)->2 * 256
        assert_eq!(xa.Z.len(), 2 * rs);
        // c=0 block then c=1 block
        assert_eq!(xa.Z[0], Fq::from(1u64));
        assert_eq!(xa.Z[rs], Fq::from(10u64));
    }
}
