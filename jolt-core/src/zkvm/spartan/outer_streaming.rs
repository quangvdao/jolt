// TODO: copying over outer.rs for now, then gradually upgrade to use the new streaming algorithm

use allocative::Allocative;
use ark_std::Zero;
use rayon::prelude::*;
use std::sync::Arc;
use tracer::instruction::Cycle;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::lagrange_poly::LagrangePolynomial;
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::univariate_skip::UniSkipState;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::r1cs::{
    constraints::OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
    evaluation::R1CSEval,
    inputs::{R1CSCycleInputs, ALL_R1CS_INPUTS},
};
use crate::zkvm::witness::VirtualPolynomial;
use crate::zkvm::bytecode::BytecodePreprocessing;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;

#[cfg(test)]
use crate::zkvm::r1cs::constraints::{R1CS_CONSTRAINTS_FIRST_GROUP, R1CS_CONSTRAINTS_SECOND_GROUP};
#[cfg(test)]
use crate::zkvm::r1cs::inputs::JoltR1CSInputs;

/// Degree bound of the sumcheck round polynomials for [`OuterRemainingStreamingSumcheckVerifier`].
const OUTER_REMAINING_DEGREE_BOUND: usize = 3;

/// SumcheckInstance for Spartan outer rounds after the univariate-skip first round.
/// Round 0 in this instance corresponds to the "streaming" round; subsequent rounds
/// use the remaining linear-time algorithm over cycle variables.
#[derive(Allocative)]
pub struct OuterRemainingStreamingSumcheckProver<F: JoltField> {
    #[allocative(skip)]
    bytecode_preprocessing: BytecodePreprocessing,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    split_eq_poly: GruenSplitEqPolynomial<F>,
    az: DensePolynomial<F>,
    bz: DensePolynomial<F>,
    /// The first round evals (t0, t_inf) computed from a streaming pass over the trace
    first_round_evals: (F, F),
    #[allocative(skip)]
    params: OuterRemainingStreamingSumcheckParams<F>,
    /// Challenges received via bind() (latest is last)
    received_challenges: Vec<F::Challenge>,
    /// Index of the current window in the schedule (post-uniskip rounds). Additive scaffolding.
    #[allow(dead_code)]
    current_window_idx: usize,
    /// Optional precomputed evaluation-basis grid for the current window. Additive scaffolding.
    #[allocative(skip)]
    #[allow(dead_code)]
    window_poly: Option<MultiQuadraticPolynomial<F>>,
    /// If true, materialize windows from already-bound az/bz (linear-time mode).
    /// If false, build windows by streaming from the trace (streaming mode).
    materialize_mode: bool,
}

impl<F: JoltField> OuterRemainingStreamingSumcheckProver<F> {
    /// Iterative boolean-to-ternary expansion for a window of size `omega`, in colex order with X_1 LSD.
    ///
    /// Inputs:
    /// - `a_bool`, `b_bool`: evaluations at {0,1}^omega in colex (X_1 LSD), length = 2^omega
    /// - `w_bool`: E_in weights per boolean corner, same order/length
    /// - `omit_ones`: if true, the 1-columns are set to zero at each dimension (space saving)
    ///
    /// Output:
    /// - Vec of length 3^omega in colex with X_1 LSD, where each entry equals
    ///   A(t) * B(t) * W(t) with per-dimension ∞ weight folded as (W0 + W1).
    #[inline]
    fn expand_boolean_to_ternary_straightline(
        a_bool: &[F],
        b_bool: &[F],
        w_bool: &[F],
        omega: usize,
        omit_ones: bool,
    ) -> Vec<F> {
        debug_assert_eq!(a_bool.len(), 1 << omega);
        debug_assert_eq!(b_bool.len(), 1 << omega);
        debug_assert_eq!(w_bool.len(), 1 << omega);

        let mut a_cur: Vec<F> = a_bool.to_vec();
        let mut b_cur: Vec<F> = b_bool.to_vec();
        let mut w_cur: Vec<F> = w_bool.to_vec();

        let mut pair_span = 1usize; // colex: pairs are contiguous at first, then stride grows by 3 each dim
        for _d in 0..omega {
            let l = a_cur.len();
            debug_assert!(l % (2 * pair_span) == 0);
            let out_len = (l / 2) * 3;
            let mut a_next: Vec<F> = unsafe_allocate_zero_vec(out_len);
            let mut b_next: Vec<F> = unsafe_allocate_zero_vec(out_len);
            let mut w_next: Vec<F> = unsafe_allocate_zero_vec(out_len);

            let mut out_idx = 0usize;
            let mut base = 0usize;
            while base < l {
                let end = base + 2 * pair_span;
                let mut off = 0usize;
                while off < pair_span {
                    let i0 = base + off;
                    let i1 = i0 + pair_span;
                    let a0 = a_cur[i0];
                    let a1 = a_cur[i1];
                    let b0 = b_cur[i0];
                    let b1 = b_cur[i1];
                    let w0 = w_cur[i0];
                    let w1 = w_cur[i1];

                    let da = a1 - a0;
                    let db = b1 - b0;
                    let ws = w0 + w1;

                    a_next[out_idx] = a0;
                    b_next[out_idx] = b0;
                    w_next[out_idx] = w0;
                    out_idx += 1;

                    a_next[out_idx] = if omit_ones { F::zero() } else { a1 };
                    b_next[out_idx] = if omit_ones { F::zero() } else { b1 };
                    w_next[out_idx] = if omit_ones { F::zero() } else { w1 };
                    out_idx += 1;

                    a_next[out_idx] = da;
                    b_next[out_idx] = db;
                    w_next[out_idx] = ws;
                    out_idx += 1;

                    off += 1;
                }
                base = end;
            }

            a_cur = a_next;
            b_cur = b_next;
            w_cur = w_next;
            pair_span *= 3;
        }

        let mut q: Vec<F> = unsafe_allocate_zero_vec(a_cur.len());
        for i in 0..q.len() {
            q[i] = a_cur[i] * b_cur[i] * w_cur[i];
        }
        q
    }
    #[tracing::instrument(skip_all, name = "OuterRemainingStreamingSumcheckProver::gen")]
    pub fn gen<PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, PCS>,
        num_cycles_bits: usize,
        uni: &UniSkipState<F>,
    ) -> Self {
        let (preprocessing, _, trace, _program_io, _final_mem) = state_manager.get_prover_data();

        let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&uni.r0);

        let tau_high = uni.tau[uni.tau.len() - 1];
        let tau_low = &uni.tau[..uni.tau.len() - 1];

        let lagrange_tau_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&uni.r0, &tau_high);

        let split_eq_poly: GruenSplitEqPolynomial<F> =
            GruenSplitEqPolynomial::<F>::new_with_scaling(
                tau_low,
                BindingOrder::LowToHigh,
                Some(lagrange_tau_r0),
            );

        let (t0, t_inf, az_bound, bz_bound) =
            Self::compute_first_quadratic_evals_and_bound_polys(
                &preprocessing.bytecode,
                trace,
                &lagrange_evals_r,
                &split_eq_poly,
            );

        Self {
            split_eq_poly,
            bytecode_preprocessing: preprocessing.bytecode.clone(),
            trace: state_manager.get_trace_arc(),
            az: az_bound,
            bz: bz_bound,
            first_round_evals: (t0, t_inf),
            params: OuterRemainingStreamingSumcheckParams::new(num_cycles_bits, uni),
            received_challenges: Vec::new(),
            current_window_idx: 0,
            window_poly: None,
            materialize_mode: false,
        }
    }

    /// Switch between streaming (false) and materialized (true) window construction.
    #[inline]
    pub fn set_materialize_mode(&mut self, enabled: bool) {
        self.materialize_mode = enabled;
    }

    #[inline]
    fn enter_window(&mut self, round_in_instance: usize) {
        let s = OuterRemainingStreamingSumcheckParams::<F>::schedule_round_index(round_in_instance);
        self.current_window_idx = self.params.schedule.window_index(s);
        // For ω=1 windows, materialize a single-variable Q by aggregating q(0) and q(∞) (quadratic coeff).
        // We deliberately omit q(1) for ω=1; gruen_evals_deg_3 derives it from the previous-claim s(0)+s(1).
        let omega = self.params.window_length(round_in_instance);
        if omega == 1 {
            let (q0, qinf) = if round_in_instance == 0 {
                self.first_round_evals
            } else {
                self.remaining_quadratic_evals()
            };
            self.window_poly = Some(MultiQuadraticPolynomial::new_omega1_from_q0_and_e(q0, qinf));
        } else if omega >= 2 {
            // Derive group binding (if any). For the very first streaming round there is no group bind yet.
            let r_group_opt = if self.received_challenges.is_empty() {
                None
            } else {
                Some(self.received_challenges[0])
            };
            let poly = self.materialize_window_poly_omegak(omega, r_group_opt);
            self.window_poly = Some(poly);
        } else {
            // No window to enter (ω=0).
            self.window_poly = None;
        }
    }

    /// Materialize an `omega`-variate MultiQuadraticPolynomial Q at an arbitrary window start.
    ///
    /// For each x_out in parallel:
    ///   - Iterate over x_in blocks of size 2^omega (colex with X_1 LSD).
    ///   - For each block, enumerate boolean corners m ∈ {0,1}^omega, build
    ///     A_bool[m], B_bool[m] at r_uniskip with group gating:
    ///       * If `r_group_opt` is Some(rg): combine with rg.
    ///       * Else: select group by the LSB of the full x_in index (group bit).
    ///     W_bool[m] is taken from `E_in_current` slice for that corner.
    ///   - Expand boolean arrays to ternary {0,1,∞}^omega via straightline expansion
    ///     and accumulate across blocks; multiply once by E_out[x_out].
    /// Reduce across x_out by summation; return colex (X_1 LSD) `MultiQuadraticPolynomial`.
    fn materialize_window_poly_omegak(
        &self,
        omega: usize,
        r_group_opt: Option<F::Challenge>,
    ) -> MultiQuadraticPolynomial<F> {
        // Precompute Lagrange weights over the uniskip base domain at r0 (used in streaming path)
        let lagrange_evals_r =
            LagrangePolynomial::<F>::evals::<F::Challenge, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE>(
                &self.params.r0_uniskip,
            );
        let eout = self.split_eq_poly.E_out_current();
        let ein = self.split_eq_poly.E_in_current();

        // Require enough window bits (placeholder restriction: window must fit inside current E_in tail)
        let in_len = ein.len();
        let block_len = 1usize << omega;
        if in_len < block_len || (in_len % block_len != 0) {
            debug_assert!(
                false,
                "materialize_window_poly_omegak: unsupported layout (omega={}, E_in_len={})",
                omega,
                in_len
            );
            let mut ternary_len = 1usize;
            for _ in 0..omega { ternary_len *= 3; }
            return MultiQuadraticPolynomial::new(vec![F::zero(); ternary_len], omega);
        }
        let in_blocks = in_len / block_len;

        let out_len = eout.len();
        let iter_num_x_in_vars = in_len.log_2();

        // Parallel per x_out; locally accumulate ternary grid length 3^omega
        let q_vec: Vec<F> = (0..out_len)
            .into_par_iter()
            .map(|x_out| {
                let weight_out = eout[x_out];
                let mut ternary_len = 1usize;
                for _ in 0..omega { ternary_len *= 3; }
                let mut loc_q: Vec<F> = vec![F::zero(); ternary_len];
                for g in 0..in_blocks {
                    let base = g << omega;
                    // Deep switch: choose source of (A,B) per-corner inside the tightest loop.
                    let mut a_bool: Vec<F> = unsafe_allocate_zero_vec(block_len);
                    let mut b_bool: Vec<F> = unsafe_allocate_zero_vec(block_len);
                    let mut w_bool: Vec<F> = unsafe_allocate_zero_vec(block_len);
                    for corner in 0..block_len {
                        let idx_in = base | corner;
                        let g_idx = (x_out << iter_num_x_in_vars) | idx_in;
                        let (a, b) = if self.materialize_mode {
                            // Use bound az/bz pairs at (x_out, idx_in)
                            let az0 = self.az[2 * g_idx];
                            let az1 = self.az[2 * g_idx + 1];
                            let bz0 = self.bz[2 * g_idx];
                            let bz1 = self.bz[2 * g_idx + 1];
                            if let Some(rg) = r_group_opt {
                                (az0 + rg * (az1 - az0), bz0 + rg * (bz1 - bz0))
                            } else {
                                if (idx_in & 1) == 0 { (az0, bz0) } else { (az1, bz1) }
                            }
                        } else {
                            // Stream from the trace at the physical row index
                            let row_idx = g_idx;
                                        let row = R1CSCycleInputs::from_trace::<F>(
                                            &self.bytecode_preprocessing,
                                            &self.trace,
                                            row_idx,
                                        );
                            let eval = R1CSEval::<F>::from_cycle_inputs(&row);
                            let a0 = eval.az_at_r_first_group(&lagrange_evals_r);
                            let b0 = eval.bz_at_r_first_group(&lagrange_evals_r);
                            let a1 = eval.az_at_r_second_group(&lagrange_evals_r);
                            let b1 = eval.bz_at_r_second_group(&lagrange_evals_r);
                            if let Some(rg) = r_group_opt {
                                (a0 + rg * (a1 - a0), b0 + rg * (b1 - b0))
                            } else {
                                if (idx_in & 1) == 0 { (a0, b0) } else { (a1, b1) }
                            }
                        };
                        a_bool[corner] = a;
                        b_bool[corner] = b;
                        w_bool[corner] = ein[idx_in];
                    }
                    let q_block = Self::expand_boolean_to_ternary_straightline(
                        &a_bool,
                        &b_bool,
                        &w_bool,
                        omega,
                        false, // keep q(1, ·) for in-window collapse
                    );
                    for i in 0..ternary_len {
                        loc_q[i] += q_block[i];
                    }
                }
                for i in 0..loc_q.len() {
                    loc_q[i] = weight_out * loc_q[i];
                }
                loc_q
            })
            .reduce(
                || {
                    let mut ternary_len = 1usize;
                    for _ in 0..omega { ternary_len *= 3; }
                    vec![F::zero(); ternary_len]
                },
                |mut a, b| {
                    for i in 0..a.len() { a[i] += b[i]; }
                    a
                }
            );
        MultiQuadraticPolynomial::new(q_vec, omega)
    }

    /// Compute the quadratic evaluations for the streaming round (right after univariate skip).
    ///
    /// This uses the streaming algorithm to compute the sum-check polynomial for the round
    /// right after the univariate skip round.
    ///
    /// Recall that we need to compute
    ///
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    ///       unbound_coeffs_a(x_out, x_in, 0, r) * unbound_coeffs_b(x_out, x_in, 0, r)`
    ///
    /// and
    ///
    /// `t_i(∞) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] * (unbound_coeffs_a(x_out,
    /// x_in, ∞, r) * unbound_coeffs_b(x_out, x_in, ∞, r))`
    ///
    /// Here the "_a,b" subscript indicates the coefficients of `unbound_coeffs` corresponding to
    /// Az and Bz respectively. Note that we index with x_out being the MSB here.
    ///
    /// Importantly, since the eval at `r` is not cached, we will need to recompute it via another
    /// sum
    ///
    /// `unbound_coeffs_{a,b}(x_out, x_in, {0,∞}, r) = \sum_{y in D} Lagrange(r, y) *
    /// unbound_coeffs_{a,b}(x_out, x_in, {0,∞}, y)`
    ///
    /// (and the eval at ∞ is computed as (eval at 1) - (eval at 0))
    #[inline]
    fn compute_first_quadratic_evals_and_bound_polys(
        preprocess: &BytecodePreprocessing,
        trace: &[Cycle],
        lagrange_evals_r: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        split_eq_poly: &GruenSplitEqPolynomial<F>,
    ) -> (F, F, DensePolynomial<F>, DensePolynomial<F>) {
        let num_x_out_vals = split_eq_poly.E_out_current_len();
        let num_x_in_vals = split_eq_poly.E_in_current_len();
        let iter_num_x_in_vars = num_x_in_vals.log_2();

        let groups_exact = num_x_out_vals
            .checked_mul(num_x_in_vals)
            .expect("overflow computing groups_exact");

        // Preallocate interleaved buffers once ([lo, hi] per entry)
        let mut az_bound: Vec<F> = unsafe_allocate_zero_vec(2 * groups_exact);
        let mut bz_bound: Vec<F> = unsafe_allocate_zero_vec(2 * groups_exact);

        // Parallel over x_out groups using exact-sized mutable chunks, with per-worker fold
        let (t0_acc_unr, t_inf_acc_unr) = az_bound
            .par_chunks_exact_mut(2 * num_x_in_vals)
            .zip(bz_bound.par_chunks_exact_mut(2 * num_x_in_vals))
            .enumerate()
            .fold(
                || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                |(mut acc0, mut acci), (x_out_val, (az_chunk, bz_chunk))| {
                    let mut inner_sum0 = F::Unreduced::<9>::zero();
                    let mut inner_sum_inf = F::Unreduced::<9>::zero();
                    for x_in_val in 0..num_x_in_vals {
                        let current_step_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                        let row_inputs = R1CSCycleInputs::from_trace::<F>(
                            preprocess,
                            trace,
                            current_step_idx,
                        );
                        let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);
                        let az0 = eval.az_at_r_first_group(lagrange_evals_r);
                        let bz0 = eval.bz_at_r_first_group(lagrange_evals_r);
                        let az1 = eval.az_at_r_second_group(lagrange_evals_r);
                        let bz1 = eval.bz_at_r_second_group(lagrange_evals_r);
                        let p0 = az0 * bz0;
                        let slope = (az1 - az0) * (bz1 - bz0);
                        let e_in = split_eq_poly.E_in_current()[x_in_val];
                        inner_sum0 += e_in.mul_unreduced::<9>(p0);
                        inner_sum_inf += e_in.mul_unreduced::<9>(slope);
                        let off = 2 * x_in_val;
                        az_chunk[off] = az0;
                        az_chunk[off + 1] = az1;
                        bz_chunk[off] = bz0;
                        bz_chunk[off + 1] = bz1;
                    }
                    let e_out = split_eq_poly.E_out_current()[x_out_val];
                    let reduced0 = F::from_montgomery_reduce::<9>(inner_sum0);
                    let reduced_inf = F::from_montgomery_reduce::<9>(inner_sum_inf);
                    acc0 += e_out.mul_unreduced::<9>(reduced0);
                    acci += e_out.mul_unreduced::<9>(reduced_inf);
                    (acc0, acci)
                },
            )
            .reduce(
                || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                |a, b| (a.0 + b.0, a.1 + b.1),
            );

        (
            F::from_montgomery_reduce::<9>(t0_acc_unr),
            F::from_montgomery_reduce::<9>(t_inf_acc_unr),
            DensePolynomial::new(az_bound),
            DensePolynomial::new(bz_bound),
        )
    }

    // No special binding path needed; az/bz hold interleaved [lo,hi] ready for binding

    /// Compute the polynomial for each of the remaining rounds, using the
    /// linear-time algorithm with split-eq optimizations.
    ///
    /// At this point, we have computed the `bound_coeffs` for the current round.
    /// We need to compute:
    ///
    /// `t_i(0) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// (az_bound[x_out, x_in, 0] * bz_bound[x_out, x_in, 0] - cz_bound[x_out, x_in, 0])`
    ///
    /// and
    ///
    /// `t_i(∞) = \sum_{x_out} E_out[x_out] \sum_{x_in} E_in[x_in] *
    /// az_bound[x_out, x_in, ∞] * bz_bound[x_out, x_in, ∞]`
    ///
    /// (ordering of indices is MSB to LSB, so x_out is the MSB and x_in is the LSB)
    #[inline]
    fn remaining_quadratic_evals(&self) -> (F, F) {
        let eq_poly = &self.split_eq_poly;

        let n = self.az.len();
        debug_assert_eq!(n, self.bz.len());
        if eq_poly.E_in_current_len() == 1 {
            // groups are pairs (0,1)
            let groups = n / 2;
            let (t0_unr, tinf_unr) = (0..groups)
                .into_par_iter()
                .map(|g| {
                    let az0 = self.az[2 * g];
                    let az1 = self.az[2 * g + 1];
                    let bz0 = self.bz[2 * g];
                    let bz1 = self.bz[2 * g + 1];
                    let eq = eq_poly.E_out_current()[g];
                    let p0 = az0 * bz0;
                    let slope = (az1 - az0) * (bz1 - bz0);
                    let t0_unr = eq.mul_unreduced::<9>(p0);
                    let tinf_unr = eq.mul_unreduced::<9>(slope);
                    (t0_unr, tinf_unr)
                })
                .reduce(
                    || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                    |a, b| (a.0 + b.0, a.1 + b.1),
                );
            (
                F::from_montgomery_reduce::<9>(t0_unr),
                F::from_montgomery_reduce::<9>(tinf_unr),
            )
        } else {
            let num_x1_bits = eq_poly.E_in_current_len().log_2();
            let x1_len = eq_poly.E_in_current_len();
            let x2_len = eq_poly.E_out_current_len();
            let (sum0_unr, suminf_unr) = (0..x2_len)
                .into_par_iter()
                .map(|x2| {
                    let mut inner0_unr = F::Unreduced::<9>::zero();
                    let mut inner_inf_unr = F::Unreduced::<9>::zero();
                    for x1 in 0..x1_len {
                        let g = (x2 << num_x1_bits) | x1;
                        let az0 = self.az[2 * g];
                        let az1 = self.az[2 * g + 1];
                        let bz0 = self.bz[2 * g];
                        let bz1 = self.bz[2 * g + 1];
                        let e_in = eq_poly.E_in_current()[x1];
                        let p0 = az0 * bz0;
                        let slope = (az1 - az0) * (bz1 - bz0);
                        inner0_unr += e_in.mul_unreduced::<9>(p0);
                        inner_inf_unr += e_in.mul_unreduced::<9>(slope);
                    }
                    let e_out = eq_poly.E_out_current()[x2];
                    let inner0_red = F::from_montgomery_reduce::<9>(inner0_unr);
                    let inner_inf_red = F::from_montgomery_reduce::<9>(inner_inf_unr);
                    let t0_unr = e_out.mul_unreduced::<9>(inner0_red);
                    let tinf_unr = e_out.mul_unreduced::<9>(inner_inf_red);
                    (t0_unr, tinf_unr)
                })
                .reduce(
                    || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                    |a, b| (a.0 + b.0, a.1 + b.1),
                );
            (
                F::from_montgomery_reduce::<9>(sum0_unr),
                F::from_montgomery_reduce::<9>(suminf_unr),
            )
        }
    }

    pub fn final_sumcheck_evals(&self) -> [F; 2] {
        let az0 = if !self.az.is_empty() {
            self.az[0]
        } else {
            F::zero()
        };
        let bz0 = if !self.bz.is_empty() {
            self.bz[0]
        } else {
            F::zero()
        };
        [az0, bz0]
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for OuterRemainingStreamingSumcheckProver<F>
{
    fn degree(&self) -> usize {
        OUTER_REMAINING_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim
    }

    #[tracing::instrument(skip_all, name = "OuterRemainingStreamingSumcheckProver::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        // Enter a new window if this round is the start of one (after applying prior binding)
        if self.params.is_window_start(round) {
            self.enter_window(round);
        }
        // If a window poly exists, use it to answer this round (ω>=1); else fall back to legacy paths.
        // For ω=1, window_poly stores q(0) and q(∞) explicitly; for ω>=2, sum over all triples.
        let (t0, t_inf) = if let Some(w) = self.window_poly.as_ref() {
            if w.num_vars() == 1 {
                let (q0, qinf) = w.omega1_q0_and_e();
                (q0, qinf)
            } else {
                // Aggregate t(0) and t(∞) by summing over the first-dimension triples across the full grid.
                // q(1) aggregate is derived by the verifier via previous_claim (gruen_evals_deg_3).
                let mut sum_q0 = F::zero();
                let mut sum_qinf = F::zero();
                for triple in w.chunks_dim0() {
                    sum_q0 += triple[0];
                    sum_qinf += triple[2];
                }
                (sum_q0, sum_qinf)
            }
        } else if round == 0 {
            self.first_round_evals
        } else {
            self.remaining_quadratic_evals()
        };
        let evals = self
            .split_eq_poly
            .gruen_evals_deg_3(t0, t_inf, previous_claim);
        vec![evals[0], evals[1], evals[2]]
    }

    #[tracing::instrument(skip_all, name = "OuterRemainingStreamingSumcheckProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        // Ingest challenge
        self.received_challenges.push(r_j);
        // Apply binding immediately each round (canonical sumcheck ordering)
        rayon::join(
            || self.az.bind_parallel(r_j, BindingOrder::LowToHigh),
            || self.bz.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
        self.split_eq_poly.bind(r_j);
        // If we have a window grid, collapse along the first (LSD) dimension using degree-2 Lagrange basis on {0,1,∞}.
        if let Some(w) = self.window_poly.as_mut() {
            let r: F = r_j.into();
            let l0 = F::one() - r;
            let l1 = r;
            let linf = r * r - r;
            w.collapse_first_dim_in_place(|(q0, q1, qinf)| l0 * q0 + l1 * q1 + linf * qinf);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.get_opening_point(sumcheck_challenges);

        // Append Az, Bz claims and corresponding opening point
        let claims = self.final_sumcheck_evals();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SpartanAz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
            claims[0],
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SpartanBz,
            SumcheckId::SpartanOuter,
            opening_point.clone(),
            claims[1],
        );

        // Handle witness openings at r_cycle (use consistent split length)
        let (r_cycle, _rx_var) = opening_point.r.split_at(self.params.num_cycles_bits);

        // Compute claimed witness evals and append virtual openings for all R1CS inputs
        let claimed_witness_evals =
            R1CSEval::compute_claimed_inputs(&self.bytecode_preprocessing, &self.trace, r_cycle);

        #[cfg(test)]
        {
            // Recompute Az,Bz at the final opening point USING ONLY the claimed witness MLEs z(r_cycle),
            // then compare to the prover's final Az,Bz claims. This validates the consistency wiring
            // between the outer sumcheck and the witness openings.

            // Prover's final Az,Bz claims (after all bindings)
            let claims = self.final_sumcheck_evals();

            // Extract streaming-round challenge r_stream from the opening point tail (after r_cycle)
            let (_, rx_tail) = opening_point.r.split_at(self.params.num_cycles_bits);
            let r_stream = rx_tail[0];

            // Build z(r_cycle) vector extended with a trailing 1 for the constant column
            let const_col = JoltR1CSInputs::num_inputs();
            let mut z_cycle_ext = claimed_witness_evals.to_vec();
            z_cycle_ext.push(F::one());

            // Lagrange weights over the univariate-skip base domain at r0
            let w = LagrangePolynomial::<F>::evals::<F::Challenge, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE>(
                &self.params.r0_uniskip,
            );

            // Group 0 fused Az,Bz via dot product of LC with z(r_cycle)
            let mut az_g0 = F::zero();
            let mut bz_g0 = F::zero();
            for i in 0..R1CS_CONSTRAINTS_FIRST_GROUP.len() {
                let lc_a = &R1CS_CONSTRAINTS_FIRST_GROUP[i].cons.a;
                let lc_b = &R1CS_CONSTRAINTS_FIRST_GROUP[i].cons.b;
                az_g0 += w[i] * lc_a.dot_eq_ry::<F>(&z_cycle_ext, const_col);
                bz_g0 += w[i] * lc_b.dot_eq_ry::<F>(&z_cycle_ext, const_col);
            }

            // Group 1 fused Az,Bz (use same Lagrange weights order as construction)
            let mut az_g1 = F::zero();
            let mut bz_g1 = F::zero();
            let g2_len = core::cmp::min(
                R1CS_CONSTRAINTS_SECOND_GROUP.len(),
                OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
            );
            for i in 0..g2_len {
                let lc_a = &R1CS_CONSTRAINTS_SECOND_GROUP[i].cons.a;
                let lc_b = &R1CS_CONSTRAINTS_SECOND_GROUP[i].cons.b;
                az_g1 += w[i] * lc_a.dot_eq_ry::<F>(&z_cycle_ext, const_col);
                bz_g1 += w[i] * lc_b.dot_eq_ry::<F>(&z_cycle_ext, const_col);
            }

            // Bind by r_stream to match the outer streaming combination used for final Az,Bz
            let az_final = az_g0 + r_stream * (az_g1 - az_g0);
            let bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0);

            assert_eq!(
                az_final, claims[0],
                "Az final eval mismatch vs claims from evaluating R1CS inputs at r_cycle: recomputed={} claimed={}",
                az_final, claims[0]
            );
            assert_eq!(
                bz_final, claims[1],
                "Bz final eval mismatch vs claims from evaluating R1CS inputs at r_cycle: recomputed={} claimed={}",
                bz_final, claims[1]
            );
        }

        for (i, input) in ALL_R1CS_INPUTS.iter().enumerate() {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                OpeningPoint::new(r_cycle.to_vec()),
                claimed_witness_evals[i],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}


struct OuterRemainingStreamingSumcheckParams<F: JoltField> {
    /// Number of cycle bits for splitting opening points (consistent across prover/verifier)
    /// Total number of rounds is `1 + num_cycles_bits`
    num_cycles_bits: usize,
    /// Window schedule encoded as exclusive end indices for post-uniskip rounds
    /// Invariant: strictly increasing, last == num_cycles_bits
    #[allow(dead_code)]
    schedule: WindowSchedule,
    /// The tau vector (length `2 + num_cycles_bits`, sampled at the beginning for Lagrange + eq poly)
    #[allow(dead_code)]
    tau: Vec<F::Challenge>,
    /// The univariate-skip first round challenge
    r0_uniskip: F::Challenge,
    /// Claim after the univariate-skip first round, updated every round
    input_claim: F,
}

impl<F: JoltField> OuterRemainingStreamingSumcheckParams<F> {
    fn new(num_cycles_bits: usize, uni: &UniSkipState<F>) -> Self {
        Self {
            num_cycles_bits,
            schedule: WindowSchedule::single_rounds(num_cycles_bits),
            tau: uni.tau.clone(),
            r0_uniskip: uni.r0,
            input_claim: uni.claim_after_first,
        }
    }

    fn num_rounds(&self) -> usize {
        // After the uni-skip first round, this instance covers exactly the
        // cycle-bit rounds. The streaming round (round 0 here) is the first of
        // these, so the total number of rounds equals the number of cycle bits.
        self.num_cycles_bits
    }

    fn get_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let r_tail = sumcheck_challenges;
        let r_full = [&[self.r0_uniskip], r_tail].concat();
        OpeningPoint::<LITTLE_ENDIAN, F>::new(r_full).match_endianness()
    }

    /// Map a round index within this instance to the schedule index.
    /// This instance covers only post-uniskip rounds, starting at 0.
    #[allow(dead_code)]
    fn schedule_round_index(round_in_instance: usize) -> usize {
        round_in_instance
    }

    /// Get the (start,end) window bounds for a given schedule round index.
    #[allow(dead_code)]
    fn window_bounds_for_schedule_round(&self, sched_round: usize) -> (usize, usize) {
        let idx = self.schedule.window_index(sched_round);
        self.schedule.window_bounds(idx)
    }

    /// Return true if `round_in_instance` is the first round of its window.
    #[allow(dead_code)]
    fn is_window_start(&self, round_in_instance: usize) -> bool {
        let s = Self::schedule_round_index(round_in_instance);
        let (start, _) = self.window_bounds_for_schedule_round(s);
        s == start
    }

    /// Return window length ω for the window containing `round_in_instance`.
    #[allow(dead_code)]
    fn window_length(&self, round_in_instance: usize) -> usize {
        let s = Self::schedule_round_index(round_in_instance);
        let (start, end) = self.window_bounds_for_schedule_round(s);
        end - start
    }
}

/// Window schedule for streaming prover: exclusive end indices per window.
/// For example, with `ends = [2, 5, 8]`, windows are [0,2), [2,5), [5,8).
struct WindowSchedule {
    ends: Vec<usize>,
}

impl WindowSchedule {
    /// Construct a schedule where each window contains exactly one round.
    /// This preserves the legacy per-round behavior.
    fn single_rounds(num_cycles_bits: usize) -> Self {
        let mut ends = Vec::with_capacity(num_cycles_bits);
        for i in 1..=num_cycles_bits {
            ends.push(i);
        }
        let sched = Self { ends };
        sched.validate(num_cycles_bits);
        sched
    }

    /// Validate invariants under debug builds.
    fn validate(&self, num_cycles_bits: usize) {
        if self.ends.is_empty() {
            debug_assert!(num_cycles_bits == 0, "empty schedule only valid when num_cycles_bits == 0");
            return;
        }
        // Strictly increasing and ends last matches num_cycles_bits
        let mut prev = 0usize;
        for (i, &e) in self.ends.iter().enumerate() {
            debug_assert!(e > prev, "WindowSchedule.ends must be strictly increasing at index {}", i);
            prev = e;
        }
        debug_assert_eq!(
            *self.ends.last().unwrap(),
            num_cycles_bits,
            "last end must equal num_cycles_bits"
        );
    }

    /// Return index of the window containing `round` (0-based, post-uniskip).
    #[allow(dead_code)]
    fn window_index(&self, round: usize) -> usize {
        debug_assert!(
            !self.ends.is_empty() && round < *self.ends.last().unwrap(),
            "round out of bounds"
        );
        match self.ends.binary_search(&round) {
            Ok(i) => i,
            Err(i) => i,
        }
    }

    /// Return (start, end) bounds for window `idx`.
    #[allow(dead_code)]
    fn window_bounds(&self, idx: usize) -> (usize, usize) {
        debug_assert!(idx < self.ends.len(), "window index out of bounds");
        let start = if idx == 0 { 0 } else { self.ends[idx - 1] };
        let end = self.ends[idx];
        (start, end)
    }
}

/// Placeholder for evaluation-basis grid for a window of `num_vars` rounds (d=2 => 3^num_vars points).
/// The grid is {0,1,infty}^{num_vars}
#[allow(dead_code)]
struct MultiQuadraticPolynomial<F: JoltField> {
    evals: Vec<F>,
    num_vars: usize,
}

/// Storage for a window's multivariate quadratic Q over `num_vars` window variables.
///
/// Mathematical object:
///   Q(X_w, ..., X_1) = Σ_{x'} eq(tau', x') · Az(x', X_w, ..., X_1, r_bound) · Bz(x', X_w, ..., X_1, r_bound)
///
/// Storage layout (flat Vec in base-3 colex order):
/// - We store evaluations of Q on the grid {0, 1, ∞}^{num_vars}.
/// - The rightmost variable X_1 is the least-significant digit (LSD) in base-3.
/// - Indexing is colexicographic with base-3 digits (trits) [t_0, t_1, ..., t_{w-1}]
///   where t_0 corresponds to X_1 and t_{w-1} corresponds to X_w.
/// - Concretely, the linear index is:
///       idx = Σ_{i=0}^{w-1} (t_i · 3^i), with t_i ∈ {0,1,2} standing for {0,1,∞}.
///
/// Rationale:
/// - Binding order in this file is LowToHigh for cycle bits, i.e., X_1 is bound first.
/// - With X_1 as LSD, each contiguous triple [Q(0), Q(1), Q(∞)] for X_1 is stored contiguously.
///   This enables stride-1 access to compute the round cubic and to collapse along X_1 in-place.
/// - After collapsing the first dimension, X_2 becomes the new LSD with the same property,
///   allowing repeated rounds without transposes or gathers.
#[allow(dead_code)]
impl<F: JoltField> MultiQuadraticPolynomial<F> {
    /// Construct from a flat evaluation vector in colex base-3 order with X_1 as LSD.
    ///
    /// Invariant: evals.len() must be exactly 3^num_vars.
    pub fn new(evals: Vec<F>, num_vars: usize) -> Self {
        debug_assert_eq!(
            evals.len(),
            Self::pow3(num_vars),
            "MultiQuadraticPolynomial: eval length must be 3^num_vars"
        );
        Self { evals, num_vars }
    }

    /// Construct an ω=1 polynomial from q(0) and the quadratic coefficient e = q_∞
    /// while omitting q(1). The middle entry is set to zero as a placeholder and
    /// must NOT be used; round computation derives q(1) implicitly from the
    /// previous-claim s(0)+s(1) per gruen_evals_deg_3.
    ///
    /// Layout: [q(0), 0, q(∞)]
    pub fn new_omega1_from_q0_and_e(q0: F, q_inf: F) -> Self {
        Self {
            evals: vec![q0, F::zero(), q_inf],
            num_vars: 1,
        }
    }

    /// Number of variables in the window (w).
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Total number of stored evaluations (= 3^w).
    #[inline]
    pub fn len(&self) -> usize {
        self.evals.len()
    }

    /// True iff no variables remain (empty grid).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.evals.is_empty()
    }

    /// Return the linear index in base-3 colex order (X_1 is LSD).
    ///
    /// - `trits_lsd[i] ∈ {0,1,2}` encodes the value of X_{i+1} ∈ {0,1,∞}.
    /// - `trits_lsd.len()` must equal `self.num_vars`.
    #[inline]
    pub fn idx_colex(&self, trits_lsd: &[u8]) -> usize {
        debug_assert_eq!(
            trits_lsd.len(),
            self.num_vars,
            "idx_colex: trit length must match num_vars"
        );
        let mut acc = 0usize;
        let mut stride = 1usize;
        for &t in trits_lsd {
            debug_assert!(t < 3, "idx_colex: trits must be in 0..=2");
            acc += (t as usize) * stride;
            stride *= 3;
        }
        acc
    }

    /// Access an evaluation by its colex base-3 address (X_1 is LSD).
    ///
    /// Example: `get_at(&[0, 2])` returns Q(X_2=∞, X_1=0).
    #[inline]
    pub fn get_at(&self, trits_lsd: &[u8]) -> &F {
        let idx = self.idx_colex(trits_lsd);
        &self.evals[idx]
    }

    /// For ω=1 only: return (q(0), q(∞)). Panics in debug if num_vars != 1.
    #[inline]
    pub fn omega1_q0_and_e(&self) -> (F, F) {
        debug_assert_eq!(self.num_vars, 1, "omega1_q0_and_e: num_vars must be 1");
        (self.evals[0], self.evals[2])
    }

    /// Iterate over contiguous triples [Q(0), Q(1), Q(∞)] along X_1 (the first bound variable).
    ///
    /// Each triple corresponds to fixing (X_w, ..., X_2) and varying X_1 ∈ {0,1,∞}.
    /// The iterator yields chunks of length 3 in the order of colex parent indices.
    #[inline]
    pub fn chunks_dim0(&self) -> core::slice::ChunksExact<'_, F> {
        self.evals.chunks_exact(3)
    }

    /// In-place collapse of the first (LSD) dimension X_1 using a provided combining function.
    ///
    /// Typical usage patterns:
    /// - Bind X_1 to a verifier challenge r: combine((q0, q1, qinf)) = L0(r)*q0 + L1(r)*q1 + L∞(r)*qinf,
    ///   where [L0, L1, L∞] are the degree-≤2 univariate Lagrange basis polynomials through {0,1,∞}.
    /// - Extract window endpoints: combine((q0, q1, qinf)) = q0 for t(0), or (q1 − q0)·(something) for slopes.
    ///
    /// After collapsing, `num_vars` decreases by 1 and `evals` is truncated to 3^{w-1}.
    #[inline]
    pub fn collapse_first_dim_in_place(&mut self, mut combine: impl FnMut((F, F, F)) -> F) {
        debug_assert!(
            self.num_vars > 0,
            "collapse_first_dim_in_place: no variables to collapse"
        );
        let out_len = self.evals.len() / 3;
        for i in 0..out_len {
            let base = 3 * i;
            let q0 = self.evals[base];
            let q1 = self.evals[base + 1];
            let qinf = self.evals[base + 2];
            self.evals[i] = combine((q0, q1, qinf));
        }
        self.evals.truncate(out_len);
        self.num_vars -= 1;
    }

    /// Compute 3^n for small n (n ≤ 64 in practice). Panics on overflow in debug builds.
    #[inline]
    fn pow3(n: usize) -> usize {
        // Small n in this protocol; iterative multiply avoids pow() cast pitfalls.
        let mut acc: usize = 1;
        for _ in 0..n {
            acc = acc
                .checked_mul(3)
                .expect("pow3 overflow (unexpectedly large window)");
        }
        acc
    }
}
