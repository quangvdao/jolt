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
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::r1cs::{
    constraints::OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
    evaluation::R1CSEval,
    inputs::{R1CSCycleInputs, ALL_R1CS_INPUTS},
};
use crate::zkvm::witness::VirtualPolynomial;
use crate::zkvm::spartan::outer::OuterRemainingSumcheckParams;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;

#[cfg(test)]
use crate::zkvm::r1cs::constraints::{R1CS_CONSTRAINTS_FIRST_GROUP, R1CS_CONSTRAINTS_SECOND_GROUP};
#[cfg(test)]
use crate::zkvm::r1cs::inputs::JoltR1CSInputs;

// DO NOT DELETE
// One algorithm to rule them all: arbitrary bind & eval next rounds, taking source from either original trace or already-bound poly evals
// Then we specialize to these cases.
// General outline:
// A. `bind` can simply:
//   - ingest challenge and store them for later use
//   - bind the current window evals to the challenge (evals are guaranteed to exist by the time this is called, via some previous call to `compute_prover_message`)
// B. `compute_prover_message` is the real work
// 1. Determine if we are at the start of a new window
// 2. If so, we need to compute the evaluations needed for this window,
//    - If we are in materialized mode, we will use the already-bound poly evals
//    - If we are not, we will need to stream each bound poly eval from trace, and:
//        - If we need to materialize for this window, we will store the just-computed bound evals
//        - Otherwise, we throw them away
//        - In either case, we collect enough of these evals at once (2^window_size), do extrapolation, multiply, and add
//          them to the current window evals
// 3. Now that the bound window evals are guaranteed to exist, we will use that to compute the evaluations for this round

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
    az: Option<DensePolynomial<F>>,
    bz: Option<DensePolynomial<F>>,
    #[allocative(skip)]
    params: OuterRemainingSumcheckParams<F>,
    /// Challenges received via bind() (latest is last)
    received_challenges: Vec<F::Challenge>,
    /// Window schedule encoded as exclusive end indices for post-uniskip rounds
    /// Invariant: strictly increasing, last == num_cycles_bits
    #[allow(dead_code)]
    schedule: WindowSchedule,
    /// Index of the current window in the schedule (post-uniskip rounds). Additive scaffolding.
    #[allow(dead_code)]
    current_window_idx: usize,
    /// Optional precomputed evaluation-basis grid for the current window. Additive scaffolding.
    #[allocative(skip)]
    #[allow(dead_code)]
    window_poly: Option<MultiQuadraticPolynomial<F>>,
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
        let (preprocessing, _, _trace, _program_io, _final_mem) = state_manager.get_prover_data();

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

        Self {
            split_eq_poly,
            bytecode_preprocessing: preprocessing.bytecode.clone(),
            trace: state_manager.get_trace_arc(),
            // Dummy az and bz polynomials, to be populated in the window start where materialization happens
            az: None,
            bz: None,
            params: OuterRemainingSumcheckParams::new(num_cycles_bits, uni),
            received_challenges: Vec::new(),
            // Linear-time only schedule for now
            schedule: WindowSchedule::single_rounds(num_cycles_bits),
            current_window_idx: 0,
            window_poly: None,
        }
    }

    pub fn final_sumcheck_evals(&self) -> [F; 2] {
        let az0 = match &self.az {
            Some(az) if !az.is_empty() => az[0],
            _ => F::zero(),
        };
        let bz0 = match &self.bz {
            Some(bz) if !bz.is_empty() => bz[0],
            _ => F::zero(),
        };
        [az0, bz0]
    }

    /// NEW! All-in-one function that either: iterate over the trace, OR use the already-computed
    /// bound evals, to compute the multiquadratic evals for the current window (which this one is a
    /// part of), and optionally store the bound evals if needed. This is invoked at the beginning of
    /// every window.
    /// 
    /// We make this to be mutable on self only for now, cuz self contains all the needed info.
    /// May make things more implicit later on.
    fn compute_window_evals_for_new_window(&mut self, current_round: usize) {
        // Pseudo-code: (DO NOT DELETE, IMPLEMENT BELOW THIS COMMENT)
        // 1. Check two flags: (cross-check with window schedule)
        //   - whether az / bz are already present
        //   - if az/bz are None, whether in this window we need to materialize az & bz
        //     If we need to materialize, we will allocate vectors for az_bound/bz_bound for this round.
        // 2. Look at the window schedule - what is the current round, and what is the length of the window?
        //   - This will determine the right split of E_out and E_in. NOTE: if window size is MORE than one round,
        //     the current `E_in_current` will NOT be correct. We need to shift it based on the window length.
        // 3. Key thing to keep in mind: we partition the number of variables into 4 groups
        //   - out variables
        //   - in variables
        //   - active variables (the ones within this window length)
        //   - bound variables (the ones that are already bound)
        // SpartanOuter specific: when we start in RemainingSumcheck, we ALREADY have one variable bound (r0_uniskip)
        // Picture to keep in mind: [x_out, x_in, X_ACTIVE, r_bound]
        // where X_ACTIVE = (X_w, ..., X_1), and r_bound = (..., r0_uniskip)
        // 3. Main loop:
        //   - Parallel iteration over x_out
        //   - For each x_out, iterate over x_in (assuming x_in isn't already fully bound; if it is,
        //     we can collapse / skip this iteration)
        //   - Now we iterate over 2^w to build the bound evals over all x_active_base \in {0,1}^w
        //     Recall that each eval has the form: {Az, Bz}(x_out, x_in, x_active_base, r_bound)
        //   - Here we special case:
        //     - if az/bz are already present, we can grab this eval directly from az/bz
        //     - if az/bz are not present, we need to compute this eval from the trace
        //       - This computation will call an auxiliary function that iterates over the trace PRECISELY
        //         over the 2^|num_bound_vars| slice corresponding to (x_out, x_in, x_active_base),
        //         and computes the eval for the given (x_out, x_in, x_active_base, r_bound).
        //       - If we need to materialize az/bz in this window, we store the evals in the right index of az_bound/bz_bound
        //       (will need to use `unsafe` here cuz the compiler isn't smart enough to figure out writes are safe / disjoint between threads)
        //   - Now we have the az/bz evals for all x_active_base \in {0,1}^w.
        //     Initialize accumulators (e.g. extended evals) for all {0,1,infty}^w.
        //     We call an auxiliary function that:
        //     - Iterates over all ternary indices starting from the base and extending to the {0,1,infty}^w grid
        //     - For each ternary index x_active_ext, computes the az/bz eval for the given (x_out, x_in, x_active_ext, r_bound), using the memoized evals from the previous indices
        //     - Do a fused multiply-accumulate to compute
        //       accum[x_active_ext] += e_in.mul_unreduced::<9>(az_ext * bz_ext)
        //   - Now the accumulators for the x_in are fully computed. We reduce them, then unreduced-multiply by E_out, sum them up, then reduce them at the end.
        // 4. Now we are done: the final accumulated values are the evals for the MultiQuadraticPolynomial.

        // 0) Identify window; only act at window starts
        let window_idx = self.schedule.window_index(current_round);
        let (win_start, win_end) = self.schedule.window_bounds(window_idx);
        if current_round != win_start {
            return;
        }
        let omega = win_end - win_start;
        debug_assert!(omega > 0, "window must contain at least one round");

        // 1) Decide materialization policy for this window
        let will_materialize = self.schedule_requires_materialization(window_idx);

        // 2) Build weights on {0,1}^ω for the active window
        let w_bool = self.compute_active_window_bool_weights(omega);
        debug_assert_eq!(w_bool.len(), 1 << omega, "w_bool must be 2^omega");
        let omit_ones = true;

        // 3^omega length
        let mut grid_len = 1usize;
        for _ in 0..omega {
            grid_len = grid_len.checked_mul(3).expect("overflow in 3^omega");
        }

        // 3) Fold over split-eq weights, expand {0,1}^ω → {0,1,∞}^ω, accumulate unreduced
        let e_out = self.split_eq_poly.E_out_current();
        let e_in = self.split_eq_poly.E_in_current();
        let in_len = self.split_eq_poly.E_in_current_len();
        let mut acc_unreduced: Vec<F::Unreduced<9>> = vec![F::Unreduced::<9>::zero(); grid_len];
        for x_out in 0..e_out.len() {
            let mut inner: Vec<F::Unreduced<9>> = vec![F::Unreduced::<9>::zero(); grid_len];
            if in_len <= 1 {
                // Fully bound inner
                let e_in_val = F::one();
                // Placeholder Az,Bz over {0,1}^ω
                let a_bool = vec![F::zero(); 1usize << omega];
                let b_bool = vec![F::zero(); 1usize << omega];
                let q_grid = Self::expand_boolean_to_ternary_straightline(
                    &a_bool,
                    &b_bool,
                    &w_bool,
                    omega,
                    omit_ones,
                );
                debug_assert_eq!(q_grid.len(), grid_len);
                for i in 0..grid_len {
                    inner[i] += e_in_val.mul_unreduced::<9>(q_grid[i]);
                }
            } else {
                for x_in in 0..in_len {
                    let e_in_val = e_in[x_in];
                    // Placeholder Az,Bz over {0,1}^ω for this (x_out, x_in)
                    let a_bool = vec![F::zero(); 1usize << omega];
                    let b_bool = vec![F::zero(); 1usize << omega];
                    let q_grid = Self::expand_boolean_to_ternary_straightline(
                        &a_bool,
                        &b_bool,
                        &w_bool,
                        omega,
                        omit_ones,
                    );
                    debug_assert_eq!(q_grid.len(), grid_len);
                    for i in 0..grid_len {
                        inner[i] += e_in_val.mul_unreduced::<9>(q_grid[i]);
                    }
                }
            }
            let e_out_val = e_out[x_out];
            for i in 0..grid_len {
                let reduced = F::from_montgomery_reduce::<9>(inner[i]);
                acc_unreduced[i] += e_out_val.mul_unreduced::<9>(reduced);
            }
        }

        // 4) Reduce and store window polynomial
        let mut evals = unsafe_allocate_zero_vec::<F>(grid_len);
        for i in 0..grid_len {
            evals[i] = F::from_montgomery_reduce::<9>(acc_unreduced[i]);
        }
        self.window_poly = Some(MultiQuadraticPolynomial::new(evals, omega));
        self.current_window_idx = window_idx;

        // 5) Finalize any materialization if requested
        if will_materialize {
            self.finalize_materialization_for_window(window_idx, omega);
        }
    }

    /// Decide whether this window should materialize bound Az/Bz for reuse.
    #[inline]
    fn schedule_requires_materialization(&self, _window_idx: usize) -> bool {
        // Stub policy: default to not materializing. Customize when wiring full schedule logic.
        false
    }

    /// Attempt to fetch pre-materialized Az,Bz boolean-base evals for group g (length 2^omega each).
    #[inline]
    fn fetch_materialized_group_bool_evals(&self, _g: usize, _omega: usize) -> Option<(Vec<F>, Vec<F>)> {
        // Stub: no cache by default.
        None
    }

    /// Compute Az,Bz boolean-base evals for group g by streaming the trace across the bound slice.
    #[inline]
    fn compute_group_bool_evals_from_trace(&self, _g: usize, omega: usize) -> (Vec<F>, Vec<F>) {
        // Stub: return zeroed vectors of the correct length. Replace with trace iteration.
        let len = 1usize << omega;
        (vec![F::zero(); len], vec![F::zero(); len])
    }

    /// Optionally store the computed boolean-base evals to support reuse in this window.
    #[inline]
    fn maybe_store_materialized_group_bool_evals(
        &mut self,
        _g: usize,
        _omega: usize,
        _a_bool: &[F],
        _b_bool: &[F],
    ) {
        // Stub: no-op by default.
    }

    /// Return weights on {0,1}^ω (colex, X_1 LSD) for the active window variables only.
    #[inline]
    fn compute_active_window_bool_weights(&self, omega: usize) -> Vec<F> {
        // Stub: uniform weights as placeholder; replace with correct active-window eq weights.
        vec![F::one(); 1usize << omega]
    }

    /// Finalize any persistent state after materializing this window (e.g., prepare az/bz layout).
    #[inline]
    fn finalize_materialization_for_window(&mut self, _window_idx: usize, _omega: usize) {
        // Stub: no-op by default.
    }

    /// Produce the 3 evaluations for the current round from the window grid.
    ///
    /// Stubbed for now: replace with logic that reads the first-dimension triples
    /// of `self.window_poly` and combines with `previous_claim` and split-eq weights.
    #[inline]
    fn get_prover_message_from_window_evals(&self, _previous_claim: F) -> Vec<F> {
        // Stub: return zeros; wire with actual computation later.
        vec![F::zero(), F::zero(), F::zero()]
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

    #[tracing::instrument(
        skip_all,
        name = "OuterRemainingStreamingSumcheckProver::compute_prover_message"
    )]
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        // Pseudo-code: (DO NOT DELETE, IMPLEMENT BELOW THIS COMMENT)
        // 1. Check whether this round is the first in the next window (using the window schedule)
        //   - If so, we need to compute the evaluations needed for this window.
        //     - We call the auxiliary function `compute_window_evals_for_new_window` to do this.
        // 2. Now that the bound window evals are guaranteed to exist, we will use that to compute the evaluations for this round
        //    - We call an auxiliary function `get_prover_message_from_window_evals` to do this,
        //      which will return the three evaluations for the current round (e.g., this needs to generalize the `gruen_evals_deg_3` function)

        // If this is a window start, materialize the window grid
        if self.schedule.is_window_start(round) {
            self.compute_window_evals_for_new_window(round);
        }
        // Use the window grid to compute this round's prover message (stubbed)
        self.get_prover_message_from_window_evals(previous_claim)
        
        // Old stuff, commetned out
        // let (t0, t_inf) = if round == 0 {
        //     let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
        //         F::Challenge,
        //         OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        //     >(&self.params.r0_uniskip);
        //     let (t0, t_inf, az_bound, bz_bound) =
        //         Self::compute_first_quadratic_evals_and_bound_polys(
        //             &self.bytecode_preprocessing,
        //             &self.trace,
        //             &lagrange_evals_r,
        //             &self.split_eq_poly,
        //         );
        //     self.az = az_bound;
        //     self.bz = bz_bound;
        //     (t0, t_inf)
        // } else {
        //     self.remaining_quadratic_evals()
        // };
        // let evals = self
        //     .split_eq_poly
        //     .gruen_evals_deg_3(t0, t_inf, previous_claim);
        // vec![evals[0], evals[1], evals[2]]
    }

    #[tracing::instrument(skip_all, name = "OuterRemainingStreamingSumcheckProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        // Pseudo-code: (DO NOT DELETE, IMPLEMENT BELOW THIS COMMENT)
        // 1. Ingest challenge
        // 2. If this round is in the middle of a window (i.e. if it's [a,b)), then
        //   - Bind the multi-quadratic evals
        // 3. If az & bz are already materialized (e.g. not `None`), bind them
        // 4. Bind the eq_poly for next round

        // Ingest challenge
        self.received_challenges.push(r_j);
        // If materialized, apply binding (no-op if empty)
        rayon::join(
            || {
                if let Some(az) = self.az.as_mut() {
                    az.bind_parallel(r_j, BindingOrder::LowToHigh)
                }
            },
            || {
                if let Some(bz) = self.bz.as_mut() {
                    bz.bind_parallel(r_j, BindingOrder::LowToHigh)
                }
            },
        );
        self.split_eq_poly.bind(r_j);
        // If we have a window grid, collapse along the first (LSD) dimension using degree-2 Lagrange basis on {0,1,∞}.
        // TODO: think about encapsulating this better, expose a `bind` method for `w`, only pass in `r`
        if let Some(w) = self.window_poly.as_mut() {
            // Only collapse during the interior of the window; at the next window start a new grid is built.
            if !self.schedule.is_window_start(round) {
                let r: F = r_j.into();
                let l0 = F::one() - r;
                let l1 = r;
                let linf = r * r - r;
                w.collapse_first_dim_in_place(|(q0, q1, qinf)| l0 * q0 + l1 * q1 + linf * qinf);
            }
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

/// Window schedule for streaming prover: exclusive end indices per window.
/// Also stores the round index at which to materialize the window (must be one of the ends)
/// For example, with `ends = [2, 5, 8]`, windows are [0,2), [2,5), [5,8).
/// If `materialize_round = 5`, the window is materialized at round 5.
#[derive(Allocative)]
struct WindowSchedule {
    ends: Vec<usize>,
    materialize_round: usize,
}

impl WindowSchedule {
    /// Construct a schedule where each window contains exactly one round,
    /// and materialization happens right away.
    /// This gives the linear-time sum-check behavior.
    fn single_rounds(num_cycles_bits: usize) -> Self {
        let mut ends = Vec::with_capacity(num_cycles_bits);
        for i in 1..=num_cycles_bits {
            ends.push(i);
        }
        let sched = Self {
            ends,
            materialize_round: 0,
        };
        #[cfg(test)]
        {
            sched.validate(num_cycles_bits);
        }
        sched
    }

    /// Validate invariants under debug builds.
    #[cfg(test)]
    fn validate(&self, num_cycles_bits: usize) {
        if self.ends.is_empty() {
            debug_assert!(
                num_cycles_bits == 0,
                "empty schedule only valid when num_cycles_bits == 0"
            );
            return;
        }
        // Strictly increasing and ends last matches num_cycles_bits
        let mut prev = 0usize;
        for (i, &e) in self.ends.iter().enumerate() {
            debug_assert!(
                e > prev,
                "WindowSchedule.ends must be strictly increasing at index {}",
                i
            );
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

    /// Return window length ω for the window containing `round_in_instance`.
    #[allow(dead_code)]
    fn window_length(&self, round_in_instance: usize) -> usize {
        let s = Self::schedule_round_index(round_in_instance);
        let (start, end) = self.window_bounds_for_schedule_round(s);
        end - start
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
        let idx = self.window_index(sched_round);
        self.window_bounds(idx)
    }

    /// Return true if `round_in_instance` is the first round of its window.
    #[allow(dead_code)]
    fn is_window_start(&self, round_in_instance: usize) -> bool {
        let s = Self::schedule_round_index(round_in_instance);
        let (start, _) = self.window_bounds_for_schedule_round(s);
        s == start
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
