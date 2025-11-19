//! Streaming Spartan outer sumcheck (design notes).
//!
//! This module currently implements `OuterRemainingStreamingSumcheckProver` as a thin
//! wrapper around the canonical `OuterRemainingSumcheckProver`, so the behavior is
//! identical to the non‑streaming outer.  The long‑term goal is to replace this
//! wrapper with a true streaming implementation whose design is sketched below.
//!
//! ---
//!
//! One algorithm to rule them all: arbitrary bind & eval next rounds, taking source from
//! either original trace or already‑bound poly evals.  The plan is:
//!
//! - `bind` should:
//!   - ingest the new challenge and store it in `received_challenges`,
//!   - bind the current window evals to that challenge (window evals are guaranteed to
//!     exist by the time this is called, via a previous call to `compute_message`).
//! - `compute_message` does the real work:
//!   1. Determine if we are at the start of a new window (using a `WindowSchedule`).
//!   2. If so, compute all evaluations needed for this window:
//!      - If the window is in *materialized* mode, reuse already‑bound Az/Bz polys.
//!      - Otherwise, stream each bound eval from the trace, optionally storing them
//!        into backing storage if the schedule says to materialize this window.
//!      - In either case, once we have `2^ω` points for the window, extrapolate to the
//!        `{0,1,∞}^ω` grid, multiply Az·Bz·Eq, and accumulate into the window grid.
//!   3. Once the window grid is ready, compute this round’s cubic from the first‑dim
//!      triples `[Q(0), Q(1), Q(∞)]` and the previous claim (generalizing
//!      `GruenSplitEqPolynomial::gruen_poly_deg_3`).
//!
//! More concretely, `compute_window_evals_for_new_window` is meant to:
//!
//! 1. Check two flags, consistent with the schedule:
//!    - whether dense Az / Bz are already present (from a prior materialized window),
//!    - if not, whether the current window *should* materialize Az/Bz for reuse.
//! 2. Look at the window schedule to determine the active window `[start,end)` and
//!    derive its length `ω = end − start`.  This also determines how we split
//!    `E_out` and `E_in` into:
//!    - static “outer” variables `x_out`,
//!    - static “inner” variables `x_in`,
//!    - active window variables `X_active` (the ω variables we’re about to bind),
//!    - already‑bound variables `r_bound` (which live in `received_challenges` and
//!      the univariate‑skip state).
//! 3. For each group `(x_out, x_in)`:
//!    - Iterate over all `x_active_base ∈ {0,1}^ω` to obtain
//!      `{Az,Bz}(x_out, x_in, x_active_base, r_bound)`.
//!      - If Az/Bz are already materialized, read this directly from `self.az/self.bz`.
//!      - Otherwise, stream from the trace by iterating over the appropriate slice
//!        of row indices and calling `R1CSEval` at the univariate‑skip window, then
//!        optionally store those evals into `self.az/self.bz` if this window is to
//!        be materialized.
//!    - With all boolean‑cube evals for this group in hand, run an in‑place expansion
//!      `{0,1}^ω → {0,1,∞}^ω`:
//!         - start from `a_bool,b_bool,w_bool` over `{0,1}^ω`,
//!         - at each dimension, map pairs `(a0,a1),(b0,b1),(w0,w1)` to
//!           `[a0, a1, a1−a0]`, `[b0, b1, b1−b0]`, `[w0, w1, w0+w1]`,
//!           optionally zeroing out the “1” column for space‑saving,
//!         - after ω steps, this yields `A(t),B(t),W(t)` over `{0,1,∞}^ω`.
//!    - Multiply and accumulate into the window grid:
//!      `Q(t) += E_out(x_out) · E_in(x_in) · A(t) · B(t) · W(t)`, with accumulation
//!      done in typed unreduced form and only reduced once per grid entry.
//! 4. After the parallel fold over all `(x_out,x_in)`, reduce the unreduced grid and
//!    store it as `MultiQuadraticPolynomial { evals, num_vars = ω }`, where the evals
//!    live in base‑3 colex order with `X_1` as the least‑significant trit.  This
//!    layout ensures that each `[Q(0),Q(1),Q(∞)]` triple along the first dimension
//!    is contiguous and can be collapsed or inspected in stride‑1 fashion.
//! 5. If the schedule says this window is materialized, reshape the boolean‑cube
//!    Az/Bz evals for all groups into dense `DensePolynomial`s and stash them in
//!    `self.az/self.bz` for subsequent rounds.
//!
//! In `compute_message`, the intended flow is:
//!
//! 1. If `round` is the first round of a new window (`schedule.is_window_start`),
//!    call `compute_window_evals_for_new_window(round)` to (re)build `window_poly`.
//! 2. Extract this round’s three evaluations from `window_poly`:
//!    - treat `window_poly.evals` as triples `[q0,q1,q∞]` along `X_1`,
//!    - form a degree‑2 `q(X)` at this round using `q0` and `q∞` (and implicitly `q1`
//!      via the previous‑round claim, as in `gruen_poly_deg_3`),
//!    - feed those into an outer linear‐eq factor `l(X)` to obtain a cubic `s(X)`.
//! 3. Return `UniPoly::from_evals_and_hint(previous_claim, &evals[..3])`, where
//!    `evals` are the three evaluations of `s(X)` at `{0,2,3}` needed for the
//!    interpolation scheme used in this codebase.
//!
//! In `ingest_challenge`, the design is:
//!
//! 1. Append `r_j` to `received_challenges`.
//! 2. If `self.az/self.bz` are materialized, bind them in place using the standard
//!    `DensePolynomial::bind_parallel` with `BindingOrder::LowToHigh`.
//! 3. Bind the split eq polynomial `split_eq_poly` for use in the next round.
//! 4. If `window_poly` exists and `round` is *not* the start of a new window, collapse
//!    its first dimension in place using the degree‑2 Lagrange basis for {0,1,∞}:
//!       `L0(r) = 1 − r`, `L1(r) = r`, `L∞(r) = r^2 − r`,
//!    i.e. update each triple `(q0,q1,q∞)` to `L0(r)*q0 + L1(r)*q1 + L∞(r)*q∞`, then
//!    shrink the eval vector from length `3^ω` to `3^{ω−1}`.  This makes `X_2`
//!    become the new least‑significant trit, preserving the triple structure for the
//!    next round without any transposes.
//!
//! The implementation below currently does **not** implement this full streaming
//! behavior; instead it simply forwards to `OuterRemainingSumcheckProver` for
//! correctness.  The comments above are preserved as a design document for a
//! future, fully‑wired streaming implementation.

use allocative::Allocative;
use std::sync::Arc;
use tracer::instruction::Cycle;

use crate::field::JoltField;
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::univariate_skip::UniSkipState;
use crate::transcripts::Transcript;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::spartan::outer::OuterRemainingSumcheckProver;

/// Degree bound of the sumcheck round polynomials for the streaming wrapper.
const OUTER_REMAINING_DEGREE_BOUND: usize = 3;

/// Wrapper prover for a "streaming" outer sumcheck.
///
/// This implementation currently delegates all logic to the canonical
/// `OuterRemainingSumcheckProver`, so it is functionally identical to the
/// non-streaming outer.  It exists as a hook for future streaming
/// optimizations while preserving the same correctness condition and API
/// surface as the canonical outer.
#[derive(Allocative)]
pub struct OuterRemainingStreamingSumcheckProver<F: JoltField> {
    #[allocative(skip)]
    inner: OuterRemainingSumcheckProver<F>,
}

impl<F: JoltField> OuterRemainingStreamingSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "OuterRemainingStreamingSumcheckProver::gen")]
    pub fn gen(
        trace: Arc<Vec<Cycle>>,
        bytecode_preprocessing: &BytecodePreprocessing,
        num_cycles_bits: usize,
        uni: &UniSkipState<F>,
    ) -> Self {
        // The canonical outer determines `num_cycles_bits` from the trace length;
        // keep a debug check here to ensure the caller-supplied value stays in sync.
        let expected_bits = trace.len().ilog2() as usize;
        debug_assert_eq!(
            num_cycles_bits, expected_bits,
            "streaming outer: num_cycles_bits mismatch (got {}, expected {})",
            num_cycles_bits, expected_bits
        );

        let inner = OuterRemainingSumcheckProver::gen(trace, bytecode_preprocessing, uni);
        Self { inner }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for OuterRemainingStreamingSumcheckProver<F>
{
    fn degree(&self) -> usize {
        OUTER_REMAINING_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        <OuterRemainingSumcheckProver<F> as SumcheckInstanceProver<F, T>>::num_rounds(&self.inner)
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        <OuterRemainingSumcheckProver<F> as SumcheckInstanceProver<F, T>>::input_claim(
            &self.inner,
            accumulator,
        )
    }

    #[tracing::instrument(
        skip_all,
        name = "OuterRemainingStreamingSumcheckProver::compute_message"
    )]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        <OuterRemainingSumcheckProver<F> as SumcheckInstanceProver<F, T>>::compute_message(
            &mut self.inner,
            round,
            previous_claim,
        )
    }

    #[tracing::instrument(skip_all, name = "OuterRemainingStreamingSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        <OuterRemainingSumcheckProver<F> as SumcheckInstanceProver<F, T>>::ingest_challenge(
            &mut self.inner,
            r_j,
            round,
        )
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        <OuterRemainingSumcheckProver<F> as SumcheckInstanceProver<F, T>>::cache_openings(
            &self.inner,
            accumulator,
            transcript,
            sumcheck_challenges,
        )
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}


