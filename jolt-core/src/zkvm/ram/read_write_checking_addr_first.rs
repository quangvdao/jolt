use crate::poly::opening_proof::OpeningAccumulator;

use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::zkvm::ram::sparse_matrix_poly_addr_first::SparseMatrixPolynomialAddrFirst;
use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator, SumcheckId},
        split_eq_poly::GruenSplitEqPolynomial,
    },
    transcripts::Transcript,
    utils::math::Math,
    zkvm::dag::state_manager::StateManager,
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;

/// Degree bound of the sumcheck round polynomials in the address-first RAM read-write checker.
const DEGREE_BOUND: usize = 3;

use super::read_write_checking::ReadWriteCheckingParams;

/// Prover for an address-first variant of the RAM read-write checking sumcheck.
///
/// This implementation is **experimental** and currently only sets up the data
/// structures and plumbing. The actual prover messages and binding logic are
/// left as TODOs and will be filled in incrementally.
#[derive(Allocative)]
pub struct RamReadWriteCheckingProverAddrFirst<F: JoltField> {
    /// Initial memory state, as field elements.
    val_init: Vec<F>,
    /// Sparse representation of ra(k, j) and Val(k, j), suitable for binding
    /// address variables first.
    sparse_matrix: SparseMatrixPolynomialAddrFirst<F>,
    /// Increment polynomial inc(j) over cycle variables only.
    inc: MultilinearPolynomial<F>,
    /// Precomputed table eq(r_cycle_stage_1, j) for all j ∈ {0,1}^log T.
    /// This is used as a per-row weight in the address rounds.
    eq_cycle_evals: Vec<F>,
    /// Dense ra(j) polynomial over cycle variables after all address bits
    /// have been bound and the sparse matrix has been materialized.
    ra_cycles: Option<MultilinearPolynomial<F>>,
    /// Dense Val(j) polynomial over cycle variables, same as above.
    val_cycles: Option<MultilinearPolynomial<F>>,
    /// Gruen split representation of eq(r_cycle_stage_1, j) for the cycle phase.
    eq_cycle_split: GruenSplitEqPolynomial<F>,
    /// Parameters (K, T, gamma, etc.), reused from the cycle-first checker.
    #[allocative(skip)]
    params: ReadWriteCheckingParams<F>,
}

impl<F: JoltField> RamReadWriteCheckingProverAddrFirst<F> {
    #[tracing::instrument(skip_all, name = "RamReadWriteCheckingProverAddrFirst::gen")]
    pub fn gen(
        initial_memory_state: &[u64],
        state_manager: &mut StateManager<'_, F, impl CommitmentScheme<Field = F>>,
        opening_accumulator: &ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let (preprocessing, _, trace, _, _) = state_manager.get_prover_data();

        let params = ReadWriteCheckingParams::new(
            state_manager.ram_K,
            trace.len(),
            opening_accumulator,
            transcript,
        );

        // r_cycle randomness from Spartan outer; used to weight cycle indices
        // by eq(r_cycle, j) in the first phase.
        let eq_cycle_evals = EqPolynomial::evals(&params.r_cycle_stage_1.r);
        let eq_cycle_split =
            GruenSplitEqPolynomial::new(&params.r_cycle_stage_1.r, BindingOrder::LowToHigh);

        let inc =
            CommittedPolynomial::RamInc.generate_witness(preprocessing, trace, state_manager.ram_d);
        let sparse_matrix =
            SparseMatrixPolynomialAddrFirst::new(&trace, &preprocessing.memory_layout);
        let val_init = initial_memory_state
            .par_iter()
            .map(|x| F::from_u64(*x))
            .collect();

        Self {
            val_init,
            sparse_matrix,
            inc,
            eq_cycle_evals,
            ra_cycles: None,
            val_cycles: None,
            eq_cycle_split,
            params,
        }
    }

    /// Phase 1 (address rounds) prover message: bind current address bit only.
    fn phase1_compute_prover_message(
        &mut self,
        _round: usize,
        _previous_claim: F,
    ) -> Vec<F> {
        // Address-binding phase: current variable is an address bit.
        //
        // We keep the address phase purely sparse and independent of the
        // eq(r_cycle, j) weights. At this stage the polynomial being
        // checked is still a function of both address and cycle bits,
        // but eq is only attached in the cycle phase via `eq_cycle_split`.
        //
        // Concretely, we compute a degree-≤2 univariate in the address bit:
        //
        //   g_addr(t) = Σ_j Σ_{address pairs in row j}
        //                   RA_t(j) ⋅ (VAL_t(j) + γ ⋅ (inc(j) + VAL_t(j)))
        //
        // for t ∈ {0, 2, 3}. The dependency on cycle bits (through inc and
        // the row index) is carried symbolically and will be combined with
        // eq(r_cycle, ·) in the later cycle phase.
        //
        // This loop is parallelized over rows; each row produces its local
        // [g(0), g(2), g(3)] triple which we then sum.
        let gamma = self.params.gamma;
        let inc = &self.inc;

        let evals = self
            .sparse_matrix
            .entries
            .par_chunk_by(|a, b| a.row == b.row)
            .map(|row_entries| {
                debug_assert!(!row_entries.is_empty());
                let row = row_entries[0].row;
                let inc_j = inc.get_coeff(row);

                let mut row_evals = [F::zero(); 3];

                let mut i = 0;
                while i < row_entries.len() {
                    let col = row_entries[i].col;
                    let base = col / 2;
                    let even_col = base * 2;
                    let odd_col = even_col + 1;

                    if col == even_col {
                        let ra_even = row_entries[i].ra_coeff;
                        let val_even = row_entries[i].val_coeff;
                        i += 1;
                        let (ra_odd, val_odd) = if i < row_entries.len()
                            && row_entries[i].col == odd_col
                        {
                            let ra = row_entries[i].ra_coeff;
                            let val = row_entries[i].val_coeff;
                            i += 1;
                            (ra, val)
                        } else {
                            // No explicit odd entry: treat ra_odd = 0 and
                            // Val_odd = Val_even. Since ra_odd = 0, the
                            // exact choice of Val_odd does not matter for
                            // correctness of ra ⋅ (Val + γ (inc + Val)).
                            (F::zero(), val_even)
                        };

                        let dra = ra_odd - ra_even;
                        let dval = val_odd - val_even;

                        let ra_0 = ra_even;
                        let val_0 = val_even;
                        let term_0 =
                            ra_0 * (val_0 + gamma * (inc_j + val_0));

                        let ra_2 = ra_even + dra + dra;
                        let val_2 = val_even + dval + dval;
                        let term_2 =
                            ra_2 * (val_2 + gamma * (inc_j + val_2));

                        let ra_3 = ra_2 + dra;
                        let val_3 = val_2 + dval;
                        let term_3 =
                            ra_3 * (val_3 + gamma * (inc_j + val_3));

                        row_evals[0] += term_0;
                        row_evals[1] += term_2;
                        row_evals[2] += term_3;
                    } else {
                        debug_assert_eq!(col, odd_col);
                        let ra_odd = row_entries[i].ra_coeff;
                        let val_odd = row_entries[i].val_coeff;
                        i += 1;

                        let ra_even = F::zero();
                        let val_even = val_odd;

                        let dra = ra_odd - ra_even;
                        let dval = val_odd - val_even;

                        let ra_0 = ra_even;
                        let val_0 = val_even;
                        let term_0 =
                            ra_0 * (val_0 + gamma * (inc_j + val_0));

                        let ra_2 = ra_even + dra + dra;
                        let val_2 = val_even + dval + dval;
                        let term_2 =
                            ra_2 * (val_2 + gamma * (inc_j + val_2));

                        let ra_3 = ra_2 + dra;
                        let val_3 = val_2 + dval;
                        let term_3 =
                            ra_3 * (val_3 + gamma * (inc_j + val_3));

                        row_evals[0] += term_0;
                        row_evals[1] += term_2;
                        row_evals[2] += term_3;
                    }
                }

                row_evals
            })
            .reduce(
                || [F::zero(); 3],
                |mut acc, row_evals| {
                    acc[0] += row_evals[0];
                    acc[1] += row_evals[1];
                    acc[2] += row_evals[2];
                    acc
                },
            );

        evals.to_vec()
    }

    /// Phase 2 (cycle rounds) prover message: work over dense ra/Val/inc with split-eq.
    fn phase2_compute_prover_message(&mut self, previous_claim: F) -> Vec<F> {
        // Cycle-binding phase: use dense ra/Val/inc over cycles together
        // with the Gruen split eq polynomial. We follow the same pattern
        // as `HammingBooleanitySumcheckProver`, computing only the
        // constant and quadratic coefficients of the inner polynomial and
        // letting `gruen_evals_deg_3` handle the cubic lift.
        let ra = self
            .ra_cycles
            .as_ref()
            .expect("ra_cycles must be materialized before cycle phase");
        let val = self
            .val_cycles
            .as_ref()
            .expect("val_cycles must be materialized before cycle phase");

        let gamma = self.params.gamma;

        // Accumulate constant term c0 and quadratic coefficient e of
        // q(X) = Σ_j ra(j, X) * (Val(j, X) + γ (inc(j, X) + Val(j, X))).
        //
        // For each group g we look at cycles (2g, 2g+1) and encode the
        // inner polynomial via evaluations at 0 and ∞ (direction), exactly
        // as in `SparseMatrixPolynomial::prover_message_contribution`.
        let [q_constant, q_quadratic] =
            self.eq_cycle_split
                .par_fold_out_in_unreduced::<9, { DEGREE_BOUND - 1 }>(&|g| {
                    let j0 = 2 * g;
                    let j1 = j0 + 1;

                    let ra0 = ra.get_bound_coeff(j0);
                    let ra1 = ra.get_bound_coeff(j1);
                    let val0 = val.get_bound_coeff(j0);
                    let val1 = val.get_bound_coeff(j1);
                    let inc0 = self.inc.get_bound_coeff(j0);
                    let inc1 = self.inc.get_bound_coeff(j1);

                    // Encode as value at 0 and "direction" at ∞.
                    let ra_evals = [ra0, ra1 - ra0];
                    let val_evals = [val0, val1 - val0];
                    let inc_evals = [inc0, inc1 - inc0];

                    [
                        ra_evals[0]
                            * (val_evals[0]
                                + gamma * (inc_evals[0] + val_evals[0])),
                        ra_evals[1]
                            * (val_evals[1]
                                + gamma * (inc_evals[1] + val_evals[1])),
                    ]
                });

        let evals = self
            .eq_cycle_split
            .gruen_evals_deg_3(q_constant, q_quadratic, previous_claim);
        evals.to_vec()
    }

    /// Phase 1 (address rounds) binding: bind current address bit in the sparse matrix.
    /// On the final address round we also materialize dense `ra(j)` and `Val(j)`
    /// over cycles for use in Phase 2.
    fn phase1_bind(&mut self, r_j: F::Challenge, round: usize) {
        self.sparse_matrix.bind_address_bit(r_j);

        let num_addr_rounds = self.params.K.log_2();
        if round == num_addr_rounds - 1 {
            // All address bits have been bound. At this point each row
            // corresponds to a single virtual address, so we can materialize
            // dense polynomials over cycles.
            debug_assert!(self.ra_cycles.is_none());
            debug_assert!(self.val_cycles.is_none());

            let sparse = std::mem::take(&mut self.sparse_matrix);
            let (ra_ml, val_ml) = sparse.materialize_over_cycles(self.params.T);
            self.ra_cycles = Some(ra_ml);
            self.val_cycles = Some(val_ml);
        }
    }

    /// Phase 2 (cycle rounds) binding: materialize and then bind dense polynomials.
    fn phase2_bind(&mut self, r_j: F::Challenge, _cycle_round: usize) {
        // Bind dense polynomials over cycle variables.
        let ra = self
            .ra_cycles
            .as_mut()
            .expect("ra_cycles must be materialized before cycle binding");
        let val = self
            .val_cycles
            .as_mut()
            .expect("val_cycles must be materialized before cycle binding");

        ra.bind_parallel(r_j, BindingOrder::LowToHigh);
        val.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.inc.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_cycle_split.bind(r_j);
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for RamReadWriteCheckingProverAddrFirst<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    #[tracing::instrument(
        skip_all,
        name = "RamReadWriteCheckingProverAddrFirst::compute_prover_message"
    )]
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        let num_addr_rounds = self.params.K.log_2();

        if round < num_addr_rounds {
            self.phase1_compute_prover_message(round, previous_claim)
        } else {
            self.phase2_compute_prover_message(previous_claim)
        }
    }

    #[tracing::instrument(
        skip_all,
        name = "RamReadWriteCheckingProverAddrFirst::bind"
    )]
    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        let num_addr_rounds = self.params.K.log_2();

        if round < num_addr_rounds {
            self.phase1_bind(r_j, round);
        } else {
            let cycle_round = round - num_addr_rounds;
            self.phase2_bind(r_j, cycle_round);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Mirror the cycle-first checker: at the end of sumcheck we open
        // RamVal and RamRa as virtual polynomials over (address, cycle)
        // and RamInc as a dense committed polynomial over cycles.
        //
        // For the addr-first variant, `ra_cycles` and `val_cycles` are dense
        // polynomials over cycle variables only. After all rounds they have
        // been fully bound, so their final_sumcheck_claims give the values
        // used in the sumcheck polynomial at the final cycle point.
        let opening_point = self
            .params
            .get_opening_point_addr_first(sumcheck_challenges);

        let val_claim = self
            .val_cycles
            .as_ref()
            .expect("val_cycles must be materialized before cache_openings")
            .final_sumcheck_claim();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
            val_claim,
        );

        let ra_claim = self
            .ra_cycles
            .as_ref()
            .expect("ra_cycles must be materialized before cache_openings")
            .final_sumcheck_claim();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
            ra_claim,
        );

        let (_, r_cycle) = opening_point.split_at(self.params.K.log_2());
        let inc_claim = self.inc.final_sumcheck_claim();
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
            r_cycle.r,
            inc_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Verifier for the address-first RAM read-write checking sumcheck.
///
/// This is currently a thin wrapper around `ReadWriteCheckingParams` and
/// mirrors the interface of `RamReadWriteCheckingVerifier`. The expected
/// claim and opening logic will track the cycle-first implementation once
/// the prover logic is fully implemented.
pub struct RamReadWriteCheckingVerifierAddrFirst<F: JoltField> {
    params: ReadWriteCheckingParams<F>,
}

impl<F: JoltField> RamReadWriteCheckingVerifierAddrFirst<F> {
    pub fn new(
        ram_K: usize,
        trace_len: usize,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        Self {
            params: ReadWriteCheckingParams::new(
                ram_K,
                trace_len,
                opening_accumulator,
                transcript,
            ),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RamReadWriteCheckingVerifierAddrFirst<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // For now we mirror the cycle-first verifier's expected output claim:
        //
        //   Σ_{k,j} eq(r_cycle, j) ⋅ ra(k, j) ⋅
        //       (Val(k, j) + γ ⋅ (inc(j) + Val(k, j)))
        //
        // As the address-first prover is completed, this method should remain
        // the same, since reordering sumcheck variables does not change the
        // target polynomial.
        let r = self
            .params
            .get_opening_point_addr_first(sumcheck_challenges);
        let (_, r_cycle) = r.split_at(self.params.K.log_2());

        let eq_eval_cycle = EqPolynomial::mle_endian(&self.params.r_cycle_stage_1, &r_cycle);

        let (_, ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, val_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );

        eq_eval_cycle * ra_claim * (val_claim + self.params.gamma * (val_claim + inc_claim))
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Mirror the cycle-first verifier: we expect RamVal and RamRa as
        // virtual polynomials over (address, cycle), and RamInc as a dense
        // committed polynomial over cycles.
        let opening_point = self
            .params
            .get_opening_point_addr_first(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
        );

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
        );
        let (_, r_cycle) = opening_point.split_at(self.params.K.log_2());
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
            r_cycle.r,
        );
    }
}


