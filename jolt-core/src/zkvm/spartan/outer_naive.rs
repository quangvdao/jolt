#![cfg(feature = "prover")]
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, unipoly::UniPoly};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;

use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::r1cs::constraints::R1CSConstraint;
use crate::zkvm::r1cs::evaluation::{BaselineConstraintEval, R1CSEval};
use crate::zkvm::r1cs::inputs::{R1CSCycleInputs, ALL_R1CS_INPUTS};
use crate::zkvm::witness::VirtualPolynomial;
use crate::{field::JoltField, transcripts::Transcript, utils::math::Math};
use allocative::Allocative;
use rayon::prelude::*;
use std::sync::Arc;
use tracer::instruction::Cycle;

// =======================
// SumcheckInstance (Prover) for outer_linear_time (no uni-skip, no SVO)
// =======================

#[derive(Allocative)]
pub struct OuterNaiveSumcheckProver<F: JoltField> {
    /// Bytecode preprocessing (for witness evaluation at r_cycle)
    #[allocative(skip)]
    bytecode_preprocessing: BytecodePreprocessing,
    /// Full execution trace (for witness evaluation at r_cycle)
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    /// Dense equality polynomial eq(τ, ·) over (cycle, constraint) variables
    eq: DensePolynomial<F>,
    /// Dense Az, Bz multilinear polynomials over (cycle, constraint) variables
    az: DensePolynomial<F>,
    bz: DensePolynomial<F>,
    /// Total rounds = step_vars + constraint_vars
    total_rounds: usize,
    /// Number of step/cycle variables (used to split r_cycle from full opening point)
    num_step_vars: usize,
}

impl<F: JoltField> OuterNaiveSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "OuterNaiveSumcheckProver::gen")]
    pub fn gen<ProofTranscript: Transcript>(
        bytecode_preprocessing: &BytecodePreprocessing,
        trace: Arc<Vec<Cycle>>,
        uniform_constraints: &[R1CSConstraint],
        padded_num_constraints: usize,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let num_steps = trace.len();
        let num_step_vars = if num_steps > 0 { num_steps.log_2() } else { 0 };
        let num_constraint_vars = if padded_num_constraints > 0 {
            padded_num_constraints.log_2()
        } else {
            0
        };
        let total_num_vars = num_step_vars + num_constraint_vars;

        // Sample τ for entire outer sumcheck (no uni-skip)
        let tau: Vec<F::Challenge> = transcript.challenge_vector_optimized::<F>(total_num_vars);

        // Build dense eq(τ, ·) polynomial over (cycle, constraint) Boolean cube.
        let eq_evals: Vec<F> = EqPolynomial::<F>::evals(&tau);
        let eq = DensePolynomial::new(eq_evals);

        // Build dense Az, Bz polynomials over the same domain (cycle, constraint).
        let (az, bz) = Self::build_dense_polynomials(
            bytecode_preprocessing,
            &trace,
            uniform_constraints,
            num_step_vars,
            num_constraint_vars,
            padded_num_constraints,
        );

        Self {
            bytecode_preprocessing: bytecode_preprocessing.clone(),
            trace,
            eq,
            az,
            bz,
            total_rounds: total_num_vars,
            num_step_vars,
        }
    }

    /// Build dense Az, Bz polynomials over (cycle, constraint) variables by streaming over the trace.
    ///
    /// Layout matches the baseline implementation:
    /// - High bits index the cycle/step.
    /// - Low bits index the constraint.
    fn build_dense_polynomials(
        bytecode_preprocessing: &BytecodePreprocessing,
        trace: &[Cycle],
        uniform_constraints: &[R1CSConstraint],
        num_step_vars: usize,
        num_constraint_vars: usize,
        padded_num_constraints: usize,
    ) -> (DensePolynomial<F>, DensePolynomial<F>) {
        let num_steps = trace.len();
        let num_cycles_padded = if num_step_vars == 0 {
            1usize
        } else {
            1usize << num_step_vars
        };

        let domain_size = num_cycles_padded
            .checked_mul(padded_num_constraints)
            .expect("overflow computing naive outer domain size");

        debug_assert!(
            num_steps <= num_cycles_padded,
            "trace length ({num_steps}) must be <= padded cycles ({num_cycles_padded})"
        );

        let total_vars = num_step_vars + num_constraint_vars;
        debug_assert_eq!(
            domain_size,
            1usize << total_vars,
            "naive outer: domain_size != 2^{total_vars}"
        );

        let mut az_vals = unsafe_allocate_zero_vec(domain_size);
        let mut bz_vals = unsafe_allocate_zero_vec(domain_size);

        az_vals
            .par_iter_mut()
            .zip(bz_vals.par_iter_mut())
            .enumerate()
            .for_each(|(d, (az_ref, bz_ref))| {
                let step_idx = d / padded_num_constraints;
                let constraint_idx = d % padded_num_constraints;

                if step_idx < num_steps && constraint_idx < uniform_constraints.len() {
                    let row =
                        R1CSCycleInputs::from_trace::<F>(bytecode_preprocessing, trace, step_idx);
                    let cons = &uniform_constraints[constraint_idx];
                    *az_ref = BaselineConstraintEval::eval_az(cons, &row);
                    *bz_ref = BaselineConstraintEval::eval_bz(cons, &row);
                } else {
                    *az_ref = F::zero();
                    *bz_ref = F::zero();
                }
            });

        (DensePolynomial::new(az_vals), DensePolynomial::new(bz_vals))
    }

    /// Naively compute evaluations of the current round sumcheck polynomial
    ///   g(t) = Σ_x eq(r_1,..,r_{j-1}, t, x) · Az(r_1,..,r_{j-1}, t, x) · Bz(...)
    /// at t ∈ {0, 1, 2, 3}, using only the current dense coeffs and the linearity
    /// of each multilinear polynomial in the current variable (no cloning).
    ///
    /// For any multilinear f in this variable, with f(0) = low, f(1) = high:
    ///   diff = high - low,
    ///   f(2) = f(1) + diff,
    ///   f(3) = f(2) + diff.
    fn naive_round_evals(&self) -> [F; 4] {
        let len = self.eq.len();
        debug_assert_eq!(len, self.az.len());
        debug_assert_eq!(len, self.bz.len());
        debug_assert!(len >= 2 && len.is_power_of_two());

        let num_pairs = len / 2;

        let (g0, g1, g2, g3) = (0..num_pairs)
            .into_par_iter()
            .map(|j| {
                let i0 = 2 * j;
                let i1 = i0 + 1;

                let eq0 = self.eq[i0];
                let eq1 = self.eq[i1];
                let az0 = self.az[i0];
                let az1 = self.az[i1];
                let bz0 = self.bz[i0];
                let bz1 = self.bz[i1];

                let eq_diff = eq1 - eq0;
                let az_diff = az1 - az0;
                let bz_diff = bz1 - bz0;

                // t = 0
                let g0 = eq0 * az0 * bz0;
                // t = 1
                let g1 = eq1 * az1 * bz1;

                // t = 2: f(2) = f(1) + diff
                let eq2 = eq1 + eq_diff;
                let az2 = az1 + az_diff;
                let bz2 = bz1 + bz_diff;
                let g2 = eq2 * az2 * bz2;

                // t = 3: f(3) = f(2) + diff
                let eq3 = eq2 + eq_diff;
                let az3 = az2 + az_diff;
                let bz3 = bz2 + bz_diff;
                let g3 = eq3 * az3 * bz3;

                (g0, g1, g2, g3)
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero(), F::zero()),
                |(a0, a1, a2, a3), (b0, b1, b2, b3)| (a0 + b0, a1 + b1, a2 + b2, a3 + b3),
            );

        [g0, g1, g2, g3]
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for OuterNaiveSumcheckProver<F> {
    fn degree(&self) -> usize {
        3
    }
    fn num_rounds(&self) -> usize {
        self.total_rounds
    }
    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "OuterNaiveSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        // Naive version: directly evaluate g(t) at t = 0,1,2,3 using dense eq, Az, Bz,
        // exploiting linearity in the current variable (no split-eq shortcuts).
        let evals = self.naive_round_evals();
        debug_assert_eq!(evals.len(), 4);
        // Sanity: previous_claim should equal g(0) + g(1); we trust the caller and
        // reconstruct the cubic from the four explicit evaluations.
        UniPoly::from_evals(&evals)
    }

    #[tracing::instrument(skip_all, name = "OuterNaiveSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        // Bind eq, Az, Bz in lockstep (standard dense-poly binding).
        self.eq.bind_parallel(r_j, BindingOrder::LowToHigh);
        rayon::join(
            || self.az.bind_parallel(r_j, BindingOrder::LowToHigh),
            || self.bz.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Opening point uses the sumcheck challenges; endianness matched by OpeningPoint impl
        let opening_point: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();

        // Witness openings at r_cycle: stream from trace and evaluate at r_cycle
        let (r_cycle, _rx_var) = opening_point.r.split_at(self.num_step_vars);
        let r_cycle_point: OpeningPoint<BIG_ENDIAN, F> = OpeningPoint::new(r_cycle.to_vec());
        let claimed_witness_evals = R1CSEval::compute_claimed_inputs_naive(
            &self.bytecode_preprocessing,
            &self.trace,
            &r_cycle_point,
        );

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
}
