#![allow(clippy::too_many_arguments)]
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::{
    dense_mlpoly::DensePolynomial, split_eq_poly::GruenSplitEqPolynomial, unipoly::UniPoly,
};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;

use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::r1cs::evaluation::{BaselineConstraintEval, R1CSEval};
use crate::zkvm::r1cs::inputs::{R1CSCycleInputs, ALL_R1CS_INPUTS};
use crate::zkvm::witness::VirtualPolynomial;
use crate::{
    field::JoltField,
    transcripts::Transcript,
    utils::math::Math,
    zkvm::r1cs::constraints::R1CSConstraint,
};
use num_traits::Zero;
use allocative::Allocative;
use rayon::prelude::*;
use std::sync::Arc;
use tracer::instruction::Cycle;

/// Split-eq + delayed reduction prover for the Spartan outer sumcheck.
///
/// Identical to `OuterSplitEqSumcheckProver` except the endpoint computation
/// uses `mul_unreduced` / `from_montgomery_reduce` to defer modular reduction
/// across the parallel sum, saving one Montgomery reduction per summand.
#[derive(Allocative)]
pub struct OuterDelayedReductionSumcheckProver<F: JoltField> {
    #[allocative(skip)]
    bytecode_preprocessing: BytecodePreprocessing,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    eq_poly: GruenSplitEqPolynomial<F>,
    az: DensePolynomial<F>,
    bz: DensePolynomial<F>,
    total_rounds: usize,
    num_step_vars: usize,
    #[allocative(skip)]
    uniform_constraints: Vec<R1CSConstraint>,
    padded_num_constraints: usize,
}

impl<F: JoltField> OuterDelayedReductionSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "OuterDelayedReductionSumcheckProver::gen")]
    pub fn gen<ProofTranscript: Transcript>(
        bytecode_preprocessing: &BytecodePreprocessing,
        trace: Arc<Vec<Cycle>>,
        uniform_constraints: &[R1CSConstraint],
        padded_num_constraints: usize,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let num_steps = trace.len();
        let num_step_vars = if num_steps > 0 { num_steps.log_2() } else { 0 };
        let num_constraint_vars = padded_num_constraints.max(1).log_2();
        let total_num_vars = num_step_vars + num_constraint_vars;

        let tau: Vec<F::Challenge> = transcript.challenge_vector_optimized::<F>(total_num_vars);
        let eq_poly = GruenSplitEqPolynomial::new(&tau, BindingOrder::LowToHigh);

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
            eq_poly,
            az,
            bz,
            total_rounds: total_num_vars,
            num_step_vars,
            uniform_constraints: uniform_constraints.to_vec(),
            padded_num_constraints,
        }
    }

    fn build_dense_polynomials(
        bytecode_preprocessing: &BytecodePreprocessing,
        trace: &[Cycle],
        uniform_constraints: &[R1CSConstraint],
        num_step_vars: usize,
        _num_constraint_vars: usize,
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
            .expect("overflow computing outer domain size");

        let mut az_vals = vec![F::zero(); domain_size];
        let mut bz_vals = vec![F::zero(); domain_size];

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
                }
            });

        (DensePolynomial::new(az_vals), DensePolynomial::new(bz_vals))
    }

    fn compute_endpoints_delayed(&self) -> (F, F) {
        let eq_poly = &self.eq_poly;
        let n = self.az.len();
        debug_assert_eq!(n, self.bz.len());

        if eq_poly.E_in_current_len() == 1 {
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
                    (eq.mul_unreduced::<9>(p0), eq.mul_unreduced::<9>(slope))
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
                    (
                        e_out.mul_unreduced::<9>(inner0_red),
                        e_out.mul_unreduced::<9>(inner_inf_red),
                    )
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
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for OuterDelayedReductionSumcheckProver<F>
{
    fn degree(&self) -> usize {
        3
    }
    fn num_rounds(&self) -> usize {
        self.total_rounds
    }
    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        F::zero()
    }

    #[tracing::instrument(
        skip_all,
        name = "OuterDelayedReductionSumcheckProver::compute_message"
    )]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let (t0, tinf) = self.compute_endpoints_delayed();
        self.eq_poly.gruen_poly_deg_3(t0, tinf, previous_claim)
    }

    #[tracing::instrument(
        skip_all,
        name = "OuterDelayedReductionSumcheckProver::ingest_challenge"
    )]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        rayon::join(
            || self.az.bind_parallel(r_j, BindingOrder::LowToHigh),
            || self.bz.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
        self.eq_poly.bind(r_j);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();

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

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, _flamegraph: &mut allocative::FlameGraphBuilder) {}
}
