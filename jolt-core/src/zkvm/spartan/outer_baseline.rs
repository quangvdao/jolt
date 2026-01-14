use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::{
    dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial,
    multilinear_polynomial::MultilinearPolynomial, split_eq_poly::GruenSplitEqPolynomial,
    unipoly::UniPoly,
};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;

use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::r1cs::evaluation::{BaselineConstraintEval, R1CSEval};
use crate::zkvm::r1cs::inputs::{R1CSCycleInputs, ALL_R1CS_INPUTS};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::witness::VirtualPolynomial;
use crate::{
    field::{JoltField, OptimizedMul},
    transcripts::Transcript,
    utils::math::Math,
    zkvm::r1cs::constraints::R1CSConstraint,
};
use allocative::Allocative;
use rayon::prelude::*;
use std::sync::Arc;
use tracer::instruction::Cycle;

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct SparseCoefficient<T> {
    pub(crate) index: usize,
    pub(crate) value: T,
}

// =======================o
// SumcheckInstance (Verifier) for baseline outer (no uni-skip)
// =======================
pub struct OuterBaselineSumcheckVerifier<F: JoltField> {
    num_step_bits: usize,
    total_rounds: usize,
    tau: Vec<F::Challenge>,
    key: UniformSpartanKey<F>,
    _phantom: core::marker::PhantomData<F>,
}

impl<F: JoltField> OuterBaselineSumcheckVerifier<F> {
    pub fn new(
        num_step_bits: usize,
        num_constraint_bits: usize,
        tau: Vec<F::Challenge>,
        key: UniformSpartanKey<F>,
    ) -> Self {
        Self {
            num_step_bits,
            total_rounds: num_step_bits + num_constraint_bits,
            tau,
            key,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for OuterBaselineSumcheckVerifier<F>
{
    fn degree(&self) -> usize {
        3
    }
    fn num_rounds(&self) -> usize {
        self.total_rounds
    }
    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        F::zero()
    }
    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Recover all z_i(r_cycle) openings for the Spartan outer instance.
        let r1cs_input_evals = ALL_R1CS_INPUTS.map(|input| {
            accumulator
                .get_virtual_polynomial_opening((&input).into(), SumcheckId::SpartanOuter)
                .1
        });

        // Inner sum-product over R1CS rows at a row-binding point derived from the
        // first two sumcheck challenges.  This matches the layout expected by
        // `UniformSpartanKey::evaluate_inner_sum_product_at_point`.
        debug_assert!(
            sumcheck_challenges.len() >= 2,
            "baseline outer: expected at least two challenges for row binding"
        );
        let rx_constr = &[sumcheck_challenges[0], sumcheck_challenges[1]];
        let inner_sum_prod = self
            .key
            .evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals);

        // Full Eq kernel over the (cycle,constraint) variables, using the same
        // τ vector that parameterizes the baseline prover's equality polynomial.
        let r_rev: Vec<F::Challenge> = sumcheck_challenges.iter().rev().copied().collect();
        let eq_tau_r = EqPolynomial::<F>::mle(&self.tau, &r_rev);

        eq_tau_r * inner_sum_prod
    }
    fn cache_openings(
        &self,
        accumulator: &mut crate::poly::opening_proof::VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();

        // Witness openings at r_cycle
        let (r_cycle, _rx_var) = opening_point.r.split_at(self.num_step_bits);
        ALL_R1CS_INPUTS.iter().for_each(|input| {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                OpeningPoint::new(r_cycle.to_vec()),
            );
        });
    }
}

impl<T> Allocative for SparseCoefficient<T> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<T> From<(usize, T)> for SparseCoefficient<T> {
    fn from(x: (usize, T)) -> Self {
        Self {
            index: x.0,
            value: x.1,
        }
    }
}

// Baseline: Original-style interleaved polynomial (single Vec storage)
#[derive(Default, Debug, Clone, Allocative)]
pub struct BaselineSpartanInterleavedPolynomial<F: JoltField> {
    pub(crate) unbound_coeffs: Vec<SparseCoefficient<F>>,
    pub(crate) bound_coeffs: Vec<SparseCoefficient<F>>,
    binding_scratch_space: Vec<SparseCoefficient<F>>,
}

impl<F: JoltField> BaselineSpartanInterleavedPolynomial<F> {
    /// Compute endpoints (t0, tinf) for the first round from unbound coefficients.
    /// Layout per logical block: [az0, bz0, az1, bz1] (stride = 4).
    pub fn endpoints_unbound_first_round(&self, eq_poly: &GruenSplitEqPolynomial<F>) -> (F, F) {
        let block_size = self
            .unbound_coeffs
            .len()
            .div_ceil(rayon::current_num_threads())
            .next_multiple_of(4);
        let chunks: Vec<_> = self
            .unbound_coeffs
            .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
            .collect();

        if eq_poly.E_in_current_len() == 1 {
            chunks
                .par_iter()
                .flat_map_iter(|chunk| {
                    chunk
                        .chunk_by(|x, y| x.index / 4 == y.index / 4)
                        .map(|sparse_block| {
                            let block_index = sparse_block[0].index / 4;
                            let mut block = [F::zero(); 4];
                            for coeff in sparse_block {
                                block[coeff.index % 4] = coeff.value;
                            }

                            let az0 = block[0];
                            let bz0 = block[1];
                            let az1 = block[2];
                            let bz1 = block[3];

                            let p0 = az0 * bz0;
                            let slope = (az1 - az0) * (bz1 - bz0);
                            let eq = eq_poly.E_out_current()[block_index];
                            (eq.mul_0_optimized(p0), eq.mul_0_optimized(slope))
                        })
                })
                .reduce(
                    || (F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                )
        } else {
            let num_x_in_bits = eq_poly.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;

            chunks
                .par_iter()
                .map(|chunk| {
                    let mut eval_point_0 = F::zero();
                    let mut eval_point_infty = F::zero();

                    let mut inner_sums = (F::zero(), F::zero());
                    let mut prev_x_out = 0;

                    for sparse_block in chunk.chunk_by(|x, y| x.index / 4 == y.index / 4) {
                        let block_index = sparse_block[0].index / 4;
                        let x_in = block_index & x_bitmask;
                        let E_in_eval = eq_poly.E_in_current()[x_in];
                        let x_out = block_index >> num_x_in_bits;

                        if x_out != prev_x_out {
                            let E_out_eval = eq_poly.E_out_current()[prev_x_out];
                            eval_point_0 += E_out_eval * inner_sums.0;
                            eval_point_infty += E_out_eval * inner_sums.1;

                            inner_sums = (F::zero(), F::zero());
                            prev_x_out = x_out;
                        }

                        let mut block = [F::zero(); 4];
                        for coeff in sparse_block {
                            block[coeff.index % 4] = coeff.value;
                        }

                        let az0 = block[0];
                        let bz0 = block[1];
                        let az1 = block[2];
                        let bz1 = block[3];

                        inner_sums.0 += E_in_eval.mul_0_optimized(az0 * bz0);
                        inner_sums.1 += E_in_eval.mul_0_optimized((az1 - az0) * (bz1 - bz0));
                    }

                    eval_point_0 += eq_poly.E_out_current()[prev_x_out] * inner_sums.0;
                    eval_point_infty += eq_poly.E_out_current()[prev_x_out] * inner_sums.1;

                    (eval_point_0, eval_point_infty)
                })
                .reduce(
                    || (F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                )
        }
    }

    /// Compute endpoints (t0, tinf) for remaining rounds from bound coefficients.
    pub fn endpoints_bound_remaining(&self, eq_poly: &GruenSplitEqPolynomial<F>) -> (F, F) {
        let block_size = self
            .bound_coeffs
            .len()
            .div_ceil(rayon::current_num_threads())
            .next_multiple_of(4);
        let chunks: Vec<_> = self
            .bound_coeffs
            .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
            .collect();

        if eq_poly.E_in_current_len() == 1 {
            chunks
                .par_iter()
                .flat_map_iter(|chunk| {
                    chunk
                        .chunk_by(|x, y| x.index / 4 == y.index / 4)
                        .map(|sparse_block| {
                            let block_index = sparse_block[0].index / 4;
                            let mut block = [F::zero(); 4];
                            for coeff in sparse_block {
                                block[coeff.index % 4] = coeff.value;
                            }

                            let az0 = block[0];
                            let bz0 = block[1];
                            let az1 = block[2];
                            let bz1 = block[3];

                            let az_eval_infty = az1 - az0;
                            let bz_eval_infty = bz1 - bz0;

                            let eq_evals = eq_poly.E_out_current()[block_index];

                            (
                                eq_evals.mul_0_optimized(az0.mul_0_optimized(bz0)),
                                eq_evals
                                    .mul_0_optimized(az_eval_infty.mul_0_optimized(bz_eval_infty)),
                            )
                        })
                })
                .reduce(
                    || (F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                )
        } else {
            let num_x_in_bits = eq_poly.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;

            chunks
                .par_iter()
                .map(|chunk| {
                    let mut eval_point_0 = F::zero();
                    let mut eval_point_infty = F::zero();

                    let mut inner_sums = (F::zero(), F::zero());
                    let mut prev_x_out = 0;

                    for sparse_block in chunk.chunk_by(|x, y| x.index / 4 == y.index / 4) {
                        let block_index = sparse_block[0].index / 4;
                        let x_in = block_index & x_bitmask;
                        let E_in_eval = eq_poly.E_in_current()[x_in];
                        let x_out = block_index >> num_x_in_bits;

                        if x_out != prev_x_out {
                            let E_out_eval = eq_poly.E_out_current()[prev_x_out];
                            eval_point_0 += E_out_eval * inner_sums.0;
                            eval_point_infty += E_out_eval * inner_sums.1;

                            inner_sums = (F::zero(), F::zero());
                            prev_x_out = x_out;
                        }

                        let mut block = [F::zero(); 4];
                        for coeff in sparse_block {
                            block[coeff.index % 4] = coeff.value;
                        }

                        let az0 = block[0];
                        let bz0 = block[1];
                        let az1 = block[2];
                        let bz1 = block[3];

                        let az_eval_infty = az1 - az0;
                        let bz_eval_infty = bz1 - bz0;

                        inner_sums.0 += E_in_eval.mul_0_optimized(az0.mul_0_optimized(bz0));
                        inner_sums.1 +=
                            E_in_eval.mul_0_optimized(az_eval_infty.mul_0_optimized(bz_eval_infty));
                    }

                    eval_point_0 += eq_poly.E_out_current()[prev_x_out] * inner_sums.0;
                    eval_point_infty += eq_poly.E_out_current()[prev_x_out] * inner_sums.1;

                    (eval_point_0, eval_point_infty)
                })
                .reduce(
                    || (F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1),
                )
        }
    }

    /// Bind coefficients in-place by r.
    /// If `unbound_coeffs` is non-empty, performs first-round binding unbound->bound; otherwise binds bound coeffs.
    pub fn bind_inplace(&mut self, r: F::Challenge) {
        if !self.is_bound() {
            // First round binding from unbound -> bound
            let block_size = self
                .unbound_coeffs
                .len()
                .div_ceil(rayon::current_num_threads())
                .next_multiple_of(4);
            let chunks: Vec<_> = self
                .unbound_coeffs
                .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
                .collect();

            let output_sizes: Vec<_> = chunks
                .par_iter()
                .map(|chunk| Self::binding_output_length(chunk))
                .collect();

            let total_output_len: usize = output_sizes.iter().sum();
            self.bound_coeffs = vec![
                SparseCoefficient {
                    index: 0,
                    value: F::zero()
                };
                total_output_len
            ];

            let mut output_slices: Vec<&mut [SparseCoefficient<F>]> =
                Vec::with_capacity(chunks.len());
            let mut remainder = self.bound_coeffs.as_mut_slice();
            for slice_len in output_sizes {
                let (first, second) = remainder.split_at_mut(slice_len);
                output_slices.push(first);
                remainder = second;
            }
            debug_assert_eq!(remainder.len(), 0);

            chunks
                .par_iter()
                .zip_eq(output_slices.into_par_iter())
                .for_each(|(unbound_coeffs, output_slice)| {
                    let mut output_index = 0;
                    for block in unbound_coeffs.chunk_by(|x, y| x.index / 4 == y.index / 4) {
                        let block_index = block[0].index / 4;

                        let mut az_coeff: (Option<F>, Option<F>) = (None, None);
                        let mut bz_coeff: (Option<F>, Option<F>) = (None, None);

                        for coeff in block {
                            match coeff.index % 4 {
                                0 => az_coeff.0 = Some(coeff.value),
                                1 => bz_coeff.0 = Some(coeff.value),
                                2 => az_coeff.1 = Some(coeff.value),
                                3 => bz_coeff.1 = Some(coeff.value),
                                _ => unreachable!(),
                            }
                        }
                        if az_coeff != (None, None) {
                            let (low, high) = (
                                az_coeff.0.unwrap_or(F::zero()),
                                az_coeff.1.unwrap_or(F::zero()),
                            );
                            output_slice[output_index] =
                                (2 * block_index, low + r * (high - low)).into();
                            output_index += 1;
                        }
                        if bz_coeff != (None, None) {
                            let (low, high) = (
                                bz_coeff.0.unwrap_or(F::zero()),
                                bz_coeff.1.unwrap_or(F::zero()),
                            );
                            output_slice[output_index] =
                                (2 * block_index + 1, low + r * (high - low)).into();
                            output_index += 1;
                        }
                    }
                    debug_assert_eq!(output_index, output_slice.len())
                });

            self.unbound_coeffs.clear();
            self.unbound_coeffs.shrink_to_fit();
        } else {
            // Remaining rounds: bind bound coeffs using r into scratch, then swap
            let block_size = self
                .bound_coeffs
                .len()
                .div_ceil(rayon::current_num_threads())
                .next_multiple_of(4);
            let chunks: Vec<_> = self
                .bound_coeffs
                .par_chunk_by(|x, y| x.index / block_size == y.index / block_size)
                .collect();

            let output_sizes: Vec<_> = chunks
                .par_iter()
                .map(|chunk| Self::binding_output_length(chunk))
                .collect();

            let total_output_len: usize = output_sizes.iter().sum();
            self.binding_scratch_space.resize(
                total_output_len,
                SparseCoefficient {
                    index: 0,
                    value: F::zero(),
                },
            );

            let mut output_slices: Vec<&mut [SparseCoefficient<F>]> =
                Vec::with_capacity(chunks.len());
            let mut remainder = self.binding_scratch_space.as_mut_slice();
            for slice_len in output_sizes {
                let (first, second) = remainder.split_at_mut(slice_len);
                output_slices.push(first);
                remainder = second;
            }
            debug_assert_eq!(remainder.len(), 0);

            chunks
                .par_iter()
                .zip_eq(output_slices.into_par_iter())
                .for_each(|(coeffs, output_slice)| {
                    let mut output_index = 0;
                    for block in coeffs.chunk_by(|x, y| x.index / 4 == y.index / 4) {
                        let block_index = block[0].index / 4;

                        let mut az_coeff: (Option<F>, Option<F>) = (None, None);
                        let mut bz_coeff: (Option<F>, Option<F>) = (None, None);

                        for coeff in block {
                            match coeff.index % 4 {
                                0 => az_coeff.0 = Some(coeff.value),
                                1 => bz_coeff.0 = Some(coeff.value),
                                2 => az_coeff.1 = Some(coeff.value),
                                3 => bz_coeff.1 = Some(coeff.value),
                                _ => unreachable!(),
                            }
                        }
                        if az_coeff != (None, None) {
                            let (low, high) = (
                                az_coeff.0.unwrap_or(F::zero()),
                                az_coeff.1.unwrap_or(F::zero()),
                            );
                            output_slice[output_index] =
                                (2 * block_index, low + r * (high - low)).into();
                            output_index += 1;
                        }
                        if bz_coeff != (None, None) {
                            let (low, high) = (
                                bz_coeff.0.unwrap_or(F::zero()),
                                bz_coeff.1.unwrap_or(F::zero()),
                            );
                            output_slice[output_index] =
                                (2 * block_index + 1, low + r * (high - low)).into();
                            output_index += 1;
                        }
                    }
                    debug_assert_eq!(output_index, output_slice.len())
                });

            std::mem::swap(&mut self.bound_coeffs, &mut self.binding_scratch_space);
        }
    }
    pub fn new(
        uniform_constraints: &[R1CSConstraint],
        flattened_polynomials: &[MultilinearPolynomial<F>],
        padded_num_constraints: usize,
    ) -> Self {
        let num_steps = flattened_polynomials[0].len();
        let num_chunks = rayon::current_num_threads().next_power_of_two() * 4;
        let chunk_size = num_steps.div_ceil(num_chunks);

        let unbound_coeffs: Vec<SparseCoefficient<F>> = (0..num_chunks)
            .into_par_iter()
            .flat_map_iter(|chunk_index| {
                let mut coeffs = Vec::with_capacity(chunk_size * padded_num_constraints * 2);
                let start = chunk_size * chunk_index;
                let end = std::cmp::min(chunk_size * (chunk_index + 1), num_steps);
                for step_index in start..end {
                    for (constraint_index, constraint) in uniform_constraints.iter().enumerate() {
                        let global_index =
                            2 * (step_index * padded_num_constraints + constraint_index);

                        let az_coeff = constraint.a.evaluate_row(flattened_polynomials, step_index);
                        let bz_coeff = constraint.b.evaluate_row(flattened_polynomials, step_index);
                        if !az_coeff.is_zero() {
                            coeffs.push((global_index, az_coeff).into());
                        }
                        if !bz_coeff.is_zero() {
                            coeffs.push((global_index + 1, bz_coeff).into());
                        }
                    }
                }
                coeffs
            })
            .collect();

        Self {
            unbound_coeffs,
            bound_coeffs: vec![],
            binding_scratch_space: vec![],
        }
    }

    pub fn is_bound(&self) -> bool {
        !self.bound_coeffs.is_empty()
    }

    fn binding_output_length<T>(coeffs: &[SparseCoefficient<T>]) -> usize {
        let mut output_size = 0;
        for block in coeffs.chunk_by(|x, y| x.index / 4 == y.index / 4) {
            let mut az_found = false;
            let mut bz_found = false;
            for coeff in block {
                match coeff.index % 2 {
                    0 => {
                        if !az_found {
                            az_found = true;
                            output_size += 1;
                        }
                    }
                    1 => {
                        if !bz_found {
                            bz_found = true;
                            output_size += 1;
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
        output_size
    }

    pub fn final_sumcheck_evals(&self) -> [F; 2] {
        let mut final_az_eval = F::zero();
        let mut final_bz_eval = F::zero();
        for i in 0..2 {
            if let Some(coeff) = self.bound_coeffs.get(i) {
                match coeff.index {
                    0 => final_az_eval = coeff.value,
                    1 => final_bz_eval = coeff.value,
                    _ => {}
                }
            }
        }
        [final_az_eval, final_bz_eval]
    }
}

// =======================
// Streaming Baseline: No materialized polys, no optimizations
// =======================

// =======================
// SumcheckInstance (Prover) for outer_baseline (no uni-skip, no optimizations, streaming)
// =======================

#[derive(Allocative)]
pub struct OuterBaselineSumcheckProver<F: JoltField> {
    #[allocative(skip)]
    bytecode_preprocessing: BytecodePreprocessing,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    /// Split-eq helper tracking binding state
    eq_poly: GruenSplitEqPolynomial<F>,
    /// Dense Az, Bz multilinear polynomials over (cycle, constraint) variables
    az: DensePolynomial<F>,
    bz: DensePolynomial<F>,
    /// Total rounds = step_vars + constraint_vars
    total_rounds: usize,
    /// Number of step/cycle variables (used to split r_cycle from full opening point)
    num_step_vars: usize,
    /// Uniform constraints (all cycles use same constraint set)
    #[allocative(skip)]
    uniform_constraints: Vec<R1CSConstraint>,
    /// Padded number of constraints
    padded_num_constraints: usize,
}

impl<F: JoltField> OuterBaselineSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "OuterBaselineSumcheckProver::gen")]
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

        // Sample tau for entire outer sumcheck (no uni-skip)
        let tau: Vec<F::Challenge> = transcript.challenge_vector_optimized::<F>(total_num_vars);

        // Initialize eq-poly (no uni-skip / small-value scaling)
        let eq_poly = GruenSplitEqPolynomial::new(&tau, BindingOrder::LowToHigh);

        // Build dense Az, Bz polynomials over the (cycle, constraint) Boolean cube.
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

    /// Build dense Az, Bz polynomials over (cycle, constraint) variables by streaming over the trace.
    ///
    /// Domain layout (big-endian in bit order):
    /// - Let `num_step_vars = log2(num_steps_padded)` and `num_constraint_vars = log2(padded_num_constraints)`.
    /// - For each index `d in [0, 2^{num_step_vars + num_constraint_vars})`:
    ///     - Interpret the high `num_step_vars` bits of `d` as the cycle/step index.
    ///     - Interpret the low `num_constraint_vars` bits as the constraint index.
    ///     - Evaluate Az, Bz at that (step, constraint) pair, or 0 if out-of-range (padding).
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
            .expect("overflow computing baseline outer domain size");

        // Sanity: the live trace is a prefix of the padded cycle domain.
        debug_assert!(
            num_steps <= num_cycles_padded,
            "trace length ({num_steps}) must be <= padded cycles ({num_cycles_padded})"
        );

        let total_vars = num_step_vars + num_constraint_vars;
        debug_assert_eq!(
            domain_size,
            1usize << total_vars,
            "baseline outer: domain_size != 2^{total_vars}"
        );

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
                } else {
                    *az_ref = F::zero();
                    *bz_ref = F::zero();
                }
            });

        (DensePolynomial::new(az_vals), DensePolynomial::new(bz_vals))
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for OuterBaselineSumcheckProver<F> {
    fn degree(&self) -> usize {
        3
    }
    fn num_rounds(&self) -> usize {
        self.total_rounds
    }
    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "OuterBaselineSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let (t0, tinf) = self.compute_endpoints();
        // Use the same Gruen helper as the canonical outer to build the cubic round polynomial.
        self.eq_poly.gruen_poly_deg_3(t0, tinf, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "OuterBaselineSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        // Bind Az, Bz and the split-eq helper in lockstep (standard dense-poly binding).
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

impl<F: JoltField> OuterBaselineSumcheckProver<F> {
    /// Compute (t0, t_inf) endpoints for current round from dense Az/Bz polynomials.
    ///
    /// This mirrors `OuterRemainingSumcheckProver::remaining_quadratic_evals`, but
    /// uses direct field arithmetic (no delayed Montgomery reduction).
    fn compute_endpoints(&self) -> (F, F) {
        let eq_poly = &self.eq_poly;

        let n = self.az.len();
        debug_assert_eq!(n, self.bz.len());

        if eq_poly.E_in_current_len() == 1 {
            // groups are pairs (0,1)
            let groups = n / 2;
            (0..groups)
                .into_par_iter()
                .map(|g| {
                    let az0 = self.az[2 * g];
                    let az1 = self.az[2 * g + 1];
                    let bz0 = self.bz[2 * g];
                    let bz1 = self.bz[2 * g + 1];
                    let eq = eq_poly.E_out_current()[g];

                    let p0 = az0 * bz0;
                    let slope = (az1 - az0) * (bz1 - bz0);
                    (eq * p0, eq * slope)
                })
                .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
        } else {
            let num_x1_bits = eq_poly.E_in_current_len().log_2();
            let x1_len = eq_poly.E_in_current_len();
            let x2_len = eq_poly.E_out_current_len();

            (0..x2_len)
                .into_par_iter()
                .map(|x2| {
                    let mut inner0 = F::zero();
                    let mut inner_inf = F::zero();
                    for x1 in 0..x1_len {
                        let g = (x2 << num_x1_bits) | x1;
                        let az0 = self.az[2 * g];
                        let az1 = self.az[2 * g + 1];
                        let bz0 = self.bz[2 * g];
                        let bz1 = self.bz[2 * g + 1];
                        let e_in = eq_poly.E_in_current()[x1];
                        let p0 = az0 * bz0;
                        let slope = (az1 - az0) * (bz1 - bz0);
                        inner0 += e_in * p0;
                        inner_inf += e_in * slope;
                    }
                    let e_out = eq_poly.E_out_current()[x2];
                    (e_out * inner0, e_out * inner_inf)
                })
                .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
        }
    }
}
