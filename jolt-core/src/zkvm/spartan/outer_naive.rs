use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, LITTLE_ENDIAN,
};
use crate::poly::{
    multilinear_polynomial::MultilinearPolynomial, split_eq_poly::GruenSplitEqPolynomial,
};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;

use crate::zkvm::witness::VirtualPolynomial;
use crate::{
    field::{JoltField, OptimizedMul},
    transcripts::Transcript,
    utils::math::Math,
    zkvm::r1cs::constraints::R1CSConstraint,
};
use allocative::Allocative;
use rayon::prelude::*;

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct SparseCoefficient<T> {
    pub(crate) index: usize,
    pub(crate) value: T,
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
pub struct NaiveSpartanInterleavedPolynomial<F: JoltField> {
    pub(crate) unbound_coeffs: Vec<SparseCoefficient<F>>,
    pub(crate) bound_coeffs: Vec<SparseCoefficient<F>>,
    binding_scratch_space: Vec<SparseCoefficient<F>>,
}

impl<F: JoltField> NaiveSpartanInterleavedPolynomial<F> {
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
            self.bound_coeffs = Vec::with_capacity(total_output_len);
            unsafe { self.bound_coeffs.set_len(total_output_len) }

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
            if self.binding_scratch_space.is_empty() {
                self.binding_scratch_space = Vec::with_capacity(total_output_len);
            }
            unsafe { self.binding_scratch_space.set_len(total_output_len) }

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

                        let (az_coeff, bz_coeff) =
                            constraint.evaluate_row(flattened_polynomials, step_index);
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
// SumcheckInstance (Prover) for outer_linear_time (no uni-skip, no SVO)
// =======================

#[derive(Allocative)]
pub struct OuterBaselineSumcheckProver<F: JoltField> {
    /// Split-eq helper tracking binding state
    eq_poly: GruenSplitEqPolynomial<F>,
    /// Interleaved Az/Bz holder and binding workspace
    poly: NaiveSpartanInterleavedPolynomial<F>,
    /// Total rounds = step_vars + constraint_vars
    total_rounds: usize,
}

impl<F: JoltField> OuterBaselineSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "OuterBaselineSumcheckProver::gen_from_polys")]
    pub fn gen_from_polys<ProofTranscript: Transcript>(
        uniform_constraints: &[R1CSConstraint],
        flattened_polynomials: &[MultilinearPolynomial<F>],
        padded_num_constraints: usize,
        transcript: &mut ProofTranscript,
    ) -> Self {
        // Determine step and constraint vars
        let num_steps = flattened_polynomials[0].len();
        let num_step_vars = if num_steps > 0 { num_steps.log_2() } else { 0 };
        let num_constraint_vars = if padded_num_constraints > 0 {
            padded_num_constraints.log_2()
        } else {
            0
        };
        let total_num_vars = num_step_vars + num_constraint_vars;

        // Sample tau for entire outer sumcheck (no uni-skip)
        let tau: Vec<F::Challenge> = transcript.challenge_vector_optimized::<F>(total_num_vars);

        // Initialize eq-poly (no SVO partitioning)
        let eq_poly = GruenSplitEqPolynomial::new(&tau, BindingOrder::LowToHigh);

        // Materialize baseline interleaved polynomial
        let poly = NaiveSpartanInterleavedPolynomial::new(
            uniform_constraints,
            flattened_polynomials,
            padded_num_constraints,
        );

        Self {
            eq_poly,
            poly,
            total_rounds: total_num_vars,
        }
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

    #[tracing::instrument(skip_all, name = "OuterBaselineSumcheckProver::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, previous_claim: F) -> Vec<F> {
        let (t0, tinf) = if !self.poly.is_bound() {
            self.poly.endpoints_unbound_first_round(&self.eq_poly)
        } else {
            self.poly.endpoints_bound_remaining(&self.eq_poly)
        };
        let evals = self.eq_poly.gruen_evals_deg_3(t0, tinf, previous_claim);
        vec![evals[0], evals[1], evals[2]]
    }

    #[tracing::instrument(skip_all, name = "OuterBaselineSumcheckProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        self.poly.bind_inplace(r_j);
        self.eq_poly.bind(r_j);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Opening point uses the sumcheck challenges; endianness matched by OpeningPoint impl
        let opening_point =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();

        // Append Az, Bz claims and corresponding opening point
        let claims = self.poly.final_sumcheck_evals();
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
            opening_point,
            claims[1],
        );
    }
}
