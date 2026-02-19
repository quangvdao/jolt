#![allow(clippy::too_many_arguments)]
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::{
    dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, multilinear_polynomial::BindingOrder,
    split_eq_poly::GruenSplitEqPolynomial, unipoly::UniPoly,
};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::zkvm::r1cs::evaluation::R1CSEval;
use crate::zkvm::r1cs::inputs::ALL_R1CS_INPUTS;
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::witness::VirtualPolynomial;
use crate::{
    field::JoltField,
    transcripts::Transcript,
    utils::small_value::accum::{SignedUnreducedAccum, UnreducedProduct},
    utils::{math::Math, small_value::svo_helpers},
    zkvm::bytecode::BytecodePreprocessing,
    zkvm::r1cs::{
        constraints::R1CS_CONSTRAINTS, evaluation::eval_az_bz_batch_from_row,
        inputs::R1CSCycleInputs,
    },
};
use allocative::Allocative;
use ark_ff::biginteger::{S160, S96 as I8OrI96};
use num_traits::Zero;
use rayon::prelude::*;
use std::sync::Arc;
use tracer::instruction::Cycle;

/// Number of rounds to use for small value optimization.
/// Testing & estimation shows that 3 rounds is the best tradeoff
pub const NUM_SVO_ROUNDS: usize = 3;

pub const NUM_NONTRIVIAL_TERNARY_POINTS: usize =
    svo_helpers::num_non_trivial_ternary_points(NUM_SVO_ROUNDS);
pub const NUM_ACCUMS_EVAL_ZERO: usize = svo_helpers::num_accums_eval_zero(NUM_SVO_ROUNDS);
pub const NUM_ACCUMS_EVAL_INFTY: usize = svo_helpers::num_accums_eval_infty(NUM_SVO_ROUNDS);

/// Number of Y-assignments per SVO block. Equal to 2^NUM_SVO_ROUNDS.
/// This is the size of the subspace over the prefix Y used in small-value optimization.
pub const Y_SVO_SPACE_SIZE: usize = 1 << NUM_SVO_ROUNDS;

/// Number of interleaved coefficients per logical block for a fixed (x_out, x_in):
///  - 2 polynomials (Az, Bz)
///  - 2 evaluations at x_next ∈ {0, 1}
///  - Y_SVO_SPACE_SIZE assignments of Y
///    So total = 4 * Y_SVO_SPACE_SIZE.
pub const Y_SVO_RELATED_COEFF_BLOCK_SIZE: usize = 4 * Y_SVO_SPACE_SIZE;

/// Bit-width of a logical block. Computed as log2(4) + NUM_SVO_ROUNDS = 2 + NUM_SVO_ROUNDS.
/// Use this for fast block id calculation via shifts
pub const Y_SVO_RELATED_COEFF_BLOCK_SIZE_SHIFT: usize = 2 + NUM_SVO_ROUNDS;

/// Bitmask for the local offset within a block. Equals (1 << SHIFT) - 1.
/// Use this to extract local offsets via bit-and, instead of modulo.
pub const Y_SVO_RELATED_COEFF_BLOCK_SIZE_MASK: usize =
    (1usize << Y_SVO_RELATED_COEFF_BLOCK_SIZE_SHIFT) - 1;

/// SVO precomputation output: accumulators for the first NUM_SVO_ROUNDS rounds.
pub struct SvoPrecomputeResult<F: JoltField> {
    pub svo_accums_zero: [F; NUM_ACCUMS_EVAL_ZERO],
    pub svo_accums_infty: [F; NUM_ACCUMS_EVAL_INFTY],
}

/// Compute the SVO accumulators for the first NUM_SVO_ROUNDS rounds.
///
/// See the doc on the old `svo_sumcheck_round` for the mathematical formulation.
#[tracing::instrument(skip_all, name = "svo_precompute")]
pub fn svo_precompute<F: JoltField>(
    preprocess: &BytecodePreprocessing,
    trace: &[Cycle],
    tau: &[F::Challenge],
) -> SvoPrecomputeResult<F> {
    let padded_num_constraints = R1CS_CONSTRAINTS.len().next_power_of_two();
    let num_steps = trace.len();
    let num_step_vars = if num_steps > 0 { num_steps.log_2() } else { 0 };
    let num_constraint_vars = if padded_num_constraints > 0 {
        padded_num_constraints.log_2()
    } else {
        0
    };
    let total_num_vars = num_step_vars + num_constraint_vars;

    assert_eq!(tau.len(), total_num_vars);
    assert!(NUM_SVO_ROUNDS <= num_constraint_vars);

    let num_non_svo_constraint_vars = num_constraint_vars.saturating_sub(NUM_SVO_ROUNDS);
    let num_non_svo_z_vars = num_step_vars + num_non_svo_constraint_vars;

    let potential_x_out_vars = total_num_vars / 2 - NUM_SVO_ROUNDS;
    let iter_num_x_out_vars = std::cmp::min(potential_x_out_vars, num_step_vars);
    let iter_num_x_in_vars = num_non_svo_z_vars - iter_num_x_out_vars;
    let iter_num_x_in_step_vars = num_step_vars - iter_num_x_out_vars;
    let iter_num_x_in_constraint_vars = num_non_svo_constraint_vars;
    assert_eq!(
        iter_num_x_in_vars,
        iter_num_x_in_step_vars + iter_num_x_in_constraint_vars
    );

    let num_uniform_r1cs_constraints = R1CS_CONSTRAINTS.len();
    let rem_num_uniform_r1cs_constraints = num_uniform_r1cs_constraints % Y_SVO_SPACE_SIZE;

    let eq_poly = GruenSplitEqPolynomial::new_for_small_value(
        tau,
        iter_num_x_out_vars,
        iter_num_x_in_vars,
        NUM_SVO_ROUNDS,
        Some(F::MONTGOMERY_R_SQUARE),
    );
    let E_in_evals = eq_poly.E_in_current();
    let E_out_vec = &eq_poly.E_out_vec;
    assert_eq!(E_out_vec.len(), NUM_SVO_ROUNDS);

    let num_x_out_vals = 1usize << iter_num_x_out_vars;
    let num_x_in_step_vals = 1usize << iter_num_x_in_step_vars;

    struct ChunkOutput<F: JoltField> {
        zero: [F; NUM_ACCUMS_EVAL_ZERO],
        infty: [F; NUM_ACCUMS_EVAL_INFTY],
    }

    let num_parallel_chunks = std::cmp::min(
        std::cmp::max(num_x_out_vals, 1),
        rayon::current_num_threads().next_power_of_two() * 8,
    );
    let x_out_chunk_size = std::cmp::max(1, num_x_out_vals.div_ceil(num_parallel_chunks));

    let chunk_outputs: Vec<ChunkOutput<F>> = (0..num_parallel_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let x_out_start = chunk_idx * x_out_chunk_size;
            let x_out_end = std::cmp::min((chunk_idx + 1) * x_out_chunk_size, num_x_out_vals);

            let mut chunk_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
            let mut chunk_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];

            for x_out_val in x_out_start..x_out_end {
                let mut tA_pos = [UnreducedProduct::<F>::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];
                let mut tA_neg = [UnreducedProduct::<F>::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];
                let mut x_out_svo_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
                let mut x_out_svo_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];

                for x_in_step_val in 0..num_x_in_step_vals {
                    let step_idx = (x_out_val << iter_num_x_in_step_vars) | x_in_step_val;
                    let mut constraint_val = 0;
                    let row = R1CSCycleInputs::from_trace::<F>(preprocess, trace, step_idx);

                    let mut az_block = [I8OrI96::zero(); Y_SVO_SPACE_SIZE];
                    let mut bz_block = [S160::zero(); Y_SVO_SPACE_SIZE];

                    for chunk in R1CS_CONSTRAINTS.chunks(Y_SVO_SPACE_SIZE) {
                        let sz = chunk.len();
                        eval_az_bz_batch_from_row::<F>(
                            chunk,
                            &row,
                            &mut az_block[..sz],
                            &mut bz_block[..sz],
                        );

                        if chunk.len() == Y_SVO_SPACE_SIZE {
                            let x_in =
                                (x_in_step_val << iter_num_x_in_constraint_vars) | constraint_val;
                            svo_helpers::compute_and_update_tA_inplace::<NUM_SVO_ROUNDS, F>(
                                &az_block,
                                &bz_block,
                                &E_in_evals[x_in],
                                &mut tA_pos,
                                &mut tA_neg,
                            );
                            constraint_val += 1;
                            az_block = [I8OrI96::zero(); Y_SVO_SPACE_SIZE];
                            bz_block = [S160::zero(); Y_SVO_SPACE_SIZE];
                        }
                    }

                    if rem_num_uniform_r1cs_constraints > 0 {
                        let x_in =
                            (x_in_step_val << iter_num_x_in_constraint_vars) | constraint_val;
                        svo_helpers::compute_and_update_tA_inplace::<NUM_SVO_ROUNDS, F>(
                            &az_block,
                            &bz_block,
                            &E_in_evals[x_in],
                            &mut tA_pos,
                            &mut tA_neg,
                        );
                    }
                }

                let mut tA_sum = [F::zero(); NUM_NONTRIVIAL_TERNARY_POINTS];
                for i in 0..NUM_NONTRIVIAL_TERNARY_POINTS {
                    let pos = F::from_montgomery_reduce::<8>(tA_pos[i]);
                    let neg = F::from_montgomery_reduce::<8>(tA_neg[i]);
                    tA_sum[i] = pos - neg;
                }

                svo_helpers::distribute_tA_to_svo_accumulators_generic::<NUM_SVO_ROUNDS, F>(
                    &tA_sum,
                    x_out_val,
                    E_out_vec,
                    &mut x_out_svo_zero,
                    &mut x_out_svo_infty,
                );

                for i in 0..NUM_ACCUMS_EVAL_ZERO {
                    chunk_zero[i] += x_out_svo_zero[i];
                }
                for i in 0..NUM_ACCUMS_EVAL_INFTY {
                    chunk_infty[i] += x_out_svo_infty[i];
                }
            }

            ChunkOutput {
                zero: chunk_zero,
                infty: chunk_infty,
            }
        })
        .collect();

    let mut final_zero = [F::zero(); NUM_ACCUMS_EVAL_ZERO];
    let mut final_infty = [F::zero(); NUM_ACCUMS_EVAL_INFTY];
    for out in chunk_outputs {
        for i in 0..NUM_ACCUMS_EVAL_ZERO {
            final_zero[i] += out.zero[i];
        }
        for i in 0..NUM_ACCUMS_EVAL_INFTY {
            final_infty[i] += out.infty[i];
        }
    }

    SvoPrecomputeResult {
        svo_accums_zero: final_zero,
        svo_accums_infty: final_infty,
    }
}

/// Streaming round: computes (t(0), t(∞)) and materializes dense Az/Bz polynomials
/// for subsequent binding rounds.
///
/// The dense polynomials are indexed so that for block g:
///   az[2*g]   = Az at x_next=0 (folded over SVO y-vars at r_challenges)
///   az[2*g+1] = Az at x_next=1
/// (likewise for bz). This matches DensePolynomial's LowToHigh binding convention.
#[tracing::instrument(skip_all, name = "compute_streaming_round_dense")]
pub fn compute_streaming_round_dense<F: JoltField>(
    preprocess: &BytecodePreprocessing,
    trace: &[Cycle],
    eq_poly: &GruenSplitEqPolynomial<F>,
    r_challenges: &[F::Challenge],
) -> ((F, F), DensePolynomial<F>, DensePolynomial<F>) {
    let mut r_rev = r_challenges.to_vec();
    r_rev.reverse();
    let eq_r_evals = EqPolynomial::<F>::evals_with_scaling(&r_rev, Some(F::MONTGOMERY_R_SQUARE));

    let num_x_out_vals = eq_poly.E_out_current_len();
    let iter_num_x_out_vars = if num_x_out_vals > 0 {
        num_x_out_vals.log_2()
    } else {
        0
    };
    let num_steps = trace.len();
    let num_step_vars = if num_steps > 0 { num_steps.log_2() } else { 0 };
    debug_assert!(iter_num_x_out_vars <= num_step_vars);
    let iter_num_x_in_step_vars = num_step_vars - iter_num_x_out_vars;
    let num_x_in_step_vals = if iter_num_x_in_step_vars > 0 {
        1usize << iter_num_x_in_step_vars
    } else {
        1
    };

    let num_uniform_r1cs_constraints = R1CS_CONSTRAINTS.len();
    let y_blocks_in_constraints = if num_uniform_r1cs_constraints > 0 {
        num_uniform_r1cs_constraints.div_ceil(Y_SVO_SPACE_SIZE)
    } else {
        0
    };
    let num_block_pairs_per_step =
        (R1CS_CONSTRAINTS.len().next_power_of_two()) >> (NUM_SVO_ROUNDS + 1);

    let total_blocks = num_steps * num_block_pairs_per_step;
    let dense_len = 2 * total_blocks;
    let mut az_vals = vec![F::zero(); dense_len];
    let mut bz_vals = vec![F::zero(); dense_len];

    let mut sum0 = F::zero();
    let mut sumInf = F::zero();

    for x_out_val in 0..num_x_out_vals {
        let mut inner_sum0 = F::Unreduced::<9>::zero();
        let mut inner_sumInf = F::Unreduced::<9>::zero();
        for x_in_step_val in 0..num_x_in_step_vals {
            let current_step_idx = (x_out_val << iter_num_x_in_step_vars) | x_in_step_val;
            let row_inputs = R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);

            for block_pair_idx in 0..num_block_pairs_per_step {
                let mut az_acc = [
                    SignedUnreducedAccum::<F>::new(),
                    SignedUnreducedAccum::<F>::new(),
                ];
                let mut bz_acc = [
                    SignedUnreducedAccum::<F>::new(),
                    SignedUnreducedAccum::<F>::new(),
                ];

                for k in 0..2 {
                    let chunk_index = (block_pair_idx << 1) | k;
                    if chunk_index >= y_blocks_in_constraints {
                        continue;
                    }
                    let start = chunk_index * Y_SVO_SPACE_SIZE;
                    let end =
                        core::cmp::min(start + Y_SVO_SPACE_SIZE, num_uniform_r1cs_constraints);
                    let uniform_svo_chunk = &R1CS_CONSTRAINTS[start..end];
                    let chunk_size = uniform_svo_chunk.len();

                    let mut binary_az_block = [I8OrI96::zero(); Y_SVO_SPACE_SIZE];
                    let mut binary_bz_block = [S160::zero(); Y_SVO_SPACE_SIZE];
                    eval_az_bz_batch_from_row::<F>(
                        uniform_svo_chunk,
                        &row_inputs,
                        &mut binary_az_block[..chunk_size],
                        &mut binary_bz_block[..chunk_size],
                    );

                    let x_next_val = k;
                    for idx in 0..chunk_size {
                        let eq = eq_r_evals[idx];
                        az_acc[x_next_val].fmadd_az(&eq, binary_az_block[idx]);
                        bz_acc[x_next_val].fmadd_bz(&eq, binary_bz_block[idx]);
                    }
                }

                let az0 = az_acc[0].reduce_to_field();
                let bz0 = bz_acc[0].reduce_to_field();
                let az1 = az_acc[1].reduce_to_field();
                let bz1 = bz_acc[1].reduce_to_field();

                let p0 = az0 * bz0;
                let slope = (az1 - az0) * (bz1 - bz0);

                let block_id = current_step_idx * num_block_pairs_per_step + block_pair_idx;
                let num_streaming_x_in_vars = eq_poly.E_in_current_len().log_2();
                let x_out_idx = block_id >> num_streaming_x_in_vars;
                let x_in_idx = block_id & ((1 << num_streaming_x_in_vars) - 1);

                let e_out = if x_out_idx < eq_poly.E_out_current_len() {
                    eq_poly.E_out_current()[x_out_idx]
                } else {
                    F::zero()
                };
                let e_in = if eq_poly.E_in_current_len() == 0 {
                    F::one()
                } else if eq_poly.E_in_current_len() == 1 {
                    eq_poly.E_in_current()[0]
                } else if x_in_idx < eq_poly.E_in_current_len() {
                    eq_poly.E_in_current()[x_in_idx]
                } else {
                    F::zero()
                };

                inner_sum0 += e_in.mul_unreduced::<9>(p0);
                inner_sumInf += e_in.mul_unreduced::<9>(slope);

                az_vals[2 * block_id] = az0;
                az_vals[2 * block_id + 1] = az1;
                bz_vals[2 * block_id] = bz0;
                bz_vals[2 * block_id + 1] = bz1;

                let red0 = F::from_montgomery_reduce::<9>(inner_sum0);
                let redi = F::from_montgomery_reduce::<9>(inner_sumInf);
                sum0 += e_out * red0;
                sumInf += e_out * redi;
                inner_sum0 = F::Unreduced::zero();
                inner_sumInf = F::Unreduced::zero();
            }
        }
    }

    (
        (sum0, sumInf),
        DensePolynomial::new(az_vals),
        DensePolynomial::new(bz_vals),
    )
}

/// Compute endpoints (t0, t_inf) from dense Az/Bz polynomials using the split-eq structure.
#[inline]
fn dense_compute_endpoints<F: JoltField>(
    eq_poly: &GruenSplitEqPolynomial<F>,
    az: &DensePolynomial<F>,
    bz: &DensePolynomial<F>,
) -> (F, F) {
    let n = az.len();
    debug_assert_eq!(n, bz.len());

    if eq_poly.E_in_current_len() == 1 {
        let groups = n / 2;
        (0..groups)
            .into_par_iter()
            .map(|g| {
                let az0 = az[2 * g];
                let az1 = az[2 * g + 1];
                let bz0 = bz[2 * g];
                let bz1 = bz[2 * g + 1];
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
                    let az0 = az[2 * g];
                    let az1 = az[2 * g + 1];
                    let bz0 = bz[2 * g];
                    let bz1 = bz[2 * g + 1];
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

#[derive(Allocative)]
pub struct OuterRoundBatchedSumcheckProver<F: JoltField> {
    #[allocative(skip)]
    preprocess: BytecodePreprocessing,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    eq_poly: GruenSplitEqPolynomial<F>,
    eq_small_value: GruenSplitEqPolynomial<F>,
    svo_accums_zero: [F; NUM_ACCUMS_EVAL_ZERO],
    svo_accums_infty: [F; NUM_ACCUMS_EVAL_INFTY],
    r_svo: Vec<F::Challenge>,
    lagrange_coeffs: Vec<F>,
    /// Dense Az polynomial, populated at the streaming round and bound thereafter.
    /// `None` during SVO rounds, `Some` from the streaming round onward.
    az: Option<DensePolynomial<F>>,
    /// Dense Bz polynomial, populated at the streaming round and bound thereafter.
    bz: Option<DensePolynomial<F>>,
    total_rounds: usize,
    num_cycle_bits: usize,
}

impl<F: JoltField> OuterRoundBatchedSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "OuterRoundBatchedSumcheckProver::gen")]
    pub fn gen<T: Transcript>(
        trace: Arc<Vec<Cycle>>,
        bytecode_preprocessing: &BytecodePreprocessing,
        transcript: &mut T,
    ) -> Self {
        let num_steps = trace.len();
        let num_step_vars = if num_steps > 0 { num_steps.log_2() } else { 0 };
        let padded_num_constraints = R1CS_CONSTRAINTS.len().next_power_of_two();
        let num_constraint_vars = if padded_num_constraints > 0 {
            padded_num_constraints.log_2()
        } else {
            0
        };
        let total_num_vars = num_step_vars + num_constraint_vars;

        let tau: Vec<F::Challenge> = transcript.challenge_vector_optimized::<F>(total_num_vars);

        let svo = svo_precompute::<F>(bytecode_preprocessing, trace.as_ref(), &tau);

        let eq_poly = GruenSplitEqPolynomial::new(&tau, BindingOrder::LowToHigh);

        let potential_x_out_vars = total_num_vars / 2 - NUM_SVO_ROUNDS;
        let iter_num_x_out_vars = core::cmp::min(potential_x_out_vars, num_step_vars);
        let iter_num_x_in_vars = (num_step_vars + num_constraint_vars - NUM_SVO_ROUNDS)
            .saturating_sub(iter_num_x_out_vars);
        let eq_small_value = GruenSplitEqPolynomial::new_for_small_value(
            &tau,
            iter_num_x_out_vars,
            iter_num_x_in_vars,
            NUM_SVO_ROUNDS,
            Some(F::MONTGOMERY_R_SQUARE),
        );

        Self {
            preprocess: bytecode_preprocessing.clone(),
            trace,
            eq_poly,
            eq_small_value,
            svo_accums_zero: svo.svo_accums_zero,
            svo_accums_infty: svo.svo_accums_infty,
            r_svo: Vec::with_capacity(NUM_SVO_ROUNDS),
            lagrange_coeffs: vec![F::one()],
            az: None,
            bz: None,
            total_rounds: total_num_vars,
            num_cycle_bits: num_step_vars,
        }
    }

    #[inline]
    fn compute_svo_quadratic_evals(&self, round: usize) -> (F, F) {
        debug_assert!(round < NUM_SVO_ROUNDS);
        let (offsets_infty, offsets_zero) =
            svo_helpers::precompute_accumulator_offsets::<NUM_SVO_ROUNDS>();
        let start_inf = offsets_infty[round];
        let len_inf = svo_helpers::pow(3, round);
        let start_zero = offsets_zero[round];
        let len_zero = if round == 0 {
            0
        } else {
            svo_helpers::pow(3, round) - svo_helpers::pow(2, round)
        };

        let mut eval_inf = F::zero();
        for i in 0..len_inf {
            eval_inf += self.svo_accums_infty[start_inf + i] * self.lagrange_coeffs[i];
        }

        let mut eval_zero = F::zero();
        if len_zero > 0 {
            let mut non_binary_idx = 0usize;
            let pow3 = svo_helpers::pow(3, round);
            for k in 0..pow3 {
                let mut is_non_binary = false;
                let mut tmp = k;
                for _ in 0..round {
                    if tmp % 3 == 2 {
                        is_non_binary = true;
                        break;
                    }
                    tmp /= 3;
                }
                if is_non_binary {
                    eval_zero +=
                        self.svo_accums_zero[start_zero + non_binary_idx] * self.lagrange_coeffs[k];
                    non_binary_idx += 1;
                }
            }
        }
        (eval_zero, eval_inf)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for OuterRoundBatchedSumcheckProver<F>
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

    #[tracing::instrument(skip_all, name = "OuterRoundBatchedSumcheckProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let (t0, tinf) = if round < NUM_SVO_ROUNDS {
            self.compute_svo_quadratic_evals(round)
        } else if round == NUM_SVO_ROUNDS {
            let ((t0, tinf), az, bz) = compute_streaming_round_dense::<F>(
                &self.preprocess,
                &self.trace,
                &self.eq_poly,
                &self.r_svo,
            );
            self.az = Some(az);
            self.bz = Some(bz);
            (t0, tinf)
        } else {
            dense_compute_endpoints(
                &self.eq_poly,
                self.az.as_ref().expect("Az not yet populated"),
                self.bz.as_ref().expect("Bz not yet populated"),
            )
        };

        self.eq_poly.gruen_poly_deg_3(t0, tinf, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "OuterRoundBatchedSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if round < NUM_SVO_ROUNDS {
            self.r_svo.push(r_j);
            if round + 1 < NUM_SVO_ROUNDS {
                let lag = [F::one() - r_j, r_j.into(), r_j * (r_j - F::one())];
                let prev = core::mem::take(&mut self.lagrange_coeffs);
                let mut next = Vec::with_capacity(prev.len() * 3);
                for c in lag.iter() {
                    for p in prev.iter() {
                        next.push(*c * *p);
                    }
                }
                self.lagrange_coeffs = next;
            }
            self.eq_poly.bind(r_j);
            return;
        }

        // Streaming round and all remaining rounds: bind the dense Az/Bz and eq_poly
        let az = self.az.as_mut().expect("Az not yet populated");
        let bz = self.bz.as_mut().expect("Bz not yet populated");
        rayon::join(
            || az.bind_parallel(r_j, BindingOrder::LowToHigh),
            || bz.bind_parallel(r_j, BindingOrder::LowToHigh),
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

        let (r_cycle, _rx_var) = opening_point.r.split_at(self.num_cycle_bits);
        let r_cycle_point =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(r_cycle.to_vec()).match_endianness();
        let claimed_witness_evals =
            R1CSEval::compute_claimed_inputs(&self.preprocess, &self.trace, &r_cycle_point);
        for (i, input) in crate::zkvm::r1cs::inputs::ALL_R1CS_INPUTS
            .iter()
            .enumerate()
        {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                r_cycle_point.clone(),
                claimed_witness_evals[i],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct OuterRoundBatchedSumcheckVerifier<F: JoltField> {
    num_cycle_bits: usize,
    total_rounds: usize,
    tau: Vec<F::Challenge>,
    key: UniformSpartanKey<F>,
    _phantom: core::marker::PhantomData<F>,
}

impl<F: JoltField> OuterRoundBatchedSumcheckVerifier<F> {
    pub fn new(
        num_cycle_bits: usize,
        num_constraint_bits: usize,
        tau: Vec<F::Challenge>,
        key: UniformSpartanKey<F>,
    ) -> Self {
        Self {
            num_cycle_bits,
            total_rounds: num_cycle_bits + num_constraint_bits,
            tau,
            key,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for OuterRoundBatchedSumcheckVerifier<F>
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
        let r1cs_input_evals = ALL_R1CS_INPUTS.map(|input| {
            accumulator
                .get_virtual_polynomial_opening((&input).into(), SumcheckId::SpartanOuter)
                .1
        });

        debug_assert!(
            sumcheck_challenges.len() >= 2,
            "round-batched outer: expected at least two challenges for row binding"
        );
        let rx_constr = &[sumcheck_challenges[0], sumcheck_challenges[1]];
        let inner_sum_prod = self
            .key
            .evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals);

        let r_rev: Vec<F::Challenge> = sumcheck_challenges.iter().rev().copied().collect();
        let eq_tau_r = EqPolynomial::<F>::mle(&self.tau, &r_rev);

        eq_tau_r * inner_sum_prod
    }
    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();

        let (r_cycle, _rx_var) = opening_point.r.split_at(self.num_cycle_bits);
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
