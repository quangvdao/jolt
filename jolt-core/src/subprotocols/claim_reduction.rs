//! Generic claim reduction sumcheck.
//!
//! This module provides a framework for reducing multiple polynomial opening claims
//! to a single claim using sumcheck. It supports:
//! - Dense polynomial claims
//! - One-hot polynomial claims (with address and cycle variables)
//! - Instruction lookup claims (fused trace-based)
//!
//! The sumcheck proceeds in three phases:
//! 1. Address phase: binds address variables (only for one-hot claims)
//! 2. Cycle phase 1 (prefix-suffix): uses prefix-suffix optimization
//! 3. Cycle phase 2 (linear): standard linear sumcheck for remaining variables

use std::sync::{Arc, RwLock};

use allocative::Allocative;
use ark_std::Zero;
use common::constants::XLEN;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding,
};
use crate::poly::one_hot_polynomial::{EqAddressState, EqCycleState, OneHotPolynomial};
use crate::poly::ra_poly::RaPolynomial;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
use crate::utils::math::Math;
use crate::utils::thread::{drop_in_background_thread, unsafe_allocate_zero_vec};
use crate::zkvm::instruction::LookupQuery;
use crate::{
    poly::opening_proof::{
        OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
        BIG_ENDIAN, LITTLE_ENDIAN,
    },
    subprotocols::sumcheck_prover::SumcheckInstanceProver,
    transcripts::Transcript,
};

/// Degree bound of the sumcheck round polynomials in claim reduction.
const DEGREE_BOUND: usize = 2;

// ============================================================================
// Parameters
// ============================================================================

/// Parameters for a generic claim reduction sumcheck.
#[derive(Clone, Default, Allocative)]
pub struct ClaimReductionSumcheckParams<F: JoltField> {
    pub num_address_vars: usize,
    pub num_cycle_vars: usize,
    pub r_address: Vec<F::Challenge>,
    pub r_cycle: Vec<F::Challenge>,
    pub degree: usize,
    /// The input claim for this sumcheck instance
    pub input_claim: F,
}

impl<F: JoltField> SumcheckInstanceParams<F> for ClaimReductionSumcheckParams<F> {
    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn num_rounds(&self) -> usize {
        self.num_address_vars + self.num_cycle_vars
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        // For claim reduction, we bind LowToHigh, so the challenges are in little-endian order
        // and need to be reversed for big-endian opening points
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

// ============================================================================
// ClaimSource trait and implementations
// ============================================================================

/// A claim to be reduced in the sumcheck protocol.
///
/// This trait abstracts over different polynomial types (dense, one-hot, trace-based).
/// Each claim source provides:
/// - A coefficient for RLC batching
/// - Methods to compute sumcheck messages in each phase
/// - Methods to bind variables in each phase
pub trait ClaimSource<F: JoltField>: Send + Sync {
    /// Returns the RLC coefficient for this claim.
    fn coeff(&self) -> F;

    /// Compute the sumcheck message for address phase.
    /// Returns [w(0), w(2)] evaluations (unscaled by coeff).
    fn compute_address_message(&self, round: usize) -> [F; 2];

    /// Bind an address variable.
    fn bind_address(&mut self, r: F::Challenge, round: usize);

    /// Initialize the Q buffer for prefix-suffix sumcheck.
    ///
    /// Computes: Q[x_lo] += coeff * \sum_{x_hi} P(x_lo, x_hi) * eq_suffix[x_hi]
    fn initialize_prefix_suffix_Q(
        &self,
        q_buffer: &mut [F],
        eq_suffix_evals: &[F],
        prefix_n_vars: usize,
        suffix_n_vars: usize,
    );

    /// Compute the sumcheck message for linear phase.
    /// Returns [w(0), w(2)] evaluations (unscaled by coeff).
    fn compute_linear_message(&self, round: usize, eq_evals: &[F]) -> [F; 2];

    /// Bind a cycle variable.
    fn bind_cycle(&mut self, r: F::Challenge, round: usize);

    /// Returns the final claimed value after all variables are bound.
    fn final_claim(&self) -> F;
}

// ============================================================================
// DenseClaimSource
// ============================================================================

/// A dense polynomial claim source.
#[derive(Clone)]
pub struct DenseClaimSource<F: JoltField> {
    pub poly: MultilinearPolynomial<F>,
    pub coefficient: F,
    /// Shared eq state for cycle variables
    pub eq_cycle_state: Arc<RwLock<EqCycleState<F>>>,
}

impl<F: JoltField> DenseClaimSource<F> {
    pub fn new(
        poly: MultilinearPolynomial<F>,
        coefficient: F,
        eq_cycle_state: Arc<RwLock<EqCycleState<F>>>,
    ) -> Self {
        Self {
            poly,
            coefficient,
            eq_cycle_state,
        }
    }
}

impl<F: JoltField> ClaimSource<F> for DenseClaimSource<F> {
    fn coeff(&self) -> F {
        self.coefficient
    }

    fn compute_address_message(&self, _round: usize) -> [F; 2] {
        // Dense claims don't participate in address phase
        [F::zero(), F::zero()]
    }

    fn bind_address(&mut self, _r: F::Challenge, _round: usize) {
        // No-op for dense claims
    }

    fn initialize_prefix_suffix_Q(
        &self,
        q_buffer: &mut [F],
        eq_suffix_evals: &[F],
        prefix_n_vars: usize,
        suffix_n_vars: usize,
    ) {
        let poly_len = self.poly.len();
        debug_assert_eq!(poly_len, (1 << prefix_n_vars) * (1 << suffix_n_vars));

        const BLOCK_SIZE: usize = 32;
        let coeff = self.coefficient;

        q_buffer
            .par_chunks_mut(BLOCK_SIZE)
            .enumerate()
            .for_each(|(chunk_i, q_chunk)| {
                let mut q_acc = [F::zero(); BLOCK_SIZE];

                for x_hi in 0..(1 << suffix_n_vars) {
                    let suffix_eq = eq_suffix_evals[x_hi];
                    let base_idx = x_hi << prefix_n_vars;

                    for i in 0..q_chunk.len() {
                        let x_lo = chunk_i * BLOCK_SIZE + i;
                        q_acc[i] += self.poly.get_coeff(base_idx + x_lo) * suffix_eq;
                    }
                }

                for (i, q) in q_chunk.iter_mut().enumerate() {
                    *q += coeff * q_acc[i];
                }
            });
    }

    fn compute_linear_message(&self, _round: usize, eq_evals: &[F]) -> [F; 2] {
        let len = self.poly.len();
        debug_assert_eq!(eq_evals.len(), len);

        // Compute [w(0), w(2)] where w(x) = sum_i p(i_hi, x, i_lo) * eq(i)
        // The polynomial is indexed as P[2*i + b] for low bit b
        // eq[2*i] corresponds to x=0 case, eq[2*i+1] to x=1 case
        let half_len = len / 2;
        let (eval_0, eval_2) = (0..half_len)
            .into_par_iter()
            .map(|i| {
                let p0 = self.poly.get_bound_coeff(2 * i);
                let p1 = self.poly.get_bound_coeff(2 * i + 1);
                let eq0 = eq_evals[2 * i];
                let eq1 = eq_evals[2 * i + 1];

                // For eval_0: contribution from x=0 terms
                let term_0 = p0 * eq0;

                // For eval_2: need P(x=2) * eq(x=2)
                let p2 = p1 + p1 - p0;
                let eq2 = eq1 + eq1 - eq0;
                let term_2 = p2 * eq2;

                (term_0, term_2)
            })
            .reduce(
                || (F::zero(), F::zero()),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            );

        [eval_0, eval_2]
    }

    fn bind_cycle(&mut self, r: F::Challenge, round: usize) {
        // Bind the shared eq state if not already bound
        let mut shared_eq = self.eq_cycle_state.write().unwrap();
        if shared_eq.num_variables_bound <= round {
            shared_eq.D.bind(r);
            shared_eq.num_variables_bound += 1;
        }
        drop(shared_eq);

        // Always bind the polynomial
        self.poly.bind_parallel(r, BindingOrder::LowToHigh);
    }

    fn final_claim(&self) -> F {
        self.poly.final_sumcheck_claim()
    }
}

// ============================================================================
// OneHotClaimSource
// ============================================================================

/// A one-hot polynomial claim source, wrapping the existing one-hot machinery.
#[derive(Clone, Allocative)]
pub struct OneHotClaimSource<F: JoltField> {
    pub log_T: usize,
    pub polynomial: OneHotPolynomial<F>,
    pub coefficient: F,
    pub eq_address_state: Arc<RwLock<EqAddressState<F>>>,
    pub eq_cycle_state: Arc<RwLock<EqCycleState<F>>>,
    /// The G array (used during address phase)
    G: Vec<F>,
    /// The H array (RaPolynomial, created after address phase)
    H: Arc<RwLock<RaPolynomial<u16, F>>>,
}

impl<F: JoltField> OneHotClaimSource<F> {
    #[tracing::instrument(skip_all, name = "OneHotClaimSource::new")]
    pub fn new(
        polynomial: OneHotPolynomial<F>,
        coefficient: F,
        eq_address_state: Arc<RwLock<EqAddressState<F>>>,
        eq_cycle_state: Arc<RwLock<EqCycleState<F>>>,
    ) -> Self {
        let nonzero_indices = &polynomial.nonzero_indices;
        let T = nonzero_indices.len();
        let K = polynomial.K;

        // Compute G using the eq_cycle_state's GruenSplitEqPolynomial
        let eq = eq_cycle_state.read().unwrap();
        let E_in = eq.D.E_in_current();
        let E_out = eq.D.E_out_current();
        let w_current = eq.D.get_current_w();
        let factor_0 = F::one() - w_current;
        let factor_1: F = w_current.into();

        // Precompute merged inner weights
        let in_len = E_in.len();
        let x_in_bits = in_len.log_2();
        let merged_in_unreduced: Vec<F::Unreduced<9>> = {
            let mut merged: Vec<F::Unreduced<9>> = unsafe_allocate_zero_vec(2 * in_len);
            merged
                .par_chunks_exact_mut(2)
                .zip(E_in.par_iter())
                .for_each(|(chunk, &low)| {
                    chunk[0] = low.mul_unreduced::<9>(factor_0);
                    chunk[1] = low.mul_unreduced::<9>(factor_1);
                });
            merged
        };

        let G = E_out
            .par_iter()
            .enumerate()
            .fold(
                || unsafe_allocate_zero_vec(K),
                |mut partial, (x_out, &e_out)| {
                    let mut local_unreduced: Vec<F::Unreduced<9>> = unsafe_allocate_zero_vec(K);
                    let x_out_base = x_out << (x_in_bits + 1);

                    for x_in in 0..in_len {
                        let j0 = x_out_base + (x_in << 1);
                        let j1 = j0 + 1;
                        let off = 2 * x_in;
                        let add0_unr = merged_in_unreduced[off];
                        let add1_unr = merged_in_unreduced[off + 1];

                        if let Some(k0) = nonzero_indices[j0] {
                            local_unreduced[k0 as usize] += add0_unr;
                        }
                        if let Some(k1) = nonzero_indices[j1] {
                            local_unreduced[k1 as usize] += add1_unr;
                        }
                    }

                    for idx in 0..K {
                        if local_unreduced[idx] != F::Unreduced::<9>::zero() {
                            let reduced = F::from_montgomery_reduce::<9>(local_unreduced[idx]);
                            partial[idx] += e_out * reduced;
                        }
                    }

                    partial
                },
            )
            .reduce(
                || unsafe_allocate_zero_vec(K),
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            );

        drop(eq);

        Self {
            log_T: T.log_2(),
            polynomial,
            coefficient,
            eq_address_state,
            eq_cycle_state,
            G,
            H: Arc::new(RwLock::new(RaPolynomial::None)),
        }
    }
}

impl<F: JoltField> ClaimSource<F> for OneHotClaimSource<F> {
    fn coeff(&self) -> F {
        self.coefficient
    }

    fn compute_address_message(&self, round: usize) -> [F; 2] {
        let shared_eq_address = self.eq_address_state.read().unwrap();
        let K = self.polynomial.K;
        let log_K = K.log_2();

        if round < log_K {
            let m = round + 1;
            let B = &shared_eq_address.B;
            let F_table = &shared_eq_address.F;
            let G = &self.G;

            let unreduced_evals = (0..B.len() / 2)
                .into_par_iter()
                .map(|k_prime| {
                    let B_evals = B.sumcheck_evals_array::<2>(k_prime, BindingOrder::LowToHigh);
                    let inner_sum = G[k_prime << m..(k_prime + 1) << m]
                        .par_iter()
                        .enumerate()
                        .map(|(k, &G_k)| {
                            let k_m = k >> (m - 1);
                            let F_k = F_table[k % (1 << (m - 1))];
                            let G_times_F = G_k * F_k;

                            let eval_c0 = if k_m == 0 { G_times_F } else { F::zero() };
                            let eval_c2 = if k_m == 0 {
                                -G_times_F
                            } else {
                                G_times_F + G_times_F
                            };

                            [eval_c0, eval_c2]
                        })
                        .fold_with([F::zero(); 2], |running, new| {
                            [running[0] + new[0], running[1] + new[1]]
                        })
                        .reduce(
                            || [F::zero(); 2],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    [
                        B_evals[0].mul_unreduced::<9>(inner_sum[0]),
                        B_evals[1].mul_unreduced::<9>(inner_sum[1]),
                    ]
                })
                .reduce(
                    || [F::Unreduced::<9>::zero(); 2],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                );

            [
                F::from_montgomery_reduce(unreduced_evals[0]),
                F::from_montgomery_reduce(unreduced_evals[1]),
            ]
        } else {
            [F::zero(), F::zero()]
        }
    }

    fn bind_address(&mut self, r: F::Challenge, round: usize) {
        let K = self.polynomial.K;
        let log_K = K.log_2();

        let mut shared_eq_address = self.eq_address_state.write().unwrap();

        // Bind shared address state if not already bound
        if shared_eq_address.num_variables_bound <= round {
            shared_eq_address
                .B
                .bind_parallel(r, BindingOrder::LowToHigh);
            shared_eq_address.F.update(r);
            shared_eq_address.num_variables_bound += 1;
        }

        // Transition from G to H at the last address round
        if round == log_K - 1 {
            let mut lock = self.H.write().unwrap();
            if matches!(*lock, RaPolynomial::None) {
                *lock = RaPolynomial::new(
                    self.polynomial.nonzero_indices.clone(),
                    shared_eq_address.F.clone_values(),
                );
            }

            // Drop G in background
            let g = std::mem::take(&mut self.G);
            drop_in_background_thread(g);
        }
    }

    fn initialize_prefix_suffix_Q(
        &self,
        _q_buffer: &mut [F],
        _eq_suffix_evals: &[F],
        _prefix_n_vars: usize,
        _suffix_n_vars: usize,
    ) {
        // One-hot claims don't use the prefix-suffix optimization in the same way
        // They use the address phase instead
    }

    fn compute_linear_message(&self, round: usize, _eq_evals: &[F]) -> [F; 2] {
        let shared_eq_address = self.eq_address_state.read().unwrap();
        let shared_eq_cycle = self.eq_cycle_state.read().unwrap();
        let B = &shared_eq_address.B;
        let d_gruen = &shared_eq_cycle.D;
        let eq_r_address_claim = B.final_sumcheck_claim();
        let H = self.H.read().unwrap();

        let _ = round; // Used for tracking, but not in this computation

        let [gruen_eval_0] =
            d_gruen.par_fold_out_in_unreduced::<9, 1>(&|g| [H.get_bound_coeff(2 * g)]);

        // Get evaluations at 0 and 2 using Gruen polynomial
        let w = d_gruen.get_current_w();
        let current_scalar = d_gruen.get_current_scalar();
        let eq_eval_1 = current_scalar * w;
        let eq_eval_0 = current_scalar - eq_eval_1;
        let eq_m = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_m;

        // q(0) = gruen_eval_0, need q(2)
        // Linear interpolation: q(x) is linear, so q(2) = 2*q(1) - q(0)
        // But we need to compute q(1) from previous_claim first...
        // This is handled by the caller using from_evals_and_hint

        // For now, return the raw evaluations (will be scaled in compute_message)
        [eq_r_address_claim * gruen_eval_0, eq_r_address_claim * eq_eval_2]
    }

    fn bind_cycle(&mut self, r: F::Challenge, round: usize) {
        let mut shared_eq_cycle = self.eq_cycle_state.write().unwrap();

        // Bind shared cycle state if not already bound
        if shared_eq_cycle.num_variables_bound <= round {
            shared_eq_cycle.D.bind(r);
            shared_eq_cycle.num_variables_bound += 1;
        }
        drop(shared_eq_cycle);

        // Bind H
        let mut H = self.H.write().unwrap();
        let log_K = self.polynomial.K.log_2();
        if H.len().log_2() == self.log_T + log_K - round {
            H.bind_parallel(r, BindingOrder::LowToHigh);
        }
    }

    fn final_claim(&self) -> F {
        self.H.read().unwrap().final_sumcheck_claim()
    }
}

// ============================================================================
// InstructionLookupClaimSource
// ============================================================================

/// Instruction lookup claim source with optimized unreduced multiplication.
/// Uses fused `out + gamma * left + gamma^2 * right` computation.
#[derive(Allocative)]
pub struct InstructionLookupClaimSource<F: JoltField> {
    #[allocative(skip)]
    pub trace: Arc<Vec<Cycle>>,
    pub gamma: F,
    pub gamma_sqr: F,
    pub coefficient: F,
    /// Materialized polynomials for linear phase (created during transition)
    pub lookup_output_poly: Option<MultilinearPolynomial<F>>,
    pub left_lookup_operand_poly: Option<MultilinearPolynomial<F>>,
    pub right_lookup_operand_poly: Option<MultilinearPolynomial<F>>,
    pub eq_poly: Option<MultilinearPolynomial<F>>,
}

impl<F: JoltField> InstructionLookupClaimSource<F> {
    pub fn new(trace: Arc<Vec<Cycle>>, gamma: F, gamma_sqr: F, coefficient: F) -> Self {
        Self {
            trace,
            gamma,
            gamma_sqr,
            coefficient,
            lookup_output_poly: None,
            left_lookup_operand_poly: None,
            right_lookup_operand_poly: None,
            eq_poly: None,
        }
    }

    /// Transition to linear phase by materializing the polynomials
    #[tracing::instrument(skip_all, name = "InstructionLookupClaimSource::transition_to_linear")]
    pub fn transition_to_linear(
        &mut self,
        sumcheck_challenges: &[F::Challenge],
        r_spartan: &OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let n_remaining_rounds = r_spartan.len() - sumcheck_challenges.len();
        let r_prefix: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();

        let eq_evals = EqPolynomial::evals(&r_prefix.r);
        let mut lookup_output_poly = unsafe_allocate_zero_vec(1 << n_remaining_rounds);
        let mut left_lookup_operand_poly = unsafe_allocate_zero_vec(1 << n_remaining_rounds);
        let mut right_lookup_operand_poly = unsafe_allocate_zero_vec(1 << n_remaining_rounds);

        (
            &mut lookup_output_poly,
            &mut left_lookup_operand_poly,
            &mut right_lookup_operand_poly,
            self.trace.par_chunks(eq_evals.len()),
        )
            .into_par_iter()
            .for_each(
                |(
                    lookup_output_eval,
                    left_lookup_operand_eval,
                    right_lookup_operand_eval,
                    trace_chunk,
                )| {
                    let mut lookup_output_eval_unreduced = F::Unreduced::<6>::zero();
                    let mut left_lookup_operand_eval_unreduced = F::Unreduced::<6>::zero();
                    let mut right_lookup_operand_eval_unreduced = F::Unreduced::<7>::zero();

                    for (i, cycle) in trace_chunk.iter().enumerate() {
                        let (left_lookup, right_lookup) =
                            LookupQuery::<XLEN>::to_lookup_operands(cycle);
                        let lookup_output = LookupQuery::<XLEN>::to_lookup_output(cycle);

                        lookup_output_eval_unreduced +=
                            eq_evals[i].mul_u64_unreduced(lookup_output);
                        left_lookup_operand_eval_unreduced +=
                            eq_evals[i].mul_u64_unreduced(left_lookup);
                        right_lookup_operand_eval_unreduced +=
                            eq_evals[i].mul_u128_unreduced(right_lookup);
                    }

                    *lookup_output_eval = F::from_barrett_reduce(lookup_output_eval_unreduced);
                    *left_lookup_operand_eval =
                        F::from_barrett_reduce(left_lookup_operand_eval_unreduced);
                    *right_lookup_operand_eval =
                        F::from_barrett_reduce(right_lookup_operand_eval_unreduced);
                },
            );

        let (r_hi, r_lo) = r_spartan.split_at(r_spartan.len() / 2);
        let eq_prefix_eval = EqPolynomial::mle_endian(&r_prefix, &r_lo);
        let eq_suffix_evals = EqPolynomial::evals_parallel(&r_hi.r, Some(eq_prefix_eval));

        self.lookup_output_poly = Some(lookup_output_poly.into());
        self.left_lookup_operand_poly = Some(left_lookup_operand_poly.into());
        self.right_lookup_operand_poly = Some(right_lookup_operand_poly.into());
        self.eq_poly = Some(eq_suffix_evals.into());
    }
}

impl<F: JoltField> ClaimSource<F> for InstructionLookupClaimSource<F> {
    fn coeff(&self) -> F {
        self.coefficient
    }

    fn compute_address_message(&self, _round: usize) -> [F; 2] {
        // Instruction lookups don't have address phase
        [F::zero(), F::zero()]
    }

    fn bind_address(&mut self, _r: F::Challenge, _round: usize) {
        // No-op
    }

    fn initialize_prefix_suffix_Q(
        &self,
        q_buffer: &mut [F],
        eq_suffix_evals: &[F],
        prefix_n_vars: usize,
        suffix_n_vars: usize,
    ) {
        const BLOCK_SIZE: usize = 32;
        let gamma = self.gamma;
        let gamma_sqr = self.gamma_sqr;
        let coeff = self.coefficient;
        let trace = &self.trace;

        q_buffer
            .par_chunks_mut(BLOCK_SIZE)
            .enumerate()
            .for_each(|(chunk_i, q_chunk)| {
                let mut q_lookup_output = [F::Unreduced::<6>::zero(); BLOCK_SIZE];
                let mut q_left_lookup_operand = [F::Unreduced::<6>::zero(); BLOCK_SIZE];
                let mut q_right_lookup_operand = [F::Unreduced::<7>::zero(); BLOCK_SIZE];

                for x_hi in 0..(1 << suffix_n_vars) {
                    for i in 0..q_chunk.len() {
                        let x_lo = chunk_i * BLOCK_SIZE + i;
                        let x = x_lo + (x_hi << prefix_n_vars);
                        let cycle = &trace[x];
                        let (left_lookup, right_lookup) =
                            LookupQuery::<XLEN>::to_lookup_operands(cycle);
                        let lookup_output = LookupQuery::<XLEN>::to_lookup_output(cycle);

                        q_lookup_output[i] +=
                            eq_suffix_evals[x_hi].mul_u64_unreduced(lookup_output);
                        q_left_lookup_operand[i] +=
                            eq_suffix_evals[x_hi].mul_u64_unreduced(left_lookup);
                        q_right_lookup_operand[i] +=
                            eq_suffix_evals[x_hi].mul_u128_unreduced(right_lookup);
                    }
                }

                for (i, q) in q_chunk.iter_mut().enumerate() {
                    let fused = F::from_barrett_reduce(q_lookup_output[i])
                        + gamma * F::from_barrett_reduce(q_left_lookup_operand[i])
                        + gamma_sqr * F::from_barrett_reduce(q_right_lookup_operand[i]);
                    *q += coeff * fused;
                }
            });
    }

    fn compute_linear_message(&self, _round: usize, _eq_evals: &[F]) -> [F; 2] {
        let lookup_output_poly = self.lookup_output_poly.as_ref().unwrap();
        let left_lookup_operand_poly = self.left_lookup_operand_poly.as_ref().unwrap();
        let right_lookup_operand_poly = self.right_lookup_operand_poly.as_ref().unwrap();
        let eq_poly = self.eq_poly.as_ref().unwrap();

        let half_n = lookup_output_poly.len() / 2;
        let gamma = self.gamma;
        let gamma_sqr = self.gamma_sqr;

        let (eval_0, eval_2) = (0..half_n)
            .into_par_iter()
            .map(|j| {
                let lookup_output_evals =
                    lookup_output_poly.sumcheck_evals_array::<2>(j, BindingOrder::LowToHigh);
                let left_lookup_operand_evals =
                    left_lookup_operand_poly.sumcheck_evals_array::<2>(j, BindingOrder::LowToHigh);
                let right_lookup_operand_evals =
                    right_lookup_operand_poly.sumcheck_evals_array::<2>(j, BindingOrder::LowToHigh);
                let eq_evals = eq_poly.sumcheck_evals_array::<2>(j, BindingOrder::LowToHigh);

                let fused_0 = lookup_output_evals[0]
                    + gamma * left_lookup_operand_evals[0]
                    + gamma_sqr * right_lookup_operand_evals[0];
                let fused_2 = lookup_output_evals[1]
                    + gamma * left_lookup_operand_evals[1]
                    + gamma_sqr * right_lookup_operand_evals[1];

                (eq_evals[0] * fused_0, eq_evals[1] * fused_2)
            })
            .reduce(
                || (F::zero(), F::zero()),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            );

        [eval_0, eval_2]
    }

    fn bind_cycle(&mut self, r: F::Challenge, _round: usize) {
        if let Some(ref mut poly) = self.lookup_output_poly {
            poly.bind_parallel(r, BindingOrder::LowToHigh);
        }
        if let Some(ref mut poly) = self.left_lookup_operand_poly {
            poly.bind_parallel(r, BindingOrder::LowToHigh);
        }
        if let Some(ref mut poly) = self.right_lookup_operand_poly {
            poly.bind_parallel(r, BindingOrder::LowToHigh);
        }
        if let Some(ref mut poly) = self.eq_poly {
            poly.bind_parallel(r, BindingOrder::LowToHigh);
        }
    }

    fn final_claim(&self) -> F {
        let out = self.lookup_output_poly.as_ref().unwrap().final_sumcheck_claim();
        let left = self.left_lookup_operand_poly.as_ref().unwrap().final_sumcheck_claim();
        let right = self.right_lookup_operand_poly.as_ref().unwrap().final_sumcheck_claim();
        out + self.gamma * left + self.gamma_sqr * right
    }
}

// ============================================================================
// Prover state
// ============================================================================

/// Prover state for the generic claim reduction sumcheck.
///
/// Unlike the previous implementation, we don't use a `Finished` state.
/// Instead, we track rounds explicitly and ensure we handle exactly `num_rounds()` calls.
#[derive(Allocative)]
pub enum ClaimReductionSumcheckProver<F: JoltField> {
    /// Address phase (for one-hot claims only)
    Address(AddressPhaseProver<F>),
    /// Cycle phase 1: prefix-suffix optimization
    CyclePrefixSuffix(CyclePrefixSuffixProver<F>),
    /// Cycle phase 2: linear sumcheck
    CycleLinear(CycleLinearProver<F>),
}

#[derive(Allocative)]
pub struct AddressPhaseProver<F: JoltField> {
    #[allocative(skip)]
    claims: Vec<Box<dyn ClaimSource<F>>>,
    params: ClaimReductionSumcheckParams<F>,
    round: usize,
}

#[derive(Allocative)]
pub struct CyclePrefixSuffixProver<F: JoltField> {
    #[allocative(skip)]
    claims: Vec<Box<dyn ClaimSource<F>>>,
    params: ClaimReductionSumcheckParams<F>,
    /// P = eq(r_lo, x_lo)
    P: MultilinearPolynomial<F>,
    /// Q = \sum_{x_hi} (coeff_i * Claims_i(x_lo, x_hi)) * eq(r_hi, x_hi)
    Q: MultilinearPolynomial<F>,
    /// Number of prefix rounds completed
    round: usize,
    /// Number of prefix variables (fixed at initialization)
    prefix_len: usize,
    /// Sumcheck challenges from prefix-suffix phase (for transition to linear)
    sumcheck_challenges: Vec<F::Challenge>,
}

#[derive(Allocative)]
pub struct CycleLinearProver<F: JoltField> {
    #[allocative(skip)]
    claims: Vec<Box<dyn ClaimSource<F>>>,
    params: ClaimReductionSumcheckParams<F>,
    /// Current eq polynomial evaluations (with Gruen optimization)
    eq_evals: GruenSplitEqPolynomial<F>,
    /// Number of linear rounds completed
    round: usize,
    /// Total number of linear rounds (suffix_len)
    total_linear_rounds: usize,
}

impl<F: JoltField> ClaimReductionSumcheckProver<F> {
    /// Create a new claim reduction sumcheck prover.
    pub fn new(
        claims: Vec<Box<dyn ClaimSource<F>>>,
        params: ClaimReductionSumcheckParams<F>,
    ) -> Self {
        if params.num_address_vars > 0 {
            Self::Address(AddressPhaseProver {
                claims,
                params,
                round: 0,
            })
        } else {
            Self::transition_to_prefix_suffix(claims, params, vec![])
        }
    }

    fn transition_to_prefix_suffix(
        claims: Vec<Box<dyn ClaimSource<F>>>,
        params: ClaimReductionSumcheckParams<F>,
        sumcheck_challenges: Vec<F::Challenge>,
    ) -> Self {
        let n_cycle = params.num_cycle_vars;
        let prefix_len = n_cycle / 2;
        let suffix_len = n_cycle - prefix_len;

        if prefix_len == 0 {
            return Self::transition_to_linear(claims, params, sumcheck_challenges);
        }

        // Split r_cycle into suffix (high, first half) and prefix (low, second half)
        // Note: r_cycle is in big-endian order for the point, so first half is "high" bits
        let (r_hi, r_lo) = params.r_cycle.split_at(suffix_len);

        let eq_prefix_evals = EqPolynomial::evals(r_lo);
        let eq_suffix_evals = EqPolynomial::evals(r_hi);

        let mut q_buffer = unsafe_allocate_zero_vec(1 << prefix_len);

        // Each claim adds its contribution (scaled by its coefficient) to q_buffer
        for claim in &claims {
            claim.initialize_prefix_suffix_Q(&mut q_buffer, &eq_suffix_evals, prefix_len, suffix_len);
        }

        Self::CyclePrefixSuffix(CyclePrefixSuffixProver {
            claims,
            params,
            P: MultilinearPolynomial::from(eq_prefix_evals),
            Q: MultilinearPolynomial::from(q_buffer),
            round: 0,
            prefix_len,
            sumcheck_challenges,
        })
    }

    fn transition_to_linear(
        claims: Vec<Box<dyn ClaimSource<F>>>,
        params: ClaimReductionSumcheckParams<F>,
        sumcheck_challenges: Vec<F::Challenge>,
    ) -> Self {
        let n_cycle = params.num_cycle_vars;
        let prefix_len = n_cycle / 2;
        let suffix_len = n_cycle - prefix_len;

        // Build the eq polynomial for the suffix, incorporating the prefix challenges
        // The eq polynomial should be: eq(r_cycle, (prefix_challenges, x_suffix))
        // where prefix_challenges have already been bound
        let (r_hi, r_lo) = params.r_cycle.split_at(suffix_len);

        // Compute eq(prefix_challenges, r_lo) as a scalar
        let eq_prefix_eval = if !sumcheck_challenges.is_empty() {
            EqPolynomial::mle(&sumcheck_challenges, r_lo)
        } else {
            F::one()
        };

        // Compute eq(r_hi, x_suffix) with scaling by eq_prefix_eval
        let eq_evals = GruenSplitEqPolynomial::new_with_scaling(
            r_hi,
            BindingOrder::LowToHigh,
            Some(eq_prefix_eval),
        );

        Self::CycleLinear(CycleLinearProver {
            claims,
            params,
            eq_evals,
            round: 0,
            total_linear_rounds: suffix_len,
        })
    }

    /// Compute the prover's message for a specific round.
    /// This is an inherent method that mirrors the trait method for easier testing.
    pub fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        self.compute_message_impl(round, previous_claim)
    }

    /// Ingest a verifier challenge.
    /// This is an inherent method that mirrors the trait method for easier testing.
    pub fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.ingest_challenge_impl(r_j, round)
    }

    fn compute_message_impl(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        match self {
            Self::Address(p) => {
                let (eval_0, eval_2) = p
                    .claims
                    .par_iter()
                    .map(|c| {
                        let evals = c.compute_address_message(p.round);
                        let coeff = c.coeff();
                        (coeff * evals[0], coeff * evals[1])
                    })
                    .reduce(
                        || (F::zero(), F::zero()),
                        |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
                    );
                UniPoly::from_evals_and_hint(previous_claim, &[eval_0, eval_2])
            }
            Self::CyclePrefixSuffix(p) => {
                let (val_0, val_2) = (0..p.P.len() / 2)
                    .into_par_iter()
                    .map(|j| {
                        let p_evals = p.P.sumcheck_evals_array::<2>(j, BindingOrder::LowToHigh);
                        let q_evals = p.Q.sumcheck_evals_array::<2>(j, BindingOrder::LowToHigh);
                        let p0 = p_evals[0];
                        let p2 = p_evals[1];
                        let q0 = q_evals[0];
                        let q2 = q_evals[1];
                        (p0 * q0, p2 * q2)
                    })
                    .reduce(
                        || (F::zero(), F::zero()),
                        |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
                    );
                UniPoly::from_evals_and_hint(previous_claim, &[val_0, val_2])
            }
            Self::CycleLinear(p) => {
                // Get the full eq polynomial (merged from Gruen split representation)
                let eq_evals: Vec<F> = p.eq_evals.merge().Z.to_vec();
                let (eval_0, eval_2) = p
                    .claims
                    .par_iter()
                    .map(|c| {
                        let evals = c.compute_linear_message(p.round, &eq_evals);
                        let coeff = c.coeff();
                        (coeff * evals[0], coeff * evals[1])
                    })
                    .reduce(
                        || (F::zero(), F::zero()),
                        |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
                    );
                UniPoly::from_evals_and_hint(previous_claim, &[eval_0, eval_2])
            }
        }
    }

    fn ingest_challenge_impl(&mut self, r_j: F::Challenge, _round: usize) {
        match self {
            Self::Address(p) => {
                for c in &mut p.claims {
                    c.bind_address(r_j, p.round);
                }
                p.round += 1;
                if p.round == p.params.num_address_vars {
                    let claims = std::mem::take(&mut p.claims);
                    let params = std::mem::take(&mut p.params);
                    *self = Self::transition_to_prefix_suffix(claims, params, vec![]);
                }
            }
            Self::CyclePrefixSuffix(p) => {
                // Bind P and Q polynomials (the compressed representations)
                p.P.bind_parallel(r_j, BindingOrder::LowToHigh);
                p.Q.bind_parallel(r_j, BindingOrder::LowToHigh);

                // Also bind the claim sources' polynomials so they're ready for linear phase
                for c in &mut p.claims {
                    c.bind_cycle(r_j, p.round);
                }

                p.round += 1;
                p.sumcheck_challenges.push(r_j);
                if p.round == p.prefix_len {
                    let claims = std::mem::take(&mut p.claims);
                    let params = std::mem::take(&mut p.params);
                    let challenges = std::mem::take(&mut p.sumcheck_challenges);
                    *self = Self::transition_to_linear(claims, params, challenges);
                }
            }
            Self::CycleLinear(p) => {
                p.eq_evals.bind(r_j);
                for c in &mut p.claims {
                    c.bind_cycle(r_j, p.round);
                }
                p.round += 1;
            }
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ClaimReductionSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        match self {
            Self::Address(p) => &p.params,
            Self::CyclePrefixSuffix(p) => &p.params,
            Self::CycleLinear(p) => &p.params,
        }
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        self.compute_message_impl(round, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.ingest_challenge_impl(r_j, round)
    }

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // Override in specific implementations if needed
    }
}

// ============================================================================
// Verifier
// ============================================================================

/// Describes the type and parameters of a claim being verified.
#[derive(Clone, Debug)]
pub enum ClaimKind {
    /// Dense polynomial opening
    Dense,
    /// One-hot polynomial opening with address variables
    OneHot { num_address_vars: usize },
    /// Instruction lookup with gamma parameters
    InstructionLookup,
}

/// Descriptor for a claim on the verifier side.
#[derive(Clone)]
pub struct ClaimDescriptor<F: JoltField> {
    pub kind: ClaimKind,
    pub coefficient: F,
    pub opening_point: OpeningPoint<BIG_ENDIAN, F>,
    pub claim: F,
}

/// Verifier for claim reduction sumcheck.
pub struct ClaimReductionSumcheckVerifier<F: JoltField> {
    params: ClaimReductionSumcheckParams<F>,
    claims: Vec<ClaimDescriptor<F>>,
}

impl<F: JoltField> ClaimReductionSumcheckVerifier<F> {
    pub fn new(params: ClaimReductionSumcheckParams<F>, claims: Vec<ClaimDescriptor<F>>) -> Self {
        Self { params, claims }
    }
}

impl<F: JoltField, T: Transcript> crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>
    for ClaimReductionSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        _accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Compute the expected output claim by evaluating eq polynomials
        // and combining with the per-claim coefficients and claims

        let mut total = F::zero();

        for descriptor in &self.claims {
            // Compute eq(opening_point, sumcheck_challenges)
            let eq_eval = EqPolynomial::mle(&descriptor.opening_point.r, sumcheck_challenges);

            // The contribution is: coefficient * claim * eq_eval
            total += descriptor.coefficient * descriptor.claim * eq_eval;
        }

        total
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // Override in specific implementations if needed
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::dense_mlpoly::DensePolynomial;
    use ark_bn254::Fr;
    use ark_std::{test_rng, One};
    use rand_core::RngCore;

    /// Test that DenseClaimSource produces correct sumcheck messages
    #[test]
    fn test_dense_claim_source_linear_phase() {
        const LOG_N: usize = 4;
        let n = 1 << LOG_N;
        let mut rng = test_rng();

        // Create a random polynomial
        let poly_coeffs: Vec<Fr> = (0..n).map(|_| Fr::from(rng.next_u64())).collect();
        let poly = DensePolynomial::new(poly_coeffs.clone());

        // Create random opening point
        let r_cycle: Vec<<Fr as JoltField>::Challenge> =
            (0..LOG_N).map(|_| <Fr as JoltField>::Challenge::random(&mut rng)).collect();

        // Create eq state
        let eq_cycle_state = Arc::new(RwLock::new(EqCycleState::new(&r_cycle)));

        // Create claim source
        let mut claim_source = DenseClaimSource::new(
            MultilinearPolynomial::from(poly_coeffs.clone()),
            Fr::one(),
            eq_cycle_state,
        );

        // Compute the expected claim: sum_x P(x) * eq(r, x)
        let eq_evals = EqPolynomial::evals(&r_cycle);
        let expected_input_claim: Fr = (0..n).map(|i| poly_coeffs[i] * eq_evals[i]).sum();

        // Run through linear phase rounds
        let mut previous_claim = expected_input_claim;
        let mut eq_poly = DensePolynomial::new(eq_evals);

        let mut sumcheck_challenges = vec![];
        for round in 0..LOG_N {
            // Get full eq evals
            let eq_evals: Vec<Fr> = eq_poly.Z.to_vec();
            let half = eq_poly.len() / 2;

            // Compute message
            let [eval_0, eval_2] = claim_source.compute_linear_message(round, &eq_evals);

            // Verify eval_0: sum over i of P[2*i] * eq[2*i]
            let expected_0: Fr = (0..half).map(|i| claim_source.poly.get_bound_coeff(2*i) * eq_evals[2*i]).sum();
            assert_eq!(eval_0, expected_0, "round {round}: eval_0 mismatch");

            // Build univariate and verify
            let uni = UniPoly::from_evals_and_hint(previous_claim, &[eval_0, eval_2]);
            assert_eq!(uni.eval_at_zero() + uni.eval_at_one(), previous_claim);

            // Generate random challenge
            let r = <Fr as JoltField>::Challenge::random(&mut rng);

            // Update claim
            previous_claim = uni.evaluate(&r);
            sumcheck_challenges.push(r);

            // Bind
            claim_source.bind_cycle(r, round);
            eq_poly.bind_parallel(r, BindingOrder::LowToHigh);
        }

        // Final claim should match: P(challenges) where challenges are the sumcheck challenges
        // Compute expected value by manually binding the polynomial
        let mut expected_poly = DensePolynomial::new(poly_coeffs.clone());
        for r in &sumcheck_challenges {
            expected_poly.bind_parallel(*r, BindingOrder::LowToHigh);
        }
        let expected = expected_poly.Z[0];
        assert_eq!(claim_source.final_claim(), expected);
    }

    /// Test prefix-suffix sumcheck correctness
    #[test]
    fn test_prefix_suffix_sumcheck() {
        const LOG_N: usize = 6;
        let n = 1 << LOG_N;
        let mut rng = test_rng();

        // Create a random polynomial
        let poly_coeffs: Vec<Fr> = (0..n).map(|_| Fr::from(rng.next_u64())).collect();

        // Create random opening point
        let r_cycle: Vec<<Fr as JoltField>::Challenge> =
            (0..LOG_N).map(|_| <Fr as JoltField>::Challenge::random(&mut rng)).collect();

        let eq_cycle_state = Arc::new(RwLock::new(EqCycleState::new(&r_cycle)));

        let claim_source = DenseClaimSource::new(
            MultilinearPolynomial::from(poly_coeffs.clone()),
            Fr::one(),
            eq_cycle_state,
        );

        // Create params
        let eq_evals: Vec<Fr> = EqPolynomial::evals(&r_cycle);
        let input_claim: Fr = (0..n).map(|i| -> Fr { poly_coeffs[i] * eq_evals[i] }).sum();

        let params = ClaimReductionSumcheckParams {
            num_address_vars: 0,
            num_cycle_vars: LOG_N,
            r_address: vec![],
            r_cycle: r_cycle.clone(),
            degree: DEGREE_BOUND,
            input_claim,
        };

        let claims: Vec<Box<dyn ClaimSource<Fr>>> = vec![Box::new(claim_source)];
        let mut prover = ClaimReductionSumcheckProver::new(claims, params);

        // Run sumcheck
        let mut previous_claim = input_claim;
        let mut sumcheck_challenges = vec![];

        for round in 0..LOG_N {
            let uni = prover.compute_message(round, previous_claim);

            // Verify degree bound
            assert!(uni.degree() <= DEGREE_BOUND);

            // Verify sum
            assert_eq!(
                uni.eval_at_zero() + uni.eval_at_one(),
                previous_claim,
                "round {round}: sum check failed"
            );

            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            previous_claim = uni.evaluate(&r);
            sumcheck_challenges.push(r);
            prover.ingest_challenge(r, round);
        }

        // Verify final claim
        // The sumcheck proves: sum_x P(x) * eq(r_cycle, x) = P(challenges) * eq(r_cycle, challenges)
        // Compute expected poly value by manually binding
        let mut expected_poly = DensePolynomial::new(poly_coeffs.clone());
        for r in &sumcheck_challenges {
            expected_poly.bind_parallel(*r, BindingOrder::LowToHigh);
        }
        let expected_poly_eval = expected_poly.Z[0];

        // Compute expected eq value by manually binding
        let mut expected_eq = DensePolynomial::<Fr>::new(EqPolynomial::evals(&r_cycle));
        for r in &sumcheck_challenges {
            expected_eq.bind_parallel(*r, BindingOrder::LowToHigh);
        }
        let eq_eval = expected_eq.Z[0];

        // The final claim should be P(challenges) * eq(r_cycle, challenges)
        assert_eq!(previous_claim, expected_poly_eval * eq_eval);
    }
}
