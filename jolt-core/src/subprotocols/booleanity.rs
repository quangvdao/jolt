//! Booleanity Sumcheck
//!
//! This module implements a single booleanity sumcheck that handles all three families:
//! - Instruction RA polynomials
//! - Bytecode RA polynomials  
//! - RAM RA polynomials
//!
//! By combining them into a single sumcheck, all families share the same `r_address` and `r_cycle`,
//! which is required by the HammingWeightClaimReduction sumcheck in Stage 7.
//!
//! ## Sumcheck Relation
//!
//! The booleanity sumcheck proves:
//! ```text
//! 0 = Σ_{k,j} eq(r_address, k) · eq(r_cycle, j) · Σ_i γ_i · (ra_i(k,j)² - ra_i(k,j))
//! ```
//!
//! Where i ranges over all RA polynomials from all three families.

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use rayon::prelude::*;
use std::iter::zip;

use common::jolt_device::MemoryLayout;
use tracer::instruction::Cycle;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::BindingOrder,
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        shared_ra_polys::{
            assert_ra_bounds, compute_all_G_and_ra_indices, RaIndices, SharedRaPolynomials,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{expanding_table::ExpandingTable, thread::drop_in_background_thread},
    zkvm::{
        bytecode::BytecodePreprocessing,
        config::OneHotParams,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};

/// Degree bound of the sumcheck round polynomials.
const DEGREE_BOUND: usize = 3;

/// Parameters for the booleanity sumcheck.
#[derive(Allocative, Clone)]
pub struct BooleanitySumcheckParams<F: JoltField> {
    /// Log of chunk size (shared across all families)
    pub log_k_chunk: usize,
    /// Log of trace length
    pub log_t: usize,
    /// Single batching challenge γ.
    /// We derive per-polynomial batching coefficients as \( \gamma^{2i} \) for i = 0, 1, ...
    pub gamma: F::Challenge,
    /// Per-polynomial batching coefficients \( \gamma^{2i} \) (in the base field).
    pub gamma_powers_square: Vec<F>,
    /// Address binding point (shared across all families)
    pub r_address: Vec<F::Challenge>,
    /// Cycle binding point (shared across all families)
    pub r_cycle: Vec<F::Challenge>,
    /// Polynomial types for all families
    pub polynomial_types: Vec<CommittedPolynomial>,
    /// OneHotParams for SharedRaPolynomials
    pub one_hot_params: OneHotParams,
}

impl<F: JoltField> SumcheckInstanceParams<F> for BooleanitySumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_k_chunk + self.log_t
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = sumcheck_challenges.to_vec();
        opening_point[..self.log_k_chunk].reverse();
        opening_point[self.log_k_chunk..].reverse();
        opening_point.into()
    }
}

impl<F: JoltField> BooleanitySumcheckParams<F> {
    /// Create booleanity params by taking r_cycle and r_address from Stage 5.
    ///
    /// Stage 5 produces challenges in order: address (LOG_K_INSTRUCTION) => cycle (log_t).
    /// We extract the last log_k_chunk challenges for r_address and all of r_cycle.
    /// (this is a somewhat arbitrary choice; any prior randomness would work)
    pub fn new(
        log_t: usize,
        one_hot_params: &OneHotParams,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let log_k_chunk = one_hot_params.log_k_chunk;
        let instruction_d = one_hot_params.instruction_d;
        let bytecode_d = one_hot_params.bytecode_d;
        let ram_d = one_hot_params.ram_d;
        let total_d = instruction_d + bytecode_d + ram_d;
        let log_k_instruction = one_hot_params.lookups_ra_virtual_log_k_chunk;

        // Get Stage 5 opening point: order is address (LOG_K_INSTRUCTION) => cycle (log_t)
        // The stored point is in BIG_ENDIAN format (after normalize_opening_point reversed it)
        let (stage5_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa(0),
            SumcheckId::InstructionReadRaf,
        );

        // Extract r_address and r_cycle.
        //
        // NOTE: `stage5_point.r` is stored in BIG_ENDIAN format (each segment was reversed by
        // `normalize_opening_point`). For internal eq evaluations we want LowToHigh (LE) order
        // because `GruenSplitEqPolynomial` is instantiated with `BindingOrder::LowToHigh`.
        debug_assert!(
            stage5_point.r.len() == log_k_instruction + log_t,
            "InstructionReadRaf opening point length mismatch: got {}, expected {} (= log_k_instruction {} + log_t {})",
            stage5_point.r.len(),
            log_k_instruction + log_t,
            log_k_instruction,
            log_t
        );

        // Address segment: BE -> LE
        let mut stage5_addr = stage5_point.r[..log_k_instruction].to_vec();
        stage5_addr.reverse();

        // Cycle segment: BE -> LE
        let mut r_cycle = stage5_point.r[log_k_instruction..].to_vec();
        r_cycle.reverse();

        // Take the last `log_k_chunk` address challenges (in LE order). If Stage 5 provided fewer,
        // fall back to sampling additional challenges so prover/verifier stay in sync.
        let r_address = if stage5_addr.len() >= log_k_chunk {
            stage5_addr[stage5_addr.len() - log_k_chunk..].to_vec()
        } else {
            let mut r = stage5_addr;
            let extra = transcript.challenge_vector_optimized::<F>(log_k_chunk - r.len());
            r.extend(extra);
            r
        };

        // Build polynomial types and family mapping
        let mut polynomial_types = Vec::with_capacity(total_d);

        for i in 0..instruction_d {
            polynomial_types.push(CommittedPolynomial::InstructionRa(i));
        }
        for i in 0..bytecode_d {
            polynomial_types.push(CommittedPolynomial::BytecodeRa(i));
        }
        for i in 0..ram_d {
            polynomial_types.push(CommittedPolynomial::RamRa(i));
        }

        // Sample a single batching challenge γ, and derive per-polynomial weights γ^{2i}.
        let mut gamma = transcript.challenge_scalar_optimized::<F>();
        let mut gamma_f: F = gamma.into();
        // Avoid the degenerate gamma=0 case (vanishing weights + non-invertible scaling).
        if gamma_f.is_zero() {
            gamma = F::Challenge::from(1_u128);
            gamma_f = gamma.into();
        }

        // Compute gamma_powers_square (verifier needs these for expected_output_claim)
        let gamma_sq = gamma_f.square();
        let mut gamma_powers_square = Vec::with_capacity(total_d);
        let mut gamma2_i = F::one();
        for _ in 0..total_d {
            gamma_powers_square.push(gamma2_i);
            gamma2_i *= gamma_sq;
        }

        Self {
            log_k_chunk,
            log_t,
            gamma,
            gamma_powers_square,
            r_address,
            r_cycle,
            polynomial_types,
            one_hot_params: one_hot_params.clone(),
        }
    }
}

/// Booleanity Sumcheck Prover.
#[derive(Allocative)]
pub struct BooleanitySumcheckProver<F: JoltField> {
    /// Per-polynomial powers γ^i (in the base field).
    /// Used to pre-scale the address eq tables for phase 2.
    gamma_powers: Vec<F>,
    /// Per-polynomial inverse powers γ^{-i} (in the base field).
    /// Used to unscale cached committed-polynomial openings.
    gamma_powers_inv: Vec<F>,
    /// B: split-eq over address-chunk variables (phase 1, LowToHigh).
    B: GruenSplitEqPolynomial<F>,
    /// D: split-eq over time/cycle variables (phase 2, LowToHigh).
    D: GruenSplitEqPolynomial<F>,
    /// G[i][k] = Σ_j eq(r_cycle, j) · ra_i(k, j) for all RA polynomials
    G: Vec<Vec<F>>,
    /// Shared H polynomials for phase 2 (initialized at transition)
    H: Option<SharedRaPolynomials<F>>,
    /// F: Expanding table for phase 1
    F: ExpandingTable<F>,
    /// eq(r_address, r_address) at end of phase 1
    eq_r_r: F,
    /// RA indices (non-transposed, one per cycle)
    ra_indices: Vec<RaIndices>,
    pub params: BooleanitySumcheckParams<F>,
}

impl<F: JoltField> BooleanitySumcheckProver<F> {
    /// Initialize a BooleanitySumcheckProver with all three families.
    ///
    /// All heavy computation is done here:
    /// - Compute G polynomials and RA indices in a single pass over the trace
    /// - Initialize split-eq polynomials for address (B) and cycle (D) variables
    /// - Initialize expanding table for phase 1
    #[tracing::instrument(skip_all, name = "BooleanitySumcheckProver::initialize")]
    pub fn initialize(
        params: BooleanitySumcheckParams<F>,
        trace: &[Cycle],
        bytecode: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
    ) -> Self {
        // Compute G and RA indices in a single pass over the trace
        let (G, ra_indices) = compute_all_G_and_ra_indices::<F>(
            trace,
            bytecode,
            memory_layout,
            &params.one_hot_params,
            &params.r_cycle,
        );

        // Initialize split-eq polynomials for address and cycle variables
        let B = GruenSplitEqPolynomial::new(&params.r_address, BindingOrder::LowToHigh);
        let D = GruenSplitEqPolynomial::new(&params.r_cycle, BindingOrder::LowToHigh);

        // Initialize expanding table for phase 1
        let k_chunk = 1 << params.log_k_chunk;
        let mut F_table = ExpandingTable::new(k_chunk, BindingOrder::LowToHigh);
        F_table.reset(F::one());

        // Compute prover-only fields: gamma_powers (γ^i) and gamma_powers_inv (γ^{-i})
        let num_polys = params.polynomial_types.len();
        let gamma_f: F = params.gamma.into();
        let mut gamma_powers = Vec::with_capacity(num_polys);
        let mut gamma_powers_inv = Vec::with_capacity(num_polys);
        let mut rho_i = F::one();
        for _ in 0..num_polys {
            gamma_powers.push(rho_i);
            gamma_powers_inv.push(
                rho_i
                    .inverse()
                    .expect("gamma_powers[i] is nonzero (gamma != 0)"),
            );
            rho_i *= gamma_f;
        }

        Self {
            gamma_powers,
            gamma_powers_inv,
            B,
            D,
            G,
            ra_indices,
            H: None,
            F: F_table,
            eq_r_r: F::zero(),
            params,
        }
    }

    fn compute_phase1_message(&self, round: usize, previous_claim: F) -> UniPoly<F> {
        let m = round + 1;
        let B = &self.B;
        let N = self.params.polynomial_types.len();

        // Compute quadratic coefficients via generic split-eq fold
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = B
            .par_fold_out_in_unreduced::<9, { DEGREE_BOUND - 1 }>(&|k_prime| {
                let coeffs = (0..N)
                    .into_par_iter()
                    .map(|i| {
                        let G_i = &self.G[i];
                        let inner_sum = G_i[k_prime << m..(k_prime + 1) << m]
                            .par_iter()
                            .enumerate()
                            .map(|(k, &G_k)| {
                                let k_m = k >> (m - 1);
                                let F_k = self.F[k & ((1 << (m - 1)) - 1)];
                                let G_times_F = G_k * F_k;

                                let eval_infty = G_times_F * F_k;
                                let eval_0 = if k_m == 0 {
                                    eval_infty - G_times_F
                                } else {
                                    F::zero()
                                };
                                [eval_0, eval_infty]
                            })
                            .fold_with(
                                [F::Unreduced::<5>::zero(); DEGREE_BOUND - 1],
                                |running, new| {
                                    [
                                        running[0] + new[0].as_unreduced_ref(),
                                        running[1] + new[1].as_unreduced_ref(),
                                    ]
                                },
                            )
                            .reduce(
                                || [F::Unreduced::zero(); DEGREE_BOUND - 1],
                                |running, new| [running[0] + new[0], running[1] + new[1]],
                            );

                        let gamma_2i = self.params.gamma_powers_square[i];
                        [
                            gamma_2i * F::from_barrett_reduce(inner_sum[0]),
                            gamma_2i * F::from_barrett_reduce(inner_sum[1]),
                        ]
                    })
                    .reduce(
                        || [F::zero(); DEGREE_BOUND - 1],
                        |running, new| [running[0] + new[0], running[1] + new[1]],
                    );
                coeffs
            });

        B.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
    }

    fn compute_phase2_message(&self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let D = &self.D;
        let H = self.H.as_ref().expect("H should be initialized in phase 2");
        let num_polys = H.num_polys();

        // Compute quadratic coefficients via generic split-eq fold (handles both E_in cases).
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = D
            .par_fold_out_in_unreduced::<9, { DEGREE_BOUND - 1 }>(&|j_prime| {
                // Accumulate in unreduced form to minimize per-term reductions
                let mut acc_c = F::Unreduced::<9>::zero();
                let mut acc_e = F::Unreduced::<9>::zero();
                for i in 0..num_polys {
                    let h_0 = H.get_bound_coeff(i, 2 * j_prime);
                    let h_1 = H.get_bound_coeff(i, 2 * j_prime + 1);
                    let b = h_1 - h_0;

                    // Phase-2 optimization: H is pre-scaled by rho_i = gamma^i, so gamma^{2i}
                    // factors are already accounted for:
                    //   gamma^{2i}*h0*(h0-1) = (rho*h0) * (rho*h0 - rho)
                    //   gamma^{2i}*b*b       = (rho*b) * (rho*b)
                    let rho = self.gamma_powers[i];
                    acc_c += h_0.mul_unreduced::<9>(h_0 - rho);
                    acc_e += b.mul_unreduced::<9>(b);
                }
                [
                    F::from_montgomery_reduce::<9>(acc_c),
                    F::from_montgomery_reduce::<9>(acc_e),
                ]
            });

        // previous_claim is s(0)+s(1) of the scaled polynomial; divide out eq_r_r to get inner claim
        let adjusted_claim = previous_claim * self.eq_r_r.inverse().unwrap();
        let gruen_poly =
            D.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], adjusted_claim);

        gruen_poly * self.eq_r_r
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for BooleanitySumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "BooleanitySumcheckProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < self.params.log_k_chunk {
            self.compute_phase1_message(round, previous_claim)
        } else {
            self.compute_phase2_message(round, previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "BooleanitySumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if round < self.params.log_k_chunk {
            // Phase 1: Bind B and update F
            self.B.bind(r_j);
            self.F.update(r_j);

            // Transition to phase 2
            if round == self.params.log_k_chunk - 1 {
                self.eq_r_r = self.B.get_current_scalar();

                // Initialize SharedRaPolynomials with per-poly pre-scaled eq tables (by rho_i)
                let F_table = std::mem::take(&mut self.F);
                let ra_indices = std::mem::take(&mut self.ra_indices);
                let base_eq = F_table.clone_values();
                let num_polys = self.params.polynomial_types.len();
                debug_assert!(
                    num_polys == self.gamma_powers.len(),
                    "gamma_powers length mismatch: got {}, expected {}",
                    self.gamma_powers.len(),
                    num_polys
                );
                let tables: Vec<Vec<F>> = (0..num_polys)
                    .into_par_iter()
                    .map(|i| {
                        let rho = self.gamma_powers[i];
                        base_eq.iter().map(|v| rho * *v).collect()
                    })
                    .collect();
                self.H = Some(SharedRaPolynomials::new(
                    tables,
                    ra_indices,
                    self.params.one_hot_params.clone(),
                ));

                // Drop G arrays
                let g = std::mem::take(&mut self.G);
                drop_in_background_thread(g);
            }
        } else {
            // Phase 2: Bind D and H
            self.D.bind(r_j);
            if let Some(ref mut h) = self.H {
                h.bind_in_place(r_j, BindingOrder::LowToHigh);
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let H = self.H.as_ref().expect("H should be initialized");
        // H is scaled by rho_i; unscale so cached openings match the committed polynomials.
        let claims: Vec<F> = (0..H.num_polys())
            .map(|i| H.final_sumcheck_claim(i) * self.gamma_powers_inv[i])
            .collect();

        // All polynomials share the same opening point (r_address, r_cycle)
        // Use a single SumcheckId for all
        accumulator.append_sparse(
            transcript,
            self.params.polynomial_types.clone(),
            SumcheckId::Booleanity,
            opening_point.r[..self.params.log_k_chunk].to_vec(),
            opening_point.r[self.params.log_k_chunk..].to_vec(),
            claims,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Parameters for the cycle-first booleanity sumcheck.
///
/// Same data as `BooleanitySumcheckParams`, but `normalize_opening_point` accounts
/// for the reversed binding order: cycle variables first, then address variables.
#[derive(Allocative, Clone)]
pub struct BooleanityCycleFirstParams<F: JoltField> {
    pub inner: BooleanitySumcheckParams<F>,
}

impl<F: JoltField> SumcheckInstanceParams<F> for BooleanityCycleFirstParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.inner.log_t + self.inner.log_k_chunk
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let log_t = self.inner.log_t;
        let mut cycle_be = sumcheck_challenges[..log_t].to_vec();
        let mut addr_be = sumcheck_challenges[log_t..].to_vec();
        cycle_be.reverse();
        addr_be.reverse();
        let mut point = addr_be;
        point.extend(cycle_be);
        point.into()
    }
}

/// Default number of sparse Phase 1 rounds for cycle-first booleanity.
///
/// Balances memory (dense allocation = T × K × N / 2^S) against compute
/// (S extra sparse rounds at O(T·N) each). `log_K + 4` gives ~92 MB peak
/// for typical parameters.
pub fn default_sparse_rounds(log_t: usize, log_k_chunk: usize) -> usize {
    (log_k_chunk + 4).min(log_t)
}

/// Cycle-first Booleanity Sumcheck Prover (3-Phase Design).
///
/// Phase 1 (sparse cycle, S rounds): Binds cycle variables using the sparse
/// `ra_indices` + `eq_split` representation. Each round does O(T·N) work.
///
/// Transition: Materializes a dense table `H_dense` of size (T/2^S) × K × N.
///
/// Phase 2 (dense cycle, log_T - S rounds): Binds remaining cycle variables
/// on the dense table. Total work O(T·K·N/2^S) (geometric decay).
///
/// Phase 3 (address, log_K rounds): Binds address variables on the K-vectors.
/// Total work O(K·N·log_K), negligible.
#[derive(Allocative)]
pub struct BooleanitySumcheckProverCycleFirst<F: JoltField> {
    gamma_powers_inv: Vec<F>,
    /// Split-eq over cycle variables (Phase 1 and Phase 2, LowToHigh).
    D: GruenSplitEqPolynomial<F>,
    /// Split-eq over address variables (Phase 3, LowToHigh).
    B: GruenSplitEqPolynomial<F>,
    /// eq(r_address, k) for k ∈ [K_chunk]. Static weight for Phase 1 and Phase 2.
    eq_addr: Vec<F>,
    /// Sparse address data per cycle (size T). Phase 1 only, dropped at materialization.
    ra_indices: Vec<RaIndices>,
    /// Expanding table for bound cycle challenges. Grows from 1 to 2^S during Phase 1.
    eq_split: ExpandingTable<F>,
    /// Dense table: `H_dense[i]` has `dense_rows * K` entries, cycle-major layout.
    /// `H_dense[i][g * K + k]` = weight at address k for merged row g.
    /// Initialized at materialization, used in Phase 2 and Phase 3.
    H_dense: Vec<Vec<F>>,
    /// Current number of rows in `H_dense` (halves each Phase 2 round).
    dense_rows: usize,
    /// Accumulated eq(r_cycle, bound_cycles) scalar. Set when all cycle variables are bound.
    eq_r_r: F,
    /// Number of sparse Phase 1 rounds (S).
    sparse_rounds: usize,
    pub params: BooleanityCycleFirstParams<F>,
}

impl<F: JoltField> BooleanitySumcheckProverCycleFirst<F> {
    #[tracing::instrument(skip_all, name = "BooleanitySumcheckProverCycleFirst::initialize")]
    pub fn initialize(
        params: BooleanitySumcheckParams<F>,
        trace: &[Cycle],
        bytecode: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
        sparse_rounds: usize,
    ) -> Self {
        assert_ra_bounds(&params.one_hot_params);
        let sparse_rounds = sparse_rounds.max(1).min(params.log_t);

        let ra_indices: Vec<RaIndices> = trace
            .par_iter()
            .map(|cycle| {
                RaIndices::from_cycle(cycle, bytecode, memory_layout, &params.one_hot_params)
            })
            .collect();

        let r_addr_rev: Vec<F::Challenge> = params.r_address.iter().copied().rev().collect();
        let eq_addr = EqPolynomial::<F>::evals(&r_addr_rev);

        let D = GruenSplitEqPolynomial::new(&params.r_cycle, BindingOrder::LowToHigh);
        let B = GruenSplitEqPolynomial::new(&params.r_address, BindingOrder::LowToHigh);

        let eq_split_capacity = 1usize << sparse_rounds;
        let mut eq_split = ExpandingTable::new(eq_split_capacity, BindingOrder::LowToHigh);
        eq_split.reset(F::one());

        let num_polys = params.polynomial_types.len();
        let gamma_f: F = params.gamma.into();
        let mut gamma_powers_inv = Vec::with_capacity(num_polys);
        let mut rho_i = F::one();
        for _ in 0..num_polys {
            gamma_powers_inv.push(
                rho_i
                    .inverse()
                    .expect("gamma_powers[i] is nonzero (gamma != 0)"),
            );
            rho_i *= gamma_f;
        }

        Self {
            gamma_powers_inv,
            D,
            B,
            eq_addr,
            ra_indices,
            eq_split,
            H_dense: Vec::new(),
            dense_rows: 0,
            eq_r_r: F::zero(),
            sparse_rounds,
            params: BooleanityCycleFirstParams { inner: params },
        }
    }

    /// Phase 1: sparse cycle binding message.
    fn compute_sparse_message(&self, round: usize, previous_claim: F) -> UniPoly<F> {
        let D = &self.D;
        let N = self.params.inner.polynomial_types.len();
        let k_chunk = 1 << self.params.inner.log_k_chunk;

        if round == 0 {
            let quadratic_coeffs: [F; DEGREE_BOUND - 1] = D
                .par_fold_out_in_unreduced::<9, { DEGREE_BOUND - 1 }>(&|j_prime| {
                    let j0 = 2 * j_prime;
                    let j1 = 2 * j_prime + 1;
                    let idx0 = &self.ra_indices[j0];
                    let idx1 = &self.ra_indices[j1];
                    let one_hot = &self.params.inner.one_hot_params;

                    let mut q_inf = F::zero();
                    for i in 0..N {
                        let k0 = idx0.get_index(i, one_hot);
                        let k1 = idx1.get_index(i, one_hot);
                        if k0 != k1 {
                            let a0 = k0.map_or(F::zero(), |k| self.eq_addr[k as usize]);
                            let a1 = k1.map_or(F::zero(), |k| self.eq_addr[k as usize]);
                            q_inf += self.params.inner.gamma_powers_square[i] * (a0 + a1);
                        }
                    }
                    [F::zero(), q_inf]
                });

            D.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
        } else {
            let m = round;
            let half_block = 1usize << m;
            let block_size = half_block << 1;

            let quadratic_coeffs: [F; DEGREE_BOUND - 1] = D
                .par_fold_out_in_unreduced::<9, { DEGREE_BOUND - 1 }>(&|j_prime| {
                    let base = j_prime * block_size;
                    let one_hot = &self.params.inner.one_hot_params;

                    let mut q0_total = F::zero();
                    let mut qinf_total = F::zero();
                    let mut even_w = vec![F::zero(); k_chunk];
                    let mut odd_w = vec![F::zero(); k_chunk];
                    let mut touched = Vec::with_capacity(k_chunk.min(block_size));
                    let mut touched_mask = vec![0u8; k_chunk];

                    for i in 0..N {
                        // Coalescing skip: all indices identical → booleanity is zero
                        let first_idx = self.ra_indices[base].get_index(i, one_hot);
                        let all_same = (1..block_size)
                            .all(|b| self.ra_indices[base + b].get_index(i, one_hot) == first_idx);
                        if all_same {
                            continue;
                        }

                        let mut q0_i = F::zero();
                        let mut qinf_i = F::zero();

                        for b in 0..half_block {
                            let w = self.eq_split[b];
                            let j_even = base + b;
                            let j_odd = base + half_block + b;

                            if let Some(k) = self.ra_indices[j_even].get_index(i, one_hot) {
                                let k = k as usize;
                                if touched_mask[k] == 0 {
                                    touched.push(k);
                                }
                                touched_mask[k] |= 1;
                                even_w[k] += w;
                            }
                            if let Some(k) = self.ra_indices[j_odd].get_index(i, one_hot) {
                                let k = k as usize;
                                if touched_mask[k] == 0 {
                                    touched.push(k);
                                }
                                touched_mask[k] |= 2;
                                odd_w[k] += w;
                            }
                        }

                        for &k in touched.iter() {
                            let eq_k = self.eq_addr[k];
                            let ew = even_w[k];
                            let ow = odd_w[k];
                            let diff = ow - ew;
                            q0_i += eq_k * ew * (ew - F::one());
                            qinf_i += eq_k * diff * diff;
                        }

                        let gamma_2i = self.params.inner.gamma_powers_square[i];
                        q0_total += gamma_2i * q0_i;
                        qinf_total += gamma_2i * qinf_i;

                        for &k in touched.iter() {
                            even_w[k] = F::zero();
                            odd_w[k] = F::zero();
                            touched_mask[k] = 0;
                        }
                        touched.clear();
                    }

                    [q0_total, qinf_total]
                });

            D.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
        }
    }

    /// Phase 2: dense cycle binding message.
    ///
    /// For each row pair in `H_dense`, computes the booleanity quadratic weighted by
    /// `eq_addr`, then uses D's Gruen polynomial for the eq_cycle factor.
    fn compute_dense_cycle_message(&self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let D = &self.D;
        let N = self.params.inner.polynomial_types.len();
        let K = 1 << self.params.inner.log_k_chunk;

        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = D
            .par_fold_out_in_unreduced::<9, { DEGREE_BOUND - 1 }>(&|j_prime| {
                let mut q0 = F::zero();
                let mut qinf = F::zero();

                for i in 0..N {
                    let gamma_2i = self.params.inner.gamma_powers_square[i];
                    let H_i = &self.H_dense[i];
                    for k in 0..K {
                        let h_even = H_i[2 * j_prime * K + k];
                        let h_odd = H_i[(2 * j_prime + 1) * K + k];
                        let diff = h_odd - h_even;
                        let w = gamma_2i * self.eq_addr[k];
                        q0 += w * h_even * (h_even - F::one());
                        qinf += w * diff * diff;
                    }
                }

                [q0, qinf]
            });

        D.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
    }

    /// Phase 3: address binding message.
    ///
    /// After all cycle variables are bound, each `H_dense[i]` is a shrinking address
    /// vector. B's Gruen polynomial handles the eq_addr factor; the callback provides
    /// the booleanity quadratic per address pair.
    fn compute_address_message(&self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let B = &self.B;
        let N = self.params.inner.polynomial_types.len();

        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = B
            .par_fold_out_in_unreduced::<9, { DEGREE_BOUND - 1 }>(&|k_prime| {
                let mut q0 = F::zero();
                let mut qinf = F::zero();

                for i in 0..N {
                    let gamma_2i = self.params.inner.gamma_powers_square[i];
                    let h_even = self.H_dense[i][2 * k_prime];
                    let h_odd = self.H_dense[i][2 * k_prime + 1];
                    let diff = h_odd - h_even;
                    q0 += gamma_2i * h_even * (h_even - F::one());
                    qinf += gamma_2i * diff * diff;
                }

                [q0, qinf]
            });

        let adjusted_claim = previous_claim * self.eq_r_r.inverse().unwrap();
        let gruen_poly =
            B.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], adjusted_claim);
        gruen_poly * self.eq_r_r
    }

    /// Materialize sparse RA indices + eq_split into a dense table.
    ///
    /// Produces `H_dense[i]` with `(T/2^S) * K` entries per polynomial.
    #[tracing::instrument(
        skip_all,
        name = "BooleanitySumcheckProverCycleFirst::materialize_dense"
    )]
    fn materialize_dense(&mut self) {
        let K = 1 << self.params.inner.log_k_chunk;
        let N = self.params.inner.polynomial_types.len();
        let block_size = 1usize << self.sparse_rounds;
        let num_groups = self.ra_indices.len() / block_size;
        let one_hot = &self.params.inner.one_hot_params;

        self.H_dense = (0..N)
            .into_par_iter()
            .map(|i| {
                let mut h = vec![F::zero(); num_groups * K];
                for g in 0..num_groups {
                    for b in 0..block_size {
                        let j = g * block_size + b;
                        let w = self.eq_split[b];
                        if let Some(k) = self.ra_indices[j].get_index(i, one_hot) {
                            h[g * K + k as usize] += w;
                        }
                    }
                }
                h
            })
            .collect();

        self.dense_rows = num_groups;

        let ra = std::mem::take(&mut self.ra_indices);
        let eq = std::mem::take(&mut self.eq_split);
        drop_in_background_thread((ra, eq));
    }

    /// Bind `H_dense` over the cycle dimension (Phase 2).
    fn bind_dense_cycle(&mut self, r_j: F::Challenge) {
        let K = 1 << self.params.inner.log_k_chunk;
        let new_rows = self.dense_rows / 2;
        let r: F = r_j.into();

        self.H_dense.par_iter_mut().for_each(|h| {
            for g in 0..new_rows {
                for k in 0..K {
                    let even = h[2 * g * K + k];
                    let odd = h[(2 * g + 1) * K + k];
                    h[g * K + k] = even + r * (odd - even);
                }
            }
            h.truncate(new_rows * K);
        });

        self.dense_rows = new_rows;
    }

    /// Bind `H_dense` over the address dimension (Phase 3).
    fn bind_dense_address(&mut self, r_j: F::Challenge) {
        let r: F = r_j.into();

        self.H_dense.par_iter_mut().for_each(|h| {
            let new_len = h.len() / 2;
            for k in 0..new_len {
                let even = h[2 * k];
                let odd = h[2 * k + 1];
                h[k] = even + r * (odd - even);
            }
            h.truncate(new_len);
        });
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for BooleanitySumcheckProverCycleFirst<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "BooleanitySumcheckProverCycleFirst::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let S = self.sparse_rounds;
        let log_t = self.params.inner.log_t;

        if round < S {
            self.compute_sparse_message(round, previous_claim)
        } else if round < log_t {
            self.compute_dense_cycle_message(round, previous_claim)
        } else {
            self.compute_address_message(round, previous_claim)
        }
    }

    #[tracing::instrument(
        skip_all,
        name = "BooleanitySumcheckProverCycleFirst::ingest_challenge"
    )]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        let S = self.sparse_rounds;
        let log_t = self.params.inner.log_t;

        if round < S {
            self.D.bind(r_j);
            self.eq_split.update(r_j);

            if round == S - 1 {
                self.materialize_dense();
            }
        } else if round < log_t {
            self.D.bind(r_j);
            self.bind_dense_cycle(r_j);
        } else {
            self.B.bind(r_j);
            self.bind_dense_address(r_j);
        }

        if round == log_t - 1 {
            self.eq_r_r = self.D.get_current_scalar();
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let log_k = self.params.inner.log_k_chunk;
        let num_polys = self.params.inner.polynomial_types.len();

        let claims: Vec<F> = (0..num_polys)
            .map(|i| {
                debug_assert_eq!(self.H_dense[i].len(), 1);
                self.H_dense[i][0]
            })
            .collect();

        accumulator.append_sparse(
            transcript,
            self.params.inner.polynomial_types.clone(),
            SumcheckId::Booleanity,
            opening_point.r[..log_k].to_vec(),
            opening_point.r[log_k..].to_vec(),
            claims,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Cycle-first Booleanity Sumcheck Verifier.
pub struct BooleanitySumcheckCycleFirstVerifier<F: JoltField> {
    params: BooleanityCycleFirstParams<F>,
}

impl<F: JoltField> BooleanitySumcheckCycleFirstVerifier<F> {
    pub fn new(params: BooleanityCycleFirstParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for BooleanitySumcheckCycleFirstVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let ra_claims: Vec<F> = self
            .params
            .inner
            .polynomial_types
            .iter()
            .map(|poly_type| {
                accumulator
                    .get_committed_polynomial_opening(*poly_type, SumcheckId::Booleanity)
                    .1
            })
            .collect();

        // Build combined r in cycle-first sumcheck order:
        // challenges are [cycle(log_t), address(log_k)].
        let combined_r: Vec<F::Challenge> = self
            .params
            .inner
            .r_cycle
            .iter()
            .cloned()
            .rev()
            .chain(self.params.inner.r_address.iter().cloned().rev())
            .collect();

        EqPolynomial::<F>::mle(sumcheck_challenges, &combined_r)
            * zip(&self.params.inner.gamma_powers_square, ra_claims)
                .map(|(gamma_2i, ra)| (ra.square() - ra) * gamma_2i)
                .sum::<F>()
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_sparse(
            transcript,
            self.params.inner.polynomial_types.clone(),
            SumcheckId::Booleanity,
            opening_point.r,
        );
    }
}

/// Booleanity Sumcheck Verifier.
pub struct BooleanitySumcheckVerifier<F: JoltField> {
    params: BooleanitySumcheckParams<F>,
}

impl<F: JoltField> BooleanitySumcheckVerifier<F> {
    pub fn new(params: BooleanitySumcheckParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for BooleanitySumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let ra_claims: Vec<F> = self
            .params
            .polynomial_types
            .iter()
            .map(|poly_type| {
                accumulator
                    .get_committed_polynomial_opening(*poly_type, SumcheckId::Booleanity)
                    .1
            })
            .collect();

        let combined_r: Vec<F::Challenge> = self
            .params
            .r_address
            .iter()
            .cloned()
            .rev()
            .chain(self.params.r_cycle.iter().cloned().rev())
            .collect();

        EqPolynomial::<F>::mle(sumcheck_challenges, &combined_r)
            * zip(&self.params.gamma_powers_square, ra_claims)
                .map(|(gamma_2i, ra)| (ra.square() - ra) * gamma_2i)
                .sum::<F>()
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_sparse(
            transcript,
            self.params.polynomial_types.clone(),
            SumcheckId::Booleanity,
            opening_point.r,
        );
    }
}
