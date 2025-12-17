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
    field::{JoltField, OptimizedMul},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::BindingOrder,
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        shared_ra_polys::{compute_all_G_and_ra_indices, RaIndices, SharedRaEqTableBank, SharedRaPolynomials},
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

/// Number of suffix bits of `E_in` to peel and pre-scale into batching coefficients.
///
/// This is a pure prover-side optimization knob (no transcript impact). Increasing this trades
/// a small amount of extra preprocessing (scaling `gammas` by a tiny `E_active` table) for a
/// ~`2^bits` reduction in the number of `E_in * value` multiplications inside the split-eq fold.
const E_IN_PRESCALE_BITS: usize = 2;

/// Parameters for the booleanity sumcheck.
pub struct BooleanitySumcheckParams<F: JoltField> {
    /// Log of chunk size (shared across all families)
    pub log_k_chunk: usize,
    /// Log of trace length
    pub log_t: usize,
    /// Batching challenges (γ_i for each RA polynomial)
    pub gammas: Vec<F::Challenge>,
    /// Address binding point (shared across all families)
    pub r_address: Vec<F::Challenge>,
    /// Cycle binding point (shared across all families)
    pub r_cycle: Vec<F::Challenge>,
    /// Polynomial types for all families
    pub polynomial_types: Vec<CommittedPolynomial>,
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

        // Sample batching challenges (TODO: we can also reuse prior challenges from Stage 5)
        let gammas = transcript.challenge_vector_optimized::<F>(total_d);

        Self {
            log_k_chunk,
            log_t,
            gammas,
            r_address,
            r_cycle,
            polynomial_types,
        }
    }
}

/// Booleanity Sumcheck Prover.
#[derive(Allocative)]
pub struct BooleanitySumcheckProver<F: JoltField> {
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
    /// OneHotParams for SharedRaPolynomials
    #[allocative(skip)]
    one_hot_params: OneHotParams,
    #[allocative(skip)]
    params: BooleanitySumcheckParams<F>,
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
        one_hot_params: &OneHotParams,
    ) -> Self {
        // Compute G and RA indices in a single pass over the trace
        let (G, ra_indices) = compute_all_G_and_ra_indices::<F>(
            trace,
            bytecode,
            memory_layout,
            one_hot_params,
            &params.r_cycle,
        );

        // Initialize split-eq polynomials for address and cycle variables
        let B = GruenSplitEqPolynomial::new(&params.r_address, BindingOrder::LowToHigh);
        let D = GruenSplitEqPolynomial::new(&params.r_cycle, BindingOrder::LowToHigh);

        // Initialize expanding table for phase 1
        let k_chunk = 1 << params.log_k_chunk;
        let mut F_table = ExpandingTable::new(k_chunk, BindingOrder::LowToHigh);
        F_table.reset(F::one());

        Self {
            B,
            D,
            G,
            ra_indices,
            one_hot_params: one_hot_params.clone(),
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

        // Peel a few fixed `E_in` bits and fold their weights into the batching coefficients.
        // This reduces the number of `E_in * value` multiplications in the split-eq fold.
        let e_active = B.e_in_active_evals(E_IN_PRESCALE_BITS);
        let gammas_scaled: Vec<Vec<F>> = e_active
            .iter()
            .map(|&s| {
                self.params
                    .gammas
                    .iter()
                    .map(|&gamma| gamma.mul_01_optimized(s))
                    .collect()
            })
            .collect();

        // Compute quadratic coefficients via split-eq fold (with peeled suffix bits).
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = B.par_fold_out_in_unreduced_peel_in_bits::<
            9,
            { DEGREE_BOUND - 1 },
        >(E_IN_PRESCALE_BITS, &|k_prime, active_idx| {
            let gammas = &gammas_scaled[active_idx];
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

                    [gammas[i] * F::from_barrett_reduce(inner_sum[0]), gammas[i]
                        * F::from_barrett_reduce(inner_sum[1])]
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

        // Peel a few fixed `E_in` bits. We'll push these peeled weights into *address eq tables*
        // (SharedRaPolynomials) rather than into `gammas`, so that `gamma` stays a Challenge and
        // we keep the MontChallenge optimized multiplication in the hot loop.
        let e_active = D.e_in_active_evals(E_IN_PRESCALE_BITS);

        // Precompute `s * eq(r_address, ·)` (and its Round2/3 variants) for each peeled scalar s.
        // This shares the same `indices` array as `H`, i.e., does not duplicate per-cycle data.
        let scaled_tables: Option<Vec<SharedRaEqTableBank<F>>> = H.precompute_scaled_eq_tables(&e_active);

        let gammas = &self.params.gammas[..num_polys];

        // Compute quadratic coefficients via split-eq fold (with peeled suffix bits).
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = D.par_fold_out_in_unreduced_peel_in_bits::<
            9,
            { DEGREE_BOUND - 1 },
        >(E_IN_PRESCALE_BITS, &|j_prime, active_idx| {
            // If we no longer have shared eq tables (RoundN), fall back to multiplying by s
            // directly inside this per-g computation. This is late in the sumcheck when the
            // fold is small, so it's acceptable.
            let s = e_active[active_idx];

            // Accumulate with a small Barrett accumulator (we're summing field elements, not
            // unreduced Montgomery products).
            let mut acc_c = F::Unreduced::<5>::zero();
            let mut acc_e = F::Unreduced::<5>::zero();

            for (i, &gamma) in gammas.iter().enumerate() {
                // Unscaled h values.
                let h0 = H.get_bound_coeff(i, 2 * j_prime);
                let h1 = H.get_bound_coeff(i, 2 * j_prime + 1);
                let delta = h1 - h0;

                // Scaled h values (s * h) via pre-scaled eq tables if available.
                let (sh0, sh1) = if let Some(ref banks) = scaled_tables {
                    let sh0 = H.get_bound_coeff_with_scaled_tables(i, 2 * j_prime, &banks[active_idx]);
                    let sh1 =
                        H.get_bound_coeff_with_scaled_tables(i, 2 * j_prime + 1, &banks[active_idx]);
                    (sh0, sh1)
                } else {
                    (s * h0, s * h1)
                };

                // Compute s * (h0^2 - h0) using h0 and (s*h0):
                //   s*(h0^2 - h0) = (s*h0)*h0 - (s*h0) = (s*h0)*(h0 - 1)
                let c_val = sh0 * (h0 - F::one());

                // Compute s * (delta^2) using delta and s*delta:
                //   s*delta^2 = delta*(s*delta), where s*delta = (s*h1) - (s*h0)
                let sdelta = sh1 - sh0;
                let e_val = delta * sdelta;

                // Keep gamma as Challenge and use its optimized mul path.
                let c_term = gamma.mul_01_optimized(c_val);
                let e_term = gamma.mul_01_optimized(e_val);

                acc_c += *c_term.as_unreduced_ref();
                acc_e += *e_term.as_unreduced_ref();
            }

            [
                F::from_barrett_reduce::<5>(acc_c),
                F::from_barrett_reduce::<5>(acc_e),
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

                // Initialize SharedRaPolynomials with shared eq table
                let F_table = std::mem::take(&mut self.F);
                let ra_indices = std::mem::take(&mut self.ra_indices);
                let one_hot_params = self.one_hot_params.clone();
                self.H = Some(SharedRaPolynomials::new(
                    F_table.clone_values(),
                    ra_indices,
                    one_hot_params,
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
        let claims: Vec<F> = (0..H.num_polys())
            .map(|i| H.final_sumcheck_claim(i))
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
            * zip(&self.params.gammas, ra_claims)
                .map(|(gamma, ra)| (ra.square() - ra) * gamma)
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
