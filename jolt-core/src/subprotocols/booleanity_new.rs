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
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        shared_ra_polys::{compute_all_G_and_ra_indices, RaIndices},
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{
        expanding_table::ExpandingTable,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
    zkvm::{
        bytecode::BytecodePreprocessing,
        config::OneHotParams,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};

/// Degree bound of the sumcheck round polynomials.
const DEGREE_BOUND: usize = 3;

// ============================================================================
// Sidon-basis batching (gamma powers) parameters
// ============================================================================

// Regime-specific Sidon sets (minimize max element) for:
// - log_k_chunk = 4: need >= 43 distinct weights → |S|=9 gives 45 pair-sums
// - log_k_chunk = 8: need >= 25 distinct weights → |S|=7 gives 28 pair-sums
const GAMMA_BASIS_LOG_K_CHUNK_4: [usize; 9] = [0, 1, 5, 12, 25, 27, 35, 41, 44];
const GAMMA_BASIS_LOG_K_CHUNK_8: [usize; 7] = [0, 1, 4, 10, 18, 23, 25];

fn sidon_basis_for_log_k_chunk(log_k_chunk: usize) -> &'static [usize] {
    match log_k_chunk {
        4 => &GAMMA_BASIS_LOG_K_CHUNK_4,
        8 => &GAMMA_BASIS_LOG_K_CHUNK_8,
        _ => panic!(
            "Unsupported log_k_chunk {log_k_chunk} for Sidon-basis batching; expected 4 or 8"
        ),
    }
}

fn sidon_pairs_sorted_by_sum(basis_exponents: &[usize]) -> Vec<(u8, u8)> {
    let m = basis_exponents.len();
    assert!(m <= u8::MAX as usize);

    let mut triples: Vec<(usize, u8, u8)> = Vec::with_capacity(m * (m + 1) / 2);
    for a in 0..m {
        for b in a..m {
            let sum = basis_exponents[a] + basis_exponents[b];
            triples.push((sum, a as u8, b as u8));
        }
    }
    triples.sort_by(|(sum1, a1, b1), (sum2, a2, b2)| (sum1, a1, b1).cmp(&(sum2, a2, b2)));

    triples.into_iter().map(|(_sum, a, b)| (a, b)).collect()
}

// ============================================================================
// Gamma-basis Shared RA polynomials (only keep basis-scaled tables through Round3)
// ============================================================================

#[derive(Allocative)]
enum GammaBasisSharedRaPolynomials<F: JoltField> {
    Round1(GammaBasisRound1<F>),
    Round2(GammaBasisRound2<F>),
    Round3(GammaBasisRound3<F>),
    RoundN(Vec<MultilinearPolynomial<F>>),
}

#[derive(Allocative)]
struct GammaBasisRound1<F: JoltField> {
    /// Flattened tables: tables[basis_idx * K + k] = γ^{s_basis_idx} * eq(r_address, k)
    tables: Vec<F>,
    basis_size: usize,
    k_chunk: usize,
    indices: Vec<RaIndices>,
    num_polys: usize,
    #[allocative(skip)]
    one_hot_params: OneHotParams,
}

#[derive(Allocative)]
struct GammaBasisRound2<F: JoltField> {
    tables_0: Vec<F>,
    tables_1: Vec<F>,
    basis_size: usize,
    k_chunk: usize,
    indices: Vec<RaIndices>,
    num_polys: usize,
    #[allocative(skip)]
    one_hot_params: OneHotParams,
    binding_order: BindingOrder,
}

#[derive(Allocative, Default)]
struct GammaBasisRound3<F: JoltField> {
    tables_00: Vec<F>,
    tables_01: Vec<F>,
    tables_10: Vec<F>,
    tables_11: Vec<F>,
    basis_size: usize,
    k_chunk: usize,
    indices: Vec<RaIndices>,
    num_polys: usize,
    #[allocative(skip)]
    one_hot_params: OneHotParams,
    binding_order: BindingOrder,
}

impl<F: JoltField> GammaBasisSharedRaPolynomials<F> {
    fn new(
        eq_table: Vec<F>,
        indices: Vec<RaIndices>,
        one_hot_params: OneHotParams,
        gamma_basis_pows: &[F],
    ) -> Self {
        assert!(
            !gamma_basis_pows.is_empty(),
            "gamma_basis_pows must include exponent 0"
        );
        let basis_size = gamma_basis_pows.len();
        let k_chunk = eq_table.len();
        debug_assert_eq!(
            k_chunk, one_hot_params.k_chunk,
            "eq_table length must equal one_hot_params.k_chunk"
        );

        // tables[basis_idx * K + k] = gamma_basis_pows[basis_idx] * eq_table[k]
        let mut tables: Vec<F> = unsafe_allocate_zero_vec(basis_size * k_chunk);
        for (basis_idx, &scale) in gamma_basis_pows.iter().enumerate() {
            let base = basis_idx * k_chunk;
            if basis_idx == 0 && scale.is_one() {
                tables[base..base + k_chunk].copy_from_slice(&eq_table);
            } else {
                tables[base..base + k_chunk]
                    .iter_mut()
                    .zip(eq_table.iter())
                    .for_each(|(dst, &v)| *dst = scale * v);
            }
        }

        let num_polys =
            one_hot_params.instruction_d + one_hot_params.bytecode_d + one_hot_params.ram_d;

        Self::Round1(GammaBasisRound1 {
            tables,
            basis_size,
            k_chunk,
            indices,
            num_polys,
            one_hot_params,
        })
    }

    #[inline]
    fn num_polys(&self) -> usize {
        match self {
            Self::Round1(r) => r.num_polys,
            Self::Round2(r) => r.num_polys,
            Self::Round3(r) => r.num_polys,
            Self::RoundN(polys) => polys.len(),
        }
    }

    #[inline]
    fn is_materialized(&self) -> bool {
        matches!(self, Self::RoundN(_))
    }

    #[inline]
    fn get_bound_coeff_unscaled(&self, poly_idx: usize, j: usize) -> F {
        match self {
            Self::Round1(r) => r.get_scaled(0, poly_idx, j),
            Self::Round2(r) => r.get_scaled(0, poly_idx, j),
            Self::Round3(r) => r.get_scaled(0, poly_idx, j),
            Self::RoundN(polys) => polys[poly_idx].get_bound_coeff(j),
        }
    }

    #[inline]
    fn get_bound_coeff_scaled(&self, basis_idx: usize, poly_idx: usize, j: usize) -> F {
        match self {
            Self::Round1(r) => r.get_scaled(basis_idx, poly_idx, j),
            Self::Round2(r) => r.get_scaled(basis_idx, poly_idx, j),
            Self::Round3(r) => r.get_scaled(basis_idx, poly_idx, j),
            Self::RoundN(_) => {
                panic!("get_bound_coeff_scaled called after materialization (RoundN)")
            }
        }
    }

    fn final_sumcheck_claim(&self, poly_idx: usize) -> F {
        match self {
            Self::RoundN(polys) => polys[poly_idx].final_sumcheck_claim(),
            _ => panic!("final_sumcheck_claim called before RoundN"),
        }
    }

    fn bind_in_place(&mut self, r: F::Challenge, order: BindingOrder) {
        // Avoid moving out of `&mut` enum variants by swapping `self` with a temporary.
        let prev = std::mem::replace(self, Self::RoundN(Vec::new()));
        match prev {
            Self::Round1(r1) => {
                *self = Self::Round2(r1.bind(r, order));
            }
            Self::Round2(r2) => {
                *self = Self::Round3(r2.bind(r, order));
            }
            Self::Round3(r3) => {
                *self = Self::RoundN(r3.bind(r, order));
            }
            Self::RoundN(mut polys) => {
                polys.par_iter_mut().for_each(|p| p.bind_parallel(r, order));
                *self = Self::RoundN(polys);
            }
        }
    }
}

impl<F: JoltField> GammaBasisRound1<F> {
    #[inline]
    fn get_scaled(&self, basis_idx: usize, poly_idx: usize, j: usize) -> F {
        self.indices[j]
            .get_index(poly_idx, &self.one_hot_params)
            .map_or(F::zero(), |k| self.tables[basis_idx * self.k_chunk + k as usize])
    }

    fn bind(self, r0: F::Challenge, order: BindingOrder) -> GammaBasisRound2<F> {
        let eq_0_r0 = EqPolynomial::mle(&[F::zero()], &[r0]);
        let eq_1_r0 = EqPolynomial::mle(&[F::one()], &[r0]);

        let mut tables_0 = self.tables.clone();
        let mut tables_1 = self.tables;
        rayon::join(
            || tables_0.par_iter_mut().for_each(|f| *f *= eq_0_r0),
            || tables_1.par_iter_mut().for_each(|f| *f *= eq_1_r0),
        );

        GammaBasisRound2 {
            tables_0,
            tables_1,
            basis_size: self.basis_size,
            k_chunk: self.k_chunk,
            indices: self.indices,
            num_polys: self.num_polys,
            one_hot_params: self.one_hot_params,
            binding_order: order,
        }
    }
}

impl<F: JoltField> GammaBasisRound2<F> {
    #[inline]
    fn get_scaled(&self, basis_idx: usize, poly_idx: usize, j: usize) -> F {
        match self.binding_order {
            BindingOrder::HighToLow => {
                let mid = self.indices.len() / 2;
                let h_0 = self.indices[j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_0[basis_idx * self.k_chunk + k as usize]);
                let h_1 = self.indices[mid + j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_1[basis_idx * self.k_chunk + k as usize]);
                h_0 + h_1
            }
            BindingOrder::LowToHigh => {
                let h_0 = self.indices[2 * j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_0[basis_idx * self.k_chunk + k as usize]);
                let h_1 = self.indices[2 * j + 1]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_1[basis_idx * self.k_chunk + k as usize]);
                h_0 + h_1
            }
        }
    }

    fn bind(self, r1: F::Challenge, order: BindingOrder) -> GammaBasisRound3<F> {
        assert_eq!(order, self.binding_order);
        let eq_0_r1 = EqPolynomial::mle(&[F::zero()], &[r1]);
        let eq_1_r1 = EqPolynomial::mle(&[F::one()], &[r1]);

        let mut tables_00 = self.tables_0.clone();
        let mut tables_01 = self.tables_0;
        let mut tables_10 = self.tables_1.clone();
        let mut tables_11 = self.tables_1;

        rayon::join(
            || {
                rayon::join(
                    || tables_00.par_iter_mut().for_each(|f| *f *= eq_0_r1),
                    || tables_01.par_iter_mut().for_each(|f| *f *= eq_1_r1),
                )
            },
            || {
                rayon::join(
                    || tables_10.par_iter_mut().for_each(|f| *f *= eq_0_r1),
                    || tables_11.par_iter_mut().for_each(|f| *f *= eq_1_r1),
                )
            },
        );

        GammaBasisRound3 {
            tables_00,
            tables_01,
            tables_10,
            tables_11,
            basis_size: self.basis_size,
            k_chunk: self.k_chunk,
            indices: self.indices,
            num_polys: self.num_polys,
            one_hot_params: self.one_hot_params,
            binding_order: order,
        }
    }
}

impl<F: JoltField> GammaBasisRound3<F> {
    #[inline]
    fn get_scaled(&self, basis_idx: usize, poly_idx: usize, j: usize) -> F {
        match self.binding_order {
            BindingOrder::HighToLow => {
                let quarter = self.indices.len() / 4;
                let h_00 = self.indices[j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_00[basis_idx * self.k_chunk + k as usize]);
                let h_01 = self.indices[quarter + j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_01[basis_idx * self.k_chunk + k as usize]);
                let h_10 = self.indices[2 * quarter + j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_10[basis_idx * self.k_chunk + k as usize]);
                let h_11 = self.indices[3 * quarter + j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_11[basis_idx * self.k_chunk + k as usize]);
                h_00 + h_01 + h_10 + h_11
            }
            BindingOrder::LowToHigh => {
                // Offset bits are (r2,r1,r0) with r0 bound first; Round3 tables are indexed by (r1,r0).
                let h_00 = self.indices[4 * j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_00[basis_idx * self.k_chunk + k as usize]);
                let h_10 = self.indices[4 * j + 1]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_10[basis_idx * self.k_chunk + k as usize]);
                let h_01 = self.indices[4 * j + 2]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_01[basis_idx * self.k_chunk + k as usize]);
                let h_11 = self.indices[4 * j + 3]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_11[basis_idx * self.k_chunk + k as usize]);
                h_00 + h_10 + h_01 + h_11
            }
        }
    }

    #[tracing::instrument(skip_all, name = "GammaBasisRound3::bind")]
    fn bind(self, r2: F::Challenge, order: BindingOrder) -> Vec<MultilinearPolynomial<F>> {
        assert_eq!(order, self.binding_order);

        // Materialize ONLY the unscaled polynomials (basis_idx = 0) and drop the rest.
        let eq_0_r2 = EqPolynomial::mle(&[F::zero()], &[r2]);
        let eq_1_r2 = EqPolynomial::mle(&[F::one()], &[r2]);

        // Extract unscaled (basis 0) K-sized tables.
        let k = self.k_chunk;
        let base = 0;
        let f_00 = self.tables_00[base * k..(base + 1) * k].to_vec();
        let f_01 = self.tables_01[base * k..(base + 1) * k].to_vec();
        let f_10 = self.tables_10[base * k..(base + 1) * k].to_vec();
        let f_11 = self.tables_11[base * k..(base + 1) * k].to_vec();

        // Create 8 unscaled tables: F_ABC where A=r0, B=r1, C=r2
        let mut f_000 = f_00.clone();
        let mut f_001 = f_00;
        let mut f_010 = f_01.clone();
        let mut f_011 = f_01;
        let mut f_100 = f_10.clone();
        let mut f_101 = f_10;
        let mut f_110 = f_11.clone();
        let mut f_111 = f_11;

        rayon::join(
            || {
                rayon::join(
                    || {
                        rayon::join(
                            || f_000.par_iter_mut().for_each(|x| *x *= eq_0_r2),
                            || f_001.par_iter_mut().for_each(|x| *x *= eq_1_r2),
                        )
                    },
                    || {
                        rayon::join(
                            || f_010.par_iter_mut().for_each(|x| *x *= eq_0_r2),
                            || f_011.par_iter_mut().for_each(|x| *x *= eq_1_r2),
                        )
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        rayon::join(
                            || f_100.par_iter_mut().for_each(|x| *x *= eq_0_r2),
                            || f_101.par_iter_mut().for_each(|x| *x *= eq_1_r2),
                        )
                    },
                    || {
                        rayon::join(
                            || f_110.par_iter_mut().for_each(|x| *x *= eq_0_r2),
                            || f_111.par_iter_mut().for_each(|x| *x *= eq_1_r2),
                        )
                    },
                )
            },
        );

        let f_tables = [&f_000, &f_100, &f_010, &f_110, &f_001, &f_101, &f_011, &f_111];

        let num_polys = self.num_polys;
        let indices = &self.indices;
        let one_hot_params = &self.one_hot_params;
        let new_len = indices.len() / 8;

        (0..num_polys)
            .into_par_iter()
            .map(|poly_idx| {
                let coeffs: Vec<F> = match order {
                    BindingOrder::LowToHigh => (0..new_len)
                        .map(|j| {
                            (0..8)
                                .map(|offset| {
                                    indices[8 * j + offset]
                                        .get_index(poly_idx, one_hot_params)
                                        .map_or(F::zero(), |k| f_tables[offset][k as usize])
                                })
                                .sum()
                        })
                        .collect(),
                    BindingOrder::HighToLow => {
                        let eighth = indices.len() / 8;
                        (0..new_len)
                            .map(|j| {
                                (0..8)
                                    .map(|seg| {
                                        indices[seg * eighth + j]
                                            .get_index(poly_idx, one_hot_params)
                                            .map_or(F::zero(), |k| f_tables[seg][k as usize])
                                    })
                                    .sum()
                            })
                            .collect()
                    }
                };
                MultilinearPolynomial::from(coeffs)
            })
            .collect()
    }
}

/// Parameters for the booleanity sumcheck.
pub struct BooleanitySumcheckParams<F: JoltField> {
    /// Log of chunk size (shared across all families)
    pub log_k_chunk: usize,
    /// Log of trace length
    pub log_t: usize,
    /// Sidon basis exponents S (same for all polynomials)
    pub gamma_basis_exponents: Vec<usize>,
    /// Precomputed γ^s for each s in `gamma_basis_exponents` (field elements)
    pub gamma_basis_pows: Vec<F>,
    /// Per-polynomial Sidon decomposition indices (a_idx, b_idx) into `gamma_basis_*`
    pub gamma_pairs: Vec<(u8, u8)>,
    /// Per-polynomial batching weights w_i = γ^(s_a + s_b)
    pub weights: Vec<F>,
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

        // --------------------------------------------------------------------
        // Sidon-basis batching: sample ONE gamma and derive N distinct weights.
        // --------------------------------------------------------------------
        let basis_exponents = sidon_basis_for_log_k_chunk(log_k_chunk).to_vec();

        // Sample γ as an optimized challenge (small) but convert powers into field elements.
        let gamma_chal = transcript.challenge_scalar_optimized::<F>();
        let max_basis = *basis_exponents
            .iter()
            .max()
            .expect("gamma basis must be non-empty");

        // pow_cache[e] = γ^e as a field element, built via optimized field*challenge multiplies.
        let mut pow_cache: Vec<F> = vec![F::zero(); max_basis + 1];
        pow_cache[0] = F::one();
        for e in 1..=max_basis {
            pow_cache[e] = pow_cache[e - 1] * gamma_chal;
        }

        let gamma_basis_pows: Vec<F> = basis_exponents.iter().map(|&e| pow_cache[e]).collect();
        debug_assert_eq!(gamma_basis_pows[0], F::one(), "basis must start with exponent 0");

        let all_pairs = sidon_pairs_sorted_by_sum(&basis_exponents);
        assert!(
            all_pairs.len() >= total_d,
            "Sidon basis too small: need {total_d} unique weights, but basis provides only {}",
            all_pairs.len()
        );
        let gamma_pairs: Vec<(u8, u8)> = all_pairs.into_iter().take(total_d).collect();

        // weights[i] = γ^(s_a + s_b) = γ^s_a * γ^s_b
        let weights: Vec<F> = gamma_pairs
            .iter()
            .map(|&(a, b)| {
                let a = a as usize;
                let b = b as usize;
                gamma_basis_pows[a] * gamma_basis_pows[b]
            })
            .collect();

        Self {
            log_k_chunk,
            log_t,
            gamma_basis_exponents: basis_exponents,
            gamma_basis_pows,
            gamma_pairs,
            weights,
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
    H: Option<GammaBasisSharedRaPolynomials<F>>,
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

                        [
                            self.params.weights[i] * F::from_barrett_reduce(inner_sum[0]),
                            self.params.weights[i] * F::from_barrett_reduce(inner_sum[1]),
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
        let use_gamma_basis_tables = !H.is_materialized();

        // Compute quadratic coefficients via generic split-eq fold (handles both E_in cases).
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = D
            .par_fold_out_in_unreduced::<9, { DEGREE_BOUND - 1 }>(&|j_prime| {
                // Accumulate in unreduced form to minimize per-term reductions
                let mut acc_c = F::Unreduced::<9>::zero();
                let mut acc_e = F::Unreduced::<9>::zero();
                if use_gamma_basis_tables {
                    for i in 0..num_polys {
                        let (a_idx_u8, b_idx_u8) = self.params.gamma_pairs[i];
                        let a_idx = a_idx_u8 as usize;
                        let b_idx = b_idx_u8 as usize;

                        // Lookup scaled coefficients:
                        //   h0_a = γ^{s_a} h0, h0_b = γ^{s_b} h0
                        //   h1_a = γ^{s_a} h1, h1_b = γ^{s_b} h1
                        let j0 = 2 * j_prime;
                        let j1 = j0 + 1;
                        let h0_a = H.get_bound_coeff_scaled(a_idx, i, j0);
                        let h0_b = H.get_bound_coeff_scaled(b_idx, i, j0);
                        let h1_a = H.get_bound_coeff_scaled(a_idx, i, j1);
                        let h1_b = H.get_bound_coeff_scaled(b_idx, i, j1);

                        // c term:
                        //   γ^{a+b} (h0^2 - h0) = (γ^a h0) * (γ^b h0 - γ^b)
                        let inner = h0_b - self.params.gamma_basis_pows[b_idx];
                        acc_c += h0_a.mul_unreduced::<9>(inner);

                        // e term:
                        //   γ^{a+b} b^2 = (γ^a b) * (γ^b b)
                        // where b = h1 - h0 and (γ^a b) = (γ^a h1) - (γ^a h0)
                        let da = h1_a - h0_a;
                        let db = h1_b - h0_b;
                        acc_e += da.mul_unreduced::<9>(db);
                    }
                } else {
                    // After materialization (RoundN), fall back to multiplying by full field weights.
                    for (i, weight) in self.params.weights.iter().enumerate().take(num_polys) {
                        let h_0 = H.get_bound_coeff_unscaled(i, 2 * j_prime);
                        let h_1 = H.get_bound_coeff_unscaled(i, 2 * j_prime + 1);
                        let b = h_1 - h_0;

                        let w_h0 = *weight * h_0;
                        let c_unr = w_h0.mul_unreduced::<9>(h_0 - F::one());
                        acc_c += c_unr;

                        let w_b = *weight * b;
                        acc_e += w_b.mul_unreduced::<9>(b);
                    }
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

                // Initialize shared RA polynomials with basis-scaled eq tables
                let F_table = std::mem::take(&mut self.F);
                let ra_indices = std::mem::take(&mut self.ra_indices);
                let one_hot_params = self.one_hot_params.clone();
                self.H = Some(GammaBasisSharedRaPolynomials::new(
                    F_table.clone_values(),
                    ra_indices,
                    one_hot_params,
                    &self.params.gamma_basis_pows,
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
            * zip(&self.params.weights, ra_claims)
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
