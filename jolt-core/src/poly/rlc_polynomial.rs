use crate::field::{BarrettReduce, FMAdd, JoltField};
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::shared_ra_polys::{MAX_BYTECODE_D, MAX_INSTRUCTION_D, MAX_RAM_D};
use crate::utils::accumulation::Acc6S;
use crate::utils::math::s64_from_diff_u64s;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::instruction::LookupQuery;
use crate::zkvm::ram::remap_address;
use crate::zkvm::witness::CommittedPolynomial;
use allocative::Allocative;
use ark_bn254::{Fr, G1Projective};
use ark_ec::CurveGroup;
use common::constants::XLEN;
use common::jolt_device::MemoryLayout;
use itertools::Itertools;
use rayon::prelude::*;
use std::sync::Arc;
use tracer::ChunksIterator;
use tracer::{instruction::Cycle, LazyTraceIterator};
use tracing::trace_span;

#[derive(Clone, Debug)]
pub struct RLCStreamingData {
    pub bytecode: BytecodePreprocessing,
    pub memory_layout: MemoryLayout,
}

/// Source of trace data for streaming VMV computation.
#[derive(Clone, Debug)]
pub enum TraceSource {
    /// Pre-materialized trace in memory (default, efficient single pass)
    Materialized(Arc<Vec<Cycle>>),
    /// Lazy trace iterator (experimental, re-runs tracer)
    /// Boxed to avoid large enum size difference (LazyTraceIterator is ~34KB)
    Lazy(Box<LazyTraceIterator>),
}

impl TraceSource {
    /// Returns the trace length if known (materialized traces only).
    /// Lazy traces require iteration to determine length.
    pub fn len(&self) -> Option<usize> {
        match self {
            TraceSource::Materialized(trace) => Some(trace.len()),
            TraceSource::Lazy(_) => None,
        }
    }

    /// Returns whether the trace is empty if known (materialized traces only).
    /// Lazy traces require iteration to determine emptiness.
    pub fn is_empty(&self) -> Option<bool> {
        match self {
            TraceSource::Materialized(trace) => Some(trace.is_empty()),
            TraceSource::Lazy(_) => None,
        }
    }
}

/// Streaming context for RLC evaluation
#[derive(Clone, Debug)]
pub struct StreamingRLCContext<F: JoltField> {
    pub dense_polys: Vec<(CommittedPolynomial, F)>,
    pub onehot_polys: Vec<(CommittedPolynomial, F)>,
    pub trace_source: TraceSource,
    pub preprocessing: Arc<RLCStreamingData>,
    pub one_hot_params: OneHotParams,
}

/// `RLCPolynomial` represents a multilinear polynomial comprised of a
/// random linear combination of multiple polynomials, potentially with
/// different sizes.
#[derive(Default, Clone, Debug, Allocative)]
pub struct RLCPolynomial<F: JoltField> {
    /// Random linear combination of dense (i.e. length T) polynomials.
    /// Empty if using streaming mode.
    pub dense_rlc: Vec<F>,
    /// Random linear combination of one-hot polynomials (length T x K
    /// for some K). Instead of pre-emptively combining these polynomials,
    /// as we do for `dense_rlc`, we store a vector of (coefficient, polynomial)
    /// pairs and lazily handle the linear combination in `commit_rows`
    /// and `vector_matrix_product`.
    pub one_hot_rlc: Vec<(F, Arc<MultilinearPolynomial<F>>)>,
    /// When present, dense_rlc and one_hot_rlc are not materialized.
    #[allocative(skip)]
    pub streaming_context: Option<Arc<StreamingRLCContext<F>>>,
}

impl<F: JoltField> PartialEq for RLCPolynomial<F> {
    fn eq(&self, other: &Self) -> bool {
        // Compare materialized data only (streaming context is ephemeral)
        self.dense_rlc == other.dense_rlc && self.one_hot_rlc == other.one_hot_rlc
    }
}

impl<F: JoltField> RLCPolynomial<F> {
    pub fn new() -> Self {
        Self {
            dense_rlc: unsafe_allocate_zero_vec(DoryGlobals::get_T()),
            one_hot_rlc: vec![],
            streaming_context: None,
        }
    }

    /// Constructs an `RLCPolynomial` as a linear combination of `polynomials` with the provided
    /// `coefficients`.
    ///
    /// This is a legacy helper (used by some commitment backends) that eagerly combines dense
    /// polynomials into `dense_rlc` and stores one-hot polynomials lazily in `one_hot_rlc`.
    #[allow(unused_variables)]
    pub fn linear_combination(
        poly_ids: Vec<CommittedPolynomial>,
        polynomials: Vec<Arc<MultilinearPolynomial<F>>>,
        coefficients: &[F],
        streaming_context: Option<Arc<StreamingRLCContext<F>>>,
    ) -> Self {
        debug_assert_eq!(polynomials.len(), coefficients.len());
        debug_assert_eq!(polynomials.len(), poly_ids.len());

        // Collect indices of dense (non-one-hot) polynomials.
        let dense_indices: Vec<usize> = polynomials
            .iter()
            .enumerate()
            .filter(|(_, p)| !matches!(p.as_ref(), MultilinearPolynomial::OneHot(_)))
            .map(|(i, _)| i)
            .collect();

        // Eagerly materialize the dense linear combination (if any).
        let dense_rlc = if dense_indices.is_empty() {
            vec![]
        } else {
            let max_len = dense_indices
                .iter()
                .map(|&i| polynomials[i].as_ref().original_len())
                .max()
                .unwrap();

            (0..max_len)
                .into_par_iter()
                .map(|idx| {
                    let mut acc = F::zero();
                    for &poly_idx in &dense_indices {
                        let poly = polynomials[poly_idx].as_ref();
                        let coeff = coefficients[poly_idx];

                        if idx < poly.original_len() {
                            acc += poly.get_scaled_coeff(idx, coeff);
                        }
                    }
                    acc
                })
                .collect()
        };

        // Store one-hot polynomials lazily.
        let mut one_hot_rlc = Vec::new();
        for (i, poly) in polynomials.iter().enumerate() {
            if matches!(poly.as_ref(), MultilinearPolynomial::OneHot(_)) {
                one_hot_rlc.push((coefficients[i], poly.clone()));
            }
        }

        Self {
            dense_rlc,
            one_hot_rlc,
            streaming_context,
        }
    }

    /// Creates a streaming RLC polynomial from polynomial IDs and coefficients.
    /// O(sqrt(T)) space - streams directly from trace without materializing polynomials.
    ///
    /// # Arguments
    /// * `one_hot_params` - Parameters for one-hot polynomial chunking
    /// * `preprocessing` - Bytecode and memory layout for address computation
    /// * `trace_source` - Either materialized trace (default) or lazy trace (experimental)
    /// * `poly_ids` - List of polynomial identifiers
    /// * `coefficients` - RLC coefficients for each polynomial
    #[tracing::instrument(skip_all)]
    pub fn new_streaming(
        one_hot_params: OneHotParams,
        preprocessing: Arc<RLCStreamingData>,
        trace_source: TraceSource,
        poly_ids: Vec<CommittedPolynomial>,
        coefficients: &[F],
    ) -> Self {
        debug_assert_eq!(poly_ids.len(), coefficients.len());

        let mut dense_polys = Vec::new();
        let mut onehot_polys = Vec::new();

        for (poly_id, coeff) in poly_ids.iter().zip(coefficients.iter()) {
            match poly_id {
                CommittedPolynomial::RdInc | CommittedPolynomial::RamInc => {
                    dense_polys.push((*poly_id, *coeff));
                }
                CommittedPolynomial::InstructionRa(_)
                | CommittedPolynomial::BytecodeRa(_)
                | CommittedPolynomial::RamRa(_) => {
                    onehot_polys.push((*poly_id, *coeff));
                }
            }
        }

        Self {
            dense_rlc: vec![],   // Not materialized in streaming mode
            one_hot_rlc: vec![], // Not materialized in streaming mode
            streaming_context: Some(Arc::new(StreamingRLCContext {
                dense_polys,
                onehot_polys,
                trace_source,
                preprocessing,
                one_hot_params,
            })),
        }
    }

    /// Materializes a streaming RLC polynomial for testing purposes.
    #[cfg(test)]
    pub fn materialize(
        &self,
        _poly_ids: &[CommittedPolynomial],
        polynomials: &[Arc<MultilinearPolynomial<F>>],
        coefficients: &[F],
    ) -> Self {
        if self.streaming_context.is_none() {
            return self.clone();
        }

        let mut result = RLCPolynomial::<F>::new();
        let dense_indices: Vec<usize> = polynomials
            .iter()
            .enumerate()
            .filter(|(_, p)| !matches!(p.as_ref(), MultilinearPolynomial::OneHot(_)))
            .map(|(i, _)| i)
            .collect();

        if !dense_indices.is_empty() {
            let dense_len = result.dense_rlc.len();

            result.dense_rlc = (0..dense_len)
                .into_par_iter()
                .map(|i| {
                    let mut acc = F::zero();
                    for &poly_idx in &dense_indices {
                        let poly = polynomials[poly_idx].as_ref();
                        let coeff = coefficients[poly_idx];

                        if i < poly.original_len() {
                            acc += poly.get_scaled_coeff(i, coeff);
                        }
                    }
                    acc
                })
                .collect();
        }

        for (i, poly) in polynomials.iter().enumerate() {
            if matches!(poly.as_ref(), MultilinearPolynomial::OneHot(_)) {
                result.one_hot_rlc.push((coefficients[i], poly.clone()));
            }
        }

        result
    }

    /// Commits to the rows of `RLCPolynomial`, viewing its coefficients
    /// as a matrix (used in Dory).
    /// We do so by computing the row commitments for the individual
    /// polynomials comprising the linear combination, and taking the
    /// linear combination of the resulting commitments.
    // TODO(moodlezoup): we should be able to cache the row commitments
    // for each underlying polynomial and take a linear combination of those
    #[tracing::instrument(skip_all, name = "RLCPolynomial::commit_rows")]
    pub fn commit_rows<G: CurveGroup<ScalarField = F> + VariableBaseMSM>(
        &self,
        bases: &[G::Affine],
    ) -> Vec<G> {
        let num_rows = DoryGlobals::get_max_num_rows();
        tracing::debug!("Committing to RLC polynomial with {num_rows} rows");
        let row_len = DoryGlobals::get_num_columns();

        let mut row_commitments = vec![G::zero(); num_rows];

        // Compute the row commitments for dense submatrix
        self.dense_rlc
            .par_chunks(row_len)
            .zip(row_commitments.par_iter_mut())
            .for_each(|(dense_row, commitment)| {
                let msm_result: G =
                    VariableBaseMSM::msm_field_elements(&bases[..dense_row.len()], dense_row)
                        .unwrap();
                *commitment += msm_result
            });

        // Compute the row commitments for one-hot polynomials
        for (coeff, poly) in self.one_hot_rlc.iter() {
            let mut new_row_commitments: Vec<G> = match poly.as_ref() {
                MultilinearPolynomial::OneHot(one_hot) => one_hot.commit_rows(bases),
                _ => panic!("Expected OneHot polynomial in one_hot_rlc"),
            };

            // TODO(moodlezoup): Avoid resize
            new_row_commitments.resize(num_rows, G::zero());

            let updated_row_commitments: &mut [G1Projective] = unsafe {
                std::slice::from_raw_parts_mut(
                    new_row_commitments.as_mut_ptr() as *mut G1Projective,
                    new_row_commitments.len(),
                )
            };

            let current_row_commitments: &[G1Projective] = unsafe {
                std::slice::from_raw_parts(
                    row_commitments.as_ptr() as *const G1Projective,
                    row_commitments.len(),
                )
            };

            let coeff_fr = unsafe { *(&raw const *coeff as *const Fr) };

            let _span = trace_span!("vector_scalar_mul_add_gamma_g1_online");
            let _enter = _span.enter();

            // Scales the row commitments for the current polynomial by
            // its coefficient
            jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
                updated_row_commitments,
                coeff_fr,
                current_row_commitments,
            );

            let _ = std::mem::replace(&mut row_commitments, new_row_commitments);
        }

        row_commitments
    }

    /// Computes a vector-matrix product, viewing the coefficients of the
    /// polynomial as a matrix (used in Dory).
    /// We do so by computing the vector-matrix product for the individual
    /// polynomials comprising the linear combination, and taking the
    /// linear combination of the resulting products.
    #[tracing::instrument(skip_all, name = "RLCPolynomial::vector_matrix_product")]
    pub fn vector_matrix_product(&self, left_vec: &[F]) -> Vec<F> {
        let num_columns = DoryGlobals::get_num_columns();

        // Compute the vector-matrix product for dense submatrix
        let mut result: Vec<F> = if let Some(ctx) = &self.streaming_context {
            // Streaming mode: generate rows on-demand from trace
            self.streaming_vector_matrix_product(left_vec, num_columns, Arc::clone(ctx))
        } else {
            // Linear space mode: use pre-computed dense_rlc
            (0..num_columns)
                .into_par_iter()
                .map(|col_index| {
                    self.dense_rlc
                        .iter()
                        .skip(col_index)
                        .step_by(num_columns)
                        .zip(left_vec.iter())
                        .map(|(&a, &b)| -> F { a * b })
                        .sum::<F>()
                })
                .collect()
        };

        // Compute the vector-matrix product for one-hot polynomials (linear space)
        for (coeff, poly) in self.one_hot_rlc.iter() {
            match poly.as_ref() {
                MultilinearPolynomial::OneHot(one_hot) => {
                    one_hot.vector_matrix_product(left_vec, *coeff, &mut result);
                }
                _ => panic!("Expected OneHot polynomial in one_hot_rlc"),
            }
        }

        result
    }

    /// Extract dense polynomial value from a cycle (debug builds only).
    #[inline]
    #[cfg(debug_assertions)]
    fn extract_dense_value(poly_id: &CommittedPolynomial, cycle: &Cycle) -> F {
        match poly_id {
            CommittedPolynomial::RdInc => {
                let (_, pre_value, post_value) = cycle.rd_write();
                F::from_i128(post_value as i128 - pre_value as i128)
            }
            CommittedPolynomial::RamInc => match cycle.ram_access() {
                tracer::instruction::RAMAccess::Write(write) => {
                    F::from_i128(write.post_value as i128 - write.pre_value as i128)
                }
                tracer::instruction::RAMAccess::Read(_) | tracer::instruction::RAMAccess::NoOp => {
                    F::zero()
                }
            },
            CommittedPolynomial::InstructionRa(_)
            | CommittedPolynomial::BytecodeRa(_)
            | CommittedPolynomial::RamRa(_) => {
                panic!("One-hot polynomials should not be passed to extract_dense_value")
            }
        }
    }

    /// Extract one-hot index k from a cycle for a given polynomial (debug builds only).
    #[inline]
    #[cfg(debug_assertions)]
    fn extract_onehot_k(
        poly_id: &CommittedPolynomial,
        cycle: &Cycle,
        preprocessing: &RLCStreamingData,
        one_hot_params: &OneHotParams,
    ) -> Option<usize> {
        match poly_id {
            CommittedPolynomial::InstructionRa(idx) => {
                let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                Some(one_hot_params.lookup_index_chunk(lookup_index, *idx) as usize)
            }
            CommittedPolynomial::BytecodeRa(idx) => {
                let pc = preprocessing.bytecode.get_pc(cycle);
                Some(one_hot_params.bytecode_pc_chunk(pc, *idx) as usize)
            }
            CommittedPolynomial::RamRa(idx) => remap_address(
                cycle.ram_access().address() as u64,
                &preprocessing.memory_layout,
            )
            .map(|address| one_hot_params.ram_address_chunk(address, *idx) as usize),
            CommittedPolynomial::RdInc | CommittedPolynomial::RamInc => {
                panic!("Dense polynomials should not be passed to extract_onehot_k")
            }
        }
    }

    /// Validates that NoOp cycles produce expected values (zeros for dense, specific k for one-hot).
    /// Called in debug builds when processing padding rows.
    #[cfg(debug_assertions)]
    fn assert_noop_invariants(ctx: &StreamingRLCContext<F>) {
        for (poly_id, _) in &ctx.dense_polys {
            debug_assert_eq!(
                Self::extract_dense_value(poly_id, &Cycle::NoOp),
                F::zero(),
                "Expected zero dense value for NoOp, got non-zero for {poly_id:?}"
            );
        }
        for (poly_id, _) in &ctx.onehot_polys {
            let k = Self::extract_onehot_k(
                poly_id,
                &Cycle::NoOp,
                &ctx.preprocessing,
                &ctx.one_hot_params,
            );
            match poly_id {
                CommittedPolynomial::InstructionRa(_) | CommittedPolynomial::BytecodeRa(_) => {
                    debug_assert_eq!(
                        k,
                        Some(0),
                        "Expected Some(0) for {poly_id:?} on NoOp, got {k:?}"
                    );
                }
                CommittedPolynomial::RamRa(_) => {
                    debug_assert_eq!(k, None, "Expected None for RamRa on NoOp, got {k:?}");
                }
                _ => unreachable!(),
            }
        }
    }

    /// Streaming VMP implementation that generates rows on-demand from trace.
    /// Achieves O(sqrt(n)) space complexity by lazily generating the witness.
    /// Single pass through trace for both dense and one-hot polynomials.
    fn streaming_vector_matrix_product(
        &self,
        left_vec: &[F],
        num_columns: usize,
        ctx: Arc<StreamingRLCContext<F>>,
    ) -> Vec<F> {
        let T = DoryGlobals::get_T();

        match &ctx.trace_source {
            TraceSource::Materialized(trace) => {
                self.materialized_vector_matrix_product(left_vec, num_columns, trace, &ctx, T)
            }
            TraceSource::Lazy(lazy_trace) => self.lazy_vector_matrix_product(
                left_vec,
                num_columns,
                (**lazy_trace).clone(),
                &ctx,
                T,
            ),
        }
    }

    /// Single-pass VMV over materialized trace. Parallelizes by dividing rows evenly across threads.
    #[tracing::instrument(skip_all)]
    fn materialized_vector_matrix_product(
        &self,
        left_vec: &[F],
        num_columns: usize,
        trace: &[Cycle],
        ctx: &StreamingRLCContext<F>,
        T: usize,
    ) -> Vec<F> {
        let num_rows = T / num_columns;
        let trace_len = trace.len();

        // Setup: precompute coefficients, row factors, and folded one-hot tables.
        let setup = VmvSetup::new(ctx, left_vec, num_rows);

        // Divide rows evenly among threads.
        let num_threads = rayon::current_num_threads();
        let rows_per_thread = num_rows.div_ceil(num_threads);
        let chunk_ranges: Vec<(usize, usize)> = (0..num_threads)
            .map(|t| {
                let start = t * rows_per_thread;
                let end = std::cmp::min(start + rows_per_thread, num_rows);
                (start, end)
            })
            .filter(|(start, end)| start < end)
            .collect();

        let (dense_accs, onehot_accs) = chunk_ranges
            .into_par_iter()
            .map(|(row_start, row_end)| {
                let (mut dense_accs, mut onehot_accs) =
                    VmvCoeffs::<F>::create_accumulators(num_columns);

                for row_idx in row_start..row_end {
                    let chunk_start = row_idx * num_columns;
                    let row_weight = left_vec[row_idx];

                    // Build per-cycle context with row-scaled coefficients.
                    let cycle_ctx = CycleContext {
                        scaled_rd_inc: row_weight * setup.coeffs.rd_inc_coeff,
                        scaled_ram_inc: row_weight * setup.coeffs.ram_inc_coeff,
                        row_factor: setup.row_factors[row_idx],
                        folded_tables: &setup.folded_onehot,
                    };

                    // Split into valid trace range vs padding range.
                    let valid_end = std::cmp::min(chunk_start + num_columns, trace_len);
                    let row_cycles = if chunk_start < valid_end {
                        &trace[chunk_start..valid_end]
                    } else {
                        &trace[0..0] // Fully padded row
                    };

                    // Process valid trace elements.
                    for (col_idx, cycle) in row_cycles.iter().enumerate() {
                        setup.coeffs.process_cycle(
                            cycle,
                            &cycle_ctx,
                            &mut dense_accs[col_idx],
                            &mut onehot_accs[col_idx],
                        );
                    }

                    // Validate padding invariants: NoOp cycles must produce zeros.
                    #[cfg(debug_assertions)]
                    if row_cycles.len() < num_columns {
                        Self::assert_noop_invariants(ctx);
                    }
                }

                (dense_accs, onehot_accs)
            })
            .reduce(
                || VmvCoeffs::<F>::create_accumulators(num_columns),
                VmvCoeffs::<F>::merge_accumulators,
            );

        VmvCoeffs::<F>::finalize(dense_accs, onehot_accs, num_columns)
    }

    /// Lazy VMV over lazy trace iterator (experimental, re-runs tracer).
    #[tracing::instrument(skip_all)]
    fn lazy_vector_matrix_product(
        &self,
        left_vec: &[F],
        num_columns: usize,
        lazy_trace: LazyTraceIterator,
        ctx: &StreamingRLCContext<F>,
        T: usize,
    ) -> Vec<F> {
        let num_rows = T / num_columns;

        // Setup: precompute coefficients, row factors, and folded one-hot tables.
        let setup = VmvSetup::new(ctx, left_vec, num_rows);

        let (dense_accs, onehot_accs) = lazy_trace
            .pad_using(T, |_| Cycle::NoOp)
            .iter_chunks(num_columns)
            .enumerate()
            .par_bridge()
            .fold(
                || VmvCoeffs::<F>::create_accumulators(num_columns),
                |(mut dense_accs, mut onehot_accs), (row_idx, chunk)| {
                    let row_weight = left_vec[row_idx];

                    // Build per-cycle context with row-scaled coefficients.
                    let cycle_ctx = CycleContext {
                        scaled_rd_inc: row_weight * setup.coeffs.rd_inc_coeff,
                        scaled_ram_inc: row_weight * setup.coeffs.ram_inc_coeff,
                        row_factor: setup.row_factors[row_idx],
                        folded_tables: &setup.folded_onehot,
                    };

                    // Process columns within chunk sequentially.
                    for (col_idx, cycle) in chunk.iter().enumerate() {
                        setup.coeffs.process_cycle(
                            cycle,
                            &cycle_ctx,
                            &mut dense_accs[col_idx],
                            &mut onehot_accs[col_idx],
                        );
                    }

                    (dense_accs, onehot_accs)
                },
            )
            .reduce(
                || VmvCoeffs::<F>::create_accumulators(num_columns),
                VmvCoeffs::<F>::merge_accumulators,
            );

        VmvCoeffs::<F>::finalize(dense_accs, onehot_accs, num_columns)
    }
}

// ============================================================================
// VMV Helper Types - Preprocessed coefficients for streaming vector-matrix product
// ============================================================================

/// Precomputed tables for the one-hot VMV fast path that **folds per-polynomial coefficients
/// (γ-powers) into the K-sized eq table**.
///
/// Let `left_vec` (length `K * rows_per_k`) be the Dory left-side Lagrange vector, arranged as
/// `left_vec[k * rows_per_k + row]`.
///
/// When `T >= K` (the typical Jolt regime), the row index bits are `[k_bits || row_bits]`, so:
///
/// ```text
/// left_vec[k,row] = eq_k[k] * eq_row[row]
/// ```
///
/// We can precompute `eq_k` from `left_vec` and build per-polynomial tables:
///
/// ```text
/// table_poly[k] = coeff_poly * eq_k[k]
/// ```
///
/// Then for any cycle in row `row`, the one-hot contribution is:
///
/// ```text
/// sum_poly coeff_poly * left_vec[k_poly,row]
///   = eq_row[row] * sum_poly table_poly[k_poly]
/// ```
///
/// This moves γ-multiplications out of the hot per-cycle loop (O(T * #polys) multiplications)
/// into a tiny precompute (O(K * #polys) multiplications).
struct VmvFoldedOneHotTables<F: JoltField> {
    k_chunk: usize,
    instruction_d: usize,
    bytecode_d: usize,
    ram_d: usize,
    /// Flattened tables in unified polynomial order:
    /// `[instruction_0..d, bytecode_0..d, ram_0..d]`, each table has length `k_chunk`.
    /// Index: `tables[(poly_idx * k_chunk) + k]`.
    tables: Vec<F>,
}

/// Per-cycle context for VMV computation.
///
/// Groups scaled coefficients and precomputed tables needed to process
/// a single cycle, reducing the number of arguments passed to `process_cycle`.
struct CycleContext<'a, F: JoltField> {
    /// Row weight * rd_inc coefficient
    scaled_rd_inc: F,
    /// Row weight * ram_inc coefficient
    scaled_ram_inc: F,
    /// Factor from row_factors table (eq_row[row])
    row_factor: F,
    /// Reference to the precomputed folded one-hot tables
    folded_tables: &'a VmvFoldedOneHotTables<F>,
}

/// Preprocessed coefficients for VMV computation.
///
/// **Invariant**: ALL CommittedPolynomials (except advice) are always present.
/// This allows us to use non-optional coefficients and remove conditional checks.
struct VmvCoeffs<'a, F: JoltField> {
    // Dense polynomial coefficients (always present)
    rd_inc_coeff: F,
    ram_inc_coeff: F,

    // One-hot RA polynomial coefficients as dense arrays indexed by chunk_idx.
    // coeffs[i] = γ^j for the polynomial with chunk index i.
    instruction_coeffs: [F; MAX_INSTRUCTION_D],
    bytecode_coeffs: [F; MAX_BYTECODE_D],
    ram_coeffs: [F; MAX_RAM_D],

    // Precomputed shifts for chunk extraction: shift[i] = log_k * (d - 1 - i)
    instruction_shifts: [usize; MAX_INSTRUCTION_D],
    bytecode_shifts: [usize; MAX_BYTECODE_D],
    ram_shifts: [usize; MAX_RAM_D],

    // Actual dimensions used
    instruction_d: usize,
    bytecode_d: usize,
    ram_d: usize,

    // Cached parameters
    k_chunk_mask_usize: usize,
    k_chunk_mask_u64: u64,
    k_chunk_mask_u128: u128,

    // References to preprocessing data
    bytecode: &'a BytecodePreprocessing,
    memory_layout: &'a MemoryLayout,
}

impl<'a, F: JoltField> VmvCoeffs<'a, F> {
    /// Compute `(row_factors, eq_k)` from the Dory left vector.
    ///
    /// - `eq_k[k] = Σ_row left_vec[k,row]`
    /// - `row_factors[row] = Σ_k left_vec[k,row]`
    ///
    /// In the typical `T >= K` regime, these correspond to `eq_k` over the address bits and
    /// `eq_row` over the cycle high bits respectively, and satisfy:
    /// `left_vec[k,row] = eq_k[k] * row_factors[row]`.
    #[inline]
    fn compute_row_factors_and_eq_k(
        left_vec: &[F],
        rows_per_k: usize,
        k_chunk: usize,
    ) -> (Vec<F>, Vec<F>) {
        debug_assert!(
            left_vec.len() >= k_chunk * rows_per_k,
            "left_vec too short: len={} need_at_least={}",
            left_vec.len(),
            k_chunk * rows_per_k
        );

        // Row factors: length rows_per_k, zero-initialized.
        let mut row_factors: Vec<F> = unsafe_allocate_zero_vec(rows_per_k);
        // Eq table over k: length k_chunk.
        let mut eq_k: Vec<F> = unsafe_allocate_zero_vec(k_chunk);

        // Single sequential pass for good cache behavior: scan left_vec contiguously.
        for k in 0..k_chunk {
            let base = k * rows_per_k;
            let mut sum_k = F::zero();
            for row in 0..rows_per_k {
                let v = left_vec[base + row];
                sum_k += v;
                row_factors[row] += v;
            }
            eq_k[k] = sum_k;
        }

        (row_factors, eq_k)
    }

    /// Build per-polynomial folded one-hot tables: `table_poly[k] = coeff_poly * eq_k[k]`.
    #[inline]
    fn build_folded_onehot_tables(&self, eq_k: &[F]) -> VmvFoldedOneHotTables<F> {
        let k_chunk = eq_k.len();
        debug_assert_eq!(
            k_chunk,
            self.k_chunk_mask_usize + 1,
            "eq_k length must match k_chunk"
        );

        let instruction_d = self.instruction_d;
        let bytecode_d = self.bytecode_d;
        let ram_d = self.ram_d;
        let num_polys = instruction_d + bytecode_d + ram_d;

        let mut tables: Vec<F> = unsafe_allocate_zero_vec(num_polys * k_chunk);

        // Instruction tables
        for i in 0..instruction_d {
            let coeff = self.instruction_coeffs[i];
            if coeff.is_zero() {
                continue;
            }
            let base = i * k_chunk;
            for k in 0..k_chunk {
                tables[base + k] = coeff * eq_k[k];
            }
        }

        // Bytecode tables
        let bytecode_base_poly = instruction_d;
        for i in 0..bytecode_d {
            let coeff = self.bytecode_coeffs[i];
            if coeff.is_zero() {
                continue;
            }
            let base = (bytecode_base_poly + i) * k_chunk;
            for k in 0..k_chunk {
                tables[base + k] = coeff * eq_k[k];
            }
        }

        // RAM tables
        let ram_base_poly = instruction_d + bytecode_d;
        for i in 0..ram_d {
            let coeff = self.ram_coeffs[i];
            if coeff.is_zero() {
                continue;
            }
            let base = (ram_base_poly + i) * k_chunk;
            for k in 0..k_chunk {
                tables[base + k] = coeff * eq_k[k];
            }
        }

        VmvFoldedOneHotTables {
            k_chunk,
            instruction_d,
            bytecode_d,
            ram_d,
            tables,
        }
    }

    /// Build preprocessed coefficients from streaming context.
    /// Assumes ALL CommittedPolynomials are present.
    fn new(ctx: &'a StreamingRLCContext<F>) -> Self {
        let one_hot_params = &ctx.one_hot_params;
        let log_k_chunk = one_hot_params.log_k_chunk;
        let instruction_d = one_hot_params.instruction_d;
        let bytecode_d = one_hot_params.bytecode_d;
        let ram_d = one_hot_params.ram_d;

        // Initialize all coefficients to zero
        let mut rd_inc_coeff = F::zero();
        let mut ram_inc_coeff = F::zero();
        let mut instruction_coeffs = [F::zero(); MAX_INSTRUCTION_D];
        let mut bytecode_coeffs = [F::zero(); MAX_BYTECODE_D];
        let mut ram_coeffs = [F::zero(); MAX_RAM_D];

        // Accumulate dense coefficients
        for (poly_id, coeff) in ctx.dense_polys.iter() {
            match poly_id {
                CommittedPolynomial::RdInc => rd_inc_coeff += *coeff,
                CommittedPolynomial::RamInc => ram_inc_coeff += *coeff,
                _ => unreachable!("one-hot polynomial found in dense_polys"),
            }
        }

        // Accumulate one-hot coefficients by chunk index
        for (poly_id, coeff) in ctx.onehot_polys.iter() {
            match poly_id {
                CommittedPolynomial::InstructionRa(idx) => instruction_coeffs[*idx] += *coeff,
                CommittedPolynomial::BytecodeRa(idx) => bytecode_coeffs[*idx] += *coeff,
                CommittedPolynomial::RamRa(idx) => ram_coeffs[*idx] += *coeff,
                _ => unreachable!("dense polynomial found in onehot_polys"),
            }
        }

        // Precompute shifts for chunk extraction
        let mut instruction_shifts = [0usize; MAX_INSTRUCTION_D];
        let mut bytecode_shifts = [0usize; MAX_BYTECODE_D];
        let mut ram_shifts = [0usize; MAX_RAM_D];

        for i in 0..instruction_d {
            instruction_shifts[i] = log_k_chunk * (instruction_d - 1 - i);
        }
        for i in 0..bytecode_d {
            bytecode_shifts[i] = log_k_chunk * (bytecode_d - 1 - i);
        }
        for i in 0..ram_d {
            ram_shifts[i] = log_k_chunk * (ram_d - 1 - i);
        }

        let k_chunk_mask_usize = one_hot_params.k_chunk - 1;

        Self {
            rd_inc_coeff,
            ram_inc_coeff,
            instruction_coeffs,
            bytecode_coeffs,
            ram_coeffs,
            instruction_shifts,
            bytecode_shifts,
            ram_shifts,
            instruction_d,
            bytecode_d,
            ram_d,
            k_chunk_mask_usize,
            k_chunk_mask_u64: k_chunk_mask_usize as u64,
            k_chunk_mask_u128: k_chunk_mask_usize as u128,
            bytecode: &ctx.preprocessing.bytecode,
            memory_layout: &ctx.preprocessing.memory_layout,
        }
    }

    /// Process a single cycle, accumulating into dense and one-hot accumulators.
    #[inline(always)]
    fn process_cycle(
        &self,
        cycle: &Cycle,
        ctx: &CycleContext<'_, F>,
        dense_acc: &mut Acc6S<F>,
        onehot_acc: &mut F::Unreduced<9>,
    ) {
        // Dense polynomials: always accumulate scaled_coeff * (post - pre)
        let (_, pre_value, post_value) = cycle.rd_write();
        let diff = s64_from_diff_u64s(post_value, pre_value);
        dense_acc.fmadd(&ctx.scaled_rd_inc, &diff);

        if let tracer::instruction::RAMAccess::Write(write) = cycle.ram_access() {
            let diff = s64_from_diff_u64s(write.post_value, write.pre_value);
            dense_acc.fmadd(&ctx.scaled_ram_inc, &diff);
        }

        // One-hot polynomials (γ-folded): accumulate inner sum using the pre-folded K tables,
        // then multiply once by the per-row factor.
        let k_chunk = ctx.folded_tables.k_chunk;
        let tables = &ctx.folded_tables.tables;

        let mut inner_sum = F::zero();

        // Instruction RA chunks
        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
        for i in 0..ctx.folded_tables.instruction_d {
            let k =
                ((lookup_index >> self.instruction_shifts[i]) & self.k_chunk_mask_u128) as usize;
            debug_assert!(k < k_chunk);
            inner_sum += tables[i * k_chunk + k];
        }

        // Bytecode RA chunks
        let pc = self.bytecode.get_pc(cycle);
        let bytecode_base = ctx.folded_tables.instruction_d;
        for i in 0..ctx.folded_tables.bytecode_d {
            let k = (pc >> self.bytecode_shifts[i]) & self.k_chunk_mask_usize;
            debug_assert!(k < k_chunk);
            inner_sum += tables[(bytecode_base + i) * k_chunk + k];
        }

        // RAM RA chunks (may be None for non-memory cycles)
        let address = cycle.ram_access().address() as u64;
        if let Some(remapped) = remap_address(address, self.memory_layout) {
            let ram_base = ctx.folded_tables.instruction_d + ctx.folded_tables.bytecode_d;
            for i in 0..ctx.folded_tables.ram_d {
                let k = ((remapped >> self.ram_shifts[i]) & self.k_chunk_mask_u64) as usize;
                debug_assert!(k < k_chunk);
                inner_sum += tables[(ram_base + i) * k_chunk + k];
            }
        }

        *onehot_acc += ctx.row_factor.mul_unreduced::<9>(inner_sum);
    }

    /// Create empty accumulators for a VMV reduction. Always creates both.
    #[inline]
    fn create_accumulators(num_columns: usize) -> (Vec<Acc6S<F>>, Vec<F::Unreduced<9>>) {
        (
            unsafe_allocate_zero_vec(num_columns),
            unsafe_allocate_zero_vec(num_columns),
        )
    }

    /// Merge two accumulator pairs (used in parallel reduce). Always merges both.
    #[inline]
    fn merge_accumulators(
        (mut dense_a, mut onehot_a): (Vec<Acc6S<F>>, Vec<F::Unreduced<9>>),
        (dense_b, onehot_b): (Vec<Acc6S<F>>, Vec<F::Unreduced<9>>),
    ) -> (Vec<Acc6S<F>>, Vec<F::Unreduced<9>>) {
        for (a, b) in dense_a.iter_mut().zip(dense_b.iter()) {
            *a = *a + *b;
        }
        for (a, b) in onehot_a.iter_mut().zip(onehot_b.iter()) {
            *a += *b;
        }
        (dense_a, onehot_a)
    }

    /// Finalize accumulators: reduce and combine dense + onehot contributions (parallel).
    fn finalize(
        dense_accs: Vec<Acc6S<F>>,
        onehot_accs: Vec<F::Unreduced<9>>,
        num_columns: usize,
    ) -> Vec<F> {
        (0..num_columns)
            .into_par_iter()
            .map(|col_idx| {
                dense_accs[col_idx].barrett_reduce()
                    + F::from_montgomery_reduce::<9>(onehot_accs[col_idx])
            })
            .collect()
    }
}

/// Precomputed VMV setup shared between materialized and lazy paths.
///
/// Encapsulates the common preprocessing needed for streaming vector-matrix
/// product computation: coefficient tables, row factors, and folded one-hot tables.
struct VmvSetup<'a, F: JoltField> {
    /// Preprocessed polynomial coefficients.
    coeffs: VmvCoeffs<'a, F>,
    /// Per-row eq factors (sum over k).
    row_factors: Vec<F>,
    /// Precomputed folded one-hot tables.
    folded_onehot: VmvFoldedOneHotTables<F>,
}

impl<'a, F: JoltField> VmvSetup<'a, F> {
    /// Creates a new VMV setup from streaming context and Dory left vector.
    fn new(ctx: &'a StreamingRLCContext<F>, left_vec: &[F], num_rows: usize) -> Self {
        let coeffs = VmvCoeffs::new(ctx);
        let k_chunk = ctx.one_hot_params.k_chunk;

        let (row_factors, eq_k) =
            VmvCoeffs::<F>::compute_row_factors_and_eq_k(left_vec, num_rows, k_chunk);
        let folded_onehot = coeffs.build_folded_onehot_tables(&eq_k);

        debug_assert!(
            left_vec.len() >= k_chunk * num_rows,
            "left_vec too short for one-hot VMV: len={} need_at_least={}",
            left_vec.len(),
            k_chunk * num_rows
        );

        Self {
            coeffs,
            row_factors,
            folded_onehot,
        }
    }
}
