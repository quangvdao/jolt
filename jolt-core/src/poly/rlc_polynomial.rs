use crate::field::{BarrettReduce, FMAdd, JoltField};
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::accumulation::Acc6S;
use crate::utils::math::s64_from_diff_u64s;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::instruction::LookupQuery;
use crate::zkvm::ram::remap_address;
use crate::zkvm::{bytecode::BytecodePreprocessing, witness::CommittedPolynomial};
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
    pub fn len(&self) -> usize {
        match self {
            TraceSource::Materialized(trace) => trace.len(),
            // Lazy trace length is not known upfront (would require full iteration)
            TraceSource::Lazy(_) => panic!("Cannot get length of lazy trace"),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            TraceSource::Materialized(trace) => trace.is_empty(),
            TraceSource::Lazy(_) => panic!("Cannot check emptiness of lazy trace"),
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
        use crate::utils::small_scalar::SmallScalar;

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
                            match poly {
                                MultilinearPolynomial::LargeScalars(p) => {
                                    acc += p.Z[idx] * coeff;
                                }
                                MultilinearPolynomial::U8Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                MultilinearPolynomial::U16Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                MultilinearPolynomial::U32Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                MultilinearPolynomial::U64Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                MultilinearPolynomial::I64Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                MultilinearPolynomial::U128Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                MultilinearPolynomial::I128Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                MultilinearPolynomial::S128Scalars(p) => {
                                    acc += p.coeffs[idx].field_mul(coeff);
                                }
                                _ => unreachable!(
                                    "unexpected polynomial variant in RLC linear_combination"
                                ),
                            }
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
        use crate::utils::small_scalar::SmallScalar;

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
                            match poly {
                                MultilinearPolynomial::U8Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::U16Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::U32Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::U64Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::I64Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::U128Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::I128Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::S128Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::LargeScalars(p) => {
                                    acc += p.Z[i] * coeff;
                                }
                                _ => unreachable!(),
                            }
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

    /// Extract dense polynomial value from a cycle
    #[inline]
    #[cfg(test)]
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

    /// Extract one-hot index k from a cycle for a given polynomial
    #[inline]
    #[cfg(test)]
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

        // Divide rows evenly among threads - one allocation per thread
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

        // --------------------------------------------------------------------
        // Preprocess coefficients into dense vs one-hot groups to avoid per-cycle
        // `CommittedPolynomial` matching and redundant decoding work.
        // --------------------------------------------------------------------
        let mut rd_inc_coeff: Option<F> = None;
        let mut ram_inc_coeff: Option<F> = None;
        for (poly_id, coeff) in ctx.dense_polys.iter() {
            match poly_id {
                CommittedPolynomial::RdInc => {
                    rd_inc_coeff = Some(rd_inc_coeff.unwrap_or(F::zero()) + *coeff);
                }
                CommittedPolynomial::RamInc => {
                    ram_inc_coeff = Some(ram_inc_coeff.unwrap_or(F::zero()) + *coeff);
                }
                CommittedPolynomial::InstructionRa(_)
                | CommittedPolynomial::BytecodeRa(_)
                | CommittedPolynomial::RamRa(_) => {
                    unreachable!("one-hot polynomial found in dense_polys")
                }
            }
        }

        let mut instruction_ra: Vec<(usize, F)> = Vec::new();
        let mut bytecode_ra: Vec<(usize, F)> = Vec::new();
        let mut ram_ra: Vec<(usize, F)> = Vec::new();
        for (poly_id, coeff) in ctx.onehot_polys.iter() {
            match poly_id {
                CommittedPolynomial::InstructionRa(idx) => instruction_ra.push((*idx, *coeff)),
                CommittedPolynomial::BytecodeRa(idx) => bytecode_ra.push((*idx, *coeff)),
                CommittedPolynomial::RamRa(idx) => ram_ra.push((*idx, *coeff)),
                CommittedPolynomial::RdInc | CommittedPolynomial::RamInc => {
                    unreachable!("dense polynomial found in onehot_polys")
                }
            }
        }

        let has_dense = rd_inc_coeff.is_some() || ram_inc_coeff.is_some();
        let has_onehot =
            !instruction_ra.is_empty() || !bytecode_ra.is_empty() || !ram_ra.is_empty();
        if has_onehot {
            // OneHotPolynomial::vector_matrix_product indexes left_vec as k * rows_per_k + row_idx.
            // Here rows_per_k == num_rows and k is in [0, K).
            debug_assert!(
                left_vec.len() >= ctx.one_hot_params.k_chunk * num_rows,
                "left_vec too short for one-hot VMV: len={} need_at_least={}",
                left_vec.len(),
                ctx.one_hot_params.k_chunk * num_rows
            );
        }

        // Cache commonly-used params to reduce indirection in hot loops.
        let one_hot_params = &ctx.one_hot_params;
        let log_k_chunk = one_hot_params.log_k_chunk;
        let k_chunk_mask_usize = one_hot_params.k_chunk - 1;
        let k_chunk_mask_u64 = k_chunk_mask_usize as u64;
        let k_chunk_mask_u128 = k_chunk_mask_usize as u128;
        let instruction_d = one_hot_params.instruction_d;
        let bytecode_d = one_hot_params.bytecode_d;
        let ram_d = one_hot_params.ram_d;
        let bytecode = &ctx.preprocessing.bytecode;
        let memory_layout = &ctx.preprocessing.memory_layout;

        let (dense_accs, onehot_accs) = chunk_ranges
            .into_par_iter()
            .map(|(row_start, row_end)| {
                // One allocation per chunk (<= num_threads):
                // - dense_accs accumulates scaled (post-pre) increments in 6-limb signed form
                // - onehot_accs accumulates field products in 9-limb form (Montgomery domain)
                let mut dense_accs: Vec<Acc6S<F>> = if has_dense {
                    unsafe_allocate_zero_vec(num_columns)
                } else {
                    Vec::new()
                };
                let mut onehot_accs: Vec<F::Unreduced<9>> = if has_onehot {
                    unsafe_allocate_zero_vec(num_columns)
                } else {
                    Vec::new()
                };

                for row_idx in row_start..row_end {
                    let chunk_start = row_idx * num_columns;
                    let row_weight = left_vec[row_idx];

                    // Row-scaled dense coefficients (no per-row allocation).
                    let scaled_rd_inc = rd_inc_coeff.map(|c| row_weight * c);
                    let scaled_ram_inc = ram_inc_coeff.map(|c| row_weight * c);

                    // For one-hot access: left_vec[k * num_rows + row_idx] == *(row_base + k*num_rows)
                    let row_base_ptr = if has_onehot {
                        // SAFETY: row_idx < num_rows and debug_assert above ensures left_vec is large enough.
                        Some(unsafe { left_vec.as_ptr().add(row_idx) })
                    } else {
                        None
                    };

                    // Split into valid trace range vs padding range (avoid branch in hot loop)
                    let valid_end = std::cmp::min(chunk_start + num_columns, trace_len);
                    let row_cycles = if chunk_start < valid_end {
                        &trace[chunk_start..valid_end]
                    } else {
                        // Fully padded row (trace shorter than T): no in-bounds cycles.
                        &trace[0..0]
                    };

                    // Process valid trace elements (no branch needed)
                    for (col_idx, cycle) in row_cycles.iter().enumerate() {
                        // ----------------
                        // Dense polynomials: accumulate scaled_coeff * (post - pre) via Acc6S + S64 diffs.
                        // Delay Barrett reduction to the end of the full VMV.
                        // ----------------
                        if has_dense {
                            let dense_acc = &mut dense_accs[col_idx];

                            if let Some(scaled) = &scaled_rd_inc {
                                let (_, pre_value, post_value) = cycle.rd_write();
                                let diff = s64_from_diff_u64s(post_value, pre_value);
                                dense_acc.fmadd(scaled, &diff);
                            }

                            if let Some(scaled) = &scaled_ram_inc {
                                match cycle.ram_access() {
                                    tracer::instruction::RAMAccess::Write(write) => {
                                        let diff =
                                            s64_from_diff_u64s(write.post_value, write.pre_value);
                                        dense_acc.fmadd(scaled, &diff);
                                    }
                                    tracer::instruction::RAMAccess::Read(_)
                                    | tracer::instruction::RAMAccess::NoOp => {}
                                }
                            }
                        }

                        // ----------------
                        // One-hot polynomials: accumulate field products in 9-limb form.
                        // Delay Montgomery reduction to the end of the full VMV.
                        // ----------------
                        if has_onehot {
                            let row_base_ptr = row_base_ptr.expect("row_base_ptr missing");
                            let onehot_acc = &mut onehot_accs[col_idx];

                            if !instruction_ra.is_empty() {
                                let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                                for (idx, coeff) in instruction_ra.iter() {
                                    debug_assert!(*idx < instruction_d);
                                    let shift = log_k_chunk * (instruction_d - 1 - *idx);
                                    let k = ((lookup_index >> shift) & k_chunk_mask_u128) as usize;
                                    // SAFETY: k < K and debug_assert above ensures bounds.
                                    let left_val = unsafe { *row_base_ptr.add(k * num_rows) };
                                    *onehot_acc += left_val.mul_unreduced::<9>(*coeff);
                                }
                            }

                            if !bytecode_ra.is_empty() {
                                let pc = bytecode.get_pc(cycle);
                                for (idx, coeff) in bytecode_ra.iter() {
                                    debug_assert!(*idx < bytecode_d);
                                    let shift = log_k_chunk * (bytecode_d - 1 - *idx);
                                    let k = (pc >> shift) & k_chunk_mask_usize;
                                    let left_val = unsafe { *row_base_ptr.add(k * num_rows) };
                                    *onehot_acc += left_val.mul_unreduced::<9>(*coeff);
                                }
                            }

                            if !ram_ra.is_empty() {
                                let address = cycle.ram_access().address() as u64;
                                if let Some(remapped) = remap_address(address, memory_layout) {
                                    for (idx, coeff) in ram_ra.iter() {
                                        debug_assert!(*idx < ram_d);
                                        let shift = log_k_chunk * (ram_d - 1 - *idx);
                                        let k = ((remapped >> shift) & k_chunk_mask_u64) as usize;
                                        let left_val = unsafe { *row_base_ptr.add(k * num_rows) };
                                        *onehot_acc += left_val.mul_unreduced::<9>(*coeff);
                                    }
                                }
                            }
                        }
                    }

                    // Process padding (NoOp cycles) - typically rare or none
                    // For NoOp: dense polys (RdInc, RamInc) are always zero,
                    // and one-hot polys contribute consistently:
                    // - InstructionRa/BytecodeRa: Some(0) (fetch NoOp at index/PC 0)
                    // - RamRa: None (no RAM access)
                    // Since these are constant per-row, we skip the per-column loop entirely.
                    #[cfg(test)]
                    if row_cycles.len() < num_columns {
                        // Verify dense polynomials are zero for NoOp
                        for (poly_id, _) in &ctx.dense_polys {
                            debug_assert_eq!(
                                Self::extract_dense_value(poly_id, &Cycle::NoOp),
                                F::zero(),
                                "Expected zero dense value for NoOp, got non-zero for {poly_id:?}"
                            );
                        }

                        // Verify one-hot polynomials have expected values for NoOp
                        for (poly_id, _) in &ctx.onehot_polys {
                            let k = Self::extract_onehot_k(
                                poly_id,
                                &Cycle::NoOp,
                                &ctx.preprocessing,
                                &ctx.one_hot_params,
                            );
                            match poly_id {
                                CommittedPolynomial::InstructionRa(_)
                                | CommittedPolynomial::BytecodeRa(_) => {
                                    debug_assert_eq!(
                                        k,
                                        Some(0),
                                        "Expected Some(0) for {poly_id:?} on NoOp, got {k:?}"
                                    );
                                }
                                CommittedPolynomial::RamRa(_) => {
                                    debug_assert_eq!(
                                        k, None,
                                        "Expected None for RamRa on NoOp, got {k:?}"
                                    );
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }

                (dense_accs, onehot_accs)
            })
            .reduce(
                || {
                    let dense_accs: Vec<Acc6S<F>> = if has_dense {
                        unsafe_allocate_zero_vec(num_columns)
                    } else {
                        Vec::new()
                    };
                    let onehot_accs: Vec<F::Unreduced<9>> = if has_onehot {
                        unsafe_allocate_zero_vec(num_columns)
                    } else {
                        Vec::new()
                    };
                    (dense_accs, onehot_accs)
                },
                |(mut dense_a, mut onehot_a), (dense_b, onehot_b)| {
                    if has_dense {
                        for (a, b) in dense_a.iter_mut().zip(dense_b.iter()) {
                            *a = *a + *b;
                        }
                    }
                    if has_onehot {
                        for (a, b) in onehot_a.iter_mut().zip(onehot_b.iter()) {
                            *a += *b;
                        }
                    }
                    (dense_a, onehot_a)
                },
            );

        // Finalize: reduce once per column and combine dense + onehot contributions (parallel).
        (0..num_columns)
            .into_par_iter()
            .map(|col_idx| {
                let mut val = F::zero();
                if has_dense {
                    val += dense_accs[col_idx].barrett_reduce();
                }
                if has_onehot {
                    val += F::from_montgomery_reduce::<9>(onehot_accs[col_idx]);
                }
                val
            })
            .collect()
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

        // Preprocess coefficients (same grouping as materialized path).
        let mut rd_inc_coeff: Option<F> = None;
        let mut ram_inc_coeff: Option<F> = None;
        for (poly_id, coeff) in ctx.dense_polys.iter() {
            match poly_id {
                CommittedPolynomial::RdInc => {
                    rd_inc_coeff = Some(rd_inc_coeff.unwrap_or(F::zero()) + *coeff);
                }
                CommittedPolynomial::RamInc => {
                    ram_inc_coeff = Some(ram_inc_coeff.unwrap_or(F::zero()) + *coeff);
                }
                _ => unreachable!("unexpected poly_id in dense_polys for streaming VMV"),
            }
        }

        let mut instruction_ra: Vec<(usize, F)> = Vec::new();
        let mut bytecode_ra: Vec<(usize, F)> = Vec::new();
        let mut ram_ra: Vec<(usize, F)> = Vec::new();
        for (poly_id, coeff) in ctx.onehot_polys.iter() {
            match poly_id {
                CommittedPolynomial::InstructionRa(idx) => instruction_ra.push((*idx, *coeff)),
                CommittedPolynomial::BytecodeRa(idx) => bytecode_ra.push((*idx, *coeff)),
                CommittedPolynomial::RamRa(idx) => ram_ra.push((*idx, *coeff)),
                _ => unreachable!("unexpected poly_id in onehot_polys for streaming VMV"),
            }
        }

        let has_dense = rd_inc_coeff.is_some() || ram_inc_coeff.is_some();
        let has_onehot =
            !instruction_ra.is_empty() || !bytecode_ra.is_empty() || !ram_ra.is_empty();
        if has_onehot {
            debug_assert!(
                left_vec.len() >= ctx.one_hot_params.k_chunk * num_rows,
                "left_vec too short for one-hot VMV: len={} need_at_least={}",
                left_vec.len(),
                ctx.one_hot_params.k_chunk * num_rows
            );
        }

        // Cache commonly-used params to reduce indirection in hot loops.
        let one_hot_params = &ctx.one_hot_params;
        let log_k_chunk = one_hot_params.log_k_chunk;
        let k_chunk_mask_usize = one_hot_params.k_chunk - 1;
        let k_chunk_mask_u64 = k_chunk_mask_usize as u64;
        let k_chunk_mask_u128 = k_chunk_mask_usize as u128;
        let instruction_d = one_hot_params.instruction_d;
        let bytecode_d = one_hot_params.bytecode_d;
        let ram_d = one_hot_params.ram_d;
        let bytecode = &ctx.preprocessing.bytecode;
        let memory_layout = &ctx.preprocessing.memory_layout;

        let (dense_accs, onehot_accs) = lazy_trace
            .pad_using(T, |_| Cycle::NoOp)
            .iter_chunks(num_columns)
            .enumerate()
            .par_bridge()
            .fold(
                || {
                    let dense_accs: Vec<Acc6S<F>> = if has_dense {
                        unsafe_allocate_zero_vec(num_columns)
                    } else {
                        Vec::new()
                    };
                    let onehot_accs: Vec<F::Unreduced<9>> = if has_onehot {
                        unsafe_allocate_zero_vec(num_columns)
                    } else {
                        Vec::new()
                    };
                    (dense_accs, onehot_accs)
                },
                |(mut dense_accs, mut onehot_accs), (row_idx, chunk)| {
                    let row_weight = left_vec[row_idx];
                    let scaled_rd_inc = rd_inc_coeff.map(|c| row_weight * c);
                    let scaled_ram_inc = ram_inc_coeff.map(|c| row_weight * c);
                    let row_base_ptr = if has_onehot {
                        Some(unsafe { left_vec.as_ptr().add(row_idx) })
                    } else {
                        None
                    };

                    // Process columns within chunk sequentially (avoid nested parallelism).
                    for (col_idx, cycle) in chunk.iter().enumerate() {
                        if has_dense {
                            let dense_acc = &mut dense_accs[col_idx];

                            if let Some(scaled) = &scaled_rd_inc {
                                let (_, pre_value, post_value) = cycle.rd_write();
                                let diff = s64_from_diff_u64s(post_value, pre_value);
                                dense_acc.fmadd(scaled, &diff);
                            }

                            if let Some(scaled) = &scaled_ram_inc {
                                match cycle.ram_access() {
                                    tracer::instruction::RAMAccess::Write(write) => {
                                        let diff =
                                            s64_from_diff_u64s(write.post_value, write.pre_value);
                                        dense_acc.fmadd(scaled, &diff);
                                    }
                                    tracer::instruction::RAMAccess::Read(_)
                                    | tracer::instruction::RAMAccess::NoOp => {}
                                }
                            }
                        }

                        if has_onehot {
                            let row_base_ptr = row_base_ptr.expect("row_base_ptr missing");
                            let onehot_acc = &mut onehot_accs[col_idx];

                            if !instruction_ra.is_empty() {
                                let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                                for (idx, coeff) in instruction_ra.iter() {
                                    debug_assert!(*idx < instruction_d);
                                    let shift = log_k_chunk * (instruction_d - 1 - *idx);
                                    let k = ((lookup_index >> shift) & k_chunk_mask_u128) as usize;
                                    let left_val = unsafe { *row_base_ptr.add(k * num_rows) };
                                    *onehot_acc += left_val.mul_unreduced::<9>(*coeff);
                                }
                            }

                            if !bytecode_ra.is_empty() {
                                let pc = bytecode.get_pc(cycle);
                                for (idx, coeff) in bytecode_ra.iter() {
                                    debug_assert!(*idx < bytecode_d);
                                    let shift = log_k_chunk * (bytecode_d - 1 - *idx);
                                    let k = (pc >> shift) & k_chunk_mask_usize;
                                    let left_val = unsafe { *row_base_ptr.add(k * num_rows) };
                                    *onehot_acc += left_val.mul_unreduced::<9>(*coeff);
                                }
                            }

                            if !ram_ra.is_empty() {
                                let address = cycle.ram_access().address() as u64;
                                if let Some(remapped) = remap_address(address, memory_layout) {
                                    for (idx, coeff) in ram_ra.iter() {
                                        debug_assert!(*idx < ram_d);
                                        let shift = log_k_chunk * (ram_d - 1 - *idx);
                                        let k = ((remapped >> shift) & k_chunk_mask_u64) as usize;
                                        let left_val = unsafe { *row_base_ptr.add(k * num_rows) };
                                        *onehot_acc += left_val.mul_unreduced::<9>(*coeff);
                                    }
                                }
                            }
                        }
                    }

                    (dense_accs, onehot_accs)
                },
            )
            .reduce(
                || {
                    let dense_accs: Vec<Acc6S<F>> = if has_dense {
                        unsafe_allocate_zero_vec(num_columns)
                    } else {
                        Vec::new()
                    };
                    let onehot_accs: Vec<F::Unreduced<9>> = if has_onehot {
                        unsafe_allocate_zero_vec(num_columns)
                    } else {
                        Vec::new()
                    };
                    (dense_accs, onehot_accs)
                },
                |(mut dense_a, mut onehot_a), (dense_b, onehot_b)| {
                    if has_dense {
                        for (a, b) in dense_a.iter_mut().zip(dense_b.iter()) {
                            *a = *a + *b;
                        }
                    }
                    if has_onehot {
                        for (a, b) in onehot_a.iter_mut().zip(onehot_b.iter()) {
                            *a += *b;
                        }
                    }
                    (dense_a, onehot_a)
                },
            );

        // Finalize: reduce once per column and combine dense + onehot contributions (parallel).
        (0..num_columns)
            .into_par_iter()
            .map(|col_idx| {
                let mut val = F::zero();
                if has_dense {
                    val += dense_accs[col_idx].barrett_reduce();
                }
                if has_onehot {
                    val += F::from_montgomery_reduce::<9>(onehot_accs[col_idx]);
                }
                val
            })
            .collect()
    }
}
