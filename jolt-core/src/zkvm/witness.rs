#![allow(static_mut_refs)]

use allocative::Allocative;
use common::constants::XLEN;
use common::jolt_device::MemoryLayout;
use itertools::Itertools;
use jolt_field::{Fr as SourceField, FromPrimitiveInt};
use jolt_openings::{
    BatchCommitmentSource, CommitmentSource, OneHotEntries, OneHotIndex, OneHotRow, SourceRow,
};
use jolt_poly::{MultilinearPoly, Polynomial as SourcePolynomial};
use rayon::prelude::*;
use tracer::{instruction::Cycle, ChunksIterator};

use crate::zkvm::bytecode::{get_pc_for_cycle, BytecodePreprocessing};
use crate::zkvm::config::OneHotParams;
use crate::zkvm::instruction::InstructionFlags;
use crate::zkvm::verifier::JoltSharedPreprocessing;
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    poly::{
        commitment::dory::{DoryContext, DoryGlobals, DoryLayout},
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        one_hot_polynomial::OneHotPolynomial,
    },
    zkvm::ram::remap_address,
};

use super::instruction::{CircuitFlags, LookupQuery};

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum CommittedPolynomial {
    /*  Twist/Shout witnesses */
    /// Inc polynomial for the registers instance of Twist
    RdInc,
    /// Inc polynomial for the RAM instance of Twist
    RamInc,
    /// One-hot ra polynomial for the instruction lookups instance of Shout.
    /// There are d=8 of these polynomials, `InstructionRa(0) .. InstructionRa(7)`
    InstructionRa(usize),
    /// One-hot ra polynomial for the bytecode instance of Shout
    BytecodeRa(usize),
    /// One-hot ra/wa polynomial for the RAM instance of Twist
    /// Note that for RAM, ra and wa are the same polynomial because
    /// there is at most one load or store per cycle.
    RamRa(usize),
    /// Trusted advice polynomial - committed before proving, verifier has commitment.
    /// Length cannot exceed max_trace_length.
    TrustedAdvice,
    /// Untrusted advice polynomial - committed during proving, commitment in proof.
    /// Length cannot exceed max_trace_length.
    UntrustedAdvice,
}

/// Returns a list of symbols representing all committed polynomials.
pub fn all_committed_polynomials(one_hot_params: &OneHotParams) -> Vec<CommittedPolynomial> {
    let mut polynomials = vec![CommittedPolynomial::RdInc, CommittedPolynomial::RamInc];
    for i in 0..one_hot_params.instruction_d {
        polynomials.push(CommittedPolynomial::InstructionRa(i));
    }
    for i in 0..one_hot_params.ram_d {
        polynomials.push(CommittedPolynomial::RamRa(i));
    }
    for i in 0..one_hot_params.bytecode_d {
        polynomials.push(CommittedPolynomial::BytecodeRa(i));
    }
    polynomials
}

/// Commitment-source view over a materialized Jolt multilinear polynomial.
///
/// The blanket `MultilinearPoly` source implementation is intentionally
/// backend-neutral and exposes rows as field elements. Jolt's materialized
/// witness/advice polynomials often retain smaller scalar encodings, and Dory
/// can commit those rows without first expanding them into field elements. This
/// adapter keeps that compact row information available for direct,
/// non-streaming commits while still delegating evaluation, one-hot traversal,
/// and opening-time row folds to the underlying polynomial.
pub struct PolynomialCommitmentSource<'a, F: JoltField>(pub &'a MultilinearPolynomial<F>);

fn address_major_dense_shape() -> Option<(usize, usize)> {
    (DoryGlobals::current_context() == DoryContext::Main
        && DoryGlobals::get_layout() == DoryLayout::AddressMajor)
        .then(|| {
            (
                DoryGlobals::address_major_cycles_per_row(),
                DoryGlobals::k_from_matrix_shape(),
            )
        })
}

impl<F> CommitmentSource<F> for PolynomialCommitmentSource<'_, F>
where
    F: JoltField + jolt_field::Field + ChallengeFieldOps<F> + FieldChallengeOps<F>,
    for<'a> &'a F::Challenge: Into<F>,
{
    fn num_vars(&self) -> usize {
        MultilinearPoly::num_vars(self.0)
    }

    fn evaluate(&self, point: &[F]) -> F {
        PolynomialEvaluation::evaluate(self.0, point)
    }

    fn for_each_row<V>(&self, sigma: usize, mut visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, F>),
    {
        match self.0 {
            MultilinearPolynomial::I128Scalars(poly) => {
                let strided = address_major_dense_shape();
                let row_len = strided.map_or(1usize << sigma, |(row_len, _)| row_len);
                for (row_index, row) in poly.coeffs.chunks(row_len).enumerate() {
                    if let Some((_, column_stride)) = strided {
                        visit(
                            row_index,
                            SourceRow::StridedI128 {
                                values: row,
                                column_stride,
                            },
                        );
                    } else {
                        visit(row_index, SourceRow::I128(row));
                    }
                }
            }
            MultilinearPolynomial::U64Scalars(poly) => {
                let strided = address_major_dense_shape();
                let row_len = strided.map_or(1usize << sigma, |(row_len, _)| row_len);
                for (row_index, row) in poly.coeffs.chunks(row_len).enumerate() {
                    if let Some((_, column_stride)) = strided {
                        visit(
                            row_index,
                            SourceRow::StridedU64 {
                                values: row,
                                column_stride,
                            },
                        );
                    } else {
                        visit(row_index, SourceRow::U64(row));
                    }
                }
            }
            _ => {
                if let Some((row_len, column_stride)) = address_major_dense_shape() {
                    let row_sigma = row_len.trailing_zeros() as usize;
                    MultilinearPoly::for_each_row(self.0, row_sigma, &mut |row_index, row| {
                        visit(
                            row_index,
                            SourceRow::StridedFieldElements {
                                values: row,
                                column_stride,
                            },
                        );
                    });
                } else {
                    MultilinearPoly::for_each_row(self.0, sigma, &mut |row_index, row| {
                        visit(row_index, SourceRow::FieldElements(row));
                    });
                }
            }
        }
    }

    fn map_rows<R, V>(&self, sigma: usize, visit: V) -> Vec<R>
    where
        R: Send,
        V: for<'row> Fn(usize, SourceRow<'row, F>) -> R + Send + Sync,
    {
        match self.0 {
            MultilinearPolynomial::I128Scalars(poly) => {
                let strided = address_major_dense_shape();
                let row_len = strided.map_or(1usize << sigma, |(row_len, _)| row_len);
                poly.coeffs
                    .par_chunks(row_len)
                    .enumerate()
                    .map(|(row_index, row)| {
                        if let Some((_, column_stride)) = strided {
                            visit(
                                row_index,
                                SourceRow::StridedI128 {
                                    values: row,
                                    column_stride,
                                },
                            )
                        } else {
                            visit(row_index, SourceRow::I128(row))
                        }
                    })
                    .collect()
            }
            MultilinearPolynomial::U64Scalars(poly) => {
                let strided = address_major_dense_shape();
                let row_len = strided.map_or(1usize << sigma, |(row_len, _)| row_len);
                poly.coeffs
                    .par_chunks(row_len)
                    .enumerate()
                    .map(|(row_index, row)| {
                        if let Some((_, column_stride)) = strided {
                            visit(
                                row_index,
                                SourceRow::StridedU64 {
                                    values: row,
                                    column_stride,
                                },
                            )
                        } else {
                            visit(row_index, SourceRow::U64(row))
                        }
                    })
                    .collect()
            }
            _ => {
                let mut rows = Vec::new();
                self.for_each_row(sigma, |row_index, row| {
                    rows.push(visit(row_index, row));
                });
                rows
            }
        }
    }

    fn is_one_hot(&self) -> bool {
        matches!(self.0, MultilinearPolynomial::OneHot(_)) || MultilinearPoly::is_one_hot(self.0)
    }

    fn for_each_one<V>(&self, mut visit: V)
    where
        V: FnMut(usize),
    {
        match self.0 {
            MultilinearPolynomial::OneHot(poly) => {
                let layout = DoryGlobals::get_layout();
                let t = poly.nonzero_indices.len();
                for (cycle, address) in poly.nonzero_indices.iter().enumerate() {
                    if let Some(address) = address {
                        visit(layout.address_cycle_to_index(*address as usize, cycle, poly.K, t));
                    }
                }
            }
            _ => MultilinearPoly::for_each_one(self.0, &mut visit),
        }
    }

    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        MultilinearPoly::fold_rows(self.0, left, sigma)
    }
}

/// Trace-backed batch source for the current CycleMajor witness commitments.
///
/// This adapter exposes Jolt's streaming witness path through the
/// source-oriented PCS API. It does not materialize witness polynomials in
/// the hot batch path: [`map_rows`](BatchCommitmentSource::map_rows) clones the
/// lazy trace iterator, pads it exactly as the current prover does, chunks it by
/// the existing Dory row width, and derives all requested committed-polynomial
/// rows from each trace chunk before moving to the next chunk.
///
/// The single-source methods exist for PCS backends that do not use batch row
/// traversal. They may materialize a source for evaluation or non-native row
/// shapes, so CycleMajor Dory should use the batch traversal to preserve the
/// current no-regression streaming behavior.
pub struct CycleMajorTraceBatch<'a, I>
where
    I: Iterator<Item = Cycle> + Clone + Send + Sync,
{
    trace: I,
    preprocessing: &'a JoltSharedPreprocessing,
    one_hot_params: &'a OneHotParams,
    source_ids: Vec<CommittedPolynomial>,
    padded_trace_len: usize,
    row_len: usize,
}

impl<'a, I> CycleMajorTraceBatch<'a, I>
where
    I: Iterator<Item = Cycle> + Clone + Send + Sync,
{
    pub fn new(
        trace: I,
        preprocessing: &'a JoltSharedPreprocessing,
        one_hot_params: &'a OneHotParams,
        source_ids: Vec<CommittedPolynomial>,
        padded_trace_len: usize,
        row_len: usize,
    ) -> Self {
        assert!(
            row_len != 0,
            "CycleMajor trace commitment rows must be non-empty"
        );
        assert!(
            padded_trace_len.is_multiple_of(row_len),
            "padded trace length ({padded_trace_len}) must be divisible by row length ({row_len})",
        );

        Self {
            trace,
            preprocessing,
            one_hot_params,
            source_ids,
            padded_trace_len,
            row_len,
        }
    }

    fn log_trace_len(&self) -> usize {
        self.padded_trace_len.trailing_zeros() as usize
    }

    fn one_hot_num_vars(&self) -> usize {
        self.log_trace_len() + self.one_hot_params.log_k_chunk
    }

    fn log_domain_size(&self) -> u8 {
        self.one_hot_params.log_k_chunk as u8
    }

    fn for_each_native_row<V>(&self, id: CommittedPolynomial, mut visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, SourceField>),
    {
        self.trace
            .clone()
            .pad_using(self.padded_trace_len, |_| Cycle::NoOp)
            .iter_chunks(self.row_len)
            .enumerate()
            .for_each(|(row_index, row_cycles)| {
                let row = self.row_for_id(id, &row_cycles);
                row.with_borrowed(self.log_domain_size(), |borrowed| {
                    visit(row_index, borrowed);
                });
            });
    }

    fn row_for_id(&self, id: CommittedPolynomial, row_cycles: &[Cycle]) -> OwnedSourceRow {
        match id {
            CommittedPolynomial::RdInc => OwnedSourceRow::I128(
                row_cycles
                    .iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write().unwrap_or_default();
                        post_value as i128 - pre_value as i128
                    })
                    .collect(),
            ),
            CommittedPolynomial::RamInc => OwnedSourceRow::I128(
                row_cycles
                    .iter()
                    .map(|cycle| match cycle.ram_access() {
                        tracer::instruction::RAMAccess::Write(write) => {
                            write.post_value as i128 - write.pre_value as i128
                        }
                        _ => 0,
                    })
                    .collect(),
            ),
            CommittedPolynomial::InstructionRa(idx) => OwnedSourceRow::OnePerColumn(
                row_cycles
                    .iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        self.one_hot_index(
                            self.one_hot_params.lookup_index_chunk(lookup_index, idx),
                        )
                    })
                    .collect(),
            ),
            CommittedPolynomial::BytecodeRa(idx) => OwnedSourceRow::OnePerColumn(
                row_cycles
                    .iter()
                    .map(|cycle| {
                        let pc = get_pc_for_cycle(&self.preprocessing.bytecode, cycle);
                        self.one_hot_index(self.one_hot_params.bytecode_pc_chunk(pc, idx))
                    })
                    .collect(),
            ),
            CommittedPolynomial::RamRa(idx) => OwnedSourceRow::MaybeZero(
                row_cycles
                    .iter()
                    .map(|cycle| {
                        remap_address(
                            cycle.ram_access().address() as u64,
                            &self.preprocessing.memory_layout,
                        )
                        .map(|address| {
                            self.one_hot_index(self.one_hot_params.ram_address_chunk(address, idx))
                        })
                    })
                    .collect(),
            ),
            CommittedPolynomial::TrustedAdvice | CommittedPolynomial::UntrustedAdvice => {
                panic!("advice polynomials are not part of the CycleMajor trace batch")
            }
        }
    }

    fn one_hot_index(&self, value: u8) -> OneHotIndex {
        match OneHotIndex::new(value, self.log_domain_size()) {
            Some(index) => index,
            None => panic!(
                "one-hot index {value} does not fit log-domain size {}",
                self.log_domain_size()
            ),
        }
    }

    fn materialize_source(&self, id: CommittedPolynomial) -> SourcePolynomial<SourceField> {
        let mut dense = Vec::new();
        match id {
            CommittedPolynomial::RdInc | CommittedPolynomial::RamInc => {
                dense.reserve(self.padded_trace_len);
                self.for_each_native_row(id, |_, row| match row {
                    SourceRow::I128(values) => {
                        dense.extend(
                            values
                                .iter()
                                .map(|&value| <SourceField as FromPrimitiveInt>::from_i128(value)),
                        );
                    }
                    SourceRow::FieldElements(_)
                    | SourceRow::StridedFieldElements { .. }
                    | SourceRow::U64(_)
                    | SourceRow::StridedU64 { .. }
                    | SourceRow::StridedI128 { .. }
                    | SourceRow::OneHot(_) => {
                        panic!("increment rows must be emitted as i128 source rows");
                    }
                });
            }
            CommittedPolynomial::InstructionRa(_)
            | CommittedPolynomial::BytecodeRa(_)
            | CommittedPolynomial::RamRa(_) => {
                dense.resize(
                    self.padded_trace_len * self.one_hot_params.k_chunk,
                    <SourceField as FromPrimitiveInt>::from_u64(0),
                );
                let mut cycle_offset = 0;
                self.for_each_native_row(id, |_, row| match row {
                    SourceRow::OneHot(row) => {
                        let entries_len = match row.entries {
                            OneHotEntries::OnePerColumn(indices) => {
                                for (column, index) in indices.iter().enumerate() {
                                    dense[index.get() * self.padded_trace_len
                                        + cycle_offset
                                        + column] = <SourceField as FromPrimitiveInt>::from_u64(1);
                                }
                                indices.len()
                            }
                            OneHotEntries::MaybeZero(indices) => {
                                for (column, index) in indices.iter().enumerate() {
                                    if let Some(index) = index {
                                        dense[index.get() * self.padded_trace_len
                                            + cycle_offset
                                            + column] =
                                            <SourceField as FromPrimitiveInt>::from_u64(1);
                                    }
                                }
                                indices.len()
                            }
                        };
                        cycle_offset += entries_len;
                    }
                    SourceRow::FieldElements(_)
                    | SourceRow::StridedFieldElements { .. }
                    | SourceRow::I128(_)
                    | SourceRow::StridedI128 { .. }
                    | SourceRow::U64(_)
                    | SourceRow::StridedU64 { .. } => {
                        panic!("RA rows must be emitted as one-hot source rows");
                    }
                });
            }
            CommittedPolynomial::TrustedAdvice | CommittedPolynomial::UntrustedAdvice => {
                panic!("advice polynomials are not part of the CycleMajor trace batch")
            }
        }
        SourcePolynomial::new(dense)
    }
}

impl<I> BatchCommitmentSource<SourceField> for CycleMajorTraceBatch<'_, I>
where
    I: Iterator<Item = Cycle> + Clone + Send + Sync,
{
    type Id = CommittedPolynomial;

    type Source<'a>
        = CycleMajorTraceSource<'a, I>
    where
        Self: 'a;

    fn source_ids(&self) -> &[Self::Id] {
        &self.source_ids
    }

    fn num_vars(&self, id: Self::Id) -> usize {
        match id {
            CommittedPolynomial::RdInc | CommittedPolynomial::RamInc => self.log_trace_len(),
            CommittedPolynomial::InstructionRa(_)
            | CommittedPolynomial::BytecodeRa(_)
            | CommittedPolynomial::RamRa(_) => self.one_hot_num_vars(),
            CommittedPolynomial::TrustedAdvice | CommittedPolynomial::UntrustedAdvice => {
                panic!("advice polynomials are not part of the CycleMajor trace batch")
            }
        }
    }

    fn source(&self, id: Self::Id) -> Self::Source<'_> {
        CycleMajorTraceSource { batch: self, id }
    }

    fn map_rows<R, V>(&self, sigma: usize, ids: &[Self::Id], visit: V) -> Vec<Vec<R>>
    where
        R: Send,
        V: for<'row> Fn(Self::Id, SourceRow<'row, SourceField>) -> R + Send + Sync,
    {
        let row_len = 1usize << sigma;
        assert_eq!(
            row_len, self.row_len,
            "CycleMajor trace batch currently preserves the native streaming row width",
        );

        let rows: Vec<_> = self
            .trace
            .clone()
            .pad_using(self.padded_trace_len, |_| Cycle::NoOp)
            .iter_chunks(self.row_len)
            .enumerate()
            .par_bridge()
            .map(|(row_index, row_cycles)| {
                let row = ids
                    .iter()
                    .map(|&id| {
                        let owned = self.row_for_id(id, &row_cycles);
                        owned.with_borrowed(self.log_domain_size(), |borrowed| visit(id, borrowed))
                    })
                    .collect();
                (row_index, row)
            })
            .collect();

        let row_count = self.padded_trace_len / self.row_len;
        let mut ordered_rows: Vec<Option<Vec<R>>> = (0..row_count).map(|_| None).collect();
        for (row_index, row) in rows {
            ordered_rows[row_index] = Some(row);
        }
        ordered_rows
            .into_iter()
            .enumerate()
            .map(|(row_index, row)| match row {
                Some(row) => row,
                None => panic!("missing CycleMajor trace row {row_index}"),
            })
            .collect()
    }
}

pub struct CycleMajorTraceSource<'a, I>
where
    I: Iterator<Item = Cycle> + Clone + Send + Sync,
{
    batch: &'a CycleMajorTraceBatch<'a, I>,
    id: CommittedPolynomial,
}

impl<I> CommitmentSource<SourceField> for CycleMajorTraceSource<'_, I>
where
    I: Iterator<Item = Cycle> + Clone + Send + Sync,
{
    fn num_vars(&self) -> usize {
        self.batch.num_vars(self.id)
    }

    fn evaluate(&self, point: &[SourceField]) -> SourceField {
        self.batch.materialize_source(self.id).evaluate(point)
    }

    fn for_each_row<V>(&self, sigma: usize, mut visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, SourceField>),
    {
        if (1usize << sigma) == self.batch.row_len {
            self.batch.for_each_native_row(self.id, visit);
        } else {
            let materialized = self.batch.materialize_source(self.id);
            MultilinearPoly::for_each_row(&materialized, sigma, &mut |row_index, row| {
                visit(row_index, SourceRow::FieldElements(row));
            });
        }
    }

    fn fold_rows(&self, left: &[SourceField], sigma: usize) -> Vec<SourceField> {
        MultilinearPoly::fold_rows(&self.batch.materialize_source(self.id), left, sigma)
    }
}

enum OwnedSourceRow {
    I128(Vec<i128>),
    OnePerColumn(Vec<OneHotIndex>),
    MaybeZero(Vec<Option<OneHotIndex>>),
}

impl OwnedSourceRow {
    fn with_borrowed<R>(
        &self,
        log_domain_size: u8,
        visit: impl for<'row> FnOnce(SourceRow<'row, SourceField>) -> R,
    ) -> R {
        match self {
            Self::I128(values) => visit(SourceRow::I128(values)),
            Self::OnePerColumn(indices) => visit(SourceRow::OneHot(OneHotRow {
                log_domain_size,
                entries: OneHotEntries::OnePerColumn(indices),
            })),
            Self::MaybeZero(indices) => visit(SourceRow::OneHot(OneHotRow {
                log_domain_size,
                entries: OneHotEntries::MaybeZero(indices),
            })),
        }
    }
}

#[cfg(test)]
mod cycle_major_trace_batch_tests {
    use std::sync::Arc;

    use common::jolt_device::MemoryLayout;
    use jolt_openings::BatchCommitmentSource;
    use tracer::instruction::Cycle;

    use crate::zkvm::{
        bytecode::BytecodePreprocessing,
        config::{OneHotConfig, OneHotParams},
        ram::RAMPreprocessing,
        verifier::JoltSharedPreprocessing,
    };

    use super::{CommittedPolynomial, CycleMajorTraceBatch, OneHotEntries, SourceField, SourceRow};
    use jolt_field::FromPrimitiveInt;

    #[derive(Debug, PartialEq, Eq)]
    enum RowSnapshot {
        I128(Vec<i128>),
        U64(Vec<u64>),
        StridedI128(Vec<i128>, usize),
        StridedU64(Vec<u64>, usize),
        OnePerColumn(Vec<usize>),
        MaybeZero(Vec<Option<usize>>),
        FieldElements(Vec<SourceField>),
        StridedFieldElements(Vec<SourceField>, usize),
    }

    fn snapshot(row: SourceRow<'_, SourceField>) -> RowSnapshot {
        match row {
            SourceRow::I128(values) => RowSnapshot::I128(values.to_vec()),
            SourceRow::StridedI128 {
                values,
                column_stride,
            } => RowSnapshot::StridedI128(values.to_vec(), column_stride),
            SourceRow::U64(values) => RowSnapshot::U64(values.to_vec()),
            SourceRow::StridedU64 {
                values,
                column_stride,
            } => RowSnapshot::StridedU64(values.to_vec(), column_stride),
            SourceRow::OneHot(row) => match row.entries {
                OneHotEntries::OnePerColumn(indices) => {
                    RowSnapshot::OnePerColumn(indices.iter().map(|index| index.get()).collect())
                }
                OneHotEntries::MaybeZero(indices) => RowSnapshot::MaybeZero(
                    indices
                        .iter()
                        .map(|index| index.map(|index| index.get()))
                        .collect(),
                ),
            },
            SourceRow::FieldElements(values) => RowSnapshot::FieldElements(values.to_vec()),
            SourceRow::StridedFieldElements {
                values,
                column_stride,
            } => RowSnapshot::StridedFieldElements(values.to_vec(), column_stride),
        }
    }

    fn preprocessing() -> JoltSharedPreprocessing {
        JoltSharedPreprocessing {
            bytecode: Arc::new(BytecodePreprocessing::default()),
            ram: RAMPreprocessing::default(),
            memory_layout: MemoryLayout::default(),
            max_padded_trace_length: 8,
        }
    }

    fn one_hot_params() -> OneHotParams {
        OneHotParams::from_config(
            &OneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
            16,
            16,
        )
    }

    #[test]
    fn map_rows_matches_existing_cycle_major_row_shapes() {
        let preprocessing = preprocessing();
        let one_hot_params = one_hot_params();
        let ids = vec![
            CommittedPolynomial::RdInc,
            CommittedPolynomial::RamInc,
            CommittedPolynomial::InstructionRa(0),
            CommittedPolynomial::BytecodeRa(0),
            CommittedPolynomial::RamRa(0),
        ];
        let batch = CycleMajorTraceBatch::new(
            vec![Cycle::NoOp; 8].into_iter(),
            &preprocessing,
            &one_hot_params,
            ids.clone(),
            8,
            4,
        );

        let rows = batch.map_rows(2, &ids, |_, row| snapshot(row));

        assert_eq!(batch.source_ids(), ids.as_slice());
        assert_eq!(rows.len(), 2);
        for row in rows {
            assert_eq!(
                row,
                vec![
                    RowSnapshot::I128(vec![0, 0, 0, 0]),
                    RowSnapshot::I128(vec![0, 0, 0, 0]),
                    RowSnapshot::OnePerColumn(vec![0, 0, 0, 0]),
                    RowSnapshot::OnePerColumn(vec![0, 0, 0, 0]),
                    RowSnapshot::MaybeZero(vec![None, None, None, None]),
                ]
            );
        }
    }

    #[test]
    fn materialized_one_hot_source_uses_hot_index_major_order() {
        let preprocessing = preprocessing();
        let one_hot_params = one_hot_params();
        let batch = CycleMajorTraceBatch::new(
            vec![Cycle::NoOp; 8].into_iter(),
            &preprocessing,
            &one_hot_params,
            vec![CommittedPolynomial::InstructionRa(0)],
            8,
            4,
        );

        let poly = batch.materialize_source(CommittedPolynomial::InstructionRa(0));
        let evals = poly.evaluations();

        assert_eq!(evals.len(), 8 * one_hot_params.k_chunk);
        assert!(evals[..8]
            .iter()
            .all(|&value| value == SourceField::from_u64(1)));
        assert!(evals[8..]
            .iter()
            .all(|&value| value == SourceField::from_u64(0)));
    }
}

impl CommittedPolynomial {
    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness")]
    pub fn generate_witness<F>(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
        trace: &[Cycle],
        one_hot_params: Option<&OneHotParams>,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
    {
        match self {
            CommittedPolynomial::BytecodeRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let pc = get_pc_for_cycle(bytecode_preprocessing, cycle);
                        Some(one_hot_params.bytecode_pc_chunk(pc, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
            CommittedPolynomial::RamRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        remap_address(cycle.ram_access().address() as u64, memory_layout)
                            .map(|address| one_hot_params.ram_address_chunk(address, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
            CommittedPolynomial::RdInc => {
                let coeffs: Vec<i128> = trace
                    .par_iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write().unwrap_or_default();
                        post_value as i128 - pre_value as i128
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::RamInc => {
                let coeffs: Vec<i128> = trace
                    .par_iter()
                    .map(|cycle| {
                        let ram_op = cycle.ram_access();
                        match ram_op {
                            tracer::instruction::RAMAccess::Write(write) => {
                                write.post_value as i128 - write.pre_value as i128
                            }
                            _ => 0,
                        }
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::InstructionRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        Some(one_hot_params.lookup_index_chunk(lookup_index, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
            CommittedPolynomial::TrustedAdvice | CommittedPolynomial::UntrustedAdvice => {
                panic!("Advice polynomials should not use generate_witness")
            }
        }
    }

    pub fn get_onehot_k(&self, one_hot_params: &OneHotParams) -> Option<usize> {
        match self {
            CommittedPolynomial::InstructionRa(_)
            | CommittedPolynomial::BytecodeRa(_)
            | CommittedPolynomial::RamRa(_) => Some(one_hot_params.k_chunk),
            _ => None,
        }
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum VirtualPolynomial {
    PC,
    UnexpandedPC,
    NextPC,
    NextUnexpandedPC,
    NextIsNoop,
    NextIsVirtual,
    NextIsFirstInSequence,
    LeftLookupOperand,
    RightLookupOperand,
    LeftInstructionInput,
    RightInstructionInput,
    Product,
    ShouldJump,
    ShouldBranch,
    Rd,
    Imm,
    Rs1Value,
    Rs2Value,
    RdWriteValue,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    LookupOutput,
    InstructionRaf,
    InstructionRafFlag,
    InstructionRa(usize),
    RegistersVal,
    RamAddress,
    RamRa,
    RamReadValue,
    RamWriteValue,
    RamVal,
    RamValInit,
    RamValFinal,
    RamHammingWeight,
    UnivariateSkip,
    OpFlags(CircuitFlags),
    InstructionFlags(InstructionFlags),
    LookupTableFlag(usize),
}
