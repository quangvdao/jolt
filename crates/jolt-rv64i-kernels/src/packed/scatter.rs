//! Cycle weights are scattered by bytecode row through 256 destination ranges.
//!
//! The plan counts each deterministic cycle chunk's destinations once. A scatter
//! emits rows and weights in cycle order into disjoint portions of one buffer,
//! then applies each range's pairs to its own output slice. Counts and offsets
//! do not depend on the Rayon pool; all combinations are field XORs.

use std::mem::size_of;
use std::sync::Arc;

use jolt_field::F128;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
use thiserror::Error;

use crate::par::{CycleChunks, ParError};
use crate::source::{CycleSource, ValidatedTrace};

const RANGES: usize = 256;

/// Unrepresentable scatter dimensions or incorrectly sized caller buffers.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ScatterError {
    #[error("bytecode row count {rows} requires an index above {limit}")]
    RowCount { rows: usize, limit: usize },
    #[error("cycle count {cycles} exceeds the count limit {limit}")]
    CycleCount { cycles: usize, limit: usize },
    #[error("scatter {buffer} with {len} elements of {element_size} bytes cannot be represented")]
    StorageSize {
        buffer: &'static str,
        len: usize,
        element_size: usize,
    },
    #[error("scatter {buffer} has {actual} elements, expected {expected}")]
    BufferLength {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error(transparent)]
    Geometry(#[from] ParError),
}

fn check_storage<T>(len: usize, buffer: &'static str) -> Result<(), ScatterError> {
    let element_size = size_of::<T>();
    if len
        .checked_mul(element_size)
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(ScatterError::StorageSize {
            buffer,
            len,
            element_size,
        });
    }
    Ok(())
}

#[derive(Default)]
struct Segment<'a> {
    rows: &'a mut [u32],
    weights: &'a mut [F128],
}

/// Counts and offsets for summing cycle weights at their bytecode row indices.
///
/// The source's immutable dimensions and indices must already be checked by
/// [`ValidatedTrace`]. Prefix counts use four bytes per chunk and row range;
/// scatter scratch consists of one packed row and one field weight per cycle.
pub struct ScatterPlan<S: CycleSource> {
    trace: Arc<ValidatedTrace<S>>,
    prefix: Vec<[u32; RANGES]>,
    offsets: [usize; RANGES + 1],
    cycles: usize,
    rows: usize,
    chunk_len: usize,
    range_shift: u32,
}

impl<S: CycleSource> ScatterPlan<S> {
    /// Count destinations in chunks chosen by [`CycleChunks`] at round zero.
    ///
    /// Counts must fit `u32`, packed row indices must fit `u32`, and all scratch
    /// and output capacities must fit a Rust slice. One row and fewer than 256
    /// rows are supported; unused destination ranges are empty.
    pub fn new(trace: Arc<ValidatedTrace<S>>) -> Result<Self, ScatterError> {
        let source = trace.source();
        let cycles = source.cycles();
        let rows = source.bytecode_rows();
        if rows - 1 > u32::MAX as usize {
            return Err(ScatterError::RowCount {
                rows,
                limit: u32::MAX as usize,
            });
        }
        if cycles > u32::MAX as usize {
            return Err(ScatterError::CycleCount {
                cycles,
                limit: u32::MAX as usize,
            });
        }
        let geometry = CycleChunks::new(cycles.ilog2() as usize, 0)?;
        let chunk_len = geometry.chunk_len();
        let chunks = cycles / chunk_len;
        check_storage::<[u32; RANGES]>(chunks + 1, "counts")?;
        check_storage::<u32>(cycles, "pair rows")?;
        check_storage::<F128>(cycles, "pair weights")?;
        check_storage::<F128>(rows, "output")?;
        let range_shift = rows.ilog2().saturating_sub(8);
        let mut prefix = vec![[0_u32; RANGES]; chunks + 1];
        for chunk in 0..chunks {
            prefix[chunk + 1] = prefix[chunk];
            for cycle in chunk * chunk_len..(chunk + 1) * chunk_len {
                prefix[chunk + 1][source.bytecode_index(cycle) >> range_shift] += 1;
            }
        }
        let mut offsets = [0; RANGES + 1];
        for range in 0..RANGES {
            offsets[range + 1] = offsets[range] + prefix[chunks][range] as usize;
        }
        Ok(Self {
            trace,
            prefix,
            offsets,
            cycles,
            rows,
            chunk_len,
            range_shift,
        })
    }

    /// Number of cycles whose weights a scatter consumes.
    pub fn cycles(&self) -> usize {
        self.cycles
    }

    /// Number of output elements, including bytecode rows no cycle visits.
    pub fn bytecode_rows(&self) -> usize {
        self.rows
    }

    /// Return `out[k] = Σ_{j: bytecode_index(j) = k} weight(j)`.
    ///
    /// Allocates one row buffer and one weight buffer of the cycle count and a
    /// zeroed output of the bytecode row count; it allocates nothing per chunk,
    /// cycle or pair. The weight function is called once per cycle and must be
    /// a deterministic function of its argument for a deterministic result.
    pub fn scatter(
        &self,
        weight: impl Fn(usize) -> F128 + Sync,
    ) -> Result<Vec<F128>, ScatterError> {
        let mut rows = vec![0; self.cycles];
        let mut weights = vec![F128::from_raw(0); self.cycles];
        let mut output = vec![F128::from_raw(0); self.rows];
        self.scatter_into(weight, &mut rows, &mut weights, &mut output)?;
        Ok(output)
    }

    /// XOR the scatter into caller-provided output using caller-provided pairs.
    ///
    /// Both pair buffers must have exactly [`Self::cycles`] elements and the
    /// output exactly [`Self::bytecode_rows`] elements. The pair buffers are
    /// overwritten; output's initial contribution is retained. A caller wanting
    /// the scatter alone supplies zeroed output. Lengths are checked before any
    /// mutation and no allocation or zero-fill occurs in this pass.
    pub fn scatter_into(
        &self,
        weight: impl Fn(usize) -> F128 + Sync,
        rows: &mut [u32],
        weights: &mut [F128],
        output: &mut [F128],
    ) -> Result<(), ScatterError> {
        for (buffer, expected, actual) in [
            ("pair rows", self.cycles, rows.len()),
            ("pair weights", self.cycles, weights.len()),
            ("output", self.rows, output.len()),
        ] {
            if actual != expected {
                return Err(ScatterError::BufferLength {
                    buffer,
                    expected,
                    actual,
                });
            }
        }
        let mut remaining_rows = &mut *rows;
        let mut remaining_weights = &mut *weights;
        let segments = std::array::from_fn(|range| {
            let len = self.offsets[range + 1] - self.offsets[range];
            let (range_rows, next_rows) = std::mem::take(&mut remaining_rows).split_at_mut(len);
            let (range_weights, next_weights) =
                std::mem::take(&mut remaining_weights).split_at_mut(len);
            remaining_rows = next_rows;
            remaining_weights = next_weights;
            Segment {
                rows: range_rows,
                weights: range_weights,
            }
        });
        self.emit(&weight, segments, 0, self.prefix.len() - 1);
        let range_len = 1 << self.range_shift;
        output
            .par_chunks_mut(range_len)
            .enumerate()
            .for_each(|(range, output)| {
                let start = self.offsets[range];
                let end = self.offsets[range + 1];
                let range_start = range << self.range_shift;
                for (&row, &value) in rows[start..end].iter().zip(&weights[start..end]) {
                    output[row as usize - range_start] += value;
                }
            });
        Ok(())
    }

    fn emit(
        &self,
        weight: &(impl Fn(usize) -> F128 + Sync),
        mut segments: [Segment<'_>; RANGES],
        start: usize,
        end: usize,
    ) {
        if end - start == 1 {
            let source = self.trace.source();
            let mut cursors = [0; RANGES];
            for cycle in start * self.chunk_len..end * self.chunk_len {
                let row = source.bytecode_index(cycle);
                let range = row >> self.range_shift;
                let cursor = cursors[range];
                segments[range].rows[cursor] = row as u32;
                segments[range].weights[cursor] = weight(cycle);
                cursors[range] += 1;
            }
            return;
        }
        let middle = start + (end - start) / 2;
        let left = std::array::from_fn(|range| {
            let segment = std::mem::take(&mut segments[range]);
            let split = (self.prefix[middle][range] - self.prefix[start][range]) as usize;
            let (left_rows, right_rows) = segment.rows.split_at_mut(split);
            let (left_weights, right_weights) = segment.weights.split_at_mut(split);
            segments[range] = Segment {
                rows: right_rows,
                weights: right_weights,
            };
            Segment {
                rows: left_rows,
                weights: left_weights,
            }
        });
        let _ = rayon::join(
            || self.emit(weight, left, start, middle),
            || self.emit(weight, segments, middle, end),
        );
    }
}
