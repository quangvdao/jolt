//! Cycle weights are scattered by bytecode row through 256 destination ranges.
//!
//! The plan counts each deterministic cycle chunk's destinations once. A scatter
//! emits rows and weights in cycle order into disjoint portions of one buffer,
//! then applies each range's pairs to its own output slice. Counts and offsets
//! do not depend on the Rayon pool; all combinations are field XORs.
//! Each chunk owns one contiguous pair segment, grouped by destination range.

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
    #[error("chunk {chunk} range {range} count {count} exceeds {limit}")]
    Count {
        chunk: usize,
        range: usize,
        count: u64,
        limit: usize,
    },
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

/// Counts and offsets for summing cycle weights at their bytecode row indices.
///
/// The source's immutable dimensions and indices must already be checked by
/// [`ValidatedTrace`]. Counts and offsets each use four bytes per chunk and row
/// range. Pair scratch holds one packed row and one field weight per cycle; a
/// flat cursor buffer holds one four-byte cursor per chunk and row range.
pub struct ScatterPlan<S: CycleSource> {
    trace: Arc<ValidatedTrace<S>>,
    counts: Vec<[u32; RANGES]>,
    offsets: Vec<[u32; RANGES]>,
    active: Vec<u8>,
    active_offsets: Vec<usize>,
    cycles: usize,
    rows: usize,
    chunk_len: usize,
    cursor_len: usize,
    range_shift: u32,
}

impl<S: CycleSource> ScatterPlan<S> {
    /// Count destinations in chunks chosen by [`CycleChunks`] at round zero.
    ///
    /// Per-chunk counts and offsets must fit `u32`, packed row indices must fit
    /// `u32`, and all scratch and output capacities must fit a Rust slice. One
    /// row and fewer than 256 rows are supported; unused ranges are empty.
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
        let geometry = CycleChunks::new(cycles.ilog2() as usize, 0)?;
        let chunk_len = geometry.chunk_len();
        let chunks = cycles / chunk_len;
        check_storage::<[u32; RANGES]>(chunks, "counts")?;
        check_storage::<[u32; RANGES]>(chunks, "offsets")?;
        check_storage::<usize>(chunks + 1, "active offsets")?;
        let cursor_len = chunks * RANGES;
        check_storage::<u32>(cursor_len, "cursors")?;
        check_storage::<u8>(cursor_len.min(cycles), "active ranges")?;
        check_storage::<u32>(cycles, "pair rows")?;
        check_storage::<F128>(cycles, "pair weights")?;
        check_storage::<F128>(rows, "output")?;
        let range_shift = rows.ilog2().saturating_sub(8);
        let mut counts = vec![[0_u32; RANGES]; chunks];
        for (chunk, counts) in counts.iter_mut().enumerate() {
            for cycle in chunk * chunk_len..(chunk + 1) * chunk_len {
                let range = source.bytecode_index(cycle) >> range_shift;
                counts[range] = counts[range].checked_add(1).ok_or(ScatterError::Count {
                    chunk,
                    range,
                    count: u64::from(counts[range]) + 1,
                    limit: u32::MAX as usize,
                })?;
            }
        }
        let mut offsets = vec![[0_u32; RANGES]; chunks];
        let mut active = Vec::with_capacity(cursor_len.min(cycles));
        let mut active_offsets = Vec::with_capacity(chunks + 1);
        active_offsets.push(0);
        for (chunk, (counts, offsets)) in counts.iter().zip(&mut offsets).enumerate() {
            let mut next = 0_u32;
            for (range, (&count, offset)) in counts.iter().zip(offsets).enumerate() {
                *offset = next;
                next = next.checked_add(count).ok_or(ScatterError::Count {
                    chunk,
                    range,
                    count: u64::from(next) + u64::from(count),
                    limit: u32::MAX as usize,
                })?;
                if count != 0 {
                    active.push(range as u8);
                }
            }
            active_offsets.push(active.len());
        }
        Ok(Self {
            trace,
            counts,
            offsets,
            active,
            active_offsets,
            cycles,
            rows,
            chunk_len,
            cursor_len,
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

    /// Required length of the reusable cursor buffer passed to a scatter.
    /// Only destination ranges occurring in a chunk are initialized or read.
    pub fn cursor_len(&self) -> usize {
        self.cursor_len
    }

    /// Return `out[k] = Σ_{j: bytecode_index(j) = k} weight(j)`.
    ///
    /// Allocates one row buffer and one weight buffer of the cycle count and a
    /// zeroed output of the bytecode row count and one flat cursor buffer;
    /// it allocates nothing per chunk, cycle or pair. The weight function is
    /// called once per cycle and must be
    /// a deterministic function of its argument for a deterministic result.
    pub fn scatter(
        &self,
        weight: impl Fn(usize) -> F128 + Sync,
    ) -> Result<Vec<F128>, ScatterError> {
        let mut rows = vec![0; self.cycles];
        let mut weights = vec![F128::from_raw(0); self.cycles];
        let mut output = vec![F128::from_raw(0); self.rows];
        let mut cursors = vec![0; self.cursor_len];
        self.scatter_into(weight, &mut rows, &mut weights, &mut output, &mut cursors)?;
        Ok(output)
    }

    /// XOR the scatter into caller-provided output using caller-provided pairs.
    ///
    /// Both pair buffers must have exactly [`Self::cycles`] elements and the
    /// output exactly [`Self::bytecode_rows`] elements, and cursors exactly
    /// [`Self::cursor_len`] elements. The pair buffers are
    /// overwritten; output's initial contribution is retained. A caller wanting
    /// the scatter alone supplies zeroed output. Lengths are checked before any
    /// mutation and no allocation or zero-fill occurs in this pass.
    pub fn scatter_into(
        &self,
        weight: impl Fn(usize) -> F128 + Sync,
        rows: &mut [u32],
        weights: &mut [F128],
        output: &mut [F128],
        cursors: &mut [u32],
    ) -> Result<(), ScatterError> {
        for (buffer, expected, actual) in [
            ("pair rows", self.cycles, rows.len()),
            ("pair weights", self.cycles, weights.len()),
            ("output", self.rows, output.len()),
            ("cursors", self.cursor_len, cursors.len()),
        ] {
            if actual != expected {
                return Err(ScatterError::BufferLength {
                    buffer,
                    expected,
                    actual,
                });
            }
        }
        let source = self.trace.source();
        rows.par_chunks_mut(self.chunk_len)
            .zip(weights.par_chunks_mut(self.chunk_len))
            .zip(cursors.par_chunks_mut(RANGES))
            .enumerate()
            .for_each(|(chunk, ((rows, weights), cursors))| {
                let offsets = &self.offsets[chunk];
                for &range in
                    &self.active[self.active_offsets[chunk]..self.active_offsets[chunk + 1]]
                {
                    cursors[range as usize] = offsets[range as usize];
                }
                let start = chunk * self.chunk_len;
                for cycle in start..start + self.chunk_len {
                    let row = source.bytecode_index(cycle);
                    let cursor = &mut cursors[row >> self.range_shift];
                    let offset = *cursor as usize;
                    rows[offset] = row as u32;
                    weights[offset] = weight(cycle);
                    *cursor += 1;
                }
            });
        let range_len = 1 << self.range_shift;
        output
            .par_chunks_mut(range_len)
            .enumerate()
            .for_each(|(range, output)| {
                let range_start = range << self.range_shift;
                for (chunk, (counts, offsets)) in self.counts.iter().zip(&self.offsets).enumerate()
                {
                    let count = counts[range] as usize;
                    if count != 0 {
                        let start = chunk * self.chunk_len + offsets[range] as usize;
                        let end = start + count;
                        for (&row, &value) in rows[start..end].iter().zip(&weights[start..end]) {
                            output[row as usize - range_start] += value;
                        }
                    }
                }
            });
        Ok(())
    }
}
