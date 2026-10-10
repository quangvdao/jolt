//! Cached cycle routing through 256 bytecode-row ranges.
//!
//! Slots are chunk-major; range-relative row offsets follow the grouped slots.
//! Segment descriptors are range-major, so application scans its metadata
//! sequentially. A scatter emits only weights and reads no source indices.

use std::mem::size_of;
use std::sync::Arc;

use jolt_field::F128;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut};
use thiserror::Error;

use crate::par::CycleChunks;
use crate::source::{CycleSource, ValidatedTrace};

const RANGES: usize = 256;

/// Unrepresentable scatter dimensions or incorrectly sized caller buffers.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ScatterError {
    #[error("bytecode row count {rows} exceeds {limit}")]
    RowCount { rows: usize, limit: usize },
    #[error("cycle chunk length {len} exceeds {limit}")]
    ChunkLength { len: usize, limit: usize },
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

#[derive(Clone, Copy, Default)]
struct Segment {
    start: u32,
    len: u32,
}

/// Cached partition for summing cycle weights at their bytecode row indices.
///
/// The immutable source must already be checked by [`ValidatedTrace`]. Each
/// cycle has a two-byte chunk-local slot; each slot has a two-byte range-relative
/// row offset. Each chunk/range descriptor has a four-byte start and length.
/// Chunks hold at most 65,536 cycles, so their counts and exclusive prefix sums
/// fit `u32`; slot indices fit `u16`. At most 2^24 rows give at most 65,536 rows
/// per range. Construction checks these bounds and representable capacities.
pub struct ScatterPlan<S: CycleSource> {
    _trace: Arc<ValidatedTrace<S>>,
    slots: Vec<u16>,
    row_offsets: Vec<u16>,
    segments: Vec<Segment>,
    cycles: usize,
    rows: usize,
    chunk_len: usize,
    range_len: usize,
}

impl<S: CycleSource> ScatterPlan<S> {
    /// Count and place destinations in [`CycleChunks`] at round zero.
    /// Rejects more than 2^24 rows or chunks longer than 2^16 cycles before
    /// allocation. One row and fewer than 256 rows use one-row ranges.
    #[expect(
        clippy::expect_used,
        reason = "validated power-of-two cycle count has a representable exponent and round zero"
    )]
    pub fn new(trace: Arc<ValidatedTrace<S>>) -> Result<Self, ScatterError> {
        let source = trace.source();
        let cycles = source.cycles();
        let rows = source.bytecode_rows();
        if rows > 1 << 24 {
            return Err(ScatterError::RowCount {
                rows,
                limit: 1 << 24,
            });
        }
        let chunk_len = CycleChunks::new(cycles.ilog2() as usize, 0)
            .expect("validated cycle geometry")
            .chunk_len();
        if chunk_len > 1 << 16 {
            // The least validated input reaching this error has 2^33 cycles.
            return Err(ScatterError::ChunkLength {
                len: chunk_len,
                limit: 1 << 16,
            });
        }
        let chunks = cycles / chunk_len;
        check_storage::<Segment>(chunks * RANGES, "segments")?;
        check_storage::<u16>(cycles, "slots")?;
        check_storage::<u16>(cycles, "row offsets")?;
        check_storage::<F128>(cycles, "weights")?;
        check_storage::<F128>(rows, "output")?;
        let range_shift = rows.ilog2().saturating_sub(8);
        let range_len = 1 << range_shift;
        let mut slots = vec![0_u16; cycles];
        let mut row_offsets = vec![0_u16; cycles];
        let mut segments = vec![Segment::default(); chunks * RANGES];
        for chunk in 0..chunks {
            let start = chunk * chunk_len;
            let mut counts = [0_u32; RANGES];
            for cycle in start..start + chunk_len {
                counts[source.bytecode_index(cycle) >> range_shift] += 1;
            }
            let mut cursors = [0_u32; RANGES];
            let mut next = 0_u32;
            for (range, &len) in counts.iter().enumerate() {
                segments[range * chunks + chunk] = Segment { start: next, len };
                cursors[range] = next;
                next += len;
            }
            for (cycle, slot) in slots[start..start + chunk_len].iter_mut().enumerate() {
                let row = source.bytecode_index(start + cycle);
                let cursor = &mut cursors[row >> range_shift];
                *slot = *cursor as u16;
                row_offsets[start + *cursor as usize] = (row & (range_len - 1)) as u16;
                *cursor += 1;
            }
        }
        Ok(Self {
            _trace: trace,
            slots,
            row_offsets,
            segments,
            cycles,
            rows,
            chunk_len,
            range_len,
        })
    }

    /// Number of cycle weights consumed by a scatter.
    pub fn cycles(&self) -> usize {
        self.cycles
    }
    /// Number of output elements, including unvisited bytecode rows.
    pub fn bytecode_rows(&self) -> usize {
        self.rows
    }

    /// Return `out[k] = Σ_{j: bytecode_index(j) = k} weight(j)`.
    /// Allocates one buffer of cycle weights and one zeroed output. No allocation
    /// occurs per cycle, pair or chunk. The callback runs once per cycle and must
    /// be a deterministic function of its argument for deterministic output.
    pub fn scatter(
        &self,
        weight: impl Fn(usize) -> F128 + Sync,
    ) -> Result<Vec<F128>, ScatterError> {
        let mut weights = vec![F128::from_raw(0); self.cycles];
        let mut output = vec![F128::from_raw(0); self.rows];
        self.scatter_into(weight, &mut weights, &mut output)?;
        Ok(output)
    }

    /// XOR the scatter into existing output through a caller's weight buffer.
    /// Checks both exact lengths before writing either buffer. Weights are
    /// overwritten; the output contribution is retained. No allocation or
    /// zero-fill occurs. The callback has the contract of [`Self::scatter`].
    pub fn scatter_into(
        &self,
        weight: impl Fn(usize) -> F128 + Sync,
        weights: &mut [F128],
        output: &mut [F128],
    ) -> Result<(), ScatterError> {
        for (buffer, expected, actual) in [
            ("weights", self.cycles, weights.len()),
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
        self.emit(&weight, weights);
        self.apply(weights, output);
        Ok(())
    }

    fn emit(&self, weight: &(impl Fn(usize) -> F128 + Sync), weights: &mut [F128]) {
        match self.chunk_len {
            4096 => self.emit_blocks::<4096>(weight, weights),
            8192 => self.emit_blocks::<8192>(weight, weights),
            16384 => self.emit_blocks::<16384>(weight, weights),
            32768 => self.emit_blocks::<32768>(weight, weights),
            65536 => self.emit_blocks::<65536>(weight, weights),
            _ => weights
                .par_chunks_mut(self.chunk_len)
                .zip(self.slots.par_chunks(self.chunk_len))
                .enumerate()
                .for_each(|(chunk, (weights, slots))| {
                    let start = chunk * self.chunk_len;
                    for (cycle, &slot) in slots.iter().enumerate() {
                        weights[usize::from(slot)] = weight(start + cycle);
                    }
                }),
        }
    }

    fn emit_blocks<const N: usize>(
        &self,
        weight: &(impl Fn(usize) -> F128 + Sync),
        weights: &mut [F128],
    ) {
        weights
            .as_chunks_mut::<N>()
            .0
            .par_iter_mut()
            .zip(self.slots.par_chunks(N))
            .enumerate()
            .for_each(|(chunk, (weights, slots))| {
                let start = chunk * N;
                for (cycle, &slot) in slots.iter().enumerate() {
                    // new establishes slot < N; the mask exposes that bound to code generation.
                    weights[usize::from(slot) & (N - 1)] = weight(start + cycle);
                }
            });
    }

    fn apply(&self, weights: &[F128], output: &mut [F128]) {
        match self.range_len {
            4096 => self.apply_blocks::<4096>(weights, output),
            65536 => self.apply_blocks::<65536>(weights, output),
            _ => {
                let chunks = self.cycles / self.chunk_len;
                output
                    .par_chunks_mut(self.range_len)
                    .zip(self.segments.par_chunks(chunks))
                    .for_each(|(output, segments)| {
                        for (chunk, segment) in segments.iter().enumerate() {
                            let start = chunk * self.chunk_len + segment.start as usize;
                            let end = start + segment.len as usize;
                            for (&row, &weight) in self.row_offsets[start..end]
                                .iter()
                                .zip(&weights[start..end])
                            {
                                output[usize::from(row)] += weight;
                            }
                        }
                    });
            }
        }
    }

    fn apply_blocks<const N: usize>(&self, weights: &[F128], output: &mut [F128]) {
        let chunks = self.cycles / self.chunk_len;
        output
            .as_chunks_mut::<N>()
            .0
            .par_iter_mut()
            .zip(self.segments.par_chunks(chunks))
            .for_each(|(output, segments)| {
                for (chunk, segment) in segments.iter().enumerate() {
                    let start = chunk * self.chunk_len + segment.start as usize;
                    let end = start + segment.len as usize;
                    for (&row, &weight) in self.row_offsets[start..end]
                        .iter()
                        .zip(&weights[start..end])
                    {
                        // new establishes row < N for this range; no routing is recomputed here.
                        output[usize::from(row) & (N - 1)] += weight;
                    }
                }
            });
    }
}
