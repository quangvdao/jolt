//! Cached cycle routing through 256 bytecode-row ranges.
//!
//! Slots are chunk-major; range-relative row offsets follow the grouped slots.
//! Segment descriptors are range-major, so application scans its metadata
//! sequentially. A scatter emits only weights and reads no source indices.

use std::mem::size_of;
use std::ops::Range;
use std::sync::Arc;

use jolt_field::F128;
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut};
use thiserror::Error;

use crate::par::CycleChunks;
use crate::source::{CycleSource, ValidatedTrace};

const RANGES: usize = 256;

type EmissionChunk<'a> = (Range<usize>, &'a [u16], &'a mut [F128]);

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
    /// Count and place destinations in parallel over [`CycleChunks`] at round
    /// zero, independently of the thread count. Each chunk owns disjoint slot
    /// and row-offset slices. A reusable chunk-local index buffer avoids reading
    /// the source twice; prefix arrays retain the range ends for segment assembly.
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
        let mut prefixes = vec![[0_u32; RANGES]; chunks];
        slots
            .par_chunks_mut(chunk_len)
            .zip(row_offsets.par_chunks_mut(chunk_len))
            .zip(prefixes.par_iter_mut())
            .enumerate()
            .for_each_init(
                || Vec::with_capacity(chunk_len),
                |indices, (chunk, ((slots, row_offsets), counts))| {
                    let start = chunk * chunk_len;
                    indices.clear();
                    for cycle in start..start + chunk_len {
                        let row = source.bytecode_index(cycle);
                        counts[row >> range_shift] += 1;
                        indices.push(row as u32);
                    }
                    let mut cursors = [0_u32; RANGES];
                    let mut next = 0_u32;
                    for (cursor, count) in cursors.iter_mut().zip(counts.iter_mut()) {
                        *cursor = next;
                        next += *count;
                        *count = next;
                    }
                    for (slot, &row) in slots.iter_mut().zip(indices.iter()) {
                        let row = row as usize;
                        let cursor = &mut cursors[row >> range_shift];
                        *slot = *cursor as u16;
                        row_offsets[*cursor as usize] = (row & (range_len - 1)) as u16;
                        *cursor += 1;
                    }
                },
            );
        segments
            .par_chunks_mut(chunks)
            .enumerate()
            .for_each(|(range, segments)| {
                for (segment, counts) in segments.iter_mut().zip(&prefixes) {
                    let start = if range == 0 { 0 } else { counts[range - 1] };
                    *segment = Segment {
                        start,
                        len: counts[range] - start,
                    };
                }
            });
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

    /// Checked, disjoint emission chunks for a pass fused with other cycle work.
    ///
    /// Checks the exact weight-buffer length before yielding any mutable slice.
    /// Each item binds this plan's global cycle interval to its cycle-ordered
    /// local slots and its exact mutable segment of the caller's buffer. The
    /// intervals are the chunks of `CycleChunks::new(log_t, 0)`. Local slots
    /// form a permutation of `0..segment.len()`: cycle `interval.start + i`
    /// writes its weight to `segment[usize::from(slots[i])]`.
    ///
    /// Initializing every slot with that cycle's weight is required of the
    /// caller, not checked here, and incorrect folded claims are detected by
    /// the verifier. The mutable borrow prevents overlapping application while
    /// the emission iterator or its segments remain in use.
    pub fn emission_chunks<'a>(
        &'a self,
        weights: &'a mut [F128],
    ) -> Result<impl IndexedParallelIterator<Item = EmissionChunk<'a>> + 'a, ScatterError> {
        self.check_weight_length(weights.len())?;
        Ok(self.chunks(weights))
    }

    /// Apply a previously emitted chunk-major buffer, XORing into `output`.
    ///
    /// Checks both exact lengths before changing the output. The caller must
    /// have initialized every slot using [`Self::emission_chunks`];
    /// this weight-to-cycle association is required of the caller, not checked
    /// here, and incorrect folded claims are detected by the verifier.
    pub fn apply_buffer(&self, weights: &[F128], output: &mut [F128]) -> Result<(), ScatterError> {
        self.check_buffers(weights.len(), output.len())?;
        self.apply(weights, output);
        Ok(())
    }

    /// Return `out[k] = Σ_{j: bytecode_index(j) = k} weight(j)`.
    /// Allocates one buffer of cycle weights and one zeroed output. No allocation
    /// occurs per cycle, pair or chunk. The callback runs once per cycle and must
    /// be a deterministic function of its argument for deterministic output.
    pub fn scatter(
        &self,
        weight: impl Fn(usize) -> F128 + Sync,
    ) -> Result<Vec<F128>, ScatterError> {
        let mut weights = unsafe_allocate_zero_vec(self.cycles);
        let mut output = unsafe_allocate_zero_vec(self.rows);
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
        self.check_buffers(weights.len(), output.len())?;
        self.emit(&weight, weights);
        self.apply(weights, output);
        Ok(())
    }

    fn check_buffers(&self, weights: usize, output: usize) -> Result<(), ScatterError> {
        self.check_weight_length(weights)?;
        if output != self.rows {
            return Err(ScatterError::BufferLength {
                buffer: "output",
                expected: self.rows,
                actual: output,
            });
        }
        Ok(())
    }

    fn check_weight_length(&self, actual: usize) -> Result<(), ScatterError> {
        if actual != self.cycles {
            return Err(ScatterError::BufferLength {
                buffer: "weights",
                expected: self.cycles,
                actual,
            });
        }
        Ok(())
    }

    fn chunks<'a>(
        &'a self,
        weights: &'a mut [F128],
    ) -> impl IndexedParallelIterator<Item = EmissionChunk<'a>> + 'a {
        self.slots
            .par_chunks(self.chunk_len)
            .zip(weights.par_chunks_mut(self.chunk_len))
            .enumerate()
            .map(move |(chunk, (slots, weights))| {
                let start = chunk * self.chunk_len;
                (start..start + self.chunk_len, slots, weights)
            })
    }

    fn emit(&self, weight: &(impl Fn(usize) -> F128 + Sync), weights: &mut [F128]) {
        self.chunks(weights).for_each(|(cycles, slots, weights)| {
            let Some(last) = weights.len().checked_sub(1) else {
                return;
            };
            for (cycle, &slot) in slots.iter().enumerate() {
                // new makes the clamp an identity; its bound removes per-cycle panic paths.
                weights[usize::from(slot).min(last)] = weight(cycles.start + cycle);
            }
        });
    }

    fn apply(&self, weights: &[F128], output: &mut [F128]) {
        let chunks = self.cycles / self.chunk_len;
        output
            .par_chunks_mut(self.range_len)
            .zip(self.segments.par_chunks(chunks))
            .for_each(|(output, segments)| {
                let Some(last) = output.len().checked_sub(1) else {
                    return;
                };
                for (chunk, segment) in segments.iter().enumerate() {
                    let start = chunk * self.chunk_len + segment.start as usize;
                    let end = start + segment.len as usize;
                    for (&row, &weight) in self.row_offsets[start..end]
                        .iter()
                        .zip(&weights[start..end])
                    {
                        // new bounds row within this range; the clamp is an identity.
                        output[usize::from(row).min(last)] += weight;
                    }
                }
            });
    }
}
