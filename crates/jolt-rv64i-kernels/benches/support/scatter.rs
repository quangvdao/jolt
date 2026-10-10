//! Cycle-order pair emission into one range-partitioned scatter buffer.
//!
//! The scatter plan in `specs/rv64i-binary-prover-kernels.md` counts each
//! 4096-cycle chunk's destinations in 256 row ranges before timing. Row and
//! weight buffers are range-contiguous; recursive safe splitting hands each
//! chunk its own portion of every range. Timing includes the cycle-order row
//! and weight writes, followed by range-local updates that read only the pairs.

use std::hint::black_box;
use std::sync::Arc;

use jolt_field::F128;
use jolt_rv64i_kernels::source::CycleSource;
use jolt_rv64i_kernels::synth::SyntheticTrace;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
use thiserror::Error;

const CHUNK: usize = 4096;
const ROWS: usize = 1 << 20;
const RANGES: usize = 256;
const RANGE_ROWS: usize = ROWS / RANGES;

#[derive(Debug, Error)]
pub enum ScatterError {
    #[error("partitioned scatter needs {expected} bytecode rows, got {actual}")]
    Rows { expected: usize, actual: usize },
}

#[derive(Default)]
struct Segment<'a> {
    rows: &'a mut [u32],
    weights: &'a mut [F128],
}

struct ScatterPlan {
    prefix: Vec<[usize; RANGES]>,
    offsets: [usize; RANGES + 1],
    cycles: usize,
}

impl ScatterPlan {
    fn new(source: &SyntheticTrace) -> Self {
        let cycles = source.cycles();
        let chunks = cycles.div_ceil(CHUNK);
        let mut prefix = vec![[0; RANGES]; chunks + 1];
        for chunk in 0..chunks {
            prefix[chunk + 1] = prefix[chunk];
            for cycle in chunk * CHUNK..((chunk + 1) * CHUNK).min(cycles) {
                prefix[chunk + 1][source.bytecode_index(cycle) / RANGE_ROWS] += 1;
            }
        }
        let mut offsets = [0; RANGES + 1];
        for range in 0..RANGES {
            offsets[range + 1] = offsets[range] + prefix[chunks][range];
        }
        Self {
            prefix,
            offsets,
            cycles,
        }
    }

    fn emit(
        &self,
        source: &SyntheticTrace,
        mut segments: [Segment<'_>; RANGES],
        start: usize,
        end: usize,
    ) {
        if end - start == 1 {
            let mut cursors = [0; RANGES];
            for cycle in start * CHUNK..(end * CHUNK).min(self.cycles) {
                let row = black_box(source.bytecode_index(cycle));
                let range = row / RANGE_ROWS;
                let weight = F128::from_raw(
                    u128::from(black_box(source.trace_word(0, cycle)))
                        | (u128::from(black_box(source.trace_word(1, cycle))) << 64),
                );
                let cursor = cursors[range];
                segments[range].rows[cursor] = black_box(row as u32);
                segments[range].weights[cursor] = black_box(weight);
                cursors[range] += 1;
            }
            return;
        }
        let middle = start + (end - start) / 2;
        let left = std::array::from_fn(|range| {
            let segment = std::mem::take(&mut segments[range]);
            let split = self.prefix[middle][range] - self.prefix[start][range];
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
            || self.emit(source, left, start, middle),
            || self.emit(source, segments, middle, end),
        );
    }
}

pub struct PartitionedScatter {
    source: Arc<SyntheticTrace>,
    plan: ScatterPlan,
    rows: Vec<u32>,
    weights: Vec<F128>,
    pub(crate) output: Vec<F128>,
}

impl PartitionedScatter {
    pub fn new(source: Arc<SyntheticTrace>) -> Result<Self, ScatterError> {
        if source.bytecode_rows() != ROWS {
            return Err(ScatterError::Rows {
                expected: ROWS,
                actual: source.bytecode_rows(),
            });
        }
        let plan = ScatterPlan::new(source.as_ref());
        Ok(Self {
            rows: vec![0; plan.cycles],
            weights: vec![F128::from_raw(0); plan.cycles],
            output: vec![F128::from_raw(0); ROWS],
            source,
            plan,
        })
    }

    pub fn cycles(&self) -> usize {
        self.plan.cycles
    }

    pub fn run(&mut self) -> F128 {
        let source = black_box(self.source.as_ref());
        let mut rows = self.rows.as_mut_slice();
        let mut weights = self.weights.as_mut_slice();
        let segments = std::array::from_fn(|range| {
            let length = self.plan.offsets[range + 1] - self.plan.offsets[range];
            let (range_rows, remaining_rows) = std::mem::take(&mut rows).split_at_mut(length);
            let (range_weights, remaining_weights) =
                std::mem::take(&mut weights).split_at_mut(length);
            rows = remaining_rows;
            weights = remaining_weights;
            Segment {
                rows: range_rows,
                weights: range_weights,
            }
        });
        self.plan
            .emit(source, segments, 0, self.plan.prefix.len() - 1);
        let rows = black_box(self.rows.as_slice());
        let weights = black_box(self.weights.as_slice());
        let offsets = &self.plan.offsets;
        let result = self
            .output
            .par_chunks_mut(RANGE_ROWS)
            .enumerate()
            .map(|(range, output)| {
                let range_start = range * RANGE_ROWS;
                for pair in offsets[range]..offsets[range + 1] {
                    output[rows[pair] as usize - range_start] += weights[pair];
                }
                let _ = black_box(&*output);
                output[0]
            })
            .reduce(|| F128::from_raw(0), |left, right| left + right);
        black_box(result)
    }
}
