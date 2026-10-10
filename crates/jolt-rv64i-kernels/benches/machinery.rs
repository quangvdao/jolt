//! Packed passes at log_t=22, five samples, on warmed one- and twelve-thread
//! pools. Lift prices one whole word. Bucket prices one nibble update in the
//! complete no-byte-selector fold layout. Scatter prices one cycle, excluding
//! plan construction and pair/output allocation. The requirement row uses
//! consecutive `all_rows` destinations; `scatter_permuted` uses a fixed seeded
//! permutation of all bytecode rows, repeated over cycles, with no requirement. Merge prices each of the
//! `(2W - 1) * layout_len` zero-fill and tree-merge element operations.

pub mod support;

use jolt_field::F128;
use jolt_rv64i_kernels::packed::buckets::{BucketError, NibbleBuckets};
use jolt_rv64i_kernels::packed::lift::{LiftError, WordLift};
use jolt_rv64i_kernels::packed::pool::{PoolError, ScratchPool};
use jolt_rv64i_kernels::packed::scatter::{ScatterError, ScatterPlan};
use jolt_rv64i_kernels::source::{CycleSource, SourceError, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthError, SynthProfile, SyntheticTrace};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;
use support::{run_machinery, MachineryKernel, RunnerError};
use thiserror::Error;

// Five shape ranges, each selector's word slots followed by nibble positions;
// the first shape also reserves eight 16-entry digit/flag positions per selector.
const OFFSETS: [usize; 5] = [
    0,
    64 * (5 * 256 + 128),
    64 * (5 * 256 + 128) + 512 * 256,
    64 * (5 * 256 + 128) + 512 * 256 + 128 * 2 * 256,
    64 * (5 * 256 + 128) + 512 * 256 + 128 * 2 * 256 + 512 * 3 * 256,
];
const LAYOUT: usize = OFFSETS[4] + 2 * 256;
const CHUNK: usize = 4096;

#[derive(Debug, Error)]
enum MachineryError {
    #[error(transparent)]
    Lift(#[from] LiftError),
    #[error(transparent)]
    Bucket(#[from] BucketError),
    #[error(transparent)]
    Pool(#[from] PoolError),
    #[error(transparent)]
    Scatter(#[from] ScatterError),
    #[error(transparent)]
    Source(#[from] SourceError),
    #[error(transparent)]
    Synth(#[from] SynthError),
    #[error("unknown machinery case {name}")]
    Case { name: String },
}

/// Shares packed values while optionally permuting bytecode destinations.
/// Row-based digits follow the selected row, preserving the source contract.
struct MachinerySource {
    base: Arc<SyntheticTrace>,
    permutation: Option<Vec<u32>>,
}
impl CycleSource for MachinerySource {
    fn cycles(&self) -> usize {
        self.base.cycles()
    }
    fn trace_words(&self) -> usize {
        self.base.trace_words()
    }
    fn trace_word(&self, word: usize, cycle: usize) -> u64 {
        self.base.trace_word(word, cycle)
    }
    fn bytecode_rows(&self) -> usize {
        self.base.bytecode_rows()
    }
    fn bytecode_words(&self) -> usize {
        self.base.bytecode_words()
    }
    fn bytecode_word(&self, word: usize, row: usize) -> u64 {
        self.base.bytecode_word(word, row)
    }
    fn bytecode_index(&self, cycle: usize) -> usize {
        if cycle >= self.cycles() {
            return 0;
        }
        self.permutation.as_ref().map_or_else(
            || self.base.bytecode_index(cycle),
            |rows| rows[cycle & (rows.len() - 1)] as usize,
        )
    }
    fn digit_columns(&self) -> usize {
        self.base.digit_columns()
    }
    fn bits(&self, column: usize) -> usize {
        self.base.bits(column)
    }
    fn by_row(&self, column: usize) -> bool {
        self.base.by_row(column)
    }
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        if cycle >= self.cycles() {
            return None;
        }
        if self.by_row(column) {
            self.row_digit(column, self.bytecode_index(cycle))
        } else {
            self.base.digit(column, cycle)
        }
    }
    fn row_digit(&self, column: usize, row: usize) -> Option<usize> {
        self.base.row_digit(column, row)
    }
}

struct Inputs {
    trace: Arc<ValidatedTrace<MachinerySource>>,
    permuted: Arc<ValidatedTrace<MachinerySource>>,
    words: Vec<u64>,
}

impl Inputs {
    fn new(log_t: usize) -> Result<Arc<Self>, MachineryError> {
        let base = Arc::new(SyntheticTrace::new(
            SynthProfile::AllRows,
            log_t,
            1 << 20,
            51,
        )?);
        let trace = Arc::new(ValidatedTrace::new(Arc::new(MachinerySource {
            base: Arc::clone(&base),
            permutation: None,
        }))?);
        let mut permutation: Vec<u32> = (0..1_u32 << 20).collect();
        let mut shuffle = ChaCha20Rng::seed_from_u64(52);
        for end in (1..permutation.len()).rev() {
            let index = shuffle.next_u32() as usize % (end + 1);
            permutation.swap(end, index);
        }
        let permuted = Arc::new(ValidatedTrace::new(Arc::new(MachinerySource {
            base,
            permutation: Some(permutation),
        }))?);
        let mut rng = ChaCha20Rng::seed_from_u64(53);
        let words = (0..trace.source().cycles())
            .map(|_| rng.next_u64())
            .collect();
        Ok(Arc::new(Self {
            trace,
            permuted,
            words,
        }))
    }
    fn weight(&self, cycle: usize) -> F128 {
        F128::from_raw(
            u128::from(self.trace.source().trace_word(0, cycle))
                | (u128::from(self.trace.source().trace_word(1, cycle)) << 64),
        )
    }
}

enum Machinery {
    Lift {
        inputs: Arc<Inputs>,
        lift: Box<WordLift>,
    },
    Bucket {
        inputs: Arc<Inputs>,
        pool: ScratchPool,
        operations: usize,
    },
    Scatter {
        inputs: Arc<Inputs>,
        plan: Box<ScatterPlan<MachinerySource>>,
        plan_ns: f64,
        weights: Vec<F128>,
        output: Vec<F128>,
    },
    Merge {
        pool: ScratchPool,
        threads: usize,
    },
}

impl Machinery {
    fn new(name: &str, inputs: Arc<Inputs>, threads: usize) -> Result<Self, MachineryError> {
        let populated = || -> Result<ScratchPool, PoolError> {
            let pool = ScratchPool::new(LAYOUT)?;
            let mut guards: Vec<_> = (0..threads)
                .map(|_| pool.take())
                .collect::<Result<_, _>>()?;
            for guard in &mut guards {
                guard.fill(F128::from_raw(1));
            }
            drop(guards);
            Ok(pool)
        };
        match name {
            "lift" => {
                let mut rng = ChaCha20Rng::seed_from_u64(54);
                let weights = std::array::from_fn(|_| {
                    F128::from_raw(u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64))
                });
                Ok(Self::Lift {
                    inputs,
                    lift: Box::new(WordLift::new(&weights)?),
                })
            }
            "bucket" => {
                let source = inputs.trace.source();
                let operations = (0..source.cycles())
                    .map(|cycle| {
                        88 + 16 * usize::from(source.digit(13, cycle).is_some())
                            + 32 * usize::from(source.digit(14, cycle).is_some())
                            + 48 * usize::from(source.digit(15, cycle).is_some())
                            + 32 * usize::from(
                                source.digit(16, cycle).is_some()
                                    && source.digit(19, cycle).is_some(),
                            )
                    })
                    .sum();
                Ok(Self::Bucket {
                    inputs,
                    pool: populated()?,
                    operations,
                })
            }
            "scatter" | "scatter_permuted" => {
                let trace = if name == "scatter" {
                    &inputs.trace
                } else {
                    &inputs.permuted
                };
                let start = Instant::now();
                let plan = Box::new(ScatterPlan::new(Arc::clone(trace))?);
                let plan_ns = start.elapsed().as_nanos() as f64;
                Ok(Self::Scatter {
                    weights: vec![F128::from_raw(0); plan.cycles()],
                    output: vec![F128::from_raw(0); plan.bytecode_rows()],
                    inputs,
                    plan,
                    plan_ns,
                })
            }
            "merge" => Ok(Self::Merge {
                pool: populated()?,
                threads,
            }),
            _ => Err(MachineryError::Case {
                name: name.to_owned(),
            }),
        }
    }

    #[inline]
    fn word(
        buckets: &mut NibbleBuckets<'_>,
        base: usize,
        slot: usize,
        word: u64,
        e: F128,
    ) -> Result<(), BucketError> {
        for (byte, value) in word.to_le_bytes().into_iter().enumerate() {
            buckets.xor(base + slot * 16 + byte * 2, usize::from(value & 15), e)?;
            buckets.xor(base + slot * 16 + byte * 2 + 1, usize::from(value >> 4), e)?;
        }
        Ok(())
    }
}

impl MachineryKernel for Machinery {
    type Error = MachineryError;
    fn operations(&self) -> usize {
        match self {
            Self::Lift { inputs, .. } => inputs.words.len(),
            Self::Bucket { operations, .. } => *operations,
            Self::Scatter { plan, .. } => plan.cycles(),
            Self::Merge { threads, .. } => (2 * threads - 1) * LAYOUT,
        }
    }
    fn plan_construction_ns(&self) -> Option<f64> {
        match self {
            Self::Scatter { plan_ns, .. } => Some(*plan_ns),
            _ => None,
        }
    }
    fn run(&mut self) -> Result<F128, MachineryError> {
        match self {
            Self::Lift { inputs, lift } => Ok(black_box(&inputs.words)
                .par_chunks(CHUNK)
                .map(|words| {
                    words
                        .iter()
                        .fold(F128::from_raw(0), |sum, &word| sum + lift.lift(word))
                })
                .reduce(|| F128::from_raw(0), |a, b| a + b)),
            Self::Scatter {
                inputs,
                plan,
                weights,
                output,
                ..
            } => {
                let plan = black_box(plan.as_ref());
                let inputs = black_box(inputs.as_ref());
                plan.scatter_into(|cycle| inputs.weight(cycle), weights, output)?;
                let _ = black_box(&*output);
                Ok(output[0])
            }
            Self::Merge { pool, .. } => {
                let pool = black_box(&*pool);
                pool.zero()?;
                let output = pool.merge()?;
                let _ = black_box(&output);
                Ok(output[0])
            }
            Self::Bucket { inputs, pool, .. } => {
                let source = black_box(inputs.trace.source());
                let pool = black_box(&*pool);
                (0..source.cycles().div_ceil(CHUNK))
                    .into_par_iter()
                    .try_for_each(|chunk| -> Result<(), MachineryError> {
                        let mut guard = pool.take()?;
                        let mut buckets = NibbleBuckets::new(&mut guard)?;
                        for cycle in chunk * CHUNK..((chunk + 1) * CHUNK).min(source.cycles()) {
                            let e = inputs.weight(cycle);
                            let selector = source.digit(12, cycle).unwrap_or(0);
                            for (slot, word) in [0, 1, 2, 4, 5].into_iter().enumerate() {
                                Self::word(
                                    &mut buckets,
                                    selector * (5 * 16 + 8),
                                    slot,
                                    source.trace_word(word, cycle),
                                    e,
                                )?;
                            }
                            for (slot, column) in (5..12).enumerate() {
                                buckets.xor(
                                    selector * 88 + 80 + slot,
                                    source.digit(column, cycle).unwrap_or(0),
                                    e,
                                )?;
                            }
                            let flags = usize::from(source.digit(18, cycle).is_some())
                                | (usize::from(source.digit(19, cycle).is_some()) << 1)
                                | (usize::from(source.digit(20, cycle).is_some()) << 2);
                            buckets.xor(selector * 88 + 87, flags, e)?;
                            let low = source.digit(10, cycle).unwrap_or(0);
                            let high = source.digit(11, cycle).unwrap_or(0);
                            if let Some(kind) = source.digit(13, cycle) {
                                Self::word(
                                    &mut buckets,
                                    OFFSETS[1] / 16 + (low + 8 * high + 64 * kind) * 16,
                                    0,
                                    source.trace_word(0, cycle),
                                    e,
                                )?;
                            }
                            if let Some(kind) = source.digit(14, cycle) {
                                for (slot, word) in [3, 1].into_iter().enumerate() {
                                    Self::word(
                                        &mut buckets,
                                        OFFSETS[2] / 16 + (low + 8 * kind) * 2 * 16,
                                        slot,
                                        source.trace_word(word, cycle),
                                        e,
                                    )?;
                                }
                            }
                            let row = source.bytecode_index(cycle);
                            if let Some(kind) = source.digit(15, cycle) {
                                for slot in 0..3 {
                                    let word = if slot < 2 {
                                        source.trace_word(slot, cycle)
                                    } else {
                                        source.bytecode_word(0, row)
                                    };
                                    Self::word(
                                        &mut buckets,
                                        OFFSETS[3] / 16 + (low + 8 * high + 64 * kind) * 3 * 16,
                                        slot,
                                        word,
                                        e,
                                    )?;
                                }
                            }
                            if source.digit(16, cycle).is_some()
                                && source.digit(19, cycle).is_some()
                            {
                                for slot in 0..2 {
                                    Self::word(
                                        &mut buckets,
                                        OFFSETS[4] / 16,
                                        slot,
                                        source.bytecode_word(slot + 1, row),
                                        e,
                                    )?;
                                }
                            }
                        }
                        Ok(())
                    })?;
                let _ = black_box(pool);
                Ok(F128::from_raw(0))
            }
        }
    }
}

fn main() -> Result<(), RunnerError> {
    run_machinery(
        &[
            ("lift", Some(4.8)),
            ("bucket", Some(0.9)),
            ("scatter", Some(2.1)),
            ("scatter_permuted", None),
            ("merge", Some(0.45)),
        ],
        Inputs::new,
        Machinery::new,
    )
}
