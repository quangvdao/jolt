//! Unit costs on seeded packed traces, before the packed passes are available.
//!
//! `lookup` issues 48 byte reads for the outer lanes, 48 for six word lifts,
//! 37 for the digit builder, and 49 for the byte builder. Ordinary layouts have
//! disjoint 256-entry banks and, at 5/69 KiB, a final 64-entry bank. That smaller
//! bank masks byte indices to six bits. Two precomputed bank schedules alternate
//! by chunk, covering a footprint with more banks than read streams. Smaller
//! footprints reuse banks. The byte builder uses interleaved bank pairs and
//! singleton banks; at 196 KiB these are exactly twenty pairs and nine singles.
//! The 5 KiB byte layout uses a 128-position pair and a 64-position singleton,
//! masking indices to seven/six bits. No timed lookup uses division or modulo.
//! Requested KiB describes allocated banks, not the effective cache working set:
//! only indices observed in the trace are read. In particular, high RAM chunks,
//! flags, PC bytes and one-hot row bytes may access much smaller subsets. The
//! active address set is the union of `base + ((value & mask) << shift)` over
//! observed values and the two schedules. No synthetic bits expand digit values.
//! Byte and digit distributions otherwise retain their trace locality. The
//! byte builder interleaves the two weights on bytes
//! 0–7 and 17–28, so a shared byte index selects adjacent weight entries. The
//! digit builder counts two eight-byte Inc lifts, twelve chunk lookups plus
//! KeysDiffer, and seven chunk lookups plus KeysDiffer; the other two flags
//! require no lookup. This measures reads and XORs, with index decoding included.
//!
//! `bucket` counts each F128 XOR update. Column buckets use all 32 row bytes.
//! Fold buckets use the five router banks and their activity digits, including
//! the two bytecode branch words and Imm in Compare. Variant has five trace
//! words and eight metadata buckets (seven chunks and the three flags together).
//! Its layouts occupy 10.383, 11.477 and 19.133 MiB per worker. A seeded trace
//! word selects the requested share on the hot eight selectors. The hot-eight
//! layout varies its byte-update fraction; the all-nibble and all-byte layouts
//! keep fractions zero and one while the share changes locality. The separate
//! pass over four bytecode words per visited row is excluded from this unit.
//!
//! `scatter` reports time per input cycle, including a worker-table tree merge.
//! Local and all-rows traces supply the actual 2^16 and 2^20 destinations. Direct
//! scatter uses two relaxed atomic u64 XORs per F128. Partitioned scatter gathers
//! weights in a precomputed destination order, then gives each row range to one
//! worker: the cycle-id and weight arrays occupy 20 bytes per cycle. This safe
//! read reordering substitutes for writing pairs through a ScatterPlan.
//!
//! `fmadd` prepares the same twenty masked operand pairs at every chain length,
//! accumulates zero, one, two, four, eight or twenty terms on the chunk stack,
//! reduces nonempty chains there, and XORs each result into the chunk total.
//! The runner fits per-cycle medians at nonzero lengths by least squares: A is
//! the slope, R is the intercept minus the zero-length preparation/XOR baseline.
//! The maximum absolute residual reports departures from this affine model.
//! `merge` reports time per zero-filled or merged F128 element; its two variants
//! count W*N and (2W-1)*N operations on N=10 MiB/16, respectively. All chunk sizes
//! are fixed at 4096 cycles and no unit allocates during an individual chunk.
//!
//! | Record metric | Unit-table symbol | Included work |
//! |---|---|---|
//! | lookup ns | L per lookup | index decoding and XOR |
//! | bucket ns | Bk per F128 update | byte/nibble decoding |
//! | scatter ns | sct per cycle | updates and worker merge; setup excluded |
//! | fmadd fit slope_ns | A per term | fused stack chains, fixed preparation removed by fit |
//! | fmadd fit reduction_ns | R per chain | fit intercept minus zero-length baseline |
//! | merge ns | mrg per element | zero-fill, or zero-fill plus tree merge |
//!
//! `arithmetic/products` measures independent reduced products from trace words.
//! `mul_x_raw_shift_substitute` uses the raw shift and modulus-0x87 conditional
//! XOR because the field's mul_x helper is not yet available. `word_monomial_mix`
//! mirrors the outer monomial rounds: two three-stage Moebius transforms, AND,
//! then stride-eight gather, totalling nine shifts, eleven ANDs, six XORs and
//! three ORs (29 word operations). The spec does not fix the ratio within 410 w.
//! `readout` reads the same column/fold bank geometry as the update probes:
//! 128 reads per byte output bit, eight per nibble bit, individual indicator
//! cells, flag-bit sums, and selector One totals. It excludes row-only banks.
//!
//! Its table-layout, stream-storage and atomic
//! substitutions are part of each unit cost, rather than isolated instructions.
//!
//! | Lookup pattern | Passage mirrored in the kernel specification | Accesses per cycle |
//! |---|---|---|
//! | outer | Bitwise outer sum-check, Rounds 7 and 8: materialise six lane words with WordLift | 48 byte lookups |
//! | source_lift | Routers, source_lift: lift each of the six trace words once at r_bit | 48 byte lookups |
//! | g_digits | Reduction of committed claims, g_pass_digits: word bytes and indicator digits | 16 byte and 21 digit lookups |
//! | g_bytes | Reduction of committed claims, g_pass_bytes: row-byte tables with interleaved weights | 49 lookups over 29 byte positions |
//!
//! Every entry is one 16-byte F128. Ordinary full banks have eight index bits;
//! their 64-entry tails have six. Interleaved full pairs have eight index bits
//! and two adjacent entries per index; the 5 KiB pair has seven index bits.
//! Digit lookups retain their original zero-to-four-bit value ranges within
//! those banks. The following table gives bank counts for every footprint;
//! each ordinary pattern uses its column and g_bytes uses the paired column.
//!
//! | KiB | Ordinary banks (F128 entries) | Interleaved byte layout |
//! |---|---|---|
//! | 5 | 256 + 64 | one 128-position pair + 64 singleton |
//! | 32 | 8 × 256 | three 256-position pairs + two 256 singletons |
//! | 64 | 16 × 256 | six 256-position pairs + four 256 singletons |
//! | 69 | 17 × 256 + 64 | six pairs + five 256 singletons + 64 singleton |
//! | 96 | 24 × 256 | nine pairs + six 256 singletons |
//! | 196 | 49 × 256 | twenty pairs + nine 256 singletons |
//!
//! Scatter destination coverage reaches the stated powers of two at log_t=22;
//! smaller CLI streams visit at most their cycle count. Unit groups accepted by
//! --units are lookup,bucket,scatter,fmadd,merge, or all.

pub mod support;

use std::hint::black_box;
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use jolt_field::{Accumulator, Field, WithAccumulator, F128};
use jolt_rv64i_kernels::source::{CycleSource, LaneSource};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use thiserror::Error;

use support::{run_probe, ProbeCase, ProbeKernel, RunnerError};

type F128Accumulator = <F128 as WithAccumulator>::Accumulator;

const CHUNK: usize = 4096;
const ROWS: usize = 1 << 20;
const ROW_RANGES: usize = 256;
const BOTH: &[SynthProfile] = &[SynthProfile::Local, SynthProfile::AllRows];
const LOCAL: &[SynthProfile] = &[SynthProfile::Local];
const ALL_ROWS: &[SynthProfile] = &[SynthProfile::AllRows];

#[derive(Debug, Error)]
enum ProbeError {
    #[error("unknown probe case {unit}/{variant}")]
    Case { unit: String, variant: String },
    #[error("probe worker count {threads} must be positive")]
    Threads { threads: usize },
    #[error("scatter needs {expected} bytecode rows, got {actual}")]
    Rows { expected: usize, actual: usize },
}

#[derive(Clone, Copy)]
enum LookupPattern {
    Outer,
    Lift,
    Digits,
    Bytes,
}

impl LookupPattern {
    fn reads(self) -> usize {
        match self {
            Self::Outer | Self::Lift => 48,
            Self::Digits => 37,
            Self::Bytes => 49,
        }
    }
}

struct Lookup {
    source: Arc<SyntheticTrace>,
    table: Vec<F128>,
    pattern: LookupPattern,
    accesses: [[Access; 49]; 2],
}

#[derive(Clone, Copy, Default)]
struct Access {
    base: usize,
    mask: usize,
    shift: usize,
}

impl Lookup {
    fn new(source: Arc<SyntheticTrace>, pattern: LookupPattern, kib: usize) -> Self {
        let entries = kib * 1024 / size_of::<F128>();
        let accesses = Self::accesses(pattern, entries);
        Self {
            source,
            table: field_values(entries),
            pattern,
            accesses,
        }
    }

    fn accesses(pattern: LookupPattern, entries: usize) -> [[Access; 49]; 2] {
        let mut layouts = [[Access::default(); 49]; 2];
        if matches!(pattern, LookupPattern::Bytes) {
            let mut pairs = Vec::new();
            let mut singles = Vec::new();
            let mut base = 0;
            if entries == 320 {
                pairs.push(Access {
                    base,
                    mask: 127,
                    shift: 1,
                });
                base += 256;
            } else {
                let pair_count = (entries / 256) * 20 / 49;
                for _ in 0..pair_count {
                    pairs.push(Access {
                        base,
                        mask: 255,
                        shift: 1,
                    });
                    base += 512;
                }
            }
            while base + 256 <= entries {
                singles.push(Access {
                    base,
                    mask: 255,
                    shift: 0,
                });
                base += 256;
            }
            if base < entries {
                singles.push(Access {
                    base,
                    mask: 63,
                    shift: 0,
                });
            }
            let mut stream = 0;
            let mut pair = 0;
            let mut single = 0;
            for uses in BYTE_USES {
                if uses == 2 {
                    let first = pairs[pair % pairs.len()];
                    layouts[0][stream] = first;
                    layouts[0][stream + 1] = Access {
                        base: first.base + 1,
                        ..first
                    };
                    pair += 1;
                } else {
                    layouts[0][stream] = singles[single % singles.len()];
                    single += 1;
                }
                stream += uses;
            }
            layouts[1] = layouts[0];
        } else {
            let mut banks = Vec::new();
            let mut base = 0;
            while base + 256 <= entries {
                banks.push(Access {
                    base,
                    mask: 255,
                    shift: 0,
                });
                base += 256;
            }
            if base < entries {
                banks.push(Access {
                    base,
                    mask: 63,
                    shift: 0,
                });
            }
            for (phase, layout) in layouts.iter_mut().enumerate() {
                for (stream, access) in layout.iter_mut().enumerate() {
                    *access = banks[(stream + phase * pattern.reads()) % banks.len()];
                }
            }
        }
        layouts
    }

    #[inline]
    fn entry(&self, accesses: &[Access; 49], stream: usize, value: usize) -> F128 {
        let access = accesses[stream];
        self.table[access.base + ((value & access.mask) << access.shift)]
    }

    fn run(&self) -> F128 {
        let source = black_box(&self.source);
        let _ = black_box(&self.table);
        (0..CycleSource::cycles(source.as_ref()).div_ceil(CHUNK))
            .into_par_iter()
            .map(|chunk| {
                let start = chunk * CHUNK;
                let end = (start + CHUNK).min(CycleSource::cycles(source.as_ref()));
                let accesses = &self.accesses[chunk & 1];
                let mut sum = F128::from_raw(0);
                for cycle in start..end {
                    let mut stream = 0;
                    match self.pattern {
                        LookupPattern::Outer => {
                            for group in source.lanes(cycle) {
                                for word in group {
                                    for byte in word.to_le_bytes() {
                                        sum += self.entry(accesses, stream, usize::from(byte));
                                        stream += 1;
                                    }
                                }
                            }
                        }
                        LookupPattern::Lift => {
                            for word in 0..6 {
                                for byte in source.trace_word(word, cycle).to_le_bytes() {
                                    sum += self.entry(accesses, stream, usize::from(byte));
                                    stream += 1;
                                }
                            }
                        }
                        LookupPattern::Digits => {
                            let inc = source.trace_word(5, cycle).to_le_bytes();
                            for _ in 0..2 {
                                for byte in inc {
                                    sum += self.entry(accesses, stream, usize::from(byte));
                                    stream += 1;
                                }
                            }
                            for columns in [&DIGITS_FIRST[..], &DIGITS_SECOND[..]] {
                                for &column in columns {
                                    let digit = source.digit(column, cycle).map_or(0, |value| {
                                        if column == 18 {
                                            1
                                        } else {
                                            value
                                        }
                                    });
                                    sum += self.entry(accesses, stream, digit);
                                    stream += 1;
                                }
                            }
                        }
                        LookupPattern::Bytes => {
                            let row = source.rows()[cycle];
                            // Adjacent reads of a shared row byte model interleaved weights.
                            for byte in 0..29 {
                                let value = usize::from(row[byte / 8].to_le_bytes()[byte % 8]);
                                for _ in 0..BYTE_USES[byte] {
                                    sum += self.entry(accesses, stream, value);
                                    stream += 1;
                                }
                            }
                        }
                    }
                }
                sum
            })
            .reduce(|| F128::from_raw(0), |left, right| left + right)
    }
}

const DIGITS_FIRST: [usize; 13] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18];
const DIGITS_SECOND: [usize; 8] = [5, 6, 7, 8, 9, 10, 11, 18];
const BYTE_USES: [usize; 29] = [
    2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
];

#[derive(Clone, Copy)]
enum BucketLayout {
    Column,
    Fold { byte_selectors: usize, share: usize },
}

impl BucketLayout {
    fn geometry(self) -> (usize, [usize; 5]) {
        match self {
            BucketLayout::Column => (32 * 256, [0; 5]),
            BucketLayout::Fold { byte_selectors, .. } => {
                let variant = byte_selectors * 5 * 8 * 256 + (64 - byte_selectors) * 5 * 16 * 16;
                let metadata = variant;
                let shift = metadata + 64 * 8 * 16;
                let memory = shift + 512 * 16 * 16;
                let compare = memory + 128 * 2 * 16 * 16;
                let branch = compare + 512 * 3 * 16 * 16;
                (
                    branch + 2 * 16 * 16,
                    [metadata, shift, memory, compare, branch],
                )
            }
        }
    }
}

struct Bucket {
    source: Arc<SyntheticTrace>,
    layout: BucketLayout,
    scratch: Vec<Mutex<Vec<F128>>>,
    offsets: [usize; 5],
    operations: usize,
}

impl Bucket {
    fn new(source: Arc<SyntheticTrace>, layout: BucketLayout, threads: usize) -> Self {
        let (entries, offsets) = layout.geometry();
        let operations = match layout {
            BucketLayout::Column => CycleSource::cycles(source.as_ref()) * 32,
            BucketLayout::Fold {
                byte_selectors,
                share,
            } => (0..CycleSource::cycles(source.as_ref()).div_ceil(CHUNK))
                .into_par_iter()
                .map(|chunk| {
                    let start = chunk * CHUNK;
                    let end = (start + CHUNK).min(CycleSource::cycles(source.as_ref()));
                    (start..end)
                        .map(|cycle| {
                            let selector = Self::selector(&source, cycle, share);
                            let variant = if selector < byte_selectors { 40 } else { 80 };
                            variant
                                + 8
                                + 16 * usize::from(source.digit(13, cycle).is_some())
                                + 32 * usize::from(source.digit(14, cycle).is_some())
                                + 48 * usize::from(source.digit(15, cycle).is_some())
                                + 32 * usize::from(
                                    source.digit(16, cycle).is_some()
                                        && source.digit(19, cycle).is_some(),
                                )
                        })
                        .sum::<usize>()
                })
                .sum(),
        };
        Self {
            source,
            layout,
            scratch: (0..threads)
                .map(|_| Mutex::new(vec![F128::from_raw(0); entries]))
                .collect(),
            offsets,
            operations,
        }
    }

    #[inline]
    fn selector(source: &SyntheticTrace, cycle: usize, share: usize) -> usize {
        let natural = source.digit(12, cycle).unwrap_or(0);
        if source.trace_word(0, cycle) as usize % 100 < share {
            natural % 8
        } else {
            8 + natural % 56
        }
    }

    fn run(&mut self) -> F128 {
        let source = black_box(&self.source);
        let _ = black_box(&self.scratch);
        (0..CycleSource::cycles(source.as_ref()).div_ceil(CHUNK))
            .into_par_iter()
            .for_each(|chunk| {
                let worker = rayon::current_thread_index().unwrap_or(0);
                let mut buckets = self.scratch[worker]
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                let start = chunk * CHUNK;
                let end = (start + CHUNK).min(CycleSource::cycles(source.as_ref()));
                for cycle in start..end {
                    let weight = trace_value(source, cycle);
                    match self.layout {
                        BucketLayout::Column => {
                            for (word, &value) in source.rows()[cycle].iter().enumerate() {
                                for (byte, value) in value.to_le_bytes().into_iter().enumerate() {
                                    buckets[(word * 8 + byte) * 256 + usize::from(value)] += weight;
                                }
                            }
                        }
                        BucketLayout::Fold {
                            byte_selectors,
                            share,
                        } => {
                            let selector = Self::selector(source, cycle, share);
                            let by_byte = selector < byte_selectors;
                            let variant_base = if by_byte {
                                selector * 5 * 8 * 256
                            } else {
                                byte_selectors * 5 * 8 * 256
                                    + (selector - byte_selectors) * 5 * 16 * 16
                            };
                            for (word_slot, word) in [0, 1, 2, 4, 5].into_iter().enumerate() {
                                bucket_word(
                                    &mut buckets,
                                    variant_base,
                                    word_slot,
                                    source.trace_word(word, cycle),
                                    by_byte,
                                    weight,
                                );
                            }
                            for (slot, column) in (5..12).enumerate() {
                                let digit = source.digit(column, cycle).unwrap_or(0);
                                buckets[self.offsets[0] + (selector * 8 + slot) * 16 + digit] +=
                                    weight;
                            }
                            let flags = usize::from(source.digit(18, cycle).is_some())
                                | (usize::from(source.digit(19, cycle).is_some()) << 1)
                                | (usize::from(source.digit(20, cycle).is_some()) << 2);
                            buckets[self.offsets[0] + (selector * 8 + 7) * 16 + flags] += weight;
                            let low = source.digit(10, cycle).unwrap_or(0);
                            let high = source.digit(11, cycle).unwrap_or(0);
                            if let Some(kind) = source.digit(13, cycle) {
                                let base = self.offsets[1] + (low + 8 * high + 64 * kind) * 16 * 16;
                                bucket_word(
                                    &mut buckets,
                                    base,
                                    0,
                                    source.trace_word(0, cycle),
                                    false,
                                    weight,
                                );
                            }
                            if let Some(kind) = source.digit(14, cycle) {
                                let base = self.offsets[2] + (low + 8 * kind) * 2 * 16 * 16;
                                for (slot, word) in [3, 1].into_iter().enumerate() {
                                    bucket_word(
                                        &mut buckets,
                                        base,
                                        slot,
                                        source.trace_word(word, cycle),
                                        false,
                                        weight,
                                    );
                                }
                            }
                            let row = source.bytecode_index(cycle);
                            if let Some(kind) = source.digit(15, cycle) {
                                let base =
                                    self.offsets[3] + (low + 8 * high + 64 * kind) * 3 * 16 * 16;
                                for slot in 0..3 {
                                    let word = if slot < 2 {
                                        source.trace_word(slot, cycle)
                                    } else {
                                        source.bytecode_word(0, row)
                                    };
                                    bucket_word(&mut buckets, base, slot, word, false, weight);
                                }
                            }
                            if source.digit(16, cycle).is_some()
                                && source.digit(19, cycle).is_some()
                            {
                                for slot in 0..2 {
                                    bucket_word(
                                        &mut buckets,
                                        self.offsets[4],
                                        slot,
                                        source.bytecode_word(slot + 1, row),
                                        false,
                                        weight,
                                    );
                                }
                            }
                        }
                    }
                }
            });
        self.scratch.iter().fold(F128::from_raw(0), |sum, buckets| {
            let buckets = buckets.lock().unwrap_or_else(PoisonError::into_inner);
            let _ = black_box(&*buckets);
            sum + buckets[0]
        })
    }
}

#[inline]
fn bucket_word(
    buckets: &mut [F128],
    base: usize,
    slot: usize,
    word: u64,
    by_byte: bool,
    weight: F128,
) {
    if by_byte {
        for (position, value) in word.to_le_bytes().into_iter().enumerate() {
            buckets[base + (slot * 8 + position) * 256 + usize::from(value)] += weight;
        }
    } else {
        for position in 0..16 {
            let value = ((word >> (position * 4)) & 15) as usize;
            buckets[base + (slot * 16 + position) * 16 + value] += weight;
        }
    }
}

#[derive(Clone, Copy)]
enum ScatterMethod {
    Direct,
    Worker,
    Partitioned,
}

enum ScatterStorage {
    Direct(Vec<[AtomicU64; 2]>),
    Worker(Vec<Mutex<Vec<F128>>>),
    Partitioned {
        order: Vec<u32>,
        weights: Vec<F128>,
        offsets: Vec<usize>,
        output: Vec<F128>,
    },
}

struct Scatter {
    source: Arc<SyntheticTrace>,
    storage: ScatterStorage,
}

impl Scatter {
    fn new(
        source: Arc<SyntheticTrace>,
        method: ScatterMethod,
        threads: usize,
    ) -> Result<Self, ProbeError> {
        if source.bytecode_rows() != ROWS {
            return Err(ProbeError::Rows {
                expected: ROWS,
                actual: source.bytecode_rows(),
            });
        }
        let storage = match method {
            ScatterMethod::Direct => ScatterStorage::Direct(
                (0..ROWS)
                    .map(|_| [AtomicU64::new(0), AtomicU64::new(0)])
                    .collect(),
            ),
            ScatterMethod::Worker => ScatterStorage::Worker(
                (0..threads)
                    .map(|_| Mutex::new(vec![F128::from_raw(0); ROWS]))
                    .collect(),
            ),
            ScatterMethod::Partitioned => {
                let mut offsets = vec![0; ROW_RANGES + 1];
                for cycle in 0..CycleSource::cycles(source.as_ref()) {
                    offsets[source.bytecode_index(cycle) / (ROWS / ROW_RANGES) + 1] += 1;
                }
                for range in 0..ROW_RANGES {
                    offsets[range + 1] += offsets[range];
                }
                let mut cursor = offsets[..ROW_RANGES].to_vec();
                let mut order = vec![0; CycleSource::cycles(source.as_ref())];
                for cycle in 0..CycleSource::cycles(source.as_ref()) {
                    let range = source.bytecode_index(cycle) / (ROWS / ROW_RANGES);
                    order[cursor[range]] = cycle as u32;
                    cursor[range] += 1;
                }
                ScatterStorage::Partitioned {
                    order,
                    weights: vec![F128::from_raw(0); CycleSource::cycles(source.as_ref())],
                    offsets,
                    output: vec![F128::from_raw(0); ROWS],
                }
            }
        };
        Ok(Self { source, storage })
    }

    fn run(&mut self) -> F128 {
        let source = black_box(&self.source);
        match &mut self.storage {
            ScatterStorage::Direct(output) => {
                let _ = black_box(&*output);
                (0..CycleSource::cycles(source.as_ref()).div_ceil(CHUNK))
                    .into_par_iter()
                    .for_each(|chunk| {
                        let start = chunk * CHUNK;
                        let end = (start + CHUNK).min(CycleSource::cycles(source.as_ref()));
                        for cycle in start..end {
                            let row = source.bytecode_index(cycle);
                            let value = trace_value(source, cycle).to_raw();
                            let _ = output[row][0].fetch_xor(value as u64, Ordering::Relaxed);
                            let _ =
                                output[row][1].fetch_xor((value >> 64) as u64, Ordering::Relaxed);
                        }
                    });
                F128::from_raw(
                    u128::from(output[0][0].load(Ordering::Relaxed))
                        | (u128::from(output[0][1].load(Ordering::Relaxed)) << 64),
                )
            }
            ScatterStorage::Worker(tables) => {
                let _ = black_box(&*tables);
                (0..CycleSource::cycles(source.as_ref()).div_ceil(CHUNK))
                    .into_par_iter()
                    .for_each(|chunk| {
                        let worker = rayon::current_thread_index().unwrap_or(0);
                        let mut table = tables[worker]
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner);
                        let start = chunk * CHUNK;
                        let end = (start + CHUNK).min(CycleSource::cycles(source.as_ref()));
                        for cycle in start..end {
                            table[source.bytecode_index(cycle)] += trace_value(source, cycle);
                        }
                    });
                let mut stride = 1;
                while stride < tables.len() {
                    tables.par_chunks_mut(stride * 2).for_each(|pair| {
                        if pair.len() > stride {
                            let (left, right) = pair.split_at_mut(stride);
                            let left = left[0].get_mut().unwrap_or_else(PoisonError::into_inner);
                            let right = right[0].get_mut().unwrap_or_else(PoisonError::into_inner);
                            left.par_chunks_mut(CHUNK)
                                .zip(right.par_chunks(CHUNK))
                                .for_each(|(left, right)| {
                                    for (left, &right) in left.iter_mut().zip(right) {
                                        *left += right;
                                    }
                                });
                        }
                    });
                    stride *= 2;
                }
                let table = tables[0].get_mut().unwrap_or_else(PoisonError::into_inner);
                let _ = black_box(&*table);
                table[0]
            }
            ScatterStorage::Partitioned {
                order,
                weights,
                offsets,
                output,
            } => {
                let _ = black_box(&*order);
                weights
                    .par_chunks_mut(CHUNK)
                    .zip(order.par_chunks(CHUNK))
                    .for_each(|(weights, order)| {
                        for (weight, &cycle) in weights.iter_mut().zip(order) {
                            *weight = trace_value(source, cycle as usize);
                        }
                    });
                output
                    .par_chunks_mut(ROWS / ROW_RANGES)
                    .enumerate()
                    .for_each(|(range, output)| {
                        let start = offsets[range];
                        let end = offsets[range + 1];
                        for (&cycle, &weight) in order[start..end].iter().zip(&weights[start..end])
                        {
                            let row = source.bytecode_index(cycle as usize) % (ROWS / ROW_RANGES);
                            output[row] += weight;
                        }
                    });
                let _ = black_box(&*output);
                output[0]
            }
        }
    }
}

struct Fmadd {
    source: Arc<SyntheticTrace>,
    masks: [(F128, F128); 20],
    terms: usize,
}

impl Fmadd {
    fn new(source: Arc<SyntheticTrace>, terms: usize) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(0x0066_6d61_6464);
        Self {
            source,
            masks: std::array::from_fn(|_| (F128::random(&mut rng), F128::random(&mut rng))),
            terms,
        }
    }

    fn run(&self) -> F128 {
        let source = black_box(&self.source);
        let masks = black_box(&self.masks);
        (0..CycleSource::cycles(source.as_ref()).div_ceil(CHUNK))
            .into_par_iter()
            .map(|chunk| {
                let start = chunk * CHUNK;
                let end = (start + CHUNK).min(CycleSource::cycles(source.as_ref()));
                let mut total = F128::from_raw(0);
                for cycle in start..end {
                    let a = black_box(trace_value(source, cycle));
                    let b = black_box(F128::from_raw(
                        u128::from(source.trace_word(2, cycle))
                            | (u128::from(source.trace_word(3, cycle)) << 64),
                    ));
                    // Every length, including zero, prepares the same twenty pairs.
                    let operands: [(F128, F128); 20] =
                        black_box(std::array::from_fn(|i| (a + masks[i].0, b + masks[i].1)));
                    let result = if self.terms == 0 {
                        operands[0].0 + operands[0].1
                    } else {
                        let mut accumulator = F128Accumulator::default();
                        for &(a, b) in &operands[..self.terms] {
                            accumulator.fmadd(a, b);
                        }
                        accumulator.reduce()
                    };
                    total += result;
                }
                black_box(total)
            })
            .reduce(|| F128::from_raw(0), |a, b| a + b)
    }
}

#[derive(Clone, Copy)]
enum ArithmeticKind {
    Product,
    MulX,
    Word,
}

struct Arithmetic {
    source: Arc<SyntheticTrace>,
    kind: ArithmeticKind,
}

impl Arithmetic {
    fn word_mix(mut a: u64, mut b: u64) -> u64 {
        for (shift, mask) in [
            (1, 0xaaaa_aaaa_aaaa_aaaa),
            (2, 0xcccc_cccc_cccc_cccc),
            (4, 0xf0f0_f0f0_f0f0_f0f0),
        ] {
            a ^= (a << shift) & mask;
            b ^= (b << shift) & mask;
        }
        let mut value = (a & b) & 0x0101_0101_0101_0101;
        for (shift, mask) in [
            (7, 0x0003_0003_0003_0003),
            (14, 0x0000_000f_0000_000f),
            (28, 0xff),
        ] {
            value = (value | (value >> shift)) & mask;
        }
        value
    }

    fn run(&self) -> F128 {
        let source = black_box(&self.source);
        (0..CycleSource::cycles(source.as_ref()).div_ceil(CHUNK))
            .into_par_iter()
            .map(|chunk| {
                let start = chunk * CHUNK;
                let end = (start + CHUNK).min(CycleSource::cycles(source.as_ref()));
                let mut total = F128::from_raw(0);
                for cycle in start..end {
                    let a = black_box(trace_value(source, cycle));
                    let result = match self.kind {
                        ArithmeticKind::Product => {
                            let b = black_box(F128::from_raw(
                                u128::from(source.trace_word(2, cycle))
                                    | (u128::from(source.trace_word(3, cycle)) << 64),
                            ));
                            a * b
                        }
                        ArithmeticKind::MulX => {
                            let raw = a.to_raw();
                            F128::from_raw((raw << 1) ^ (0x87 & 0_u128.wrapping_sub(raw >> 127)))
                        }
                        ArithmeticKind::Word => F128::from_raw(u128::from(Self::word_mix(
                            a.to_raw() as u64,
                            (a.to_raw() >> 64) as u64,
                        ))),
                    };
                    total += black_box(result);
                }
                total
            })
            .reduce(|| F128::from_raw(0), |a, b| a + b)
    }
}

struct ReadSpec {
    base: usize,
    width: usize,
    bit: Option<usize>,
}

struct Readout {
    buckets: Vec<F128>,
    specs: Vec<ReadSpec>,
    output: Vec<F128>,
    operations: usize,
}

impl Readout {
    fn word(specs: &mut Vec<ReadSpec>, base: usize, words: usize, bits: usize) {
        let width = 1 << bits;
        for position in 0..words * 64 / bits {
            for bit in 0..bits {
                specs.push(ReadSpec {
                    base: base + position * width,
                    width,
                    bit: Some(bit),
                });
            }
        }
    }

    fn new(layout: BucketLayout) -> Self {
        let (entries, offsets) = layout.geometry();
        let mut specs = Vec::new();
        match layout {
            BucketLayout::Column => Self::word(&mut specs, 0, 4, 8),
            BucketLayout::Fold { byte_selectors, .. } => {
                for selector in 0..64 {
                    let by_byte = selector < byte_selectors;
                    let base = if by_byte {
                        selector * 5 * 8 * 256
                    } else {
                        byte_selectors * 5 * 8 * 256 + (selector - byte_selectors) * 5 * 16 * 16
                    };
                    Self::word(&mut specs, base, 5, if by_byte { 8 } else { 4 });
                    for digit in 0..7 {
                        for value in 1..if digit < 5 { 16 } else { 8 } {
                            specs.push(ReadSpec {
                                base: offsets[0] + (selector * 8 + digit) * 16 + value,
                                width: 1,
                                bit: None,
                            });
                        }
                    }
                    for bit in 0..3 {
                        specs.push(ReadSpec {
                            base: offsets[0] + (selector * 8 + 7) * 16,
                            width: 8,
                            bit: Some(bit),
                        });
                    }
                    specs.push(ReadSpec {
                        base,
                        width: if by_byte { 256 } else { 16 },
                        bit: None,
                    });
                }
                Self::word(&mut specs, offsets[1], 512, 4);
                Self::word(&mut specs, offsets[2], 128 * 2, 4);
                Self::word(&mut specs, offsets[3], 512 * 3, 4);
                Self::word(&mut specs, offsets[4], 2, 4);
                for selector in 0..512 {
                    specs.push(ReadSpec {
                        base: offsets[3] + selector * 3 * 16 * 16,
                        width: 16,
                        bit: None,
                    });
                }
            }
        }
        let operations = specs
            .iter()
            .map(|spec| {
                if spec.bit.is_some() {
                    spec.width / 2
                } else {
                    spec.width
                }
            })
            .sum();
        let output = vec![F128::from_raw(0); specs.len()];
        Self {
            buckets: field_values(entries),
            specs,
            output,
            operations,
        }
    }

    fn run(&mut self) -> F128 {
        let buckets = black_box(&self.buckets);
        let specs = black_box(&self.specs);
        self.output
            .par_iter_mut()
            .zip(specs.par_iter())
            .for_each(|(output, spec)| {
                let mut total = F128::from_raw(0);
                for value in 0..spec.width {
                    if spec.bit.is_none_or(|bit| value & (1 << bit) != 0) {
                        total += buckets[spec.base + value];
                    }
                }
                *output = total;
            });
        black_box(&self.output)
            .par_iter()
            .copied()
            .reduce(|| F128::from_raw(0), |a, b| a + b)
    }
}

struct Merge {
    arrays: Vec<Vec<F128>>,
    tree: bool,
}

impl Merge {
    fn new(threads: usize, tree: bool) -> Self {
        Self {
            arrays: (0..threads)
                .map(|_| vec![F128::from_raw(1); 10 * 1024 * 1024 / size_of::<F128>()])
                .collect(),
            tree,
        }
    }

    fn operations(&self) -> usize {
        let arrays = if self.tree {
            2 * self.arrays.len() - 1
        } else {
            self.arrays.len()
        };
        arrays * self.arrays[0].len()
    }

    fn run(&mut self) -> F128 {
        let arrays = black_box(&mut self.arrays);
        arrays.par_iter_mut().for_each(|array| {
            array
                .par_chunks_mut(CHUNK)
                .for_each(|chunk| chunk.fill(F128::from_raw(0)));
        });
        let _ = black_box(&mut *arrays);
        if self.tree {
            let mut stride = 1;
            while stride < arrays.len() {
                arrays.par_chunks_mut(stride * 2).for_each(|pair| {
                    if pair.len() > stride {
                        let (left, right) = pair.split_at_mut(stride);
                        left[0]
                            .par_chunks_mut(CHUNK)
                            .zip(right[0].par_chunks(CHUNK))
                            .for_each(|(left, right)| {
                                for (left, &right) in left.iter_mut().zip(right) {
                                    *left += right;
                                }
                            });
                    }
                });
                stride *= 2;
            }
        }
        let _ = black_box(&*arrays);
        arrays[0][0]
    }
}

enum Unit {
    Lookup(Box<Lookup>),
    Bucket(Bucket),
    Scatter(Scatter),
    Fmadd(Box<Fmadd>),
    Merge(Merge),
    Arithmetic(Arithmetic),
    Readout(Readout),
}

impl Unit {
    fn new(
        case: &ProbeCase,
        source: Arc<SyntheticTrace>,
        threads: usize,
    ) -> Result<Self, ProbeError> {
        if threads == 0 {
            return Err(ProbeError::Threads { threads });
        }
        let invalid = || ProbeError::Case {
            unit: case.unit.to_owned(),
            variant: case.variant.clone(),
        };
        match case.unit {
            "lookup" => {
                let (pattern, size) = case.variant.rsplit_once('_').ok_or_else(invalid)?;
                let kib = size
                    .strip_suffix("kib")
                    .and_then(|size| size.parse::<usize>().ok())
                    .filter(|size| [5, 32, 64, 69, 96, 196].contains(size))
                    .ok_or_else(invalid)?;
                let pattern = match pattern {
                    "outer" => LookupPattern::Outer,
                    "source_lift" => LookupPattern::Lift,
                    "g_digits" => LookupPattern::Digits,
                    "g_bytes" => LookupPattern::Bytes,
                    _ => return Err(invalid()),
                };
                Ok(Self::Lookup(Box::new(Lookup::new(source, pattern, kib))))
            }
            "bucket" => {
                let layout = if case.variant == "column_128kib" {
                    BucketLayout::Column
                } else {
                    let (layout, share) = case.variant.split_once("_share_").ok_or_else(invalid)?;
                    let byte_selectors = match layout {
                        "fold_none" => 0,
                        "fold_hot8" => 8,
                        "fold_all" => 64,
                        _ => return Err(invalid()),
                    };
                    let share = share
                        .parse::<usize>()
                        .ok()
                        .filter(|share| [0, 25, 50, 75, 100].contains(share))
                        .ok_or_else(invalid)?;
                    BucketLayout::Fold {
                        byte_selectors,
                        share,
                    }
                };
                Ok(Self::Bucket(Bucket::new(source, layout, threads)))
            }
            "scatter" => {
                let (method, rows) = case.variant.split_once("_rows_").ok_or_else(invalid)?;
                if rows != "16" && rows != "20" {
                    return Err(invalid());
                }
                let method = match method {
                    "direct_atomic_halves" => ScatterMethod::Direct,
                    "worker_tree" => ScatterMethod::Worker,
                    "partitioned_gather" => ScatterMethod::Partitioned,
                    _ => return Err(invalid()),
                };
                Ok(Self::Scatter(Scatter::new(source, method, threads)?))
            }
            "fmadd" => {
                let terms = case
                    .variant
                    .strip_prefix("chain_")
                    .and_then(|terms| terms.parse::<usize>().ok())
                    .filter(|terms| [0, 1, 2, 4, 8, 20].contains(terms))
                    .ok_or_else(invalid)?;
                Ok(Self::Fmadd(Box::new(Fmadd::new(source, terms))))
            }
            "arithmetic" => {
                let kind = match case.variant.as_str() {
                    "products" => ArithmeticKind::Product,
                    "mul_x_raw_shift_substitute" => ArithmeticKind::MulX,
                    "word_monomial_mix" => ArithmeticKind::Word,
                    _ => return Err(invalid()),
                };
                Ok(Self::Arithmetic(Arithmetic { source, kind }))
            }
            "readout" => {
                let layout = match case.variant.as_str() {
                    "column_128kib" => BucketLayout::Column,
                    "fold_none" => BucketLayout::Fold {
                        byte_selectors: 0,
                        share: 0,
                    },
                    "fold_hot8" => BucketLayout::Fold {
                        byte_selectors: 8,
                        share: 0,
                    },
                    "fold_all" => BucketLayout::Fold {
                        byte_selectors: 64,
                        share: 0,
                    },
                    _ => return Err(invalid()),
                };
                Ok(Self::Readout(Readout::new(layout)))
            }
            "merge" => {
                let tree = match case.variant.as_str() {
                    "zero_fill_10mib" => false,
                    "zero_fill_tree_10mib" => true,
                    _ => return Err(invalid()),
                };
                Ok(Self::Merge(Merge::new(threads, tree)))
            }
            _ => Err(invalid()),
        }
    }
}

impl ProbeKernel for Unit {
    fn operations(&self) -> [usize; 2] {
        match self {
            Self::Lookup(unit) => [
                CycleSource::cycles(unit.source.as_ref()) * unit.pattern.reads(),
                0,
            ],
            Self::Bucket(unit) => [unit.operations, 0],
            Self::Scatter(unit) => [CycleSource::cycles(unit.source.as_ref()), 0],
            Self::Fmadd(unit) => [CycleSource::cycles(unit.source.as_ref()), 0],
            Self::Merge(unit) => [unit.operations(), 0],
            Self::Arithmetic(unit) => [
                CycleSource::cycles(unit.source.as_ref())
                    * if matches!(unit.kind, ArithmeticKind::Word) {
                        29
                    } else {
                        1
                    },
                0,
            ],
            Self::Readout(unit) => [unit.operations, 0],
        }
    }

    fn run(&mut self) -> F128 {
        match self {
            Self::Lookup(unit) => unit.run(),
            Self::Bucket(unit) => unit.run(),
            Self::Scatter(unit) => unit.run(),
            Self::Fmadd(unit) => unit.run(),
            Self::Merge(unit) => unit.run(),
            Self::Arithmetic(unit) => unit.run(),
            Self::Readout(unit) => unit.run(),
        }
    }

    fn chain_terms(&self) -> Option<usize> {
        match self {
            Self::Fmadd(unit) => Some(unit.terms),
            _ => None,
        }
    }
}

#[inline]
fn trace_value(source: &SyntheticTrace, cycle: usize) -> F128 {
    F128::from_raw(
        u128::from(source.trace_word(0, cycle)) | (u128::from(source.trace_word(1, cycle)) << 64),
    )
}

fn field_values(entries: usize) -> Vec<F128> {
    let mut rng = ChaCha20Rng::seed_from_u64(0x0074_6162_6c65);
    (0..entries).map(|_| F128::random(&mut rng)).collect()
}

fn main() -> Result<(), RunnerError> {
    let mut cases = Vec::new();
    for kib in [5, 32, 64, 69, 96, 196] {
        for pattern in ["outer", "source_lift", "g_digits", "g_bytes"] {
            cases.push(ProbeCase {
                unit: "lookup",
                variant: format!("{pattern}_{kib}kib"),
                profiles: BOTH,
            });
        }
    }
    cases.push(ProbeCase {
        unit: "bucket",
        variant: "column_128kib".to_owned(),
        profiles: BOTH,
    });
    for layout in ["none", "hot8", "all"] {
        for share in [0, 25, 50, 75, 100] {
            cases.push(ProbeCase {
                unit: "bucket",
                variant: format!("fold_{layout}_share_{share}"),
                profiles: BOTH,
            });
        }
    }
    for rows in [16, 20] {
        for method in ["direct_atomic_halves", "worker_tree", "partitioned_gather"] {
            cases.push(ProbeCase {
                unit: "scatter",
                variant: format!("{method}_rows_{rows}"),
                profiles: if rows == 16 { LOCAL } else { ALL_ROWS },
            });
        }
    }
    for terms in [0, 1, 2, 4, 8, 20] {
        cases.push(ProbeCase {
            unit: "fmadd",
            variant: format!("chain_{terms}"),
            profiles: BOTH,
        });
    }
    for variant in ["zero_fill_10mib", "zero_fill_tree_10mib"] {
        cases.push(ProbeCase {
            unit: "merge",
            variant: variant.to_owned(),
            profiles: BOTH,
        });
    }
    for variant in [
        "products",
        "mul_x_raw_shift_substitute",
        "word_monomial_mix",
    ] {
        cases.push(ProbeCase {
            unit: "arithmetic",
            variant: variant.to_owned(),
            profiles: BOTH,
        });
    }
    for variant in ["column_128kib", "fold_none", "fold_hot8", "fold_all"] {
        cases.push(ProbeCase {
            unit: "readout",
            variant: variant.to_owned(),
            profiles: BOTH,
        });
    }
    run_probe(&cases, Unit::new)
}
