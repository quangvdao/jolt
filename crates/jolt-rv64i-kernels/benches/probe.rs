//! Unit-cost calibration on opaque seeded packed traces.
//!
//! Construction, source generation, domain/count planning, allocation and initial
//! zeroing are outside kernel timing. Each record prints its sample median and
//! min/max in wall-clock nanoseconds divided by its operation count. Twelve-thread
//! figures are a scaling column, never single-thread unit prices or CPU time.
//! The final unit-table uses the largest requested stream with one thread and
//! the representative ids below; excluded cases/one-thread runs say not_measured.
//! Constants in this bench retain both specification estimates for comparison.
//!
//! | Unit | Representative record (before size/thread suffix) | Timed work |
//! |---|---|---|
//! | M | arithmetic/products/local | independent reduced trace-word products and chunk XOR |
//! | A | fmadd/fit/local, slope_ns | least-squares slope of fused stack chains |
//! | R | fmadd/fit/local, reduction_ns | fit intercept minus zero-length preparation/XOR baseline |
//! | L | lookup/g_digits_69kib/local | the canonical digit builder's 37 reads/XORs |
//! | Bk | bucket/fold_none_share_0/all_rows | default nibble-layout updates, selectors prepared first |
//! | sct | sct/partitioned_emit_rows_20/all_rows | cycle-order pair emission and range-local application |
//! | mrg | readout/column_128kib/independent | per-bit bucket read-out, normalized by reads/XORs |
//! | X | arithmetic/mul_x_raw_shift_substitute/local | raw shift and conditional modulus XOR |
//! | w | arithmetic/word_monomial_mix/local | representative outer monomial word operations |
//!
//! Fused chains have lengths 0,1,2,4,8,20. Every cycle prepares the same twenty
//! masked operand pairs at every length. Nonempty chains accumulate and reduce
//! on the stack, then XOR the result into the chunk total; zero uses one prepared
//! operand and the same total XOR, with no multiplication/reduction. Fits use
//! nonzero-length per-cycle medians and print slope, intercept, the zero control,
//! their difference and the largest absolute residual. Reduction estimates may
//! be negative on a noisy run; no estimate is clamped or represented as latency.
//!
//! mul_x is absent from this branch's field API. Its substitute implements the
//! specified polynomial-basis shift/reduction using modulus mask 0x87. The word
//! stream mirrors outer rounds 1--3: two three-stage Moebius transforms, an AND
//! product and stride-eight gather (nine shifts, eleven ANDs, six XORs, three
//! ORs, 29 operations). The spec does not define an exact ratio behind its 410 w;
//! this representative mix is explicit rather than claiming whole-core coverage.
//!
//! Canonical lookup patterns and the specification passages they mirror:
//! | Pattern | Passage/suboperation | L covered |
//! |---|---|---|
//! | outer_materialise_32kib | Bitwise outer, Rounds 7 and 8: six lane WordLifts | 48 of outer's 253 per cycle |
//! | word_lift_bytes_32kib | Routers, source_lift: six trace-word WordLifts | 48 of source_lift's 59 per cycle |
//! | g_digits_69kib | Reduction, g_pass_digits: two Inc lifts and compact indicator tables | all 37 per cycle |
//! | g_bytes_196kib | Reduction, g_pass_bytes: interleaved weighted row bytes | all 49 per cycle |
//!
//! Every entry is 16 bytes. Canonical word lifts use eight 256-entry banks,
//! eight index bits, shared by the six words. Canonical digits have sixteen
//! distinct 256-entry byte banks plus twenty-one separately weighted compact
//! banks: fifteen of width 16, four of width 8, one of width 2, one of width 8
//! for three combined flags. They total 4,378 entries/70,048 bytes (69 KiB rounded).
//! Each stream has a fixed bank: no rotation or byte/digit aliasing. Canonical
//! bytes use twenty 256-position interleaved pairs and nine 256-entry singletons;
//! positions 0--7 and 17--28 have two adjacent weights for one eight-bit index.
//! Table values are seeded field entries: the cost is loads/XORs, not table building.
//!
//! Other footprints are labelled pressure, not builder implementations. Ordinary
//! layouts use 256-entry/eight-bit banks with 64-entry/six-bit tails at 5/69 KiB,
//! reuse banks in small sets, and alternate two fixed bank assignments by chunk.
//! Interleaved pressure layouts reuse pairs/singletons; the 5 KiB case uses one
//! 128-position pair (seven bits) and 64-entry singleton (six bits). No timed
//! lookup divides/mods indices. Every lookup prints allocated_bytes and
//! addressable_entries: a conservative union of synthetic value domains across
//! stream assignments, computed before timing. It is not an empirical hit or
//! cache-residency count; sparse one-hot bytes and high zero RAM digits remain sparse.
//!
//! Column bucket updates use 32 row bytes and 128 KiB per worker. Fold updates
//! mirror fold_pass's five cycle banks: five Variant words, seven chunk buckets
//! and combined flags, plus active Shift, Memory, Compare and Branch words.
//! Layouts occupy 10.383/11.477/19.133 MiB per worker. Requested hot shares are
//! remapped once at construction; timing reads a selector byte, without remapping.
//! Bucket read-out uses those same layouts: 128 reads per byte bit, eight per
//! nibble bit, direct indicator cells, combined-flag sums and selector One totals.
//! Both fold streams exclude its separate four-word-per-visited-bytecode-row pass.
//!
//! sct counts destinations outside timing. The timed stream writes rows and
//! weights in cycle order through per-chunk cursors into one range-contiguous
//! buffer, then applies buffered pairs without source reads or a worker-table
//! merge. The scatter group's direct atomic halves, worker tables with merge,
//! and destination-ordered gather are comparisons; none prices sct. Coverage at
//! log_t=22 is 2^16 local and 2^20 all_rows destinations in a 2^20-row output;
//! shorter streams visit at most their cycle count. Rows/weights cost 20 bytes
//! per cycle. Stack slice views split safely using the cached per-chunk counts.
//!
//! Fixed-size readout/merge cases generate no trace and run once per thread count,
//! under independent/fixed. Zero-fill is W*N operations on 10 MiB arrays;
//! zero-fill plus tree is (2W-1)*N; merge alone is (W-1)*N and is skipped at W=1.
//! Both tree callers use one allocation-free stride-doubling helper. Timing
//! includes scheduling and final checksums, excludes allocation/first touch.
//! All trace-driven chunks contain 4096 cycles, independent of the thread count.
//! --units accepts lookup,bucket,scatter,sct,fmadd,arithmetic,readout,merge, or all.

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

use support::merge::tree_merge;
use support::scatter::{PartitionedScatter, ScatterError};
use support::{run_probe, ProbeCase, ProbeKernel, ProbeRecord, RunnerError};

type F128Accumulator = <F128 as WithAccumulator>::Accumulator;

const CHUNK: usize = 4096;
const ROWS: usize = 1 << 20;
const ROW_RANGES: usize = 256;
const BOTH: &[SynthProfile] = &[SynthProfile::Local, SynthProfile::AllRows];
const LOCAL: &[SynthProfile] = &[SynthProfile::Local];
const ALL_ROWS: &[SynthProfile] = &[SynthProfile::AllRows];

#[derive(Debug, Error)]
enum ProbeError {
    #[error(transparent)]
    Scatter(#[from] ScatterError),
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
    addressable_entries: usize,
}

#[derive(Clone, Copy, Default)]
struct Access {
    base: usize,
    mask: usize,
    shift: usize,
}

impl Lookup {
    fn new(source: Arc<SyntheticTrace>, pattern: LookupPattern, kib: usize) -> Self {
        let canonical_digits = matches!(pattern, LookupPattern::Digits) && kib == 69;
        let entries = if canonical_digits {
            4378
        } else {
            kib * 1024 / size_of::<F128>()
        };
        let accesses = if canonical_digits {
            Self::digit_accesses()
        } else {
            Self::accesses(pattern, entries)
        };
        let addressable_entries = Self::domain_bound(&source, pattern, &accesses, entries);
        Self {
            source,
            table: field_values(entries),
            pattern,
            accesses,
            addressable_entries,
        }
    }

    fn digit_accesses() -> [[Access; 49]; 2] {
        let mut access = [Access::default(); 49];
        let mut base = 0;
        for entry in &mut access[..16] {
            *entry = Access {
                base,
                mask: 255,
                shift: 0,
            };
            base += 256;
        }
        for (group, columns) in [&DIGITS_FIRST[..], &DIGITS_SECOND[..]]
            .into_iter()
            .enumerate()
        {
            for (index, &column) in columns.iter().enumerate() {
                let width = if column == 18 {
                    if group == 0 {
                        2
                    } else {
                        8
                    }
                } else if column < 10 {
                    16
                } else {
                    8
                };
                let stream = 16
                    + if group == 0 {
                        index
                    } else {
                        DIGITS_FIRST.len() + index
                    };
                access[stream] = Access {
                    base,
                    mask: width - 1,
                    shift: 0,
                };
                base += width;
            }
        }
        [access; 2]
    }

    fn domain_bound(
        source: &SyntheticTrace,
        pattern: LookupPattern,
        accesses: &[[Access; 49]; 2],
        entries: usize,
    ) -> usize {
        let visited = match source.profile() {
            SynthProfile::Local => source.bytecode_rows() / 16,
            SynthProfile::AllRows | SynthProfile::UniformDigits => source.bytecode_rows(),
        }
        .max(1)
        .min(CycleSource::cycles(source));
        let mut bytecode_digits = [[false; 16]; 5];
        let mut pc_bytes = [[false; 256]; 8];
        let first = source.bytecode_index(0);
        for index in 0..visited {
            let row = (first + index) & (source.bytecode_rows() - 1);
            for (column, values) in bytecode_digits.iter_mut().enumerate() {
                values[(row >> (column * 4)) & 15] = true;
            }
            for (byte, value) in ((row as u64) * 4).to_le_bytes().into_iter().enumerate() {
                pc_bytes[byte][usize::from(value)] = true;
            }
        }
        // NextPC can read the row just beyond a shorter CLI stream.
        let next = (first + visited) & (source.bytecode_rows() - 1);
        for (byte, value) in ((next as u64) * 4).to_le_bytes().into_iter().enumerate() {
            pc_bytes[byte][usize::from(value)] = true;
        }
        let digits: [Vec<usize>; 12] = std::array::from_fn(|column| {
            if column < 5 {
                bytecode_digits[column]
                    .iter()
                    .enumerate()
                    .filter_map(|(v, &possible)| possible.then_some(v))
                    .collect()
            } else if column == 8 || column == 9 {
                vec![0]
            } else {
                (0..if column < 10 { 16 } else { 8 }).collect()
            }
        });
        let domains: Vec<Vec<usize>> = match pattern {
            LookupPattern::Outer => (0..48).map(|_| (0..256).collect()).collect(),
            LookupPattern::Lift => (0..48)
                .map(|stream| {
                    if stream / 8 == 4 {
                        pc_bytes[stream % 8]
                            .iter()
                            .enumerate()
                            .filter_map(|(v, &possible)| possible.then_some(v))
                            .collect()
                    } else {
                        (0..256).collect()
                    }
                })
                .collect(),
            LookupPattern::Digits => {
                let mut domains: Vec<Vec<usize>> = (0..16).map(|_| (0..256).collect()).collect();
                for (group, columns) in [&DIGITS_FIRST[..], &DIGITS_SECOND[..]]
                    .into_iter()
                    .enumerate()
                {
                    for &column in columns {
                        domains.push(if column == 18 {
                            (0..if group == 0 { 2 } else { 8 }).collect()
                        } else {
                            digits[column].clone()
                        });
                    }
                }
                domains
            }
            LookupPattern::Bytes => {
                let mut bytes: [Vec<usize>; 32] = std::array::from_fn(|byte| {
                    if byte < 8 {
                        (0..256).collect()
                    } else {
                        vec![0]
                    }
                });
                for (column, values) in digits.iter().enumerate() {
                    let start = if column < 10 {
                        64 + 15 * column
                    } else {
                        214 + 7 * (column - 10)
                    };
                    for (byte, domain) in bytes.iter_mut().enumerate().skip(8) {
                        let choices: Vec<_> = values
                            .iter()
                            .map(|&value| {
                                if value != 0 && (start + value - 1) / 8 == byte {
                                    1 << ((start + value - 1) % 8)
                                } else {
                                    0
                                }
                            })
                            .collect();
                        let mut possible = [false; 256];
                        for &old in domain.iter() {
                            for &choice in &choices {
                                possible[old | choice] = true;
                            }
                        }
                        *domain = possible
                            .iter()
                            .enumerate()
                            .filter_map(|(v, &possible)| possible.then_some(v))
                            .collect();
                    }
                }
                for bit in 228..231 {
                    let byte = bit / 8;
                    let old = bytes[byte].clone();
                    bytes[byte].extend(old.iter().map(|v| v | (1 << (bit % 8))));
                }
                let mut domains = Vec::new();
                for byte in 0..29 {
                    for _ in 0..BYTE_USES[byte] {
                        domains.push(bytes[byte].clone());
                    }
                }
                domains
            }
        };
        let mut reachable = vec![false; entries];
        for layout in accesses {
            for (stream, domain) in domains.iter().enumerate() {
                let access = layout[stream];
                for &value in domain {
                    reachable[access.base + ((value & access.mask) << access.shift)] = true;
                }
            }
        }
        reachable.iter().filter(|&&reachable| reachable).count()
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
                            for (group, columns) in [&DIGITS_FIRST[..], &DIGITS_SECOND[..]]
                                .into_iter()
                                .enumerate()
                            {
                                for &column in columns {
                                    let digit = if column == 18 {
                                        if group == 0 {
                                            usize::from(source.digit(18, cycle).is_some())
                                        } else {
                                            (18..21).enumerate().fold(0, |bits, (bit, column)| {
                                                bits | (usize::from(
                                                    source.digit(column, cycle).is_some(),
                                                ) << bit)
                                            })
                                        }
                                    } else {
                                        source.digit(column, cycle).unwrap_or(0)
                                    };
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
    selectors: Vec<u8>,
    operations: usize,
}

impl Bucket {
    fn new(source: Arc<SyntheticTrace>, layout: BucketLayout, threads: usize) -> Self {
        let (entries, offsets) = layout.geometry();
        let selectors: Vec<u8> = match layout {
            BucketLayout::Column => Vec::new(),
            BucketLayout::Fold { share, .. } => (0..CycleSource::cycles(source.as_ref()))
                .into_par_iter()
                .map(|cycle| Self::selector(&source, cycle, share) as u8)
                .collect(),
        };
        let operations = match layout {
            BucketLayout::Column => CycleSource::cycles(source.as_ref()) * 32,
            BucketLayout::Fold { byte_selectors, .. } => (0..CycleSource::cycles(source.as_ref())
                .div_ceil(CHUNK))
                .into_par_iter()
                .map(|chunk| {
                    let start = chunk * CHUNK;
                    let end = (start + CHUNK).min(CycleSource::cycles(source.as_ref()));
                    (start..end)
                        .map(|cycle| {
                            let selector = usize::from(selectors[cycle]);
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
            selectors,
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
        let selectors = black_box(&self.selectors);
        (0..CycleSource::cycles(source.as_ref()).div_ceil(CHUNK))
            .into_par_iter()
            .for_each(|chunk| {
                let worker = rayon::current_thread_index().unwrap_or(0);
                let mut buckets = self.scratch[worker]
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                let start = chunk * CHUNK;
                let end = (start + CHUNK).min(CycleSource::cycles(source.as_ref()));
                #[expect(clippy::needless_range_loop, reason = "the shared cycle loop also serves the column layout, which has no selector vector")]
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
                        BucketLayout::Fold { byte_selectors, .. } => {
                            let selector = usize::from(selectors[cycle]);
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
    Gather,
}

enum ScatterStorage {
    Direct(Vec<[AtomicU64; 2]>),
    Worker(Vec<Mutex<Vec<F128>>>),
    Gather {
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
            ScatterMethod::Gather => {
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
                ScatterStorage::Gather {
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
                tree_merge(tables, CHUNK, |table| {
                    table
                        .get_mut()
                        .unwrap_or_else(PoisonError::into_inner)
                        .as_mut_slice()
                });
                let table = tables[0].get_mut().unwrap_or_else(PoisonError::into_inner);
                let _ = black_box(&*table);
                table[0]
            }
            ScatterStorage::Gather {
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
                        operands[0].0
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum MergeMode {
    Zero,
    ZeroTree,
    Tree,
}

struct Merge {
    arrays: Vec<Vec<F128>>,
    mode: MergeMode,
}

impl Merge {
    fn new(threads: usize, mode: MergeMode) -> Self {
        Self {
            arrays: (0..threads)
                .map(|_| vec![F128::from_raw(1); 10 * 1024 * 1024 / size_of::<F128>()])
                .collect(),
            mode,
        }
    }

    fn operations(&self) -> usize {
        let arrays = match self.mode {
            MergeMode::Zero => self.arrays.len(),
            MergeMode::ZeroTree => 2 * self.arrays.len() - 1,
            MergeMode::Tree => self.arrays.len() - 1,
        };
        arrays * self.arrays[0].len()
    }

    fn run(&mut self) -> F128 {
        let arrays = black_box(&mut self.arrays);
        if self.mode != MergeMode::Tree {
            arrays.par_iter_mut().for_each(|array| {
                array
                    .par_chunks_mut(CHUNK)
                    .for_each(|chunk| chunk.fill(F128::from_raw(0)));
            });
        }
        let _ = black_box(&mut *arrays);
        if self.mode != MergeMode::Zero {
            tree_merge(arrays, CHUNK, Vec::as_mut_slice);
        }
        let _ = black_box(&*arrays);
        arrays[0][0]
    }
}

enum Unit {
    Lookup(Box<Lookup>),
    Bucket(Bucket),
    Scatter(Scatter),
    Partitioned(Box<PartitionedScatter>),
    Fmadd(Box<Fmadd>),
    Merge(Merge),
    Arithmetic(Arithmetic),
    Readout(Readout),
}

impl Unit {
    fn new(
        case: &ProbeCase,
        source: Option<Arc<SyntheticTrace>>,
        threads: usize,
    ) -> Result<Self, ProbeError> {
        if threads == 0 {
            return Err(ProbeError::Threads { threads });
        }
        let invalid = || ProbeError::Case {
            unit: case.unit.to_owned(),
            variant: case.variant.clone(),
        };
        let trace = || source.as_ref().map(Arc::clone).ok_or_else(invalid);
        match case.unit {
            "lookup" => {
                let variant = case
                    .variant
                    .strip_prefix("pressure_")
                    .unwrap_or(&case.variant);
                let (pattern, size) = variant.rsplit_once('_').ok_or_else(invalid)?;
                let kib = size
                    .strip_suffix("kib")
                    .and_then(|size| size.parse::<usize>().ok())
                    .filter(|size| [5, 32, 64, 69, 96, 196].contains(size))
                    .ok_or_else(invalid)?;
                let pattern = match pattern {
                    "outer_materialise" => LookupPattern::Outer,
                    "word_lift_bytes" => LookupPattern::Lift,
                    "g_digits" => LookupPattern::Digits,
                    "g_bytes" => LookupPattern::Bytes,
                    _ => return Err(invalid()),
                };
                Ok(Self::Lookup(Box::new(Lookup::new(trace()?, pattern, kib))))
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
                Ok(Self::Bucket(Bucket::new(trace()?, layout, threads)))
            }
            "scatter" => {
                let (method, rows) = case.variant.split_once("_rows_").ok_or_else(invalid)?;
                if rows != "16" && rows != "20" {
                    return Err(invalid());
                }
                let method = match method {
                    "direct_atomic_halves" => ScatterMethod::Direct,
                    "worker_tree" => ScatterMethod::Worker,
                    "gather" => ScatterMethod::Gather,
                    _ => return Err(invalid()),
                };
                Ok(Self::Scatter(Scatter::new(trace()?, method, threads)?))
            }
            "sct" => Ok(Self::Partitioned(Box::new(PartitionedScatter::new(
                trace()?,
            )?))),
            "fmadd" => {
                let terms = case
                    .variant
                    .strip_prefix("chain_")
                    .and_then(|terms| terms.parse::<usize>().ok())
                    .filter(|terms| [0, 1, 2, 4, 8, 20].contains(terms))
                    .ok_or_else(invalid)?;
                Ok(Self::Fmadd(Box::new(Fmadd::new(trace()?, terms))))
            }
            "arithmetic" => {
                let kind = match case.variant.as_str() {
                    "products" => ArithmeticKind::Product,
                    "mul_x_raw_shift_substitute" => ArithmeticKind::MulX,
                    "word_monomial_mix" => ArithmeticKind::Word,
                    _ => return Err(invalid()),
                };
                Ok(Self::Arithmetic(Arithmetic {
                    source: trace()?,
                    kind,
                }))
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
                let mode = match case.variant.as_str() {
                    "zero_fill_10mib" => MergeMode::Zero,
                    "zero_fill_tree_10mib" => MergeMode::ZeroTree,
                    "tree_only_10mib" => MergeMode::Tree,
                    _ => return Err(invalid()),
                };
                Ok(Self::Merge(Merge::new(threads, mode)))
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
            Self::Partitioned(unit) => [unit.cycles(), 0],
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
            Self::Partitioned(unit) => unit.run(),
            Self::Fmadd(unit) => unit.run(),
            Self::Merge(unit) => unit.run(),
            Self::Arithmetic(unit) => unit.run(),
            Self::Readout(unit) => unit.run(),
        }
    }

    fn lookup_layout(&self) -> Option<(usize, usize)> {
        match self {
            Self::Lookup(unit) => Some((
                unit.table.len() * size_of::<F128>(),
                unit.addressable_entries,
            )),
            _ => None,
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

#[derive(Clone, Copy)]
enum PriceMetric {
    Time,
    Slope,
    Reduction,
}

struct UnitPrice {
    unit: &'static str,
    record: &'static str,
    estimates: [f64; 2],
    metric: PriceMetric,
}

const UNIT_PRICES: [UnitPrice; 9] = [
    UnitPrice {
        unit: "M",
        record: "probe/arithmetic/products/local/",
        estimates: [1.83, 0.9],
        metric: PriceMetric::Time,
    },
    UnitPrice {
        unit: "A",
        record: "probe/fmadd/fit/local/",
        estimates: [1.10, 0.7],
        metric: PriceMetric::Slope,
    },
    UnitPrice {
        unit: "R",
        record: "probe/fmadd/fit/local/",
        estimates: [0.7, 0.2],
        metric: PriceMetric::Reduction,
    },
    UnitPrice {
        unit: "L",
        record: "probe/lookup/g_digits_69kib/local/",
        estimates: [0.4, 0.4],
        metric: PriceMetric::Time,
    },
    UnitPrice {
        unit: "Bk",
        record: "probe/bucket/fold_none_share_0/all_rows/",
        estimates: [0.6, 0.6],
        metric: PriceMetric::Time,
    },
    UnitPrice {
        unit: "sct",
        record: "probe/sct/partitioned_emit_rows_20/all_rows/",
        estimates: [1.4, 1.4],
        metric: PriceMetric::Time,
    },
    UnitPrice {
        unit: "mrg",
        record: "probe/readout/column_128kib/independent/",
        estimates: [0.3, 0.3],
        metric: PriceMetric::Time,
    },
    UnitPrice {
        unit: "X",
        record: "probe/arithmetic/mul_x_raw_shift_substitute/local/",
        estimates: [0.2, 0.2],
        metric: PriceMetric::Time,
    },
    UnitPrice {
        unit: "w",
        record: "probe/arithmetic/word_monomial_mix/local/",
        estimates: [0.05, 0.05],
        metric: PriceMetric::Time,
    },
];

impl UnitPrice {
    #[expect(
        clippy::print_stdout,
        reason = "unit-table is the probe's calibration summary contract"
    )]
    fn print_table(records: &[ProbeRecord]) {
        println!("unit-table");
        for price in &UNIT_PRICES {
            let record = records
                .iter()
                .filter(|record| record.id.starts_with(price.record) && record.id.ends_with("/1"))
                .max_by_key(|record| {
                    record
                        .id
                        .split('/')
                        .nth(4)
                        .and_then(|log_t| log_t.parse::<usize>().ok())
                        .unwrap_or(0)
                });
            let value = record.and_then(|record| match price.metric {
                PriceMetric::Time => Some(record.median),
                PriceMetric::Slope => record.fit.as_ref().map(|fit| fit.slope),
                PriceMetric::Reduction => record.fit.as_ref().map(|fit| fit.reduction),
            });
            if let Some((record, value)) = record.zip(value) {
                println!(
                    "unit-table {} record={} median_ns={value:.6} spec_1_ns={:.6} spec_2_ns={:.6}",
                    price.unit, record.id, price.estimates[0], price.estimates[1]
                );
            } else {
                println!("unit-table {} record={} median_ns=not_measured spec_1_ns={:.6} spec_2_ns={:.6} reason=matching_single_thread_record_not_requested", price.unit, price.record, price.estimates[0], price.estimates[1]);
            }
        }
    }
}

fn main() -> Result<(), RunnerError> {
    let mut cases = Vec::new();
    for kib in [5, 32, 64, 69, 96, 196] {
        for pattern in [
            "outer_materialise",
            "word_lift_bytes",
            "g_digits",
            "g_bytes",
        ] {
            cases.push(ProbeCase {
                unit: "lookup",
                variant: if (kib == 32
                    && ["outer_materialise", "word_lift_bytes"].contains(&pattern))
                    || (kib == 69 && pattern == "g_digits")
                    || (kib == 196 && pattern == "g_bytes")
                {
                    format!("{pattern}_{kib}kib")
                } else {
                    format!("pressure_{pattern}_{kib}kib")
                },
                profiles: BOTH,
                minimum_threads: 1,
            });
        }
    }
    cases.push(ProbeCase {
        unit: "bucket",
        variant: "column_128kib".to_owned(),
        profiles: BOTH,
        minimum_threads: 1,
    });
    for layout in ["none", "hot8", "all"] {
        for share in [0, 25, 50, 75, 100] {
            cases.push(ProbeCase {
                unit: "bucket",
                variant: format!("fold_{layout}_share_{share}"),
                profiles: BOTH,
                minimum_threads: 1,
            });
        }
    }
    for rows in [16, 20] {
        for method in ["direct_atomic_halves", "worker_tree", "gather"] {
            cases.push(ProbeCase {
                unit: "scatter",
                variant: format!("{method}_rows_{rows}"),
                profiles: if rows == 16 { LOCAL } else { ALL_ROWS },
                minimum_threads: 1,
            });
        }
    }
    for rows in [16, 20] {
        cases.push(ProbeCase {
            unit: "sct",
            variant: format!("partitioned_emit_rows_{rows}"),
            profiles: if rows == 16 { LOCAL } else { ALL_ROWS },
            minimum_threads: 1,
        });
    }
    for terms in [0, 1, 2, 4, 8, 20] {
        cases.push(ProbeCase {
            unit: "fmadd",
            variant: format!("chain_{terms}"),
            profiles: BOTH,
            minimum_threads: 1,
        });
    }
    for variant in ["zero_fill_10mib", "zero_fill_tree_10mib", "tree_only_10mib"] {
        cases.push(ProbeCase {
            unit: "merge",
            variant: variant.to_owned(),
            profiles: &[],
            minimum_threads: if variant == "tree_only_10mib" { 2 } else { 1 },
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
            minimum_threads: 1,
        });
    }
    for variant in ["column_128kib", "fold_none", "fold_hot8", "fold_all"] {
        cases.push(ProbeCase {
            unit: "readout",
            variant: variant.to_owned(),
            profiles: &[],
            minimum_threads: 1,
        });
    }
    let records = run_probe(&cases, Unit::new)?;
    UnitPrice::print_table(&records);
    Ok(())
}
