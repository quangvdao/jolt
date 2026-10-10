//! Unit-cost candidates and contextual kernel streams on seeded inputs.
//!
//! Construction, preparation, allocation and initial zeroing precede timing.
//! Records report median/min/max wall time divided by operation count; multiple
//! threads report scaling, not unit prices. The summary selects single-thread
//! records, or says not_measured when they were omitted. A specification row is
//! replaced only from a quiet-machine run, with every model and threshold
//! recomputed in the same change. No record here changes an estimate.
//!
//! | Unit | Candidate or context | Timed region |
//! |---|---|---|
//! | M | arithmetic/product_hot | hot prepared independent reduced products, checksum |
//! | A | arithmetic/chain_hot/256 and /1024 | identical runtime term loop; incremental ns/term |
//! | R | arithmetic/reduce_hot and /reduce_hot_control (context) | hot accumulator reductions versus opaque-lane checksum; signed difference |
//! | L | lookup/g_digits_69kib; other canonical layouts retained | fixed-bank field loads/XORs with necessary source decoding |
//! | Bk | bucket/fold_none_share_0/all_rows | model's no-byte-bucket layout updates, prepared selectors |
//! | sct | sct/partitioned_emit_rows_20/all_rows | cached-slot weight emission and buffered range application |
//! | mrg | merge/zero_fill_10mib, /tree_only_10mib, readout/* (context) | separately counted fills, two-array merges, selected-half reads/XORs |
//! | X | arithmetic/mul_x_hot_raw_shift_substitute | independent hot 128-bit shifts and conditional modulus XOR |
//! | w | arithmetic/word_monomial_hot | two live transforms, two coefficient shifts, AND, gather; 30 logical word operations |
//!
//! Hot arithmetic runs once per thread setting without a trace. The pair block
//! is 32 KiB; X uses 2 KiB, words 3 KiB and unreduced accumulators 6 KiB on native
//! aarch64. Inputs are opaque once per repeated block and only a block checksum
//! escapes. One runtime-length chain body serves 1,2,4,8,20,256,1024; short chains
//! report chain/term totals, with no affine fit. The two longest chains alone
//! give the incremental A candidate after assembly inspection confirms the same
//! term body. Boundary calls, loads and checksums remain disclosed overhead.
//! Reduction's safe control merges all opaque accumulator lanes, with no reduce:
//! three lanes are XORed versus the reduced checksum's one. Their difference
//! remains context because these checksum instruction mixes are unmatched.
//! Fused trace chains retain their operand preparation and per-cycle opacity;
//! their ids are contextual totals and are not arithmetic unit candidates.
//!
//! mul_x is absent from the field API; the named raw-shift substitute implements
//! the polynomial-basis operation with modulus mask 0x87. The word block computes
//! two three-stage Möbius transforms (18 logical operations), live subset
//! alignment (2), AND/mask (2), and stride-eight gather (8). Its consumed output
//! depends on every stage; native shifted operands may fuse logical operations.
//! This is one outer monomial coefficient product, not the whole outer core.
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
//! Lookup and bucket records price their calibration loops, rather than library
//! lift or bucket calls; partitioned scatter and pool merges call the library.
//! Column bucket updates use 32 row bytes and 131,584 bytes per worker. Fold updates
//! mirror fold_pass's five cycle banks: five Variant words, seven chunk buckets
//! and combined flags, plus active Shift, Memory, Compare and Branch words.
//! Layouts occupy 10.423/11.521/19.207 MiB per worker. Requested hot shares are
//! remapped once at construction; timing reads a selector byte, without remapping.
//! Bucket read-out uses those same layouts: 128 reads per byte bit, eight per
//! nibble bit, direct indicator cells, combined-flag sums and selector One totals.
//! Both fold streams exclude its separate four-word-per-visited-bytecode-row pass.
//!
//! sct calls the library ScatterPlan: construction caches each cycle's u16 slot
//! and each slot's u16 row offset. Timing emits weights through those slots into
//! chunk-contiguous range segments and applies them through cached offsets, with
//! no source index reads, cursors, row buffer or worker-table merge. The scatter group's direct atomic halves, worker tables with merge,
//! and destination-ordered gather are comparisons; none prices sct. Coverage at
//! log_t=22 is 2^16 local and 2^20 all_rows destinations in a 2^20-row output;
//! shorter streams visit at most their cycle count. Persistent routing costs
//! four bytes per cycle plus u32 range descriptors; pass scratch is 16 bytes per cycle.
//!
//! Fixed-size readout/merge cases generate no trace and run once per thread count,
//! under independent/fixed. Zero-fill is W*N operations on 10 MiB arrays;
//! zero-fill plus tree is (2W-1)*N using ScratchPool. Merge-only always uses two
//! prepared 10 MiB arrays, executing N load/load/XOR/store operations even at
//! one thread. Read-out has its own footprint and selected-read count. No single
//! record is the common mrg estimate. Timing includes scheduling and checksums,
//! excludes allocation/first touch. Trace streams use 4096-cycle chunks, except
//! ScatterPlan, which owns its deterministic geometry.
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

use jolt_rv64i_kernels::packed::buckets::{BucketPlacement, ByteBuckets, NibbleBuckets};
use jolt_rv64i_kernels::packed::pool::{PoolError, ScratchPool};
use jolt_rv64i_kernels::packed::scatter::{ScatterError, ScatterPlan};
use jolt_rv64i_kernels::router::fold::FoldCalibration;
use jolt_rv64i_kernels::router::shape::RouterError;
use jolt_rv64i_kernels::source::{SourceError, ValidatedTrace};
use support::arithmetic::HotArithmetic;
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
    #[error(transparent)]
    Pool(#[from] PoolError),
    #[error(transparent)]
    Source(#[from] SourceError),
    #[error(transparent)]
    Router(#[from] RouterError),
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
    layout: LookupLayout,
    addressable_entries: usize,
}

enum LookupLayout {
    Word,
    Digits,
    Bytes,
    Pressure(Box<[[Access; 49]; 2]>),
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
        let layout = match (pattern, kib) {
            (LookupPattern::Outer | LookupPattern::Lift, 32) => LookupLayout::Word,
            (LookupPattern::Digits, 69) => LookupLayout::Digits,
            (LookupPattern::Bytes, 196) => LookupLayout::Bytes,
            _ => LookupLayout::Pressure(Box::new(accesses)),
        };
        Self {
            source,
            table: field_values(entries),
            pattern,
            layout,
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
            SynthProfile::AllRows
            | SynthProfile::UniformDigits
            | SynthProfile::SmallValues
            | SynthProfile::Chained => source.bytecode_rows(),
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
        match &self.layout {
            LookupLayout::Word => {
                if matches!(self.pattern, LookupPattern::Outer) {
                    self.run_words::<true>()
                } else {
                    self.run_words::<false>()
                }
            }
            LookupLayout::Digits => self.run_digits(),
            LookupLayout::Bytes => self.run_bytes(),
            LookupLayout::Pressure(accesses) => self.run_pressure(accesses),
        }
    }

    fn run_chunks(&self, value: impl Fn(&SyntheticTrace, usize) -> F128 + Sync) -> F128 {
        let source = black_box(&self.source);
        (0..CycleSource::cycles(source.as_ref()).div_ceil(CHUNK))
            .into_par_iter()
            .map(|chunk| {
                let start = chunk * CHUNK;
                let end = (start + CHUNK).min(CycleSource::cycles(source.as_ref()));
                let mut sum = F128::from_raw(0);
                for cycle in start..end {
                    sum += value(source, cycle);
                }
                sum
            })
            .reduce(|| F128::from_raw(0), |left, right| left + right)
    }

    #[inline(never)]
    fn run_words<const OUTER: bool>(&self) -> F128 {
        let (banks, _) = black_box(self.table.as_slice()).as_chunks::<256>();
        self.run_chunks(|source, cycle| {
            let mut sum = F128::from_raw(0);
            if OUTER {
                for group in source.lanes(cycle) {
                    for word in group {
                        for (bank, byte) in banks.iter().zip(word.to_le_bytes()) {
                            sum += bank[usize::from(byte)];
                        }
                    }
                }
            } else {
                for word in 0..6 {
                    for (bank, byte) in banks
                        .iter()
                        .zip(source.trace_word(word, cycle).to_le_bytes())
                    {
                        sum += bank[usize::from(byte)];
                    }
                }
            }
            sum
        })
    }

    #[inline(never)]
    fn run_digits(&self) -> F128 {
        let table = black_box(self.table.as_slice());
        let (byte_banks, _) = table[..4096].as_chunks::<256>();
        self.run_chunks(|source, cycle| {
            let mut sum = F128::from_raw(0);
            let inc = source.trace_word(5, cycle).to_le_bytes();
            for banks in byte_banks.chunks_exact(8) {
                for (bank, byte) in banks.iter().zip(inc) {
                    sum += bank[usize::from(byte)];
                }
            }
            macro_rules! digit {
                ($column:literal, $base:literal) => {
                    sum += table[$base + source.digit($column, cycle).unwrap_or(0)];
                };
            }
            digit!(0, 4096);
            digit!(1, 4112);
            digit!(2, 4128);
            digit!(3, 4144);
            digit!(4, 4160);
            digit!(5, 4176);
            digit!(6, 4192);
            digit!(7, 4208);
            digit!(8, 4224);
            digit!(9, 4240);
            digit!(10, 4256);
            digit!(11, 4264);
            sum += table[4272 + usize::from(source.digit(18, cycle).is_some())];
            digit!(5, 4274);
            digit!(6, 4290);
            digit!(7, 4306);
            digit!(8, 4322);
            digit!(9, 4338);
            digit!(10, 4354);
            digit!(11, 4362);
            let flags = usize::from(source.digit(18, cycle).is_some())
                | (usize::from(source.digit(19, cycle).is_some()) << 1)
                | (usize::from(source.digit(20, cycle).is_some()) << 2);
            sum += table[4370 + flags];
            sum
        })
    }

    #[inline(never)]
    fn run_bytes(&self) -> F128 {
        let table = black_box(self.table.as_slice());
        self.run_chunks(|source, cycle| {
            let row = source.rows()[cycle];
            let mut sum = F128::from_raw(0);
            macro_rules! pair {
                ($word:literal, $shift:literal, $base:literal) => {
                    let index = (((row[$word] >> $shift) & 255) as usize) << 1;
                    sum += table[$base + index];
                    sum += table[$base + index + 1];
                };
            }
            macro_rules! single {
                ($word:literal, $shift:literal, $base:literal) => {
                    sum += table[$base + (((row[$word] >> $shift) & 255) as usize)];
                };
            }
            pair!(0, 0, 0);
            pair!(0, 8, 512);
            pair!(0, 16, 1024);
            pair!(0, 24, 1536);
            pair!(0, 32, 2048);
            pair!(0, 40, 2560);
            pair!(0, 48, 3072);
            pair!(0, 56, 3584);
            single!(1, 0, 10240);
            single!(1, 8, 10496);
            single!(1, 16, 10752);
            single!(1, 24, 11008);
            single!(1, 32, 11264);
            single!(1, 40, 11520);
            single!(1, 48, 11776);
            single!(1, 56, 12032);
            single!(2, 0, 12288);
            pair!(2, 8, 4096);
            pair!(2, 16, 4608);
            pair!(2, 24, 5120);
            pair!(2, 32, 5632);
            pair!(2, 40, 6144);
            pair!(2, 48, 6656);
            pair!(2, 56, 7168);
            pair!(3, 0, 7680);
            pair!(3, 8, 8192);
            pair!(3, 16, 8704);
            pair!(3, 24, 9216);
            pair!(3, 32, 9728);
            sum
        })
    }

    fn run_pressure(&self, layouts: &[[Access; 49]; 2]) -> F128 {
        let source = black_box(&self.source);
        let _ = black_box(&self.table);
        (0..CycleSource::cycles(source.as_ref()).div_ceil(CHUNK))
            .into_par_iter()
            .map(|chunk| {
                let start = chunk * CHUNK;
                let end = (start + CHUNK).min(CycleSource::cycles(source.as_ref()));
                let accesses = &layouts[chunk & 1];
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
    fn calibration(self) -> Result<Option<FoldCalibration>, RouterError> {
        match self {
            BucketLayout::Column => Ok(None),
            BucketLayout::Fold { byte_selectors, .. } => {
                FoldCalibration::new(byte_selectors).map(Some)
            }
        }
    }
}

struct Bucket {
    source: Arc<SyntheticTrace>,
    layout: BucketLayout,
    scratch: Vec<Mutex<Vec<F128>>>,
    selectors: Vec<u8>,
    operations: usize,
    calibration: Option<BucketCalibration>,
}

struct BucketCalibration {
    bases: [Vec<usize>; 5],
    metadata: [usize; 64],
}

impl BucketCalibration {
    fn new(calibration: &FoldCalibration) -> Result<Self, RouterError> {
        let mut bases: [Vec<usize>; 5] = std::array::from_fn(|_| Vec::new());
        for (shape, (bases, selectors)) in bases.iter_mut().zip(calibration.selectors()).enumerate()
        {
            *bases = (0..selectors)
                .map(|selector| calibration.shape_base(shape, selector))
                .collect::<Result<_, _>>()?;
        }
        let metadata: Vec<usize> = (0..calibration.selectors()[0])
            .map(|selector| calibration.variant_metadata_base(selector))
            .collect::<Result<_, _>>()?;
        let metadata =
            metadata
                .try_into()
                .map_err(|metadata: Vec<usize>| RouterError::TableLength {
                    table: "calibration metadata",
                    expected: 64,
                    actual: metadata.len(),
                })?;
        Ok(Self { bases, metadata })
    }

    #[inline]
    fn base(&self, shape: usize, selector: usize) -> usize {
        let bases = &self.bases[shape];
        bases[selector.min(bases.len() - 1)]
    }
}

impl Bucket {
    fn new(
        source: Arc<SyntheticTrace>,
        layout: BucketLayout,
        threads: usize,
    ) -> Result<Self, ProbeError> {
        let fold = layout.calibration()?;
        let entries = fold.as_ref().map_or(
            4 * BucketPlacement::Byte.word_entries(),
            FoldCalibration::entries,
        );
        let calibration = fold.as_ref().map(BucketCalibration::new).transpose()?;
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
        Ok(Self {
            source,
            layout,
            scratch: (0..threads)
                .map(|_| Mutex::new(vec![F128::from_raw(0); entries]))
                .collect(),
            selectors,
            operations,
            calibration,
        })
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

    #[expect(
        clippy::unwrap_used,
        reason = "fold bucket cases construct calibration before timing; column cases do not use it"
    )]
    fn run(&mut self) -> F128 {
        let source = black_box(&self.source);
        let _ = black_box(&self.scratch);
        let selectors = black_box(&self.selectors);
        let fold = black_box(self.calibration.as_ref());
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
                                    buckets[BucketPlacement::Byte.position_offset(
                                        word * ByteBuckets::POSITIONS_PER_WORD + byte,
                                    ) + usize::from(value)] += weight;
                                }
                            }
                        }
                        BucketLayout::Fold { byte_selectors, .. } => {
                            let selector = usize::from(selectors[cycle]);
                            let by_byte = selector < byte_selectors;
                            let fold = fold.unwrap();
                            let variant_base = fold.base(0, selector);
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
                                buckets[fold.metadata[selector & 63] + slot * 16 + digit] +=
                                    weight;
                            }
                            let flags = usize::from(source.digit(18, cycle).is_some())
                                | (usize::from(source.digit(19, cycle).is_some()) << 1)
                                | (usize::from(source.digit(20, cycle).is_some()) << 2);
                            buckets[fold.metadata[selector & 63] + 7 * 16 + flags] += weight;
                            let low = source.digit(10, cycle).unwrap_or(0);
                            let high = source.digit(11, cycle).unwrap_or(0);
                            if let Some(kind) = source.digit(13, cycle) {
                                let base = fold.base(1, low + 8 * high + 64 * kind);
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
                                let base = fold.base(2, low + 8 * kind);
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
                                    fold.base(3, low + 8 * high + 64 * kind);
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
                                        fold.base(4, 0),
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
    let placement = BucketPlacement::from_bytes(by_byte);
    let base = base + slot * placement.word_entries();
    if by_byte {
        for (position, value) in word.to_le_bytes().into_iter().enumerate() {
            buckets[base + placement.position_offset(position) + usize::from(value)] += weight;
        }
    } else {
        for position in 0..NibbleBuckets::POSITIONS_PER_WORD {
            let value = ((word >> (position * NibbleBuckets::BITS_PER_POSITION))
                & (NibbleBuckets::ENTRIES_PER_POSITION - 1) as u64)
                as usize;
            buckets[base + placement.position_offset(position) + value] += weight;
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
    Worker(ScratchPool),
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

#[expect(
    clippy::unwrap_used,
    reason = "sequential chunk bodies respect the constructed scratch bound; no merge overlaps loans"
)]
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
            ScatterMethod::Worker => {
                let pool = ScratchPool::new(ROWS)?;
                let guards: Vec<_> = (0..threads)
                    .map(|_| pool.take())
                    .collect::<Result<_, _>>()?;
                drop(guards);
                ScatterStorage::Worker(pool)
            }
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
            ScatterStorage::Worker(pool) => {
                (0..CycleSource::cycles(source.as_ref()).div_ceil(CHUNK))
                    .into_par_iter()
                    .for_each(|chunk| {
                        let mut table = pool.take().unwrap();
                        let start = chunk * CHUNK;
                        let end = (start + CHUNK).min(CycleSource::cycles(source.as_ref()));
                        for cycle in start..end {
                            table[source.bytecode_index(cycle)] += trace_value(source, cycle);
                        }
                    });
                let table = pool.merge().unwrap();
                let _ = black_box(&table);
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
    fn word(specs: &mut Vec<ReadSpec>, base: usize, words: usize, by_byte: bool) {
        let placement = BucketPlacement::from_bytes(by_byte);
        let bits = if by_byte {
            ByteBuckets::BITS_PER_POSITION
        } else {
            NibbleBuckets::BITS_PER_POSITION
        };
        let width = 1 << bits;
        for word in 0..words {
            for position in 0..64 / bits {
                for bit in 0..bits {
                    specs.push(ReadSpec {
                        base: base
                            + word * placement.word_entries()
                            + placement.position_offset(position),
                        width,
                        bit: Some(bit),
                    });
                }
            }
        }
    }

    fn new(layout: BucketLayout) -> Result<Self, ProbeError> {
        let calibration = layout.calibration()?;
        let entries = calibration.as_ref().map_or(
            4 * BucketPlacement::Byte.word_entries(),
            FoldCalibration::entries,
        );
        let mut specs = Vec::new();
        match layout {
            BucketLayout::Column => Self::word(&mut specs, 0, 4, true),
            BucketLayout::Fold { byte_selectors, .. } => {
                let fold = calibration.as_ref().ok_or_else(|| ProbeError::Case {
                    unit: "readout".to_owned(),
                    variant: "calibration".to_owned(),
                })?;
                let selectors = fold.selectors();
                let word_sets = fold.word_sets();
                for selector in 0..selectors[0] {
                    let by_byte = selector < byte_selectors;
                    let base = fold.variant_base(selector)?;
                    Self::word(&mut specs, base, word_sets[0], by_byte);
                    for digit in 0..7 {
                        for value in 1..if digit < 5 { 16 } else { 8 } {
                            specs.push(ReadSpec {
                                base: fold.variant_metadata_base(selector)? + digit * 16 + value,
                                width: 1,
                                bit: None,
                            });
                        }
                    }
                    for bit in 0..3 {
                        specs.push(ReadSpec {
                            base: fold.variant_metadata_base(selector)? + 7 * 16,
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
                for shape in 1..selectors.len() {
                    for selector in 0..selectors[shape] {
                        Self::word(
                            &mut specs,
                            fold.shape_base(shape, selector)?,
                            word_sets[shape],
                            false,
                        );
                    }
                }
                for selector in 0..selectors[3] {
                    specs.push(ReadSpec {
                        base: fold.shape_base(3, selector)?,
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
        Ok(Self {
            buckets: field_values(entries),
            specs,
            output,
            operations,
        })
    }

    fn run(&mut self) -> F128 {
        let buckets = black_box(&self.buckets);
        let specs = black_box(&self.specs);
        self.output
            .par_iter_mut()
            .zip(specs.par_iter())
            .for_each(|(output, spec)| {
                let bucket = &buckets[spec.base..spec.base + spec.width];
                let mut total = F128::from_raw(0);
                match spec.bit {
                    Some(bit) => {
                        let half = 1 << bit;
                        for period in bucket.chunks_exact(2 * half) {
                            for &value in &period[half..] {
                                total += value;
                            }
                        }
                    }
                    None => {
                        for &value in bucket {
                            total += value;
                        }
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

enum MergeStorage {
    Pool(ScratchPool),
    Pair { left: Vec<F128>, right: Vec<F128> },
}

struct Merge {
    storage: MergeStorage,
    mode: MergeMode,
    arrays: usize,
    len: usize,
}

#[expect(
    clippy::unwrap_used,
    reason = "benchmark phases own the pool exclusively and all loans end during construction"
)]
impl Merge {
    fn new(threads: usize, mode: MergeMode) -> Result<Self, PoolError> {
        let len = 10 * 1024 * 1024 / size_of::<F128>();
        let arrays = if mode == MergeMode::Tree { 2 } else { threads };
        let storage = if mode == MergeMode::Tree {
            MergeStorage::Pair {
                left: field_values(len),
                right: field_values(len),
            }
        } else {
            let pool = ScratchPool::new(len)?;
            let mut guards: Vec<_> = (0..arrays).map(|_| pool.take()).collect::<Result<_, _>>()?;
            for guard in &mut guards {
                guard.fill(F128::from_raw(1));
            }
            drop(guards);
            MergeStorage::Pool(pool)
        };
        Ok(Self {
            storage,
            mode,
            arrays,
            len,
        })
    }

    fn operations(&self) -> usize {
        let arrays = match self.mode {
            MergeMode::Zero => self.arrays,
            MergeMode::ZeroTree => 2 * self.arrays - 1,
            MergeMode::Tree => self.arrays - 1,
        };
        arrays * self.len
    }

    fn layout(&self) -> (usize, usize) {
        (self.len * size_of::<F128>(), self.arrays)
    }

    fn run(&mut self) -> F128 {
        match &mut self.storage {
            MergeStorage::Pair { left, right } => {
                let left = black_box(left);
                let right = black_box(right);
                left.par_chunks_mut(CHUNK)
                    .zip(right.par_chunks(CHUNK))
                    .for_each(|(left, right)| {
                        for (left, &right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                    });
                let _ = black_box(&left);
                left[0]
            }
            MergeStorage::Pool(pool) => {
                let pool = black_box(pool);
                pool.zero().unwrap();
                if self.mode == MergeMode::ZeroTree {
                    let result = pool.merge().unwrap();
                    let _ = black_box(&result);
                    result[0]
                } else {
                    black_box(F128::from_raw(0))
                }
            }
        }
    }
}

struct PartitionedScatter {
    source: Arc<SyntheticTrace>,
    plan: ScatterPlan<SyntheticTrace>,
    weights: Vec<F128>,
    output: Vec<F128>,
}

impl PartitionedScatter {
    fn new(source: Arc<SyntheticTrace>) -> Result<Self, ProbeError> {
        let validated = Arc::new(ValidatedTrace::new(Arc::clone(&source))?);
        let plan = ScatterPlan::new(validated)?;
        Ok(Self {
            weights: vec![F128::from_raw(0); plan.cycles()],
            output: vec![F128::from_raw(0); plan.bytecode_rows()],
            source,
            plan,
        })
    }
    fn cycles(&self) -> usize {
        self.plan.cycles()
    }
    #[expect(
        clippy::unwrap_used,
        reason = "the constructor sizes every buffer for this immutable plan"
    )]
    fn run(&mut self) -> F128 {
        let source = black_box(&self.source);
        self.plan
            .scatter_into(
                |cycle| trace_value(source, cycle),
                &mut self.weights,
                &mut self.output,
            )
            .unwrap();
        let _ = black_box(&self.output);
        self.output[0]
    }
}

enum Unit {
    Lookup(Box<Lookup>),
    Bucket(Box<Bucket>),
    Scatter(Scatter),
    Partitioned(Box<PartitionedScatter>),
    Fmadd(Box<Fmadd>),
    Merge(Merge),
    Hot(HotArithmetic),
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
                let variant = case.variant.as_str();
                let layout = if variant == "column_128kib" {
                    BucketLayout::Column
                } else {
                    let (layout, share) = variant.split_once("_share_").ok_or_else(invalid)?;
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
                Ok(Self::Bucket(Box::new(Bucket::new(
                    trace()?,
                    layout,
                    threads,
                )?)))
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
            "arithmetic" if case.variant == "product_hot" => {
                Ok(Self::Hot(HotArithmetic::product()))
            }
            "arithmetic" if case.variant.starts_with("chain_hot/") => {
                let terms = case
                    .variant
                    .strip_prefix("chain_hot/")
                    .and_then(|n| n.parse::<usize>().ok())
                    .filter(|n| [1, 2, 4, 8, 20, 256, 1024].contains(n))
                    .ok_or_else(invalid)?;
                Ok(Self::Hot(HotArithmetic::chain(terms)))
            }
            "arithmetic" if case.variant.starts_with("reduce_hot") => {
                if !["reduce_hot", "reduce_hot_control"].contains(&case.variant.as_str()) {
                    return Err(invalid());
                }
                Ok(Self::Hot(HotArithmetic::reduction(
                    case.variant == "reduce_hot_control",
                )))
            }
            "arithmetic" if case.variant == "mul_x_hot_raw_shift_substitute" => {
                Ok(Self::Hot(HotArithmetic::mul_x()))
            }
            "arithmetic" if case.variant == "word_monomial_hot" => {
                Ok(Self::Hot(HotArithmetic::word()))
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
                Ok(Self::Readout(Readout::new(layout)?))
            }
            "merge" => {
                let mode = match case.variant.as_str() {
                    "zero_fill_10mib" => MergeMode::Zero,
                    "zero_fill_tree_10mib" => MergeMode::ZeroTree,
                    "tree_only_10mib" => MergeMode::Tree,
                    _ => return Err(invalid()),
                };
                Ok(Self::Merge(Merge::new(threads, mode)?))
            }
            _ => Err(invalid()),
        }
    }
}

impl ProbeKernel for Unit {
    fn operations(&self) -> usize {
        match self {
            Self::Lookup(unit) => CycleSource::cycles(unit.source.as_ref()) * unit.pattern.reads(),
            Self::Bucket(unit) => unit.operations,
            Self::Scatter(unit) => CycleSource::cycles(unit.source.as_ref()),
            Self::Partitioned(unit) => unit.cycles(),
            Self::Fmadd(unit) => CycleSource::cycles(unit.source.as_ref()),
            Self::Merge(unit) => unit.operations(),
            Self::Readout(unit) => unit.operations,
            Self::Hot(unit) => unit.operations(),
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
            Self::Readout(unit) => unit.run(),
            Self::Hot(unit) => unit.run(),
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

    fn memory_layout(&self) -> Option<(usize, usize)> {
        match self {
            Self::Merge(unit) => Some(unit.layout()),
            Self::Readout(unit) => Some((unit.buckets.len() * size_of::<F128>(), 1)),
            _ => None,
        }
    }

    fn chain_terms(&self) -> Option<usize> {
        match self {
            Self::Hot(unit) => unit.terms(),
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

// Point-one and provisional point-two estimates; measurement never updates them.
const ESTIMATES: [[f64; 2]; 9] = [
    [1.83, 0.9],
    [1.1, 0.7],
    [0.7, 0.2],
    [0.4, 0.4],
    [0.6, 0.6],
    [1.4, 1.4],
    [0.3, 0.3],
    [0.2, 0.2],
    [0.05, 0.05],
];

struct UnitCandidates;

impl UnitCandidates {
    fn find<'a>(records: &'a [ProbeRecord], prefix: &str) -> Option<&'a ProbeRecord> {
        records
            .iter()
            .filter(|r| r.id.starts_with(prefix) && r.id.ends_with("/1"))
            .max_by_key(|r| {
                r.id.rsplit('/')
                    .nth(1)
                    .and_then(|n| n.parse::<usize>().ok())
                    .unwrap_or(0)
            })
    }

    #[expect(
        clippy::print_stdout,
        reason = "candidate records are the probe summary"
    )]
    fn record(
        records: &[ProbeRecord],
        unit: &str,
        prefix: &str,
        disposition: &str,
        estimates: [f64; 2],
    ) {
        if let Some(r) = Self::find(records, prefix) {
            print!(
                "unit-candidates {unit} {disposition} record={} median_ns={:.6} spec {},{}",
                r.id, r.median, estimates[0], estimates[1]
            );
            if let Some((bytes, arrays)) = r.layout {
                print!(" bytes_per_array={bytes} arrays={arrays}");
            }
            println!();
        } else {
            println!("unit-candidates {unit} {disposition} record={prefix} median_ns=not_measured spec {},{}",estimates[0],estimates[1]);
        }
    }

    #[expect(
        clippy::print_stdout,
        reason = "candidate records are the probe summary"
    )]
    fn print(records: &[ProbeRecord]) {
        println!("unit-candidates");
        Self::record(
            records,
            "M",
            "probe/arithmetic/product_hot/",
            "candidate",
            ESTIMATES[0],
        );
        let short = Self::find(records, "probe/arithmetic/chain_hot/256/");
        let long = Self::find(records, "probe/arithmetic/chain_hot/1024/");
        if let Some((a, b)) = short.zip(long) {
            println!(
                "unit-candidates A candidate records={},{} incremental_ns={:.6} spec {},{}",
                a.id,
                b.id,
                (b.median - a.median) / 768.0,
                ESTIMATES[1][0],
                ESTIMATES[1][1]
            );
        } else {
            println!("unit-candidates A candidate records=chain_hot/256,chain_hot/1024 incremental_ns=not_measured spec {},{}",ESTIMATES[1][0],ESTIMATES[1][1]);
        }
        Self::record(
            records,
            "R",
            "probe/arithmetic/reduce_hot/",
            "context",
            ESTIMATES[2],
        );
        Self::record(
            records,
            "R",
            "probe/arithmetic/reduce_hot_control/",
            "context",
            ESTIMATES[2],
        );
        if let Some((a, b)) = Self::find(records, "probe/arithmetic/reduce_hot/")
            .zip(Self::find(records, "probe/arithmetic/reduce_hot_control/"))
        {
            println!("unit-candidates R context records={},{} difference_ns={:.6} spec {},{} reason=checksum_lane_mixes_differ",a.id,b.id,a.median-b.median,ESTIMATES[2][0],ESTIMATES[2][1]);
        }
        for prefix in [
            "probe/lookup/g_digits_69kib/local/",
            "probe/lookup/word_lift_bytes_32kib/local/",
            "probe/lookup/outer_materialise_32kib/local/",
            "probe/lookup/g_bytes_196kib/local/",
        ] {
            Self::record(records, "L", prefix, "candidate", ESTIMATES[3]);
        }
        Self::record(
            records,
            "Bk",
            "probe/bucket/fold_none_share_0/all_rows/",
            "candidate",
            ESTIMATES[4],
        );
        Self::record(
            records,
            "sct",
            "probe/sct/partitioned_emit_rows_20/all_rows/",
            "candidate",
            ESTIMATES[5],
        );
        for prefix in [
            "probe/merge/zero_fill_10mib/",
            "probe/merge/tree_only_10mib/",
            "probe/readout/column_128kib/",
            "probe/readout/fold_none/",
            "probe/readout/fold_hot8/",
            "probe/readout/fold_all/",
        ] {
            Self::record(records, "mrg", prefix, "context", ESTIMATES[6]);
        }
        Self::record(
            records,
            "X",
            "probe/arithmetic/mul_x_hot_raw_shift_substitute/",
            "candidate",
            ESTIMATES[7],
        );
        Self::record(
            records,
            "w",
            "probe/arithmetic/word_monomial_hot/",
            "candidate",
            ESTIMATES[8],
        );
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
            minimum_threads: 1,
        });
    }
    for variant in ["product_hot", "reduce_hot", "reduce_hot_control"] {
        cases.push(ProbeCase {
            unit: "arithmetic",
            variant: variant.to_owned(),
            profiles: &[],
            minimum_threads: 1,
        });
    }
    for n in [1, 2, 4, 8, 20, 256, 1024] {
        cases.push(ProbeCase {
            unit: "arithmetic",
            variant: format!("chain_hot/{n}"),
            profiles: &[],
            minimum_threads: 1,
        });
    }
    for variant in ["mul_x_hot_raw_shift_substitute", "word_monomial_hot"] {
        cases.push(ProbeCase {
            unit: "arithmetic",
            variant: variant.to_owned(),
            profiles: &[],
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
    UnitCandidates::print(&records);
    Ok(())
}
