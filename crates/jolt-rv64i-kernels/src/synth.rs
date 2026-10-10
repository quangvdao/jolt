//! Seeded packed traces with eight trace words and four bytecode words.
//! Trace words are Rs1Value, Rs2Value, RdPreValue, RamReadValue, NextPC and Inc;
//! words 6 and 7 are Rs1Value on a clear KeysDiffer flag and Rs2Value on a
//! set flag; bytecode words are Imm, FallThroughPC, PCPlusImm and PC.
//!
//! | Row columns | Contents | Digit columns | Width |
//! |---|---|---|---|
//! | 0–63 | Inc | — | 64 |
//! | 64–138 | five bytecode chunks, 15 indicators each | 0–4 | 4 |
//! | 139–213 | five RAM chunks, 15 indicators each | 5–9 | 4 |
//! | 214–220, 221–227 | low and high position chunks | 10–11 | 3 |
//! | 228, 229, 230 | KeysDiffer, ShouldBranch, JalrLowBit | 18–20 | 0 |
//! | 231–255 | zero padding | — | — |
//!
//! Chunk digit zero is `Some(0)` and has no stored indicator; the other
//! digits set column `start + digit - 1`. An absent flag has a zero column.
//! Columns 12–17 are bytecode selectors Variant (6 bits), ShiftKind (3),
//! AccessKind (4), KeyKind (3), Branch (0), Store (0); they are represented
//! by bytecode metadata rather than columns of the committed row.
//!
//! Local traces visit `max(1, bytecode_rows / 16)` consecutive rows, rotating
//! their start by the seed; all-rows traces visit every row when cycles suffice.
//! The scaled `log_t = log2(bytecode_rows) + 2` rule preserves the reference
//! ratio of four cycles per bytecode row. RAM accesses use 4,096 words, with
//! 90% in a 64-word window. Register writes occur on three quarters of cycles.
//! Uniform digits have zero trace words; columns 0–20 are present, uniform
//! over their bit width and independent of the bytecode row.
//!
//! | Profile | Value-word distribution | Other fields |
//! |---|---|---|
//! | `local`, `all_rows` | Uniform 64-bit words | Locality described above |
//! | `uniform_digits` | Zero trace words, uniform 64-bit Imm | Columns 0–20 are independent uniform digits |
//! | `chained` | Uniform 64-bit words | `all_rows` with predecessor register and store dependencies |
//! | `small_values` | Rs1Value, Rs2Value, RdPreValue, RamReadValue and Imm each choose 0, 8, 16, 32 or 64 bits with equal probability, then a uniform word of that width | Identical to `all_rows` at the same seed |
//!
//! Fixed chunks of 4,096 cycles use separate ChaCha20 streams indexed by the
//! chunk, making generation independent of the rayon pool.

use crate::memory::MAX_ADDRESS_BITS;
use crate::reduction::ColumnMap;
use crate::source::{CycleSource, LaneSource};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::mem::size_of;
use std::ops::Range;
use std::sync::Arc;
use thiserror::Error;

const GENERATION_CHUNK: usize = 4096;
const PACKED_DIGIT_COLUMNS: usize = 21;
const DIGIT_COLUMNS: usize = 29;
const PRESENT: u8 = 0x80;
const WIDTHS: [usize; DIGIT_COLUMNS] = [
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 6, 3, 4, 3, 0, 0, 0, 0, 0, 5, 5, 5, 4, 4, 4, 4, 4,
];

const WORD_BITS: usize = 64;
const INC_WORD: usize = 5;
const INDICATOR_COLUMNS: usize = 12;
const FLAG_FIRST: usize = 18;
const INDICATOR_STARTS: [usize; INDICATOR_COLUMNS] = {
    let mut starts = [0; INDICATOR_COLUMNS];
    let mut start = WORD_BITS;
    let mut column = 0;
    while column < INDICATOR_COLUMNS {
        starts[column] = start;
        start += (1 << WIDTHS[column]) - 1;
        column += 1;
    }
    starts
};
const FLAGS_START: usize =
    INDICATOR_STARTS[INDICATOR_COLUMNS - 1] + (1 << WIDTHS[INDICATOR_COLUMNS - 1]) - 1;

/// Synthetic instruction and locality distributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthProfile {
    Local,
    AllRows,
    UniformDigits,
    SmallValues,
    Chained,
}

impl SynthProfile {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::AllRows => "all_rows",
            Self::UniformDigits => "uniform_digits",
            Self::SmallValues => "small_values",
            Self::Chained => "chained",
        }
    }
}

/// Invalid synthetic trace dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SynthError {
    #[error("cycle exponent {log_t} is outside 1..=32")]
    CycleExponent { log_t: usize },
    #[error("bytecode row count {rows} must be a power of two at most 2^20")]
    BytecodeRows { rows: usize },
    #[error("synthetic table size cannot be represented for exponent {log_t}")]
    TableSize { log_t: usize },
    #[error("synthetic RAM address bits {bits} must be in 1..={supported}")]
    AddressBits { bits: usize, supported: usize },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Cycle {
    words: [u64; 5],
    digits: [u8; DIGIT_COLUMNS],
    bytecode: u32,
    tail: u8,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct BytecodeRow {
    words: [u64; 4],
    selectors: [u8; 6],
}

/// Flat word and packed digit arrays; each digit byte carries a presence bit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticTrace {
    profile: SynthProfile,
    rows: Vec<[u64; 4]>,
    cycles: Vec<Cycle>,
    bytecode: Vec<BytecodeRow>,
}

impl SyntheticTrace {
    pub fn new(
        profile: SynthProfile,
        log_t: usize,
        bytecode_rows: usize,
        seed: u64,
    ) -> Result<Self, SynthError> {
        if !(1..=32).contains(&log_t) {
            return Err(SynthError::CycleExponent { log_t });
        }
        if !bytecode_rows.is_power_of_two() || bytecode_rows > 1 << 20 {
            return Err(SynthError::BytecodeRows {
                rows: bytecode_rows,
            });
        }
        let count = 1_usize
            .checked_shl(log_t as u32)
            .ok_or(SynthError::TableSize { log_t })?;
        if count
            .checked_mul(size_of::<Cycle>() + size_of::<[u64; 4]>())
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(SynthError::TableSize { log_t });
        }
        let mut bytecode = vec![BytecodeRow::default(); bytecode_rows];
        bytecode
            .par_chunks_mut(GENERATION_CHUNK)
            .enumerate()
            .for_each(|(chunk, rows)| {
                let mut rng = seeded_rng(seed);
                rng.set_stream((chunk as u64) | (1_u64 << 63));
                let mut values_rng = (profile == SynthProfile::SmallValues).then(|| {
                    let mut rng = seeded_rng(seed);
                    rng.set_stream((chunk as u64) | (3_u64 << 62));
                    rng
                });
                for (offset, row) in rows.iter_mut().enumerate() {
                    let index = chunk * GENERATION_CHUNK + offset;
                    let class = rng.next_u32() % 100;
                    let variant = match class {
                        0..=24 => 28 + rng.next_u32() % 7,
                        25..=34 => 35 + rng.next_u32() % 4,
                        35..=39 => 16 + rng.next_u32() % 12,
                        40..=47 => 12 + rng.next_u32() % 4,
                        48..=59 => 39 + rng.next_u32() % 6,
                        _ => rng.next_u32() % 12,
                    } as u8;
                    let shift = if (16..28).contains(&variant) {
                        PRESENT | ((variant - 16) % 3 + if variant >= 22 { 3 } else { 0 })
                    } else {
                        0
                    };
                    let access = if (28..39).contains(&variant) {
                        PRESENT | (variant - 28)
                    } else {
                        0
                    };
                    let key = match variant {
                        12..=15 => PRESENT | (variant - 12),
                        39..=40 => PRESENT | 4,
                        41..=42 => PRESENT,
                        43..=44 => PRESENT | 2,
                        _ => 0,
                    };
                    let branch = if (39..45).contains(&variant) {
                        PRESENT
                    } else {
                        0
                    };
                    let store = if (35..39).contains(&variant) {
                        PRESENT
                    } else {
                        0
                    };
                    row.selectors = [PRESENT | variant, shift, access, key, branch, store];
                    let pc = (index as u64) * 4;
                    row.words = [rng.next_u64(), pc + 4, pc.wrapping_add(rng.next_u64()), pc];
                    if let Some(rng) = &mut values_rng {
                        row.words[0] = small_value(rng);
                    }
                }
            });
        if profile == SynthProfile::Chained {
            let mut previous_rd = (bytecode[bytecode_rows - 1].words[0] >> 10) & 31;
            for row in &mut bytecode {
                let rd = (row.words[0] >> 10) & 31;
                row.words[0] = (row.words[0] & !31) | previous_rd;
                previous_rd = rd;
            }
        }
        let mut rows = vec![[0; 4]; count];
        let mut cycles = vec![Cycle::default(); count];
        let visited = match profile {
            SynthProfile::Local => (bytecode_rows / 16).max(1),
            SynthProfile::AllRows
            | SynthProfile::UniformDigits
            | SynthProfile::SmallValues
            | SynthProfile::Chained => bytecode_rows,
        };
        let row_offset = seed as usize & (bytecode_rows - 1);
        cycles
            .par_chunks_mut(GENERATION_CHUNK)
            .zip(rows.par_chunks_mut(GENERATION_CHUNK))
            .enumerate()
            .for_each(|(chunk, (cycles, rows))| {
                let mut rng = seeded_rng(seed);
                rng.set_stream(chunk as u64);
                let mut values_rng = (profile == SynthProfile::SmallValues).then(|| {
                    let mut rng = seeded_rng(seed);
                    rng.set_stream((chunk as u64) | (1_u64 << 62));
                    rng
                });
                for (offset, (cycle, row)) in cycles.iter_mut().zip(rows).enumerate() {
                    let j = chunk * GENERATION_CHUNK + offset;
                    let k = (row_offset + j % visited) & (bytecode_rows - 1);
                    cycle.bytecode = k as u32;
                    if profile == SynthProfile::UniformDigits {
                        for (column, digit) in
                            cycle.digits[..PACKED_DIGIT_COLUMNS].iter_mut().enumerate()
                        {
                            *digit =
                                PRESENT | (rng.next_u32() as u8 & ((1_u8 << WIDTHS[column]) - 1));
                        }
                    } else {
                        let code = &bytecode[k];
                        cycle.digits[12..18].copy_from_slice(&code.selectors);
                        for (c, digit) in cycle.digits[..5].iter_mut().enumerate() {
                            *digit = PRESENT | ((k >> (4 * c)) & 15) as u8;
                        }
                        let ram = if rng.next_u32().is_multiple_of(10) {
                            rng.next_u32() & 4095
                        } else {
                            rng.next_u32() & 63
                        };
                        for (c, digit) in cycle.digits[5..10].iter_mut().enumerate() {
                            *digit = PRESENT | ((ram >> (4 * c)) & 15) as u8;
                        }
                        cycle.digits[10] = PRESENT | (rng.next_u32() as u8 & 7);
                        cycle.digits[11] = PRESENT | (rng.next_u32() as u8 & 7);
                        cycle.digits[18] = if rng.next_u32() & 1 == 0 { 0 } else { PRESENT };
                        cycle.digits[19] = if cycle.digits[16] != 0 && rng.next_u32() & 1 != 0 {
                            PRESENT
                        } else {
                            0
                        };
                        cycle.digits[20] = if rng.next_u32() & 1 == 0 { 0 } else { PRESENT };
                        for word in &mut cycle.words[..4] {
                            *word = rng.next_u64();
                        }
                        if let Some(rng) = &mut values_rng {
                            for word in &mut cycle.words[..4] {
                                *word = small_value(rng);
                            }
                        }
                        cycle.words[4] = bytecode
                            [(row_offset + (j + 1) % visited) & (bytecode_rows - 1)]
                            .words[3];
                        if rng.next_u32() & 3 != 0 {
                            row[0] = rng.next_u64();
                        }
                        let a = rng.next_u32() as u8 & 3;
                        let b = rng.next_u32() as u8 & 3;
                        cycle.tail = a | (b << 2) | ((a & b) << 4);
                    }
                    for (column, &packed) in cycle.digits[..INDICATOR_COLUMNS].iter().enumerate() {
                        if profile != SynthProfile::Chained || !(5..10).contains(&column) {
                            write_indicator(row, column, packed);
                        }
                    }
                    for column in FLAG_FIRST..PACKED_DIGIT_COLUMNS {
                        if cycle.digits[column] != 0 {
                            let bit = FLAGS_START + column - FLAG_FIRST;
                            row[bit / WORD_BITS] |= 1 << (bit % WORD_BITS);
                        }
                    }
                }
            });
        if profile == SynthProfile::Chained {
            let mut previous_store: Option<[u8; 5]> = None;
            for cycle in &mut cycles {
                if let Some(chunks) = previous_store {
                    cycle.digits[5..10].copy_from_slice(&chunks);
                }
                previous_store = (cycle.digits[17] != 0).then(|| {
                    let mut chunks = [0; 5];
                    chunks.copy_from_slice(&cycle.digits[5..10]);
                    chunks
                });
            }
            rows.par_iter_mut().zip(&cycles).for_each(|(row, cycle)| {
                for column in 5..10 {
                    write_indicator(row, column, cycle.digits[column]);
                }
            });
        }
        cycles.par_iter_mut().for_each(|cycle| {
            let word = bytecode[cycle.bytecode as usize].words[0];
            for (register, digit) in cycle.digits[21..24].iter_mut().enumerate() {
                *digit = PRESENT | ((word >> (5 * register)) & 31) as u8;
            }
            for chunk in 0..5 {
                cycle.digits[24 + chunk] = if cycle.digits[14] != 0 {
                    cycle.digits[5 + chunk]
                } else {
                    PRESENT
                };
            }
        });
        Ok(Self {
            profile,
            rows,
            cycles,
            bytecode,
        })
    }

    /// Maps the encoded Inc word, nonzero digit indicators and flag presence
    /// to their packed-row columns, for every synthetic profile. Padding and
    /// bytecode-only selectors have no entry; the row encoder uses this layout.
    pub fn column_map() -> Vec<ColumnMap> {
        let mut map = vec![ColumnMap::Word {
            start: 0,
            trace_word: INC_WORD,
        }];
        map.extend(
            INDICATOR_STARTS
                .iter()
                .enumerate()
                .map(|(column, &start)| ColumnMap::Indicators { start, column }),
        );
        map.push(ColumnMap::Flags {
            start: FLAGS_START,
            columns: (FLAG_FIRST..PACKED_DIGIT_COLUMNS).collect(),
        });
        map
    }

    /// First packed-row bit for a column's nonzero indicators, whose lengths
    /// are `2^bits(column) - 1`. Returns `None` for flags, selectors and invalid
    /// columns; digit zero has no stored indicator.
    pub fn indicator_start(column: usize) -> Option<usize> {
        INDICATOR_STARTS.get(column).copied()
    }

    /// RAM chunks low first, rs1/rs2/rd, and the optional Store flag.
    pub const fn memory_columns() -> ([usize; 5], [usize; 3], usize) {
        ([24, 25, 26, 27, 28], [21, 22, 23], 17)
    }

    /// Key flag, gated key words, and power columns for a 32-row block.
    pub const fn packed_columns() -> (usize, [usize; 2], [usize; 12]) {
        (18, [6, 7], [0, 1, 2, 3, 4, 24, 25, 26, 27, 28, 10, 11])
    }

    pub fn rows(&self) -> &[[u64; 4]] {
        &self.rows
    }
    pub fn profile(&self) -> SynthProfile {
        self.profile
    }
}

impl LaneSource for SyntheticTrace {
    fn cycles(&self) -> usize {
        self.cycles.len()
    }
    #[inline]
    fn lanes(&self, cycle: usize) -> [[u64; 3]; 2] {
        self.cycles.get(cycle).map_or([[0; 3]; 2], |c| {
            [
                [c.words[0], c.words[1], c.words[0] & c.words[1]],
                [c.words[2], c.words[3], c.words[2] & c.words[3]],
            ]
        })
    }
    #[inline]
    fn tail(&self, cycle: usize) -> u8 {
        self.cycles.get(cycle).map_or(0, |c| c.tail)
    }
}

impl CycleSource for SyntheticTrace {
    fn cycles(&self) -> usize {
        self.cycles.len()
    }
    fn trace_words(&self) -> usize {
        8
    }
    #[inline]
    fn trace_word(&self, word: usize, cycle: usize) -> u64 {
        if word == INC_WORD {
            self.rows.get(cycle).map_or(0, |row| row[0])
        } else if word == 6 || word == 7 {
            self.cycles.get(cycle).map_or(0, |c| {
                let keys_differ = c.digits[18] != 0;
                if keys_differ == (word == 7) {
                    c.words[word - 6]
                } else {
                    0
                }
            })
        } else {
            self.cycles
                .get(cycle)
                .and_then(|c| c.words.get(word))
                .copied()
                .unwrap_or(0)
        }
    }
    fn bytecode_rows(&self) -> usize {
        self.bytecode.len()
    }
    fn bytecode_words(&self) -> usize {
        4
    }
    #[inline]
    fn bytecode_word(&self, word: usize, row: usize) -> u64 {
        self.bytecode
            .get(row)
            .and_then(|r| r.words.get(word))
            .copied()
            .unwrap_or(0)
    }
    #[inline]
    fn bytecode_index(&self, cycle: usize) -> usize {
        self.cycles.get(cycle).map_or(0, |c| c.bytecode as usize)
    }
    fn digit_columns(&self) -> usize {
        DIGIT_COLUMNS
    }
    #[inline]
    fn bits(&self, column: usize) -> usize {
        WIDTHS.get(column).copied().unwrap_or(0)
    }
    #[inline]
    fn by_row(&self, column: usize) -> bool {
        (21..24).contains(&column)
            || (self.profile != SynthProfile::UniformDigits
                && (column < 5 || (12..18).contains(&column)))
    }
    #[inline]
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        self.cycles
            .get(cycle)
            .and_then(|row| row.digits.get(column))
            .and_then(|&byte| {
                let encoded = encoded_digit(byte);
                (encoded != 0).then(|| usize::from(encoded - 1))
            })
    }
    fn digits(&self, cycles: Range<usize>, out: &mut [u16]) {
        if cycles.len().checked_mul(DIGIT_COLUMNS) != Some(out.len()) {
            out.fill(0);
            return;
        }
        for (cycle, output) in cycles.zip(out.chunks_exact_mut(DIGIT_COLUMNS)) {
            if let Some(row) = self.cycles.get(cycle) {
                for (slot, &byte) in output.iter_mut().zip(&row.digits) {
                    *slot = encoded_digit(byte);
                }
            } else {
                output.fill(0);
            }
        }
    }
    #[inline]
    fn row_digit(&self, column: usize, row: usize) -> Option<usize> {
        if !self.by_row(column) || row >= self.bytecode.len() {
            return None;
        }
        if column < 5 {
            Some((row >> (4 * column)) & 15)
        } else if (21..24).contains(&column) {
            Some(((self.bytecode[row].words[0] >> (5 * (column - 21))) & 31) as usize)
        } else {
            self.bytecode[row]
                .selectors
                .get(column - 12)
                .and_then(|&d| {
                    let encoded = encoded_digit(d);
                    (encoded != 0).then(|| usize::from(encoded - 1))
                })
        }
    }
}

fn write_indicator(row: &mut [u64; 4], column: usize, packed: u8) {
    let digit = usize::from(packed & !PRESENT);
    if digit != 0 {
        let bit = INDICATOR_STARTS[column] + digit - 1;
        row[bit / WORD_BITS] |= 1 << (bit % WORD_BITS);
    }
}

#[inline]
fn encoded_digit(byte: u8) -> u16 {
    let value = u16::from(byte & !PRESENT) + 1;
    value & 0_u16.wrapping_sub(u16::from(byte >> 7))
}

fn seeded_rng(seed: u64) -> ChaCha20Rng {
    let mut bytes = [0; 32];
    bytes[..8].copy_from_slice(&seed.to_le_bytes());
    ChaCha20Rng::from_seed(bytes)
}

fn small_value(rng: &mut ChaCha20Rng) -> u64 {
    let widths = [0, 8, 16, 32, 64];
    let choice = loop {
        let choice = rng.next_u32();
        if choice != u32::MAX {
            break choice as usize % widths.len();
        }
    };
    let width = widths[choice];
    rng.next_u64() & u64::MAX.checked_shr(64 - width).unwrap_or(0)
}

/// Seeded nonzero initial RAM and its store replay for the synthetic trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticMemory {
    pub initial: Vec<(u64, u64)>,
    pub final_words: Arc<Vec<u64>>,
    pub mask: Range<usize>,
    pub io: Vec<u64>,
}

impl SyntheticMemory {
    /// Replays normalized RAM columns 24–28 modulo `2^a`, optional Store
    /// column 17 and Inc word 5. The caller supplies normalized RAM digits
    /// whose presence and ranges `ValidatedTrace::prepare` has checked.
    /// The mask covers the first `min(2^a, 4096)` final words.
    pub fn new<S: CycleSource>(trace: &S, a: usize, seed: u64) -> Result<Self, SynthError> {
        if !(1..=MAX_ADDRESS_BITS).contains(&a) {
            return Err(SynthError::AddressBits {
                bits: a,
                supported: MAX_ADDRESS_BITS,
            });
        }
        let words = 1_usize
            .checked_shl(a as u32)
            .ok_or(SynthError::TableSize { log_t: a })?;
        if words
            .checked_mul(size_of::<(u64, u64)>())
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(SynthError::TableSize { log_t: a });
        }
        let (ram, _, store) = SyntheticTrace::memory_columns();
        let mut shifts = [0; 5];
        let mut bits = 0;
        for (chunk, &column) in ram.iter().enumerate() {
            shifts[chunk] = bits;
            let width = trace.bits(column);
            if width > MAX_ADDRESS_BITS - bits {
                return Err(SynthError::AddressBits {
                    bits: bits.saturating_add(width),
                    supported: MAX_ADDRESS_BITS,
                });
            }
            bits += width;
        }
        let mut rng = seeded_rng(seed);
        let initial: Vec<_> = (0..words)
            .map(|index| {
                let word = loop {
                    let word = rng.next_u64();
                    if word != 0 {
                        break word;
                    }
                };
                (index as u64, word)
            })
            .collect();
        let mut final_words: Vec<_> = initial.iter().map(|&(_, word)| word).collect();
        for j in 0..trace.cycles() {
            if trace.digit(store, j).is_some() {
                let address = ram.iter().enumerate().fold(0, |address, (chunk, &column)| {
                    address | (trace.digit(column, j).unwrap_or(0) << shifts[chunk])
                }) & (words - 1);
                final_words[address] ^= trace.trace_word(INC_WORD, j);
            }
        }
        let mask = 0..words.min(1 << 12);
        let io = final_words[mask.clone()].to_vec();
        Ok(Self {
            initial,
            final_words: Arc::new(final_words),
            mask,
            io,
        })
    }
}
