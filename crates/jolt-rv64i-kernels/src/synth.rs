//! Seeded packed traces with six trace words and four bytecode words.
//! Trace words are Rs1Value, Rs2Value, RdPreValue, RamReadValue, NextPC and Inc;
//! bytecode words are Imm, FallThroughPC, PCPlusImm and PC.
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
//! Uniform digits have no trace words and every column is independent of the
//! bytecode row; all digits are present and uniform over their bit width.
//!
//! | Profile | Value-word distribution | Other fields |
//! |---|---|---|
//! | `local`, `all_rows` | Uniform 64-bit words | Locality described above |
//! | `uniform_digits` | Zero trace words, uniform 64-bit Imm | Independent uniform digits |
//! | `small_values` | Rs1Value, Rs2Value, RdPreValue, RamReadValue and Imm each choose 0, 8, 16, 32 or 64 bits with equal probability, then a uniform word of that width | Identical to `all_rows` at the same seed |
//!
//! Fixed chunks of 4,096 cycles use separate ChaCha20 streams indexed by the
//! chunk, making generation independent of the rayon pool.

use crate::reduction::ColumnMap;
use crate::source::{CycleSource, LaneSource};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::mem::size_of;
use thiserror::Error;

const GENERATION_CHUNK: usize = 4096;
const DIGIT_COLUMNS: usize = 21;
const PRESENT: u8 = 0x80;
const WIDTHS: [usize; DIGIT_COLUMNS] = [
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 6, 3, 4, 3, 0, 0, 0, 0, 0,
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
}

impl SynthProfile {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::AllRows => "all_rows",
            Self::UniformDigits => "uniform_digits",
            Self::SmallValues => "small_values",
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
        let mut rows = vec![[0; 4]; count];
        let mut cycles = vec![Cycle::default(); count];
        let visited = match profile {
            SynthProfile::Local => (bytecode_rows / 16).max(1),
            SynthProfile::AllRows | SynthProfile::UniformDigits | SynthProfile::SmallValues => {
                bytecode_rows
            }
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
                        for (column, digit) in cycle.digits.iter_mut().enumerate() {
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
                        let digit = usize::from(packed & !PRESENT);
                        if digit != 0 {
                            let start = INDICATOR_STARTS[column];
                            let bit = start + digit - 1;
                            row[bit / 64] |= 1 << (bit % 64);
                        }
                    }
                    for column in FLAG_FIRST..DIGIT_COLUMNS {
                        if cycle.digits[column] != 0 {
                            let bit = FLAGS_START + column - FLAG_FIRST;
                            row[bit / WORD_BITS] |= 1 << (bit % WORD_BITS);
                        }
                    }
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
            columns: (FLAG_FIRST..DIGIT_COLUMNS).collect(),
        });
        map
    }

    /// First packed-row bit for a column's nonzero indicators, whose lengths
    /// are `2^bits(column) - 1`. Returns `None` for flags, selectors and invalid
    /// columns; digit zero has no stored indicator.
    pub fn indicator_start(column: usize) -> Option<usize> {
        INDICATOR_STARTS.get(column).copied()
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
        6
    }
    #[inline]
    fn trace_word(&self, word: usize, cycle: usize) -> u64 {
        if word == INC_WORD {
            self.rows.get(cycle).map_or(0, |row| row[0])
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
        self.profile != SynthProfile::UniformDigits && (column < 5 || (12..18).contains(&column))
    }
    #[inline]
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        self.cycles
            .get(cycle)
            .and_then(|c| c.digits.get(column))
            .and_then(|&d| {
                if d & PRESENT != 0 {
                    Some(usize::from(d & !PRESENT))
                } else {
                    None
                }
            })
    }
    #[inline]
    fn row_digit(&self, column: usize, row: usize) -> Option<usize> {
        if !self.by_row(column) || row >= self.bytecode.len() {
            return None;
        }
        if column < 5 {
            Some((row >> (4 * column)) & 15)
        } else {
            self.bytecode[row]
                .selectors
                .get(column - 12)
                .and_then(|&d| {
                    if d & PRESENT != 0 {
                        Some(usize::from(d & !PRESENT))
                    } else {
                        None
                    }
                })
        }
    }
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
