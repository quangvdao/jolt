//! The committed row uses `64 + n(log_K_bytecode) + n(log_K_ram) + 14 + 3`
//! columns, where `n(a) = 15 floor(a/4) + 2^(a mod 4) - 1`.

#![expect(
    non_snake_case,
    reason = "layout sizing names follow protocol notation"
)]

use thiserror::Error;

/// Number of committed columns per cycle.
pub const BITS_COLUMNS: usize = 256;
/// Column `c` is bit `c mod 64` of word `floor(c/64)`.
pub type BitsRow = [u64; 4];

/// Indicator count `n(a) = 15 floor(a/4) + 2^(a mod 4) - 1`.
/// Saturates at `usize::MAX` if the mathematical count is unrepresentable.
#[inline]
pub const fn chunk_indicators(index_bits: usize) -> usize {
    (index_bits / 4)
        .saturating_mul(15)
        .saturating_add((1 << (index_bits % 4)) - 1)
}

/// An invalid chunk descriptor, digit, or digit-bit position.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ChunkError {
    #[error("chunk digit width {bits} is outside 1..=4")]
    WidthOutOfRange { bits: u8 },
    #[error("chunk at column {start} with digit width {bits} exceeds 256 columns")]
    ColumnRangeOutOfRange { start: u16, bits: u8 },
    #[error("chunk digit {digit} does not fit digit width {bits}")]
    DigitOutOfRange { digit: u8, bits: u8 },
    #[error("chunk digit-bit position {bit} is outside digit width {bits}")]
    DigitBitOutOfRange { bit: u8, bits: u8 },
}

/// A digit of width `bits` stores indicators for digits `1..2^bits` beginning
/// at `start`; digit zero is represented by their complemented parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    start: u16,
    bits: u8,
}

impl Chunk {
    /// Checks `1 <= bits <= 4` and `start + 2^bits - 1 <= 256`.
    pub const fn new(start: u16, bits: u8) -> Result<Self, ChunkError> {
        if bits == 0 || bits > 4 {
            return Err(ChunkError::WidthOutOfRange { bits });
        }
        if start as usize + ((1_usize << bits) - 1) > BITS_COLUMNS {
            return Err(ChunkError::ColumnRangeOutOfRange { start, bits });
        }
        Ok(Self::constant(start, bits))
    }
    pub(crate) const fn constant(start: u16, bits: u8) -> Self {
        Self { start, bits }
    }
    /// First stored indicator's column.
    #[inline]
    pub const fn start(self) -> u16 {
        self.start
    }
    /// Digit width in `1..=4`.
    #[inline]
    pub const fn bits(self) -> u8 {
        self.bits
    }
    /// Number of stored indicators, `2^bits - 1`.
    #[inline]
    pub const fn indicators(self) -> usize {
        (1_usize << self.bits) - 1
    }
    #[inline]
    fn mask(self, i: u8) -> u16 {
        const MASKS: [u16; 4] = [0x5555, 0x6666, 0x7878, 0x7f80];
        MASKS.get(usize::from(i)).copied().unwrap_or(0) & ((1_u16 << self.indicators()) - 1)
    }
    /// Bit `k-1` is set exactly when nonzero digit `k` has digit bit `i` set.
    #[inline]
    pub fn digit_bit_mask(self, i: u8) -> Result<u16, ChunkError> {
        if i >= self.bits {
            return Err(ChunkError::DigitBitOutOfRange {
                bit: i,
                bits: self.bits,
            });
        }
        Ok(self.mask(i))
    }
    /// Stored indicator `k` appears at bit `k-1` of the result.
    #[inline]
    pub fn stored(self, row: &BitsRow) -> u16 {
        let start = usize::from(self.start);
        let offset = start % 64;
        let low = row.get(start / 64).copied().unwrap_or(0) >> offset;
        let high = if offset + self.indicators() > 64 {
            row.get(start / 64 + 1).copied().unwrap_or(0) << (64 - offset)
        } else {
            0
        };
        ((low | high) & ((1_u64 << self.indicators()) - 1)) as u16
    }
    /// All indicators, `(stored << 1) | (1 XOR parity(stored))`.
    #[inline]
    pub fn full(self, row: &BitsRow) -> u16 {
        let stored = self.stored(row);
        (stored << 1) | (1 ^ (stored.count_ones() as u16 & 1))
    }
    /// Linear digit bit `parity(stored AND digit_bit_mask(i))`.
    #[inline]
    pub fn digit_bit(self, row: &BitsRow, i: u8) -> Result<bool, ChunkError> {
        Ok((self.stored(row) & self.digit_bit_mask(i)?).count_ones() & 1 != 0)
    }
    /// Clears this chunk and writes the indicator for `digit` (none for zero).
    /// An out-of-range digit leaves the row unchanged.
    #[inline]
    pub fn write_digit(self, row: &mut BitsRow, digit: u8) -> Result<(), ChunkError> {
        if usize::from(digit) > self.indicators() {
            return Err(ChunkError::DigitOutOfRange {
                digit,
                bits: self.bits,
            });
        }
        self.write_validated_digit(row, digit);
        Ok(())
    }
    #[inline]
    fn write_validated_digit(self, row: &mut BitsRow, digit: u8) {
        let start = usize::from(self.start);
        let offset = start % 64;
        let mask = (1_u64 << self.indicators()) - 1;
        let hot = if digit == 0 { 0 } else { 1_u64 << (digit - 1) };
        if let Some(word) = row.get_mut(start / 64) {
            *word = (*word & !(mask << offset)) | (hot << offset);
        }
        if offset + self.indicators() > 64 {
            if let Some(word) = row.get_mut(start / 64 + 1) {
                *word = (*word & !(mask >> (64 - offset))) | (hot >> (64 - offset));
            }
        }
    }
}

#[inline]
pub(crate) fn bit(row: &BitsRow, column: usize) -> bool {
    row.get(column / 64)
        .is_some_and(|word| word & (1_u64 << (column % 64)) != 0)
}
#[inline]
pub(crate) fn set_bit(row: &mut BitsRow, column: usize, value: bool) {
    if let Some(word) = row.get_mut(column / 64) {
        let mask = 1_u64 << (column % 64);
        *word = (*word & !mask) | (u64::from(value) << (column % 64));
    }
}

/// A violated size, RAM placement, or committed index bound.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LayoutError {
    #[error("{name} exponent {log_size} is outside its permitted range")]
    LogSizeOutOfRange { name: &'static str, log_size: usize },
    #[error("lowest address {lowest_address} is not a multiple of 8")]
    LowestAddressNotAligned { lowest_address: u64 },
    #[error("RAM at lowest address {lowest_address} with exponent {log_K_ram} exceeds 2^64")]
    RamRangeOverflow {
        lowest_address: u64,
        log_K_ram: usize,
    },
    #[error("committed row needs {used_columns} columns, exceeding 256")]
    BitsRowOverflow { used_columns: usize },
    #[error("{name} index {index} does not fit exponent {log_size}")]
    IndexOutOfRange {
        name: &'static str,
        index: u64,
        log_size: usize,
    },
}

/// Column ownership: `Inc`, bytecode chunks, RAM chunks, two 3-bit `Pos`
/// chunks, `KeysDiffer`, `ShouldBranch`, `JalrLowBit`, then spare columns.
#[derive(Debug, Clone)]
pub struct Layout {
    log_K_bytecode: usize,
    log_K_ram: usize,
    lowest_address: u64,
    bytecode_ra: Vec<Chunk>,
    ram_ra: Vec<Chunk>,
    pos_ra: [Chunk; 2],
    keys_differ: usize,
}
impl Layout {
    /// Checks exponents in `1..=32` and `1..=61`, 8-byte RAM alignment,
    /// `LowestAddress + 8 * 2^log_K_ram <= 2^64`, then row capacity.
    pub fn new(
        log_K_bytecode: usize,
        log_K_ram: usize,
        lowest_address: u64,
    ) -> Result<Self, LayoutError> {
        for (name, log_size, max) in [
            ("log_K_bytecode", log_K_bytecode, 32),
            ("log_K_ram", log_K_ram, 61),
        ] {
            if log_size == 0 || log_size > max {
                return Err(LayoutError::LogSizeOutOfRange { name, log_size });
            }
        }
        if !lowest_address.is_multiple_of(8) {
            return Err(LayoutError::LowestAddressNotAligned { lowest_address });
        }
        if u128::from(lowest_address) + (8_u128 << log_K_ram) > 1_u128 << 64 {
            return Err(LayoutError::RamRangeOverflow {
                lowest_address,
                log_K_ram,
            });
        }
        let used_columns = 81 + chunk_indicators(log_K_bytecode) + chunk_indicators(log_K_ram);
        if used_columns > BITS_COLUMNS {
            return Err(LayoutError::BitsRowOverflow { used_columns });
        }
        let mut start = 64_u16;
        let bytecode_ra = Self::address_chunks(log_K_bytecode, &mut start);
        let ram_ra = Self::address_chunks(log_K_ram, &mut start);
        let pos_ra = [Chunk::constant(start, 3), Chunk::constant(start + 7, 3)];
        Ok(Self {
            log_K_bytecode,
            log_K_ram,
            lowest_address,
            bytecode_ra,
            ram_ra,
            pos_ra,
            keys_differ: usize::from(start) + 14,
        })
    }
    fn address_chunks(log_size: usize, start: &mut u16) -> Vec<Chunk> {
        let mut out = Vec::with_capacity(log_size.div_ceil(4));
        for from in (0..log_size).step_by(4) {
            let chunk = Chunk::constant(*start, (log_size - from).min(4) as u8);
            *start += chunk.indicators() as u16;
            out.push(chunk);
        }
        out
    }
    /// Bytecode address exponent.
    #[inline]
    pub fn log_K_bytecode(&self) -> usize {
        self.log_K_bytecode
    }
    /// RAM word address exponent.
    #[inline]
    pub fn log_K_ram(&self) -> usize {
        self.log_K_ram
    }
    /// Base address subtracted from memory displacements.
    #[inline]
    pub fn lowest_address(&self) -> u64 {
        self.lowest_address
    }
    /// Bytecode address chunks in least-significant order.
    #[inline]
    pub fn bytecode_ra(&self) -> &[Chunk] {
        &self.bytecode_ra
    }
    /// RAM word address chunks in least-significant order.
    #[inline]
    pub fn ram_ra(&self) -> &[Chunk] {
        &self.ram_ra
    }
    /// Low and high 3-bit `Pos` chunks.
    #[inline]
    pub fn pos_ra(&self) -> [Chunk; 2] {
        self.pos_ra
    }
    /// Chunks in row order: bytecode, RAM, low Pos, high Pos.
    #[inline]
    pub fn chunks(&self) -> impl Iterator<Item = Chunk> + '_ {
        self.bytecode_ra
            .iter()
            .chain(&self.ram_ra)
            .chain(&self.pos_ra)
            .copied()
    }
    /// `KeysDiffer` column.
    #[inline]
    pub fn keys_differ(&self) -> usize {
        self.keys_differ
    }
    /// `ShouldBranch` column.
    #[inline]
    pub fn should_branch(&self) -> usize {
        self.keys_differ + 1
    }
    /// `JalrLowBit` column.
    #[inline]
    pub fn jalr_low_bit(&self) -> usize {
        self.keys_differ + 2
    }
    /// Used columns, `81 + n(log_K_bytecode) + n(log_K_ram)`.
    #[inline]
    pub fn used_columns(&self) -> usize {
        self.keys_differ + 3
    }
    /// Increment word at columns `0..64`.
    #[inline]
    pub fn inc(&self, row: &BitsRow) -> u64 {
        let [inc, ..] = *row;
        inc
    }
    #[inline]
    fn index(chunks: &[Chunk], row: &BitsRow) -> u64 {
        let mut out = 0;
        let mut from = 0;
        for chunk in chunks {
            let stored = chunk.stored(row);
            for i in 0..chunk.bits {
                out |= u64::from((stored & chunk.mask(i)).count_ones() & 1 != 0)
                    << (from + usize::from(i));
            }
            from += usize::from(chunk.bits);
        }
        out
    }
    /// Bytecode index reconstructed linearly from digit indicator parities.
    #[inline]
    pub fn bytecode_index(&self, row: &BitsRow) -> u64 {
        Self::index(&self.bytecode_ra, row)
    }
    /// RAM word index reconstructed linearly from digit indicator parities.
    #[inline]
    pub fn ram_index(&self, row: &BitsRow) -> u64 {
        Self::index(&self.ram_ra, row)
    }
    /// Six-bit position reconstructed linearly from the two Pos chunks.
    #[inline]
    pub fn pos(&self, row: &BitsRow) -> u8 {
        Self::index(&self.pos_ra, row) as u8
    }
    #[inline]
    fn write_index(
        chunks: &[Chunk],
        row: &mut BitsRow,
        index: u64,
        name: &'static str,
        log_size: usize,
    ) -> Result<(), LayoutError> {
        if index >= 1_u64 << log_size {
            return Err(LayoutError::IndexOutOfRange {
                name,
                index,
                log_size,
            });
        }
        let mut remaining = index;
        for chunk in chunks {
            let digit = (remaining & chunk.indicators() as u64) as u8;
            chunk.write_validated_digit(row, digit);
            remaining >>= chunk.bits;
        }
        Ok(())
    }
    /// Clears and writes bytecode chunks; rejects an index outside `2^log_K_bytecode`.
    #[inline]
    pub fn write_bytecode_index(&self, row: &mut BitsRow, index: u64) -> Result<(), LayoutError> {
        Self::write_index(
            &self.bytecode_ra,
            row,
            index,
            "bytecode",
            self.log_K_bytecode,
        )
    }
    /// Clears and writes RAM chunks; rejects an index outside `2^log_K_ram`.
    #[inline]
    pub fn write_ram_index(&self, row: &mut BitsRow, index: u64) -> Result<(), LayoutError> {
        Self::write_index(&self.ram_ra, row, index, "RAM", self.log_K_ram)
    }
    /// Clears and writes Pos chunks; rejects a position outside `0..64`.
    #[inline]
    pub fn write_pos(&self, row: &mut BitsRow, pos: u8) -> Result<(), LayoutError> {
        Self::write_index(&self.pos_ra, row, u64::from(pos), "Pos", 6)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test fixtures use checked valid descriptors"
)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_reference_columns() {
        for (bits, expected) in [
            (0, 0),
            (1, 1),
            (2, 3),
            (3, 7),
            (4, 15),
            (20, 75),
            (21, 76),
            (22, 78),
            (23, 82),
            (24, 90),
            (27, 97),
            (28, 105),
        ] {
            assert_eq!(chunk_indicators(bits), expected);
        }
        for (ram, expected) in [(20, 231), (23, 238), (24, 246), (27, 253)] {
            assert_eq!(Layout::new(20, ram, 0).unwrap().used_columns(), expected);
        }
        let layout = Layout::new(20, 20, 0).unwrap();
        assert_eq!(
            layout
                .bytecode_ra()
                .iter()
                .map(|c| (c.start(), c.bits()))
                .collect::<Vec<_>>(),
            [(64, 4), (79, 4), (94, 4), (109, 4), (124, 4)]
        );
        assert_eq!(
            layout
                .ram_ra()
                .iter()
                .map(|c| (c.start(), c.bits()))
                .collect::<Vec<_>>(),
            [(139, 4), (154, 4), (169, 4), (184, 4), (199, 4)]
        );
        assert_eq!(
            layout.pos_ra().map(|c| (c.start(), c.bits())),
            [(214, 3), (221, 3)]
        );
        assert_eq!(
            (
                layout.keys_differ(),
                layout.should_branch(),
                layout.jalr_low_bit()
            ),
            (228, 229, 230)
        );
        assert_eq!(layout.chunks().count(), 12);
        assert_eq!(3 * 64 + 3 * 64 + 2 * 64 + 3 * 64 + 11 + 3, 718);
        assert_eq!(chunk_indicators(20) * 2 + 14 + 1, 165);
    }
    #[test]
    fn layout_checks_and_boundaries() {
        for (bytecode, ram, name, value) in [
            (0, 1, "log_K_bytecode", 0),
            (33, 1, "log_K_bytecode", 33),
            (1, 0, "log_K_ram", 0),
            (1, 62, "log_K_ram", 62),
        ] {
            assert_eq!(
                Layout::new(bytecode, ram, 0).unwrap_err(),
                LayoutError::LogSizeOutOfRange {
                    name,
                    log_size: value
                }
            );
        }
        assert_eq!(
            Layout::new(20, 20, 4).unwrap_err(),
            LayoutError::LowestAddressNotAligned { lowest_address: 4 }
        );
        let lowest_address = u64::MAX - (1 << 22) + 1;
        assert_eq!(
            Layout::new(20, 20, lowest_address).unwrap_err(),
            LayoutError::RamRangeOverflow {
                lowest_address,
                log_K_ram: 20
            }
        );
        assert!(Layout::new(20, 20, u64::MAX - (1 << 23) + 1).is_ok());
        assert_eq!(
            Layout::new(20, 28, 0).unwrap_err(),
            LayoutError::BitsRowOverflow { used_columns: 261 }
        );
    }
    #[test]
    fn checked_chunk_domains() {
        assert_eq!(
            Chunk::new(0, 0),
            Err(ChunkError::WidthOutOfRange { bits: 0 })
        );
        assert_eq!(
            Chunk::new(0, 5),
            Err(ChunkError::WidthOutOfRange { bits: 5 })
        );
        assert_eq!(
            Chunk::new(242, 4),
            Err(ChunkError::ColumnRangeOutOfRange {
                start: 242,
                bits: 4
            })
        );
        for bits in 1..=4 {
            let chunk = Chunk::new(63, bits).unwrap();
            let mut row = [u64::MAX; 4];
            assert_eq!(
                chunk.digit_bit_mask(bits),
                Err(ChunkError::DigitBitOutOfRange { bit: bits, bits })
            );
            assert_eq!(
                chunk.digit_bit(&row, bits),
                Err(ChunkError::DigitBitOutOfRange { bit: bits, bits })
            );
            let before = row;
            assert_eq!(
                chunk.write_digit(&mut row, 1 << bits),
                Err(ChunkError::DigitOutOfRange {
                    digit: 1 << bits,
                    bits
                })
            );
            assert_eq!(row, before);
            for i in 0..bits {
                let expected = (1_u16..(1 << bits))
                    .filter(|digit| digit & (1 << i) != 0)
                    .fold(0_u16, |mask, digit| mask | (1 << (digit - 1)));
                assert_eq!(chunk.digit_bit_mask(i).unwrap(), expected);
            }
            for digit in 0..(1 << bits) {
                chunk.write_digit(&mut row, digit).unwrap();
                assert_eq!(chunk.full(&row), 1 << digit);
                for i in 0..bits {
                    assert_eq!(chunk.digit_bit(&row, i).unwrap(), digit & (1 << i) != 0);
                }
            }
        }
    }
    #[test]
    fn indicator_parity_is_linear_on_every_pattern() {
        for bits in [3, 4] {
            let chunk = Chunk::new(60, bits).unwrap();
            for pattern in 0..(1_u16 << chunk.indicators()) {
                let mut row = [0; 4];
                for digit in 1_u8..(1 << bits) {
                    set_bit(
                        &mut row,
                        60 + usize::from(digit) - 1,
                        pattern & (1 << (digit - 1)) != 0,
                    );
                }
                assert_eq!(chunk.stored(&row), pattern);
                for i in 0..bits {
                    let expected = (1..(1 << bits))
                        .filter(|digit| pattern & (1 << (digit - 1)) != 0)
                        .fold(false, |parity, digit| parity ^ (digit & (1 << i) != 0));
                    assert_eq!(chunk.digit_bit(&row, i).unwrap(), expected);
                }
            }
        }
    }
    #[test]
    fn index_writers_clear_and_preserve_other_columns() {
        let layout = Layout::new(5, 7, 0).unwrap();
        for bytecode in 0..32 {
            for ram in 0..128 {
                let mut row = [u64::MAX; 4];
                layout.write_bytecode_index(&mut row, bytecode).unwrap();
                layout.write_ram_index(&mut row, ram).unwrap();
                for pos in 0..64 {
                    layout.write_pos(&mut row, pos).unwrap();
                    assert_eq!(
                        (
                            layout.bytecode_index(&row),
                            layout.ram_index(&row),
                            layout.pos(&row)
                        ),
                        (bytecode, ram, pos)
                    );
                }
                assert_eq!(layout.inc(&row), u64::MAX);
                assert!(bit(&row, layout.jalr_low_bit()));
            }
        }
        let mut row = [0; 4];
        for result in [
            layout.write_bytecode_index(&mut row, 32),
            layout.write_ram_index(&mut row, 128),
            layout.write_pos(&mut row, 64),
        ] {
            assert!(matches!(result, Err(LayoutError::IndexOutOfRange { .. })));
        }
        assert_eq!(row, [0; 4]);
        for bytecode in 1..=32 {
            for ram in 1..=61 {
                if let Ok(layout) = Layout::new(bytecode, ram, 0) {
                    for chunk in layout.chunks() {
                        assert_eq!(Chunk::new(chunk.start(), chunk.bits()), Ok(chunk));
                    }
                }
            }
        }
    }
}
