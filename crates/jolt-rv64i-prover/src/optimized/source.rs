//! Packed witness access for the kernels' validation and digit passes.

use crate::error::Rv64iProverError;
use crate::plane::{CycleWords, DecodedCycle, DigitField, DigitFields, Rv64iWitness};
#[cfg(feature = "allocative")]
use allocative::Allocative;
use jolt_rv64i_arith::{Bytecode, BytecodeRow, Chunk, Layout, Variant};
use jolt_rv64i_kernels::reduction::ColumnMap;
use jolt_rv64i_kernels::source::CycleSource;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
enum DigitRead {
    Field(DigitField),
    Flag(DigitField),
    Kind(KindTable),
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
enum KindTable {
    Shift,
    Access,
    Key,
    Branch,
}
impl KindTable {
    const ALL: [Self; 4] = [Self::Shift, Self::Access, Self::Key, Self::Branch];

    fn bits(self) -> usize {
        match self {
            Self::Shift | Self::Key => 3,
            Self::Access => 4,
            Self::Branch => 0,
        }
    }

    fn digit(self, variant: Variant) -> Option<u8> {
        match self {
            Self::Shift => variant.shift().map(|shift| shift.kind.index() as u8),
            Self::Access => variant
                .access()
                .and_then(|access| access.kind)
                .map(|kind| kind.index() as u8),
            Self::Key => variant.key_kind().map(|kind| kind.index() as u8),
            Self::Branch => variant.branch().map(|_| 0),
        }
    }
}

/// The sole numbering of digit columns, trace words and bytecode words.
/// Committed correspondence is derived from `Layout`; indices returned by the
/// named accessors are also the indices accepted by `WitnessSource`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct WitnessColumns {
    bytecode: Vec<usize>,
    ram: Vec<usize>,
    pos: [usize; 2],
    variant: usize,
    shift_kind: usize,
    access_kind: usize,
    key_kind: usize,
    branch: usize,
    keys_differ: usize,
    should_branch: usize,
    jalr_low_bit: usize,
    reads: Vec<DigitRead>,
    widths: Vec<usize>,
    committed: [Option<(usize, usize)>; 256],
    #[cfg_attr(feature = "allocative", allocative(skip))]
    map: Vec<ColumnMap>,
}
impl WitnessColumns {
    const RS1_VALUE: usize = 0;
    const RS2_VALUE: usize = 1;
    const RD_PRE_VALUE: usize = 2;
    const RAM_READ_VALUE: usize = 3;
    const NEXT_PC: usize = 4;
    const INC_WORD: usize = 5;
    const IMM: usize = 0;
    const FALL_THROUGH_PC: usize = 1;
    const PC_PLUS_IMM: usize = 2;
    const PC: usize = 3;

    pub fn new(layout: &Layout) -> Self {
        let fields = DigitFields::new(layout);
        let mut reads = Vec::new();
        let mut widths = Vec::new();
        let mut committed = [None; 256];
        let mut map = vec![ColumnMap::Word {
            start: 0,
            trace_word: Self::inc_word(),
        }];
        let mut add = |chunk: Chunk, field: DigitField| {
            let column = reads.len();
            reads.push(DigitRead::Field(field));
            widths.push(field.bits());
            let start = usize::from(chunk.start());
            map.push(ColumnMap::Indicators { start, column });
            for value in 1..=chunk.indicators() {
                committed[start + value - 1] = Some((column, value));
            }
            column
        };
        let bytecode = layout
            .bytecode_ra()
            .iter()
            .copied()
            .zip(fields.bytecode_fields().iter().copied())
            .map(|(chunk, field)| add(chunk, field))
            .collect();
        let ram = layout
            .ram_ra()
            .iter()
            .copied()
            .zip(fields.ram_fields().iter().copied())
            .map(|(chunk, field)| add(chunk, field))
            .collect();
        let pos_fields = fields.pos_fields();
        let chunks = layout.pos_ra();
        let pos = [add(chunks[0], pos_fields[0]), add(chunks[1], pos_fields[1])];
        let variant = reads.len();
        reads.push(DigitRead::Field(fields.variant()));
        widths.push(fields.variant().bits());
        let shift_kind = reads.len();
        let access_kind = shift_kind + 1;
        let key_kind = shift_kind + 2;
        let branch = shift_kind + 3;
        for kind in KindTable::ALL {
            reads.push(DigitRead::Kind(kind));
            widths.push(kind.bits());
        }
        let keys_differ = reads.len();
        let should_branch = keys_differ + 1;
        let jalr_low_bit = keys_differ + 2;
        for (field, start) in [
            (fields.keys_differ(), layout.keys_differ()),
            (fields.should_branch(), layout.should_branch()),
            (fields.jalr_low_bit(), layout.jalr_low_bit()),
        ] {
            let column = reads.len();
            reads.push(DigitRead::Flag(field));
            widths.push(0);
            committed[start] = Some((column, 0));
        }
        map.push(ColumnMap::Flags {
            start: layout.keys_differ(),
            columns: vec![keys_differ, should_branch, jalr_low_bit],
        });
        Self {
            bytecode,
            ram,
            pos,
            variant,
            shift_kind,
            access_kind,
            key_kind,
            branch,
            keys_differ,
            should_branch,
            jalr_low_bit,
            reads,
            widths,
            committed,
            map,
        }
    }
    pub fn bytecode_chunk(&self, chunk: usize) -> Option<usize> {
        self.bytecode.get(chunk).copied()
    }
    pub fn ram_chunk(&self, chunk: usize) -> Option<usize> {
        self.ram.get(chunk).copied()
    }
    pub fn bytecode_chunks(&self) -> &[usize] {
        &self.bytecode
    }
    pub fn ram_chunks(&self) -> &[usize] {
        &self.ram
    }
    pub fn pos(&self, digit: usize) -> Option<usize> {
        self.pos.get(digit).copied()
    }
    pub fn variant(&self) -> usize {
        self.variant
    }
    pub fn shift_kind(&self) -> usize {
        self.shift_kind
    }
    pub fn access_kind(&self) -> usize {
        self.access_kind
    }
    pub fn key_kind(&self) -> usize {
        self.key_kind
    }
    pub fn branch(&self) -> usize {
        self.branch
    }
    pub fn keys_differ(&self) -> usize {
        self.keys_differ
    }
    pub fn should_branch(&self) -> usize {
        self.should_branch
    }
    pub fn jalr_low_bit(&self) -> usize {
        self.jalr_low_bit
    }
    pub fn committed(&self, column: usize) -> Option<(usize, usize)> {
        self.committed.get(column).copied().flatten()
    }
    pub fn column_map(&self) -> &[ColumnMap] {
        &self.map
    }
    #[inline]
    fn trace_value(word: usize, row: &CycleWords) -> u64 {
        match word {
            Self::RS1_VALUE => row.rs1_value,
            Self::RS2_VALUE => row.rs2_value,
            Self::RD_PRE_VALUE => row.rd_pre_value,
            Self::RAM_READ_VALUE => row.ram_read_value,
            Self::NEXT_PC => row.next_pc,
            _ => 0,
        }
    }
    #[inline]
    fn bytecode_value(word: usize, row: &BytecodeRow) -> u64 {
        match word {
            Self::IMM => row.imm,
            Self::FALL_THROUGH_PC => row.fall_through_pc,
            Self::PC_PLUS_IMM => row.pc_plus_imm,
            Self::PC => row.pc,
            _ => 0,
        }
    }
    pub const fn rs1_value() -> usize {
        Self::RS1_VALUE
    }
    pub const fn rs2_value() -> usize {
        Self::RS2_VALUE
    }
    pub const fn rd_pre_value() -> usize {
        Self::RD_PRE_VALUE
    }
    pub const fn ram_read_value() -> usize {
        Self::RAM_READ_VALUE
    }
    pub const fn next_pc() -> usize {
        Self::NEXT_PC
    }
    pub const fn inc_word() -> usize {
        Self::INC_WORD
    }
    pub const fn imm() -> usize {
        Self::IMM
    }
    pub const fn fall_through_pc() -> usize {
        Self::FALL_THROUGH_PC
    }
    pub const fn pc_plus_imm() -> usize {
        Self::PC_PLUS_IMM
    }
    pub const fn pc() -> usize {
        Self::PC
    }
}

/// Shared replay buffers and small lookup tables; construction copies no cycles.
/// Word, digit, cycle and row indices outside the source return zero or absence.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct WitnessSource {
    words: Arc<[CycleWords]>,
    decoded: Arc<[DecodedCycle]>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    bytecode: Arc<Bytecode>,
    fields: DigitFields,
    columns: WitnessColumns,
    kinds: [[Option<u8>; 64]; KindTable::ALL.len()],
}
impl WitnessSource {
    pub fn new(witness: &Rv64iWitness) -> Result<Self, Rv64iProverError> {
        if witness.decoded.len() != witness.bits.len() {
            return Err(Rv64iProverError::DecodedLength {
                expected: witness.bits.len(),
                found: witness.decoded.len(),
            });
        }
        if witness.words.len() != witness.bits.len() {
            return Err(Rv64iProverError::RowCount {
                expected: witness.bits.len(),
                bits: witness.bits.len(),
                words: witness.words.len(),
            });
        }
        let mut kinds = [[None; 64]; KindTable::ALL.len()];
        for kind in KindTable::ALL {
            for variant in Variant::ALL {
                kinds[kind as usize][variant.index()] = kind.digit(variant);
            }
        }
        Ok(Self {
            words: Arc::clone(&witness.words),
            decoded: Arc::clone(&witness.decoded),
            bytecode: Arc::clone(&witness.bytecode),
            fields: DigitFields::new(&witness.layout),
            columns: WitnessColumns::new(&witness.layout),
            kinds,
        })
    }
    pub fn columns(&self) -> &WitnessColumns {
        &self.columns
    }
}
impl CycleSource for WitnessSource {
    fn cycles(&self) -> usize {
        self.decoded.len()
    }
    fn trace_words(&self) -> usize {
        WitnessColumns::inc_word() + 1
    }
    #[inline]
    fn trace_word(&self, word: usize, cycle: usize) -> u64 {
        if word == WitnessColumns::inc_word() {
            return self.decoded.get(cycle).map_or(0, |row| row.inc);
        }
        self.words
            .get(cycle)
            .map_or(0, |row| WitnessColumns::trace_value(word, row))
    }
    fn bytecode_rows(&self) -> usize {
        self.bytecode.rows().len()
    }
    fn bytecode_words(&self) -> usize {
        WitnessColumns::pc() + 1
    }
    #[inline]
    fn bytecode_word(&self, word: usize, row: usize) -> u64 {
        self.bytecode
            .rows()
            .get(row)
            .map_or(0, |row| WitnessColumns::bytecode_value(word, row))
    }
    #[inline]
    fn bytecode_index(&self, cycle: usize) -> usize {
        self.decoded
            .get(cycle)
            .map_or(0, |row| self.fields.bytecode_index().read(row) as usize)
    }
    fn digit_columns(&self) -> usize {
        self.columns.reads.len()
    }
    #[inline]
    fn bits(&self, column: usize) -> usize {
        self.columns.widths.get(column).copied().unwrap_or(0)
    }
    #[inline]
    fn by_row(&self, column: usize) -> bool {
        column == self.columns.variant()
    }
    #[inline]
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        let row = self.decoded.get(cycle)?;
        match *self.columns.reads.get(column)? {
            DigitRead::Field(field) => Some(field.read(row) as usize),
            DigitRead::Flag(field) => (field.read(row) != 0).then_some(0),
            DigitRead::Kind(kind) => self
                .kinds
                .get(kind as usize)?
                .get(self.fields.variant().read(row) as usize)
                .copied()
                .flatten()
                .map(usize::from),
        }
    }
    #[inline]
    fn row_digit(&self, column: usize, row: usize) -> Option<usize> {
        if column != self.columns.variant() {
            return None;
        }
        self.bytecode
            .rows()
            .get(row)?
            .variant
            .map(|variant| variant.index())
    }
}
