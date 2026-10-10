//! Packed witness access for the kernels' validation and digit passes.

use crate::error::Rv64iProverError;
use crate::plane::{CycleWords, DecodedCycle, DigitField, DigitFields, Rv64iWitness};
#[cfg(feature = "allocative")]
use allocative::Allocative;
use jolt_field::F128;
use jolt_kernels::KernelError;
use jolt_rv64i_arith::{Bytecode, BytecodeRow, Chunk, Layout, Variant};
use jolt_rv64i_kernels::packed::scatter::ScatterPlan;
use jolt_rv64i_kernels::reduction::ColumnMap;
use jolt_rv64i_kernels::source::{
    CycleSource, OptionalGroup, PrepareRequest, PresentGroup, ValidatedTrace,
};
use std::fmt::Display;
use std::ops::Range;
use std::sync::Arc;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
struct ColumnReads {
    fields: Vec<DigitField>,
    variant: DigitField,
    kinds: [KindTable; 4],
    flags: [FlagRead; 3],
}
impl ColumnReads {
    fn variant_column(&self) -> usize {
        self.fields.len()
    }

    fn kind_start(&self) -> usize {
        self.variant_column() + 1
    }

    fn kind_column(&self, kind: KindTable) -> usize {
        self.kind_start() + kind.index()
    }

    fn flag_start(&self) -> usize {
        self.kind_start() + self.kinds.len()
    }

    fn flag_column(&self, flag: FlagTable) -> usize {
        self.flag_start() + flag.index()
    }

    fn len(&self) -> usize {
        self.flag_start() + self.flags.len()
    }
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
#[repr(usize)]
enum KindTable {
    Shift,
    Access,
    Key,
    Branch,
}
impl KindTable {
    const ALL: [Self; 4] = [Self::Shift, Self::Access, Self::Key, Self::Branch];

    fn index(self) -> usize {
        self as usize
    }

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

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
#[repr(usize)]
enum FlagTable {
    KeysDiffer,
    ShouldBranch,
    JalrLowBit,
}
impl FlagTable {
    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
struct FlagRead {
    flag: FlagTable,
    field: DigitField,
    committed: usize,
}

#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
struct EncodedField {
    shift: usize,
    mask: u64,
    add: u16,
}
impl EncodedField {
    fn new(field: DigitField, add: u16) -> Self {
        Self {
            shift: field.shift(),
            mask: field.mask(),
            add,
        }
    }

    // WitnessColumns::new supplies fields of at most six bits, including malformed rows.
    #[inline]
    fn read(self, digits: u64) -> u16 {
        ((digits >> self.shift) & self.mask) as u16 + self.add
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
struct DigitDecoder {
    fields: Vec<EncodedField>,
    kinds: [[u16; KindTable::ALL.len()]; 64],
    flags: [EncodedField; 3],
    variant: EncodedField,
}
impl DigitDecoder {
    fn new(columns: &WitnessColumns) -> Self {
        let reads = &columns.reads;
        let fields = reads
            .fields
            .iter()
            .copied()
            .map(|field| EncodedField::new(field, 1))
            .collect();
        let mut kinds = [[0; KindTable::ALL.len()]; 64];
        for table in reads.kinds {
            for variant in Variant::ALL {
                kinds[variant.index()][table.index()] =
                    table.digit(variant).map_or(0, |digit| u16::from(digit) + 1);
            }
        }
        let mut flags = [EncodedField::default(); 3];
        for read in reads.flags {
            flags[read.flag.index()] = EncodedField::new(read.field, 0);
        }
        Self {
            variant: EncodedField::new(reads.variant, 1),
            fields,
            kinds,
            flags,
        }
    }

    #[inline]
    fn variant(&self, digits: u64) -> usize {
        // The table's mask also exposes the six-bit index bound to the compiler.
        usize::from(self.variant.read(digits).wrapping_sub(1)) & (self.kinds.len() - 1)
    }

    #[inline]
    fn digit(&self, column: usize, digits: u64) -> Option<usize> {
        let encoded = if let Some(field) = self.fields.get(column) {
            field.read(digits)
        } else if column == self.fields.len() {
            self.variant.read(digits)
        } else {
            let column = column - self.fields.len() - 1;
            if let Some(&kind) = self.kinds[self.variant(digits)].get(column) {
                kind
            } else {
                self.flags.get(column - KindTable::ALL.len())?.read(digits)
            }
        };
        (encoded != 0).then(|| usize::from(encoded - 1))
    }

    #[inline]
    fn cycles<const FIELDS: usize, const COLUMNS: usize>(
        &self,
        cycles: &[DecodedCycle],
        out: &mut [u16],
    ) {
        let Ok(fields): Result<&[EncodedField; FIELDS], _> = self.fields.as_slice().try_into()
        else {
            out.fill(0);
            return;
        };
        for (row, output) in cycles.iter().zip(out.as_chunks_mut::<COLUMNS>().0) {
            let digits = row.digits;
            for (column, slot) in output[..FIELDS].iter_mut().enumerate() {
                *slot = fields[column].read(digits);
            }
            let variant = self.variant.read(digits);
            output[FIELDS] = variant;
            let variant = usize::from(variant.wrapping_sub(1)) & (self.kinds.len() - 1);
            output[FIELDS + 1..FIELDS + 1 + KindTable::ALL.len()]
                .copy_from_slice(&self.kinds[variant]);
            for (slot, field) in output[FIELDS + 1 + KindTable::ALL.len()..]
                .iter_mut()
                .zip(self.flags)
            {
                *slot = field.read(digits);
            }
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
    reads: ColumnReads,
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
        let mut read_fields = Vec::new();
        let mut widths = Vec::new();
        let mut committed = [None; 256];
        let mut map = vec![ColumnMap::Word {
            start: 0,
            trace_word: Self::inc_word(),
        }];
        let mut add = |chunk: Chunk, field: DigitField| {
            let column = read_fields.len();
            read_fields.push(field);
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
        let reads = ColumnReads {
            fields: read_fields,
            variant: fields.variant(),
            kinds: KindTable::ALL,
            flags: [
                FlagRead {
                    flag: FlagTable::KeysDiffer,
                    field: fields.keys_differ(),
                    committed: layout.keys_differ(),
                },
                FlagRead {
                    flag: FlagTable::ShouldBranch,
                    field: fields.should_branch(),
                    committed: layout.should_branch(),
                },
                FlagRead {
                    flag: FlagTable::JalrLowBit,
                    field: fields.jalr_low_bit(),
                    committed: layout.jalr_low_bit(),
                },
            ],
        };
        let variant = reads.variant_column();
        widths.push(reads.variant.bits());
        let shift_kind = reads.kind_column(KindTable::Shift);
        let access_kind = reads.kind_column(KindTable::Access);
        let key_kind = reads.kind_column(KindTable::Key);
        let branch = reads.kind_column(KindTable::Branch);
        widths.resize(reads.len(), 0);
        for table in reads.kinds {
            widths[reads.kind_column(table)] = table.bits();
        }
        let keys_differ = reads.flag_column(FlagTable::KeysDiffer);
        let should_branch = reads.flag_column(FlagTable::ShouldBranch);
        let jalr_low_bit = reads.flag_column(FlagTable::JalrLowBit);
        for read in reads.flags {
            committed[read.committed] = Some((reads.flag_column(read.flag), 0));
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
    pub fn pos_digits(&self) -> usize {
        self.pos.len()
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
    decoder: DigitDecoder,
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
        let columns = WitnessColumns::new(&witness.layout);
        let decoder = DigitDecoder::new(&columns);
        Ok(Self {
            words: Arc::clone(&witness.words),
            decoded: Arc::clone(&witness.decoded),
            bytecode: Arc::clone(&witness.bytecode),
            fields: DigitFields::new(&witness.layout),
            columns,
            decoder,
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
        self.decoder.digit(column, row.digits)
    }
    fn digits(&self, cycles: Range<usize>, out: &mut [u16]) {
        if cycles.len().checked_mul(self.digit_columns()) != Some(out.len()) {
            out.fill(0);
            return;
        }
        let valid_end = cycles.end.min(self.decoded.len());
        if cycles.start >= valid_end {
            out.fill(0);
            return;
        }
        let rows = &self.decoded[cycles.start..valid_end];
        let valid_len = rows.len() * self.digit_columns();
        let (out, absent) = out.split_at_mut(valid_len);
        absent.fill(0);
        // Layout::new admits four through fifteen fields before Variant.
        macro_rules! decode {
            ($($fields:literal),*) => {
                match self.decoder.fields.len() {
                    $($fields => self.decoder.cycles::<$fields, { $fields + 1 + KindTable::ALL.len() + 3 }>(rows, out),)*
                    _ => out.fill(0),
                }
            };
        }
        decode!(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceGroup {
    BytecodeChunks,
    RamChunks,
    Selectors,
}
impl SourceGroup {
    fn name(self) -> &'static str {
        match self {
            Self::BytecodeChunks => "bytecode chunk group",
            Self::RamChunks => "RAM chunk group",
            Self::Selectors => "selector group",
        }
    }
}

#[cfg_attr(feature = "allocative", derive(Allocative), allocative(bound = ""))]
enum OwnedGroup<G> {
    NotRequested,
    Held(#[cfg_attr(feature = "allocative", allocative(skip))] G),
    Taken,
}
impl<G> OwnedGroup<G> {
    fn take(&mut self, group: SourceGroup) -> Result<G, KernelError<F128>> {
        match std::mem::replace(self, Self::Taken) {
            Self::Held(value) => Ok(value),
            Self::NotRequested => {
                *self = Self::NotRequested;
                Err(SharedSource::geometry(format!(
                    "{} was not requested",
                    group.name()
                )))
            }
            Self::Taken => Err(SharedSource::geometry(format!(
                "{} was already taken",
                group.name()
            ))),
        }
    }
}

#[cfg_attr(feature = "allocative", derive(Allocative))]
enum SharedPlan {
    NotBuilt,
    Held(#[cfg_attr(feature = "allocative", allocative(skip))] Arc<ScatterPlan<WitnessSource>>),
    Released,
}

#[cfg_attr(feature = "allocative", derive(Allocative))]
struct PreparedSource {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    trace: Arc<ValidatedTrace<WitnessSource>>,
    bytecode: OwnedGroup<PresentGroup>,
    ram: OwnedGroup<PresentGroup>,
    selectors: OwnedGroup<OptionalGroup>,
    plan: SharedPlan,
}

/// Cross-batch state inserted empty with `ProofSession::state_or_insert_with`.
/// A first caller in batch 3a or 3b passes the selector columns listed by
/// `RoutersCycleCore::columns` for all five shapes, regardless of optimised
/// slots: preparation writes that optional
/// group and each chunk group with at most seven columns. A first caller in
/// batch 6b passes `None`, requesting only the eligible chunk groups, because
/// batches 3a and 3b are behind it. A later router request after that tail-first
/// preparation fails with `InvalidGeometry`; it never adds a second walk.
/// `BytecodeReadCycle` and `RamRaProduct` take their respective present groups;
/// the first optimised `RouterCycle` prepare takes selectors for `RoutersCycleCore`.
/// Untaken groups drop with the session.
/// The session belongs to one witness, and router callers supply the same
/// canonical selector list throughout it. Taken groups and a released plan
/// are never reconstructed.
#[derive(Default)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct SharedSource {
    prepared: Option<PreparedSource>,
}
impl SharedSource {
    fn geometry(reason: impl Display) -> KernelError<F128> {
        KernelError::InvalidGeometry {
            reason: reason.to_string(),
        }
    }

    /// Prepares on a miss and returns the same trace `Arc` on later requests.
    /// `Some` supplies the router selector list; `None` selects a tail request.
    /// Failure leaves the previous state intact.
    pub fn prepare(
        &mut self,
        witness: &Rv64iWitness,
        selectors: Option<Vec<usize>>,
    ) -> Result<Arc<ValidatedTrace<WitnessSource>>, KernelError<F128>> {
        if let Some(prepared) = &self.prepared {
            if selectors.is_some() && matches!(prepared.selectors, OwnedGroup::NotRequested) {
                return Err(Self::geometry("selector group was not requested"));
            }
            return Ok(Arc::clone(&prepared.trace));
        }
        let source = Arc::new(WitnessSource::new(witness).map_err(Self::geometry)?);
        let columns = source.columns();
        let bytecode_requested = columns.bytecode_chunks().len() <= 7;
        let ram_requested = columns.ram_chunks().len() <= 7;
        let mut present = Vec::with_capacity(2);
        if bytecode_requested {
            present.push(columns.bytecode_chunks().to_vec());
        }
        if ram_requested {
            present.push(columns.ram_chunks().to_vec());
        }
        let request = PrepareRequest {
            present,
            optional: selectors.into_iter().collect(),
        };
        let (trace, groups) = ValidatedTrace::prepare(source, request).map_err(Self::geometry)?;
        let mut present = groups.present.into_iter();
        let mut next_present = |requested| {
            if requested {
                present
                    .next()
                    .map(OwnedGroup::Held)
                    .ok_or_else(|| Self::geometry("preparation omitted a requested chunk group"))
            } else {
                Ok(OwnedGroup::NotRequested)
            }
        };
        let bytecode = next_present(bytecode_requested)?;
        let ram = next_present(ram_requested)?;
        let selectors = groups
            .optional
            .into_iter()
            .next()
            .map_or(OwnedGroup::NotRequested, OwnedGroup::Held);
        let trace = Arc::new(trace);
        self.prepared = Some(PreparedSource {
            trace: Arc::clone(&trace),
            bytecode,
            ram,
            selectors,
            plan: SharedPlan::NotBuilt,
        });
        Ok(trace)
    }

    fn prepared_mut(&mut self) -> Result<&mut PreparedSource, KernelError<F128>> {
        self.prepared
            .as_mut()
            .ok_or_else(|| Self::geometry("source has not been prepared"))
    }

    /// Moves held bytecode chunk bytes into their core; absent or taken is an error.
    pub fn take_bytecode_group(&mut self) -> Result<PresentGroup, KernelError<F128>> {
        self.prepared_mut()?
            .bytecode
            .take(SourceGroup::BytecodeChunks)
    }

    /// Moves held RAM chunk bytes into their core; absent or taken is an error.
    pub fn take_ram_group(&mut self) -> Result<PresentGroup, KernelError<F128>> {
        self.prepared_mut()?.ram.take(SourceGroup::RamChunks)
    }

    /// Moves held selector bytes into the cycle core; absent or taken is an error.
    pub fn take_selector_group(&mut self) -> Result<OptionalGroup, KernelError<F128>> {
        self.prepared_mut()?.selectors.take(SourceGroup::Selectors)
    }

    /// Requires preparation; builds the plan once and shares it until release.
    /// Access after release fails instead of rebuilding cycle-sized storage.
    pub fn plan(&mut self) -> Result<Arc<ScatterPlan<WitnessSource>>, KernelError<F128>> {
        let prepared = self.prepared_mut()?;
        match &prepared.plan {
            SharedPlan::Held(plan) => return Ok(Arc::clone(plan)),
            SharedPlan::Released => return Err(Self::geometry("scatter plan was released")),
            SharedPlan::NotBuilt => {}
        }
        let plan = Arc::new(ScatterPlan::new(Arc::clone(&prepared.trace)).map_err(Self::geometry)?);
        prepared.plan = SharedPlan::Held(Arc::clone(&plan));
        Ok(plan)
    }

    /// Drops the session's held plan reference. Repeated release is harmless;
    /// release before a plan was built is invalid geometry.
    pub fn release_plan(&mut self) -> Result<(), KernelError<F128>> {
        let prepared = self.prepared_mut()?;
        if matches!(prepared.plan, SharedPlan::NotBuilt) {
            return Err(Self::geometry("scatter plan has not been built"));
        }
        prepared.plan = SharedPlan::Released;
        Ok(())
    }
}
