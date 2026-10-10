//! Replayed witness data: XOR increments start from zero registers and canonical
//! initial RAM. `from_bits` and `from_facts` establish the pre-state reads and
//! fetched-row successors; direct construction owes that same contract.

use crate::error::{FactField, Rv64iProverError};
#[cfg(feature = "allocative")]
use allocative::Allocative;
#[cfg(feature = "test-utils")]
use common::{constants::RAM_START_ADDRESS, jolt_device::MemoryLayout};
use jolt_field::JoltField;
use jolt_kernels::WitnessPlane;
use jolt_rv64i_arith::decode::SourceParts;
use jolt_rv64i_arith::{
    BaseWords, BitsBuilder, BitsRow, Bytecode, BytecodeRow, Chunk, CycleError, CycleFacts, Layout,
    Variant, WitnessRow,
};
use jolt_rv64i_verifier::{commitment::BitsCommitmentScheme, statement::CheckedInputs};
#[cfg(feature = "test-utils")]
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

/// Sixteen-byte replay cache: the increment and low-first bytecode/RAM indices,
/// six position bits, `KeysDiffer`, `ShouldBranch`, `JalrLowBit` and six variant
/// bits. `Layout::new` bounds the total index width by 49 bits.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct DecodedCycle {
    pub inc: u64,
    pub digits: u64,
}

/// A checked shift and mask inside `DecodedCycle::digits`.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct DigitField {
    shift: usize,
    bits: usize,
    mask: u64,
}
impl DigitField {
    fn new(shift: usize, bits: usize) -> Self {
        Self {
            shift,
            bits,
            mask: u64::MAX >> (64 - bits),
        }
    }
    #[inline]
    pub fn read(self, row: &DecodedCycle) -> u64 {
        (row.digits >> self.shift) & self.mask
    }
    pub fn shift(self) -> usize {
        self.shift
    }
    pub fn bits(self) -> usize {
        self.bits
    }
    #[inline]
    fn pack(self, value: u64) -> u64 {
        (value & self.mask) << self.shift
    }
}

/// The sole packing geometry. Chunks accumulate their layout widths low first;
/// position, flags and variant follow the indices. `Layout::new` bounds their width.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct DigitFields {
    bytecode_index: DigitField,
    ram_index: DigitField,
    bytecode: [DigitField; 16],
    bytecode_len: usize,
    ram: [DigitField; 16],
    ram_len: usize,
    pos: [DigitField; 2],
    position: DigitField,
    keys_differ: DigitField,
    should_branch: DigitField,
    jalr_low_bit: DigitField,
    variant: DigitField,
}
impl DigitFields {
    pub fn new(layout: &Layout) -> Self {
        let b = layout.log_K_bytecode();
        let a = layout.log_K_ram();
        // Layout::new pins b + a <= 49, so all packed fields fit.
        debug_assert!(b + a + 15 <= 64);
        let mut shift = 0;
        let mut chunks = |chunks: &[Chunk]| {
            let mut fields = [DigitField::new(0, 1); 16];
            for (output, chunk) in fields.iter_mut().zip(chunks) {
                *output = DigitField::new(shift, usize::from(chunk.bits()));
                shift += output.bits;
            }
            fields
        };
        let bytecode = chunks(layout.bytecode_ra());
        let ram = chunks(layout.ram_ra());
        Self {
            bytecode_index: DigitField::new(0, b),
            ram_index: DigitField::new(b, a),
            bytecode,
            ram,
            bytecode_len: layout.bytecode_ra().len(),
            ram_len: layout.ram_ra().len(),
            pos: [DigitField::new(shift, 3), DigitField::new(shift + 3, 3)],
            position: DigitField::new(shift, 6),
            keys_differ: DigitField::new(shift + 6, 1),
            should_branch: DigitField::new(shift + 7, 1),
            jalr_low_bit: DigitField::new(shift + 8, 1),
            variant: DigitField::new(shift + 9, 6),
        }
    }
    pub fn bytecode_fields(&self) -> &[DigitField] {
        &self.bytecode[..self.bytecode_len]
    }
    pub fn ram_fields(&self) -> &[DigitField] {
        &self.ram[..self.ram_len]
    }
    pub fn bytecode_index(&self) -> DigitField {
        self.bytecode_index
    }
    pub fn ram_index(&self) -> DigitField {
        self.ram_index
    }
    pub fn pos_fields(&self) -> [DigitField; 2] {
        self.pos
    }
    pub fn pos(&self, digit: usize) -> Option<DigitField> {
        self.pos.get(digit).copied()
    }
    pub fn keys_differ(&self) -> DigitField {
        self.keys_differ
    }
    pub fn should_branch(&self) -> DigitField {
        self.should_branch
    }
    pub fn jalr_low_bit(&self) -> DigitField {
        self.jalr_low_bit
    }
    pub fn variant(&self) -> DigitField {
        self.variant
    }

    #[inline]
    fn pack(&self, index: u64, variant: Variant, parts: &SourceParts) -> DecodedCycle {
        DecodedCycle {
            inc: parts.inc,
            digits: self.bytecode_index.pack(index)
                | self.ram_index.pack(parts.ram_index)
                | self.position.pack(u64::from(parts.pos))
                | self.keys_differ.pack(u64::from(parts.keys_differ))
                | self.should_branch.pack(u64::from(parts.should_branch))
                | self.jalr_low_bit.pack(u64::from(parts.jalr_low_bit))
                | self.variant.pack(variant.index() as u64),
        }
    }
}

/// Replayed pre-state reads and successor PC for one committed cycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct CycleWords {
    /// Value selected by the fetched row's first register selector.
    pub rs1_value: u64,
    /// Value selected by the fetched row's second register selector.
    pub rs2_value: u64,
    /// Destination value before its XOR update.
    pub rd_pre_value: u64,
    /// Word at the committed RAM index, including index zero without an access.
    pub ram_read_value: u64,
    /// Next fetched PC, or the supplied final PC on the last cycle.
    pub next_pc: u64,
}
impl CycleWords {
    /// Forms the destination post-value from the replay's pre-value and increment.
    pub fn base_words(&self, store: bool, inc: u64) -> BaseWords {
        BaseWords {
            rs1_value: self.rs1_value,
            rs2_value: self.rs2_value,
            rd_write_value: self.rd_pre_value ^ if store { 0 } else { inc },
            ram_read_value: self.ram_read_value,
            next_pc: self.next_pc,
        }
    }
}
/// Shared committed rows and replayed words, with retained dense final RAM.
/// Direct construction owes nonzero power-of-two equal row lengths, matching layout and bytecode,
/// canonical initial RAM, and the same replay contract as the constructors; execution constraints
/// and a real fixed-point stall suffix remain obligations of the host and protocol relations.
#[derive(Clone, Debug)]
pub struct Rv64iWitness {
    /// Committed-column and address geometry.
    pub layout: Layout,
    /// Public fetched table shared with preprocessing.
    pub bytecode: Arc<Bytecode>,
    /// Nonzero power-of-two committed rows, shared without copying by proving and commitment.
    pub bits: Arc<[BitsRow]>,
    /// One pre-state record per committed cycle.
    pub words: Arc<[CycleWords]>,
    /// Packed digits written in the replay, with one row per committed cycle.
    pub decoded: Arc<[DecodedCycle]>,
    /// Number of fetched cycles of each variant; unused variant indices stay zero.
    pub variant_cycles: [u64; 64],
    /// Increasing nonzero initial RAM words; all omitted words start at zero.
    pub initial_ram: Vec<(u64, u64)>,
    /// Dense RAM after the final XOR update, with exactly `2^log_K_ram` words.
    pub final_ram: Vec<u64>,
    /// Successor after the final cycle; it must name a valid bytecode row.
    pub final_pc: u64,
}
/// A borrowed cycle walk with its packed geometry prepared once.
pub struct WitnessCycles<'w> {
    witness: &'w Rv64iWitness,
    fields: DigitFields,
}

/// Fetched instruction and decoded inputs, without reading a committed row.
pub struct CycleParts<'w> {
    pub fetched: &'w BytecodeRow,
    pub variant: Variant,
    pub base: BaseWords,
    pub sources: SourceParts,
}

impl<'w> WitnessCycles<'w> {
    /// Rejects an absent cycle or invalid fetch before decoding its inputs.
    #[inline]
    pub fn parts(&self, cycle: usize) -> Result<CycleParts<'w>, Rv64iProverError> {
        let witness = self.witness;
        if cycle >= witness.bits.len() {
            return Err(Rv64iProverError::CycleIndex {
                cycle,
                rows: witness.bits.len(),
            });
        }
        let decoded = witness
            .decoded
            .get(cycle)
            .ok_or_else(|| Rv64iProverError::CycleIndex {
                cycle,
                rows: witness.decoded.len(),
            })?;
        let index = self.fields.bytecode_index().read(decoded);
        let fetched = usize::try_from(index)
            .ok()
            .and_then(|i| witness.bytecode.rows().get(i))
            .ok_or_else(|| Rv64iProverError::InvalidBytecode { cycle, index })?;
        let variant = fetched
            .variant
            .ok_or_else(|| Rv64iProverError::InvalidBytecode { cycle, index })?;
        let words = witness
            .words
            .get(cycle)
            .ok_or_else(|| Rv64iProverError::CycleIndex {
                cycle,
                rows: witness.words.len(),
            })?;
        let sources = SourceParts {
            inc: decoded.inc,
            ram_index: self.fields.ram_index().read(decoded),
            pos: self.fields.position.read(decoded) as u8,
            keys_differ: self.fields.keys_differ().read(decoded) != 0,
            should_branch: self.fields.should_branch().read(decoded) != 0,
            jalr_low_bit: self.fields.jalr_low_bit().read(decoded) != 0,
        };
        Ok(CycleParts {
            fetched,
            variant,
            base: words.base_words(variant.is_store(), sources.inc),
            sources,
        })
    }

    /// Composes decoded inputs with the committed row through `WitnessRow::compute`.
    #[inline]
    pub fn row(&self, cycle: usize) -> Result<WitnessRow, Rv64iProverError> {
        let parts = self.parts(cycle)?;
        let bits = self
            .witness
            .bits
            .get(cycle)
            .ok_or(Rv64iProverError::CycleIndex {
                cycle,
                rows: self.witness.bits.len(),
            })?;
        Ok(WitnessRow::compute(
            &self.witness.layout,
            parts.fetched,
            &parts.base,
            bits,
        ))
    }
}

impl Rv64iWitness {
    /// Prepares packed geometry once for a borrowed walk of the cycles.
    pub fn cycles(&self) -> WitnessCycles<'_> {
        WitnessCycles {
            witness: self,
            fields: DigitFields::new(&self.layout),
        }
    }

    /// Replays shared committed rows from zero registers and canonical initial RAM without copying them.
    /// Rejects invalid geometry, fetches, chunk indicators, initial words or RAM allocation; it does
    /// not authenticate instruction transitions or check public outputs.
    pub fn from_bits(
        layout: Layout,
        bytecode: Arc<Bytecode>,
        bits: Arc<[BitsRow]>,
        initial_ram: Vec<(u64, u64)>,
        final_pc: u64,
    ) -> Result<Self, Rv64iProverError> {
        let mut witness = Self::prepare(layout, bytecode, bits, initial_ram, final_pc)?;
        witness.replay(None)?;
        Ok(witness)
    }

    /// Generates committed rows and checks present operands in one forward replay; normalized `NOOP`
    /// selector reads include index zero because original operand presence is erased, and RAM facts
    /// are read only on accesses. Returns the first difference before generating that row or a geometry,
    /// fetch, row-generation or allocation error; outputs and instruction transitions remain unchecked.
    pub fn from_facts(
        layout: Layout,
        bytecode: Arc<Bytecode>,
        facts: &[CycleFacts],
        initial_ram: Vec<(u64, u64)>,
    ) -> Result<Self, Rv64iProverError> {
        let final_pc = facts
            .last()
            .ok_or(Rv64iProverError::RowCount {
                expected: 1,
                bits: 0,
                words: 0,
            })?
            .next_pc;
        let final_ram = Self::initial_state(&layout, &initial_ram)?;
        let bits = (0..facts.len()).map(|_| [0; 4]).collect();
        let mut witness =
            Self::prepare_with_ram(layout, bytecode, bits, initial_ram, final_ram, final_pc)?;
        witness.replay(Some(facts))?;
        Ok(witness)
    }

    fn initial_state(
        layout: &Layout,
        initial_ram: &[(u64, u64)],
    ) -> Result<Vec<u64>, Rv64iProverError> {
        let count = Self::ram_words(layout)?;
        let mut previous = None;
        for &(index, value) in initial_ram {
            if value == 0 || index >= count as u64 || previous.is_some_and(|p| p >= index) {
                return Err(Rv64iProverError::InitialRam { index });
            }
            previous = Some(index);
        }
        let mut ram = Vec::new();
        ram.try_reserve_exact(count)
            .map_err(|source| Rv64iProverError::RamAllocation {
                log_K_ram: layout.log_K_ram(),
                source,
            })?;
        ram.resize(count, 0);
        for &(index, value) in initial_ram {
            *ram.get_mut(index as usize)
                .ok_or(Rv64iProverError::InitialRam { index })? = value;
        }
        Ok(ram)
    }

    fn prepare(
        layout: Layout,
        bytecode: Arc<Bytecode>,
        bits: Arc<[BitsRow]>,
        initial_ram: Vec<(u64, u64)>,
        final_pc: u64,
    ) -> Result<Self, Rv64iProverError> {
        let final_ram = Self::initial_state(&layout, &initial_ram)?;
        Self::prepare_with_ram(layout, bytecode, bits, initial_ram, final_ram, final_pc)
    }

    fn prepare_with_ram(
        layout: Layout,
        bytecode: Arc<Bytecode>,
        bits: Arc<[BitsRow]>,
        initial_ram: Vec<(u64, u64)>,
        final_ram: Vec<u64>,
        final_pc: u64,
    ) -> Result<Self, Rv64iProverError> {
        if !bits.len().is_power_of_two() {
            let expected =
                bits.len()
                    .checked_next_power_of_two()
                    .ok_or(Rv64iProverError::TraceDimension {
                        log_T: usize::BITS as usize,
                    })?;
            return Err(Rv64iProverError::RowCount {
                expected,
                bits: bits.len(),
                words: 0,
            });
        }
        let _ = BitsBuilder::new(&layout, &bytecode)?;
        let _ = bytecode.final_pc_index(final_pc)?;
        let decoded = (0..bits.len()).map(|_| DecodedCycle::default()).collect();
        let words = (0..bits.len()).map(|_| CycleWords::default()).collect();
        Ok(Self {
            layout,
            bytecode,
            bits,
            words,
            decoded,
            variant_cycles: [0; 64],
            initial_ram,
            final_ram,
            final_pc,
        })
    }

    fn replay(&mut self, facts: Option<&[CycleFacts]>) -> Result<(), Rv64iProverError> {
        let fields = DigitFields::new(&self.layout);
        let decoded = Arc::get_mut(&mut self.decoded).ok_or(Rv64iProverError::SharedBuffer)?;
        let builder = BitsBuilder::new(&self.layout, &self.bytecode)?;
        let mut rows = match facts {
            Some(facts) => ReplayRows::Facts {
                bits: Arc::get_mut(&mut self.bits).ok_or(Rv64iProverError::SharedBuffer)?,
                facts,
            },
            None => ReplayRows::Bits(&self.bits),
        };
        let words = Arc::get_mut(&mut self.words).ok_or(Rv64iProverError::SharedBuffer)?;
        let mut registers = [0_u64; 32];
        for cycle in 0..words.len() {
            let input = match &mut rows {
                ReplayRows::Bits(bits) => {
                    let (index, parts) = SourceParts::from_checked_bits(&self.layout, &bits[cycle])
                        .map_err(|error| Rv64iProverError::MultipleIndicators {
                            cycle,
                            start: error.start,
                        })?;
                    ReplayCycle::Bits { index, parts }
                }
                ReplayRows::Facts { bits, facts } => ReplayCycle::Facts {
                    fact: &facts[cycle],
                    committed: &mut bits[cycle],
                },
            };
            let (index, fact) = match &input {
                ReplayCycle::Bits { index, .. } => (*index, None),
                ReplayCycle::Facts { fact, .. } => (u64::from(fact.bytecode_index), Some(*fact)),
            };
            let row = usize::try_from(index)
                .ok()
                .and_then(|i| self.bytecode.rows().get(i))
                .filter(|r| r.variant.is_some())
                .ok_or(Rv64iProverError::InvalidBytecode { cycle, index })?;
            let variant = row
                .variant
                .ok_or(Rv64iProverError::InvalidBytecode { cycle, index })?;
            let access = variant.access().is_some();
            let ram_index = match &input {
                ReplayCycle::Facts { fact, .. } => {
                    if access {
                        fact.ram_word_index
                    } else {
                        0
                    }
                }
                ReplayCycle::Bits { parts, .. } => parts.ram_index,
            };
            let read = |register: u8| {
                registers
                    .get(usize::from(register))
                    .copied()
                    .ok_or(Rv64iProverError::Register { register })
            };
            let mut word = CycleWords {
                rs1_value: read(row.rs1)?,
                rs2_value: read(row.rs2)?,
                rd_pre_value: read(row.rd)?,
                ram_read_value: 0,
                next_pc: self.final_pc,
            };
            if let Some(fact) = fact {
                for ((field, expected, found), present) in [
                    (FactField::Rs1Value, word.rs1_value, fact.rs1_value),
                    (FactField::Rs2Value, word.rs2_value, fact.rs2_value),
                    (FactField::RdPreValue, word.rd_pre_value, fact.rd_pre_value),
                ]
                .into_iter()
                .zip(Self::register_operands(variant))
                {
                    if present && expected != found {
                        return Err(Rv64iProverError::FactMismatch {
                            cycle,
                            field,
                            expected,
                            found,
                        });
                    }
                }
            }
            word.ram_read_value = self
                .final_ram
                .get(ram_index as usize)
                .copied()
                .ok_or(Rv64iProverError::InitialRam { index: ram_index })?;
            if let Some(fact) = fact {
                if access && word.ram_read_value != fact.ram_pre_value {
                    return Err(Rv64iProverError::FactMismatch {
                        cycle,
                        field: FactField::RamPreValue,
                        expected: word.ram_read_value,
                        found: fact.ram_pre_value,
                    });
                }
            }
            let parts = match input {
                ReplayCycle::Bits { parts, .. } => parts,
                ReplayCycle::Facts { fact, committed } => {
                    let (bits, parts) = builder
                        .bits_row_with_parts(fact)
                        .map_err(|error| CycleError { cycle, error })?;
                    *committed = bits;
                    parts
                }
            };
            if let Some(previous) = cycle.checked_sub(1).and_then(|i| words.get_mut(i)) {
                previous.next_pc = row.pc;
            }
            decoded[cycle] = fields.pack(index, variant, &parts);
            self.variant_cycles[variant.index()] += 1;
            let inc = decoded[cycle].inc;
            if row.variant.is_some_and(|v| v.is_store()) {
                *self
                    .final_ram
                    .get_mut(ram_index as usize)
                    .ok_or(Rv64iProverError::InitialRam { index: ram_index })? ^= inc;
            } else {
                *registers
                    .get_mut(usize::from(row.rd))
                    .ok_or(Rv64iProverError::Register { register: row.rd })? ^= inc;
            }
            words[cycle] = word;
        }
        Ok(())
    }

    fn register_operands(variant: Variant) -> [bool; 3] {
        match variant {
            Variant::ECALL | Variant::EBREAK => [false; 3],
            Variant::JAL | Variant::JAL_X0 | Variant::LUI | Variant::AUIPC => [false, false, true],
            Variant::SB
            | Variant::SH
            | Variant::SW
            | Variant::SD
            | Variant::BEQ
            | Variant::BNE
            | Variant::BLT
            | Variant::BGE
            | Variant::BLTU
            | Variant::BGEU => [true, true, false],
            Variant::ADD
            | Variant::SUB
            | Variant::ADDW
            | Variant::SUBW
            | Variant::AND
            | Variant::OR
            | Variant::XOR
            | Variant::SLT
            | Variant::SLTU
            | Variant::SLL
            | Variant::SRL
            | Variant::SRA
            | Variant::SLLW
            | Variant::SRLW
            | Variant::SRAW => [true; 3],
            // NOOP preserves selectors but merges operand shapes; index-zero selectors also read the replayed register state.
            Variant::NOOP => [true; 3],
            Variant::ADDI
            | Variant::ADDIW
            | Variant::ANDI
            | Variant::ORI
            | Variant::XORI
            | Variant::SLTI
            | Variant::SLTIU
            | Variant::SLLI
            | Variant::SRLI
            | Variant::SRAI
            | Variant::SLLIW
            | Variant::SRLIW
            | Variant::SRAIW
            | Variant::LB
            | Variant::LH
            | Variant::LW
            | Variant::LD
            | Variant::LBU
            | Variant::LHU
            | Variant::LWU
            | Variant::JALR
            | Variant::JALR_X0
            | Variant::LOAD1_X0
            | Variant::LOAD2_X0
            | Variant::LOAD4_X0
            | Variant::LOAD8_X0 => [true, false, true],
        }
    }

    pub(crate) fn ram_words(layout: &Layout) -> Result<usize, Rv64iProverError> {
        1_usize
            .checked_shl(layout.log_K_ram() as u32)
            .ok_or(Rv64iProverError::RamDimension {
                log_K_ram: layout.log_K_ram(),
            })
    }

    /// Checks the retained final RAM against every public I/O-mask word in ascending order.
    /// Returns a layout or RAM-length error first, otherwise the first differing word; omitted
    /// public segments contribute zero, including termination on a panic statement.
    pub fn check_outputs<S: BitsCommitmentScheme>(
        &self,
        checked: &CheckedInputs<'_, S>,
    ) -> Result<(), Rv64iProverError> {
        let layout = checked.layout();
        if self.layout.log_K_bytecode() != layout.log_K_bytecode()
            || self.layout.log_K_ram() != layout.log_K_ram()
            || self.layout.lowest_address() != layout.lowest_address()
        {
            return Err(Rv64iProverError::OutputLayoutMismatch);
        }
        let expected = Self::ram_words(layout)?;
        if self.final_ram.len() != expected {
            return Err(Rv64iProverError::FinalRamLength {
                expected,
                found: self.final_ram.len(),
            });
        }
        let io = checked.io();
        for index in io.io_mask_start..io.io_mask_end {
            let expected = io
                .segments
                .iter()
                .find_map(|segment| {
                    index
                        .checked_sub(segment.start_index)
                        .and_then(|offset| usize::try_from(offset).ok())
                        .and_then(|offset| segment.words.get(offset))
                        .copied()
                })
                .unwrap_or(0);
            let found = self.final_ram[index as usize];
            if expected != found {
                return Err(Rv64iProverError::OutputMismatch {
                    index: index as u64,
                    expected,
                    found,
                });
            }
        }
        Ok(())
    }
    /// Generates unconstrained seeded rows and replays their XOR updates for protocol fixtures.
    /// The final cycle sets the termination word to one; geometry, missing stores, inadmissible
    /// store addresses and RAM-allocation failures are typed errors.
    #[cfg(feature = "test-utils")]
    pub fn synthetic(
        seed: u64,
        layout: Layout,
        bytecode: Arc<Bytecode>,
        memory_layout: &MemoryLayout,
        initial_ram: Vec<(u64, u64)>,
        log_T: usize,
    ) -> Result<Self, Rv64iProverError> {
        let count = 1_usize
            .checked_shl(u32::try_from(log_T).unwrap_or(u32::MAX))
            .ok_or(Rv64iProverError::TraceDimension { log_T })?;
        let valid: Vec<_> = bytecode
            .rows()
            .iter()
            .enumerate()
            .filter(|(_, r)| r.variant.is_some())
            .collect();
        let store = valid
            .iter()
            .find(|(_, r)| r.variant.is_some_and(|v| v.is_store()))
            .ok_or(Rv64iProverError::MissingStore)?
            .0;
        let termination = memory_layout.remapped_word_address(memory_layout.termination)?;
        let io_start = memory_layout.remapped_word_address(memory_layout.input_start)?;
        let io_end = memory_layout.remapped_word_address(RAM_START_ADDRESS)?;
        let output_start = memory_layout.remapped_word_address(memory_layout.output_start)?;
        let output_end = memory_layout.remapped_word_address(memory_layout.output_end)?;
        let limit = 1_u64 << layout.log_K_ram();
        let ordinary = (0..io_start.min(limit))
            .chain(output_start..output_end.min(limit))
            .chain(io_end..limit)
            .next()
            .ok_or(Rv64iProverError::StoreAddress)?;
        let termination_value = initial_ram
            .iter()
            .find(|(i, _)| *i == termination)
            .map_or(0, |(_, v)| *v);
        let mut rng = StdRng::seed_from_u64(seed);
        let final_ram = Self::initial_state(&layout, &initial_ram)?;
        let mut bits: Arc<[BitsRow]> = (0..count).map(|_| [0; 4]).collect();
        let rows = Arc::get_mut(&mut bits).ok_or(Rv64iProverError::SharedBuffer)?;
        for (cycle, output) in rows.iter_mut().enumerate() {
            let (index, row) = if cycle + 1 == count {
                (
                    store,
                    bytecode
                        .rows()
                        .get(store)
                        .ok_or(Rv64iProverError::MissingStore)?,
                )
            } else {
                *valid
                    .get(rng.gen_range(0..valid.len()))
                    .ok_or(Rv64iProverError::MissingStore)?
            };
            let mut committed = rng.gen::<BitsRow>();
            layout.write_bytecode_index(&mut committed, index as u64)?;
            let address = if cycle + 1 == count {
                termination
            } else if row.variant.is_some_and(|v| v.is_store()) {
                ordinary
            } else if row.variant.is_some_and(|v| v.access().is_some()) {
                rng.gen_range(0..limit)
            } else {
                0
            };
            layout.write_ram_index(&mut committed, address)?;
            layout.write_pos(&mut committed, rng.gen_range(0..64))?;
            if cycle + 1 == count {
                committed[0] = termination_value ^ 1;
            }
            *output = committed;
        }
        let final_pc = valid
            .get(rng.gen_range(0..valid.len()))
            .ok_or(Rv64iProverError::MissingStore)?
            .1
            .pc;
        let mut witness =
            Self::prepare_with_ram(layout, bytecode, bits, initial_ram, final_ram, final_pc)?;
        witness.replay(None)?;
        Ok(witness)
    }
}
enum ReplayCycle<'a> {
    Bits {
        index: u64,
        parts: SourceParts,
    },
    Facts {
        fact: &'a CycleFacts,
        committed: &'a mut BitsRow,
    },
}

enum ReplayRows<'a> {
    Bits(&'a [BitsRow]),
    Facts {
        bits: &'a mut [BitsRow],
        facts: &'a [CycleFacts],
    },
}

/// Borrowed witness plane used by the RV64I stage kernels without cloning trace storage.
pub struct Rv64iPlane;
impl<F: JoltField> WitnessPlane<F> for Rv64iPlane {
    type Ref<'w>
        = &'w Rv64iWitness
    where
        F: 'w;
}
