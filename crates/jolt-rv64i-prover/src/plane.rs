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
use jolt_rv64i_kernels::par::CycleChunks;
use jolt_rv64i_verifier::{commitment::BitsCommitmentScheme, statement::CheckedInputs};
#[cfg(feature = "test-utils")]
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
    ParallelSlice, ParallelSliceMut,
};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
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
    pub fn mask(self) -> u64 {
        self.mask
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
    /// Packed digits produced with committed rows, with one row per committed cycle.
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
        witness.replay(())?;
        Ok(witness)
    }

    /// Generates committed and decoded rows and checks present operands in parallel chunks,
    /// using chunk-entry states from a prefix of XOR summaries. Normalized `NOOP`
    /// selector reads include index zero because original operand presence is erased, and RAM facts
    /// are read only on accesses. Returns the first difference or a geometry,
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
        witness.replay(facts)?;
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

    #[expect(
        clippy::expect_used,
        reason = "prepare_with_ram checked nonzero power-of-two witness geometry"
    )]
    #[inline(never)]
    fn replay<'a, R: ReplayRequest<'a>>(&mut self, request: R) -> Result<(), Rv64iProverError> {
        let facts = request.facts();
        let context = ReplayContext {
            layout: &self.layout,
            bytecode: &self.bytecode,
        };
        if rayon::current_num_threads() == 1 {
            let mut state = ReplayState::new(self.final_pc, 0);
            state.ram_zero = self.final_ram[0];
            let decoded = Arc::get_mut(&mut self.decoded).ok_or(Rv64iProverError::SharedBuffer)?;
            let words = Arc::get_mut(&mut self.words).ok_or(Rv64iProverError::SharedBuffer)?;
            let result = if let Some(facts) = facts {
                ReplayContext::run_facts(
                    context.layout,
                    context.bytecode,
                    facts,
                    Arc::get_mut(&mut self.bits).ok_or(Rv64iProverError::SharedBuffer)?,
                    words,
                    decoded,
                    &mut state,
                    self.final_ram.as_mut_slice(),
                )
            } else {
                ReplayContext::run_bits(
                    context.layout,
                    context.bytecode,
                    self.bits.as_ref(),
                    words,
                    decoded,
                    &mut state,
                    self.final_ram.as_mut_slice(),
                )
            };
            result?;
            self.variant_cycles = state.counts;
            return Ok(());
        }
        let chunk_len = CycleChunks::new(self.bits.len().ilog2() as usize, 0)
            .expect("checked witness geometry")
            .chunk_len();
        let source = facts.map_or(ReplayRows::Bits(&self.bits), ReplayRows::Facts);
        let mut chunks: Vec<_> = (0..self.bits.len() / chunk_len)
            .into_par_iter()
            .map(|chunk| {
                context.summarize(
                    source,
                    chunk * chunk_len,
                    chunk_len,
                    self.final_pc,
                    self.final_ram.len(),
                )
            })
            .collect();
        let mut registers = [0_u64; 32];
        for chunk in 0..chunks.len() {
            let next_pc = chunks
                .get(chunk + 1)
                .and_then(|next| next.first_pc)
                .unwrap_or(self.final_pc);
            let state = &mut chunks[chunk];
            state.next_pc = next_pc;
            for (entry, running) in state.registers.iter_mut().zip(&mut registers) {
                let delta = *entry;
                *entry = *running;
                *running ^= delta;
            }
            let delta = state.ram_zero;
            state.ram_zero = self.final_ram[0];
            self.final_ram[0] ^= delta;
            for (&index, entry) in &mut state.ram {
                let delta = *entry;
                *entry = self.final_ram[index as usize];
                self.final_ram[index as usize] ^= delta;
            }
        }
        let decoded = Arc::get_mut(&mut self.decoded).ok_or(Rv64iProverError::SharedBuffer)?;
        let words = Arc::get_mut(&mut self.words).ok_or(Rv64iProverError::SharedBuffer)?;
        let error = if let Some(facts) = facts {
            let bits = Arc::get_mut(&mut self.bits).ok_or(Rv64iProverError::SharedBuffer)?;
            bits.par_chunks_mut(chunk_len)
                .zip(facts.par_chunks(chunk_len))
                .zip(words.par_chunks_mut(chunk_len))
                .zip(decoded.par_chunks_mut(chunk_len))
                .zip(chunks.par_iter_mut())
                .enumerate()
                .map(|(chunk, ((((bits, facts), words), decoded), state))| {
                    ReplayContext::run_facts(
                        context.layout,
                        context.bytecode,
                        facts,
                        bits,
                        words,
                        decoded,
                        state,
                        ChunkMemory,
                    )
                    .err()
                    .map(|error| (chunk * chunk_len, error))
                })
                .reduce(|| None, first_replay_fault)
        } else {
            self.bits
                .par_chunks(chunk_len)
                .zip(words.par_chunks_mut(chunk_len))
                .zip(decoded.par_chunks_mut(chunk_len))
                .zip(chunks.par_iter_mut())
                .enumerate()
                .map(|(chunk, (((bits, words), decoded), state))| {
                    ReplayContext::run_bits(
                        context.layout,
                        context.bytecode,
                        bits,
                        words,
                        decoded,
                        state,
                        ChunkMemory,
                    )
                    .err()
                    .map(|error| (chunk * chunk_len, error))
                })
                .reduce(|| None, first_replay_fault)
        };
        if let Some((_, error)) = error {
            return Err(error);
        }
        for state in chunks {
            for (total, count) in self.variant_cycles.iter_mut().zip(state.counts) {
                *total += count;
            }
        }
        Ok(())
    }

    #[inline(always)]
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
        witness.replay(())?;
        Ok(witness)
    }
}
type ReplayFault = (usize, Rv64iProverError);

trait ReplayRequest<'a> {
    fn facts(self) -> Option<&'a [CycleFacts]>;
}
impl<'a> ReplayRequest<'a> for () {
    fn facts(self) -> Option<&'a [CycleFacts]> {
        None
    }
}
impl<'a> ReplayRequest<'a> for &'a [CycleFacts] {
    fn facts(self) -> Option<&'a [CycleFacts]> {
        Some(self)
    }
}

// Local scans return their first fault, so ordering chunks orders faulting cycles.
fn first_replay_fault(
    left: Option<ReplayFault>,
    right: Option<ReplayFault>,
) -> Option<ReplayFault> {
    match (left, right) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (left, right) => left.or(right),
    }
}

#[derive(Default)]
struct RamHasher(u64);
impl Hasher for RamHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
    }
    fn write_u64(&mut self, value: u64) {
        self.0 = value.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(32);
    }
}

struct ReplayState {
    registers: [u64; 32],
    ram_zero: u64,
    ram: HashMap<u64, u64, BuildHasherDefault<RamHasher>>,
    counts: [u64; 64],
    start: usize,
    first_pc: Option<u64>,
    next_pc: u64,
}
impl ReplayState {
    fn new(next_pc: u64, start: usize) -> Self {
        Self {
            registers: [0; 32],
            ram_zero: 0,
            ram: HashMap::default(),
            counts: [0; 64],
            start,
            first_pc: None,
            next_pc,
        }
    }
}

#[derive(Clone, Copy)]
enum ReplayRows<'a> {
    Bits(&'a [BitsRow]),
    Facts(&'a [CycleFacts]),
}
impl<'a> ReplayRows<'a> {
    #[inline]
    fn input(self, cycle: usize, layout: &Layout) -> Result<ReplayCycle<'a>, Rv64iProverError> {
        match self {
            Self::Bits(bits) => {
                let (index, parts) =
                    SourceParts::from_checked_bits(layout, &bits[cycle]).map_err(|error| {
                        Rv64iProverError::MultipleIndicators {
                            cycle,
                            start: error.start,
                        }
                    })?;
                Ok(ReplayCycle::Bits { index, parts })
            }
            Self::Facts(facts) => Ok(ReplayCycle::Facts(&facts[cycle])),
        }
    }
}

enum ReplayCycle<'a> {
    Bits { index: u64, parts: SourceParts },
    Facts(&'a CycleFacts),
}
impl ReplayCycle<'_> {
    #[inline(always)]
    fn index(&self) -> u64 {
        match self {
            Self::Bits { index, .. } => *index,
            Self::Facts(fact) => u64::from(fact.bytecode_index),
        }
    }
    #[inline(always)]
    fn ram_index(&self, variant: Variant) -> u64 {
        match self {
            Self::Bits { parts, .. } => parts.ram_index,
            Self::Facts(fact) => {
                if variant.access().is_some() {
                    fact.ram_word_index
                } else {
                    0
                }
            }
        }
    }
    #[inline(always)]
    fn inc(&self, variant: Variant) -> u64 {
        match self {
            Self::Bits { parts, .. } => parts.inc,
            Self::Facts(fact) => fact.increment(variant),
        }
    }
}

enum ReplayUpdate {
    Register { register: u8, inc: u64 },
    Ram { index: u64, inc: u64 },
}
impl ReplayUpdate {
    #[inline(always)]
    fn new(variant: Variant, register: u8, index: u64, inc: u64) -> Self {
        if variant.is_store() {
            Self::Ram { index, inc }
        } else {
            Self::Register { register, inc }
        }
    }

    #[inline(always)]
    fn apply<M: ReplayMemory>(
        self,
        registers: &mut [u64; 32],
        memory: &mut M,
        state: &mut ReplayState,
    ) -> Result<(), Rv64iProverError> {
        let (value, inc) = match self {
            Self::Register { register, inc } => (
                registers
                    .get_mut(usize::from(register))
                    .ok_or(Rv64iProverError::Register { register })?,
                inc,
            ),
            Self::Ram { index, inc } => (memory.cell(state, index)?, inc),
        };
        *value ^= inc;
        Ok(())
    }
}

trait ReplayMemory {
    const CHUNKED: bool;
    fn read(&self, state: &ReplayState, index: u64) -> Result<u64, Rv64iProverError>;
    fn cell<'s>(
        &'s mut self,
        state: &'s mut ReplayState,
        index: u64,
    ) -> Result<&'s mut u64, Rv64iProverError>;
}
impl ReplayMemory for &mut [u64] {
    const CHUNKED: bool = false;
    #[inline]
    fn read(&self, _state: &ReplayState, index: u64) -> Result<u64, Rv64iProverError> {
        usize::try_from(index)
            .ok()
            .and_then(|index| self.get(index))
            .copied()
            .ok_or(Rv64iProverError::InitialRam { index })
    }
    #[inline]
    fn cell<'s>(
        &'s mut self,
        _state: &'s mut ReplayState,
        index: u64,
    ) -> Result<&'s mut u64, Rv64iProverError> {
        usize::try_from(index)
            .ok()
            .and_then(|index| self.get_mut(index))
            .ok_or(Rv64iProverError::InitialRam { index })
    }
}
struct ChunkMemory;
impl ReplayMemory for ChunkMemory {
    const CHUNKED: bool = true;
    #[inline]
    fn read(&self, state: &ReplayState, index: u64) -> Result<u64, Rv64iProverError> {
        if index == 0 {
            Ok(state.ram_zero)
        } else {
            state
                .ram
                .get(&index)
                .copied()
                .ok_or(Rv64iProverError::InitialRam { index })
        }
    }
    #[inline]
    fn cell<'s>(
        &'s mut self,
        state: &'s mut ReplayState,
        index: u64,
    ) -> Result<&'s mut u64, Rv64iProverError> {
        if index == 0 {
            Ok(&mut state.ram_zero)
        } else {
            state
                .ram
                .get_mut(&index)
                .ok_or(Rv64iProverError::InitialRam { index })
        }
    }
}
struct SummaryMemory;
impl ReplayMemory for SummaryMemory {
    const CHUNKED: bool = true;
    #[inline]
    fn read(&self, state: &ReplayState, index: u64) -> Result<u64, Rv64iProverError> {
        ChunkMemory.read(state, index)
    }
    #[inline]
    fn cell<'s>(
        &'s mut self,
        state: &'s mut ReplayState,
        index: u64,
    ) -> Result<&'s mut u64, Rv64iProverError> {
        if index == 0 {
            Ok(&mut state.ram_zero)
        } else {
            Ok(state.ram.entry(index).or_insert(0))
        }
    }
}
trait ReplayInput {
    fn rows(&self) -> ReplayRows<'_>;
    fn destination(&mut self) -> Option<&mut [BitsRow]>;
}
impl ReplayInput for &[BitsRow] {
    #[inline]
    fn rows(&self) -> ReplayRows<'_> {
        ReplayRows::Bits(self)
    }
    #[inline]
    fn destination(&mut self) -> Option<&mut [BitsRow]> {
        None
    }
}
struct FactRows<'a> {
    bits: &'a mut [BitsRow],
    facts: &'a [CycleFacts],
}
impl ReplayInput for FactRows<'_> {
    #[inline]
    fn rows(&self) -> ReplayRows<'_> {
        ReplayRows::Facts(self.facts)
    }
    #[inline]
    fn destination(&mut self) -> Option<&mut [BitsRow]> {
        Some(self.bits)
    }
}

struct ReplayContext<'a> {
    layout: &'a Layout,
    bytecode: &'a Bytecode,
}
impl<'a> ReplayContext<'a> {
    #[inline(always)]
    fn fetched(
        &self,
        index: u64,
        cycle: usize,
    ) -> Result<(&BytecodeRow, Variant), Rv64iProverError> {
        let row = usize::try_from(index)
            .ok()
            .and_then(|index| self.bytecode.rows().get(index))
            .ok_or(Rv64iProverError::InvalidBytecode { cycle, index })?;
        let variant = row
            .variant
            .ok_or(Rv64iProverError::InvalidBytecode { cycle, index })?;
        Ok((row, variant))
    }

    fn summarize(
        &self,
        source: ReplayRows<'_>,
        start: usize,
        len: usize,
        final_pc: u64,
        ram_words: usize,
    ) -> ReplayState {
        let mut state = ReplayState::new(final_pc, start);
        let mut registers = state.registers;
        // Truncated or unchecked XOR summaries may corrupt later entry values;
        // those chunks cannot outrank the earlier fault returned by run.
        for cycle in start..start + len {
            let Ok(input) = source.input(cycle, self.layout) else {
                break;
            };
            let Ok((row, variant)) = self.fetched(input.index(), cycle) else {
                break;
            };
            if cycle == start {
                state.first_pc = Some(row.pc);
            }
            let index = input.ram_index(variant);
            if index >= ram_words as u64 {
                break;
            }
            let update = ReplayUpdate::new(variant, row.rd, index, input.inc(variant));
            if matches!(update, ReplayUpdate::Register { .. }) && index != 0 {
                let _ = state.ram.entry(index).or_insert(0);
            }
            if update
                .apply(&mut registers, &mut SummaryMemory, &mut state)
                .is_err()
            {
                break;
            }
        }
        state.registers = registers;
        state
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "flat borrowed geometry and output slices retain alias information in the hot loop"
    )]
    #[inline(never)]
    fn run_facts<M: ReplayMemory>(
        layout: &'a Layout,
        bytecode: &'a Bytecode,
        facts: &[CycleFacts],
        bits: &mut [BitsRow],
        words: &mut [CycleWords],
        decoded: &mut [DecodedCycle],
        state: &mut ReplayState,
        memory: M,
    ) -> Result<(), Rv64iProverError> {
        Self { layout, bytecode }.run(
            state.start,
            FactRows { bits, facts },
            words,
            decoded,
            state,
            memory,
        )
    }

    #[inline(never)]
    fn run_bits<M: ReplayMemory>(
        layout: &'a Layout,
        bytecode: &'a Bytecode,
        bits: &[BitsRow],
        words: &mut [CycleWords],
        decoded: &mut [DecodedCycle],
        state: &mut ReplayState,
        memory: M,
    ) -> Result<(), Rv64iProverError> {
        Self { layout, bytecode }.run(state.start, bits, words, decoded, state, memory)
    }

    #[inline(always)]
    fn run<R: ReplayInput, M: ReplayMemory>(
        &self,
        start: usize,
        mut input: R,
        words: &mut [CycleWords],
        decoded: &mut [DecodedCycle],
        state: &mut ReplayState,
        mut memory: M,
    ) -> Result<(), Rv64iProverError> {
        let fields = DigitFields::new(self.layout);
        let builder = BitsBuilder::new(self.layout, self.bytecode)?;
        let next_pc = state.next_pc;
        let mut registers = state.registers;
        let mut counts = state.counts;
        for offset in 0..words.len() {
            let cycle = if M::CHUNKED { start + offset } else { offset };
            let source = input.rows();
            let input_cycle = source
                .input(offset, self.layout)
                .map_err(|error| match error {
                    Rv64iProverError::MultipleIndicators { start, .. } => {
                        Rv64iProverError::MultipleIndicators { cycle, start }
                    }
                    error => error,
                })?;
            let index = input_cycle.index();
            let fact = match &input_cycle {
                ReplayCycle::Facts(fact) => Some(*fact),
                ReplayCycle::Bits { .. } => None,
            };
            let (row, variant) = self.fetched(index, cycle)?;
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
                next_pc,
            };
            if let Some(fact) = fact {
                for ((field, expected, found), present) in [
                    (FactField::Rs1Value, word.rs1_value, fact.rs1_value),
                    (FactField::Rs2Value, word.rs2_value, fact.rs2_value),
                    (FactField::RdPreValue, word.rd_pre_value, fact.rd_pre_value),
                ]
                .into_iter()
                .zip(Rv64iWitness::register_operands(variant))
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
            let ram_index = input_cycle.ram_index(variant);
            word.ram_read_value = memory.read(state, ram_index)?;
            if let Some(fact) = fact {
                if variant.access().is_some() && word.ram_read_value != fact.ram_pre_value {
                    return Err(Rv64iProverError::FactMismatch {
                        cycle,
                        field: FactField::RamPreValue,
                        expected: word.ram_read_value,
                        found: fact.ram_pre_value,
                    });
                }
            }
            let parts = match input_cycle {
                ReplayCycle::Bits { parts, .. } => parts,
                ReplayCycle::Facts(fact) => {
                    let (bits, parts) = builder
                        .bits_row_with_parts(fact)
                        .map_err(|error| CycleError { cycle, error })?;
                    if let Some(output) = input.destination() {
                        output[offset] = bits;
                    }
                    parts
                }
            };
            if let Some(previous) = offset.checked_sub(1).and_then(|i| words.get_mut(i)) {
                previous.next_pc = row.pc;
            }
            decoded[offset] = fields.pack(index, variant, &parts);
            counts[variant.index()] += 1;
            ReplayUpdate::new(variant, row.rd, ram_index, parts.inc).apply(
                &mut registers,
                &mut memory,
                state,
            )?;
            words[offset] = word;
        }
        state.registers = registers;
        state.counts = counts;
        Ok(())
    }
}

/// Borrowed witness plane used by the RV64I stage kernels without cloning trace storage.
pub struct Rv64iPlane;
impl<F: JoltField> WitnessPlane<F> for Rv64iPlane {
    type Ref<'w>
        = &'w Rv64iWitness
    where
        F: 'w;
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "malformed fixtures fail their tests")]
mod tests {
    use super::{DigitFields, Rv64iWitness};
    use crate::error::{FactField, Rv64iProverError};
    use jolt_program::image::decode::decode_instruction;
    use jolt_riscv::RV64I;
    use jolt_rv64i_arith::{Bytecode, CycleError, CycleFacts, Layout, Variant, WitnessError};
    use rayon::ThreadPoolBuilder;
    use std::sync::Arc;

    #[test]
    fn parallel_fact_generation_keeps_counts_and_first_fault_across_chunks() {
        let layout = Layout::new(1, 5, 0).unwrap();
        let instruction = decode_instruction(0x0000_006f, 0, false, RV64I).unwrap();
        let bytecode = Arc::new(Bytecode::preprocess(&[instruction], &layout).unwrap());
        let facts = vec![CycleFacts::default(); 1 << 14];
        for threads in [1, 12] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let witness = pool
                .install(|| {
                    Rv64iWitness::from_facts(layout.clone(), Arc::clone(&bytecode), &facts, vec![])
                })
                .unwrap();
            assert_eq!(
                witness.variant_cycles[Variant::JAL_X0.index()],
                facts.len() as u64
            );
            assert_eq!(
                witness.variant_cycles.iter().sum::<u64>(),
                facts.len() as u64
            );
            let fields = DigitFields::new(&layout);
            for (bits, decoded) in witness.bits.iter().zip(witness.decoded.iter()) {
                assert_eq!(*bits, [0; 4]);
                assert_eq!(
                    fields.variant().read(decoded),
                    Variant::JAL_X0.index() as u64
                );
                assert_eq!(decoded.inc, 0);
            }
            let mut changed = facts.clone();
            changed[8197].bytecode_index = u32::MAX;
            changed[4107].bytecode_index = u32::MAX;
            let result = pool.install(|| {
                Rv64iWitness::from_facts(layout.clone(), Arc::clone(&bytecode), &changed, vec![])
            });
            assert!(
                matches!(result, Err(Rv64iProverError::InvalidBytecode { cycle: 4107, index }) if index == u64::from(u32::MAX))
            );
            let load = decode_instruction(0x0000_3083, 0, false, RV64I).unwrap();
            let load_bytecode = Arc::new(Bytecode::preprocess(&[load], &layout).unwrap());
            let mut accesses = facts.clone();
            accesses[8197].ram_word_index = 2;
            accesses[4107].ram_word_index = 1;
            let result = pool.install(|| {
                Rv64iWitness::from_facts(layout.clone(), load_bytecode, &accesses, vec![])
            });
            assert!(matches!(
                result,
                Err(Rv64iProverError::Cycle(CycleError {
                    cycle: 4107,
                    error: WitnessError::RamWordIndexMismatch {
                        expected: 0,
                        found: 1
                    },
                }))
            ));
            changed[513].rd_pre_value = 1;
            let result = pool.install(|| {
                Rv64iWitness::from_facts(layout.clone(), Arc::clone(&bytecode), &changed, vec![])
            });
            assert!(matches!(
                result,
                Err(Rv64iProverError::FactMismatch {
                    cycle: 513,
                    field: FactField::RdPreValue,
                    expected: 0,
                    found: 1
                })
            ));
        }
    }

    #[test]
    fn chunk_entry_reads_preserve_first_fault_and_literal_xor_state() {
        let layout = Layout::new(3, 5, 0).unwrap();
        let instructions: Vec<_> = [
            0x0000_006f,
            0x0000_00b7,
            0x0000_8113,
            0x0010_3023,
            0x0000_3183,
        ]
        .into_iter()
        .enumerate()
        .map(|(index, word)| decode_instruction(word, 4 * index as u64, false, RV64I).unwrap())
        .collect();
        let bytecode = Arc::new(Bytecode::preprocess(&instructions, &layout).unwrap());
        for threads in [1, 12] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for writer in [4093, 4095] {
                let mut facts = vec![CycleFacts::default(); 1 << 14];
                facts[writer] = CycleFacts {
                    bytecode_index: 1,
                    rd_post_value: 5,
                    ..CycleFacts::default()
                };
                facts[4094] = CycleFacts {
                    bytecode_index: 3,
                    rs2_value: if writer == 4093 { 5 } else { 0 },
                    ram_pre_value: 9,
                    ram_post_value: 7,
                    ..CycleFacts::default()
                };
                facts[4096] = CycleFacts {
                    bytecode_index: 2,
                    rs1_value: 5,
                    rd_post_value: 5,
                    ..CycleFacts::default()
                };
                facts[8192] = CycleFacts {
                    bytecode_index: 4,
                    ram_pre_value: 7,
                    rd_post_value: 7,
                    ..CycleFacts::default()
                };
                let witness = pool
                    .install(|| {
                        Rv64iWitness::from_facts(
                            layout.clone(),
                            Arc::clone(&bytecode),
                            &facts,
                            vec![(0, 9), (1, 11)],
                        )
                    })
                    .unwrap();
                assert_eq!(witness.words[4096].rs1_value, 5);
                assert_eq!(witness.words[4094].ram_read_value, 9);
                assert_eq!(witness.words[4095].ram_read_value, 7);
                assert_eq!(witness.words[8192].ram_read_value, 7);
                assert_eq!(witness.words[8193].rd_pre_value, 0);
                assert_eq!(witness.final_ram[0], 7);
                assert_eq!(witness.final_ram[1], 11);
                assert_eq!(witness.words[4095].next_pc, 8);
                for (cycle, field, expected, found) in [
                    (4096, FactField::Rs1Value, 5, 6),
                    (8192, FactField::RamPreValue, 7, 8),
                ] {
                    let mut changed = facts.clone();
                    changed[16383].bytecode_index = u32::MAX;
                    match field {
                        FactField::Rs1Value => changed[cycle].rs1_value = found,
                        FactField::RamPreValue => changed[cycle].ram_pre_value = found,
                        _ => unreachable!(),
                    }
                    let result = pool.install(|| {
                        Rv64iWitness::from_facts(
                            layout.clone(),
                            Arc::clone(&bytecode),
                            &changed,
                            vec![(0, 9), (1, 11)],
                        )
                    });
                    assert!(
                        matches!(result, Err(Rv64iProverError::FactMismatch { cycle: actual_cycle, field: actual_field, expected: actual_expected, found: actual_found }) if (actual_cycle, actual_field, actual_expected, actual_found) == (cycle, field, expected, found))
                    );
                }
            }
            let mut bits = vec![[0; 4]; 1 << 14];
            for (cycle, index, inc) in [(4093, 1, 5), (4094, 3, 14), (4096, 2, 5), (8192, 4, 7)] {
                layout
                    .write_bytecode_index(&mut bits[cycle], index)
                    .unwrap();
                bits[cycle][0] = inc;
            }
            bits[8][0] = 3;
            layout.write_ram_index(&mut bits[4097], 1).unwrap();
            let witness = pool
                .install(|| {
                    Rv64iWitness::from_bits(
                        layout.clone(),
                        Arc::clone(&bytecode),
                        bits.into(),
                        vec![(0, 9), (1, 11)],
                        0,
                    )
                })
                .unwrap();
            assert_eq!(witness.words[9].rd_pre_value, 3);
            assert_eq!(witness.words[4096].rs1_value, 5);
            assert_eq!(witness.words[4097].ram_read_value, 11);
            assert_eq!(witness.words[8192].ram_read_value, 7);
            assert_eq!(witness.final_ram[0], 7);
            assert_eq!(witness.final_ram[1], 11);
        }
    }

    #[test]
    fn sparse_ram_writer_at_chunk_boundary_keeps_first_fault() {
        let layout = Layout::new(2, 5, 0).unwrap();
        let instructions: Vec<_> = [0x0000_006f, 0x0000_3423, 0x0080_3183]
            .into_iter()
            .enumerate()
            .map(|(index, word)| decode_instruction(word, 4 * index as u64, false, RV64I).unwrap())
            .collect();
        let bytecode = Arc::new(Bytecode::preprocess(&instructions, &layout).unwrap());
        for threads in [1, 12] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut facts = vec![CycleFacts::default(); 1 << 14];
            facts[4095] = CycleFacts {
                bytecode_index: 1,
                ram_word_index: 1,
                ram_pre_value: 11,
                ram_post_value: 7,
                ..CycleFacts::default()
            };
            facts[4096] = CycleFacts {
                bytecode_index: 2,
                ram_word_index: 1,
                ram_pre_value: 7,
                rd_post_value: 7,
                ..CycleFacts::default()
            };
            let witness = pool
                .install(|| {
                    Rv64iWitness::from_facts(
                        layout.clone(),
                        Arc::clone(&bytecode),
                        &facts,
                        vec![(0, 9), (1, 11)],
                    )
                })
                .unwrap();
            assert_eq!(witness.words[4095].ram_read_value, 11);
            assert_eq!(witness.words[4096].ram_read_value, 7);
            assert_eq!(witness.words[4097].ram_read_value, 9);
            assert_eq!(witness.final_ram[..2], [9, 7]);
            facts[4096].ram_pre_value = 8;
            facts[16383].bytecode_index = u32::MAX;
            let result = pool.install(|| {
                Rv64iWitness::from_facts(
                    layout.clone(),
                    Arc::clone(&bytecode),
                    &facts,
                    vec![(0, 9), (1, 11)],
                )
            });
            assert!(matches!(
                result,
                Err(Rv64iProverError::FactMismatch {
                    cycle: 4096,
                    field: FactField::RamPreValue,
                    expected: 7,
                    found: 8
                })
            ));
        }
    }
}

#[cfg(test)]
#[path = "replay_tests.rs"]
mod replay_tests;
