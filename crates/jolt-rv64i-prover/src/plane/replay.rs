//! Chunk transition summaries, boundary prefix and authoritative local replay.

use super::{CycleWords, DecodedCycle, DigitFields, Rv64iWitness};
use crate::error::{FactField, Rv64iProverError};
use jolt_rv64i_arith::decode::SourceParts;
use jolt_rv64i_arith::{
    BitsBuilder, BitsRow, Bytecode, BytecodeRow, CycleError, CycleFacts, Layout, Variant,
};
use jolt_rv64i_kernels::par::CycleChunks;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
    ParallelSlice, ParallelSliceMut,
};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

impl Rv64iWitness {
    #[expect(
        clippy::expect_used,
        reason = "prepare_with_ram checked nonzero power-of-two witness geometry"
    )]
    #[inline(never)]
    pub(super) fn replay(&mut self, facts: Option<&[CycleFacts]>) -> Result<(), Rv64iProverError> {
        let context = ReplayContext {
            layout: &self.layout,
            bytecode: &self.bytecode,
        };
        let chunk_len = CycleChunks::new(self.bits.len().ilog2() as usize, 0)
            .expect("checked witness geometry")
            .chunk_len();
        if rayon::current_num_threads() == 1 || chunk_len == self.bits.len() {
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
}

type ReplayFault = (usize, Rv64iProverError);

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
#[path = "../replay_tests.rs"]
mod replay_tests;
