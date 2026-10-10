//! Interpreted program fixtures and transcript records for the complete RV64I protocol.
#![expect(
    non_snake_case,
    reason = "protocol dimensions follow mathematical notation"
)]
#![expect(
    clippy::unwrap_used,
    reason = "fixture failures fail the enclosing test"
)]

use self::replay::State;
use common::{
    constants::RAM_START_ADDRESS,
    jolt_device::{JoltDevice, MemoryConfig, MemoryLayout},
};
use jolt_field::F128;
use jolt_rv64i_arith::{CycleFacts, Layout};
use jolt_rv64i_prover::{commitment::transparent::TransparentBits, plane::Rv64iWitness};
use jolt_rv64i_verifier::{
    preprocessing::VerifierPreprocessing,
    statement::{CheckedInputs, Statement},
    transcript::Rv64iTranscript,
};
use jolt_transcript::{AppendToTranscript, Label, Transcript};
use std::sync::Arc;

#[path = "../../../jolt-rv64i-arith/tests/suite/common/asm.rs"]
pub mod asm;
#[path = "../../../jolt-rv64i-arith/tests/suite/common/harness.rs"]
pub mod harness;
#[expect(
    dead_code,
    reason = "the shared interpreter exposes setup and execution helpers for the whole corpus"
)]
#[path = "../../../jolt-rv64i-arith/tests/suite/common/interp.rs"]
pub mod interp;
#[path = "../../../jolt-rv64i-arith/tests/suite/common/replay.rs"]
pub mod replay;

#[derive(Clone, Copy, Debug)]
pub enum Program {
    CountingLoop,
    CountingLoopWithoutTermination,
    ByteCopy,
    CallsReturns,
    ShiftXor,
    BranchLadder,
}
pub const PROGRAMS: [Program; 5] = [
    Program::CountingLoop,
    Program::ByteCopy,
    Program::CallsReturns,
    Program::ShiftXor,
    Program::BranchLadder,
];
impl Program {
    pub fn name(self) -> &'static str {
        match self {
            Self::CountingLoop => "counting_loop",
            Self::CountingLoopWithoutTermination => "counting_loop_without_termination",
            Self::ByteCopy => "byte_copy",
            Self::CallsReturns => "calls_returns",
            Self::ShiftXor => "shift_xor",
            Self::BranchLadder => "branch_ladder",
        }
    }
    fn words(self) -> Vec<u32> {
        match self {
            Self::CountingLoop => vec![
                asm::auipc(2, 0),
                asm::addi(2, 2, -8),
                asm::addi(1, 0, 0),
                asm::addi(3, 0, 3),
                asm::addi(1, 1, 1),
                asm::blt(1, 3, -4),
                asm::addi(4, 0, 1),
                asm::sd(2, 4, 0),
                asm::jal(0, 0),
            ],
            Self::CountingLoopWithoutTermination => {
                let mut words = Self::CountingLoop.words();
                words[7] = asm::sd(2, 4, -8);
                words
            }
            Self::ByteCopy => vec![
                asm::auipc(2, 0),
                asm::addi(2, 2, -8),
                asm::addi(5, 2, -24),
                asm::addi(6, 2, -16),
                asm::lb(1, 5, 0),
                asm::sb(6, 1, 0),
                asm::lh(1, 5, 0),
                asm::sh(6, 1, 0),
                asm::lw(1, 5, 0),
                asm::sw(6, 1, 0),
                asm::ld(1, 5, 0),
                asm::sd(6, 1, 0),
                asm::lbu(7, 5, 0),
                asm::lhu(8, 5, 0),
                asm::lwu(9, 5, 0),
                asm::lb(0, 5, 0),
                asm::lh(0, 5, 0),
                asm::lw(0, 5, 0),
                asm::ld(0, 5, 0),
                asm::addi(4, 0, 1),
                asm::sd(2, 4, 0),
                asm::jal(0, 0),
            ],
            Self::CallsReturns => vec![
                asm::auipc(2, 0),
                asm::addi(2, 2, -8),
                asm::addi(1, 0, 5),
                asm::addi(3, 0, 2),
                asm::jal(5, 12),
                asm::auipc(7, 0),
                asm::jalr(6, 7, 64),
                asm::add(10, 1, 3),
                asm::sub(11, 1, 3),
                asm::addw(12, 1, 3),
                asm::addiw(13, 1, -2),
                asm::subw(14, 1, 3),
                asm::and(15, 1, 3),
                asm::andi(16, 1, 3),
                asm::or(17, 1, 3),
                asm::ori(18, 1, 8),
                asm::xor(19, 1, 3),
                asm::xori(20, 1, 7),
                asm::lui(21, 1),
                asm::fence(0, 0),
                asm::jalr(0, 5, 0),
                asm::addi(4, 0, 1),
                asm::sd(2, 4, 0),
                asm::jal(0, 0),
            ],
            Self::ShiftXor => vec![
                asm::auipc(2, 0),
                asm::addi(2, 2, -8),
                asm::addi(1, 0, -9),
                asm::addi(3, 0, 3),
                asm::sll(4, 1, 3),
                asm::xor(1, 1, 4),
                asm::srl(4, 1, 3),
                asm::xor(1, 1, 4),
                asm::sra(4, 1, 3),
                asm::xor(1, 1, 4),
                asm::slli(4, 1, 5),
                asm::srli(5, 4, 2),
                asm::srai(6, 1, 4),
                asm::sllw(7, 1, 3),
                asm::srlw(8, 1, 3),
                asm::sraw(9, 1, 3),
                asm::slliw(10, 1, 2),
                asm::srliw(11, 1, 2),
                asm::sraiw(12, 1, 2),
                asm::xor(1, 5, 12),
                asm::addi(4, 0, 1),
                asm::sd(2, 4, 0),
                asm::ecall(),
            ],
            Self::BranchLadder => vec![
                asm::auipc(2, 0),
                asm::addi(2, 2, -8),
                asm::addi(1, 0, -1),
                asm::addi(3, 0, 1),
                asm::slt(4, 1, 3),
                asm::slti(5, 1, 1),
                asm::sltu(6, 1, 3),
                asm::sltiu(7, 3, -1),
                asm::beq(1, 3, 8),
                asm::addi(0, 0, 0),
                asm::bne(1, 3, 8),
                asm::addi(0, 0, 0),
                asm::blt(1, 3, 8),
                asm::addi(0, 0, 0),
                asm::bge(1, 3, 8),
                asm::addi(0, 0, 0),
                asm::bltu(1, 3, 8),
                asm::addi(0, 0, 0),
                asm::bgeu(1, 3, 8),
                asm::addi(0, 0, 0),
                asm::addi(4, 0, 1),
                asm::sd(2, 4, 0),
                asm::ebreak(),
            ],
        }
    }
}

pub fn counting_loop() -> (
    Statement,
    VerifierPreprocessing<TransparentBits>,
    Rv64iWitness,
) {
    counting_loop_at(6)
}
pub fn counting_loop_at(
    log_T: u8,
) -> (
    Statement,
    VerifierPreprocessing<TransparentBits>,
    Rv64iWitness,
) {
    program_fixture(Program::CountingLoop, log_T)
}
pub fn counting_loop_facts() -> Vec<CycleFacts> {
    program_fixture_with_facts(Program::CountingLoop, 6).3
}
pub fn program_fixture(
    program: Program,
    log_T: u8,
) -> (
    Statement,
    VerifierPreprocessing<TransparentBits>,
    Rv64iWitness,
) {
    let (statement, preprocessing, witness, _) = program_fixture_with_facts(program, log_T);
    (statement, preprocessing, witness)
}
fn program_fixture_with_facts(
    program_kind: Program,
    log_T: u8,
) -> (
    Statement,
    VerifierPreprocessing<TransparentBits>,
    Rv64iWitness,
    Vec<CycleFacts>,
) {
    let words = program_kind.words();
    let program_size = (words.len() * 4).next_multiple_of(8) as u64;
    let memory_layout = MemoryLayout::try_new(&MemoryConfig {
        max_input_size: 8,
        max_output_size: 8,
        max_trusted_advice_size: 0,
        max_untrusted_advice_size: 0,
        stack_size: 0,
        heap_size: 0,
        program_size: Some(program_size),
    })
    .unwrap();
    let b = words.len().next_power_of_two().trailing_zeros() as usize;
    let image_end = 4 + words.len().div_ceil(2);
    let a = 5_usize.max(image_end.next_power_of_two().trailing_zeros() as usize);
    let layout = Layout::new(b, a, memory_layout.get_lowest_address()).unwrap();
    let program: Vec<_> = words
        .iter()
        .enumerate()
        .map(|(i, word)| (RAM_START_ADDRESS + 4 * i as u64, *word))
        .collect();
    let bytecode = Arc::new(harness::bytecode(&program, &layout));
    let device = JoltDevice {
        inputs: if matches!(program_kind, Program::ByteCopy) {
            vec![0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0]
        } else {
            vec![0x12, 0x34]
        },
        outputs: vec![],
        panic: matches!(program_kind, Program::CountingLoopWithoutTermination),
        memory_layout,
        ..JoltDevice::default()
    };
    let mut statement = Statement {
        log_T,
        entry_pc: RAM_START_ADDRESS,
        device,
    };
    let image: Vec<_> = words
        .chunks(2)
        .enumerate()
        .map(|(i, pair)| {
            let value = u64::from(pair[0]) | (u64::from(pair.get(1).copied().unwrap_or(0)) << 32);
            (4 + i as u64, value)
        })
        .collect();
    let preprocessing = VerifierPreprocessing::new(Arc::clone(&bytecode), image, ()).unwrap();
    let last_pc = program.last().unwrap().0;
    let initial_ram = CheckedInputs::of_statement(&preprocessing, &statement, a as u8, last_pc)
        .unwrap()
        .initial_ram()
        .to_vec();
    let mut initial = State::new(RAM_START_ADDRESS);
    for &(index, value) in &initial_ram {
        initial.set_ram_word(index, value);
    }
    let mut machine = harness::machine(&program, &layout, &initial);
    let facts: Vec<_> = (0..(1_usize << log_T))
        .map(|_| {
            let record = machine.step().unwrap();
            harness::facts(&record, bytecode.index_of_pc(record.pc).unwrap())
        })
        .collect();
    if matches!(program_kind, Program::ByteCopy) {
        statement.device.outputs = machine.ram_word(1).unwrap().to_le_bytes().to_vec();
        assert_eq!(statement.device.outputs, statement.device.inputs);
    }
    assert_eq!(machine.pc(), last_pc);
    assert_eq!(
        machine.ram_word(3).unwrap(),
        u64::from(!matches!(
            program_kind,
            Program::CountingLoopWithoutTermination
        ))
    );
    assert_eq!(
        machine.ram_word(2).unwrap(),
        u64::from(statement.device.panic)
    );
    if matches!(program_kind, Program::CountingLoop) {
        assert_eq!(machine.registers()[1], 3);
    }
    let witness = Rv64iWitness::from_facts(layout, bytecode, &facts, initial_ram).unwrap();
    let trace = replay::replay(
        &witness.layout,
        &witness.bytecode,
        &witness.bits,
        &initial,
        witness.final_pc,
    )
    .unwrap();
    assert_eq!(trace.final_state.pc, machine.pc());
    for (index, word) in witness.final_ram.iter().enumerate() {
        assert_eq!(
            *word,
            trace
                .final_state
                .ram
                .get(&(index as u64))
                .copied()
                .unwrap_or(0)
        );
    }
    (statement, preprocessing, witness, facts)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Append(Vec<u8>),
    Challenge(F128),
    Scalar(F128),
}
#[derive(Default)]
pub struct RecordedTranscript {
    inner: Rv64iTranscript,
    pub events: Vec<Event>,
    pub batch_states: Vec<[u8; 32]>,
    pub batch_event_ends: Vec<usize>,
    pub openings: usize,
}
impl RecordedTranscript {
    pub fn inner_at(&self, event_end: usize) -> Rv64iTranscript {
        let mut inner = Rv64iTranscript::new(b"jolt-rv64i-binary-v0");
        for event in self.events.iter().take(event_end) {
            match event {
                Event::Append(bytes) => inner.append_bytes(bytes),
                Event::Challenge(value) => assert_eq!(inner.challenge(), *value),
                Event::Scalar(value) => assert_eq!(inner.challenge_scalar(), *value),
            }
        }
        inner
    }
    pub fn challenge_count(&self) -> usize {
        self.events
            .iter()
            .filter(|event| matches!(event, Event::Challenge(_) | Event::Scalar(_)))
            .count()
    }
    pub fn label_count(&self, label: &[u8]) -> usize {
        self.events.iter().filter(|event| matches!(event, Event::Append(bytes) if bytes.len() == 32 && bytes.starts_with(label) && bytes.get(label.len()..24).is_some_and(|padding| padding.iter().all(|byte| *byte == 0)))).count()
    }
    pub fn fork(&self) -> Self {
        Self {
            inner: self.inner_at(self.events.len()),
            events: self.events.clone(),
            batch_states: self.batch_states.clone(),
            batch_event_ends: self.batch_event_ends.clone(),
            openings: self.openings,
        }
    }
}
impl Transcript for RecordedTranscript {
    type Challenge = F128;
    fn new(label: &'static [u8]) -> Self {
        Self {
            inner: Rv64iTranscript::new(label),
            ..Self::default()
        }
    }
    fn append_bytes(&mut self, bytes: &[u8]) {
        self.inner.append_bytes(bytes);
        self.events.push(Event::Append(bytes.to_vec()));
    }
    fn append_labeled<A: AppendToTranscript>(&mut self, label: &'static [u8], value: &A) {
        self.append(&Label(label));
        self.append(value);
        if label == b"opening_claim" {
            self.openings += 1;
            if [6, 8, 13, 31, 38, 42, 43, 299].contains(&self.openings) {
                self.batch_states.push(self.inner.state());
                self.batch_event_ends.push(self.events.len());
            }
        }
    }
    fn challenge(&mut self) -> F128 {
        let value = self.inner.challenge();
        self.events.push(Event::Challenge(value));
        value
    }
    fn challenge_scalar(&mut self) -> F128 {
        let value = self.inner.challenge_scalar();
        self.events.push(Event::Scalar(value));
        value
    }
    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
}
