use std::collections::BTreeMap;
use std::sync::Arc;

use common::{constants::RAM_START_ADDRESS, jolt_device::MemoryConfig};
use jolt_program::{
    execution::{OwnedTrace, RamAccess, RegisterState, SourceTraceRow, TraceOutput},
    image::{decode_elf_with_mode, DecodeMode},
};
use jolt_riscv::RV64I;
use jolt_rv64i_arith::{RowSystem, WitnessRow};
use jolt_rv64i_prover::{
    commitment::transparent::TransparentBits, error::Rv64iProverError, plane::Rv64iWitness,
};
use jolt_rv64i_trace::{adapt, preprocess, trace, AdapterError, Execution};
use jolt_rv64i_verifier::{
    preprocessing::VerifierPreprocessing,
    statement::{CheckedInputs, Statement},
};
use tracer::emulator::elf_analyzer::test_elf::{build_elf64, StrtabOrder};

#[expect(
    dead_code,
    reason = "the maintained encoder also serves instruction kinds outside these fixtures"
)]
#[path = "../../../jolt-rv64i-arith/tests/suite/common/asm.rs"]
pub mod asm;

pub const ENTRY: u64 = RAM_START_ADDRESS;
pub const DATA: u64 = ENTRY + 0x1000;

pub struct Fixture {
    pub words: Vec<u32>,
    pub inputs: Vec<u8>,
    pub mode: DecodeMode,
    pub config: MemoryConfig,
}

impl Fixture {
    pub fn new(words: Vec<u32>) -> Self {
        Self {
            words,
            inputs: Vec::new(),
            mode: DecodeMode::Strict,
            config: MemoryConfig {
                max_input_size: 8,
                max_output_size: 8,
                max_trusted_advice_size: 0,
                max_untrusted_advice_size: 0,
                stack_size: 256,
                heap_size: 0x2000,
                program_size: None,
            },
        }
    }

    pub fn elf(&self) -> Vec<u8> {
        build_elf64(&self.words, &[], StrtabOrder::GnuLd)
    }

    pub fn prepare(&self) -> Prepared {
        let elf = self.elf();
        let program = preprocess(&elf, self.config, self.mode).unwrap();
        let output = trace(&elf, &self.inputs, &program.memory_config, self.mode).unwrap();
        let entry_pc = program.entry_pc;
        let preprocessing =
            VerifierPreprocessing::new(program.bytecode, program.image, ()).unwrap();
        let decoded = decode_elf_with_mode(&elf, RV64I, self.mode).unwrap();
        for row in output.trace.rows() {
            let instruction = &decoded.instructions[row.instruction_index() as usize];
            let operands = instruction.row().operands;
            let bytecode = &preprocessing.bytecode().rows()[row.instruction_index() as usize];
            assert_eq!(row.pc(), instruction.row().address as u64);
            assert_eq!(row.registers().rs1.map(|read| read.register), operands.rs1);
            assert_eq!(row.registers().rs2.map(|read| read.register), operands.rs2);
            assert_eq!(row.registers().rd.map(|write| write.register), operands.rd);
            assert_eq!(bytecode.rs1, operands.rs1.unwrap_or(0));
            assert_eq!(bytecode.rs2, operands.rs2.unwrap_or(0));
            assert_eq!(bytecode.rd, operands.rd.unwrap_or(0));
            assert_eq!(
                bytecode.variant.unwrap().access().is_some(),
                !matches!(row.ram_access(), RamAccess::NoOp)
            );
        }
        Prepared {
            preprocessing,
            output,
            entry_pc,
        }
    }

    pub fn complete(&self) -> Completed {
        let prepared = self.prepare();
        let execution = prepared.adapt(prepared.output.trace.rows()).unwrap();
        let witness = prepared.witness(&execution).unwrap();
        let completed = Completed {
            prepared,
            execution,
            witness,
        };
        completed
            .prepared
            .check(&completed.execution, &completed.witness);
        completed
    }
}

pub struct Prepared {
    pub preprocessing: VerifierPreprocessing<TransparentBits>,
    pub output: TraceOutput<OwnedTrace<SourceTraceRow>>,
    pub entry_pc: u64,
}

impl Prepared {
    pub fn adapt(&self, rows: &[SourceTraceRow]) -> Result<Execution, AdapterError> {
        adapt(
            self.preprocessing.bytecode(),
            self.preprocessing.image(),
            &self.output.device.memory_layout,
            self.entry_pc,
            rows,
        )
    }

    pub fn statement(&self, execution: &Execution) -> Statement {
        Statement {
            log_T: execution.log_T(),
            entry_pc: self.entry_pc,
            device: self.output.device.clone(),
        }
    }

    pub fn witness(&self, execution: &Execution) -> Result<Rv64iWitness, Rv64iProverError> {
        let statement = self.statement(execution);
        let checked = CheckedInputs::of_statement(
            &self.preprocessing,
            &statement,
            execution.log_K_ram,
            execution.final_pc(),
        )?;
        Rv64iWitness::from_facts(
            checked.layout().clone(),
            Arc::clone(self.preprocessing.shared_bytecode()),
            &execution.facts,
            checked.initial_ram().to_vec(),
        )
    }

    pub fn check(&self, execution: &Execution, witness: &Rv64iWitness) {
        for (source, fact) in self.output.trace.rows().iter().zip(&execution.facts) {
            assert_eq!(
                fact.bytecode_index as usize,
                self.preprocessing
                    .bytecode()
                    .index_of_pc(source.pc())
                    .unwrap()
            );
        }
        let system = RowSystem::new(&witness.layout);
        for (cycle, bits) in witness.bits.iter().enumerate() {
            let index = witness.layout.bytecode_index(bits) as usize;
            let row = &witness.bytecode.rows()[index];
            let base = witness.words[cycle]
                .base_words(row.variant.unwrap().is_store(), witness.layout.inc(bits));
            let checked = system.check(&WitnessRow::compute(&witness.layout, row, &base, bits));
            assert!(
                checked.is_ok(),
                "cycle {cycle} at {:#x}: {checked:?}",
                row.pc
            );
            assert_eq!(
                execution.facts[cycle].bytecode_index as usize,
                witness.bytecode.index_of_pc(row.pc).unwrap()
            );
        }
        let memory: BTreeMap<_, _> = self
            .output
            .final_memory
            .as_ref()
            .unwrap()
            .bytes
            .iter()
            .map(|&(offset, byte)| (ENTRY + offset, byte))
            .collect();
        let layout = &self.output.device.memory_layout;
        let mask_end = layout.remapped_word_address(ENTRY).unwrap() as usize;
        for (index, &word) in witness.final_ram.iter().enumerate().skip(mask_end) {
            let address = witness.layout.lowest_address() + 8 * index as u64;
            let expected = u64::from_le_bytes(std::array::from_fn(|offset| {
                memory.get(&(address + offset as u64)).copied().unwrap_or(0)
            }));
            assert_eq!(word, expected, "final RAM word {index}");
        }
        for &address in memory.keys() {
            assert!(
                (address - witness.layout.lowest_address()) / 8 < witness.final_ram.len() as u64
            );
        }
    }
}

pub struct Completed {
    pub prepared: Prepared,
    pub execution: Execution,
    pub witness: Rv64iWitness,
}

pub fn address(words: &mut Vec<u32>, rd: u8, address: u64) {
    let upper = (address + 0x800) & !0xfff;
    words.extend([
        asm::lui(rd, (upper as u32 as i32) >> 12),
        asm::slli(rd, rd, 32),
        asm::srli(rd, rd, 32),
        asm::addi(rd, rd, (address as i64 - upper as i64) as i32),
    ]);
}

pub fn replace(
    row: SourceTraceRow,
    index: u32,
    registers: RegisterState,
    ram: RamAccess,
) -> SourceTraceRow {
    SourceTraceRow::new(index, row.pc(), row.next_pc(), registers, ram)
}
