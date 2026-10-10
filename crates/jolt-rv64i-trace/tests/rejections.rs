#![cfg(feature = "emulator")]
#![expect(
    clippy::unwrap_used,
    reason = "fixture failures fail the enclosing test"
)]

mod support;

use std::sync::Arc;

use jolt_program::{
    execution::{RamAccess, RamRead, SourceTraceError, TraceError},
    image::{decode_elf_with_mode, DecodeMode},
    ProgramError,
};
use jolt_riscv::{SourceInstructionKind as Kind, RV64I};
use jolt_rv64i_arith::{Bytecode, Layout, WitnessError};
use jolt_rv64i_prover::{
    commitment::transparent::TransparentBits,
    error::{FactField, Rv64iProverError},
    plane::Rv64iWitness,
};
use jolt_rv64i_trace::{adapt, preprocess, trace, AdapterError};
use jolt_rv64i_verifier::{preprocessing::VerifierPreprocessing, statement::CheckedInputs};
use support::{address, asm, replace, Fixture, DATA, ENTRY};

#[test]
fn identity_rejections_are_returned_by_adapt() {
    let fixture = Fixture::new(vec![asm::addi(1, 0, 1), asm::addi(2, 1, 2), asm::jal(0, 0)]);
    let prepared = fixture.prepare();
    let mut rows = prepared.output.trace.rows().to_vec();
    let original = rows[1];
    for index in [0, 3, prepared.preprocessing.bytecode().rows().len() as u32] {
        rows[1] = replace(original, index, original.registers(), original.ram_access());
        assert!(matches!(prepared.adapt(&rows),
            Err(AdapterError::InstructionIndex { cycle: 1, pc, index: found })
            if pc == ENTRY + 4 && found == index));
    }
    let mut image = decode_elf_with_mode(&fixture.elf(), RV64I, DecodeMode::Strict).unwrap();
    let _ = image.instructions.remove(0);
    let bytecode = Bytecode::preprocess(
        &image.instructions,
        &Layout::new(2, 5, prepared.preprocessing.bytecode().lowest_address()).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        adapt(
            &bytecode,
            prepared.preprocessing.image(),
            &prepared.output.device.memory_layout,
            ENTRY,
            prepared.output.trace.rows()
        ),
        Err(AdapterError::InstructionIndex {
            cycle: 0,
            pc: ENTRY,
            index: 0
        })
    ));
}

#[test]
fn ram_address_and_initial_word_rejections_name_the_conversion_or_replay_stage() {
    let mut words = Vec::new();
    address(&mut words, 5, DATA);
    words.extend([asm::ld(6, 5, 0), asm::jal(0, 0)]);
    let prepared = Fixture::new(words).prepare();
    let mut rows = prepared.output.trace.rows().to_vec();
    let original = rows[4];
    let lowest = prepared.preprocessing.bytecode().lowest_address();
    rows[4] = replace(
        original,
        original.instruction_index(),
        original.registers(),
        RamAccess::Read(RamRead {
            address: lowest - 8,
            value: 0,
        }),
    );
    assert!(matches!(prepared.adapt(&rows),
        Err(AdapterError::AddressBelowRam { cycle: 4, address }) if address == lowest - 8));
    rows[4] = replace(
        original,
        original.instruction_index(),
        original.registers(),
        RamAccess::Read(RamRead {
            address: DATA + 4,
            value: 0,
        }),
    );
    assert!(matches!(prepared.adapt(&rows),
        Err(AdapterError::UnalignedWordAddress { cycle: 4, address }) if address == DATA + 4));
    rows[4] = replace(
        original,
        original.instruction_index(),
        original.registers(),
        RamAccess::Read(RamRead {
            address: DATA + 8,
            value: 0,
        }),
    );
    let execution = prepared.adapt(&rows).unwrap();
    assert!(matches!(prepared.witness(&execution),
    Err(Rv64iProverError::Cycle(error)) if error.cycle == 4 &&
    error.error == WitnessError::RamWordIndexMismatch {
        expected: (DATA - lowest) / 8, found: (DATA + 8 - lowest) / 8,
    }));

    let mut image_words = vec![asm::auipc(5, 0), asm::ld(6, 5, 16), asm::jal(0, 0), 0];
    image_words.extend([0, 0]);
    let image_prepared = Fixture::new(image_words).prepare();
    let execution = image_prepared
        .adapt(image_prepared.output.trace.rows())
        .unwrap();
    let mut image = image_prepared.preprocessing.image().to_vec();
    let word_index = (ENTRY + 16 - image_prepared.preprocessing.bytecode().lowest_address()) / 8;
    assert!(!image.iter().any(|&(index, _)| index == word_index));
    image.push((word_index, 0x80));
    image.sort_unstable_by_key(|&(index, _)| index);
    let changed = VerifierPreprocessing::<TransparentBits>::new(
        Arc::clone(image_prepared.preprocessing.shared_bytecode()),
        image,
        (),
    )
    .unwrap();
    let statement = image_prepared.statement(&execution);
    let checked = CheckedInputs::of_statement(
        &changed,
        &statement,
        execution.log_K_ram,
        execution.final_pc(),
    )
    .unwrap();
    assert!(matches!(
        Rv64iWitness::from_facts(
            checked.layout().clone(),
            Arc::clone(changed.shared_bytecode()),
            &execution.facts,
            checked.initial_ram().to_vec()
        ),
        Err(Rv64iProverError::FactMismatch {
            cycle: 1,
            field: FactField::RamPreValue,
            expected: 0x80,
            found: 0,
        })
    ));
}

#[test]
fn missing_branch_operand_is_rejected_by_from_facts() {
    let prepared = Fixture::new(vec![
        asm::addi(1, 0, 7),
        asm::addi(2, 0, 7),
        asm::beq(1, 2, 8),
        asm::addi(3, 0, 1),
        asm::jal(0, 0),
    ])
    .prepare();
    let mut rows = prepared.output.trace.rows().to_vec();
    let original = rows[2];
    let mut registers = original.registers();
    registers.rs2 = None;
    rows[2] = replace(
        original,
        original.instruction_index(),
        registers,
        original.ram_access(),
    );
    let execution = prepared.adapt(&rows).unwrap();
    assert!(matches!(
        prepared.witness(&execution),
        Err(Rv64iProverError::FactMismatch {
            cycle: 2,
            field: FactField::Rs2Value,
            expected: 7,
            found: 0,
        })
    ));
}

#[test]
fn entry_empty_and_missing_stall_are_adapt_errors() {
    let fixture = Fixture::new(vec![
        asm::addi(1, 0, 9),
        asm::addi(1, 1, -1),
        asm::bne(1, 0, -4),
        asm::jal(0, 0),
    ]);
    let prepared = fixture.prepare();
    let rows = prepared.output.trace.rows();
    assert!(rows.len() > 16);
    assert!(
        matches!(adapt(prepared.preprocessing.bytecode(), prepared.preprocessing.image(),
        &prepared.output.device.memory_layout, ENTRY + 4, rows),
        Err(AdapterError::EntryPc { cycle: 0, pc: ENTRY, entry_pc }) if entry_pc == ENTRY + 4)
    );
    assert!(matches!(prepared.adapt(&[]), Err(AdapterError::EmptyTrace)));
    assert!(matches!(prepared.adapt(&rows[..15]),
        Err(AdapterError::NoStall { cycle: 14, pc }) if pc == rows[14].pc()));
    let exact = prepared.adapt(&rows[..16]).unwrap();
    assert_eq!(exact.facts.len(), 16);
    assert_eq!(exact.final_pc(), rows[15].next_pc());
    let witness = prepared.witness(&exact).unwrap();
    prepared.check(&exact, &witness);
}

#[test]
fn instruction_set_rejections_name_preprocess_or_trace() {
    for (word, kind) in [
        (0x0220_81b3, Kind::MUL),
        (0x1000_b1af, Kind::LRD),
        (0x0010_9173, Kind::CSRRW),
    ] {
        let fixture = Fixture::new(vec![word, asm::jal(0, 0)]);
        assert!(
            matches!(preprocess(&fixture.elf(), fixture.config, DecodeMode::Strict),
            Err(AdapterError::Program(ProgramError::IllegalSourceInstruction(found))) if found == kind)
        );
    }
    let malformed = Fixture::new(vec![0x0000_007f, asm::jal(0, 0)]);
    assert!(matches!(
        preprocess(&malformed.elf(), malformed.config, DecodeMode::Strict),
        Err(AdapterError::Program(ProgramError::MalformedImage(_)))
    ));
    let compressed = Fixture::new(vec![0x0000_0085, asm::jal(0, 0)]);
    assert!(matches!(
        preprocess(&compressed.elf(), compressed.config, DecodeMode::Strict),
        Err(AdapterError::Program(
            ProgramError::IllegalCompressedInstruction { address: ENTRY }
        ))
    ));
    let misaligned = Fixture::new(vec![0x0093_0000, 0x006f_0010, 0]);
    assert!(matches!(
        preprocess(&misaligned.elf(), misaligned.config, DecodeMode::Strict),
        Err(AdapterError::Program(ProgramError::MalformedImage(_)))
    ));
    for (word, kind) in [(asm::ecall(), Kind::ECALL), (asm::ebreak(), Kind::EBREAK)] {
        let reached = Fixture::new(vec![word, asm::jal(0, 0)]);
        let program = preprocess(&reached.elf(), reached.config, DecodeMode::Strict).unwrap();
        assert!(
            matches!(trace(&reached.elf(), &[], &program.memory_config, DecodeMode::Strict),
            Err(AdapterError::Trace(TraceError::SourceTrace(
                SourceTraceError::UnsupportedInstruction { pc: ENTRY, kind: found }))) if found == kind)
        );
        let _ = Fixture::new(vec![asm::jal(0, 8), word, asm::jal(0, 0)]).complete();
    }
    let mut unreached = Fixture::new(vec![asm::jal(0, 8), 0x0220_81b3, asm::jal(0, 0)]);
    unreached.mode = DecodeMode::DataHoles;
    let _ = unreached.complete();
    let mut reached = Fixture::new(vec![0x0220_81b3, asm::jal(0, 0)]);
    reached.mode = DecodeMode::DataHoles;
    let program = preprocess(&reached.elf(), reached.config, reached.mode).unwrap();
    assert!(matches!(
        trace(&reached.elf(), &[], &program.memory_config, reached.mode),
        Err(AdapterError::Trace(TraceError::SourceTrace(
            SourceTraceError::PcOutsideProgram { pc: ENTRY }
        )))
    ));
}
