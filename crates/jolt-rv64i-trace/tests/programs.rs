#![cfg(feature = "emulator")]
#![expect(
    clippy::unwrap_used,
    reason = "fixture failures fail the enclosing test"
)]

mod support;

use common::jolt_device::MemoryLayout;
use jolt_program::{
    execution::{RamAccess, RamRead, RegisterState},
    image::{decode::decode_instruction, DecodeMode},
};
use jolt_riscv::{SourceInstructionKind as Kind, RV64I};
use jolt_rv64i_arith::{Bytecode, Layout, LayoutError, MAX_LOG_K_BYTECODE};
use jolt_rv64i_trace::{adapt, preprocess, AdapterError};
use jolt_rv64i_verifier::{error::Rv64iVerifierError, statement::CheckedInputs};
use rayon::ThreadPoolBuilder;
use support::{address, asm, replace, Fixture, DATA, ENTRY};

const INPUT_WORD: u64 = 0xf8e7_d6c5_b4a3_9281;

type MemoryInstruction = fn(u8, u8, i32) -> u32;

#[test]
fn alu_alias_loop_crosses_power_of_two_lengths() {
    for count in [31, 32, 33] {
        let mut words = vec![
            asm::addi(5, 0, 3),
            asm::addi(2, 0, 1),
            asm::add(3, 2, 2),
            asm::add(3, 3, 2),
            asm::add(2, 3, 2),
            asm::add(2, 2, 2),
            asm::addi(5, 5, -1),
            asm::bne(5, 0, -20),
        ];
        words.extend(std::iter::repeat_n(asm::addi(0, 0, 0), count - 21));
        words.push(asm::jal(0, 0));
        let run = Fixture::new(words).complete();
        assert_eq!(run.prepared.output.trace.rows().len(), count);
        assert_eq!(run.execution.facts.len(), count.next_power_of_two());
    }
}

#[test]
fn loads_and_stores_cover_every_width_and_aligned_offset() {
    let mut words = Vec::new();
    address(&mut words, 5, DATA);
    address(&mut words, 6, ENTRY - 32);
    words.extend([asm::ld(7, 6, 0), asm::sd(5, 7, 0)]);
    let loads: [(MemoryInstruction, usize); 7] = [
        (asm::lb, 1),
        (asm::lbu, 1),
        (asm::lh, 2),
        (asm::lhu, 2),
        (asm::lw, 4),
        (asm::lwu, 4),
        (asm::ld, 8),
    ];
    for (load, width) in loads {
        for offset in (0..8).step_by(width) {
            words.push(load(8, 5, offset));
        }
        words.push(load(0, 5, 0));
    }
    for (store, width) in [
        (asm::sb as MemoryInstruction, 1),
        (asm::sh, 2),
        (asm::sw, 4),
        (asm::sd, 8),
    ] {
        for offset in (0..8).step_by(width) {
            words.push(store(5, 7, offset));
        }
    }
    words.push(asm::jal(0, 0));
    let mut fixture = Fixture::new(words);
    fixture.inputs = INPUT_WORD.to_le_bytes().to_vec();
    let _ = fixture.complete();
}

#[test]
fn calls_save_and_restore_a_stack_frame() {
    let mut words = vec![0; 4];
    words.extend([
        asm::addi(3, 0, 5),
        asm::jal(1, 8),
        asm::jal(0, 0),
        asm::addi(2, 2, -16),
        asm::sd(2, 1, 8),
        asm::sd(2, 3, 0),
        asm::addi(3, 3, 1),
        asm::ld(1, 2, 8),
        asm::ld(3, 2, 0),
        asm::addi(2, 2, 16),
        asm::jalr(0, 1, 0),
    ]);
    let fixture = Fixture::new(words.clone());
    let program = preprocess(&fixture.elf(), fixture.config, DecodeMode::Strict).unwrap();
    let layout = MemoryLayout::try_new(&program.memory_config).unwrap();
    let mut prologue = Vec::new();
    address(
        &mut prologue,
        2,
        (layout.stack_end + layout.stack_size) & !15,
    );
    words[..4].copy_from_slice(&prologue);
    let _ = Fixture::new(words).complete();
}

#[test]
fn input_output_and_termination_agree_with_public_memory() {
    let mut words = Vec::new();
    address(&mut words, 5, ENTRY - 32);
    address(&mut words, 6, ENTRY - 24);
    address(&mut words, 7, ENTRY - 8);
    words.extend([
        asm::ld(8, 5, 0),
        asm::sd(6, 8, 0),
        asm::addi(9, 0, 1),
        asm::sd(7, 9, 0),
        asm::jal(0, 0),
    ]);
    let mut fixture = Fixture::new(words);
    fixture.inputs = INPUT_WORD.to_le_bytes().to_vec();
    let run = fixture.complete();
    let statement = run.prepared.statement(&run.execution);
    let checked = CheckedInputs::of_statement(
        &run.prepared.preprocessing,
        &statement,
        run.execution.log_K_ram,
        run.execution.final_pc(),
    )
    .unwrap();
    run.witness.check_outputs(&checked).unwrap();
    assert_eq!(run.prepared.output.device.outputs, INPUT_WORD.to_le_bytes());
    assert_eq!(run.witness.layout.lowest_address(), ENTRY - 32);
    assert!(run.execution.facts.len() > run.prepared.output.trace.rows().len());
    for (facts, words) in run.execution.facts.iter().zip(run.witness.words.iter()) {
        let row = &run.witness.bytecode.rows()[facts.bytecode_index as usize];
        if row.variant.unwrap().access().is_none() {
            assert_eq!(
                (
                    facts.ram_word_index,
                    facts.ram_pre_value,
                    facts.ram_post_value
                ),
                (0, 0, 0)
            );
            assert_eq!(words.ram_read_value, INPUT_WORD);
        }
    }
}

#[test]
fn stall_forms_repeat_in_the_state_they_leave() {
    for word in [asm::jal(1, 0), asm::beq(0, 0, 0)] {
        let run = Fixture::new(vec![asm::addi(3, 0, 7), asm::addi(4, 0, 9), word]).complete();
        assert_eq!(run.prepared.output.trace.rows().len(), 3);
        assert_eq!(run.execution.facts.len(), 4);
    }
    for (rd, rs1, immediate) in [(1, 2, 0), (0, 1, 0), (1, 1, -4), (1, 1, -3)] {
        let mut words = vec![asm::auipc(rs1, 0), asm::addi(rs1, rs1, 8 - immediate)];
        words.push(asm::jalr(rd, rs1, immediate));
        let run = Fixture::new(words).complete();
        assert_eq!(run.prepared.output.trace.rows().len(), 3);
        assert_eq!(run.execution.facts.len(), 4);
        assert_eq!(run.execution.final_pc(), ENTRY + 8);
    }
    let padded = Fixture::new(vec![
        asm::auipc(1, 0),
        asm::addi(1, 1, 8),
        asm::jalr(1, 1, 0),
    ])
    .prepare();
    assert!(matches!(padded.adapt(padded.output.trace.rows()),
        Err(AdapterError::StallNotFixedPoint { cycle: 2, pc, .. }) if pc == ENTRY + 8));
    let exact = Fixture::new(vec![
        asm::auipc(1, 0),
        asm::addi(1, 1, 12),
        asm::addi(0, 0, 0),
        asm::jalr(1, 1, 0),
    ])
    .complete();
    assert_eq!(exact.execution.facts.len(), 4);
    assert_eq!(exact.execution.final_pc(), ENTRY + 12);
}

#[test]
fn instruction_list_identity_survives_omitted_words() {
    for (omitted, mode) in [
        (0, DecodeMode::Strict),
        (0x0220_81b3, DecodeMode::DataHoles),
    ] {
        let mut fixture = Fixture::new(vec![
            asm::jal(0, 8),
            omitted,
            asm::addi(3, 0, 1),
            asm::jal(0, 0),
        ]);
        fixture.mode = mode;
        let run = fixture.complete();
        let second = run.prepared.output.trace.rows()[1];
        assert_eq!(second.instruction_index(), 1);
        assert_eq!((second.pc() - ENTRY) / 4, 2);
        assert_eq!(run.execution.facts[1].bytecode_index, 1);
    }
}

#[test]
fn operand_presence_matches_all_fifty_traced_kinds() {
    let corpus = [
        (Kind::LUI, asm::lui(3, 1)),
        (Kind::AUIPC, asm::auipc(3, 1)),
        (Kind::JAL, asm::jal(3, 4)),
        (Kind::JALR, asm::jalr(3, 1, 4)),
        (Kind::BEQ, asm::beq(1, 2, 4)),
        (Kind::BNE, asm::bne(1, 2, 4)),
        (Kind::BLT, asm::blt(1, 2, 4)),
        (Kind::BGE, asm::bge(1, 2, 4)),
        (Kind::BLTU, asm::bltu(1, 2, 4)),
        (Kind::BGEU, asm::bgeu(1, 2, 4)),
        (Kind::LB, asm::lb(3, 1, 0)),
        (Kind::LH, asm::lh(3, 1, 0)),
        (Kind::LW, asm::lw(3, 1, 0)),
        (Kind::LD, asm::ld(3, 1, 0)),
        (Kind::LBU, asm::lbu(3, 1, 0)),
        (Kind::LHU, asm::lhu(3, 1, 0)),
        (Kind::LWU, asm::lwu(3, 1, 0)),
        (Kind::SB, asm::sb(1, 2, 0)),
        (Kind::SH, asm::sh(1, 2, 0)),
        (Kind::SW, asm::sw(1, 2, 0)),
        (Kind::SD, asm::sd(1, 2, 0)),
        (Kind::ADDI, asm::addi(3, 1, 1)),
        (Kind::SLTI, asm::slti(3, 1, 1)),
        (Kind::SLTIU, asm::sltiu(3, 1, 1)),
        (Kind::XORI, asm::xori(3, 1, 1)),
        (Kind::ORI, asm::ori(3, 1, 1)),
        (Kind::ANDI, asm::andi(3, 1, 1)),
        (Kind::SLLI, asm::slli(3, 1, 1)),
        (Kind::SRLI, asm::srli(3, 1, 1)),
        (Kind::SRAI, asm::srai(3, 1, 1)),
        (Kind::ADD, asm::add(3, 1, 2)),
        (Kind::SUB, asm::sub(3, 1, 2)),
        (Kind::SLL, asm::sll(3, 1, 2)),
        (Kind::SLT, asm::slt(3, 1, 2)),
        (Kind::SLTU, asm::sltu(3, 1, 2)),
        (Kind::XOR, asm::xor(3, 1, 2)),
        (Kind::SRL, asm::srl(3, 1, 2)),
        (Kind::SRA, asm::sra(3, 1, 2)),
        (Kind::OR, asm::or(3, 1, 2)),
        (Kind::AND, asm::and(3, 1, 2)),
        (Kind::ADDIW, asm::addiw(3, 1, 1)),
        (Kind::SLLIW, asm::slliw(3, 1, 1)),
        (Kind::SRLIW, asm::srliw(3, 1, 1)),
        (Kind::SRAIW, asm::sraiw(3, 1, 1)),
        (Kind::ADDW, asm::addw(3, 1, 2)),
        (Kind::SUBW, asm::subw(3, 1, 2)),
        (Kind::SLLW, asm::sllw(3, 1, 2)),
        (Kind::SRLW, asm::srlw(3, 1, 2)),
        (Kind::SRAW, asm::sraw(3, 1, 2)),
        (Kind::FENCE, asm::fence(15, 15)),
    ];
    assert_eq!(corpus.len(), 50);
    let source: Vec<_> = corpus
        .iter()
        .enumerate()
        .map(|(index, &(kind, word))| {
            let source = decode_instruction(word, ENTRY + index as u64 * 4, false, RV64I).unwrap();
            assert_eq!(source.kind(), kind);
            source
        })
        .collect();
    let bytecode = Bytecode::preprocess(&source, &Layout::new(6, 5, ENTRY - 32).unwrap()).unwrap();
    for (instruction, row) in source.iter().zip(bytecode.rows()) {
        let operands = instruction.row().operands;
        assert_eq!(
            (row.rs1, row.rs2, row.rd),
            (
                operands.rs1.unwrap_or(0),
                operands.rs2.unwrap_or(0),
                operands.rd.unwrap_or(0)
            )
        );
        assert_eq!(
            row.variant.unwrap().access().is_some(),
            matches!(
                instruction.kind(),
                Kind::LB
                    | Kind::LBU
                    | Kind::LH
                    | Kind::LHU
                    | Kind::LW
                    | Kind::LWU
                    | Kind::LD
                    | Kind::SB
                    | Kind::SH
                    | Kind::SW
                    | Kind::SD
            )
        );
    }
}

#[test]
fn ram_exponent_boundaries_are_candidates_for_the_statement_checker() {
    let mut words = Vec::new();
    address(&mut words, 5, DATA);
    words.extend([asm::ld(1, 5, 0), asm::jal(0, 0)]);
    let prepared = Fixture::new(words).prepare();
    let lowest = prepared.preprocessing.bytecode().lowest_address();
    let mut rows = prepared.output.trace.rows().to_vec();
    let base = rows[4];
    for (index, expected) in [(31, 5), (32, 6), (63, 6), (64, 7)] {
        rows[4] = replace(
            base,
            base.instruction_index(),
            base.registers(),
            RamAccess::Read(RamRead {
                address: lowest + index * 8,
                value: 0,
            }),
        );
        assert_eq!(prepared.adapt(&rows).unwrap().log_K_ram, expected);
    }
    rows[4] = replace(
        base,
        base.instruction_index(),
        base.registers(),
        RamAccess::Read(RamRead {
            address: lowest + (1 << 47) * 8,
            value: 0,
        }),
    );
    assert!(matches!(
        prepared.adapt(&rows),
        Err(AdapterError::Layout(LayoutError::BitsRowOverflow { .. }))
    ));
    for b in 1..=MAX_LOG_K_BYTECODE {
        assert!(matches!(
            Layout::new(b, 48, lowest),
            Err(LayoutError::BitsRowOverflow { .. })
        ));
    }
    rows[4] = replace(
        base,
        base.instruction_index(),
        base.registers(),
        RamAccess::Read(RamRead {
            address: lowest + (1 << 20) * 8,
            value: 0,
        }),
    );
    let execution = prepared.adapt(&rows).unwrap();
    let statement = prepared.statement(&execution);
    assert!(matches!(
        CheckedInputs::of_statement(
            &prepared.preprocessing,
            &statement,
            execution.log_K_ram,
            execution.final_pc()
        ),
        Err(Rv64iVerifierError::RamTooLarge)
    ));
}

#[test]
fn parallel_chunks_preserve_facts_maximum_and_first_error() {
    let mut words = Vec::new();
    address(&mut words, 5, DATA);
    words.extend([
        asm::lui(6, 12),
        asm::ld(7, 5, 0),
        asm::addi(6, 6, -1),
        asm::bne(6, 0, -8),
        asm::jal(0, 0),
    ]);
    let prepared = Fixture::new(words).prepare();
    let mut rows = prepared.output.trace.rows().to_vec();
    assert!(rows.len() > 1 << 17);
    let pools: Vec<_> = [1, 8]
        .into_iter()
        .map(|threads| {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        })
        .collect();
    let loads: Vec<_> = rows
        .iter()
        .enumerate()
        .filter_map(|(cycle, row)| matches!(row.ram_access(), RamAccess::Read(_)).then_some(cycle))
        .collect();
    let earlier = *loads.iter().find(|&&cycle| cycle > 100).unwrap();
    let later = *loads
        .iter()
        .find(|&&cycle| cycle > (1 << 16) + 100)
        .unwrap();
    let lowest = prepared.preprocessing.bytecode().lowest_address();
    for cycle in [earlier, later] {
        let old = rows[cycle];
        rows[cycle] = replace(
            old,
            old.instruction_index(),
            old.registers(),
            RamAccess::Read(RamRead {
                address: lowest + 8 * (1 << 15),
                value: 0,
            }),
        );
        let one = pools[0].install(|| prepared.adapt(&rows)).unwrap();
        let eight = pools[1].install(|| prepared.adapt(&rows)).unwrap();
        assert_eq!(one.facts, eight.facts);
        assert_eq!(one.log_K_ram, eight.log_K_ram);
        assert_eq!(one.log_K_ram, 16);
        rows[cycle] = old;
    }
    let old = rows[earlier];
    rows[earlier] = replace(
        old,
        old.instruction_index(),
        old.registers(),
        RamAccess::Read(RamRead {
            address: lowest - 8,
            value: 0,
        }),
    );
    let old = rows[later];
    rows[later] = replace(old, u32::MAX, RegisterState::default(), old.ram_access());
    for pool in pools {
        assert!(matches!(pool.install(|| prepared.adapt(&rows)),
            Err(AdapterError::AddressBelowRam { cycle, address })
                if cycle == earlier && address == lowest - 8));
    }
    assert!(matches!(
        adapt(
            prepared.preprocessing.bytecode(),
            prepared.preprocessing.image(),
            &prepared.output.device.memory_layout,
            ENTRY + 4,
            &rows
        ),
        Err(AdapterError::EntryPc { cycle: 0, .. })
    ));
}
