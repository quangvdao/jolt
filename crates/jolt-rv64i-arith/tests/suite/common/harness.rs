//! Bridges independent instruction words and oracle records to the crate's inputs.

#![expect(clippy::panic, reason = "invalid fixtures fail their enclosing tests")]

use jolt_program::image::decode::decode_instruction;
use jolt_riscv::RV64IMAC_JOLT;
use jolt_rv64i_arith::{Bytecode, CycleFacts, Layout};

use super::interp::{Machine, Record};
use super::replay::State;

/// Retains the program list's order as its committed bytecode index order.
/// Invalid fixture encodings and preprocessing errors fail the caller's test.
pub fn bytecode(program: &[(u64, u32)], layout: &Layout) -> Bytecode {
    let instructions = program
        .iter()
        .map(|&(pc, word)| {
            decode_instruction(word, pc, false, RV64IMAC_JOLT)
                .unwrap_or_else(|error| panic!("fixture instruction at {pc:#x}: {error:?}"))
        })
        .collect::<Vec<_>>();
    Bytecode::preprocess(&instructions, layout)
        .unwrap_or_else(|error| panic!("fixture bytecode: {error:?}"))
}

/// Copies the independent oracle record; RAM fields are zero without an access.
/// `index` is the instruction's position in the preprocessed public list.
pub fn facts(record: &Record, index: usize) -> CycleFacts {
    let (ram_word_index, ram_pre_value, ram_post_value) =
        record.access.map_or((0, 0, 0), |access| {
            (access.word_index, access.word_before, access.word_after)
        });
    CycleFacts {
        bytecode_index: u32::try_from(index).unwrap_or_else(|_| {
            panic!("fixture bytecode index {index} exceeds the 32-bit address domain")
        }),
        rs1_value: record.rs1_value,
        rs2_value: record.rs2_value,
        rd_pre_value: record.rd_pre_value,
        rd_post_value: record.rd_post_value,
        ram_word_index,
        ram_pre_value,
        ram_post_value,
        next_pc: record.next_pc,
    }
}

/// Sets the independent machine's entry PC, registers and sparse RAM. Reserves
/// capacity for at least 64 new words, sufficient for the short mutation fixtures.
pub fn machine(program: &[(u64, u32)], layout: &Layout, initial: &State) -> Machine {
    let mut machine = Machine::new(
        program,
        initial.pc,
        layout.lowest_address(),
        layout.log_K_ram() as u8,
        initial.ram.len() + program.len() + 64,
    )
    .unwrap_or_else(|error| panic!("fixture machine: {error:?}"));
    for (register, &value) in initial.registers.iter().enumerate() {
        machine
            .set_register(register as u8, value)
            .unwrap_or_else(|error| panic!("fixture register: {error:?}"));
    }
    for (&index, &value) in &initial.ram {
        machine
            .set_ram_word(index, value)
            .unwrap_or_else(|error| panic!("fixture RAM: {error:?}"));
    }
    machine
}
