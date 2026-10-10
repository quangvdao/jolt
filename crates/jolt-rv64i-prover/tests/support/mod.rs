//! A counting loop built and replayed with the arithmetisation's machine oracle.
#![expect(
    clippy::unwrap_used,
    reason = "fixture failures fail the enclosing test"
)]

use common::{
    constants::RAM_START_ADDRESS,
    jolt_device::{JoltDevice, MemoryConfig, MemoryLayout},
};
use jolt_rv64i_arith::{CycleFacts, Layout};
use jolt_rv64i_prover::{commitment::transparent::TransparentBits, plane::Rv64iWitness};
use jolt_rv64i_verifier::{preprocessing::VerifierPreprocessing, statement::Statement};
use self::replay::State;
use std::sync::Arc;

#[expect(
    dead_code,
    reason = "the shared encoder contains instructions for the whole arithmetisation corpus"
)]
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

pub fn counting_loop() -> (
    Statement,
    VerifierPreprocessing<TransparentBits>,
    Rv64iWitness,
) {
    let (statement, preprocessing, witness, _) = counting_loop_fixture();
    (statement, preprocessing, witness)
}

pub fn counting_loop_facts() -> Vec<CycleFacts> {
    counting_loop_fixture().3
}

fn counting_loop_fixture() -> (
    Statement,
    VerifierPreprocessing<TransparentBits>,
    Rv64iWitness,
    Vec<CycleFacts>,
) {
    let memory_layout = MemoryLayout::try_new(&MemoryConfig {
        max_input_size: 8,
        max_output_size: 8,
        max_trusted_advice_size: 0,
        max_untrusted_advice_size: 0,
        stack_size: 0,
        heap_size: 0,
        program_size: Some(40),
    })
    .unwrap();
    let layout = Layout::new(4, 5, memory_layout.get_lowest_address()).unwrap();
    let words = [
        asm::auipc(2, 0),
        asm::addi(2, 2, -8),
        asm::addi(1, 0, 0),
        asm::addi(3, 0, 3),
        asm::addi(1, 1, 1),
        asm::blt(1, 3, -4),
        asm::addi(4, 0, 1),
        asm::sd(2, 4, 0),
        asm::jal(0, 0),
    ];
    let program: Vec<_> = words
        .iter()
        .enumerate()
        .map(|(i, word)| (RAM_START_ADDRESS + 4 * i as u64, *word))
        .collect();
    let bytecode = harness::bytecode(&program, &layout);
    let device = JoltDevice {
        inputs: vec![0x12, 0x34],
        outputs: vec![],
        memory_layout,
        ..JoltDevice::default()
    };
    let statement = Statement {
        log_T: 6,
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
    let preprocessing = VerifierPreprocessing::new(bytecode.clone(), image.clone(), ()).unwrap();
    let initial_ram: Vec<_> = std::iter::once((0, 0x3412)).chain(image).collect();
    let mut initial = State::new(RAM_START_ADDRESS);
    for &(index, value) in &initial_ram {
        initial.set_ram_word(index, value);
    }
    let mut machine = harness::machine(&program, &layout, &initial);
    let facts: Vec<_> = (0..64)
        .map(|_| {
            let record = machine.step().unwrap();
            harness::facts(&record, bytecode.index_of_pc(record.pc).unwrap())
        })
        .collect();
    let witness =
        Rv64iWitness::from_facts(layout, Arc::new(bytecode), &facts, initial_ram).unwrap();
    let trace = replay::replay(
        &witness.layout,
        &witness.bytecode,
        &witness.bits,
        &initial,
        witness.final_pc,
    )
    .unwrap();
    assert_eq!(trace.final_state.pc, RAM_START_ADDRESS + 32);
    assert_eq!(trace.final_state.registers[1], 3);
    assert_eq!(trace.final_state.ram.get(&3), Some(&1));
    (statement, preprocessing, witness, facts)
}
