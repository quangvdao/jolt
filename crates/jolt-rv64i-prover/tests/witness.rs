//! Witness ownership, replayed facts and public output contracts.
#![expect(
    clippy::unwrap_used,
    reason = "invalid fixtures fail the enclosing test"
)]

#[expect(
    dead_code,
    reason = "shared machine helpers serve the complete protocol corpus"
)]
mod support;

use common::{
    constants::RAM_START_ADDRESS,
    jolt_device::{MemoryConfig, MemoryLayout},
};
use jolt_field::{One, Zero, F128};
use jolt_rv64i_arith::{BitsRow, CycleFacts, Layout};
use jolt_rv64i_prover::{
    commitment::transparent::TransparentBits,
    error::{FactField, Rv64iProverError},
    plane::Rv64iWitness,
    reference::views::{base_word_with_lift, ram_val_final_with_lift, BaseWord},
};
use jolt_rv64i_verifier::{
    points::WordLift, preprocessing::VerifierPreprocessing, statement::CheckedInputs,
};
use std::sync::Arc;
use support::replay::State;

#[test]
fn facts_match_the_arithmetisation_replay_and_keep_nonaccess_ram_reads() {
    let (_, _, witness) = support::counting_loop();
    let facts = support::counting_loop_facts();
    let rebuilt = Rv64iWitness::from_facts(
        witness.layout.clone(),
        Arc::clone(&witness.bytecode),
        &facts,
        witness.initial_ram.clone(),
    )
    .unwrap();
    let initial = State {
        pc: RAM_START_ADDRESS,
        ram: witness.initial_ram.iter().copied().collect(),
        ..State::default()
    };
    let replay = support::replay::replay(
        &rebuilt.layout,
        &rebuilt.bytecode,
        &rebuilt.bits,
        &initial,
        rebuilt.final_pc,
    )
    .unwrap();
    for (cycle, bits) in rebuilt.bits.iter().enumerate() {
        let pre = if cycle == 0 {
            &initial
        } else {
            &replay.cycles[cycle - 1]
        };
        let row = &rebuilt.bytecode.rows()[rebuilt.layout.bytecode_index(bits) as usize];
        let expected = support::replay::base_words(
            &rebuilt.layout,
            row,
            bits,
            &pre.registers,
            &pre.ram,
            replay.cycles[cycle].pc,
        );
        assert_eq!(
            rebuilt.words[cycle]
                .base_words(row.variant.unwrap().is_store(), rebuilt.layout.inc(bits)),
            expected
        );
    }
    assert_eq!(rebuilt.final_ram.len(), 32);
    for (index, &found) in rebuilt.final_ram.iter().enumerate() {
        assert_eq!(
            found,
            replay
                .final_state
                .ram
                .get(&(index as u64))
                .copied()
                .unwrap_or(0)
        );
    }
    assert_eq!(
        (
            facts[0].ram_word_index,
            facts[0].ram_pre_value,
            facts[0].ram_post_value
        ),
        (0, 0, 0)
    );
    assert_eq!(rebuilt.words[0].ram_read_value, 0x3412);
    let read_bit = |point: [F128; 6]| {
        let lift = WordLift::new(&point).unwrap();
        base_word_with_lift(&rebuilt, BaseWord::RamReadValue, &lift).evaluate(&[F128::zero(); 6])
    };
    let (zero, one) = (F128::zero(), F128::one());
    assert_eq!(read_bit([zero; 6]), zero);
    assert_eq!(read_bit([zero, one, zero, zero, zero, zero]), zero);
    assert_eq!(read_bit([one, zero, zero, zero, zero, zero]), one);
}

#[test]
fn fact_differences_name_the_first_cycle_and_register_field() {
    let (_, _, witness) = support::counting_loop();
    for (cycle, field) in [
        (1, FactField::Rs1Value),
        (1, FactField::RdPreValue),
        (5, FactField::Rs2Value),
    ] {
        let mut facts = support::counting_loop_facts();
        let expected = match field {
            FactField::Rs1Value => facts[cycle].rs1_value,
            FactField::Rs2Value => facts[cycle].rs2_value,
            FactField::RdPreValue => facts[cycle].rd_pre_value,
            FactField::RamPreValue => unreachable!(),
        };
        match field {
            FactField::Rs1Value => facts[cycle].rs1_value ^= 1,
            FactField::Rs2Value => facts[cycle].rs2_value ^= 1,
            FactField::RdPreValue => facts[cycle].rd_pre_value ^= 1,
            FactField::RamPreValue => unreachable!(),
        }
        facts[63].bytecode_index = u32::MAX;
        let result = Rv64iWitness::from_facts(
            witness.layout.clone(),
            Arc::clone(&witness.bytecode),
            &facts,
            witness.initial_ram.clone(),
        );
        assert!(matches!(result, Err(Rv64iProverError::FactMismatch {
            cycle: found_cycle, field: found_field, expected: found_expected, found,
        }) if found_cycle == cycle && found_field == field && found_expected == expected && found == expected ^ 1));
    }
}

#[test]
fn absent_operands_and_nonaccess_ram_facts_do_not_replace_replayed_reads() {
    let (_, _, source) = support::counting_loop();
    let mut facts = support::counting_loop_facts();
    facts[0].rs1_value = u64::MAX;
    facts[0].rs2_value = u64::MAX;
    facts[0].ram_word_index = u64::MAX;
    facts[0].ram_pre_value = u64::MAX;
    facts[0].ram_post_value = u64::MAX;
    facts[63].rs1_value = u64::MAX;
    facts[63].rs2_value = u64::MAX;
    let witness =
        Rv64iWitness::from_facts(source.layout, source.bytecode, &facts, source.initial_ram)
            .unwrap();
    assert_eq!(witness.words[0].rs1_value, 0);
    assert_eq!(witness.words[0].rs2_value, 0);
    assert_eq!(witness.words[0].ram_read_value, 0x3412);
    assert_eq!(witness.final_ram[3], 1);
}

#[test]
fn load_facts_reject_ram_pre_values_and_register_reads_before_row_generation() {
    let (_, _, source) = support::counting_loop();
    let layout = source.layout;
    let program = [
        (RAM_START_ADDRESS, support::asm::auipc(2, 0)),
        (RAM_START_ADDRESS + 4, support::asm::addi(2, 2, -32)),
        (RAM_START_ADDRESS + 8, support::asm::ld(1, 2, 0)),
        (RAM_START_ADDRESS + 12, support::asm::jal(0, 0)),
    ];
    let bytecode = Arc::new(support::harness::bytecode(&program, &layout));
    let mut initial = State::new(RAM_START_ADDRESS);
    initial.set_ram_word(0, 0x3412);
    let mut machine = support::harness::machine(&program, &layout, &initial);
    let facts: Vec<_> = (0..4)
        .map(|_| {
            let record = machine.step().unwrap();
            support::harness::facts(&record, bytecode.index_of_pc(record.pc).unwrap())
        })
        .collect();
    for field in [FactField::RamPreValue, FactField::Rs1Value] {
        let mut changed = facts.clone();
        let expected = if field == FactField::RamPreValue {
            changed[2].ram_pre_value ^= 1;
            facts[2].ram_pre_value
        } else {
            changed[2].rs1_value ^= 1;
            changed[2].ram_word_index = u64::MAX;
            facts[2].rs1_value
        };
        let result = Rv64iWitness::from_facts(
            layout.clone(),
            Arc::clone(&bytecode),
            &changed,
            vec![(0, 0x3412)],
        );
        assert!(matches!(result, Err(Rv64iProverError::FactMismatch {
            cycle: 2, field: found_field, expected: found_expected, found,
        }) if found_field == field && found_expected == expected && found == expected ^ 1));
    }
}

#[test]
fn output_check_names_the_first_public_word_and_rejects_layout_mismatch() {
    let (statement, preprocessing, witness) = support::counting_loop();
    let checked =
        CheckedInputs::of_statement(&preprocessing, &statement, 5, witness.final_pc).unwrap();
    witness.check_outputs(&checked).unwrap();
    let mut changed = statement.clone();
    changed.device.outputs.push(1);
    let checked_output =
        CheckedInputs::of_statement(&preprocessing, &changed, 5, witness.final_pc).unwrap();
    assert!(matches!(
        witness.check_outputs(&checked_output),
        Err(Rv64iProverError::OutputMismatch {
            index: 1,
            expected: 1,
            found: 0,
        })
    ));
    let mut bits = witness.bits.to_vec();
    let store_cycle = bits
        .iter()
        .position(|bits| {
            witness.bytecode.rows()[witness.layout.bytecode_index(bits) as usize]
                .variant
                .unwrap()
                .is_store()
        })
        .unwrap();
    witness
        .layout
        .write_bytecode_index(&mut bits[store_cycle], 8)
        .unwrap();
    witness
        .layout
        .write_ram_index(&mut bits[store_cycle], 0)
        .unwrap();
    witness.layout.write_pos(&mut bits[store_cycle], 0).unwrap();
    bits[store_cycle][0] = 0;
    let without_termination = Rv64iWitness::from_bits(
        witness.layout.clone(),
        Arc::clone(&witness.bytecode),
        bits.into(),
        witness.initial_ram.clone(),
        witness.final_pc,
    )
    .unwrap();
    assert!(!without_termination.bytecode.rows()[without_termination
        .layout
        .bytecode_index(&without_termination.bits[store_cycle])
        as usize]
        .variant
        .unwrap()
        .is_store());
    assert!(matches!(
        without_termination.check_outputs(&checked),
        Err(Rv64iProverError::OutputMismatch {
            index: 3,
            expected: 1,
            found: 0,
        })
    ));
    let mut wrong_layout = witness.clone();
    wrong_layout.layout = Layout::new(3, 5, witness.layout.lowest_address()).unwrap();
    assert!(matches!(
        wrong_layout.check_outputs(&checked),
        Err(Rv64iProverError::OutputLayoutMismatch)
    ));
    let mut wrong_ram = witness;
    let _ = wrong_ram.final_ram.pop();
    assert!(matches!(
        wrong_ram.check_outputs(&checked),
        Err(Rv64iProverError::FinalRamLength {
            expected: 32,
            found: 31
        })
    ));
    assert!(matches!(
        ram_val_final_with_lift(&wrong_ram, &WordLift::new(&[F128::zero(); 6]).unwrap()),
        Err(Rv64iProverError::FinalRamLength {
            expected: 32,
            found: 31
        })
    ));
}

#[test]
fn preprocessing_and_bit_replay_retain_the_supplied_arcs() {
    let (_, source, witness) = support::counting_loop();
    let bytecode = Arc::clone(&witness.bytecode);
    let preprocessing = VerifierPreprocessing::<TransparentBits>::new(
        Arc::clone(&bytecode),
        source.image().to_vec(),
        (),
    )
    .unwrap();
    assert!(Arc::ptr_eq(&bytecode, preprocessing.shared_bytecode()));
    let bits = Arc::clone(&witness.bits);
    let replayed = Rv64iWitness::from_bits(
        witness.layout.clone(),
        Arc::clone(&bytecode),
        Arc::clone(&bits),
        witness.initial_ram.clone(),
        witness.final_pc,
    )
    .unwrap();
    assert!(Arc::ptr_eq(&bits, &replayed.bits));
    assert!(Arc::ptr_eq(&bytecode, &replayed.bytecode));
}

#[test]
fn constructors_return_a_typed_error_for_unallocatable_ram() {
    let memory = MemoryLayout::try_new(&MemoryConfig {
        max_input_size: 8,
        max_output_size: 8,
        max_trusted_advice_size: 0,
        max_untrusted_advice_size: 0,
        stack_size: 0,
        heap_size: 0,
        program_size: Some(8),
    })
    .unwrap();
    let layout = Layout::new(1, 47, memory.get_lowest_address()).unwrap();
    let program = [
        (RAM_START_ADDRESS, support::asm::sd(1, 2, 0)),
        (RAM_START_ADDRESS + 4, support::asm::jal(0, 0)),
    ];
    let bytecode = Arc::new(support::harness::bytecode(&program, &layout));
    let bits: Arc<[BitsRow]> = (0..1).map(|_| [0; 4]).collect();
    assert!(matches!(
        Rv64iWitness::from_bits(
            layout.clone(),
            Arc::clone(&bytecode),
            bits,
            vec![],
            RAM_START_ADDRESS
        ),
        Err(Rv64iProverError::RamAllocation { log_K_ram: 47, .. })
    ));
    let facts = [CycleFacts {
        next_pc: RAM_START_ADDRESS,
        ..CycleFacts::default()
    }];
    assert!(matches!(
        Rv64iWitness::from_facts(layout.clone(), Arc::clone(&bytecode), &facts, vec![]),
        Err(Rv64iProverError::RamAllocation { log_K_ram: 47, .. })
    ));
    assert!(matches!(
        Rv64iWitness::synthetic(1, layout, bytecode, &memory, vec![], 1),
        Err(Rv64iProverError::RamAllocation { log_K_ram: 47, .. })
    ));
}
