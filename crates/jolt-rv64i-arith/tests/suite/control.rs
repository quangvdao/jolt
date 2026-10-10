#![expect(
    clippy::unwrap_used,
    reason = "test fixtures require successful construction"
)]

use jolt_rv64i_arith::{BaseWords, BitsBuilder, CycleFacts, Layout, RowSystem, WitnessRow};

use super::common::{
    asm, harness,
    interp::Error,
    replay::{self, State},
};

#[test]
fn jalr_low_bit_local_and_execution_contracts() {
    let layout = Layout::new(1, 1, 0x1000).unwrap();
    let program = [(0, asm::jalr(0, 1, 1))];
    let bytecode = harness::bytecode(&program, &layout);
    let initial = State::new(0);
    let mut oracle = harness::machine(&program, &layout, &initial);
    let record = oracle.step().unwrap();
    assert_eq!(record.next_pc, 0);
    let facts = harness::facts(&record, 0);
    let mut bits = BitsBuilder::new(&layout, &bytecode)
        .unwrap()
        .bits_row(&facts)
        .unwrap();
    let system = RowSystem::new(&layout);
    for (next_pc, low) in [(0, true), (1, false)] {
        let column = layout.jalr_low_bit();
        bits[column / 64] =
            (bits[column / 64] & !(1 << (column % 64))) | (u64::from(low) << (column % 64));
        let base = BaseWords {
            next_pc,
            ..BaseWords::default()
        };
        assert!(system
            .failing_rows(&WitnessRow::compute(
                &layout,
                &bytecode.rows()[0],
                &base,
                &bits
            ))
            .is_empty());
        assert_eq!(
            replay::replay(&layout, &bytecode, &[bits], &initial, next_pc).is_ok(),
            low
        );
        assert_eq!(bytecode.final_pc_index(next_pc).is_ok(), low);
    }
}

#[test]
fn no_access_ignores_ram_at_nonzero_base_and_load_ignores_post() {
    let layout = Layout::new(2, 2, 0x1000).unwrap();
    let program = [
        (0, asm::addi(3, 1, 1)),
        (4, asm::ld(3, 2, 0)),
        (8, asm::ecall()),
    ];
    let bytecode = harness::bytecode(&program, &layout);
    let builder = BitsBuilder::new(&layout, &bytecode).unwrap();
    let system = RowSystem::new(&layout);
    let mut initial = State::new(0);
    initial.registers[1] = 6;
    initial.registers[2] = 0x1000;
    let mut oracle = harness::machine(&program, &layout, &initial);
    let fact = harness::facts(&oracle.step().unwrap(), 0);
    let bits = builder.bits_row(&fact).unwrap();
    assert!(system
        .failing_rows(&WitnessRow::compute(
            &layout,
            &bytecode.rows()[0],
            &BaseWords::from_facts(&fact),
            &bits
        ))
        .is_empty());
    for value in [1, 7, u64::MAX, 0x1234_5678_9abc_def0] {
        let altered = CycleFacts {
            ram_word_index: value,
            ram_pre_value: value,
            ram_post_value: !value,
            ..fact
        };
        assert_eq!(builder.bits_row(&altered).unwrap(), bits);
        assert!(system
            .failing_rows(&WitnessRow::compute(
                &layout,
                &bytecode.rows()[0],
                &BaseWords::from_facts(&altered),
                &bits
            ))
            .is_empty());
    }
    let load = harness::facts(&oracle.step().unwrap(), 1);
    let bits = builder.bits_row(&load).unwrap();
    for value in [0, 1, 7, u64::MAX, 0x1234_5678_9abc_def0] {
        assert_eq!(
            builder
                .bits_row(&CycleFacts {
                    ram_post_value: value,
                    ..load
                })
                .unwrap(),
            bits
        );
    }
}

#[test]
fn successor_membership_rejects_inner_and_final_escape() {
    let layout = Layout::new(3, 1, 0).unwrap();
    let program = [
        (0, asm::addi(1, 0, 1)),
        (4, asm::jal(0, 12)),
        (8, asm::ecall()),
        (12, asm::ecall()),
    ];
    let bytecode = harness::bytecode(&program, &layout);
    let initial = State::new(0);
    let mut oracle = harness::machine(&program, &layout, &initial);
    assert_eq!(oracle.step().unwrap().next_pc, 4);
    assert!(matches!(
        oracle.step(),
        Err(Error::InvalidTarget { target: 16, .. })
    ));
    let builder = BitsBuilder::new(&layout, &bytecode).unwrap();
    let first = builder
        .bits_row(&CycleFacts {
            rd_post_value: 1,
            next_pc: 4,
            ..CycleFacts::default()
        })
        .unwrap();
    let jump = builder
        .bits_row(&CycleFacts {
            bytecode_index: 1,
            next_pc: 16,
            ..CycleFacts::default()
        })
        .unwrap();
    let local = WitnessRow::compute(
        &layout,
        &bytecode.rows()[1],
        &BaseWords {
            next_pc: 16,
            ..BaseWords::default()
        },
        &jump,
    );
    assert!(RowSystem::new(&layout).failing_rows(&local).is_empty());
    for next_index in 0..bytecode.rows().len() {
        let mut next = [0; 4];
        layout
            .write_bytecode_index(&mut next, next_index as u64)
            .unwrap();
        assert!(replay::replay(&layout, &bytecode, &[first, jump, next], &initial, 0).is_err());
    }
    assert!(bytecode.final_pc_index(16).is_err());
    assert!(replay::replay(&layout, &bytecode, &[first, jump], &initial, 16).is_err());
    let program = [(0, asm::addi(1, 0, 1))];
    let bytecode = harness::bytecode(&program, &layout);
    let mut oracle = harness::machine(&program, &layout, &initial);
    assert!(matches!(
        oracle.run(1),
        Err(Error::MissingInstruction { pc: 4 })
    ));
    let bits = BitsBuilder::new(&layout, &bytecode)
        .unwrap()
        .bits_row(&CycleFacts {
            rd_post_value: 1,
            next_pc: 4,
            ..CycleFacts::default()
        })
        .unwrap();
    let local = WitnessRow::compute(
        &layout,
        &bytecode.rows()[0],
        &BaseWords {
            rd_write_value: 1,
            next_pc: 4,
            ..BaseWords::default()
        },
        &bits,
    );
    assert!(RowSystem::new(&layout).failing_rows(&local).is_empty());
    assert!(bytecode.final_pc_index(4).is_err());
    assert!(replay::replay(&layout, &bytecode, &[bits], &initial, 4).is_err());
}

#[test]
fn oracle_rejects_wrapping_ram_statement() {
    use super::common::interp::Machine;
    let lowest_address = u64::MAX - (1 << 22) + 1;
    assert!(Layout::new(1, 20, lowest_address).is_err());
    assert_eq!(
        Machine::new(
            &[(0, 0x0080_3003), (4, 0x0000_0073)],
            0,
            lowest_address,
            20,
            0
        ),
        Err(Error::InvalidRam {
            lowest_address,
            log_k_ram: 20
        })
    );
}
