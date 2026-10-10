#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "exhaustive contract tests fail on invalid setup or an unexpected oracle result"
)]

use super::common::{asm, harness, interp, replay};
use jolt_rv64i_arith::{
    BaseWords, BitsBuilder, BitsRow, BytecodeRow, CycleFacts, Layout, RowSystem, WitnessError,
    WitnessRow,
};
use std::collections::BTreeMap;

type MemoryEncoder = fn(u8, u8, i32) -> u32;

const LOADS: [(MemoryEncoder, u8); 7] = [
    (asm::lb, 1),
    (asm::lh, 2),
    (asm::lw, 4),
    (asm::ld, 8),
    (asm::lbu, 1),
    (asm::lhu, 2),
    (asm::lwu, 4),
];
const STORES: [(MemoryEncoder, u8); 4] = [(asm::sb, 1), (asm::sh, 2), (asm::sw, 4), (asm::sd, 8)];

fn instructions() -> Vec<(u32, u8)> {
    let mut words: Vec<_> = LOADS
        .iter()
        .map(|&(encode, width)| (encode(3, 1, 0), width))
        .collect();
    words.extend(
        STORES
            .iter()
            .map(|&(encode, width)| (encode(1, 2, 0), width)),
    );
    words.extend([
        (asm::lb(0, 1, 0), 1),
        (asm::lh(0, 1, 0), 2),
        (asm::lw(0, 1, 0), 4),
        (asm::ld(0, 1, 0), 8),
    ]);
    words
}

fn flip(bits: &mut BitsRow, col: usize) {
    bits[col / 64] ^= 1u64 << (col % 64);
}

fn local_holds(system: &RowSystem, z: &WitnessRow) -> bool {
    for lane in system.lane_rows() {
        let [a, b, c] = lane.values(z);
        if (a & b) ^ c != 0 {
            return false;
        }
    }
    // All candidate selectors are written through the checked one-hot writers.
    system.packed_rows().iter().take(8).all(|row| {
        let [a, b, c] = row.values(z);
        a * b == c
    })
}

fn witness(
    layout: &Layout,
    row: &BytecodeRow,
    bits: &BitsRow,
    initial: &replay::State,
    next_pc: u64,
) -> WitnessRow {
    let base = replay::base_words(layout, row, bits, &initial.registers, &initial.ram, next_pc);
    WitnessRow::compute(layout, row, &base, bits)
}

#[test]
fn all_59_legal_memory_pairs_reject_committed_and_supplied_mutations() {
    let cases = instructions();
    assert_eq!(
        cases
            .iter()
            .map(|(_, width)| 8 / usize::from(*width))
            .sum::<usize>(),
        59
    );
    let program: Vec<_> = cases
        .iter()
        .enumerate()
        .map(|(i, &(word, _))| (i as u64 * 4, word))
        .chain([(60, asm::jal(0, 0))])
        .collect();
    for (log_bytecode, log_ram, low) in [
        (20, 20, (1u64 << 31) - (1 << 16)),
        (20, 20, 0),
        (21, 21, u64::MAX - ((8u64 << 21) - 1)),
        (22, 24, 0),
        (20, 23, 0),
        (20, 27, 0),
    ] {
        let layout = Layout::new(log_bytecode, log_ram, low).unwrap();
        let bytecode = harness::bytecode(&program, &layout);
        let system = RowSystem::new(&layout);
        let builder = BitsBuilder::new(&layout, &bytecode).unwrap();
        let mut count = 0;
        for (index, &(_, width)) in cases.iter().enumerate() {
            let row = &bytecode.rows()[index];
            let variant = row.variant.unwrap();
            for offset in (0u8..8).step_by(usize::from(width)) {
                count += 1;
                let mut initial = replay::State {
                    pc: index as u64 * 4,
                    registers: [0; 32],
                    ram: BTreeMap::from([(1, 0x81a2_c3e4_f596_b7d8)]),
                };
                initial.registers[1] = low.wrapping_add(8 + u64::from(offset));
                initial.registers[2] = 0x19b8_d7f6_3512_4a6e;
                initial.registers[3] = 0x7134_589a_bcdf_e206;
                let mut oracle = harness::machine(&program, &layout, &initial);
                let record = oracle.step().unwrap();
                let facts = harness::facts(&record, index);
                let honest = builder.bits_row(&facts).unwrap();
                let z = witness(&layout, row, &honest, &initial, record.next_pc);
                assert!(
                    system.failing_rows(&z).is_empty(),
                    "{variant:?} offset {offset}"
                );
                for bit in 0..64 {
                    let mut changed = honest;
                    flip(&mut changed, bit);
                    let base = replay::base_words(
                        &layout,
                        row,
                        &changed,
                        &initial.registers,
                        &initial.ram,
                        record.next_pc,
                    );
                    if variant.is_store() {
                        assert_ne!(
                            facts.ram_pre_value ^ layout.inc(&changed),
                            facts.ram_post_value
                        );
                    } else {
                        assert_ne!(base.rd_write_value, record.rd_post_value);
                    }
                    assert!(
                        !system
                            .failing_rows(&WitnessRow::compute(&layout, row, &base, &changed))
                            .is_empty(),
                        "{variant:?} Inc bit {bit}"
                    );
                    let mut base = BaseWords::from_facts(&facts);
                    base.rd_write_value ^= 1u64 << bit;
                    assert!(
                        !system
                            .failing_rows(&WitnessRow::compute(&layout, row, &base, &honest))
                            .is_empty(),
                        "{variant:?} supplied destination bit {bit}"
                    );
                }
                let low_pos = layout.pos_ra()[0];
                for bit in 0..low_pos.indicators() {
                    let mut changed = honest;
                    flip(&mut changed, usize::from(low_pos.start()) + bit);
                    assert!(
                        !system
                            .failing_rows(&witness(
                                &layout,
                                row,
                                &changed,
                                &initial,
                                record.next_pc
                            ))
                            .is_empty(),
                        "{variant:?} Pos indicator {bit}"
                    );
                }
                for other in 0..8 {
                    if other != offset {
                        let mut changed = honest;
                        layout.write_pos(&mut changed, other).unwrap();
                        assert!(!system
                            .failing_rows(&witness(
                                &layout,
                                row,
                                &changed,
                                &initial,
                                record.next_pc
                            ))
                            .is_empty());
                    }
                }
                for chunk in layout.ram_ra() {
                    for bit in 0..chunk.indicators() {
                        let mut changed = honest;
                        flip(&mut changed, usize::from(chunk.start()) + bit);
                        assert!(
                            !system
                                .failing_rows(&witness(
                                    &layout,
                                    row,
                                    &changed,
                                    &initial,
                                    record.next_pc
                                ))
                                .is_empty(),
                            "{variant:?} RAM indicator {bit}"
                        );
                    }
                }
                for bit in 0..layout.log_K_ram() {
                    let mut changed = honest;
                    layout
                        .write_ram_index(&mut changed, facts.ram_word_index ^ (1u64 << bit))
                        .unwrap();
                    assert!(
                        !system
                            .failing_rows(&witness(
                                &layout,
                                row,
                                &changed,
                                &initial,
                                record.next_pc
                            ))
                            .is_empty(),
                        "{variant:?} RAM index bit {bit}"
                    );
                }
            }
        }
        assert_eq!(count, 59);
    }
}

#[test]
fn small_ram_exhausts_addresses_indices_and_positions() {
    for log_ram in 1..=5 {
        let words = 1u64 << log_ram;
        let bytes = words * 8;
        for low in [0, (1u64 << 31) - bytes / 2, u64::MAX - (bytes - 1)] {
            let layout = Layout::new(1, log_ram, low).unwrap();
            let system = RowSystem::new(&layout);
            let mut cases = Vec::new();
            for &(encode, width) in &LOADS {
                for rd in [0, 3] {
                    cases.push((encode(rd, 1, 0), width));
                }
            }
            cases.extend(
                STORES
                    .iter()
                    .map(|&(encode, width)| (encode(1, 2, 0), width)),
            );
            for (word, width) in cases {
                let program = [(0, word), (4, asm::jal(0, 0))];
                let bytecode = harness::bytecode(&program, &layout);
                let row = &bytecode.rows()[0];
                let builder = BitsBuilder::new(&layout, &bytecode).unwrap();
                for displacement in -24i64..(bytes as i64 + 24) {
                    let address = low.wrapping_add(displacement as u64);
                    let mut initial = replay::State {
                        pc: 0,
                        registers: [0; 32],
                        ram: (0..words)
                            .map(|i| (i, 0x9876_5432_10fe_dcba ^ i.wrapping_mul(0x1937_9bad)))
                            .collect(),
                    };
                    initial.registers[1] = address;
                    initial.registers[2] = 0x0123_4567_89ab_cdef;
                    initial.registers[3] = 0xdead_beef_7634_1250;
                    let mut oracle = harness::machine(&program, &layout, &initial);
                    let outcome = oracle.step();
                    let facts = outcome.as_ref().map_or_else(
                        |_| CycleFacts {
                            rs1_value: address,
                            rs2_value: initial.registers[2],
                            rd_pre_value: initial.registers[usize::from(row.rd)],
                            ram_word_index: address.wrapping_sub(low) >> 3,
                            next_pc: 4,
                            ..CycleFacts::default()
                        },
                        |record| harness::facts(record, 0),
                    );
                    let generated = builder.bits_row(&facts);
                    match &outcome {
                        Ok(_) => assert!(generated.is_ok()),
                        Err(interp::Error::OutsideRam { .. }) => assert!(matches!(
                            generated,
                            Err(WitnessError::AddressOutsideRam { .. }
                                | WitnessError::UnalignedAccess { .. })
                        )),
                        Err(interp::Error::UnalignedAccess { .. }) => assert!(
                            matches!(generated, Err(WitnessError::UnalignedAccess { width: found, .. }) if found == width)
                        ),
                        Err(error) => panic!("unexpected oracle error {error:?}"),
                    }
                    let inc = if row.variant.unwrap().is_store() {
                        facts.ram_pre_value ^ facts.ram_post_value
                    } else {
                        facts.rd_pre_value ^ facts.rd_post_value
                    };
                    let mut accepted = 0;
                    for index in 0..words {
                        for pos in 0u8..64 {
                            let mut bits = [inc, 0, 0, 0];
                            layout.write_bytecode_index(&mut bits, 0).unwrap();
                            layout.write_ram_index(&mut bits, index).unwrap();
                            layout.write_pos(&mut bits, pos).unwrap();
                            let holds =
                                local_holds(&system, &witness(&layout, row, &bits, &initial, 4));
                            let expected = outcome.as_ref().is_ok_and(|record| {
                                let access = record.access.unwrap();
                                index == access.word_index
                                    && u64::from(pos & 7) == access.address.wrapping_sub(low) % 8
                            });
                            assert_eq!(holds, expected, "log_ram={log_ram} low={low:#x} word={word:#x} address={address:#x} index={index} pos={pos}");
                            accepted += usize::from(holds);
                        }
                    }
                    assert_eq!(accepted, if outcome.is_ok() { 8 } else { 0 });
                    assert_eq!(generated.is_ok(), accepted != 0);
                }
            }
        }
    }
}
