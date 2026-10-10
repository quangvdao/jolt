#![expect(
    clippy::unwrap_used,
    reason = "test fixtures require successful construction"
)]

use std::collections::BTreeMap;

use jolt_rv64i_arith::{BaseWords, BitsBuilder, BitsRow, Layout, RowSystem, Variant, WitnessRow};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::common::{
    harness,
    replay::{self, State},
};

struct Fixture {
    words: &'static [u32],
    entry: u64,
    registers: &'static [(usize, u64)],
    layout: (usize, usize, u64),
    cycles: usize,
}
const FIXTURES: [Fixture; 5] = [
    Fixture {
        words: &[
            0x0020_81b3,
            0xfff0_8193,
            0x4020_81b3,
            0x0020_81bb,
            0xfff0_819b,
            0x4020_81bb,
            0x0020_f1b3,
            0x07f0_f193,
            0x0020_e1b3,
            0x07f0_e193,
            0x0020_c1b3,
            0xfff0_c193,
            0x0020_a1b3,
            0x0000_2193,
            0x0020_b1b3,
            0x0000_3193,
            0x0020_91b3,
            0x0020_d1b3,
            0x4020_d1b3,
            0x03f0_9193,
            0x03f0_d193,
            0x43f0_d193,
            0x0020_91bb,
            0x0020_d1bb,
            0x4020_d1bb,
            0x01f0_919b,
            0x01f0_d19b,
            0x41f0_d19b,
            0x0020_8263,
            0x0020_9263,
            0x0020_c263,
            0x0020_d263,
            0x0020_e263,
            0x0020_f263,
            0x0040_01ef,
            0xabcd_e1b7,
            0x0000_1197,
            0x0000_000f,
            0x0020_8033,
            0x0000_006f,
        ],
        entry: 0x100,
        registers: &[(1, 0x8000_0000_0000_003f), (2, 0x0123_4567_89ab_cdef)],
        layout: (6, 3, 0x1000),
        cycles: 40,
    },
    Fixture {
        words: &[
            0x0072_0183,
            0x0062_1183,
            0x0042_2183,
            0x0002_3183,
            0x0072_4183,
            0x0062_5183,
            0x0042_6183,
            0x0022_03a3,
            0x0022_1323,
            0x0022_2223,
            0x0022_3023,
            0x0072_0003,
            0x0062_1003,
            0x0042_2003,
            0x0002_3003,
            0x0000_006f,
        ],
        entry: 0x100,
        registers: &[(4, 0x1000), (2, 0x0123_4567_89ab_cdef)],
        layout: (5, 3, 0x1000),
        cycles: 16,
    },
    Fixture {
        words: &[0x0040_81e7, 0x0080_8067, 0x0000_006f],
        entry: 0x100,
        registers: &[(1, 0x100)],
        layout: (2, 3, 0x1000),
        cycles: 3,
    },
    Fixture {
        words: &[0x0000_0073],
        entry: 0x100,
        registers: &[],
        layout: (1, 3, 0x1000),
        cycles: 1,
    },
    Fixture {
        words: &[0x0010_0073],
        entry: 0x100,
        registers: &[],
        layout: (1, 3, 0x1000),
        cycles: 1,
    },
];

fn setup(f: &Fixture) -> (Layout, Vec<(u64, u32)>, State) {
    let layout = Layout::new(f.layout.0, f.layout.1, f.layout.2).unwrap();
    let program = f
        .words
        .iter()
        .enumerate()
        .map(|(i, &word)| (f.entry + 4 * i as u64, word))
        .collect();
    let mut state = State {
        pc: f.entry,
        registers: [0; 32],
        ram: BTreeMap::from([(0, 0x8877_6655_4433_2211)]),
    };
    for &(r, v) in f.registers {
        state.registers[r] = v;
    }
    (layout, program, state)
}
fn one_hot(layout: &Layout, bits: &BitsRow) -> bool {
    layout.chunks().all(|c| c.full(bits).is_power_of_two())
}
fn free_column(
    layout: &Layout,
    row: &jolt_rv64i_arith::BytecodeRow,
    fact: &jolt_rv64i_arith::CycleFacts,
    column: usize,
) -> bool {
    let in_chunk = |c: jolt_rv64i_arith::Chunk| {
        (usize::from(c.start())..usize::from(c.start()) + c.indicators()).contains(&column)
    };
    let variant = row.variant.unwrap();
    let access = variant.access().is_some();
    let keys_equal = if matches!(variant, Variant::SLTI | Variant::SLTIU) {
        fact.rs1_value == row.imm
    } else {
        fact.rs1_value == fact.rs2_value
    };
    let pos_free =
        variant.shift().is_none() && !access && (variant.key_kind().is_none() || keys_equal);
    let [low, high] = layout.pos_ra();
    (!access && layout.ram_ra().iter().copied().any(in_chunk))
        || (pos_free && (in_chunk(low) || in_chunk(high)))
        || (access && in_chunk(high))
        || (column == layout.should_branch() && variant.branch().is_none())
        || (column == layout.jalr_low_bit() && !matches!(variant, Variant::JALR | Variant::JALR_X0))
        || column >= layout.used_columns()
}

#[test]
fn literal_whole_traces_and_every_committed_bit() {
    let mut seen = [false; 58];
    for fixture in &FIXTURES {
        let (layout, program, initial) = setup(fixture);
        let bytecode = harness::bytecode(&program, &layout);
        let builder = BitsBuilder::new(&layout, &bytecode).unwrap();
        let mut machine = harness::machine(&program, &layout, &initial);
        let mut facts = Vec::new();
        let mut expected = Vec::new();
        let mut ram = initial.ram.clone();
        for _ in 0..fixture.cycles {
            let record = machine.step().unwrap();
            let index = bytecode.index_of_pc(record.pc).unwrap();
            let variant = bytecode.rows()[index].variant.unwrap();
            seen[variant.index()] = true;
            facts.push(harness::facts(&record, index));
            if let Some(access) = record.access {
                if access.is_store {
                    if access.word_after == 0 {
                        let _ = ram.remove(&access.word_index);
                    } else {
                        let _ = ram.insert(access.word_index, access.word_after);
                    }
                }
            }
            expected.push(State {
                pc: machine.pc(),
                registers: *machine.registers(),
                ram: ram.clone(),
            });
        }
        machine.run(0).unwrap();
        let mut bits = vec![[0; 4]; facts.len()];
        builder.fill(&facts, &mut bits).unwrap();
        let honest = replay::replay(&layout, &bytecode, &bits, &initial, machine.pc()).unwrap();
        assert_eq!(honest.cycles, expected);
        for cycle in 0..bits.len() {
            let variant = bytecode.rows()[facts[cycle].bytecode_index as usize]
                .variant
                .unwrap();
            for column in 0..256 {
                bits[cycle][column / 64] ^= 1 << (column % 64);
                let accepted = replay::replay(&layout, &bytecode, &bits, &initial, machine.pc());
                let free = free_column(
                    &layout,
                    &bytecode.rows()[facts[cycle].bytecode_index as usize],
                    &facts[cycle],
                    column,
                ) && one_hot(&layout, &bits[cycle]);
                assert_eq!(
                    accepted.is_ok(),
                    free,
                    "{variant:?} cycle={cycle} column={column}"
                );
                if let Ok(trace) = accepted {
                    assert_eq!(
                        trace.cycles, expected,
                        "{variant:?} cycle={cycle} column={column}"
                    );
                }
                bits[cycle][column / 64] ^= 1 << (column % 64);
            }
        }
    }
    assert!(seen.into_iter().all(|v| v));
}

#[test]
fn honest_oracle_corpus_over_six_layouts() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x636f_7270_7573);
    let mut seen = [false; 58];
    let mut count = 0;
    for (bc, ram, lowest) in [
        (6, 1, 0),
        (6, 3, 0x1000),
        (7, 5, 0x8000),
        (8, 7, 0x8000_0000),
        (6, 20, 0x7fff_0000),
        (7, 23, 0x1_0000_0000),
    ] {
        let layout = Layout::new(bc, ram, lowest).unwrap();
        let rows = RowSystem::new(&layout);
        for round in 0..300 {
            for f in &FIXTURES {
                let (_, mut program, mut initial) = setup(f);
                initial.ram = BTreeMap::from([(0, rng.next_u64())]);
                if f.words.len() > 3 {
                    initial.registers[1] = rng.next_u64();
                    initial.registers[2] = rng.next_u64();
                    initial.registers[4] = lowest;
                }
                if round % 2 == 1 && f.words.len() > 3 {
                    for (_, word) in &mut program {
                        if matches!(*word & 0x7f, 0x13 | 0x1b | 0x33 | 0x3b | 0x37 | 0x17 | 0x03) {
                            *word &= !(31 << 7);
                        }
                    }
                }
                let bytecode = harness::bytecode(&program, &layout);
                let builder = BitsBuilder::new(&layout, &bytecode).unwrap();
                let mut machine = harness::machine(&program, &layout, &initial);
                for _ in 0..f.cycles {
                    let record = machine.step().unwrap();
                    let index = bytecode.index_of_pc(record.pc).unwrap();
                    let facts = harness::facts(&record, index);
                    let bits = builder.bits_row(&facts).unwrap();
                    let row = &bytecode.rows()[index];
                    seen[row.variant.unwrap().index()] = true;
                    let z =
                        WitnessRow::compute(&layout, row, &BaseWords::from_facts(&facts), &bits);
                    assert!(
                        rows.failing_rows(&z).is_empty(),
                        "{:?} {facts:?}",
                        row.variant
                    );
                    count += 1;
                }
                machine.run(0).unwrap();
            }
        }
    }
    assert!(count >= 100_000, "{count}");
    assert!(seen.into_iter().all(|v| v));
}
