#![expect(
    clippy::unwrap_used,
    reason = "exhaustive contract tests fail on invalid setup"
)]

use super::common::{asm, harness};
use jolt_rv64i_arith::{
    BaseWords, BitsBuilder, BitsRow, BytecodeRow, Layout, PackedRow, RowGroup, RowSystem, Variant,
    WitnessRow,
};

fn set_bit(bits: &mut BitsRow, column: usize, value: bool) {
    let word = bits.get_mut(column / 64).unwrap();
    let mask = 1_u64 << (column % 64);
    *word = (*word & !mask) | (u64::from(value) * mask);
}

fn packed_hold(rows: &[&PackedRow], z: &WitnessRow) -> bool {
    rows.iter().all(|row| {
        let [a, b, c] = row.values(z);
        a * b == c
    })
}

fn adder_bytes(shift: u32, sums: u64) {
    let layout = Layout::new(1, 1, 0).unwrap();
    let bytecode = harness::bytecode(&[(0, asm::add(3, 1, 2)), (4, asm::jal(0, 0))], &layout);
    let row = bytecode.rows().first().unwrap();
    let system = RowSystem::new(&layout);
    let adder = system
        .lane_rows()
        .iter()
        .find(|row| row.group == RowGroup::Adder)
        .unwrap();
    for a in 0..256_u64 {
        for b in 0..256_u64 {
            for sum in 0..sums {
                let base = BaseWords {
                    rs1_value: a << shift,
                    rs2_value: b << shift,
                    rd_write_value: sum << shift,
                    ram_read_value: 0,
                    next_pc: 4,
                };
                let z = WitnessRow::compute(&layout, row, &base, &[sum << shift, 0, 0, 0]);
                let [az, bz, cz] = adder.values(&z);
                assert_eq!(
                    (az & bz) ^ cz == 0,
                    base.rd_write_value == base.rs1_value.wrapping_add(base.rs2_value),
                    "a={a}, b={b}, sum={sum}, shift={shift}"
                );
            }
        }
    }
}

#[test]
fn adder_exhausts_all_top_byte_triples() {
    adder_bytes(56, 256);
}

#[test]
fn adder_exhausts_low_byte_operands_and_nine_bit_sums() {
    adder_bytes(0, 512);
}

fn comparison_bytes(shift: u32) {
    let layout = Layout::new(3, 1, 0).unwrap();
    let bytecode = harness::bytecode(
        &[
            (0, asm::slt(3, 1, 2)),
            (4, asm::sltu(3, 1, 2)),
            (8, asm::beq(1, 2, 4)),
            (12, asm::bne(1, 2, 4)),
            (16, asm::jal(0, 0)),
        ],
        &layout,
    );
    let system = RowSystem::new(&layout);
    let comparisons: Vec<&PackedRow> = system
        .packed_rows()
        .iter()
        .filter(|row| {
            matches!(
                row.group,
                RowGroup::LessThan
                    | RowGroup::KeyBits
                    | RowGroup::KeysAgreeAbove
                    | RowGroup::KeysEqual
            )
        })
        .collect();
    let branches: Vec<&PackedRow> = system
        .packed_rows()
        .iter()
        .filter(|row| {
            matches!(
                row.group,
                RowGroup::LessThan
                    | RowGroup::KeyBits
                    | RowGroup::KeysAgreeAbove
                    | RowGroup::KeysEqual
                    | RowGroup::ControlResidual
            )
        })
        .collect();
    assert_eq!(comparisons.len(), 4);
    assert_eq!(branches.len(), 5);
    let mut positional = [[0; 4]; 64];
    for (pos, bits) in positional.iter_mut().enumerate() {
        layout.write_pos(bits, pos as u8).unwrap();
    }
    for x in 0..256_u64 {
        for y in 0..256_u64 {
            let a = x << shift;
            let b = y << shift;
            for row in bytecode.rows().iter().take(4) {
                let variant = row.variant.unwrap();
                let branch = matches!(variant, Variant::BEQ | Variant::BNE);
                let expected_result = if variant == Variant::SLT {
                    (a as i64) < (b as i64)
                } else if variant == Variant::SLTU {
                    a < b
                } else if variant == Variant::BEQ {
                    a == b
                } else {
                    assert_eq!(variant, Variant::BNE);
                    a != b
                };
                let targeted = if branch { &branches } else { &comparisons };
                for (pos, initial) in positional.iter().enumerate() {
                    for keys_differ in [false, true] {
                        for result in [false, true] {
                            let mut bits = *initial;
                            set_bit(&mut bits, layout.keys_differ(), keys_differ);
                            set_bit(&mut bits, layout.should_branch(), branch && result);
                            let base = BaseWords {
                                rs1_value: a,
                                rs2_value: b,
                                rd_write_value: u64::from(!branch && result),
                                ram_read_value: 0,
                                next_pc: row.fall_through_pc,
                            };
                            *bits.first_mut().unwrap() = base.rd_write_value;
                            let z = WitnessRow::compute(&layout, row, &base, &bits);
                            let expected = keys_differ == (a != b)
                                && (a == b || ((a ^ b) >> pos) == 1)
                                && result == expected_result;
                            assert_eq!(packed_hold(targeted, &z), expected,
                                "{variant:?}, a={a:#x}, b={b:#x}, pos={pos}, keys_differ={keys_differ}, result={result}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn comparison_exhausts_top_byte_pairs_positions_and_flags() {
    comparison_bytes(56);
}

#[test]
fn comparison_exhausts_low_byte_pairs_positions_and_flags() {
    comparison_bytes(0);
}

#[test]
fn decode_table_rejects_all_thirty_three_wrong_variants() {
    use super::common::replay::State;
    use std::collections::BTreeMap;

    let cases = [
        (asm::sub(3, 1, 2), Variant::SUB, Variant::ADD, 1, 1, 0),
        (asm::add(3, 1, 2), Variant::ADD, Variant::SUB, 1, 1, 0),
        (
            asm::addw(3, 1, 2),
            Variant::ADDW,
            Variant::ADD,
            0x7fff_ffff,
            1,
            0,
        ),
        (asm::addi(3, 1, 2), Variant::ADDI, Variant::ADD, 1, 0, 0),
        (asm::and(3, 1, 2), Variant::AND, Variant::OR, 3, 5, 0),
        (asm::or(3, 1, 2), Variant::OR, Variant::XOR, 3, 5, 0),
        (
            asm::slt(3, 1, 2),
            Variant::SLT,
            Variant::SLTU,
            1 << 63,
            1,
            0,
        ),
        (
            asm::sltu(3, 1, 2),
            Variant::SLTU,
            Variant::SLT,
            1 << 63,
            1,
            0,
        ),
        (
            asm::slti(3, 1, 1),
            Variant::SLTI,
            Variant::SLTIU,
            1 << 63,
            0,
            0,
        ),
        (asm::sra(3, 1, 2), Variant::SRA, Variant::SRL, 1 << 63, 1, 0),
        (asm::srl(3, 1, 2), Variant::SRL, Variant::SRA, 1 << 63, 1, 0),
        (asm::sll(3, 1, 2), Variant::SLL, Variant::SRL, 1, 1, 0),
        (
            asm::sraw(3, 1, 2),
            Variant::SRAW,
            Variant::SRLW,
            0x8000_0000,
            1,
            0,
        ),
        (
            asm::sllw(3, 1, 2),
            Variant::SLLW,
            Variant::SLL,
            0x4000_0000,
            1,
            0,
        ),
        (
            asm::srai(3, 1, 1),
            Variant::SRAI,
            Variant::SRAIW,
            1 << 63,
            0,
            0,
        ),
        (asm::lb(3, 1, 0), Variant::LB, Variant::LBU, 0x1000, 0, 0x80),
        (
            asm::lh(3, 1, 0),
            Variant::LH,
            Variant::LHU,
            0x1000,
            0,
            0x8000,
        ),
        (
            asm::lw(3, 1, 0),
            Variant::LW,
            Variant::LWU,
            0x1000,
            0,
            0x8000_0000,
        ),
        (
            asm::lwu(3, 1, 0),
            Variant::LWU,
            Variant::LW,
            0x1000,
            0,
            0x8000_0000,
        ),
        (
            asm::ld(3, 1, 0),
            Variant::LD,
            Variant::LW,
            0x1000,
            0,
            1 << 32,
        ),
        (asm::sb(1, 2, 1), Variant::SB, Variant::SH, 0x1000, 5, 0),
        (
            asm::sd(1, 2, 0),
            Variant::SD,
            Variant::SW,
            0x1000,
            1 << 40,
            0,
        ),
        (asm::beq(1, 2, 8), Variant::BEQ, Variant::BNE, 1, 1, 0),
        (asm::blt(1, 2, 8), Variant::BLT, Variant::BGE, 0, 1, 0),
        (
            asm::blt(1, 2, 8),
            Variant::BLT,
            Variant::BLTU,
            1 << 63,
            1,
            0,
        ),
        (
            asm::bgeu(1, 2, 8),
            Variant::BGEU,
            Variant::BGE,
            1 << 63,
            1,
            0,
        ),
        (asm::jal(3, 8), Variant::JAL, Variant::JAL_X0, 0, 0, 0),
        (
            asm::jalr(3, 1, 0),
            Variant::JALR,
            Variant::JALR_X0,
            0x109,
            0,
            0,
        ),
        (asm::lui(3, 1), Variant::LUI, Variant::AUIPC, 0, 0, 0),
        (asm::ecall(), Variant::ECALL, Variant::NOOP, 0, 0, 0),
        (asm::fence(0, 0), Variant::NOOP, Variant::ECALL, 0, 0, 0),
        (
            asm::lb(0, 1, 1),
            Variant::LOAD1_X0,
            Variant::LOAD2_X0,
            0x1000,
            0,
            0x1234,
        ),
        (
            asm::ld(0, 1, 0),
            Variant::LOAD8_X0,
            Variant::LD,
            0x1000,
            0,
            5,
        ),
    ];
    assert_eq!(cases.len(), 33);
    let layout = Layout::new(2, 1, 0x1000).unwrap();
    let system = RowSystem::new(&layout);
    for (word, honest, wrong, a, b, ram_word) in cases {
        let program = [
            (0x100, word),
            (0x104, asm::jal(0, 0)),
            (0x108, asm::jal(0, 0)),
        ];
        let bytecode = harness::bytecode(&program, &layout);
        let row = bytecode.rows().first().unwrap();
        assert_eq!(row.variant, Some(honest));
        let initial = State {
            pc: 0x100,
            registers: std::array::from_fn(|i| {
                if i == 1 {
                    a
                } else if i == 2 {
                    b
                } else {
                    0
                }
            }),
            ram: BTreeMap::from([(0, ram_word)]),
        };
        let mut machine = harness::machine(&program, &layout, &initial);
        let record = machine.step().unwrap();
        let facts = harness::facts(&record, 0);
        let bits = BitsBuilder::new(&layout, &bytecode)
            .unwrap()
            .bits_row(&facts)
            .unwrap();
        let base = BaseWords::from_facts(&facts);
        assert!(
            system
                .failing_rows(&WitnessRow::compute(&layout, row, &base, &bits))
                .is_empty(),
            "honest {honest:?}"
        );
        let swapped = BytecodeRow {
            variant: Some(wrong),
            ..*row
        };
        assert!(
            !system
                .failing_rows(&WitnessRow::compute(&layout, &swapped, &base, &bits))
                .is_empty(),
            "{honest:?}/{wrong:?}"
        );
    }
}
