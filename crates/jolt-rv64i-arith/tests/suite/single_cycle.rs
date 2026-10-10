#![expect(
    clippy::unwrap_used,
    reason = "exhaustive contract tests fail on invalid setup or an unexpected oracle result"
)]

use super::common::{
    asm, harness,
    replay::{self, State},
};
use jolt_rv64i_arith::{BitsBuilder, BitsRow, Layout, RowSystem, Variant, WitnessRow};

fn instruction(variant: Variant, extreme: i32) -> Option<u32> {
    let immediate = match extreme {
        -1 => -2048,
        1 => 2047,
        _ => 1,
    };
    let shift = match extreme {
        -1 => 0,
        1 => 63,
        _ => 1,
    };
    let word_shift = shift & 31;
    let upper = match extreme {
        -1 => -524_288,
        1 => 524_287,
        _ => 1,
    };
    let branch = match extreme {
        -1 => -4096,
        1 => 4094,
        _ => 8,
    };
    let jump = match extreme {
        -1 => -1_048_576,
        1 => 1_048_574,
        _ => 8,
    };
    let jalr = match extreme {
        -1 => -2048,
        1 => 2047,
        _ => 0,
    };
    Some(match variant {
        Variant::ADD => asm::add(3, 1, 2),
        Variant::ADDI => asm::addi(3, 1, immediate),
        Variant::SUB => asm::sub(3, 1, 2),
        Variant::ADDW => asm::addw(3, 1, 2),
        Variant::ADDIW => asm::addiw(3, 1, immediate),
        Variant::SUBW => asm::subw(3, 1, 2),
        Variant::AND => asm::and(3, 1, 2),
        Variant::ANDI => asm::andi(3, 1, immediate),
        Variant::OR => asm::or(3, 1, 2),
        Variant::ORI => asm::ori(3, 1, immediate),
        Variant::XOR => asm::xor(3, 1, 2),
        Variant::XORI => asm::xori(3, 1, immediate),
        Variant::SLT => asm::slt(3, 1, 2),
        Variant::SLTI => asm::slti(3, 1, immediate),
        Variant::SLTU => asm::sltu(3, 1, 2),
        Variant::SLTIU => asm::sltiu(3, 1, immediate),
        Variant::SLL => asm::sll(3, 1, 2),
        Variant::SRL => asm::srl(3, 1, 2),
        Variant::SRA => asm::sra(3, 1, 2),
        Variant::SLLI => asm::slli(3, 1, shift),
        Variant::SRLI => asm::srli(3, 1, shift),
        Variant::SRAI => asm::srai(3, 1, shift),
        Variant::SLLW => asm::sllw(3, 1, 2),
        Variant::SRLW => asm::srlw(3, 1, 2),
        Variant::SRAW => asm::sraw(3, 1, 2),
        Variant::SLLIW => asm::slliw(3, 1, word_shift),
        Variant::SRLIW => asm::srliw(3, 1, word_shift),
        Variant::SRAIW => asm::sraiw(3, 1, word_shift),
        Variant::BEQ => asm::beq(1, 2, branch),
        Variant::BNE => asm::bne(1, 2, branch),
        Variant::BLT => asm::blt(1, 2, branch),
        Variant::BGE => asm::bge(1, 2, branch),
        Variant::BLTU => asm::bltu(1, 2, branch),
        Variant::BGEU => asm::bgeu(1, 2, branch),
        Variant::JAL => asm::jal(3, jump),
        Variant::JALR => asm::jalr(3, 1, jalr),
        Variant::LUI => asm::lui(3, upper),
        Variant::AUIPC => asm::auipc(3, upper),
        Variant::NOOP => asm::fence(15, 15),
        Variant::ECALL => asm::ecall(),
        Variant::EBREAK => asm::ebreak(),
        Variant::JAL_X0 => asm::jal(0, jump),
        Variant::JALR_X0 => asm::jalr(0, 1, jalr),
        Variant::LB
        | Variant::LH
        | Variant::LW
        | Variant::LD
        | Variant::LBU
        | Variant::LHU
        | Variant::LWU
        | Variant::SB
        | Variant::SH
        | Variant::SW
        | Variant::SD
        | Variant::LOAD1_X0
        | Variant::LOAD2_X0
        | Variant::LOAD4_X0
        | Variant::LOAD8_X0 => return None,
    })
}

fn bit(bits: &mut BitsRow, col: usize, value: bool) {
    let mask = 1u64 << (col % 64);
    bits[col / 64] = (bits[col / 64] & !mask) | if value { mask } else { 0 };
}

fn local_holds(system: &RowSystem, z: &WitnessRow) -> bool {
    for lane in system.lane_rows() {
        let [a, b, c] = lane.values(z);
        if (a & b) ^ c != 0 {
            return false;
        }
    }
    // Checked writers pin every chunk one-hot throughout this enumeration.
    system.packed_rows().iter().take(8).all(|row| {
        let [a, b, c] = row.values(z);
        a * b == c
    })
}

fn boundary_candidates(extreme: i32) {
    const BOUNDARIES: [u64; 9] = [
        0,
        1,
        (1 << 31) - 1,
        1 << 31,
        (1 << 32) - 1,
        1 << 32,
        (1 << 63) - 1,
        1 << 63,
        u64::MAX,
    ];
    let mut operands = Vec::new();
    for (i, &a) in BOUNDARIES.iter().enumerate() {
        operands.push((a, a));
        operands.push((a, BOUNDARIES[(i + 1) % BOUNDARIES.len()]));
    }
    operands.extend([(8, 1), (9, 1)]);
    let layout = Layout::new(2, 1, 0).unwrap();
    let system = RowSystem::new(&layout);
    let variants: Vec<_> = Variant::ALL
        .into_iter()
        .filter(|variant| variant.access().is_none())
        .collect();
    assert_eq!(variants.len(), 43);
    for variant in variants {
        let word = instruction(variant, extreme).unwrap();
        let program = [
            (0, word),
            (4, asm::jal(0, 0)),
            (8, asm::jal(0, 0)),
            (12, asm::jal(0, 0)),
        ];
        let bytecode = harness::bytecode(&program, &layout);
        let row = &bytecode.rows()[0];
        assert_eq!(row.variant, Some(variant));
        for &(a, b) in &operands {
            let mut initial = State::new(0);
            initial.registers[1] = a;
            initial.registers[2] = b;
            initial.registers[3] = 0xc3b2_a190_7865_4fed;
            let mut oracle = harness::machine(&program, &layout, &initial);
            let outcome = oracle.step();
            let expected = outcome
                .as_ref()
                .ok()
                .map(|record| (record.rd_post_value, record.next_pc));
            if let Ok(record) = &outcome {
                let facts = harness::facts(record, 0);
                let honest = BitsBuilder::new(&layout, &bytecode)
                    .unwrap()
                    .bits_row(&facts)
                    .unwrap();
                let base = replay::base_words(
                    &layout,
                    row,
                    &honest,
                    &initial.registers,
                    &initial.ram,
                    record.next_pc,
                );
                assert!(system
                    .failing_rows(&WitnessRow::compute(&layout, row, &base, &honest))
                    .is_empty());
            }
            let result = expected.map_or(0, |(rd, _)| rd);
            let mut destinations =
                vec![result, result ^ 1, result ^ 2, result ^ (1u64 << 63), 0, 1];
            destinations.sort_unstable();
            destinations.dedup();
            let mut accepted = 0;
            for pos in 0u8..64 {
                for flags in 0u8..8 {
                    let mut selectors = [0; 4];
                    layout.write_bytecode_index(&mut selectors, 0).unwrap();
                    layout.write_pos(&mut selectors, pos).unwrap();
                    bit(&mut selectors, layout.keys_differ(), flags & 1 != 0);
                    bit(&mut selectors, layout.should_branch(), flags & 2 != 0);
                    bit(&mut selectors, layout.jalr_low_bit(), flags & 4 != 0);
                    for &rd in &destinations {
                        let mut bits = selectors;
                        bits[0] = initial.registers[usize::from(row.rd)] ^ rd;
                        for next_pc in [0, 4, 8, 12] {
                            let base = replay::base_words(
                                &layout,
                                row,
                                &bits,
                                &initial.registers,
                                &initial.ram,
                                next_pc,
                            );
                            assert_eq!(base.rd_write_value, rd);
                            if local_holds(
                                &system,
                                &WitnessRow::compute(&layout, row, &base, &bits),
                            ) {
                                assert_eq!(Some((rd, next_pc)), expected, "variant={variant:?} extreme={extreme} a={a:#x} b={b:#x} pos={pos} flags={flags}");
                                accepted += 1;
                            }
                        }
                    }
                }
            }
            assert_eq!(
                accepted != 0,
                expected.is_some(),
                "variant={variant:?} extreme={extreme} a={a:#x} b={b:#x}"
            );
        }
    }
}

#[test]
fn all_43_nonaccess_variants_exhaust_auxiliaries_at_boundary_operands() {
    boundary_candidates(0);
}
#[test]
fn all_43_nonaccess_variants_exhaust_auxiliaries_at_negative_immediates() {
    boundary_candidates(-1);
}
#[test]
fn all_43_nonaccess_variants_exhaust_auxiliaries_at_positive_immediates() {
    boundary_candidates(1);
}
