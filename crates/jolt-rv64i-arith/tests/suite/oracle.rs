#![expect(
    clippy::unwrap_used,
    reason = "literal contract assertions in test code"
)]

use super::common::asm;

#[test]
fn encoder_matches_literal_rv64i_words() {
    let cases = [
        (asm::lui(1, -524_288), 0x8000_00b7),
        (asm::auipc(1, -524_288), 0x8000_0097),
        (asm::jal(1, -4), 0xffdf_f0ef),
        (asm::jalr(1, 2, -1), 0xfff1_00e7),
        (asm::beq(2, 3, -4), 0xfe31_0ee3),
        (asm::bne(2, 3, -4), 0xfe31_1ee3),
        (asm::blt(2, 3, -4), 0xfe31_4ee3),
        (asm::bge(2, 3, -4), 0xfe31_5ee3),
        (asm::bltu(2, 3, -4), 0xfe31_6ee3),
        (asm::bgeu(2, 3, -4), 0xfe31_7ee3),
        (asm::lb(1, 2, -8), 0xff81_0083),
        (asm::lh(1, 2, -8), 0xff81_1083),
        (asm::lw(1, 2, -8), 0xff81_2083),
        (asm::lbu(1, 2, -8), 0xff81_4083),
        (asm::lhu(1, 2, -8), 0xff81_5083),
        (asm::lwu(1, 2, -8), 0xff81_6083),
        (asm::ld(1, 2, -8), 0xff81_3083),
        (asm::sb(2, 3, -8), 0xfe31_0c23),
        (asm::sh(2, 3, -8), 0xfe31_1c23),
        (asm::sw(2, 3, -8), 0xfe31_2c23),
        (asm::sd(2, 3, -8), 0xfe31_3c23),
        (asm::addi(1, 2, -1), 0xfff1_0093),
        (asm::slti(1, 2, -1), 0xfff1_2093),
        (asm::sltiu(1, 2, -1), 0xfff1_3093),
        (asm::xori(1, 2, -1), 0xfff1_4093),
        (asm::ori(1, 2, -1), 0xfff1_6093),
        (asm::andi(1, 2, -1), 0xfff1_7093),
        (asm::slli(1, 2, 63), 0x03f1_1093),
        (asm::srli(1, 2, 63), 0x03f1_5093),
        (asm::srai(1, 2, 63), 0x43f1_5093),
        (asm::add(1, 2, 3), 0x0031_00b3),
        (asm::sub(1, 2, 3), 0x4031_00b3),
        (asm::sll(1, 2, 3), 0x0031_10b3),
        (asm::slt(1, 2, 3), 0x0031_20b3),
        (asm::sltu(1, 2, 3), 0x0031_30b3),
        (asm::xor(1, 2, 3), 0x0031_40b3),
        (asm::srl(1, 2, 3), 0x0031_50b3),
        (asm::sra(1, 2, 3), 0x4031_50b3),
        (asm::or(1, 2, 3), 0x0031_60b3),
        (asm::and(1, 2, 3), 0x0031_70b3),
        (asm::addiw(1, 2, -1), 0xfff1_009b),
        (asm::slliw(1, 2, 31), 0x01f1_109b),
        (asm::srliw(1, 2, 31), 0x01f1_509b),
        (asm::sraiw(1, 2, 31), 0x41f1_509b),
        (asm::addw(1, 2, 3), 0x0031_00bb),
        (asm::subw(1, 2, 3), 0x4031_00bb),
        (asm::sllw(1, 2, 3), 0x0031_10bb),
        (asm::srlw(1, 2, 3), 0x0031_50bb),
        (asm::sraw(1, 2, 3), 0x4031_50bb),
        (asm::fence(15, 15), 0x0ff0_000f),
        (asm::ecall(), 0x0000_0073),
        (asm::ebreak(), 0x0010_0073),
        (asm::addi(1, 0, 1), 0x0010_0093),
        (asm::jal(0, 0), 0x0000_006f),
        (asm::beq(31, 30, -4096), 0x81ef_8063),
        (asm::bne(31, 30, 4094), 0x7fef_9fe3),
        (asm::jal(31, -1_048_576), 0x8000_0fef),
        (asm::jal(31, 1_048_574), 0x7fff_ffef),
        (asm::lui(31, 524_287), 0x7fff_ffb7),
        (asm::addi(31, 31, -2048), 0x800f_8f93),
        (asm::addi(31, 31, 2047), 0x7fff_8f93),
        (asm::sb(31, 30, -2048), 0x81ef_8023),
        (asm::sb(31, 30, 2047), 0x7fef_8fa3),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, expected);
    }
}

use super::common::interp::{Access, Error, Machine, Record};

fn machine(word: u32) -> Machine {
    Machine::new(
        &[(0x100, word), (0x104, asm::jal(0, 0))],
        0x100,
        0x1000,
        27,
        32,
    )
    .unwrap()
}

#[test]
fn arithmetic_known_answers() {
    let cases = [
        (asm::lui(1, -524_288), 0, 0, 0xffff_ffff_8000_0000),
        (asm::auipc(1, -524_288), 0, 0, 0xffff_ffff_8000_0100),
        (asm::addi(1, 2, -1), 0, 0, 0xffff_ffff_ffff_ffff),
        (asm::slti(1, 2, 1), 0x8000_0000_0000_0000, 0, 1),
        (asm::sltiu(1, 2, 1), 0x8000_0000_0000_0000, 0, 0),
        (asm::sltiu(1, 2, -1), 1, 0, 1),
        (asm::xori(1, 2, -1), 0xff00, 0, 0xffff_ffff_ffff_00ff),
        (asm::ori(1, 2, 15), 0xf0, 0, 0xff),
        (asm::andi(1, 2, 15), 0xf0, 0, 0),
        (asm::slli(1, 2, 63), 1, 0, 0x8000_0000_0000_0000),
        (asm::srli(1, 2, 63), 0x8000_0000_0000_0000, 0, 1),
        (
            asm::srai(1, 2, 63),
            0x8000_0000_0000_0000,
            0,
            0xffff_ffff_ffff_ffff,
        ),
        (asm::add(1, 2, 3), 0x7fff_ffff, 1, 0x8000_0000),
        (asm::add(1, 2, 3), 0x1_0000_0000, 1, 0x1_0000_0001),
        (asm::sub(1, 2, 3), 0, 1, 0xffff_ffff_ffff_ffff),
        (asm::sll(1, 2, 3), 1, 64, 1),
        (asm::slt(1, 2, 3), 0x8000_0000_0000_0000, 1, 1),
        (asm::sltu(1, 2, 3), 0x8000_0000_0000_0000, 1, 0),
        (asm::xor(1, 2, 3), 0xf0, 0x0f, 0xff),
        (asm::srl(1, 2, 3), 0x8000_0000_0000_0000, 127, 1),
        (
            asm::sra(1, 2, 3),
            0x8000_0000_0000_0000,
            127,
            0xffff_ffff_ffff_ffff,
        ),
        (asm::or(1, 2, 3), 0xf0, 0x33, 0xf3),
        (asm::and(1, 2, 3), 0xf0, 0x33, 0x30),
        (asm::addiw(1, 2, 1), 0x7fff_ffff, 0, 0xffff_ffff_8000_0000),
        (asm::addiw(1, 2, 0), 0x1_0000_0000, 0, 0),
        (asm::slliw(1, 2, 31), 1, 0, 0xffff_ffff_8000_0000),
        (asm::srliw(1, 2, 31), 0xffff_ffff_8000_0000, 0, 1),
        (asm::sraiw(1, 2, 31), 0x8000_0000, 0, 0xffff_ffff_ffff_ffff),
        (asm::addw(1, 2, 3), 0x7fff_ffff, 1, 0xffff_ffff_8000_0000),
        (asm::addw(1, 2, 3), 0x1_0000_0000, 1, 1),
        (asm::subw(1, 2, 3), 0, 1, 0xffff_ffff_ffff_ffff),
        (asm::sllw(1, 2, 3), 1, 63, 0xffff_ffff_8000_0000),
        (asm::srlw(1, 2, 3), 0xffff_ffff_8000_0000, 63, 1),
        (asm::sraw(1, 2, 3), 0x8000_0000, 63, 0xffff_ffff_ffff_ffff),
    ];
    for (word, a, b, expected) in cases {
        let mut m = machine(word);
        m.set_register(2, a).unwrap();
        m.set_register(3, b).unwrap();
        let record = m.step().unwrap();
        assert_eq!(m.registers()[1], expected, "word {word:#010x}");
        assert_eq!(record.rd_post_value, expected);
        assert_eq!(m.pc(), 0x104);
        assert_eq!(m.ram_word(0).unwrap(), 0);
    }
}

#[test]
fn shift_boundaries() {
    for (amount, left, logical, arithmetic) in [
        (0, 1, 0x8000_0000_0000_0000, 0x8000_0000_0000_0000),
        (31, 0x8000_0000, 0x1_0000_0000, 0xffff_ffff_0000_0000),
        (32, 0x1_0000_0000, 0x8000_0000, 0xffff_ffff_8000_0000),
        (63, 0x8000_0000_0000_0000, 1, 0xffff_ffff_ffff_ffff),
    ] {
        for (word, input, expected) in [
            (asm::slli(1, 2, amount), 1, left),
            (asm::srli(1, 2, amount), 0x8000_0000_0000_0000, logical),
            (asm::srai(1, 2, amount), 0x8000_0000_0000_0000, arithmetic),
            (asm::sll(1, 2, 3), 1, left),
            (asm::srl(1, 2, 3), 0x8000_0000_0000_0000, logical),
            (asm::sra(1, 2, 3), 0x8000_0000_0000_0000, arithmetic),
        ] {
            let mut m = machine(word);
            m.set_register(2, input).unwrap();
            m.set_register(3, amount as u64 + 64).unwrap();
            let _ = m.step().unwrap();
            assert_eq!(m.registers()[1], expected);
            assert_eq!(m.pc(), 0x104);
        }
    }
    for (amount, left, logical, arithmetic) in [
        (0, 1, 0xffff_ffff_8000_0000, 0xffff_ffff_8000_0000),
        (31, 0xffff_ffff_8000_0000, 1, 0xffff_ffff_ffff_ffff),
        (32, 1, 0xffff_ffff_8000_0000, 0xffff_ffff_8000_0000),
        (63, 0xffff_ffff_8000_0000, 1, 0xffff_ffff_ffff_ffff),
        (95, 0xffff_ffff_8000_0000, 1, 0xffff_ffff_ffff_ffff),
    ] {
        for (word, input, expected) in [
            (asm::sllw(1, 2, 3), 0x1_0000_0001, left),
            (asm::srlw(1, 2, 3), 0x1234_5678_8000_0000, logical),
            (asm::sraw(1, 2, 3), 0x1234_5678_8000_0000, arithmetic),
        ] {
            let mut m = machine(word);
            m.set_register(2, input).unwrap();
            m.set_register(3, amount).unwrap();
            let _ = m.step().unwrap();
            assert_eq!(m.registers()[1], expected);
        }
    }
}

#[test]
fn loads_at_every_legal_offset() {
    let cases: &[(u32, u64, u8, u64, u64)] = &[
        (asm::lb(1, 2, 0), 0x1000, 1, 0x11, 0xffff_ffff_ffff_ff80),
        (asm::lb(1, 2, 1), 0x1001, 1, 0x22, 0xffff_ffff_ffff_ff81),
        (asm::lb(1, 2, 2), 0x1002, 1, 0x33, 0xffff_ffff_ffff_ff82),
        (asm::lb(1, 2, 3), 0x1003, 1, 0x44, 0xffff_ffff_ffff_ff83),
        (asm::lb(1, 2, 4), 0x1004, 1, 0x55, 0xffff_ffff_ffff_ff84),
        (asm::lb(1, 2, 5), 0x1005, 1, 0x66, 0xffff_ffff_ffff_ff85),
        (asm::lb(1, 2, 6), 0x1006, 1, 0x77, 0xffff_ffff_ffff_ff86),
        (
            asm::lb(1, 2, 7),
            0x1007,
            1,
            0xffff_ffff_ffff_ff88,
            0xffff_ffff_ffff_ff87,
        ),
        (asm::lbu(1, 2, 0), 0x1000, 1, 0x11, 0x80),
        (asm::lbu(1, 2, 1), 0x1001, 1, 0x22, 0x81),
        (asm::lbu(1, 2, 2), 0x1002, 1, 0x33, 0x82),
        (asm::lbu(1, 2, 3), 0x1003, 1, 0x44, 0x83),
        (asm::lbu(1, 2, 4), 0x1004, 1, 0x55, 0x84),
        (asm::lbu(1, 2, 5), 0x1005, 1, 0x66, 0x85),
        (asm::lbu(1, 2, 6), 0x1006, 1, 0x77, 0x86),
        (asm::lbu(1, 2, 7), 0x1007, 1, 0x88, 0x87),
        (asm::lh(1, 2, 0), 0x1000, 2, 0x2211, 0xffff_ffff_ffff_8180),
        (asm::lh(1, 2, 2), 0x1002, 2, 0x4433, 0xffff_ffff_ffff_8382),
        (asm::lh(1, 2, 4), 0x1004, 2, 0x6655, 0xffff_ffff_ffff_8584),
        (
            asm::lh(1, 2, 6),
            0x1006,
            2,
            0xffff_ffff_ffff_8877,
            0xffff_ffff_ffff_8786,
        ),
        (asm::lhu(1, 2, 0), 0x1000, 2, 0x2211, 0x8180),
        (asm::lhu(1, 2, 2), 0x1002, 2, 0x4433, 0x8382),
        (asm::lhu(1, 2, 4), 0x1004, 2, 0x6655, 0x8584),
        (asm::lhu(1, 2, 6), 0x1006, 2, 0x8877, 0x8786),
        (
            asm::lw(1, 2, 0),
            0x1000,
            4,
            0x4433_2211,
            0xffff_ffff_8382_8180,
        ),
        (
            asm::lw(1, 2, 4),
            0x1004,
            4,
            0xffff_ffff_8877_6655,
            0xffff_ffff_8786_8584,
        ),
        (asm::lwu(1, 2, 0), 0x1000, 4, 0x4433_2211, 0x8382_8180),
        (asm::lwu(1, 2, 4), 0x1004, 4, 0x8877_6655, 0x8786_8584),
        (
            asm::ld(1, 2, 0),
            0x1000,
            8,
            0x8877_6655_4433_2211,
            0x8786_8584_8382_8180,
        ),
    ];
    for &(word, address, width, positive, negative) in cases {
        for (initial, expected) in [
            (0x8877_6655_4433_2211, positive),
            (0x8786_8584_8382_8180, negative),
        ] {
            let mut m = machine(word);
            m.set_register(2, 0x1000).unwrap();
            m.set_ram_word(0, initial).unwrap();
            let record = m.step().unwrap();
            assert_eq!(m.registers()[1], expected);
            assert_eq!(m.ram_word(0).unwrap(), initial);
            assert_eq!(m.pc(), 0x104);
            assert_eq!(
                record.access,
                Some(Access {
                    is_store: false,
                    address,
                    width,
                    word_index: 0,
                    word_before: initial,
                    word_after: initial
                })
            );
        }
    }
}

#[test]
fn stores_preserve_other_bytes() {
    for (word, expected) in [
        (asm::sb(2, 3, 0), 0x8877_6655_4433_22ef),
        (asm::sb(2, 3, 1), 0x8877_6655_4433_ef11),
        (asm::sb(2, 3, 2), 0x8877_6655_44ef_2211),
        (asm::sb(2, 3, 3), 0x8877_6655_ef33_2211),
        (asm::sb(2, 3, 4), 0x8877_66ef_4433_2211),
        (asm::sb(2, 3, 5), 0x8877_ef55_4433_2211),
        (asm::sb(2, 3, 6), 0x88ef_6655_4433_2211),
        (asm::sb(2, 3, 7), 0xef77_6655_4433_2211),
        (asm::sh(2, 3, 0), 0x8877_6655_4433_cdef),
        (asm::sh(2, 3, 2), 0x8877_6655_cdef_2211),
        (asm::sh(2, 3, 4), 0x8877_cdef_4433_2211),
        (asm::sh(2, 3, 6), 0xcdef_6655_4433_2211),
        (asm::sw(2, 3, 0), 0x8877_6655_89ab_cdef),
        (asm::sw(2, 3, 4), 0x89ab_cdef_4433_2211),
        (asm::sd(2, 3, 0), 0x0123_4567_89ab_cdef),
    ] {
        let mut m = machine(word);
        m.set_register(2, 0x1000).unwrap();
        m.set_register(3, 0x0123_4567_89ab_cdef).unwrap();
        m.set_ram_word(0, 0x8877_6655_4433_2211).unwrap();
        let record = m.step().unwrap();
        assert_eq!(m.ram_word(0).unwrap(), expected);
        assert_eq!(record.access.unwrap().word_after, expected);
        assert_eq!(m.registers()[1], 0);
        assert_eq!(m.pc(), 0x104);
    }
}

#[test]
fn branch_and_jump_answers() {
    for (encode, a, b, taken) in [
        (asm::beq as fn(u8, u8, i32) -> u32, 1, 1, true),
        (asm::beq, 1, 2, false),
        (asm::bne, 1, 2, true),
        (asm::bne, 1, 1, false),
        (asm::blt, 0x8000_0000_0000_0000, 1, true),
        (asm::blt, 1, 0x8000_0000_0000_0000, false),
        (asm::bge, 0x8000_0000_0000_0000, 1, false),
        (asm::bge, 1, 0x8000_0000_0000_0000, true),
        (asm::bge, 1, 1, true),
        (asm::bltu, 0x8000_0000_0000_0000, 1, false),
        (asm::bltu, 1, 0x8000_0000_0000_0000, true),
        (asm::bgeu, 0x8000_0000_0000_0000, 1, true),
        (asm::bgeu, 1, 0x8000_0000_0000_0000, false),
        (asm::bgeu, 1, 1, true),
    ] {
        for (offset, destination) in [(8, 0x108), (-8, 0xf8)] {
            let mut m = Machine::new(
                &[(0x100, encode(2, 3, offset)), (destination, asm::ecall())],
                0x100,
                0x1000,
                0,
                0,
            )
            .unwrap();
            m.set_register(2, a).unwrap();
            m.set_register(3, b).unwrap();
            let record = m.step().unwrap();
            assert_eq!(m.pc(), if taken { destination } else { 0x104 });
            assert_eq!((record.rd_pre_value, record.rd_post_value), (0, 0));
            assert_eq!(m.ram_word(0).unwrap(), 0);
            assert_eq!(m.registers()[2], a);
            assert_eq!(m.registers()[3], b);
        }
    }
    for (word, initial, expected_rd, target) in [
        (asm::jal(1, 8), 0, 0x104, 0x108),
        (asm::jal(1, -8), 0, 0x104, 0xf8),
        (asm::jal(0, 8), 0, 0, 0x108),
        (asm::jalr(1, 1, 0), 0x109, 0x104, 0x108),
        (asm::jalr(0, 1, 0), 0x109, 0x109, 0x108),
        (asm::jalr(1, 1, -1), 0x109, 0x104, 0x108),
    ] {
        let mut m = Machine::new(
            &[(0x100, word), (target, asm::ecall())],
            0x100,
            0x1000,
            0,
            0,
        )
        .unwrap();
        m.set_register(1, initial).unwrap();
        let record = m.step().unwrap();
        assert_eq!(m.registers()[1], expected_rd);
        assert_eq!(m.registers()[0], 0);
        assert_eq!(record.next_pc, target);
        assert_eq!(m.pc(), target);
        assert_eq!(m.ram_word(0).unwrap(), 0);
    }
}

#[test]
fn fence_and_system_answers() {
    for (word, expected_pc) in [
        (asm::fence(15, 15), 0x104),
        (asm::ecall(), 0x100),
        (asm::ebreak(), 0x100),
    ] {
        let mut m = machine(word);
        m.set_register(1, 123).unwrap();
        m.set_ram_word(0, 456).unwrap();
        assert_eq!(
            m.step().unwrap(),
            Record {
                pc: 0x100,
                word,
                rs1_value: 0,
                rs2_value: 0,
                rd_pre_value: 0,
                rd_post_value: 0,
                access: None,
                next_pc: expected_pc
            }
        );
        assert_eq!(m.registers()[1], 123);
        assert_eq!(m.ram_word(0).unwrap(), 456);
        assert_eq!(m.pc(), expected_pc);
    }
}

#[test]
fn records_for_all_formats() {
    for (word, a, b, old, new, next_pc) in [
        (asm::add(1, 2, 3), 10, 7, 91, 17, 0x104),
        (asm::addi(1, 2, -3), 10, 0, 91, 7, 0x104),
        (asm::slli(1, 2, 3), 10, 0, 91, 80, 0x104),
        (
            asm::lui(1, -524_288),
            0,
            0,
            91,
            0xffff_ffff_8000_0000,
            0x104,
        ),
        (asm::jal(1, 4), 0, 0, 91, 0x104, 0x104),
        (asm::beq(2, 3, 4), 10, 7, 0, 0, 0x104),
    ] {
        let mut m = machine(word);
        m.set_register(1, 91).unwrap();
        m.set_register(2, 10).unwrap();
        m.set_register(3, 7).unwrap();
        assert_eq!(
            m.step().unwrap(),
            Record {
                pc: 0x100,
                word,
                rs1_value: a,
                rs2_value: b,
                rd_pre_value: old,
                rd_post_value: new,
                access: None,
                next_pc
            }
        );
    }
    let mut m = machine(asm::sh(2, 3, 2));
    m.set_register(2, 0x1000).unwrap();
    m.set_register(3, 0xdead_beef).unwrap();
    m.set_ram_word(0, 0x8877_6655_4433_2211).unwrap();
    assert_eq!(
        m.step().unwrap(),
        Record {
            pc: 0x100,
            word: 0x0031_1123,
            rs1_value: 0x1000,
            rs2_value: 0xdead_beef,
            rd_pre_value: 0,
            rd_post_value: 0,
            access: Some(Access {
                is_store: true,
                address: 0x1002,
                width: 2,
                word_index: 0,
                word_before: 0x8877_6655_4433_2211,
                word_after: 0x8877_6655_beef_2211
            }),
            next_pc: 0x104
        }
    );
    let mut m = machine(asm::lb(0, 2, 7));
    m.set_register(0, 99).unwrap();
    m.set_register(2, 0x1000).unwrap();
    m.set_ram_word(0, 0x8877_6655_4433_2211).unwrap();
    assert_eq!(
        m.step().unwrap(),
        Record {
            pc: 0x100,
            word: 0x0071_0003,
            rs1_value: 0x1000,
            rs2_value: 0,
            rd_pre_value: 0,
            rd_post_value: 0,
            access: Some(Access {
                is_store: false,
                address: 0x1007,
                width: 1,
                word_index: 0,
                word_before: 0x8877_6655_4433_2211,
                word_after: 0x8877_6655_4433_2211
            }),
            next_pc: 0x104
        }
    );
    assert_eq!(m.registers()[0], 0);
}

fn error_unchanged(mut m: Machine, expected: Error) {
    let before = format!("{m:?}");
    assert_eq!(m.step(), Err(expected));
    assert_eq!(format!("{m:?}"), before);
}

#[test]
fn missing_invalid_and_missing_successors() {
    let mut missing = machine(asm::ecall());
    missing.set_pc(0x108);
    error_unchanged(missing, Error::MissingInstruction { pc: 0x108 });
    for word in [
        0,
        0x0000_0001,
        0x0231_00b3,
        0x0010_10f3,
        0x0201_109b,
        0x0201_10b3,
        0x0000_100f,
        0x0020_0073,
        0x4001_1093,
        0x0401_5093,
        0x0001_7083,
        0x0031_4023,
        0x0031_2063,
        0x0001_10e7,
    ] {
        error_unchanged(machine(word), Error::InvalidInstruction { pc: 0x100, word });
    }
    for word in [asm::jal(1, 8), asm::jalr(1, 2, 0), asm::beq(2, 2, 8)] {
        let mut m = machine(word);
        m.set_register(2, 0x108).unwrap();
        m.set_register(1, 71).unwrap();
        error_unchanged(
            m,
            Error::InvalidTarget {
                pc: 0x100,
                target: 0x108,
            },
        );
    }
    let mut m = machine(asm::bne(2, 2, 8));
    assert_eq!(m.step().unwrap().next_pc, 0x104);
    let bad_target = Machine::new(&[(0x100, asm::jal(1, 8)), (0x108, 0)], 0x100, 0, 0, 0).unwrap();
    error_unchanged(
        bad_target,
        Error::InvalidTarget {
            pc: 0x100,
            target: 0x108,
        },
    );
    error_unchanged(
        machine(asm::jal(1, 2)),
        Error::InvalidTarget {
            pc: 0x100,
            target: 0x102,
        },
    );
    let mut fallthrough = Machine::new(&[(0x100, asm::addi(1, 0, 1))], 0x100, 0, 0, 0).unwrap();
    assert_eq!(
        fallthrough.run(1),
        Err(Error::MissingInstruction { pc: 0x104 })
    );
    assert_eq!(fallthrough.registers()[1], 1);
    assert_eq!(fallthrough.pc(), 0x104);
    let mut padding =
        Machine::new(&[(0x100, asm::addi(1, 0, 1)), (0x104, 0)], 0x100, 0, 0, 0).unwrap();
    assert_eq!(
        padding.run(1),
        Err(Error::InvalidInstruction { pc: 0x104, word: 0 })
    );
    assert_eq!(
        padding.run(0),
        Err(Error::InvalidInstruction { pc: 0x104, word: 0 })
    );
}

#[test]
fn ram_boundaries_and_alignment() {
    for (word, address, width) in [
        (asm::lb(1, 2, 0), 0xfff, 1),
        (asm::sb(2, 3, 0), 0x1008, 1),
        (asm::ld(1, 2, 0), 0x1008, 8),
    ] {
        let mut m = Machine::new(&[(0x100, word)], 0x100, 0x1000, 0, 8).unwrap();
        m.set_register(2, address).unwrap();
        error_unchanged(m, Error::OutsideRam { address, width });
    }
    for (word, width) in [
        (asm::lh(1, 2, 0), 2),
        (asm::lw(1, 2, 0), 4),
        (asm::ld(1, 2, 0), 8),
        (asm::sh(2, 3, 0), 2),
        (asm::sw(2, 3, 0), 4),
        (asm::sd(2, 3, 0), 8),
    ] {
        let mut m = machine(word);
        m.set_register(2, 0x1001).unwrap();
        error_unchanged(
            m,
            Error::UnalignedAccess {
                address: 0x1001,
                width,
            },
        );
    }
    let mut last_byte = Machine::new(&[(0x100, asm::lbu(1, 2, 7))], 0x100, 0x1000, 0, 0).unwrap();
    last_byte.set_register(2, 0x1000).unwrap();
    assert_eq!(last_byte.step().unwrap().access.unwrap().address, 0x1007);
    assert_eq!(last_byte.registers()[1], 0);
    let mut top = Machine::new(
        &[
            (0x100, asm::ld(1, 2, 0)),
            (0x104, asm::sd(2, 3, 0)),
            (0x108, asm::ecall()),
        ],
        0x100,
        0xffff_ffff_ffff_fff0,
        1,
        8,
    )
    .unwrap();
    top.set_register(2, 0xffff_ffff_ffff_fff8).unwrap();
    top.set_register(3, 0x0123_4567_89ab_cdef).unwrap();
    top.set_ram_word(1, 0x8877_6655_4433_2211).unwrap();
    let load = top.step().unwrap();
    assert_eq!(load.access.unwrap().word_index, 1);
    assert_eq!(top.registers()[1], 0x8877_6655_4433_2211);
    top.run(1).unwrap();
    assert_eq!(top.ram_word(1).unwrap(), 0x0123_4567_89ab_cdef);
    assert_eq!(top.pc(), 0x108);
    top.set_pc(0x100);
    top.set_register(2, 0).unwrap();
    error_unchanged(
        top,
        Error::OutsideRam {
            address: 0,
            width: 8,
        },
    );
    let mut negative_offset = machine(asm::ld(1, 2, -8));
    negative_offset.set_register(2, 0x1008).unwrap();
    negative_offset
        .set_ram_word(0, 0x8877_6655_4433_2211)
        .unwrap();
    let _ = negative_offset.step().unwrap();
    assert_eq!(negative_offset.registers()[1], 0x8877_6655_4433_2211);
}

#[test]
fn setup_errors_and_sparse_capacity() {
    assert_eq!(
        Machine::new(&[], 0, 4, 0, 0),
        Err(Error::InvalidRam {
            lowest_address: 4,
            log_k_ram: 0
        })
    );
    assert_eq!(
        Machine::new(&[], 0, 0, 62, 0),
        Err(Error::InvalidRam {
            lowest_address: 0,
            log_k_ram: 62
        })
    );
    assert_eq!(
        Machine::new(&[], 0, 0xffff_ffff_ffff_fff8, 1, 0),
        Err(Error::InvalidRam {
            lowest_address: 0xffff_ffff_ffff_fff8,
            log_k_ram: 1
        })
    );
    assert_eq!(
        Machine::new(&[(2, 0)], 0, 0, 0, 0),
        Err(Error::UnalignedProgram { address: 2 })
    );
    assert_eq!(
        Machine::new(&[(4, 0), (4, 1)], 0, 0, 0, 0),
        Err(Error::DuplicateInstruction { address: 4 })
    );
    let mut m = machine(asm::ecall());
    assert_eq!(
        m.set_register(32, 5),
        Err(Error::InvalidRegister { register: 32 })
    );
    assert_eq!(
        m.ram_word(0x0800_0000),
        Err(Error::InvalidRamIndex { index: 0x0800_0000 })
    );
    assert_eq!(
        m.set_ram_word(0x0800_0000, 5),
        Err(Error::InvalidRamIndex { index: 0x0800_0000 })
    );
    let mut m = Machine::new(&[(0x100, asm::sd(2, 3, 0))], 0x100, 0x1000, 27, 0).unwrap();
    m.set_register(2, 0x1000).unwrap();
    error_unchanged(m, Error::RamCapacity { index: 0 });
    assert!(Error::OutsideRam {
        address: 0x1008,
        width: 1
    }
    .to_string()
    .contains("4104"));
}

#[test]
fn sums_one_through_ten() {
    let program = [
        (0, asm::addi(1, 0, 1)),
        (4, asm::addi(2, 0, 11)),
        (8, asm::add(3, 3, 1)),
        (12, asm::addi(1, 1, 1)),
        (16, asm::blt(1, 2, -8)),
        (20, asm::sd(0, 3, 0)),
        (24, asm::jal(0, 0)),
    ];
    let mut m = Machine::new(&program, 0, 0, 27, 1).unwrap();
    m.run(33).unwrap();
    assert_eq!(m.pc(), 24);
    assert_eq!(m.ram_word(0).unwrap(), 55);
    assert_eq!(
        m.registers(),
        &[
            0, 11, 11, 55, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0
        ]
    );
    assert_eq!(
        m.step().unwrap(),
        Record {
            pc: 24,
            word: 0x0000_006f,
            rs1_value: 0,
            rs2_value: 0,
            rd_pre_value: 0,
            rd_post_value: 0,
            access: None,
            next_pc: 24
        }
    );
    assert_eq!(m.pc(), 24);
}

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static COUNT_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: every operation delegates the unchanged allocator contract to System.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT_ALLOCATIONS.with(|enabled| {
            if enabled.get() {
                ALLOCATIONS.with(|count| count.set(count.get() + 1));
            }
        });
        // SAFETY: layout is supplied by the caller of GlobalAlloc::alloc.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller supplies a live System allocation and its layout.
        unsafe {
            System.dealloc(ptr, layout);
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        COUNT_ALLOCATIONS.with(|enabled| {
            if enabled.get() {
                ALLOCATIONS.with(|count| count.set(count.get() + 1));
            }
        });
        // SAFETY: the caller supplies a live allocation and a valid new size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[test]
fn steps_allocate_nothing_even_on_first_store() {
    let program = [
        (0, asm::sd(0, 1, 0)),
        (4, asm::ld(2, 0, 0)),
        (8, asm::addi(1, 1, 1)),
        (12, asm::jal(0, -12)),
    ];
    let mut m = Machine::new(&program, 0, 0, 27, 1).unwrap();
    COUNT_ALLOCATIONS.with(|enabled| enabled.set(true));
    let result = m.run(4096);
    COUNT_ALLOCATIONS.with(|enabled| enabled.set(false));
    result.unwrap();
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
    assert_eq!(m.registers()[1], 1024);
    assert_eq!(m.registers()[2], 1023);
    assert_eq!(m.ram_word(0).unwrap(), 1023);
    assert_eq!(m.pc(), 0);
}

#[test]
fn encoder_rejects_out_of_range_operands() {
    for invalid in [
        (|| asm::add(32, 1, 2)) as fn() -> u32,
        || asm::addi(1, 0, 2048),
        || asm::addi(1, 0, -2049),
        || asm::lui(1, 524_288),
        || asm::sb(1, 2, -2049),
        || asm::beq(1, 2, 3),
        || asm::beq(1, 2, 4096),
        || asm::jal(1, 3),
        || asm::jal(1, 1_048_576),
        || asm::slli(1, 2, -1),
        || asm::slli(1, 2, 64),
        || asm::slliw(1, 2, 32),
        || asm::fence(16, 0),
    ] {
        assert!(std::panic::catch_unwind(invalid).is_err());
    }
}
