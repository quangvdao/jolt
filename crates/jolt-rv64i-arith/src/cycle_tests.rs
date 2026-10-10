#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures fail loudly and index fixed-size arrays"
)]

use jolt_program::image::decode::decode_instruction;
use jolt_riscv::{
    NormalizedOperands, SourceInstruction, SourceInstructionKind as Kind, SourceInstructionRow,
    RV64IMAC_JOLT,
};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::{BitsBuilder, CycleFacts, WitnessError};
use crate::bytecode::{Bytecode, BytecodeColumn, BytecodeRow};
use crate::decode::{eval, load_form, shift_form, store_form, Source, Sources};
use crate::layout::{chunk_indicators, set_bit, BitsRow, Chunk, Layout};
use crate::rows::{RowGroup, RowSystem};
use crate::variant::{AccessKind, ShiftKind, Variant};
use crate::words::{BaseWords, Lane, WitnessRow, Words};

fn program(layout: &Layout) -> Bytecode {
    let words = [
        0x0020_81b3,
        0x0020_f1b3,
        0x0020_b1b3,
        0x0020_b1b3,
        0x0020_b1b3,
        0x0020_8463,
        0x0020_c1b3,
        0x0020_b023,
        0x0010_8193,
        0x0010_9193,
        0x0010_9193,
        0x0030_8183,
        0x0020_9323,
        0x0010_8067,
        0xfe20_9ce3,
        0x0010_0073,
    ];
    let instructions: Vec<SourceInstruction> = words
        .iter()
        .enumerate()
        .map(|(i, word)| {
            decode_instruction(*word, 0x1000 + 4 * i as u64, false, RV64IMAC_JOLT).unwrap()
        })
        .collect();
    Bytecode::preprocess(&instructions, layout).unwrap()
}

fn facts(index: u32, rs1: u64, rs2: u64, post: u64) -> CycleFacts {
    CycleFacts {
        bytecode_index: index,
        rs1_value: rs1,
        rs2_value: rs2,
        rd_post_value: post,
        next_pc: 0x1004 + u64::from(index) * 4,
        ..CycleFacts::default()
    }
}

fn cases() -> [(CycleFacts, BitsRow); 15] {
    [
        (facts(0, 1, 1, 2), [2, 0, 0, 0]),
        (facts(1, 3, 5, 1), [1, 1, 0, 0]),
        (facts(2, 1, 2, 1), [1, 0x1000_4000_0002, 0, 0]),
        (facts(3, 6, 5, 0), [0, 0x1000_4000_0004, 0, 0]),
        (facts(4, 2, 1, 0), [0, 0x1000_4000_0008, 0, 0]),
        (facts(5, 1, 2, 0), [0, 0x1000_4000_0010, 0, 0]),
        (facts(6, 1, 2, 3), [3, 0x20, 0, 0]),
        (
            CycleFacts {
                ram_word_index: 1,
                ram_post_value: 5,
                ..facts(7, 8, 5, 0)
            },
            [5, 0x8040, 0, 0],
        ),
        (facts(8, 0, 0, 1), [1, 0x80, 0, 0]),
        (facts(9, 1, 0, 2), [2, 0x4000_0100, 0, 0]),
        (facts(10, 1, 0, 2), [2, 0x4000_0200, 0, 0]),
        (
            CycleFacts {
                rd_pre_value: 7,
                ram_word_index: 1,
                ram_pre_value: 0x8000_0000,
                ram_post_value: u64::MAX,
                ..facts(11, 8, 0, 0xffff_ffff_ffff_ff80)
            },
            [0xffff_ffff_ffff_ff87, 0x0001_0000_8400, 0, 0],
        ),
        (
            CycleFacts {
                ram_word_index: 1,
                ram_pre_value: 0x1122_3344_5566_7788,
                ram_post_value: 0xabcd_3344_5566_7788,
                ..facts(12, 8, 0xabcd, 0)
            },
            [0xbaef_0000_0000_0000, 0x0008_0000_8800, 0, 0],
        ),
        (
            CycleFacts {
                next_pc: 0x1000,
                ..facts(13, 0x1000, 0, 0)
            },
            [0, 0x4000_0000_1000, 0, 0],
        ),
        (
            CycleFacts {
                next_pc: 0x1030,
                ..facts(14, 1, 2, 0)
            },
            [0, 0x3000_4000_2000, 0, 0],
        ),
    ]
}

#[test]
fn hand_computed_cycles_and_words() {
    let layout = Layout::new(4, 4, 0).unwrap();
    let bytecode = program(&layout);
    let builder = BitsBuilder::new(&layout, &bytecode).unwrap();
    let system = RowSystem::new(&layout);
    for (fact, expected) in cases() {
        let bits = builder.bits_row(&fact).unwrap();
        assert_eq!(bits, expected, "cycle {}", fact.bytecode_index);
        let row = &bytecode.rows()[fact.bytecode_index as usize];
        let base = BaseWords::from_facts(&fact);
        let z = WitnessRow::compute(&layout, row, &base, &bits);
        assert_eq!(
            z,
            WitnessRow::assemble(&Words::compute(&layout, row, &base, &bits), &bits)
        );
        assert!(
            system.failing_rows(&z).is_empty(),
            "cycle {}: {:?}",
            fact.bytecode_index,
            system.check(&z)
        );
        assert_eq!(z.lane(Lane::Bits0), bits[0]);
        assert_eq!(z.lane(Lane::Bits1), bits[1]);
        let mut wrong = bits;
        wrong[0] ^= 1;
        let mut wrong_base = base;
        if row.variant.unwrap().is_store() {
            let reconstructed_post = fact.ram_pre_value ^ layout.inc(&wrong);
            assert_ne!(reconstructed_post, fact.ram_post_value);
        } else {
            wrong_base.rd_write_value = fact.rd_pre_value ^ layout.inc(&wrong);
        }
        assert!(!system
            .failing_rows(&WitnessRow::compute(&layout, row, &wrong_base, &wrong))
            .is_empty());
    }
    let (load, bits) = cases()[11];
    let words = Words::compute(
        &layout,
        &bytecode.rows()[11],
        &BaseWords::from_facts(&load),
        &bits,
    );
    assert_eq!(words.add_sum, 11);
    assert_eq!(words.load_output, 0xffff_ffff_ffff_ff80);
    let (store, bits) = cases()[12];
    assert_eq!(
        Words::compute(
            &layout,
            &bytecode.rows()[12],
            &BaseWords::from_facts(&store),
            &bits
        )
        .store_inc,
        0xbaef_0000_0000_0000
    );
    let (add, bits) = cases()[0];
    let words = Words::compute(
        &layout,
        &bytecode.rows()[0],
        &BaseWords::from_facts(&add),
        &bits,
    );
    assert_eq!(
        (
            words.carry,
            words.carry_left,
            words.carry_right,
            words.carry_step
        ),
        (2, 3, 3, 3)
    );
}

#[test]
fn each_row_group_has_an_independent_failure_signal() {
    let layout = Layout::new(4, 4, 0).unwrap();
    let bytecode = program(&layout);
    let builder = BitsBuilder::new(&layout, &bytecode).unwrap();
    let system = RowSystem::new(&layout);
    let groups = [
        RowGroup::Adder,
        RowGroup::And,
        RowGroup::LessThan,
        RowGroup::KeyBits,
        RowGroup::KeysAgreeAbove,
        RowGroup::KeysEqual,
        RowGroup::RdResidual,
        RowGroup::RamResidual,
        RowGroup::NextPCResidual,
        RowGroup::ControlResidual,
        RowGroup::OneHot,
    ];
    for (i, group) in groups.into_iter().enumerate() {
        let (mut fact, _) = cases()[i];
        match i {
            0 => fact.rd_post_value = 3,
            1 => fact.rd_post_value = 7,
            2 => fact.rd_post_value = 0,
            3 | 4 => fact.rd_post_value = 1,
            5 => fact.next_pc = 0x101c,
            6 => fact.rd_post_value = 0,
            7 => fact.ram_post_value = 0,
            8 => fact.next_pc = 0x1000,
            9 => fact.rd_post_value = 4,
            10 => fact.rd_post_value = 13,
            _ => unreachable!(),
        }
        let mut bits = builder.bits_row(&fact).unwrap();
        match i {
            3 => layout.write_pos(&mut bits, 2).unwrap(),
            4 => layout.write_pos(&mut bits, 0).unwrap(),
            5 => {
                layout.write_pos(&mut bits, 2).unwrap();
                set_bit(&mut bits, layout.keys_differ(), false);
                set_bit(&mut bits, layout.should_branch(), true);
            }
            9 => layout.write_pos(&mut bits, 2).unwrap(),
            10 => {
                layout.write_pos(&mut bits, 0).unwrap();
                set_bit(&mut bits, usize::from(layout.pos_ra()[0].start()) + 1, true);
                set_bit(&mut bits, usize::from(layout.pos_ra()[0].start()) + 2, true);
            }
            _ => {}
        }
        let row = &bytecode.rows()[i];
        let mut base = BaseWords::from_facts(&fact);
        if !row.variant.unwrap().is_store() {
            base.rd_write_value = fact.rd_pre_value ^ layout.inc(&bits);
        } else {
            assert_eq!(fact.ram_pre_value ^ layout.inc(&bits), fact.ram_post_value);
        }
        let failures = system.failing_rows(&WitnessRow::compute(&layout, row, &base, &bits));
        assert!(!failures.is_empty(), "{group:?}");
        for failure in failures.iter() {
            assert_eq!(
                system.group_of(failure),
                Some(group),
                "cycle {i}, row {failure}"
            );
        }
    }
}

#[test]
fn jalr_local_contract_leaves_low_bit_split_to_surrounding_protocol() {
    let layout = Layout::new(1, 1, 0).unwrap();
    let instruction = decode_instruction(0x0010_8067, 0, false, RV64IMAC_JOLT).unwrap();
    let bytecode = Bytecode::preprocess(&[instruction], &layout).unwrap();
    let builder = BitsBuilder::new(&layout, &bytecode).unwrap();
    let system = RowSystem::new(&layout);
    let fact = facts(0, 0, 0, 0);
    let mut bits = builder.bits_row(&fact).unwrap();
    for (next_pc, low_bit) in [(0, true), (1, false)] {
        set_bit(&mut bits, layout.jalr_low_bit(), low_bit);
        let base = BaseWords {
            next_pc,
            ..BaseWords::from_facts(&fact)
        };
        assert!(system
            .failing_rows(&WitnessRow::compute(
                &layout,
                &bytecode.rows()[0],
                &base,
                &bits
            ))
            .is_empty());
    }
}

#[test]
fn fill_is_stateless_and_reports_first_error() {
    let layout = Layout::new(4, 4, 0).unwrap();
    let bytecode = program(&layout);
    let builder = BitsBuilder::new(&layout, &bytecode).unwrap();
    let facts: Vec<_> = cases().into_iter().map(|(fact, _)| fact).collect();
    let mut whole = [[u64::MAX; 4]; 15];
    builder.fill(&facts, &mut whole).unwrap();
    for split in 0..=facts.len() {
        let mut chunked = [[0; 4]; 15];
        let (left, right) = chunked.split_at_mut(split);
        builder.fill(&facts[..split], left).unwrap();
        builder.fill(&facts[split..], right).unwrap();
        assert_eq!(whole, chunked);
    }
    let mut untouched = [[17; 4]; 1];
    let error = builder.fill(&facts, &mut untouched).unwrap_err();
    assert_eq!(error.cycle, 1);
    assert_eq!(
        error.error,
        WitnessError::LengthMismatch { facts: 15, rows: 1 }
    );
    assert_eq!(untouched, [[17; 4]; 1]);
    let mut invalid = facts;
    invalid[3].bytecode_index = 16;
    let mut out = [[0; 4]; 15];
    assert_eq!(builder.fill(&invalid, &mut out).unwrap_err().cycle, 3);
    assert_eq!(&out[..3], &whole[..3]);
    assert_eq!(out[3], [0; 4]);
    assert!(matches!(
        builder.bits_row(&invalid[3]),
        Err(WitnessError::BytecodeIndexOutOfRange { bytecode_index: 16 })
    ));
    let short = Bytecode::preprocess(&[], &layout).unwrap();
    assert!(matches!(
        BitsBuilder::new(&layout, &short)
            .unwrap()
            .bits_row(&CycleFacts::default()),
        Err(WitnessError::InvalidRow { bytecode_index: 0 })
    ));
    let other = Layout::new(3, 4, 0).unwrap();
    assert!(matches!(
        BitsBuilder::new(&other, &bytecode),
        Err(WitnessError::LayoutMismatch)
    ));
    let shifted = Layout::new(4, 4, 8).unwrap();
    assert!(matches!(
        BitsBuilder::new(&shifted, &bytecode),
        Err(WitnessError::LayoutMismatch)
    ));
}

#[test]
fn access_errors_and_ignored_facts_are_pinned() {
    let layout = Layout::new(4, 4, 0).unwrap();
    let bytecode = program(&layout);
    let builder = BitsBuilder::new(&layout, &bytecode).unwrap();
    let store = CycleFacts {
        rs1_value: 128,
        ..cases()[7].0
    };
    assert_eq!(
        builder.bits_row(&store),
        Err(WitnessError::AddressOutsideRam {
            relative_address: 128
        })
    );
    let store = CycleFacts {
        rs1_value: 9,
        ..cases()[7].0
    };
    assert_eq!(
        builder.bits_row(&store),
        Err(WitnessError::UnalignedAccess {
            relative_address: 9,
            width: 8
        })
    );
    let store = CycleFacts {
        ram_word_index: 2,
        ..cases()[7].0
    };
    assert_eq!(
        builder.bits_row(&store),
        Err(WitnessError::RamWordIndexMismatch {
            expected: 1,
            found: 2
        })
    );
    let (load, expected) = cases()[11];
    for post in [0, 1, u64::MAX] {
        assert_eq!(
            builder
                .bits_row(&CycleFacts {
                    ram_post_value: post,
                    ..load
                })
                .unwrap(),
            expected
        );
    }
    let fact = facts(0, 1, 1, 2);
    let bits = builder.bits_row(&fact).unwrap();
    let changed = CycleFacts {
        ram_word_index: u64::MAX,
        ram_pre_value: u64::MAX,
        ram_post_value: u64::MAX,
        next_pc: 19,
        ..fact
    };
    assert_eq!(builder.bits_row(&changed).unwrap(), bits);
    let original = BaseWords::from_facts(&fact);
    let changed = BaseWords {
        ram_read_value: u64::MAX,
        ..original
    };
    assert_eq!(
        WitnessRow::compute(&layout, &bytecode.rows()[0], &original, &bits),
        WitnessRow::compute(&layout, &bytecode.rows()[0], &changed, &bits)
    );
    let invalid = Words::compute(&layout, &Default::default(), &original, &[u64::MAX; 4]);
    assert_eq!(
        invalid,
        Words {
            control_residual: 1,
            ..Words::default()
        }
    );
}

#[test]
fn random_public_inputs_are_total() {
    let layout = Layout::new(4, 4, 0).unwrap();
    let bytecode = program(&layout);
    let builder = BitsBuilder::new(&layout, &bytecode).unwrap();
    let system = RowSystem::new(&layout);
    let mut rng = ChaCha8Rng::seed_from_u64(0x726f_7773);
    assert_eq!(chunk_indicators(usize::MAX), usize::MAX);
    for &kind in Kind::ALL {
        let source = SourceInstruction::new(
            kind,
            SourceInstructionRow {
                address: rng.next_u64() as usize,
                operands: NormalizedOperands {
                    rs1: Some(rng.next_u32() as u8),
                    rs2: Some(rng.next_u32() as u8),
                    rd: Some(rng.next_u32() as u8),
                    imm: (u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64())) as i128,
                },
                inline: None,
                is_compressed: false,
            },
        );
        let _row = BytecodeRow::from_source(&source, rng.next_u64());
        let _preprocessed = Bytecode::preprocess(&[source], &layout);
        let _variant = Variant::from_source(kind, rng.next_u32() & 1 != 0);
    }
    for i in 0..4096 {
        let _random_layout = Layout::new(
            rng.next_u32() as usize,
            rng.next_u32() as usize,
            rng.next_u64(),
        );
        let _random_chunk = Chunk::new(rng.next_u32() as u16, rng.next_u32() as u8);
        let fact = CycleFacts {
            bytecode_index: if i % 2 == 0 {
                rng.next_u32()
            } else {
                rng.next_u32() % 16
            },
            rs1_value: rng.next_u64(),
            rs2_value: rng.next_u64(),
            rd_pre_value: rng.next_u64(),
            rd_post_value: rng.next_u64(),
            ram_word_index: rng.next_u64(),
            ram_pre_value: rng.next_u64(),
            ram_post_value: rng.next_u64(),
            next_pc: rng.next_u64(),
        };
        let _result = builder.bits_row(&fact);
        let _fill_result = builder.fill(&[fact], &mut [[0; 4]]);
        let bits = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];
        let row = BytecodeRow {
            variant: bytecode.rows()[i % 16].variant,
            pc: rng.next_u64(),
            imm: rng.next_u64(),
            fall_through_pc: rng.next_u64(),
            pc_plus_imm: rng.next_u64(),
            rs1: rng.next_u32() as u8,
            rs2: rng.next_u32() as u8,
            rd: rng.next_u32() as u8,
        };
        let base = BaseWords::from_facts(&fact);
        let src = Sources::new(&layout, &row, &base, &bits);
        for source in Source::ALL {
            let _word = src.get(source);
        }
        let words = Words::compute(&layout, &row, &base, &bits);
        let _computed = WitnessRow::compute(&layout, &row, &base, &bits);
        let _assembled = WitnessRow::assemble(&words, &bits);
        let random = WitnessRow(std::array::from_fn(|_| rng.next_u64()));
        let failures = system.failing_rows(&random);
        let _check_result = system.check(&random);
        assert!(!failures.contains(usize::MAX));
        assert_eq!(random.bit(usize::MAX), None);
        assert_eq!(system.group_of(usize::MAX), None);
        for lane in system.lane_rows() {
            let _values = lane.values(&random);
        }
        for packed in system.packed_rows() {
            let _values = packed.values(&random);
        }
        for column in BytecodeColumn::ALL {
            let _column_value = row.column(column);
        }
        let _pc_index = bytecode.index_of_pc(rng.next_u64());
        let _final_index = bytecode.final_pc_index(rng.next_u64());
        for chunk in layout.chunks() {
            let _stored = chunk.stored(&bits);
            let _full = chunk.full(&bits);
            let _mask = chunk.digit_bit_mask(rng.next_u32() as u8);
            let _digit = chunk.digit_bit(&bits, rng.next_u32() as u8);
            let _write = chunk.write_digit(&mut [0; 4], rng.next_u32() as u8);
        }
        let _inc = layout.inc(&bits);
        let _bc = layout.bytecode_index(&bits);
        let _ram = layout.ram_index(&bits);
        let _pos = layout.pos(&bits);
        let _bc_write = layout.write_bytecode_index(&mut [0; 4], rng.next_u64());
        let _ram_write = layout.write_ram_index(&mut [0; 4], rng.next_u64());
        let _pos_write = layout.write_pos(&mut [0; 4], rng.next_u32() as u8);
        let pos = rng.next_u32() as u8;
        for kind in ShiftKind::ALL {
            if let Ok(form) = shift_form(kind, pos) {
                let _word = eval(form.terms(), &src);
            }
        }
        for kind in AccessKind::ALL {
            let _load = load_form(kind, pos);
            let _store = store_form(kind, pos);
        }
    }
}
