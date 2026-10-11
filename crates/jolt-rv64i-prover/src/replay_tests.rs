#![expect(clippy::unwrap_used, reason = "malformed fixtures fail their tests")]

use super::{DigitFields, Rv64iWitness};
use crate::error::{FactField, Rv64iProverError};
use jolt_program::image::decode::decode_instruction;
use jolt_riscv::RV64I;
use jolt_rv64i_arith::{Bytecode, CycleFacts, Layout, Variant};
use rayon::ThreadPoolBuilder;
use std::sync::Arc;

fn narrow_store_bytecode() -> (Layout, Arc<Bytecode>) {
    let layout = Layout::new(2, 5, 0).unwrap();
    let instructions: Vec<_> = [0x0000_006f, 0x0000_0423, 0x0000_1523, 0x0080_3183]
        .into_iter()
        .enumerate()
        .map(|(index, word)| decode_instruction(word, 4 * index as u64, false, RV64I).unwrap())
        .collect();
    let bytecode = Arc::new(Bytecode::preprocess(&instructions, &layout).unwrap());
    (layout, bytecode)
}

fn assert_narrow_store_literals(
    witness: &Rv64iWitness,
    byte_store: usize,
    halfword_store: usize,
    loads: &[usize],
) {
    assert_eq!(
        witness.words[byte_store].ram_read_value,
        0x8877_6655_4433_2211
    );
    assert_eq!(
        witness.words[halfword_store].ram_read_value,
        0x8877_6655_4433_2201
    );
    assert_eq!(
        witness.words[byte_store].ram_read_value.to_le_bytes(),
        [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
    );
    assert_eq!(
        witness.words[halfword_store].ram_read_value.to_le_bytes(),
        [0x01, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
    );
    for &cycle in loads {
        assert_eq!(witness.words[cycle].ram_read_value, 0x8877_6655_4411_2201);
        assert_eq!(
            witness.words[cycle].ram_read_value.to_le_bytes(),
            [0x01, 0x22, 0x11, 0x44, 0x55, 0x66, 0x77, 0x88]
        );
    }
    let mut expected_ram = vec![0; 32];
    expected_ram[1] = 0x8877_6655_4411_2201;
    assert_eq!(witness.final_ram, expected_ram);
    assert_eq!(witness.variant_cycles[Variant::SB.index()], 1);
    assert_eq!(witness.variant_cycles[Variant::SH.index()], 1);
    assert_eq!(
        witness.variant_cycles[Variant::LD.index()],
        loads.len() as u64
    );
    assert_eq!(
        witness.variant_cycles[Variant::JAL_X0.index()],
        8190 - loads.len() as u64
    );
    assert_eq!(witness.variant_cycles.iter().sum::<u64>(), 8192);
    let fields = DigitFields::new(&witness.layout);
    assert_eq!(witness.decoded[byte_store].inc, 0x10);
    assert_eq!(witness.decoded[halfword_store].inc, 0x0022_0000);
    assert_eq!(fields.pos(0).unwrap().read(&witness.decoded[byte_store]), 0);
    assert_eq!(
        fields
            .pos(0)
            .unwrap()
            .read(&witness.decoded[halfword_store]),
        2
    );
    assert_eq!(witness.words[byte_store].next_pc, 8);
    assert_eq!(witness.words[halfword_store].next_pc, 12);
    assert_eq!(witness.words[8191].next_pc, 0);
}

#[test]
fn adjacent_chunk_narrow_stores_preserve_literal_whole_words() {
    let (layout, bytecode) = narrow_store_bytecode();
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut bits = vec![[0; 4]; 8192];
        for (cycle, index, inc, pos) in [
            (4095, 1, 0x10, 0),
            (4096, 2, 0x0022_0000, 2),
            (4097, 3, 0x8877_6655_4411_2201, 0),
        ] {
            layout
                .write_bytecode_index(&mut bits[cycle], index)
                .unwrap();
            layout.write_ram_index(&mut bits[cycle], 1).unwrap();
            layout.write_pos(&mut bits[cycle], pos).unwrap();
            bits[cycle][0] = inc;
        }
        let witness = pool
            .install(|| {
                Rv64iWitness::from_bits(
                    layout.clone(),
                    Arc::clone(&bytecode),
                    bits.into(),
                    vec![(1, 0x8877_6655_4433_2211)],
                    0,
                )
            })
            .unwrap();
        assert_narrow_store_literals(&witness, 4095, 4096, &[4097]);
        assert_eq!(witness.words[4094].next_pc, 4);
        assert_eq!(witness.words[4097].next_pc, 0);
        assert_eq!(witness.words[4098].rd_pre_value, 0);

        let mut facts = vec![CycleFacts::default(); 8192];
        facts[4095] = CycleFacts {
            bytecode_index: 1,
            ram_word_index: 1,
            ram_pre_value: 0x8877_6655_4433_2211,
            ram_post_value: 0x8877_6655_4433_2201,
            next_pc: 8,
            ..CycleFacts::default()
        };
        facts[4096] = CycleFacts {
            bytecode_index: 2,
            ram_word_index: 1,
            ram_pre_value: 0x8877_6655_4433_2201,
            ram_post_value: 0x8877_6655_4411_2201,
            next_pc: 12,
            ..CycleFacts::default()
        };
        facts[4097] = CycleFacts {
            bytecode_index: 3,
            ram_word_index: 1,
            ram_pre_value: 0x8877_6655_4411_2201,
            rd_post_value: 0x8877_6655_4411_2201,
            ..CycleFacts::default()
        };
        let witness = pool
            .install(|| {
                Rv64iWitness::from_facts(
                    layout.clone(),
                    Arc::clone(&bytecode),
                    &facts,
                    vec![(1, 0x8877_6655_4433_2211)],
                )
            })
            .unwrap();
        assert_narrow_store_literals(&witness, 4095, 4096, &[4097]);
        assert_eq!(witness.words[4094].next_pc, 4);
        assert_eq!(witness.words[4097].next_pc, 0);
        facts[4096].ram_pre_value = 0x8877_6655_4433_2211;
        facts[8191].bytecode_index = u32::MAX;
        let result = pool.install(|| {
            Rv64iWitness::from_facts(
                layout.clone(),
                Arc::clone(&bytecode),
                &facts,
                vec![(1, 0x8877_6655_4433_2211)],
            )
        });
        assert!(matches!(
            result,
            Err(Rv64iProverError::FactMismatch {
                cycle: 4096,
                field: FactField::RamPreValue,
                expected: 0x8877_6655_4433_2201,
                found: 0x8877_6655_4433_2211,
            })
        ));
    }
}

#[test]
fn repeated_chunk_stores_accumulate_literal_xor_deltas() {
    let (layout, bytecode) = narrow_store_bytecode();
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut bits = vec![[0; 4]; 8192];
        for (cycle, index, inc, pos) in [
            (100, 1, 0x10, 0),
            (101, 2, 0x0022_0000, 2),
            (102, 3, 0x8877_6655_4411_2201, 0),
            (4097, 3, 0, 0),
        ] {
            layout
                .write_bytecode_index(&mut bits[cycle], index)
                .unwrap();
            layout.write_ram_index(&mut bits[cycle], 1).unwrap();
            layout.write_pos(&mut bits[cycle], pos).unwrap();
            bits[cycle][0] = inc;
        }
        let witness = pool
            .install(|| {
                Rv64iWitness::from_bits(
                    layout.clone(),
                    Arc::clone(&bytecode),
                    bits.into(),
                    vec![(1, 0x8877_6655_4433_2211)],
                    0,
                )
            })
            .unwrap();
        assert_narrow_store_literals(&witness, 100, 101, &[102, 4097]);
        assert_eq!(witness.words[4095].next_pc, 0);
        assert_eq!(witness.words[4096].next_pc, 12);
        assert_eq!(witness.words[4097].rd_pre_value, 0x8877_6655_4411_2201);

        let mut facts = vec![CycleFacts::default(); 8192];
        facts[100] = CycleFacts {
            bytecode_index: 1,
            ram_word_index: 1,
            ram_pre_value: 0x8877_6655_4433_2211,
            ram_post_value: 0x8877_6655_4433_2201,
            next_pc: 8,
            ..CycleFacts::default()
        };
        facts[101] = CycleFacts {
            bytecode_index: 2,
            ram_word_index: 1,
            ram_pre_value: 0x8877_6655_4433_2201,
            ram_post_value: 0x8877_6655_4411_2201,
            next_pc: 12,
            ..CycleFacts::default()
        };
        facts[102] = CycleFacts {
            bytecode_index: 3,
            ram_word_index: 1,
            ram_pre_value: 0x8877_6655_4411_2201,
            rd_post_value: 0x8877_6655_4411_2201,
            ..CycleFacts::default()
        };
        facts[4097] = CycleFacts {
            bytecode_index: 3,
            ram_word_index: 1,
            ram_pre_value: 0x8877_6655_4411_2201,
            rd_pre_value: 0x8877_6655_4411_2201,
            rd_post_value: 0x8877_6655_4411_2201,
            ..CycleFacts::default()
        };
        let witness = pool
            .install(|| {
                Rv64iWitness::from_facts(
                    layout.clone(),
                    Arc::clone(&bytecode),
                    &facts,
                    vec![(1, 0x8877_6655_4433_2211)],
                )
            })
            .unwrap();
        assert_narrow_store_literals(&witness, 100, 101, &[102, 4097]);
        assert_eq!(witness.words[4095].next_pc, 0);
        assert_eq!(witness.words[4096].next_pc, 12);
        assert_eq!(witness.words[4097].rd_pre_value, 0x8877_6655_4411_2201);
        facts[4097].ram_pre_value = 0x8877_6655_4411_2211;
        facts[8191].bytecode_index = u32::MAX;
        let result = pool.install(|| {
            Rv64iWitness::from_facts(
                layout.clone(),
                Arc::clone(&bytecode),
                &facts,
                vec![(1, 0x8877_6655_4433_2211)],
            )
        });
        assert!(matches!(
            result,
            Err(Rv64iProverError::FactMismatch {
                cycle: 4097,
                field: FactField::RamPreValue,
                expected: 0x8877_6655_4411_2201,
                found: 0x8877_6655_4411_2211,
            })
        ));
    }
}
