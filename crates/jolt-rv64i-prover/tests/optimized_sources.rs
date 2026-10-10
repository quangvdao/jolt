//! Committed-table ground truth for the replay cache and packed kernel source.
#![expect(clippy::unwrap_used, reason = "fixture failures fail the test")]

#[expect(
    dead_code,
    reason = "shared helpers serve all protocol acceptance tests"
)]
mod support;

use common::constants::RAM_START_ADDRESS;
use jolt_field::{Zero, F128};
use jolt_kernels::{KernelError, ProofSession};
use jolt_rv64i_arith::{BitsRow, Layout, WitnessRow, WITNESS_COLUMNS};
use jolt_rv64i_kernels::{
    reduction::g_pass_digits,
    source::{CycleSource, PresentGroup},
};
use jolt_rv64i_prover::{
    backend::Rv64iBackend,
    error::Rv64iProverError,
    optimized::source::{SharedSource, WitnessColumns, WitnessSource},
    plane::{DecodedCycle, DigitFields, Rv64iWitness},
    prover::{prove, ProverPreprocessing},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;
use support::PROGRAMS;

fn witnesses() -> Vec<Rv64iWitness> {
    let mut witnesses: Vec<_> = PROGRAMS
        .into_iter()
        .map(|program| support::program_fixture(program, 6).2)
        .collect();
    let (statement, _, small) = support::counting_loop();
    for (b, a) in [(4, 5), (10, 14)] {
        let layout = Layout::new(b, a, small.layout.lowest_address()).unwrap();
        let program = [
            (RAM_START_ADDRESS, support::asm::sd(1, 2, 0)),
            (RAM_START_ADDRESS + 4, support::asm::sll(3, 4, 5)),
            (RAM_START_ADDRESS + 8, support::asm::slt(6, 7, 8)),
            (RAM_START_ADDRESS + 12, support::asm::bne(9, 10, 4)),
        ];
        let bytecode = Arc::new(support::harness::bytecode(&program, &layout));
        witnesses.push(
            Rv64iWitness::synthetic(
                0x6469_6769_7473,
                layout,
                bytecode,
                &statement.device.memory_layout,
                small.initial_ram.clone(),
                6,
            )
            .unwrap(),
        );
    }
    witnesses
}

fn bit(row: &BitsRow, column: usize) -> bool {
    row[column / 64] >> (column % 64) & 1 != 0
}

fn check_decoded(witness: &Rv64iWitness) {
    let layout = &witness.layout;
    let fields = DigitFields::new(layout);
    let cycles = witness.cycles();
    let mut counts = [0_u64; 64];
    assert_eq!(witness.decoded.len(), witness.bits.len());
    for (cycle, (decoded, committed)) in witness.decoded.iter().zip(witness.bits.iter()).enumerate()
    {
        assert_eq!(decoded.inc, layout.inc(committed));
        assert_eq!(
            fields.bytecode_index().read(decoded),
            layout.bytecode_index(committed)
        );
        assert_eq!(
            fields.ram_index().read(decoded),
            layout.ram_index(committed)
        );
        for (chunk, field) in layout
            .bytecode_ra()
            .iter()
            .zip(fields.bytecode_fields())
            .chain(layout.ram_ra().iter().zip(fields.ram_fields()))
            .chain(layout.pos_ra().iter().zip(fields.pos_fields().iter()))
        {
            assert_eq!(field.bits(), usize::from(chunk.bits()));
            assert_eq!(
                field.read(decoded),
                u64::from(chunk.full(committed).trailing_zeros())
            );
        }
        let pos = layout.pos(committed);
        assert_eq!(
            fields.pos(0).unwrap().read(decoded),
            u64::from(pos) & layout.pos_ra()[0].indicators() as u64
        );
        assert_eq!(
            fields.pos(1).unwrap().read(decoded),
            u64::from(pos) >> layout.pos_ra()[0].bits()
        );
        for (field, column) in [
            (fields.keys_differ(), layout.keys_differ()),
            (fields.should_branch(), layout.should_branch()),
            (fields.jalr_low_bit(), layout.jalr_low_bit()),
        ] {
            assert_eq!(field.read(decoded), u64::from(bit(committed, column)));
        }
        let row = &witness.bytecode.rows()[layout.bytecode_index(committed) as usize];
        let variant = row.variant.unwrap();
        assert_eq!(fields.variant().read(decoded), variant.index() as u64);
        counts[variant.index()] += 1;
        let expected = WitnessRow::compute(
            layout,
            row,
            &witness.words[cycle].base_words(variant.is_store(), layout.inc(committed)),
            committed,
        );
        let found = cycles.row(cycle).unwrap();
        for column in 0..WITNESS_COLUMNS {
            assert_eq!(
                found.bit(column),
                expected.bit(column),
                "cycle {cycle}, column {column}"
            );
        }
    }
    assert_eq!(witness.variant_cycles, counts);
}

#[test]
fn decoded_rows_match_committed_rows_in_both_constructors() {
    assert_eq!(std::mem::size_of::<DecodedCycle>(), 16);
    let fields = DigitFields::new(&Layout::new(3, 46, 0).unwrap());
    assert_eq!((fields.variant().shift(), fields.variant().bits()), (58, 6));
    for witness in witnesses() {
        check_decoded(&witness);
        let from_bits = Rv64iWitness::from_bits(
            witness.layout.clone(),
            Arc::clone(&witness.bytecode),
            Arc::clone(&witness.bits),
            witness.initial_ram.clone(),
            witness.final_pc,
        )
        .unwrap();
        check_decoded(&from_bits);
        assert_eq!(witness.decoded, from_bits.decoded);
        assert_eq!(witness.variant_cycles, from_bits.variant_cycles);
    }
}

fn check_source(witness: &Rv64iWitness) {
    let source = Arc::new(WitnessSource::new(witness).unwrap());
    let columns = source.columns();
    let layout = &witness.layout;
    assert_eq!(source.cycles(), witness.bits.len());
    assert_eq!(source.bytecode_rows(), witness.bytecode.rows().len());
    for column in 0..source.digit_columns() {
        assert_eq!(source.by_row(column), column == columns.variant());
    }
    for (cycle, committed) in witness.bits.iter().enumerate() {
        let index = layout.bytecode_index(committed) as usize;
        let row = &witness.bytecode.rows()[index];
        let variant = row.variant.unwrap();
        assert_eq!(source.bytecode_index(cycle), index);
        assert_eq!(
            source.row_digit(columns.variant(), index),
            Some(variant.index())
        );
        for (word, expected) in [
            (WitnessColumns::rs1_value(), witness.words[cycle].rs1_value),
            (WitnessColumns::rs2_value(), witness.words[cycle].rs2_value),
            (
                WitnessColumns::rd_pre_value(),
                witness.words[cycle].rd_pre_value,
            ),
            (
                WitnessColumns::ram_read_value(),
                witness.words[cycle].ram_read_value,
            ),
            (WitnessColumns::next_pc(), witness.words[cycle].next_pc),
            (WitnessColumns::inc_word(), layout.inc(committed)),
        ] {
            assert_eq!(source.trace_word(word, cycle), expected);
        }
        for (column, width, expected) in [
            (columns.variant(), 6, Some(variant.index())),
            (
                columns.shift_kind(),
                3,
                variant.shift().map(|shift| shift.kind.index()),
            ),
            (
                columns.access_kind(),
                4,
                variant
                    .access()
                    .and_then(|access| access.kind)
                    .map(|kind| kind.index()),
            ),
            (
                columns.key_kind(),
                3,
                variant.key_kind().map(|kind| kind.index()),
            ),
            (columns.branch(), 0, variant.branch().map(|_| 0)),
            (
                columns.keys_differ(),
                0,
                bit(committed, layout.keys_differ()).then_some(0),
            ),
            (
                columns.should_branch(),
                0,
                bit(committed, layout.should_branch()).then_some(0),
            ),
            (
                columns.jalr_low_bit(),
                0,
                bit(committed, layout.jalr_low_bit()).then_some(0),
            ),
        ] {
            assert_eq!(source.bits(column), width);
            assert_eq!(source.digit(column, cycle), expected);
        }
        for (chunk, column) in layout
            .bytecode_ra()
            .iter()
            .zip(columns.bytecode_chunks())
            .chain(layout.ram_ra().iter().zip(columns.ram_chunks()))
        {
            assert_eq!(source.bits(*column), usize::from(chunk.bits()));
            assert_eq!(
                source.digit(*column, cycle),
                Some(chunk.full(committed).trailing_zeros() as usize)
            );
        }
        for (digit, chunk) in layout.pos_ra().into_iter().enumerate() {
            let column = columns.pos(digit).unwrap();
            assert_eq!(source.bits(column), usize::from(chunk.bits()));
            assert_eq!(
                source.digit(column, cycle),
                Some(chunk.full(committed).trailing_zeros() as usize)
            );
        }
        for column in 0..layout.used_columns() {
            let expected = if column < u64::BITS as usize {
                assert_eq!(columns.committed(column), None);
                source.trace_word(WitnessColumns::inc_word(), cycle) >> column & 1 != 0
            } else {
                let (digit, value) = columns.committed(column).unwrap();
                source.digit(digit, cycle) == Some(value)
            };
            assert_eq!(
                bit(committed, column),
                expected,
                "cycle {cycle}, column {column}"
            );
        }
    }
    for (index, row) in witness.bytecode.rows().iter().enumerate() {
        for (word, expected) in [
            (WitnessColumns::imm(), row.imm),
            (WitnessColumns::fall_through_pc(), row.fall_through_pc),
            (WitnessColumns::pc_plus_imm(), row.pc_plus_imm),
            (WitnessColumns::pc(), row.pc),
        ] {
            assert_eq!(source.bytecode_word(word, index), expected);
        }
        assert_eq!(
            source.row_digit(columns.variant(), index),
            row.variant.map(|variant| variant.index())
        );
    }
    for column in layout.used_columns()..256 {
        assert_eq!(columns.committed(column), None);
    }
    let mut rng = StdRng::seed_from_u64(0x7765_6967_6874);
    let mut weights = vec![F128::zero(); 256];
    for weight in &mut weights[..layout.used_columns()] {
        *weight = F128::from_raw(rng.gen());
    }
    let mut shared = SharedSource::default();
    let selector_columns = selector_columns(columns);
    let trace = shared
        .prepare(witness, Some(selector_columns.clone()))
        .unwrap();
    let bytecode = shared.take_bytecode_group().unwrap();
    let ram = shared.take_ram_group().unwrap();
    check_present_group(&bytecode, &source, columns.bytecode_chunks());
    check_present_group(&ram, &source, columns.ram_chunks());
    let selectors = shared.take_selector_group().unwrap();
    assert_eq!(selectors.columns(), selector_columns);
    assert_eq!(selectors.cycles(), source.cycles());
    assert_eq!(
        selectors.bytes().len(),
        source.cycles() * selector_columns.len()
    );
    for (cycle, bytes) in selectors
        .bytes()
        .chunks_exact(selector_columns.len())
        .enumerate()
    {
        for (&column, &found) in selector_columns.iter().zip(bytes) {
            let expected = source.digit(column, cycle).map_or(0, |digit| digit + 1);
            assert_eq!(
                usize::from(found),
                expected,
                "cycle {cycle}, column {column}"
            );
        }
    }
    let tables =
        g_pass_digits(&trace, columns.column_map(), std::slice::from_ref(&weights)).unwrap();
    for (cycle, committed) in witness.bits.iter().enumerate() {
        let expected = weights
            .iter()
            .enumerate()
            .filter(|(column, _)| bit(committed, *column))
            .fold(F128::zero(), |sum, (_, weight)| sum + *weight);
        assert_eq!(tables[0][cycle], expected);
    }
}

fn selector_columns(columns: &WitnessColumns) -> Vec<usize> {
    vec![
        columns.variant(),
        columns.pos(0).unwrap(),
        columns.pos(1).unwrap(),
        columns.shift_kind(),
        columns.access_kind(),
        columns.key_kind(),
        columns.branch(),
        columns.should_branch(),
    ]
}

fn check_present_group(group: &PresentGroup, source: &WitnessSource, columns: &[usize]) {
    assert_eq!(group.columns(), columns);
    assert_eq!(group.cycles(), source.cycles());
    assert_eq!(group.bytes().len(), source.cycles() * columns.len());
    for (cycle, bytes) in group.bytes().chunks_exact(columns.len()).enumerate() {
        for (&column, &found) in columns.iter().zip(bytes) {
            assert_eq!(
                usize::from(found),
                source.digit(column, cycle).unwrap(),
                "cycle {cycle}, column {column}"
            );
        }
    }
}

#[test]
fn source_matches_committed_columns_words_and_weighted_sum() {
    for witness in witnesses() {
        check_source(&witness);
        let from_bits = Rv64iWitness::from_bits(
            witness.layout.clone(),
            Arc::clone(&witness.bytecode),
            Arc::clone(&witness.bits),
            witness.initial_ram.clone(),
            witness.final_pc,
        )
        .unwrap();
        check_source(&from_bits);
    }
}

#[test]
fn bulk_digits_match_scalar_encoding_and_zero_wrong_lengths() {
    for witness in witnesses() {
        let source = WitnessSource::new(&witness).unwrap();
        let columns = source.digit_columns();
        for cycles in [
            0..source.cycles(),
            3..17,
            11..61,
            source.cycles() - 2..source.cycles() + 2,
        ] {
            let len = cycles.len() * columns;
            let mut output = vec![u16::MAX; len];
            source.digits(cycles.clone(), &mut output);
            for (cycle, row) in cycles.clone().zip(output.chunks_exact(columns)) {
                for (column, &found) in row.iter().enumerate() {
                    let expected = source.digit(column, cycle).map_or(0, |digit| {
                        digit.saturating_add(1).min(usize::from(u16::MAX)) as u16
                    });
                    assert_eq!(found, expected, "cycle {cycle}, column {column}");
                }
            }
            for wrong_len in [len - 1, len + 1] {
                let mut output = vec![u16::MAX; wrong_len];
                source.digits(cycles.clone(), &mut output);
                assert!(output.iter().all(|&slot| slot == 0));
            }
        }
        let mut output = [u16::MAX; 7];
        source.digits(0..usize::MAX, &mut output);
        assert_eq!(output, [0; 7]);
    }
}

#[test]
fn shared_source_enforces_group_ownership_and_shared_lifetimes() {
    let (_, _, witness) = support::counting_loop();
    let columns = WitnessColumns::new(&witness.layout);
    let selectors = selector_columns(&columns);
    let mut session = ProofSession::default();
    let shared = session.state_or_insert_with(SharedSource::default);
    assert!(matches!(
        shared.take_bytecode_group(),
        Err(KernelError::InvalidGeometry { .. })
    ));
    assert!(matches!(
        shared.take_ram_group(),
        Err(KernelError::InvalidGeometry { .. })
    ));
    assert!(matches!(
        shared.take_selector_group(),
        Err(KernelError::InvalidGeometry { .. })
    ));
    assert!(matches!(
        shared.plan(),
        Err(KernelError::InvalidGeometry { .. })
    ));
    let trace = shared.prepare(&witness, Some(selectors.clone())).unwrap();
    let warm = shared.prepare(&witness, None).unwrap();
    let router_warm = shared.prepare(&witness, Some(selectors.clone())).unwrap();
    assert!(Arc::ptr_eq(&trace, &warm));
    assert!(Arc::ptr_eq(&trace, &router_warm));
    let _bytecode = shared.take_bytecode_group().unwrap();
    let _ram = shared.take_ram_group().unwrap();
    let _selectors = shared.take_selector_group().unwrap();
    assert!(matches!(
        shared.take_bytecode_group(),
        Err(KernelError::InvalidGeometry { .. })
    ));
    assert!(matches!(
        shared.take_ram_group(),
        Err(KernelError::InvalidGeometry { .. })
    ));
    assert!(matches!(
        shared.take_selector_group(),
        Err(KernelError::InvalidGeometry { .. })
    ));
    assert!(matches!(
        shared.release_plan(),
        Err(KernelError::InvalidGeometry { .. })
    ));
    let plan = shared.plan().unwrap();
    let plan_again = shared.plan().unwrap();
    assert!(Arc::ptr_eq(&plan, &plan_again));
    let weak_plan = Arc::downgrade(&plan);
    shared.release_plan().unwrap();
    shared.release_plan().unwrap();
    assert!(matches!(
        shared.plan(),
        Err(KernelError::InvalidGeometry { .. })
    ));
    assert!(weak_plan.upgrade().is_some());
    drop(plan);
    drop(plan_again);
    assert!(weak_plan.upgrade().is_none());
    let after_take = shared.prepare(&witness, Some(selectors.clone())).unwrap();
    assert!(Arc::ptr_eq(&trace, &after_take));
    assert!(matches!(
        shared.take_bytecode_group(),
        Err(KernelError::InvalidGeometry { .. })
    ));
    assert!(matches!(
        shared.take_ram_group(),
        Err(KernelError::InvalidGeometry { .. })
    ));
    assert!(matches!(
        shared.take_selector_group(),
        Err(KernelError::InvalidGeometry { .. })
    ));
    let weak_trace = Arc::downgrade(&trace);
    let weak_source = Arc::downgrade(trace.source());
    drop(trace);
    drop(warm);
    drop(router_warm);
    drop(after_take);
    assert!(weak_trace.upgrade().is_some());
    assert!(weak_source.upgrade().is_some());
    drop(session);
    assert!(weak_trace.upgrade().is_none());
    assert!(weak_source.upgrade().is_none());

    let mut tail_session = ProofSession::default();
    let tail = tail_session.state_or_insert_with(SharedSource::default);
    let trace = tail.prepare(&witness, None).unwrap();
    assert!(matches!(
        tail.take_selector_group(),
        Err(KernelError::InvalidGeometry { .. })
    ));
    assert!(matches!(
        tail.prepare(&witness, Some(selectors)),
        Err(KernelError::InvalidGeometry { .. })
    ));
    let warm = tail.prepare(&witness, None).unwrap();
    assert!(Arc::ptr_eq(&trace, &warm));
    let _bytecode = tail.take_bytecode_group().unwrap();
    let _ram = tail.take_ram_group().unwrap();
    assert!(matches!(
        tail.take_selector_group(),
        Err(KernelError::InvalidGeometry { .. })
    ));
}

#[test]
fn source_indices_are_total_and_wrong_decoded_length_is_rejected() {
    let (statement, preprocessing, mut witness) = support::counting_loop();
    let source = WitnessSource::new(&witness).unwrap();
    for cycle in [source.cycles(), usize::MAX] {
        for word in 0..=source.trace_words() {
            assert_eq!(source.trace_word(word, cycle), 0);
        }
        assert_eq!(source.bytecode_index(cycle), 0);
        for column in 0..=source.digit_columns() {
            assert_eq!(source.digit(column, cycle), None);
        }
    }
    for word in [source.trace_words(), usize::MAX] {
        assert_eq!(source.trace_word(word, 0), 0);
    }
    for row in [source.bytecode_rows(), usize::MAX] {
        for word in 0..=source.bytecode_words() {
            assert_eq!(source.bytecode_word(word, row), 0);
        }
        assert_eq!(source.row_digit(source.columns().variant(), row), None);
    }
    for word in [source.bytecode_words(), usize::MAX] {
        assert_eq!(source.bytecode_word(word, 0), 0);
    }
    for column in [source.digit_columns(), usize::MAX] {
        assert_eq!(source.bits(column), 0);
        assert!(!source.by_row(column));
        assert_eq!(source.digit(column, 0), None);
        assert_eq!(source.row_digit(column, 0), None);
        assert_eq!(source.columns().committed(column), None);
    }
    assert!(matches!(
        witness.cycles().row(usize::MAX),
        Err(Rv64iProverError::CycleIndex { .. })
    ));
    witness.decoded = witness.decoded[..63].iter().copied().collect();
    assert!(matches!(
        WitnessSource::new(&witness),
        Err(Rv64iProverError::DecodedLength {
            expected: 64,
            found: 63
        })
    ));
    let result = prove(
        &ProverPreprocessing {
            verifier: preprocessing,
            scheme: (),
        },
        &statement,
        &witness,
        &Rv64iBackend::reference(),
    );
    assert!(matches!(
        result,
        Err(Rv64iProverError::DecodedLength {
            expected: 64,
            found: 63
        })
    ));
}
