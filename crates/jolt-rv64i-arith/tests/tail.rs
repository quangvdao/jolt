//! Binary tail wire order and the nonzero tail of absent rails.
#![expect(
    clippy::unwrap_used,
    reason = "invalid fixtures fail their enclosing tests"
)]

use jolt_rv64i_arith::decode::SourceParts;
use jolt_rv64i_arith::words::F2Words;
use jolt_rv64i_arith::{BaseWords, BytecodeRow, Layout, RowSystem, Sources, Variant};

#[test]
fn reference_tail_table_matches_wire_literal() {
    let rows = RowSystem::new(&Layout::new(20, 20, 0).unwrap());
    assert_eq!(
        rows.f2_tail().unwrap().table(),
        &[
            0x08, 0x0a, 0x0e, 0x0c, 0x18, 0x1a, 0x1e, 0x1c, 0x0b, 0x09, 0x0d, 0x0f, 0x1b, 0x19,
            0x1d, 0x1f
        ],
    );
}

#[test]
fn absent_rails_keep_one_and_keys_differ_in_the_tail() {
    let rows = RowSystem::new(&Layout::new(20, 20, 0).unwrap());
    let table = rows.f2_tail().unwrap();
    let row = BytecodeRow {
        variant: Some(Variant::NOOP),
        ..BytecodeRow::default()
    };
    for keys_differ in [false, true] {
        let sources = Sources::from_parts(
            &row,
            &BaseWords::default(),
            SourceParts {
                inc: 0,
                ram_index: 0,
                pos: 0,
                keys_differ,
                should_branch: false,
                jalr_low_bit: false,
            },
        );
        let words = F2Words::compute(&row, &sources, 0).unwrap();
        let lanes = rows.f2_lanes(&words);
        let tail = table.value(&words, keys_differ);
        assert_eq!(lanes, [[0; 3]; 2]);
        assert_eq!(
            tail,
            8 | (3 * u8::from(keys_differ)),
            "Rails::None still has row 129's ONE"
        );
    }
}
