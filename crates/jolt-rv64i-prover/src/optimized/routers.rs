//! Routers adapters for the packed RV64I kernels.

use crate::optimized::source::WitnessColumns;
use jolt_rv64i_arith::Layout;
use jolt_rv64i_kernels::router::shape::{
    BitEntry, RouteEntry, RouterError, RouterShape, RouterShapeRequest, SelectorFactor, WordSlot,
};
use jolt_rv64i_verifier::public::routes::{
    bank, selector_slots, source_slots, BankWord, Factor, RouteTensors, ROUTERS,
};

fn word_slot(word: BankWord) -> WordSlot {
    match word {
        BankWord::Rs1Value => WordSlot::Trace(WitnessColumns::rs1_value()),
        BankWord::Rs2Value => WordSlot::Trace(WitnessColumns::rs2_value()),
        BankWord::RdPreValue => WordSlot::Trace(WitnessColumns::rd_pre_value()),
        BankWord::RamReadValue => WordSlot::Trace(WitnessColumns::ram_read_value()),
        BankWord::NextPC => WordSlot::Trace(WitnessColumns::next_pc()),
        BankWord::Inc => WordSlot::Trace(WitnessColumns::inc_word()),
        BankWord::Imm => WordSlot::Bytecode(WitnessColumns::imm()),
        BankWord::FallThroughPC => WordSlot::Bytecode(WitnessColumns::fall_through_pc()),
        BankWord::PCPlusImm => WordSlot::Bytecode(WitnessColumns::pc_plus_imm()),
        BankWord::PC => WordSlot::Bytecode(WitnessColumns::pc()),
        BankWord::One => WordSlot::Bits(vec![BitEntry::One]),
    }
}

fn factor_column(columns: &WitnessColumns, factor: Factor) -> Result<usize, RouterError> {
    Ok(match factor {
        Factor::Variant => columns.variant(),
        Factor::Pos(digit) => columns.pos(usize::from(digit)).ok_or(RouterError::Column {
            column: usize::from(digit),
            columns: 2,
        })?,
        Factor::ShiftKind => columns.shift_kind(),
        Factor::AccessKind => columns.access_kind(),
        Factor::KeyKind => columns.key_kind(),
        Factor::Branch => columns.branch(),
        Factor::ShouldBranch => columns.should_branch(),
    })
}

/// Converts the verifier-owned banks and variable maps into checked shapes in
/// `ROUTERS` order. With tensors, each support is copied once; without them no
/// route entries are walked. Passes additionally check columns against the source.
pub fn router_shapes(
    columns: &WitnessColumns,
    layout: &Layout,
    tensors: Option<&RouteTensors>,
) -> Result<Vec<RouterShape>, RouterError> {
    ROUTERS
        .into_iter()
        .map(|router| {
            let description = bank(router, layout);
            let mut words: Vec<_> = description.words.iter().copied().map(word_slot).collect();
            if let Some(committed) = description.committed {
                let mut entries = Vec::with_capacity(committed.len() + 1);
                for y in committed {
                    let (column, value) = columns.committed(y).ok_or(RouterError::Column {
                        column: y,
                        columns: layout.used_columns(),
                    })?;
                    entries.push(BitEntry::Indicator { column, value });
                }
                entries.push(BitEntry::One);
                words.extend(
                    entries
                        .chunks(64)
                        .map(|entries| WordSlot::Bits(entries.to_vec())),
                );
            }
            words.resize(words.len().next_power_of_two(), WordSlot::Zero);
            let selectors = selector_slots(router);
            let pos_width = |digit: u8| {
                layout
                    .pos_ra()
                    .get(usize::from(digit))
                    .map(|chunk| usize::from(chunk.bits()))
                    .ok_or(RouterError::Column {
                        column: usize::from(digit),
                        columns: layout.pos_ra().len(),
                    })
            };
            let mut pos_bits = 0;
            for factor in description.factors {
                if let Factor::Pos(digit) = *factor {
                    pos_bits += pos_width(digit)?;
                }
            }
            let kind_bits =
                selectors
                    .len()
                    .checked_sub(pos_bits)
                    .ok_or(RouterError::FactorWidth {
                        column: 0,
                        expected: pos_bits,
                        actual: selectors.len(),
                    })?;
            let mut start = 0;
            let mut factors = Vec::with_capacity(description.factors.len());
            for &factor in description.factors {
                let width = match factor {
                    Factor::Pos(digit) => pos_width(digit)?,
                    Factor::Branch | Factor::ShouldBranch => 0,
                    Factor::Variant | Factor::ShiftKind | Factor::AccessKind | Factor::KeyKind => {
                        kind_bits
                    }
                };
                let end = start + width;
                let slots = selectors.get(start..end).ok_or(RouterError::FactorWidth {
                    column: factor_column(columns, factor)?,
                    expected: end,
                    actual: selectors.len(),
                })?;
                factors.push(SelectorFactor {
                    column: factor_column(columns, factor)?,
                    slots: slots.to_vec(),
                });
                start = end;
            }
            let route = tensors.map_or_else(Vec::new, |tensors| {
                tensors
                    .entries(router)
                    .iter()
                    .map(|entry| RouteEntry {
                        output: entry.column,
                        source: entry.source,
                        selector: entry.selector,
                    })
                    .collect()
            });
            RouterShape::new(RouterShapeRequest {
                slots: 17,
                bank: words,
                factors,
                word_slots: source_slots(router).get(6..).unwrap_or(&[]).to_vec(),
                log_outputs: 10,
                route,
            })
        })
        .collect()
}
