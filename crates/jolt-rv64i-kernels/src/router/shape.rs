//! Checked router banks and low-first variable maps shared by every router phase.

use crate::source::{CycleSource, ValidatedTrace};
use std::ops::Range;
use thiserror::Error;

/// One source bit; absent digits contribute zero to both digit entry kinds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitEntry {
    Indicator { column: usize, value: usize },
    DigitBit { column: usize, bit: usize },
    One,
    Zero,
}

/// A 64-bit source word; unspecified entries of `Bits` are zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WordSlot {
    Trace(usize),
    Bytecode(usize),
    Bits(Vec<BitEntry>),
    Zero,
}

/// A one-hot digit factor, with its low-first index bits in the stated slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectorFactor {
    pub column: usize,
    pub slots: Vec<usize>,
}

/// The variable at one occupied slot. Selector bits are low-first within a factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotVariable {
    Bit(usize),
    Word(usize),
    Selector { factor: usize, bit: usize },
}

/// Invalid router geometry, source references or pass storage.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RouterError {
    #[error("bank length {len} is not a nonzero power of two")]
    BankLength { len: usize },
    #[error("word slot {slot} has {entries} bit entries, exceeding 64")]
    BitsEntries { slot: usize, entries: usize },
    #[error("factor count {count} is outside 1..=3")]
    Factors { count: usize },
    #[error("bank needs {expected} word variables, got {actual}")]
    WordVariables { expected: usize, actual: usize },
    #[error("slot {slot} is repeated")]
    SlotRepeated { slot: usize },
    #[error("slot {slot} is outside {slots} slots")]
    SlotRange { slot: usize, slots: usize },
    #[error("bit variable {bit} occupies slot {slot}, expected {bit}")]
    BitSlot { bit: usize, slot: usize },
    #[error("route {triple:?} has {axis} index outside {bound}")]
    Route {
        triple: (usize, usize, usize),
        axis: &'static str,
        bound: usize,
    },
    #[error("{bank} word {index} is outside {words} words")]
    WordIndex {
        bank: &'static str,
        index: usize,
        words: usize,
    },
    #[error("column {column} is outside {columns} columns")]
    Column { column: usize, columns: usize },
    #[error("factor column {column} needs {expected} slots, got {actual}")]
    FactorWidth {
        column: usize,
        expected: usize,
        actual: usize,
    },
    #[error("{kind} entry {value} is outside bound {bound} on column {column}")]
    Entry {
        column: usize,
        kind: &'static str,
        value: usize,
        bound: usize,
    },
    #[error("point has {actual} coordinates, expected {expected}")]
    PointLength { expected: usize, actual: usize },
    #[error("{table} table has {actual} elements, expected {expected}")]
    TableLength {
        table: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("dimension {variables} cannot represent a field table")]
    Dimension { variables: usize },
    #[error("layout selector {selector} is repeated or outside {bound} values for shape {shape}")]
    Layout {
        shape: usize,
        selector: usize,
        bound: usize,
    },
}

/// Immutable bank, factors and routing tensor geometry.
/// `new` checks every source-independent bound. Source references are checked
/// by the passes. No additional honest-input condition is required of the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterShape {
    slots: usize,
    bank: Vec<WordSlot>,
    factors: Vec<SelectorFactor>,
    word_slots: Vec<usize>,
    slot_map: Vec<(usize, SlotVariable)>,
    idle: Vec<usize>,
    log_outputs: usize,
    route: Vec<(usize, usize, usize)>,
}

pub(crate) fn table_len(variables: usize) -> Result<usize, RouterError> {
    if variables >= usize::BITS as usize - 5 {
        return Err(RouterError::Dimension { variables });
    }
    Ok(1 << variables)
}

impl RouterShape {
    /// Validates the complete bank, distinct slots, bit slots 0–5, and every
    /// route triple. Factor slots and word slots list their index bits low first.
    /// `route` is a set: repeated triples are removed.
    pub fn new(
        slots: usize,
        bank: Vec<WordSlot>,
        factors: Vec<SelectorFactor>,
        bit_slots: [usize; 6],
        word_slots: Vec<usize>,
        log_outputs: usize,
        mut route: Vec<(usize, usize, usize)>,
    ) -> Result<Self, RouterError> {
        if !bank.len().is_power_of_two() {
            return Err(RouterError::BankLength { len: bank.len() });
        }
        for (slot, word) in bank.iter().enumerate() {
            if let WordSlot::Bits(entries) = word {
                if entries.len() > 64 {
                    return Err(RouterError::BitsEntries {
                        slot,
                        entries: entries.len(),
                    });
                }
            }
        }
        if !(1..=3).contains(&factors.len()) {
            return Err(RouterError::Factors {
                count: factors.len(),
            });
        }
        let expected = bank.len().ilog2() as usize;
        if word_slots.len() != expected {
            return Err(RouterError::WordVariables {
                expected,
                actual: word_slots.len(),
            });
        }
        let _ = table_len(slots)?;
        let _ = table_len(log_outputs)?;
        let mut slot_map = Vec::new();
        for (bit, slot) in bit_slots.into_iter().enumerate() {
            if slot != bit {
                return Err(RouterError::BitSlot { bit, slot });
            }
            slot_map.push((slot, SlotVariable::Bit(bit)));
        }
        slot_map.extend(
            word_slots
                .iter()
                .enumerate()
                .map(|(bit, &slot)| (slot, SlotVariable::Word(bit))),
        );
        for (factor, value) in factors.iter().enumerate() {
            slot_map.extend(
                value
                    .slots
                    .iter()
                    .enumerate()
                    .map(|(bit, &slot)| (slot, SlotVariable::Selector { factor, bit })),
            );
        }
        slot_map.sort_unstable_by_key(|&(slot, _)| slot);
        for (index, &(slot, _)) in slot_map.iter().enumerate() {
            if slot >= slots {
                return Err(RouterError::SlotRange { slot, slots });
            }
            if index != 0 && slot == slot_map[index - 1].0 {
                return Err(RouterError::SlotRepeated { slot });
            }
        }
        let selectors = table_len(factors.iter().map(|f| f.slots.len()).sum())?;
        let sources = table_len(6 + expected)?;
        for &triple in &route {
            for (axis, value, bound) in [
                ("output", triple.0, 1 << log_outputs),
                ("source", triple.1, sources),
                ("selector", triple.2, selectors),
            ] {
                if value >= bound {
                    return Err(RouterError::Route {
                        triple,
                        axis,
                        bound,
                    });
                }
            }
        }
        route.sort_unstable();
        route.dedup();
        let idle = (0..slots)
            .filter(|slot| !slot_map.iter().any(|&(used, _)| used == *slot))
            .collect();
        Ok(Self {
            slots,
            bank,
            factors,
            word_slots,
            slot_map,
            idle,
            log_outputs,
            route,
        })
    }

    /// Number of shared short-sumcheck slots, including idle slots.
    pub fn slots(&self) -> usize {
        self.slots
    }
    /// Source words in bank index order.
    pub fn bank(&self) -> &[WordSlot] {
        &self.bank
    }
    /// Factors in mixed-radix order, low factor first.
    pub fn factors(&self) -> &[SelectorFactor] {
        &self.factors
    }
    /// Word index bits, low first.
    pub fn word_slots(&self) -> &[usize] {
        &self.word_slots
    }
    /// Occupied slots in increasing order, the Fold table's variable order.
    pub fn slot_map(&self) -> &[(usize, SlotVariable)] {
        &self.slot_map
    }
    /// Unoccupied short-sumcheck slots in increasing order.
    pub fn idle_slots(&self) -> &[usize] {
        &self.idle
    }
    /// Base-two logarithm of the output domain.
    pub fn log_outputs(&self) -> usize {
        self.log_outputs
    }
    /// The checked, sorted routing support; fold construction never reads it.
    pub fn route(&self) -> &[(usize, usize, usize)] {
        &self.route
    }
    /// Number of mixed-radix selector values, including unreachable values.
    pub fn selectors(&self) -> usize {
        1 << self.factors.iter().map(|f| f.slots.len()).sum::<usize>()
    }
    /// Complete Fold table length in increasing slot order.
    pub fn fold_len(&self) -> usize {
        1 << self.slot_map.len()
    }

    pub(crate) fn check_source<S: CycleSource>(&self, source: &S) -> Result<(), RouterError> {
        let check_column = |column| {
            if column >= source.digit_columns() {
                Err(RouterError::Column {
                    column,
                    columns: source.digit_columns(),
                })
            } else {
                Ok(())
            }
        };
        for f in &self.factors {
            check_column(f.column)?;
            if f.slots.len() != source.bits(f.column) {
                return Err(RouterError::FactorWidth {
                    column: f.column,
                    expected: source.bits(f.column),
                    actual: f.slots.len(),
                });
            }
        }
        for word in &self.bank {
            match word {
                WordSlot::Trace(index) | WordSlot::Bytecode(index) => {
                    let (bank, words) = if matches!(word, WordSlot::Trace(_)) {
                        ("trace", source.trace_words())
                    } else {
                        ("bytecode", source.bytecode_words())
                    };
                    if *index >= words {
                        return Err(RouterError::WordIndex {
                            bank,
                            index: *index,
                            words,
                        });
                    }
                }
                WordSlot::Bits(entries) => {
                    for entry in entries {
                        match *entry {
                            BitEntry::Indicator { column, value } => {
                                check_column(column)?;
                                let bound = 1 << source.bits(column);
                                if value >= bound {
                                    return Err(RouterError::Entry {
                                        column,
                                        kind: "indicator",
                                        value,
                                        bound,
                                    });
                                }
                            }
                            BitEntry::DigitBit { column, bit } => {
                                check_column(column)?;
                                let bound = source.bits(column);
                                if bit >= bound {
                                    return Err(RouterError::Entry {
                                        column,
                                        kind: "digit bit",
                                        value: bit,
                                        bound,
                                    });
                                }
                            }
                            BitEntry::One | BitEntry::Zero => {}
                        }
                    }
                }
                WordSlot::Zero => {}
            }
        }
        Ok(())
    }

    pub(crate) fn selector<S: CycleSource>(
        &self,
        source: &S,
        index: usize,
        row: bool,
    ) -> Option<usize> {
        let mut h = 0;
        let mut shift = 0;
        for factor in &self.factors {
            let digit = if row {
                source.row_digit(factor.column, index)
            } else {
                source.digit(factor.column, index)
            }?;
            h |= digit << shift;
            shift += factor.slots.len();
        }
        Some(h)
    }
}

/// Counts cycles that spell each selector; absent factors do not contribute.
/// The source is validated, and all shape source references are checked here.
/// Select frequent values by descending count, breaking ties by increasing value.
pub fn selector_counts<S: CycleSource>(
    source: &ValidatedTrace<S>,
    shape: &RouterShape,
) -> Result<Vec<usize>, RouterError> {
    let source = source.source();
    shape.check_source(source.as_ref())?;
    let mut counts = vec![0; shape.selectors()];
    for cycle in 0..source.cycles() {
        if let Some(h) = shape.selector(source.as_ref(), cycle, false) {
            counts[h] += 1;
        }
    }
    Ok(counts)
}

/// The five protocol banks and slot maps over `SyntheticTrace` columns, in
/// Variant, Shift, Memory, Compare, Branch order. Routing supports start empty;
/// callers construct their public tensors separately. No relation type is used.
#[cfg(feature = "test-utils")]
pub fn synthetic_router_shapes() -> Result<Vec<RouterShape>, RouterError> {
    let mut entries = Vec::new();
    for column in 5..12 {
        for value in 1..if column < 10 { 16 } else { 8 } {
            entries.push(BitEntry::Indicator { column, value });
        }
    }
    for column in 18..21 {
        entries.push(BitEntry::Indicator { column, value: 0 });
    }
    entries.push(BitEntry::One);
    let mut variant = vec![
        WordSlot::Trace(0),
        WordSlot::Trace(1),
        WordSlot::Trace(2),
        WordSlot::Bytecode(0),
        WordSlot::Bytecode(1),
        WordSlot::Bytecode(2),
        WordSlot::Bytecode(3),
        WordSlot::Trace(4),
        WordSlot::Trace(5),
    ];
    variant.extend(
        entries
            .chunks(64)
            .map(|entries| WordSlot::Bits(entries.to_vec())),
    );
    variant.resize(16, WordSlot::Zero);
    let factor = |column, slots: Range<usize>| SelectorFactor {
        column,
        slots: slots.collect(),
    };
    Ok(vec![
        RouterShape::new(
            17,
            variant,
            vec![factor(12, 11..17)],
            [0, 1, 2, 3, 4, 5],
            vec![6, 7, 8, 9],
            10,
            vec![],
        )?,
        RouterShape::new(
            17,
            vec![WordSlot::Trace(0)],
            vec![factor(10, 6..9), factor(11, 9..12), factor(13, 12..15)],
            [0, 1, 2, 3, 4, 5],
            vec![],
            10,
            vec![],
        )?,
        RouterShape::new(
            17,
            vec![WordSlot::Trace(3), WordSlot::Trace(1)],
            vec![factor(10, 6..9), factor(14, 13..17)],
            [0, 1, 2, 3, 4, 5],
            vec![12],
            10,
            vec![],
        )?,
        RouterShape::new(
            17,
            vec![
                WordSlot::Trace(0),
                WordSlot::Trace(1),
                WordSlot::Bytecode(0),
                WordSlot::Bits(vec![BitEntry::One]),
            ],
            vec![factor(10, 6..9), factor(11, 9..12), factor(15, 14..17)],
            [0, 1, 2, 3, 4, 5],
            vec![12, 13],
            10,
            vec![],
        )?,
        RouterShape::new(
            17,
            vec![WordSlot::Bytecode(1), WordSlot::Bytecode(2)],
            vec![factor(16, 6..6), factor(19, 6..6)],
            [0, 1, 2, 3, 4, 5],
            vec![12],
            10,
            vec![],
        )?,
    ])
}
