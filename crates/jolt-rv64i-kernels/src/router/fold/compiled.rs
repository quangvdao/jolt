use super::super::shape::{SelectorFactor, SlotVariable};
use super::{BitEntry, NibbleBuckets, RouterShape, ShapeLayout, WordSlot};
use crate::source::CycleSource;

#[derive(Debug, Clone)]
pub(super) enum BankStorage {
    Cycle(usize),
    Row(usize),
    Bits(Vec<ReadBit>),
    Zero,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum ReadBit {
    One,
    Value {
        offset: usize,
        value: usize,
    },
    Bit {
        offset: usize,
        bound: usize,
        bit: usize,
    },
    Zero,
}

#[derive(Debug, Clone)]
pub(super) struct CompiledShape {
    bank: Vec<WordSlot>,
    factors: Vec<SelectorFactor>,
    slots: Vec<(usize, SlotVariable)>,
    columns: Vec<(usize, usize, bool)>,
    pub(super) bank_storage: Vec<BankStorage>,
    pub(super) destinations: Vec<usize>,
}

impl CompiledShape {
    #[expect(
        clippy::expect_used,
        reason = "checked bank slots and entries have canonical storage"
    )]
    pub(super) fn new<S: CycleSource>(
        shape: &RouterShape,
        layout: &ShapeLayout,
        source: &S,
    ) -> Self {
        let mut factor_indices = [(0, 0); 3];
        let mut shift = 0;
        for (index, factor) in shape.factors().iter().enumerate() {
            factor_indices[index] = (factor.column, shift);
            shift += factor.slots.len();
        }
        let bank_storage = shape
            .bank()
            .iter()
            .enumerate()
            .map(|(slot, word)| match word {
                WordSlot::Trace(_) | WordSlot::Bytecode(_) => {
                    if let Some(index) = layout.words.iter().position(|(s, _)| *s == slot) {
                        BankStorage::Cycle(index)
                    } else {
                        BankStorage::Row(
                            layout
                                .row_words
                                .iter()
                                .position(|(s, _)| *s == slot)
                                .expect("checked word storage"),
                        )
                    }
                }
                WordSlot::Zero => BankStorage::Zero,
                WordSlot::Bits(entries) => BankStorage::Bits(
                    entries
                        .iter()
                        .map(|entry| match *entry {
                            BitEntry::One => ReadBit::One,
                            BitEntry::Zero => ReadBit::Zero,
                            BitEntry::Indicator { column, value } => {
                                if let Some((base, _)) = layout.digit_base(0, column) {
                                    ReadBit::Value {
                                        offset: base - layout.metadata,
                                        value,
                                    }
                                } else {
                                    let (base, bit) =
                                        layout.flag_base(0, column).expect("checked flag storage");
                                    ReadBit::Bit {
                                        offset: base - layout.metadata,
                                        bound: NibbleBuckets::ENTRIES_PER_POSITION,
                                        bit,
                                    }
                                }
                            }
                            BitEntry::DigitBit { column, bit } => {
                                let (base, bound) =
                                    layout.digit_base(0, column).expect("checked digit storage");
                                ReadBit::Bit {
                                    offset: base - layout.metadata,
                                    bound,
                                    bit,
                                }
                            }
                        })
                        .collect(),
                ),
            })
            .collect();
        let destinations = (0..layout.selectors)
            .flat_map(|h| {
                (0..shape.bank().len()).map(move |word| {
                    shape.slot_map().iter().enumerate().skip(6).fold(
                        0,
                        |destination, (position, (_, variable))| {
                            let value = match *variable {
                                SlotVariable::Bit(_) => 0,
                                SlotVariable::Word(bit) => word >> bit,
                                SlotVariable::Selector { factor, bit } => {
                                    h >> (factor_indices[factor].1 + bit)
                                }
                            };
                            destination | ((value & 1) << position)
                        },
                    )
                })
            })
            .collect();
        let mut columns = Vec::new();
        for column in shape
            .factors()
            .iter()
            .map(|factor| factor.column)
            .chain(layout.digits.iter().map(|&(column, _)| column))
            .chain(layout.flags.iter().copied())
        {
            if !columns.iter().any(|&(c, _, _)| c == column) {
                columns.push((column, source.bits(column), source.by_row(column)));
            }
        }
        Self {
            bank: shape.bank().to_vec(),
            factors: shape.factors().to_vec(),
            slots: shape.slot_map().to_vec(),
            columns,
            bank_storage,
            destinations,
        }
    }

    pub(super) fn matches<S: CycleSource>(&self, shape: &RouterShape, source: &S) -> bool {
        self.bank == shape.bank()
            && self.factors == shape.factors()
            && self.slots == shape.slot_map()
            && self.columns.iter().all(|&(column, bits, by_row)| {
                source.bits(column) == bits && source.by_row(column) == by_row
            })
    }
}
