use super::super::shape::{SelectorFactor, SlotVariable};
use super::{BitEntry, NibbleBuckets, RouterShape, ShapeLayout, WordSlot, F128, ZERO};
use crate::packed::buckets::ByteBuckets;
use crate::source::CycleSource;

const INLINE_WORDS: usize = 16;

#[derive(Debug, Clone, Copy, Default)]
struct WordReader {
    column: usize,
    position: usize,
}

#[derive(Debug, Clone)]
struct Readers {
    inline: [WordReader; INLINE_WORDS],
    len: usize,
    overflow: Vec<WordReader>,
}
impl Readers {
    fn new(readers: Vec<WordReader>) -> Self {
        if readers.len() <= INLINE_WORDS {
            let mut inline = [WordReader::default(); INLINE_WORDS];
            inline[..readers.len()].copy_from_slice(&readers);
            Self {
                inline,
                len: readers.len(),
                overflow: Vec::new(),
            }
        } else {
            Self {
                inline: [WordReader::default(); INLINE_WORDS],
                len: 0,
                overflow: readers,
            }
        }
    }
    #[inline]
    fn as_slice(&self) -> &[WordReader] {
        if self.overflow.is_empty() {
            &self.inline[..self.len]
        } else {
            &self.overflow
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Selector {
    word_base: usize,
    row_base: usize,
    metadata: usize,
    bytes: bool,
}
#[derive(Debug, Clone, Copy)]
struct Digit {
    column: usize,
    offset: usize,
    bound: usize,
}
#[derive(Debug, Clone, Copy)]
struct Flags {
    columns: [usize; NibbleBuckets::BITS_PER_POSITION],
    len: usize,
    offset: usize,
}
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
    factor_indices: [(usize, usize); 3],
    factor_count: usize,
    selectors: Vec<Selector>,
    trace_words: Readers,
    bytecode_words: Readers,
    digits: Vec<Digit>,
    flags: Vec<Flags>,
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
        let selectors = (0..layout.selectors)
            .map(|h| Selector {
                word_base: layout.bases[h],
                row_base: layout.row_bases[h],
                metadata: layout.metadata + h * layout.meta_len,
                bytes: layout.byte_flags[h],
            })
            .collect();
        let mut trace_words = Vec::new();
        let mut bytecode_words = Vec::new();
        for (position, (_, word)) in layout.words.iter().enumerate() {
            match word {
                WordSlot::Trace(column) => trace_words.push(WordReader {
                    column: *column,
                    position,
                }),
                WordSlot::Bytecode(column) => bytecode_words.push(WordReader {
                    column: *column,
                    position,
                }),
                WordSlot::Bits(_) | WordSlot::Zero => {}
            }
        }
        let mut offset = 0;
        let digits: Vec<_> = layout
            .digits
            .iter()
            .map(|&(column, bound)| {
                let digit = Digit {
                    column,
                    offset,
                    bound,
                };
                offset += bound;
                digit
            })
            .collect();
        let flags: Vec<_> = layout
            .flags
            .chunks(NibbleBuckets::BITS_PER_POSITION)
            .map(|columns| {
                let mut group = Flags {
                    columns: [0; NibbleBuckets::BITS_PER_POSITION],
                    len: columns.len(),
                    offset,
                };
                group.columns[..columns.len()].copy_from_slice(columns);
                offset += NibbleBuckets::ENTRIES_PER_POSITION;
                group
            })
            .collect();
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
                                if let Some(digit) = digits.iter().find(|d| d.column == column) {
                                    ReadBit::Value {
                                        offset: digit.offset,
                                        value,
                                    }
                                } else {
                                    let index = layout
                                        .flags
                                        .iter()
                                        .position(|&c| c == column)
                                        .expect("checked flag storage");
                                    ReadBit::Bit {
                                        offset: layout.digits.iter().map(|(_, n)| n).sum::<usize>()
                                            + index / NibbleBuckets::BITS_PER_POSITION
                                                * NibbleBuckets::ENTRIES_PER_POSITION,
                                        bound: NibbleBuckets::ENTRIES_PER_POSITION,
                                        bit: index % NibbleBuckets::BITS_PER_POSITION,
                                    }
                                }
                            }
                            BitEntry::DigitBit { column, bit } => {
                                let digit = digits
                                    .iter()
                                    .find(|d| d.column == column)
                                    .expect("checked digit storage");
                                ReadBit::Bit {
                                    offset: digit.offset,
                                    bound: digit.bound,
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
            factor_indices,
            factor_count: shape.factors().len(),
            selectors,
            trace_words: Readers::new(trace_words),
            bytecode_words: Readers::new(bytecode_words),
            digits,
            flags,
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
    #[inline]
    fn selector<S: CycleSource, const N: usize, const ROW: bool>(
        &self,
        source: &S,
        index: usize,
    ) -> Option<usize> {
        let mut h = 0;
        let mut present = true;
        for &(column, shift) in &self.factor_indices[..N] {
            let digit = if ROW {
                source.row_digit(column, index)
            } else {
                source.digit(column, index)
            };
            present &= digit.is_some();
            h |= digit.unwrap_or(0) << shift;
        }
        present.then_some(h)
    }
    pub(super) fn cycle<S: CycleSource>(
        &self,
        source: &S,
        index: usize,
        e: F128,
        buckets: &mut ChunkBuckets<'_>,
        totals: Option<usize>,
    ) -> bool {
        match self.factor_count {
            1 => self.cycle_n::<S, 1>(source, index, e, buckets, totals),
            2 => self.cycle_n::<S, 2>(source, index, e, buckets, totals),
            _ => self.cycle_n::<S, 3>(source, index, e, buckets, totals),
        }
    }
    #[inline]
    fn cycle_n<S: CycleSource, const N: usize>(
        &self,
        source: &S,
        index: usize,
        e: F128,
        buckets: &mut ChunkBuckets<'_>,
        totals: Option<usize>,
    ) -> bool {
        let Some(h) = self.selector::<S, N, false>(source, index) else {
            return false;
        };
        let entry = self.selectors[h];
        if entry.bytes {
            self.words::<S, true>(source, index, entry.word_base, e, buckets);
        } else {
            self.words::<S, false>(source, index, entry.word_base, e, buckets);
        }
        for digit in &self.digits {
            if let Some(value) = source.digit(digit.column, index) {
                buckets.xor(
                    entry.metadata + digit.offset + value.min(digit.bound - 1),
                    e,
                );
            }
        }
        for flags in &self.flags {
            let mut value = 0;
            for (bit, &column) in flags.columns[..flags.len].iter().enumerate() {
                value |= usize::from(source.digit(column, index).is_some()) << bit;
            }
            buckets.xor(entry.metadata + flags.offset + value, e);
        }
        if let Some(base) = totals {
            buckets.xor(base + h, e);
        }
        true
    }
    #[inline]
    fn words<S: CycleSource, const BYTES: bool>(
        &self,
        source: &S,
        cycle: usize,
        base: usize,
        e: F128,
        buckets: &mut ChunkBuckets<'_>,
    ) {
        let size = super::word_entries(BYTES);
        for reader in self.trace_words.as_slice() {
            buckets.word::<BYTES>(
                base + reader.position * size,
                source.trace_word(reader.column, cycle),
                e,
            );
        }
        if !self.bytecode_words.as_slice().is_empty() {
            let row = source.bytecode_index(cycle);
            for reader in self.bytecode_words.as_slice() {
                buckets.word::<BYTES>(
                    base + reader.position * size,
                    source.bytecode_word(reader.column, row),
                    e,
                );
            }
        }
    }
    pub(super) fn cycles<S: CycleSource>(
        &self,
        source: &S,
        start: usize,
        weights: &[F128],
        buckets: &mut ChunkBuckets<'_>,
        totals: Option<usize>,
        fallbacks: &[(usize, usize)],
    ) {
        match self.factor_count {
            1 => self.cycles_n::<S, 1>(source, start, weights, buckets, totals, fallbacks),
            2 => self.cycles_n::<S, 2>(source, start, weights, buckets, totals, fallbacks),
            _ => self.cycles_n::<S, 3>(source, start, weights, buckets, totals, fallbacks),
        }
    }
    fn cycles_n<S: CycleSource, const N: usize>(
        &self,
        source: &S,
        start: usize,
        weights: &[F128],
        buckets: &mut ChunkBuckets<'_>,
        totals: Option<usize>,
        fallbacks: &[(usize, usize)],
    ) {
        for (offset, &e) in weights.iter().enumerate() {
            let cycle = start + offset;
            if !self.cycle_n::<S, N>(source, cycle, e, buckets, totals) {
                for &(column, base) in fallbacks {
                    if let Some(value) = source.digit(column, cycle) {
                        buckets.xor(base + value, e);
                    }
                }
            }
        }
    }
    pub(super) fn rows<S: CycleSource>(
        &self,
        source: &S,
        start: usize,
        weights: &[F128],
        words: &[(usize, usize)],
        buckets: &mut ChunkBuckets<'_>,
    ) {
        match self.factor_count {
            1 => self.rows_n::<S, 1>(source, start, weights, words, buckets),
            2 => self.rows_n::<S, 2>(source, start, weights, words, buckets),
            _ => self.rows_n::<S, 3>(source, start, weights, words, buckets),
        }
    }
    fn rows_n<S: CycleSource, const N: usize>(
        &self,
        source: &S,
        start: usize,
        weights: &[F128],
        words: &[(usize, usize)],
        buckets: &mut ChunkBuckets<'_>,
    ) {
        for (offset, &e) in weights.iter().enumerate() {
            if e == ZERO {
                continue;
            }
            let row = start + offset;
            let Some(h) = self.selector::<S, N, true>(source, row) else {
                continue;
            };
            let entry = self.selectors[h];
            if entry.bytes {
                for (position, &(_, word)) in words.iter().enumerate() {
                    buckets.word::<true>(
                        entry.row_base + position * ByteBuckets::ELEMENTS_PER_WORD,
                        source.bytecode_word(word, row),
                        e,
                    );
                }
            } else {
                for (position, &(_, word)) in words.iter().enumerate() {
                    buckets.word::<false>(
                        entry.row_base + position * NibbleBuckets::ELEMENTS_PER_WORD,
                        source.bytecode_word(word, row),
                        e,
                    );
                }
            }
        }
    }
}

pub(super) struct ChunkBuckets<'a> {
    positions: NibbleBuckets<'a>,
    tail: &'a mut [F128],
}
impl<'a> ChunkBuckets<'a> {
    #[expect(
        clippy::expect_used,
        reason = "the prefix is cut at a whole position boundary"
    )]
    pub(super) fn new(storage: &'a mut [F128]) -> Self {
        let whole = storage.len() / NibbleBuckets::ENTRIES_PER_POSITION
            * NibbleBuckets::ENTRIES_PER_POSITION;
        let (positions, tail) = storage.split_at_mut(whole);
        let positions = NibbleBuckets::new(positions).expect("whole positions");
        Self { positions, tail }
    }
    #[inline]
    pub(super) fn xor(&mut self, index: usize, e: F128) {
        let position = index / NibbleBuckets::ENTRIES_PER_POSITION;
        if position < self.positions.positions_mut().len() {
            self.positions.positions_mut()[position]
                [index & (NibbleBuckets::ENTRIES_PER_POSITION - 1)] += e;
        } else {
            let last = self.tail.len() - 1;
            self.tail[(index
                - self.positions.positions_mut().len() * NibbleBuckets::ENTRIES_PER_POSITION)
                .min(last)] += e;
        }
    }
    #[inline]
    #[expect(
        clippy::expect_used,
        reason = "word slices have the fixed checked layout dimensions"
    )]
    fn word<const BYTES: bool>(&mut self, base: usize, mut word: u64, e: F128) {
        if base & (NibbleBuckets::ENTRIES_PER_POSITION - 1) != 0 {
            let bits = if BYTES {
                ByteBuckets::BITS_PER_POSITION
            } else {
                NibbleBuckets::BITS_PER_POSITION
            };
            let width = if BYTES {
                ByteBuckets::ENTRIES_PER_POSITION
            } else {
                NibbleBuckets::ENTRIES_PER_POSITION
            };
            for position in 0..if BYTES {
                ByteBuckets::POSITIONS_PER_WORD
            } else {
                NibbleBuckets::POSITIONS_PER_WORD
            } {
                self.xor(base + position * width + (word as usize & (width - 1)), e);
                word >>= bits;
            }
            return;
        }
        let start = base / NibbleBuckets::ENTRIES_PER_POSITION;
        let positions = self.positions.positions_mut();
        if BYTES {
            let word_positions: &mut [[F128; NibbleBuckets::ENTRIES_PER_POSITION];
                     ByteBuckets::ELEMENTS_PER_WORD
                         / NibbleBuckets::ENTRIES_PER_POSITION] = (&mut positions[start
                ..start + ByteBuckets::ELEMENTS_PER_WORD / NibbleBuckets::ENTRIES_PER_POSITION])
                .try_into()
                .expect("one byte word");
            for position in word_positions.as_chunks_mut::<{ ByteBuckets::ENTRIES_PER_POSITION / NibbleBuckets::ENTRIES_PER_POSITION }>().0 {
                let value = (word & (ByteBuckets::ENTRIES_PER_POSITION as u64 - 1)) as usize;
                position[value >> NibbleBuckets::BITS_PER_POSITION][value & (NibbleBuckets::ENTRIES_PER_POSITION - 1)] += e;
                word >>= ByteBuckets::BITS_PER_POSITION;
            }
        } else {
            let word_positions: &mut [[F128; NibbleBuckets::ENTRIES_PER_POSITION];
                     NibbleBuckets::POSITIONS_PER_WORD] = (&mut positions
                [start..start + NibbleBuckets::POSITIONS_PER_WORD])
                .try_into()
                .expect("one nibble word");
            for position in word_positions {
                position[(word & (NibbleBuckets::ENTRIES_PER_POSITION as u64 - 1)) as usize] += e;
                word >>= NibbleBuckets::BITS_PER_POSITION;
            }
        }
    }
}
