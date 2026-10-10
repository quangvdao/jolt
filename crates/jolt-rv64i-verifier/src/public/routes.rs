//! Sparse binary route tensors generated from the RV64I decode forms.

use crate::ids::Router;
use crate::points::{self, PointsError};
use jolt_field::JoltField;
use jolt_rv64i_arith::decode::RdWriteSource;
use jolt_rv64i_arith::{
    load_form, shift_form, store_form, AccessKind, Chunk, Layout, Rails, Source, Term, Variant,
    BRANCH_FORM,
};
use std::collections::BTreeSet;
use std::ops::Range;

/// Number of variables in the common router short space.
pub const SHORT_VARIABLES: usize = 17;

pub const ROUTERS: [Router; 5] = [
    Router::Variant,
    Router::Shift,
    Router::Memory,
    Router::Compare,
    Router::Branch,
];

/// A named word in a router's source bank; `One` has only bit zero set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankWord {
    Rs1Value,
    Rs2Value,
    RdPreValue,
    RamReadValue,
    NextPC,
    Inc,
    Imm,
    FallThroughPC,
    PCPlusImm,
    PC,
    One,
}

/// Selector digits in least-significant-factor order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Factor {
    Variant,
    Pos(u8),
    ShiftKind,
    AccessKind,
    KeyKind,
    Branch,
    ShouldBranch,
}

/// Width of a named selector factor, shared by tensor generation and shapes.
pub fn factor_bits(factor: Factor, layout: &Layout) -> Result<usize, PointsError> {
    Ok(match factor {
        Factor::Variant => 6,
        Factor::Pos(index) => layout
            .pos_ra()
            .get(usize::from(index))
            .copied()
            .ok_or(PointsError::Index {
                index: usize::from(index),
                variables: 1,
            })?
            .bits()
            .into(),
        Factor::ShiftKind | Factor::KeyKind => 3,
        Factor::AccessKind => 4,
        Factor::Branch | Factor::ShouldBranch => 0,
    })
}

/// Words precede committed entries, which are followed by one constant.
/// A bank without committed entries represents constants through `words`.
#[derive(Clone, Debug)]
pub struct Bank {
    pub words: &'static [BankWord],
    pub committed: Option<Range<usize>>,
    pub factors: &'static [Factor],
}

/// The shared source and selector geometry used by tensor generation and proving.
pub fn bank(router: Router, layout: &Layout) -> Bank {
    match router {
        Router::Variant => Bank {
            words: &[
                BankWord::Rs1Value,
                BankWord::Rs2Value,
                BankWord::RdPreValue,
                BankWord::Imm,
                BankWord::FallThroughPC,
                BankWord::PCPlusImm,
                BankWord::PC,
                BankWord::NextPC,
                BankWord::Inc,
            ],
            committed: Some(
                layout
                    .ram_ra()
                    .first()
                    .map_or(64, |c| usize::from(c.start()))..layout.used_columns(),
            ),
            factors: &[Factor::Variant],
        },
        Router::Shift => Bank {
            words: &[BankWord::Rs1Value],
            committed: None,
            factors: &[Factor::Pos(0), Factor::Pos(1), Factor::ShiftKind],
        },
        Router::Memory => Bank {
            words: &[BankWord::RamReadValue, BankWord::Rs2Value],
            committed: None,
            factors: &[Factor::Pos(0), Factor::AccessKind],
        },
        Router::Compare => Bank {
            words: &[
                BankWord::Rs1Value,
                BankWord::Rs2Value,
                BankWord::Imm,
                BankWord::One,
            ],
            committed: None,
            factors: &[Factor::Pos(0), Factor::Pos(1), Factor::KeyKind],
        },
        Router::Branch => Bank {
            words: &[BankWord::FallThroughPC, BankWord::PCPlusImm],
            committed: None,
            factors: &[Factor::Branch, Factor::ShouldBranch],
        },
    }
}

pub fn source_slots(router: Router) -> &'static [usize] {
    match router {
        Router::Variant => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        Router::Shift => &[0, 1, 2, 3, 4, 5],
        Router::Memory | Router::Branch => &[0, 1, 2, 3, 4, 5, 12],
        Router::Compare => &[0, 1, 2, 3, 4, 5, 12, 13],
    }
}
pub fn selector_slots(router: Router) -> &'static [usize] {
    match router {
        Router::Variant => &[11, 12, 13, 14, 15, 16],
        Router::Shift => &[6, 7, 8, 9, 10, 11, 12, 13, 14],
        Router::Memory => &[6, 7, 8, 13, 14, 15, 16],
        Router::Compare => &[6, 7, 8, 9, 10, 11, 14, 15, 16],
        Router::Branch => &[],
    }
}
pub fn idle_slots(router: Router) -> &'static [usize] {
    match router {
        Router::Variant => &[10],
        Router::Shift => &[15, 16],
        Router::Memory => &[9, 10, 11],
        Router::Compare => &[],
        Router::Branch => &[6, 7, 8, 9, 10, 11, 13, 14, 15, 16],
    }
}
pub fn restriction<F: Copy>(router: Router, point: &[F]) -> Result<Vec<F>, PointsError> {
    if point.len() != SHORT_VARIABLES {
        return Err(PointsError::Dimension {
            expected: SHORT_VARIABLES,
            actual: point.len(),
        });
    }
    source_slots(router)
        .iter()
        .chain(selector_slots(router))
        .map(|&slot| {
            point.get(slot).copied().ok_or(PointsError::Dimension {
                expected: SHORT_VARIABLES,
                actual: point.len(),
            })
        })
        .collect()
}
/// Recovers the shared short point from the router with no idle coordinates.
pub(crate) fn short_point<F: JoltField>(compare: &[F]) -> Result<Vec<F>, PointsError> {
    if compare.len() != SHORT_VARIABLES {
        return Err(PointsError::Dimension {
            expected: SHORT_VARIABLES,
            actual: compare.len(),
        });
    }
    let mut point = vec![F::zero(); SHORT_VARIABLES];
    for (slot, value) in source_slots(Router::Compare)
        .iter()
        .chain(selector_slots(Router::Compare))
        .zip(compare)
    {
        let target = point.get_mut(*slot).ok_or(PointsError::Dimension {
            expected: SHORT_VARIABLES,
            actual: *slot,
        })?;
        *target = *value;
    }
    Ok(point)
}
pub fn projected_index(router: Router, index: usize) -> (usize, usize) {
    let project = |slots: &[usize]| {
        slots
            .iter()
            .enumerate()
            .fold(0, |value, (i, slot)| value | (((index >> slot) & 1) << i))
    };
    (
        project(source_slots(router)),
        project(selector_slots(router)),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RouteEntry {
    pub column: usize,
    pub source: usize,
    pub selector: usize,
}
#[derive(Clone, Debug)]
pub struct RouteTensors {
    tensors: [Vec<RouteEntry>; 5],
}
impl RouteTensors {
    pub fn new(layout: &Layout) -> Result<Self, PointsError> {
        let mut builder = Builder {
            layout,
            entries: std::array::from_fn(|_| BTreeSet::new()),
        };
        for variant in Variant::ALL {
            let h = builder.selector(Router::Variant, &[(Factor::Variant, variant.index())])?;
            let line = variant.line();
            builder.toggle(Router::Variant, 16, builder.one(Router::Variant)?, h);
            match line.rails {
                Rails::None => {}
                Rails::Adder { left, right, sum } => {
                    for form in [right, sum] {
                        builder.form(Router::Variant, 64, h, variant, form)?;
                    }
                    for form in [left, sum] {
                        builder.form(Router::Variant, 128, h, variant, form)?;
                    }
                    for form in [left, right, sum] {
                        for term in form {
                            for wire in term.wires() {
                                if wire.out < 63 {
                                    builder.wire(
                                        Router::Variant,
                                        192 + usize::from(wire.out),
                                        h,
                                        variant,
                                        wire.source,
                                        wire.bit,
                                    )?;
                                }
                                if wire.out > 0 {
                                    builder.wire(
                                        Router::Variant,
                                        191 + usize::from(wire.out),
                                        h,
                                        variant,
                                        wire.source,
                                        wire.bit,
                                    )?;
                                }
                                if wire.out == 0 {
                                    builder.wire(
                                        Router::Variant,
                                        255,
                                        h,
                                        variant,
                                        wire.source,
                                        wire.bit,
                                    )?;
                                }
                            }
                        }
                    }
                }
                Rails::And { left, right, out } => {
                    builder.form(Router::Variant, 256, h, variant, left)?;
                    builder.form(Router::Variant, 320, h, variant, right)?;
                    builder.form(Router::Variant, 384, h, variant, out)?;
                }
                Rails::Compare { keys, less_than } => {
                    let (left, right) = keys.keys();
                    for form in [left, right] {
                        builder.form(Router::Variant, 448, h, variant, form)?;
                    }
                    for term in less_than {
                        for wire in term.wires().filter(|wire| wire.out == 0) {
                            builder.wire(Router::Variant, 3, h, variant, wire.source, wire.bit)?;
                        }
                    }
                }
            }
            builder.form(Router::Variant, 16, h, variant, line.control)?;
            for bit in 0..64 {
                builder.wire(
                    Router::Variant,
                    576 + usize::from(bit),
                    h,
                    variant,
                    Source::RdWriteValue,
                    bit,
                )?;
                builder.wire(
                    Router::Variant,
                    704 + usize::from(bit),
                    h,
                    variant,
                    Source::NextPC,
                    bit,
                )?;
                if variant.is_store() {
                    builder.wire(
                        Router::Variant,
                        640 + usize::from(bit),
                        h,
                        variant,
                        Source::Inc,
                        bit,
                    )?;
                }
            }
            builder.form(Router::Variant, 576, h, variant, line.rd_expected)?;
            builder.form(Router::Variant, 704, h, variant, line.next_pc_expected)?;
        }
        let shifts: BTreeSet<_> = Variant::ALL
            .into_iter()
            .filter_map(|v| v.shift().map(|s| s.kind as usize))
            .collect();
        for kind in shifts {
            let shift = Variant::ALL
                .into_iter()
                .filter_map(|v| v.shift())
                .find(|s| s.kind as usize == kind)
                .ok_or(PointsError::Index {
                    index: kind,
                    variables: 3,
                })?;
            for pos in 0..64_u8 {
                builder.form(
                    Router::Shift,
                    576,
                    builder.selector(
                        Router::Shift,
                        &[
                            (Factor::Pos(0), usize::from(pos & 7)),
                            (Factor::Pos(1), usize::from(pos >> 3)),
                            (Factor::ShiftKind, kind),
                        ],
                    )?,
                    Variant::NOOP,
                    shift_form(shift.kind, pos)
                        .map_err(|_| PointsError::Index {
                            index: usize::from(pos),
                            variables: 6,
                        })?
                        .terms(),
                )?;
            }
        }
        let kinds: BTreeSet<_> = Variant::ALL
            .into_iter()
            .filter_map(|v| v.access().and_then(|a| a.kind).map(|k| k as usize))
            .collect();
        for index in kinds {
            let kind: AccessKind = Variant::ALL
                .into_iter()
                .filter_map(|v| v.access().and_then(|a| a.kind))
                .find(|k| *k as usize == index)
                .ok_or(PointsError::Index {
                    index,
                    variables: 4,
                })?;
            for pos in 0..8_u8 {
                let h = builder.selector(
                    Router::Memory,
                    &[
                        (Factor::Pos(0), usize::from(pos)),
                        (Factor::AccessKind, index),
                    ],
                )?;
                builder.form(
                    Router::Memory,
                    576,
                    h,
                    Variant::NOOP,
                    load_form(kind, pos)
                        .map_err(|_| PointsError::Index {
                            index: h,
                            variables: 7,
                        })?
                        .terms(),
                )?;
                builder.form(
                    Router::Memory,
                    640,
                    h,
                    Variant::NOOP,
                    store_form(kind, pos)
                        .map_err(|_| PointsError::Index {
                            index: h,
                            variables: 7,
                        })?
                        .terms(),
                )?;
            }
        }
        let keys: BTreeSet<_> = Variant::ALL
            .into_iter()
            .filter_map(|v| v.key_kind().map(|k| k as usize))
            .collect();
        for kind in keys {
            let key = Variant::ALL
                .into_iter()
                .filter_map(|v| v.key_kind())
                .find(|k| *k as usize == kind)
                .ok_or(PointsError::Index {
                    index: kind,
                    variables: 3,
                })?;
            let (left, right) = key.keys();
            for pos in 0..64_u8 {
                let h = builder.selector(
                    Router::Compare,
                    &[
                        (Factor::Pos(0), usize::from(pos & 7)),
                        (Factor::Pos(1), usize::from(pos >> 3)),
                        (Factor::KeyKind, kind),
                    ],
                )?;
                for (form, column) in [(left, 1), (right, 2)] {
                    for term in form {
                        for wire in term.wires() {
                            if wire.out == pos {
                                builder.wire(
                                    Router::Compare,
                                    column,
                                    h,
                                    Variant::NOOP,
                                    wire.source,
                                    wire.bit,
                                )?;
                            }
                            if wire.out > pos {
                                builder.wire(
                                    Router::Compare,
                                    512 + usize::from(wire.out),
                                    h,
                                    Variant::NOOP,
                                    wire.source,
                                    wire.bit,
                                )?;
                            }
                        }
                    }
                }
            }
        }
        builder.form(
            Router::Branch,
            704,
            builder.selector(
                Router::Branch,
                &[(Factor::Branch, 0), (Factor::ShouldBranch, 0)],
            )?,
            Variant::NOOP,
            BRANCH_FORM,
        )?;
        Ok(Self {
            tensors: builder.entries.map(|set| set.into_iter().collect()),
        })
    }
    pub fn entries(&self, router: Router) -> &[RouteEntry] {
        self.tensors.get(router as usize).map_or(&[], Vec::as_slice)
    }
    /// Number of nonzero entries after XOR cancellation.
    pub fn nonzero_entries(&self) -> usize {
        self.tensors.iter().map(Vec::len).sum()
    }
    /// Evaluates a tensor with two products per nonzero and one idle factor.
    pub(crate) fn weight<F: JoltField>(
        &self,
        router: Router,
        columns: &[F],
        x: &[F],
    ) -> Result<F, PointsError> {
        let restricted = restriction(router, x)?;
        let source_len = source_slots(router).len();
        let (source, selector) = restricted.split_at(source_len);
        let sources = points::equality_table(source)?;
        let selectors = points::equality_table(selector)?;
        let mut sum = F::zero();
        for entry in self.entries(router) {
            let c = columns
                .get(entry.column)
                .ok_or(PointsError::MissingColumn {
                    column: entry.column,
                })?;
            let s = sources.get(entry.source).ok_or(PointsError::Index {
                index: entry.source,
                variables: source_len,
            })?;
            let h = selectors.get(entry.selector).ok_or(PointsError::Index {
                index: entry.selector,
                variables: selector.len(),
            })?;
            sum += *c * *s * *h;
        }
        for slot in idle_slots(router) {
            sum *= F::one()
                + *x.get(*slot).ok_or(PointsError::Dimension {
                    expected: SHORT_VARIABLES,
                    actual: x.len(),
                })?;
        }
        Ok(sum)
    }
}
struct Builder<'a> {
    layout: &'a Layout,
    entries: [BTreeSet<RouteEntry>; 5],
}
impl Builder<'_> {
    fn committed_index(&self, column: usize) -> Result<usize, PointsError> {
        let bank = bank(Router::Variant, self.layout);
        let range = bank
            .committed
            .ok_or(PointsError::MissingColumn { column })?;
        if !range.contains(&column) {
            return Err(PointsError::MissingColumn { column });
        }
        Ok(64 * bank.words.len() + column - range.start)
    }
    fn word(&self, router: Router, word: BankWord, bit: u8) -> Result<usize, PointsError> {
        bank(router, self.layout)
            .words
            .iter()
            .position(|entry| *entry == word)
            .map(|slot| slot * 64 + usize::from(bit))
            .ok_or(PointsError::MissingColumn {
                column: usize::from(bit),
            })
    }
    fn one(&self, router: Router) -> Result<usize, PointsError> {
        let bank = bank(router, self.layout);
        match bank.committed {
            Some(range) => Ok(64 * bank.words.len() + range.len()),
            None => self.word(router, BankWord::One, 0),
        }
    }
    fn selector(&self, router: Router, values: &[(Factor, usize)]) -> Result<usize, PointsError> {
        let mut selector = 0;
        let mut shift = 0;
        for factor in bank(router, self.layout).factors {
            let bits = factor_bits(*factor, self.layout)?;
            let value = values
                .iter()
                .find(|(entry, _)| entry == factor)
                .map(|(_, value)| *value)
                .ok_or(PointsError::Index {
                    index: shift,
                    variables: shift + bits,
                })?;
            if value >= (1 << bits) {
                return Err(PointsError::Index {
                    index: value,
                    variables: bits,
                });
            }
            selector |= value << shift;
            shift += bits;
        }
        Ok(selector)
    }
    fn toggle(&mut self, router: Router, column: usize, source: usize, selector: usize) {
        let entry = RouteEntry {
            column,
            source,
            selector,
        };
        if let Some(set) = self.entries.get_mut(router as usize) {
            if !set.insert(entry) {
                let _ = set.remove(&entry);
            }
        }
    }
    fn indicator(
        &mut self,
        column: usize,
        h: usize,
        chunk: Chunk,
        bit: u8,
    ) -> Result<(), PointsError> {
        let mask = chunk.digit_bit_mask(bit).map_err(|_| PointsError::Index {
            index: usize::from(bit),
            variables: usize::from(chunk.bits()),
        })?;
        for digit in 1..=chunk.indicators() {
            if (mask >> (digit - 1)) & 1 != 0 {
                self.toggle(
                    Router::Variant,
                    column,
                    self.committed_index(usize::from(chunk.start()) + digit - 1)?,
                    h,
                );
            }
        }
        Ok(())
    }
    fn form(
        &mut self,
        r: Router,
        base: usize,
        h: usize,
        v: Variant,
        form: &[Term],
    ) -> Result<(), PointsError> {
        for term in form {
            for wire in term.wires() {
                self.wire(r, base + usize::from(wire.out), h, v, wire.source, wire.bit)?;
            }
        }
        Ok(())
    }
    fn wire(
        &mut self,
        r: Router,
        c: usize,
        h: usize,
        v: Variant,
        source: Source,
        bit: u8,
    ) -> Result<(), PointsError> {
        let i = usize::from(bit);
        if r != Router::Variant {
            let word = match source {
                Source::Rs1Value => BankWord::Rs1Value,
                Source::Rs2Value => BankWord::Rs2Value,
                Source::Imm => BankWord::Imm,
                Source::RamReadValue => BankWord::RamReadValue,
                Source::FallThroughPC => BankWord::FallThroughPC,
                Source::PCPlusImm => BankWord::PCPlusImm,
                Source::One => BankWord::One,
                Source::RdWriteValue
                | Source::PC
                | Source::NextPC
                | Source::Inc
                | Source::RamAddress
                | Source::Pos
                | Source::KeysDiffer
                | Source::ShouldBranch
                | Source::JalrLowBit => return Err(PointsError::MissingColumn { column: c }),
            };
            if word != BankWord::One || bit == 0 {
                self.toggle(r, c, self.word(r, word, bit)?, h);
            }
            return Ok(());
        }
        match source {
            Source::RdWriteValue => {
                for term in v.rd_write_sources() {
                    self.toggle(
                        r,
                        c,
                        match term {
                            RdWriteSource::RdPreValue => self.word(r, BankWord::RdPreValue, bit)?,
                            RdWriteSource::Inc => self.word(r, BankWord::Inc, bit)?,
                        },
                        h,
                    );
                }
            }
            Source::Rs1Value => self.toggle(r, c, self.word(r, BankWord::Rs1Value, bit)?, h),
            Source::Rs2Value => self.toggle(r, c, self.word(r, BankWord::Rs2Value, bit)?, h),
            Source::Imm => self.toggle(r, c, self.word(r, BankWord::Imm, bit)?, h),
            Source::FallThroughPC => {
                self.toggle(r, c, self.word(r, BankWord::FallThroughPC, bit)?, h);
            }
            Source::PCPlusImm => self.toggle(r, c, self.word(r, BankWord::PCPlusImm, bit)?, h),
            Source::PC => self.toggle(r, c, self.word(r, BankWord::PC, bit)?, h),
            Source::NextPC => self.toggle(r, c, self.word(r, BankWord::NextPC, bit)?, h),
            Source::Inc => self.toggle(r, c, self.word(r, BankWord::Inc, bit)?, h),
            Source::One => {
                if bit == 0 {
                    self.toggle(r, c, self.one(r)?, h);
                }
            }
            Source::KeysDiffer | Source::ShouldBranch | Source::JalrLowBit => {
                if bit == 0 {
                    let y = match source {
                        Source::KeysDiffer => self.layout.keys_differ(),
                        Source::ShouldBranch => self.layout.should_branch(),
                        Source::JalrLowBit => self.layout.jalr_low_bit(),
                        Source::Rs1Value
                        | Source::Rs2Value
                        | Source::RdWriteValue
                        | Source::Imm
                        | Source::FallThroughPC
                        | Source::PCPlusImm
                        | Source::PC
                        | Source::NextPC
                        | Source::Inc
                        | Source::RamReadValue
                        | Source::RamAddress
                        | Source::Pos
                        | Source::One => return Err(PointsError::MissingColumn { column: c }),
                    };
                    self.toggle(r, c, self.committed_index(y)?, h);
                }
            }
            Source::Pos => {
                if bit < 6 {
                    let chunk = self
                        .layout
                        .pos_ra()
                        .get(i / 3)
                        .copied()
                        .ok_or(PointsError::MissingColumn { column: c })?;
                    self.indicator(c, h, chunk, bit % 3)?;
                }
            }
            Source::RamAddress => {
                if bit < 3 {
                    self.indicator(
                        c,
                        h,
                        self.layout
                            .pos_ra()
                            .first()
                            .copied()
                            .ok_or(PointsError::MissingColumn { column: c })?,
                        bit,
                    )?;
                } else {
                    let mut start = 3;
                    for chunk in self.layout.ram_ra() {
                        let end = start + usize::from(chunk.bits());
                        if i >= start && i < end {
                            self.indicator(c, h, *chunk, (i - start) as u8)?;
                        }
                        start = end;
                    }
                }
            }
            Source::RamReadValue => return Err(PointsError::MissingColumn { column: c }),
        }
        Ok(())
    }
}
