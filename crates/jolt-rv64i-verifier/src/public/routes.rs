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

pub const ROUTERS: [Router; 5] = [
    Router::Variant,
    Router::Shift,
    Router::Memory,
    Router::Compare,
    Router::Branch,
];

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
    if point.len() != 17 {
        return Err(PointsError::Dimension {
            expected: 17,
            actual: point.len(),
        });
    }
    source_slots(router)
        .iter()
        .chain(selector_slots(router))
        .map(|&slot| {
            point.get(slot).copied().ok_or(PointsError::Dimension {
                expected: 17,
                actual: point.len(),
            })
        })
        .collect()
}
/// Recovers the shared short point from the router with no idle coordinates.
pub(crate) fn short_point<F: JoltField>(compare: &[F]) -> Result<Vec<F>, PointsError> {
    if compare.len() != 17 {
        return Err(PointsError::Dimension {
            expected: 17,
            actual: compare.len(),
        });
    }
    let mut point = vec![F::zero(); 17];
    for (slot, value) in source_slots(Router::Compare)
        .iter()
        .chain(selector_slots(Router::Compare))
        .zip(compare)
    {
        let target = point.get_mut(*slot).ok_or(PointsError::Dimension {
            expected: 17,
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
            let h = variant.index();
            let line = variant.line();
            builder.toggle(Router::Variant, 16, builder.one(), h);
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
                    usize::from(pos) | (kind << 6),
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
                let h = usize::from(pos) | (index << 3);
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
                let h = usize::from(pos) | (kind << 6);
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
        builder.form(Router::Branch, 704, 0, Variant::NOOP, BRANCH_FORM)?;
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
        let sources = equality_table(source)?;
        let selectors = equality_table(selector)?;
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
                    expected: 17,
                    actual: x.len(),
                })?;
        }
        Ok(sum)
    }
}
pub(crate) fn equality_table<F: JoltField>(point: &[F]) -> Result<Vec<F>, PointsError> {
    let (low, high) = point.split_at(point.len().min(6));
    let low = points::eq_table(low)?;
    let high = points::eq_table(high)?;
    Ok(high
        .into_iter()
        .flat_map(|h| low.iter().map(move |l| *l * h))
        .collect())
}
struct Builder<'a> {
    layout: &'a Layout,
    entries: [BTreeSet<RouteEntry>; 5],
}
impl Builder<'_> {
    fn g(&self) -> usize {
        self.layout
            .ram_ra()
            .first()
            .map_or(64, |c| usize::from(c.start()))
    }
    fn one(&self) -> usize {
        576 + self.layout.used_columns() - self.g()
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
                    576 + usize::from(chunk.start()) + digit - 1 - self.g(),
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
            let slot = match (r, source) {
                (Router::Shift | Router::Compare, Source::Rs1Value)
                | (Router::Memory, Source::RamReadValue)
                | (Router::Branch, Source::FallThroughPC) => 0,
                (Router::Memory | Router::Compare, Source::Rs2Value)
                | (Router::Branch, Source::PCPlusImm) => 1,
                (Router::Compare, Source::Imm) => 2,
                (Router::Compare, Source::One) => 3,
                _ => return Err(PointsError::MissingColumn { column: c }),
            };
            if source != Source::One || bit == 0 {
                self.toggle(r, c, slot * 64 + i, h);
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
                            RdWriteSource::RdPreValue => 128 + i,
                            RdWriteSource::Inc => 512 + i,
                        },
                        h,
                    );
                }
            }
            Source::Rs1Value => self.toggle(r, c, i, h),
            Source::Rs2Value => self.toggle(r, c, 64 + i, h),
            Source::Imm => self.toggle(r, c, 192 + i, h),
            Source::FallThroughPC => self.toggle(r, c, 256 + i, h),
            Source::PCPlusImm => self.toggle(r, c, 320 + i, h),
            Source::PC => self.toggle(r, c, 384 + i, h),
            Source::NextPC => self.toggle(r, c, 448 + i, h),
            Source::Inc => self.toggle(r, c, 512 + i, h),
            Source::One => {
                if bit == 0 {
                    self.toggle(r, c, self.one(), h);
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
                    self.toggle(r, c, 576 + y - self.g(), h);
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
