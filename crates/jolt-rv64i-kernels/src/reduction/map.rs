use super::tables::{fill_table, Encoding};
use super::ReductionError;
use crate::source::CycleSource;
use jolt_field::F128;
use std::ops::Range;

/// A source range in the 256 committed columns. Indicator digit zero and absence
/// both encode zero; digit `k >= 1` selects column `start + k - 1`.
/// Flags span one column per zero-bit digit and encode its presence, in list order.
/// Flag groups contain one through eight source columns.
#[derive(Clone, Debug)]
pub enum ColumnMap {
    Word { start: usize, trace_word: usize },
    Indicators { start: usize, column: usize },
    Flags { start: usize, columns: Vec<usize> },
}

#[derive(Clone, Copy)]
pub(super) struct WordRange {
    pub word: usize,
    pub offset: usize,
}
#[derive(Clone, Copy)]
pub(super) struct DigitRange {
    pub column: usize,
    pub offset: usize,
    pub mask: usize,
}
#[derive(Clone, Copy)]
pub(super) struct FlagRange {
    pub columns: [usize; 8],
    pub count: usize,
    pub offset: usize,
    pub mask: usize,
}
pub(super) struct Group<R, const K: usize> {
    pub destinations: [usize; K],
    pub ranges: Vec<R>,
    pub arena: Vec<[F128; K]>,
}
pub(super) struct Groups<const K: usize> {
    pub words: Vec<Group<WordRange, K>>,
    pub indicators: Vec<Group<DigitRange, K>>,
    pub flags: Vec<Group<FlagRange, K>>,
}
impl<const K: usize> Default for Groups<K> {
    fn default() -> Self {
        Self {
            words: Vec::new(),
            indicators: Vec::new(),
            flags: Vec::new(),
        }
    }
}
impl<const K: usize> Groups<K> {
    fn group<R>(groups: &mut Vec<Group<R, K>>, destinations: [usize; K]) -> &mut Group<R, K> {
        let position = groups
            .iter()
            .position(|group| group.destinations == destinations);
        if let Some(position) = position {
            return &mut groups[position];
        }
        groups.push(Group {
            destinations,
            ranges: Vec::new(),
            arena: Vec::new(),
        });
        let index = groups.len() - 1;
        &mut groups[index]
    }
    fn compact(&mut self) {
        for group in &mut self.words {
            group.arena.shrink_to_fit();
        }
        for group in &mut self.indicators {
            group.arena.shrink_to_fit();
        }
        for group in &mut self.flags {
            group.arena.shrink_to_fit();
        }
    }
    fn add(
        &mut self,
        entry: &ColumnMap,
        range: Range<usize>,
        destinations: [usize; K],
        weights: &[Vec<F128>],
    ) {
        match entry {
            ColumnMap::Word { trace_word, .. } => {
                let group = Self::group(&mut self.words, destinations);
                let offset = group.arena.len();
                for byte in 0..8 {
                    fill_table(
                        &mut group.arena,
                        weights,
                        &destinations,
                        range.start + 8 * byte,
                        8,
                        Encoding::Bits,
                    );
                }
                group.ranges.push(WordRange {
                    word: *trace_word,
                    offset,
                });
            }
            ColumnMap::Indicators { column, .. } => {
                let group = Self::group(&mut self.indicators, destinations);
                let offset = group.arena.len();
                let bits = (range.len() + 1).ilog2() as usize;
                fill_table(
                    &mut group.arena,
                    weights,
                    &destinations,
                    range.start,
                    bits,
                    Encoding::Indicators,
                );
                group.ranges.push(DigitRange {
                    column: *column,
                    offset,
                    mask: range.len(),
                });
            }
            ColumnMap::Flags { columns, .. } => {
                let group = Self::group(&mut self.flags, destinations);
                let offset = group.arena.len();
                fill_table(
                    &mut group.arena,
                    weights,
                    &destinations,
                    range.start,
                    columns.len(),
                    Encoding::Bits,
                );
                let mut fixed = [0; 8];
                fixed[..columns.len()].copy_from_slice(columns);
                group.ranges.push(FlagRange {
                    columns: fixed,
                    count: columns.len(),
                    offset,
                    mask: (1 << columns.len()) - 1,
                });
            }
        }
    }
}
pub(super) struct Plan {
    pub one: Groups<1>,
    pub two: Groups<2>,
    pub three: Groups<3>,
    pub four: Groups<4>,
}
impl Plan {
    pub fn compile<S: CycleSource>(
        source: &S,
        map: &[ColumnMap],
        weights: &[Vec<F128>],
    ) -> Result<Self, ReductionError> {
        if weights.len() > 4 {
            return Err(ReductionError::WeightCount {
                count: weights.len(),
            });
        }
        for (weight, values) in weights.iter().enumerate() {
            if values.len() != 256 {
                return Err(ReductionError::WeightLength {
                    weight,
                    actual: values.len(),
                });
            }
        }
        let mut covered = [false; 256];
        let mut ranges = Vec::with_capacity(map.len());
        for entry in map.iter().cloned() {
            let (start, length) = match entry.clone() {
                ColumnMap::Word { start, trace_word } => {
                    if trace_word >= source.trace_words() {
                        return Err(ReductionError::MapTraceWord {
                            trace_word,
                            words: source.trace_words(),
                        });
                    }
                    (start, 64)
                }
                ColumnMap::Indicators { start, column } => {
                    if column >= source.digit_columns() {
                        return Err(ReductionError::MapColumn {
                            column,
                            columns: source.digit_columns(),
                        });
                    }
                    (start, (1_usize << source.bits(column)) - 1)
                }
                ColumnMap::Flags { start, columns } => {
                    if columns.is_empty() || columns.len() > 8 {
                        return Err(ReductionError::MapFlags {
                            count: columns.len(),
                            offending_column: None,
                        });
                    }
                    for &column in &columns {
                        if column >= source.digit_columns() {
                            return Err(ReductionError::MapColumn {
                                column,
                                columns: source.digit_columns(),
                            });
                        }
                        if source.bits(column) != 0 {
                            return Err(ReductionError::MapFlags {
                                count: columns.len(),
                                offending_column: Some((column, source.bits(column))),
                            });
                        }
                    }
                    (start, columns.len())
                }
            };
            if start > 256 || length > 256 - start {
                return Err(ReductionError::MapRange { start, length });
            }
            for (offset, flag) in covered[start..start + length].iter_mut().enumerate() {
                if *flag {
                    return Err(ReductionError::MapOverlap {
                        column: start + offset,
                    });
                }
                *flag = true;
            }
            ranges.push((entry, start..start + length));
        }
        for (weight, values) in weights.iter().enumerate() {
            for (column, (&value, &covered)) in values.iter().zip(&covered).enumerate() {
                if !covered && value != F128::from_raw(0) {
                    return Err(ReductionError::Uncovered { weight, column });
                }
            }
        }

        let mut plan = Self {
            one: Groups::default(),
            two: Groups::default(),
            three: Groups::default(),
            four: Groups::default(),
        };
        for (entry, range) in ranges {
            let mut supported = [0; 4];
            let mut count = 0;
            for (slot, weight) in weights.iter().enumerate() {
                if weight[range.clone()]
                    .iter()
                    .any(|&v| v != F128::from_raw(0))
                {
                    supported[count] = slot;
                    count += 1;
                }
            }
            match count {
                0 => {}
                1 => plan.one.add(&entry, range, [supported[0]], weights),
                2 => plan
                    .two
                    .add(&entry, range, [supported[0], supported[1]], weights),
                3 => plan.three.add(
                    &entry,
                    range,
                    [supported[0], supported[1], supported[2]],
                    weights,
                ),
                _ => plan.four.add(&entry, range, supported, weights),
            }
        }
        plan.one.compact();
        plan.two.compact();
        plan.three.compact();
        plan.four.compact();
        Ok(plan)
    }
}
