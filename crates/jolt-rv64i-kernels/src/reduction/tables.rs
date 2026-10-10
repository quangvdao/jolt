use super::map::{Groups, Plan};
use super::{ColumnMap, ReductionError};
use crate::par::CycleChunks;
use crate::source::{CycleSource, ValidatedTrace};
use jolt_field::F128;
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;

const ZERO: F128 = F128::from_raw(0);

pub(super) enum Encoding {
    Bits,
    Indicators,
}

pub(super) fn fill_table<const K: usize>(
    arena: &mut Vec<[F128; K]>,
    weights: &[Vec<F128>],
    destinations: &[usize; K],
    start: usize,
    bits: usize,
    encoding: Encoding,
) {
    let offset = arena.len();
    arena.resize(offset + (1 << bits), [ZERO; K]);
    let table = &mut arena[offset..];
    for index in 1_usize..table.len() {
        match encoding {
            Encoding::Bits => {
                let bit = index.trailing_zeros() as usize;
                let prior = table[index & (index - 1)];
                table[index] = std::array::from_fn(|slot| {
                    prior[slot] + weights[destinations[slot]][start + bit]
                });
            }
            Encoding::Indicators => {
                // A digit selects a one-hot column, not the bits of its index.
                table[index] =
                    std::array::from_fn(|slot| weights[destinations[slot]][start + index - 1]);
            }
        }
    }
}

struct WordView<'a, const K: usize> {
    word: usize,
    tables: &'a [[[F128; K]; 256]; 8],
}
struct DigitView<'a, const K: usize> {
    column: usize,
    table: &'a [[F128; K]],
    mask: usize,
}
struct FlagView<'a, const K: usize> {
    columns: [usize; 8],
    count: usize,
    table: &'a [[F128; K]],
    mask: usize,
}
struct Views<R, const K: usize> {
    destinations: [usize; K],
    ranges: Vec<R>,
}
struct GroupViews<'a, const K: usize> {
    words: Vec<Views<WordView<'a, K>, K>>,
    indicators: Vec<Views<DigitView<'a, K>, K>>,
    flags: Vec<Views<FlagView<'a, K>, K>>,
}
impl<const K: usize> Groups<K> {
    #[expect(
        clippy::expect_used,
        reason = "compiled arena offsets and byte-table dimensions are fixed"
    )]
    fn views(&self) -> GroupViews<'_, K> {
        GroupViews {
            words: self
                .words
                .iter()
                .map(|group| Views {
                    destinations: group.destinations,
                    ranges: group
                        .ranges
                        .iter()
                        .map(|range| WordView {
                            word: range.word,
                            tables: group.arena[range.offset..range.offset + 2048]
                                .as_chunks::<256>()
                                .0
                                .try_into()
                                .expect("eight complete byte tables"),
                        })
                        .collect(),
                })
                .collect(),
            indicators: self
                .indicators
                .iter()
                .map(|group| Views {
                    destinations: group.destinations,
                    ranges: group
                        .ranges
                        .iter()
                        .map(|range| DigitView {
                            column: range.column,
                            table: &group.arena[range.offset..=range.offset + range.mask],
                            mask: range.mask,
                        })
                        .collect(),
                })
                .collect(),
            flags: self
                .flags
                .iter()
                .map(|group| Views {
                    destinations: group.destinations,
                    ranges: group
                        .ranges
                        .iter()
                        .map(|range| FlagView {
                            columns: range.columns,
                            count: range.count,
                            table: &group.arena[range.offset..=range.offset + range.mask],
                            mask: range.mask,
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

impl<const K: usize> GroupViews<'_, K> {
    #[inline(always)]
    #[expect(
        clippy::expect_used,
        reason = "compiled destinations are distinct and below four"
    )]
    fn destinations<'a>(
        output: &'a mut [[F128; 256]; 4],
        destinations: &[usize; K],
    ) -> [&'a mut [F128; 256]; K] {
        output
            .get_disjoint_mut(*destinations)
            .expect("distinct output columns")
    }
    #[inline(always)]
    #[expect(
        clippy::expect_used,
        reason = "the table dispatcher selects its exact fixed width"
    )]
    fn fixed<const W: usize, I: Fn(usize) -> usize>(
        table: &[[F128; K]],
        output: &mut [&mut [F128; 256]; K],
        len: usize,
        index: I,
    ) {
        let table: &[[F128; K]; W] = table.try_into().expect("fixed table width");
        for cycle in 0..len.min(256) {
            let values = table[index(cycle) & (W - 1)];
            for (destination, value) in output.iter_mut().zip(values) {
                destination[cycle] += value;
            }
        }
    }
    fn indexed<I: Fn(usize) -> usize>(
        table: &[[F128; K]],
        output: &mut [&mut [F128; 256]; K],
        len: usize,
        index: I,
    ) {
        macro_rules! widths {
            ($($width:literal),*) => {
                match table.len() {
                    $($width => Self::fixed::<$width, _>(table, output, len, index),)*
                    _ => {},
                }
            };
        }
        widths!(1, 2, 4, 8, 16, 32, 64, 128, 256);
    }
    fn accumulate<S: CycleSource>(
        &self,
        source: &S,
        start: usize,
        len: usize,
        output: &mut [[F128; 256]; 4],
    ) {
        for group in &self.words {
            let mut destinations = Self::destinations(output, &group.destinations);
            for range in &group.ranges {
                for cycle in 0..len.min(256) {
                    let bytes = source.trace_word(range.word, start + cycle).to_le_bytes();
                    let mut value = [ZERO; K];
                    for (table, byte) in range.tables.iter().zip(bytes) {
                        for (sum, value) in value.iter_mut().zip(table[usize::from(byte)]) {
                            *sum += value;
                        }
                    }
                    for (destination, value) in destinations.iter_mut().zip(value) {
                        destination[cycle] += value;
                    }
                }
            }
        }
        for group in &self.indicators {
            let mut destinations = Self::destinations(output, &group.destinations);
            for range in &group.ranges {
                Self::indexed(range.table, &mut destinations, len, |cycle| {
                    source.digit(range.column, start + cycle).unwrap_or(0) & range.mask
                });
            }
        }
        for group in &self.flags {
            let mut destinations = Self::destinations(output, &group.destinations);
            for range in &group.ranges {
                Self::indexed(range.table, &mut destinations, len, |cycle| {
                    let mut mask = 0;
                    for (bit, &column) in range.columns[..range.count].iter().enumerate() {
                        mask |= usize::from(source.digit(column, start + cycle).is_some()) << bit;
                    }
                    mask & range.mask
                });
            }
        }
    }
}
struct Pass<'a> {
    one: GroupViews<'a, 1>,
    two: GroupViews<'a, 2>,
    three: GroupViews<'a, 3>,
    four: GroupViews<'a, 4>,
}
impl Pass<'_> {
    #[expect(
        clippy::expect_used,
        reason = "the caller dispatches exactly N validated outputs with identical geometry"
    )]
    fn run<const N: usize, S: CycleSource>(
        &self,
        source: &S,
        tables: &mut [Vec<F128>],
        chunk: usize,
    ) {
        let tables: &mut [Vec<F128>; N] = tables.try_into().expect("N outputs");
        let mut iterators = tables.each_mut().map(|table| table.chunks_mut(chunk));
        let mut views: Vec<[&mut [F128]; N]> = (0..source.cycles() / chunk)
            .map(|_| std::array::from_fn(|slot| iterators[slot].next().expect("complete chunks")))
            .collect();
        views
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, outputs)| {
                let start = index * chunk;
                let chunk_len = outputs[0].len();
                let mut tiles = outputs.each_mut().map(|output| output.chunks_mut(256));
                let mut offset = 0;
                while offset < chunk_len {
                    let mut output = [[ZERO; 256]; 4];
                    let len = 256.min(chunk_len - offset);
                    self.one
                        .accumulate(source, start + offset, len, &mut output);
                    self.two
                        .accumulate(source, start + offset, len, &mut output);
                    self.three
                        .accumulate(source, start + offset, len, &mut output);
                    self.four
                        .accumulate(source, start + offset, len, &mut output);
                    let mut targets: [&mut [F128]; N] =
                        std::array::from_fn(|slot| tiles[slot].next().expect("complete tile"));
                    for (target, column) in targets.iter_mut().zip(&output) {
                        target.copy_from_slice(&column[..len]);
                    }
                    offset += len;
                }
            });
    }
}

/// Build up to four tables `G_i[j] = sum_y weights[i][y] * Bits[y,j]` in one
/// parallel cycle pass. Map ranges are disjoint and cover every nonzero weight.
/// Words span 64 columns, indicators `2^bits - 1`, and flags encode presence.
/// Source immutability and agreement of the map with the committed bits are
/// required of the caller, not checked; the verifier detects a false statement
/// through its final batched check.
pub fn g_pass_digits<S: CycleSource>(
    trace: &ValidatedTrace<S>,
    map: &[ColumnMap],
    weights: &[Vec<F128>],
) -> Result<Vec<Vec<F128>>, ReductionError> {
    let source = trace.source();
    let plan = Plan::compile(source.as_ref(), map, weights)?;
    let mut tables: Vec<Vec<F128>> = (0..weights.len())
        .map(|_| unsafe_allocate_zero_vec(source.cycles()))
        .collect();
    if weights.is_empty() {
        return Ok(tables);
    }
    let chunk = CycleChunks::new(source.cycles().ilog2() as usize, 0)
        .map_err(|_| ReductionError::TableLength {
            table: 0,
            actual: source.cycles(),
            expected: None,
        })?
        .chunk_len();
    let pass = Pass {
        one: plan.one.views(),
        two: plan.two.views(),
        three: plan.three.views(),
        four: plan.four.views(),
    };
    match weights.len() {
        1 => pass.run::<1, _>(source.as_ref(), &mut tables, chunk),
        2 => pass.run::<2, _>(source.as_ref(), &mut tables, chunk),
        3 => pass.run::<3, _>(source.as_ref(), &mut tables, chunk),
        _ => pass.run::<4, _>(source.as_ref(), &mut tables, chunk),
    }
    Ok(tables)
}
