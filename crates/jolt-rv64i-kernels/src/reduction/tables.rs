use super::{ColumnMap, ReductionError};
use crate::packed::lift::WordLift;
use crate::par::CycleChunks;
use crate::source::{CycleSource, ValidatedTrace};
use jolt_field::F128;
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;

fn check_weights(weights: &[Vec<F128>]) -> Result<(), ReductionError> {
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
    Ok(())
}

fn output_views(tables: &mut [Vec<F128>], chunk: usize) -> Vec<(usize, &mut [F128])> {
    let mut views: Vec<_> = tables
        .iter_mut()
        .flat_map(|table| table.chunks_mut(chunk).enumerate())
        .collect();
    views.sort_by_key(|&(index, _)| index);
    views
}

#[derive(Default)]
struct Sums {
    a: F128,
    b: F128,
    c: F128,
    d: F128,
}
impl Sums {
    #[inline(always)]
    fn values(self) -> [F128; 4] {
        [self.a, self.b, self.c, self.d]
    }
}

enum GroupTables {
    Eight([Option<Box<[F128; 8]>>; 4]),
    Sixteen([Option<Box<[F128; 16]>>; 4]),
    Byte([Option<Box<[F128; 256]>>; 4]),
    Other([Option<Vec<F128>>; 4]),
}
impl GroupTables {
    fn new(tables: Vec<(usize, Vec<F128>)>) -> Self {
        match tables.first().map(|(_, table)| table.len()) {
            Some(8) => Self::Eight(Self::fixed_tables(tables)),
            Some(16) => Self::Sixteen(Self::fixed_tables(tables)),
            Some(256) => Self::Byte(Self::fixed_tables(tables)),
            _ => {
                let mut entries = std::array::from_fn(|_| None);
                for (weight, table) in tables {
                    entries[weight] = Some(table);
                }
                Self::Other(entries)
            }
        }
    }
    fn fixed_tables<const N: usize>(
        tables: Vec<(usize, Vec<F128>)>,
    ) -> [Option<Box<[F128; N]>>; 4] {
        let mut entries = std::array::from_fn(|_| None);
        for (weight, table) in tables {
            let mut values = Box::new([F128::from_raw(0); N]);
            values.copy_from_slice(&table);
            entries[weight] = Some(values);
        }
        entries
    }
    #[inline(always)]
    fn add_fixed<const N: usize>(
        entries: &[Option<Box<[F128; N]>>; 4],
        index: usize,
        sums: &mut Sums,
    ) {
        let index = index & (N - 1);
        if let Some(table) = &entries[0] {
            sums.a += table[index];
        }
        if let Some(table) = &entries[1] {
            sums.b += table[index];
        }
        if let Some(table) = &entries[2] {
            sums.c += table[index];
        }
        if let Some(table) = &entries[3] {
            sums.d += table[index];
        }
    }
    #[inline(always)]
    fn add(&self, index: usize, sums: &mut Sums) {
        match self {
            Self::Eight(entries) => Self::add_fixed(entries, index, sums),
            Self::Sixteen(entries) => Self::add_fixed(entries, index, sums),
            Self::Byte(entries) => Self::add_fixed(entries, index, sums),
            Self::Other(entries) => {
                if let Some(table) = &entries[0] {
                    sums.a += table[index];
                }
                if let Some(table) = &entries[1] {
                    sums.b += table[index];
                }
                if let Some(table) = &entries[2] {
                    sums.c += table[index];
                }
                if let Some(table) = &entries[3] {
                    sums.d += table[index];
                }
            }
        }
    }
}
enum GroupLift {
    Word {
        trace_word: usize,
        lifts: [Option<Box<WordLift>>; 4],
    },
    Indicators {
        column: usize,
        tables: GroupTables,
    },
    Flags {
        columns: Vec<usize>,
        tables: GroupTables,
    },
}

#[inline(always)]
fn group_values<S: CycleSource>(source: &S, groups: &[GroupLift], cycle: usize) -> [F128; 4] {
    let mut sums = Sums::default();
    for group in groups {
        match group {
            GroupLift::Word { trace_word, lifts } => {
                let word = source.trace_word(*trace_word, cycle);
                if let Some(lift) = &lifts[0] {
                    sums.a += lift.lift(word);
                }
                if let Some(lift) = &lifts[1] {
                    sums.b += lift.lift(word);
                }
                if let Some(lift) = &lifts[2] {
                    sums.c += lift.lift(word);
                }
                if let Some(lift) = &lifts[3] {
                    sums.d += lift.lift(word);
                }
            }
            GroupLift::Indicators { column, tables } => {
                tables.add(source.digit(*column, cycle).unwrap_or(0), &mut sums);
            }
            GroupLift::Flags { columns, tables } => {
                let mask = columns.iter().enumerate().fold(0, |mask, (bit, &column)| {
                    mask | (usize::from(source.digit(column, cycle).is_some()) << bit)
                });
                tables.add(mask, &mut sums);
            }
        }
    }
    sums.values()
}

/// Build every `G_i[j] = sum_y weights[i][y] * Bits[y,j]` in one cycle pass.
/// The validated source must remain immutable. Map ranges must be disjoint and
/// cover every nonzero weight; words span 64 columns, indicators `2^bits - 1`.
pub fn g_pass_digits<S: CycleSource>(
    trace: &ValidatedTrace<S>,
    map: &[ColumnMap],
    weights: &[Vec<F128>],
) -> Result<Vec<Vec<F128>>, ReductionError> {
    check_weights(weights)?;
    let source = trace.source();
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
    let groups: Vec<_> = ranges
        .iter()
        .map(|(entry, range)| {
            let supported: Vec<_> = weights
                .iter()
                .enumerate()
                .filter(|(_, weight)| {
                    weight[range.clone()]
                        .iter()
                        .any(|&v| v != F128::from_raw(0))
                })
                .collect();
            match entry {
                ColumnMap::Word { trace_word, .. } => {
                    let mut lifts = std::array::from_fn(|_| None);
                    for (index, weight) in supported {
                        let mut values = [F128::from_raw(0); 64];
                        values.copy_from_slice(&weight[range.clone()]);
                        lifts[index] = Some(Box::new(WordLift::new(&values)));
                    }
                    GroupLift::Word {
                        trace_word: *trace_word,
                        lifts,
                    }
                }
                ColumnMap::Indicators { column, .. } => GroupLift::Indicators {
                    column: *column,
                    tables: GroupTables::new(
                        supported
                            .into_iter()
                            .map(|(index, weight)| {
                                let mut values = Vec::with_capacity(range.len() + 1);
                                values.push(F128::from_raw(0));
                                values.extend_from_slice(&weight[range.clone()]);
                                (index, values)
                            })
                            .collect(),
                    ),
                },
                ColumnMap::Flags { columns, .. } => GroupLift::Flags {
                    columns: columns.clone(),
                    tables: GroupTables::new(
                        supported
                            .into_iter()
                            .map(|(index, weight)| {
                                let mut values = vec![F128::from_raw(0); 1 << columns.len()];
                                for mask in 1_usize..values.len() {
                                    let bit = mask.trailing_zeros() as usize;
                                    values[mask] =
                                        values[mask & (mask - 1)] + weight[range.start + bit];
                                }
                                (index, values)
                            })
                            .collect(),
                    ),
                },
            }
        })
        .collect();
    let mut tables = (0..weights.len())
        .map(|_| unsafe_allocate_zero_vec(source.cycles()))
        .collect::<Vec<Vec<F128>>>();
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
    let mut views = output_views(&mut tables, chunk);
    views
        .par_chunks_mut(weights.len())
        .enumerate()
        .for_each(|(index, outputs)| {
            if let [(_, a), (_, b), (_, c)] = outputs {
                for (cycle, ((a, b), c)) in
                    a.iter_mut().zip(b.iter_mut()).zip(c.iter_mut()).enumerate()
                {
                    let values = group_values(source.as_ref(), &groups, index * chunk + cycle);
                    *a = values[0];
                    *b = values[1];
                    *c = values[2];
                }
                return;
            }

            for cycle in 0..outputs[0].1.len() {
                let values = group_values(source.as_ref(), &groups, index * chunk + cycle);
                for ((_, output), value) in outputs.iter_mut().zip(values) {
                    output[cycle] = value;
                }
            }
        });
    drop(views);
    Ok(tables)
}
