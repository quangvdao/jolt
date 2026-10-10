//! Weighted committed-bit tables and their low-variable-first claim reduction.

use crate::packed::lift::WordLift;
use crate::par::CycleChunks;
use crate::round::eq::{eq_table, split_eq};
use crate::round::RoundError;
use crate::source::{CycleSource, ValidatedTrace};
use jolt_field::{Accumulator, F128Accumulator, F128};
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::prelude::*;
use thiserror::Error;

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

/// Rejected reduction geometry or a diagnostic claim mismatch.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ReductionError {
    #[error("table {table} has invalid length {actual}; expected {expected:?}")]
    TableLength {
        table: usize,
        actual: usize,
        expected: Option<usize>,
    },
    #[error("leg {leg} selects table {table} outside {tables} tables")]
    LegTable {
        leg: usize,
        table: usize,
        tables: usize,
    },
    #[error("leg {leg} point has {actual} coordinates, expected {expected}")]
    LegPoint {
        leg: usize,
        actual: usize,
        expected: usize,
    },
    #[error("weight {weight} has {actual} columns, expected 256")]
    WeightLength { weight: usize, actual: usize },
    #[error("map range at {start} of length {length} exceeds 256 columns")]
    MapRange { start: usize, length: usize },
    #[error("map ranges overlap at column {column}")]
    MapOverlap { column: usize },
    #[error("map trace word {trace_word} is outside {words} words")]
    MapTraceWord { trace_word: usize, words: usize },
    #[error("map digit column {column} is outside {columns} columns")]
    MapColumn { column: usize, columns: usize },
    #[error("flag group of {count} columns has invalid column and width {offending_column:?}")]
    MapFlags {
        count: usize,
        offending_column: Option<(usize, usize)>,
    },
    #[error("weight {weight} is nonzero on uncovered column {column}")]
    Uncovered { weight: usize, column: usize },
    #[error("leg {leg} claims {expected:?}, but its table gives {actual:?}")]
    Claim {
        leg: usize,
        expected: F128,
        actual: F128,
    },
    #[error("reduction values require completed rounds")]
    Unfinished,
    #[error(transparent)]
    Round(#[from] RoundError),
}

fn check_weights(weights: &[Vec<F128>]) -> Result<(), ReductionError> {
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

enum DigitLift {
    Flags {
        columns: Vec<usize>,
        values: Vec<F128>,
    },
    Word {
        trace_word: usize,
        lift: Box<WordLift>,
    },
    Indicators {
        column: usize,
        values: Vec<F128>,
    },
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
    fn add(&mut self, weight: usize, value: F128) {
        match weight & 3 {
            0 => self.a += value,
            1 => self.b += value,
            2 => self.c += value,
            _ => self.d += value,
        }
    }
    #[inline(always)]
    fn values(self) -> [F128; 4] {
        [self.a, self.b, self.c, self.d]
    }
}

enum GroupEntry {
    Eight(Box<[F128; 8]>),
    Sixteen(Box<[F128; 16]>),
    Byte(Box<[F128; 256]>),
    Other(Vec<F128>),
}
impl GroupEntry {
    fn new(table: Vec<F128>) -> Self {
        match table.len() {
            8 => {
                let mut entries = Box::new([F128::from_raw(0); 8]);
                entries.copy_from_slice(&table);
                Self::Eight(entries)
            }
            16 => {
                let mut entries = Box::new([F128::from_raw(0); 16]);
                entries.copy_from_slice(&table);
                Self::Sixteen(entries)
            }
            256 => {
                let mut entries = Box::new([F128::from_raw(0); 256]);
                entries.copy_from_slice(&table);
                Self::Byte(entries)
            }
            _ => Self::Other(table),
        }
    }
    #[inline(always)]
    fn value(&self, index: usize) -> F128 {
        match self {
            Self::Eight(table) => table[index & 7],
            Self::Sixteen(table) => table[index & 15],
            Self::Byte(table) => table[index & 255],
            Self::Other(table) => table[index],
        }
    }
}
struct GroupTables {
    entries: [Option<GroupEntry>; 4],
}
impl GroupTables {
    fn new(tables: Vec<(usize, Vec<F128>)>) -> Self {
        let mut entries = std::array::from_fn(|_| None);
        for (weight, table) in tables {
            entries[weight] = Some(GroupEntry::new(table));
        }
        Self { entries }
    }
    #[inline(always)]
    fn add(&self, index: usize, sums: &mut Sums) {
        if let Some(table) = &self.entries[0] {
            sums.a += table.value(index);
        }
        if let Some(table) = &self.entries[1] {
            sums.b += table.value(index);
        }
        if let Some(table) = &self.entries[2] {
            sums.c += table.value(index);
        }
        if let Some(table) = &self.entries[3] {
            sums.d += table.value(index);
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
    let lifts: Vec<Vec<DigitLift>> = if weights.len() <= 4 {
        Vec::new()
    } else {
        weights
            .iter()
            .map(|weight| {
                ranges
                    .iter()
                    .filter(|(_, range)| {
                        weight[range.clone()]
                            .iter()
                            .any(|&w| w != F128::from_raw(0))
                    })
                    .map(|(entry, range)| match entry.clone() {
                        ColumnMap::Word { trace_word, .. } => {
                            let mut values = [F128::from_raw(0); 64];
                            values.copy_from_slice(&weight[range.clone()]);
                            DigitLift::Word {
                                trace_word,
                                lift: Box::new(WordLift::new(&values)),
                            }
                        }
                        ColumnMap::Indicators { column, .. } => {
                            let mut values = Vec::with_capacity(range.len() + 1);
                            values.push(F128::from_raw(0));
                            values.extend_from_slice(&weight[range.clone()]);
                            DigitLift::Indicators { column, values }
                        }
                        ColumnMap::Flags { columns, .. } => {
                            let mut values = vec![F128::from_raw(0); 1 << columns.len()];
                            for mask in 1_usize..values.len() {
                                let bit = mask.trailing_zeros() as usize;
                                values[mask] =
                                    values[mask & (mask - 1)] + weight[range.start + bit];
                            }
                            DigitLift::Flags { columns, values }
                        }
                    })
                    .collect()
            })
            .collect()
    };
    let groups: Vec<_> = if weights.len() <= 4 {
        ranges
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
            .collect()
    } else {
        Vec::new()
    };
    let mut tables = vec![vec![F128::from_raw(0); source.cycles()]; weights.len()];
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
                if weights.len() <= 4 {
                    let values = group_values(source.as_ref(), &groups, index * chunk + cycle);
                    for ((_, output), value) in outputs.iter_mut().zip(values) {
                        output[cycle] = value;
                    }
                    continue;
                }
                for ((_, output), lifts) in outputs.iter_mut().zip(&lifts) {
                    let mut value = F128::from_raw(0);
                    for lift in lifts {
                        value += match lift {
                            DigitLift::Flags { columns, values } => {
                                let mask =
                                    columns.iter().enumerate().fold(0, |mask, (bit, &column)| {
                                        mask | (usize::from(
                                            source.digit(column, index * chunk + cycle).is_some(),
                                        ) << bit)
                                    });
                                values[mask]
                            }
                            DigitLift::Word { trace_word, lift } => {
                                lift.lift(source.trace_word(*trace_word, index * chunk + cycle))
                            }
                            DigitLift::Indicators { column, values } => {
                                values[source.digit(*column, index * chunk + cycle).unwrap_or(0)]
                            }
                        };
                    }
                    output[cycle] = value;
                }
            }
        });
    drop(views);
    Ok(tables)
}

struct ByteLift {
    position: usize,
    weights: Vec<usize>,
    entries: Vec<F128>,
}

#[inline(always)]
fn byte_values(row: &[u64; 4], lifts: &[ByteLift]) -> [F128; 4] {
    let mut sums = Sums::default();
    for lift in lifts {
        let byte = ((row[(lift.position / 8) & 3] >> (8 * (lift.position % 8))) & 255) as usize;
        let entries = &lift.entries[byte * lift.weights.len()..(byte + 1) * lift.weights.len()];
        for (&weight, &value) in lift.weights.iter().zip(entries) {
            sums.add(weight, value);
        }
    }
    sums.values()
}

/// Build weighted tables from packed rows, reading each cycle once. A byte's
/// entries for all supported weight vectors are interleaved under one index.
/// Rejects non-power-of-two row counts and weights other than 256 columns.
pub fn g_pass_bytes(
    rows: &[[u64; 4]],
    weights: &[Vec<F128>],
) -> Result<Vec<Vec<F128>>, ReductionError> {
    check_weights(weights)?;
    if !rows.len().is_power_of_two() {
        return Err(ReductionError::TableLength {
            table: 0,
            actual: rows.len(),
            expected: None,
        });
    }
    let mut lifts = Vec::new();
    for position in 0..32 {
        let supported: Vec<_> = weights
            .iter()
            .enumerate()
            .filter(|(_, w)| {
                w[position * 8..position * 8 + 8]
                    .iter()
                    .any(|&v| v != F128::from_raw(0))
            })
            .map(|(i, _)| i)
            .collect();
        if supported.is_empty() {
            continue;
        }
        let mut entries = vec![F128::from_raw(0); 256 * supported.len()];
        for byte in 1_usize..256 {
            let bit = byte.trailing_zeros() as usize;
            let prior = byte & (byte - 1);
            for (slot, &weight) in supported.iter().enumerate() {
                entries[byte * supported.len() + slot] =
                    entries[prior * supported.len() + slot] + weights[weight][position * 8 + bit];
            }
        }
        lifts.push(ByteLift {
            position,
            weights: supported,
            entries,
        });
    }
    let mut tables = vec![vec![F128::from_raw(0); rows.len()]; weights.len()];
    if weights.is_empty() {
        return Ok(tables);
    }
    let chunk = CycleChunks::new(rows.len().ilog2() as usize, 0)
        .map_err(|_| ReductionError::TableLength {
            table: 0,
            actual: rows.len(),
            expected: None,
        })?
        .chunk_len();
    let mut views = output_views(&mut tables, chunk);
    views
        .par_chunks_mut(weights.len())
        .zip(rows.par_chunks(chunk))
        .for_each(|(outputs, rows)| {
            if let [(_, a), (_, b), (_, c)] = outputs {
                for (((a, b), c), row) in a.iter_mut().zip(b.iter_mut()).zip(c.iter_mut()).zip(rows)
                {
                    let sums = byte_values(row, &lifts);
                    *a = sums[0];
                    *b = sums[1];
                    *c = sums[2];
                }
                return;
            }

            for (cycle, row) in rows.iter().enumerate() {
                if weights.len() <= 4 {
                    let sums = byte_values(row, &lifts);
                    for ((_, output), value) in outputs.iter_mut().zip(sums) {
                        output[cycle] = value;
                    }
                    continue;
                }

                for lift in &lifts {
                    let byte =
                        ((row[lift.position / 8] >> (8 * (lift.position % 8))) & 255) as usize;
                    let entries =
                        &lift.entries[byte * lift.weights.len()..(byte + 1) * lift.weights.len()];
                    for (&weight, &value) in lift.weights.iter().zip(entries) {
                        outputs[weight].1[cycle] += value;
                    }
                }
            }
        });
    drop(views);
    Ok(tables)
}

struct Leg {
    table: usize,
    point: Vec<F128>,
    coefficient: F128,
    quotient: F128,
    claim: F128,
    eq: GruenSplitEqPolynomial<F128>,
    q: [F128; 2],
}

enum State {
    Round(usize),
    LastBind,
    Finished(Vec<F128>),
    Failed,
}

/// Degree-two reduction of weighted equality claims, with one bind per table
/// even when several legs share it. Equality and challenges are low-variable-first.
pub struct ReductionCore {
    tables: Vec<Vec<F128>>,
    scratch: Vec<Vec<F128>>,
    legs: Vec<Leg>,
    partials: Vec<F128Accumulator>,
    rounds: usize,
    state: State,
}

impl ReductionCore {
    /// A leg is `(table_index, point, coefficient, claim)`; all tables have the
    /// same nonzero power-of-two length and points have its logarithm coordinates.
    /// Each claim must equal `sum_j eq(point,j)*table[j]`: this is required of the
    /// caller, not checked here, and a violation is detected by the verifier.
    pub fn new(
        tables: Vec<Vec<F128>>,
        legs: Vec<(usize, Vec<F128>, F128, F128)>,
    ) -> Result<Self, ReductionError> {
        let length = tables.first().map_or(0, Vec::len);
        for (table, values) in tables.iter().enumerate() {
            if !values.len().is_power_of_two() || values.len() != length {
                return Err(ReductionError::TableLength {
                    table,
                    actual: values.len(),
                    expected: Some(length),
                });
            }
        }
        if length == 0 {
            return Err(ReductionError::TableLength {
                table: 0,
                actual: 0,
                expected: None,
            });
        }
        let rounds = length.ilog2() as usize;
        let mut checked = Vec::with_capacity(legs.len());
        for (leg, (table, point, coefficient, claim)) in legs.into_iter().enumerate() {
            if table >= tables.len() {
                return Err(ReductionError::LegTable {
                    leg,
                    table,
                    tables: tables.len(),
                });
            }
            if point.len() != rounds {
                return Err(ReductionError::LegPoint {
                    leg,
                    actual: point.len(),
                    expected: rounds,
                });
            }
            let eq = split_eq(&point, None)?;
            checked.push(Leg {
                table,
                point,
                coefficient,
                quotient: claim,
                claim,
                eq,
                q: [F128::from_raw(0); 2],
            });
        }
        if rounds == 0 {
            return Err(ReductionError::Round(RoundError::EmptyPoint));
        }
        let scratch = tables
            .iter()
            .map(|_| vec![F128::from_raw(0); length / 2])
            .collect();
        let chunks = length
            / CycleChunks::new(rounds, 0)
                .map_err(|_| ReductionError::TableLength {
                    table: 0,
                    actual: length,
                    expected: None,
                })?
                .chunk_len();
        let partials = vec![F128Accumulator::default(); 2 * chunks * checked.len()];
        Ok(Self {
            tables,
            scratch,
            legs: checked,
            partials,
            rounds,
            state: State::Round(0),
        })
    }

    /// Returns one table extension after `finish_rounds`, dropping all dense
    /// tables and second buffers before the finished state becomes observable.
    pub fn final_values(&self) -> Result<&[F128], ReductionError> {
        match &self.state {
            State::Finished(values) => Ok(values),
            _ => Err(ReductionError::Unfinished),
        }
    }

    /// Before any round, diagnose the first leg whose supplied claim disagrees
    /// with its defining equality sum. This optional scan is outside proving.
    pub fn check_claims(&self) -> Result<(), ReductionError> {
        if !matches!(self.state, State::Round(0)) {
            return Err(ReductionError::Unfinished);
        }
        for (leg, value) in self.legs.iter().enumerate() {
            let actual = eq_table(&value.point, None)
                .iter()
                .zip(&self.tables[value.table])
                .fold(F128::from_raw(0), |sum, (&eq, &g)| sum + eq * g);
            if actual != value.claim {
                return Err(ReductionError::Claim {
                    leg,
                    expected: value.claim,
                    actual,
                });
            }
        }
        Ok(())
    }

    fn round_sums(&mut self, bind: Option<F128>, round: usize) {
        let length = 1_usize << (self.rounds - round);
        let chunk = CycleChunks::new(self.rounds, round).map_or(length, CycleChunks::chunk_len);
        let count = length / chunk;
        let tables = self.tables.len();
        let leg_count = self.legs.len();
        let legs = &self.legs;
        let mut views: Vec<_> = if bind.is_some() {
            for scratch in &mut self.scratch {
                scratch.truncate(length);
            }
            self.tables
                .iter()
                .zip(&mut self.scratch)
                .flat_map(|(input, output)| {
                    input
                        .chunks(2 * chunk)
                        .zip(output.chunks_mut(chunk))
                        .enumerate()
                        .map(|(index, (input, output))| (index, input, output))
                })
                .collect()
        } else {
            self.tables
                .iter_mut()
                .flat_map(|input| {
                    input
                        .chunks_mut(chunk)
                        .enumerate()
                        .map(|(index, output)| (index, &[][..], output))
                })
                .collect()
        };
        views.sort_by_key(|&(index, _, _)| index);
        if !legs.is_empty() {
            views
                .par_chunks_mut(tables)
                .zip(self.partials[..2 * count * leg_count].par_chunks_mut(2 * leg_count))
                .enumerate()
                .for_each(|(index, (views, partials))| {
                    if leg_count <= 4 {
                        Self::accumulate_chunk::<4>(views, legs, partials, index, chunk, bind);
                    } else {
                        Self::accumulate_large_chunk(views, legs, partials, index, chunk, bind);
                    }
                });
        } else if let Some(challenge) = bind {
            views.par_chunks_mut(tables).for_each(|views| {
                for (_, input, output) in views {
                    for (dest, pair) in output.iter_mut().zip(input.chunks_exact(2)) {
                        *dest = pair[0] + challenge * (pair[0] + pair[1]);
                    }
                }
            });
        }
        drop(views);
        if bind.is_some() {
            std::mem::swap(&mut self.tables, &mut self.scratch);
        }
        for (index, leg) in self.legs.iter_mut().enumerate() {
            let mut sum = F128Accumulator::default();
            for chunk in self.partials[..2 * count * leg_count].chunks_exact(2 * leg_count) {
                sum.merge(chunk[index]);
            }
            let b = sum.reduce();
            leg.q = [leg.quotient + leg.point[round] * b, b];
        }
    }

    fn accumulate_chunk<const N: usize>(
        views: &mut [(usize, &[F128], &mut [F128])],
        legs: &[Leg],
        partials: &mut [F128Accumulator],
        index: usize,
        chunk: usize,
        bind: Option<F128>,
    ) {
        let mut total = [F128Accumulator::default(); N];
        let mut sums = [F128Accumulator::default(); N];
        Self::accumulate_blocks(
            views,
            legs,
            (&mut total[..legs.len()], &mut sums[..legs.len()]),
            (index, chunk),
            bind,
        );
        partials[..legs.len()].copy_from_slice(&total[..legs.len()]);
    }

    fn accumulate_large_chunk(
        views: &mut [(usize, &[F128], &mut [F128])],
        legs: &[Leg],
        partials: &mut [F128Accumulator],
        index: usize,
        chunk: usize,
        bind: Option<F128>,
    ) {
        let (total, sums) = partials.split_at_mut(legs.len());
        total.fill(F128Accumulator::default());
        Self::accumulate_blocks(views, legs, (total, sums), (index, chunk), bind);
    }

    fn accumulate_blocks(
        views: &mut [(usize, &[F128], &mut [F128])],
        legs: &[Leg],
        (total, sums): (&mut [F128Accumulator], &mut [F128Accumulator]),
        (index, chunk): (usize, usize),
        bind: Option<F128>,
    ) {
        let inner_len = legs[0].eq.e_in_current_len();
        let block_len = 2 * inner_len;
        let first_block = index * (chunk / 2) / inner_len;
        for block in 0..chunk / block_len {
            sums.fill(F128Accumulator::default());
            let start = block * block_len;
            for (table, (_, input, output)) in views.iter_mut().enumerate() {
                let output = &mut output[start..start + block_len];
                if let Some(challenge) = bind {
                    let input = &input[2 * start..2 * (start + block_len)];
                    for (pair_index, (dest, source)) in output
                        .chunks_exact_mut(2)
                        .zip(input.chunks_exact(4))
                        .enumerate()
                    {
                        dest[0] = source[0] + challenge * (source[0] + source[1]);
                        dest[1] = source[2] + challenge * (source[2] + source[3]);
                        let delta = dest[0] + dest[1];
                        for (slot, leg) in legs
                            .iter()
                            .enumerate()
                            .filter(|(_, leg)| leg.table == table)
                        {
                            sums[slot].fmadd(leg.eq.e_in_current()[pair_index], delta);
                        }
                    }
                } else {
                    for (pair_index, pair) in output.chunks_exact(2).enumerate() {
                        let delta = pair[0] + pair[1];
                        for (slot, leg) in legs
                            .iter()
                            .enumerate()
                            .filter(|(_, leg)| leg.table == table)
                        {
                            sums[slot].fmadd(leg.eq.e_in_current()[pair_index], delta);
                        }
                    }
                }
            }
            for (slot, leg) in legs.iter().enumerate() {
                total[slot].fmadd(
                    leg.eq.e_out_current()[first_block + block],
                    sums[slot].reduce(),
                );
            }
        }
    }

    fn bind_legs(&mut self, challenge: F128) {
        for leg in &mut self.legs {
            leg.quotient = leg.q[0] + challenge * leg.q[1];
            leg.eq.bind(challenge);
        }
    }
}

impl ProveRounds<F128> for ReductionCore {
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        previous_claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        let State::Round(expected) = self.state else {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: self.rounds,
                got: round,
            });
        };
        if round != expected {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected,
                got: round,
            });
        }
        if (round == 0) != bind.is_none() {
            return Err(SumcheckError::MissingEvaluationSource {
                kind: "reduction previous challenge",
            });
        }
        self.state = State::Failed;
        if let Some(challenge) = bind {
            self.bind_legs(challenge);
        }
        self.round_sums(bind, round);
        let mut coefficients = [F128::from_raw(0); 3];
        for leg in &self.legs {
            let message = leg.eq.round_poly_from_q_coeffs(&leg.q);
            for (dest, &coefficient) in coefficients.iter_mut().zip(message.coefficients()) {
                *dest += leg.coefficient * coefficient;
            }
        }
        let actual = coefficients[1] + coefficients[2];
        if actual != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual,
            });
        }
        self.state = if round + 1 == self.rounds {
            State::LastBind
        } else {
            State::Round(round + 1)
        };
        Ok(UnivariatePoly::new(coefficients.to_vec()))
    }
    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        if !matches!(self.state, State::LastBind) {
            return Err(SumcheckError::MissingEvaluationSource {
                kind: "reduction final challenge",
            });
        }
        self.bind_legs(bind);
        let values = self
            .tables
            .iter()
            .map(|table| table[0] + bind * (table[0] + table[1]))
            .collect();
        self.state = State::Finished(values);
        self.tables.clear();
        self.scratch.clear();
        self.legs.clear();
        self.partials.clear();
        self.tables.shrink_to_fit();
        self.scratch.shrink_to_fit();
        self.legs.shrink_to_fit();
        self.partials.shrink_to_fit();
        Ok(())
    }
}

#[cfg(all(test, feature = "test-utils"))]
#[expect(clippy::unwrap_used, reason = "test fixtures fail by panicking")]
mod tests {
    use super::ReductionCore;
    use crate::oracle::{mle_at, round_polynomial};
    use crate::synth::{SynthProfile, SyntheticTrace};
    use jolt_field::{Field, F128};
    use jolt_sumcheck::ProveRounds;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn eq(point: &[F128], vertex: usize) -> F128 {
        point
            .iter()
            .enumerate()
            .map(|(i, &t)| F128::from_raw(1) + t + F128::from_raw(((vertex >> i) & 1) as u128))
            .product()
    }

    fn quotient(table: &[F128], bound: &[F128], suffix: &[F128]) -> F128 {
        (0..1 << suffix.len())
            .map(|vertex| {
                let point: Vec<_> = bound
                    .iter()
                    .copied()
                    .chain((0..suffix.len()).map(|i| F128::from_raw(((vertex >> i) & 1) as u128)))
                    .collect();
                eq(suffix, vertex) * mle_at(table, &point).unwrap()
            })
            .sum()
    }

    #[test]
    fn zero_scalar_preserves_nonzero_quotient_and_shared_table_binding() {
        let rounds = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(7619);
        let trace = SyntheticTrace::new(SynthProfile::Local, rounds, 4, 98).unwrap();
        let weights: Vec<_> = (0..64).map(|_| F128::random(&mut rng)).collect();
        let table: Vec<_> = trace
            .rows()
            .iter()
            .map(|row| {
                weights
                    .iter()
                    .enumerate()
                    .filter(|(bit, _)| row[0] & (1 << bit) != 0)
                    .fold(F128::from_raw(0), |sum, (_, &w)| sum + w)
            })
            .collect();
        let points = [
            vec![
                F128::from_raw(0),
                F128::from_raw(1),
                F128::from_raw(91),
                F128::from_raw(117),
            ],
            vec![
                F128::from_raw(43),
                F128::from_raw(67),
                F128::from_raw(109),
                F128::from_raw(139),
            ],
        ];
        let challenges = [
            F128::from_raw(1),
            F128::from_raw(151),
            F128::from_raw(173),
            F128::from_raw(191),
        ];
        for count in [1, 2] {
            let legs: Vec<_> = points[..count]
                .iter()
                .enumerate()
                .map(|(i, point)| {
                    (
                        0,
                        point.clone(),
                        F128::from_raw(37 + i as u128),
                        quotient(&table, &[], point),
                    )
                })
                .collect();
            let eq_tables: Vec<Vec<_>> = points[..count]
                .iter()
                .map(|t| (0..table.len()).map(|j| eq(t, j)).collect())
                .collect();
            let leaves: Vec<_> = std::iter::once(table.as_slice())
                .chain(eq_tables.iter().map(Vec::as_slice))
                .collect();
            let mut claim: F128 = legs.iter().map(|(_, _, k, c)| *k * c).sum();
            let mut core = ReductionCore::new(vec![table.clone()], legs.clone()).unwrap();
            for round in 0..rounds {
                let message = core
                    .prove_round(
                        if round == 0 {
                            None
                        } else {
                            Some(challenges[round - 1])
                        },
                        round,
                        claim,
                    )
                    .unwrap();
                let expected = round_polynomial(&leaves, &challenges[..round], 2, |v| {
                    legs.iter()
                        .enumerate()
                        .map(|(i, (_, _, k, _))| *k * v[0] * v[i + 1])
                        .sum()
                })
                .unwrap();
                assert_eq!(message, expected);
                for (i, leg) in core.legs.iter().enumerate() {
                    assert_eq!(
                        leg.quotient,
                        quotient(&table, &challenges[..round], &points[i][round..])
                    );
                    let expected = quotient(&table, &challenges[..=round], &points[i][round + 1..]);
                    assert_eq!(leg.q[0] + challenges[round] * leg.q[1], expected);
                    if i == 0 {
                        assert_ne!(expected, F128::from_raw(0));
                    }
                    if round > 0 {
                        assert_eq!(leg.eq.current_scalar() == F128::from_raw(0), i == 0);
                    }
                }
                claim = message.evaluate(challenges[round]);
            }
            core.finish_rounds(challenges[rounds - 1]).unwrap();
            assert_eq!(
                core.final_values().unwrap(),
                [mle_at(&table, &challenges).unwrap()]
            );
        }
    }
}
