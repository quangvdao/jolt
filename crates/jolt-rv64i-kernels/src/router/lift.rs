//! Source lifting sums out bit and word variables and retains shared trace-word lifts.

use std::time::{Duration, Instant};

use jolt_field::{Accumulator, F128Accumulator, F128};
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;

use super::shape::{table_len, BitEntry, RouterError, RouterShape, WordSlot};
use crate::packed::lift::WordLift;
use crate::par::CycleChunks;
use crate::round::eq::eq_table;
use crate::source::{CycleSource, ValidatedTrace};

const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);
const TILE: usize = 32;
const SHAPES_PER_TILE: usize = 8;

/// Each distinct trace word used by a bank, lifted once at the common bit point.
/// Tables are in `word_indices()` order and retain their full cycle domain.
/// Agreement with the source subsequently passed to `claims_pass` is required
/// of the caller, not checked, and incorrect claims are detected by the verifier.
pub struct RetainedWordLifts {
    pub(crate) lift: WordLift,
    pub(crate) words: Vec<usize>,
    pub(crate) tables: Vec<Vec<F128>>,
}

impl RetainedWordLifts {
    /// Source word indices, sorted increasingly without repetitions.
    pub fn word_indices(&self) -> &[usize] {
        &self.words
    }

    /// The complete lifted cycle tables, in `word_indices()` order.
    pub fn tables(&self) -> &[Vec<F128>] {
        &self.tables
    }
}

/// Source tables in shape order and retained trace-word tables for claims.
/// Row tables are temporary and have been dropped before this value is returned.
pub struct SourceLiftOutput {
    pub source_tables: Vec<Vec<F128>>,
    pub lifts: RetainedWordLifts,
    pub row_tables_time: Duration,
    pub cycles_time: Duration,
}

#[derive(Clone, Copy)]
struct Term {
    shape: usize,
    coefficient: F128,
}
struct WordPlan {
    word: usize,
    terms: Vec<Term>,
}
struct DigitPlan {
    column: usize,
    table: Vec<F128>,
}
struct FlagEntry {
    column: usize,
    coefficient: F128,
}
struct Flags<const N: usize, const SIZE: usize> {
    columns: [usize; N],
    table: [F128; SIZE],
}
impl<const N: usize, const SIZE: usize> Flags<N, SIZE> {
    fn new(entries: &[FlagEntry; N]) -> Self {
        assert_eq!(SIZE, 1 << N);
        Self {
            columns: std::array::from_fn(|index| entries[index].column),
            table: std::array::from_fn(|index| {
                entries
                    .iter()
                    .enumerate()
                    .filter(|&(bit, _)| index & (1 << bit) != 0)
                    .fold(ZERO, |sum, (_, entry)| sum + entry.coefficient)
            }),
        }
    }
    fn add<S: CycleSource>(&self, source: &S, start: usize, output: &mut [F128]) {
        for (cycle, output) in output.iter_mut().enumerate() {
            let mut index = 0;
            for (bit, &column) in self.columns.iter().enumerate() {
                index |= usize::from(source.digit(column, start + cycle).is_some()) << bit;
            }
            *output += self.table[index & (SIZE - 1)];
        }
    }
}
struct ShapePlan {
    constant: F128,
    trace_terms: usize,
    bytecode_terms: usize,
    row_table: Option<usize>,
    direct_trace: bool,
    digits: Vec<DigitPlan>,
    flags_one: Vec<Flags<1, 2>>,
    flags_two: Vec<Flags<2, 4>>,
    flags_three: Vec<Flags<3, 8>>,
}
struct Plan {
    shapes: Vec<ShapePlan>,
    trace: Vec<WordPlan>,
    bytecode: Vec<WordPlan>,
}

impl Plan {
    fn add_word(words: &mut Vec<WordPlan>, word: usize, shape: usize, coefficient: F128) {
        if coefficient == ZERO {
            return;
        }
        if let Some(plan) = words.iter_mut().find(|plan| plan.word == word) {
            if let Some(term) = plan.terms.iter_mut().find(|term| term.shape == shape) {
                term.coefficient += coefficient;
            } else {
                plan.terms.push(Term { shape, coefficient });
            }
        } else {
            words.push(WordPlan {
                word,
                terms: vec![Term { shape, coefficient }],
            });
        }
    }

    fn compile<S: CycleSource>(
        source: &S,
        shapes: &[RouterShape],
        point: &[F128],
        bits: &[F128],
    ) -> Result<Self, RouterError> {
        let mut plan = Self {
            shapes: Vec::with_capacity(shapes.len()),
            trace: Vec::new(),
            bytecode: Vec::new(),
        };
        let mut row_count = 0;
        for (shape_index, shape) in shapes.iter().enumerate() {
            shape.check_source(source)?;
            if point.len() != shape.slots() {
                return Err(RouterError::PointLength {
                    expected: shape.slots(),
                    actual: point.len(),
                });
            }
            let word_point: Vec<_> = shape.word_slots().iter().map(|&slot| point[slot]).collect();
            let weights = eq_table(&word_point, None);
            let mut request = ShapePlan {
                constant: ZERO,
                trace_terms: 0,
                bytecode_terms: 0,
                row_table: None,
                direct_trace: false,
                digits: Vec::new(),
                flags_one: Vec::new(),
                flags_two: Vec::new(),
                flags_three: Vec::new(),
            };
            let mut flags: Vec<FlagEntry> = Vec::new();
            for (word, &weight) in shape.bank().iter().zip(&weights) {
                match word {
                    WordSlot::Trace(word) => {
                        Self::add_word(&mut plan.trace, *word, shape_index, weight);
                    }
                    WordSlot::Bytecode(word) => {
                        if request.row_table.is_none() {
                            request.row_table = Some(row_count);
                            row_count += 1;
                        }
                        Self::add_word(&mut plan.bytecode, *word, shape_index, weight);
                    }
                    WordSlot::Bits(entries) => {
                        for (entry, &bit_weight) in entries.iter().zip(bits) {
                            let coefficient = weight * bit_weight;
                            match *entry {
                                BitEntry::One => request.constant += coefficient,
                                BitEntry::Zero => {}
                                BitEntry::Indicator { column, value }
                                    if source.bits(column) == 0 =>
                                {
                                    if let Some(flag) =
                                        flags.iter_mut().find(|flag| flag.column == column)
                                    {
                                        flag.coefficient += coefficient;
                                    } else {
                                        flags.push(FlagEntry {
                                            column,
                                            coefficient,
                                        });
                                    }
                                    debug_assert_eq!(value, 0);
                                }
                                BitEntry::Indicator { column, value } => {
                                    let digit = Self::digit(&mut request, source, column)?;
                                    digit.table[value + 1] += coefficient;
                                }
                                BitEntry::DigitBit { column, bit } => {
                                    let digit = Self::digit(&mut request, source, column)?;
                                    for (value, entry) in digit.table[1..].iter_mut().enumerate() {
                                        if value & (1 << bit) != 0 {
                                            *entry += coefficient;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    WordSlot::Zero => {}
                }
            }
            for flags in flags.chunks(3) {
                match flags {
                    [a] => request.flags_one.push(Flags::new(&[FlagEntry {
                        column: a.column,
                        coefficient: a.coefficient,
                    }])),
                    [a, b] => request.flags_two.push(Flags::new(&[
                        FlagEntry {
                            column: a.column,
                            coefficient: a.coefficient,
                        },
                        FlagEntry {
                            column: b.column,
                            coefficient: b.coefficient,
                        },
                    ])),
                    [a, b, c] => request.flags_three.push(Flags::new(&[
                        FlagEntry {
                            column: a.column,
                            coefficient: a.coefficient,
                        },
                        FlagEntry {
                            column: b.column,
                            coefficient: b.coefficient,
                        },
                        FlagEntry {
                            column: c.column,
                            coefficient: c.coefficient,
                        },
                    ])),
                    _ => {}
                }
            }
            plan.shapes.push(request);
        }
        // Words with a zero weight still need their lift retained for claims.
        for shape in shapes {
            for word in shape.bank() {
                if let WordSlot::Trace(word) = word {
                    if !plan.trace.iter().any(|plan| plan.word == *word) {
                        plan.trace.push(WordPlan {
                            word: *word,
                            terms: Vec::new(),
                        });
                    }
                }
            }
        }
        plan.trace.sort_unstable_by_key(|word| word.word);
        plan.bytecode.sort_unstable_by_key(|word| word.word);
        for word in &mut plan.trace {
            word.terms.retain(|term| term.coefficient != ZERO);
            for term in &word.terms {
                plan.shapes[term.shape].trace_terms += 1;
            }
        }
        for word in &mut plan.bytecode {
            word.terms.retain(|term| term.coefficient != ZERO);
            for term in &word.terms {
                plan.shapes[term.shape].bytecode_terms += 1;
            }
        }
        for (index, shape) in plan.shapes.iter_mut().enumerate() {
            shape.direct_trace = shape.trace_terms == 1
                && plan.trace.iter().any(|word| {
                    word.terms
                        .iter()
                        .any(|term| term.shape == index && term.coefficient == ONE)
                });
        }
        Ok(plan)
    }

    fn digit<'a, S: CycleSource>(
        shape: &'a mut ShapePlan,
        source: &S,
        column: usize,
    ) -> Result<&'a mut DigitPlan, RouterError> {
        if let Some(index) = shape.digits.iter().position(|digit| digit.column == column) {
            return Ok(&mut shape.digits[index]);
        }
        let len = table_len(source.bits(column))?;
        shape.digits.push(DigitPlan {
            column,
            // Zero is absence; a present digit d reads entry d + 1.
            table: vec![ZERO; len + 1],
        });
        let index = shape.digits.len() - 1;
        Ok(&mut shape.digits[index])
    }

    fn rows<S: CycleSource>(
        &self,
        source: &S,
        lift: &WordLift,
        rows: &mut [Vec<F128>],
        chunk: usize,
    ) {
        let row_count = rows.len();
        let mut targets = chunk_views(rows, chunk);
        let shapes = self.shapes.len();
        let row_shapes: Vec<_> = self
            .shapes
            .iter()
            .enumerate()
            .filter_map(|(index, shape)| shape.row_table.map(|row| (index, row)))
            .collect();
        targets
            .par_chunks_mut(row_count)
            .enumerate()
            .for_each(|(index, outputs)| {
                let start = index * chunk;
                let len = outputs[0].len();
                for offset in (0..len).step_by(TILE) {
                    let count = TILE.min(len - offset);
                    for base in (0..shapes).step_by(SHAPES_PER_TILE) {
                        let group = SHAPES_PER_TILE.min(shapes - base);
                        let mut sums = [[F128Accumulator::default(); TILE]; SHAPES_PER_TILE];
                        for word in &self.bytecode {
                            let mut values = [ZERO; TILE];
                            for (row, value) in values[..count].iter_mut().enumerate() {
                                *value = lift
                                    .lift(source.bytecode_word(word.word, start + offset + row));
                            }
                            for term in word
                                .terms
                                .iter()
                                .filter(|term| term.shape >= base && term.shape < base + group)
                            {
                                let sums = &mut sums[term.shape - base];
                                for (sum, &value) in sums[..count].iter_mut().zip(&values) {
                                    sum.fmadd(value, term.coefficient);
                                }
                            }
                        }
                        for &(shape_index, row) in row_shapes
                            .iter()
                            .filter(|&&(shape, _)| shape >= base && shape < base + group)
                        {
                            let shape = &self.shapes[shape_index];
                            for (output, sum) in outputs[row][offset..offset + count]
                                .iter_mut()
                                .zip(sums[shape_index - base].iter().copied())
                            {
                                *output = shape.constant
                                    + if shape.bytecode_terms == 0 {
                                        ZERO
                                    } else {
                                        sum.reduce()
                                    };
                            }
                        }
                    }
                }
            });
    }

    fn cycles<S: CycleSource>(
        &self,
        source: &S,
        lift: &WordLift,
        rows: &[Vec<F128>],
        lifts: &mut [Vec<F128>],
        tables: &mut [Vec<F128>],
        chunk: usize,
    ) {
        let word_count = lifts.len();
        let mut lift_views = chunk_views(lifts, chunk);
        let mut outputs = chunk_views(tables, chunk);
        let shape_count = self.shapes.len();
        let mut lift_chunks = lift_views.chunks_mut(word_count.max(1));
        let mut target_chunks = outputs.chunks_mut(shape_count);
        let chunks: Vec<_> = (0..source.cycles() / chunk)
            .map(|_| {
                let words = lift_chunks.next().unwrap_or_default();
                let output = target_chunks.next().unwrap_or_default();
                (words, output)
            })
            .collect();
        chunks
            .into_par_iter()
            .enumerate()
            .for_each(|(index, (words, outputs))| {
                let start = index * chunk;
                let len = outputs[0].len();
                for offset in (0..len).step_by(TILE) {
                    let count = TILE.min(len - offset);
                    for (word, table) in self.trace.iter().zip(words.iter_mut()) {
                        for (cycle, value) in table[offset..offset + count].iter_mut().enumerate() {
                            *value =
                                lift.lift(source.trace_word(word.word, start + offset + cycle));
                        }
                    }
                    for (shape, output) in self.shapes.iter().zip(outputs.iter_mut()) {
                        output[offset..offset + count].fill(if shape.row_table.is_none() {
                            shape.constant
                        } else {
                            ZERO
                        });
                    }
                    for base in (0..shape_count).step_by(SHAPES_PER_TILE) {
                        let group = SHAPES_PER_TILE.min(shape_count - base);
                        let mut sums = [[F128Accumulator::default(); TILE]; SHAPES_PER_TILE];
                        for (word, values) in self.trace.iter().zip(words.iter()) {
                            for term in word
                                .terms
                                .iter()
                                .filter(|term| term.shape >= base && term.shape < base + group)
                            {
                                if self.shapes[term.shape].direct_trace {
                                    for (output, &value) in outputs[term.shape]
                                        [offset..offset + count]
                                        .iter_mut()
                                        .zip(&values[offset..offset + count])
                                    {
                                        *output += value;
                                    }
                                } else {
                                    for (sum, &value) in sums[term.shape - base][..count]
                                        .iter_mut()
                                        .zip(&values[offset..offset + count])
                                    {
                                        sum.fmadd(value, term.coefficient);
                                    }
                                }
                            }
                        }
                        for (local, shape) in self.shapes[base..base + group].iter().enumerate() {
                            let output = &mut outputs[base + local][offset..offset + count];
                            if shape.trace_terms != 0 && !shape.direct_trace {
                                for (output, sum) in
                                    output.iter_mut().zip(sums[local].iter().copied())
                                {
                                    *output += sum.reduce();
                                }
                            }
                            if let Some(row) = shape.row_table {
                                let row_table = &rows[row];
                                for (cycle, output) in output.iter_mut().enumerate() {
                                    *output +=
                                        row_table[source.bytecode_index(start + offset + cycle)];
                                }
                            }
                            for digit in &shape.digits {
                                for (cycle, output) in output.iter_mut().enumerate() {
                                    let value = source
                                        .digit(digit.column, start + offset + cycle)
                                        .map_or(0, |digit| digit + 1);
                                    *output += digit.table[value];
                                }
                            }
                            for flags in &shape.flags_one {
                                flags.add(source, start + offset, output);
                            }
                            for flags in &shape.flags_two {
                                flags.add(source, start + offset, output);
                            }
                            for flags in &shape.flags_three {
                                flags.add(source, start + offset, output);
                            }
                        }
                    }
                }
            });
    }
}

#[expect(
    clippy::expect_used,
    reason = "each equally sized table has exactly one slice per checked chunk"
)]
fn chunk_views(tables: &mut [Vec<F128>], chunk: usize) -> Vec<&mut [F128]> {
    let len = tables.first().map_or(0, Vec::len);
    let mut iterators: Vec<_> = tables
        .iter_mut()
        .map(|table| table.chunks_mut(chunk))
        .collect();
    let mut views = Vec::with_capacity(len.div_ceil(chunk) * iterators.len());
    for _ in 0..len.div_ceil(chunk) {
        views.extend(
            iterators
                .iter_mut()
                .map(|iterator| iterator.next().expect("equal complete chunks")),
        );
    }
    views
}

/// Build `Source_rho(x|src,j)` for every shape and retain each distinct trace
/// word's lift at `x[0..6]`. Checks all shape source references and point lengths.
/// Each shape combines its trace lifts by one unreduced sum, its digit lookup
/// tables, and a temporary bytecode-row table that also contains its constant.
/// Source immutability and the source's agreement with committed bits are
/// required of the caller, not checked, and false claims are detected by the verifier.
pub fn source_lift<S: CycleSource>(
    trace: &ValidatedTrace<S>,
    shapes: &[RouterShape],
    x: &[F128],
) -> Result<SourceLiftOutput, RouterError> {
    let source = trace.source().as_ref();
    let bit_point = x.get(..6).ok_or(RouterError::PointLength {
        expected: shapes.first().map_or(6, RouterShape::slots),
        actual: x.len(),
    })?;
    let bit_weights = eq_table(bit_point, None);
    let weights = std::array::from_fn(|bit| bit_weights[bit]);
    let lift = WordLift::new(&weights);
    let plan = Plan::compile(source, shapes, x, &bit_weights)?;
    let log_t = source.cycles().ilog2() as usize;
    let _ = table_len(log_t)?;
    let _ = table_len(source.bytecode_rows().ilog2() as usize)?;
    let geometry = CycleChunks::new(log_t, 0)?;
    let mut source_tables: Vec<Vec<F128>> = shapes
        .iter()
        .map(|_| unsafe_allocate_zero_vec(source.cycles()))
        .collect();
    let mut lifts: Vec<Vec<F128>> = plan
        .trace
        .iter()
        .map(|_| unsafe_allocate_zero_vec(source.cycles()))
        .collect();
    let row_start = Instant::now();
    let mut rows: Vec<Vec<F128>> = plan
        .shapes
        .iter()
        .filter(|shape| shape.row_table.is_some())
        .map(|_| unsafe_allocate_zero_vec(source.bytecode_rows()))
        .collect();
    if !rows.is_empty() {
        plan.rows(source, &lift, &mut rows, geometry.chunk_len());
    }
    let row_tables_time = row_start.elapsed();
    let cycles_start = Instant::now();
    if !shapes.is_empty() {
        plan.cycles(
            source,
            &lift,
            &rows,
            &mut lifts,
            &mut source_tables,
            geometry.chunk_len(),
        );
    }
    let cycles_time = cycles_start.elapsed();
    Ok(SourceLiftOutput {
        source_tables,
        lifts: RetainedWordLifts {
            lift,
            words: plan.trace.iter().map(|word| word.word).collect(),
            tables: lifts,
        },
        row_tables_time,
        cycles_time,
    })
}
