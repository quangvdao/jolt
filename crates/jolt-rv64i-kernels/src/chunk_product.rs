//! Cycle sum-check for `sum_j W[j] product_c eq(a_c, digit(c,j))`.
//! Points and challenges list the low index variable first.

use crate::par::CycleChunks;
use crate::round::eq::{eq_table, split_eq};
use crate::round::{
    coefficients_from_nodes, linear_at_nodes, quadratic, quadratic_at_nodes, RoundError,
};
use crate::source::{CycleSource, DigitColumns};
use jolt_field::{Accumulator, F128Accumulator, F128};
use jolt_kernels::optimized::lazy_ra::{ChunkIndexSource, LazyFoldedRa, LazyRaError};
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;
use std::alloc::Layout;
use thiserror::Error;

const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);

/// A dense cycle table or table-free equality terms, and the terms accepted by
/// [`combined_weight`]. `EqTerms` claims include their coefficients: each is
/// `sum_j coefficient * eq(point,j) * product_c Ra_c[j]`.
/// Their correctness is required of the caller, not checked, and detected by
/// the verifier at its final evaluation check. An empty list denotes zero.
#[derive(Clone, Debug)]
pub enum ChunkWeight {
    /// Exactly one field element per cycle.
    Dense(Vec<F128>),
    /// `(coefficient, low-variable-first cycle point, weighted claim)`.
    /// Honest weighted claims are required of the caller, not checked, and
    /// detected by the verifier at its final evaluation check.
    EqTerms(Vec<(F128, Vec<F128>, F128)>),
    /// An equality term for `combined_weight`, not a core weight.
    Eq { coefficient: F128, point: Vec<F128> },
    /// Equality at `j-1` for `j>0`, zero at cycle zero; for `combined_weight`.
    Next { coefficient: F128, point: Vec<F128> },
}

/// Malformed chunk geometry, weight terms or incomplete final evaluation.
#[derive(Debug, Error)]
pub enum ChunkProductError {
    #[error("{columns} columns supplied; expected 1..=7")]
    Columns { columns: usize },
    #[error("column {column} has {bits} bits; at most 8 are supported")]
    ColumnWidth { column: usize, bits: usize },
    #[error("{buffer} allocation has an unrepresentable element count or byte size")]
    AllocationSize { buffer: &'static str },
    #[error("expected {expected} chunk points, got {actual}")]
    PointCount { expected: usize, actual: usize },
    #[error("column {column} point has {actual} coordinates, expected {expected}")]
    PointLength {
        column: usize,
        expected: usize,
        actual: usize,
    },
    #[error("weight has {actual} entries, expected {expected}")]
    WeightLength { expected: usize, actual: usize },
    #[error("term {term} point has {actual} coordinates, expected {expected}")]
    TermPoint {
        term: usize,
        expected: usize,
        actual: usize,
    },
    #[error("column {column} has no digit at cycle {cycle}")]
    MissingDigit { column: usize, cycle: usize },
    #[error("term {term} has an unsupported weight kind")]
    TermKind { term: usize },
    #[error("cycle exponent {log_t} cannot size a field table")]
    LogSize { log_t: usize },
    #[error(transparent)]
    Round(#[from] RoundError),
    #[error(transparent)]
    LazyRa(#[from] LazyRaError),
    #[error("final values require all rounds and the last bind")]
    Unfinished,
}

struct HalfTerm {
    low: Vec<F128>,
    high: Vec<F128>,
    next: bool,
}

/// Builds `W[j] = sum_i coefficient_i * E_i(j)` with `Eq` and `Next` terms.
/// Rejects other variants and points of a length different from `log_t`.
/// Boolean points contribute at one index; other terms use two half tables,
/// unreduced products, and one reduction per cycle. Source-independent and
/// valid for an empty term list or a zero-dimensional cycle domain.
pub fn combined_weight(
    log_t: usize,
    terms: &[ChunkWeight],
) -> Result<Vec<F128>, ChunkProductError> {
    if log_t >= usize::BITS as usize - 5 {
        return Err(ChunkProductError::LogSize { log_t });
    }
    let low_bits = log_t.div_ceil(2);
    let mask = (1 << low_bits) - 1;
    let mut halves = Vec::new();
    let mut singletons = Vec::new();
    for (term, weight) in terms.iter().enumerate() {
        let (coefficient, point, next) = match weight {
            ChunkWeight::Eq { coefficient, point } => (*coefficient, point, false),
            ChunkWeight::Next { coefficient, point } => (*coefficient, point, true),
            ChunkWeight::Dense(_) | ChunkWeight::EqTerms(_) => {
                return Err(ChunkProductError::TermKind { term })
            }
        };
        if point.len() != log_t {
            return Err(ChunkProductError::TermPoint {
                term,
                expected: log_t,
                actual: point.len(),
            });
        }
        if point.iter().all(|&value| value == ZERO || value == ONE) {
            let index = point.iter().enumerate().fold(0, |index, (bit, &value)| {
                index | (usize::from(value == ONE) << bit)
            });
            if !next || index + 1 < 1 << log_t {
                singletons.push((index + usize::from(next), coefficient));
            }
        } else {
            halves.push(HalfTerm {
                low: eq_table(&point[..low_bits], None),
                high: eq_table(&point[low_bits..], Some(coefficient)),
                next,
            });
        }
    }
    let geometry = CycleChunks::new(log_t, 0).map_err(|_| ChunkProductError::LogSize { log_t })?;
    let mut output = vec![ZERO; geometry.len()];
    output
        .par_chunks_mut(geometry.chunk_len())
        .enumerate()
        .for_each(|(chunk, output)| {
            let start = chunk * geometry.chunk_len();
            for (offset, value) in output.iter_mut().enumerate() {
                let cycle = start + offset;
                let mut sum = F128Accumulator::default();
                for term in &halves {
                    if term.next && cycle == 0 {
                        continue;
                    }
                    let index = cycle - usize::from(term.next);
                    sum.fmadd(term.low[index & mask], term.high[index >> low_bits]);
                }
                *value = sum.reduce();
            }
        });
    for (index, coefficient) in singletons {
        output[index] += coefficient;
    }
    Ok(output)
}

struct EqTerm {
    eq: GruenSplitEqPolynomial<F128>,
    claim: F128,
    message: Option<UnivariatePoly<F128>>,
}

enum WeightState {
    Dense {
        table: Vec<F128>,
        scratch: Vec<F128>,
    },
    Terms {
        terms: Vec<EqTerm>,
        scratch: Vec<F128Accumulator>,
    },
}

enum State {
    Round(usize),
    LastBind,
    Finished((F128, Vec<F128>)),
    Failed,
}

fn checked_len<T>(
    elements: Option<usize>,
    buffer: &'static str,
) -> Result<usize, ChunkProductError> {
    let elements = elements.ok_or(ChunkProductError::AllocationSize { buffer })?;
    let _layout =
        Layout::array::<T>(elements).map_err(|_| ChunkProductError::AllocationSize { buffer })?;
    Ok(elements)
}

struct PresentDigits {
    digits: Vec<u8>,
    bits: [usize; 7],
    d: usize,
    cycles: usize,
}

impl PresentDigits {
    fn new<S: CycleSource>(
        columns: &DigitColumns<S>,
        bits: [usize; 7],
        geometry: CycleChunks,
    ) -> Result<Self, ChunkProductError> {
        let d = columns.num_polys();
        let cycles = columns.cycles();
        let len = checked_len::<u8>(cycles.checked_mul(d), "present digits")?;
        let mut digits = unsafe_allocate_zero_vec(len);
        let missing = digits
            .par_chunks_mut(geometry.chunk_len() * d)
            .enumerate()
            .find_map_first(|(chunk, output)| {
                let start = chunk * geometry.chunk_len();
                for (offset, row) in output.chunks_exact_mut(d).enumerate() {
                    let cycle = start + offset;
                    for (column, digit) in row.iter_mut().enumerate() {
                        match columns.index(column, cycle) {
                            Some(index) => *digit = index as u8,
                            None => return Some(ChunkProductError::MissingDigit { column, cycle }),
                        }
                    }
                }
                None
            });
        if let Some(error) = missing {
            return Err(error);
        }
        Ok(Self {
            digits,
            bits,
            d,
            cycles,
        })
    }
}

impl ChunkIndexSource for PresentDigits {
    fn num_polys(&self) -> usize {
        self.d
    }
    fn cycles(&self) -> usize {
        self.cycles
    }
    #[inline]
    fn index(&self, column: usize, cycle: usize) -> Option<usize> {
        Some(self.digits[cycle * self.d + column] as usize)
    }
    fn index_bound(&self, column: usize) -> Option<usize> {
        Some(1 << self.bits[column])
    }
}

/// Degree `d+1` cycle sum-check for one through seven digit columns.
/// The source must remain immutable under the `CycleSource` contract. All
/// digits are checked present at construction; range validation is owned by
/// `DigitColumns`. With `EqTerms`, honest individual weighted claims are
/// required of the caller, not checked, and detected by the verifier.
pub struct ChunkProductCore {
    columns: LazyFoldedRa<F128, PresentDigits>,
    weight: WeightState,
    log_t: usize,
    d: usize,
    state: State,
}

impl ChunkProductCore {
    /// Validates column/point counts, point dimensions, weight lengths and term
    /// dimensions before constructing the lazy family. `Eq` and `Next` are
    /// construction terms and are rejected here; first call `combined_weight`.
    pub fn new<S: CycleSource>(
        columns: DigitColumns<S>,
        points: Vec<Vec<F128>>,
        weight: ChunkWeight,
    ) -> Result<Self, ChunkProductError> {
        let d = columns.num_polys();
        if !(1..=7).contains(&d) {
            return Err(ChunkProductError::Columns { columns: d });
        }
        if points.len() != d {
            return Err(ChunkProductError::PointCount {
                expected: d,
                actual: points.len(),
            });
        }
        let mut bits = [0; 7];
        for (column, point) in points.iter().enumerate() {
            let expected = columns.source().bits(columns.columns()[column]);
            if expected > 8 {
                return Err(ChunkProductError::ColumnWidth {
                    column,
                    bits: expected,
                });
            }
            bits[column] = expected;
            if point.len() != expected {
                return Err(ChunkProductError::PointLength {
                    column,
                    expected,
                    actual: point.len(),
                });
            }
        }
        let cycles = columns.cycles();
        let log_t = cycles.ilog2() as usize;
        let weight = match weight {
            ChunkWeight::Dense(table) => {
                if table.len() != cycles {
                    return Err(ChunkProductError::WeightLength {
                        expected: cycles,
                        actual: table.len(),
                    });
                }
                WeightState::Dense {
                    table,
                    scratch: vec![ZERO; cycles / 2],
                }
            }
            ChunkWeight::EqTerms(input) => {
                let mut terms = Vec::with_capacity(input.len());
                for (term, (coefficient, point, claim)) in input.into_iter().enumerate() {
                    if point.len() != log_t {
                        return Err(ChunkProductError::TermPoint {
                            term,
                            expected: log_t,
                            actual: point.len(),
                        });
                    }
                    terms.push(EqTerm {
                        eq: split_eq(&point, Some(coefficient))?,
                        claim,
                        message: None,
                    });
                }
                let geometry =
                    CycleChunks::new(log_t, 0).map_err(|_| ChunkProductError::LogSize { log_t })?;
                let scratch = if matches!(terms.len(), 0 | 1 | 2 | 5) {
                    Vec::new()
                } else {
                    vec![
                        F128Accumulator::default();
                        checked_len::<F128Accumulator>(
                            geometry
                                .len()
                                .div_ceil(geometry.chunk_len())
                                .checked_mul(terms.len())
                                .and_then(|len| len.checked_mul(8)),
                            "equality term scratch",
                        )?
                    ]
                };
                WeightState::Terms { terms, scratch }
            }
            ChunkWeight::Eq { .. } | ChunkWeight::Next { .. } => {
                return Err(ChunkProductError::TermKind { term: 0 })
            }
        };
        let geometry =
            CycleChunks::new(log_t, 0).map_err(|_| ChunkProductError::LogSize { log_t })?;
        let present = PresentDigits::new(&columns, bits, geometry)?;
        let columns = LazyFoldedRa::try_new(
            points.iter().map(|point| eq_table(point, None)).collect(),
            present,
        )?;
        let state = if log_t == 0 {
            let w = match &weight {
                WeightState::Dense { table, .. } => table[0],
                WeightState::Terms { .. } => ZERO,
            };
            State::Finished((w, columns.final_values()))
        } else {
            State::Round(0)
        };
        Ok(Self {
            columns,
            weight,
            log_t,
            d,
            state,
        })
    }

    /// Returns `(W(r'), [Ra_0(r'), ...])` after the last challenge is bound.
    /// Earlier calls return `Unfinished`; the challenge point is low-first.
    pub fn final_values(&self) -> Result<(F128, Vec<F128>), ChunkProductError> {
        match &self.state {
            State::Finished(values) => Ok(values.clone()),
            _ => Err(ChunkProductError::Unfinished),
        }
    }

    fn dense_round<const D: usize, const N: usize>(
        &mut self,
        bind: Option<F128>,
        round: usize,
        claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        let WeightState::Dense { table, scratch } = &mut self.weight else {
            return Err(missing("dense chunk weight"));
        };
        let geometry =
            CycleChunks::new(self.log_t, round + 1).map_err(|_| missing("chunk round geometry"))?;
        let pairs = geometry.len();
        let chunk_pairs = geometry.chunk_len();
        let columns = &self.columns;
        let sums = if let Some(r) = bind {
            scratch.truncate(pairs * 2);
            let result = scratch
                .par_chunks_mut(chunk_pairs * 2)
                .zip(table.par_chunks(chunk_pairs * 4))
                .enumerate()
                .map(|(chunk, (output, input))| {
                    let mut sums = [F128Accumulator::default(); 8];
                    for (offset, (output, input)) in output
                        .chunks_exact_mut(2)
                        .zip(input.chunks_exact(4))
                        .enumerate()
                    {
                        let w0 = input[0] + r * (input[0] + input[1]);
                        let w1 = input[2] + r * (input[2] + input[3]);
                        output[0] = w0;
                        output[1] = w1;
                        let mut factors = [[ZERO; 2]; 8];
                        factors[0] = [w0, w0 + w1];
                        let mut values = [(ZERO, ZERO); D];
                        columns.lo_hi_all(chunk * chunk_pairs + offset, &mut values);
                        for (factor, (lo, hi)) in factors[1..=D].iter_mut().zip(values) {
                            *factor = [lo, lo + hi];
                        }
                        accumulate_product::<D, N>(&factors, &mut sums);
                    }
                    sums
                })
                .reduce(
                    || [F128Accumulator::default(); 8],
                    |left, right| merge_sums(left, &right),
                );
            std::mem::swap(table, scratch);
            result
        } else {
            table
                .par_chunks(chunk_pairs * 2)
                .enumerate()
                .map(|(chunk, input)| {
                    let mut sums = [F128Accumulator::default(); 8];
                    for (offset, input) in input.chunks_exact(2).enumerate() {
                        let mut factors = [[ZERO; 2]; 8];
                        factors[0] = [input[0], input[0] + input[1]];
                        let mut values = [(ZERO, ZERO); D];
                        columns.lo_hi_all(chunk * chunk_pairs + offset, &mut values);
                        for (factor, (lo, hi)) in factors[1..=D].iter_mut().zip(values) {
                            *factor = [lo, lo + hi];
                        }
                        accumulate_product::<D, N>(&factors, &mut sums);
                    }
                    sums
                })
                .reduce(
                    || [F128Accumulator::default(); 8],
                    |left, right| merge_sums(left, &right),
                )
        };
        let sums = sums.map(Accumulator::reduce);
        let coefficients =
            coefficients_from_nodes(D + 1, sums[0], sums[1], claim + sums[0], &sums[2..2 + N])
                .map_err(|_| missing("chunk interpolation"))?;
        Ok(UnivariatePoly::new(coefficients))
    }

    fn terms_round<const D: usize, const N: usize>(
        &mut self,
        round: usize,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        let WeightState::Terms { terms, scratch } = &mut self.weight else {
            return Err(missing("chunk equality terms"));
        };
        if terms.is_empty() {
            return Ok(UnivariatePoly::new(vec![ZERO; D + 2]));
        }
        let geometry =
            CycleChunks::new(self.log_t, round + 1).map_err(|_| missing("chunk round geometry"))?;
        let chunk_pairs = geometry.chunk_len();
        let columns = &self.columns;
        let stride = terms.len() * 8;
        let reduced = match terms.len() {
            1 => stack_term_sums::<D, N, 1>(&self.columns, terms, geometry),
            2 => stack_term_sums::<D, N, 2>(&self.columns, terms, geometry),
            5 => stack_term_sums::<D, N, 5>(&self.columns, terms, geometry),
            _ => {
                scratch.truncate(geometry.len().div_ceil(chunk_pairs) * stride);
                scratch
                    .par_chunks_mut(stride)
                    .enumerate()
                    .for_each(|(chunk, output)| {
                        let mut q_values = [[ZERO; 8]; 128];
                        output.fill(F128Accumulator::default());
                        let start = chunk * chunk_pairs;
                        let block_len = terms[0].eq.e_in_current_len();
                        // Equality's outer factor is constant on the block, so
                        // reducing tiles separately bounds stack use independently of m.
                        for block_start in (start..start + chunk_pairs).step_by(block_len) {
                            for tile_start in (0..block_len).step_by(q_values.len()) {
                                let tile_len = (block_len - tile_start).min(q_values.len());
                                let tile = &mut q_values[..tile_len];
                                for (offset, q) in tile.iter_mut().enumerate() {
                                    let mut factors = [[ZERO; 2]; 8];
                                    let mut values = [(ZERO, ZERO); D];
                                    columns
                                        .lo_hi_all(block_start + tile_start + offset, &mut values);
                                    for (factor, (lo, hi)) in factors[..D].iter_mut().zip(values) {
                                        *factor = [lo, lo + hi];
                                    }
                                    let (left, right, groups) = product_points::<D, N>(&factors, D);
                                    *q = std::array::from_fn(|i| {
                                        if i < N + 2 && groups > 1 {
                                            left[i] * right[i]
                                        } else {
                                            left[i]
                                        }
                                    });
                                }
                                for (term, partial) in terms.iter().zip(output.chunks_exact_mut(8))
                                {
                                    let weights =
                                        &term.eq.e_in_current()[tile_start..tile_start + tile_len];
                                    let mut inner = [F128Accumulator::default(); 8];
                                    for (q, &weight) in tile.iter().zip(weights) {
                                        for i in 0..N + 2 {
                                            inner[i].fmadd(q[i], weight);
                                        }
                                    }
                                    let outer = term.eq.e_out_current()[block_start / block_len];
                                    let mut total = [F128Accumulator::default(); 8];
                                    for i in 0..N + 2 {
                                        total[i].fmadd(inner[i].reduce(), outer);
                                    }
                                    for (partial, total) in partial.iter_mut().zip(total) {
                                        partial.merge(total);
                                    }
                                }
                            }
                        }
                    });
                (0..terms.len())
                    .map(|index| {
                        let mut sums = [F128Accumulator::default(); 8];
                        for chunk in scratch.chunks_exact(stride) {
                            for (sum, &partial) in
                                sums.iter_mut().zip(&chunk[index * 8..index * 8 + 8])
                            {
                                sum.merge(partial);
                            }
                        }
                        sums.map(Accumulator::reduce)
                    })
                    .collect()
            }
        };
        let mut coefficients = vec![ZERO; D + 2];
        for (index, term) in terms.iter_mut().enumerate() {
            let sums = reduced[index];
            let q_coeffs = if D == 1 {
                vec![sums[0], sums[1]]
            } else {
                let at_one = term
                    .eq
                    .recover_q_one(sums[0], term.claim, || {
                        let inner = term.eq.e_in_current();
                        let outer = term.eq.e_out_current();
                        outer
                            .par_chunks(chunk_pairs / inner.len())
                            .enumerate()
                            .map(|(chunk, outer)| {
                                let mut total = F128Accumulator::default();
                                for (offset, &out) in outer.iter().enumerate() {
                                    let block = chunk * (chunk_pairs / inner.len()) + offset;
                                    let mut sum = F128Accumulator::default();
                                    for (offset, &weight) in inner.iter().enumerate() {
                                        let row = block * inner.len() + offset;
                                        let mut product = columns.value(0, 2 * row + 1);
                                        for column in 1..D {
                                            product *= columns.value(column, 2 * row + 1);
                                        }
                                        sum.fmadd(product, weight);
                                    }
                                    total.fmadd(sum.reduce(), out);
                                }
                                total
                            })
                            .reduce(F128Accumulator::default, |mut left, right| {
                                left.merge(right);
                                left
                            })
                            .reduce()
                    })
                    .map_err(|actual| SumcheckError::RoundCheckFailed {
                        round,
                        expected: term.claim,
                        actual,
                    })?;
                coefficients_from_nodes(D, sums[0], sums[1], at_one, &sums[2..2 + N])
                    .map_err(|_| missing("chunk term interpolation"))?
            };
            let message = term.eq.round_poly_from_q_coeffs(&q_coeffs);
            for (output, &value) in coefficients.iter_mut().zip(message.coefficients()) {
                *output += value;
            }
            term.message = Some(message);
        }
        Ok(UnivariatePoly::new(coefficients))
    }
}

fn stack_term_sums<const D: usize, const N: usize, const M: usize>(
    columns: &LazyFoldedRa<F128, PresentDigits>,
    terms: &[EqTerm],
    geometry: CycleChunks,
) -> Vec<[F128; 8]> {
    let block_len = terms[0].eq.e_in_current_len();
    let chunk_pairs = geometry.chunk_len();
    let sums = (0..geometry.len() / chunk_pairs)
        .into_par_iter()
        .map(|chunk| {
            let mut totals = [[F128Accumulator::default(); 8]; M];
            let start = chunk * chunk_pairs;
            for block_start in (start..start + chunk_pairs).step_by(block_len) {
                let mut inner = [[F128Accumulator::default(); 8]; M];
                for offset in 0..block_len {
                    let mut factors = [[ZERO; 2]; 8];
                    let mut values = [(ZERO, ZERO); D];
                    columns.lo_hi_all(block_start + offset, &mut values);
                    for (factor, (lo, hi)) in factors[..D].iter_mut().zip(values) {
                        *factor = [lo, lo + hi];
                    }
                    let (left, right, groups) = product_points::<D, N>(&factors, D);
                    let q: [F128; 8] = std::array::from_fn(|i| {
                        if i < N + 2 && groups > 1 {
                            left[i] * right[i]
                        } else {
                            left[i]
                        }
                    });
                    for (term, sums) in terms.iter().zip(&mut inner) {
                        let weight = term.eq.e_in_current()[offset];
                        for i in 0..N + 2 {
                            sums[i].fmadd(q[i], weight);
                        }
                    }
                }
                let block = block_start / block_len;
                for ((term, total), inner) in terms.iter().zip(&mut totals).zip(inner) {
                    let outer = term.eq.e_out_current()[block];
                    for i in 0..N + 2 {
                        total[i].fmadd(inner[i].reduce(), outer);
                    }
                }
            }
            totals
        })
        .reduce(
            || [[F128Accumulator::default(); 8]; M],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(&right) {
                    *left = merge_sums(*left, right);
                }
                left
            },
        );
    sums.into_iter()
        .map(|sum| sum.map(Accumulator::reduce))
        .collect()
}

fn missing(kind: &'static str) -> SumcheckError<F128> {
    SumcheckError::MissingEvaluationSource { kind }
}

fn merge_sums(
    mut left: [F128Accumulator; 8],
    right: &[F128Accumulator; 8],
) -> [F128Accumulator; 8] {
    for (left, right) in left.iter_mut().zip(right) {
        left.merge(*right);
    }
    left
}

#[inline(always)]
fn factor_points<const N: usize>(
    factors: &[[F128; 2]; 8],
    group: usize,
    count: usize,
) -> [F128; 8] {
    let mut points = [ZERO; 8];
    if 2 * group + 1 < count {
        let q = quadratic(factors[2 * group], factors[2 * group + 1]);
        points[0] = q[0];
        points[1] = q[2];
        points[2..2 + N].copy_from_slice(&quadratic_at_nodes(q)[..N]);
    } else {
        let linear = factors[2 * group];
        points[0] = linear[0];
        points[1] = linear[1];
        points[2..2 + N].copy_from_slice(&linear_at_nodes(linear)[..N]);
    }
    points
}

#[inline(always)]
fn product_points<const D: usize, const N: usize>(
    factors: &[[F128; 2]; 8],
    count: usize,
) -> ([F128; 8], [F128; 8], usize) {
    let groups = count.div_ceil(2);
    let mut left = factor_points::<N>(factors, 0, count);
    let mut right = [ZERO; 8];
    if groups > 1 {
        right = factor_points::<N>(factors, 1, count);
        if groups > 2 {
            for i in 0..N + 2 {
                left[i] *= right[i];
            }
            right = factor_points::<N>(factors, 2, count);
            if groups > 3 {
                for i in 0..N + 2 {
                    left[i] *= right[i];
                }
                right = factor_points::<N>(factors, 3, count);
            }
        }
    }
    (left, right, groups)
}

#[inline(always)]
fn accumulate_product<const D: usize, const N: usize>(
    factors: &[[F128; 2]; 8],
    sums: &mut [F128Accumulator; 8],
) {
    let (left, right, groups) = product_points::<D, N>(factors, D + 1);
    for i in 0..N + 2 {
        if groups == 1 {
            sums[i].add(left[i]);
        } else {
            sums[i].fmadd(left[i], right[i]);
        }
    }
}

impl ProveRounds<F128> for ChunkProductCore {
    fn num_rounds(&self) -> usize {
        self.log_t
    }

    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        previous_claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        let State::Round(expected) = self.state else {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: self.log_t,
                got: round,
            });
        };
        if expected != round {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected,
                got: round,
            });
        }
        if (round == 0) != bind.is_none() {
            return Err(missing("chunk previous challenge"));
        }
        self.state = State::Failed;
        if let Some(r) = bind {
            self.columns.bind(r);
            if let WeightState::Terms { terms, .. } = &mut self.weight {
                for term in terms {
                    if let Some(message) = &term.message {
                        term.claim = message.evaluate(r);
                    }
                    term.eq.bind(r);
                }
            }
        }
        let result = match &self.weight {
            WeightState::Dense { .. } => match self.d {
                1 => self.dense_round::<1, 0>(bind, round, previous_claim),
                2 => self.dense_round::<2, 1>(bind, round, previous_claim),
                3 => self.dense_round::<3, 2>(bind, round, previous_claim),
                4 => self.dense_round::<4, 3>(bind, round, previous_claim),
                5 => self.dense_round::<5, 4>(bind, round, previous_claim),
                6 => self.dense_round::<6, 5>(bind, round, previous_claim),
                _ => self.dense_round::<7, 6>(bind, round, previous_claim),
            },
            WeightState::Terms { .. } => match self.d {
                1 => self.terms_round::<1, 0>(round),
                2 => self.terms_round::<2, 0>(round),
                3 => self.terms_round::<3, 1>(round),
                4 => self.terms_round::<4, 2>(round),
                5 => self.terms_round::<5, 3>(round),
                6 => self.terms_round::<6, 4>(round),
                _ => self.terms_round::<7, 5>(round),
            },
        };
        if result.is_ok() {
            self.state = if round + 1 == self.log_t {
                State::LastBind
            } else {
                State::Round(round + 1)
            };
        }
        result
    }

    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        if !matches!(self.state, State::LastBind) {
            return Err(missing("chunk final challenge"));
        }
        self.columns.bind(bind);
        let weight = match &mut self.weight {
            WeightState::Dense { table, scratch } => {
                let value = table[0] + bind * (table[0] + table[1]);
                drop(std::mem::take(table));
                drop(std::mem::take(scratch));
                value
            }
            WeightState::Terms { terms, scratch } => {
                let value = terms
                    .iter_mut()
                    .map(|term| {
                        term.eq.bind(bind);
                        term.eq.current_scalar()
                    })
                    .sum();
                terms.clear();
                drop(std::mem::take(scratch));
                value
            }
        };
        self.state = State::Finished((weight, self.columns.final_values()));
        Ok(())
    }
}

#[cfg(test)]
mod allocation_tests {
    use super::{checked_len, ChunkProductError};
    use jolt_field::F128Accumulator;

    #[test]
    fn overflowing_scratch_count_and_byte_size_are_typed_errors() {
        for elements in [None, Some(usize::MAX)] {
            assert!(matches!(
                checked_len::<F128Accumulator>(elements, "equality term scratch"),
                Err(ChunkProductError::AllocationSize {
                    buffer: "equality term scratch"
                })
            ));
        }
    }
}
