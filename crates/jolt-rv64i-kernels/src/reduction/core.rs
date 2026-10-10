use super::ReductionError;
use crate::par::CycleChunks;
use crate::round::eq::{eq_table, split_eq};
use crate::round::RoundError;
use jolt_field::{Accumulator, F128Accumulator, F128};
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;

/// One input equality claim, selecting an input table and a low-variable-first
/// point. Its defining equality sum is required of the
/// caller, not checked by construction. A false claim produces messages of
/// another polynomial and is detected by the verifier's final batched check,
/// except when its coefficient is zero or legs at one point cancel.
/// [`ReductionCore::check_claims`] diagnoses each individual leg instead.
#[derive(Clone, Debug)]
pub struct ReductionLeg {
    pub table: usize,
    pub point: Vec<F128>,
    pub coefficient: F128,
    pub claim: F128,
}

#[derive(Clone, Copy)]
struct TableLeg<'a> {
    slot: usize,
    inner: &'a [F128],
    outer: &'a [F128],
}

struct LegState {
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
    legs: Vec<LegState>,
    partials: Vec<F128Accumulator>,
    rounds: usize,
    state: State,
}

impl ReductionCore {
    /// Accept at most eight legs. All tables have the same power-of-two length
    /// of at least two, and points have its logarithm coordinates.
    /// Each claim must equal `sum_j eq(point,j)*table[j]`: this is required of the
    /// caller, not checked here; the verifier detects a false batched statement
    /// at its final check. Zero coefficients or cancelling legs at one point
    /// can hide an individual false claim; `check_claims` diagnoses it.
    pub fn new(tables: Vec<Vec<F128>>, legs: Vec<ReductionLeg>) -> Result<Self, ReductionError> {
        if legs.len() > 8 {
            return Err(ReductionError::LegCount { count: legs.len() });
        }
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
        for (
            leg,
            ReductionLeg {
                table,
                point,
                coefficient,
                claim,
            },
        ) in legs.into_iter().enumerate()
        {
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
            checked.push(LegState {
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
            .map(|_| unsafe_allocate_zero_vec(length / 2))
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

    /// Returns one value per input table, in table order, after `finish_rounds`,
    /// dropping all dense tables and second buffers before the finished state
    /// becomes observable.
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
        let table_legs: Vec<Vec<_>> = (0..tables)
            .map(|table| {
                legs.iter()
                    .enumerate()
                    .filter(|(_, leg)| leg.table == table)
                    .map(|(slot, leg)| TableLeg {
                        slot,
                        inner: leg.eq.e_in_current(),
                        outer: leg.eq.e_out_current(),
                    })
                    .collect()
            })
            .collect();
        let mut views = Vec::with_capacity(count * tables);
        if bind.is_some() {
            for scratch in &mut self.scratch {
                scratch.truncate(length);
            }
            let mut columns: Vec<_> = self
                .tables
                .iter()
                .zip(&mut self.scratch)
                .map(|(input, output)| input.chunks(2 * chunk).zip(output.chunks_mut(chunk)))
                .collect();
            for _ in 0..count {
                for column in &mut columns {
                    views.extend(column.next());
                }
            }
        } else {
            let mut columns: Vec<_> = self
                .tables
                .iter_mut()
                .map(|table| table.chunks_mut(chunk))
                .collect();
            for _ in 0..count {
                for column in &mut columns {
                    views.extend(column.next().map(|output| (&[][..], output)));
                }
            }
        }
        if !legs.is_empty() {
            views
                .par_chunks_mut(tables)
                .zip(self.partials[..2 * count * leg_count].par_chunks_mut(2 * leg_count))
                .enumerate()
                .for_each(|(index, (views, partials))| {
                    Self::accumulate_chunk(views, &table_legs, partials, index, chunk, bind);
                });
        } else if let Some(challenge) = bind {
            views.par_chunks_mut(tables).for_each(|views| {
                for (input, output) in views {
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

    fn accumulate_chunk(
        views: &mut [(&[F128], &mut [F128])],
        table_legs: &[Vec<TableLeg<'_>>],
        partials: &mut [F128Accumulator],
        index: usize,
        chunk: usize,
        bind: Option<F128>,
    ) {
        let mut total = [F128Accumulator::default(); 8];
        for ((input, output), legs) in views.iter_mut().zip(table_legs) {
            let Some(first) = legs.first() else {
                if let Some(challenge) = bind {
                    for (dest, pair) in output.iter_mut().zip(input.chunks_exact(2)) {
                        *dest = pair[0] + challenge * (pair[0] + pair[1]);
                    }
                }
                continue;
            };
            let block_len = 2 * first.inner.len();
            let first_block = index * (chunk / block_len);
            for (block, output) in output.chunks_exact_mut(block_len).enumerate() {
                let input = if bind.is_some() {
                    &input[2 * block * block_len..2 * (block + 1) * block_len]
                } else {
                    &[][..]
                };
                match legs.as_slice() {
                    [leg] => {
                        let sum = Self::sum_one(input, output, leg.inner, bind);
                        total[leg.slot].fmadd(leg.outer[first_block + block], sum.reduce());
                    }
                    [first, second] => {
                        let [a, b] = Self::sum_two(input, output, first.inner, second.inner, bind);
                        total[first.slot].fmadd(first.outer[first_block + block], a.reduce());
                        total[second.slot].fmadd(second.outer[first_block + block], b.reduce());
                    }
                    _ => {
                        let sums = Self::sum_many(input, output, legs, bind);
                        for (leg, sum) in legs.iter().zip(sums) {
                            total[leg.slot].fmadd(leg.outer[first_block + block], sum.reduce());
                        }
                    }
                }
            }
        }
        let leg_count = partials.len() / 2;
        partials[..leg_count].copy_from_slice(&total[..leg_count]);
    }

    #[inline]
    fn sum_one(
        input: &[F128],
        output: &mut [F128],
        inner: &[F128],
        bind: Option<F128>,
    ) -> F128Accumulator {
        let mut sum = F128Accumulator::default();
        if let Some(challenge) = bind {
            for ((dest, source), &eq) in output
                .chunks_exact_mut(2)
                .zip(input.chunks_exact(4))
                .zip(inner)
            {
                dest[0] = source[0] + challenge * (source[0] + source[1]);
                dest[1] = source[2] + challenge * (source[2] + source[3]);
                sum.fmadd(eq, dest[0] + dest[1]);
            }
        } else {
            for (pair, &eq) in output.chunks_exact(2).zip(inner) {
                sum.fmadd(eq, pair[0] + pair[1]);
            }
        }
        sum
    }

    #[inline]
    fn sum_two(
        input: &[F128],
        output: &mut [F128],
        first: &[F128],
        second: &[F128],
        bind: Option<F128>,
    ) -> [F128Accumulator; 2] {
        let mut a = F128Accumulator::default();
        let mut b = F128Accumulator::default();
        if let Some(challenge) = bind {
            for (((dest, source), &first), &second) in output
                .chunks_exact_mut(2)
                .zip(input.chunks_exact(4))
                .zip(first)
                .zip(second)
            {
                dest[0] = source[0] + challenge * (source[0] + source[1]);
                dest[1] = source[2] + challenge * (source[2] + source[3]);
                let delta = dest[0] + dest[1];
                a.fmadd(first, delta);
                b.fmadd(second, delta);
            }
        } else {
            for ((pair, &first), &second) in output.chunks_exact(2).zip(first).zip(second) {
                let delta = pair[0] + pair[1];
                a.fmadd(first, delta);
                b.fmadd(second, delta);
            }
        }
        [a, b]
    }

    #[inline]
    fn sum_many(
        input: &[F128],
        output: &mut [F128],
        legs: &[TableLeg<'_>],
        bind: Option<F128>,
    ) -> [F128Accumulator; 8] {
        let mut sums = [F128Accumulator::default(); 8];
        if let Some(challenge) = bind {
            for (pair, (dest, source)) in output
                .chunks_exact_mut(2)
                .zip(input.chunks_exact(4))
                .enumerate()
            {
                dest[0] = source[0] + challenge * (source[0] + source[1]);
                dest[1] = source[2] + challenge * (source[2] + source[3]);
                let delta = dest[0] + dest[1];
                for (leg, sum) in legs.iter().zip(&mut sums) {
                    sum.fmadd(leg.inner[pair], delta);
                }
            }
        } else {
            for (pair, values) in output.chunks_exact(2).enumerate() {
                let delta = values[0] + values[1];
                for (leg, sum) in legs.iter().zip(&mut sums) {
                    sum.fmadd(leg.inner[pair], delta);
                }
            }
        }
        sums
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
    use super::{ReductionCore, ReductionLeg};
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
                .map(|(i, point)| ReductionLeg {
                    table: 0,
                    point: point.clone(),
                    coefficient: F128::from_raw(37 + i as u128),
                    claim: quotient(&table, &[], point),
                })
                .collect();
            let eq_tables: Vec<Vec<_>> = points[..count]
                .iter()
                .map(|t| (0..table.len()).map(|j| eq(t, j)).collect())
                .collect();
            let leaves: Vec<_> = std::iter::once(table.as_slice())
                .chain(eq_tables.iter().map(Vec::as_slice))
                .collect();
            let mut claim: F128 = legs.iter().map(|leg| leg.coefficient * leg.claim).sum();
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
                        .map(|(i, leg)| leg.coefficient * v[0] * v[i + 1])
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
