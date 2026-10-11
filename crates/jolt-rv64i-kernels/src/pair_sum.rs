//! Low-variable-first sum-check of a sum of dense table products.

use crate::par::CycleChunks;
use jolt_field::{Accumulator, F128Accumulator, F128};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;
use thiserror::Error;

/// Invalid pair count or dense table geometry.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PairSumError {
    #[error("pair count {count} is outside 1..=8")]
    Pairs { count: usize },
    #[error("pair {pair} table {table} has length {actual}; it must match common length {expected} and be a power of two >= 2")]
    Length {
        pair: usize,
        table: &'static str,
        expected: usize,
        actual: usize,
    },
}

struct Pair {
    h: Vec<F128>,
    r: Vec<F128>,
    h_buffer: Vec<F128>,
    r_buffer: Vec<F128>,
}

enum State {
    Round(usize),
    LastBind,
    Finished(Vec<(F128, F128)>),
}

impl Pair {
    fn round_sums(&mut self, bind: Option<F128>, chunk_len: usize) -> [F128Accumulator; 2] {
        match bind {
            Some(challenge) => self.bind_and_sums(challenge, chunk_len),
            None => self.sums(chunk_len),
        }
    }

    fn bind_and_sums(&mut self, challenge: F128, chunk_len: usize) -> [F128Accumulator; 2] {
        let half = self.h.len() / 2;
        self.h_buffer.truncate(half);
        self.r_buffer.truncate(half);
        let sums = self
            .h_buffer
            .par_chunks_mut(chunk_len)
            .zip(self.r_buffer.par_chunks_mut(chunk_len))
            .zip(self.h.par_chunks(2 * chunk_len))
            .zip(self.r.par_chunks(2 * chunk_len))
            .map(|(((h_out, r_out), h), r)| {
                let mut sums = [F128Accumulator::default(); 2];
                for (((h_out, r_out), h), r) in h_out
                    .chunks_exact_mut(2)
                    .zip(r_out.chunks_exact_mut(2))
                    .zip(h.chunks_exact(4))
                    .zip(r.chunks_exact(4))
                {
                    let h0 = h[0] + challenge * (h[0] + h[1]);
                    let h1 = h[2] + challenge * (h[2] + h[3]);
                    let r0 = r[0] + challenge * (r[0] + r[1]);
                    let r1 = r[2] + challenge * (r[2] + r[3]);
                    h_out[0] = h0;
                    h_out[1] = h1;
                    r_out[0] = r0;
                    r_out[1] = r1;
                    sums[0].fmadd(h0, r0);
                    sums[1].fmadd(h0 + h1, r0 + r1);
                }
                sums
            })
            .reduce(
                || [F128Accumulator::default(); 2],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        left.merge(right);
                    }
                    left
                },
            );
        std::mem::swap(&mut self.h, &mut self.h_buffer);
        std::mem::swap(&mut self.r, &mut self.r_buffer);
        sums
    }

    fn sums(&self, chunk_len: usize) -> [F128Accumulator; 2] {
        self.h
            .par_chunks(chunk_len)
            .zip(self.r.par_chunks(chunk_len))
            .map(|(h, r)| {
                let mut sums = [F128Accumulator::default(); 2];
                for (h, r) in h.chunks_exact(2).zip(r.chunks_exact(2)) {
                    sums[0].fmadd(h[0], r[0]);
                    sums[1].fmadd(h[0] + h[1], r[0] + r[1]);
                }
                sums
            })
            .reduce(
                || [F128Accumulator::default(); 2],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        left.merge(right);
                    }
                    left
                },
            )
    }
}

/// Degree-two sum-check of `sum_t sum_k H_t[k] * R_t[k]`.
/// The caller supplies the honest input claim; recovering the value at one from
/// that claim leaves its agreement with the tables to the verifier's final check.
pub struct PairSumCore {
    pairs: Vec<Pair>,
    rounds: usize,
    state: State,
}

impl PairSumCore {
    /// Takes one to eight pairs of equally sized power-of-two tables of length at least two.
    pub fn new(pairs: Vec<(Vec<F128>, Vec<F128>)>) -> Result<Self, PairSumError> {
        if !(1..=8).contains(&pairs.len()) {
            return Err(PairSumError::Pairs { count: pairs.len() });
        }
        let length = pairs[0].0.len();
        for (pair, (h, r)) in pairs.iter().enumerate() {
            for (table, actual) in [("H", h.len()), ("R", r.len())] {
                if actual < 2 || !actual.is_power_of_two() || actual != length {
                    return Err(PairSumError::Length {
                        pair,
                        table,
                        expected: length,
                        actual,
                    });
                }
            }
        }
        Ok(Self {
            pairs: pairs
                .into_iter()
                .map(|(h, r)| Pair {
                    h,
                    r,
                    h_buffer: unsafe_allocate_zero_vec(length / 2),
                    r_buffer: unsafe_allocate_zero_vec(length / 2),
                })
                .collect(),
            rounds: length.ilog2() as usize,
            state: State::Round(0),
        })
    }

    /// Returns each `(H(point), R(point))` in input order after `finish_rounds`;
    /// before completion, the slice is empty.
    pub fn final_values(&self) -> &[(F128, F128)] {
        match &self.state {
            State::Finished(values) => values,
            State::Round(_) | State::LastBind => &[],
        }
    }
}

impl ProveRounds<F128> for PairSumCore {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        previous_claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        let expected = match self.state {
            State::Round(next) => next,
            State::LastBind | State::Finished(_) => self.rounds,
        };
        if round != expected || round >= self.rounds {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected,
                got: round,
            });
        }
        if (round == 0) != bind.is_none() {
            return Err(SumcheckError::MissingEvaluationSource {
                kind: "pair sum previous challenge",
            });
        }
        // PairSumCore::new and the round checks above pin this geometry.
        let chunks = CycleChunks::new(self.rounds, round).map_err(|_| {
            SumcheckError::WrongNumberOfRounds {
                expected: self.rounds,
                got: round,
            }
        })?;
        let merge = |mut sums: [F128Accumulator; 2], partial: [F128Accumulator; 2]| {
            for (sum, partial) in sums.iter_mut().zip(partial) {
                sum.merge(partial);
            }
            sums
        };
        let sums = if chunks.len() > chunks.chunk_len() {
            self.pairs
                .par_iter_mut()
                .map(|pair| pair.round_sums(bind, chunks.chunk_len()))
                .reduce(|| [F128Accumulator::default(); 2], merge)
        } else {
            self.pairs
                .iter_mut()
                .map(|pair| pair.round_sums(bind, chunks.chunk_len()))
                .fold([F128Accumulator::default(); 2], merge)
        };
        let [constant, leading] = sums.map(Accumulator::reduce);
        self.state = if round + 1 == self.rounds {
            State::LastBind
        } else {
            State::Round(round + 1)
        };
        Ok(UnivariatePoly::new(vec![
            constant,
            previous_claim + leading,
            leading,
        ]))
    }

    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        if !matches!(self.state, State::LastBind) {
            return Err(SumcheckError::MissingEvaluationSource {
                kind: "pair sum final challenge",
            });
        }
        let values = self
            .pairs
            .iter()
            .map(|pair| {
                (
                    pair.h[0] + bind * (pair.h[0] + pair.h[1]),
                    pair.r[0] + bind * (pair.r[0] + pair.r[1]),
                )
            })
            .collect();
        drop(std::mem::take(&mut self.pairs));
        self.state = State::Finished(values);
        Ok(())
    }
}
