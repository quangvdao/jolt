//! Dense degree-two example: the sum of the products of two multilinear tables.

use jolt_field::{Accumulator, F128Accumulator, F128};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExampleError {
    #[error("dense product table length {length} is not a positive power of two")]
    InvalidLength { length: usize },
    #[error("dense product tables have lengths {a} and {b}")]
    LengthMismatch { a: usize, b: usize },
    #[error("dense product final values require {rounds} completed rounds")]
    Unfinished { rounds: usize },
}

enum State {
    Round(usize),
    LastBind,
    Finished([F128; 2]),
    Failed,
}

/// Two dense leaves, bound low variable first, with a second buffer per leaf.
pub struct DenseProductCore {
    a: Vec<F128>,
    b: Vec<F128>,
    a_scratch: Vec<F128>,
    b_scratch: Vec<F128>,
    rounds: usize,
    initial_claim: F128,
    state: State,
}

impl DenseProductCore {
    /// Checks equal power-of-two lengths before allocating the binding buffers.
    pub fn new(a: Vec<F128>, b: Vec<F128>) -> Result<Self, ExampleError> {
        if a.len() != b.len() {
            return Err(ExampleError::LengthMismatch {
                a: a.len(),
                b: b.len(),
            });
        }
        if !a.len().is_power_of_two() {
            return Err(ExampleError::InvalidLength { length: a.len() });
        }
        let rounds = a.len().ilog2() as usize;
        let initial_claim = a
            .par_chunks(4096)
            .zip(b.par_chunks(4096))
            .map(|(a, b)| {
                let mut sum = F128Accumulator::default();
                for (&a, &b) in a.iter().zip(b) {
                    sum.fmadd(a, b);
                }
                sum
            })
            .reduce(F128Accumulator::default, |mut left, right| {
                left.merge(right);
                left
            })
            .reduce();
        let state = if rounds == 0 {
            State::Finished([a[0], b[0]])
        } else {
            State::Round(0)
        };
        let half = a.len() / 2;
        Ok(Self {
            a,
            b,
            a_scratch: vec![F128::from_raw(0); half],
            b_scratch: vec![F128::from_raw(0); half],
            rounds,
            initial_claim,
            state,
        })
    }

    pub fn initial_claim(&self) -> F128 {
        self.initial_claim
    }

    /// Returns both leaf extensions after the final challenge has been bound.
    pub fn final_values(&self) -> Result<[F128; 2], ExampleError> {
        match self.state {
            State::Finished(values) => Ok(values),
            State::Round(_) | State::LastBind | State::Failed => Err(ExampleError::Unfinished {
                rounds: self.rounds,
            }),
        }
    }

    fn bind(&mut self, challenge: F128) {
        let half = self.a.len() / 2;
        self.a_scratch.truncate(half);
        self.b_scratch.truncate(half);
        self.a_scratch
            .par_iter_mut()
            .zip(self.b_scratch.par_iter_mut())
            .zip(self.a.par_chunks_exact(2).zip(self.b.par_chunks_exact(2)))
            .for_each(|((a_out, b_out), (a, b))| {
                *a_out = a[0] + challenge * (a[0] + a[1]);
                *b_out = b[0] + challenge * (b[0] + b[1]);
            });
        std::mem::swap(&mut self.a, &mut self.a_scratch);
        std::mem::swap(&mut self.b, &mut self.b_scratch);
    }
}

impl ProveRounds<F128> for DenseProductCore {
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
        match (round, bind) {
            (0, None) => {}
            (0, Some(_)) | (_, None) => {
                return Err(SumcheckError::MissingEvaluationSource {
                    kind: "dense product previous challenge",
                });
            }
            (_, Some(_)) => {}
        }
        self.state = State::Failed;
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let sums = self
            .a
            .par_chunks(4096)
            .zip(self.b.par_chunks(4096))
            .map(|(a, b)| {
                let mut coefficients = [F128Accumulator::default(); 3];
                for (a, b) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
                    let delta_a = a[0] + a[1];
                    let delta_b = b[0] + b[1];
                    coefficients[0].fmadd(a[0], b[0]);
                    coefficients[1].fmadd(a[0], delta_b);
                    coefficients[1].fmadd(delta_a, b[0]);
                    coefficients[2].fmadd(delta_a, delta_b);
                }
                coefficients
            })
            .reduce(
                || [F128Accumulator::default(); 3],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        left.merge(right);
                    }
                    left
                },
            )
            .map(Accumulator::reduce);
        let actual = sums[1] + sums[2];
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
        Ok(UnivariatePoly::new(sums.to_vec()))
    }

    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        if !matches!(self.state, State::LastBind) {
            return Err(SumcheckError::MissingEvaluationSource {
                kind: "dense product final challenge",
            });
        }
        self.bind(bind);
        self.state = State::Finished([self.a[0], self.b[0]]);
        drop(std::mem::take(&mut self.a));
        drop(std::mem::take(&mut self.b));
        drop(std::mem::take(&mut self.a_scratch));
        drop(std::mem::take(&mut self.b_scratch));
        Ok(())
    }
}
