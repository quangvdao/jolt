//! Adjacent-pair sumcheck rounds after the streaming bridge round.

use crate::parallel::MIN_TASK;
use jolt_field::{Accumulator, WithAccumulator, Zero, F192};
use jolt_rv64i_verifier::whir::error::{try_vec, WhirError, WhirPart};
use rayon::prelude::*;

type ProductAccumulator = <F192 as WithAccumulator>::Accumulator;

/// Dense message and weight, with the low variables bound first.
pub(crate) struct RoundState {
    pub(crate) message: Vec<F192>,
    pub(crate) weight: Vec<F192>,
}

impl RoundState {
    pub(crate) fn new(message: Vec<F192>, weight: Vec<F192>) -> Result<Self, WhirError> {
        if !message.len().is_power_of_two() {
            return Err(WhirError::Shape {
                part: WhirPart::FinalValues,
                expected: message
                    .len()
                    .checked_next_power_of_two()
                    .unwrap_or(usize::MAX),
                actual: message.len(),
            });
        }
        if weight.len() != message.len() {
            return Err(WhirError::Shape {
                part: WhirPart::FinalValues,
                expected: message.len(),
                actual: weight.len(),
            });
        }
        Ok(Self { message, weight })
    }

    pub(crate) fn coefficients(&self) -> Result<[F192; 2], WhirError> {
        self.validate_round()?;
        let coefficients = self
            .message
            .par_chunks_exact(2)
            .zip(self.weight.par_chunks_exact(2))
            .fold(
                || [ProductAccumulator::default(); 2],
                |mut sums, (f, w)| {
                    let [f0, f1] = [f[0], f[1]];
                    let [w0, w1] = [w[0], w[1]];
                    sums[0].fmadd(f0, w0);
                    sums[1].fmadd(f0 + f1, w0 + w1);
                    sums
                },
            )
            .reduce(
                || [ProductAccumulator::default(); 2],
                |mut a, b| {
                    for (a, b) in a.iter_mut().zip(b) {
                        a.merge(b);
                    }
                    a
                },
            );
        Ok(coefficients.map(Accumulator::reduce))
    }

    /// A level boundary releases both old allocations before the next oracle.
    pub(crate) fn fold(&mut self, challenge: F192, release: bool) -> Result<(), WhirError> {
        self.validate_round()?;
        let len = self.message.len() / 2;
        if release {
            let mut message = try_vec(WhirPart::FinalValues, len)?;
            let mut weight = try_vec(WhirPart::FinalValues, len)?;
            message.resize(len, F192::zero());
            weight.resize(len, F192::zero());
            message
                .par_iter_mut()
                .zip(weight.par_iter_mut())
                .zip(self.message.par_chunks_exact(2))
                .zip(self.weight.par_chunks_exact(2))
                .for_each(|(((out_f, out_w), f), w)| {
                    *out_f = f[0] + challenge * (f[0] + f[1]);
                    *out_w = w[0] + challenge * (w[0] + w[1]);
                });
            self.message = message;
            self.weight = weight;
        } else {
            let _ = rayon::join(
                || Self::fold_in_place(&mut self.message, challenge),
                || Self::fold_in_place(&mut self.weight, challenge),
            );
        }
        Ok(())
    }

    fn validate_round(&self) -> Result<(), WhirError> {
        if self.message.len() < 2 || !self.message.len().is_power_of_two() {
            return Err(WhirError::Shape {
                part: WhirPart::Rounds,
                expected: 2,
                actual: self.message.len(),
            });
        }
        if self.weight.len() != self.message.len() {
            return Err(WhirError::Shape {
                part: WhirPart::FinalValues,
                expected: self.message.len(),
                actual: self.weight.len(),
            });
        }
        Ok(())
    }

    fn fold_in_place(values: &mut Vec<F192>, challenge: F192) {
        let cutoff = if rayon::current_num_threads() > 1 {
            2 * MIN_TASK
        } else {
            2
        };
        Self::fold_prefix(values, challenge, cutoff);
        values.truncate(values.len() / 2);
    }

    fn fold_prefix(values: &mut [F192], challenge: F192, cutoff: usize) {
        if values.len() <= cutoff {
            for index in 0..values.len() / 2 {
                values[index] =
                    values[2 * index] + challenge * (values[2 * index] + values[2 * index + 1]);
            }
            return;
        }
        let (left, right) = values.split_at_mut(values.len() / 2);
        Self::fold_prefix(left, challenge, cutoff);
        // The original left half is consumed before its upper quarter becomes
        // output storage for the original right half. Every output is written
        // once, with disjoint input/output borrows and no compaction pass.
        let (_, output) = left.split_at_mut(left.len() / 2);
        output
            .par_iter_mut()
            .zip(right.par_chunks_exact(2))
            .for_each(|(out, pair)| *out = pair[0] + challenge * (pair[0] + pair[1]));
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests assert valid round shapes")]
mod tests {
    use super::RoundState;
    use crate::induce::inner_product;
    use jolt_field::{ExtField, One, Zero, F192, F64};
    use jolt_rv64i_verifier::whir::error::{WhirError, WhirPart};
    use rayon::ThreadPoolBuilder;

    #[test]
    fn round_coefficients_and_folds_obey_sumcheck_identity() {
        for threads in [1, 12] {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    for release in [false, true] {
                        let f: Vec<_> = (1..=65536)
                            .map(|i| F192::lift_base(F64::from_raw(i)))
                            .collect();
                        let w: Vec<_> = (1..=65536)
                            .map(|i| F192::lift_base(F64::from_raw(i * 31)))
                            .collect();
                        let mut state = RoundState::new(f, w).unwrap();
                        let old_capacity = state.message.capacity();
                        while state.message.len() > 1 {
                            let sigma = inner_product(&state.message, &state.weight).unwrap();
                            let [u0, u2] = state.coefficients().unwrap();
                            let a = F192::from_base_fn(|i| F64::from_raw(7 + i as u64));
                            state.fold(a, release).unwrap();
                            assert_eq!(
                                inner_product(&state.message, &state.weight).unwrap(),
                                u0 + (sigma + u2) * a + u2 * a * a
                            );
                        }
                        if release {
                            assert_eq!(state.message.capacity(), 1);
                        } else {
                            assert_eq!(state.message.capacity(), old_capacity);
                        }
                    }
                });
        }
    }

    #[test]
    fn boolean_challenges_select_the_adjacent_endpoint() {
        for a in [F192::zero(), F192::one()] {
            let mut state = RoundState::new(
                vec![F192::zero(), F192::one()],
                vec![F192::one(), F192::zero()],
            )
            .unwrap();
            state.fold(a, false).unwrap();
            assert_eq!(state.message, [a]);
            assert_eq!(state.weight, [F192::one() + a]);
        }
    }

    #[test]
    fn malformed_round_buffers_return_shapes() {
        assert!(matches!(
            RoundState::new(Vec::new(), Vec::new()),
            Err(WhirError::Shape {
                part: WhirPart::FinalValues,
                ..
            })
        ));
        assert!(matches!(
            RoundState::new(vec![F192::one(); 2], Vec::new()),
            Err(WhirError::Shape {
                part: WhirPart::FinalValues,
                ..
            })
        ));
        let state = RoundState::new(vec![F192::one()], vec![F192::one()]).unwrap();
        assert!(matches!(
            state.coefficients(),
            Err(WhirError::Shape {
                part: WhirPart::Rounds,
                ..
            })
        ));
    }
}
