//! Shared low-first slot sum-check over complete folds and sparse public routing tensors.

use super::shape::{RouteEntry, RouterError, RouterShape};
use crate::round::eq::eq_table;
use jolt_field::{Accumulator, F128Accumulator, F128};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use rayon::prelude::*;

const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);

struct ShapeState {
    occupied: Vec<usize>,
    fold: Vec<F128>,
    weight: Vec<F128>,
    fold_buffer: Vec<F128>,
    weight_buffer: Vec<F128>,
    sum: F128,
    scalar: F128,
    polynomial: [F128; 3],
}

impl ShapeState {
    fn bind(&mut self, slot: usize, challenge: F128) {
        if self.occupied.binary_search(&slot).is_err() {
            self.scalar *= ONE + challenge;
            return;
        }
        self.sum =
            self.polynomial[0] + challenge * (self.polynomial[1] + challenge * self.polynomial[2]);
        let half = self.fold.len() / 2;
        self.fold_buffer.truncate(half);
        self.weight_buffer.truncate(half);
        self.fold_buffer
            .par_chunks_mut(4096)
            .zip(self.weight_buffer.par_chunks_mut(4096))
            .zip(self.fold.par_chunks(8192))
            .zip(self.weight.par_chunks(8192))
            .for_each(|(((fold_out, weight_out), fold), weight)| {
                for (((fold_out, weight_out), fold), weight) in fold_out
                    .iter_mut()
                    .zip(weight_out)
                    .zip(fold.chunks_exact(2))
                    .zip(weight.chunks_exact(2))
                {
                    *fold_out = fold[0] + challenge * (fold[0] + fold[1]);
                    *weight_out = weight[0] + challenge * (weight[0] + weight[1]);
                }
            });
        std::mem::swap(&mut self.fold, &mut self.fold_buffer);
        std::mem::swap(&mut self.weight, &mut self.weight_buffer);
    }

    fn message(&mut self, slot: usize) -> [F128; 3] {
        if self.occupied.binary_search(&slot).is_err() {
            let constant = self.scalar * self.sum;
            return [constant, constant, ZERO];
        }
        let [constant, leading] = self
            .fold
            .par_chunks(4096)
            .zip(self.weight.par_chunks(4096))
            .map(|(fold, weight)| {
                let mut sums = [F128Accumulator::default(); 2];
                for (fold, weight) in fold.chunks_exact(2).zip(weight.chunks_exact(2)) {
                    sums[0].fmadd(fold[0], weight[0]);
                    sums[1].fmadd(fold[0] + fold[1], weight[0] + weight[1]);
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
            .map(Accumulator::reduce);
        self.polynomial = [constant, self.sum + leading, leading];
        self.polynomial.map(|value| self.scalar * value)
    }
}

/// One degree-two member over shared slots, storing only each shape's occupied cube.
/// Complete fold tables and public route supports are checked before allocation.
pub struct RouterShortCore {
    shapes: Vec<ShapeState>,
    slots: usize,
    next: usize,
    values: Option<Vec<(F128, F128)>>,
    failed: bool,
}

impl RouterShortCore {
    /// Builds each `W` by scattering `eq(w,o)` over the shape's route set.
    /// Checks one complete fold per shape, common slot counts and output-point
    /// dimensions. Agreement of the supplied folds with the committed source is
    /// required of the caller, not checked. Detection rests on the verifier's final
    /// evaluation check against the committed source, with the sum-check's soundness error.
    pub fn new(
        shapes: &[RouterShape],
        w: &[F128],
        folds: Vec<Vec<F128>>,
    ) -> Result<Self, RouterError> {
        let first = shapes.first().ok_or(RouterError::EmptyShapes)?;
        if folds.len() != shapes.len() {
            return Err(RouterError::TableLength {
                table: "fold list",
                expected: shapes.len(),
                actual: folds.len(),
            });
        }
        let slots = first.slots();
        let mut states = Vec::with_capacity(shapes.len());
        if w.len() != first.log_outputs() {
            return Err(RouterError::PointLength {
                expected: first.log_outputs(),
                actual: w.len(),
            });
        }
        let output_weights = eq_table(w, None);
        for (shape, fold) in shapes.iter().zip(folds) {
            if shape.slots() != slots {
                return Err(RouterError::SlotCount {
                    expected: slots,
                    actual: shape.slots(),
                });
            }
            if w.len() != shape.log_outputs() {
                return Err(RouterError::PointLength {
                    expected: shape.log_outputs(),
                    actual: w.len(),
                });
            }
            if fold.len() != shape.fold_len() {
                return Err(RouterError::TableLength {
                    table: "fold",
                    expected: shape.fold_len(),
                    actual: fold.len(),
                });
            }
            let mut weight = unsafe_allocate_zero_vec(shape.fold_len());
            for &RouteEntry {
                output,
                source,
                selector,
            } in shape.route()
            {
                let index = shape.fold_index(source, selector);
                let eq = output_weights[output];
                weight[index] += eq;
            }
            let sum = fold
                .par_chunks(4096)
                .zip(weight.par_chunks(4096))
                .map(|(fold, weight)| {
                    let mut sum = F128Accumulator::default();
                    for (&fold, &weight) in fold.iter().zip(weight) {
                        sum.fmadd(fold, weight);
                    }
                    sum
                })
                .reduce(F128Accumulator::default, |mut left, right| {
                    left.merge(right);
                    left
                })
                .reduce();
            let half = fold.len() / 2;
            states.push(ShapeState {
                occupied: shape.slot_map().iter().map(|&(slot, _)| slot).collect(),
                fold,
                weight,
                fold_buffer: unsafe_allocate_zero_vec(half),
                weight_buffer: unsafe_allocate_zero_vec(half),
                sum,
                scalar: ONE,
                polynomial: [ZERO; 3],
            });
        }
        Ok(Self {
            shapes: states,
            slots,
            next: 0,
            values: None,
            failed: false,
        })
    }

    /// After the final bind, returns `(Fold(x), Idle(x) * W(x))` in shape order.
    /// Calls before completion return `RouterError::Unfinished`.
    pub fn final_values(&self) -> Result<Vec<(F128, F128)>, RouterError> {
        self.values.clone().ok_or(RouterError::Unfinished)
    }
}

impl ProveRounds<F128> for RouterShortCore {
    fn num_rounds(&self) -> usize {
        self.slots
    }
    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        previous_claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        if self.failed || self.values.is_some() || round != self.next || round >= self.slots {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: self.next,
                got: round,
            });
        }
        if (round == 0) != bind.is_none() {
            return Err(SumcheckError::MissingEvaluationSource {
                kind: "short previous challenge",
            });
        }
        if let Some(challenge) = bind {
            for shape in &mut self.shapes {
                shape.bind(round - 1, challenge);
            }
        }
        let mut coefficients = [ZERO; 3];
        for shape in &mut self.shapes {
            for (dest, value) in coefficients.iter_mut().zip(shape.message(round)) {
                *dest += value;
            }
        }
        let actual = coefficients[1] + coefficients[2];
        if actual != previous_claim {
            self.failed = true;
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual,
            });
        }
        self.next += 1;
        Ok(UnivariatePoly::new(coefficients.to_vec()))
    }
    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        if self.next != self.slots || self.values.is_some() || self.failed {
            return Err(SumcheckError::MissingEvaluationSource {
                kind: "short final challenge",
            });
        }
        for shape in &mut self.shapes {
            shape.bind(self.slots - 1, bind);
            drop(std::mem::take(&mut shape.fold_buffer));
            drop(std::mem::take(&mut shape.weight_buffer));
        }
        self.values = Some(
            self.shapes
                .iter()
                .map(|shape| (shape.fold[0], shape.scalar * shape.weight[0]))
                .collect(),
        );
        drop(std::mem::take(&mut self.shapes));
        Ok(())
    }
}
