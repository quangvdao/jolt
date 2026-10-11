//! Initial RAM contains input and image words, never final public I/O contents.

use crate::commitment::BitsCommitmentScheme;
use crate::points::{self, PointsError, WordLift};
use crate::statement::CheckedInputs;
use jolt_field::{JoltField, F128};

/// Evaluates canonical sparse initial RAM with two split address tables.
pub fn evaluate<S: BitsCommitmentScheme>(
    checked: &CheckedInputs<'_, S>,
    a_ram: &[F128],
    r_bit: &[F128],
) -> Result<F128, PointsError> {
    InitialRamEvaluation::new(checked, a_ram, r_bit)?.evaluate(checked.initial_ram())
}

/// Reusable word lift and split address weights for evaluating borrowed portions of initial RAM.
pub struct InitialRamEvaluation<F: JoltField> {
    lift: WordLift<F>,
    low: Vec<F>,
    high: Vec<F>,
    split: usize,
    variables: usize,
}

impl<F: JoltField> InitialRamEvaluation<F> {
    /// Checks the RAM and six-bit point dimensions before preparing the shared weights.
    pub fn new<S: BitsCommitmentScheme>(
        checked: &CheckedInputs<'_, S>,
        a_ram: &[F],
        r_bit: &[F],
    ) -> Result<Self, PointsError> {
        if a_ram.len() != checked.log_K_ram() {
            return Err(PointsError::Dimension {
                expected: checked.log_K_ram(),
                actual: a_ram.len(),
            });
        }
        let lift = WordLift::new(r_bit)?;
        let (low, high) = points::split_eq_tables(a_ram)?;
        Ok(Self {
            lift,
            low,
            high,
            split: a_ram.len() / 2,
            variables: a_ram.len(),
        })
    }

    /// Sums the weighted words, returning the first out-of-domain index in slice order.
    #[inline]
    pub fn evaluate(&self, words: &[(u64, u64)]) -> Result<F, PointsError> {
        let low = self.low.as_slice();
        let high = self.high.as_slice();
        let lift = &self.lift;
        let split = self.split;
        let variables = self.variables;
        let mask = low.len() - 1;
        words.iter().try_fold(F::zero(), |sum, &(index, word)| {
            let index = usize::try_from(index).map_err(|_| PointsError::Index {
                index: usize::MAX,
                variables,
            })?;
            let left = low
                .get(index & mask)
                .ok_or(PointsError::Index { index, variables })?;
            let right = high
                .get(index >> split)
                .ok_or(PointsError::Index { index, variables })?;
            Ok(sum + (*left * *right) * lift.evaluate(word))
        })
    }
}
