//! Initial RAM contains input and image words, never final public I/O contents.

use crate::commitment::BitsCommitmentScheme;
use crate::points::{self, PointsError, WordLift};
use crate::statement::CheckedInputs;
use jolt_field::{Zero, F128};
use jolt_poly::EqPolynomial;

/// Evaluates canonical sparse initial RAM with two split address tables.
pub fn evaluate<S: BitsCommitmentScheme>(
    checked: &CheckedInputs<'_, S>,
    a_ram: &[F128],
    r_bit: &[F128],
) -> Result<F128, PointsError> {
    if a_ram.len() != checked.log_K_ram() {
        return Err(PointsError::Dimension {
            expected: checked.log_K_ram(),
            actual: a_ram.len(),
        });
    }
    if r_bit.len() != 6 {
        return Err(PointsError::Dimension {
            expected: 6,
            actual: r_bit.len(),
        });
    }
    let lift = WordLift::new(r_bit)?;
    let split = a_ram.len() / 2;
    let (lo, hi) = a_ram.split_at(split);
    let low = EqPolynomial::new(points::to_high_to_low(lo)).evaluations();
    let high = EqPolynomial::new(points::to_high_to_low(hi)).evaluations();
    let mask = low.len() - 1;
    checked
        .initial_ram()
        .iter()
        .try_fold(F128::zero(), |sum, &(index, word)| {
            let index = usize::try_from(index).map_err(|_| PointsError::Index {
                index: usize::MAX,
                variables: a_ram.len(),
            })?;
            let left = low.get(index & mask).ok_or(PointsError::Index {
                index,
                variables: a_ram.len(),
            })?;
            let right = high.get(index >> split).ok_or(PointsError::Index {
                index,
                variables: a_ram.len(),
            })?;
            Ok(sum + lift.evaluate(word) * *left * *right)
        })
}
