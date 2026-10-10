//! Equality tables and Gruen rounds on points with the low index bit first.

use super::RoundError;
use jolt_field::F128;
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial};

fn msb_first(point: &[F128]) -> Vec<F128> {
    point.iter().rev().copied().collect()
}

/// Entry `k` is `scale * eq(point, k)`, pairing coordinate `i` with bit `i`.
/// `None` means unit scale; an empty point gives the single entry `scale`.
pub fn eq_table(point: &[F128], scale: Option<F128>) -> Vec<F128> {
    EqPolynomial::evals(&msb_first(point), scale)
}

/// Round `i` consumes coordinate `i`; inner entries index low pair bits and
/// outer entries high pair bits. `None` means unit scale. Rejects empty points
/// before the shared type's `current_linear_evals` can read a missing coordinate.
pub fn split_eq(
    point: &[F128],
    scale: Option<F128>,
) -> Result<GruenSplitEqPolynomial<F128>, RoundError> {
    if point.is_empty() {
        return Err(RoundError::EmptyPoint);
    }
    Ok(GruenSplitEqPolynomial::new_with_scaling(
        &msb_first(point),
        BindingOrder::LowToHigh,
        scale,
    ))
}
