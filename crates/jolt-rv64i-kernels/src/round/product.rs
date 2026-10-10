//! Products of linear factors are assembled as quadratics and evaluated at binary-field nodes.

use jolt_field::F128;

/// Ascending coefficients of `(left[0] + X left[1]) (right[0] + X right[1])`.
/// Karatsuba uses exactly three field multiplications. Evaluate the result with
/// [`super::eval_at_node`] and multiply the resulting values pointwise in a core.
#[inline]
pub fn quadratic(left: [F128; 2], right: [F128; 2]) -> [F128; 3] {
    let constant = left[0] * right[0];
    let leading = left[1] * right[1];
    let middle = (left[0] + left[1]) * (right[0] + right[1]) + constant + leading;
    [constant, middle, leading]
}
