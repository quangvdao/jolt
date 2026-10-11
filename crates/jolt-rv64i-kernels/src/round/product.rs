//! Products of linear factors are assembled as quadratics and evaluated at binary-field nodes.

use jolt_field::F128;

/// Ascending coefficients of `(left[0] + X left[1]) (right[0] + X right[1])`.
/// Karatsuba uses exactly three field multiplications. A loop over pairs uses
/// [`quadratic_at_nodes`] and multiplies the resulting values pointwise.
#[inline]
pub fn quadratic(left: [F128; 2], right: [F128; 2]) -> [F128; 3] {
    let constant = left[0] * right[0];
    let leading = left[1] * right[1];
    let middle = (left[0] + left[1]) * (right[0] + right[1]) + constant + leading;
    [constant, middle, leading]
}

/// Values of `q(t) = a + b*t + c*t^2` at raw nodes 2 through 7, in that order.
/// Write `X` for the field generator: `q(2) = a + X*(b + X*c)`,
/// `q(4) = a + X^2*(b + X^2*c)`, `q(3) = q(2) + b + c`,
/// `q(5) = q(4) + b + c`, `q(6) = q(2) + q(4) + a`,
/// and `q(7) = q(6) + b + c` because squaring is additive.
/// Uses five `mul_x` shifts and otherwise XORs, without field multiplications:
/// `X*c` is shared, so nodes 2 and 3 alone take two shifts.
/// A caller wanting fewer nodes reads a prefix, fixing its length outside the
/// loop over pairs so the compiler can discard unused values.
#[inline]
pub fn quadratic_at_nodes(q: [F128; 3]) -> [F128; 6] {
    let [a, b, c] = q;
    let xc = c.mul_x();
    let at_two = a + (b + xc).mul_x();
    let at_four = a + (b + xc.mul_x()).mul_x().mul_x();
    let at_six = at_two + at_four + a;
    let odd_delta = b + c;
    [
        at_two,
        at_two + odd_delta,
        at_four,
        at_four + odd_delta,
        at_six,
        at_six + odd_delta,
    ]
}

/// Values of `l(t) = a + b*t` at raw nodes 2 through 7, in that order.
/// With `X` the field generator, `l(2) = a + X*b`, `l(4) = a + X^2*b`,
/// `l(3) = l(2) + b`, `l(5) = l(4) + b`, `l(6) = l(2) + l(4) + a`,
/// and `l(7) = l(6) + b`. Uses two `mul_x` shifts and otherwise XORs,
/// without field multiplications. A caller wanting fewer nodes reads a prefix,
/// fixing its length outside the loop over pairs.
#[inline]
pub fn linear_at_nodes(l: [F128; 2]) -> [F128; 6] {
    let [a, b] = l;
    let xb = b.mul_x();
    let x2b = xb.mul_x();
    let at_two = a + xb;
    let at_four = a + x2b;
    let at_six = at_two + at_four + a;
    [at_two, at_two + b, at_four, at_four + b, at_six, at_six + b]
}
