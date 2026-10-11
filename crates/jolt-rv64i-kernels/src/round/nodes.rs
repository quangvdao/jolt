//! Fixed-degree interpolation at polynomial-basis nodes, not integer ring images.

use super::RoundError;
use jolt_field::F128;

type Matrix = [[F128; 7]; 7];
const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);

// For degree d and node k, each scalar is 1 / product_{j != k, j < d}(k + j),
// with k,j raw field elements. F128 inversion is not const, so only these
// scalars are precomputed; matrix() assembles the Lagrange bases at compile
// time using the shared const mul_x reduction.
const MATRICES: [Matrix; 7] = [
    matrix([F128::from_raw(0x0000_0000_0000_0000_0000_0000_0000_0001)]),
    matrix([
        F128::from_raw(0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ff82),
        F128::from_raw(0x7fff_ffff_ffff_ffff_ffff_ffff_ffff_ffc1),
    ]),
    matrix([
        F128::from_raw(0x7fff_ffff_ffff_ffff_ffff_ffff_ffff_ffc1),
        F128::from_raw(0x7fff_ffff_ffff_ffff_ffff_ffff_ffff_ffc1),
        F128::from_raw(0x7fff_ffff_ffff_ffff_ffff_ffff_ffff_ffc1),
    ]),
    matrix([
        F128::from_raw(0x1999_9999_9999_9999_9999_9999_9999_9995),
        F128::from_raw(0x9555_5555_5555_5555_5555_5555_5555_551c),
        F128::from_raw(0x1249_2492_4924_9249_2492_4924_9249_249b),
        F128::from_raw(0x417a_17a1_7a17_a17a_17a1_7a17_a17a_1780),
    ]),
    matrix([
        F128::from_raw(0xc666_6666_6666_6666_6666_6666_6666_6607),
        F128::from_raw(0x871c_71c7_1c71_c71c_71c7_1c71_c71c_7187),
        F128::from_raw(0x871c_71c7_1c71_c71c_71c7_1c71_c71c_7187),
        F128::from_raw(0x417a_17a1_7a17_a17a_17a1_7a17_a17a_1780),
        F128::from_raw(0x417a_17a1_7a17_a17a_17a1_7a17_a17a_1780),
    ]),
    matrix([
        F128::from_raw(0x417a_17a1_7a17_a17a_17a1_7a17_a17a_1780),
        F128::from_raw(0x61c7_1c71_c71c_71c7_1c71_c71c_71c7_1c40),
        F128::from_raw(0x7e53_e53e_53e5_3e53_e53e_53e5_3e53_e500),
        F128::from_raw(0x20bd_0bd0_bd0b_d0bd_0bd0_bd0b_d0bd_0bc0),
        F128::from_raw(0x3f29_f29f_29f2_9f29_f29f_29f2_9f29_f280),
        F128::from_raw(0x1f94_f94f_94f9_4f94_f94f_94f9_4f94_f940),
    ]),
    matrix([
        F128::from_raw(0x1f94_f94f_94f9_4f94_f94f_94f9_4f94_f940),
        F128::from_raw(0x1f94_f94f_94f9_4f94_f94f_94f9_4f94_f940),
        F128::from_raw(0x1f94_f94f_94f9_4f94_f94f_94f9_4f94_f940),
        F128::from_raw(0x1f94_f94f_94f9_4f94_f94f_94f9_4f94_f940),
        F128::from_raw(0x1f94_f94f_94f9_4f94_f94f_94f9_4f94_f940),
        F128::from_raw(0x1f94_f94f_94f9_4f94_f94f_94f9_4f94_f940),
        F128::from_raw(0x1f94_f94f_94f9_4f94_f94f_94f9_4f94_f940),
    ]),
];

const fn matrix<const N: usize>(inverse_denominators: [F128; N]) -> Matrix {
    let degree = N + 1;
    let mut result = [[ZERO; 7]; 7];
    let mut column = 0;
    while column < degree - 1 {
        let node = column + 1;
        let mut basis = [ZERO; 8];
        basis[0] = ONE;
        let mut length = 1;
        let mut other = 0;
        while other < degree {
            if other != node {
                let mut lower = ZERO;
                let mut i = 0;
                while i <= length {
                    let old = basis[i];
                    basis[i] = F128::from_raw(lower.to_raw() ^ mul_node(old, other as u8).to_raw());
                    lower = old;
                    i += 1;
                }
                length += 1;
            }
            other += 1;
        }
        let mut row = 0;
        while row < degree - 1 {
            let mut bits = basis[row + 1].to_raw();
            let mut value = inverse_denominators[column];
            let mut product = 0;
            while bits != 0 {
                if bits & 1 != 0 {
                    product ^= value.to_raw();
                }
                bits >>= 1;
                value = value.mul_x();
            }
            result[row][column] = F128::from_raw(product);
            row += 1;
        }
        column += 1;
    }
    result
}

/// Evaluate monomial coefficients at `F128::from_raw(node)`.
/// An empty coefficient slice returns zero. Every degree costs zero field
/// multiplications: Horner's small-node products use `mul_x` and XOR. This is
/// for a value per round; a loop over pairs uses [`super::quadratic_at_nodes`]
/// or [`super::linear_at_nodes`] to share shifts across nodes. A node that is
/// not a constant costs a branch per bit of the node.
#[inline]
pub fn eval_at_node(coefficients: &[F128], node: u8) -> F128 {
    coefficients.iter().rev().fold(ZERO, |value, &coefficient| {
        coefficient + mul_node(value, node)
    })
}

#[inline]
const fn mul_node(mut value: F128, mut node: u8) -> F128 {
    let mut result = 0;
    while node != 0 {
        if node & 1 != 0 {
            result ^= value.to_raw();
        }
        node >>= 1;
        if node != 0 {
            value = value.mul_x();
        }
    }
    F128::from_raw(result)
}

/// Recover degree `2..=8` coefficients from `p(0)`, `p`'s leading
/// coefficient, `p(1)` and the `degree - 2` values at raw nodes `2..degree`.
/// Returns exactly `degree + 1` coefficients in ascending order. The vector is
/// the message's storage: pass it directly to `UnivariatePoly::new`, or borrow
/// it for `round_poly_from_q_coeffs`, without trimming or copying.
/// Runtime degrees need no const-generic dispatch or allocation by callers.
///
/// Fixed matrices are module constants assembled at compile time from Lagrange
/// bases and precomputed inverse denominators, using the const `F128::mul_x`.
/// Runs once per round and member, allocating only the returned message vector;
/// no call initializes a cache or performs inversion.
/// Multiplications by zero and one are skipped.
/// Degrees 2, 3, 4, 5, 6, 7, 8 cost respectively 0, 4, 8, 15, 23, 36, 48
/// field multiplications; subtracting the leading term uses shifts and XORs.
pub fn coefficients_from_nodes(
    degree: usize,
    at_zero: F128,
    leading: F128,
    at_one: F128,
    nodes: &[F128],
) -> Result<Vec<F128>, RoundError> {
    if !(2..=8).contains(&degree) {
        return Err(RoundError::Degree { degree });
    }
    let expected = degree - 2;
    if nodes.len() != expected {
        return Err(RoundError::Nodes {
            expected,
            actual: nodes.len(),
        });
    }
    let mut residuals = [ZERO; 7];
    residuals[0] = at_one + at_zero + leading;
    for (i, &value) in nodes.iter().enumerate() {
        let node = (i + 2) as u8;
        let mut leading_at_node = leading;
        for _ in 0..degree {
            leading_at_node = mul_node(leading_at_node, node);
        }
        residuals[i + 1] = value + at_zero + leading_at_node;
    }
    let mut coefficients = vec![ZERO; degree + 1];
    coefficients[0] = at_zero;
    coefficients[degree] = leading;
    for (row, entries) in MATRICES[degree - 2].iter().take(degree - 1).enumerate() {
        for (&entry, &value) in entries.iter().zip(&residuals).take(degree - 1) {
            if entry == ONE {
                coefficients[row + 1] += value;
            } else if entry != ZERO {
                coefficients[row + 1] += entry * value;
            }
        }
    }
    Ok(coefficients)
}
