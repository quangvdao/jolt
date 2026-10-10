//! Fixed-degree interpolation at polynomial-basis nodes, not integer ring images.

use super::RoundError;
use jolt_field::F128;
use jolt_poly::lagrange::interpolate_nodes_to_coeffs;
use std::sync::LazyLock;

type Matrix = [[F128; 7]; 7];
const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);

// F128 multiplication and inversion are not const; each fixed matrix is built
// once by the shared interpolator, outside the per-call reconstruction loop.
#[expect(
    clippy::expect_used,
    reason = "fixed distinct raw nodes satisfy the shared interpolator's contract"
)]
static MATRICES: LazyLock<[Matrix; 7]> = LazyLock::new(|| {
    std::array::from_fn(|offset| {
        let degree = offset + 2;
        let nodes: Vec<_> = (0..degree).map(|i| F128::from_raw(i as u128)).collect();
        let mut matrix = [[ZERO; 7]; 7];
        for column in 0..degree - 1 {
            let mut values = vec![ZERO; degree];
            values[column + 1] = ONE;
            let coefficients = interpolate_nodes_to_coeffs(&nodes, &values)
                .expect("raw nodes 0..degree are distinct and have exactly degree values");
            for row in 0..degree - 1 {
                matrix[row][column] = coefficients[row + 1];
            }
        }
        matrix
    })
});

/// Evaluate monomial coefficients at `F128::from_raw(node)`.
/// Linear and quadratic factors use the same entry point. Every degree costs
/// zero field multiplications: Horner's small-node products use `mul_x` and XOR.
#[inline]
pub fn eval_at_node(coefficients: &[F128], node: u8) -> F128 {
    coefficients.iter().rev().fold(ZERO, |value, &coefficient| {
        coefficient + mul_node(value, node)
    })
}

#[inline]
fn mul_node(mut value: F128, mut node: u8) -> F128 {
    let mut result = ZERO;
    while node != 0 {
        if node & 1 != 0 {
            result += value;
        }
        node >>= 1;
        if node != 0 {
            value = value.mul_x();
        }
    }
    result
}

/// Recover degree `2..=8` coefficients from `p(0)`, `p`'s leading
/// coefficient, `p(1)` and the `degree - 2` values at raw nodes `2..degree`.
/// Returns coefficients in ascending order; entries above `degree` are zero.
/// Runtime degrees need no const-generic dispatch or allocation by callers.
///
/// Matrices are cached in a `LazyLock`, because F128 multiplication/inversion
/// are not const. Its first use builds all seven matrices with the shared
/// interpolator; later calls allocate nothing and perform no inversion.
/// Multiplications by zero and one are skipped. Excluding initialization,
/// degrees 2, 3, 4, 5, 6, 7, 8 cost respectively 0, 4, 8, 15, 23, 36, 48
/// field multiplications; subtracting the leading term uses shifts and XORs.
pub fn coefficients_from_nodes(
    degree: usize,
    at_zero: F128,
    leading: F128,
    at_one: F128,
    nodes: &[F128],
) -> Result<[F128; 9], RoundError> {
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
    let mut coefficients = [ZERO; 9];
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
