//! Dense ground truth for multilinear extensions and sum-check round messages.
//!
//! Tables use low-bit-first coordinates. Each leaf is copied and bound directly
//! by `f_0 + r * (f_0 + f_1)`. A round evaluates the caller's summand at distinct
//! raw `F128` elements and interpolates; it shares no packed representation or
//! round assembly with the kernels.

use jolt_field::F128;
use jolt_poly::lagrange::{interpolate_nodes_to_coeffs, LagrangeNodesError};
use jolt_poly::UnivariatePoly;
use thiserror::Error;

/// Invalid dense oracle geometry or interpolation inputs.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OracleError {
    #[error("leaf {leaf} length {length} is not a nonzero power of two")]
    TableLength { leaf: usize, length: usize },
    #[error("leaf {leaf} has length {actual}, expected {expected}")]
    LeafLength {
        leaf: usize,
        expected: usize,
        actual: usize,
    },
    #[error("a round requires at least one leaf")]
    EmptyLeaves,
    #[error("point length {actual} differs from table dimension {expected}")]
    PointLength { expected: usize, actual: usize },
    #[error("bound point length {bound} leaves no round in dimension {variables}")]
    NoRound { bound: usize, variables: usize },
    #[error("degree {degree} exceeds the supported maximum of 8")]
    DegreeOutOfRange { degree: usize },
    #[error(transparent)]
    Interpolation(#[from] LagrangeNodesError),
}

/// Evaluate a dense multilinear table at a point, low index bit first.
/// Rejects non-power-of-two tables and points with the wrong dimension.
pub fn mle_at(table: &[F128], point: &[F128]) -> Result<F128, OracleError> {
    let variables = table_variables(table, 0)?;
    if point.len() != variables {
        return Err(OracleError::PointLength {
            expected: variables,
            actual: point.len(),
        });
    }
    let mut bound = table.to_vec();
    for &r in point {
        bind(&mut bound, r);
    }
    Ok(bound[0])
}

/// Compute `sum_x summand(leaves(bound, X, x))` in coefficient form.
///
/// Every leaf spans the same Boolean cube; `bound` fixes its low variables.
/// The caller states the summand and its degree in the next variable. Exactly
/// `degree + 1` distinct polynomial-basis field elements are used as nodes.
/// Degrees above eight, the maximum kernel degree, are rejected before allocation.
pub fn round_polynomial(
    leaves: &[&[F128]],
    bound: &[F128],
    degree: usize,
    summand: impl Fn(&[F128]) -> F128,
) -> Result<UnivariatePoly<F128>, OracleError> {
    let first = leaves.first().ok_or(OracleError::EmptyLeaves)?;
    let variables = table_variables(first, 0)?;
    for (leaf, table) in leaves.iter().enumerate().skip(1) {
        let _ = table_variables(table, leaf)?;
        if table.len() != first.len() {
            return Err(OracleError::LeafLength {
                leaf,
                expected: first.len(),
                actual: table.len(),
            });
        }
    }
    if bound.len() >= variables {
        return Err(OracleError::NoRound {
            bound: bound.len(),
            variables,
        });
    }
    if degree > 8 {
        return Err(OracleError::DegreeOutOfRange { degree });
    }
    let node_count = degree + 1;
    let mut tables: Vec<Vec<F128>> = leaves.iter().map(|leaf| leaf.to_vec()).collect();
    for table in &mut tables {
        for &r in bound {
            bind(table, r);
        }
    }
    let nodes: Vec<F128> = (0..node_count)
        .map(|raw| F128::from_raw(raw as u128))
        .collect();
    let mut values = vec![F128::from_raw(0); node_count];
    let mut leaf_values = vec![F128::from_raw(0); tables.len()];
    for (&node, value) in nodes.iter().zip(&mut values) {
        for row in 0..(tables[0].len() / 2) {
            for (leaf, table) in leaf_values.iter_mut().zip(&tables) {
                let lo = table[2 * row];
                *leaf = lo + node * (lo + table[2 * row + 1]);
            }
            *value += summand(&leaf_values);
        }
    }
    Ok(UnivariatePoly::new(interpolate_nodes_to_coeffs(
        &nodes, &values,
    )?))
}

fn table_variables(table: &[F128], leaf: usize) -> Result<usize, OracleError> {
    if !table.len().is_power_of_two() {
        return Err(OracleError::TableLength {
            leaf,
            length: table.len(),
        });
    }
    Ok(table.len().ilog2() as usize)
}

fn bind(table: &mut Vec<F128>, r: F128) {
    let half = table.len() / 2;
    for row in 0..half {
        let lo = table[2 * row];
        table[row] = lo + r * (lo + table[2 * row + 1]);
    }
    table.truncate(half);
}
