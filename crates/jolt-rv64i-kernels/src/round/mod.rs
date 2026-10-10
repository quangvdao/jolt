//! Round messages use monomial coefficients from distinct binary-field nodes and a leading coefficient.

pub mod eq;
pub mod nodes;
pub mod product;

pub use nodes::{coefficients_from_nodes, eval_at_node};
pub use product::quadratic;

use thiserror::Error;

/// Invalid round interpolation or equality geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum RoundError {
    #[error("degree {degree} is outside 2..=8")]
    Degree { degree: usize },
    #[error("expected {expected} node values, got {actual}")]
    Nodes { expected: usize, actual: usize },
    #[error("split equality requires a nonempty point")]
    EmptyPoint,
}
