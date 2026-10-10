//! Weighted committed-bit tables and their low-variable-first claim reduction.

mod core;
mod map;
mod tables;

pub use core::{ReductionCore, ReductionLeg};
pub use map::ColumnMap;
pub use tables::g_pass_digits;

use crate::round::RoundError;
use jolt_field::F128;
use thiserror::Error;

/// Rejected reduction geometry or a diagnostic claim mismatch.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ReductionError {
    #[error("{count} weight vectors exceed the maximum of four")]
    WeightCount { count: usize },
    #[error("{count} reduction legs exceed the maximum of eight")]
    LegCount { count: usize },
    #[error("table {table} has invalid length {actual}; expected {expected:?}")]
    TableLength {
        table: usize,
        actual: usize,
        expected: Option<usize>,
    },
    #[error("leg {leg} selects table {table} outside {tables} tables")]
    LegTable {
        leg: usize,
        table: usize,
        tables: usize,
    },
    #[error("leg {leg} point has {actual} coordinates, expected {expected}")]
    LegPoint {
        leg: usize,
        actual: usize,
        expected: usize,
    },
    #[error("weight {weight} has {actual} columns, expected 256")]
    WeightLength { weight: usize, actual: usize },
    #[error("map range at {start} of length {length} exceeds 256 columns")]
    MapRange { start: usize, length: usize },
    #[error("map ranges overlap at column {column}")]
    MapOverlap { column: usize },
    #[error("map trace word {trace_word} is outside {words} words")]
    MapTraceWord { trace_word: usize, words: usize },
    #[error("map digit column {column} is outside {columns} columns")]
    MapColumn { column: usize, columns: usize },
    #[error("flag group of {count} columns has invalid column and width {offending_column:?}")]
    MapFlags {
        count: usize,
        offending_column: Option<(usize, usize)>,
    },
    #[error("weight {weight} is nonzero on uncovered column {column}")]
    Uncovered { weight: usize, column: usize },
    #[error("leg {leg} claims {expected:?}, but its table gives {actual:?}")]
    Claim {
        leg: usize,
        expected: F128,
        actual: F128,
    },
    #[error("reduction values require completed rounds")]
    Unfinished,
    #[error(transparent)]
    Round(#[from] RoundError),
}
