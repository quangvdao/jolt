//! Jagged polynomial transform and related protocols
//!
//! This module implements the jagged-to-dense polynomial bijection transform
//! and related sumcheck protocols following "Jagged Polynomial Commitments" paper.
//!
//! - Bijection: Generic jagged-to-dense polynomial bijection transform
//! - Branching program: Width-4 read-once branching program for efficient g-function evaluation
//! - Jagged: Jagged transform sumcheck (sparse to dense reduction)
//! - Jagged assist: Batch MLE verification optimization (Theorem 1.5)

pub mod assist;
pub mod bijection;
pub mod branching_program;
pub mod sumcheck;

pub use assist::{
    JaggedAssistEvalPoint, JaggedAssistParams, JaggedAssistProof, JaggedAssistProver,
    JaggedAssistVerifier,
};
pub use bijection::{ConstraintMapping, JaggedTransform, VarCountJaggedBijection};
pub use branching_program::{
    bit_to_field, get_coordinate_info, CoordType, JaggedBranchingProgram, Point,
};
pub use sumcheck::{JaggedSumcheckParams, JaggedSumcheckProver, JaggedSumcheckVerifier};
