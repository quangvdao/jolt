//! Constraint system management
//!
//! This module contains the constraint system infrastructure:
//! - Constraint config: Central configuration for constraint variable counts
//! - Constraint system: Matrix building and constraint batching
//! - Constraint list sumcheck: Generic sumcheck over a list of constraints

pub mod config;
pub mod sumcheck;
pub mod system;

pub use config::{ConstraintSystemConfig, CONFIG};
pub use sumcheck::{
    sequential_opening_specs, ConstraintListProver, ConstraintListProverSpec, ConstraintListSpec,
    ConstraintListVerifier, ConstraintListVerifierSpec, OpeningSpec,
};
pub use system::{
    ConstraintLocator, ConstraintSystem, ConstraintType, G1AddNative, G1ScalarMulNative,
    G2AddNative, G2ScalarMulNative, GtMulNativeRows, PolyType, RecursionMatrixShape,
    RecursionMetadataBuilder,
};
