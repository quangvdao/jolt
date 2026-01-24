//! Constraint system management
//!
//! This module contains the constraint system infrastructure:
//! - Constraint config: Central configuration for constraint variable counts
//! - Constraint system: Matrix building and constraint batching
//! - Constraint list sumcheck: Generic sumcheck over a list of constraints

pub mod constraint_config;
pub mod constraint_list_sumcheck;
pub mod constraints_sys;

pub use constraint_config::{ConstraintSystemConfig, CONFIG};
pub use constraint_list_sumcheck::{
    sequential_opening_specs, ConstraintListProver, ConstraintListProverSpec, ConstraintListSpec,
    ConstraintListVerifier, ConstraintListVerifierSpec, OpeningSpec,
};
pub use constraints_sys::{
    ConstraintSystem, ConstraintType, DoryMatrixBuilder, PolyType, RecursionMetadataBuilder,
};
