//! Stage 3: Direct Evaluation Protocol (Virtualization)
//!
//! This stage combines virtual claims into M(r_s, r_x) without a sumcheck.
//! It verifies the virtual polynomial claims from earlier stages.

pub mod virtualization;

pub use virtualization::{
    extract_virtual_claims_from_accumulator, DirectEvaluationParams, DirectEvaluationProver,
    DirectEvaluationVerifier,
};
