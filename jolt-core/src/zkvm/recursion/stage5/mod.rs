//! Stage 5: Jagged Assist Sumcheck
//!
//! Batch verification protocol that reduces verifier cost for evaluating multiple
//! polynomial openings from O(K × bits) to O(bits) operations.

pub mod jagged_assist;

pub use jagged_assist::{
    JaggedAssistEvalPoint, JaggedAssistParams, JaggedAssistProof, JaggedAssistProver,
    JaggedAssistVerifier,
};
