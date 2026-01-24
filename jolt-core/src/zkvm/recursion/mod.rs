//! Recursion SNARK implementation for Dory commitment scheme
//!
//! This module implements a multi-stage SNARK protocol with optimizations for proving
//! recursion constraints that arise from the Dory polynomial commitment scheme.
//!
//! ## Protocol Overview
//!
//! The recursion SNARK consists of multiple sumcheck stages plus PCS opening:
//!
//! ### G1 Operations
//! Proves constraints for G1 addition and scalar multiplication.
//!
//! ### G2 Operations
//! Proves constraints for G2 addition and scalar multiplication.
//!
//! ### GT Operations
//! Proves constraints for GT exponentiation and multiplication.
//!
//! ### Pairing Operations
//! Proves Multi-Miller loop constraints.
//!
//! ### Virtualization (Direct Evaluation)
//! Verifies virtual polynomial claims using direct evaluation over the matrix.
//!
//! ### Jagged Transform
//! Uses sumcheck to open the sparse constraint matrix at a random point.
//!
//! ### Jagged Assist (Optimization)
//! Batch verification protocol that reduces verifier cost for evaluating multiple
//! polynomial openings from O(K × bits) to O(bits) operations.
//!
//! ## Module Structure
//! - `constraints/`: Constraint system management and configuration
//! - `g1/`: G1 curve operations (addition, scalar multiplication)
//! - `g2/`: G2 curve operations (addition, scalar multiplication)
//! - `gt/`: GT group operations (exponentiation, multiplication, claim reduction)
//! - `pairing/`: Pairing operations (multi-miller loop)
//! - `jagged/`: Jagged transform and assist protocols
//! - `virtualization`: Direct evaluation protocol
//! - `utils/`: Shared utilities and helpers
//! - `prover`: Unified prover orchestrating all stages
//! - `verifier`: Unified verifier for the complete protocol

pub mod constraints;
pub mod curve;
pub mod g1;
pub mod g2;
pub mod gt;
pub mod jagged;
pub mod pairing;
pub mod prover;
pub mod utils;
pub mod verifier;
pub mod virtualization;
pub mod witness;

#[cfg(test)]
mod tests;

// Re-export constraint types
pub use constraints::{
    ConstraintSystem, ConstraintSystemConfig, ConstraintType, DoryMatrixBuilder, PolyType,
    RecursionMetadataBuilder, CONFIG,
};

// Re-export jagged types
pub use jagged::{ConstraintMapping, JaggedTransform, VarCountJaggedBijection};

// Re-export prover/verifier
pub use prover::{RecursionProof, RecursionProofResult, RecursionProver};
pub use verifier::{RecursionVerifier, RecursionVerifierInput};

// G1 exports
pub use g1::{G1ScalarMulProver, G1ScalarMulVerifier};

// G2 exports
pub use g2::{G2ScalarMulProver, G2ScalarMulVerifier};

// GT exports
pub use gt::{GtMulProver, GtMulVerifier};
pub use gt::{
    GtExpClaimReductionParams, GtExpClaimReductionProver,
    GtExpClaimReductionVerifier, GtExpProver, GtExpVerifier,
};
pub use gt::{ShiftClaim, ShiftRhoParams, ShiftRhoProver, ShiftRhoVerifier};

// Virtualization exports
pub use virtualization::{
    extract_virtual_claims_from_accumulator, DirectEvaluationParams, DirectEvaluationProver,
    DirectEvaluationVerifier,
};

// Jagged exports
pub use jagged::{
    JaggedAssistEvalPoint, JaggedAssistParams, JaggedAssistProof, JaggedAssistProver,
    JaggedAssistVerifier,
};
pub use jagged::{JaggedSumcheckParams, JaggedSumcheckProver, JaggedSumcheckVerifier};

pub use witness::{
    DoryRecursionWitness, G1ScalarMulWitness, GTExpWitness, GTMulWitness, WitnessData,
};

/// Max `dense_num_vars` for recursion Hyrax setup.
///
/// This bounds the size of the dense polynomial opened via Hyrax in the recursion proof.
/// It is used to size cached Hyrax Pedersen generators in preprocessing (prover + verifier).
/// Supports up to \(2^{22}\) constraints (see `zkvm/verifier.rs` recursion verification path).
pub const MAX_RECURSION_DENSE_NUM_VARS: usize = 22;
