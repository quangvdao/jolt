//! Recursion SNARK implementation for Dory commitment scheme
//!
//! This module implements a multi-stage SNARK protocol with optimizations for proving
//! recursion constraints that arise from the Dory polynomial commitment scheme.
//!
//! ## Protocol Overview
//!
//! The recursion SNARK consists of five stages plus PCS opening:
//!
//! ### Stage 1: Packed GT Exp Sumcheck
//! Proves constraints for pairing group exponentiation in a packed base-4 form.
//!
//! ### Stage 2: Batched Constraint Sumchecks
//! Shift + reduction + GT mul + G1/G2 scalar mul + G1/G2 add, sharing challenges.
//!
//! ### Stage 3: Direct Evaluation
//! Verifies virtual polynomial claims using direct evaluation over the matrix.
//!
//! ### Stage 4: Jagged Transform
//! Uses sumcheck to open the sparse constraint matrix at a random point.
//!
//! ### Stage 5: Jagged Assist (Optimization)
//! Batch verification protocol that reduces verifier cost for evaluating multiple
//! polynomial openings from O(K × bits) to O(bits) operations.
//!
//! ## Module Structure
//! - `constraints_sys`: Constraint system management and matrix building
//! - `constraint_config`: Central configuration for constraint variable counts
//! - `stage1/`: Stage 1 - Packed GT exponentiation sumcheck
//! - `stage2/`: Stage 2 - Batched constraint sumchecks
//! - `stage3/`: Stage 3 - Direct evaluation (virtualization)
//! - `stage4/`: Stage 4 - Jagged transform sumcheck
//! - `stage5/`: Stage 5 - Jagged assist sumcheck
//! - `utils/`: Shared utilities and helpers
//! - `recursion_prover`: Unified prover orchestrating all stages
//! - `recursion_verifier`: Unified verifier for the complete protocol

pub mod bijection;
pub mod constraint_config;
pub mod constraints_sys;
pub mod curve;
pub mod recursion_prover;
pub mod recursion_verifier;
pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod stage4;
pub mod stage5;
pub mod utils;
pub mod witness;

#[cfg(test)]
mod tests;

// Re-export main types
pub use bijection::{ConstraintMapping, JaggedTransform, VarCountJaggedBijection};
pub use constraint_config::{ConstraintSystemConfig, CONFIG};
pub use constraints_sys::{
    ConstraintSystem, ConstraintType, DoryMatrixBuilder, PolyType, RecursionMetadataBuilder,
};
pub use recursion_prover::{RecursionProof, RecursionProver};
pub use recursion_verifier::{RecursionVerifier, RecursionVerifierInput};

// Stage 1 exports
pub use stage1::gt_exp::{PackedGtExpProver, PackedGtExpVerifier};

// Stage 2 exports
pub use stage2::{
    g1_scalar_mul::{G1ScalarMulProver, G1ScalarMulVerifier},
    g2_scalar_mul::{G2ScalarMulProver, G2ScalarMulVerifier},
    gt_mul::{GtMulProver, GtMulVerifier},
    packed_gt_exp_reduction::{
        PackedGtExpClaimReductionParams, PackedGtExpClaimReductionProver,
        PackedGtExpClaimReductionVerifier,
    },
    shift_rho::{ShiftClaim, ShiftRhoParams, ShiftRhoProver, ShiftRhoVerifier},
};

// Stage 3 exports
pub use stage3::virtualization::{
    extract_virtual_claims_from_accumulator, DirectEvaluationParams, DirectEvaluationProver,
    DirectEvaluationVerifier,
};

// Stage 4 exports
pub use stage4::jagged::{JaggedSumcheckParams, JaggedSumcheckProver, JaggedSumcheckVerifier};

// Stage 5 exports
pub use stage5::jagged_assist::{
    JaggedAssistEvalPoint, JaggedAssistParams, JaggedAssistProof, JaggedAssistProver,
    JaggedAssistVerifier,
};

pub use witness::{
    DoryRecursionWitness, G1ScalarMulWitness, GTExpWitness, GTMulWitness, WitnessData,
};
