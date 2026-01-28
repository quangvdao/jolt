//! Witness generation for Dory recursion proofs
//!
//! This module contains witness generators for various group operations:
//! - G1 addition and scalar multiplication
//! - G2 addition and scalar multiplication
//! - GT exponentiation and multiplication
//! - Multi-Miller loop (experimental)

pub mod g1_add;
pub mod g1_scalar_mul;
pub mod g2_add;
pub mod g2_scalar_mul;
pub mod gt_exp;
pub mod gt_mul;

pub use g1_add::G1AdditionSteps;
pub use g1_scalar_mul::ScalarMultiplicationSteps;
pub use g2_add::G2AdditionSteps;
pub use g2_scalar_mul::G2ScalarMultiplicationSteps;
pub use gt_exp::Base4ExponentiationSteps;
pub use gt_mul::MultiplicationSteps;
