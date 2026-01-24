//! Witness generation for Dory recursion proofs
//!
//! This module contains witness generators for various group operations:
//! - G1 scalar multiplication
//! - G2 scalar multiplication
//! - GT exponentiation
//! - GT multiplication
//! - Multi-Miller loop (experimental)

pub mod g1_scalar_mul;
pub mod g2_scalar_mul;
pub mod gt_exp;
pub mod gt_mul;
#[cfg(feature = "experimental-pairing-recursion")]
pub mod multi_miller_loop;

pub use g1_scalar_mul::ScalarMultiplicationSteps;
pub use g2_scalar_mul::G2ScalarMultiplicationSteps;
pub use gt_exp::Base4ExponentiationSteps;
pub use gt_mul::MultiplicationSteps;
#[cfg(feature = "experimental-pairing-recursion")]
pub use multi_miller_loop::MultiMillerLoopSteps;
