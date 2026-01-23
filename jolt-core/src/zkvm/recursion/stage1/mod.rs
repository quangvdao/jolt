//! Stage 1: Packed GT Exponentiation Sumcheck
//!
//! This stage proves GT exponentiation constraints in a packed base-4 form.
//! All other constraint sumchecks are batched in Stage 2.

pub mod gt_exp;

pub use gt_exp::{PackedGtExpProver, PackedGtExpVerifier};
