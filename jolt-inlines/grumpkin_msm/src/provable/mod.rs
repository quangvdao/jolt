//! Provable (sound) Grumpkin arithmetic and MSM on stable limb layouts.
//!
//! This module is **not** based on trace-time patching. All computations are performed in
//! guest Rust using fixed-width limb arithmetic modulo the Grumpkin base field.

pub mod curve;
pub mod fq;
pub mod msm;

pub use msm::grumpkin_msm_2048;

#[cfg(test)]
mod tests;
