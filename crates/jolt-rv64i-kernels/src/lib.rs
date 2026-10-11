//! Packed-bit sum-check kernels over `F128`.
//!
//! Sources share packed lane words, cycle words and digit columns. Cores bind
//! variables from the low index bit upwards and return monomial coefficients.
//! Each constructor and pass checks its input geometry with a typed error.
//! The `test-utils` feature provides seeded traces and a dense summation oracle.

#![forbid(unsafe_code)]

pub mod chunk_product;
pub mod column_pass;
pub mod outer_f2;
pub mod packed;
pub mod pair_sum;
pub mod par;
pub mod reduction;
pub mod round;
pub mod router;
pub mod source;

#[cfg(feature = "test-utils")]
pub mod oracle;
#[cfg(feature = "test-utils")]
pub mod synth;
