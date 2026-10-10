//! Packed-bit sum-check kernels over `F128`.
//!
//! Sources share packed lane words, cycle words and digit columns. Cores bind
//! variables from the low index bit upwards and return monomial coefficients.
//! Each constructor and pass checks its input geometry with a typed error.
//! The `test-utils` feature provides seeded traces and a dense summation oracle.

#![forbid(unsafe_code)]

include!("src/lib.rs");
