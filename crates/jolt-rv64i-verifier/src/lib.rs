//! An experiment in RV64I hash-based Jolt over binary fields.
//! Identifiers and points define the protocol geometry; claims hold symbolic
//! relations, stages hold concrete relations, and public data comes from checked
//! statements and preprocessing. Commitment, proof, and transcript modules own
//! the verifier boundary. Points and indices are least-significant-bit first;
//! sum-check binds low to high, with columns before cycles in the Bits table.

#![forbid(unsafe_code)]
#![expect(
    non_snake_case,
    reason = "protocol dimensions retain their mathematical names"
)]
#![deny(
    clippy::indexing_slicing,
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::wildcard_enum_match_arm
)]

pub mod claims;
pub mod commitment;
pub mod error;
pub mod ids;
pub mod points;
pub mod preprocessing;
pub mod proof;
pub mod public;
pub mod stages;
pub mod statement;
pub mod transcript;
pub mod verifier;
