//! Binary-field RV64I bit-table commitment protocol.
//!
//! The committed bits are packed in pairs of `F64` coefficients in `F192`.
//! A position-major additive code is authenticated by a BLAKE2s Merkle root;
//! a shared out-of-domain lane sample selects at most one nearby message under
//! the conditional ledger. Opening reduces the `F128` partial evaluations by
//! a bit-transposition bridge, folds low lane variables by sumcheck, and adds
//! sampled and authenticated query claims at each level. The final message is
//! checked directly against the last oracle, then the closing identity.
//! Messages precede their challenges; wire lengths and typed shapes are checked
//! before indexing or allocation, with geometry checked first.
//!
//! Parameters, transcript order, wire encoding and conditional security claims
//! are specified in the [commitment specification](../../../../specs/rv64i-binary-commitment.md),
//! sections 2–6 and 8.

pub mod bridge;
pub mod error;
pub mod merkle;
pub mod params;
