//! Routers fold source bits under selectors, prove the complete fold in shared slots and reduce each fold over cycles.

pub mod claims;
pub mod columns;
pub mod cycle;
pub mod fold;
pub mod lift;
pub mod shape;
pub mod short;

pub use columns::routed_columns;
