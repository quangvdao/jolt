//! Bit-level constraint system for RV64I over the binary field, one cycle per
//! executed instruction.

// Layouts, bytecode rows and constraint rows are verifier inputs: the crate
// carries the lint set of the verifier closure (specs/verifier-closure-lints.md).
#![forbid(unsafe_code)]
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

pub mod bytecode;
pub mod cycle;
pub mod decode;
pub mod layout;
pub mod rows;
pub mod variant;
pub mod words;

pub use bytecode::{
    Bytecode, BytecodeColumn, BytecodeError, BytecodeRow, BytecodeRowError, BYTECODE_ROW_BITS,
};
pub use cycle::{BitsBuilder, CycleError, CycleFacts, WitnessError};
pub use decode::{
    eval, load_form, shift_form, store_form, Form, FormError, Line, Rails, ShortForm, Source,
    Sources, Term, TermError, Wire, BRANCH_FORM,
};
pub use layout::{chunk_indicators, BitsRow, Chunk, ChunkError, Layout, LayoutError, BITS_COLUMNS};
pub use rows::{
    LaneRows, PackedForm, PackedRow, PackedTerm, PackedTermError, RowFailure, RowGroup, RowSet,
    RowSystem,
};
pub use variant::{Access, AccessKind, BranchCondition, KeyKind, Shift, ShiftKind, Variant};
pub use words::{column, BaseWords, Lane, WitnessRow, Words, WITNESS_COLUMNS};
