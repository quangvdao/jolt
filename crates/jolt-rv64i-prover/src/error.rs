//! Typed failures of witness construction and proving.
use common::jolt_device::MemoryLayoutError;
use jolt_field::F128;
use jolt_prover::ProverError;
use jolt_rv64i_arith::{BytecodeError, CycleError, LayoutError, WitnessError};
use jolt_rv64i_verifier::error::Rv64iVerifierError;
use jolt_rv64i_verifier::points::PointsError;
use std::error::Error as StdError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Rv64iProverError {
    #[error(transparent)]
    Points(#[from] PointsError),
    #[error(transparent)]
    Prover(#[from] ProverError<F128>),
    #[error(transparent)]
    Verifier(#[from] Rv64iVerifierError),
    #[error(transparent)]
    Cycle(#[from] CycleError),
    #[error(transparent)]
    Witness(#[from] WitnessError),
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error(transparent)]
    Bytecode(#[from] BytecodeError),
    #[error(transparent)]
    MemoryLayout(#[from] MemoryLayoutError),
    #[error("witness row count {rows} is not a nonzero power of two")]
    RowCount { rows: usize },
    #[error("cycle {cycle} selects an invalid bytecode index {index}")]
    InvalidBytecode { cycle: usize, index: u64 },
    #[error("cycle {cycle} has multiple stored indicators in chunk {start}")]
    MultipleIndicators { cycle: usize, start: u16 },
    #[error("initial RAM is not canonical at word index {index}")]
    InitialRam { index: u64 },
    #[error("register selector {register} is outside the register table")]
    Register { register: u8 },
    #[error("synthetic witness requires a valid store row")]
    MissingStore,
    #[error("synthetic witness has no admissible ordinary store address")]
    StoreAddress,
    #[error("synthetic trace exponent {log_T} cannot be represented")]
    TraceDimension { log_T: usize },
    #[error("commitment scheme failed: {0}")]
    Scheme(Box<dyn StdError + Send + Sync>),
}
