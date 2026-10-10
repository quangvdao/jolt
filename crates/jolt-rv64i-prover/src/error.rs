//! Typed failures of witness construction and proving.
use common::jolt_device::MemoryLayoutError;
use jolt_field::F128;
use jolt_prover::ProverError;
use jolt_rv64i_arith::{BytecodeError, CycleError, LayoutError, WitnessError};
use jolt_rv64i_verifier::error::Rv64iVerifierError;
use jolt_rv64i_verifier::points::PointsError;
use std::collections::TryReserveError;
use std::error::Error as StdError;
use thiserror::Error;

/// The pre-state fact that disagrees with the witness replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactField {
    /// First source-register read.
    Rs1Value,
    /// Second source-register read.
    Rs2Value,
    /// Destination-register pre-value.
    RdPreValue,
    /// RAM pre-word on a cycle with an access.
    RamPreValue,
}

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
    #[error(
        "witness has {bits} bit rows and {words} word rows, expected {expected} rows in each table"
    )]
    RowCount {
        expected: usize,
        bits: usize,
        words: usize,
    },
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
    #[error("RAM allocation for exponent {log_K_ram} failed: {source}")]
    RamAllocation {
        log_K_ram: usize,
        source: TryReserveError,
    },
    #[error("cycle {cycle} fact {field:?} differs: expected {expected}, found {found}")]
    FactMismatch {
        cycle: usize,
        field: FactField,
        expected: u64,
        found: u64,
    },
    #[error("witness layout differs from the checked layout")]
    OutputLayoutMismatch,
    #[error("public I/O word {index} differs: expected {expected}, found {found}")]
    OutputMismatch {
        index: u64,
        expected: u64,
        found: u64,
    },
    #[error("final RAM has {found} words, expected {expected}")]
    FinalRamLength { expected: usize, found: usize },
    #[error("a constructor's new witness buffer is already shared")]
    SharedBuffer,
    #[error("RAM view with {words} words and {cycles} cycles cannot be represented on this host")]
    RamViewSize { words: usize, cycles: usize },
    #[error("RAM view allocation for {words} words and {cycles} cycles failed: {source}")]
    RamViewAllocation {
        words: usize,
        cycles: usize,
        source: TryReserveError,
    },
    #[error("store at cycle {cycle} selects RAM word {index} outside the RAM domain")]
    StoreRam { cycle: usize, index: u64 },
    #[error("RAM exponent {log_K_ram} cannot be represented on this host")]
    RamDimension { log_K_ram: usize },
    #[error("witness initial RAM differs from the checked initial RAM")]
    InitialRamMismatch,
    #[error("batch {batch} failed: {source}")]
    Batch {
        batch: &'static str,
        #[source]
        source: Box<Rv64iProverError>,
    },
    #[error("cycle {cycle} is absent from a witness table of {rows} rows")]
    CycleIndex { cycle: usize, rows: usize },
}

impl Rv64iProverError {
    pub(crate) fn in_batch(self, batch: &'static str) -> Self {
        Self::Batch {
            batch,
            source: Box::new(self),
        }
    }
}
