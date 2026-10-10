//! Typed failures at the statement, proof, preprocessing and commitment boundaries.

use std::error::Error as StdError;

use common::jolt_device::MemoryLayoutError;
use jolt_program::preprocess::RamDomainError;
use jolt_rv64i_arith::{BytecodeError, LayoutError};
use jolt_verifier::error::VerifierError;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PreprocessingError {
    #[error("program image repeats word index {index}")]
    RepeatedImageWord { index: u64 },
    #[error("program image word index {index} follows larger index {previous}")]
    ImageOutOfOrder { previous: u64, index: u64 },
    #[error("program image word {index} has zero value")]
    ZeroImageWord { index: u64 },
    #[error("preprocessing length is not representable by the wire format")]
    Length,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProofDecodeError {
    #[error("unsupported proof version")]
    InvalidVersion,
    #[error("proof dimensions are outside the protocol domain")]
    Dimensions,
    #[error("proof encoding is truncated")]
    Truncated,
    #[error("proof length exceeds its remaining encoding")]
    Length,
    #[error("commitment scheme rejected its wire encoding")]
    Scheme,
    #[error("proof encoding has trailing bytes")]
    TrailingBytes,
    #[error("proof has an invalid round or value shape")]
    ProofShape,
}

#[derive(Debug, Error)]
pub enum Rv64iVerifierError {
    #[error("trace exponent must be positive")]
    TraceExponentZero,
    #[error("trace exponent {log_T} exceeds the protocol limit")]
    TraceExponentTooLarge { log_T: u8 },
    #[error("RAM exponent {log_K_ram} is below five")]
    RamExponentTooSmall { log_K_ram: u8 },
    #[error("protocol layout is invalid: {0}")]
    Layout(#[source] LayoutError),
    #[error("RAM address range exceeds 2^64")]
    RamRangeOverflow,
    #[error("private advice must be empty")]
    AdviceNotEmpty,
    #[error("memory configuration is invalid: {0}")]
    MemoryLayout(#[source] MemoryLayoutError),
    #[error("memory layout is not its canonical construction")]
    NonCanonicalMemoryLayout,
    #[error("bytecode and memory layout have different lowest addresses")]
    LowestAddressMismatch,
    #[error("memory layout has no supported RAM bound: {0}")]
    RamBound(#[source] RamDomainError),
    #[error("RAM domain exceeds the memory layout bound")]
    RamTooLarge,
    #[error("public inputs exceed their memory region")]
    InputsTooLong,
    #[error("public outputs exceed their memory region")]
    OutputsTooLong,
    #[error("public I/O memory cannot be constructed: {0}")]
    PublicIo(#[source] MemoryLayoutError),
    #[error("public initial RAM cannot be constructed: {0}")]
    InitialRam(#[source] MemoryLayoutError),
    #[error("public I/O word range exceeds the RAM domain")]
    IoRangeTooLarge,
    #[error("program image word {index} is inside the I/O range")]
    ImageInsideIo { index: u64 },
    #[error("program image word {index} is outside the RAM domain")]
    ImageOutsideRam { index: u64 },
    #[error("final PC is invalid: {0}")]
    FinalPc(#[source] BytecodeError),
    #[error("proof shape is invalid: {0}")]
    ProofShape(#[source] ProofDecodeError),
    #[error("proof decoding failed: {0}")]
    ProofDecode(#[from] ProofDecodeError),
    #[error("sumcheck verification failed: {0}")]
    Verifier(#[from] VerifierError),
    #[error("commitment phase failed: {0}")]
    CommitPhase(#[source] Box<dyn StdError + Send + Sync>),
    #[error("Bits opening failed: {0}")]
    Opening(#[source] Box<dyn StdError + Send + Sync>),
}
