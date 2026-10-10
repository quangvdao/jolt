use common::jolt_device::MemoryLayoutError;
use jolt_program::{execution::TraceError, ProgramError};
use jolt_rv64i_arith::{BytecodeError, LayoutError, RowFailure, WitnessError};
use thiserror::Error;

/// The stage that failed when checking a repeated stall fact.
#[derive(Debug, Error)]
pub enum StallCause {
    #[error(transparent)]
    Bits(#[from] WitnessError),
    #[error(transparent)]
    Row(#[from] RowFailure),
}

/// Decode, layout, identity and padding failures. Cycle errors name the earliest
/// failing absolute cycle regardless of parallel scheduling.
#[derive(Debug, Error)]
pub enum AdapterError {
    #[error(transparent)]
    Program(#[from] ProgramError),
    #[error(transparent)]
    Trace(#[from] TraceError),
    #[error(transparent)]
    MemoryLayout(#[from] MemoryLayoutError),
    #[error(transparent)]
    Bytecode(#[from] BytecodeError),
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error("bytecode row {index} at {pc:#x} is invalid")]
    InvalidBytecodeRow { index: usize, pc: u64 },
    #[error("the trace is empty")]
    EmptyTrace,
    #[error("trace length {rows} cannot be padded on this platform")]
    TraceTooLarge { rows: usize },
    #[error("allocating {facts} cycle facts failed")]
    Allocation { facts: usize },
    #[error("cycle {cycle} starts at {pc:#x}, not entry {entry_pc:#x}")]
    EntryPc {
        cycle: usize,
        pc: u64,
        entry_pc: u64,
    },
    #[error("cycle {cycle} at {pc:#x} has instruction index {index}")]
    InstructionIndex { cycle: usize, pc: u64, index: u32 },
    #[error("cycle {cycle} accesses {address:#x} below RAM")]
    AddressBelowRam { cycle: usize, address: u64 },
    #[error("cycle {cycle} has unaligned word address {address:#x}")]
    UnalignedWordAddress { cycle: usize, address: u64 },
    #[error("cycle {cycle} at {pc:#x} does not stall")]
    NoStall { cycle: usize, pc: u64 },
    #[error("cycle {cycle} at {pc:#x} cannot repeat: {cause}")]
    StallNotFixedPoint {
        cycle: usize,
        pc: u64,
        cause: StallCause,
    },
}
