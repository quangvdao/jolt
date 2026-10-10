//! Adapts an RV64I program image and architectural execution rows to cycle
//! facts. Decoding, conversion and stall validation preserve instruction-list
//! identity; register and RAM replay belong to the consumer of the facts.
#![forbid(unsafe_code)]
#![expect(non_snake_case, reason = "dimension names follow the cycle layout")]

mod adapt;
mod error;
mod preprocess;

pub use adapt::{adapt, Execution};
pub use error::{AdapterError, StallCause};
pub use preprocess::{preprocess, Program};

#[cfg(feature = "emulator")]
use common::jolt_device::MemoryConfig;
#[cfg(feature = "emulator")]
use jolt_program::{
    execution::{OwnedTrace, SourceTraceRow, TraceOutput},
    image::DecodeMode,
};

/// Executes with the same decode mode used for preprocessing. The supplied
/// configuration must cover the loaded ELF; no advice is supplied.
#[cfg(feature = "emulator")]
pub fn trace(
    elf: &[u8],
    inputs: &[u8],
    memory_config: &MemoryConfig,
    decode: DecodeMode,
) -> Result<TraceOutput<OwnedTrace<SourceTraceRow>>, AdapterError> {
    use jolt_program::execution::{JoltProgram, TraceInputs};
    use jolt_riscv::RV64I;
    use tracer::SourceTracerBackend;
    let program = JoltProgram::from_elf_bytes_with_profile(elf.to_vec(), RV64I);
    let mut backend = SourceTracerBackend::default().with_decode_mode(decode);
    Ok(program.trace_with(
        &mut backend,
        TraceInputs::new(inputs.to_vec(), Vec::new(), Vec::new(), *memory_config),
    )?)
}
