//! Preamble grammar: padded 32-byte labels and words, with big-endian counts
//! and word values, then raw input, output and program-digest bodies. Each body,
//! including an empty segment, is one append call. The 36 calls precede the
//! scheme's commit phase; the initial domain is `jolt-rv64i-binary-v0`.

use crate::{commitment::BitsCommitmentScheme, statement::CheckedInputs};
use jolt_field::F128;
use jolt_transcript::{Blake2bTranscript, Label, LabelWithCount, Transcript, U64Word};

/// The binary-field transcript used by this protocol's reference front end.
pub type Rv64iTranscript = Blake2bTranscript<F128>;
/// Version-zero domain separator; changing the fixed preamble grammar requires a new domain.
pub const PROTOCOL_LABEL: &[u8] = b"jolt-rv64i-binary-v0";

/// Creates and absorbs the checked front-end preamble, without drawing a challenge.
pub fn preamble<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
) -> T {
    let mut transcript = T::new(PROTOCOL_LABEL);
    append_preamble(checked, &mut transcript);
    transcript
}

/// Absorbs the 36 checked preamble bodies in their wire order, including empty public segments, without drawing a challenge.
/// The caller initializes the transcript with `PROTOCOL_LABEL` and performs the scheme's commit phase immediately afterwards.
pub fn append_preamble<S: BitsCommitmentScheme, T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_, S>,
    transcript: &mut T,
) {
    let statement = checked.statement();
    let device = &statement.device;
    let m = &device.memory_layout;
    transcript.append(&Label(b"params"));
    for word in [
        checked.log_T() as u64,
        checked.log_K_bytecode() as u64,
        checked.log_K_ram() as u64,
        checked.layout().lowest_address(),
    ] {
        transcript.append(&U64Word(word));
    }
    transcript.append(&Label(b"statement"));
    transcript.append(&U64Word(statement.entry_pc));
    for word in [
        m.program_size,
        m.max_trusted_advice_size,
        m.trusted_advice_start,
        m.trusted_advice_end,
        m.max_untrusted_advice_size,
        m.untrusted_advice_start,
        m.untrusted_advice_end,
        m.max_input_size,
        m.max_output_size,
        m.input_start,
        m.input_end,
        m.output_start,
        m.output_end,
        m.stack_size,
        m.stack_end,
        m.heap_size,
        m.heap_end,
        m.panic,
        m.termination,
        m.io_end,
    ] {
        transcript.append(&U64Word(word));
    }
    transcript.append(&LabelWithCount(b"inputs", device.inputs.len() as u64));
    transcript.append_bytes(&device.inputs);
    transcript.append(&LabelWithCount(b"outputs", device.outputs.len() as u64));
    transcript.append_bytes(&device.outputs);
    transcript.append(&U64Word(u64::from(device.panic)));
    transcript.append(&Label(b"program"));
    transcript.append_bytes(checked.preprocessing().digest());
    transcript.append(&Label(b"final_pc"));
    transcript.append(&U64Word(checked.final_pc()));
}
