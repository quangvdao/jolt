use super::program::Program;
use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::transcripts::Transcript;
use crate::zkvm::bytecode::PreprocessingError;
use crate::zkvm::proof_serialization::JoltProof;
use crate::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use crate::zkvm::verifier::JoltSharedPreprocessing;
use crate::zkvm::witness::{CommittedPolynomial, CycleMajorTraceBatch};
use crate::zkvm::{JoltCommitmentScheme, ProverDebugInfo};
use common::jolt_device::MemoryLayout;
use jolt_crypto::Bn254;
use jolt_dory::DoryScheme;
use jolt_field::{Field, Fr};
use jolt_openings::{BatchCommitmentSource, CommitmentScheme};
use jolt_transcript::Transcript as OpeningsTranscript;
use tracer::{JoltDevice, LazyTraceIterator};

#[allow(clippy::type_complexity)]
#[cfg(feature = "prover")]
pub fn preprocess(
    guest: &Program,
    max_trace_length: usize,
) -> Result<JoltProverPreprocessing<Fr, Bn254, DoryScheme>, PreprocessingError> {
    let (bytecode, memory_init, program_size, e_entry) = guest.decode();

    let mut memory_config = guest.memory_config;
    memory_config.program_size = Some(program_size);
    let memory_layout = MemoryLayout::new(&memory_config);
    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode,
        memory_layout,
        memory_init,
        max_trace_length,
        e_entry,
    )?;
    Ok(JoltProverPreprocessing::new(shared_preprocessing))
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
#[cfg(feature = "prover")]
pub fn prove<
    F: JoltField + Field,
    C: JoltCurve<F = F>,
    PCS: JoltCommitmentScheme<F, C>,
    FS: Transcript + OpeningsTranscript<Challenge = F>,
>(
    guest: &Program,
    inputs_bytes: &[u8],
    untrusted_advice_bytes: &[u8],
    trusted_advice_bytes: &[u8],
    trusted_advice_commitment: Option<PCS::Output>,
    trusted_advice_hint: Option<<PCS as CommitmentScheme>::OpeningHint>,
    output_bytes: &mut [u8],
    preprocessing: &JoltProverPreprocessing<F, C, PCS>,
) -> (
    JoltProof<F, C, PCS, FS>,
    JoltDevice,
    Option<ProverDebugInfo<F, FS, PCS>>,
)
where
    for<'challenge> &'challenge F::Challenge: Into<F>,
    for<'batch> CycleMajorTraceBatch<'batch, LazyTraceIterator>:
        BatchCommitmentSource<F, Id = CommittedPolynomial>,
{
    let prover = JoltCpuProver::<F, C, PCS, FS>::gen_from_elf(
        preprocessing,
        &guest.elf_contents,
        inputs_bytes,
        untrusted_advice_bytes,
        trusted_advice_bytes,
        trusted_advice_commitment,
        trusted_advice_hint,
        None,
    );
    let io_device = prover.program_io.clone();
    let (proof, debug_info) = prover.prove();
    output_bytes[..io_device.outputs.len()].copy_from_slice(&io_device.outputs);
    (proof, io_device, debug_info)
}
