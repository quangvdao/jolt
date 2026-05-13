use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::utils::errors::ProofVerifyError;
use crate::zkvm::bytecode::PreprocessingError;
use crate::zkvm::verifier::BlindfoldSetup;

use crate::guest::program::Program;
use crate::transcripts::Transcript;
use crate::zkvm::proof_serialization::JoltProof;
use crate::zkvm::verifier::JoltSharedPreprocessing;
use crate::zkvm::verifier::JoltVerifier;
use crate::zkvm::verifier::JoltVerifierPreprocessing;
use common::jolt_device::MemoryConfig;
use common::jolt_device::MemoryLayout;
use jolt_crypto::Bn254;
use jolt_dory::{DoryScheme, DoryVerifierSetup};
use jolt_field::Fr;

pub fn preprocess(
    guest: &Program,
    max_trace_length: usize,
    verifier_setup: DoryVerifierSetup,
    blindfold_setup: Option<BlindfoldSetup<Bn254>>,
) -> Result<JoltVerifierPreprocessing<Fr, Bn254, DoryScheme>, PreprocessingError> {
    let shared = preprocess_shared(guest, max_trace_length)?;
    Ok(JoltVerifierPreprocessing::new(
        shared,
        verifier_setup,
        blindfold_setup,
    ))
}

fn preprocess_shared(
    guest: &Program,
    max_trace_length: usize,
) -> Result<JoltSharedPreprocessing, PreprocessingError> {
    let (bytecode, memory_init, program_size, e_entry) = guest.decode();

    let mut memory_config = guest.memory_config;
    memory_config.program_size = Some(program_size);
    let memory_layout = MemoryLayout::new(&memory_config);
    JoltSharedPreprocessing::new(
        bytecode,
        memory_layout,
        memory_init,
        max_trace_length,
        e_entry,
    )
}

pub fn verify<
    F: JoltField + jolt_field::Field,
    C: JoltCurve<F = F>,
    PCS: crate::zkvm::JoltCommitmentScheme<F, C>,
    FS: Transcript + jolt_transcript::Transcript<Challenge = F>,
>(
    inputs_bytes: &[u8],
    trusted_advice_commitment: Option<PCS::Output>,
    outputs_bytes: &[u8],
    proof: JoltProof<F, C, PCS, FS>,
    preprocessing: &JoltVerifierPreprocessing<F, C, PCS>,
) -> Result<(), ProofVerifyError> {
    use common::jolt_device::JoltDevice;
    let memory_layout = &preprocessing.shared.memory_layout;
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
        max_trusted_advice_size: memory_layout.max_trusted_advice_size,
        max_input_size: memory_layout.max_input_size,
        max_output_size: memory_layout.max_output_size,
        stack_size: memory_layout.stack_size,
        heap_size: memory_layout.heap_size,
        program_size: Some(memory_layout.program_size),
    };
    let mut io_device = JoltDevice::new(&memory_config);

    io_device.inputs = inputs_bytes.to_vec();
    io_device.outputs = outputs_bytes.to_vec();

    let verifier = JoltVerifier::new(
        preprocessing,
        proof,
        io_device,
        trusted_advice_commitment,
        None,
    )?;
    verifier.verify()
}
