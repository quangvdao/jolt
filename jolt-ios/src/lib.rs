use std::{
    os::raw::{c_char, c_uchar},
    panic::{catch_unwind, AssertUnwindSafe},
    slice,
};

use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryConfig;
use jolt_core::{
    guest::{program::Program, prover, verifier},
    poly::commitment::dory::DoryCommitmentScheme,
    transcripts::{Blake2bTranscript, KeccakTranscript, Transcript},
    zkvm::proof_serialization::JoltProof,
};

type F = Fr;
type PCS = DoryCommitmentScheme;

const DEFAULT_MEMORY_SIZE: u64 = 16 * 1024 * 1024;
const DEFAULT_STACK_SIZE: u64 = 4 * 1024 * 1024;
const DEFAULT_TRACE_LENGTH: usize = 1 << 20;
const DEFAULT_MAX_OUTPUT_SIZE: usize = 4 * 1024 * 1024;

/// Status codes for FFI operations
#[repr(C)]
pub enum JoltStatusCode {
    Success = 0,
    ErrorInvalidInput = 1,
    ErrorProofGeneration = 2,
    ErrorVerification = 3,
    ErrorSerialization = 4,
    ErrorOutOfMemory = 5,
    ErrorNotImplemented = 99,
}

/// Result structure for proof generation
#[repr(C)]
pub struct JoltProofResult {
    pub status: JoltStatusCode,
    pub cycles: u64,
    pub proof_data: *mut u8,
    pub proof_len: usize,
    pub output_data: *mut u8,
    pub output_len: usize,
}

impl JoltProofResult {
    fn success(cycles: u64, proof: Vec<u8>, outputs: Vec<u8>) -> Self {
        let (proof_data, proof_len) = leak_vec(proof);
        let (output_data, output_len) = leak_vec(outputs);
        Self {
            status: JoltStatusCode::Success,
            cycles,
            proof_data,
            proof_len,
            output_data,
            output_len,
        }
    }

    const fn failure(status: JoltStatusCode) -> Self {
        Self {
            status,
            cycles: 0,
            proof_data: std::ptr::null_mut(),
            proof_len: 0,
            output_data: std::ptr::null_mut(),
            output_len: 0,
        }
    }
}

/// Result structure for verification
#[repr(C)]
pub struct JoltVerifyResult {
    pub status: JoltStatusCode,
    pub is_valid: bool,
}

impl JoltVerifyResult {
    const fn success(is_valid: bool) -> Self {
        Self {
            status: JoltStatusCode::Success,
            is_valid,
        }
    }

    const fn failure(status: JoltStatusCode) -> Self {
        Self {
            status,
            is_valid: false,
        }
    }
}

/// Transcript choice for prover/verifier
#[repr(u32)]
pub enum JoltTranscriptFlavor {
    Blake2b = 0,
    Keccak = 1,
}

impl Default for JoltTranscriptFlavor {
    fn default() -> Self {
        Self::Blake2b
    }
}

/// Free memory allocated by Jolt
///
/// # Safety
/// - `ptr` must have been allocated by this library
/// - `ptr` and `len` must match a previous allocation
/// - Must only be called once per allocation
#[no_mangle]
pub unsafe extern "C" fn jolt_free_buffer(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// Generate a proof for the supplied ELF and inputs.
///
/// All buffers passed to this function must remain valid for the duration of the call.
/// The returned proof and output buffers are owned by the caller, who must free them
/// with [`jolt_free_buffer`].
#[no_mangle]
pub unsafe extern "C" fn jolt_prove(
    elf_bytes: *const c_uchar,
    elf_len: usize,
    input_bytes: *const c_uchar,
    input_len: usize,
    max_input_size: usize,
    max_output_size: usize,
    max_trace_length: usize,
    transcript: JoltTranscriptFlavor,
) -> JoltProofResult {
    match catch_unwind(AssertUnwindSafe(|| {
        prove_impl(
            elf_bytes,
            elf_len,
            input_bytes,
            input_len,
            max_input_size,
            max_output_size,
            max_trace_length,
            transcript,
        )
    })) {
        Ok(result) => result,
        Err(_) => JoltProofResult::failure(JoltStatusCode::ErrorProofGeneration),
    }
}

/// Verify a proof.
///
/// The caller must pass the same transcript flavor, memory limits, and ELF that were used
/// when generating the proof.
#[no_mangle]
pub unsafe extern "C" fn jolt_verify(
    proof_data: *const c_uchar,
    proof_len: usize,
    elf_bytes: *const c_uchar,
    elf_len: usize,
    input_bytes: *const c_uchar,
    input_len: usize,
    output_bytes: *const c_uchar,
    output_len: usize,
    max_input_size: usize,
    max_output_size: usize,
    max_trace_length: usize,
    transcript: JoltTranscriptFlavor,
) -> JoltVerifyResult {
    match catch_unwind(AssertUnwindSafe(|| {
        verify_impl(
            proof_data,
            proof_len,
            elf_bytes,
            elf_len,
            input_bytes,
            input_len,
            output_bytes,
            output_len,
            max_input_size,
            max_output_size,
            max_trace_length,
            transcript,
        )
    })) {
        Ok(result) => result,
        Err(_) => JoltVerifyResult::failure(JoltStatusCode::ErrorVerification),
    }
}

/// Get the version string
///
/// Returns a null-terminated C string. Do NOT free this string.
#[no_mangle]
pub extern "C" fn jolt_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Get a description of the current build configuration
///
/// Returns a null-terminated C string. Do NOT free this string.
#[no_mangle]
pub extern "C" fn jolt_build_info() -> *const c_char {
    concat!(
        "jolt-ios v",
        env!("CARGO_PKG_VERSION"),
        " (guest prover API)\0"
    )
    .as_ptr() as *const c_char
}

fn prove_impl(
    elf_bytes: *const c_uchar,
    elf_len: usize,
    input_bytes: *const c_uchar,
    input_len: usize,
    max_input_size: usize,
    max_output_size: usize,
    max_trace_length: usize,
    transcript: JoltTranscriptFlavor,
) -> JoltProofResult {
    if elf_bytes.is_null() || elf_len == 0 {
        return JoltProofResult::failure(JoltStatusCode::ErrorInvalidInput);
    }
    if max_input_size == 0 || input_len > max_input_size {
        return JoltProofResult::failure(JoltStatusCode::ErrorInvalidInput);
    }

    let trace_length = match resolve_trace_length(max_trace_length) {
        Ok(len) => len,
        Err(status) => return JoltProofResult::failure(status),
    };

    let (output_cap, output_cap_u64) = match resolve_output_size(max_output_size) {
        Ok(values) => values,
        Err(status) => return JoltProofResult::failure(status),
    };

    let elf = unsafe { slice::from_raw_parts(elf_bytes, elf_len) };
    let inputs = unsafe { read_optional_slice(input_bytes, input_len) };
    let inputs = match inputs {
        Ok(slice) => slice,
        Err(status) => return JoltProofResult::failure(status),
    };

    let memory_config = match build_memory_config(max_input_size, output_cap_u64) {
        Ok(cfg) => cfg,
        Err(status) => return JoltProofResult::failure(status),
    };

    let program = Program::new(elf, &memory_config);

    let result = match transcript as u32 {
        0 => prove_with_transcript::<Blake2bTranscript>(&program, inputs, output_cap, trace_length),
        1 => prove_with_transcript::<KeccakTranscript>(&program, inputs, output_cap, trace_length),
        _ => Err(JoltStatusCode::ErrorInvalidInput),
    };

    match result {
        Ok((cycles, proof, outputs)) => JoltProofResult::success(cycles, proof, outputs),
        Err(status) => JoltProofResult::failure(status),
    }
}

fn verify_impl(
    proof_data: *const c_uchar,
    proof_len: usize,
    elf_bytes: *const c_uchar,
    elf_len: usize,
    input_bytes: *const c_uchar,
    input_len: usize,
    output_bytes: *const c_uchar,
    output_len: usize,
    max_input_size: usize,
    max_output_size: usize,
    max_trace_length: usize,
    transcript: JoltTranscriptFlavor,
) -> JoltVerifyResult {
    if proof_data.is_null() || proof_len == 0 || elf_bytes.is_null() || elf_len == 0 {
        return JoltVerifyResult::failure(JoltStatusCode::ErrorInvalidInput);
    }
    if max_input_size == 0 {
        return JoltVerifyResult::failure(JoltStatusCode::ErrorInvalidInput);
    }

    let trace_length = match resolve_trace_length(max_trace_length) {
        Ok(len) => len,
        Err(status) => return JoltVerifyResult::failure(status),
    };

    let (output_cap, output_cap_u64) = match resolve_output_size(max_output_size) {
        Ok(values) => values,
        Err(status) => return JoltVerifyResult::failure(status),
    };

    if input_len > max_input_size || output_len > output_cap {
        return JoltVerifyResult::failure(JoltStatusCode::ErrorInvalidInput);
    }

    let proof_bytes = unsafe { slice::from_raw_parts(proof_data, proof_len) };
    let elf = unsafe { slice::from_raw_parts(elf_bytes, elf_len) };
    let inputs = match unsafe { read_optional_slice(input_bytes, input_len) } {
        Ok(slice) => slice,
        Err(status) => return JoltVerifyResult::failure(status),
    };
    let outputs = match unsafe { read_optional_slice(output_bytes, output_len) } {
        Ok(slice) => slice,
        Err(status) => return JoltVerifyResult::failure(status),
    };

    let memory_config = match build_memory_config(max_input_size, output_cap_u64) {
        Ok(cfg) => cfg,
        Err(status) => return JoltVerifyResult::failure(status),
    };

    let program = Program::new(elf, &memory_config);

    let result = match transcript as u32 {
        0 => verify_with_transcript::<Blake2bTranscript>(
            &program,
            proof_bytes,
            inputs,
            outputs,
            trace_length,
        ),
        1 => verify_with_transcript::<KeccakTranscript>(
            &program,
            proof_bytes,
            inputs,
            outputs,
            trace_length,
        ),
        _ => Err(JoltStatusCode::ErrorInvalidInput),
    };

    match result {
        Ok(is_valid) => JoltVerifyResult::success(is_valid),
        Err(status) => JoltVerifyResult::failure(status),
    }
}

fn prove_with_transcript<FS: Transcript>(
    program: &Program,
    inputs: &[u8],
    max_output_size: usize,
    max_trace_length: usize,
) -> Result<(u64, Vec<u8>, Vec<u8>), JoltStatusCode> {
    let preprocessing = prover::preprocess(program, max_trace_length);
    let mut output_buffer = vec![0u8; max_output_size];

    let (proof, io_device, _) = prover::prove::<F, PCS, FS>(
        program,
        inputs,
        &[],
        &[],
        None,
        &mut output_buffer,
        &preprocessing,
    );

    let cycles = proof.trace_length as u64;

    let mut proof_bytes = Vec::new();
    proof
        .serialize_compressed(&mut proof_bytes)
        .map_err(|_| JoltStatusCode::ErrorSerialization)?;

    Ok((cycles, proof_bytes, io_device.outputs))
}

fn verify_with_transcript<FS: Transcript>(
    program: &Program,
    proof_bytes: &[u8],
    inputs: &[u8],
    outputs: &[u8],
    max_trace_length: usize,
) -> Result<bool, JoltStatusCode> {
    let proof = JoltProof::<F, PCS, FS>::deserialize_compressed(proof_bytes)
        .map_err(|_| JoltStatusCode::ErrorSerialization)?;

    let preprocessing = verifier::preprocess(program, max_trace_length);
    let result =
        verifier::verify::<F, PCS, FS>(inputs, None, outputs, proof, &preprocessing).is_ok();
    Ok(result)
}

unsafe fn read_optional_slice<'a>(
    ptr: *const c_uchar,
    len: usize,
) -> Result<&'a [u8], JoltStatusCode> {
    if len == 0 {
        Ok(&[])
    } else if ptr.is_null() {
        Err(JoltStatusCode::ErrorInvalidInput)
    } else {
        Ok(slice::from_raw_parts(ptr, len))
    }
}

fn build_memory_config(
    max_input_size: usize,
    max_output_size: u64,
) -> Result<MemoryConfig, JoltStatusCode> {
    let max_input_size =
        u64::try_from(max_input_size).map_err(|_| JoltStatusCode::ErrorInvalidInput)?;

    Ok(MemoryConfig {
        max_input_size,
        max_output_size,
        max_trusted_advice_size: 0,
        max_untrusted_advice_size: 0,
        stack_size: DEFAULT_STACK_SIZE,
        memory_size: DEFAULT_MEMORY_SIZE,
        program_size: None,
    })
}

fn resolve_trace_length(requested: usize) -> Result<usize, JoltStatusCode> {
    let base = if requested == 0 {
        DEFAULT_TRACE_LENGTH
    } else {
        requested
    };
    base.checked_next_power_of_two()
        .ok_or(JoltStatusCode::ErrorInvalidInput)
}

fn resolve_output_size(max_output_size: usize) -> Result<(usize, u64), JoltStatusCode> {
    let resolved = if max_output_size == 0 {
        DEFAULT_MAX_OUTPUT_SIZE
    } else {
        max_output_size
    };
    let resolved_u64 = u64::try_from(resolved).map_err(|_| JoltStatusCode::ErrorInvalidInput)?;
    Ok((resolved, resolved_u64))
}

fn leak_vec(data: Vec<u8>) -> (*mut u8, usize) {
    let mut boxed = data.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    let len = boxed.len();
    std::mem::forget(boxed);
    (ptr, len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn test_version() {
        unsafe {
            let version = jolt_version();
            let c_str = CStr::from_ptr(version);
            let version_str = c_str.to_str().unwrap();
            assert!(!version_str.is_empty());
            assert_eq!(version_str, env!("CARGO_PKG_VERSION"));
        }
    }

    #[test]
    fn test_build_info() {
        unsafe {
            let info = jolt_build_info();
            let c_str = CStr::from_ptr(info);
            let info_str = c_str.to_str().unwrap();
            assert!(info_str.contains("jolt-ios"));
        }
    }

    #[test]
    fn test_free_buffer() {
        unsafe {
            // Create a test buffer
            let test_data = vec![1u8, 2, 3, 4, 5];
            let mut boxed = test_data.into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            let len = boxed.len();
            std::mem::forget(boxed);

            // Free it
            jolt_free_buffer(ptr, len);
        }
    }
}
