use jni::objects::{JByteArray, JClass, JObject};
use jni::sys::{jint, jstring};
use jni::JNIEnv;

// Note: This FFI layer currently requires further development to work without the `host` feature.
// The jolt-core library's high-level API (Program, preprocessing, etc.) is gated behind the `host` feature,
// which pulls in dependencies (reqwest, tokio, file I/O) that may not be suitable for Android.
//
// To make this production-ready for Android:
// 1. Either enable `host` feature and ensure all dependencies work on Android
// 2. Or implement lower-level proof construction using only `minimal` + `prover` features
// 3. Handle ELF parsing, trace generation, and preprocessing manually

/// Status codes for FFI operations
#[repr(i32)]
pub enum JoltStatusCode {
    Success = 0,
    ErrorInvalidInput = 1,
    ErrorProofGeneration = 2,
    ErrorVerification = 3,
    ErrorSerialization = 4,
    ErrorOutOfMemory = 5,
    ErrorNotImplemented = 99,
}

/// Get the library version string
///
/// Java signature: `public static native String getVersion();`
#[no_mangle]
pub extern "C" fn Java_com_jolt_JoltProver_getVersion(env: JNIEnv, _class: JClass) -> jstring {
    let version = env!("CARGO_PKG_VERSION");
    env.new_string(version)
        .expect("Couldn't create Java string")
        .into_raw()
}

/// Get build information
///
/// Java signature: `public static native String getBuildInfo();`
#[no_mangle]
pub extern "C" fn Java_com_jolt_JoltProver_getBuildInfo(env: JNIEnv, _class: JClass) -> jstring {
    let info = concat!(
        "jolt-android v",
        env!("CARGO_PKG_VERSION"),
        " (minimal API - needs implementation)"
    );
    env.new_string(info)
        .expect("Couldn't create Java string")
        .into_raw()
}

/// Generate a proof from an ELF binary
///
/// Java signature:
/// ```java
/// public static native ProofResult prove(
///     byte[] elfBytes,
///     byte[] inputBytes,
///     int maxInputSize,
///     int maxOutputSize
/// );
/// ```
///
/// Returns a ProofResult object with fields:
/// - `int status` - Status code
/// - `long cycles` - Number of cycles executed
/// - `byte[] proofData` - Serialized proof (null if error)
///
/// # Note
/// This is a stub implementation. A full implementation would:
/// 1. Parse the ELF binary
/// 2. Set up the program with the provided inputs
/// 3. Generate an execution trace
/// 4. Run preprocessing
/// 5. Generate the proof
/// 6. Serialize and return the proof
#[no_mangle]
pub extern "C" fn Java_com_jolt_JoltProver_prove<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    _elf_bytes: JByteArray<'local>,
    _input_bytes: JByteArray<'local>,
    _max_input_size: jint,
    _max_output_size: jint,
) -> JObject<'local> {
    // TODO: Implement proof generation
    // This requires either:
    // A) Using host feature and ensuring all dependencies work on Android, or
    // B) Manually implementing program loading, trace generation, and preprocessing
    //    using only the lower-level jolt-core APIs

    // For now, return a ProofResult indicating not implemented
    let proof_result_class = env
        .find_class("com/jolt/JoltProver$ProofResult")
        .expect("Couldn't find ProofResult class");

    env.new_object(
        proof_result_class,
        "(IJ[B)V",
        &[
            (JoltStatusCode::ErrorNotImplemented as i32).into(),
            0i64.into(),
            (&JObject::null()).into(),
        ],
    )
    .expect("Couldn't create ProofResult object")
}

/// Verify a proof
///
/// Java signature:
/// ```java
/// public static native VerifyResult verify(
///     byte[] proofData,
///     byte[] preprocessingData,
///     byte[] expectedOutput
/// );
/// ```
///
/// Returns a VerifyResult object with fields:
/// - `int status` - Status code
/// - `boolean isValid` - Whether the proof is valid
///
/// # Note
/// This is a stub implementation. A full implementation would:
/// 1. Deserialize the preprocessing data
/// 2. Deserialize the proof
/// 3. Run the verifier
/// 4. Check outputs match expectations
#[no_mangle]
pub extern "C" fn Java_com_jolt_JoltProver_verify<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    _proof_data: JByteArray<'local>,
    _preprocessing_data: JByteArray<'local>,
    _expected_output: JByteArray<'local>,
) -> JObject<'local> {
    // TODO: Implement verification

    let verify_result_class = env
        .find_class("com/jolt/JoltProver$VerifyResult")
        .expect("Couldn't find VerifyResult class");

    env.new_object(
        verify_result_class,
        "(IZ)V",
        &[
            (JoltStatusCode::ErrorNotImplemented as i32).into(),
            false.into(),
        ],
    )
    .expect("Couldn't create VerifyResult object")
}

/// Test function to verify JNI is working
///
/// Java signature: `public static native int testAdd(int a, int b);`
#[no_mangle]
pub extern "C" fn Java_com_jolt_JoltProver_testAdd(
    _env: JNIEnv,
    _class: JClass,
    a: jint,
    b: jint,
) -> jint {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_codes() {
        assert_eq!(JoltStatusCode::Success as i32, 0);
        assert_eq!(JoltStatusCode::ErrorNotImplemented as i32, 99);
    }
}
