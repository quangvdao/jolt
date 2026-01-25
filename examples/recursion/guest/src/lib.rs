// #![cfg_attr(feature = "guest", no_std)]

use jolt_sdk::{self as jolt};

extern crate alloc;

use ark_serialize::{CanonicalDeserialize, Compress, Validate};
use jolt::{JoltDevice, JoltVerifierPreprocessing, RV64IMACProof, RV64IMACVerifier, F, PCS};

#[cfg(feature = "proof-bundle")]
use jolt::serialized_bundle::ProofBundleReader;
use jolt::{end_cycle_tracking, start_cycle_tracking};

mod embedded_bytes {
    include!("./embedded_bytes.rs");
}

include!("./provable_macro.rs");

// Serialized input mode (default)
#[cfg(all(not(feature = "native-input"), not(feature = "proof-bundle")))]
provable_with_config! {
fn verify(bytes: &[u8]) -> u32 {
    // In embed mode, the host passes an empty input slice and we fall back to the embedded bytes.
    // In non-embed mode, always prefer the explicit input bytes (even if embedded bytes exist).
    let data_bytes = if !bytes.is_empty() {
        bytes
    } else {
        embedded_bytes::EMBEDDED_BYTES
    };

    let mut cursor = std::io::Cursor::new(data_bytes);

    start_cycle_tracking("deserialize preprocessing");
    let verifier_preprocessing: JoltVerifierPreprocessing<F, PCS> =
        JoltVerifierPreprocessing::<F, PCS>::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No).unwrap();
    end_cycle_tracking("deserialize preprocessing");

    start_cycle_tracking("deserialize count of proofs");
    // Deserialize number of proofs to verify
    let n: u32 = u32::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No).unwrap();
    end_cycle_tracking("deserialize count of proofs");

    let mut all_valid = true;
    for _ in 0..n {
        start_cycle_tracking("deserialize proof");
        let proof = RV64IMACProof::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No).unwrap();
        end_cycle_tracking("deserialize proof");

        start_cycle_tracking("deserialize device");
        let device = JoltDevice::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No).unwrap();
        end_cycle_tracking("deserialize device");

        start_cycle_tracking("verification");
        let verifier = RV64IMACVerifier::new(&verifier_preprocessing, proof, device, None, None);
        let is_valid = verifier.is_ok_and(|verifier| {
            let result = verifier.verify();
            core::hint::black_box(result).is_ok()
        });
        end_cycle_tracking("verification");

        all_valid = all_valid && is_valid;
    }

    all_valid as u32
}
}

// Proof-bundle input mode (feature-gated)
#[cfg(all(not(feature = "native-input"), feature = "proof-bundle"))]
provable_with_config! {
fn verify(bytes: &[u8]) -> u32 {
    // In embed mode, the host passes an empty input slice and we fall back to the embedded bytes.
    // In non-embed mode, always prefer the explicit input bytes (even if embedded bytes exist).
    let data_bytes = if !bytes.is_empty() {
        bytes
    } else {
        embedded_bytes::EMBEDDED_BYTES
    };

    // Parse the proof bundle container.
    let mut bundle = ProofBundleReader::new(data_bytes).unwrap();

    start_cycle_tracking("deserialize preprocessing");
    let compress = if bundle.version() == 1 {
        Compress::Yes
    } else {
        Compress::No
    };
    let verifier_preprocessing: JoltVerifierPreprocessing<F, PCS> =
        JoltVerifierPreprocessing::<F, PCS>::deserialize_with_mode(
            &mut std::io::Cursor::new(bundle.preprocessing_bytes()),
            compress,
            Validate::No,
        )
        .unwrap();
    end_cycle_tracking("deserialize preprocessing");

    let mut all_valid = true;
    while let Some(entry) = bundle.next_entry().unwrap() {
        start_cycle_tracking("deserialize proof");
        // Proof bytes are already in the proof-record encoding; verify by streaming.
        end_cycle_tracking("deserialize proof");

        start_cycle_tracking("deserialize device");
        let device =
            JoltDevice::deserialize_with_mode(&mut std::io::Cursor::new(entry.device_bytes), Compress::Yes, Validate::No)
                .unwrap();
        end_cycle_tracking("deserialize device");

        start_cycle_tracking("verification");
        let is_valid = jolt::streaming_verifier::rv64imac_verify_from_proof_bytes(
            &verifier_preprocessing,
            device,
            None,
            entry.proof_bytes,
        )
        .is_ok();
        let is_valid = core::hint::black_box(is_valid);
        end_cycle_tracking("verification");

        all_valid = all_valid && is_valid;
    }

    all_valid as u32
}
}

// Pre-deserialized input mode (native-input feature)
#[cfg(feature = "native-input")]
provable_with_config! {
fn verify(
    verifier_preprocessing: &JoltVerifierPreprocessing<F, PCS>,
    proofs: &[(RV64IMACProof, JoltDevice)]
) -> u32 {
    // NOTE: this feature is currently not supported end-to-end because the guest macro
    // uses postcard/serde for argument decoding, while these types use ark_serialize.
    // Keep a placeholder body to avoid accidentally relying on it.
    let _ = (verifier_preprocessing, proofs);
    0
}
}
