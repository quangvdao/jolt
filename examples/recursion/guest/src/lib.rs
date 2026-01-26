// NOTE: recursion-guest currently requires `std` due to transitive dependencies
// (e.g. dory uses std-only crates like `once_cell`).
//
// Trust model:
// This guest program is intended to consume *trusted* inputs produced by a trusted host pipeline
// (e.g. `jolt_sdk::decompress_transport_bytes_to_guest_bytes`). Do not treat the guest encoding as
// a general wire format.

use jolt_sdk::{self as jolt};

extern crate alloc;

use jolt::{
    end_cycle_tracking, start_cycle_tracking, GuestDeserialize, JoltDevice,
    JoltVerifierPreprocessing, RV64IMACProof, RV64IMACVerifier, F, PCS,
};

mod embedded_bytes {
    include!("./embedded_bytes.rs");
}

include!("./provable_macro.rs");

provable_with_config! {
fn verify(bytes: &[u8]) -> u32 {
    // In embed mode, the host passes an empty input slice and we fall back to the embedded bytes.
    // In non-embed mode, always prefer the explicit input bytes (even if embedded bytes exist).
    let data_bytes = if !bytes.is_empty() {
        bytes
    } else {
        embedded_bytes::EMBEDDED_BYTES
    };

    // `&mut &[u8]` implements arkworks' `Read` and advances as bytes are consumed (no `std::io`).
    let mut cursor: &[u8] = data_bytes;

    start_cycle_tracking("deserialize preprocessing");
    let verifier_preprocessing: JoltVerifierPreprocessing<F, PCS> =
        JoltVerifierPreprocessing::<F, PCS>::guest_deserialize(&mut cursor).unwrap();
    end_cycle_tracking("deserialize preprocessing");

    start_cycle_tracking("deserialize count of proofs");
    // Deserialize number of proofs to verify
    let n: u32 = u32::guest_deserialize(&mut cursor).unwrap();
    end_cycle_tracking("deserialize count of proofs");

    let mut all_valid = true;
    for _ in 0..n {
        start_cycle_tracking("deserialize proof");
        let proof = RV64IMACProof::guest_deserialize(&mut cursor).unwrap();
        end_cycle_tracking("deserialize proof");

        start_cycle_tracking("deserialize device");
        let device = JoltDevice::guest_deserialize(&mut cursor).unwrap();
        end_cycle_tracking("deserialize device");

        start_cycle_tracking("verification");
        let verifier = RV64IMACVerifier::new(&verifier_preprocessing, proof, device, None, None);
        let is_valid = verifier.is_ok_and(|verifier| {
            // Recursion verification was extracted to `jolt-recursion`; guest verification only
            // supports the base Jolt verifier path.
            let result = verifier.verify();
            core::hint::black_box(result).is_ok()
        });
        end_cycle_tracking("verification");

        all_valid = all_valid && is_valid;
    }

    all_valid as u32
}
}
