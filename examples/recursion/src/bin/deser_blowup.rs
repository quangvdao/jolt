use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use jolt_sdk::{JoltDevice, JoltVerifierPreprocessing, RV64IMACProof, F, PCS};
use std::{env, fs, io::Cursor};

fn report<T: CanonicalSerialize>(label: &str, v: &T) {
    let compressed = v.serialized_size(Compress::Yes);
    let uncompressed = v.serialized_size(Compress::No);
    let ratio = uncompressed as f64 / compressed as f64;
    println!("{label}");
    println!("  compressed:   {compressed:>12} bytes");
    println!("  uncompressed: {uncompressed:>12} bytes");
    println!("  ratio:        {ratio:>12.4}x");
}

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "output/fibonacci-guest_proofs.bin".to_string());

    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    println!("input file: {path}");
    println!("file size:  {} bytes", bytes.len());

    let mut cursor = Cursor::new(bytes.as_slice());

    let preprocessing: JoltVerifierPreprocessing<F, PCS> =
        JoltVerifierPreprocessing::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No)
            .expect("deserialize preprocessing");

    report("preprocessing.total", &preprocessing);
    report(
        "preprocessing.generators (PCS::VerifierSetup)",
        &preprocessing.generators,
    );
    report("preprocessing.shared", &preprocessing.shared);
    report(
        "preprocessing.hyrax_recursion_setup",
        &preprocessing.hyrax_recursion_setup,
    );
    report("preprocessing.program", &preprocessing.program);

    let n: u32 = u32::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No)
        .expect("deserialize proof count");
    println!("proof count: {n}");

    let mut proofs_compressed_total = 0usize;
    let mut proofs_uncompressed_total = 0usize;
    let mut devices_compressed_total = 0usize;
    let mut devices_uncompressed_total = 0usize;

    for i in 0..n {
        let proof: RV64IMACProof =
            RV64IMACProof::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No)
                .expect("deserialize proof");
        let device: JoltDevice =
            JoltDevice::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No)
                .expect("deserialize device");

        let proof_c = proof.serialized_size(Compress::Yes);
        let proof_u = proof.serialized_size(Compress::No);
        let dev_c = device.serialized_size(Compress::Yes);
        let dev_u = device.serialized_size(Compress::No);

        proofs_compressed_total += proof_c;
        proofs_uncompressed_total += proof_u;
        devices_compressed_total += dev_c;
        devices_uncompressed_total += dev_u;

        if i == 0 {
            report("proof[0].total", &proof);
            if let Some(ref payload) = proof.recursion {
                report(
                    "proof[0].recursion.stage10_recursion_metadata",
                    &payload.stage10_recursion_metadata,
                );
                report("proof[0].recursion.recursion_proof", &payload.recursion_proof);
            } else {
                println!("proof[0].recursion: None");
            }
            report("device[0].total", &device);
        }
    }

    println!("totals across {n} proofs/devices:");
    println!("  proofs:   compressed={proofs_compressed_total} uncompressed={proofs_uncompressed_total} ratio={:.4}x",
        proofs_uncompressed_total as f64 / proofs_compressed_total as f64
    );
    println!("  devices:  compressed={devices_compressed_total} uncompressed={devices_uncompressed_total} ratio={:.4}x",
        devices_uncompressed_total as f64 / devices_compressed_total as f64
    );

    let pos = cursor.position() as usize;
    println!("cursor position: {pos} bytes");
    println!(
        "remaining bytes:  {}",
        cursor.get_ref().len().saturating_sub(pos)
    );
}
