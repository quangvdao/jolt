use jolt_sdk::{CommitmentScheme, PCS};
use std::time::Instant;
use tracing::info;
use tracing_subscriber::fmt;

pub fn main() {
    fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_sha3(target_dir);
    let shared_preprocessing = guest::preprocess_shared_sha3(&mut program).unwrap();
    let prover_preprocessing = guest::preprocess_prover_sha3(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_sha3(
        shared_preprocessing,
        <PCS as CommitmentScheme>::project_verifier_setup(&prover_preprocessing.generators),
        None,
    );

    let prove_sha3 = guest::build_prover_sha3(program, prover_preprocessing);
    let verify_sha3 = guest::build_verifier_sha3(verifier_preprocessing);

    let input = b"Hello, world!";
    let native_output = guest::sha3(input);
    let now = Instant::now();
    let (output, proof, program_io) = prove_sha3(input);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha3(input, output, program_io.panic, proof);

    assert_eq!(output, native_output, "output mismatch");
    info!("output: {}", hex::encode(output));
    info!("native_output: {}", hex::encode(native_output));
    info!("valid: {is_valid}");
}
