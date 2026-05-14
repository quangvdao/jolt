use jolt_sdk::{CommitmentScheme, PCS};
use std::time::Instant;
use tracing::info;
use tracing_subscriber::fmt;

pub fn main() {
    fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_muldiv(target_dir);

    let shared_preprocessing = guest::preprocess_shared_muldiv(&mut program).unwrap();
    let prover_preprocessing = guest::preprocess_prover_muldiv(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_muldiv(
        shared_preprocessing,
        <PCS as CommitmentScheme>::project_verifier_setup(&prover_preprocessing.generators),
        None,
    );

    let prove = guest::build_prover_muldiv(program, prover_preprocessing);
    let verify = guest::build_verifier_muldiv(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove(12031293, 17, 92);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify(12031293, 17, 92, output, program_io.panic, proof);

    info!("output: {output}");
    info!("valid: {is_valid}");
}
