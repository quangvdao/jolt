use jolt_sdk::{CommitmentScheme, PCS};
use spinners::{Spinner, Spinners};
use tracing::info;
use tracing_subscriber::fmt;

macro_rules! step {
    ($msg:expr, $action:expr) => {{
        let mut sp = Spinner::new(Spinners::Dots9, $msg.to_string());
        let result = $action;
        sp.stop_with_message(format!("✓ {}", $msg));
        result
    }};
}

pub fn btreemap() {
    let target_dir = "/tmp/jolt-guest-targets";

    let mut program = step!("Compiling guest code", {
        guest::compile_btreemap(target_dir)
    });

    let shared_preprocessing = step!("Preprocessing shared", {
        guest::preprocess_shared_btreemap(&mut program).unwrap()
    });

    let prover_preprocessing = step!("Preprocessing prover", {
        guest::preprocess_prover_btreemap(shared_preprocessing.clone())
    });

    let verifier_preprocessing = step!("Preprocessing verifier", {
        guest::preprocess_verifier_btreemap(
            shared_preprocessing,
            <PCS as CommitmentScheme>::project_verifier_setup(&prover_preprocessing.generators),
            None,
        )
    });

    let prove = step!("Building prover", {
        guest::build_prover_btreemap(program, prover_preprocessing)
    });

    let verify = step!("Building verifier", {
        guest::build_verifier_btreemap(verifier_preprocessing)
    });

    let n = 50;
    let (output, proof, io_device) = step!("Proving", { prove(n) });
    assert!(output >= 1);

    let is_valid = step!("Verifying", { verify(n, output, io_device.panic, proof) });
    assert!(is_valid);
}

fn main() {
    fmt::init();

    info!("BTreeMap");
    btreemap();
}
