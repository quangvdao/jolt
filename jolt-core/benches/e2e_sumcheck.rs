use std::time::Instant;

use ark_bn254::Fr;
use jolt_core::field::fp128::JoltFp128;
use jolt_core::field::JoltField;
use jolt_core::host;
use jolt_core::poly::commitment::mock::MockCommitScheme;
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use jolt_core::zkvm::verifier::JoltSharedPreprocessing;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

fn build_prover<F: JoltField>(
    label: &str,
) -> JoltCpuProver<'static, F, MockCommitScheme<F>, Blake2bTranscript> {
    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&100000u32).unwrap();
    let (bytecode, init_memory_state, _) = program.decode();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared = JoltSharedPreprocessing::new(
        bytecode,
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 22,
    );

    let preprocessing = JoltProverPreprocessing::<F, MockCommitScheme<F>>::new(shared);
    let preprocessing = Box::leak(Box::new(preprocessing));

    let elf_contents = program.get_elf_contents().expect("elf contents is None");

    let prover: JoltCpuProver<F, MockCommitScheme<F>, Blake2bTranscript> =
        JoltCpuProver::gen_from_elf(
            preprocessing,
            &elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
            None,
        );

    println!("{label}: trace_len = {}", prover.trace.len());
    prover
}

fn main() {
    let out_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("benchmark-runs/perfetto_traces");
    std::fs::create_dir_all(&out_dir).expect("failed to create output dir");

    // --- BN254 ---
    {
        let trace_file = out_dir.join("sumcheck_bn254.json");
        println!("BN254 trace → {}", trace_file.display());
        let (chrome_layer, _guard) = ChromeLayerBuilder::new()
            .include_args(true)
            .file(trace_file)
            .build();
        let subscriber = tracing_subscriber::registry().with(chrome_layer);
        let _default = tracing::subscriber::set_default(subscriber);

        let prover = build_prover::<Fr>("BN254");
        let start = Instant::now();
        prover.prove_sumchecks_only();
        let elapsed = start.elapsed();
        println!("BN254: {:.3}s", elapsed.as_secs_f64());
    }

    // --- Fp128 ---
    {
        let trace_file = out_dir.join("sumcheck_fp128.json");
        println!("Fp128 trace → {}", trace_file.display());
        let (chrome_layer, _guard) = ChromeLayerBuilder::new()
            .include_args(true)
            .file(trace_file)
            .build();
        let subscriber = tracing_subscriber::registry().with(chrome_layer);
        let _default = tracing::subscriber::set_default(subscriber);

        let prover = build_prover::<JoltFp128>("Fp128");
        let start = Instant::now();
        prover.prove_sumchecks_only();
        let elapsed = start.elapsed();
        println!("Fp128: {:.3}s", elapsed.as_secs_f64());
    }
}
