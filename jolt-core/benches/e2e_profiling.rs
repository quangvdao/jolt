use std::sync::Arc;

use ark_serialize::CanonicalSerialize;
use jolt_core::host;
use jolt_core::zkvm::config::ProgramMode;
use jolt_core::zkvm::prover::JoltProverPreprocessing;
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifierPreprocessing};
use jolt_core::zkvm::{RV64IMACProver, RV64IMACVerifier};
use std::fs;
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "recursion")]
mod recursion_sizes {
    use super::*;
    use ark_serialize::Compress;

    fn sz<T: CanonicalSerialize>(v: &T, compress: Compress) -> usize {
        v.serialized_size(compress)
    }

    fn log_size(label: &str, bytes: usize) {
        println!("{label}: {bytes} bytes");
    }

    fn log_sizes<T: CanonicalSerialize>(label: &str, v: &T) -> (usize, usize) {
        let c = sz(v, Compress::Yes);
        let u = sz(v, Compress::No);
        println!("{label}: {c} bytes (compressed), {u} bytes (uncompressed)");
        (c, u)
    }

    pub fn log_jolt_proof_breakdown(
        proof: &jolt_core::zkvm::proof_serialization::JoltProof<
            jolt_core::ark_bn254::Fr,
            jolt_core::poly::commitment::dory::DoryCommitmentScheme,
            jolt_core::transcripts::Blake2bTranscript,
        >,
    ) {
        println!("--- base JoltProof size breakdown (ark_serialize canonical) ---");
        log_sizes("opening_claims", &proof.opening_claims);
        log_sizes("commitments", &proof.commitments);
        log_sizes(
            "stage1_uni_skip_first_round_proof",
            &proof.stage1_uni_skip_first_round_proof,
        );
        log_sizes("stage1_sumcheck_proof", &proof.stage1_sumcheck_proof);
        log_sizes(
            "stage2_uni_skip_first_round_proof",
            &proof.stage2_uni_skip_first_round_proof,
        );
        log_sizes("stage2_sumcheck_proof", &proof.stage2_sumcheck_proof);
        log_sizes("stage3_sumcheck_proof", &proof.stage3_sumcheck_proof);
        log_sizes("stage4_sumcheck_proof", &proof.stage4_sumcheck_proof);
        log_sizes("stage5_sumcheck_proof", &proof.stage5_sumcheck_proof);
        log_sizes("stage6a_sumcheck_proof", &proof.stage6a_sumcheck_proof);
        log_sizes("stage6b_sumcheck_proof", &proof.stage6b_sumcheck_proof);
        log_sizes("stage7_sumcheck_proof", &proof.stage7_sumcheck_proof);
        log_sizes("joint_opening_proof", &proof.joint_opening_proof);
        log_sizes(
            "untrusted_advice_commitment",
            &proof.untrusted_advice_commitment,
        );
        log_size("trace_length", sz(&proof.trace_length, Compress::Yes));
        log_size("ram_K", sz(&proof.ram_K, Compress::Yes));
        log_size("bytecode_K", sz(&proof.bytecode_K, Compress::Yes));
        log_size("program_mode", sz(&proof.program_mode, Compress::Yes));
        log_size("rw_config", sz(&proof.rw_config, Compress::Yes));
        log_size("one_hot_config", sz(&proof.one_hot_config, Compress::Yes));
        log_size("dory_layout", sz(&proof.dory_layout, Compress::Yes));
        log_size("TOTAL (compressed)", sz(proof, Compress::Yes));
        log_size("TOTAL (uncompressed)", sz(proof, Compress::No));
    }

    pub fn log_recursion_artifact_breakdown(
        artifact: &jolt_core::zkvm::recursion::RecursionArtifact<
            jolt_core::transcripts::Blake2bTranscript,
        >,
    ) {
        println!("--- RecursionArtifact size breakdown (ark_serialize canonical) ---");
        log_sizes("stage8_combine_hint", &artifact.stage8_combine_hint);
        log_sizes("pairing_boundary", &artifact.pairing_boundary);
        log_sizes("non_input_base_hints", &artifact.non_input_base_hints);

        println!("--- recursion SNARK proof (inner) ---");
        log_sizes("proof.stage1_proof", &artifact.proof.stage1_proof);
        log_sizes("proof.stage2_proof", &artifact.proof.stage2_proof);
        log_sizes(
            "proof.stage3_packed_eval",
            &artifact.proof.stage3_packed_eval,
        );
        log_sizes("proof.opening_proof", &artifact.proof.opening_proof);
        log_sizes("proof.opening_claims", &artifact.proof.opening_claims);
        log_sizes("proof.dense_commitment", &artifact.proof.dense_commitment);

        log_size("TOTAL (compressed)", sz(artifact, Compress::Yes));
        log_size("TOTAL (uncompressed)", sz(artifact, Compress::No));
    }
}

// Empirically measured cycles per operation for RV64IMAC
const CYCLES_PER_SHA256: f64 = 3396.0;
const CYCLES_PER_SHA3: f64 = 4330.0;
const CYCLES_PER_BTREEMAP_OP: f64 = 1550.0;
const CYCLES_PER_FIBONACCI_UNIT: f64 = 12.0;
const SAFETY_MARGIN: f64 = 0.9; // Use 90% of max trace capacity

/// Calculate number of operations to target a specific cycle count
fn scale_to_target_ops(target_cycles: usize, cycles_per_op: f64) -> u32 {
    std::cmp::max(1, (target_cycles as f64 / cycles_per_op) as u32)
}

#[derive(Debug, Copy, Clone, clap::ValueEnum, strum_macros::Display)]
#[strum(serialize_all = "kebab-case")]
pub enum BenchType {
    #[value(name = "btreemap")]
    BTreeMap,
    Fibonacci,
    Sha2,
    Sha3,
    #[strum(serialize = "SHA2 Chain")]
    Sha2Chain,
    #[strum(serialize = "SHA3 Chain")]
    Sha3Chain,
}

pub fn benchmarks(
    bench_type: BenchType,
    committed: bool,
    recursion: bool,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::BTreeMap => btreemap(committed, recursion),
        BenchType::Sha2 => sha2(committed, recursion),
        BenchType::Sha3 => sha3(committed, recursion),
        BenchType::Sha2Chain => sha2_chain(committed, recursion),
        BenchType::Sha3Chain => sha3_chain(committed, recursion),
        BenchType::Fibonacci => fibonacci(committed, recursion),
    }
}

fn fibonacci(committed: bool, recursion: bool) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example(
        "fibonacci-guest",
        postcard::to_stdvec(&400000u32).unwrap(),
        committed,
        recursion,
    )
}

fn sha2(committed: bool, recursion: bool) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    #[cfg(feature = "host")]
    use jolt_inlines_sha2 as _;
    prove_example(
        "sha2-guest",
        postcard::to_stdvec(&vec![5u8; 2048]).unwrap(),
        committed,
        recursion,
    )
}

fn sha3(committed: bool, recursion: bool) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    #[cfg(feature = "host")]
    use jolt_inlines_keccak256 as _;
    prove_example(
        "sha3-guest",
        postcard::to_stdvec(&vec![5u8; 2048]).unwrap(),
        committed,
        recursion,
    )
}

fn btreemap(committed: bool, recursion: bool) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example(
        "btreemap-guest",
        postcard::to_stdvec(&50u32).unwrap(),
        committed,
        recursion,
    )
}

fn sha2_chain(committed: bool, recursion: bool) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    #[cfg(feature = "host")]
    use jolt_inlines_sha2 as _;
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    let iters = scale_to_target_ops(
        ((1 << 24) as f64 * SAFETY_MARGIN) as usize,
        CYCLES_PER_SHA256,
    );
    inputs.append(&mut postcard::to_stdvec(&iters).unwrap());
    prove_example("sha2-chain-guest", inputs, committed, recursion)
}

fn sha3_chain(committed: bool, recursion: bool) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    #[cfg(feature = "host")]
    extern crate jolt_inlines_keccak256;
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&20u32).unwrap());
    prove_example("sha3-chain-guest", inputs, committed, recursion)
}

pub fn master_benchmark(
    bench_type: BenchType,
    bench_scale: usize,
    target_trace_size: Option<usize>,
    recursion: bool,
    committed: bool,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    // Ensure SHA2 inline library is linked and auto-registered
    #[cfg(feature = "host")]
    extern crate jolt_inlines_sha2;
    #[cfg(feature = "host")]
    extern crate jolt_inlines_keccak256;

    if let Err(e) = fs::create_dir_all("benchmark-runs/results") {
        eprintln!("Warning: Failed to create benchmark-runs/results directory: {e}");
    }

    let task = move || {
        let max_trace_length = 1 << bench_scale;
        let bench_target =
            target_trace_size.unwrap_or(((1 << bench_scale) as f64 * SAFETY_MARGIN) as usize);

        // Map benchmark type to canonical name + input closure
        let (bench_name, input_fn): (&str, fn(usize) -> Vec<u8>) = match bench_type {
            BenchType::Fibonacci => ("fibonacci", |target| {
                postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_FIBONACCI_UNIT))
                    .unwrap()
            }),
            BenchType::Sha2Chain => ("sha2-chain", |target| {
                let iterations = scale_to_target_ops(target, CYCLES_PER_SHA256);
                [
                    postcard::to_stdvec(&[5u8; 32]).unwrap(),
                    postcard::to_stdvec(&iterations).unwrap(),
                ]
                .concat()
            }),
            BenchType::Sha3Chain => ("sha3-chain", |target| {
                let iterations = scale_to_target_ops(target, CYCLES_PER_SHA3);
                [
                    postcard::to_stdvec(&[5u8; 32]).unwrap(),
                    postcard::to_stdvec(&iterations).unwrap(),
                ]
                .concat()
            }),
            BenchType::BTreeMap => ("btreemap", |target| {
                postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_BTREEMAP_OP)).unwrap()
            }),
            BenchType::Sha2 => panic!("Use sha2-chain instead"),
            BenchType::Sha3 => panic!("Use sha3-chain instead"),
        };

        tracing::info!("Running {bench_name} benchmark at scale 2^{bench_scale}");

        // Derive names from canonical bench_name
        let guest_name = format!("{bench_name}-guest");
        // Generate input and run benchmark
        let input = input_fn(bench_target);
        let (duration, proof_size, proof_size_comp, trace_length) = prove_example_with_trace(
            &guest_name,
            input,
            max_trace_length,
            bench_name,
            bench_scale,
            recursion,
            committed,
        );

        let proving_hz = trace_length as f64 / duration.as_secs_f64();
        let padded_proving_hz = trace_length.next_power_of_two() as f64 / duration.as_secs_f64();
        println!(
            "{} (2^{}): Prover completed in {:.2}s ({:.1} kHz / padded {:.1} kHz)",
            bench_name,
            bench_scale,
            duration.as_secs_f64(),
            proving_hz / 1000.0,
            padded_proving_hz / 1000.0,
        );

        // Write CSV
        let summary_line = format!(
            "{},{},{:.2},{},{:.2},{},{}\n",
            bench_name,
            bench_scale,
            duration.as_secs_f64(),
            trace_length.next_power_of_two(),
            padded_proving_hz,
            proof_size,
            proof_size_comp
        );

        // Write individual result file for resume detection
        let individual_file = format!("benchmark-runs/results/{bench_name}_{bench_scale}.csv");
        if let Err(e) = fs::write(&individual_file, &summary_line) {
            eprintln!("Failed to write individual result file {individual_file}: {e}");
        }

        // Also append to consolidated timings file
        if let Err(e) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("benchmark-runs/results/timings.csv")
            .and_then(|mut f| f.write_all(summary_line.as_bytes()))
        {
            eprintln!("Failed to write consolidated timing: {e}");
        }
    };

    vec![(
        tracing::info_span!("MasterBenchmark"),
        Box::new(task) as Box<dyn FnOnce()>,
    )]
}

fn prove_example(
    example_name: &str,
    serialized_input: Vec<u8>,
    committed: bool,
    recursion: bool,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut tasks = Vec::new();
    let mut program = host::Program::new(example_name);
    let (instructions, init_memory_state, _) = program.decode();
    let (_lazy_trace, trace, _, program_io) = program.trace(&serialized_input, &[], &[]);
    let padded_trace_len = (trace.len() + 1).next_power_of_two();
    drop(trace);

    let task = move || {
        use jolt_core::zkvm::program::ProgramPreprocessing;
        let program_data = Arc::new(ProgramPreprocessing::preprocess(
            instructions,
            init_memory_state,
        ));
        let shared_preprocessing = JoltSharedPreprocessing::new(
            program_data.meta(),
            program_io.memory_layout.clone(),
            padded_trace_len,
        );

        // Choose preprocessing mode based on committed flag
        let preprocessing = if committed {
            tracing::info!("Using COMMITTED mode");
            JoltProverPreprocessing::new_committed(
                shared_preprocessing.clone(),
                Arc::clone(&program_data),
            )
        } else {
            JoltProverPreprocessing::new(shared_preprocessing.clone(), Arc::clone(&program_data))
        };

        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let program_mode = if committed {
            ProgramMode::Committed
        } else {
            ProgramMode::Full
        };
        let prover = RV64IMACProver::gen_from_elf_with_program_mode(
            &preprocessing,
            elf_contents,
            &serialized_input,
            &[],
            &[],
            None,
            None,
            program_mode,
        );
        let program_io = prover.program_io.clone();
        let (jolt_proof, _) = prover.prove();

        // Verifier preprocessing is derived from prover preprocessing.
        // This automatically uses committed mode if preprocessing was committed.
        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);

        if recursion {
            #[cfg(feature = "recursion")]
            {
                use jolt_core::transcripts::Blake2bTranscript;
                let recursion_artifact =
                    jolt_core::zkvm::recursion::prove_recursion::<Blake2bTranscript>(
                        &verifier_preprocessing,
                        program_io.clone(),
                        None,
                        &jolt_proof,
                    )
                    .expect("Failed to generate recursion artifact");

                recursion_sizes::log_jolt_proof_breakdown(&jolt_proof);
                recursion_sizes::log_recursion_artifact_breakdown(&recursion_artifact);

                jolt_core::zkvm::recursion::verify_recursion::<Blake2bTranscript>(
                    &verifier_preprocessing,
                    program_io,
                    None,
                    &jolt_proof,
                    &recursion_artifact,
                )
                .expect("Recursion verification failed");
                return;
            }
            #[cfg(not(feature = "recursion"))]
            {
                println!(
                    "recursion requested, but jolt-core was built without `--features recursion`"
                );
            }
        }

        let verifier =
            RV64IMACVerifier::new(&verifier_preprocessing, jolt_proof, program_io, None, None)
                .expect("Failed to create verifier");
        verifier.verify().unwrap();
    };

    let span_name = if committed {
        "Example_E2E_Committed"
    } else {
        "Example_E2E"
    };
    tasks.push((
        tracing::info_span!("{}", span_name),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn prove_example_with_trace(
    example_name: &str,
    serialized_input: Vec<u8>,
    max_trace_length: usize,
    _bench_name: &str,
    _scale: usize,
    recursion: bool,
    committed: bool,
) -> (std::time::Duration, usize, usize, usize) {
    let mut program = host::Program::new(example_name);
    let (instructions, init_memory_state, _) = program.decode();
    let (_, trace, _, program_io) = program.trace(&serialized_input, &[], &[]);

    assert!(
        trace.len().next_power_of_two() <= max_trace_length,
        "Trace is longer than expected"
    );

    use jolt_core::zkvm::program::ProgramPreprocessing;
    let program_data = Arc::new(ProgramPreprocessing::preprocess(
        instructions,
        init_memory_state,
    ));
    let shared_preprocessing = JoltSharedPreprocessing::new(
        program_data.meta(),
        program_io.memory_layout.clone(),
        trace.len().next_power_of_two(),
    );
    let preprocessing = if committed {
        tracing::info!("Using COMMITTED mode");
        JoltProverPreprocessing::new_committed(shared_preprocessing, Arc::clone(&program_data))
    } else {
        JoltProverPreprocessing::new(shared_preprocessing, Arc::clone(&program_data))
    };

    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");

    let span = tracing::info_span!("E2E").entered();
    let program_mode = if committed {
        ProgramMode::Committed
    } else {
        ProgramMode::Full
    };
    let prover = RV64IMACProver::gen_from_elf_with_program_mode(
        &preprocessing,
        elf_contents,
        &serialized_input,
        &[],
        &[],
        None,
        None,
        program_mode,
    );
    let now = Instant::now();
    let (jolt_proof, _) = prover.prove();
    let prove_duration = now.elapsed();
    drop(span);
    let proof_size = jolt_proof.serialized_size(ark_serialize::Compress::Yes);

    // Stage 8: Dory opening proof (curve points - benefits from compression)
    let stage8_size_compressed = jolt_proof
        .joint_opening_proof
        .serialized_size(ark_serialize::Compress::Yes);
    let stage8_size_uncompressed = jolt_proof
        .joint_opening_proof
        .serialized_size(ark_serialize::Compress::No);

    // Commitments (curve points - benefits from compression)
    let commitments_size_compressed = jolt_proof
        .commitments
        .serialized_size(ark_serialize::Compress::Yes);
    let commitments_size_uncompressed = jolt_proof
        .commitments
        .serialized_size(ark_serialize::Compress::No);

    // Estimate proof size with full Dory compression (assuming ~3x compression ratio)
    let proof_size_full_compressed = proof_size - stage8_size_compressed
        + (stage8_size_uncompressed / 3)
        - commitments_size_compressed
        + (commitments_size_uncompressed / 3);

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);

    #[cfg(feature = "recursion")]
    let mut total_size = proof_size;
    #[cfg(not(feature = "recursion"))]
    let total_size = proof_size;

    #[cfg(feature = "recursion")]
    let mut total_size_full_compressed = proof_size_full_compressed;
    #[cfg(not(feature = "recursion"))]
    let total_size_full_compressed = proof_size_full_compressed;

    if recursion {
        #[cfg(feature = "recursion")]
        {
            use jolt_core::transcripts::Blake2bTranscript;
            let recursion_artifact =
                jolt_core::zkvm::recursion::prove_recursion::<Blake2bTranscript>(
                    &verifier_preprocessing,
                    program_io.clone(),
                    None,
                    &jolt_proof,
                )
                .expect("Failed to generate recursion artifact");

            recursion_sizes::log_jolt_proof_breakdown(&jolt_proof);
            recursion_sizes::log_recursion_artifact_breakdown(&recursion_artifact);

            let rec_size = recursion_artifact.serialized_size(ark_serialize::Compress::Yes);
            total_size += rec_size;
            total_size_full_compressed += rec_size;

            jolt_core::zkvm::recursion::verify_recursion::<Blake2bTranscript>(
                &verifier_preprocessing,
                program_io,
                None,
                &jolt_proof,
                &recursion_artifact,
            )
            .expect("Recursion verification failed");
        }
        #[cfg(not(feature = "recursion"))]
        {
            println!("recursion requested, but jolt-core was built without `--features recursion`");
            let verifier =
                RV64IMACVerifier::new(&verifier_preprocessing, jolt_proof, program_io, None, None)
                    .expect("Failed to create verifier");
            verifier.verify().unwrap();
        }
    } else {
        let verifier =
            RV64IMACVerifier::new(&verifier_preprocessing, jolt_proof, program_io, None, None)
                .expect("Failed to create verifier");
        verifier.verify().unwrap();
    }

    (
        prove_duration,
        total_size,
        total_size_full_compressed,
        trace.len(),
    )
}
