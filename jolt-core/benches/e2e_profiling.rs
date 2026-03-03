use ark_serialize::CanonicalSerialize;
use jolt_core::host;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::zkvm::prover::JoltProverPreprocessing;
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifierPreprocessing};
use jolt_core::zkvm::{
    HachiPcs, RV64IMACHachiProver, RV64IMACHachiVerifier, RV64IMACProver, RV64IMACVerifier,
};
use std::fs;
use std::io::Write;
use std::time::Instant;

#[derive(Debug, Copy, Clone, clap::ValueEnum, strum_macros::Display, PartialEq)]
#[strum(serialize_all = "kebab-case")]
pub enum PcsChoice {
    Dory,
    Hachi,
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
    pcs: PcsChoice,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::BTreeMap => btreemap(pcs),
        BenchType::Sha2 => sha2(pcs),
        BenchType::Sha3 => sha3(pcs),
        BenchType::Sha2Chain => sha2_chain(pcs),
        BenchType::Sha3Chain => sha3_chain(pcs),
        BenchType::Fibonacci => fibonacci(pcs),
    }
}

fn fibonacci(pcs: PcsChoice) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example(
        "fibonacci-guest",
        postcard::to_stdvec(&400000u32).unwrap(),
        pcs,
    )
}

fn sha2(pcs: PcsChoice) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    #[cfg(feature = "host")]
    use jolt_inlines_sha2 as _;
    prove_example(
        "sha2-guest",
        postcard::to_stdvec(&vec![5u8; 2048]).unwrap(),
        pcs,
    )
}

fn sha3(pcs: PcsChoice) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    #[cfg(feature = "host")]
    use jolt_inlines_keccak256 as _;
    prove_example(
        "sha3-guest",
        postcard::to_stdvec(&vec![5u8; 2048]).unwrap(),
        pcs,
    )
}

fn btreemap(pcs: PcsChoice) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example("btreemap-guest", postcard::to_stdvec(&50u32).unwrap(), pcs)
}

fn sha2_chain(pcs: PcsChoice) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    #[cfg(feature = "host")]
    use jolt_inlines_sha2 as _;
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    let iters = scale_to_target_ops(
        ((1 << 24) as f64 * SAFETY_MARGIN) as usize,
        CYCLES_PER_SHA256,
    );
    inputs.append(&mut postcard::to_stdvec(&iters).unwrap());
    prove_example("sha2-chain-guest", inputs, pcs)
}

fn sha3_chain(pcs: PcsChoice) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    #[cfg(feature = "host")]
    extern crate jolt_inlines_keccak256;
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&20u32).unwrap());
    prove_example("sha3-chain-guest", inputs, pcs)
}

pub fn master_benchmark(
    bench_type: BenchType,
    bench_scale: usize,
    target_trace_size: Option<usize>,
    pcs: PcsChoice,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
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

        let pcs_label = match pcs {
            PcsChoice::Dory => "dory",
            PcsChoice::Hachi => "hachi",
        };
        tracing::info!("Running {bench_name} benchmark at scale 2^{bench_scale} with {pcs_label}");

        let guest_name = format!("{bench_name}-guest");
        let input = input_fn(bench_target);
        let (duration, proof_size, proof_size_comp, trace_length) = match pcs {
            PcsChoice::Dory => prove_example_with_trace_dory(
                &guest_name,
                input,
                max_trace_length,
                bench_name,
                bench_scale,
            ),
            PcsChoice::Hachi => {
                prove_example_with_trace_hachi(&guest_name, input, max_trace_length)
            }
        };

        let proving_hz = trace_length as f64 / duration.as_secs_f64();
        let padded_proving_hz = trace_length.next_power_of_two() as f64 / duration.as_secs_f64();
        println!(
            "{} (2^{}, {}): Prover completed in {:.2}s ({:.1} kHz / padded {:.1} kHz)",
            bench_name,
            bench_scale,
            pcs_label,
            duration.as_secs_f64(),
            proving_hz / 1000.0,
            padded_proving_hz / 1000.0,
        );

        let summary_line = format!(
            "{},{},{},{:.2},{},{:.2},{},{}\n",
            bench_name,
            pcs_label,
            bench_scale,
            duration.as_secs_f64(),
            trace_length.next_power_of_two(),
            padded_proving_hz,
            proof_size,
            proof_size_comp
        );

        let individual_file =
            format!("benchmark-runs/results/{bench_name}_{pcs_label}_{bench_scale}.csv");
        if let Err(e) = fs::write(&individual_file, &summary_line) {
            eprintln!("Failed to write individual result file {individual_file}: {e}");
        }

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
    pcs: PcsChoice,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match pcs {
        PcsChoice::Dory => prove_example_dory(example_name, serialized_input),
        PcsChoice::Hachi => prove_example_hachi(example_name, serialized_input),
    }
}

fn prove_example_dory(
    example_name: &str,
    serialized_input: Vec<u8>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut program = host::Program::new(example_name);
    let (bytecode, init_memory_state, _) = program.decode();
    let (_lazy_trace, trace, _, program_io) = program.trace(&serialized_input, &[], &[]);
    let padded_trace_len = (trace.len() + 1).next_power_of_two();
    drop(trace);

    let task = move || {
        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode,
            program_io.memory_layout.clone(),
            init_memory_state,
            padded_trace_len,
        );
        let preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover = RV64IMACProver::gen_from_elf(
            &preprocessing,
            elf_contents,
            &serialized_input,
            &[],
            &[],
            None,
            None,
            None,
        );
        let program_io = prover.program_io.clone();
        let (jolt_proof, _) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            shared_preprocessing,
            preprocessing.generators.to_verifier_setup(),
        );
        let verifier =
            RV64IMACVerifier::new(&verifier_preprocessing, jolt_proof, program_io, None, None)
                .expect("Failed to create verifier");
        verifier.verify().unwrap();
    };

    vec![(
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    )]
}

fn prove_example_hachi(
    example_name: &str,
    serialized_input: Vec<u8>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut program = host::Program::new(example_name);
    let (bytecode, init_memory_state, _) = program.decode();
    let (_lazy_trace, trace, _, program_io) = program.trace(&serialized_input, &[], &[]);
    let padded_trace_len = (trace.len() + 1).next_power_of_two();
    drop(trace);

    let task = move || {
        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode,
            program_io.memory_layout.clone(),
            init_memory_state,
            padded_trace_len,
        );
        let preprocessing =
            JoltProverPreprocessing::<_, HachiPcs>::new(shared_preprocessing.clone());

        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover = RV64IMACHachiProver::gen_from_elf(
            &preprocessing,
            elf_contents,
            &serialized_input,
            &[],
            &[],
            None,
            None,
            None,
        );
        let program_io = prover.program_io.clone();
        let (jolt_proof, _) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            shared_preprocessing,
            HachiPcs::setup_verifier(&preprocessing.generators),
        );
        let verifier =
            RV64IMACHachiVerifier::new(&verifier_preprocessing, jolt_proof, program_io, None, None)
                .expect("Failed to create verifier");
        verifier.verify().unwrap();
    };

    vec![(
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    )]
}

fn prove_example_with_trace_dory(
    example_name: &str,
    serialized_input: Vec<u8>,
    max_trace_length: usize,
    _bench_name: &str,
    _scale: usize,
) -> (std::time::Duration, usize, usize, usize) {
    let mut program = host::Program::new(example_name);
    let (bytecode, init_memory_state, _) = program.decode();
    let (_, trace, _, program_io) = program.trace(&serialized_input, &[], &[]);

    assert!(
        trace.len().next_power_of_two() <= max_trace_length,
        "Trace is longer than expected"
    );

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        program_io.memory_layout.clone(),
        init_memory_state,
        trace.len().next_power_of_two(),
    );
    let preprocessing = JoltProverPreprocessing::new(shared_preprocessing);

    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");

    let span = tracing::info_span!("E2E").entered();
    let prover = RV64IMACProver::gen_from_elf(
        &preprocessing,
        elf_contents,
        &serialized_input,
        &[],
        &[],
        None,
        None,
        None,
    );
    let now = Instant::now();
    let (jolt_proof, _) = prover.prove();
    let prove_duration = now.elapsed();
    drop(span);
    let proof_size = jolt_proof.serialized_size(ark_serialize::Compress::Yes);

    let stage8_size_compressed = jolt_proof
        .joint_opening_proof
        .serialized_size(ark_serialize::Compress::Yes);
    let stage8_size_uncompressed = jolt_proof
        .joint_opening_proof
        .serialized_size(ark_serialize::Compress::No);

    let commitments_size_compressed = jolt_proof
        .commitments
        .serialized_size(ark_serialize::Compress::Yes);
    let commitments_size_uncompressed = jolt_proof
        .commitments
        .serialized_size(ark_serialize::Compress::No);

    let proof_size_full_compressed = proof_size - stage8_size_compressed
        + (stage8_size_uncompressed / 3)
        - commitments_size_compressed
        + (commitments_size_uncompressed / 3);

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
    let verifier =
        RV64IMACVerifier::new(&verifier_preprocessing, jolt_proof, program_io, None, None)
            .expect("Failed to create verifier");
    verifier.verify().unwrap();

    (
        prove_duration,
        proof_size,
        proof_size_full_compressed,
        trace.len(),
    )
}

fn prove_example_with_trace_hachi(
    example_name: &str,
    serialized_input: Vec<u8>,
    max_trace_length: usize,
) -> (std::time::Duration, usize, usize, usize) {
    let mut program = host::Program::new(example_name);
    let (bytecode, init_memory_state, _) = program.decode();
    let (_, trace, _, program_io) = program.trace(&serialized_input, &[], &[]);

    assert!(
        trace.len().next_power_of_two() <= max_trace_length,
        "Trace is longer than expected"
    );

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        program_io.memory_layout.clone(),
        init_memory_state,
        trace.len().next_power_of_two(),
    );
    let preprocessing = JoltProverPreprocessing::<_, HachiPcs>::new(shared_preprocessing.clone());

    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");

    let span = tracing::info_span!("E2E").entered();
    let prover = RV64IMACHachiProver::gen_from_elf(
        &preprocessing,
        elf_contents,
        &serialized_input,
        &[],
        &[],
        None,
        None,
        None,
    );
    let now = Instant::now();
    let (jolt_proof, _) = prover.prove();
    let prove_duration = now.elapsed();
    drop(span);
    let proof_size = jolt_proof.serialized_size(ark_serialize::Compress::Yes);

    let verifier_preprocessing = JoltVerifierPreprocessing::new(
        shared_preprocessing,
        HachiPcs::setup_verifier(&preprocessing.generators),
    );
    let verifier =
        RV64IMACHachiVerifier::new(&verifier_preprocessing, jolt_proof, program_io, None, None)
            .expect("Failed to create verifier");
    verifier.verify().unwrap();

    (prove_duration, proof_size, proof_size, trace.len())
}
