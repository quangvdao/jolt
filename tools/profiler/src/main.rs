//! Jolt Profiler with recursion support
//!
//! This is a standalone profiler that can profile both regular Jolt proving
//! and recursion proving. It avoids the circular dependency issue between
//! jolt-core and jolt-recursion.

use std::any::Any;
use std::fs;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use ark_serialize::CanonicalSerialize;
use chrono::Local;
use clap::{Args, Parser, Subcommand, ValueEnum};
use jolt_core::host;
use jolt_core::poly::commitment::dory::{DoryGlobals, DoryLayout};
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::config::ProgramMode;
use jolt_core::zkvm::prover::JoltProverPreprocessing;
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifierPreprocessing};
use jolt_core::zkvm::{RV64IMACProver, RV64IMACVerifier};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::{fmt::format::FmtSpan, prelude::*, EnvFilter};

// Empirically measured cycles per operation for RV64IMAC
const CYCLES_PER_SHA256: f64 = 3396.0;
const CYCLES_PER_SHA3: f64 = 4330.0;
const CYCLES_PER_BTREEMAP_OP: f64 = 1550.0;
const CYCLES_PER_FIBONACCI_UNIT: f64 = 12.0;
const SAFETY_MARGIN: f64 = 0.9;

/// CLI-friendly layout enum that maps to DoryLayout
#[derive(Debug, Clone, Copy, Default, ValueEnum, PartialEq, Eq)]
pub enum LayoutArg {
    #[default]
    CycleMajor,
    AddressMajor,
}

impl From<LayoutArg> for DoryLayout {
    fn from(arg: LayoutArg) -> Self {
        match arg {
            LayoutArg::CycleMajor => DoryLayout::CycleMajor,
            LayoutArg::AddressMajor => DoryLayout::AddressMajor,
        }
    }
}

#[derive(Debug, Copy, Clone, ValueEnum, strum_macros::Display)]
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

#[derive(Parser, Debug)]
#[command(name = "jolt-profiler")]
#[command(about = "Jolt profiler with recursion support")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Profile(ProfileArgs),
    Benchmark(BenchmarkArgs),
}

#[derive(Args, Debug, Clone)]
struct ProfileArgs {
    #[clap(short, long, value_enum)]
    format: Option<Vec<Format>>,

    #[clap(long, value_enum)]
    name: BenchType,

    #[clap(long, default_value = "false")]
    committed: bool,

    #[clap(long, default_value = "false")]
    recursion: bool,

    #[clap(long, value_enum, default_value = "cycle-major")]
    layout: LayoutArg,

    #[clap(short, long)]
    scale: Option<usize>,

    #[clap(short, long)]
    target_trace_size: Option<usize>,
}

#[derive(Args, Debug)]
struct BenchmarkArgs {
    #[clap(flatten)]
    profile_args: ProfileArgs,
}

#[derive(Debug, Clone, ValueEnum, PartialEq)]
enum Format {
    Default,
    Chrome,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Profile(args) => trace(args),
        Commands::Benchmark(args) => run_benchmark(args),
    }
}

fn normalize_bench_name(name: &str) -> String {
    name.to_lowercase().replace(" ", "_")
}

fn setup_tracing(formats: Option<Vec<Format>>, trace_name: &str) -> Vec<Box<dyn Any>> {
    let mut layers = Vec::new();

    let log_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_filter(EnvFilter::from_default_env())
        .boxed();
    layers.push(log_layer);

    let mut guards: Vec<Box<dyn Any>> = vec![];

    if let Some(format) = formats {
        if format.contains(&Format::Default) {
            let collector_layer = tracing_subscriber::fmt::layer()
                .with_span_events(FmtSpan::CLOSE)
                .compact()
                .with_target(false)
                .with_file(false)
                .with_line_number(false)
                .with_thread_ids(false)
                .with_thread_names(false)
                .boxed();
            layers.push(collector_layer);
        }
        if format.contains(&Format::Chrome) {
            let trace_file = format!("benchmark-runs/perfetto_traces/{trace_name}.json");
            std::fs::create_dir_all("benchmark-runs/perfetto_traces").ok();
            let (chrome_layer, guard) = ChromeLayerBuilder::new()
                .include_args(true)
                .file(trace_file)
                .build();
            layers.push(chrome_layer.boxed());
            guards.push(Box::new(guard));
            tracing::info!(
                "Chrome tracing enabled. Output: benchmark-runs/perfetto_traces/{trace_name}.json"
            );
        }
    }

    tracing_subscriber::registry().with(layers).init();
    guards
}

fn trace(args: ProfileArgs) {
    let bench_name = normalize_bench_name(&args.name.to_string());
    let mode_suffix = if args.committed { "_committed" } else { "" };
    let recursion_suffix = if args.recursion { "_recursion" } else { "" };
    let layout_suffix = match args.layout {
        LayoutArg::CycleMajor => "",
        LayoutArg::AddressMajor => "_addr_major",
    };
    let scale_suffix = args.scale.map_or(String::new(), |s| format!("_2^{s}"));
    let timestamp = Local::now().format("%Y%m%d-%H%M");
    let trace_name = format!(
        "{bench_name}{mode_suffix}{recursion_suffix}{layout_suffix}{scale_suffix}_{timestamp}"
    );
    let _guards = setup_tracing(args.format.clone(), &trace_name);

    let layout: DoryLayout = args.layout.into();
    DoryGlobals::set_layout(layout);
    tracing::info!("Using Dory layout: {:?}", layout);

    let tasks = if let Some(scale) = args.scale {
        tracing::info!("Running with scale 2^{}", scale);
        master_benchmark(
            args.name,
            scale,
            args.target_trace_size,
            args.recursion,
            args.committed,
        )
    } else {
        benchmarks(args.name, args.committed, args.recursion)
    };

    for (span, bench) in tasks.into_iter() {
        span.in_scope(|| {
            bench();
            tracing::info!("Bench Complete");
        });
    }
}

fn run_benchmark(args: BenchmarkArgs) {
    let scale = match (args.profile_args.scale, args.profile_args.target_trace_size) {
        (Some(s), _) => s,
        (None, Some(target)) => target.next_power_of_two().trailing_zeros() as usize,
        (None, None) => {
            eprintln!("Error: Must provide either --scale or --target-trace-size");
            std::process::exit(1);
        }
    };

    let bench_name = normalize_bench_name(&args.profile_args.name.to_string());
    let layout_suffix = match args.profile_args.layout {
        LayoutArg::CycleMajor => "",
        LayoutArg::AddressMajor => "_addr_major",
    };
    let recursion_suffix = if args.profile_args.recursion {
        "_recursion"
    } else {
        ""
    };
    let trace_name = format!("{bench_name}{recursion_suffix}{layout_suffix}_{scale}");
    let _guards = setup_tracing(args.profile_args.format, &trace_name);

    let layout: DoryLayout = args.profile_args.layout.into();
    DoryGlobals::set_layout(layout);
    tracing::info!("Using Dory layout: {:?}", layout);

    for (span, bench) in master_benchmark(
        args.profile_args.name,
        scale,
        args.profile_args.target_trace_size,
        args.profile_args.recursion,
        args.profile_args.committed,
    )
    .into_iter()
    {
        span.in_scope(|| {
            bench();
            tracing::info!("Benchmark Complete");
        });
    }
}

// ============================================================================
// Benchmark implementations with actual recursion support
// ============================================================================

fn scale_to_target_ops(target_cycles: usize, cycles_per_op: f64) -> u32 {
    std::cmp::max(1, (target_cycles as f64 / cycles_per_op) as u32)
}

pub fn benchmarks(
    bench_type: BenchType,
    committed: bool,
    recursion: bool,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::BTreeMap => prove_example(
            "btreemap-guest",
            postcard::to_stdvec(&50u32).unwrap(),
            committed,
            recursion,
        ),
        BenchType::Sha2 => prove_example(
            "sha2-guest",
            postcard::to_stdvec(&vec![5u8; 2048]).unwrap(),
            committed,
            recursion,
        ),
        BenchType::Sha3 => prove_example(
            "sha3-guest",
            postcard::to_stdvec(&vec![5u8; 2048]).unwrap(),
            committed,
            recursion,
        ),
        BenchType::Sha2Chain => {
            let mut inputs = vec![];
            inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
            let iters = scale_to_target_ops(
                ((1 << 24) as f64 * SAFETY_MARGIN) as usize,
                CYCLES_PER_SHA256,
            );
            inputs.append(&mut postcard::to_stdvec(&iters).unwrap());
            prove_example("sha2-chain-guest", inputs, committed, recursion)
        }
        BenchType::Sha3Chain => {
            let mut inputs = vec![];
            inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
            inputs.append(&mut postcard::to_stdvec(&20u32).unwrap());
            prove_example("sha3-chain-guest", inputs, committed, recursion)
        }
        BenchType::Fibonacci => prove_example(
            "fibonacci-guest",
            postcard::to_stdvec(&400000u32).unwrap(),
            committed,
            recursion,
        ),
    }
}

pub fn master_benchmark(
    bench_type: BenchType,
    bench_scale: usize,
    target_trace_size: Option<usize>,
    recursion: bool,
    committed: bool,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
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

        tracing::info!("Running {bench_name} benchmark at scale 2^{bench_scale}");

        let guest_name = format!("{bench_name}-guest");
        let input = input_fn(bench_target);
        let (duration, proof_size, proof_size_comp, trace_length) =
            prove_example_with_trace(&guest_name, input, max_trace_length, recursion, committed);

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

        let individual_file = format!("benchmark-runs/results/{bench_name}_{bench_scale}.csv");
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
    committed: bool,
    recursion: bool,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
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

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);

        if recursion {
            // Actual recursion proving and verification
            let _recursion_span = tracing::info_span!("recursion_proving").entered();
            let recursion_proof = jolt_core::zkvm::recursion::prove_recursion::<Blake2bTranscript>(
                &verifier_preprocessing,
                program_io.clone(),
                None,
                &jolt_proof,
            )
            .expect("Failed to generate recursion proof");
            drop(_recursion_span);

            let _verify_span = tracing::info_span!("recursion_verification").entered();
            jolt_core::zkvm::recursion::verify_recursion::<Blake2bTranscript>(
                &verifier_preprocessing,
                program_io,
                None,
                &jolt_proof,
                &recursion_proof,
            )
            .expect("Recursion verification failed");
        } else {
            let verifier =
                RV64IMACVerifier::new(&verifier_preprocessing, jolt_proof, program_io, None, None)
                    .expect("Failed to create verifier");
            verifier.verify().unwrap();
        }
    };

    let span_name = if committed {
        "Example_E2E_Committed"
    } else {
        "Example_E2E"
    };
    vec![(
        tracing::info_span!("{}", span_name),
        Box::new(task) as Box<dyn FnOnce()>,
    )]
}

fn prove_example_with_trace(
    example_name: &str,
    serialized_input: Vec<u8>,
    max_trace_length: usize,
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

    // Calculate proof size stats before potentially consuming jolt_proof
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

    if recursion {
        // Actual recursion proving and verification
        let _recursion_span = tracing::info_span!("recursion_proving").entered();
        let recursion_proof = jolt_core::zkvm::recursion::prove_recursion::<Blake2bTranscript>(
            &verifier_preprocessing,
            program_io.clone(),
            None,
            &jolt_proof,
        )
        .expect("Failed to generate recursion proof");
        drop(_recursion_span);

        let _verify_span = tracing::info_span!("recursion_verification").entered();
        jolt_core::zkvm::recursion::verify_recursion::<Blake2bTranscript>(
            &verifier_preprocessing,
            program_io,
            None,
            &jolt_proof,
            &recursion_proof,
        )
        .expect("Recursion verification failed");
    } else {
        let verifier =
            RV64IMACVerifier::new(&verifier_preprocessing, jolt_proof, program_io, None, None)
                .expect("Failed to create verifier");
        verifier.verify().unwrap();
    }

    (
        prove_duration,
        proof_size,
        proof_size_full_compressed,
        trace.len(),
    )
}
