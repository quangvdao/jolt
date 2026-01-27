use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;
use clap::{Parser, Subcommand, ValueEnum};
use jolt_core::zkvm::recursion::RecursionArtifact;
use jolt_sdk::guest::program::Program as JoltGuestProgram;
use jolt_sdk::guest::{prover, verifier};
use jolt_sdk::host;
use jolt_sdk::{DoryGlobals, DoryLayout, JoltDevice, MemoryConfig, RV64IMACProof, Serializable};
use std::cmp::PartialEq;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{error, info};

/// CLI-friendly layout enum that maps to DoryLayout
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub enum LayoutArg {
    /// Cycle-major layout (default): index = address * T + cycle
    #[default]
    CycleMajor,
    /// Address-major layout: index = cycle * K + address
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

fn get_guest_src_dir() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let guest_src_dir = manifest_dir.join("guest").join("src");

    guest_src_dir.canonicalize().unwrap_or(guest_src_dir)
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate proofs for guest programs
    Generate {
        /// Example to run (fibonacci or muldiv)
        #[arg(long, value_name = "EXAMPLE")]
        example: String,
        /// Working directory for output files
        #[arg(long, value_name = "DIRECTORY", default_value = "output")]
        workdir: PathBuf,
        /// Use committed program mode (verifier only gets commitments, not full program)
        #[arg(long, default_value_t = false)]
        committed: bool,
        /// Enable recursion proving/verifying for the *inner* proofs (strict extension payload).
        #[arg(long, default_value_t = false)]
        recursion: bool,
        /// Dory matrix layout (cycle-major or address-major)
        #[arg(long, value_enum, default_value_t = LayoutArg::CycleMajor)]
        layout: LayoutArg,
        /// Target trace size as power of 2 (e.g., 25 for ~33M cycles). If not specified, uses default small input.
        #[arg(long, value_name = "POWER")]
        scale: Option<u8>,
    },
    /// Verify proofs and optionally embed them
    Verify {
        /// Example to verify (fibonacci or muldiv)
        #[arg(long, value_name = "EXAMPLE")]
        example: String,
        /// Working directory containing proof files
        #[arg(long, value_name = "DIRECTORY", default_value = "output")]
        workdir: PathBuf,
        /// Embed proof data to specified directory
        #[arg(long, value_name = "DIRECTORY", num_args = 0..=1)]
        embed: Option<Option<PathBuf>>,
        /// Use committed program mode (verifier only gets commitments, not full program)
        #[arg(long, default_value_t = false)]
        committed: bool,
        /// Enable cycle tracking markers in the recursion guest (off by default).
        #[arg(long, default_value_t = false)]
        cycle_tracking: bool,
        /// Dory matrix layout (cycle-major or address-major)
        #[arg(long, value_enum, default_value_t = LayoutArg::CycleMajor)]
        layout: LayoutArg,
    },
    /// Trace the execution of guest programs without attempting to prove them
    Trace {
        /// Example to trace (fibonacci or muldiv)
        #[arg(long, value_name = "EXAMPLE")]
        example: String,
        /// Working directory containing proof files
        #[arg(long, value_name = "DIRECTORY", default_value = "output")]
        workdir: PathBuf,
        /// Embed proof data to specified directory
        #[arg(long, value_name = "DIRECTORY", num_args = 0..=1)]
        embed: Option<Option<PathBuf>>,
        /// Trace to disk instead of memory (redues memory usage)
        #[arg(short = 'd', long = "disk", default_value_t = false)]
        trace_to_file: bool,
        /// Use committed program mode (verifier only gets commitments, not full program)
        #[arg(long, default_value_t = false)]
        committed: bool,
        /// Enable cycle tracking markers in the recursion guest (off by default).
        #[arg(long, default_value_t = false)]
        cycle_tracking: bool,
        /// Dory matrix layout (cycle-major or address-major)
        #[arg(long, value_enum, default_value_t = LayoutArg::CycleMajor)]
        layout: LayoutArg,
    },
    /// Debug proof deserialization
    Debug {
        /// Example to debug (fibonacci or muldiv)
        #[arg(long, value_name = "EXAMPLE")]
        example: String,
        /// Working directory containing proof files
        #[arg(long, value_name = "DIRECTORY", default_value = "output")]
        workdir: PathBuf,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GuestProgram {
    Fibonacci,
    Muldiv,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RunConfig {
    Prove,
    Trace,
    TraceToFile,
}

impl GuestProgram {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "fibonacci" => Some(GuestProgram::Fibonacci),
            "muldiv" => Some(GuestProgram::Muldiv),
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            GuestProgram::Fibonacci => "fibonacci-guest",
            GuestProgram::Muldiv => "muldiv-guest",
        }
    }

    fn func(&self) -> &'static str {
        match self {
            GuestProgram::Fibonacci => "fib",
            GuestProgram::Muldiv => "muldiv",
        }
    }

    fn inputs(&self) -> Vec<Vec<u8>> {
        match self {
            GuestProgram::Fibonacci => {
                vec![postcard::to_stdvec(&2u32).unwrap()]
            }
            GuestProgram::Muldiv => {
                vec![postcard::to_stdvec(&(10u32, 5u32, 2u32)).unwrap()]
            }
        }
    }

    /// Compute inputs scaled to achieve approximately `1 << scale` cycles.
    /// Uses empirical cycles-per-operation constants from benchmarks.
    fn inputs_for_scale(&self, scale: u8) -> Vec<Vec<u8>> {
        const CYCLES_PER_FIBONACCI_UNIT: f64 = 12.0;
        const SAFETY_MARGIN: f64 = 0.9; // Use 90% of max trace capacity

        let target_cycles = ((1usize << scale) as f64 * SAFETY_MARGIN) as usize;

        match self {
            GuestProgram::Fibonacci => {
                let n = std::cmp::max(1, (target_cycles as f64 / CYCLES_PER_FIBONACCI_UNIT) as u32);
                info!("Scaling fibonacci to n={n} for ~2^{scale} cycles");
                vec![postcard::to_stdvec(&n).unwrap()]
            }
            GuestProgram::Muldiv => {
                // Muldiv is a single operation, can't really scale it meaningfully
                // Just return the default input
                info!("Warning: muldiv cannot be scaled, using default input");
                vec![postcard::to_stdvec(&(10u32, 5u32, 2u32)).unwrap()]
            }
        }
    }

    /// Get max_trace_length based on scale, or default if no scale provided.
    fn get_max_trace_length_for_scale(&self, scale: Option<u8>) -> usize {
        match scale {
            Some(s) => 1usize << s,
            None => self.get_max_trace_length(false),
        }
    }

    fn get_memory_config(&self, use_embed: bool) -> MemoryConfig {
        match self {
            GuestProgram::Fibonacci => {
                if use_embed {
                    MemoryConfig {
                        max_input_size: 67108864, // 64MB
                        max_output_size: 4096,
                        max_untrusted_advice_size: 0,
                        max_trusted_advice_size: 0,
                        memory_size: 134217728, // 128MB
                        stack_size: 134217728,  // 128MB (total 256MB)
                        program_size: None,
                    }
                } else {
                    MemoryConfig {
                        max_input_size: 67108864, // 64MB
                        max_output_size: 4096,
                        max_untrusted_advice_size: 0,
                        max_trusted_advice_size: 0,
                        memory_size: 134217728, // 128MB
                        stack_size: 134217728,  // 128MB (total 256MB)
                        program_size: None,
                    }
                }
            }
            GuestProgram::Muldiv => {
                if use_embed {
                    MemoryConfig {
                        max_input_size: 67108864, // 64MB
                        max_output_size: 4096,
                        max_untrusted_advice_size: 0,
                        max_trusted_advice_size: 0,
                        memory_size: 134217728, // 128MB
                        stack_size: 134217728,  // 128MB (total 256MB)
                        program_size: None,
                    }
                } else {
                    MemoryConfig {
                        max_input_size: 67108864, // 64MB
                        max_output_size: 4096,
                        max_untrusted_advice_size: 0,
                        max_trusted_advice_size: 0,
                        memory_size: 134217728, // 128MB
                        stack_size: 134217728,  // 128MB (total 256MB)
                        program_size: None,
                    }
                }
            }
        }
    }

    fn get_max_trace_length(&self, use_embed: bool) -> usize {
        match self {
            GuestProgram::Fibonacci => {
                if use_embed {
                    67108864
                } else {
                    5000000
                }
            }
            GuestProgram::Muldiv => {
                if use_embed {
                    800000
                } else {
                    3000000
                }
            }
        }
    }
}

fn generate_provable_macro(guest: GuestProgram, use_embed: bool, output_dir: &Path) {
    let memory_config = guest.get_memory_config(use_embed);
    let max_trace_length = guest.get_max_trace_length(use_embed);

    let macro_content = format!(
        r#"macro_rules! provable_with_config {{
    ($item: item) => {{
        #[jolt::provable(
            max_input_size = {},
            max_output_size = {},
            max_untrusted_advice_size = {},
            max_trusted_advice_size = {},
            memory_size = {},
            stack_size = {},
            max_trace_length = {}
        )]
        $item
    }};
}}"#,
        memory_config.max_input_size,
        memory_config.max_output_size,
        memory_config.max_untrusted_advice_size,
        memory_config.max_trusted_advice_size,
        memory_config.memory_size,
        memory_config.stack_size,
        max_trace_length
    );

    let provable_macro_path = output_dir.join("provable_macro.rs");

    std::fs::create_dir_all(output_dir).unwrap();

    std::fs::write(&provable_macro_path, macro_content).unwrap();
    info!(
        "Generated {} with config: input={}, output={}, memory={}, stack={}, trace={}",
        provable_macro_path.display(),
        memory_config.max_input_size,
        memory_config.max_output_size,
        memory_config.memory_size,
        memory_config.stack_size,
        max_trace_length
    );
}

fn check_data_integrity(all_groups_data: &[u8]) -> (u32, u32) {
    info!("Checking data integrity...");

    // Expect canonical transport encoding:
    //   [preprocessing][u32 n][(device)(proof)(recursion_artifact?)]^n
    let mut cursor = std::io::Cursor::new(all_groups_data);

    let verifier_preprocessing =
        jolt_sdk::JoltVerifierPreprocessing::<jolt_sdk::F, jolt_sdk::PCS>::deserialize_compressed(
            &mut cursor,
        )
        .unwrap();
    let verifier_bytes = verifier_preprocessing.serialize_to_bytes().unwrap();
    info!(
        "✓ Verifier preprocessing deserialized successfully ({} bytes)",
        verifier_bytes.len()
    );

    let n = u32::deserialize_compressed(&mut cursor).unwrap();
    info!("✓ Number of proofs deserialized: {n}");

    for i in 0..n {
        match JoltDevice::deserialize_compressed(&mut cursor) {
            Ok(_) => info!("✓ Device {i} deserialized"),
            Err(e) => error!("✗ Failed to deserialize device {i}: {e:?}"),
        }
        match RV64IMACProof::deserialize_compressed(&mut cursor) {
            Ok(_) => info!("✓ Proof {i} deserialized"),
            Err(e) => error!("✗ Failed to deserialize proof {i}: {e:?}"),
        }
        match Option::<RecursionArtifact<jolt_sdk::FS>>::deserialize_compressed(&mut cursor) {
            Ok(_) => info!("✓ Recursion artifact {i} deserialized"),
            Err(e) => error!("✗ Failed to deserialize recursion artifact {i}: {e:?}"),
        }
    }

    let position = cursor.position() as usize;
    let all_data = cursor.into_inner();
    let remaining_data: Vec<u8> = all_data[position..].to_vec();
    info!("✓ Remaining data size: {} bytes", remaining_data.len());

    assert_eq!(
        remaining_data.len(),
        0,
        "Not all data was consumed during deserialization"
    );

    (n, remaining_data.len() as u32)
}

fn collect_guest_proofs(
    guest: GuestProgram,
    target_dir: &str,
    use_embed: bool,
    use_committed: bool,
    recursion: bool,
    scale: Option<u8>,
) -> Vec<u8> {
    info!("Starting collect_guest_proofs for {}", guest.name());
    info!("Using committed program mode: {use_committed}");
    info!("Scale: {:?}", scale);
    let max_trace_length = if use_embed {
        guest.get_max_trace_length(true)
    } else {
        guest.get_max_trace_length_for_scale(scale)
    };

    // This should match the example being run, it can cause layout issues if the guest's macro and our assumption here differ
    let memory_config = MemoryConfig {
        memory_size: 32768u64,
        ..Default::default()
    };

    info!("Creating program...");
    let mut program = host::Program::new(guest.name());
    program.set_func(guest.func());
    program.set_std(false);
    program.set_memory_config(memory_config);
    info!("Building program...");
    program.build(target_dir);
    info!("Getting ELF contents...");
    let elf_contents = program.get_elf_contents().unwrap();
    info!("Creating guest program...");
    let mut guest_prog = JoltGuestProgram::new(&elf_contents, &memory_config);
    guest_prog.elf = program.elf;

    info!("Preprocessing guest prover (committed={use_committed})...");
    let guest_prover_preprocessing = if use_committed {
        prover::preprocess_committed(&guest_prog, max_trace_length)
    } else {
        prover::preprocess(&guest_prog, max_trace_length)
    };
    info!("Preprocessing guest verifier...");
    let guest_verifier_preprocessing =
        jolt_sdk::JoltVerifierPreprocessing::from(&guest_prover_preprocessing);

    let inputs = match scale {
        Some(s) => guest.inputs_for_scale(s),
        None => guest.inputs(),
    };
    info!("Got inputs: {inputs:?}");
    let n = inputs.len() as u32;

    let mut all_groups_data: Vec<u8> = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut all_groups_data);
    let mut total_prove_time = 0.0;

    // Serialize verifier preprocessing and per-instance tuples.
    //
    // Default format:
    //   [preprocessing][u32 n][(device)(proof)(recursion_artifact?)]^n
    {
        guest_verifier_preprocessing
            .serialize_compressed(&mut cursor)
            .unwrap();
        u32::serialize_compressed(&n, &mut cursor).unwrap();
    }

    info!("Starting {} recursion with {}", guest.name(), n);

    for (i, input_bytes) in inputs.into_iter().enumerate() {
        info!("Processing input {i}: {:#?}", &input_bytes);

        let now = Instant::now();

        let mut output_bytes = vec![0; 4096];

        // Running tracing allows things like JOLT_BACKTRACE=1 to work properly
        info!("  Tracing...");
        guest_prog.memory_config.program_size = Some(
            guest_verifier_preprocessing
                .shared
                .memory_layout
                .program_size,
        );
        let (_, _, _, device_io) = guest_prog.trace(&input_bytes, &[], &[]);
        assert!(!device_io.panic, "Guest program panicked during tracing");

        info!("  Proving...");
        let (proof, io_device, _debug): (RV64IMACProof, _, _) = prover::prove(
            &guest_prog,
            &input_bytes,
            &[],
            &[],
            None,
            None,
            &mut output_bytes,
            &guest_prover_preprocessing,
        );
        let prove_time = now.elapsed().as_secs_f64();
        total_prove_time += prove_time;
        info!(
            "  Input: {:?}, Prove time: {:.3}s",
            &input_bytes, prove_time
        );

        let recursion_artifact: Option<RecursionArtifact<jolt_sdk::FS>> = if recursion {
            info!("  Generating recursion artifact...");
            Some(
                jolt_core::zkvm::recursion::prove_recursion::<jolt_sdk::FS>(
                    &guest_verifier_preprocessing,
                    io_device.clone(),
                    None,
                    &proof,
                )
                .expect("Failed to generate recursion artifact"),
            )
        } else {
            None
        };

        io_device.serialize_compressed(&mut cursor).unwrap();
        proof.serialize_compressed(&mut cursor).unwrap();
        recursion_artifact
            .serialize_compressed(&mut cursor)
            .unwrap();

        info!("  Verifying...");
        if let Some(ref recursion_artifact) = recursion_artifact {
            let is_valid = jolt_core::zkvm::recursion::verify_recursion::<jolt_sdk::FS>(
                &guest_verifier_preprocessing,
                io_device,
                None,
                &proof,
                recursion_artifact,
            )
            .is_ok();
            info!("  Recursion verification result: {is_valid}");
        } else {
            // Standard verification (no recursion)
            let is_valid = verifier::verify(
                &input_bytes,
                None,
                &output_bytes,
                proof,
                &guest_verifier_preprocessing,
            )
            .is_ok();
            info!("  Verification result: {is_valid}");
        }
    }
    info!("Total prove time: {total_prove_time:.3}s");

    info!("Total data size: {} bytes", all_groups_data.len());
    all_groups_data
}

fn debug_deserialize_proof_fields(proof_file: &Path) {
    use ark_serialize::CanonicalDeserialize;
    use jolt_sdk::{JoltDevice, RV64IMACProof};

    info!("Starting detailed field-by-field deserialization debug...");

    let proof_data = std::fs::read(proof_file).unwrap();
    let mut cursor = std::io::Cursor::new(&proof_data);

    // First deserialize the verifier preprocessing and count
    info!("Step 1: Deserializing verifier preprocessing...");
    match jolt_sdk::JoltVerifierPreprocessing::<jolt_sdk::F, jolt_sdk::PCS>::deserialize_compressed(
        &mut cursor,
    ) {
        Ok(preprocessing) => {
            let bytes = preprocessing.serialize_to_bytes().unwrap();
            info!("✓ Verifier preprocessing OK ({} bytes)", bytes.len());
        }
        Err(e) => {
            panic!("✗ FAILED at verifier preprocessing: {:?}", e);
        }
    }

    info!("Step 2: Deserializing proof count...");
    let n = match u32::deserialize_compressed(&mut cursor) {
        Ok(count) => {
            info!("✓ Proof count: {}", count);
            count
        }
        Err(e) => {
            panic!("✗ FAILED at proof count: {:?}", e);
        }
    };

    // Check cursor position
    let position = cursor.position() as usize;
    info!(
        "Current cursor position after preprocessing + count: {} bytes",
        position
    );

    info!("Step 3: Attempting to deserialize {} proof(s)...", n);
    for i in 0..n {
        info!(
            "Device {}: Attempting JoltDevice deserialization at position {}...",
            i,
            cursor.position()
        );

        match JoltDevice::deserialize_compressed(&mut cursor) {
            Ok(device) => {
                info!("✓ Device {} deserialized successfully", i);
                info!(
                    "  Memory layout size: {:?}",
                    device.memory_layout.memory_size
                );
                info!("  Panic state: {:?}", device.panic);
            }
            Err(e) => {
                error!(
                    "✗ FAILED to deserialize device {} at position {}: {:?}",
                    i,
                    cursor.position(),
                    e
                );

                // Try to read some bytes at current position to debug
                let current_pos = cursor.position() as usize;
                let bytes = cursor.get_ref();
                if current_pos < bytes.len() {
                    let next_bytes = &bytes[current_pos..current_pos.min(current_pos + 32)];
                    error!(
                        "  Next {} bytes at position {}: {:?}",
                        next_bytes.len(),
                        current_pos,
                        next_bytes
                    );

                    // Try to interpret as length
                    if next_bytes.len() >= 8 {
                        let mut length_bytes = [0u8; 8];
                        length_bytes.copy_from_slice(&next_bytes[0..8]);
                        let length = u64::from_le_bytes(length_bytes);
                        error!("  Interpreted as u64 length: {} (0x{:x})", length, length);
                        if length > 1_000_000_000_000 {
                            error!("  WARNING: This looks like an unreasonably large value!");
                        }
                    }
                }
                panic!("Cannot continue after device deserialization failure");
            }
        }

        info!(
            "Proof {}: Starting JoltProof deserialization at position {}...",
            i,
            cursor.position()
        );
        match RV64IMACProof::deserialize_compressed(&mut cursor) {
            Ok(proof) => {
                info!("✓ Proof {} deserialized successfully", i);
                info!("  Trace length: {}", proof.trace_length);
                info!("  RAM K: {}", proof.ram_K);
                info!("  Bytecode K: {}", proof.bytecode_K);
            }
            Err(e) => {
                error!(
                    "✗ FAILED to deserialize proof {} at position {}: {:?}",
                    i,
                    cursor.position(),
                    e
                );
                panic!("Cannot continue after proof deserialization failure");
            }
        }

        info!(
            "Recursion artifact {}: Attempting Option<RecursionArtifact> deserialization at position {}...",
            i,
            cursor.position()
        );
        match Option::<RecursionArtifact<jolt_sdk::FS>>::deserialize_compressed(&mut cursor) {
            Ok(_rec) => {
                info!("✓ Recursion artifact {} deserialized successfully", i);
            }
            Err(e) => {
                error!(
                    "✗ FAILED to deserialize recursion artifact {} at position {}: {:?}",
                    i,
                    cursor.position(),
                    e
                );
                panic!("Cannot continue after recursion artifact deserialization failure");
            }
        }
    }

    let final_position = cursor.position() as usize;
    let total_bytes = cursor.get_ref().len();
    info!(
        "Final cursor position: {} / {} bytes",
        final_position, total_bytes
    );
    info!("Remaining bytes: {}", total_bytes - final_position);

    if final_position < total_bytes {
        error!(
            "WARNING: {} bytes remain unread!",
            total_bytes - final_position
        );
    }

    info!("Debug deserialization complete!");
}

fn generate_embedded_bytes(guest: GuestProgram, guest_bytes: &[u8], n: u32, output_dir: &Path) {
    info!(
        "Generating embedded bytes for {} guest program...",
        guest.name()
    );

    let mut output = String::new();
    output.push_str(&format!(
        "// Generated embedded bytes for {} recursion guest\n",
        guest.name()
    ));
    output.push_str("pub static EMBEDDED_BYTES: &[u8] = &[\n");

    for (i, byte) in guest_bytes.iter().enumerate() {
        if i > 0 && i % 16 == 0 {
            output.push('\n');
        }
        output.push_str(&format!("0x{byte:02x}, "));
    }

    output.push_str("\n];\n");
    output.push_str(&format!("// Total embedded bytes: {}\n", guest_bytes.len()));
    output.push_str(&format!("// Number of proofs: {n}\n"));

    std::fs::create_dir_all(output_dir).unwrap();

    let filename = output_dir.join("embedded_bytes.rs");
    std::fs::write(&filename, output).unwrap();
    info!("Embedded bytes written to {}", filename.display());
}

fn save_proof_data(guest: GuestProgram, all_groups_data: &[u8], workdir: &Path) {
    info!(
        "Saving proof data for {} to {}",
        guest.name(),
        workdir.display()
    );

    std::fs::create_dir_all(workdir).unwrap();

    let proof_file = workdir.join(format!("{}_proofs.bin", guest.name()));
    std::fs::write(&proof_file, all_groups_data).unwrap();

    info!("Proof data saved to {}", proof_file.display());
    info!("Total proof data size: {} bytes", all_groups_data.len());
}

fn load_proof_data(guest: GuestProgram, workdir: &Path) -> Vec<u8> {
    info!(
        "Loading proof data for {} from {}",
        guest.name(),
        workdir.display()
    );

    let proof_file = workdir.join(format!("{}_proofs.bin", guest.name()));

    if !proof_file.exists() {
        panic!("Proof file not found: {}", proof_file.display());
    }

    let proof_data = std::fs::read(&proof_file).unwrap();
    info!(
        "Loaded proof data from {} ({} bytes)",
        proof_file.display(),
        proof_data.len()
    );

    proof_data
}

fn generate_proofs(
    guest: GuestProgram,
    workdir: &Path,
    use_committed: bool,
    recursion: bool,
    layout: DoryLayout,
    scale: Option<u8>,
) {
    info!("Generating proofs for {} guest program...", guest.name());
    info!("Using committed program mode: {use_committed}");
    info!("Using Dory layout: {layout:?}");
    info!("Scale: {:?}", scale);

    // Set the Dory layout before any preprocessing
    DoryGlobals::set_layout(layout);

    let target_dir = "/tmp/jolt-guest-targets";

    // Collect guest proofs
    let all_groups_data =
        collect_guest_proofs(guest, target_dir, false, use_committed, recursion, scale);

    // Save proof data
    save_proof_data(guest, &all_groups_data, workdir);

    info!("Proof generation completed for {}", guest.name());
}

fn run_recursion_proof(
    guest: GuestProgram,
    run_config: RunConfig,
    input_bytes: Vec<u8>,
    memory_config: MemoryConfig,
    mut max_trace_length: usize,
    _use_committed: bool, // Note: committed mode only applies to inner proof, not recursion guest
    layout: DoryLayout,
    cycle_tracking: bool,
) {
    let target_dir = "/tmp/jolt-guest-targets";

    // Set the Dory layout before any preprocessing
    DoryGlobals::set_layout(layout);

    // Note: We always use Full mode for the recursion guest itself.
    // The --committed flag controls whether the inner proof (fibonacci/muldiv) uses committed mode.

    let mut program = host::Program::new("recursion-guest");
    program.set_func("verify");
    program.set_std(true);
    program.set_memory_config(memory_config);
    if cycle_tracking {
        program.add_guest_feature("cycle-tracking");
    }
    program.build(target_dir);
    let elf_contents = program.get_elf_contents().unwrap();
    let mut recursion = JoltGuestProgram::new(&elf_contents, &memory_config);
    recursion.elf = program.elf;

    if run_config == RunConfig::Trace || run_config == RunConfig::TraceToFile {
        // shorten the max_trace_length for tracing only. Speeds up setup time for tracing purposes.
        max_trace_length = 0;
    }
    // Always use Full mode for the recursion guest (outer proof)
    let recursion_prover_preprocessing = prover::preprocess(&recursion, max_trace_length);
    let recursion_verifier_preprocessing =
        jolt_sdk::JoltVerifierPreprocessing::from(&recursion_prover_preprocessing);

    // update program_size in memory_config now that we know it
    recursion.memory_config.program_size = Some(
        recursion_verifier_preprocessing
            .shared
            .memory_layout
            .program_size,
    );

    let mut output_bytes = vec![
        0;
        recursion_verifier_preprocessing
            .shared
            .memory_layout
            .max_output_size as usize
    ];
    match run_config {
        RunConfig::Prove => {
            // Reset DoryGlobals again before proving - the trace may have polluted them
            // when running the inner proof's recursion verification inside the guest.
            DoryGlobals::reset();
            info!("DoryGlobals reset before proving (inner verification may have polluted)");

            let (proof, _io_device, _debug): (RV64IMACProof, _, _) = prover::prove(
                &recursion,
                &input_bytes,
                &[],
                &[],
                None,
                None,
                &mut output_bytes,
                &recursion_prover_preprocessing,
            );
            let is_valid = verifier::verify(
                &input_bytes,
                None,
                &output_bytes,
                proof,
                &recursion_verifier_preprocessing,
            )
            .is_ok();
            let rv = postcard::from_bytes::<u32>(&output_bytes).unwrap();
            info!("  Recursion verification result: {rv}");
            info!("  Recursion verification result: {is_valid}");
        }
        RunConfig::Trace => {
            info!("  Trace-only mode: Skipping proof generation and verification.");
            let (_, _, _, io_device) = recursion.trace(&input_bytes, &[], &[]);
            let rv = postcard::from_bytes::<u32>(&io_device.outputs).unwrap_or(0);
            info!("  Recursion output (trace-only): {rv}");
        }
        RunConfig::TraceToFile => {
            info!("  Trace-only mode: Skipping proof generation and verification. Tracing to file: /tmp/{}.trace", guest.name());
            let (_, io_device) = recursion.trace_to_file(
                &input_bytes,
                &[],
                &[],
                &format!("/tmp/{}.trace", guest.name()).into(),
            );
            let rv = postcard::from_bytes::<u32>(&io_device.outputs).unwrap_or(0);
            info!("  Recursion output (trace-only): {rv}");
        }
    }
}

fn verify_proofs(
    guest: GuestProgram,
    use_embed: bool,
    workdir: &Path,
    output_dir: &Path,
    run_config: RunConfig,
    use_committed: bool,
    layout: DoryLayout,
    cycle_tracking: bool,
) {
    info!("Verifying proofs for {} guest program...", guest.name());
    info!("Using embed mode: {use_embed}");
    info!("Using committed program mode: {use_committed}");
    info!("Using Dory layout: {layout:?}");

    generate_provable_macro(guest, use_embed, output_dir);

    let all_groups_data = load_proof_data(guest, workdir);

    let (n, _remaining) = check_data_integrity(&all_groups_data);

    // Decompress+convert (native, validated) into guest-optimized encoding.
    let guest_bytes =
        jolt_sdk::decompression::decompress_transport_bytes_to_guest_bytes(&all_groups_data)
            .expect("decompress transport -> guest bytes");

    if use_embed {
        info!("Running {} recursion with embedded bytes...", guest.name());

        generate_embedded_bytes(guest, &guest_bytes, n, output_dir);

        let memory_config = guest.get_memory_config(use_embed);

        let input_bytes = vec![];
        info!("Using empty input bytes (embedded mode)");

        // Reset DoryGlobals before running recursion proof to avoid pollution
        // from the inner proof's context (which may have different parameters).
        DoryGlobals::reset();

        // The recursion guest has a MUCH larger trace than the inner guest.
        let recursion_max_trace_length = 1 << 30; // 1B cycles (safe upper bound)
        info!("Using recursion max_trace_length: {recursion_max_trace_length}");

        run_recursion_proof(
            guest,
            run_config,
            input_bytes,
            memory_config,
            recursion_max_trace_length,
            use_committed,
            layout,
            cycle_tracking,
        );
    } else {
        info!("Running {} recursion with input data...", guest.name());

        info!("Testing basic serialization/deserialization...");
        let test_input_bytes = postcard::to_stdvec(&guest_bytes).unwrap();
        let test_deserialized: Vec<u8> = postcard::from_bytes(&test_input_bytes).unwrap();
        assert_eq!(guest_bytes, test_deserialized);
        info!("Basic serialization/deserialization test passed!");

        let mut input_bytes = vec![];
        input_bytes.append(&mut postcard::to_stdvec(&guest_bytes.as_slice()).unwrap());

        info!("Serialized input size: {} bytes", input_bytes.len());
        let memory_config = guest.get_memory_config(use_embed);

        assert!(
            input_bytes.len() < memory_config.max_input_size as usize,
            "Input size is too large"
        );

        // Reset DoryGlobals before running recursion proof to avoid pollution
        // from the inner proof's context (which may have different parameters).
        info!("Resetting DoryGlobals before recursion proof (non-embed path)");
        DoryGlobals::reset();
        info!("DoryGlobals reset complete");

        // The recursion guest has a MUCH larger trace than the inner guest.
        // Use the max_trace_length from the provable_macro.rs config (generated earlier)
        // which is sized for the recursion verifier's actual cycle count.
        // With --recursion: ~300M cycles, without: ~1.1B cycles
        // We use the memory_config.max_trace_length which was set to 5M for the inner guest,
        // but we need a larger value for the recursion guest itself.
        let recursion_max_trace_length = 1 << 30; // 1B cycles (safe upper bound)
        info!("Using recursion max_trace_length: {recursion_max_trace_length}");

        run_recursion_proof(
            guest,
            run_config,
            input_bytes,
            memory_config,
            recursion_max_trace_length,
            use_committed,
            layout,
            cycle_tracking,
        );
    }
}

fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Generate {
            example,
            workdir,
            committed,
            recursion,
            layout,
            scale,
        }) => {
            let guest = match GuestProgram::from_str(example) {
                Some(guest) => guest,
                None => {
                    info!("Unknown example: {example}. Supported examples: fibonacci, muldiv");
                    return;
                }
            };
            generate_proofs(
                guest,
                workdir,
                *committed,
                *recursion,
                (*layout).into(),
                *scale,
            );
        }
        Some(Commands::Verify {
            example,
            workdir,
            embed,
            committed,
            cycle_tracking,
            layout,
        }) => {
            let guest = match GuestProgram::from_str(example) {
                Some(guest) => guest,
                None => {
                    info!("Unknown example: {example}. Supported examples: fibonacci, muldiv");
                    return;
                }
            };
            let output_dir = embed
                .as_ref()
                .and_then(|inner| inner.as_ref())
                .cloned()
                .unwrap_or_else(get_guest_src_dir);
            verify_proofs(
                guest,
                embed.is_some(),
                workdir,
                &output_dir,
                RunConfig::Prove,
                *committed,
                (*layout).into(),
                *cycle_tracking,
            );
        }
        Some(Commands::Trace {
            example,
            workdir,
            embed,
            trace_to_file,
            committed,
            cycle_tracking,
            layout,
        }) => {
            let guest = match GuestProgram::from_str(example) {
                Some(guest) => guest,
                None => {
                    info!("Unknown example: {example}. Supported examples: fibonacci, muldiv");
                    return;
                }
            };
            let output_dir = embed
                .as_ref()
                .and_then(|inner| inner.as_ref())
                .cloned()
                .unwrap_or_else(get_guest_src_dir);
            let run_config = if *trace_to_file {
                RunConfig::TraceToFile
            } else {
                RunConfig::Trace
            };
            verify_proofs(
                guest,
                embed.is_some(),
                workdir,
                &output_dir,
                run_config,
                *committed,
                (*layout).into(),
                *cycle_tracking,
            );
        }
        Some(Commands::Debug { example, workdir }) => {
            let guest = match GuestProgram::from_str(example) {
                Some(guest) => guest,
                None => {
                    info!("Unknown example: {example}. Supported examples: fibonacci, muldiv");
                    return;
                }
            };
            let proof_file = workdir.join(format!("{}_proofs.bin", guest.name()));
            debug_deserialize_proof_fields(&proof_file);
        }
        None => {
            info!("No subcommand specified. Available commands:");
            info!("  generate --example <fibonacci|muldiv> [--workdir <DIR>] [--committed] [--recursion] [--layout <cycle-major|address-major>] [--scale <POWER>]");
            info!("  verify --example <fibonacci|muldiv> [--workdir <DIR>] [--embed <DIR>] [--committed] [--cycle-tracking] [--layout <cycle-major|address-major>]");
            info!("  trace --example <fibonacci|muldiv> [--workdir <DIR>] [--embed <DIR>] [--committed] [--cycle-tracking] [--layout <cycle-major|address-major>]");
            info!("");
            info!("Examples:");
            info!("  cargo run --release -- generate --example fibonacci");
            info!("  cargo run --release -- generate --example fibonacci --workdir ./output --committed");
            info!("  cargo run --release -- generate --example fibonacci --layout address-major");
            info!("  cargo run --release -- generate --example fibonacci --scale 25 --committed --layout address-major --recursion");
            info!("  cargo run --release -- verify --example fibonacci");
            info!("  cargo run --release -- verify --example fibonacci --workdir ./output --embed --committed");
            info!("  cargo run --release -- trace --example fibonacci --cycle-tracking");
            info!(
                "  cargo run --release -- trace --example fibonacci --embed --layout address-major"
            );
        }
    }
}
