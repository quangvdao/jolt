use crate::field::JoltField;
use crate::host;
use crate::jolt::instruction::LookupTables;
use crate::jolt::vm::rv32i_vm::{RV32IJoltVM, C, M, RV32I};
use crate::jolt::vm::Jolt;
use crate::jolt::vm::{JoltProverPreprocessing, JoltTraceStep};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::hyperkzg::HyperKZG;
use crate::poly::commitment::zeromorph::Zeromorph;
use crate::subprotocols::shout::ShoutProof;
use crate::subprotocols::sparse_dense_shout::{
    prove_sparse_dense_shout, verify_sparse_dense_shout,
};
use crate::subprotocols::twist::{TwistAlgorithm, TwistProof};
use crate::utils::math::Math;
use crate::utils::transcript::{KeccakTranscript, Transcript};
use ark_bn254::{Bn254, Fr};
use ark_std::test_rng;
use common::rv_trace::JoltDevice;
use criterion::{black_box, BatchSize, Bencher, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_core::RngCore;
use rand_distr::{Distribution, Zipf};
use serde::Serialize;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum PCSType {
    Zeromorph,
    HyperKZG,
}

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Fibonacci,
    MemoryOps,
    Sha2,
    Sha3,
    Sha2Chain,
    Sha3Chain,
    Shout,
    SparseDenseShout,
    Twist,
}

#[allow(unreachable_patterns)] // good errors on new BenchTypes
pub fn benchmarks(
    pcs_type: PCSType,
    bench_type: BenchType,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match pcs_type {
        PCSType::Zeromorph => match bench_type {
            BenchType::Sha2 => sha2::<Fr, Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>(),
            BenchType::Sha3 => sha3::<Fr, Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>(),
            BenchType::Sha2Chain => {
                sha2chain::<Fr, Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>()
            }
            BenchType::Sha3Chain => {
                sha3chain::<Fr, Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>()
            }
            BenchType::Fibonacci => {
                fibonacci::<Fr, Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>()
            }
            BenchType::MemoryOps => memory_ops::<Fr, Zeromorph<Bn254, KeccakTranscript>, KeccakTranscript>(),
            BenchType::Shout => shout::<Fr, KeccakTranscript>(),
            BenchType::Twist => twist::<Fr, KeccakTranscript>(),
            BenchType::SparseDenseShout => sparse_dense_shout::<Fr, KeccakTranscript>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        PCSType::HyperKZG => match bench_type {
            BenchType::Sha2 => sha2::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(),
            BenchType::Sha3 => sha3::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(),
            BenchType::Sha2Chain => {
                sha2chain::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>()
            }
            BenchType::Sha3Chain => {
                sha3chain::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>()
            }
            BenchType::Fibonacci => {
                fibonacci::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>()
            }
            BenchType::MemoryOps => memory_ops::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(),
            BenchType::Shout => shout::<Fr, KeccakTranscript>(),
            BenchType::Twist => twist::<Fr, KeccakTranscript>(),
            BenchType::SparseDenseShout => sparse_dense_shout::<Fr, KeccakTranscript>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        _ => panic!("PCS Type does not have a mapping"),
    }
}

fn shout<F, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    let small_value_lookup_tables = F::compute_lookup_tables();
    F::initialize_lookup_tables(small_value_lookup_tables);

    let mut tasks = Vec::new();

    const TABLE_SIZE: usize = 1 << 16;
    const NUM_LOOKUPS: usize = 1 << 20;

    let mut rng = test_rng();

    let lookup_table: Vec<F> = (0..TABLE_SIZE).map(|_| F::random(&mut rng)).collect();
    let read_addresses: Vec<usize> = (0..NUM_LOOKUPS)
        .map(|_| rng.next_u32() as usize % TABLE_SIZE)
        .collect();

    let mut prover_transcript = ProofTranscript::new(b"test_transcript");
    let r_cycle: Vec<F> = prover_transcript.challenge_vector(NUM_LOOKUPS.log_2());

    let task = move || {
        let _proof = ShoutProof::prove(
            lookup_table,
            read_addresses,
            &r_cycle,
            &mut prover_transcript,
        );
    };

    tasks.push((
        tracing::info_span!("Shout d=1"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn sparse_dense_shout<F, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    let small_value_lookup_tables = F::compute_lookup_tables();
    F::initialize_lookup_tables(small_value_lookup_tables);

    let mut tasks = Vec::new();

    const WORD_SIZE: usize = 32;
    const LOG_K: usize = 2 * WORD_SIZE;
    const LOG_T: usize = 19;
    const T: u64 = 1 << LOG_T;

    let mut rng = StdRng::seed_from_u64(12345);

    let instructions: Vec<_> = (0..T)
        .map(|_| LookupTables::random(&mut rng, None))
        .collect();

    let mut prover_transcript = ProofTranscript::new(b"test_transcript");
    let r_cycle: Vec<F> = prover_transcript.challenge_vector(LOG_T);

    let task = move || {
        let (proof, rv_claim, ra_claims, flag_claims) = prove_sparse_dense_shout::<WORD_SIZE, _, _>(
            &instructions,
            r_cycle,
            &mut prover_transcript,
        );

        let mut verifier_transcript = ProofTranscript::new(b"test_transcript");
        let r_cycle: Vec<F> = verifier_transcript.challenge_vector(LOG_T);
        let verification_result = verify_sparse_dense_shout::<WORD_SIZE, _, _>(
            proof,
            LOG_K,
            LOG_T,
            r_cycle,
            rv_claim,
            ra_claims,
            flag_claims,
            &mut verifier_transcript,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    };

    tasks.push((
        tracing::info_span!("Sparse-dense shout d=4"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn twist<F, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    let small_value_lookup_tables = F::compute_lookup_tables();
    F::initialize_lookup_tables(small_value_lookup_tables);

    let mut tasks = Vec::new();

    const K: usize = 1 << 10;
    const T: usize = 1 << 20;
    const ZIPF_S: f64 = 0.0;
    let zipf = Zipf::new(K as u64, ZIPF_S).unwrap();

    let mut rng = test_rng();

    let mut registers = [0u32; K];
    let mut read_addresses: Vec<usize> = Vec::with_capacity(T);
    let mut read_values: Vec<u32> = Vec::with_capacity(T);
    let mut write_addresses: Vec<usize> = Vec::with_capacity(T);
    let mut write_values: Vec<u32> = Vec::with_capacity(T);
    let mut write_increments: Vec<i64> = Vec::with_capacity(T);
    for _ in 0..T {
        // Random read register
        let read_address = zipf.sample(&mut rng) as usize - 1;
        // Random write register
        let write_address = zipf.sample(&mut rng) as usize - 1;
        read_addresses.push(read_address);
        write_addresses.push(write_address);
        // Read the value currently in the read register
        read_values.push(registers[read_address]);
        // Random write value
        let write_value = rng.next_u32();
        write_values.push(write_value);
        // The increment is the difference between the new value and the old value
        let write_increment = (write_value as i64) - (registers[write_address] as i64);
        write_increments.push(write_increment);
        // Write the new value to the write register
        registers[write_address] = write_value;
    }

    let mut prover_transcript = ProofTranscript::new(b"test_transcript");
    let r: Vec<F> = prover_transcript.challenge_vector(K.log_2());
    let r_prime: Vec<F> = prover_transcript.challenge_vector(T.log_2());

    let task = move || {
        let _proof = TwistProof::prove(
            read_addresses,
            read_values,
            write_addresses,
            write_values,
            write_increments,
            r.clone(),
            r_prime.clone(),
            &mut prover_transcript,
            TwistAlgorithm::Local,
        );
    };

    tasks.push((
        tracing::info_span!("Twist d=1"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn fibonacci<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    prove_example::<u32, PCS, F, ProofTranscript>("fibonacci-guest", &9u32)
}

fn memory_ops<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    prove_example::<u32, PCS, F, ProofTranscript>("memory-ops-guest", &9u32)
}

fn sha2<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    prove_example::<Vec<u8>, PCS, F, ProofTranscript>("sha2-guest", &vec![5u8; 2048])
}

fn sha3<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    prove_example::<Vec<u8>, PCS, F, ProofTranscript>("sha3-guest", &vec![5u8; 2048])
}

#[allow(dead_code)]
fn serialize_and_print_size(name: &str, item: &impl ark_serialize::CanonicalSerialize) {
    use std::fs::File;
    let mut file = File::create("temp_file").unwrap();
    item.serialize_compressed(&mut file).unwrap();
    let file_size_bytes = file.metadata().unwrap().len();
    let file_size_kb = file_size_bytes as f64 / 1024.0;
    let file_size_mb = file_size_kb / 1024.0;
    println!("{name:<30} : {file_size_mb:.3} MB");
}

fn prove_example<T: Serialize, PCS, F, ProofTranscript>(
    example_name: &str,
    input: &T,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let mut tasks = Vec::new();
    let mut program = host::Program::new(example_name);
    let inputs = postcard::to_stdvec(input).unwrap();

    let task = move || {
        let (io_device, trace) = program.trace(&inputs);
        let (bytecode, memory_init) = program.decode();

        let preprocessing: crate::jolt::vm::JoltProverPreprocessing<C, F, PCS, ProofTranscript> =
            RV32IJoltVM::prover_preprocess(
                bytecode.clone(),
                io_device.memory_layout.clone(),
                memory_init,
                1 << 18,
                1 << 18,
                1 << 18,
            );

        let (jolt_proof, jolt_commitments, verifier_io_device, _) =
            <RV32IJoltVM as Jolt<_, PCS, C, M, ProofTranscript>>::prove(
                io_device,
                trace,
                preprocessing.clone(),
            );

        println!("Proof sizing:");
        serialize_and_print_size("jolt_commitments", &jolt_commitments);
        serialize_and_print_size("jolt_proof", &jolt_proof);
        serialize_and_print_size(" jolt_proof.r1cs", &jolt_proof.r1cs);
        serialize_and_print_size(" jolt_proof.bytecode", &jolt_proof.bytecode);
        serialize_and_print_size(
            " jolt_proof.read_write_memory",
            &jolt_proof.read_write_memory,
        );
        serialize_and_print_size(
            " jolt_proof.instruction_lookups",
            &jolt_proof.instruction_lookups,
        );

        let verification_result = RV32IJoltVM::verify(
            preprocessing.shared,
            jolt_proof,
            jolt_commitments,
            verifier_io_device,
            None,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    };

    tasks.push((
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn sha2chain<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let mut tasks = Vec::new();
    let mut program = host::Program::new("sha2-chain-guest");

    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&1000u32).unwrap());

    let task = move || {
        let (io_device, trace) = program.trace(&inputs);
        let (bytecode, memory_init) = program.decode();

        let preprocessing: crate::jolt::vm::JoltProverPreprocessing<C, F, PCS, ProofTranscript> =
            RV32IJoltVM::prover_preprocess(
                bytecode.clone(),
                io_device.memory_layout.clone(),
                memory_init,
                1 << 22,
                1 << 22,
                1 << 22,
            );

        let (jolt_proof, jolt_commitments, verifier_io_device, _) =
            <RV32IJoltVM as Jolt<_, PCS, C, M, ProofTranscript>>::prove(
                io_device,
                trace,
                preprocessing.clone(),
            );
        let verification_result = RV32IJoltVM::verify(
            preprocessing.shared,
            jolt_proof,
            jolt_commitments,
            verifier_io_device,
            None,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    };

    tasks.push((
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn sha3chain<F, PCS, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    let mut tasks = Vec::new();
    let mut program = host::Program::new("sha3-chain-guest");

    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&50u32).unwrap());

    let task = move || {
        let (io_device, trace) = program.trace(&inputs);
        let (bytecode, memory_init) = program.decode();

        let preprocessing: crate::jolt::vm::JoltProverPreprocessing<C, F, PCS, ProofTranscript> =
            RV32IJoltVM::prover_preprocess(
                bytecode.clone(),
                io_device.memory_layout.clone(),
                memory_init,
                1 << 22,
                1 << 22,
                1 << 22,
            );

        let (jolt_proof, jolt_commitments, verifier_io_device, _) =
            <RV32IJoltVM as Jolt<_, PCS, C, M, ProofTranscript>>::prove(
                io_device,
                trace,
                preprocessing.clone(),
            );
        let verification_result = RV32IJoltVM::verify(
            preprocessing.shared,
            jolt_proof,
            jolt_commitments,
            verifier_io_device,
            None,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    };

    tasks.push((
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

// Define a helper struct to hold data for each benchmark iteration.
// It needs to be Clone, and its fields are what the \'prove\' routine needs.
// Ensure C_CONST is correctly scoped or passed if this struct is made generic beyond this file.
#[derive(Clone)]
struct _ProveIterationData<
    const C_CONST: usize,
    F: JoltField,
    PCS: CommitmentScheme<PT, Field = F>, // Corrected ProofTranscript to PT
    PT: Transcript,                       // Renamed ProofTranscript to PT for brevity in struct def
> {
    preprocessing: JoltProverPreprocessing<C_CONST, F, PCS, PT>,
    io_device: JoltDevice,
    trace: Vec<JoltTraceStep<RV32I>>,
}

// New benchmark function for sha2-chain-guest, focusing on the prove step
fn _benchmark_jolt_prove_only_sha2_chain(c: &mut Criterion) {
    // Define types for this specific benchmark
    // These should align with types used in other benchmarks in the file for consistency
    type F = ark_bn254::Fr; // Assuming Fr as in other examples
    type PCSImpl = crate::poly::commitment::hyperkzg::HyperKZG<
        ark_bn254::Bn254,
        crate::utils::transcript::KeccakTranscript,
    >;
    type PTImpl = crate::utils::transcript::KeccakTranscript;

    // Use constants for C and M, potentially from crate::jolt::vm::rv32i_vm or define specific ones
    const C_CONST_VAL: usize = crate::jolt::vm::rv32i_vm::C; // Default C from the VM
    const M_CONST_VAL: usize = crate::jolt::vm::rv32i_vm::M; // Default M from the VM

    // --- Part 1: One-time setup for this benchmark ---
    let example_name_str = "sha2-chain-guest";
    println!(
        "Setting up benchmark for: {} (prove step only)",
        example_name_str
    ); // Log setup

    let mut program = host::Program::new(example_name_str);

    let mut inputs_bytes = vec![];
    inputs_bytes.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs_bytes.append(&mut postcard::to_stdvec(&1000u32).unwrap());

    let (bytecode, memory_init) = program.decode();

    // Perform one trace to get memory_layout for preprocessing
    let (initial_io_device_template, _) = program.trace(&inputs_bytes);

    println!(
        "Performing one-time prover preprocessing for {}...",
        example_name_str
    );
    let prover_preprocessing: JoltProverPreprocessing<C_CONST_VAL, F, PCSImpl, PTImpl> =
        RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            initial_io_device_template.memory_layout.clone(),
            memory_init.clone(),
            1 << 22, // Max trace length for bytecode (from existing sha2chain bench)
            1 << 22, // Max trace length for memory
            1 << 22, // Max trace length for instruction lookups
        );
    println!("Prover preprocessing complete for {}.", example_name_str);

    // --- Part 2: Using iter_batched to benchmark the prove step ---
    let benchmark_id = format!("Jolt_Prove_Only_{}", example_name_str);
    c.bench_function(&benchmark_id, |b: &mut Bencher| {
        b.iter_batched(
            // Setup for each batch/iteration:
            // This closure prepares the data needed for one execution of the routine.
            || {
                // Re-trace to get fresh io_device and trace as they are consumed by \'prove\'.
                // \'program\' and \'inputs_bytes\' are captured from the outer scope.
                // If \'program.trace\' was on \'&self\', \'program\' could be shared more easily.
                // Since it\'s `&mut self`, we might need to re-create `program` or ensure
                // it\'s handled correctly if iter_batched parallelizes setup.
                // For simplicity, we re-create program here. If `host::Program::new` is expensive,
                // this part of the setup should be minimized or `program` made more shareable.
                let mut current_program = host::Program::new(example_name_str);
                let (current_io_device, current_trace) = current_program.trace(&inputs_bytes);

                _ProveIterationData::<C_CONST_VAL, F, PCSImpl, PTImpl> {
                    preprocessing: prover_preprocessing.clone(), // Clone shared preprocessed data (Rc inside makes this cheap)
                    io_device: current_io_device,                // Fresh for this iteration
                    trace: current_trace,                        // Fresh for this iteration
                }
            },
            // Routine to benchmark:
            // This closure takes the data from the setup closure and executes the code to be measured.
            |data: _ProveIterationData<C_CONST_VAL, F, PCSImpl, PTImpl>| {
                // Call the Jolt::prove method
                let (_jolt_proof, _jolt_commitments, _verifier_io_device, _) =
                    <RV32IJoltVM as Jolt<F, PCSImpl, C_CONST_VAL, M_CONST_VAL, PTImpl>>::prove(
                        black_box(data.io_device),
                        black_box(data.trace),
                        black_box(data.preprocessing), // This is a cloned version
                    );
            },
            BatchSize::SmallInput, // Adjust batch size as needed (e.g., SmallInput, MediumInput, LargeInput)
        );
    });
    println!("Benchmark registration complete for: {}", benchmark_id);
}
