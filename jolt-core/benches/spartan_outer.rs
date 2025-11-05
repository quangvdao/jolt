use ark_bn254::Fr;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, SamplingMode};
use jolt_core::{
    poly::commitment::dory::DoryCommitmentScheme,
    subprotocols::sumcheck::BatchedSumcheck,
    subprotocols::univariate_skip::{prove_uniskip_round, UniSkipState},
    transcripts::{KeccakTranscript, Transcript},
    utils::math::Math,
    zkvm::{
        dag::state_manager::StateManager,
        spartan::{outer::OuterUniSkipInstanceProver, outer_streaming::OuterRemainingStreamingSumcheckProver},
        spartan::SpartanDagProver,
        witness::AllCommittedPolynomials,
        JoltProverPreprocessing, JoltRV32IM,
    },
    poly::opening_proof::ProverOpeningAccumulator,
    subprotocols::sumcheck_prover::SumcheckInstanceProver
};
use rayon::prelude::*;


type F = Fr;
type PCS = DoryCommitmentScheme;
type ProofTranscript = KeccakTranscript;

fn setup_for_spartan(
    program_name: &str,
    num_iterations: u32,
) -> (
    StateManager<'static, F, PCS>,
    usize,
    ProverOpeningAccumulator<F>,
    ProofTranscript,
) {
    let mut program = jolt_core::host::Program::new(program_name);

    let inputs = if program_name.contains("chain") {
        let initial_hash_data = [7u8; 32];
        let guest_input_tuple = (initial_hash_data, num_iterations);
        postcard::to_stdvec(&guest_input_tuple).unwrap()
    } else {
        let simple_input = num_iterations as u64;
        postcard::to_stdvec(&simple_input).unwrap()
    };

    let (bytecode, memory_init, _) = program.decode();
    let (lazy_trace, mut trace, final_memory_state, mut io_device) =
        program.trace(&inputs, &[], &[]);

    let padded_trace_length = (trace.len() + 1).next_power_of_two();
    trace.resize(padded_trace_length, tracer::instruction::Cycle::NoOp);

    let preprocessing: JoltProverPreprocessing<F, PCS> = JoltRV32IM::prover_preprocess(
        bytecode,
        io_device.memory_layout.clone(),
        memory_init,
        padded_trace_length,
    );

    // This is needed to initialize the global list of committed polynomials, required for witness generation.
    let ram_d = jolt_core::zkvm::witness::compute_d_parameter(
        io_device.memory_layout.memory_end as usize,
    );
    let bytecode_d = jolt_core::zkvm::witness::compute_d_parameter(
        preprocessing.shared.bytecode.code_size,
    );
    let _all_committed_polys_handle = AllCommittedPolynomials::initialize(ram_d, bytecode_d);

    // truncate trailing zeros on device outputs (defensive)
    io_device.outputs.truncate(
        io_device
            .outputs
            .iter()
            .rposition(|&b| b != 0)
            .map_or(0, |pos| pos + 1),
    );

    // Build state manager for the DAG-based prover
    // Note: we leak preprocessing to extend its lifetime to 'static for the benchmark
    let preprocessing_leaked: &'static JoltProverPreprocessing<F, PCS> = Box::leak(Box::new(preprocessing));
    let state_manager: StateManager<'static, F, PCS> = StateManager::new_prover(
        preprocessing_leaked,
        lazy_trace,
        trace,
        io_device,
        None,
        final_memory_state,
    );

    let mut transcript = ProofTranscript::new(b"Jolt transcript");
    // Initialize opening accumulator and preamble here for callers
    let opening_bits = state_manager.get_trace_len().log_2();
    let mut opening_accumulator =
        jolt_core::poly::opening_proof::ProverOpeningAccumulator::<F>::new(opening_bits);
    // FS preamble must be applied before any challenges are derived
    state_manager.fiat_shamir_preamble(&mut transcript);

    (state_manager, padded_trace_length, opening_accumulator, transcript)
}

fn bench_spartan_sumcheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spartan Sumcheck");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    // powers of 2
    let num_iters = [8, 16, 32, 64, 128, 256, 512, 1024, 2048];

    for num_iterations in num_iters {
        let bench_name = format!("sha2-chain-{}", num_iterations);

        // group.bench_function(&format!("svo/{}", bench_name), |b| {
        //     b.iter_batched(
        //         || setup_for_spartan("sha2-chain-guest", num_iterations),
        //         |(
        //             flattened_polys,
        //             uniform_constraints,
        //             num_rounds,
        //             padded_num_constraints,
        //             mut transcript,
        //         )| {
        //             let tau: Vec<F> = (0..num_rounds)
        //                 .map(|_| F::from(rand::random::<u64>()))
        //                 .collect();

        //             black_box(
        //                 SumcheckInstanceProof::prove_spartan_small_value::<NUM_SVO_ROUNDS>(
        //                     num_rounds,
        //                     padded_num_constraints,
        //                     &uniform_constraints,
        //                     &flattened_polys,
        //                     &tau,
        //                     &mut transcript,
        //                 ),
        //             );
        //         },
        //         BatchSize::SmallInput,
        //     );
        // });

        group.bench_function(&format!("outer-current/{}", bench_name), |b| {
            b.iter_batched(
                || setup_for_spartan("sha2-chain-guest", num_iterations),
                |(mut state_manager, padded_trace_length, mut opening_accumulator, mut transcript)| {

                    // Stage 1 (Outer): UniSkip first round + remaining rounds
                    let mut spartan_dag = SpartanDagProver::<F>::new(padded_trace_length);
                    let _first_round = spartan_dag.stage1_uni_skip(
                        &mut state_manager,
                        &mut opening_accumulator,
                        &mut transcript,
                    );
                    let mut instances = spartan_dag.stage1_instances(
                        &mut state_manager,
                        &mut opening_accumulator,
                        &mut transcript,
                    );
                    let mut instance_refs: Vec<_> =
                        instances.iter_mut().map(|b| &mut **b).collect();
                    black_box(BatchedSumcheck::prove(
                        instance_refs,
                        &mut opening_accumulator,
                        &mut transcript,
                    ));
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function(&format!("outer-streaming/{}", bench_name), |b| {
            b.iter_batched(
                || setup_for_spartan("sha2-chain-guest", num_iterations),
                |(mut state_manager, _padded_trace_length, mut opening_accumulator, mut transcript)| {

                    // Uni-skip first round (manual, to get UniSkipState)
                    let num_rows_bits = state_manager.get_trace_len().log_2();
                    let tau = transcript.challenge_vector_optimized::<F>(num_rows_bits);
                    let mut uniskip_instance = OuterUniSkipInstanceProver::gen(&mut state_manager, &tau);
                    let (_first_round_proof, r0, claim_after_first) =
                        prove_uniskip_round(&mut uniskip_instance, &mut transcript);
                    let uni = UniSkipState { claim_after_first, r0, tau };

                    // Streaming outer-remaining instance
                    let num_cycles_bits = state_manager.get_trace_len().log_2();
                    let mut instance = OuterRemainingStreamingSumcheckProver::gen(
                        &mut state_manager,
                        num_cycles_bits,
                        &uni,
                    );
                    let mut instance_refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> =
                        vec![&mut instance];

                    black_box(BatchedSumcheck::prove(
                        instance_refs,
                        &mut opening_accumulator,
                        &mut transcript,
                    ));
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_spartan_sumcheck);
criterion_main!(benches);
