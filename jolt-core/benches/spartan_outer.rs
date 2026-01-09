use std::sync::Arc;

use ark_bn254::Fr;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, SamplingMode};
use jolt_core::{
    poly::opening_proof::ProverOpeningAccumulator,
    subprotocols::{
        streaming_schedule::{HalfSplitSchedule, LinearOnlySchedule},
        sumcheck::BatchedSumcheck,
        sumcheck_prover::SumcheckInstanceProver,
        univariate_skip::prove_uniskip_round,
    },
    transcripts::{KeccakTranscript, Transcript},
    utils::math::Math,
    zkvm::{
        bytecode::BytecodePreprocessing,
        r1cs::constraints::{R1CSConstraint, R1CS_CONSTRAINTS},
        r1cs::key::UniformSpartanKey,
        spartan::{
            outer_baseline::OuterBaselineSumcheckProver as OuterBaselineStreamingSumcheckProver,
            outer_naive::OuterNaiveSumcheckProver,
            outer_round_batched::OuterRoundBatchedSumcheckProver,
            outer::{OuterRemainingStreamingSumcheck, OuterSharedState, OuterUniSkipParams, OuterUniSkipProver},
            outer_uni_skip_linear::{OuterRemainingSumcheckProverNonStreaming, OuterUniSkipInstanceProver},
        },
    },
};
use tracer::instruction::Cycle;

// Ensure SHA2 inline library is linked so its #[ctor] auto-registers builders
#[cfg(feature = "host")]
use jolt_inlines_sha2 as _;
#[cfg(feature = "host")]
use jolt_core::host;

type F = Fr;
type ProofTranscript = KeccakTranscript;

fn setup_for_spartan(
    program_name: &str,
    num_iterations: u32,
) -> (
    Arc<Vec<Cycle>>,
    BytecodePreprocessing,
    usize,
    ProverOpeningAccumulator<F>,
    ProofTranscript,
) {
    let mut program = host::Program::new(program_name);

    let inputs = if program_name.contains("chain") {
        let initial_hash_data = [7u8; 32];
        let guest_input_tuple = (initial_hash_data, num_iterations);
        postcard::to_stdvec(&guest_input_tuple).unwrap()
    } else {
        let simple_input = num_iterations as u64;
        postcard::to_stdvec(&simple_input).unwrap()
    };

    let (bytecode, _memory_init, _) = program.decode();
    let (_lazy_trace, mut trace, _final_memory_state, mut io_device) =
        program.trace(&inputs, &[], &[]);

    let padded_trace_length = (trace.len() + 1).next_power_of_two();
    trace.resize(padded_trace_length, Cycle::NoOp);

    let bytecode_pp = BytecodePreprocessing::preprocess(bytecode);

    // Truncate trailing zeros on device outputs (defensive)
    io_device.outputs.truncate(
        io_device
            .outputs
            .iter()
            .rposition(|&b| b != 0)
            .map_or(0, |pos| pos + 1),
    );

    let transcript = ProofTranscript::new(b"Jolt transcript");
    let opening_bits = padded_trace_length.log_2();
    let opening_accumulator = ProverOpeningAccumulator::<F>::new(opening_bits);

    (Arc::new(trace), bytecode_pp, padded_trace_length, opening_accumulator, transcript)
}

fn bench_spartan_sumcheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spartan Sumcheck");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    // powers of 2
    let num_iters = [8, 16, 32, 64, 128, 256, 512, 1024, 2048];

    for num_iterations in num_iters {
        let bench_name = format!("sha2-chain-{}", num_iterations);

        group.bench_function(&format!("outer-current/{}", bench_name), |b| {
            b.iter_batched(
                || setup_for_spartan("sha2-chain-guest", num_iterations),
                |(trace, bytecode_pp, padded_trace_length, mut opening_accumulator, mut transcript)| {
                    // Stage 1 (Outer): UniSkip first round + remaining rounds in linear-only mode
                    let key = UniformSpartanKey::<F>::new(padded_trace_length);
                    let uni_skip_params = OuterUniSkipParams::<F>::new(&key, &mut transcript);
                    let mut uni_skip = OuterUniSkipProver::<F>::initialize(
                        uni_skip_params.clone(),
                        trace.as_ref(),
                        &bytecode_pp,
                    );
                    let _ = prove_uniskip_round(&mut uni_skip, &mut opening_accumulator, &mut transcript);

                    // Remaining outer rounds: linear-only schedule (no streaming windows).
                    let schedule = LinearOnlySchedule::new(uni_skip_params.tau.len() - 1);
                    let shared = OuterSharedState::<F>::new(
                        Arc::clone(&trace),
                        &bytecode_pp,
                        &uni_skip_params,
                        &opening_accumulator,
                    );
                    let mut instance: OuterRemainingStreamingSumcheck<F, LinearOnlySchedule> =
                        OuterRemainingStreamingSumcheck::new(shared, schedule);
                    let instance_refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> =
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

        group.bench_function(&format!("outer-streaming/{}", bench_name), |b| {
            b.iter_batched(
                || setup_for_spartan("sha2-chain-guest", num_iterations),
                |(trace, bytecode_pp, padded_trace_length, mut opening_accumulator, mut transcript)| {
                    // Uni-skip first round
                    let key = UniformSpartanKey::<F>::new(padded_trace_length);
                    let uni_skip_params = OuterUniSkipParams::<F>::new(&key, &mut transcript);
                    let mut uni_skip = OuterUniSkipProver::<F>::initialize(
                        uni_skip_params.clone(),
                        trace.as_ref(),
                        &bytecode_pp,
                    );
                    let _ =
                        prove_uniskip_round(&mut uni_skip, &mut opening_accumulator, &mut transcript);

                    // Remaining outer rounds: streaming schedule for degree-3 messages.
                    let num_rounds = uni_skip_params.tau.len() - 1;
                    let schedule = HalfSplitSchedule::new(num_rounds, 3);
                    let shared = OuterSharedState::<F>::new(
                        Arc::clone(&trace),
                        &bytecode_pp,
                        &uni_skip_params,
                        &opening_accumulator,
                    );
                    let mut instance: OuterRemainingStreamingSumcheck<F, HalfSplitSchedule> =
                        OuterRemainingStreamingSumcheck::new(shared, schedule);
                    let instance_refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> =
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

        group.bench_function(&format!("outer-uni-skip/{}", bench_name), |b| {
            b.iter_batched(
                || setup_for_spartan("sha2-chain-guest", num_iterations),
                |(trace, bytecode_pp, padded_trace_length, mut opening_accumulator, mut transcript)| {
                    // Stage 1 (Outer): uni-skip first round + remaining rounds using the specialized
                    // streaming-first-then-linear prover (no arbitrary window schedule machinery).
                    let key = UniformSpartanKey::<F>::new(padded_trace_length);
                    let uni_skip_params = OuterUniSkipParams::<F>::new(&key, &mut transcript);

                    // Uni-skip first round (checkpoint implementation)
                    let mut uni_skip = OuterUniSkipInstanceProver::<F>::gen(
                        trace.as_ref(),
                        &bytecode_pp,
                        &uni_skip_params.tau,
                    );
                    let _ = prove_uniskip_round(&mut uni_skip, &mut opening_accumulator, &mut transcript);

                    // Remaining rounds (checkpoint implementation)
                    let mut instance = OuterRemainingSumcheckProverNonStreaming::<F>::gen(
                        Arc::clone(&trace),
                        &bytecode_pp,
                        uni_skip_params,
                        &opening_accumulator,
                    );
                    let instance_refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> =
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

        group.bench_function(&format!("outer-round-batched/{}", bench_name), |b| {
            b.iter_batched(
                || setup_for_spartan("sha2-chain-guest", num_iterations),
                |(trace, bytecode_pp, _padded_trace_length, mut opening_accumulator, mut transcript)| {
                    let mut instance = OuterRoundBatchedSumcheckProver::<F>::gen::<ProofTranscript>(
                        Arc::clone(&trace),
                        &bytecode_pp,
                        &mut transcript,
                    );
                    let instance_refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> =
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

        group.bench_function(&format!("outer-baseline/{}", bench_name), |b| {
            b.iter_batched(
                || setup_for_spartan("sha2-chain-guest", num_iterations),
                |(trace, bytecode_pp, _padded_trace_length, mut opening_accumulator, mut transcript)| {
                    // Baseline (no uni-skip): build dense Az/Bz over (cycle,constraint)
                    let padded_num_constraints = R1CS_CONSTRAINTS.len().next_power_of_two();
                    let constraints_vec: Vec<R1CSConstraint> =
                        R1CS_CONSTRAINTS.iter().map(|c| c.cons).collect();
                    let mut instance = OuterBaselineStreamingSumcheckProver::<F>::gen(
                        &bytecode_pp,
                        Arc::clone(&trace),
                        &constraints_vec,
                        padded_num_constraints,
                        &mut transcript,
                    );
                    let instance_refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> =
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

        group.bench_function(&format!("outer-naive/{}", bench_name), |b| {
            b.iter_batched(
                || setup_for_spartan("sha2-chain-guest", num_iterations),
                |(trace, bytecode_pp, _padded_trace_length, mut opening_accumulator, mut transcript)| {
                    let padded_num_constraints = R1CS_CONSTRAINTS.len().next_power_of_two();
                    let constraints_vec: Vec<R1CSConstraint> =
                        R1CS_CONSTRAINTS.iter().map(|c| c.cons).collect();
                    let mut instance = OuterNaiveSumcheckProver::<F>::gen(
                        &bytecode_pp,
                        Arc::clone(&trace),
                        &constraints_vec,
                        padded_num_constraints,
                        &mut transcript,
                    );
                    let instance_refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> =
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
