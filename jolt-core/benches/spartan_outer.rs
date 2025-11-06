use ark_bn254::Fr;
use ark_ff::biginteger::S128;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, SamplingMode};
use jolt_core::zkvm::dag::stage::SumcheckStagesProver;
use jolt_core::{
    poly::{commitment::dory::DoryCommitmentScheme, opening_proof::ProverOpeningAccumulator},
    subprotocols::{
        sumcheck::BatchedSumcheck,
        sumcheck_prover::SumcheckInstanceProver,
        univariate_skip::{prove_uniskip_round, UniSkipState},
    },
    transcripts::{KeccakTranscript, Transcript},
    utils::math::Math,
    zkvm::{
        dag::state_manager::StateManager,
        r1cs::{
            constraints::{R1CSConstraint, R1CS_CONSTRAINTS},
            inputs::{JoltR1CSInputs, R1CSCycleInputs, ALL_R1CS_INPUTS},
        },
        spartan::{
            outer::OuterUniSkipInstanceProver, outer_baseline::OuterBaselineSumcheckProver,
            outer_round_batched::OuterRoundBatchedSumcheckProver,
            outer_streaming::OuterRemainingStreamingSumcheckProver, SpartanDagProver,
        },
        witness::AllCommittedPolynomials,
        Jolt, JoltProverPreprocessing, JoltRV64IMAC, JoltSharedPreprocessing,
    },
};
use tracer::instruction::Cycle;

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

    let preprocessing: JoltProverPreprocessing<F, PCS> = JoltRV64IMAC::prover_preprocess(
        bytecode,
        io_device.memory_layout.clone(),
        memory_init,
        padded_trace_length,
    );

    // This is needed to initialize the global list of committed polynomials, required for witness generation.
    let ram_d =
        jolt_core::zkvm::witness::compute_d_parameter(io_device.memory_layout.memory_end as usize);
    let bytecode_d =
        jolt_core::zkvm::witness::compute_d_parameter(preprocessing.shared.bytecode.code_size);
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
    let preprocessing_leaked: &'static JoltProverPreprocessing<F, PCS> =
        Box::leak(Box::new(preprocessing));
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
    let opening_accumulator =
        jolt_core::poly::opening_proof::ProverOpeningAccumulator::<F>::new(opening_bits);
    // FS preamble must be applied before any challenges are derived
    state_manager.fiat_shamir_preamble(&mut transcript);

    (
        state_manager,
        padded_trace_length,
        opening_accumulator,
        transcript,
    )
}

fn build_flattened_polynomials<F: jolt_core::field::JoltField>(
    preprocess: &JoltSharedPreprocessing,
    trace: &Vec<Cycle>,
) -> Vec<jolt_core::poly::multilinear_polynomial::MultilinearPolynomial<F>> {
    let n = trace.len();
    let mut out: Vec<jolt_core::poly::multilinear_polynomial::MultilinearPolynomial<F>> =
        Vec::with_capacity(ALL_R1CS_INPUTS.len());
    for input in ALL_R1CS_INPUTS.iter() {
        match input {
            JoltR1CSInputs::LeftInstructionInput
            | JoltR1CSInputs::LeftLookupOperand
            | JoltR1CSInputs::LookupOutput
            | JoltR1CSInputs::PC
            | JoltR1CSInputs::UnexpandedPC
            | JoltR1CSInputs::RamAddress
            | JoltR1CSInputs::Rs1Value
            | JoltR1CSInputs::Rs2Value
            | JoltR1CSInputs::RdWriteValue
            | JoltR1CSInputs::RamReadValue
            | JoltR1CSInputs::RamWriteValue
            | JoltR1CSInputs::NextPC
            | JoltR1CSInputs::NextUnexpandedPC => {
                let mut v: Vec<u64> = Vec::with_capacity(n);
                for t in 0..n {
                    let row = R1CSCycleInputs::from_trace::<F>(preprocess, trace, t);
                    let val = match input {
                        JoltR1CSInputs::LeftInstructionInput => row.left_input,
                        JoltR1CSInputs::LeftLookupOperand => row.left_lookup,
                        JoltR1CSInputs::LookupOutput => row.lookup_output,
                        JoltR1CSInputs::PC => row.pc,
                        JoltR1CSInputs::UnexpandedPC => row.unexpanded_pc,
                        JoltR1CSInputs::RamAddress => row.ram_addr,
                        JoltR1CSInputs::Rs1Value => row.rs1_read_value,
                        JoltR1CSInputs::Rs2Value => row.rs2_read_value,
                        JoltR1CSInputs::RdWriteValue => row.rd_write_value,
                        JoltR1CSInputs::RamReadValue => row.ram_read_value,
                        JoltR1CSInputs::RamWriteValue => row.ram_write_value,
                        JoltR1CSInputs::NextPC => row.next_pc,
                        JoltR1CSInputs::NextUnexpandedPC => row.next_unexpanded_pc,
                        _ => 0,
                    };
                    v.push(val);
                }
                out.push(v.into());
            }
            JoltR1CSInputs::RightInstructionInput | JoltR1CSInputs::Imm => {
                let mut v: Vec<i128> = Vec::with_capacity(n);
                for t in 0..n {
                    let row = R1CSCycleInputs::from_trace::<F>(preprocess, trace, t);
                    let val = match input {
                        JoltR1CSInputs::RightInstructionInput => row.right_input.to_i128(),
                        JoltR1CSInputs::Imm => row.imm.to_i128(),
                        _ => 0,
                    };
                    v.push(val);
                }
                out.push(v.into());
            }
            JoltR1CSInputs::Product => {
                let mut v: Vec<S128> = Vec::with_capacity(n);
                for t in 0..n {
                    let row = R1CSCycleInputs::from_trace::<F>(preprocess, trace, t);
                    v.push(row.product);
                }
                out.push(v.into());
            }
            JoltR1CSInputs::RightLookupOperand => {
                let mut v: Vec<u128> = Vec::with_capacity(n);
                for t in 0..n {
                    let row = R1CSCycleInputs::from_trace::<F>(preprocess, trace, t);
                    v.push(row.right_lookup);
                }
                out.push(v.into());
            }
            JoltR1CSInputs::WriteLookupOutputToRD
            | JoltR1CSInputs::WritePCtoRD
            | JoltR1CSInputs::ShouldBranch
            | JoltR1CSInputs::NextIsVirtual
            | JoltR1CSInputs::NextIsFirstInSequence
            | JoltR1CSInputs::ShouldJump => {
                let mut v: Vec<bool> = Vec::with_capacity(n);
                for t in 0..n {
                    let row = R1CSCycleInputs::from_trace::<F>(preprocess, trace, t);
                    let val = match input {
                        JoltR1CSInputs::WriteLookupOutputToRD => row.write_lookup_output_to_rd_addr,
                        JoltR1CSInputs::WritePCtoRD => row.write_pc_to_rd_addr,
                        JoltR1CSInputs::ShouldBranch => row.should_branch,
                        JoltR1CSInputs::NextIsVirtual => row.next_is_virtual,
                        JoltR1CSInputs::NextIsFirstInSequence => row.next_is_first_in_sequence,
                        JoltR1CSInputs::ShouldJump => row.should_jump,
                        _ => false,
                    };
                    v.push(val);
                }
                out.push(v.into());
            }
            JoltR1CSInputs::OpFlags(flag) => {
                let mut v: Vec<bool> = Vec::with_capacity(n);
                for t in 0..n {
                    let row = R1CSCycleInputs::from_trace::<F>(preprocess, trace, t);
                    v.push(row.flags[*flag as usize]);
                }
                out.push(v.into());
            }
        }
    }
    out
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
                |(
                    mut state_manager,
                    padded_trace_length,
                    mut opening_accumulator,
                    mut transcript,
                )| {
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
                    debug_assert_eq!(instances.len(), 1);
                    let mut only = instances
                        .pop()
                        .expect("expected one outer remaining instance");
                    let instance_refs: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>> =
                        vec![&mut *only];
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
                |(
                    mut state_manager,
                    _padded_trace_length,
                    mut opening_accumulator,
                    mut transcript,
                )| {
                    // Uni-skip first round (manual, to get UniSkipState)
                    let num_rows_bits = state_manager.get_trace_len().log_2();
                    let tau = transcript.challenge_vector_optimized::<F>(num_rows_bits);
                    let mut uniskip_instance =
                        OuterUniSkipInstanceProver::gen(&mut state_manager, &tau);
                    let (_first_round_proof, r0, claim_after_first) =
                        prove_uniskip_round(&mut uniskip_instance, &mut transcript);
                    let uni = UniSkipState {
                        claim_after_first,
                        r0,
                        tau,
                    };

                    // Streaming outer-remaining instance
                    let num_cycles_bits = state_manager.get_trace_len().log_2();
                    let mut instance = OuterRemainingStreamingSumcheckProver::gen(
                        &mut state_manager,
                        num_cycles_bits,
                        &uni,
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
                |(
                    mut state_manager,
                    _padded_trace_length,
                    mut opening_accumulator,
                    mut transcript,
                )| {
                    let mut instance = OuterRoundBatchedSumcheckProver::<F>::gen::<
                        PCS,
                        ProofTranscript,
                    >(&mut state_manager, &mut transcript);
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
                |(
                    state_manager,
                    _padded_trace_length,
                    mut opening_accumulator,
                    mut transcript,
                )| {
                    // Baseline (no uni-skip): build flattened polynomials and instantiate baseline prover
                    let (preprocessing, _lazy_trace, trace, _program_io, _final_mem) =
                        state_manager.get_prover_data();
                    let flattened = build_flattened_polynomials::<F>(&preprocessing.shared, trace);
                    let padded_num_constraints = R1CS_CONSTRAINTS.len().next_power_of_two();
                    let constraints_vec: Vec<R1CSConstraint> =
                        R1CS_CONSTRAINTS.iter().map(|c| c.cons).collect();
                    let mut instance = OuterBaselineSumcheckProver::<F>::gen_from_polys(
                        &constraints_vec,
                        &flattened,
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
