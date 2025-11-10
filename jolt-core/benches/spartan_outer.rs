use ark_bn254::Fr;
use ark_ff::biginteger::S128;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, SamplingMode};
use jolt_core::zkvm::dag::stage::SumcheckStagesProver;
use jolt_core::zkvm::instruction::CircuitFlags;
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
        bytecode::BytecodePreprocessing,
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
        Jolt, JoltProverPreprocessing, JoltRV64IMAC,
    },
};
use strum::IntoEnumIterator;
use tracer::instruction::Cycle;

// Ensure SHA2 inline library is linked so its #[ctor] auto-registers builders
#[cfg(feature = "host")]
use jolt_inlines_sha2 as _;

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
    let bytecode_d = jolt_core::zkvm::witness::compute_d_parameter(preprocessing.bytecode.code_size);
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
    let opening_bits = padded_trace_length.log_2();
    let opening_accumulator =
        jolt_core::poly::opening_proof::ProverOpeningAccumulator::<F>::new(opening_bits);
    // FS preamble must be applied before any challenges are derived
    jolt_core::zkvm::dag::state_manager::fiat_shamir_preamble(
        &state_manager.program_io,
        state_manager.ram_K,
        padded_trace_length,
        &mut transcript,
    );

    (
        state_manager,
        padded_trace_length,
        opening_accumulator,
        transcript,
    )
}

fn build_flattened_polynomials<F: jolt_core::field::JoltField>(
    preprocess: &BytecodePreprocessing,
    trace: &Vec<Cycle>,
) -> Vec<jolt_core::poly::multilinear_polynomial::MultilinearPolynomial<F>> {
    let n = trace.len();

    // Pre-allocate per-input buffers
    let mut left_instruction_input: Vec<u64> = Vec::with_capacity(n);
    let mut right_instruction_input: Vec<i128> = Vec::with_capacity(n);
    let mut product: Vec<S128> = Vec::with_capacity(n);

    let mut left_lookup_operand: Vec<u64> = Vec::with_capacity(n);
    let mut right_lookup_operand: Vec<u128> = Vec::with_capacity(n);
    let mut lookup_output: Vec<u64> = Vec::with_capacity(n);

    let mut pc: Vec<u64> = Vec::with_capacity(n);
    let mut unexpanded_pc: Vec<u64> = Vec::with_capacity(n);
    let mut next_pc: Vec<u64> = Vec::with_capacity(n);
    let mut next_unexpanded_pc: Vec<u64> = Vec::with_capacity(n);

    let mut imm: Vec<i128> = Vec::with_capacity(n);

    let mut ram_addr: Vec<u64> = Vec::with_capacity(n);
    let mut ram_read_value: Vec<u64> = Vec::with_capacity(n);
    let mut ram_write_value: Vec<u64> = Vec::with_capacity(n);

    let mut rs1_read_value: Vec<u64> = Vec::with_capacity(n);
    let mut rs2_read_value: Vec<u64> = Vec::with_capacity(n);
    let mut rd_write_value: Vec<u64> = Vec::with_capacity(n);

    let mut write_lookup_output_to_rd_addr: Vec<bool> = Vec::with_capacity(n);
    let mut write_pc_to_rd_addr: Vec<bool> = Vec::with_capacity(n);
    let mut should_branch: Vec<bool> = Vec::with_capacity(n);
    let mut should_jump: Vec<bool> = Vec::with_capacity(n);
    let mut next_is_virtual: Vec<bool> = Vec::with_capacity(n);
    let mut next_is_first_in_sequence: Vec<bool> = Vec::with_capacity(n);

    // Per-flag buffers
    let mut opflag_vecs: [Vec<bool>; jolt_core::zkvm::instruction::NUM_CIRCUIT_FLAGS] =
        std::array::from_fn(|_| Vec::with_capacity(n));

    // Single pass over the trace
    for t in 0..n {
        let row = R1CSCycleInputs::from_trace::<F>(preprocess, trace, t);

        // Instruction inputs and product
        left_instruction_input.push(row.left_input);
        right_instruction_input.push(row.right_input.to_i128());
        product.push(row.product);

        // Lookup operands and output
        left_lookup_operand.push(row.left_lookup);
        right_lookup_operand.push(row.right_lookup);
        lookup_output.push(row.lookup_output);

        // Registers
        rs1_read_value.push(row.rs1_read_value);
        rs2_read_value.push(row.rs2_read_value);
        rd_write_value.push(row.rd_write_value);

        // RAM
        ram_addr.push(row.ram_addr);
        ram_read_value.push(row.ram_read_value);
        ram_write_value.push(row.ram_write_value);

        // PCs
        pc.push(row.pc);
        next_pc.push(row.next_pc);
        unexpanded_pc.push(row.unexpanded_pc);
        next_unexpanded_pc.push(row.next_unexpanded_pc);

        // Immediate
        imm.push(row.imm.to_i128());

        // Derived booleans
        write_lookup_output_to_rd_addr.push(row.write_lookup_output_to_rd_addr);
        write_pc_to_rd_addr.push(row.write_pc_to_rd_addr);
        should_branch.push(row.should_branch);
        should_jump.push(row.should_jump);
        next_is_virtual.push(row.next_is_virtual);
        next_is_first_in_sequence.push(row.next_is_first_in_sequence);

        // Op flags
        for flag in CircuitFlags::iter() {
            let idx = flag as usize;
            opflag_vecs[idx].push(row.flags[flag]);
        }
    }

    // Assemble output in ALL_R1CS_INPUTS canonical order
    let mut out: Vec<jolt_core::poly::multilinear_polynomial::MultilinearPolynomial<F>> =
        Vec::with_capacity(ALL_R1CS_INPUTS.len());

    for input in ALL_R1CS_INPUTS.iter() {
        match input {
            JoltR1CSInputs::LeftInstructionInput => {
                out.push(std::mem::take(&mut left_instruction_input).into())
            }
            JoltR1CSInputs::RightInstructionInput => {
                out.push(std::mem::take(&mut right_instruction_input).into())
            }
            JoltR1CSInputs::Product => out.push(std::mem::take(&mut product).into()),
            JoltR1CSInputs::WriteLookupOutputToRD => {
                out.push(std::mem::take(&mut write_lookup_output_to_rd_addr).into())
            }
            JoltR1CSInputs::WritePCtoRD => {
                out.push(std::mem::take(&mut write_pc_to_rd_addr).into())
            }
            JoltR1CSInputs::ShouldBranch => out.push(std::mem::take(&mut should_branch).into()),
            JoltR1CSInputs::PC => out.push(std::mem::take(&mut pc).into()),
            JoltR1CSInputs::UnexpandedPC => out.push(std::mem::take(&mut unexpanded_pc).into()),
            JoltR1CSInputs::Imm => out.push(std::mem::take(&mut imm).into()),
            JoltR1CSInputs::RamAddress => out.push(std::mem::take(&mut ram_addr).into()),
            JoltR1CSInputs::Rs1Value => out.push(std::mem::take(&mut rs1_read_value).into()),
            JoltR1CSInputs::Rs2Value => out.push(std::mem::take(&mut rs2_read_value).into()),
            JoltR1CSInputs::RdWriteValue => out.push(std::mem::take(&mut rd_write_value).into()),
            JoltR1CSInputs::RamReadValue => out.push(std::mem::take(&mut ram_read_value).into()),
            JoltR1CSInputs::RamWriteValue => out.push(std::mem::take(&mut ram_write_value).into()),
            JoltR1CSInputs::LeftLookupOperand => {
                out.push(std::mem::take(&mut left_lookup_operand).into())
            }
            JoltR1CSInputs::RightLookupOperand => {
                out.push(std::mem::take(&mut right_lookup_operand).into())
            }
            JoltR1CSInputs::NextUnexpandedPC => {
                out.push(std::mem::take(&mut next_unexpanded_pc).into())
            }
            JoltR1CSInputs::NextPC => out.push(std::mem::take(&mut next_pc).into()),
            JoltR1CSInputs::NextIsVirtual => out.push(std::mem::take(&mut next_is_virtual).into()),
            JoltR1CSInputs::NextIsFirstInSequence => {
                out.push(std::mem::take(&mut next_is_first_in_sequence).into())
            }
            JoltR1CSInputs::LookupOutput => out.push(std::mem::take(&mut lookup_output).into()),
            JoltR1CSInputs::ShouldJump => out.push(std::mem::take(&mut should_jump).into()),
            JoltR1CSInputs::OpFlags(flag) => {
                let idx = *flag as usize;
                out.push(std::mem::take(&mut opflag_vecs[idx]).into());
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
                    padded_trace_length,
                    mut opening_accumulator,
                    mut transcript,
                )| {
                    // Uni-skip first round (manual, to get UniSkipState)
                    let num_rows_bits = padded_trace_length.log_2();
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
                    let num_cycles_bits = padded_trace_length.log_2();
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
                |(state_manager, _padded_trace_length, mut opening_accumulator, mut transcript)| {
                    // Baseline (no uni-skip): build flattened polynomials and instantiate baseline prover
                    let (preprocessing, _lazy_trace, trace, _program_io, _final_mem) =
                        state_manager.get_prover_data();
                    let flattened = build_flattened_polynomials::<F>(&preprocessing.bytecode, trace);
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
