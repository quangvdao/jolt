use ark_bn254::{Fr};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, SamplingMode};
use jolt_core::{
    poly::{
        commitment::dory::DoryCommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::{KeccakTranscript, Transcript},
    utils::{math::Math, small_value::NUM_SVO_ROUNDS},
    zkvm::{
        r1cs::{
            constraints::{JoltRV32IMConstraints, R1CSConstraints},
            key::UniformSpartanKey,
        },
        witness::{AllCommittedPolynomials, CommittedPolynomial},
        Jolt, JoltProverPreprocessing, JoltRV32IM,
    },
};

type F = Fr;
type PCS = DoryCommitmentScheme;
type ProofTranscript = KeccakTranscript;

fn setup_for_spartan(
    program_name: &str,
    num_iterations: u32,
) -> (
    Vec<MultilinearPolynomial<F>>,
    Vec<jolt_core::zkvm::r1cs::builder::Constraint>,
    UniformSpartanKey<F>,
    usize,
    usize,
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
    let (mut trace, _, io_device) = program.trace(&inputs);

    let padded_trace_length = (trace.len() + 1).next_power_of_two();
    trace.resize(
        padded_trace_length,
        tracer::instruction::RV32IMCycle::NoOp,
    );

    let preprocessing: JoltProverPreprocessing<F, PCS> = JoltRV32IM::prover_preprocess(
        bytecode,
        io_device.memory_layout.clone(),
        memory_init,
        padded_trace_length,
    );

    let ram_d = jolt_core::zkvm::witness::compute_d_parameter(io_device.memory_layout.memory_end as usize);
    let bytecode_d =
        jolt_core::zkvm::witness::compute_d_parameter(preprocessing.shared.bytecode.code_size);
    let _all_committed_polys_handle = AllCommittedPolynomials::initialize(ram_d, bytecode_d);

    let mut flattened_polys: Vec<MultilinearPolynomial<F>> = Vec::new();
    let mut witness_polys = CommittedPolynomial::generate_witness_batch(
        &AllCommittedPolynomials::iter().cloned().collect::<Vec<_>>(),
        &preprocessing,
        &trace,
    );

    for poly in AllCommittedPolynomials::iter() {
        flattened_polys.push(witness_polys.remove(poly).unwrap());
    }

    let uniform_builder =
        JoltRV32IMConstraints::construct_constraints(padded_trace_length);
    let uniform_constraints = uniform_builder.uniform_builder.get_constraints();
    let uniform_key = UniformSpartanKey::from_builder(&uniform_builder);
    let num_rounds_x = uniform_key.num_steps.log_2();
    let padded_num_constraints = uniform_key.num_cons_total;
    let transcript = ProofTranscript::new(b"Jolt transcript");

    (
        flattened_polys,
        uniform_constraints,
        uniform_key,
        num_rounds_x,
        padded_num_constraints,
        transcript,
    )
}

fn bench_spartan_svo(c: &mut Criterion) {
    let mut group = c.benchmark_group("spartan_svo");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    group.bench_function("sha2-chain-100", |b| {
        b.iter_batched(
            || setup_for_spartan("sha2-chain-guest", 100),
            |(
                flattened_polys,
                uniform_constraints,
                _uniform_key,
                num_rounds_x,
                padded_num_constraints,
                mut transcript,
            )| {
                let tau: Vec<F> = (0..num_rounds_x)
                    .map(|_| F::from(rand::random::<u64>()))
                    .collect();

                let (proof, r, final_evals) = black_box(
                    SumcheckInstanceProof::prove_spartan_small_value::<NUM_SVO_ROUNDS>(
                        num_rounds_x,
                        padded_num_constraints,
                        &uniform_constraints,
                        &flattened_polys,
                        &tau,
                        &mut transcript,
                    ),
                );

                (proof, r, final_evals)
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_spartan_gruen(c: &mut Criterion) {
    let mut group = c.benchmark_group("spartan_gruen");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    group.bench_function("sha2-chain-100", |b| {
        b.iter_batched(
            || setup_for_spartan("sha2-chain-guest", 100),
            |(
                flattened_polys,
                uniform_constraints,
                _uniform_key,
                num_rounds_x,
                padded_num_constraints,
                mut transcript,
            )| {
                let tau: Vec<F> = (0..num_rounds_x)
                    .map(|_| F::from(rand::random::<u64>()))
                    .collect();

                let (proof, r, final_evals) = black_box(
                    SumcheckInstanceProof::prove_spartan_with_gruen(
                        num_rounds_x,
                        padded_num_constraints,
                        &uniform_constraints,
                        &flattened_polys,
                        &tau,
                        &mut transcript,
                    ),
                );

                (proof, r, final_evals)
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_spartan_svo, bench_spartan_gruen);
criterion_main!(benches);
