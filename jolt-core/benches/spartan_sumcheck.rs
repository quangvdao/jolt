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
            builder::CombinedUniformBuilder,
            constraints::{JoltRV32IMConstraints, R1CSConstraints},
            inputs::ALL_R1CS_INPUTS,
            key::UniformSpartanKey,
        },
        witness::AllCommittedPolynomials,
        Jolt, JoltProverPreprocessing, JoltRV32IM,
    },
};
use rayon::prelude::*;

#[cfg(feature = "host")]
extern crate sha2_inline;

type F = Fr;
type PCS = DoryCommitmentScheme;
type ProofTranscript = KeccakTranscript;

fn setup_for_spartan(
    program_name: &str,
    num_iterations: u32,
) -> (
    Vec<MultilinearPolynomial<F>>,
    Vec<jolt_core::zkvm::r1cs::builder::Constraint>,
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

    // This is needed to initialize the global list of committed polynomials, required for witness generation.
    let ram_d = jolt_core::zkvm::witness::compute_d_parameter(io_device.memory_layout.memory_end as usize);
    let bytecode_d =
        jolt_core::zkvm::witness::compute_d_parameter(preprocessing.shared.bytecode.code_size);
    let _all_committed_polys_handle = AllCommittedPolynomials::initialize(ram_d, bytecode_d);

    let flattened_polys: Vec<MultilinearPolynomial<F>> = ALL_R1CS_INPUTS
        .par_iter()
        .map(|input| input.generate_witness(&trace, &preprocessing))
        .collect();

    let uniform_builder: CombinedUniformBuilder<F> =
        JoltRV32IMConstraints::construct_constraints(padded_trace_length);
    let uniform_constraints = uniform_builder.uniform_builder.get_constraints();
    let uniform_key = UniformSpartanKey::from_builder(&uniform_builder);

    let num_rounds = uniform_key.num_steps.log_2() + uniform_key.num_cons_total.log_2();
    let padded_num_constraints = uniform_key.num_cons_total;
    let transcript = ProofTranscript::new(b"Jolt transcript");

    (
        flattened_polys,
        uniform_constraints,
        num_rounds,
        padded_num_constraints,
        transcript,
    )
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

        group.bench_function(&format!("gruen/{}", bench_name), |b| {
            b.iter_batched(
                || setup_for_spartan("sha2-chain-guest", num_iterations),
                |(
                    flattened_polys,
                    uniform_constraints,
                    num_rounds,
                    padded_num_constraints,
                    mut transcript,
                )| {
                    let tau: Vec<F> = (0..num_rounds)
                        .map(|_| F::from(rand::random::<u64>()))
                        .collect();

                    black_box(
                        SumcheckInstanceProof::prove_spartan_with_gruen(
                            num_rounds,
                            padded_num_constraints,
                            &uniform_constraints,
                            &flattened_polys,
                            &tau,
                            &mut transcript,
                        ),
                    );
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function(&format!("legacy/{}", bench_name), |b| {
            b.iter_batched(
                || setup_for_spartan("sha2-chain-guest", num_iterations),
                |(
                    flattened_polys,
                    uniform_constraints,
                    num_rounds,
                    padded_num_constraints,
                    mut transcript,
                )| {
                    let tau: Vec<F> = (0..num_rounds)
                        .map(|_| F::from(rand::random::<u64>()))
                        .collect();

                    black_box(
                        SumcheckInstanceProof::prove_spartan_with_legacy(
                            num_rounds,
                            padded_num_constraints,
                            &uniform_constraints,
                            &flattened_polys,
                            &tau,
                            &mut transcript,
                        ),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_spartan_sumcheck);
criterion_main!(benches);
