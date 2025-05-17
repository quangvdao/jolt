// TODO: import the logic needed to build program trace, preprocess it into jolt witness
// then run Spartan first sumcheck on it

use ark_bn254::{Bn254, Fr};
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, Bencher, Criterion, SamplingMode,
};
use jolt_core::{
    host,
    jolt::vm::{
        rv32i_vm::{RV32IJoltVM, C},
        Jolt, JoltPolynomials, JoltProverPreprocessing, JoltTraceStep,
    },
    poly::{
        commitment::hyperkzg::HyperKZG,
        multilinear_polynomial::MultilinearPolynomial,
        split_eq_poly::{NewSplitEqPolynomial, SplitEqPolynomial},
    },
    r1cs::{
        builder::CombinedUniformBuilder,
        inputs::{ConstraintInput, JoltR1CSInputs},
        key::UniformSpartanKey,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::transcript::{KeccakTranscript, Transcript},
};
use postcard;

type F = Fr;
type PCS = HyperKZG<Bn254, KeccakTranscript>;
type ProofTranscript = KeccakTranscript;

fn setup_for_spartan(
    num_iterations: u32,
) -> (
    JoltPolynomials<F>,
    ProofTranscript,
    CombinedUniformBuilder<C, F, JoltR1CSInputs>,
    UniformSpartanKey<C, JoltR1CSInputs, F>,
) {
    let mut program = host::Program::new("sha2-chain-guest");

    // Prepare input for sha2_chain guest: (input_data: [u8; 32], iterations: u32)
    let initial_hash_data = [0u8; 32]; // Example initial data
    let guest_input_tuple = (initial_hash_data, num_iterations);
    let inputs = postcard::to_stdvec(&guest_input_tuple).unwrap();

    let (io_device, mut trace) = program.trace(&inputs);
    let (bytecode, memory_init) = program.decode();

    let max_trace_len = 1 << 18; // Adjust if needed for the specific program

    let mut preprocessing: JoltProverPreprocessing<C, F, PCS, ProofTranscript> =
        RV32IJoltVM::prover_preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init.clone(),
            max_trace_len,
            max_trace_len,
            max_trace_len,
        );

    let trace_len = trace.len();
    let padded_trace_length = trace_len.next_power_of_two();

    RV32IJoltVM::initialize_lookup_tables(&mut preprocessing);

    JoltTraceStep::pad(&mut trace);

    let mut transcript = ProofTranscript::new(b"Jolt transcript");

    // Comment this out for now since there are type annotation errors
    // In any case, since we are benchmarking only, we do not need this for security
    // RV32IJoltVM::fiat_shamir_preamble(
    //     &mut transcript,
    //     &io_device,
    //     &io_device.memory_layout,
    //     trace_len,
    // );

    let (r1cs_builder, spartan_key, jolt_polynomials) = RV32IJoltVM::construct_data_for_spartan(
        io_device,
        padded_trace_length,
        trace,
        &preprocessing,
    );

    transcript.append_scalar(&spartan_key.vk_digest);

    (jolt_polynomials, transcript, r1cs_builder, spartan_key)
}

fn bench_spartan_sumchecks_in_file(c: &mut Criterion) {
    // Define a range or list of iteration counts you want to benchmark
    // 64, 128, 256, 512,
    let iteration_counts = vec![2048]; // Example values

    for &num_iters in iteration_counts.iter() {
        println!(
            "Running one-time setup for Spartan sumcheck benchmarks with {} iterations...",
            num_iters
        );
        // Pass num_iters to setup
        let (jolt_polynomials, mut transcript, r1cs_builder, spartan_key) =
            setup_for_spartan(num_iters);
        println!("Setup complete for {} iterations.", num_iters);

        let flattened_polys: Vec<&MultilinearPolynomial<F>> = JoltR1CSInputs::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(&jolt_polynomials))
            .collect();

        let num_rounds_x = spartan_key.num_rows_bits();
        println!("num_rounds_x: {}", num_rounds_x);

        println!("trace length: {:?}", flattened_polys[0].len());

        let padded_rows_per_step = r1cs_builder.padded_rows_per_step();
        let uniform_constraints = r1cs_builder.uniform_builder.constraints.clone();
        let cross_step_constraints = r1cs_builder.offset_equality_constraints.clone();

        /* Sumcheck 1: Outer sumcheck */

        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();

        // Use a group name that reflects the parameterization
        let mut group = c.benchmark_group(format!(
            "SpartanFirstSumcheckStandalone_iters_{}",
            num_iters
        ));

        group.sample_size(10);
        group.sampling_mode(SamplingMode::Flat);

        group.bench_function(
            "Original (SpartanInterleaved + SplitEq)",
            |b: &mut Bencher| {
                b.iter_batched(
                    || {
                        let new_transcript = transcript.clone();
                        return new_transcript;
                    },
                    |mut transcript| {
                        let mut eq_poly = SplitEqPolynomial::new(&tau);

                        let mut az_bz_cz_poly =
                            black_box(r1cs_builder.compute_spartan_Az_Bz_Cz(&flattened_polys));
                        black_box(SumcheckInstanceProof::prove_spartan_cubic(
                            num_rounds_x,
                            black_box(&mut eq_poly),
                            black_box(&mut az_bz_cz_poly),
                            black_box(&mut transcript),
                        ));
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_function(
            "Gruen (SpartanInterleaved + NewSplitEq)",
            |b: &mut Bencher| {
                b.iter_batched(
                    || {
                        let new_transcript = transcript.clone();
                        return new_transcript;
                    },
                    |mut transcript| {
                        let mut eq_poly = NewSplitEqPolynomial::new(&tau);

                        let mut az_bz_cz_poly =
                            black_box(r1cs_builder.compute_spartan_Az_Bz_Cz(&flattened_polys));
                        black_box(SumcheckInstanceProof::prove_spartan_cubic_with_gruen(
                            num_rounds_x,
                            black_box(&mut eq_poly),
                            black_box(&mut az_bz_cz_poly),
                            black_box(&mut transcript),
                        ));
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_function(
            "Gruen + SVO (NewSpartanInterleaved + NewSplitEq)",
            |b: &mut Bencher| {
                b.iter_batched(
                    || {
                        let new_transcript = transcript.clone();
                        return new_transcript;
                    },
                    |mut transcript| {
                        black_box(SumcheckInstanceProof::prove_spartan_small_value::<3>(
                            num_rounds_x,
                            padded_rows_per_step,
                            black_box(&uniform_constraints),
                            black_box(&cross_step_constraints),
                            black_box(&flattened_polys),
                            black_box(&tau),
                            black_box(&mut transcript),
                        ));
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.finish();
    }
}
criterion_group!(spartan_sumcheck_benches, bench_spartan_sumchecks_in_file);
criterion_main!(spartan_sumcheck_benches);
