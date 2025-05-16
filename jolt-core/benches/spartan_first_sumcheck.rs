// TODO: import the logic needed to build program trace, preprocess it into jolt witness
// then run Spartan first sumcheck on it

use ark_bn254::{Bn254, Fr};
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, Bencher, Criterion,
};
use jolt_core::{
    field::JoltField,
    host,
    jolt::{
        vm::{
            rv32i_vm::{RV32IJoltVM, C, M, RV32I}, Jolt, JoltPolynomials, JoltProverPreprocessing, JoltTraceStep
        },
    },
    poly::{
        commitment::hyperkzg::HyperKZG,
        multilinear_polynomial::MultilinearPolynomial,
        split_eq_poly::{NewSplitEqPolynomial, SplitEqPolynomial},
    },
    r1cs::{
        builder::CombinedUniformBuilder, inputs::{ConstraintInput, JoltR1CSInputs}, key::UniformSpartanKey,
        constraints::JoltRV32IMConstraints
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::transcript::{KeccakTranscript, Transcript},
};
use postcard;

type F = Fr;
type PCS = HyperKZG<Bn254, KeccakTranscript>;
type ProofTranscript = KeccakTranscript;

fn setup_for_spartan() ->
    (JoltPolynomials<F>,
        ProofTranscript,
        CombinedUniformBuilder<C, F, JoltR1CSInputs>,
        UniformSpartanKey<C, JoltR1CSInputs, F>) {
    let mut program = host::Program::new("fibonacci-guest");
    let fib_arg = 9u32; // Example input for setup
    let inputs = postcard::to_stdvec(&fib_arg).unwrap();

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

    let (r1cs_builder, spartan_key, jolt_polynomials) = RV32IJoltVM::construct_data_for_spartan(io_device, padded_trace_length, trace, &preprocessing);

    transcript.append_scalar(&spartan_key.vk_digest);

    (jolt_polynomials, transcript, r1cs_builder, spartan_key)
}

fn bench_spartan_sumchecks_in_file(c: &mut Criterion) {
    println!("Running one-time setup for Spartan sumcheck benchmarks...");
    let (jolt_polynomials, mut transcript, r1cs_builder, spartan_key) = setup_for_spartan();
    println!("Setup complete.");

    let flattened_polys: Vec<&MultilinearPolynomial<F>> = JoltR1CSInputs::flatten::<C>()
    .iter()
    .map(|var| var.get_ref(&jolt_polynomials))
    .collect();

    let num_rounds_x = spartan_key.num_rows_bits();

    /* Sumcheck 1: Outer sumcheck */

    let tau = (0..num_rounds_x)
        .map(|_i| transcript.challenge_scalar())
        .collect::<Vec<F>>();

    let mut group = c.benchmark_group("SpartanFirstSumcheckStandalone");

    group.bench_function("Original (SpartanInterleaved + SplitEq)", |b: &mut Bencher| {
        b.iter_batched(
            || {
                let new_transcript = transcript.clone();
                return new_transcript;
            },
            |mut transcript| {
                let mut eq_poly = SplitEqPolynomial::new(&tau);

                let mut az_bz_cz_poly =
                    r1cs_builder.compute_spartan_Az_Bz_Cz(&flattened_polys);
                SumcheckInstanceProof::prove_spartan_cubic(
                    num_rounds_x,
                    &mut eq_poly,
                    &mut az_bz_cz_poly,
                    &mut transcript,
                );
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Gruen (SpartanInterleaved + NewSplitEq)", |b: &mut Bencher| {
        b.iter_batched(
            || {
                let new_transcript = transcript.clone();
                return new_transcript;
            },
            |mut transcript| {
                let mut eq_poly = NewSplitEqPolynomial::new(&tau);

                let mut az_bz_cz_poly =
                    r1cs_builder.compute_spartan_Az_Bz_Cz(&flattened_polys);
                SumcheckInstanceProof::prove_spartan_cubic_with_gruen(
                    num_rounds_x,
                    black_box(&mut eq_poly),
                    black_box(&mut az_bz_cz_poly),
                    black_box(&mut transcript),
                );
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Gruen + SVO (NewSpartanInterleaved + NewSplitEq)", |b: &mut Bencher| {
        b.iter_batched(
            || {
                let new_transcript = transcript.clone();
                return new_transcript;
            },
            |mut transcript| {
                SumcheckInstanceProof::prove_spartan_small_value::<3>(
                    num_rounds_x,
                    r1cs_builder.padded_rows_per_step(),
                    &r1cs_builder.uniform_builder.constraints,
                    &r1cs_builder.offset_equality_constraints,
                    &flattened_polys,
                    &tau,
                    &mut transcript,
                )
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(spartan_sumcheck_benches, bench_spartan_sumchecks_in_file);
criterion_main!(spartan_sumcheck_benches);
