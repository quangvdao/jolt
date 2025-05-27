use ark_bn254::{Bn254, Fr};
use ark_ff::{One, Zero};
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
        commitment::hyperkzg::HyperKZG, multilinear_polynomial::MultilinearPolynomial,
        spartan_interleaved_poly::NewSpartanInterleavedPolynomial,
        split_eq_poly::{GruenSplitEqPolynomial, SplitEqPolynomial},
    },
    r1cs::{
        builder::CombinedUniformBuilder,
        inputs::{ConstraintInput, JoltR1CSInputs},
        key::UniformSpartanKey,
    },
    subprotocols::sumcheck::{process_eq_sumcheck_round, SumcheckInstanceProof},
    utils::transcript::{KeccakTranscript, Transcript},
};
use postcard;

type F = Fr;
type PCS = HyperKZG<Bn254, KeccakTranscript>;
type ProofTranscript = KeccakTranscript;

fn setup_for_spartan(
    program_name: &str,
    num_iterations: u32,
) -> (
    JoltPolynomials<F>,
    ProofTranscript,
    CombinedUniformBuilder<C, F, JoltR1CSInputs>,
    UniformSpartanKey<C, JoltR1CSInputs, F>,
) {
    let mut program = host::Program::new(program_name);

    // Prepare input based on program type
    let inputs = if program_name.contains("chain") {
        // Chain-type programs take input data and number of iterations
        let initial_hash_data = [7u8; 32]; // Example initial data
        let guest_input_tuple = (initial_hash_data, num_iterations);
        postcard::to_stdvec(&guest_input_tuple).unwrap()
    } else {
        // For non-chain programs, just use a simple input
        // This can be customized based on what each program expects
        let simple_input = num_iterations as u64;
        postcard::to_stdvec(&simple_input).unwrap()
    };

    let (io_device, mut trace) = program.trace(&inputs);
    let (bytecode, memory_init) = program.decode();

    // Doesn't seem to matter cuz we don't run any PCS commit
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

/// This bench compares the performance of three versions of Spartan sumcheck:
/// 1. Original (before Gruen's or small-value optimization). This is the version in Jolt before our changes
/// 2. Gruen (no small-value optimization). This is the middle version for ``fair'' benchmarking that
/// takes into account only the speedup due to SVO.
/// 3. Gruen + 3 SVO rounds (NewSpartanInterleaved + NewSplitEq). This is our optimized implementation.
fn bench_spartan_sumchecks_in_file(c: &mut Criterion) {
    // Define programs to benchmark
    let programs = vec![
        // "sha3-chain-guest",
        // "fibonacci-guest",
        "sha2-chain-guest",
    ];

    // Define iteration counts for chain programs
    let chain_iteration_counts = vec![8, 16, 32, 64, 128, 256, 512, 1024, 2048];

    // Define iteration counts for non-chain programs (this is just a dummy value)
    let non_chain_iterations = vec![0];

    for program_name in programs {
        // Choose iteration counts based on program type
        let iteration_counts = if program_name.contains("chain") {
            &chain_iteration_counts
        } else {
            &non_chain_iterations
        };

        for &num_iters in iteration_counts {
            println!(
                "Running one-time setup for Spartan sumcheck with program '{}' and {} iterations...",
                program_name, num_iters
            );

            // Pass program_name and num_iters to setup
            let (jolt_polynomials, mut transcript, r1cs_builder, spartan_key) =
                setup_for_spartan(program_name, num_iters);

            println!(
                "Setup complete for {} with {} iterations.",
                program_name, num_iters
            );

            let flattened_polys: Vec<&MultilinearPolynomial<F>> = JoltR1CSInputs::flatten::<C>()
                .iter()
                .map(|var| var.get_ref(&jolt_polynomials))
                .collect();

            let num_rounds_x = spartan_key.num_rows_bits();
            println!("num_rounds_x: {}", num_rounds_x);

            println!("(padded) trace length: {:?}", flattened_polys[0].len());

            let padded_rows_per_step = r1cs_builder.padded_rows_per_step();
            let uniform_constraints = r1cs_builder.uniform_builder.constraints.clone();
            let cross_step_constraints = r1cs_builder.offset_equality_constraints.clone();

            /* Sumcheck 1: Outer sumcheck */

            let tau = (0..num_rounds_x)
                .map(|_i| transcript.challenge_scalar())
                .collect::<Vec<F>>();

            // Use a group name that reflects both program and parameterization
            let mut group = c.benchmark_group(format!(
                "SpartanSumcheck_{}_iters_{}",
                program_name.replace("-guest", ""),
                num_iters
            ));

            group.sample_size(10);
            // group.sampling_mode(SamplingMode::Flat);

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
                            black_box(SumcheckInstanceProof::prove_spartan_cubic_original(
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
                "Gruen (GruenSpartanInterleaved + GruenSplitEq)",
                |b: &mut Bencher| {
                    b.iter_batched(
                        || {
                            let new_transcript = transcript.clone();
                            return new_transcript;
                        },
                        |mut transcript| {
                            let mut eq_poly = GruenSplitEqPolynomial::new(&tau);

                            let mut az_bz_cz_poly =
                                black_box(r1cs_builder.compute_spartan_Az_Bz_Cz_gruen(&flattened_polys));
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

            // group.bench_function(
            //     "Gruen + 1 SVO round (NewSpartanInterleaved + NewSplitEq)",
            //     |b: &mut Bencher| {
            //         b.iter_batched(
            //             || {
            //                 let new_transcript = transcript.clone();
            //                 return new_transcript;
            //             },
            //             |mut transcript| {
            //                 SumcheckInstanceProof::prove_spartan_small_value::<1>(
            //                     num_rounds_x,
            //                     padded_rows_per_step,
            //                     &uniform_constraints,
            //                     &cross_step_constraints,
            //                     &flattened_polys,
            //                     &tau,
            //                     &mut transcript,
            //                 );
            //             },
            //             BatchSize::SmallInput,
            //         );
            //     },
            // );

            // group.bench_function(
            //     "Gruen + 2 SVO rounds (NewSpartanInterleaved + NewSplitEq)",
            //     |b: &mut Bencher| {
            //         b.iter_batched(
            //             || {
            //                 let new_transcript = transcript.clone();
            //                 return new_transcript;
            //             },
            //             |mut transcript| {
            //                 SumcheckInstanceProof::prove_spartan_small_value::<2>(
            //                     num_rounds_x,
            //                     padded_rows_per_step,
            //                     &uniform_constraints,
            //                     &cross_step_constraints,
            //                     &flattened_polys,
            //                     &tau,
            //                     &mut transcript,
            //                 );
            //             },
            //             BatchSize::SmallInput,
            //         );
            //     },
            // );

            group.bench_function(
                "Gruen + 3 SVO rounds (NewSpartanInterleaved + NewSplitEq)",
                |b: &mut Bencher| {
                    b.iter_batched(
                        || {
                            let new_transcript = transcript.clone();
                            return new_transcript;
                        },
                        |mut transcript| {
                            SumcheckInstanceProof::prove_spartan_small_value::<3>(
                                num_rounds_x,
                                padded_rows_per_step,
                                &uniform_constraints,
                                &cross_step_constraints,
                                &flattened_polys,
                                &tau,
                                &mut transcript,
                            );
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            // group.bench_function(
            //     "Gruen + 4 SVO rounds (NewSpartanInterleaved + NewSplitEq)",
            //     |b: &mut Bencher| {
            //         b.iter_batched(
            //             || {
            //                 let new_transcript = transcript.clone();
            //                 return new_transcript;
            //             },
            //             |mut transcript| {
            //                 SumcheckInstanceProof::prove_spartan_small_value::<4>(
            //                     num_rounds_x,
            //                     padded_rows_per_step,
            //                     &uniform_constraints,
            //                     &cross_step_constraints,
            //                     &flattened_polys,
            //                     &tau,
            //                     &mut transcript,
            //                 );
            //             },
            //             BatchSize::SmallInput,
            //         );
            //     },
            // );

            group.finish();
        }
    }
}

fn bench_spartan_svo_components(c: &mut Criterion) {
    // Define the program and iteration counts to test
    let program_name = "sha2-chain-guest";
    let iteration_counts = vec![65];

    for &num_iters in &iteration_counts {
        println!(
            "Running setup for SVO component benchmarking with program '{}' and {} iterations...",
            program_name, num_iters
        );

        // Perform initial setup
        let (jolt_polynomials, mut transcript, r1cs_builder, spartan_key) =
            setup_for_spartan(program_name, num_iters);

        // Get the data needed for sumcheck
        let flattened_polys: Vec<&MultilinearPolynomial<F>> = JoltR1CSInputs::flatten::<C>()
            .iter()
            .map(|var| var.get_ref(&jolt_polynomials))
            .collect();

        let num_rounds_x = spartan_key.num_rows_bits();
        let padded_rows_per_step = r1cs_builder.padded_rows_per_step();
        let uniform_constraints = r1cs_builder.uniform_builder.constraints.clone();
        let cross_step_constraints = r1cs_builder.offset_equality_constraints.clone();

        let tau = (0..num_rounds_x)
            .map(|_i| transcript.challenge_scalar())
            .collect::<Vec<F>>();

        // Create a benchmark group for this specific iteration count
        let mut group = c.benchmark_group(format!("SpartanSVO_Components_{}_iters", num_iters));

        group.sample_size(10);
        group.sampling_mode(SamplingMode::Flat);

        // Benchmark 1: Precomputation phase
        group.bench_function("1_precomputation", |b: &mut Bencher| {
            b.iter_batched(
                || (),
                |_| {
                    black_box(
                        NewSpartanInterleavedPolynomial::<3, F>::new_with_precompute(
                            padded_rows_per_step,
                            &uniform_constraints,
                            &cross_step_constraints,
                            &flattened_polys,
                            &tau,
                        ),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        // For the streaming and remaining rounds, we need the output from precomputation
        let (accums_zero, accums_infty, mut az_bz_cz_poly) =
            NewSpartanInterleavedPolynomial::<3, F>::new_with_precompute(
                padded_rows_per_step,
                &uniform_constraints,
                &cross_step_constraints,
                &flattened_polys,
                &tau,
            );

        // Benchmark 2: SVO rounds (simulate them by calculating the quadratic evals and Lagrange coeffs)
        group.bench_function("2_svo_rounds", |b: &mut Bencher| {
            b.iter_batched(
                || {
                    // Clone data needed for benchmarking
                    let r_challenges = Vec::new();
                    let polys = Vec::new();
                    let claim = F::zero();
                    let transcript_clone = transcript.clone();
                    let accums_zero_clone = accums_zero.clone();
                    let accums_infty_clone = accums_infty.clone();

                    (
                        r_challenges,
                        polys,
                        claim,
                        transcript_clone,
                        accums_zero_clone,
                        accums_infty_clone,
                    )
                },
                |(
                    mut r_challenges,
                    mut polys,
                    mut claim,
                    mut transcript_clone,
                    accums_zero,
                    accums_infty,
                )| {
                    let mut eq_poly = GruenSplitEqPolynomial::new(&tau);
                    let mut lagrange_coeffs: Vec<F> = vec![F::one()];
                    let mut current_acc_zero_offset = 0;
                    let mut current_acc_infty_offset = 0;

                    // Perform the 3 SVO rounds (the same logic as in prove_spartan_small_value)
                    for i in 0..3 {
                        let mut quadratic_eval_0 = F::zero();
                        let mut quadratic_eval_infty = F::zero();

                        let num_vars_in_v_config = i;
                        let num_lagrange_coeffs_for_round =
                            3_usize.checked_pow(num_vars_in_v_config as u32).unwrap();

                        // Compute quadratic_eval_infty
                        let num_accs_infty_curr_round =
                            3_usize.checked_pow(num_vars_in_v_config as u32).unwrap();
                        if num_accs_infty_curr_round > 0 {
                            let accums_infty_slice = &accums_infty[current_acc_infty_offset
                                ..current_acc_infty_offset + num_accs_infty_curr_round];
                            for k in 0..num_lagrange_coeffs_for_round {
                                if k < accums_infty_slice.len() && k < lagrange_coeffs.len() {
                                    quadratic_eval_infty +=
                                        accums_infty_slice[k] * lagrange_coeffs[k];
                                }
                            }
                        }
                        current_acc_infty_offset += num_accs_infty_curr_round;

                        // Compute quadratic_eval_0
                        let num_accs_zero_curr_round = if num_vars_in_v_config == 0 {
                            0
                        } else {
                            3_usize.checked_pow(num_vars_in_v_config as u32).unwrap()
                                - 2_usize.checked_pow(num_vars_in_v_config as u32).unwrap()
                        };

                        if num_accs_zero_curr_round > 0 {
                            let accums_zero_slice = &accums_zero[current_acc_zero_offset
                                ..current_acc_zero_offset + num_accs_zero_curr_round];

                            // The logic to compute quadratic_eval_0 using accums_zero_slice
                            // This is simplified for benchmarking purposes
                            for val in accums_zero_slice {
                                quadratic_eval_0 += *val;
                            }
                        }
                        current_acc_zero_offset += num_accs_zero_curr_round;

                        let r_i = process_eq_sumcheck_round(
                            (quadratic_eval_0, quadratic_eval_infty),
                            &mut eq_poly,
                            &mut polys,
                            &mut r_challenges,
                            &mut claim,
                            &mut transcript_clone,
                        );

                        // Update Lagrange coefficients
                        if i < 2 {
                            let lagrange_coeffs_r_i = [F::one() - r_i, r_i, r_i * (r_i - F::one())];
                            lagrange_coeffs = lagrange_coeffs_r_i
                                .iter()
                                .flat_map(|lagrange_coeff| {
                                    lagrange_coeffs
                                        .iter()
                                        .map(move |coeff| *lagrange_coeff * *coeff)
                                })
                                .collect();
                        }
                    }

                    black_box((r_challenges, polys, claim))
                },
                BatchSize::SmallInput,
            )
        });

        // For benchmarking the streaming round, simulate SVO rounds to get the proper state
        let mut eq_poly = GruenSplitEqPolynomial::new(&tau);
        let mut r_challenges = Vec::new();
        let mut polys = Vec::new();
        let mut claim = F::zero();
        let mut lagrange_coeffs: Vec<F> = vec![F::one()];
        let mut current_acc_infty_offset = 0;

        for i in 0..3 {
            let quadratic_eval_0 = F::zero();
            let mut quadratic_eval_infty = F::zero();

            let num_vars_in_v_config = i;
            let num_lagrange_coeffs_for_round =
                3_usize.checked_pow(num_vars_in_v_config as u32).unwrap();

            // Compute quadratic_eval_infty (simplified calculation for setup)
            let num_accs_infty_curr_round =
                3_usize.checked_pow(num_vars_in_v_config as u32).unwrap();
            if num_accs_infty_curr_round > 0 {
                let accums_infty_slice = &accums_infty[current_acc_infty_offset
                    ..current_acc_infty_offset + num_accs_infty_curr_round];
                for k in 0..num_lagrange_coeffs_for_round {
                    if k < accums_infty_slice.len() && k < lagrange_coeffs.len() {
                        quadratic_eval_infty += accums_infty_slice[k] * lagrange_coeffs[k];
                    }
                }
            }
            current_acc_infty_offset += num_accs_infty_curr_round;

            // Skip full calculation for setup
            let r_i = process_eq_sumcheck_round(
                (quadratic_eval_0, quadratic_eval_infty),
                &mut eq_poly,
                &mut polys,
                &mut r_challenges,
                &mut claim,
                &mut transcript,
            );

            if i < 2 {
                let lagrange_coeffs_r_i = [F::one() - r_i, r_i, r_i * (r_i - F::one())];
                lagrange_coeffs = lagrange_coeffs_r_i
                    .iter()
                    .flat_map(|lagrange_coeff| {
                        lagrange_coeffs
                            .iter()
                            .map(move |coeff| *lagrange_coeff * *coeff)
                    })
                    .collect();
            }
        }

        // Benchmark 3: Streaming sumcheck round
        group.bench_function("3_streaming_round", |b: &mut Bencher| {
            b.iter_batched(
                || {
                    // Clone necessary data
                    let az_bz_cz_poly_clone = az_bz_cz_poly.clone();
                    let eq_poly_clone = eq_poly.clone();
                    let r_challenges_clone = r_challenges.clone();
                    let polys_clone = polys.clone();
                    let claim_clone = claim.clone();
                    let transcript_clone = transcript.clone();

                    (
                        az_bz_cz_poly_clone,
                        eq_poly_clone,
                        r_challenges_clone,
                        polys_clone,
                        claim_clone,
                        transcript_clone,
                    )
                },
                |(
                    mut az_bz_cz_poly_clone,
                    mut eq_poly_clone,
                    mut r_challenges_clone,
                    mut polys_clone,
                    mut claim_clone,
                    mut transcript_clone,
                )| {
                    black_box(az_bz_cz_poly_clone.streaming_sumcheck_round(
                        &mut eq_poly_clone,
                        &mut transcript_clone,
                        &mut r_challenges_clone,
                        &mut polys_clone,
                        &mut claim_clone,
                    ))
                },
                BatchSize::SmallInput,
            )
        });

        // Perform the streaming sumcheck round to prepare for the remaining rounds
        az_bz_cz_poly.streaming_sumcheck_round(
            &mut eq_poly,
            &mut transcript,
            &mut r_challenges,
            &mut polys,
            &mut claim,
        );

        // Benchmark 4: Remaining sumcheck rounds
        if num_rounds_x > 4 {
            // Only if we have remaining rounds
            group.bench_function("4_remaining_rounds", |b: &mut Bencher| {
                b.iter_batched(
                    || {
                        // Clone necessary data
                        let az_bz_cz_poly_clone = az_bz_cz_poly.clone();
                        let eq_poly_clone = eq_poly.clone();
                        let r_challenges_clone = r_challenges.clone();
                        let polys_clone = polys.clone();
                        let claim_clone = claim.clone();
                        let transcript_clone = transcript.clone();

                        (
                            az_bz_cz_poly_clone,
                            eq_poly_clone,
                            r_challenges_clone,
                            polys_clone,
                            claim_clone,
                            transcript_clone,
                        )
                    },
                    |(
                        mut az_bz_cz_poly_clone,
                        mut eq_poly_clone,
                        mut r_challenges_clone,
                        mut polys_clone,
                        mut claim_clone,
                        mut transcript_clone,
                    )| {
                        // Perform all remaining rounds
                        for _ in 4..num_rounds_x {
                            az_bz_cz_poly_clone.remaining_sumcheck_round(
                                &mut eq_poly_clone,
                                &mut transcript_clone,
                                &mut r_challenges_clone,
                                &mut polys_clone,
                                &mut claim_clone,
                            );
                        }
                        black_box(())
                    },
                    BatchSize::SmallInput,
                )
            });
        }

        group.finish();
    }
}

criterion_group!(spartan_sumcheck_benches, bench_spartan_sumchecks_in_file);
criterion_group!(spartan_svo_components, bench_spartan_svo_components);
criterion_main!(spartan_sumcheck_benches);
