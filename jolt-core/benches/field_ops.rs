use ark_bn254::Fr;
use ark_ff::{UniformRand, BigInteger};
use ark_std::rand::{rngs::StdRng, SeedableRng, Rng};
use criterion::{criterion_group, criterion_main, Criterion, black_box};
use jolt_core::field::JoltField;

/// Benchmark 1: Primitive finite field operations
///
/// As specified in Section 8.1: "We first consider the cost of primitive finite field operations.
/// As baselines, we benchmark addition and multiplication operations.
/// We compare this to the cost of small-small and small-big multiplications."
fn bench_primitive_field_ops(c: &mut Criterion) {
    const SAMPLES: usize = 1000;
    let mut rng = StdRng::seed_from_u64(0u64);

    // Generate random field elements
    let a_samples: Vec<Fr> = (0..SAMPLES).map(|_| Fr::rand(&mut rng)).collect();
    let b_samples: Vec<Fr> = (0..SAMPLES).map(|_| Fr::rand(&mut rng)).collect();

    // Generate small values for small-small and small-big multiplications
    let small_u64_samples: Vec<u64> = (0..SAMPLES).map(|_| rng.gen_range(0..1000)).collect();
    // let small_i64_samples: Vec<i64> = (0..SAMPLES).map(|_| rng.gen_range(-1000..1000)).collect();

    let mut group = c.benchmark_group("Primitive Field Operations");

    // Small-small multiplication
    group.bench_function("small_small_mul_u64", |bench| {
        let small_a = small_u64_samples[0];
        let small_b = small_u64_samples[1];
        bench.iter(|| small_a * small_b)
    });

    // Small-small multiplication (10 operations)
    group.bench_function("small_small_mul_u64_10x", |bench| {
        let a = small_u64_samples[0];
        let b = small_u64_samples[1];
        bench.iter(|| {
            for _ in 0..10 {
                let result = black_box(a) * black_box(b);
                black_box(result);
            }
        })
    });

    // Baseline: Addition
    group.bench_function("field_addition", |bench| {
        let a = a_samples[0];
        let b = b_samples[0];
        bench.iter(|| black_box(a) + black_box(b))
    });

    // Baseline: Addition (10 operations)
    group.bench_function("field_addition_10x", |bench| {
        let a = a_samples[0];
        let b = b_samples[0];
        bench.iter(|| {
            for _ in 0..10 {
                let result = black_box(a) + black_box(b);
                black_box(result);
            }
        })
    });

    // Baseline: Multiplication (big-by-big)
    group.bench_function("field_multiplication", |bench| {
        let a = a_samples[0];
        let b = b_samples[0];
        bench.iter(|| black_box(a) * black_box(b))
    });

    // Baseline: Multiplication (10 operations)
    group.bench_function("field_multiplication_10x", |bench| {
        let a = a_samples[0];
        let b = b_samples[0];
        bench.iter(|| {
            for _ in 0..10 {
                let result = black_box(a) * black_box(b);
                black_box(result);
            }
        })
    });

    // Small-big multiplication: u64
    group.bench_function("small_big_mul_u64", |bench| {
        let a = a_samples[0];
        let small = small_u64_samples[0];
        bench.iter(|| {
            <Fr as JoltField>::mul_u64(&black_box(a), black_box(small))
        })
    });

    // Small-big multiplication: u64 (10 operations)
    group.bench_function("small_big_mul_u64_10x", |bench| {
        let a = a_samples[0];
        let small = small_u64_samples[0];
        bench.iter(|| {
            for _ in 0..10 {
                let result = <Fr as JoltField>::mul_u64(&black_box(a), black_box(small));
                black_box(result);
            }
        })
    });

    // // Small-big multiplication: i64
    // group.bench_function("small_big_mul_i64", |bench| {
    //     let a = a_samples[0];
    //     let small = small_i64_samples[0];
    //     bench.iter(|| {
    //         <Fr as JoltField>::mul_i64(black_box(a), black_box(small))
    //     })
    // });

    // // Small-big multiplication: i64 (10 operations)
    // group.bench_function("small_big_mul_i64_10x", |bench| {
    //     let a = a_samples[0];
    //     let small = small_i64_samples[0];
    //     bench.iter(|| {
    //         for _ in 0..10 {
    //             _ = <Fr as JoltField>::mul_i64(black_box(a), black_box(small));
    //         }
    //     })
    // });

    // Breakdown of mul_u64: BigInt multiplication + Barrett reduction
    group.bench_function("bigint_mul_u64_w_carry (1st half of mul_u64)", |bench| {
        let a = a_samples[0];
        let small = small_u64_samples[0];
        bench.iter(|| {
            // Extract the BigInt from the field element and multiply
            ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&black_box(a.0), black_box(small))
        })
    });

    // BigInt multiplication (10 operations)
    group.bench_function("bigint_mul_u64_w_carry_10x", |bench| {
        let a = a_samples[0];
        let small = small_u64_samples[0];
        bench.iter(|| {
            for _ in 0..10 {
                let result = ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&black_box(a.0), black_box(small));
                black_box(result);
            }
        })
    });

    group.bench_function("barrett_reduce_nplus1 (2nd half of mul_u64)", |bench| {
        // Pre-compute BigInt<5> values for reduction
        let bigint_nplus1_samples: Vec<_> = (0..SAMPLES).map(|i| {
            ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&a_samples[i].0, small_u64_samples[i])
        }).collect();

        bench.iter(|| {
            Fr::from_unchecked_nplus1::<5>(black_box(bigint_nplus1_samples[0]))
        })
    });

    // Barrett reduction (10 operations)
    group.bench_function("barrett_reduce_nplus1_10x", |bench| {
        // Pre-compute BigInt<5> values for reduction
        let bigint_nplus1_samples: Vec<_> = (0..SAMPLES).map(|i| {
            ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&a_samples[i].0, small_u64_samples[i])
        }).collect();

        bench.iter(|| {
            for _ in 0..10 {
                let result = Fr::from_unchecked_nplus1::<5>(black_box(bigint_nplus1_samples[0]));
                black_box(result);
            }
        })
    });

    // Breakdown of regular field multiplication: BigInt multiplication + Montgomery reduction
    group.bench_function("bigint_mul_nxn_to_2n (1st half of field mul)", |bench| {
        let a = a_samples[0];
        let b = b_samples[0];
        bench.iter(|| {
            // N x N -> 2N limb multiplication
            black_box(a.0).mul_trunc::<4,8>(black_box(&b.0))
        })
    });

    // BigInt multiplication NxN->2N (10 operations)
    group.bench_function("bigint_mul_nxn_to_2n_10x", |bench| {
        let a = a_samples[0].0;
        let b = b_samples[0].0;
        bench.iter(|| {
            for _ in 0..10 {
                let result = black_box(a).mul_trunc::<4,8>(black_box(&b));
                black_box(result);
            }
        })
    });

    group.bench_function("montgomery_reduce_2n (2nd half of field mul)", |bench| {
        // Pre-compute BigInt<8> values (2N limbs) for Montgomery reduction
        let bigint_2n_samples: Vec<ark_ff::BigInt<8>> = (0..SAMPLES).map(|i| {
            a_samples[i].0.mul_trunc::<4,8>(&b_samples[i].0)
        }).collect();

        bench.iter(|| {
            Fr::montgomery_reduce_2n::<8>(black_box(bigint_2n_samples[0]))
        })
    });

    // Montgomery reduction (10 operations)
    group.bench_function("montgomery_reduce_2n_10x", |bench| {
        // Pre-compute BigInt<8> values (2N limbs) for Montgomery reduction
        let bigint_2n_samples: Vec<ark_ff::BigInt<8>> = (0..SAMPLES).map(|i| {
            a_samples[i].0.mul_trunc::<4,8>(&b_samples[i].0)
        }).collect();

        let sample = bigint_2n_samples[0];
        bench.iter(|| {
            for _ in 0..10 {
                let result = Fr::montgomery_reduce_2n::<8>(black_box(sample));
                black_box(result);
            }
        })
    });

    // // Field conversion benchmarks
    // group.bench_function("field_from_u64", |bench| {
    //     bench.iter(|| {
    //         black_box(<Fr as JoltField>::from_u64(black_box(small_u64_samples[0])))
    //     })
    // });

    // // Field conversion from u64 (10 operations)
    // group.bench_function("field_from_u64_10x", |bench| {
    //     let small = small_u64_samples[0];
    //     bench.iter(|| {
    //         let mut result = Fr::from(0u64);
    //         for _ in 0..10 {
    //             result = black_box(<Fr as JoltField>::from_u64(black_box(small)));
    //         }
    //         black_box(result)
    //     })
    // });

    // group.bench_function("field_from_i64", |bench| {
    //     let mut i = 0;
    //     bench.iter(|| {
    //         i = (i + 1) % SAMPLES;
    //         black_box(<Fr as JoltField>::mul_i64(black_box(Fr::from(1u64)), black_box(small_i64_samples[i])))
    //     })
    // });

    // // Field conversion from i64 (10 operations)
    // group.bench_function("field_from_i64_10x", |bench| {
    //     let small = small_i64_samples[0];
    //     bench.iter(|| {
    //         let mut result = Fr::from(1u64);
    //         for _ in 0..10 {
    //             result = black_box(<Fr as JoltField>::mul_i64(black_box(result), black_box(small)));
    //         }
    //         black_box(result)
    //     })
    // });

    group.finish();
}

criterion_group!(benches, bench_primitive_field_ops);
criterion_main!(benches);
