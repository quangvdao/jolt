use ark_bn254::Fr;
use ark_ff::{BigInteger, UniformRand};
use ark_std::{
    rand::{rngs::StdRng, Rng, SeedableRng},
    One, Zero,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
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
    let a_samples: Vec<Fr> = (0..SAMPLES).map(|_| Fr::random(&mut rng)).collect();
    let b_samples: Vec<Fr> = (0..SAMPLES).map(|_| Fr::random(&mut rng)).collect();

    // Generate small values for small-small and small-big multiplications
    let small_u64_samples: Vec<u64> = (0..SAMPLES).map(|_| rng.gen_range(0..1000)).collect();
    let small_i64_samples: Vec<i64> = (0..SAMPLES).map(|_| rng.gen_range(-1000..1000)).collect();

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

    // Small-big multiplication: u64 using JoltField
    group.bench_function("small_big_mul_u64", |bench| {
        let a = a_samples[0];
        let small = small_u64_samples[0];
        bench.iter(|| <Fr as JoltField>::mul_u64(&a, black_box(small)))
    });

    // Small-big multiplication: u64 (10 operations)
    group.bench_function("small_big_mul_u64_10x", |bench| {
        let a = a_samples[0];
        let small = small_u64_samples[0];
        bench.iter(|| {
            for _ in 0..10 {
                let result = <Fr as JoltField>::mul_u64(&a, black_box(small));
                black_box(result);
            }
        })
    });

    // Small-big multiplication: i64 using JoltField
    group.bench_function("small_big_mul_i64", |bench| {
        let a = a_samples[0];
        let small = small_i64_samples[0];
        bench.iter(|| <Fr as JoltField>::mul_i64(&a, black_box(small)))
    });

    // Small-big multiplication: i64 (10 operations)
    group.bench_function("small_big_mul_i64_10x", |bench| {
        let a = a_samples[0];
        let small = small_i64_samples[0];
        bench.iter(|| {
            for _ in 0..10 {
                let result = <Fr as JoltField>::mul_i64(&a, black_box(small));
                black_box(result);
            }
        })
    });

    // Small-big multiplication: u128 using JoltField
    group.bench_function("small_big_mul_u128", |bench| {
        let a = a_samples[0];
        let small = small_u64_samples[0] as u128;
        bench.iter(|| <Fr as JoltField>::mul_u128(&a, black_box(small)))
    });

    // Small-big multiplication: i128 using JoltField
    group.bench_function("small_big_mul_i128", |bench| {
        let a = a_samples[0];
        let small = small_i64_samples[0] as i128;
        bench.iter(|| <Fr as JoltField>::mul_i128(&a, black_box(small)))
    });

    // Field conversion benchmarks using JoltField
    group.bench_function("field_from_u64", |bench| {
        let small = small_u64_samples[0];
        bench.iter(|| black_box(<Fr as JoltField>::from_u64(black_box(small))))
    });

    // Field conversion from u64 (10 operations)
    group.bench_function("field_from_u64_10x", |bench| {
        let small = small_u64_samples[0];
        bench.iter(|| {
            let mut result = Fr::zero();
            for _ in 0..10 {
                result = black_box(<Fr as JoltField>::from_u64(black_box(small)));
            }
            black_box(result)
        })
    });

    group.bench_function("field_from_i64", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            black_box(<Fr as JoltField>::from_i64(black_box(small_i64_samples[i])))
        })
    });

    // Field conversion from i64 (10 operations)
    group.bench_function("field_from_i64_10x", |bench| {
        let small = small_i64_samples[0];
        bench.iter(|| {
            let mut result = Fr::one();
            for _ in 0..10 {
                result = black_box(<Fr as JoltField>::from_i64(black_box(small)));
            }
            black_box(result)
        })
    });

    // Power of 2 multiplication using JoltField
    group.bench_function("mul_pow_2", |bench| {
        let a = a_samples[0];
        let pow = 64usize;
        bench.iter(|| <Fr as JoltField>::mul_pow_2(&a, black_box(pow)))
    });

    // Power of 2 multiplication (10 operations)
    group.bench_function("mul_pow_2_10x", |bench| {
        let a = a_samples[0];
        let pow = 64usize;
        bench.iter(|| {
            for _ in 0..10 {
                let result = <Fr as JoltField>::mul_pow_2(&a, black_box(pow));
                black_box(result);
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_primitive_field_ops);
criterion_main!(benches);
