use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_std::{
    rand::{rngs::StdRng, Rng, SeedableRng},
    One, Zero,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jolt_core::field::JoltField;
use jolt_core::subprotocols::univariate::{
    product_eval_univariate_accumulate, product_eval_univariate_accumulate_naive,
    product_eval_univariate_full_zero_based,
};

/// Benchmark 3: Univariate product evaluation
///
/// As specified in Section 8.1: "Finally, we benchmark the cost of computing the product
/// of d linear polynomials, returning the result in evaluation form."
fn bench_univariate_product(c: &mut Criterion) {
    const SAMPLES: usize = 1000;
    let mut rng = StdRng::seed_from_u64(0u64);

    // Test different degrees
    let degrees = [2, 4, 8, 16, 32];

    for &d in &degrees {
        let mut group = c.benchmark_group(&format!("Univariate Product (d={})", d));

        // Generate random linear polynomials as pairs (p(0), p(1))
        let pairs: Vec<(Fr, Fr)> = (0..d)
            .map(|_| {
                let p0 = Fr::random(&mut rng);
                let p1 = Fr::random(&mut rng);
                (p0, p1)
            })
            .collect();

        // Convert to array format for accumulate functions
        let pairs_array: [(Fr, Fr); 32] = if d <= 32 {
            let mut arr = [(Fr::zero(), Fr::zero()); 32];
            for (i, &pair) in pairs.iter().enumerate() {
                arr[i] = pair;
            }
            arr
        } else {
            [(Fr::zero(), Fr::zero()); 32] // Won't be used for d > 32
        };

        // Naive O(D^2) accumulate approach
        if d <= 32 {
            group.bench_function("naive_accumulate", |bench| {
                let pairs = pairs_array;
                bench.iter(|| {
                    let mut sums = vec![Fr::zero(); if d > 1 { d } else { 1 }];
                    product_eval_univariate_accumulate_naive::<Fr, 32>(&pairs, &mut sums);
                    black_box(sums)
                })
            });
        }

        // Fast accumulate using optimized kernels
        if d <= 32 {
            group.bench_function("fast_accumulate", |bench| {
                let pairs = pairs_array;
                bench.iter(|| {
                    let mut sums = vec![Fr::zero(); if d > 1 { d } else { 1 }];
                    product_eval_univariate_accumulate::<Fr, 32>(&pairs, &mut sums);
                    black_box(sums)
                })
            });
        }

        // Full evaluation (including g(0))
        group.bench_function("full_evaluation", |bench| {
            let pairs = pairs.clone();
            bench.iter(|| {
                let result = product_eval_univariate_full_zero_based(&pairs);
                black_box(result)
            })
        });

        // Benchmark with different polynomial patterns
        let constant_pairs: Vec<(Fr, Fr)> = (0..d)
            .map(|_| {
                let c = Fr::random(&mut rng);
                (c, c) // p(x) = c (constant polynomial)
            })
            .collect();

        group.bench_function(&format!("constant_polys_{}x", d), |bench| {
            let pairs = constant_pairs.clone();
            bench.iter(|| {
                let result = product_eval_univariate_full_zero_based(&pairs);
                black_box(result)
            })
        });

        let identity_pairs: Vec<(Fr, Fr)> = (0..d)
            .map(|_| {
                (Fr::zero(), Fr::one()) // p(x) = x (identity polynomial)
            })
            .collect();

        group.bench_function(&format!("identity_polys_{}x", d), |bench| {
            let pairs = identity_pairs.clone();
            bench.iter(|| {
                let result = product_eval_univariate_full_zero_based(&pairs);
                black_box(result)
            })
        });

        // Benchmark with small coefficients
        let small_pairs: Vec<(Fr, Fr)> = (0..d)
            .map(|_| {
                let p0 = Fr::from_u64(rng.gen_range(1..10));
                let p1 = Fr::from_u64(rng.gen_range(1..10));
                (p0, p1)
            })
            .collect();

        group.bench_function(&format!("small_coeffs_{}x", d), |bench| {
            let pairs = small_pairs.clone();
            bench.iter(|| {
                let result = product_eval_univariate_full_zero_based(&pairs);
                black_box(result)
            })
        });

        // Benchmark accumulation into existing buffer
        if d <= 32 {
            group.bench_function("accumulate_into_buffer", |bench| {
                let pairs = pairs_array;
                bench.iter(|| {
                    let mut sums = vec![Fr::zero(); if d > 1 { d } else { 1 }];
                    // Pre-fill with some values
                    for s in sums.iter_mut() {
                        *s = Fr::random(&mut rng);
                    }
                    product_eval_univariate_accumulate::<Fr, 32>(&pairs, &mut sums);
                    black_box(sums)
                })
            });
        }

        // Benchmark multiple evaluations
        let num_evaluations = 10;
        let multiple_pairs: Vec<Vec<(Fr, Fr)>> = (0..num_evaluations)
            .map(|_| {
                (0..d)
                    .map(|_| {
                        let p0 = Fr::random(&mut rng);
                        let p1 = Fr::random(&mut rng);
                        (p0, p1)
                    })
                    .collect()
            })
            .collect();

        group.bench_function(
            &format!("multiple_evaluations_{}x{}", num_evaluations, d),
            |bench| {
                let pairs_list = multiple_pairs.clone();
                bench.iter(|| {
                    let mut results = Vec::new();
                    for pairs in &pairs_list {
                        let result = product_eval_univariate_full_zero_based(pairs);
                        results.push(result);
                    }
                    black_box(results)
                })
            },
        );

        group.finish();
    }
}

criterion_group!(benches, bench_univariate_product);
criterion_main!(benches);
