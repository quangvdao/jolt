use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_std::{
    rand::{rngs::StdRng, SeedableRng},
    One, Zero,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jolt_core::field::JoltField;
use jolt_core::subprotocols::multivariate::{
    multi_product_eval, multivariate_product_evaluations_accumulate,
    multivariate_product_evaluations_accumulate_buffered, multivariate_product_evaluations_naive,
};

/// Benchmark 4: Multivariate product evaluation
///
/// As specified in Section 8.1: "Finally, we benchmark the cost of computing the product
/// of d multilinear polynomials in v variables, returning the result in evaluation form."
fn bench_multivariate_product(c: &mut Criterion) {
    const SAMPLES: usize = 1000;
    let mut rng = StdRng::seed_from_u64(0u64);

    // Test different parameter combinations
    let test_cases = [
        (2, 3),  // v=2, d=3
        (3, 4),  // v=3, d=4
        (4, 8),  // v=4, d=8
        (5, 16), // v=5, d=16
        (6, 32), // v=6, d=32
    ];

    for &(v, d) in &test_cases {
        let mut group = c.benchmark_group(&format!("Multivariate Product (v={}, d={})", v, d));

        // Generate random multilinear polynomials on {0,1}^v
        let polynomials: Vec<Vec<Fr>> = (0..d)
            .map(|j| {
                let n = 1usize << v;
                (0..n)
                    .map(|i| {
                        // Vary each polynomial slightly for realistic testing
                        let base = ((i as u64) * 17 + (j as u64) * 101 + 5) % 100;
                        Fr::from_u64(base)
                    })
                    .collect()
            })
            .collect();

        // Naive approach: extrapolate multilinear polynomials first, then multiply
        group.bench_function("naive_extrapolate_then_multiply", |bench| {
            let polys = polynomials.clone();
            bench.iter(|| {
                let result = multivariate_product_evaluations_naive::<Fr>(v, &polys, d);
                black_box(result)
            })
        });

        // New algorithm basic: recursive approach without extrapolation optimization
        group.bench_function("new_algorithm_basic", |bench| {
            let polys = polynomials.clone();
            bench.iter(|| {
                let result = multi_product_eval::<Fr>(v, &polys, d);
                black_box(result)
            })
        });

        // Accumulate version: adds to existing buffer
        let out_len = (d + 1).pow(v as u32) as usize;

        group.bench_function("accumulate_into_buffer", |bench| {
            let polys = polynomials.clone();
            bench.iter(|| {
                let mut sums = vec![Fr::zero(); out_len];
                multivariate_product_evaluations_accumulate::<Fr>(v, &polys, d, &mut sums);
                black_box(sums)
            })
        });

        // Buffered version: uses pre-allocated work buffers
        group.bench_function("buffered_accumulate", |bench| {
            let polys = polynomials.clone();
            bench.iter(|| {
                let mut sums = vec![Fr::zero(); out_len];
                let mut prod_buf = vec![Fr::zero(); out_len];
                let mut work_buf1 = vec![Fr::zero(); out_len];
                let mut work_buf2 = vec![Fr::zero(); out_len];
                let mut work_buf3 = vec![Fr::zero(); out_len];
                multivariate_product_evaluations_accumulate_buffered::<Fr>(
                    v,
                    &polys,
                    d,
                    &mut sums,
                    &mut prod_buf,
                    &mut work_buf1,
                    &mut work_buf2,
                    &mut work_buf3,
                );
                black_box(sums)
            })
        });

        // Benchmark with different polynomial patterns
        let constant_polys: Vec<Vec<Fr>> = (0..d)
            .map(|j| {
                let n = 1usize << v;
                let c = Fr::from_u64((j + 1) as u64);
                vec![c; n] // Constant polynomial p(x) = c
            })
            .collect();

        group.bench_function(&format!("constant_polys_v{}_d{}", v, d), |bench| {
            let polys = constant_polys.clone();
            bench.iter(|| {
                let result = multi_product_eval::<Fr>(v, &polys, d);
                black_box(result)
            })
        });

        let identity_polys: Vec<Vec<Fr>> = (0..d)
            .map(|j| {
                let n = 1usize << v;
                (0..n)
                    .map(|i| if i == 0 { Fr::zero() } else { Fr::one() })
                    .collect()
            })
            .collect();

        group.bench_function(&format!("identity_polys_v{}_d{}", v, d), |bench| {
            let polys = identity_polys.clone();
            bench.iter(|| {
                let result = multi_product_eval::<Fr>(v, &polys, d);
                black_box(result)
            })
        });

        // Benchmark with small coefficients
        let small_polys: Vec<Vec<Fr>> = (0..d)
            .map(|j| {
                let n = 1usize << v;
                (0..n)
                    .map(|i| {
                        let val = ((i as u64) + (j as u64)) % 10;
                        Fr::from_u64(val)
                    })
                    .collect()
            })
            .collect();

        group.bench_function(&format!("small_coeffs_v{}_d{}", v, d), |bench| {
            let polys = small_polys.clone();
            bench.iter(|| {
                let result = multi_product_eval::<Fr>(v, &polys, d);
                black_box(result)
            })
        });

        // Benchmark multiple evaluations
        let num_evaluations = 5;
        let multiple_polys: Vec<Vec<Vec<Fr>>> = (0..num_evaluations)
            .map(|_| {
                (0..d)
                    .map(|j| {
                        let n = 1usize << v;
                        (0..n)
                            .map(|i| {
                                let base = ((i as u64) * 17 + (j as u64) * 101 + 5) % 100;
                                Fr::from_u64(base)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        group.bench_function(
            &format!("multiple_evaluations_{}x_v{}_d{}", num_evaluations, v, d),
            |bench| {
                let polys_list = multiple_polys.clone();
                bench.iter(|| {
                    let mut results = Vec::new();
                    for polys in &polys_list {
                        let result = multi_product_eval::<Fr>(v, polys, d);
                        results.push(result);
                    }
                    black_box(results)
                })
            },
        );

        // Benchmark degree scaling (fixed v, varying d)
        if v <= 4 {
            // Avoid memory issues with large v
            for &test_d in &[1, 2, 4, 8] {
                if test_d <= d {
                    // Only test if we have enough polynomials
                    let test_polys = &polynomials[..test_d];

                    group.bench_function(&format!("degree_scaling_v{}_d{}", v, test_d), |bench| {
                        let polys = test_polys.to_vec();
                        bench.iter(|| {
                            let result = multi_product_eval::<Fr>(v, &polys, test_d);
                            black_box(result)
                        })
                    });
                }
            }
        }

        // Benchmark variable scaling (fixed d, varying v)
        if d <= 8 {
            // Avoid memory issues with large d
            for &test_v in &[1, 2, 3, 4] {
                if test_v <= v {
                    // Only test if we have enough variables
                    let test_polys: Vec<Vec<Fr>> = polynomials
                        .iter()
                        .take(d)
                        .map(|poly| {
                            let n = 1usize << test_v;
                            poly[..n].to_vec()
                        })
                        .collect();

                    group.bench_function(
                        &format!("variable_scaling_v{}_d{}", test_v, d),
                        |bench| {
                            let polys = test_polys.clone();
                            bench.iter(|| {
                                let result = multi_product_eval::<Fr>(test_v, &polys, d);
                                black_box(result)
                            })
                        },
                    );
                }
            }
        }

        group.finish();
    }
}

criterion_group!(benches, bench_multivariate_product);
criterion_main!(benches);
