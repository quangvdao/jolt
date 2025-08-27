use ark_bn254::Fr;
use ark_ff::Zero;
use ark_ff::UniformRand;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use criterion::{criterion_group, criterion_main, Criterion, black_box};
use jolt_core::subprotocols::univariate::{
    product_eval_univariate_accumulate,
    product_eval_univariate_accumulate_naive,
};

/// Benchmark 3: Univariate product evaluation
///
/// This benchmarks our specialized univariate product handling for computing the product
/// of D linear polynomials, as implemented in univariate.rs. This is a key component
/// of the multivariate product algorithm and benefits from our optimized evaluation-based
/// approach with hardcoded formulas for powers of two.
fn bench_univariate_product(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0u64);

    // Test degrees as specified: 3, 4, 8, 16, 32
    let degrees = [3, 4, 8, 16, 32];

    for &d in &degrees {
        let mut group = c.benchmark_group(&format!("Univariate Product (d={})", d));

        // Generate random linear polynomial pairs (a0, a1) where p(x) = a0 + (a1-a0)*x
        let pairs: Vec<(Fr, Fr)> = (0..d)
            .map(|_| {
                let a0 = Fr::rand(&mut rng);
                let a1 = Fr::rand(&mut rng);
                (a0, a1)
            })
            .collect();

        // Precompute fixed-size array once per group; measure everything after pair construction
        match d {
            3 => {
                let arr: [(Fr, Fr); 3] = pairs[..3].try_into().unwrap();
                let mut sums_naive = vec![Fr::zero(); 3];
                let mut sums_opt = vec![Fr::zero(); 3];

                group.bench_function("accumulate_naive_O(d^2)", |bench| {
                    bench.iter(|| {
                        product_eval_univariate_accumulate_naive::<Fr, 3>(black_box(&arr), black_box(&mut sums_naive));
                    })
                });

                group.bench_function("accumulate", |bench| {
                    bench.iter(|| {
                        sums_opt.fill(Fr::zero());
                        product_eval_univariate_accumulate::<Fr, 3>(black_box(&arr), black_box(&mut sums_opt));
                    })
                });
            }
            4 => {
                let arr: [(Fr, Fr); 4] = pairs[..4].try_into().unwrap();
                let mut sums_naive = vec![Fr::zero(); 4];
                let mut sums_opt = vec![Fr::zero(); 4];

                group.bench_function("accumulate_naive_O(d^2)", |bench| {
                    bench.iter(|| {
                        product_eval_univariate_accumulate_naive::<Fr, 4>(black_box(&arr), black_box(&mut sums_naive));
                    })
                });

                group.bench_function("accumulate", |bench| {
                    bench.iter(|| {
                        product_eval_univariate_accumulate::<Fr, 4>(black_box(&arr), black_box(&mut sums_opt));
                    })
                });
            }
            8 => {
                let arr: [(Fr, Fr); 8] = pairs[..8].try_into().unwrap();
                let mut sums_naive = vec![Fr::zero(); 8];
                let mut sums_opt = vec![Fr::zero(); 8];

                group.bench_function("accumulate_naive_O(d^2)", |bench| {
                    bench.iter(|| {
                        product_eval_univariate_accumulate_naive::<Fr, 8>(black_box(&arr), black_box(&mut sums_naive));
                    })
                });

                group.bench_function("accumulate", |bench| {
                    bench.iter(|| {
                        product_eval_univariate_accumulate::<Fr, 8>(black_box(&arr), black_box(&mut sums_opt));
                    })
                });
            }
            16 => {
                let arr: [(Fr, Fr); 16] = pairs[..16].try_into().unwrap();
                let mut sums_naive = vec![Fr::zero(); 16];
                let mut sums_opt = vec![Fr::zero(); 16];

                group.bench_function("accumulate_naive_O(d^2)", |bench| {
                    bench.iter(|| {
                        product_eval_univariate_accumulate_naive::<Fr, 16>(black_box(&arr), black_box(&mut sums_naive));
                    })
                });

                group.bench_function("accumulate", |bench| {
                    bench.iter(|| {
                        product_eval_univariate_accumulate::<Fr, 16>(black_box(&arr), black_box(&mut sums_opt));
                    })
                });
            }
            32 => {
                let arr: [(Fr, Fr); 32] = pairs[..32].try_into().unwrap();
                let mut sums_naive = vec![Fr::zero(); 32];
                let mut sums_opt = vec![Fr::zero(); 32];

                group.bench_function("accumulate_naive_O(d^2)", |bench| {
                    bench.iter(|| {
                        product_eval_univariate_accumulate_naive::<Fr, 32>(black_box(&arr), black_box(&mut sums_naive));
                    })
                });

                group.bench_function("accumulate", |bench| {
                    bench.iter(|| {
                        product_eval_univariate_accumulate::<Fr, 32>(black_box(&arr), black_box(&mut sums_opt));
                    })
                });
            }
            _ => unreachable!(),
        }

        group.finish();
    }
}

criterion_group!(benches, bench_univariate_product);
criterion_main!(benches);
