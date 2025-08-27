use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_std::{
    rand::{rngs::StdRng, Rng, SeedableRng},
    One, Zero,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jolt_core::field::JoltField;

/// Benchmark 2: Linear combinations with small coefficients
///
/// As specified in Section 8.1: "Next we benchmark the cost of computing a linear combination
/// of field elements with small coefficients using three approaches: naive, small-big, and single-reduction."
fn bench_linear_combinations(c: &mut Criterion) {
    const SAMPLES: usize = 1000;
    let mut rng = StdRng::seed_from_u64(0u64);

    // Generate random field elements and coefficients for different sizes
    let sizes = [2, 4, 8, 16, 32];

    for &size in &sizes {
        let mut group = c.benchmark_group(&format!("Linear Combinations ({} terms)", size));

        // Generate test data for this size
        let elements: Vec<Fr> = (0..size).map(|_| Fr::random(&mut rng)).collect();
        let coeffs: Vec<u64> = (0..size).map(|_| rng.gen_range(1..100)).collect();

        // Convert to pairs format for single-reduction
        let pairs: Vec<(Fr, u64)> = elements
            .iter()
            .zip(coeffs.iter())
            .map(|(elem, coeff)| (*elem, *coeff))
            .collect();

        // Split into positive and negative for signed combinations
        let mid = size / 2;
        let pos_pairs: Vec<(Fr, u64)> = pairs[..mid].to_vec();
        let neg_pairs: Vec<(Fr, u64)> = pairs[mid..].to_vec();

        // Naive approach: decompose into series of additions and multiplications
        group.bench_function(&format!("naive_add_mul_{}x", size), |bench| {
            let elems = elements.clone();
            let coeffs = coeffs.clone();
            bench.iter(|| {
                let mut result = Fr::zero();
                for (elem, coeff) in elems.iter().zip(coeffs.iter()) {
                    result += *elem * Fr::from_u64(*coeff);
                }
                black_box(result)
            })
        });

        // Small-big approach: using mul_u64 operations
        group.bench_function(&format!("small_big_mul_{}x", size), |bench| {
            let elems = elements.clone();
            let coeffs = coeffs.clone();
            bench.iter(|| {
                let mut result = Fr::zero();
                for (elem, coeff) in elems.iter().zip(coeffs.iter()) {
                    result += <Fr as JoltField>::mul_u64(elem, *coeff);
                }
                black_box(result)
            })
        });

        // Single-reduction approach: using linear_combination_u64
        group.bench_function(&format!("single_reduction_{}x", size), |bench| {
            let pairs = pairs.clone();
            bench.iter(|| {
                let result = <Fr as JoltField>::linear_combination_u64(&pairs);
                black_box(result)
            })
        });

        // Signed linear combinations with positive/negative terms
        group.bench_function(&format!("signed_combination_{}x", size), |bench| {
            let pos = pos_pairs.clone();
            let neg = neg_pairs.clone();
            bench.iter(|| {
                let result = <Fr as JoltField>::linear_combination_i64(&pos, &neg);
                black_box(result)
            })
        });

        // Specialized optimizations for small sizes
        if size == 2 {
            group.bench_function("optimized_2_term", |bench| {
                let (elem1, coeff1) = (elements[0], coeffs[0]);
                let (elem2, coeff2) = (elements[1], coeffs[1]);
                bench.iter(|| {
                    let result = <Fr as JoltField>::mul_u64(&elem1, coeff1)
                        + <Fr as JoltField>::mul_u64(&elem2, coeff2);
                    black_box(result)
                })
            });
        }

        if size == 3 {
            group.bench_function("optimized_3_term", |bench| {
                let (elem1, coeff1) = (elements[0], coeffs[0]);
                let (elem2, coeff2) = (elements[1], coeffs[1]);
                let (elem3, coeff3) = (elements[2], coeffs[2]);
                bench.iter(|| {
                    let result = <Fr as JoltField>::mul_u64(&elem1, coeff1)
                        + <Fr as JoltField>::mul_u64(&elem2, coeff2)
                        + <Fr as JoltField>::mul_u64(&elem3, coeff3);
                    black_box(result)
                })
            });
        }

        // Benchmark with different coefficient ranges
        let small_coeffs: Vec<u64> = (0..size).map(|_| rng.gen_range(1..10)).collect();
        let small_pairs: Vec<(Fr, u64)> = elements
            .iter()
            .zip(small_coeffs.iter())
            .map(|(elem, coeff)| (*elem, *coeff))
            .collect();

        group.bench_function(&format!("small_coeffs_{}x", size), |bench| {
            let pairs = small_pairs.clone();
            bench.iter(|| {
                let result = <Fr as JoltField>::linear_combination_u64(&pairs);
                black_box(result)
            })
        });

        // Benchmark with mixed positive/negative coefficients
        let mixed_coeffs: Vec<i64> = (0..size)
            .map(|i| {
                if i % 2 == 0 {
                    rng.gen_range(1..50)
                } else {
                    -rng.gen_range(1..50)
                }
            })
            .collect();

        let mixed_pos: Vec<(Fr, u64)> = mixed_coeffs
            .iter()
            .enumerate()
            .filter(|(_, &coeff)| coeff > 0)
            .map(|(i, &coeff)| (elements[i], coeff as u64))
            .collect();
        let mixed_neg: Vec<(Fr, u64)> = mixed_coeffs
            .iter()
            .enumerate()
            .filter(|(_, &coeff)| coeff < 0)
            .map(|(i, &coeff)| (elements[i], (-coeff) as u64))
            .collect();

        group.bench_function(&format!("mixed_signs_{}x", size), |bench| {
            let pos = mixed_pos.clone();
            let neg = mixed_neg.clone();
            bench.iter(|| {
                let result = <Fr as JoltField>::linear_combination_i64(&pos, &neg);
                black_box(result)
            })
        });

        group.finish();
    }
}

criterion_group!(benches, bench_linear_combinations);
criterion_main!(benches);
