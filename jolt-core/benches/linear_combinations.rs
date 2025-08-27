use ark_bn254::Fr;
use ark_ff::{UniformRand, Zero};
use ark_std::rand::{rngs::StdRng, SeedableRng, Rng};
use criterion::{criterion_group, criterion_main, Criterion, black_box};
use jolt_core::field::JoltField;

/// Benchmark 2: Linear combinations with small coefficients
///
/// As specified in Section 8.1: "Next we benchmark the cost of computing a linear combination of
/// field elements with small coefficients in 3 ways.
/// The first is the naive approach of decomposing the linear combination into a series of additions and multiplications.
/// The second uses the small-big multiplications.
/// The third uses our single-reduction algorithm."
fn bench_linear_combinations(c: &mut Criterion) {
    const SAMPLES: usize = 1000;
    let mut rng = StdRng::seed_from_u64(0u64);

    // Test different sizes of linear combinations
    let sizes = [2, 4, 8, 16, 32];

    for &size in &sizes {
        let mut group = c.benchmark_group(&format!("Linear Combinations (n={})", size));

        // Generate test data for this size
        let field_elements: Vec<Vec<Fr>> = (0..SAMPLES)
            .map(|_| (0..size).map(|_| Fr::rand(&mut rng)).collect())
            .collect();

        let small_coeffs: Vec<Vec<u64>> = (0..SAMPLES)
            .map(|_| (0..size).map(|_| rng.gen_range(0..1000)).collect())
            .collect();

        // Method 1: Naive approach - series of additions and multiplications (first convert to field, then multiply)
        group.bench_function("naive_add_mul", |bench| {
            let elements = &field_elements[0];
            let coeffs = &small_coeffs[0];
            bench.iter(|| {
                let mut result = Fr::zero();
                for (elem, coeff) in elements.iter().zip(coeffs.iter()) {
                    result += *elem * Fr::from(*coeff);
                }
                black_box(result)
            })
        });

        // Method 2: Small-big multiplications
        group.bench_function("small_big_mul", |bench| {
            let elements = &field_elements[0];
            let coeffs = &small_coeffs[0];
            bench.iter(|| {
                let mut result = Fr::zero();
                for (elem, coeff) in elements.iter().zip(coeffs.iter()) {
                    result += <Fr as JoltField>::mul_u64(elem, *coeff);
                }
                black_box(result)
            })
        });

        // Precompute (elem, coeff) pairs to avoid per-iter allocation
        let pairs_samples: Vec<Vec<(Fr, u64)>> = (0..SAMPLES)
            .map(|s| field_elements[s].iter().zip(small_coeffs[s].iter())
                .map(|(elem, coeff)| (*elem, *coeff))
                .collect())
            .collect();

        // Method 3: Single-reduction algorithm (our optimized linear combination)
        group.bench_function("single_reduction", |bench| {
            let pairs = &pairs_samples[0];
            bench.iter(|| {
                black_box(<Fr as JoltField>::linear_combination_u64(pairs))
            })
        });

        // Note: Specialized 2-term and 3-term optimized versions are available in arkworks
        // but not exposed through the FieldMulSmall trait in this crate

        group.finish();
    }
}

/// Additional benchmark for signed linear combinations (positive and negative terms)
fn bench_signed_linear_combinations(c: &mut Criterion) {
    const SAMPLES: usize = 1000;
    let mut rng = StdRng::seed_from_u64(0u64);

    let size = 4; // Test with 2+2 terms (2 positive, 2 negative)

    let mut group = c.benchmark_group("Signed Linear Combinations");

    // Generate test data
    let pos_elements: Vec<Vec<Fr>> = (0..SAMPLES)
        .map(|_| (0..size/2).map(|_| Fr::rand(&mut rng)).collect())
        .collect();

    let neg_elements: Vec<Vec<Fr>> = (0..SAMPLES)
        .map(|_| (0..size/2).map(|_| Fr::rand(&mut rng)).collect())
        .collect();

    let pos_coeffs: Vec<Vec<u64>> = (0..SAMPLES)
        .map(|_| (0..size/2).map(|_| rng.gen_range(0..1000)).collect())
        .collect();

    let neg_coeffs: Vec<Vec<u64>> = (0..SAMPLES)
        .map(|_| (0..size/2).map(|_| rng.gen_range(0..1000)).collect())
        .collect();

    // Naive approach with explicit subtraction
    group.bench_function("naive_add_mul (2 positive, 2 negative)", |bench| {
        let pos_elements_sample = &pos_elements[0];
        let pos_coeffs_sample = &pos_coeffs[0];
        let neg_elements_sample = &neg_elements[0];
        let neg_coeffs_sample = &neg_coeffs[0];
        bench.iter(|| {
            let mut result = Fr::zero();
            // Add positive terms
            for (elem, coeff) in pos_elements_sample.iter().zip(pos_coeffs_sample.iter()) {
                result += *elem * Fr::from(*coeff);
            }
            // Subtract negative terms
            for (elem, coeff) in neg_elements_sample.iter().zip(neg_coeffs_sample.iter()) {
                result -= *elem * Fr::from(*coeff);
            }
            black_box(result)
        })
    });

    // Precompute pairs for signed version to avoid per-iter allocation
    let pos_pairs_samples: Vec<Vec<(Fr, u64)>> = (0..SAMPLES)
        .map(|s| pos_elements[s].iter().zip(pos_coeffs[s].iter())
            .map(|(elem, coeff)| (*elem, *coeff))
            .collect())
        .collect();
    let neg_pairs_samples: Vec<Vec<(Fr, u64)>> = (0..SAMPLES)
        .map(|s| neg_elements[s].iter().zip(neg_coeffs[s].iter())
            .map(|(elem, coeff)| (*elem, *coeff))
            .collect())
        .collect();

    // Optimized signed linear combination
    group.bench_function("single_reduction (2 positive, 2 negative)", |bench| {
        let pos_pairs = &pos_pairs_samples[0];
        let neg_pairs = &neg_pairs_samples[0];
        bench.iter(|| {
            black_box(<Fr as JoltField>::linear_combination_i64(pos_pairs, neg_pairs))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_linear_combinations, bench_signed_linear_combinations);
criterion_main!(benches);
