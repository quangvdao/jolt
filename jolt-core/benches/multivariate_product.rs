use ark_bn254::Fr;
use criterion::{criterion_group, criterion_main, Criterion, black_box};
use jolt_core::subprotocols::multivariate::{
    multi_product_eval,
    multivariate_product_evaluations_naive,
};

/// Consolidated Multivariate Product Benchmarks
///
/// This file consolidates what were previously three separate benchmark functions:
/// - bench_multivariate_product (original test cases)
/// - bench_degree_scaling (varying degrees with fixed v=3)
/// - bench_variable_scaling (varying variables with fixed d=4)
///
/// Benefits of consolidation:
/// 1. Eliminates ~150 lines of duplicated code
/// 2. Single source of truth for polynomial generation and benchmark structure
/// 3. Easier maintenance and consistency across different test types
/// 4. Flexible configuration through test_configs vector
///
/// Note: This benchmark only uses paper-aligned implementations with the correct
/// U_d = {0,1} if d=1, {0,1,...,d-1,∞} if d≥2 definition.
fn bench_multivariate_product_consolidated(c: &mut Criterion) {
    let test_configs = vec![
        // Specific combinations: v=1,2,3,4 with d=4,8,16,32
        (1, 4, "v=1_d=4", 0u64),
        (1, 8, "v=1_d=8", 0u64),
        (1, 16, "v=1_d=16", 0u64),
        (1, 32, "v=1_d=32", 0u64),
        (2, 4, "v=2_d=4", 0u64),
        (2, 8, "v=2_d=8", 0u64),
        (2, 16, "v=2_d=16", 0u64),
        (2, 32, "v=2_d=32", 0u64),
        (3, 4, "v=3_d=4", 0u64),
        (3, 8, "v=3_d=8", 0u64),
        (3, 16, "v=3_d=16", 0u64),
        (3, 32, "v=3_d=32", 0u64),
        (4, 4, "v=4_d=4", 0u64),
        (4, 8, "v=4_d=8", 0u64),
        (4, 16, "v=4_d=16", 0u64),
        (4, 32, "v=4_d=32", 0u64),

        // Commented out original test cases
        // (2, 3, "v=2_d=3", 0u64),
        // (3, 4, "v=3_d=4", 0u64),
        // (4, 8, "v=4_d=8", 0u64),
        // (5, 16, "v=5_d=16", 0u64),
        // (6, 32, "v=6_d=32", 0u64),

        // Commented out degree scaling tests
        // (3, 2, "degree_scaling_d=2", 1u64),
        // (3, 3, "degree_scaling_d=3", 1u64),
        // (3, 4, "degree_scaling_d=4", 1u64),
        // (3, 6, "degree_scaling_d=6", 1u64),
        // (3, 8, "degree_scaling_d=8", 1u64),
        // (3, 12, "degree_scaling_d=12", 1u64),
        // (3, 16, "degree_scaling_d=16", 1u64),

        // Commented out variable scaling tests
        // (2, 4, "variable_scaling_v=2", 2u64),
        // (3, 4, "variable_scaling_v=3", 2u64),
        // (4, 4, "variable_scaling_v=4", 2u64),
        // (5, 4, "variable_scaling_v=5", 2u64),
        // (6, 4, "variable_scaling_v=6", 2u64),
        // (7, 4, "variable_scaling_v=7", 2u64),
    ];

    for (v, d, group_suffix, seed_offset) in test_configs {
        // Skip very large cases that might be too slow for benchmarking
        let max_size = if group_suffix.contains("degree_scaling") { 50_000 } else { 100_000 };
        if (d + 1_usize).pow(v as u32) > max_size {
            continue;
        }

        // Generate group name based on the type of test
        let group_name = if group_suffix.contains("v=") && group_suffix.contains("d=") && !group_suffix.contains("scaling") {
            format!("Multivariate Product ({})", group_suffix.replace("_", ", "))
        } else if group_suffix.contains("degree_scaling") {
            format!("Degree Scaling (v=3, {})", group_suffix.replace("degree_scaling_", "").replace("_", "="))
        } else if group_suffix.contains("variable_scaling") {
            format!("Variable Scaling ({}, d=4)", group_suffix.replace("variable_scaling_", "").replace("_", "="))
        } else {
            format!("Multivariate Product ({})", group_suffix)
        };

        let mut group = c.benchmark_group(group_name);

        // Generate random multilinear polynomials on {0,1}^v
        let n = 1usize << v;

        // Use different seeds for different test types to ensure variety
        let base_seed = 7919u64 + seed_offset * 1000;
        let polynomials: Vec<Vec<Fr>> = (0..d)
            .map(|j| {
                (0..n)
                    .map(|i| {
                        let seed = ((i as u64) * 17 + (j as u64) * 101 + 5 + seed_offset * 100) % base_seed;
                        Fr::from(seed)
                    })
                    .collect()
            })
            .collect();

        // Method 1: Naive approach (extrapolate first, then multiply)
        group.bench_function("naive_extrapolate_then_multiply", |bench| {
            bench.iter(|| {
                black_box(multivariate_product_evaluations_naive::<Fr>(
                    black_box(v),
                    black_box(&polynomials),
                    black_box(d)
                ))
            })
        });

        // Method 2: Our new algorithm with extrapolation technique
        group.bench_function("optimized_with_extrapolation", |bench| {
            bench.iter(|| {
                black_box(multi_product_eval::<Fr>(
                    black_box(v),
                    black_box(&polynomials),
                    black_box(d)
                ))
            })
        });

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_multivariate_product_consolidated
);
criterion_main!(benches);
