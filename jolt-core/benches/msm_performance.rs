//! Benchmark for MSM (Multi-Scalar Multiplication) performance
//! Testing u64 and i128 scalar types with varying sparsity levels.

use ark_bn254::{G1Affine, G1Projective};
use ark_ec::scalar_mul::variable_base::{msm_i128, msm_u64};
use ark_ec::CurveGroup;
use ark_ff::UniformRand;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

/// Generate random G1 affine points for bases
fn generate_bases(n: usize, rng: &mut ChaCha20Rng) -> Vec<G1Affine> {
    (0..n)
        .map(|_| G1Projective::rand(rng).into_affine())
        .collect()
}

/// Generate u64 scalars with given sparsity (fraction of zeros)
fn generate_u64_scalars(n: usize, sparsity: f64, rng: &mut ChaCha20Rng) -> Vec<u64> {
    (0..n)
        .map(|_| {
            if rng.gen::<f64>() < sparsity {
                0u64
            } else {
                rng.gen::<u64>()
            }
        })
        .collect()
}

/// Generate i128 scalars with given sparsity (fraction of zeros)
fn generate_i128_scalars(n: usize, sparsity: f64, rng: &mut ChaCha20Rng) -> Vec<i128> {
    (0..n)
        .map(|_| {
            if rng.gen::<f64>() < sparsity {
                0i128
            } else {
                rng.gen::<i128>()
            }
        })
        .collect()
}

/// Benchmark MSM with u64 scalars
fn bench_msm_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm_u64");
    group.sample_size(10);

    let sparsities = [
        (0.0, "dense"),
        (0.1, "10pct_sparse"),
        (0.5, "50pct_sparse"),
        (0.8, "80pct_sparse"),
    ];
    let sizes = [14, 16, 18, 20]; // 2^14 = 16384, 2^16 = 65536, 2^18 = 262144, 2^20 = 1048576

    for &(sparsity, sparsity_name) in &sparsities {
        for &exp in &sizes {
            let n = 1usize << exp;
            let mut rng = ChaCha20Rng::seed_from_u64(12345);

            let bases = generate_bases(n, &mut rng);
            let scalars = generate_u64_scalars(n, sparsity, &mut rng);

            let id = format!("2^{exp}_{sparsity_name}");
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new("msm", &id), &n, |b, _| {
                b.iter(|| msm_u64::<G1Projective>(&bases, &scalars, true))
            });
        }
    }

    group.finish();
}

/// Benchmark MSM with i128 scalars
fn bench_msm_i128(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm_i128");
    group.sample_size(10);

    let sparsities = [
        (0.0, "dense"),
        (0.1, "10pct_sparse"),
        (0.5, "50pct_sparse"),
        (0.8, "80pct_sparse"),
    ];
    let sizes = [14, 16, 18, 20];

    for &(sparsity, sparsity_name) in &sparsities {
        for &exp in &sizes {
            let n = 1usize << exp;
            let mut rng = ChaCha20Rng::seed_from_u64(12345);

            let bases = generate_bases(n, &mut rng);
            let scalars = generate_i128_scalars(n, sparsity, &mut rng);

            let id = format!("2^{exp}_{sparsity_name}");
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new("msm", &id), &n, |b, _| {
                b.iter(|| msm_i128::<G1Projective>(&bases, &scalars, true))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_msm_u64, bench_msm_i128);
criterion_main!(benches);
