use ark_bn254::Fr;
use ark_std::{test_rng, UniformRand};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_core::field::JoltField;
use rand::Rng;
use rayon::prelude::*;

/*
fn benchmark_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_operations");
    let mut rng = rand::thread_rng();
    let mut field_rng = test_rng();

    let sizes = [100_000, 1_000_000];

    for size in sizes {
        // Generate test data
        let arr_a_u64: Vec<u64> = (0..size).map(|_| rng.gen()).collect();
        let arr_b_u64: Vec<u64> = (0..size).map(|_| rng.gen()).collect();
        let arr_a_signed_u64: Vec<SignedU64> = (0..size)
            .map(|_| SignedU64::new(rng.gen(), rng.gen_bool(0.5)))
            .collect();
        let arr_b_signed_u64: Vec<SignedU64> = (0..size)
            .map(|_| SignedU64::new(rng.gen(), rng.gen_bool(0.5)))
            .collect();
        let arr_a_fr: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut field_rng)).collect();
        let arr_b_fr: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut field_rng)).collect();

        // Sequential u64 multiplication
        group.bench_with_input(BenchmarkId::new("Sequential::u64", size), &size, |b, _| {
            b.iter(|| {
                arr_a_u64
                    .iter()
                    .zip(arr_b_u64.iter())
                    .map(|(&x, &y)| x * y)
                    .collect::<Vec<_>>()
            })
        });

        // Parallel u64 multiplication
        group.bench_with_input(BenchmarkId::new("Parallel::u64", size), &size, |b, _| {
            b.iter(|| {
                arr_a_u64
                    .par_iter()
                    .zip(arr_b_u64.par_iter())
                    .map(|(&x, &y)| x * y)
                    .collect::<Vec<_>>()
            })
        });

        // Sequential signed u64 multiplication
        group.bench_with_input(
            BenchmarkId::new("Sequential::signed_u64", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_signed_u64
                        .iter()
                        .zip(arr_b_signed_u64.iter())
                        .map(|(&x, &y)| x * y)
                        .collect::<Vec<_>>()
                })
            },
        );

        // Parallel signed u64 multiplication
        group.bench_with_input(
            BenchmarkId::new("Parallel::signed_u64", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_signed_u64
                        .par_iter()
                        .zip(arr_b_signed_u64.par_iter())
                        .map(|(&x, &y)| x * y)
                        .collect::<Vec<_>>()
                })
            },
        );
        /*

        // Sequential field addition
        group.bench_with_input(
            BenchmarkId::new("Sequential::field_add", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_fr
                        .iter()
                        .zip(arr_b_fr.iter())
                        .map(|(x, y)| x + y)
                        .collect::<Vec<_>>()
                })
            },
        );

        // Parallel field addition
        group.bench_with_input(
            BenchmarkId::new("Parallel::field_add", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_fr
                        .par_iter()
                        .zip(arr_b_fr.par_iter())
                        .map(|(x, y)| x + y)
                        .collect::<Vec<_>>()
                })
            },
        );

        // Sequential field multiplication
        group.bench_with_input(
            BenchmarkId::new("Sequential::field_mul", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_fr
                        .iter()
                        .zip(arr_b_fr.iter())
                        .map(|(&x, &y)| x * y)
                        .collect::<Vec<_>>()
                })
            },
        );

        // Parallel field multiplication
        group.bench_with_input(
            BenchmarkId::new("Parallel::field_mul", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_fr
                        .par_iter()
                        .zip(arr_b_fr.par_iter())
                        .map(|(&x, &y)| x * y)
                        .collect::<Vec<_>>()
                })
            },
        );

        // Sequential field squaring
        group.bench_with_input(
            BenchmarkId::new("Sequential::field_square", size),
            &size,
            |b, _| b.iter(|| arr_a_fr.iter().map(|x| x.square()).collect::<Vec<_>>()),
        );

        // Parallel field squaring
        group.bench_with_input(
            BenchmarkId::new("Parallel::field_square", size),
            &size,
            |b, _| b.iter(|| arr_a_fr.par_iter().map(|x| x.square()).collect::<Vec<_>>()),
        );
         */

        // Sequential field * u64
        group.bench_with_input(
            BenchmarkId::new("Sequential::field_mul_u64", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_fr
                        .iter()
                        .zip(arr_b_u64.iter())
                        .map(|(&x, &y)| x.mul_u64(y))
                        .collect::<Vec<_>>()
                })
            },
        );

        // Parallel field * u64
        group.bench_with_input(
            BenchmarkId::new("Parallel::field_mul_u64", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_fr
                        .par_iter()
                        .zip(arr_b_u64.par_iter())
                        .map(|(&x, &y)| x.mul_u64(y))
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}
 */

fn benchmark_bulk_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_multiplication");
    let mut rng = rand::thread_rng();
    let mut field_rng = test_rng();

    let sizes = [1000000];

    for size in sizes {
        // bitwise operations
        let arr_a_u64: Vec<bool> = (0..size).map(|_| rng.gen()).collect();
        let arr_b_u64: Vec<bool> = (0..size).map(|_| rng.gen()).collect();
        group.bench_with_input(
            BenchmarkId::new("BulkAnd::bool(64 bit arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_u64
                        .iter()
                        .zip(arr_b_u64.iter())
                        .map(|(&x, &y)| x && y)
                        .collect::<Vec<_>>()
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("BulkXor::bool(64 bit arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_u64
                        .iter()
                        .zip(arr_b_u64.iter())
                        .map(|(&x, &y)| x ^ y)
                        .collect::<Vec<_>>()
                })
            },
        );

        // u64 bulk operations
        let arr_a_u64: Vec<u64> = (0..size).map(|_| rng.gen()).collect();
        let arr_b_u64: Vec<u64> = (0..size).map(|_| rng.gen()).collect();
        // Bulk addition
        group.bench_with_input(
            BenchmarkId::new("BulkAdd::u64(64 bit arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_u64
                        .iter()
                        .zip(arr_b_u64.iter())
                        .map(|(&x, &y)| criterion::black_box(x) + criterion::black_box(y))
                        .collect::<Vec<_>>()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("BulkMultiply::u64(64 bit arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_u64
                        .iter()
                        .zip(arr_b_u64.iter())
                        .map(|(&x, &y)| criterion::black_box(x) * criterion::black_box(y))
                        .collect::<Vec<_>>()
                })
            },
        );

        // i64 bulk operations
        let arr_a_i64: Vec<i64> = (0..size).map(|_| rng.gen()).collect();
        let arr_b_i64: Vec<i64> = (0..size).map(|_| rng.gen()).collect();
        group.bench_with_input(
            BenchmarkId::new("BulkMultiply::i64(64 bit arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_i64
                        .iter()
                        .zip(arr_b_i64.iter())
                        .map(|(&x, &y)| criterion::black_box(x) * criterion::black_box(y))
                        .collect::<Vec<_>>()
                })
            },
        );

        // // signed u64 bulk multiplication
        // let arr_a_signed_u64: Vec<SignedU64> = (0..size)
        //     .map(|_| SignedU64::new(rng.gen(), rng.gen_bool(0.5)))
        //     .collect();
        // let arr_b_signed_u64: Vec<SignedU64> = (0..size)
        //     .map(|_| SignedU64::new(rng.gen(), rng.gen_bool(0.5)))
        //     .collect();
        // group.bench_with_input(
        //     BenchmarkId::new("BulkMultiply::signed_u64(64 bit arrays)", size),
        //     &size,
        //     |b, _| {
        //         b.iter(|| {
        //             arr_a_signed_u64
        //                 .iter()
        //                 .zip(arr_b_signed_u64.iter())
        //                 .map(|(&x, &y)| criterion::black_box(x) * criterion::black_box(y))
        //                 .collect::<Vec<_>>()
        //         })
        //     },
        // );

        // u128 bulk operations
        let arr_a: Vec<u128> = (0..size).map(|_| rng.gen()).collect();
        let arr_b: Vec<u128> = (0..size).map(|_| rng.gen()).collect();
        group.bench_with_input(
            BenchmarkId::new("BulkMultiply::u128(128 bit arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a
                        .iter()
                        .zip(arr_b.iter())
                        .map(|(&x, &y)| criterion::black_box(x) * criterion::black_box(y))
                        .collect::<Vec<_>>()
                })
            },
        );

        // u128 comparison with (u64::MAX as u128)
        let arr_a_u128: Vec<u128> = (0..size).map(|_| rng.gen()).collect();
        group.bench_with_input(
            BenchmarkId::new("BulkCompareToU64Max::u128(128 bit arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_u128
                        .iter()
                        .map(|&x| criterion::black_box(x) < u64::MAX as u128)
                        .collect::<Vec<_>>()
                })
            },
        );

        // i128 determine if positive
        let arr_a_i128: Vec<i128> = (0..size).map(|_| rng.gen()).collect();
        group.bench_with_input(
            BenchmarkId::new("BulkIsPositive::i128(128 bit arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a_i128
                        .iter()
                        .map(|&x| criterion::black_box(x) > 0)
                        .collect::<Vec<_>>()
                })
            },
        );

        // // i128 bulk operations
        // let arr_a_i128: Vec<i128> = (0..size).map(|_| rng.gen()).collect();
        // let arr_b_i128: Vec<i128> = (0..size).map(|_| rng.gen()).collect();
        // group.bench_with_input(
        //     BenchmarkId::new("BulkMultiply::i128(128 bit arrays)", size),
        //     &size,
        //     |b, _| {
        //         b.iter(|| {
        //             arr_a_i128
        //                 .iter()
        //                 .zip(arr_b_i128.iter())
        //                 .map(|(&x, &y)| criterion::black_box(x) * criterion::black_box(y))
        //                 .collect::<Vec<_>>()
        //         })
        //     },
        // );

        // Field bulk operations
        let arr_a: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut field_rng)).collect();
        let arr_b: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut field_rng)).collect();
        group.bench_with_input(
            BenchmarkId::new("BulkAdd::Fr(BN254 field element arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a
                        .iter()
                        .zip(arr_b.iter())
                        .map(|(&x, &y)| criterion::black_box(x) + criterion::black_box(y))
                        .collect::<Vec<_>>()
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("BulkMultiply::Fr(BN254 field element arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a
                        .iter()
                        .zip(arr_b.iter())
                        .map(|(&x, &y)| criterion::black_box(x) * criterion::black_box(y))
                        .collect::<Vec<_>>()
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("BulkSquare::Fr(BN254 field element arrays)", size),
            &size,
            |b, _| b.iter(|| arr_a.iter().map(|x| x.square()).collect::<Vec<_>>()),
        );
        // group.bench_with_input(
        //     BenchmarkId::new("BulkInverse::Fr(BN254 field element arrays)", size),
        //     &size,
        //     |b, _| b.iter(|| arr_a.iter().map(|x| x.inverse()).collect::<Vec<_>>()),
        // );
        group.bench_with_input(
            BenchmarkId::new("BulkMulU64::Fr(BN254 field element arrays)", size),
            &size,
            |b, _| {
                b.iter(|| {
                    arr_a
                        .iter()
                        .zip(arr_b_u64.iter())
                        .map(|(x, y)| x.mul_u64(criterion::black_box(*y)))
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

// criterion_group!(
//     name = benches;
//     config = Criterion::default()
//         .sample_size(10)
//         .warm_up_time(std::time::Duration::from_millis(500));
//     targets = benchmark_parallel_operations
// );
// criterion_main!(benches);

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .sample_size(20)
        .warm_up_time(std::time::Duration::from_secs(2));

    benchmark_bulk_multiplication(&mut criterion);

    criterion.final_summary();
}
