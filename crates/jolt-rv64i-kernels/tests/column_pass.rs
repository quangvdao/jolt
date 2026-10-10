#![cfg(feature = "test-utils")]
#![expect(
    clippy::unwrap_used,
    reason = "fixture construction and independent oracle assertions may panic on failure"
)]

#[path = "../benches/support/allocator.rs"]
mod allocator;

use allocator::{AllocationMeasurement, CountingAllocator};
use jolt_field::{Field, F128};
use jolt_rv64i_kernels::column_pass::{column_pass, ColumnPassError};
use jolt_rv64i_kernels::oracle::mle_at;
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::ThreadPoolBuilder;

#[test]
fn all_columns_equal_the_defining_sum_at_seeded_and_boolean_points() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x0c01_1128);
    for log_t in 0..=8 {
        let rows: Vec<[u64; 4]> = (0..1 << log_t)
            .map(|_| std::array::from_fn(|_| rng.next_u64()))
            .collect();
        let tables: Vec<Vec<F128>> = (0..256)
            .map(|y| {
                rows.iter()
                    .map(|row| F128::from_raw(u128::from((row[y / 64] >> (y % 64)) & 1)))
                    .collect()
            })
            .collect();
        let seeded: Vec<_> = (0..log_t).map(|_| F128::random(&mut rng)).collect();
        for point in std::iter::once(seeded).chain((0..1 << log_t).map(|vertex| {
            (0..log_t)
                .map(|bit| F128::from_raw(((vertex >> bit) & 1) as u128))
                .collect()
        })) {
            let columns = column_pass(&rows, &point).unwrap();
            for (y, table) in tables.iter().enumerate() {
                assert_eq!(columns[y], mle_at(table, &point).unwrap());
            }
        }
    }
}

#[test]
fn multichunk_column_sums_match_the_definition_on_one_and_twelve_threads() {
    let log_t = 13;
    let mut rng = ChaCha20Rng::seed_from_u64(0xc011_130c);
    let rows: Vec<[u64; 4]> = (0..1 << log_t)
        .map(|_| std::array::from_fn(|_| rng.next_u64()))
        .collect();
    let point: Vec<_> = (0..log_t).map(|_| F128::random(&mut rng)).collect();
    let expected: [F128; 256] = std::array::from_fn(|y| {
        let table: Vec<_> = rows
            .iter()
            .map(|row| F128::from_raw(u128::from((row[y / 64] >> (y % 64)) & 1)))
            .collect();
        mle_at(&table, &point).unwrap()
    });
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let columns = pool.install(|| column_pass(&rows, &point)).unwrap();
        assert_eq!(columns, expected);
    }
}

#[test]
fn malformed_row_count_is_rejected() {
    for count in [0, 12] {
        assert_eq!(
            column_pass(&vec![[0; 4]; count], &[]),
            Err(ColumnPassError::Rows { rows: count })
        );
    }
}

#[test]
fn malformed_cycle_point_length_is_rejected() {
    assert_eq!(
        column_pass(&[[0; 4]; 16], &[F128::from_raw(0); 5]),
        Err(ColumnPassError::PointLength {
            expected: 4,
            actual: 5,
        })
    );
}

#[test]
fn scratch_is_bounded_and_released_on_return() {
    for (threads, log_t) in [(1, 8), (1, 14), (12, 14)] {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_scoped(
                |thread| thread.run(),
                |pool| {
                    // Warm each worker's work-stealing bookkeeping, including
                    // workers that the four-chunk fixture may otherwise leave idle.
                    let _ = pool.broadcast(|_| {
                        rayon::join(
                            || {
                                let _ = rayon::yield_now();
                            },
                            || (),
                        )
                    });
                    let trace = pool
                        .install(|| SyntheticTrace::new(SynthProfile::Local, log_t, 256, 0xc011))
                        .unwrap();
                    let point = vec![F128::from_raw(0x713); log_t];
                    let _ = pool.install(|| column_pass(trace.rows(), &point).unwrap());
                    // The worker invocation must return before checking release;
                    // Rayon frees completed scheduling jobs after their bodies return.
                    let baseline = CountingAllocator::live_bytes();
                    let measurement = AllocationMeasurement::begin();
                    let columns = pool.install(|| column_pass(trace.rows(), &point).unwrap());
                    let stats = measurement.finish();
                    assert_eq!(columns.len(), 256);
                    assert!(stats.allocs <= 256, "{} allocations", stats.allocs);
                    let allowance = 256 * 16 + threads * 8192 * 16 + (1 << log_t) * 16;
                    assert!(
                        stats.peak_bytes <= allowance,
                        "{} peak bytes",
                        stats.peak_bytes
                    );
                    assert_eq!(stats.final_bytes, 0);
                    assert_eq!(CountingAllocator::live_bytes(), baseline);
                },
            )
            .unwrap();
    }
}
