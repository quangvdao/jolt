#![cfg(feature = "test-utils")]

#[expect(
    clippy::unwrap_used,
    reason = "test setup and contract failures must fail the test"
)]
mod tests {
    use jolt_field::F128;
    use jolt_rv64i_kernels::packed::bits::{gather, moebius, BitsError};
    use jolt_rv64i_kernels::packed::buckets::{
        BucketError, ByteBuckets, DigitHistogram, NibbleBuckets,
    };
    use jolt_rv64i_kernels::packed::lift::{LiftError, NibbleLift, WordLift};
    use jolt_rv64i_kernels::packed::pool::{PoolError, ScratchPool};
    use jolt_rv64i_kernels::packed::scatter::{ScatterError, ScatterPlan};
    use jolt_rv64i_kernels::source::{CycleSource, ValidatedTrace};
    use rand_chacha::rand_core::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;

    use std::sync::Arc;

    fn field(rng: &mut ChaCha20Rng) -> F128 {
        F128::from_raw(u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64))
    }

    #[test]
    fn lifts_equal_weighted_bits() {
        let mut rng = ChaCha20Rng::seed_from_u64(73);
        let weights: [F128; 64] = std::array::from_fn(|_| field(&mut rng));
        let lift = WordLift::new(&weights).unwrap();
        let words: Vec<_> = (0..64)
            .map(|bit| 1_u64 << bit)
            .chain([0, u64::MAX])
            .chain((0..10_000).map(|_| rng.next_u64()))
            .collect();
        for word in words.iter().copied() {
            let expected = weights
                .iter()
                .enumerate()
                .filter(|(bit, _)| word & (1 << bit) != 0)
                .fold(F128::from_raw(0), |sum, (_, &weight)| sum + weight);
            assert_eq!(lift.lift(word), expected);
        }
        for significant in [0, 1, 2, 4, 8, 16, 32, 64] {
            let lift = NibbleLift::new(&weights[..significant]).unwrap();
            for &word in &words {
                let expected = weights[..significant]
                    .iter()
                    .enumerate()
                    .filter(|(bit, _)| word & (1 << bit) != 0)
                    .fold(F128::from_raw(0), |sum, (_, &weight)| sum + weight);
                assert_eq!(lift.lift(word), expected);
            }
        }
    }

    #[test]
    fn buckets_and_histograms_equal_direct_sums_and_merged_halves() {
        let mut rng = ChaCha20Rng::seed_from_u64(74);
        let events: Vec<_> = (0..10_000)
            .map(|_| {
                (
                    rng.next_u32() as usize % 32,
                    rng.next_u32() as usize & 255,
                    field(&mut rng),
                )
            })
            .collect();
        for width in [4, 8] {
            let bound = 1 << width;
            let len = 32 * bound;
            let mut expected_bits = vec![[F128::from_raw(0); 8]; 32];
            let mut expected_totals = [F128::from_raw(0); 32];
            let mut expected_digits = vec![F128::from_raw(0); bound];
            for &(position, value, e) in &events {
                let value = value & (bound - 1);
                expected_totals[position] += e;
                expected_digits[value] += e;
                for (bit, sum) in expected_bits[position].iter_mut().enumerate().take(width) {
                    if value & (1 << bit) != 0 {
                        *sum += e;
                    }
                }
            }
            let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
            pool.install(|| {
                let scratch = ScratchPool::new(len).unwrap();
                let mut halves = [scratch.take().unwrap(), scratch.take().unwrap()];
                for (half, sequence) in halves.iter_mut().zip(events.chunks(5000)) {
                    if width == 4 {
                        let mut buckets = NibbleBuckets::new(half).unwrap();
                        for &(position, value, e) in sequence {
                            buckets.xor(position, value & 15, e).unwrap();
                        }
                    } else {
                        let mut buckets = ByteBuckets::new(half).unwrap();
                        for &(position, value, e) in sequence {
                            buckets.xor(position, value, e).unwrap();
                        }
                    }
                }
                drop(halves);
                let mut merged = scratch.merge().unwrap();
                if width == 4 {
                    let buckets = NibbleBuckets::new(&mut merged).unwrap();
                    for position in 0..32 {
                        assert_eq!(
                            buckets.bits(position).unwrap().as_slice(),
                            &expected_bits[position][..4]
                        );
                        assert_eq!(buckets.total(position).unwrap(), expected_totals[position]);
                    }
                } else {
                    let buckets = ByteBuckets::new(&mut merged).unwrap();
                    for position in 0..32 {
                        assert_eq!(buckets.bits(position).unwrap(), expected_bits[position]);
                        assert_eq!(buckets.total(position).unwrap(), expected_totals[position]);
                    }
                }
                let hist_pool = ScratchPool::new(bound).unwrap();
                let mut halves = [hist_pool.take().unwrap(), hist_pool.take().unwrap()];
                for (half, sequence) in halves.iter_mut().zip(events.chunks(5000)) {
                    let mut hist = DigitHistogram::new(half, width).unwrap();
                    for &(_, value, e) in sequence {
                        hist.xor(value & (bound - 1), e).unwrap();
                    }
                }
                drop(halves);
                let mut merged = hist_pool.merge().unwrap();
                assert_eq!(
                    DigitHistogram::new(&mut merged, width).unwrap().sums(),
                    expected_digits
                );
            });
        }
    }

    #[test]
    fn transforms_equal_subset_sum_and_compaction_definitions() {
        let mut rng = ChaCha20Rng::seed_from_u64(75);
        for word in [0, u64::MAX]
            .into_iter()
            .chain((0..100).map(|_| rng.next_u64()))
        {
            for k in 0..=6 {
                let mask = (1_usize << k) - 1;
                let mut expected = 0;
                for position in 0..64 {
                    let low = position & mask;
                    for subset in 0..=low {
                        if subset & !low == 0 {
                            expected ^= ((word >> ((position & !mask) | subset)) & 1) << position;
                        }
                    }
                }
                let transformed = moebius(word, k).unwrap();
                assert_eq!(transformed, expected);
                assert_eq!(moebius(transformed, k).unwrap(), word);
            }
            for m in 0..=6 {
                let stride = 1 << m;
                let expected = (0..64 / stride)
                    .fold(0, |sum, bit| sum | (((word >> (bit * stride)) & 1) << bit));
                assert_eq!(gather(word, m).unwrap(), expected);
            }
        }
    }

    #[test]
    fn scratch_chunks_are_bounded_and_second_pass_has_no_first_pass_contribution() {
        let pool = ThreadPoolBuilder::new().num_threads(12).build().unwrap();
        pool.install(|| {
            let scratch = ScratchPool::new(37).unwrap();
            for pass in 0..2 {
                let mut expected = [F128::from_raw(0); 37];
                for chunk in 0..64 {
                    for (entry, sum) in expected.iter_mut().enumerate() {
                        *sum += F128::from_raw((pass * 10_000 + chunk * 37 + entry) as u128);
                    }
                }
                (0..64).into_par_iter().for_each(|chunk| {
                    let mut guard = scratch.take().unwrap();
                    for (entry, sum) in guard.iter_mut().enumerate() {
                        *sum += F128::from_raw((pass * 10_000 + chunk * 37 + entry) as u128);
                    }
                    assert!(scratch.allocated_arrays().unwrap() <= 12);
                });
                assert_eq!(scratch.merge().unwrap(), expected);
                assert_eq!(scratch.allocated_arrays().unwrap(), 0);
            }
        });
    }

    #[test]
    fn scratch_exhaustion_and_lent_merge_are_typed_and_preserve_arrays() {
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            let scratch = ScratchPool::new(4).unwrap();
            let mut first = scratch.take().unwrap();
            first[0] = F128::from_raw(9);
            let second = scratch.take().unwrap();
            assert!(matches!(scratch.take(), Err(PoolError::Exhausted)));
            assert_eq!(scratch.allocated_arrays().unwrap(), 2);
            assert!(matches!(
                scratch.merge(),
                Err(PoolError::MergeWhileLent { lent: 2 })
            ));
            drop(second);
            drop(first);
            assert_eq!(
                scratch.merge().unwrap(),
                [
                    F128::from_raw(9),
                    F128::from_raw(0),
                    F128::from_raw(0),
                    F128::from_raw(0)
                ]
            );
            let guard = scratch.take().unwrap();
            assert!(guard.iter().all(|&e| e == F128::from_raw(0)));
        });
    }

    struct Rows {
        indices: Vec<usize>,
        rows: usize,
    }
    impl CycleSource for Rows {
        fn cycles(&self) -> usize {
            self.indices.len()
        }
        fn trace_words(&self) -> usize {
            0
        }
        fn trace_word(&self, _: usize, _: usize) -> u64 {
            0
        }
        fn bytecode_rows(&self) -> usize {
            self.rows
        }
        fn bytecode_words(&self) -> usize {
            0
        }
        fn bytecode_word(&self, _: usize, _: usize) -> u64 {
            0
        }
        fn bytecode_index(&self, cycle: usize) -> usize {
            self.indices.get(cycle).copied().unwrap_or(0)
        }
        fn digit_columns(&self) -> usize {
            0
        }
        fn bits(&self, _: usize) -> usize {
            0
        }
        fn by_row(&self, _: usize) -> bool {
            false
        }
        fn digit(&self, _: usize, _: usize) -> Option<usize> {
            None
        }
        fn row_digit(&self, _: usize, _: usize) -> Option<usize> {
            None
        }
    }

    #[test]
    fn scatter_equals_cycle_order_summation_in_small_and_large_domains() {
        for (rows, visited, cycles) in [
            (256, 16, 8192),
            (64, 64, 8192),
            (1, 1, 1),
            (1 << 20, 1 << 20, 1 << 20),
        ] {
            let mut indices: Vec<_> = (0..cycles)
                .map(|cycle| (cycle * 17) & (visited - 1))
                .collect();
            if rows == 1 << 20 {
                let mut rng = ChaCha20Rng::seed_from_u64(76);
                for end in (1..indices.len()).rev() {
                    let index = rng.next_u32() as usize % (end + 1);
                    indices.swap(end, index);
                }
            }
            let source = Arc::new(Rows { indices, rows });
            let weights: Vec<_> = (0..cycles)
                .map(|cycle| F128::from_raw((cycle as u128) * 31 + 1))
                .collect();
            let mut expected = vec![F128::from_raw(0); rows];
            for (cycle, &weight) in weights.iter().enumerate() {
                expected[source.bytecode_index(cycle)] += weight;
            }
            let validated = Arc::new(ValidatedTrace::new(source).unwrap());
            let mut previous = None;
            for threads in [1, 12] {
                let pool = ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let result = pool.install(|| {
                    ScatterPlan::new(Arc::clone(&validated))
                        .unwrap()
                        .scatter(|cycle| weights[cycle])
                        .unwrap()
                });
                assert_eq!(result, expected);
                if let Some(previous) = &previous {
                    assert_eq!(&result, previous);
                }
                previous = Some(result);
            }
        }
    }

    #[test]
    fn malformed_transform_and_bucket_inputs_return_named_errors() {
        assert_eq!(moebius(0, 7), Err(BitsError::MoebiusBits { k: 7 }));
        assert_eq!(gather(0, 7), Err(BitsError::GatherBits { m: 7 }));
        assert!(matches!(
            NibbleLift::new(&[F128::from_raw(0); 65]),
            Err(LiftError::WeightCount { count: 65 })
        ));
        assert!(matches!(
            NibbleBuckets::new(&mut [F128::from_raw(0); 17]),
            Err(BucketError::Length {
                len: 17,
                entries: 16
            })
        ));
        let mut storage = [F128::from_raw(0); 16];
        let mut buckets = NibbleBuckets::new(&mut storage).unwrap();
        assert_eq!(
            buckets.xor(1, 0, F128::from_raw(1)),
            Err(BucketError::Position {
                position: 1,
                positions: 1
            })
        );
        assert_eq!(
            buckets.xor(0, 16, F128::from_raw(1)),
            Err(BucketError::Value {
                value: 16,
                bound: 16
            })
        );
        assert_eq!(buckets.bit(0, 4), Err(BucketError::Bit { bit: 4, bits: 4 }));
        assert!(matches!(
            DigitHistogram::new(&mut storage, usize::BITS as usize),
            Err(BucketError::Width { .. })
        ));
        assert!(matches!(
            DigitHistogram::new(&mut storage, 3),
            Err(BucketError::HistogramLength { len: 16, bound: 8 })
        ));
        assert!(matches!(
            ScratchPool::new(usize::MAX),
            Err(PoolError::Length { len: usize::MAX })
        ));
    }

    #[test]
    fn odd_tail_merge_and_zero_fill_equal_direct_array_sums() {
        for workers in [1, 3, 5, 12] {
            ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    let pool = ScratchPool::new(37).unwrap();
                    let mut guards: Vec<_> = (0..workers).map(|_| pool.take().unwrap()).collect();
                    let mut expected = [F128::from_raw(0); 37];
                    for (worker, guard) in guards.iter_mut().enumerate() {
                        for (entry, sum) in guard.iter_mut().enumerate() {
                            *sum = F128::from_raw((worker * 37 + entry) as u128);
                            expected[entry] += *sum;
                        }
                    }
                    drop(guards);
                    assert_eq!(pool.merge().unwrap(), expected);
                    let mut guards: Vec<_> = (0..workers).map(|_| pool.take().unwrap()).collect();
                    for guard in &mut guards {
                        guard.fill(F128::from_raw(1));
                    }
                    drop(guards);
                    pool.zero().unwrap();
                    assert_eq!(pool.merge().unwrap(), [F128::from_raw(0); 37]);
                });
        }
    }

    #[test]
    fn scatter_rejects_buffer_lengths_before_mutation() {
        let source = Arc::new(
            ValidatedTrace::new(Arc::new(Rows {
                indices: vec![0],
                rows: 1,
            }))
            .unwrap(),
        );
        let plan = ScatterPlan::new(source).unwrap();
        let mut rows = [77];
        let mut weights = [F128::from_raw(13)];
        let mut output = [];
        let mut cursors = vec![0; plan.cursor_len()];
        assert_eq!(
            plan.scatter_into(
                |_| F128::from_raw(1),
                &mut rows,
                &mut weights,
                &mut output,
                &mut cursors
            ),
            Err(ScatterError::BufferLength {
                buffer: "output",
                expected: 1,
                actual: 0
            })
        );
        assert_eq!(rows, [77]);
        assert_eq!(weights, [F128::from_raw(13)]);
    }
}
