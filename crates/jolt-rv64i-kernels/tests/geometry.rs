#![cfg(feature = "test-utils")]
#![expect(clippy::unwrap_used, reason = "valid geometry fixtures must construct")]

use jolt_rv64i_kernels::par::{CycleChunks, ParError};
use rayon::ThreadPoolBuilder;

#[test]
fn chunks_cover_whole_equality_blocks_independently_of_the_pool() {
    let one = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let twelve = ThreadPoolBuilder::new().num_threads(12).build().unwrap();
    for log_t in 0..=24 {
        for round in 0..=log_t {
            let chunks = CycleChunks::new(log_t, round).unwrap();
            let a: Vec<_> = one.install(|| chunks.ranges().collect());
            let b: Vec<_> = twelve.install(|| chunks.ranges().collect());
            assert_eq!(a, b);
            assert_eq!(a.first().unwrap().start, 0);
            assert_eq!(a.last().unwrap().end, 1 << (log_t - round));
            assert_eq!(chunks.low_bits() + chunks.high_bits(), log_t - round);
            assert_eq!(chunks.len(), 1 << (log_t - round));
            for range in &a {
                assert_eq!(range.start % chunks.block_len(), 0);
                assert_eq!(range.end % chunks.block_len(), 0);
                assert_eq!(range.len(), chunks.chunk_len());
            }
            for pair in a.windows(2) {
                assert_eq!(pair[0].end, pair[1].start);
            }
            let point: Vec<_> = (0..log_t - round).collect();
            let (low, high) = chunks.split_point(&point).unwrap();
            assert_eq!(low.len(), chunks.low_bits());
            assert_eq!(high.len(), chunks.high_bits());
            assert_eq!(low.iter().chain(high).copied().collect::<Vec<_>>(), point);
        }
    }
}

#[test]
fn malformed_geometry_returns_typed_errors() {
    assert!(matches!(
        CycleChunks::new(usize::MAX, 0),
        Err(ParError::LogSize {
            log_t: usize::MAX,
            ..
        })
    ));
    assert_eq!(
        CycleChunks::new(5, 6),
        Err(ParError::Round { log_t: 5, round: 6 })
    );
    assert_eq!(
        CycleChunks::new(5, 2).unwrap().split_point::<u8>(&[]),
        Err(ParError::PointLength {
            expected: 3,
            actual: 0
        })
    );
}
