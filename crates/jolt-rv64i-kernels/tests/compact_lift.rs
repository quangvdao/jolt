#![cfg(feature = "test-utils")]
#![expect(
    clippy::unwrap_used,
    reason = "test fixtures and assertions may panic on failure"
)]

#[path = "../src/packed/lift.rs"]
#[expect(
    dead_code,
    reason = "the source inclusion exposes crate-private compact lifts; public word and nibble lifts are tested separately"
)]
mod lift;

use jolt_field::{Field, F128};
use lift::{compact_table, CompactLift, LiftError};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn weighted_bits(weights: &[F128], word: u64) -> F128 {
    weights
        .iter()
        .enumerate()
        .filter(|(bit, _)| word & (1_u64 << bit) != 0)
        .map(|(_, &weight)| weight)
        .sum()
}

fn check_lift<const N: usize, const TABLES: usize>(
    count: usize,
    bits: usize,
    arena: &mut Vec<F128>,
    rng: &mut ChaCha20Rng,
) {
    let weights: Vec<_> = (0..count).map(|_| F128::random(rng)).collect();
    let start = arena.len();
    let lift = CompactLift::new(&weights, bits, arena).unwrap();
    assert_eq!(arena.len() - start, N * TABLES);
    let view = lift.view::<N, TABLES>(arena).unwrap();
    for word in [0, u64::MAX]
        .into_iter()
        .chain((0..64).map(|bit| 1_u64 << bit))
        .chain((0..1000).map(|_| rng.next_u64()))
    {
        assert_eq!(view.lift(word), weighted_bits(&weights, word));
    }
}

#[test]
fn compact_lifts_equal_the_weighted_bit_definition_in_every_layout() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x0c0a_1ac7);
    let mut arena = vec![F128::from_raw(0x53); 7];
    check_lift::<256, 4>(32, 8, &mut arena, &mut rng);
    check_lift::<256, 2>(16, 8, &mut arena, &mut rng);
    check_lift::<16, 4>(16, 4, &mut arena, &mut rng);
    check_lift::<16, 2>(8, 4, &mut arena, &mut rng);
    check_lift::<16, 1>(4, 4, &mut arena, &mut rng);
    check_lift::<4, 1>(2, 2, &mut arena, &mut rng);
    check_lift::<2, 1>(1, 1, &mut arena, &mut rng);
    check_lift::<16, 2>(5, 4, &mut arena, &mut rng);
    check_lift::<256, 2>(9, 8, &mut arena, &mut rng);
    check_lift::<4, 0>(0, 2, &mut arena, &mut rng);
}

#[test]
fn compact_lifts_keep_independent_offsets_in_a_shared_arena() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x000a_2e0a);
    let a: Vec<_> = (0..9).map(|_| F128::random(&mut rng)).collect();
    let b: Vec<_> = (0..5).map(|_| F128::random(&mut rng)).collect();
    let mut arena = vec![F128::from_raw(0x71); 3];
    let first = CompactLift::new(&a, 8, &mut arena).unwrap();
    let second = CompactLift::new(&b, 4, &mut arena).unwrap();
    let first = first.view::<256, 2>(&arena).unwrap();
    let second = second.view::<16, 2>(&arena).unwrap();
    for word in [0, u64::MAX]
        .into_iter()
        .chain((0..1000).map(|_| rng.next_u64()))
    {
        assert_eq!(first.lift(word), weighted_bits(&a, word));
        assert_eq!(second.lift(word), weighted_bits(&b, word));
    }
}

fn check_table<const N: usize>(weights: &[F128]) {
    let table = compact_table::<N>(weights).unwrap();
    for (word, &value) in table.iter().enumerate() {
        assert_eq!(value, weighted_bits(weights, word as u64));
    }
}

#[test]
fn compact_tables_equal_the_weighted_bit_definition_including_padding() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x0007_ab1e);
    let weights: Vec<_> = (0..8).map(|_| F128::random(&mut rng)).collect();
    for count in 0..=1 {
        check_table::<2>(&weights[..count]);
    }
    for count in 0..=2 {
        check_table::<4>(&weights[..count]);
    }
    for count in 0..=4 {
        check_table::<16>(&weights[..count]);
    }
    for count in 0..=8 {
        check_table::<256>(&weights[..count]);
    }
}

#[test]
fn compact_lifts_reject_malformed_weights_widths_views_and_arenas() {
    let one = F128::from_raw(1);
    let mut arena = vec![one; 3];
    assert!(matches!(
        CompactLift::new(&[one; 65], 8, &mut arena),
        Err(LiftError::WeightCount { count: 65 })
    ));
    assert!(matches!(
        CompactLift::new(&[one; 3], 3, &mut arena),
        Err(LiftError::Width { bits: 3 })
    ));
    assert_eq!(arena, [one; 3]);
    let lift = CompactLift::new(&[one; 9], 8, &mut arena).unwrap();
    assert!(matches!(
        lift.view::<16, 32>(&arena),
        Err(LiftError::Layout)
    ));
    assert!(matches!(
        lift.view::<256, 1>(&arena),
        Err(LiftError::Layout)
    ));
    assert!(matches!(
        lift.view::<256, 2>(&arena[..arena.len() - 1]),
        Err(LiftError::Layout)
    ));
    assert!(matches!(lift.view::<256, 2>(&[]), Err(LiftError::Layout)));
    assert!(matches!(
        lift.view::<256, { usize::MAX }>(&arena),
        Err(LiftError::Layout)
    ));
    assert!(matches!(
        compact_table::<8>(&[one; 3]),
        Err(LiftError::Width { bits: 3 })
    ));
    assert!(matches!(
        compact_table::<16>(&[one; 5]),
        Err(LiftError::TableWeightCount {
            capacity: 4,
            count: 5
        })
    ));
}
