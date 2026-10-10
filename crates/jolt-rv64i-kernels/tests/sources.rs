#![cfg(feature = "test-utils")]
#![expect(
    clippy::unwrap_used,
    reason = "test setup failures are assertion failures"
)]

use jolt_rv64i_kernels::source::{CycleSource, DigitColumns, LaneSource, SourceError};
use jolt_rv64i_kernels::synth::{SynthError, SynthProfile, SyntheticTrace};
use rayon::ThreadPoolBuilder;
use std::collections::BTreeSet;
use std::sync::Arc;

#[test]
fn profiles_depend_only_on_seed_and_dimensions() {
    let one = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let twelve = ThreadPoolBuilder::new().num_threads(12).build().unwrap();
    for profile in [
        SynthProfile::Local,
        SynthProfile::AllRows,
        SynthProfile::UniformDigits,
    ] {
        let a = one.install(|| SyntheticTrace::new(profile, 14, 1 << 12, 91).unwrap());
        let b = one.install(|| SyntheticTrace::new(profile, 14, 1 << 12, 91).unwrap());
        let c = twelve.install(|| SyntheticTrace::new(profile, 14, 1 << 12, 91).unwrap());
        let different = one.install(|| SyntheticTrace::new(profile, 14, 1 << 12, 92).unwrap());
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_ne!(a, different);
    }
}

#[test]
fn lanes_tail_and_committed_indicators_match_their_definitions() {
    for profile in [
        SynthProfile::Local,
        SynthProfile::AllRows,
        SynthProfile::UniformDigits,
    ] {
        let trace = SyntheticTrace::new(profile, 12, 1 << 12, 53).unwrap();
        for (cycle, row) in trace.rows().iter().enumerate() {
            for [a, b, c] in trace.lanes(cycle) {
                assert_eq!(c, a & b);
            }
            let tail = trace.tail(cycle);
            assert_eq!((tail >> 4) & 3, (tail & 3) & ((tail >> 2) & 3));
            assert_eq!(trace.trace_word(5, cycle), row[0]);
            for column in 0..trace.digit_columns() {
                let digit = trace.digit(column, cycle);
                assert!(digit.is_none_or(|d| d < 1 << trace.bits(column)));
                if trace.by_row(column) {
                    assert_eq!(digit, trace.row_digit(column, trace.bytecode_index(cycle)));
                }
                if column < 12 {
                    let start = if column < 10 {
                        64 + 15 * column
                    } else {
                        214 + 7 * (column - 10)
                    };
                    let mut indicators = 0;
                    for d in 1..1 << trace.bits(column) {
                        let bit = start + d - 1;
                        let present = (row[bit / 64] >> (bit % 64)) & 1;
                        assert_eq!(present, u64::from(digit == Some(d)));
                        indicators += present;
                    }
                    assert_eq!(indicators == 0, digit == Some(0));
                } else if column >= 18 {
                    let bit = 228 + column - 18;
                    let flag = (row[bit / 64] >> (bit % 64)) & 1;
                    assert_eq!(flag == 0, digit.is_none());
                    assert_eq!(flag, u64::from(digit == Some(0)));
                }
            }
            assert_eq!(row[3] >> 39, 0);
        }
    }
}

#[test]
fn instruction_mix_register_writes_and_ram_locality_match_profile() {
    for profile in [SynthProfile::Local, SynthProfile::AllRows] {
        let trace = SyntheticTrace::new(profile, 16, 1 << 20, 1209).unwrap();
        let mut counts = [0; 9];
        for cycle in 0..CycleSource::cycles(&trace) {
            let variant = trace.digit(12, cycle).unwrap();
            counts[0] += usize::from((28..35).contains(&variant));
            counts[1] += usize::from((35..39).contains(&variant));
            counts[2] += usize::from(trace.digit(13, cycle).is_some());
            counts[3] += usize::from((12..16).contains(&variant));
            counts[4] += usize::from(trace.digit(16, cycle).is_some());
            counts[5] += usize::from(trace.digit(19, cycle).is_some());
            counts[6] += usize::from(trace.trace_word(5, cycle) != 0);
            counts[7] += usize::from(trace.digit(15, cycle).is_some());
            let ram = (5..10)
                .map(|c| trace.digit(c, cycle).unwrap() << (4 * (c - 5)))
                .sum::<usize>();
            assert!(ram < 1 << 12);
            if trace.digit(14, cycle).is_some() {
                counts[8] += usize::from(ram < 64);
            }
        }
        for (actual, expected) in counts[..8]
            .iter()
            .zip([0.25, 0.10, 0.05, 0.08, 0.12, 0.06, 0.75, 0.20])
        {
            assert!((*actual as f64 / 65536.0 - expected).abs() < 0.012);
        }
        let accesses = counts[0] + counts[1];
        assert!((counts[8] as f64 / accesses as f64 - 0.90).abs() < 0.015);
    }
}

#[test]
fn scaled_locality_visits_one_sixteenth_or_every_bytecode_row() {
    // Four cycles per row preserves (log_t, log_rows) = (22, 20).
    for (profile, expected) in [
        (SynthProfile::Local, 1 << 8),
        (SynthProfile::AllRows, 1 << 12),
    ] {
        let trace = SyntheticTrace::new(profile, 14, 1 << 12, 173).unwrap();
        let visited: BTreeSet<_> = (0..CycleSource::cycles(&trace))
            .map(|j| trace.bytecode_index(j))
            .collect();
        assert_eq!(visited.len(), expected);
    }
}

#[test]
fn uniform_digits_are_present_and_uniform_without_row_dependence() {
    let trace = SyntheticTrace::new(SynthProfile::UniformDigits, 16, 1 << 10, 829).unwrap();
    for column in 0..trace.digit_columns() {
        assert!(!trace.by_row(column));
        let bound = 1 << trace.bits(column);
        let mut counts = vec![0; bound];
        for cycle in 0..CycleSource::cycles(&trace) {
            let digit = trace.digit(column, cycle).unwrap();
            counts[digit] += 1;
            assert_eq!(trace.lanes(cycle), [[0; 3]; 2]);
            assert_eq!(trace.tail(cycle), 0);
        }
        for count in counts {
            assert!((count as f64 / 65536.0 - 1.0 / bound as f64).abs() < 0.006);
        }
    }
}

#[test]
fn malformed_dimensions_columns_and_indices_are_total() {
    assert!(matches!(
        SyntheticTrace::new(SynthProfile::Local, 0, 16, 0),
        Err(SynthError::CycleExponent { log_t: 0 })
    ));
    assert!(matches!(
        SyntheticTrace::new(SynthProfile::Local, 33, 16, 0),
        Err(SynthError::CycleExponent { log_t: 33 })
    ));
    assert!(matches!(
        SyntheticTrace::new(SynthProfile::Local, 4, 3, 0),
        Err(SynthError::BytecodeRows { rows: 3 })
    ));
    let trace = Arc::new(SyntheticTrace::new(SynthProfile::Local, 4, 16, 0).unwrap());
    assert!(matches!(
        DigitColumns::new(Arc::clone(&trace), vec![21]),
        Err(SourceError::Column { column: 21, .. })
    ));
    let selected = DigitColumns::new(Arc::clone(&trace), vec![0, 10, 12, 19]).unwrap();
    assert_eq!(selected.columns(), &[0, 10, 12, 19]);
    assert_eq!(selected.index_bound(0), Some(16));
    assert_eq!(selected.index_bound(1), Some(8));
    assert_eq!(selected.index_bound(2), Some(64));
    assert_eq!(selected.index_bound(3), Some(1));
    assert_eq!(selected.index_bound(4), None);
    assert_eq!(selected.index(4, 0), None);
    assert_eq!(selected.index(0, 16), None);
    assert_eq!(trace.trace_word(6, 0), 0);
    assert_eq!(trace.bytecode_word(4, 0), 0);
    assert_eq!(trace.row_digit(0, 16), None);
    assert_eq!(trace.digit(21, 0), None);
}
