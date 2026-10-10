#![cfg(feature = "test-utils")]
#![expect(
    clippy::unwrap_used,
    reason = "test setup failures are assertion failures"
)]

use jolt_rv64i_kernels::source::{
    CycleSource, DigitColumns, LaneSource, SourceError, ValidatedTrace,
};
use jolt_rv64i_kernels::synth::{SynthError, SynthProfile, SyntheticTrace};
use rayon::ThreadPoolBuilder;
use std::collections::BTreeSet;
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    let checked = Arc::new(ValidatedTrace::new(Arc::clone(&trace)).unwrap());
    let selected = DigitColumns::from_validated(checked, vec![0, 10, 12, 19]).unwrap();
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

#[derive(Debug)]
struct SourceFixture {
    cycles: usize,
    rows: usize,
    columns: usize,
    widths: [usize; 2],
    indices: [usize; 4],
    digits: [[Option<usize>; 4]; 2],
    row_digits: [Option<usize>; 2],
    digit_reads: AtomicUsize,
    row_reads: AtomicUsize,
}

impl SourceFixture {
    fn new() -> Self {
        Self {
            cycles: 4,
            rows: 2,
            columns: 2,
            widths: [2, 2],
            indices: [0, 1, 0, 1],
            digits: [
                [Some(0), Some(1), Some(0), Some(1)],
                [Some(2), None, Some(3), Some(1)],
            ],
            row_digits: [Some(0), Some(1)],
            digit_reads: AtomicUsize::new(0),
            row_reads: AtomicUsize::new(0),
        }
    }
}

impl CycleSource for SourceFixture {
    fn cycles(&self) -> usize {
        self.cycles
    }
    fn trace_words(&self) -> usize {
        1
    }
    fn trace_word(&self, _word: usize, _cycle: usize) -> u64 {
        0
    }
    fn bytecode_rows(&self) -> usize {
        self.rows
    }
    fn bytecode_words(&self) -> usize {
        1
    }
    fn bytecode_word(&self, _word: usize, _row: usize) -> u64 {
        0
    }
    fn bytecode_index(&self, cycle: usize) -> usize {
        self.indices.get(cycle).copied().unwrap_or(0)
    }
    fn digit_columns(&self) -> usize {
        self.columns
    }
    fn bits(&self, column: usize) -> usize {
        self.widths.get(column).copied().unwrap_or(0)
    }
    fn by_row(&self, column: usize) -> bool {
        column == 0
    }
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        let _previous = self.digit_reads.fetch_add(1, Ordering::Relaxed);
        self.digits
            .get(column)
            .and_then(|values| values.get(cycle))
            .copied()
            .flatten()
    }
    fn row_digit(&self, column: usize, row: usize) -> Option<usize> {
        let _previous = self.row_reads.fetch_add(1, Ordering::Relaxed);
        if column == 0 {
            self.row_digits.get(row).copied().flatten()
        } else {
            None
        }
    }
}

#[test]
fn source_validation_rejects_each_malformed_source_contract() {
    let mut source = SourceFixture::new();
    source.cycles = 3;
    assert!(matches!(
        ValidatedTrace::new(Arc::new(source)),
        Err(SourceError::CycleCount { cycles: 3 })
    ));

    let mut source = SourceFixture::new();
    source.rows = 3;
    assert!(matches!(
        ValidatedTrace::new(Arc::new(source)),
        Err(SourceError::BytecodeRows { rows: 3 })
    ));

    let mut source = SourceFixture::new();
    source.widths[1] = usize::BITS as usize;
    assert!(matches!(
        ValidatedTrace::new(Arc::new(source)),
        Err(SourceError::Width { column: 1, .. })
    ));

    let mut source = SourceFixture::new();
    source.indices[2] = 2;
    assert!(matches!(
        ValidatedTrace::new(Arc::new(source)),
        Err(SourceError::BytecodeIndex {
            cycle: 2,
            row: 2,
            rows: 2
        })
    ));

    let mut source = SourceFixture::new();
    source.digits[1][3] = Some(4);
    assert!(matches!(
        DigitColumns::new(Arc::new(source), vec![0]),
        Err(SourceError::Digit {
            column: 1,
            cycle: 3,
            digit: 4,
            bound: 4
        })
    ));

    let mut source = SourceFixture::new();
    source.row_digits[1] = Some(4);
    assert!(matches!(
        ValidatedTrace::new(Arc::new(source)),
        Err(SourceError::RowDigitRange {
            column: 0,
            row: 1,
            digit: 4,
            bound: 4
        })
    ));

    let mut source = SourceFixture::new();
    source.digits[0][2] = Some(1);
    assert!(matches!(
        ValidatedTrace::new(Arc::new(source)),
        Err(SourceError::RowDigit {
            column: 0,
            cycle: 2,
            row: 0
        })
    ));

    let validated = Arc::new(ValidatedTrace::new(Arc::new(SourceFixture::new())).unwrap());
    assert!(matches!(
        DigitColumns::from_validated(validated, vec![2]),
        Err(SourceError::Column {
            column: 2,
            columns: 2
        })
    ));
}

#[test]
fn validated_source_is_scanned_once_and_reused_by_column_selections() {
    let source = Arc::new(SourceFixture::new());
    let validated = Arc::new(ValidatedTrace::new(Arc::clone(&source)).unwrap());
    assert!(Arc::ptr_eq(validated.source(), &source));
    assert_eq!(source.digit_reads.load(Ordering::Relaxed), 8);
    assert_eq!(source.row_reads.load(Ordering::Relaxed), 2);
    let first = DigitColumns::from_validated(Arc::clone(&validated), vec![0, 1]).unwrap();
    let second = DigitColumns::from_validated(validated, vec![1, 0, 1]).unwrap();
    assert_eq!(source.digit_reads.load(Ordering::Relaxed), 8);
    assert_eq!(source.row_reads.load(Ordering::Relaxed), 2);
    assert_eq!(first.cycles(), 4);
    assert_eq!(second.num_polys(), 3);
    assert_eq!(first.index_bound(0), Some(4));
    assert_eq!(second.index(0, 1), None);
}

#[test]
fn validation_rejects_unrepresentable_scratch_sizes_without_allocating() {
    let mut source = SourceFixture::new();
    source.columns = usize::MAX;
    assert!(matches!(
        ValidatedTrace::new(Arc::new(source)),
        Err(SourceError::ValidationScratchSize {
            len: usize::MAX,
            ..
        })
    ));

    let columns = isize::MAX as usize / size_of::<usize>();
    let mut source = SourceFixture::new();
    source.columns = columns;
    assert!(matches!(ValidatedTrace::new(Arc::new(source)),
        Err(SourceError::ValidationScratchSize { len, element_size, .. })
        if len == columns && element_size > size_of::<usize>()));

    let rows = 1_usize << (usize::BITS - 1);
    let mut source = SourceFixture::new();
    source.rows = rows;
    assert!(matches!(ValidatedTrace::new(Arc::new(source)),
        Err(SourceError::ValidationScratchSize { len, element_size, .. })
        if len == rows && element_size == size_of::<usize>()));
}
