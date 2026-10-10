#![cfg(feature = "test-utils")]
#![expect(
    clippy::unwrap_used,
    reason = "test setup failures are assertion failures"
)]

use jolt_rv64i_kernels::par::CycleChunks;
use jolt_rv64i_kernels::source::{
    CycleSource, LaneSource, PrepareRequest, SourceError, ValidatedTrace,
};
use jolt_rv64i_kernels::synth::{SynthError, SynthProfile, SyntheticTrace};
use rayon::ThreadPoolBuilder;
use std::collections::BTreeSet;
use std::mem::size_of;
use std::ops::Range;
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
        if cycle < self.cycles {
            self.indices[cycle % self.indices.len()]
        } else {
            0
        }
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
            .and_then(|values| (cycle < self.cycles).then(|| values[cycle % values.len()]))
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
    source.widths[1] = 16;
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
        ValidatedTrace::new(Arc::new(source)),
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
}

#[test]
fn preparation_reads_each_digit_once_for_repeated_group_columns() {
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for cycles in [4, 1 << 14] {
            pool.install(|| {
                let mut fixture = SourceFixture::new();
                fixture.cycles = cycles;
                let source = Arc::new(fixture);
                let (validated, groups) = ValidatedTrace::prepare(
                    Arc::clone(&source),
                    PrepareRequest {
                        present: vec![vec![0], vec![0, 0]],
                        optional: vec![vec![1, 0, 1]],
                    },
                )
                .unwrap();
                assert!(Arc::ptr_eq(validated.source(), &source));
                assert_eq!(source.digit_reads.load(Ordering::Relaxed), 2 * cycles);
                assert_eq!(source.row_reads.load(Ordering::Relaxed), 2);
                assert_eq!(groups.present.len(), 2);
                assert_eq!(groups.present[0].columns(), &[0]);
                assert_eq!(groups.present[0].widths(), &[2]);
                assert_eq!(groups.present[0].cycles(), cycles);
                assert_eq!(groups.present[0].bytes(), [0, 1, 0, 1].repeat(cycles / 4));
                assert_eq!(groups.present[1].columns(), &[0, 0]);
                assert_eq!(groups.present[1].widths(), &[2, 2]);
                assert_eq!(groups.present[1].cycles(), cycles);
                assert_eq!(
                    groups.present[1].bytes(),
                    [0, 0, 1, 1, 0, 0, 1, 1].repeat(cycles / 4)
                );
                assert_eq!(groups.optional[0].columns(), &[1, 0, 1]);
                assert_eq!(groups.optional[0].widths(), &[2, 2, 2]);
                assert_eq!(groups.optional[0].cycles(), cycles);
                assert_eq!(
                    groups.optional[0].bytes(),
                    [3, 1, 3, 0, 2, 0, 4, 1, 4, 2, 2, 2].repeat(cycles / 4)
                );
            });
        }
    }
}

struct LiteralRowSource {
    digits: Vec<[Option<usize>; 3]>,
    digit_reads: Vec<[AtomicUsize; 3]>,
}

impl CycleSource for LiteralRowSource {
    fn cycles(&self) -> usize {
        self.digits.len()
    }
    fn trace_words(&self) -> usize {
        0
    }
    fn trace_word(&self, _: usize, _: usize) -> u64 {
        0
    }
    fn bytecode_rows(&self) -> usize {
        2
    }
    fn bytecode_words(&self) -> usize {
        0
    }
    fn bytecode_word(&self, _: usize, _: usize) -> u64 {
        0
    }
    fn bytecode_index(&self, cycle: usize) -> usize {
        [0, 1, 1, 0][cycle % 4]
    }
    fn digit_columns(&self) -> usize {
        3
    }
    fn bits(&self, _: usize) -> usize {
        2
    }
    fn by_row(&self, column: usize) -> bool {
        column == 0 || column == 2
    }
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        let _previous = self.digit_reads[cycle][column].fetch_add(1, Ordering::Relaxed);
        self.digits[cycle][column]
    }
    fn row_digit(&self, column: usize, row: usize) -> Option<usize> {
        [[Some(0), None, Some(2)], [Some(1), None, Some(3)]][row][column]
    }
}

#[test]
fn preparation_checks_distinct_row_columns_without_rereading_valid_digits() {
    for threads in [1, 12] {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let source = Arc::new(LiteralRowSource {
                    digits: vec![
                        [Some(0), Some(1), Some(2)],
                        [Some(1), None, Some(3)],
                        [Some(1), Some(2), Some(3)],
                        [Some(0), Some(3), Some(2)],
                    ],
                    digit_reads: (0..4)
                        .map(|_| std::array::from_fn(|_| AtomicUsize::new(0)))
                        .collect(),
                });
                let (_, groups) = ValidatedTrace::prepare(
                    Arc::clone(&source),
                    PrepareRequest {
                        present: vec![vec![2, 0, 2]],
                        optional: vec![vec![1, 2, 0]],
                    },
                )
                .unwrap();
                assert_eq!(
                    groups.present[0].bytes(),
                    &[2, 0, 2, 3, 1, 3, 3, 1, 3, 2, 0, 2]
                );
                assert_eq!(
                    groups.optional[0].bytes(),
                    &[2, 3, 1, 0, 4, 2, 3, 4, 2, 4, 3, 1]
                );
                for reads in source.digit_reads.iter().flatten() {
                    assert_eq!(reads.load(Ordering::Relaxed), 1);
                }
            });
    }
}

#[test]
fn validation_reports_the_only_fault_at_the_final_partial_tile_cycle() {
    for threads in [1, 12] {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let mut digits = [
                    [Some(0), Some(1), Some(2)],
                    [Some(1), None, Some(3)],
                    [Some(1), Some(2), Some(3)],
                    [Some(0), Some(3), Some(2)],
                ]
                .repeat(2048);
                digits[8191][1] = Some(4);
                let source = Arc::new(LiteralRowSource {
                    digits,
                    digit_reads: (0..8192)
                        .map(|_| std::array::from_fn(|_| AtomicUsize::new(0)))
                        .collect(),
                });
                assert_eq!(
                    ValidatedTrace::new(source).err(),
                    Some(SourceError::Digit {
                        column: 1,
                        cycle: 8191,
                        digit: 4,
                        bound: 4,
                    })
                );
            });
    }
}

#[test]
fn preparation_rejects_request_columns_and_widths_before_digit_reads() {
    for (request, expected) in [
        (
            PrepareRequest {
                present: vec![vec![2]],
                optional: vec![],
            },
            SourceError::Column {
                column: 2,
                columns: 2,
            },
        ),
        (
            PrepareRequest {
                present: vec![],
                optional: vec![vec![2]],
            },
            SourceError::Column {
                column: 2,
                columns: 2,
            },
        ),
        (
            PrepareRequest {
                present: vec![vec![1]],
                optional: vec![],
            },
            SourceError::GroupWidth {
                column: 1,
                bits: 9,
                max_bits: 8,
            },
        ),
        (
            PrepareRequest {
                present: vec![],
                optional: vec![vec![1]],
            },
            SourceError::GroupWidth {
                column: 1,
                bits: 9,
                max_bits: 7,
            },
        ),
    ] {
        let mut fixture = SourceFixture::new();
        fixture.widths[1] = 9;
        let source = Arc::new(fixture);
        assert_eq!(
            ValidatedTrace::prepare(Arc::clone(&source), request).err(),
            Some(expected)
        );
        assert_eq!(source.digit_reads.load(Ordering::Relaxed), 0);
        assert_eq!(source.row_reads.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn preparation_encodes_full_width_present_and_optional_bytes() {
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut source = SourceFixture::new();
            source.widths = [8, 7];
            source.row_digits = [Some(255), Some(0)];
            source.digits = [
                [Some(255), Some(0), Some(255), Some(0)],
                [Some(127), None, Some(0), Some(1)],
            ];
            let (_, groups) = ValidatedTrace::prepare(
                Arc::new(source),
                PrepareRequest {
                    present: vec![vec![0]],
                    optional: vec![vec![1]],
                },
            )
            .unwrap();
            assert_eq!(groups.present[0].bytes(), &[255, 0, 255, 0]);
            assert_eq!(groups.present[0].widths(), &[8]);
            assert_eq!(groups.optional[0].bytes(), &[128, 0, 1, 2]);
            assert_eq!(groups.optional[0].widths(), &[7]);
        });
    }
}

struct GroupEncodingSource;

impl CycleSource for GroupEncodingSource {
    fn cycles(&self) -> usize {
        8192
    }
    fn trace_words(&self) -> usize {
        0
    }
    fn trace_word(&self, _: usize, _: usize) -> u64 {
        0
    }
    fn bytecode_rows(&self) -> usize {
        2
    }
    fn bytecode_words(&self) -> usize {
        0
    }
    fn bytecode_word(&self, _: usize, _: usize) -> u64 {
        0
    }
    fn bytecode_index(&self, cycle: usize) -> usize {
        cycle % 2
    }
    fn digit_columns(&self) -> usize {
        21
    }
    fn bits(&self, column: usize) -> usize {
        if column == 0 {
            8
        } else {
            7
        }
    }
    fn by_row(&self, column: usize) -> bool {
        column < 5 || (12..18).contains(&column)
    }
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        if cycle >= self.cycles() {
            None
        } else {
            self.row_digit(column, cycle % 2)
        }
    }
    fn row_digit(&self, column: usize, row: usize) -> Option<usize> {
        if row >= self.bytecode_rows() || (column == 13 && row == 1) {
            None
        } else {
            [
                255, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            ]
            .get(column)
            .copied()
        }
    }
}

#[test]
fn preparation_keeps_literal_group_encodings_across_tiles_and_chunks() {
    let cases: [(bool, &[usize], &[u8]); 12] = [
        (true, &[0, 1, 2, 3, 4], &[255, 1, 2, 3, 4, 255, 1, 2, 3, 4]),
        (true, &[4, 0, 3, 1, 2], &[4, 255, 3, 1, 2, 4, 255, 3, 1, 2]),
        (
            true,
            &[0, 0, 3, 1, 3],
            &[255, 255, 3, 1, 3, 255, 255, 3, 1, 3],
        ),
        (
            true,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[255, 1, 2, 3, 4, 5, 6, 7, 255, 1, 2, 3, 4, 5, 6, 7],
        ),
        (
            true,
            &[7, 0, 5, 2, 6, 1, 4, 3],
            &[7, 255, 5, 2, 6, 1, 4, 3, 7, 255, 5, 2, 6, 1, 4, 3],
        ),
        (
            true,
            &[0, 0, 3, 3, 2, 5, 2, 7],
            &[255, 255, 3, 3, 2, 5, 2, 7, 255, 255, 3, 3, 2, 5, 2, 7],
        ),
        (
            false,
            &[10, 11, 12, 13, 14],
            &[11, 12, 13, 14, 15, 11, 12, 13, 0, 15],
        ),
        (
            false,
            &[14, 10, 13, 11, 12],
            &[15, 11, 14, 12, 13, 15, 11, 0, 12, 13],
        ),
        (
            false,
            &[13, 13, 10, 14, 10],
            &[14, 14, 11, 15, 11, 0, 0, 11, 15, 11],
        ),
        (
            false,
            &[10, 11, 12, 13, 14, 15, 16, 17],
            &[
                11, 12, 13, 14, 15, 16, 17, 18, 11, 12, 13, 0, 15, 16, 17, 18,
            ],
        ),
        (
            false,
            &[17, 10, 15, 13, 16, 11, 14, 12],
            &[
                18, 11, 16, 14, 17, 12, 15, 13, 18, 11, 16, 0, 17, 12, 15, 13,
            ],
        ),
        (
            false,
            &[13, 13, 10, 14, 10, 15, 16, 17],
            &[14, 14, 11, 15, 11, 16, 17, 18, 0, 0, 11, 15, 11, 16, 17, 18],
        ),
    ];
    let source = Arc::new(GroupEncodingSource);
    for threads in [1, 12] {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                for (present, columns, expected) in cases {
                    let request = if present {
                        PrepareRequest {
                            present: vec![columns.to_vec()],
                            optional: vec![],
                        }
                    } else {
                        PrepareRequest {
                            present: vec![],
                            optional: vec![columns.to_vec()],
                        }
                    };
                    let (_, groups) =
                        ValidatedTrace::prepare(Arc::clone(&source), request).unwrap();
                    let bytes = if present {
                        groups.present[0].bytes()
                    } else {
                        groups.optional[0].bytes()
                    };
                    assert_eq!(
                        bytes,
                        expected.repeat(4096),
                        "present={present}, columns={columns:?}"
                    );
                }
                let (_, groups) = ValidatedTrace::prepare(
                    Arc::clone(&source),
                    PrepareRequest {
                        present: vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]],
                        optional: vec![vec![12, 10, 11, 13, 14, 15, 16, 19]],
                    },
                )
                .unwrap();
                assert_eq!(groups.present[0].bytes(), [255, 1, 2, 3, 4].repeat(8192));
                assert_eq!(groups.present[1].bytes(), [5, 6, 7, 8, 9].repeat(8192));
                assert_eq!(
                    groups.optional[0].bytes(),
                    [13, 11, 12, 14, 15, 16, 17, 20, 13, 11, 12, 0, 15, 16, 17, 20].repeat(4096)
                );
            });
    }
}

#[test]
fn validation_rejects_unrepresentable_scratch_sizes_without_allocating() {
    let mut source = SourceFixture::new();
    source.columns = usize::MAX;
    assert!(matches!(
        ValidatedTrace::new(Arc::new(source)),
        Err(SourceError::ColumnCapacity {
            columns: usize::MAX,
            max_columns: 128,
        })
    ));

    let columns = isize::MAX as usize / size_of::<usize>();
    let mut source = SourceFixture::new();
    source.columns = columns;
    assert!(matches!(ValidatedTrace::new(Arc::new(source)),
        Err(SourceError::ColumnCapacity { columns: actual, max_columns: 128 })
        if actual == columns));

    let rows = 1_usize << (usize::BITS - 1);
    let mut source = SourceFixture::new();
    source.rows = rows;
    assert!(matches!(ValidatedTrace::new(Arc::new(source)),
        Err(SourceError::ValidationScratchSize { len, element_size, .. })
        if len == rows && element_size == size_of::<u16>()));
}

struct OrderedFaults {
    row_fault: bool,
    same_cycle_row_fault: bool,
    same_column_range_fault: bool,
    invalid_index: bool,
}

impl CycleSource for OrderedFaults {
    fn cycles(&self) -> usize {
        1 << 14
    }
    fn trace_words(&self) -> usize {
        0
    }
    fn trace_word(&self, _word: usize, _cycle: usize) -> u64 {
        0
    }
    fn bytecode_rows(&self) -> usize {
        1 << 13
    }
    fn bytecode_words(&self) -> usize {
        0
    }
    fn bytecode_word(&self, _word: usize, _row: usize) -> u64 {
        0
    }
    fn bytecode_index(&self, cycle: usize) -> usize {
        if self.invalid_index && cycle == 8190 {
            self.bytecode_rows()
        } else if cycle < self.cycles() {
            cycle & ((1 << 13) - 1)
        } else {
            0
        }
    }
    fn digit_columns(&self) -> usize {
        3
    }
    fn bits(&self, column: usize) -> usize {
        match column {
            0 | 1 => 2,
            2 => 8,
            _ => 0,
        }
    }
    fn by_row(&self, column: usize) -> bool {
        column == 0 || column == 2
    }
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        if cycle >= self.cycles() {
            return None;
        }
        match (column, cycle) {
            (0, 8190) if self.same_column_range_fault => Some(4),
            (0, 8190) if self.same_cycle_row_fault => Some(1),
            (0, 8192) => Some(1),
            (1, 8190) => Some(4),
            (0 | 1, _) => Some(0),
            (2, _) => Some(255),
            _ => None,
        }
    }
    fn row_digit(&self, column: usize, row: usize) -> Option<usize> {
        if row >= self.bytecode_rows() {
            return None;
        }
        match (column, row) {
            (0, 4095) if self.row_fault => Some(4),
            (2, 4094) if self.row_fault => Some(256),
            (0, _) => Some(0),
            (2, _) => Some(255),
            _ => None,
        }
    }
}

#[test]
fn parallel_validation_returns_the_earliest_row_or_cycle_fault_on_every_pool() {
    assert_eq!(CycleChunks::new(14, 0).unwrap().ranges().len(), 4);
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let source = Arc::new(OrderedFaults {
                row_fault: false,
                same_cycle_row_fault: false,
                same_column_range_fault: false,
                invalid_index: false,
            });
            assert_eq!(
                ValidatedTrace::new(source).err(),
                Some(SourceError::Digit {
                    column: 1,
                    cycle: 8190,
                    digit: 4,
                    bound: 4,
                })
            );
            let source = Arc::new(OrderedFaults {
                row_fault: true,
                same_cycle_row_fault: false,
                same_column_range_fault: false,
                invalid_index: false,
            });
            assert_eq!(
                ValidatedTrace::new(source).err(),
                Some(SourceError::RowDigitRange {
                    column: 2,
                    row: 4094,
                    digit: 256,
                    bound: 256,
                })
            );
            let source = Arc::new(OrderedFaults {
                row_fault: false,
                same_cycle_row_fault: true,
                same_column_range_fault: false,
                invalid_index: false,
            });
            assert_eq!(
                ValidatedTrace::new(source).err(),
                Some(SourceError::RowDigit {
                    column: 0,
                    cycle: 8190,
                    row: 8190,
                })
            );
            let source = Arc::new(OrderedFaults {
                row_fault: false,
                same_cycle_row_fault: false,
                same_column_range_fault: true,
                invalid_index: false,
            });
            assert_eq!(
                ValidatedTrace::new(source).err(),
                Some(SourceError::Digit {
                    column: 0,
                    cycle: 8190,
                    digit: 4,
                    bound: 4,
                })
            );
            let source = Arc::new(OrderedFaults {
                row_fault: false,
                same_cycle_row_fault: true,
                same_column_range_fault: false,
                invalid_index: true,
            });
            assert_eq!(
                ValidatedTrace::new(source).err(),
                Some(SourceError::BytecodeIndex {
                    cycle: 8190,
                    row: 8192,
                    rows: 8192,
                })
            );
        });
    }
}

struct MissingFaults {
    range_cycle: usize,
}

impl CycleSource for MissingFaults {
    fn cycles(&self) -> usize {
        1 << 14
    }
    fn trace_words(&self) -> usize {
        0
    }
    fn trace_word(&self, _: usize, _: usize) -> u64 {
        0
    }
    fn bytecode_rows(&self) -> usize {
        1
    }
    fn bytecode_words(&self) -> usize {
        0
    }
    fn bytecode_word(&self, _: usize, _: usize) -> u64 {
        0
    }
    fn bytecode_index(&self, _: usize) -> usize {
        0
    }
    fn digit_columns(&self) -> usize {
        2
    }
    fn bits(&self, _: usize) -> usize {
        2
    }
    fn by_row(&self, _: usize) -> bool {
        false
    }
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        match (column, cycle) {
            (0, cycle) if cycle == self.range_cycle => Some(4),
            (1, 8190) => None,
            (0 | 1, cycle) if cycle < self.cycles() => Some(0),
            _ => None,
        }
    }
    fn row_digit(&self, _: usize, _: usize) -> Option<usize> {
        None
    }
}

#[test]
fn missing_digit_rank_preserves_cycle_column_and_chunk_order_on_every_pool() {
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for (range_cycle, expected) in [
                (
                    8189,
                    SourceError::Digit {
                        column: 0,
                        cycle: 8189,
                        digit: 4,
                        bound: 4,
                    },
                ),
                (
                    8190,
                    SourceError::Digit {
                        column: 0,
                        cycle: 8190,
                        digit: 4,
                        bound: 4,
                    },
                ),
                (
                    8192,
                    SourceError::MissingDigit {
                        column: 1,
                        cycle: 8190,
                    },
                ),
            ] {
                let source = Arc::new(MissingFaults { range_cycle });
                assert_eq!(
                    ValidatedTrace::prepare(
                        source,
                        PrepareRequest {
                            present: vec![vec![1]],
                            optional: vec![],
                        }
                    )
                    .err(),
                    Some(expected)
                );
            }
            let mut source = SourceFixture::new();
            source.digits[0][2] = None;
            assert_eq!(
                ValidatedTrace::prepare(
                    Arc::new(source),
                    PrepareRequest {
                        present: vec![vec![0]],
                        optional: vec![],
                    }
                )
                .err(),
                Some(SourceError::RowDigit {
                    column: 0,
                    cycle: 2,
                    row: 0
                })
            );
        });
    }
}

#[test]
fn bulk_digits_default_saturates_and_zeroes_malformed_output() {
    let mut source = SourceFixture::new();
    source.digits[1][0] = Some(usize::MAX);
    let mut out = [99; 8];
    source.digits(0..4, &mut out);
    assert_eq!(out, [1, u16::MAX, 2, 0, 1, 4, 2, 2]);
    let before = source.digit_reads.load(Ordering::Relaxed);
    let mut short = [99; 7];
    source.digits(0..4, &mut short);
    assert_eq!(short, [0; 7]);
    source.digits(0..4, &mut []);
    assert_eq!(source.digit_reads.load(Ordering::Relaxed), before);
}

struct BulkFixture {
    source: SourceFixture,
    encoded: [[u16; 2]; 4],
}

impl CycleSource for BulkFixture {
    fn cycles(&self) -> usize {
        self.source.cycles()
    }
    fn trace_words(&self) -> usize {
        self.source.trace_words()
    }
    fn trace_word(&self, word: usize, cycle: usize) -> u64 {
        self.source.trace_word(word, cycle)
    }
    fn bytecode_rows(&self) -> usize {
        self.source.bytecode_rows()
    }
    fn bytecode_words(&self) -> usize {
        self.source.bytecode_words()
    }
    fn bytecode_word(&self, word: usize, row: usize) -> u64 {
        self.source.bytecode_word(word, row)
    }
    fn bytecode_index(&self, cycle: usize) -> usize {
        self.source.bytecode_index(cycle)
    }
    fn digit_columns(&self) -> usize {
        2
    }
    fn bits(&self, column: usize) -> usize {
        self.source.bits(column)
    }
    fn by_row(&self, column: usize) -> bool {
        self.source.by_row(column)
    }
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        self.source.digit(column, cycle)
    }
    fn row_digit(&self, column: usize, row: usize) -> Option<usize> {
        self.source.row_digit(column, row)
    }
    fn digits(&self, cycles: Range<usize>, out: &mut [u16]) {
        if cycles.len().checked_mul(2) != Some(out.len()) {
            out.fill(0);
            return;
        }
        for (cycle, output) in cycles.zip(out.chunks_exact_mut(2)) {
            output.copy_from_slice(&self.encoded[cycle % 4]);
        }
    }
}

#[test]
fn rejected_bulk_tiles_use_scalar_faults_and_view_precedence_on_every_pool() {
    for threads in [1, 12] {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let mut source = BulkFixture {
                    source: SourceFixture::new(),
                    encoded: [[1, 3], [2, 0], [1, 4], [2, 2]],
                };
                source.encoded[0][0] = 5;
                assert_eq!(
                    ValidatedTrace::new(Arc::new(source)).err(),
                    Some(SourceError::DigitView {
                        column: 0,
                        cycle: 0,
                        encoded: 5,
                        digit: Some(0)
                    })
                );
                let mut source = BulkFixture {
                    source: SourceFixture::new(),
                    encoded: [[1, 3], [2, 0], [1, 4], [2, 2]],
                };
                source.encoded[0][1] = 5;
                source.source.indices[0] = 2;
                assert_eq!(
                    ValidatedTrace::new(Arc::new(source)).err(),
                    Some(SourceError::BytecodeIndex {
                        cycle: 0,
                        row: 2,
                        rows: 2
                    })
                );
                let mut source = SourceFixture::new();
                source.widths[1] = 15;
                source.digits[1][0] = Some(usize::MAX);
                assert_eq!(
                    ValidatedTrace::new(Arc::new(source)).err(),
                    Some(SourceError::Digit {
                        column: 1,
                        cycle: 0,
                        digit: usize::MAX,
                        bound: 32768
                    })
                );
            });
    }
}

#[test]
fn tile_column_capacity_is_checked_before_digit_reads_and_empty_columns_work() {
    let mut source = SourceFixture::new();
    source.columns = 129;
    let source = Arc::new(source);
    assert_eq!(
        ValidatedTrace::new(Arc::clone(&source)).err(),
        Some(SourceError::ColumnCapacity {
            columns: 129,
            max_columns: 128
        })
    );
    assert_eq!(source.digit_reads.load(Ordering::Relaxed), 0);
    assert_eq!(source.row_reads.load(Ordering::Relaxed), 0);
    let mut source = SourceFixture::new();
    source.columns = 0;
    assert!(ValidatedTrace::new(Arc::new(source)).is_ok());
}
