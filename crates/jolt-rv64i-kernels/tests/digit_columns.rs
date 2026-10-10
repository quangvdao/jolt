#![cfg(feature = "test-utils")]
#![expect(
    clippy::unwrap_used,
    reason = "test fixture failures are assertion failures"
)]

use jolt_field::F128;
use jolt_kernels::optimized::lazy_ra::{ChunkIndexSource, LazyFoldedRa, LazyRaError};
use jolt_rv64i_kernels::oracle::mle_at;
use jolt_rv64i_kernels::round::eq::eq_table;
use jolt_rv64i_kernels::source::{CycleSource, DigitColumns};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct CountedTrace {
    trace: SyntheticTrace,
    digit_reads: AtomicUsize,
}

impl CycleSource for CountedTrace {
    fn cycles(&self) -> usize {
        CycleSource::cycles(&self.trace)
    }
    fn trace_words(&self) -> usize {
        self.trace.trace_words()
    }
    fn trace_word(&self, word: usize, cycle: usize) -> u64 {
        self.trace.trace_word(word, cycle)
    }
    fn bytecode_rows(&self) -> usize {
        self.trace.bytecode_rows()
    }
    fn bytecode_words(&self) -> usize {
        self.trace.bytecode_words()
    }
    fn bytecode_word(&self, word: usize, row: usize) -> u64 {
        self.trace.bytecode_word(word, row)
    }
    fn bytecode_index(&self, cycle: usize) -> usize {
        self.trace.bytecode_index(cycle)
    }
    fn digit_columns(&self) -> usize {
        self.trace.digit_columns()
    }
    fn bits(&self, column: usize) -> usize {
        self.trace.bits(column)
    }
    fn by_row(&self, column: usize) -> bool {
        self.trace.by_row(column)
    }
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        let _ = self.digit_reads.fetch_add(1, Ordering::Relaxed);
        self.trace.digit(column, cycle)
    }
    fn row_digit(&self, column: usize, row: usize) -> Option<usize> {
        self.trace.row_digit(column, row)
    }
}

#[test]
fn bounded_digit_columns_skip_scan_and_bind_to_dense_summation() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xfa2e);
    let mut saw_absent = false;
    for profile in [
        SynthProfile::Local,
        SynthProfile::AllRows,
        SynthProfile::UniformDigits,
    ] {
        for log_t in [3, 4, 6] {
            let source = Arc::new(CountedTrace {
                trace: SyntheticTrace::new(profile, log_t, 16, 0xc41).unwrap(),
                digit_reads: AtomicUsize::new(0),
            });
            let selected = vec![0, 10, 13, 18];
            let columns = DigitColumns::new(source.clone(), selected.clone()).unwrap();
            assert_eq!(ChunkIndexSource::num_polys(&columns), selected.len());
            assert_eq!(ChunkIndexSource::cycles(&columns), 1 << log_t);
            let points: Vec<Vec<_>> = selected
                .iter()
                .map(|&c| {
                    (0..source.bits(c))
                        .map(|i| {
                            F128::from_raw((u128::from(rng.next_u64()) << 64) | (i as u128 + 2))
                        })
                        .collect()
                })
                .collect();
            let tables: Vec<_> = points.iter().map(|point| eq_table(point, None)).collect();
            let dense: Vec<Vec<_>> = selected
                .iter()
                .zip(&points)
                .enumerate()
                .map(|(column, (&c, point))| {
                    assert_eq!(
                        ChunkIndexSource::index_bound(&columns, column),
                        Some(1 << source.bits(c))
                    );
                    (0..1 << log_t)
                        .map(|cycle| {
                            let digit = source.trace.digit(c, cycle);
                            assert_eq!(ChunkIndexSource::index(&columns, column, cycle), digit);
                            if let Some(index) = digit {
                                let mut vertex = vec![F128::from_raw(0); 1 << source.bits(c)];
                                vertex[index] = F128::from_raw(1);
                                mle_at(&vertex, point).unwrap()
                            } else {
                                saw_absent = true;
                                F128::from_raw(0)
                            }
                        })
                        .collect()
                })
                .collect();
            let reads = source.digit_reads.load(Ordering::Relaxed);
            let mut lazy = LazyFoldedRa::try_new(tables, columns).unwrap();
            assert_eq!(source.digit_reads.load(Ordering::Relaxed), reads);
            let challenges: Vec<_> = (0..log_t)
                .map(|i| F128::from_raw((u128::from(rng.next_u64()) << 64) | (i as u128 + 19)))
                .collect();
            for bound in 0..=log_t {
                for (column, table) in dense.iter().enumerate() {
                    for suffix in 0..1 << (log_t - bound) {
                        let mut point = challenges[..bound].to_vec();
                        point.extend(
                            (0..log_t - bound).map(|i| F128::from_raw(((suffix >> i) & 1) as u128)),
                        );
                        assert_eq!(lazy.value(column, suffix), mle_at(table, &point).unwrap());
                    }
                }
                if bound < log_t {
                    lazy.bind(challenges[bound]);
                }
            }
            assert_eq!(
                lazy.final_values(),
                dense
                    .iter()
                    .map(|table| mle_at(table, &challenges).unwrap())
                    .collect::<Vec<_>>()
            );
        }
    }
    assert!(saw_absent);
}

#[test]
fn digit_bound_rejects_short_table_without_reading_indices() {
    let source = Arc::new(CountedTrace {
        trace: SyntheticTrace::new(SynthProfile::Local, 3, 16, 7).unwrap(),
        digit_reads: AtomicUsize::new(0),
    });
    let columns = DigitColumns::new(source.clone(), vec![0]).unwrap();
    let reads = source.digit_reads.load(Ordering::Relaxed);
    assert!(matches!(
        LazyFoldedRa::try_new(vec![vec![F128::from_raw(0); 15]], columns),
        Err(LazyRaError::IndexBoundExceedsTable {
            poly: 0,
            bound: 16,
            len: 15
        })
    ));
    assert_eq!(source.digit_reads.load(Ordering::Relaxed), reads);
}
