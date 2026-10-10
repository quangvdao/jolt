#![cfg(feature = "test-utils")]

#[path = "../benches/support/allocator.rs"]
mod allocator;

#[expect(
    clippy::unwrap_used,
    reason = "invalid fixtures and contract failures fail the tests"
)]
mod tests {
    use super::allocator::{
        AllocationMeasurement, AllocationStats, CountingAllocator, RAYON_WORKER_ALLOWANCE,
    };
    use jolt_field::F128;
    use jolt_rv64i_kernels::memory::{
        address_column, fold_words, inc_lift, row_weights, MemoryError, MemoryTrace, RowWeight,
        MAX_ADDRESS_BITS,
    };
    use jolt_rv64i_kernels::packed::lift::WordLift;
    use jolt_rv64i_kernels::packed::scatter::ScatterPlan;
    use jolt_rv64i_kernels::source::{
        CycleSource, OptionalGroup, PrepareRequest, PresentGroup, ValidatedTrace,
    };
    use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticMemory, SyntheticTrace};
    use rand_chacha::rand_core::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use rayon::{ThreadPool, ThreadPoolBuilder};
    use std::mem::size_of;
    use std::sync::Arc;

    const REGISTERS: [[usize; 3]; 8] = [
        [1, 2, 3],
        [3, 4, 0],
        [5, 6, 0],
        [0, 7, 8],
        [8, 9, 10],
        [10, 11, 12],
        [12, 13, 14],
        [14, 15, 1],
    ];
    const ADDRESSES: [usize; 8] = [0x54321, 0x54321, 0, 0, 0x102, 0, 0x102, 0x102];

    struct Fixture {
        base: SyntheticTrace,
        widths: [usize; 29],
        ram: Vec<usize>,
    }

    impl Fixture {
        fn new(log_t: usize, ram_widths: &[usize]) -> Self {
            let base = SyntheticTrace::new(
                SynthProfile::AllRows,
                log_t,
                if log_t >= 6 { 64 } else { 8 },
                739,
            )
            .unwrap();
            let mut widths = std::array::from_fn(|column| base.bits(column));
            widths[24..29].fill(0);
            widths[24..24 + ram_widths.len()].copy_from_slice(ram_widths);
            Self {
                base,
                widths,
                ram: (24..24 + ram_widths.len()).collect(),
            }
        }

        fn address(&self, cycle: usize) -> usize {
            ADDRESSES[cycle % 8]
                & ((1
                    << self
                        .ram
                        .iter()
                        .map(|&column| self.bits(column))
                        .sum::<usize>())
                    - 1)
        }

        fn registers(row: usize) -> [usize; 3] {
            if row < 8 {
                REGISTERS[row]
            } else {
                [row % 32, (row + 1) % 32, (row + 2) % 32]
            }
        }

        fn groups(
            self: Arc<Self>,
            ram: Vec<usize>,
            registers: Vec<usize>,
            store: Vec<usize>,
        ) -> (
            Arc<ValidatedTrace<Self>>,
            PresentGroup,
            PresentGroup,
            OptionalGroup,
        ) {
            let (trace, groups) = ValidatedTrace::prepare(
                self,
                PrepareRequest {
                    present: vec![ram, registers],
                    optional: vec![store],
                },
            )
            .unwrap();
            let mut present = groups.present.into_iter();
            (
                Arc::new(trace),
                present.next().unwrap(),
                present.next().unwrap(),
                groups.optional.into_iter().next().unwrap(),
            )
        }

        fn prepared(self: Arc<Self>) -> (Arc<ValidatedTrace<Self>>, MemoryTrace) {
            let ram_columns = self.ram.clone();
            let (trace, ram, registers, store) =
                self.groups(ram_columns, vec![21, 22, 23], vec![17]);
            (trace, MemoryTrace::new(ram, registers, store).unwrap())
        }
    }

    impl CycleSource for Fixture {
        fn cycles(&self) -> usize {
            self.base.cycles()
        }
        fn trace_words(&self) -> usize {
            self.base.trace_words()
        }
        fn trace_word(&self, word: usize, cycle: usize) -> u64 {
            if cycle >= self.cycles() || word >= self.trace_words() {
                return 0;
            }
            if word == 5 {
                self.base.trace_word(word, cycle) | 1
            } else {
                self.base.trace_word(word, cycle)
            }
        }
        fn bytecode_rows(&self) -> usize {
            self.base.bytecode_rows()
        }
        fn bytecode_words(&self) -> usize {
            self.base.bytecode_words()
        }
        fn bytecode_word(&self, word: usize, row: usize) -> u64 {
            if row >= self.bytecode_rows() {
                return 0;
            }
            let value = self.base.bytecode_word(word, row);
            if word == 0 {
                let [rs1, rs2, rd] = Self::registers(row);
                (value & !0x7fff) | (rs1 | rs2 << 5 | rd << 10) as u64
            } else {
                value
            }
        }
        fn bytecode_index(&self, cycle: usize) -> usize {
            cycle % self.bytecode_rows()
        }
        fn digit_columns(&self) -> usize {
            self.widths.len()
        }
        fn bits(&self, column: usize) -> usize {
            self.widths.get(column).copied().unwrap_or(0)
        }
        fn by_row(&self, column: usize) -> bool {
            self.base.by_row(column)
        }
        fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
            if column >= self.digit_columns() || cycle >= self.cycles() {
                return None;
            }
            if (24..29).contains(&column) {
                let shift: usize = self.widths[24..column].iter().sum();
                Some((self.address(cycle) >> shift) & ((1 << self.bits(column)) - 1))
            } else if self.by_row(column) {
                self.row_digit(column, self.bytecode_index(cycle))
            } else {
                self.base.digit(column, cycle)
            }
        }
        fn row_digit(&self, column: usize, row: usize) -> Option<usize> {
            if row >= self.bytecode_rows() {
                return None;
            }
            match column {
                21..=23 => Some(Self::registers(row)[column - 21]),
                12 => Some(match row % 8 {
                    2 | 4 | 6 => 35,
                    0 | 1 | 7 => 28,
                    _ => 0,
                }),
                13 | 15 | 16 => None,
                14 => match row % 8 {
                    2 | 4 | 6 => Some(7),
                    0 | 1 | 7 => Some(0),
                    _ => None,
                },
                17 => matches!(row % 8, 2 | 4 | 6).then_some(0),
                _ => self.base.row_digit(column, row),
            }
        }
    }

    fn field(rng: &mut ChaCha20Rng) -> F128 {
        F128::from_raw(u128::from(rng.next_u64()) | u128::from(rng.next_u64()) << 64)
    }

    fn equality(point: &[F128], index: usize) -> F128 {
        point
            .iter()
            .enumerate()
            .fold(F128::from_raw(1), |product, (bit, &coordinate)| {
                product
                    * (F128::from_raw(1)
                        + coordinate
                        + F128::from_raw(((index >> bit) & 1) as u128))
            })
    }

    fn lifted(word: u64, weights: &[F128; 64]) -> F128 {
        weights
            .iter()
            .enumerate()
            .filter(|(bit, _)| word & (1 << bit) != 0)
            .fold(F128::from_raw(0), |sum, (_, &weight)| sum + weight)
    }

    fn check_cycle_views(log_t: usize, widths: &[usize]) {
        let source = Arc::new(Fixture::new(log_t, widths));
        let mut rng = ChaCha20Rng::seed_from_u64(91);
        let weights = std::array::from_fn(|_| field(&mut rng));
        let lift = WordLift::new(&weights);
        let address_point: Vec<_> = (0..widths.iter().sum::<usize>())
            .map(|_| field(&mut rng))
            .collect();
        let cycle_point: Vec<_> = (0..log_t).map(|_| field(&mut rng)).collect();
        let (trace, memory) = source.clone().prepared();
        assert_eq!(memory.cycles(), 1 << log_t);
        assert_eq!(memory.address_bits(), widths.iter().sum::<usize>());
        let increments = inc_lift(&trace, 5, &lift).unwrap();
        for (cycle, &value) in increments.iter().enumerate() {
            assert_eq!(
                value,
                lifted(source.trace_word(5, cycle), &weights),
                "inc cycle {cycle}"
            );
        }
        let addresses = address_column(&memory, &address_point).unwrap();
        for (cycle, &value) in addresses.iter().enumerate() {
            assert_eq!(
                value,
                equality(&address_point, source.address(cycle)),
                "address cycle {cycle}"
            );
        }
        let plan = ScatterPlan::new(trace).unwrap();
        for weight in [RowWeight::Eq(&cycle_point), RowWeight::Next(&cycle_point)] {
            let mut expected = vec![F128::from_raw(0); source.bytecode_rows()];
            for cycle in 0..source.cycles() {
                let value = match weight {
                    RowWeight::Eq(point) => equality(point, cycle),
                    RowWeight::Next(_) if cycle == 0 => F128::from_raw(0),
                    RowWeight::Next(point) => equality(point, cycle - 1),
                };
                expected[source.bytecode_index(cycle)] += value;
            }
            assert_eq!(row_weights(&plan, weight).unwrap(), expected);
        }
        let mut expected = Vec::new();
        for cycle in 0..source.cycles() {
            for &column in &source.ram {
                expected.push(source.digit(column, cycle).unwrap() as u8);
            }
        }
        let (trace, ram, registers, store) =
            source
                .clone()
                .groups(source.ram.clone(), vec![21, 22, 23], vec![17]);
        drop(trace);
        let pointer = ram.bytes().as_ptr();
        let memory = Arc::new(MemoryTrace::new(ram, registers, store).unwrap());
        let ram = MemoryTrace::into_ram_chunks(memory).unwrap();
        assert_eq!(ram.bytes(), expected);
        assert_eq!(ram.bytes().as_ptr(), pointer);
    }

    #[test]
    fn memory_views_equal_direct_sums_on_separating_trace() {
        for log_t in [3, 8] {
            for widths in [&[1][..], &[4, 4], &[8, 4], &[4; 5]] {
                check_cycle_views(log_t, widths);
            }
        }
        let mut rng = ChaCha20Rng::seed_from_u64(92);
        let words: Vec<_> = (0..4096).map(|_| rng.next_u64()).collect();
        let weights = std::array::from_fn(|_| field(&mut rng));
        let lift = WordLift::new(&weights);
        let lifted_words: Vec<_> = words.iter().map(|&word| lifted(word, &weights)).collect();
        let point: Vec<_> = (0..12).map(|_| field(&mut rng)).collect();
        for coordinates in 0..=12 {
            let point = &point[..coordinates];
            let block = 1 << coordinates;
            let expected: Vec<_> = lifted_words
                .chunks(block)
                .map(|words| {
                    words
                        .iter()
                        .enumerate()
                        .fold(F128::from_raw(0), |sum, (low, &word)| {
                            sum + equality(point, low) * word
                        })
                })
                .collect();
            assert_eq!(
                fold_words(&words, &lift, point).unwrap(),
                expected,
                "fold coordinates {coordinates}"
            );
        }
    }

    #[test]
    fn views_equal_definitions_across_parallel_chunks() {
        for threads in [1, 12] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                check_cycle_views(13, &[2]);
                let weights = std::array::from_fn(|bit| F128::from_raw(bit as u128 * 17 + 3));
                let words: Vec<_> = (0..8192_u64)
                    .map(|word| word.wrapping_mul(0x9e37_79b9_7f4a_7c15))
                    .collect();
                let point: Vec<_> = (0..13).map(|bit| F128::from_raw(bit * 13 + 2)).collect();
                let expected = words
                    .iter()
                    .enumerate()
                    .fold(F128::from_raw(0), |sum, (index, &word)| {
                        sum + equality(&point, index) * lifted(word, &weights)
                    });
                assert_eq!(
                    fold_words(&words, &WordLift::new(&weights), &point).unwrap(),
                    vec![expected]
                );
            });
        }
    }

    #[test]
    fn synthetic_memory_replays_stores_on_the_exposed_address_domain() {
        for (widths, address_bits) in [
            (&[1][..], 1),
            (&[4, 4][..], 8),
            (&[8, 4][..], 12),
            (&[4; 5][..], 20),
            (&[4; 5][..], 8),
        ] {
            let source = Fixture::new(8, widths);
            let memory = SyntheticMemory::new(&source, address_bits, 93).unwrap();
            let word_count = 1 << address_bits;
            assert_eq!(memory.initial.len(), word_count);
            for (index, &(address, word)) in memory.initial.iter().enumerate() {
                assert_eq!(address, index as u64);
                assert_ne!(word, 0);
            }
            let mut expected: Vec<_> = memory.initial.iter().map(|&(_, word)| word).collect();
            for cycle in 0..source.cycles() {
                if source.digit(17, cycle).is_some() {
                    expected[source.address(cycle) & (word_count - 1)] ^=
                        source.trace_word(5, cycle);
                }
            }
            assert_eq!(memory.final_words.as_slice(), expected);
            assert_eq!(memory.mask, 0..word_count.min(4096));
            assert_eq!(memory.io, expected[memory.mask.clone()]);
            if address_bits == 8 && widths.iter().sum::<usize>() == 20 {
                let (_, narrow) = Arc::new(Fixture::new(8, &[4, 4])).prepared();
                let point: Vec<_> = (2..10).map(F128::from_raw).collect();
                for (cycle, &value) in address_column(&narrow, &point).unwrap().iter().enumerate() {
                    assert_eq!(value, equality(&point, source.address(cycle) & 255));
                }
            }
        }
    }

    #[test]
    fn synthetic_memory_columns_and_chained_dependencies_hold() {
        assert_eq!(
            SyntheticTrace::memory_columns(),
            ([24, 25, 26, 27, 28], [21, 22, 23], 17)
        );
        assert_eq!(
            SyntheticTrace::packed_columns(),
            (18, [6, 7], [0, 1, 2, 3, 4, 24, 25, 26, 27, 28, 10, 11])
        );
        for profile in [
            SynthProfile::Local,
            SynthProfile::AllRows,
            SynthProfile::UniformDigits,
            SynthProfile::SmallValues,
            SynthProfile::Chained,
        ] {
            let trace = SyntheticTrace::new(profile, 13, 64, 94).unwrap();
            assert_eq!(trace.digit_columns(), 29);
            assert_eq!(trace.trace_words(), 8);
            for column in 21..24 {
                assert!(trace.by_row(column));
                assert_eq!(trace.bits(column), 5);
                for row in 0..trace.bytecode_rows() {
                    assert_eq!(
                        trace.row_digit(column, row),
                        Some(((trace.bytecode_word(0, row) >> (5 * (column - 21))) & 31) as usize)
                    );
                }
            }
            for cycle in 0..trace.cycles() {
                let keys_differ = trace.digit(18, cycle).is_some();
                assert_eq!(
                    trace.trace_word(6, cycle),
                    if keys_differ {
                        0
                    } else {
                        trace.trace_word(0, cycle)
                    }
                );
                assert_eq!(
                    trace.trace_word(7, cycle),
                    if keys_differ {
                        trace.trace_word(1, cycle)
                    } else {
                        0
                    }
                );
                for column in 24..29 {
                    assert_eq!(trace.bits(column), 4);
                    assert_eq!(
                        trace.digit(column, cycle),
                        if trace.digit(14, cycle).is_some() {
                            trace.digit(column - 19, cycle)
                        } else {
                            Some(0)
                        }
                    );
                }
                if profile == SynthProfile::Chained {
                    if cycle != 0 {
                        assert_eq!(trace.digit(21, cycle), trace.digit(23, cycle - 1));
                        if trace.digit(17, cycle - 1).is_some() {
                            for column in 5..10 {
                                assert_eq!(
                                    trace.digit(column, cycle),
                                    trace.digit(column, cycle - 1)
                                );
                            }
                        }
                    }
                    for column in 5..10 {
                        let start = SyntheticTrace::indicator_start(column).unwrap();
                        let digit = trace.digit(column, cycle).unwrap();
                        for value in 1..16 {
                            let bit = start + value - 1;
                            assert_eq!(
                                (trace.rows()[cycle][bit / 64] >> (bit % 64)) & 1,
                                u64::from(digit == value)
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn malformed_memory_groups_return_typed_geometry_errors() {
        let reject = |widths: &[usize],
                      registers: Vec<usize>,
                      store: Vec<usize>,
                      modify: Option<(usize, usize)>| {
            let mut source = Fixture::new(3, widths);
            if let Some((column, width)) = modify {
                source.widths[column] = width;
            }
            let source = Arc::new(source);
            let (_, ram, registers, store) =
                source.clone().groups(source.ram.clone(), registers, store);
            MemoryTrace::new(ram, registers, store).unwrap_err()
        };
        assert_eq!(MAX_ADDRESS_BITS, 32);
        assert_eq!(
            reject(&[], vec![21, 22, 23], vec![17], None),
            MemoryError::AddressBits { bits: 0 }
        );
        assert_eq!(
            reject(&[8, 8, 8, 8, 1], vec![21, 22, 23], vec![17], None),
            MemoryError::UnsupportedAddressBits {
                bits: 33,
                supported: 32
            }
        );
        assert_eq!(
            reject(&[1], vec![21, 22], vec![17], None),
            MemoryError::RegisterColumns { columns: 2 }
        );
        assert_eq!(
            reject(&[1], vec![21, 22, 23], vec![17], Some((21, 4))),
            MemoryError::RegisterWidth {
                column: 21,
                bits: 4
            }
        );
        assert_eq!(
            reject(&[1], vec![21, 22, 23], vec![17, 17], None),
            MemoryError::StoreColumns { columns: 2 }
        );
        assert_eq!(
            reject(&[1], vec![21, 22, 23], vec![17], Some((17, 1))),
            MemoryError::StoreWidth {
                column: 17,
                bits: 1
            }
        );
        let source = Arc::new(Fixture::new(3, &[1]));
        let (_, ram, _, store) = source.groups(vec![24], vec![21, 22, 23], vec![17]);
        let (_, _, registers, _) =
            Arc::new(Fixture::new(2, &[1])).groups(vec![24], vec![21, 22, 23], vec![17]);
        assert_eq!(
            MemoryTrace::new(ram, registers, store).unwrap_err(),
            MemoryError::Cycles {
                group: "registers",
                expected: 8,
                actual: 4
            }
        );
    }

    #[test]
    fn ram_chunks_require_unique_trace_ownership() {
        let source = Arc::new(Fixture::new(3, &[4, 4]));
        let (_, memory) = source.prepared();
        let memory = Arc::new(memory);
        let other = memory.clone();
        assert_eq!(
            MemoryTrace::into_ram_chunks(other).unwrap_err(),
            MemoryError::InUse { handles: 1 }
        );
        assert_eq!(MemoryTrace::into_ram_chunks(memory).unwrap().cycles(), 8);
    }

    #[test]
    fn views_reject_wrong_words_and_point_dimensions() {
        let source = Arc::new(Fixture::new(3, &[4, 4]));
        let (trace, memory) = source.clone().prepared();
        let lift = WordLift::new(&[F128::from_raw(1); 64]);
        assert_eq!(
            inc_lift(&trace, source.trace_words(), &lift).unwrap_err(),
            MemoryError::Word {
                word: source.trace_words(),
                words: source.trace_words()
            }
        );
        assert_eq!(
            fold_words(&[0; 12], &lift, &[]).unwrap_err(),
            MemoryError::Words {
                words: 12,
                point: 0
            }
        );
        assert_eq!(
            fold_words(&[0; 4], &lift, &[F128::from_raw(1); 3]).unwrap_err(),
            MemoryError::Words { words: 4, point: 3 }
        );
        assert_eq!(
            address_column(&memory, &[F128::from_raw(1); 9]).unwrap_err(),
            MemoryError::PointLength {
                expected: 8,
                actual: 9
            }
        );
        let plan = ScatterPlan::new(trace).unwrap();
        assert_eq!(
            row_weights(&plan, RowWeight::Eq(&[F128::from_raw(1); 2])).unwrap_err(),
            MemoryError::PointLength {
                expected: 3,
                actual: 2
            }
        );
    }

    fn allocation_measurements(pool: &ThreadPool, log_t: usize) -> [usize; 5] {
        pool.install(|| {
            let source = Arc::new(Fixture::new(log_t, &[2]));
            let (trace, memory) = source.clone().prepared();
            let plan = ScatterPlan::new(trace.clone()).unwrap();
            let weights = std::array::from_fn(|bit| F128::from_raw(bit as u128 + 3));
            let lift = WordLift::new(&weights);
            let words: Vec<_> = (0..source.cycles())
                .map(|cycle| source.trace_word(5, cycle))
                .collect();
            let point: Vec<_> = (0..log_t)
                .map(|coordinate| F128::from_raw(coordinate as u128 + 2))
                .collect();
            let address_point = [F128::from_raw(2), F128::from_raw(3)];
            let view = |which| match which {
                0 => inc_lift(&trace, 5, &lift).unwrap(),
                1 => fold_words(&words, &lift, &point[..8]).unwrap(),
                2 => address_column(&memory, &address_point).unwrap(),
                3 => row_weights(&plan, RowWeight::Eq(&point)).unwrap(),
                _ => row_weights(&plan, RowWeight::Next(&point)).unwrap(),
            };
            for which in 0..5 {
                drop(view(which));
            }
            let runtime_bytes = RAYON_WORKER_ALLOWANCE.bytes * pool.current_num_threads();
            std::array::from_fn(|which| {
                let before = CountingAllocator::live_bytes();
                let measurement = AllocationMeasurement::begin();
                let output = view(which);
                let AllocationStats {
                    final_bytes,
                    allocs,
                    peak_bytes,
                } = measurement.finish();
                let output_bytes = output.capacity() * size_of::<F128>();
                assert!(
                    final_bytes <= output_bytes + runtime_bytes,
                    "view {which}, log_t {log_t}: retained {final_bytes}, output {output_bytes}"
                );
                assert!((before..=before + output_bytes + runtime_bytes)
                    .contains(&CountingAllocator::live_bytes()));
                assert!(peak_bytes >= output_bytes);
                allocs
            })
        })
    }

    #[test]
    fn views_allocate_only_outputs_and_bounded_scratch() {
        let one = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        for log_t in [8, 14] {
            for (view, allocations) in allocation_measurements(&one, log_t).into_iter().enumerate()
            {
                assert!(
                    allocations <= 16,
                    "view {view}, log_t {log_t}: {allocations} allocations"
                );
            }
        }
        let twelve = ThreadPoolBuilder::new().num_threads(12).build().unwrap();
        let small = allocation_measurements(&twelve, 13);
        let large = allocation_measurements(&twelve, 19);
        let allowance = RAYON_WORKER_ALLOWANCE.allocs * twelve.current_num_threads();
        for (view, (small, large)) in small.into_iter().zip(large).enumerate() {
            assert!(
                large <= small + allowance,
                "view {view}: allocation growth {small} -> {large}"
            );
        }
    }
}
