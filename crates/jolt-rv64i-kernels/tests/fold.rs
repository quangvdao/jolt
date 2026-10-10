#![cfg(feature = "test-utils")]

#[path = "../benches/support/allocator.rs"]
mod allocator;

#[expect(
    clippy::unwrap_used,
    reason = "fixtures and contract assertions must fail the test"
)]
mod tests {
    use super::allocator::{AllocationMeasurement, CountingAllocator};
    use jolt_field::F128;
    use jolt_rv64i_kernels::oracle::mle_at;
    use jolt_rv64i_kernels::packed::scatter::ScatterPlan;
    use jolt_rv64i_kernels::router::fold::{fold_pass, FoldLayout, FoldOutput};
    use jolt_rv64i_kernels::router::shape::{
        selector_counts, synthetic_router_shapes, BitEntry, RouterError, RouterShape,
        SelectorFactor, SlotVariable, WordSlot,
    };
    use jolt_rv64i_kernels::source::{CycleSource, ValidatedTrace};
    use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
    use rand_chacha::rand_core::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use rayon::ThreadPoolBuilder;
    use std::mem::size_of;
    use std::sync::Arc;
    use std::time::Duration;

    const ZERO: F128 = F128::from_raw(0);
    const ONE: F128 = F128::from_raw(1);

    #[derive(Clone)]
    struct Trace {
        words: Vec<[u64; 6]>,
        bytecode: Vec<[u64; 4]>,
        indices: Vec<usize>,
        widths: Vec<usize>,
        by_row: Vec<bool>,
        digits: Vec<Vec<Option<usize>>>,
        row_digits: Vec<Vec<Option<usize>>>,
    }

    impl Trace {
        fn synthetic(log_t: usize, seed: u64) -> Self {
            let source = SyntheticTrace::new(
                SynthProfile::AllRows,
                log_t,
                1 << log_t.saturating_sub(2),
                seed,
            )
            .unwrap();
            let cycles = CycleSource::cycles(&source);
            let rows = source.bytecode_rows();
            let columns = source.digit_columns();
            let mut trace = Self {
                words: (0..cycles)
                    .map(|j| std::array::from_fn(|word| source.trace_word(word, j)))
                    .collect(),
                bytecode: (0..rows)
                    .map(|row| std::array::from_fn(|word| source.bytecode_word(word, row)))
                    .collect(),
                indices: (0..cycles).map(|j| source.bytecode_index(j)).collect(),
                widths: (0..columns).map(|c| source.bits(c)).collect(),
                by_row: (0..columns).map(|c| source.by_row(c)).collect(),
                digits: (0..columns)
                    .map(|c| (0..cycles).map(|j| source.digit(c, j)).collect())
                    .collect(),
                row_digits: (0..columns)
                    .map(|c| (0..rows).map(|row| source.row_digit(c, row)).collect())
                    .collect(),
            };
            for column in 5..12 {
                for j in 0..cycles {
                    if (j + column).is_multiple_of(7) {
                        trace.digits[column][j] = None;
                    }
                }
            }
            for shape in synthetic_router_shapes().unwrap() {
                let mut seen = vec![false; shape.selectors()];
                for j in 0..cycles {
                    if let Some(h) = selected(&trace, &shape, j) {
                        if !seen[h] {
                            seen[h] = true;
                            trace.words[j].fill(u64::MAX);
                            trace.bytecode[trace.indices[j]].fill(u64::MAX);
                        }
                    }
                }
            }
            trace
        }

        fn small(present: Vec<Option<usize>>) -> Self {
            Self {
                words: vec![[0; 6]; present.len()],
                bytecode: vec![[0; 4]; 1],
                indices: vec![0; present.len()],
                widths: vec![0],
                by_row: vec![false],
                digits: vec![present],
                row_digits: vec![vec![None]],
            }
        }
    }

    impl CycleSource for Trace {
        fn cycles(&self) -> usize {
            self.words.len()
        }
        fn trace_words(&self) -> usize {
            6
        }
        fn trace_word(&self, word: usize, j: usize) -> u64 {
            self.words
                .get(j)
                .and_then(|words| words.get(word))
                .copied()
                .unwrap_or(0)
        }
        fn bytecode_rows(&self) -> usize {
            self.bytecode.len()
        }
        fn bytecode_words(&self) -> usize {
            4
        }
        fn bytecode_word(&self, word: usize, row: usize) -> u64 {
            self.bytecode
                .get(row)
                .and_then(|words| words.get(word))
                .copied()
                .unwrap_or(0)
        }
        fn bytecode_index(&self, j: usize) -> usize {
            self.indices.get(j).copied().unwrap_or(0)
        }
        fn digit_columns(&self) -> usize {
            self.widths.len()
        }
        fn bits(&self, c: usize) -> usize {
            self.widths.get(c).copied().unwrap_or(0)
        }
        fn by_row(&self, c: usize) -> bool {
            self.by_row.get(c).copied().unwrap_or(false)
        }
        fn digit(&self, c: usize, j: usize) -> Option<usize> {
            self.digits
                .get(c)
                .and_then(|digits| digits.get(j))
                .copied()
                .flatten()
        }
        fn row_digit(&self, c: usize, row: usize) -> Option<usize> {
            self.row_digits
                .get(c)
                .and_then(|digits| digits.get(row))
                .copied()
                .flatten()
        }
    }

    fn random_field(rng: &mut ChaCha20Rng) -> F128 {
        F128::from_raw(u128::from(rng.next_u64()) | u128::from(rng.next_u64()) << 64)
    }

    fn selected<S: CycleSource>(source: &S, shape: &RouterShape, j: usize) -> Option<usize> {
        let mut h = 0;
        let mut stride = 1;
        for factor in shape.factors() {
            h += stride * source.digit(factor.column, j)?;
            stride *= 1 << factor.slots.len();
        }
        Some(h)
    }

    fn word_at<S: CycleSource>(source: &S, word: &WordSlot, j: usize) -> u64 {
        match word {
            WordSlot::Trace(word) => source.trace_word(*word, j),
            WordSlot::Bytecode(word) => source.bytecode_word(*word, source.bytecode_index(j)),
            WordSlot::Zero => 0,
            WordSlot::Bits(entries) => entries.iter().enumerate().fold(0, |word, (bit, entry)| {
                let value = match entry {
                    BitEntry::One => true,
                    BitEntry::Zero => false,
                    BitEntry::Indicator { column, value } => {
                        source.digit(*column, j) == Some(*value)
                    }
                    BitEntry::DigitBit { column, bit } => source
                        .digit(*column, j)
                        .is_some_and(|digit| digit & (1 << bit) != 0),
                };
                word | (u64::from(value) << bit)
            }),
        }
    }

    fn expected<S: CycleSource>(
        source: &S,
        shapes: &[RouterShape],
        point: &[F128],
        columns: &[usize],
    ) -> FoldOutput {
        let weights: Vec<_> = (0..source.cycles())
            .map(|j| {
                if source.cycles() <= 256 {
                    let mut indicator = vec![ZERO; source.cycles()];
                    indicator[j] = ONE;
                    mle_at(&indicator, point).unwrap()
                } else {
                    point.iter().enumerate().fold(ONE, |weight, (bit, &r)| {
                        weight * (ONE + r + F128::from_raw(((j >> bit) & 1) as u128))
                    })
                }
            })
            .collect();
        let folds = shapes
            .iter()
            .map(|shape| {
                let sources = shape.bank().len() * 64;
                let mut canonical = vec![ZERO; sources * shape.selectors()];
                for (j, &e) in weights.iter().enumerate() {
                    if let Some(h) = selected(source, shape, j) {
                        for (word_index, word) in shape.bank().iter().enumerate() {
                            let mut value = word_at(source, word, j);
                            while value != 0 {
                                let bit = value.trailing_zeros() as usize;
                                canonical[h * sources + word_index * 64 + bit] += e;
                                value &= value - 1;
                            }
                        }
                    }
                }
                (0..shape.fold_len())
                    .map(|index| {
                        let mut source_index = 0;
                        let mut digits = [0; 3];
                        for (position, &(_, variable)) in shape.slot_map().iter().enumerate() {
                            let value = (index >> position) & 1;
                            match variable {
                                SlotVariable::Bit(bit) => source_index |= value << bit,
                                SlotVariable::Word(bit) => source_index |= value << (6 + bit),
                                SlotVariable::Selector { factor, bit } => {
                                    digits[factor] |= value << bit;
                                }
                            }
                        }
                        let mut h = 0;
                        let mut stride = 1;
                        for (factor, digit) in shape.factors().iter().zip(digits) {
                            h += digit * stride;
                            stride *= 1 << factor.slots.len();
                        }
                        canonical[source_index + h * sources]
                    })
                    .collect()
            })
            .collect();
        let mut ra_fold = vec![ZERO; source.bytecode_rows()];
        let mut histograms: Vec<_> = columns
            .iter()
            .map(|&c| vec![ZERO; 1 << source.bits(c)])
            .collect();
        for (j, &e) in weights.iter().enumerate() {
            ra_fold[source.bytecode_index(j)] += e;
            for (&column, histogram) in columns.iter().zip(&mut histograms) {
                if let Some(digit) = source.digit(column, j) {
                    histogram[digit] += e;
                }
            }
        }
        FoldOutput {
            folds,
            ra_fold,
            histograms,
        }
    }

    fn check_output(actual: &FoldOutput, expected: &FoldOutput) {
        assert_eq!(actual.folds, expected.folds);
        assert_eq!(actual.ra_fold, expected.ra_fold);
        assert_eq!(actual.histograms, expected.histograms);
    }

    fn with_routes(shape: &RouterShape, rng: &mut ChaCha20Rng) -> RouterShape {
        let route = (0..16)
            .map(|_| {
                (
                    rng.next_u32() as usize & ((1 << shape.log_outputs()) - 1),
                    rng.next_u32() as usize & (shape.bank().len() * 64 - 1),
                    rng.next_u32() as usize & (shape.selectors() - 1),
                )
            })
            .collect();
        RouterShape::new(
            shape.slots(),
            shape.bank().to_vec(),
            shape.factors().to_vec(),
            [0, 1, 2, 3, 4, 5],
            shape.word_slots().to_vec(),
            shape.log_outputs(),
            route,
        )
        .unwrap()
    }

    #[test]
    fn complete_fold_matches_definitions_for_all_shapes_routes_points_and_layouts() {
        ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                let mut rng = ChaCha20Rng::seed_from_u64(818);
                for log_t in 3..=8 {
                    let source = Arc::new(Trace::synthetic(log_t, log_t as u64 + 821));
                    let trace = Arc::new(ValidatedTrace::new(Arc::clone(&source)).unwrap());
                    let plan = ScatterPlan::new(Arc::clone(&trace)).unwrap();
                    let mut shapes = synthetic_router_shapes().unwrap();
                    shapes.push(
                        RouterShape::new(
                            9,
                            vec![WordSlot::Bits(vec![
                                BitEntry::Indicator {
                                    column: 5,
                                    value: 0,
                                },
                                BitEntry::DigitBit { column: 5, bit: 3 },
                                BitEntry::One,
                                BitEntry::Zero,
                                BitEntry::Indicator {
                                    column: 18,
                                    value: 0,
                                },
                            ])],
                            vec![SelectorFactor {
                                column: 10,
                                slots: vec![6, 7, 8],
                            }],
                            [0, 1, 2, 3, 4, 5],
                            vec![],
                            0,
                            vec![],
                        )
                        .unwrap(),
                    );
                    let columns: Vec<_> = (0..source.digit_columns()).collect();
                    let points = [
                        (0..log_t)
                            .map(|_| random_field(&mut rng))
                            .collect::<Vec<_>>(),
                        (0..log_t)
                            .map(|_| F128::from_raw(u128::from(rng.next_u32() & 1)))
                            .collect(),
                    ];
                    for point in points {
                        let oracle = expected(source.as_ref(), &shapes, &point, &columns);
                        for routed in [false, true] {
                            let shapes: Vec<_> = shapes
                                .iter()
                                .map(|shape| {
                                    if routed {
                                        with_routes(shape, &mut rng)
                                    } else {
                                        shape.clone()
                                    }
                                })
                                .collect();
                            for choice in 0..3 {
                                let values: Vec<_> = shapes
                                    .iter()
                                    .map(|shape| match choice {
                                        0 => vec![],
                                        1 => {
                                            (0..shape.selectors()).filter(|h| h % 3 == 0).collect()
                                        }
                                        _ => (0..shape.selectors()).collect(),
                                    })
                                    .collect();
                                let layout = FoldLayout::new(&trace, &shapes, &values).unwrap();
                                check_output(
                                    &fold_pass(&trace, &shapes, &point, &plan, &layout, &columns)
                                        .unwrap(),
                                    &oracle,
                                );
                            }
                        }
                    }
                    for shape in &shapes {
                        let mut counts = vec![0; shape.selectors()];
                        for j in 0..source.cycles() {
                            if let Some(h) = selected(source.as_ref(), shape, j) {
                                counts[h] += 1;
                            }
                        }
                        assert_eq!(selector_counts(&trace, shape).unwrap(), counts);
                    }
                }
            });
    }

    fn small_shape(word: WordSlot) -> RouterShape {
        RouterShape::new(
            6,
            vec![word],
            vec![SelectorFactor {
                column: 0,
                slots: vec![],
            }],
            [0, 1, 2, 3, 4, 5],
            vec![],
            0,
            vec![(0, 0, 0)],
        )
        .unwrap()
    }

    #[test]
    fn complete_fold_smallest_case_keeps_unrouted_bit_one() {
        let mut source = Trace::small(vec![Some(0); 8]);
        let mut rng = ChaCha20Rng::seed_from_u64(826);
        for words in &mut source.words {
            words[0] = rng.next_u64();
        }
        source.words[0][0] = 2;
        let trace = Arc::new(ValidatedTrace::new(Arc::new(source)).unwrap());
        let plan = ScatterPlan::new(Arc::clone(&trace)).unwrap();
        let shapes = vec![small_shape(WordSlot::Trace(0))];
        let layout = FoldLayout::new(&trace, &shapes, &[vec![]]).unwrap();
        let result = fold_pass(&trace, &shapes, &[ZERO; 3], &plan, &layout, &[0]).unwrap();
        let mut literal = vec![ZERO; 64];
        literal[1] = ONE;
        assert_eq!(result.folds, vec![literal]);
        assert_eq!(result.ra_fold, vec![ONE]);
        assert_eq!(result.histograms, vec![vec![ONE]]);
    }

    #[test]
    fn complete_fold_constant_bank_includes_present_factor_totals() {
        let mut rng = ChaCha20Rng::seed_from_u64(827);
        for digits in [
            vec![Some(0); 8],
            vec![None, Some(0), None, Some(0), Some(0), None, Some(0), None],
        ] {
            let all_present = digits.iter().all(Option::is_some);
            let source = Arc::new(Trace::small(digits));
            let trace = Arc::new(ValidatedTrace::new(Arc::clone(&source)).unwrap());
            let plan = ScatterPlan::new(Arc::clone(&trace)).unwrap();
            let shapes = vec![small_shape(WordSlot::Bits(vec![BitEntry::One]))];
            for point in [
                (0..3).map(|_| random_field(&mut rng)).collect::<Vec<_>>(),
                vec![ONE, ZERO, ONE],
            ] {
                let oracle = expected(source.as_ref(), &shapes, &point, &[0]);
                if all_present {
                    let mut literal = vec![ZERO; 64];
                    literal[0] = ONE;
                    assert_eq!(oracle.folds, vec![literal]);
                }
                for values in [vec![], vec![0]] {
                    let layout = FoldLayout::new(&trace, &shapes, &[values]).unwrap();
                    check_output(
                        &fold_pass(&trace, &shapes, &point, &plan, &layout, &[0]).unwrap(),
                        &oracle,
                    );
                }
            }
        }
    }

    #[test]
    fn complete_fold_row_only_words_preserve_constant_and_cycle_digit_bits() {
        let mut source = Trace::small(vec![Some(0); 8]);
        source.by_row[0] = true;
        source.row_digits[0][0] = Some(0);
        source.bytecode[0][0] = 0x8123_4567_89ab_cdef;
        source.widths.push(0);
        source.by_row.push(false);
        source.digits.push(
            (0..8)
                .map(|j| if j % 2 == 0 { None } else { Some(0) })
                .collect(),
        );
        source.row_digits.push(vec![None]);
        let source = Arc::new(source);
        let trace = Arc::new(ValidatedTrace::new(Arc::clone(&source)).unwrap());
        let plan = ScatterPlan::new(Arc::clone(&trace)).unwrap();
        let shapes = vec![RouterShape::new(
            7,
            vec![
                WordSlot::Bytecode(0),
                WordSlot::Bits(vec![
                    BitEntry::One,
                    BitEntry::Indicator {
                        column: 1,
                        value: 0,
                    },
                ]),
            ],
            vec![SelectorFactor {
                column: 0,
                slots: vec![],
            }],
            [0, 1, 2, 3, 4, 5],
            vec![6],
            0,
            vec![],
        )
        .unwrap()];
        let mut rng = ChaCha20Rng::seed_from_u64(832);
        for point in [
            (0..3).map(|_| random_field(&mut rng)).collect::<Vec<_>>(),
            vec![ZERO; 3],
            vec![ONE, ZERO, ZERO],
        ] {
            let oracle = expected(source.as_ref(), &shapes, &point, &[0, 1]);
            for values in [vec![], vec![0]] {
                let layout = FoldLayout::new(&trace, &shapes, &[values]).unwrap();
                check_output(
                    &fold_pass(&trace, &shapes, &point, &plan, &layout, &[0, 1]).unwrap(),
                    &oracle,
                );
            }
        }
    }

    #[test]
    fn complete_fold_two_chunks_matches_one_oracle_on_one_and_twelve_threads() {
        let source = Arc::new(Trace::synthetic(13, 828));
        let trace = Arc::new(ValidatedTrace::new(Arc::clone(&source)).unwrap());
        let shapes = synthetic_router_shapes().unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(829);
        let point: Vec<_> = (0..13).map(|_| random_field(&mut rng)).collect();
        let columns: Vec<_> = (0..source.digit_columns()).collect();
        let oracle = expected(source.as_ref(), &shapes, &point, &columns);
        for threads in [1, 12] {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let plan = ScatterPlan::new(Arc::clone(&trace)).unwrap();
                    let values: Vec<_> = shapes
                        .iter()
                        .map(|shape| (0..shape.selectors().min(8)).collect())
                        .collect();
                    let layout = FoldLayout::new(&trace, &shapes, &values).unwrap();
                    check_output(
                        &fold_pass(&trace, &shapes, &point, &plan, &layout, &columns).unwrap(),
                        &oracle,
                    );
                });
        }
    }

    fn output_bytes(output: &FoldOutput) -> usize {
        output.folds.capacity() * size_of::<Vec<F128>>()
            + output.histograms.capacity() * size_of::<Vec<F128>>()
            + (output.ra_fold.capacity()
                + output.folds.iter().map(Vec::capacity).sum::<usize>()
                + output.histograms.iter().map(Vec::capacity).sum::<usize>())
                * size_of::<F128>()
    }

    #[test]
    fn fold_allocation_count_peak_and_returned_bytes_meet_pass_budget() {
        for (log_t, threads) in [(8, 1), (14, 1), (14, 12)] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            // Contended mutexes can initialize persistent thread parking state on any worker.
            std::thread::park_timeout(Duration::from_nanos(1));
            drop(pool.broadcast(|_| std::thread::park_timeout(Duration::from_nanos(1))));
            pool.install(|| {
                let trace =
                    Arc::new(ValidatedTrace::new(Arc::new(Trace::synthetic(log_t, 830))).unwrap());
                let shapes = synthetic_router_shapes().unwrap();
                let plan = ScatterPlan::new(Arc::clone(&trace)).unwrap();
                let values: Vec<_> = shapes.iter().map(|_| vec![]).collect();
                let layout = FoldLayout::new(&trace, &shapes, &values).unwrap();
                let point = vec![F128::from_raw(7); log_t];
                let columns: Vec<_> = (0..trace.source().digit_columns()).collect();
                drop(fold_pass(&trace, &shapes, &point, &plan, &layout, &columns).unwrap());
                let baseline = CountingAllocator::live_bytes();
                let measurement = AllocationMeasurement::begin();
                let result = fold_pass(&trace, &shapes, &point, &plan, &layout, &columns).unwrap();
                let stats = measurement.finish();
                let returned = output_bytes(&result);
                if threads == 1 {
                    assert!(stats.allocs <= 256, "{} allocations", stats.allocs);
                }
                assert_eq!(stats.final_bytes, returned);
                assert_eq!(CountingAllocator::live_bytes(), baseline + returned);
                let hist_entries = columns
                    .iter()
                    .map(|&c| 1 << trace.source().bits(c))
                    .sum::<usize>();
                let buckets = (layout.entries() + hist_entries) * threads * size_of::<F128>();
                let scatter = trace.source().cycles() * size_of::<F128>();
                assert!(
                    stats.peak_bytes <= returned + buckets + scatter,
                    "peak {} exceeds {}",
                    stats.peak_bytes,
                    returned + buckets + scatter
                );
                drop(result);
                assert_eq!(CountingAllocator::live_bytes(), baseline);
            });
        }
    }

    #[test]
    fn router_shape_rejects_malformed_banks_slots_factors_routes_and_dimensions() {
        let factor = SelectorFactor {
            column: 0,
            slots: vec![],
        };
        let make = |slots, bank, factors, bit_slots, word_slots, outputs, route| {
            RouterShape::new(slots, bank, factors, bit_slots, word_slots, outputs, route)
        };
        assert_eq!(
            make(
                8,
                vec![WordSlot::Zero; 3],
                vec![factor.clone()],
                [0, 1, 2, 3, 4, 5],
                vec![6],
                0,
                vec![]
            ),
            Err(RouterError::BankLength { len: 3 })
        );
        assert_eq!(
            make(
                6,
                vec![WordSlot::Bits(vec![BitEntry::Zero; 65])],
                vec![factor.clone()],
                [0, 1, 2, 3, 4, 5],
                vec![],
                0,
                vec![]
            ),
            Err(RouterError::BitsEntries {
                slot: 0,
                entries: 65
            })
        );
        for count in [0, 4] {
            assert_eq!(
                make(
                    6,
                    vec![WordSlot::Zero],
                    vec![factor.clone(); count],
                    [0, 1, 2, 3, 4, 5],
                    vec![],
                    0,
                    vec![]
                ),
                Err(RouterError::Factors { count })
            );
        }
        assert_eq!(
            make(
                8,
                vec![WordSlot::Zero; 4],
                vec![factor.clone()],
                [0, 1, 2, 3, 4, 5],
                vec![6],
                0,
                vec![]
            ),
            Err(RouterError::WordVariables {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            make(
                7,
                vec![WordSlot::Zero; 2],
                vec![factor.clone()],
                [0, 1, 2, 3, 4, 5],
                vec![5],
                0,
                vec![]
            ),
            Err(RouterError::SlotRepeated { slot: 5 })
        );
        assert_eq!(
            make(
                7,
                vec![WordSlot::Zero; 2],
                vec![factor.clone()],
                [0, 1, 2, 3, 4, 5],
                vec![7],
                0,
                vec![]
            ),
            Err(RouterError::SlotRange { slot: 7, slots: 7 })
        );
        assert_eq!(
            make(
                7,
                vec![WordSlot::Zero],
                vec![factor.clone()],
                [6, 1, 2, 3, 4, 5],
                vec![],
                0,
                vec![]
            ),
            Err(RouterError::BitSlot { bit: 0, slot: 6 })
        );
        for (triple, axis, bound) in [
            ((1, 0, 0), "output", 1),
            ((0, 64, 0), "source", 64),
            ((0, 0, 1), "selector", 1),
        ] {
            assert_eq!(
                make(
                    6,
                    vec![WordSlot::Zero],
                    vec![factor.clone()],
                    [0, 1, 2, 3, 4, 5],
                    vec![],
                    0,
                    vec![triple]
                ),
                Err(RouterError::Route {
                    triple,
                    axis,
                    bound
                })
            );
        }
        let variables = usize::BITS as usize - 5;
        assert_eq!(
            make(
                variables,
                vec![WordSlot::Zero],
                vec![factor.clone()],
                [0, 1, 2, 3, 4, 5],
                vec![],
                0,
                vec![]
            ),
            Err(RouterError::Dimension { variables })
        );
        assert_eq!(
            make(
                6,
                vec![WordSlot::Zero],
                vec![factor],
                [0, 1, 2, 3, 4, 5],
                vec![],
                variables,
                vec![]
            ),
            Err(RouterError::Dimension { variables })
        );
    }

    #[test]
    fn router_source_validation_names_invalid_words_columns_widths_and_entries() {
        let trace = ValidatedTrace::new(Arc::new(Trace::synthetic(3, 831))).unwrap();
        for (word, error) in [
            (
                WordSlot::Trace(6),
                RouterError::WordIndex {
                    bank: "trace",
                    index: 6,
                    words: 6,
                },
            ),
            (
                WordSlot::Bytecode(4),
                RouterError::WordIndex {
                    bank: "bytecode",
                    index: 4,
                    words: 4,
                },
            ),
            (
                WordSlot::Bits(vec![BitEntry::Indicator {
                    column: 21,
                    value: 0,
                }]),
                RouterError::Column {
                    column: 21,
                    columns: 21,
                },
            ),
            (
                WordSlot::Bits(vec![BitEntry::DigitBit { column: 21, bit: 0 }]),
                RouterError::Column {
                    column: 21,
                    columns: 21,
                },
            ),
            (
                WordSlot::Bits(vec![BitEntry::Indicator {
                    column: 5,
                    value: 16,
                }]),
                RouterError::Entry {
                    column: 5,
                    kind: "indicator",
                    value: 16,
                    bound: 16,
                },
            ),
            (
                WordSlot::Bits(vec![BitEntry::DigitBit { column: 5, bit: 4 }]),
                RouterError::Entry {
                    column: 5,
                    kind: "digit bit",
                    value: 4,
                    bound: 4,
                },
            ),
        ] {
            let shape = RouterShape::new(
                6,
                vec![word],
                vec![SelectorFactor {
                    column: 18,
                    slots: vec![],
                }],
                [0, 1, 2, 3, 4, 5],
                vec![],
                0,
                vec![],
            )
            .unwrap();
            assert_eq!(selector_counts(&trace, &shape), Err(error));
        }
        let column = RouterShape::new(
            6,
            vec![WordSlot::Zero],
            vec![SelectorFactor {
                column: 21,
                slots: vec![],
            }],
            [0, 1, 2, 3, 4, 5],
            vec![],
            0,
            vec![],
        )
        .unwrap();
        assert_eq!(
            selector_counts(&trace, &column),
            Err(RouterError::Column {
                column: 21,
                columns: 21
            })
        );
        let width = RouterShape::new(
            9,
            vec![WordSlot::Zero],
            vec![SelectorFactor {
                column: 5,
                slots: vec![6, 7, 8],
            }],
            [0, 1, 2, 3, 4, 5],
            vec![],
            0,
            vec![],
        )
        .unwrap();
        assert_eq!(
            selector_counts(&trace, &width),
            Err(RouterError::FactorWidth {
                column: 5,
                expected: 4,
                actual: 3
            })
        );
    }

    #[test]
    fn fold_rejects_points_layouts_histogram_columns_and_scatter_dimensions() {
        let trace =
            Arc::new(ValidatedTrace::new(Arc::new(Trace::small(vec![Some(0); 8]))).unwrap());
        let shapes = vec![small_shape(WordSlot::Trace(0))];
        let plan = ScatterPlan::new(Arc::clone(&trace)).unwrap();
        let layout = FoldLayout::new(&trace, &shapes, &[vec![]]).unwrap();
        assert_eq!(
            fold_pass(&trace, &shapes, &[ZERO; 4], &plan, &layout, &[]).unwrap_err(),
            RouterError::PointLength {
                expected: 3,
                actual: 4
            }
        );
        assert_eq!(
            FoldLayout::new(&trace, &shapes, &[]).unwrap_err(),
            RouterError::TableLength {
                table: "layout shapes",
                expected: 1,
                actual: 0
            }
        );
        for selector in [0, 1] {
            let values = if selector == 0 { vec![0, 0] } else { vec![1] };
            assert_eq!(
                FoldLayout::new(&trace, &shapes, &[values]).unwrap_err(),
                RouterError::Layout {
                    shape: 0,
                    selector,
                    bound: 1
                }
            );
        }
        assert_eq!(
            fold_pass(&trace, &shapes, &[ZERO; 3], &plan, &layout, &[1]).unwrap_err(),
            RouterError::Column {
                column: 1,
                columns: 1
            }
        );
        let empty = FoldLayout::new(&trace, &[], &[]).unwrap();
        assert_eq!(
            fold_pass(&trace, &shapes, &[ZERO; 3], &plan, &empty, &[]).unwrap_err(),
            RouterError::TableLength {
                table: "layout shapes",
                expected: 1,
                actual: 0
            }
        );
        let other_shape = vec![small_shape(WordSlot::Bits(vec![BitEntry::One]))];
        let geometry = FoldLayout::new(&trace, &other_shape, &[vec![]]).unwrap();
        assert_eq!(
            fold_pass(&trace, &shapes, &[ZERO; 3], &plan, &geometry, &[]).unwrap_err(),
            RouterError::TableLength {
                table: "layout geometry",
                expected: layout.entries(),
                actual: geometry.entries()
            }
        );
        let shorter =
            Arc::new(ValidatedTrace::new(Arc::new(Trace::small(vec![Some(0); 4]))).unwrap());
        let shorter_plan = ScatterPlan::new(shorter).unwrap();
        assert_eq!(
            fold_pass(&trace, &shapes, &[ZERO; 3], &shorter_plan, &layout, &[]).unwrap_err(),
            RouterError::TableLength {
                table: "scatter cycles",
                expected: 8,
                actual: 4
            }
        );
        let mut more_rows = Trace::small(vec![Some(0); 8]);
        more_rows.bytecode.push([0; 4]);
        more_rows.row_digits[0].push(None);
        let more_rows = Arc::new(ValidatedTrace::new(Arc::new(more_rows)).unwrap());
        let more_rows_plan = ScatterPlan::new(more_rows).unwrap();
        assert_eq!(
            fold_pass(&trace, &shapes, &[ZERO; 3], &more_rows_plan, &layout, &[]).unwrap_err(),
            RouterError::TableLength {
                table: "scatter rows",
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn fold_rejects_unrepresentable_bucket_and_histogram_storage() {
        let mut source = Trace::small(vec![Some(0); 8]);
        source.widths.extend([usize::BITS as usize - 6, 1]);
        source.by_row.extend([false, false]);
        source.digits.extend([vec![None; 8], vec![Some(0); 8]]);
        source.row_digits.extend([vec![None], vec![None]]);
        let trace = Arc::new(ValidatedTrace::new(Arc::new(source)).unwrap());
        let shape = RouterShape::new(
            7,
            vec![WordSlot::Bits(vec![BitEntry::Indicator {
                column: 1,
                value: 0,
            }])],
            vec![SelectorFactor {
                column: 2,
                slots: vec![6],
            }],
            [0, 1, 2, 3, 4, 5],
            vec![],
            0,
            vec![],
        )
        .unwrap();
        assert_eq!(
            FoldLayout::new(&trace, &[shape], &[vec![]]).unwrap_err(),
            RouterError::Dimension {
                variables: usize::BITS as usize
            }
        );
        let shapes = vec![small_shape(WordSlot::Trace(0))];
        let layout = FoldLayout::new(&trace, &shapes, &[vec![]]).unwrap();
        let plan = ScatterPlan::new(Arc::clone(&trace)).unwrap();
        assert_eq!(
            fold_pass(&trace, &shapes, &[ZERO; 3], &plan, &layout, &[1, 1]).unwrap_err(),
            RouterError::Dimension {
                variables: usize::BITS as usize
            }
        );
    }
}
