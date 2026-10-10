#![cfg(feature = "test-utils")]
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test fixtures and assertions fail by panicking"
)]

use jolt_field::{Field, F128};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_kernels::oracle::{mle_at, round_polynomial};
use jolt_rv64i_kernels::reduction::{
    g_pass_digits, ColumnMap, ReductionCore, ReductionError, ReductionLeg,
};
use jolt_rv64i_kernels::round::RoundError;
use jolt_rv64i_kernels::source::{CycleSource, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, BooleanHypercube, ClearProof, ClearSumcheckRecorder,
    ProveRounds, SequentialRounds, SumcheckClaim, SumcheckProof, SumcheckRecorder,
    SumcheckVerifier, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
#[path = "../benches/support/allocator.rs"]
mod allocator;
use allocator::{AllocationMeasurement, CountingAllocator, RAYON_WORKER_ALLOWANCE};

fn weights(rng: &mut ChaCha20Rng) -> Vec<Vec<F128>> {
    (0..4)
        .map(|support| {
            (0..256)
                .map(|y| {
                    let nonzero = match support {
                        0 => true,
                        1 => (64..=228).contains(&y),
                        2 => y < 64 || (139..=230).contains(&y),
                        _ => y < 64,
                    };
                    if nonzero {
                        F128::random(rng)
                    } else {
                        F128::from_raw(0)
                    }
                })
                .collect()
        })
        .collect()
}

fn defining_tables(rows: &[[u64; 4]], weights: &[Vec<F128>]) -> Vec<Vec<F128>> {
    weights
        .iter()
        .map(|weight| {
            rows.iter()
                .map(|row| {
                    weight
                        .iter()
                        .enumerate()
                        .filter(|(y, _)| row[y / 64] & (1 << (y % 64)) != 0)
                        .fold(F128::from_raw(0), |sum, (_, &weight)| sum + weight)
                })
                .collect()
        })
        .collect()
}

fn eq(point: &[F128], vertex: usize) -> F128 {
    point
        .iter()
        .enumerate()
        .map(|(i, &t)| F128::from_raw(1) + t + F128::from_raw(((vertex >> i) & 1) as u128))
        .product()
}

#[test]
fn digit_tables_equal_summation_on_indicator_rows() {
    let mut rng = ChaCha20Rng::seed_from_u64(718);
    for log_t in [3, 8] {
        let trace = Arc::new(
            SyntheticTrace::new(
                SynthProfile::Local,
                log_t,
                1 << log_t.saturating_sub(2),
                313,
            )
            .unwrap(),
        );
        let weights = weights(&mut rng);
        let validated = ValidatedTrace::new(trace.clone()).unwrap();
        let supported_weights: Vec<_> = [1, 2, 3, 1]
            .into_iter()
            .map(|index| weights[index].clone())
            .collect();
        for count in 1..=4 {
            let weights = &supported_weights[..count];
            assert_eq!(
                g_pass_digits(&validated, &SyntheticTrace::column_map(), weights).unwrap(),
                defining_tables(trace.rows(), weights),
                "{count} weights at log_t={log_t}"
            );
        }
        if log_t == 8 {
            let covered_support_weights: Vec<Vec<_>> = (0..4)
                .map(|_| {
                    (0..256)
                        .map(|column| {
                            if column < 231 {
                                F128::random(&mut rng)
                            } else {
                                F128::from_raw(0)
                            }
                        })
                        .collect()
                })
                .collect();
            assert_eq!(
                g_pass_digits(
                    &validated,
                    &SyntheticTrace::column_map(),
                    &covered_support_weights
                )
                .unwrap(),
                defining_tables(trace.rows(), &covered_support_weights)
            );
        }
    }
}

#[test]
fn digit_tables_equal_summation_for_all_columns_and_each_weight_count() {
    let mut rng = ChaCha20Rng::seed_from_u64(3718);
    let map: Vec<_> = (0..4)
        .map(|trace_word| ColumnMap::Word {
            start: 64 * trace_word,
            trace_word,
        })
        .collect();
    let weights: Vec<Vec<_>> = (0..4)
        .map(|_| {
            (0..256)
                .map(|_| F128::from_raw(1 + u128::from(rng.next_u64())))
                .collect()
        })
        .collect();
    for log_t in [3, 8] {
        let source = Arc::new(PackedRows(
            (0..1 << log_t)
                .map(|_| std::array::from_fn(|_| rng.next_u64()))
                .collect(),
        ));
        let validated = ValidatedTrace::new(source.clone()).unwrap();
        for count in 1..=4 {
            let weights = &weights[..count];
            assert_eq!(
                g_pass_digits(&validated, &map, weights).unwrap(),
                defining_tables(&source.0, weights),
                "{count} full-support weights at log_t={log_t}"
            );
        }
    }
}

struct PackedRows(Vec<[u64; 4]>);

impl CycleSource for PackedRows {
    fn cycles(&self) -> usize {
        self.0.len()
    }
    fn trace_words(&self) -> usize {
        4
    }
    fn trace_word(&self, word: usize, cycle: usize) -> u64 {
        self.0
            .get(cycle)
            .and_then(|row| row.get(word))
            .copied()
            .unwrap_or(0)
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

struct Fixture {
    tables: Vec<Vec<F128>>,
    legs: Vec<ReductionLeg>,
}

impl Fixture {
    fn new(log_t: usize, shared: bool) -> Self {
        let trace = SyntheticTrace::new(
            SynthProfile::Local,
            log_t,
            1 << log_t.saturating_sub(2),
            387,
        )
        .unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(771 + log_t as u64);
        let tables = defining_tables(trace.rows(), &weights(&mut rng)[1..]);
        Self::from_tables(tables, log_t, shared, &mut rng)
    }

    fn from_tables(
        tables: Vec<Vec<F128>>,
        log_t: usize,
        shared: bool,
        rng: &mut ChaCha20Rng,
    ) -> Self {
        let legs = (0..if shared { 4 } else { 3 })
            .map(|leg| {
                let table = leg.min(2);
                let point: Vec<_> = (0..log_t).map(|_| F128::random(rng)).collect();
                let claim = tables[table]
                    .iter()
                    .enumerate()
                    .map(|(j, &g)| eq(&point, j) * g)
                    .sum();
                ReductionLeg {
                    table,
                    point,
                    coefficient: F128::random(rng),
                    claim,
                }
            })
            .collect();
        Self { tables, legs }
    }
    fn claim(&self) -> F128 {
        self.legs
            .iter()
            .map(|leg| leg.coefficient * leg.claim)
            .sum()
    }
    fn equality_tables(&self) -> Vec<Vec<F128>> {
        self.legs
            .iter()
            .map(|leg| {
                (0..self.tables[0].len())
                    .map(|j| eq(&leg.point, j))
                    .collect()
            })
            .collect()
    }
    fn expected_round(&self, bound: &[F128]) -> UnivariatePoly<F128> {
        let equalities = self.equality_tables();
        let leaves: Vec<_> = self
            .tables
            .iter()
            .chain(&equalities)
            .map(Vec::as_slice)
            .collect();
        round_polynomial(&leaves, bound, 2, |v| {
            self.legs
                .iter()
                .enumerate()
                .map(|(index, leg)| leg.coefficient * v[leg.table] * v[self.tables.len() + index])
                .sum()
        })
        .unwrap()
    }
    fn final_claim(&self, point: &[F128]) -> F128 {
        let eqs = self.equality_tables();
        self.legs
            .iter()
            .enumerate()
            .map(|(index, leg)| {
                leg.coefficient
                    * mle_at(&self.tables[leg.table], point).unwrap()
                    * mle_at(&eqs[index], point).unwrap()
            })
            .sum()
    }
}

fn prove_and_verify(fixture: &Fixture, honest: bool) {
    let rounds = fixture.legs[0].point.len();
    let mut core = ReductionCore::new(fixture.tables.clone(), fixture.legs.clone()).unwrap();
    if honest {
        core.check_claims().unwrap();
    } else {
        assert!(matches!(
            core.check_claims(),
            Err(ReductionError::Claim { leg: 1, .. })
        ));
    }
    let mut recorder = ClearSumcheckRecorder::<F128>::new();
    let mut prover = Blake2bTranscript::<F128>::new(b"binary-reduction");
    let coefficient = F128::from_raw(127);
    let prelude = BatchPrelude::try_new(
        vec![BatchMember {
            input_claim: fixture.claim(),
            coefficient,
            rounds,
            offset: 0,
        }],
        rounds,
        2,
    )
    .unwrap();
    let proved = prove_batch(
        &prelude,
        &mut [&mut core],
        &mut SequentialRounds,
        &mut recorder,
        &mut prover,
    )
    .unwrap();
    let recorded = recorder.finish(&proved.member_claims, &mut prover).unwrap();
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = recorded.proof else {
        panic!("clear proof expected")
    };
    let mut verifier = Blake2bTranscript::<F128>::new(b"binary-reduction");
    let reduced = SumcheckVerifier::verify_compressed(
        &SumcheckClaim::new(rounds, 2, prelude.claimed_sum),
        &proof,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut verifier,
    )
    .unwrap();
    assert_eq!(reduced.point.as_slice(), proved.challenges);
    let expected = fixture.final_claim(&proved.challenges);
    if honest {
        let mut claim = prelude.claimed_sum;
        for (round, message) in proof.round_polynomials.iter().enumerate() {
            let expected = fixture.expected_round(&proved.challenges[..round]);
            let expected = UnivariatePoly::new(
                expected
                    .coefficients()
                    .iter()
                    .map(|&c| coefficient * c)
                    .collect(),
            );
            assert_eq!(message.decompress(claim), expected);
            claim = expected.evaluate(proved.challenges[round]);
        }
        assert_eq!(reduced.value, coefficient * expected);
    } else {
        assert_ne!(reduced.value, coefficient * expected);
    }
    let values: Vec<_> = fixture
        .tables
        .iter()
        .map(|table| mle_at(table, &proved.challenges).unwrap())
        .collect();
    assert_eq!(core.final_values().unwrap(), values);
}

#[test]
fn reduction_three_legs_and_four_shared_legs_match_oracle_and_verify() {
    for rounds in 1..=8 {
        for shared in [false, true] {
            prove_and_verify(&Fixture::new(rounds, shared), true);
        }
    }
}

#[test]
fn changed_claim_fails_verifier_final_check_and_diagnostic_names_leg() {
    let mut fixture = Fixture::new(8, true);
    fixture.legs[1].claim += F128::from_raw(1);
    prove_and_verify(&fixture, false);
}

#[test]
fn reduction_rejects_malformed_tables_legs_weights_and_maps() {
    let zero = F128::from_raw(0);
    let one = F128::from_raw(1);
    assert!(matches!(
        ReductionCore::new(vec![vec![zero; 12]], vec![]),
        Err(ReductionError::TableLength { actual: 12, .. })
    ));
    assert!(matches!(
        ReductionCore::new(
            vec![vec![zero; 8]; 3],
            vec![ReductionLeg {
                table: 3,
                point: vec![zero; 3],
                coefficient: one,
                claim: zero
            }]
        ),
        Err(ReductionError::LegTable {
            leg: 0,
            table: 3,
            ..
        })
    ));
    assert!(matches!(
        ReductionCore::new(
            vec![vec![zero; 8]],
            vec![ReductionLeg {
                table: 0,
                point: vec![zero; 2],
                coefficient: one,
                claim: zero
            }]
        ),
        Err(ReductionError::LegPoint {
            actual: 2,
            expected: 3,
            ..
        })
    ));
    assert!(matches!(
        ReductionCore::new(
            vec![vec![zero]],
            vec![ReductionLeg {
                table: 0,
                point: vec![],
                coefficient: one,
                claim: zero
            }]
        ),
        Err(ReductionError::Round(RoundError::EmptyPoint))
    ));
    let trace = Arc::new(SyntheticTrace::new(SynthProfile::Local, 3, 2, 817).unwrap());
    let validated = ValidatedTrace::new(trace.clone()).unwrap();
    assert!(matches!(
        g_pass_digits(
            &validated,
            &SyntheticTrace::column_map(),
            &[vec![zero; 255]]
        ),
        Err(ReductionError::WeightLength { actual: 255, .. })
    ));
    assert!(matches!(
        g_pass_digits(
            &validated,
            &[ColumnMap::Word {
                start: 200,
                trace_word: 0
            }],
            &[]
        ),
        Err(ReductionError::MapRange { start: 200, .. })
    ));
    assert!(matches!(
        g_pass_digits(
            &validated,
            &[
                ColumnMap::Word {
                    start: 64,
                    trace_word: 0
                },
                ColumnMap::Indicators {
                    start: 64,
                    column: 0
                }
            ],
            &[]
        ),
        Err(ReductionError::MapOverlap { column: 64 })
    ));
    assert!(matches!(
        g_pass_digits(
            &validated,
            &[ColumnMap::Word {
                start: 0,
                trace_word: trace.trace_words()
            }],
            &[]
        ),
        Err(ReductionError::MapTraceWord { .. })
    ));
    assert!(matches!(
        g_pass_digits(
            &validated,
            &[ColumnMap::Indicators {
                start: 0,
                column: trace.digit_columns()
            }],
            &[]
        ),
        Err(ReductionError::MapColumn { .. })
    ));
    let mut uncovered = vec![zero; 256];
    uncovered[255] = one;
    assert!(matches!(
        g_pass_digits(&validated, &SyntheticTrace::column_map(), &[uncovered]),
        Err(ReductionError::Uncovered { column: 255, .. })
    ));
    let core = ReductionCore::new(
        vec![vec![zero; 8]],
        vec![ReductionLeg {
            table: 0,
            point: vec![zero; 3],
            coefficient: one,
            claim: zero,
        }],
    )
    .unwrap();
    assert!(matches!(
        core.final_values(),
        Err(ReductionError::Unfinished)
    ));
}

#[test]
fn flag_groups_validate_width_count_ranges_overlap_and_columns() {
    let trace = ValidatedTrace::new(Arc::new(OneBitDigit)).unwrap();
    assert_eq!(
        g_pass_digits(
            &trace,
            &[ColumnMap::Flags {
                start: 0,
                columns: vec![0]
            }],
            &[]
        ),
        Err(ReductionError::MapFlags {
            count: 1,
            offending_column: Some((0, 1))
        })
    );

    let trace = Arc::new(SyntheticTrace::new(SynthProfile::Local, 3, 2, 19).unwrap());
    let trace = ValidatedTrace::new(trace).unwrap();
    for columns in [vec![], vec![18; 9]] {
        assert!(matches!(
            g_pass_digits(
                &trace,
                &[ColumnMap::Flags {
                    start: 228,
                    columns
                }],
                &[]
            ),
            Err(ReductionError::MapFlags { .. })
        ));
    }
    assert!(matches!(
        g_pass_digits(
            &trace,
            &[ColumnMap::Flags {
                start: 256,
                columns: vec![18]
            }],
            &[]
        ),
        Err(ReductionError::MapRange { start: 256, .. })
    ));
    assert!(matches!(
        g_pass_digits(
            &trace,
            &[ColumnMap::Flags {
                start: 228,
                columns: vec![18, trace.source().digit_columns()]
            }],
            &[]
        ),
        Err(ReductionError::MapColumn { column, .. }) if column == trace.source().digit_columns()
    ));
    assert!(matches!(
        g_pass_digits(
            &trace,
            &[
                ColumnMap::Flags {
                    start: 64,
                    columns: vec![18]
                },
                ColumnMap::Indicators {
                    start: 64,
                    column: 0
                }
            ],
            &[]
        ),
        Err(ReductionError::MapOverlap { column: 64 })
    ));
}

#[test]
fn reduction_round_allocations_are_bounded_and_do_not_grow_with_chunks() {
    // Keep workers alive across measurements so teardown cannot lower live bytes.
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        let mut counts = [0; 2];
        for rounds in [8, 13, 14, 17] {
            let fixture = Fixture::new(rounds, true);
            let runtime_allocs = RAYON_WORKER_ALLOWANCE.allocs * pool.current_num_threads();
            let runtime_bytes = RAYON_WORKER_ALLOWANCE.bytes * pool.current_num_threads();
            let mut claim = fixture.claim();
            let challenges: Vec<_> = (0..rounds)
                .map(|i| F128::from_raw(79 + i as u128))
                .collect();
            // Exclude only fixture-owned storage; the core releases its dense inputs.
            let resident = CountingAllocator::live_bytes();
            let mut core =
                ReductionCore::new(fixture.tables.clone(), fixture.legs.clone()).unwrap();
            let measurement = AllocationMeasurement::begin();
            for round in 0..rounds {
                let message = core
                    .prove_round(
                        if round == 0 {
                            None
                        } else {
                            Some(challenges[round - 1])
                        },
                        round,
                        claim,
                    )
                    .unwrap();
                claim = message.evaluate(challenges[round]);
            }
            core.finish_rounds(challenges[rounds - 1]).unwrap();
            let stats = measurement.finish();
            assert!(
                stats.allocs <= 16 * rounds + 64 + runtime_allocs,
                "{} allocations for {rounds} rounds",
                stats.allocs
            );
            if rounds == 13 {
                counts[0] = stats.allocs;
            }
            if rounds == 17 {
                counts[1] = stats.allocs;
            }
            assert!(stats.peak_bytes >= stats.final_bytes);
            assert!(stats.peak_bytes <= 64 * 1024 + runtime_bytes);
            let final_values_bytes = std::mem::size_of_val(core.final_values().unwrap());
            assert!((resident + final_values_bytes
                ..=resident + final_values_bytes + runtime_bytes)
                .contains(&CountingAllocator::live_bytes()));
            drop(core);
            assert!(
                (resident..=resident + runtime_bytes).contains(&CountingAllocator::live_bytes())
            );
        }
        // Four extra rounds add fixed metadata; round-chunks grow from 14 to 74.
        assert!(counts[1] <= counts[0] + 16 * (17 - 13) + RAYON_WORKER_ALLOWANCE.allocs);
    });
}

struct OneBitDigit;
impl CycleSource for OneBitDigit {
    fn cycles(&self) -> usize {
        2
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
        1
    }
    fn bits(&self, column: usize) -> usize {
        usize::from(column == 0)
    }
    fn by_row(&self, _: usize) -> bool {
        false
    }
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        (column == 0 && cycle < 2).then_some(0)
    }
    fn row_digit(&self, _: usize, _: usize) -> Option<usize> {
        None
    }
}

#[test]
fn eight_reduction_legs_match_the_definition() {
    for log_t in 1..=8 {
        let mut fixture = Fixture::new(log_t, true);
        for leg in 0..4 {
            let point: Vec<_> = (0..log_t)
                .map(|i| F128::from_raw(211 + 8 * leg as u128 + i as u128))
                .collect();
            let claim = fixture.tables[1]
                .iter()
                .enumerate()
                .map(|(j, &g)| eq(&point, j) * g)
                .sum();
            fixture.legs.push(ReductionLeg {
                table: 1,
                point,
                coefficient: F128::from_raw(73 + leg as u128),
                claim,
            });
        }
        prove_and_verify(&fixture, true);
    }
}

#[test]
fn reduction_rejects_weight_and_leg_counts_above_the_bounds() {
    let zero = F128::from_raw(0);
    let trace = ValidatedTrace::new(Arc::new(
        SyntheticTrace::new(SynthProfile::Local, 3, 2, 17).unwrap(),
    ))
    .unwrap();
    assert_eq!(
        g_pass_digits(
            &trace,
            &SyntheticTrace::column_map(),
            &vec![vec![zero; 256]; 5]
        ),
        Err(ReductionError::WeightCount { count: 5 })
    );
    assert!(matches!(
        ReductionCore::new(
            vec![vec![zero; 8]],
            vec![
                ReductionLeg {
                    table: 0,
                    point: vec![zero; 3],
                    coefficient: zero,
                    claim: zero
                };
                9
            ]
        ),
        Err(ReductionError::LegCount { count: 9 })
    ));
}

#[test]
fn two_chunk_tables_round_messages_and_final_values_match_the_definition_on_each_pool() {
    let log_t = 13;
    let source = Arc::new(SyntheticTrace::new(SynthProfile::Local, log_t, 1 << 11, 918).unwrap());
    let trace = ValidatedTrace::new(source.clone()).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(173);
    let weights = weights(&mut rng);
    let weights = &weights[1..];
    let map = SyntheticTrace::column_map();
    let expected_tables = defining_tables(source.rows(), weights);
    let fixture = Fixture::from_tables(expected_tables, log_t, true, &mut rng);
    let challenges: Vec<_> = (0..log_t).map(|_| F128::random(&mut rng)).collect();
    let expected_messages: Vec<_> = (0..log_t)
        .map(|round| fixture.expected_round(&challenges[..round]))
        .collect();
    let expected_final_values: Vec<_> = fixture
        .tables
        .iter()
        .map(|table| mle_at(table, &challenges).unwrap())
        .collect();
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let tables = g_pass_digits(&trace, &map, weights).unwrap();
            assert_eq!(tables, fixture.tables);
            let mut core = ReductionCore::new(tables, fixture.legs.clone()).unwrap();
            core.check_claims().unwrap();
            let mut claim = fixture.claim();
            for (round, expected) in expected_messages.iter().enumerate() {
                let message = core
                    .prove_round(
                        round.checked_sub(1).map(|previous| challenges[previous]),
                        round,
                        claim,
                    )
                    .unwrap();
                assert_eq!(message.coefficients(), expected.coefficients());
                claim = expected.evaluate(challenges[round]);
            }
            core.finish_rounds(challenges[log_t - 1]).unwrap();
            assert_eq!(core.final_values().unwrap(), expected_final_values);
        });
    }
}

#[test]
fn digit_builder_allocations_are_bounded_per_pass_and_only_outputs_remain() {
    // Keep workers alive across measurements so teardown cannot lower live bytes.
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        let mut counts = [0; 2];
        for log_t in [8, 14, 16] {
            let source = Arc::new(
                SyntheticTrace::new(
                    SynthProfile::Local,
                    log_t,
                    1 << log_t.saturating_sub(2),
                    713,
                )
                .unwrap(),
            );
            let trace = ValidatedTrace::new(source.clone()).unwrap();
            let mut rng = ChaCha20Rng::seed_from_u64(818);
            let weights = weights(&mut rng);
            let weights = &weights[1..];
            let map = SyntheticTrace::column_map();
            let expected = defining_tables(source.rows(), weights);
            drop(g_pass_digits(&trace, &map, weights).unwrap());
            let runtime_allocs = RAYON_WORKER_ALLOWANCE.allocs * pool.current_num_threads();
            let runtime_bytes = RAYON_WORKER_ALLOWANCE.bytes * pool.current_num_threads();
            let resident = CountingAllocator::live_bytes();
            let measurement = AllocationMeasurement::begin();
            let tables = g_pass_digits(&trace, &map, weights).unwrap();
            let stats = measurement.finish();
            assert!(
                stats.allocs <= 256 + runtime_allocs,
                "{} allocations at log_t={log_t}",
                stats.allocs
            );
            if log_t == 14 {
                counts[0] = stats.allocs;
            }
            if log_t == 16 {
                counts[1] = stats.allocs;
            }
            let output_bytes = tables.capacity() * std::mem::size_of::<Vec<F128>>()
                + tables
                    .iter()
                    .map(|table| table.capacity() * std::mem::size_of::<F128>())
                    .sum::<usize>();
            assert!(stats.peak_bytes <= output_bytes + 128 * 1024 + runtime_bytes);
            assert!((output_bytes..=output_bytes + runtime_bytes).contains(&stats.final_bytes));
            assert_eq!(tables, expected);
            drop(tables);
            assert!(
                (resident..=resident + runtime_bytes).contains(&CountingAllocator::live_bytes())
            );
        }
        // Lookup/view counts are fixed while tiles grow from 64 to 256.
        assert!(counts[1] <= counts[0] + RAYON_WORKER_ALLOWANCE.allocs);
    });
}

#[test]
fn reduction_zero_coefficients_and_same_point_shared_legs_match_oracle() {
    for log_t in 1..=8 {
        for variant in 1..3 {
            let mut fixture = Fixture::new(log_t, false);
            match variant {
                1 => fixture.legs[1].coefficient = F128::from_raw(0),
                2 => {
                    fixture.legs[2] = ReductionLeg {
                        table: fixture.legs[1].table,
                        point: fixture.legs[1].point.clone(),
                        coefficient: F128::from_raw(73),
                        claim: fixture.legs[1].claim,
                    }
                }
                _ => {}
            }
            prove_and_verify(&fixture, true);
        }
    }
}
