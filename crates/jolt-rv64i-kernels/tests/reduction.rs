#![cfg(feature = "test-utils")]
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test fixtures and assertions fail by panicking"
)]

use jolt_field::{Field, F128};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_kernels::oracle::{mle_at, round_polynomial};
use jolt_rv64i_kernels::reduction::{g_pass_digits, ColumnMap, ReductionCore, ReductionError};
use jolt_rv64i_kernels::round::RoundError;
use jolt_rv64i_kernels::source::{CycleSource, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, BooleanHypercube, ClearProof, ClearSumcheckRecorder,
    ProveRounds, SequentialRounds, SumcheckClaim, SumcheckProof, SumcheckRecorder,
    SumcheckVerifier, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
#[path = "../benches/support/allocator.rs"]
mod allocator;
use allocator::{AllocationMeasurement, CountingAllocator};

fn map() -> Vec<ColumnMap> {
    let mut map = vec![ColumnMap::Word {
        start: 0,
        trace_word: 5,
    }];
    map.extend((0..10).map(|column| ColumnMap::Indicators {
        start: 64 + 15 * column,
        column,
    }));
    map.extend((10..12).map(|column| ColumnMap::Indicators {
        start: 214 + 7 * (column - 10),
        column,
    }));
    map.push(ColumnMap::Flags {
        start: 228,
        columns: vec![18, 19, 20],
    });
    map
}

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
        assert_eq!(
            g_pass_digits(&validated, &map(), &weights[1..]).unwrap(),
            defining_tables(trace.rows(), &weights[1..])
        );
        let five_weights: Vec<_> = [1, 2, 3, 1, 2]
            .into_iter()
            .map(|index| weights[index].clone())
            .collect();
        assert_eq!(
            g_pass_digits(&validated, &map(), &five_weights).unwrap(),
            defining_tables(trace.rows(), &five_weights)
        );
    }
}

struct Fixture {
    tables: Vec<Vec<F128>>,
    legs: Vec<(usize, Vec<F128>, F128, F128)>,
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
        let legs = (0..if shared { 4 } else { 3 })
            .map(|leg| {
                let table = leg.min(2);
                let point: Vec<_> = (0..log_t).map(|_| F128::random(&mut rng)).collect();
                let claim = tables[table]
                    .iter()
                    .enumerate()
                    .map(|(j, &g)| eq(&point, j) * g)
                    .sum();
                (table, point, F128::random(&mut rng), claim)
            })
            .collect();
        Self { tables, legs }
    }
    fn claim(&self) -> F128 {
        self.legs.iter().map(|(_, _, k, claim)| *k * claim).sum()
    }
    fn equality_tables(&self) -> Vec<Vec<F128>> {
        self.legs
            .iter()
            .map(|(_, point, _, _)| (0..self.tables[0].len()).map(|j| eq(point, j)).collect())
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
                .map(|(leg, (table, _, k, _))| *k * v[*table] * v[self.tables.len() + leg])
                .sum()
        })
        .unwrap()
    }
    fn final_claim(&self, point: &[F128]) -> F128 {
        let eqs = self.equality_tables();
        self.legs
            .iter()
            .enumerate()
            .map(|(leg, (table, _, k, _))| {
                *k * mle_at(&self.tables[*table], point).unwrap()
                    * mle_at(&eqs[leg], point).unwrap()
            })
            .sum()
    }
}

fn prove_and_verify(fixture: &Fixture, honest: bool) {
    let rounds = fixture.legs[0].1.len();
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
    for rounds in 3..=8 {
        for shared in [false, true] {
            prove_and_verify(&Fixture::new(rounds, shared), true);
        }
    }
}

#[test]
fn changed_claim_fails_verifier_final_check_and_diagnostic_names_leg() {
    let mut fixture = Fixture::new(8, true);
    fixture.legs[1].3 += F128::from_raw(1);
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
        ReductionCore::new(vec![vec![zero; 8]; 3], vec![(3, vec![zero; 3], one, zero)]),
        Err(ReductionError::LegTable {
            leg: 0,
            table: 3,
            ..
        })
    ));
    assert!(matches!(
        ReductionCore::new(vec![vec![zero; 8]], vec![(0, vec![zero; 2], one, zero)]),
        Err(ReductionError::LegPoint {
            actual: 2,
            expected: 3,
            ..
        })
    ));
    assert!(matches!(
        ReductionCore::new(vec![vec![zero]], vec![(0, vec![], one, zero)]),
        Err(ReductionError::Round(RoundError::EmptyPoint))
    ));
    let trace = Arc::new(SyntheticTrace::new(SynthProfile::Local, 3, 2, 817).unwrap());
    let validated = ValidatedTrace::new(trace.clone()).unwrap();
    assert!(matches!(
        g_pass_digits(&validated, &map(), &[vec![zero; 255]]),
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
        g_pass_digits(&validated, &map(), &[uncovered]),
        Err(ReductionError::Uncovered { column: 255, .. })
    ));
    let core =
        ReductionCore::new(vec![vec![zero; 8]], vec![(0, vec![zero; 3], one, zero)]).unwrap();
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
                columns: vec![18, 21]
            }],
            &[]
        ),
        Err(ReductionError::MapColumn { column: 21, .. })
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
fn reduction_round_allocations_are_bounded_at_both_sizes() {
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        for rounds in [8, 14] {
            let fixture = Fixture::new(rounds, true);
            let mut core =
                ReductionCore::new(fixture.tables.clone(), fixture.legs.clone()).unwrap();
            let resident = CountingAllocator::live_bytes();
            let mut claim = fixture.claim();
            let challenges: Vec<_> = (0..rounds)
                .map(|i| F128::from_raw(79 + i as u128))
                .collect();
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
                stats.allocs <= 16 * rounds + 64,
                "{} allocations for {rounds} rounds",
                stats.allocs
            );
            assert!(stats.peak_bytes >= stats.final_bytes);
            assert!(CountingAllocator::live_bytes() < resident);
        }
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
fn reduction_with_more_than_four_legs_matches_the_unbounded_contract() {
    let mut fixture = Fixture::new(8, true);
    let point: Vec<_> = (0..8).map(|i| F128::from_raw(211 + i as u128)).collect();
    let claim = fixture.tables[1]
        .iter()
        .enumerate()
        .map(|(j, &g)| eq(&point, j) * g)
        .sum();
    fixture.legs.push((1, point, F128::from_raw(73), claim));
    prove_and_verify(&fixture, true);
}
