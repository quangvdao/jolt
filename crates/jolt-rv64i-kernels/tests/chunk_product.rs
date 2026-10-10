#![cfg(feature = "test-utils")]
#![expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "test assertions and fixture construction may panic on failure"
)]

use jolt_field::{Field, F128};
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_rv64i_kernels::chunk_product::{
    combined_weight, ChunkProductCore, ChunkProductError, ChunkWeight, ChunkWeightTerm, EqTerm,
};
use jolt_rv64i_kernels::oracle::{mle_at, round_polynomial};
use jolt_rv64i_kernels::round::RoundError;
use jolt_rv64i_kernels::source::{CycleSource, DigitColumns, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, BooleanHypercube, ClearProof, ClearSumcheckRecorder,
    ProveRounds, ProvedBatch, SequentialRounds, SumcheckClaim, SumcheckError, SumcheckProof,
    SumcheckRecorder, SumcheckVerifier, OPENING_CLAIM_TRANSCRIPT_LABEL,
    SUMCHECK_CLAIM_TRANSCRIPT_LABEL, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::sync::Arc;

const LABEL: &[u8] = b"rv64i-chunk-product-acceptance";
const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);

struct UniformColumns {
    trace: SyntheticTrace,
    columns: usize,
    top_bits: usize,
    missing: bool,
    cycles_override: Option<usize>,
}

impl CycleSource for UniformColumns {
    fn cycles(&self) -> usize {
        self.cycles_override.unwrap_or_else(|| self.trace.cycles())
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
        self.columns
    }
    fn bits(&self, column: usize) -> usize {
        if column + 1 == self.columns {
            self.top_bits
        } else {
            4
        }
    }
    fn by_row(&self, _: usize) -> bool {
        false
    }
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        if column >= self.columns || (self.missing && column == 0 && cycle == 3) {
            None
        } else if self.cycles_override == Some(1) && cycle == 0 {
            Some(0)
        } else {
            self.trace
                .digit(column, cycle)
                .map(|digit| digit & ((1 << self.bits(column)) - 1))
        }
    }
    fn row_digit(&self, _: usize, _: usize) -> Option<usize> {
        None
    }
}

fn columns(log_t: usize, d: usize, top_bits: usize, missing: bool) -> DigitColumns<UniformColumns> {
    let source = Arc::new(UniformColumns {
        trace: SyntheticTrace::new(SynthProfile::UniformDigits, log_t, 1, 0xc8_0011).unwrap(),
        columns: d,
        top_bits,
        missing,
        cycles_override: None,
    });
    DigitColumns::from_validated(
        Arc::new(ValidatedTrace::new(source).unwrap()),
        (0..d).collect(),
    )
    .unwrap()
}

fn equality(point: &[F128], index: usize) -> F128 {
    point.iter().enumerate().fold(ONE, |value, (bit, &w)| {
        value * (ONE + w + F128::from_raw(((index >> bit) & 1) as u128))
    })
}

fn random_point(length: usize, rng: &mut ChaCha20Rng) -> Vec<F128> {
    (0..length).map(|_| F128::random(rng)).collect()
}

fn terms(log_t: usize, count: usize, next: bool, rng: &mut ChaCha20Rng) -> Vec<ChunkWeightTerm> {
    (0..count)
        .map(|term| {
            let coefficient = F128::random(rng);
            let point = if term == 0 {
                (0..log_t)
                    .map(|bit| F128::from_raw((bit & 1) as u128))
                    .collect()
            } else {
                random_point(log_t, rng)
            };
            if next && term + 1 == count {
                ChunkWeightTerm::Next { coefficient, point }
            } else {
                ChunkWeightTerm::Eq { coefficient, point }
            }
        })
        .collect()
}

fn defining_weight(log_t: usize, terms: &[ChunkWeightTerm]) -> Vec<F128> {
    (0..1 << log_t)
        .map(|cycle| {
            terms
                .iter()
                .map(|term| match term {
                    ChunkWeightTerm::Eq { coefficient, point } => {
                        *coefficient * equality(point, cycle)
                    }
                    ChunkWeightTerm::Next { coefficient, point } => cycle
                        .checked_sub(1)
                        .map_or(ZERO, |previous| *coefficient * equality(point, previous)),
                })
                .sum()
        })
        .collect()
}

struct RecordingCore {
    inner: ChunkProductCore,
    messages: Vec<UnivariatePoly<F128>>,
}

impl ProveRounds<F128> for RecordingCore {
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        previous_claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        let message = self.inner.prove_round(bind, round, previous_claim)?;
        self.messages.push(message.clone());
        Ok(message)
    }

    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        self.inner.finish_rounds(bind)
    }
}

struct Fixture {
    leaves: Vec<Vec<F128>>,
    prelude: BatchPrelude<F128>,
    proved: ProvedBatch<F128>,
    proof: SumcheckProof<F128, ()>,
    raw_messages: Vec<UnivariatePoly<F128>>,
    final_values: (F128, Vec<F128>),
    transcript_state: [u8; 32],
}

impl Fixture {
    fn prove(
        columns: DigitColumns<UniformColumns>,
        points: Vec<Vec<F128>>,
        terms: &[ChunkWeightTerm],
        eq_terms: bool,
        change_claim: bool,
    ) -> Result<Self, SumcheckError<F128>> {
        let log_t = columns.cycles().ilog2() as usize;
        let d = points.len();
        let mut leaves = vec![defining_weight(log_t, terms)];
        for (column, point) in points.iter().enumerate() {
            leaves.push(
                (0..columns.cycles())
                    .map(|cycle| equality(point, columns.index(column, cycle).unwrap()))
                    .collect(),
            );
        }
        let honest_claim = (0..columns.cycles())
            .map(|cycle| leaves.iter().map(|leaf| leaf[cycle]).product::<F128>())
            .sum();
        let mut input_claim = honest_claim;
        let weight = if eq_terms {
            let mut weighted_terms = terms
                .iter()
                .map(|term| {
                    let ChunkWeightTerm::Eq { coefficient, point } = term else {
                        unreachable!()
                    };
                    let claim = (0..columns.cycles())
                        .map(|cycle| {
                            *coefficient
                                * equality(point, cycle)
                                * leaves[1..].iter().map(|leaf| leaf[cycle]).product::<F128>()
                        })
                        .sum();
                    EqTerm {
                        coefficient: *coefficient,
                        point: point.clone(),
                        claim,
                    }
                })
                .collect::<Vec<_>>();
            if change_claim {
                weighted_terms[0].claim += ONE;
                input_claim += ONE;
            }
            ChunkWeight::EqTerms(weighted_terms)
        } else {
            let built = combined_weight(log_t, terms).unwrap();
            assert_eq!(built, leaves[0]);
            ChunkWeight::Dense(built)
        };
        let mut core = RecordingCore {
            inner: ChunkProductCore::new(columns, points, weight).unwrap(),
            messages: Vec::with_capacity(log_t),
        };
        let mut recorder = ClearSumcheckRecorder::<F128>::new();
        let mut transcript = Blake2bTranscript::<F128>::new(LABEL);
        recorder.absorb_input_claims(&[input_claim], &mut transcript);
        let prelude = BatchPrelude::try_new(
            vec![BatchMember {
                input_claim,
                coefficient: transcript.challenge_scalar(),
                rounds: log_t,
                offset: 0,
            }],
            log_t,
            d + 1,
        )
        .unwrap();
        let proved = prove_batch(
            &prelude,
            &mut [&mut core],
            &mut SequentialRounds,
            &mut recorder,
            &mut transcript,
        )?;
        let recorded = recorder.finish(&proved.member_claims, &mut transcript)?;
        Ok(Self {
            leaves,
            prelude,
            proved,
            proof: recorded.proof,
            final_values: core.inner.final_values().unwrap(),
            raw_messages: core.messages,
            transcript_state: transcript.state(),
        })
    }

    fn verify(&self, proof: &SumcheckProof<F128, ()>) -> bool {
        let mut transcript = Blake2bTranscript::<F128>::new(LABEL);
        let member = &self.prelude.members[0];
        transcript.append_labeled(SUMCHECK_CLAIM_TRANSCRIPT_LABEL, &member.input_claim);
        assert_eq!(transcript.challenge_scalar(), member.coefficient);
        let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
            unreachable!()
        };
        let Ok(reduced) = SumcheckVerifier::verify_compressed(
            &SumcheckClaim::new(
                self.prelude.max_num_vars,
                self.leaves.len(),
                self.prelude.claimed_sum,
            ),
            proof,
            BooleanHypercube,
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            &mut transcript,
        ) else {
            return false;
        };
        let native = self
            .leaves
            .iter()
            .map(|leaf| mle_at(leaf, reduced.point.as_slice()).unwrap())
            .product::<F128>();
        if reduced.value != member.coefficient * native {
            return false;
        }
        transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, &native);
        assert_eq!(transcript.state(), self.transcript_state);
        true
    }

    fn assert_definition(&self) {
        let coefficient = self.prelude.members[0].coefficient;
        let SumcheckProof::Clear(ClearProof::Compressed(proof)) = &self.proof else {
            unreachable!()
        };
        assert_eq!(self.raw_messages.len(), self.proved.challenges.len());
        assert_eq!(proof.round_polynomials.len(), self.proved.challenges.len());
        let leaves = self.leaves.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut claim = self.prelude.claimed_sum;
        for (round, message) in proof.round_polynomials.iter().enumerate() {
            let expected = round_polynomial(
                &leaves,
                &self.proved.challenges[..round],
                leaves.len(),
                |values| values.iter().copied().product::<F128>(),
            )
            .unwrap();
            assert_eq!(self.raw_messages[round], expected);
            let expected_coefficients = expected
                .coefficients()
                .iter()
                .map(|&value| coefficient * value)
                .collect::<Vec<_>>();
            let mut coefficients = message.decompress(claim).coefficients().to_vec();
            assert!(coefficients.len() <= expected_coefficients.len());
            coefficients.resize(expected_coefficients.len(), ZERO);
            assert_eq!(coefficients, expected_coefficients);
            claim = coefficient * expected.evaluate(self.proved.challenges[round]);
        }
        let expected = self
            .leaves
            .iter()
            .map(|leaf| mle_at(leaf, &self.proved.challenges).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(self.final_values.0, expected[0]);
        assert_eq!(self.final_values.1, expected[1..]);
        let final_value = expected.iter().copied().product::<F128>();
        assert_eq!(self.proved.member_claims, [final_value]);
        assert_eq!(self.proved.final_claim, coefficient * final_value);
        assert_eq!(claim, self.proved.final_claim);
        assert!(self.verify(&self.proof));
    }

    fn assert_changed_coefficient_rejected(&self) {
        let mut proof = self.proof.clone();
        let SumcheckProof::Clear(ClearProof::Compressed(clear)) = &mut proof else {
            unreachable!()
        };
        let message = clear.round_polynomials.last_mut().unwrap();
        let mut coefficients = message.coeffs_except_linear_term().to_vec();
        coefficients[0] += ONE;
        *message = CompressedPoly::new(coefficients);
        assert!(!self.verify(&proof));
    }
}

#[test]
fn combined_weight_equals_defining_sum_and_next_starts_at_zero() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xc8_0001);
    for log_t in 3..=8 {
        for count in [1, 2, 5] {
            for next in [false, true] {
                let terms = terms(log_t, count, next, &mut rng);
                assert_eq!(
                    combined_weight(log_t, &terms).unwrap(),
                    defining_weight(log_t, &terms)
                );
            }
        }
        let term = ChunkWeightTerm::Next {
            coefficient: F128::random(&mut rng),
            point: random_point(log_t, &mut rng),
        };
        let weight = combined_weight(log_t, &[term]).unwrap();
        assert_eq!(weight[0], ZERO);
    }
}

#[test]
fn dense_chunk_rounds_final_values_and_proof_match_defining_sum() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xc8_0002);
    for log_t in 3..=8 {
        for d in 1..=7 {
            for top_bits in 1..=4 {
                for count in [1, 2, 5] {
                    let source = columns(log_t, d, top_bits, false);
                    let points = (0..d)
                        .map(|column| random_point(source.source().bits(column), &mut rng))
                        .collect();
                    let terms = terms(log_t, count, true, &mut rng);
                    let fixture = Fixture::prove(source, points, &terms, false, false).unwrap();
                    fixture.assert_definition();
                    fixture.assert_changed_coefficient_rejected();
                }
            }
        }
    }
}

#[test]
fn eq_terms_rounds_final_values_and_proof_match_defining_sum_including_one_column() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xc8_0003);
    for log_t in 3..=8 {
        for d in 1..=7 {
            for top_bits in 1..=4 {
                for count in [1, 2] {
                    let source = columns(log_t, d, top_bits, false);
                    let points = (0..d)
                        .map(|column| random_point(source.source().bits(column), &mut rng))
                        .collect();
                    let terms = terms(log_t, count, false, &mut rng);
                    let fixture = Fixture::prove(source, points, &terms, true, false).unwrap();
                    fixture.assert_definition();
                    fixture.assert_changed_coefficient_rejected();
                }
            }
        }
    }
}

#[test]
fn general_eq_term_counts_match_defining_rounds_final_values_and_proof() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xc8_0005);
    for (log_t, d, count) in [
        (8, 1, 3),
        (8, 1, 4),
        (8, 1, 6),
        (8, 5, 3),
        (8, 5, 4),
        (8, 5, 6),
        (3, 2, 2000),
    ] {
        let source = columns(log_t, d, 2, false);
        let points = (0..d)
            .map(|column| random_point(source.source().bits(column), &mut rng))
            .collect();
        let terms = terms(log_t, count, false, &mut rng);
        let fixture = Fixture::prove(source, points, &terms, true, false).unwrap();
        fixture.assert_definition();
        fixture.assert_changed_coefficient_rejected();
    }
}

#[test]
fn general_eq_terms_cross_cache_tiles_and_cycle_chunks() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xc8_0006);
    let source = columns(17, 2, 2, false);
    let points = (0..2)
        .map(|column| random_point(source.source().bits(column), &mut rng))
        .collect();
    let terms = terms(17, 3, false, &mut rng);
    let fixture = Fixture::prove(source, points, &terms, true, false).unwrap();
    fixture.assert_definition();
    fixture.assert_changed_coefficient_rejected();
}

#[test]
fn dense_chunk_product_crosses_cycle_chunks() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xc8_0007);
    let source = columns(14, 2, 1, false);
    let points = (0..2)
        .map(|column| random_point(source.source().bits(column), &mut rng))
        .collect();
    let terms = terms(14, 2, true, &mut rng);
    let fixture = Fixture::prove(source, points, &terms, false, false).unwrap();
    fixture.assert_definition();
    fixture.assert_changed_coefficient_rejected();
}

#[test]
fn small_cycle_domains_and_zero_weights_match_defining_sum() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xc8_0008);
    for log_t in [1, 2] {
        for d in [1, 2] {
            for (eq_terms, zero_coefficients) in [(false, 0), (true, 0), (false, 2), (true, 1)] {
                let source = columns(log_t, d, 1, false);
                let points = (0..d)
                    .map(|column| random_point(source.source().bits(column), &mut rng))
                    .collect();
                let mut terms = terms(log_t, 2, false, &mut rng);
                for term in terms.iter_mut().take(zero_coefficients) {
                    let ChunkWeightTerm::Eq { coefficient, .. } = term else {
                        unreachable!()
                    };
                    *coefficient = ZERO;
                }
                let fixture = Fixture::prove(source, points, &terms, eq_terms, false).unwrap();
                fixture.assert_definition();
                fixture.assert_changed_coefficient_rejected();
            }
        }
    }
}

#[test]
fn later_zero_equality_prefix_matches_defining_sum() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xc8_0009);
    let source = columns(4, 2, 2, false);
    let points = (0..2)
        .map(|column| random_point(source.source().bits(column), &mut rng))
        .collect::<Vec<_>>();
    let term_point = vec![F128::from_raw(2), ZERO, ONE, ZERO];
    let term = ChunkWeightTerm::Eq {
        coefficient: ONE,
        point: term_point.clone(),
    };
    let mut terms = terms(4, 3, false, &mut rng);
    terms[0] = term;
    let leaves = std::iter::once(defining_weight(4, &terms))
        .chain(points.iter().enumerate().map(|(column, point)| {
            (0..source.cycles())
                .map(|cycle| equality(point, source.index(column, cycle).unwrap()))
                .collect()
        }))
        .collect::<Vec<_>>();
    let input_terms = terms
        .iter()
        .map(|term| {
            let ChunkWeightTerm::Eq { coefficient, point } = term else {
                unreachable!()
            };
            let claim = (0..source.cycles())
                .map(|cycle| {
                    *coefficient
                        * equality(point, cycle)
                        * leaves[1..].iter().map(|leaf| leaf[cycle]).product::<F128>()
                })
                .sum();
            EqTerm {
                coefficient: *coefficient,
                point: point.clone(),
                claim,
            }
        })
        .collect();
    let mut core =
        ChunkProductCore::new(source, points, ChunkWeight::EqTerms(input_terms)).unwrap();
    let challenges = [
        F128::from_raw(4),
        F128::from_raw(8),
        ONE + term_point[2],
        F128::from_raw(14),
    ];
    let prefix_before = term_point[..2]
        .iter()
        .zip(&challenges[..2])
        .map(|(&coordinate, &challenge)| ONE + coordinate + challenge)
        .product::<F128>();
    assert_ne!(prefix_before, ZERO);
    assert_eq!(prefix_before * (ONE + term_point[2] + challenges[2]), ZERO);
    let mut claim = (0..leaves[0].len())
        .map(|cycle| leaves.iter().map(|leaf| leaf[cycle]).product::<F128>())
        .sum();
    let oracle_leaves = leaves.iter().map(Vec::as_slice).collect::<Vec<_>>();
    for (round, &challenge) in challenges.iter().enumerate() {
        let expected = round_polynomial(&oracle_leaves, &challenges[..round], 3, |values| {
            values.iter().copied().product::<F128>()
        })
        .unwrap();
        let bind = round.checked_sub(1).map(|previous| challenges[previous]);
        let message = core.prove_round(bind, round, claim).unwrap();
        assert_eq!(message, expected);
        claim = expected.evaluate(challenge);
    }
    core.finish_rounds(challenges[3]).unwrap();
    let expected = leaves
        .iter()
        .map(|leaf| mle_at(leaf, &challenges).unwrap())
        .collect::<Vec<_>>();
    let final_values = core.final_values().unwrap();
    assert_eq!(final_values.0, expected[0]);
    assert_eq!(final_values.1, expected[1..]);
    assert_eq!(claim, expected.iter().copied().product::<F128>());
}

#[test]
fn changed_eq_term_claim_is_rejected_for_every_column_count() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xc8_0004);
    for d in 1..=7 {
        for count in [1, 2] {
            let source = columns(8, d, 4, false);
            let points = (0..d).map(|_| random_point(4, &mut rng)).collect();
            let terms = (0..count)
                .map(|_| ChunkWeightTerm::Eq {
                    coefficient: F128::random(&mut rng),
                    point: random_point(8, &mut rng),
                })
                .collect::<Vec<_>>();
            match Fixture::prove(source, points, &terms, true, true) {
                Ok(fixture) => assert!(!fixture.verify(&fixture.proof)),
                Err(SumcheckError::RoundCheckFailed { .. }) => {}
                Err(error) => panic!("claim tampering produced an unrelated failure: {error:?}"),
            }
        }
    }
}

#[test]
fn column_count_rejects_zero_and_eight() {
    for d in [0, 8] {
        assert!(matches!(
            ChunkProductCore::new(columns(3, d, 4, false), vec![vec![ONE; 4]; d], ChunkWeight::Dense(vec![ONE; 8])),
            Err(ChunkProductError::Columns { columns }) if columns == d
        ));
    }
}

#[test]
fn point_count_rejects_one_point_for_two_columns() {
    assert!(matches!(
        ChunkProductCore::new(
            columns(3, 2, 4, false),
            vec![vec![ONE; 4]],
            ChunkWeight::Dense(vec![ONE; 8])
        ),
        Err(ChunkProductError::PointCount {
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn point_length_rejects_three_coordinates_for_four_bits() {
    assert!(matches!(
        ChunkProductCore::new(
            columns(3, 1, 4, false),
            vec![vec![ONE; 3]],
            ChunkWeight::Dense(vec![ONE; 8])
        ),
        Err(ChunkProductError::PointLength {
            column: 0,
            expected: 4,
            actual: 3
        })
    ));
}

#[test]
fn dense_weight_length_rejects_half_a_table() {
    assert!(matches!(
        ChunkProductCore::new(
            columns(3, 1, 4, false),
            vec![vec![ONE; 4]],
            ChunkWeight::Dense(vec![ONE; 4])
        ),
        Err(ChunkProductError::WeightLength {
            expected: 8,
            actual: 4
        })
    ));
}

#[test]
fn term_point_rejects_extra_coordinate_in_builder_and_core() {
    assert!(matches!(
        combined_weight(
            3,
            &[ChunkWeightTerm::Eq {
                coefficient: ONE,
                point: vec![ONE; 4]
            }]
        ),
        Err(ChunkProductError::TermPoint {
            term: 0,
            expected: 3,
            actual: 4
        })
    ));
    assert!(matches!(
        ChunkProductCore::new(
            columns(3, 1, 4, false),
            vec![vec![ONE; 4]],
            ChunkWeight::EqTerms(vec![EqTerm {
                coefficient: ONE,
                point: vec![ONE; 4],
                claim: ZERO
            }])
        ),
        Err(ChunkProductError::TermPoint {
            term: 0,
            expected: 3,
            actual: 4
        })
    ));
}

#[test]
fn missing_digit_is_rejected_with_column_and_cycle() {
    assert!(matches!(
        ChunkProductCore::new(
            columns(3, 1, 4, true),
            vec![vec![ONE; 4]],
            ChunkWeight::Dense(vec![ONE; 8])
        ),
        Err(ChunkProductError::MissingDigit {
            column: 0,
            cycle: 3
        })
    ));
}

#[test]
fn log_size_rejects_an_unrepresentable_table() {
    for exponent in [usize::BITS as usize, usize::BITS as usize - 5] {
        assert!(matches!(
            combined_weight(exponent, &[]),
            Err(ChunkProductError::LogSize { log_t }) if log_t == exponent
        ));
    }
}

#[test]
fn column_width_rejects_oversized_tables_on_one_cycle() {
    for bits in [9, 59] {
        let source = Arc::new(UniformColumns {
            trace: SyntheticTrace::new(SynthProfile::UniformDigits, 1, 1, 0xc8_0011).unwrap(),
            columns: 1,
            top_bits: bits,
            missing: false,
            cycles_override: Some(1),
        });
        let validated = Arc::new(ValidatedTrace::new(source).unwrap());
        let selected = DigitColumns::from_validated(validated, vec![0]).unwrap();
        assert_eq!(selected.index(0, 0), Some(0));
        assert!(matches!(
            ChunkProductCore::new(
                selected,
                vec![vec![ZERO; bits]],
                ChunkWeight::Dense(vec![ONE])
            ),
            Err(ChunkProductError::ColumnWidth { column: 0, bits: rejected }) if rejected == bits
        ));
    }
}

#[test]
fn round_wrapper_reports_empty_eq_point() {
    let source = Arc::new(UniformColumns {
        trace: SyntheticTrace::new(SynthProfile::UniformDigits, 1, 1, 0xc8_0011).unwrap(),
        columns: 1,
        top_bits: 4,
        missing: false,
        cycles_override: Some(1),
    });
    let validated = Arc::new(ValidatedTrace::new(source).unwrap());
    let selected = DigitColumns::from_validated(validated, vec![0]).unwrap();
    assert!(matches!(
        ChunkProductCore::new(
            selected,
            vec![vec![ONE; 4]],
            ChunkWeight::EqTerms(vec![EqTerm {
                coefficient: ONE,
                point: vec![],
                claim: ZERO
            }])
        ),
        Err(ChunkProductError::Round(RoundError::EmptyPoint))
    ));
}

#[test]
fn final_values_require_completed_rounds() {
    let core = ChunkProductCore::new(
        columns(3, 1, 4, false),
        vec![vec![ONE; 4]],
        ChunkWeight::Dense(vec![ONE; 8]),
    )
    .unwrap();
    assert!(matches!(
        core.final_values(),
        Err(ChunkProductError::Unfinished)
    ));
}

#[test]
fn malformed_round_order_returns_sumcheck_errors() {
    let mut core = ChunkProductCore::new(
        columns(3, 1, 4, false),
        vec![vec![ONE; 4]],
        ChunkWeight::Dense(vec![ONE; 8]),
    )
    .unwrap();
    assert!(core.prove_round(None, 1, ZERO).is_err());
    assert!(core.prove_round(Some(ONE), 0, ZERO).is_err());
    assert!(core.finish_rounds(ONE).is_err());
}
