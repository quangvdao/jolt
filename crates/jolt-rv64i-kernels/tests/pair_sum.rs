#![cfg(feature = "test-utils")]
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "acceptance fixtures and assertions fail by panicking"
)]

#[path = "../benches/support/allocator.rs"]
mod allocator;

use allocator::{AllocationMeasurement, CountingAllocator, RAYON_WORKER_ALLOWANCE};
use jolt_field::{Field, F128};
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_rv64i_kernels::oracle::{mle_at, round_polynomial};
use jolt_rv64i_kernels::packed::scatter::ScatterPlan;
use jolt_rv64i_kernels::pair_sum::{PairSumCore, PairSumError};
use jolt_rv64i_kernels::router::fold::{fold_pass, FoldLayout};
use jolt_rv64i_kernels::router::routed_columns;
use jolt_rv64i_kernels::router::shape::{
    synthetic_router_shapes, BitEntry, RouteEntry, RouterError, RouterShape, RouterShapeRequest,
    WordSlot,
};
use jolt_rv64i_kernels::source::{CycleSource, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, BooleanHypercube, ClearProof, ClearSumcheckRecorder,
    ProveRounds, SequentialRounds, SumcheckClaim, SumcheckError, SumcheckProof, SumcheckRecorder,
    SumcheckVerifier, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::ThreadPoolBuilder;
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);
const LABEL: &[u8] = b"rv64i-pair-sum-definition";

struct Fixture {
    pairs: Vec<(Vec<F128>, Vec<F128>)>,
}

impl Fixture {
    fn new(count: usize, rounds: usize) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(617 + count as u64 * 32 + rounds as u64);
        Self {
            pairs: (0..count)
                .map(|_| {
                    (
                        (0..1 << rounds).map(|_| F128::random(&mut rng)).collect(),
                        (0..1 << rounds).map(|_| F128::random(&mut rng)).collect(),
                    )
                })
                .collect(),
        }
    }

    fn claim(&self) -> F128 {
        self.pairs
            .iter()
            .flat_map(|(h, r)| h.iter().zip(r).map(|(&h, &r)| h * r))
            .sum()
    }

    fn message(&self, bound: &[F128]) -> UnivariatePoly<F128> {
        let leaves: Vec<_> = self
            .pairs
            .iter()
            .flat_map(|(h, r)| [h.as_slice(), r.as_slice()])
            .collect();
        round_polynomial(&leaves, bound, 2, |values| {
            values.chunks_exact(2).map(|pair| pair[0] * pair[1]).sum()
        })
        .unwrap()
    }

    fn values(&self, point: &[F128]) -> Vec<(F128, F128)> {
        self.pairs
            .iter()
            .map(|(h, r)| (mle_at(h, point).unwrap(), mle_at(r, point).unwrap()))
            .collect()
    }

    fn verify(&self, proof: &SumcheckProof<F128, ()>, rounds: usize) -> bool {
        let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
            panic!("compressed clear proof required")
        };
        let mut transcript = Blake2bTranscript::<F128>::new(LABEL);
        SumcheckVerifier::verify_compressed(
            &SumcheckClaim::new(rounds, 2, self.claim()),
            proof,
            BooleanHypercube,
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            &mut transcript,
        )
        .is_ok_and(|reduced| {
            reduced.value
                == self
                    .values(reduced.point.as_slice())
                    .iter()
                    .map(|&(h, r)| h * r)
                    .sum()
        })
    }

    fn prove_and_check(&self, rounds: usize) {
        let mut core = PairSumCore::new(self.pairs.clone()).unwrap();
        let prelude = BatchPrelude::try_new(
            vec![BatchMember {
                input_claim: self.claim(),
                coefficient: ONE,
                rounds,
                offset: 0,
            }],
            rounds,
            2,
        )
        .unwrap();
        let mut recorder = ClearSumcheckRecorder::<F128>::new();
        let mut transcript = Blake2bTranscript::<F128>::new(LABEL);
        let proved = prove_batch(
            &prelude,
            &mut [&mut core],
            &mut SequentialRounds,
            &mut recorder,
            &mut transcript,
        )
        .unwrap();
        let recorded = recorder
            .finish(&proved.member_claims, &mut transcript)
            .unwrap();
        let SumcheckProof::Clear(ClearProof::Compressed(proof)) = &recorded.proof else {
            panic!("compressed clear proof required")
        };
        let mut claim = self.claim();
        for (round, message) in proof.round_polynomials.iter().enumerate() {
            let expected = self.message(&proved.challenges[..round]);
            assert_eq!(message.decompress(claim), expected, "round {round}");
            claim = expected.evaluate(proved.challenges[round]);
        }
        assert_eq!(core.final_values(), self.values(&proved.challenges));
        assert!(self.verify(&recorded.proof, rounds));
        let mut changed = recorded.proof;
        let SumcheckProof::Clear(ClearProof::Compressed(proof)) = &mut changed else {
            panic!("compressed clear proof required")
        };
        let last = proof.round_polynomials.last_mut().unwrap();
        let mut coefficients = last.coeffs_except_linear_term().to_vec();
        coefficients[0] += ONE;
        *last = CompressedPoly::new(coefficients);
        assert!(!self.verify(&changed, rounds));
    }
}

#[test]
fn pair_sums_all_pair_counts_and_dimensions_match_oracle_and_reject_tampering() {
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        for count in 1..=8 {
            for rounds in 1..=10 {
                Fixture::new(count, rounds).prove_and_check(rounds);
            }
        }
    });
}

#[test]
fn pair_sum_smallest_literal_and_round_lifecycle() {
    let fixture = Fixture {
        pairs: vec![(vec![ONE, F128::from_raw(2)], vec![F128::from_raw(3), ONE])],
    };
    assert_eq!(fixture.claim(), ONE);
    let mut core = PairSumCore::new(fixture.pairs.clone()).unwrap();
    assert!(core.final_values().is_empty());
    assert!(matches!(
        core.finish_rounds(ONE),
        Err(SumcheckError::MissingEvaluationSource { .. })
    ));
    assert!(matches!(
        core.prove_round(None, 1, ONE),
        Err(SumcheckError::WrongNumberOfRounds {
            expected: 0,
            got: 1
        })
    ));
    assert!(matches!(
        core.prove_round(Some(ONE), 0, ONE),
        Err(SumcheckError::MissingEvaluationSource { .. })
    ));
    let message = core.prove_round(None, 0, ONE).unwrap();
    assert_eq!(message.coefficients(), [3, 7, 6].map(F128::from_raw));
    assert!(matches!(
        core.prove_round(None, 0, ONE),
        Err(SumcheckError::WrongNumberOfRounds { .. })
    ));
    let challenge = F128::from_raw(19);
    core.finish_rounds(challenge).unwrap();
    assert_eq!(core.final_values(), fixture.values(&[challenge]));
    assert!(matches!(
        core.finish_rounds(challenge),
        Err(SumcheckError::MissingEvaluationSource { .. })
    ));
    assert!(matches!(
        core.prove_round(Some(challenge), 1, ONE),
        Err(SumcheckError::WrongNumberOfRounds { .. })
    ));
    let mut core = PairSumCore::new(Fixture::new(2, 2).pairs).unwrap();
    let _ = core.prove_round(None, 0, ZERO).unwrap();
    assert!(matches!(
        core.prove_round(None, 1, ZERO),
        Err(SumcheckError::MissingEvaluationSource { .. })
    ));
    let _ = core.prove_round(Some(challenge), 1, ZERO).unwrap();
    core.finish_rounds(challenge).unwrap();
}

#[test]
fn pair_sum_rejects_pair_counts_and_each_invalid_table_geometry() {
    for count in [0, 9] {
        assert!(
            matches!(PairSumCore::new(vec![(vec![ZERO; 2], vec![ZERO; 2]); count]), Err(PairSumError::Pairs { count: actual }) if actual == count)
        );
    }
    for length in [0, 1, 12] {
        assert!(
            matches!(PairSumCore::new(vec![(vec![ZERO; length], vec![ZERO; length])]), Err(PairSumError::Length { pair: 0, table: "H", actual, .. }) if actual == length)
        );
    }
    assert!(matches!(
        PairSumCore::new(vec![(vec![ZERO; 8], vec![ZERO; 4])]),
        Err(PairSumError::Length {
            pair: 0,
            table: "R",
            expected: 8,
            actual: 4
        })
    ));
    assert!(matches!(
        PairSumCore::new(vec![
            (vec![ZERO; 8], vec![ZERO; 8]),
            (vec![ZERO; 4], vec![ZERO; 4])
        ]),
        Err(PairSumError::Length {
            pair: 1,
            table: "H",
            expected: 8,
            actual: 4
        })
    ));
}

#[test]
fn pair_sum_messages_and_final_values_match_oracle_on_one_and_twelve_threads() {
    let rounds = 13;
    let fixture = Fixture::new(8, rounds);
    let challenges: Vec<_> = (0..rounds)
        .map(|i| F128::from_raw(211 + i as u128))
        .collect();
    let messages: Vec<_> = (0..rounds)
        .map(|round| fixture.message(&challenges[..round]))
        .collect();
    let values = fixture.values(&challenges);
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut core = PairSumCore::new(fixture.pairs.clone()).unwrap();
            let mut claim = fixture.claim();
            for (round, expected) in messages.iter().enumerate() {
                let message = core
                    .prove_round(round.checked_sub(1).map(|i| challenges[i]), round, claim)
                    .unwrap();
                assert_eq!(message.coefficients(), expected.coefficients());
                claim = expected.evaluate(challenges[round]);
            }
            core.finish_rounds(challenges[rounds - 1]).unwrap();
            assert_eq!(core.final_values(), values);
        });
    }
}

#[test]
fn pair_sum_allocations_do_not_grow_with_chunks_and_only_final_pairs_remain() {
    for (threads, dimensions) in [(1, [8, 14]), (12, [13, 19])] {
        // Keeping the pool alive prevents teardown from hiding retained storage.
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let _ = pool.broadcast(|_| ());
        pool.install(|| {
            let count_allowance = if threads == 12 {
                RAYON_WORKER_ALLOWANCE.allocs * threads
            } else {
                0
            };
            let mut counts = [0; 2];
            for (index, rounds) in dimensions.into_iter().enumerate() {
                let resident = CountingAllocator::live_bytes();
                let fixture = Fixture::new(3, rounds);
                let mut claim = fixture.claim();
                let mut core = PairSumCore::new(fixture.pairs).unwrap();
                let measurement = AllocationMeasurement::begin();
                for round in 0..rounds {
                    let challenge = F128::from_raw(61 + round as u128);
                    let message = core
                        .prove_round(
                            round.checked_sub(1).map(|i| F128::from_raw(61 + i as u128)),
                            round,
                            claim,
                        )
                        .unwrap();
                    claim = message.evaluate(challenge);
                }
                core.finish_rounds(F128::from_raw(60 + rounds as u128))
                    .unwrap();
                let stats = measurement.finish();
                counts[index] = stats.allocs;
                assert!(
                    stats.allocs <= 16 * rounds + 64 + count_allowance,
                    "{} allocations at {threads} threads, {rounds} rounds",
                    stats.allocs
                );
                let output_bytes = size_of_val(core.final_values());
                let live = CountingAllocator::live_bytes();
                assert!(
                    (resident + output_bytes
                        ..=resident + output_bytes + RAYON_WORKER_ALLOWANCE.bytes * threads)
                        .contains(&live),
                    "retained {} bytes for {output_bytes} output bytes at {threads} threads, {rounds} rounds",
                    live.saturating_sub(resident)
                );
                drop(core);
                assert!(
                    (resident..=resident + RAYON_WORKER_ALLOWANCE.bytes * threads)
                        .contains(&CountingAllocator::live_bytes())
                );
            }
            assert!(
                counts[1] <= counts[0] + 16 * (dimensions[1] - dimensions[0]) + count_allowance
            );
        });
    }
}

fn equality(point: &[F128], index: usize) -> F128 {
    point.iter().enumerate().fold(ONE, |weight, (bit, &r)| {
        weight * (ONE + r + F128::from_raw(((index >> bit) & 1) as u128))
    })
}

fn source_word<S: CycleSource>(source: &S, word: &WordSlot, cycle: usize) -> u64 {
    match word {
        WordSlot::Trace(word) => source.trace_word(*word, cycle),
        WordSlot::Bytecode(word) => source.bytecode_word(*word, source.bytecode_index(cycle)),
        WordSlot::Zero => 0,
        WordSlot::Bits(entries) => entries.iter().enumerate().fold(0, |word, (bit, entry)| {
            let set = match *entry {
                BitEntry::Indicator { column, value } => source.digit(column, cycle) == Some(value),
                BitEntry::DigitBit { column, bit } => source
                    .digit(column, cycle)
                    .is_some_and(|value| value & (1 << bit) != 0),
                BitEntry::One => true,
                BitEntry::Zero => false,
            };
            word | (u64::from(set) << bit)
        }),
    }
}

fn selector<S: CycleSource>(source: &S, shape: &RouterShape, cycle: usize) -> Option<usize> {
    let mut selector = 0;
    let mut shift = 0;
    for factor in shape.factors() {
        selector |= source.digit(factor.column, cycle)? << shift;
        shift += factor.slots.len();
    }
    Some(selector)
}

struct RoutedFixture {
    trace: Arc<ValidatedTrace<SyntheticTrace>>,
    shapes: Vec<RouterShape>,
    point: Vec<F128>,
}

impl RoutedFixture {
    fn new(log_t: usize) -> Self {
        let source = Arc::new(
            SyntheticTrace::new(SynthProfile::AllRows, log_t, 1 << (log_t - 2), 812).unwrap(),
        );
        let trace = Arc::new(ValidatedTrace::new(source.clone()).unwrap());
        let mut rng = ChaCha20Rng::seed_from_u64(1182);
        let shapes = synthetic_router_shapes()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(shape_index, base)| {
                let mut route: Vec<_> = (0..48)
                    .map(|_| {
                        let raw = F128::random(&mut rng).to_raw();
                        RouteEntry {
                            output: raw as usize & ((1 << base.log_outputs()) - 1),
                            source: (raw >> 32) as usize & (64 * base.bank().len() - 1),
                            selector: (raw >> 64) as usize & (base.selectors() - 1),
                        }
                    })
                    .collect();
                let mut long_route = false;
                for cycle in 0..source.cycles().min(8) {
                    let Some(selector) = selector(source.as_ref(), &base, cycle) else {
                        continue;
                    };
                    for (word, entry) in base.bank().iter().enumerate() {
                        let bits = source_word(source.as_ref(), entry, cycle);
                        if bits != 0 {
                            route.push(RouteEntry {
                                output: 64 * cycle + word,
                                source: 64 * word + bits.trailing_zeros() as usize,
                                selector,
                            });
                            if !long_route {
                                let omitted = (bits.count_ones() & 1 == 0)
                                    .then_some(bits.trailing_zeros() as usize);
                                route.extend(
                                    (0..u64::BITS as usize)
                                        .filter(|&bit| Some(bit) != omitted)
                                        .map(|bit| RouteEntry {
                                            output: (1 << base.log_outputs()) - 1 - shape_index,
                                            source: u64::BITS as usize * word + bit,
                                            selector,
                                        }),
                                );
                                long_route = true;
                            }
                        }
                    }
                }
                RouterShape::new(RouterShapeRequest {
                    slots: base.slots(),
                    bank: base.bank().to_vec(),
                    factors: base.factors().to_vec(),
                    word_slots: base.word_slots().to_vec(),
                    log_outputs: base.log_outputs(),
                    route,
                })
                .unwrap()
            })
            .collect();
        Self {
            trace,
            shapes,
            point: (0..log_t).map(|_| F128::random(&mut rng)).collect(),
        }
    }

    fn folds(&self) -> Vec<Vec<F128>> {
        let plan = ScatterPlan::new(self.trace.clone()).unwrap();
        let layout =
            FoldLayout::new(&self.trace, &self.shapes, &vec![vec![]; self.shapes.len()]).unwrap();
        fold_pass(&self.trace, &self.shapes, &self.point, &plan, &layout, &[])
            .unwrap()
            .folds
    }

    fn expected(&self) -> Vec<F128> {
        let source = self.trace.source();
        let mut output = vec![ZERO; 1 << self.shapes[0].log_outputs()];
        for cycle in 0..source.cycles() {
            let weight = equality(&self.point, cycle);
            for shape in &self.shapes {
                let Some(selector) = selector(source.as_ref(), shape, cycle) else {
                    continue;
                };
                for entry in shape.route() {
                    let bits =
                        source_word(source.as_ref(), &shape.bank()[entry.source / 64], cycle);
                    if selector == entry.selector && bits & (1 << (entry.source % 64)) != 0 {
                        output[entry.output] += weight;
                    }
                }
            }
        }
        output
    }
}

#[test]
fn routed_columns_of_five_complete_folds_match_direct_cycle_summation() {
    for log_t in [3, 8] {
        let fixture = RoutedFixture::new(log_t);
        let expected = fixture.expected();
        assert!(expected.iter().any(|&value| value != ZERO));
        assert_ne!(*expected.last().unwrap(), ZERO);
        for threads in [1, 12] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                assert_eq!(
                    routed_columns(&fixture.shapes, &fixture.folds()).unwrap(),
                    expected
                );
            });
        }
        let empty_shapes = synthetic_router_shapes().unwrap();
        assert!(routed_columns(&empty_shapes, &fixture.folds())
            .unwrap()
            .iter()
            .all(|&value| value == ZERO));
    }
}

#[test]
fn routed_columns_reject_missing_folds_incomplete_tables_and_output_domain_mismatch() {
    assert!(matches!(
        routed_columns(&[], &[]),
        Err(RouterError::EmptyShapes)
    ));
    let shapes = synthetic_router_shapes().unwrap();
    let mut folds: Vec<_> = shapes
        .iter()
        .map(|shape| vec![ZERO; shape.fold_len()])
        .collect();
    assert!(matches!(
        routed_columns(&shapes, &folds[..4]),
        Err(RouterError::TableLength {
            expected: 5,
            actual: 4,
            ..
        })
    ));
    folds[2].truncate(shapes[2].fold_len() / 2);
    assert!(
        matches!(routed_columns(&shapes, &folds), Err(RouterError::TableLength { expected, actual, .. }) if expected == shapes[2].fold_len() && actual * 2 == expected)
    );
    let base = &shapes[0];
    let other = RouterShape::new(RouterShapeRequest {
        slots: base.slots(),
        bank: base.bank().to_vec(),
        factors: base.factors().to_vec(),
        word_slots: base.word_slots().to_vec(),
        log_outputs: base.log_outputs() + 1,
        route: vec![],
    })
    .unwrap();
    assert!(matches!(
        routed_columns(
            &[base.clone(), other],
            &vec![vec![ZERO; base.fold_len()]; 2]
        ),
        Err(RouterError::TableLength {
            expected: 1024,
            actual: 2048,
            ..
        })
    ));
}

#[test]
fn routed_columns_allocations_are_bounded_and_retain_only_output() {
    for (threads, dimensions) in [(1, [8, 14]), (12, [13, 19])] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let _ = pool.broadcast(|_| ());
        pool.install(|| {
            let count_allowance = if threads == 12 {
                RAYON_WORKER_ALLOWANCE.allocs * threads
            } else {
                0
            };
            let mut counts = [0; 2];
            for (index, log_t) in dimensions.into_iter().enumerate() {
                let fixture = RoutedFixture::new(log_t);
                let folds = fixture.folds();
                let resident = CountingAllocator::live_bytes();
                let measurement = AllocationMeasurement::begin();
                let output = routed_columns(&fixture.shapes, &folds).unwrap();
                let stats = measurement.finish();
                counts[index] = stats.allocs;
                assert!(stats.allocs <= 16 + count_allowance);
                let bytes = output.capacity() * size_of::<F128>();
                assert!(
                    (bytes..=bytes + RAYON_WORKER_ALLOWANCE.bytes * threads)
                        .contains(&stats.final_bytes),
                    "retained {} bytes for {bytes} output bytes, peak {}",
                    stats.final_bytes,
                    stats.peak_bytes
                );
                drop(output);
                assert!(
                    (resident..=resident + RAYON_WORKER_ALLOWANCE.bytes * threads)
                        .contains(&CountingAllocator::live_bytes())
                );
            }
            assert!(counts[1] <= counts[0] + count_allowance);
        });
    }
}
