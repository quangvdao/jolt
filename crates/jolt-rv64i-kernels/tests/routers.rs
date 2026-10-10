#![cfg(feature = "test-utils")]
#![expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "test assertions and independently constructed acceptance fixtures fail by panicking"
)]

#[path = "../benches/support/allocator.rs"]
mod allocator;

use allocator::{AllocationMeasurement, CountingAllocator, RAYON_WORKER_ALLOWANCE};
use jolt_field::{Field, F128};
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_rv64i_kernels::oracle::{mle_at, round_polynomial};
use jolt_rv64i_kernels::packed::scatter::ScatterPlan;
use jolt_rv64i_kernels::par::CycleChunks;
use jolt_rv64i_kernels::round::RoundError;
use jolt_rv64i_kernels::router::claims::claims_pass;
use jolt_rv64i_kernels::router::cycle::{RouterCycleMember, RoutersCycleCore};
use jolt_rv64i_kernels::router::fold::{fold_pass, FoldLayout};
use jolt_rv64i_kernels::router::lift::{source_lift, RetainedWordLifts};
use jolt_rv64i_kernels::router::shape::{
    synthetic_router_shapes, BitEntry, RouteEntry, RouterError, RouterShape, RouterShapeRequest,
    SelectorFactor, SlotVariable, WordSlot,
};
use jolt_rv64i_kernels::router::short::RouterShortCore;
use jolt_rv64i_kernels::source::{CycleSource, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, BooleanHypercube, ClearProof, ClearSumcheckRecorder,
    MemberFinish, MemberRound, ProveRounds, RoundScheduler, SequentialRounds, SumcheckClaim,
    SumcheckError, SumcheckProof, SumcheckRecorder, SumcheckVerifier,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::{ThreadPoolBuilder, Yield};
use std::mem::size_of;
use std::sync::Arc;

const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);
const LABEL: &[u8] = b"rv64i-routers-definition";

fn shape(request: RouterShapeRequest) -> RouterShape {
    RouterShape::new(request).unwrap()
}

fn routed_shapes(rng: &mut ChaCha20Rng) -> Vec<RouterShape> {
    synthetic_router_shapes()
        .unwrap()
        .into_iter()
        .map(|base| {
            let route = (0..48)
                .map(|_| {
                    let value = F128::random(rng).to_raw();
                    RouteEntry {
                        output: value as usize & ((1 << base.log_outputs()) - 1),
                        source: (value >> 32) as usize & (64 * base.bank().len() - 1),
                        selector: (value >> 64) as usize & (base.selectors() - 1),
                    }
                })
                .collect();
            shape(RouterShapeRequest {
                slots: base.slots(),
                bank: base.bank().to_vec(),
                factors: base.factors().to_vec(),
                word_slots: base.word_slots().to_vec(),
                log_outputs: base.log_outputs(),
                route,
            })
        })
        .collect()
}

fn point(length: usize, rng: &mut ChaCha20Rng) -> Vec<F128> {
    (0..length).map(|_| F128::random(rng)).collect()
}

fn equality(point: &[F128], index: usize) -> F128 {
    point.iter().enumerate().fold(ONE, |value, (bit, &r)| {
        value * (ONE + r + F128::from_raw(((index >> bit) & 1) as u128))
    })
}

fn restriction(shape: &RouterShape, x: &[F128]) -> Vec<F128> {
    shape.slot_map().iter().map(|&(slot, _)| x[slot]).collect()
}

fn table_index(shape: &RouterShape, source: usize, selector: usize) -> usize {
    let mut factor_shift = [0; 3];
    for factor in 1..shape.factors().len() {
        factor_shift[factor] = factor_shift[factor - 1] + shape.factors()[factor - 1].slots.len();
    }
    shape
        .slot_map()
        .iter()
        .enumerate()
        .fold(0, |index, (position, &(_, variable))| {
            let bit = match variable {
                SlotVariable::Bit(bit) => source >> bit,
                SlotVariable::Word(bit) => source >> (6 + bit),
                SlotVariable::Selector { factor, bit } => selector >> (factor_shift[factor] + bit),
            } & 1;
            index | (bit << position)
        })
}

fn selector<S: CycleSource>(source: &S, shape: &RouterShape, cycle: usize) -> Option<usize> {
    let mut value = 0;
    let mut shift = 0;
    for factor in shape.factors() {
        value |= source.digit(factor.column, cycle)? << shift;
        shift += factor.slots.len();
    }
    Some(value)
}

fn source_word<S: CycleSource>(source: &S, word: &WordSlot, cycle: usize) -> u64 {
    match word {
        WordSlot::Trace(index) => source.trace_word(*index, cycle),
        WordSlot::Bytecode(index) => source.bytecode_word(*index, source.bytecode_index(cycle)),
        WordSlot::Zero => 0,
        WordSlot::Bits(entries) => entries.iter().enumerate().fold(0, |value, (bit, entry)| {
            let present = match *entry {
                BitEntry::Indicator { column, value } => source.digit(column, cycle) == Some(value),
                BitEntry::DigitBit { column, bit } => source
                    .digit(column, cycle)
                    .is_some_and(|digit| (digit >> bit) & 1 != 0),
                BitEntry::One => true,
                BitEntry::Zero => false,
            };
            value | (u64::from(present) << bit)
        }),
    }
}

fn word_extension(word: u64, bit_weights: &[F128]) -> F128 {
    bit_weights
        .iter()
        .enumerate()
        .filter_map(|(bit, &weight)| ((word >> bit) & 1 != 0).then_some(weight))
        .sum()
}

struct Definition {
    folds: Vec<Vec<F128>>,
    sources: Vec<Vec<F128>>,
    cycle_leaves: Vec<Vec<Vec<F128>>>,
}

impl Definition {
    fn new<S: CycleSource>(
        source: &S,
        shapes: &[RouterShape],
        r_cycle: &[F128],
        x: &[F128],
    ) -> Self {
        let cycle_weights: Vec<_> = (0..source.cycles())
            .map(|cycle| equality(r_cycle, cycle))
            .collect();
        let bit_weights: Vec<_> = (0..64).map(|bit| equality(&x[..6], bit)).collect();
        let mut folds = Vec::new();
        let mut sources = Vec::new();
        let mut cycle_leaves = Vec::new();
        for shape in shapes {
            let mut fold = vec![ZERO; shape.fold_len()];
            let word_point: Vec<_> = shape.word_slots().iter().map(|&slot| x[slot]).collect();
            let word_weights: Vec<_> = (0..shape.bank().len())
                .map(|word| equality(&word_point, word))
                .collect();
            let mut source_table = Vec::with_capacity(source.cycles());
            for (cycle, &weight) in cycle_weights.iter().enumerate() {
                let h = selector(source, shape, cycle);
                let mut lifted = ZERO;
                for (word_index, word) in shape.bank().iter().enumerate() {
                    let mut bits = source_word(source, word, cycle);
                    lifted += word_weights[word_index] * word_extension(bits, &bit_weights);
                    if let Some(h) = h {
                        while bits != 0 {
                            let bit = bits.trailing_zeros() as usize;
                            fold[table_index(shape, bit + 64 * word_index, h)] += weight;
                            bits &= bits - 1;
                        }
                    }
                }
                source_table.push(lifted);
            }
            let mut leaves = vec![cycle_weights.clone(), source_table.clone()];
            for factor in shape.factors() {
                let factor_point: Vec<_> = factor.slots.iter().map(|&slot| x[slot]).collect();
                leaves.push(
                    (0..source.cycles())
                        .map(|cycle| {
                            source
                                .digit(factor.column, cycle)
                                .map_or(ZERO, |digit| equality(&factor_point, digit))
                        })
                        .collect(),
                );
            }
            folds.push(fold);
            sources.push(source_table);
            cycle_leaves.push(leaves);
        }
        Self {
            folds,
            sources,
            cycle_leaves,
        }
    }

    fn claims(&self, shapes: &[RouterShape], x: &[F128]) -> Vec<F128> {
        self.folds
            .iter()
            .zip(shapes)
            .map(|(fold, shape)| mle_at(fold, &restriction(shape, x)).unwrap())
            .collect()
    }

    fn cycle_messages(&self, bound: &[F128]) -> Vec<UnivariatePoly<F128>> {
        self.cycle_leaves
            .iter()
            .map(|leaves| {
                let slices: Vec<_> = leaves.iter().map(Vec::as_slice).collect();
                round_polynomial(&slices, bound, leaves.len(), |values| {
                    values.iter().copied().product()
                })
                .unwrap()
            })
            .collect()
    }

    fn assert_claims_pass<S: CycleSource>(
        trace: &ValidatedTrace<S>,
        lifts: &RetainedWordLifts,
        plan: &ScatterPlan<S>,
        r_bit: &[F128],
        r_prime: &[F128],
    ) {
        let source = trace.source();
        let words: Vec<_> = lifts.word_indices().iter().rev().copied().collect();
        let output = claims_pass(trace, lifts, &words, plan, r_prime).unwrap();
        let bit_weights: Vec<_> = (0..64).map(|bit| equality(r_bit, bit)).collect();
        let cycle_weights: Vec<_> = (0..source.cycles())
            .map(|cycle| equality(r_prime, cycle))
            .collect();
        let mut row_weights = vec![ZERO; source.bytecode_rows()];
        for (cycle, &weight) in cycle_weights.iter().enumerate() {
            row_weights[source.bytecode_index(cycle)] += weight;
        }
        assert_eq!(output.row_weights, row_weights);
        let trace_values: Vec<F128> = words
            .iter()
            .map(|&word| {
                (0..source.cycles())
                    .map(|cycle| {
                        cycle_weights[cycle]
                            * word_extension(source.trace_word(word, cycle), &bit_weights)
                    })
                    .sum()
            })
            .collect();
        assert_eq!(output.trace_words, trace_values);
        let bytecode_values: Vec<F128> = (0..source.bytecode_words())
            .map(|word| {
                (0..source.bytecode_rows())
                    .map(|row| {
                        row_weights[row]
                            * word_extension(source.bytecode_word(word, row), &bit_weights)
                    })
                    .sum()
            })
            .collect();
        assert_eq!(output.bytecode_words, bytecode_values);
        for (&word, table) in lifts.word_indices().iter().zip(lifts.tables()) {
            let expected: Vec<_> = (0..source.cycles())
                .map(|cycle| word_extension(source.trace_word(word, cycle), &bit_weights))
                .collect();
            assert_eq!(*table, expected);
        }
    }
}

struct ShortDefinition {
    leaves: Vec<Vec<F128>>,
    summand: Vec<F128>,
    weights: Vec<Vec<F128>>,
}

impl ShortDefinition {
    fn new(shapes: &[RouterShape], w: &[F128], folds: &[Vec<F128>]) -> Self {
        let mut leaves = Vec::new();
        let mut summand = vec![ZERO; 1 << shapes[0].slots()];
        let mut weights = Vec::new();
        for (shape, fold) in shapes.iter().zip(folds) {
            let mut weight = vec![ZERO; shape.fold_len()];
            for &RouteEntry {
                output,
                source,
                selector,
            } in shape.route()
            {
                weight[table_index(shape, source, selector)] += equality(w, output);
            }
            let mut dense_w = Vec::with_capacity(summand.len());
            let mut dense_fold = Vec::with_capacity(summand.len());
            let mut dense_idle = Vec::with_capacity(summand.len());
            for (u, sum) in summand.iter_mut().enumerate() {
                let index = shape
                    .slot_map()
                    .iter()
                    .enumerate()
                    .fold(0, |index, (bit, &(slot, _))| {
                        index | (((u >> slot) & 1) << bit)
                    });
                let idle = if shape.idle_slots().iter().any(|&slot| (u >> slot) & 1 != 0) {
                    ZERO
                } else {
                    ONE
                };
                dense_w.push(weight[index]);
                dense_fold.push(fold[index]);
                dense_idle.push(idle);
                *sum += idle * weight[index] * fold[index];
            }
            leaves.extend([dense_idle, dense_w, dense_fold]);
            weights.push(weight);
        }
        Self {
            leaves,
            summand,
            weights,
        }
    }

    fn claim(&self) -> F128 {
        self.summand.iter().copied().sum()
    }

    fn round(&self, bound: &[F128]) -> UnivariatePoly<F128> {
        let leaves: Vec<_> = self.leaves.iter().map(Vec::as_slice).collect();
        round_polynomial(&leaves, bound, 2, |values| {
            values.chunks_exact(3).map(|v| v[0] * v[1] * v[2]).sum()
        })
        .unwrap()
    }

    fn final_values(
        &self,
        shapes: &[RouterShape],
        folds: &[Vec<F128>],
        x: &[F128],
    ) -> Vec<(F128, F128)> {
        shapes
            .iter()
            .zip(folds)
            .zip(&self.weights)
            .map(|((shape, fold), weight)| {
                let idle = shape
                    .idle_slots()
                    .iter()
                    .map(|&slot| ONE + x[slot])
                    .product::<F128>();
                let restricted = restriction(shape, x);
                (
                    mle_at(fold, &restricted).unwrap(),
                    idle * mle_at(weight, &restricted).unwrap(),
                )
            })
            .collect()
    }
}

struct Recording<C> {
    inner: C,
    messages: Vec<UnivariatePoly<F128>>,
}

impl<C: ProveRounds<F128>> ProveRounds<F128> for Recording<C> {
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }
    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        let message = self.inner.prove_round(bind, round, claim)?;
        self.messages.push(message.clone());
        Ok(message)
    }
    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        self.inner.finish_rounds(bind)
    }
}

struct ReverseRounds;
impl RoundScheduler<F128> for ReverseRounds {
    fn batch_prove_round(
        &mut self,
        work: &mut [MemberRound<'_, F128>],
    ) -> Result<(), SumcheckError<F128>> {
        for item in work.iter_mut().rev() {
            item.run()?;
        }
        Ok(())
    }
    fn batch_finish_rounds(
        &mut self,
        work: &mut [MemberFinish<'_, F128>],
    ) -> Result<(), SumcheckError<F128>> {
        for item in work.iter_mut().rev() {
            item.run()?;
        }
        Ok(())
    }
}

fn verify(
    proof: &SumcheckProof<F128, ()>,
    prelude: &BatchPrelude<F128>,
    final_value: impl Fn(&[F128]) -> F128,
) -> bool {
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
        panic!("compressed clear proof required")
    };
    let mut transcript = Blake2bTranscript::<F128>::new(LABEL);
    let Ok(reduced) = SumcheckVerifier::verify_compressed(
        &SumcheckClaim::new(
            prelude.max_num_vars,
            prelude.max_degree,
            prelude.claimed_sum,
        ),
        proof,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut transcript,
    ) else {
        return false;
    };
    reduced.value == final_value(reduced.point.as_slice())
}

#[test]
fn short_seventeen_slots_match_defining_sum_final_values_and_verify() {
    let mut rng = ChaCha20Rng::seed_from_u64(1401);
    let source = SyntheticTrace::new(SynthProfile::AllRows, 5, 8, 1401).unwrap();
    let shapes = routed_shapes(&mut rng);
    let r_cycle = point(5, &mut rng);
    let x = point(17, &mut rng);
    let w = point(10, &mut rng);
    let definition = Definition::new(&source, &shapes, &r_cycle, &x);
    let short = ShortDefinition::new(&shapes, &w, &definition.folds);
    assert_eq!(short.summand.len(), 1 << 17);
    let mut core = Recording {
        inner: RouterShortCore::new(&shapes, &w, definition.folds.clone()).unwrap(),
        messages: Vec::new(),
    };
    let prelude = BatchPrelude::try_new(
        vec![BatchMember {
            input_claim: short.claim(),
            coefficient: F128::from_raw(97),
            rounds: 17,
            offset: 0,
        }],
        17,
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
    for (round, message) in core.messages.iter().enumerate() {
        assert_eq!(
            message.coefficients(),
            short.round(&proved.challenges[..round]).coefficients()
        );
    }
    let expected = short.final_values(&shapes, &definition.folds, &proved.challenges);
    assert_eq!(core.inner.final_values().unwrap(), expected);
    assert!(verify(&recorded.proof, &prelude, |x| F128::from_raw(97)
        * short
            .final_values(&shapes, &definition.folds, x)
            .iter()
            .map(|&(fold, weight)| fold * weight)
            .sum::<F128>()));
}

fn prove_cycle_fixture<S: CycleSource>(
    trace: &ValidatedTrace<S>,
    shapes: &[RouterShape],
    r_cycle: &[F128],
    x: &[F128],
    definition: &Definition,
    reverse: bool,
) {
    let lifted = source_lift(trace, shapes, x).unwrap();
    assert_eq!(lifted.source_tables, definition.sources);
    let core = RoutersCycleCore::new(trace, shapes, r_cycle, x, lifted.source_tables).unwrap();
    let mut members: Vec<_> = core
        .members()
        .into_iter()
        .map(|inner| Recording {
            inner,
            messages: Vec::new(),
        })
        .collect();
    let claims = definition.claims(shapes, x);
    let coefficients: Vec<_> = (0..shapes.len())
        .map(|index| {
            if index == 0 && shapes.len() > 1 && reverse {
                ZERO
            } else {
                F128::from_raw(index as u128 + 29)
            }
        })
        .collect();
    let degree = shapes
        .iter()
        .map(|shape| 2 + shape.factors().len())
        .max()
        .unwrap();
    let prelude = BatchPrelude::try_new(
        claims
            .iter()
            .zip(&coefficients)
            .map(|(&input_claim, &coefficient)| BatchMember {
                input_claim,
                coefficient,
                rounds: r_cycle.len(),
                offset: 0,
            })
            .collect(),
        r_cycle.len(),
        degree,
    )
    .unwrap();
    let mut recorder = ClearSumcheckRecorder::<F128>::new();
    let mut transcript = Blake2bTranscript::<F128>::new(LABEL);
    let mut handles: Vec<&mut dyn ProveRounds<F128>> = members
        .iter_mut()
        .map(|member| member as &mut dyn ProveRounds<F128>)
        .collect();
    let mut forward = SequentialRounds;
    let mut backward = ReverseRounds;
    let scheduler: &mut dyn RoundScheduler<F128> =
        if reverse { &mut backward } else { &mut forward };
    let proved = prove_batch(
        &prelude,
        &mut handles,
        scheduler,
        &mut recorder,
        &mut transcript,
    )
    .unwrap();
    let recorded = recorder
        .finish(&proved.member_claims, &mut transcript)
        .unwrap();
    for round in 0..r_cycle.len() {
        let expected = definition.cycle_messages(&proved.challenges[..round]);
        for (member, message) in members.iter().zip(expected) {
            assert_eq!(
                member.messages[round].coefficients(),
                message.coefficients()
            );
        }
    }
    for (member, leaves) in members.iter().zip(&definition.cycle_leaves) {
        let expected: Vec<_> = leaves
            .iter()
            .map(|table| mle_at(table, &proved.challenges).unwrap())
            .collect();
        assert_eq!(
            member.inner.final_values().unwrap(),
            (expected[1], expected[2..].to_vec())
        );
    }
    let final_value = |point: &[F128]| {
        definition
            .cycle_leaves
            .iter()
            .zip(&coefficients)
            .map(|(leaves, &coefficient)| {
                coefficient
                    * leaves
                        .iter()
                        .map(|table| mle_at(table, point).unwrap())
                        .product::<F128>()
            })
            .sum::<F128>()
    };
    assert!(verify(&recorded.proof, &prelude, final_value));
    let mut changed = recorded.proof.clone();
    let SumcheckProof::Clear(ClearProof::Compressed(clear)) = &mut changed else {
        panic!("compressed clear proof required")
    };
    let last = clear.round_polynomials.last_mut().unwrap();
    let mut coefficients = last.coeffs_except_linear_term().to_vec();
    coefficients[0] += ONE;
    *last = CompressedPoly::new(coefficients);
    assert!(!verify(&changed, &prelude, final_value));
    let plan = ScatterPlan::new(Arc::new(
        ValidatedTrace::new(trace.source().clone()).unwrap(),
    ))
    .unwrap();
    Definition::assert_claims_pass(trace, &lifted.lifts, &plan, &x[..6], &proved.challenges);
}

#[test]
fn cycle_batch_sequential_and_reverse_messages_claims_and_tamper_match_definitions() {
    let mut rng = ChaCha20Rng::seed_from_u64(1402);
    let shapes = synthetic_router_shapes().unwrap();
    for log_t in [1, 2, 5, 8] {
        let source = Arc::new(
            SyntheticTrace::new(
                SynthProfile::AllRows,
                log_t,
                1 << log_t.saturating_sub(2),
                1402,
            )
            .unwrap(),
        );
        let trace = ValidatedTrace::new(source.clone()).unwrap();
        let r_cycle = point(log_t, &mut rng);
        let x = point(17, &mut rng);
        let definition = Definition::new(source.as_ref(), &shapes, &r_cycle, &x);
        for reverse in [false, true] {
            prove_cycle_fixture(&trace, &shapes, &r_cycle, &x, &definition, reverse);
        }
    }
}

struct TinyTrace {
    words: Vec<u64>,
    digits: Vec<Option<usize>>,
}

impl CycleSource for TinyTrace {
    fn cycles(&self) -> usize {
        self.words.len()
    }
    fn trace_words(&self) -> usize {
        1
    }
    fn trace_word(&self, word: usize, cycle: usize) -> u64 {
        if word == 0 {
            self.words.get(cycle).copied().unwrap_or(0)
        } else {
            0
        }
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
    fn bits(&self, _: usize) -> usize {
        0
    }
    fn by_row(&self, _: usize) -> bool {
        false
    }
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        if column == 0 {
            self.digits.get(cycle).copied().flatten()
        } else {
            None
        }
    }
    fn row_digit(&self, _: usize, _: usize) -> Option<usize> {
        None
    }
}

fn tiny_shape(bank: WordSlot, slots: usize, route: Vec<RouteEntry>) -> RouterShape {
    shape(RouterShapeRequest {
        slots,
        bank: vec![bank],
        factors: vec![SelectorFactor {
            column: 0,
            slots: vec![],
        }],
        word_slots: vec![],
        log_outputs: 0,
        route,
    })
}

#[test]
fn complete_fold_smallest_case_keeps_unrouted_bit_and_literal_quadratic() {
    let source = Arc::new(TinyTrace {
        words: vec![2, 0xd591, 0x7168, 0xa938, 0x8921, 0xb442, 0x3714, 0x6542],
        digits: vec![Some(0); 8],
    });
    let trace = Arc::new(ValidatedTrace::new(source.clone()).unwrap());
    let shapes = vec![tiny_shape(
        WordSlot::Trace(0),
        6,
        vec![RouteEntry {
            output: 0,
            source: 0,
            selector: 0,
        }],
    )];
    let plan = ScatterPlan::new(trace.clone()).unwrap();
    let layout = FoldLayout::new(&trace, &shapes, &[vec![]]).unwrap();
    let fold = fold_pass(&trace, &shapes, &[ZERO; 3], &plan, &layout, &[]).unwrap();
    let mut expected = vec![ZERO; 64];
    expected[1] = ONE;
    assert_eq!(fold.folds, [expected.clone()]);
    let mut short = RouterShortCore::new(&shapes, &[], fold.folds).unwrap();
    let message = short.prove_round(None, 0, ZERO).unwrap();
    assert_eq!(message.coefficients(), [ZERO, ONE, ONE]);
    assert_eq!(message.evaluate(F128::from_raw(2)), F128::from_raw(6));
    let x = vec![F128::from_raw(71); 6];
    let definition = Definition::new(source.as_ref(), &shapes, &[ZERO; 3], &x);
    assert_eq!(definition.folds, [expected]);
    prove_cycle_fixture(&trace, &shapes, &[ZERO; 3], &x, &definition, true);
}

#[test]
fn factor_supported_on_one_side_has_literal_cubic() {
    let t = F128::from_raw(0x0081_a976);
    let source = Arc::new(TinyTrace {
        words: vec![1, 0],
        digits: vec![None, Some(0)],
    });
    let trace = ValidatedTrace::new(source.clone()).unwrap();
    let shapes = vec![tiny_shape(WordSlot::Trace(0), 6, vec![])];
    let x = [ZERO; 6];
    let lifted = source_lift(&trace, &shapes, &x).unwrap();
    assert_eq!(lifted.source_tables, [vec![ONE, ZERO]]);
    let core = RoutersCycleCore::new(&trace, &shapes, &[t], &x, lifted.source_tables).unwrap();
    let mut members = core.members();
    let message = members[0].prove_round(None, 0, ZERO).unwrap();
    assert_eq!(message.coefficients(), [ZERO, ONE + t, t, ONE]);
    let challenge = F128::from_raw(0x1189);
    members[0].finish_rounds(challenge).unwrap();
    assert_eq!(
        members[0].final_values().unwrap(),
        (ONE + challenge, vec![challenge])
    );
    let definition = Definition::new(source.as_ref(), &shapes, &[t], &x);
    prove_cycle_fixture(&trace, &shapes, &[t], &x, &definition, true);
}

#[test]
fn complete_fold_constant_bank_covers_absence_and_boolean_cycle_points() {
    let mut rng = ChaCha20Rng::seed_from_u64(1403);
    let x = point(6, &mut rng);
    for missing in [false, true] {
        let source = Arc::new(TinyTrace {
            words: vec![0; 8],
            digits: (0..8)
                .map(|cycle| {
                    if missing && cycle % 3 == 0 {
                        None
                    } else {
                        Some(0)
                    }
                })
                .collect(),
        });
        let trace = Arc::new(ValidatedTrace::new(source.clone()).unwrap());
        let shapes = vec![tiny_shape(
            WordSlot::Bits(vec![BitEntry::One]),
            6,
            vec![RouteEntry {
                output: 0,
                source: 0,
                selector: 0,
            }],
        )];
        let plan = ScatterPlan::new(trace.clone()).unwrap();
        let layout = FoldLayout::new(&trace, &shapes, &[vec![]]).unwrap();
        let mut points = vec![point(3, &mut rng)];
        points.extend((0..8).map(|index| {
            (0..3)
                .map(|bit| F128::from_raw(((index >> bit) & 1) as u128))
                .collect()
        }));
        for r_cycle in points {
            let fold = fold_pass(&trace, &shapes, &r_cycle, &plan, &layout, &[]).unwrap();
            let expected = (0..8)
                .filter(|&cycle| source.digit(0, cycle).is_some())
                .map(|cycle| equality(&r_cycle, cycle))
                .sum::<F128>();
            let mut table = vec![ZERO; 64];
            table[0] = expected;
            assert_eq!(fold.folds, [table]);
            let lifted = source_lift(&trace, &shapes, &x).unwrap();
            assert_eq!(
                lifted.source_tables,
                [vec![x.iter().map(|&r| ONE + r).product::<F128>(); 8]]
            );
            let definition = Definition::new(source.as_ref(), &shapes, &r_cycle, &x);
            prove_cycle_fixture(&trace, &shapes, &r_cycle, &x, &definition, false);
        }
    }
}

#[test]
fn cycle_scalar_becoming_zero_and_short_idle_zero_match_definitions() {
    let mut rng = ChaCha20Rng::seed_from_u64(1404);
    let source = Arc::new(TinyTrace {
        words: vec![7, 11, 17, 23, 31, 41, 43, 53],
        digits: vec![Some(0); 8],
    });
    let trace = ValidatedTrace::new(source.clone()).unwrap();
    let shapes = vec![tiny_shape(
        WordSlot::Trace(0),
        7,
        vec![RouteEntry {
            output: 0,
            source: 0,
            selector: 0,
        }],
    )];
    let r_cycle = vec![F128::from_raw(83), ZERO, ONE];
    let mut x = point(7, &mut rng);
    x[6] = ONE;
    let definition = Definition::new(source.as_ref(), &shapes, &r_cycle, &x);
    let short_definition = ShortDefinition::new(&shapes, &[], &definition.folds);
    let mut short = RouterShortCore::new(&shapes, &[], definition.folds.clone()).unwrap();
    let mut claim = short_definition.claim();
    for round in 0..7 {
        let expected = short_definition.round(&x[..round]);
        let message = short
            .prove_round(round.checked_sub(1).map(|r| x[r]), round, claim)
            .unwrap();
        assert_eq!(message.coefficients(), expected.coefficients());
        claim = expected.evaluate(x[round]);
    }
    short.finish_rounds(x[6]).unwrap();
    assert_eq!(
        short.final_values().unwrap(),
        short_definition.final_values(&shapes, &definition.folds, &x)
    );
    assert_eq!(short.final_values().unwrap()[0].1, ZERO);
    let challenges = vec![ONE + r_cycle[0], F128::from_raw(109), F128::from_raw(127)];
    let core =
        RoutersCycleCore::new(&trace, &shapes, &r_cycle, &x, definition.sources.clone()).unwrap();
    let mut member = core.members().remove(0);
    let mut claim = definition.claims(&shapes, &x)[0];
    for round in 0..3 {
        let expected = definition.cycle_messages(&challenges[..round]).remove(0);
        let message = member
            .prove_round(round.checked_sub(1).map(|r| challenges[r]), round, claim)
            .unwrap();
        assert_eq!(message.coefficients(), expected.coefficients());
        if round != 0 {
            assert!(message
                .coefficients()
                .iter()
                .all(|&coefficient| coefficient == ZERO));
        }
        claim = expected.evaluate(challenges[round]);
    }
    member.finish_rounds(challenges[2]).unwrap();
    assert_eq!(
        member.final_values().unwrap().0,
        mle_at(&definition.sources[0], &challenges).unwrap()
    );
}

#[test]
fn malformed_router_sources_and_points_return_named_errors() {
    let source = Arc::new(SyntheticTrace::new(SynthProfile::AllRows, 3, 2, 1405).unwrap());
    let trace = Arc::new(ValidatedTrace::new(source.clone()).unwrap());
    let shapes = synthetic_router_shapes().unwrap();
    let x = [ZERO; 17];
    let r_cycle = [ZERO; 3];
    let plan = ScatterPlan::new(trace.clone()).unwrap();
    assert!(matches!(
        source_lift(&trace, &shapes, &x[..16]),
        Err(RouterError::PointLength {
            expected: 17,
            actual: 16
        })
    ));
    let lifted = source_lift(&trace, &shapes, &x).unwrap();
    let partial_lifts = source_lift(&trace, &shapes[..1], &x).unwrap();
    assert!(matches!(
        claims_pass(&trace, &partial_lifts.lifts, &[3], &plan, &r_cycle),
        Err(RouterError::MissingRetainedWord { word: 3 })
    ));
    assert!(matches!(
        claims_pass(&trace, &lifted.lifts, &[0], &plan, &r_cycle[..2]),
        Err(RouterError::PointLength {
            expected: 3,
            actual: 2
        })
    ));
    assert!(matches!(
        claims_pass(
            &trace,
            &lifted.lifts,
            &[source.trace_words()],
            &plan,
            &r_cycle
        ),
        Err(RouterError::WordIndex { index: 6, .. })
    ));
    assert!(matches!(
        RouterShortCore::new(
            &shapes,
            &[ZERO; 11],
            shapes
                .iter()
                .map(|shape| vec![ZERO; shape.fold_len()])
                .collect()
        ),
        Err(RouterError::PointLength {
            expected: 10,
            actual: 11
        })
    ));
    let mut folds: Vec<_> = shapes
        .iter()
        .map(|shape| vec![ZERO; shape.fold_len()])
        .collect();
    folds[0].truncate(shapes[0].fold_len() / 2);
    assert!(
        matches!(RouterShortCore::new(&shapes, &[ZERO; 10], folds), Err(RouterError::TableLength { actual, .. }) if actual == shapes[0].fold_len() / 2)
    );
    let mut source_tables = vec![vec![ZERO; 8]; 5];
    source_tables[0].truncate(4);
    assert!(matches!(
        RoutersCycleCore::new(&trace, &shapes, &r_cycle, &x, source_tables),
        Err(RouterError::TableLength {
            expected: 8,
            actual: 4,
            ..
        })
    ));
    assert!(matches!(
        RoutersCycleCore::new(&trace, &shapes, &[ZERO; 4], &x, vec![vec![ZERO; 8]; 5]),
        Err(RouterError::PointLength {
            expected: 3,
            actual: 4
        })
    ));
    assert!(matches!(
        RoutersCycleCore::new(&trace, &shapes, &r_cycle, &x[..16], vec![vec![ZERO; 8]; 5]),
        Err(RouterError::PointLength {
            expected: 17,
            actual: 16
        })
    ));
    let factor = || SelectorFactor {
        column: 12,
        slots: (6..12).collect(),
    };
    let invalid_shapes = [
        (
            WordSlot::Trace(source.trace_words()),
            factor(),
            RouterError::WordIndex {
                bank: "trace",
                index: 6,
                words: 6,
            },
        ),
        (
            WordSlot::Bytecode(source.bytecode_words()),
            factor(),
            RouterError::WordIndex {
                bank: "bytecode",
                index: 4,
                words: 4,
            },
        ),
        (
            WordSlot::Zero,
            SelectorFactor {
                column: 21,
                slots: vec![],
            },
            RouterError::Column {
                column: 21,
                columns: 21,
            },
        ),
        (
            WordSlot::Bits(vec![BitEntry::Indicator {
                column: 21,
                value: 0,
            }]),
            factor(),
            RouterError::Column {
                column: 21,
                columns: 21,
            },
        ),
        (
            WordSlot::Bits(vec![BitEntry::DigitBit { column: 21, bit: 0 }]),
            factor(),
            RouterError::Column {
                column: 21,
                columns: 21,
            },
        ),
        (
            WordSlot::Zero,
            SelectorFactor {
                column: 0,
                slots: (6..9).collect(),
            },
            RouterError::FactorWidth {
                column: 0,
                expected: 4,
                actual: 3,
            },
        ),
        (
            WordSlot::Bits(vec![BitEntry::Indicator {
                column: 0,
                value: 16,
            }]),
            factor(),
            RouterError::Entry {
                column: 0,
                kind: "indicator",
                value: 16,
                bound: 16,
            },
        ),
        (
            WordSlot::Bits(vec![BitEntry::DigitBit { column: 0, bit: 4 }]),
            factor(),
            RouterError::Entry {
                column: 0,
                kind: "digit bit",
                value: 4,
                bound: 4,
            },
        ),
    ];
    for (bank, factor, expected) in invalid_shapes {
        let invalid = vec![shape(RouterShapeRequest {
            slots: 17,
            bank: vec![bank],
            factors: vec![factor],
            word_slots: vec![],
            log_outputs: 0,
            route: vec![],
        })];
        assert_eq!(source_lift(&trace, &invalid, &x).err().unwrap(), expected);
        assert_eq!(
            RoutersCycleCore::new(&trace, &invalid, &r_cycle, &x, vec![vec![ZERO; 8]])
                .err()
                .unwrap(),
            expected
        );
    }
}

#[test]
fn router_pipeline_two_and_four_chunks_match_one_oracle_per_size_on_one_and_twelve_threads() {
    let mut rng = ChaCha20Rng::seed_from_u64(1406);
    for log_t in [13_usize, 14] {
        let source = Arc::new(
            SyntheticTrace::new(SynthProfile::AllRows, log_t, 1 << (log_t - 2), 1406).unwrap(),
        );
        let trace = Arc::new(ValidatedTrace::new(source.clone()).unwrap());
        let shapes = routed_shapes(&mut rng);
        let r_cycle = point(log_t, &mut rng);
        let x = point(17, &mut rng);
        let r_prime = point(log_t, &mut rng);
        let w = point(10, &mut rng);
        assert_eq!(
            CycleChunks::new(log_t, 0).unwrap().ranges().len(),
            1 << (log_t - 12)
        );
        let definition = Definition::new(source.as_ref(), &shapes, &r_cycle, &x);
        let short_definition = ShortDefinition::new(&shapes, &w, &definition.folds);
        let short_messages: Vec<_> = (0..17)
            .map(|round| short_definition.round(&x[..round]))
            .collect();
        let short_final = short_definition.final_values(&shapes, &definition.folds, &x);
        let cycle_messages: Vec<_> = (0..log_t)
            .map(|round| definition.cycle_messages(&r_prime[..round]))
            .collect();
        let cycle_final: Vec<_> = definition
            .cycle_leaves
            .iter()
            .map(|leaves| {
                leaves
                    .iter()
                    .map(|table| mle_at(table, &r_prime).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect();
        let bit_weights: Vec<_> = (0..64).map(|bit| equality(&x[..6], bit)).collect();
        let trace_lifts: Vec<Vec<_>> = (0..source.trace_words())
            .map(|word| {
                (0..source.cycles())
                    .map(|cycle| word_extension(source.trace_word(word, cycle), &bit_weights))
                    .collect()
            })
            .collect();
        let trace_claims: Vec<_> = trace_lifts
            .iter()
            .map(|table| mle_at(table, &r_prime).unwrap())
            .collect();
        let mut expected_rows = vec![ZERO; source.bytecode_rows()];
        let mut fold_rows = expected_rows.clone();
        for cycle in 0..source.cycles() {
            expected_rows[source.bytecode_index(cycle)] += equality(&r_prime, cycle);
            fold_rows[source.bytecode_index(cycle)] += equality(&r_cycle, cycle);
        }
        let bytecode_claims: Vec<F128> = (0..source.bytecode_words())
            .map(|word| {
                (0..source.bytecode_rows())
                    .map(|row| {
                        expected_rows[row]
                            * word_extension(source.bytecode_word(word, row), &bit_weights)
                    })
                    .sum()
            })
            .collect();
        let plan = ScatterPlan::new(trace.clone()).unwrap();
        let layout = FoldLayout::new(&trace, &shapes, &vec![vec![]; shapes.len()]).unwrap();
        for threads in [1, 12] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let folded = fold_pass(&trace, &shapes, &r_cycle, &plan, &layout, &[]).unwrap();
                assert_eq!(folded.folds, definition.folds);
                assert_eq!(folded.ra_fold, fold_rows);
                let mut short = RouterShortCore::new(&shapes, &w, folded.folds).unwrap();
                let mut claim = short_definition.claim();
                for (round, expected) in short_messages.iter().enumerate() {
                    let actual = short
                        .prove_round(round.checked_sub(1).map(|round| x[round]), round, claim)
                        .unwrap();
                    assert_eq!(actual.coefficients(), expected.coefficients());
                    claim = expected.evaluate(x[round]);
                }
                short.finish_rounds(x[16]).unwrap();
                assert_eq!(short.final_values().unwrap(), short_final);
                let lifted = source_lift(&trace, &shapes, &x).unwrap();
                assert_eq!(lifted.source_tables, definition.sources);
                for (&word, table) in lifted
                    .lifts
                    .word_indices()
                    .iter()
                    .zip(lifted.lifts.tables())
                {
                    assert_eq!(*table, trace_lifts[word]);
                }
                let core =
                    RoutersCycleCore::new(&trace, &shapes, &r_cycle, &x, lifted.source_tables)
                        .unwrap();
                let mut members = core.members();
                let mut claims = definition.claims(&shapes, &x);
                for (round, expected) in cycle_messages.iter().enumerate() {
                    for member in (0..members.len()).rev() {
                        let message = members[member]
                            .prove_round(
                                round.checked_sub(1).map(|round| r_prime[round]),
                                round,
                                claims[member],
                            )
                            .unwrap();
                        assert_eq!(message.coefficients(), expected[member].coefficients());
                        claims[member] = expected[member].evaluate(r_prime[round]);
                    }
                }
                for member in members.iter_mut().rev() {
                    member.finish_rounds(r_prime[log_t - 1]).unwrap();
                }
                for (member, expected) in members.iter().zip(&cycle_final) {
                    assert_eq!(
                        member.final_values().unwrap(),
                        (expected[1], expected[2..].to_vec())
                    );
                }
                let words: Vec<_> = (0..source.trace_words()).collect();
                let claims = claims_pass(&trace, &lifted.lifts, &words, &plan, &r_prime).unwrap();
                assert_eq!(claims.trace_words, trace_claims);
                assert_eq!(claims.bytecode_words, bytecode_claims);
                assert_eq!(claims.row_weights, expected_rows);
            });
        }
    }
}

fn drive_members(members: &mut [RouterCycleMember], claims: &mut [F128], challenges: &[F128]) {
    for (round, &challenge) in challenges.iter().enumerate() {
        for (member, claim) in members.iter_mut().zip(claims.iter_mut()) {
            let message = member
                .prove_round(
                    round.checked_sub(1).map(|round| challenges[round]),
                    round,
                    *claim,
                )
                .unwrap();
            *claim = message.evaluate(challenge);
        }
    }
    for member in members {
        member
            .finish_rounds(challenges[challenges.len() - 1])
            .unwrap();
    }
}

#[test]
fn router_lift_claims_and_core_allocation_bounds_hold_through_thirty_two_chunks() {
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        let workers = pool.current_num_threads();
        let allowance_allocs = RAYON_WORKER_ALLOWANCE.allocs * workers;
        let allowance_bytes = RAYON_WORKER_ALLOWANCE.bytes * workers;
        let mut rng = ChaCha20Rng::seed_from_u64(1407);
        let shapes = synthetic_router_shapes().unwrap();
        let x = point(17, &mut rng);
        let mut lift_allocations = None;
        let mut claim_allocations = None;
        let mut cycle_allocations = None;
        for log_t in [8, 14, 19] {
            let source = Arc::new(
                SyntheticTrace::new(
                    SynthProfile::AllRows,
                    log_t,
                    1 << log_t.saturating_sub(2),
                    1407,
                )
                .unwrap(),
            );
            let trace = Arc::new(ValidatedTrace::new(source.clone()).unwrap());
            let plan = ScatterPlan::new(trace.clone()).unwrap();
            let r_cycle = point(log_t, &mut rng);
            let challenges = point(log_t, &mut rng);
            if log_t == 19 {
                assert!(CycleChunks::new(log_t, 0).unwrap().ranges().len() >= 32);
            }
            drop(source_lift(&trace, &shapes, &x).unwrap());
            let baseline = CountingAllocator::live_bytes();
            let measurement = AllocationMeasurement::begin();
            let lifted = source_lift(&trace, &shapes, &x).unwrap();
            let stats = measurement.finish();
            let outputs = lifted.source_tables.capacity() * size_of::<Vec<F128>>()
                + std::mem::size_of_val(lifted.lifts.tables())
                + std::mem::size_of_val(lifted.lifts.word_indices())
                + lifted
                    .source_tables
                    .iter()
                    .chain(lifted.lifts.tables())
                    .map(|table| table.capacity() * size_of::<F128>())
                    .sum::<usize>();
            assert!(
                stats.allocs <= 256 + allowance_allocs,
                "lift allocations {} at log_t={log_t}",
                stats.allocs
            );
            if let Some(first) = lift_allocations {
                assert!(stats.allocs <= first + allowance_allocs, "lift allocations scale with chunks at log_t={log_t}: {} versus {first}", stats.allocs);
            } else {
                lift_allocations = Some(stats.allocs);
            }
            assert!((outputs..=outputs + allowance_bytes).contains(&stats.final_bytes));
            let claims: Vec<_> = lifted
                .source_tables
                .iter()
                .zip(&shapes)
                .map(|(table, shape)| {
                    (0..source.cycles())
                        .map(|cycle| {
                            let selectors = shape
                                .factors()
                                .iter()
                                .map(|factor| {
                                    let p: Vec<_> =
                                        factor.slots.iter().map(|&slot| x[slot]).collect();
                                    source
                                        .digit(factor.column, cycle)
                                        .map_or(ZERO, |digit| equality(&p, digit))
                                })
                                .product::<F128>();
                            equality(&r_cycle, cycle) * table[cycle] * selectors
                        })
                        .sum()
                })
                .collect();
            let core =
                RoutersCycleCore::new(&trace, &shapes, &r_cycle, &x, lifted.source_tables).unwrap();
            let mut members = core.members();
            let mut claims = claims;
            let measurement = AllocationMeasurement::begin();
            drive_members(&mut members, &mut claims, &challenges);
            let stats = measurement.finish();
            if log_t >= 14 {
                let pass_chunks: usize = (1..=log_t)
                    .map(|round| CycleChunks::new(log_t, round).unwrap().ranges().len())
                    .sum();
                if let Some((first_log_t, first_allocs, first_chunks)) = cycle_allocations {
                    // One jobs vector, eight shared dense selector binds, and
                    // coefficient/message vectors for each of the five members.
                    let per_round = 1 + 8 + 2 * shapes.len();
                    let growth_bound = per_round * (log_t - first_log_t) + allowance_allocs;
                    assert!(CycleChunks::new(log_t, 1).unwrap().ranges().len() >= 32);
                    assert!(pass_chunks - first_chunks > growth_bound);
                    assert!(
                        stats.allocs <= first_allocs + growth_bound,
                        "cycle allocations scale with chunks: {} at log_t={log_t} versus {first_allocs} at log_t={first_log_t}, growth bound {growth_bound}",
                        stats.allocs
                    );
                } else {
                    cycle_allocations = Some((log_t, stats.allocs, pass_chunks));
                }
            }
            let words: Vec<_> = (0..source.trace_words()).collect();
            drop(claims_pass(&trace, &lifted.lifts, &words, &plan, &challenges).unwrap());
            let measurement = AllocationMeasurement::begin();
            let output = claims_pass(&trace, &lifted.lifts, &words, &plan, &challenges).unwrap();
            let stats = measurement.finish();
            assert!(
                stats.allocs <= 256 + allowance_allocs,
                "claims allocations {} at log_t={log_t}",
                stats.allocs
            );
            if let Some(first) = claim_allocations {
                assert!(stats.allocs <= first + allowance_allocs, "claims allocations scale with chunks at log_t={log_t}: {} versus {first}", stats.allocs);
            } else {
                claim_allocations = Some(stats.allocs);
            }
            let output_bytes = (output.trace_words.capacity()
                + output.bytecode_words.capacity()
                + output.row_weights.capacity())
                * size_of::<F128>();
            let chunks = CycleChunks::new(log_t, 0).unwrap();
            let half_entries = (1 << chunks.low_bits()) + (1 << chunks.high_bits());
            let scratch_bytes = (source.cycles() + half_entries) * size_of::<F128>()
                + words.len() * size_of::<&[F128]>();
            assert!(stats.peak_bytes <= output_bytes + scratch_bytes + allowance_bytes);
            assert!((output_bytes..=output_bytes + allowance_bytes).contains(&stats.final_bytes));
            drop(output);
            drop(members);
            drop(core);
            drop(lifted.lifts);
            drop(claims);
            drop(words);
            // LazyFoldedRa queues destruction at its fourth bind; this one-worker
            // pool must finish that owned work before checking released capacities.
            while rayon::yield_now() == Some(Yield::Executed) {}
            let after_drop = CountingAllocator::live_bytes();
            assert!(
                (baseline..=baseline + allowance_bytes).contains(&after_drop),
                "log_t={log_t}: baseline={baseline}, after_drop={after_drop}, worker_allowance={allowance_bytes}"
            );
        }
        let folds: Vec<_> = shapes
            .iter()
            .map(|shape| vec![ZERO; shape.fold_len()])
            .collect();
        let mut short = RouterShortCore::new(&shapes, &[ZERO; 10], folds).unwrap();
        let mut claim = ZERO;
        let measurement = AllocationMeasurement::begin();
        for round in 0_usize..17 {
            let message = short
                .prove_round(round.checked_sub(1).map(|round| x[round]), round, claim)
                .unwrap();
            claim = message.evaluate(x[round]);
        }
        short.finish_rounds(x[16]).unwrap();
        let stats = measurement.finish();
        assert!(stats.allocs <= 16 * 17 + 64 + allowance_allocs);

    });
}

#[test]
fn malformed_router_shape_collections_and_unfinished_values_return_errors() {
    assert!(matches!(
        RouterShortCore::new(&[], &[], vec![]),
        Err(RouterError::EmptyShapes)
    ));
    let shapes = synthetic_router_shapes().unwrap();
    assert!(matches!(
        RouterShortCore::new(&shapes, &[ZERO; 10], vec![]),
        Err(RouterError::TableLength {
            expected: 5,
            actual: 0,
            ..
        })
    ));
    let core = RouterShortCore::new(
        &shapes,
        &[ZERO; 10],
        shapes
            .iter()
            .map(|shape| vec![ZERO; shape.fold_len()])
            .collect(),
    )
    .unwrap();
    assert!(matches!(core.final_values(), Err(RouterError::Unfinished)));
    let source = Arc::new(TinyTrace {
        words: vec![0; 2],
        digits: vec![Some(0); 2],
    });
    let trace = ValidatedTrace::new(source).unwrap();
    let mixed = vec![
        tiny_shape(WordSlot::Trace(0), 6, vec![]),
        tiny_shape(WordSlot::Trace(0), 7, vec![]),
    ];
    assert!(matches!(
        RouterShortCore::new(&mixed, &[], vec![vec![ZERO; 64]; 2]),
        Err(RouterError::SlotCount {
            expected: 6,
            actual: 7
        })
    ));
    let core = RoutersCycleCore::new(
        &trace,
        &mixed[..1],
        &[ZERO],
        &[ZERO; 6],
        vec![vec![ZERO; 2]],
    )
    .unwrap();
    assert!(matches!(
        core.members()[0].final_values(),
        Err(RouterError::Unfinished)
    ));
    assert!(matches!(
        RoutersCycleCore::new(&trace, &[], &[ZERO], &[], vec![]),
        Err(RouterError::EmptyShapes)
    ));
    assert!(RoutersCycleCore::new(&trace, &mixed[..1], &[ZERO], &[ZERO; 6], vec![]).is_err());
}

struct WideColumns {
    width: usize,
}

impl CycleSource for WideColumns {
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
        2
    }
    fn bits(&self, column: usize) -> usize {
        if column == 1 {
            self.width
        } else {
            0
        }
    }
    fn by_row(&self, _: usize) -> bool {
        false
    }
    fn digit(&self, column: usize, cycle: usize) -> Option<usize> {
        (column < 2 && cycle < 2).then_some(0)
    }
    fn row_digit(&self, _: usize, _: usize) -> Option<usize> {
        None
    }
}

#[test]
fn router_dimension_compact_width_and_empty_cycle_point_are_rejected() {
    let width = usize::BITS as usize - 5;
    let trace = ValidatedTrace::new(Arc::new(WideColumns { width })).unwrap();
    let shapes = vec![tiny_shape(
        WordSlot::Bits(vec![BitEntry::Indicator {
            column: 1,
            value: 0,
        }]),
        6,
        vec![],
    )];
    assert!(
        matches!(source_lift(&trace, &shapes, &[ZERO; 6]), Err(RouterError::Dimension { variables }) if variables == width)
    );
    let trace = ValidatedTrace::new(Arc::new(WideColumns { width: 8 })).unwrap();
    let shapes = vec![shape(RouterShapeRequest {
        slots: 14,
        bank: vec![WordSlot::Zero],
        factors: vec![SelectorFactor {
            column: 1,
            slots: (6..14).collect(),
        }],
        word_slots: vec![],
        log_outputs: 0,
        route: vec![],
    })];
    assert!(matches!(
        RoutersCycleCore::new(&trace, &shapes, &[ZERO], &[ZERO; 14], vec![vec![ZERO; 2]]),
        Err(RouterError::FactorCapacity {
            column: 1,
            bound: 7,
            width: 8
        })
    ));
    let trace = ValidatedTrace::new(Arc::new(TinyTrace {
        words: vec![0],
        digits: vec![Some(0)],
    }))
    .unwrap();
    let shapes = vec![tiny_shape(WordSlot::Zero, 6, vec![])];
    assert!(matches!(
        RoutersCycleCore::new(&trace, &shapes, &[], &[ZERO; 6], vec![vec![ZERO]]),
        Err(RouterError::Round(RoundError::EmptyPoint))
    ));
}

#[test]
fn cycle_members_reject_disagreeing_round_and_final_challenges_without_consuming_handles() {
    let source = Arc::new(TinyTrace {
        words: vec![1, 0, 1, 1],
        digits: vec![Some(0); 4],
    });
    let trace = ValidatedTrace::new(source.clone()).unwrap();
    let shapes = vec![tiny_shape(WordSlot::Trace(0), 6, vec![]); 2];
    let r_cycle = [F128::from_raw(181), F128::from_raw(193)];
    let x = [ZERO; 6];
    let challenges = [F128::from_raw(211), F128::from_raw(227)];
    let definition = Definition::new(source.as_ref(), &shapes, &r_cycle, &x);
    let core =
        RoutersCycleCore::new(&trace, &shapes, &r_cycle, &x, definition.sources.clone()).unwrap();
    let mut members = core.members();
    let mut claims = definition.claims(&shapes, &x);
    assert!(members[0].finish_rounds(challenges[0]).is_err());
    let expected = definition.cycle_messages(&[]);
    let first = members[0].prove_round(None, 0, claims[0]).unwrap();
    assert_eq!(first.coefficients(), expected[0].coefficients());
    assert!(matches!(
        members[1].prove_round(Some(challenges[0]), 0, claims[1]),
        Err(SumcheckError::MissingEvaluationSource { .. })
    ));
    let second = members[1].prove_round(None, 0, claims[1]).unwrap();
    assert_eq!(second.coefficients(), expected[1].coefficients());
    claims[0] = first.evaluate(challenges[0]);
    claims[1] = second.evaluate(challenges[0]);
    let expected = definition.cycle_messages(&challenges[..1]);
    let first = members[0]
        .prove_round(Some(challenges[0]), 1, claims[0])
        .unwrap();
    assert_eq!(first.coefficients(), expected[0].coefficients());
    assert!(matches!(
        members[1].prove_round(Some(challenges[1]), 1, claims[1]),
        Err(SumcheckError::MissingEvaluationSource { .. })
    ));
    assert!(matches!(
        members[1].prove_round(None, 1, claims[1]),
        Err(SumcheckError::MissingEvaluationSource { .. })
    ));
    let second = members[1]
        .prove_round(Some(challenges[0]), 1, claims[1])
        .unwrap();
    assert_eq!(second.coefficients(), expected[1].coefficients());
    members[0].finish_rounds(challenges[1]).unwrap();
    assert!(members[1].finish_rounds(challenges[0]).is_err());
    members[1].finish_rounds(challenges[1]).unwrap();
    for member in members {
        assert_eq!(
            member.final_values().unwrap().0,
            mle_at(&definition.sources[0], &challenges).unwrap()
        );
    }
}

#[test]
fn nine_cycle_members_and_repeated_word_claims_match_definitions_on_generic_paths() {
    let mut rng = ChaCha20Rng::seed_from_u64(1408);
    let source = Arc::new(TinyTrace {
        words: vec![
            0x2f13, 0x751b, 0x916d, 0x836f, 0xa481, 0xc697, 0xe2b9, 0x13df,
        ],
        digits: vec![Some(0); 8],
    });
    let trace = Arc::new(ValidatedTrace::new(source.clone()).unwrap());
    let shapes = vec![tiny_shape(WordSlot::Trace(0), 6, vec![]); 9];
    let r_cycle = point(3, &mut rng);
    let x = point(6, &mut rng);
    let definition = Definition::new(source.as_ref(), &shapes, &r_cycle, &x);
    for reverse in [false, true] {
        prove_cycle_fixture(&trace, &shapes, &r_cycle, &x, &definition, reverse);
    }
    let lifted = source_lift(&trace, &shapes, &x).unwrap();
    let plan = ScatterPlan::new(trace.clone()).unwrap();
    let r_prime = point(3, &mut rng);
    let output = claims_pass(&trace, &lifted.lifts, &[0; 9], &plan, &r_prime).unwrap();
    let bit_weights: Vec<_> = (0..64).map(|bit| equality(&x, bit)).collect();
    let expected: F128 = (0..source.cycles())
        .map(|cycle| {
            equality(&r_prime, cycle) * word_extension(source.trace_word(0, cycle), &bit_weights)
        })
        .sum();
    assert_eq!(output.trace_words, vec![expected; 9]);
    assert!(output.bytecode_words.is_empty());
    assert_eq!(
        output.row_weights,
        vec![(0..source.cycles())
            .map(|cycle| equality(&r_prime, cycle))
            .sum::<F128>()]
    );
}

#[test]
fn singular_cycle_batch_endpoints_match_definitions_for_all_factor_counts() {
    let mut rng = ChaCha20Rng::seed_from_u64(1409);
    let source = Arc::new(SyntheticTrace::new(SynthProfile::AllRows, 8, 64, 1409).unwrap());
    let trace = ValidatedTrace::new(source.clone()).unwrap();
    let shapes = synthetic_router_shapes().unwrap();
    let x = point(17, &mut rng);
    let mut r_cycle = point(8, &mut rng);
    r_cycle[0] = ZERO;
    r_cycle[1] = ONE;
    let definition = Definition::new(source.as_ref(), &shapes, &r_cycle, &x);
    for reverse in [false, true] {
        prove_cycle_fixture(&trace, &shapes, &r_cycle, &x, &definition, reverse);
    }
}

#[test]
fn digit_bit_and_explicit_zero_bank_entries_match_definitions() {
    let mut rng = ChaCha20Rng::seed_from_u64(1410);
    let source = Arc::new(SyntheticTrace::new(SynthProfile::AllRows, 8, 64, 1410).unwrap());
    let trace = ValidatedTrace::new(source.clone()).unwrap();
    let shapes = vec![shape(RouterShapeRequest {
        slots: 12,
        bank: vec![
            WordSlot::Trace(0),
            WordSlot::Bits(vec![
                BitEntry::DigitBit { column: 5, bit: 3 },
                BitEntry::Zero,
                BitEntry::One,
            ]),
            WordSlot::Bytecode(0),
            WordSlot::Zero,
        ],
        factors: vec![SelectorFactor {
            column: 5,
            slots: (6..10).collect(),
        }],
        word_slots: vec![10, 11],
        log_outputs: 0,
        route: vec![],
    })];
    let r_cycle = point(8, &mut rng);
    let x = point(12, &mut rng);
    let definition = Definition::new(source.as_ref(), &shapes, &r_cycle, &x);
    for reverse in [false, true] {
        prove_cycle_fixture(&trace, &shapes, &r_cycle, &x, &definition, reverse);
    }
}

#[test]
fn nine_distinct_factor_columns_match_definitions_on_uncached_gathers() {
    let mut rng = ChaCha20Rng::seed_from_u64(1411);
    let source = Arc::new(SyntheticTrace::new(SynthProfile::AllRows, 8, 64, 1411).unwrap());
    let trace = ValidatedTrace::new(source.clone()).unwrap();
    let shapes: Vec<_> = (0..9)
        .map(|column| {
            shape(RouterShapeRequest {
                slots: 10,
                bank: vec![WordSlot::Trace(column % 4)],
                factors: vec![SelectorFactor {
                    column,
                    slots: (6..10).collect(),
                }],
                word_slots: vec![],
                log_outputs: 0,
                route: vec![],
            })
        })
        .collect();
    let r_cycle = point(8, &mut rng);
    let x = point(10, &mut rng);
    let definition = Definition::new(source.as_ref(), &shapes, &r_cycle, &x);
    for reverse in [false, true] {
        prove_cycle_fixture(&trace, &shapes, &r_cycle, &x, &definition, reverse);
    }
}

#[test]
fn nine_bytecode_shapes_match_definitions_across_row_tile_batches() {
    let mut rng = ChaCha20Rng::seed_from_u64(1412);
    let source = Arc::new(SyntheticTrace::new(SynthProfile::AllRows, 8, 64, 1412).unwrap());
    let trace = ValidatedTrace::new(source.clone()).unwrap();
    let shapes: Vec<_> = (0..9)
        .map(|index| {
            shape(RouterShapeRequest {
                slots: 14,
                bank: vec![
                    WordSlot::Bytecode(index % 4),
                    WordSlot::Bytecode((index + 1) % 4),
                    WordSlot::Bits(vec![BitEntry::One]),
                    WordSlot::Zero,
                ],
                factors: vec![SelectorFactor {
                    column: 12,
                    slots: (6..12).collect(),
                }],
                word_slots: vec![12, 13],
                log_outputs: 0,
                route: vec![],
            })
        })
        .collect();
    let r_cycle = point(8, &mut rng);
    let x = point(14, &mut rng);
    let definition = Definition::new(source.as_ref(), &shapes, &r_cycle, &x);
    for reverse in [false, true] {
        prove_cycle_fixture(&trace, &shapes, &r_cycle, &x, &definition, reverse);
    }
}

#[test]
fn singular_cycle_endpoint_with_zero_source_member_matches_definitions() {
    let mut rng = ChaCha20Rng::seed_from_u64(1413);
    let source = Arc::new(TinyTrace {
        words: vec![
            0x2f13, 0x751b, 0x916d, 0x836f, 0xa481, 0xc697, 0xe2b9, 0x13df,
        ],
        digits: vec![Some(0); 8],
    });
    let trace = ValidatedTrace::new(source.clone()).unwrap();
    let shapes = vec![
        tiny_shape(WordSlot::Zero, 6, vec![]),
        tiny_shape(WordSlot::Trace(0), 6, vec![]),
    ];
    let r_cycle = vec![ZERO, ONE, F128::random(&mut rng)];
    let x = point(6, &mut rng);
    let definition = Definition::new(source.as_ref(), &shapes, &r_cycle, &x);
    assert!(definition.sources[0].iter().all(|&value| value == ZERO));
    assert!(definition.sources[1].iter().any(|&value| value != ZERO));
    for reverse in [false, true] {
        prove_cycle_fixture(&trace, &shapes, &r_cycle, &x, &definition, reverse);
    }
}

#[test]
fn cycle_subset_members_match_definitions_with_dropped_and_undriven_handles() {
    let mut rng = ChaCha20Rng::seed_from_u64(1414);
    let source = Arc::new(SyntheticTrace::new(SynthProfile::AllRows, 8, 64, 1414).unwrap());
    let trace = ValidatedTrace::new(source.clone()).unwrap();
    let shapes = synthetic_router_shapes().unwrap();
    let r_cycle = point(8, &mut rng);
    let x = point(17, &mut rng);
    let challenges = point(8, &mut rng);
    let definition = Definition::new(source.as_ref(), &shapes, &r_cycle, &x);
    let lifted = source_lift(&trace, &shapes, &x).unwrap();
    let core = RoutersCycleCore::new(&trace, &shapes, &r_cycle, &x, lifted.source_tables).unwrap();
    let mut members = core.members().into_iter();
    drop(members.next().unwrap());
    let mut first = members.next().unwrap();
    drop(members.next().unwrap());
    let mut third = members.next().unwrap();
    let undriven = members.next().unwrap();
    let mut claims = definition.claims(&shapes, &x);
    for (round, &challenge) in challenges.iter().enumerate() {
        let expected = definition.cycle_messages(&challenges[..round]);
        for (member, index) in [(&mut third, 3), (&mut first, 1)] {
            let message = member
                .prove_round(
                    round.checked_sub(1).map(|previous| challenges[previous]),
                    round,
                    claims[index],
                )
                .unwrap();
            assert_eq!(message.coefficients(), expected[index].coefficients());
            claims[index] = message.evaluate(challenge);
        }
    }
    third.finish_rounds(challenges[7]).unwrap();
    first.finish_rounds(challenges[7]).unwrap();
    for (member, index) in [(&third, 3), (&first, 1)] {
        let leaves = &definition.cycle_leaves[index];
        let expected = (
            mle_at(&leaves[1], &challenges).unwrap(),
            leaves[2..]
                .iter()
                .map(|table| mle_at(table, &challenges).unwrap())
                .collect(),
        );
        assert_eq!(member.final_values().unwrap(), expected);
    }
    drop(undriven);
}
