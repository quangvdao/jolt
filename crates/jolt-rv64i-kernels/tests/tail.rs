#![cfg(feature = "test-utils")]
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "acceptance fixtures and assertions fail by panicking"
)]

#[path = "../benches/support/allocator.rs"]
mod allocator;

use allocator::{AllocationMeasurement, CountingAllocator, RAYON_WORKER_ALLOWANCE};
use jolt_field::{CanonicalBytes, Field, F128};
use jolt_poly::UnivariatePoly;
use jolt_rv64i_kernels::chunk_product::{
    combined_weight, ChunkProductCore, ChunkWeight, ChunkWeightTerm,
};
use jolt_rv64i_kernels::column_pass::column_pass;
use jolt_rv64i_kernels::oracle::{mle_at, round_polynomial};
use jolt_rv64i_kernels::par::CycleChunks;
use jolt_rv64i_kernels::reduction::{g_pass_digits, ReductionCore, ReductionLeg};
use jolt_rv64i_kernels::source::{CycleSource, PrepareRequest, PresentGroup, ValidatedTrace};
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, BooleanHypercube, ClearProof, ClearSumcheckRecorder,
    MemberFinish, MemberRound, ProveRounds, ProvedBatch, RoundScheduler, SequentialRounds,
    SumcheckClaim, SumcheckError, SumcheckProof, SumcheckRecorder, SumcheckVerifier,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::ThreadPoolBuilder;
use std::mem::size_of;
use std::sync::Arc;

const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);
const LABEL: &[u8] = b"rv64i-tail";
const BATCH: [F128; 3] = [F128::from_raw(71), F128::from_raw(97), F128::from_raw(113)];

fn eq(point: &[F128], vertex: usize) -> F128 {
    point
        .iter()
        .enumerate()
        .map(|(bit, &w)| ONE + w + F128::from_raw(((vertex >> bit) & 1) as u128))
        .product()
}

fn point(length: usize, rng: &mut ChaCha20Rng) -> Vec<F128> {
    (0..length).map(|_| F128::random(rng)).collect()
}

struct Definition {
    leaves: Vec<Vec<F128>>,
    legs: Vec<ReductionLeg>,
    rounds: usize,
}

impl Definition {
    fn sum(&self, member: usize, values: &[F128]) -> F128 {
        match member {
            0 | 1 => values[6 * member..6 * member + 6].iter().copied().product(),
            _ => self
                .legs
                .iter()
                .enumerate()
                .map(|(i, leg)| leg.coefficient * values[12 + leg.table] * values[15 + i])
                .sum(),
        }
    }

    fn claims(&self) -> [F128; 3] {
        std::array::from_fn(|member| {
            (0..1 << self.rounds)
                .map(|j| {
                    let values: Vec<_> = self.leaves.iter().map(|leaf| leaf[j]).collect();
                    self.sum(member, &values)
                })
                .sum()
        })
    }

    fn final_claims(&self, point: &[F128]) -> [F128; 3] {
        let values: Vec<_> = self
            .leaves
            .iter()
            .map(|leaf| mle_at(leaf, point).unwrap())
            .collect();
        std::array::from_fn(|member| self.sum(member, &values))
    }
}

struct Fixture {
    trace: Arc<ValidatedTrace<SyntheticTrace>>,
    groups: [PresentGroup; 2],
    points: [Vec<Vec<F128>>; 2],
    terms: [Vec<ChunkWeightTerm>; 2],
    weights: Vec<Vec<F128>>,
    definition: Definition,
}

impl Fixture {
    fn new(log_t: usize) -> Self {
        let source =
            Arc::new(SyntheticTrace::new(SynthProfile::Local, log_t, 256, 0x7a11).unwrap());
        let (trace, groups) = ValidatedTrace::prepare(
            Arc::clone(&source),
            PrepareRequest {
                present: vec![(0..5).collect(), (5..10).collect()],
                optional: vec![],
            },
        )
        .unwrap();
        let trace = Arc::new(trace);
        let groups = groups.present.try_into().unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(0x7a11_5eed);
        let points =
            std::array::from_fn(|_| (0..5).map(|_| point(4, &mut rng)).collect::<Vec<_>>());
        let terms = std::array::from_fn(|member| {
            (0..if member == 0 { 5 } else { 2 })
                .map(|i| {
                    let coefficient = F128::random(&mut rng);
                    let point = if member == 0 && i == 0 {
                        (0..log_t)
                            .map(|bit| F128::from_raw((bit & 1) as u128))
                            .collect()
                    } else {
                        point(log_t, &mut rng)
                    };
                    if member == 0 && i == 4 {
                        ChunkWeightTerm::Next { coefficient, point }
                    } else {
                        ChunkWeightTerm::Eq { coefficient, point }
                    }
                })
                .collect::<Vec<_>>()
        });
        let weights: Vec<Vec<_>> = (0..3)
            .map(|support| {
                (0..256)
                    .map(|y| {
                        let active = match support {
                            0 => (64..=228).contains(&y),
                            1 => y < 64 || (139..=230).contains(&y),
                            _ => y < 64,
                        };
                        if active {
                            F128::random(&mut rng)
                        } else {
                            ZERO
                        }
                    })
                    .collect()
            })
            .collect();
        let mut leaves: Vec<Vec<F128>> = Vec::new();
        for member in 0..2 {
            leaves.push(
                (0..source.cycles())
                    .map(|j| {
                        terms[member]
                            .iter()
                            .map(|term| match term {
                                ChunkWeightTerm::Eq { coefficient, point } => {
                                    *coefficient * eq(point, j)
                                }
                                ChunkWeightTerm::Next { coefficient, point } => j
                                    .checked_sub(1)
                                    .map_or(ZERO, |j| *coefficient * eq(point, j)),
                            })
                            .sum()
                    })
                    .collect(),
            );
            for (c, a) in points[member].iter().enumerate() {
                leaves.push(
                    (0..source.cycles())
                        .map(|j| eq(a, source.digit(5 * member + c, j).unwrap()))
                        .collect(),
                );
            }
        }
        for weight in &weights {
            leaves.push(
                source
                    .rows()
                    .iter()
                    .map(|row| {
                        weight
                            .iter()
                            .enumerate()
                            .filter(|(y, _)| row[y / 64] & (1 << (y % 64)) != 0)
                            .map(|(_, &w)| w)
                            .sum()
                    })
                    .collect(),
            );
        }
        let legs: Vec<_> = (0..3)
            .map(|table| {
                let point = point(log_t, &mut rng);
                let claim = leaves[12 + table]
                    .iter()
                    .enumerate()
                    .map(|(j, &g)| eq(&point, j) * g)
                    .sum();
                ReductionLeg {
                    table,
                    point,
                    coefficient: F128::random(&mut rng),
                    claim,
                }
            })
            .collect();
        for leg in &legs {
            leaves.push((0..source.cycles()).map(|j| eq(&leg.point, j)).collect());
        }
        Self {
            trace,
            groups,
            points,
            terms,
            weights,
            definition: Definition {
                leaves,
                legs,
                rounds: log_t,
            },
        }
    }

    fn cores(&self) -> (ChunkProductCore, ChunkProductCore, ReductionCore) {
        let tables =
            g_pass_digits(&self.trace, &SyntheticTrace::column_map(), &self.weights).unwrap();
        assert_eq!(tables, self.definition.leaves[12..15]);
        let weights = self
            .terms
            .each_ref()
            .map(|terms| combined_weight(self.definition.rounds, terms).unwrap());
        assert_eq!(weights[0], self.definition.leaves[0]);
        assert_eq!(weights[1], self.definition.leaves[6]);
        let [first, second] = weights;
        let chunk = |member: usize, weight| {
            ChunkProductCore::new(
                self.groups[member].clone(),
                self.points[member].clone(),
                ChunkWeight::Dense(weight),
            )
            .unwrap()
        };
        (
            chunk(0, first),
            chunk(1, second),
            ReductionCore::new(tables, self.definition.legs.clone()).unwrap(),
        )
    }

    fn prelude(&self) -> BatchPrelude<F128> {
        BatchPrelude::try_new(
            self.definition
                .claims()
                .into_iter()
                .zip(BATCH)
                .map(|(input_claim, coefficient)| BatchMember {
                    input_claim,
                    coefficient,
                    rounds: self.definition.rounds,
                    offset: 0,
                })
                .collect(),
            self.definition.rounds,
            6,
        )
        .unwrap()
    }

    fn assert_columns(
        &self,
        r: &[F128],
        columns: &[F128; 256],
        chunks: &[(F128, Vec<F128>); 2],
        g: &[F128],
        proved: &ProvedBatch<F128>,
    ) {
        for (member, (_, values)) in chunks.iter().enumerate() {
            for (c, a) in self.points[member].iter().enumerate() {
                let zero = eq(a, 0);
                let start = SyntheticTrace::indicator_start(5 * member + c).unwrap();
                let expected = zero
                    + (1..16)
                        .map(|k| (eq(a, k) + zero) * columns[start + k - 1])
                        .sum::<F128>();
                assert_eq!(values[c], expected);
            }
            assert_eq!(
                chunks[member].0,
                mle_at(&self.definition.leaves[6 * member], r).unwrap()
            );
        }
        for (table, weight) in self.weights.iter().enumerate() {
            assert_eq!(
                g[table],
                weight
                    .iter()
                    .zip(columns)
                    .map(|(&l, &c)| l * c)
                    .sum::<F128>()
            );
        }
        let expected = self.definition.final_claims(r);
        assert_eq!(proved.member_claims, expected);
        assert_eq!(
            proved.final_claim,
            expected
                .into_iter()
                .zip(BATCH)
                .map(|(claim, coefficient)| claim * coefficient)
                .sum::<F128>()
        );
    }
}

struct Recorded<C> {
    inner: C,
    messages: Vec<UnivariatePoly<F128>>,
}
impl<C: ProveRounds<F128>> ProveRounds<F128> for Recorded<C> {
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

struct OracleCore<'a> {
    definition: &'a Definition,
    member: usize,
    bound: Vec<F128>,
}
impl ProveRounds<F128> for OracleCore<'_> {
    fn num_rounds(&self) -> usize {
        self.definition.rounds
    }
    fn prove_round(
        &mut self,
        bind: Option<F128>,
        _: usize,
        _: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        if let Some(r) = bind {
            self.bound.push(r);
        }
        let leaves: Vec<_> = self.definition.leaves.iter().map(Vec::as_slice).collect();
        Ok(round_polynomial(
            &leaves,
            &self.bound,
            if self.member < 2 { 6 } else { 2 },
            |v| self.definition.sum(self.member, v),
        )
        .unwrap())
    }
    fn finish_rounds(&mut self, r: F128) -> Result<(), SumcheckError<F128>> {
        self.bound.push(r);
        Ok(())
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
        finishes: &mut [MemberFinish<'_, F128>],
    ) -> Result<(), SumcheckError<F128>> {
        for item in finishes.iter_mut().rev() {
            item.run()?;
        }
        Ok(())
    }
}

fn compressed_round_bytes(proof: &SumcheckProof<F128, ()>) -> Vec<Vec<u8>> {
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
        panic!("clear compressed proof expected")
    };
    proof
        .round_polynomials
        .iter()
        .map(|poly| {
            poly.coeffs_except_linear_term()
                .iter()
                .flat_map(CanonicalBytes::to_bytes_le_vec)
                .collect()
        })
        .collect()
}

fn prove(
    members: &mut [&mut dyn ProveRounds<F128>],
    prelude: &BatchPrelude<F128>,
    scheduler: &mut dyn RoundScheduler<F128>,
) -> (ProvedBatch<F128>, SumcheckProof<F128, ()>) {
    let mut transcript = Blake2bTranscript::new(LABEL);
    let mut recorder = ClearSumcheckRecorder::new();
    let proved = prove_batch(prelude, members, scheduler, &mut recorder, &mut transcript).unwrap();
    let proof = recorder
        .finish(&proved.member_claims, &mut transcript)
        .unwrap()
        .proof;
    (proved, proof)
}

fn verify(
    prelude: &BatchPrelude<F128>,
    proved: &ProvedBatch<F128>,
    proof: &SumcheckProof<F128, ()>,
) {
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
        panic!("clear proof expected")
    };
    let mut transcript = Blake2bTranscript::new(LABEL);
    let reduced = SumcheckVerifier::verify_compressed(
        &SumcheckClaim::new(prelude.max_num_vars, 6, prelude.claimed_sum),
        proof,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut transcript,
    )
    .unwrap();
    assert_eq!(reduced.point.as_slice(), proved.challenges);
    assert_eq!(reduced.value, proved.final_claim);
}

fn acceptance(log_t: usize, threads: &[usize], dense_bits: bool, reverse_orders: &[bool]) {
    let fixture = Fixture::new(log_t);
    let prelude = fixture.prelude();
    let mut oracle: Vec<_> = (0..3)
        .map(|member| Recorded {
            inner: OracleCore {
                definition: &fixture.definition,
                member,
                bound: Vec::new(),
            },
            messages: Vec::new(),
        })
        .collect();
    let mut members: Vec<&mut dyn ProveRounds<F128>> = oracle
        .iter_mut()
        .map(|core| core as &mut dyn ProveRounds<F128>)
        .collect();
    let (expected, expected_proof) = prove(&mut members, &prelude, &mut SequentialRounds);
    let expected_columns: [F128; 256] = std::array::from_fn(|y| {
        fixture
            .trace
            .source()
            .rows()
            .iter()
            .enumerate()
            .filter(|(_, row)| row[y / 64] & (1 << (y % 64)) != 0)
            .map(|(j, _)| eq(&expected.challenges, j))
            .sum()
    });
    for &threads in threads {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for &reverse_order in reverse_orders {
            pool.install(|| {
                let (a, b, g) = fixture.cores();
                let mut a = Recorded {
                    inner: a,
                    messages: Vec::new(),
                };
                let mut b = Recorded {
                    inner: b,
                    messages: Vec::new(),
                };
                let mut g = Recorded {
                    inner: g,
                    messages: Vec::new(),
                };
                let mut sequential = SequentialRounds;
                let mut reverse = ReverseRounds;
                let scheduler: &mut dyn RoundScheduler<F128> = if reverse_order {
                    &mut reverse
                } else {
                    &mut sequential
                };
                let (proved, proof) = prove(&mut [&mut a, &mut b, &mut g], &prelude, scheduler);
                assert_eq!(a.messages, oracle[0].messages);
                assert_eq!(b.messages, oracle[1].messages);
                assert_eq!(g.messages, oracle[2].messages);
                assert_eq!(proved, expected);
                assert_eq!(proof, expected_proof);
                assert_eq!(
                    compressed_round_bytes(&proof),
                    compressed_round_bytes(&expected_proof)
                );
                let columns =
                    column_pass(fixture.trace.source().rows(), &proved.challenges).unwrap();
                assert_eq!(columns, expected_columns);
                let chunks = [
                    a.inner.final_values().unwrap(),
                    b.inner.final_values().unwrap(),
                ];
                fixture.assert_columns(
                    &proved.challenges,
                    &columns,
                    &chunks,
                    g.inner.final_values().unwrap(),
                    &proved,
                );
                verify(&prelude, &proved, &proof);
                if dense_bits {
                    let mut rng = ChaCha20Rng::seed_from_u64(0x7a11_b175);
                    let rho = point(8, &mut rng);
                    let dense: Vec<_> = fixture
                        .trace
                        .source()
                        .rows()
                        .iter()
                        .flat_map(|row| {
                            (0..256).map(move |y| {
                                F128::from_raw(u128::from((row[y / 64] >> (y % 64)) & 1))
                            })
                        })
                        .collect();
                    let opening: Vec<_> = rho.iter().chain(&proved.challenges).copied().collect();
                    assert_eq!(
                        columns
                            .iter()
                            .enumerate()
                            .map(|(y, &c)| eq(&rho, y) * c)
                            .sum::<F128>(),
                        mle_at(&dense, &opening).unwrap()
                    );
                }
            });
        }
    }
}

#[test]
fn tail_end_to_end_matches_committed_bits_and_verifies() {
    acceptance(8, &[1], true, &[false]);
}

#[test]
fn tail_reverse_rounds_and_finishes_match_one_dense_oracle() {
    acceptance(8, &[1], true, &[false, true]);
}

#[test]
fn next_weight_selects_the_successor_at_both_ends_and_inside() {
    let values = [11, 13, 17, 19, 23, 29, 31, 37].map(F128::from_raw);
    // Goal "Chunk products": a Boolean point selects f[p + 1], with no wrap.
    for (point, selected) in [
        ([ZERO, ZERO, ZERO], F128::from_raw(13)),
        ([ONE, ONE, ZERO], F128::from_raw(23)),
        ([ONE, ONE, ONE], ZERO),
    ] {
        let weight = combined_weight(
            3,
            &[ChunkWeightTerm::Next {
                coefficient: ONE,
                point: point.to_vec(),
            }],
        )
        .unwrap();
        assert_eq!(
            weight.iter().zip(values).map(|(&w, f)| w * f).sum::<F128>(),
            selected
        );
        assert_eq!(weight[0], ZERO);
    }
}

#[test]
fn tail_multichunk_rounds_and_passes_match_one_oracle_on_each_pool() {
    acceptance(13, &[1, 12], false, &[false]);
}

#[test]
fn tail_small_domains_match_definitions() {
    for log_t in [1, 2] {
        acceptance(log_t, &[1, 12], true, &[false]);
    }
}

fn measure_rounds(
    core: &mut dyn ProveRounds<F128>,
    mut claim: F128,
    challenges: &[F128],
    threads: usize,
) {
    let measurement = AllocationMeasurement::begin();
    for (round, &r) in challenges.iter().enumerate() {
        let message = core
            .prove_round(round.checked_sub(1).map(|i| challenges[i]), round, claim)
            .unwrap();
        claim = message.evaluate(r);
    }
    core.finish_rounds(*challenges.last().unwrap()).unwrap();
    let stats = measurement.finish();
    assert!(
        stats.allocs <= 16 * challenges.len() + 64 + threads * RAYON_WORKER_ALLOWANCE.allocs,
        "{} allocations",
        stats.allocs
    );
}

#[test]
fn tail_allocations_and_scratch_are_bounded_and_passes_release_storage() {
    // Keep pool teardown and LazyFoldedRa's spawned drops outside pass baselines.
    let mut pools = Vec::with_capacity(2);
    let mut fixtures = Vec::with_capacity(5);
    for (threads, log_sizes) in [(1, &[8, 14, 17][..]), (12, &[14, 17][..])] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let _ = pool.broadcast(|_| {
            rayon::join(
                || {
                    let _ = rayon::yield_now();
                },
                || (),
            )
        });
        let pool_index = pools.len();
        pools.push(pool);
        let pool = &pools[pool_index];
        for &log_t in log_sizes {
            let fixture = pool.install(|| Fixture::new(log_t));
            let map = SyntheticTrace::column_map();
            let challenges = vec![F128::from_raw(79); log_t];
            let allowance = threads * RAYON_WORKER_ALLOWANCE.bytes;
            let allocs = threads * RAYON_WORKER_ALLOWANCE.allocs;
            assert!(log_t != 17 || CycleChunks::new(log_t, 0).unwrap().ranges().len() >= 32);
            let baseline = CountingAllocator::live_bytes();
            let measurement = AllocationMeasurement::begin();
            let tables =
                pool.install(|| g_pass_digits(&fixture.trace, &map, &fixture.weights).unwrap());
            let stats = measurement.finish();
            let output = tables.capacity() * size_of::<Vec<F128>>()
                + tables
                    .iter()
                    .map(|table| table.capacity() * size_of::<F128>())
                    .sum::<usize>();
            assert!(stats.allocs <= 256 + allocs);
            assert!(
                (output..=output + allowance).contains(&stats.final_bytes),
                "g threads={threads}, log_t={log_t}: {} retained bytes for {output} output bytes",
                stats.final_bytes
            );
            drop(tables);
            assert!(
                (baseline..=baseline + allowance).contains(&CountingAllocator::live_bytes()),
                "g release threads={threads}, log_t={log_t}: baseline={baseline}, live={}",
                CountingAllocator::live_bytes()
            );
            for terms in &fixture.terms {
                let baseline = CountingAllocator::live_bytes();
                let measurement = AllocationMeasurement::begin();
                let weight = pool.install(|| combined_weight(log_t, terms).unwrap());
                let stats = measurement.finish();
                assert!(stats.allocs <= 256 + allocs);
                assert!(
                    (weight.capacity() * size_of::<F128>()
                        ..=weight.capacity() * size_of::<F128>() + allowance)
                        .contains(&stats.final_bytes),
                    "threads={threads}, log_t={log_t}: {} retained bytes for {} output bytes",
                    stats.final_bytes,
                    weight.capacity() * size_of::<F128>()
                );
                drop(weight);
                assert!(
                    (baseline..=baseline + allowance).contains(&CountingAllocator::live_bytes()),
                    "weight release threads={threads}, log_t={log_t}: baseline={baseline}, live={}",
                    CountingAllocator::live_bytes()
                );
            }
            let baseline = CountingAllocator::live_bytes();
            let measurement = AllocationMeasurement::begin();
            let columns =
                pool.install(|| column_pass(fixture.trace.source().rows(), &challenges).unwrap());
            let stats = measurement.finish();
            assert_eq!(columns.len(), 256);
            assert!(stats.allocs <= 256 + allocs);
            assert!(
                stats.peak_bytes <= 256 * 16 + threads * 8192 * 16 + (1 << log_t) * 16 + allowance
            );
            assert!(stats.final_bytes <= allowance);
            assert!(
                (baseline..=baseline + allowance).contains(&CountingAllocator::live_bytes()),
                "column release threads={threads}, log_t={log_t}: baseline={baseline}, live={}",
                CountingAllocator::live_bytes()
            );
            fixtures.push((pool_index, threads, fixture, challenges));
        }
    }
    for (pool_index, threads, fixture, challenges) in fixtures {
        let pool = &pools[pool_index];
        pool.install(|| {
            let (mut a, mut b, mut g) = fixture.cores();
            let claims = fixture.definition.claims();
            measure_rounds(&mut a, claims[0], &challenges, threads);
            measure_rounds(&mut b, claims[1], &challenges, threads);
            measure_rounds(&mut g, claims[2], &challenges, threads);
        });
    }
}

#[test]
fn tail_pass_allocation_counts_do_not_grow_with_chunk_count() {
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let _ = pool.broadcast(|_| {
        rayon::join(
            || {
                let _ = rayon::yield_now();
            },
            || (),
        )
    });
    let sizes = [14, 18];
    let chunks = sizes.map(|log_t| CycleChunks::new(log_t, 0).unwrap().ranges().len());
    assert!(chunks[1] >= 4 * chunks[0]);
    let base = pool.install(|| Fixture::new(8));
    let map = SyntheticTrace::column_map();
    let traces = sizes.map(|log_t| {
        pool.install(|| {
            let source =
                Arc::new(SyntheticTrace::new(SynthProfile::Local, log_t, 256, 0x7a11).unwrap());
            ValidatedTrace::new(source).unwrap()
        })
    });
    let terms = sizes.map(|log_t| {
        let mut terms = base.terms.clone();
        for term in terms.iter_mut().flatten() {
            let point = match term {
                ChunkWeightTerm::Eq { point, .. } | ChunkWeightTerm::Next { point, .. } => point,
            };
            point.resize(log_t, point[0]);
        }
        terms
    });
    let points = sizes.map(|log_t| vec![F128::from_raw(79); log_t]);
    let names = [
        "g_pass_digits",
        "combined_weight_five",
        "combined_weight_two",
        "column_pass",
    ];
    // Fixed map/support/term counts and one scratch loan give size-independent
    // kernel counts. Only Rayon's named per-worker bookkeeping may add calls.
    let fixed_growth = RAYON_WORKER_ALLOWANCE.allocs * pool.current_num_threads();
    for sample in 0..8 {
        let mut counts = [[0; 4]; 2];
        let order = if sample % 2 == 0 { [0, 1] } else { [1, 0] };
        for index in order {
            let measurement = AllocationMeasurement::begin();
            let tables =
                pool.install(|| g_pass_digits(&traces[index], &map, &base.weights).unwrap());
            counts[index][0] = measurement.finish().allocs;
            drop(tables);
            for member in 0..2 {
                let measurement = AllocationMeasurement::begin();
                let weight =
                    pool.install(|| combined_weight(sizes[index], &terms[index][member]).unwrap());
                counts[index][member + 1] = measurement.finish().allocs;
                drop(weight);
            }
            let measurement = AllocationMeasurement::begin();
            let columns = pool
                .install(|| column_pass(traces[index].source().rows(), &points[index]).unwrap());
            counts[index][3] = measurement.finish().allocs;
            let _ = std::hint::black_box(columns);
        }
        for (pass, name) in names.iter().enumerate() {
            assert!(counts[1][pass] <= counts[0][pass] + fixed_growth,
                "{name}: {} to {} allocations for {} to {} chunks; fixed growth limit {fixed_growth}",
                counts[0][pass], counts[1][pass], chunks[0], chunks[1]);
        }
    }
}

#[test]
fn tail_first_multichunk_pair_round_matches_the_oracle_on_each_pool() {
    acceptance(14, &[1, 12], false, &[false]);
}

#[test]
fn tail_zero_coefficients_and_later_zero_scalar_preserve_final_values() {
    let mut fixture = Fixture::new(8);
    let ChunkWeightTerm::Eq {
        coefficient,
        point: term_point,
    } = &mut fixture.terms[1][0]
    else {
        panic!("first RAM weight term is an equality")
    };
    for (j, value) in fixture.definition.leaves[6].iter_mut().enumerate() {
        *value += *coefficient * eq(term_point, j);
    }
    *coefficient = ZERO;
    fixture.definition.legs[1].coefficient = ZERO;
    let mut rng = ChaCha20Rng::seed_from_u64(0x7a11_2e20);
    let mut challenges = point(8, &mut rng);
    challenges[0] = ONE + fixture.definition.legs[0].point[0];
    let leaves: Vec<_> = fixture
        .definition
        .leaves
        .iter()
        .map(Vec::as_slice)
        .collect();
    let expected: [Vec<_>; 3] = std::array::from_fn(|member| {
        (0..8)
            .map(|round| {
                round_polynomial(
                    &leaves,
                    &challenges[..round],
                    if member < 2 { 6 } else { 2 },
                    |v| fixture.definition.sum(member, v),
                )
                .unwrap()
            })
            .collect()
    });
    let claims = fixture.definition.claims();
    let finals: Vec<_> = fixture
        .definition
        .leaves
        .iter()
        .map(|leaf| mle_at(leaf, &challenges).unwrap())
        .collect();
    for threads in [1, 12] {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let (mut a, mut b, mut g) = fixture.cores();
                for (member, core) in [&mut a as &mut dyn ProveRounds<F128>, &mut b, &mut g]
                    .into_iter()
                    .enumerate()
                {
                    let mut claim = claims[member];
                    for (round, message) in expected[member].iter().enumerate() {
                        let actual = core
                            .prove_round(round.checked_sub(1).map(|i| challenges[i]), round, claim)
                            .unwrap();
                        assert_eq!(actual, *message);
                        claim = message.evaluate(challenges[round]);
                    }
                    core.finish_rounds(challenges[7]).unwrap();
                }
                let first = a.final_values().unwrap();
                let second = b.final_values().unwrap();
                assert_eq!(first.0, finals[0]);
                assert_eq!(first.1, finals[1..6]);
                assert_eq!(second.0, finals[6]);
                assert_eq!(second.1, finals[7..12]);
                assert_eq!(g.final_values().unwrap(), &finals[12..15]);
            });
    }
}
