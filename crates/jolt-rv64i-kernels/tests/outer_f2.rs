#![cfg(feature = "test-utils")]
#![expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "test fixtures and assertions may panic on failure"
)]

#[path = "../benches/support/allocator.rs"]
#[expect(
    dead_code,
    reason = "the shared benchmark allocator also exposes byte metrics unused by this count test"
)]
mod allocator;

use allocator::AllocationMeasurement;
use jolt_field::{Field, F128};
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_rv64i_kernels::oracle::{mle_at, round_polynomial};
use jolt_rv64i_kernels::outer_f2::{OuterError, OuterF2Core, OuterF2Options};
use jolt_rv64i_kernels::source::LaneSource;
use jolt_rv64i_kernels::synth::{SynthProfile, SyntheticTrace};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, BooleanHypercube, ClearProof, ClearSumcheckRecorder,
    ProveRounds, ProvedBatch, SequentialRounds, SumcheckClaim, SumcheckError, SumcheckProof,
    SumcheckRecorder, SumcheckVerifier, SUMCHECK_CLAIM_TRANSCRIPT_LABEL,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;

const ZERO: F128 = F128::from_raw(0);
const ONE: F128 = F128::from_raw(1);
const LABEL: &[u8] = b"bitwise-outer-f2";

struct Definition {
    leaves: [Vec<F128>; 4],
}

impl Definition {
    fn new(source: &impl LaneSource, tau: &[F128]) -> Self {
        let mut leaves: [Vec<F128>; 4] = std::array::from_fn(|_| vec![ZERO; 256 * source.cycles()]);
        let row_eq: Vec<_> = (0..256).map(|row| eq_at_index(&tau[..8], row)).collect();
        for cycle in 0..source.cycles() {
            let words = source.lanes(cycle);
            let tail = source.tail(cycle);
            let cycle_eq = eq_at_index(&tau[8..], cycle);
            for (row, &row_weight) in row_eq.iter().enumerate() {
                let index = 256 * cycle + row;
                for (lane, table) in leaves.iter_mut().take(3).enumerate() {
                    let bit = if row < 128 {
                        (words[row / 64][lane] >> (row % 64)) & 1
                    } else if row < 130 {
                        u64::from((tail >> (2 * lane + row - 128)) & 1)
                    } else {
                        0
                    };
                    table[index] = F128::from_raw(u128::from(bit));
                }
                leaves[3][index] = row_weight * cycle_eq;
            }
        }
        Self { leaves }
    }

    fn round(&self, point: &[F128]) -> UnivariatePoly<F128> {
        let leaves: Vec<_> = self.leaves.iter().map(Vec::as_slice).collect();
        round_polynomial(&leaves, point, 3, |v| v[3] * (v[0] * v[1] + v[2])).unwrap()
    }

    fn values(&self, point: &[F128]) -> [F128; 3] {
        std::array::from_fn(|lane| mle_at(&self.leaves[lane], point).unwrap())
    }

    fn final_claim(&self, point: &[F128]) -> F128 {
        let [a, b, c] = self.values(point);
        mle_at(&self.leaves[3], point).unwrap() * (a * b + c)
    }
}

fn eq_at_index(point: &[F128], index: usize) -> F128 {
    point.iter().enumerate().fold(ONE, |value, (bit, &t)| {
        value * (ONE + t + F128::from_raw(((index >> bit) & 1) as u128))
    })
}

struct OracleRounds<'a> {
    definition: &'a Definition,
    point: Vec<F128>,
    messages: Vec<UnivariatePoly<F128>>,
}

impl ProveRounds<F128> for OracleRounds<'_> {
    fn num_rounds(&self) -> usize {
        self.definition.leaves[0].len().ilog2() as usize
    }

    fn prove_round(
        &mut self,
        bind: Option<F128>,
        round: usize,
        claim: F128,
    ) -> Result<UnivariatePoly<F128>, SumcheckError<F128>> {
        if let Some(bind) = bind {
            self.point.push(bind);
        }
        assert_eq!(round, self.point.len());
        let message = self.definition.round(&self.point);
        assert_eq!(message.evaluate(ZERO) + message.evaluate(ONE), claim);
        self.messages.push(message.clone());
        Ok(message)
    }

    fn finish_rounds(&mut self, bind: F128) -> Result<(), SumcheckError<F128>> {
        self.point.push(bind);
        Ok(())
    }
}

struct Batch {
    prelude: BatchPrelude<F128>,
    proved: ProvedBatch<F128>,
    proof: SumcheckProof<F128, ()>,
}

#[derive(Debug)]
enum Rejection {
    Sumcheck(SumcheckError<F128>),
    FinalIdentity,
}

impl Batch {
    fn prove(
        core: &mut dyn ProveRounds<F128>,
        input_claim: F128,
    ) -> Result<Self, SumcheckError<F128>> {
        let mut recorder = ClearSumcheckRecorder::<F128>::new();
        let mut transcript = Blake2bTranscript::<F128>::new(LABEL);
        recorder.absorb_input_claims(&[input_claim], &mut transcript);
        let rounds = core.num_rounds();
        let prelude = BatchPrelude::try_new(
            vec![BatchMember {
                input_claim,
                coefficient: transcript.challenge_scalar(),
                rounds,
                offset: 0,
            }],
            rounds,
            3,
        )?;
        let proved = prove_batch(
            &prelude,
            &mut [core],
            &mut SequentialRounds,
            &mut recorder,
            &mut transcript,
        )?;
        let recorded = recorder.finish(&proved.member_claims, &mut transcript)?;
        Ok(Self {
            prelude,
            proved,
            proof: recorded.proof,
        })
    }

    fn verify(
        &self,
        proof: &SumcheckProof<F128, ()>,
        native: impl FnOnce(&[F128]) -> F128,
    ) -> Result<(), Rejection> {
        let mut transcript = Blake2bTranscript::<F128>::new(LABEL);
        let member = &self.prelude.members[0];
        transcript.append_labeled(SUMCHECK_CLAIM_TRANSCRIPT_LABEL, &member.input_claim);
        assert_eq!(transcript.challenge_scalar(), member.coefficient);
        let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
            panic!("the clear recorder must produce compressed rounds")
        };
        let reduced = SumcheckVerifier::verify_compressed(
            &SumcheckClaim::new(self.prelude.max_num_vars, 3, self.prelude.claimed_sum),
            proof,
            BooleanHypercube,
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            &mut transcript,
        )
        .map_err(Rejection::Sumcheck)?;
        if reduced.value != member.coefficient * native(reduced.point.as_slice()) {
            return Err(Rejection::FinalIdentity);
        }
        Ok(())
    }
}

fn options() -> impl Iterator<Item = OuterF2Options> {
    (2..=6).flat_map(|monomial_rounds| {
        [false, true].into_iter().flat_map(move |nibble_round_2| {
            [false, true]
                .into_iter()
                .map(move |folded_group_weights| OuterF2Options {
                    monomial_rounds,
                    nibble_round_2,
                    folded_group_weights,
                })
        })
    })
}

fn trace(log_t: usize) -> Arc<SyntheticTrace> {
    Arc::new(SyntheticTrace::new(SynthProfile::Local, log_t, 16, 0x5eed).unwrap())
}

fn seeded_point(rounds: usize, seed: u64) -> Vec<F128> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..rounds).map(|_| F128::random(&mut rng)).collect()
}

fn assert_rounds(
    core: &mut impl ProveRounds<F128>,
    point: &[F128],
    expected: &[UnivariatePoly<F128>],
) {
    assert_eq!(core.num_rounds(), expected.len());
    let mut claim = ZERO;
    for (round, (&challenge, expected)) in point.iter().zip(expected).enumerate() {
        let message = core
            .prove_round(round.checked_sub(1).map(|i| point[i]), round, claim)
            .unwrap();
        assert_eq!(
            message.coefficients(),
            expected.coefficients(),
            "round {round}"
        );
        assert_eq!(message.coefficients().len(), 4);
        claim = message.evaluate(challenge);
    }
    core.finish_rounds(*point.last().unwrap()).unwrap();
}

#[test]
fn outer_messages_and_boolean_points_match_summation_for_every_option_and_verify() {
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        for log_t in 3..=8 {
            let source = trace(log_t);
            assert_eq!(OuterF2Core::<SyntheticTrace>::check_rows(&source), Ok(()));
            let rounds = 8 + log_t;
            let point = seeded_point(rounds, 0xcafe + log_t as u64);
            let taus = [
                seeded_point(rounds, 0x7a00 + log_t as u64),
                vec![ZERO; rounds],
                vec![ONE; rounds],
                (0..rounds)
                    .map(|i| F128::from_raw((i & 1) as u128))
                    .collect(),
            ];
            for tau in taus {
                let definition = Definition::new(source.as_ref(), &tau);
                let messages: Vec<_> = (0..rounds)
                    .map(|round| definition.round(&point[..round]))
                    .collect();
                let values = definition.values(&point);
                let mut oracle = OracleRounds {
                    definition: &definition,
                    point: Vec::new(),
                    messages: Vec::new(),
                };
                let expected_batch = Batch::prove(&mut oracle, ZERO).unwrap();
                let batch_values = definition.values(&expected_batch.proved.challenges);
                let batch_claim = definition.final_claim(&expected_batch.proved.challenges);
                for options in options() {
                    let mut core = OuterF2Core::new(Arc::clone(&source), &tau, options).unwrap();
                    assert_rounds(&mut core, &point, &messages);
                    assert_eq!(core.final_values(), values);
                    let mut core = OuterF2Core::new(Arc::clone(&source), &tau, options).unwrap();
                    let batch = Batch::prove(&mut core, ZERO).unwrap();
                    assert_eq!(batch.proved.challenges, oracle.point);
                    assert_eq!(core.final_values(), batch_values);
                    assert_eq!(batch.proved.member_claims, [batch_claim]);
                    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = &batch.proof else {
                        panic!("the clear recorder must produce compressed rounds")
                    };
                    let coefficient = batch.prelude.members[0].coefficient;
                    let mut claim = batch.prelude.claimed_sum;
                    for (round, (message, expected)) in proof
                        .round_polynomials
                        .iter()
                        .zip(&oracle.messages)
                        .enumerate()
                    {
                        let expected = UnivariatePoly::new(
                            expected
                                .coefficients()
                                .iter()
                                .map(|&c| coefficient * c)
                                .collect(),
                        );
                        let mut coefficients = message.decompress(claim).into_coefficients();
                        coefficients.resize(4, ZERO);
                        assert_eq!(coefficients, expected.coefficients());
                        claim = expected.evaluate(batch.proved.challenges[round]);
                    }
                    batch
                        .verify(&batch.proof, |point| {
                            assert_eq!(point, oracle.point);
                            batch_claim
                        })
                        .unwrap();
                }
                for zero_factor_index in [0, 8] {
                    let mut zero_factor_point = point.clone();
                    zero_factor_point[zero_factor_index] = ONE + tau[zero_factor_index];
                    let earlier_scale: F128 = tau[..zero_factor_index]
                        .iter()
                        .zip(&zero_factor_point)
                        .map(|(&t, &r)| ONE + t + r)
                        .product();
                    assert_ne!(earlier_scale, ZERO);
                    let zero_factor_messages: Vec<_> = (0..rounds)
                        .map(|round| definition.round(&zero_factor_point[..round]))
                        .collect();
                    let zero_factor_values = definition.values(&zero_factor_point);
                    for options in options() {
                        let mut core =
                            OuterF2Core::new(Arc::clone(&source), &tau, options).unwrap();
                        assert_rounds(&mut core, &zero_factor_point, &zero_factor_messages);
                        assert_eq!(core.final_values(), zero_factor_values);
                    }
                }
            }
        }
    });
}

struct CorruptLane {
    source: Arc<SyntheticTrace>,
    cycle: usize,
    tail: bool,
}

impl LaneSource for CorruptLane {
    fn cycles(&self) -> usize {
        self.source.cycles()
    }

    fn lanes(&self, cycle: usize) -> [[u64; 3]; 2] {
        let mut lanes = self.source.lanes(cycle);
        if cycle == self.cycle && !self.tail {
            lanes[1][2] ^= 1 << 37;
        }
        lanes
    }

    fn tail(&self, cycle: usize) -> u8 {
        self.source.tail(cycle)
            ^ if cycle == self.cycle && self.tail {
                1 << 5
            } else {
                0
            }
    }
}

#[test]
fn outer_bad_rows_coefficients_and_input_claim_are_rejected() {
    let source = trace(5);
    let tau = seeded_point(13, 0x7a00);
    for tail in [false, true] {
        let bad = Arc::new(CorruptLane {
            source: Arc::clone(&source),
            cycle: 7,
            tail,
        });
        assert_eq!(OuterF2Core::<CorruptLane>::check_rows(&bad), Err(7));
        let definition = Definition::new(bad.as_ref(), &tau);
        let mut core = OuterF2Core::new(bad, &tau, OuterF2Options::default()).unwrap();
        let batch = Batch::prove(&mut core, ZERO).unwrap();
        assert!(matches!(
            batch.verify(&batch.proof, |point| definition.final_claim(point)),
            Err(Rejection::FinalIdentity)
        ));
    }
    let definition = Definition::new(source.as_ref(), &tau);
    let mut core = OuterF2Core::new(Arc::clone(&source), &tau, OuterF2Options::default()).unwrap();
    let batch = Batch::prove(&mut core, ZERO).unwrap();
    let mut altered = batch.proof.clone();
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = &mut altered else {
        panic!("the clear recorder must produce compressed rounds")
    };
    let last = proof.round_polynomials.last_mut().unwrap();
    let mut coefficients = last.coeffs_except_linear_term().to_vec();
    coefficients[0] += ONE;
    *last = CompressedPoly::new(coefficients);
    match batch.verify(&altered, |point| definition.final_claim(point)) {
        Err(Rejection::FinalIdentity) => {}
        Err(Rejection::Sumcheck(error)) => panic!("expected final identity rejection: {error:?}"),
        Ok(()) => panic!("changed coefficient was accepted"),
    }
    let mut core = OuterF2Core::new(source, &tau, OuterF2Options::default()).unwrap();
    match Batch::prove(&mut core, ONE) {
        Err(SumcheckError::RoundCheckFailed { .. }) => {}
        Err(error) => panic!("unexpected prover rejection: {error:?}"),
        Ok(batch) => assert!(batch
            .verify(&batch.proof, |point| definition.final_claim(point))
            .is_err()),
    }
}

struct TwelveCycles;

impl LaneSource for TwelveCycles {
    fn cycles(&self) -> usize {
        12
    }
    fn lanes(&self, _cycle: usize) -> [[u64; 3]; 2] {
        [[0; 3]; 2]
    }
    fn tail(&self, _cycle: usize) -> u8 {
        0
    }
}

#[test]
fn outer_constructor_reports_each_rejected_geometry() {
    let source = trace(3);
    assert!(matches!(
        OuterF2Core::new(Arc::clone(&source), &[ZERO; 10], OuterF2Options::default()),
        Err(OuterError::PointLength {
            expected: 11,
            actual: 10
        })
    ));
    for rounds in [1, 7] {
        assert!(matches!(
            OuterF2Core::new(Arc::clone(&source), &[ZERO; 11], OuterF2Options {
                monomial_rounds: rounds, ..OuterF2Options::default()
            }),
            Err(OuterError::MonomialRounds { rounds: actual }) if actual == rounds
        ));
    }
    assert!(matches!(
        OuterF2Core::new(
            Arc::new(TwelveCycles),
            &[ZERO; 12],
            OuterF2Options::default()
        ),
        Err(OuterError::Cycles { cycles: 12 })
    ));
}

#[test]
fn outer_messages_on_one_and_twelve_threads_match_summation() {
    let source = trace(14);
    let tau = seeded_point(22, 0x7a14);
    let point = seeded_point(22, 0xca14);
    let definition = Definition::new(source.as_ref(), &tau);
    let messages: Vec<_> = (0..22)
        .map(|round| definition.round(&point[..round]))
        .collect();
    let values = definition.values(&point);
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut core =
                OuterF2Core::new(Arc::clone(&source), &tau, OuterF2Options::default()).unwrap();
            assert_rounds(&mut core, &point, &messages);
            assert_eq!(core.final_values(), values);
        });
    }
}

#[test]
fn outer_round_allocation_count_is_bounded() {
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        for log_t in [8, 14] {
            let source = trace(log_t);
            let rounds = 8 + log_t;
            let tau = seeded_point(rounds, 0x7a00 + log_t as u64);
            let point = seeded_point(rounds, 0xca00 + log_t as u64);
            for options in options() {
                let mut core = OuterF2Core::new(Arc::clone(&source), &tau, options).unwrap();
                let measurement = AllocationMeasurement::begin();
                let mut claim = ZERO;
                for (round, &challenge) in point.iter().enumerate() {
                    let message = core
                        .prove_round(round.checked_sub(1).map(|i| point[i]), round, claim)
                        .unwrap();
                    claim = message.evaluate(challenge);
                }
                core.finish_rounds(*point.last().unwrap()).unwrap();
                let stats = measurement.finish();
                assert!(
                    stats.allocs <= 16 * rounds + 64,
                    "log_t={log_t}, options={options:?}: {} allocations",
                    stats.allocs
                );
            }
        }
    });
}
