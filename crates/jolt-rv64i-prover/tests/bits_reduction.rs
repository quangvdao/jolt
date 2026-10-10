//! The one-member fixture is not the wire of protocol batch 6b: it exercises
//! reduction, column absorption and opening without the two product members.
#![expect(clippy::unwrap_used, reason = "tests fail on invalid fixtures")]

mod support;
use jolt_crypto::NoCommitment;
use jolt_field::{One, Ring, Zero, F128};
use jolt_kernels::ProofSession;
use jolt_poly::Polynomial;
use jolt_prover::driver::{Proved, StageProver};
use jolt_rv64i_arith::{BitsRow, Layout};
use jolt_rv64i_prover::commitment::{
    transparent::{TransparentBits, TransparentCommitment, TransparentError, TransparentOpening},
    BitsCommitmentProver,
};
use jolt_rv64i_prover::plane::Rv64iWitness;
use jolt_rv64i_prover::stages::fixture::{ReductionOnlyKernels, ReductionOnlySumchecks};
use jolt_rv64i_verifier::claims::bits_reduction::BitsReductionInputClaims;
use jolt_rv64i_verifier::commitment::{BitsCommitmentScheme, BitsGeometry, BitsOpening, BitsWire};
use jolt_rv64i_verifier::points::{eq_index, to_high_to_low};
use jolt_rv64i_verifier::stages::fixture::{
    ReductionOnlyInputClaims, ReductionOnlyInputPoints,
    ReductionOnlySumchecks as VerifierReductionOnlySumchecks,
};
use jolt_rv64i_verifier::stages::stage6b::bits_reduction::BitsReduction;
use jolt_sumcheck::{ClearSumcheckRecorder, SequentialRounds};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

type BinaryTranscript = Blake2bTranscript<F128>;
type FixtureProof = Proved<F128, ReductionOnlySumchecks<F128>, NoCommitment>;

fn bit(row: &BitsRow, y: usize) -> F128 {
    F128::from_u64((row[y / 64] >> (y % 64)) & 1)
}
fn point(rng: &mut StdRng, n: usize) -> Vec<F128> {
    (0..n).map(|_| F128::from_raw(rng.gen())).collect()
}

struct Fixture {
    witness: Rv64iWitness,
    batch: ReductionOnlySumchecks<F128>,
    inputs: ReductionOnlyInputClaims<F128>,
    input_points: ReductionOnlyInputPoints<F128>,
    r1: Vec<F128>,
    r3: Vec<F128>,
    r5: Vec<F128>,
    w: Vec<F128>,
    x: Vec<F128>,
}
impl Fixture {
    fn new() -> Self {
        let (_, _, source) = support::counting_loop();
        let mut rng = StdRng::seed_from_u64(0x0641_0b17);
        let layout = source.layout.clone();
        let valid: Vec<_> = source
            .bytecode
            .rows()
            .iter()
            .enumerate()
            .filter(|(_, r)| r.variant.is_some())
            .map(|(i, _)| i)
            .collect();
        let mut bits = Vec::new();
        for _ in 0..32 {
            let mut row = rng.gen::<BitsRow>();
            layout
                .write_bytecode_index(&mut row, valid[rng.gen_range(0..valid.len())] as u64)
                .unwrap();
            layout
                .write_ram_index(&mut row, rng.gen_range(0..1_u64 << layout.log_K_ram()))
                .unwrap();
            layout.write_pos(&mut row, rng.gen_range(0..64)).unwrap();
            bits.push(row);
        }
        let witness = Rv64iWitness::from_bits(
            layout.clone(),
            Arc::clone(&source.bytecode),
            bits.into(),
            source.initial_ram.clone(),
            source.final_pc,
        )
        .unwrap();
        let r1 = point(&mut rng, 5);
        let r3 = point(&mut rng, 5);
        let r5 = point(&mut rng, 5);
        let w = point(&mut rng, 10);
        let x = point(&mut rng, 17);
        let relation = BitsReduction::new(
            &layout,
            r1.clone(),
            r3.clone(),
            r5.clone(),
            w.clone(),
            x.clone(),
        )
        .unwrap();
        let input_points = ReductionOnlyInputPoints {
            bits_reduction: relation.input_points(),
        };
        let mut values = [F128::zero(); 6];
        for (j, row) in witness.bits.iter().enumerate() {
            let linear = definitions(&layout, &w, &x, row);
            for (u, value) in values.iter_mut().enumerate() {
                *value += eq_index(
                    match u {
                        0 => &r1,
                        5 => &r5,
                        _ => &r3,
                    },
                    j,
                )
                .unwrap()
                    * linear[u];
            }
        }
        let [direct_columns, variant_bits, pos_ra_0, pos_ra_1, should_branch, inc] = values;
        let inputs = ReductionOnlyInputClaims {
            bits_reduction: BitsReductionInputClaims {
                direct_columns,
                variant_bits,
                pos_ra_0,
                pos_ra_1,
                should_branch,
                inc,
            },
        };
        Self {
            witness,
            batch: ReductionOnlySumchecks(VerifierReductionOnlySumchecks {
                bits_reduction: relation,
            }),
            inputs,
            input_points,
            r1,
            r3,
            r5,
            w,
            x,
        }
    }
    fn transcript(&self) -> (BinaryTranscript, TransparentCommitment) {
        let mut transcript = BinaryTranscript::new(b"rv64i-reduction-fixture");
        let (commitment, _) = TransparentBits::commit(
            &(),
            BitsGeometry { log_T: 5 },
            &self.witness.bits,
            &mut transcript,
        )
        .unwrap();
        (transcript, commitment)
    }
    fn prove(
        &self,
    ) -> (
        FixtureProof,
        BinaryTranscript,
        TransparentOpening,
        Vec<F128>,
    ) {
        let (mut transcript, _) = self.transcript();
        let challenges = self.batch.draw_challenges(&mut transcript).unwrap();
        let proof = self
            .batch
            .prove(
                &ReductionOnlyKernels::default(),
                &mut ProofSession::default(),
                &mut SequentialRounds,
                &self.witness,
                &self.inputs,
                &self.input_points,
                &challenges,
                ClearSumcheckRecorder::<F128, NoCommitment>::new(),
                &mut transcript,
            )
            .unwrap();
        let rho = transcript.challenge_vector(8);
        let cycle = &proof.output_points.bits_reduction.columns[0];
        let opening = BitsOpening {
            geometry: BitsGeometry { log_T: 5 },
            column_point: &rho,
            cycle_point: cycle,
            columns: &proof.output_claims.bits_reduction.columns,
        };
        let mut state_transcript = BinaryTranscript::new(b"rv64i-reduction-fixture");
        let (_, state) = TransparentBits::commit(
            &(),
            BitsGeometry { log_T: 5 },
            &self.witness.bits,
            &mut state_transcript,
        )
        .unwrap();
        let opening_proof = TransparentBits::open(&(), state, &opening, &mut transcript).unwrap();
        (proof, transcript, opening_proof, rho)
    }
}

/// Direct per-cycle evaluations of the six functionals of §8.17.
fn definitions(layout: &Layout, w: &[F128], x: &[F128], row: &BitsRow) -> [F128; 6] {
    let mut values = [F128::zero(); 6];
    for y in 64..=layout.keys_differ() {
        values[0] += eq_index(w, 768 + y).unwrap() * bit(row, y);
    }
    for y in 0..64 {
        let v = eq_index(&x[..6], y).unwrap() * bit(row, y);
        values[1] += eq_index(&x[6..10], 8).unwrap() * v;
        values[5] += v;
    }
    let g = usize::from(layout.ram_ra()[0].start());
    for y in g..layout.used_columns() {
        values[1] += eq_index(&x[..10], 576 + y - g).unwrap() * bit(row, y);
    }
    for (d, ch) in layout.pos_ra().into_iter().enumerate() {
        let p = &x[6 + 3 * d..9 + 3 * d];
        let zero = eq_index(p, 0).unwrap();
        values[2 + d] = zero;
        for k in 1..=ch.indicators() {
            values[2 + d] +=
                (eq_index(p, k).unwrap() + zero) * bit(row, usize::from(ch.start()) + k - 1);
        }
    }
    values[4] = bit(row, layout.should_branch());
    values
}

#[test]
fn fixture_reduction_opens_seeded_bits_and_terminal_values() {
    let fixture = Fixture::new();
    let (proof, prover_transcript, opening_proof, rho) = fixture.prove();
    let (mut transcript, commitment) = fixture.transcript();
    let mut commit_transcript = BinaryTranscript::new(b"rv64i-reduction-fixture");
    let state = TransparentBits::verify_commit(
        &(),
        BitsGeometry { log_T: 5 },
        &commitment,
        &mut commit_transcript,
    )
    .unwrap();
    let challenges = fixture.batch.draw_challenges(&mut transcript).unwrap();
    let points = fixture
        .batch
        .verify_clear(
            &fixture.inputs,
            &fixture.input_points,
            &challenges,
            &proof.output_claims,
            &proof.recorded.proof,
            &mut transcript,
            0,
        )
        .unwrap();
    fixture
        .batch
        .append_output_claims(&mut transcript, &proof.output_claims);
    let verifier_rho = transcript.challenge_vector(8);
    assert_eq!(rho, verifier_rho);
    let cycle = &points.bits_reduction.columns[0];
    let columns = &proof.output_claims.bits_reduction.columns;
    let opening = BitsOpening {
        geometry: BitsGeometry { log_T: 5 },
        column_point: &rho,
        cycle_point: cycle,
        columns,
    };
    TransparentBits::verify_opening(&(), state, &opening, &opening_proof, &mut transcript).unwrap();
    assert_eq!(transcript.state(), prover_transcript.state());
    for (y, value) in columns.iter().enumerate() {
        let table = Polynomial::new(fixture.witness.bits.iter().map(|r| bit(r, y)).collect());
        assert_eq!(*value, table.evaluate(&to_high_to_low(cycle)));
    }
    let table = Polynomial::new(
        fixture
            .witness
            .bits
            .iter()
            .flat_map(|r| (0..256).map(move |y| bit(r, y)))
            .collect(),
    );
    let full: Vec<_> = rho.iter().chain(cycle).copied().collect();
    assert_eq!(opening.value(), table.evaluate(&to_high_to_low(&full)));
    let coefficient = &challenges.bits_reduction;
    let coefficients = [
        coefficient.direct_columns,
        coefficient.variant_bits,
        coefficient.pos_ra_0,
        coefficient.pos_ra_1,
        coefficient.should_branch,
        coefficient.inc,
    ];
    let mut direct_terminal = F128::zero();
    for y in 0..256 {
        let weights = Polynomial::new(
            (0..32)
                .map(|j| {
                    let mut unit = [0; 4];
                    unit[y / 64] |= 1_u64 << (y % 64);
                    let mut zero = [F128::zero(); 6];
                    let values =
                        definitions(&fixture.witness.layout, &fixture.w, &fixture.x, &unit);
                    let constants =
                        definitions(&fixture.witness.layout, &fixture.w, &fixture.x, &[0; 4]);
                    for u in 0..6 {
                        zero[u] = coefficients[u]
                            * (values[u] + constants[u])
                            * eq_index(
                                match u {
                                    0 => &fixture.r1,
                                    5 => &fixture.r5,
                                    _ => &fixture.r3,
                                },
                                j,
                            )
                            .unwrap();
                    }
                    zero.into_iter().sum()
                })
                .collect(),
        );
        direct_terminal += weights.evaluate(&to_high_to_low(cycle)) * columns[y];
    }
    let derived_terminal: F128 = (0..256)
        .map(|y| {
            fixture
                .batch
                .bits_reduction
                .column_weight(y, cycle, &challenges.bits_reduction)
                .unwrap()
                * columns[y]
        })
        .sum();
    assert_eq!(direct_terminal, derived_terminal);
}

#[test]
fn fixture_rejects_changed_column_and_input_claim() {
    let fixture = Fixture::new();
    let (proof, _, _, _) = fixture.prove();
    for change_column in [true, false] {
        let (mut transcript, _) = fixture.transcript();
        let challenges = fixture.batch.draw_challenges(&mut transcript).unwrap();
        let mut outputs = proof.output_claims.clone();
        let mut inputs = fixture.inputs.clone();
        if change_column {
            outputs.bits_reduction.columns[0] += F128::one();
        } else {
            inputs.bits_reduction.inc += F128::one();
        }
        assert!(fixture
            .batch
            .verify_clear(
                &inputs,
                &fixture.input_points,
                &challenges,
                &outputs,
                &proof.recorded.proof,
                &mut transcript,
                0
            )
            .is_err());
    }
}

#[test]
fn transparent_scheme_rejects_table_column_lengths_and_oversized_geometry() {
    let fixture = Fixture::new();
    let (proof, _, opening_proof, rho) = fixture.prove();
    let cycle = &proof.output_points.bits_reduction.columns[0];
    let columns = &proof.output_claims.bits_reduction.columns;
    let geometry = BitsGeometry { log_T: 5 };
    let (_, commitment) = fixture.transcript();
    let opening = BitsOpening {
        geometry,
        column_point: &rho,
        cycle_point: cycle,
        columns,
    };
    let mut transcript = BinaryTranscript::new(b"rv64i-reduction-fixture");
    let state =
        TransparentBits::verify_commit(&(), geometry, &commitment, &mut transcript).unwrap();
    let mut changed = opening_proof.0.to_vec();
    changed[0][0] ^= 1;
    assert!(TransparentBits::verify_opening(
        &(),
        state,
        &opening,
        &TransparentOpening(changed.into()),
        &mut transcript
    )
    .is_err());
    let state =
        TransparentBits::verify_commit(&(), geometry, &commitment, &mut transcript).unwrap();
    let mut changed = columns.clone();
    changed[0] += F128::one();
    let wrong = BitsOpening {
        columns: &changed,
        ..opening
    };
    assert!(
        TransparentBits::verify_opening(&(), state, &wrong, &opening_proof, &mut transcript)
            .is_err()
    );
    let mut bytes = Vec::new();
    opening_proof.write(&mut bytes);
    for length in 0..bytes.len() {
        assert!(TransparentOpening::read(&bytes[..length], geometry).is_none());
    }
    bytes.push(0);
    assert!(TransparentOpening::read(&bytes, geometry).is_none());
    let oversized = BitsGeometry { log_T: 21 };
    assert!(matches!(
        TransparentBits::commit(&(), oversized, &fixture.witness.bits, &mut transcript),
        Err(TransparentError::Dimension { log_T: 21 })
    ));
    assert!(matches!(
        TransparentBits::verify_commit(&(), oversized, &commitment, &mut transcript),
        Err(TransparentError::Dimension { log_T: 21 })
    ));
    assert!(TransparentOpening::read(&[], oversized).is_none());
}

#[test]
fn fixture_rejects_column_point_drawn_before_column_absorption() {
    let fixture = Fixture::new();
    let (proof, _, _, _) = fixture.prove();
    let (mut early, _) = fixture.transcript();
    let challenges = fixture.batch.draw_challenges(&mut early).unwrap();
    let _ = fixture
        .batch
        .verify_clear(
            &fixture.inputs,
            &fixture.input_points,
            &challenges,
            &proof.output_claims,
            &proof.recorded.proof,
            &mut early,
            0,
        )
        .unwrap();
    let early_rho = early.challenge_vector(8);
    fixture
        .batch
        .append_output_claims(&mut early, &proof.output_claims);
    let (mut correct, commitment) = fixture.transcript();
    let challenges = fixture.batch.draw_challenges(&mut correct).unwrap();
    let _ = fixture
        .batch
        .verify_clear(
            &fixture.inputs,
            &fixture.input_points,
            &challenges,
            &proof.output_claims,
            &proof.recorded.proof,
            &mut correct,
            0,
        )
        .unwrap();
    fixture
        .batch
        .append_output_claims(&mut correct, &proof.output_claims);
    let rho = correct.challenge_vector(8);
    assert_ne!(early_rho, rho);
    let cycle = &proof.output_points.bits_reduction.columns[0];
    let mut columns = proof.output_claims.bits_reduction.columns.clone();
    // This nonzero degree-one difference vanishes at the premature column point.
    let delta = Polynomial::new(
        (0..256)
            .map(|y| F128::from_u64((y & 1) as u64) + early_rho[0])
            .collect(),
    );
    for (y, column) in columns.iter_mut().enumerate() {
        *column += F128::from_u64((y & 1) as u64) + early_rho[0];
    }
    assert_eq!(delta.evaluate(&to_high_to_low(&early_rho)), F128::zero());
    assert_ne!(delta.evaluate(&to_high_to_low(&rho)), F128::zero());
    let malicious = BitsOpening {
        geometry: BitsGeometry { log_T: 5 },
        column_point: &rho,
        cycle_point: cycle,
        columns: &columns,
    };
    let mut twin = BinaryTranscript::new(b"rv64i-reduction-fixture");
    let state =
        TransparentBits::verify_commit(&(), malicious.geometry, &commitment, &mut twin).unwrap();
    assert!(TransparentBits::verify_opening(
        &(),
        state,
        &malicious,
        &TransparentOpening(Arc::clone(&fixture.witness.bits)),
        &mut correct
    )
    .is_err());
}

#[test]
fn witness_reads_follow_the_replayed_pre_state_and_reject_noncanonical_rows() {
    use jolt_rv64i_prover::error::Rv64iProverError;
    use std::collections::BTreeMap;
    let fixture = Fixture::new();
    let witness = &fixture.witness;
    let mut registers = [0_u64; 32];
    let mut ram: BTreeMap<_, _> = witness.initial_ram.iter().copied().collect();
    for (j, bits) in witness.bits.iter().enumerate() {
        let row = &witness.bytecode.rows()[witness.layout.bytecode_index(bits) as usize];
        let next = if j + 1 < witness.bits.len() {
            witness.bytecode.rows()[witness.layout.bytecode_index(&witness.bits[j + 1]) as usize].pc
        } else {
            witness.final_pc
        };
        let base = support::replay::base_words(&witness.layout, row, bits, &registers, &ram, next);
        let store = row.variant.unwrap().is_store();
        assert_eq!(
            witness.words[j].base_words(store, witness.layout.inc(bits)),
            base
        );
        if store {
            let index = witness.layout.ram_index(bits);
            *ram.entry(index).or_default() = base.ram_read_value ^ witness.layout.inc(bits);
        } else {
            registers[usize::from(row.rd)] = base.rd_write_value;
        }
    }
    let (_, _, honest) = support::counting_loop();
    let rebuilt = Rv64iWitness::from_facts(
        honest.layout.clone(),
        Arc::clone(&honest.bytecode),
        &support::counting_loop_facts(),
        honest.initial_ram.clone(),
    )
    .unwrap();
    let initial = support::replay::State {
        pc: honest.bytecode.rows()[honest.layout.bytecode_index(&honest.bits[0]) as usize].pc,
        ram: honest.initial_ram.iter().copied().collect(),
        ..support::replay::State::default()
    };
    let replay = support::replay::replay(
        &honest.layout,
        &honest.bytecode,
        &honest.bits,
        &initial,
        honest.final_pc,
    )
    .unwrap();
    assert_eq!(rebuilt.final_pc, replay.final_state.pc);
    for (j, bits) in rebuilt.bits.iter().enumerate() {
        let pre = if j == 0 {
            &initial
        } else {
            &replay.cycles[j - 1]
        };
        let row = &rebuilt.bytecode.rows()[rebuilt.layout.bytecode_index(bits) as usize];
        let expected = support::replay::base_words(
            &rebuilt.layout,
            row,
            bits,
            &pre.registers,
            &pre.ram,
            replay.cycles[j].pc,
        );
        assert_eq!(
            rebuilt.words[j].base_words(row.variant.unwrap().is_store(), rebuilt.layout.inc(bits)),
            expected
        );
    }
    let build = |bits: Arc<[BitsRow]>, initial_ram| {
        Rv64iWitness::from_bits(
            witness.layout.clone(),
            Arc::clone(&witness.bytecode),
            bits,
            initial_ram,
            witness.final_pc,
        )
    };
    assert!(matches!(
        build(vec![[0; 4]; 3].into(), vec![]),
        Err(Rv64iProverError::RowCount { rows: 3 })
    ));
    for memory in [
        vec![(0, 0)],
        vec![(1, 1), (0, 2)],
        vec![(0, 1), (0, 2)],
        vec![(1_u64 << witness.layout.log_K_ram(), 1)],
    ] {
        assert!(matches!(
            build(Arc::clone(&witness.bits), memory),
            Err(Rv64iProverError::InitialRam { .. })
        ));
    }
    let mut invalid = witness.bits.to_vec();
    witness
        .layout
        .write_bytecode_index(&mut invalid[0], 9)
        .unwrap();
    assert!(matches!(
        build(invalid.into(), witness.initial_ram.clone()),
        Err(Rv64iProverError::InvalidBytecode { cycle: 0, index: 9 })
    ));
    let mut malformed = witness.bits.to_vec();
    let descriptor = witness.layout.pos_ra()[0];
    let start = usize::from(descriptor.start());
    malformed[0][start / 64] |= 1_u64 << (start % 64);
    let next = start + 1;
    malformed[0][next / 64] |= 1_u64 << (next % 64);
    assert!(matches!(
        build(malformed.into(), witness.initial_ram.clone()),
        Err(Rv64iProverError::MultipleIndicators { .. })
    ));
}
