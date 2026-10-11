//! Packed outer lanes and batch-one protocol compatibility.
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "fixture failures fail their enclosing test"
)]
#[expect(
    dead_code,
    reason = "shared machine helpers serve the complete protocol corpus"
)]
mod support;

use jolt_field::F128;
use jolt_kernels::{KernelError, ProofSession};
use jolt_prover::ProverError;
use jolt_rv64i_arith::{RowSystem, Variant};
use jolt_rv64i_kernels::{outer_f2::OuterF2Core, par::CycleChunks, source::LaneSource};
use jolt_rv64i_prover::{
    backend::Rv64iBackend,
    commitment::transparent::TransparentBits,
    error::Rv64iProverError,
    optimized::outer::{SpartanOuterF2Prepare, WitnessLanes},
    plane::{DigitFields, Rv64iWitness},
    prover::{prove_with_transcript, ProverPreprocessing},
    stages::stage1,
};
use jolt_rv64i_verifier::{proof::OuterValues, stages::stage1::Output, statement::CheckedInputs};
use jolt_sumcheck::ClearProof;
use jolt_verifier::VerifierError;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use support::{RecordedTranscript, ReferenceBatches, PROGRAMS};

fn outer_backend() -> Rv64iBackend {
    let mut backend = Rv64iBackend::reference();
    backend.stage1.spartan_outer_f2 = Box::new(SpartanOuterF2Prepare);
    backend
}

fn check_lanes(witness: &Rv64iWitness) {
    let rows = RowSystem::new(&witness.layout);
    let table = rows.f2_tail().unwrap();
    let cycles = witness.cycles();
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let lanes = pool.install(|| WitnessLanes::new(witness)).unwrap();
        assert_eq!(lanes.cycles(), witness.bits.len());
        for cycle in 0..lanes.cycles() {
            let row = cycles.row(cycle).unwrap();
            let expected = rows.lane_rows().map(|family| family.values(&row));
            assert_eq!(lanes.lanes(cycle), expected, "cycle {cycle}");
            let tail = lanes.tail(cycle);
            for (offset, equation) in rows.packed_rows().iter().take(2).enumerate() {
                let values = equation.values(&row);
                for (column, value) in values.into_iter().enumerate() {
                    assert_eq!(
                        u128::from((tail >> (2 * column + offset)) & 1),
                        value.to_raw(),
                        "cycle {cycle}, row {}, column {column}",
                        128 + offset,
                    );
                }
            }
            assert_eq!(tail >> 6, 0, "cycle {cycle}");
            assert_eq!(
                WitnessLanes::cycle(&cycles, &rows, &table, cycle).unwrap(),
                (expected, tail),
                "cycle {cycle}",
            );
        }
        assert_eq!(OuterF2Core::check_rows(&lanes), Ok(()));
        for cycle in [lanes.cycles(), usize::MAX] {
            assert_eq!(lanes.lanes(cycle), [[0; 3]; 2]);
            assert_eq!(lanes.tail(cycle), 0);
            assert!(WitnessLanes::cycle(&cycles, &rows, &table, cycle).is_err());
        }
    }
}

#[test]
fn lanes_match_row_definitions_on_programs_and_separating_jalr() {
    for program in PROGRAMS {
        check_lanes(&support::program_fixture(program, 6).2);
    }
    let (_, _, witness) = support::separating_fixture();
    check_lanes(&witness);
    let rows = RowSystem::new(&witness.layout);
    let table = rows.f2_tail().unwrap();
    let cycle = witness
        .bits
        .iter()
        .position(|bits| {
            witness.bytecode.rows()[witness.layout.bytecode_index(bits) as usize].variant
                == Some(Variant::JALR)
        })
        .unwrap();
    let cycles = witness.cycles();
    let before = WitnessLanes::cycle(&cycles, &rows, &table, cycle).unwrap();
    let mut changed = witness.clone();
    Arc::make_mut(&mut changed.words)[cycle].next_pc ^= 4;
    let changed_cycles = changed.cycles();
    let after = WitnessLanes::cycle(&changed_cycles, &rows, &table, cycle).unwrap();
    assert_ne!(before.0[0], after.0[0], "JALR adder must read NextPC");
    let row = changed_cycles.row(cycle).unwrap();
    assert_eq!(after.0[0], rows.lane_rows()[0].values(&row));
}

#[test]
fn lanes_read_decoded_keys_differ_without_reading_committed_bits() {
    let (_, _, witness) = support::separating_fixture();
    let rows = RowSystem::new(&witness.layout);
    let table = rows.f2_tail().unwrap();
    let cycles = witness.cycles();
    let key_cycle = (0..witness.bits.len())
        .find(|&cycle| cycles.parts(cycle).unwrap().variant.key_kind().is_some())
        .unwrap();
    let mut changed = witness.clone();
    let column = witness.layout.keys_differ();
    Arc::make_mut(&mut changed.bits)[key_cycle][column / 64] ^= 1u64 << (column % 64);
    assert!(Arc::ptr_eq(&witness.decoded, &changed.decoded));
    assert_eq!(witness.decoded, changed.decoded);
    let changed_cycles = changed.cycles();
    assert_ne!(
        rows.packed_rows()[0].values(&cycles.row(key_cycle).unwrap()),
        rows.packed_rows()[0].values(&changed_cycles.row(key_cycle).unwrap()),
        "the committed KeysDiffer mutation must distinguish the full evaluator",
    );
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let original = pool.install(|| WitnessLanes::new(&witness)).unwrap();
        let altered = pool.install(|| WitnessLanes::new(&changed)).unwrap();
        for cycle in 0..witness.bits.len() {
            let expected = (original.lanes(cycle), original.tail(cycle));
            assert_eq!(
                (altered.lanes(cycle), altered.tail(cycle)),
                expected,
                "cycle {cycle}, pool with {threads} threads",
            );
            assert_eq!(
                WitnessLanes::cycle(&changed_cycles, &rows, &table, cycle).unwrap(),
                WitnessLanes::cycle(&cycles, &rows, &table, cycle).unwrap(),
                "public cycle {cycle}",
            );
            assert_eq!(
                WitnessLanes::cycle(&changed_cycles, &rows, &table, cycle).unwrap(),
                expected,
                "public cycle {cycle}, pool with {threads} threads",
            );
        }
    }
}

fn assert_distinct(values: &[F128], name: &str) {
    for (index, value) in values.iter().enumerate() {
        assert!(
            !values[..index].contains(value),
            "{name} values {index} and {} coincide",
            values[..index]
                .iter()
                .position(|other| other == value)
                .unwrap_or(index),
        );
    }
}

fn assert_separates(reference: &ReferenceBatches) {
    let folds = &reference.stage3a.proof.values;
    assert_distinct(
        &[
            folds.variant,
            folds.shift,
            folds.memory,
            folds.compare,
            folds.branch,
        ],
        "batch 3a folds",
    );
    let router = &reference.stage3b.proof.values;
    assert_distinct(
        &[
            router.rs1_value,
            router.rs2_value,
            router.rd_pre_value,
            router.imm,
            router.fall_through_pc,
            router.pc_plus_imm,
            router.pc,
            router.next_pc,
            router.variant_bits,
            router.variant,
            router.shift_kind,
            router.pos_ra_0,
            router.pos_ra_1,
            router.ram_read_value,
            router.access_kind,
            router.key_kind,
            router.branch,
            router.should_branch,
        ],
        "batch 3b outputs",
    );
    let tail = reference
        .stage6b
        .inputs
        .batch
        .expand(&reference.stage6b.proof.values.0)
        .unwrap();
    let chunks: Vec<_> = tail
        .bytecode_read_cycle
        .chunks
        .iter()
        .chain(&tail.ram_ra_product.chunks)
        .copied()
        .collect();
    assert_distinct(&chunks, "batch 6b chunks");
    let input = &reference.stage6b.inputs.claims.bits_reduction;
    let values = [
        input.direct_columns,
        input.variant_bits,
        input.pos_ra_0,
        input.pos_ra_1,
        input.should_branch,
        input.inc,
    ];
    assert!(values.iter().all(|value| value.to_raw() != 0));
    assert_distinct(&values, "BitsReduction inputs");
}

fn assert_outputs(found: &Output, expected: &Output, wire: &OuterValues) {
    macro_rules! claims {
        ($member:ident, $az:ident, $bz:ident, $cz:ident) => {
            for (name, found, expected, transmitted) in [
                (
                    "az",
                    found.claims.$member.az,
                    expected.claims.$member.az,
                    wire.$az,
                ),
                (
                    "bz",
                    found.claims.$member.bz,
                    expected.claims.$member.bz,
                    wire.$bz,
                ),
                (
                    "cz",
                    found.claims.$member.cz,
                    expected.claims.$member.cz,
                    wire.$cz,
                ),
            ] {
                assert_eq!(
                    found,
                    expected,
                    "batch 1 first output {}.{name}",
                    stringify!($member)
                );
                assert_eq!(
                    found,
                    transmitted,
                    "batch 1 wire output {}.{name}",
                    stringify!($member)
                );
            }
            for (name, found, expected) in [
                ("az", &found.points.$member.az, &expected.points.$member.az),
                ("bz", &found.points.$member.bz, &expected.points.$member.bz),
                ("cz", &found.points.$member.cz, &expected.points.$member.cz),
            ] {
                assert_eq!(
                    found,
                    expected,
                    "batch 1 output point {}.{name}",
                    stringify!($member)
                );
            }
        };
    }
    claims!(spartan_outer_f2, az_f2, bz_f2, cz_f2);
    claims!(spartan_outer_f128, az_f128, bz_f128, cz_f128);
}

#[test]
fn batch_one_matches_kept_reference_rounds_and_outputs() {
    let (statement, preprocessing, witness) = support::separating_fixture();
    let reference = support::reference_batches(&statement, &preprocessing, &witness);
    assert_separates(&reference);
    let checked = CheckedInputs::of_statement(
        &preprocessing,
        &statement,
        witness.layout.log_K_ram() as u8,
        witness.final_pc,
    )
    .unwrap();
    let backend = outer_backend();
    let mut transcript = reference.stage1.transcript.fork();
    let (proof, output) = stage1::prove(
        &checked,
        &witness,
        &backend.stage1,
        &mut ProofSession::default(),
        &mut transcript,
    )
    .unwrap();
    let ClearProof::Compressed(found) = proof.rounds.as_clear().unwrap() else {
        panic!("batch 1 is not compressed clear");
    };
    let ClearProof::Compressed(expected) = reference.stage1.proof.rounds.as_clear().unwrap() else {
        panic!("reference batch 1 is not compressed clear");
    };
    assert_eq!(
        found.round_polynomials.len(),
        expected.round_polynomials.len(),
        "batch 1 first differing round {} (round count)",
        found
            .round_polynomials
            .len()
            .min(expected.round_polynomials.len()),
    );
    for (round, (found, expected)) in found
        .round_polynomials
        .iter()
        .zip(&expected.round_polynomials)
        .enumerate()
    {
        assert_eq!(found, expected, "batch 1 first differing round {round}");
    }
    assert_outputs(&output, &reference.stage1.output, &proof.values);
    assert_eq!(proof.values, reference.stage1.proof.values);
}

// The fixed transcript pins this outcome; invariant 9 leaves detection probabilistic
// over the outer weight and round challenges, rather than a production row check.
#[test]
fn violated_adder_row_is_detected_algebraically_in_batch_one() {
    let (statement, verifier, mut witness) = support::counting_loop();
    let cycle = witness
        .bits
        .iter()
        .position(|bits| {
            witness.bytecode.rows()[witness.layout.bytecode_index(bits) as usize].variant
                == Some(Variant::ADDI)
        })
        .unwrap();
    Arc::make_mut(&mut witness.words)[cycle].rs1_value ^= 1;
    let lanes = WitnessLanes::new(&witness).unwrap();
    assert_eq!(OuterF2Core::check_rows(&lanes), Err(cycle));
    let result = prove_with_transcript::<TransparentBits, RecordedTranscript>(
        &ProverPreprocessing {
            verifier,
            scheme: (),
        },
        &statement,
        &witness,
        &outer_backend(),
    );
    let error = result.err().unwrap();
    let Rv64iProverError::Batch { batch, source } = error else {
        panic!("expected batch 1 failure, found {error:?}");
    };
    assert_eq!(batch, "1");
    assert!(matches!(
        *source,
        Rv64iProverError::Prover(ProverError::Verifier(
            VerifierError::StageClaimSumcheckFailed { .. }
        ))
    ));
}

#[test]
fn malformed_lanes_report_first_cycle_on_every_pool_and_invalid_geometry() {
    let (statement, verifier, mut witness) = support::counting_loop_at(14);
    let honest = witness.clone();
    let rows = RowSystem::new(&witness.layout);
    let table = rows.f2_tail().unwrap();
    let geometry = CycleChunks::new(usize::from(statement.log_T), 0).unwrap();
    let cycles: Vec<_> = geometry
        .ranges()
        .skip(1)
        .take(2)
        .map(|range| range.start + 1)
        .collect();
    assert_eq!(cycles.len(), 2);
    let field = DigitFields::new(&witness.layout).bytecode_index();
    let mask = u64::MAX >> (64 - field.bits());
    let invalid_index = witness.bytecode.rows().len() - 1;
    assert_eq!(witness.bytecode.rows()[invalid_index].variant, None);
    for &cycle in &cycles {
        let decoded = &mut Arc::make_mut(&mut witness.decoded)[cycle];
        decoded.digits =
            (decoded.digits & !(mask << field.shift())) | ((invalid_index as u64) << field.shift());
    }
    let view = witness.cycles();
    let expected = WitnessLanes::cycle(&view, &rows, &table, cycles[0])
        .err()
        .unwrap()
        .to_string();
    assert_eq!(
        expected,
        Rv64iProverError::InvalidBytecode {
            cycle: cycles[0],
            index: invalid_index as u64,
        }
        .to_string()
    );
    for threads in [1, 12] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let error = pool.install(|| WitnessLanes::new(&witness)).err().unwrap();
        assert_eq!(error.to_string(), expected, "pool with {threads} threads");
    }
    let error = prove_with_transcript::<TransparentBits, RecordedTranscript>(
        &ProverPreprocessing {
            verifier,
            scheme: (),
        },
        &statement,
        &witness,
        &outer_backend(),
    )
    .err()
    .unwrap();
    let Rv64iProverError::Batch { batch, source } = error else {
        panic!("expected batch 1 geometry failure, found {error:?}");
    };
    assert_eq!(batch, "1");
    let Rv64iProverError::Prover(ProverError::Kernel(KernelError::InvalidGeometry { reason })) =
        *source
    else {
        panic!("expected InvalidGeometry, found {source:?}");
    };
    assert_eq!(reason, expected);
    let mut shortened = honest.clone();
    shortened.decoded = shortened.decoded[..shortened.decoded.len() - 1]
        .to_vec()
        .into();
    assert!(WitnessLanes::new(&shortened).is_err());
    let mut shortened = honest;
    shortened.words = shortened.words[..shortened.words.len() - 1].to_vec().into();
    assert!(WitnessLanes::new(&shortened).is_err());
}
