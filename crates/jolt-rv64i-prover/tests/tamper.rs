//! Whole-protocol rejection of independently changed proof and statement cells.
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "invalid fixtures fail the enclosing test"
)]
#[expect(
    dead_code,
    reason = "shared machine helpers serve the complete protocol corpus"
)]
mod support;
use jolt_crypto::NoCommitment;
use jolt_field::{One, Zero, F128};
use jolt_poly::CompressedPoly;
use jolt_rv64i_prover::{
    backend::Rv64iBackend,
    commitment::transparent::{TransparentBits, TransparentError},
    prover::{prove_with_transcript, ProverPreprocessing},
};
use jolt_rv64i_verifier::{
    error::{ProofDecodeError, Rv64iVerifierError},
    preprocessing::VerifierPreprocessing,
    proof::{BatchProof, Rv64iProof},
    statement::{CheckedInputs, Statement},
    transcript::Rv64iTranscript,
    verifier::{verify, verify_with_transcript},
};
use jolt_sumcheck::{ClearProof, ClearSumcheckProof, CompressedSumcheckProof, SumcheckProof};
use jolt_verifier::VerifierError;
use std::sync::Arc;

type Proof = Rv64iProof<TransparentBits>;
fn fixture() -> (Statement, ProverPreprocessing<TransparentBits>, Proof) {
    let (mut statement, verifier, witness) = support::counting_loop_at(8);
    statement.device.outputs = vec![0; 8];
    let preprocessing = ProverPreprocessing {
        verifier,
        scheme: (),
    };
    let (proof, _) = prove_with_transcript::<TransparentBits, Rv64iTranscript>(
        &preprocessing,
        &statement,
        &witness,
        &Rv64iBackend::reference(),
    )
    .unwrap();
    (statement, preprocessing, proof)
}
fn copy(proof: &Proof) -> Proof {
    Proof::from_bytes(&proof.to_bytes(), 8, 4).unwrap()
}
fn rejection(
    preprocessing: &VerifierPreprocessing<TransparentBits>,
    statement: &Statement,
    proof: &Proof,
) -> Rv64iVerifierError {
    match verify::<TransparentBits>(preprocessing, statement, proof) {
        Err(error) => error,
        Ok(()) => panic!("changed protocol input was accepted"),
    }
}
fn batch_error(error: Rv64iVerifierError, expected: &str) {
    match error {
        Rv64iVerifierError::Batch { batch, source } => {
            assert_eq!(batch, expected);
            assert!(
                matches!(
                    *source,
                    Rv64iVerifierError::Verifier(VerifierError::StageClaimOutputMismatch { .. })
                        | Rv64iVerifierError::Opening(_)
                ),
                "unexpected batch failure: {source}"
            );
        }
        other => panic!("expected batch {expected}, found {other}"),
    }
}
fn rounds<V>(batch: &mut BatchProof<V>) -> &mut CompressedSumcheckProof<F128> {
    match &mut batch.rounds {
        SumcheckProof::Clear(ClearProof::Compressed(rounds)) => rounds,
        _ => panic!("fixture uses compressed clear rounds"),
    }
}
fn shape_error(
    preprocessing: &VerifierPreprocessing<TransparentBits>,
    statement: &Statement,
    proof: &Proof,
) {
    assert!(matches!(
        CheckedInputs::new(preprocessing, statement, proof),
        Err(Rv64iVerifierError::ProofShape(ProofDecodeError::ProofShape))
    ));
}

#[test]
fn each_of_the_299_wire_values_is_bound() {
    let (statement, preprocessing, proof) = fixture();
    for changed in 0..299 {
        let mut altered = copy(&proof);
        let mut cells = vec![
            &mut altered.stage1.values.az_f2,
            &mut altered.stage1.values.bz_f2,
            &mut altered.stage1.values.cz_f2,
            &mut altered.stage1.values.az_f128,
            &mut altered.stage1.values.bz_f128,
            &mut altered.stage1.values.cz_f128,
            &mut altered.stage2.values.witness_routed,
            &mut altered.stage2.values.direct_columns,
            &mut altered.stage3a.values.variant,
            &mut altered.stage3a.values.shift,
            &mut altered.stage3a.values.memory,
            &mut altered.stage3a.values.compare,
            &mut altered.stage3a.values.branch,
            &mut altered.stage3b.values.rs1_value,
            &mut altered.stage3b.values.rs2_value,
            &mut altered.stage3b.values.rd_pre_value,
            &mut altered.stage3b.values.imm,
            &mut altered.stage3b.values.fall_through_pc,
            &mut altered.stage3b.values.pc_plus_imm,
            &mut altered.stage3b.values.pc,
            &mut altered.stage3b.values.next_pc,
            &mut altered.stage3b.values.variant_bits,
            &mut altered.stage3b.values.variant,
            &mut altered.stage3b.values.shift_kind,
            &mut altered.stage3b.values.pos_ra_0,
            &mut altered.stage3b.values.pos_ra_1,
            &mut altered.stage3b.values.ram_read_value,
            &mut altered.stage3b.values.access_kind,
            &mut altered.stage3b.values.key_kind,
            &mut altered.stage3b.values.branch,
            &mut altered.stage3b.values.should_branch,
            &mut altered.stage4.values.rs1_ra,
            &mut altered.stage4.values.rs2_ra,
            &mut altered.stage4.values.rd_wa,
            &mut altered.stage4.values.registers_val,
            &mut altered.stage4.values.ram_ra,
            &mut altered.stage4.values.ram_val,
            &mut altered.stage4.values.ram_val_final,
            &mut altered.stage5.values.rd_wa,
            &mut altered.stage5.values.store,
            &mut altered.stage5.values.inc,
            &mut altered.stage5.values.ram_ra,
            &mut altered.stage6a.values.address_claim,
        ];
        cells.extend(altered.stage6b.values.0.iter_mut());
        assert_eq!(cells.len(), 299);
        *cells[changed] += F128::one();
        drop(cells);
        let expected = match changed {
            0..6 => "1",
            6..8 => "2",
            8..13 => "3a",
            13..31 => "3b",
            31..38 => "4",
            38..42 => "5",
            42 => "6a",
            _ => "6b",
        };
        batch_error(
            rejection(&preprocessing.verifier, &statement, &altered),
            expected,
        );
    }
}

#[test]
fn malformed_in_memory_rounds_and_columns_fail_before_transcript() {
    let (statement, preprocessing, proof) = fixture();
    for count in [255, 257] {
        let mut altered = copy(&proof);
        altered.stage6b.values.0.resize(count, F128::zero());
        shape_error(&preprocessing.verifier, &statement, &altered);
    }
    let mut altered = copy(&proof);
    altered.stage1.rounds =
        SumcheckProof::<F128, NoCommitment>::Clear(ClearProof::Full(ClearSumcheckProof::default()));
    shape_error(&preprocessing.verifier, &statement, &altered);
    for coeffs in [
        vec![],
        vec![F128::one(); 4],
        vec![F128::one(), F128::zero()],
    ] {
        let mut altered = copy(&proof);
        rounds(&mut altered.stage1).round_polynomials[0] = CompressedPoly::new(coeffs);
        shape_error(&preprocessing.verifier, &statement, &altered);
    }
}

#[test]
fn each_batch_binds_its_round_coefficients_and_round_count() {
    let (statement, preprocessing, proof) = fixture();
    macro_rules! check {
        ($field:ident,$name:literal) => {{
            let mut altered = copy(&proof);
            let round = &mut rounds(&mut altered.$field).round_polynomials[0];
            let mut coefficients = round.coeffs_except_linear_term().to_vec();
            coefficients[0] += F128::one();
            *round = CompressedPoly::new(coefficients);
            batch_error(
                rejection(&preprocessing.verifier, &statement, &altered),
                $name,
            );
            let mut altered = copy(&proof);
            let _ = rounds(&mut altered.$field).round_polynomials.pop();
            shape_error(&preprocessing.verifier, &statement, &altered);
        }};
    }
    check!(stage1, "1");
    check!(stage2, "2");
    check!(stage3a, "3a");
    check!(stage3b, "3b");
    check!(stage4, "4");
    check!(stage5, "5");
    check!(stage6a, "6a");
    check!(stage6b, "6b");
}

#[test]
fn proof_header_commitment_opening_and_preprocessing_are_bound() {
    let (statement, preprocessing, proof) = fixture();
    let mut altered = copy(&proof);
    altered.log_K_ram = 6;
    assert!(matches!(
        rejection(&preprocessing.verifier, &statement, &altered),
        Rv64iVerifierError::RamTooLarge
    ));
    let mut altered = copy(&proof);
    altered.final_pc = statement.entry_pc;
    batch_error(
        rejection(&preprocessing.verifier, &statement, &altered),
        "1",
    );
    for pc in [0, statement.entry_pc + 1] {
        let mut altered = copy(&proof);
        altered.final_pc = pc;
        assert!(matches!(
            rejection(&preprocessing.verifier, &statement, &altered),
            Rv64iVerifierError::FinalPc(_)
        ));
    }
    let mut altered = copy(&proof);
    altered.bits_commitment.0[0] ^= 1;
    batch_error(
        rejection(&preprocessing.verifier, &statement, &altered),
        "1",
    );
    let mut altered = copy(&proof);
    Arc::make_mut(&mut altered.opening.0)[0][0] ^= 1;
    match rejection(&preprocessing.verifier, &statement, &altered) {
        Rv64iVerifierError::Batch {
            batch: "6b",
            source,
        } => match *source {
            Rv64iVerifierError::Opening(error) => assert!(matches!(
                error.downcast_ref::<TransparentError>(),
                Some(TransparentError::Digest)
            )),
            other => panic!("unexpected failure: {other}"),
        },
        other => panic!("unexpected failure: {other}"),
    }
    let checked = CheckedInputs::new(&preprocessing.verifier, &statement, &proof).unwrap();
    let mut program: Vec<_> = preprocessing
        .verifier
        .bytecode()
        .rows()
        .iter()
        .filter(|row| row.variant.is_some())
        .map(|row| {
            let offset = row.pc - checked.layout().lowest_address();
            let word = preprocessing
                .verifier
                .image()
                .iter()
                .find(|(index, _)| *index == offset / 8)
                .unwrap()
                .1;
            (row.pc, (word >> (8 * (offset % 8))) as u32)
        })
        .collect();
    program[0].1 = support::asm::addi(1, 0, 7);
    let changed_bytecode = support::harness::bytecode(&program, checked.layout());
    let changed = VerifierPreprocessing::new(
        changed_bytecode,
        preprocessing.verifier.image().to_vec(),
        (),
    )
    .unwrap();
    assert_ne!(changed.digest(), preprocessing.verifier.digest());
    batch_error(rejection(&changed, &statement, &proof), "1");
}

#[test]
fn every_public_statement_field_is_bound() {
    let (statement, preprocessing, proof) = fixture();
    let mut altered = statement.clone();
    altered.entry_pc += 4;
    batch_error(rejection(&preprocessing.verifier, &altered, &proof), "1");
    let mut altered = statement.clone();
    altered.device.inputs[0] ^= 1;
    batch_error(rejection(&preprocessing.verifier, &altered, &proof), "1");
    let mut altered = statement.clone();
    altered.device.outputs[0] ^= 1;
    batch_error(rejection(&preprocessing.verifier, &altered, &proof), "1");
    let mut altered = statement.clone();
    altered.device.panic = true;
    batch_error(rejection(&preprocessing.verifier, &altered, &proof), "1");
    let mut altered = statement.clone();
    altered.log_T = 7;
    shape_error(&preprocessing.verifier, &altered, &proof);
    for field in 0..20 {
        let mut altered = statement.clone();
        let m = &mut altered.device.memory_layout;
        let cells = [
            &mut m.program_size,
            &mut m.max_trusted_advice_size,
            &mut m.trusted_advice_start,
            &mut m.trusted_advice_end,
            &mut m.max_untrusted_advice_size,
            &mut m.untrusted_advice_start,
            &mut m.untrusted_advice_end,
            &mut m.max_input_size,
            &mut m.max_output_size,
            &mut m.input_start,
            &mut m.input_end,
            &mut m.output_start,
            &mut m.output_end,
            &mut m.stack_size,
            &mut m.stack_end,
            &mut m.heap_size,
            &mut m.heap_end,
            &mut m.panic,
            &mut m.termination,
            &mut m.io_end,
        ];
        *cells[field] ^= 1;
        assert!(
            matches!(
                rejection(&preprocessing.verifier, &altered, &proof),
                Rv64iVerifierError::NonCanonicalMemoryLayout | Rv64iVerifierError::MemoryLayout(_)
            ),
            "memory field {field}"
        );
    }
}

use jolt_rv64i_arith::BitsRow;
use jolt_rv64i_prover::commitment::{
    transparent::{TransparentCommitment, TransparentOpening, TransparentState},
    BitsCommitmentProver,
};
use jolt_rv64i_verifier::{
    commitment::{BitsCommitmentScheme, BitsGeometry, BitsOpening},
    points::eq_index,
    stages::{stage1, stage2, stage3a, stage3b, stage4, stage5, stage6a, stage6b},
    transcript::preamble,
};
use jolt_transcript::Transcript;
use support::{Event, RecordedTranscript};

struct PointBoundBits;
impl BitsCommitmentScheme for PointBoundBits {
    type VerifierSetup = ();
    type Commitment = TransparentCommitment;
    type VerifierState = (BitsGeometry, TransparentCommitment);
    type OpeningProof = TransparentOpening;
    type Error = TransparentError;
    fn verify_commit<T: Transcript<Challenge = F128>>(
        (): &(),
        geometry: BitsGeometry,
        commitment: &TransparentCommitment,
        transcript: &mut T,
    ) -> Result<Self::VerifierState, Self::Error> {
        TransparentBits::verify_commit(&(), geometry, commitment, transcript)
    }
    fn verify_opening<T: Transcript<Challenge = F128>>(
        (): &(),
        state: Self::VerifierState,
        opening: &BitsOpening<'_>,
        proof: &TransparentOpening,
        transcript: &mut T,
    ) -> Result<(), Self::Error> {
        TransparentBits::verify_opening(&(), state, opening, proof, transcript)
    }
}
impl BitsCommitmentProver for PointBoundBits {
    type ProverSetup = ();
    type ProverState = TransparentState;
    fn commit<T: Transcript<Challenge = F128>>(
        (): &(),
        geometry: BitsGeometry,
        bits: &Arc<[BitsRow]>,
        transcript: &mut T,
    ) -> Result<(Self::Commitment, Self::ProverState), Self::Error> {
        TransparentBits::commit(&(), geometry, bits, transcript)
    }
    fn open<T: Transcript<Challenge = F128>>(
        (): &(),
        state: TransparentState,
        opening: &BitsOpening<'_>,
        transcript: &mut T,
    ) -> Result<TransparentOpening, Self::Error> {
        TransparentBits::open(&(), state, opening, transcript)
    }
}

#[test]
fn columns_must_be_absorbed_before_drawing_the_opening_point() {
    let (statement, source, witness) = support::counting_loop_at(8);
    let preprocessing = ProverPreprocessing {
        verifier: VerifierPreprocessing::<PointBoundBits>::new(
            Arc::clone(source.shared_bytecode()),
            source.image().to_vec(),
            (),
        )
        .unwrap(),
        scheme: (),
    };
    let (mut proof, recorded) = prove_with_transcript::<PointBoundBits, RecordedTranscript>(
        &preprocessing,
        &statement,
        &witness,
        &Rv64iBackend::reference(),
    )
    .unwrap();
    let checked = CheckedInputs::new(&preprocessing.verifier, &statement, &proof).unwrap();
    let mut transcript: Rv64iTranscript = preamble(&checked);
    let geometry = BitsGeometry { log_T: 8 };
    let state =
        PointBoundBits::verify_commit(&(), geometry, &proof.bits_commitment, &mut transcript)
            .unwrap();
    let s1 = stage1::verify::verify(&checked, &proof.stage1, &mut transcript).unwrap();
    let s2 = stage2::verify::verify(&checked, &proof.stage2, &mut transcript, &s1).unwrap();
    let s3a = stage3a::verify::verify(&checked, &proof.stage3a, &mut transcript, &s2).unwrap();
    let s3b =
        stage3b::verify::verify(&checked, &proof.stage3b, &mut transcript, &s1, &s3a).unwrap();
    let s4 = stage4::verify::verify(&checked, &proof.stage4, &mut transcript, &s3a, &s3b).unwrap();
    let s5 = stage5::verify::verify(&checked, &proof.stage5, &mut transcript, &s3a, &s4).unwrap();
    let s6a = stage6a::verify::verify(
        &checked,
        &proof.stage6a,
        &mut transcript,
        &s3a,
        &s3b,
        &s4,
        &s5,
    )
    .unwrap();
    let inputs =
        stage6b::verify::from_upstream(&checked, &s1, &s2, &s3a, &s3b, &s4, &s5, &s6a).unwrap();
    let challenges = inputs.batch.draw_challenges(&mut transcript).unwrap();
    let honest = inputs.batch.expand(&proof.stage6b.values.0).unwrap();
    let points = inputs
        .batch
        .verify_clear(
            &inputs.claims,
            &inputs.points,
            &challenges,
            &honest,
            &proof.stage6b.rounds,
            &mut transcript,
            6,
        )
        .unwrap();
    let cycle = &points.bits_reduction.columns[0];
    let first_column_event=recorded.events.iter().enumerate().filter(|(_,event)|matches!(event,Event::Append(bytes) if bytes.len()==32 && bytes.starts_with(b"opening_claim"))).nth(43).unwrap().0;
    assert_eq!(
        transcript.state(),
        recorded.inner_at(first_column_event).state()
    );
    let early_rho = transcript.challenge_vector(8);
    let columns = [254, 255];
    for column in columns {
        assert_eq!(
            inputs
                .batch
                .bits_reduction
                .column_weight(column, cycle, &challenges.bits_reduction)
                .unwrap(),
            F128::zero()
        );
        assert!(column >= witness.layout.used_columns());
    }
    let differences = [
        eq_index(&early_rho, 255).unwrap(),
        eq_index(&early_rho, 254).unwrap(),
    ];
    assert!(differences
        .iter()
        .all(|difference| *difference != F128::zero()));
    let original = proof.stage6b.values.0.clone();
    for (column, difference) in columns.into_iter().zip(differences) {
        proof.stage6b.values.0[column] += difference;
    }
    let malicious = inputs.batch.expand(&proof.stage6b.values.0).unwrap();
    assert_eq!(honest.bytecode_read_cycle, malicious.bytecode_read_cycle);
    assert_eq!(honest.ram_ra_product, malicious.ram_ra_product);
    let early = BitsOpening {
        geometry,
        column_point: &early_rho,
        cycle_point: cycle,
        columns: &proof.stage6b.values.0,
    };
    let early_honest = BitsOpening {
        columns: &original,
        ..early
    };
    assert_eq!(early.value(), early_honest.value());
    inputs
        .batch
        .append_output_claims(&mut transcript, &malicious);
    PointBoundBits::verify_opening(&(), state, &early, &proof.opening, &mut transcript).unwrap();
    let mut proper = recorded.inner_at(first_column_event);
    inputs.batch.append_output_claims(&mut proper, &malicious);
    let rho = proper.challenge_vector(8);
    assert_ne!(rho, early_rho);
    let proper_claim = BitsOpening {
        column_point: &rho,
        ..early
    };
    let proper_honest = BitsOpening {
        column_point: &rho,
        columns: &original,
        ..early
    };
    assert_ne!(proper_claim.value(), proper_honest.value());
    match verify_with_transcript::<PointBoundBits, Rv64iTranscript>(
        &preprocessing.verifier,
        &statement,
        &proof,
    ) {
        Err(Rv64iVerifierError::Batch {
            batch: "6b",
            source,
        }) => match *source {
            Rv64iVerifierError::Opening(error) => assert!(matches!(
                error.downcast_ref::<TransparentError>(),
                Some(TransparentError::Evaluation)
            )),
            error => panic!("expected opening evaluation rejection, found {error}"),
        },
        Err(error) => panic!("unexpected rejection: {error}"),
        Ok(_) => panic!("premature opening point was accepted"),
    }
}
