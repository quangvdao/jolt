//! Complete binary-field RV64I protocol execution on interpreted programs.
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "fixture failures fail their enclosing test"
)]
#![expect(
    non_snake_case,
    reason = "protocol dimensions follow mathematical notation"
)]
#[expect(
    dead_code,
    reason = "shared machine helpers serve the complete protocol corpus"
)]
mod support;
use jolt_prover::ProverError;
use jolt_rv64i_arith::Variant;
use jolt_rv64i_prover::error::Rv64iProverError;
use jolt_rv64i_prover::{
    backend::Rv64iBackend,
    commitment::transparent::TransparentBits,
    prover::{prove_with_transcript, ProverPreprocessing},
};
use jolt_rv64i_verifier::error::Rv64iVerifierError;
use jolt_rv64i_verifier::verifier::verify_with_transcript;
use jolt_rv64i_verifier::{preprocessing::VerifierPreprocessing, whir::WhirBits};
use jolt_sumcheck::SumcheckError;
use jolt_transcript::Transcript;
use jolt_verifier::VerifierError;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use support::{Program, ProvingFailure, ProvingFailureFixture, RecordedTranscript, PROGRAMS};

#[expect(
    clippy::print_stdout,
    reason = "the acceptance corpus records per-program proving and verification timings"
)]
fn run_program(program: Program, log_T: u8) -> HashSet<Variant> {
    let (statement, source, witness) = support::program_fixture(program, log_T);
    let variants = witness
        .bits
        .iter()
        .map(|row| {
            witness.bytecode.rows()[witness.layout.bytecode_index(row) as usize]
                .variant
                .unwrap()
        })
        .collect();
    let valid_rows = witness
        .bytecode
        .rows()
        .iter()
        .filter(|row| row.variant.is_some())
        .count();
    assert_eq!(
        witness.layout.log_K_bytecode(),
        valid_rows.next_power_of_two().trailing_zeros() as usize,
        "{} uses the smallest bytecode cube",
        program.name()
    );
    let last_ram_word = witness.initial_ram.last().unwrap().0 as usize;
    assert_eq!(
        witness.layout.log_K_ram(),
        5_usize.max((last_ram_word + 1).next_power_of_two().trailing_zeros() as usize),
        "{} uses the smallest RAM cube",
        program.name()
    );
    let preprocessing = ProverPreprocessing {
        verifier: VerifierPreprocessing::<WhirBits>::new(
            Arc::clone(source.shared_bytecode()),
            source.image().to_vec(),
            (),
        )
        .unwrap(),
        scheme: (),
    };
    let backend = Rv64iBackend::reference();
    let start = Instant::now();
    let (proof, prover) = prove_with_transcript::<WhirBits, RecordedTranscript>(
        &preprocessing,
        &statement,
        &witness,
        &backend,
    )
    .unwrap();
    let prove_time = start.elapsed();
    let start = Instant::now();
    let verifier = verify_with_transcript::<WhirBits, RecordedTranscript>(
        &preprocessing.verifier,
        &statement,
        &proof,
    )
    .unwrap();
    let verify_time = start.elapsed();
    println!(
        "{} t={log_T} b={} a={} prove={:.6}s verify={:.6}s",
        program.name(),
        witness.layout.log_K_bytecode(),
        witness.layout.log_K_ram(),
        prove_time.as_secs_f64(),
        verify_time.as_secs_f64()
    );
    assert_eq!(prover.batch_states.len(), 8);
    assert_eq!(prover.batch_states, verifier.batch_states);
    assert_eq!(prover.state(), verifier.state());
    variants
}

#[test]
fn five_programs_all_sizes_and_variants() {
    let mut variants = HashSet::new();
    for program in PROGRAMS {
        for log_T in [6, 8, 10] {
            variants.extend(run_program(program, log_T));
        }
    }
    assert_eq!(variants, HashSet::from(Variant::ALL));
}

fn assert_failed_batch(error: Rv64iProverError, expected: &str) {
    match error {
        Rv64iProverError::Batch { batch, source } => {
            assert_eq!(batch, expected);
            assert!(matches!(
                *source,
                Rv64iProverError::Prover(ProverError::Sumcheck(SumcheckError::RoundCheckFailed {
                    round: 0,
                    ..
                }))
            ));
        }
        other => panic!("expected batch {expected} rejection, found {other:?}"),
    }
}
fn assert_statement_binding_rejection(error: Rv64iVerifierError) {
    match error {
        Rv64iVerifierError::Batch { batch, source } => {
            assert_eq!(batch, "1");
            assert!(matches!(
                *source,
                Rv64iVerifierError::Verifier(VerifierError::StageClaimOutputMismatch { .. })
            ));
        }
        other => panic!("expected statement binding rejection in batch 1, found {other:?}"),
    }
}

/// The verifier rejection checks statement binding into the transcript, not the equation.
#[test]
fn wrong_output_is_rejected_by_proving_and_statement_binding() {
    let ProvingFailureFixture {
        statement,
        changed,
        verifier,
        witness,
        expected_batch,
    } = support::proving_failure_fixture(ProvingFailure::WrongOutput);
    let preprocessing = ProverPreprocessing {
        verifier,
        scheme: (),
    };
    let backend = Rv64iBackend::reference();
    let (proof, _) = prove_with_transcript::<TransparentBits, RecordedTranscript>(
        &preprocessing,
        &statement,
        &witness,
        &backend,
    )
    .unwrap();
    let error = prove_with_transcript::<TransparentBits, RecordedTranscript>(
        &preprocessing,
        &changed,
        &witness,
        &backend,
    )
    .err()
    .unwrap();
    assert_failed_batch(error, expected_batch);
    let error = verify_with_transcript::<TransparentBits, RecordedTranscript>(
        &preprocessing.verifier,
        &changed,
        &proof,
    )
    .err()
    .unwrap();
    assert_statement_binding_rejection(error);
}

/// The verifier rejection checks statement binding into the transcript, not the equation.
#[test]
fn missing_termination_is_rejected_by_proving_and_statement_binding() {
    let ProvingFailureFixture {
        statement,
        changed,
        verifier,
        witness,
        expected_batch,
    } = support::proving_failure_fixture(ProvingFailure::MissingTermination);
    assert!(statement.device.panic);
    assert_eq!(witness.final_ram[3], 0);
    assert!(witness.bits.iter().all(|row| {
        let fetched = &witness.bytecode.rows()[witness.layout.bytecode_index(row) as usize];
        !fetched.variant.unwrap().is_store() || witness.layout.ram_index(row) != 3
    }));
    let preprocessing = ProverPreprocessing {
        verifier,
        scheme: (),
    };
    let backend = Rv64iBackend::reference();
    let (proof, _) = prove_with_transcript::<TransparentBits, RecordedTranscript>(
        &preprocessing,
        &statement,
        &witness,
        &backend,
    )
    .unwrap();
    let error = prove_with_transcript::<TransparentBits, RecordedTranscript>(
        &preprocessing,
        &changed,
        &witness,
        &backend,
    )
    .err()
    .unwrap();
    assert_failed_batch(error, expected_batch);
    let error = verify_with_transcript::<TransparentBits, RecordedTranscript>(
        &preprocessing.verifier,
        &changed,
        &proof,
    )
    .err()
    .unwrap();
    assert_statement_binding_rejection(error);
}

/// The verifier rejection checks statement binding into the transcript, not the equation.
#[test]
fn wrong_entry_is_rejected_by_proving_and_statement_binding() {
    let ProvingFailureFixture {
        statement,
        changed,
        verifier,
        witness,
        expected_batch,
    } = support::proving_failure_fixture(ProvingFailure::WrongEntry);
    let preprocessing = ProverPreprocessing {
        verifier,
        scheme: (),
    };
    let backend = Rv64iBackend::reference();
    let (proof, _) = prove_with_transcript::<TransparentBits, RecordedTranscript>(
        &preprocessing,
        &statement,
        &witness,
        &backend,
    )
    .unwrap();
    let error = prove_with_transcript::<TransparentBits, RecordedTranscript>(
        &preprocessing,
        &changed,
        &witness,
        &backend,
    )
    .err()
    .unwrap();
    assert_failed_batch(error, expected_batch);
    let error = verify_with_transcript::<TransparentBits, RecordedTranscript>(
        &preprocessing.verifier,
        &changed,
        &proof,
    )
    .err()
    .unwrap();
    assert_statement_binding_rejection(error);
}
