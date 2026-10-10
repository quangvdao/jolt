//! Ordered proving over shared packed witnesses and the eight kernel registries.
use crate::{
    backend::Rv64iBackend,
    commitment::BitsCommitmentProver,
    error::Rv64iProverError,
    plane::Rv64iWitness,
    stages::{stage1, stage2, stage3a, stage3b, stage4, stage5, stage6a, stage6b},
};
use jolt_field::F128;
use jolt_kernels::ProofSession;
use jolt_rv64i_verifier::{
    commitment::{BitsGeometry, BitsOpening},
    preprocessing::VerifierPreprocessing,
    proof::Rv64iProof,
    statement::{CheckedInputs, Statement},
    transcript::{preamble, Rv64iTranscript},
};
use jolt_transcript::Transcript;

/// Shared verifier preprocessing and scheme-specific prover setup for a single program.
pub struct ProverPreprocessing<S: BitsCommitmentProver> {
    pub verifier: VerifierPreprocessing<S>,
    pub scheme: S::ProverSetup,
}

/// Proves the statement using the reference protocol transcript and the supplied kernel registries.
/// Returns the first witness, batch or commitment error; output agreement is checked by the protocol.
pub fn prove<S: BitsCommitmentProver>(
    preprocessing: &ProverPreprocessing<S>,
    statement: &Statement,
    witness: &Rv64iWitness,
    backend: &Rv64iBackend<F128>,
) -> Result<Rv64iProof<S>, Rv64iProverError> {
    prove_with_transcript::<S, Rv64iTranscript>(preprocessing, backend, statement, witness)
        .map(|(proof, _)| proof)
}

/// Proves the checked statement using the witness's RAM exponent and final PC.
/// Checks row count, layout and initial RAM before commitment; public outputs remain protocol obligations.
pub fn prove_with_transcript<S: BitsCommitmentProver, T: Transcript<Challenge = F128>>(
    preprocessing: &ProverPreprocessing<S>,
    backend: &Rv64iBackend,
    statement: &Statement,
    witness: &Rv64iWitness,
) -> Result<(Rv64iProof<S>, T), Rv64iProverError> {
    let log_K_ram =
        u8::try_from(witness.layout.log_K_ram()).map_err(|_| Rv64iProverError::RamDimension {
            log_K_ram: witness.layout.log_K_ram(),
        })?;
    let checked = CheckedInputs::of_statement(
        &preprocessing.verifier,
        statement,
        log_K_ram,
        witness.final_pc,
    )?;
    if witness.bits.len() != 1_usize << checked.log_T() || witness.words.len() != witness.bits.len()
    {
        return Err(Rv64iProverError::RowCount {
            rows: witness.bits.len(),
        });
    }
    if witness.layout.log_K_bytecode() != checked.log_K_bytecode()
        || witness.layout.log_K_ram() != checked.log_K_ram()
        || witness.layout.lowest_address() != checked.layout().lowest_address()
    {
        return Err(Rv64iProverError::OutputLayoutMismatch);
    }
    if witness.initial_ram != checked.initial_ram() {
        return Err(Rv64iProverError::InitialRamMismatch);
    }
    let mut transcript = preamble::<S, T>(&checked);
    let geometry = BitsGeometry {
        log_T: checked.log_T(),
    };
    let (bits_commitment, state) = S::commit(
        &preprocessing.scheme,
        geometry,
        &witness.bits,
        &mut transcript,
    )
    .map_err(|error| Rv64iProverError::Scheme(Box::new(error)))?;
    let mut session = ProofSession::default();
    let (stage1, s1) = stage1::prove(
        &checked,
        witness,
        &backend.stage1,
        &mut session,
        &mut transcript,
    )
    .map_err(|error| error.in_batch("1"))?;
    let (stage2, s2) = stage2::prove(
        &checked,
        witness,
        &backend.stage2,
        &mut session,
        &mut transcript,
        &s1,
    )
    .map_err(|error| error.in_batch("2"))?;
    let (stage3a, s3a) = stage3a::prove(
        &checked,
        witness,
        &backend.stage3a,
        &mut session,
        &mut transcript,
        &s2,
    )
    .map_err(|error| error.in_batch("3a"))?;
    let (stage3b, s3b) = stage3b::prove(
        &checked,
        witness,
        &backend.stage3b,
        &mut session,
        &mut transcript,
        &s1,
        &s3a,
    )
    .map_err(|error| error.in_batch("3b"))?;
    let (stage4, s4) = stage4::prove(
        &checked,
        witness,
        &backend.stage4,
        &mut session,
        &mut transcript,
        &s3a,
        &s3b,
    )
    .map_err(|error| error.in_batch("4"))?;
    let (stage5, s5) = stage5::prove(
        &checked,
        witness,
        &backend.stage5,
        &mut session,
        &mut transcript,
        &s3a,
        &s4,
    )
    .map_err(|error| error.in_batch("5"))?;
    let (stage6a, s6a) = stage6a::prove(
        &checked,
        witness,
        &backend.stage6a,
        &mut session,
        &mut transcript,
        &s3a,
        &s3b,
        &s4,
        &s5,
    )
    .map_err(|error| error.in_batch("6a"))?;
    let (stage6b, s6b) = stage6b::prove(
        &checked,
        witness,
        &backend.stage6b,
        &mut session,
        &mut transcript,
        &s1,
        &s2,
        &s3a,
        &s3b,
        &s4,
        &s5,
        &s6a,
    )
    .map_err(|error| error.in_batch("6b"))?;
    let rho = transcript.challenge_vector(8);
    let opening = S::open(
        &preprocessing.scheme,
        state,
        &BitsOpening {
            geometry,
            column_point: &rho,
            cycle_point: &s6b.point,
            columns: &stage6b.values.0,
        },
        &mut transcript,
    )
    .map_err(|error| Rv64iProverError::Scheme(Box::new(error)))?;
    Ok((
        Rv64iProof {
            log_K_ram,
            final_pc: witness.final_pc,
            bits_commitment,
            stage1,
            stage2,
            stage3a,
            stage3b,
            stage4,
            stage5,
            stage6a,
            stage6b,
            opening,
        },
        transcript,
    ))
}
