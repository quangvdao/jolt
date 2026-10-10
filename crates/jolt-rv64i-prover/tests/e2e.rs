//! Complete binary-field RV64I protocol execution on interpreted programs.
#[expect(dead_code, reason = "shared machine helpers serve the complete protocol corpus")]
mod support;
use jolt_rv64i_prover::{
    backend::Rv64iBackend,
    commitment::transparent::TransparentBits,
    prover::{prove, ProverPreprocessing},
};
use jolt_rv64i_verifier::{transcript::Rv64iTranscript, verifier::verify};
use jolt_transcript::Transcript;
#[test]
fn counting_loop_end_to_end() {
    let (statement, verifier, witness) = support::counting_loop();
    let preprocessing = ProverPreprocessing {
        verifier,
        scheme: (),
    };
    let (proof, mut prover) = prove::<TransparentBits, Rv64iTranscript>(
        &preprocessing,
        &Rv64iBackend::reference(),
        &statement,
        &witness,
    )
    .unwrap();
    let mut verifier =
        verify::<TransparentBits, Rv64iTranscript>(&preprocessing.verifier, &statement, &proof)
            .unwrap();
    assert_eq!(prover.challenge(), verifier.challenge());
}
