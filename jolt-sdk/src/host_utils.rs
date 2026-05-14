#[cfg(feature = "host")]
pub use jolt_core::host;
#[cfg(feature = "host")]
pub use jolt_core::zkvm::proof_serialization::serialize_and_print_size;
#[cfg(feature = "host")]
pub use jolt_core::zkvm::{prover::JoltProverPreprocessing, RV64IMACProver};
#[cfg(feature = "host")]
pub use jolt_program::execution::{
    ExecutionBackend, OwnedTrace, TraceError, TraceInputs, TraceOutput, TraceSource,
};
#[cfg(feature = "host")]
pub use tracer::TracerBackend;

pub use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
pub use jolt_core::curve::JoltCurve;
pub use jolt_core::field::JoltField;
pub use jolt_core::guest;
pub use jolt_core::zkvm::{
    bytecode::PreprocessingError, proof_serialization::JoltProof,
    verifier::JoltSharedPreprocessing, verifier::JoltVerifierPreprocessing, RV64IMACProof,
    RV64IMACVerifier, Serializable,
};
pub use jolt_core::AdviceTape;
pub use jolt_crypto::{Bn254 as Curve, Commitment};
pub use jolt_dory::DoryScheme as PCS;
pub use jolt_field::Fr as F;

// Re-exports needed by the provable macro
pub use jolt_core::poly::commitment::dory::{DoryContext, DoryGlobals};
pub use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
pub use jolt_core::zkvm::ram::populate_memory_states;
pub use jolt_core::zkvm::verifier::BlindfoldSetup;
pub use jolt_core::zkvm::witness::PolynomialCommitmentSource;
pub use jolt_openings::{CommitmentScheme, CommitmentSchemeVerifier};
