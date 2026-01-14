use super::program::Program;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;
use crate::poly::commitment::dory::DoryCommitmentScheme;
use crate::transcripts::Transcript;
use crate::zkvm::config::{OuterStage1RemainderImpl, OuterStreamingScheduleKind};
use crate::zkvm::proof_serialization::{JoltProof, SpartanOuterStage1Kind};
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::zkvm::ProverDebugInfo;
use common::jolt_device::MemoryLayout;
use tracer::JoltDevice;

#[cfg(feature = "prover")]
fn spartan_outer_stage1_kind_from_env() -> SpartanOuterStage1Kind {
    let kind = std::env::var("SPARTAN_OUTER_STAGE1_KIND")
        .unwrap_or_else(|_| "uniskip".to_string())
        .to_lowercase();

    match kind.as_str() {
        "uniskip" => {
            let remainder_impl = std::env::var("OUTER_STAGE1_REMAINDER_IMPL")
                .unwrap_or_else(|_| "streaming".to_string())
                .to_lowercase();
            let schedule = std::env::var("OUTER_STAGE1_SCHEDULE")
                .unwrap_or_else(|_| "linear-only".to_string())
                .to_lowercase();

            let remainder_impl = match remainder_impl.as_str() {
                "streaming" => OuterStage1RemainderImpl::Streaming,
                "streaming-mtable" | "streaming_mtable" | "mtable" => {
                    OuterStage1RemainderImpl::StreamingMTable
                }
                "checkpoint" | "nonstreaming-checkpoint" | "non_streaming_checkpoint" => {
                    OuterStage1RemainderImpl::NonStreamingCheckpoint
                }
                other => {
                    tracing::warn!(
                        "Unknown OUTER_STAGE1_REMAINDER_IMPL={other}, defaulting to streaming"
                    );
                    OuterStage1RemainderImpl::Streaming
                }
            };

            let schedule = match schedule.as_str() {
                "linear-only" | "linear_only" => OuterStreamingScheduleKind::LinearOnly,
                "half-split" | "half_split" => OuterStreamingScheduleKind::HalfSplit,
                other => {
                    tracing::warn!(
                        "Unknown OUTER_STAGE1_SCHEDULE={other}, defaulting to linear-only"
                    );
                    OuterStreamingScheduleKind::LinearOnly
                }
            };

            SpartanOuterStage1Kind::UniSkipPlusRemainder {
                remainder_impl,
                schedule,
            }
        }
        "full-baseline" | "full_baseline" => SpartanOuterStage1Kind::FullBaseline,
        "full-naive" | "full_naive" => SpartanOuterStage1Kind::FullNaive,
        "full-round-batched" | "full_round_batched" | "full-roundbatched" => {
            SpartanOuterStage1Kind::FullRoundBatched
        }
        other => {
            tracing::warn!("Unknown SPARTAN_OUTER_STAGE1_KIND={other}, defaulting to uniskip");
            SpartanOuterStage1Kind::UniSkipPlusRemainder {
                remainder_impl: OuterStage1RemainderImpl::Streaming,
                schedule: OuterStreamingScheduleKind::LinearOnly,
            }
        }
    }
}

#[allow(clippy::type_complexity)]
#[cfg(feature = "prover")]
pub fn preprocess(
    guest: &Program,
    max_trace_length: usize,
) -> JoltProverPreprocessing<ark_bn254::Fr, DoryCommitmentScheme> {
    use crate::zkvm::verifier::JoltSharedPreprocessing;

    let (bytecode, memory_init, program_size) = guest.decode();

    let mut memory_config = guest.memory_config;
    memory_config.program_size = Some(program_size);
    let memory_layout = MemoryLayout::new(&memory_config);
    let shared_preprocessing =
        JoltSharedPreprocessing::new(bytecode, memory_layout, memory_init, max_trace_length);
    JoltProverPreprocessing::new(shared_preprocessing)
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
#[cfg(feature = "prover")]
pub fn prove<F: JoltField, PCS: StreamingCommitmentScheme<Field = F>, FS: Transcript>(
    guest: &Program,
    inputs_bytes: &[u8],
    untrusted_advice_bytes: &[u8],
    trusted_advice_bytes: &[u8],
    trusted_advice_commitment: Option<<PCS as CommitmentScheme>::Commitment>,
    trusted_advice_hint: Option<<PCS as CommitmentScheme>::OpeningProofHint>,
    output_bytes: &mut [u8],
    preprocessing: &JoltProverPreprocessing<F, PCS>,
) -> (
    JoltProof<F, PCS, FS>,
    JoltDevice,
    Option<ProverDebugInfo<F, FS, PCS>>,
) {
    use crate::zkvm::prover::JoltCpuProver;

    let prover = JoltCpuProver::gen_from_elf(
        preprocessing,
        &guest.elf_contents,
        inputs_bytes,
        untrusted_advice_bytes,
        trusted_advice_bytes,
        trusted_advice_commitment,
        trusted_advice_hint,
    )
    .with_spartan_outer_stage1_kind(spartan_outer_stage1_kind_from_env());
    let io_device = prover.program_io.clone();
    let (proof, debug_info) = prover.prove();
    output_bytes[..io_device.outputs.len()].copy_from_slice(&io_device.outputs);
    (proof, io_device, debug_info)
}
