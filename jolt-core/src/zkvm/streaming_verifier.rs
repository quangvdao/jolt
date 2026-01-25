//! Streaming verifier support for proof bundles.
//!
//! Goal: minimize guest-side *deserialization* cycles by avoiding construction of a full
//! `JoltProof` object (which includes allocations like `BTreeMap` + large `Vec`s), while
//! preserving transcript equivalence.
//!
//! Call-graph (ground truth; see `jolt-core/src/zkvm/verifier.rs`):
//! - `JoltVerifier::new` populates `opening_accumulator` from `proof.opening_claims` and
//!   sets up transcript and params.
//! - `JoltVerifier::verify` runs stages 1..8 and then recursion (stage 8 sub-step) when enabled.
//! - `verify_stage8_with_recursion` performs PCS verification (Dory with hint), then constructs
//!   `RecursionVerifierInput` from `proof.stage10_recursion_metadata`, then calls
//!   `RecursionVerifier::verify` on `proof.recursion_proof`.
//!
//! Key proof/preprocessing fields consumed (non-exhaustive; see `jolt-core/src/zkvm/verifier.rs`
//! and `jolt-core/src/zkvm/recursion/verifier.rs`):
//! - Preprocessing: `shared.memory_layout`, `program` (Committed/Full), `generators` (Dory setup),
//!   `hyrax_recursion_setup` (Hyrax verifier setup for recursion PCS).
//! - Proof: `opening_claims`, `commitments`, stage1..stage7 proofs, stage8 opening proof/hints,
//!   stage10 recursion metadata, `recursion_proof`.
//!
//! This module provides a `verify_from_proof_bytes` entrypoint that consumes a proof-bundle
//! proof encoding (not arkworks canonical `JoltProof` encoding) and drives verification while
//! parsing.

use crate::{
    field::JoltField,
    poly::commitment::commitment_scheme::{CommitmentScheme, RecursionExt},
    poly::commitment::dory::DoryLayout,
    poly::opening_proof::{OpeningId, OpeningPoint, VerifierOpeningAccumulator},
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::{
        config::{OneHotConfig, OneHotParams, ProgramMode, ReadWriteConfig},
        fiat_shamir_preamble,
        r1cs::key::UniformSpartanKey,
        spartan::{verify_stage1_uni_skip, verify_stage2_uni_skip},
        witness::CommittedPolynomial,
    },
};

use crate::zkvm::serialized_bundle::PROOF_RECORD_VERSION;
use crate::zkvm::streaming_decode::SliceReader;
use anyhow::Context;
use jolt_platform::{end_cycle_tracking, start_cycle_tracking};

// Keep marker labels identical to `zkvm/verifier.rs` for trace comparability.
// These must be static strings: the tracer keys markers by the guest string pointer.
const CYCLE_VERIFY_STAGE1: &str = "jolt_verify_stage1";
const CYCLE_VERIFY_STAGE2: &str = "jolt_verify_stage2";
const CYCLE_VERIFY_STAGE3: &str = "jolt_verify_stage3";
const CYCLE_VERIFY_STAGE4: &str = "jolt_verify_stage4";
const CYCLE_VERIFY_STAGE5: &str = "jolt_verify_stage5";
const CYCLE_VERIFY_STAGE6A: &str = "jolt_verify_stage6a";
const CYCLE_VERIFY_STAGE6B: &str = "jolt_verify_stage6b";
const CYCLE_VERIFY_STAGE7: &str = "jolt_verify_stage7";
const CYCLE_VERIFY_STAGE8: &str = "jolt_verify_stage8";
const CYCLE_VERIFY_STAGE8_DORY_PCS: &str = "jolt_verify_stage8_dory_pcs";
const CYCLE_VERIFY_STAGE8_RECURSION: &str = "jolt_verify_stage8_recursion";

// Extra Stage 8 submarkers to attribute "stage8 other" costs.
const CYCLE_VERIFY_STAGE8_DECODE_OPENING_PROOF: &str = "jolt_verify_stage8_decode_opening_proof";
const CYCLE_VERIFY_STAGE8_DECODE_COMBINE_HINT: &str = "jolt_verify_stage8_decode_combine_hint";
const CYCLE_VERIFY_STAGE8_DECODE_PCS_HINT: &str = "jolt_verify_stage8_decode_pcs_hint";
const CYCLE_VERIFY_STAGE8_DECODE_RECURSION_METADATA: &str =
    "jolt_verify_stage8_decode_recursion_metadata";
const CYCLE_VERIFY_STAGE8_DECODE_RECURSION_PROOF: &str = "jolt_verify_stage8_decode_recursion_proof";
const CYCLE_VERIFY_STAGE8_BUILD_RECURSION_INPUT: &str = "jolt_verify_stage8_build_recursion_input";
const CYCLE_VERIFY_STAGE8_CREATE_RECURSION_VERIFIER: &str =
    "jolt_verify_stage8_create_recursion_verifier";

struct CycleMarkerGuard(&'static str);
impl CycleMarkerGuard {
    #[inline(always)]
    fn new(label: &'static str) -> Self {
        start_cycle_tracking(label);
        Self(label)
    }
}
impl Drop for CycleMarkerGuard {
    #[inline(always)]
    fn drop(&mut self) {
        end_cycle_tracking(self.0);
    }
}

/// Streaming verifier state that mirrors `zkvm::verifier::JoltVerifier`, but consumes a
/// proof-bundle proof record instead of a `JoltProof` struct.
pub struct StreamingJoltVerifier<
    'a,
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + RecursionExt<F>,
    T: Transcript,
> where
    <PCS as RecursionExt<F>>::Hint: Send + Sync + Clone + 'static,
{
    preprocessing: &'a crate::zkvm::verifier::JoltVerifierPreprocessing<F, PCS>,
    trusted_advice_commitment: Option<PCS::Commitment>,
    program_io: tracer::JoltDevice,

    // Proof-configs (must match canonical verifier behavior)
    trace_length: usize,
    ram_K: usize,
    bytecode_K: usize,
    program_mode: ProgramMode,
    rw_config: ReadWriteConfig,
    one_hot_config: OneHotConfig,
    dory_layout: DoryLayout,

    transcript: T,
    opening_accumulator: VerifierOpeningAccumulator<F>,
    /// Cached across Stage 6b → 7.
    advice_reduction_verifier_trusted:
        Option<crate::zkvm::claim_reductions::AdviceClaimReductionVerifier<F>>,
    /// Cached across Stage 6b → 7.
    advice_reduction_verifier_untrusted:
        Option<crate::zkvm::claim_reductions::AdviceClaimReductionVerifier<F>>,
    /// Cached across Stage 6b → 7 (Committed mode only).
    bytecode_reduction_verifier:
        Option<crate::zkvm::claim_reductions::BytecodeClaimReductionVerifier<F>>,
    spartan_key: UniformSpartanKey<F>,
    one_hot_params: OneHotParams,

    // Commitments required by Stage 8
    commitments: Vec<PCS::Commitment>,
    untrusted_advice_commitment: Option<PCS::Commitment>,

    /// Whether this proof-record contains the optional recursion payload.
    has_recursion: bool,
}

impl<'a, F, PCS, T> StreamingJoltVerifier<'a, F, PCS, T>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>
        + RecursionExt<F, Hint = crate::poly::commitment::dory::recursion::JoltHintMap>,
    T: Transcript,
    <PCS as RecursionExt<F>>::Hint: Send + Sync + Clone + 'static,
{
    pub fn from_proof_bytes<'b>(
        preprocessing: &'a crate::zkvm::verifier::JoltVerifierPreprocessing<F, PCS>,
        program_io: tracer::JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        mut r: SliceReader<'b>,
    ) -> Result<(Self, SliceReader<'b>), ProofVerifyError> {
        // Record version
        let version = r
            .read_u32_le()
            .map_err(|_| ProofVerifyError::InternalError)?;
        if version != 2 && version != PROOF_RECORD_VERSION {
            return Err(ProofVerifyError::InternalError);
        }

        let trace_length = r
            .read_usize_u64_le()
            .map_err(|_| ProofVerifyError::InternalError)?;
        let ram_K = r
            .read_usize_u64_le()
            .map_err(|_| ProofVerifyError::InternalError)?;
        let bytecode_K = r
            .read_usize_u64_le()
            .map_err(|_| ProofVerifyError::InternalError)?;

        let program_mode: ProgramMode = r
            .read_canonical(ark_serialize::Compress::Yes, ark_serialize::Validate::No)
            .map_err(|_| ProofVerifyError::InternalError)?;
        let rw_config: ReadWriteConfig = r
            .read_canonical(ark_serialize::Compress::Yes, ark_serialize::Validate::No)
            .map_err(|_| ProofVerifyError::InternalError)?;
        let one_hot_config: OneHotConfig = r
            .read_canonical(ark_serialize::Compress::Yes, ark_serialize::Validate::No)
            .map_err(|_| ProofVerifyError::InternalError)?;
        let dory_layout: DoryLayout = r
            .read_canonical(ark_serialize::Compress::Yes, ark_serialize::Validate::No)
            .map_err(|_| ProofVerifyError::InternalError)?;

        // v2: legacy (always-recursive) record format
        // v3: tagged record format (base vs recursion)
        let has_recursion = if version == 2 {
            true
        } else {
            let b = r.read_u8().map_err(|_| ProofVerifyError::InternalError)?;
            match b {
                0 => false,
                1 => true,
                _ => return Err(ProofVerifyError::InternalError),
            }
        };

        // Memory layout checks (mirrors `JoltVerifier::new`).
        if program_io.memory_layout != preprocessing.shared.memory_layout {
            return Err(ProofVerifyError::MemoryLayoutMismatch);
        }
        if program_io.inputs.len() > preprocessing.shared.memory_layout.max_input_size as usize {
            return Err(ProofVerifyError::InputTooLarge);
        }
        if program_io.outputs.len() > preprocessing.shared.memory_layout.max_output_size as usize {
            return Err(ProofVerifyError::OutputTooLarge);
        }

        // Populate opening accumulator directly from the encoded opening-claims table.
        let mut opening_accumulator = VerifierOpeningAccumulator::new(trace_length.log_2());
        let n_claims = r
            .read_u32_le()
            .map_err(|_| ProofVerifyError::InternalError)? as usize;
        for _ in 0..n_claims {
            let key: OpeningId = r
                .read_canonical(ark_serialize::Compress::Yes, ark_serialize::Validate::No)
                .map_err(|_| ProofVerifyError::InternalError)?;
            let claim: F = r
                .read_canonical(ark_serialize::Compress::Yes, ark_serialize::Validate::No)
                .map_err(|_| ProofVerifyError::InternalError)?;
            opening_accumulator
                .openings
                .insert(key, (OpeningPoint::default(), claim));
        }

        // Commitments (length-delimited).
        let n_commitments = r
            .read_u32_le()
            .map_err(|_| ProofVerifyError::InternalError)? as usize;
        let mut commitments = Vec::with_capacity(n_commitments);
        for _ in 0..n_commitments {
            let c: PCS::Commitment = r
                .read_canonical(ark_serialize::Compress::Yes, ark_serialize::Validate::No)
                .map_err(|_| ProofVerifyError::InternalError)?;
            commitments.push(c);
        }

        let untrusted_advice_commitment: Option<PCS::Commitment> = r
            .read_canonical(ark_serialize::Compress::Yes, ark_serialize::Validate::No)
            .map_err(|_| ProofVerifyError::InternalError)?;

        // Transcript init (mirrors non-test `JoltVerifier::new`).
        let transcript = T::new(b"Jolt");

        let spartan_key = UniformSpartanKey::new(trace_length.next_power_of_two());

        // Validate configs from the proof (mirrors `JoltVerifier::new`).
        one_hot_config
            .validate()
            .map_err(ProofVerifyError::InvalidOneHotConfig)?;
        rw_config
            .validate(trace_length.log_2(), ram_K.log_2())
            .map_err(ProofVerifyError::InvalidReadWriteConfig)?;
        let one_hot_params = OneHotParams::from_config(&one_hot_config, bytecode_K, ram_K);

        Ok((
            Self {
                preprocessing,
                trusted_advice_commitment,
                program_io,
                trace_length,
                ram_K,
                bytecode_K,
                program_mode,
                rw_config,
                one_hot_config,
                dory_layout,
                transcript,
                opening_accumulator,
                advice_reduction_verifier_trusted: None,
                advice_reduction_verifier_untrusted: None,
                bytecode_reduction_verifier: None,
                spartan_key,
                one_hot_params,
                commitments,
                untrusted_advice_commitment,
                has_recursion,
            },
            r,
        ))
    }

    /// Verify Stage 1 (uni-skip + batched sumcheck), consuming bytes from the proof record.
    fn verify_stage1_from_reader(&mut self, r: &mut SliceReader<'_>) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE1);
        // Stage 1 uni-skip proof (canonical)
        let uni_skip: crate::subprotocols::univariate_skip::UniSkipFirstRoundProof<F, T> = r
            .read_canonical(ark_serialize::Compress::Yes, ark_serialize::Validate::No)
            .map_err(|e| anyhow::anyhow!("decode stage1 uni-skip: {e:?}"))?;

        let uni_skip_params = verify_stage1_uni_skip(
            &uni_skip,
            &self.spartan_key,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 1 univariate skip first round")?;

        let spartan_outer_remaining =
            crate::zkvm::spartan::outer::OuterRemainingSumcheckVerifier::new(
                self.spartan_key,
                self.trace_length,
                uni_skip_params,
                &self.opening_accumulator,
            );

        verify_batched_sumcheck_streaming(
            r,
            vec![&spartan_outer_remaining],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 1")?;

        Ok(())
    }

    /// Verify Stage 2 (uni-skip + batched sumcheck), consuming bytes from the proof record.
    fn verify_stage2_from_reader(&mut self, r: &mut SliceReader<'_>) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE2);
        let uni_skip: crate::subprotocols::univariate_skip::UniSkipFirstRoundProof<F, T> = r
            .read_canonical(ark_serialize::Compress::Yes, ark_serialize::Validate::No)
            .map_err(|e| anyhow::anyhow!("decode stage2 uni-skip: {e:?}"))?;

        let uni_skip_params = verify_stage2_uni_skip(
            &uni_skip,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 2 univariate skip first round")?;

        let spartan_product_virtual_remainder =
            crate::zkvm::spartan::product::ProductVirtualRemainderVerifier::new(
                self.trace_length,
                uni_skip_params,
                &self.opening_accumulator,
            );
        let ram_raf_evaluation =
            crate::zkvm::ram::raf_evaluation::RafEvaluationSumcheckVerifier::new(
                &self.program_io.memory_layout,
                &self.one_hot_params,
                &self.opening_accumulator,
            );
        let ram_read_write_checking =
            crate::zkvm::ram::read_write_checking::RamReadWriteCheckingVerifier::new(
                &self.opening_accumulator,
                &mut self.transcript,
                &self.one_hot_params,
                self.trace_length,
                &self.rw_config,
            );
        let ram_output_check = crate::zkvm::ram::output_check::OutputSumcheckVerifier::new(
            self.ram_K,
            &self.program_io,
            &mut self.transcript,
        );
        let instruction_claim_reduction =
            crate::zkvm::claim_reductions::InstructionLookupsClaimReductionSumcheckVerifier::new(
                self.trace_length,
                &self.opening_accumulator,
                &mut self.transcript,
            );

        verify_batched_sumcheck_streaming(
            r,
            vec![
                &spartan_product_virtual_remainder,
                &ram_raf_evaluation,
                &ram_read_write_checking,
                &ram_output_check,
                &instruction_claim_reduction,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 2")?;

        Ok(())
    }

    fn verify_stage3_from_reader(&mut self, r: &mut SliceReader<'_>) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE3);
        let spartan_shift = crate::zkvm::spartan::shift::ShiftSumcheckVerifier::new(
            self.trace_length.log_2(),
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let spartan_instruction_input =
            crate::zkvm::spartan::instruction_input::InstructionInputSumcheckVerifier::new(
                &self.opening_accumulator,
                &mut self.transcript,
            );
        let spartan_registers_claim_reduction =
            crate::zkvm::claim_reductions::RegistersClaimReductionSumcheckVerifier::new(
                self.trace_length,
                &self.opening_accumulator,
                &mut self.transcript,
            );

        verify_batched_sumcheck_streaming(
            r,
            vec![
                &spartan_shift
                    as &dyn crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>,
                &spartan_instruction_input,
                &spartan_registers_claim_reduction,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 3")?;

        Ok(())
    }

    fn verify_stage4_from_reader(&mut self, r: &mut SliceReader<'_>) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE4);
        crate::zkvm::ram::verifier_accumulate_advice::<F>(
            self.ram_K,
            &self.program_io,
            self.untrusted_advice_commitment.is_some(),
            self.trusted_advice_commitment.is_some(),
            &mut self.opening_accumulator,
            &mut self.transcript,
            self.rw_config
                .needs_single_advice_opening(self.trace_length.log_2()),
        );
        if self.program_mode == ProgramMode::Committed {
            crate::zkvm::ram::verifier_accumulate_program_image::<F>(
                self.ram_K,
                &self.program_io,
                &mut self.opening_accumulator,
                &mut self.transcript,
                self.rw_config
                    .needs_single_advice_opening(self.trace_length.log_2()),
            );
        }

        let registers_read_write_checking =
            crate::zkvm::registers::read_write_checking::RegistersReadWriteCheckingVerifier::new(
                self.trace_length,
                &self.opening_accumulator,
                &mut self.transcript,
                &self.rw_config,
            );

        let program_image_words = self.preprocessing.program.program_image_words();
        let ram_val_evaluation =
            crate::zkvm::ram::val_evaluation::ValEvaluationSumcheckVerifier::new(
                &self.preprocessing.shared.program_meta,
                program_image_words,
                &self.program_io,
                self.trace_length,
                self.ram_K,
                self.program_mode,
                &self.opening_accumulator,
            );
        let ram_val_final = crate::zkvm::ram::val_final::ValFinalSumcheckVerifier::new(
            &self.preprocessing.shared.program_meta,
            program_image_words,
            &self.program_io,
            self.trace_length,
            self.ram_K,
            self.program_mode,
            &self.opening_accumulator,
            &self.rw_config,
        );

        verify_batched_sumcheck_streaming(
            r,
            vec![
                &registers_read_write_checking
                    as &dyn crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>,
                &ram_val_evaluation,
                &ram_val_final,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 4")?;

        Ok(())
    }

    fn verify_stage5_from_reader(&mut self, r: &mut SliceReader<'_>) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE5);
        let n_cycle_vars = self.trace_length.log_2();
        let registers_val_evaluation =
            crate::zkvm::registers::val_evaluation::ValEvaluationSumcheckVerifier::new(
                &self.opening_accumulator,
            );
        let ram_ra_reduction =
            crate::zkvm::claim_reductions::RamRaClaimReductionSumcheckVerifier::new(
                self.trace_length,
                &self.one_hot_params,
                &self.opening_accumulator,
                &mut self.transcript,
            );
        let lookups_read_raf =
            crate::zkvm::instruction_lookups::read_raf_checking::InstructionReadRafSumcheckVerifier::new(
                n_cycle_vars,
                &self.one_hot_params,
                &self.opening_accumulator,
                &mut self.transcript,
            );

        verify_batched_sumcheck_streaming(
            r,
            vec![
                &registers_val_evaluation
                    as &dyn crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>,
                &ram_ra_reduction,
                &lookups_read_raf,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 5")?;

        Ok(())
    }

    fn verify_stage6a_from_reader(
        &mut self,
        r: &mut SliceReader<'_>,
    ) -> Result<
        (
            crate::zkvm::bytecode::read_raf_checking::BytecodeReadRafSumcheckParams<F>,
            crate::subprotocols::booleanity::BooleanitySumcheckParams<F>,
        ),
        anyhow::Error,
    > {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE6A);
        let n_cycle_vars = self.trace_length.log_2();
        let program_preprocessing = match self.program_mode {
            ProgramMode::Committed => {
                let _ = self.preprocessing.program.as_committed()?;
                None
            }
            ProgramMode::Full => self.preprocessing.program.full().map(|p| p.as_ref()),
        };
        let bytecode_read_raf =
            crate::zkvm::bytecode::read_raf_checking::BytecodeReadRafAddressSumcheckVerifier::new(
                program_preprocessing,
                n_cycle_vars,
                &self.one_hot_params,
                &self.opening_accumulator,
                &mut self.transcript,
                self.program_mode,
            )?;
        let booleanity_params = crate::subprotocols::booleanity::BooleanitySumcheckParams::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let booleanity = crate::subprotocols::booleanity::BooleanityAddressSumcheckVerifier::new(
            booleanity_params,
        );

        verify_batched_sumcheck_streaming(
            r,
            vec![&bytecode_read_raf, &booleanity],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 6a")?;

        Ok((bytecode_read_raf.into_params(), booleanity.into_params()))
    }

    fn verify_stage6b_from_reader(
        &mut self,
        r: &mut SliceReader<'_>,
        bytecode_read_raf_params: crate::zkvm::bytecode::read_raf_checking::BytecodeReadRafSumcheckParams<F>,
        booleanity_params: crate::subprotocols::booleanity::BooleanitySumcheckParams<F>,
    ) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE6B);
        let booleanity = crate::subprotocols::booleanity::BooleanityCycleSumcheckVerifier::new(
            booleanity_params,
        );
        let ram_hamming_booleanity =
            crate::zkvm::ram::hamming_booleanity::HammingBooleanitySumcheckVerifier::new(
                &self.opening_accumulator,
            );
        let ram_ra_virtual = crate::zkvm::ram::ra_virtual::RamRaVirtualSumcheckVerifier::new(
            self.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_ra_virtual =
            crate::zkvm::instruction_lookups::ra_virtual::RaSumcheckVerifier::new(
                &self.one_hot_params,
                &self.opening_accumulator,
                &mut self.transcript,
            );
        let inc_reduction = crate::zkvm::claim_reductions::IncClaimReductionSumcheckVerifier::new(
            self.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        if self.program_mode == ProgramMode::Committed {
            let params = crate::zkvm::claim_reductions::BytecodeClaimReductionParams::new(
                &bytecode_read_raf_params,
                &self.opening_accumulator,
                &mut self.transcript,
            );
            self.bytecode_reduction_verifier =
                Some(crate::zkvm::claim_reductions::BytecodeClaimReductionVerifier::new(params));
        } else {
            self.bytecode_reduction_verifier = None;
        }

        if self.trusted_advice_commitment.is_some() {
            self.advice_reduction_verifier_trusted = Some(
                crate::zkvm::claim_reductions::AdviceClaimReductionVerifier::new(
                    crate::zkvm::claim_reductions::AdviceKind::Trusted,
                    &self.program_io.memory_layout,
                    self.trace_length,
                    &self.opening_accumulator,
                    &mut self.transcript,
                    self.rw_config
                        .needs_single_advice_opening(self.trace_length.log_2()),
                ),
            );
        }
        if self.untrusted_advice_commitment.is_some() {
            self.advice_reduction_verifier_untrusted = Some(
                crate::zkvm::claim_reductions::AdviceClaimReductionVerifier::new(
                    crate::zkvm::claim_reductions::AdviceKind::Untrusted,
                    &self.program_io.memory_layout,
                    self.trace_length,
                    &self.opening_accumulator,
                    &mut self.transcript,
                    self.rw_config
                        .needs_single_advice_opening(self.trace_length.log_2()),
                ),
            );
        }

        let program_image_reduction = if self.program_mode == ProgramMode::Committed {
            let trusted = self
                .preprocessing
                .program
                .as_committed()
                .expect("program commitments missing in committed mode");
            let padded_len_words = trusted.program_image_num_words;
            let log_t = self.trace_length.log_2();
            let m = padded_len_words.log_2();
            if m > log_t {
                return Err(ProofVerifyError::InvalidBytecodeConfig(format!(
                    "program-image claim reduction requires m=log2(padded_len_words) <= log_T (got m={m}, log_T={log_t})"
                ))
                .into());
            }
            let params = crate::zkvm::claim_reductions::ProgramImageClaimReductionParams::new(
                &self.program_io,
                self.preprocessing.shared.min_bytecode_address(),
                padded_len_words,
                self.ram_K,
                self.trace_length,
                &self.rw_config,
                &self.opening_accumulator,
                &mut self.transcript,
            );
            Some(crate::zkvm::claim_reductions::ProgramImageClaimReductionVerifier { params })
        } else {
            None
        };

        let bytecode_read_raf =
            crate::zkvm::bytecode::read_raf_checking::BytecodeReadRafCycleSumcheckVerifier::new(
                bytecode_read_raf_params,
            );

        let mut instances: Vec<
            &dyn crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>,
        > = vec![
            &bytecode_read_raf,
            &ram_hamming_booleanity,
            &booleanity,
            &ram_ra_virtual,
            &lookups_ra_virtual,
            &inc_reduction,
        ];
        if let Some(ref bytecode) = self.bytecode_reduction_verifier {
            instances.push(bytecode);
        }
        if let Some(ref advice) = self.advice_reduction_verifier_trusted {
            instances.push(advice);
        }
        if let Some(ref advice) = self.advice_reduction_verifier_untrusted {
            instances.push(advice);
        }
        if let Some(ref prog) = program_image_reduction {
            instances.push(prog);
        }

        verify_batched_sumcheck_streaming(
            r,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 6b")?;

        Ok(())
    }

    fn verify_stage7_from_reader(&mut self, r: &mut SliceReader<'_>) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE7);
        let hw_verifier = crate::zkvm::claim_reductions::HammingWeightClaimReductionVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let mut instances: Vec<
            &dyn crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>,
        > = vec![&hw_verifier];

        if let Some(bytecode_reduction_verifier) = self.bytecode_reduction_verifier.as_mut() {
            bytecode_reduction_verifier.params.borrow_mut().phase =
                crate::zkvm::claim_reductions::BytecodeReductionPhase::LaneVariables;
            instances.push(bytecode_reduction_verifier);
        }
        if let Some(advice_reduction_verifier_trusted) =
            self.advice_reduction_verifier_trusted.as_mut()
        {
            let mut params = advice_reduction_verifier_trusted.params.borrow_mut();
            if params.num_address_phase_rounds() > 0 {
                params.phase =
                    crate::zkvm::claim_reductions::advice::ReductionPhase::AddressVariables;
                instances.push(advice_reduction_verifier_trusted);
            }
        }
        if let Some(advice_reduction_verifier_untrusted) =
            self.advice_reduction_verifier_untrusted.as_mut()
        {
            let mut params = advice_reduction_verifier_untrusted.params.borrow_mut();
            if params.num_address_phase_rounds() > 0 {
                params.phase =
                    crate::zkvm::claim_reductions::advice::ReductionPhase::AddressVariables;
                instances.push(advice_reduction_verifier_untrusted);
            }
        }

        verify_batched_sumcheck_streaming(
            r,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 7")?;

        Ok(())
    }

    fn compute_joint_commitment(
        &self,
        commitment_map: &mut std::collections::HashMap<CommittedPolynomial, PCS::Commitment>,
        state: &crate::poly::opening_proof::DoryOpeningState<F>,
    ) -> PCS::Commitment {
        let mut rlc_map = std::collections::HashMap::new();
        for (gamma, (poly, _claim)) in state
            .gamma_powers
            .iter()
            .zip(state.polynomial_claims.iter())
        {
            *rlc_map.entry(*poly).or_insert(F::zero()) += *gamma;
        }

        let (coeffs, commitments): (Vec<F>, Vec<PCS::Commitment>) = rlc_map
            .into_iter()
            .map(|(k, v)| (v, commitment_map.remove(&k).unwrap()))
            .unzip();

        PCS::combine_commitments(&commitments, &coeffs)
    }

    fn compute_stage8_opening_point_and_claims(
        &self,
    ) -> (
        OpeningPoint<{ crate::poly::opening_proof::BIG_ENDIAN }, F>,
        Vec<(CommittedPolynomial, F)>,
        Vec<F>,
    ) {
        use crate::poly::opening_proof::{
            compute_advice_lagrange_factor, OpeningAccumulator, SumcheckId,
        };

        let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        let r_address_stage7 = &opening_point.r[..log_k_chunk];

        let mut polynomial_claims = Vec::new();

        // Dense polynomials (IncClaimReduction)
        let (_, ram_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
        );
        let (_, rd_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
        );

        let lagrange_factor: F = r_address_stage7.iter().map(|r| F::one() - *r).product();
        polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
        polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));

        // Sparse polynomials: all RA polys (HammingWeightClaimReduction)
        for i in 0..self.one_hot_params.instruction_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::InstructionRa(i), claim));
        }
        for i in 0..self.one_hot_params.bytecode_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::BytecodeRa(i), claim));
        }
        for i in 0..self.one_hot_params.ram_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::RamRa(i), claim));
        }

        // Advice polynomials (if present)
        if let Some((advice_point, advice_claim)) = self.opening_accumulator.get_advice_opening(
            crate::zkvm::claim_reductions::AdviceKind::Trusted,
            SumcheckId::AdviceClaimReduction,
        ) {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
            polynomial_claims.push((
                CommittedPolynomial::TrustedAdvice,
                advice_claim * lagrange_factor,
            ));
        }
        if let Some((advice_point, advice_claim)) = self.opening_accumulator.get_advice_opening(
            crate::zkvm::claim_reductions::AdviceKind::Untrusted,
            SumcheckId::AdviceClaimReduction,
        ) {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
            polynomial_claims.push((
                CommittedPolynomial::UntrustedAdvice,
                advice_claim * lagrange_factor,
            ));
        }

        // Bytecode chunk polynomials (Committed mode)
        if self.program_mode == ProgramMode::Committed {
            let (bytecode_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeChunk(0),
                SumcheckId::BytecodeClaimReduction,
            );
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &bytecode_point.r);

            let num_chunks = crate::zkvm::bytecode::chunks::total_lanes()
                .div_ceil(self.one_hot_params.k_chunk);
            for i in 0..num_chunks {
                let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeChunk(i),
                    SumcheckId::BytecodeClaimReduction,
                );
                polynomial_claims.push((
                    CommittedPolynomial::BytecodeChunk(i),
                    claim * lagrange_factor,
                ));
            }
        }

        // Program-image polynomial (Committed mode)
        if self.program_mode == ProgramMode::Committed {
            let (prog_point, prog_claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::ProgramImageInit,
                SumcheckId::ProgramImageClaimReduction,
            );
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &prog_point.r);
            polynomial_claims.push((
                CommittedPolynomial::ProgramImageInit,
                prog_claim * lagrange_factor,
            ));
        }

        let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
        (opening_point, polynomial_claims, claims)
    }

    fn verify_stage8_with_pcs_hint_inner(
        &mut self,
        stage8_opening_proof: &PCS::Proof,
        stage8_combine_hint: &Option<ark_bn254::Fq12>,
        stage8_hint: &<PCS as RecursionExt<F>>::Hint,
    ) -> Result<(), anyhow::Error> {
        use crate::poly::opening_proof::{
            compute_advice_lagrange_factor, DoryOpeningState, OpeningAccumulator, SumcheckId,
        };
        use crate::zkvm::witness::all_committed_polynomials;
        use itertools::Itertools;

        let (opening_point, polynomial_claims, claims) = {
            let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(0),
                SumcheckId::HammingWeightClaimReduction,
            );
            let log_k_chunk = self.one_hot_params.log_k_chunk;
            let r_address_stage7 = &opening_point.r[..log_k_chunk];

            let mut polynomial_claims = Vec::new();

            // Dense polynomials (IncClaimReduction)
            let (_, ram_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::IncClaimReduction,
            );
            let (_, rd_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RdInc,
                SumcheckId::IncClaimReduction,
            );

            let lagrange_factor: F = r_address_stage7.iter().map(|r| F::one() - *r).product();
            polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
            polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));

            // Sparse polynomials: all RA polys (HammingWeightClaimReduction)
            for i in 0..self.one_hot_params.instruction_d {
                let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::HammingWeightClaimReduction,
                );
                polynomial_claims.push((CommittedPolynomial::InstructionRa(i), claim));
            }
            for i in 0..self.one_hot_params.bytecode_d {
                let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeRa(i),
                    SumcheckId::HammingWeightClaimReduction,
                );
                polynomial_claims.push((CommittedPolynomial::BytecodeRa(i), claim));
            }
            for i in 0..self.one_hot_params.ram_d {
                let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::RamRa(i),
                    SumcheckId::HammingWeightClaimReduction,
                );
                polynomial_claims.push((CommittedPolynomial::RamRa(i), claim));
            }

            // Advice polynomials (if present)
            if let Some((advice_point, advice_claim)) = self.opening_accumulator.get_advice_opening(
                crate::zkvm::claim_reductions::AdviceKind::Trusted,
                SumcheckId::AdviceClaimReduction,
            ) {
                let lagrange_factor =
                    compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
                polynomial_claims.push((
                    CommittedPolynomial::TrustedAdvice,
                    advice_claim * lagrange_factor,
                ));
            }
            if let Some((advice_point, advice_claim)) = self.opening_accumulator.get_advice_opening(
                crate::zkvm::claim_reductions::AdviceKind::Untrusted,
                SumcheckId::AdviceClaimReduction,
            ) {
                let lagrange_factor =
                    compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
                polynomial_claims.push((
                    CommittedPolynomial::UntrustedAdvice,
                    advice_claim * lagrange_factor,
                ));
            }

            // Bytecode chunk polynomials (Committed mode)
            if self.program_mode == ProgramMode::Committed {
                let (bytecode_point, _) =
                    self.opening_accumulator.get_committed_polynomial_opening(
                        CommittedPolynomial::BytecodeChunk(0),
                        SumcheckId::BytecodeClaimReduction,
                    );
                let lagrange_factor =
                    compute_advice_lagrange_factor::<F>(&opening_point.r, &bytecode_point.r);

                let num_chunks = crate::zkvm::bytecode::chunks::total_lanes()
                    .div_ceil(self.one_hot_params.k_chunk);
                for i in 0..num_chunks {
                    let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                        CommittedPolynomial::BytecodeChunk(i),
                        SumcheckId::BytecodeClaimReduction,
                    );
                    polynomial_claims.push((
                        CommittedPolynomial::BytecodeChunk(i),
                        claim * lagrange_factor,
                    ));
                }
            }

            // Program-image polynomial (Committed mode)
            if self.program_mode == ProgramMode::Committed {
                let (prog_point, prog_claim) =
                    self.opening_accumulator.get_committed_polynomial_opening(
                        CommittedPolynomial::ProgramImageInit,
                        SumcheckId::ProgramImageClaimReduction,
                    );
                let lagrange_factor =
                    compute_advice_lagrange_factor::<F>(&opening_point.r, &prog_point.r);
                polynomial_claims.push((
                    CommittedPolynomial::ProgramImageInit,
                    prog_claim * lagrange_factor,
                ));
            }

            let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
            (opening_point, polynomial_claims, claims)
        };

        let gamma_powers: Vec<F> = {
            self.transcript.append_scalars(&claims);
            self.transcript.challenge_scalar_powers(claims.len())
        };

        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers: gamma_powers.clone(),
            polynomial_claims,
        };

        let joint_commitment = if let Some(combine_hint) = stage8_combine_hint {
            PCS::combine_with_hint_fq12(combine_hint)
        } else {
            let mut commitments_map = std::collections::HashMap::new();
            for (polynomial, commitment) in all_committed_polynomials(&self.one_hot_params)
                .into_iter()
                .zip_eq(&self.commitments)
            {
                commitments_map.insert(polynomial, commitment.clone());
            }

            if let Some(ref commitment) = self.trusted_advice_commitment {
                if state
                    .polynomial_claims
                    .iter()
                    .any(|(p, _)| *p == CommittedPolynomial::TrustedAdvice)
                {
                    commitments_map.insert(CommittedPolynomial::TrustedAdvice, commitment.clone());
                }
            }
            if let Some(ref commitment) = self.untrusted_advice_commitment {
                if state
                    .polynomial_claims
                    .iter()
                    .any(|(p, _)| *p == CommittedPolynomial::UntrustedAdvice)
                {
                    commitments_map
                        .insert(CommittedPolynomial::UntrustedAdvice, commitment.clone());
                }
            }

            if self.program_mode == ProgramMode::Committed {
                if let Ok(committed) = self.preprocessing.program.as_committed() {
                    for (idx, commitment) in committed.bytecode_commitments.iter().enumerate() {
                        commitments_map
                            .entry(CommittedPolynomial::BytecodeChunk(idx))
                            .or_insert_with(|| commitment.clone());
                    }

                    if state
                        .polynomial_claims
                        .iter()
                        .any(|(p, _)| *p == CommittedPolynomial::ProgramImageInit)
                    {
                        commitments_map.insert(
                            CommittedPolynomial::ProgramImageInit,
                            committed.program_image_commitment.clone(),
                        );
                    }
                }
            }

            self.compute_joint_commitment(&mut commitments_map, &state)
        };

        let joint_claim: F = gamma_powers
            .iter()
            .zip(claims.iter())
            .map(|(gamma, claim)| *gamma * *claim)
            .sum();

        PCS::verify_with_hint(
            stage8_opening_proof,
            &self.preprocessing.generators,
            &mut self.transcript,
            &opening_point.r,
            &joint_claim,
            &joint_commitment,
            stage8_hint,
        )
        .map_err(|_| anyhow::anyhow!("Stage 8 (hint)"))?;

        Ok(())
    }

    fn verify_stage8_native_inner(
        &mut self,
        stage8_opening_proof: &PCS::Proof,
    ) -> Result<(), anyhow::Error> {
        use crate::poly::opening_proof::DoryOpeningState;
        use crate::zkvm::witness::all_committed_polynomials;
        use itertools::Itertools;

        let (opening_point, polynomial_claims, claims) = self.compute_stage8_opening_point_and_claims();

        let gamma_powers: Vec<F> = {
            self.transcript.append_scalars(&claims);
            self.transcript.challenge_scalar_powers(claims.len())
        };

        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers: gamma_powers.clone(),
            polynomial_claims,
        };

        let mut commitments_map = std::collections::HashMap::new();
        for (polynomial, commitment) in all_committed_polynomials(&self.one_hot_params)
            .into_iter()
            .zip_eq(&self.commitments)
        {
            commitments_map.insert(polynomial, commitment.clone());
        }

        if let Some(ref commitment) = self.trusted_advice_commitment {
            if state
                .polynomial_claims
                .iter()
                .any(|(p, _)| *p == CommittedPolynomial::TrustedAdvice)
            {
                commitments_map.insert(CommittedPolynomial::TrustedAdvice, commitment.clone());
            }
        }
        if let Some(ref commitment) = self.untrusted_advice_commitment {
            if state
                .polynomial_claims
                .iter()
                .any(|(p, _)| *p == CommittedPolynomial::UntrustedAdvice)
            {
                commitments_map.insert(CommittedPolynomial::UntrustedAdvice, commitment.clone());
            }
        }

        if self.program_mode == ProgramMode::Committed {
            if let Ok(committed) = self.preprocessing.program.as_committed() {
                for (idx, commitment) in committed.bytecode_commitments.iter().enumerate() {
                    commitments_map
                        .entry(CommittedPolynomial::BytecodeChunk(idx))
                        .or_insert_with(|| commitment.clone());
                }

                if state
                    .polynomial_claims
                    .iter()
                    .any(|(p, _)| *p == CommittedPolynomial::ProgramImageInit)
                {
                    commitments_map.insert(
                        CommittedPolynomial::ProgramImageInit,
                        committed.program_image_commitment.clone(),
                    );
                }
            }
        }

        let joint_commitment = self.compute_joint_commitment(&mut commitments_map, &state);

        let joint_claim: F = gamma_powers
            .iter()
            .zip(claims.iter())
            .map(|(gamma, claim)| *gamma * *claim)
            .sum();

        PCS::verify(
            stage8_opening_proof,
            &self.preprocessing.generators,
            &mut self.transcript,
            &opening_point.r,
            &joint_claim,
            &joint_commitment,
        )
        .map_err(|_| anyhow::anyhow!("Stage 8"))?;

        Ok(())
    }

    fn verify_stage8_from_reader(&mut self, r: &mut SliceReader<'_>) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8);

        // Initialize Dory globals so the Dory PCS verifier can reorder points correctly.
        let _dory_guard = if self.program_mode == ProgramMode::Committed {
            let committed = self.preprocessing.program.as_committed()?;
            crate::poly::commitment::dory::DoryGlobals::initialize_main_context_with_num_columns(
                1 << self.one_hot_params.log_k_chunk,
                self.trace_length.next_power_of_two(),
                committed.bytecode_num_columns,
                Some(self.dory_layout),
            )
        } else {
            crate::poly::commitment::dory::DoryGlobals::initialize_context(
                1 << self.one_hot_params.log_k_chunk,
                self.trace_length.next_power_of_two(),
                crate::poly::commitment::dory::DoryContext::Main,
                Some(self.dory_layout),
            )
        };

        let stage8_opening_proof: PCS::Proof = r
            .read_canonical(ark_serialize::Compress::Yes, ark_serialize::Validate::No)
            .map_err(|e| anyhow::anyhow!("decode stage8_opening_proof: {e:?}"))?;

        self.verify_stage8_native_inner(&stage8_opening_proof)?;
        Ok(())
    }

    fn verify_stage8_with_recursion_from_reader(
        &mut self,
        r: &mut SliceReader<'_>,
    ) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8);

        // Initialize Dory globals so the Dory PCS verifier can reorder points correctly.
        let _dory_guard = if self.program_mode == ProgramMode::Committed {
            let committed = self.preprocessing.program.as_committed()?;
            crate::poly::commitment::dory::DoryGlobals::initialize_main_context_with_num_columns(
                1 << self.one_hot_params.log_k_chunk,
                self.trace_length.next_power_of_two(),
                committed.bytecode_num_columns,
                Some(self.dory_layout),
            )
        } else {
            crate::poly::commitment::dory::DoryGlobals::initialize_context(
                1 << self.one_hot_params.log_k_chunk,
                self.trace_length.next_power_of_two(),
                crate::poly::commitment::dory::DoryContext::Main,
                Some(self.dory_layout),
            )
        };

        // 1) Decode Stage 8 opening proof.
        let stage8_opening_proof: PCS::Proof = {
            let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8_DECODE_OPENING_PROOF);
            r.read_canonical(ark_serialize::Compress::Yes, ark_serialize::Validate::No)
                .map_err(|e| anyhow::anyhow!("decode stage8_opening_proof: {e:?}"))?
        };
        let _stage8_combine_hint: Option<ark_bn254::Fq12> = {
            let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8_DECODE_COMBINE_HINT);
            r.read_canonical(ark_serialize::Compress::Yes, ark_serialize::Validate::No)
                .map_err(|e| anyhow::anyhow!("decode stage8_combine_hint: {e:?}"))?
        };
        // RecursionProofBundle starts here: version + hint map + (later) metadata + recursion proof.
        let _stage9_hint: <PCS as RecursionExt<F>>::Hint = {
            let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8_DECODE_PCS_HINT);
            let v = r
                .read_u32_le()
                .map_err(|e| anyhow::anyhow!("decode RecursionProofBundle version: {e:?}"))?;
            if v != crate::zkvm::recursion_proof_bundle::RECURSION_PROOF_BUNDLE_VERSION {
                return Err(anyhow::anyhow!("unsupported RecursionProofBundle version: {v}"));
            }
            crate::zkvm::recursion_proof_bundle::read_hint_map_record(r)
                .map_err(|e| anyhow::anyhow!("decode stage9_pcs_hint record: {e:?}"))?
        };

        // 2) Decode recursion payload.
        let metadata: crate::zkvm::proof_serialization::RecursionConstraintMetadata = {
            let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8_DECODE_RECURSION_METADATA);
            crate::zkvm::recursion_proof_bundle::read_recursion_constraint_metadata_record(r)
                .map_err(|e| anyhow::anyhow!("decode stage10_recursion_metadata record: {e:?}"))?
        };

        type HyraxPCS = crate::poly::commitment::hyrax::Hyrax<1, ark_grumpkin::Projective>;
        let recursion_proof: crate::zkvm::recursion::prover::RecursionProof<
            ark_bn254::Fq,
            T,
            HyraxPCS,
        > = {
            let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8_DECODE_RECURSION_PROOF);
            crate::zkvm::recursion_proof_bundle::read_recursion_proof_record::<T>(r)
                .map_err(|e| anyhow::anyhow!("decode recursion_proof record: {e:?}"))?
        };

        // 3) Stage 8 Fiat–Shamir replay (no PCS checks): advance transcript to match
        // the prover's post-Stage8 state expected by the recursion SNARK verifier.
        {
            let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8_DORY_PCS);
            let (_opening_point, _poly_claims, claims) = self.compute_stage8_opening_point_and_claims();
            self.transcript.append_scalars(&claims);
            let _gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());
            PCS::replay_opening_proof_transcript(&stage8_opening_proof, &mut self.transcript)
                .map_err(|e| anyhow::anyhow!("Stage 8 PCS FS replay failed: {e:?}"))?;
        }

        // 4) Build recursion verifier input (alloc-heavy: clones large vectors).
        let verifier_input = {
            let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8_BUILD_RECURSION_INPUT);
            let constraint_types = metadata.constraint_types.clone();
            let num_constraints = constraint_types.len();
            let num_constraints_padded = num_constraints.next_power_of_two();
            use crate::zkvm::recursion::constraints::system::PolyType;
            let num_rows_unpadded = PolyType::NUM_TYPES * num_constraints_padded;
            let num_s_vars = (num_rows_unpadded as f64).log2().ceil() as usize;
            let num_constraint_vars = 11;
            let num_vars = num_s_vars + num_constraint_vars;
            crate::zkvm::recursion::verifier::RecursionVerifierInput {
                constraint_types,
                num_vars,
                num_constraint_vars,
                num_s_vars,
                num_constraints,
                num_constraints_padded,
                jagged_bijection: metadata.jagged_bijection.clone(),
                jagged_mapping: metadata.jagged_mapping.clone(),
                matrix_rows: metadata.matrix_rows.clone(),
                gt_exp_public_inputs: metadata.gt_exp_public_inputs.clone(),
                g1_scalar_mul_public_inputs: metadata.g1_scalar_mul_public_inputs.clone(),
                g2_scalar_mul_public_inputs: metadata.g2_scalar_mul_public_inputs.clone(),
            }
        };

        let recursion_verifier = {
            let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8_CREATE_RECURSION_VERIFIER);
            crate::zkvm::recursion::verifier::RecursionVerifier::<ark_bn254::Fq>::new(
                verifier_input,
            )
        };

        if metadata.dense_num_vars > crate::zkvm::recursion::MAX_RECURSION_DENSE_NUM_VARS {
            return Err(anyhow::anyhow!(
                "dense_num_vars {} exceeds max {}",
                metadata.dense_num_vars,
                crate::zkvm::recursion::MAX_RECURSION_DENSE_NUM_VARS
            ));
        }

        let ok = {
            let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8_RECURSION);
            recursion_verifier
                .verify::<T, HyraxPCS>(
                    &recursion_proof,
                    &mut self.transcript,
                    &recursion_proof.dense_commitment,
                    &self.preprocessing.hyrax_recursion_setup,
                )
                .map_err(|e| anyhow::anyhow!("Recursion verification failed: {e:?}"))?
        };

        if !ok {
            return Err(anyhow::anyhow!("Recursion proof verification failed"));
        }

        Ok(())
    }
}

/// Streaming batched-sumcheck verification for the proof-record encoding.
///
/// Encoded sumcheck proof format:
/// - u32 num_rounds
/// - repeated num_rounds times:
///   - u32 coeffs_len
///   - repeated coeffs_len times: field element (canonical compressed)
fn verify_batched_sumcheck_streaming<F: JoltField, ProofTranscript: Transcript>(
    r: &mut SliceReader<'_>,
    sumcheck_instances: Vec<
        &dyn crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, ProofTranscript>,
    >,
    opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut ProofTranscript,
) -> Result<Vec<F::Challenge>, ProofVerifyError> {
    let max_degree = sumcheck_instances
        .iter()
        .map(|sumcheck| sumcheck.degree())
        .max()
        .unwrap();
    let max_num_rounds = sumcheck_instances
        .iter()
        .map(|sumcheck| sumcheck.num_rounds())
        .max()
        .unwrap();

    // Append input claims to transcript
    sumcheck_instances.iter().for_each(|sumcheck| {
        let input_claim = sumcheck.input_claim(opening_accumulator);
        transcript.append_scalar(&input_claim);
    });
    let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

    let claim: F = sumcheck_instances
        .iter()
        .zip(batching_coeffs.iter())
        .map(|(sumcheck, coeff)| {
            let num_rounds = sumcheck.num_rounds();
            let input_claim = sumcheck.input_claim(opening_accumulator);
            input_claim.mul_pow_2(max_num_rounds - num_rounds) * *coeff
        })
        .sum();

    // Read proof rounds
    let proof_rounds = r
        .read_u32_le()
        .map_err(|_| ProofVerifyError::InternalError)? as usize;
    if proof_rounds != max_num_rounds {
        return Err(ProofVerifyError::InvalidInputLength(
            max_num_rounds,
            proof_rounds,
        ));
    }

    let mut e = claim;
    let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
    let mut coeffs: Vec<F> = Vec::new();

    for _round in 0..max_num_rounds {
        transcript.append_message(b"UniPoly_begin");

        let coeffs_len = r
            .read_u32_le()
            .map_err(|_| ProofVerifyError::InternalError)? as usize;
        if coeffs_len == 0 || coeffs_len > max_degree {
            return Err(ProofVerifyError::InvalidInputLength(max_degree, coeffs_len));
        }

        coeffs.clear();
        coeffs.reserve_exact(coeffs_len);
        for _ in 0..coeffs_len {
            let ci: F = r
                .read_canonical(ark_serialize::Compress::Yes, ark_serialize::Validate::No)
                .map_err(|_| ProofVerifyError::InternalError)?;
            transcript.append_scalar(&ci);
            coeffs.push(ci);
        }

        transcript.append_message(b"UniPoly_end");

        let r_i: F::Challenge = transcript.challenge_scalar_optimized::<F>();
        r_sumcheck.push(r_i);

        // Update the running claim using the compressed polynomial and hint `e`.
        // Mirrors `CompressedUniPoly::eval_from_hint`.
        let hint = e;
        let c0 = coeffs[0];
        let mut linear_term = hint - c0 - c0;
        for i in 1..coeffs_len {
            linear_term -= coeffs[i];
        }

        let x = r_i;
        let mut running_point: F = x.into();
        let mut running_sum = c0 + x * linear_term;
        for i in 1..coeffs_len {
            running_point = running_point * &x; // x^{i+1}
            running_sum += coeffs[i] * running_point;
        }
        e = running_sum;
    }

    let output_claim = e;

    let expected_output_claim: F = sumcheck_instances
        .iter()
        .zip(batching_coeffs.iter())
        .map(|(sumcheck, coeff)| {
            let offset = sumcheck.round_offset(max_num_rounds);
            let r_slice = &r_sumcheck[offset..offset + sumcheck.num_rounds()];
            sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
            let claim = sumcheck.expected_output_claim(opening_accumulator, r_slice);
            claim * *coeff
        })
        .sum();

    if output_claim != expected_output_claim {
        return Err(ProofVerifyError::SumcheckVerificationError);
    }

    Ok(r_sumcheck)
}

/// Verify from a proof-record byte slice (inside a proof bundle).
///
/// This currently implements stages 1–2 as the first integration step. Later stages are
/// added in follow-up patches.
pub fn verify_from_proof_bytes<F, PCS, ProofTranscript>(
    preprocessing: &crate::zkvm::verifier::JoltVerifierPreprocessing<F, PCS>,
    program_io: tracer::JoltDevice,
    trusted_advice_commitment: Option<PCS::Commitment>,
    proof_bytes: &[u8],
) -> Result<(), anyhow::Error>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>
        + RecursionExt<F, Hint = crate::poly::commitment::dory::recursion::JoltHintMap>,
    ProofTranscript: Transcript,
    <PCS as RecursionExt<F>>::Hint: Send + Sync + Clone + 'static,
{
    let (mut v, mut r) = StreamingJoltVerifier::<F, PCS, ProofTranscript>::from_proof_bytes(
        preprocessing,
        program_io,
        trusted_advice_commitment,
        SliceReader::new(proof_bytes),
    )
    .map_err(|e| anyhow::anyhow!("init streaming verifier: {e:?}"))?;

    fiat_shamir_preamble(&v.program_io, v.ram_K, v.trace_length, &mut v.transcript);

    for c in &v.commitments {
        v.transcript.append_serializable(c);
    }
    if let Some(ref c) = v.untrusted_advice_commitment {
        v.transcript.append_serializable(c);
    }
    if let Some(ref c) = v.trusted_advice_commitment {
        v.transcript.append_serializable(c);
    }
    if v.program_mode == ProgramMode::Committed {
        let trusted = v.preprocessing.program.as_committed()?;
        for commitment in &trusted.bytecode_commitments {
            v.transcript.append_serializable(commitment);
        }
        v.transcript
            .append_serializable(&trusted.program_image_commitment);
    }

    v.verify_stage1_from_reader(&mut r)?;
    v.verify_stage2_from_reader(&mut r)?;
    v.verify_stage3_from_reader(&mut r)?;
    v.verify_stage4_from_reader(&mut r)?;
    v.verify_stage5_from_reader(&mut r)?;
    let (bytecode_read_raf_params, booleanity_params) = v.verify_stage6a_from_reader(&mut r)?;
    v.verify_stage6b_from_reader(&mut r, bytecode_read_raf_params, booleanity_params)?;
    v.verify_stage7_from_reader(&mut r)?;
    if v.has_recursion {
        v.verify_stage8_with_recursion_from_reader(&mut r)?;
    } else {
        v.verify_stage8_from_reader(&mut r)?;
    }

    if r.remaining() != 0 {
        return Err(anyhow::anyhow!(
            "proof record has trailing bytes: {}",
            r.remaining()
        ));
    }

    Ok(())
}

/// Convenience wrapper for the default RV64IMAC stack (Fr + Dory + Blake2bTranscript).
pub fn rv64imac_verify_from_proof_bytes(
    preprocessing: &crate::zkvm::verifier::JoltVerifierPreprocessing<
        ark_bn254::Fr,
        crate::poly::commitment::dory::DoryCommitmentScheme,
    >,
    program_io: tracer::JoltDevice,
    trusted_advice_commitment: Option<
        <crate::poly::commitment::dory::DoryCommitmentScheme as CommitmentScheme>::Commitment,
    >,
    proof_bytes: &[u8],
) -> Result<(), anyhow::Error> {
    verify_from_proof_bytes::<
        ark_bn254::Fr,
        crate::poly::commitment::dory::DoryCommitmentScheme,
        crate::transcripts::Blake2bTranscript,
    >(
        preprocessing,
        program_io,
        trusted_advice_commitment,
        proof_bytes,
    )
}
