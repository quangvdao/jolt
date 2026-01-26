use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use ark_bn254::Fq;
use ark_grumpkin::Projective as GrumpkinProjective;

use crate::poly::commitment::{
    commitment_scheme::{CommitmentScheme, RecursionExt},
    dory::{DoryContext, DoryGlobals},
    hyrax::{Hyrax, PedersenGenerators},
};
// Dory-specific imports for recursion mode (used at runtime when PCS is Dory)
#[allow(unused_imports)]
use crate::poly::commitment::dory::{
    derive_from_dory_ast, derive_plan_with_hints, wrappers::ArkDoryProof, wrappers::ArkGT,
    wrappers::ArkworksVerifierSetup,
};
use crate::poly::rlc_utils::{compute_joint_claim, compute_rlc_coefficients};
use crate::subprotocols::sumcheck::BatchedSumcheck;
use crate::zkvm::bytecode::chunks::total_lanes;
use crate::zkvm::claim_reductions::advice::ReductionPhase;
use crate::zkvm::claim_reductions::RegistersClaimReductionSumcheckVerifier;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::config::ProgramMode;
use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};
use crate::zkvm::program::{
    ProgramMetadata, ProgramPreprocessing, TrustedProgramCommitments, VerifierProgram,
};
#[cfg(feature = "prover")]
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::zkvm::ram::val_final::ValFinalSumcheckVerifier;
use crate::zkvm::ram::verifier_accumulate_program_image;
use crate::zkvm::recursion::MAX_RECURSION_DENSE_NUM_VARS;
use crate::zkvm::witness::all_committed_polynomials;
use crate::zkvm::Serializable;
use crate::zkvm::{
    bytecode::read_raf_checking::{
        BytecodeReadRafAddressSumcheckVerifier, BytecodeReadRafCycleSumcheckVerifier,
        BytecodeReadRafSumcheckParams,
    },
    claim_reductions::{
        AdviceClaimReductionVerifier, AdviceKind, BytecodeClaimReductionParams,
        BytecodeClaimReductionVerifier, BytecodeReductionPhase,
        HammingWeightClaimReductionVerifier, IncClaimReductionSumcheckVerifier,
        InstructionLookupsClaimReductionSumcheckVerifier, ProgramImageClaimReductionParams,
        ProgramImageClaimReductionVerifier, RamRaClaimReductionSumcheckVerifier,
    },
    fiat_shamir_preamble,
    instruction_lookups::{
        ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier,
        read_raf_checking::InstructionReadRafSumcheckVerifier,
    },
    proof_serialization::{JoltProof, RecursionPayload},
    r1cs::key::UniformSpartanKey,
    ram::{
        hamming_booleanity::HammingBooleanitySumcheckVerifier,
        output_check::OutputSumcheckVerifier, ra_virtual::RamRaVirtualSumcheckVerifier,
        raf_evaluation::RafEvaluationSumcheckVerifier as RamRafEvaluationSumcheckVerifier,
        read_write_checking::RamReadWriteCheckingVerifier,
        val_evaluation::ValEvaluationSumcheckVerifier as RamValEvaluationSumcheckVerifier,
        verifier_accumulate_advice,
    },
    recursion::verifier::RecursionVerifier,
    registers::{
        read_write_checking::RegistersReadWriteCheckingVerifier,
        val_evaluation::ValEvaluationSumcheckVerifier as RegistersValEvaluationSumcheckVerifier,
    },
    spartan::{
        instruction_input::InstructionInputSumcheckVerifier, outer::OuterRemainingSumcheckVerifier,
        product::ProductVirtualRemainderVerifier, shift::ShiftSumcheckVerifier,
        verify_stage1_uni_skip, verify_stage2_uni_skip,
    },
    ProverDebugInfo,
};
use crate::{
    field::JoltField,
    poly::opening_proof::{
        compute_advice_lagrange_factor, DoryOpeningState, OpeningAccumulator, OpeningPoint,
        SumcheckId, VerifierOpeningAccumulator,
    },
    pprof_scope,
    subprotocols::{
        booleanity::{
            BooleanityAddressSumcheckVerifier, BooleanityCycleSumcheckVerifier,
            BooleanitySumcheckParams,
        },
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::witness::CommittedPolynomial,
};
use anyhow::Context;
#[allow(unused_imports)]
use ark_ec::AffineRepr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
#[allow(unused_imports)]
use dory::backends::arkworks::{ArkG1, ArkG2, BN254};
#[allow(unused_imports)]
use dory::primitives::arithmetic::PairingCurve;
use itertools::Itertools;
use tracer::JoltDevice;

use jolt_platform::{end_cycle_tracking, start_cycle_tracking};

// Cycle-marker labels must be static strings: the tracer keys markers by the guest string pointer.
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

/// Minimal snapshot of verifier state needed to drive Stage 8 transcript replay.
///
/// Built after Stage 7, before any Stage 8 transcript mutations.
/// Used in recursion mode to replay the Fiat-Shamir transcript without
/// performing actual PCS verification.
#[derive(Clone, Debug)]
struct DoryVerifySnapshot<F: JoltField> {
    /// Unified opening point (big-endian).
    opening_point:
        crate::poly::opening_proof::OpeningPoint<{ crate::poly::opening_proof::BIG_ENDIAN }, F>,
    /// Ordered (polynomial, claim) pairs in prover-matching order.
    polynomial_claims: Vec<(CommittedPolynomial, F)>,
    /// Ordered claims for transcript replay.
    /// Order matters: must match the prover's ordering for gamma-power sampling.
    claims: Vec<F>,
}
pub struct JoltVerifier<'a, F: JoltField, PCS: RecursionExt<F>, ProofTranscript: Transcript> {
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub program_io: JoltDevice,
    pub proof: JoltProof<F, PCS, ProofTranscript>,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    pub transcript: ProofTranscript,
    pub opening_accumulator: VerifierOpeningAccumulator<F>,
    /// The advice claim reduction sumcheck effectively spans two stages (6 and 7).
    /// Cache the verifier state here between stages.
    advice_reduction_verifier_trusted: Option<AdviceClaimReductionVerifier<F>>,
    /// The advice claim reduction sumcheck effectively spans two stages (6 and 7).
    /// Cache the verifier state here between stages.
    advice_reduction_verifier_untrusted: Option<AdviceClaimReductionVerifier<F>>,
    /// The bytecode claim reduction sumcheck effectively spans two stages (6b and 7).
    /// Cache the verifier state here between stages.
    bytecode_reduction_verifier: Option<BytecodeClaimReductionVerifier<F>>,
    pub spartan_key: UniformSpartanKey<F>,
    pub one_hot_params: OneHotParams,
}

impl<
        'a,
        F: JoltField,
        PCS: CommitmentScheme<Field = F> + RecursionExt<F>,
        ProofTranscript: Transcript,
    > JoltVerifier<'a, F, PCS, ProofTranscript>
{
    pub fn new(
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        proof: JoltProof<F, PCS, ProofTranscript>,
        mut program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        _debug_info: Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) -> Result<Self, ProofVerifyError> {
        // Memory layout checks
        if program_io.memory_layout != preprocessing.shared.memory_layout {
            return Err(ProofVerifyError::MemoryLayoutMismatch);
        }
        if program_io.inputs.len() > preprocessing.shared.memory_layout.max_input_size as usize {
            return Err(ProofVerifyError::InputTooLarge);
        }
        if program_io.outputs.len() > preprocessing.shared.memory_layout.max_output_size as usize {
            return Err(ProofVerifyError::OutputTooLarge);
        }

        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let mut opening_accumulator = VerifierOpeningAccumulator::new(proof.trace_length.log_2());
        // Populate claims in the verifier accumulator
        for (key, (_, claim)) in &proof.opening_claims.0 {
            opening_accumulator
                .openings
                .insert(*key, (OpeningPoint::default(), *claim));
        }

        #[cfg(test)]
        let mut transcript = ProofTranscript::new(b"Jolt");
        #[cfg(not(test))]
        let transcript = ProofTranscript::new(b"Jolt");

        #[cfg(test)]
        {
            if let Some(debug_info) = _debug_info {
                transcript.compare_to(debug_info.transcript);
                opening_accumulator.compare_to(debug_info.opening_accumulator);
            }
        }

        let spartan_key = UniformSpartanKey::new(proof.trace_length.next_power_of_two());

        // Validate configs from the proof
        proof
            .one_hot_config
            .validate()
            .map_err(ProofVerifyError::InvalidOneHotConfig)?;

        proof
            .rw_config
            .validate(proof.trace_length.log_2(), proof.ram_K.log_2())
            .map_err(ProofVerifyError::InvalidReadWriteConfig)?;

        // Construct full params from the validated config
        let one_hot_params =
            OneHotParams::from_config(&proof.one_hot_config, proof.bytecode_K, proof.ram_K);

        if proof.program_mode == ProgramMode::Committed {
            let committed = preprocessing.program.as_committed()?;
            if committed.log_k_chunk != proof.one_hot_config.log_k_chunk {
                return Err(ProofVerifyError::InvalidBytecodeConfig(format!(
                    "bytecode log_k_chunk mismatch: commitments={}, proof={}",
                    committed.log_k_chunk, proof.one_hot_config.log_k_chunk
                )));
            }
            if committed.bytecode_len != preprocessing.shared.bytecode_size() {
                return Err(ProofVerifyError::InvalidBytecodeConfig(format!(
                    "bytecode length mismatch: commitments={}, shared={}",
                    committed.bytecode_len,
                    preprocessing.shared.bytecode_size()
                )));
            }
            let k_chunk = 1usize << (committed.log_k_chunk as usize);
            let expected_chunks = total_lanes().div_ceil(k_chunk);
            if committed.bytecode_commitments.len() != expected_chunks {
                return Err(ProofVerifyError::InvalidBytecodeConfig(format!(
                    "expected {expected_chunks} bytecode commitments, got {}",
                    committed.bytecode_commitments.len()
                )));
            }
        }

        Ok(Self {
            trusted_advice_commitment,
            program_io,
            proof,
            preprocessing,
            transcript,
            opening_accumulator,
            advice_reduction_verifier_trusted: None,
            advice_reduction_verifier_untrusted: None,
            bytecode_reduction_verifier: None,
            spartan_key,
            one_hot_params,
        })
    }

    #[tracing::instrument(skip_all)]
    pub fn verify(mut self, recursion: bool) -> Result<(), anyhow::Error> {
        let _pprof_verify = pprof_scope!("verify");

        fiat_shamir_preamble(
            &self.program_io,
            self.proof.ram_K,
            self.proof.trace_length,
            &mut self.transcript,
        );

        // Append commitments to transcript
        for commitment in &self.proof.commitments {
            self.transcript.append_serializable(commitment);
        }
        // Append untrusted advice commitment to transcript
        if let Some(ref untrusted_advice_commitment) = self.proof.untrusted_advice_commitment {
            self.transcript
                .append_serializable(untrusted_advice_commitment);
        }
        // Append trusted advice commitment to transcript
        if let Some(ref trusted_advice_commitment) = self.trusted_advice_commitment {
            self.transcript
                .append_serializable(trusted_advice_commitment);
        }
        if self.proof.program_mode == ProgramMode::Committed {
            let trusted = self.preprocessing.program.as_committed()?;
            for commitment in &trusted.bytecode_commitments {
                self.transcript.append_serializable(commitment);
            }
            self.transcript
                .append_serializable(&trusted.program_image_commitment);
        }

        self.verify_stage1()?;
        self.verify_stage2()?;
        self.verify_stage3()?;
        self.verify_stage4()?;
        self.verify_stage5()?;
        let (bytecode_read_raf_params, booleanity_params) = self.verify_stage6a()?;
        self.verify_stage6b(bytecode_read_raf_params, booleanity_params)?;
        self.verify_stage7()?;
        if recursion {
            // Recursion mode: require payload and skip native Stage 8 verification.
            if self.proof.recursion.is_none() {
                return Err(anyhow::anyhow!(
                    "recursion mode requested but proof has no recursion payload"
                ));
            }
            self.verify_stage8_with_recursion()?;
        } else {
            // Normal mode: require no recursion payload and run native Stage 8 PCS verification.
            if self.proof.recursion.is_some() {
                return Err(anyhow::anyhow!(
                    "non-recursion mode requested but proof contains a recursion payload"
                ));
            }
            self.verify_stage8()?;
        }

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "verify_stage1")]
    fn verify_stage1(&mut self) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE1);
        let uni_skip_params = verify_stage1_uni_skip(
            &self.proof.stage1_uni_skip_first_round_proof,
            &self.spartan_key,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 1 univariate skip first round")?;

        let spartan_outer_remaining = OuterRemainingSumcheckVerifier::new(
            self.spartan_key,
            self.proof.trace_length,
            uni_skip_params,
            &self.opening_accumulator,
        );

        let _r_stage1 = BatchedSumcheck::verify(
            &self.proof.stage1_sumcheck_proof,
            vec![&spartan_outer_remaining],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 1")?;

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "verify_stage2")]
    fn verify_stage2(&mut self) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE2);
        let uni_skip_params = verify_stage2_uni_skip(
            &self.proof.stage2_uni_skip_first_round_proof,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 2 univariate skip first round")?;

        let spartan_product_virtual_remainder = ProductVirtualRemainderVerifier::new(
            self.proof.trace_length,
            uni_skip_params,
            &self.opening_accumulator,
        );
        let ram_raf_evaluation = RamRafEvaluationSumcheckVerifier::new(
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &self.opening_accumulator,
        );
        let ram_read_write_checking = RamReadWriteCheckingVerifier::new(
            &self.opening_accumulator,
            &mut self.transcript,
            &self.one_hot_params,
            self.proof.trace_length,
            &self.proof.rw_config,
        );
        let ram_output_check =
            OutputSumcheckVerifier::new(self.proof.ram_K, &self.program_io, &mut self.transcript);
        let instruction_claim_reduction = InstructionLookupsClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let _r_stage2 = BatchedSumcheck::verify(
            &self.proof.stage2_sumcheck_proof,
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

    #[tracing::instrument(skip_all, name = "verify_stage3")]
    fn verify_stage3(&mut self) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE3);
        let spartan_shift = ShiftSumcheckVerifier::new(
            self.proof.trace_length.log_2(),
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let spartan_instruction_input =
            InstructionInputSumcheckVerifier::new(&self.opening_accumulator, &mut self.transcript);
        let spartan_registers_claim_reduction = RegistersClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let _r_stage3 = BatchedSumcheck::verify(
            &self.proof.stage3_sumcheck_proof,
            vec![
                &spartan_shift as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &spartan_instruction_input,
                &spartan_registers_claim_reduction,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 3")?;

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "verify_stage4")]
    fn verify_stage4(&mut self) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE4);
        verifier_accumulate_advice::<F>(
            self.proof.ram_K,
            &self.program_io,
            self.proof.untrusted_advice_commitment.is_some(),
            self.trusted_advice_commitment.is_some(),
            &mut self.opening_accumulator,
            &mut self.transcript,
            self.proof
                .rw_config
                .needs_single_advice_opening(self.proof.trace_length.log_2()),
        );
        if self.proof.program_mode == ProgramMode::Committed {
            verifier_accumulate_program_image::<F>(
                self.proof.ram_K,
                &self.program_io,
                &mut self.opening_accumulator,
                &mut self.transcript,
                self.proof
                    .rw_config
                    .needs_single_advice_opening(self.proof.trace_length.log_2()),
            );
        }
        let registers_read_write_checking = RegistersReadWriteCheckingVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
            &self.proof.rw_config,
        );
        // In Full mode, get the program image words from the preprocessing
        let program_image_words = self.preprocessing.program.program_image_words();
        let ram_val_evaluation = RamValEvaluationSumcheckVerifier::new(
            &self.preprocessing.shared.program_meta,
            program_image_words,
            &self.program_io,
            self.proof.trace_length,
            self.proof.ram_K,
            self.proof.program_mode,
            &self.opening_accumulator,
        );
        let ram_val_final = ValFinalSumcheckVerifier::new(
            &self.preprocessing.shared.program_meta,
            program_image_words,
            &self.program_io,
            self.proof.trace_length,
            self.proof.ram_K,
            self.proof.program_mode,
            &self.opening_accumulator,
            &self.proof.rw_config,
        );

        let _r_stage4 = BatchedSumcheck::verify(
            &self.proof.stage4_sumcheck_proof,
            vec![
                &registers_read_write_checking as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &ram_val_evaluation,
                &ram_val_final,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 4")?;

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "verify_stage5")]
    fn verify_stage5(&mut self) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE5);
        let n_cycle_vars = self.proof.trace_length.log_2();
        let registers_val_evaluation =
            RegistersValEvaluationSumcheckVerifier::new(&self.opening_accumulator);
        let ram_ra_reduction = RamRaClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_read_raf = InstructionReadRafSumcheckVerifier::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let _r_stage5 = BatchedSumcheck::verify(
            &self.proof.stage5_sumcheck_proof,
            vec![
                &registers_val_evaluation as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &ram_ra_reduction,
                &lookups_read_raf,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 5")?;

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "verify_stage6a")]
    fn verify_stage6a(
        &mut self,
    ) -> Result<
        (
            BytecodeReadRafSumcheckParams<F>,
            BooleanitySumcheckParams<F>,
        ),
        anyhow::Error,
    > {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE6A);
        let n_cycle_vars = self.proof.trace_length.log_2();
        let program_preprocessing = match self.proof.program_mode {
            ProgramMode::Committed => {
                // Ensure we have committed program commitments for committed mode.
                let _ = self.preprocessing.program.as_committed()?;
                None
            }
            ProgramMode::Full => self.preprocessing.program.full().map(|p| p.as_ref()),
        };
        let bytecode_read_raf = BytecodeReadRafAddressSumcheckVerifier::new(
            program_preprocessing,
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
            self.proof.program_mode,
        )?;
        let booleanity_params = BooleanitySumcheckParams::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let booleanity = BooleanityAddressSumcheckVerifier::new(booleanity_params);

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> =
            vec![&bytecode_read_raf, &booleanity];

        let _r_stage6a = BatchedSumcheck::verify(
            &self.proof.stage6a_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 6a")?;
        Ok((bytecode_read_raf.into_params(), booleanity.into_params()))
    }

    #[tracing::instrument(skip_all, name = "verify_stage6b")]
    fn verify_stage6b(
        &mut self,
        bytecode_read_raf_params: BytecodeReadRafSumcheckParams<F>,
        booleanity_params: BooleanitySumcheckParams<F>,
    ) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE6B);
        // Initialize Stage 6b cycle verifiers from scratch (Option B).
        let booleanity = BooleanityCycleSumcheckVerifier::new(booleanity_params);
        let ram_hamming_booleanity =
            HammingBooleanitySumcheckVerifier::new(&self.opening_accumulator);
        let ram_ra_virtual = RamRaVirtualSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_ra_virtual = LookupsRaSumcheckVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let inc_reduction = IncClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // Bytecode claim reduction (Phase 1 in Stage 6b): consumes Val_s(r_bc) from Stage 6a and
        // caches an intermediate claim for Stage 7.
        //
        // IMPORTANT: This must be sampled *after* other Stage 6b params (e.g. lookup/inc gammas),
        // to match the prover's transcript order.
        if self.proof.program_mode == ProgramMode::Committed {
            let bytecode_reduction_params = BytecodeClaimReductionParams::new(
                &bytecode_read_raf_params,
                &self.opening_accumulator,
                &mut self.transcript,
            );
            self.bytecode_reduction_verifier = Some(BytecodeClaimReductionVerifier::new(
                bytecode_reduction_params,
            ));
        } else {
            // Legacy mode: do not run the bytecode claim reduction.
            self.bytecode_reduction_verifier = None;
        }

        // Advice claim reduction (Phase 1 in Stage 6b): trusted and untrusted are separate instances.
        if self.trusted_advice_commitment.is_some() {
            self.advice_reduction_verifier_trusted = Some(AdviceClaimReductionVerifier::new(
                AdviceKind::Trusted,
                &self.program_io.memory_layout,
                self.proof.trace_length,
                &self.opening_accumulator,
                &mut self.transcript,
                self.proof
                    .rw_config
                    .needs_single_advice_opening(self.proof.trace_length.log_2()),
            ));
        }
        if self.proof.untrusted_advice_commitment.is_some() {
            self.advice_reduction_verifier_untrusted = Some(AdviceClaimReductionVerifier::new(
                AdviceKind::Untrusted,
                &self.program_io.memory_layout,
                self.proof.trace_length,
                &self.opening_accumulator,
                &mut self.transcript,
                self.proof
                    .rw_config
                    .needs_single_advice_opening(self.proof.trace_length.log_2()),
            ));
        }

        // Program-image claim reduction (Stage 6b): binds staged Stage 4 scalar program-image claims
        // to the trusted commitment, caching an opening of ProgramImageInit.
        let program_image_reduction = if self.proof.program_mode == ProgramMode::Committed {
            let trusted = self
                .preprocessing
                .program
                .as_committed()
                .expect("program commitments missing in committed mode");
            let padded_len_words = trusted.program_image_num_words;
            let log_t = self.proof.trace_length.log_2();
            let m = padded_len_words.log_2();
            if m > log_t {
                return Err(ProofVerifyError::InvalidBytecodeConfig(format!(
                    "program-image claim reduction requires m=log2(padded_len_words) <= log_T (got m={m}, log_T={log_t})"
                ))
                .into());
            }
            let params = ProgramImageClaimReductionParams::new(
                &self.program_io,
                self.preprocessing.shared.min_bytecode_address(),
                padded_len_words,
                self.proof.ram_K,
                self.proof.trace_length,
                &self.proof.rw_config,
                &self.opening_accumulator,
                &mut self.transcript,
            );
            Some(ProgramImageClaimReductionVerifier { params })
        } else {
            None
        };

        let bytecode_read_raf = BytecodeReadRafCycleSumcheckVerifier::new(bytecode_read_raf_params);

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = vec![
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

        let _r_stage6b = BatchedSumcheck::verify(
            &self.proof.stage6b_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 6b")?;

        Ok(())
    }

    /// Stage 7: HammingWeight claim reduction verification.
    #[tracing::instrument(skip_all, name = "verify_stage7")]
    fn verify_stage7(&mut self) -> Result<(), anyhow::Error> {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE7);
        // Create verifier for HammingWeightClaimReduction
        // (r_cycle and r_addr_bool are extracted from Booleanity opening internally)
        let hw_verifier = HammingWeightClaimReductionVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> =
            vec![&hw_verifier];

        if let Some(bytecode_reduction_verifier) = self.bytecode_reduction_verifier.as_mut() {
            bytecode_reduction_verifier.params.borrow_mut().phase =
                BytecodeReductionPhase::LaneVariables;
            instances.push(bytecode_reduction_verifier);
        }
        if let Some(advice_reduction_verifier_trusted) =
            self.advice_reduction_verifier_trusted.as_mut()
        {
            let mut params = advice_reduction_verifier_trusted.params.borrow_mut();
            if params.num_address_phase_rounds() > 0 {
                // Transition phase
                params.phase = ReductionPhase::AddressVariables;
                instances.push(advice_reduction_verifier_trusted);
            }
        }
        if let Some(advice_reduction_verifier_untrusted) =
            self.advice_reduction_verifier_untrusted.as_mut()
        {
            let mut params = advice_reduction_verifier_untrusted.params.borrow_mut();
            if params.num_address_phase_rounds() > 0 {
                // Transition phase
                params.phase = ReductionPhase::AddressVariables;
                instances.push(advice_reduction_verifier_untrusted);
            }
        }

        let _r_address_stage7 = BatchedSumcheck::verify(
            &self.proof.stage7_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 7")?;

        Ok(())
    }

    fn build_dory_verify_snapshot(&self) -> Result<DoryVerifySnapshot<F>, anyhow::Error> {
        let _span = tracing::info_span!("stage8_build_dory_verify_snapshot").entered();

        // Get the unified opening point from HammingWeightClaimReduction.
        // This contains (r_address_stage7 || r_cycle_stage6) in big-endian.
        let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        let r_address_stage7 = &opening_point.r[..log_k_chunk];

        // 1) Collect all (polynomial, claim) pairs in prover-matching order.
        let mut polynomial_claims = Vec::new();

        // Dense polynomials: RamInc and RdInc (from IncClaimReduction in Stage 6)
        let (_, ram_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
        );
        let (_, rd_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
        );

        // Apply Lagrange factor for dense polys.
        // Note: r_address is in big-endian, Lagrange factor uses ∏(1 - r_i).
        let lagrange_factor: F = r_address_stage7.iter().map(|r| F::one() - *r).product();
        polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
        polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));

        // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
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

        // Advice polynomials (if present): fold into the Stage 8 batch via a Lagrange embedding
        // so the verifier samples the same gamma powers as the prover.
        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReduction)
        {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
            polynomial_claims.push((
                CommittedPolynomial::TrustedAdvice,
                advice_claim * lagrange_factor,
            ));
        }
        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_advice_opening(AdviceKind::Untrusted, SumcheckId::AdviceClaimReduction)
        {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
            polynomial_claims.push((
                CommittedPolynomial::UntrustedAdvice,
                advice_claim * lagrange_factor,
            ));
        }

        // Bytecode chunk polynomials: committed in Bytecode context and embedded into the
        // main opening point by fixing the extra cycle variables to 0.
        if self.proof.program_mode == ProgramMode::Committed {
            let (bytecode_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeChunk(0),
                SumcheckId::BytecodeClaimReduction,
            );
            #[cfg(test)]
            {
                let log_t = opening_point.r.len() - log_k_chunk;
                let log_k = bytecode_point.r.len() - log_k_chunk;
                if log_k == log_t {
                    assert_eq!(
                        bytecode_point.r, opening_point.r,
                        "BytecodeChunk opening point must equal unified opening point when log_K == log_T"
                    );
                }
            }
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &bytecode_point.r);

            let num_chunks = total_lanes().div_ceil(self.one_hot_params.k_chunk);
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

        // Program-image polynomial: opened by ProgramImageClaimReduction in Stage 6b.
        // Embed into the top-left block of the main matrix (same trick as advice).
        if self.proof.program_mode == ProgramMode::Committed {
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

        Ok(DoryVerifySnapshot {
            opening_point,
            polynomial_claims,
            claims,
        })
    }

    /// Stage 8: Dory batch opening verification.
    fn verify_stage8(&mut self) -> Result<(), anyhow::Error> {
        // Initialize DoryGlobals with the layout from the proof.
        // In committed mode, we must also match the Main-context sigma used to derive trusted
        // bytecode commitments, otherwise Stage 8 batching will be inconsistent.
        let _guard = if self.proof.program_mode == ProgramMode::Committed {
            let committed = self.preprocessing.program.as_committed()?;
            DoryGlobals::initialize_main_context_with_num_columns(
                1 << self.one_hot_params.log_k_chunk,
                self.proof.trace_length.next_power_of_two(),
                committed.bytecode_num_columns,
                Some(self.proof.dory_layout),
            )
        } else {
            DoryGlobals::initialize_context(
                1 << self.one_hot_params.log_k_chunk,
                self.proof.trace_length.next_power_of_two(),
                DoryContext::Main,
                Some(self.proof.dory_layout),
            )
        };

        // Get the unified opening point from HammingWeightClaimReduction
        // This contains (r_address_stage7 || r_cycle_stage6) in big-endian
        let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        let r_address_stage7 = &opening_point.r[..log_k_chunk];

        // 1. Collect all (polynomial, claim) pairs
        let mut polynomial_claims = Vec::new();

        // Dense polynomials: RamInc and RdInc (from IncClaimReduction in Stage 6)
        let (_, ram_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
        );
        let (_, rd_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
        );

        // Apply Lagrange factor for dense polys
        // Note: r_address is in big-endian, Lagrange factor uses ∏(1 - r_i)
        let lagrange_factor: F = r_address_stage7.iter().map(|r| F::one() - *r).product();

        polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
        polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));

        // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
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

        // Advice polynomials: TrustedAdvice and UntrustedAdvice (from AdviceClaimReduction in Stage 6)
        // These are committed with smaller dimensions, so we apply Lagrange factors to embed
        // them in the top-left block of the main Dory matrix.
        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReduction)
        {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
            polynomial_claims.push((
                CommittedPolynomial::TrustedAdvice,
                advice_claim * lagrange_factor,
            ));
        }

        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_advice_opening(AdviceKind::Untrusted, SumcheckId::AdviceClaimReduction)
        {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
            polynomial_claims.push((
                CommittedPolynomial::UntrustedAdvice,
                advice_claim * lagrange_factor,
            ));
        }

        // Bytecode chunk polynomials: committed in Bytecode context and embedded into the
        // main opening point by fixing the extra cycle variables to 0.
        if self.proof.program_mode == ProgramMode::Committed {
            let (bytecode_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeChunk(0),
                SumcheckId::BytecodeClaimReduction,
            );
            let log_t = opening_point.r.len() - log_k_chunk;
            let log_k = bytecode_point.r.len() - log_k_chunk;
            if log_k > log_t {
                return Err(ProofVerifyError::InvalidBytecodeConfig(format!(
                    "bytecode folding requires log_T >= log_K (got log_T={log_t}, log_K={log_k})"
                ))
                .into());
            }
            #[cfg(test)]
            {
                if log_k == log_t {
                    assert_eq!(
                        bytecode_point.r, opening_point.r,
                        "BytecodeChunk opening point must equal unified opening point when log_K == log_T"
                    );
                } else {
                    let (r_lane_main, r_cycle_main) = opening_point.split_at(log_k_chunk);
                    let (r_lane_bc, r_cycle_bc) = bytecode_point.split_at(log_k_chunk);
                    debug_assert_eq!(r_lane_main.r, r_lane_bc.r);
                    debug_assert_eq!(&r_cycle_main.r[(log_t - log_k)..], r_cycle_bc.r.as_slice());
                }
            }
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &bytecode_point.r);

            let num_chunks = total_lanes().div_ceil(self.one_hot_params.k_chunk);
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

        // Program-image polynomial: opened by ProgramImageClaimReduction in Stage 6b.
        // Embed into the top-left block of the main matrix (same trick as advice).
        if self.proof.program_mode == ProgramMode::Committed {
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

        // 2. Sample gamma and compute powers for RLC
        let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
        self.transcript.append_scalars(&claims);
        let gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());

        // Build state for computing joint commitment/claim
        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers: gamma_powers.clone(),
            polynomial_claims,
        };

        // Build commitments map
        let mut commitments_map = HashMap::new();
        for (polynomial, commitment) in all_committed_polynomials(&self.one_hot_params)
            .into_iter()
            .zip_eq(&self.proof.commitments)
        {
            commitments_map.insert(polynomial, commitment.clone());
        }

        // Add advice commitments if they're part of the batch
        if let Some(ref commitment) = self.trusted_advice_commitment {
            if state
                .polynomial_claims
                .iter()
                .any(|(p, _)| *p == CommittedPolynomial::TrustedAdvice)
            {
                commitments_map.insert(CommittedPolynomial::TrustedAdvice, commitment.clone());
            }
        }
        if let Some(ref commitment) = self.proof.untrusted_advice_commitment {
            if state
                .polynomial_claims
                .iter()
                .any(|(p, _)| *p == CommittedPolynomial::UntrustedAdvice)
            {
                commitments_map.insert(CommittedPolynomial::UntrustedAdvice, commitment.clone());
            }
        }

        if self.proof.program_mode == ProgramMode::Committed {
            let committed = self.preprocessing.program.as_committed()?;
            for (idx, commitment) in committed.bytecode_commitments.iter().enumerate() {
                commitments_map
                    .entry(CommittedPolynomial::BytecodeChunk(idx))
                    .or_insert_with(|| commitment.clone());
            }

            // Add trusted program-image commitment if it's part of the batch.
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

        // Compute joint commitment: Σ γ_i · C_i
        let joint_commitment = self.compute_joint_commitment(&mut commitments_map, &state);

        // Compute joint claim: Σ γ_i · claim_i
        let joint_claim: F = gamma_powers
            .iter()
            .zip(claims.iter())
            .map(|(gamma, claim)| *gamma * claim)
            .sum();

        // Verify joint opening
        PCS::verify(
            &self.proof.stage8_opening_proof,
            &self.preprocessing.generators,
            &mut self.transcript,
            &opening_point.r,
            &joint_claim,
            &joint_commitment,
        )
        .context("Stage 8 (joint)")?;

        Ok(())
    }

    /// Compute joint commitment for the batch opening.
    fn compute_joint_commitment(
        &self,
        commitment_map: &mut HashMap<CommittedPolynomial, PCS::Commitment>,
        state: &DoryOpeningState<F>,
    ) -> PCS::Commitment {
        // Accumulate gamma coefficients per polynomial
        let mut rlc_map = HashMap::new();
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
    /// Verify Stage 8 with recursion proof
    #[tracing::instrument(skip_all, name = "verify_stage8_with_recursion")]
    fn verify_stage8_with_recursion(&mut self) -> Result<(), anyhow::Error>
    where
        PCS: RecursionExt<F>,
        PCS::Ast: 'static,
        PCS::Commitment: 'static,
        PCS::Proof: 'static,
        PCS::VerifierSetup: 'static,
    {
        let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8);
        let payload: &RecursionPayload<F, PCS, ProofTranscript> =
            self.proof.recursion.as_ref().ok_or_else(|| {
                anyhow::anyhow!("recursion payload is required in recursion mode")
            })?;

        // 1) Reconstruct the Stage 8 batching state + build the symbolic AST at the pre-Stage8
        // transcript state. Then replay the PCS transcript interactions (no native PCS checks).
        let dory_snap = self.build_dory_verify_snapshot()?;
        let (
            _gamma_powers,
            _joint_claim,
            joint_commitment,
            combine_coeffs,
            combine_commitments,
            ast,
        ) = {
            let _span = tracing::info_span!("stage8_reconstruct_ast_and_batching").entered();
            let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8_DORY_PCS);

            // Must match prover ordering: append claims then sample gamma powers.
            self.transcript.append_scalars(&dory_snap.claims);
            let gamma_powers: Vec<F> = self
                .transcript
                .challenge_scalar_powers(dory_snap.claims.len());

            // Build state for computing joint commitment/claim.
            let state = DoryOpeningState {
                opening_point: dory_snap.opening_point.r.clone(),
                gamma_powers: gamma_powers.clone(),
                polynomial_claims: dory_snap.polynomial_claims.clone(),
            };

            // Build commitments map (must match Stage 8 native verifier path).
            let mut commitments_map = HashMap::new();
            for (polynomial, commitment) in all_committed_polynomials(&self.one_hot_params)
                .into_iter()
                .zip_eq(&self.proof.commitments)
            {
                commitments_map.insert(polynomial, commitment.clone());
            }

            // Add advice commitments if they're part of the batch.
            if let Some(ref commitment) = self.trusted_advice_commitment {
                if state
                    .polynomial_claims
                    .iter()
                    .any(|(p, _)| *p == CommittedPolynomial::TrustedAdvice)
                {
                    commitments_map.insert(CommittedPolynomial::TrustedAdvice, commitment.clone());
                }
            }
            if let Some(ref commitment) = self.proof.untrusted_advice_commitment {
                if state
                    .polynomial_claims
                    .iter()
                    .any(|(p, _)| *p == CommittedPolynomial::UntrustedAdvice)
                {
                    commitments_map
                        .insert(CommittedPolynomial::UntrustedAdvice, commitment.clone());
                }
            }

            if self.proof.program_mode == ProgramMode::Committed {
                let committed = self.preprocessing.program.as_committed()?;
                for (idx, commitment) in committed.bytecode_commitments.iter().enumerate() {
                    commitments_map
                        .entry(CommittedPolynomial::BytecodeChunk(idx))
                        .or_insert_with(|| commitment.clone());
                }

                // Add trusted program-image commitment if it's part of the batch.
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

            // Deterministic combine plan (must match prover's BTreeMap iteration).
            let joint_claim: F = compute_joint_claim(&gamma_powers, &dory_snap.claims);
            let rlc_map = compute_rlc_coefficients(&gamma_powers, state.polynomial_claims.clone());
            let (combine_coeffs, combine_commitments): (Vec<F>, Vec<PCS::Commitment>) = rlc_map
                .into_iter()
                .map(|(poly, coeff)| {
                    (
                        coeff,
                        commitments_map
                            .get(&poly)
                            .expect("missing commitment for polynomial in batch")
                            .clone(),
                    )
                })
                .unzip();
            // Recursion mode: prefer the prover-supplied Stage-8 combine hint to avoid expensive
            // `PCS::combine_commitments` on the verifier.
            //
            // NOTE: This is not sound until wiring/boundary constraints bind the hinted value to
            // the Dory AST (tracked separately).
            let joint_commitment = match payload.stage8_combine_hint.as_ref() {
                Some(hint_fq12) => PCS::combine_with_hint_fq12(hint_fq12),
                None => PCS::combine_commitments(&combine_commitments, &combine_coeffs),
            };

            // Build AST on a transcript clone at the pre-Stage8-proof state.
            let mut ast_transcript = self.transcript.clone();
            let ast = PCS::build_symbolic_ast(
                &self.proof.stage8_opening_proof,
                &self.preprocessing.generators,
                &mut ast_transcript,
                &state.opening_point,
                &joint_claim,
                &joint_commitment,
            )
            .map_err(|e| anyhow::anyhow!("Stage 8 symbolic AST build failed: {e:?}"))?;

            // Now replay the PCS opening proof's transcript interactions on the real transcript.
            PCS::replay_opening_proof_transcript(
                &self.proof.stage8_opening_proof,
                &mut self.transcript,
            )
            .map_err(|e| anyhow::anyhow!("Stage 8 PCS FS replay failed: {e:?}"))?;

            (
                gamma_powers,
                joint_claim,
                joint_commitment,
                combine_coeffs,
                combine_commitments,
                ast,
            )
        };

        // 2) Derive RecursionVerifierInput (constraint plan + public inputs) from the AST.
        //
        // NOTE: We treat pairing boundary (multi-pairing inputs) as prover-supplied hints for now.
        // Binding those values to the Dory AST requires wiring/boundary constraints which are not
        // yet implemented.
        let derived_plan = {
            let _span = tracing::info_span!("stage8_derive_recursion_instance_plan").entered();
            use std::any::Any;

            // Convert combine coeffs to BN254 Fr for bit extraction.
            let combine_coeffs_fr: Vec<ark_bn254::Fr> = combine_coeffs
                .iter()
                .map(|c| {
                    (c as &dyn Any)
                        .downcast_ref::<ark_bn254::Fr>()
                        .copied()
                        .ok_or_else(|| anyhow::anyhow!("recursion mode requires F = ark_bn254::Fr"))
                })
                .collect::<Result<_, _>>()?;

            // Dory-specific plan derivation (recursion mode currently only supported for Dory).
            let ast_graph = (&ast as &dyn Any)
                .downcast_ref::<dory::recursion::ast::AstGraph<dory::backends::arkworks::BN254>>()
                .ok_or_else(|| anyhow::anyhow!("recursion mode requires Dory AST"))?;
            let dory_proof = (&self.proof.stage8_opening_proof as &dyn Any)
                .downcast_ref::<ArkDoryProof>()
                .ok_or_else(|| anyhow::anyhow!("recursion mode requires Dory opening proof"))?;
            let dory_setup = (&self.preprocessing.generators as &dyn Any)
                .downcast_ref::<ArkworksVerifierSetup>()
                .ok_or_else(|| anyhow::anyhow!("recursion mode requires Dory verifier setup"))?;
            let joint_commitment_dory = (&joint_commitment as &dyn Any)
                .downcast_ref::<ArkGT>()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("recursion mode requires Dory commitment type"))?;
            let combine_commitments_dory: Vec<ArkGT> = combine_commitments
                .iter()
                .map(|c| {
                    (c as &dyn Any)
                        .downcast_ref::<ArkGT>()
                        .cloned()
                        .ok_or_else(|| {
                            anyhow::anyhow!("recursion mode requires Dory commitment type")
                        })
                })
                .collect::<Result<_, _>>()?;

            derive_plan_with_hints(
                ast_graph,
                dory_proof,
                dory_setup,
                joint_commitment_dory,
                &combine_commitments_dory,
                &combine_coeffs_fr,
                &payload.non_input_base_hints,
                payload.pairing_boundary.clone(),
                joint_commitment_dory.0,
            )
            .map_err(|e| anyhow::anyhow!("AST->instance-plan derivation failed: {e:?}"))?
        };

        if derived_plan.dense_num_vars > MAX_RECURSION_DENSE_NUM_VARS {
            return Err(anyhow::anyhow!(
                "dense_num_vars {} exceeds max {}",
                derived_plan.dense_num_vars,
                MAX_RECURSION_DENSE_NUM_VARS
            ));
        }

        // 3) Verify recursion proof.
        let recursion_proof = &payload.recursion_proof;
        let recursion_verifier = {
            let _span = tracing::info_span!("stage8_create_recursion_verifier").entered();
            RecursionVerifier::<Fq>::new(derived_plan.verifier_input)
        };

        type HyraxPCS = Hyrax<1, GrumpkinProjective>;
        let hyrax_verifier_setup = &self.preprocessing.hyrax_recursion_setup;

        let verification_result = {
            let _span = tracing::info_span!("stage8_recursion_verifier_verify").entered();
            let _cycle = CycleMarkerGuard::new(CYCLE_VERIFY_STAGE8_RECURSION);
            recursion_verifier
                .verify::<ProofTranscript, HyraxPCS>(
                    recursion_proof,
                    &mut self.transcript,
                    &recursion_proof.dense_commitment,
                    hyrax_verifier_setup,
                )
                .map_err(|e| anyhow::anyhow!("Recursion verification failed: {e:?}"))?
        };
        if !verification_result {
            return Err(anyhow::anyhow!("Recursion proof verification failed"));
        }

        // 4) External pairing check (pairing boundary treated as prover hint for now).
        let got = &payload.pairing_boundary;

        // Minimal binding: recompute multi-pairing and check equality.
        let lhs = {
            let g1s = [
                ArkG1(got.p1_g1.into_group()),
                ArkG1(got.p2_g1.into_group()),
                ArkG1(got.p3_g1.into_group()),
            ];
            let g2s = [
                ArkG2(got.p1_g2.into_group()),
                ArkG2(got.p2_g2.into_group()),
                ArkG2(got.p3_g2.into_group()),
            ];
            BN254::multi_pair(&g1s, &g2s)
        };
        if lhs.0 != got.rhs {
            return Err(anyhow::anyhow!("external pairing check failed"));
        }

        Ok(())
    }
}

/// Shared preprocessing between prover and verifier.
///
/// Contains O(1) metadata about the program. Does NOT contain the full program data.
/// - Full program data is in `JoltProverPreprocessing.program`.
/// - Verifier program (Full or Committed) is in `JoltVerifierPreprocessing.program`.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltSharedPreprocessing {
    /// Program metadata (bytecode size, program image info).
    pub program_meta: ProgramMetadata,
    pub memory_layout: MemoryLayout,
    pub max_padded_trace_length: usize,
}
impl JoltSharedPreprocessing {
    /// Create shared preprocessing from program metadata.
    ///
    /// # Arguments
    /// - `program_meta`: Program metadata (from `ProgramPreprocessing::meta()`)
    /// - `memory_layout`: Memory layout configuration
    /// - `max_padded_trace_length`: Maximum trace length for generator sizing
    #[tracing::instrument(skip_all, name = "JoltSharedPreprocessing::new")]
    pub fn new(
        program_meta: ProgramMetadata,
        memory_layout: MemoryLayout,
        max_padded_trace_length: usize,
    ) -> JoltSharedPreprocessing {
        Self {
            program_meta,
            memory_layout,
            max_padded_trace_length,
        }
    }

    /// Bytecode size (power-of-2 padded).
    /// Legacy accessor - use `program_meta.bytecode_len` directly.
    pub fn bytecode_size(&self) -> usize {
        self.program_meta.bytecode_len
    }

    /// Minimum bytecode address.
    /// Legacy accessor - use `program_meta.min_bytecode_address` directly.
    pub fn min_bytecode_address(&self) -> u64 {
        self.program_meta.min_bytecode_address
    }

    /// Program image length (unpadded words).
    /// Legacy accessor - use `program_meta.program_image_len_words` directly.
    pub fn program_image_len_words(&self) -> usize {
        self.program_meta.program_image_len_words
    }
}

impl GuestSerialize for JoltSharedPreprocessing {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.program_meta.guest_serialize(w)?;
        self.memory_layout.guest_serialize(w)?;
        self.max_padded_trace_length.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for JoltSharedPreprocessing {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            program_meta: ProgramMetadata::guest_deserialize(r)?,
            memory_layout: MemoryLayout::guest_deserialize(r)?,
            max_padded_trace_length: usize::guest_deserialize(r)?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub shared: JoltSharedPreprocessing,
    /// Cached Hyrax setup for recursion verification
    pub hyrax_recursion_setup: PedersenGenerators<GrumpkinProjective>,
    /// Program information for verification.
    ///
    /// In Full mode: contains full program preprocessing (bytecode + program image).
    /// In Committed mode: contains only commitments (succinct).
    pub program: VerifierProgram<PCS>,
}

impl<F, PCS> GuestSerialize for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    PCS::VerifierSetup: GuestSerialize,
    PedersenGenerators<GrumpkinProjective>: GuestSerialize,
    VerifierProgram<PCS>: GuestSerialize,
{
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.generators.guest_serialize(w)?;
        self.shared.guest_serialize(w)?;
        self.hyrax_recursion_setup.guest_serialize(w)?;
        self.program.guest_serialize(w)?;
        Ok(())
    }
}

impl<F, PCS> GuestDeserialize for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    PCS::VerifierSetup: GuestDeserialize,
    PedersenGenerators<GrumpkinProjective>: GuestDeserialize,
    VerifierProgram<PCS>: GuestDeserialize,
{
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            generators: PCS::VerifierSetup::guest_deserialize(r)?,
            shared: JoltSharedPreprocessing::guest_deserialize(r)?,
            hyrax_recursion_setup: PedersenGenerators::<GrumpkinProjective>::guest_deserialize(r)?,
            program: VerifierProgram::<PCS>::guest_deserialize(r)?,
        })
    }
}

impl<F, PCS> CanonicalSerialize for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.generators.serialize_with_mode(&mut writer, compress)?;
        self.shared.serialize_with_mode(&mut writer, compress)?;
        self.hyrax_recursion_setup
            .serialize_with_mode(&mut writer, compress)?;
        self.program.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.generators.serialized_size(compress)
            + self.shared.serialized_size(compress)
            + self.hyrax_recursion_setup.serialized_size(compress)
            + self.program.serialized_size(compress)
    }
}

impl<F, PCS> ark_serialize::Valid for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.generators.check()?;
        self.shared.check()?;
        self.hyrax_recursion_setup.check()?;
        self.program.check()
    }
}

impl<F, PCS> CanonicalDeserialize for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let generators =
            PCS::VerifierSetup::deserialize_with_mode(&mut reader, compress, validate)?;
        let shared =
            JoltSharedPreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
        let hyrax_recursion_setup =
            PedersenGenerators::deserialize_with_mode(&mut reader, compress, validate)?;
        let program = VerifierProgram::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self {
            generators,
            shared,
            hyrax_recursion_setup,
            program,
        })
    }
}

impl<F, PCS> Serializable for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
}

impl<F, PCS> JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> JoltVerifierPreprocessing<F, PCS> {
    /// Create verifier preprocessing in Full mode (verifier has full program).
    #[tracing::instrument(skip_all, name = "JoltVerifierPreprocessing::new_full")]
    pub fn new_full(
        shared: JoltSharedPreprocessing,
        generators: PCS::VerifierSetup,
        program: Arc<ProgramPreprocessing>,
    ) -> JoltVerifierPreprocessing<F, PCS> {
        // Precompute Hyrax generators for recursion verification
        type HyraxPCS = Hyrax<1, GrumpkinProjective>;
        let hyrax_prover_setup =
            <HyraxPCS as CommitmentScheme>::setup_prover(MAX_RECURSION_DENSE_NUM_VARS);
        let hyrax_recursion_setup =
            <HyraxPCS as CommitmentScheme>::setup_verifier(&hyrax_prover_setup);

        Self {
            generators,
            shared: shared.clone(),
            hyrax_recursion_setup,
            program: VerifierProgram::Full(program),
        }
    }

    /// Create verifier preprocessing in Committed mode with trusted commitments.
    ///
    /// This is the "fast path" for online verification. The `TrustedProgramCommitments`
    /// type guarantees (at the type level) that these commitments were derived from
    /// actual program via `TrustedProgramCommitments::derive()`.
    ///
    /// # Trust Model
    /// The caller must ensure the commitments were honestly derived (e.g., loaded from
    /// a trusted file or received from trusted preprocessing).
    #[tracing::instrument(skip_all, name = "JoltVerifierPreprocessing::new_committed")]
    pub fn new_committed(
        shared: JoltSharedPreprocessing,
        generators: PCS::VerifierSetup,
        hyrax_recursion_setup: PedersenGenerators<GrumpkinProjective>,
        program_commitments: TrustedProgramCommitments<PCS>,
    ) -> JoltVerifierPreprocessing<F, PCS> {
        Self {
            generators,
            shared,
            hyrax_recursion_setup,
            program: VerifierProgram::Committed(program_commitments),
        }
    }
}

#[cfg(feature = "prover")]
impl<F: JoltField, PCS: CommitmentScheme<Field = F>> From<&JoltProverPreprocessing<F, PCS>>
    for JoltVerifierPreprocessing<F, PCS>
{
    fn from(prover_preprocessing: &JoltProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&prover_preprocessing.generators);
        let shared = prover_preprocessing.shared.clone();
        let hyrax_recursion_setup = prover_preprocessing.hyrax_recursion_setup.clone();
        // Choose VerifierProgram variant based on whether prover has program commitments
        let program = match &prover_preprocessing.program_commitments {
            Some(commitments) => VerifierProgram::Committed(commitments.clone()),
            None => VerifierProgram::Full(Arc::clone(&prover_preprocessing.program)),
        };
        Self {
            generators,
            shared,
            hyrax_recursion_setup,
            program,
        }
    }
}
