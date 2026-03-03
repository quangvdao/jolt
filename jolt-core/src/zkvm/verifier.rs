use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::{DoryContext, DoryGlobals};
use crate::subprotocols::sumcheck::BatchedSumcheck;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::claim_reductions::advice::ReductionPhase;
use crate::zkvm::claim_reductions::RegistersClaimReductionSumcheckVerifier;
use crate::zkvm::config::OneHotParams;
#[cfg(feature = "prover")]
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::zkvm::ram::RAMPreprocessing;
use crate::zkvm::witness::all_committed_polynomials;
use crate::zkvm::Serializable;
use crate::zkvm::{
    bytecode::read_raf_checking::BytecodeReadRafSumcheckVerifier,
    claim_reductions::{
        AdviceClaimReductionVerifier, AdviceKind, HammingWeightClaimReductionVerifier,
        IncClaimReductionSumcheckVerifier, InstructionLookupsClaimReductionSumcheckVerifier,
        RamRaClaimReductionSumcheckVerifier,
    },
    fiat_shamir_preamble,
    instruction_lookups::{
        ra_virtual::RaSumcheckVerifier as LookupsRaSumcheckVerifier,
        read_raf_checking::InstructionReadRafSumcheckVerifier,
    },
    proof_serialization::JoltProof,
    r1cs::key::UniformSpartanKey,
    ram::{
        compute_min_ram_K, gen_ram_initial_memory_state,
        hamming_booleanity::HammingBooleanitySumcheckVerifier,
        output_check::OutputSumcheckVerifier, ra_virtual::RamRaVirtualSumcheckVerifier,
        raf_evaluation::RafEvaluationSumcheckVerifier as RamRafEvaluationSumcheckVerifier,
        read_write_checking::RamReadWriteCheckingVerifier, val_check::RamValCheckSumcheckVerifier,
        verifier_accumulate_advice,
    },
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
        compute_advice_lagrange_factor, OpeningAccumulator, OpeningPoint, SumcheckId,
        VerifierOpeningAccumulator,
    },
    pprof_scope,
    subprotocols::{
        booleanity::{BooleanitySumcheckParams, BooleanitySumcheckVerifier},
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::witness::CommittedPolynomial,
};
use anyhow::Context;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use itertools::Itertools;
use tracer::instruction::Instruction;
use tracer::JoltDevice;

pub struct JoltVerifier<
    'a,
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
> {
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
    pub spartan_key: UniformSpartanKey<F>,
    pub one_hot_params: OneHotParams,
}

impl<'a, F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript>
    JoltVerifier<'a, F, PCS, ProofTranscript>
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

        let min_ram_K = compute_min_ram_K(
            &preprocessing.shared.ram,
            &preprocessing.shared.memory_layout,
        );
        if !proof.ram_K.is_power_of_two() || proof.ram_K < min_ram_K {
            return Err(ProofVerifyError::InvalidRamK(proof.ram_K, min_ram_K));
        }

        proof
            .rw_config
            .validate(proof.trace_length.log_2(), proof.ram_K.log_2())
            .map_err(ProofVerifyError::InvalidReadWriteConfig)?;

        // Construct full params from the validated config.
        let bytecode_K = preprocessing.shared.bytecode.code_size;
        let one_hot_params =
            OneHotParams::from_config(&proof.one_hot_config, bytecode_K, proof.ram_K);

        Ok(Self {
            trusted_advice_commitment,
            program_io,
            proof,
            preprocessing,
            transcript,
            opening_accumulator,
            advice_reduction_verifier_trusted: None,
            advice_reduction_verifier_untrusted: None,
            spartan_key,
            one_hot_params,
        })
    }

    #[tracing::instrument(skip_all)]
    pub fn verify(mut self) -> Result<(), anyhow::Error> {
        let _pprof_verify = pprof_scope!("verify");
        tracing::info!("starting verify");

        fiat_shamir_preamble(
            &self.program_io,
            self.proof.ram_K,
            self.proof.trace_length,
            &mut self.transcript,
        );

        // Append commitments to transcript
        for commitment in &self.proof.commitments {
            self.transcript
                .append_serializable(b"commitment", commitment);
        }
        // Append untrusted advice commitment to transcript
        if let Some(ref untrusted_advice_commitment) = self.proof.untrusted_advice_commitment {
            self.transcript
                .append_serializable(b"untrusted_advice", untrusted_advice_commitment);
        }
        // Append trusted advice commitment to transcript
        if let Some(ref trusted_advice_commitment) = self.trusted_advice_commitment {
            self.transcript
                .append_serializable(b"trusted_advice", trusted_advice_commitment);
        }

        self.verify_stage1()?;
        self.verify_stage2()?;
        self.verify_stage3()?;
        self.verify_stage4()?;
        self.verify_stage5()?;
        self.verify_stage6()?;
        self.verify_stage7()?;
        self.verify_stage8()?;

        Ok(())
    }

    fn verify_stage1(&mut self) -> Result<(), anyhow::Error> {
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

    fn verify_stage2(&mut self) -> Result<(), anyhow::Error> {
        let uni_skip_params = verify_stage2_uni_skip(
            &self.proof.stage2_uni_skip_first_round_proof,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 2 univariate skip first round")?;

        let ram_read_write_checking = RamReadWriteCheckingVerifier::new(
            &self.opening_accumulator,
            &mut self.transcript,
            &self.one_hot_params,
            self.proof.trace_length,
            &self.proof.rw_config,
        );

        let spartan_product_virtual_remainder = ProductVirtualRemainderVerifier::new(
            self.proof.trace_length,
            uni_skip_params,
            &self.opening_accumulator,
        );

        let instruction_claim_reduction = InstructionLookupsClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let ram_raf_evaluation = RamRafEvaluationSumcheckVerifier::new(
            &self.program_io.memory_layout,
            &self.one_hot_params,
            self.proof.trace_length,
            &self.proof.rw_config,
            &self.opening_accumulator,
        );

        let ram_output_check = OutputSumcheckVerifier::new(
            self.proof.ram_K,
            &self.program_io,
            &mut self.transcript,
            self.proof.trace_length,
            &self.proof.rw_config,
        );

        let _r_stage2 = BatchedSumcheck::verify(
            &self.proof.stage2_sumcheck_proof,
            vec![
                &ram_read_write_checking,
                &spartan_product_virtual_remainder,
                &instruction_claim_reduction,
                &ram_raf_evaluation,
                &ram_output_check,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 2")?;

        Ok(())
    }

    fn verify_stage3(&mut self) -> Result<(), anyhow::Error> {
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
                &spartan_shift,
                &spartan_instruction_input,
                &spartan_registers_claim_reduction,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 3")?;

        Ok(())
    }

    fn verify_stage4(&mut self) -> Result<(), anyhow::Error> {
        let registers_read_write_checking = RegistersReadWriteCheckingVerifier::new(
            self.proof.trace_length,
            &self.opening_accumulator,
            &mut self.transcript,
            &self.proof.rw_config,
        );
        verifier_accumulate_advice::<F>(
            self.proof.ram_K,
            &self.program_io,
            self.proof.untrusted_advice_commitment.is_some(),
            self.trusted_advice_commitment.is_some(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        // Domain-separate the batching challenge.
        self.transcript.append_bytes(b"ram_val_check_gamma", &[]);
        let ram_val_check_gamma: F = self.transcript.challenge_scalar::<F>();
        let initial_ram_state = gen_ram_initial_memory_state::<F>(
            self.proof.ram_K,
            &self.preprocessing.shared.ram,
            &self.program_io,
        );
        let ram_val_check = RamValCheckSumcheckVerifier::new(
            &initial_ram_state,
            &self.program_io,
            self.proof.trace_length,
            self.proof.ram_K,
            &self.proof.rw_config,
            ram_val_check_gamma,
            &self.opening_accumulator,
        );

        let _r_stage4 = BatchedSumcheck::verify(
            &self.proof.stage4_sumcheck_proof,
            vec![
                &registers_read_write_checking as &dyn SumcheckInstanceVerifier<F, ProofTranscript>,
                &ram_val_check,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 4")?;

        Ok(())
    }

    fn verify_stage5(&mut self) -> Result<(), anyhow::Error> {
        let n_cycle_vars = self.proof.trace_length.log_2();

        let lookups_read_raf = InstructionReadRafSumcheckVerifier::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let ram_ra_reduction = RamRaClaimReductionSumcheckVerifier::new(
            self.proof.trace_length,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let registers_val_evaluation =
            RegistersValEvaluationSumcheckVerifier::new(&self.opening_accumulator);

        let _r_stage5 = BatchedSumcheck::verify(
            &self.proof.stage5_sumcheck_proof,
            vec![
                &lookups_read_raf,
                &ram_ra_reduction,
                &registers_val_evaluation,
            ],
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 5")?;

        Ok(())
    }

    fn verify_stage6(&mut self) -> Result<(), anyhow::Error> {
        let n_cycle_vars = self.proof.trace_length.log_2();
        let bytecode_read_raf = BytecodeReadRafSumcheckVerifier::gen(
            &self.preprocessing.shared.bytecode,
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let ram_hamming_booleanity =
            HammingBooleanitySumcheckVerifier::new(&self.opening_accumulator);
        let booleanity_params = BooleanitySumcheckParams::new(
            n_cycle_vars,
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
            PCS::uses_onehot_inc(),
        );

        let booleanity = BooleanitySumcheckVerifier::new(booleanity_params);
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
            PCS::uses_onehot_inc(),
        );

        // Advice claim reduction (Phase 1 in Stage 6): trusted and untrusted are separate instances.
        if self.trusted_advice_commitment.is_some() {
            self.advice_reduction_verifier_trusted = Some(AdviceClaimReductionVerifier::new(
                AdviceKind::Trusted,
                &self.program_io.memory_layout,
                self.proof.trace_length,
                &self.opening_accumulator,
            ));
        }
        if self.proof.untrusted_advice_commitment.is_some() {
            self.advice_reduction_verifier_untrusted = Some(AdviceClaimReductionVerifier::new(
                AdviceKind::Untrusted,
                &self.program_io.memory_layout,
                self.proof.trace_length,
                &self.opening_accumulator,
            ));
        }

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> = vec![
            &bytecode_read_raf,
            &booleanity,
            &ram_hamming_booleanity,
            &ram_ra_virtual,
            &lookups_ra_virtual,
            &inc_reduction,
        ];
        if let Some(ref advice) = self.advice_reduction_verifier_trusted {
            instances.push(advice);
        }
        if let Some(ref advice) = self.advice_reduction_verifier_untrusted {
            instances.push(advice);
        }

        let _r_stage6 = BatchedSumcheck::verify(
            &self.proof.stage6_sumcheck_proof,
            instances,
            &mut self.opening_accumulator,
            &mut self.transcript,
        )
        .context("Stage 6")?;

        Ok(())
    }

    /// Stage 7: HammingWeight claim reduction verification.
    fn verify_stage7(&mut self) -> Result<(), anyhow::Error> {
        // Create verifier for HammingWeightClaimReduction
        // (r_cycle and r_addr_bool are extracted from Booleanity opening internally)
        let hw_verifier = HammingWeightClaimReductionVerifier::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
            PCS::uses_onehot_inc(),
        );

        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>> =
            vec![&hw_verifier];
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

    /// Stage 8: Dory batch opening verification.
    fn verify_stage8(&mut self) -> Result<(), anyhow::Error> {
        let _guard = PCS::dory_layout(&self.proof.pcs_config).map(|layout| {
            DoryGlobals::initialize_context(
                1 << self.one_hot_params.log_k_chunk,
                self.proof.trace_length.next_power_of_two(),
                DoryContext::Main,
                Some(layout),
            )
        });

        // Get the unified opening point from HammingWeightClaimReduction
        // This contains (r_address_stage7 || r_cycle_stage6) in big-endian
        let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        let r_address_stage7 = &opening_point.r[..log_k_chunk];

        let mut polynomial_claims = Vec::new();

        if PCS::uses_onehot_inc() {
            for i in 0..self.one_hot_params.inc_onehot_d() {
                let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::RdIncRa(i),
                    SumcheckId::HammingWeightClaimReduction,
                );
                polynomial_claims.push((CommittedPolynomial::RdIncRa(i), claim));
            }
            let (_, rd_msb_claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RdIncMsb,
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::RdIncMsb, rd_msb_claim));
            for i in 0..self.one_hot_params.inc_onehot_d() {
                let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::RamIncRa(i),
                    SumcheckId::HammingWeightClaimReduction,
                );
                polynomial_claims.push((CommittedPolynomial::RamIncRa(i), claim));
            }
            let (_, ram_msb_claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamIncMsb,
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::RamIncMsb, ram_msb_claim));
        } else {
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
        }

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

        let main_polys = all_committed_polynomials(&self.one_hot_params, PCS::uses_onehot_inc());
        let mut commitments_map: HashMap<CommittedPolynomial, PCS::Commitment> =
            if let Some(arity) = PCS::packed_main_commitment_arity() {
                if arity != 1 {
                    return Err(ProofVerifyError::InvalidInputLength(1, arity).into());
                }
                if self.proof.commitments.len() != arity {
                    return Err(ProofVerifyError::InvalidInputLength(
                        arity,
                        self.proof.commitments.len(),
                    )
                    .into());
                }
                main_polys
                    .clone()
                    .into_iter()
                    .map(|p| (p, self.proof.commitments[0].clone()))
                    .collect()
            } else {
                if self.proof.commitments.len() != main_polys.len() {
                    return Err(ProofVerifyError::InvalidInputLength(
                        main_polys.len(),
                        self.proof.commitments.len(),
                    )
                    .into());
                }
                main_polys
                    .iter()
                    .copied()
                    .zip_eq(&self.proof.commitments)
                    .map(|(p, c)| (p, c.clone()))
                    .collect()
            };
        if let Some(ref commitment) = self.trusted_advice_commitment {
            commitments_map.insert(CommittedPolynomial::TrustedAdvice, commitment.clone());
        }
        if let Some(ref commitment) = self.proof.untrusted_advice_commitment {
            commitments_map.insert(CommittedPolynomial::UntrustedAdvice, commitment.clone());
        }

        if PCS::packed_main_commitment_arity().is_some() {
            let mut claim_map: HashMap<CommittedPolynomial, F> = HashMap::new();
            for (poly, claim) in polynomial_claims {
                let prev = claim_map.insert(poly, claim);
                if prev.is_some() {
                    return Err(ProofVerifyError::InternalError.into());
                }
            }

            let mut sorted_claims = Vec::with_capacity(main_polys.len() + 2);
            for poly in &main_polys {
                let claim = claim_map
                    .remove(poly)
                    .ok_or(ProofVerifyError::InternalError)?;
                sorted_claims.push(claim);
            }

            let mut individual_ids = Vec::new();
            for advice_poly in [
                CommittedPolynomial::TrustedAdvice,
                CommittedPolynomial::UntrustedAdvice,
            ] {
                if let Some(claim) = claim_map.remove(&advice_poly) {
                    sorted_claims.push(claim);
                    individual_ids.push(advice_poly);
                }
            }
            if !claim_map.is_empty() {
                return Err(ProofVerifyError::InternalError.into());
            }

            let packed_commitment = commitments_map
                .remove(main_polys.first().ok_or(ProofVerifyError::InternalError)?)
                .ok_or(ProofVerifyError::InternalError)?;
            let mut commitment_refs = Vec::with_capacity(1 + individual_ids.len());
            commitment_refs.push(packed_commitment);
            for id in &individual_ids {
                commitment_refs.push(
                    commitments_map
                        .remove(id)
                        .ok_or(ProofVerifyError::InternalError)?,
                );
            }
            let commitment_ref_slice: Vec<&PCS::Commitment> = commitment_refs.iter().collect();

            return PCS::default()
                .batch_verify(
                    &self.proof.joint_opening_proof,
                    &self.preprocessing.generators,
                    &mut self.transcript,
                    &opening_point.r,
                    &commitment_ref_slice,
                    &sorted_claims,
                    &[],
                )
                .context("Stage 8");
        }

        // 2. Sample gamma and compute powers for RLC
        let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
        self.transcript.append_scalars(b"rlc_claims", &claims);
        let gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());

        // Accumulate gamma coefficients per unique polynomial (BTreeMap for deterministic ordering)
        let mut rlc_map = BTreeMap::new();
        for (gamma, (poly, claim)) in gamma_powers.iter().zip(polynomial_claims.iter()) {
            let prev = rlc_map.insert(*poly, (*gamma, *claim));
            if prev.is_some() {
                return Err(ProofVerifyError::InternalError.into());
            }
        }

        let mut poly_ids: Vec<CommittedPolynomial> = main_polys
            .iter()
            .copied()
            .filter(|poly| rlc_map.contains_key(poly))
            .collect();
        for advice_poly in [
            CommittedPolynomial::TrustedAdvice,
            CommittedPolynomial::UntrustedAdvice,
        ] {
            if rlc_map.contains_key(&advice_poly) {
                poly_ids.push(advice_poly);
            }
        }
        let coeffs_and_claims: Vec<(F, F)> = poly_ids
            .iter()
            .map(|poly| *rlc_map.get(poly).expect("missing RLC entry"))
            .collect();
        let (coeffs, sorted_claims): (Vec<F>, Vec<F>) = coeffs_and_claims.into_iter().unzip();

        let commitment_refs: Vec<PCS::Commitment> = poly_ids
            .iter()
            .map(|id| commitments_map.remove(id).unwrap())
            .collect();
        let commitment_ref_slice: Vec<&PCS::Commitment> = commitment_refs.iter().collect();

        PCS::default()
            .batch_verify(
                &self.proof.joint_opening_proof,
                &self.preprocessing.generators,
                &mut self.transcript,
                &opening_point.r,
                &commitment_ref_slice,
                &sorted_claims,
                &coeffs,
            )
            .context("Stage 8")
    }
}

#[derive(Debug, Clone)]
pub struct JoltSharedPreprocessing {
    pub bytecode: Arc<BytecodePreprocessing>,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
    pub max_padded_trace_length: usize,
}

impl CanonicalSerialize for JoltSharedPreprocessing {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        // Serialize the inner BytecodePreprocessing (not the Arc wrapper)
        self.bytecode
            .as_ref()
            .serialize_with_mode(&mut writer, compress)?;
        self.ram.serialize_with_mode(&mut writer, compress)?;
        self.memory_layout
            .serialize_with_mode(&mut writer, compress)?;
        self.max_padded_trace_length
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.bytecode.serialized_size(compress)
            + self.ram.serialized_size(compress)
            + self.memory_layout.serialized_size(compress)
            + self.max_padded_trace_length.serialized_size(compress)
    }
}

impl CanonicalDeserialize for JoltSharedPreprocessing {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let bytecode =
            BytecodePreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
        let ram = RAMPreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
        let memory_layout = MemoryLayout::deserialize_with_mode(&mut reader, compress, validate)?;
        let max_padded_trace_length =
            usize::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self {
            bytecode: Arc::new(bytecode),
            ram,
            memory_layout,
            max_padded_trace_length,
        })
    }
}

impl ark_serialize::Valid for JoltSharedPreprocessing {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.bytecode.check()?;
        self.ram.check()?;
        self.memory_layout.check()
    }
}

impl JoltSharedPreprocessing {
    #[tracing::instrument(skip_all, name = "JoltSharedPreprocessing::new")]
    pub fn new(
        bytecode: Vec<Instruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_padded_trace_length: usize,
    ) -> JoltSharedPreprocessing {
        let bytecode = Arc::new(BytecodePreprocessing::preprocess(bytecode));
        let ram = RAMPreprocessing::preprocess(memory_init);
        Self {
            bytecode,
            ram,
            memory_layout,
            max_padded_trace_length,
        }
    }
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub shared: JoltSharedPreprocessing,
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
    #[tracing::instrument(skip_all, name = "JoltVerifierPreprocessing::new")]
    pub fn new(
        shared: JoltSharedPreprocessing,
        generators: PCS::VerifierSetup,
    ) -> JoltVerifierPreprocessing<F, PCS> {
        Self {
            generators,
            shared: shared.clone(),
        }
    }
}

#[cfg(feature = "prover")]
impl<F: JoltField, PCS: CommitmentScheme<Field = F>> From<&JoltProverPreprocessing<F, PCS>>
    for JoltVerifierPreprocessing<F, PCS>
{
    fn from(prover_preprocessing: &JoltProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&prover_preprocessing.generators);
        Self {
            generators,
            shared: prover_preprocessing.shared.clone(),
        }
    }
}
