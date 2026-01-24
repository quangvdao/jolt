#[cfg(test)]
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::{
    subprotocols::streaming_schedule::LinearOnlySchedule,
    zkvm::{claim_reductions::advice::ReductionPhase, config::OneHotConfig},
};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};

use crate::poly::commitment::hyrax::{Hyrax, PedersenGenerators};
use ark_bn254::{Fq, Fq12, Fr};
use ark_grumpkin::Projective as GrumpkinProjective;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;

use crate::zkvm::recursion::{RecursionProofResult, MAX_RECURSION_DENSE_NUM_VARS};

#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;
#[cfg(feature = "allocative")]
use crate::utils::profiling::{print_data_structure_heap_usage, write_flamegraph_svg};
use crate::{
    field::JoltField,
    guest,
    poly::{
        commitment::{
            commitment_scheme::StreamingCommitmentScheme,
            dory::{DoryContext, DoryGlobals, DoryLayout},
        },
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{
            compute_advice_lagrange_factor, DoryOpeningState, OpeningAccumulator,
            ProverOpeningAccumulator, SumcheckId,
        },
        rlc_polynomial::{RLCStreamingData, TraceSource},
        rlc_utils::{compute_joint_claim, compute_rlc_coefficients},
    },
    pprof_scope,
    subprotocols::{
        booleanity::{
            BooleanityAddressSumcheckProver, BooleanityCycleSumcheckProver,
            BooleanitySumcheckParams,
        },
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        univariate_skip::{prove_uniskip_round, UniSkipFirstRoundProof},
    },
    transcripts::Transcript,
    utils::{math::Math, thread::drop_in_background_thread},
    zkvm::{
        bytecode::{chunks::total_lanes, read_raf_checking::BytecodeReadRafSumcheckParams},
        claim_reductions::{
            AdviceClaimReductionParams, AdviceClaimReductionProver, AdviceKind,
            BytecodeClaimReductionParams, BytecodeClaimReductionProver, BytecodeReductionPhase,
            HammingWeightClaimReductionParams, HammingWeightClaimReductionProver,
            IncClaimReductionSumcheckParams, IncClaimReductionSumcheckProver,
            InstructionLookupsClaimReductionSumcheckParams,
            InstructionLookupsClaimReductionSumcheckProver, ProgramImageClaimReductionParams,
            ProgramImageClaimReductionProver, RaReductionParams, RamRaClaimReductionSumcheckProver,
            RegistersClaimReductionSumcheckParams, RegistersClaimReductionSumcheckProver,
        },
        config::{OneHotParams, ProgramMode, ReadWriteConfig},
        instruction_lookups::{
            ra_virtual::InstructionRaSumcheckParams,
            read_raf_checking::InstructionReadRafSumcheckParams,
        },
        program::{ProgramPreprocessing, TrustedProgramCommitments, TrustedProgramHints},
        ram::{
            hamming_booleanity::HammingBooleanitySumcheckParams,
            output_check::OutputSumcheckParams,
            populate_memory_states, prover_accumulate_program_image,
            ra_virtual::RamRaVirtualParams,
            raf_evaluation::RafEvaluationSumcheckParams,
            read_write_checking::RamReadWriteCheckingParams,
            remap_address,
            val_evaluation::{
                ValEvaluationSumcheckParams,
                ValEvaluationSumcheckProver as RamValEvaluationSumcheckProver,
            },
            val_final::{ValFinalSumcheckParams, ValFinalSumcheckProver},
        },
        registers::{
            read_write_checking::RegistersReadWriteCheckingParams,
            val_evaluation::RegistersValEvaluationSumcheckParams,
        },
        spartan::{
            instruction_input::InstructionInputParams,
            outer::{OuterUniSkipParams, OuterUniSkipProver},
            product::{
                ProductVirtualRemainderParams, ProductVirtualUniSkipParams,
                ProductVirtualUniSkipProver,
            },
            shift::ShiftSumcheckParams,
        },
        verifier::JoltSharedPreprocessing,
        witness::all_committed_polynomials,
        Serializable,
    },
};
use crate::{
    poly::commitment::commitment_scheme::{CommitmentScheme, RecursionExt},
    zkvm::{
        bytecode::read_raf_checking::{
            BytecodeReadRafAddressSumcheckProver, BytecodeReadRafCycleSumcheckProver,
        },
        fiat_shamir_preamble,
        instruction_lookups::{
            ra_virtual::InstructionRaSumcheckProver as LookupsRaSumcheckProver,
            read_raf_checking::InstructionReadRafSumcheckProver,
        },
        proof_serialization::{Claims, JoltProof},
        r1cs::key::UniformSpartanKey,
        ram::{
            gen_ram_memory_states, hamming_booleanity::HammingBooleanitySumcheckProver,
            output_check::OutputSumcheckProver, prover_accumulate_advice,
            ra_virtual::RamRaVirtualSumcheckProver,
            raf_evaluation::RafEvaluationSumcheckProver as RamRafEvaluationSumcheckProver,
            read_write_checking::RamReadWriteCheckingProver,
        },
        recursion::prover::RecursionProver,
        registers::{
            read_write_checking::RegistersReadWriteCheckingProver,
            val_evaluation::ValEvaluationSumcheckProver as RegistersValEvaluationSumcheckProver,
        },
        spartan::{
            instruction_input::InstructionInputSumcheckProver,
            outer::{OuterRemainingStreamingSumcheck, OuterSharedState},
            product::ProductVirtualRemainderProver,
            shift::ShiftSumcheckProver,
        },
        witness::CommittedPolynomial,
        ProverDebugInfo,
    },
};

#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::jolt_device::MemoryConfig;
use itertools::zip_eq;
use rayon::prelude::*;
use tracer::{
    emulator::memory::Memory, instruction::Cycle, ChunksIterator, JoltDevice, LazyTraceIterator,
};

/// Data structure to hold Stage 8 values that need to be passed between functions
pub struct Stage8ProofData<
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + RecursionExt<F>,
    ProofTranscript: Transcript,
> {
    /// The gamma powers computed after appending claims
    pub gamma_powers: Vec<F>,
    /// The joint claim: Σ γ_i · claim_i
    pub joint_claim: F,
    /// The joint commitment computed from gamma_powers
    pub joint_commitment: PCS::Commitment,
    /// Forked transcript state right after gamma sampling (before PCS::prove)
    pub witness_gen_transcript: ProofTranscript,
    /// The opening state with all polynomial claims
    pub opening_state: DoryOpeningState<F>,
    /// The witnesses from the first witness_gen call
    pub witnesses: <PCS as RecursionExt<F>>::Witness,
    /// Witness for combine_commitments offloading
    pub combine_witness: Option<crate::zkvm::recursion::witness::GTCombineWitness>,
}

/// Jolt CPU prover for RV64IMAC.
pub struct JoltCpuProver<
    'a,
    F: JoltField,
    PCS: StreamingCommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
> {
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub program_io: JoltDevice,
    pub lazy_trace: LazyTraceIterator,
    pub trace: Arc<Vec<Cycle>>,
    pub advice: JoltAdvice<F, PCS>,
    /// The advice claim reduction sumcheck effectively spans two stages (6 and 7).
    /// Cache the prover state here between stages.
    advice_reduction_prover_trusted: Option<AdviceClaimReductionProver<F>>,
    /// The advice claim reduction sumcheck effectively spans two stages (6 and 7).
    /// Cache the prover state here between stages.
    advice_reduction_prover_untrusted: Option<AdviceClaimReductionProver<F>>,
    /// The bytecode claim reduction sumcheck effectively spans two stages (6b and 7).
    /// Cache the prover state here between stages.
    bytecode_reduction_prover: Option<BytecodeClaimReductionProver<F>>,
    /// Bytecode read RAF params, cached between Stage 6a and 6b.
    bytecode_read_raf_params: Option<BytecodeReadRafSumcheckParams<F>>,
    /// Booleanity params, cached between Stage 6a and 6b.
    booleanity_params: Option<BooleanitySumcheckParams<F>>,
    pub unpadded_trace_len: usize,
    pub padded_trace_len: usize,
    pub transcript: ProofTranscript,
    pub opening_accumulator: ProverOpeningAccumulator<F>,
    pub spartan_key: UniformSpartanKey<F>,
    pub initial_ram_state: Vec<u64>,
    pub final_ram_state: Vec<u64>,
    pub one_hot_params: OneHotParams,
    pub rw_config: ReadWriteConfig,
    pub commitments: Vec<PCS::Commitment>,
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    /// First-class selection of full vs committed bytecode mode.
    pub program_mode: ProgramMode,
}
impl<
        'a,
        F: JoltField,
        PCS: StreamingCommitmentScheme<Field = F> + RecursionExt<F>,
        ProofTranscript: Transcript,
    > JoltCpuProver<'a, F, PCS, ProofTranscript>
where
    <PCS as RecursionExt<F>>::Hint: Send + Sync + 'static,
{
    pub fn gen_from_elf(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        elf_contents: &[u8],
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
        trusted_advice_commitment: Option<PCS::Commitment>,
        trusted_advice_hint: Option<PCS::OpeningProofHint>,
    ) -> Self {
        Self::gen_from_elf_with_program_mode(
            preprocessing,
            elf_contents,
            inputs,
            untrusted_advice,
            trusted_advice,
            trusted_advice_commitment,
            trusted_advice_hint,
            ProgramMode::Full,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gen_from_elf_with_program_mode(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        elf_contents: &[u8],
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
        trusted_advice_commitment: Option<PCS::Commitment>,
        trusted_advice_hint: Option<PCS::OpeningProofHint>,
        program_mode: ProgramMode,
    ) -> Self {
        let memory_config = MemoryConfig {
            max_untrusted_advice_size: preprocessing.shared.memory_layout.max_untrusted_advice_size,
            max_trusted_advice_size: preprocessing.shared.memory_layout.max_trusted_advice_size,
            max_input_size: preprocessing.shared.memory_layout.max_input_size,
            max_output_size: preprocessing.shared.memory_layout.max_output_size,
            stack_size: preprocessing.shared.memory_layout.stack_size,
            memory_size: preprocessing.shared.memory_layout.memory_size,
            program_size: Some(preprocessing.shared.memory_layout.program_size),
        };

        let (lazy_trace, trace, final_memory_state, program_io) = {
            let _pprof_trace = pprof_scope!("trace");
            guest::program::trace(
                elf_contents,
                None,
                inputs,
                untrusted_advice,
                trusted_advice,
                &memory_config,
            )
        };

        let num_riscv_cycles: usize = trace
            .par_iter()
            .map(|cycle| {
                // Count the cycle if the instruction is not part of a inline sequence
                // (`virtual_sequence_remaining` is `None`) or if it's the first instruction
                // in a inline sequence (`virtual_sequence_remaining` is `Some(0)`)
                if let Some(virtual_sequence_remaining) =
                    cycle.instruction().normalize().virtual_sequence_remaining
                {
                    if virtual_sequence_remaining > 0 {
                        return 0;
                    }
                }
                1
            })
            .sum();
        tracing::info!(
            "{num_riscv_cycles} raw RISC-V instructions + {} virtual instructions = {} total cycles",
            trace.len() - num_riscv_cycles,
            trace.len(),
        );

        Self::gen_from_trace_with_program_mode(
            preprocessing,
            lazy_trace,
            trace,
            program_io,
            trusted_advice_commitment,
            trusted_advice_hint,
            final_memory_state,
            program_mode,
        )
    }

    /// Adjusts the padded trace length to ensure the main Dory matrix is large enough
    /// to embed "extra" (non-trace-streamed) polynomials as the top-left block.
    ///
    /// Returns the adjusted padded_trace_len that satisfies:
    /// - `sigma_main >= max_sigma_a`
    /// - `nu_main >= max_nu_a`
    ///
    /// Panics if `max_padded_trace_length` is too small for the configured sizes.
    #[allow(clippy::too_many_arguments)]
    fn adjust_trace_length_for_advice(
        mut padded_trace_len: usize,
        max_padded_trace_length: usize,
        max_trusted_advice_size: u64,
        max_untrusted_advice_size: u64,
        has_trusted_advice: bool,
        has_untrusted_advice: bool,
        has_program_image: bool,
        program_image_len_words_padded: usize,
    ) -> usize {
        // Canonical advice shape policy (balanced):
        // - advice_vars = log2(advice_len)
        // - sigma_a = ceil(advice_vars/2)
        // - nu_a    = advice_vars - sigma_a
        let mut max_sigma_a = 0usize;
        let mut max_nu_a = 0usize;

        if has_trusted_advice {
            let (sigma_a, nu_a) =
                DoryGlobals::advice_sigma_nu_from_max_bytes(max_trusted_advice_size as usize);
            max_sigma_a = max_sigma_a.max(sigma_a);
            max_nu_a = max_nu_a.max(nu_a);
        }
        if has_untrusted_advice {
            let (sigma_a, nu_a) =
                DoryGlobals::advice_sigma_nu_from_max_bytes(max_untrusted_advice_size as usize);
            max_sigma_a = max_sigma_a.max(sigma_a);
            max_nu_a = max_nu_a.max(nu_a);
        }

        if has_program_image {
            let prog_vars = program_image_len_words_padded.log_2();
            let (sigma_p, nu_p) = DoryGlobals::balanced_sigma_nu(prog_vars);
            max_sigma_a = max_sigma_a.max(sigma_p);
            max_nu_a = max_nu_a.max(nu_p);
        }

        if max_sigma_a == 0 && max_nu_a == 0 {
            return padded_trace_len;
        }

        // Require main matrix dimensions to be large enough to embed advice as the top-left
        // block: sigma_main >= sigma_a and nu_main >= nu_a.
        //
        // This loop doubles padded_trace_len until the main Dory matrix is large enough.
        // Each doubling increases log_t by 1, which increases total_vars by 1 (since
        // log_k_chunk stays constant for a given log_t range), increasing both sigma_main
        // and nu_main by roughly 0.5 each iteration.
        while {
            let log_t = padded_trace_len.log_2();
            let log_k_chunk = OneHotConfig::new(log_t).log_k_chunk as usize;
            let (sigma_main, nu_main) = DoryGlobals::main_sigma_nu(log_k_chunk, log_t);
            sigma_main < max_sigma_a || nu_main < max_nu_a
        } {
            if padded_trace_len >= max_padded_trace_length {
                // This is a configuration error: the preprocessing was set up with
                // max_padded_trace_length too small for the configured advice sizes.
                // Cannot recover at runtime - user must fix their configuration.
                let log_t = padded_trace_len.log_2();
                let log_k_chunk = OneHotConfig::new(log_t).log_k_chunk as usize;
                let total_vars = log_k_chunk + log_t;
                let (sigma_main, nu_main) = DoryGlobals::main_sigma_nu(log_k_chunk, log_t);
                panic!(
                    "Configuration error: trace too small to embed advice into Dory batch opening.\n\
                    Current: (sigma_main={sigma_main}, nu_main={nu_main}) from total_vars={total_vars} (log_t={log_t}, log_k_chunk={log_k_chunk})\n\
                    Required: (sigma_a={max_sigma_a}, nu_a={max_nu_a}) for advice embedding\n\
                    Solutions:\n\
                    1. Increase max_trace_length in preprocessing (currently {max_padded_trace_length})\n\
                    2. Reduce max_trusted_advice_size or max_untrusted_advice_size\n\
                    3. Run a program with more cycles"
                );
            }
            padded_trace_len = (padded_trace_len * 2).min(max_padded_trace_length);
        }

        padded_trace_len
    }

    pub fn gen_from_trace(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        lazy_trace: LazyTraceIterator,
        trace: Vec<Cycle>,
        program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        trusted_advice_hint: Option<PCS::OpeningProofHint>,
        final_memory_state: Memory,
    ) -> Self {
        Self::gen_from_trace_with_program_mode(
            preprocessing,
            lazy_trace,
            trace,
            program_io,
            trusted_advice_commitment,
            trusted_advice_hint,
            final_memory_state,
            ProgramMode::Full,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gen_from_trace_with_program_mode(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        lazy_trace: LazyTraceIterator,
        mut trace: Vec<Cycle>,
        mut program_io: JoltDevice,
        trusted_advice_commitment: Option<PCS::Commitment>,
        trusted_advice_hint: Option<PCS::OpeningProofHint>,
        final_memory_state: Memory,
        program_mode: ProgramMode,
    ) -> Self {
        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        // Setup trace length and padding
        let unpadded_trace_len = trace.len();
        let padded_trace_len = if unpadded_trace_len < 256 {
            256 // ensures that T >= k^{1/D}
        } else {
            (trace.len() + 1).next_power_of_two()
        };

        // In Committed mode, Stage 8 folds bytecode chunk openings into the *joint* opening.
        // That folding currently requires log_T >= log_K_bytecode, so we ensure the padded trace
        // length is at least the (power-of-two padded) bytecode size.
        //
        // For CycleMajor layout, bytecode chunks are committed with bytecode_T for coefficient
        // indexing. The main context's T must be >= bytecode_T for row indices to align correctly
        // during Stage 8 VMP computation.
        let padded_trace_len = if program_mode == ProgramMode::Committed {
            let trusted = preprocessing
                .program_commitments
                .as_ref()
                .expect("program commitments missing in committed preprocessing");
            padded_trace_len
                .max(preprocessing.shared.bytecode_size())
                .max(trusted.bytecode_T) // Ensure T >= bytecode_T for CycleMajor row alignment
        } else {
            padded_trace_len
        };
        // In Committed mode, ProgramImageClaimReduction uses `m = log2(padded_len_words)` rounds and is
        // back-loaded into Stage 6b, so we require log_T >= m. A sufficient condition is T >= padded_len_words.
        let (has_program_image, program_image_len_words_padded) =
            if program_mode == ProgramMode::Committed {
                let trusted = preprocessing
                    .program_commitments
                    .as_ref()
                    .expect("program commitments missing in committed preprocessing");
                (true, trusted.program_image_num_words)
            } else {
                (false, 0usize)
            };
        let padded_trace_len = if has_program_image {
            padded_trace_len.max(program_image_len_words_padded)
        } else {
            padded_trace_len
        };
        // We may need extra padding so the main Dory matrix has enough (row, col) variables
        // to embed advice commitments committed in their own preprocessing-only contexts.
        let has_trusted_advice = !program_io.trusted_advice.is_empty();
        let has_untrusted_advice = !program_io.untrusted_advice.is_empty();

        let padded_trace_len = Self::adjust_trace_length_for_advice(
            padded_trace_len,
            preprocessing.shared.max_padded_trace_length,
            preprocessing.shared.memory_layout.max_trusted_advice_size,
            preprocessing.shared.memory_layout.max_untrusted_advice_size,
            has_trusted_advice,
            has_untrusted_advice,
            has_program_image,
            program_image_len_words_padded,
        );

        trace.resize(padded_trace_len, Cycle::NoOp);

        // Calculate K for DoryGlobals initialization
        let ram_K = trace
            .par_iter()
            .filter_map(|cycle| {
                remap_address(
                    cycle.ram_access().address() as u64,
                    &preprocessing.shared.memory_layout,
                )
            })
            .max()
            .unwrap_or(0)
            .max(
                remap_address(
                    preprocessing.program.min_bytecode_address,
                    &preprocessing.shared.memory_layout,
                )
                .unwrap_or(0)
                    + {
                        let base = preprocessing.program.program_image_words.len() as u64;
                        if has_program_image {
                            (program_image_len_words_padded as u64).max(base)
                        } else {
                            base
                        }
                    }
                    + 1,
            )
            .next_power_of_two() as usize;

        let transcript = ProofTranscript::new(b"Jolt");
        let opening_accumulator = ProverOpeningAccumulator::new(trace.len().log_2());

        let spartan_key = UniformSpartanKey::new(trace.len());

        let (initial_ram_state, final_ram_state) = gen_ram_memory_states::<F>(
            ram_K,
            preprocessing.program.min_bytecode_address,
            &preprocessing.program.program_image_words,
            &program_io,
            &final_memory_state,
        );

        let log_T = trace.len().log_2();
        let ram_log_K = ram_K.log_2();
        let rw_config = ReadWriteConfig::new(log_T, ram_log_K);
        let one_hot_params = if program_mode == ProgramMode::Committed {
            let committed = preprocessing
                .program_commitments
                .as_ref()
                .expect("program commitments missing in committed mode");
            let config = OneHotConfig::from_log_k_chunk(committed.log_k_chunk as usize);
            OneHotParams::from_config(&config, preprocessing.shared.bytecode_size(), ram_K)
        } else {
            OneHotParams::new(log_T, preprocessing.shared.bytecode_size(), ram_K)
        };

        Self {
            preprocessing,
            program_io,
            lazy_trace,
            trace: trace.into(),
            advice: JoltAdvice {
                untrusted_advice_polynomial: None,
                trusted_advice_commitment,
                trusted_advice_polynomial: None,
                untrusted_advice_hint: None,
                trusted_advice_hint,
            },
            advice_reduction_prover_trusted: None,
            advice_reduction_prover_untrusted: None,
            bytecode_reduction_prover: None,
            bytecode_read_raf_params: None,
            booleanity_params: None,
            unpadded_trace_len,
            padded_trace_len,
            transcript,
            opening_accumulator,
            spartan_key,
            initial_ram_state,
            final_ram_state,
            one_hot_params,
            rw_config,
            commitments: vec![],
            untrusted_advice_commitment: None,
            program_mode,
        }
    }

    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all)]
    pub fn prove(
        mut self,
    ) -> (
        JoltProof<F, PCS, ProofTranscript>,
        Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) {
        let _pprof_prove = pprof_scope!("prove");

        let start = Instant::now();
        fiat_shamir_preamble(
            &self.program_io,
            self.one_hot_params.ram_k,
            self.trace.len(),
            &mut self.transcript,
        );

        tracing::info!(
            "bytecode size: {}",
            self.preprocessing.shared.bytecode_size()
        );

        let (commitments, mut opening_proof_hints) = self.generate_and_commit_witness_polynomials();
        self.commitments = commitments.clone();
        let untrusted_advice_commitment = self.generate_and_commit_untrusted_advice();
        self.untrusted_advice_commitment = untrusted_advice_commitment.clone();
        self.generate_and_commit_trusted_advice();

        if self.program_mode == ProgramMode::Committed {
            if let Some(trusted) = &self.preprocessing.program_commitments {
                // Append bytecode chunk commitments
                for commitment in &trusted.bytecode_commitments {
                    self.transcript.append_serializable(commitment);
                }
                // Append program image commitment
                self.transcript
                    .append_serializable(&trusted.program_image_commitment);
                #[cfg(test)]
                {
                    // Sanity: re-commit the program image polynomial and ensure it matches the trusted commitment.
                    // Must use the same padded size and context as TrustedProgramCommitments::derive().
                    let poly =
                        TrustedProgramCommitments::<PCS>::build_program_image_polynomial_padded::<F>(
                            &self.preprocessing.program,
                            trusted.program_image_num_words,
                        );
                    // Recompute log_k_chunk and max_log_t to get Main's sigma.
                    let max_t_any: usize = self
                        .preprocessing
                        .shared
                        .max_padded_trace_length
                        .max(self.preprocessing.shared.bytecode_size())
                        .next_power_of_two();
                    let max_log_t = max_t_any.log_2();
                    let log_k_chunk = if max_log_t < common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T
                    {
                        4
                    } else {
                        8
                    };
                    // Use the explicit context initialization to match TrustedProgramCommitments::derive()
                    let (sigma_main, _) = DoryGlobals::main_sigma_nu(log_k_chunk, max_log_t);
                    let main_num_columns = 1usize << sigma_main;
                    DoryGlobals::initialize_program_image_context_with_num_columns(
                        1usize << log_k_chunk,
                        trusted.program_image_num_words,
                        main_num_columns,
                    );
                    let _ctx = DoryGlobals::with_context(DoryContext::ProgramImage);
                    let mle = MultilinearPolynomial::from(poly);
                    let (recommit, _hint) = PCS::commit(&mle, &self.preprocessing.generators);
                    assert_eq!(
                        recommit, trusted.program_image_commitment,
                        "ProgramImageInit commitment mismatch vs polynomial used in proving"
                    );
                }
            }
        }

        // Add advice hints for batched Stage 8 opening
        if let Some(hint) = self.advice.trusted_advice_hint.take() {
            opening_proof_hints.insert(CommittedPolynomial::TrustedAdvice, hint);
        }
        if let Some(hint) = self.advice.untrusted_advice_hint.take() {
            opening_proof_hints.insert(CommittedPolynomial::UntrustedAdvice, hint);
        }
        if self.program_mode == ProgramMode::Committed {
            if let Some(hints) = self.preprocessing.program_hints.as_ref() {
                for (idx, hint) in hints.bytecode_hints.iter().enumerate() {
                    opening_proof_hints
                        .insert(CommittedPolynomial::BytecodeChunk(idx), hint.clone());
                }
            }
            if let Some(hints) = self.preprocessing.program_hints.as_ref() {
                opening_proof_hints.insert(
                    CommittedPolynomial::ProgramImageInit,
                    hints.program_image_hint.clone(),
                );
            }
        }

        let (stage1_uni_skip_first_round_proof, stage1_sumcheck_proof) = self.prove_stage1();
        let (stage2_uni_skip_first_round_proof, stage2_sumcheck_proof) = self.prove_stage2();
        let stage3_sumcheck_proof = self.prove_stage3();
        let stage4_sumcheck_proof = self.prove_stage4();
        let stage5_sumcheck_proof = self.prove_stage5();
        let stage6a_sumcheck_proof = self.prove_stage6a();
        let stage6b_sumcheck_proof = self.prove_stage6b();
        let stage7_sumcheck_proof = self.prove_stage7();

        // Stage 8: Dory batch opening proof with hint generation
        let (joint_opening_proof, stage8_pcs_hint, stage8_proof_data, stage8_combine_hint) =
            self.prove_stage8(opening_proof_hints, true);
        let stage8_pcs_hint = stage8_pcs_hint.expect("Stage 8 hint required for recursion");
        let stage8_proof_data =
            stage8_proof_data.expect("Stage 8 proof data required for recursion");

        // Stages 9-13: Consolidated recursion proof generation
        // This handles:
        // - Stage 9: Create RecursionProver from witnesses + sample gamma/delta
        // - Stage 10: Extract constraint metadata
        // - Stage 11: Build dense polynomial + commit (BEFORE sumchecks for soundness)
        // - Stage 12: Run recursion sumchecks (stages 1-5 of RecursionProver)
        // - Stage 13: Generate Hyrax opening proof
        let (recursion_proof, recursion_constraint_metadata) = self
            .prove_stage_recursive(stage8_proof_data)
            .expect("Failed to generate recursion proof");

        #[cfg(test)]
        assert!(
            self.opening_accumulator
                .appended_virtual_openings
                .borrow()
                .is_empty(),
            "Not all virtual openings have been proven, missing: {:?}",
            self.opening_accumulator.appended_virtual_openings.borrow()
        );

        #[cfg(test)]
        let debug_info = Some(ProverDebugInfo {
            transcript: self.transcript.clone(),
            opening_accumulator: self.opening_accumulator.clone(),
            prover_setup: self.preprocessing.generators.clone(),
        });
        #[cfg(not(test))]
        let debug_info = None;

        let proof = JoltProof {
            opening_claims: Claims(self.opening_accumulator.openings.clone()),
            commitments,
            untrusted_advice_commitment,
            stage1_uni_skip_first_round_proof,
            stage1_sumcheck_proof,
            stage2_uni_skip_first_round_proof,
            stage2_sumcheck_proof,
            stage3_sumcheck_proof,
            stage4_sumcheck_proof,
            stage5_sumcheck_proof,
            stage6a_sumcheck_proof,
            stage6b_sumcheck_proof,
            stage7_sumcheck_proof,
            stage8_opening_proof: joint_opening_proof,
            stage8_combine_hint,
            stage9_pcs_hint: Some(stage8_pcs_hint),
            stage10_recursion_metadata: recursion_constraint_metadata,
            recursion_proof,
            trusted_advice_val_evaluation_proof: None,
            trusted_advice_val_final_proof: None,
            untrusted_advice_val_evaluation_proof: None,
            untrusted_advice_val_final_proof: None,
            trace_length: self.trace.len(),
            ram_K: self.one_hot_params.ram_k,
            bytecode_K: self.one_hot_params.bytecode_k,
            program_mode: self.program_mode,
            rw_config: self.rw_config.clone(),
            one_hot_config: self.one_hot_params.to_config(),
            dory_layout: DoryGlobals::get_layout(),
        };

        let prove_duration = start.elapsed();

        tracing::info!(
            "Proved in {:.1}s ({:.1} kHz / padded {:.1} kHz)",
            prove_duration.as_secs_f64(),
            self.unpadded_trace_len as f64 / prove_duration.as_secs_f64() / 1000.0,
            self.padded_trace_len as f64 / prove_duration.as_secs_f64() / 1000.0,
        );

        (proof, debug_info)
    }

    #[tracing::instrument(skip_all, name = "generate_and_commit_witness_polynomials")]
    fn generate_and_commit_witness_polynomials(
        &mut self,
    ) -> (
        Vec<PCS::Commitment>,
        HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
    ) {
        let _guard = if self.program_mode == ProgramMode::Committed {
            let committed = self
                .preprocessing
                .program_commitments
                .as_ref()
                .expect("program commitments missing in committed mode");
            DoryGlobals::initialize_main_context_with_num_columns(
                1 << self.one_hot_params.log_k_chunk,
                self.padded_trace_len,
                committed.bytecode_num_columns,
                Some(DoryGlobals::get_layout()),
            )
        } else {
            DoryGlobals::initialize_context(
                1 << self.one_hot_params.log_k_chunk,
                self.padded_trace_len,
                DoryContext::Main,
                Some(DoryGlobals::get_layout()),
            )
        };

        let polys = all_committed_polynomials(&self.one_hot_params);
        let T = DoryGlobals::get_T();

        // For AddressMajor, use non-streaming commit path since streaming assumes CycleMajor layout
        let (commitments, hint_map) = if DoryGlobals::get_layout() == DoryLayout::AddressMajor {
            tracing::debug!(
                "Using non-streaming commit path for AddressMajor layout with {} polynomials",
                polys.len()
            );

            // Materialize the trace for non-streaming commit
            let trace: Vec<Cycle> = self
                .lazy_trace
                .clone()
                .pad_using(T, |_| Cycle::NoOp)
                .collect();

            // Generate witnesses and commit using the regular (non-streaming) path
            let (commitments, hints): (Vec<_>, Vec<_>) = polys
                .par_iter()
                .map(|poly_id| {
                    let witness: MultilinearPolynomial<F> = poly_id.generate_witness(
                        &self.preprocessing.program,
                        &self.preprocessing.shared.memory_layout,
                        &trace,
                        Some(&self.one_hot_params),
                    );
                    PCS::commit(&witness, &self.preprocessing.generators)
                })
                .unzip();

            let hint_map = HashMap::from_iter(zip_eq(polys, hints));
            (commitments, hint_map)
        } else {
            // CycleMajor: use streaming
            let row_len = DoryGlobals::get_num_columns();
            let num_rows = T / DoryGlobals::get_max_num_rows();

            tracing::debug!(
                "Generating and committing {} witness polynomials with T={}, row_len={}, num_rows={}",
                polys.len(),
                T,
                row_len,
                num_rows
            );

            // Tier 1: Compute row commitments for each polynomial
            let mut row_commitments: Vec<Vec<PCS::ChunkState>> = vec![vec![]; num_rows];

            self.lazy_trace
                .clone()
                .pad_using(T, |_| Cycle::NoOp)
                .iter_chunks(row_len)
                .zip(row_commitments.iter_mut())
                .par_bridge()
                .for_each(|(chunk, row_tier1_commitments)| {
                    let res: Vec<_> = polys
                        .par_iter()
                        .map(|poly| {
                            poly.stream_witness_and_commit_rows::<_, PCS>(
                                &self.preprocessing.generators,
                                &self.preprocessing.shared,
                                &self.preprocessing.program,
                                &chunk,
                                &self.one_hot_params,
                            )
                        })
                        .collect();
                    *row_tier1_commitments = res;
                });

            // Transpose: row_commitments[row][poly] -> tier1_per_poly[poly][row]
            let tier1_per_poly: Vec<Vec<PCS::ChunkState>> = (0..polys.len())
                .into_par_iter()
                .map(|poly_idx| {
                    row_commitments
                        .iter()
                        .flat_map(|row| row.get(poly_idx).cloned())
                        .collect()
                })
                .collect();

            // Tier 2: Compute final commitments from tier1 commitments
            let (commitments, hints): (Vec<_>, Vec<_>) = tier1_per_poly
                .into_par_iter()
                .zip(&polys)
                .map(|(tier1_commitments, poly)| {
                    let onehot_k = poly.get_onehot_k(&self.one_hot_params);
                    PCS::aggregate_chunks(
                        &self.preprocessing.generators,
                        onehot_k,
                        &tier1_commitments,
                    )
                })
                .unzip();

            let hint_map = HashMap::from_iter(zip_eq(polys, hints));
            (commitments, hint_map)
        };

        // Append commitments to transcript
        for commitment in &commitments {
            self.transcript.append_serializable(commitment);
        }

        (commitments, hint_map)
    }

    fn generate_and_commit_untrusted_advice(&mut self) -> Option<PCS::Commitment> {
        if self.program_io.untrusted_advice.is_empty() {
            return None;
        }

        // Commit untrusted advice in its dedicated Dory context, using a preprocessing-only
        // matrix shape derived deterministically from the advice length (balanced dims).

        let mut untrusted_advice_vec =
            vec![0; self.program_io.memory_layout.max_untrusted_advice_size as usize / 8];

        populate_memory_states(
            0,
            &self.program_io.untrusted_advice,
            Some(&mut untrusted_advice_vec),
            None,
        );

        let poly = MultilinearPolynomial::from(untrusted_advice_vec);
        let advice_len = poly.len().next_power_of_two().max(1);

        let _guard =
            DoryGlobals::initialize_context(1, advice_len, DoryContext::UntrustedAdvice, None);
        let _ctx = DoryGlobals::with_context(DoryContext::UntrustedAdvice);
        let (commitment, hint) = PCS::commit(&poly, &self.preprocessing.generators);
        self.transcript.append_serializable(&commitment);

        self.advice.untrusted_advice_polynomial = Some(poly);
        self.advice.untrusted_advice_hint = Some(hint);

        Some(commitment)
    }

    fn generate_and_commit_trusted_advice(&mut self) {
        if self.program_io.trusted_advice.is_empty() {
            return;
        }

        let mut trusted_advice_vec =
            vec![0; self.program_io.memory_layout.max_trusted_advice_size as usize / 8];

        populate_memory_states(
            0,
            &self.program_io.trusted_advice,
            Some(&mut trusted_advice_vec),
            None,
        );

        let poly = MultilinearPolynomial::from(trusted_advice_vec);
        self.advice.trusted_advice_polynomial = Some(poly);
        self.transcript
            .append_serializable(self.advice.trusted_advice_commitment.as_ref().unwrap());
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage1(
        &mut self,
    ) -> (
        UniSkipFirstRoundProof<F, ProofTranscript>,
        SumcheckInstanceProof<F, ProofTranscript>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 1 baseline");

        tracing::info!("Stage 1 proving");
        let uni_skip_params = OuterUniSkipParams::new(&self.spartan_key, &mut self.transcript);
        let mut uni_skip = OuterUniSkipProver::initialize(
            uni_skip_params.clone(),
            &self.trace,
            &self.preprocessing.program,
        );
        let first_round_proof = prove_uniskip_round(
            &mut uni_skip,
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        // Every sum-check with num_rounds > 1 requires a schedule
        // which dictates the compute_message and bind methods.
        // Using LinearOnlySchedule to benchmark linear-only mode (no streaming).
        // Outer remaining sumcheck has degree 3 (multiquadratic)
        // Number of rounds = tau.len() - 1 (cycle variables only)
        let schedule = LinearOnlySchedule::new(uni_skip_params.tau.len() - 1);
        let shared = OuterSharedState::new(
            Arc::clone(&self.trace),
            &self.preprocessing.program,
            &uni_skip_params,
            &self.opening_accumulator,
        );
        let mut spartan_outer_remaining: OuterRemainingStreamingSumcheck<_, _> =
            OuterRemainingStreamingSumcheck::new(shared, schedule);

        let (sumcheck_proof, _r_stage1) = BatchedSumcheck::prove(
            vec![&mut spartan_outer_remaining],
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        (first_round_proof, sumcheck_proof)
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage2(
        &mut self,
    ) -> (
        UniSkipFirstRoundProof<F, ProofTranscript>,
        SumcheckInstanceProof<F, ProofTranscript>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 2 baseline");

        // Stage 2a: Prove univariate-skip first round for product virtualization
        let uni_skip_params =
            ProductVirtualUniSkipParams::new(&self.opening_accumulator, &mut self.transcript);
        let mut uni_skip =
            ProductVirtualUniSkipProver::initialize(uni_skip_params.clone(), &self.trace);
        let first_round_proof = prove_uniskip_round(
            &mut uni_skip,
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        // Initialization params
        let spartan_product_virtual_remainder_params = ProductVirtualRemainderParams::new(
            self.trace.len(),
            uni_skip_params,
            &self.opening_accumulator,
        );
        let ram_raf_evaluation_params = RafEvaluationSumcheckParams::new(
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &self.opening_accumulator,
        );
        let ram_read_write_checking_params = RamReadWriteCheckingParams::new(
            &self.opening_accumulator,
            &mut self.transcript,
            &self.one_hot_params,
            self.trace.len(),
            &self.rw_config,
        );
        let ram_output_check_params = OutputSumcheckParams::new(
            self.one_hot_params.ram_k,
            &self.program_io,
            &mut self.transcript,
        );
        let instruction_claim_reduction_params =
            InstructionLookupsClaimReductionSumcheckParams::new(
                self.trace.len(),
                &self.opening_accumulator,
                &mut self.transcript,
            );

        // Initialization
        let spartan_product_virtual_remainder = ProductVirtualRemainderProver::initialize(
            spartan_product_virtual_remainder_params,
            Arc::clone(&self.trace),
        );
        let ram_raf_evaluation = RamRafEvaluationSumcheckProver::initialize(
            ram_raf_evaluation_params,
            &self.trace,
            &self.program_io.memory_layout,
        );
        let ram_read_write_checking = RamReadWriteCheckingProver::initialize(
            ram_read_write_checking_params,
            &self.trace,
            &self.preprocessing.program,
            &self.program_io.memory_layout,
            &self.initial_ram_state,
        );
        let ram_output_check = OutputSumcheckProver::initialize(
            ram_output_check_params,
            &self.initial_ram_state,
            &self.final_ram_state,
            &self.program_io.memory_layout,
        );
        let instruction_claim_reduction =
            InstructionLookupsClaimReductionSumcheckProver::initialize(
                instruction_claim_reduction_params,
                Arc::clone(&self.trace),
            );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "ProductVirtualRemainderProver",
                &spartan_product_virtual_remainder,
            );
            print_data_structure_heap_usage("RamRafEvaluationSumcheckProver", &ram_raf_evaluation);
            print_data_structure_heap_usage("RamReadWriteCheckingProver", &ram_read_write_checking);
            print_data_structure_heap_usage("OutputSumcheckProver", &ram_output_check);
            print_data_structure_heap_usage(
                "InstructionLookupsClaimReductionSumcheckProver",
                &instruction_claim_reduction,
            );
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(spartan_product_virtual_remainder),
            Box::new(ram_raf_evaluation),
            Box::new(ram_read_write_checking),
            Box::new(ram_output_check),
            Box::new(instruction_claim_reduction),
        ];

        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage2_start_flamechart.svg");
        tracing::info!("Stage 2 proving");
        let (sumcheck_proof, _r_stage2) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage2_end_flamechart.svg");
        drop_in_background_thread(instances);

        (first_round_proof, sumcheck_proof)
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage3(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 3 baseline");

        // Initialization params
        let spartan_shift_params = ShiftSumcheckParams::new(
            self.trace.len().log_2(),
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let spartan_instruction_input_params =
            InstructionInputParams::new(&self.opening_accumulator, &mut self.transcript);
        let spartan_registers_claim_reduction_params = RegistersClaimReductionSumcheckParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // Initialize
        let spartan_shift = ShiftSumcheckProver::initialize(
            spartan_shift_params,
            Arc::clone(&self.trace),
            &self.preprocessing.program,
        );
        let spartan_instruction_input = InstructionInputSumcheckProver::initialize(
            spartan_instruction_input_params,
            &self.trace,
            &self.opening_accumulator,
        );
        let spartan_registers_claim_reduction = RegistersClaimReductionSumcheckProver::initialize(
            spartan_registers_claim_reduction_params,
            Arc::clone(&self.trace),
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("ShiftSumcheckProver", &spartan_shift);
            print_data_structure_heap_usage(
                "InstructionInputSumcheckProver",
                &spartan_instruction_input,
            );
            print_data_structure_heap_usage(
                "RegistersClaimReductionSumcheckProver",
                &spartan_registers_claim_reduction,
            );
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(spartan_shift),
            Box::new(spartan_instruction_input),
            Box::new(spartan_registers_claim_reduction),
        ];

        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage3_start_flamechart.svg");
        tracing::info!("Stage 3 proving");
        let (sumcheck_proof, _r_stage3) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage3_end_flamechart.svg");
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage4(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 4 baseline");

        prover_accumulate_advice(
            &self.advice.untrusted_advice_polynomial,
            &self.advice.trusted_advice_polynomial,
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &mut self.opening_accumulator,
            &mut self.transcript,
            self.rw_config
                .needs_single_advice_opening(self.trace.len().log_2()),
        );
        if self.program_mode == ProgramMode::Committed {
            let trusted = self
                .preprocessing
                .program_commitments
                .as_ref()
                .expect("program commitments missing in committed mode");
            prover_accumulate_program_image::<F>(
                self.one_hot_params.ram_k,
                self.preprocessing.program.min_bytecode_address,
                &self.preprocessing.program.program_image_words,
                &self.program_io,
                trusted.program_image_num_words,
                &mut self.opening_accumulator,
                &mut self.transcript,
                self.rw_config
                    .needs_single_advice_opening(self.trace.len().log_2()),
            );
        }

        let registers_read_write_checking_params = RegistersReadWriteCheckingParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
            &self.rw_config,
        );
        let ram_val_evaluation_params = ValEvaluationSumcheckParams::new_from_prover(
            &self.one_hot_params,
            &self.opening_accumulator,
            &self.initial_ram_state,
            self.trace.len(),
        );
        let ram_val_final_params =
            ValFinalSumcheckParams::new_from_prover(self.trace.len(), &self.opening_accumulator);

        let registers_read_write_checking = RegistersReadWriteCheckingProver::initialize(
            registers_read_write_checking_params,
            self.trace.clone(),
            &self.preprocessing.program,
            &self.program_io.memory_layout,
        );
        let ram_val_evaluation = RamValEvaluationSumcheckProver::initialize(
            ram_val_evaluation_params,
            &self.trace,
            &self.preprocessing.program,
            &self.program_io.memory_layout,
        );
        let ram_val_final = ValFinalSumcheckProver::initialize(
            ram_val_final_params,
            &self.trace,
            &self.preprocessing.program,
            &self.program_io.memory_layout,
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "RegistersReadWriteCheckingProver",
                &registers_read_write_checking,
            );
            print_data_structure_heap_usage("RamValEvaluationSumcheckProver", &ram_val_evaluation);
            print_data_structure_heap_usage("ValFinalSumcheckProver", &ram_val_final);
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(registers_read_write_checking),
            Box::new(ram_val_evaluation),
            Box::new(ram_val_final),
        ];

        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage4_start_flamechart.svg");
        tracing::info!("Stage 4 proving");
        let (sumcheck_proof, _r_stage4) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage4_end_flamechart.svg");
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage5(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 5 baseline");
        let registers_val_evaluation_params =
            RegistersValEvaluationSumcheckParams::new(&self.opening_accumulator);
        let ram_ra_reduction_params = RaReductionParams::new(
            self.trace.len(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_read_raf_params = InstructionReadRafSumcheckParams::new(
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let registers_val_evaluation = RegistersValEvaluationSumcheckProver::initialize(
            registers_val_evaluation_params,
            &self.trace,
            &self.preprocessing.program,
            &self.program_io.memory_layout,
        );
        let ram_ra_reduction = RamRaClaimReductionSumcheckProver::initialize(
            ram_ra_reduction_params,
            &self.trace,
            &self.program_io.memory_layout,
            &self.one_hot_params,
        );
        let lookups_read_raf = InstructionReadRafSumcheckProver::initialize(
            lookups_read_raf_params,
            Arc::clone(&self.trace),
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "RegistersValEvaluationSumcheckProver",
                &registers_val_evaluation,
            );
            print_data_structure_heap_usage("RamRaClaimReductionSumcheckProver", &ram_ra_reduction);
            print_data_structure_heap_usage("InstructionReadRafSumcheckProver", &lookups_read_raf);
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(registers_val_evaluation),
            Box::new(ram_ra_reduction),
            Box::new(lookups_read_raf),
        ];

        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage5_start_flamechart.svg");
        tracing::info!("Stage 5 proving");
        let (sumcheck_proof, _r_stage5) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage5_end_flamechart.svg");
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage6a(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 6a baseline");

        let mut bytecode_read_raf_params = BytecodeReadRafSumcheckParams::gen(
            &self.preprocessing.program,
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        bytecode_read_raf_params.use_staged_val_claims =
            self.program_mode == ProgramMode::Committed;

        let booleanity_params = BooleanitySumcheckParams::new(
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let mut bytecode_read_raf = BytecodeReadRafAddressSumcheckProver::initialize(
            bytecode_read_raf_params.clone(),
            Arc::clone(&self.trace),
            Arc::clone(&self.preprocessing.program),
        );
        let mut booleanity = BooleanityAddressSumcheckProver::initialize(
            booleanity_params.clone(),
            &self.trace,
            &self.preprocessing.program,
            &self.program_io.memory_layout,
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "BytecodeReadRafAddressSumcheckProver",
                &bytecode_read_raf,
            );
            print_data_structure_heap_usage("BooleanityAddressSumcheckProver", &booleanity);
        }

        let mut instances: Vec<&mut dyn SumcheckInstanceProver<_, _>> =
            vec![&mut bytecode_read_raf, &mut booleanity];

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage6a_start_flamechart.svg");
        tracing::info!("Stage 6a proving");
        let (sumcheck_proof, _r_stage6a) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage6a_end_flamechart.svg");

        // Cache params for Stage 6b
        self.bytecode_read_raf_params = Some(bytecode_read_raf_params);
        self.booleanity_params = Some(booleanity_params);

        sumcheck_proof
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage6b(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 6b baseline");

        let bytecode_read_raf_params = self
            .bytecode_read_raf_params
            .take()
            .expect("bytecode_read_raf_params must be set by prove_stage6a");
        let booleanity_params = self
            .booleanity_params
            .take()
            .expect("booleanity_params must be set by prove_stage6a");

        let ram_hamming_booleanity_params =
            HammingBooleanitySumcheckParams::new(&self.opening_accumulator);

        let ram_ra_virtual_params = RamRaVirtualParams::new(
            self.trace.len(),
            &self.one_hot_params,
            &self.opening_accumulator,
        );
        let lookups_ra_virtual_params = InstructionRaSumcheckParams::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let inc_reduction_params = IncClaimReductionSumcheckParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // Bytecode claim reduction (Phase 1 in Stage 6b): consumes Val_s(r_bc) from Stage 6a and
        // caches an intermediate claim for Stage 7.
        if self.program_mode == ProgramMode::Committed {
            let bytecode_reduction_params = BytecodeClaimReductionParams::new(
                &bytecode_read_raf_params,
                &self.opening_accumulator,
                &mut self.transcript,
            );
            self.bytecode_reduction_prover = Some(BytecodeClaimReductionProver::initialize(
                bytecode_reduction_params,
                Arc::clone(&self.preprocessing.program),
            ));
        } else {
            // Legacy mode: do not run the bytecode claim reduction.
            self.bytecode_reduction_prover = None;
        }

        // Advice claim reduction (Phase 1 in Stage 6b): trusted and untrusted are separate instances.
        if self.advice.trusted_advice_polynomial.is_some() {
            let trusted_advice_params = AdviceClaimReductionParams::new(
                AdviceKind::Trusted,
                &self.program_io.memory_layout,
                self.trace.len(),
                &self.opening_accumulator,
                &mut self.transcript,
                self.rw_config
                    .needs_single_advice_opening(self.trace.len().log_2()),
            );
            // Note: We clone the advice polynomial here because Stage 8 needs the original polynomial
            // A future optimization could use Arc<MultilinearPolynomial> with copy-on-write.
            self.advice_reduction_prover_trusted = {
                let poly = self
                    .advice
                    .trusted_advice_polynomial
                    .clone()
                    .expect("trusted advice params exist but polynomial is missing");
                Some(AdviceClaimReductionProver::initialize(
                    trusted_advice_params,
                    poly,
                ))
            };
        }

        if self.advice.untrusted_advice_polynomial.is_some() {
            let untrusted_advice_params = AdviceClaimReductionParams::new(
                AdviceKind::Untrusted,
                &self.program_io.memory_layout,
                self.trace.len(),
                &self.opening_accumulator,
                &mut self.transcript,
                self.rw_config
                    .needs_single_advice_opening(self.trace.len().log_2()),
            );
            // Note: We clone the advice polynomial here because Stage 8 needs the original polynomial
            // A future optimization could use Arc<MultilinearPolynomial> with copy-on-write.
            self.advice_reduction_prover_untrusted = {
                let poly = self
                    .advice
                    .untrusted_advice_polynomial
                    .clone()
                    .expect("untrusted advice params exist but polynomial is missing");
                Some(AdviceClaimReductionProver::initialize(
                    untrusted_advice_params,
                    poly,
                ))
            };
        }

        // Initialize Stage 6b cycle provers from scratch (Option B).
        let mut bytecode_read_raf = BytecodeReadRafCycleSumcheckProver::initialize(
            bytecode_read_raf_params,
            Arc::clone(&self.trace),
            Arc::clone(&self.preprocessing.program),
            &self.opening_accumulator,
        );
        let mut booleanity = BooleanityCycleSumcheckProver::initialize(
            booleanity_params,
            &self.trace,
            &self.preprocessing.program,
            &self.program_io.memory_layout,
            &self.opening_accumulator,
        );
        let mut ram_hamming_booleanity =
            HammingBooleanitySumcheckProver::initialize(ram_hamming_booleanity_params, &self.trace);

        let mut ram_ra_virtual = RamRaVirtualSumcheckProver::initialize(
            ram_ra_virtual_params,
            &self.trace,
            &self.program_io.memory_layout,
            &self.one_hot_params,
        );
        let mut lookups_ra_virtual =
            LookupsRaSumcheckProver::initialize(lookups_ra_virtual_params, &self.trace);
        let mut inc_reduction =
            IncClaimReductionSumcheckProver::initialize(inc_reduction_params, self.trace.clone());

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "BytecodeReadRafCycleSumcheckProver",
                &bytecode_read_raf,
            );
            print_data_structure_heap_usage(
                "ram HammingBooleanitySumcheckProver",
                &ram_hamming_booleanity,
            );
            print_data_structure_heap_usage("BooleanityCycleSumcheckProver", &booleanity);
            print_data_structure_heap_usage("RamRaSumcheckProver", &ram_ra_virtual);
            print_data_structure_heap_usage("LookupsRaSumcheckProver", &lookups_ra_virtual);
            print_data_structure_heap_usage("IncClaimReductionSumcheckProver", &inc_reduction);
            if let Some(ref advice) = self.advice_reduction_prover_trusted {
                print_data_structure_heap_usage("AdviceClaimReductionProver(trusted)", advice);
            }
            if let Some(ref advice) = self.advice_reduction_prover_untrusted {
                print_data_structure_heap_usage("AdviceClaimReductionProver(untrusted)", advice);
            }
        }

        let mut instances: Vec<&mut dyn SumcheckInstanceProver<_, _>> = vec![
            &mut bytecode_read_raf,
            &mut ram_hamming_booleanity,
            &mut booleanity,
            &mut ram_ra_virtual,
            &mut lookups_ra_virtual,
            &mut inc_reduction,
        ];
        if let Some(bytecode) = self.bytecode_reduction_prover.as_mut() {
            instances.push(bytecode);
        }
        if let Some(advice) = self.advice_reduction_prover_trusted.as_mut() {
            instances.push(advice);
        }
        if let Some(advice) = self.advice_reduction_prover_untrusted.as_mut() {
            instances.push(advice);
        }
        // Program-image claim reduction (Stage 6b): binds staged Stage 4 program-image scalar claims
        // to the trusted commitment via a degree-2 sumcheck, caching an opening of ProgramImageInit.
        let mut program_image_reduction: Option<ProgramImageClaimReductionProver<F>> = None;
        if self.program_mode == ProgramMode::Committed {
            let trusted = self
                .preprocessing
                .program_commitments
                .as_ref()
                .expect("program commitments missing in committed mode");
            let padded_len_words = trusted.program_image_num_words;
            let log_t = self.trace.len().log_2();
            let m = padded_len_words.log_2();
            assert!(
                m <= log_t,
                "program-image claim reduction requires m=log2(padded_len_words) <= log_T (got m={m}, log_T={log_t})"
            );
            let params = ProgramImageClaimReductionParams::new(
                &self.program_io,
                self.preprocessing.program.min_bytecode_address,
                padded_len_words,
                self.one_hot_params.ram_k,
                self.trace.len(),
                &self.rw_config,
                &self.opening_accumulator,
                &mut self.transcript,
            );
            // Build padded coefficients for ProgramWord polynomial.
            let mut coeffs = self.preprocessing.program.program_image_words.clone();
            coeffs.resize(padded_len_words, 0u64);
            program_image_reduction =
                Some(ProgramImageClaimReductionProver::initialize(params, coeffs));
        }
        if let Some(ref mut prog) = program_image_reduction {
            instances.push(prog);
        }

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage6b_start_flamechart.svg");
        tracing::info!("Stage 6b proving");

        let (sumcheck_proof, _r_stage6b) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage6b_end_flamechart.svg");
        drop_in_background_thread(bytecode_read_raf);
        drop_in_background_thread(ram_hamming_booleanity);
        drop_in_background_thread(booleanity);
        drop_in_background_thread(ram_ra_virtual);
        drop_in_background_thread(lookups_ra_virtual);
        drop_in_background_thread(inc_reduction);

        if let Some(prog) = program_image_reduction {
            drop_in_background_thread(prog);
        }

        sumcheck_proof
    }

    /// Stage 7: HammingWeight + ClaimReduction sumcheck (only log_k_chunk rounds).
    #[tracing::instrument(skip_all)]
    fn prove_stage7(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        // Create params and prover for HammingWeightClaimReduction
        // (r_cycle and r_addr_bool are extracted from Booleanity opening internally)
        let hw_params = HammingWeightClaimReductionParams::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let hw_prover = HammingWeightClaimReductionProver::initialize(
            hw_params,
            &self.trace,
            &self.preprocessing.shared,
            &self.preprocessing.program,
            &self.one_hot_params,
        );

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("HammingWeightClaimReductionProver", &hw_prover);

        // Run Stage 7 batched sumcheck (address rounds only).
        // Includes HammingWeightClaimReduction plus lane/address-phase reductions (if needed).
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> =
            vec![Box::new(hw_prover)];

        if let Some(mut bytecode_reduction_prover) = self.bytecode_reduction_prover.take() {
            bytecode_reduction_prover.params.phase = BytecodeReductionPhase::LaneVariables;
            instances.push(Box::new(bytecode_reduction_prover));
        }

        if let Some(mut advice_reduction_prover_trusted) =
            self.advice_reduction_prover_trusted.take()
        {
            if advice_reduction_prover_trusted
                .params
                .num_address_phase_rounds()
                > 0
            {
                // Transition phase
                advice_reduction_prover_trusted.params.phase = ReductionPhase::AddressVariables;
                instances.push(Box::new(advice_reduction_prover_trusted));
            }
        }
        if let Some(mut advice_reduction_prover_untrusted) =
            self.advice_reduction_prover_untrusted.take()
        {
            if advice_reduction_prover_untrusted
                .params
                .num_address_phase_rounds()
                > 0
            {
                // Transition phase
                advice_reduction_prover_untrusted.params.phase = ReductionPhase::AddressVariables;
                instances.push(Box::new(advice_reduction_prover_untrusted));
            }
        }

        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage7_start_flamechart.svg");
        tracing::info!("Stage 7 proving");
        let (sumcheck_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_boxed_instance_flamegraph_svg(&instances, "stage7_end_flamechart.svg");
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    /// Stage 8: Dory batch opening proof.
    /// Builds streaming RLC polynomial directly from trace (no witness regeneration needed).
    /// Stage 8: Dory batch opening proof with optional PCS recursion hint generation
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all)]
    fn prove_stage8(
        &mut self,
        opening_proof_hints: HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
        generate_recursion_hint: bool,
    ) -> (
        PCS::Proof,
        Option<<PCS as RecursionExt<F>>::Hint>,
        Option<Stage8ProofData<F, PCS, ProofTranscript>>,
        Option<Fq12>, // combine_hint for homomorphic combining offload
    )
    where
        PCS: RecursionExt<F>,
    {
        tracing::info!(
            "Stage 8 proving (Dory batch opening) - recursion hint: {}",
            generate_recursion_hint
        );

        let _guard = tracing::info_span!("stage8_init_dory_context").in_scope(|| {
            let start = Instant::now();
            let guard = if self.program_mode == ProgramMode::Committed {
                let committed = self
                    .preprocessing
                    .program_commitments
                    .as_ref()
                    .expect("program commitments missing in committed mode");
                DoryGlobals::initialize_main_context_with_num_columns(
                    self.one_hot_params.k_chunk,
                    self.padded_trace_len,
                    committed.bytecode_num_columns,
                    Some(DoryGlobals::get_layout()),
                )
            } else {
                DoryGlobals::initialize_context(
                    self.one_hot_params.k_chunk,
                    self.padded_trace_len,
                    DoryContext::Main,
                    Some(DoryGlobals::get_layout()),
                )
            };
            tracing::info!(
                elapsed_ms = start.elapsed().as_millis(),
                k_chunk = self.one_hot_params.k_chunk,
                padded_trace_len = self.padded_trace_len,
                "Initialized DoryGlobals for Stage 8"
            );
            guard
        });

        // Get the unified opening point from HammingWeightClaimReduction
        // This contains (r_address_stage7 || r_cycle_stage6) in big-endian
        let (opening_point, log_k_chunk) = tracing::info_span!("stage8_get_unified_opening_point")
            .in_scope(|| {
                let start = Instant::now();
                let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(0),
                    SumcheckId::HammingWeightClaimReduction,
                );
                let log_k_chunk = self.one_hot_params.log_k_chunk;
                tracing::info!(
                    elapsed_ms = start.elapsed().as_millis(),
                    log_k_chunk,
                    opening_point_vars = opening_point.r.len(),
                    "Fetched unified opening point for Stage 8"
                );
                (opening_point, log_k_chunk)
            });
        let r_address_stage7 = &opening_point.r[..log_k_chunk];

        // 1. Collect all (polynomial, claim) pairs
        let mut polynomial_claims = Vec::new();

        // Dense polynomials: RamInc and RdInc (from IncClaimReduction in Stage 6)
        // These are at r_cycle_stage6 only (length log_T)
        let ((_ram_inc_point, ram_inc_claim), (_rd_inc_point, rd_inc_claim)) =
            tracing::info_span!("stage8_collect_dense_inc_claims").in_scope(|| {
                let start = Instant::now();
                let ram = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::RamInc,
                    SumcheckId::IncClaimReduction,
                );
                let rd = self.opening_accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::RdInc,
                    SumcheckId::IncClaimReduction,
                );
                tracing::info!(
                    elapsed_ms = start.elapsed().as_millis(),
                    "Collected dense Inc claims (RamInc/RdInc)"
                );
                (ram, rd)
            });

        #[cfg(test)]
        {
            // Verify that Inc openings are at the same point as r_cycle from HammingWeightClaimReduction
            let r_cycle_stage6 = &opening_point.r[log_k_chunk..];

            debug_assert_eq!(
                _ram_inc_point.r.as_slice(),
                r_cycle_stage6,
                "RamInc opening point should match r_cycle from HammingWeightClaimReduction"
            );
            debug_assert_eq!(
                _rd_inc_point.r.as_slice(),
                r_cycle_stage6,
                "RdInc opening point should match r_cycle from HammingWeightClaimReduction"
            );
        }

        // Apply Lagrange factor for dense polys: ∏_{i<log_k_chunk} (1 - r_address[i])
        // Because dense polys have fewer variables, we need to account for this
        // Note: r_address is in big-endian, Lagrange factor uses ∏(1 - r_i)
        let lagrange_factor: F =
            tracing::info_span!("stage8_dense_lagrange_factor").in_scope(|| {
                let start = Instant::now();
                let lf: F = r_address_stage7.iter().map(|r| F::one() - *r).product();
                tracing::info!(
                    elapsed_ms = start.elapsed().as_millis(),
                    log_k_chunk,
                    "Computed dense Lagrange factor for embedding"
                );
                lf
            });

        polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
        polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));

        // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
        // These are at (r_address_stage7, r_cycle_stage6)
        tracing::info_span!("stage8_collect_sparse_ra_claims").in_scope(|| {
            let start = Instant::now();
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
            tracing::info!(
                elapsed_ms = start.elapsed().as_millis(),
                instruction_d = self.one_hot_params.instruction_d,
                bytecode_d = self.one_hot_params.bytecode_d,
                ram_d = self.one_hot_params.ram_d,
                "Collected sparse RA claims"
            );
        });

        // Advice polynomials: TrustedAdvice and UntrustedAdvice (from AdviceClaimReduction in Stage 6)
        // These are committed with smaller dimensions, so we apply Lagrange factors to embed
        // them in the top-left block of the main Dory matrix.
        tracing::info_span!("stage8_collect_advice_claims").in_scope(|| {
            let start = Instant::now();
            if let Some((advice_point, advice_claim)) = self
                .opening_accumulator
                .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReduction)
            {
                #[cfg(test)]
                {
                    let advice_poly = self.advice.trusted_advice_polynomial.as_ref().unwrap();
                    let expected_eval = advice_poly.evaluate(&advice_point.r);
                    assert_eq!(expected_eval, advice_claim);
                }
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
                #[cfg(test)]
                {
                    let advice_poly = self.advice.untrusted_advice_polynomial.as_ref().unwrap();
                    let expected_eval = advice_poly.evaluate(&advice_point.r);
                    assert_eq!(expected_eval, advice_claim);
                }
                let lagrange_factor =
                    compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
                polynomial_claims.push((
                    CommittedPolynomial::UntrustedAdvice,
                    advice_claim * lagrange_factor,
                ));
            }

            tracing::info!(
                elapsed_ms = start.elapsed().as_millis(),
                total_claims = polynomial_claims.len(),
                "Collected advice claims (if present)"
            );
        });

        // Bytecode chunk polynomials: committed in Bytecode context and embedded into the
        // main opening point by fixing the extra cycle variables to 0.
        if self.program_mode == ProgramMode::Committed {
            let (bytecode_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeChunk(0),
                SumcheckId::BytecodeClaimReduction,
            );
            let log_t = opening_point.r.len() - log_k_chunk;
            let log_k = bytecode_point.r.len() - log_k_chunk;
            assert!(
                log_k <= log_t,
                "bytecode folding requires log_T >= log_K (got log_T={log_t}, log_K={log_k})"
            );
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

        // 2. Sample gamma and compute powers for RLC
        let (claims, gamma_powers) =
            tracing::info_span!("stage8_sample_gamma_powers").in_scope(|| {
                let start = Instant::now();
                let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
                self.transcript.append_scalars(&claims);
                let gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());
                tracing::info!(
                    elapsed_ms = start.elapsed().as_millis(),
                    num_claims = claims.len(),
                    "Sampled gamma powers for Stage 8 RLC"
                );
                (claims, gamma_powers)
            });

        // Build DoryOpeningState
        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers: gamma_powers.clone(),
            polynomial_claims,
        };

        let streaming_data = Arc::new(RLCStreamingData {
            program: Arc::clone(&self.preprocessing.program),
            memory_layout: self.preprocessing.shared.memory_layout.clone(),
        });

        // Build advice polynomials map for RLC
        let mut advice_polys = HashMap::new();
        if let Some(poly) = self.advice.trusted_advice_polynomial.take() {
            advice_polys.insert(CommittedPolynomial::TrustedAdvice, poly);
        }
        if let Some(poly) = self.advice.untrusted_advice_polynomial.take() {
            advice_polys.insert(CommittedPolynomial::UntrustedAdvice, poly);
        }
        if self.program_mode == ProgramMode::Committed {
            let trusted = self
                .preprocessing
                .program_commitments
                .as_ref()
                .expect("program commitments missing in committed mode");
            // Use the padded size from the trusted commitments (may be larger than program's own padded size)
            let program_image_poly =
                TrustedProgramCommitments::<PCS>::build_program_image_polynomial_padded::<F>(
                    &self.preprocessing.program,
                    trusted.program_image_num_words,
                );
            advice_polys.insert(
                CommittedPolynomial::ProgramImageInit,
                MultilinearPolynomial::from(program_image_poly),
            );
        }

        // Build streaming RLC polynomial directly (no witness poly regeneration!)
        // Use materialized trace (default, single pass) instead of lazy trace
        //
        // bytecode_T: The T value used for bytecode coefficient indexing.
        // In Committed mode, use the value stored in trusted commitments.
        // In Full mode, use bytecode_len (original behavior).
        let bytecode_T = if self.program_mode == ProgramMode::Committed {
            let trusted = self
                .preprocessing
                .program_commitments
                .as_ref()
                .expect("program commitments missing in committed mode");
            trusted.bytecode_T
        } else {
            self.preprocessing.program.bytecode_len()
        };
        let (joint_poly, hint) = tracing::info_span!("stage8_build_streaming_rlc").in_scope(|| {
            let start = Instant::now();
            let out = state.build_streaming_rlc::<PCS>(
                self.one_hot_params.clone(),
                TraceSource::Materialized(Arc::clone(&self.trace)),
                streaming_data,
                opening_proof_hints,
                advice_polys,
                bytecode_T,
            );
            tracing::info!(
                elapsed_ms = start.elapsed().as_millis(),
                "Built streaming RLC polynomial for Stage 8"
            );
            out
        });

        // Ensure Dory is in the Main context for proving the joint opening.
        // (Some helper paths may temporarily switch contexts for advice handling.)
        let _ctx_main = DoryGlobals::with_context(DoryContext::Main);

        // Fork the transcript state if we need to generate recursion hint
        let witness_gen_transcript = if generate_recursion_hint {
            Some(self.transcript.clone())
        } else {
            None
        };

        // Dory opening proof at the unified point
        let opening_proof = tracing::info_span!("stage8_pcs_prove").in_scope(|| {
            let start = Instant::now();
            let proof = PCS::prove(
                &self.preprocessing.generators,
                &joint_poly,
                &opening_point.r,
                Some(hint),
                &mut self.transcript,
            );
            tracing::info!(
                elapsed_ms = start.elapsed().as_millis(),
                "Generated Stage 8 PCS opening proof"
            );
            proof
        });

        // Generate recursion hint if requested
        let (stage8_hint, proof_data, stage8_combine_hint) = if generate_recursion_hint {
            tracing::info_span!("stage8_generate_recursion_hint").in_scope(|| {
                // Compute claim: Σ γ_i · claim_i using shared utility
                let joint_claim = tracing::info_span!("stage8_joint_claim").in_scope(|| {
                    let start = Instant::now();
                    let jc = compute_joint_claim(&gamma_powers, &claims);
                    tracing::info!(
                        elapsed_ms = start.elapsed().as_millis(),
                        num_claims = claims.len(),
                        "Computed Stage 8 joint claim"
                    );
                    jc
                });

                // Compute joint commitment: Σ γ_i · C_i
                let (coeffs, comms) = tracing::info_span!("stage8_collect_rlc_coeffs_and_comms")
                    .in_scope(|| {
                        let start = Instant::now();

                        let mut commitments_map = HashMap::new();
                        let all_polys = all_committed_polynomials(&self.one_hot_params);
                        debug_assert_eq!(
                            all_polys.len(),
                            self.commitments.len(),
                            "commitment vector length mismatch"
                        );
                        for (poly, commitment) in all_polys.into_iter().zip(self.commitments.iter())
                        {
                            commitments_map.insert(poly, commitment.clone());
                        }

                        // Include advice commitments if they were folded into the Stage 8 RLC.
                        if state
                            .polynomial_claims
                            .iter()
                            .any(|(p, _)| *p == CommittedPolynomial::TrustedAdvice)
                        {
                            if let Some(ref commitment) = self.advice.trusted_advice_commitment {
                                commitments_map
                                    .insert(CommittedPolynomial::TrustedAdvice, commitment.clone());
                            }
                        }
                        if state
                            .polynomial_claims
                            .iter()
                            .any(|(p, _)| *p == CommittedPolynomial::UntrustedAdvice)
                        {
                            if let Some(ref commitment) = self.untrusted_advice_commitment {
                                commitments_map.insert(
                                    CommittedPolynomial::UntrustedAdvice,
                                    commitment.clone(),
                                );
                            }
                        }

                        // Include program commitments in committed mode (BytecodeChunk and ProgramImageInit)
                        if let Some(ref program_commitments) =
                            self.preprocessing.program_commitments
                        {
                            for (idx, commitment) in
                                program_commitments.bytecode_commitments.iter().enumerate()
                            {
                                commitments_map.insert(
                                    CommittedPolynomial::BytecodeChunk(idx),
                                    commitment.clone(),
                                );
                            }
                            commitments_map.insert(
                                CommittedPolynomial::ProgramImageInit,
                                program_commitments.program_image_commitment.clone(),
                            );
                        }

                        // Compute RLC coefficients using shared utility
                        let rlc_map = compute_rlc_coefficients(
                            &state.gamma_powers,
                            state.polynomial_claims.iter().cloned(),
                        );
                        let (coeffs, comms): (Vec<F>, Vec<PCS::Commitment>) = rlc_map
                            .into_iter()
                            .map(|(poly, coeff)| (coeff, commitments_map.remove(&poly).unwrap()))
                            .unzip();

                        tracing::info!(
                            elapsed_ms = start.elapsed().as_millis(),
                            num_terms = coeffs.len(),
                            "Collected RLC coeffs and commitments"
                        );
                        (coeffs, comms)
                    });

                // Generate combine witness for homomorphic combining offload
                let (combine_witness, combine_hint) =
                    tracing::info_span!("stage8_generate_combine_witness").in_scope(|| {
                        let start = Instant::now();
                        let out = PCS::generate_combine_witness(&comms, &coeffs);
                        tracing::info!(
                            elapsed_ms = start.elapsed().as_millis(),
                            exp_ops = out.0.exp_witnesses.len(),
                            mul_ops = out.0.mul_witnesses.len(),
                            "Generated combine witness for homomorphic combining"
                        );
                        out
                    });

                // combine_witness will be passed to Stage 9 via proof_data
                let joint_commitment =
                    tracing::info_span!("stage8_combine_with_hint").in_scope(|| {
                        let start = Instant::now();
                        let jc = PCS::combine_with_hint(&combine_hint);
                        tracing::info!(
                            elapsed_ms = start.elapsed().as_millis(),
                            "Computed joint commitment via combine hint"
                        );
                        jc
                    });

                // Generate a hint for verifying the opening proof (via the PCS's recursion extension).
                let verifier_setup = tracing::info_span!("stage8_setup_verifier").in_scope(|| {
                    let start = Instant::now();
                    let vs = PCS::setup_verifier(&self.preprocessing.generators);
                    tracing::info!(
                        elapsed_ms = start.elapsed().as_millis(),
                        "Derived PCS verifier setup for witness_gen"
                    );
                    vs
                });

                let (witnesses, stage8_hint) = tracing::info_span!("stage8_pcs_witness_gen")
                    .in_scope(|| {
                        let start = Instant::now();
                        let out = PCS::witness_gen(
                            &opening_proof,
                            &verifier_setup,
                            &mut witness_gen_transcript.clone().unwrap(),
                            &opening_point.r,
                            &joint_claim,
                            &joint_commitment,
                        )
                        .expect("PCS::witness_gen failed for Stage 8 opening proof");
                        tracing::info!(
                            elapsed_ms = start.elapsed().as_millis(),
                            "Generated recursion witness + hint for Stage 8 opening"
                        );
                        out
                    });

                // Create Stage8ProofData with all computed values
                let proof_data = tracing::info_span!("stage8_pack_proof_data").in_scope(|| {
                    let start = Instant::now();
                    let pd = Stage8ProofData {
                        gamma_powers: gamma_powers.clone(),
                        joint_claim,
                        joint_commitment: joint_commitment.clone(),
                        witness_gen_transcript: witness_gen_transcript.unwrap(),
                        opening_state: state.clone(),
                        witnesses,
                        combine_witness: Some(combine_witness),
                    };
                    tracing::info!(
                        elapsed_ms = start.elapsed().as_millis(),
                        "Packed Stage8ProofData"
                    );
                    pd
                });

                // Extract the Fq12 from combine_hint for serialization
                let combine_hint_fq12 = tracing::info_span!("stage8_combine_hint_to_fq12")
                    .in_scope(|| {
                        let start = Instant::now();
                        let x = PCS::combine_hint_to_fq12(&combine_hint);
                        tracing::info!(
                            elapsed_ms = start.elapsed().as_millis(),
                            "Converted combine hint to Fq12 for serialization"
                        );
                        x
                    });

                (Some(stage8_hint), Some(proof_data), Some(combine_hint_fq12))
            })
        } else {
            (None, None, None)
        };

        (opening_proof, stage8_hint, proof_data, stage8_combine_hint)
    }

    /// Consolidated recursion proof generation (Stages 9-13).
    ///
    /// This method consolidates the following stages:
    /// - Stage 9: Create RecursionProver from witnesses + sample gamma/delta
    /// - Stage 10: Extract constraint metadata
    /// - Stage 11: Build dense polynomial + commit (BEFORE sumchecks for soundness)
    /// - Stage 12: Run recursion sumchecks (stages 1-5 of RecursionProver)
    /// - Stage 13: Generate Hyrax opening proof
    ///
    /// Returns the recursion proof and constraint metadata for the verifier.
    #[tracing::instrument(skip_all, name = "prove_stage_recursive")]
    fn prove_stage_recursive(
        &mut self,
        stage8_proof_data: Stage8ProofData<F, PCS, ProofTranscript>,
    ) -> RecursionProofResult<ProofTranscript, Hyrax<1, GrumpkinProjective>>
    where
        PCS: RecursionExt<F>,
        <PCS as RecursionExt<F>>::Hint: Send + Sync + 'static,
    {
        tracing::info!("Starting consolidated recursion proof (Stages 9-13)");

        // ============ STAGE 9: Create RecursionProver ============
        // Sample gamma/delta challenges from the main transcript
        let (gamma, delta) = tracing::info_span!("sample_recursion_challenges").in_scope(|| {
            let g: Fq = self.transcript.challenge_scalar();
            let d: Fq = self.transcript.challenge_scalar();
            tracing::info!("Sampled gamma and delta challenges");
            (g, d)
        });

        // Verify type compatibility at runtime
        if std::any::TypeId::of::<F>() != std::any::TypeId::of::<Fr>() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Recursion SNARK requires F to be Fr (BN254 scalar field)",
            )));
        }

        // Optionally check PCS type
        let pcs_name = std::any::type_name::<PCS>();
        if !pcs_name.contains("DoryCommitmentScheme") {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Recursion SNARK requires DoryCommitmentScheme",
            )));
        }

        // Convert PCS witnesses to Dory witness collection type
        let witness_collection = tracing::info_span!("convert_witness_collection").in_scope(|| {
            // These unsafe transmutes are necessary because we're working with generic types
            let collection = unsafe {
                &*(&stage8_proof_data.witnesses as *const <PCS as RecursionExt<F>>::Witness
                    as *const dory::recursion::WitnessCollection<
                        crate::poly::commitment::dory::recursion::JoltWitness,
                    >)
            };
            tracing::info!("Converted PCS witnesses to Dory witness collection");
            collection
        });

        // Create RecursionProver using pre-generated witnesses
        let recursion_prover = tracing::info_span!("create_recursion_prover").in_scope(|| {
            let prover = RecursionProver::<Fq>::new_from_witnesses(
                witness_collection,
                stage8_proof_data.combine_witness.clone(),
                gamma,
                delta,
            )?;
            tracing::info!("Created RecursionProver from witnesses");
            Ok::<_, Box<dyn std::error::Error>>(prover)
        })?;

        // ============ STAGES 10-13: Delegate to RecursionProver::prove_full ============
        // This method handles:
        // - Stage 10: Build constraint metadata
        // - Stage 11: Build dense polynomial + commit (BEFORE sumchecks)
        // - Stage 12: Run all recursion sumchecks (stages 1-5)
        // - Stage 13: Generate Hyrax opening proof
        type HyraxPCS = Hyrax<1, GrumpkinProjective>;
        let hyrax_prover_setup = &self.preprocessing.hyrax_recursion_setup;

        let (recursion_proof, metadata) = tracing::info_span!("recursion_prover_prove_full")
            .in_scope(|| {
                recursion_prover
                    .prove_full::<ProofTranscript, HyraxPCS>(
                        &mut self.transcript,
                        hyrax_prover_setup,
                    )
                    .expect("Failed to run RecursionProver::prove_full")
            });

        tracing::info!("Completed consolidated recursion proof");
        Ok((recursion_proof, metadata))
    }
}

pub struct JoltAdvice<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub untrusted_advice_polynomial: Option<MultilinearPolynomial<F>>,
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub trusted_advice_polynomial: Option<MultilinearPolynomial<F>>,
    /// Hint for untrusted advice (for batched Dory opening)
    pub untrusted_advice_hint: Option<PCS::OpeningProofHint>,
    /// Hint for trusted advice (for batched Dory opening)
    pub trusted_advice_hint: Option<PCS::OpeningProofHint>,
}

#[cfg(feature = "allocative")]
fn write_boxed_instance_flamegraph_svg(
    instances: &[Box<dyn SumcheckInstanceProver<impl JoltField, impl Transcript>>],
    path: impl AsRef<Path>,
) {
    let mut flamegraph = FlameGraphBuilder::default();
    for instance in instances {
        instance.update_flamegraph(&mut flamegraph)
    }
    write_flamegraph_svg(flamegraph, path);
}

#[cfg(feature = "allocative")]
fn write_instance_flamegraph_svg(
    instances: &[&mut dyn SumcheckInstanceProver<impl JoltField, impl Transcript>],
    path: impl AsRef<Path>,
) {
    let mut flamegraph = FlameGraphBuilder::default();
    for instance in instances {
        instance.update_flamegraph(&mut flamegraph)
    }
    write_flamegraph_svg(flamegraph, path);
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProverPreprocessing<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub generators: PCS::ProverSetup,
    pub shared: JoltSharedPreprocessing,
    pub hyrax_recursion_setup: PedersenGenerators<GrumpkinProjective>,
    /// Full program preprocessing (prover always has full access for witness computation).
    pub program: Arc<ProgramPreprocessing>,
    /// Trusted program commitments (only in Committed mode).
    ///
    /// In Full mode: None (verifier has full program).
    /// In Committed mode: Some(trusted) for bytecode + program-image commitments.
    pub program_commitments: Option<TrustedProgramCommitments<PCS>>,
    /// Opening proof hints for program commitments (only in Committed mode).
    pub program_hints: Option<TrustedProgramHints<PCS>>,
}

impl<F, PCS> JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    /// Setup generators based on trace length (Main context).
    fn setup_generators(shared: &JoltSharedPreprocessing) -> PCS::ProverSetup {
        use common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T;
        let max_T: usize = shared.max_padded_trace_length.next_power_of_two();
        let max_log_T = max_T.log_2();
        // Use the maximum possible log_k_chunk for generator setup
        let max_log_k_chunk = if max_log_T < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };
        PCS::setup_prover(max_log_k_chunk + max_log_T)
    }

    /// Setup generators for Committed mode, ensuring capacity for both:
    /// - Main context up to `max_padded_trace_length`
    /// - Bytecode context up to `bytecode_size`
    /// - ProgramImage context up to the padded program-image word length
    fn setup_generators_committed(
        shared: &JoltSharedPreprocessing,
        program: &ProgramPreprocessing,
    ) -> PCS::ProverSetup {
        use common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T;
        let prog_len_words_padded = program.program_image_len_words_padded();
        let max_t_any: usize = shared
            .max_padded_trace_length
            .max(shared.bytecode_size())
            .max(prog_len_words_padded)
            .next_power_of_two();
        let max_log_t_any = max_t_any.log_2();
        let max_log_k_chunk = if max_log_t_any < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };
        PCS::setup_prover(max_log_k_chunk + max_log_t_any)
    }

    /// Create prover preprocessing in Full mode (no commitments).
    ///
    /// Use this when the verifier will have access to full program.
    #[tracing::instrument(skip_all, name = "JoltProverPreprocessing::new")]
    pub fn new(
        shared: JoltSharedPreprocessing,
        program: Arc<ProgramPreprocessing>,
    ) -> JoltProverPreprocessing<F, PCS> {
        let generators = Self::setup_generators(&shared);
        // Precompute Hyrax setup for recursion proving.
        type HyraxPCS = Hyrax<1, GrumpkinProjective>;
        let hyrax_recursion_setup =
            <HyraxPCS as CommitmentScheme>::setup_prover(MAX_RECURSION_DENSE_NUM_VARS);
        JoltProverPreprocessing {
            generators,
            shared,
            hyrax_recursion_setup,
            program,
            program_commitments: None,
            program_hints: None,
        }
    }

    /// Create prover preprocessing in Committed mode (with program commitments).
    ///
    /// Use this when the verifier should only receive commitments (succinct verification).
    /// Computes commitments + hints for all bytecode chunk polynomials and program image during preprocessing.
    #[tracing::instrument(skip_all, name = "JoltProverPreprocessing::new_committed")]
    pub fn new_committed(
        shared: JoltSharedPreprocessing,
        program: Arc<ProgramPreprocessing>,
    ) -> JoltProverPreprocessing<F, PCS> {
        let generators = Self::setup_generators_committed(&shared, &program);
        // Precompute Hyrax setup for recursion proving.
        type HyraxPCS = Hyrax<1, GrumpkinProjective>;
        let hyrax_recursion_setup =
            <HyraxPCS as CommitmentScheme>::setup_prover(MAX_RECURSION_DENSE_NUM_VARS);
        let max_t_any: usize = shared
            .max_padded_trace_length
            .max(shared.bytecode_size())
            .next_power_of_two();
        let max_log_t = max_t_any.log_2();
        let log_k_chunk = if max_log_t < common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };
        let (program_commitments, program_hints) =
            TrustedProgramCommitments::derive(&program, &generators, log_k_chunk, max_t_any);
        JoltProverPreprocessing {
            generators,
            shared,
            hyrax_recursion_setup,
            program,
            program_commitments: Some(program_commitments),
            program_hints: Some(program_hints),
        }
    }

    /// Check if this preprocessing is in Committed mode.
    pub fn is_committed_mode(&self) -> bool {
        self.program_commitments.is_some()
    }

    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_prover_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_prover_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> Serializable
    for JoltProverPreprocessing<F, PCS>
{
}
