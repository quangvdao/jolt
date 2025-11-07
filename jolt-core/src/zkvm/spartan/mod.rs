use std::sync::Arc;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::{
    OpeningAccumulator, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
};
use crate::subprotocols::sumcheck::UniSkipFirstRoundProof;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::univariate_skip::{prove_uniskip_round, UniSkipState};
use crate::transcripts::Transcript;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::zkvm::dag::stage::SumcheckStagesProver;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::r1cs::constraints::{
    OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::spartan::inner::InnerSumcheckProver;
use crate::zkvm::spartan::instruction_input::InstructionInputSumcheckProver;
use crate::zkvm::spartan::outer::{OuterRemainingSumcheckProver, OuterUniSkipInstanceProver};
use crate::zkvm::spartan::outer_baseline::OuterBaselineSumcheckProver;
use crate::zkvm::spartan::outer_round_batched::OuterRoundBatchedSumcheckProver;
use crate::zkvm::spartan::outer_streaming::OuterRemainingStreamingSumcheckProver;
use crate::zkvm::spartan::product::{
    ProductVirtualInnerProver, ProductVirtualRemainderProver, ProductVirtualUniSkipInstanceParams,
};
use crate::zkvm::spartan::shift::ShiftSumcheckProver;
use crate::zkvm::witness::VirtualPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::zkvm::r1cs::constraints::{R1CSConstraint, R1CS_CONSTRAINTS};
use crate::zkvm::r1cs::inputs::{JoltR1CSInputs, ALL_R1CS_INPUTS};
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::transcripts::AppendToTranscript;
use tracer::instruction::Cycle;
use ark_ff::biginteger::S128;

use product::{
    ProductVirtualUniSkipInstanceProver, PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
    PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
};

pub mod inner;
pub mod instruction_input;
pub mod outer;
pub mod outer_baseline;
pub mod outer_naive;
pub mod outer_round_batched;
pub mod outer_streaming;
pub mod product;
pub mod shift;

/// Select which outer-remaining implementation to use.
#[derive(Copy, Clone, Debug)]
pub enum OuterImpl {
    Current,
    Streaming,
    Baseline,
    RoundBatched,
}
/// Global selection for Spartan Stage 1 remainder.
pub const OUTER_IMPL: OuterImpl = OuterImpl::Streaming;

pub struct SpartanDagProver<F: JoltField> {
    /// Cached key to avoid recomputation across stages
    key: Arc<UniformSpartanKey<F>>,
    /// Handoff state from univariate skip first round (shared by prover and verifier)
    /// Consists of the `tau` vector for Lagrange / eq evals, the claim from univariate skip round,
    /// and the challenge r0 from the univariate skip round
    /// This is first used in stage 1 and then reused in stage 2
    uni_skip_state: Option<UniSkipState<F>>,
}

impl<F: JoltField> SpartanDagProver<F> {
    pub fn new(padded_trace_length: usize) -> Self {
        Self {
            key: Arc::new(UniformSpartanKey::new(padded_trace_length)),
            uni_skip_state: None,
        }
    }

    // Stage 1: Outer sumcheck with uni-skip first round
    pub fn stage1_uni_skip<T: Transcript>(
        &mut self,
        state_manager: &mut StateManager<'_, F, impl CommitmentScheme<Field = F>>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> UniSkipFirstRoundProof<F, T> {
        // For Baseline and RoundBatched, do not perform uni-skip: append a trivial zero polynomial.
        if matches!(OUTER_IMPL, OuterImpl::Baseline | OuterImpl::RoundBatched) {
            let zero_poly = crate::poly::unipoly::UniPoly::<F>::from_coeff(vec![F::zero()]);
            // Keep transcript alignment: append the polynomial and derive r0
            zero_poly.append_to_transcript(transcript);
            let _ = transcript.challenge_scalar_optimized::<F>();
            return UniSkipFirstRoundProof::new(zero_poly);
        }
        let num_rounds_x: usize = self.key.num_rows_bits();

        // Transcript and tau
        let tau = transcript.challenge_vector_optimized::<F>(num_rounds_x);

        // Prove uni-skip first round
        let mut uniskip_instance = OuterUniSkipInstanceProver::gen(state_manager, &tau);
        let (first_round_proof, r0, claim_after_first) =
            prove_uniskip_round(&mut uniskip_instance, transcript);

        self.uni_skip_state = Some(UniSkipState {
            claim_after_first,
            r0,
            tau,
        });

        first_round_proof
    }

    // Stage 2: Product virtualization uni-skip first round
    pub fn stage2_uni_skip<T: Transcript>(
        &mut self,
        state_manager: &mut StateManager<'_, F, impl CommitmentScheme<Field = F>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> UniSkipFirstRoundProof<F, T> {
        let num_cycle_vars: usize = self.key.num_cycle_vars();

        // Reuse r_cycle from Stage 1 (outer) for τ_low, and sample τ_high
        let r_cycle = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Product, SumcheckId::SpartanOuter)
            .0
            .r;
        debug_assert_eq!(r_cycle.len(), num_cycle_vars);
        let tau_high = transcript.challenge_scalar_optimized::<F>();
        let mut tau = r_cycle;
        tau.push(tau_high);

        let mut uniskip_instance =
            ProductVirtualUniSkipInstanceProver::gen(state_manager, opening_accumulator, &tau);
        let (first_round_proof, r0, claim_after_first) =
            prove_uniskip_round(&mut uniskip_instance, transcript);

        self.uni_skip_state = Some(UniSkipState {
            claim_after_first,
            r0,
            tau,
        });
        first_round_proof
    }
}

/// Build flattened polynomials (per R1CS input) from the trace, for baseline outer prover.
#[allow(clippy::too_many_lines)]
fn build_flattened_polynomials<F: JoltField>(
    preprocess: &BytecodePreprocessing,
    trace: &Vec<Cycle>,
) -> Vec<MultilinearPolynomial<F>> {
    let n = trace.len();

    // Pre-allocate per-input buffers
    let mut left_instruction_input: Vec<u64> = Vec::with_capacity(n);
    let mut right_instruction_input: Vec<i128> = Vec::with_capacity(n);
    let mut product: Vec<S128> = Vec::with_capacity(n);

    let mut left_lookup_operand: Vec<u64> = Vec::with_capacity(n);
    let mut right_lookup_operand: Vec<u128> = Vec::with_capacity(n);
    let mut lookup_output: Vec<u64> = Vec::with_capacity(n);

    let mut pc: Vec<u64> = Vec::with_capacity(n);
    let mut unexpanded_pc: Vec<u64> = Vec::with_capacity(n);
    let mut next_pc: Vec<u64> = Vec::with_capacity(n);
    let mut next_unexpanded_pc: Vec<u64> = Vec::with_capacity(n);

    let mut imm: Vec<i128> = Vec::with_capacity(n);

    let mut ram_addr: Vec<u64> = Vec::with_capacity(n);
    let mut ram_read_value: Vec<u64> = Vec::with_capacity(n);
    let mut ram_write_value: Vec<u64> = Vec::with_capacity(n);

    let mut rs1_read_value: Vec<u64> = Vec::with_capacity(n);
    let mut rs2_read_value: Vec<u64> = Vec::with_capacity(n);
    let mut rd_write_value: Vec<u64> = Vec::with_capacity(n);

    let mut write_lookup_output_to_rd_addr: Vec<bool> = Vec::with_capacity(n);
    let mut write_pc_to_rd_addr: Vec<bool> = Vec::with_capacity(n);
    let mut should_branch: Vec<bool> = Vec::with_capacity(n);
    let mut should_jump: Vec<bool> = Vec::with_capacity(n);
    let mut next_is_virtual: Vec<bool> = Vec::with_capacity(n);
    let mut next_is_first_in_sequence: Vec<bool> = Vec::with_capacity(n);

    // Per-flag buffers
    let mut opflag_vecs: [Vec<bool>; crate::zkvm::instruction::NUM_CIRCUIT_FLAGS] =
        std::array::from_fn(|_| Vec::with_capacity(n));

    // Single pass over the trace
    for t in 0..n {
        let row = crate::zkvm::r1cs::inputs::R1CSCycleInputs::from_trace::<F>(preprocess, trace, t);

        // Instruction inputs and product
        left_instruction_input.push(row.left_input);
        right_instruction_input.push(row.right_input.to_i128());
        product.push(row.product);

        // Lookup operands and output
        left_lookup_operand.push(row.left_lookup);
        right_lookup_operand.push(row.right_lookup);
        lookup_output.push(row.lookup_output);

        // Registers
        rs1_read_value.push(row.rs1_read_value);
        rs2_read_value.push(row.rs2_read_value);
        rd_write_value.push(row.rd_write_value);

        // RAM
        ram_addr.push(row.ram_addr);
        ram_read_value.push(row.ram_read_value);
        ram_write_value.push(row.ram_write_value);

        // PCs
        pc.push(row.pc);
        next_pc.push(row.next_pc);
        unexpanded_pc.push(row.unexpanded_pc);
        next_unexpanded_pc.push(row.next_unexpanded_pc);

        // Immediate
        imm.push(row.imm.to_i128());

        // Derived booleans
        write_lookup_output_to_rd_addr.push(row.write_lookup_output_to_rd_addr);
        write_pc_to_rd_addr.push(row.write_pc_to_rd_addr);
        should_branch.push(row.should_branch);
        should_jump.push(row.should_jump);
        next_is_virtual.push(row.next_is_virtual);
        next_is_first_in_sequence.push(row.next_is_first_in_sequence);

        // Op flags
        for idx in 0..crate::zkvm::instruction::NUM_CIRCUIT_FLAGS {
            opflag_vecs[idx].push(row.flags[idx]);
        }
    }

    // Assemble output in ALL_R1CS_INPUTS canonical order
    let mut out: Vec<MultilinearPolynomial<F>> = Vec::with_capacity(ALL_R1CS_INPUTS.len());
    for input in ALL_R1CS_INPUTS.iter() {
        match input {
            JoltR1CSInputs::LeftInstructionInput => out.push(left_instruction_input.clone().into()),
            JoltR1CSInputs::RightInstructionInput => out.push(right_instruction_input.clone().into()),
            JoltR1CSInputs::Product => out.push(product.clone().into()),
            JoltR1CSInputs::WriteLookupOutputToRD => {
                out.push(write_lookup_output_to_rd_addr.clone().into())
            }
            JoltR1CSInputs::WritePCtoRD => out.push(write_pc_to_rd_addr.clone().into()),
            JoltR1CSInputs::ShouldBranch => out.push(should_branch.clone().into()),
            JoltR1CSInputs::PC => out.push(pc.clone().into()),
            JoltR1CSInputs::UnexpandedPC => out.push(unexpanded_pc.clone().into()),
            JoltR1CSInputs::Imm => out.push(imm.clone().into()),
            JoltR1CSInputs::RamAddress => out.push(ram_addr.clone().into()),
            JoltR1CSInputs::Rs1Value => out.push(rs1_read_value.clone().into()),
            JoltR1CSInputs::Rs2Value => out.push(rs2_read_value.clone().into()),
            JoltR1CSInputs::RdWriteValue => out.push(rd_write_value.clone().into()),
            JoltR1CSInputs::RamReadValue => out.push(ram_read_value.clone().into()),
            JoltR1CSInputs::RamWriteValue => out.push(ram_write_value.clone().into()),
            JoltR1CSInputs::LeftLookupOperand => out.push(left_lookup_operand.clone().into()),
            JoltR1CSInputs::RightLookupOperand => out.push(right_lookup_operand.clone().into()),
            JoltR1CSInputs::NextUnexpandedPC => out.push(next_unexpanded_pc.clone().into()),
            JoltR1CSInputs::NextPC => out.push(next_pc.clone().into()),
            JoltR1CSInputs::NextIsVirtual => out.push(next_is_virtual.clone().into()),
            JoltR1CSInputs::NextIsFirstInSequence => {
                out.push(next_is_first_in_sequence.clone().into())
            }
            JoltR1CSInputs::LookupOutput => out.push(lookup_output.clone().into()),
            JoltR1CSInputs::ShouldJump => out.push(should_jump.clone().into()),
            JoltR1CSInputs::OpFlags(flag) => {
                let idx = *flag as usize;
                out.push(opflag_vecs[idx].clone().into());
            }
        }
    }
    out
}

impl<F, ProofTranscript, PCS> SumcheckStagesProver<F, ProofTranscript, PCS> for SpartanDagProver<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    fn stage1_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        // Stage 1 remainder: outer-remaining
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> = Vec::new();
        let n_cycles = self.key.num_cycle_vars();
        match OUTER_IMPL {
            OuterImpl::Streaming => {
                let st = self
                    .uni_skip_state
                    .take()
                    .expect("stage1_uni_skip must run before Streaming outer instances");
                let outer_remaining =
                    OuterRemainingStreamingSumcheckProver::gen(state_manager, n_cycles, &st);
                instances.push(Box::new(outer_remaining));
            }
            OuterImpl::Current => {
                let st = self
                    .uni_skip_state
                    .take()
                    .expect("stage1_uni_skip must run before Current outer instances");
                let outer_remaining =
                    OuterRemainingSumcheckProver::gen(state_manager, n_cycles, &st);
                instances.push(Box::new(outer_remaining));
            }
            OuterImpl::Baseline => {
                // No uni-skip: build flattened polynomials from the trace and constraints
                let (preprocessing, _lazy_trace, trace, _program_io, _final_mem) =
                    state_manager.get_prover_data();
                let flattened = build_flattened_polynomials::<F>(&preprocessing.bytecode, trace);
                let padded_num_constraints = R1CS_CONSTRAINTS.len().next_power_of_two();
                let constraints_vec: Vec<R1CSConstraint> =
                    R1CS_CONSTRAINTS.iter().map(|c| c.cons).collect();
                let outer_baseline = OuterBaselineSumcheckProver::<F>::gen_from_polys(
                    &constraints_vec,
                    &flattened,
                    padded_num_constraints,
                    transcript,
                );
                instances.push(Box::new(outer_baseline));
            }
            OuterImpl::RoundBatched => {
                // No uni-skip: directly instantiate round-batched prover
                let outer_round_batched =
                    OuterRoundBatchedSumcheckProver::<F>::gen(state_manager, transcript);
                instances.push(Box::new(outer_round_batched));
            }
        }
        instances
    }

    fn stage2_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, PCS>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        // Stage 2 remainder: inner + product remainder
        let key = self.key.clone();
        let inner_sumcheck = InnerSumcheckProver::gen(opening_accumulator, key, transcript);

        let st = self
            .uni_skip_state
            .take()
            .expect("stage2_prover_uni_skip must run before stage2_prover_instances");
        let n_cycle_vars = self.key.num_cycle_vars();
        let product_virtual_remainder =
            ProductVirtualRemainderProver::gen(state_manager, n_cycle_vars, &st);

        vec![
            Box::new(inner_sumcheck),
            Box::new(product_virtual_remainder),
        ]
    }

    fn stage3_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, PCS>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        /*  Sumcheck 3: Batched sumcheck for NextUnexpandedPC and NextPC verification
            Proves: NextUnexpandedPC(r_cycle) + r * NextPC(r_cycle) =
                    \sum_t (UnexpandedPC(t) + r * PC(t)) * eq_plus_one(r_cycle, t)

            This batched sumcheck simultaneously proves:
            1. NextUnexpandedPC(r_cycle) = \sum_t UnexpandedPC(t) * eq_plus_one(r_cycle, t)
            2. NextPC(r_cycle) = \sum_t PC(t) * eq_plus_one(r_cycle, t)
        */
        let shift_sumcheck =
            ShiftSumcheckProver::gen(state_manager, opening_accumulator, transcript);
        let instruction_input_sumcheck =
            InstructionInputSumcheckProver::gen(state_manager, opening_accumulator, transcript);
        let product_virtual_claim_check =
            ProductVirtualInnerProver::new(opening_accumulator, transcript);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("Spartan ShiftSumcheck", &shift_sumcheck);
            print_data_structure_heap_usage(
                "InstructionInputSumcheck",
                &instruction_input_sumcheck,
            );
        }

        vec![
            Box::new(shift_sumcheck),
            Box::new(instruction_input_sumcheck),
            Box::new(product_virtual_claim_check),
        ]
    }
}

/// Stage 1a: Verify first round of Spartan outer sum-check with univariate skip
pub fn verify_stage1_uni_skip<F: JoltField, T: Transcript>(
    proof: &UniSkipFirstRoundProof<F, T>,
    key: &UniformSpartanKey<F>,
    transcript: &mut T,
) -> Result<UniSkipState<F>, anyhow::Error> {
    let num_rounds_x = key.num_rows_bits();

    let tau = transcript.challenge_vector_optimized::<F>(num_rounds_x);

    let input_claim = F::zero();
    let (r0, claim_after_first) = proof
        .verify::<OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE, OUTER_FIRST_ROUND_POLY_NUM_COEFFS>(
            OUTER_FIRST_ROUND_POLY_NUM_COEFFS - 1,
            input_claim,
            transcript,
        )
        .map_err(|_| anyhow::anyhow!("UniSkip first-round verification failed"))?;

    Ok(UniSkipState {
        claim_after_first,
        r0,
        tau,
    })
}

pub fn verify_stage2_uni_skip<F: JoltField, T: Transcript>(
    proof: &UniSkipFirstRoundProof<F, T>,
    key: &UniformSpartanKey<F>,
    opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<UniSkipState<F>, anyhow::Error> {
    let num_cycle_vars: usize = key.num_cycle_vars();

    // Reuse r_cycle from Stage 1 (outer) for τ_low, and sample τ_high
    let r_cycle = opening_accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::Product, SumcheckId::SpartanOuter)
        .0
        .r;
    debug_assert_eq!(r_cycle.len(), num_cycle_vars);
    let tau_high: F::Challenge = transcript.challenge_scalar_optimized::<F>();
    let mut tau: Vec<F::Challenge> = r_cycle;
    tau.push(tau_high);

    let uniskip_params = ProductVirtualUniSkipInstanceParams::new(opening_accumulator, &tau);
    let input_claim = uniskip_params.input_claim();
    let (r0, claim_after_first) = proof
        .verify::<PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE, PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS>(
            PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS - 1,
            input_claim,
            transcript,
        )
        .map_err(|_| anyhow::anyhow!("ProductVirtual uni-skip first-round verification failed"))?;

    Ok(UniSkipState {
        claim_after_first,
        r0,
        tau,
    })
}
