//! Unified verifier for the recursion SNARK protocol
//!
//! This module provides a high-level verifier that verifies:
//! - Stage 1: Packed GT exp sumcheck
//! - Stage 2: Batched constraint sumchecks (shift + claim reduction + remaining constraints)
//! - Stage 3: Virtualization direct evaluation
//! - Stage 4: Jagged transform sumcheck
//! - Stage 5: Jagged assist sumcheck
//!
//! The verifier returns an opening accumulator for PCS verification.

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::{OpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    },
    transcripts::Transcript,
    zkvm::witness::VirtualPolynomial,
};
use ark_bn254::Fq;
use ark_std::Zero;

use super::{
    bijection::{ConstraintMapping, VarCountJaggedBijection},
    constraints_sys::ConstraintType,
    curve::Bn254Recursion,
    recursion_prover::RecursionProof,
    stage1::gt_exp::{PackedGtExpParams, PackedGtExpPublicInputs, PackedGtExpVerifier},
    stage2::{
        g1_add::G1AddParams,
        g1_scalar_mul::G1ScalarMulPublicInputs,
        g2_add::G2AddParams,
        g2_scalar_mul::G2ScalarMulPublicInputs,
        gt_mul::{GtMulParams, GtMulVerifier, GtMulVerifierSpec},
        packed_gt_exp_reduction::{
            PackedGtExpClaimReductionParams, PackedGtExpClaimReductionVerifier,
        },
        shift_rho::{ShiftRhoParams, ShiftRhoVerifier},
        shift_scalar_mul::{
            g1_shift_params, g2_shift_params, ShiftG1ScalarMulVerifier, ShiftG2ScalarMulVerifier,
        },
    },
    stage3::virtualization::{
        extract_virtual_claims_from_accumulator, DirectEvaluationParams, DirectEvaluationVerifier,
    },
    stage4::jagged::{JaggedSumcheckParams, JaggedSumcheckVerifier},
    stage5::jagged_assist::JaggedAssistVerifier,
};
use crate::subprotocols::{sumcheck::BatchedSumcheck, sumcheck_verifier::SumcheckInstanceVerifier};

use jolt_platform::{end_cycle_tracking, start_cycle_tracking};

// Cycle-marker labels must be static strings: the tracer keys markers by the guest string pointer.
const CYCLE_RECURSION_STAGE1: &str = "jolt_recursion_stage1";
const CYCLE_RECURSION_STAGE2: &str = "jolt_recursion_stage2";
const CYCLE_RECURSION_STAGE3: &str = "jolt_recursion_stage3";
const CYCLE_RECURSION_STAGE4: &str = "jolt_recursion_stage4";
const CYCLE_RECURSION_STAGE5: &str = "jolt_recursion_stage5";
const CYCLE_RECURSION_PCS_OPENING: &str = "jolt_recursion_pcs_opening";

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

/// Input required by the verifier
#[derive(Clone, Debug)]
pub struct RecursionVerifierInput {
    /// Constraint types to verify
    pub constraint_types: Vec<ConstraintType>,
    /// Number of variables in the constraint system
    pub num_vars: usize,
    /// Number of constraint variables (x variables) in the matrix
    pub num_constraint_vars: usize,
    /// Number of s-variables for virtualization
    pub num_s_vars: usize,
    /// Total number of constraints
    pub num_constraints: usize,
    /// Padded number of constraints
    pub num_constraints_padded: usize,
    /// Jagged bijection for Stage 3
    pub jagged_bijection: VarCountJaggedBijection,
    /// Mapping for decoding polynomial indices to matrix rows
    pub jagged_mapping: ConstraintMapping,
    /// Precomputed matrix row indices for each polynomial index
    pub matrix_rows: Vec<usize>,
    /// Public inputs for packed GT exp (base Fq12 and scalar bits)
    pub gt_exp_public_inputs: Vec<PackedGtExpPublicInputs>,
    /// Public inputs for G1 scalar multiplication (scalar per G1ScalarMul constraint)
    pub g1_scalar_mul_public_inputs: Vec<G1ScalarMulPublicInputs>,
    /// Public inputs for G2 scalar multiplication (scalar per G2ScalarMul constraint)
    pub g2_scalar_mul_public_inputs: Vec<G2ScalarMulPublicInputs>,
}

/// Unified verifier for the recursion SNARK
pub struct RecursionVerifier<F: JoltField = Fq> {
    /// Input parameters for verification
    input: RecursionVerifierInput,
    /// Phantom data for the field type
    _phantom: std::marker::PhantomData<F>,
}

impl RecursionVerifier<Fq> {
    /// Create a new recursion verifier
    pub fn new(input: RecursionVerifierInput) -> Self {
        Self {
            input,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Verify the full two-stage recursion proof and PCS opening
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify")]
    pub fn verify<T: Transcript, PCS: CommitmentScheme<Field = Fq>>(
        &self,
        proof: &RecursionProof<Fq, T, PCS>,
        transcript: &mut T,
        matrix_commitment: &PCS::Commitment,
        verifier_setup: &PCS::VerifierSetup,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // Initialize opening accumulator
        let mut accumulator = VerifierOpeningAccumulator::<Fq>::new(self.input.num_vars);

        // Populate accumulator with opening claims from proof
        for (key, value) in &proof.opening_claims {
            accumulator.openings.insert(*key, value.clone());
        }

        // NOTE: The caller is responsible for appending the commitment to the transcript
        // BEFORE calling this method. The prover commits BEFORE running sumchecks,
        // so the verifier must also append the commitment before verification.
        // See: JoltVerifier::verify_stage8_with_recursion() and e2e_test.rs

        // ============ STAGE 1: Packed GT Exp ============
        let _cycle_stage1 = CycleMarkerGuard::new(CYCLE_RECURSION_STAGE1);
        let _r_stage1_packed = tracing::info_span!("verify_recursion_stage1").in_scope(|| {
            tracing::info!("Verifying Stage 1: Packed GT exp sumcheck");
            self.verify_stage1(&proof.stage1_proof, transcript, &mut accumulator)
        })?;
        drop(_cycle_stage1);

        // ============ STAGE 2: Batched Constraint Sumchecks ============
        let _cycle_stage2 = CycleMarkerGuard::new(CYCLE_RECURSION_STAGE2);
        let r_x = tracing::info_span!("verify_recursion_stage2").in_scope(|| {
            tracing::info!("Verifying Stage 2: Batched constraint sumchecks");
            self.verify_stage2(&proof.stage2_proof, transcript, &mut accumulator)
        })?;
        drop(_cycle_stage2);

        // Debug hook: allow stopping after Stage 2 to isolate failures.
        #[cfg(test)]
        if std::env::var("JOLT_RECURSION_STOP_AFTER_STAGE2").is_ok() {
            return Ok(true);
        }

        // ============ STAGE 3: Virtualization Direct Evaluation ============
        let _cycle_stage3 = CycleMarkerGuard::new(CYCLE_RECURSION_STAGE3);
        let r_s = tracing::info_span!("verify_recursion_stage3").in_scope(|| {
            tracing::info!("Verifying Stage 3: Virtualization direct evaluation");
            self.verify_stage3(transcript, &mut accumulator, &r_x, proof.stage3_m_eval)
        })?;
        drop(_cycle_stage3);

        // ============ STAGE 4: Jagged Transform Sumcheck ============
        let _cycle_stage4 = CycleMarkerGuard::new(CYCLE_RECURSION_STAGE4);
        let r_dense = tracing::info_span!("verify_recursion_stage4").in_scope(|| {
            tracing::info!("Verifying Stage 4: Jagged transform sumcheck");
            self.verify_stage4(
                &proof.stage4_proof,
                &proof.stage5_proof,
                transcript,
                &mut accumulator,
                &r_s,
            )
        })?;
        drop(_cycle_stage4);

        // ============ STAGE 5: Jagged Assist ============
        let _cycle_stage5 = CycleMarkerGuard::new(CYCLE_RECURSION_STAGE5);
        tracing::info_span!("verify_recursion_stage5").in_scope(|| {
            tracing::info!("Verifying Stage 5: Jagged assist");
            self.verify_stage5(
                &proof.stage5_proof,
                transcript,
                &mut accumulator,
                &r_dense,
                &r_x,
            )
        })?;
        drop(_cycle_stage5);

        // ============ PCS OPENING VERIFICATION ============
        let _cycle_pcs = CycleMarkerGuard::new(CYCLE_RECURSION_PCS_OPENING);
        tracing::info_span!("verify_recursion_pcs_opening").in_scope(|| {
            tracing::info!("Verifying PCS opening proof");
            // Verify opening proof using PCS
            accumulator.verify_single::<T, PCS>(
                &proof.opening_proof,
                matrix_commitment.clone(),
                verifier_setup,
                transcript,
            )
        })?;
        drop(_cycle_pcs);

        Ok(true)
    }

    /// Verify Stage 1: Packed GT exp sumcheck
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage1")]
    fn verify_stage1<T: Transcript>(
        &self,
        proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<Fq, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
    ) -> Result<Vec<<Fq as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        if self.input.gt_exp_public_inputs.is_empty() {
            return Err("No PackedGtExp constraints to verify in Stage 1".into());
        }

        let params = PackedGtExpParams::new();
        let verifier =
            PackedGtExpVerifier::new(params, self.input.gt_exp_public_inputs.clone(), transcript);

        let r_stage1 = BatchedSumcheck::verify(proof, vec![&verifier], accumulator, transcript)?;
        Ok(r_stage1)
    }

    /// Verify Stage 2: Batched constraint sumchecks.
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage2")]
    fn verify_stage2<T: Transcript>(
        &self,
        proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<Fq, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
    ) -> Result<Vec<<Fq as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        let env_flag_default = |name: &str, default: bool| -> bool {
            std::env::var(name)
                .ok()
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(default)
        };
        let enable_shift_rho = env_flag_default("JOLT_RECURSION_ENABLE_SHIFT_RHO", true);
        let enable_shift_g1_scalar_mul =
            env_flag_default("JOLT_RECURSION_ENABLE_SHIFT_G1_SCALAR_MUL", true);
        let enable_shift_g2_scalar_mul =
            env_flag_default("JOLT_RECURSION_ENABLE_SHIFT_G2_SCALAR_MUL", true);
        let enable_claim_reduction = env_flag_default("JOLT_RECURSION_ENABLE_PGX_REDUCTION", true);

        let mut verifiers: Vec<Box<dyn SumcheckInstanceVerifier<Fq, T>>> = Vec::new();

        // Count constraints by type and collect per-type sequential indices (matching the extractor).
        let mut num_gt_exp = 0usize;
        let mut num_gt_mul = 0usize;
        let mut num_g1_scalar_mul = 0usize;
        let mut num_g2_scalar_mul = 0usize;
        let mut num_g1_add = 0usize;
        let mut num_g2_add = 0usize;

        let mut gt_mul_indices = Vec::new();
        let mut g1_scalar_mul_base_points = Vec::new();
        let mut g1_scalar_mul_indices = Vec::new();
        let mut g2_scalar_mul_base_points = Vec::new();
        let mut g2_scalar_mul_indices = Vec::new();
        let mut g1_add_indices = Vec::new();
        let mut g2_add_indices = Vec::new();

        for constraint in self.input.constraint_types.iter() {
            match constraint {
                ConstraintType::PackedGtExp => {
                    num_gt_exp += 1;
                }
                ConstraintType::GtMul => {
                    gt_mul_indices.push(num_gt_mul);
                    num_gt_mul += 1;
                }
                ConstraintType::G1ScalarMul { base_point } => {
                    g1_scalar_mul_base_points.push(*base_point);
                    g1_scalar_mul_indices.push(num_g1_scalar_mul);
                    num_g1_scalar_mul += 1;
                }
                ConstraintType::G2ScalarMul { base_point } => {
                    g2_scalar_mul_base_points.push(*base_point);
                    g2_scalar_mul_indices.push(num_g2_scalar_mul);
                    num_g2_scalar_mul += 1;
                }
                ConstraintType::G1Add => {
                    g1_add_indices.push(num_g1_add);
                    num_g1_add += 1;
                }
                ConstraintType::G2Add => {
                    g2_add_indices.push(num_g2_add);
                    num_g2_add += 1;
                }
            }
        }

        // Packed GT exp auxiliary subprotocols (shift rho + claim reduction).
        if num_gt_exp > 0 {
            let claim_indices: Vec<usize> = (0..num_gt_exp).collect();
            if enable_shift_rho {
                let shift_verifier = ShiftRhoVerifier::<Fq>::new(
                    ShiftRhoParams::new(num_gt_exp),
                    claim_indices.clone(),
                    transcript,
                );
                verifiers.push(Box::new(shift_verifier));
            }

            if enable_claim_reduction {
                let reduction_verifier = PackedGtExpClaimReductionVerifier::<Fq>::new(
                    PackedGtExpClaimReductionParams::new(2 * num_gt_exp),
                    claim_indices,
                    transcript,
                );
                verifiers.push(Box::new(reduction_verifier));
            }
        }

        // GT mul
        if num_gt_mul > 0 {
            let params = GtMulParams::new(num_gt_mul);
            let spec = GtMulVerifierSpec::<Bn254Recursion>::new(params);
            let verifier =
                GtMulVerifier::<Bn254Recursion>::from_spec(spec, gt_mul_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // G1 scalar mul shift: link A and A_next (x/y)
        if enable_shift_g1_scalar_mul && num_g1_scalar_mul > 0 {
            let mut pairs: Vec<(VirtualPolynomial, VirtualPolynomial)> =
                Vec::with_capacity(num_g1_scalar_mul * 2);
            for i in 0..num_g1_scalar_mul {
                pairs.push((
                    VirtualPolynomial::g1_scalar_mul_xa(i),
                    VirtualPolynomial::g1_scalar_mul_xa_next(i),
                ));
                pairs.push((
                    VirtualPolynomial::g1_scalar_mul_ya(i),
                    VirtualPolynomial::g1_scalar_mul_ya_next(i),
                ));
            }
            verifiers.push(Box::new(ShiftG1ScalarMulVerifier::<Fq>::new(
                g1_shift_params(pairs.len()),
                pairs,
                transcript,
            )));
        }

        // G1 scalar mul
        if num_g1_scalar_mul > 0 {
            use super::stage2::g1_scalar_mul::{
                G1ScalarMulParams, G1ScalarMulVerifier, G1ScalarMulVerifierSpec,
            };

            let params = G1ScalarMulParams::new(num_g1_scalar_mul);
            debug_assert_eq!(
                self.input.g1_scalar_mul_public_inputs.len(),
                num_g1_scalar_mul,
                "RecursionVerifierInput.g1_scalar_mul_public_inputs must match number of G1ScalarMul constraints"
            );
            let spec = G1ScalarMulVerifierSpec::new(
                params,
                self.input.g1_scalar_mul_public_inputs.clone(),
                g1_scalar_mul_base_points,
            );
            let verifier = G1ScalarMulVerifier::from_spec(spec, g1_scalar_mul_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // G2 scalar mul shift: link A and A_next (x/y, c0/c1)
        if enable_shift_g2_scalar_mul && num_g2_scalar_mul > 0 {
            let mut pairs: Vec<(VirtualPolynomial, VirtualPolynomial)> =
                Vec::with_capacity(num_g2_scalar_mul * 4);
            for i in 0..num_g2_scalar_mul {
                pairs.push((
                    VirtualPolynomial::g2_scalar_mul_xa_c0(i),
                    VirtualPolynomial::g2_scalar_mul_xa_next_c0(i),
                ));
                pairs.push((
                    VirtualPolynomial::g2_scalar_mul_xa_c1(i),
                    VirtualPolynomial::g2_scalar_mul_xa_next_c1(i),
                ));
                pairs.push((
                    VirtualPolynomial::g2_scalar_mul_ya_c0(i),
                    VirtualPolynomial::g2_scalar_mul_ya_next_c0(i),
                ));
                pairs.push((
                    VirtualPolynomial::g2_scalar_mul_ya_c1(i),
                    VirtualPolynomial::g2_scalar_mul_ya_next_c1(i),
                ));
            }
            verifiers.push(Box::new(ShiftG2ScalarMulVerifier::<Fq>::new(
                g2_shift_params(pairs.len()),
                pairs,
                transcript,
            )));
        }

        // G2 scalar mul
        if num_g2_scalar_mul > 0 {
            use super::stage2::g2_scalar_mul::{
                G2ScalarMulParams, G2ScalarMulVerifier, G2ScalarMulVerifierSpec,
            };

            let params = G2ScalarMulParams::new(num_g2_scalar_mul);
            debug_assert_eq!(
                self.input.g2_scalar_mul_public_inputs.len(),
                num_g2_scalar_mul,
                "RecursionVerifierInput.g2_scalar_mul_public_inputs must match number of G2ScalarMul constraints"
            );
            let spec = G2ScalarMulVerifierSpec::new(
                params,
                self.input.g2_scalar_mul_public_inputs.clone(),
                g2_scalar_mul_base_points,
            );
            let verifier = G2ScalarMulVerifier::from_spec(spec, g2_scalar_mul_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // G1 add
        if num_g1_add > 0 {
            use super::stage2::g1_add::{G1AddVerifier, G1AddVerifierSpec};
            let params = G1AddParams::new(num_g1_add);
            let spec = G1AddVerifierSpec::new(params);
            let verifier = G1AddVerifier::from_spec(spec, g1_add_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // G2 add
        if num_g2_add > 0 {
            use super::stage2::g2_add::{G2AddVerifier, G2AddVerifierSpec};
            let params = G2AddParams::new(num_g2_add);
            let spec = G2AddVerifierSpec::new(params);
            let verifier = G2AddVerifier::from_spec(spec, g2_add_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // TODO: wiring/boundary constraints.

        if verifiers.is_empty() {
            return Err("No constraints to verify in Stage 2".into());
        }

        let verifier_refs: Vec<&dyn SumcheckInstanceVerifier<Fq, T>> =
            verifiers.iter().map(|v| &**v).collect();
        let r_x = BatchedSumcheck::verify(proof, verifier_refs, accumulator, transcript)?;
        Ok(r_x)
    }

    /// Verify Stage 3: Direct evaluation protocol (virtualization).
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage3")]
    fn verify_stage3<T: Transcript>(
        &self,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        r_x: &[<Fq as crate::field::JoltField>::Challenge],
        stage3_m_eval: Fq,
    ) -> Result<Vec<<Fq as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        let accumulator_fq: &mut VerifierOpeningAccumulator<Fq> = accumulator;
        let r_x_fq: Vec<Fq> = r_x.iter().map(|c| (*c).into()).collect();

        let virtual_claims = extract_virtual_claims_from_accumulator(
            accumulator_fq,
            &self.input.constraint_types,
            &self.input.gt_exp_public_inputs,
        );

        let params = DirectEvaluationParams::new(
            self.input.num_s_vars,
            self.input.num_constraints,
            self.input.num_constraints_padded,
            self.input.num_constraint_vars,
        );

        let verifier = DirectEvaluationVerifier::new(params, virtual_claims, r_x_fq);
        let m_eval_fq: Fq = stage3_m_eval;
        let r_s = verifier
            .verify(transcript, accumulator_fq, m_eval_fq)
            .map_err(Box::<dyn std::error::Error>::from)?;

        let r_s_challenges: Vec<<Fq as JoltField>::Challenge> =
            r_s.into_iter().rev().map(|f| f.into()).collect();

        Ok(r_s_challenges)
    }

    /// Verify Stage 4: Jagged transform sumcheck.
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage4")]
    fn verify_stage4<T: Transcript>(
        &self,
        stage4_proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<Fq, T>,
        stage5_proof: &super::stage5::jagged_assist::JaggedAssistProof<Fq, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        r_s: &[<Fq as crate::field::JoltField>::Challenge],
    ) -> Result<Vec<<Fq as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        let (_, sparse_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::DorySparseConstraintMatrix,
            SumcheckId::RecursionVirtualization,
        );

        let r_s_final: Vec<Fq> = r_s
            .iter()
            .take(self.input.num_s_vars)
            .map(|c| (*c).into())
            .collect();

        let dense_size = <VarCountJaggedBijection as crate::zkvm::recursion::bijection::JaggedTransform<Fq>>::dense_size(&self.input.jagged_bijection);
        let num_dense_vars = dense_size.next_power_of_two().trailing_zeros() as usize;

        let params = JaggedSumcheckParams::new(
            self.input.num_s_vars,
            self.input.num_constraint_vars,
            num_dense_vars,
        );

        // Convert per-polynomial claimed evals → per-row evals.
        let num_rows = 1usize << self.input.num_s_vars;
        let mut claimed_evaluations_by_row = vec![Fq::zero(); num_rows];
        for (poly_idx, claimed_eval) in stage5_proof.claimed_evaluations.iter().enumerate() {
            let matrix_row = self.input.matrix_rows[poly_idx];
            if matrix_row < num_rows {
                claimed_evaluations_by_row[matrix_row] += *claimed_eval;
            }
        }

        let verifier = JaggedSumcheckVerifier::new(
            r_s_final,
            sparse_claim,
            params,
            claimed_evaluations_by_row,
        );

        let r_dense =
            BatchedSumcheck::verify(stage4_proof, vec![&verifier], accumulator, transcript)?;

        Ok(r_dense)
    }

    /// Verify Stage 5: Jagged assist (batch MLE verification).
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage5")]
    fn verify_stage5<T: Transcript>(
        &self,
        stage5_proof: &super::stage5::jagged_assist::JaggedAssistProof<Fq, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        r_dense: &[<Fq as crate::field::JoltField>::Challenge],
        r_x: &[<Fq as crate::field::JoltField>::Challenge],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let r_dense_fq: Vec<Fq> = r_dense.iter().map(|c| (*c).into()).collect();
        let r_x_prev: Vec<Fq> = r_x.iter().map(|c| (*c).into()).collect();

        let dense_size = <VarCountJaggedBijection as crate::zkvm::recursion::bijection::JaggedTransform<Fq>>::dense_size(&self.input.jagged_bijection);
        let num_dense_vars = dense_size.next_power_of_two().trailing_zeros() as usize;
        let num_bits = std::cmp::max(self.input.num_constraint_vars, num_dense_vars);

        let assist_verifier = JaggedAssistVerifier::<Fq, T>::new(
            stage5_proof.claimed_evaluations.clone(),
            r_x_prev,
            r_dense_fq,
            &self.input.jagged_bijection,
            num_bits,
            transcript,
        );

        let _r_assist = BatchedSumcheck::verify(
            &stage5_proof.sumcheck_proof,
            vec![&assist_verifier],
            accumulator,
            transcript,
        )?;

        Ok(())
    }
}
