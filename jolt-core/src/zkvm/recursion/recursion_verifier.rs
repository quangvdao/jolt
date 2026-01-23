//! Unified verifier for the two-stage recursion SNARK protocol
//!
//! This module provides a high-level verifier that verifies:
//! - Stage 1: Constraint sumchecks (GT exp, GT mul, G1 scalar mul)
//! - Stage 2: Virtualization sumcheck
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
    stage1::{
        g1_add::G1AddParams,
        g1_scalar_mul::{G1ScalarMulParams, G1ScalarMulPublicInputs},
        g2_add::G2AddParams,
        g2_scalar_mul::{G2ScalarMulParams, G2ScalarMulPublicInputs},
        gt_exp::{PackedGtExpParams, PackedGtExpPublicInputs, PackedGtExpVerifier},
        gt_mul::{GtMulParams, GtMulVerifier, GtMulVerifierSpec},
    },
    stage2::virtualization::{
        extract_virtual_claims_from_accumulator, DirectEvaluationParams, DirectEvaluationVerifier,
    },
    stage3::{
        jagged::{JaggedSumcheckParams, JaggedSumcheckVerifier},
        jagged_assist::JaggedAssistVerifier,
    },
};
use crate::subprotocols::{sumcheck::BatchedSumcheck, sumcheck_verifier::SumcheckInstanceVerifier};

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

        // ============ STAGE 1: Verify Constraint Sumchecks ============
        let r_stage1 = tracing::info_span!("verify_recursion_stage1").in_scope(|| {
            tracing::info!("Verifying Stage 1: Constraint sumchecks");
            self.verify_stage1(
                &proof.stage1_proof,
                transcript,
                &mut accumulator,
                proof.gamma,
                proof.delta,
            )
        })?;

        // ============ STAGE 2: Verify Virtualization Sumcheck ============
        let r_stage2 = tracing::info_span!("verify_recursion_stage2").in_scope(|| {
            tracing::info!("Verifying Stage 2: Direct evaluation");
            self.verify_stage2(transcript, &mut accumulator, &r_stage1, proof.stage2_m_eval)
        })?;

        // // ============ STAGE 3: Verify Jagged Transform Sumcheck + Stage 3b: Jagged Assist ============
        let _r_stage3 = tracing::info_span!("verify_recursion_stage3").in_scope(|| {
            tracing::info!("Verifying Stage 3: Jagged transform sumcheck + Jagged Assist");
            self.verify_stage3(
                &proof.stage3_proof,
                &proof.stage3b_proof,
                transcript,
                &mut accumulator,
                &r_stage1,
                &r_stage2,
            )
        })?;

        // ============ PCS OPENING VERIFICATION ============
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

        Ok(true)
    }

    /// Verify Stage 1: Constraint sumchecks
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage1")]
    fn verify_stage1<T: Transcript>(
        &self,
        proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<Fq, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        _gamma: Fq,
        _delta: Fq,
    ) -> Result<Vec<<Fq as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        // Create verifiers for each constraint type
        let mut verifiers: Vec<Box<dyn SumcheckInstanceVerifier<Fq, T>>> = Vec::new();

        // Count constraints by type
        let mut num_gt_exp = 0;
        let mut num_gt_mul = 0;
        let mut num_g1_scalar_mul = 0;
        let mut num_g2_scalar_mul = 0;
        let mut num_g1_add = 0;
        let mut num_g2_add = 0;

        // Collect constraint information
        let mut gt_mul_indices = Vec::new();
        let mut g1_scalar_mul_base_points = Vec::new();
        let mut g1_scalar_mul_indices = Vec::new();
        let mut g2_scalar_mul_base_points = Vec::new();
        let mut g2_scalar_mul_indices = Vec::new();
        let mut g1_add_indices = Vec::new();
        let mut g2_add_indices = Vec::new();

        // Count constraints and collect base points per type.
        // Use sequential indices (0, 1, 2...) within each type to match Stage 2's
        // extract_virtual_claims_from_accumulator which uses separate counters per type.
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

        // Add packed GT exp verifier if we have packed GT exp constraints
        // Each PackedGtExp constraint = 1 witness (covers all 254 steps)
        if num_gt_exp > 0 {
            let params = PackedGtExpParams::new();
            let verifier = PackedGtExpVerifier::new(
                params,
                self.input.gt_exp_public_inputs.clone(),
                transcript,
            );
            verifiers.push(Box::new(verifier));
        }

        // Add GT mul verifier if we have GT mul constraints
        if num_gt_mul > 0 {
            let params = GtMulParams::new(num_gt_mul);
            let spec = GtMulVerifierSpec::<Bn254Recursion>::new(params);
            let verifier =
                GtMulVerifier::<Bn254Recursion>::from_spec(spec, gt_mul_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // Add G1 scalar mul verifier if we have G1 scalar mul constraints
        if num_g1_scalar_mul > 0 {
            use super::stage1::g1_scalar_mul::{G1ScalarMulVerifier, G1ScalarMulVerifierSpec};

            let params = G1ScalarMulParams::new(num_g1_scalar_mul);
            debug_assert_eq!(
                self.input.g1_scalar_mul_public_inputs.len(),
                num_g1_scalar_mul,
                "RecursionVerifierInput.g1_scalar_mul_public_inputs must match number of G1ScalarMul constraints"
            );
            let spec = G1ScalarMulVerifierSpec::new(
                params,
                g1_scalar_mul_base_points,
                self.input.g1_scalar_mul_public_inputs.clone(),
            );
            let verifier = G1ScalarMulVerifier::from_spec(spec, g1_scalar_mul_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // Add G2 scalar mul verifier if we have G2 scalar mul constraints
        if num_g2_scalar_mul > 0 {
            use super::stage1::g2_scalar_mul::{G2ScalarMulVerifier, G2ScalarMulVerifierSpec};

            let params = G2ScalarMulParams::new(num_g2_scalar_mul);
            debug_assert_eq!(
                self.input.g2_scalar_mul_public_inputs.len(),
                num_g2_scalar_mul,
                "RecursionVerifierInput.g2_scalar_mul_public_inputs must match number of G2ScalarMul constraints"
            );
            let spec = G2ScalarMulVerifierSpec::new(
                params,
                g2_scalar_mul_base_points,
                self.input.g2_scalar_mul_public_inputs.clone(),
            );
            let verifier = G2ScalarMulVerifier::from_spec(spec, g2_scalar_mul_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // Add G1 add verifier
        if num_g1_add > 0 {
            use super::stage1::g1_add::{G1AddVerifier, G1AddVerifierSpec};
            let params = G1AddParams::new(num_g1_add);
            let spec = G1AddVerifierSpec::new(params);
            let verifier = G1AddVerifier::from_spec(spec, g1_add_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // Add G2 add verifier
        if num_g2_add > 0 {
            use super::stage1::g2_add::{G2AddVerifier, G2AddVerifierSpec};
            let params = G2AddParams::new(num_g2_add);
            let spec = G2AddVerifierSpec::new(params);
            let verifier = G2AddVerifier::from_spec(spec, g2_add_indices, transcript);
            verifiers.push(Box::new(verifier));
        }

        // TODO: Add Boundary/Wiring Sumcheck Verifier (initial/final states + copy constraints)
        // Currently removed due to polynomial size mismatch bug; will be redesigned with packing.

        if verifiers.is_empty() {
            return Err("No constraints to verify in Stage 1".into());
        }

        // Run batched sumcheck verification for all verifiers
        let verifier_refs: Vec<&dyn SumcheckInstanceVerifier<Fq, T>> =
            verifiers.iter().map(|v| &**v).collect();

        let r_stage1 = BatchedSumcheck::verify(proof, verifier_refs, accumulator, transcript)?;

        Ok(r_stage1)
    }

    /// Verify Stage 2: Direct evaluation protocol
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage2")]
    fn verify_stage2<T: Transcript>(
        &self,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        r_stage1: &[<Fq as crate::field::JoltField>::Challenge],
        stage2_m_eval: Fq,
    ) -> Result<Vec<<Fq as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        // Since we know F = Fq, we can work directly with Fq types
        let accumulator_fq: &mut VerifierOpeningAccumulator<Fq> = accumulator;

        // Convert r_stage1 challenges to Fq field elements
        // SAFETY: We verified F = Fq above, so F::Challenge = Fq::Challenge
        let r_x: Vec<Fq> = r_stage1.iter().map(|c| (*c).into()).collect();

        // Extract virtual claims from Stage 1
        let virtual_claims = extract_virtual_claims_from_accumulator(
            accumulator_fq,
            &self.input.constraint_types,
            &self.input.gt_exp_public_inputs,
        );

        // Create parameters
        let params = DirectEvaluationParams::new(
            self.input.num_s_vars,
            self.input.num_constraints,
            self.input.num_constraints_padded,
            self.input.num_constraint_vars,
        );

        // Create and run verifier
        let verifier = DirectEvaluationVerifier::new(params, virtual_claims, r_x);

        // Convert stage2_m_eval from F to Fq
        // SAFETY: We verified F = Fq above
        let m_eval_fq: Fq = stage2_m_eval;

        let r_s = verifier
            .verify(transcript, accumulator_fq, m_eval_fq)
            .map_err(Box::<dyn std::error::Error>::from)?;

        // Convert r_s to challenges for Stage 3 compatibility
        // Stage 3 expects them in reverse order
        // SAFETY: We verified F = Fq above
        let r_stage2: Vec<<Fq as JoltField>::Challenge> =
            r_s.into_iter().rev().map(|f| f.into()).collect();

        Ok(r_stage2)
    }

    /// Verify Stage 3: Jagged transform sumcheck + Stage 3b: Jagged Assist
    #[tracing::instrument(skip_all, name = "RecursionVerifier::verify_stage3")]
    fn verify_stage3<T: Transcript>(
        &self,
        stage3_proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<Fq, T>,
        stage3b_proof: &super::stage3::jagged_assist::JaggedAssistProof<Fq, T>,
        transcript: &mut T,
        accumulator: &mut VerifierOpeningAccumulator<Fq>,
        r_stage1: &[<Fq as crate::field::JoltField>::Challenge],
        r_stage2: &[<Fq as crate::field::JoltField>::Challenge],
    ) -> Result<Vec<<Fq as crate::field::JoltField>::Challenge>, Box<dyn std::error::Error>> {
        let _get_claim_span = tracing::info_span!("stage3_get_sparse_claim").entered();
        // Get the Stage 2 opening claim (sparse matrix claim)
        let (_, sparse_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::DorySparseConstraintMatrix,
            SumcheckId::RecursionVirtualization,
        );
        drop(_get_claim_span);

        let _convert_challenges_span = tracing::info_span!("stage3_convert_challenges").entered();
        // Convert challenges to field elements
        let r_s_final: Vec<Fq> = r_stage2
            .iter()
            .take(self.input.num_s_vars)
            .map(|c| (*c).into())
            .collect();
        let r_x_prev: Vec<Fq> = r_stage1.iter().map(|c| (*c).into()).collect();
        drop(_convert_challenges_span);

        let _dense_size_span = tracing::info_span!("stage3_compute_dense_size").entered();
        // Calculate number of dense variables based on the true dense size
        let dense_size = <VarCountJaggedBijection as crate::zkvm::recursion::bijection::JaggedTransform<Fq>>::dense_size(&self.input.jagged_bijection);
        let num_dense_vars = dense_size.next_power_of_two().trailing_zeros() as usize;
        drop(_dense_size_span);

        let _create_params_span = tracing::info_span!("stage3_create_params").entered();
        // Create jagged sumcheck parameters
        let params = JaggedSumcheckParams::new(
            self.input.num_s_vars,
            self.input.num_constraint_vars,
            num_dense_vars,
        );
        drop(_create_params_span);

        // Convert per-polynomial claimed evaluations to per-row claimed evaluations
        // stage3b_proof.claimed_evaluations[k] = v_k = ĝ(r_x, r_dense, t_{k-1}, t_k)
        // We need: claimed_evaluations[y] = Σ_{k: matrix_row[k]==y} v_k
        let _poly_to_row_span = tracing::info_span!(
            "stage3_poly_to_row_conversion",
            num_polys = stage3b_proof.claimed_evaluations.len()
        )
        .entered();
        let num_rows = 1usize << self.input.num_s_vars;
        let mut claimed_evaluations = vec![Fq::zero(); num_rows];

        for (poly_idx, claimed_eval) in stage3b_proof.claimed_evaluations.iter().enumerate() {
            let matrix_row = self.input.matrix_rows[poly_idx];
            if matrix_row < num_rows {
                claimed_evaluations[matrix_row] += *claimed_eval;
            }
        }
        drop(_poly_to_row_span);

        let _create_verifier_span = tracing::info_span!(
            "stage3_create_verifier",
            num_polys = self.input.jagged_bijection.num_polynomials(),
            num_matrix_rows = self.input.matrix_rows.len()
        )
        .entered();
        // Create jagged sumcheck verifier with claimed evaluations for cheap f̂_jagged
        let verifier =
            JaggedSumcheckVerifier::new(r_s_final, sparse_claim, params, claimed_evaluations);
        drop(_create_verifier_span);

        let _batched_sumcheck_span =
            tracing::info_span!("stage3_batched_sumcheck_verify").entered();
        let r_stage3 =
            BatchedSumcheck::verify(stage3_proof, vec![&verifier], accumulator, transcript)?;
        drop(_batched_sumcheck_span);

        // ============ STAGE 3b: Verify Jagged Assist (Batch MLE Verification) ============
        let _stage3b_span = tracing::info_span!("stage3b_jagged_assist_verify").entered();

        // Convert r_stage3 (dense challenges) to F
        let r_dense: Vec<Fq> = r_stage3.iter().map(|c| (*c).into()).collect();

        // Compute num_bits for branching program
        let num_bits = std::cmp::max(self.input.num_constraint_vars, num_dense_vars);

        // Create Jagged Assist verifier - iterates over K polynomials (not rows!)
        let assist_verifier = JaggedAssistVerifier::<Fq, T>::new(
            stage3b_proof.claimed_evaluations.clone(),
            r_x_prev,
            r_dense,
            &self.input.jagged_bijection,
            num_bits,
            transcript,
        );

        // Verify Jagged Assist sumcheck
        let _r_assist = BatchedSumcheck::verify(
            &stage3b_proof.sumcheck_proof,
            vec![&assist_verifier],
            accumulator,
            transcript,
        )?;

        drop(_stage3b_span);

        Ok(r_stage3)
    }
}
