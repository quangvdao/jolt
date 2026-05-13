//! Dory polynomial commitment scheme implementation

use super::dory_globals::{DoryGlobals, DoryLayout};
use super::jolt_dory_routines::JoltG1Routines;
use super::wrappers::{
    ark_to_jolt, jolt_to_ark, ArkDoryProof, ArkFr, ArkG1, ArkGT, ArkworksProverSetup,
    ArkworksVerifierSetup, JoltToDoryTranscript, BN254,
};
use crate::{
    curve::JoltCurve,
    field::JoltField,
    poly::commitment::commitment_scheme::{
        BatchOpeningScheme, CommitmentScheme, SourceBatchCommitmentScheme, ZkOpeningSupport,
    },
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use ark_bn254::G1Projective;
use ark_ff::Zero;
use dory::primitives::{
    arithmetic::{Field as DoryField, Group},
    poly::{MultilinearLagrange, Polynomial},
};
use jolt_crypto::Bn254G1;
use jolt_openings::{BatchCommitmentSource, CommitmentSource, SourceRow};
use rayon::prelude::*;
use std::borrow::Borrow;
use tracing::trace_span;

#[derive(Clone)]
pub struct DoryCommitmentScheme;

#[derive(Clone, Debug, PartialEq)]
pub struct DoryOpeningProofHint {
    row_commitments: Vec<ArkG1>,
    commit_blind: ArkFr,
}

impl DoryOpeningProofHint {
    fn new(row_commitments: Vec<ArkG1>, commit_blind: ArkFr) -> Self {
        Self {
            row_commitments,
            commit_blind,
        }
    }

    fn into_parts(self) -> (Vec<ArkG1>, ArkFr) {
        (self.row_commitments, self.commit_blind)
    }
}

pub fn bind_opening_inputs<F: JoltField, ProofTranscript: Transcript>(
    transcript: &mut ProofTranscript,
    opening_point: &[F::Challenge],
    opening: &F,
) {
    let mut point_scalars = Vec::with_capacity(opening_point.len());
    for point in opening_point {
        let scalar: F = (*point).into();
        point_scalars.push(scalar);
    }
    transcript.append_scalars(b"dory_opening_point", &point_scalars);

    transcript.append_scalar(b"dory_opening_eval", opening);
}

#[cfg(feature = "zk")]
pub fn bind_opening_inputs_zk<F: JoltField, C: JoltCurve<F = F>, ProofTranscript: Transcript>(
    transcript: &mut ProofTranscript,
    opening_point: &[F::Challenge],
    y_com: &C::G1,
) {
    let mut point_scalars = Vec::with_capacity(opening_point.len());
    for point in opening_point {
        let scalar: F = (*point).into();
        point_scalars.push(scalar);
    }
    transcript.append_scalars(b"dory_opening_point", &point_scalars);

    transcript.append_commitment(b"dory_eval_commitment", y_com);
}

impl CommitmentScheme for DoryCommitmentScheme {
    type Field = ark_bn254::Fr;
    type ProverSetup = ArkworksProverSetup;
    type VerifierSetup = ArkworksVerifierSetup;
    type Commitment = ArkGT;
    type Proof = ArkDoryProof;
    type BatchedProof = Vec<ArkDoryProof>;
    type OpeningProofHint = DoryOpeningProofHint;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_prover").entered();
        #[cfg(test)]
        DoryGlobals::configure_test_cache_root();

        #[cfg(not(target_arch = "wasm32"))]
        let setup = ArkworksProverSetup::new_from_urs(max_num_vars);
        #[cfg(target_arch = "wasm32")]
        let setup = ArkworksProverSetup::new(max_num_vars);

        // The prepared-point cache in dory-pcs is global and can only be initialized once.
        // In unit tests, multiple setups with different sizes are created, so initializing the
        // cache with a small setup can break later tests that need more generators.
        // We therefore disable cache initialization in `cfg(test)` builds.
        #[cfg(not(test))]
        DoryGlobals::init_prepared_cache(&setup.g1_vec, &setup.g2_vec);

        setup
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_verifier").entered();
        setup.to_verifier_setup()
    }

    fn commit(
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let _span = trace_span!("DoryCommitmentScheme::commit").entered();

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        #[cfg(feature = "zk")]
        type DoryMode = dory::ZK;
        #[cfg(not(feature = "zk"))]
        type DoryMode = dory::Transparent;

        let (tier_2, row_commitments, commit_blind) =
            <MultilinearPolynomial<ark_bn254::Fr> as Polynomial<ArkFr>>::commit::<
                BN254,
                DoryMode,
                JoltG1Routines,
            >(poly, nu, sigma, setup)
            .expect("commitment should succeed");

        (
            tier_2,
            DoryOpeningProofHint::new(row_commitments, commit_blind),
        )
    }

    fn batch_commit<U>(
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<ark_bn254::Fr>> + Sync,
    {
        let _span = trace_span!("DoryCommitmentScheme::batch_commit").entered();

        polys
            .par_iter()
            .map(|poly| Self::commit(poly.borrow(), gens))
            .collect()
    }

    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> (Self::Proof, Option<Self::Field>) {
        let _span = trace_span!("DoryCommitmentScheme::prove").entered();

        let hint = convert_old_dory_hint(hint.unwrap_or_else(|| Self::commit(poly, setup).1));
        let setup = jolt_dory::DoryProverSetup(setup.clone());
        let source = CoreOpeningSource(poly);
        let (nu, sigma) = current_dory_shape();
        let point = convert_opening_point(opening_point);
        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        #[cfg(feature = "zk")]
        {
            let (proof, _y_com, y_blinding) = jolt_dory::DoryScheme::open_zk_source_with_shape(
                &source,
                &point,
                nu,
                sigma,
                &setup,
                hint,
                &mut dory_transcript,
            );
            (proof.0, Some(ark_bn254::Fr::from(y_blinding)))
        }
        #[cfg(not(feature = "zk"))]
        {
            let proof = jolt_dory::DoryScheme::open_source_with_shape(
                &source,
                &point,
                nu,
                sigma,
                &setup,
                hint,
                &mut dory_transcript,
            );
            (proof.0, None)
        }
    }

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        opening: &ark_bn254::Fr,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let _span = trace_span!("DoryCommitmentScheme::verify").entered();

        let proof = jolt_dory::DoryProof(proof.clone());
        let setup = jolt_dory::DoryVerifierSetup(setup.clone());
        let commitment = jolt_dory::DoryCommitment::from_dory_pcs(*commitment);
        let point = convert_opening_point(opening_point);
        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        #[cfg(feature = "zk")]
        let result = {
            let _ = opening;
            jolt_dory::DoryScheme::verify_zk_with_shape(
                &commitment,
                &point,
                &proof,
                &setup,
                &mut dory_transcript,
            )
        };
        #[cfg(not(feature = "zk"))]
        let result = jolt_dory::DoryScheme::verify_with_shape(
            &commitment,
            &point,
            jolt_field::Fr::from(*opening),
            &proof,
            &setup,
            &mut dory_transcript,
        );

        result.map_err(|_| ProofVerifyError::InternalError)
    }

    fn protocol_name() -> &'static [u8] {
        b"Dory"
    }

    /// In Dory, the opening proof hint consists of the Pedersen commitments to the rows
    /// of the polynomial coefficient matrix. In the context of a batch opening proof, we
    /// can homomorphically combine the row commitments for multiple polynomials into the
    /// row commitments for the RLC of those polynomials. This is more efficient than computing
    /// the row commitments for the RLC from scratch.
    ///
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_hints")]
    fn combine_hints(
        hints: Vec<Self::OpeningProofHint>,
        coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        let num_rows = DoryGlobals::get_max_num_rows();

        let mut rlc_hint = vec![ArkG1(G1Projective::zero()); num_rows];
        let mut rlc_commit_blind = <ArkFr as DoryField>::zero();
        for (coeff, hint) in coeffs.iter().zip(hints.into_iter()) {
            let DoryOpeningProofHint {
                mut row_commitments,
                commit_blind,
            } = hint;
            row_commitments.resize(num_rows, ArkG1(G1Projective::zero()));
            let ark_coeff = jolt_to_ark(coeff);
            rlc_commit_blind = rlc_commit_blind + ark_coeff * commit_blind;

            let row_commitment_projects: &mut [G1Projective] = unsafe {
                std::slice::from_raw_parts_mut(
                    row_commitments.as_mut_ptr() as *mut G1Projective,
                    row_commitments.len(),
                )
            };

            let rlc_row_commitments: &[G1Projective] = unsafe {
                std::slice::from_raw_parts(rlc_hint.as_ptr() as *const G1Projective, rlc_hint.len())
            };

            let _span = trace_span!("vector_scalar_mul_add_gamma_g1_online");
            let _enter = _span.enter();

            jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
                row_commitment_projects,
                *coeff,
                rlc_row_commitments,
            );

            let _ = std::mem::replace(&mut rlc_hint, row_commitments);
        }

        DoryOpeningProofHint::new(rlc_hint, rlc_commit_blind)
    }

    /// Homomorphically combines multiple commitments using a random linear combination.
    /// Computes: sum_i(coeff_i * commitment_i) for the GT elements.
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_commitments")]
    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let _span = trace_span!("DoryCommitmentScheme::combine_commitments").entered();

        // Combine GT elements using parallel RLC
        let commitments_vec: Vec<&ArkGT> = commitments.iter().map(|c| c.borrow()).collect();
        coeffs
            .par_iter()
            .zip(commitments_vec.par_iter())
            .map(|(coeff, commitment)| {
                let ark_coeff = jolt_to_ark(coeff);
                ark_coeff * **commitment
            })
            .reduce(ArkGT::identity, |a, b| a + b)
    }
}

impl SourceBatchCommitmentScheme for DoryCommitmentScheme {
    fn commit_batch<B: BatchCommitmentSource<jolt_field::Fr>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)> {
        let setup = jolt_dory::DoryProverSetup(setup.clone());
        <jolt_dory::DoryScheme as jolt_openings::CommitmentScheme>::commit_batch(batch, ids, &setup)
            .into_iter()
            .map(convert_new_dory_commitment_and_hint)
            .collect()
    }

    #[cfg(feature = "zk")]
    fn commit_batch_zk<B: BatchCommitmentSource<jolt_field::Fr>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)> {
        let setup = jolt_dory::DoryProverSetup(setup.clone());
        <jolt_dory::DoryScheme as jolt_openings::ZkOpeningScheme>::commit_batch_zk(
            batch, ids, &setup,
        )
        .into_iter()
        .map(convert_new_dory_commitment_and_hint)
        .collect()
    }
}

impl BatchOpeningScheme for DoryCommitmentScheme {
    fn prove_batch<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> (Self::BatchedProof, Option<Self::Field>) {
        let hint = convert_old_dory_hint(
            hint.expect("Dory batch opening requires the combined opening hint"),
        );
        let setup = jolt_dory::DoryProverSetup(setup.clone());
        let source = CoreOpeningSource(poly);
        let (nu, sigma) = current_dory_shape();
        let point = convert_opening_point(opening_point);
        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        #[cfg(feature = "zk")]
        {
            let (proof, _y_com, y_blinding) = jolt_dory::DoryScheme::open_zk_source_with_shape(
                &source,
                &point,
                nu,
                sigma,
                &setup,
                hint,
                &mut dory_transcript,
            );
            (vec![proof.0], Some(ark_bn254::Fr::from(y_blinding)))
        }
        #[cfg(not(feature = "zk"))]
        {
            let proof = jolt_dory::DoryScheme::open_source_with_shape(
                &source,
                &point,
                nu,
                sigma,
                &setup,
                hint,
                &mut dory_transcript,
            );
            (vec![proof.0], None)
        }
    }

    fn verify_batch<ProofTranscript: Transcript>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let [proof] = proof.as_slice() else {
            return Err(ProofVerifyError::InvalidOpeningProof);
        };
        let proof = jolt_dory::DoryProof(proof.clone());
        let setup = jolt_dory::DoryVerifierSetup(setup.clone());
        let commitment = jolt_dory::DoryCommitment::from_dory_pcs(*commitment);
        let point = convert_opening_point(opening_point);
        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        #[cfg(feature = "zk")]
        let result = {
            let _ = opening;
            jolt_dory::DoryScheme::verify_zk_with_shape(
                &commitment,
                &point,
                &proof,
                &setup,
                &mut dory_transcript,
            )
        };
        #[cfg(not(feature = "zk"))]
        let result = jolt_dory::DoryScheme::verify_with_shape(
            &commitment,
            &point,
            jolt_field::Fr::from(*opening),
            &proof,
            &setup,
            &mut dory_transcript,
        );

        result.map_err(|_| ProofVerifyError::InternalError)
    }
}

struct CoreOpeningSource<'a>(&'a MultilinearPolynomial<ark_bn254::Fr>);

impl CommitmentSource<jolt_field::Fr> for CoreOpeningSource<'_> {
    fn num_vars(&self) -> usize {
        if matches!(self.0, MultilinearPolynomial::RLC(_)) {
            let (nu, sigma) = current_dory_shape();
            nu + sigma
        } else {
            self.0.get_num_vars()
        }
    }

    fn evaluate(&self, point: &[jolt_field::Fr]) -> jolt_field::Fr {
        let point: Vec<ark_bn254::Fr> = point
            .iter()
            .rev()
            .copied()
            .map(ark_bn254::Fr::from)
            .collect();
        jolt_field::Fr::from(PolynomialEvaluation::evaluate(self.0, point.as_slice()))
    }

    fn for_each_row<V>(&self, sigma: usize, mut visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, jolt_field::Fr>),
    {
        assert!(
            !matches!(self.0, MultilinearPolynomial::RLC(_)),
            "streaming RLC openings require a precomputed Dory hint"
        );
        let num_cols = 1usize << sigma;
        let len = self.0.original_len();
        let mut row = Vec::with_capacity(num_cols);
        for row_index in 0..len.div_ceil(num_cols) {
            row.clear();
            let start = row_index * num_cols;
            let end = (start + num_cols).min(len);
            row.extend((start..end).map(|idx| jolt_field::Fr::from(self.0.get_coeff(idx))));
            visit(row_index, SourceRow::FieldElements(&row));
        }
    }

    fn fold_rows(&self, left: &[jolt_field::Fr], sigma: usize) -> Vec<jolt_field::Fr> {
        let left: Vec<ArkFr> = left
            .iter()
            .copied()
            .map(ark_bn254::Fr::from)
            .map(|value| jolt_to_ark(&value))
            .collect();
        let nu = self.num_vars().saturating_sub(sigma);
        <MultilinearPolynomial<ark_bn254::Fr> as MultilinearLagrange<ArkFr>>::vector_matrix_product(
            self.0, &left, nu, sigma,
        )
        .iter()
        .map(|value| jolt_field::Fr::from(ark_to_jolt(value)))
        .collect()
    }
}

fn current_dory_shape() -> (usize, usize) {
    (
        DoryGlobals::get_max_num_rows().log_2(),
        DoryGlobals::get_num_columns().log_2(),
    )
}

fn convert_opening_point(
    opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
) -> Vec<jolt_field::Fr> {
    reorder_opening_point_for_layout::<ark_bn254::Fr>(opening_point)
        .iter()
        .map(|point| jolt_field::Fr::from(ark_bn254::Fr::from(*point)))
        .collect()
}

fn convert_old_dory_hint(hint: DoryOpeningProofHint) -> jolt_dory::DoryHint {
    let (row_commitments, commit_blind) = hint.into_parts();
    let row_commitments = row_commitments
        .into_iter()
        .map(|commitment| Bn254G1::from(commitment.0))
        .collect();
    let commit_blind = jolt_field::Fr::from(ark_to_jolt(&commit_blind));
    jolt_dory::DoryHint::from_parts(row_commitments, commit_blind)
}

fn convert_new_dory_commitment_and_hint(
    (commitment, hint): (jolt_dory::DoryCommitment, jolt_dory::DoryHint),
) -> (ArkGT, DoryOpeningProofHint) {
    let commitment = ArkGT(commitment.0.into());
    let (row_commitments, commit_blind) = hint.into_parts();
    let row_commitments = row_commitments
        .into_iter()
        .map(|commitment| ArkG1(commitment.into_inner()))
        .collect();
    let commit_blind = jolt_to_ark(&ark_bn254::Fr::from(commit_blind));
    (
        commitment,
        DoryOpeningProofHint::new(row_commitments, commit_blind),
    )
}

impl<C: JoltCurve> ZkOpeningSupport<C> for DoryCommitmentScheme
where
    C::G1: From<ArkG1>,
{
    fn batch_eval_commitment(proof: &Self::BatchedProof) -> Option<C::G1> {
        let [proof] = proof.as_slice() else {
            return None;
        };
        #[cfg(feature = "zk")]
        {
            proof.y_com.as_ref().copied().map(C::G1::from)
        }
        #[cfg(not(feature = "zk"))]
        {
            let _ = proof;
            None
        }
    }

    fn eval_commitment_gens(setup: &Self::ProverSetup) -> Option<(C::G1, C::G1)> {
        let g1_0 = setup.0.g1_vec.first().copied().map(C::G1::from)?;
        let h1 = C::G1::from(setup.0.h1);
        Some((g1_0, h1))
    }

    fn eval_commitment_gens_verifier(setup: &Self::VerifierSetup) -> Option<(C::G1, C::G1)> {
        let g1_0 = C::G1::from(setup.0.g1_0);
        let h1 = C::G1::from(setup.0.h1);
        Some((g1_0, h1))
    }

    #[cfg(feature = "zk")]
    fn zk_generators(setup: &Self::ProverSetup, count: usize) -> Option<(Vec<C::G1>, C::G1)> {
        let count = std::cmp::min(count, setup.0.g1_vec.len());
        let g1s = setup.0.g1_vec[..count]
            .iter()
            .map(|g| C::G1::from(*g))
            .collect();
        let h1 = C::G1::from(setup.0.h1);
        Some((g1s, h1))
    }
}

/// Reorders opening_point for AddressMajor layout.
///
/// For AddressMajor layout, reorders opening_point from [r_address, r_cycle] to [r_cycle, r_address].
/// This ensures that after Dory's reversal and splitting:
/// - Column (right) vector gets address variables (matching AddressMajor column indexing)
/// - Row (left) vector gets cycle variables (matching AddressMajor row indexing)
///
/// For CycleMajor layout, returns the point unchanged.
fn reorder_opening_point_for_layout<F: JoltField>(
    opening_point: &[F::Challenge],
) -> Vec<F::Challenge> {
    if DoryGlobals::get_layout() == DoryLayout::AddressMajor {
        let log_T = DoryGlobals::get_T().log_2();
        let log_K = opening_point.len().saturating_sub(log_T);
        let (r_address, r_cycle) = opening_point.split_at(log_K);
        [r_cycle, r_address].concat()
    } else {
        opening_point.to_vec()
    }
}
