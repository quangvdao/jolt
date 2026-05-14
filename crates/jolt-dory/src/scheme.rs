//! Dory PCS implementing the `jolt-openings` trait hierarchy.

#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::unimplemented,
    reason = "ZK proof y_com/y_blinding are Dory-mode invariants; dory::prove/verify errors are caller-precondition violations surfaced via panic; the dory adapter's commit is unreachable because DoryScheme pre-computes row commitments"
)]

use dory::backends::arkworks::ArkworksProverSetup;
use dory::mode::Transparent;
use dory::primitives::arithmetic::{
    DoryRoutines, Field as DoryField, Group as DoryGroup, PairingCurve,
};
use dory::primitives::poly::{MultilinearLagrange, Polynomial as DoryPolynomial};
use dory::primitives::transcript::Transcript as DoryTranscript;
use dory::Mode;
use jolt_crypto::ec::bn254::batch_addition::batch_g1_additions_multi_affine;
use jolt_crypto::{Bn254G1, Bn254GT, Commitment, DeriveSetup, JoltGroup, PedersenSetup};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::{
    homomorphic_prove_batch, homomorphic_verify_batch, AdditivelyHomomorphic,
    AdditivelyHomomorphicVerifier, BatchCommitmentSource, CommitmentScheme,
    CommitmentSchemeVerifier, CommitmentSource, EvaluationCommitmentProver,
    EvaluationCommitmentScheme, OpeningClaim, OpeningsError, ProverClaim, PublicVerifierSetup,
    ShapedCommitmentScheme, ShapedZkOpeningScheme, SourceRow, ZkOpeningScheme,
    ZkOpeningSchemeVerifier,
};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};
use rayon::prelude::*;

use ark_bn254::{G1Affine, G1Projective};
use ark_ec::CurveGroup;
use dory::backends::arkworks::ArkG1 as ArkG1Struct;

use crate::routines::{JoltG1Routines, JoltG2Routines};
use crate::transcript::JoltToDoryTranscript;
use crate::types::{DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryVerifierSetup};

// All jolt types below are #[repr(transparent)] over the same arkworks
// inner type as their dory-pcs counterpart, guaranteeing identical layout.

/// Dory-pcs's arkworks scalar wrapper.
///
/// Most callers should use the backend-neutral `jolt_field::Fr` APIs on
/// `DoryScheme`. This type is exposed for Dory-native integration points that
/// already implement dory-pcs polynomial traits and need to avoid converting
/// large opening-time vectors through the generic source abstraction.
pub type ArkFr = dory::backends::arkworks::ArkFr;
pub(crate) type ArkG1 = dory::backends::arkworks::ArkG1;
pub(crate) type ArkGT = dory::backends::arkworks::ArkGT;
type InnerBN254 = dory::backends::arkworks::BN254;

// All conversion functions below rely on repr(transparent) layout identity
// between jolt and dory-pcs wrappers over the same arkworks inner type.

#[inline]
pub(crate) fn jolt_fr_to_ark(f: &Fr) -> ArkFr {
    // SAFETY: Fr and ArkFr are both repr(transparent) over ark_bn254::Fr.
    unsafe { std::mem::transmute_copy(f) }
}

#[inline]
pub(crate) fn ark_to_jolt_fr(ark: &ArkFr) -> Fr {
    // SAFETY: same layout as jolt_fr_to_ark.
    unsafe { std::mem::transmute_copy(ark) }
}

#[inline]
pub(crate) fn jolt_gt_to_ark(gt: &Bn254GT) -> ArkGT {
    // SAFETY: Bn254GT and ArkGT are both repr(transparent) over Fq12.
    unsafe { std::mem::transmute_copy(gt) }
}

#[inline]
pub(crate) fn ark_to_jolt_gt(ark: &ArkGT) -> Bn254GT {
    // SAFETY: same layout as jolt_gt_to_ark.
    unsafe { std::mem::transmute_copy(ark) }
}

#[inline]
pub(crate) fn jolt_g1_vec_to_ark(v: Vec<Bn254G1>) -> Vec<ArkG1> {
    // SAFETY: Bn254G1 and ArkG1 have identical size/align (repr(transparent)
    // over G1Projective), so Vec layout is identical.
    unsafe { std::mem::transmute(v) }
}

#[inline]
pub(crate) fn ark_to_jolt_g1_vec(v: Vec<ArkG1>) -> Vec<Bn254G1> {
    // SAFETY: same layout as jolt_g1_vec_to_ark.
    unsafe { std::mem::transmute(v) }
}

#[inline]
pub(crate) fn ark_to_jolt_g1(ark: ArkG1) -> Bn254G1 {
    // SAFETY: Bn254G1 and ArkG1 are both repr(transparent) over G1Projective.
    unsafe { std::mem::transmute(ark) }
}

#[derive(Clone)]
pub struct DoryScheme;

impl DoryScheme {
    #[tracing::instrument(skip_all, name = "DoryScheme::setup_prover", fields(max_num_vars))]
    pub fn setup_prover(max_num_vars: usize) -> DoryProverSetup {
        #[cfg(not(target_arch = "wasm32"))]
        let setup = ArkworksProverSetup::new_from_urs(max_num_vars);
        #[cfg(target_arch = "wasm32")]
        let setup = ArkworksProverSetup::new(max_num_vars);
        DoryProverSetup(setup)
    }

    /// Derives the verifier SRS (a subset of the prover SRS).
    #[tracing::instrument(skip_all, name = "DoryScheme::setup_verifier", fields(max_num_vars))]
    pub fn setup_verifier(max_num_vars: usize) -> DoryVerifierSetup {
        let prover_setup = Self::setup_prover(max_num_vars);
        DoryVerifierSetup(prover_setup.0.to_verifier_setup())
    }

    fn commit_with_mode<S, M>(source: &S, setup: &ArkworksProverSetup) -> (DoryCommitment, DoryHint)
    where
        S: CommitmentSource<Fr> + ?Sized,
        M: Mode,
    {
        let row_commitments = compute_row_commitments(source, setup);
        finish_row_commitments::<M>(row_commitments, setup)
    }

    fn commit_with_shape_mode<S, M>(
        source: &S,
        nu: usize,
        sigma: usize,
        setup: &ArkworksProverSetup,
    ) -> (DoryCommitment, DoryHint)
    where
        S: CommitmentSource<Fr> + ?Sized,
        M: Mode,
    {
        let row_commitments = compute_row_commitments_with_shape(source, nu, sigma, setup);
        finish_row_commitments::<M>(row_commitments, setup)
    }

    /// Commits a source using an explicit Dory matrix shape.
    ///
    /// The generic [`CommitmentScheme::commit`] entrypoint chooses a balanced
    /// shape from the source's variable count. Jolt's cycle-major streaming
    /// path sometimes commits a shorter dense source in the larger matrix
    /// shape dictated by the trace/one-hot batch. This Dory-specific entrypoint
    /// makes that layout choice explicit: the source is traversed with
    /// `2^sigma` columns, and one-hot sources reserve `2^nu` row slots.
    #[tracing::instrument(skip_all, name = "DoryScheme::commit_with_shape")]
    pub fn commit_with_shape<S>(
        source: &S,
        nu: usize,
        sigma: usize,
        setup: &DoryProverSetup,
    ) -> (DoryCommitment, DoryHint)
    where
        S: CommitmentSource<Fr> + ?Sized,
    {
        Self::commit_with_shape_mode::<S, Transparent>(source, nu, sigma, &setup.0)
    }

    /// Commits a hiding source using an explicit Dory matrix shape.
    #[tracing::instrument(skip_all, name = "DoryScheme::commit_zk_with_shape")]
    pub fn commit_zk_with_shape<S>(
        source: &S,
        nu: usize,
        sigma: usize,
        setup: &DoryProverSetup,
    ) -> (DoryCommitment, DoryHint)
    where
        S: CommitmentSource<Fr> + ?Sized,
    {
        Self::commit_with_shape_mode::<S, dory::ZK>(source, nu, sigma, &setup.0)
    }

    fn commit_batch_with_mode<B, M>(
        batch: &B,
        ids: &[B::Id],
        setup: &ArkworksProverSetup,
    ) -> Vec<(DoryCommitment, DoryHint)>
    where
        B: BatchCommitmentSource<Fr>,
        M: Mode,
    {
        if ids.is_empty() {
            return Vec::new();
        }

        let max_num_vars = ids
            .iter()
            .map(|&id| batch.num_vars(id))
            .max()
            .expect("ids is non-empty");
        let sigma = max_num_vars.div_ceil(2);
        let ctx = CommitRowContext::new(setup, 1usize << sigma);
        let row_major = batch.map_rows(sigma, ids, |_, row| commit_source_row(row, &ctx));

        let mut chunks_by_source: Vec<Vec<DoryChunkCommitment>> = (0..ids.len())
            .map(|_| Vec::with_capacity(row_major.len()))
            .collect();
        for row in row_major {
            assert_eq!(
                row.len(),
                ids.len(),
                "batch source returned a ragged row of committed sources",
            );
            for (source_chunks, chunk) in chunks_by_source.iter_mut().zip(row) {
                source_chunks.push(chunk);
            }
        }

        chunks_by_source
            .into_iter()
            .map(|chunks| aggregate_batch_chunks::<M>(chunks, setup))
            .collect()
    }

    fn open_source_with_mode<S, T, M>(
        source: &S,
        point: &[Fr],
        nu: usize,
        sigma: usize,
        setup: &ArkworksProverSetup,
        hint: DoryHint,
        transcript: &mut T,
    ) -> (DoryProof, Option<Fr>)
    where
        S: CommitmentSource<Fr> + ?Sized,
        T: DoryTranscript<Curve = InnerBN254>,
        M: Mode,
    {
        let adapter = DorySourceAdapter::new(source);
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();

        Self::open_dory_source_with_mode::<_, _, M>(
            &adapter, &ark_point, nu, sigma, setup, hint, transcript,
        )
    }

    fn open_dory_source_with_mode<S, T, M>(
        source: &S,
        ark_point: &[ArkFr],
        nu: usize,
        sigma: usize,
        setup: &ArkworksProverSetup,
        hint: DoryHint,
        transcript: &mut T,
    ) -> (DoryProof, Option<Fr>)
    where
        S: DoryPolynomial<ArkFr> + MultilinearLagrange<ArkFr>,
        T: DoryTranscript<Curve = InnerBN254>,
        M: Mode,
    {
        let (row_commitments, commit_blind) = hint.into_ark_parts();
        let (proof, y_blinding) =
            dory::prove::<ArkFr, InnerBN254, JoltG1Routines, JoltG2Routines, _, _, M>(
                source,
                ark_point,
                row_commitments,
                commit_blind,
                nu,
                sigma,
                setup,
                transcript,
            )
            .unwrap_or_else(|e| panic!("dory::prove failed: {e:?}"));

        (
            DoryProof(proof),
            y_blinding.map(|blind| ark_to_jolt_fr(&blind)),
        )
    }

    /// Opens a transparent Dory commitment for an arbitrary commitment source.
    ///
    /// This entrypoint is for protocol layers that already know the Dory matrix
    /// shape and already have a row-commitment hint. It preserves streaming
    /// opening paths without forcing the source through `DoryScheme::Polynomial`.
    #[tracing::instrument(skip_all, name = "DoryScheme::open_source_with_shape")]
    pub fn open_source_with_shape<S, T>(
        source: &S,
        point: &[Fr],
        nu: usize,
        sigma: usize,
        setup: &DoryProverSetup,
        hint: DoryHint,
        transcript: &mut T,
    ) -> DoryProof
    where
        S: CommitmentSource<Fr> + ?Sized,
        T: DoryTranscript<Curve = InnerBN254>,
    {
        let (proof, _blind) = Self::open_source_with_mode::<S, T, Transparent>(
            source, point, nu, sigma, &setup.0, hint, transcript,
        );
        proof
    }

    /// Opens a ZK/hiding Dory commitment for an arbitrary commitment source.
    ///
    /// Returns the proof, the hiding commitment to the evaluation, and the
    /// evaluation blinding scalar consumed later by BlindFold.
    #[tracing::instrument(skip_all, name = "DoryScheme::open_zk_source_with_shape")]
    pub fn open_zk_source_with_shape<S, T>(
        source: &S,
        point: &[Fr],
        nu: usize,
        sigma: usize,
        setup: &DoryProverSetup,
        hint: DoryHint,
        transcript: &mut T,
    ) -> (DoryProof, Bn254G1, Fr)
    where
        S: CommitmentSource<Fr> + ?Sized,
        T: DoryTranscript<Curve = InnerBN254>,
    {
        let (proof, y_blinding) = Self::open_source_with_mode::<S, T, dory::ZK>(
            source, point, nu, sigma, &setup.0, hint, transcript,
        );
        let y_com = ark_to_jolt_g1(proof.0.y_com.expect("ZK proof must contain y_com"));
        let blinding = y_blinding.expect("ZK proof must return y_blinding");
        (proof, y_com, blinding)
    }

    /// Verifies a transparent Dory opening using an already Dory-compatible
    /// transcript adapter.
    ///
    /// This is the verifier-side counterpart to `open_source_with_shape` for
    /// protocol layers that still own their transcript type but delegate Dory
    /// verification to this crate.
    #[tracing::instrument(skip_all, name = "DoryScheme::verify_with_shape")]
    pub fn verify_with_shape<T>(
        commitment: &DoryCommitment,
        point: &[Fr],
        eval: Fr,
        proof: &DoryProof,
        setup: &DoryVerifierSetup,
        transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        T: DoryTranscript<Curve = InnerBN254>,
    {
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let ark_eval = jolt_fr_to_ark(&eval);
        let ark_commitment = jolt_gt_to_ark(&commitment.0);

        dory::verify::<ArkFr, InnerBN254, JoltG1Routines, JoltG2Routines, _>(
            ark_commitment,
            ark_eval,
            &ark_point,
            &proof.0,
            setup.0.clone().into_inner(),
            transcript,
        )
        .map_err(|_| OpeningsError::VerificationFailed)
    }

    /// Verifies a ZK/hiding Dory opening using an already Dory-compatible
    /// transcript adapter.
    ///
    /// In ZK mode the evaluation is hidden and Dory verifies against the
    /// evaluation commitment embedded in the proof.
    #[tracing::instrument(skip_all, name = "DoryScheme::verify_zk_with_shape")]
    pub fn verify_zk_with_shape<T>(
        commitment: &DoryCommitment,
        point: &[Fr],
        proof: &DoryProof,
        setup: &DoryVerifierSetup,
        transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        T: DoryTranscript<Curve = InnerBN254>,
    {
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let dummy_eval = <ArkFr as DoryField>::zero();
        let ark_commitment = jolt_gt_to_ark(&commitment.0);

        dory::verify::<ArkFr, InnerBN254, JoltG1Routines, JoltG2Routines, _>(
            ark_commitment,
            dummy_eval,
            &ark_point,
            &proof.0,
            setup.0.clone().into_inner(),
            transcript,
        )
        .map_err(|_| OpeningsError::VerificationFailed)
    }
}

impl DeriveSetup<DoryProverSetup> for PedersenSetup<Bn254G1> {
    fn derive(source: &DoryProverSetup, capacity: usize) -> Self {
        assert!(
            capacity <= source.0.g1_vec.len(),
            "Pedersen capacity ({}) exceeds Dory SRS size ({})",
            capacity,
            source.0.g1_vec.len(),
        );
        let generators = ark_to_jolt_g1_vec(source.0.g1_vec[..capacity].to_vec());
        let blinding = ark_to_jolt_g1(source.0.h1);
        PedersenSetup::new(generators, blinding)
    }
}

impl Commitment for DoryScheme {
    type Output = DoryCommitment;
}

impl CommitmentSchemeVerifier for DoryScheme {
    type Field = Fr;
    type Proof = DoryProof;
    type BatchProof = Vec<DoryProof>;
    type VerifierSetup = DoryVerifierSetup;

    #[tracing::instrument(skip_all, name = "DoryScheme::verify")]
    fn verify(
        commitment: &Self::Output,
        point: &[Fr],
        eval: Fr,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        Self::verify_with_shape(commitment, point, eval, proof, setup, &mut dory_transcript)
    }

    fn verify_batch(
        claims: Vec<OpeningClaim<Self::Field, Self>>,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        homomorphic_verify_batch::<Self, _>(claims, proof, setup, transcript)
    }

    fn verify_fused_batch(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        let [proof] = proof.as_slice() else {
            return Err(OpeningsError::VerificationFailed);
        };
        Self::verify(commitment, point, eval, proof, setup, transcript)
    }

    fn bind_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        eval: &Self::Field,
    ) {
        transcript.append(&LabelWithCount(b"dory_opening_point", point.len() as u64));
        for p in point {
            p.append_to_transcript(transcript);
        }
        transcript.append(&Label(b"dory_opening_eval"));
        eval.append_to_transcript(transcript);
    }
}

impl PublicVerifierSetup for DoryScheme {
    type PublicParams = usize;

    fn verifier_setup(max_num_vars: Self::PublicParams) -> DoryVerifierSetup {
        Self::setup_verifier(max_num_vars)
    }
}

impl CommitmentScheme for DoryScheme {
    type ProverSetup = DoryProverSetup;
    type OpeningHint = DoryHint;
    type SetupParams = usize;

    fn setup(max_num_vars: Self::SetupParams) -> (DoryProverSetup, DoryVerifierSetup) {
        let prover = Self::setup_prover(max_num_vars);
        let verifier = Self::project_verifier_setup(&prover);
        (prover, verifier)
    }

    fn project_verifier_setup(prover_setup: &DoryProverSetup) -> DoryVerifierSetup {
        DoryVerifierSetup(prover_setup.0.to_verifier_setup())
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::commit")]
    fn commit<S: CommitmentSource<Fr> + ?Sized>(
        source: &S,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit_with_mode::<S, Transparent>(source, &setup.0)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::commit_batch")]
    fn commit_batch<B: BatchCommitmentSource<Self::Field>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Output, Self::OpeningHint)> {
        Self::commit_batch_with_mode::<B, Transparent>(batch, ids, &setup.0)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::open")]
    fn open<S>(
        poly: &S,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof
    where
        S: CommitmentSource<Self::Field> + ?Sized,
    {
        let num_vars = point.len();
        let sigma = num_vars.div_ceil(2);
        let nu = num_vars - sigma;

        let hint = match hint {
            Some(hint) => hint,
            None => DoryHint::new(
                ark_to_jolt_g1_vec(compute_row_commitments(poly, &setup.0)),
                Fr::from_u64(0),
            ),
        };
        debug_assert!(
            hint.commit_blind == Fr::from_u64(0),
            "commit_blind should be 0 for transparent mode"
        );
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        Self::open_source_with_shape(poly, point, nu, sigma, setup, hint, &mut dory_transcript)
    }

    fn prove_batch<S>(
        claims: Vec<ProverClaim<Self::Field, S>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::BatchProof
    where
        S: CommitmentSource<Self::Field>,
    {
        homomorphic_prove_batch::<Self, _, _>(claims, hints, setup, transcript)
    }

    fn prove_fused_batch<S>(
        polynomial: &S,
        point: &[Self::Field],
        eval: Self::Field,
        hint: Option<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::BatchProof
    where
        S: CommitmentSource<Self::Field> + ?Sized,
    {
        vec![Self::open(polynomial, point, eval, setup, hint, transcript)]
    }
}

impl ShapedCommitmentScheme for DoryScheme {
    fn commit_with_shape<S: CommitmentSource<Fr> + ?Sized>(
        source: &S,
        nu: usize,
        sigma: usize,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        DoryScheme::commit_with_shape(source, nu, sigma, setup)
    }
}

impl AdditivelyHomomorphicVerifier for DoryScheme {
    #[tracing::instrument(skip_all, name = "DoryScheme::combine")]
    fn combine(commitments: &[Self::Output], scalars: &[Self::Field]) -> Self::Output {
        assert_eq!(commitments.len(), scalars.len());

        let combined = commitments
            .par_iter()
            .zip(scalars.par_iter())
            .map(|(c, s)| jolt_fr_to_ark(s) * jolt_gt_to_ark(&c.0))
            .reduce(ArkGT::identity, |acc, x| acc + x);

        DoryCommitment(ark_to_jolt_gt(&combined))
    }
}

impl AdditivelyHomomorphic for DoryScheme {
    #[tracing::instrument(skip_all, name = "DoryScheme::combine_hints")]
    fn combine_hints(hints: Vec<Self::OpeningHint>, scalars: &[Self::Field]) -> Self::OpeningHint {
        assert_eq!(hints.len(), scalars.len());
        assert!(!hints.is_empty(), "combine_hints: empty hint set");

        let num_rows = hints
            .iter()
            .map(|hint| hint.row_commitments.len())
            .max()
            .unwrap_or(0);

        let combined_blind = hints
            .iter()
            .zip(scalars.iter())
            .map(|(hint, &scalar)| scalar * hint.commit_blind)
            .sum();

        let combined: Vec<Bn254G1> = (0..num_rows)
            .into_par_iter()
            .map(|row| {
                let mut acc = Bn254G1::default();
                for (hint, &scalar) in hints.iter().zip(scalars.iter()) {
                    if let Some(row_commitment) = hint.row_commitments.get(row) {
                        acc += row_commitment.scalar_mul(&scalar);
                    }
                }
                acc
            })
            .collect();

        DoryHint::new(combined, combined_blind)
    }
}

impl ZkOpeningSchemeVerifier for DoryScheme {
    type HidingCommitment = Bn254G1;

    #[tracing::instrument(skip_all, name = "DoryScheme::verify_zk")]
    fn verify_zk(
        commitment: &Self::Output,
        point: &[Fr],
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        Self::verify_zk_with_shape(commitment, point, proof, setup, &mut dory_transcript)
    }

    fn verify_batch_zk(
        claims: Vec<OpeningClaim<Self::Field, Self>>,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        let [claim] = claims.as_slice() else {
            return Err(OpeningsError::VerificationFailed);
        };
        let [proof] = proof.as_slice() else {
            return Err(OpeningsError::VerificationFailed);
        };
        Self::verify_zk(&claim.commitment, &claim.point, proof, setup, transcript)
    }

    fn verify_fused_batch_zk(
        commitment: &Self::Output,
        point: &[Self::Field],
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        let [proof] = proof.as_slice() else {
            return Err(OpeningsError::VerificationFailed);
        };
        Self::verify_zk(commitment, point, proof, setup, transcript)
    }

    fn bind_zk_opening_inputs(
        transcript: &mut impl Transcript<Challenge = Self::Field>,
        point: &[Self::Field],
        hiding_commitment: &Self::HidingCommitment,
    ) {
        transcript.append(&LabelWithCount(b"dory_opening_point", point.len() as u64));
        for p in point {
            p.append_to_transcript(transcript);
        }
        transcript.append(&Label(b"dory_eval_commitment"));
        hiding_commitment.append_to_transcript(transcript);
    }
}

impl ZkOpeningScheme for DoryScheme {
    type Blind = Fr;

    fn commit_zk<S: CommitmentSource<Fr> + ?Sized>(
        source: &S,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit_with_mode::<S, dory::ZK>(source, &setup.0)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::commit_batch_zk")]
    fn commit_batch_zk<B: BatchCommitmentSource<Self::Field>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Output, Self::OpeningHint)> {
        Self::commit_batch_with_mode::<B, dory::ZK>(batch, ids, &setup.0)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::open_zk")]
    fn open_zk<S>(
        poly: &S,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind)
    where
        S: CommitmentSource<Self::Field> + ?Sized,
    {
        let num_vars = point.len();
        let sigma = num_vars.div_ceil(2);
        let nu = num_vars - sigma;
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        Self::open_zk_source_with_shape(poly, point, nu, sigma, setup, hint, &mut dory_transcript)
    }

    fn prove_batch_zk<S>(
        claims: Vec<ProverClaim<Self::Field, S>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::BatchProof, Self::HidingCommitment, Self::Blind)
    where
        S: CommitmentSource<Self::Field>,
    {
        let [claim] = claims.as_slice() else {
            panic!("Dory ZK batch opening expects one already-combined claim");
        };
        let [hint] = hints.as_slice() else {
            panic!("Dory ZK batch opening expects one already-combined hint");
        };
        Self::prove_fused_batch_zk(
            &claim.polynomial,
            &claim.point,
            claim.eval,
            hint.clone(),
            setup,
            transcript,
        )
    }

    fn prove_fused_batch_zk<S>(
        polynomial: &S,
        point: &[Self::Field],
        eval: Self::Field,
        hint: Self::OpeningHint,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::BatchProof, Self::HidingCommitment, Self::Blind)
    where
        S: CommitmentSource<Self::Field> + ?Sized,
    {
        let (proof, y_com, y_blinding) =
            Self::open_zk(polynomial, point, eval, setup, hint, transcript);
        (vec![proof], y_com, y_blinding)
    }
}

impl ShapedZkOpeningScheme for DoryScheme {
    fn commit_zk_with_shape<S: CommitmentSource<Fr> + ?Sized>(
        source: &S,
        nu: usize,
        sigma: usize,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        DoryScheme::commit_zk_with_shape(source, nu, sigma, setup)
    }
}

impl EvaluationCommitmentScheme<Bn254G1> for DoryScheme {
    fn batch_eval_commitment(proof: &Self::BatchProof) -> Option<Bn254G1> {
        let [proof] = proof.as_slice() else {
            return None;
        };
        proof.0.y_com.as_ref().copied().map(ark_to_jolt_g1)
    }

    fn eval_commitment_gens_verifier(setup: &Self::VerifierSetup) -> Option<(Bn254G1, Bn254G1)> {
        Some((ark_to_jolt_g1(setup.0.g1_0), ark_to_jolt_g1(setup.0.h1)))
    }
}

impl EvaluationCommitmentProver<Bn254G1> for DoryScheme {
    fn eval_commitment_gens(setup: &Self::ProverSetup) -> Option<(Bn254G1, Bn254G1)> {
        let g1_0 = setup.0.g1_vec.first().copied().map(ark_to_jolt_g1)?;
        Some((g1_0, ark_to_jolt_g1(setup.0.h1)))
    }

    fn zk_generators(setup: &Self::ProverSetup, count: usize) -> Option<(Vec<Bn254G1>, Bn254G1)> {
        let count = std::cmp::min(count, setup.0.g1_vec.len());
        let g1s = ark_to_jolt_g1_vec(setup.0.g1_vec[..count].to_vec());
        Some((g1s, ark_to_jolt_g1(setup.0.h1)))
    }
}

enum DoryChunkCommitment {
    Dense(ArkG1),
    OneHot(Vec<ArkG1>),
}

struct CommitRowContext<'a> {
    setup: &'a ArkworksProverSetup,
    g1_bases_affine: Vec<G1Affine>,
}

impl<'a> CommitRowContext<'a> {
    fn new(setup: &'a ArkworksProverSetup, row_len: usize) -> Self {
        Self {
            setup,
            g1_bases_affine: g1_bases_affine(setup, row_len),
        }
    }
}

/// Dense commit: full MSM per row, parallel over rows.
fn commit_rows_dense<S: CommitmentSource<Fr> + ?Sized>(
    source: &S,
    sigma: usize,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let num_cols = 1usize << sigma;
    let ctx = CommitRowContext::new(setup, num_cols);

    let chunks = source.map_rows(sigma, |_, row| commit_source_row(row, &ctx));
    flatten_chunks(chunks)
}

fn commit_field_row(values: &[Fr], setup: &ArkworksProverSetup) -> ArkG1 {
    assert!(
        values.len() <= setup.g1_vec.len(),
        "Dory row length ({}) exceeds G1 SRS size ({})",
        values.len(),
        setup.g1_vec.len(),
    );
    let scalars: Vec<ArkFr> = values.iter().map(jolt_fr_to_ark).collect();
    JoltG1Routines::msm(&setup.g1_vec[..scalars.len()], &scalars)
}

fn strided_affine_bases(
    values_len: usize,
    column_stride: usize,
    ctx: &CommitRowContext<'_>,
) -> Vec<G1Affine> {
    assert!(
        column_stride > 0,
        "Dory strided row column stride must be nonzero"
    );
    assert!(
        values_len == 0 || (values_len - 1) * column_stride < ctx.g1_bases_affine.len(),
        "Dory strided row length ({values_len}) with stride ({column_stride}) exceeds G1 SRS row size ({})",
        ctx.g1_bases_affine.len(),
    );
    ctx.g1_bases_affine
        .iter()
        .step_by(column_stride)
        .take(values_len)
        .copied()
        .collect()
}

fn commit_strided_field_row(
    values: &[Fr],
    column_stride: usize,
    ctx: &CommitRowContext<'_>,
) -> ArkG1 {
    assert!(
        column_stride > 0,
        "Dory strided row column stride must be nonzero"
    );
    assert!(
        values.is_empty() || (values.len() - 1) * column_stride < ctx.setup.g1_vec.len(),
        "Dory strided row length ({}) with stride ({column_stride}) exceeds G1 SRS size ({})",
        values.len(),
        ctx.setup.g1_vec.len(),
    );
    let bases: Vec<ArkG1Struct> = ctx
        .setup
        .g1_vec
        .iter()
        .step_by(column_stride)
        .take(values.len())
        .copied()
        .collect();
    let scalars: Vec<ArkFr> = values.iter().map(jolt_fr_to_ark).collect();
    JoltG1Routines::msm(&bases, &scalars)
}

fn commit_i128_row(values: &[i128], ctx: &CommitRowContext<'_>) -> ArkG1 {
    assert!(
        values.len() <= ctx.g1_bases_affine.len(),
        "Dory row length ({}) exceeds G1 SRS size ({})",
        values.len(),
        ctx.g1_bases_affine.len(),
    );
    ArkG1Struct(ark_ec::scalar_mul::variable_base::msm_i128::<G1Projective>(
        &ctx.g1_bases_affine[..values.len()],
        values,
        true,
    ))
}

fn commit_strided_i128_row(
    values: &[i128],
    column_stride: usize,
    ctx: &CommitRowContext<'_>,
) -> ArkG1 {
    let bases = strided_affine_bases(values.len(), column_stride, ctx);
    ArkG1Struct(ark_ec::scalar_mul::variable_base::msm_i128::<G1Projective>(
        &bases, values, true,
    ))
}

fn commit_u64_row(values: &[u64], ctx: &CommitRowContext<'_>) -> ArkG1 {
    assert!(
        values.len() <= ctx.g1_bases_affine.len(),
        "Dory row length ({}) exceeds G1 SRS size ({})",
        values.len(),
        ctx.g1_bases_affine.len(),
    );
    ArkG1Struct(ark_ec::scalar_mul::variable_base::msm_u64::<G1Projective>(
        &ctx.g1_bases_affine[..values.len()],
        values,
        true,
    ))
}

fn commit_strided_u64_row(
    values: &[u64],
    column_stride: usize,
    ctx: &CommitRowContext<'_>,
) -> ArkG1 {
    let bases = strided_affine_bases(values.len(), column_stride, ctx);
    ArkG1Struct(ark_ec::scalar_mul::variable_base::msm_u64::<G1Projective>(
        &bases, values, true,
    ))
}

/// One-hot commit: O(T) group additions for unit-valued one-hot polynomials.
fn commit_rows_one_hot<S: CommitmentSource<Fr> + ?Sized>(
    source: &S,
    num_rows: usize,
    num_cols: usize,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let g1_bases = &setup.g1_vec[..num_cols];

    let mut cols_per_row: Vec<Vec<usize>> = vec![Vec::new(); num_rows];
    source.for_each_one(|flat_idx: usize| {
        let row = flat_idx / num_cols;
        let col = flat_idx % num_cols;
        debug_assert!(
            row < num_rows && col < num_cols,
            "for_each_one out-of-bounds flat_idx: row={row} num_rows={num_rows} col={col} num_cols={num_cols}",
        );
        cols_per_row[row].push(col);
    });

    cols_per_row
        .par_iter()
        .map(|cols| {
            cols.iter()
                .fold(<InnerBN254 as PairingCurve>::G1::identity(), |acc, &col| {
                    <InnerBN254 as PairingCurve>::G1::add(&acc, &g1_bases[col])
                })
        })
        .collect()
}

fn commit_one_hot_row(row: jolt_openings::OneHotRow<'_>, ctx: &CommitRowContext<'_>) -> Vec<ArkG1> {
    let k = 1usize << row.log_domain_size;
    let num_columns = match row.entries {
        jolt_openings::OneHotEntries::OnePerColumn(indices) => indices.len(),
        jolt_openings::OneHotEntries::MaybeZero(indices) => indices.len(),
    };
    assert!(
        num_columns <= ctx.g1_bases_affine.len(),
        "Dory one-hot row length ({}) exceeds G1 SRS size ({})",
        num_columns,
        ctx.g1_bases_affine.len(),
    );

    let mut columns_by_hot_index: Vec<Vec<usize>> = vec![Vec::new(); k];
    match row.entries {
        jolt_openings::OneHotEntries::OnePerColumn(indices) => {
            for (column, hot_index) in indices.iter().enumerate() {
                columns_by_hot_index[hot_index.get()].push(column);
            }
        }
        jolt_openings::OneHotEntries::MaybeZero(indices) => {
            for (column, hot_index) in indices.iter().enumerate() {
                if let Some(hot_index) = hot_index {
                    columns_by_hot_index[hot_index.get()].push(column);
                }
            }
        }
    }

    batch_g1_additions_multi_affine(&ctx.g1_bases_affine[..num_columns], &columns_by_hot_index)
        .into_iter()
        .map(|affine| ArkG1Struct(affine.into()))
        .collect()
}

fn g1_bases_affine(setup: &ArkworksProverSetup, len: usize) -> Vec<G1Affine> {
    setup.g1_vec[..len]
        .par_iter()
        .map(|base| base.0.into_affine())
        .collect()
}

fn commit_source_row(row: SourceRow<'_, Fr>, ctx: &CommitRowContext<'_>) -> DoryChunkCommitment {
    match row {
        SourceRow::FieldElements(values) => {
            DoryChunkCommitment::Dense(commit_field_row(values, ctx.setup))
        }
        SourceRow::StridedFieldElements {
            values,
            column_stride,
        } => DoryChunkCommitment::Dense(commit_strided_field_row(values, column_stride, ctx)),
        SourceRow::I128(values) => DoryChunkCommitment::Dense(commit_i128_row(values, ctx)),
        SourceRow::StridedI128 {
            values,
            column_stride,
        } => DoryChunkCommitment::Dense(commit_strided_i128_row(values, column_stride, ctx)),
        SourceRow::U64(values) => DoryChunkCommitment::Dense(commit_u64_row(values, ctx)),
        SourceRow::StridedU64 {
            values,
            column_stride,
        } => DoryChunkCommitment::Dense(commit_strided_u64_row(values, column_stride, ctx)),
        SourceRow::OneHot(row) => DoryChunkCommitment::OneHot(commit_one_hot_row(row, ctx)),
    }
}

fn flatten_chunks(chunks: Vec<DoryChunkCommitment>) -> Vec<ArkG1> {
    let Some(first) = chunks.first() else {
        return Vec::new();
    };

    match first {
        DoryChunkCommitment::Dense(_) => chunks
            .into_iter()
            .map(|chunk| match chunk {
                DoryChunkCommitment::Dense(row_commitment) => row_commitment,
                DoryChunkCommitment::OneHot(_) => {
                    panic!("source mixed dense and one-hot rows during commitment");
                }
            })
            .collect(),
        DoryChunkCommitment::OneHot(first) => {
            let rows_per_hot_index = chunks.len();
            let k = first.len();
            let mut row_commitments =
                vec![<InnerBN254 as PairingCurve>::G1::identity(); rows_per_hot_index * k];
            for (chunk_index, chunk) in chunks.into_iter().enumerate() {
                match chunk {
                    DoryChunkCommitment::OneHot(commitments) => {
                        assert_eq!(
                            commitments.len(),
                            k,
                            "source changed one-hot domain size during commitment",
                        );
                        for (hot_index, row_commitment) in commitments.into_iter().enumerate() {
                            row_commitments[chunk_index + hot_index * rows_per_hot_index] =
                                row_commitment;
                        }
                    }
                    DoryChunkCommitment::Dense(_) => {
                        panic!("source mixed dense and one-hot rows during commitment");
                    }
                }
            }
            row_commitments
        }
    }
}

fn aggregate_batch_chunks<M: Mode>(
    chunks: Vec<DoryChunkCommitment>,
    setup: &ArkworksProverSetup,
) -> (DoryCommitment, DoryHint) {
    assert!(!chunks.is_empty(), "cannot aggregate an empty source");

    match &chunks[0] {
        DoryChunkCommitment::Dense(_) => {
            let mut row_commitments = Vec::with_capacity(chunks.len());
            for chunk in chunks {
                match chunk {
                    DoryChunkCommitment::Dense(row_commitment) => {
                        row_commitments.push(row_commitment);
                    }
                    DoryChunkCommitment::OneHot(_) => {
                        panic!("batch source mixed dense and one-hot rows for one source");
                    }
                }
            }
            finish_row_commitments::<M>(row_commitments, setup)
        }
        DoryChunkCommitment::OneHot(first) => {
            let rows_per_hot_index = chunks.len();
            let k = first.len();
            let mut row_commitments =
                vec![<InnerBN254 as PairingCurve>::G1::identity(); rows_per_hot_index * k];

            for (chunk_index, chunk) in chunks.into_iter().enumerate() {
                match chunk {
                    DoryChunkCommitment::OneHot(commitments) => {
                        assert_eq!(
                            commitments.len(),
                            k,
                            "batch source changed one-hot domain size within one source",
                        );
                        for (hot_index, row_commitment) in commitments.into_iter().enumerate() {
                            row_commitments[chunk_index + hot_index * rows_per_hot_index] =
                                row_commitment;
                        }
                    }
                    DoryChunkCommitment::Dense(_) => {
                        panic!("batch source mixed dense and one-hot rows for one source");
                    }
                }
            }
            finish_row_commitments::<M>(row_commitments, setup)
        }
    }
}

fn finish_row_commitments<M: Mode>(
    row_commitments: Vec<ArkG1>,
    setup: &ArkworksProverSetup,
) -> (DoryCommitment, DoryHint) {
    let (tier_2, commit_blind) = commit_rows_tier_2::<M>(&row_commitments, setup);
    (
        DoryCommitment(ark_to_jolt_gt(&tier_2)),
        DoryHint::new(
            ark_to_jolt_g1_vec(row_commitments),
            ark_to_jolt_fr(&commit_blind),
        ),
    )
}

fn compute_row_commitments<S: CommitmentSource<Fr> + ?Sized>(
    source: &S,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let num_vars = source.num_vars();
    let sigma = num_vars.div_ceil(2);
    let nu = num_vars - sigma;

    compute_row_commitments_with_shape(source, nu, sigma, setup)
}

fn compute_row_commitments_with_shape<S: CommitmentSource<Fr> + ?Sized>(
    source: &S,
    nu: usize,
    sigma: usize,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let num_cols = 1usize << sigma;
    let num_rows = 1usize << nu;
    if source.is_one_hot() {
        commit_rows_one_hot(source, num_rows, num_cols, setup)
    } else {
        commit_rows_dense(source, sigma, setup)
    }
}

pub(crate) fn commit_rows_tier_2<M: Mode>(
    row_commitments: &[ArkG1],
    setup: &ArkworksProverSetup,
) -> (ArkGT, ArkFr) {
    let g2_bases = &setup.g2_vec[..row_commitments.len()];
    let tier_2 = <InnerBN254 as PairingCurve>::multi_pair_g2_setup(row_commitments, g2_bases);
    let commit_blind = M::sample::<ArkFr>();
    let tier_2 = M::mask(tier_2, &setup.ht, &commit_blind);
    (tier_2, commit_blind)
}

impl DoryHint {
    fn into_ark_parts(self) -> (Vec<ArkG1>, ArkFr) {
        (
            jolt_g1_vec_to_ark(self.row_commitments),
            jolt_fr_to_ark(&self.commit_blind),
        )
    }
}

/// Adapts [`CommitmentSource<Fr>`] to dory-pcs's polynomial traits
/// without materializing the full evaluation table.
struct DorySourceAdapter<'a, S: CommitmentSource<Fr> + ?Sized> {
    source: &'a S,
}

impl<'a, S: CommitmentSource<Fr> + ?Sized> DorySourceAdapter<'a, S> {
    fn new(source: &'a S) -> Self {
        Self { source }
    }
}

impl<S: CommitmentSource<Fr> + ?Sized> DoryPolynomial<ArkFr> for DorySourceAdapter<'_, S> {
    fn num_vars(&self) -> usize {
        self.source.num_vars()
    }

    fn evaluate(&self, point: &[ArkFr]) -> ArkFr {
        let native_point: Vec<Fr> = point.iter().rev().map(ark_to_jolt_fr).collect();
        jolt_fr_to_ark(&self.source.evaluate(&native_point))
    }

    fn commit<E, Mo, M1>(
        &self,
        _nu: usize,
        _sigma: usize,
        _setup: &dory::setup::ProverSetup<E>,
    ) -> Result<(E::GT, Vec<E::G1>, ArkFr), dory::error::DoryError>
    where
        E: PairingCurve,
        Mo: dory::mode::Mode,
        M1: DoryRoutines<E::G1>,
        E::G1: DoryGroup<Scalar = ArkFr>,
    {
        unimplemented!(
            "DoryScheme pre-computes row commitments before invoking dory::prove; \
             dory::Polynomial::commit on this adapter is not exercised"
        )
    }
}

impl<S: CommitmentSource<Fr> + ?Sized> MultilinearLagrange<ArkFr> for DorySourceAdapter<'_, S> {
    fn vector_matrix_product(&self, left_vec: &[ArkFr], _nu: usize, sigma: usize) -> Vec<ArkFr> {
        let native_left: Vec<Fr> = left_vec.iter().map(ark_to_jolt_fr).collect();
        let result = self.source.fold_rows(&native_left, sigma);
        result.iter().map(jolt_fr_to_ark).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_crypto::{Bn254, JoltGroup, Pedersen, VectorCommitment};
    use jolt_field::{FromPrimitiveInt, RandomSampling};
    use jolt_openings::SourceRow;
    use jolt_poly::Polynomial;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    struct FoldOnlySource {
        poly: Polynomial<Fr>,
    }

    struct U64Source {
        evaluations: Vec<u64>,
    }

    struct StridedU64Source {
        rows: Vec<Vec<u64>>,
        dense: Polynomial<Fr>,
        column_stride: usize,
    }

    impl U64Source {
        fn field_evaluation(&self, point: &[Fr]) -> Fr {
            let dense: Vec<Fr> = self
                .evaluations
                .iter()
                .map(|&value| Fr::from_u64(value))
                .collect();
            dense.evaluate(point)
        }
    }

    impl CommitmentSource<Fr> for U64Source {
        fn num_vars(&self) -> usize {
            self.evaluations.len().ilog2() as usize
        }

        fn evaluate(&self, point: &[Fr]) -> Fr {
            self.field_evaluation(point)
        }

        fn for_each_row<V>(&self, sigma: usize, mut visit: V)
        where
            V: for<'row> FnMut(usize, SourceRow<'row, Fr>),
        {
            let row_len = 1usize << sigma;
            for (row_index, row) in self.evaluations.chunks(row_len).enumerate() {
                visit(row_index, SourceRow::U64(row));
            }
        }

        fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
            let row_len = 1usize << sigma;
            let mut result = vec![Fr::from_u64(0); row_len];
            for (row_index, row) in self.evaluations.chunks(row_len).enumerate() {
                let weight = left[row_index];
                for (dest, &value) in result.iter_mut().zip(row) {
                    *dest += Fr::from_u64(value) * weight;
                }
            }
            result
        }
    }

    impl CommitmentSource<Fr> for StridedU64Source {
        fn num_vars(&self) -> usize {
            self.dense.num_vars()
        }

        fn evaluate(&self, point: &[Fr]) -> Fr {
            self.dense.evaluate(point)
        }

        fn for_each_row<V>(&self, _sigma: usize, mut visit: V)
        where
            V: for<'row> FnMut(usize, SourceRow<'row, Fr>),
        {
            for (row_index, row) in self.rows.iter().enumerate() {
                visit(
                    row_index,
                    SourceRow::StridedU64 {
                        values: row,
                        column_stride: self.column_stride,
                    },
                );
            }
        }

        fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
            self.dense.fold_rows(left, sigma)
        }
    }

    impl CommitmentSource<Fr> for FoldOnlySource {
        fn num_vars(&self) -> usize {
            self.poly.num_vars()
        }

        fn evaluate(&self, point: &[Fr]) -> Fr {
            self.poly.evaluate(point)
        }

        fn for_each_row<V>(&self, _sigma: usize, _visit: V)
        where
            V: for<'row> FnMut(usize, SourceRow<'row, Fr>),
        {
            panic!("single-claim prove_batch must not materialize source rows")
        }

        fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
            self.poly.fold_rows(left, sigma)
        }
    }

    #[test]
    fn commit_open_verify_round_trip() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"test");
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        );

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"test");
        let result = DoryScheme::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(result.is_ok(), "Verification failed: {result:?}");
    }

    #[test]
    fn u64_source_commit_open_verify_round_trip() {
        let num_vars = 4;
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        let source = U64Source {
            evaluations: (0..(1 << num_vars)).map(|value| value as u64).collect(),
        };
        let point: Vec<Fr> = (0..num_vars)
            .map(|idx| Fr::from_u64((idx + 2) as u64))
            .collect();
        let eval = source.evaluate(&point);

        let (commitment, hint) = DoryScheme::commit(&source, &prover_setup);
        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"test-u64");
        let proof = DoryScheme::open(
            &source,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        );

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"test-u64");
        let result = DoryScheme::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn strided_u64_commit_matches_dense_zero_padded_rows() {
        let prover_setup = DoryScheme::setup_prover(6);
        let mut dense = vec![Fr::from_u64(0); 16];
        dense[0] = Fr::from_u64(3);
        dense[4] = Fr::from_u64(5);
        dense[8] = Fr::from_u64(7);
        dense[12] = Fr::from_u64(11);

        let source = StridedU64Source {
            rows: vec![vec![3, 5], vec![7, 11]],
            dense: Polynomial::new(dense.clone()),
            column_stride: 4,
        };
        let dense_source = Polynomial::new(dense);

        let (strided_commitment, strided_hint) =
            DoryScheme::commit_with_shape(&source, 1, 3, &prover_setup);
        let (dense_commitment, dense_hint) =
            DoryScheme::commit_with_shape(&dense_source, 1, 3, &prover_setup);

        assert_eq!(strided_commitment, dense_commitment);
        assert_eq!(strided_hint.row_commitments, dense_hint.row_commitments);
    }

    #[test]
    fn combine_commitments_homomorphic() {
        let num_vars = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(300);

        let prover_setup = DoryScheme::setup_prover(num_vars);

        let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
        let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);

        let (commit_a, _) = DoryScheme::commit(poly_a.evaluations(), &prover_setup);
        let (commit_b, _) = DoryScheme::commit(poly_b.evaluations(), &prover_setup);

        let sum_evals: Vec<Fr> = poly_a
            .evaluations()
            .iter()
            .zip(poly_b.evaluations().iter())
            .map(|(a, b)| *a + *b)
            .collect();
        let (commit_sum_direct, _) = DoryScheme::commit(&sum_evals, &prover_setup);

        let combined = DoryScheme::combine(
            &[commit_a, commit_b],
            &[
                <Fr as FromPrimitiveInt>::from_u64(1),
                <Fr as FromPrimitiveInt>::from_u64(1),
            ],
        );

        assert_eq!(
            commit_sum_direct, combined,
            "combine([1,1]) must match commitment to sum"
        );
    }

    #[test]
    fn combine_hints_zero_pads_ragged_rows() {
        let g = Bn254::g1_generator();
        let h = g.scalar_mul(&Fr::from_u64(11));
        let k = g.scalar_mul(&Fr::from_u64(13));
        let a = Fr::from_u64(2);
        let b = Fr::from_u64(7);

        let hint_a = DoryHint::new(vec![g], Fr::from_u64(3));
        let hint_b = DoryHint::new(vec![h, k], Fr::from_u64(5));

        let combined = DoryScheme::combine_hints(vec![hint_a, hint_b], &[a, b]);

        assert_eq!(combined.row_commitments.len(), 2);
        assert_eq!(
            combined.row_commitments[0],
            g.scalar_mul(&a) + h.scalar_mul(&b)
        );
        assert_eq!(combined.row_commitments[1], k.scalar_mul(&b));
        assert_eq!(
            combined.commit_blind,
            a * Fr::from_u64(3) + b * Fr::from_u64(5),
            "combined hint blind must match the same linear combination"
        );
    }

    #[test]
    fn generic_open_preserves_jolt_point_order() {
        let num_vars = 3;
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        // f(x0, x1, x2) = x0. Reversing the opening point evaluates x2
        // instead, so this catches accidental Dory/Jolt point-order swaps.
        let evals = vec![
            Fr::from_u64(0),
            Fr::from_u64(0),
            Fr::from_u64(0),
            Fr::from_u64(0),
            Fr::from_u64(1),
            Fr::from_u64(1),
            Fr::from_u64(1),
            Fr::from_u64(1),
        ];
        let poly = Polynomial::new(evals);
        let point = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let eval = poly.evaluate(&point);

        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"point-order");
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prove_transcript,
        );

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"point-order");
        let result = DoryScheme::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(
            result.is_ok(),
            "generic Dory opening must evaluate at the caller's Jolt-order point"
        );
    }

    #[test]
    fn single_claim_prove_batch_opens_source_without_materializing_rows() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(414);
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let source = FoldOnlySource { poly: poly.clone() };
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let (commitment, hint) = DoryScheme::commit(&poly, &prover_setup);

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"single-batch");
        let proof = DoryScheme::prove_batch(
            vec![ProverClaim {
                polynomial: source,
                point: point.clone(),
                eval,
            }],
            vec![hint],
            &prover_setup,
            &mut prove_transcript,
        );

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"single-batch");
        DoryScheme::verify_batch(
            vec![OpeningClaim {
                commitment,
                point,
                eval,
            }],
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        )
        .expect("single-claim batch proof should verify");
    }

    #[test]
    fn fused_batch_opens_source_without_batch_transcript_rlc() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(415);
        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::project_verifier_setup(&prover_setup);

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let source = FoldOnlySource { poly: poly.clone() };
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let (commitment, hint) = DoryScheme::commit(&poly, &prover_setup);

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"fused-batch");
        let proof = DoryScheme::prove_fused_batch(
            &source,
            &point,
            eval,
            Some(hint),
            &prover_setup,
            &mut prove_transcript,
        );

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"fused-batch");
        DoryScheme::verify_fused_batch(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        )
        .expect("already-fused batch proof should verify");
    }

    #[test]
    fn zk_open_verify_round_trip() {
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(600);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(prover_setup.0.to_verifier_setup());

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);

        let (commitment, hint) =
            <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"zk-test");
        let (proof, _eval_com, _blinding) = DoryScheme::open_zk(
            &poly,
            &point,
            eval,
            &prover_setup,
            hint,
            &mut prove_transcript,
        );

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"zk-test");
        let result = DoryScheme::verify_zk(
            &commitment,
            &point,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        );
        assert!(result.is_ok(), "ZK verification failed: {result:?}");
    }

    #[test]
    fn zk_single_claim_batch_round_trip() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(601);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::project_verifier_setup(&prover_setup);

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let (commitment, hint) = DoryScheme::commit_zk(&poly, &prover_setup);

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"zk-batch");
        let (proof, y_com, _blind) = DoryScheme::prove_batch_zk(
            vec![ProverClaim {
                polynomial: poly,
                point: point.clone(),
                eval,
            }],
            vec![hint],
            &prover_setup,
            &mut prove_transcript,
        );
        assert_eq!(
            DoryScheme::batch_eval_commitment(&proof),
            Some(y_com),
            "batch proof should expose the hidden evaluation commitment"
        );

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"zk-batch");
        DoryScheme::verify_batch_zk(
            vec![OpeningClaim {
                commitment,
                point,
                eval,
            }],
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        )
        .expect("ZK batch proof should verify");
    }

    #[test]
    fn zk_fused_batch_round_trip() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(602);

        let prover_setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryScheme::project_verifier_setup(&prover_setup);

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let point: Vec<Fr> = (0..num_vars)
            .map(|_| <Fr as RandomSampling>::random(&mut rng))
            .collect();
        let eval = poly.evaluate(&point);
        let (commitment, hint) = DoryScheme::commit_zk(&poly, &prover_setup);

        let mut prove_transcript = jolt_transcript::Blake2bTranscript::new(b"zk-fused-batch");
        let (proof, y_com, _blind) = DoryScheme::prove_fused_batch_zk(
            &poly,
            &point,
            eval,
            hint,
            &prover_setup,
            &mut prove_transcript,
        );
        assert_eq!(DoryScheme::batch_eval_commitment(&proof), Some(y_com));

        let mut verify_transcript = jolt_transcript::Blake2bTranscript::new(b"zk-fused-batch");
        DoryScheme::verify_fused_batch_zk(
            &commitment,
            &point,
            &proof,
            &verifier_setup,
            &mut verify_transcript,
        )
        .expect("ZK fused batch proof should verify");
    }

    #[test]
    fn extract_vc_setup_produces_valid_pedersen_setup() {
        let num_vars = 6;
        let prover_setup = DoryScheme::setup_prover(num_vars);

        let capacity = 5;
        let vc_setup = PedersenSetup::<Bn254G1>::derive(&prover_setup, capacity);

        assert_eq!(
            <Pedersen<Bn254G1> as VectorCommitment>::capacity(&vc_setup),
            capacity,
        );

        let values = vec![
            <Fr as FromPrimitiveInt>::from_u64(1),
            <Fr as FromPrimitiveInt>::from_u64(2),
            <Fr as FromPrimitiveInt>::from_u64(3),
        ];
        let blinding = <Fr as FromPrimitiveInt>::from_u64(42);
        let commitment =
            <Pedersen<Bn254G1> as VectorCommitment>::commit(&vc_setup, &values, &blinding);
        assert!(<Pedersen<Bn254G1> as VectorCommitment>::verify(
            &vc_setup,
            &commitment,
            &values,
            &blinding,
        ),);
    }
}
