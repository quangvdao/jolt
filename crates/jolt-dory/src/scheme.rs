//! Dory PCS implementing the `jolt-openings` trait hierarchy.

#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::unimplemented,
    reason = "ZK proof y_com/y_blinding are Dory-mode invariants; dory::prove/verify errors are caller-precondition violations surfaced via panic; the dory adapter's commit is unreachable because DoryScheme pre-computes row commitments"
)]

use dory::backends::arkworks::{ArkworksProverSetup, G1Routines, G2Routines};
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
    CommitmentSchemeVerifier, CommitmentSource, OpeningClaim, OpeningsError, ProverClaim,
    PublicVerifierSetup, SourceRow, ZkOpeningScheme, ZkOpeningSchemeVerifier,
};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};
use rayon::prelude::*;

use ark_bn254::{G1Affine, G1Projective};
use ark_ec::CurveGroup;
use dory::backends::arkworks::ArkG1 as ArkG1Struct;

use crate::transcript::JoltToDoryTranscript;
use crate::types::{DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryVerifierSetup};

// All jolt types below are #[repr(transparent)] over the same arkworks
// inner type as their dory-pcs counterpart, guaranteeing identical layout.

pub(crate) type ArkFr = dory::backends::arkworks::ArkFr;
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
        DoryProverSetup(ArkworksProverSetup::new_from_urs(max_num_vars))
    }

    /// Derives the verifier SRS (a subset of the prover SRS).
    #[tracing::instrument(skip_all, name = "DoryScheme::setup_verifier", fields(max_num_vars))]
    pub fn setup_verifier(max_num_vars: usize) -> DoryVerifierSetup {
        let prover_setup = Self::setup_prover(max_num_vars);
        DoryVerifierSetup(prover_setup.0.to_verifier_setup())
    }

    fn commit_with_mode<S, M>(source: &S, setup: &DoryProverSetup) -> (DoryCommitment, DoryHint)
    where
        S: CommitmentSource<Fr> + ?Sized,
        M: Mode,
    {
        let row_commitments = compute_row_commitments(source, setup);
        let (tier_2, commit_blind) = commit_rows_tier_2::<M>(&row_commitments, setup);

        (
            DoryCommitment(ark_to_jolt_gt(&tier_2)),
            DoryHint::new(
                ark_to_jolt_g1_vec(row_commitments),
                ark_to_jolt_fr(&commit_blind),
            ),
        )
    }

    fn commit_batch_with_mode<B, M>(
        batch: &B,
        ids: &[B::Id],
        setup: &DoryProverSetup,
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
        let row_major = batch.map_rows(sigma, ids, |_, row| commit_source_row(row, &setup.0));

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
        setup: &DoryProverSetup,
        hint: DoryHint,
        transcript: &mut T,
    ) -> (DoryProof, Option<Fr>)
    where
        S: CommitmentSource<Fr> + ?Sized,
        T: DoryTranscript<Curve = InnerBN254>,
        M: Mode,
    {
        let adapter = DorySourceAdapter::new(source);
        let (row_commitments, commit_blind) = hint.into_ark_parts();
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();

        let (proof, y_blinding) =
            dory::prove::<ArkFr, InnerBN254, G1Routines, G2Routines, _, _, M>(
                &adapter,
                &ark_point,
                row_commitments,
                commit_blind,
                nu,
                sigma,
                &setup.0,
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
            source, point, nu, sigma, setup, hint, transcript,
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
            source, point, nu, sigma, setup, hint, transcript,
        );
        let y_com = ark_to_jolt_g1(proof.0.y_com.expect("ZK proof must contain y_com"));
        let blinding = y_blinding.expect("ZK proof must return y_blinding");
        (proof, y_com, blinding)
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
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        let ark_eval = jolt_fr_to_ark(&eval);
        let ark_commitment = jolt_gt_to_ark(&commitment.0);
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        dory::verify::<ArkFr, InnerBN254, G1Routines, G2Routines, _>(
            ark_commitment,
            ark_eval,
            &ark_point,
            &proof.0,
            setup.0.clone().into_inner(),
            &mut dory_transcript,
        )
        .map_err(|_| OpeningsError::VerificationFailed)
    }

    fn verify_batch(
        claims: Vec<OpeningClaim<Self::Field, Self>>,
        proof: &Self::BatchProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Result<(), OpeningsError> {
        homomorphic_verify_batch::<Self, _>(claims, proof, setup, transcript)
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
    type Polynomial = jolt_poly::Polynomial<Fr>;
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
        Self::commit_with_mode::<S, Transparent>(source, setup)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::commit_batch")]
    fn commit_batch<B: BatchCommitmentSource<Self::Field>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Output, Self::OpeningHint)> {
        Self::commit_batch_with_mode::<B, Transparent>(batch, ids, setup)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::open")]
    fn open(
        poly: &Self::Polynomial,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        hint: Option<Self::OpeningHint>,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::Proof {
        let num_vars = point.len();
        let sigma = num_vars.div_ceil(2);
        let nu = num_vars - sigma;

        let hint = match hint {
            Some(hint) => hint,
            None => DoryHint::new(
                ark_to_jolt_g1_vec(compute_row_commitments(poly, setup)),
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

    fn prove_batch(
        claims: Vec<ProverClaim<Self::Field, Self::Polynomial>>,
        hints: Vec<Self::OpeningHint>,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> Self::BatchProof {
        homomorphic_prove_batch::<Self, _>(claims, hints, setup, transcript)
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

        let num_rows = hints[0].row_commitments.len();
        assert!(
            hints.iter().all(|h| h.row_commitments.len() == num_rows),
            "combine_hints: ragged hint lengths",
        );

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
                    acc += hint.row_commitments[row].scalar_mul(&scalar);
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
        let ark_point: Vec<ArkFr> = point.iter().rev().map(jolt_fr_to_ark).collect();
        // In ZK mode dory::verify reads the evaluation commitment from `proof.y_com`,
        // so the caller-side eval is unused here.
        let dummy_eval = <ArkFr as DoryField>::zero();
        let ark_commitment = jolt_gt_to_ark(&commitment.0);
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        dory::verify::<ArkFr, InnerBN254, G1Routines, G2Routines, _>(
            ark_commitment,
            dummy_eval,
            &ark_point,
            &proof.0,
            setup.0.clone().into_inner(),
            &mut dory_transcript,
        )
        .map_err(|_| OpeningsError::VerificationFailed)
    }
}

impl ZkOpeningScheme for DoryScheme {
    type Blind = Fr;

    fn commit_zk<S: CommitmentSource<Fr> + ?Sized>(
        source: &S,
        setup: &Self::ProverSetup,
    ) -> (Self::Output, Self::OpeningHint) {
        Self::commit_with_mode::<S, dory::ZK>(source, setup)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::commit_batch_zk")]
    fn commit_batch_zk<B: BatchCommitmentSource<Self::Field>>(
        batch: &B,
        ids: &[B::Id],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Output, Self::OpeningHint)> {
        Self::commit_batch_with_mode::<B, dory::ZK>(batch, ids, setup)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::open_zk")]
    fn open_zk(
        poly: &Self::Polynomial,
        point: &[Fr],
        _eval: Fr,
        setup: &Self::ProverSetup,
        hint: Self::OpeningHint,
        transcript: &mut impl Transcript<Challenge = Self::Field>,
    ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
        let num_vars = point.len();
        let sigma = num_vars.div_ceil(2);
        let nu = num_vars - sigma;
        let mut dory_transcript = JoltToDoryTranscript::new(transcript);
        Self::open_zk_source_with_shape(poly, point, nu, sigma, setup, hint, &mut dory_transcript)
    }
}

enum DoryChunkCommitment {
    Dense(ArkG1),
    OneHot(Vec<ArkG1>),
}

/// Dense commit: full MSM per row, parallel over rows.
fn commit_rows_dense<S: CommitmentSource<Fr> + ?Sized>(
    source: &S,
    sigma: usize,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let num_cols = 1usize << sigma;
    let g1_bases = &setup.g1_vec[..num_cols];

    let mut rows: Vec<Vec<Fr>> = Vec::new();
    source.for_each_row(sigma, |_, row| {
        rows.push(source_row_to_dense(row, num_cols));
    });

    rows.par_iter()
        .map(|row| {
            let scalars: Vec<ArkFr> = row.iter().map(jolt_fr_to_ark).collect();
            G1Routines::msm(&g1_bases[..scalars.len()], &scalars)
        })
        .collect()
}

fn commit_field_row(values: &[Fr], setup: &ArkworksProverSetup) -> ArkG1 {
    assert!(
        values.len() <= setup.g1_vec.len(),
        "Dory row length ({}) exceeds G1 SRS size ({})",
        values.len(),
        setup.g1_vec.len(),
    );
    let scalars: Vec<ArkFr> = values.iter().map(jolt_fr_to_ark).collect();
    G1Routines::msm(&setup.g1_vec[..scalars.len()], &scalars)
}

fn commit_i128_row(values: &[i128], setup: &ArkworksProverSetup) -> ArkG1 {
    assert!(
        values.len() <= setup.g1_vec.len(),
        "Dory row length ({}) exceeds G1 SRS size ({})",
        values.len(),
        setup.g1_vec.len(),
    );
    let g1_bases = g1_bases_affine(setup, values.len());
    ArkG1Struct(ark_ec::scalar_mul::variable_base::msm_i128::<G1Projective>(
        &g1_bases, values, true,
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

fn commit_one_hot_row(
    row: jolt_openings::OneHotRow<'_>,
    setup: &ArkworksProverSetup,
) -> Vec<ArkG1> {
    let k = 1usize << row.log_domain_size;
    let num_columns = match row.entries {
        jolt_openings::OneHotEntries::OnePerColumn(indices) => indices.len(),
        jolt_openings::OneHotEntries::MaybeZero(indices) => indices.len(),
    };
    assert!(
        num_columns <= setup.g1_vec.len(),
        "Dory one-hot row length ({}) exceeds G1 SRS size ({})",
        num_columns,
        setup.g1_vec.len(),
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

    let g1_bases = g1_bases_affine(setup, num_columns);
    batch_g1_additions_multi_affine(&g1_bases, &columns_by_hot_index)
        .into_iter()
        .map(|affine| ArkG1Struct(affine.into()))
        .collect()
}

fn g1_bases_affine(setup: &ArkworksProverSetup, len: usize) -> Vec<G1Affine> {
    setup.g1_vec[..len]
        .iter()
        .map(|base| base.0.into_affine())
        .collect()
}

fn commit_source_row(row: SourceRow<'_, Fr>, setup: &ArkworksProverSetup) -> DoryChunkCommitment {
    match row {
        SourceRow::FieldElements(values) => {
            DoryChunkCommitment::Dense(commit_field_row(values, setup))
        }
        SourceRow::I128(values) => DoryChunkCommitment::Dense(commit_i128_row(values, setup)),
        SourceRow::OneHot(row) => DoryChunkCommitment::OneHot(commit_one_hot_row(row, setup)),
    }
}

fn aggregate_batch_chunks<M: Mode>(
    chunks: Vec<DoryChunkCommitment>,
    setup: &DoryProverSetup,
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
    setup: &DoryProverSetup,
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
    setup: &DoryProverSetup,
) -> Vec<ArkG1> {
    let num_vars = source.num_vars();
    let sigma = num_vars.div_ceil(2);
    let num_cols = 1usize << sigma;
    let num_rows = 1usize << (num_vars - sigma);

    if source.is_one_hot() {
        commit_rows_one_hot(source, num_rows, num_cols, &setup.0)
    } else {
        commit_rows_dense(source, sigma, &setup.0)
    }
}

fn source_row_to_dense(row: SourceRow<'_, Fr>, expected_len: usize) -> Vec<Fr> {
    match row {
        SourceRow::FieldElements(values) => {
            assert_eq!(values.len(), expected_len);
            values.to_vec()
        }
        SourceRow::I128(values) => {
            assert_eq!(values.len(), expected_len);
            values.iter().map(|&value| Fr::from_i128(value)).collect()
        }
        SourceRow::OneHot(row) => {
            let domain_size = 1usize << row.log_domain_size;
            let mut dense = Vec::new();
            match row.entries {
                jolt_openings::OneHotEntries::OnePerColumn(indices) => {
                    dense.resize(indices.len() * domain_size, Fr::from_u64(0));
                    for (col, hot_index) in indices.iter().enumerate() {
                        dense[hot_index.get() * indices.len() + col] = Fr::from_u64(1);
                    }
                }
                jolt_openings::OneHotEntries::MaybeZero(indices) => {
                    dense.resize(indices.len() * domain_size, Fr::from_u64(0));
                    for (col, hot_index) in indices.iter().enumerate() {
                        if let Some(hot_index) = hot_index {
                            dense[hot_index.get() * indices.len() + col] = Fr::from_u64(1);
                        }
                    }
                }
            }
            assert_eq!(dense.len(), expected_len);
            dense
        }
    }
}

pub(crate) fn commit_rows_tier_2<M: Mode>(
    row_commitments: &[ArkG1],
    setup: &DoryProverSetup,
) -> (ArkGT, ArkFr) {
    let g2_bases = &setup.0.g2_vec[..row_commitments.len()];
    let tier_2 = <InnerBN254 as PairingCurve>::multi_pair_g2_setup(row_commitments, g2_bases);
    let commit_blind = M::sample::<ArkFr>();
    let tier_2 = M::mask(tier_2, &setup.0.ht, &commit_blind);
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

/// Bridges [`CommitmentSource<Fr>`] to dory-pcs's polynomial traits
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
        let native_point: Vec<Fr> = point.iter().map(ark_to_jolt_fr).collect();
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
    use jolt_crypto::{Pedersen, VectorCommitment};
    use jolt_field::{FromPrimitiveInt, RandomSampling};
    use jolt_poly::Polynomial;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

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
