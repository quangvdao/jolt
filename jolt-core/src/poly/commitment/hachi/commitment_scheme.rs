use std::{env::var_os, marker::PhantomData, mem::take, slice::from_raw_parts};

use super::packed_layout::{choose_packed_bit_layout, PackedBitLayout};
use super::packed_poly::{build_packed_poly, JoltPackedPoly};
use super::wrappers::{jolt_to_hachi, ArkBridge, Fp128, JoltToHachiTranscript};
use crate::curve::JoltCurve;
use crate::field::fp128::JoltFp128;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{
    CommitmentScheme, PolynomialBatchSource, StreamingCommitmentScheme, ZkEvalCommitment,
};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::BatchPolynomialSource;
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;
use crate::utils::small_scalar::SmallScalar;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::HACHI_ONEHOT_CHUNK_THRESHOLD_LOG_T;
use hachi_pcs::algebra::ring::CyclotomicRing;
use hachi_pcs::protocol::commitment::{
    compute_num_digits, compute_num_digits_fold, CommitmentConfig, Fp128D64BoundedCommitmentConfig,
    HachiCommitmentCore, HachiCommitmentLayout, HachiScheduleInputs, RingCommitment,
};
use hachi_pcs::protocol::opening_point::BasisMode;
use hachi_pcs::protocol::proof::{HachiCommitmentHint, HachiProof};
use hachi_pcs::protocol::{HachiCommitmentScheme, HachiProverSetup, HachiVerifierSetup};
use hachi_pcs::CommitmentScheme as HachiCommitmentSchemeTrait;
use hachi_pcs::{CanonicalField, DensePoly, FieldCore, FromSmallInt, OneHotIndex, OneHotPoly};
use rayon::prelude::*;

/// Keep the initial Hachi decomposition basis fixed at 3 so the first-round
/// layouts match the setup envelope used elsewhere in the scheme.
const INITIAL_LOG_BASIS: u32 = 3;

/// Mirror Hachi's bounded D=64 profile while keeping Jolt's level-0 basis fixed.
pub type Fp128Bounded64Config<const LOG_COMMIT_BOUND: u32> =
    Fp128D64BoundedCommitmentConfig<LOG_COMMIT_BOUND, INITIAL_LOG_BASIS>;

pub type Fp128OneHot64Config = Fp128Bounded64Config<1>;

fn level0_layout_params<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    log_basis: u32,
) -> (usize, usize) {
    let current_w_len = 1usize.checked_shl(max_num_vars as u32).unwrap_or(0);
    let params = Cfg::level_params_with_log_basis(
        HachiScheduleInputs {
            max_num_vars,
            level: 0,
            current_w_len,
        },
        log_basis,
    );
    (params.n_a, params.challenge_l1_mass)
}

fn optimal_advice_m_r_split<Cfg: CommitmentConfig>(
    reduced_vars: usize,
    log_basis: u32,
) -> (usize, usize) {
    if reduced_vars <= 2 || reduced_vars >= 53 {
        let r = reduced_vars / 2;
        return (reduced_vars - r, r);
    }

    let alpha = Cfg::D.trailing_zeros() as usize;
    let max_num_vars = reduced_vars
        .checked_add(alpha)
        .expect("advice layout variable count overflow");
    let (n_a, challenge_l1_mass) = level0_layout_params::<Cfg>(max_num_vars, log_basis);
    let delta_open = compute_num_digits(128, log_basis) as u64;
    let delta_commit = compute_num_digits(64, log_basis) as u64;
    let c1 = delta_open + n_a as u64 * delta_commit;

    let mut best_r = reduced_vars / 2;
    let mut best_cost = u64::MAX;
    for r in 1..reduced_vars {
        let m = reduced_vars - r;
        let delta_fold = compute_num_digits_fold(r, challenge_l1_mass, log_basis) as u64;
        let cost = c1 * (1u64 << r) + delta_commit * delta_fold * (1u64 << m);
        if cost < best_cost {
            best_cost = cost;
            best_r = r;
        }
    }

    (reduced_vars - best_r, best_r)
}

#[derive(Clone, Default)]
pub struct JoltHachiCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HachiBatchedProof<const D: usize> {
    pub packed_poly_proof: ArkBridge<HachiProof<Fp128>>,
    pub num_packed_polys: u32,
    pub log_k: u32,
    pub individual_proofs: Vec<ArkBridge<HachiProof<Fp128>>>,
}

#[derive(Clone, Debug)]
pub struct JoltHachiBatchHint<const D: usize> {
    hachi_hint: HachiCommitmentHint<Fp128, D>,
    packed_layout: PackedBitLayout,
    num_packed_polys: usize,
    log_k: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JoltHachiOpeningHint<const D: usize> {
    hachi_hint: HachiCommitmentHint<Fp128, D>,
    ring_coeffs: Vec<CyclotomicRing<Fp128, D>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HachiChunkState<const D: usize> {
    Dense(Vec<CyclotomicRing<Fp128, D>>),
    OneHot {
        onehot_k: usize,
        indices: Vec<Option<u8>>,
    },
}

fn to_hachi_opening_point<const D: usize>(point: &[JoltFp128]) -> Vec<Fp128> {
    point.iter().rev().map(jolt_to_hachi).collect()
}

fn to_hachi_packed_opening_point<const D: usize>(
    opening_point: &[JoltFp128],
    rho: &[JoltFp128],
    packed_layout: PackedBitLayout,
) -> Vec<Fp128> {
    let reversed: Vec<Fp128> = opening_point.iter().rev().map(jolt_to_hachi).collect();
    let log_k = packed_layout.log_k();
    assert!(
        log_k <= reversed.len(),
        "packed opening point expects log_k <= num_vars (log_k={log_k}, num_vars={})",
        reversed.len()
    );
    let log_t = reversed.len() - log_k;
    let rho_le: Vec<Fp128> = rho.iter().rev().map(jolt_to_hachi).collect();
    packed_layout.reorder_packed_point(&reversed[..log_t], &reversed[log_t..], &rho_le)
}

fn advice_commit_layout<Cfg: CommitmentConfig>(
    m_vars: usize,
    r_vars: usize,
    log_basis: u32,
) -> HachiCommitmentLayout {
    let alpha = Cfg::D.trailing_zeros() as usize;
    let max_num_vars = m_vars
        .checked_add(r_vars)
        .and_then(|vars| vars.checked_add(alpha))
        .expect("advice layout variable count overflow");
    let (n_a, challenge_l1_mass) = level0_layout_params::<Cfg>(max_num_vars, log_basis);
    HachiCommitmentLayout::new_with_decomp(
        m_vars,
        r_vars,
        n_a,
        compute_num_digits(64, log_basis),
        compute_num_digits(128, log_basis),
        compute_num_digits_fold(r_vars, challenge_l1_mass, log_basis),
        log_basis,
    )
    .unwrap()
}

/// Compute the advice commit layout using the polynomial's own optimal m/r split
/// rather than inheriting from the setup envelope.
fn compute_advice_layout<const D: usize, Cfg: CommitmentConfig>(
    poly_num_vars: usize,
) -> HachiCommitmentLayout {
    let alpha = D.trailing_zeros() as usize;
    let reduced_vars = poly_num_vars.saturating_sub(alpha);
    if reduced_vars <= 1 {
        return advice_commit_layout::<Cfg>(reduced_vars.max(1), 0, INITIAL_LOG_BASIS);
    }
    // Advice polynomials have log_commit_bound=64 even when the main one-hot
    // witness uses a narrower commit bound, so choose the split from that cost model.
    let (m_vars, r_vars) = optimal_advice_m_r_split::<Cfg>(reduced_vars, INITIAL_LOG_BASIS);
    advice_commit_layout::<Cfg>(m_vars, r_vars, INITIAL_LOG_BASIS)
}

fn choose_packed_layout_for_shape<const D: usize, Cfg: CommitmentConfig>(
    log_k: usize,
    log_t: usize,
    log_packed: usize,
) -> (PackedBitLayout, HachiCommitmentLayout) {
    let packed_layout = choose_packed_bit_layout::<D, Cfg>(log_k, log_t, log_packed);
    let hachi_layout = packed_layout.into_hachi_layout::<Cfg>(INITIAL_LOG_BASIS);
    (packed_layout, hachi_layout)
}

fn choose_packed_layout_for_dims<const D: usize, Cfg: CommitmentConfig>(
    num_cycles: usize,
    num_polys: usize,
    onehot_k: usize,
) -> (PackedBitLayout, HachiCommitmentLayout) {
    assert!(
        num_cycles.is_power_of_two(),
        "packed Hachi layout expects num_cycles to be a power of two (got {num_cycles})"
    );
    assert!(
        onehot_k.is_power_of_two(),
        "packed Hachi layout expects onehot_k to be a power of two (got {onehot_k})"
    );
    let log_k = onehot_k.trailing_zeros() as usize;
    let log_t = num_cycles.trailing_zeros() as usize;
    let log_packed = num_polys.next_power_of_two().trailing_zeros() as usize;
    choose_packed_layout_for_shape::<D, Cfg>(log_k, log_t, log_packed)
}

fn compute_packed_setup_layouts<const D: usize, Cfg>(
    max_log_t: usize,
    max_log_k: usize,
    log_packed: usize,
) -> Vec<HachiCommitmentLayout>
where
    Cfg: CommitmentConfig + Default,
{
    let advice_num_vars = max_log_k + max_log_t;
    let advice_layout = compute_advice_layout::<D, Cfg>(advice_num_vars);
    let mut setup_layouts = vec![advice_layout];
    let packed_log_ks = JoltHachiCommitmentScheme::<D, Cfg>::supported_log_k_chunks(max_log_k);
    if var_os("HACHI_SETUP_DIAGNOSTICS").is_some() {
        eprintln!(
            "[jolt hachi setup] max_log_t={max_log_t}, max_log_k={max_log_k}, log_packed={log_packed}"
        );
        eprintln!("  advice_layout={advice_layout:?}");
    }
    for log_k in packed_log_ks {
        let (_, packed_layout) =
            choose_packed_layout_for_shape::<D, Cfg>(log_k, max_log_t, log_packed);
        if var_os("HACHI_SETUP_DIAGNOSTICS").is_some() {
            eprintln!("  packed_layout(log_k={log_k})={packed_layout:?}");
        }
        setup_layouts.push(packed_layout);
    }
    setup_layouts
}

fn hachi_commit_dense<const D: usize, Cfg: CommitmentConfig>(
    ring_coeffs: Vec<CyclotomicRing<Fp128, D>>,
    setup: &HachiProverSetup<Fp128, D>,
    layout: &HachiCommitmentLayout,
) -> (RingCommitment<Fp128, D>, JoltHachiOpeningHint<D>) {
    let mut dense_poly = DensePoly::from_ring_coeffs(ring_coeffs);
    let (commitment, hachi_hint) = <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<
        Fp128,
        D,
    >>::commit(&dense_poly, setup, layout)
    .expect("Hachi commit failed");
    let ring_coeffs = take(&mut dense_poly.coeffs);
    (
        commitment,
        JoltHachiOpeningHint {
            hachi_hint,
            ring_coeffs,
        },
    )
}

fn hachi_commit_onehot<const D: usize, Cfg: CommitmentConfig, I: OneHotIndex>(
    onehot_k: usize,
    indices: Vec<Option<I>>,
    setup: &HachiProverSetup<Fp128, D>,
    layout: &HachiCommitmentLayout,
) -> (RingCommitment<Fp128, D>, JoltHachiOpeningHint<D>) {
    let onehot_poly =
        OneHotPoly::<Fp128, D, I>::new(onehot_k, indices, layout.r_vars, layout.m_vars)
            .expect("OneHotPoly construction failed");
    let (commitment, hachi_hint) = <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<
        Fp128,
        D,
    >>::commit(&onehot_poly, setup, layout)
    .expect("Hachi commit_onehot failed");
    (
        commitment,
        JoltHachiOpeningHint {
            hachi_hint,
            ring_coeffs: vec![],
        },
    )
}

#[tracing::instrument(skip_all, name = "fused_build_and_commit")]
fn fused_build_and_commit<const D: usize, Cfg, F, B>(
    index_fn: F,
    batch_fn: B,
    num_cycles: usize,
    num_polys: usize,
    packed_layout: PackedBitLayout,
    batch_layout: &HachiCommitmentLayout,
    setup: &HachiProverSetup<Fp128, D>,
) -> (RingCommitment<Fp128, D>, HachiCommitmentHint<Fp128, D>)
where
    Cfg: CommitmentConfig,
    F: Fn(usize, usize) -> Option<u8> + Clone + Send + Sync,
    B: Fn(usize, usize, &mut [Option<u8>]) + Clone + Send + Sync,
{
    let packed_poly = JoltPackedPoly {
        packed_layout,
        index_fn,
        batch_fn,
        num_cycles,
        num_polys,
    };

    <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::commit(
        &packed_poly,
        setup,
        batch_layout,
    )
    .expect("Hachi packed poly commit failed")
}

impl<const D: usize, Cfg> CommitmentScheme for JoltHachiCommitmentScheme<D, Cfg>
where
    Cfg: CommitmentConfig + Default,
{
    type Field = JoltFp128;
    type Config = ();
    type ProverSetup = ArkBridge<HachiProverSetup<Fp128, D>>;
    type VerifierSetup = ArkBridge<HachiVerifierSetup<Fp128>>;
    type Commitment = ArkBridge<RingCommitment<Fp128, D>>;
    type Proof = ArkBridge<HachiProof<Fp128>>;
    type BatchedProof = HachiBatchedProof<D>;
    type OpeningProofHint = JoltHachiOpeningHint<D>;
    type BatchOpeningHint = JoltHachiBatchHint<D>;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        ArkBridge(
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::setup_prover(
                max_num_vars,
            ),
        )
    }

    fn setup_prover_from_shape(
        max_log_t: usize,
        max_log_k: usize,
        log_packed: Option<usize>,
    ) -> Self::ProverSetup {
        let setup_layouts =
            compute_packed_setup_layouts::<D, Cfg>(max_log_t, max_log_k, log_packed.unwrap_or(0));
        let (setup, _) = HachiCommitmentCore::setup_with_layouts::<Fp128, D, Cfg>(&setup_layouts)
            .expect("Hachi packed setup failed");
        ArkBridge(setup)
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        ArkBridge(
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::setup_verifier(
                &setup.0,
            ),
        )
    }

    fn from_proof(_proof: &Self::BatchedProof) -> Self {
        Self::default()
    }

    fn config(&self) -> &() {
        &()
    }

    fn commit(
        &self,
        poly: &MultilinearPolynomial<JoltFp128>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let poly_num_vars = poly.len().trailing_zeros() as usize;
        let layout = compute_advice_layout::<D, Cfg>(poly_num_vars);

        let (commitment, hint) = if let MultilinearPolynomial::OneHot(onehot) = poly {
            let indices: Vec<Option<u8>> = onehot.nonzero_indices.as_ref().clone();
            hachi_commit_onehot::<D, Cfg, u8>(onehot.K, indices, &setup.0, &layout)
        } else {
            let ring_coeffs = poly_to_ring_coeffs::<D>(poly);
            hachi_commit_dense::<D, Cfg>(ring_coeffs, &setup.0, &layout)
        };
        (ArkBridge(commitment), hint)
    }

    fn batch_commit<S: PolynomialBatchSource<JoltFp128>>(
        &self,
        source: &S,
        setup: &Self::ProverSetup,
    ) -> (Vec<Self::Commitment>, Self::BatchOpeningHint) {
        assert!(source.num_polys() > 0);
        let num_cycles = source
            .num_cycles()
            .expect("batch_commit requires lazy source");
        let onehot_k = source.onehot_k().unwrap();
        let num_polys = source.num_polys();
        let (packed_layout, batch_layout) =
            choose_packed_layout_for_dims::<D, Cfg>(num_cycles, num_polys, onehot_k);

        let index_fn = |c: usize, p: usize| source.onehot_index(c, p);
        let batch_fn = |c: usize, p_start: usize, buf: &mut [Option<u8>]| {
            source.batch_onehot_indices(c, p_start, buf)
        };
        let (commitment, hachi_hint) = fused_build_and_commit::<D, Cfg, _, _>(
            index_fn,
            batch_fn,
            num_cycles,
            num_polys,
            packed_layout,
            &batch_layout,
            &setup.0,
        );

        let hint = JoltHachiBatchHint {
            hachi_hint,
            packed_layout,
            num_packed_polys: num_polys,
            log_k: packed_layout.log_k(),
        };
        (vec![ArkBridge(commitment)], hint)
    }

    fn prove<ProofTranscript: Transcript>(
        &self,
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<JoltFp128>,
        opening_point: &[JoltFp128],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
        commitment: &Self::Commitment,
    ) -> (Self::Proof, Option<Self::Field>) {
        let hint = hint.expect("prove() requires a hint");
        let hachi_point = to_hachi_opening_point::<D>(opening_point);
        let mut adapter = JoltToHachiTranscript::new(transcript);

        let layout = compute_advice_layout::<D, Cfg>(opening_point.len());

        let proof = if let MultilinearPolynomial::OneHot(onehot) = poly {
            let indices: Vec<Option<u8>> = onehot.nonzero_indices.as_ref().clone();
            let onehot_poly =
                OneHotPoly::<Fp128, D, u8>::new(onehot.K, indices, layout.r_vars, layout.m_vars)
                    .expect("OneHotPoly construction failed");
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::prove(
                &setup.0,
                &onehot_poly,
                &hachi_point,
                hint.hachi_hint,
                &mut adapter,
                &commitment.0,
                BasisMode::Lagrange,
                &layout,
            )
        } else {
            let dense_poly = if hint.ring_coeffs.is_empty() {
                let ring_coeffs = poly_to_ring_coeffs::<D>(poly);
                DensePoly::from_ring_coeffs(ring_coeffs)
            } else {
                DensePoly::from_ring_coeffs(hint.ring_coeffs)
            };
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::prove(
                &setup.0,
                &dense_poly,
                &hachi_point,
                hint.hachi_hint,
                &mut adapter,
                &commitment.0,
                BasisMode::Lagrange,
                &layout,
            )
        }
        .expect("Hachi prove failed");
        (ArkBridge(proof), None)
    }

    fn verify<ProofTranscript: Transcript>(
        &self,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[JoltFp128],
        opening: &JoltFp128,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let hachi_point = to_hachi_opening_point::<D>(opening_point);
        let hachi_opening = jolt_to_hachi(opening);
        let mut adapter = JoltToHachiTranscript::new(transcript);

        let layout = compute_advice_layout::<D, Cfg>(opening_point.len());

        <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::verify(
            &proof.0,
            &setup.0,
            &mut adapter,
            &hachi_point,
            &hachi_opening,
            &commitment.0,
            BasisMode::Lagrange,
            &layout,
        )
        .map_err(|_| ProofVerifyError::InternalError)
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_prove<ProofTranscript: Transcript, PolySource: BatchPolynomialSource<JoltFp128>>(
        &self,
        setup: &Self::ProverSetup,
        poly_source: &PolySource,
        batch_hint: Self::BatchOpeningHint,
        individual_hints: Vec<Self::OpeningProofHint>,
        commitments: &[&Self::Commitment],
        opening_point: &[JoltFp128],
        claims: &[JoltFp128],
        _coeffs: &[JoltFp128],
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        let num_individual = individual_hints.len();
        let num_packed = batch_hint.num_packed_polys;
        assert!(
            num_packed > 0,
            "batch_prove requires at least one packed claim"
        );
        assert_eq!(
            claims.len(),
            num_packed + num_individual,
            "batch_prove claims must be [packed_claims, individual_claims]"
        );
        assert_eq!(
            commitments.len(),
            1 + num_individual,
            "batch_prove commitments must be [packed_commitment, individual_commitments...]"
        );

        let packed_claims = &claims[..num_packed];
        let num_padded = num_packed.next_power_of_two();
        let r_vars = (num_padded as u32).trailing_zeros() as usize;

        transcript.append_bytes(b"hachi_packed_num", &(num_packed as u64).to_le_bytes());
        let rho: Vec<JoltFp128> = transcript.challenge_vector(r_vars);

        let eq_table = EqPolynomial::<JoltFp128>::evals(&rho);
        let combined_claim: JoltFp128 = packed_claims
            .iter()
            .zip(eq_table.iter())
            .map(|(&v, &eq)| v * eq)
            .sum();

        let log_k = batch_hint.log_k;
        assert!(
            log_k <= opening_point.len(),
            "batch_prove log_k exceeds opening point length (log_k={log_k}, point_len={})",
            opening_point.len()
        );
        let packed_layout = batch_hint.packed_layout;
        let packed_hachi_layout = packed_layout.into_hachi_layout::<Cfg>(INITIAL_LOG_BASIS);
        let packed_point = to_hachi_packed_opening_point::<D>(opening_point, &rho, packed_layout);

        let hachi_combined = jolt_to_hachi(&combined_claim);
        let packed_commitment = commitments[0];

        transcript.append_bytes(
            b"hachi_packed_claim",
            &hachi_combined.to_canonical_u128().to_le_bytes(),
        );

        let num_cycles = poly_source
            .num_cycles()
            .expect("batch_prove requires lazy source");
        let num_polys = poly_source.num_polys().unwrap();
        assert_eq!(
            num_polys, num_packed,
            "batch_prove packed poly count mismatch (commit={num_packed}, prove={num_polys})"
        );

        let index_fn = |c: usize, p: usize| poly_source.onehot_index(c, p);
        let batch_fn = |c: usize, p_start: usize, buf: &mut [Option<u8>]| {
            poly_source.batch_onehot_indices(c, p_start, buf)
        };
        let packed_poly =
            build_packed_poly(index_fn, batch_fn, num_cycles, num_polys, packed_layout);

        let mut adapter = JoltToHachiTranscript::new(transcript);

        let packed_proof =
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::prove(
                &setup.0,
                &packed_poly,
                &packed_point,
                batch_hint.hachi_hint,
                &mut adapter,
                &packed_commitment.0,
                BasisMode::Lagrange,
                &packed_hachi_layout,
            )
            .expect("Hachi packed poly prove failed");

        let hachi_point = to_hachi_opening_point::<D>(opening_point);
        let indiv_layout = compute_advice_layout::<D, Cfg>(opening_point.len());

        let individual_commitments = &commitments[1..];
        let individual_proofs: Vec<ArkBridge<HachiProof<Fp128>>> = individual_hints
            .into_iter()
            .zip(individual_commitments.iter())
            .enumerate()
            .map(|(i, (hint, commitment))| {
                transcript.append_bytes(b"hachi_individual_item", &(i as u64).to_le_bytes());
                let individual_poly = DensePoly::from_ring_coeffs(hint.ring_coeffs);
                let mut adapter = JoltToHachiTranscript::new(transcript);
                let proof =
                    <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::prove(
                        &setup.0,
                        &individual_poly,
                        &hachi_point,
                        hint.hachi_hint,
                        &mut adapter,
                        &commitment.0,
                        BasisMode::Lagrange,
                        &indiv_layout,
                    )
                    .expect("Hachi individual prove failed");
                ArkBridge(proof)
            })
            .collect();

        HachiBatchedProof {
            packed_poly_proof: ArkBridge(packed_proof),
            num_packed_polys: num_packed as u32,
            log_k: log_k as u32,
            individual_proofs,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_verify<ProofTranscript: Transcript>(
        &self,
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[JoltFp128],
        commitments: &[&Self::Commitment],
        claims: &[JoltFp128],
        _coeffs: &[JoltFp128],
    ) -> Result<(), ProofVerifyError> {
        let num_packed = proof.num_packed_polys as usize;
        if num_packed == 0 {
            return Err(ProofVerifyError::InvalidInputLength(1, num_packed));
        }
        if num_packed > claims.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                num_packed,
                claims.len(),
            ));
        }

        let packed_claims = &claims[..num_packed];
        let individual_claims = &claims[num_packed..];
        if commitments.len() != 1 + individual_claims.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                1 + individual_claims.len(),
                commitments.len(),
            ));
        }
        if proof.individual_proofs.len() != individual_claims.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                individual_claims.len(),
                proof.individual_proofs.len(),
            ));
        }

        let num_padded = num_packed.next_power_of_two();
        let selector_vars = (num_padded as u32).trailing_zeros() as usize;

        transcript.append_bytes(b"hachi_packed_num", &(num_packed as u64).to_le_bytes());
        let rho: Vec<JoltFp128> = transcript.challenge_vector(selector_vars);

        let eq_table = EqPolynomial::<JoltFp128>::evals(&rho);
        let combined_claim: JoltFp128 = packed_claims
            .iter()
            .zip(eq_table.iter())
            .map(|(&v, &eq)| v * eq)
            .sum();

        let log_k = proof.log_k as usize;
        if log_k > opening_point.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                opening_point.len(),
                log_k,
            ));
        }
        let log_t = opening_point.len() - log_k;
        let (packed_layout, packed_hachi_layout) =
            choose_packed_layout_for_shape::<D, Cfg>(log_k, log_t, selector_vars);
        let packed_point = to_hachi_packed_opening_point::<D>(opening_point, &rho, packed_layout);

        let hachi_combined = jolt_to_hachi(&combined_claim);
        let packed_commitment = commitments[0];

        transcript.append_bytes(
            b"hachi_packed_claim",
            &hachi_combined.to_canonical_u128().to_le_bytes(),
        );
        let mut adapter = JoltToHachiTranscript::new(transcript);
        <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::verify(
            &proof.packed_poly_proof.0,
            &setup.0,
            &mut adapter,
            &packed_point,
            &hachi_combined,
            &packed_commitment.0,
            BasisMode::Lagrange,
            &packed_hachi_layout,
        )
        .map_err(|_| ProofVerifyError::InternalError)?;

        let hachi_point = to_hachi_opening_point::<D>(opening_point);
        let indiv_layout = compute_advice_layout::<D, Cfg>(opening_point.len());

        let individual_commitments = &commitments[1..];
        for i in 0..individual_claims.len() {
            transcript.append_bytes(b"hachi_individual_item", &(i as u64).to_le_bytes());
            let hachi_claim = jolt_to_hachi(&individual_claims[i]);
            let mut adapter = JoltToHachiTranscript::new(transcript);
            <HachiCommitmentScheme<D, Cfg> as HachiCommitmentSchemeTrait<Fp128, D>>::verify(
                &proof.individual_proofs[i].0,
                &setup.0,
                &mut adapter,
                &hachi_point,
                &hachi_claim,
                &individual_commitments[i].0,
                BasisMode::Lagrange,
                &indiv_layout,
            )
            .map_err(|_| ProofVerifyError::InternalError)?;
        }

        Ok(())
    }

    fn split_batch_hint(_batch_hint: &Self::BatchOpeningHint) -> Vec<Self::OpeningProofHint> {
        vec![]
    }

    fn protocol_name() -> &'static [u8] {
        b"Hachi"
    }

    fn packed_main_commitment_arity() -> Option<usize> {
        Some(1)
    }

    fn uses_onehot_inc() -> bool {
        true
    }

    fn log_k_chunk_for_trace(log_T: usize) -> usize {
        if log_T >= HACHI_ONEHOT_CHUNK_THRESHOLD_LOG_T {
            8
        } else {
            4
        }
    }

    fn supported_log_k_chunks(max_log_k: usize) -> Vec<usize> {
        if max_log_k > 4 {
            vec![4, max_log_k]
        } else {
            vec![max_log_k]
        }
    }

    fn validate_batch_proof_shape(
        proof: &Self::BatchedProof,
        one_hot_log_k_chunk: usize,
    ) -> Result<(), ProofVerifyError> {
        let proof_log_k = proof.log_k as usize;
        if proof_log_k != one_hot_log_k_chunk {
            return Err(ProofVerifyError::InvalidInputLength(
                one_hot_log_k_chunk,
                proof_log_k,
            ));
        }
        Ok(())
    }
}

impl<const D: usize, Cfg> StreamingCommitmentScheme for JoltHachiCommitmentScheme<D, Cfg>
where
    Cfg: CommitmentConfig + Default,
{
    type ChunkState = HachiChunkState<D>;

    #[allow(non_snake_case)]
    fn streaming_chunk_size(&self, _K: usize, _T: usize) -> Option<usize> {
        None
    }

    fn process_chunk<T: SmallScalar>(
        &self,
        _setup: &Self::ProverSetup,
        _chunk: &[T],
    ) -> Self::ChunkState {
        unreachable!("Hachi uses batch_commit via PolynomialBatchSource, not streaming")
    }

    fn process_chunk_onehot(
        &self,
        _setup: &Self::ProverSetup,
        _onehot_k: usize,
        _chunk: &[Option<usize>],
    ) -> Self::ChunkState {
        unreachable!("Hachi uses batch_commit via PolynomialBatchSource, not streaming")
    }

    fn aggregate_chunks(
        &self,
        _setup: &Self::ProverSetup,
        _onehot_k: Option<usize>,
        _tier1_commitments: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        unreachable!("Hachi uses batch_commit via PolynomialBatchSource, not streaming")
    }

    fn aggregate_streaming_batch(
        &self,
        _setup: &Self::ProverSetup,
        _onehot_ks: &[Option<usize>],
        _tier1_per_poly: &[Vec<Self::ChunkState>],
    ) -> Option<(Vec<Self::Commitment>, Self::BatchOpeningHint)> {
        None
    }

    fn streaming_batch_hint(_hints: Vec<Self::OpeningProofHint>) -> Self::BatchOpeningHint {
        panic!("Hachi uses batch_commit via PolynomialBatchSource, not streaming")
    }
}

pub(super) fn poly_to_ring_coeffs<const D: usize>(
    poly: &MultilinearPolynomial<JoltFp128>,
) -> Vec<CyclotomicRing<Fp128, D>> {
    match poly {
        MultilinearPolynomial::LargeScalars(p) => {
            // SAFETY: JoltFp128 is repr(transparent) over Fp128.
            let field_coeffs: &[Fp128] =
                unsafe { from_raw_parts(p.Z.as_ptr() as *const Fp128, p.Z.len()) };
            pack_field_to_ring::<D>(field_coeffs)
        }
        MultilinearPolynomial::BoolScalars(p) => {
            pack_scalars::<D, _, _>(&p.coeffs, |&b| if b { Fp128::one() } else { Fp128::zero() })
        }
        MultilinearPolynomial::U8Scalars(p) => {
            pack_scalars::<D, _, _>(&p.coeffs, |&v| Fp128::from_u64(v as u64))
        }
        MultilinearPolynomial::U16Scalars(p) => {
            pack_scalars::<D, _, _>(&p.coeffs, |&v| Fp128::from_u64(v as u64))
        }
        MultilinearPolynomial::U32Scalars(p) => {
            pack_scalars::<D, _, _>(&p.coeffs, |&v| Fp128::from_u64(v as u64))
        }
        MultilinearPolynomial::U64Scalars(p) => {
            pack_scalars::<D, _, _>(&p.coeffs, |&v| Fp128::from_u64(v))
        }
        MultilinearPolynomial::U128Scalars(p) => pack_scalars::<D, _, _>(&p.coeffs, |&v| {
            <Fp128 as CanonicalField>::from_canonical_u128_reduced(v)
        }),
        MultilinearPolynomial::I64Scalars(p) => pack_scalars::<D, _, _>(&p.coeffs, |&v| {
            if v >= 0 {
                Fp128::from_u64(v as u64)
            } else {
                -Fp128::from_u64(v.unsigned_abs())
            }
        }),
        MultilinearPolynomial::I128Scalars(p) => pack_scalars::<D, _, _>(&p.coeffs, |&v| {
            let jolt = JoltFp128::from_i128(v);
            jolt_to_hachi(&jolt)
        }),
        MultilinearPolynomial::S128Scalars(p) => pack_scalars::<D, _, _>(&p.coeffs, |v| {
            if let Some(i) = v.to_i128() {
                let jolt = JoltFp128::from_i128(i);
                jolt_to_hachi(&jolt)
            } else {
                let mag = v.magnitude_as_u128();
                let f = <Fp128 as CanonicalField>::from_canonical_u128_reduced(mag);
                if v.is_positive {
                    f
                } else {
                    -f
                }
            }
        }),
        MultilinearPolynomial::OneHot(_) | MultilinearPolynomial::RLC(_) => {
            panic!("OneHot and RLC polynomials cannot be materialized for Hachi commit")
        }
    }
}

fn pack_scalars<const D: usize, T: Sync, F: Fn(&T) -> Fp128 + Sync + Send>(
    scalars: &[T],
    convert: F,
) -> Vec<CyclotomicRing<Fp128, D>> {
    let par_grain = D * 256;
    scalars
        .par_chunks(par_grain)
        .flat_map_iter(|big_chunk| {
            big_chunk.chunks(D).map(|chunk| {
                let mut coeffs = [Fp128::zero(); D];
                for (i, scalar) in chunk.iter().enumerate() {
                    coeffs[i] = convert(scalar);
                }
                CyclotomicRing::from_coefficients(coeffs)
            })
        })
        .collect()
}

fn pack_field_to_ring<const D: usize>(field_coeffs: &[Fp128]) -> Vec<CyclotomicRing<Fp128, D>> {
    field_coeffs
        .par_chunks(D)
        .map(|chunk| {
            let mut coeffs = [Fp128::zero(); D];
            coeffs[..chunk.len()].copy_from_slice(chunk);
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect()
}

impl<const D: usize, Cfg, C> ZkEvalCommitment<C> for JoltHachiCommitmentScheme<D, Cfg>
where
    Cfg: CommitmentConfig + Default,
    C: JoltCurve<F = <Self as CommitmentScheme>::Field>,
{
    fn eval_commitment(_proof: &Self::BatchedProof) -> Option<C::G1> {
        None
    }
    fn eval_commitment_gens(_setup: &Self::ProverSetup) -> Option<(C::G1, C::G1)> {
        None
    }
    fn eval_commitment_gens_verifier(_setup: &Self::VerifierSetup) -> Option<(C::G1, C::G1)> {
        None
    }
}
