//! This file implements the Hyrax polynomial commitment scheme used in snark composition
use super::commitment_scheme::CommitmentScheme;
use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::transcripts::{AppendToTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::{compute_dotproduct, mul_0_1_optimized};
use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};
use ark_ec::CurveGroup;
use ark_grumpkin;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::vec::Vec;
use ark_std::Zero;
use jolt_platform::{end_cycle_tracking, start_cycle_tracking};
use num_integer::Roots;
use rand::SeedableRng;
use rayon::prelude::*;
use std::borrow::Borrow;
use tracing::trace_span;

// Cycle-marker labels must be static strings: the tracer keys markers by the guest string pointer.
const CYCLE_HYRAX_OPENING_VERIFY: &str = "jolt_hyrax_opening_verify";
const CYCLE_HYRAX_MSM1: &str = "jolt_hyrax_msm1";
const CYCLE_HYRAX_MSM2: &str = "jolt_hyrax_msm2";

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

/// Pedersen generators for commitment scheme
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct PedersenGenerators<G: CurveGroup> {
    pub(crate) generators: Vec<G::Affine>,
}

impl<G: CurveGroup> GuestSerialize for PedersenGenerators<G>
where
    G::Affine: GuestSerialize,
{
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.generators.guest_serialize(w)
    }
}

impl<G: CurveGroup> GuestDeserialize for PedersenGenerators<G>
where
    G::Affine: GuestDeserialize,
{
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            generators: Vec::<G::Affine>::guest_deserialize(r)?,
        })
    }
}

impl<G: CurveGroup> PedersenGenerators<G> {
    pub fn new(size: usize, label: &[u8]) -> Self {
        // Deterministic generation using label as seed
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(
            label
                .iter()
                .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64)),
        );

        Self {
            generators: (0..size).map(|_| G::rand(&mut rng).into_affine()).collect(),
        }
    }
}

/// Hyrax commits to a multilinear polynomial by interpreting its coefficients as a
/// matrix. Given the number of variables in the polynomial, and the desired "aspect
/// ratio", returns the column and row size of that matrix.
pub fn matrix_dimensions(num_vars: usize, matrix_aspect_ratio: usize) -> (usize, usize) {
    if num_vars == 0 {
        panic!("Hyrax matrix_dimensions called with num_vars = 0!");
    }

    let mut row_size = (num_vars / 2).pow2();
    row_size = (row_size * matrix_aspect_ratio.sqrt()).next_power_of_two();

    let right_num_vars = std::cmp::min(row_size.log_2(), num_vars - 1);
    row_size = right_num_vars.pow2();
    let left_num_vars = num_vars - right_num_vars;
    let col_size = left_num_vars.pow2();

    (col_size, row_size)
}

#[derive(Default, Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyraxCommitment<const RATIO: usize, G: CurveGroup> {
    pub row_commitments: Vec<G>,
}

impl<const RATIO: usize, G: CurveGroup> GuestSerialize for HyraxCommitment<RATIO, G>
where
    G: GuestSerialize,
{
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.row_commitments.guest_serialize(w)
    }
}

impl<const RATIO: usize, G: CurveGroup> GuestDeserialize for HyraxCommitment<RATIO, G>
where
    G: GuestDeserialize,
{
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            row_commitments: Vec::<G>::guest_deserialize(r)?,
        })
    }
}

impl<const RATIO: usize, G: CurveGroup> HyraxCommitment<RATIO, G>
where
    G::ScalarField: JoltField,
{
    #[tracing::instrument(skip_all, name = "HyraxCommitment::commit")]
    pub fn commit(
        poly: &DensePolynomial<G::ScalarField>,
        generators: &PedersenGenerators<G>,
    ) -> Self {
        Self::commit_with_unpadded_len(poly, generators, poly.len())
    }

    /// Commit to a dense polynomial while optionally skipping work on a known-zero padded suffix.
    ///
    /// `unpadded_len` must satisfy:
    /// - `unpadded_len <= poly.len()`
    /// - `poly[unpadded_len..]` is all zeros (debug-asserted)
    ///
    /// This is particularly useful when the caller constructs `poly` by packing real data
    /// and then padding with zeros to the next power of two (as in recursion prefix packing).
    #[tracing::instrument(skip_all, name = "HyraxCommitment::commit_with_unpadded_len")]
    pub fn commit_with_unpadded_len(
        poly: &DensePolynomial<G::ScalarField>,
        generators: &PedersenGenerators<G>,
        unpadded_len: usize,
    ) -> Self {
        let n = poly.len();
        assert!(
            unpadded_len <= n,
            "Hyrax commit_with_unpadded_len: unpadded_len={unpadded_len} > poly.len()={n}"
        );

        debug_assert!(
            poly.Z[unpadded_len..]
                .iter()
                .all(|x| *x == G::ScalarField::zero()),
            "Hyrax commit_with_unpadded_len: expected poly[unpadded_len..] to be all-zero padding"
        );

        let ell = n.log_2();
        let (L_size, R_size) = matrix_dimensions(ell, 1);
        assert_eq!(L_size * R_size, n);

        let full_rows = unpadded_len / R_size;
        let rem = unpadded_len % R_size;

        let gens = &generators.generators[..R_size];

        // This commitment method is used on the prover side (host). Keep it simple and
        // parallelize over rows while running a sequential MSM per row to avoid nested
        // rayon oversubscription in upstream callsites.
        let mut row_commitments: Vec<G> = poly
            .Z
            .par_chunks(R_size)
            .take(full_rows)
            .map(|row| {
                VariableBaseMSM::msm_sequential(gens, row)
                    .expect("row length should match generator length")
            })
            .collect();

        // Handle a partial last row (if any) by truncating both the row and generators;
        // this is equivalent to padding the row with zeros.
        if rem != 0 {
            let row_start = full_rows * R_size;
            let row = &poly.Z[row_start..row_start + rem];
            let row_commitment = VariableBaseMSM::msm_sequential(&generators.generators[..rem], row)
                .expect("partial row length should match generator length");
            row_commitments.push(row_commitment);
        }

        // The remaining rows are all-zero padding; their commitments are the identity.
        row_commitments.resize(L_size, G::zero());

        Self { row_commitments }
    }
}

impl<const RATIO: usize, G: CurveGroup> AppendToTranscript for HyraxCommitment<RATIO, G> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        for i in 0..self.row_commitments.len() {
            transcript.append_point(&self.row_commitments[i]);
        }
    }
}

/// A Hyrax opening proof for a single polynomial opened at a single point.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyraxOpeningProof<const RATIO: usize, G: CurveGroup> {
    pub vector_matrix_product: Vec<G::ScalarField>,
}

impl<const RATIO: usize, G: CurveGroup> GuestSerialize for HyraxOpeningProof<RATIO, G>
where
    G::ScalarField: GuestSerialize,
{
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.vector_matrix_product.guest_serialize(w)
    }
}

impl<const RATIO: usize, G: CurveGroup> GuestDeserialize for HyraxOpeningProof<RATIO, G>
where
    G::ScalarField: GuestDeserialize,
{
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            vector_matrix_product: Vec::<G::ScalarField>::guest_deserialize(r)?,
        })
    }
}

/// See Section 14.3 of Thaler's Proofs, Arguments, and Zero-Knowledge
impl<const RATIO: usize, F: JoltField, G: CurveGroup<ScalarField = F>> HyraxOpeningProof<RATIO, G> {
    #[tracing::instrument(skip_all, name = "HyraxOpeningProof::prove")]
    pub fn prove(
        poly: &DensePolynomial<G::ScalarField>,
        opening_point: &[G::ScalarField], // point at which the polynomial is evaluated
        ratio: usize,
    ) -> HyraxOpeningProof<RATIO, G> {
        assert_eq!(poly.get_num_vars(), opening_point.len());

        // compute the L and R vectors
        let (L_size, _R_size) = matrix_dimensions(poly.get_num_vars(), ratio);
        let L = EqPolynomial::evals(&opening_point[..L_size.log_2()]);

        // compute vector-matrix product between L and Z viewed as a matrix
        let vector_matrix_product = Self::vector_matrix_product(poly, &L, ratio);

        HyraxOpeningProof {
            vector_matrix_product,
        }
    }

    pub fn verify(
        &self,
        pedersen_generators: &PedersenGenerators<G>,
        opening_point: &[G::ScalarField], // point at which the polynomial is evaluated
        opening: &G::ScalarField,         // evaluation \widetilde{Z}(r)
        commitment: &HyraxCommitment<RATIO, G>,
    ) -> Result<(), ProofVerifyError> {
        let _cycle = CycleMarkerGuard::new(CYCLE_HYRAX_OPENING_VERIFY);
        // compute L and R
        let (L_size, R_size) = matrix_dimensions(opening_point.len(), RATIO);

        tracing::debug!(
            L_size,
            R_size,
            opening_point_len = opening_point.len(),
            "Hyrax verify matrix dimensions"
        );

        let L: Vec<G::ScalarField> = EqPolynomial::evals(&opening_point[..L_size.log_2()]);
        let R: Vec<G::ScalarField> = EqPolynomial::evals(&opening_point[L_size.log_2()..]);
        tracing::debug!("Computed EqPolynomial evals for L and R");

        // Verifier-derived commitment to u * a = \prod Com(u_j)^{a_j}
        let normalized_commitments = G::normalize_batch(&commitment.row_commitments);
        tracing::debug!(
            num_bases = normalized_commitments.len(),
            "Normalized row commitments"
        );

        // In ZKVM guest environment, use grumpkin MSM (GLV + Pippenger) so that
        // group ops exercise the grumpkin inlines (division + GLV decomposition).
        let msm_eq = if cfg!(target_arch = "riscv64") || cfg!(target_arch = "riscv32") {
            use ark_ec::AffineRepr;
            use core::any::TypeId;
            use jolt_inlines_grumpkin::msm::{msm_glv_const, DEFAULT_GLV_WINDOW_BITS};
            use jolt_inlines_grumpkin::{GrumpkinFq, GrumpkinFr, GrumpkinPoint};

            debug_assert_eq!(
                TypeId::of::<G>(),
                TypeId::of::<ark_grumpkin::Projective>(),
                "grumpkin MSM backend only supported for G = ark_grumpkin::Projective"
            );

            let to_grumpkin_point = |p: &G::Affine| -> GrumpkinPoint {
                // SAFETY: guarded by the TypeId check above in debug builds; this code path is only
                // intended for the recursion verifier where Hyrax uses Grumpkin.
                let p: &ark_grumpkin::Affine =
                    unsafe { &*(p as *const G::Affine as *const ark_grumpkin::Affine) };
                if p.is_zero() {
                    GrumpkinPoint::infinity()
                } else {
                    GrumpkinPoint::new_unchecked(GrumpkinFq::new(p.x), GrumpkinFq::new(p.y))
                }
            };

            let to_grumpkin_scalar = |s: &G::ScalarField| -> GrumpkinFr {
                // SAFETY: same as above.
                let s: &ark_grumpkin::Fr =
                    unsafe { &*(s as *const G::ScalarField as *const ark_grumpkin::Fr) };
                GrumpkinFr::new(*s)
            };

            let bases_1: Vec<GrumpkinPoint> = normalized_commitments
                .iter()
                .map(to_grumpkin_point)
                .collect();
            let scalars_1: Vec<GrumpkinFr> = L.iter().map(to_grumpkin_scalar).collect();

            let _msm1_cycle = CycleMarkerGuard::new(CYCLE_HYRAX_MSM1);
            let homomorphically_derived_commitment: GrumpkinPoint =
                msm_glv_const::<GrumpkinPoint, { DEFAULT_GLV_WINDOW_BITS }>(&scalars_1, &bases_1);
            drop(_msm1_cycle);
            tracing::debug!(
                num_bases = bases_1.len(),
                "MSM #1: homomorphically derived commitment (grumpkin GLV MSM)"
            );

            let bases_2: Vec<GrumpkinPoint> = pedersen_generators.generators[..R_size]
                .iter()
                .map(to_grumpkin_point)
                .collect();
            let scalars_2: Vec<GrumpkinFr> = self
                .vector_matrix_product
                .iter()
                .map(to_grumpkin_scalar)
                .collect();

            let _msm2_cycle = CycleMarkerGuard::new(CYCLE_HYRAX_MSM2);
            let product_commitment: GrumpkinPoint =
                msm_glv_const::<GrumpkinPoint, { DEFAULT_GLV_WINDOW_BITS }>(&scalars_2, &bases_2);
            drop(_msm2_cycle);
            tracing::debug!(
                num_bases = bases_2.len(),
                vector_matrix_product_len = self.vector_matrix_product.len(),
                "MSM #2: product commitment (grumpkin GLV MSM)"
            );

            homomorphically_derived_commitment == product_commitment
        } else {
            let _msm1_cycle = CycleMarkerGuard::new(CYCLE_HYRAX_MSM1);
            let homomorphically_derived_commitment: G =
                VariableBaseMSM::msm(&normalized_commitments, &MultilinearPolynomial::from(L))
                    .unwrap();
            drop(_msm1_cycle);
            tracing::debug!(
                num_bases = normalized_commitments.len(),
                "MSM #1: homomorphically derived commitment"
            );

            let _msm2_cycle = CycleMarkerGuard::new(CYCLE_HYRAX_MSM2);
            let product_commitment = VariableBaseMSM::msm_field_elements(
                &pedersen_generators.generators[..R_size],
                &self.vector_matrix_product,
            )
            .unwrap();
            drop(_msm2_cycle);
            tracing::debug!(
                num_bases = R_size,
                vector_matrix_product_len = self.vector_matrix_product.len(),
                "MSM #2: product commitment"
            );

            homomorphically_derived_commitment == product_commitment
        };

        let dot_product = compute_dotproduct(&self.vector_matrix_product, &R);
        tracing::debug!("Computed dot product");

        if msm_eq && (dot_product == *opening) {
            Ok(())
        } else {
            Err(ProofVerifyError::InternalError)
        }
    }

    #[tracing::instrument(skip_all, name = "HyraxOpeningProof::vector_matrix_product")]
    fn vector_matrix_product(
        poly: &DensePolynomial<G::ScalarField>,
        L: &[G::ScalarField],
        ratio: usize,
    ) -> Vec<G::ScalarField> {
        let (_, R_size) = matrix_dimensions(poly.get_num_vars(), ratio);

        poly.evals_ref()
            .par_chunks(R_size)
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .map(|x| mul_0_1_optimized(&L[i], x))
                    .collect::<Vec<G::ScalarField>>()
            })
            .reduce(
                || vec![G::ScalarField::zero(); R_size],
                |mut acc: Vec<_>, row| {
                    acc.iter_mut().zip(row).for_each(|(x, y)| *x += y);
                    acc
                },
            )
    }
}

/// A Hyrax opening proof for multiple polynomials opened at the same point.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedHyraxOpeningProof<const RATIO: usize, G: CurveGroup> {
    pub joint_proof: HyraxOpeningProof<RATIO, G>,
}

/// See Section 16.1 of Thaler's Proofs, Arguments, and Zero-Knowledge
impl<const RATIO: usize, F: JoltField, G: CurveGroup<ScalarField = F>>
    BatchedHyraxOpeningProof<RATIO, G>
{
    #[tracing::instrument(skip_all, name = "BatchedHyraxOpeningProof::prove")]
    pub fn prove<ProofTranscript: Transcript>(
        polynomials: &[&DensePolynomial<G::ScalarField>],
        opening_point: &[G::ScalarField],
        openings: &[G::ScalarField],
        transcript: &mut ProofTranscript,
    ) -> Self {
        tracing::info!(
            num_polynomials = polynomials.len(),
            poly_size = polynomials[0].len(),
            "Starting batched Hyrax opening proof"
        );

        transcript.append_scalars(openings);

        let rlc_coefficients: Vec<_> = transcript.challenge_vector(polynomials.len());

        let _span = trace_span!("Compute RLC of polynomials");
        let _enter = _span.enter();

        let poly_len = polynomials[0].len();

        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = poly_len / num_chunks;

        let rlc_poly = if chunk_size > 0 {
            (0..num_chunks)
                .into_par_iter()
                .flat_map_iter(|chunk_index| {
                    let mut chunk = vec![G::ScalarField::zero(); chunk_size];
                    for (coeff, poly) in rlc_coefficients.iter().zip(polynomials.iter()) {
                        for (rlc, poly_eval) in chunk
                            .iter_mut()
                            .zip(poly.evals_ref()[chunk_index * chunk_size..].iter())
                        {
                            *rlc += mul_0_1_optimized(poly_eval, coeff);
                        }
                    }
                    chunk
                })
                .collect::<Vec<_>>()
        } else {
            rlc_coefficients
                .par_iter()
                .zip(polynomials.par_iter())
                .map(|(coeff, poly)| poly.evals_ref().iter().map(|eval| *coeff * *eval).collect())
                .reduce(
                    || vec![G::ScalarField::zero(); poly_len],
                    |running, new| {
                        debug_assert_eq!(running.len(), new.len());
                        running
                            .iter()
                            .zip(new.iter())
                            .map(|(r, n)| *r + *n)
                            .collect()
                    },
                )
        };

        drop(_enter);
        drop(_span);

        tracing::debug!("RLC computation complete");

        let joint_proof =
            HyraxOpeningProof::prove(&DensePolynomial::new(rlc_poly), opening_point, RATIO);

        tracing::debug!("Joint proof generation complete");

        Self { joint_proof }
    }

    #[tracing::instrument(skip_all, name = "BatchedHyraxOpeningProof::verify")]
    pub fn verify<ProofTranscript: Transcript>(
        &self,
        pedersen_generators: &PedersenGenerators<G>,
        opening_point: &[G::ScalarField],
        openings: &[G::ScalarField],
        commitments: &[&HyraxCommitment<RATIO, G>],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        assert_eq!(openings.len(), commitments.len());
        let (L_size, _R_size) = matrix_dimensions(opening_point.len(), RATIO);
        commitments.iter().enumerate().for_each(|(i, commitment)| {
            assert_eq!(
                L_size,
                commitment.row_commitments.len(),
                "Row commitment {}/{} wrong length.",
                i,
                commitments.len()
            )
        });

        transcript.append_scalars(openings);

        let rlc_coefficients: Vec<_> = transcript.challenge_vector(openings.len());

        let rlc_eval = compute_dotproduct(&rlc_coefficients, openings);

        let rlc_commitment = rlc_coefficients
            .par_iter()
            .zip(commitments.par_iter())
            .map(|(coeff, commitment)| {
                commitment
                    .row_commitments
                    .iter()
                    .map(|row_commitment| *row_commitment * coeff)
                    .collect()
            })
            .reduce(
                || vec![G::zero(); L_size],
                |running, new| {
                    debug_assert_eq!(running.len(), new.len());
                    running
                        .iter()
                        .zip(new.iter())
                        .map(|(r, n)| *r + n)
                        .collect()
                },
            );

        self.joint_proof.verify(
            pedersen_generators,
            opening_point,
            &rlc_eval,
            &HyraxCommitment {
                row_commitments: rlc_commitment,
            },
        )
    }
}

/// Wrapper struct for Hyrax to implement CommitmentScheme trait
#[derive(Clone)]
pub struct Hyrax<const RATIO: usize, G: CurveGroup>(std::marker::PhantomData<G>);

impl<const RATIO: usize, G: CurveGroup> CommitmentScheme for Hyrax<RATIO, G>
where
    G::ScalarField: JoltField,
{
    type Field = G::ScalarField;
    type ProverSetup = PedersenGenerators<G>;
    type VerifierSetup = PedersenGenerators<G>;
    type Commitment = HyraxCommitment<RATIO, G>;
    type Proof = HyraxOpeningProof<RATIO, G>;
    type BatchedProof = BatchedHyraxOpeningProof<RATIO, G>;
    type OpeningProofHint = ();

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let (_left, right) = matrix_dimensions(max_num_vars, RATIO);
        PedersenGenerators::new(right, b"Jolt v1 Hyrax generators")
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        // For Hyrax, verifier uses the same setup as prover
        setup.clone()
    }

    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let dense_poly = match poly {
            MultilinearPolynomial::LargeScalars(dense) => dense.clone(),
            MultilinearPolynomial::RLC(rlc) => {
                // For RLC polynomials, use the materialized dense representation
                DensePolynomial::new(rlc.dense_rlc.clone())
            }
            _ => panic!("Hyrax only supports dense and RLC polynomials"),
        };
        let commitment = HyraxCommitment::commit(&dense_poly, setup);
        (commitment, ())
    }

    fn batch_commit<U>(
        _polys: &[U],
        _gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        unimplemented!("Hyrax batch commit not implemented")
    }

    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        assert_eq!(commitments.len(), coeffs.len());

        if let Some(first) = commitments.first() {
            let row_count = first.borrow().row_commitments.len();

            let row_commitments = (0..row_count)
                .map(|row_idx| {
                    commitments
                        .iter()
                        .zip(coeffs)
                        .map(|(commitment, &coeff)| {
                            commitment.borrow().row_commitments[row_idx] * coeff
                        })
                        .sum()
                })
                .collect();

            HyraxCommitment { row_commitments }
        } else {
            HyraxCommitment::default()
        }
    }

    fn combine_hints(
        _hints: Vec<Self::OpeningProofHint>,
        _coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        // Hyrax doesn't use hints
    }

    fn prove<ProofTranscript: Transcript>(
        _setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        _hint: Option<Self::OpeningProofHint>,
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let dense_poly_owned;
        let dense_poly = match poly {
            MultilinearPolynomial::LargeScalars(dense) => dense,
            MultilinearPolynomial::RLC(rlc) => {
                // For RLC polynomials, use the materialized dense representation
                dense_poly_owned = DensePolynomial::new(rlc.dense_rlc.clone());
                &dense_poly_owned
            }
            _ => panic!("Hyrax only supports dense and RLC polynomials"),
        };

        let opening_point_field: Vec<G::ScalarField> = opening_point
            .iter()
            // .rev() // Hyrax uses opposite endian-ness
            .map(|challenge| (*challenge).into())
            .collect();

        HyraxOpeningProof::prove(dense_poly, &opening_point_field, RATIO)
    }

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        _transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let opening_point_field: Vec<G::ScalarField> = opening_point
            .iter()
            .map(|challenge| (*challenge).into())
            .collect();

        proof.verify(setup, &opening_point_field, opening, commitment)
    }

    fn protocol_name() -> &'static [u8] {
        b"hyrax"
    }
}
