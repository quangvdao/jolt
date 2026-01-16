//! Dory polynomial commitment scheme implementation

use super::dory_globals::DoryGlobals;
use super::jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
use super::wrappers::{
    jolt_to_ark, ArkDoryProof, ArkFr, ArkG1, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
    JoltToDoryTranscript, BN254,
};
use crate::{
    field::JoltField,
    poly::commitment::commitment_scheme::{CommitmentScheme, StreamingCommitmentScheme},
    poly::multilinear_polynomial::MultilinearPolynomial,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math, small_scalar::SmallScalar},
};
use ark_bn254::{G1Affine, G1Projective};
use ark_ec::CurveGroup;
use ark_ff::Zero;
use core::hint::black_box;
use dory::primitives::{
    arithmetic::{Group, PairingCurve},
    poly::Polynomial,
};
use jolt_platform::{end_cycle_tracking, start_cycle_tracking};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};
use std::borrow::Borrow;
use tracing::trace_span;

#[cfg(not(feature = "host"))]
fn gt_exp_inline(coeff: &ArkFr, base: &ArkGT) -> ArkGT {
    use ark_bn254::{Fq, Fq12, Fq2, Fq6};
    use ark_ff::{BigInt, One, PrimeField};
    use core::marker::PhantomData;
    use jolt_inlines_dory_gt_exp::sdk::{bn254_gt_mul_into, bn254_gt_sqr_into};
    use jolt_inlines_dory_gt_exp::{FR_LIMBS_U64, GT_LIMBS_U64};

    #[inline(always)]
    fn fq_to_limbs_mont(x: &Fq) -> [u64; 4] {
        x.0 .0
    }

    #[inline(always)]
    fn fq_from_limbs_mont(limbs: [u64; 4]) -> Fq {
        ark_ff::Fp::<_, 4>(BigInt(limbs), PhantomData)
    }

    #[inline(always)]
    fn fq2_to_limbs_mont(x: &Fq2) -> [u64; 8] {
        let c0 = fq_to_limbs_mont(&x.c0);
        let c1 = fq_to_limbs_mont(&x.c1);
        [c0[0], c0[1], c0[2], c0[3], c1[0], c1[1], c1[2], c1[3]]
    }

    #[inline(always)]
    fn fq2_from_limbs_mont(limbs: [u64; 8]) -> Fq2 {
        Fq2 {
            c0: fq_from_limbs_mont([limbs[0], limbs[1], limbs[2], limbs[3]]),
            c1: fq_from_limbs_mont([limbs[4], limbs[5], limbs[6], limbs[7]]),
        }
    }

    #[inline(always)]
    fn fq6_to_limbs_mont(x: &Fq6) -> [u64; 24] {
        let c0 = fq2_to_limbs_mont(&x.c0);
        let c1 = fq2_to_limbs_mont(&x.c1);
        let c2 = fq2_to_limbs_mont(&x.c2);
        let mut out = [0u64; 24];
        out[0..8].copy_from_slice(&c0);
        out[8..16].copy_from_slice(&c1);
        out[16..24].copy_from_slice(&c2);
        out
    }

    #[inline(always)]
    fn fq6_from_limbs_mont(limbs: [u64; 24]) -> Fq6 {
        Fq6 {
            c0: fq2_from_limbs_mont([
                limbs[0], limbs[1], limbs[2], limbs[3], limbs[4], limbs[5], limbs[6], limbs[7],
            ]),
            c1: fq2_from_limbs_mont([
                limbs[8], limbs[9], limbs[10], limbs[11], limbs[12], limbs[13], limbs[14],
                limbs[15],
            ]),
            c2: fq2_from_limbs_mont([
                limbs[16], limbs[17], limbs[18], limbs[19], limbs[20], limbs[21], limbs[22],
                limbs[23],
            ]),
        }
    }

    #[inline(always)]
    fn fq12_to_limbs_mont(x: &Fq12) -> [u64; 48] {
        let c0 = fq6_to_limbs_mont(&x.c0);
        let c1 = fq6_to_limbs_mont(&x.c1);
        let mut out = [0u64; 48];
        out[0..24].copy_from_slice(&c0);
        out[24..48].copy_from_slice(&c1);
        out
    }

    #[inline(always)]
    fn fq12_from_limbs_mont(limbs: [u64; 48]) -> Fq12 {
        Fq12 {
            c0: fq6_from_limbs_mont(limbs[0..24].try_into().unwrap()),
            c1: fq6_from_limbs_mont(limbs[24..48].try_into().unwrap()),
        }
    }

    let exp_limbs: [u64; FR_LIMBS_U64] = coeff.0.into_bigint().0;

    #[inline(always)]
    fn exp_bit(exp: &[u64; FR_LIMBS_U64], bit_idx: usize) -> u64 {
        let limb = exp[bit_idx / 64];
        (limb >> (bit_idx % 64)) & 1
    }

    // Find the highest set bit (skip leading zeros). If exp == 0, return 1 in Fq12.
    let msb: Option<usize> = (0..FR_LIMBS_U64).rev().find_map(|limb_idx| {
        let limb = exp_limbs[limb_idx];
        if limb == 0 {
            None
        } else {
            Some(limb_idx * 64 + (63usize.saturating_sub(limb.leading_zeros() as usize)))
        }
    });

    // acc := 1 in Fq12 (Montgomery form)
    let one_limbs = fq_to_limbs_mont(&Fq::one());
    let mut one_gt_limbs = [0u64; GT_LIMBS_U64];
    one_gt_limbs[0..4].copy_from_slice(&one_limbs);

    let Some(msb) = msb else {
        return ArkGT(fq12_from_limbs_mont(one_gt_limbs));
    };

    let base_limbs: [u64; GT_LIMBS_U64] = fq12_to_limbs_mont(&base.0);

    // Sliding-window exponentiation (left-to-right) in guest code using GT_SQR/GT_MUL inlines.
    // This reduces the number of GT multiplications relative to bit-by-bit square-and-multiply.
    //
    // Window size tradeoff:
    // - larger windows reduce muls in the main loop, but increase precomputation muls
    // - w=4 is a good first default for 256-bit exponents in a zkVM setting
    const WINDOW: usize = 4;
    const TABLE_SIZE: usize = 1 << (WINDOW - 1); // odd powers: 1,3,5,...,(2^w-1)

    // Precompute odd powers: base^(2i+1) for i in [0, TABLE_SIZE).
    let mut odd_pows = [[0u64; GT_LIMBS_U64]; TABLE_SIZE];
    odd_pows[0] = base_limbs;

    let mut base_sq = [0u64; GT_LIMBS_U64];
    bn254_gt_sqr_into(&mut base_sq, &base_limbs); // base^2
    for i in 1..TABLE_SIZE {
        // base^(2i+1) = base^(2(i-1)+1) * base^2
        let prev = odd_pows[i - 1];
        bn254_gt_mul_into(&mut odd_pows[i], &prev, &base_sq);
    }

    #[inline(always)]
    fn window_at(exp: &[u64; FR_LIMBS_U64], i: usize) -> (usize, u32) {
        let mut l = core::cmp::min(WINDOW, i + 1);
        while l > 1 && exp_bit(exp, i + 1 - l) == 0 {
            l -= 1;
        }
        let mut u: u32 = 0;
        for j in 0..l {
            u = (u << 1) | (exp_bit(exp, i - j) as u32);
        }
        debug_assert!((u & 1) == 1, "window value must be odd");
        (l, u)
    }

    // Initialize acc with the first (MSB) window, avoiding redundant squarings of 1.
    let (l0, u0) = window_at(&exp_limbs, msb);
    let mut acc_limbs = odd_pows[(u0 as usize) >> 1];

    let mut tmp = [0u64; GT_LIMBS_U64];
    let mut i: isize = msb as isize - l0 as isize;
    while i >= 0 {
        if exp_bit(&exp_limbs, i as usize) == 0 {
            // acc := acc^2
            bn254_gt_sqr_into(&mut tmp, &acc_limbs);
            core::mem::swap(&mut acc_limbs, &mut tmp);
            i -= 1;
        } else {
            let (l, u) = window_at(&exp_limbs, i as usize);
            for _ in 0..l {
                bn254_gt_sqr_into(&mut tmp, &acc_limbs);
                core::mem::swap(&mut acc_limbs, &mut tmp);
            }
            // u is odd, so index is (u - 1)/2 == u >> 1.
            let idx = (u as usize) >> 1;
            bn254_gt_mul_into(&mut tmp, &acc_limbs, &odd_pows[idx]);
            core::mem::swap(&mut acc_limbs, &mut tmp);
            i -= l as isize;
        }
    }

    ArkGT(fq12_from_limbs_mont(acc_limbs))
}

#[derive(Clone)]
pub struct DoryCommitmentScheme;

impl DoryCommitmentScheme {
    /// Derived GT op counts for Dory opening verification.
    ///
    /// This is based on the algorithm in `dory-pcs`:
    /// - Each reduce round performs 10 GT exponentiations and 11 GT multiplications.
    /// - Final verification performs 5 GT exponentiations and 8 GT multiplications.
    fn derived_opening_gt_op_counts(proof: &ArkDoryProof) -> (u64, u64) {
        let rounds = proof.sigma as u64;
        let exp_ops = 10 * rounds + 5;
        let mul_ops = 11 * rounds + 8;
        (mul_ops, exp_ops)
    }
}

impl CommitmentScheme for DoryCommitmentScheme {
    type Field = ark_bn254::Fr;
    type ProverSetup = ArkworksProverSetup;
    type VerifierSetup = ArkworksVerifierSetup;
    type Commitment = ArkGT;
    type Proof = ArkDoryProof;
    type BatchedProof = Vec<ArkDoryProof>;
    type OpeningProofHint = Vec<ArkG1>;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_prover").entered();
        let mut hasher = Sha3_256::new();
        hasher.update(b"Jolt Dory URS seed");
        let hash_result = hasher.finalize();
        let seed: [u8; 32] = hash_result.into();
        let mut rng = ChaCha20Rng::from_seed(seed);
        let setup = {
            #[cfg(feature = "prover")]
            {
                ArkworksProverSetup::new_from_urs(&mut rng, max_num_vars)
            }
            #[cfg(not(feature = "prover"))]
            {
                ArkworksProverSetup::new(&mut rng, max_num_vars)
            }
        };

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

        let (tier_2, row_commitments) = <MultilinearPolynomial<ark_bn254::Fr> as Polynomial<
            ArkFr,
        >>::commit::<BN254, JoltG1Routines>(
            poly, nu, sigma, setup
        )
        .expect("commitment should succeed");

        (tier_2, row_commitments)
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
    ) -> Self::Proof {
        let _span = trace_span!("DoryCommitmentScheme::prove").entered();

        let row_commitments = hint.unwrap_or_else(|| {
            let (_commitment, row_commitments) = Self::commit(poly, setup);
            row_commitments
        });

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        // Dory uses the opposite endian-ness as Jolt
        let ark_point: Vec<ArkFr> = opening_point
            .iter()
            .rev()  // Reverse the order for Dory
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();

        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        dory::prove::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _, _>(
            poly,
            &ark_point,
            row_commitments,
            nu,
            sigma,
            setup,
            &mut dory_transcript,
        )
        .expect("proof generation should succeed")
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
        const DORY_OPENING_VERIFY_SPAN: &str = "dory_opening_verify";

        // Dory uses the opposite endian-ness as Jolt
        let ark_point: Vec<ArkFr> = opening_point
            .iter()
            .rev()  // Reverse the order for Dory
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();
        let ark_eval: ArkFr = jolt_to_ark(opening);

        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        let (opening_gt_mul_ops, opening_gt_exp_ops) =
            DoryCommitmentScheme::derived_opening_gt_op_counts(proof);
        jolt_platform::jolt_println!(
            "dory_opening_gt_ops (derived): rounds={}, mul={}, exp={}",
            proof.sigma,
            opening_gt_mul_ops,
            opening_gt_exp_ops
        );

        start_cycle_tracking(DORY_OPENING_VERIFY_SPAN);
        dory::verify::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _>(
            *commitment,
            ark_eval,
            &ark_point,
            proof,
            setup.clone().into_inner(),
            &mut dory_transcript,
        )
        .map_err(|_| ProofVerifyError::InternalError)?;
        end_cycle_tracking(DORY_OPENING_VERIFY_SPAN);

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Dory"
    }

    /// In Dory, the opening proof hint consists of the Pedersen commitments to the rows
    /// of the polynomial coefficient matrix. In the context of a batch opening proof, we
    /// can homomorphically combine the row commitments for multiple polynomials into the
    /// row commitments for the RLC of those polynomials. This is more efficient than computing
    /// the row commitments for the RLC from scratch.
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_hints")]
    fn combine_hints(
        hints: Vec<Self::OpeningProofHint>,
        coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        let num_rows = DoryGlobals::get_max_num_rows();

        let mut rlc_hint = vec![ArkG1(G1Projective::zero()); num_rows];
        for (coeff, mut hint) in coeffs.iter().zip(hints.into_iter()) {
            hint.resize(num_rows, ArkG1(G1Projective::zero()));

            let row_commitments: &mut [G1Projective] = unsafe {
                std::slice::from_raw_parts_mut(hint.as_mut_ptr() as *mut G1Projective, hint.len())
            };

            let rlc_row_commitments: &[G1Projective] = unsafe {
                std::slice::from_raw_parts(rlc_hint.as_ptr() as *const G1Projective, rlc_hint.len())
            };

            let _span = trace_span!("vector_scalar_mul_add_gamma_g1_online");
            let _enter = _span.enter();

            // Scales the row commitments for the current polynomial by
            // its coefficient
            jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
                row_commitments,
                *coeff,
                rlc_row_commitments,
            );

            let _ = std::mem::replace(&mut rlc_hint, hint);
        }

        rlc_hint
    }

    /// Homomorphically combines multiple commitments using a random linear combination.
    /// Computes: sum_i(coeff_i * commitment_i) for the GT elements.
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_commitments")]
    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let _span = trace_span!("DoryCommitmentScheme::combine_commitments").entered();
        const GT_EXP_SPAN: &str = "dory_gt_exp_ops";
        const GT_MUL_SPAN: &str = "dory_gt_mul_ops";
        let exp_ops = coeffs.len() as u64;
        let mul_ops = coeffs.len().saturating_sub(1) as u64;
        jolt_platform::jolt_println!("dory_gt_combine_counts: exp={}, mul={}", exp_ops, mul_ops);
        start_cycle_tracking(GT_EXP_SPAN);

        // Combine GT elements using parallel RLC
        let commitments_vec: Vec<&ArkGT> = commitments.iter().map(|c| c.borrow()).collect();
        let exp_terms: Vec<ArkGT> = coeffs
            .par_iter()
            .zip(commitments_vec.par_iter())
            .map(|(coeff, commitment)| {
                let ark_coeff = jolt_to_ark(coeff);
                #[cfg(feature = "host")]
                {
                    black_box(ark_coeff) * black_box(**commitment)
                }
                #[cfg(not(feature = "host"))]
                {
                    // Use the BN254_GT_EXP inline inside the zkVM guest.
                    gt_exp_inline(&ark_coeff, commitment)
                }
            })
            .collect();
        end_cycle_tracking(GT_EXP_SPAN);

        start_cycle_tracking(GT_MUL_SPAN);
        let combined = exp_terms
            .par_iter()
            .cloned()
            .reduce(ArkGT::identity, |a, b| black_box(a) + black_box(b));
        end_cycle_tracking(GT_MUL_SPAN);
        combined
    }
}

impl StreamingCommitmentScheme for DoryCommitmentScheme {
    type ChunkState = Vec<ArkG1>; // Tier 1 commitment chunks

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::compute_tier1_commitment")]
    fn process_chunk<T: SmallScalar>(setup: &Self::ProverSetup, chunk: &[T]) -> Self::ChunkState {
        debug_assert_eq!(chunk.len(), DoryGlobals::get_num_columns());

        let row_len = DoryGlobals::get_num_columns();
        let g1_slice =
            unsafe { std::slice::from_raw_parts(setup.g1_vec.as_ptr(), setup.g1_vec.len()) };

        let g1_bases: Vec<G1Affine> = g1_slice[..row_len]
            .iter()
            .map(|g| g.0.into_affine())
            .collect();

        let row_commitment =
            ArkG1(T::msm(&g1_bases[..chunk.len()], chunk).expect("MSM calculation failed."));
        vec![row_commitment]
    }

    #[tracing::instrument(
        skip_all,
        name = "DoryCommitmentScheme::compute_tier1_commitment_onehot"
    )]
    fn process_chunk_onehot(
        setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState {
        let K = onehot_k;

        let row_len = DoryGlobals::get_num_columns();
        let g1_slice =
            unsafe { std::slice::from_raw_parts(setup.g1_vec.as_ptr(), setup.g1_vec.len()) };

        let g1_bases: Vec<G1Affine> = g1_slice[..row_len]
            .iter()
            .map(|g| g.0.into_affine())
            .collect();

        let mut indices_per_k: Vec<Vec<usize>> = vec![Vec::new(); K];
        for (col_index, k) in chunk.iter().enumerate() {
            if let Some(k) = k {
                indices_per_k[*k].push(col_index);
            }
        }

        let results = jolt_optimizations::batch_g1_additions_multi(&g1_bases, &indices_per_k);

        let mut row_commitments = vec![ArkG1(G1Projective::zero()); K];
        for (k, result) in results.into_iter().enumerate() {
            if !indices_per_k[k].is_empty() {
                row_commitments[k] = ArkG1(G1Projective::from(result));
            }
        }
        row_commitments
    }

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::compute_tier2_commitment")]
    fn aggregate_chunks(
        setup: &Self::ProverSetup,
        onehot_k: Option<usize>,
        chunks: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        if let Some(K) = onehot_k {
            let row_len = DoryGlobals::get_num_columns();
            let T = DoryGlobals::get_T();
            let rows_per_k = T / row_len;
            let num_rows = K * T / row_len;

            let mut row_commitments = vec![ArkG1(G1Projective::zero()); num_rows];
            for (chunk_index, commitments) in chunks.iter().enumerate() {
                row_commitments
                    .par_iter_mut()
                    .skip(chunk_index)
                    .step_by(rows_per_k)
                    .zip(commitments.par_iter())
                    .for_each(|(dest, src)| *dest = *src);
            }

            let g2_bases = &setup.g2_vec[..row_commitments.len()];
            let tier_2 = <BN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);

            (tier_2, row_commitments)
        } else {
            let row_commitments: Vec<ArkG1> =
                chunks.iter().flat_map(|chunk| chunk.clone()).collect();

            let g2_bases = &setup.g2_vec[..row_commitments.len()];
            let tier_2 = <BN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);

            (tier_2, row_commitments)
        }
    }
}
