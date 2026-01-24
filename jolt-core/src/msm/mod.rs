use std::borrow::Borrow;

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::utils::errors::ProofVerifyError;
use ark_ec::scalar_mul::variable_base::{
    msm_binary, msm_i128, msm_i64, msm_s128, msm_s64, msm_u128, msm_u16, msm_u32, msm_u64, msm_u8,
    VariableBaseMSM as ArkVariableBaseMSM,
};
use ark_ec::{CurveGroup, ScalarMul};
use ark_ff::biginteger::{S128, S64};
use ark_ff::{One, PrimeField, Zero};
use rayon::prelude::*;

/// The result of this function is only approximately `ln(a)`.
///
/// Mirrors `ark_ec::scalar_mul::ln_without_floats`:
/// `ln(a) ≈ log2(a) * ln(2)`, with `ln(2) ≈ 0.69`.
#[inline]
fn ln_without_floats(a: usize) -> usize {
    debug_assert!(a > 0);
    // floor(log2(a)) * 69 / 100
    let log2 = (usize::BITS - 1 - a.leading_zeros()) as usize;
    log2 * 69 / 100
}

#[inline(always)]
fn extract_window_from_bigint_limbs(
    limbs: &[u64],
    window_start: usize,
    window_size: usize,
) -> usize {
    if window_size == 0 {
        return 0;
    }

    let limb_idx = window_start / 64;
    let bit_idx = window_start % 64;

    let lo = limbs.get(limb_idx).copied().unwrap_or(0);
    let hi = limbs.get(limb_idx + 1).copied().unwrap_or(0);

    let buf: u128 = ((hi as u128) << 64) | (lo as u128);
    let shifted = buf >> (bit_idx as u32);

    // window_size is expected to be small (Pippenger heuristic), but guard anyway.
    let mask = if window_size >= 128 {
        u128::MAX
    } else {
        (1u128 << (window_size as u32)) - 1
    };

    (shifted & mask) as usize
}

// A very light wrapper around Ark5.0 VariableBaseMSM
pub trait VariableBaseMSM: ArkVariableBaseMSM
where
    Self: ScalarMul, // technically implied by ArkVariableBaseMSM, but explicitly mentioned to be
    // consistent with current Jolt Msm implementation.
    Self::ScalarField: JoltField,
{
    #[tracing::instrument(skip_all)]
    fn msm<U>(bases: &[Self::MulBase], poly: &U) -> Result<Self, ProofVerifyError>
    where
        U: Borrow<MultilinearPolynomial<Self::ScalarField>> + Sync,
    {
        match poly.borrow() {
            MultilinearPolynomial::LargeScalars(poly) => {
                let scalars: &[Self::ScalarField] = poly.evals_ref();
                ArkVariableBaseMSM::msm(bases, scalars).map_err(|_bad_index| {
                    ProofVerifyError::KeyLengthError(bases.len(), scalars.len())
                })
            }

            MultilinearPolynomial::U8Scalars(poly) => (bases.len() == poly.coeffs.len())
                .then(|| {
                    let scalars = &poly.coeffs;
                    if scalars.par_iter().all(|&s| s == 0) {
                        Self::zero()
                    } else if scalars.par_iter().all(|&s| s <= 1) {
                        let bool_scalars: Vec<bool> = scalars.par_iter().map(|&s| s == 1).collect();
                        msm_binary::<Self>(bases, &bool_scalars, false)
                    } else {
                        msm_u8::<Self>(bases, scalars, false)
                    }
                })
                .ok_or(ProofVerifyError::KeyLengthError(
                    bases.len(),
                    poly.coeffs.len(),
                )),
            MultilinearPolynomial::U16Scalars(poly) => (bases.len() == poly.coeffs.len())
                .then(|| msm_u16::<Self>(bases, &poly.coeffs, false))
                .ok_or(ProofVerifyError::KeyLengthError(
                    bases.len(),
                    poly.coeffs.len(),
                )),
            MultilinearPolynomial::U32Scalars(poly) => (bases.len() == poly.coeffs.len())
                .then(|| msm_u32::<Self>(bases, &poly.coeffs, false))
                .ok_or(ProofVerifyError::KeyLengthError(
                    bases.len(),
                    poly.coeffs.len(),
                )),

            MultilinearPolynomial::U64Scalars(poly) => (bases.len() == poly.coeffs.len())
                .then(|| msm_u64::<Self>(bases, &poly.coeffs, false))
                .ok_or(ProofVerifyError::KeyLengthError(
                    bases.len(),
                    poly.coeffs.len(),
                )),

            MultilinearPolynomial::I64Scalars(poly) => (bases.len() == poly.coeffs.len())
                .then(|| msm_i64::<Self>(bases, &poly.coeffs, false))
                .ok_or(ProofVerifyError::KeyLengthError(
                    bases.len(),
                    poly.coeffs.len(),
                )),
            _ => unimplemented!("This variant of MultilinearPolynomial is not yet handled"),
        }
    }

    #[tracing::instrument(skip_all)]
    fn msm_field_elements(
        bases: &[Self::MulBase],
        scalars: &[Self::ScalarField],
    ) -> Result<Self, ProofVerifyError> {
        ArkVariableBaseMSM::msm_serial(bases, scalars)
            .map_err(|_bad_index| ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    /// Truly sequential Pippenger MSM (no rayon), intended for **nested parallelism**:
    /// e.g. when the caller parallelizes over many MSM instances (rows), each MSM should
    /// avoid spawning its own worker threads.
    ///
    /// This is based on arkworks' bucketed Pippenger routine, using the same window heuristic:
    /// `ln_without_floats(n) + 2` (and `3` for `n < 32`).
    #[tracing::instrument(skip_all)]
    fn msm_sequential(
        bases: &[Self::MulBase],
        scalars: &[Self::ScalarField],
    ) -> Result<Self, ProofVerifyError>
    where
        Self::ScalarField: PrimeField,
    {
        if bases.len() != scalars.len() {
            return Err(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()));
        }
        if bases.is_empty() {
            return Ok(Self::zero());
        }

        let n = bases.len();
        let window_size = if n < 32 { 3 } else { ln_without_floats(n) + 2 };

        // Guard against pathological `window_size` (would cause overflow / huge allocation).
        let two_to_c = match 1usize.checked_shl(window_size as u32) {
            Some(v) if v >= 2 => v,
            _ => {
                // Fallback to naive if the computed window is unusable.
                let mut result = Self::zero();
                for (b, s) in bases.iter().zip(scalars.iter()) {
                    result += *b * s;
                }
                return Ok(result);
            }
        };

        let scalar_bits = <Self::ScalarField as PrimeField>::MODULUS_BIT_SIZE as usize;

        // Precompute bigint scalars sequentially (arkworks' `msm_serial` still uses `cfg_into_iter!`).
        let bigints: Vec<<Self::ScalarField as PrimeField>::BigInt> =
            scalars.iter().map(|s| s.into_bigint()).collect();

        let mut window_sums = Vec::with_capacity(scalar_bits.div_ceil(window_size));
        for w_start in (0..scalar_bits).step_by(window_size) {
            let mut res = Self::zero();

            // We don't need the zero bucket, so we only allocate 2^c - 1 buckets.
            let mut buckets = vec![Self::zero(); two_to_c - 1];

            for ((base, scalar), bigint) in bases.iter().zip(scalars.iter()).zip(bigints.iter()) {
                if scalar.is_zero() {
                    continue;
                }
                if scalar.is_one() {
                    // Only process unit scalars once in the first window.
                    if w_start == 0 {
                        res += base;
                    }
                    continue;
                }

                let limbs = bigint.as_ref();
                let window_val = extract_window_from_bigint_limbs(limbs, w_start, window_size);

                if window_val != 0 {
                    // (Recall: `buckets` doesn't have a zero bucket.)
                    buckets[window_val - 1] += base;
                }
            }

            // `running_sum` = sum_{j in i..num_buckets} bucket[j], iterating backwards.
            let mut running_sum = Self::zero();
            for b in buckets.into_iter().rev() {
                running_sum += &b;
                res += &running_sum;
            }

            window_sums.push(res);
        }

        // Combine window sums from high to low (excluding lowest), shifting by `window_size` each step.
        let lowest = window_sums.first().cloned().unwrap_or_else(Self::zero);

        let mut total = Self::zero();
        for sum_i in window_sums.iter().skip(1).rev() {
            total += sum_i;
            for _ in 0..window_size {
                total.double_in_place();
            }
        }

        let mut out = lowest;
        out += total;
        Ok(out)
    }

    #[tracing::instrument(skip_all)]
    fn msm_u8(bases: &[Self::MulBase], scalars: &[u8]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| {
                if scalars.par_iter().all(|&s| s <= 1) {
                    let bool_scalars: Vec<bool> = scalars.par_iter().map(|&s| s == 1).collect();
                    msm_binary::<Self>(bases, &bool_scalars, true)
                } else {
                    msm_u8::<Self>(bases, scalars, true)
                }
            })
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_u16(bases: &[Self::MulBase], scalars: &[u16]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_u16::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_u32(bases: &[Self::MulBase], scalars: &[u32]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_u32::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_u64(bases: &[Self::MulBase], scalars: &[u64]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_u64::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_u128(bases: &[Self::MulBase], scalars: &[u128]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_u128::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_i64(bases: &[Self::MulBase], scalars: &[i64]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_i64::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_s64(bases: &[Self::MulBase], scalars: &[S64]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_s64::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_i128(bases: &[Self::MulBase], scalars: &[i128]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_i128::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    #[tracing::instrument(skip_all)]
    fn msm_s128(bases: &[Self::MulBase], scalars: &[S128]) -> Result<Self, ProofVerifyError> {
        (bases.len() == scalars.len())
            .then(|| msm_s128::<Self>(bases, scalars, true))
            .ok_or(ProofVerifyError::KeyLengthError(bases.len(), scalars.len()))
    }

    fn batch_msm<U>(bases: &[Self::MulBase], polys: &[U]) -> Vec<Self>
    where
        U: Borrow<MultilinearPolynomial<Self::ScalarField>> + Sync,
    {
        polys
            .par_iter()
            .map(|poly| VariableBaseMSM::msm(&bases[..poly.borrow().len()], poly).unwrap())
            .collect()
    }

    fn batch_msm_univariate(
        bases: &[Self::MulBase],
        polys: &[UniPoly<Self::ScalarField>],
    ) -> Vec<Self> {
        polys
            .par_iter()
            .map(|poly| {
                VariableBaseMSM::msm_field_elements(&bases[..poly.coeffs.len()], &poly.coeffs)
                    .unwrap()
            })
            .collect()
    }
}

// Implement VariableBaseMSM For any type G (like G1Projective) that implements the CurveGroup trait.
impl<F: JoltField, G: CurveGroup<ScalarField = F>> VariableBaseMSM for G {}
