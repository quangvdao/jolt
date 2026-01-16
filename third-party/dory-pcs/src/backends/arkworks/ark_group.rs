//! Group implementations for BN254 curve (G1, G2, GT)

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use super::ark_field::ArkFr;
use crate::primitives::arithmetic::{DoryRoutines, Group};
use ark_bn254::{Fq12, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::{Field as ArkField, One, PrimeField, UniformRand, Zero as ArkZero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::ops::{Add, Mul, Neg, Sub};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use rand_core::RngCore;

static GT_OP_TRACKING_ENABLED: AtomicBool = AtomicBool::new(false);
static GT_MUL_COUNT: AtomicU64 = AtomicU64::new(0);
static GT_EXP_COUNT: AtomicU64 = AtomicU64::new(0);

pub fn set_gt_op_tracking(enabled: bool) {
    GT_OP_TRACKING_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn reset_gt_op_counts() {
    GT_MUL_COUNT.store(0, Ordering::Relaxed);
    GT_EXP_COUNT.store(0, Ordering::Relaxed);
}

pub fn get_gt_op_counts() -> (u64, u64) {
    (
        GT_MUL_COUNT.load(Ordering::Relaxed),
        GT_EXP_COUNT.load(Ordering::Relaxed),
    )
}

#[inline]
fn track_gt_mul() {
    if GT_OP_TRACKING_ENABLED.load(Ordering::Relaxed) {
        GT_MUL_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
fn track_gt_exp() {
    if GT_OP_TRACKING_ENABLED.load(Ordering::Relaxed) {
        GT_EXP_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ArkG1(pub G1Projective);

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ArkG2(pub G2Projective);

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ArkGT(pub Fq12);

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
mod jolt_gt {
    use super::*;
    use ark_bn254::{Fq, Fq2, Fq6};
    use ark_ff::BigInt;
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

    #[inline(always)]
    fn exp_bit(exp: &[u64; FR_LIMBS_U64], bit_idx: usize) -> u64 {
        let limb = exp[bit_idx / 64];
        (limb >> (bit_idx % 64)) & 1
    }

    #[inline(always)]
    fn window_at<const WINDOW: usize>(exp: &[u64; FR_LIMBS_U64], i: usize) -> (usize, u32) {
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

    #[inline(always)]
    fn gt_mul(a: &Fq12, b: &Fq12) -> Fq12 {
        let a_limbs: [u64; GT_LIMBS_U64] = fq12_to_limbs_mont(a);
        let b_limbs: [u64; GT_LIMBS_U64] = fq12_to_limbs_mont(b);
        let mut out = [0u64; GT_LIMBS_U64];
        bn254_gt_mul_into(&mut out, &a_limbs, &b_limbs);
        fq12_from_limbs_mont(out)
    }

    #[inline(always)]
    pub fn gt_scale(base: &Fq12, k: &ArkFr) -> Fq12 {
        let exp_limbs: [u64; FR_LIMBS_U64] = k.0.into_bigint().0;

        // Highest set bit; if exp==0 return 1.
        let msb: Option<usize> = (0..FR_LIMBS_U64).rev().find_map(|limb_idx| {
            let limb = exp_limbs[limb_idx];
            if limb == 0 {
                None
            } else {
                Some(limb_idx * 64 + (63usize.saturating_sub(limb.leading_zeros() as usize)))
            }
        });

        let Some(msb) = msb else {
            return Fq12::one();
        };

        // Tuned default: w=5 (cheap squaring, expensive mul).
        const WINDOW: usize = 5;
        const TABLE_SIZE: usize = 1 << (WINDOW - 1);

        let base_limbs: [u64; GT_LIMBS_U64] = fq12_to_limbs_mont(base);

        // Precompute odd powers: base^(2i+1).
        let mut odd_pows = [[0u64; GT_LIMBS_U64]; TABLE_SIZE];
        odd_pows[0] = base_limbs;
        let mut base_sq = [0u64; GT_LIMBS_U64];
        bn254_gt_sqr_into(&mut base_sq, &base_limbs);
        for i in 1..TABLE_SIZE {
            let prev = odd_pows[i - 1];
            bn254_gt_mul_into(&mut odd_pows[i], &prev, &base_sq);
        }

        // Double-buffer accumulator to avoid 48-limb swaps.
        #[inline(always)]
        fn sqr_toggle(
            buf0: &mut [u64; GT_LIMBS_U64],
            buf1: &mut [u64; GT_LIMBS_U64],
            cur_is_0: &mut bool,
        ) {
            if *cur_is_0 {
                bn254_gt_sqr_into(buf1, buf0);
            } else {
                bn254_gt_sqr_into(buf0, buf1);
            }
            *cur_is_0 = !*cur_is_0;
        }

        #[inline(always)]
        fn mul_toggle(
            buf0: &mut [u64; GT_LIMBS_U64],
            buf1: &mut [u64; GT_LIMBS_U64],
            rhs: &[u64; GT_LIMBS_U64],
            cur_is_0: &mut bool,
        ) {
            if *cur_is_0 {
                bn254_gt_mul_into(buf1, buf0, rhs);
            } else {
                bn254_gt_mul_into(buf0, buf1, rhs);
            }
            *cur_is_0 = !*cur_is_0;
        }

        // Initialize acc with first (MSB) window.
        let (l0, u0) = window_at::<WINDOW>(&exp_limbs, msb);
        let mut buf0 = odd_pows[(u0 as usize) >> 1];
        let mut buf1 = [0u64; GT_LIMBS_U64];
        let mut cur_is_0 = true;

        let mut i: isize = msb as isize - l0 as isize;
        while i >= 0 {
            if exp_bit(&exp_limbs, i as usize) == 0 {
                sqr_toggle(&mut buf0, &mut buf1, &mut cur_is_0);
                i -= 1;
            } else {
                let (l, u) = window_at::<WINDOW>(&exp_limbs, i as usize);
                for _ in 0..l {
                    sqr_toggle(&mut buf0, &mut buf1, &mut cur_is_0);
                }
                let idx = (u as usize) >> 1;
                mul_toggle(&mut buf0, &mut buf1, &odd_pows[idx], &mut cur_is_0);
                i -= l as isize;
            }
        }

        let acc_limbs = if cur_is_0 { buf0 } else { buf1 };
        fq12_from_limbs_mont(acc_limbs)
    }

    #[inline(always)]
    pub fn gt_add(lhs: &Fq12, rhs: &Fq12) -> Fq12 {
        gt_mul(lhs, rhs)
    }
}

impl Group for ArkG1 {
    type Scalar = ArkFr;

    fn identity() -> Self {
        ArkG1(ArkZero::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArkG1(self.0 + rhs.0)
    }

    fn neg(&self) -> Self {
        ArkG1(-self.0)
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        ArkG1(self.0 * k.0)
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        ArkG1(G1Projective::rand(rng))
    }
}

impl Add for ArkG1 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        ArkG1(self.0 + rhs.0)
    }
}

impl Sub for ArkG1 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        ArkG1(self.0 - rhs.0)
    }
}

impl Neg for ArkG1 {
    type Output = Self;
    fn neg(self) -> Self {
        ArkG1(-self.0)
    }
}

impl<'a> Add<&'a ArkG1> for ArkG1 {
    type Output = ArkG1;
    fn add(self, rhs: &'a ArkG1) -> ArkG1 {
        ArkG1(self.0 + rhs.0)
    }
}

impl<'a> Sub<&'a ArkG1> for ArkG1 {
    type Output = ArkG1;
    fn sub(self, rhs: &'a ArkG1) -> ArkG1 {
        ArkG1(self.0 - rhs.0)
    }
}

impl Mul<ArkG1> for ArkFr {
    type Output = ArkG1;
    fn mul(self, rhs: ArkG1) -> ArkG1 {
        ArkG1(rhs.0 * self.0)
    }
}

impl<'a> Mul<&'a ArkG1> for ArkFr {
    type Output = ArkG1;
    fn mul(self, rhs: &'a ArkG1) -> ArkG1 {
        ArkG1(rhs.0 * self.0)
    }
}

impl Group for ArkG2 {
    type Scalar = ArkFr;

    fn identity() -> Self {
        ArkG2(ArkZero::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArkG2(self.0 + rhs.0)
    }

    fn neg(&self) -> Self {
        ArkG2(-self.0)
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        ArkG2(self.0 * k.0)
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        ArkG2(G2Projective::rand(rng))
    }
}

impl Add for ArkG2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        ArkG2(self.0 + rhs.0)
    }
}

impl Sub for ArkG2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        ArkG2(self.0 - rhs.0)
    }
}

impl Neg for ArkG2 {
    type Output = Self;
    fn neg(self) -> Self {
        ArkG2(-self.0)
    }
}

impl<'a> Add<&'a ArkG2> for ArkG2 {
    type Output = ArkG2;
    fn add(self, rhs: &'a ArkG2) -> ArkG2 {
        ArkG2(self.0 + rhs.0)
    }
}

impl<'a> Sub<&'a ArkG2> for ArkG2 {
    type Output = ArkG2;
    fn sub(self, rhs: &'a ArkG2) -> ArkG2 {
        ArkG2(self.0 - rhs.0)
    }
}

impl Mul<ArkG2> for ArkFr {
    type Output = ArkG2;
    fn mul(self, rhs: ArkG2) -> ArkG2 {
        ArkG2(rhs.0 * self.0)
    }
}

impl<'a> Mul<&'a ArkG2> for ArkFr {
    type Output = ArkG2;
    fn mul(self, rhs: &'a ArkG2) -> ArkG2 {
        ArkG2(rhs.0 * self.0)
    }
}

impl Group for ArkGT {
    type Scalar = ArkFr;

    fn identity() -> Self {
        ArkGT(Fq12::one())
    }

    fn add(&self, rhs: &Self) -> Self {
        track_gt_mul();
        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
        {
            ArkGT(jolt_gt::gt_add(&self.0, &rhs.0))
        }
        #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
        {
            ArkGT(self.0 * rhs.0)
        }
    }

    fn neg(&self) -> Self {
        ArkGT(ArkField::inverse(&self.0).expect("GT inverse"))
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        track_gt_exp();
        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
        {
            ArkGT(jolt_gt::gt_scale(&self.0, k))
        }
        #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
        {
            ArkGT(self.0.pow(k.0.into_bigint()))
        }
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        ArkGT(Fq12::rand(rng))
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Add for ArkGT {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        track_gt_mul();
        // GT is a multiplicative group, so group addition is field multiplication
        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
        {
            ArkGT(jolt_gt::gt_add(&self.0, &rhs.0))
        }
        #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
        {
            ArkGT(self.0 * rhs.0)
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Sub for ArkGT {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        track_gt_mul();
        // GT is a multiplicative group, so group subtraction is multiplication by inverse
        ArkGT(self.0 * rhs.0.inverse().expect("GT inverse"))
    }
}

impl Neg for ArkGT {
    type Output = Self;
    fn neg(self) -> Self {
        ArkGT(self.0.inverse().expect("GT inverse"))
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a> Add<&'a ArkGT> for ArkGT {
    type Output = ArkGT;
    fn add(self, rhs: &'a ArkGT) -> ArkGT {
        track_gt_mul();
        // GT is a multiplicative group, so group addition is field multiplication
        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
        {
            ArkGT(jolt_gt::gt_add(&self.0, &rhs.0))
        }
        #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
        {
            ArkGT(self.0 * rhs.0)
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a> Sub<&'a ArkGT> for ArkGT {
    type Output = ArkGT;
    fn sub(self, rhs: &'a ArkGT) -> ArkGT {
        track_gt_mul();
        // GT is a multiplicative group, so group subtraction is multiplication by inverse
        ArkGT(self.0 * rhs.0.inverse().expect("GT inverse"))
    }
}

impl Mul<ArkGT> for ArkFr {
    type Output = ArkGT;
    fn mul(self, rhs: ArkGT) -> ArkGT {
        track_gt_exp();
        ArkGT(rhs.0.pow(self.0.into_bigint()))
    }
}

impl<'a> Mul<&'a ArkGT> for ArkFr {
    type Output = ArkGT;
    fn mul(self, rhs: &'a ArkGT) -> ArkGT {
        track_gt_exp();
        ArkGT(rhs.0.pow(self.0.into_bigint()))
    }
}

pub struct G1Routines;

impl DoryRoutines<ArkG1> for G1Routines {
    #[tracing::instrument(skip_all, name = "G1::msm", fields(len = bases.len()))]
    fn msm(bases: &[ArkG1], scalars: &[ArkFr]) -> ArkG1 {
        assert_eq!(
            bases.len(),
            scalars.len(),
            "MSM requires equal length vectors"
        );

        if bases.is_empty() {
            return ArkG1::identity();
        }

        let bases_affine: Vec<G1Affine> = bases.iter().map(|b| b.0.into_affine()).collect();
        let scalars_fr: Vec<ark_bn254::Fr> = scalars.iter().map(|s| s.0).collect();

        ArkG1(G1Projective::msm(&bases_affine, &scalars_fr).expect("MSM failed"))
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG1, scalars: &[ArkFr]) -> Vec<ArkG1> {
        scalars.iter().map(|s| base.scale(s)).collect()
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG1], vs: &mut [ArkG1], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "Lengths must match");

        for (v, base) in vs.iter_mut().zip(bases.iter()) {
            *v = v.add(&base.scale(scalar));
        }
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG1], addends: &[ArkG1], scalar: &ArkFr) {
        assert_eq!(vs.len(), addends.len(), "Lengths must match");

        for (v, addend) in vs.iter_mut().zip(addends.iter()) {
            *v = v.scale(scalar).add(addend);
        }
    }
}

pub struct G2Routines;

impl DoryRoutines<ArkG2> for G2Routines {
    #[tracing::instrument(skip_all, name = "G2::msm", fields(len = bases.len()))]
    fn msm(bases: &[ArkG2], scalars: &[ArkFr]) -> ArkG2 {
        assert_eq!(
            bases.len(),
            scalars.len(),
            "MSM requires equal length vectors"
        );

        if bases.is_empty() {
            return ArkG2::identity();
        }

        let bases_affine: Vec<G2Affine> = bases.iter().map(|b| b.0.into_affine()).collect();
        let scalars_fr: Vec<ark_bn254::Fr> = scalars.iter().map(|s| s.0).collect();

        ArkG2(G2Projective::msm(&bases_affine, &scalars_fr).expect("MSM failed"))
    }

    fn fixed_base_vector_scalar_mul(base: &ArkG2, scalars: &[ArkFr]) -> Vec<ArkG2> {
        scalars.iter().map(|s| base.scale(s)).collect()
    }

    fn fixed_scalar_mul_bases_then_add(bases: &[ArkG2], vs: &mut [ArkG2], scalar: &ArkFr) {
        assert_eq!(bases.len(), vs.len(), "Lengths must match");

        for (v, base) in vs.iter_mut().zip(bases.iter()) {
            *v = v.add(&base.scale(scalar));
        }
    }

    fn fixed_scalar_mul_vs_then_add(vs: &mut [ArkG2], addends: &[ArkG2], scalar: &ArkFr) {
        assert_eq!(vs.len(), addends.len(), "Lengths must match");

        for (v, addend) in vs.iter_mut().zip(addends.iter()) {
            *v = v.scale(scalar).add(addend);
        }
    }
}
