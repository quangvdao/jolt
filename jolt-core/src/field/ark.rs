use super::{FieldOps, JoltField, UnreducedInteger};
#[cfg(feature = "challenge-254-bit")]
use crate::field::challenge::Mont254BitChallenge;
#[cfg(not(feature = "challenge-254-bit"))]
use crate::field::challenge::MontU128Challenge;
use crate::utils::thread::unsafe_allocate_zero_vec;
use ark_ff::{prelude::*, BigInt, BigInteger, MontConfig, PrimeField, UniformRand};
use num_traits::Zero;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};

impl FieldOps for ark_bn254::Fr {}
impl FieldOps<&ark_bn254::Fr, ark_bn254::Fr> for &ark_bn254::Fr {}
impl FieldOps<&ark_bn254::Fr, ark_bn254::Fr> for ark_bn254::Fr {}

impl<const N: usize> UnreducedInteger for BigInt<N> {}

/// Redundant product accumulator for BN254: 8 u128 slots at positions [0, 64, 128, ..., 448].
///
/// Partial products from a 256x256 (4-limb × 4-limb) multiply are folded directly
/// into positional slots WITHOUT carry propagation between slots. Each slot uses
/// u128, giving headroom for accumulating ~2^61 products before overflow.
/// Normalization to `BigInt<9>` happens only at reduction time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FoldedBn254Accum(pub [u128; 8]);

impl fmt::Display for FoldedBn254Accum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FoldedBn254Accum({:?})", &self.0)
    }
}

impl Ord for FoldedBn254Accum {
    fn cmp(&self, other: &Self) -> Ordering {
        for i in (0..8).rev() {
            match self.0[i].cmp(&other.0[i]) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }
}

impl PartialOrd for FoldedBn254Accum {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Zero for FoldedBn254Accum {
    #[inline(always)]
    fn zero() -> Self {
        Self([0u128; 8])
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|&l| l == 0)
    }
}

impl From<u128> for FoldedBn254Accum {
    #[inline]
    fn from(val: u128) -> Self {
        Self([val, 0, 0, 0, 0, 0, 0, 0])
    }
}

impl Add for FoldedBn254Accum {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
            self.0[5] + rhs.0[5],
            self.0[6] + rhs.0[6],
            self.0[7] + rhs.0[7],
        ])
    }
}

impl<'a> Add<&'a Self> for FoldedBn254Accum {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: &'a Self) -> Self {
        self + *rhs
    }
}

impl Sub for FoldedBn254Accum {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
            self.0[2].wrapping_sub(rhs.0[2]),
            self.0[3].wrapping_sub(rhs.0[3]),
            self.0[4].wrapping_sub(rhs.0[4]),
            self.0[5].wrapping_sub(rhs.0[5]),
            self.0[6].wrapping_sub(rhs.0[6]),
            self.0[7].wrapping_sub(rhs.0[7]),
        ])
    }
}

impl<'a> Sub<&'a Self> for FoldedBn254Accum {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: &'a Self) -> Self {
        self - *rhs
    }
}

impl AddAssign for FoldedBn254Accum {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
        self.0[3] += rhs.0[3];
        self.0[4] += rhs.0[4];
        self.0[5] += rhs.0[5];
        self.0[6] += rhs.0[6];
        self.0[7] += rhs.0[7];
    }
}

impl<'a> AddAssign<&'a Self> for FoldedBn254Accum {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Self) {
        *self += *rhs;
    }
}

impl SubAssign for FoldedBn254Accum {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_sub(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_sub(rhs.0[1]);
        self.0[2] = self.0[2].wrapping_sub(rhs.0[2]);
        self.0[3] = self.0[3].wrapping_sub(rhs.0[3]);
        self.0[4] = self.0[4].wrapping_sub(rhs.0[4]);
        self.0[5] = self.0[5].wrapping_sub(rhs.0[5]);
        self.0[6] = self.0[6].wrapping_sub(rhs.0[6]);
        self.0[7] = self.0[7].wrapping_sub(rhs.0[7]);
    }
}

impl<'a> SubAssign<&'a Self> for FoldedBn254Accum {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Self) {
        *self -= *rhs;
    }
}

impl UnreducedInteger for FoldedBn254Accum {}

impl AddAssign<BigInt<4>> for FoldedBn254Accum {
    #[inline(always)]
    fn add_assign(&mut self, rhs: BigInt<4>) {
        self.0[0] += rhs.0[0] as u128;
        self.0[1] += rhs.0[1] as u128;
        self.0[2] += rhs.0[2] as u128;
        self.0[3] += rhs.0[3] as u128;
    }
}

impl AddAssign<BigInt<8>> for FoldedBn254Accum {
    #[inline(always)]
    fn add_assign(&mut self, rhs: BigInt<8>) {
        self.0[0] += rhs.0[0] as u128;
        self.0[1] += rhs.0[1] as u128;
        self.0[2] += rhs.0[2] as u128;
        self.0[3] += rhs.0[3] as u128;
        self.0[4] += rhs.0[4] as u128;
        self.0[5] += rhs.0[5] as u128;
        self.0[6] += rhs.0[6] as u128;
        self.0[7] += rhs.0[7] as u128;
    }
}

impl FoldedBn254Accum {
    /// Folded 256x256 multiply: produces 16 partial products and folds them
    /// directly into 8 positional slots WITHOUT carry propagation.
    ///
    /// 16 `mul` + 16 `umulh` instructions, all independent. Zero carry chain.
    #[inline(always)]
    pub fn from_mul(a: BigInt<4>, b: BigInt<4>) -> Self {
        let (a0, a1, a2, a3) = (a.0[0], a.0[1], a.0[2], a.0[3]);
        let (b0, b1, b2, b3) = (b.0[0], b.0[1], b.0[2], b.0[3]);

        let p00 = (a0 as u128) * (b0 as u128);
        let p01 = (a0 as u128) * (b1 as u128);
        let p10 = (a1 as u128) * (b0 as u128);
        let p02 = (a0 as u128) * (b2 as u128);
        let p11 = (a1 as u128) * (b1 as u128);
        let p20 = (a2 as u128) * (b0 as u128);
        let p03 = (a0 as u128) * (b3 as u128);
        let p12 = (a1 as u128) * (b2 as u128);
        let p21 = (a2 as u128) * (b1 as u128);
        let p30 = (a3 as u128) * (b0 as u128);
        let p13 = (a1 as u128) * (b3 as u128);
        let p22 = (a2 as u128) * (b2 as u128);
        let p31 = (a3 as u128) * (b1 as u128);
        let p23 = (a2 as u128) * (b3 as u128);
        let p32 = (a3 as u128) * (b2 as u128);
        let p33 = (a3 as u128) * (b3 as u128);

        #[inline(always)]
        fn lo(v: u128) -> u128 {
            (v as u64) as u128
        }
        #[inline(always)]
        fn hi(v: u128) -> u128 {
            (v >> 64) as u64 as u128
        }

        Self([
            lo(p00),
            hi(p00) + lo(p01) + lo(p10),
            hi(p01) + hi(p10) + lo(p02) + lo(p11) + lo(p20),
            hi(p02) + hi(p11) + hi(p20) + lo(p03) + lo(p12) + lo(p21) + lo(p30),
            hi(p03) + hi(p12) + hi(p21) + hi(p30) + lo(p13) + lo(p22) + lo(p31),
            hi(p13) + hi(p22) + hi(p31) + lo(p23) + lo(p32),
            hi(p23) + hi(p32) + lo(p33),
            hi(p33),
        ])
    }

    /// Normalize to `BigInt<9>` by carry-propagating the u128 hi-halves.
    #[inline]
    pub fn normalize(self) -> BigInt<9> {
        let p0 = self.0[0];
        let p1 = self.0[1] + (p0 >> 64);
        let p2 = self.0[2] + (p1 >> 64);
        let p3 = self.0[3] + (p2 >> 64);
        let p4 = self.0[4] + (p3 >> 64);
        let p5 = self.0[5] + (p4 >> 64);
        let p6 = self.0[6] + (p5 >> 64);
        let p7 = self.0[7] + (p6 >> 64);
        BigInt::new([
            p0 as u64,
            p1 as u64,
            p2 as u64,
            p3 as u64,
            p4 as u64,
            p5 as u64,
            p6 as u64,
            p7 as u64,
            (p7 >> 64) as u64,
        ])
    }
}

impl JoltField for ark_bn254::Fr {
    const NUM_BYTES: usize = 32;
    const NUM_LIMBS: usize = 4;

    // SAFETY: Transmuting from the Montgomery R constants from arkworks,
    // which are guaranteed to be valid field elements in Montgomery form.
    const MONTGOMERY_R: Self =
        unsafe { std::mem::transmute(<ark_bn254::FrConfig as MontConfig<4>>::R) };
    const MONTGOMERY_R_SQUARE: Self =
        unsafe { std::mem::transmute(<ark_bn254::FrConfig as MontConfig<4>>::R2) };

    type UnreducedElem = BigInt<4>;
    type UnreducedMulU64 = BigInt<5>;
    type UnreducedMulU128 = BigInt<6>;
    type UnreducedMulU128Accum = BigInt<7>;
    type UnreducedProduct = BigInt<8>;
    type UnreducedProductAccum = FoldedBn254Accum;

    type SmallValueLookupTables = [Vec<Self>; 2];

    #[cfg(not(feature = "challenge-254-bit"))]
    type Challenge = MontU128Challenge<ark_bn254::Fr>;
    #[cfg(feature = "challenge-254-bit")]
    type Challenge = Mont254BitChallenge<ark_bn254::Fr>;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        <Self as UniformRand>::rand(rng)
    }

    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
        let mut lookup_tables = [
            unsafe_allocate_zero_vec(1 << 16),
            unsafe_allocate_zero_vec(1 << 16),
        ];

        for i in 0..2 {
            let bitshift = 16 * i;
            let unit = <Self as JoltField>::from_u64(1 << bitshift);
            lookup_tables[i] = (0..(1 << 16))
                .into_par_iter()
                .map(|j| unit * <Self as JoltField>::from_u64(j))
                .collect();
        }

        lookup_tables
    }

    #[inline]
    fn from_bool(val: bool) -> Self {
        if val {
            Self::one()
        } else {
            Self::zero()
        }
    }

    #[inline]
    fn from_u8(n: u8) -> Self {
        <Self as PrimeField>::from_u64::<5>(n as u64).unwrap()
    }

    #[inline]
    fn from_u16(n: u16) -> Self {
        <Self as PrimeField>::from_u64::<5>(n as u64).unwrap()
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        <Self as PrimeField>::from_u64::<5>(n as u64).unwrap()
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        if n <= u16::MAX as u64 {
            <Self as JoltField>::from_u16(n as u16)
        } else if n <= u32::MAX as u64 {
            <Self as JoltField>::from_u32(n as u32)
        } else {
            <Self as PrimeField>::from_u64::<5>(n).unwrap()
        }
    }

    #[inline]
    fn from_i64(val: i64) -> Self {
        if val.is_negative() {
            let val = val.unsigned_abs();
            if val <= u16::MAX as u64 {
                -<Self as JoltField>::from_u16(val as u16)
            } else if val <= u32::MAX as u64 {
                -<Self as JoltField>::from_u32(val as u32)
            } else {
                -<Self as JoltField>::from_u64(val)
            }
        } else {
            let val = val as u64;
            if val <= u16::MAX as u64 {
                <Self as JoltField>::from_u16(val as u16)
            } else if val <= u32::MAX as u64 {
                <Self as JoltField>::from_u32(val as u32)
            } else {
                <Self as JoltField>::from_u64(val)
            }
        }
    }

    #[inline]
    fn from_i128(val: i128) -> Self {
        if val.is_negative() {
            let val = val.unsigned_abs();
            if val <= u16::MAX as u128 {
                -<Self as JoltField>::from_u16(val as u16)
            } else if val <= u32::MAX as u128 {
                -<Self as JoltField>::from_u32(val as u32)
            } else if val <= u64::MAX as u128 {
                -<Self as JoltField>::from_u64(val as u64)
            } else {
                let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
                -<Self as PrimeField>::from_bigint(bigint).unwrap()
            }
        } else {
            let val = val as u128;
            if val <= u16::MAX as u128 {
                <Self as JoltField>::from_u16(val as u16)
            } else if val <= u32::MAX as u128 {
                <Self as JoltField>::from_u32(val as u32)
            } else if val <= u64::MAX as u128 {
                <Self as JoltField>::from_u64(val as u64)
            } else {
                let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
                <Self as PrimeField>::from_bigint(bigint).unwrap()
            }
        }
    }

    #[inline]
    fn from_u128(val: u128) -> Self {
        if val <= u16::MAX as u128 {
            <Self as JoltField>::from_u16(val as u16)
        } else if val <= u32::MAX as u128 {
            <Self as JoltField>::from_u32(val as u32)
        } else if val <= u64::MAX as u128 {
            <Self as JoltField>::from_u64(val as u64)
        } else {
            let bigint = BigInt::new([val as u64, (val >> 64) as u64, 0, 0]);
            <Self as PrimeField>::from_bigint(bigint).unwrap()
        }
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        let bigint = <Self as PrimeField>::into_bigint(*self);
        let limbs: &[u64] = bigint.as_ref();
        let result = limbs[0];

        if <Self as JoltField>::from_u64(result) != *self {
            None
        } else {
            Some(result)
        }
    }

    #[inline]
    fn square(&self) -> Self {
        <Self as ark_ff::Field>::square(self)
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        <Self as ark_ff::Field>::inverse(self)
    }

    #[inline]
    fn from_bytes(bytes: &[u8]) -> Self {
        ark_bn254::Fr::from_le_bytes_mod_order(bytes)
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        <Self as PrimeField>::into_bigint(*self).num_bits()
    }

    #[inline(always)]
    fn to_unreduced(&self) -> Self::UnreducedElem {
        self.0
    }

    #[inline]
    fn mul_u64(&self, n: u64) -> Self {
        if n == 0 || self.is_zero() {
            Self::zero()
        } else if n == 1 {
            *self
        } else {
            ark_ff::Fp::mul_u64::<5>(*self, n)
        }
    }

    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        ark_ff::Fp::mul_i64::<5>(*self, n)
    }

    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        ark_ff::Fp::mul_u128::<5, 6>(*self, n)
    }

    #[inline]
    fn mul_i128(&self, n: i128) -> Self {
        if n == 0 || self.is_zero() {
            Self::zero()
        } else if n == 1 {
            *self
        } else {
            ark_ff::Fp::mul_i128::<5, 6>(*self, n)
        }
    }

    #[inline]
    fn mul_u64_unreduced(self, other: u64) -> BigInt<5> {
        self.0.mul_trunc::<1, 5>(&BigInt::new([other]))
    }

    #[inline]
    fn mul_u128_unreduced(self, other: u128) -> BigInt<6> {
        self.0
            .mul_trunc::<2, 6>(&BigInt::new([other as u64, (other >> 64) as u64]))
    }

    #[inline]
    fn mul_to_product(self, other: Self) -> BigInt<8> {
        self.0.mul_trunc::<4, 8>(&other.0)
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> FoldedBn254Accum {
        FoldedBn254Accum::from_mul(self.0, other.0)
    }

    #[inline]
    fn unreduced_mul_u64(a: &BigInt<4>, b: u64) -> BigInt<5> {
        a.mul_u64_w_carry(b)
    }

    #[inline]
    fn unreduced_mul_to_product_accum(a: &BigInt<4>, b: &BigInt<4>) -> FoldedBn254Accum {
        FoldedBn254Accum::from_mul(*a, *b)
    }

    #[inline]
    fn mul_to_accum_mag<const M: usize>(&self, mag: &BigInt<M>) -> BigInt<7> {
        self.0.mul_trunc::<M, 7>(mag)
    }

    #[inline]
    fn mul_to_product_mag<const M: usize>(&self, mag: &BigInt<M>) -> BigInt<8> {
        self.0.mul_trunc::<M, 8>(mag)
    }

    #[inline]
    fn reduce_mul_u64(x: BigInt<5>) -> Self {
        ark_bn254::Fr::from_barrett_reduce::<5, 5>(x)
    }

    #[inline]
    fn reduce_mul_u128(x: BigInt<6>) -> Self {
        ark_bn254::Fr::from_barrett_reduce::<6, 5>(x)
    }

    #[inline]
    fn reduce_mul_u128_accum(x: BigInt<7>) -> Self {
        ark_bn254::Fr::from_barrett_reduce::<7, 5>(x)
    }

    #[inline]
    fn reduce_product(x: BigInt<8>) -> Self {
        ark_bn254::Fr::from_montgomery_reduce::<8, 5>(x)
    }

    #[inline]
    fn reduce_product_accum(x: FoldedBn254Accum) -> Self {
        ark_bn254::Fr::from_montgomery_reduce::<9, 5>(x.normalize())
    }
}

#[cfg(test)]
mod tests {
    use crate::field::JoltField;
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use ark_std::One;
    use rand_chacha::rand_core::RngCore;

    #[test]
    fn implicit_montgomery_conversion() {
        let mut rng = test_rng();
        for _ in 0..256 {
            let x = rng.next_u64();
            assert_eq!(
                <Fr as JoltField>::from_u64(x),
                JoltField::mul_u64(&Fr::one(), x)
            );
        }

        for _ in 0..256 {
            let x = rng.next_u64();
            let y = Fr::random(&mut rng);
            assert_eq!(
                y * <Fr as JoltField>::from_u64(x),
                JoltField::mul_u64(&y, x)
            );
        }
    }
}
