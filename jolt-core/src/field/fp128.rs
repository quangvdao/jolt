use super::{FieldOps, JoltField, UnreducedInteger};
use ark_ff::BigInt;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use hachi_pcs::algebra::Prime128M8M4M1M0;
use hachi_pcs::{CanonicalField, FieldCore, FieldSampling};
use num_traits::{One, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Jolt-compatible wrapper around hachi's 128-bit Solinas prime field.
///
/// Uses the prime `p = 2^128 - 275` (C=275), the smallest offset in the
/// hachi prime family, yielding the fastest Solinas reduction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct JoltFp128(pub Prime128M8M4M1M0);

impl Hash for JoltFp128 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_canonical_u128().hash(state);
    }
}

impl fmt::Display for JoltFp128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.to_canonical_u128())
    }
}

impl allocative::Allocative for JoltFp128 {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

// ---------------------------------------------------------------------------
// Core arithmetic operators
// ---------------------------------------------------------------------------

impl Add for JoltFp128 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<'a> Add<&'a JoltFp128> for JoltFp128 {
    type Output = JoltFp128;
    #[inline]
    fn add(self, rhs: &'a JoltFp128) -> JoltFp128 {
        Self(self.0 + rhs.0)
    }
}

impl Sub for JoltFp128 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl<'a> Sub<&'a JoltFp128> for JoltFp128 {
    type Output = JoltFp128;
    #[inline]
    fn sub(self, rhs: &'a JoltFp128) -> JoltFp128 {
        Self(self.0 - rhs.0)
    }
}

impl Mul for JoltFp128 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl<'a> Mul<&'a JoltFp128> for JoltFp128 {
    type Output = JoltFp128;
    #[inline]
    fn mul(self, rhs: &'a JoltFp128) -> JoltFp128 {
        Self(self.0 * rhs.0)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div for JoltFp128 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = FieldCore::inv(rhs.0).expect("division by zero");
        Self(self.0 * inv)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a> Div<&'a JoltFp128> for JoltFp128 {
    type Output = JoltFp128;
    #[inline]
    fn div(self, rhs: &'a JoltFp128) -> JoltFp128 {
        let inv = FieldCore::inv(rhs.0).expect("division by zero");
        Self(self.0 * inv)
    }
}

impl Neg for JoltFp128 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl AddAssign for JoltFp128 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0 + rhs.0;
    }
}

impl<'a> AddAssign<&'a JoltFp128> for JoltFp128 {
    #[inline]
    fn add_assign(&mut self, rhs: &'a JoltFp128) {
        self.0 = self.0 + rhs.0;
    }
}

impl SubAssign for JoltFp128 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0 - rhs.0;
    }
}

impl<'a> SubAssign<&'a JoltFp128> for JoltFp128 {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a JoltFp128) {
        self.0 = self.0 - rhs.0;
    }
}

impl MulAssign for JoltFp128 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = self.0 * rhs.0;
    }
}

// ---------------------------------------------------------------------------
// num_traits
// ---------------------------------------------------------------------------

impl Zero for JoltFp128 {
    fn zero() -> Self {
        Self(FieldCore::zero())
    }
    fn is_zero(&self) -> bool {
        FieldCore::is_zero(&self.0)
    }
}

impl One for JoltFp128 {
    fn one() -> Self {
        Self(FieldCore::one())
    }
    fn is_one(&self) -> bool {
        self.0.to_canonical_u128() == 1
    }
}

impl Sum for JoltFp128 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

impl<'a> Sum<&'a Self> for JoltFp128 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.copied().fold(Self::zero(), |a, b| a + b)
    }
}

impl Product for JoltFp128 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| a * b)
    }
}

impl<'a> Product<&'a Self> for JoltFp128 {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.copied().fold(Self::one(), |a, b| a * b)
    }
}

// ---------------------------------------------------------------------------
// FieldOps marker
// ---------------------------------------------------------------------------

impl FieldOps for JoltFp128 {}
impl FieldOps<&JoltFp128, JoltFp128> for JoltFp128 {}

impl PartialOrd for JoltFp128 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for JoltFp128 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.to_canonical_u128().cmp(&other.0.to_canonical_u128())
    }
}

/// Reconstruct a field element from the low 128 bits of an arbitrary-width limb array.
#[inline]
fn from_limb_slice(limbs: &[u64]) -> JoltFp128 {
    let lo = limbs.first().copied().unwrap_or(0);
    let hi = limbs.get(1).copied().unwrap_or(0);
    JoltFp128::from_u128((lo as u128) | ((hi as u128) << 64))
}

/// Reconstruct `Prime128M8M4M1M0` from a canonical 2-limb representation.
/// The input must already be a valid canonical field element (i.e. < p).
#[inline(always)]
fn limbs_to_fp128(limbs: [u64; 2]) -> Prime128M8M4M1M0 {
    let val = limbs[0] as u128 | ((limbs[1] as u128) << 64);
    Prime128M8M4M1M0::from_canonical_u128(val)
}

impl<const N: usize> From<[u64; N]> for JoltFp128 {
    #[inline]
    fn from(limbs: [u64; N]) -> Self {
        from_limb_slice(&limbs)
    }
}

impl<const N: usize> From<BigInt<N>> for JoltFp128 {
    #[inline]
    fn from(bigint: BigInt<N>) -> Self {
        from_limb_slice(&bigint.0)
    }
}

/// Fixed-width unreduced limb array for Fp128 delayed reduction.
///
/// Wraps `[u64; N]` in little-endian limb order. Supports limb-wise addition
/// for accumulating widened products before a single Solinas fold at the end.
///
/// Width mapping (Fp128, NUM_LIMBS = 2):
///   N=2 → UnreducedElem (field element identity)
///   N=3 → UnreducedMulU64 (field × u64)
///   N=4 → UnreducedMulU128 / UnreducedProduct (field × u128, field × field)
///   N=5 → UnreducedMulU128Accum / UnreducedProductAccum (accumulator headroom)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnreducedFp128<const N: usize>(pub [u64; N]);

impl<const N: usize> Default for UnreducedFp128<N> {
    #[inline(always)]
    fn default() -> Self {
        Self([0u64; N])
    }
}

impl<const N: usize> fmt::Display for UnreducedFp128<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UnreducedFp128({:?})", &self.0)
    }
}

impl<const N: usize> Ord for UnreducedFp128<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        for i in (0..N).rev() {
            match self.0[i].cmp(&other.0[i]) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }
}

impl<const N: usize> PartialOrd for UnreducedFp128<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> Zero for UnreducedFp128<N> {
    #[inline(always)]
    fn zero() -> Self {
        Self([0u64; N])
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|&l| l == 0)
    }
}

impl<const N: usize> From<u128> for UnreducedFp128<N> {
    #[inline]
    fn from(val: u128) -> Self {
        let mut limbs = [0u64; N];
        limbs[0] = val as u64;
        if N > 1 {
            limbs[1] = (val >> 64) as u64;
        }
        Self(limbs)
    }
}

impl<const N: usize> Add for UnreducedFp128<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let mut out = [0u64; N];
        let mut carry: u128 = 0;
        for i in 0..N {
            let s = self.0[i] as u128 + rhs.0[i] as u128 + carry;
            out[i] = s as u64;
            carry = s >> 64;
        }
        Self(out)
    }
}

impl<'a, const N: usize> Add<&'a Self> for UnreducedFp128<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: &'a Self) -> Self {
        self + *rhs
    }
}

impl<const N: usize> Sub for UnreducedFp128<N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let mut out = [0u64; N];
        let mut borrow: i128 = 0;
        for i in 0..N {
            let d = self.0[i] as i128 - rhs.0[i] as i128 + borrow;
            out[i] = d as u64;
            borrow = d >> 64;
        }
        Self(out)
    }
}

impl<'a, const N: usize> Sub<&'a Self> for UnreducedFp128<N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: &'a Self) -> Self {
        self - *rhs
    }
}

impl<const N: usize> AddAssign for UnreducedFp128<N> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<'a, const N: usize> AddAssign<&'a Self> for UnreducedFp128<N> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Self) {
        *self = *self + *rhs;
    }
}

impl<const N: usize> SubAssign for UnreducedFp128<N> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<'a, const N: usize> SubAssign<&'a Self> for UnreducedFp128<N> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Self) {
        *self = *self - *rhs;
    }
}

impl<const N: usize> UnreducedInteger for UnreducedFp128<N> {}

macro_rules! impl_cross_width_add {
    ($wide:literal, $narrow:literal) => {
        impl Add<UnreducedFp128<$narrow>> for UnreducedFp128<$wide> {
            type Output = UnreducedFp128<$wide>;
            #[inline(always)]
            fn add(mut self, rhs: UnreducedFp128<$narrow>) -> Self::Output {
                let mut carry: u128 = 0;
                for i in 0..$narrow {
                    let s = self.0[i] as u128 + rhs.0[i] as u128 + carry;
                    self.0[i] = s as u64;
                    carry = s >> 64;
                }
                for i in $narrow..$wide {
                    let s = self.0[i] as u128 + carry;
                    self.0[i] = s as u64;
                    carry = s >> 64;
                }
                self
            }
        }

        impl AddAssign<UnreducedFp128<$narrow>> for UnreducedFp128<$wide> {
            #[inline(always)]
            fn add_assign(&mut self, rhs: UnreducedFp128<$narrow>) {
                *self = *self + rhs;
            }
        }
    };
}

impl_cross_width_add!(3, 2);
impl_cross_width_add!(4, 2);
impl_cross_width_add!(4, 3);
impl_cross_width_add!(5, 2);
impl_cross_width_add!(5, 3);
impl_cross_width_add!(5, 4);

// ---------------------------------------------------------------------------
// Arkworks serialization (16 bytes, little-endian)
// ---------------------------------------------------------------------------

impl Valid for JoltFp128 {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for JoltFp128 {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        let val = self.0.to_canonical_u128();
        writer
            .write_all(&val.to_le_bytes())
            .map_err(SerializationError::IoError)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        16
    }
}

impl CanonicalDeserialize for JoltFp128 {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        let mut buf = [0u8; 16];
        reader
            .read_exact(&mut buf)
            .map_err(SerializationError::IoError)?;
        let val = u128::from_le_bytes(buf);
        Ok(Self(
            <Prime128M8M4M1M0 as CanonicalField>::from_canonical_u128_reduced(val),
        ))
    }
}

// ---------------------------------------------------------------------------
// UniformRand (arkworks)
// ---------------------------------------------------------------------------

impl ark_ff::UniformRand for JoltFp128 {
    fn rand<R: rand::Rng + ?Sized>(rng: &mut R) -> Self {
        // Can't forward to FieldSampling::sample because it requires R: Sized.
        // Inline the rejection-free reduction approach instead.
        let lo = rng.next_u64();
        let hi = rng.next_u64();
        let val = (lo as u128) | ((hi as u128) << 64);
        Self(<Prime128M8M4M1M0 as CanonicalField>::from_canonical_u128_reduced(val))
    }
}

// ---------------------------------------------------------------------------
// From<u128> (needed for Challenge = Self)
// ---------------------------------------------------------------------------

impl From<u128> for JoltFp128 {
    #[inline]
    fn from(val: u128) -> Self {
        Self(<Prime128M8M4M1M0 as CanonicalField>::from_canonical_u128_reduced(val))
    }
}

// ---------------------------------------------------------------------------
// JoltField implementation
// ---------------------------------------------------------------------------

impl JoltField for JoltFp128 {
    const NUM_BYTES: usize = 16;
    const NUM_LIMBS: usize = 2;

    // SAFETY: Prime128M8M4M1M0 is repr(transparent) over u128.
    // With trivial Montgomery (R=1), these are the multiplicative identity.
    const MONTGOMERY_R: Self = unsafe { std::mem::transmute(1u128) };
    const MONTGOMERY_R_SQUARE: Self = unsafe { std::mem::transmute(1u128) };

    type UnreducedElem = UnreducedFp128<2>;
    type UnreducedMulU64 = UnreducedFp128<3>;
    type UnreducedMulU128 = UnreducedFp128<4>;
    type UnreducedMulU128Accum = UnreducedFp128<5>;
    type UnreducedProduct = UnreducedFp128<4>;
    type UnreducedProductAccum = UnreducedFp128<5>;

    type SmallValueLookupTables = Vec<u8>;
    type Challenge = Self;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        Self(FieldSampling::sample(rng))
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
        Self(<Prime128M8M4M1M0 as CanonicalField>::from_u64(n as u64))
    }

    #[inline]
    fn from_u16(n: u16) -> Self {
        Self(<Prime128M8M4M1M0 as CanonicalField>::from_u64(n as u64))
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        Self(<Prime128M8M4M1M0 as CanonicalField>::from_u64(n as u64))
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        Self(<Prime128M8M4M1M0 as CanonicalField>::from_u64(n))
    }

    #[inline]
    fn from_i64(val: i64) -> Self {
        Self(<Prime128M8M4M1M0 as CanonicalField>::from_i64(val))
    }

    #[inline]
    fn from_i128(val: i128) -> Self {
        if val >= 0 {
            Self::from_u128(val as u128)
        } else {
            -Self::from_u128(val.unsigned_abs())
        }
    }

    #[inline]
    fn from_u128(val: u128) -> Self {
        Self(<Prime128M8M4M1M0 as CanonicalField>::from_canonical_u128_reduced(val))
    }

    #[inline]
    fn square(&self) -> Self {
        Self(self.0.square())
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let mut buf = [0u8; 16];
        let len = bytes.len().min(16);
        buf[..len].copy_from_slice(&bytes[..len]);
        let val = u128::from_le_bytes(buf);
        Self(<Prime128M8M4M1M0 as CanonicalField>::from_canonical_u128_reduced(val))
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        FieldCore::inv(self.0).map(Self)
    }

    fn to_u64(&self) -> Option<u64> {
        let val = self.0.to_canonical_u128();
        if val <= u64::MAX as u128 {
            Some(val as u64)
        } else {
            None
        }
    }

    fn num_bits(&self) -> u32 {
        let val = self.0.to_canonical_u128();
        128 - val.leading_zeros()
    }

    #[inline]
    fn to_unreduced(&self) -> Self::UnreducedElem {
        UnreducedFp128(self.0.to_limbs())
    }

    #[inline]
    fn mul_u64_unreduced(self, other: u64) -> Self::UnreducedMulU64 {
        UnreducedFp128(self.0.mul_wide_u64(other))
    }

    #[inline]
    fn mul_u128_unreduced(self, other: u128) -> Self::UnreducedMulU128 {
        UnreducedFp128(self.0.mul_wide_u128(other))
    }

    #[inline]
    fn mul_to_product(self, other: Self) -> Self::UnreducedProduct {
        UnreducedFp128(self.0.mul_wide(other.0))
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Self::UnreducedProductAccum {
        let [a, b, c, d] = self.0.mul_wide(other.0);
        UnreducedFp128([a, b, c, d, 0])
    }

    #[inline]
    fn unreduced_mul_u64(a: &Self::UnreducedElem, b: u64) -> Self::UnreducedMulU64 {
        let elem = limbs_to_fp128(a.0);
        UnreducedFp128(elem.mul_wide_u64(b))
    }

    #[inline]
    fn unreduced_mul_to_product_accum(
        a: &Self::UnreducedElem,
        b: &Self::UnreducedElem,
    ) -> Self::UnreducedProductAccum {
        let fa = limbs_to_fp128(a.0);
        let fb = limbs_to_fp128(b.0);
        let [w0, w1, w2, w3] = fa.mul_wide(fb);
        UnreducedFp128([w0, w1, w2, w3, 0])
    }

    #[inline]
    fn mul_to_accum_mag<const M: usize>(&self, mag: &BigInt<M>) -> Self::UnreducedMulU128Accum {
        let reduced = *self * JoltFp128::from(*mag);
        let limbs = reduced.0.to_limbs();
        UnreducedFp128([limbs[0], limbs[1], 0, 0, 0])
    }

    #[inline]
    fn mul_to_product_mag<const M: usize>(&self, mag: &BigInt<M>) -> Self::UnreducedProduct {
        let reduced = *self * JoltFp128::from(*mag);
        let limbs = reduced.0.to_limbs();
        UnreducedFp128([limbs[0], limbs[1], 0, 0])
    }

    #[inline]
    fn reduce_mul_u64(x: Self::UnreducedMulU64) -> Self {
        Self(Prime128M8M4M1M0::solinas_reduce(&x.0))
    }

    #[inline]
    fn reduce_mul_u128(x: Self::UnreducedMulU128) -> Self {
        Self(Prime128M8M4M1M0::solinas_reduce(&x.0))
    }

    #[inline]
    fn reduce_mul_u128_accum(x: Self::UnreducedMulU128Accum) -> Self {
        Self(Prime128M8M4M1M0::solinas_reduce(&x.0))
    }

    #[inline]
    fn reduce_product(x: Self::UnreducedProduct) -> Self {
        Self(Prime128M8M4M1M0::solinas_reduce(&x.0))
    }

    #[inline]
    fn reduce_product_accum(x: Self::UnreducedProductAccum) -> Self {
        Self(Prime128M8M4M1M0::solinas_reduce(&x.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::JoltField;

    #[test]
    fn basic_arithmetic() {
        let a = JoltFp128::from_u64(7);
        let b = JoltFp128::from_u64(11);
        let c = a + b;
        assert_eq!(c, JoltFp128::from_u64(18));

        let d = a * b;
        assert_eq!(d, JoltFp128::from_u64(77));

        let e = b - a;
        assert_eq!(e, JoltFp128::from_u64(4));

        let f = a * a.inverse().unwrap();
        assert_eq!(f, JoltFp128::one());
    }

    #[test]
    fn from_conversions() {
        assert_eq!(JoltFp128::from_bool(true), JoltFp128::one());
        assert_eq!(JoltFp128::from_bool(false), JoltFp128::zero());

        let neg = JoltFp128::from_i64(-5);
        assert_eq!(neg + JoltFp128::from_u64(5), JoltFp128::zero());

        let neg128 = JoltFp128::from_i128(-100);
        assert_eq!(neg128 + JoltFp128::from_u128(100), JoltFp128::zero());
    }

    #[test]
    fn serialization_roundtrip() {
        let val = JoltFp128::from_u64(123456789);
        let mut buf = vec![];
        val.serialize_uncompressed(&mut buf).unwrap();
        assert_eq!(buf.len(), 16);
        let restored = JoltFp128::deserialize_uncompressed(&buf[..]).unwrap();
        assert_eq!(val, restored);
    }

    #[test]
    fn challenge_from_u128() {
        let f = <JoltFp128 as JoltField>::Challenge::from(42u128);
        assert_eq!(f, JoltFp128::from_u64(42));
    }
}
