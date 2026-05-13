use super::{FieldOps, JoltField};
#[cfg(feature = "challenge-254-bit")]
use crate::field::challenge::Mont254BitChallenge;
#[cfg(not(feature = "challenge-254-bit"))]
use crate::field::challenge::MontU128Challenge;
use crate::field::folded_accum::{
    Folded256MulU128, Folded256MulU128Accum, Folded256MulU64, Folded256Product,
    Folded256ProductAccum,
};
use crate::utils::thread::unsafe_allocate_zero_vec;
use ark_ff::BigInt;
use jolt_field::Fr;
use rayon::prelude::*;

impl FieldOps for Fr {}
impl FieldOps<&Fr, Fr> for &Fr {}
impl FieldOps<&Fr, Fr> for Fr {}

impl JoltField for Fr {
    const NUM_BYTES: usize = 32;
    const NUM_LIMBS: usize = 4;

    const MONTGOMERY_R: Self =
        unsafe { std::mem::transmute(<ark_bn254::Fr as JoltField>::MONTGOMERY_R) };
    const MONTGOMERY_R_SQUARE: Self =
        unsafe { std::mem::transmute(<ark_bn254::Fr as JoltField>::MONTGOMERY_R_SQUARE) };

    type UnreducedElem = BigInt<4>;
    type UnreducedMulU64 = Folded256MulU64;
    type UnreducedMulU128 = Folded256MulU128;
    type UnreducedMulU128Accum = Folded256MulU128Accum;
    type UnreducedProduct = Folded256Product;
    type UnreducedProductAccum = Folded256ProductAccum;
    type SmallValueLookupTables = [Vec<Self>; 2];

    #[cfg(not(feature = "challenge-254-bit"))]
    type Challenge = MontU128Challenge<Fr>;
    #[cfg(feature = "challenge-254-bit")]
    type Challenge = Mont254BitChallenge<Fr>;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        <Self as jolt_field::RandomSampling>::random(rng)
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
        <Self as jolt_field::FromPrimitiveInt>::from_bool(val)
    }

    #[inline]
    fn from_u8(n: u8) -> Self {
        <Self as jolt_field::FromPrimitiveInt>::from_u64(n as u64)
    }

    #[inline]
    fn from_u16(n: u16) -> Self {
        <Self as jolt_field::FromPrimitiveInt>::from_u64(n as u64)
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        <Self as jolt_field::FromPrimitiveInt>::from_u64(n as u64)
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        <Self as jolt_field::FromPrimitiveInt>::from_u64(n)
    }

    #[inline]
    fn from_i64(val: i64) -> Self {
        <Self as jolt_field::FromPrimitiveInt>::from_i64(val)
    }

    #[inline]
    fn from_i128(val: i128) -> Self {
        <Self as jolt_field::FromPrimitiveInt>::from_i128(val)
    }

    #[inline]
    fn from_u128(val: u128) -> Self {
        <Self as jolt_field::FromPrimitiveInt>::from_u128(val)
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        <Self as jolt_field::CanonicalU64>::to_canonical_u64_checked(self)
    }

    #[inline]
    fn square(&self) -> Self {
        <Self as jolt_field::RingCore>::square(self)
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        <Self as jolt_field::Invertible>::inverse(self)
    }

    #[inline]
    fn from_bytes(bytes: &[u8]) -> Self {
        <Self as jolt_field::ReducingBytes>::from_le_bytes_mod_order(bytes)
    }

    #[inline]
    fn num_bits(&self) -> u32 {
        <Self as jolt_field::CanonicalBitLength>::num_bits(self)
    }

    #[inline(always)]
    fn to_unreduced(&self) -> Self::UnreducedElem {
        let inner: ark_bn254::Fr = (*self).into();
        inner.0
    }

    #[inline]
    fn mul_u64(&self, n: u64) -> Self {
        Self::from(<ark_bn254::Fr as JoltField>::mul_u64(&(*self).into(), n))
    }

    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        Self::from(<ark_bn254::Fr as JoltField>::mul_i64(&(*self).into(), n))
    }

    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        Self::from(<ark_bn254::Fr as JoltField>::mul_u128(&(*self).into(), n))
    }

    #[inline]
    fn mul_i128(&self, n: i128) -> Self {
        Self::from(<ark_bn254::Fr as JoltField>::mul_i128(&(*self).into(), n))
    }

    #[inline]
    fn mul_u64_unreduced(self, other: u64) -> Folded256MulU64 {
        <ark_bn254::Fr as JoltField>::mul_u64_unreduced(self.into(), other)
    }

    #[inline]
    fn mul_u128_unreduced(self, other: u128) -> Folded256MulU128 {
        <ark_bn254::Fr as JoltField>::mul_u128_unreduced(self.into(), other)
    }

    #[inline]
    fn mul_to_product(self, other: Self) -> Folded256Product {
        <ark_bn254::Fr as JoltField>::mul_to_product(self.into(), other.into())
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Folded256ProductAccum {
        <ark_bn254::Fr as JoltField>::mul_to_product_accum(self.into(), other.into())
    }

    #[inline]
    fn unreduced_mul_u64(a: &BigInt<4>, b: u64) -> Folded256MulU64 {
        <ark_bn254::Fr as JoltField>::unreduced_mul_u64(a, b)
    }

    #[inline]
    fn unreduced_mul_to_product_accum(a: &BigInt<4>, b: &BigInt<4>) -> Folded256ProductAccum {
        <ark_bn254::Fr as JoltField>::unreduced_mul_to_product_accum(a, b)
    }

    #[inline]
    fn mul_to_accum_mag<const M: usize>(&self, mag: &BigInt<M>) -> Folded256MulU128Accum {
        <ark_bn254::Fr as JoltField>::mul_to_accum_mag(&(*self).into(), mag)
    }

    #[inline]
    fn mul_to_product_mag<const M: usize>(&self, mag: &BigInt<M>) -> Folded256Product {
        <ark_bn254::Fr as JoltField>::mul_to_product_mag(&(*self).into(), mag)
    }

    #[inline]
    fn reduce_mul_u64(x: Folded256MulU64) -> Self {
        Self::from(<ark_bn254::Fr as JoltField>::reduce_mul_u64(x))
    }

    #[inline]
    fn reduce_mul_u128(x: Folded256MulU128) -> Self {
        Self::from(<ark_bn254::Fr as JoltField>::reduce_mul_u128(x))
    }

    #[inline]
    fn reduce_mul_u128_accum(x: Folded256MulU128Accum) -> Self {
        Self::from(<ark_bn254::Fr as JoltField>::reduce_mul_u128_accum(x))
    }

    #[inline]
    fn reduce_product(x: Folded256Product) -> Self {
        Self::from(<ark_bn254::Fr as JoltField>::reduce_product(x))
    }

    #[inline]
    fn reduce_product_accum(x: Folded256ProductAccum) -> Self {
        Self::from(<ark_bn254::Fr as JoltField>::reduce_product_accum(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::JoltField;
    use rand_chacha::rand_core::RngCore;

    #[test]
    fn jolt_fr_matches_ark_fr_for_core_field_ops() {
        let mut rng = ark_std::test_rng();
        for _ in 0..128 {
            let x = rng.next_u64();
            let y = rng.next_u64();

            let ark_x = <ark_bn254::Fr as JoltField>::from_u64(x);
            let ark_y = <ark_bn254::Fr as JoltField>::from_u64(y);
            let jolt_x = <Fr as JoltField>::from_u64(x);
            let jolt_y = <Fr as JoltField>::from_u64(y);

            assert_eq!(ark_bn254::Fr::from(jolt_x + jolt_y), ark_x + ark_y);
            assert_eq!(ark_bn254::Fr::from(jolt_x * jolt_y), ark_x * ark_y);
            assert_eq!(
                ark_bn254::Fr::from(jolt_x.mul_u128(y as u128)),
                <ark_bn254::Fr as JoltField>::mul_u128(&ark_x, y as u128),
            );
        }
    }
}
