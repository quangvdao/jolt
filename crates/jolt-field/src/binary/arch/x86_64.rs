use std::arch::x86_64::{
    __m128i, _mm_clmulepi64_si128, _mm_cvtsi64_si128, _mm_shuffle_epi32, _mm_slli_epi64,
    _mm_slli_si128, _mm_srli_epi64, _mm_srli_si128, _mm_xor_si128,
};

#[cfg(target_feature = "gfni")]
use std::arch::x86_64::{
    _mm_cvtsi128_si64, _mm_gf2p8affine_epi64_epi8, _mm_set1_epi64x, _mm_set_epi64x,
    _mm_setzero_si128, _mm_unpacklo_epi64, _mm_unpacklo_epi8,
};

#[cfg(all(target_feature = "avx2", target_feature = "vpclmulqdq"))]
use std::arch::x86_64::{
    _mm256_broadcastsi128_si256, _mm256_castsi256_si128, _mm256_clmulepi64_epi128,
    _mm256_extracti128_si256, _mm256_permute4x64_epi64,
};

#[derive(Clone, Copy)]
pub(super) struct Word(__m128i);

pub(super) type Unreduced64 = Word;
pub(super) const SCALAR_ACCUMULATOR64: bool = false;
pub(super) const KARATSUBA128: bool = true;
pub(super) const SHIFT_SQUARE128: bool = !cfg!(target_feature = "gfni");
pub(super) const KARATSUBA_ACCUMULATOR128: bool =
    !cfg!(all(target_feature = "avx2", target_feature = "vpclmulqdq"));

impl Word {
    #[inline]
    pub(super) fn reduce128([t0, t1, t2]: [Self; 3]) -> u128 {
        #[cfg(target_feature = "gfni")]
        {
            // SAFETY: GFNI is enabled by cfg; SSE2 is baseline. The matrices
            // are derived from the portable reducer. All three transforms
            // depend only on the high half, so their latencies overlap.
            unsafe {
                let low = t0 ^ t1.low_to_high();
                let high = t2 ^ t1.high_to_low();
                let first = _mm_gf2p8affine_epi64_epi8::<0>(
                    high.0,
                    _mm_set1_epi64x(const { affine_matrix::<true>(0, 0) } as i64),
                );
                let carry = _mm_gf2p8affine_epi64_epi8::<0>(
                    high.0,
                    _mm_set1_epi64x(const { affine_matrix::<true>(0, 8) } as i64),
                );
                let overflow_image = _mm_gf2p8affine_epi64_epi8::<0>(
                    _mm_srli_epi64::<56>(_mm_shuffle_epi32::<0xee>(high.0)),
                    _mm_set_epi64x(
                        const { affine_matrix::<true>(120, 8) } as i64,
                        const { affine_matrix::<true>(120, 0) } as i64,
                    ),
                );
                let overflow =
                    _mm_unpacklo_epi8(overflow_image, _mm_srli_si128::<8>(overflow_image));
                (low ^ Self(first) ^ Self(_mm_slli_si128::<1>(carry)) ^ Self(overflow)).to_u128()
            }
        }
        #[cfg(not(target_feature = "gfni"))]
        super::kernels::reduce128_products([t0, t1, t2])
    }

    #[inline]
    pub(super) fn tail128(self) -> Self {
        #[cfg(target_feature = "gfni")]
        {
            // SAFETY: GFNI is enabled by cfg; SSE2 is baseline. The two
            // matrices map a byte to its product and carry under the portable
            // modulus. Zeroing the high lane prevents low-byte contamination.
            unsafe {
                let image = _mm_gf2p8affine_epi64_epi8::<0>(
                    _mm_shuffle_epi32::<0xee>(self.0),
                    _mm_set_epi64x(
                        const { affine_matrix::<true>(0, 8) } as i64,
                        const { affine_matrix::<true>(0, 0) } as i64,
                    ),
                );
                Self(_mm_xor_si128(
                    _mm_unpacklo_epi64(image, _mm_setzero_si128()),
                    _mm_slli_si128::<1>(_mm_srli_si128::<8>(image)),
                ))
            }
        }
        #[cfg(not(target_feature = "gfni"))]
        self.mul_hl(Self::from_u64(
            const { super::portable::reduce128([0, 1]) } as u64,
        ))
    }

    #[inline]
    pub(super) fn schoolbook128(self, rhs: Self) -> [Self; 4] {
        #[cfg(all(target_feature = "avx2", target_feature = "vpclmulqdq"))]
        {
            // SAFETY: cfg enables AVX2 and VPCLMULQDQ. Each 128-bit lane
            // computes one independent limb product; the immediate exchanges
            // the two lanes for the cross products.
            unsafe {
                let a = _mm256_permute4x64_epi64::<0xd8>(_mm256_broadcastsi128_si256(self.0));
                let b = _mm256_permute4x64_epi64::<0xd8>(_mm256_broadcastsi128_si256(rhs.0));
                let diagonal = _mm256_clmulepi64_epi128::<0>(a, b);
                let cross = _mm256_clmulepi64_epi128::<0>(a, _mm256_permute4x64_epi64::<0x4e>(b));
                [
                    Self(_mm256_castsi256_si128(diagonal)),
                    Self(_mm256_castsi256_si128(cross)),
                    Self(_mm256_extracti128_si256::<1>(cross)),
                    Self(_mm256_extracti128_si256::<1>(diagonal)),
                ]
            }
        }
        #[cfg(not(all(target_feature = "avx2", target_feature = "vpclmulqdq")))]
        [
            self.mul_ll(rhs),
            self.mul_lh(rhs),
            self.mul_hl(rhs),
            self.mul_hh(rhs),
        ]
    }

    #[inline]
    pub(super) fn word_products128(self, rhs: Self) -> [Self; 2] {
        #[cfg(all(target_feature = "avx2", target_feature = "vpclmulqdq"))]
        {
            // SAFETY: cfg enables AVX2 and VPCLMULQDQ. rhs's low limb is
            // broadcast to both lanes; the two products use self's two limbs.
            unsafe {
                let a = _mm256_permute4x64_epi64::<0xd8>(_mm256_broadcastsi128_si256(self.0));
                let b = _mm256_broadcastsi128_si256(rhs.0);
                let product = _mm256_clmulepi64_epi128::<0>(a, b);
                [
                    Self(_mm256_castsi256_si128(product)),
                    Self(_mm256_extracti128_si256::<1>(product)),
                ]
            }
        }
        #[cfg(not(all(target_feature = "avx2", target_feature = "vpclmulqdq")))]
        [self.mul_ll(rhs), self.mul_hl(rhs)]
    }

    #[inline]
    pub(super) fn reduce64(self) -> u64 {
        #[cfg(target_feature = "gfni")]
        {
            // SAFETY: the cfg guarantees GFNI, and SSE2 is baseline. Affine
            // transforms act on each byte independently; the shifts place
            // byte carries and the x^64 overflow in their reduced positions.
            unsafe {
                let high = _mm_shuffle_epi32::<0xee>(self.0);
                let image = _mm_gf2p8affine_epi64_epi8::<0>(
                    high,
                    _mm_set_epi64x(
                        const { affine_matrix::<false>(0, 8) } as i64,
                        const { affine_matrix::<false>(0, 0) } as i64,
                    ),
                );
                let carry = _mm_slli_si128::<1>(_mm_srli_si128::<8>(image));
                let overflow = _mm_gf2p8affine_epi64_epi8::<0>(
                    _mm_srli_epi64::<56>(high),
                    _mm_set1_epi64x(const { affine_matrix::<false>(56, 0) } as i64),
                );
                let folded =
                    _mm_xor_si128(_mm_xor_si128(self.0, image), _mm_xor_si128(carry, overflow));
                _mm_cvtsi128_si64(folded) as u64
            }
        }
        #[cfg(not(target_feature = "gfni"))]
        super::portable::reduce64(self.to_u128())
    }

    #[inline]
    pub(super) fn from_unreduced64(value: Unreduced64) -> Self {
        value
    }

    #[inline]
    pub(super) fn into_unreduced64(self) -> Unreduced64 {
        self
    }

    #[inline]
    pub(super) fn from_u64(value: u64) -> Self {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { Self(_mm_cvtsi64_si128(value as i64)) }
    }

    #[inline]
    pub(super) fn from_u128(value: u128) -> Self {
        // A full-width load cannot forward two separate scalar stores.
        // Keep lane loads distinct until packing, without forcing a GP move.
        // SAFETY: SSE2 is baseline; AVX packing is selected only with AVX.
        // Both instructions interleave the low lanes without touching memory.
        unsafe {
            let mut low = _mm_cvtsi64_si128(value as i64);
            let high = _mm_cvtsi64_si128((value >> 64) as i64);
            #[cfg(target_feature = "avx")]
            std::arch::asm!(
                "vpunpcklqdq {low}, {low}, {high}",
                low = inout(xmm_reg) low,
                high = in(xmm_reg) high,
                options(pure, nomem, nostack, preserves_flags),
            );
            #[cfg(not(target_feature = "avx"))]
            std::arch::asm!(
                "punpcklqdq {low}, {high}",
                low = inout(xmm_reg) low,
                high = in(xmm_reg) high,
                options(pure, nomem, nostack, preserves_flags),
            );
            Self(low)
        }
    }

    #[inline]
    pub(super) fn from_accumulator_u128(value: u128) -> Self {
        // SAFETY: both types have 128 bits, all bit patterns are valid, and x86_64 is little-endian.
        unsafe { Self(std::mem::transmute::<u128, __m128i>(value)) }
    }

    #[inline]
    pub(super) fn swap64(self) -> Self {
        // SAFETY: SSE2 is baseline on x86_64; 0x4e exchanges the two 64-bit lanes.
        unsafe { Self(_mm_shuffle_epi32::<0x4e>(self.0)) }
    }

    #[inline]
    pub(super) fn to_u128(self) -> u128 {
        // SAFETY: both types have 128 bits, all bit patterns are valid, and x86_64 is little-endian.
        unsafe { std::mem::transmute::<__m128i, u128>(self.0) }
    }

    #[inline]
    pub(super) fn xor(self, rhs: Self) -> Self {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { Self(_mm_xor_si128(self.0, rhs.0)) }
    }

    #[inline]
    pub(super) fn mul_ll(self, rhs: Self) -> Self {
        // SAFETY: the module cfg guarantees pclmulqdq.
        unsafe { Self(_mm_clmulepi64_si128::<0x00>(self.0, rhs.0)) }
    }

    #[cfg(not(all(target_feature = "avx2", target_feature = "vpclmulqdq")))]
    #[inline]
    pub(super) fn mul_lh(self, rhs: Self) -> Self {
        // SAFETY: the module cfg guarantees pclmulqdq.
        unsafe { Self(_mm_clmulepi64_si128::<0x10>(self.0, rhs.0)) }
    }

    #[cfg(not(all(
        target_feature = "avx2",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    #[inline]
    pub(super) fn mul_hl(self, rhs: Self) -> Self {
        // SAFETY: the module cfg guarantees pclmulqdq.
        unsafe { Self(_mm_clmulepi64_si128::<0x01>(self.0, rhs.0)) }
    }

    #[inline]
    pub(super) fn mul_hh(self, rhs: Self) -> Self {
        // SAFETY: the module cfg guarantees pclmulqdq.
        unsafe { Self(_mm_clmulepi64_si128::<0x11>(self.0, rhs.0)) }
    }

    #[inline]
    pub(super) fn high_to_low(self) -> Self {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { Self(_mm_srli_si128::<8>(self.0)) }
    }

    #[inline]
    pub(super) fn shl<const N: i32>(self) -> Self {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { Self(_mm_slli_epi64::<N>(self.0)) }
    }

    #[inline]
    pub(super) fn shr<const N: i32>(self) -> Self {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { Self(_mm_srli_epi64::<N>(self.0)) }
    }

    #[inline]
    pub(super) fn low_to_high(self) -> Self {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { Self(_mm_slli_si128::<8>(self.0)) }
    }
}

#[cfg(target_feature = "gfni")]
const fn affine_matrix<const FIELD128: bool>(input_shift: u32, output_shift: u32) -> u64 {
    let mut matrix = 0;
    let mut input_bit = 0;
    while input_bit < 8 {
        let image = if FIELD128 {
            super::portable::reduce128([0, 1u128 << (input_shift + input_bit)])
        } else {
            super::portable::reduce64(1u128 << (64 + input_shift + input_bit)) as u128
        };
        let mut output_bit = 0;
        while output_bit < 8 {
            let bit = (image >> (output_shift + output_bit)) & 1;
            // GFNI numbers the matrix's output rows from the most significant byte.
            matrix |= (bit as u64) << ((7 - output_bit) * 8 + input_bit);
            output_bit += 1;
        }
        input_bit += 1;
    }
    matrix
}
