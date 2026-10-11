use super::super::butterfly::{base_tail, extension_tail};
use super::super::{F192, F64};
use super::fold64;
use crate::ExtField;
#[cfg(all(
    all(target_feature = "avx2", target_feature = "vpclmulqdq"),
    not(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "vpclmulqdq"
    ))
))]
use std::arch::x86_64::{
    _mm256_clmulepi64_epi128, _mm256_gf2p8affine_epi64_epi8, _mm256_loadu_si256,
    _mm256_set1_epi64x, _mm256_set_epi64x, _mm256_shuffle_epi32, _mm256_slli_si256,
    _mm256_srli_epi64, _mm256_srli_si256, _mm256_storeu_si256, _mm256_unpacklo_epi64,
    _mm256_xor_si256,
};
#[cfg(not(any(
    all(target_feature = "avx2", target_feature = "vpclmulqdq"),
    all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "vpclmulqdq"
    )
)))]
use std::arch::x86_64::{
    _mm_clmulepi64_si128, _mm_gf2p8affine_epi64_epi8, _mm_loadu_si128, _mm_set1_epi64x,
    _mm_set_epi64x, _mm_shuffle_epi32, _mm_slli_si128, _mm_srli_epi64, _mm_srli_si128,
    _mm_storeu_si128, _mm_unpacklo_epi64, _mm_xor_si128,
};

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "vpclmulqdq"
))]
use std::arch::x86_64::{
    _mm512_bslli_epi128, _mm512_bsrli_epi128, _mm512_clmulepi64_epi128,
    _mm512_gf2p8affine_epi64_epi8, _mm512_loadu_si512, _mm512_set1_epi64, _mm512_set_epi64,
    _mm512_shuffle_epi32, _mm512_srli_epi64, _mm512_storeu_si512, _mm512_unpacklo_epi64,
    _mm512_xor_si512,
};

#[inline]
pub(in crate::binary) fn base_butterfly(top: &mut [F64], bot: &mut [F64], twiddle: F64) {
    if twiddle.to_raw() >> 61 == 0 {
        base_batches::<true>(top, bot, twiddle);
    } else {
        base_batches::<false>(top, bot, twiddle);
    }
}

#[inline]
fn base_batches<const SHORT: bool>(top: &mut [F64], bot: &mut [F64], twiddle: F64) {
    let mut top_chunks = top.chunks_exact_mut(16);
    let mut bot_chunks = bot.chunks_exact_mut(16);
    for (top, bot) in top_chunks.by_ref().zip(bot_chunks.by_ref()) {
        let mut raw_top: [u64; 16] = std::array::from_fn(|i| top[i].to_raw());
        let mut raw_bot: [u64; 16] = std::array::from_fn(|i| bot[i].to_raw());
        // SAFETY: the module requires PCLMULQDQ and GFNI; the kernel's cfg
        // selects a supported vector width. These disjoint initialized arrays
        // have lengths divisible by the number of words loaded per vector.
        unsafe { butterfly_words::<SHORT>(&mut raw_top, &mut raw_bot, twiddle.to_raw()) };
        for (value, raw) in top.iter_mut().zip(raw_top) {
            *value = F64::from_raw(raw);
        }
        for (value, raw) in bot.iter_mut().zip(raw_bot) {
            *value = F64::from_raw(raw);
        }
    }
    base_tail(
        top_chunks.into_remainder(),
        bot_chunks.into_remainder(),
        twiddle,
    );
}

#[inline]
pub(in crate::binary) fn extension_butterfly(top: &mut [F192], bot: &mut [F192], twiddle: F64) {
    if twiddle.to_raw() >> 61 == 0 {
        extension_batches::<true>(top, bot, twiddle);
    } else {
        #[cfg(any(
            all(target_feature = "avx2", target_feature = "vpclmulqdq"),
            all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "vpclmulqdq"
            )
        ))]
        extension_batches::<false>(top, bot, twiddle);
        #[cfg(not(any(
            all(target_feature = "avx2", target_feature = "vpclmulqdq"),
            all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "vpclmulqdq"
            )
        )))]
        extension_tail(top, bot, twiddle);
    }
}

#[inline]
fn extension_batches<const SHORT: bool>(top: &mut [F192], bot: &mut [F192], twiddle: F64) {
    let mut top_chunks = top.chunks_exact_mut(8);
    let mut bot_chunks = bot.chunks_exact_mut(8);
    for (top, bot) in top_chunks.by_ref().zip(bot_chunks.by_ref()) {
        let mut raw_top: [u64; 24] =
            std::array::from_fn(|i| top[i / 3].base_coefficient(i % 3).to_raw());
        let mut raw_bot: [u64; 24] =
            std::array::from_fn(|i| bot[i / 3].base_coefficient(i % 3).to_raw());
        // SAFETY: the selected instructions are enabled by the module and
        // kernel cfgs. These genuine u64 arrays contain all 24 coefficients;
        // their disjoint initialized storage permits unaligned vector access.
        unsafe { butterfly_words::<SHORT>(&mut raw_top, &mut raw_bot, twiddle.to_raw()) };
        for (i, value) in top.iter_mut().enumerate() {
            *value = F192::from_base_fn(|j| F64::from_raw(raw_top[3 * i + j]));
        }
        for (i, value) in bot.iter_mut().enumerate() {
            *value = F192::from_base_fn(|j| F64::from_raw(raw_bot[3 * i + j]));
        }
    }
    extension_tail(
        top_chunks.into_remainder(),
        bot_chunks.into_remainder(),
        twiddle,
    );
}

macro_rules! word_kernel {
    ($words:literal, $load:ident, $store:ident, $splat:ident, $matrix:expr,
     $mul:ident, $high:ident, $affine:ident, $take_carry:ident, $carry:ident, $overflow:ident,
     $xor:ident, $join:ident) => {
        /// Both slices contain complete vectors; SHORT requires twiddle degree <=60.
        #[inline]
        unsafe fn butterfly_words<const SHORT: bool>(
            top: &mut [u64],
            bot: &mut [u64],
            twiddle: u64,
        ) {
            // SAFETY: the wrappers provide disjoint, initialized slices with
            // lengths divisible by this vector width. Loads/stores are unaligned;
            // the enclosing cfgs enable every intrinsic and fix all lane indices.
            unsafe {
                let tw = $splat(twiddle as i64);
                for (top, bot) in top
                    .chunks_exact_mut($words)
                    .zip(bot.chunks_exact_mut($words))
                {
                    let bottom = $load(bot.as_ptr().cast());
                    let products = [$mul::<0>(bottom, tw), $mul::<0x11>(bottom, tw)];
                    let reduced = fold64::<_, SHORT>(
                        products,
                        |p| p.map(|v| $high::<0xee>(v)),
                        |p, low, high| p.map(|v| $affine::<0>(v, ($matrix)(low, high))),
                        |p| p.map(|v| $carry::<1>($take_carry::<8>(v))),
                        |p| p.map(|v| $overflow::<56>(v)),
                        |a, b| std::array::from_fn(|i| $xor(a[i], b[i])),
                    );
                    let product = $join(reduced[0], reduced[1]);
                    let next = $xor($load(top.as_ptr().cast()), product);
                    $store(top.as_mut_ptr().cast(), next);
                    $store(bot.as_mut_ptr().cast(), $xor(bottom, next));
                }
            }
        }
    };
}

#[cfg(not(any(
    all(target_feature = "avx2", target_feature = "vpclmulqdq"),
    all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "vpclmulqdq"
    )
)))]
word_kernel!(
    2,
    _mm_loadu_si128,
    _mm_storeu_si128,
    _mm_set1_epi64x,
    |low, high| _mm_set_epi64x(high, low),
    _mm_clmulepi64_si128,
    _mm_shuffle_epi32,
    _mm_gf2p8affine_epi64_epi8,
    _mm_srli_si128,
    _mm_slli_si128,
    _mm_srli_epi64,
    _mm_xor_si128,
    _mm_unpacklo_epi64
);

#[cfg(all(
    all(target_feature = "avx2", target_feature = "vpclmulqdq"),
    not(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "vpclmulqdq"
    ))
))]
word_kernel!(
    4,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    _mm256_set1_epi64x,
    |low, high| _mm256_set_epi64x(high, low, high, low),
    _mm256_clmulepi64_epi128,
    _mm256_shuffle_epi32,
    _mm256_gf2p8affine_epi64_epi8,
    _mm256_srli_si256,
    _mm256_slli_si256,
    _mm256_srli_epi64,
    _mm256_xor_si256,
    _mm256_unpacklo_epi64
);

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "vpclmulqdq"
))]
word_kernel!(
    8,
    _mm512_loadu_si512,
    _mm512_storeu_si512,
    _mm512_set1_epi64,
    |low, high| _mm512_set_epi64(high, low, high, low, high, low, high, low),
    _mm512_clmulepi64_epi128,
    _mm512_shuffle_epi32,
    _mm512_gf2p8affine_epi64_epi8,
    _mm512_bsrli_epi128,
    _mm512_bslli_epi128,
    _mm512_srli_epi64,
    _mm512_xor_si512,
    _mm512_unpacklo_epi64
);
