// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Modified from leanVM crates/pcs/src/ntt/additive_ntt_f64.rs and
// crates/pcs/src/whir_ntt_ext.rs at revision
// 48a904208d682848dac0e18ef8b01ebfc40df9ad.
// Notices: crates/jolt-rv64i-pcs/THIRD_PARTY_NOTICES.md.

//! Batched coefficient butterflies without assumptions about field representation.

#![expect(
    unsafe_code,
    reason = "this feature-gated architecture module confines NEON intrinsics"
)]

use super::Encoder;
use jolt_field::{ExtField, F192, F64};
#[cfg(not(target_feature = "sha3"))]
use std::arch::aarch64::vtrn1q_u64;
use std::arch::aarch64::{
    uint64x2_t, vdupq_n_u64, veorq_u64, vgetq_lane_u64, vld1q_u64, vmull_high_p64, vmull_p64,
    vreinterpretq_p64_u64, vreinterpretq_u64_p128, vst1q_u64,
};

#[inline]
pub(super) fn base_butterfly(top: &mut [F64], bot: &mut [F64], twiddle: F64) {
    if twiddle.to_raw() >> 61 == 0 {
        base_batches::<true>(top, bot, twiddle);
    } else {
        base_batches::<false>(top, bot, twiddle);
    }
}

#[inline]
fn base_batches<const SHORT: bool>(top: &mut [F64], bot: &mut [F64], twiddle: F64) {
    let count = top.len().min(bot.len());
    let (top, _) = top.split_at_mut(count);
    let (bot, _) = bot.split_at_mut(count);
    let mut top_chunks = top.chunks_exact_mut(8);
    let mut bot_chunks = bot.chunks_exact_mut(8);
    for (top, bot) in top_chunks.by_ref().zip(bot_chunks.by_ref()) {
        let mut raw_top: [u64; 8] = std::array::from_fn(|i| top[i].to_raw());
        let mut raw_bot: [u64; 8] = std::array::from_fn(|i| bot[i].to_raw());
        // SAFETY: the enclosing module cfg enables aes; both genuine u64 arrays
        // contain the eight coefficients accessed by the kernel.
        unsafe { butterfly_pairs::<4, SHORT>(&mut raw_top, &mut raw_bot, twiddle.to_raw()) };
        for (value, raw) in top.iter_mut().zip(raw_top) {
            *value = F64::from_raw(raw);
        }
        for (value, raw) in bot.iter_mut().zip(raw_bot) {
            *value = F64::from_raw(raw);
        }
    }
    Encoder::safe_butterfly(
        top_chunks.into_remainder(),
        bot_chunks.into_remainder(),
        twiddle,
    );
}

#[inline]
pub(super) fn extension_butterfly(top: &mut [F192], bot: &mut [F192], twiddle: F64) {
    if twiddle.to_raw() >> 61 == 0 {
        extension_batches::<true>(top, bot, twiddle);
    } else {
        extension_batches::<false>(top, bot, twiddle);
    }
}

#[inline]
fn extension_batches<const SHORT: bool>(top: &mut [F192], bot: &mut [F192], twiddle: F64) {
    let count = top.len().min(bot.len());
    let (top, _) = top.split_at_mut(count);
    let (bot, _) = bot.split_at_mut(count);
    let mut top_chunks = top.chunks_exact_mut(4);
    let mut bot_chunks = bot.chunks_exact_mut(4);
    for (top, bot) in top_chunks.by_ref().zip(bot_chunks.by_ref()) {
        let mut raw_top: [u64; 12] =
            std::array::from_fn(|i| top[i / 3].base_coefficient(i % 3).to_raw());
        let mut raw_bot: [u64; 12] =
            std::array::from_fn(|i| bot[i / 3].base_coefficient(i % 3).to_raw());
        // SAFETY: the enclosing module cfg enables aes; these are genuine u64
        // arrays contain all twelve coefficients accessed by the kernel.
        unsafe { butterfly_pairs::<6, SHORT>(&mut raw_top, &mut raw_bot, twiddle.to_raw()) };
        for (i, value) in top.iter_mut().enumerate() {
            *value = F192::from_base_fn(|j| F64::from_raw(raw_top[3 * i + j]));
        }
        for (i, value) in bot.iter_mut().enumerate() {
            *value = F192::from_base_fn(|j| F64::from_raw(raw_bot[3 * i + j]));
        }
    }
    Encoder::safe_butterfly(
        top_chunks.into_remainder(),
        bot_chunks.into_remainder(),
        twiddle,
    );
}

/// The caller supplies at least `2 * PAIRS` words in each slice and enables aes.
#[inline]
unsafe fn butterfly_pairs<const PAIRS: usize, const SHORT: bool>(
    top: &mut [u64],
    bot: &mut [u64],
    twiddle: u64,
) {
    // SAFETY: the callers provide valid, disjoint u64 slices of the required
    // lengths; NEON loads/stores permit their ordinary u64 alignment. The target
    // feature covers polynomial multiplication and all lane indices are fixed.
    unsafe {
        let tw = vdupq_n_u64(twiddle);
        let modulus = vdupq_n_u64(0x1b);
        let bottom: [uint64x2_t; PAIRS] =
            std::array::from_fn(|i| vld1q_u64(bot.as_ptr().add(2 * i)));
        let products = bottom.map(|value| {
            (
                vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(value), twiddle)),
                vreinterpretq_u64_p128(vmull_high_p64(
                    vreinterpretq_p64_u64(value),
                    vreinterpretq_p64_u64(tw),
                )),
            )
        });
        let reduced = products.map(|(low, high)| reduce_pair::<SHORT>(low, high, modulus));
        for i in 0..PAIRS {
            let original_top = vld1q_u64(top.as_ptr().add(2 * i));
            let next = veorq_u64(original_top, reduced[i]);
            vst1q_u64(top.as_mut_ptr().add(2 * i), next);
            vst1q_u64(bot.as_mut_ptr().add(2 * i), veorq_u64(bottom[i], next));
        }
    }
}

/// The caller enables aes, as guaranteed by the enclosing module cfg.
#[inline(always)]
unsafe fn reduce_pair<const SHORT: bool>(
    low: uint64x2_t,
    high: uint64x2_t,
    modulus: uint64x2_t,
) -> uint64x2_t {
    // x^64 = 0x1b in K. Fold each product's high half by 0x1b, then fold that
    // product's at-most-four-bit overflow once more; the second fold fits in
    // eight bits. This is the same two-fold law as jolt-field's reduce_word64.
    let first_low = vreinterpretq_u64_p128(vmull_high_p64(
        vreinterpretq_p64_u64(low),
        vreinterpretq_p64_u64(modulus),
    ));
    let first_high = vreinterpretq_u64_p128(vmull_high_p64(
        vreinterpretq_p64_u64(high),
        vreinterpretq_p64_u64(modulus),
    ));
    // SHORT is selected once by the wrapper when the twiddle degree is <=60.
    // A degree-63 coefficient then gives degree <=123, so the product's high
    // half has degree <=59 and its first fold by 0x1b already fits in 64 bits.
    if SHORT {
        #[cfg(target_feature = "sha3")]
        {
            let result;
            // SAFETY: the enclosing cfg enables aes. Only vector registers
            // are read/written; the asm has no memory effects.
            unsafe {
                std::arch::asm!(
                    "eor {low:v}.16b, {low:v}.16b, {first_low:v}.16b",
                    "eor {high:v}.16b, {high:v}.16b, {first_high:v}.16b",
                    "trn1 {low:v}.2d, {low:v}.2d, {high:v}.2d",
                    low = inout(vreg) low => result,
                    high = inout(vreg) high => _,
                    first_low = in(vreg) first_low,
                    first_high = in(vreg) first_high,
                    options(pure, nomem, nostack),
                );
            }
            return result;
        }
        #[cfg(not(target_feature = "sha3"))]
        {
            return vtrn1q_u64(veorq_u64(low, first_low), veorq_u64(high, first_high));
        }
    }
    let second_low = vreinterpretq_u64_p128(vmull_high_p64(
        vreinterpretq_p64_u64(first_low),
        vreinterpretq_p64_u64(modulus),
    ));
    let second_high = vreinterpretq_u64_p128(vmull_high_p64(
        vreinterpretq_p64_u64(first_high),
        vreinterpretq_p64_u64(modulus),
    ));
    #[cfg(target_feature = "sha3")]
    {
        let result;
        // SAFETY: the enclosing cfg enables aes, and this branch requires sha3.
        // Only vector registers are read/written; the asm has no memory effects.
        unsafe {
            std::arch::asm!(
                "eor3 {low:v}.16b, {low:v}.16b, {first_low:v}.16b, {second_low:v}.16b",
                "eor3 {high:v}.16b, {high:v}.16b, {first_high:v}.16b, {second_high:v}.16b",
                "trn1 {low:v}.2d, {low:v}.2d, {high:v}.2d",
                low = inout(vreg) low => result,
                high = inout(vreg) high => _,
                first_low = in(vreg) first_low,
                first_high = in(vreg) first_high,
                second_low = in(vreg) second_low,
                second_high = in(vreg) second_high,
                options(pure, nomem, nostack),
            );
        }
        result
    }
    #[cfg(not(target_feature = "sha3"))]
    {
        vtrn1q_u64(
            veorq_u64(veorq_u64(low, first_low), second_low),
            veorq_u64(veorq_u64(high, first_high), second_high),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{base_butterfly, extension_butterfly, Encoder};
    use jolt_field::{ExtField, F192, F64};

    #[test]
    fn batched_butterflies_match_safe_path_at_every_width() {
        for width in (0..=512).chain([1024, 2048, 4096]) {
            for raw_twiddle in [
                0,
                1,
                1 << 60,
                (1 << 61) - 1,
                1 << 61,
                u64::MAX,
                0xa53c_2f17_8e91_6bd4,
            ] {
                let twiddle = F64::from_raw(raw_twiddle);
                let original_top: Vec<_> = (0..width)
                    .map(|i| {
                        F64::from_raw((0x9e37_79b9_7f4a_7c15u64 ^ i as u64).rotate_left(i as u32))
                    })
                    .collect();
                let original_bot: Vec<_> = (0..width)
                    .map(|i| {
                        F64::from_raw((0xd1b5_4a32_d192_ed03u64 ^ i as u64).rotate_right(i as u32))
                    })
                    .collect();
                let mut top = original_top.clone();
                let mut bot = original_bot.clone();
                base_butterfly(&mut top, &mut bot, twiddle);
                let mut expected_top = original_top.clone();
                let mut expected_bot = original_bot.clone();
                Encoder::safe_butterfly(&mut expected_top, &mut expected_bot, twiddle);
                assert_eq!(top, expected_top, "K width={width}");
                assert_eq!(bot, expected_bot, "K width={width}");
                let original_top: Vec<_> = original_top
                    .iter()
                    .map(|&v| {
                        F192::from_base_fn(|j| F64::from_raw(v.to_raw().rotate_left(17 * j as u32)))
                    })
                    .collect();
                let original_bot: Vec<_> = original_bot
                    .iter()
                    .map(|&v| {
                        F192::from_base_fn(|j| {
                            F64::from_raw(v.to_raw().rotate_right(23 * j as u32))
                        })
                    })
                    .collect();
                let mut top = original_top.clone();
                let mut bot = original_bot.clone();
                extension_butterfly(&mut top, &mut bot, twiddle);
                let mut expected_top = original_top.clone();
                let mut expected_bot = original_bot.clone();
                Encoder::safe_butterfly(&mut expected_top, &mut expected_bot, twiddle);
                assert_eq!(top, expected_top, "E width={width}");
                assert_eq!(bot, expected_bot, "E width={width}");
            }
        }
    }
}
