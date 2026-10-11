// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Modified from leanVM crates/pcs/src/ntt/additive_ntt_f64.rs and
// crates/pcs/src/whir_ntt_ext.rs at revision
// 48a904208d682848dac0e18ef8b01ebfc40df9ad.
// Notices: crates/jolt-rv64i-pcs/THIRD_PARTY_NOTICES.md.

//! Paired coefficient butterflies over initialized word arrays.

use super::arch::fold64;
use super::butterfly::{base_tail, extension_tail};
use super::reduction::MODULUS64;
use super::{F192, F64};
use crate::ExtField;
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
    base_tail(
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
        // arrays containing all twelve coefficients accessed by the kernel.
        unsafe { butterfly_pairs::<6, SHORT>(&mut raw_top, &mut raw_bot, twiddle.to_raw()) };
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
        let modulus = vdupq_n_u64(MODULUS64);
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
        let reduced = products.map(|(low, high)| {
            fold64::<_, _, SHORT>(
                [low, high],
                |pair| {
                    pair.map(|p| {
                        vreinterpretq_u64_p128(vmull_high_p64(
                            vreinterpretq_p64_u64(p),
                            vreinterpretq_p64_u64(modulus),
                        ))
                    })
                },
                finish_short,
                finish_full,
            )
        });
        for i in 0..PAIRS {
            let original_top = vld1q_u64(top.as_ptr().add(2 * i));
            let next = veorq_u64(original_top, reduced[i]);
            vst1q_u64(top.as_mut_ptr().add(2 * i), next);
            vst1q_u64(bot.as_mut_ptr().add(2 * i), veorq_u64(bottom[i], next));
        }
    }
}

#[inline(always)]
fn finish_short(
    [low, high]: [uint64x2_t; 2],
    [first_low, first_high]: [uint64x2_t; 2],
) -> uint64x2_t {
    // SAFETY: this module's cfg enables aes. Only vector registers are touched.
    unsafe {
        #[cfg(target_feature = "sha3")]
        {
            let result;
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
            result
        }
        #[cfg(not(target_feature = "sha3"))]
        {
            vtrn1q_u64(veorq_u64(low, first_low), veorq_u64(high, first_high))
        }
    }
}

#[inline(always)]
fn finish_full(
    [low, high]: [uint64x2_t; 2],
    [first_low, first_high]: [uint64x2_t; 2],
    [second_low, second_high]: [uint64x2_t; 2],
) -> uint64x2_t {
    // SAFETY: this module's cfg enables aes; the assembly branch also requires
    // sha3. All operations are register-only, with no memory effects.
    unsafe {
        #[cfg(target_feature = "sha3")]
        {
            let result;
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
}
