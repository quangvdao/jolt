// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Modified from leanVM crates/pcs/src/ntt/additive_ntt_f64.rs,
// crates/pcs/src/whir_ntt_ext.rs and crates/pcs/src/whir_induce.rs at
// revision 48a904208d682848dac0e18ef8b01ebfc40df9ad.
// Notices: crates/jolt-rv64i-pcs/THIRD_PARTY_NOTICES.md.

use super::{F192, F64};
use crate::ExtField;

impl F64 {
    /// Replaces `(a, b)` by `(a + b * twiddle, b + a + b * twiddle)`.
    /// Operates on the common prefix, checking both slice lengths before
    /// batching; any longer suffix is left untouched. No temporary allocation.
    #[inline]
    pub fn butterfly_assign(top: &mut [Self], bot: &mut [Self], twiddle: Self) {
        let count = top.len().min(bot.len());
        let (top, _) = top.split_at_mut(count);
        let (bot, _) = bot.split_at_mut(count);
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        super::butterfly_aarch64::base_butterfly(top, bot, twiddle);
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        base_tail(top, bot, twiddle);
    }
}

impl F192 {
    /// Replaces `(a, b)` by `(a + b * twiddle, b + a + b * twiddle)`, scaling
    /// each of the three base coefficients independently. Operates on the
    /// checked common prefix; any longer suffix is untouched. No allocation.
    #[inline]
    pub fn butterfly_assign(top: &mut [Self], bot: &mut [Self], twiddle: F64) {
        let count = top.len().min(bot.len());
        let (top, _) = top.split_at_mut(count);
        let (bot, _) = bot.split_at_mut(count);
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        super::butterfly_aarch64::extension_butterfly(top, bot, twiddle);
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        extension_tail(top, bot, twiddle);
    }
}

#[inline]
pub(super) fn base_tail(top: &mut [F64], bot: &mut [F64], twiddle: F64) {
    let (top_chunks, top_tail) = top.as_chunks_mut::<4>();
    let (bot_chunks, bot_tail) = bot.as_chunks_mut::<4>();
    for (top, bot) in top_chunks.iter_mut().zip(bot_chunks) {
        let products = bot.map(|b| b * twiddle);
        for ((top, bot), product) in top.iter_mut().zip(bot).zip(products) {
            let next = *top + product;
            *bot += next;
            *top = next;
        }
    }
    for (top, bot) in top_tail.iter_mut().zip(bot_tail) {
        let next = *top + *bot * twiddle;
        *bot += next;
        *top = next;
    }
}

#[inline]
pub(super) fn extension_tail(top: &mut [F192], bot: &mut [F192], twiddle: F64) {
    for (top, bot) in top.iter_mut().zip(bot) {
        let next = *top + bot.mul_base(twiddle);
        *bot += next;
        *top = next;
    }
}

#[cfg(test)]
mod tests {
    use super::{F192, F64};
    use crate::binary::portable;
    use crate::ExtField;

    #[test]
    fn batched_butterflies_match_portable_at_every_width() {
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
                let raw_top: Vec<_> = (0..width)
                    .map(|i| (0x9e37_79b9_7f4a_7c15u64 ^ i as u64).rotate_left(i as u32))
                    .collect();
                let raw_bot: Vec<_> = (0..width)
                    .map(|i| (0xd1b5_4a32_d192_ed03u64 ^ i as u64).rotate_right(i as u32))
                    .collect();
                let expected_top: Vec<_> = raw_top
                    .iter()
                    .zip(&raw_bot)
                    .map(|(&a, &b)| a ^ portable::multiply64(b, raw_twiddle))
                    .collect();
                let expected_bot: Vec<_> = expected_top
                    .iter()
                    .zip(&raw_bot)
                    .map(|(&a, &b)| a ^ b)
                    .collect();
                let mut top: Vec<_> = raw_top.iter().copied().map(F64::from_raw).collect();
                let mut bot: Vec<_> = raw_bot.iter().copied().map(F64::from_raw).collect();
                F64::butterfly_assign(&mut top, &mut bot, twiddle);
                assert_eq!(
                    top.iter().map(|v| v.to_raw()).collect::<Vec<_>>(),
                    expected_top,
                    "K width={width}"
                );
                assert_eq!(
                    bot.iter().map(|v| v.to_raw()).collect::<Vec<_>>(),
                    expected_bot,
                    "K width={width}"
                );
                let a: Vec<_> = raw_top
                    .iter()
                    .map(|v| std::array::from_fn::<_, 3, _>(|j| v.rotate_left(17 * j as u32)))
                    .collect();
                let b: Vec<_> = raw_bot
                    .iter()
                    .map(|v| std::array::from_fn::<_, 3, _>(|j| v.rotate_right(23 * j as u32)))
                    .collect();
                let mut top: Vec<_> = a
                    .iter()
                    .map(|v| F192::from_base_fn(|j| F64::from_raw(v[j])))
                    .collect();
                let mut bot: Vec<_> = b
                    .iter()
                    .map(|v| F192::from_base_fn(|j| F64::from_raw(v[j])))
                    .collect();
                F192::butterfly_assign(&mut top, &mut bot, twiddle);
                for (((top, bot), a), b) in top.iter().zip(&bot).zip(&a).zip(&b) {
                    for j in 0..3 {
                        let next = a[j] ^ portable::multiply64(b[j], raw_twiddle);
                        assert_eq!(top.base_coefficient(j).to_raw(), next, "E width={width}");
                        assert_eq!(
                            bot.base_coefficient(j).to_raw(),
                            next ^ b[j],
                            "E width={width}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn butterfly_common_prefix_preserves_offset_guards_and_suffixes() {
        let a = F64::from_raw(3);
        let b = F64::from_raw(5);
        let tw = F64::from_raw(7);
        for (top_len, bot_len) in [(0, 9), (9, 0), (7, 11), (11, 7), (17, 20), (20, 17)] {
            let mut top = vec![a; top_len + 2];
            let mut bot = vec![b; bot_len + 2];
            F64::butterfly_assign(&mut top[1..=top_len], &mut bot[1..=bot_len], tw);
            let count = top_len.min(bot_len);
            for (i, v) in top.iter().enumerate() {
                assert_eq!(
                    v.to_raw(),
                    if (1..=count).contains(&i) {
                        3 ^ portable::multiply64(5, 7)
                    } else {
                        3
                    }
                );
            }
            for (i, v) in bot.iter().enumerate() {
                assert_eq!(
                    v.to_raw(),
                    if (1..=count).contains(&i) {
                        5 ^ 3 ^ portable::multiply64(5, 7)
                    } else {
                        5
                    }
                );
            }
            let a = F192::from_base_fn(|i| F64::from_raw(3 + i as u64));
            let b = F192::from_base_fn(|i| F64::from_raw(5 + i as u64));
            let mut top = vec![a; top_len + 2];
            let mut bot = vec![b; bot_len + 2];
            F192::butterfly_assign(&mut top[1..=top_len], &mut bot[1..=bot_len], tw);
            for i in 0..top.len().max(bot.len()) {
                for j in 0..3 {
                    let a = 3 + j as u64;
                    let b = 5 + j as u64;
                    let p = portable::multiply64(b, 7);
                    if let Some(v) = top.get(i) {
                        assert_eq!(
                            v.base_coefficient(j).to_raw(),
                            if (1..=count).contains(&i) { a ^ p } else { a }
                        );
                    }
                    if let Some(v) = bot.get(i) {
                        assert_eq!(
                            v.base_coefficient(j).to_raw(),
                            if (1..=count).contains(&i) {
                                a ^ b ^ p
                            } else {
                                b
                            }
                        );
                    }
                }
            }
        }
    }
}
