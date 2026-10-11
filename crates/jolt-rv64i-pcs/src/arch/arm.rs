// Modified from leanVM crates/primitives/src/hash.rs at
// 48a904208d682848dac0e18ef8b01ebfc40df9ad.
// Notices: crates/jolt-rv64i-pcs/THIRD_PARTY_NOTICES.md.

use super::Lanes32;
use core::arch::aarch64::*;

#[derive(Clone, Copy)]
pub(super) struct Neon(uint32x4_t);

impl Lanes32 for Neon {
    const WIDTH: usize = 4;
    // Four interleaved groups beat one and two in the tree-phase benchmark.
    const GROUPS: usize = 4;

    #[inline(always)]
    unsafe fn load(p: *const u32) -> Self {
        // SAFETY: aarch64 NEON cfg supplies ISA; caller supplies 4 readable u32.
        Self(unsafe { vld1q_u32(p) })
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u32) {
        // SAFETY: aarch64 NEON cfg supplies ISA; caller supplies 4 writable u32.
        unsafe { vst1q_u32(p, self.0) }
    }
    #[inline(always)]
    fn splat(x: u32) -> Self {
        // SAFETY: the module cfg guarantees aarch64 NEON.
        Self(unsafe { vdupq_n_u32(x) })
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        // SAFETY: the module cfg guarantees aarch64 NEON.
        Self(unsafe { vaddq_u32(self.0, o.0) })
    }
    #[inline(always)]
    fn xor(self, o: Self) -> Self {
        // SAFETY: the module cfg guarantees aarch64 NEON.
        Self(unsafe { veorq_u32(self.0, o.0) })
    }
    #[inline(always)]
    fn rotr<const N: u32>(self) -> Self {
        Self(rot4::<N>(self.0))
    }

    #[inline(always)]
    unsafe fn transpose(inputs: &[*const u8], off: usize, buf: &mut [u32]) {
        debug_assert_eq!(inputs.len(), 4);
        debug_assert!(buf.len() >= 64);
        // SAFETY: cfg guarantees NEON; each input spans 64 bytes at off,
        // and buf has 64 writable u32; byte loads accept unaligned inputs.
        unsafe {
            let r: [uint32x4_t; 16] = std::array::from_fn(|i| {
                vreinterpretq_u32_u8(vld1q_u8(inputs[i / 4].add(off + 16 * (i % 4))))
            });
            for q in 0..4 {
                let [a, b, c, d] = [r[q], r[4 + q], r[8 + q], r[12 + q]];
                for (j, o) in transpose4(a, b, c, d).into_iter().enumerate() {
                    vst1q_u32(buf.as_mut_ptr().add(16 * q + 4 * j), o);
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn store_digests(h: &[Self; 8], out: *mut u8) {
        // SAFETY: cfg guarantees NEON; caller supplies 4 * 32 writable bytes.
        unsafe {
            for half in 0..2 {
                let [a, b, c, d] = [
                    h[4 * half],
                    h[4 * half + 1],
                    h[4 * half + 2],
                    h[4 * half + 3],
                ];
                for (lane, o) in transpose4(a.0, b.0, c.0, d.0).into_iter().enumerate() {
                    vst1q_u8(out.add(lane * 32 + 16 * half), vreinterpretq_u8_u32(o));
                }
            }
        }
    }
}

#[inline(always)]
fn rot4<const N: u32>(v: uint32x4_t) -> uint32x4_t {
    // SAFETY: cfg guarantees NEON; ROT8 holds 16 readable bytes.
    unsafe {
        match N {
            16 => vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(v))),
            8 => {
                static ROT8: [u8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];
                vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(v), vld1q_u8(ROT8.as_ptr())))
            }
            12 => rot_sri::<12, 20>(v),
            7 => rot_sri::<7, 25>(v),
            _ => {
                let count = vdupq_n_s32((N % 32) as i32);
                vorrq_u32(
                    vshlq_u32(v, vnegq_s32(count)),
                    vshlq_u32(v, vsubq_s32(vdupq_n_s32(32), count)),
                )
            }
        }
    }
}

#[inline(always)]
fn rot_sri<const N: u32, const SHL: i32>(v: uint32x4_t) -> uint32x4_t {
    // SAFETY: cfg guarantees NEON; the assembly uses only vector registers
    // and has no memory or stack effects; N and SHL are fixed rotation amounts.
    unsafe {
        let mut out = vshlq_n_u32::<SHL>(v);
        std::arch::asm!(
            "sri {out:v}.4s, {v:v}.4s, #{n}",
            out = inout(vreg) out,
            v = in(vreg) v,
            n = const N,
            options(pure, nomem, nostack)
        );
        out
    }
}

#[inline(always)]
fn transpose4(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t, d: uint32x4_t) -> [uint32x4_t; 4] {
    // SAFETY: the module cfg guarantees aarch64 NEON.
    unsafe {
        let (ab0, ab1) = (vtrn1q_u32(a, b), vtrn2q_u32(a, b));
        let (cd0, cd1) = (vtrn1q_u32(c, d), vtrn2q_u32(c, d));
        let pair = |x, y| {
            (
                vreinterpretq_u32_u64(vtrn1q_u64(
                    vreinterpretq_u64_u32(x),
                    vreinterpretq_u64_u32(y),
                )),
                vreinterpretq_u32_u64(vtrn2q_u64(
                    vreinterpretq_u64_u32(x),
                    vreinterpretq_u64_u32(y),
                )),
            )
        };
        let ((o0, o2), (o1, o3)) = (pair(ab0, cd0), pair(ab1, cd1));
        [o0, o1, o2, o3]
    }
}
