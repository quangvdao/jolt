//! Eight independent BLAKE2s states per AVX2 vector.
//!
//! Modified from `leanVM`'s `crates/primitives/src/hash.rs` at revision
//! `48a904208d682848dac0e18ef8b01ebfc40df9ad`.
//! Notices: `crates/jolt-rv64i-pcs/THIRD_PARTY_NOTICES.md`.

use core::arch::x86_64;
use core::arch::x86_64::__m256i;

use super::Lanes32;

#[derive(Clone, Copy)]
pub(super) struct Avx2(__m256i);

impl Lanes32 for Avx2 {
    const WIDTH: usize = 8;

    #[inline(always)]
    unsafe fn load(p: *const u32) -> Self {
        // SAFETY: the module cfg guarantees x86_64 AVX2; the caller supplies
        // eight readable u32 values. The intrinsic permits unaligned access.
        Self(unsafe { x86_64::_mm256_loadu_si256(p.cast()) })
    }

    #[inline(always)]
    unsafe fn store(self, p: *mut u32) {
        // SAFETY: the module cfg guarantees x86_64 AVX2; the caller supplies
        // eight writable u32 values. The intrinsic permits unaligned access.
        unsafe { x86_64::_mm256_storeu_si256(p.cast(), self.0) }
    }

    #[inline(always)]
    fn splat(x: u32) -> Self {
        // SAFETY: the module cfg guarantees x86_64 AVX2; this instruction has
        // no memory operand or additional precondition.
        Self(unsafe { x86_64::_mm256_set1_epi32(x as i32) })
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: the module cfg guarantees x86_64 AVX2; both operands are
        // initialized vectors and this instruction accesses no memory.
        Self(unsafe { x86_64::_mm256_add_epi32(self.0, other.0) })
    }

    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        // SAFETY: the module cfg guarantees x86_64 AVX2; both operands are
        // initialized vectors and this instruction accesses no memory.
        Self(unsafe { x86_64::_mm256_xor_si256(self.0, other.0) })
    }

    #[inline(always)]
    fn rotr<const N: u32>(self) -> Self {
        const ROT16: [i8; 16] = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13];
        const ROT8: [i8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];
        // SAFETY: the module cfg guarantees x86_64 AVX2 (and hence SSE2).
        // Each shuffle mask owns the 16 readable bytes loaded unaligned;
        // all other instructions operate solely on initialized vectors.
        unsafe {
            let shuffle = |mask: [i8; 16]| {
                let half = x86_64::_mm_loadu_si128(mask.as_ptr().cast());
                Self(x86_64::_mm256_shuffle_epi8(
                    self.0,
                    x86_64::_mm256_set_m128i(half, half),
                ))
            };
            match N {
                16 => shuffle(ROT16),
                8 => shuffle(ROT8),
                12 => Self(x86_64::_mm256_or_si256(
                    x86_64::_mm256_srli_epi32::<12>(self.0),
                    x86_64::_mm256_slli_epi32::<20>(self.0),
                )),
                7 => Self(x86_64::_mm256_or_si256(
                    x86_64::_mm256_srli_epi32::<7>(self.0),
                    x86_64::_mm256_slli_epi32::<25>(self.0),
                )),
                _ => {
                    let right = x86_64::_mm256_set1_epi32((N % 32) as i32);
                    let left = x86_64::_mm256_set1_epi32(((32 - N % 32) % 32) as i32);
                    Self(x86_64::_mm256_or_si256(
                        x86_64::_mm256_srlv_epi32(self.0, right),
                        x86_64::_mm256_sllv_epi32(self.0, left),
                    ))
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn transpose(inputs: &[*const u8], off: usize, buf: &mut [u32]) {
        debug_assert_eq!(inputs.len(), Self::WIDTH);
        debug_assert!(buf.len() >= 16 * Self::WIDTH);
        // SAFETY: the module cfg guarantees x86_64 AVX2. The caller supplies
        // eight inputs, each readable for 64 bytes at off, and at least 128
        // writable words in buf. Each half loads 32 of those bytes; each store
        // writes eight words at an index in 0..=120. Accesses are unaligned.
        unsafe {
            for half in 0..2 {
                let rows: [__m256i; 8] = std::array::from_fn(|lane| {
                    x86_64::_mm256_loadu_si256(inputs[lane].add(off + 32 * half).cast())
                });
                let mut pairs = [x86_64::_mm256_setzero_si256(); 8];
                for pair in 0..4 {
                    pairs[2 * pair] =
                        x86_64::_mm256_unpacklo_epi32(rows[2 * pair], rows[2 * pair + 1]);
                    pairs[2 * pair + 1] =
                        x86_64::_mm256_unpackhi_epi32(rows[2 * pair], rows[2 * pair + 1]);
                }
                let quads: [__m256i; 8] = [
                    x86_64::_mm256_unpacklo_epi64(pairs[0], pairs[2]),
                    x86_64::_mm256_unpackhi_epi64(pairs[0], pairs[2]),
                    x86_64::_mm256_unpacklo_epi64(pairs[1], pairs[3]),
                    x86_64::_mm256_unpackhi_epi64(pairs[1], pairs[3]),
                    x86_64::_mm256_unpacklo_epi64(pairs[4], pairs[6]),
                    x86_64::_mm256_unpackhi_epi64(pairs[4], pairs[6]),
                    x86_64::_mm256_unpacklo_epi64(pairs[5], pairs[7]),
                    x86_64::_mm256_unpackhi_epi64(pairs[5], pairs[7]),
                ];
                for word in 0..4 {
                    let index = 8 * half + word;
                    x86_64::_mm256_storeu_si256(
                        buf.as_mut_ptr().add(index * 8).cast(),
                        x86_64::_mm256_permute2x128_si256::<0x20>(quads[word], quads[word + 4]),
                    );
                    x86_64::_mm256_storeu_si256(
                        buf.as_mut_ptr().add((index + 4) * 8).cast(),
                        x86_64::_mm256_permute2x128_si256::<0x31>(quads[word], quads[word + 4]),
                    );
                }
            }
        }
    }
}
