// Modified from leanVM crates/primitives/src/hash.rs at
// 48a904208d682848dac0e18ef8b01ebfc40df9ad.
// Notices: crates/jolt-rv64i-pcs/THIRD_PARTY_NOTICES.md.
//! Lane-transposed BLAKE2s; byte layouts and counters match RFC 7693.

use jolt_rv64i_verifier::whir::{
    error::{checked_product, WhirError, WhirPart},
    merkle::{hash_leaf, Digest},
};

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_endian = "little"
))]
mod arm;
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_endian = "little"
))]
use arm::Neon;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_endian = "little"
))]
mod x86;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_endian = "little"
))]
use x86::Avx2;

#[cfg(any(
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ),
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_endian = "little"
    )
))]
const IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

#[cfg(any(
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ),
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_endian = "little"
    )
))]
mod kernel {
    use super::IV;

    pub(super) trait Lanes32: Copy {
        const WIDTH: usize;
        const GROUPS: usize = 1;
        // Backend cfg must guarantee its ISA for every operation. All raw
        // pointers passed to load/store span WIDTH words; transpose inputs
        // span 64 bytes at off, and buf spans 16 * WIDTH words.
        unsafe fn load(p: *const u32) -> Self;
        unsafe fn store(self, p: *mut u32);
        fn splat(x: u32) -> Self;
        fn add(self, other: Self) -> Self;
        fn xor(self, other: Self) -> Self;
        fn rotr<const N: u32>(self) -> Self;
        unsafe fn transpose(inputs: &[*const u8], off: usize, buf: &mut [u32]);

        #[inline(always)]
        unsafe fn store_digests(h: &[Self; 8], out: *mut u8) {
            let mut words = [0u32; 8 * 8];
            for (i, value) in h.iter().enumerate() {
                // SAFETY: backend cfg supplies ISA; WIDTH <= 8 and words holds 8 * WIDTH words.
                unsafe { value.store(words.as_mut_ptr().add(i * Self::WIDTH)) };
            }
            for lane in 0..Self::WIDTH {
                for i in 0..8 {
                    let bytes = words[i * Self::WIDTH + lane].to_le_bytes();
                    // SAFETY: backend cfg supplies ISA; caller supplies WIDTH * 32 writable bytes.
                    unsafe {
                        out.add(lane * 32 + i * 4)
                            .copy_from_nonoverlapping(bytes.as_ptr(), 4);
                    };
                }
            }
        }
    }

    macro_rules! g {
        ($v:ident, $m:ident, $a:expr, $b:expr, $c:expr, $d:expr, $x:expr, $y:expr) => {{
            $v[$a] = $v[$a].add($v[$b]).add(S::load($m.add($x * S::WIDTH)));
            $v[$d] = $v[$d].xor($v[$a]).rotr::<16>();
            $v[$c] = $v[$c].add($v[$d]);
            $v[$b] = $v[$b].xor($v[$c]).rotr::<12>();
            $v[$a] = $v[$a].add($v[$b]).add(S::load($m.add($y * S::WIDTH)));
            $v[$d] = $v[$d].xor($v[$a]).rotr::<8>();
            $v[$c] = $v[$c].add($v[$d]);
            $v[$b] = $v[$b].xor($v[$c]).rotr::<7>();
        }};
    }

    macro_rules! round {
        ($v:ident, $m:ident, [$s0:expr, $s1:expr, $s2:expr, $s3:expr, $s4:expr, $s5:expr, $s6:expr, $s7:expr,
      $s8:expr, $s9:expr, $s10:expr, $s11:expr, $s12:expr, $s13:expr, $s14:expr, $s15:expr]) => {{
            g!($v, $m, 0, 4, 8, 12, $s0, $s1);
            g!($v, $m, 1, 5, 9, 13, $s2, $s3);
            g!($v, $m, 2, 6, 10, 14, $s4, $s5);
            g!($v, $m, 3, 7, 11, 15, $s6, $s7);
            g!($v, $m, 0, 5, 10, 15, $s8, $s9);
            g!($v, $m, 1, 6, 11, 12, $s10, $s11);
            g!($v, $m, 2, 7, 8, 13, $s12, $s13);
            g!($v, $m, 3, 4, 9, 14, $s14, $s15);
        }};
    }

    macro_rules! round_fns {
    ($($name:ident [$($s:expr),*],)*) => {
        $(
            #[inline(never)]
            unsafe fn $name<S: Lanes32>(v: &mut [S; 16], m: *const u32) {
                // SAFETY: the compiled backend cfg supplies its ISA; m has 16 * WIDTH readable words.
                unsafe { round!(v, m, [$($s),*]) }
            }
        )*
    };
}

    round_fns! {
        round_0 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        round_1 [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
        round_2 [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
        round_3 [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
        round_4 [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
        round_5 [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
        round_6 [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
        round_7 [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        round_8 [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
        round_9 [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    }

    macro_rules! rounds {
        ($v:ident, $m:ident) => {{
            round!(
                $v,
                $m,
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            );
            round!(
                $v,
                $m,
                [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3]
            );
            round!(
                $v,
                $m,
                [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4]
            );
            round!(
                $v,
                $m,
                [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8]
            );
            round!(
                $v,
                $m,
                [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13]
            );
            round!(
                $v,
                $m,
                [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9]
            );
            round!(
                $v,
                $m,
                [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11]
            );
            round!(
                $v,
                $m,
                [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10]
            );
            round!(
                $v,
                $m,
                [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5]
            );
            round!(
                $v,
                $m,
                [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0]
            );
        }};
    }

    #[inline(always)]
    unsafe fn compress_lanes<S: Lanes32>(h: &mut [S; 8], m: *const u32, t: u64, last: bool) {
        let mut v = [
            h[0],
            h[1],
            h[2],
            h[3],
            h[4],
            h[5],
            h[6],
            h[7],
            S::splat(IV[0]),
            S::splat(IV[1]),
            S::splat(IV[2]),
            S::splat(IV[3]),
            S::splat(IV[4] ^ t as u32),
            S::splat(IV[5] ^ (t >> 32) as u32),
            S::splat(if last { !IV[6] } else { IV[6] }),
            S::splat(IV[7]),
        ];
        // SAFETY: backend cfg supplies the ISA; m holds 16 * WIDTH transposed words.
        unsafe { rounds!(v, m) };
        for i in 0..8 {
            h[i] = h[i].xor(v[i]).xor(v[i + 8]);
        }
    }

    #[inline(always)]
    unsafe fn compress_groups<S: Lanes32, const G: usize>(
        h: &mut [[S; 8]; G],
        m: [*const u32; G],
        t: u64,
        last: bool,
    ) {
        let init = |h: &[S; 8]| {
            [
                h[0],
                h[1],
                h[2],
                h[3],
                h[4],
                h[5],
                h[6],
                h[7],
                S::splat(IV[0]),
                S::splat(IV[1]),
                S::splat(IV[2]),
                S::splat(IV[3]),
                S::splat(IV[4] ^ t as u32),
                S::splat(IV[5] ^ (t >> 32) as u32),
                S::splat(if last { !IV[6] } else { IV[6] }),
                S::splat(IV[7]),
            ]
        };
        let mut v: [[S; 16]; G] = std::array::from_fn(|g| init(&h[g]));
        // SAFETY: backend cfg supplies the ISA; every m[g] holds 16 * WIDTH words.
        unsafe {
            macro_rules! round_all {
            ($($name:ident),*) => { $( for g in 0..G { $name(&mut v[g], m[g]); } )* };
        }
            round_all!(
                round_0, round_1, round_2, round_3, round_4, round_5, round_6, round_7, round_8,
                round_9
            );
        }
        for g in 0..G {
            for i in 0..8 {
                h[g][i] = h[g][i].xor(v[g][i]).xor(v[g][i + 8]);
            }
        }
    }

    #[inline(always)]
    pub(super) fn hash_many_with<S: Lanes32>(data: &[u8], len: usize, out: &mut [[u8; 32]]) {
        let groups = out.len() / S::WIDTH;
        let mut group = 0;
        while group + S::GROUPS <= groups {
            if S::GROUPS == 4 {
                hash_groups::<S, 4>(
                    &data[group * S::WIDTH * len..],
                    len,
                    &mut out[group * S::WIDTH..(group + 4) * S::WIDTH],
                );
                group += 4;
            } else {
                hash_groups::<S, 1>(
                    &data[group * S::WIDTH * len..],
                    len,
                    &mut out[group * S::WIDTH..(group + 1) * S::WIDTH],
                );
                group += 1;
            }
        }
        if S::GROUPS > 1 && group + 2 <= groups {
            hash_groups::<S, 2>(
                &data[group * S::WIDTH * len..],
                len,
                &mut out[group * S::WIDTH..(group + 2) * S::WIDTH],
            );
            group += 2;
        }
        while group < groups {
            hash_groups::<S, 1>(
                &data[group * S::WIDTH * len..],
                len,
                &mut out[group * S::WIDTH..(group + 1) * S::WIDTH],
            );
            group += 1;
        }
        for (input, slot) in data[group * S::WIDTH * len..]
            .chunks_exact(len.max(1))
            .zip(&mut out[group * S::WIDTH..])
        {
            *slot = super::hash_leaf(input);
        }
        if len == 0 {
            for slot in &mut out[group * S::WIDTH..] {
                *slot = super::hash_leaf(&[]);
            }
        }
    }

    #[inline(always)]
    fn hash_groups<S: Lanes32, const G: usize>(data: &[u8], len: usize, out: &mut [[u8; 32]]) {
        let mut initial = IV;
        initial[0] ^= 0x0101_0020;
        let mut h: [[S; 8]; G] = [std::array::from_fn(|i| S::splat(initial[i])); G];
        let mut buffers = [[0u32; 16 * 8]; G];
        let mut padded = [[0u8; 64]; 16];
        let blocks = len.div_ceil(64).max(1);
        for block in 0..blocks {
            let off = block * 64;
            let end = len.min(off + 64);
            let mut messages = [std::ptr::null(); G];
            for g in 0..G {
                let mut inputs = [std::ptr::null(); 8];
                for (lane, pointer) in inputs.iter_mut().take(S::WIDTH).enumerate() {
                    let index = g * S::WIDTH + lane;
                    let input = &data[index * len + off..index * len + end];
                    *pointer = if end - off == 64 {
                        input.as_ptr()
                    } else {
                        padded[index][..input.len()].copy_from_slice(input);
                        padded[index].as_ptr()
                    };
                }
                // SAFETY: backend cfg supplies ISA; each pointer spans a full block (padded if partial),
                // and buffers[g] holds at least 16 * WIDTH aligned words.
                unsafe { S::transpose(&inputs[..S::WIDTH], 0, &mut buffers[g]) };
                messages[g] = buffers[g].as_ptr();
            }
            // SAFETY: backend cfg supplies ISA; each transposed message has 16 * WIDTH words.
            unsafe {
                if G == 1 {
                    compress_lanes::<S>(&mut h[0], messages[0], end as u64, block + 1 == blocks);
                } else {
                    compress_groups::<S, G>(&mut h, messages, end as u64, block + 1 == blocks);
                }
            }
        }
        for (g, state) in h.iter().enumerate() {
            // SAFETY: backend cfg supplies ISA; out spans G * WIDTH initialized 32-byte arrays,
            // and group g's writable window is disjoint from the other groups.
            unsafe { S::store_digests(state, out.as_mut_ptr().add(g * S::WIDTH).cast()) };
        }
    }
}

#[cfg(any(
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ),
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_endian = "little"
    )
))]
use kernel::Lanes32;

#[inline]
pub(super) fn hash_many(data: &[u8], len: usize, out: &mut [Digest]) -> Result<(), WhirError> {
    let expected = checked_product(WhirPart::Leaves, &[len, out.len()])?;
    if data.len() != expected {
        return Err(WhirError::Shape {
            part: WhirPart::Leaves,
            expected,
            actual: data.len(),
        });
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    kernel::hash_many_with::<Neon>(data, len, out);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_endian = "little"
    ))]
    kernel::hash_many_with::<Avx2>(data, len, out);
    #[cfg(not(any(
        all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little"
        ),
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_endian = "little"
        )
    )))]
    for (index, slot) in out.iter_mut().enumerate() {
        *slot = hash_leaf(&data[index * len..(index + 1) * len]);
    }
    Ok(())
}

pub(super) fn hash_pairs(read: &[Digest], out: &mut [Digest]) -> Result<(), WhirError> {
    let expected = checked_product(WhirPart::Digests, &[2, out.len()])?;
    if read.len() != expected {
        return Err(WhirError::Shape {
            part: WhirPart::Digests,
            expected,
            actual: read.len(),
        });
    }
    let bytes = checked_product(WhirPart::Digests, &[read.len(), 32])?;
    // SAFETY: [u8;32] has byte alignment and no padding; read is initialized for exactly bytes bytes.
    // This changes no byte order and borrows only immutable digest storage.
    let data = unsafe { std::slice::from_raw_parts(read.as_ptr().cast(), bytes) };
    hash_many(data, 64, out)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests assert valid batched inputs")]
mod tests {
    use super::{hash_many, hash_pairs};
    use jolt_rv64i_verifier::whir::{
        error::{WhirError, WhirPart},
        merkle::{hash_leaf, Digest},
    };

    #[test]
    fn batched_hashes_match_definition_every_count_and_ragged_tail() {
        let mut seed = 0xcafe_0123_4567_89abu64;
        for count in (0..=33).chain([63, 65, 1023, 1024, 1025]) {
            for len in [0, 1, 3, 16, 63, 64, 65, 192, 384, 512, 513, 1024] {
                for pattern in 0..4 {
                    let mut input = vec![0; count * len + 1];
                    for byte in &mut input[1..] {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        *byte = match pattern {
                            0 => 0,
                            1 => 255,
                            2 => 0x80 ^ (seed as u8 & 1),
                            _ => seed as u8,
                        };
                    }
                    let data = &input[1..];
                    let mut output = vec![[0; 32]; count];
                    hash_many(data, len, &mut output).unwrap();
                    for (i, digest) in output.iter().enumerate() {
                        assert_eq!(*digest, hash_leaf(&data[i * len..(i + 1) * len]));
                    }
                    if len == 64 {
                        let children: Vec<Digest> = data
                            .chunks_exact(32)
                            .map(|b| b.try_into().unwrap())
                            .collect();
                        let mut parents = vec![[0; 32]; count];
                        hash_pairs(&children, &mut parents).unwrap();
                        assert_eq!(parents, output);
                    }
                }
            }
        }
    }

    #[test]
    fn published_vectors_run_through_full_simd_batches() {
        for (input, hex) in [
            (
                &b""[..],
                "69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9",
            ),
            (
                &b"abc"[..],
                "508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982",
            ),
        ] {
            let digits: Vec<u8> = hex.chars().map(|c| c.to_digit(16).unwrap() as u8).collect();
            let expected: Digest = std::array::from_fn(|i| digits[2 * i] * 16 + digits[2 * i + 1]);
            let mut output = [[0; 32]; 16];
            hash_many(&input.repeat(16), input.len(), &mut output).unwrap();
            assert_eq!(output, [expected; 16]);
        }
    }

    #[test]
    fn batched_boundaries_report_typed_shape_and_overflow() {
        assert_eq!(
            hash_many(&[], 512, &mut [[0; 32]]),
            Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: 512,
                actual: 0
            })
        );
        assert_eq!(
            hash_many(&[], usize::MAX, &mut [[0; 32]; 2]),
            Err(WhirError::LengthOverflow {
                part: WhirPart::Leaves
            })
        );
        assert_eq!(
            hash_pairs(&[[0; 32]], &mut [[0; 32]]),
            Err(WhirError::Shape {
                part: WhirPart::Digests,
                expected: 2,
                actual: 1
            })
        );
    }
}
