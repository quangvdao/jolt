//! Word-parallel XOR subset sums and compaction of regularly spaced bits.

use thiserror::Error;

/// A position-bit count outside the six index bits of a 64-bit word.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BitsError {
    /// The Möbius transform requested more than six position bits.
    #[error("Möbius position-bit count {k} exceeds 6")]
    MoebiusBits {
        /// The supplied number of low position bits.
        k: usize,
    },
    /// Compaction requested a window larger than a 64-bit word.
    #[error("gather window-bit count {m} exceeds 6")]
    GatherBits {
        /// The supplied base-two logarithm of the window size.
        m: usize,
    },
}

/// Applies the XOR Möbius transform independently in each `2^k`-bit window.
///
/// For every window `v` and position `s < 2^k`, output bit `2^k·v + s` is
/// `⊕_{t ⊆ s} word[2^k·v + t]`, where subset inclusion is on the binary
/// position indices. One shift and mask per level implements the transform;
/// applying it twice is the identity. Returns `BitsError::MoebiusBits` for
/// `k > 6`; `k = 0` returns the input unchanged.
#[inline]
pub fn moebius(mut word: u64, k: usize) -> Result<u64, BitsError> {
    if k > 6 {
        return Err(BitsError::MoebiusBits { k });
    }
    const MASKS: [u64; 6] = [
        0xaaaa_aaaa_aaaa_aaaa,
        0xcccc_cccc_cccc_cccc,
        0xf0f0_f0f0_f0f0_f0f0,
        0xff00_ff00_ff00_ff00,
        0xffff_0000_ffff_0000,
        0xffff_ffff_0000_0000,
    ];
    for (level, &mask) in MASKS.iter().take(k).enumerate() {
        word ^= (word << (1 << level)) & mask;
    }
    Ok(word)
}

/// Compacts bit zero of each `2^m`-bit window into consecutive low bits.
///
/// Output bit `v < 64 / 2^m` is input bit `2^m·v`; all higher output bits are
/// zero, and all other input bits are ignored. For a significant bit at offset
/// `a` in each window, call `gather(word >> a, m)`. Shifts, masks and ORs compact
/// whole groups of positions without a loop over individual bits. Returns
/// `BitsError::GatherBits` for `m > 6`; `m = 0` returns the input unchanged.
#[inline]
pub fn gather(word: u64, m: usize) -> Result<u64, BitsError> {
    let packed = match m {
        0 => word,
        1 => compact(
            word & 0x5555_5555_5555_5555,
            [
                (1, 0x3333_3333_3333_3333),
                (2, 0x0f0f_0f0f_0f0f_0f0f),
                (4, 0x00ff_00ff_00ff_00ff),
                (8, 0x0000_ffff_0000_ffff),
                (16, 0x0000_0000_ffff_ffff),
            ],
        ),
        2 => compact(
            word & 0x1111_1111_1111_1111,
            [
                (3, 0x0303_0303_0303_0303),
                (6, 0x000f_000f_000f_000f),
                (12, 0x0000_00ff_0000_00ff),
                (24, 0x0000_0000_0000_ffff),
            ],
        ),
        3 => compact(
            word & 0x0101_0101_0101_0101,
            [
                (7, 0x0003_0003_0003_0003),
                (14, 0x0000_000f_0000_000f),
                (28, 0x0000_0000_0000_00ff),
            ],
        ),
        4 => compact(
            word & 0x0001_0001_0001_0001,
            [(15, 0x0000_0003_0000_0003), (30, 0x0000_0000_0000_000f)],
        ),
        5 => compact(word & 0x0000_0001_0000_0001, [(31, 3)]),
        6 => word & 1,
        _ => return Err(BitsError::GatherBits { m }),
    };
    Ok(packed)
}

#[inline]
fn compact<const N: usize>(mut word: u64, stages: [(u32, u64); N]) -> u64 {
    for (shift, mask) in stages {
        word = (word | (word >> shift)) & mask;
    }
    word
}
