// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
// Modified from leanVM crates/fiat_shamir/src/merkle.rs at
// 48a904208d682848dac0e18ef8b01ebfc40df9ad.
// Notices: crates/jolt-rv64i-pcs/THIRD_PARTY_NOTICES.md.
//! BLAKE2s-256 authentication and canonical multiproofs (specification §2, §6).

use super::error::{checked_product, try_vec, WhirError, WhirPart};
use blake2::{Blake2s256, Digest as BlakeDigest};

/// The complete 256-bit BLAKE2s output, in the hash's byte order.
pub type Digest = [u8; 32];

/// Hash the canonical leaf bytes, with no prefix, key, or personalization.
pub fn hash_leaf(bytes: &[u8]) -> Digest {
    Blake2s256::digest(bytes).into()
}

/// Hash the left digest followed by the right digest, with no prefix.
pub fn hash_node(left: &Digest, right: &Digest) -> Digest {
    let mut hash = Blake2s256::new();
    hash.update(left);
    hash.update(right);
    hash.finalize().into()
}

/// Missing siblings as `(height above leaves, index within layer)`, in wire
/// order. Positions must be nonempty, sorted, distinct, and below `2^depth`.
/// Shared with the prover so sibling order and digest counts have one owner.
pub fn sibling_indices(
    depth: usize,
    positions: &[usize],
    level: usize,
) -> Result<Vec<(usize, usize)>, WhirError> {
    let width = u32::try_from(depth)
        .ok()
        .and_then(|shift| 1usize.checked_shl(shift))
        .ok_or(WhirError::LengthOverflow {
            part: WhirPart::Leaves,
        })?;
    if positions.is_empty()
        || positions.last().is_some_and(|&p| p >= width)
        || positions.windows(2).any(|pair| pair.first() >= pair.get(1))
    {
        return Err(WhirError::MerkleAuthentication { level });
    }
    let mut active = try_vec(WhirPart::Leaves, positions.len())?;
    active.extend_from_slice(positions);
    let q = positions.len();
    let ceil_log_q = usize::BITS - (q - 1).leading_zeros();
    let balanced_width = 1usize
        .checked_shl(ceil_log_q)
        .ok_or(WhirError::LengthOverflow {
            part: WhirPart::Digests,
        })?;
    let bound = checked_product(WhirPart::Digests, &[q, depth - ceil_log_q as usize])?
        .checked_add(balanced_width - q)
        .ok_or(WhirError::LengthOverflow {
            part: WhirPart::Digests,
        })?;
    let mut siblings = try_vec(WhirPart::Digests, bound)?;
    for height in 0..depth {
        let mut read = 0;
        let mut write = 0;
        while let Some(&p) = active.get(read) {
            if active.get(read + 1) == Some(&(p ^ 1)) {
                read += 2;
            } else {
                siblings.push((height, p ^ 1));
                read += 1;
            }
            if let Some(slot) = active.get_mut(write) {
                *slot = p >> 1;
            }
            write += 1;
        }
        active.truncate(write);
    }
    Ok(siblings)
}

/// Authenticate one leaf per sorted distinct position. Leaf widths are checked
/// first, then exact leaf/digest counts, then the root. Opening every position
/// (including a depth-zero tree) requires no sibling digest.
pub fn verify_multiproof<L: AsRef<[u8]>>(
    root: &Digest,
    depth: usize,
    leaf_bytes: usize,
    positions: &[usize],
    leaves: &[L],
    digests: &[Digest],
    level: usize,
) -> Result<(), WhirError> {
    for leaf in leaves {
        if leaf.as_ref().len() != leaf_bytes {
            return Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: leaf_bytes,
                actual: leaf.as_ref().len(),
            });
        }
    }
    if leaves.len() != positions.len() {
        return Err(WhirError::CountMismatch {
            level,
            part: WhirPart::Leaves,
            expected: positions.len(),
            actual: leaves.len(),
        });
    }
    let siblings = sibling_indices(depth, positions, level)?;
    if digests.len() != siblings.len() {
        return Err(WhirError::CountMismatch {
            level,
            part: WhirPart::Digests,
            expected: siblings.len(),
            actual: digests.len(),
        });
    }
    let mut nodes = try_vec(WhirPart::Leaves, positions.len())?;
    nodes.extend(
        positions
            .iter()
            .copied()
            .zip(leaves.iter().map(|leaf| hash_leaf(leaf.as_ref()))),
    );
    let mut supplied = digests.iter();
    for _ in 0..depth {
        let mut read = 0;
        let mut write = 0;
        while let Some(&(index, digest)) = nodes.get(read) {
            let paired = nodes.get(read + 1).filter(|(p, _)| *p == (index ^ 1));
            let sibling = if let Some((_, sibling)) = paired {
                read += 2;
                *sibling
            } else {
                read += 1;
                *supplied
                    .next()
                    .ok_or(WhirError::MerkleAuthentication { level })?
            };
            let parent = if index & 1 == 0 {
                hash_node(&digest, &sibling)
            } else {
                hash_node(&sibling, &digest)
            };
            if let Some(slot) = nodes.get_mut(write) {
                *slot = (index >> 1, parent);
            }
            write += 1;
        }
        nodes.truncate(write);
    }
    if nodes.first() != Some(&(0, *root)) || supplied.next().is_some() {
        return Err(WhirError::MerkleAuthentication { level });
    }
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "fixed reference arrays and test assertions"
)]
mod tests {
    use super::{hash_leaf, hash_node, sibling_indices, verify_multiproof, Digest};
    use crate::whir::error::{WhirError, WhirPart};

    // Independent scalar implementation of the algorithm in RFC 7693 §3.
    fn reference_hash(input: &[u8]) -> Digest {
        const IV: [u32; 8] = [
            0x6a09_e667,
            0xbb67_ae85,
            0x3c6e_f372,
            0xa54f_f53a,
            0x510e_527f,
            0x9b05_688c,
            0x1f83_d9ab,
            0x5be0_cd19,
        ];
        const SIGMA: [[usize; 16]; 10] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
            [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
            [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
            [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
            [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
            [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
            [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
            [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
            [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
        ];
        fn mix(v: &mut [u32; 16], [a, b, c, d]: [usize; 4], x: u32, y: u32) {
            v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
            v[d] = (v[d] ^ v[a]).rotate_right(16);
            v[c] = v[c].wrapping_add(v[d]);
            v[b] = (v[b] ^ v[c]).rotate_right(12);
            v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
            v[d] = (v[d] ^ v[a]).rotate_right(8);
            v[c] = v[c].wrapping_add(v[d]);
            v[b] = (v[b] ^ v[c]).rotate_right(7);
        }
        let mut h = IV;
        h[0] ^= 0x0101_0020;
        let blocks = input.len().div_ceil(64).max(1);
        for block in 0..blocks {
            let start = block * 64;
            let end = input.len().min(start + 64);
            let mut bytes = [0; 64];
            bytes[..end - start].copy_from_slice(&input[start..end]);
            let m: [u32; 16] = std::array::from_fn(|i| {
                u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
            });
            let mut v = [0; 16];
            v[..8].copy_from_slice(&h);
            v[8..].copy_from_slice(&IV);
            v[12] ^= end as u32;
            v[13] ^= ((end as u64) >> 32) as u32;
            if block + 1 == blocks {
                v[14] = !v[14];
            }
            for s in SIGMA {
                for (i, indices) in [
                    [0, 4, 8, 12],
                    [1, 5, 9, 13],
                    [2, 6, 10, 14],
                    [3, 7, 11, 15],
                    [0, 5, 10, 15],
                    [1, 6, 11, 12],
                    [2, 7, 8, 13],
                    [3, 4, 9, 14],
                ]
                .into_iter()
                .enumerate()
                {
                    mix(&mut v, indices, m[s[2 * i]], m[s[2 * i + 1]]);
                }
            }
            for i in 0..8 {
                h[i] ^= v[i] ^ v[i + 8];
            }
        }
        let mut out = [0; 32];
        for (dst, word) in out.chunks_exact_mut(4).zip(h) {
            dst.copy_from_slice(&word.to_le_bytes());
        }
        out
    }

    fn hex_digest(hex: &str) -> Digest {
        let nibbles: Vec<u8> = hex.chars().map(|c| c.to_digit(16).unwrap() as u8).collect();
        std::array::from_fn(|i| (nibbles[2 * i] << 4) | nibbles[2 * i + 1])
    }

    #[test]
    fn blake2s_published_vectors_and_scalar_reference() {
        // RFC 7693 appendix B (abc), and the BLAKE2 unkeyed empty-input vector.
        for (bytes, hex) in [
            (
                &b"abc"[..],
                "508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982",
            ),
            (
                &b""[..],
                "69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9",
            ),
        ] {
            assert_eq!(hash_leaf(bytes), hex_digest(hex));
            assert_eq!(reference_hash(bytes), hex_digest(hex));
        }
        for length in 0..=1024 {
            let input: Vec<u8> = (0..length).map(|i| (i * 131 + 17) as u8).collect();
            assert_eq!(hash_leaf(&input), reference_hash(&input));
        }
        let left = std::array::from_fn(|i| i as u8);
        let right = std::array::from_fn(|i| (i + 32) as u8);
        assert_eq!(
            hash_node(&left, &right),
            reference_hash(&[left, right].concat())
        );
    }

    #[test]
    fn canonical_smallest_multiproofs() {
        assert_eq!(sibling_indices(0, &[0], 0), Ok(vec![]));
        assert_eq!(sibling_indices(1, &[0], 0), Ok(vec![(0, 1)]));
        assert_eq!(sibling_indices(1, &[1], 0), Ok(vec![(0, 0)]));
        assert_eq!(sibling_indices(2, &[0, 2], 0), Ok(vec![(0, 1), (0, 3)]));
        assert_eq!(sibling_indices(2, &[0, 1], 0), Ok(vec![(1, 1)]));
        assert_eq!(
            sibling_indices(3, &[1, 4, 5], 0),
            Ok(vec![(0, 0), (1, 1), (1, 3)])
        );
        assert_eq!(sibling_indices(2, &[0, 1, 2, 3], 0), Ok(vec![]));
    }

    #[test]
    fn malformed_positions_and_shapes_have_typed_errors() {
        for positions in [&[][..], &[0, 0], &[1, 0], &[0, 4], &[usize::MAX]] {
            assert_eq!(
                sibling_indices(2, positions, 7),
                Err(WhirError::MerkleAuthentication { level: 7 })
            );
        }
        assert_eq!(
            sibling_indices(usize::BITS as usize, &[0], 0),
            Err(WhirError::LengthOverflow {
                part: WhirPart::Leaves
            })
        );
        assert_eq!(
            verify_multiproof(&[0; 32], 0, 16, &[0], &[vec![0; 15]], &[], 7),
            Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: 16,
                actual: 15
            })
        );
    }
}
