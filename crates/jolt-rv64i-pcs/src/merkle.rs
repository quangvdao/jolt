// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
// Modified from leanVM crates/pcs/src/merkle.rs and
// crates/fiat_shamir/src/merkle.rs at 48a904208d682848dac0e18ef8b01ebfc40df9ad.
// Notices: crates/jolt-rv64i-pcs/THIRD_PARTY_NOTICES.md.
//! Contiguous Merkle trees: leaf hashes first, then successive parent layers.

use crate::parallel;
use jolt_field::CanonicalBytes;
#[cfg(not(feature = "arch"))]
use jolt_rv64i_verifier::whir::merkle::{hash_leaf, hash_node};
use jolt_rv64i_verifier::whir::{
    error::{checked_product, try_vec, WhirError, WhirPart},
    merkle::{sibling_indices, Digest},
};
use rayon::prelude::*;

/// Retained by the commitment prover until its oracle's multiproof is written.
/// The tree owns exactly `2 * num_leaves - 1` digests, without a leaf-byte copy.
#[derive(Debug)]
pub struct MerkleTree {
    nodes: Vec<Digest>,
    num_leaves: usize,
}

impl MerkleTree {
    /// Hashes a position-major typed codeword with canonical serialization
    /// bounded to one batch per worker. Leaves are at most 512 bytes.
    pub fn build_canonical<F: CanonicalBytes + Sync>(
        data: &[F],
        entries_per_leaf: usize,
    ) -> Result<Self, WhirError> {
        let leaf_bytes = checked_product(WhirPart::Leaves, &[entries_per_leaf, F::NUM_BYTES])?;
        if leaf_bytes == 0 || leaf_bytes > 512 {
            return Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: 512,
                actual: leaf_bytes,
            });
        }
        if !data.len().is_multiple_of(entries_per_leaf) {
            return Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: entries_per_leaf,
                actual: data.len(),
            });
        }
        let mut tree = Self::empty(data.len() / entries_per_leaf)?;
        let fill = |first_leaf: usize, outputs: &mut [Digest]| {
            let bytes_len =
                checked_product(WhirPart::Leaves, &[outputs.len().min(1024), leaf_bytes])?;
            let mut bytes = try_vec(WhirPart::Leaves, bytes_len)?;
            bytes.resize(bytes_len, 0);
            for (batch, outputs) in outputs.chunks_mut(1024).enumerate() {
                let used = outputs.len() * leaf_bytes;
                let bytes = &mut bytes[..used];
                let first = (first_leaf + batch * 1024) * entries_per_leaf;
                let values = &data[first..first + outputs.len() * entries_per_leaf];
                for (value, encoding) in values.iter().zip(bytes.chunks_exact_mut(F::NUM_BYTES)) {
                    value.to_bytes_le(encoding);
                }
                #[cfg(feature = "arch")]
                crate::arch::hash_many(bytes, leaf_bytes, outputs)?;
                #[cfg(not(feature = "arch"))]
                for (leaf, digest) in bytes.chunks_exact(leaf_bytes).zip(outputs) {
                    *digest = hash_leaf(leaf);
                }
            }
            Ok::<_, WhirError>(())
        };
        if parallel::enabled(tree.num_leaves) {
            let group = if tree.num_leaves / 4096 >= 4 * rayon::current_num_threads() {
                4096
            } else {
                1024
            };
            tree.nodes[..tree.num_leaves]
                .par_chunks_mut(group)
                .enumerate()
                .try_for_each(|(group_index, outputs)| fill(group_index * group, outputs))?;
        } else {
            fill(0, &mut tree.nodes[..tree.num_leaves])?;
        }
        tree.fill_parents()?;
        Ok(tree)
    }

    /// Build from equal-width canonical byte leaves. The input must contain a
    /// positive power of two leaves of nonzero width, including 512 and 384.
    pub fn build(data: &[u8], leaf_bytes: usize) -> Result<Self, WhirError> {
        if leaf_bytes == 0 || !data.len().is_multiple_of(leaf_bytes) {
            return Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: leaf_bytes.max(1),
                actual: data.len(),
            });
        }
        #[cfg(feature = "arch")]
        {
            let mut tree = Self::empty(data.len() / leaf_bytes)?;
            tree.nodes[..tree.num_leaves]
                .par_chunks_mut(1024)
                .enumerate()
                .try_for_each(|(group, outputs)| {
                    let start = group * 1024 * leaf_bytes;
                    let input = &data[start..start + outputs.len() * leaf_bytes];
                    crate::arch::hash_many(input, leaf_bytes, outputs)
                })?;
            tree.fill_parents()?;
            Ok(tree)
        }
        #[cfg(not(feature = "arch"))]
        Self::from_leaves(data.len() / leaf_bytes, |index| {
            hash_leaf(&data[index * leaf_bytes..(index + 1) * leaf_bytes])
        })
    }

    /// Build from a leaf hasher, called once per index in parallel. The caller
    /// can serialize field words little-endian on its stack without allocating
    /// another codeword. `num_leaves` must be a positive power of two.
    pub fn from_leaves(
        num_leaves: usize,
        leaf: impl Fn(usize) -> Digest + Sync,
    ) -> Result<Self, WhirError> {
        let mut tree = Self::empty(num_leaves)?;
        tree.nodes[..num_leaves]
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, slot)| *slot = leaf(index));
        tree.fill_parents()?;
        Ok(tree)
    }

    fn empty(num_leaves: usize) -> Result<Self, WhirError> {
        if !num_leaves.is_power_of_two() {
            return Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: 1,
                actual: num_leaves,
            });
        }
        let len = checked_product(WhirPart::Digests, &[num_leaves, 2])? - 1;
        let mut nodes = try_vec(WhirPart::Digests, len)?;
        nodes.resize(len, [0; 32]);
        Ok(Self { nodes, num_leaves })
    }

    fn fill_parents(&mut self) -> Result<(), WhirError> {
        let mut start = 0;
        let mut width = self.num_leaves;
        while width > 1 {
            let (read, write) = self.nodes.split_at_mut(start + width);
            #[cfg(feature = "arch")]
            if parallel::enabled(width / 2) {
                write[..width / 2]
                    .par_chunks_mut(1024)
                    .enumerate()
                    .try_for_each(|(group, outputs)| {
                        let first = start + group * 2048;
                        crate::arch::hash_pairs(&read[first..first + outputs.len() * 2], outputs)
                    })?;
            } else {
                crate::arch::hash_pairs(&read[start..start + width], &mut write[..width / 2])?;
            }
            #[cfg(not(feature = "arch"))]
            if parallel::enabled(width / 2) {
                read[start..]
                    .par_chunks_exact(2)
                    .zip(write[..width / 2].par_iter_mut())
                    .for_each(|(pair, slot)| *slot = hash_node(&pair[0], &pair[1]));
            } else {
                read[start..]
                    .chunks_exact(2)
                    .zip(write[..width / 2].iter_mut())
                    .for_each(|(pair, slot)| *slot = hash_node(&pair[0], &pair[1]));
            }
            start += width;
            width /= 2;
        }
        Ok(())
    }

    /// The root authenticates the exact byte leaves supplied to the builder.
    pub fn root(&self) -> &Digest {
        &self.nodes[self.nodes.len() - 1]
    }

    /// Extract only sibling digests; callers retain the leaves from the oracle.
    /// Positions are nonempty, sorted, distinct and in range. The returned
    /// digests follow the verifier's bottom-up, left-to-right wire order.
    pub fn multiproof(&self, positions: &[usize], level: usize) -> Result<Vec<Digest>, WhirError> {
        let siblings =
            sibling_indices(self.num_leaves.trailing_zeros() as usize, positions, level)?;
        let mut digests = try_vec(WhirPart::Digests, siblings.len())?;
        for (height, index) in siblings {
            let layer_start = 2 * (self.num_leaves - (self.num_leaves >> height));
            digests.push(self.nodes[layer_start + index]);
        }
        Ok(digests)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests assert valid trees and proofs")]
mod tests {
    use super::MerkleTree;
    use blake2::{Blake2s256, Digest as BlakeDigest};
    use jolt_field::{CanonicalBytes, ExtField, F192, F64};
    use jolt_rv64i_verifier::whir::{
        error::{WhirError, WhirPart},
        merkle::{sibling_indices, verify_multiproof, Digest},
    };
    use rayon::ThreadPoolBuilder;
    use std::collections::BTreeSet;

    fn definition_tree(leaves: &[Vec<u8>]) -> Vec<Vec<Digest>> {
        let mut layers = vec![leaves
            .iter()
            .map(|l| Blake2s256::digest(l).into())
            .collect::<Vec<Digest>>()];
        while layers.last().unwrap().len() > 1 {
            let parents = layers
                .last()
                .unwrap()
                .chunks_exact(2)
                .map(|pair| Blake2s256::digest([pair[0], pair[1]].concat()).into())
                .collect();
            layers.push(parents);
        }
        layers
    }

    fn canonical_root<F: CanonicalBytes + Sync>(values: &[F], entries: usize) {
        let leaves: Vec<Vec<u8>> = values
            .chunks_exact(entries)
            .map(|leaf| {
                leaf.iter()
                    .flat_map(CanonicalBytes::to_bytes_le_vec)
                    .collect()
            })
            .collect();
        let definition = definition_tree(&leaves);
        assert_eq!(
            MerkleTree::build_canonical(values, entries).unwrap().root(),
            &definition.last().unwrap()[0]
        );
    }

    #[test]
    fn canonical_typed_codewords_match_independent_hash_definition() {
        for leaves in [1, 2, 8, 1024, 2048] {
            for width in [1, 2, 64] {
                let values: Vec<_> = (0..leaves * width)
                    .map(|i| F64::from_raw(i as u64 * 0x19a7))
                    .collect();
                canonical_root(&values, width);
            }
            for width in [1, 8, 16] {
                let values: Vec<_> = (0..leaves * width)
                    .map(|i| F192::from_base_fn(|c| F64::from_raw(i as u64 * 0x19a7 + c as u64)))
                    .collect();
                canonical_root(&values, width);
            }
        }
    }

    #[test]
    fn grouped_serialization_reuse_matches_independent_hash_definition() {
        let values: Vec<_> = (0..32768 * 16)
            .map(|i| F192::from_base_fn(|c| F64::from_raw(i as u64 * 0x19a7 + c as u64)))
            .collect();
        for threads in [1, 2] {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| canonical_root(&values, 16));
        }
    }

    #[test]
    fn canonical_typed_codeword_shapes_fail_before_hashing() {
        for (words, width, expected, actual) in [
            (1, 0, 512, 0),
            (65, 65, 512, 520),
            (3, 2, 2, 3),
            (0, 1, 1, 0),
            (3, 1, 1, 3),
        ] {
            let values = vec![F64::from_raw(1); words];
            assert!(
                matches!(MerkleTree::build_canonical(&values, width), Err(WhirError::Shape { part: WhirPart::Leaves, expected: e, actual: a }) if e == expected && a == actual)
            );
        }
    }

    #[test]
    fn smallest_tree_roots_are_literals() {
        assert_eq!(
            *MerkleTree::build(b"a", 1).unwrap().root(),
            [
                0x4a, 0x0d, 0x12, 0x98, 0x73, 0x40, 0x30, 0x37, 0xc2, 0xcd, 0x9b, 0x90, 0x48, 0x20,
                0x36, 0x87, 0xf6, 0x23, 0x3f, 0xb6, 0x73, 0x89, 0x56, 0xe0, 0x34, 0x9b, 0xd4, 0x32,
                0x0f, 0xec, 0x3e, 0x90
            ]
        );
        assert_eq!(
            *MerkleTree::build(b"ab", 1).unwrap().root(),
            [
                0x2d, 0x12, 0xd4, 0xf7, 0xa2, 0xc2, 0xc9, 0xe0, 0x2f, 0xc6, 0x30, 0x0b, 0x0d, 0x23,
                0xc7, 0x72, 0x45, 0x7a, 0xa5, 0xe3, 0x0d, 0x1d, 0x69, 0xe7, 0xb5, 0x89, 0xb8, 0xa4,
                0x8a, 0xfe, 0x54, 0x25
            ]
        );
        assert_eq!(
            *MerkleTree::build(b"abcd", 1).unwrap().root(),
            [
                0x25, 0x28, 0xde, 0x40, 0x90, 0x3e, 0xc0, 0x22, 0x45, 0xc5, 0x1a, 0x6c, 0xf9, 0x2a,
                0x35, 0xd3, 0xe9, 0xbc, 0x79, 0xf8, 0x71, 0xf0, 0xec, 0xba, 0x80, 0x3a, 0x7b, 0x59,
                0xfe, 0x39, 0xd5, 0x45
            ]
        );
    }

    #[test]
    fn roots_match_definition_all_depths_and_leaf_widths() {
        for threads in [1, 12] {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    for depth in 0..=12 {
                        for width in [16, 64, 192, 384, 512] {
                            let leaves: Vec<Vec<u8>> = (0..1usize << depth)
                                .map(|p| {
                                    (0..width)
                                        .map(|i| (p * 37 + i * 131 + (p >> 8)) as u8)
                                        .collect()
                                })
                                .collect();
                            let expected = definition_tree(&leaves);
                            let tree = MerkleTree::build(&leaves.concat(), width).unwrap();
                            assert_eq!(tree.root(), &expected.last().unwrap()[0]);
                            assert_eq!(tree.nodes, expected.concat());
                        }
                    }
                });
        }
        let leaves = [b"a".to_vec(), b"b".to_vec(), b"c".to_vec(), b"d".to_vec()];
        let h = |data: &[u8]| -> Digest { Blake2s256::digest(data).into() };
        let left = h(&[h(b"a"), h(b"b")].concat());
        let right = h(&[h(b"c"), h(b"d")].concat());
        assert_eq!(
            MerkleTree::build(&leaves.concat(), 1).unwrap().root(),
            &h(&[left, right].concat())
        );
    }

    #[test]
    fn every_small_multiproof_matches_paths_and_rejects_tampering() {
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        for depth in 0..=4 {
            let width = 16;
            let n = 1usize << depth;
            let leaves: Vec<Vec<u8>> = (0..n).map(|p| vec![p as u8; width]).collect();
            let layers = definition_tree(&leaves);
            let tree = pool
                .install(|| MerkleTree::build(&leaves.concat(), width))
                .unwrap();
            for mask in 1u32..1 << n {
                let positions: Vec<usize> = (0..n).filter(|p| mask & (1 << p) != 0).collect();
                let opened: Vec<Vec<u8>> = positions.iter().map(|&p| leaves[p].clone()).collect();
                let proof = tree.multiproof(&positions, 3).unwrap();
                // Independent per-leaf paths: retain a sibling only when no
                // queried leaf lies in its subtree, then sort by wire order.
                let mut missing = BTreeSet::new();
                for &position in &positions {
                    let mut path_root: Digest = Blake2s256::digest(&leaves[position]).into();
                    for (height, layer) in layers.iter().take(depth).enumerate() {
                        let sibling = (position >> height) ^ 1;
                        if !positions.iter().any(|&p| p >> height == sibling) {
                            let _inserted = missing.insert((height, sibling));
                        }
                        let pair = if position >> height & 1 == 0 {
                            [path_root, layer[sibling]]
                        } else {
                            [layer[sibling], path_root]
                        };
                        path_root = Blake2s256::digest(pair.concat()).into();
                    }
                    assert_eq!(&path_root, tree.root());
                }
                let expected: Vec<Digest> = missing.iter().map(|&(h, p)| layers[h][p]).collect();
                assert_eq!(proof, expected);
                assert_eq!(
                    proof.len(),
                    sibling_indices(depth, &positions, 3).unwrap().len()
                );
                assert_eq!(
                    verify_multiproof(tree.root(), depth, width, &positions, &opened, &proof, 3),
                    Ok(())
                );
                for i in 0..proof.len() {
                    let mut changed = proof.clone();
                    changed[i][0] ^= 1;
                    assert_eq!(
                        verify_multiproof(
                            tree.root(),
                            depth,
                            width,
                            &positions,
                            &opened,
                            &changed,
                            3
                        ),
                        Err(WhirError::MerkleAuthentication { level: 3 })
                    );
                }
                for i in 0..opened.len() {
                    let mut changed = opened.clone();
                    changed[i][0] ^= 1;
                    assert_eq!(
                        verify_multiproof(
                            tree.root(),
                            depth,
                            width,
                            &positions,
                            &changed,
                            &proof,
                            3
                        ),
                        Err(WhirError::MerkleAuthentication { level: 3 })
                    );
                }
                if !proof.is_empty() {
                    assert_eq!(
                        verify_multiproof(
                            tree.root(),
                            depth,
                            width,
                            &positions,
                            &opened,
                            &proof[..proof.len() - 1],
                            3
                        ),
                        Err(WhirError::CountMismatch {
                            level: 3,
                            part: WhirPart::Digests,
                            expected: proof.len(),
                            actual: proof.len() - 1
                        })
                    );
                }
                let mut extra = proof.clone();
                extra.push([0; 32]);
                assert_eq!(
                    verify_multiproof(tree.root(), depth, width, &positions, &opened, &extra, 3),
                    Err(WhirError::CountMismatch {
                        level: 3,
                        part: WhirPart::Digests,
                        expected: proof.len(),
                        actual: proof.len() + 1
                    })
                );
            }
        }
    }

    #[test]
    fn tight_maximum_sibling_count_is_attained() {
        for depth in 0..=4 {
            let n = 1usize << depth;
            let mut maxima = vec![0; n + 1];
            for mask in 1u32..1 << n {
                let positions: Vec<usize> = (0..n).filter(|p| mask & (1 << p) != 0).collect();
                maxima[positions.len()] = maxima[positions.len()]
                    .max(sibling_indices(depth, &positions, 0).unwrap().len());
            }
            for (q, &maximum) in maxima.iter().enumerate().skip(1) {
                let ceil_log = usize::BITS - (q - 1).leading_zeros();
                assert_eq!(
                    maximum,
                    q * (depth - ceil_log as usize) + (1 << ceil_log) - q
                );
            }
        }
        for (depth, q, expected) in [
            (19, 260, 2852),
            (18, 65, 778),
            (17, 37, 434),
            (16, 26, 292),
            (15, 20, 212),
        ] {
            let mut positions: Vec<usize> = (0..q)
                .map(|p: usize| p.reverse_bits() >> (usize::BITS as usize - depth))
                .collect();
            positions.sort_unstable();
            assert_eq!(
                sibling_indices(depth, &positions, 0).unwrap().len(),
                expected
            );
        }
    }

    #[test]
    fn leaf_counts_positions_and_root_faults_are_typed() {
        let data: Vec<u8> = (0..64).collect();
        let tree = MerkleTree::build(&data, 16).unwrap();
        let positions = [0, 2];
        let leaves = [&data[..16], &data[32..48]];
        let proof = tree.multiproof(&positions, 9).unwrap();
        for actual in [1, 3] {
            let malformed = vec![leaves[0]; actual];
            assert_eq!(
                verify_multiproof(tree.root(), 2, 16, &positions, &malformed, &proof, 9),
                Err(WhirError::CountMismatch {
                    level: 9,
                    part: WhirPart::Leaves,
                    expected: 2,
                    actual
                })
            );
        }
        for malformed in [[1, 2], [0, 3], [0, 0], [2, 0], [0, 4]] {
            assert_eq!(
                verify_multiproof(tree.root(), 2, 16, &malformed, &leaves, &proof, 9),
                Err(WhirError::MerkleAuthentication { level: 9 })
            );
        }
        let mut root = *tree.root();
        root[31] ^= 1;
        assert_eq!(
            verify_multiproof(&root, 2, 16, &positions, &leaves, &proof, 9),
            Err(WhirError::MerkleAuthentication { level: 9 })
        );
        assert!(MerkleTree::build(&[], 16).is_err());
        assert!(MerkleTree::build(&data, 0).is_err());
        assert!(MerkleTree::build(&data, 15).is_err());
        assert!(MerkleTree::build(&data[..48], 16).is_err());
        assert!(matches!(
            MerkleTree::from_leaves(1usize << (usize::BITS - 1), |_| [0; 32]),
            Err(WhirError::LengthOverflow {
                part: WhirPart::Digests
            })
        ));
    }

    #[test]
    fn tree_and_proof_are_identical_on_one_and_twelve_threads() {
        let data: Vec<u8> = (0..512 * 1024).map(|i| (i * 31) as u8).collect();
        let build = |threads| {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| MerkleTree::build(&data, 512))
                .unwrap()
        };
        let one = build(1);
        let twelve = build(12);
        assert_eq!(one.nodes, twelve.nodes);
        assert_eq!(
            one.multiproof(&[0, 3, 127, 1023], 0).unwrap(),
            twelve.multiproof(&[0, 3, 127, 1023], 0).unwrap()
        );
    }
}
