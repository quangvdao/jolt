//! Canonical commitment and opening encodings from specification §6.

use super::{
    error::{checked_product, try_vec, WhirError, WhirPart},
    merkle::Digest,
    params::{Queries, Schedule},
};
use crate::commitment::{BitsGeometry, BitsWire};
use jolt_field::{CanonicalBytes, CanonicalEncoding, F192};

/// Commit-time root and out-of-domain lane evaluations, in lane order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhirCommitment {
    pub root: Digest,
    pub lane_values: Vec<F192>,
}

/// The message sent after a level's fold rounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WhirLevelMessage {
    /// The next oracle's root and one evaluation of its message.
    Intermediate { root: Digest, value: F192 },
    /// The last folded message, in low-variable-first index order.
    Final { values: Vec<F192> },
}

/// A level's messages and Merkle multiproof, with canonical raw leaf bytes.
/// Position order is ascending; positions themselves are drawn by the verifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhirLevelProof {
    pub rounds: Vec<[F192; 2]>,
    pub message: WhirLevelMessage,
    pub leaves: Vec<Vec<u8>>,
    pub digests: Vec<Digest>,
}

/// Opening messages in protocol order; the geometry determines every fixed size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhirOpeningProof {
    pub levels: Vec<WhirLevelProof>,
    pub closing_rounds: Vec<[F192; 2]>,
}

impl WhirCommitment {
    pub(crate) fn validate(&self, schedule: &Schedule) -> Result<(), WhirError> {
        let first = schedule.levels().first().ok_or(WhirError::Shape {
            part: WhirPart::Levels,
            expected: 1,
            actual: 0,
        })?;
        Self::check_length(WhirPart::LaneValues, first.lanes()?, self.lane_values.len())
    }

    fn check_length(part: WhirPart, expected: usize, actual: usize) -> Result<(), WhirError> {
        if expected == actual {
            Ok(())
        } else {
            Err(WhirError::Shape {
                part,
                expected,
                actual,
            })
        }
    }

    fn decode(bytes: &[u8], schedule: &Schedule) -> Option<Self> {
        let lanes = schedule.levels().first()?.lanes().ok()?;
        let length =
            32usize.checked_add(checked_product(WhirPart::LaneValues, &[24, lanes]).ok()?)?;
        if bytes.len() != length {
            return None;
        }
        let mut reader = Reader::new(bytes);
        let root = reader.digest()?;
        let lane_values = reader.elements(WhirPart::LaneValues, lanes)?;
        Some(Self { root, lane_values })
    }

    /// Decode a commitment against an explicit test schedule, without changing its wire grammar.
    #[cfg(feature = "test-utils")]
    pub fn read_with_schedule(bytes: &[u8], schedule: &Schedule) -> Option<Self> {
        Self::decode(bytes, schedule)
    }
}

impl BitsWire for WhirCommitment {
    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.root);
        for value in &self.lane_values {
            let mut bytes = [0; 24];
            value.to_bytes_le(&mut bytes);
            out.extend_from_slice(&bytes);
        }
    }

    fn read(bytes: &[u8], geometry: BitsGeometry) -> Option<Self> {
        Self::decode(bytes, &Schedule::new(geometry).ok()?)
    }
}

impl WhirOpeningProof {
    pub(crate) fn validate(&self, schedule: &Schedule) -> Result<(), WhirError> {
        WhirCommitment::check_length(WhirPart::Levels, schedule.levels().len(), self.levels.len())?;
        for (i, (proof, level)) in self.levels.iter().zip(schedule.levels()).enumerate() {
            WhirCommitment::check_length(WhirPart::Rounds, level.k, proof.rounds.len())?;
            let is_last = i + 1 == schedule.levels().len();
            match &proof.message {
                WhirLevelMessage::Intermediate { .. } if !is_last => {}
                WhirLevelMessage::Final { values } if is_last => {
                    WhirCommitment::check_length(
                        WhirPart::FinalValues,
                        Self::final_count(schedule).ok_or(WhirError::LengthOverflow {
                            part: WhirPart::FinalValues,
                        })?,
                        values.len(),
                    )?;
                }
                WhirLevelMessage::Intermediate { .. } | WhirLevelMessage::Final { .. } => {
                    return Err(WhirError::Shape {
                        part: WhirPart::Levels,
                        expected: usize::from(is_last),
                        actual: usize::from(matches!(
                            proof.message,
                            WhirLevelMessage::Final { .. }
                        )),
                    });
                }
            }
            for leaf in &proof.leaves {
                WhirCommitment::check_length(WhirPart::Leaves, level.leaf_bytes, leaf.len())?;
            }
        }
        WhirCommitment::check_length(WhirPart::Rounds, schedule.res(), self.closing_rounds.len())
    }

    fn final_count(schedule: &Schedule) -> Option<usize> {
        1usize.checked_shl(u32::try_from(schedule.res()).ok()?)
    }

    fn fixed_message_bytes(schedule: &Schedule, i: usize) -> Option<usize> {
        if i.checked_add(1)? == schedule.levels().len() {
            checked_product(WhirPart::FinalValues, &[24, Self::final_count(schedule)?]).ok()
        } else {
            Some(56)
        }
    }

    // No proof-owned allocation is reachable until this complete walk succeeds.
    fn first_pass(bytes: &[u8], schedule: &Schedule) -> Option<()> {
        let mut reader = Reader::new(bytes);
        for (i, level) in schedule.levels().iter().enumerate() {
            reader.skip(checked_product(WhirPart::Rounds, &[48, level.k]).ok()?)?;
            reader.skip(Self::fixed_message_bytes(schedule, i)?)?;
            let leaves = reader.count()?;
            let positions = 1usize.checked_shl(u32::try_from(level.d).ok()?)?;
            match level.queries {
                Queries::All if leaves != positions => return None,
                Queries::Count(queries) => {
                    if leaves == 0 || leaves > usize::try_from(queries).ok()?.min(positions) {
                        return None;
                    }
                }
                Queries::All => {}
            }
            reader.skip(checked_product(WhirPart::Leaves, &[leaves, level.leaf_bytes]).ok()?)?;
            let digests = reader.count()?;
            match level.queries {
                Queries::All if digests != 0 => return None,
                Queries::Count(_) => {
                    if digests > checked_product(WhirPart::Digests, &[leaves, level.d]).ok()? {
                        return None;
                    }
                }
                Queries::All => {}
            }
            reader.skip(checked_product(WhirPart::Digests, &[32, digests]).ok()?)?;
        }
        reader.skip(checked_product(WhirPart::Rounds, &[48, schedule.res()]).ok()?)?;
        reader.finished().then_some(())
    }

    fn decode(bytes: &[u8], schedule: &Schedule) -> Option<Self> {
        Self::first_pass(bytes, schedule)?;
        let mut reader = Reader::new(bytes);
        let mut levels = try_vec(WhirPart::Levels, schedule.levels().len()).ok()?;
        for (i, level) in schedule.levels().iter().enumerate() {
            let rounds = reader.rounds(level.k)?;
            let message = if i + 1 == schedule.levels().len() {
                WhirLevelMessage::Final {
                    values: reader.elements(WhirPart::FinalValues, Self::final_count(schedule)?)?,
                }
            } else {
                WhirLevelMessage::Intermediate {
                    root: reader.digest()?,
                    value: reader.element()?,
                }
            };
            let count = reader.count()?;
            let mut leaves = try_vec(WhirPart::Leaves, count).ok()?;
            for _ in 0..count {
                let bytes = reader.take(level.leaf_bytes)?;
                let mut leaf = try_vec(WhirPart::Leaves, bytes.len()).ok()?;
                leaf.extend_from_slice(bytes);
                leaves.push(leaf);
            }
            let count = reader.count()?;
            let mut digests = try_vec(WhirPart::Digests, count).ok()?;
            for _ in 0..count {
                digests.push(reader.digest()?);
            }
            levels.push(WhirLevelProof {
                rounds,
                message,
                leaves,
                digests,
            });
        }
        let closing_rounds = reader.rounds(schedule.res())?;
        Some(Self {
            levels,
            closing_rounds,
        })
    }

    /// Decode an opening against an explicit test schedule, including the last fold of three.
    #[cfg(feature = "test-utils")]
    pub fn read_with_schedule(bytes: &[u8], schedule: &Schedule) -> Option<Self> {
        Self::decode(bytes, schedule)
    }
}

impl BitsWire for WhirOpeningProof {
    fn write(&self, out: &mut Vec<u8>) {
        for level in &self.levels {
            for round in &level.rounds {
                for value in round {
                    let mut bytes = [0; 24];
                    value.to_bytes_le(&mut bytes);
                    out.extend_from_slice(&bytes);
                }
            }
            match &level.message {
                WhirLevelMessage::Intermediate { root, value } => {
                    out.extend_from_slice(root);
                    let mut bytes = [0; 24];
                    value.to_bytes_le(&mut bytes);
                    out.extend_from_slice(&bytes);
                }
                WhirLevelMessage::Final { values } => {
                    for value in values {
                        let mut bytes = [0; 24];
                        value.to_bytes_le(&mut bytes);
                        out.extend_from_slice(&bytes);
                    }
                }
            }
            out.extend_from_slice(&(level.leaves.len() as u32).to_le_bytes());
            for leaf in &level.leaves {
                out.extend_from_slice(leaf);
            }
            out.extend_from_slice(&(level.digests.len() as u32).to_le_bytes());
            for digest in &level.digests {
                out.extend_from_slice(digest);
            }
        }
        for round in &self.closing_rounds {
            for value in round {
                let mut bytes = [0; 24];
                value.to_bytes_le(&mut bytes);
                out.extend_from_slice(&bytes);
            }
        }
    }

    fn read(bytes: &[u8], geometry: BitsGeometry) -> Option<Self> {
        Self::decode(bytes, &Schedule::new(geometry).ok()?)
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Option<&'a [u8]> {
        let end = self.offset.checked_add(length)?;
        let bytes = self.bytes.get(self.offset..end)?;
        self.offset = end;
        Some(bytes)
    }

    fn skip(&mut self, length: usize) -> Option<()> {
        let _ = self.take(length)?;
        Some(())
    }

    fn finished(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn count(&mut self) -> Option<usize> {
        usize::try_from(u32::from_le_bytes(self.take(4)?.try_into().ok()?)).ok()
    }

    fn digest(&mut self) -> Option<Digest> {
        self.take(32)?.try_into().ok()
    }

    fn element(&mut self) -> Option<F192> {
        F192::from_bytes_le_checked(self.take(24)?)
    }

    fn elements(&mut self, part: WhirPart, count: usize) -> Option<Vec<F192>> {
        let mut values = try_vec(part, count).ok()?;
        for _ in 0..count {
            values.push(self.element()?);
        }
        Some(values)
    }

    fn rounds(&mut self, count: usize) -> Option<Vec<[F192; 2]>> {
        let mut rounds = try_vec(WhirPart::Rounds, count).ok()?;
        for _ in 0..count {
            rounds.push([self.element()?, self.element()?]);
        }
        Some(rounds)
    }
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "tests assert checked fixture dimensions and successful decoding"
)]
mod tests {
    use super::{WhirCommitment, WhirLevelMessage, WhirLevelProof, WhirOpeningProof};
    use crate::{
        commitment::{BitsGeometry, BitsWire},
        whir::{
            error::{WhirError, WhirPart},
            params::{Queries, Schedule},
        },
    };
    use jolt_field::{CanonicalEncoding, F192};
    use std::fmt::Debug;

    struct Random(u64);

    impl Random {
        fn byte(&mut self) -> u8 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 as u8
        }

        fn element(&mut self) -> F192 {
            let bytes: [u8; 24] = std::array::from_fn(|_| self.byte());
            F192::from_bytes_le_checked(&bytes).unwrap()
        }

        fn proof(&mut self, schedule: &Schedule) -> WhirOpeningProof {
            let levels = schedule
                .levels()
                .iter()
                .enumerate()
                .map(|(i, level)| {
                    let count = match level.queries {
                        Queries::All => 1 << level.d,
                        Queries::Count(queries) => (queries as usize).min(3),
                    };
                    let message = if i + 1 == schedule.levels().len() {
                        WhirLevelMessage::Final {
                            values: (0..1 << schedule.res()).map(|_| self.element()).collect(),
                        }
                    } else {
                        WhirLevelMessage::Intermediate {
                            root: std::array::from_fn(|_| self.byte()),
                            value: self.element(),
                        }
                    };
                    WhirLevelProof {
                        rounds: (0..level.k)
                            .map(|_| [self.element(), self.element()])
                            .collect(),
                        message,
                        leaves: (0..count)
                            .map(|_| (0..level.leaf_bytes).map(|_| self.byte()).collect())
                            .collect(),
                        digests: (0..if level.queries == Queries::All { 0 } else { 3 })
                            .map(|_| std::array::from_fn(|_| self.byte()))
                            .collect(),
                    }
                })
                .collect();
            let closing_rounds = (0..schedule.res())
                .map(|_| [self.element(), self.element()])
                .collect();
            WhirOpeningProof {
                levels,
                closing_rounds,
            }
        }

        fn commitment(&mut self, schedule: &Schedule) -> WhirCommitment {
            WhirCommitment {
                root: std::array::from_fn(|_| self.byte()),
                lane_values: (0..schedule.levels()[0].lanes().unwrap())
                    .map(|_| self.element())
                    .collect(),
            }
        }
    }

    fn assert_complete_encoding<T: BitsWire + PartialEq + Debug>(
        value: &T,
        geometry: BitsGeometry,
    ) {
        let mut bytes = Vec::new();
        value.write(&mut bytes);
        assert_eq!(T::read(&bytes, geometry).as_ref(), Some(value));
        for length in 0..bytes.len() {
            assert!(
                T::read(&bytes[..length], geometry).is_none(),
                "prefix {length}"
            );
        }
        for extension in 0..=255 {
            bytes.push(extension);
            assert!(T::read(&bytes, geometry).is_none(), "extension {extension}");
            let _ = bytes.pop();
        }
    }

    #[test]
    fn wire_roundtrips_prefixes_and_all_one_byte_extensions_t1_through_t14() {
        let mut random = Random(0x48e6_470d_2367_0b91);
        for log_T in 1..=14 {
            let geometry = BitsGeometry { log_T };
            let schedule = Schedule::new(geometry).unwrap();
            let commitment = random.commitment(&schedule);
            let proof = random.proof(&schedule);
            assert_eq!(commitment.validate(&schedule), Ok(()));
            assert_eq!(proof.validate(&schedule), Ok(()));
            assert_complete_encoding(&commitment, geometry);
            assert_complete_encoding(&proof, geometry);
        }
    }

    #[test]
    fn wire_length_bounds_reject_in_the_allocation_free_pass() {
        let mut random = Random(0xe701_4d94_3c39_0821);
        for log_T in [1, 6, 10, 11, 14] {
            let geometry = BitsGeometry { log_T };
            let schedule = Schedule::new(geometry).unwrap();
            let proof = random.proof(&schedule);
            let mut bytes = Vec::new();
            proof.write(&mut bytes);
            assert_eq!(WhirOpeningProof::first_pass(&bytes, &schedule), Some(()));
            let mut offset = 0;
            for (i, (level, value)) in schedule.levels().iter().zip(&proof.levels).enumerate() {
                offset += 48 * level.k
                    + if i + 1 == schedule.levels().len() {
                        24 * (1 << schedule.res())
                    } else {
                        56
                    };
                let count_offset = offset;
                let count = value.leaves.len();
                let invalid = match level.queries {
                    Queries::All => [0, (count - 1) as u32, (count + 1) as u32, u32::MAX],
                    Queries::Count(queries) => [0, queries + 1, (1u32 << level.d) + 1, u32::MAX],
                };
                for length in invalid {
                    let mut malformed = bytes.clone();
                    malformed[count_offset..count_offset + 4]
                        .copy_from_slice(&length.to_le_bytes());
                    assert_eq!(WhirOpeningProof::first_pass(&malformed, &schedule), None);
                    assert!(WhirOpeningProof::read(&malformed, geometry).is_none());
                    // The same invalid header is rejected without supplying its purported body.
                    malformed.truncate(count_offset + 4);
                    assert_eq!(WhirOpeningProof::first_pass(&malformed, &schedule), None);
                }
                offset += 4 + count * level.leaf_bytes;
                let invalid = match level.queries {
                    Queries::All => [1, u32::MAX],
                    Queries::Count(_) => [(count * level.d + 1) as u32, u32::MAX],
                };
                for length in invalid {
                    let mut malformed = bytes.clone();
                    malformed[offset..offset + 4].copy_from_slice(&length.to_le_bytes());
                    assert_eq!(WhirOpeningProof::first_pass(&malformed, &schedule), None);
                    assert!(WhirOpeningProof::read(&malformed, geometry).is_none());
                    malformed.truncate(offset + 4);
                    assert_eq!(WhirOpeningProof::first_pass(&malformed, &schedule), None);
                }
                offset += 4 + 32 * value.digests.len();
            }
        }
    }

    #[test]
    fn wire_accepted_mutations_and_random_strings_have_unique_encodings() {
        let mut random = Random(0x201d_e759_918b_530c);
        let geometry = BitsGeometry { log_T: 6 };
        let schedule = Schedule::new(geometry).unwrap();
        let mut bytes = Vec::new();
        random.proof(&schedule).write(&mut bytes);
        for position in 0..bytes.len() {
            for difference in 1..=u8::MAX {
                bytes[position] ^= difference;
                if let Some(decoded) = WhirOpeningProof::read(&bytes, geometry) {
                    let mut canonical = Vec::new();
                    decoded.write(&mut canonical);
                    assert_eq!(canonical, bytes);
                }
                bytes[position] ^= difference;
            }
        }
        for trial in 0..10_000 {
            let length = (usize::from(random.byte()) << 4) | usize::from(random.byte());
            let bytes: Vec<u8> = (0..length).map(|_| random.byte()).collect();
            let geometry = BitsGeometry { log_T: trial % 35 };
            if let Some(decoded) = WhirOpeningProof::read(&bytes, geometry) {
                let mut canonical = Vec::new();
                decoded.write(&mut canonical);
                assert_eq!(canonical, bytes);
            }
            if let Some(decoded) = WhirCommitment::read(&bytes, geometry) {
                let mut canonical = Vec::new();
                decoded.write(&mut canonical);
                assert_eq!(canonical, bytes);
            }
        }
    }

    #[test]
    fn wire_fixed_words_are_little_endian_and_have_no_alternative_form() {
        let geometry = BitsGeometry { log_T: 1 };
        let mut random = Random(0x17a9_9033_4eda_650b);
        let schedule = Schedule::new(geometry).unwrap();
        let proof = random.proof(&schedule);
        let mut bytes = Vec::new();
        proof.write(&mut bytes);
        assert_eq!(&bytes[96..100], &[8, 0, 0, 0]);
        assert_eq!(&bytes[228..232], &[0, 0, 0, 0]);
        bytes[96..100].copy_from_slice(&[0, 0, 0, 8]);
        assert!(WhirOpeningProof::read(&bytes, geometry).is_none());
        bytes[96..100].copy_from_slice(&[8, 0, 0, 0]);
        bytes.insert(100, 0);
        assert!(WhirOpeningProof::read(&bytes, geometry).is_none());
    }

    #[test]
    fn typed_wire_shapes_report_the_first_fault() {
        let mut random = Random(0x592a_b571_192e_0445);
        let schedule = Schedule::new(BitsGeometry { log_T: 6 }).unwrap();
        let mut commitment = random.commitment(&schedule);
        let _ = commitment.lane_values.pop();
        assert_eq!(
            commitment.validate(&schedule),
            Err(WhirError::Shape {
                part: WhirPart::LaneValues,
                expected: 32,
                actual: 31,
            })
        );
        let original = random.proof(&schedule);
        let mut proof = original.clone();
        proof.levels.clear();
        assert_eq!(
            proof.validate(&schedule),
            Err(WhirError::Shape {
                part: WhirPart::Levels,
                expected: 1,
                actual: 0,
            })
        );
        let mut proof = original.clone();
        let _ = proof.levels[0].rounds.pop();
        assert_eq!(
            proof.validate(&schedule),
            Err(WhirError::Shape {
                part: WhirPart::Rounds,
                expected: 5,
                actual: 4,
            })
        );
        let mut proof = original.clone();
        if let WhirLevelMessage::Final { values } = &mut proof.levels[0].message {
            let _ = values.pop();
        }
        assert_eq!(
            proof.validate(&schedule),
            Err(WhirError::Shape {
                part: WhirPart::FinalValues,
                expected: 4,
                actual: 3,
            })
        );
        let mut proof = original.clone();
        let _ = proof.levels[0].leaves[0].pop();
        assert_eq!(
            proof.validate(&schedule),
            Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: 512,
                actual: 511,
            })
        );
        let mut proof = original.clone();
        let _ = proof.closing_rounds.pop();
        assert_eq!(
            proof.validate(&schedule),
            Err(WhirError::Shape {
                part: WhirPart::Rounds,
                expected: 2,
                actual: 1,
            })
        );
        let mut proof = original;
        proof.levels[0].message = WhirLevelMessage::Intermediate {
            root: [0; 32],
            value: random.element(),
        };
        assert_eq!(
            proof.validate(&schedule),
            Err(WhirError::Shape {
                part: WhirPart::Levels,
                expected: 1,
                actual: 0,
            })
        );
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn wire_explicit_last_fold_three_roundtrips() {
        let geometry = BitsGeometry { log_T: 11 };
        let schedule = Schedule::from_levels(
            geometry,
            &[(5, 7, 8, Queries::All), (3, 4, 7, Queries::All)],
        )
        .unwrap();
        let mut random = Random(0x7311_27ab_09fa_6201);
        let commitment = random.commitment(&schedule);
        let proof = random.proof(&schedule);
        assert_eq!(proof.levels[1].leaves[0].len(), 192);
        assert_eq!(proof.validate(&schedule), Ok(()));
        let mut bytes = Vec::new();
        commitment.write(&mut bytes);
        assert_eq!(
            WhirCommitment::read_with_schedule(&bytes, &schedule),
            Some(commitment)
        );
        bytes.clear();
        proof.write(&mut bytes);
        assert_eq!(
            WhirOpeningProof::read_with_schedule(&bytes, &schedule),
            Some(proof)
        );
    }
}
