//! Challenge bytes and labels of `specs/rv64i-binary-commitment.md`, §6.

use super::{
    error::{checked_product, try_vec, WhirError, WhirPart},
    params::{Level, Queries},
};
use crate::commitment::squeeze_bytes;
use jolt_field::{CanonicalBytes, CanonicalEncoding, F128, F192};
use jolt_transcript::{Label, Transcript};

/// Commit-phase root label.
pub const COMMIT_LABEL: &[u8] = b"whir_commit";
/// Out-of-domain lane evaluation label.
pub const OOD_LABEL: &[u8] = b"whir_ood";
/// Opening-phase label, preceding the bridge challenge.
pub const OPEN_LABEL: &[u8] = b"whir_open";
/// Sumcheck round label, shared by lane and closing rounds.
pub const ROUND_LABEL: &[u8] = b"whir_round";
/// Later-level root label.
pub const ROOT_LABEL: &[u8] = b"whir_root";
/// Final-message label.
pub const FINAL_LABEL: &[u8] = b"whir_final";

/// Absorbs one label and one concatenation of canonical elements. Up to 32
/// elements use stack storage; larger explicit schedules reserve checked bytes.
pub fn append_elements<T: Transcript<Challenge = F128>>(
    transcript: &mut T,
    label: &'static [u8],
    values: &[F192],
) -> Result<(), WhirError> {
    let len = checked_product(WhirPart::FinalValues, &[values.len(), F192::NUM_BYTES])?;
    let mut stack = [0; 32 * 24];
    let mut heap;
    let bytes = if let Some(bytes) = stack.get_mut(..len) {
        bytes
    } else {
        heap = try_vec(WhirPart::FinalValues, len)?;
        heap.resize(len, 0);
        heap.as_mut_slice()
    };
    for (value, encoding) in values.iter().zip(bytes.chunks_exact_mut(F192::NUM_BYTES)) {
        value.to_bytes_le(encoding);
    }
    transcript.append(&Label(label));
    transcript.append_bytes(bytes);
    Ok(())
}

/// Coefficients of the entering sample, ascending distinct queries, and optional
/// commit sample. Consume one query coefficient per position before the commit
/// coefficient; its consuming method closes the ordered walk.
pub struct ClaimCoefficients {
    lambda: F192,
    next: F192,
}

impl ClaimCoefficients {
    /// The entering sample has coefficient lambda; the first query has lambda².
    pub fn new(lambda: F192) -> Self {
        Self {
            lambda,
            next: lambda * lambda,
        }
    }

    /// Coefficient of the new oracle's out-of-domain sample.
    pub fn sample(&self) -> F192 {
        self.lambda
    }

    /// Coefficient of the next distinct query, in ascending position order.
    pub fn next_query(&mut self) -> F192 {
        let coefficient = self.next;
        self.next *= self.lambda;
        coefficient
    }

    /// Coefficient of the commit sample after every distinct query is consumed.
    pub fn commit_sample(self) -> F192 {
        self.next
    }
}

/// Draws one element with two 16-byte draws, discarding the last eight bytes.
#[expect(
    clippy::expect_used,
    reason = "every 24-byte string is a canonical F192 encoding"
)]
pub fn draw_element<T: Transcript<Challenge = F128>>(transcript: &mut T) -> F192 {
    let mut bytes = [0; 24];
    squeeze_bytes(transcript, &mut bytes);
    F192::from_bytes_le_checked(&bytes).expect("fixed-width F192 encoding")
}

/// Draws coordinates in index order, keeping each element's draw boundary.
pub fn draw_point<T: Transcript<Challenge = F128>>(
    transcript: &mut T,
    len: usize,
) -> Result<Vec<F192>, WhirError> {
    let mut point = try_vec(WhirPart::Rounds, len)?;
    for _ in 0..len {
        point.push(draw_element(transcript));
    }
    Ok(point)
}

/// Draws positions with replacement in one byte call, then sorts and deduplicates.
/// An all-position level consumes no transcript bytes.
pub fn draw_positions<T: Transcript<Challenge = F128>>(
    transcript: &mut T,
    level: &Level,
) -> Result<Vec<usize>, WhirError> {
    let exponent = u32::try_from(level.d).map_err(|_| WhirError::LengthOverflow {
        part: WhirPart::Leaves,
    })?;
    if exponent > u32::BITS {
        return Err(WhirError::LengthOverflow {
            part: WhirPart::Leaves,
        });
    }
    let domain_len = 1usize
        .checked_shl(exponent)
        .ok_or(WhirError::LengthOverflow {
            part: WhirPart::Leaves,
        })?;
    match level.queries {
        Queries::All => {
            let mut positions = try_vec(WhirPart::Leaves, domain_len)?;
            positions.extend(0..domain_len);
            Ok(positions)
        }
        Queries::Count(count) => {
            let count = usize::try_from(count).map_err(|_| WhirError::LengthOverflow {
                part: WhirPart::Leaves,
            })?;
            if count == 0 {
                return Err(WhirError::Shape {
                    part: WhirPart::Leaves,
                    expected: 1,
                    actual: 0,
                });
            }
            let byte_len = checked_product(WhirPart::Leaves, &[count, 4])?;
            let mut bytes = try_vec(WhirPart::Leaves, byte_len)?;
            let mut positions = try_vec(WhirPart::Leaves, count)?;
            bytes.resize(byte_len, 0);
            squeeze_bytes(transcript, &mut bytes);
            for chunk in bytes.chunks_exact(4) {
                let mut word = [0; 4];
                word.copy_from_slice(chunk);
                let raw = usize::try_from(u32::from_le_bytes(word)).map_err(|_| {
                    WhirError::LengthOverflow {
                        part: WhirPart::Leaves,
                    }
                })?;
                positions.push(raw & (domain_len - 1));
            }
            positions.sort_unstable();
            positions.dedup();
            Ok(positions)
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests assert successful challenge draws"
)]
mod tests {
    use super::{
        append_elements, draw_element, draw_point, draw_positions, ClaimCoefficients, COMMIT_LABEL,
        FINAL_LABEL, OOD_LABEL, OPEN_LABEL, ROOT_LABEL, ROUND_LABEL,
    };
    use crate::whir::{
        error::{WhirError, WhirPart},
        params::{Level, Queries},
    };
    use jolt_field::{CanonicalBytes, CanonicalEncoding, ExtField, F128, F192, F64};
    use jolt_transcript::Transcript;

    #[derive(Default)]
    struct ByteTranscript {
        draw: usize,
        fixtures: Vec<[u8; 16]>,
        absorbed: Vec<Vec<u8>>,
    }

    impl Transcript for ByteTranscript {
        type Challenge = F128;

        fn new(label: &'static [u8]) -> Self {
            Self {
                absorbed: vec![label.to_vec()],
                ..Self::default()
            }
        }

        fn append_bytes(&mut self, bytes: &[u8]) {
            self.absorbed.push(bytes.to_vec());
        }

        fn challenge(&mut self) -> F128 {
            let bytes = self.fixtures.get(self.draw).copied().unwrap_or_else(|| {
                let mut bytes = [0; 16];
                for (i, byte) in bytes.iter_mut().enumerate() {
                    *byte = (16 * self.draw + i).to_le_bytes().first().copied().unwrap();
                }
                bytes
            });
            self.draw += 1;
            F128::from_bytes_le_checked(&bytes).unwrap()
        }

        fn state(&self) -> [u8; 32] {
            [0; 32]
        }
    }

    #[test]
    fn claim_coefficients_match_literal_sample_query_commit_exponents() {
        let lambda = F192::lift_base(F64::from_raw(2));
        for (queries, literals) in [
            (0, vec![2, 4]),
            (1, vec![2, 4, 8]),
            (5, vec![2, 4, 8, 16, 32, 64, 128]),
        ] {
            let mut coefficients = ClaimCoefficients::new(lambda);
            let mut actual = vec![coefficients.sample()];
            for _ in 0..queries {
                actual.push(coefficients.next_query());
            }
            actual.push(coefficients.commit_sample());
            let expected: Vec<_> = literals
                .into_iter()
                .map(|raw| F192::lift_base(F64::from_raw(raw)))
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn element_absorption_keeps_literal_single_body_across_stack_bound() {
        let element = F192::from_base_fn(|i| F64::from_raw([1, 2, 3].get(i).copied().unwrap()));
        let encoded = [
            1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
        ];
        let mut label = [0; 32];
        label.get_mut(..10).unwrap().copy_from_slice(b"whir_round");
        for count in [0, 1, 2, 32, 33] {
            let mut transcript = ByteTranscript::default();
            append_elements(&mut transcript, ROUND_LABEL, &vec![element; count]).unwrap();
            assert_eq!(
                transcript.absorbed,
                vec![label.to_vec(), encoded.repeat(count)]
            );
            assert_eq!(transcript.draw, 0);
        }
    }

    #[test]
    fn element_and_point_keep_literal_coordinate_boundaries() {
        let mut transcript = ByteTranscript::default();
        let point = draw_point(&mut transcript, 2).unwrap();
        let expected = [
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23,
            ],
            [
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                53, 54, 55,
            ],
        ];
        for (element, expected) in point.iter().zip(expected) {
            let mut bytes = [0; 24];
            element.to_bytes_le(&mut bytes);
            assert_eq!(bytes, expected);
        }
        assert_eq!(transcript.draw, 4);
        assert!(transcript.absorbed.is_empty());
        let mut bytes = [0; 24];
        draw_element(&mut transcript).to_bytes_le(&mut bytes);
        assert_eq!(
            bytes,
            [
                64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                85, 86, 87
            ]
        );
        assert_eq!(transcript.draw, 6);
    }

    #[test]
    fn positions_match_literal_d7_q5_bytes() {
        let mut transcript = ByteTranscript {
            fixtures: vec![
                [
                    0xef, 0xbe, 0xad, 0xde, 0x67, 0x45, 0x23, 0x01, 0x81, 0x00, 0xff, 0xff, 0x67,
                    0x00, 0x00, 0x80,
                ],
                [
                    0x04, 0x03, 0x02, 0x01, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
            ],
            ..ByteTranscript::default()
        };
        let level = Level {
            k: 0,
            c: 6,
            d: 7,
            queries: Queries::Count(5),
            leaf_bytes: 16,
        };
        assert_eq!(
            draw_positions(&mut transcript, &level).unwrap(),
            [1, 4, 103, 111]
        );
        assert_eq!(transcript.draw, 2);
        assert!(transcript.absorbed.is_empty());
        let mut next = [0; 24];
        draw_element(&mut transcript).to_bytes_le(&mut next);
        assert_eq!(next.first(), Some(&32));
        assert_eq!(transcript.draw, 4);
    }

    #[test]
    fn all_positions_draw_nothing_including_smallest_geometry() {
        let mut transcript = ByteTranscript::default();
        let level = Level {
            k: 0,
            c: 2,
            d: 3,
            queries: Queries::All,
            leaf_bytes: 16,
        };
        assert_eq!(
            draw_positions(&mut transcript, &level).unwrap(),
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert!(draw_point(&mut transcript, 0).unwrap().is_empty());
        assert_eq!(transcript.draw, 0);
        assert!(transcript.absorbed.is_empty());
    }

    #[test]
    fn labels_are_literal_and_disjoint_from_front_end() {
        let labels = [
            COMMIT_LABEL,
            OOD_LABEL,
            OPEN_LABEL,
            ROUND_LABEL,
            ROOT_LABEL,
            FINAL_LABEL,
        ];
        let expected: [&[u8]; 6] = [
            b"whir_commit",
            b"whir_ood",
            b"whir_open",
            b"whir_round",
            b"whir_root",
            b"whir_final",
        ];
        assert_eq!(labels, expected);
        let front_end: [&[u8]; 9] = [
            b"params",
            b"statement",
            b"inputs",
            b"outputs",
            b"program",
            b"final_pc",
            b"sumcheck_claim",
            b"sumcheck_poly",
            b"opening_claim",
        ];
        for label in labels {
            assert!(!front_end.contains(&label));
            assert_eq!(labels.iter().filter(|other| **other == label).count(), 1);
        }
    }

    #[test]
    fn challenge_lengths_and_dimensions_fail_before_draws() {
        let mut transcript = ByteTranscript::default();
        assert_eq!(
            draw_point(&mut transcript, usize::MAX),
            Err(WhirError::LengthOverflow {
                part: WhirPart::Rounds
            })
        );
        for d in [33, usize::MAX] {
            let level = Level {
                k: 0,
                c: 2,
                d,
                queries: Queries::Count(5),
                leaf_bytes: 16,
            };
            assert_eq!(
                draw_positions(&mut transcript, &level),
                Err(WhirError::LengthOverflow {
                    part: WhirPart::Leaves
                })
            );
        }
        let level = Level {
            k: 0,
            c: 2,
            d: 3,
            queries: Queries::Count(0),
            leaf_bytes: 16,
        };
        assert_eq!(
            draw_positions(&mut transcript, &level),
            Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: 1,
                actual: 0
            })
        );
        assert_eq!(transcript.draw, 0);
    }
}
