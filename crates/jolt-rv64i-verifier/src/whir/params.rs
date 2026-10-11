// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
// CREDIT: https://github.com/bcc-research/bolt-rs, MIT.
// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Ported from bolt-rs (https://github.com/bcc-research/bolt-rs,
// `whir_recursive.rs`).
// Modified from leanVM crates/pcs/src/whir_config.rs at revision
// 48a904208d682848dac0e18ef8b01ebfc40df9ad: frozen parameters and exact certificates.
// See crates/jolt-rv64i-pcs/THIRD_PARTY_NOTICES.md.

//! Geometry and frozen query counts of `specs/rv64i-binary-commitment.md`, §5.
//! The exact derivation and security certificates run only in the tests.

use super::error::{checked_product, try_vec, WhirError, WhirPart};
use crate::commitment::BitsGeometry;

/// Every position is opened, or this many independent positions are drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Queries {
    /// Open every position, with no position challenges.
    All,
    /// Draw positions with replacement; repeated positions are opened once.
    Count(u32),
}

/// One committed oracle, with low lane variables and high message variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Level {
    /// Number of lane variables folded at this level.
    pub k: usize,
    /// Number of variables in each lane's message.
    pub c: usize,
    /// Logarithm of the number of encoded positions.
    pub d: usize,
    /// Position sampling rule, fixed by the geometry.
    pub queries: Queries,
    /// Canonical bytes in one leaf: 16 per lane at level 0, 24 thereafter.
    pub leaf_bytes: usize,
}

impl Level {
    /// Number of lanes, rejecting an exponent that cannot fit in `usize`.
    pub fn lanes(&self) -> Result<usize, WhirError> {
        Self::checked_lanes(self.k)
    }

    fn checked_lanes(k: usize) -> Result<usize, WhirError> {
        let exponent = u32::try_from(k).map_err(|_| WhirError::LengthOverflow {
            part: WhirPart::Leaves,
        })?;
        1usize
            .checked_shl(exponent)
            .ok_or(WhirError::LengthOverflow {
                part: WhirPart::Leaves,
            })
    }

    fn new(
        k: usize,
        c: usize,
        d: usize,
        queries: Queries,
        symbol_bytes: usize,
    ) -> Result<Self, WhirError> {
        let leaf_bytes =
            checked_product(WhirPart::Leaves, &[Self::checked_lanes(k)?, symbol_bytes])?;
        Ok(Self {
            k,
            c,
            d,
            queries,
            leaf_bytes,
        })
    }
}

/// Validated schedule; `levels().len()` is the number of levels `R`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    mu: usize,
    res: usize,
    levels: Vec<Level>,
}

impl Schedule {
    /// Frozen production parameters for cycle exponents in `1..=32`.
    pub fn new(geometry: BitsGeometry) -> Result<Self, WhirError> {
        let t = geometry.log_T;
        if !(1..=32).contains(&t) {
            return Err(WhirError::UnsupportedGeometry { log_T: t });
        }
        let query_counts = QUERY_COUNTS
            .get(t - 1)
            .ok_or(WhirError::UnsupportedGeometry { log_T: t })?;
        let mu = t + 1;
        let mut k = 5.min(mu - 2);
        let mut c = mu - k;
        let mut d = c + 1;
        let mut levels = try_vec(WhirPart::Levels, query_counts.len())?;
        for (i, queries) in query_counts.iter().copied().enumerate() {
            if i != 0 {
                k = if t >= 31 && c <= 8 { 3 } else { 4 };
                c -= k;
                d -= 1;
            }
            levels.push(Level::new(k, c, d, queries, if i == 0 { 16 } else { 24 })?);
        }
        Ok(Self { mu, res: c, levels })
    }

    /// Checks a supported geometry against this schedule, returning its first
    /// level. A mismatch reports the schedule's geometry as expected.
    pub fn validate_geometry(&self, geometry: BitsGeometry) -> Result<&Level, WhirError> {
        if !(1..=32).contains(&geometry.log_T) {
            return Err(WhirError::UnsupportedGeometry {
                log_T: geometry.log_T,
            });
        }
        if self.mu != geometry.log_T + 1 {
            return Err(WhirError::GeometryMismatch {
                expected: BitsGeometry {
                    log_T: self.mu.saturating_sub(1),
                },
                actual: geometry,
            });
        }
        self.levels.first().ok_or(WhirError::Shape {
            part: WhirPart::Levels,
            expected: 1,
            actual: 0,
        })
    }

    /// Number of packed-table variables, `log_T + 1`.
    pub fn mu(&self) -> usize {
        self.mu
    }

    /// Number of variables in the final message.
    pub fn res(&self) -> usize {
        self.res
    }

    /// The committed levels, in transcript order.
    pub fn levels(&self) -> &[Level] {
        &self.levels
    }

    /// Variables bound before level `i`; `i = R` is the closing-round offset.
    /// Returns `None` if `i > R`.
    pub fn offset(&self, i: usize) -> Option<usize> {
        self.levels
            .get(..i)
            .map(|levels| levels.iter().map(|level| level.k).sum())
    }

    /// Explicit test schedule with chain validation, carrying no soundness claim.
    /// Dimensions, rates and query counts may differ from production parameters;
    /// leaf sizes and allocation lengths must still be representable.
    #[cfg(feature = "test-utils")]
    pub fn from_levels(
        geometry: BitsGeometry,
        levels: &[(usize, usize, usize, Queries)],
    ) -> Result<Self, WhirError> {
        if levels.is_empty() {
            return Err(WhirError::Shape {
                part: WhirPart::Levels,
                expected: 1,
                actual: 0,
            });
        }
        let mu = geometry
            .log_T
            .checked_add(1)
            .ok_or(WhirError::LengthOverflow {
                part: WhirPart::Rounds,
            })?;
        let mut remaining = mu;
        let mut result = try_vec(WhirPart::Levels, levels.len())?;
        for (i, &(k, c, d, queries)) in levels.iter().enumerate() {
            let actual = k.checked_add(c).ok_or(WhirError::LengthOverflow {
                part: WhirPart::Rounds,
            })?;
            if actual != remaining {
                return Err(WhirError::Shape {
                    part: WhirPart::Rounds,
                    expected: remaining,
                    actual,
                });
            }
            result.push(Level::new(k, c, d, queries, if i == 0 { 16 } else { 24 })?);
            remaining = c;
        }
        Ok(Self {
            mu,
            res: remaining,
            levels: result,
        })
    }
}

// Spec §5; integer constants are the only query derivation in production.
const QUERY_COUNTS: [&[Queries]; 32] = [
    &[Queries::All],
    &[Queries::All],
    &[Queries::All],
    &[Queries::All],
    &[Queries::All],
    &[Queries::All],
    &[Queries::All],
    &[Queries::All],
    &[Queries::All],
    &[Queries::All, Queries::Count(59)],
    &[Queries::Count(254), Queries::Count(62)],
    &[Queries::Count(256), Queries::Count(63)],
    &[Queries::Count(257), Queries::Count(64)],
    &[Queries::Count(257), Queries::Count(64), Queries::Count(35)],
    &[Queries::Count(258), Queries::Count(64), Queries::Count(36)],
    &[Queries::Count(258), Queries::Count(65), Queries::Count(37)],
    &[Queries::Count(258), Queries::Count(65), Queries::Count(37)],
    &[
        Queries::Count(259),
        Queries::Count(65),
        Queries::Count(37),
        Queries::Count(25),
    ],
    &[
        Queries::Count(259),
        Queries::Count(65),
        Queries::Count(37),
        Queries::Count(26),
    ],
    &[
        Queries::Count(259),
        Queries::Count(65),
        Queries::Count(37),
        Queries::Count(26),
    ],
    &[
        Queries::Count(260),
        Queries::Count(65),
        Queries::Count(37),
        Queries::Count(26),
    ],
    &[
        Queries::Count(260),
        Queries::Count(65),
        Queries::Count(37),
        Queries::Count(26),
        Queries::Count(20),
    ],
    &[
        Queries::Count(261),
        Queries::Count(65),
        Queries::Count(37),
        Queries::Count(26),
        Queries::Count(20),
    ],
    &[
        Queries::Count(262),
        Queries::Count(65),
        Queries::Count(37),
        Queries::Count(26),
        Queries::Count(20),
    ],
    &[
        Queries::Count(262),
        Queries::Count(65),
        Queries::Count(37),
        Queries::Count(26),
        Queries::Count(20),
    ],
    &[
        Queries::Count(263),
        Queries::Count(65),
        Queries::Count(37),
        Queries::Count(26),
        Queries::Count(20),
        Queries::Count(16),
    ],
    &[
        Queries::Count(264),
        Queries::Count(65),
        Queries::Count(37),
        Queries::Count(26),
        Queries::Count(20),
        Queries::Count(17),
    ],
    &[
        Queries::Count(266),
        Queries::Count(65),
        Queries::Count(37),
        Queries::Count(26),
        Queries::Count(21),
        Queries::Count(17),
    ],
    &[
        Queries::Count(267),
        Queries::Count(65),
        Queries::Count(38),
        Queries::Count(26),
        Queries::Count(21),
        Queries::Count(17),
    ],
    &[
        Queries::Count(268),
        Queries::Count(66),
        Queries::Count(38),
        Queries::Count(27),
        Queries::Count(21),
        Queries::Count(17),
        Queries::Count(14),
    ],
    &[
        Queries::Count(271),
        Queries::Count(66),
        Queries::Count(38),
        Queries::Count(27),
        Queries::Count(21),
        Queries::Count(17),
        Queries::Count(15),
    ],
    &[
        Queries::Count(273),
        Queries::Count(66),
        Queries::Count(38),
        Queries::Count(27),
        Queries::Count(21),
        Queries::Count(17),
        Queries::Count(15),
    ],
];

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "literal vectors and exact certificate tests use checked fixtures"
)]
mod tests {
    use super::{Level, Queries, Schedule};
    use crate::commitment::BitsGeometry;
    use crate::whir::error::{WhirError, WhirPart};
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::{One, Zero};

    // Each tuple is (k, c, d, queries, m), transcribed from the full §5 table.
    type SpecLevel = (usize, usize, usize, Queries, u64);
    const SPEC_LEVELS: [&[SpecLevel]; 32] = [
        &[(0, 2, 3, Queries::All, 3)],
        &[(1, 2, 3, Queries::All, 3)],
        &[(2, 2, 3, Queries::All, 3)],
        &[(3, 2, 3, Queries::All, 3)],
        &[(4, 2, 3, Queries::All, 3)],
        &[(5, 2, 3, Queries::All, 3)],
        &[(5, 3, 4, Queries::All, 3)],
        &[(5, 4, 5, Queries::All, 3)],
        &[(5, 5, 6, Queries::All, 3)],
        &[
            (5, 6, 7, Queries::All, 3),
            (4, 2, 6, Queries::Count(59), 38),
        ],
        &[
            (5, 7, 8, Queries::Count(254), 838),
            (4, 3, 7, Queries::Count(62), 45),
        ],
        &[
            (5, 8, 9, Queries::Count(256), 511),
            (4, 4, 8, Queries::Count(63), 97),
        ],
        &[
            (5, 9, 10, Queries::Count(257), 430),
            (4, 5, 9, Queries::Count(64), 63),
        ],
        &[
            (5, 10, 11, Queries::Count(257), 544),
            (4, 6, 10, Queries::Count(64), 127),
            (4, 2, 9, Queries::Count(35), 29),
        ],
        &[
            (5, 11, 12, Queries::Count(258), 341),
            (4, 7, 11, Queries::Count(64), 255),
            (4, 3, 10, Queries::Count(36), 35),
        ],
        &[
            (5, 12, 13, Queries::Count(258), 356),
            (4, 8, 12, Queries::Count(65), 43),
            (4, 4, 11, Queries::Count(37), 17),
        ],
        &[
            (5, 13, 14, Queries::Count(258), 364),
            (4, 9, 13, Queries::Count(65), 45),
            (4, 5, 12, Queries::Count(37), 23),
        ],
        &[
            (5, 14, 15, Queries::Count(259), 247),
            (4, 10, 14, Queries::Count(65), 46),
            (4, 6, 13, Queries::Count(37), 28),
            (4, 2, 12, Queries::Count(25), 16),
        ],
        &[
            (5, 15, 16, Queries::Count(259), 248),
            (4, 11, 15, Queries::Count(65), 46),
            (4, 7, 14, Queries::Count(37), 31),
            (4, 3, 13, Queries::Count(26), 8),
        ],
        &[
            (5, 16, 17, Queries::Count(259), 249),
            (4, 12, 16, Queries::Count(65), 47),
            (4, 8, 15, Queries::Count(37), 33),
            (4, 4, 14, Queries::Count(26), 12),
        ],
        &[
            (5, 17, 18, Queries::Count(260), 187),
            (4, 13, 17, Queries::Count(65), 47),
            (4, 9, 16, Queries::Count(37), 34),
            (4, 5, 15, Queries::Count(26), 14),
        ],
        &[
            (5, 18, 19, Queries::Count(260), 187),
            (4, 14, 18, Queries::Count(65), 47),
            (4, 10, 17, Queries::Count(37), 35),
            (4, 6, 16, Queries::Count(26), 16),
            (4, 2, 15, Queries::Count(20), 5),
        ],
        &[
            (5, 19, 20, Queries::Count(261), 151),
            (4, 15, 19, Queries::Count(65), 47),
            (4, 11, 18, Queries::Count(37), 35),
            (4, 7, 17, Queries::Count(26), 17),
            (4, 3, 16, Queries::Count(20), 7),
        ],
        &[
            (5, 20, 21, Queries::Count(262), 126),
            (4, 16, 20, Queries::Count(65), 47),
            (4, 12, 19, Queries::Count(37), 35),
            (4, 8, 18, Queries::Count(26), 18),
            (4, 4, 17, Queries::Count(20), 10),
        ],
        &[
            (5, 21, 22, Queries::Count(262), 126),
            (4, 17, 21, Queries::Count(65), 47),
            (4, 13, 20, Queries::Count(37), 36),
            (4, 9, 19, Queries::Count(26), 18),
            (4, 5, 18, Queries::Count(20), 12),
        ],
        &[
            (5, 22, 23, Queries::Count(263), 108),
            (4, 18, 22, Queries::Count(65), 47),
            (4, 14, 21, Queries::Count(37), 36),
            (4, 10, 20, Queries::Count(26), 19),
            (4, 6, 19, Queries::Count(20), 13),
            (4, 2, 18, Queries::Count(16), 7),
        ],
        &[
            (5, 23, 24, Queries::Count(264), 95),
            (4, 19, 23, Queries::Count(65), 47),
            (4, 15, 22, Queries::Count(37), 36),
            (4, 11, 21, Queries::Count(26), 19),
            (4, 7, 20, Queries::Count(20), 14),
            (4, 3, 19, Queries::Count(17), 3),
        ],
        &[
            (5, 24, 25, Queries::Count(266), 77),
            (4, 20, 24, Queries::Count(65), 47),
            (4, 16, 23, Queries::Count(37), 36),
            (4, 12, 22, Queries::Count(26), 19),
            (4, 8, 21, Queries::Count(21), 4),
            (4, 4, 20, Queries::Count(17), 3),
        ],
        &[
            (5, 25, 26, Queries::Count(267), 70),
            (4, 21, 25, Queries::Count(65), 47),
            (4, 17, 24, Queries::Count(38), 11),
            (4, 13, 23, Queries::Count(26), 19),
            (4, 9, 22, Queries::Count(21), 4),
            (4, 5, 21, Queries::Count(17), 3),
        ],
        &[
            (5, 26, 27, Queries::Count(268), 64),
            (4, 22, 26, Queries::Count(66), 24),
            (4, 18, 25, Queries::Count(38), 11),
            (4, 14, 24, Queries::Count(27), 6),
            (4, 10, 23, Queries::Count(21), 4),
            (4, 6, 22, Queries::Count(17), 3),
            (4, 2, 21, Queries::Count(14), 3),
        ],
        &[
            (5, 27, 28, Queries::Count(271), 52),
            (4, 23, 27, Queries::Count(66), 24),
            (4, 19, 26, Queries::Count(38), 11),
            (4, 15, 25, Queries::Count(27), 6),
            (4, 11, 24, Queries::Count(21), 4),
            (4, 7, 23, Queries::Count(17), 3),
            (3, 4, 22, Queries::Count(15), 3),
        ],
        &[
            (5, 28, 29, Queries::Count(273), 46),
            (4, 24, 28, Queries::Count(66), 24),
            (4, 20, 27, Queries::Count(38), 11),
            (4, 16, 26, Queries::Count(27), 6),
            (4, 12, 25, Queries::Count(21), 4),
            (4, 8, 24, Queries::Count(17), 3),
            (3, 5, 23, Queries::Count(15), 3),
        ],
    ];

    #[test]
    fn schedule_matches_full_spec_table() {
        for (index, expected) in SPEC_LEVELS.iter().enumerate() {
            let t = index + 1;
            let schedule = Schedule::new(BitsGeometry { log_T: t }).unwrap();
            assert_eq!(schedule.mu(), t + 1, "t={t}");
            assert_eq!(schedule.levels().len(), expected.len(), "t={t}");
            assert_eq!(schedule.res(), expected.last().unwrap().1, "t={t}");
            let mut offset = 0;
            for (i, (level, &(k, c, d, queries, _))) in
                schedule.levels().iter().zip(*expected).enumerate()
            {
                assert_eq!(
                    (level.k, level.c, level.d, level.queries),
                    (k, c, d, queries),
                    "t={t}, level={i}"
                );
                assert_eq!(level.lanes().unwrap(), 1usize << k);
                assert_eq!(
                    level.leaf_bytes,
                    (1usize << k) * if i == 0 { 16 } else { 24 }
                );
                assert_eq!(schedule.offset(i), Some(offset));
                offset += k;
            }
            assert_eq!(
                schedule.offset(expected.len()),
                Some(t + 1 - schedule.res())
            );
            assert_eq!(schedule.offset(expected.len() + 1), None);
            assert!(matches!(schedule.res(), 2..=5));
        }
    }

    #[test]
    fn schedule_handles_degenerate_sizes_and_final_fold_three() {
        let smallest = Schedule::new(BitsGeometry { log_T: 1 }).unwrap();
        assert_eq!(
            smallest.levels(),
            &[Level {
                k: 0,
                c: 2,
                d: 3,
                queries: Queries::All,
                leaf_bytes: 16
            }]
        );
        assert_eq!(smallest.offset(1), Some(0));
        assert_eq!(smallest.res(), smallest.mu());
        for t in 1..=10 {
            assert_eq!(
                Schedule::new(BitsGeometry { log_T: t }).unwrap().levels()[0].queries,
                Queries::All
            );
        }
        for t in 31..=32 {
            let schedule = Schedule::new(BitsGeometry { log_T: t }).unwrap();
            let last = schedule.levels().last().unwrap();
            assert_eq!(last.k, 3);
            assert_eq!(last.lanes().unwrap(), 8);
            assert_eq!(last.leaf_bytes, 192);
            assert_eq!(last.c, t - 27);
            assert_eq!(last.d, t - 9);
        }
    }

    #[test]
    fn schedule_rejects_unsupported_geometry() {
        for log_T in [0, 33, usize::MAX] {
            assert_eq!(
                Schedule::new(BitsGeometry { log_T }),
                Err(WhirError::UnsupportedGeometry { log_T })
            );
        }
    }

    #[test]
    fn lane_count_rejects_unrepresentable_exponents() {
        for k in [usize::BITS as usize, usize::MAX] {
            let level = Level {
                k,
                c: 2,
                d: 3,
                queries: Queries::All,
                leaf_bytes: 16,
            };
            assert_eq!(
                level.lanes(),
                Err(WhirError::LengthOverflow {
                    part: WhirPart::Leaves
                })
            );
        }
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn explicit_schedule_checks_chain_without_claiming_soundness() {
        let geometry = BitsGeometry { log_T: 11 };
        let levels = [(5, 7, 8, Queries::All), (3, 4, 7, Queries::All)];
        let schedule = Schedule::from_levels(geometry, &levels).unwrap();
        assert_eq!(schedule.mu(), 12);
        assert_eq!(schedule.res(), 4);
        assert_eq!(schedule.offset(1), Some(5));
        assert_eq!(schedule.levels()[1].lanes().unwrap(), 8);
        assert_eq!(schedule.levels()[1].leaf_bytes, 192);
        assert_eq!(
            Schedule::from_levels(geometry, &[]),
            Err(WhirError::Shape {
                part: WhirPart::Levels,
                expected: 1,
                actual: 0
            })
        );
        for (bad_levels, expected, actual) in [
            ([(4, 7, 8, Queries::All), (3, 4, 7, Queries::All)], 12, 11),
            ([(5, 7, 8, Queries::All), (2, 4, 7, Queries::All)], 7, 6),
        ] {
            assert_eq!(
                Schedule::from_levels(geometry, &bad_levels),
                Err(WhirError::Shape {
                    part: WhirPart::Rounds,
                    expected,
                    actual
                })
            );
        }
        // A rate-one level and zero drawn queries are deliberately admitted by
        // this constructor: the later test caller owns these soundness choices.
        let nonproduction = Schedule::from_levels(
            geometry,
            &[(5, 7, 7, Queries::Count(0)), (3, 4, 4, Queries::All)],
        )
        .unwrap();
        assert_eq!(nonproduction.levels()[0].d, 7);
        assert_eq!(nonproduction.levels()[0].queries, Queries::Count(0));
        assert_eq!(
            Schedule::from_levels(
                BitsGeometry { log_T: usize::MAX },
                &[(0, 0, 0, Queries::All)]
            ),
            Err(WhirError::LengthOverflow {
                part: WhirPart::Rounds
            })
        );
    }

    fn integer(value: u64) -> BigRational {
        BigRational::from_integer(BigInt::from(value))
    }

    fn power_of_two(exponent: usize) -> BigRational {
        BigRational::from_integer(BigInt::one() << exponent)
    }

    struct Certificate<'a> {
        level: &'a Level,
        rho: BigRational,
        n: u64,
        sample_variables: usize,
        batching_degree: Option<u64>,
    }

    impl Certificate<'_> {
        fn list(&self, m: u64) -> BigRational {
            integer(m) / (integer(2) * &self.rho)
        }

        fn sample_numerator(&self, m: u64) -> BigRational {
            let list = self.list(m);
            &list * (&list - integer(1)) * integer(self.sample_variables as u64) / integer(2)
        }

        fn fold_below(&self, m: u64, round: usize, bound: &BigRational) -> bool {
            // U/sqrt(rho) - B contains the only irrational term. Keeping
            // R nonnegative before squaring gives an equivalent rational test.
            let h = integer(m) + BigRational::new(BigInt::one(), BigInt::from(2));
            let h5 = BigRational::new(h.numer().pow(5), h.denom().pow(5));
            let u = integer(self.n) * (integer(2) * h5 / (integer(3) * &self.rho) + &h) + &h;
            let b = integer(self.n) * &h * integer(m + 1) / integer(m);
            let factor = power_of_two(self.level.k - round);
            let r = bound - integer(2) * self.list(m) + &factor * b;
            let lhs = &factor * u;
            r >= BigRational::zero() && &lhs * &lhs <= &self.rho * &r * &r
        }

        fn admissible(&self, m: u64) -> bool {
            let bound = power_of_two(62);
            m >= 3
                && (self.level.k == 0 || self.fold_below(m, 1, &bound))
                && self.sample_numerator(m) <= bound
                && self
                    .batching_degree
                    .is_none_or(|degree| integer(degree) * self.list(m) <= bound)
        }

        fn position_below(&self, m: u64, queries: u32) -> bool {
            let ratio =
                &self.rho * BigRational::new(BigInt::from(m + 1).pow(2), BigInt::from(m).pow(2));
            (ratio.numer().pow(queries) << 256) <= ratio.denom().pow(queries)
        }

        fn query_count(&self, m: u64) -> u32 {
            let mut hi = 1;
            while !self.position_below(m, hi) {
                hi *= 2;
            }
            let mut lo = 0;
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if self.position_below(m, mid) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            hi
        }

        fn largest_admissible(&self) -> u64 {
            assert!(self.admissible(3));
            let mut lo = 3;
            let mut hi = 6;
            while self.admissible(hi) {
                lo = hi;
                hi *= 2;
            }
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if self.admissible(mid) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            lo
        }

        fn derive(&self) -> (Queries, u64, u64) {
            let largest = self.largest_admissible();
            let count = self.query_count(largest);
            if u64::from(count) >= self.n {
                return (Queries::All, 3, largest);
            }
            let mut lo = 2;
            let mut hi = largest;
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if self.position_below(mid, count) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            (Queries::Count(count), hi, largest)
        }
    }

    #[test]
    fn exact_rule_certifies_full_parameter_table_and_security_rows() {
        let field_size = power_of_two(192);
        let algebraic_target = BigRational::one() / power_of_two(130);
        assert!(integer(127) / &field_size <= algebraic_target);
        assert!(integer(2) / &field_size <= algebraic_target);
        for (index, expected) in SPEC_LEVELS.iter().enumerate() {
            let t = index + 1;
            let schedule = Schedule::new(BitsGeometry { log_T: t }).unwrap();
            let mut previous_positions = 0;
            let mut fold_rows = 0;
            for (i, (level, &(_, _, _, expected_queries, expected_m))) in
                schedule.levels().iter().zip(*expected).enumerate()
            {
                let certificate = Certificate {
                    level,
                    rho: BigRational::new(
                        (BigInt::one() << level.c) - BigInt::one(),
                        BigInt::one() << level.d,
                    ),
                    n: 1u64 << level.d,
                    sample_variables: if i == 0 {
                        level.c
                    } else {
                        schedule.levels()[i - 1].c
                    },
                    batching_degree: if i == 0 {
                        None
                    } else {
                        Some(previous_positions + if i == 1 { 2 } else { 1 })
                    },
                };
                let (queries, m, largest) = certificate.derive();
                assert_eq!(
                    (queries, m),
                    (expected_queries, expected_m),
                    "t={t}, level={i}"
                );
                assert!(certificate.admissible(m), "t={t}, level={i}, m={m}");
                assert!(certificate.admissible(largest));
                assert!(!certificate.admissible(largest + 1));
                let raw_count = certificate.query_count(m);
                assert!(certificate.position_below(m, raw_count));
                assert!(!certificate.position_below(m, raw_count - 1));
                let best_effective = u64::from(certificate.query_count(largest)).min(certificate.n);
                let effective = u64::from(raw_count).min(certificate.n);
                assert_eq!(
                    effective, best_effective,
                    "largest admissible m gives no smaller count"
                );
                match queries {
                    Queries::All => {
                        assert_eq!(effective, certificate.n);
                        assert_eq!(m, 3);
                        assert!(BigRational::zero() <= BigRational::one() / power_of_two(128));
                    }
                    Queries::Count(count) => {
                        assert_eq!(raw_count, count);
                        assert!(certificate.position_below(m, count));
                        if m > 3 {
                            assert!(
                                !certificate.position_below(m - 1, count),
                                "no smaller m attains this count"
                            );
                        }
                    }
                }
                assert!(certificate.sample_numerator(m) / &field_size <= algebraic_target);
                if let Some(degree) = certificate.batching_degree {
                    assert!(
                        integer(degree) * certificate.list(m) / &field_size <= algebraic_target
                    );
                }
                for round in 1..=level.k {
                    assert!(certificate.fold_below(m, round, &(&algebraic_target * &field_size)));
                    fold_rows += 1;
                }
                if t == 1 {
                    assert_eq!(fold_rows, 0, "k_0 = 0 has no fold row");
                }
                previous_positions = effective;
            }
        }
    }

    #[test]
    #[expect(
        clippy::print_stdout,
        reason = "the spec requires reporting the declared-budget collision bound without a target assertion"
    )]
    fn reports_declared_budget_collision_bound() {
        let budget = BigInt::one() << 64;
        let collision = BigRational::new(&budget * (&budget - BigInt::one()), BigInt::one() << 257);
        println!("BLAKE2s-256 ideal-hash collision bound at q_h = 2^64: {collision}");
    }
}
