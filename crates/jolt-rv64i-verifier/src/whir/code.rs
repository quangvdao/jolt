//! The additive domain and novel polynomial basis of §2 and §9 of
//! `specs/rv64i-binary-commitment.md`.

#[cfg(any(test, feature = "test-utils"))]
use super::error::checked_product;
use super::error::{try_vec, WhirError, WhirPart};
use jolt_field::{ExtField, Field, One, Zero, F192, F64};

/// The triangular constants `Ŵ_l(β_i)`, excluding the implicit zeroes below
/// the diagonal and ones on it. A call owns this table; it has no cache.
#[derive(Debug)]
pub struct DomainTable {
    c: usize,
    d: usize,
    rows: Vec<Vec<F64>>,
}

impl DomainTable {
    /// Builds constants for `0 <= c_0 <= d_0 <= 32`. Positions are `u32`,
    /// hence dimension 32 includes every representable position.
    pub fn new(c_0: usize, d_0: usize) -> Result<Self, WhirError> {
        Self::check_dimensions(c_0, d_0)?;
        let mut rows = try_vec(WhirPart::Levels, c_0)?;
        let mut evaluations = [F64::zero(); 32];
        for (i, value) in evaluations.iter_mut().take(d_0).enumerate() {
            *value = F64::from_raw(1u64 << i);
        }
        for l in 0..c_0 {
            let denominator = evaluations.get(l).copied().ok_or(WhirError::Shape {
                part: WhirPart::Levels,
                expected: d_0,
                actual: l,
            })?;
            let inverse = denominator.inverse().ok_or(WhirError::Shape {
                part: WhirPart::Levels,
                expected: 1,
                actual: 0,
            })?;
            let mut row = try_vec(WhirPart::Levels, d_0 - l - 1)?;
            for value in evaluations.iter_mut().take(d_0).skip(l + 1) {
                row.push(*value * inverse);
                *value *= *value + denominator;
            }
            rows.push(row);
        }
        Ok(Self {
            c: c_0,
            d: d_0,
            rows,
        })
    }

    fn check_dimensions(c: usize, d: usize) -> Result<(), WhirError> {
        if d > 32 {
            return Err(WhirError::Shape {
                part: WhirPart::Levels,
                expected: 32,
                actual: d,
            });
        }
        if c > d {
            return Err(WhirError::Shape {
                part: WhirPart::Levels,
                expected: d,
                actual: c,
            });
        }
        Ok(())
    }

    /// Checks that this table covers a level's message and domain dimensions.
    pub fn validate_dimensions(&self, c: usize, d: usize) -> Result<(), WhirError> {
        Self::check_dimensions(c, d)?;
        if c > self.c {
            return Err(WhirError::Shape {
                part: WhirPart::Levels,
                expected: self.c,
                actual: c,
            });
        }
        if d > self.d {
            return Err(WhirError::Shape {
                part: WhirPart::Levels,
                expected: self.d,
                actual: d,
            });
        }
        Ok(())
    }

    fn contains_position(&self, x: u32) -> bool {
        self.d == 32 || u64::from(x) < (1u64 << self.d)
    }

    /// Evaluates `Ŵ_l(x)` by linearity over `F_2`. Callers validate the level
    /// with [`Self::validate_dimensions`]; an uncovered argument returns zero.
    pub fn w_hat(&self, l: usize, x: u32) -> F64 {
        let Some(row) = self.rows.get(l) else {
            return F64::zero();
        };
        if !self.contains_position(x) {
            return F64::zero();
        }
        let bits = u64::from(x) >> l;
        let mut result = F64::from_raw(bits & 1);
        for (i, constant) in row.iter().enumerate() {
            if (bits >> (i + 1)) & 1 != 0 {
                result += *constant;
            }
        }
        result
    }

    /// Evaluates the multilinear extension of `W_x` at a low-variable-first
    /// point. An uncovered position, dimension, or wrong point length returns
    /// zero; verifier callers check these shapes before evaluating claims.
    pub fn query_weight(&self, c: usize, x: u32, q: &[F192]) -> F192 {
        if c > self.c || q.len() != c || !self.contains_position(x) {
            return F192::zero();
        }
        let Some((first, rest)) = q.split_first() else {
            return F192::one();
        };
        let initial = F192::one() + *first + first.mul_base(self.w_hat(0, x));
        rest.iter().enumerate().fold(initial, |weight, (l, q_l)| {
            weight * (F192::one() + *q_l + q_l.mul_base(self.w_hat(l + 1, x)))
        })
    }
}

/// Direct polynomial evaluation of position-major lanes. This definition
/// uses products over the roots of each subspace polynomial, independently
/// of the triangular table and the additive transform.
#[cfg(any(test, feature = "test-utils"))]
pub fn encode_by_definition(
    message: &[F192],
    c: usize,
    d: usize,
    lanes: usize,
) -> Result<Vec<F192>, WhirError> {
    DomainTable::check_dimensions(c, d)?;
    if lanes == 0 || !lanes.is_power_of_two() {
        return Err(WhirError::Shape {
            part: WhirPart::LaneValues,
            expected: lanes.checked_next_power_of_two().unwrap_or(usize::MAX),
            actual: lanes,
        });
    }
    let message_positions = 1usize
        .checked_shl(c as u32)
        .ok_or(WhirError::LengthOverflow {
            part: WhirPart::FinalValues,
        })?;
    let domain_positions = 1usize
        .checked_shl(d as u32)
        .ok_or(WhirError::LengthOverflow {
            part: WhirPart::Leaves,
        })?;
    let expected = checked_product(WhirPart::FinalValues, &[message_positions, lanes])?;
    if message.len() != expected {
        return Err(WhirError::Shape {
            part: WhirPart::FinalValues,
            expected,
            actual: message.len(),
        });
    }
    let len = checked_product(WhirPart::Leaves, &[domain_positions, lanes])?;
    let mut out = try_vec(WhirPart::Leaves, len)?;
    let mut inverses = try_vec(WhirPart::Levels, c)?;
    for l in 0..c {
        let beta = F64::from_raw(1u64 << l);
        let denominator = (0..(1u64 << l)).fold(F64::one(), |value, root| {
            value * (beta + F64::from_raw(root))
        });
        inverses.push(denominator.inverse().ok_or(WhirError::Shape {
            part: WhirPart::Levels,
            expected: 1,
            actual: 0,
        })?);
    }
    let mut weights = try_vec(WhirPart::Levels, c)?;
    let mut position = try_vec(WhirPart::LaneValues, lanes)?;
    position.resize(lanes, F192::zero());
    for x in 0..domain_positions {
        weights.clear();
        for (l, inverse) in inverses.iter().enumerate() {
            let subspace = (0..(1u64 << l)).fold(F64::one(), |value, root| {
                value * F64::from_raw(x as u64 ^ root)
            });
            weights.push(subspace * *inverse);
        }
        position.fill(F192::zero());
        for (w, values) in message.chunks_exact(lanes).enumerate() {
            let basis = weights
                .iter()
                .enumerate()
                .fold(F64::one(), |value, (l, weight)| {
                    if (w >> l) & 1 != 0 {
                        value * *weight
                    } else {
                        value
                    }
                });
            for (sum, coefficient) in position.iter_mut().zip(values) {
                *sum += coefficient.mul_base(basis);
            }
        }
        out.extend_from_slice(&position);
    }
    Ok(out)
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "tests use fixed small dimensions and assert successful constructors"
)]
mod tests {
    use super::{encode_by_definition, DomainTable};
    use crate::whir::error::{WhirError, WhirPart};
    use jolt_field::{ExtField, One, Zero, F192, F64};

    #[test]
    fn code_smallest_literal_words() {
        for lane in 0..2 {
            let unit = |word| {
                F192::from_base_fn(|i| {
                    if i == lane {
                        F64::from_raw(word)
                    } else {
                        F64::zero()
                    }
                })
            };
            for (message, expected) in [([1, 0], [1, 1, 1, 1]), ([0, 1], [0, 1, 2, 3])] {
                assert_eq!(
                    encode_by_definition(&message.map(unit), 1, 2, 1).unwrap(),
                    expected.map(unit)
                );
            }
            for (w, expected) in [(0, [1, 1]), (1, [2, 4]), (2, [1, 6]), (3, [2, 24])] {
                let mut message = [F192::zero(); 4];
                message[w] = unit(1);
                let output = encode_by_definition(&message, 2, 3, 1).unwrap();
                assert_eq!(output[2], unit(expected[0]));
                assert_eq!(output[4], unit(expected[1]));
            }
        }
        let table = DomainTable::new(2, 3).unwrap();
        assert_eq!(
            (0..5)
                .map(|x| table.w_hat(1, x).to_raw())
                .collect::<Vec<_>>(),
            [0, 0, 1, 1, 6]
        );
    }

    #[test]
    fn query_weight_matches_direct_equality_sum() {
        for d in 0..=6 {
            for c in 0..=d {
                let table = DomainTable::new(c, d).unwrap();
                let q: Vec<_> = (0..c)
                    .map(|l| {
                        F192::from_base_fn(|i| {
                            F64::from_raw(0x0123_4567_89ab_cdef_u64.rotate_left((l * 3 + i) as u32))
                        })
                    })
                    .collect();
                let lanes = 1usize << c;
                let mut units = vec![F192::zero(); lanes * lanes];
                for w in 0..lanes {
                    units[w * lanes + w] = F192::one();
                }
                let basis = encode_by_definition(&units, c, d, lanes).unwrap();
                for x in 0..(1usize << d) {
                    let mut sum = F192::zero();
                    for w in 0..lanes {
                        let eq = q.iter().enumerate().fold(F192::one(), |value, (l, q_l)| {
                            value
                                * if (w >> l) & 1 == 0 {
                                    F192::one() + *q_l
                                } else {
                                    *q_l
                                }
                        });
                        sum += eq * basis[x * lanes + w];
                    }
                    assert_eq!(table.query_weight(c, x as u32, &q), sum);
                }
            }
        }
    }

    #[test]
    fn code_dimensions_reject_typed_faults() {
        for (c, d, expected, actual) in [
            (2, 1, 1, 2),
            (0, 33, 32, 33),
            (usize::MAX, 32, 32, usize::MAX),
        ] {
            assert!(
                matches!(DomainTable::new(c, d), Err(WhirError::Shape { part: WhirPart::Levels, expected: got_expected, actual: got_actual }) if got_expected == expected && got_actual == actual)
            );
        }
        let table = DomainTable::new(2, 3).unwrap();
        assert_eq!(
            table.validate_dimensions(3, 3),
            Err(WhirError::Shape {
                part: WhirPart::Levels,
                expected: 2,
                actual: 3
            })
        );
        assert_eq!(
            table.validate_dimensions(2, 4),
            Err(WhirError::Shape {
                part: WhirPart::Levels,
                expected: 3,
                actual: 4
            })
        );
        assert_eq!(
            encode_by_definition(&[], 1, 2, 1),
            Err(WhirError::Shape {
                part: WhirPart::FinalValues,
                expected: 2,
                actual: 0
            })
        );
        assert_eq!(
            encode_by_definition(&[], 1, 2, 0),
            Err(WhirError::Shape {
                part: WhirPart::LaneValues,
                expected: 1,
                actual: 0
            })
        );
        assert_eq!(
            encode_by_definition(&[], 1, 2, 3),
            Err(WhirError::Shape {
                part: WhirPart::LaneValues,
                expected: 4,
                actual: 3
            })
        );
        assert_eq!(table.w_hat(usize::MAX, u32::MAX), F64::zero());
        assert_eq!(table.w_hat(1, u32::MAX), F64::zero());
        assert_eq!(table.query_weight(usize::MAX, 0, &[]), F192::zero());
        assert_eq!(table.query_weight(2, 0, &[]), F192::zero());
        assert_eq!(table.query_weight(0, u32::MAX, &[]), F192::zero());
    }

    #[test]
    fn domain_table_top_dimension_and_constant_code() {
        let table = DomainTable::new(32, 32).unwrap();
        for l in 0..32 {
            assert_eq!(table.w_hat(l, 1u32 << l), F64::one());
            assert_eq!(table.w_hat(l, (1u32 << l) - 1), F64::zero());
        }
        assert!(table.validate_dimensions(32, 32).is_ok());
        let table = DomainTable::new(0, 0).unwrap();
        assert_eq!(table.query_weight(0, 0, &[]), F192::one());
        let value = F192::from_base_fn(|i| F64::from_raw(9 + i as u64));
        assert_eq!(encode_by_definition(&[value], 0, 3, 1).unwrap(), [value; 8]);
    }
}
