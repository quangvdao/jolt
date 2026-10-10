//! Low-variable-first points for the RV64I binary protocol.
//!
//! `eq(x,y)` is the equality extension, `lt(x,y)` extends `[x < y]`, and
//! `next(x,y)` extends `[y = x + 1]` without wrap. `lift` extends the 64 bits
//! of a word. `chunk` extends a full digit selector, reconstructing digit zero
//! as the complement of the stored indicators. Lengths are checked before
//! calling the polynomial layer's infallible evaluators.

use jolt_field::JoltField;
use jolt_poly::{EqPlusOnePolynomial, EqPolynomial, LtPolynomial};
use jolt_rv64i_arith::Chunk;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PointsError {
    #[error("point has {actual} coordinates, expected {expected}")]
    Dimension { expected: usize, actual: usize },
    #[error("index {index} does not fit a point of {variables} coordinates")]
    Index { index: usize, variables: usize },
    #[error("column {column} is absent from the column values")]
    MissingColumn { column: usize },
}

pub fn to_high_to_low<F: Copy>(point: &[F]) -> Vec<F> {
    point.iter().rev().copied().collect()
}

pub fn eq<F: JoltField>(x: &[F], y: &[F]) -> Result<F, PointsError> {
    if x.len() != y.len() {
        return Err(PointsError::Dimension {
            expected: x.len(),
            actual: y.len(),
        });
    }
    Ok(EqPolynomial::mle(x, y))
}

pub fn eq_index<F: JoltField>(point: &[F], index: usize) -> Result<F, PointsError> {
    if point.len() > usize::BITS as usize
        || (point.len() < usize::BITS as usize && index >= 1_usize << point.len())
    {
        return Err(PointsError::Index {
            index,
            variables: point.len(),
        });
    }
    let vertex: Vec<F> = (0..point.len())
        .map(|i| F::from_u64(((index >> i) & 1) as u64))
        .collect();
    eq(point, &vertex)
}

/// Materializes at most 64 equality weights in low-variable-first index order.
/// The dimension bound precedes the polynomial layer's allocation and shift.
pub(crate) fn eq_table<F: JoltField>(point: &[F]) -> Result<Vec<F>, PointsError> {
    if point.len() > 6 {
        return Err(PointsError::Dimension {
            expected: 6,
            actual: point.len(),
        });
    }
    Ok(EqPolynomial::new(point.iter().rev().copied().collect()).evaluations())
}

pub fn lt<F: JoltField>(x: &[F], y: &[F]) -> Result<F, PointsError> {
    if x.len() != y.len() {
        return Err(PointsError::Dimension {
            expected: x.len(),
            actual: y.len(),
        });
    }
    let x_high: Vec<F> = x.iter().rev().copied().collect();
    let y_high: Vec<F> = y.iter().rev().copied().collect();
    Ok(LtPolynomial::evaluate(&x_high, &y_high))
}

pub fn next<F: JoltField>(x: &[F], y: &[F]) -> Result<F, PointsError> {
    if x.len() != y.len() {
        return Err(PointsError::Dimension {
            expected: x.len(),
            actual: y.len(),
        });
    }
    Ok(EqPlusOnePolynomial::new(to_high_to_low(x)).evaluate(&to_high_to_low(y)))
}

pub fn lift<F: JoltField>(word: u64, point: &[F]) -> Result<F, PointsError> {
    if point.len() != 6 {
        return Err(PointsError::Dimension {
            expected: 6,
            actual: point.len(),
        });
    }
    (0..64)
        .filter(|bit| word & (1_u64 << bit) != 0)
        .try_fold(F::zero(), |sum, bit| Ok(sum + eq_index(point, bit)?))
}

pub fn chunk<F: JoltField>(
    descriptor: Chunk,
    point: &[F],
    columns: &[F],
) -> Result<F, PointsError> {
    let expected = usize::from(descriptor.bits());
    if point.len() != expected {
        return Err(PointsError::Dimension {
            expected,
            actual: point.len(),
        });
    }
    let zero = eq_index(point, 0)?;
    (1..=descriptor.indicators()).try_fold(zero, |sum, digit| {
        let column = usize::from(descriptor.start()) + digit - 1;
        let value = columns
            .get(column)
            .ok_or(PointsError::MissingColumn { column })?;
        Ok(sum + (eq_index(point, digit)? + zero) * *value)
    })
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "algebraic tests use bounded cubes and fail by assertions"
)]
mod tests {
    use super::*;
    use jolt_field::{One, Ring, Zero, F128};
    use jolt_poly::Polynomial;

    fn vertex(index: usize, variables: usize) -> Vec<F128> {
        (0..variables)
            .map(|i| F128::from_u64(((index >> i) & 1) as u64))
            .collect()
    }
    fn seeded(seed: &mut u128, variables: usize) -> Vec<F128> {
        (0..variables)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(0xda94_2042_e4dd_58b5)
                    .wrapping_add(0x9e37_79b9_7f4a_7c15);
                F128::from_raw(*seed)
            })
            .collect()
    }
    fn basis(point: &[F128], index: usize) -> F128 {
        point
            .iter()
            .enumerate()
            .map(|(bit, value)| {
                if (index >> bit) & 1 == 1 {
                    *value
                } else {
                    F128::one() + *value
                }
            })
            .product()
    }

    #[test]
    fn point_extensions_equal_direct_boolean_cube_sums() {
        let mut seed = 0x737b_9587_4657_c5fd_5f97_008d_0da4_2e83;
        for variables in 1..=6 {
            let x = seeded(&mut seed, variables);
            let y = seeded(&mut seed, variables);
            let mut equality = F128::zero();
            let mut less = F128::zero();
            let mut successor = F128::zero();
            for a in 0..1 << variables {
                for b in 0..1 << variables {
                    let weight = basis(&x, a) * basis(&y, b);
                    if a == b {
                        equality += weight;
                    }
                    if a < b {
                        less += weight;
                    }
                    if b == a + 1 {
                        successor += weight;
                    }
                }
            }
            assert_eq!(eq(&x, &y).unwrap(), equality);
            assert_eq!(lt(&x, &y).unwrap(), less);
            assert_eq!(next(&x, &y).unwrap(), successor);
            let mass: F128 = (0..1 << variables)
                .map(|b| next(&x, &vertex(b, variables)).unwrap())
                .sum();
            assert_eq!(mass, F128::one() + basis(&x, (1 << variables) - 1));
            assert_eq!(
                next(&x, &vec![F128::zero(); variables]).unwrap(),
                F128::zero()
            );
        }
    }

    #[test]
    fn point_extensions_are_boolean_indicators() {
        for variables in 1..=4 {
            for a in 0..1 << variables {
                for b in 0..1 << variables {
                    let x = vertex(a, variables);
                    let y = vertex(b, variables);
                    assert_eq!(eq(&x, &y).unwrap(), F128::from_u64(u64::from(a == b)));
                    assert_eq!(lt(&x, &y).unwrap(), F128::from_u64(u64::from(a < b)));
                    assert_eq!(next(&x, &y).unwrap(), F128::from_u64(u64::from(b == a + 1)));
                }
            }
        }
    }

    #[test]
    fn chunk_reconstructs_full_digit_table_extension() {
        let mut seed = 0x9b43_c7de_d2a6_04ba_fe70_8b55_e455_dcfa;
        for bits in 1..=4 {
            let ch = Chunk::new(23, bits).unwrap();
            let point = seeded(&mut seed, usize::from(bits));
            let mut columns = vec![F128::zero(); 256];
            let values = seeded(&mut seed, ch.indicators());
            columns[23..23 + values.len()].copy_from_slice(&values);
            let mut full = vec![F128::one() + values.iter().copied().sum::<F128>()];
            full.extend(values);
            assert_eq!(
                chunk(ch, &point, &columns).unwrap(),
                Polynomial::new(full).evaluate(&to_high_to_low(&point))
            );
            for digit in 0..1 << bits {
                let mut one_hot = vec![F128::zero(); 256];
                if digit != 0 {
                    one_hot[23 + digit - 1] = F128::one();
                }
                let table: Vec<F128> = (0..1 << bits)
                    .map(|i| F128::from_u64(u64::from(i == digit)))
                    .collect();
                assert_eq!(
                    chunk(ch, &point, &one_hot).unwrap(),
                    Polynomial::new(table).evaluate(&to_high_to_low(&point))
                );
            }
        }
    }

    #[test]
    fn word_lift_equals_bit_table_evaluation() {
        let mut seed = 0xb51a_3f42_a967_7e01;
        let point = seeded(&mut seed, 6);
        let word = 0xf023_51a5_4802_7feb;
        let table = (0..64).map(|i| F128::from_u64((word >> i) & 1)).collect();
        assert_eq!(
            lift(word, &point).unwrap(),
            Polynomial::new(table).evaluate(&to_high_to_low(&point))
        );
    }

    #[test]
    fn point_helpers_reject_wrong_shapes() {
        let p = vec![F128::one(); 3];
        let q = vec![F128::zero(); 2];
        for result in [
            eq(&p, &q),
            lt(&p, &q),
            next(&p, &q),
            lift(3, &p),
            chunk(Chunk::new(20, 2).unwrap(), &p, &[]),
        ] {
            assert!(matches!(result, Err(PointsError::Dimension { .. })));
        }
        assert!(matches!(eq_index(&p, 8), Err(PointsError::Index { .. })));
        assert!(matches!(
            eq_index(&vec![F128::one(); usize::BITS as usize + 1], 0),
            Err(PointsError::Index { .. })
        ));
        assert!(matches!(
            chunk(Chunk::new(20, 2).unwrap(), &q, &[]),
            Err(PointsError::MissingColumn { .. })
        ));
    }
}
