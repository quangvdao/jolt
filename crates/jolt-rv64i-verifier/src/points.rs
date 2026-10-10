//! Low-variable-first points for the RV64I binary protocol.
//!
//! `eq(x,y)` is the equality extension, `lt(x,y)` extends `[x < y]`, and
//! `next(x,y)` extends `[y = x + 1]` without wrap. `lift` extends the 64 bits
//! of a word. `chunk` extends a full digit selector, reconstructing digit zero
//! as the complement of the stored indicators. Lengths are checked before
//! calling the polynomial layer's infallible evaluators.
//!
//! `eq` and `lt` use the factor `1 + x + y`, which is the equality extension of
//! one coordinate only in characteristic 2. The `JoltField` bound does not
//! enforce that; the protocol instantiates every helper at `F128`.

use jolt_field::JoltField;
use jolt_poly::EqPlusOnePolynomial;
use jolt_rv64i_arith::{BitsRow, Chunk};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Invalid point dimensions, integer vertices or required column indices.
pub enum PointsError {
    #[error("point has {actual} coordinates, expected {expected}")]
    /// The provided coordinate count differs from the required count.
    Dimension { expected: usize, actual: usize },
    #[error("index {index} does not fit a point of {variables} coordinates")]
    /// The integer vertex is outside the representable point domain.
    Index { index: usize, variables: usize },
    #[error("column {column} is absent from the column values")]
    /// A chunk indicator references a column absent from the supplied values.
    MissingColumn { column: usize },
    #[error("cannot allocate an equality table for {variables} coordinates")]
    /// The table's capacity or byte size is unrepresentable, or reservation failed.
    Allocation { variables: usize },
}

/// Converts a low-variable-first point to the most-significant-variable-first order of `jolt-poly`.
pub fn to_high_to_low<F: Copy>(point: &[F]) -> Vec<F> {
    point.iter().rev().copied().collect()
}

/// Evaluates the equality extension on low-variable-first points of equal length,
/// with one multiplication per coordinate. Returns `Dimension` if their lengths differ.
pub fn eq<F: JoltField>(x: &[F], y: &[F]) -> Result<F, PointsError> {
    if x.len() != y.len() {
        return Err(PointsError::Dimension {
            expected: x.len(),
            actual: y.len(),
        });
    }
    Ok(x.iter().zip(y).map(|(x, y)| F::one() + *x + *y).product())
}

/// Evaluates equality with the integer vertex whose bit `i` is coordinate `i`.
/// Returns `Index` if the index does not fit or the point exceeds the machine-word bit width.
pub fn eq_index<F: JoltField>(point: &[F], index: usize) -> Result<F, PointsError> {
    if point.len() > usize::BITS as usize
        || (point.len() < usize::BITS as usize && index >= 1_usize << point.len())
    {
        return Err(PointsError::Index {
            index,
            variables: point.len(),
        });
    }
    Ok(point
        .iter()
        .enumerate()
        .map(|(bit, coordinate)| {
            if (index >> bit) & 1 != 0 {
                *coordinate
            } else {
                F::one() - *coordinate
            }
        })
        .product())
}

/// Allocates one equality table in low-variable-first index order, expanding it
/// serially with `2^n - 2` multiplications for a nonempty point. Rejects dimensions that cannot be
/// shifted on this host before allocating.
#[expect(
    clippy::indexing_slicing,
    reason = "the doubling loop stays inside the checked final table length"
)]
pub fn equality_table<F: JoltField>(point: &[F]) -> Result<Vec<F>, PointsError> {
    let maximum = usize::BITS as usize - 1;
    if point.len() > maximum {
        return Err(PointsError::Dimension {
            expected: maximum,
            actual: point.len(),
        });
    }
    let allocation_error = || PointsError::Allocation {
        variables: point.len(),
    };
    let shift = u32::try_from(point.len()).map_err(|_| allocation_error())?;
    let elements = 1_usize.checked_shl(shift).ok_or_else(allocation_error)?;
    let maximum_bytes = usize::try_from(isize::MAX).map_err(|_| allocation_error())?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<F>())
        .ok_or_else(allocation_error)?;
    if elements > maximum_bytes || bytes > maximum_bytes {
        return Err(allocation_error());
    }
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(elements)
        .map_err(|_| allocation_error())?;
    weights.resize(elements, F::zero());
    if let Some((first, rest)) = point.split_first() {
        weights[0] = F::one() - *first;
        weights[1] = *first;
        let mut width = 2;
        for coordinate in rest {
            for index in 0..width {
                let upper = weights[index] * *coordinate;
                weights[index + width] = upper;
                weights[index] -= upper;
            }
            width *= 2;
        }
    } else {
        weights[0] = F::one();
    }
    Ok(weights)
}

/// Materializes at most 1,024 equality weights in low-variable-first index order.
pub(crate) fn eq_table<F: JoltField>(point: &[F]) -> Result<Vec<F>, PointsError> {
    if point.len() > 10 {
        return Err(PointsError::Dimension {
            expected: 10,
            actual: point.len(),
        });
    }
    equality_table(point)
}

/// Holds the two address halves for a pass, with no intermediate prefix copies.
pub(crate) fn split_eq_tables<F: JoltField>(point: &[F]) -> Result<(Vec<F>, Vec<F>), PointsError> {
    let (low, high) = point.split_at(point.len() / 2);
    Ok((equality_table(low)?, equality_table(high)?))
}

/// Evaluates the extension of unsigned `x < y`, with low-variable-first integer bits.
/// Returns `Dimension` if the point lengths differ.
pub fn lt<F: JoltField>(x: &[F], y: &[F]) -> Result<F, PointsError> {
    if x.len() != y.len() {
        return Err(PointsError::Dimension {
            expected: x.len(),
            actual: y.len(),
        });
    }
    Ok(x.iter().zip(y).fold(F::zero(), |less, (x, y)| {
        if y.is_zero() {
            (F::one() - *x) * less
        } else if *y == F::one() {
            F::one() - *x + *x * less
        } else {
            (F::one() - *x) * *y + (F::one() + *x + *y) * less
        }
    }))
}

/// Evaluates the extension of `y = x + 1` without wrap, with low-variable-first integer bits.
/// Returns `Dimension` if the point lengths differ.
pub fn next<F: JoltField>(x: &[F], y: &[F]) -> Result<F, PointsError> {
    if x.len() != y.len() {
        return Err(PointsError::Dimension {
            expected: x.len(),
            actual: y.len(),
        });
    }
    Ok(EqPlusOnePolynomial::new(to_high_to_low(x)).evaluate(&to_high_to_low(y)))
}

/// Evaluates `next(point, index)` over the Boolean cube in low-variable-first
/// order. Shifting the equality table omits the all-ones source and forbids wrap.
pub fn next_table<F: JoltField>(point: &[F]) -> Result<Vec<F>, PointsError> {
    let mut weights = equality_table(point)?;
    for index in (0..weights.len()).rev() {
        let value = shifted_next_weight(&weights, index).ok_or(PointsError::Index {
            index,
            variables: point.len(),
        })?;
        *weights.get_mut(index).ok_or(PointsError::Index {
            index,
            variables: point.len(),
        })? = value;
    }
    Ok(weights)
}

/// Reads the nonwrapping successor weight from an existing equality table.
/// Returns `None` for an index outside that table.
pub fn shifted_next_weight<F: JoltField>(weights: &[F], index: usize) -> Option<F> {
    if index >= weights.len() {
        None
    } else if let Some(previous) = index.checked_sub(1) {
        weights.get(previous).copied()
    } else {
        Some(F::zero())
    }
}

/// A word lift against one reusable 64-entry equality table of a six-coordinate
/// low-variable-first bit point. Evaluating a word sums the weights of its set bits
/// without multiplying.
#[derive(Clone)]
pub struct WordLift<F: JoltField> {
    weights: [F; 64],
}

impl<F: JoltField> WordLift<F> {
    /// Returns `Dimension` unless the point has six coordinates.
    pub fn new(point: &[F]) -> Result<Self, PointsError> {
        if point.len() != 6 {
            return Err(PointsError::Dimension {
                expected: 6,
                actual: point.len(),
            });
        }
        let weights =
            eq_table(point)?
                .try_into()
                .map_err(|values: Vec<F>| PointsError::Dimension {
                    expected: 64,
                    actual: values.len(),
                })?;
        Ok(Self { weights })
    }

    pub fn evaluate(&self, word: u64) -> F {
        self.weights
            .iter()
            .enumerate()
            .filter(|(bit, _)| word & (1_u64 << bit) != 0)
            .map(|(_, weight)| *weight)
            .sum()
    }
}

/// Evaluates the 64-bit table of a word at a six-coordinate low-variable-first bit point.
/// Returns `Dimension` for any other point length.
pub fn lift<F: JoltField>(word: u64, point: &[F]) -> Result<F, PointsError> {
    Ok(WordLift::new(point)?.evaluate(word))
}

/// Evaluates the full digit selector in low-variable-first digit order, reconstructing digit zero from stored columns.
/// Returns `Dimension` for the wrong digit width or `MissingColumn` when a required indicator is absent.
pub fn chunk<F: JoltField>(
    descriptor: Chunk,
    point: &[F],
    columns: &[F],
) -> Result<F, PointsError> {
    ChunkWeights::new(descriptor, point)?.evaluate(columns)
}

/// Prepared digit weights shared by a pass over stored chunk indicators.
pub struct ChunkWeights<F: JoltField> {
    descriptor: Chunk,
    weights: Vec<F>,
}

impl<F: JoltField> ChunkWeights<F> {
    /// Checks the digit width before building its equality table.
    pub fn new(descriptor: Chunk, point: &[F]) -> Result<Self, PointsError> {
        let expected = usize::from(descriptor.bits());
        if point.len() != expected {
            return Err(PointsError::Dimension {
                expected,
                actual: point.len(),
            });
        }
        let mut weights = eq_table(point)?;
        if let Some((zero, differences)) = weights.split_first_mut() {
            for difference in differences {
                *difference -= *zero;
            }
        }
        Ok(Self {
            descriptor,
            weights,
        })
    }

    /// Evaluates the affine selector from packed indicators, including rows with
    /// multiple indicators set, without field multiplications.
    pub fn evaluate_packed(&self, row: &BitsRow) -> F {
        let stored = self.descriptor.stored(row);
        let Some((zero, differences)) = self.weights.split_first() else {
            return F::zero();
        };
        differences
            .iter()
            .enumerate()
            .filter(|(indicator, _)| stored & (1_u16 << indicator) != 0)
            .fold(*zero, |value, (_, difference)| value + *difference)
    }

    /// Reconstructs digit zero from the stored indicators; missing columns return an error.
    pub fn evaluate(&self, columns: &[F]) -> Result<F, PointsError> {
        let variables = usize::from(self.descriptor.bits());
        let zero = self.weights.first().copied().ok_or(PointsError::Index {
            index: 0,
            variables,
        })?;
        (1..=self.descriptor.indicators()).try_fold(zero, |sum, digit| {
            let column = usize::from(self.descriptor.start()) + digit - 1;
            let value = columns
                .get(column)
                .ok_or(PointsError::MissingColumn { column })?;
            let weight = self.weights.get(digit).copied().ok_or(PointsError::Index {
                index: digit,
                variables,
            })?;
            Ok(sum + weight * *value)
        })
    }
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

    #[test]
    fn packed_chunk_preserves_affine_multiple_indicators() {
        let chunk = Chunk::new(63, 2).unwrap();
        let weights = ChunkWeights::new(chunk, &[F128::from_raw(2), F128::from_raw(4)]).unwrap();
        for (row, literal) in [
            ([0, 0, 0, 0], 15),
            ([1_u64 << 63, 0, 0, 0], 10),
            ([0, 1, 0, 0], 12),
            ([0, 2, 0, 0], 8),
            ([1_u64 << 63, 1, 0, 0], 9),
        ] {
            assert_eq!(weights.evaluate_packed(&row), F128::from_raw(literal));
        }
    }

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
        let expected = Polynomial::new(table).evaluate(&to_high_to_low(&point));
        assert_eq!(lift(word, &point).unwrap(), expected);
        assert_eq!(WordLift::new(&point).unwrap().evaluate(word), expected);
    }

    #[test]
    fn equality_table_uses_low_variable_first_indices() {
        assert_eq!(
            equality_table(&[F128::zero(), F128::one()]).unwrap(),
            vec![F128::zero(), F128::zero(), F128::one(), F128::zero()]
        );
        assert_eq!(equality_table::<F128>(&[]).unwrap(), vec![F128::one()]);
        assert!(matches!(
            equality_table(&vec![F128::one(); usize::BITS as usize]),
            Err(PointsError::Dimension { .. })
        ));
    }

    #[test]
    fn equality_tables_reject_unrepresentable_byte_capacity() {
        let point = [F128::zero(); usize::BITS as usize - 5];
        for result in [equality_table(&point), next_table(&point)] {
            assert!(matches!(result, Err(PointsError::Allocation { .. })));
        }
    }

    #[test]
    fn split_address_tables_fit_the_twenty_two_bit_budget() {
        let (low, high) = split_eq_tables(&[F128::zero(); 22]).unwrap();
        assert_eq!((low.len(), high.len()), (2048, 2048));
        assert_eq!(low.len() + high.len(), 4096);
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
