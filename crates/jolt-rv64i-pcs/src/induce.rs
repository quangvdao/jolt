//! Equality tables and sparse query weights for the level interface.

use crate::commit::power;
use crate::measure::{self, Event, Phase};
use crate::ntt::Encoder;
use crate::parallel::{self, MIN_TASK};
use jolt_field::{Accumulator, WithAccumulator, Zero, F192};
use jolt_rv64i_verifier::whir::challenge::ClaimCoefficients;
use jolt_rv64i_verifier::whir::code::DomainTable;
use jolt_rv64i_verifier::whir::error::{try_vec, WhirError, WhirPart};
use jolt_rv64i_verifier::whir::params::Level;
use rayon::prelude::*;

type ProductAccumulator = <F192 as WithAccumulator>::Accumulator;

pub(crate) fn equality_table(point: &[F192], scale: F192) -> Result<Vec<F192>, WhirError> {
    let len = power(WhirPart::FinalValues, point.len())?;
    let mut table = try_vec(WhirPart::FinalValues, len)?;
    table.resize(len, F192::zero());
    equality_table_into(&mut table, point, scale)?;
    Ok(table)
}

/// Rebuilds a scaled table in its existing exact allocation.
pub(crate) fn equality_table_into(
    table: &mut [F192],
    point: &[F192],
    scale: F192,
) -> Result<(), WhirError> {
    let len = power(WhirPart::FinalValues, point.len())?;
    if table.len() != len {
        return Err(WhirError::Shape {
            part: WhirPart::FinalValues,
            expected: len,
            actual: table.len(),
        });
    }
    table[0] = scale;
    let mut width = 1;
    for coordinate in point {
        let (low, high) = table[..2 * width].split_at_mut(width);
        let expand = |(low, high): (&mut F192, &mut F192)| {
            *high = *low * *coordinate;
            *low += *high;
        };
        if parallel::enabled(width) {
            low.par_iter_mut()
                .zip(high.par_iter_mut())
                .with_min_len(MIN_TASK)
                .for_each(expand);
        } else {
            low.iter_mut().zip(high.iter_mut()).for_each(expand);
        }
        width *= 2;
    }
    Ok(())
}

pub(crate) fn inner_product(message: &[F192], weight: &[F192]) -> Result<F192, WhirError> {
    if message.len() != weight.len() {
        return Err(WhirError::Shape {
            part: WhirPart::FinalValues,
            expected: message.len(),
            actual: weight.len(),
        });
    }
    let accumulate = |mut sum: ProductAccumulator, (&f, &w): (&F192, &F192)| {
        sum.fmadd(f, w);
        sum
    };
    let sum = if parallel::enabled(message.len()) {
        message
            .par_iter()
            .zip(weight.par_iter())
            .with_min_len(MIN_TASK)
            .fold(ProductAccumulator::default, accumulate)
            .reduce(ProductAccumulator::default, |mut a, b| {
                a.merge(b);
                a
            })
    } else {
        message
            .iter()
            .zip(weight.iter())
            .fold(ProductAccumulator::default(), accumulate)
    };
    Ok(sum.reduce())
}

/// Claims entering together at one level interface.
pub(crate) struct ClaimWeights<'a> {
    pub(crate) sample_table: Vec<F192>,
    pub(crate) lambda: F192,
    pub(crate) previous: &'a Level,
    pub(crate) positions: &'a [usize],
    pub(crate) commit_point: Option<&'a [F192]>,
}

/// Only one domain buffer is passed to the encoder transpose.
pub(crate) fn add_claim_weights(
    weight: &mut [F192],
    table: &DomainTable,
    claims: ClaimWeights<'_>,
    observer: &mut impl FnMut(Event),
) -> Result<(), WhirError> {
    let ClaimWeights {
        sample_table,
        lambda,
        previous,
        positions,
        commit_point,
    } = claims;
    let expected = power(WhirPart::FinalValues, previous.c)?;
    for actual in [weight.len(), sample_table.len()] {
        if actual != expected {
            return Err(WhirError::Shape {
                part: WhirPart::FinalValues,
                expected,
                actual,
            });
        }
    }
    if let Some(point) = commit_point {
        if point.len() != previous.c {
            return Err(WhirError::Shape {
                part: WhirPart::CyclePoint,
                expected: previous.c,
                actual: point.len(),
            });
        }
    }
    let domain_len = power(WhirPart::Leaves, previous.d)?;
    let mut last = None;
    for &position in positions {
        if position >= domain_len {
            return Err(WhirError::Shape {
                part: WhirPart::Leaves,
                expected: domain_len,
                actual: position,
            });
        }
        if let Some(previous_position) = last {
            if position <= previous_position {
                return Err(WhirError::Shape {
                    part: WhirPart::Leaves,
                    expected: previous_position + 1,
                    actual: position,
                });
            }
        }
        last = Some(position);
    }
    let mut coefficients = ClaimCoefficients::new(lambda);
    let sample_coefficient = coefficients.sample();
    let (induced, commit_coefficient) = measure::run(observer, Phase::InducedWeights, || {
        let encoder = Encoder::new(table, previous.c, previous.d, 1)?;
        let mut domain = try_vec(WhirPart::Leaves, domain_len)?;
        domain.resize(domain_len, F192::zero());
        for &position in positions {
            domain[position] = coefficients.next_query();
        }
        let induced = encoder.transpose(&mut domain)?;
        drop(domain);
        Ok::<_, WhirError>((induced, coefficients.commit_sample()))
    })?;
    measure::run(observer, Phase::EqualityAndSamples, || {
        let commit_table = commit_point
            .map(|point| equality_table(point, commit_coefficient))
            .transpose()?;
        if let Some(commit_table) = commit_table {
            let merge =
                |(((weight, &sample), &query), &commit): (((&mut F192, &F192), &F192), &F192)| {
                    *weight += sample_coefficient * sample + query + commit;
                };
            if parallel::enabled(weight.len()) {
                weight
                    .par_iter_mut()
                    .zip(sample_table.par_iter())
                    .zip(induced.par_iter())
                    .zip(commit_table.par_iter())
                    .with_min_len(MIN_TASK)
                    .for_each(merge);
            } else {
                weight
                    .iter_mut()
                    .zip(sample_table.iter())
                    .zip(induced.iter())
                    .zip(commit_table.iter())
                    .for_each(merge);
            }
        } else {
            let merge = |((weight, &sample), &query): ((&mut F192, &F192), &F192)| {
                *weight += sample_coefficient * sample + query;
            };
            if parallel::enabled(weight.len()) {
                weight
                    .par_iter_mut()
                    .zip(sample_table.par_iter())
                    .zip(induced.par_iter())
                    .with_min_len(MIN_TASK)
                    .for_each(merge);
            } else {
                weight
                    .iter_mut()
                    .zip(sample_table.iter())
                    .zip(induced.iter())
                    .for_each(merge);
            }
        }
        Ok::<_, WhirError>(())
    })
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests assert supported table dimensions"
)]
mod tests {
    use super::{add_claim_weights, equality_table, inner_product, ClaimWeights};
    use crate::ntt::Encoder;
    use jolt_field::{ExtField, One, Zero, F192, F64};
    use jolt_rv64i_verifier::points::eq_index;
    use jolt_rv64i_verifier::whir::challenge::ClaimCoefficients;
    use jolt_rv64i_verifier::whir::code::DomainTable;
    use jolt_rv64i_verifier::whir::error::{WhirError, WhirPart};
    use jolt_rv64i_verifier::whir::params::{Level, Queries};

    #[test]
    fn scaled_equality_table_uses_low_variable_first_vertices() {
        let scale = F192::from_base_fn(|i| F64::from_raw(13 + i as u64));
        for n in 0..=14 {
            let point: Vec<_> = (0..n)
                .map(|i| F192::from_base_fn(|j| F64::from_raw(27 + i as u64 + j as u64)))
                .collect();
            let table = equality_table(&point, scale).unwrap();
            assert_eq!(table.iter().copied().sum::<F192>(), scale);
            for (vertex, &value) in table.iter().enumerate() {
                assert_eq!(value, scale * eq_index(&point, vertex).unwrap());
            }
        }
        assert_eq!(
            equality_table(&[F192::one(); usize::BITS as usize], F192::one()),
            Err(WhirError::LengthOverflow {
                part: WhirPart::FinalValues
            })
        );
    }

    #[test]
    fn induced_weight_contracts_to_ordered_claims() {
        let table = DomainTable::new(4, 6).unwrap();
        let level = Level {
            k: 2,
            c: 4,
            d: 6,
            queries: Queries::Count(5),
            leaf_bytes: 96,
        };
        let positions = [0, 7, 23, 49, 63];
        let sample = [F192::lift_base(F64::from_raw(13)); 4];
        let commit = [F192::lift_base(F64::from_raw(19)); 4];
        let message: Vec<_> = (0..16)
            .map(|i| F192::lift_base(F64::from_raw(31 + i)))
            .collect();
        let code = Encoder::new(&table, 4, 6, 1)
            .unwrap()
            .encode_extension(&message)
            .unwrap();
        let lambda = F192::from_base_fn(|i| F64::from_raw(5 + i as u64));
        for commit_point in [None, Some(commit.as_slice())] {
            let sample_table = equality_table(&sample, F192::one()).unwrap();
            let mut coefficients = ClaimCoefficients::new(lambda);
            let mut expected =
                coefficients.sample() * inner_product(&message, &sample_table).unwrap();
            for &position in &positions {
                expected += coefficients.next_query() * code[position];
            }
            if let Some(point) = commit_point {
                expected += coefficients.commit_sample()
                    * inner_product(&message, &equality_table(point, F192::one()).unwrap())
                        .unwrap();
            }
            let mut weight = vec![F192::zero(); 16];
            add_claim_weights(
                &mut weight,
                &table,
                ClaimWeights {
                    sample_table,
                    lambda,
                    previous: &level,
                    positions: &positions,
                    commit_point,
                },
                &mut |_| {},
            )
            .unwrap();
            assert_eq!(inner_product(&message, &weight).unwrap(), expected);
        }
    }

    #[test]
    fn induced_weights_validate_before_mutating_existing_weight() {
        let table = DomainTable::new(2, 3).unwrap();
        let level = Level {
            k: 1,
            c: 2,
            d: 3,
            queries: Queries::All,
            leaf_bytes: 32,
        };
        for positions in [vec![8], vec![3, 3], vec![4, 1]] {
            let mut weight = vec![F192::one(); 4];
            assert!(matches!(
                add_claim_weights(
                    &mut weight,
                    &table,
                    ClaimWeights {
                        sample_table: vec![F192::one(); 4],
                        lambda: F192::one(),
                        previous: &level,
                        positions: &positions,
                        commit_point: None
                    },
                    &mut |_| {}
                ),
                Err(WhirError::Shape {
                    part: WhirPart::Leaves,
                    ..
                })
            ));
            assert_eq!(weight, [F192::one(); 4]);
        }
    }
}
