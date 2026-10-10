//! Public I/O mask and word extensions for the binary-field RV64I protocol.

use jolt_field::JoltField;
use jolt_poly::EqPolynomial;
use jolt_program::preprocess::PublicIoMemory;

use crate::points::{self, PointsError, WordLift};

fn address_domain(variables: usize) -> Result<usize, PointsError> {
    let maximum = 61.min(usize::BITS as usize - 1);
    if variables > maximum {
        return Err(PointsError::Dimension {
            expected: maximum,
            actual: variables,
        });
    }
    Ok(1_usize << variables)
}

fn index_in_domain(index: u128, variables: usize, domain: usize) -> Result<usize, PointsError> {
    let index = usize::try_from(index).map_err(|_| PointsError::Index {
        index: usize::MAX,
        variables,
    })?;
    if index > domain {
        return Err(PointsError::Index { index, variables });
    }
    Ok(index)
}

pub(crate) fn validate_io(io: &PublicIoMemory, variables: usize) -> Result<(), PointsError> {
    let domain = address_domain(variables)?;
    let start = index_in_domain(io.io_mask_start, variables, domain)?;
    let end = index_in_domain(io.io_mask_end, variables, domain)?;
    if start > end {
        return Err(PointsError::Index {
            index: start,
            variables,
        });
    }
    for segment in &io.segments {
        let start = index_in_domain(segment.start_index, variables, domain)?;
        let end = start
            .checked_add(segment.words.len())
            .ok_or(PointsError::Index {
                index: usize::MAX,
                variables,
            })?;
        if end > domain {
            return Err(PointsError::Index {
                index: end,
                variables,
            });
        }
    }
    Ok(())
}

fn below<F: JoltField>(address: &[F], end: usize, domain: usize) -> Result<F, PointsError> {
    if end == domain {
        return Ok(F::one());
    }
    let vertex: Vec<F> = (0..address.len())
        .map(|bit| F::from_u64(((end >> bit) & 1) as u64))
        .collect();
    points::lt(address, &vertex)
}

/// Evaluates the interval `[io_mask_start, io_mask_end)` at an address point.
/// A boundary equal to the address-domain size extends to the constant one.
pub fn io_mask<F: JoltField>(io: &PublicIoMemory, address: &[F]) -> Result<F, PointsError> {
    validate_io(io, address.len())?;
    let domain = address_domain(address.len())?;
    let start = index_in_domain(io.io_mask_start, address.len(), domain)?;
    let end = index_in_domain(io.io_mask_end, address.len(), domain)?;
    Ok(below(address, end, domain)? + below(address, start, domain)?)
}

/// Evaluates public I/O words at `address ++ bit`, least significant bit first.
/// The two equality tables hold at most `2^(ceil(a/2)+1)` entries in total;
/// each segment word uses one word lift and two field multiplications.
pub fn val_io<F: JoltField>(
    io: &PublicIoMemory,
    address: &[F],
    bit: &[F],
) -> Result<F, PointsError> {
    validate_io(io, address.len())?;
    let domain = address_domain(address.len())?;
    if bit.len() != 6 {
        return Err(PointsError::Dimension {
            expected: 6,
            actual: bit.len(),
        });
    }
    let lift = WordLift::new(bit)?;
    let low_bits = address.len() / 2;
    let (low, high) = address.split_at(low_bits);
    let low_eq = EqPolynomial::new(low.iter().rev().copied().collect()).evaluations();
    let high_eq = EqPolynomial::new(high.iter().rev().copied().collect()).evaluations();
    let low_mask = (1_usize << low_bits) - 1;
    let mut value = F::zero();
    for segment in &io.segments {
        let start = index_in_domain(segment.start_index, address.len(), domain)?;
        for (offset, word) in segment.words.iter().enumerate() {
            let index = start + offset;
            let low_weight = low_eq.get(index & low_mask).ok_or(PointsError::Index {
                index,
                variables: address.len(),
            })?;
            let high_weight = high_eq.get(index >> low_bits).ok_or(PointsError::Index {
                index,
                variables: address.len(),
            })?;
            value += *low_weight * *high_weight * lift.evaluate(*word);
        }
    }
    Ok(value)
}

/// Returns the word placed by the public segments, or zero outside them.
/// Segment packing is owned by `PublicIoMemory::new`; this reads those words.
pub fn word_at(io: &PublicIoMemory, index: usize) -> u64 {
    io.segments
        .iter()
        .find_map(|segment| {
            let offset = (index as u128).checked_sub(segment.start_index)?;
            let offset = usize::try_from(offset).ok()?;
            segment.words.get(offset).copied()
        })
        .unwrap_or(0)
}
