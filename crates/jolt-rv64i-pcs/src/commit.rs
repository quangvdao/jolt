//! Commit-time encoding, authentication and lane sample from specification §9.

use crate::measure::{self, Event, Phase, ReleasePoint};
use crate::{induce::equality_table, merkle::MerkleTree, ntt::Encoder};
use jolt_field::{Accumulator, CanonicalBytes, One, WithAccumulator, F128, F192, F64};
use jolt_rv64i_verifier::{
    commitment::BitsGeometry,
    whir::{
        challenge::{draw_point, COMMIT_LABEL, OOD_LABEL},
        code::DomainTable,
        error::{checked_product, try_vec, WhirError, WhirPart},
        params::{Level, Schedule},
        wire::WhirCommitment,
    },
};
use jolt_transcript::{Label, Transcript};
use rayon::prelude::*;
use std::sync::Arc;

type ProductAccumulator = <F192 as WithAccumulator>::Accumulator;

/// Commit state consumed by one opening; rows share the caller's allocation.
pub struct ProverState {
    pub(crate) rows: Arc<[[u64; 4]]>,
    pub(crate) codeword: Vec<F64>,
    pub(crate) tree: MerkleTree,
    pub(crate) point: Vec<F192>,
    pub(crate) lane_values: Vec<F192>,
    pub(crate) geometry: BitsGeometry,
}

/// The transcript must bind the geometry before this call.
pub fn commit<T: Transcript<Challenge = F128>>(
    geometry: BitsGeometry,
    rows: &Arc<[[u64; 4]]>,
    transcript: &mut T,
) -> Result<(WhirCommitment, ProverState), WhirError> {
    commit_impl(
        geometry,
        Schedule::new(geometry)?,
        rows,
        transcript,
        &mut |_| {},
    )
}

/// Explicit test schedule with at most 32 initial lanes; no soundness claim is
/// made for its parameters.
#[cfg(any(test, feature = "test-utils"))]
pub fn commit_with_schedule<T: Transcript<Challenge = F128>>(
    geometry: BitsGeometry,
    schedule: Schedule,
    rows: &Arc<[[u64; 4]]>,
    transcript: &mut T,
) -> Result<(WhirCommitment, ProverState), WhirError> {
    commit_impl(geometry, schedule, rows, transcript, &mut |_| {})
}

/// Observes joined phases and release points for the measurement example.
#[cfg(feature = "test-utils")]
pub fn commit_observed<T: Transcript<Challenge = F128>>(
    geometry: BitsGeometry,
    rows: &Arc<[[u64; 4]]>,
    transcript: &mut T,
    observer: &mut impl FnMut(Event),
) -> Result<(WhirCommitment, ProverState), WhirError> {
    commit_impl(
        geometry,
        Schedule::new(geometry)?,
        rows,
        transcript,
        observer,
    )
}

fn commit_impl<T: Transcript<Challenge = F128>>(
    geometry: BitsGeometry,
    schedule: Schedule,
    rows: &Arc<[[u64; 4]]>,
    transcript: &mut T,
    observer: &mut impl FnMut(Event),
) -> Result<(WhirCommitment, ProverState), WhirError> {
    let first = validate_schedule(geometry, &schedule)?;
    let count = power(WhirPart::Rows, geometry.log_T)?;
    if rows.len() != count {
        return Err(WhirError::Shape {
            part: WhirPart::Rows,
            expected: count,
            actual: rows.len(),
        });
    }
    let lanes = first.lanes()?;
    let (_table, codeword) = measure::run(observer, Phase::LevelZeroEncode, || {
        let table = DomainTable::new(first.c, first.d)?;
        for level in schedule.levels() {
            table.validate_dimensions(level.c, level.d)?;
        }
        let codeword = Encoder::new(&table, first.c, first.d, lanes)?.encode_rows(rows)?;
        Ok::<_, WhirError>((table, codeword))
    })?;
    let tree = measure::run(observer, Phase::LevelZeroTree, || {
        MerkleTree::build_canonical(&codeword, checked_product(WhirPart::Leaves, &[lanes, 2])?)
    })?;
    observer(Event::Start(Phase::CommitSample));
    let root = *tree.root();
    transcript.append(&Label(COMMIT_LABEL));
    transcript.append_bytes(&root);
    let point = draw_point(transcript, first.c)?;
    let weights = equality_table(&point, F192::one())?;
    let sums = rows
        .par_iter()
        .enumerate()
        .with_min_len(1024)
        .fold(
            || {
                let mut accumulators = try_vec(WhirPart::LaneValues, lanes)?;
                accumulators.resize(lanes, ProductAccumulator::default());
                Ok::<_, WhirError>(accumulators)
            },
            |accumulators, (index, &[a0, a1, b0, b1])| {
                let mut accumulators = accumulators?;
                for (half, pair) in [[a0, a1], [b0, b1]].into_iter().enumerate() {
                    let symbol = index * 2 + half;
                    accumulators[symbol % lanes]
                        .fmadd_base_pair(weights[symbol / lanes], pair.map(F64::from_raw));
                }
                Ok(accumulators)
            },
        )
        .try_reduce(Vec::new, |mut sums, other| {
            if sums.is_empty() {
                return Ok(other);
            }
            for (sum, other) in sums.iter_mut().zip(other) {
                sum.merge(other);
            }
            Ok(sums)
        })?;
    let mut lane_values = try_vec(WhirPart::LaneValues, lanes)?;
    lane_values.extend(sums.into_iter().take(lanes).map(Accumulator::reduce));
    drop(weights);
    append_elements(transcript, OOD_LABEL, &lane_values)?;
    let mut retained_values = try_vec(WhirPart::LaneValues, lanes)?;
    retained_values.extend_from_slice(&lane_values);
    observer(Event::End(Phase::CommitSample));
    observer(Event::Released(ReleasePoint::Commit));
    Ok((
        WhirCommitment { root, lane_values },
        ProverState {
            rows: Arc::clone(rows),
            codeword,
            tree,
            point,
            lane_values: retained_values,
            geometry,
        },
    ))
}

pub(crate) fn validate_schedule(
    geometry: BitsGeometry,
    schedule: &Schedule,
) -> Result<&Level, WhirError> {
    if !(1..=32).contains(&geometry.log_T) {
        return Err(WhirError::UnsupportedGeometry {
            log_T: geometry.log_T,
        });
    }
    if schedule.mu() != geometry.log_T + 1 {
        return Err(WhirError::GeometryMismatch {
            expected: BitsGeometry {
                log_T: schedule.mu().saturating_sub(1),
            },
            actual: geometry,
        });
    }
    let first = schedule.levels().first().ok_or(WhirError::Shape {
        part: WhirPart::Levels,
        expected: 1,
        actual: 0,
    })?;
    if first.lanes()? > 32 {
        return Err(WhirError::Shape {
            part: WhirPart::LaneValues,
            expected: 32,
            actual: first.lanes()?,
        });
    }
    Ok(first)
}

pub(crate) fn power(part: WhirPart, exponent: usize) -> Result<usize, WhirError> {
    let exponent = u32::try_from(exponent).map_err(|_| WhirError::LengthOverflow { part })?;
    1usize
        .checked_shl(exponent)
        .ok_or(WhirError::LengthOverflow { part })
}

pub(crate) fn append_elements<T: Transcript<Challenge = F128>>(
    transcript: &mut T,
    label: &'static [u8],
    values: &[F192],
) -> Result<(), WhirError> {
    let mut bytes = try_vec(
        WhirPart::FinalValues,
        checked_product(WhirPart::FinalValues, &[values.len(), F192::NUM_BYTES])?,
    )?;
    for value in values {
        let mut encoded = [0; 24];
        value.to_bytes_le(&mut encoded);
        bytes.extend_from_slice(&encoded);
    }
    transcript.append(&Label(label));
    transcript.append_bytes(&bytes);
    Ok(())
}
