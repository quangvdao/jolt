//! Commit-time encoding, authentication and lane sample from specification §9.

use crate::measure::{self, Event, Phase, ReleasePoint};
use crate::{induce::equality_table, merkle::MerkleTree, ntt::Encoder};
use jolt_field::{Accumulator, One, WithAccumulator, F128, F192, F64};
use jolt_rv64i_verifier::{
    commitment::BitsGeometry,
    whir::{
        challenge::{append_elements, draw_point, COMMIT_LABEL, OOD_LABEL},
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

/// Observes joined phases and release points for the WHIR benchmark runner.
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
    let first = schedule.validate_geometry(geometry)?;
    let lanes = initial_lanes(first)?;
    let count = power(WhirPart::Rows, geometry.log_T)?;
    if rows.len() != count {
        return Err(WhirError::Shape {
            part: WhirPart::Rows,
            expected: count,
            actual: rows.len(),
        });
    }
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
    let rows_per_position = (lanes / 2).max(1);
    let weights_per_position = if lanes == 1 { 2 } else { 1 };
    let sums = rows
        .par_chunks_exact(rows_per_position)
        .zip(weights.par_chunks_exact(weights_per_position))
        .with_min_len((1024 / rows_per_position).max(1))
        .fold(
            || {
                let mut accumulators = try_vec(WhirPart::LaneValues, lanes)?;
                accumulators.resize(lanes, ProductAccumulator::default());
                Ok::<_, WhirError>(accumulators)
            },
            |accumulators, (rows, weights)| {
                let mut accumulators = accumulators?;
                if lanes == 1 {
                    for (pair, &weight) in rows[0].chunks_exact(2).zip(weights) {
                        accumulators[0].fmadd_base_pair(
                            weight,
                            [F64::from_raw(pair[0]), F64::from_raw(pair[1])],
                        );
                    }
                } else {
                    for (row, sums) in rows.iter().zip(accumulators.chunks_exact_mut(2)) {
                        for (pair, sum) in row.chunks_exact(2).zip(sums) {
                            sum.fmadd_base_pair(
                                weights[0],
                                [F64::from_raw(pair[0]), F64::from_raw(pair[1])],
                            );
                        }
                    }
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

pub(crate) fn initial_lanes(first: &Level) -> Result<usize, WhirError> {
    let lanes = first.lanes()?;
    if lanes > 32 {
        return Err(WhirError::Shape {
            part: WhirPart::LaneValues,
            expected: 32,
            actual: lanes,
        });
    }
    Ok(lanes)
}

pub(crate) fn power(part: WhirPart, exponent: usize) -> Result<usize, WhirError> {
    let exponent = u32::try_from(exponent).map_err(|_| WhirError::LengthOverflow { part })?;
    1usize
        .checked_shl(exponent)
        .ok_or(WhirError::LengthOverflow { part })
}
