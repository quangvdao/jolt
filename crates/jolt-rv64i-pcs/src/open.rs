//! Opening orchestration: bridge, lane rounds, authenticated oracles and closing.

use crate::{
    bridge::{BridgeTables, FoldedBridge},
    commit::{initial_lanes, power, ProverState},
    induce::{add_claim_weights, equality_table, inner_product, ClaimWeights},
    measure::{self, Event, Phase, ReleasePoint},
    merkle::MerkleTree,
    ntt::Encoder,
    rounds::RoundState,
};
use jolt_field::{CanonicalBytes, One, F128, F192, F64};
use jolt_rv64i_verifier::{
    commitment::BitsOpening,
    whir::{
        bridge::slice_claims,
        challenge::{
            append_elements, draw_element, draw_point, draw_positions, FINAL_LABEL, OOD_LABEL,
            OPEN_LABEL, ROOT_LABEL, ROUND_LABEL,
        },
        code::DomainTable,
        error::{checked_product, try_vec, WhirError, WhirPart},
        params::{Level, Schedule},
        wire::{WhirLevelMessage, WhirLevelProof, WhirOpeningProof},
    },
};
use jolt_transcript::{Label, Transcript};

/// Consumes the commit state after the caller has bound the complete request.
pub fn open<T: Transcript<Challenge = F128>>(
    state: ProverState,
    opening: &BitsOpening<'_>,
    transcript: &mut T,
) -> Result<WhirOpeningProof, WhirError> {
    let schedule = Schedule::new(state.geometry)?;
    open_impl(schedule, state, opening, transcript, &mut |_| {})
}

/// Use the same explicit test schedule supplied at commit.
#[cfg(any(test, feature = "test-utils"))]
pub fn open_with_schedule<T: Transcript<Challenge = F128>>(
    schedule: Schedule,
    state: ProverState,
    opening: &BitsOpening<'_>,
    transcript: &mut T,
) -> Result<WhirOpeningProof, WhirError> {
    open_impl(schedule, state, opening, transcript, &mut |_| {})
}

/// Observes joined phases and allocation release points for the WHIR benchmark runner.
#[cfg(feature = "test-utils")]
pub fn open_observed<T: Transcript<Challenge = F128>>(
    state: ProverState,
    opening: &BitsOpening<'_>,
    transcript: &mut T,
    observer: &mut impl FnMut(Event),
) -> Result<WhirOpeningProof, WhirError> {
    let schedule = Schedule::new(state.geometry)?;
    open_impl(schedule, state, opening, transcript, observer)
}

fn open_impl<T: Transcript<Challenge = F128>>(
    schedule: Schedule,
    state: ProverState,
    opening: &BitsOpening<'_>,
    transcript: &mut T,
    observer: &mut impl FnMut(Event),
) -> Result<WhirOpeningProof, WhirError> {
    if state.geometry != opening.geometry {
        return Err(WhirError::GeometryMismatch {
            expected: state.geometry,
            actual: opening.geometry,
        });
    }
    let _ = slice_claims(opening)?;
    let first = schedule.validate_geometry(state.geometry)?;
    let lanes = initial_lanes(first)?;
    for (part, expected, actual) in [
        (WhirPart::LaneValues, lanes, state.lane_values.len()),
        (WhirPart::Rounds, first.c, state.point.len()),
    ] {
        if expected != actual {
            return Err(WhirError::Shape {
                part,
                expected,
                actual,
            });
        }
    }
    let (table, point) = measure::run(observer, Phase::BridgeTables, || {
        let table = DomainTable::new(first.c, first.d)?;
        for level in schedule.levels() {
            table.validate_dimensions(level.c, level.d)?;
        }
        let mut point = try_vec(WhirPart::CyclePoint, schedule.mu())?;
        point.extend(opening.column_point.last().copied());
        point.extend_from_slice(opening.cycle_point);
        Ok::<_, WhirError>((table, point))
    })?;
    let mut levels = try_vec(WhirPart::Levels, schedule.levels().len())?;
    let mut rounds = try_vec(WhirPart::Rounds, first.k)?;
    transcript.append(&Label(OPEN_LABEL));
    let alpha = draw_element(transcript);
    let folded = if first.k == 0 {
        let rows = state
            .rows
            .as_ref()
            .try_into()
            .map_err(|_| WhirError::Shape {
                part: WhirPart::Rows,
                expected: 2,
                actual: state.rows.len(),
            })?;
        let point = point.as_slice().try_into().map_err(|_| WhirError::Shape {
            part: WhirPart::CyclePoint,
            expected: 2,
            actual: point.len(),
        })?;
        FoldedBridge::dense(rows, point, alpha)?
    } else {
        let tables = measure::run(observer, Phase::BridgeTables, || {
            BridgeTables::new(&point, alpha)
        })?;
        let first_round = measure::run(observer, Phase::BridgeFirstPass, || {
            tables.first_round(&state.rows)
        })?;
        observer(Event::Released(ReleasePoint::FirstPass));
        let coefficients = first_round.coefficients();
        append_elements(transcript, ROUND_LABEL, &coefficients)?;
        let challenge = draw_element(transcript);
        rounds.push(coefficients);
        let folded = measure::run(observer, Phase::BridgeFirstFold, || {
            first_round.fold(&state.rows, challenge)
        })?;
        observer(Event::Released(ReleasePoint::FirstFold));
        folded
    };
    let mut dense = RoundState::new(folded.message, folded.weight)?;
    let ProverState {
        rows,
        codeword,
        tree,
        point: commit_point,
        lane_values,
        geometry: _,
    } = state;
    drop(rows);
    drop(lane_values);
    let mut oracle = Oracle {
        values: OracleValues::Base(codeword),
        tree,
    };
    let mut pending: Option<Pending> = None;
    for (index, level) in schedule.levels().iter().enumerate() {
        if let Some(Pending {
            previous,
            positions,
            sample,
        }) = pending.take()
        {
            let lambda = draw_element(transcript);
            add_claim_weights(
                &mut dense.weight,
                &table,
                ClaimWeights {
                    sample_table: sample,
                    lambda,
                    previous: &previous,
                    positions: &positions,
                    commit_point: (index == 1).then_some(commit_point.as_slice()),
                },
                observer,
            )?;
            observer(Event::Released(ReleasePoint::Induction(index)));
        }
        observer(Event::Start(Phase::LaterRounds));
        for round in rounds.len()..level.k {
            let coefficients = dense.coefficients()?;
            append_elements(transcript, ROUND_LABEL, &coefficients)?;
            let challenge = draw_element(transcript);
            dense.fold(
                challenge,
                round + 1 == level.k && index + 1 < schedule.levels().len(),
            )?;
            rounds.push(coefficients);
        }
        observer(Event::End(Phase::LaterRounds));
        observer(Event::Released(ReleasePoint::LevelFold(index)));
        let (message, next) = if let Some(next_level) = schedule.levels().get(index + 1) {
            let values = measure::run(observer, Phase::LaterEncodes, || {
                Encoder::new(&table, next_level.c, next_level.d, next_level.lanes()?)?
                    .encode_extension(&dense.message)
            })?;
            let tree = measure::run(observer, Phase::LaterTrees, || {
                MerkleTree::build_canonical(&values, next_level.lanes()?)
            })?;
            observer(Event::Released(ReleasePoint::NextOracle(index + 1)));
            let root = *tree.root();
            transcript.append(&Label(ROOT_LABEL));
            transcript.append_bytes(&root);
            let sample_point = draw_point(transcript, level.c)?;
            let (sample, value) = measure::run(observer, Phase::EqualityAndSamples, || {
                let sample = equality_table(&sample_point, F192::one())?;
                let value = inner_product(&dense.message, &sample)?;
                Ok::<_, WhirError>((sample, value))
            })?;
            observer(Event::Released(ReleasePoint::NextSample(index + 1)));
            append_elements(transcript, OOD_LABEL, &[value])?;
            (
                WhirLevelMessage::Intermediate { root, value },
                Some((
                    Oracle {
                        values: OracleValues::Extension(values),
                        tree,
                    },
                    sample,
                )),
            )
        } else {
            let mut values = try_vec(
                WhirPart::FinalValues,
                power(WhirPart::FinalValues, schedule.res())?,
            )?;
            values.extend_from_slice(&dense.message);
            append_elements(transcript, FINAL_LABEL, &values)?;
            (WhirLevelMessage::Final { values }, None)
        };
        observer(Event::Start(Phase::Queries));
        let positions = draw_positions(transcript, level)?;
        let leaves = oracle.leaves(level, &positions)?;
        let digests = oracle.tree.multiproof(&positions, index)?;
        levels.push(WhirLevelProof {
            rounds,
            message,
            leaves,
            digests,
        });
        observer(Event::End(Phase::Queries));
        // Moving the replacement drops the previous oracle and its whole tree
        // after its evidence is written, before the next level's scratch exists.
        if let Some((next_oracle, sample)) = next {
            oracle = next_oracle;
            observer(Event::Released(ReleasePoint::OldOracle(index)));
            pending = Some(Pending {
                previous: *level,
                positions,
                sample,
            });
            let next_level = schedule.levels().get(index + 1).ok_or(WhirError::Shape {
                part: WhirPart::Levels,
                expected: index + 2,
                actual: schedule.levels().len(),
            })?;
            rounds = try_vec(WhirPart::Rounds, next_level.k)?;
        } else {
            drop(oracle);
            observer(Event::Released(ReleasePoint::OldOracle(index)));
            break;
        }
    }
    let mut closing_rounds = try_vec(WhirPart::Rounds, schedule.res())?;
    observer(Event::Start(Phase::LaterRounds));
    for _ in 0..schedule.res() {
        let coefficients = dense.coefficients()?;
        append_elements(transcript, ROUND_LABEL, &coefficients)?;
        let challenge = draw_element(transcript);
        dense.fold(challenge, false)?;
        closing_rounds.push(coefficients);
    }
    observer(Event::End(Phase::LaterRounds));
    Ok(WhirOpeningProof {
        levels,
        closing_rounds,
    })
}

struct Pending {
    previous: Level,
    positions: Vec<usize>,
    sample: Vec<F192>,
}

#[cfg(test)]
#[path = "open_tests.rs"]
mod tests;

enum OracleValues {
    Base(Vec<F64>),
    Extension(Vec<F192>),
}

struct Oracle {
    values: OracleValues,
    tree: MerkleTree,
}

impl Oracle {
    fn leaves(&self, level: &Level, positions: &[usize]) -> Result<Vec<Vec<u8>>, WhirError> {
        let mut leaves = try_vec(WhirPart::Leaves, positions.len())?;
        for &position in positions {
            let mut leaf = try_vec(WhirPart::Leaves, level.leaf_bytes)?;
            leaf.resize(level.leaf_bytes, 0);
            match &self.values {
                OracleValues::Base(values) => Self::encode_leaf(values, position, &mut leaf)?,
                OracleValues::Extension(values) => Self::encode_leaf(values, position, &mut leaf)?,
            }
            leaves.push(leaf);
        }
        Ok(leaves)
    }

    fn encode_leaf<F: CanonicalBytes>(
        values: &[F],
        position: usize,
        leaf: &mut [u8],
    ) -> Result<(), WhirError> {
        let width = leaf.len() / F::NUM_BYTES;
        let start = checked_product(WhirPart::Leaves, &[position, width])?;
        let end = start.checked_add(width).ok_or(WhirError::LengthOverflow {
            part: WhirPart::Leaves,
        })?;
        let values = values.get(start..end).ok_or(WhirError::Shape {
            part: WhirPart::Leaves,
            expected: end,
            actual: values.len(),
        })?;
        for (value, bytes) in values.iter().zip(leaf.chunks_exact_mut(F::NUM_BYTES)) {
            value.to_bytes_le(bytes);
        }
        Ok(())
    }
}
