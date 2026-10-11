//! Transcript-coupled verification of the commitment specification, §4–6.

use super::{
    bridge::{slice_claims, tau, BridgeWeight},
    challenge::{
        append_elements, draw_element, draw_point, draw_positions, ClaimCoefficients, COMMIT_LABEL,
        FINAL_LABEL, OOD_LABEL, OPEN_LABEL, ROOT_LABEL, ROUND_LABEL,
    },
    code::DomainTable,
    error::{try_vec, WhirError, WhirPart},
    merkle::{verify_multiproof, Digest},
    params::Schedule,
    wire::{WhirCommitment, WhirLevelMessage, WhirOpeningProof},
};
use crate::{
    commitment::{BitsCommitmentScheme, BitsGeometry, BitsOpening},
    points::{eq, eq_index, equality_table, PointsError},
};
use jolt_field::WithAccumulator;
use jolt_field::{Accumulator, CanonicalEncoding, Zero, F128, F192, F64};
use jolt_transcript::{Label, Transcript};

type ProductAccumulator = <F192 as WithAccumulator>::Accumulator;

/// Hash-based bit-table commitment with the frozen schedule of specification §5.
pub struct WhirBits;

/// Commit-time geometry, authenticated root and lane sample. Consumed by opening.
pub struct WhirVerifierState {
    geometry: BitsGeometry,
    root: Digest,
    point: Vec<F192>,
    lane_values: Vec<F192>,
}

impl BitsCommitmentScheme for WhirBits {
    type VerifierSetup = ();
    type Commitment = WhirCommitment;
    type VerifierState = WhirVerifierState;
    type OpeningProof = WhirOpeningProof;
    type Error = WhirError;

    fn verify_commit<T: Transcript<Challenge = F128>>(
        _setup: &(),
        geometry: BitsGeometry,
        commitment: &WhirCommitment,
        transcript: &mut T,
    ) -> Result<WhirVerifierState, WhirError> {
        Self::commit_with_schedule(geometry, Schedule::new(geometry)?, commitment, transcript)
    }

    fn verify_opening<T: Transcript<Challenge = F128>>(
        _setup: &(),
        state: WhirVerifierState,
        opening: &BitsOpening<'_>,
        proof: &WhirOpeningProof,
        transcript: &mut T,
    ) -> Result<(), WhirError> {
        Self::opening(
            Schedule::new(state.geometry)?,
            state,
            opening,
            proof,
            transcript,
        )
    }
}

impl WhirBits {
    /// Explicit-schedule test entry point. The transcript must already bind the
    /// geometry, just as at the production trait entry point.
    #[cfg(feature = "test-utils")]
    pub fn verify_commit_with_schedule<T: Transcript<Challenge = F128>>(
        geometry: BitsGeometry,
        schedule: Schedule,
        commitment: &WhirCommitment,
        transcript: &mut T,
    ) -> Result<WhirVerifierState, WhirError> {
        Self::commit_with_schedule(geometry, schedule, commitment, transcript)
    }

    /// Verifies with the same explicit schedule used by the test commit entry
    /// point. This test-only schedule has no soundness claim. The transcript
    /// must bind the opening request before this call.
    #[cfg(feature = "test-utils")]
    pub fn verify_opening_with_schedule<T: Transcript<Challenge = F128>>(
        schedule: Schedule,
        state: WhirVerifierState,
        opening: &BitsOpening<'_>,
        proof: &WhirOpeningProof,
        transcript: &mut T,
    ) -> Result<(), WhirError> {
        Self::opening(schedule, state, opening, proof, transcript)
    }

    fn commit_with_schedule<T: Transcript<Challenge = F128>>(
        geometry: BitsGeometry,
        schedule: Schedule,
        commitment: &WhirCommitment,
        transcript: &mut T,
    ) -> Result<WhirVerifierState, WhirError> {
        let first = schedule.validate_geometry(geometry)?;
        commitment.validate(&schedule)?;
        let expected = first.lanes()?;
        // Check all explicit-schedule dimensions before transcript draws or shifts.
        let domain = DomainTable::new(first.c, first.d)?;
        for level in schedule.levels() {
            domain.validate_dimensions(level.c, level.d)?;
        }
        let mut lane_values = try_vec(WhirPart::LaneValues, expected)?;
        lane_values.extend_from_slice(&commitment.lane_values);
        transcript.append(&Label(COMMIT_LABEL));
        transcript.append_bytes(&commitment.root);
        let point = draw_point(transcript, first.c)?;
        append_elements(transcript, OOD_LABEL, &lane_values)?;
        Ok(WhirVerifierState {
            geometry,
            root: commitment.root,
            point,
            lane_values,
        })
    }

    fn rounds<T: Transcript<Challenge = F128>>(
        rounds: &[[F192; 2]],
        sigma: &mut F192,
        q: &mut Vec<F192>,
        transcript: &mut T,
    ) -> Result<(), WhirError> {
        for &[u_0, u_2] in rounds {
            append_elements(transcript, ROUND_LABEL, &[u_0, u_2])?;
            let a = draw_element(transcript);
            *sigma = u_0 + (*sigma + u_2) * a + u_2 * a * a;
            q.push(a);
        }
        Ok(())
    }

    fn opening<T: Transcript<Challenge = F128>>(
        schedule: Schedule,
        state: WhirVerifierState,
        opening: &BitsOpening<'_>,
        proof: &WhirOpeningProof,
        transcript: &mut T,
    ) -> Result<(), WhirError> {
        if opening.geometry != state.geometry {
            return Err(WhirError::GeometryMismatch {
                expected: state.geometry,
                actual: opening.geometry,
            });
        }
        let first = schedule
            .validate_geometry(state.geometry)
            .map_err(|error| {
                if let WhirError::GeometryMismatch { expected, actual } = error {
                    WhirError::GeometryMismatch {
                        expected: actual,
                        actual: expected,
                    }
                } else {
                    error
                }
            })?;
        let slices = slice_claims(opening)?;
        proof.validate(&schedule)?;
        for (part, expected, actual) in [
            (
                WhirPart::LaneValues,
                first.lanes()?,
                state.lane_values.len(),
            ),
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
        let domain = DomainTable::new(first.c, first.d)?;
        for level in schedule.levels() {
            domain.validate_dimensions(level.c, level.d)?;
        }
        let mut q = try_vec(WhirPart::Rounds, schedule.mu())?;
        let mut batches = try_vec(WhirPart::Levels, schedule.levels().len().saturating_sub(1))?;
        let mut pending: Option<ClaimBatch> = None;
        let mut root = state.root;
        transcript.append(&Label(OPEN_LABEL));
        let alpha = draw_element(transcript);
        let mut sigma = tau(&slices, alpha);
        for (i, (level, evidence)) in schedule.levels().iter().zip(&proof.levels).enumerate() {
            if let Some(claims) = pending.take() {
                let lambda = draw_element(transcript);
                sigma += claims.combination(lambda);
                batches.push((lambda, claims));
            }
            let offset = q.len();
            Self::rounds(&evidence.rounds, &mut sigma, &mut q, transcript)?;
            let lane_point = q.get(offset..).ok_or(WhirError::Shape {
                part: WhirPart::Rounds,
                expected: level.k,
                actual: q.len(),
            })?;
            let weights = equality_table(lane_point).map_err(point_error)?;
            let next = match &evidence.message {
                WhirLevelMessage::Intermediate { root, value } => {
                    transcript.append(&Label(ROOT_LABEL));
                    transcript.append_bytes(root);
                    let point = draw_point(transcript, level.c)?;
                    append_elements(transcript, OOD_LABEL, &[*value])?;
                    Some((*root, *value, point))
                }
                WhirLevelMessage::Final { values } => {
                    append_elements(transcript, FINAL_LABEL, values)?;
                    None
                }
            };
            let positions = draw_positions(transcript, level)?;
            verify_multiproof(
                &root,
                level.d,
                level.leaf_bytes,
                &positions,
                &evidence.leaves,
                &evidence.digests,
                i,
            )?;
            let mut values = try_vec(WhirPart::Leaves, positions.len())?;
            for leaf in &evidence.leaves {
                let mut sum = ProductAccumulator::default();
                if i == 0 {
                    for (weight, bytes) in weights.iter().zip(leaf.chunks_exact(16)) {
                        let (low, high) = bytes.split_at(8);
                        let b_0 = F64::from_bytes_le_checked(low)
                            .ok_or(WhirError::MerkleAuthentication { level: i })?;
                        let b_1 = F64::from_bytes_le_checked(high)
                            .ok_or(WhirError::MerkleAuthentication { level: i })?;
                        sum.fmadd_base_pair(*weight, [b_0, b_1]);
                    }
                } else {
                    for (weight, bytes) in weights.iter().zip(leaf.chunks_exact(24)) {
                        let symbol = F192::from_bytes_le_checked(bytes)
                            .ok_or(WhirError::MerkleAuthentication { level: i })?;
                        sum.fmadd(*weight, symbol);
                    }
                }
                values.push(sum.reduce());
            }
            let commit_sample = if i == 0 {
                Some(
                    weights
                        .iter()
                        .zip(&state.lane_values)
                        .map(|(w, v)| *w * *v)
                        .sum(),
                )
            } else {
                None
            };
            if let Some((next_root, value, point)) = next {
                root = next_root;
                pending = Some(ClaimBatch {
                    offset: q.len(),
                    c: level.c,
                    positions,
                    values,
                    point,
                    value,
                    commit_sample,
                });
            } else if let WhirLevelMessage::Final {
                values: final_values,
            } = &evidence.message
            {
                for (&position, value) in positions.iter().zip(values) {
                    if value != domain.evaluate_message(level.c, position as u32, final_values)? {
                        return Err(WhirError::FinalCodeMismatch { position });
                    }
                }
                if let Some(sample) = commit_sample {
                    if evaluate(final_values, &state.point)? != sample {
                        return Err(WhirError::CommitSampleMismatch);
                    }
                }
            }
        }
        Self::rounds(&proof.closing_rounds, &mut sigma, &mut q, transcript)?;
        let mut bridge = BridgeWeight::new();
        for (r, a) in opening
            .column_point
            .last()
            .into_iter()
            .chain(opening.cycle_point)
            .zip(&q)
        {
            bridge.absorb(*r, *a);
        }
        let mut omega = bridge.finish(alpha);
        for (lambda, claims) in batches {
            let suffix = q
                .get(claims.offset..)
                .ok_or(WhirError::ClosingIdentityMismatch)?;
            omega += claims.weight(lambda, suffix, &state.point, &domain)?;
        }
        let final_values = match proof.levels.last().map(|level| &level.message) {
            Some(WhirLevelMessage::Final { values }) => values,
            Some(WhirLevelMessage::Intermediate { .. }) | None => {
                return Err(WhirError::ClosingIdentityMismatch)
            }
        };
        let suffix = q
            .get(schedule.mu() - schedule.res()..)
            .ok_or(WhirError::ClosingIdentityMismatch)?;
        if sigma != evaluate(final_values, suffix)? * omega {
            return Err(WhirError::ClosingIdentityMismatch);
        }
        Ok(())
    }
}

struct ClaimBatch {
    offset: usize,
    c: usize,
    positions: Vec<usize>,
    values: Vec<F192>,
    point: Vec<F192>,
    value: F192,
    commit_sample: Option<F192>,
}

impl ClaimBatch {
    fn combination(&self, lambda: F192) -> F192 {
        let mut coefficients = ClaimCoefficients::new(lambda);
        let mut sum = coefficients.sample() * self.value;
        for value in &self.values {
            sum += coefficients.next_query() * *value;
        }
        if let Some(sample) = self.commit_sample {
            sum += coefficients.commit_sample() * sample;
        }
        sum
    }

    fn weight(
        &self,
        lambda: F192,
        q: &[F192],
        commit_point: &[F192],
        domain: &DomainTable,
    ) -> Result<F192, WhirError> {
        let mut coefficients = ClaimCoefficients::new(lambda);
        let mut sum = coefficients.sample() * eq(&self.point, q).map_err(point_error)?;
        for position in &self.positions {
            sum += coefficients.next_query() * domain.query_weight(self.c, *position as u32, q);
        }
        if self.commit_sample.is_some() {
            sum += coefficients.commit_sample() * eq(commit_point, q).map_err(point_error)?;
        }
        Ok(sum)
    }
}

fn point_error(error: PointsError) -> WhirError {
    match error {
        PointsError::Allocation { .. } => WhirError::Allocation {
            part: WhirPart::LaneValues,
        },
        PointsError::Dimension { expected, actual } => WhirError::Shape {
            part: WhirPart::Rounds,
            expected,
            actual,
        },
        PointsError::Index { index, variables } => WhirError::Shape {
            part: WhirPart::FinalValues,
            expected: variables,
            actual: index,
        },
        PointsError::MissingColumn { column } => WhirError::Shape {
            part: WhirPart::Columns,
            expected: 256,
            actual: column,
        },
    }
}

fn evaluate(values: &[F192], point: &[F192]) -> Result<F192, WhirError> {
    values
        .iter()
        .enumerate()
        .try_fold(F192::zero(), |sum, (w, value)| {
            Ok(sum + *value * eq_index(point, w).map_err(point_error)?)
        })
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "tests assert checked geometry and deliberately mutate bounded fixtures"
)]
mod tests {
    use super::WhirBits;
    use crate::{
        commitment::{BitsCommitmentScheme, BitsGeometry, BitsOpening, BitsWire},
        transcript::Rv64iTranscript,
        whir::{
            error::{WhirError, WhirPart},
            merkle::{hash_leaf, hash_node, Digest},
            params::{Queries, Schedule},
            wire::{WhirCommitment, WhirLevelMessage, WhirLevelProof, WhirOpeningProof},
        },
    };
    use jolt_field::{One, Zero, F128, F192};
    use jolt_transcript::Transcript;

    // The zero word has a constant hash at each tree height, independent of
    // its queried positions. This fixture constructs no proving algorithm.
    fn zero_tree(leaf_bytes: usize, depth: usize) -> Vec<Digest> {
        let mut hashes = vec![hash_leaf(&vec![0; leaf_bytes])];
        for _ in 0..depth {
            let child = *hashes.last().unwrap();
            hashes.push(hash_node(&child, &child));
        }
        hashes
    }

    fn zero_fixture(t: usize) -> (WhirCommitment, WhirOpeningProof) {
        let schedule = Schedule::new(BitsGeometry { log_T: t }).unwrap();
        zero_fixture_schedule(&schedule)
    }

    fn zero_fixture_schedule(schedule: &Schedule) -> (WhirCommitment, WhirOpeningProof) {
        let first = &schedule.levels()[0];
        let commitment = WhirCommitment {
            root: zero_tree(first.leaf_bytes, first.d)[first.d],
            lane_values: vec![F192::zero(); first.lanes().unwrap()],
        };
        let levels = schedule
            .levels()
            .iter()
            .enumerate()
            .map(|(i, level)| {
                let message = if let Some(next) = schedule.levels().get(i + 1) {
                    WhirLevelMessage::Intermediate {
                        root: zero_tree(next.leaf_bytes, next.d)[next.d],
                        value: F192::zero(),
                    }
                } else {
                    WhirLevelMessage::Final {
                        values: vec![F192::zero(); 1 << schedule.res()],
                    }
                };
                let (count, digests) = match level.queries {
                    Queries::All => (1 << level.d, vec![]),
                    Queries::Count(_) => {
                        (1, zero_tree(level.leaf_bytes, level.d)[..level.d].to_vec())
                    }
                };
                WhirLevelProof {
                    rounds: vec![[F192::zero(); 2]; level.k],
                    message,
                    leaves: vec![vec![0; level.leaf_bytes]; count],
                    digests,
                }
            })
            .collect();
        (
            commitment,
            WhirOpeningProof {
                levels,
                closing_rounds: vec![[F192::zero(); 2]; schedule.res()],
            },
        )
    }

    fn check<T: Transcript<Challenge = F128>>(
        t: usize,
        commitment: &WhirCommitment,
        proof: &WhirOpeningProof,
        transcript: &mut T,
    ) -> Result<(), WhirError> {
        let geometry = BitsGeometry { log_T: t };
        let state = WhirBits::verify_commit(&(), geometry, commitment, transcript)?;
        WhirBits::verify_opening(
            &(),
            state,
            &BitsOpening {
                geometry,
                column_point: &[F128::zero(); 8],
                cycle_point: &vec![F128::zero(); t],
                columns: &[F128::zero(); 256],
            },
            proof,
            transcript,
        )
    }

    #[derive(Default)]
    struct Recording {
        calls: Vec<String>,
        draws: usize,
    }
    impl Transcript for Recording {
        type Challenge = F128;
        fn new(_: &'static [u8]) -> Self {
            Self::default()
        }
        fn append_bytes(&mut self, bytes: &[u8]) {
            if let Some(label) = [
                "whir_commit",
                "whir_ood",
                "whir_open",
                "whir_round",
                "whir_root",
                "whir_final",
            ]
            .into_iter()
            .find(|label| bytes.starts_with(label.as_bytes()))
            {
                let mut expected = [0; 32];
                expected[..label.len()].copy_from_slice(label.as_bytes());
                assert_eq!(bytes, expected);
                self.calls.push(label.to_string());
            } else {
                self.calls.push(format!("B{}", bytes.len()));
            }
        }
        fn challenge(&mut self) -> F128 {
            self.calls.push("D".to_string());
            self.draws += 1;
            F128::zero()
        }
        fn state(&self) -> [u8; 32] {
            [0; 32]
        }
    }

    #[test]
    fn transcript_literal_call_order_small_and_numeric_levels() {
        // Counts of successive D calls are literal scalar draws (two per E),
        // not logical E challenges. Numeric queries make ceil(Q/4) draws.
        for (t, expected) in [
            (6, "whir_commit B32 D4 whir_ood B768 whir_open D2 whir_round B48 D2 whir_round B48 D2 whir_round B48 D2 whir_round B48 D2 whir_round B48 D2 whir_final B96 whir_round B48 D2 whir_round B48 D2"),
            (10, "whir_commit B32 D12 whir_ood B768 whir_open D2 whir_round B48 D2 whir_round B48 D2 whir_round B48 D2 whir_round B48 D2 whir_round B48 D2 whir_root B32 D12 whir_ood B24 D2 whir_round B48 D2 whir_round B48 D2 whir_round B48 D2 whir_round B48 D2 whir_final B96 D15 whir_round B48 D2 whir_round B48 D2"),
            (11, "whir_commit B32 D14 whir_ood B768 whir_open D2 whir_round B48 D2 whir_round B48 D2 whir_round B48 D2 whir_round B48 D2 whir_round B48 D2 whir_root B32 D14 whir_ood B24 D66 whir_round B48 D2 whir_round B48 D2 whir_round B48 D2 whir_round B48 D2 whir_final B192 D16 whir_round B48 D2 whir_round B48 D2 whir_round B48 D2"),
        ] {
            let (commitment, proof) = zero_fixture(t);
            let mut recording = Recording::default();
            assert_eq!(check(t, &commitment, &proof, &mut recording), Ok(()));
            let mut compact = Vec::new();
            let mut draws = 0;
            for call in recording.calls {
                if call == "D" { draws += 1; } else {
                    if draws > 0 { compact.push(format!("D{draws}")); draws = 0; }
                    compact.push(call);
                }
            }
            if draws > 0 { compact.push(format!("D{draws}")); }
            assert_eq!(compact.join(" "), expected, "t={t}");
        }
    }

    #[test]
    fn verifier_zero_word_supported_small_geometries() {
        for t in 1..=9 {
            let (commitment, proof) = zero_fixture(t);
            assert_eq!(
                check(
                    t,
                    &commitment,
                    &proof,
                    &mut Rv64iTranscript::new(b"whir-test")
                ),
                Ok(())
            );
        }
    }

    #[cfg(feature = "test-utils")]
    #[test]
    fn verifier_explicit_schedule_last_fold_three() {
        let geometry = BitsGeometry { log_T: 11 };
        let schedule = Schedule::from_levels(
            geometry,
            &[(5, 7, 8, Queries::All), (3, 4, 7, Queries::All)],
        )
        .unwrap();
        let (commitment, proof) = zero_fixture_schedule(&schedule);
        assert_eq!(schedule.levels()[1].lanes().unwrap(), 8);
        assert_eq!(proof.levels[1].leaves[0].len(), 192);
        let mut transcript = Rv64iTranscript::new(b"whir-explicit");
        let state = WhirBits::verify_commit_with_schedule(
            geometry,
            schedule.clone(),
            &commitment,
            &mut transcript,
        )
        .unwrap();
        assert_eq!(
            WhirBits::verify_opening_with_schedule(
                schedule,
                state,
                &BitsOpening {
                    geometry,
                    column_point: &[F128::zero(); 8],
                    cycle_point: &[F128::zero(); 11],
                    columns: &[F128::zero(); 256],
                },
                &proof,
                &mut transcript
            ),
            Ok(())
        );
    }

    #[test]
    fn verifier_geometry_and_request_shapes() {
        let (commitment, proof) = zero_fixture(6);
        for t in [0, 33] {
            assert!(
                matches!(WhirBits::verify_commit(&(), BitsGeometry { log_T: t }, &commitment, &mut Recording::default()), Err(WhirError::UnsupportedGeometry { log_T }) if log_T == t)
            );
        }
        let wrong = WhirCommitment {
            root: commitment.root,
            lane_values: vec![F192::zero(); 31],
        };
        assert!(matches!(
            WhirBits::verify_commit(
                &(),
                BitsGeometry { log_T: 6 },
                &wrong,
                &mut Recording::default()
            ),
            Err(WhirError::Shape {
                part: WhirPart::LaneValues,
                expected: 32,
                actual: 31
            })
        ));
        for (t, column_count, cycle_count, columns_count, error) in [
            (
                7,
                8,
                7,
                256,
                WhirError::GeometryMismatch {
                    expected: BitsGeometry { log_T: 6 },
                    actual: BitsGeometry { log_T: 7 },
                },
            ),
            (
                6,
                7,
                6,
                256,
                WhirError::Shape {
                    part: WhirPart::ColumnPoint,
                    expected: 8,
                    actual: 7,
                },
            ),
            (
                6,
                8,
                5,
                256,
                WhirError::Shape {
                    part: WhirPart::CyclePoint,
                    expected: 6,
                    actual: 5,
                },
            ),
            (
                6,
                8,
                6,
                255,
                WhirError::Shape {
                    part: WhirPart::Columns,
                    expected: 256,
                    actual: 255,
                },
            ),
        ] {
            let mut transcript = Recording::default();
            let state = WhirBits::verify_commit(
                &(),
                BitsGeometry { log_T: 6 },
                &commitment,
                &mut transcript,
            )
            .unwrap();
            let before = transcript.calls.len();
            let request = BitsOpening {
                geometry: BitsGeometry { log_T: t },
                column_point: &vec![F128::zero(); column_count],
                cycle_point: &vec![F128::zero(); cycle_count],
                columns: &vec![F128::zero(); columns_count],
            };
            assert_eq!(
                WhirBits::verify_opening(&(), state, &request, &proof, &mut transcript),
                Err(error)
            );
            assert_eq!(transcript.calls.len(), before);
        }
    }

    #[test]
    fn verifier_typed_proof_shapes_before_transcript() {
        let (commitment, proof) = zero_fixture(6);
        for (part, expected, actual, changed) in [
            (WhirPart::Levels, 1, 0, {
                let mut p = proof.clone();
                p.levels.clear();
                p
            }),
            (WhirPart::Rounds, 5, 4, {
                let mut p = proof.clone();
                let _ = p.levels[0].rounds.pop();
                p
            }),
            (WhirPart::FinalValues, 4, 3, {
                let mut p = proof.clone();
                p.levels[0].message = WhirLevelMessage::Final {
                    values: vec![F192::zero(); 3],
                };
                p
            }),
            (WhirPart::Leaves, 512, 511, {
                let mut p = proof.clone();
                let _ = p.levels[0].leaves[0].pop();
                p
            }),
            (WhirPart::Rounds, 2, 1, {
                let mut p = proof.clone();
                let _ = p.closing_rounds.pop();
                p
            }),
        ] {
            let mut transcript = Recording::default();
            let result = check(6, &commitment, &changed, &mut transcript);
            assert_eq!(
                result,
                Err(WhirError::Shape {
                    part,
                    expected,
                    actual
                })
            );
            assert_eq!(transcript.calls.last().unwrap(), "B768");
        }
        // The shape of the last level wins over an earlier bad authentication.
        let (commitment, mut proof) = zero_fixture(11);
        proof.levels[0].leaves[0][0] = 1;
        proof.levels[1].rounds.clear();
        assert_eq!(
            check(11, &commitment, &proof, &mut Recording::default()),
            Err(WhirError::Shape {
                part: WhirPart::Rounds,
                expected: 4,
                actual: 0
            })
        );
    }

    #[test]
    fn verifier_all_position_counts_and_numeric_counts() {
        for t in [1, 6, 10, 11] {
            let (commitment, proof) = zero_fixture(t);
            let expected = proof.levels[0].leaves.len();
            for append in [false, true] {
                let mut altered = proof.clone();
                if append {
                    let leaf = altered.levels[0].leaves[0].clone();
                    altered.levels[0].leaves.push(leaf);
                } else {
                    let _ = altered.levels[0].leaves.pop();
                }
                assert_eq!(
                    check(t, &commitment, &altered, &mut Recording::default()),
                    Err(WhirError::CountMismatch {
                        level: 0,
                        part: WhirPart::Leaves,
                        expected,
                        actual: if append { expected + 1 } else { expected - 1 }
                    })
                );
            }
            let mut altered = proof.clone();
            altered.levels[0].digests.push([0; 32]);
            assert_eq!(
                check(t, &commitment, &altered, &mut Recording::default()),
                Err(WhirError::CountMismatch {
                    level: 0,
                    part: WhirPart::Digests,
                    expected: proof.levels[0].digests.len(),
                    actual: proof.levels[0].digests.len() + 1
                })
            );
            if t == 11 {
                altered.levels[0]
                    .digests
                    .truncate(proof.levels[0].digests.len() - 1);
                assert_eq!(
                    check(t, &commitment, &altered, &mut Recording::default()),
                    Err(WhirError::CountMismatch {
                        level: 0,
                        part: WhirPart::Digests,
                        expected: proof.levels[0].digests.len(),
                        actual: proof.levels[0].digests.len() - 1
                    })
                );
            }
        }
    }

    #[test]
    fn verifier_terminal_errors_have_exact_variants() {
        let (commitment, proof) = zero_fixture(6);
        let mut altered = commitment.clone();
        altered.root[0] ^= 1;
        assert_eq!(
            check(6, &altered, &proof, &mut Recording::default()),
            Err(WhirError::MerkleAuthentication { level: 0 })
        );
        altered = commitment.clone();
        altered.lane_values[0] = F192::one();
        assert_eq!(
            check(6, &altered, &proof, &mut Recording::default()),
            Err(WhirError::CommitSampleMismatch)
        );
        let mut altered = proof.clone();
        if let WhirLevelMessage::Final { values } = &mut altered.levels[0].message {
            values[0] = F192::one();
        }
        assert_eq!(
            check(6, &commitment, &altered, &mut Recording::default()),
            Err(WhirError::FinalCodeMismatch { position: 0 })
        );
        let mut altered = proof.clone();
        altered.closing_rounds[1][0] = F192::one();
        assert_eq!(
            check(6, &commitment, &altered, &mut Recording::default()),
            Err(WhirError::ClosingIdentityMismatch)
        );
    }

    #[test]
    fn verifier_random_strings_and_every_single_byte_mutation_do_not_panic() {
        let geometry = BitsGeometry { log_T: 6 };
        let (commitment, proof) = zero_fixture(6);
        let mut bytes = Vec::new();
        proof.write(&mut bytes);
        for index in 0..bytes.len() {
            for difference in 1..=u8::MAX {
                bytes[index] ^= difference;
                if let Some(parsed) = WhirOpeningProof::read(&bytes, geometry) {
                    let _result = check(
                        6,
                        &commitment,
                        &parsed,
                        &mut Rv64iTranscript::new(b"whir-test"),
                    );
                }
                bytes[index] ^= difference;
            }
        }
        let mut bytes = Vec::new();
        commitment.write(&mut bytes);
        for index in 0..bytes.len() {
            for difference in 1..=u8::MAX {
                bytes[index] ^= difference;
                let parsed = WhirCommitment::read(&bytes, geometry).unwrap();
                let _result = check(6, &parsed, &proof, &mut Rv64iTranscript::new(b"whir-test"));
                bytes[index] ^= difference;
            }
        }
        let mut seed = 0x832b_6cdd_fcf7_96e1u64;
        for i in 0..10_000 {
            let mut raw = vec![0; i % 1024];
            for byte in &mut raw {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *byte = seed as u8;
            }
            let _opening = WhirOpeningProof::read(&raw, geometry);
            let candidate =
                WhirCommitment::read(&raw, geometry).unwrap_or_else(|| WhirCommitment {
                    root: [0; 32],
                    lane_values: vec![],
                });
            let mut transcript = Rv64iTranscript::new(b"whir-random");
            let result = WhirBits::verify_commit(&(), geometry, &candidate, &mut transcript);
            if let Ok(state) = result {
                let candidate =
                    WhirOpeningProof::read(&raw, geometry).unwrap_or_else(|| WhirOpeningProof {
                        levels: vec![],
                        closing_rounds: vec![],
                    });
                let _result = WhirBits::verify_opening(
                    &(),
                    state,
                    &BitsOpening {
                        geometry,
                        column_point: &[F128::zero(); 8],
                        cycle_point: &[F128::zero(); 6],
                        columns: &[F128::zero(); 256],
                    },
                    &candidate,
                    &mut transcript,
                );
            } else {
                let _result = check(
                    6,
                    &commitment,
                    &WhirOpeningProof {
                        levels: vec![],
                        closing_rounds: vec![],
                    },
                    &mut transcript,
                );
            }
        }
    }
}
