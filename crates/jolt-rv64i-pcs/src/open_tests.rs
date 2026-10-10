#![expect(
    clippy::unwrap_used,
    reason = "test fixtures assert valid protocol shapes"
)]

use crate::commit::{commit, commit_with_schedule, ProverState};
use crate::merkle::MerkleTree;
use crate::open::{open, open_with_schedule};
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use jolt_field::{CanonicalBytes, One, Zero, F128, F192, F64};
use jolt_rv64i_verifier::commitment::{BitsCommitmentScheme, BitsGeometry, BitsOpening, BitsWire};
use jolt_rv64i_verifier::points::{eq_index, equality_table};
use jolt_rv64i_verifier::whir::error::{WhirError, WhirPart};
use jolt_rv64i_verifier::whir::params::{Queries, Schedule};
use jolt_rv64i_verifier::whir::wire::{WhirCommitment, WhirLevelMessage, WhirOpeningProof};
use jolt_rv64i_verifier::whir::WhirBits;
use jolt_transcript::{Blake2bTranscript, Label, Transcript};
use rayon::ThreadPoolBuilder;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Call {
    Append(Vec<u8>),
    Draw(F128),
}

#[derive(Default)]
struct Recorded {
    inner: Blake2bTranscript<F128>,
    calls: Vec<Call>,
}

impl Clone for Recorded {
    fn clone(&self) -> Self {
        let mut out = Self::new(b"whir_contract");
        for call in &self.calls {
            match call {
                Call::Append(bytes) => out.append_bytes(bytes),
                Call::Draw(_) => {
                    let _draw = out.challenge();
                }
            }
        }
        out
    }
}

impl Transcript for Recorded {
    type Challenge = F128;

    fn new(label: &'static [u8]) -> Self {
        Self {
            inner: Blake2bTranscript::new(label),
            calls: Vec::new(),
        }
    }
    fn append_bytes(&mut self, bytes: &[u8]) {
        self.calls.push(Call::Append(bytes.to_vec()));
        self.inner.append_bytes(bytes);
    }
    fn challenge(&mut self) -> F128 {
        let value = self.inner.challenge();
        self.calls.push(Call::Draw(value));
        value
    }
    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
}

struct Words(u64);
impl Words {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

struct Fixture {
    geometry: BitsGeometry,
    schedule: Schedule,
    commitment: WhirCommitment,
    proof: WhirOpeningProof,
    cycle: Vec<F128>,
    rho: Vec<F128>,
    columns: Vec<F128>,
    rows: Arc<[[u64; 4]]>,
    initial: Recorded,
    committed: Recorded,
    prover_committed: Recorded,
    opening: Recorded,
    final_transcript: Recorded,
}

impl Fixture {
    fn new(t: usize) -> Self {
        Self::build(t, None, |_, _| {})
    }

    fn build(
        t: usize,
        explicit: Option<Schedule>,
        mutate: impl FnOnce(&mut WhirCommitment, &mut ProverState),
    ) -> Self {
        let geometry = BitsGeometry { log_T: t };
        let schedule = explicit
            .clone()
            .unwrap_or_else(|| Schedule::new(geometry).unwrap());
        let mut rng = Words(0x729a_610b_77c2_9031);
        let rows: Arc<[[u64; 4]]> = (0..1usize << t)
            .map(|_| std::array::from_fn(|_| rng.next()))
            .collect::<Vec<_>>()
            .into();
        let mut initial = Recorded::new(b"whir_contract");
        initial.append(&Label(b"geometry"));
        initial.append_bytes(&(t as u64).to_le_bytes());
        let mut transcript = initial.clone();
        let (mut commitment, mut state) = if explicit.is_some() {
            commit_with_schedule(geometry, schedule.clone(), &rows, &mut transcript).unwrap()
        } else {
            commit(geometry, &rows, &mut transcript).unwrap()
        };
        let prover_committed = transcript.clone();
        mutate(&mut commitment, &mut state);
        // Replaying a malicious commitment lets terminal checks be exercised
        // with identical challenge histories on both sides.
        transcript = initial.clone();
        let _state = WhirBits::verify_commit_with_schedule(
            geometry,
            schedule.clone(),
            &commitment,
            &mut transcript,
        )
        .unwrap();
        let committed = transcript.clone();
        let cycle = (0..t).map(|_| transcript.challenge()).collect::<Vec<_>>();
        let weights = equality_table(&cycle).unwrap();
        let mut columns = vec![F128::zero(); 256];
        for (&weight, row) in weights.iter().zip(rows.iter()) {
            for (column, value) in columns.iter_mut().enumerate() {
                if row[column / 64] >> (column % 64) & 1 != 0 {
                    *value += weight;
                }
            }
        }
        transcript.append(&Label(b"column_values"));
        let mut bytes = Vec::new();
        for value in &columns {
            let mut encoded = [0; 16];
            value.to_bytes_le(&mut encoded);
            bytes.extend_from_slice(&encoded);
        }
        transcript.append_bytes(&bytes);
        let rho = (0..8).map(|_| transcript.challenge()).collect::<Vec<_>>();
        let opening = transcript.clone();
        let request = BitsOpening {
            geometry,
            column_point: &rho,
            cycle_point: &cycle,
            columns: &columns,
        };
        let proof = if explicit.is_some() {
            open_with_schedule(schedule.clone(), state, &request, &mut transcript).unwrap()
        } else {
            open(state, &request, &mut transcript).unwrap()
        };
        Self {
            geometry,
            schedule,
            commitment,
            proof,
            cycle,
            rho,
            columns,
            rows,
            initial,
            committed,
            prover_committed,
            opening,
            final_transcript: transcript,
        }
    }

    fn request(&self) -> BitsOpening<'_> {
        BitsOpening {
            geometry: self.geometry,
            column_point: &self.rho,
            cycle_point: &self.cycle,
            columns: &self.columns,
        }
    }

    fn verify(
        &self,
        commitment: &WhirCommitment,
        proof: &WhirOpeningProof,
        request: &BitsOpening<'_>,
    ) -> Result<(), WhirError> {
        let mut transcript = self.initial.clone();
        let state = WhirBits::verify_commit_with_schedule(
            self.geometry,
            self.schedule.clone(),
            commitment,
            &mut transcript,
        )?;
        for call in self.opening.calls.iter().skip(self.committed.calls.len()) {
            match call {
                Call::Append(bytes) => transcript.append_bytes(bytes),
                Call::Draw(_) => {
                    let _draw = transcript.challenge();
                }
            }
        }
        WhirBits::verify_opening_with_schedule(
            self.schedule.clone(),
            state,
            request,
            proof,
            &mut transcript,
        )
    }

    fn accepted(&self) {
        self.verify(&self.commitment, &self.proof, &self.request())
            .unwrap();
    }

    fn wire(&self) -> (Vec<u8>, Vec<u8>) {
        let mut commitment = Vec::new();
        let mut proof = Vec::new();
        self.commitment.write(&mut commitment);
        self.proof.write(&mut proof);
        (commitment, proof)
    }
}

fn explicit_three() -> Schedule {
    Schedule::from_levels(
        BitsGeometry { log_T: 11 },
        &[(5, 7, 8, Queries::All), (3, 4, 7, Queries::All)],
    )
    .unwrap()
}

fn changed(value: F192) -> F192 {
    value + F192::one()
}

#[test]
fn completeness_random_tables_all_small_supported_sizes() {
    for t in 1..=14 {
        let fixture = Fixture::new(t);
        fixture.accepted();
        let mut bitwise = F128::zero();
        let column_weights: [F128; 256] =
            std::array::from_fn(|column| eq_index(&fixture.rho, column).unwrap());
        for (cycle, row) in fixture.rows.iter().enumerate() {
            let cycle_weight = eq_index(&fixture.cycle, cycle).unwrap();
            for column in 0..256 {
                if row[column / 64] >> (column % 64) & 1 != 0 {
                    bitwise += cycle_weight * column_weights[column];
                }
            }
        }
        assert_eq!(fixture.request().value(), bitwise, "t={t}");
        let (commitment, proof) = fixture.wire();
        assert_eq!(
            WhirCommitment::read(&commitment, fixture.geometry),
            Some(fixture.commitment.clone())
        );
        assert_eq!(
            WhirOpeningProof::read(&proof, fixture.geometry),
            Some(fixture.proof.clone())
        );
    }
}

#[test]
fn completeness_explicit_last_fold_of_three() {
    let fixture = Fixture::build(11, Some(explicit_three()), |_, _| {});
    fixture.accepted();
    assert_eq!(fixture.schedule.levels()[1].lanes().unwrap(), 8);
    assert!(fixture.proof.levels[1]
        .leaves
        .iter()
        .all(|leaf| leaf.len() == 192));
    reject_wire_field_classes(&fixture);
}

fn reject_wire_field_classes(fixture: &Fixture) {
    let request = fixture.request();
    let mut commitment = fixture.commitment.clone();
    commitment.root[0] ^= 1;
    assert!(fixture
        .verify(&commitment, &fixture.proof, &request)
        .is_err());
    for lane in 0..fixture.commitment.lane_values.len() {
        let mut commitment = fixture.commitment.clone();
        commitment.lane_values[lane] = changed(commitment.lane_values[lane]);
        assert!(fixture
            .verify(&commitment, &fixture.proof, &request)
            .is_err());
    }
    for level in 0..fixture.proof.levels.len() {
        for round in 0..fixture.proof.levels[level].rounds.len() {
            for coefficient in 0..2 {
                let mut proof = fixture.proof.clone();
                proof.levels[level].rounds[round][coefficient] =
                    changed(proof.levels[level].rounds[round][coefficient]);
                assert!(fixture
                    .verify(&fixture.commitment, &proof, &request)
                    .is_err());
            }
        }
        match &fixture.proof.levels[level].message {
            WhirLevelMessage::Intermediate { .. } => {
                for change_root in [false, true] {
                    let mut proof = fixture.proof.clone();
                    if let WhirLevelMessage::Intermediate { root, value } =
                        &mut proof.levels[level].message
                    {
                        if change_root {
                            root[0] ^= 1;
                        } else {
                            *value = changed(*value);
                        }
                    }
                    assert!(fixture
                        .verify(&fixture.commitment, &proof, &request)
                        .is_err());
                }
            }
            WhirLevelMessage::Final { values } => {
                for index in 0..values.len() {
                    let mut proof = fixture.proof.clone();
                    if let WhirLevelMessage::Final { values } = &mut proof.levels[level].message {
                        values[index] = changed(values[index]);
                    }
                    assert!(fixture
                        .verify(&fixture.commitment, &proof, &request)
                        .is_err());
                }
            }
        }
        let mut proof = fixture.proof.clone();
        proof.levels[level].leaves[0][0] ^= 1;
        assert_eq!(
            fixture.verify(&fixture.commitment, &proof, &request),
            Err(WhirError::MerkleAuthentication { level })
        );
        if !fixture.proof.levels[level].digests.is_empty() {
            let mut proof = fixture.proof.clone();
            proof.levels[level].digests[0][0] ^= 1;
            assert_eq!(
                fixture.verify(&fixture.commitment, &proof, &request),
                Err(WhirError::MerkleAuthentication { level })
            );
        }
        for append in [false, true] {
            let mut proof = fixture.proof.clone();
            if append {
                let leaf = proof.levels[level].leaves[0].clone();
                proof.levels[level].leaves.push(leaf);
            } else {
                let _removed = proof.levels[level].leaves.pop();
            }
            assert_eq!(
                fixture.verify(&fixture.commitment, &proof, &request),
                Err(WhirError::CountMismatch {
                    level,
                    part: WhirPart::Leaves,
                    expected: fixture.proof.levels[level].leaves.len(),
                    actual: proof.levels[level].leaves.len(),
                })
            );
            let mut proof = fixture.proof.clone();
            if append {
                proof.levels[level].digests.push([0; 32]);
            } else if proof.levels[level].digests.pop().is_none() {
                continue;
            }
            assert_eq!(
                fixture.verify(&fixture.commitment, &proof, &request),
                Err(WhirError::CountMismatch {
                    level,
                    part: WhirPart::Digests,
                    expected: fixture.proof.levels[level].digests.len(),
                    actual: proof.levels[level].digests.len(),
                })
            );
        }
    }
    for round in 0..fixture.proof.closing_rounds.len() {
        for coefficient in 0..2 {
            let mut proof = fixture.proof.clone();
            proof.closing_rounds[round][coefficient] =
                changed(proof.closing_rounds[round][coefficient]);
            assert_eq!(
                fixture.verify(&fixture.commitment, &proof, &request),
                Err(WhirError::ClosingIdentityMismatch)
            );
        }
    }
    let (_, wire) = fixture.wire();
    let mut offset = 0;
    for (index, level) in fixture.proof.levels.iter().enumerate() {
        offset += 48 * level.rounds.len()
            + match &level.message {
                WhirLevelMessage::Intermediate { .. } => 56,
                WhirLevelMessage::Final { values } => 24 * values.len(),
            };
        for count_offset in [
            offset,
            offset + 4 + level.leaves.len() * fixture.schedule.levels()[index].leaf_bytes,
        ] {
            let mut mutated = wire.clone();
            mutated[count_offset] ^= 1;
            if let Some(proof) = WhirOpeningProof::read_with_schedule(&mutated, &fixture.schedule) {
                assert!(fixture
                    .verify(&fixture.commitment, &proof, &request)
                    .is_err());
            }
        }
        offset += 8
            + level.leaves.len() * fixture.schedule.levels()[index].leaf_bytes
            + 32 * level.digests.len();
    }
}

#[test]
fn rejections_each_honest_wire_field_class() {
    for t in [6, 11] {
        reject_wire_field_classes(&Fixture::new(t));
    }
}

#[test]
fn rejections_wrong_claims_points_geometry_and_stale_transcript() {
    for t in [6, 11] {
        let fixture = Fixture::new(t);
        let mut columns = fixture.columns.clone();
        columns[0] += F128::one();
        let request = BitsOpening {
            columns: &columns,
            ..fixture.request()
        };
        assert_eq!(
            fixture.verify(&fixture.commitment, &fixture.proof, &request),
            Err(WhirError::ClosingIdentityMismatch)
        );
        let a = eq_index(&fixture.rho, 0).unwrap();
        let b = eq_index(&fixture.rho, 1).unwrap();
        let mut columns = fixture.columns.clone();
        columns[0] += b;
        columns[1] += a;
        let request = BitsOpening {
            columns: &columns,
            ..fixture.request()
        };
        assert_eq!(request.value(), fixture.request().value());
        assert_eq!(
            fixture.verify(&fixture.commitment, &fixture.proof, &request),
            Err(WhirError::ClosingIdentityMismatch)
        );
        let mut rho = fixture.rho.clone();
        rho[7] += F128::one();
        let request = BitsOpening {
            column_point: &rho,
            ..fixture.request()
        };
        assert_eq!(
            fixture.verify(&fixture.commitment, &fixture.proof, &request),
            Err(WhirError::ClosingIdentityMismatch)
        );
        let mut cycle = fixture.cycle.clone();
        cycle[0] += F128::one();
        let request = BitsOpening {
            cycle_point: &cycle,
            ..fixture.request()
        };
        assert_eq!(
            fixture.verify(&fixture.commitment, &fixture.proof, &request),
            Err(WhirError::ClosingIdentityMismatch)
        );
        let request = BitsOpening {
            geometry: BitsGeometry { log_T: t + 1 },
            ..fixture.request()
        };
        assert_eq!(
            fixture.verify(&fixture.commitment, &fixture.proof, &request),
            Err(WhirError::GeometryMismatch {
                expected: fixture.geometry,
                actual: request.geometry
            })
        );
        let mut transcript = fixture.initial.clone();
        transcript.append_bytes(b"stale");
        let state = WhirBits::verify_commit_with_schedule(
            fixture.geometry,
            fixture.schedule.clone(),
            &fixture.commitment,
            &mut transcript,
        )
        .unwrap();
        for call in fixture
            .opening
            .calls
            .iter()
            .skip(fixture.committed.calls.len())
        {
            match call {
                Call::Append(bytes) => transcript.append_bytes(bytes),
                Call::Draw(_) => {
                    let _draw = transcript.challenge();
                }
            }
        }
        assert!(WhirBits::verify_opening_with_schedule(
            fixture.schedule.clone(),
            state,
            &fixture.request(),
            &fixture.proof,
            &mut transcript
        )
        .is_err());
        if fixture.proof.levels.len() > 1 {
            let mut proof = fixture.proof.clone();
            proof.levels.swap(0, 1);
            assert!(fixture
                .verify(&fixture.commitment, &proof, &fixture.request())
                .is_err());
        }
    }
}

#[test]
fn rejections_changed_row_and_far_codeword() {
    for t in [6, 11] {
        let changed_row = Fixture::build(t, None, |_, state| {
            let mut rows = state.rows.to_vec();
            rows[0][0] ^= 1;
            state.rows = rows.into();
        });
        assert!(changed_row
            .verify(
                &changed_row.commitment,
                &changed_row.proof,
                &changed_row.request()
            )
            .is_err());
        let far = Fixture::build(t, None, |commitment, state| {
            let mut rng = Words(0x310c_43a1_497e_0d9b);
            for symbol in &mut state.codeword {
                *symbol = F64::from_raw(rng.next());
            }
            let lanes = Schedule::new(state.geometry).unwrap().levels()[0]
                .lanes()
                .unwrap();
            state.tree = MerkleTree::build_canonical(&state.codeword, 2 * lanes).unwrap();
            commitment.root = *state.tree.root();
        });
        assert!(matches!(
            far.verify(&far.commitment, &far.proof, &far.request()),
            Err(WhirError::FinalCodeMismatch { .. } | WhirError::ClosingIdentityMismatch)
        ));
    }
}

#[test]
fn final_code_mismatch_names_the_first_changed_position_at_t6() {
    let fixture = Fixture::new(6);
    let mut proof = fixture.proof.clone();
    if let WhirLevelMessage::Final { values } = &mut proof.levels[0].message {
        values[0] = changed(values[0]);
    }
    assert_eq!(
        fixture.verify(&fixture.commitment, &proof, &fixture.request()),
        Err(WhirError::FinalCodeMismatch { position: 0 })
    );
}

#[test]
fn commit_sample_mismatch_reaches_its_terminal_check_at_t6() {
    let fixture = Fixture::build(6, None, |commitment, state| {
        commitment.lane_values[0] = changed(commitment.lane_values[0]);
        state.lane_values[0] = commitment.lane_values[0];
    });
    assert_eq!(
        fixture.verify(&fixture.commitment, &fixture.proof, &fixture.request()),
        Err(WhirError::CommitSampleMismatch)
    );
}

#[test]
fn prover_geometry_and_row_errors_preserve_fields() {
    let rows: Arc<[[u64; 4]]> = Arc::from([[0; 4]; 2]);
    for t in [0, 33] {
        let geometry = BitsGeometry { log_T: t };
        let mut transcript = Recorded::new(b"whir_contract");
        assert!(
            matches!(commit(geometry, &rows, &mut transcript), Err(WhirError::UnsupportedGeometry { log_T }) if log_T == t)
        );
    }
    let mut transcript = Recorded::new(b"whir_contract");
    assert!(matches!(
        commit(BitsGeometry { log_T: 2 }, &rows, &mut transcript),
        Err(WhirError::Shape {
            part: WhirPart::Rows,
            expected: 4,
            actual: 2
        })
    ));
}

#[test]
fn transcript_calls_and_states_match_literal_protocol_grammar() {
    for (t, levels, commit_coordinates, residual) in [
        (6, vec![(5, None, None)], 2, 2),
        (10, vec![(5, Some(6), None), (4, None, Some(59usize))], 6, 2),
        (11, vec![(5, Some(7), Some(254)), (4, None, Some(62))], 7, 3),
    ] {
        let fixture = Fixture::new(t);
        let mut transcript = fixture.initial.clone();
        let state =
            WhirBits::verify_commit(&(), fixture.geometry, &fixture.commitment, &mut transcript)
                .unwrap();
        assert_eq!(transcript.calls, fixture.committed.calls);
        assert_eq!(transcript.state(), fixture.committed.state());
        assert_eq!(transcript.calls, fixture.prover_committed.calls);
        assert_eq!(transcript.state(), fixture.prover_committed.state());
        let mut expected = vec![(0, b"whir_commit".to_vec()), (0, vec![0; 32])];
        expected.extend((0..2 * commit_coordinates).map(|_| (1, Vec::new())));
        expected.push((0, b"whir_ood".to_vec()));
        expected.push((0, vec![0; 768]));
        assert_grammar(
            &fixture.committed.calls[fixture.initial.calls.len()..],
            &expected,
        );
        for call in fixture
            .opening
            .calls
            .iter()
            .skip(fixture.committed.calls.len())
        {
            match call {
                Call::Append(bytes) => transcript.append_bytes(bytes),
                Call::Draw(_) => {
                    let _draw = transcript.challenge();
                }
            }
        }
        WhirBits::verify_opening(
            &(),
            state,
            &fixture.request(),
            &fixture.proof,
            &mut transcript,
        )
        .unwrap();
        assert_eq!(transcript.calls, fixture.final_transcript.calls);
        assert_eq!(transcript.state(), fixture.final_transcript.state());
        let mut expected = vec![(0, b"whir_open".to_vec()), (1, Vec::new()), (1, Vec::new())];
        for (index, (rounds, sample, queries)) in levels.into_iter().enumerate() {
            if index != 0 {
                expected.extend([(1, Vec::new()), (1, Vec::new())]);
            }
            for _ in 0..rounds {
                expected.extend([
                    (0, b"whir_round".to_vec()),
                    (0, vec![0; 48]),
                    (1, Vec::new()),
                    (1, Vec::new()),
                ]);
            }
            if let Some(coordinates) = sample {
                expected.extend([(0, b"whir_root".to_vec()), (0, vec![0; 32])]);
                expected.extend((0..2 * coordinates).map(|_| (1, Vec::new())));
                expected.extend([(0, b"whir_ood".to_vec()), (0, vec![0; 24])]);
            } else {
                expected.extend([
                    (0, b"whir_final".to_vec()),
                    (0, vec![0; 24 * (1 << residual)]),
                ]);
            }
            if let Some(queries) = queries {
                expected.extend((0..(4 * queries).div_ceil(16)).map(|_| (1, Vec::new())));
            }
        }
        for _ in 0..residual {
            expected.extend([
                (0, b"whir_round".to_vec()),
                (0, vec![0; 48]),
                (1, Vec::new()),
                (1, Vec::new()),
            ]);
        }
        assert_grammar(
            &fixture.final_transcript.calls[fixture.opening.calls.len()..],
            &expected,
        );
    }
}

fn assert_grammar(calls: &[Call], expected: &[(u8, Vec<u8>)]) {
    assert_eq!(calls.len(), expected.len());
    for (call, (kind, bytes)) in calls.iter().zip(expected) {
        match call {
            Call::Draw(_) => assert_eq!(*kind, 1),
            Call::Append(actual) => {
                assert_eq!(*kind, 0);
                if bytes.first().is_some_and(|byte| *byte != 0) {
                    let mut padded = bytes.clone();
                    padded.resize(32, 0);
                    assert_eq!(actual, &padded);
                } else {
                    assert_eq!(actual.len(), bytes.len());
                }
            }
        }
    }
}

#[test]
fn deterministic_commitment_and_opening_on_one_and_twelve_threads() {
    for t in [6, 11] {
        let mut baseline = None;
        for threads in [1, 12] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for _ in 0..2 {
                let bytes = pool.install(|| {
                    let fixture = Fixture::new(t);
                    fixture.accepted();
                    fixture.wire()
                });
                if let Some(expected) = &baseline {
                    assert_eq!(&bytes, expected);
                } else {
                    baseline = Some(bytes);
                }
            }
        }
        if t == 6 {
            let (commitment, proof) = baseline.unwrap();
            let commitment_digest: [u8; 32] = Blake2b::<U32>::digest(&commitment).into();
            let opening_digest: [u8; 32] = Blake2b::<U32>::digest(&proof).into();
            assert_eq!(
                commitment_digest,
                [
                    215, 42, 173, 58, 147, 142, 143, 159, 42, 114, 93, 166, 100, 94, 105, 216, 37,
                    224, 75, 97, 40, 31, 119, 106, 214, 185, 245, 149, 250, 165, 236, 49
                ]
            );
            assert_eq!(
                opening_digest,
                [
                    68, 235, 248, 238, 82, 154, 109, 67, 122, 4, 116, 77, 62, 121, 181, 39, 98, 83,
                    203, 12, 124, 243, 178, 32, 203, 96, 154, 194, 144, 147, 239, 210
                ]
            );
        }
    }
}
