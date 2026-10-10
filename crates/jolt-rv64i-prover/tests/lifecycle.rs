//! Commitment phases and partial openings on the complete binary-field RV64I transcript.
#![expect(
    clippy::unwrap_used,
    reason = "invalid fixtures fail the enclosing test"
)]
#[expect(
    dead_code,
    reason = "shared machine helpers serve the complete protocol corpus"
)]
mod support;

use jolt_field::{One, Zero, F128};
use jolt_rv64i_arith::{BitsRow, BITS_COLUMNS};
use jolt_rv64i_prover::backend::Rv64iBackend;
use jolt_rv64i_prover::commitment::{
    transparent::{
        TransparentBits, TransparentCommitment, TransparentError, TransparentOpening,
        TransparentState,
    },
    BitsCommitmentProver,
};
use jolt_rv64i_prover::prover::{prove_with_transcript, ProverPreprocessing};
use jolt_rv64i_verifier::commitment::{
    squeeze_bytes, BitsCommitmentScheme, BitsGeometry, BitsOpening, BitsWire,
};
use jolt_rv64i_verifier::preprocessing::VerifierPreprocessing;
use jolt_rv64i_verifier::statement::CheckedInputs;
use jolt_rv64i_verifier::transcript::{preamble, Rv64iTranscript};
use jolt_rv64i_verifier::verifier::verify_with_transcript;
use jolt_transcript::{Label, Transcript};
use std::sync::Arc;
use thiserror::Error;

fn basis(point: &[F128], index: usize) -> F128 {
    point
        .iter()
        .enumerate()
        .map(|(bit, value)| {
            if (index >> bit) & 1 == 0 {
                F128::one() + *value
            } else {
                *value
            }
        })
        .product()
}
fn columns(bits: &[BitsRow], cycle: &[F128]) -> Vec<F128> {
    (0..BITS_COLUMNS)
        .map(|column| {
            bits.iter()
                .enumerate()
                .map(|(j, row)| {
                    if (row[column / 64] >> (column % 64)) & 1 == 0 {
                        F128::zero()
                    } else {
                        basis(cycle, j)
                    }
                })
                .sum()
        })
        .collect()
}
#[derive(Debug, Error)]
enum LifecycleError {
    #[error(transparent)]
    Transparent(#[from] TransparentError),
    #[error("commitment message does not bind the scheme's challenge")]
    Message,
    #[error("partial opening {index} differs from the committed table")]
    Partial { index: usize },
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct LifecycleCommitment {
    digest: TransparentCommitment,
    message: [u8; 24],
}
impl BitsWire for LifecycleCommitment {
    fn write(&self, out: &mut Vec<u8>) {
        self.digest.write(out);
        out.extend_from_slice(&self.message);
    }
    fn read(bytes: &[u8], geometry: BitsGeometry) -> Option<Self> {
        if bytes.len() != 56 {
            return None;
        }
        Some(Self {
            digest: TransparentCommitment::read(&bytes[..32], geometry)?,
            message: bytes[32..].try_into().ok()?,
        })
    }
}
struct LifecycleBits;
impl LifecycleBits {
    fn message<T: Transcript<Challenge = F128>>(transcript: &mut T) -> [u8; 24] {
        let mut message = [0; 24];
        squeeze_bytes(transcript, &mut message);
        message
    }
    fn append_message<T: Transcript<Challenge = F128>>(transcript: &mut T, message: &[u8; 24]) {
        transcript.append(&Label(b"bits_phase_challenge"));
        transcript.append_bytes(message);
    }
    fn partials(opening: &BitsOpening<'_>, bits: &[BitsRow]) -> Result<(), LifecycleError> {
        for index in 0..64 {
            let from_columns: F128 = (0..4)
                .map(|high| {
                    basis(&opening.column_point[6..], high) * opening.columns[index + 64 * high]
                })
                .sum();
            let from_table: F128 = bits
                .iter()
                .enumerate()
                .map(|(j, row)| {
                    let column: F128 = (0..4)
                        .filter(|high| (row[*high] >> index) & 1 != 0)
                        .map(|high| basis(&opening.column_point[6..], high))
                        .sum();
                    basis(opening.cycle_point, j) * column
                })
                .sum();
            if from_columns != from_table {
                return Err(LifecycleError::Partial { index });
            }
        }
        Ok(())
    }
}
impl BitsCommitmentScheme for LifecycleBits {
    type VerifierSetup = ();
    type Commitment = LifecycleCommitment;
    type VerifierState = <TransparentBits as BitsCommitmentScheme>::VerifierState;
    type OpeningProof = TransparentOpening;
    type Error = LifecycleError;
    fn verify_commit<T: Transcript<Challenge = F128>>(
        (): &(),
        geometry: BitsGeometry,
        commitment: &LifecycleCommitment,
        transcript: &mut T,
    ) -> Result<Self::VerifierState, Self::Error> {
        let state = TransparentBits::verify_commit(&(), geometry, &commitment.digest, transcript)?;
        let challenge = Self::message(transcript);
        if challenge != commitment.message {
            return Err(LifecycleError::Message);
        }
        Self::append_message(transcript, &commitment.message);
        Ok(state)
    }
    fn verify_opening<T: Transcript<Challenge = F128>>(
        (): &(),
        state: Self::VerifierState,
        opening: &BitsOpening<'_>,
        proof: &TransparentOpening,
        transcript: &mut T,
    ) -> Result<(), Self::Error> {
        TransparentBits::verify_opening(&(), state, opening, proof, transcript)?;
        Self::partials(opening, &proof.0)
    }
}
impl BitsCommitmentProver for LifecycleBits {
    type ProverSetup = ();
    type ProverState = TransparentState;
    fn commit<T: Transcript<Challenge = F128>>(
        (): &(),
        geometry: BitsGeometry,
        bits: &Arc<[BitsRow]>,
        transcript: &mut T,
    ) -> Result<(LifecycleCommitment, Self::ProverState), Self::Error> {
        let (digest, state) = TransparentBits::commit(&(), geometry, bits, transcript)?;
        let message = Self::message(transcript);
        Self::append_message(transcript, &message);
        Ok((LifecycleCommitment { digest, message }, state))
    }
    fn open<T: Transcript<Challenge = F128>>(
        (): &(),
        state: TransparentState,
        opening: &BitsOpening<'_>,
        transcript: &mut T,
    ) -> Result<TransparentOpening, Self::Error> {
        let proof = TransparentBits::open(&(), state, opening, transcript)?;
        Self::partials(opening, &proof.0)?;
        Ok(proof)
    }
}

#[test]
fn commitment_lifecycle_binds_its_challenges_before_front_end_draws_and_checks_all_partials() {
    let (statement, source, witness) = support::counting_loop();
    let verifier = VerifierPreprocessing::<LifecycleBits>::new(
        Arc::clone(source.shared_bytecode()),
        source.image().to_vec(),
        (),
    )
    .unwrap();
    let preprocessing = ProverPreprocessing {
        verifier,
        scheme: (),
    };
    let (proof, prover) = prove_with_transcript::<LifecycleBits, Rv64iTranscript>(
        &preprocessing,
        &Rv64iBackend::reference(),
        &statement,
        &witness,
    )
    .unwrap();
    let verifier = verify_with_transcript::<LifecycleBits, Rv64iTranscript>(
        &preprocessing.verifier,
        &statement,
        &proof,
    )
    .unwrap();
    assert_eq!(prover.state(), verifier.state());
    let checked = CheckedInputs::new(&preprocessing.verifier, &statement, &proof).unwrap();
    let mut late: Rv64iTranscript = preamble(&checked);
    let _first_tau = late.challenge_vector(8 + checked.log_T());
    assert!(matches!(
        LifecycleBits::verify_commit(
            &(),
            BitsGeometry {
                log_T: checked.log_T()
            },
            &proof.bits_commitment,
            &mut late
        ),
        Err(LifecycleError::Message)
    ));
}

#[test]
fn transparent_scheme_rejects_changed_bits_columns_geometry_and_point_shapes() {
    let geometry = BitsGeometry { log_T: 3 };
    let bits: Arc<[BitsRow]> = (0..8_u64)
        .map(|j| [j.wrapping_mul(0x9137), !j, j.rotate_left(13), 0])
        .collect();
    let rho: Vec<_> = (0..8).map(|i| F128::from_raw(0x100 + i)).collect();
    let cycle: Vec<_> = (0..3).map(|i| F128::from_raw(0x200 + i)).collect();
    let values = columns(&bits, &cycle);
    let opening = BitsOpening {
        geometry,
        column_point: &rho,
        cycle_point: &cycle,
        columns: &values,
    };
    let mut transcript = Rv64iTranscript::new(b"bits-stand-in");
    let (commitment, state) =
        TransparentBits::commit(&(), geometry, &bits, &mut transcript).unwrap();
    let proof = TransparentBits::open(&(), state, &opening, &mut transcript).unwrap();
    assert!(Arc::ptr_eq(&proof.0, &bits));
    let retained = (geometry, commitment);
    TransparentBits::verify_opening(&(), retained, &opening, &proof, &mut transcript).unwrap();
    let mut cancelling = values.clone();
    cancelling[0] += basis(&rho, 1);
    cancelling[1] += basis(&rho, 0);
    let cancelled = BitsOpening {
        columns: &cancelling,
        ..opening
    };
    TransparentBits::verify_opening(&(), retained, &cancelled, &proof, &mut transcript).unwrap();
    assert!(matches!(
        LifecycleBits::verify_opening(&(), retained, &cancelled, &proof, &mut transcript),
        Err(LifecycleError::Partial { index: 0 })
    ));
    let mut changed_bits = bits.to_vec();
    changed_bits[0][0] ^= 1;
    assert!(matches!(
        TransparentBits::verify_opening(
            &(),
            retained,
            &opening,
            &TransparentOpening(changed_bits.into()),
            &mut transcript
        ),
        Err(TransparentError::Digest)
    ));
    let mut changed_values = values.clone();
    changed_values[0] += F128::one();
    assert!(matches!(
        TransparentBits::verify_opening(
            &(),
            retained,
            &BitsOpening {
                columns: &changed_values,
                ..opening
            },
            &proof,
            &mut transcript
        ),
        Err(TransparentError::Evaluation)
    ));
    assert!(matches!(
        TransparentBits::verify_opening(
            &(),
            retained,
            &BitsOpening {
                geometry: BitsGeometry { log_T: 4 },
                ..opening
            },
            &proof,
            &mut transcript
        ),
        Err(TransparentError::Geometry)
    ));
    for malformed in [
        BitsOpening {
            column_point: &rho[..7],
            ..opening
        },
        BitsOpening {
            cycle_point: &cycle[..2],
            ..opening
        },
        BitsOpening {
            columns: &values[..255],
            ..opening
        },
    ] {
        assert!(matches!(
            TransparentBits::verify_opening(&(), retained, &malformed, &proof, &mut transcript),
            Err(TransparentError::OpeningShape)
        ));
    }
    let unsupported = BitsOpening {
        geometry: BitsGeometry { log_T: 21 },
        ..opening
    };
    assert!(matches!(
        TransparentBits::verify_opening(&(), retained, &unsupported, &proof, &mut transcript),
        Err(TransparentError::Dimension { log_T: 21 })
    ));
    let (_, second_state) = TransparentBits::commit(&(), geometry, &bits, &mut transcript).unwrap();
    assert!(matches!(
        TransparentBits::open(&(), second_state, &unsupported, &mut transcript),
        Err(TransparentError::Dimension { log_T: 21 })
    ));
    let mut other_cycle = cycle.clone();
    other_cycle[0] += F128::one();
    assert!(matches!(
        TransparentBits::verify_opening(
            &(),
            retained,
            &BitsOpening {
                cycle_point: &other_cycle,
                ..opening
            },
            &proof,
            &mut transcript
        ),
        Err(TransparentError::Evaluation)
    ));
}

#[test]
fn transparent_wire_lengths_and_unsupported_exponents_are_rejected_before_table_allocation() {
    let geometry = BitsGeometry { log_T: 3 };
    let bytes = vec![0; 257];
    for length in 0..bytes.len() {
        assert_eq!(
            TransparentOpening::read(&bytes[..length], geometry).is_some(),
            length == 256
        );
        assert_eq!(
            TransparentCommitment::read(&bytes[..length], geometry).is_some(),
            length == 32
        );
    }
    assert!(TransparentOpening::read(&bytes, geometry).is_none());
    let unsupported = BitsGeometry { log_T: 21 };
    let bits: Arc<[BitsRow]> = Arc::from([]);
    let mut transcript = Rv64iTranscript::new(b"bits-stand-in-bound");
    assert!(matches!(
        TransparentBits::commit(&(), unsupported, &bits, &mut transcript),
        Err(TransparentError::Dimension { log_T: 21 })
    ));
    assert!(matches!(
        TransparentBits::verify_commit(
            &(),
            unsupported,
            &TransparentCommitment([0; 32]),
            &mut transcript
        ),
        Err(TransparentError::Dimension { log_T: 21 })
    ));
    assert!(TransparentOpening::read(&[], unsupported).is_none());
    assert!(TransparentCommitment::read(&[0; 32], unsupported).is_none());
}
