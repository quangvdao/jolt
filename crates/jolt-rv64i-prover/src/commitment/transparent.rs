//! Test stand-in for the binary-field RV64I protocol. Packed opening rows enforce
//! bit values, and collision resistance of their digest binds a unique table.
//! Commitment absorption binds that digest before the front-end challenges;
//! opening verifies its evaluation at the supplied points. The retained prover
//! state shares the rows. This scheme is not succinct: opening uses 32 bytes per
//! cycle and verification is linear in the trace length. It exists for end-to-end
//! protocol tests before a production scheme is attached, and is not for use
//! outside tests. Trace exponents above 20 are rejected before allocation.

use super::BitsCommitmentProver;
use blake2::{digest::consts::U32, Blake2b, Digest};
use jolt_field::{Zero, F128};
use jolt_rv64i_arith::{BitsRow, BITS_COLUMNS};
use jolt_rv64i_verifier::commitment::{BitsCommitmentScheme, BitsGeometry, BitsOpening, BitsWire};
use jolt_rv64i_verifier::points::{eq_index, PointsError};
use jolt_transcript::{Label, Transcript};
use std::sync::Arc;
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Digest of the cycle exponent followed by packed rows in cycle and word order.
pub struct TransparentCommitment(
    /// Exactly 32 canonical commitment bytes, absorbed after the `bits_commitment` label.
    pub [u8; 32],
);
#[derive(Clone, Debug, PartialEq, Eq)]
/// The complete packed table opened by the test scheme; its digest and multilinear evaluation are verified.
pub struct TransparentOpening(
    /// Exactly `2^log_T` rows, shared with retained prover state and encoded as four little-endian words per cycle.
    pub Arc<[BitsRow]>,
);
/// Retained prover commitment state sharing the packed table allocation until `open` consumes it.
pub struct TransparentState {
    geometry: BitsGeometry,
    commitment: TransparentCommitment,
    bits: Arc<[BitsRow]>,
}
/// Test-only bit-table scheme admitting exponents at most 20 and opening the complete packed table.
/// Commit absorbs its digest before front-end challenges; open consumes retained state after column absorption and point drawing.
pub struct TransparentBits;
#[derive(Debug, Error)]
/// Unsupported geometry, malformed table shape or a rejected digest or evaluation claim.
pub enum TransparentError {
    #[error("transparent bit-table exponent {log_T} exceeds 20")]
    /// The requested exponent exceeds the scheme's bound, checked before allocation.
    Dimension { log_T: usize },
    #[error("bit-table row count differs from its geometry")]
    /// The packed table does not contain the number of rows selected by its geometry.
    RowCount,
    #[error("opening has invalid point or column dimensions")]
    /// Opening points are not of widths 8 and `log_T`, or the column vector does not have 256 values.
    OpeningShape,
    #[error("opening geometry differs from the retained commitment geometry")]
    /// The opening's cycle exponent differs from the retained commitment geometry.
    Geometry,
    #[error("opening table digest differs from the commitment")]
    /// The disclosed packed table is not bound by the retained digest.
    Digest,
    #[error("opening value differs from the committed bit table")]
    /// The disclosed table evaluation disagrees with the supplied column-vector evaluation.
    Evaluation,
    #[error(transparent)]
    /// A supplied point does not fit the integer-index domain needed for evaluation.
    Points(#[from] PointsError),
}
impl TransparentBits {
    fn rows(geometry: BitsGeometry) -> Result<usize, TransparentError> {
        if geometry.log_T > 20 {
            return Err(TransparentError::Dimension {
                log_T: geometry.log_T,
            });
        }
        Ok(1_usize << geometry.log_T)
    }
    fn digest(
        geometry: BitsGeometry,
        bits: &[BitsRow],
    ) -> Result<TransparentCommitment, TransparentError> {
        if bits.len() != Self::rows(geometry)? {
            return Err(TransparentError::RowCount);
        }
        let mut hash = Blake2b::<U32>::new();
        hash.update((geometry.log_T as u64).to_le_bytes());
        for row in bits {
            for word in row {
                hash.update(word.to_le_bytes());
            }
        }
        Ok(TransparentCommitment(hash.finalize().into()))
    }
    fn value(opening: &BitsOpening<'_>, bits: &[BitsRow]) -> Result<F128, TransparentError> {
        if opening.column_point.len() != 8
            || opening.cycle_point.len() != opening.geometry.log_T
            || opening.columns.len() != BITS_COLUMNS
        {
            return Err(TransparentError::OpeningShape);
        }
        let mut value = F128::zero();
        let columns = (0..BITS_COLUMNS)
            .map(|y| eq_index(opening.column_point, y))
            .collect::<Result<Vec<_>, _>>()?;
        for (j, row) in bits.iter().enumerate() {
            let cycle = eq_index(opening.cycle_point, j)?;
            for (y, weight) in columns.iter().enumerate() {
                if row
                    .get(y / 64)
                    .is_some_and(|word| (word >> (y % 64)) & 1 != 0)
                {
                    value += cycle * *weight;
                }
            }
        }
        Ok(value)
    }
}
impl BitsWire for TransparentCommitment {
    /// Appends the 32-byte digest without a length prefix.
    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
    /// Accepts exactly 32 bytes for an exponent at most 20, otherwise returning `None`.
    fn read(bytes: &[u8], geometry: BitsGeometry) -> Option<Self> {
        let _ = TransparentBits::rows(geometry).ok()?;
        Some(Self(bytes.try_into().ok()?))
    }
}
impl BitsWire for TransparentOpening {
    /// Appends packed rows in cycle order, with each row's four words encoded little-endian.
    fn write(&self, out: &mut Vec<u8>) {
        for row in self.0.iter() {
            for word in row {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
    }
    /// Accepts exactly `32 * 2^log_T` bytes with exponent at most 20, otherwise returning `None` before allocation.
    fn read(bytes: &[u8], geometry: BitsGeometry) -> Option<Self> {
        let count = TransparentBits::rows(geometry).ok()?;
        if bytes.len() != count * 32 {
            return None;
        }
        let mut rows = Vec::with_capacity(count);
        for row in bytes.chunks_exact(32) {
            let mut words = [0; 4];
            for (word, bytes) in words.iter_mut().zip(row.chunks_exact(8)) {
                *word = u64::from_le_bytes(bytes.try_into().ok()?);
            }
            rows.push(words);
        }
        Some(Self(rows.into()))
    }
}
impl BitsCommitmentScheme for TransparentBits {
    type VerifierSetup = ();
    type Commitment = TransparentCommitment;
    type VerifierState = (BitsGeometry, TransparentCommitment);
    type OpeningProof = TransparentOpening;
    type Error = TransparentError;
    /// Absorbs the commitment label and digest and retains the geometry and digest for one opening check.
    /// Returns `Dimension` for an unsupported exponent before any front-end challenge is drawn.
    fn verify_commit<T: Transcript<Challenge = F128>>(
        _setup: &(),
        geometry: BitsGeometry,
        commitment: &TransparentCommitment,
        transcript: &mut T,
    ) -> Result<Self::VerifierState, TransparentError> {
        let _ = Self::rows(geometry)?;
        transcript.append(&Label(b"bits_commitment"));
        transcript.append_bytes(&commitment.0);
        Ok((geometry, *commitment))
    }
    /// Consumes retained verifier state and checks geometry, digest, shape and the full packed-table evaluation.
    /// Returns the first typed opening failure; the caller has already absorbed columns and drawn the column point.
    fn verify_opening<T: Transcript<Challenge = F128>>(
        _setup: &(),
        state: Self::VerifierState,
        opening: &BitsOpening<'_>,
        proof: &TransparentOpening,
        _transcript: &mut T,
    ) -> Result<(), TransparentError> {
        let _ = Self::rows(opening.geometry)?;
        if state.0.log_T != opening.geometry.log_T {
            return Err(TransparentError::Geometry);
        }
        if Self::digest(opening.geometry, &proof.0)? != state.1 {
            return Err(TransparentError::Digest);
        }
        if Self::value(opening, &proof.0)? != opening.value() {
            return Err(TransparentError::Evaluation);
        }
        Ok(())
    }
}
impl BitsCommitmentProver for TransparentBits {
    type ProverSetup = ();
    type ProverState = TransparentState;
    /// Validates geometry and row count, absorbs the digest and retains a clone of the same row handle.
    /// Returns `Dimension` or `RowCount` for unsupported geometry or inconsistent rows without copying the table.
    fn commit<T: Transcript<Challenge = F128>>(
        _setup: &(),
        geometry: BitsGeometry,
        bits: &Arc<[BitsRow]>,
        transcript: &mut T,
    ) -> Result<(TransparentCommitment, TransparentState), TransparentError> {
        let commitment = Self::digest(geometry, bits)?;
        let _ = Self::verify_commit(&(), geometry, &commitment, transcript)?;
        Ok((
            commitment,
            TransparentState {
                geometry,
                commitment,
                bits: Arc::clone(bits),
            },
        ))
    }
    /// Consumes retained prover state, checks the requested opening and transfers its shared row handle to the proof.
    /// Returns a typed opening error on inconsistent geometry, dimensions or evaluation.
    fn open<T: Transcript<Challenge = F128>>(
        _setup: &(),
        state: TransparentState,
        opening: &BitsOpening<'_>,
        transcript: &mut T,
    ) -> Result<TransparentOpening, TransparentError> {
        let proof = TransparentOpening(state.bits);
        Self::verify_opening(
            &(),
            (state.geometry, state.commitment),
            opening,
            &proof,
            transcript,
        )?;
        Ok(proof)
    }
}
