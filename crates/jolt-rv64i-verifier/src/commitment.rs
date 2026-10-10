//! Bit-table commitment contract: the commit phase absorbs every commitment
//! message after the preamble and before front-end challenges. The opening
//! follows absorption of all column values and the draw of the column point.
//! Scheme labels are distinct from front-end labels. Its verifier authenticates
//! one table of bits whose multilinear extension equals `BitsOpening::value` at
//! the supplied point; packed honest-prover inputs alone do not establish this.
//! Larger-field challenges use consecutive little-endian scalar encodings.

use crate::points::eq_table;
use jolt_field::{CanonicalBytes, Zero, F128};
use jolt_transcript::Transcript;
use std::error::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// A bit table has 256 columns and `2^log_T` cycles; a scheme checks that its supported geometry admits the exponent.
pub struct BitsGeometry {
    /// The cycle exponent, admitted by the front end in `1..=32` and possibly bounded further by the scheme.
    pub log_T: usize,
}

/// The full column vector and the least-significant-bit-first opening points.
/// Schemes check lengths 8, `log_T`, and 256 before evaluating this request.
pub struct BitsOpening<'a> {
    /// Geometry of the table authenticated during the commit phase.
    pub geometry: BitsGeometry,
    /// Eight low-variable-first column coordinates, drawn after the column vector is absorbed.
    pub column_point: &'a [F128],
    /// `log_T` low-variable-first cycle coordinates from the final reduction batch.
    pub cycle_point: &'a [F128],
    /// Exactly 256 column evaluations at `cycle_point`, in bit-column index order.
    pub columns: &'a [F128],
}
impl BitsOpening<'_> {
    /// Evaluation of the column vector at the column point, after shape checks.
    pub fn value(&self) -> F128 {
        let Ok(weights) = eq_table(self.column_point) else {
            return F128::zero();
        };
        self.columns
            .iter()
            .zip(weights)
            .map(|(value, weight)| weight * *value)
            .sum()
    }
}

/// Canonical encoding of a scheme commitment or opening for the supplied bit-table geometry.
/// The decoder accepts exactly complete strings emitted by the writer and rejects malformed strings without panicking.
pub trait BitsWire: Sized {
    /// Appends this object's complete canonical bytes, without an envelope length prefix.
    fn write(&self, out: &mut Vec<u8>);
    /// Decodes the whole slice for this geometry, returning `None` for unsupported geometry or malformed bytes.
    fn read(bytes: &[u8], geometry: BitsGeometry) -> Option<Self>;
}
/// Authenticates a bit table and its multilinear opening through a transcript-coupled two-phase lifecycle.
/// Commit absorption precedes all front-end challenges; opening consumes retained state after `columns` are absorbed and the column point is drawn.
pub trait BitsCommitmentScheme {
    /// Public setup borrowed throughout commitment and opening verification.
    type VerifierSetup;
    /// Canonically encoded commit-phase messages bound to the table geometry.
    type Commitment: BitsWire;
    /// State retained after commit verification and consumed by one opening check.
    type VerifierState;
    /// Canonically encoded evidence that the committed table contains bits and has the requested evaluation.
    type OpeningProof: BitsWire;
    /// Typed errors for unsupported geometry, malformed commit messages or rejected openings.
    type Error: Error + Send + Sync + 'static;
    /// Absorbs all scheme commitment messages after the preamble and returns retained verification state.
    /// Returns a scheme error for unsupported geometry or rejected messages; labels must be distinct from front-end labels.
    fn verify_commit<T: Transcript<Challenge = F128>>(
        setup: &Self::VerifierSetup,
        geometry: BitsGeometry,
        commitment: &Self::Commitment,
        transcript: &mut T,
    ) -> Result<Self::VerifierState, Self::Error>;
    /// Consumes the retained state and authenticates one bit table whose extension at the supplied points equals `opening.value()`.
    /// Checks opening dimensions and returns a scheme error if the bit-table or evaluation claim fails.
    fn verify_opening<T: Transcript<Challenge = F128>>(
        setup: &Self::VerifierSetup,
        state: Self::VerifierState,
        opening: &BitsOpening<'_>,
        proof: &Self::OpeningProof,
        transcript: &mut T,
    ) -> Result<(), Self::Error>;
}

/// Draws ceil(length / 16) challenges, keeping the requested prefix of their bytes.
pub fn squeeze_bytes<T: Transcript<Challenge = F128>>(transcript: &mut T, out: &mut [u8]) {
    for chunk in out.chunks_mut(16) {
        let mut bytes = [0; 16];
        transcript.challenge().to_bytes_le(&mut bytes);
        for (target, byte) in chunk.iter_mut().zip(bytes) {
            *target = byte;
        }
    }
}
