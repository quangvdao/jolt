//! Bit-table commitment contract: the commit phase absorbs every commitment
//! message after the preamble and before front-end challenges. The opening
//! follows absorption of all column values and the draw of the column point.
//! Scheme labels are distinct from front-end labels. Its verifier authenticates
//! one table of bits whose multilinear extension equals `BitsOpening::value` at
//! the supplied point; packed honest-prover inputs alone do not establish this.
//! Larger-field challenges use consecutive little-endian scalar encodings.

use crate::points::eq_index;
use jolt_field::{CanonicalBytes, Zero, F128};
use jolt_transcript::Transcript;
use std::error::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitsGeometry {
    pub log_T: usize,
}

/// The full column vector and the least-significant-bit-first opening points.
/// Schemes check lengths 8, `log_T`, and 256 before evaluating this request.
pub struct BitsOpening<'a> {
    pub geometry: BitsGeometry,
    pub column_point: &'a [F128],
    pub cycle_point: &'a [F128],
    pub columns: &'a [F128],
}
impl BitsOpening<'_> {
    /// Evaluation of the column vector at the column point, after shape checks.
    pub fn value(&self) -> F128 {
        self.columns
            .iter()
            .enumerate()
            .map(|(y, value)| {
                eq_index(self.column_point, y).map_or(F128::zero(), |weight| weight * *value)
            })
            .fold(F128::zero(), |sum, term| sum + term)
    }
}

/// A scheme decoder accepts exactly the bytes its writer emits for the geometry.
pub trait BitsWire: Sized {
    fn write(&self, out: &mut Vec<u8>);
    fn read(bytes: &[u8], geometry: BitsGeometry) -> Option<Self>;
}
pub trait BitsCommitmentScheme {
    type VerifierSetup;
    type Commitment: BitsWire;
    type VerifierState;
    type OpeningProof: BitsWire;
    type Error: Error + Send + Sync + 'static;
    fn verify_commit<T: Transcript<Challenge = F128>>(
        setup: &Self::VerifierSetup,
        geometry: BitsGeometry,
        commitment: &Self::Commitment,
        transcript: &mut T,
    ) -> Result<Self::VerifierState, Self::Error>;
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
