mod blake2b;
mod keccak;
#[cfg(feature = "transcript-poseidon")]
mod poseidon;
mod transcript;

pub use blake2b::Blake2bTranscript;
pub use keccak::KeccakTranscript;
#[cfg(feature = "transcript-poseidon")]
pub use poseidon::PoseidonTranscript;
pub use transcript::Transcript;

use jolt_field::Fr;
use jolt_transcript::Transcript as OpeningsTranscript;

macro_rules! impl_openings_transcript {
    ($transcript:ty) => {
        impl OpeningsTranscript for $transcript {
            type Challenge = Fr;

            fn new(label: &'static [u8]) -> Self {
                <Self as Transcript>::new(label)
            }

            fn append_bytes(&mut self, bytes: &[u8]) {
                <Self as Transcript>::raw_append_bytes(self, bytes);
            }

            fn challenge(&mut self) -> Self::Challenge {
                <Self as Transcript>::challenge_scalar::<Fr>(self)
            }

            fn state(&self) -> &[u8; 32] {
                &self.state
            }
        }
    };
}

impl_openings_transcript!(Blake2bTranscript);
impl_openings_transcript!(KeccakTranscript);

#[cfg(feature = "transcript-poseidon")]
impl_openings_transcript!(PoseidonTranscript);

#[cfg(test)]
mod tests {
    use super::{Blake2bTranscript, Transcript as CoreTranscript};
    use crate::field::JoltField;
    use jolt_field::Fr;
    use jolt_transcript::Transcript as OpeningsTranscript;
    use jolt_transcript::{AppendToTranscript, Label, LabelWithCount};

    #[test]
    fn openings_transcript_impl_preserves_core_dory_absorption_layout() {
        let scalar = Fr::from_u64(7);

        let mut core_transcript = <Blake2bTranscript as CoreTranscript>::new(b"Dory");
        CoreTranscript::append_bytes(&mut core_transcript, b"dory_bytes", b"abc");
        CoreTranscript::append_scalar(&mut core_transcript, b"dory_field", &scalar);

        let mut openings_transcript = <Blake2bTranscript as OpeningsTranscript>::new(b"Dory");
        OpeningsTranscript::append(&mut openings_transcript, &LabelWithCount(b"dory_bytes", 3));
        OpeningsTranscript::append_bytes(&mut openings_transcript, b"abc");
        OpeningsTranscript::append(&mut openings_transcript, &Label(b"dory_field"));
        scalar.append_to_transcript(&mut openings_transcript);

        assert_eq!(core_transcript.state, openings_transcript.state);

        let core_challenge = CoreTranscript::challenge_scalar::<Fr>(&mut core_transcript);
        let openings_challenge = OpeningsTranscript::challenge(&mut openings_transcript);
        assert_eq!(core_challenge, openings_challenge);
    }
}
