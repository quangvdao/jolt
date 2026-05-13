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

macro_rules! impl_openings_transcript {
    ($transcript:ty) => {
        impl jolt_transcript::Transcript for $transcript {
            type Challenge = jolt_field::Fr;

            fn new(label: &'static [u8]) -> Self {
                <Self as crate::transcripts::Transcript>::new(label)
            }

            fn append_bytes(&mut self, bytes: &[u8]) {
                <Self as crate::transcripts::Transcript>::raw_append_bytes(self, bytes);
            }

            fn challenge(&mut self) -> Self::Challenge {
                <Self as crate::transcripts::Transcript>::challenge_scalar::<jolt_field::Fr>(self)
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
    use super::{Blake2bTranscript, Transcript};
    use jolt_transcript::{AppendToTranscript, Label, LabelWithCount};

    #[test]
    fn openings_transcript_impl_preserves_core_dory_absorption_layout() {
        let scalar = <jolt_field::Fr as crate::field::JoltField>::from_u64(7);

        let mut core_transcript = Blake2bTranscript::new(b"Dory");
        core_transcript.append_bytes(b"dory_bytes", b"abc");
        core_transcript.append_scalar(b"dory_field", &scalar);

        let mut openings_transcript =
            <Blake2bTranscript as jolt_transcript::Transcript>::new(b"Dory");
        jolt_transcript::Transcript::append(
            &mut openings_transcript,
            &LabelWithCount(b"dory_bytes", 3),
        );
        jolt_transcript::Transcript::append_bytes(&mut openings_transcript, b"abc");
        jolt_transcript::Transcript::append(&mut openings_transcript, &Label(b"dory_field"));
        scalar.append_to_transcript(&mut openings_transcript);

        assert_eq!(core_transcript.state, openings_transcript.state);

        let core_challenge = core_transcript.challenge_scalar::<jolt_field::Fr>();
        let openings_challenge = jolt_transcript::Transcript::challenge(&mut openings_transcript);
        assert_eq!(core_challenge, openings_challenge);
    }
}
