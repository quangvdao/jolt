use crate::field::fp128::JoltFp128;
use crate::transcripts::Transcript as JoltTranscript;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use hachi_pcs::algebra::Prime128M8M4M1M0;
use hachi_pcs::primitives::serialization::Compress as HachiCompress;
use hachi_pcs::protocol::transcript::Transcript as HachiTranscript;
use hachi_pcs::HachiSerialize;
use std::io::{Read, Write};

pub type Fp128 = Prime128M8M4M1M0;

#[inline]
pub fn jolt_to_hachi(f: &JoltFp128) -> Fp128 {
    // SAFETY: JoltFp128 is repr(transparent) over Prime128M8M4M1M0.
    unsafe { std::mem::transmute_copy(f) }
}

#[inline]
#[allow(dead_code)]
pub fn hachi_to_jolt(f: &Fp128) -> JoltFp128 {
    // SAFETY: JoltFp128 is repr(transparent) over Prime128M8M4M1M0.
    unsafe { std::mem::transmute_copy(f) }
}

/// Bridge adapter: wraps a Jolt transcript pointer and implements Hachi's Transcript trait.
///
/// Uses a raw pointer internally because Hachi's `Transcript` trait requires `'static`,
/// but we need to borrow a Jolt transcript that has a limited lifetime. The adapter is
/// always used in a strictly scoped manner within a single prove/verify call.
pub struct JoltToHachiTranscript<T: JoltTranscript> {
    transcript: *mut T,
}

unsafe impl<T: JoltTranscript> Send for JoltToHachiTranscript<T> {}
unsafe impl<T: JoltTranscript> Sync for JoltToHachiTranscript<T> {}

impl<T: JoltTranscript> JoltToHachiTranscript<T> {
    pub fn new(transcript: &mut T) -> Self {
        Self {
            transcript: transcript as *mut T,
        }
    }

    fn inner(&mut self) -> &mut T {
        // SAFETY: The pointer is valid for the duration of the prove/verify call
        // that created this adapter.
        unsafe { &mut *self.transcript }
    }
}

impl<T: JoltTranscript> Clone for JoltToHachiTranscript<T> {
    fn clone(&self) -> Self {
        unimplemented!("JoltToHachiTranscript is not clonable; Clone bound exists for trait compat")
    }
}

impl<T: JoltTranscript> HachiTranscript<Fp128> for JoltToHachiTranscript<T> {
    fn new(_domain_label: &[u8]) -> Self {
        unimplemented!("use JoltToHachiTranscript::new(transcript) to wrap an existing transcript")
    }

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        self.inner().append_bytes(b"hachi_bytes", bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &Fp128) {
        let val = x.to_canonical_u128();
        self.inner()
            .append_bytes(b"hachi_field", &val.to_le_bytes());
    }

    fn append_serde<S: HachiSerialize>(&mut self, _label: &[u8], s: &S) {
        let mut buf = Vec::with_capacity(s.serialized_size(HachiCompress::No));
        s.serialize_uncompressed(&mut buf)
            .expect("HachiSerialize should not fail");
        self.inner().append_bytes(b"hachi_serde", &buf);
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> Fp128 {
        let jolt_challenge: JoltFp128 = self.inner().challenge_scalar();
        jolt_to_hachi(&jolt_challenge)
    }
}

/// Newtype wrapper that provides arkworks `CanonicalSerialize`/`CanonicalDeserialize`
/// for Hachi types. These are stub implementations: the actual serialization path for
/// Hachi types uses `HachiSerialize`/`HachiDeserialize`. The arkworks traits exist
/// solely to satisfy Jolt's `CommitmentScheme` associated type bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArkBridge<T: Send + Sync>(pub T);

impl<T: Default + Send + Sync> Default for ArkBridge<T> {
    fn default() -> Self {
        Self(T::default())
    }
}

impl<T: Send + Sync> Valid for ArkBridge<T> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

fn ark_to_hachi_compress(c: Compress) -> HachiCompress {
    match c {
        Compress::Yes => HachiCompress::Yes,
        Compress::No => HachiCompress::No,
    }
}

impl<T: Send + Sync + HachiSerialize> CanonicalSerialize for ArkBridge<T> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0
            .serialize_with_mode(writer, ark_to_hachi_compress(compress))
            .map_err(|e| SerializationError::IoError(std::io::Error::other(e.to_string())))
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(ark_to_hachi_compress(compress))
    }
}

impl<T: Send + Sync> CanonicalDeserialize for ArkBridge<T> {
    fn deserialize_with_mode<R: Read>(
        _reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        unimplemented!("Hachi types use HachiDeserialize, not CanonicalDeserialize")
    }
}
