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
use std::sync::Arc;

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

struct TranscriptSyncTarget<T: JoltTranscript> {
    ptr: *mut T,
}

unsafe impl<T: JoltTranscript> Send for TranscriptSyncTarget<T> {}
unsafe impl<T: JoltTranscript> Sync for TranscriptSyncTarget<T> {}

/// Bridge adapter: wraps a Jolt transcript pointer and implements Hachi's Transcript trait.
///
/// Uses a raw pointer internally because Hachi's `Transcript` trait requires `'static`,
/// but we need to borrow a Jolt transcript that has a limited lifetime. The adapter is
/// always used in a strictly scoped manner within a single prove/verify call.
pub struct JoltToHachiTranscript<T: JoltTranscript> {
    state: T,
    sync_target: Option<Arc<TranscriptSyncTarget<T>>>,
}

unsafe impl<T: JoltTranscript> Send for JoltToHachiTranscript<T> {}
unsafe impl<T: JoltTranscript> Sync for JoltToHachiTranscript<T> {}

impl<T: JoltTranscript> JoltToHachiTranscript<T> {
    pub fn new(transcript: &mut T) -> Self {
        Self {
            state: transcript.clone(),
            sync_target: Some(Arc::new(TranscriptSyncTarget {
                ptr: transcript as *mut T,
            })),
        }
    }

    fn inner(&mut self) -> &mut T {
        &mut self.state
    }

    #[inline]
    fn absorb_label(&mut self, label: &[u8]) {
        self.inner().append_bytes(b"hachi_label", label);
    }
}

impl<T: JoltTranscript> Clone for JoltToHachiTranscript<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            sync_target: self.sync_target.clone(),
        }
    }
}

impl<T: JoltTranscript> Drop for JoltToHachiTranscript<T> {
    fn drop(&mut self) {
        if let Some(target) = &self.sync_target {
            if Arc::strong_count(target) == 1 {
                // SAFETY: `sync_target` originates from `new(&mut T)` and remains valid for the
                // scoped lifetime of all adapter clones. Only the last surviving clone syncs back,
                // which preserves Hachi's clone-and-commit transcript pattern without letting
                // speculative clones overwrite the caller transcript.
                unsafe {
                    *target.ptr = self.state.clone();
                }
            }
        }
    }
}

impl<T: JoltTranscript> HachiTranscript<Fp128> for JoltToHachiTranscript<T> {
    fn new(_domain_label: &[u8]) -> Self {
        unimplemented!("use JoltToHachiTranscript::new(transcript) to wrap an existing transcript")
    }

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        self.absorb_label(_label);
        self.inner().append_bytes(b"hachi_bytes", bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &Fp128) {
        self.absorb_label(_label);
        let val = x.to_canonical_u128();
        self.inner()
            .append_bytes(b"hachi_field", &val.to_le_bytes());
    }

    fn append_serde<S: HachiSerialize>(&mut self, _label: &[u8], s: &S) {
        self.absorb_label(_label);
        let mut buf = Vec::with_capacity(s.serialized_size(HachiCompress::No));
        s.serialize_uncompressed(&mut buf)
            .expect("HachiSerialize should not fail");
        self.inner().append_bytes(b"hachi_serde", &buf);
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> Fp128 {
        self.absorb_label(_label);
        let jolt_challenge: JoltFp128 = self.inner().challenge_scalar();
        jolt_to_hachi(&jolt_challenge)
    }

    fn challenge_bytes(&mut self, _label: &[u8], len: usize) -> Vec<u8> {
        self.absorb_label(_label);
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            out.extend_from_slice(&self.inner().challenge_u128().to_le_bytes());
        }
        out.truncate(len);
        out
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
