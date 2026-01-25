use crate::zkvm::streaming_decode::{DecodeError, SliceReader};
use crate::{
    field::JoltField, poly::commitment::commitment_scheme::RecursionExt, transcripts::Transcript,
    zkvm::proof_serialization::JoltProof,
};
use ark_serialize::CanonicalSerialize;
use crate::zkvm::recursion_proof_bundle;

/// Proof bundle magic header.
///
/// Chosen to be short and unlikely to collide with canonical-serialized arkworks types.
pub const PROOF_BUNDLE_MAGIC: &[u8; 8] = b"JOLTBDL\0";

/// Proof bundle format version.
pub const PROOF_BUNDLE_VERSION: u32 = 2;

/// Proof record format version (inside the proof bundle).
pub const PROOF_RECORD_VERSION: u32 = 3;

/// A streaming view over a proof bundle byte slice.
///
/// Layout (all integers little-endian):
/// - [8]  magic
/// - [u32] version
/// - [u32] preprocessing_len
/// - [preprocessing_len] preprocessing_bytes
/// - [u32] n
/// - repeated n times:
///   - [u32] device_len
///   - [device_len] device_bytes
///   - [u32] proof_len
///   - [proof_len] proof_bytes
pub struct ProofBundleReader<'a> {
    version: u32,
    preprocessing: &'a [u8],
    remaining: SliceReader<'a>,
    remaining_entries: u32,
}

pub struct ProofBundleEntry<'a> {
    pub device_bytes: &'a [u8],
    pub proof_bytes: &'a [u8],
}

impl<'a> ProofBundleReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self, DecodeError> {
        let mut r = SliceReader::new(bytes);
        let magic = r.take(PROOF_BUNDLE_MAGIC.len())?;
        if magic != PROOF_BUNDLE_MAGIC {
            return Err(DecodeError::Invalid("bad proof bundle magic"));
        }
        let version = r.read_u32_le()?;
        if version != 1 && version != 2 {
            return Err(DecodeError::Invalid("unsupported proof bundle version"));
        }
        let preprocessing_len = r.read_u32_le()? as usize;
        let preprocessing = r.take(preprocessing_len)?;
        let n = r.read_u32_le()?;
        Ok(Self {
            version,
            preprocessing,
            remaining: r,
            remaining_entries: n,
        })
    }

    #[inline]
    pub fn version(&self) -> u32 {
        self.version
    }

    #[inline]
    pub fn preprocessing_bytes(&self) -> &'a [u8] {
        self.preprocessing
    }

    #[inline]
    pub fn remaining_entries(&self) -> u32 {
        self.remaining_entries
    }

    pub fn next_entry(&mut self) -> Result<Option<ProofBundleEntry<'a>>, DecodeError> {
        if self.remaining_entries == 0 {
            return Ok(None);
        }
        self.remaining_entries -= 1;
        let device_bytes = self.remaining.read_bytes_u32_len()?;
        let proof_bytes = self.remaining.read_bytes_u32_len()?;
        Ok(Some(ProofBundleEntry {
            device_bytes,
            proof_bytes,
        }))
    }
}

/// Write a proof bundle.
///
/// This is used on the host side to provide an input to the recursion guest verifier
/// that is amenable to streaming verification.
pub fn write_proof_bundle(
    mut out: impl ark_std::io::Write,
    preprocessing_bytes: &[u8],
    entries: &[(Vec<u8>, Vec<u8>)], // (device_bytes, proof_bytes)
) -> Result<(), ark_std::io::Error> {
    out.write_all(PROOF_BUNDLE_MAGIC)?;
    out.write_all(&PROOF_BUNDLE_VERSION.to_le_bytes())?;
    out.write_all(&(preprocessing_bytes.len() as u32).to_le_bytes())?;
    out.write_all(preprocessing_bytes)?;
    out.write_all(&(entries.len() as u32).to_le_bytes())?;
    for (device_bytes, proof_bytes) in entries {
        out.write_all(&(device_bytes.len() as u32).to_le_bytes())?;
        out.write_all(device_bytes)?;
        out.write_all(&(proof_bytes.len() as u32).to_le_bytes())?;
        out.write_all(proof_bytes)?;
    }
    Ok(())
}

#[inline]
fn write_u32_le(mut out: impl ark_std::io::Write, v: u32) -> Result<(), ark_std::io::Error> {
    out.write_all(&v.to_le_bytes())
}

#[inline]
fn write_u8(mut out: impl ark_std::io::Write, v: u8) -> Result<(), ark_std::io::Error> {
    out.write_all(&[v])
}

#[inline]
fn write_u64_le(mut out: impl ark_std::io::Write, v: u64) -> Result<(), ark_std::io::Error> {
    out.write_all(&v.to_le_bytes())
}

fn write_sumcheck_instance_proof<F: JoltField, T: Transcript>(
    mut out: impl ark_std::io::Write,
    proof: &crate::subprotocols::sumcheck::SumcheckInstanceProof<F, T>,
) -> Result<(), ark_std::io::Error> {
    // Number of rounds
    write_u32_le(&mut out, proof.compressed_polys.len() as u32)?;
    for poly in &proof.compressed_polys {
        write_u32_le(&mut out, poly.coeffs_except_linear_term.len() as u32)?;
        for c in &poly.coeffs_except_linear_term {
            c.serialize_compressed(&mut out)
                .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
        }
    }
    Ok(())
}

/// Serialize a `JoltProof` into the proof-record encoding used inside the proof bundle.
///
/// This encoding is *not* the arkworks canonical encoding for `JoltProof`. It is designed so the
/// verifier can stream-verify without needing to materialize a full `JoltProof` object.
pub fn write_proof_record<F, PCS, FS>(
    mut out: impl ark_std::io::Write,
    proof: &JoltProof<F, PCS, FS>,
) -> Result<(), ark_std::io::Error>
where
    F: JoltField,
    PCS: RecursionExt<F, Hint = crate::poly::commitment::dory::recursion::JoltHintMap>,
    FS: Transcript,
{
    // Record version
    write_u32_le(&mut out, PROOF_RECORD_VERSION)?;

    // Configs needed early by the verifier.
    write_u64_le(&mut out, proof.trace_length as u64)?;
    write_u64_le(&mut out, proof.ram_K as u64)?;
    write_u64_le(&mut out, proof.bytecode_K as u64)?;

    proof
        .program_mode
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    proof
        .rw_config
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    proof
        .one_hot_config
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    proof
        .dory_layout
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;

    // Recursion mode tag (strict extension): 0 = base proof, 1 = includes recursion payload.
    write_u8(&mut out, u8::from(proof.recursion.is_some()))?;

    // Opening claims (key -> claim), length-delimited as u32.
    write_u32_le(&mut out, proof.opening_claims.0.len() as u32)?;
    for (key, (_pt, claim)) in proof.opening_claims.0.iter() {
        key.serialize_compressed(&mut out)
            .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
        claim
            .serialize_compressed(&mut out)
            .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    }

    // Commitments (length-delimited).
    write_u32_le(&mut out, proof.commitments.len() as u32)?;
    for c in &proof.commitments {
        c.serialize_compressed(&mut out)
            .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    }

    // Untrusted advice commitment (optional).
    proof
        .untrusted_advice_commitment
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;

    // Stage 1-7: uni-skip + sumcheck proofs.
    proof
        .stage1_uni_skip_first_round_proof
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    write_sumcheck_instance_proof(&mut out, &proof.stage1_sumcheck_proof)?;

    proof
        .stage2_uni_skip_first_round_proof
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    write_sumcheck_instance_proof(&mut out, &proof.stage2_sumcheck_proof)?;

    write_sumcheck_instance_proof(&mut out, &proof.stage3_sumcheck_proof)?;
    write_sumcheck_instance_proof(&mut out, &proof.stage4_sumcheck_proof)?;
    write_sumcheck_instance_proof(&mut out, &proof.stage5_sumcheck_proof)?;
    write_sumcheck_instance_proof(&mut out, &proof.stage6a_sumcheck_proof)?;
    write_sumcheck_instance_proof(&mut out, &proof.stage6b_sumcheck_proof)?;
    write_sumcheck_instance_proof(&mut out, &proof.stage7_sumcheck_proof)?;

    // Stage 8/9: PCS opening proof + hints.
    proof
        .stage8_opening_proof
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;

    // Optional Stage 9–13 recursion payload (record format; eliminates point decompression).
    if let Some(payload) = proof.recursion.as_ref() {
        // Stage 8 combine hint (optional).
        payload
            .stage8_combine_hint
            .serialize_compressed(&mut out)
            .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;

        // RecursionProofBundle starts here: version + hint map + metadata + recursion proof.
        recursion_proof_bundle::write_recursion_proof_bundle(
            &mut out,
            &payload.stage9_pcs_hint,
            &payload.stage10_recursion_metadata,
            &payload.recursion_proof,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_bundle_roundtrip_layout() {
        let preprocessing = b"preprocessing-bytes";
        let entries: Vec<(Vec<u8>, Vec<u8>)> = vec![
            (b"device1".to_vec(), b"proof1".to_vec()),
            (b"device2".to_vec(), b"proof2".to_vec()),
        ];

        let mut buf = Vec::new();
        write_proof_bundle(&mut buf, preprocessing, &entries).unwrap();

        let mut r = ProofBundleReader::new(&buf).unwrap();
        assert_eq!(r.preprocessing_bytes(), preprocessing);
        assert_eq!(r.remaining_entries(), 2);

        let e1 = r.next_entry().unwrap().unwrap();
        assert_eq!(e1.device_bytes, b"device1");
        assert_eq!(e1.proof_bytes, b"proof1");

        let e2 = r.next_entry().unwrap().unwrap();
        assert_eq!(e2.device_bytes, b"device2");
        assert_eq!(e2.proof_bytes, b"proof2");

        assert!(r.next_entry().unwrap().is_none());
    }
}
