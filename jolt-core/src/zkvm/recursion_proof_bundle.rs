//! Record-format encoding for the recursion payload (Stage 9–13).
//!
//! Goal: avoid expensive arkworks curve-point decompression by storing:
//! - Dory HintMap results as uncompressed affine coordinates (unchecked reconstruction)
//! - Hyrax dense commitment row points as uncompressed affine coordinates (unchecked reconstruction)
//! while keeping everything self-delimiting for streaming decode.

use crate::field::JoltField;
use crate::poly::opening_proof::{OpeningId, OpeningPoint, BIG_ENDIAN};
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::zkvm::proof_serialization::RecursionConstraintMetadata;
use crate::zkvm::recursion::jagged::assist::JaggedAssistProof;
use crate::zkvm::recursion::prover::RecursionProof;
use crate::zkvm::streaming_decode::{DecodeError, SliceReader};

use ark_bn254::{Fq, Fq12};
use ark_ec::{AffineRepr, CurveGroup};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use ark_std::collections::BTreeMap;

use dory::backends::arkworks::{ArkG1, ArkG2, ArkGT, BN254};
use dory::recursion::{HintMap, HintResult, OpId as DoryOpId, OpType as DoryOpType};

use crate::poly::commitment::dory::recursion::JoltHintMap;

type HyraxPCS = crate::poly::commitment::hyrax::Hyrax<1, ark_grumpkin::Projective>;

/// RecursionProofBundle record format version.
pub const RECURSION_PROOF_BUNDLE_VERSION: u32 = 1;

const HINT_TAG_G1: u8 = 0;
const HINT_TAG_G2: u8 = 1;
const HINT_TAG_GT: u8 = 2;

#[inline]
fn write_u8(mut out: impl ark_std::io::Write, v: u8) -> Result<(), ark_std::io::Error> {
    out.write_all(&[v])
}

#[inline]
fn write_u16_le(mut out: impl ark_std::io::Write, v: u16) -> Result<(), ark_std::io::Error> {
    out.write_all(&v.to_le_bytes())
}

#[inline]
fn write_u32_le(mut out: impl ark_std::io::Write, v: u32) -> Result<(), ark_std::io::Error> {
    out.write_all(&v.to_le_bytes())
}

fn write_sumcheck_instance_proof_record<T: crate::transcripts::Transcript>(
    mut out: impl ark_std::io::Write,
    proof: &SumcheckInstanceProof<Fq, T>,
) -> Result<(), ark_std::io::Error> {
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

fn read_sumcheck_instance_proof_record<'a, T: crate::transcripts::Transcript>(
    r: &mut SliceReader<'a>,
) -> Result<SumcheckInstanceProof<Fq, T>, DecodeError> {
    use crate::poly::unipoly::CompressedUniPoly;

    let num_rounds = r.read_u32_le()? as usize;
    let mut polys = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        let coeffs_len = r.read_u32_le()? as usize;
        let mut coeffs = Vec::with_capacity(coeffs_len);
        for _ in 0..coeffs_len {
            let c: Fq = r.read_canonical(Compress::Yes, Validate::No)?;
            coeffs.push(c);
        }
        polys.push(CompressedUniPoly {
            coeffs_except_linear_term: coeffs,
        });
    }
    Ok(SumcheckInstanceProof::new(polys))
}

fn write_bn254_g1_uncompressed(
    mut out: impl ark_std::io::Write,
    p: &ArkG1,
) -> Result<(), ark_std::io::Error> {
    let affine = p.0.into_affine();
    if affine.is_zero() {
        write_u8(&mut out, 1)?;
        return Ok(());
    }
    write_u8(&mut out, 0)?;
    affine
        .x
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    affine
        .y
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    Ok(())
}

fn write_bn254_g2_uncompressed(
    mut out: impl ark_std::io::Write,
    p: &ArkG2,
) -> Result<(), ark_std::io::Error> {
    let affine = p.0.into_affine();
    if affine.is_zero() {
        write_u8(&mut out, 1)?;
        return Ok(());
    }
    write_u8(&mut out, 0)?;
    affine
        .x
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    affine
        .y
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    Ok(())
}

fn write_bn254_gt(
    mut out: impl ark_std::io::Write,
    gt: &ArkGT,
) -> Result<(), ark_std::io::Error> {
    // Field element; canonical encoding is already "uncompressed" in practice.
    gt.0.serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))
}

fn op_type_from_u8(b: u8) -> Result<DoryOpType, DecodeError> {
    Ok(match b {
        0 => DoryOpType::G1Add,
        1 => DoryOpType::G1ScalarMul,
        2 => DoryOpType::MsmG1,
        3 => DoryOpType::G2Add,
        4 => DoryOpType::G2ScalarMul,
        5 => DoryOpType::MsmG2,
        6 => DoryOpType::GtMul,
        7 => DoryOpType::GtExp,
        8 => DoryOpType::Pairing,
        9 => DoryOpType::MultiPairing,
        _ => return Err(DecodeError::Invalid("invalid dory op_type")),
    })
}

/// Write the Stage-9 PCS hint (Dory HintMap) in record form (no blob, no point compression).
pub fn write_hint_map_record(
    mut out: impl ark_std::io::Write,
    hint: &JoltHintMap,
) -> Result<(), ark_std::io::Error> {
    let num_rounds_u32: u32 = hint.0.num_rounds.try_into().map_err(|_| {
        ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, "num_rounds overflow")
    })?;

    // Sort entries for deterministic encoding.
    let mut entries: Vec<(DoryOpId, HintResult<BN254>)> =
        hint.0.iter().map(|(id, r)| (*id, r.clone())).collect();
    entries.sort_by_key(|(id, _)| *id);

    write_u32_le(&mut out, num_rounds_u32)?;
    write_u32_le(&mut out, entries.len() as u32)?;

    for (id, result) in entries {
        write_u16_le(&mut out, id.round)?;
        write_u8(&mut out, id.op_type as u8)?;
        write_u16_le(&mut out, id.index)?;

        match result {
            HintResult::G1(g1) => {
                write_u8(&mut out, HINT_TAG_G1)?;
                write_bn254_g1_uncompressed(&mut out, &g1)?;
            }
            HintResult::G2(g2) => {
                write_u8(&mut out, HINT_TAG_G2)?;
                write_bn254_g2_uncompressed(&mut out, &g2)?;
            }
            HintResult::GT(gt) => {
                write_u8(&mut out, HINT_TAG_GT)?;
                write_bn254_gt(&mut out, &gt)?;
            }
        }
    }
    Ok(())
}

/// Read the Stage-9 PCS hint (Dory HintMap) from record form.
pub fn read_hint_map_record<'a>(r: &mut SliceReader<'a>) -> Result<JoltHintMap, DecodeError> {
    let num_rounds = r.read_u32_le()? as usize;
    let n = r.read_u32_le()? as usize;

    let mut hm: HintMap<BN254> = HintMap::new(num_rounds);
    for _ in 0..n {
        let round = {
            let b = r.take(2)?;
            u16::from_le_bytes([b[0], b[1]])
        };
        let op_type = op_type_from_u8(r.read_u8()?)?;
        let index = {
            let b = r.take(2)?;
            u16::from_le_bytes([b[0], b[1]])
        };
        let id = DoryOpId::new(round, op_type, index);

        let tag = r.read_u8()?;
        let result = match tag {
            HINT_TAG_G1 => {
                let a = r.read_bn254_g1_affine_uncompressed_unchecked()?;
                let p = a.into_group();
                HintResult::G1(ArkG1(p))
            }
            HINT_TAG_G2 => {
                let a = r.read_bn254_g2_affine_uncompressed_unchecked()?;
                let p = a.into_group();
                HintResult::G2(ArkG2(p))
            }
            HINT_TAG_GT => {
                let gt: Fq12 = r.read_canonical(Compress::Yes, Validate::No)?;
                HintResult::GT(ArkGT(gt))
            }
            _ => return Err(DecodeError::Invalid("invalid hint tag")),
        };
        hm.insert(id, result);
    }
    Ok(JoltHintMap(hm))
}

fn write_grumpkin_point_uncompressed(
    mut out: impl ark_std::io::Write,
    p: &ark_grumpkin::Projective,
) -> Result<(), ark_std::io::Error> {
    let affine = p.into_affine();
    if affine.is_zero() {
        write_u8(&mut out, 1)?;
        return Ok(());
    }
    write_u8(&mut out, 0)?;
    affine
        .x
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    affine
        .y
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    Ok(())
}

fn read_grumpkin_point_uncompressed<'a>(
    r: &mut SliceReader<'a>,
) -> Result<ark_grumpkin::Projective, DecodeError> {
    let a = r.read_grumpkin_affine_uncompressed_unchecked()?;
    Ok(a.into_group())
}

fn write_openings_record(
    mut out: impl ark_std::io::Write,
    openings: &BTreeMap<OpeningId, (OpeningPoint<BIG_ENDIAN, Fq>, Fq)>,
) -> Result<(), ark_std::io::Error> {
    write_u32_le(&mut out, openings.len() as u32)?;
    for (key, (pt, claim)) in openings.iter() {
        key.serialize_compressed(&mut out)
            .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;

        write_u32_le(&mut out, pt.r.len() as u32)?;
        for c in pt.r.iter() {
            c.serialize_compressed(&mut out)
                .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
        }

        claim.serialize_compressed(&mut out)
            .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    }
    Ok(())
}

fn read_openings_record<'a>(
    r: &mut SliceReader<'a>,
) -> Result<BTreeMap<OpeningId, (OpeningPoint<BIG_ENDIAN, Fq>, Fq)>, DecodeError> {
    let n = r.read_u32_le()? as usize;
    let mut openings: BTreeMap<OpeningId, (OpeningPoint<BIG_ENDIAN, Fq>, Fq)> = BTreeMap::new();
    for _ in 0..n {
        let key: OpeningId = r.read_canonical(Compress::Yes, Validate::No)?;
        let pt_len = r.read_u32_le()? as usize;
        let mut rr = Vec::with_capacity(pt_len);
        for _ in 0..pt_len {
            let c: <Fq as JoltField>::Challenge = r.read_canonical(Compress::Yes, Validate::No)?;
            rr.push(c);
        }
        let pt = OpeningPoint::<BIG_ENDIAN, Fq>::new(rr);
        let claim: Fq = r.read_canonical(Compress::Yes, Validate::No)?;
        openings.insert(key, (pt, claim));
    }
    Ok(openings)
}

pub(crate) fn write_recursion_constraint_metadata_record(
    mut out: impl ark_std::io::Write,
    metadata: &RecursionConstraintMetadata,
) -> Result<(), ark_std::io::Error> {
    let n = metadata.serialized_size(Compress::No);
    write_u32_le(&mut out, n as u32)?;
    metadata
        .serialize_uncompressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    Ok(())
}

pub(crate) fn read_recursion_constraint_metadata_record<'a>(
    r: &mut SliceReader<'a>,
) -> Result<RecursionConstraintMetadata, DecodeError> {
    let n = r.read_u32_le()? as usize;
    let bytes = r.take(n)?;
    let mut b = bytes;
    RecursionConstraintMetadata::deserialize_with_mode(&mut b, Compress::No, Validate::No)
        .map_err(DecodeError::Ark)
}

pub(crate) fn write_recursion_proof_record<T: crate::transcripts::Transcript>(
    mut out: impl ark_std::io::Write,
    proof: &RecursionProof<Fq, T, HyraxPCS>,
) -> Result<(), ark_std::io::Error> {
    // FS sync
    proof
        .gamma
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    proof
        .delta
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;

    // Dense commitment (HyraxCommitment row points, uncompressed affine coords)
    write_u32_le(&mut out, proof.dense_commitment.row_commitments.len() as u32)?;
    for p in &proof.dense_commitment.row_commitments {
        write_grumpkin_point_uncompressed(&mut out, p)?;
    }

    // Opening claims
    write_openings_record(&mut out, &proof.opening_claims)?;

    // Stages 1–5
    write_sumcheck_instance_proof_record(&mut out, &proof.stage1_proof)?;
    write_sumcheck_instance_proof_record(&mut out, &proof.stage2_proof)?;
    proof
        .stage3_m_eval
        .serialize_compressed(&mut out)
        .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    write_sumcheck_instance_proof_record(&mut out, &proof.stage4_proof)?;

    // Jagged assist
    write_u32_le(&mut out, proof.stage5_proof.claimed_evaluations.len() as u32)?;
    for v in &proof.stage5_proof.claimed_evaluations {
        v.serialize_compressed(&mut out)
            .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    }
    write_sumcheck_instance_proof_record(&mut out, &proof.stage5_proof.sumcheck_proof)?;

    // Hyrax opening proof is just the vector-matrix product (scalars)
    write_u32_le(&mut out, proof.opening_proof.vector_matrix_product.len() as u32)?;
    for s in &proof.opening_proof.vector_matrix_product {
        s.serialize_compressed(&mut out)
            .map_err(|e| ark_std::io::Error::new(ark_std::io::ErrorKind::InvalidData, e))?;
    }

    Ok(())
}

pub(crate) fn read_recursion_proof_record<'a, T: crate::transcripts::Transcript>(
    r: &mut SliceReader<'a>,
) -> Result<RecursionProof<Fq, T, HyraxPCS>, DecodeError> {
    // FS sync
    let gamma: Fq = r.read_canonical(Compress::Yes, Validate::No)?;
    let delta: Fq = r.read_canonical(Compress::Yes, Validate::No)?;

    // Dense commitment
    let n_rows = r.read_u32_le()? as usize;
    let mut rows = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        rows.push(read_grumpkin_point_uncompressed(r)?);
    }
    let dense_commitment = crate::poly::commitment::hyrax::HyraxCommitment::<
        1,
        ark_grumpkin::Projective,
    > { row_commitments: rows };

    // Opening claims
    let opening_claims = read_openings_record(r)?;

    // Stages 1–5
    let stage1_proof = read_sumcheck_instance_proof_record::<T>(r)?;
    let stage2_proof = read_sumcheck_instance_proof_record::<T>(r)?;
    let stage3_m_eval: Fq = r.read_canonical(Compress::Yes, Validate::No)?;
    let stage4_proof = read_sumcheck_instance_proof_record::<T>(r)?;

    // Jagged assist
    let claimed_len = r.read_u32_le()? as usize;
    let mut claimed_evaluations = Vec::with_capacity(claimed_len);
    for _ in 0..claimed_len {
        claimed_evaluations.push(r.read_canonical(Compress::Yes, Validate::No)?);
    }
    let stage5_sumcheck = read_sumcheck_instance_proof_record::<T>(r)?;
    let stage5_proof: JaggedAssistProof<Fq, T> = JaggedAssistProof {
        claimed_evaluations,
        sumcheck_proof: stage5_sumcheck,
    };

    // Hyrax opening proof (vector-matrix product scalars)
    let vlen = r.read_u32_le()? as usize;
    let mut v = Vec::with_capacity(vlen);
    for _ in 0..vlen {
        v.push(r.read_canonical(Compress::Yes, Validate::No)?);
    }
    let opening_proof = crate::poly::commitment::hyrax::HyraxOpeningProof::<
        1,
        ark_grumpkin::Projective,
    > {
        vector_matrix_product: v,
    };

    Ok(RecursionProof {
        stage1_proof,
        stage2_proof,
        stage3_m_eval,
        stage4_proof,
        stage5_proof,
        opening_proof,
        gamma,
        delta,
        opening_claims,
        dense_commitment,
    })
}

/// Write the full RecursionProofBundle section (version + hint map + metadata + recursion proof).
pub fn write_recursion_proof_bundle<T: crate::transcripts::Transcript>(
    mut out: impl ark_std::io::Write,
    hint: &JoltHintMap,
    metadata: &RecursionConstraintMetadata,
    recursion_proof: &RecursionProof<Fq, T, HyraxPCS>,
) -> Result<(), ark_std::io::Error> {
    write_u32_le(&mut out, RECURSION_PROOF_BUNDLE_VERSION)?;
    write_hint_map_record(&mut out, hint)?;
    write_recursion_constraint_metadata_record(&mut out, metadata)?;
    write_recursion_proof_record(&mut out, recursion_proof)?;
    Ok(())
}

/// Read the full RecursionProofBundle section (version + hint map + metadata + recursion proof).
pub fn read_recursion_proof_bundle<'a, T: crate::transcripts::Transcript>(
    r: &mut SliceReader<'a>,
) -> Result<(JoltHintMap, RecursionConstraintMetadata, RecursionProof<Fq, T, HyraxPCS>), DecodeError>
{
    let version = r.read_u32_le()?;
    if version != RECURSION_PROOF_BUNDLE_VERSION {
        return Err(DecodeError::Invalid("unsupported RecursionProofBundle version"));
    }
    let hint = read_hint_map_record(r)?;
    let metadata = read_recursion_constraint_metadata_record(r)?;
    let proof = read_recursion_proof_record::<T>(r)?;
    Ok((hint, metadata, proof))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::G1Affine;
    use ark_ec::AffineRepr;
    use ark_ff::One;

    #[test]
    fn hint_map_record_roundtrip_small() {
        // Build a tiny hint map with 2 entries.
        let mut hm: HintMap<BN254> = HintMap::new(1);
        let id1 = DoryOpId::new(0, DoryOpType::G1Add, 0);
        let id2 = DoryOpId::new(0, DoryOpType::GtMul, 0);

        let p = G1Affine::generator().into_group();
        hm.insert(id1, HintResult::G1(ArkG1(p)));
        hm.insert(id2, HintResult::GT(ArkGT(Fq12::one())));

        let j = JoltHintMap(hm);
        let mut buf = Vec::new();
        write_hint_map_record(&mut buf, &j).unwrap();

        let mut r = SliceReader::new(&buf);
        let j2 = read_hint_map_record(&mut r).unwrap();
        assert_eq!(r.remaining(), 0);

        let mut a: Vec<_> = j.0.iter().map(|(k, v)| (*k, v.is_gt(), v.is_g1())).collect();
        let mut b: Vec<_> = j2.0.iter().map(|(k, v)| (*k, v.is_gt(), v.is_g1())).collect();
        a.sort_by_key(|(k, _, _)| *k);
        b.sort_by_key(|(k, _, _)| *k);
        assert_eq!(a, b);
    }
}

