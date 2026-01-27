use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use num::FromPrimitive;
use strum::EnumCount;

use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        commitment::dory::DoryLayout,
        opening_proof::{OpeningId, OpeningPoint, Openings, SumcheckId},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    zkvm::{
        config::{OneHotConfig, ProgramMode, ReadWriteConfig},
        instruction::{CircuitFlags, InstructionFlags},
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use crate::{
    poly::opening_proof::PolynomialId, subprotocols::univariate_skip::UniSkipFirstRoundProof,
};

#[cfg(feature = "recursion")]
use ark_bn254::{Fq12, G1Affine, G2Affine};

/// Boundary outputs for the final external pairing check in recursion mode.
///
/// The verifier must **not trust** prover-supplied values here; it should re-derive the expected
/// boundary from public inputs (proof + setup) and compare.
#[cfg(feature = "recursion")]
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct PairingBoundary {
    pub p1_g1: G1Affine,
    pub p1_g2: G2Affine,
    pub p2_g1: G1Affine,
    pub p2_g2: G2Affine,
    pub p3_g1: G1Affine,
    pub p3_g2: G2Affine,
    pub rhs: Fq12,
}

#[cfg(feature = "recursion")]
impl GuestSerialize for PairingBoundary {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.p1_g1.guest_serialize(w)?;
        self.p1_g2.guest_serialize(w)?;
        self.p2_g1.guest_serialize(w)?;
        self.p2_g2.guest_serialize(w)?;
        self.p3_g1.guest_serialize(w)?;
        self.p3_g2.guest_serialize(w)?;
        self.rhs.guest_serialize(w)?;
        Ok(())
    }
}

#[cfg(feature = "recursion")]
impl GuestDeserialize for PairingBoundary {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            p1_g1: ark_bn254::g1::G1Affine::guest_deserialize(r)?,
            p1_g2: ark_bn254::g2::G2Affine::guest_deserialize(r)?,
            p2_g1: ark_bn254::g1::G1Affine::guest_deserialize(r)?,
            p2_g2: ark_bn254::g2::G2Affine::guest_deserialize(r)?,
            p3_g1: ark_bn254::g1::G1Affine::guest_deserialize(r)?,
            p3_g2: ark_bn254::g2::G2Affine::guest_deserialize(r)?,
            rhs: Fq12::guest_deserialize(r)?,
        })
    }
}

/// Hints for recursion instance-plan derivation when an op's base/point is not an `AstOp::Input`.
///
/// These are used to avoid requiring the verifier to evaluate the full Dory verification DAG just
/// to recover bases/points for public inputs in the recursion verifier input.
///
/// **Security note**: without wiring/boundary constraints, these hints are not bound to the Dory
/// verification computation. They are intended for performance/profiling until wiring is added.
#[cfg(feature = "recursion")]
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NonInputBaseHints {
    /// One entry per Dory-traced `GTExp` op, in OpId-sorted order.
    /// `None` means the base was an `AstOp::Input` and can be resolved by the verifier.
    pub gt_exp_base_hints: Vec<Option<Fq12>>,
    /// One entry per Dory-traced `G1ScalarMul` op, in OpId-sorted order.
    pub g1_scalar_mul_base_hints: Vec<Option<G1Affine>>,
    /// One entry per Dory-traced `G2ScalarMul` op, in OpId-sorted order.
    pub g2_scalar_mul_base_hints: Vec<Option<G2Affine>>,
}

#[cfg(feature = "recursion")]
impl GuestSerialize for NonInputBaseHints {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.gt_exp_base_hints.guest_serialize(w)?;
        self.g1_scalar_mul_base_hints.guest_serialize(w)?;
        self.g2_scalar_mul_base_hints.guest_serialize(w)?;
        Ok(())
    }
}

#[cfg(feature = "recursion")]
impl GuestDeserialize for NonInputBaseHints {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            gt_exp_base_hints: Vec::<Option<Fq12>>::guest_deserialize(r)?,
            g1_scalar_mul_base_hints: Vec::<Option<G1Affine>>::guest_deserialize(r)?,
            g2_scalar_mul_base_hints: Vec::<Option<G2Affine>>::guest_deserialize(r)?,
        })
    }
}

/// Jolt proof structure organized by verification stages.
#[derive(Clone)]
pub struct JoltProof<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> {
    pub opening_claims: Claims<F>,
    pub commitments: Vec<PCS::Commitment>,

    pub stage1_uni_skip_first_round_proof: UniSkipFirstRoundProof<F, FS>,
    pub stage1_sumcheck_proof: SumcheckInstanceProof<F, FS>,

    pub stage2_uni_skip_first_round_proof: UniSkipFirstRoundProof<F, FS>,
    pub stage2_sumcheck_proof: SumcheckInstanceProof<F, FS>,

    pub stage3_sumcheck_proof: SumcheckInstanceProof<F, FS>,

    pub stage4_sumcheck_proof: SumcheckInstanceProof<F, FS>,

    pub stage5_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    pub stage6a_sumcheck_proof: SumcheckInstanceProof<F, FS>,
    pub stage6b_sumcheck_proof: SumcheckInstanceProof<F, FS>,

    pub stage7_sumcheck_proof: SumcheckInstanceProof<F, FS>,

    /// Dory polynomial commitment opening proof (joint batch opening)
    pub joint_opening_proof: PCS::Proof,

    pub untrusted_advice_commitment: Option<PCS::Commitment>,

    pub trace_length: usize,
    pub ram_K: usize,
    pub bytecode_K: usize,
    pub program_mode: ProgramMode,
    pub rw_config: ReadWriteConfig,
    pub one_hot_config: OneHotConfig,
    pub dory_layout: DoryLayout,
}

impl CanonicalSerialize for DoryLayout {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        u8::from(*self).serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        u8::from(*self).serialized_size(compress)
    }
}

impl Valid for DoryLayout {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for DoryLayout {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = u8::deserialize_with_mode(reader, compress, validate)?;
        if value > 1 {
            return Err(SerializationError::InvalidData);
        }
        Ok(DoryLayout::from(value))
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> CanonicalSerialize
    for JoltProof<F, PCS, FS>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.opening_claims
            .serialize_with_mode(&mut writer, compress)?;
        self.commitments
            .serialize_with_mode(&mut writer, compress)?;
        self.stage1_uni_skip_first_round_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage1_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2_uni_skip_first_round_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage3_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage4_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage5_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage6a_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage6b_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.stage7_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.joint_opening_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.untrusted_advice_commitment
            .serialize_with_mode(&mut writer, compress)?;
        self.trace_length
            .serialize_with_mode(&mut writer, compress)?;
        self.ram_K.serialize_with_mode(&mut writer, compress)?;
        self.bytecode_K.serialize_with_mode(&mut writer, compress)?;
        self.program_mode
            .serialize_with_mode(&mut writer, compress)?;
        self.rw_config.serialize_with_mode(&mut writer, compress)?;
        self.one_hot_config
            .serialize_with_mode(&mut writer, compress)?;
        self.dory_layout
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.opening_claims.serialized_size(compress)
            + self.commitments.serialized_size(compress)
            + self
                .stage1_uni_skip_first_round_proof
                .serialized_size(compress)
            + self.stage1_sumcheck_proof.serialized_size(compress)
            + self
                .stage2_uni_skip_first_round_proof
                .serialized_size(compress)
            + self.stage2_sumcheck_proof.serialized_size(compress)
            + self.stage3_sumcheck_proof.serialized_size(compress)
            + self.stage4_sumcheck_proof.serialized_size(compress)
            + self.stage5_sumcheck_proof.serialized_size(compress)
            + self.stage6a_sumcheck_proof.serialized_size(compress)
            + self.stage6b_sumcheck_proof.serialized_size(compress)
            + self.stage7_sumcheck_proof.serialized_size(compress)
            + self.joint_opening_proof.serialized_size(compress)
            + self.untrusted_advice_commitment.serialized_size(compress)
            + self.trace_length.serialized_size(compress)
            + self.ram_K.serialized_size(compress)
            + self.bytecode_K.serialized_size(compress)
            + self.program_mode.serialized_size(compress)
            + self.rw_config.serialized_size(compress)
            + self.one_hot_config.serialized_size(compress)
            + self.dory_layout.serialized_size(compress)
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> Valid
    for JoltProof<F, PCS, FS>
{
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> CanonicalDeserialize
    for JoltProof<F, PCS, FS>
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            opening_claims: Claims::deserialize_with_mode(&mut reader, compress, validate)?,
            commitments: Vec::deserialize_with_mode(&mut reader, compress, validate)?,
            stage1_uni_skip_first_round_proof: UniSkipFirstRoundProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage1_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage2_uni_skip_first_round_proof: UniSkipFirstRoundProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage2_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage3_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage4_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage5_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage6a_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage6b_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            stage7_sumcheck_proof: SumcheckInstanceProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            joint_opening_proof: PCS::Proof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            untrusted_advice_commitment: Option::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            trace_length: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            ram_K: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            bytecode_K: usize::deserialize_with_mode(&mut reader, compress, validate)?,
            program_mode: ProgramMode::deserialize_with_mode(&mut reader, compress, validate)?,
            rw_config: ReadWriteConfig::deserialize_with_mode(&mut reader, compress, validate)?,
            one_hot_config: OneHotConfig::deserialize_with_mode(&mut reader, compress, validate)?,
            dory_layout: DoryLayout::deserialize_with_mode(&mut reader, compress, validate)?,
        })
    }
}

#[derive(Clone)]
pub struct Claims<F: JoltField>(pub Openings<F>);

impl<F: JoltField> CanonicalSerialize for Claims<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.len().serialize_with_mode(&mut writer, compress)?;
        for (key, (_opening_point, claim)) in self.0.iter() {
            key.serialize_with_mode(&mut writer, compress)?;
            claim.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut size = self.0.len().serialized_size(compress);
        for (key, (_opening_point, claim)) in self.0.iter() {
            size += key.serialized_size(compress);
            size += claim.serialized_size(compress);
        }
        size
    }
}

impl<F: JoltField> Valid for Claims<F> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: JoltField> CanonicalDeserialize for Claims<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let size = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let mut claims = BTreeMap::new();
        for _ in 0..size {
            let key = OpeningId::deserialize_with_mode(&mut reader, compress, validate)?;
            let claim = F::deserialize_with_mode(&mut reader, compress, validate)?;
            claims.insert(key, (OpeningPoint::default(), claim));
        }

        Ok(Claims(claims))
    }
}

// Guest-optimized encoding for recursion/guest verification inputs.
impl<F> GuestSerialize for Claims<F>
where
    F: JoltField + GuestSerialize,
{
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        // Match canonical shape: len + (OpeningId, claim) pairs (opening point omitted).
        (self.0.len() as u64).guest_serialize(w)?;
        for (key, (_opening_point, claim)) in self.0.iter() {
            // `OpeningId` is small; use its canonical encoding.
            key.serialize_compressed(&mut *w)
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "OpeningId"))?;
            claim.guest_serialize(w)?;
        }
        Ok(())
    }
}

impl<F> GuestDeserialize for Claims<F>
where
    F: JoltField + GuestDeserialize,
{
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        let size_u64 = u64::guest_deserialize(r)?;
        let size = usize::try_from(size_u64).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "Claims length overflow")
        })?;
        let mut claims = BTreeMap::new();
        for _ in 0..size {
            let key = OpeningId::deserialize_compressed(&mut *r)
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "OpeningId"))?;
            let claim = F::guest_deserialize(r)?;
            claims.insert(key, (OpeningPoint::default(), claim));
        }
        Ok(Claims(claims))
    }
}

impl<F, PCS, FS> GuestSerialize for JoltProof<F, PCS, FS>
where
    F: JoltField + GuestSerialize,
    PCS: CommitmentScheme<Field = F>,
    PCS::Commitment: GuestSerialize,
    PCS::Proof: GuestSerialize,
    FS: Transcript,
{
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.opening_claims.guest_serialize(w)?;
        self.commitments.guest_serialize(w)?;
        self.stage1_uni_skip_first_round_proof.guest_serialize(w)?;
        self.stage1_sumcheck_proof.guest_serialize(w)?;
        self.stage2_uni_skip_first_round_proof.guest_serialize(w)?;
        self.stage2_sumcheck_proof.guest_serialize(w)?;
        self.stage3_sumcheck_proof.guest_serialize(w)?;
        self.stage4_sumcheck_proof.guest_serialize(w)?;
        self.stage5_sumcheck_proof.guest_serialize(w)?;
        self.stage6a_sumcheck_proof.guest_serialize(w)?;
        self.stage6b_sumcheck_proof.guest_serialize(w)?;
        self.stage7_sumcheck_proof.guest_serialize(w)?;
        self.joint_opening_proof.guest_serialize(w)?;
        self.untrusted_advice_commitment.guest_serialize(w)?;
        self.trace_length.guest_serialize(w)?;
        self.ram_K.guest_serialize(w)?;
        self.bytecode_K.guest_serialize(w)?;
        self.program_mode.guest_serialize(w)?;
        self.rw_config.guest_serialize(w)?;
        self.one_hot_config.guest_serialize(w)?;
        self.dory_layout.guest_serialize(w)?;
        Ok(())
    }
}

impl<F, PCS, FS> GuestDeserialize for JoltProof<F, PCS, FS>
where
    F: JoltField + GuestDeserialize,
    PCS: CommitmentScheme<Field = F>,
    PCS::Commitment: GuestDeserialize,
    PCS::Proof: GuestDeserialize,
    FS: Transcript,
{
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            opening_claims: Claims::<F>::guest_deserialize(r)?,
            commitments: Vec::<PCS::Commitment>::guest_deserialize(r)?,
            stage1_uni_skip_first_round_proof: UniSkipFirstRoundProof::<F, FS>::guest_deserialize(
                r,
            )?,
            stage1_sumcheck_proof: SumcheckInstanceProof::<F, FS>::guest_deserialize(r)?,
            stage2_uni_skip_first_round_proof: UniSkipFirstRoundProof::<F, FS>::guest_deserialize(
                r,
            )?,
            stage2_sumcheck_proof: SumcheckInstanceProof::<F, FS>::guest_deserialize(r)?,
            stage3_sumcheck_proof: SumcheckInstanceProof::<F, FS>::guest_deserialize(r)?,
            stage4_sumcheck_proof: SumcheckInstanceProof::<F, FS>::guest_deserialize(r)?,
            stage5_sumcheck_proof: SumcheckInstanceProof::<F, FS>::guest_deserialize(r)?,
            stage6a_sumcheck_proof: SumcheckInstanceProof::<F, FS>::guest_deserialize(r)?,
            stage6b_sumcheck_proof: SumcheckInstanceProof::<F, FS>::guest_deserialize(r)?,
            stage7_sumcheck_proof: SumcheckInstanceProof::<F, FS>::guest_deserialize(r)?,
            joint_opening_proof: PCS::Proof::guest_deserialize(r)?,
            untrusted_advice_commitment: Option::<PCS::Commitment>::guest_deserialize(r)?,
            trace_length: usize::guest_deserialize(r)?,
            ram_K: usize::guest_deserialize(r)?,
            bytecode_K: usize::guest_deserialize(r)?,
            program_mode: ProgramMode::guest_deserialize(r)?,
            rw_config: ReadWriteConfig::guest_deserialize(r)?,
            one_hot_config: OneHotConfig::guest_deserialize(r)?,
            dory_layout: DoryLayout::guest_deserialize(r)?,
        })
    }
}

// Compact encoding for OpeningId:
// Each variant uses a fused byte = BASE + sumcheck_id (1 byte total for advice, 2 bytes for committed/virtual)
// - [0, NUM_SUMCHECKS) = UntrustedAdvice(sumcheck_id)
// - [NUM_SUMCHECKS, 2*NUM_SUMCHECKS) = TrustedAdvice(sumcheck_id)
// - [2*NUM_SUMCHECKS, 3*NUM_SUMCHECKS) + poly_index = Committed(poly, sumcheck_id)
// - [3*NUM_SUMCHECKS, 4*NUM_SUMCHECKS) + poly_index = Virtual(poly, sumcheck_id)
const OPENING_ID_UNTRUSTED_ADVICE_BASE: u8 = 0;
const OPENING_ID_TRUSTED_ADVICE_BASE: u8 =
    OPENING_ID_UNTRUSTED_ADVICE_BASE + SumcheckId::COUNT as u8;
const OPENING_ID_COMMITTED_BASE: u8 = OPENING_ID_TRUSTED_ADVICE_BASE + SumcheckId::COUNT as u8;
const OPENING_ID_VIRTUAL_BASE: u8 = OPENING_ID_COMMITTED_BASE + SumcheckId::COUNT as u8;

impl CanonicalSerialize for OpeningId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            OpeningId::UntrustedAdvice(sumcheck_id) => {
                let fused = OPENING_ID_UNTRUSTED_ADVICE_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::TrustedAdvice(sumcheck_id) => {
                let fused = OPENING_ID_TRUSTED_ADVICE_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::Polynomial(PolynomialId::Committed(committed_polynomial), sumcheck_id) => {
                let fused = OPENING_ID_COMMITTED_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)?;
                committed_polynomial.serialize_with_mode(&mut writer, compress)
            }
            OpeningId::Polynomial(PolynomialId::Virtual(virtual_polynomial), sumcheck_id) => {
                let fused = OPENING_ID_VIRTUAL_BASE + (*sumcheck_id as u8);
                fused.serialize_with_mode(&mut writer, compress)?;
                virtual_polynomial.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            OpeningId::UntrustedAdvice(_) | OpeningId::TrustedAdvice(_) => 1,
            OpeningId::Polynomial(PolynomialId::Committed(committed_polynomial), _) => {
                // 1 byte fused (variant + sumcheck_id) + poly index
                1 + committed_polynomial.serialized_size(compress)
            }
            OpeningId::Polynomial(PolynomialId::Virtual(virtual_polynomial), _) => {
                // 1 byte fused (variant + sumcheck_id) + poly index
                1 + virtual_polynomial.serialized_size(compress)
            }
        }
    }
}

impl Valid for OpeningId {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for OpeningId {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let fused = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match fused {
            _ if fused < OPENING_ID_TRUSTED_ADVICE_BASE => {
                let sumcheck_id = fused - OPENING_ID_UNTRUSTED_ADVICE_BASE;
                Ok(OpeningId::UntrustedAdvice(
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ if fused < OPENING_ID_COMMITTED_BASE => {
                let sumcheck_id = fused - OPENING_ID_TRUSTED_ADVICE_BASE;
                Ok(OpeningId::TrustedAdvice(
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ if fused < OPENING_ID_VIRTUAL_BASE => {
                let sumcheck_id = fused - OPENING_ID_COMMITTED_BASE;
                let polynomial =
                    CommittedPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::Polynomial(
                    PolynomialId::Committed(polynomial),
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
            _ => {
                let sumcheck_id = fused - OPENING_ID_VIRTUAL_BASE;
                let polynomial =
                    VirtualPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(OpeningId::Polynomial(
                    PolynomialId::Virtual(polynomial),
                    SumcheckId::from_u8(sumcheck_id).ok_or(SerializationError::InvalidData)?,
                ))
            }
        }
    }
}

impl CanonicalSerialize for CommittedPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::RdInc => 0u8.serialize_with_mode(writer, compress),
            Self::RamInc => 1u8.serialize_with_mode(writer, compress),
            Self::InstructionRa(i) => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::BytecodeRa(i) => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::BytecodeChunk(i) => {
                7u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::RamRa(i) => {
                4u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*i).unwrap()).serialize_with_mode(writer, compress)
            }
            Self::TrustedAdvice => 5u8.serialize_with_mode(writer, compress),
            Self::UntrustedAdvice => 6u8.serialize_with_mode(writer, compress),
            Self::DoryDenseMatrix => 9u8.serialize_with_mode(writer, compress),
            Self::ProgramImageInit => 8u8.serialize_with_mode(writer, compress),
        }
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        match self {
            Self::RdInc
            | Self::RamInc
            | Self::TrustedAdvice
            | Self::UntrustedAdvice
            | Self::DoryDenseMatrix
            | Self::ProgramImageInit => 1,
            Self::InstructionRa(_)
            | Self::BytecodeRa(_)
            | Self::BytecodeChunk(_)
            | Self::RamRa(_) => 2,
        }
    }
}

impl Valid for CommittedPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for CommittedPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(
            match u8::deserialize_with_mode(&mut reader, compress, validate)? {
                0 => Self::RdInc,
                1 => Self::RamInc,
                2 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::InstructionRa(i as usize)
                }
                3 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::BytecodeRa(i as usize)
                }
                4 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::RamRa(i as usize)
                }
                5 => Self::TrustedAdvice,
                6 => Self::UntrustedAdvice,
                7 => {
                    let i = u8::deserialize_with_mode(reader, compress, validate)?;
                    Self::BytecodeChunk(i as usize)
                }
                8 => Self::ProgramImageInit,
                9 => Self::DoryDenseMatrix,
                _ => return Err(SerializationError::InvalidData),
            },
        )
    }
}

impl CanonicalSerialize for VirtualPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::PC => 0u8.serialize_with_mode(&mut writer, compress),
            Self::UnexpandedPC => 1u8.serialize_with_mode(&mut writer, compress),
            Self::NextPC => 2u8.serialize_with_mode(&mut writer, compress),
            Self::NextUnexpandedPC => 3u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsNoop => 4u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsVirtual => 5u8.serialize_with_mode(&mut writer, compress),
            Self::NextIsFirstInSequence => 6u8.serialize_with_mode(&mut writer, compress),
            Self::LeftLookupOperand => 7u8.serialize_with_mode(&mut writer, compress),
            Self::RightLookupOperand => 8u8.serialize_with_mode(&mut writer, compress),
            Self::LeftInstructionInput => 9u8.serialize_with_mode(&mut writer, compress),
            Self::RightInstructionInput => 10u8.serialize_with_mode(&mut writer, compress),
            Self::Product => 11u8.serialize_with_mode(&mut writer, compress),
            Self::ShouldJump => 12u8.serialize_with_mode(&mut writer, compress),
            Self::ShouldBranch => 13u8.serialize_with_mode(&mut writer, compress),
            Self::WritePCtoRD => 14u8.serialize_with_mode(&mut writer, compress),
            Self::WriteLookupOutputToRD => 15u8.serialize_with_mode(&mut writer, compress),
            Self::Rd => 16u8.serialize_with_mode(&mut writer, compress),
            Self::Imm => 17u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Value => 18u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Value => 19u8.serialize_with_mode(&mut writer, compress),
            Self::RdWriteValue => 20u8.serialize_with_mode(&mut writer, compress),
            Self::Rs1Ra => 21u8.serialize_with_mode(&mut writer, compress),
            Self::Rs2Ra => 22u8.serialize_with_mode(&mut writer, compress),
            Self::RdWa => 23u8.serialize_with_mode(&mut writer, compress),
            Self::LookupOutput => 24u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRaf => 25u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRafFlag => 26u8.serialize_with_mode(&mut writer, compress),
            Self::InstructionRa(i) => {
                27u8.serialize_with_mode(&mut writer, compress)?;
                (*i as u32).serialize_with_mode(&mut writer, compress)
            }
            Self::RegistersVal => 28u8.serialize_with_mode(&mut writer, compress),
            Self::RamAddress => 29u8.serialize_with_mode(&mut writer, compress),
            Self::RamRa => 30u8.serialize_with_mode(&mut writer, compress),
            Self::RamReadValue => 31u8.serialize_with_mode(&mut writer, compress),
            Self::RamWriteValue => 32u8.serialize_with_mode(&mut writer, compress),
            Self::RamVal => 33u8.serialize_with_mode(&mut writer, compress),
            Self::RamValInit => 34u8.serialize_with_mode(&mut writer, compress),
            Self::RamValFinal => 35u8.serialize_with_mode(&mut writer, compress),
            Self::RamHammingWeight => 36u8.serialize_with_mode(&mut writer, compress),
            Self::UnivariateSkip => 37u8.serialize_with_mode(&mut writer, compress),
            Self::OpFlags(flags) => {
                38u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::InstructionFlags(flags) => {
                39u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flags as usize).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::LookupTableFlag(flag) => {
                40u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*flag).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            // Program-image commitment variants
            Self::BytecodeValStage(stage) => {
                41u8.serialize_with_mode(&mut writer, compress)?;
                (u8::try_from(*stage).unwrap()).serialize_with_mode(&mut writer, compress)
            }
            Self::BytecodeReadRafAddrClaim => 42u8.serialize_with_mode(&mut writer, compress),
            Self::BooleanityAddrClaim => 43u8.serialize_with_mode(&mut writer, compress),
            Self::BytecodeClaimReductionIntermediate => {
                44u8.serialize_with_mode(&mut writer, compress)
            }
            Self::ProgramImageInitContributionRw => 45u8.serialize_with_mode(&mut writer, compress),
            Self::ProgramImageInitContributionRaf => {
                46u8.serialize_with_mode(&mut writer, compress)
            }
            Self::Recursion(poly) => {
                47u8.serialize_with_mode(&mut writer, compress)?;
                poly.serialize_with_mode(&mut writer, compress)
            }
            Self::DorySparseConstraintMatrix => 48u8.serialize_with_mode(&mut writer, compress),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            Self::PC
            | Self::UnexpandedPC
            | Self::NextPC
            | Self::NextUnexpandedPC
            | Self::NextIsNoop
            | Self::NextIsVirtual
            | Self::NextIsFirstInSequence
            | Self::LeftLookupOperand
            | Self::RightLookupOperand
            | Self::LeftInstructionInput
            | Self::RightInstructionInput
            | Self::Product
            | Self::ShouldJump
            | Self::ShouldBranch
            | Self::WritePCtoRD
            | Self::WriteLookupOutputToRD
            | Self::Rd
            | Self::Imm
            | Self::Rs1Value
            | Self::Rs2Value
            | Self::RdWriteValue
            | Self::Rs1Ra
            | Self::Rs2Ra
            | Self::RdWa
            | Self::LookupOutput
            | Self::InstructionRaf
            | Self::InstructionRafFlag
            | Self::RegistersVal
            | Self::RamAddress
            | Self::RamRa
            | Self::RamReadValue
            | Self::RamWriteValue
            | Self::RamVal
            | Self::RamValInit
            | Self::RamValFinal
            | Self::RamHammingWeight
            | Self::UnivariateSkip
            | Self::BytecodeReadRafAddrClaim
            | Self::BooleanityAddrClaim
            | Self::BytecodeClaimReductionIntermediate
            | Self::ProgramImageInitContributionRw
            | Self::ProgramImageInitContributionRaf => 1,
            Self::InstructionRa(_)
            | Self::OpFlags(_)
            | Self::InstructionFlags(_)
            | Self::LookupTableFlag(_)
            | Self::BytecodeValStage(_) => 2,
            Self::Recursion(poly) => 1 + poly.serialized_size(compress),
            Self::DorySparseConstraintMatrix => 1,
        }
    }
}

impl Valid for VirtualPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for VirtualPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(
            match u8::deserialize_with_mode(&mut reader, compress, validate)? {
                0 => Self::PC,
                1 => Self::UnexpandedPC,
                2 => Self::NextPC,
                3 => Self::NextUnexpandedPC,
                4 => Self::NextIsNoop,
                5 => Self::NextIsVirtual,
                6 => Self::NextIsFirstInSequence,
                7 => Self::LeftLookupOperand,
                8 => Self::RightLookupOperand,
                9 => Self::LeftInstructionInput,
                10 => Self::RightInstructionInput,
                11 => Self::Product,
                12 => Self::ShouldJump,
                13 => Self::ShouldBranch,
                14 => Self::WritePCtoRD,
                15 => Self::WriteLookupOutputToRD,
                16 => Self::Rd,
                17 => Self::Imm,
                18 => Self::Rs1Value,
                19 => Self::Rs2Value,
                20 => Self::RdWriteValue,
                21 => Self::Rs1Ra,
                22 => Self::Rs2Ra,
                23 => Self::RdWa,
                24 => Self::LookupOutput,
                25 => Self::InstructionRaf,
                26 => Self::InstructionRafFlag,
                27 => {
                    let i = u32::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::InstructionRa(i as usize)
                }
                28 => Self::RegistersVal,
                29 => Self::RamAddress,
                30 => Self::RamRa,
                31 => Self::RamReadValue,
                32 => Self::RamWriteValue,
                33 => Self::RamVal,
                34 => Self::RamValInit,
                35 => Self::RamValFinal,
                36 => Self::RamHammingWeight,
                37 => Self::UnivariateSkip,
                38 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = CircuitFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::OpFlags(flags)
                }
                39 => {
                    let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    let flags = InstructionFlags::from_repr(discriminant)
                        .ok_or(SerializationError::InvalidData)?;
                    Self::InstructionFlags(flags)
                }
                40 => {
                    let flag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::LookupTableFlag(flag as usize)
                }
                // Program-image commitment variants
                41 => {
                    let stage = u8::deserialize_with_mode(&mut reader, compress, validate)?;
                    Self::BytecodeValStage(stage as usize)
                }
                42 => Self::BytecodeReadRafAddrClaim,
                43 => Self::BooleanityAddrClaim,
                44 => Self::BytecodeClaimReductionIntermediate,
                45 => Self::ProgramImageInitContributionRw,
                46 => Self::ProgramImageInitContributionRaf,
                47 => {
                    let poly = crate::zkvm::witness::RecursionPoly::deserialize_with_mode(
                        &mut reader,
                        compress,
                        validate,
                    )?;
                    Self::Recursion(poly)
                }
                48 => Self::DorySparseConstraintMatrix,
                _ => return Err(SerializationError::InvalidData),
            },
        )
    }
}

pub fn serialize_and_print_size(
    item_name: &str,
    file_name: &str,
    item: &impl CanonicalSerialize,
) -> Result<(), SerializationError> {
    let mut file = File::create(file_name)?;
    item.serialize_compressed(&mut file)?;
    let file_size_bytes = file.metadata()?.len();
    let file_size_kb = file_size_bytes as f64 / 1024.0;
    tracing::info!("{item_name} Written to {file_name}");
    tracing::info!("{item_name} size: {file_size_kb:.1} kB");
    Ok(())
}
