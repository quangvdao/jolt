//! Version-zero proof envelope: header, bounded scheme commitment, eight batches
//! of fixed-width field elements, then a bounded scheme opening. In-memory clear
//! rounds omit the linear coefficient and trim trailing zeros to one coefficient;
//! wire rounds pad that canonical form to the batch degree. Decoding validates
//! dimensions and available bytes before allocating any round or value vector.

use crate::{
    commitment::{BitsCommitmentScheme, BitsGeometry, BitsWire},
    error::ProofDecodeError,
    statement::LOG_T_MAX,
};
use jolt_crypto::NoCommitment;
use jolt_field::{CanonicalBytes, CanonicalEncoding, Zero, F128};
use jolt_poly::CompressedPoly;
use jolt_sumcheck::{ClearProof, CompressedSumcheckProof, SumcheckProof};

#[derive(Clone, Debug)]
pub struct BatchProof<V> {
    pub rounds: SumcheckProof<F128, NoCommitment>,
    pub values: V,
}

trait WireValues: Sized {
    fn write(&self, out: &mut Vec<u8>);
    fn read(cursor: &mut Cursor<'_>) -> Result<Self, ProofDecodeError>;
}
macro_rules! values {
    ($name:ident { $($field:ident),+ $(,)? }) => {
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub struct $name { $(pub $field: F128),+ }
        impl WireValues for $name {
            fn write(&self, out: &mut Vec<u8>) { $(write_element(self.$field, out);)+ }
            fn read(cursor: &mut Cursor<'_>) -> Result<Self, ProofDecodeError> { Ok(Self { $($field: cursor.element()?),+ }) }
        }
    };
}
values!(OuterValues {
    az_f2,
    bz_f2,
    cz_f2,
    az_f128,
    bz_f128,
    cz_f128
});
values!(InnerValues {
    witness_routed,
    direct_columns
});
values!(RouterFoldValues {
    variant,
    shift,
    memory,
    compare,
    branch
});
values!(RouterCycleValues {
    rs1_value,
    rs2_value,
    rd_pre_value,
    imm,
    fall_through_pc,
    pc_plus_imm,
    pc,
    next_pc,
    variant_bits,
    variant,
    shift_kind,
    pos_ra_0,
    pos_ra_1,
    ram_read_value,
    access_kind,
    key_kind,
    branch,
    should_branch
});
values!(ReadCheckingValues {
    rs1_ra,
    rs2_ra,
    rd_wa,
    registers_val,
    ram_ra,
    ram_val,
    ram_val_final
});
values!(ValEvaluationValues {
    rd_wa,
    store,
    inc,
    ram_ra
});
values!(BytecodeAddressValue { address_claim });
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitsColumns(pub Vec<F128>);
impl WireValues for BitsColumns {
    fn write(&self, out: &mut Vec<u8>) {
        for value in &self.0 {
            write_element(*value, out);
        }
    }
    fn read(cursor: &mut Cursor<'_>) -> Result<Self, ProofDecodeError> {
        cursor.require(256 * 16)?;
        let mut values = Vec::with_capacity(256);
        for _ in 0..256 {
            values.push(cursor.element()?);
        }
        Ok(Self(values))
    }
}

pub struct Rv64iProof<S: BitsCommitmentScheme> {
    pub log_K_ram: u8,
    pub final_pc: u64,
    pub bits_commitment: S::Commitment,
    pub stage1: BatchProof<OuterValues>,
    pub stage2: BatchProof<InnerValues>,
    pub stage3a: BatchProof<RouterFoldValues>,
    pub stage3b: BatchProof<RouterCycleValues>,
    pub stage4: BatchProof<ReadCheckingValues>,
    pub stage5: BatchProof<ValEvaluationValues>,
    pub stage6a: BatchProof<BytecodeAddressValue>,
    pub stage6b: BatchProof<BitsColumns>,
    pub opening: S::OpeningProof,
}

/// The eight envelope geometries of §5, until the corresponding stage batches exist.
/// Relation member lists remain owned by their batch derives as stages are filled.
fn geometry(t: usize, b: usize, a: usize) -> [(usize, usize); 8] {
    [
        (t + 8, 3),
        (10, 2),
        (17, 2),
        (t, 5),
        (a + t, 3),
        (t, 4),
        (b, 2),
        (t, b.div_ceil(4).max(a.div_ceil(4)) + 1),
    ]
}
impl<S: BitsCommitmentScheme> Rv64iProof<S> {
    /// Rejects malformed in-memory round forms and column vectors before sum-check.
    pub fn validate_shape(
        &self,
        log_T: usize,
        log_K_bytecode: usize,
    ) -> Result<(), ProofDecodeError> {
        if !(1..=usize::from(LOG_T_MAX)).contains(&log_T)
            || !(1..=24).contains(&log_K_bytecode)
            || !(5..=61).contains(&self.log_K_ram)
        {
            return Err(ProofDecodeError::Dimensions);
        }
        let [s1, s2, s3a, s3b, s4, s5, s6a, s6b] =
            geometry(log_T, log_K_bytecode, usize::from(self.log_K_ram));
        self.stage1.validate(s1)?;
        self.stage2.validate(s2)?;
        self.stage3a.validate(s3a)?;
        self.stage3b.validate(s3b)?;
        self.stage4.validate(s4)?;
        self.stage5.validate(s5)?;
        self.stage6a.validate(s6a)?;
        self.stage6b.validate(s6b)?;
        if self.stage6b.values.0.len() != 256 {
            return Err(ProofDecodeError::ProofShape);
        }
        Ok(())
    }
    /// Serializes a shape-checked proof; an invalid in-memory proof has no encoding.
    pub fn to_bytes(&self) -> Vec<u8> {
        let t = self.stage6b.round_count();
        let b = self.stage6a.round_count();
        if self.validate_shape(t, b).is_err() {
            return Vec::new();
        }
        let mut out = vec![0, self.log_K_ram];
        out.extend_from_slice(&self.final_pc.to_le_bytes());
        write_scheme(&self.bits_commitment, &mut out);
        let [s1, s2, s3a, s3b, s4, s5, s6a, s6b] = geometry(t, b, usize::from(self.log_K_ram));
        self.stage1.write(s1.1, &mut out);
        self.stage2.write(s2.1, &mut out);
        self.stage3a.write(s3a.1, &mut out);
        self.stage3b.write(s3b.1, &mut out);
        self.stage4.write(s4.1, &mut out);
        self.stage5.write(s5.1, &mut out);
        self.stage6a.write(s6a.1, &mut out);
        self.stage6b.write(s6b.1, &mut out);
        write_scheme(&self.opening, &mut out);
        out
    }
    pub fn from_bytes(
        bytes: &[u8],
        log_T: u8,
        log_K_bytecode: u8,
    ) -> Result<Self, ProofDecodeError> {
        let mut cursor = Cursor { bytes };
        if cursor.take(1)? != [0] {
            return Err(ProofDecodeError::InvalidVersion);
        }
        let log_K_ram = *cursor.take(1)?.first().ok_or(ProofDecodeError::Truncated)?;
        if !(1..=LOG_T_MAX).contains(&log_T)
            || !(1..=24).contains(&log_K_bytecode)
            || !(5..=61).contains(&log_K_ram)
        {
            return Err(ProofDecodeError::Dimensions);
        }
        let final_pc = cursor.word()?;
        let bits_geometry = BitsGeometry {
            log_T: usize::from(log_T),
        };
        let bits_commitment =
            S::Commitment::read(cursor.scheme()?, bits_geometry).ok_or(ProofDecodeError::Scheme)?;
        let [s1, s2, s3a, s3b, s4, s5, s6a, s6b] = geometry(
            usize::from(log_T),
            usize::from(log_K_bytecode),
            usize::from(log_K_ram),
        );
        let stage1 = BatchProof::read(&mut cursor, s1)?;
        let stage2 = BatchProof::read(&mut cursor, s2)?;
        let stage3a = BatchProof::read(&mut cursor, s3a)?;
        let stage3b = BatchProof::read(&mut cursor, s3b)?;
        let stage4 = BatchProof::read(&mut cursor, s4)?;
        let stage5 = BatchProof::read(&mut cursor, s5)?;
        let stage6a = BatchProof::read(&mut cursor, s6a)?;
        let stage6b = BatchProof::read(&mut cursor, s6b)?;
        let opening = S::OpeningProof::read(cursor.scheme()?, bits_geometry)
            .ok_or(ProofDecodeError::Scheme)?;
        if !cursor.bytes.is_empty() {
            return Err(ProofDecodeError::TrailingBytes);
        }
        Ok(Self {
            log_K_ram,
            final_pc,
            bits_commitment,
            stage1,
            stage2,
            stage3a,
            stage3b,
            stage4,
            stage5,
            stage6a,
            stage6b,
            opening,
        })
    }
}
impl<V> BatchProof<V> {
    fn round_count(&self) -> usize {
        if let SumcheckProof::Clear(ClearProof::Compressed(proof)) = &self.rounds {
            proof.round_polynomials.len()
        } else {
            0
        }
    }
    fn validate(&self, (rounds, degree): (usize, usize)) -> Result<(), ProofDecodeError> {
        if let SumcheckProof::Clear(ClearProof::Compressed(proof)) = &self.rounds {
            if proof.round_polynomials.len() != rounds {
                return Err(ProofDecodeError::ProofShape);
            }
            for round in &proof.round_polynomials {
                let coefficients = round.coeffs_except_linear_term();
                if coefficients.is_empty()
                    || coefficients.len() > degree
                    || (coefficients.len() >= 2 && coefficients.last() == Some(&F128::zero()))
                {
                    return Err(ProofDecodeError::ProofShape);
                }
            }
            Ok(())
        } else {
            Err(ProofDecodeError::ProofShape)
        }
    }
    fn write(&self, degree: usize, out: &mut Vec<u8>)
    where
        V: WireValues,
    {
        if let SumcheckProof::Clear(ClearProof::Compressed(proof)) = &self.rounds {
            for round in &proof.round_polynomials {
                for value in round.coeffs_except_linear_term() {
                    write_element(*value, out);
                }
                for _ in round.coeffs_except_linear_term().len()..degree {
                    write_element(F128::zero(), out);
                }
            }
        }
        self.values.write(out);
    }
    fn read(
        cursor: &mut Cursor<'_>,
        (rounds, degree): (usize, usize),
    ) -> Result<Self, ProofDecodeError>
    where
        V: WireValues,
    {
        cursor.require(rounds * degree * 16)?;
        let mut round_polynomials = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let mut coefficients = Vec::with_capacity(degree);
            for _ in 0..degree {
                coefficients.push(cursor.element()?);
            }
            while coefficients.len() > 1 && coefficients.last() == Some(&F128::zero()) {
                let _ = coefficients.pop();
            }
            round_polynomials.push(CompressedPoly::new(coefficients));
        }
        Ok(Self {
            rounds: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
                round_polynomials,
            })),
            values: V::read(cursor)?,
        })
    }
}
fn write_element(value: F128, out: &mut Vec<u8>) {
    let mut bytes = [0; 16];
    value.to_bytes_le(&mut bytes);
    out.extend_from_slice(&bytes);
}
fn write_scheme<W: BitsWire>(wire: &W, out: &mut Vec<u8>) {
    let prefix = out.len();
    out.extend_from_slice(&[0; 8]);
    let start = out.len();
    wire.write(out);
    let length = (out.len() - start) as u64;
    if let Some(bytes) = out.get_mut(prefix..start) {
        bytes.copy_from_slice(&length.to_le_bytes());
    }
}
struct Cursor<'a> {
    bytes: &'a [u8],
}
impl<'a> Cursor<'a> {
    fn require(&self, length: usize) -> Result<(), ProofDecodeError> {
        if length <= self.bytes.len() {
            Ok(())
        } else {
            Err(ProofDecodeError::Truncated)
        }
    }
    fn take(&mut self, length: usize) -> Result<&'a [u8], ProofDecodeError> {
        self.require(length)?;
        let (head, tail) = self.bytes.split_at(length);
        self.bytes = tail;
        Ok(head)
    }
    fn word(&mut self) -> Result<u64, ProofDecodeError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| ProofDecodeError::Truncated)?,
        ))
    }
    fn element(&mut self) -> Result<F128, ProofDecodeError> {
        F128::from_bytes_le_checked(self.take(16)?).ok_or(ProofDecodeError::ProofShape)
    }
    fn scheme(&mut self) -> Result<&'a [u8], ProofDecodeError> {
        let length = usize::try_from(self.word()?).map_err(|_| ProofDecodeError::Length)?;
        if length > self.bytes.len() {
            return Err(ProofDecodeError::Length);
        }
        self.take(length)
    }
}
