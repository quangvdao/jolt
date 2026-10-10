//! Version-zero proof envelope: header, bounded scheme commitment, eight batches
//! of fixed-width field elements, then a bounded scheme opening. In-memory clear
//! rounds omit the linear coefficient and trim trailing zeros to one coefficient;
//! wire rounds pad that canonical form to the batch degree. Decoding validates
//! dimensions and available bytes before allocating any round or value vector.

use crate::stages::{stage1, stage2, stage3a, stage3b, stage4, stage5, stage6a, stage6b};
use crate::{
    commitment::{BitsCommitmentScheme, BitsGeometry, BitsWire},
    error::ProofDecodeError,
    statement::LOG_T_MAX,
};
use jolt_crypto::NoCommitment;
use jolt_field::{CanonicalBytes, CanonicalEncoding, Zero, F128};
use jolt_poly::CompressedPoly;
use jolt_rv64i_arith::Layout;
use jolt_sumcheck::BatchPrelude;
use jolt_sumcheck::{ClearProof, CompressedSumcheckProof, SumcheckProof};

#[derive(Clone, Debug)]
/// Canonical compressed clear rounds and the ordered wire values of one reduction batch.
/// `Rv64iProof::validate_shape` rejects other round forms and noncanonical coefficient counts before verification.
pub struct BatchProof<V> {
    /// Round messages with the linear coefficient omitted and trailing zeros trimmed to at least one stored coefficient.
    pub rounds: SumcheckProof<F128, NoCommitment>,
    /// Only the batch's transmitted evaluations, excluding derived values and aliased copies.
    pub values: V,
}

trait WireValues: Sized {
    fn write(&self, out: &mut Vec<u8>);
    fn read(cursor: &mut Cursor<'_>) -> Result<Self, ProofDecodeError>;
}
macro_rules! values {
    ($name:ident { $($field:ident),+ $(,)? }) => {
        /// Ordered binary-field evaluation cells transmitted by one batch; derived and aliased cells are absent.
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub struct $name {
            $(
                /// A transmitted evaluation in the batch's declared wire order.
                pub $field: F128
            ),+
        }
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
/// Evaluations of all 256 bit columns at the final cycle point, transmitted in column-index order.
pub struct BitsColumns(
    /// Exactly 256 binary-field evaluations; `validate_shape` rejects any other count.
    pub Vec<F128>,
);
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

/// Version-zero proof envelope: chosen RAM size, final PC, scheme messages and the eight ordered reduction batches.
/// Public trace and bytecode dimensions come from checked inputs; decoding validates byte shape, not the sum-check claims.
pub struct Rv64iProof<S: BitsCommitmentScheme> {
    /// The prover's RAM-word exponent, admitted in `5..=61` and bounded by checked memory geometry.
    pub log_K_ram: u8,
    /// The successor PC of the last cycle, checked to identify a valid bytecode row.
    pub final_pc: u64,
    /// Scheme commitment messages absorbed after the preamble and before front-end challenges.
    pub bits_commitment: S::Commitment,
    /// Outer row-system reductions over the binary and extension-field blocks.
    pub stage1: BatchProof<OuterValues>,
    /// Inner witness-column reduction.
    pub stage2: BatchProof<InnerValues>,
    /// Short router folds.
    pub stage3a: BatchProof<RouterFoldValues>,
    /// Cycle reductions for the five routers.
    pub stage3b: BatchProof<RouterCycleValues>,
    /// Register reads, RAM reads and public output checking.
    pub stage4: BatchProof<ReadCheckingValues>,
    /// Register and RAM value evaluation reductions.
    pub stage5: BatchProof<ValEvaluationValues>,
    /// Bytecode address reduction.
    pub stage6a: BatchProof<BytecodeAddressValue>,
    /// Bytecode cycle, RAM selector product and bit-column reductions.
    pub stage6b: BatchProof<BitsColumns>,
    /// Evidence opened after all column evaluations are absorbed and the column point is drawn.
    pub opening: S::OpeningProof,
}

// The derive callback supplies every member and its order to both geometry forms.
macro_rules! batch_geometry {
    (batch = $batch:ident, label = $label:literal, aggregates = { $($aggregates:tt)* },
     shape = $shape:ident, members = [ $({name: $member:ident, relation: $relation:ident, presence: required},)+ ]) => {
        impl<F: ::jolt_field::JoltField> $batch<F> {
            /// Returns the generated member windows and maximum round count and degree without transcript operations.
            /// Concrete member dimensions determine the schedule; invalid windows return a typed verifier error.
            pub fn geometry(&self) -> Result<::jolt_sumcheck::BatchPrelude<F>, ::jolt_verifier::VerifierError> {
                use ::jolt_verifier::stages::relations::ConcreteSumcheck as _;
                let mut max_num_vars = 0usize;
                let mut max_degree = 0usize;
                $(
                    max_num_vars = max_num_vars.max(self.$member.rounds());
                    max_degree = max_degree.max(self.$member.degree());
                )+
                let members = vec![$(
                    ::jolt_sumcheck::BatchMember {
                        input_claim: F::zero(), coefficient: F::zero(),
                        rounds: self.$member.rounds(), offset: self.$member.instance_point_offset(max_num_vars)?,
                    }
                ),+];
                ::jolt_sumcheck::BatchPrelude::try_new(members, max_num_vars, max_degree)
                    .map_err(|error| ::jolt_verifier::VerifierError::StageClaimSumcheckFailed {
                        stage: $label.to_owned(), reason: error.to_string(),
                    })
            }
        }
    };
}
pub(crate) use batch_geometry;

macro_rules! unit_batch_geometry {
    (batch = $batch:ident, label = $label:literal, aggregates = { $($aggregates:tt)* },
     shape = $shape:ident, members = [{name: $member:ident, relation: $relation:ident, presence: required},]) => {
        impl $batch<::jolt_field::F128> {
            /// Returns this one-member batch's canonical symbolic schedule without constructing public tables.
            /// The trace and layout arguments are unused because its symbolic shape is fixed.
            pub fn geometry_for(_log_T: usize, _layout: &::jolt_rv64i_arith::Layout)
                -> Result<::jolt_sumcheck::BatchPrelude<::jolt_field::F128>, ::jolt_verifier::VerifierError> {
                use ::jolt_claims::SymbolicSumcheck as _;
                use ::jolt_field::Zero as _;
                type Symbolic = <$relation<::jolt_field::F128> as ::jolt_verifier::stages::relations::ConcreteSumcheck<::jolt_field::F128>>::Symbolic;
                let symbolic = Symbolic::new(());
                let rounds = symbolic.rounds();
                let degree = symbolic.degree();
                let members = vec![::jolt_sumcheck::BatchMember {
                    input_claim: ::jolt_field::F128::zero(), coefficient: ::jolt_field::F128::zero(), rounds, offset: 0,
                }];
                ::jolt_sumcheck::BatchPrelude::try_new(members, rounds, degree)
                    .map_err(|error| ::jolt_verifier::VerifierError::StageClaimSumcheckFailed {
                        stage: $label.to_owned(), reason: error.to_string(),
                    })
            }
        }
    };
}
pub(crate) use unit_batch_geometry;

/// Reads the eight schedules from their generated member lists after validating dimensions.
/// Geometry construction uses bounded points and no statement or witness tables.
pub fn geometry(t: usize, b: usize, a: usize) -> Result<[BatchPrelude<F128>; 8], ProofDecodeError> {
    if !(1..=usize::from(LOG_T_MAX)).contains(&t)
        || !(1..=24).contains(&b)
        || !(5..=61).contains(&a)
    {
        return Err(ProofDecodeError::Dimensions);
    }
    let layout = Layout::new(b, a, 0).map_err(|_| ProofDecodeError::Dimensions)?;
    Ok([
        stage1::Stage1Sumchecks::for_geometry(t, &layout)
            .map_err(|_| ProofDecodeError::Dimensions)?
            .geometry()
            .map_err(|_| ProofDecodeError::Dimensions)?,
        stage2::Stage2Sumchecks::geometry_for(t, &layout)
            .map_err(|_| ProofDecodeError::Dimensions)?,
        stage3a::Stage3aSumchecks::geometry_for(t, &layout)
            .map_err(|_| ProofDecodeError::Dimensions)?,
        stage3b::Stage3bSumchecks::for_geometry(t, &layout)
            .map_err(|_| ProofDecodeError::Dimensions)?
            .geometry()
            .map_err(|_| ProofDecodeError::Dimensions)?,
        stage4::Stage4Sumchecks::for_geometry(t, &layout)
            .map_err(|_| ProofDecodeError::Dimensions)?
            .geometry()
            .map_err(|_| ProofDecodeError::Dimensions)?,
        stage5::Stage5Sumchecks::for_geometry(t, &layout)
            .map_err(|_| ProofDecodeError::Dimensions)?
            .geometry()
            .map_err(|_| ProofDecodeError::Dimensions)?,
        stage6a::Stage6aSumchecks::for_geometry(t, &layout)
            .map_err(|_| ProofDecodeError::Dimensions)?
            .geometry()
            .map_err(|_| ProofDecodeError::Dimensions)?,
        stage6b::Stage6bSumchecks::for_geometry(t, &layout)
            .map_err(|_| ProofDecodeError::Dimensions)?
            .geometry()
            .map_err(|_| ProofDecodeError::Dimensions)?,
    ])
}
impl<S: BitsCommitmentScheme> Rv64iProof<S> {
    /// Checks admitted dimensions, expected batch round counts, canonical compressed clear messages and exactly 256 column values.
    /// Returns `Dimensions` or `ProofShape` before any sum-check; it does not verify evaluation claims.
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
            geometry(log_T, log_K_bytecode, usize::from(self.log_K_ram))?;
        self.stage1.validate((s1.max_num_vars, s1.max_degree))?;
        self.stage2.validate((s2.max_num_vars, s2.max_degree))?;
        self.stage3a.validate((s3a.max_num_vars, s3a.max_degree))?;
        self.stage3b.validate((s3b.max_num_vars, s3b.max_degree))?;
        self.stage4.validate((s4.max_num_vars, s4.max_degree))?;
        self.stage5.validate((s5.max_num_vars, s5.max_degree))?;
        self.stage6a.validate((s6a.max_num_vars, s6a.max_degree))?;
        self.stage6b.validate((s6b.max_num_vars, s6b.max_degree))?;
        if self.stage6b.values.0.len() != 256 {
            return Err(ProofDecodeError::ProofShape);
        }
        Ok(())
    }
    /// Encodes a shape-checked proof with fixed-width padded rounds and bounded scheme byte strings.
    /// Returns an empty vector for an invalid in-memory proof.
    pub fn to_bytes(&self) -> Vec<u8> {
        let t = self.stage6b.round_count();
        let b = self.stage6a.round_count();
        if self.validate_shape(t, b).is_err() {
            return Vec::new();
        }
        let mut out = vec![0, self.log_K_ram];
        out.extend_from_slice(&self.final_pc.to_le_bytes());
        write_scheme(&self.bits_commitment, &mut out);
        let Ok([s1, s2, s3a, s3b, s4, s5, s6a, s6b]) = geometry(t, b, usize::from(self.log_K_ram))
        else {
            return Vec::new();
        };
        self.stage1.write(s1.max_degree, &mut out);
        self.stage2.write(s2.max_degree, &mut out);
        self.stage3a.write(s3a.max_degree, &mut out);
        self.stage3b.write(s3b.max_degree, &mut out);
        self.stage4.write(s4.max_degree, &mut out);
        self.stage5.write(s5.max_degree, &mut out);
        self.stage6a.write(s6a.max_degree, &mut out);
        self.stage6b.write(s6b.max_degree, &mut out);
        write_scheme(&self.opening, &mut out);
        out
    }
    /// Decodes one complete version-zero envelope using public dimensions `1..=LOG_T_MAX` and `1..=24`.
    /// Returns a typed error for invalid version or dimensions, unavailable bytes, invalid scheme encodings or trailing bytes; allocation is bounded by the input.
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
        )?;
        let stage1 = BatchProof::read(&mut cursor, (s1.max_num_vars, s1.max_degree))?;
        let stage2 = BatchProof::read(&mut cursor, (s2.max_num_vars, s2.max_degree))?;
        let stage3a = BatchProof::read(&mut cursor, (s3a.max_num_vars, s3a.max_degree))?;
        let stage3b = BatchProof::read(&mut cursor, (s3b.max_num_vars, s3b.max_degree))?;
        let stage4 = BatchProof::read(&mut cursor, (s4.max_num_vars, s4.max_degree))?;
        let stage5 = BatchProof::read(&mut cursor, (s5.max_num_vars, s5.max_degree))?;
        let stage6a = BatchProof::read(&mut cursor, (s6a.max_num_vars, s6a.max_degree))?;
        let stage6b = BatchProof::read(&mut cursor, (s6b.max_num_vars, s6b.max_degree))?;
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
