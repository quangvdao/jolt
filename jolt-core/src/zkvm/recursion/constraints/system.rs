//! Recursion constraint-system types (streaming plan output).

use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};
use crate::zkvm::recursion::g1::scalar_multiplication::G1ScalarMulPublicInputs;
use crate::zkvm::recursion::g2::scalar_multiplication::G2ScalarMulPublicInputs;
use crate::zkvm::recursion::gt::exponentiation::{GtExpPublicInputs, GtExpWitness};
use ark_bn254::{Fq, Fq2};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid};

/// Convert index to binary representation as field elements (little-endian).
pub fn index_to_binary<F: JoltField>(index: usize, num_vars: usize) -> Vec<F> {
    let mut binary = Vec::with_capacity(num_vars);
    let mut idx = index;
    for _ in 0..num_vars {
        binary.push(if idx & 1 == 1 { F::one() } else { F::zero() });
        idx >>= 1;
    }
    binary
}

/// Polynomial types (virtual claim rows) in the recursion system.
///
/// The discriminant ordering is part of the proof format: Stage 2 virtual claims are laid out
/// as `[constraint-major, poly-type-minor]` using `poly_type as usize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum PolyType {
    // Packed GT Exponentiation polynomials (11-var each, one constraint per GT exp)
    RhoPrev = 0,  // rho(s,x) - packed intermediate results (11-var)
    Quotient = 1, // quotient(s,x) - packed quotients (11-var)

    // GT Multiplication polynomials
    MulLhs = 2,
    MulRhs = 3,
    MulResult = 4,
    MulQuotient = 5,

    // G1 Scalar Multiplication polynomials (8-var)
    G1ScalarMulXA = 6,
    G1ScalarMulYA = 7,
    G1ScalarMulXT = 8,
    G1ScalarMulYT = 9,
    G1ScalarMulXANext = 10,
    G1ScalarMulYANext = 11,
    G1ScalarMulTIndicator = 12,
    G1ScalarMulAIndicator = 13,
    G1ScalarMulBit = 14, // NOT committed; kept for stable indexing

    // G2 Scalar Multiplication polynomials (Fq2 coords split into c0/c1; 8-var)
    G2ScalarMulXAC0 = 15,
    G2ScalarMulXAC1 = 16,
    G2ScalarMulYAC0 = 17,
    G2ScalarMulYAC1 = 18,
    G2ScalarMulXTC0 = 19,
    G2ScalarMulXTC1 = 20,
    G2ScalarMulYTC0 = 21,
    G2ScalarMulYTC1 = 22,
    G2ScalarMulXANextC0 = 23,
    G2ScalarMulXANextC1 = 24,
    G2ScalarMulYANextC0 = 25,
    G2ScalarMulYANextC1 = 26,
    G2ScalarMulTIndicator = 27,
    G2ScalarMulAIndicator = 28,
    G2ScalarMulBit = 29, // NOT committed; kept for stable indexing

    // G1 Addition polynomials (0-var committed polys; stored as scalars)
    G1AddXP = 30,
    G1AddYP = 31,
    G1AddPIndicator = 32,
    G1AddXQ = 33,
    G1AddYQ = 34,
    G1AddQIndicator = 35,
    G1AddXR = 36,
    G1AddYR = 37,
    G1AddRIndicator = 38,
    G1AddLambda = 39,
    G1AddInvDeltaX = 40,
    G1AddIsDouble = 41,
    G1AddIsInverse = 42,

    // G2 Addition polynomials (Fq2 coords split into c0/c1; 0-var committed polys)
    G2AddXPC0 = 43,
    G2AddXPC1 = 44,
    G2AddYPC0 = 45,
    G2AddYPC1 = 46,
    G2AddPIndicator = 47,
    G2AddXQC0 = 48,
    G2AddXQC1 = 49,
    G2AddYQC0 = 50,
    G2AddYQC1 = 51,
    G2AddQIndicator = 52,
    G2AddXRC0 = 53,
    G2AddXRC1 = 54,
    G2AddYRC0 = 55,
    G2AddYRC1 = 56,
    G2AddRIndicator = 57,
    G2AddLambdaC0 = 58,
    G2AddLambdaC1 = 59,
    G2AddInvDeltaXC0 = 60,
    G2AddInvDeltaXC1 = 61,
    G2AddIsDouble = 62,
    G2AddIsInverse = 63,

    // Pairing recursion (experimental): Multi-Miller loop packed traces
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopF = 64,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopFNext = 65,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopQuotient = 66,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopTXC0 = 67,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopTXC1 = 68,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopTYC0 = 69,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopTYC1 = 70,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopTXC0Next = 71,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopTXC1Next = 72,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopTYC0Next = 73,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopTYC1Next = 74,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopLambdaC0 = 75,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopLambdaC1 = 76,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopInvDeltaXC0 = 77,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopInvDeltaXC1 = 78,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopXP = 79,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopYP = 80,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopXQC0 = 81,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopXQC1 = 82,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopYQC0 = 83,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopYQC1 = 84,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopIsDouble = 85,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopIsAdd = 86,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopLVal = 87,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopInvTwoYC0 = 88,
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopInvTwoYC1 = 89,
}

impl PolyType {
    #[cfg(not(feature = "experimental-pairing-recursion"))]
    pub const NUM_TYPES: usize = 64;
    #[cfg(feature = "experimental-pairing-recursion")]
    pub const NUM_TYPES: usize = 90;
}

impl CanonicalSerialize for PolyType {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        // PolyType discriminants are stable and fit in u8.
        (*self as u8).serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        1
    }
}

impl CanonicalDeserialize for PolyType {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        compress: Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        let v = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let ty = match v {
            0 => PolyType::RhoPrev,
            1 => PolyType::Quotient,
            2 => PolyType::MulLhs,
            3 => PolyType::MulRhs,
            4 => PolyType::MulResult,
            5 => PolyType::MulQuotient,
            6 => PolyType::G1ScalarMulXA,
            7 => PolyType::G1ScalarMulYA,
            8 => PolyType::G1ScalarMulXT,
            9 => PolyType::G1ScalarMulYT,
            10 => PolyType::G1ScalarMulXANext,
            11 => PolyType::G1ScalarMulYANext,
            12 => PolyType::G1ScalarMulTIndicator,
            13 => PolyType::G1ScalarMulAIndicator,
            14 => PolyType::G1ScalarMulBit,
            15 => PolyType::G2ScalarMulXAC0,
            16 => PolyType::G2ScalarMulXAC1,
            17 => PolyType::G2ScalarMulYAC0,
            18 => PolyType::G2ScalarMulYAC1,
            19 => PolyType::G2ScalarMulXTC0,
            20 => PolyType::G2ScalarMulXTC1,
            21 => PolyType::G2ScalarMulYTC0,
            22 => PolyType::G2ScalarMulYTC1,
            23 => PolyType::G2ScalarMulXANextC0,
            24 => PolyType::G2ScalarMulXANextC1,
            25 => PolyType::G2ScalarMulYANextC0,
            26 => PolyType::G2ScalarMulYANextC1,
            27 => PolyType::G2ScalarMulTIndicator,
            28 => PolyType::G2ScalarMulAIndicator,
            29 => PolyType::G2ScalarMulBit,
            30 => PolyType::G1AddXP,
            31 => PolyType::G1AddYP,
            32 => PolyType::G1AddPIndicator,
            33 => PolyType::G1AddXQ,
            34 => PolyType::G1AddYQ,
            35 => PolyType::G1AddQIndicator,
            36 => PolyType::G1AddXR,
            37 => PolyType::G1AddYR,
            38 => PolyType::G1AddRIndicator,
            39 => PolyType::G1AddLambda,
            40 => PolyType::G1AddInvDeltaX,
            41 => PolyType::G1AddIsDouble,
            42 => PolyType::G1AddIsInverse,
            43 => PolyType::G2AddXPC0,
            44 => PolyType::G2AddXPC1,
            45 => PolyType::G2AddYPC0,
            46 => PolyType::G2AddYPC1,
            47 => PolyType::G2AddPIndicator,
            48 => PolyType::G2AddXQC0,
            49 => PolyType::G2AddXQC1,
            50 => PolyType::G2AddYQC0,
            51 => PolyType::G2AddYQC1,
            52 => PolyType::G2AddQIndicator,
            53 => PolyType::G2AddXRC0,
            54 => PolyType::G2AddXRC1,
            55 => PolyType::G2AddYRC0,
            56 => PolyType::G2AddYRC1,
            57 => PolyType::G2AddRIndicator,
            58 => PolyType::G2AddLambdaC0,
            59 => PolyType::G2AddLambdaC1,
            60 => PolyType::G2AddInvDeltaXC0,
            61 => PolyType::G2AddInvDeltaXC1,
            62 => PolyType::G2AddIsDouble,
            63 => PolyType::G2AddIsInverse,
            #[cfg(feature = "experimental-pairing-recursion")]
            64 => PolyType::MultiMillerLoopF,
            #[cfg(feature = "experimental-pairing-recursion")]
            65 => PolyType::MultiMillerLoopFNext,
            #[cfg(feature = "experimental-pairing-recursion")]
            66 => PolyType::MultiMillerLoopQuotient,
            #[cfg(feature = "experimental-pairing-recursion")]
            67 => PolyType::MultiMillerLoopTXC0,
            #[cfg(feature = "experimental-pairing-recursion")]
            68 => PolyType::MultiMillerLoopTXC1,
            #[cfg(feature = "experimental-pairing-recursion")]
            69 => PolyType::MultiMillerLoopTYC0,
            #[cfg(feature = "experimental-pairing-recursion")]
            70 => PolyType::MultiMillerLoopTYC1,
            #[cfg(feature = "experimental-pairing-recursion")]
            71 => PolyType::MultiMillerLoopTXC0Next,
            #[cfg(feature = "experimental-pairing-recursion")]
            72 => PolyType::MultiMillerLoopTXC1Next,
            #[cfg(feature = "experimental-pairing-recursion")]
            73 => PolyType::MultiMillerLoopTYC0Next,
            #[cfg(feature = "experimental-pairing-recursion")]
            74 => PolyType::MultiMillerLoopTYC1Next,
            #[cfg(feature = "experimental-pairing-recursion")]
            75 => PolyType::MultiMillerLoopLambdaC0,
            #[cfg(feature = "experimental-pairing-recursion")]
            76 => PolyType::MultiMillerLoopLambdaC1,
            #[cfg(feature = "experimental-pairing-recursion")]
            77 => PolyType::MultiMillerLoopInvDeltaXC0,
            #[cfg(feature = "experimental-pairing-recursion")]
            78 => PolyType::MultiMillerLoopInvDeltaXC1,
            #[cfg(feature = "experimental-pairing-recursion")]
            79 => PolyType::MultiMillerLoopXP,
            #[cfg(feature = "experimental-pairing-recursion")]
            80 => PolyType::MultiMillerLoopYP,
            #[cfg(feature = "experimental-pairing-recursion")]
            81 => PolyType::MultiMillerLoopXQC0,
            #[cfg(feature = "experimental-pairing-recursion")]
            82 => PolyType::MultiMillerLoopXQC1,
            #[cfg(feature = "experimental-pairing-recursion")]
            83 => PolyType::MultiMillerLoopYQC0,
            #[cfg(feature = "experimental-pairing-recursion")]
            84 => PolyType::MultiMillerLoopYQC1,
            #[cfg(feature = "experimental-pairing-recursion")]
            85 => PolyType::MultiMillerLoopIsDouble,
            #[cfg(feature = "experimental-pairing-recursion")]
            86 => PolyType::MultiMillerLoopIsAdd,
            #[cfg(feature = "experimental-pairing-recursion")]
            87 => PolyType::MultiMillerLoopLVal,
            #[cfg(feature = "experimental-pairing-recursion")]
            88 => PolyType::MultiMillerLoopInvTwoYC0,
            #[cfg(feature = "experimental-pairing-recursion")]
            89 => PolyType::MultiMillerLoopInvTwoYC1,
            _ => return Err(SerializationError::InvalidData),
        };
        Ok(ty)
    }
}

impl Valid for PolyType {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

/// Type of constraint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstraintType {
    /// Packed GT exponentiation constraint (one per GT exp, covers all 254 steps)
    GtExp,
    /// GT multiplication constraint
    GtMul,
    /// G1 scalar multiplication constraint with base point
    G1ScalarMul { base_point: (Fq, Fq) },
    /// G2 scalar multiplication constraint with base point (over Fq2)
    G2ScalarMul { base_point: (Fq2, Fq2) },
    /// G1 addition constraint
    G1Add,
    /// G2 addition constraint
    G2Add,
    /// Multi-Miller loop constraint (experimental)
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoop,
}

// -----------------------------------------------------------------------------
// Public mapping: ConstraintType -> committed PolyTypes (+ native num_vars)
// -----------------------------------------------------------------------------

const GT_EXP_POLYS: [PolyType; 2] = [PolyType::RhoPrev, PolyType::Quotient];
const GT_EXP_SPECS: [(PolyType, usize); 2] = [(PolyType::RhoPrev, 11), (PolyType::Quotient, 11)];

const GT_MUL_POLYS: [PolyType; 4] = [
    PolyType::MulLhs,
    PolyType::MulRhs,
    PolyType::MulResult,
    PolyType::MulQuotient,
];
const GT_MUL_SPECS: [(PolyType, usize); 4] = [
    (PolyType::MulLhs, 4),
    (PolyType::MulRhs, 4),
    (PolyType::MulResult, 4),
    (PolyType::MulQuotient, 4),
];

const G1_SCALAR_MUL_POLYS: [PolyType; 8] = [
    PolyType::G1ScalarMulXA,
    PolyType::G1ScalarMulYA,
    PolyType::G1ScalarMulXT,
    PolyType::G1ScalarMulYT,
    PolyType::G1ScalarMulXANext,
    PolyType::G1ScalarMulYANext,
    PolyType::G1ScalarMulTIndicator,
    PolyType::G1ScalarMulAIndicator,
];
const G1_SCALAR_MUL_SPECS: [(PolyType, usize); 8] = [
    (PolyType::G1ScalarMulXA, 8),
    (PolyType::G1ScalarMulYA, 8),
    (PolyType::G1ScalarMulXT, 8),
    (PolyType::G1ScalarMulYT, 8),
    (PolyType::G1ScalarMulXANext, 8),
    (PolyType::G1ScalarMulYANext, 8),
    (PolyType::G1ScalarMulTIndicator, 8),
    (PolyType::G1ScalarMulAIndicator, 8),
];

const G2_SCALAR_MUL_POLYS: [PolyType; 14] = [
    PolyType::G2ScalarMulXAC0,
    PolyType::G2ScalarMulXAC1,
    PolyType::G2ScalarMulYAC0,
    PolyType::G2ScalarMulYAC1,
    PolyType::G2ScalarMulXTC0,
    PolyType::G2ScalarMulXTC1,
    PolyType::G2ScalarMulYTC0,
    PolyType::G2ScalarMulYTC1,
    PolyType::G2ScalarMulXANextC0,
    PolyType::G2ScalarMulXANextC1,
    PolyType::G2ScalarMulYANextC0,
    PolyType::G2ScalarMulYANextC1,
    PolyType::G2ScalarMulTIndicator,
    PolyType::G2ScalarMulAIndicator,
];
const G2_SCALAR_MUL_SPECS: [(PolyType, usize); 14] = [
    (PolyType::G2ScalarMulXAC0, 8),
    (PolyType::G2ScalarMulXAC1, 8),
    (PolyType::G2ScalarMulYAC0, 8),
    (PolyType::G2ScalarMulYAC1, 8),
    (PolyType::G2ScalarMulXTC0, 8),
    (PolyType::G2ScalarMulXTC1, 8),
    (PolyType::G2ScalarMulYTC0, 8),
    (PolyType::G2ScalarMulYTC1, 8),
    (PolyType::G2ScalarMulXANextC0, 8),
    (PolyType::G2ScalarMulXANextC1, 8),
    (PolyType::G2ScalarMulYANextC0, 8),
    (PolyType::G2ScalarMulYANextC1, 8),
    (PolyType::G2ScalarMulTIndicator, 8),
    (PolyType::G2ScalarMulAIndicator, 8),
];

const G1_ADD_POLYS: [PolyType; 13] = [
    PolyType::G1AddXP,
    PolyType::G1AddYP,
    PolyType::G1AddPIndicator,
    PolyType::G1AddXQ,
    PolyType::G1AddYQ,
    PolyType::G1AddQIndicator,
    PolyType::G1AddXR,
    PolyType::G1AddYR,
    PolyType::G1AddRIndicator,
    PolyType::G1AddLambda,
    PolyType::G1AddInvDeltaX,
    PolyType::G1AddIsDouble,
    PolyType::G1AddIsInverse,
];
const G1_ADD_SPECS: [(PolyType, usize); 13] = [
    (PolyType::G1AddXP, 0),
    (PolyType::G1AddYP, 0),
    (PolyType::G1AddPIndicator, 0),
    (PolyType::G1AddXQ, 0),
    (PolyType::G1AddYQ, 0),
    (PolyType::G1AddQIndicator, 0),
    (PolyType::G1AddXR, 0),
    (PolyType::G1AddYR, 0),
    (PolyType::G1AddRIndicator, 0),
    (PolyType::G1AddLambda, 0),
    (PolyType::G1AddInvDeltaX, 0),
    (PolyType::G1AddIsDouble, 0),
    (PolyType::G1AddIsInverse, 0),
];

const G2_ADD_POLYS: [PolyType; 21] = [
    PolyType::G2AddXPC0,
    PolyType::G2AddXPC1,
    PolyType::G2AddYPC0,
    PolyType::G2AddYPC1,
    PolyType::G2AddPIndicator,
    PolyType::G2AddXQC0,
    PolyType::G2AddXQC1,
    PolyType::G2AddYQC0,
    PolyType::G2AddYQC1,
    PolyType::G2AddQIndicator,
    PolyType::G2AddXRC0,
    PolyType::G2AddXRC1,
    PolyType::G2AddYRC0,
    PolyType::G2AddYRC1,
    PolyType::G2AddRIndicator,
    PolyType::G2AddLambdaC0,
    PolyType::G2AddLambdaC1,
    PolyType::G2AddInvDeltaXC0,
    PolyType::G2AddInvDeltaXC1,
    PolyType::G2AddIsDouble,
    PolyType::G2AddIsInverse,
];
const G2_ADD_SPECS: [(PolyType, usize); 21] = [
    (PolyType::G2AddXPC0, 0),
    (PolyType::G2AddXPC1, 0),
    (PolyType::G2AddYPC0, 0),
    (PolyType::G2AddYPC1, 0),
    (PolyType::G2AddPIndicator, 0),
    (PolyType::G2AddXQC0, 0),
    (PolyType::G2AddXQC1, 0),
    (PolyType::G2AddYQC0, 0),
    (PolyType::G2AddYQC1, 0),
    (PolyType::G2AddQIndicator, 0),
    (PolyType::G2AddXRC0, 0),
    (PolyType::G2AddXRC1, 0),
    (PolyType::G2AddYRC0, 0),
    (PolyType::G2AddYRC1, 0),
    (PolyType::G2AddRIndicator, 0),
    (PolyType::G2AddLambdaC0, 0),
    (PolyType::G2AddLambdaC1, 0),
    (PolyType::G2AddInvDeltaXC0, 0),
    (PolyType::G2AddInvDeltaXC1, 0),
    (PolyType::G2AddIsDouble, 0),
    (PolyType::G2AddIsInverse, 0),
];

#[cfg(feature = "experimental-pairing-recursion")]
const MULTI_MILLER_LOOP_POLYS: [PolyType; 26] = [
    PolyType::MultiMillerLoopF,
    PolyType::MultiMillerLoopFNext,
    PolyType::MultiMillerLoopQuotient,
    PolyType::MultiMillerLoopTXC0,
    PolyType::MultiMillerLoopTXC1,
    PolyType::MultiMillerLoopTYC0,
    PolyType::MultiMillerLoopTYC1,
    PolyType::MultiMillerLoopTXC0Next,
    PolyType::MultiMillerLoopTXC1Next,
    PolyType::MultiMillerLoopTYC0Next,
    PolyType::MultiMillerLoopTYC1Next,
    PolyType::MultiMillerLoopLambdaC0,
    PolyType::MultiMillerLoopLambdaC1,
    PolyType::MultiMillerLoopInvDeltaXC0,
    PolyType::MultiMillerLoopInvDeltaXC1,
    PolyType::MultiMillerLoopXP,
    PolyType::MultiMillerLoopYP,
    PolyType::MultiMillerLoopXQC0,
    PolyType::MultiMillerLoopXQC1,
    PolyType::MultiMillerLoopYQC0,
    PolyType::MultiMillerLoopYQC1,
    PolyType::MultiMillerLoopIsDouble,
    PolyType::MultiMillerLoopIsAdd,
    PolyType::MultiMillerLoopLVal,
    PolyType::MultiMillerLoopInvTwoYC0,
    PolyType::MultiMillerLoopInvTwoYC1,
];

#[cfg(feature = "experimental-pairing-recursion")]
const MULTI_MILLER_LOOP_SPECS: [(PolyType, usize); 26] = [
    (PolyType::MultiMillerLoopF, 11),
    (PolyType::MultiMillerLoopFNext, 11),
    (PolyType::MultiMillerLoopQuotient, 11),
    (PolyType::MultiMillerLoopTXC0, 11),
    (PolyType::MultiMillerLoopTXC1, 11),
    (PolyType::MultiMillerLoopTYC0, 11),
    (PolyType::MultiMillerLoopTYC1, 11),
    (PolyType::MultiMillerLoopTXC0Next, 11),
    (PolyType::MultiMillerLoopTXC1Next, 11),
    (PolyType::MultiMillerLoopTYC0Next, 11),
    (PolyType::MultiMillerLoopTYC1Next, 11),
    (PolyType::MultiMillerLoopLambdaC0, 11),
    (PolyType::MultiMillerLoopLambdaC1, 11),
    (PolyType::MultiMillerLoopInvDeltaXC0, 11),
    (PolyType::MultiMillerLoopInvDeltaXC1, 11),
    (PolyType::MultiMillerLoopXP, 11),
    (PolyType::MultiMillerLoopYP, 11),
    (PolyType::MultiMillerLoopXQC0, 11),
    (PolyType::MultiMillerLoopXQC1, 11),
    (PolyType::MultiMillerLoopYQC0, 11),
    (PolyType::MultiMillerLoopYQC1, 11),
    (PolyType::MultiMillerLoopIsDouble, 11),
    (PolyType::MultiMillerLoopIsAdd, 11),
    (PolyType::MultiMillerLoopLVal, 11),
    (PolyType::MultiMillerLoopInvTwoYC0, 11),
    (PolyType::MultiMillerLoopInvTwoYC1, 11),
];

impl ConstraintType {
    pub fn committed_poly_types(&self) -> &'static [PolyType] {
        match self {
            ConstraintType::GtExp => &GT_EXP_POLYS,
            ConstraintType::GtMul => &GT_MUL_POLYS,
            ConstraintType::G1ScalarMul { .. } => &G1_SCALAR_MUL_POLYS,
            ConstraintType::G2ScalarMul { .. } => &G2_SCALAR_MUL_POLYS,
            ConstraintType::G1Add => &G1_ADD_POLYS,
            ConstraintType::G2Add => &G2_ADD_POLYS,
            #[cfg(feature = "experimental-pairing-recursion")]
            ConstraintType::MultiMillerLoop => &MULTI_MILLER_LOOP_POLYS,
        }
    }

    pub fn committed_poly_specs(&self) -> &'static [(PolyType, usize)] {
        match self {
            ConstraintType::GtExp => &GT_EXP_SPECS,
            ConstraintType::GtMul => &GT_MUL_SPECS,
            ConstraintType::G1ScalarMul { .. } => &G1_SCALAR_MUL_SPECS,
            ConstraintType::G2ScalarMul { .. } => &G2_SCALAR_MUL_SPECS,
            ConstraintType::G1Add => &G1_ADD_SPECS,
            ConstraintType::G2Add => &G2_ADD_SPECS,
            #[cfg(feature = "experimental-pairing-recursion")]
            ConstraintType::MultiMillerLoop => &MULTI_MILLER_LOOP_SPECS,
        }
    }
}

// -----------------------------------------------------------------------------
// Streaming recursion "plan output" types.
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RecursionMatrixShape {
    pub num_constraints: usize,
    pub num_constraints_padded: usize,
    pub num_constraint_vars: usize, // fixed 11 today (ambient x-domain)
    pub num_s_vars: usize,
    pub num_vars: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum ConstraintLocator {
    GtExp { local: usize },
    GtMul { local: usize },
    G1ScalarMul { local: usize },
    G2ScalarMul { local: usize },
    G1Add { local: usize },
    G2Add { local: usize },
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoop { local: usize },
}

#[derive(Clone, Debug)]
pub struct GtMulNativeRows {
    pub lhs: Vec<Fq>,
    pub rhs: Vec<Fq>,
    pub result: Vec<Fq>,
    pub quotient: Vec<Fq>,
}

#[derive(Clone, Debug)]
pub struct G1ScalarMulNative {
    pub base_point: (Fq, Fq),
    pub x_a: Vec<Fq>,
    pub y_a: Vec<Fq>,
    pub x_t: Vec<Fq>,
    pub y_t: Vec<Fq>,
    pub x_a_next: Vec<Fq>,
    pub y_a_next: Vec<Fq>,
    pub t_indicator: Vec<Fq>,
    pub a_indicator: Vec<Fq>,
}

#[derive(Clone, Debug)]
pub struct G2ScalarMulNative {
    pub base_point: (Fq2, Fq2),
    pub x_a_c0: Vec<Fq>,
    pub x_a_c1: Vec<Fq>,
    pub y_a_c0: Vec<Fq>,
    pub y_a_c1: Vec<Fq>,
    pub x_t_c0: Vec<Fq>,
    pub x_t_c1: Vec<Fq>,
    pub y_t_c0: Vec<Fq>,
    pub y_t_c1: Vec<Fq>,
    pub x_a_next_c0: Vec<Fq>,
    pub x_a_next_c1: Vec<Fq>,
    pub y_a_next_c0: Vec<Fq>,
    pub y_a_next_c1: Vec<Fq>,
    pub t_indicator: Vec<Fq>,
    pub a_indicator: Vec<Fq>,
}

#[derive(Clone, Copy, Debug)]
pub struct G1AddNative {
    pub x_p: Fq,
    pub y_p: Fq,
    pub ind_p: Fq,
    pub x_q: Fq,
    pub y_q: Fq,
    pub ind_q: Fq,
    pub x_r: Fq,
    pub y_r: Fq,
    pub ind_r: Fq,
    pub lambda: Fq,
    pub inv_delta_x: Fq,
    pub is_double: Fq,
    pub is_inverse: Fq,
}

#[derive(Clone, Copy, Debug)]
pub struct G2AddNative {
    pub x_p_c0: Fq,
    pub x_p_c1: Fq,
    pub y_p_c0: Fq,
    pub y_p_c1: Fq,
    pub ind_p: Fq,
    pub x_q_c0: Fq,
    pub x_q_c1: Fq,
    pub y_q_c0: Fq,
    pub y_q_c1: Fq,
    pub ind_q: Fq,
    pub x_r_c0: Fq,
    pub x_r_c1: Fq,
    pub y_r_c0: Fq,
    pub y_r_c1: Fq,
    pub ind_r: Fq,
    pub lambda_c0: Fq,
    pub lambda_c1: Fq,
    pub inv_delta_x_c0: Fq,
    pub inv_delta_x_c1: Fq,
    pub is_double: Fq,
    pub is_inverse: Fq,
}

#[derive(Clone)]
pub struct ConstraintSystem {
    pub shape: RecursionMatrixShape,
    pub constraint_types: Vec<ConstraintType>,
    pub locator_by_constraint: Vec<ConstraintLocator>,

    pub g_poly: DensePolynomial<Fq>, // native 4-var (len 16)

    // Stage 1
    pub gt_exp_witnesses: Vec<GtExpWitness<Fq>>,
    pub gt_exp_public_inputs: Vec<GtExpPublicInputs>,

    // Stage 2 native stores
    pub gt_mul_rows: Vec<GtMulNativeRows>,          // each field len 16
    pub g1_scalar_mul_rows: Vec<G1ScalarMulNative>, // each field len 256
    pub g2_scalar_mul_rows: Vec<G2ScalarMulNative>, // each field len 256
    pub g1_add_rows: Vec<G1AddNative>,              // scalars
    pub g2_add_rows: Vec<G2AddNative>,              // scalars

    // Public inputs
    pub g1_scalar_mul_public_inputs: Vec<G1ScalarMulPublicInputs>,
    pub g2_scalar_mul_public_inputs: Vec<G2ScalarMulPublicInputs>,
}

impl ConstraintSystem {
    #[inline]
    pub fn num_constraints(&self) -> usize {
        self.shape.num_constraints
    }
    #[inline]
    pub fn num_constraints_padded(&self) -> usize {
        self.shape.num_constraints_padded
    }
    #[inline]
    pub fn num_constraint_vars(&self) -> usize {
        self.shape.num_constraint_vars
    }
    #[inline]
    pub fn num_s_vars(&self) -> usize {
        self.shape.num_s_vars
    }
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.shape.num_vars
    }
}

/// Builder for recursion metadata (internal; not part of proof payload).
#[derive(Clone)]
pub struct RecursionMetadataBuilder {
    constraint_system: ConstraintSystem,
}

impl RecursionMetadataBuilder {
    pub fn from_constraint_system(constraint_system: ConstraintSystem) -> Self {
        Self { constraint_system }
    }

    pub fn build(self) -> crate::zkvm::recursion::RecursionConstraintMetadata {
        let constraint_types = self.constraint_system.constraint_types.clone();
        let layout =
            crate::zkvm::recursion::prefix_packing::PrefixPackingLayout::from_constraint_types(
                &constraint_types,
            );
        let dense_num_vars = layout.num_dense_vars;
        crate::zkvm::recursion::RecursionConstraintMetadata {
            constraint_types,
            dense_num_vars,
            gt_exp_public_inputs: self.constraint_system.gt_exp_public_inputs.clone(),
            g1_scalar_mul_public_inputs: self.constraint_system.g1_scalar_mul_public_inputs.clone(),
            g2_scalar_mul_public_inputs: self.constraint_system.g2_scalar_mul_public_inputs.clone(),
        }
    }
}

// -----------------------------------------------------------------------------
// Serialization for ConstraintType (needed across host/guest boundaries).
// -----------------------------------------------------------------------------

impl CanonicalSerialize for ConstraintType {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            ConstraintType::GtExp => 0u8.serialize_with_mode(&mut writer, compress),
            ConstraintType::GtMul => 1u8.serialize_with_mode(&mut writer, compress),
            ConstraintType::G1ScalarMul { base_point } => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                base_point.0.serialize_with_mode(&mut writer, compress)?;
                base_point.1.serialize_with_mode(&mut writer, compress)?;
                Ok(())
            }
            ConstraintType::G2ScalarMul { base_point } => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                base_point.0.serialize_with_mode(&mut writer, compress)?;
                base_point.1.serialize_with_mode(&mut writer, compress)?;
                Ok(())
            }
            ConstraintType::G1Add => 4u8.serialize_with_mode(&mut writer, compress),
            ConstraintType::G2Add => 5u8.serialize_with_mode(&mut writer, compress),
            #[cfg(feature = "experimental-pairing-recursion")]
            ConstraintType::MultiMillerLoop => 6u8.serialize_with_mode(&mut writer, compress),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            ConstraintType::GtExp => 1,
            ConstraintType::GtMul => 1,
            ConstraintType::G1ScalarMul { base_point } => {
                1 + base_point.0.serialized_size(compress) + base_point.1.serialized_size(compress)
            }
            ConstraintType::G2ScalarMul { base_point } => {
                1 + base_point.0.serialized_size(compress) + base_point.1.serialized_size(compress)
            }
            ConstraintType::G1Add => 1,
            ConstraintType::G2Add => 1,
            #[cfg(feature = "experimental-pairing-recursion")]
            ConstraintType::MultiMillerLoop => 1,
        }
    }
}

impl CanonicalDeserialize for ConstraintType {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        compress: Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        let variant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match variant {
            0 => Ok(ConstraintType::GtExp),
            1 => Ok(ConstraintType::GtMul),
            2 => {
                let x = Fq::deserialize_with_mode(&mut reader, compress, validate)?;
                let y = Fq::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(ConstraintType::G1ScalarMul { base_point: (x, y) })
            }
            3 => {
                let x = Fq2::deserialize_with_mode(&mut reader, compress, validate)?;
                let y = Fq2::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(ConstraintType::G2ScalarMul { base_point: (x, y) })
            }
            4 => Ok(ConstraintType::G1Add),
            5 => Ok(ConstraintType::G2Add),
            #[cfg(feature = "experimental-pairing-recursion")]
            6 => Ok(ConstraintType::MultiMillerLoop),
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl GuestSerialize for ConstraintType {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        match self {
            ConstraintType::GtExp => 0u8.guest_serialize(w),
            ConstraintType::GtMul => 1u8.guest_serialize(w),
            ConstraintType::G1ScalarMul { base_point } => {
                2u8.guest_serialize(w)?;
                base_point.0.guest_serialize(w)?;
                base_point.1.guest_serialize(w)?;
                Ok(())
            }
            ConstraintType::G2ScalarMul { base_point } => {
                3u8.guest_serialize(w)?;
                base_point.0.guest_serialize(w)?;
                base_point.1.guest_serialize(w)?;
                Ok(())
            }
            ConstraintType::G1Add => 4u8.guest_serialize(w),
            ConstraintType::G2Add => 5u8.guest_serialize(w),
            #[cfg(feature = "experimental-pairing-recursion")]
            ConstraintType::MultiMillerLoop => 6u8.guest_serialize(w),
        }
    }
}

impl GuestDeserialize for ConstraintType {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        match u8::guest_deserialize(r)? {
            0 => Ok(ConstraintType::GtExp),
            1 => Ok(ConstraintType::GtMul),
            2 => Ok(ConstraintType::G1ScalarMul {
                base_point: (Fq::guest_deserialize(r)?, Fq::guest_deserialize(r)?),
            }),
            3 => Ok(ConstraintType::G2ScalarMul {
                base_point: (Fq2::guest_deserialize(r)?, Fq2::guest_deserialize(r)?),
            }),
            4 => Ok(ConstraintType::G1Add),
            5 => Ok(ConstraintType::G2Add),
            #[cfg(feature = "experimental-pairing-recursion")]
            6 => Ok(ConstraintType::MultiMillerLoop),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid ConstraintType",
            )),
        }
    }
}

impl Valid for ConstraintType {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            ConstraintType::G1ScalarMul { base_point } => {
                base_point.0.check()?;
                base_point.1.check()?;
            }
            ConstraintType::G2ScalarMul { base_point } => {
                base_point.0.check()?;
                base_point.1.check()?;
            }
            _ => {}
        }
        Ok(())
    }
}

