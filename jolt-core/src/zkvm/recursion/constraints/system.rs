//! Batches constraints into a single indexed polynomial F(z, x) = Σ_i eq(z, i) * C_i(x)

use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::RecursionExt,
            dory::{
                recursion::JoltGtMulWitness, ArkDoryProof, ArkworksVerifierSetup,
                DoryCommitmentScheme,
            },
        },
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
    zkvm::recursion::{
        g1::scalar_multiplication::G1ScalarMulPublicInputs,
        g2::scalar_multiplication::G2ScalarMulPublicInputs,
        gt::exponentiation::GtExpPublicInputs,
        witness::{GTCombineWitness, GTMulOpWitness},
    },
};
use ark_bn254::{Fq, Fq2, Fr};
use ark_ff::{One, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::backends::arkworks::ArkGT;
use jolt_optimizations::{fq12_to_multilinear_evals, get_g_mle};
use rayon::prelude::*;

/// Convert index to binary representation as field elements (little-endian)
pub fn index_to_binary<F: JoltField>(index: usize, num_vars: usize) -> Vec<F> {
    let mut binary = Vec::with_capacity(num_vars);
    let mut idx = index;

    for _ in 0..num_vars {
        binary.push(if idx & 1 == 1 { F::one() } else { F::zero() });
        idx >>= 1;
    }

    // binary.reverse();
    binary
}

/// Builder for RecursionConstraintMetadata that encapsulates all metadata extraction
#[derive(Clone)]
pub struct RecursionMetadataBuilder {
    constraint_system: ConstraintSystem,
}

impl RecursionMetadataBuilder {
    /// Create a new builder from a constraint system
    pub fn from_constraint_system(constraint_system: ConstraintSystem) -> Self {
        Self { constraint_system }
    }

    /// Build the metadata, extracting all necessary information
    pub fn build(self) -> crate::zkvm::proof_serialization::RecursionConstraintMetadata {
        // Extract constraint types
        let constraint_types: Vec<_> = self
            .constraint_system
            .constraints
            .iter()
            .map(|c| c.constraint_type.clone())
            .collect();

        // Build dense polynomial and get bijection info
        let (dense_poly, jagged_bijection, jagged_mapping) =
            self.constraint_system.build_dense_polynomial();
        let dense_num_vars = dense_poly.get_num_vars();

        // Compute matrix rows for the verifier
        let num_polynomials = jagged_bijection.num_polynomials();
        let mut matrix_rows = Vec::with_capacity(num_polynomials);

        for poly_idx in 0..num_polynomials {
            let (constraint_idx, poly_type) = jagged_mapping.decode(poly_idx);
            let matrix_row = self
                .constraint_system
                .matrix
                .row_index(poly_type, constraint_idx);
            matrix_rows.push(matrix_row);
        }

        crate::zkvm::proof_serialization::RecursionConstraintMetadata {
            constraint_types,
            jagged_bijection,
            jagged_mapping,
            matrix_rows,
            dense_num_vars,
            gt_exp_public_inputs: self.constraint_system.gt_exp_public_inputs.clone(),
            g1_scalar_mul_public_inputs: self.constraint_system.g1_scalar_mul_public_inputs.clone(),
            g2_scalar_mul_public_inputs: self.constraint_system.g2_scalar_mul_public_inputs.clone(),
        }
    }
}

/// Compute constraint formula: ρ_curr - ρ_prev² × base^{b} - quotient × g
pub fn compute_constraint_formula(
    rho_curr: Fq,
    rho_prev: Fq,
    base: Fq,
    quotient: Fq,
    g_val: Fq,
    bit: bool,
) -> Fq {
    let base_power = if bit { base } else { Fq::one() };
    rho_curr - rho_prev.square() * base_power - quotient * g_val
}

/// Polynomial types stored in the recursion constraint matrix.
///
/// Notes:
/// - Packed GT exp **commits only** `RhoPrev` and `Quotient`.
/// - `rho_next` is verified separately via a **shift sumcheck**.
/// - `base(x)` and the exponent digits/bits are **public inputs** (not committed matrix rows).
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

    // G1 Scalar Multiplication polynomials (8-var padded to 11-var in the matrix)
    G1ScalarMulXA = 6,
    G1ScalarMulYA = 7,
    G1ScalarMulXT = 8,
    G1ScalarMulYT = 9,
    G1ScalarMulXANext = 10,
    G1ScalarMulYANext = 11,
    G1ScalarMulTIndicator = 12,
    G1ScalarMulAIndicator = 13,
    G1ScalarMulBit = 14,

    // G2 Scalar Multiplication polynomials (Fq2 coords split into c0/c1; 8-var padded to 11-var)
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
    G2ScalarMulBit = 29,

    // G1 Addition polynomials (11-var)
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

    // G2 Addition polynomials (Fq2 coords split into c0/c1; 11-var)
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

    // ---------------------------------------------------------------------
    // Pairing recursion (experimental): Multi-Miller loop packed traces
    // ---------------------------------------------------------------------
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
    /// Witnessed inverse of 2*y_T for doubling rows in the Multi-Miller loop trace (Fq2.c0).
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopInvTwoYC0 = 88,
    /// Witnessed inverse of 2*y_T for doubling rows in the Multi-Miller loop trace (Fq2.c1).
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoopInvTwoYC1 = 89,
}

#[cfg(not(feature = "experimental-pairing-recursion"))]
const ALL_POLY_TYPES: [PolyType; 64] = PolyType::all_base_64();
#[cfg(feature = "experimental-pairing-recursion")]
const ALL_POLY_TYPES: [PolyType; 90] = PolyType::all_with_pairing_90();

impl PolyType {
    #[cfg(not(feature = "experimental-pairing-recursion"))]
    pub const NUM_TYPES: usize = 64;
    #[cfg(feature = "experimental-pairing-recursion")]
    pub const NUM_TYPES: usize = 90;

    pub fn all() -> &'static [PolyType] {
        &ALL_POLY_TYPES
    }

    #[cfg(not(feature = "experimental-pairing-recursion"))]
    const fn all_base_64() -> [PolyType; 64] {
        [
            PolyType::RhoPrev,
            PolyType::Quotient,
            PolyType::MulLhs,
            PolyType::MulRhs,
            PolyType::MulResult,
            PolyType::MulQuotient,
            PolyType::G1ScalarMulXA,
            PolyType::G1ScalarMulYA,
            PolyType::G1ScalarMulXT,
            PolyType::G1ScalarMulYT,
            PolyType::G1ScalarMulXANext,
            PolyType::G1ScalarMulYANext,
            PolyType::G1ScalarMulTIndicator,
            PolyType::G1ScalarMulAIndicator,
            PolyType::G1ScalarMulBit,
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
            PolyType::G2ScalarMulBit,
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
        ]
    }

    #[cfg(feature = "experimental-pairing-recursion")]
    const fn all_with_pairing_90() -> [PolyType; 90] {
        [
            // Base 64
            PolyType::RhoPrev,
            PolyType::Quotient,
            PolyType::MulLhs,
            PolyType::MulRhs,
            PolyType::MulResult,
            PolyType::MulQuotient,
            PolyType::G1ScalarMulXA,
            PolyType::G1ScalarMulYA,
            PolyType::G1ScalarMulXT,
            PolyType::G1ScalarMulYT,
            PolyType::G1ScalarMulXANext,
            PolyType::G1ScalarMulYANext,
            PolyType::G1ScalarMulTIndicator,
            PolyType::G1ScalarMulAIndicator,
            PolyType::G1ScalarMulBit,
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
            PolyType::G2ScalarMulBit,
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
            // Pairing extension (24)
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
        ]
    }

    /// Get polynomial type from row index
    pub fn from_row_index(row_idx: usize, num_constraints: usize) -> Self {
        match row_idx / num_constraints {
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
            _ => panic!("Invalid row index"),
        }
    }
}

/// Giant multilinear matrix M(s, x) that stores all Dory polynomials in a single structure.
///
/// Layout: M(s, x) where s is the row index and x are the constraint variables
/// Physical layout: rows are organized as [all base] [all rho_prev] [all rho_curr] [all quotient]
/// Row index = poly_type * num_constraints_padded + constraint_index
#[derive(Clone)]
pub struct DoryMultilinearMatrix {
    /// Number of s variables (log2(num_rows))
    pub num_s_vars: usize,

    /// Number of constraint variables (x) - fixed at 4 for Fq12
    pub num_constraint_vars: usize,

    /// Number of constraint index variables (bits needed to index constraints)
    pub num_constraint_index_vars: usize,

    /// Number of constraints (before padding)
    pub num_constraints: usize,

    /// Number of constraints padded to power of 2
    pub num_constraints_padded: usize,

    /// Total number of rows: 4 * num_constraints_padded
    pub num_rows: usize,

    /// Total M variables: num_s_vars + num_constraint_vars
    pub num_vars: usize,

    /// Flattened storage: rows concatenated together
    /// Each row contains 2^num_constraint_vars evaluations
    /// Total size: num_rows * 2^num_constraint_vars
    pub evaluations: Vec<Fq>,
}

impl DoryMultilinearMatrix {
    /// Get row index for a given polynomial type and constraint index
    pub fn row_index(&self, poly_type: PolyType, constraint_idx: usize) -> usize {
        (poly_type as usize) * self.num_constraints_padded + constraint_idx
    }

    /// Get the storage offset for accessing a specific row's polynomial
    pub fn storage_offset(&self, row_index: usize) -> usize {
        row_index * (1 << self.num_constraint_vars)
    }

    /// Evaluate a specific row's polynomial at point x
    /// Note: constraint_vars is in little-endian (LSB first), but DensePolynomial::evaluate
    /// uses big-endian (MSB first), so we reverse the point.
    pub fn evaluate_row(&self, row: usize, constraint_vars: &[Fq]) -> Fq {
        let offset = self.storage_offset(row);
        let row_evals = &self.evaluations[offset..offset + (1 << self.num_constraint_vars)];

        let poly = DensePolynomial::new(row_evals.to_vec());
        // Reverse for big-endian convention used by DensePolynomial::evaluate
        let reversed: Vec<Fq> = constraint_vars.iter().rev().copied().collect();
        poly.evaluate(&reversed)
    }

    /// Evaluate M(s, x) where s selects the row and x is the evaluation point
    pub fn evaluate(&self, s_vars: &[Fq], constraint_vars: &[Fq]) -> Fq {
        assert_eq!(s_vars.len(), self.num_s_vars);
        assert_eq!(constraint_vars.len(), self.num_constraint_vars);

        let mut result = Fq::zero();
        for row in 0..self.num_rows {
            let row_binary = index_to_binary::<Fq>(row, self.num_s_vars);
            let eq_eval = EqPolynomial::mle(&row_binary, s_vars);

            let row_poly_eval = self.evaluate_row(row, constraint_vars);
            result += eq_eval * row_poly_eval;
        }
        result
    }
}

/// Builder for constructing the giant multilinear matrix.
///
/// Physical layout: rows are organized by polynomial type
/// Row index = poly_type * num_constraints_padded + constraint_index
pub struct DoryMatrixBuilder {
    num_constraint_vars: usize,
    /// Rows grouped by polynomial type (`PolyType::NUM_TYPES` types total)
    rows_by_type: [Vec<Vec<Fq>>; PolyType::NUM_TYPES],
    /// Constraint types for each constraint
    constraint_types: Vec<ConstraintType>,
}

impl DoryMatrixBuilder {
    pub fn new(num_constraint_vars: usize) -> Self {
        Self {
            num_constraint_vars,
            rows_by_type: std::array::from_fn(|_| Vec::new()),
            constraint_types: Vec::new(),
        }
    }

    /// Get the current number of constraints added to the builder
    pub fn constraint_count(&self) -> usize {
        self.constraint_types.len()
    }

    /// For a newly-added constraint, push a zero row for every polynomial type *except* `except`.
    ///
    /// This keeps the matrix layout consistent across heterogeneous constraints:
    /// for each constraint index, every `PolyType` has exactly one row (possibly all zeros).
    fn push_zero_rows_except(&mut self, row_size: usize, except: &[PolyType]) {
        let zero_row = vec![Fq::zero(); row_size];
        for &poly_type in PolyType::all() {
            if except.contains(&poly_type) {
                continue;
            }
            self.rows_by_type[poly_type as usize].push(zero_row.clone());
        }
    }

    /// Pad a 4-variable MLE to 8 variables by repeating the values.
    /// For a 4-var MLE f(x0,x1,x2,x3), the 8-var version is f(x0,x1,x2,x3,0,0,0,0).
    /// This means we repeat each 4-var evaluation 2^4 = 16 times.
    pub fn pad_4var_to_8var(mle_4var: &[Fq]) -> Vec<Fq> {
        assert_eq!(mle_4var.len(), 16, "Input must be a 4-variable MLE");
        let mut mle_8var = Vec::with_capacity(256);

        // For each 4-var evaluation, repeat it 16 times
        // This corresponds to all possible values of the last 4 variables
        for &val in mle_4var {
            for _ in 0..16 {
                mle_8var.push(val);
            }
        }

        assert_eq!(mle_8var.len(), 256);
        mle_8var
    }

    /// Pad a 4-variable MLE to 8 variables using zero padding for true jaggedness.
    pub fn pad_4var_to_8var_zero_padding<Fq: JoltField>(mle_4var: &[Fq]) -> Vec<Fq> {
        assert_eq!(mle_4var.len(), 16, "Input must be a 4-variable MLE");
        let mut mle_8var = Vec::with_capacity(256);

        // Copy original 16 values at the beginning
        mle_8var.extend_from_slice(mle_4var);

        // Pad with zeros for the remaining positions
        // This creates true jaggedness - non-zero values only at the start
        mle_8var.resize(256, Fq::zero());

        assert_eq!(mle_8var.len(), 256);
        mle_8var
    }

    /// Pad a 4-variable MLE to 11 variables using zero padding for true jaggedness.
    /// For GT mul: index = s * 16 + x (x in low bits).
    pub fn pad_4var_to_11var_zero_padding<Fq: JoltField>(mle_4var: &[Fq]) -> Vec<Fq> {
        assert_eq!(mle_4var.len(), 16, "Input must be a 4-variable MLE");
        let mut mle_11var = Vec::with_capacity(2048);

        // Copy original 16 values at the beginning
        mle_11var.extend_from_slice(mle_4var);

        // Pad with zeros for the remaining positions
        mle_11var.resize(2048, Fq::zero());

        assert_eq!(mle_11var.len(), 2048);
        mle_11var
    }

    /// Pad a 4-variable MLE to 11 variables for packed GT exp layout.
    /// Data layout: index = x * 128 + s (s in low 7 bits, x in high 4 bits).
    /// g(x) doesn't depend on s, so we replicate each g[x] across all 128 s values.
    pub fn pad_4var_to_11var_replicated<Fq: JoltField>(mle_4var: &[Fq]) -> Vec<Fq> {
        assert_eq!(mle_4var.len(), 16, "Input must be a 4-variable MLE");
        let mut mle_11var = vec![Fq::zero(); 2048];

        // For each x value (0-15), replicate g[x] across all s values (0-127)
        // index = x * 128 + s
        for x in 0..16 {
            let g_x = mle_4var[x];
            for s in 0..128 {
                mle_11var[x * 128 + s] = g_x;
            }
        }

        mle_11var
    }

    /// Pad an 8-variable MLE to 11 variables using zero padding.
    /// For G1 scalar mul: 8 vars → 11 vars.
    pub fn pad_8var_to_11var_zero_padding(mle_8var: &[Fq]) -> Vec<Fq> {
        assert_eq!(mle_8var.len(), 256, "Input must be an 8-variable MLE");
        let mut mle_11var = Vec::with_capacity(2048);

        // Copy original 256 values at the beginning
        mle_11var.extend_from_slice(mle_8var);

        // Pad with zeros for the remaining positions
        mle_11var.resize(2048, Fq::zero());

        assert_eq!(mle_11var.len(), 2048);
        mle_11var
    }

    /// Add a packed GT exponentiation witness.
    /// Creates ONE constraint per GT exp with packed polynomials (all 11-var):
    /// - Base: base(x) - 4-var padded to 11-var (public input, not committed)
    /// - RhoPrev: rho(s,x) - all intermediate results packed
    /// - RhoNext: rho_next(s,x) - shifted intermediates (NOT COMMITTED - verified via shift sumcheck)
    /// - Quotient: quotient(s,x) - all quotients packed
    /// - Digit bits: digit_lo/hi(s) - scalar digits (7-var padded to 11-var, public input)
    pub fn add_gt_exp_witness(
        &mut self,
        witness: &crate::zkvm::recursion::gt::exponentiation::GtExpWitness<Fq>,
    ) {
        assert_eq!(
            self.num_constraint_vars, 11,
            "Packed GT exp requires 11 constraint variables"
        );

        let row_size = 1 << self.num_constraint_vars; // 2048

        // All packed polynomials are already 11-var (2048 elements)
        assert_eq!(witness.base_packed.len(), row_size);
        assert_eq!(witness.rho_packed.len(), row_size);
        assert_eq!(witness.rho_next_packed.len(), row_size);
        assert_eq!(witness.quotient_packed.len(), row_size);
        assert_eq!(witness.digit_lo_packed.len(), row_size);
        assert_eq!(witness.digit_hi_packed.len(), row_size);
        assert_eq!(witness.base2_packed.len(), row_size);
        assert_eq!(witness.base3_packed.len(), row_size);

        // Add only the 2 committed GT exp polynomials (base/digits/rho_next are not committed)
        self.rows_by_type[PolyType::RhoPrev as usize].push(witness.rho_packed.clone());
        self.rows_by_type[PolyType::Quotient as usize].push(witness.quotient_packed.clone());

        // Add empty rows for all other polynomial types to maintain consistent indexing
        self.push_zero_rows_except(row_size, &GT_EXP_POLYS);

        // Store ONE constraint entry for this packed GT exp
        self.constraint_types.push(ConstraintType::GtExp);
    }

    /// Add constraint from a GT multiplication witness.
    /// Creates one constraint using:
    /// - lhs: the left operand a
    /// - rhs: the right operand b
    /// - result: the product c = a * b
    /// - quotient: Q such that a(x) * b(x) - c(x) = Q(x) * g(x)
    pub fn add_gt_mul_witness(&mut self, witness: &JoltGtMulWitness) {
        let lhs_mle_4var = fq12_to_multilinear_evals(&witness.lhs);
        let rhs_mle_4var = fq12_to_multilinear_evals(&witness.rhs);
        let result_mle_4var = fq12_to_multilinear_evals(&witness.result);
        let quotient_mle_4var = witness.quotient_mle.clone();

        assert_eq!(
            lhs_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );
        assert_eq!(
            rhs_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );
        assert_eq!(
            result_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );
        assert_eq!(
            quotient_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );

        // Handle padding from 4-var to target vars
        let (lhs_mle, rhs_mle, result_mle, quotient_mle) = if self.num_constraint_vars == 11 {
            // Pad 4-variable MLEs to 11 variables using zero padding
            (
                Self::pad_4var_to_11var_zero_padding(&lhs_mle_4var),
                Self::pad_4var_to_11var_zero_padding(&rhs_mle_4var),
                Self::pad_4var_to_11var_zero_padding(&result_mle_4var),
                Self::pad_4var_to_11var_zero_padding(&quotient_mle_4var),
            )
        } else if self.num_constraint_vars == 8 {
            // Pad 4-variable MLEs to 8 variables using zero padding
            (
                Self::pad_4var_to_8var_zero_padding(&lhs_mle_4var),
                Self::pad_4var_to_8var_zero_padding(&rhs_mle_4var),
                Self::pad_4var_to_8var_zero_padding(&result_mle_4var),
                Self::pad_4var_to_8var_zero_padding(&quotient_mle_4var),
            )
        } else if self.num_constraint_vars == 4 {
            // Use MLEs as-is
            (
                lhs_mle_4var,
                rhs_mle_4var,
                result_mle_4var,
                quotient_mle_4var,
            )
        } else {
            panic!(
                "Unsupported number of constraint variables: {}",
                self.num_constraint_vars
            );
        };

        assert_eq!(lhs_mle.len(), 1 << self.num_constraint_vars);
        assert_eq!(rhs_mle.len(), 1 << self.num_constraint_vars);
        assert_eq!(result_mle.len(), 1 << self.num_constraint_vars);
        assert_eq!(quotient_mle.len(), 1 << self.num_constraint_vars);

        // Add rows for GT mul polynomials (keeping GT exp rows empty)
        self.rows_by_type[PolyType::MulLhs as usize].push(lhs_mle);
        self.rows_by_type[PolyType::MulRhs as usize].push(rhs_mle);
        self.rows_by_type[PolyType::MulResult as usize].push(result_mle);
        self.rows_by_type[PolyType::MulQuotient as usize].push(quotient_mle);

        // Add empty rows for all other polynomial types to maintain consistent indexing
        self.push_zero_rows_except(1 << self.num_constraint_vars, &GT_MUL_POLYS);

        // Store constraint type
        self.constraint_types.push(ConstraintType::GtMul);
    }

    /// Add constraints from a G1 scalar multiplication witness.
    /// Unlike GT exp which has one MLE per step, G1 has one MLE per variable type
    /// that contains all steps.
    pub fn add_g1_scalar_mul_witness(
        &mut self,
        witness: &crate::poly::commitment::dory::recursion::JoltG1ScalarMulWitness,
    ) {
        let _n = witness.bits.len();

        // The witness MLEs contain all steps in one MLE
        assert_eq!(witness.x_a_mles.len(), 1, "Expected single MLE for x_a");
        assert_eq!(witness.y_a_mles.len(), 1, "Expected single MLE for y_a");
        assert_eq!(witness.x_t_mles.len(), 1, "Expected single MLE for x_t");
        assert_eq!(witness.y_t_mles.len(), 1, "Expected single MLE for y_t");
        assert_eq!(
            witness.x_a_next_mles.len(),
            1,
            "Expected single MLE for x_a_next"
        );
        assert_eq!(
            witness.y_a_next_mles.len(),
            1,
            "Expected single MLE for y_a_next"
        );
        assert_eq!(
            witness.t_is_infinity_mles.len(),
            1,
            "Expected single MLE for t_is_infinity"
        );
        assert_eq!(
            witness.a_is_infinity_mles.len(),
            1,
            "Expected single MLE for a_is_infinity"
        );

        // Each MLE should have 256 evaluations for 8 variables
        assert_eq!(
            witness.x_a_mles[0].len(),
            1 << 8,
            "Expected 256 evaluations for 8 variables"
        );
        assert_eq!(witness.y_a_mles[0].len(), 1 << 8);
        assert_eq!(witness.x_t_mles[0].len(), 1 << 8);
        assert_eq!(witness.y_t_mles[0].len(), 1 << 8);
        assert_eq!(witness.x_a_next_mles[0].len(), 1 << 8);
        assert_eq!(witness.y_a_next_mles[0].len(), 1 << 8);
        assert_eq!(witness.t_is_infinity_mles[0].len(), 1 << 8);
        assert_eq!(witness.a_is_infinity_mles[0].len(), 1 << 8);

        // Pad 8-var MLEs to target constraint vars if needed
        let (x_a, y_a, x_t, y_t, x_a_next, y_a_next, t_indicator, a_indicator) = if self
            .num_constraint_vars
            == 11
        {
            (
                Self::pad_8var_to_11var_zero_padding(&witness.x_a_mles[0]),
                Self::pad_8var_to_11var_zero_padding(&witness.y_a_mles[0]),
                Self::pad_8var_to_11var_zero_padding(&witness.x_t_mles[0]),
                Self::pad_8var_to_11var_zero_padding(&witness.y_t_mles[0]),
                Self::pad_8var_to_11var_zero_padding(&witness.x_a_next_mles[0]),
                Self::pad_8var_to_11var_zero_padding(&witness.y_a_next_mles[0]),
                Self::pad_8var_to_11var_zero_padding(&witness.t_is_infinity_mles[0]),
                Self::pad_8var_to_11var_zero_padding(&witness.a_is_infinity_mles[0]),
            )
        } else if self.num_constraint_vars == 8 {
            (
                witness.x_a_mles[0].clone(),
                witness.y_a_mles[0].clone(),
                witness.x_t_mles[0].clone(),
                witness.y_t_mles[0].clone(),
                witness.x_a_next_mles[0].clone(),
                witness.y_a_next_mles[0].clone(),
                witness.t_is_infinity_mles[0].clone(),
                witness.a_is_infinity_mles[0].clone(),
            )
        } else {
            panic!(
                "G1 scalar multiplication requires 8 or 11 constraint variables, but builder has {}",
                self.num_constraint_vars
            );
        };

        // Add the entire MLEs (one per variable type for this scalar multiplication)
        self.rows_by_type[PolyType::G1ScalarMulXA as usize].push(x_a);
        self.rows_by_type[PolyType::G1ScalarMulYA as usize].push(y_a);
        self.rows_by_type[PolyType::G1ScalarMulXT as usize].push(x_t);
        self.rows_by_type[PolyType::G1ScalarMulYT as usize].push(y_t);
        self.rows_by_type[PolyType::G1ScalarMulXANext as usize].push(x_a_next);
        self.rows_by_type[PolyType::G1ScalarMulYANext as usize].push(y_a_next);
        self.rows_by_type[PolyType::G1ScalarMulTIndicator as usize].push(t_indicator);
        self.rows_by_type[PolyType::G1ScalarMulAIndicator as usize].push(a_indicator);

        // Add empty rows for all other polynomial types to maintain consistent indexing
        self.push_zero_rows_except(1 << self.num_constraint_vars, &G1_SCALAR_MUL_POLYS);

        // Store one constraint entry for this G1 scalar multiplication
        // The constraint evaluation will handle accessing different indices within the MLEs
        self.constraint_types.push(ConstraintType::G1ScalarMul {
            base_point: (witness.point_base.x, witness.point_base.y),
        });
    }

    /// Add constraints from a G2 scalar multiplication witness.
    ///
    /// G2 coordinates are in Fq2, so each coordinate is provided as two 8-var MLEs (c0/c1).
    pub fn add_g2_scalar_mul_witness(
        &mut self,
        witness: &crate::poly::commitment::dory::recursion::JoltG2ScalarMulWitness,
    ) {
        // Expect a single 8-var MLE per coordinate component / indicator.
        assert_eq!(
            witness.x_a_c0_mles.len(),
            1,
            "Expected single MLE for x_a.c0"
        );
        assert_eq!(
            witness.x_a_c1_mles.len(),
            1,
            "Expected single MLE for x_a.c1"
        );
        assert_eq!(
            witness.y_a_c0_mles.len(),
            1,
            "Expected single MLE for y_a.c0"
        );
        assert_eq!(
            witness.y_a_c1_mles.len(),
            1,
            "Expected single MLE for y_a.c1"
        );
        assert_eq!(
            witness.x_t_c0_mles.len(),
            1,
            "Expected single MLE for x_t.c0"
        );
        assert_eq!(
            witness.x_t_c1_mles.len(),
            1,
            "Expected single MLE for x_t.c1"
        );
        assert_eq!(
            witness.y_t_c0_mles.len(),
            1,
            "Expected single MLE for y_t.c0"
        );
        assert_eq!(
            witness.y_t_c1_mles.len(),
            1,
            "Expected single MLE for y_t.c1"
        );
        assert_eq!(
            witness.x_a_next_c0_mles.len(),
            1,
            "Expected single MLE for x_a_next.c0"
        );
        assert_eq!(
            witness.x_a_next_c1_mles.len(),
            1,
            "Expected single MLE for x_a_next.c1"
        );
        assert_eq!(
            witness.y_a_next_c0_mles.len(),
            1,
            "Expected single MLE for y_a_next.c0"
        );
        assert_eq!(
            witness.y_a_next_c1_mles.len(),
            1,
            "Expected single MLE for y_a_next.c1"
        );
        assert_eq!(
            witness.t_is_infinity_mles.len(),
            1,
            "Expected single MLE for t_is_infinity"
        );
        assert_eq!(
            witness.a_is_infinity_mles.len(),
            1,
            "Expected single MLE for a_is_infinity"
        );

        // Each MLE should have 256 evaluations for 8 variables
        let expected_len = 1 << 8;
        assert_eq!(witness.x_a_c0_mles[0].len(), expected_len);
        assert_eq!(witness.x_a_c1_mles[0].len(), expected_len);
        assert_eq!(witness.y_a_c0_mles[0].len(), expected_len);
        assert_eq!(witness.y_a_c1_mles[0].len(), expected_len);
        assert_eq!(witness.x_t_c0_mles[0].len(), expected_len);
        assert_eq!(witness.x_t_c1_mles[0].len(), expected_len);
        assert_eq!(witness.y_t_c0_mles[0].len(), expected_len);
        assert_eq!(witness.y_t_c1_mles[0].len(), expected_len);
        assert_eq!(witness.x_a_next_c0_mles[0].len(), expected_len);
        assert_eq!(witness.x_a_next_c1_mles[0].len(), expected_len);
        assert_eq!(witness.y_a_next_c0_mles[0].len(), expected_len);
        assert_eq!(witness.y_a_next_c1_mles[0].len(), expected_len);
        assert_eq!(witness.t_is_infinity_mles[0].len(), expected_len);
        assert_eq!(witness.a_is_infinity_mles[0].len(), expected_len);

        // Pad 8-var MLEs to target constraint vars if needed
        let pad = |mle_8: &Vec<Fq>| -> Vec<Fq> {
            if self.num_constraint_vars == 11 {
                Self::pad_8var_to_11var_zero_padding(mle_8)
            } else if self.num_constraint_vars == 8 {
                mle_8.clone()
            } else {
                panic!(
                    "G2 scalar multiplication requires 8 or 11 constraint variables, but builder has {}",
                    self.num_constraint_vars
                );
            }
        };

        let x_a_c0 = pad(&witness.x_a_c0_mles[0]);
        let x_a_c1 = pad(&witness.x_a_c1_mles[0]);
        let y_a_c0 = pad(&witness.y_a_c0_mles[0]);
        let y_a_c1 = pad(&witness.y_a_c1_mles[0]);
        let x_t_c0 = pad(&witness.x_t_c0_mles[0]);
        let x_t_c1 = pad(&witness.x_t_c1_mles[0]);
        let y_t_c0 = pad(&witness.y_t_c0_mles[0]);
        let y_t_c1 = pad(&witness.y_t_c1_mles[0]);
        let x_a_next_c0 = pad(&witness.x_a_next_c0_mles[0]);
        let x_a_next_c1 = pad(&witness.x_a_next_c1_mles[0]);
        let y_a_next_c0 = pad(&witness.y_a_next_c0_mles[0]);
        let y_a_next_c1 = pad(&witness.y_a_next_c1_mles[0]);
        let t_indicator = pad(&witness.t_is_infinity_mles[0]);
        let a_indicator = pad(&witness.a_is_infinity_mles[0]);

        // Add the entire MLEs (one per variable type for this scalar multiplication)
        self.rows_by_type[PolyType::G2ScalarMulXAC0 as usize].push(x_a_c0);
        self.rows_by_type[PolyType::G2ScalarMulXAC1 as usize].push(x_a_c1);
        self.rows_by_type[PolyType::G2ScalarMulYAC0 as usize].push(y_a_c0);
        self.rows_by_type[PolyType::G2ScalarMulYAC1 as usize].push(y_a_c1);
        self.rows_by_type[PolyType::G2ScalarMulXTC0 as usize].push(x_t_c0);
        self.rows_by_type[PolyType::G2ScalarMulXTC1 as usize].push(x_t_c1);
        self.rows_by_type[PolyType::G2ScalarMulYTC0 as usize].push(y_t_c0);
        self.rows_by_type[PolyType::G2ScalarMulYTC1 as usize].push(y_t_c1);
        self.rows_by_type[PolyType::G2ScalarMulXANextC0 as usize].push(x_a_next_c0);
        self.rows_by_type[PolyType::G2ScalarMulXANextC1 as usize].push(x_a_next_c1);
        self.rows_by_type[PolyType::G2ScalarMulYANextC0 as usize].push(y_a_next_c0);
        self.rows_by_type[PolyType::G2ScalarMulYANextC1 as usize].push(y_a_next_c1);
        self.rows_by_type[PolyType::G2ScalarMulTIndicator as usize].push(t_indicator);
        self.rows_by_type[PolyType::G2ScalarMulAIndicator as usize].push(a_indicator);

        // Add empty rows for all other polynomial types to maintain consistent indexing
        self.push_zero_rows_except(1 << self.num_constraint_vars, &G2_SCALAR_MUL_POLYS);

        // Store one constraint entry for this G2 scalar multiplication
        self.constraint_types.push(ConstraintType::G2ScalarMul {
            base_point: (witness.point_base.x, witness.point_base.y),
        });
    }

    /// Add constraint from a single G1 addition witness (P + Q = R).
    ///
    /// The recursion matrix uses 11 constraint variables for a uniform layout, so each witness
    /// polynomial is an 11-var MLE (length 2^11). For G1Add, each polynomial is constant over
    /// the hypercube (the same value repeated).
    pub fn add_g1_add_witness(
        &mut self,
        witness: &crate::poly::commitment::dory::witness::g1_add::G1AdditionSteps,
    ) {
        assert_eq!(
            self.num_constraint_vars, 11,
            "G1Add expects 11 constraint vars (constant 11-var MLEs)"
        );
        let row_size = 1 << self.num_constraint_vars;
        let const_row = |v: Fq| vec![v; row_size];

        let x_p = const_row(witness.x_p);
        let y_p = const_row(witness.y_p);
        let ind_p = const_row(witness.ind_p);
        let x_q = const_row(witness.x_q);
        let y_q = const_row(witness.y_q);
        let ind_q = const_row(witness.ind_q);
        let x_r = const_row(witness.x_r);
        let y_r = const_row(witness.y_r);
        let ind_r = const_row(witness.ind_r);
        let lambda = const_row(witness.lambda);
        let inv_delta_x = const_row(witness.inv_delta_x);
        let is_double = const_row(witness.is_double);
        let is_inverse = const_row(witness.is_inverse);

        self.rows_by_type[PolyType::G1AddXP as usize].push(x_p);
        self.rows_by_type[PolyType::G1AddYP as usize].push(y_p);
        self.rows_by_type[PolyType::G1AddPIndicator as usize].push(ind_p);
        self.rows_by_type[PolyType::G1AddXQ as usize].push(x_q);
        self.rows_by_type[PolyType::G1AddYQ as usize].push(y_q);
        self.rows_by_type[PolyType::G1AddQIndicator as usize].push(ind_q);
        self.rows_by_type[PolyType::G1AddXR as usize].push(x_r);
        self.rows_by_type[PolyType::G1AddYR as usize].push(y_r);
        self.rows_by_type[PolyType::G1AddRIndicator as usize].push(ind_r);
        self.rows_by_type[PolyType::G1AddLambda as usize].push(lambda);
        self.rows_by_type[PolyType::G1AddInvDeltaX as usize].push(inv_delta_x);
        self.rows_by_type[PolyType::G1AddIsDouble as usize].push(is_double);
        self.rows_by_type[PolyType::G1AddIsInverse as usize].push(is_inverse);

        self.push_zero_rows_except(row_size, &G1_ADD_POLYS);

        self.constraint_types.push(ConstraintType::G1Add);
    }

    /// Add constraint from a single G2 addition witness (P + Q = R), with Fq2 coordinates split
    /// into (c0, c1) components in Fq.
    pub fn add_g2_add_witness(
        &mut self,
        witness: &crate::poly::commitment::dory::witness::g2_add::G2AdditionSteps,
    ) {
        assert_eq!(
            self.num_constraint_vars, 11,
            "G2Add expects 11 constraint vars (constant 11-var MLEs)"
        );
        let row_size = 1 << self.num_constraint_vars;
        let const_row = |v: Fq| vec![v; row_size];

        let x_p_c0 = const_row(witness.x_p_c0);
        let x_p_c1 = const_row(witness.x_p_c1);
        let y_p_c0 = const_row(witness.y_p_c0);
        let y_p_c1 = const_row(witness.y_p_c1);
        let ind_p = const_row(witness.ind_p);

        let x_q_c0 = const_row(witness.x_q_c0);
        let x_q_c1 = const_row(witness.x_q_c1);
        let y_q_c0 = const_row(witness.y_q_c0);
        let y_q_c1 = const_row(witness.y_q_c1);
        let ind_q = const_row(witness.ind_q);

        let x_r_c0 = const_row(witness.x_r_c0);
        let x_r_c1 = const_row(witness.x_r_c1);
        let y_r_c0 = const_row(witness.y_r_c0);
        let y_r_c1 = const_row(witness.y_r_c1);
        let ind_r = const_row(witness.ind_r);

        let lambda_c0 = const_row(witness.lambda_c0);
        let lambda_c1 = const_row(witness.lambda_c1);
        let inv_delta_x_c0 = const_row(witness.inv_delta_x_c0);
        let inv_delta_x_c1 = const_row(witness.inv_delta_x_c1);
        let is_double = const_row(witness.is_double);
        let is_inverse = const_row(witness.is_inverse);

        self.rows_by_type[PolyType::G2AddXPC0 as usize].push(x_p_c0);
        self.rows_by_type[PolyType::G2AddXPC1 as usize].push(x_p_c1);
        self.rows_by_type[PolyType::G2AddYPC0 as usize].push(y_p_c0);
        self.rows_by_type[PolyType::G2AddYPC1 as usize].push(y_p_c1);
        self.rows_by_type[PolyType::G2AddPIndicator as usize].push(ind_p);

        self.rows_by_type[PolyType::G2AddXQC0 as usize].push(x_q_c0);
        self.rows_by_type[PolyType::G2AddXQC1 as usize].push(x_q_c1);
        self.rows_by_type[PolyType::G2AddYQC0 as usize].push(y_q_c0);
        self.rows_by_type[PolyType::G2AddYQC1 as usize].push(y_q_c1);
        self.rows_by_type[PolyType::G2AddQIndicator as usize].push(ind_q);

        self.rows_by_type[PolyType::G2AddXRC0 as usize].push(x_r_c0);
        self.rows_by_type[PolyType::G2AddXRC1 as usize].push(x_r_c1);
        self.rows_by_type[PolyType::G2AddYRC0 as usize].push(y_r_c0);
        self.rows_by_type[PolyType::G2AddYRC1 as usize].push(y_r_c1);
        self.rows_by_type[PolyType::G2AddRIndicator as usize].push(ind_r);

        self.rows_by_type[PolyType::G2AddLambdaC0 as usize].push(lambda_c0);
        self.rows_by_type[PolyType::G2AddLambdaC1 as usize].push(lambda_c1);
        self.rows_by_type[PolyType::G2AddInvDeltaXC0 as usize].push(inv_delta_x_c0);
        self.rows_by_type[PolyType::G2AddInvDeltaXC1 as usize].push(inv_delta_x_c1);
        self.rows_by_type[PolyType::G2AddIsDouble as usize].push(is_double);
        self.rows_by_type[PolyType::G2AddIsInverse as usize].push(is_inverse);

        self.push_zero_rows_except(row_size, &G2_ADD_POLYS);

        self.constraint_types.push(ConstraintType::G2Add);
    }

    /// Add constraints from a Multi-Miller loop witness.
    ///
    /// One `ConstraintType::MultiMillerLoop` is added **per traced pair** in the witness.
    /// Each packed trace is an 11-var MLE (length \(2^{11}\)).
    #[cfg(feature = "experimental-pairing-recursion")]
    pub fn add_multi_miller_loop_witness(
        &mut self,
        witness: &crate::poly::commitment::dory::recursion::JoltMultiMillerLoopWitness,
    ) {
        assert_eq!(
            self.num_constraint_vars, 11,
            "MultiMillerLoop expects 11 constraint vars (11-var packed trace MLEs)"
        );
        let row_size = 1 << self.num_constraint_vars;

        let num_pairs = witness.f_packed_mles.len();
        debug_assert_eq!(witness.f_next_packed_mles.len(), num_pairs);
        debug_assert_eq!(witness.quotient_packed_mles.len(), num_pairs);

        for pair_idx in 0..num_pairs {
            // Basic shape sanity (catch accidental padding bugs early).
            debug_assert_eq!(witness.f_packed_mles[pair_idx].len(), row_size);
            debug_assert_eq!(witness.f_next_packed_mles[pair_idx].len(), row_size);
            debug_assert_eq!(witness.t_x_c0_packed_mles[pair_idx].len(), row_size);
            debug_assert_eq!(witness.t_x_c0_next_packed_mles[pair_idx].len(), row_size);
            debug_assert_eq!(witness.inv_two_y_c0_packed_mles[pair_idx].len(), row_size);
            debug_assert_eq!(witness.inv_two_y_c1_packed_mles[pair_idx].len(), row_size);

            self.rows_by_type[PolyType::MultiMillerLoopF as usize]
                .push(witness.f_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopFNext as usize]
                .push(witness.f_next_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopQuotient as usize]
                .push(witness.quotient_packed_mles[pair_idx].clone());

            self.rows_by_type[PolyType::MultiMillerLoopTXC0 as usize]
                .push(witness.t_x_c0_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopTXC1 as usize]
                .push(witness.t_x_c1_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopTYC0 as usize]
                .push(witness.t_y_c0_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopTYC1 as usize]
                .push(witness.t_y_c1_packed_mles[pair_idx].clone());

            self.rows_by_type[PolyType::MultiMillerLoopTXC0Next as usize]
                .push(witness.t_x_c0_next_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopTXC1Next as usize]
                .push(witness.t_x_c1_next_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopTYC0Next as usize]
                .push(witness.t_y_c0_next_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopTYC1Next as usize]
                .push(witness.t_y_c1_next_packed_mles[pair_idx].clone());

            self.rows_by_type[PolyType::MultiMillerLoopLambdaC0 as usize]
                .push(witness.lambda_c0_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopLambdaC1 as usize]
                .push(witness.lambda_c1_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopInvDeltaXC0 as usize]
                .push(witness.inv_dx_c0_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopInvDeltaXC1 as usize]
                .push(witness.inv_dx_c1_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopInvTwoYC0 as usize]
                .push(witness.inv_two_y_c0_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopInvTwoYC1 as usize]
                .push(witness.inv_two_y_c1_packed_mles[pair_idx].clone());

            self.rows_by_type[PolyType::MultiMillerLoopXP as usize]
                .push(witness.x_p_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopYP as usize]
                .push(witness.y_p_packed_mles[pair_idx].clone());

            self.rows_by_type[PolyType::MultiMillerLoopXQC0 as usize]
                .push(witness.x_q_c0_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopXQC1 as usize]
                .push(witness.x_q_c1_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopYQC0 as usize]
                .push(witness.y_q_c0_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopYQC1 as usize]
                .push(witness.y_q_c1_packed_mles[pair_idx].clone());

            self.rows_by_type[PolyType::MultiMillerLoopIsDouble as usize]
                .push(witness.is_double_packed_mles[pair_idx].clone());
            self.rows_by_type[PolyType::MultiMillerLoopIsAdd as usize]
                .push(witness.is_add_packed_mles[pair_idx].clone());

            self.rows_by_type[PolyType::MultiMillerLoopLVal as usize]
                .push(witness.l_val_packed_mles[pair_idx].clone());

            self.push_zero_rows_except(row_size, &MULTI_MILLER_LOOP_POLYS);

            self.constraint_types.push(ConstraintType::MultiMillerLoop);
        }
    }

    /// Add constraint from a per-operation GT multiplication witness (from combine_commitments).
    /// This is the same as `add_gt_mul_witness` but accepts `GTMulOpWitness` type.
    pub fn add_gt_mul_op_witness(&mut self, witness: &GTMulOpWitness) {
        // Skip witnesses with empty quotient MLEs
        if witness.quotient_mle.is_empty() {
            tracing::debug!(
                "[Homomorphic Combine] Skipping GT mul witness with empty quotient_mle"
            );
            return;
        }

        let lhs_mle_4var = fq12_to_multilinear_evals(&witness.lhs);
        let rhs_mle_4var = fq12_to_multilinear_evals(&witness.rhs);
        let result_mle_4var = fq12_to_multilinear_evals(&witness.result);
        let quotient_mle_4var = witness.quotient_mle.clone();

        assert_eq!(
            lhs_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );
        assert_eq!(
            rhs_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );
        assert_eq!(
            result_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );
        assert_eq!(
            quotient_mle_4var.len(),
            16,
            "GT mul witness should have 4-variable MLEs"
        );

        let (lhs_mle, rhs_mle, result_mle, quotient_mle) = if self.num_constraint_vars == 11 {
            (
                Self::pad_4var_to_11var_zero_padding(&lhs_mle_4var),
                Self::pad_4var_to_11var_zero_padding(&rhs_mle_4var),
                Self::pad_4var_to_11var_zero_padding(&result_mle_4var),
                Self::pad_4var_to_11var_zero_padding(&quotient_mle_4var),
            )
        } else if self.num_constraint_vars == 8 {
            (
                Self::pad_4var_to_8var_zero_padding(&lhs_mle_4var),
                Self::pad_4var_to_8var_zero_padding(&rhs_mle_4var),
                Self::pad_4var_to_8var_zero_padding(&result_mle_4var),
                Self::pad_4var_to_8var_zero_padding(&quotient_mle_4var),
            )
        } else if self.num_constraint_vars == 4 {
            (
                lhs_mle_4var,
                rhs_mle_4var,
                result_mle_4var,
                quotient_mle_4var,
            )
        } else {
            panic!(
                "Unsupported number of constraint variables: {}",
                self.num_constraint_vars
            );
        };

        assert_eq!(lhs_mle.len(), 1 << self.num_constraint_vars);
        assert_eq!(rhs_mle.len(), 1 << self.num_constraint_vars);
        assert_eq!(result_mle.len(), 1 << self.num_constraint_vars);
        assert_eq!(quotient_mle.len(), 1 << self.num_constraint_vars);

        self.rows_by_type[PolyType::MulLhs as usize].push(lhs_mle);
        self.rows_by_type[PolyType::MulRhs as usize].push(rhs_mle);
        self.rows_by_type[PolyType::MulResult as usize].push(result_mle);
        self.rows_by_type[PolyType::MulQuotient as usize].push(quotient_mle);

        self.push_zero_rows_except(
            1 << self.num_constraint_vars,
            &[
                PolyType::MulLhs,
                PolyType::MulRhs,
                PolyType::MulResult,
                PolyType::MulQuotient,
            ],
        );

        self.constraint_types.push(ConstraintType::GtMul);
    }

    /// Add all constraints from a GTCombineWitness (homomorphic combine offloading).
    /// Returns the packed GT exp witnesses that were created.
    pub fn add_combine_witness(
        &mut self,
        witness: &GTCombineWitness,
    ) -> Vec<crate::zkvm::recursion::gt::exponentiation::GtExpWitness<Fq>> {
        use crate::zkvm::recursion::gt::exponentiation::GtExpWitness;
        let mut packed_witnesses = Vec::new();

        tracing::info!(
            "[add_combine_witness] Processing {} GT exp witnesses",
            witness.exp_witnesses.len()
        );

        // Convert and add GT exp witnesses in packed format
        for (idx, exp_wit) in witness.exp_witnesses.iter().enumerate() {
            // Handle edge cases where exponent is 0 or 1 (no bits)
            if exp_wit.bits.is_empty() {
                tracing::info!(
                    "[add_combine_witness] GT exp witness {} has empty bits, creating minimal witness",
                    idx
                );
                let base_mle = fq12_to_multilinear_evals(&exp_wit.base);
                let base2_mle = fq12_to_multilinear_evals(&(exp_wit.base * exp_wit.base));
                let base3_mle =
                    fq12_to_multilinear_evals(&(exp_wit.base * exp_wit.base * exp_wit.base));

                let rho_mles = if exp_wit.rho_mles.is_empty() {
                    vec![fq12_to_multilinear_evals(&exp_wit.result)]
                } else {
                    exp_wit.rho_mles.clone()
                };
                let quotient_mles = exp_wit.quotient_mles.clone();

                let packed = GtExpWitness::from_steps(
                    &rho_mles,
                    &quotient_mles,
                    &exp_wit.bits,
                    &base_mle,
                    &base2_mle,
                    &base3_mle,
                );
                self.add_gt_exp_witness(&packed);
                packed_witnesses.push(packed);
                continue;
            }

            // Convert base Fq12 to 4-var MLE
            let base_mle = fq12_to_multilinear_evals(&exp_wit.base);
            let base2_mle = fq12_to_multilinear_evals(&(exp_wit.base * exp_wit.base));
            let base3_mle =
                fq12_to_multilinear_evals(&(exp_wit.base * exp_wit.base * exp_wit.base));

            // Validate and fix witness data if needed
            let num_steps = exp_wit.bits.len().div_ceil(2);
            let (rho_mles, quotient_mles) = if exp_wit.quotient_mles.len() != num_steps {
                // Fix mismatched sizes
                let mut fixed_quotients = exp_wit.quotient_mles.clone();

                // Ensure we have exactly num_steps quotient MLEs
                fixed_quotients.resize(num_steps, vec![Fq::zero(); 16]);

                // Ensure we have exactly num_steps + 1 rho MLEs
                let mut fixed_rhos = exp_wit.rho_mles.clone();
                if fixed_rhos.len() < num_steps + 1 {
                    // Pad with result MLE if needed
                    let result_mle = fq12_to_multilinear_evals(&exp_wit.result);
                    while fixed_rhos.len() < num_steps + 1 {
                        fixed_rhos.push(result_mle.clone());
                    }
                }

                (fixed_rhos, fixed_quotients)
            } else {
                (exp_wit.rho_mles.clone(), exp_wit.quotient_mles.clone())
            };

            // Create packed witness
            let packed = GtExpWitness::from_steps(
                &rho_mles,
                &quotient_mles,
                &exp_wit.bits,
                &base_mle,
                &base2_mle,
                &base3_mle,
            );

            // Add to matrix
            self.add_gt_exp_witness(&packed);
            packed_witnesses.push(packed);
        }

        // Add GT mul witnesses
        tracing::info!(
            "[add_combine_witness] Processing {} GT mul witnesses",
            witness.mul_layers.iter().map(|l| l.len()).sum::<usize>()
        );
        let mut idx = 0usize;
        for (level, layer) in witness.mul_layers.iter().enumerate() {
            for mul_wit in layer {
                tracing::debug!(
                    "[add_combine_witness] Adding GT mul witness {} (level={})",
                    idx,
                    level
                );
                self.add_gt_mul_op_witness(mul_wit);
                idx += 1;
            }
        }

        tracing::info!(
            "[add_combine_witness] Total constraints after combine witness: {}",
            self.constraint_count()
        );

        packed_witnesses
    }

    pub fn build(self) -> (DoryMultilinearMatrix, Vec<MatrixConstraint>) {
        let num_constraints = self.rows_by_type[0].len();
        assert!(num_constraints > 0, "No constraints added");

        // Debug: print constraint counts for each type
        for &poly_type in PolyType::all() {
            let count = self.rows_by_type[poly_type as usize].len();
            if count != num_constraints {
                eprintln!(
                    "Row type {poly_type:?} has {count} constraints, expected {num_constraints}"
                );
            }
        }

        for &poly_type in PolyType::all() {
            assert_eq!(
                self.rows_by_type[poly_type as usize].len(),
                num_constraints,
                "Row type {poly_type:?} has wrong number of constraints"
            );
        }
        assert_eq!(
            self.constraint_types.len(),
            num_constraints,
            "Number of constraint types must match number of constraints"
        );

        // Pad num_constraints to next power of 2
        let num_constraints_bits = (num_constraints as f64).log2().ceil() as usize;
        let num_constraints_padded = 1 << num_constraints_bits;

        // Total rows = NUM_TYPES × num_constraints_padded
        let num_rows_unpadded = PolyType::NUM_TYPES * num_constraints_padded;

        // Pad num_rows to next power of 2 for the matrix
        let num_s_vars = (num_rows_unpadded as f64).log2().ceil() as usize;
        let num_rows = 1 << num_s_vars;

        // Sanity checks
        assert!(
            num_constraints_padded >= num_constraints,
            "Padded constraints must be at least as large as actual constraints"
        );
        assert_eq!(num_rows, 1 << num_s_vars, "num_rows must be a power of 2");
        assert!(
            num_rows >= PolyType::NUM_TYPES * num_constraints_padded,
            "num_rows must be at least NUM_TYPES * num_constraints_padded"
        );
        assert_eq!(
            1 << num_constraints_bits,
            num_constraints_padded,
            "Constraints padding must be power of 2"
        );

        let row_size = 1 << self.num_constraint_vars;
        let capacity = num_rows * row_size;

        // Pre-allocate the exact size and initialize to zero
        let mut evaluations = vec![Fq::zero(); capacity];

        // Pre-compute offsets for each PolyType to enable parallel copy
        // Each PolyType writes to a contiguous block of size num_constraints_padded * row_size
        let block_size = num_constraints_padded * row_size;

        // Extract rows_by_type to avoid capturing self in the closure
        let rows_by_type = &self.rows_by_type;

        // Create a wrapper to safely share the evaluations pointer across threads
        // Safety: Each thread writes to a disjoint region based on type_idx
        #[derive(Clone, Copy)]
        struct SendPtr(*mut Fq);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}

        let send_ptr = SendPtr(evaluations.as_mut_ptr());

        (0..PolyType::NUM_TYPES)
            .into_par_iter()
            .for_each(move |type_idx| {
                let rows = &rows_by_type[type_idx];
                let base_offset = type_idx * block_size;
                let ptr = send_ptr; // Copy the SendPtr (implements Copy)

                // Copy rows for this PolyType
                for (row_idx, row) in rows.iter().enumerate() {
                    let offset = base_offset + row_idx * row_size;
                    // Safety: Each (type_idx, row_idx) writes to a unique, non-overlapping region
                    unsafe {
                        std::ptr::copy_nonoverlapping(row.as_ptr(), ptr.0.add(offset), row_size);
                    }
                }
                // Zero padding is already initialized, no need to write zeros
            });

        let matrix = DoryMultilinearMatrix {
            num_s_vars,
            num_constraint_vars: self.num_constraint_vars,
            num_constraint_index_vars: num_constraints_bits,
            num_constraints,
            num_constraints_padded,
            num_rows,
            num_vars: num_s_vars + self.num_constraint_vars,
            evaluations,
        };

        let constraints: Vec<MatrixConstraint> = self
            .constraint_types
            .into_iter()
            .enumerate()
            .map(|(idx, constraint_type)| MatrixConstraint {
                constraint_index: idx,
                constraint_type,
            })
            .collect();

        (matrix, constraints)
    }
}

/// Type of constraint
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstraintType {
    /// Packed GT exponentiation constraint (one per GT exp, covers all 254 steps)
    GtExp,
    /// GT multiplication constraint
    GtMul,
    /// G1 scalar multiplication constraint with base point
    G1ScalarMul {
        base_point: (Fq, Fq), // (x_p, y_p)
    },
    /// G2 scalar multiplication constraint with base point (over Fq2)
    G2ScalarMul {
        base_point: (Fq2, Fq2), // (x_p, y_p) in Fq2
    },
    /// G1 addition constraint
    G1Add,
    /// G2 addition constraint
    G2Add,
    /// Multi-Miller loop constraint (BN254 pairing Miller loop), one per traced pair.
    #[cfg(feature = "experimental-pairing-recursion")]
    MultiMillerLoop,
}

// -----------------------------------------------------------------------------
// Canonical mapping: ConstraintType -> committed PolyTypes (+ native variable count)
//
// This is used to keep matrix construction, jagged layout, and tests consistent.
// -----------------------------------------------------------------------------

#[cfg(not(feature = "experimental-pairing-recursion"))]
const GT_EXP_POLYS: [PolyType; 2] = [PolyType::RhoPrev, PolyType::Quotient];
#[cfg(not(feature = "experimental-pairing-recursion"))]
const GT_EXP_SPECS: [(PolyType, usize); 2] = [(PolyType::RhoPrev, 11), (PolyType::Quotient, 11)];

#[cfg(feature = "experimental-pairing-recursion")]
const GT_EXP_POLYS: [PolyType; 2] = [PolyType::RhoPrev, PolyType::Quotient];
#[cfg(feature = "experimental-pairing-recursion")]
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
    /// Committed polynomial types used by this constraint type (in PolyType enum order).
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

    /// Committed polynomial specs (poly type + native num_vars) used by this constraint type.
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

/// Canonical sparse set of committed polynomials (poly type + native variable count) for a given
/// constraint list, ordered to match matrix layout (poly-type major, then constraint index).
#[derive(Clone, Debug)]
pub struct PolyTypeSet {
    entries: Vec<(usize, PolyType, usize)>,
}

impl PolyTypeSet {
    pub fn from_constraint_types(constraint_types: &[ConstraintType]) -> Self {
        let mut entries: Vec<(usize, PolyType, usize)> = Vec::new();
        for (constraint_idx, ct) in constraint_types.iter().enumerate() {
            for &(poly_type, num_vars) in ct.committed_poly_specs() {
                entries.push((constraint_idx, poly_type, num_vars));
            }
        }

        // Sort to match matrix layout order (PolyType-major, then constraint index).
        entries
            .sort_by_key(|(constraint_idx, poly_type, _)| (*poly_type as usize, *constraint_idx));

        Self { entries }
    }

    pub fn entries(&self) -> &[(usize, PolyType, usize)] {
        &self.entries
    }

    pub fn into_entries(self) -> Vec<(usize, PolyType, usize)> {
        self.entries
    }
}

/// Constraint metadata for matrix-based evaluation.
/// Row indices are computed from constraint_index using the matrix layout.
#[derive(Clone, Debug)]
pub struct MatrixConstraint {
    /// Index of this constraint (0 to num_constraints-1)
    pub constraint_index: usize,
    /// Type of constraint (GT exp or GT mul)
    pub constraint_type: ConstraintType,
}

/// Constraint system using a giant multilinear matrix for all witness polynomials
#[derive(Clone)]
pub struct ConstraintSystem {
    /// The giant matrix M(s, x)
    pub matrix: DoryMultilinearMatrix,

    /// g(x) polynomial - precomputed as DensePolynomial
    pub g_poly: DensePolynomial<Fq>,

    /// Constraint metadata: maps constraint index to matrix rows it references
    pub constraints: Vec<MatrixConstraint>,

    /// Packed GT exp witnesses for Stage 1 prover (base-4 steps packed into 11-var MLEs)
    pub gt_exp_witnesses: Vec<crate::zkvm::recursion::gt::exponentiation::GtExpWitness<Fq>>,

    /// Public inputs for packed GT exp (base Fq12 and scalar bits) - used by verifier
    /// and Stage 2 to compute digit/base evaluations directly
    pub gt_exp_public_inputs: Vec<GtExpPublicInputs>,

    /// Public inputs for G1 scalar multiplication (the scalars, one per G1ScalarMul constraint)
    pub g1_scalar_mul_public_inputs: Vec<G1ScalarMulPublicInputs>,

    /// Public inputs for G2 scalar multiplication (the scalars, one per G2ScalarMul constraint)
    pub g2_scalar_mul_public_inputs: Vec<G2ScalarMulPublicInputs>,

    /// G1 addition witnesses for Stage 1 prover (one per `ConstraintType::G1Add`)
    pub g1_add_witnesses: Vec<crate::zkvm::recursion::g1::addition::G1AddWitness<ark_bn254::Fq>>,

    /// G2 addition witnesses for Stage 1 prover (one per `ConstraintType::G2Add`)
    pub g2_add_witnesses: Vec<crate::zkvm::recursion::g2::addition::G2AddWitness<ark_bn254::Fq>>,
}

impl ConstraintSystem {
    /// Create constraint system from witness data
    pub fn from_witness(
        constraint_types: Vec<ConstraintType>,
        g_poly: DensePolynomial<Fq>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // For now, create a simple matrix with padding
        let _num_constraints = constraint_types.len();
        let num_constraint_vars = 11; // Using 11 for packed GT exp base-4

        let mut builder = DoryMatrixBuilder::new(num_constraint_vars);

        // Add dummy data for each constraint type
        // This is a placeholder - should be replaced with actual witness data
        let zero_row = vec![Fq::zero(); 1 << num_constraint_vars];

        for constraint_type in &constraint_types {
            match constraint_type {
                ConstraintType::GtExp => {
                    // Add packed GT exp rows (2 committed polynomials - base/digits/rho_next are not committed)
                    builder.rows_by_type[PolyType::RhoPrev as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::Quotient as usize].push(zero_row.clone());

                    // Add empty rows for all other polynomial types to maintain consistent indexing
                    builder.push_zero_rows_except(1 << num_constraint_vars, &GT_EXP_POLYS);

                    builder.constraint_types.push(constraint_type.clone());
                }
                ConstraintType::GtMul => {
                    // Add GT mul rows
                    builder.rows_by_type[PolyType::MulLhs as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::MulRhs as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::MulResult as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::MulQuotient as usize].push(zero_row.clone());

                    // Add empty rows for all other polynomial types to maintain consistent indexing
                    builder.push_zero_rows_except(1 << num_constraint_vars, &GT_MUL_POLYS);

                    builder.constraint_types.push(constraint_type.clone());
                }
                ConstraintType::G1ScalarMul { .. } => {
                    // Add G1 scalar mul rows
                    builder.rows_by_type[PolyType::G1ScalarMulXA as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::G1ScalarMulYA as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::G1ScalarMulXT as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::G1ScalarMulYT as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::G1ScalarMulXANext as usize]
                        .push(zero_row.clone());
                    builder.rows_by_type[PolyType::G1ScalarMulYANext as usize]
                        .push(zero_row.clone());
                    builder.rows_by_type[PolyType::G1ScalarMulTIndicator as usize]
                        .push(zero_row.clone());
                    builder.rows_by_type[PolyType::G1ScalarMulAIndicator as usize]
                        .push(zero_row.clone());

                    builder.push_zero_rows_except(1 << num_constraint_vars, &G1_SCALAR_MUL_POLYS);

                    builder.constraint_types.push(constraint_type.clone());
                }
                ConstraintType::G2ScalarMul { .. } => {
                    // Add G2 scalar mul rows (placeholders)
                    builder.rows_by_type[PolyType::G2ScalarMulXAC0 as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::G2ScalarMulXAC1 as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::G2ScalarMulYAC0 as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::G2ScalarMulYAC1 as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::G2ScalarMulXTC0 as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::G2ScalarMulXTC1 as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::G2ScalarMulYTC0 as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::G2ScalarMulYTC1 as usize].push(zero_row.clone());
                    builder.rows_by_type[PolyType::G2ScalarMulXANextC0 as usize]
                        .push(zero_row.clone());
                    builder.rows_by_type[PolyType::G2ScalarMulXANextC1 as usize]
                        .push(zero_row.clone());
                    builder.rows_by_type[PolyType::G2ScalarMulYANextC0 as usize]
                        .push(zero_row.clone());
                    builder.rows_by_type[PolyType::G2ScalarMulYANextC1 as usize]
                        .push(zero_row.clone());
                    builder.rows_by_type[PolyType::G2ScalarMulTIndicator as usize]
                        .push(zero_row.clone());
                    builder.rows_by_type[PolyType::G2ScalarMulAIndicator as usize]
                        .push(zero_row.clone());

                    builder.push_zero_rows_except(1 << num_constraint_vars, &G2_SCALAR_MUL_POLYS);

                    builder.constraint_types.push(constraint_type.clone());
                }
                ConstraintType::G1Add | ConstraintType::G2Add => {
                    // Placeholder: add constraints are not yet wired into this constructor path.
                    // Maintain consistent indexing by appending all-zero rows for every PolyType.
                    builder.push_zero_rows_except(1 << num_constraint_vars, &[]);
                    builder.constraint_types.push(constraint_type.clone());
                }
                #[cfg(feature = "experimental-pairing-recursion")]
                ConstraintType::MultiMillerLoop => {
                    // Placeholder: pairing constraints are not yet wired into this constructor path.
                    // Maintain consistent indexing by appending all-zero rows for every PolyType.
                    builder.push_zero_rows_except(1 << num_constraint_vars, &[]);
                    builder.constraint_types.push(constraint_type.clone());
                }
            }
        }

        let (matrix, constraints) = builder.build();

        Ok(Self {
            matrix,
            g_poly,
            constraints,
            gt_exp_witnesses: Vec::new(), // No actual witnesses in from_witness (test helper)
            gt_exp_public_inputs: Vec::new(), // No actual public inputs in from_witness (test helper)
            g1_scalar_mul_public_inputs: Vec::new(),
            g2_scalar_mul_public_inputs: Vec::new(),
            g1_add_witnesses: Vec::new(),
            g2_add_witnesses: Vec::new(),
        })
    }

    /// Get the number of variables in the constraint system
    pub fn num_vars(&self) -> usize {
        self.matrix.num_vars
    }

    /// Get the number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Get the number of s variables (for virtualization)
    pub fn num_s_vars(&self) -> usize {
        self.matrix.num_s_vars
    }

    pub fn new<T>(
        proof: &ArkDoryProof,
        setup: &ArkworksVerifierSetup,
        transcript: &mut T,
        point: &[<Fr as JoltField>::Challenge],
        evaluation: &Fr,
        commitment: &ArkGT,
    ) -> Result<(Self, <DoryCommitmentScheme as RecursionExt<Fr>>::Hint), ProofVerifyError>
    where
        T: Transcript,
    {
        let (witnesses, hints) = <DoryCommitmentScheme as RecursionExt<Fr>>::witness_gen(
            proof, setup, transcript, point, evaluation, commitment,
        )?;

        // Always use 11 variables for uniform matrix structure
        // GT operations (4-var) and G1 operations (8-var) will be padded to 11 vars
        let mut builder = DoryMatrixBuilder::new(11);

        // Create packed GT exp witnesses and public inputs, add to matrix
        let mut gt_exp_witnesses = Vec::with_capacity(witnesses.gt_exp.len());
        let mut gt_exp_public_inputs = Vec::with_capacity(witnesses.gt_exp.len());
        for (_op_id, witness) in witnesses.gt_exp.iter() {
            let base_mle = fq12_to_multilinear_evals(&witness.base);
            let base2_mle = fq12_to_multilinear_evals(&(witness.base * witness.base));
            let base3_mle =
                fq12_to_multilinear_evals(&(witness.base * witness.base * witness.base));
            let packed = crate::zkvm::recursion::gt::exponentiation::GtExpWitness::from_steps(
                &witness.rho_mles,
                &witness.quotient_mles,
                &witness.bits,
                &base_mle,
                &base2_mle,
                &base3_mle,
            );
            builder.add_gt_exp_witness(&packed);
            gt_exp_witnesses.push(packed);
            gt_exp_public_inputs.push(GtExpPublicInputs::new(witness.base, witness.bits.clone()));
        }

        for (_op_id, witness) in witnesses.gt_mul.iter() {
            builder.add_gt_mul_witness(witness);
        }

        // Collect scalar public inputs for scalar-mul constraints (bits are treated as public inputs)
        let mut g1_scalar_mul_public_inputs = Vec::with_capacity(witnesses.g1_scalar_mul.len());
        let mut g2_scalar_mul_public_inputs = Vec::with_capacity(witnesses.g2_scalar_mul.len());

        // Add G1 scalar multiplication witnesses
        for (_op_id, witness) in witnesses.g1_scalar_mul.iter() {
            builder.add_g1_scalar_mul_witness(witness);
            g1_scalar_mul_public_inputs.push(G1ScalarMulPublicInputs::new(witness.scalar));
        }

        // Add G2 scalar multiplication witnesses
        for (_op_id, witness) in witnesses.g2_scalar_mul.iter() {
            builder.add_g2_scalar_mul_witness(witness);
            g2_scalar_mul_public_inputs.push(G2ScalarMulPublicInputs::new(witness.scalar));
        }

        // Add G1 add witnesses
        for (_op_id, witness) in witnesses.g1_add.iter() {
            builder.add_g1_add_witness(witness);
        }

        // Add G2 add witnesses
        for (_op_id, witness) in witnesses.g2_add.iter() {
            builder.add_g2_add_witness(witness);
        }

        let (matrix, constraints) = builder.build();

        // Get the 4-variable g(x) polynomial
        let g_mle_4var = get_g_mle();

        // Pad g(x) to match constraint vars
        let g_poly = if matrix.num_constraint_vars == 11 {
            let padded_g = DoryMatrixBuilder::pad_4var_to_11var_zero_padding(&g_mle_4var);
            DensePolynomial::new(padded_g)
        } else if matrix.num_constraint_vars == 8 {
            let padded_g = DoryMatrixBuilder::pad_4var_to_8var_zero_padding(&g_mle_4var);
            DensePolynomial::new(padded_g)
        } else {
            DensePolynomial::new(g_mle_4var)
        };

        Ok((
            Self {
                matrix,
                g_poly,
                constraints,
                gt_exp_witnesses,
                gt_exp_public_inputs,
                g1_scalar_mul_public_inputs,
                g2_scalar_mul_public_inputs,
                g1_add_witnesses: Vec::new(),
                g2_add_witnesses: Vec::new(),
            },
            hints,
        ))
    }

    /// Extract GT mul constraint data for gt_mul sumcheck
    #[allow(clippy::type_complexity)]
    pub fn extract_gt_mul_constraints(&self) -> Vec<(usize, Vec<Fq>, Vec<Fq>, Vec<Fq>, Vec<Fq>)> {
        let num_constraint_vars = self.matrix.num_constraint_vars;
        let row_size = 1 << num_constraint_vars;

        // Pre-allocate with exact capacity
        let gt_mul_count = self
            .constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, ConstraintType::GtMul))
            .count();
        let mut constraints = Vec::with_capacity(gt_mul_count);

        for (idx, constraint) in self.constraints.iter().enumerate() {
            if let ConstraintType::GtMul = constraint.constraint_type {
                let lhs = self.extract_row_poly(PolyType::MulLhs, idx, row_size);
                let rhs = self.extract_row_poly(PolyType::MulRhs, idx, row_size);
                let result = self.extract_row_poly(PolyType::MulResult, idx, row_size);
                let quotient = self.extract_row_poly(PolyType::MulQuotient, idx, row_size);

                constraints.push((constraint.constraint_index, lhs, rhs, result, quotient));
            }
        }

        constraints
    }

    /// Extract G1 scalar mul constraint data for g1_scalar_mul sumcheck.
    ///
    /// Returns a vector of [`G1ScalarMulWitness`] containing the witness polynomials
    /// for each G1 scalar multiplication constraint.
    pub fn extract_g1_scalar_mul_constraints(&self) -> Vec<G1ScalarMulWitness<Fq>> {
        let num_constraint_vars = self.matrix.num_constraint_vars;
        let row_size = 1 << num_constraint_vars;

        // Pre-allocate with exact capacity
        let g1_scalar_mul_count = self
            .constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, ConstraintType::G1ScalarMul { .. }))
            .count();
        let mut constraints = Vec::with_capacity(g1_scalar_mul_count);
        let mut local_idx = 0usize;

        for (idx, constraint) in self.constraints.iter().enumerate() {
            if let ConstraintType::G1ScalarMul { base_point } = constraint.constraint_type {
                // Extract the MLEs for this G1 scalar multiplication
                let x_a = self.extract_row_poly(PolyType::G1ScalarMulXA, idx, row_size);
                let y_a = self.extract_row_poly(PolyType::G1ScalarMulYA, idx, row_size);
                let x_t = self.extract_row_poly(PolyType::G1ScalarMulXT, idx, row_size);
                let y_t = self.extract_row_poly(PolyType::G1ScalarMulYT, idx, row_size);
                let x_a_next = self.extract_row_poly(PolyType::G1ScalarMulXANext, idx, row_size);
                let y_a_next = self.extract_row_poly(PolyType::G1ScalarMulYANext, idx, row_size);
                let t_indicator =
                    self.extract_row_poly(PolyType::G1ScalarMulTIndicator, idx, row_size);
                let a_indicator =
                    self.extract_row_poly(PolyType::G1ScalarMulAIndicator, idx, row_size);

                constraints.push(G1ScalarMulWitness {
                    // Instance index is sequential within the G1ScalarMul family.
                    constraint_index: local_idx,
                    base_point,
                    x_a,
                    y_a,
                    x_t,
                    y_t,
                    x_a_next,
                    y_a_next,
                    t_indicator,
                    a_indicator,
                });
                local_idx += 1;
            }
        }

        constraints
    }

    /// Extract G2 scalar mul constraint data for g2_scalar_mul sumcheck.
    ///
    /// Returns a vector of [`G2ScalarMulWitness`] containing the witness polynomials
    /// for each G2 scalar multiplication constraint.
    pub fn extract_g2_scalar_mul_constraints(&self) -> Vec<G2ScalarMulWitness<Fq>> {
        let num_constraint_vars = self.matrix.num_constraint_vars;
        let row_size = 1 << num_constraint_vars;

        // Pre-allocate with exact capacity
        let g2_scalar_mul_count = self
            .constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, ConstraintType::G2ScalarMul { .. }))
            .count();
        let mut constraints = Vec::with_capacity(g2_scalar_mul_count);
        let mut local_idx = 0usize;

        for (idx, constraint) in self.constraints.iter().enumerate() {
            if let ConstraintType::G2ScalarMul { base_point } = constraint.constraint_type {
                let x_a_c0 = self.extract_row_poly(PolyType::G2ScalarMulXAC0, idx, row_size);
                let x_a_c1 = self.extract_row_poly(PolyType::G2ScalarMulXAC1, idx, row_size);
                let y_a_c0 = self.extract_row_poly(PolyType::G2ScalarMulYAC0, idx, row_size);
                let y_a_c1 = self.extract_row_poly(PolyType::G2ScalarMulYAC1, idx, row_size);
                let x_t_c0 = self.extract_row_poly(PolyType::G2ScalarMulXTC0, idx, row_size);
                let x_t_c1 = self.extract_row_poly(PolyType::G2ScalarMulXTC1, idx, row_size);
                let y_t_c0 = self.extract_row_poly(PolyType::G2ScalarMulYTC0, idx, row_size);
                let y_t_c1 = self.extract_row_poly(PolyType::G2ScalarMulYTC1, idx, row_size);
                let x_a_next_c0 =
                    self.extract_row_poly(PolyType::G2ScalarMulXANextC0, idx, row_size);
                let x_a_next_c1 =
                    self.extract_row_poly(PolyType::G2ScalarMulXANextC1, idx, row_size);
                let y_a_next_c0 =
                    self.extract_row_poly(PolyType::G2ScalarMulYANextC0, idx, row_size);
                let y_a_next_c1 =
                    self.extract_row_poly(PolyType::G2ScalarMulYANextC1, idx, row_size);
                let t_indicator =
                    self.extract_row_poly(PolyType::G2ScalarMulTIndicator, idx, row_size);
                let a_indicator =
                    self.extract_row_poly(PolyType::G2ScalarMulAIndicator, idx, row_size);

                constraints.push(G2ScalarMulWitness {
                    // Instance index is sequential within the G2ScalarMul family.
                    constraint_index: local_idx,
                    base_point,
                    x_a_c0,
                    x_a_c1,
                    y_a_c0,
                    y_a_c1,
                    x_t_c0,
                    x_t_c1,
                    y_t_c0,
                    y_t_c1,
                    x_a_next_c0,
                    x_a_next_c1,
                    y_a_next_c0,
                    y_a_next_c1,
                    t_indicator,
                    a_indicator,
                });
                local_idx += 1;
            }
        }

        constraints
    }

    /// Extract G1 add witnesses for the Stage 1 G1Add sumcheck.
    pub fn extract_g1_add_constraints(
        &self,
    ) -> Vec<crate::zkvm::recursion::g1::addition::G1AddWitness<ark_bn254::Fq>> {
        let num_constraint_vars = self.matrix.num_constraint_vars;
        let row_size = 1 << num_constraint_vars;

        let g1_add_count = self
            .constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, ConstraintType::G1Add))
            .count();

        let mut constraints = Vec::with_capacity(g1_add_count);
        let mut local_idx = 0usize;

        for (idx, constraint) in self.constraints.iter().enumerate() {
            if matches!(constraint.constraint_type, ConstraintType::G1Add) {
                let x_p = self.extract_row_poly(PolyType::G1AddXP, idx, row_size);
                let y_p = self.extract_row_poly(PolyType::G1AddYP, idx, row_size);
                let ind_p = self.extract_row_poly(PolyType::G1AddPIndicator, idx, row_size);
                let x_q = self.extract_row_poly(PolyType::G1AddXQ, idx, row_size);
                let y_q = self.extract_row_poly(PolyType::G1AddYQ, idx, row_size);
                let ind_q = self.extract_row_poly(PolyType::G1AddQIndicator, idx, row_size);
                let x_r = self.extract_row_poly(PolyType::G1AddXR, idx, row_size);
                let y_r = self.extract_row_poly(PolyType::G1AddYR, idx, row_size);
                let ind_r = self.extract_row_poly(PolyType::G1AddRIndicator, idx, row_size);
                let lambda = self.extract_row_poly(PolyType::G1AddLambda, idx, row_size);
                let inv_delta_x = self.extract_row_poly(PolyType::G1AddInvDeltaX, idx, row_size);
                let is_double = self.extract_row_poly(PolyType::G1AddIsDouble, idx, row_size);
                let is_inverse = self.extract_row_poly(PolyType::G1AddIsInverse, idx, row_size);

                constraints.push(crate::zkvm::recursion::g1::addition::G1AddWitness {
                    x_p,
                    y_p,
                    ind_p,
                    x_q,
                    y_q,
                    ind_q,
                    x_r,
                    y_r,
                    ind_r,
                    lambda,
                    inv_delta_x,
                    is_double,
                    is_inverse,
                    // Instance index is sequential within the G1Add family.
                    constraint_index: local_idx,
                });
                local_idx += 1;
            }
        }

        constraints
    }

    /// Extract G2 add witnesses for the Stage 1 G2Add sumcheck.
    pub fn extract_g2_add_constraints(
        &self,
    ) -> Vec<crate::zkvm::recursion::g2::addition::G2AddWitness<ark_bn254::Fq>> {
        let num_constraint_vars = self.matrix.num_constraint_vars;
        let row_size = 1 << num_constraint_vars;

        let g2_add_count = self
            .constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, ConstraintType::G2Add))
            .count();

        let mut constraints = Vec::with_capacity(g2_add_count);
        let mut local_idx = 0usize;

        for (idx, constraint) in self.constraints.iter().enumerate() {
            if matches!(constraint.constraint_type, ConstraintType::G2Add) {
                let x_p_c0 = self.extract_row_poly(PolyType::G2AddXPC0, idx, row_size);
                let x_p_c1 = self.extract_row_poly(PolyType::G2AddXPC1, idx, row_size);
                let y_p_c0 = self.extract_row_poly(PolyType::G2AddYPC0, idx, row_size);
                let y_p_c1 = self.extract_row_poly(PolyType::G2AddYPC1, idx, row_size);
                let ind_p = self.extract_row_poly(PolyType::G2AddPIndicator, idx, row_size);

                let x_q_c0 = self.extract_row_poly(PolyType::G2AddXQC0, idx, row_size);
                let x_q_c1 = self.extract_row_poly(PolyType::G2AddXQC1, idx, row_size);
                let y_q_c0 = self.extract_row_poly(PolyType::G2AddYQC0, idx, row_size);
                let y_q_c1 = self.extract_row_poly(PolyType::G2AddYQC1, idx, row_size);
                let ind_q = self.extract_row_poly(PolyType::G2AddQIndicator, idx, row_size);

                let x_r_c0 = self.extract_row_poly(PolyType::G2AddXRC0, idx, row_size);
                let x_r_c1 = self.extract_row_poly(PolyType::G2AddXRC1, idx, row_size);
                let y_r_c0 = self.extract_row_poly(PolyType::G2AddYRC0, idx, row_size);
                let y_r_c1 = self.extract_row_poly(PolyType::G2AddYRC1, idx, row_size);
                let ind_r = self.extract_row_poly(PolyType::G2AddRIndicator, idx, row_size);

                let lambda_c0 = self.extract_row_poly(PolyType::G2AddLambdaC0, idx, row_size);
                let lambda_c1 = self.extract_row_poly(PolyType::G2AddLambdaC1, idx, row_size);
                let inv_delta_x_c0 =
                    self.extract_row_poly(PolyType::G2AddInvDeltaXC0, idx, row_size);
                let inv_delta_x_c1 =
                    self.extract_row_poly(PolyType::G2AddInvDeltaXC1, idx, row_size);
                let is_double = self.extract_row_poly(PolyType::G2AddIsDouble, idx, row_size);
                let is_inverse = self.extract_row_poly(PolyType::G2AddIsInverse, idx, row_size);

                constraints.push(crate::zkvm::recursion::g2::addition::G2AddWitness {
                    x_p_c0,
                    x_p_c1,
                    y_p_c0,
                    y_p_c1,
                    ind_p,
                    x_q_c0,
                    x_q_c1,
                    y_q_c0,
                    y_q_c1,
                    ind_q,
                    x_r_c0,
                    x_r_c1,
                    y_r_c0,
                    y_r_c1,
                    ind_r,
                    lambda_c0,
                    lambda_c1,
                    inv_delta_x_c0,
                    inv_delta_x_c1,
                    is_double,
                    is_inverse,
                    // Instance index is sequential within the G2Add family.
                    constraint_index: local_idx,
                });
                local_idx += 1;
            }
        }

        constraints
    }

    /// Extract Multi-Miller loop witnesses for the Stage 2 `MultiMillerLoop` sumcheck.
    #[cfg(feature = "experimental-pairing-recursion")]
    pub fn extract_multi_miller_loop_constraints(
        &self,
    ) -> Vec<
        crate::zkvm::recursion::pairing::multi_miller_loop::MultiMillerLoopWitness<ark_bn254::Fq>,
    > {
        let num_constraint_vars = self.matrix.num_constraint_vars;
        let row_size = 1 << num_constraint_vars;

        let mml_count = self
            .constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, ConstraintType::MultiMillerLoop))
            .count();
        let mut constraints = Vec::with_capacity(mml_count);
        let mut local_idx = 0usize;

        for (idx, constraint) in self.constraints.iter().enumerate() {
            if matches!(constraint.constraint_type, ConstraintType::MultiMillerLoop) {
                let f = self.extract_row_poly(PolyType::MultiMillerLoopF, idx, row_size);
                let f_next = self.extract_row_poly(PolyType::MultiMillerLoopFNext, idx, row_size);
                let quotient =
                    self.extract_row_poly(PolyType::MultiMillerLoopQuotient, idx, row_size);

                let t_x_c0 = self.extract_row_poly(PolyType::MultiMillerLoopTXC0, idx, row_size);
                let t_x_c1 = self.extract_row_poly(PolyType::MultiMillerLoopTXC1, idx, row_size);
                let t_y_c0 = self.extract_row_poly(PolyType::MultiMillerLoopTYC0, idx, row_size);
                let t_y_c1 = self.extract_row_poly(PolyType::MultiMillerLoopTYC1, idx, row_size);

                let t_x_c0_next =
                    self.extract_row_poly(PolyType::MultiMillerLoopTXC0Next, idx, row_size);
                let t_x_c1_next =
                    self.extract_row_poly(PolyType::MultiMillerLoopTXC1Next, idx, row_size);
                let t_y_c0_next =
                    self.extract_row_poly(PolyType::MultiMillerLoopTYC0Next, idx, row_size);
                let t_y_c1_next =
                    self.extract_row_poly(PolyType::MultiMillerLoopTYC1Next, idx, row_size);

                let lambda_c0 =
                    self.extract_row_poly(PolyType::MultiMillerLoopLambdaC0, idx, row_size);
                let lambda_c1 =
                    self.extract_row_poly(PolyType::MultiMillerLoopLambdaC1, idx, row_size);
                let inv_delta_x_c0 =
                    self.extract_row_poly(PolyType::MultiMillerLoopInvDeltaXC0, idx, row_size);
                let inv_delta_x_c1 =
                    self.extract_row_poly(PolyType::MultiMillerLoopInvDeltaXC1, idx, row_size);
                let inv_two_y_c0 =
                    self.extract_row_poly(PolyType::MultiMillerLoopInvTwoYC0, idx, row_size);
                let inv_two_y_c1 =
                    self.extract_row_poly(PolyType::MultiMillerLoopInvTwoYC1, idx, row_size);

                let x_p = self.extract_row_poly(PolyType::MultiMillerLoopXP, idx, row_size);
                let y_p = self.extract_row_poly(PolyType::MultiMillerLoopYP, idx, row_size);

                let x_q_c0 = self.extract_row_poly(PolyType::MultiMillerLoopXQC0, idx, row_size);
                let x_q_c1 = self.extract_row_poly(PolyType::MultiMillerLoopXQC1, idx, row_size);
                let y_q_c0 = self.extract_row_poly(PolyType::MultiMillerLoopYQC0, idx, row_size);
                let y_q_c1 = self.extract_row_poly(PolyType::MultiMillerLoopYQC1, idx, row_size);

                let is_double =
                    self.extract_row_poly(PolyType::MultiMillerLoopIsDouble, idx, row_size);
                let is_add = self.extract_row_poly(PolyType::MultiMillerLoopIsAdd, idx, row_size);

                let l_val = self.extract_row_poly(PolyType::MultiMillerLoopLVal, idx, row_size);

                constraints.push(
                    crate::zkvm::recursion::pairing::multi_miller_loop::MultiMillerLoopWitness {
                        f,
                        f_next,
                        quotient,
                        t_x_c0,
                        t_x_c1,
                        t_y_c0,
                        t_y_c1,
                        t_x_c0_next,
                        t_x_c1_next,
                        t_y_c0_next,
                        t_y_c1_next,
                        lambda_c0,
                        lambda_c1,
                        inv_delta_x_c0,
                        inv_delta_x_c1,
                        inv_two_y_c0,
                        inv_two_y_c1,
                        x_p,
                        y_p,
                        x_q_c0,
                        x_q_c1,
                        y_q_c0,
                        y_q_c1,
                        is_double,
                        is_add,
                        l_val,
                        // Instance index is sequential within the MultiMillerLoop family.
                        constraint_index: local_idx,
                    },
                );
                local_idx += 1;
            }
        }

        constraints
    }

    /// Extract packed GT exp constraint data for gt_exp sumcheck
    /// Returns: (constraint_index, rho, quotient) for each GtExp constraint
    /// Note: digit bits, base, and rho_next are public inputs/virtual claims
    pub fn extract_gt_exp_constraints(&self) -> Vec<(usize, Vec<Fq>, Vec<Fq>)> {
        let num_constraint_vars = self.matrix.num_constraint_vars;
        let row_size = 1 << num_constraint_vars;

        // Pre-allocate with exact capacity
        let gt_exp_count = self
            .constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, ConstraintType::GtExp))
            .count();
        let mut constraints = Vec::with_capacity(gt_exp_count);

        for (idx, constraint) in self.constraints.iter().enumerate() {
            if let ConstraintType::GtExp = constraint.constraint_type {
                // Extract the 2 committed MLEs for this packed GT exp
                // Note: RhoPrev = rho in the packed convention
                // Base, digit bits, and rho_next are not committed
                let rho = self.extract_row_poly(PolyType::RhoPrev, idx, row_size);
                let quotient = self.extract_row_poly(PolyType::Quotient, idx, row_size);

                constraints.push((constraint.constraint_index, rho, quotient));
            }
        }

        constraints
    }

    /// Helper to extract a row polynomial from the matrix
    fn extract_row_poly(
        &self,
        poly_type: PolyType,
        constraint_idx: usize,
        row_size: usize,
    ) -> Vec<Fq> {
        let type_start = (poly_type as usize) * self.matrix.num_constraints_padded * row_size;
        let row_start = type_start + constraint_idx * row_size;
        let row_end = row_start + row_size;
        self.matrix.evaluations[row_start..row_end].to_vec()
    }

    /// Debug helper to print constraint evaluation components
    #[allow(dead_code)]
    pub fn debug_constraint_eval(&self, constraint: &MatrixConstraint, x: &[Fq]) {
        let idx = constraint.constraint_index;
        match constraint.constraint_type {
            ConstraintType::GtExp => {
                let g_eval = if x.len() == 11 {
                    let x_elem_reversed: Vec<Fq> = x[7..11].iter().rev().copied().collect();
                    let g_4var = get_g_mle();
                    DensePolynomial::new(g_4var).evaluate(&x_elem_reversed)
                } else {
                    let x_reversed: Vec<Fq> = x.iter().rev().copied().collect();
                    self.g_poly.evaluate(&x_reversed)
                };

                let rho_prev_row = self.matrix.row_index(PolyType::RhoPrev, idx);
                let quotient_row = self.matrix.row_index(PolyType::Quotient, idx);

                let rho_prev = self.matrix.evaluate_row(rho_prev_row, x);
                let quotient = self.matrix.evaluate_row(quotient_row, x);

                // Get rho_next, base_eval and digit bits from packed witness
                // Need to find GT exp witness index (count GtExp constraints before this one)
                let gt_exp_idx = self
                    .constraints
                    .iter()
                    .take(idx)
                    .filter(|c| matches!(c.constraint_type, ConstraintType::GtExp))
                    .count();
                let packed = &self.gt_exp_witnesses[gt_exp_idx];
                let rho_curr = DensePolynomial::new(packed.rho_next_packed.clone()).evaluate(x);
                let base_eval = DensePolynomial::new(packed.base_packed.clone()).evaluate(x);
                let base2_eval = DensePolynomial::new(packed.base2_packed.clone()).evaluate(x);
                let base3_eval = DensePolynomial::new(packed.base3_packed.clone()).evaluate(x);
                let digit_lo_eval =
                    DensePolynomial::new(packed.digit_lo_packed.clone()).evaluate(x);
                let digit_hi_eval =
                    DensePolynomial::new(packed.digit_hi_packed.clone()).evaluate(x);

                let u = digit_lo_eval;
                let v = digit_hi_eval;
                let w0 = (Fq::one() - u) * (Fq::one() - v);
                let w1 = u * (Fq::one() - v);
                let w2 = (Fq::one() - u) * v;
                let w3 = u * v;
                let base_power = w0 + w1 * base_eval + w2 * base2_eval + w3 * base3_eval;
                let rho2 = rho_prev * rho_prev;
                let rho4 = rho2 * rho2;
                let constraint_eval = rho_curr - rho4 * base_power - quotient * g_eval;

                // Convert point to index
                let mut index = 0usize;
                for (i, &b) in x.iter().enumerate() {
                    if b == Fq::one() {
                        index |= 1 << i;
                    }
                }
                let s_index = index & 0x7F; // low 7 bits
                let x_index = (index >> 7) & 0xF; // high 4 bits

                println!("  s_index={s_index}, x_index={x_index}");
                println!("  base_eval = {base_eval:?}");
                println!("  rho_prev = {rho_prev:?}");
                println!("  rho_curr = {rho_curr:?}");
                println!("  quotient = {quotient:?}");
                println!("  digit_lo_eval = {digit_lo_eval:?}");
                println!("  digit_hi_eval = {digit_hi_eval:?}");
                println!("  g_eval = {g_eval:?}");
                println!("  base_power = {base_power:?}");
                println!("  constraint = {constraint_eval:?}");

                // Also check raw data at the specific index
                println!("  --- Raw packed data at index {index} ---");
                println!(
                    "  rho_packed[{}] = {:?}",
                    index,
                    packed.rho_packed.get(index)
                );
                println!(
                    "  rho_next_packed[{}] = {:?}",
                    index,
                    packed.rho_next_packed.get(index)
                );
                println!(
                    "  quotient_packed[{}] = {:?}",
                    index,
                    packed.quotient_packed.get(index)
                );
                println!(
                    "  digit_lo_packed[{}] = {:?}",
                    index,
                    packed.digit_lo_packed.get(index)
                );
                println!(
                    "  digit_hi_packed[{}] = {:?}",
                    index,
                    packed.digit_hi_packed.get(index)
                );
                println!(
                    "  base_packed[{}] = {:?}",
                    index,
                    packed.base_packed.get(index)
                );
                println!(
                    "  base2_packed[{}] = {:?}",
                    index,
                    packed.base2_packed.get(index)
                );
                println!(
                    "  base3_packed[{}] = {:?}",
                    index,
                    packed.base3_packed.get(index)
                );
            }
            _ => {
                println!("  Non-GtExp constraint, skipping debug");
            }
        }
    }

    /// Evaluate μ at a given point s
    /// This is used for testing/debugging - the actual evaluation should use the multilinear polynomial
    /// Note: Base and digit bits are public inputs computed by verifier, not committed polynomials
    pub fn evaluate_mu_at_binary_point(
        rho_prev_claim: Fq,
        _rho_curr_claim: Fq,
        quotient_claim: Fq,
        s_binary: &[Fq],
        num_constraints_padded: usize,
    ) -> Fq {
        // Convert binary point to index (for testing only)
        let mut s_index = 0usize;
        for (i, &bit) in s_binary.iter().enumerate() {
            if bit == Fq::one() {
                s_index |= 1 << i;
            }
        }

        // Determine which polynomial type this row index corresponds to
        let poly_type = PolyType::from_row_index(s_index, num_constraints_padded);

        match poly_type {
            PolyType::RhoPrev => rho_prev_claim,
            PolyType::Quotient => quotient_claim,
            PolyType::MulLhs => Fq::zero(),
            PolyType::MulRhs => Fq::zero(),
            PolyType::MulResult => Fq::zero(),
            PolyType::MulQuotient => Fq::zero(),
            PolyType::G1ScalarMulXA => Fq::zero(),
            PolyType::G1ScalarMulYA => Fq::zero(),
            PolyType::G1ScalarMulXT => Fq::zero(),
            PolyType::G1ScalarMulYT => Fq::zero(),
            PolyType::G1ScalarMulXANext => Fq::zero(),
            PolyType::G1ScalarMulYANext => Fq::zero(),
            PolyType::G1ScalarMulTIndicator => Fq::zero(),
            PolyType::G1ScalarMulAIndicator => Fq::zero(),
            PolyType::G1ScalarMulBit => Fq::zero(),
            _ => Fq::zero(),
        }
    }

    /// Evaluate the constraint system for Stage 1 sumcheck
    ///
    /// Takes only x variables and evaluates F(x) = Σ_i γ^i * C_i(x)
    /// This is used in the GT exponentiation sumcheck.
    pub fn evaluate_constraints_batched(&self, x_vars: &[Fq], gamma: Fq) -> Fq {
        assert_eq!(x_vars.len(), self.matrix.num_constraint_vars);

        let mut result = Fq::zero();
        let mut gamma_power = gamma;

        for constraint in self.constraints.iter() {
            let constraint_eval = self.evaluate_constraint(constraint, x_vars);
            result += gamma_power * constraint_eval;
            gamma_power *= gamma;
        }

        result
    }

    /// Evaluate the full constraint system at a point (for testing)
    ///
    /// Point structure: [x_vars, s_vars]
    /// Returns F(x, s) where s selects which constraint to evaluate
    pub fn evaluate(&self, point: &[Fq]) -> Fq {
        let num_x_vars = self.matrix.num_constraint_vars;
        let num_s_vars = self.matrix.num_s_vars;

        assert_eq!(point.len(), num_x_vars + num_s_vars);

        // Split point: [x_vars, s_vars]
        let (x_vars, s_vars) = point.split_at(num_x_vars);

        // We only evaluate constraints here, not the matrix rows
        // So we need to map s to constraint index
        let num_constraint_bits = (self.num_constraints() as f64).log2().ceil() as usize;
        let num_constraints_padded = 1 << num_constraint_bits;

        let mut result = Fq::zero();

        // For each constraint, check if s selects it
        for constraint in self.constraints.iter() {
            // s encodes both poly type and constraint index
            // For constraint evaluation, we treat all poly types of a constraint the same
            let constraint_padded_idx = constraint.constraint_index;

            // Check all 4 row indices that correspond to this constraint
            for &poly_type in PolyType::all() {
                let row_idx = (poly_type as usize) * num_constraints_padded + constraint_padded_idx;
                let row_binary = index_to_binary::<Fq>(row_idx, num_s_vars);
                let eq_eval = EqPolynomial::mle(&row_binary, s_vars);

                let constraint_eval = self.evaluate_constraint(constraint, x_vars);
                result += eq_eval * constraint_eval;
            }
        }

        result
    }

    /// Evaluate a single constraint C_i(x) using the matrix layout.
    fn evaluate_constraint(&self, constraint: &MatrixConstraint, x: &[Fq]) -> Fq {
        let idx = constraint.constraint_index;

        match constraint.constraint_type {
            ConstraintType::GtExp => {
                // For packed GT exp, g(x) only depends on element variables (high 4 bits)
                // Data layout: index = x_elem * 128 + s (s in low 7 bits, x_elem in high 4 bits)
                // So for an 11-var point, element vars are x[7..11]
                let g_eval = if x.len() == 11 {
                    // Extract element variables (high 4 bits) and evaluate 4-var g
                    // Reverse for big-endian convention used by DensePolynomial::evaluate
                    let x_elem_reversed: Vec<Fq> = x[7..11].iter().rev().copied().collect();
                    let g_4var = get_g_mle();
                    DensePolynomial::new(g_4var).evaluate(&x_elem_reversed)
                } else {
                    let x_reversed: Vec<Fq> = x.iter().rev().copied().collect();
                    self.g_poly.evaluate(&x_reversed)
                };

                let rho_prev_row = self.matrix.row_index(PolyType::RhoPrev, idx);
                let quotient_row = self.matrix.row_index(PolyType::Quotient, idx);

                let rho_prev = self.matrix.evaluate_row(rho_prev_row, x);
                let quotient = self.matrix.evaluate_row(quotient_row, x);

                // Compute rho_next on the fly from packed witness
                // During constraint evaluation, we use the precomputed rho_next_packed
                // Need to find GT exp witness index (count GtExp constraints before this one)
                let gt_exp_idx = self
                    .constraints
                    .iter()
                    .take(idx)
                    .filter(|c| matches!(c.constraint_type, ConstraintType::GtExp))
                    .count();
                let packed = &self.gt_exp_witnesses[gt_exp_idx];
                // Reverse for big-endian convention used by DensePolynomial::evaluate
                let x_reversed: Vec<Fq> = x.iter().rev().copied().collect();
                let rho_curr =
                    DensePolynomial::new(packed.rho_next_packed.clone()).evaluate(&x_reversed);
                let base_eval =
                    DensePolynomial::new(packed.base_packed.clone()).evaluate(&x_reversed);
                let base2_eval =
                    DensePolynomial::new(packed.base2_packed.clone()).evaluate(&x_reversed);
                let base3_eval =
                    DensePolynomial::new(packed.base3_packed.clone()).evaluate(&x_reversed);
                let digit_lo_eval =
                    DensePolynomial::new(packed.digit_lo_packed.clone()).evaluate(&x_reversed);
                let digit_hi_eval =
                    DensePolynomial::new(packed.digit_hi_packed.clone()).evaluate(&x_reversed);

                let u = digit_lo_eval;
                let v = digit_hi_eval;
                let w0 = (Fq::one() - u) * (Fq::one() - v);
                let w1 = u * (Fq::one() - v);
                let w2 = (Fq::one() - u) * v;
                let w3 = u * v;
                let base_power = w0 + w1 * base_eval + w2 * base2_eval + w3 * base3_eval;
                let rho2 = rho_prev * rho_prev;
                let rho4 = rho2 * rho2;

                rho_curr - rho4 * base_power - quotient * g_eval
            }
            ConstraintType::GtMul => {
                // GT mul uses 4-var polynomials with ZERO PADDING to 11-var
                // Zero padding: data in low indices 0-15, zeros in 16-2047
                // So element variables are in the LOW 4 bits: x[0..4]
                // Reverse for big-endian convention used by DensePolynomial::evaluate
                let g_eval = if x.len() == 11 {
                    let x_elem_reversed: Vec<Fq> = x[0..4].iter().rev().copied().collect();
                    let g_4var = get_g_mle();
                    DensePolynomial::new(g_4var).evaluate(&x_elem_reversed)
                } else {
                    let x_reversed: Vec<Fq> = x.iter().rev().copied().collect();
                    self.g_poly.evaluate(&x_reversed)
                };

                let lhs_row = self.matrix.row_index(PolyType::MulLhs, idx);
                let rhs_row = self.matrix.row_index(PolyType::MulRhs, idx);
                let result_row = self.matrix.row_index(PolyType::MulResult, idx);
                let quotient_row = self.matrix.row_index(PolyType::MulQuotient, idx);

                let lhs_eval = self.matrix.evaluate_row(lhs_row, x);
                let rhs_eval = self.matrix.evaluate_row(rhs_row, x);
                let result_eval = self.matrix.evaluate_row(result_row, x);
                let quotient_eval = self.matrix.evaluate_row(quotient_row, x);

                // GT mul constraint: lhs * rhs - result - quotient * g
                lhs_eval * rhs_eval - result_eval - quotient_eval * g_eval
            }
            ConstraintType::G1ScalarMul { base_point } => {
                // G1 scalar multiplication constraint evaluation (debug/testing helper).
                // Matches the stage1 bit-dependent constraints, with bits treated as public inputs.

                // Get the row indices for our MLEs
                let x_a_row = self.matrix.row_index(PolyType::G1ScalarMulXA, idx);
                let y_a_row = self.matrix.row_index(PolyType::G1ScalarMulYA, idx);
                let x_t_row = self.matrix.row_index(PolyType::G1ScalarMulXT, idx);
                let y_t_row = self.matrix.row_index(PolyType::G1ScalarMulYT, idx);
                let x_a_next_row = self.matrix.row_index(PolyType::G1ScalarMulXANext, idx);
                let y_a_next_row = self.matrix.row_index(PolyType::G1ScalarMulYANext, idx);
                let ind_t_row = self.matrix.row_index(PolyType::G1ScalarMulTIndicator, idx);
                let ind_a_row = self.matrix.row_index(PolyType::G1ScalarMulAIndicator, idx);

                // Evaluate MLEs at point x
                let x_a = self.matrix.evaluate_row(x_a_row, x);
                let y_a = self.matrix.evaluate_row(y_a_row, x);
                let x_t = self.matrix.evaluate_row(x_t_row, x);
                let y_t = self.matrix.evaluate_row(y_t_row, x);
                let x_a_next = self.matrix.evaluate_row(x_a_next_row, x);
                let y_a_next = self.matrix.evaluate_row(y_a_next_row, x);
                let ind_t = self.matrix.evaluate_row(ind_t_row, x);
                let ind_a = self.matrix.evaluate_row(ind_a_row, x);

                // Extract base point coordinates
                let (x_p, y_p) = base_point;

                // Evaluate bit MLE at x using verifier-known public input scalar.
                // `x` is little-endian for the matrix, but public-bit evaluation expects big-endian.
                let g1_scalar_mul_idx = self
                    .constraints
                    .iter()
                    .take(idx)
                    .filter(|c| matches!(c.constraint_type, ConstraintType::G1ScalarMul { .. }))
                    .count();
                let x_be: Vec<Fq> = x.iter().rev().copied().collect();
                let bit = self.g1_scalar_mul_public_inputs[g1_scalar_mul_idx]
                    .evaluate_bit_mle::<Fq>(&x_be);

                // C1: Doubling x-coordinate constraint
                // 4y_A^2(x_T + 2x_A) - 9x_A^4 = 0
                let c1 = {
                    let four = Fq::from(4u64);
                    let two = Fq::from(2u64);
                    let nine = Fq::from(9u64);

                    let y_a_sq = y_a * y_a;
                    let x_a_sq = x_a * x_a;
                    let x_a_fourth = x_a_sq * x_a_sq;

                    four * y_a_sq * (x_t + two * x_a) - nine * x_a_fourth
                };

                // C2: Doubling y-coordinate constraint
                // 3x_A^2(x_T - x_A) + 2y_A(y_T + y_A) = 0
                let c2 = {
                    let three = Fq::from(3u64);
                    let two = Fq::from(2u64);

                    let x_a_sq = x_a * x_a;
                    three * x_a_sq * (x_t - x_a) + two * y_a * (y_t + y_a)
                };

                // C3/C4: bit-dependent conditional add with T=O special case (matches stage1).
                let c3 = {
                    let one = Fq::one();
                    let c3_skip = (one - bit) * (x_a_next - x_t);
                    let c3_infinity = bit * ind_t * (x_a_next - x_p);
                    let x_diff = x_p - x_t;
                    let y_diff = y_p - y_t;
                    let chord_x = (x_a_next + x_t + x_p) * x_diff * x_diff - y_diff * y_diff;
                    let c3_add = bit * (one - ind_t) * chord_x;
                    c3_skip + c3_infinity + c3_add
                };

                let c4 = {
                    let one = Fq::one();
                    let c4_skip = (one - bit) * (y_a_next - y_t);
                    let c4_infinity = bit * ind_t * (y_a_next - y_p);
                    let x_diff = x_p - x_t;
                    let y_diff = y_p - y_t;
                    let chord_y = (y_a_next + y_t) * x_diff - y_diff * (x_t - x_a_next);
                    let c4_add = bit * (one - ind_t) * chord_y;
                    c4_skip + c4_infinity + c4_add
                };

                // C5/C6: indicator consistency (matches stage1).
                // C5: ind_A * (1 - ind_T) = 0
                let c5 = ind_a * (Fq::one() - ind_t);
                // C6: ind_T => (x_T, y_T) = (0,0), implemented as two field-independent checks
                let c6_x = ind_t * x_t;
                let c6_y = ind_t * y_t;

                // Return aggregate (should be 0 for honest witnesses).
                c1 + c2 + c3 + c4 + c5 + c6_x + c6_y
            }
            ConstraintType::G2ScalarMul { base_point } => {
                // G2 scalar multiplication constraint evaluation (over Fq2, coords split into c0/c1).

                // Row indices
                let x_a_c0_row = self.matrix.row_index(PolyType::G2ScalarMulXAC0, idx);
                let x_a_c1_row = self.matrix.row_index(PolyType::G2ScalarMulXAC1, idx);
                let y_a_c0_row = self.matrix.row_index(PolyType::G2ScalarMulYAC0, idx);
                let y_a_c1_row = self.matrix.row_index(PolyType::G2ScalarMulYAC1, idx);
                let x_t_c0_row = self.matrix.row_index(PolyType::G2ScalarMulXTC0, idx);
                let x_t_c1_row = self.matrix.row_index(PolyType::G2ScalarMulXTC1, idx);
                let y_t_c0_row = self.matrix.row_index(PolyType::G2ScalarMulYTC0, idx);
                let y_t_c1_row = self.matrix.row_index(PolyType::G2ScalarMulYTC1, idx);
                let x_a_next_c0_row = self.matrix.row_index(PolyType::G2ScalarMulXANextC0, idx);
                let x_a_next_c1_row = self.matrix.row_index(PolyType::G2ScalarMulXANextC1, idx);
                let y_a_next_c0_row = self.matrix.row_index(PolyType::G2ScalarMulYANextC0, idx);
                let y_a_next_c1_row = self.matrix.row_index(PolyType::G2ScalarMulYANextC1, idx);
                let ind_t_row = self.matrix.row_index(PolyType::G2ScalarMulTIndicator, idx);
                let ind_a_row = self.matrix.row_index(PolyType::G2ScalarMulAIndicator, idx);

                // Evaluate at x
                let x_a_c0 = self.matrix.evaluate_row(x_a_c0_row, x);
                let x_a_c1 = self.matrix.evaluate_row(x_a_c1_row, x);
                let y_a_c0 = self.matrix.evaluate_row(y_a_c0_row, x);
                let y_a_c1 = self.matrix.evaluate_row(y_a_c1_row, x);
                let x_t_c0 = self.matrix.evaluate_row(x_t_c0_row, x);
                let x_t_c1 = self.matrix.evaluate_row(x_t_c1_row, x);
                let y_t_c0 = self.matrix.evaluate_row(y_t_c0_row, x);
                let y_t_c1 = self.matrix.evaluate_row(y_t_c1_row, x);
                let x_a_next_c0 = self.matrix.evaluate_row(x_a_next_c0_row, x);
                let x_a_next_c1 = self.matrix.evaluate_row(x_a_next_c1_row, x);
                let y_a_next_c0 = self.matrix.evaluate_row(y_a_next_c0_row, x);
                let y_a_next_c1 = self.matrix.evaluate_row(y_a_next_c1_row, x);
                let ind_t = self.matrix.evaluate_row(ind_t_row, x);
                let ind_a = self.matrix.evaluate_row(ind_a_row, x);

                let x_a = Fq2::new(x_a_c0, x_a_c1);
                let y_a = Fq2::new(y_a_c0, y_a_c1);
                let x_t = Fq2::new(x_t_c0, x_t_c1);
                let y_t = Fq2::new(y_t_c0, y_t_c1);
                let x_a_next = Fq2::new(x_a_next_c0, x_a_next_c1);
                let y_a_next = Fq2::new(y_a_next_c0, y_a_next_c1);

                let (x_p, y_p) = base_point;

                // Evaluate bit at x from public inputs (big-endian point expected).
                let g2_scalar_mul_idx = self
                    .constraints
                    .iter()
                    .take(idx)
                    .filter(|c| matches!(c.constraint_type, ConstraintType::G2ScalarMul { .. }))
                    .count();
                let x_be: Vec<Fq> = x.iter().rev().copied().collect();
                let bit = self.g2_scalar_mul_public_inputs[g2_scalar_mul_idx]
                    .evaluate_bit_mle::<Fq>(&x_be);

                let fq2_from_fq = |v: Fq| Fq2::new(v, Fq::zero());

                // C1
                let c1 = {
                    let four = fq2_from_fq(Fq::from(4u64));
                    let two = fq2_from_fq(Fq::from(2u64));
                    let nine = fq2_from_fq(Fq::from(9u64));

                    let y_a_sq = y_a * y_a;
                    let x_a_sq = x_a * x_a;
                    let x_a_fourth = x_a_sq * x_a_sq;

                    four * y_a_sq * (x_t + two * x_a) - nine * x_a_fourth
                };

                // C2
                let c2 = {
                    let three = fq2_from_fq(Fq::from(3u64));
                    let two = fq2_from_fq(Fq::from(2u64));

                    let x_a_sq = x_a * x_a;
                    three * x_a_sq * (x_t - x_a) + two * y_a * (y_t + y_a)
                };

                // C3 (bit/indicator dependent)
                let c3 = {
                    let one = Fq2::one();
                    let bit2 = fq2_from_fq(bit);
                    let ind_t2 = fq2_from_fq(ind_t);

                    let c3_skip = (one - bit2) * (x_a_next - x_t);
                    let c3_infty = bit2 * ind_t2 * (x_a_next - x_p);

                    let x_diff = x_p - x_t;
                    let y_diff = y_p - y_t;
                    let chord_x = (x_a_next + x_t + x_p) * x_diff * x_diff - y_diff * y_diff;
                    let c3_add = bit2 * (one - ind_t2) * chord_x;

                    c3_skip + c3_infty + c3_add
                };

                // C4 (bit/indicator dependent)
                let c4 = {
                    let one = Fq2::one();
                    let bit2 = fq2_from_fq(bit);
                    let ind_t2 = fq2_from_fq(ind_t);

                    let c4_skip = (one - bit2) * (y_a_next - y_t);
                    let c4_infty = bit2 * ind_t2 * (y_a_next - y_p);

                    let x_diff = x_p - x_t;
                    let y_diff = y_p - y_t;
                    let chord_y = (y_a_next + y_t) * x_diff - y_diff * (x_t - x_a_next);
                    let c4_add = bit2 * (one - ind_t2) * chord_y;

                    c4_skip + c4_infty + c4_add
                };

                // Base-field constraints
                let c6 = ind_a * (Fq::one() - ind_t);
                let c7_xt_c0 = ind_t * x_t.c0;
                let c7_xt_c1 = ind_t * x_t.c1;
                let c7_yt_c0 = ind_t * y_t.c0;
                let c7_yt_c1 = ind_t * y_t.c1;

                // Return a simple aggregate (should be 0 for honest witnesses).
                c1.c0
                    + c1.c1
                    + c2.c0
                    + c2.c1
                    + c3.c0
                    + c3.c1
                    + c4.c0
                    + c4.c1
                    + c6
                    + c7_xt_c0
                    + c7_xt_c1
                    + c7_yt_c0
                    + c7_yt_c1
            }
            ConstraintType::G1Add => {
                use crate::zkvm::recursion::g1::addition::G1AddValues;
                let delta = Fq::from(7u64);

                let x_p_row = self.matrix.row_index(PolyType::G1AddXP, idx);
                let y_p_row = self.matrix.row_index(PolyType::G1AddYP, idx);
                let ind_p_row = self.matrix.row_index(PolyType::G1AddPIndicator, idx);
                let x_q_row = self.matrix.row_index(PolyType::G1AddXQ, idx);
                let y_q_row = self.matrix.row_index(PolyType::G1AddYQ, idx);
                let ind_q_row = self.matrix.row_index(PolyType::G1AddQIndicator, idx);
                let x_r_row = self.matrix.row_index(PolyType::G1AddXR, idx);
                let y_r_row = self.matrix.row_index(PolyType::G1AddYR, idx);
                let ind_r_row = self.matrix.row_index(PolyType::G1AddRIndicator, idx);
                let lambda_row = self.matrix.row_index(PolyType::G1AddLambda, idx);
                let inv_dx_row = self.matrix.row_index(PolyType::G1AddInvDeltaX, idx);
                let is_double_row = self.matrix.row_index(PolyType::G1AddIsDouble, idx);
                let is_inverse_row = self.matrix.row_index(PolyType::G1AddIsInverse, idx);

                let values = G1AddValues::<Fq> {
                    x_p: self.matrix.evaluate_row(x_p_row, x),
                    y_p: self.matrix.evaluate_row(y_p_row, x),
                    ind_p: self.matrix.evaluate_row(ind_p_row, x),
                    x_q: self.matrix.evaluate_row(x_q_row, x),
                    y_q: self.matrix.evaluate_row(y_q_row, x),
                    ind_q: self.matrix.evaluate_row(ind_q_row, x),
                    x_r: self.matrix.evaluate_row(x_r_row, x),
                    y_r: self.matrix.evaluate_row(y_r_row, x),
                    ind_r: self.matrix.evaluate_row(ind_r_row, x),
                    lambda: self.matrix.evaluate_row(lambda_row, x),
                    inv_delta_x: self.matrix.evaluate_row(inv_dx_row, x),
                    is_double: self.matrix.evaluate_row(is_double_row, x),
                    is_inverse: self.matrix.evaluate_row(is_inverse_row, x),
                };

                values.eval_constraint(delta)
            }
            ConstraintType::G2Add => {
                use crate::zkvm::recursion::g2::addition::G2AddValues;
                let delta = Fq::from(7u64);

                let x_p_c0_row = self.matrix.row_index(PolyType::G2AddXPC0, idx);
                let x_p_c1_row = self.matrix.row_index(PolyType::G2AddXPC1, idx);
                let y_p_c0_row = self.matrix.row_index(PolyType::G2AddYPC0, idx);
                let y_p_c1_row = self.matrix.row_index(PolyType::G2AddYPC1, idx);
                let ind_p_row = self.matrix.row_index(PolyType::G2AddPIndicator, idx);
                let x_q_c0_row = self.matrix.row_index(PolyType::G2AddXQC0, idx);
                let x_q_c1_row = self.matrix.row_index(PolyType::G2AddXQC1, idx);
                let y_q_c0_row = self.matrix.row_index(PolyType::G2AddYQC0, idx);
                let y_q_c1_row = self.matrix.row_index(PolyType::G2AddYQC1, idx);
                let ind_q_row = self.matrix.row_index(PolyType::G2AddQIndicator, idx);
                let x_r_c0_row = self.matrix.row_index(PolyType::G2AddXRC0, idx);
                let x_r_c1_row = self.matrix.row_index(PolyType::G2AddXRC1, idx);
                let y_r_c0_row = self.matrix.row_index(PolyType::G2AddYRC0, idx);
                let y_r_c1_row = self.matrix.row_index(PolyType::G2AddYRC1, idx);
                let ind_r_row = self.matrix.row_index(PolyType::G2AddRIndicator, idx);
                let lambda_c0_row = self.matrix.row_index(PolyType::G2AddLambdaC0, idx);
                let lambda_c1_row = self.matrix.row_index(PolyType::G2AddLambdaC1, idx);
                let inv_dx_c0_row = self.matrix.row_index(PolyType::G2AddInvDeltaXC0, idx);
                let inv_dx_c1_row = self.matrix.row_index(PolyType::G2AddInvDeltaXC1, idx);
                let is_double_row = self.matrix.row_index(PolyType::G2AddIsDouble, idx);
                let is_inverse_row = self.matrix.row_index(PolyType::G2AddIsInverse, idx);

                let values = G2AddValues::<Fq> {
                    x_p_c0: self.matrix.evaluate_row(x_p_c0_row, x),
                    x_p_c1: self.matrix.evaluate_row(x_p_c1_row, x),
                    y_p_c0: self.matrix.evaluate_row(y_p_c0_row, x),
                    y_p_c1: self.matrix.evaluate_row(y_p_c1_row, x),
                    ind_p: self.matrix.evaluate_row(ind_p_row, x),
                    x_q_c0: self.matrix.evaluate_row(x_q_c0_row, x),
                    x_q_c1: self.matrix.evaluate_row(x_q_c1_row, x),
                    y_q_c0: self.matrix.evaluate_row(y_q_c0_row, x),
                    y_q_c1: self.matrix.evaluate_row(y_q_c1_row, x),
                    ind_q: self.matrix.evaluate_row(ind_q_row, x),
                    x_r_c0: self.matrix.evaluate_row(x_r_c0_row, x),
                    x_r_c1: self.matrix.evaluate_row(x_r_c1_row, x),
                    y_r_c0: self.matrix.evaluate_row(y_r_c0_row, x),
                    y_r_c1: self.matrix.evaluate_row(y_r_c1_row, x),
                    ind_r: self.matrix.evaluate_row(ind_r_row, x),
                    lambda_c0: self.matrix.evaluate_row(lambda_c0_row, x),
                    lambda_c1: self.matrix.evaluate_row(lambda_c1_row, x),
                    inv_delta_x_c0: self.matrix.evaluate_row(inv_dx_c0_row, x),
                    inv_delta_x_c1: self.matrix.evaluate_row(inv_dx_c1_row, x),
                    is_double: self.matrix.evaluate_row(is_double_row, x),
                    is_inverse: self.matrix.evaluate_row(is_inverse_row, x),
                };

                values.eval_constraint(delta)
            }
            #[cfg(feature = "experimental-pairing-recursion")]
            ConstraintType::MultiMillerLoop => {
                unimplemented!(
                    "ConstraintSystem::debug_constraint_eval for MultiMillerLoop will be updated \
                     after the MultiMillerLoop sumcheck refactor (g/selectors become verifier-derived)"
                )
            }
        }
    }

    #[cfg(test)]
    pub fn verify_constraints_are_zero(&self) {
        // Verify that each constraint evaluates to 0 over the entire hypercube
        let num_x_points = 1 << self.matrix.num_constraint_vars;

        for constraint in &self.constraints {
            let idx = constraint.constraint_index;

            for x_val in 0..num_x_points {
                let mut x_binary = Vec::with_capacity(self.matrix.num_constraint_vars);
                let mut x = x_val;
                for _ in 0..self.matrix.num_constraint_vars {
                    x_binary.push(if x & 1 == 1 { Fq::one() } else { Fq::zero() });
                    x >>= 1;
                }

                let constraint_eval = self.evaluate_constraint(constraint, &x_binary);

                assert!(
                    constraint_eval == Fq::zero(),
                    "Constraint {idx} failed at x={x_binary:?}: got {constraint_eval}, expected 0"
                );
            }
        }
    }
}

// Manual serialization implementations for enums

use ark_serialize::{Compress, SerializationError, Valid};

impl CanonicalSerialize for PolyType {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        (*self as u8).serialize_with_mode(writer, _compress)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        1 // u8 size
    }
}

impl CanonicalDeserialize for PolyType {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        reader: R,
        _compress: Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        let val = u8::deserialize_with_mode(reader, _compress, _validate)?;
        match val {
            0 => Ok(PolyType::RhoPrev),
            1 => Ok(PolyType::Quotient),
            2 => Ok(PolyType::MulLhs),
            3 => Ok(PolyType::MulRhs),
            4 => Ok(PolyType::MulResult),
            5 => Ok(PolyType::MulQuotient),
            6 => Ok(PolyType::G1ScalarMulXA),
            7 => Ok(PolyType::G1ScalarMulYA),
            8 => Ok(PolyType::G1ScalarMulXT),
            9 => Ok(PolyType::G1ScalarMulYT),
            10 => Ok(PolyType::G1ScalarMulXANext),
            11 => Ok(PolyType::G1ScalarMulYANext),
            12 => Ok(PolyType::G1ScalarMulTIndicator),
            13 => Ok(PolyType::G1ScalarMulAIndicator),
            14 => Ok(PolyType::G1ScalarMulBit),
            15 => Ok(PolyType::G2ScalarMulXAC0),
            16 => Ok(PolyType::G2ScalarMulXAC1),
            17 => Ok(PolyType::G2ScalarMulYAC0),
            18 => Ok(PolyType::G2ScalarMulYAC1),
            19 => Ok(PolyType::G2ScalarMulXTC0),
            20 => Ok(PolyType::G2ScalarMulXTC1),
            21 => Ok(PolyType::G2ScalarMulYTC0),
            22 => Ok(PolyType::G2ScalarMulYTC1),
            23 => Ok(PolyType::G2ScalarMulXANextC0),
            24 => Ok(PolyType::G2ScalarMulXANextC1),
            25 => Ok(PolyType::G2ScalarMulYANextC0),
            26 => Ok(PolyType::G2ScalarMulYANextC1),
            27 => Ok(PolyType::G2ScalarMulTIndicator),
            28 => Ok(PolyType::G2ScalarMulAIndicator),
            29 => Ok(PolyType::G2ScalarMulBit),
            30 => Ok(PolyType::G1AddXP),
            31 => Ok(PolyType::G1AddYP),
            32 => Ok(PolyType::G1AddPIndicator),
            33 => Ok(PolyType::G1AddXQ),
            34 => Ok(PolyType::G1AddYQ),
            35 => Ok(PolyType::G1AddQIndicator),
            36 => Ok(PolyType::G1AddXR),
            37 => Ok(PolyType::G1AddYR),
            38 => Ok(PolyType::G1AddRIndicator),
            39 => Ok(PolyType::G1AddLambda),
            40 => Ok(PolyType::G1AddInvDeltaX),
            41 => Ok(PolyType::G1AddIsDouble),
            42 => Ok(PolyType::G1AddIsInverse),
            43 => Ok(PolyType::G2AddXPC0),
            44 => Ok(PolyType::G2AddXPC1),
            45 => Ok(PolyType::G2AddYPC0),
            46 => Ok(PolyType::G2AddYPC1),
            47 => Ok(PolyType::G2AddPIndicator),
            48 => Ok(PolyType::G2AddXQC0),
            49 => Ok(PolyType::G2AddXQC1),
            50 => Ok(PolyType::G2AddYQC0),
            51 => Ok(PolyType::G2AddYQC1),
            52 => Ok(PolyType::G2AddQIndicator),
            53 => Ok(PolyType::G2AddXRC0),
            54 => Ok(PolyType::G2AddXRC1),
            55 => Ok(PolyType::G2AddYRC0),
            56 => Ok(PolyType::G2AddYRC1),
            57 => Ok(PolyType::G2AddRIndicator),
            58 => Ok(PolyType::G2AddLambdaC0),
            59 => Ok(PolyType::G2AddLambdaC1),
            60 => Ok(PolyType::G2AddInvDeltaXC0),
            61 => Ok(PolyType::G2AddInvDeltaXC1),
            62 => Ok(PolyType::G2AddIsDouble),
            63 => Ok(PolyType::G2AddIsInverse),
            #[cfg(feature = "experimental-pairing-recursion")]
            64 => Ok(PolyType::MultiMillerLoopF),
            #[cfg(feature = "experimental-pairing-recursion")]
            65 => Ok(PolyType::MultiMillerLoopFNext),
            #[cfg(feature = "experimental-pairing-recursion")]
            66 => Ok(PolyType::MultiMillerLoopQuotient),
            #[cfg(feature = "experimental-pairing-recursion")]
            67 => Ok(PolyType::MultiMillerLoopTXC0),
            #[cfg(feature = "experimental-pairing-recursion")]
            68 => Ok(PolyType::MultiMillerLoopTXC1),
            #[cfg(feature = "experimental-pairing-recursion")]
            69 => Ok(PolyType::MultiMillerLoopTYC0),
            #[cfg(feature = "experimental-pairing-recursion")]
            70 => Ok(PolyType::MultiMillerLoopTYC1),
            #[cfg(feature = "experimental-pairing-recursion")]
            71 => Ok(PolyType::MultiMillerLoopTXC0Next),
            #[cfg(feature = "experimental-pairing-recursion")]
            72 => Ok(PolyType::MultiMillerLoopTXC1Next),
            #[cfg(feature = "experimental-pairing-recursion")]
            73 => Ok(PolyType::MultiMillerLoopTYC0Next),
            #[cfg(feature = "experimental-pairing-recursion")]
            74 => Ok(PolyType::MultiMillerLoopTYC1Next),
            #[cfg(feature = "experimental-pairing-recursion")]
            75 => Ok(PolyType::MultiMillerLoopLambdaC0),
            #[cfg(feature = "experimental-pairing-recursion")]
            76 => Ok(PolyType::MultiMillerLoopLambdaC1),
            #[cfg(feature = "experimental-pairing-recursion")]
            77 => Ok(PolyType::MultiMillerLoopInvDeltaXC0),
            #[cfg(feature = "experimental-pairing-recursion")]
            78 => Ok(PolyType::MultiMillerLoopInvDeltaXC1),
            #[cfg(feature = "experimental-pairing-recursion")]
            79 => Ok(PolyType::MultiMillerLoopXP),
            #[cfg(feature = "experimental-pairing-recursion")]
            80 => Ok(PolyType::MultiMillerLoopYP),
            #[cfg(feature = "experimental-pairing-recursion")]
            81 => Ok(PolyType::MultiMillerLoopXQC0),
            #[cfg(feature = "experimental-pairing-recursion")]
            82 => Ok(PolyType::MultiMillerLoopXQC1),
            #[cfg(feature = "experimental-pairing-recursion")]
            83 => Ok(PolyType::MultiMillerLoopYQC0),
            #[cfg(feature = "experimental-pairing-recursion")]
            84 => Ok(PolyType::MultiMillerLoopYQC1),
            #[cfg(feature = "experimental-pairing-recursion")]
            85 => Ok(PolyType::MultiMillerLoopIsDouble),
            #[cfg(feature = "experimental-pairing-recursion")]
            86 => Ok(PolyType::MultiMillerLoopIsAdd),
            #[cfg(feature = "experimental-pairing-recursion")]
            87 => Ok(PolyType::MultiMillerLoopLVal),
            #[cfg(feature = "experimental-pairing-recursion")]
            88 => Ok(PolyType::MultiMillerLoopInvTwoYC0),
            #[cfg(feature = "experimental-pairing-recursion")]
            89 => Ok(PolyType::MultiMillerLoopInvTwoYC1),
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl Valid for PolyType {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for ConstraintType {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            ConstraintType::GtExp => {
                0u8.serialize_with_mode(&mut writer, compress)?;
            }
            ConstraintType::GtMul => {
                1u8.serialize_with_mode(&mut writer, compress)?;
            }
            ConstraintType::G1ScalarMul { base_point } => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                base_point.0.serialize_with_mode(&mut writer, compress)?;
                base_point.1.serialize_with_mode(&mut writer, compress)?;
            }
            ConstraintType::G2ScalarMul { base_point } => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                base_point.0.serialize_with_mode(&mut writer, compress)?;
                base_point.1.serialize_with_mode(&mut writer, compress)?;
            }
            ConstraintType::G1Add => {
                4u8.serialize_with_mode(&mut writer, compress)?;
            }
            ConstraintType::G2Add => {
                5u8.serialize_with_mode(&mut writer, compress)?;
            }
            #[cfg(feature = "experimental-pairing-recursion")]
            ConstraintType::MultiMillerLoop => {
                6u8.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{DoryCommitmentScheme, DoryContext, DoryGlobals},
        },
        dense_mlpoly::DensePolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    };
    use ark_bn254::Fr;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_dory_witness_constraint_evaluation() {
        use ark_ff::UniformRand;
        use rand::thread_rng;

        DoryGlobals::reset();
        DoryGlobals::initialize_context(1 << 2, 1 << 2, DoryContext::Main, None);
        let num_vars = 4;
        let mut rng = thread_rng();

        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        let coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));
        let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        let mut prover_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &point,
            Some(hint),
            &mut prover_transcript,
        );

        let evaluation = PolynomialEvaluation::evaluate(&poly, &point);
        let mut extract_transcript = crate::transcripts::Blake2bTranscript::new(b"test");

        let _start = std::time::Instant::now();

        let (system, hints) = ConstraintSystem::new(
            &proof,
            &verifier_setup,
            &mut extract_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("System creation should succeed");

        let _elapsed = _start.elapsed();
        // Count constraints by type
        let mut gt_exp_count = 0;
        let mut gt_mul_count = 0;
        let mut g1_scalar_mul_count = 0;
        let mut g1_add_count = 0;
        let mut g2_add_count = 0;

        for constraint in &system.constraints {
            match &constraint.constraint_type {
                ConstraintType::GtExp => gt_exp_count += 1,
                ConstraintType::GtMul => gt_mul_count += 1,
                ConstraintType::G1ScalarMul { .. } => g1_scalar_mul_count += 1,
                ConstraintType::G2ScalarMul { .. } => {}
                ConstraintType::G1Add => g1_add_count += 1,
                ConstraintType::G2Add => g2_add_count += 1,
                #[cfg(feature = "experimental-pairing-recursion")]
                ConstraintType::MultiMillerLoop => {}
            }
        }

        assert!(
            g1_add_count > 0,
            "Expected at least one G1Add constraint from Dory witness_gen"
        );
        assert!(
            g2_add_count > 0,
            "Expected at least one G2Add constraint from Dory witness_gen"
        );

        let _ = (
            gt_exp_count,
            gt_mul_count,
            g1_scalar_mul_count,
            g1_add_count,
            g2_add_count,
        );
        // Instead of evaluating the full system, just evaluate constraints at random points
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let num_x_vars = system.matrix.num_constraint_vars;

        // For each constraint, test it at random x-variable points
        for (idx, constraint) in system.constraints.iter().enumerate() {
            // Test this constraint at 5 random points
            for _trial in 0..5 {
                let mut x_point = Vec::with_capacity(num_x_vars);
                for _ in 0..num_x_vars {
                    x_point.push(if rng.gen_bool(0.5) {
                        Fq::one()
                    } else {
                        Fq::zero()
                    });
                }

                let eval = system.evaluate_constraint(constraint, &x_point);
                assert_eq!(
                    eval,
                    Fq::zero(),
                    "Constraint {idx} should evaluate to 0 at boolean points"
                );
            }
        }
        let mut verify_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        DoryCommitmentScheme::verify_with_hint(
            &proof,
            &verifier_setup,
            &mut verify_transcript,
            &point,
            &evaluation,
            &commitment,
            &hints,
        )
        .expect("Verification with hint should succeed");

        let mut verify_transcript_no_hint = crate::transcripts::Blake2bTranscript::new(b"test");
        DoryCommitmentScheme::verify(
            &proof,
            &verifier_setup,
            &mut verify_transcript_no_hint,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("Verification without hint should also succeed");
    }
}

// =============================================================================
// Witness Types for Constraint Extraction
// =============================================================================
//
// These intermediate types are returned by the `extract_*_constraints` methods.
// They bundle the constraint polynomials with per-instance data (like base_point)
// needed by the prover.

/// Witness data for a single G1 scalar multiplication constraint.
#[derive(Clone, Debug)]
pub struct G1ScalarMulWitness<F> {
    pub constraint_index: usize,
    pub base_point: (Fq, Fq),
    pub x_a: Vec<F>,
    pub y_a: Vec<F>,
    pub x_t: Vec<F>,
    pub y_t: Vec<F>,
    pub x_a_next: Vec<F>,
    pub y_a_next: Vec<F>,
    pub t_indicator: Vec<F>,
    pub a_indicator: Vec<F>,
}

/// Witness data for a single G2 scalar multiplication constraint.
#[derive(Clone, Debug)]
pub struct G2ScalarMulWitness<F> {
    pub constraint_index: usize,
    pub base_point: (Fq2, Fq2),
    pub x_a_c0: Vec<F>,
    pub x_a_c1: Vec<F>,
    pub y_a_c0: Vec<F>,
    pub y_a_c1: Vec<F>,
    pub x_t_c0: Vec<F>,
    pub x_t_c1: Vec<F>,
    pub y_t_c0: Vec<F>,
    pub y_t_c1: Vec<F>,
    pub x_a_next_c0: Vec<F>,
    pub x_a_next_c1: Vec<F>,
    pub y_a_next_c0: Vec<F>,
    pub y_a_next_c1: Vec<F>,
    pub t_indicator: Vec<F>,
    pub a_indicator: Vec<F>,
}
