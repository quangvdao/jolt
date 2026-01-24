//! Generic jagged-to-dense polynomial bijection transform
//!
//! This module implements a generic bijection between sparse "jagged" multilinear polynomials
//! and dense representations, following the approach from "Jagged Polynomial Commitments".
//!
//! The "jaggedness" in this context comes from polynomials having different numbers of variables
//! (e.g., 4-var vs 8-var), requiring padding in the sparse representation. The dense representation
//! excludes these redundant padded values to achieve compression.

use crate::field::JoltField;
use ark_bn254::Fq;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::zkvm::recursion::constraints::system::{
    ConstraintSystem, ConstraintType, MatrixConstraint, PolyType,
};

/// Core trait for jagged transforms with variable-count based heights
pub trait JaggedTransform<F: JoltField> {
    /// Get the polynomial index (row in sparse representation) for a dense index
    fn row(&self, dense_idx: usize) -> usize;

    /// Get the evaluation index (col in sparse representation) for a dense index
    fn col(&self, dense_idx: usize) -> usize;

    /// Map sparse (row, col) to dense index, returns None if outside bounds
    fn sparse_to_dense(&self, row: usize, col: usize) -> Option<usize>;

    /// Total non-redundant entries in dense representation
    fn dense_size(&self) -> usize;

    /// Get native variable count for a polynomial at given row
    fn poly_num_vars(&self, poly_idx: usize) -> usize;
}

/// Represents a polynomial with a specific number of variables
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct JaggedPolynomial {
    /// Number of variables (e.g., 4 or 8)
    pub num_vars: usize,
    /// Native size: 2^num_vars
    pub native_size: usize,
}

impl JaggedPolynomial {
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            native_size: 1 << num_vars,
        }
    }
}

/// Main bijection implementation for variable-count based jaggedness
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct VarCountJaggedBijection {
    /// Information for each polynomial
    polynomials: Vec<JaggedPolynomial>,
    /// Cumulative sizes for dense indexing
    cumulative_sizes: Vec<usize>,
    /// Total dense size
    total_size: usize,
}

impl VarCountJaggedBijection {
    /// Create a new bijection from polynomial specifications
    pub fn new(polynomials: Vec<JaggedPolynomial>) -> Self {
        let mut cumulative_sizes = Vec::with_capacity(polynomials.len());
        let mut total = 0;

        for poly in &polynomials {
            total += poly.native_size;
            cumulative_sizes.push(total);
        }

        Self {
            polynomials,
            cumulative_sizes,
            total_size: total,
        }
    }

    /// Get the number of polynomials
    pub fn num_polynomials(&self) -> usize {
        self.polynomials.len()
    }

    /// Get the cumulative size at the given index
    pub fn cumulative_size(&self, idx: usize) -> usize {
        self.cumulative_sizes[idx]
    }

    /// Get the cumulative size before the given index (0 if idx is 0)
    pub fn cumulative_size_before(&self, idx: usize) -> usize {
        if idx == 0 {
            0
        } else {
            self.cumulative_sizes[idx - 1]
        }
    }
}

impl<F: JoltField> JaggedTransform<F> for VarCountJaggedBijection {
    fn row(&self, dense_idx: usize) -> usize {
        if dense_idx >= self.total_size {
            panic!(
                "Dense index {} out of bounds (total size: {})",
                dense_idx, self.total_size
            );
        }

        // Binary search to find which polynomial contains this index
        self.cumulative_sizes
            .binary_search(&(dense_idx + 1))
            .unwrap_or_else(|x| x)
    }

    fn col(&self, dense_idx: usize) -> usize {
        let poly_idx = <Self as JaggedTransform<F>>::row(self, dense_idx);
        let poly_start = if poly_idx == 0 {
            0
        } else {
            self.cumulative_sizes[poly_idx - 1]
        };
        dense_idx - poly_start
    }

    fn sparse_to_dense(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.polynomials.len() {
            return None;
        }

        let poly = &self.polynomials[row];
        if col >= poly.native_size {
            return None;
        }

        let poly_start = if row == 0 {
            0
        } else {
            self.cumulative_sizes[row - 1]
        };

        Some(poly_start + col)
    }

    fn dense_size(&self) -> usize {
        self.total_size
    }

    fn poly_num_vars(&self, poly_idx: usize) -> usize {
        self.polynomials
            .get(poly_idx)
            .map(|p| p.num_vars)
            .unwrap_or(0)
    }
}

/// Maps between polynomial indices and constraint system structure
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ConstraintMapping {
    /// Maps polynomial index to (constraint_idx, poly_type)
    poly_to_constraint: Vec<(usize, PolyType)>,
}

impl ConstraintMapping {
    /// Decode polynomial index to constraint index and polynomial type
    pub fn decode(&self, poly_idx: usize) -> (usize, PolyType) {
        self.poly_to_constraint[poly_idx]
    }

    /// Create from list of (constraint_idx, poly_type, num_vars) tuples
    pub fn from_tuples(polynomials: &[(usize, PolyType, usize)]) -> Self {
        let poly_to_constraint = polynomials
            .iter()
            .map(|(idx, poly_type, _)| (*idx, *poly_type))
            .collect();

        Self { poly_to_constraint }
    }

    /// Get the number of polynomials
    pub fn num_polynomials(&self) -> usize {
        self.poly_to_constraint.len()
    }
}

/// Builder for creating jagged bijection from constraint system
pub struct ConstraintSystemJaggedBuilder {
    /// List of (constraint_idx, poly_type, num_vars) for each polynomial
    pub polynomials: Vec<(usize, PolyType, usize)>,
}

impl ConstraintSystemJaggedBuilder {
    /// Create builder from constraint system constraints
    /// This matches the matrix layout where rows are organized by polynomial type first
    pub fn from_constraints(constraints: &[MatrixConstraint]) -> Self {
        let mut polynomials = Vec::new();

        // Get all polynomial types that are actually used
        let mut used_poly_types = std::collections::HashSet::new();
        for constraint in constraints.iter() {
            match &constraint.constraint_type {
                ConstraintType::GtExp => {
                    // Base, Bit, and RhoNext are not committed polynomials
                    used_poly_types.insert(PolyType::RhoPrev);
                    used_poly_types.insert(PolyType::Quotient);
                }
                ConstraintType::GtMul => {
                    used_poly_types.insert(PolyType::MulLhs);
                    used_poly_types.insert(PolyType::MulRhs);
                    used_poly_types.insert(PolyType::MulResult);
                    used_poly_types.insert(PolyType::MulQuotient);
                }
                ConstraintType::G1ScalarMul { .. } => {
                    used_poly_types.insert(PolyType::G1ScalarMulXA);
                    used_poly_types.insert(PolyType::G1ScalarMulYA);
                    used_poly_types.insert(PolyType::G1ScalarMulXT);
                    used_poly_types.insert(PolyType::G1ScalarMulYT);
                    used_poly_types.insert(PolyType::G1ScalarMulXANext);
                    used_poly_types.insert(PolyType::G1ScalarMulYANext);
                    used_poly_types.insert(PolyType::G1ScalarMulTIndicator);
                    used_poly_types.insert(PolyType::G1ScalarMulAIndicator);
                }
                ConstraintType::G2ScalarMul { .. } => {
                    used_poly_types.insert(PolyType::G2ScalarMulXAC0);
                    used_poly_types.insert(PolyType::G2ScalarMulXAC1);
                    used_poly_types.insert(PolyType::G2ScalarMulYAC0);
                    used_poly_types.insert(PolyType::G2ScalarMulYAC1);
                    used_poly_types.insert(PolyType::G2ScalarMulXTC0);
                    used_poly_types.insert(PolyType::G2ScalarMulXTC1);
                    used_poly_types.insert(PolyType::G2ScalarMulYTC0);
                    used_poly_types.insert(PolyType::G2ScalarMulYTC1);
                    used_poly_types.insert(PolyType::G2ScalarMulXANextC0);
                    used_poly_types.insert(PolyType::G2ScalarMulXANextC1);
                    used_poly_types.insert(PolyType::G2ScalarMulYANextC0);
                    used_poly_types.insert(PolyType::G2ScalarMulYANextC1);
                    used_poly_types.insert(PolyType::G2ScalarMulTIndicator);
                    used_poly_types.insert(PolyType::G2ScalarMulAIndicator);
                }
                ConstraintType::G1Add => {
                    used_poly_types.insert(PolyType::G1AddXP);
                    used_poly_types.insert(PolyType::G1AddYP);
                    used_poly_types.insert(PolyType::G1AddPIndicator);
                    used_poly_types.insert(PolyType::G1AddXQ);
                    used_poly_types.insert(PolyType::G1AddYQ);
                    used_poly_types.insert(PolyType::G1AddQIndicator);
                    used_poly_types.insert(PolyType::G1AddXR);
                    used_poly_types.insert(PolyType::G1AddYR);
                    used_poly_types.insert(PolyType::G1AddRIndicator);
                    used_poly_types.insert(PolyType::G1AddLambda);
                    used_poly_types.insert(PolyType::G1AddInvDeltaX);
                    used_poly_types.insert(PolyType::G1AddIsDouble);
                    used_poly_types.insert(PolyType::G1AddIsInverse);
                }
                ConstraintType::G2Add => {
                    used_poly_types.insert(PolyType::G2AddXPC0);
                    used_poly_types.insert(PolyType::G2AddXPC1);
                    used_poly_types.insert(PolyType::G2AddYPC0);
                    used_poly_types.insert(PolyType::G2AddYPC1);
                    used_poly_types.insert(PolyType::G2AddPIndicator);
                    used_poly_types.insert(PolyType::G2AddXQC0);
                    used_poly_types.insert(PolyType::G2AddXQC1);
                    used_poly_types.insert(PolyType::G2AddYQC0);
                    used_poly_types.insert(PolyType::G2AddYQC1);
                    used_poly_types.insert(PolyType::G2AddQIndicator);
                    used_poly_types.insert(PolyType::G2AddXRC0);
                    used_poly_types.insert(PolyType::G2AddXRC1);
                    used_poly_types.insert(PolyType::G2AddYRC0);
                    used_poly_types.insert(PolyType::G2AddYRC1);
                    used_poly_types.insert(PolyType::G2AddRIndicator);
                    used_poly_types.insert(PolyType::G2AddLambdaC0);
                    used_poly_types.insert(PolyType::G2AddLambdaC1);
                    used_poly_types.insert(PolyType::G2AddInvDeltaXC0);
                    used_poly_types.insert(PolyType::G2AddInvDeltaXC1);
                    used_poly_types.insert(PolyType::G2AddIsDouble);
                    used_poly_types.insert(PolyType::G2AddIsInverse);
                }
                #[cfg(feature = "experimental-pairing-recursion")]
                ConstraintType::MultiMillerLoop => {
                    used_poly_types.insert(PolyType::MultiMillerLoopF);
                    used_poly_types.insert(PolyType::MultiMillerLoopFNext);
                    used_poly_types.insert(PolyType::MultiMillerLoopQuotient);
                    used_poly_types.insert(PolyType::MultiMillerLoopTXC0);
                    used_poly_types.insert(PolyType::MultiMillerLoopTXC1);
                    used_poly_types.insert(PolyType::MultiMillerLoopTYC0);
                    used_poly_types.insert(PolyType::MultiMillerLoopTYC1);
                    used_poly_types.insert(PolyType::MultiMillerLoopTXC0Next);
                    used_poly_types.insert(PolyType::MultiMillerLoopTXC1Next);
                    used_poly_types.insert(PolyType::MultiMillerLoopTYC0Next);
                    used_poly_types.insert(PolyType::MultiMillerLoopTYC1Next);
                    used_poly_types.insert(PolyType::MultiMillerLoopLambdaC0);
                    used_poly_types.insert(PolyType::MultiMillerLoopLambdaC1);
                    used_poly_types.insert(PolyType::MultiMillerLoopInvDeltaXC0);
                    used_poly_types.insert(PolyType::MultiMillerLoopInvDeltaXC1);
                    used_poly_types.insert(PolyType::MultiMillerLoopLC0C0);
                    used_poly_types.insert(PolyType::MultiMillerLoopLC0C1);
                    used_poly_types.insert(PolyType::MultiMillerLoopLC1C0);
                    used_poly_types.insert(PolyType::MultiMillerLoopLC1C1);
                    used_poly_types.insert(PolyType::MultiMillerLoopLC2C0);
                    used_poly_types.insert(PolyType::MultiMillerLoopLC2C1);
                    used_poly_types.insert(PolyType::MultiMillerLoopXP);
                    used_poly_types.insert(PolyType::MultiMillerLoopYP);
                    used_poly_types.insert(PolyType::MultiMillerLoopXQC0);
                    used_poly_types.insert(PolyType::MultiMillerLoopXQC1);
                    used_poly_types.insert(PolyType::MultiMillerLoopYQC0);
                    used_poly_types.insert(PolyType::MultiMillerLoopYQC1);
                    used_poly_types.insert(PolyType::MultiMillerLoopIsDouble);
                    used_poly_types.insert(PolyType::MultiMillerLoopIsAdd);
                    used_poly_types.insert(PolyType::MultiMillerLoopLVal);
                    used_poly_types.insert(PolyType::MultiMillerLoopG);
                    used_poly_types.insert(PolyType::MultiMillerLoopSelector0);
                    used_poly_types.insert(PolyType::MultiMillerLoopSelector1);
                    used_poly_types.insert(PolyType::MultiMillerLoopSelector2);
                    used_poly_types.insert(PolyType::MultiMillerLoopSelector3);
                    used_poly_types.insert(PolyType::MultiMillerLoopSelector4);
                    used_poly_types.insert(PolyType::MultiMillerLoopSelector5);
                }
            }
        }

        // Iterate through polynomial types in order (matching matrix layout)
        for &poly_type in PolyType::all() {
            if !used_poly_types.contains(&poly_type) {
                continue;
            }

            // For each constraint, check if it uses this polynomial type
            for (idx, constraint) in constraints.iter().enumerate() {
                let num_vars = match &constraint.constraint_type {
                    ConstraintType::GtExp => {
                        // Packed GT exp uses RhoPrev and Quotient (all 11-var)
                        // Base, Bit, and RhoNext are not committed polynomials
                        match poly_type {
                            PolyType::RhoPrev | PolyType::Quotient => Some(11),
                            _ => None,
                        }
                    }
                    ConstraintType::GtMul => {
                        // GT mul uses MulLhs, MulRhs, MulResult, MulQuotient (4-var padded to 11)
                        match poly_type {
                            PolyType::MulLhs
                            | PolyType::MulRhs
                            | PolyType::MulResult
                            | PolyType::MulQuotient => Some(4),
                            _ => None,
                        }
                    }
                    ConstraintType::G1ScalarMul { .. } => {
                        // G1 scalar mul uses all G1ScalarMul* types (8-var padded to 11)
                        match poly_type {
                            PolyType::G1ScalarMulXA
                            | PolyType::G1ScalarMulYA
                            | PolyType::G1ScalarMulXT
                            | PolyType::G1ScalarMulYT
                            | PolyType::G1ScalarMulXANext
                            | PolyType::G1ScalarMulYANext
                            | PolyType::G1ScalarMulTIndicator
                            | PolyType::G1ScalarMulAIndicator => Some(8),
                            _ => None,
                        }
                    }
                    ConstraintType::G2ScalarMul { .. } => {
                        // G2 scalar mul uses all G2ScalarMul* types (8-var padded to 12)
                        match poly_type {
                            PolyType::G2ScalarMulXAC0
                            | PolyType::G2ScalarMulXAC1
                            | PolyType::G2ScalarMulYAC0
                            | PolyType::G2ScalarMulYAC1
                            | PolyType::G2ScalarMulXTC0
                            | PolyType::G2ScalarMulXTC1
                            | PolyType::G2ScalarMulYTC0
                            | PolyType::G2ScalarMulYTC1
                            | PolyType::G2ScalarMulXANextC0
                            | PolyType::G2ScalarMulXANextC1
                            | PolyType::G2ScalarMulYANextC0
                            | PolyType::G2ScalarMulYANextC1
                            | PolyType::G2ScalarMulTIndicator
                            | PolyType::G2ScalarMulAIndicator => Some(8),
                            _ => None,
                        }
                    }
                    ConstraintType::G1Add => match poly_type {
                        PolyType::G1AddXP
                        | PolyType::G1AddYP
                        | PolyType::G1AddPIndicator
                        | PolyType::G1AddXQ
                        | PolyType::G1AddYQ
                        | PolyType::G1AddQIndicator
                        | PolyType::G1AddXR
                        | PolyType::G1AddYR
                        | PolyType::G1AddRIndicator
                        | PolyType::G1AddLambda
                        | PolyType::G1AddInvDeltaX
                        | PolyType::G1AddIsDouble
                        | PolyType::G1AddIsInverse => {
                            // G1Add witnesses are constant over the constraint hypercube
                            // (single operation, not a trace), so these are 0-var polys.
                            Some(0)
                        }
                        _ => None,
                    },
                    ConstraintType::G2Add => match poly_type {
                        PolyType::G2AddXPC0
                        | PolyType::G2AddXPC1
                        | PolyType::G2AddYPC0
                        | PolyType::G2AddYPC1
                        | PolyType::G2AddPIndicator
                        | PolyType::G2AddXQC0
                        | PolyType::G2AddXQC1
                        | PolyType::G2AddYQC0
                        | PolyType::G2AddYQC1
                        | PolyType::G2AddQIndicator
                        | PolyType::G2AddXRC0
                        | PolyType::G2AddXRC1
                        | PolyType::G2AddYRC0
                        | PolyType::G2AddYRC1
                        | PolyType::G2AddRIndicator
                        | PolyType::G2AddLambdaC0
                        | PolyType::G2AddLambdaC1
                        | PolyType::G2AddInvDeltaXC0
                        | PolyType::G2AddInvDeltaXC1
                        | PolyType::G2AddIsDouble
                        | PolyType::G2AddIsInverse => {
                            // G2Add witnesses are constant over the constraint hypercube
                            // (single operation, not a trace), so these are 0-var polys.
                            Some(0)
                        }
                        _ => None,
                    },
                    #[cfg(feature = "experimental-pairing-recursion")]
                    ConstraintType::MultiMillerLoop => match poly_type {
                        PolyType::MultiMillerLoopF
                        | PolyType::MultiMillerLoopFNext
                        | PolyType::MultiMillerLoopQuotient
                        | PolyType::MultiMillerLoopTXC0
                        | PolyType::MultiMillerLoopTXC1
                        | PolyType::MultiMillerLoopTYC0
                        | PolyType::MultiMillerLoopTYC1
                        | PolyType::MultiMillerLoopTXC0Next
                        | PolyType::MultiMillerLoopTXC1Next
                        | PolyType::MultiMillerLoopTYC0Next
                        | PolyType::MultiMillerLoopTYC1Next
                        | PolyType::MultiMillerLoopLambdaC0
                        | PolyType::MultiMillerLoopLambdaC1
                        | PolyType::MultiMillerLoopInvDeltaXC0
                        | PolyType::MultiMillerLoopInvDeltaXC1
                        | PolyType::MultiMillerLoopLC0C0
                        | PolyType::MultiMillerLoopLC0C1
                        | PolyType::MultiMillerLoopLC1C0
                        | PolyType::MultiMillerLoopLC1C1
                        | PolyType::MultiMillerLoopLC2C0
                        | PolyType::MultiMillerLoopLC2C1
                        | PolyType::MultiMillerLoopXP
                        | PolyType::MultiMillerLoopYP
                        | PolyType::MultiMillerLoopXQC0
                        | PolyType::MultiMillerLoopXQC1
                        | PolyType::MultiMillerLoopYQC0
                        | PolyType::MultiMillerLoopYQC1
                        | PolyType::MultiMillerLoopIsDouble
                        | PolyType::MultiMillerLoopIsAdd
                        | PolyType::MultiMillerLoopLVal
                        | PolyType::MultiMillerLoopG
                        | PolyType::MultiMillerLoopSelector0
                        | PolyType::MultiMillerLoopSelector1
                        | PolyType::MultiMillerLoopSelector2
                        | PolyType::MultiMillerLoopSelector3
                        | PolyType::MultiMillerLoopSelector4
                        | PolyType::MultiMillerLoopSelector5 => Some(11),
                        _ => None,
                    },
                };

                if let Some(num_vars) = num_vars {
                    polynomials.push((idx, poly_type, num_vars));
                }
            }
        }

        Self { polynomials }
    }

    /// Build the bijection and constraint mapping
    pub fn build(self) -> (VarCountJaggedBijection, ConstraintMapping) {
        let jagged_polys: Vec<JaggedPolynomial> = self
            .polynomials
            .iter()
            .map(|(_, _, num_vars)| JaggedPolynomial::new(*num_vars))
            .collect();

        let bijection = VarCountJaggedBijection::new(jagged_polys);
        let mapping = ConstraintMapping::from_tuples(&self.polynomials);

        (bijection, mapping)
    }
}

/// Extension methods for ConstraintSystem to build dense polynomial
impl ConstraintSystem {
    /// Build dense polynomial using generic jagged transform
    /// Returns the dense polynomial, the bijection, and the mapping used to create it
    pub fn build_dense_polynomial(
        &self,
    ) -> (
        crate::poly::dense_mlpoly::DensePolynomial<Fq>,
        VarCountJaggedBijection,
        ConstraintMapping,
    ) {
        // Env-gated logging for precise virtual-poly size tracking.
        //
        // - JOLT_RECURSION_LOG_POLY_SIZES=0/false (default): off
        // - JOLT_RECURSION_LOG_POLY_SIZES=1/true: summary only
        // - JOLT_RECURSION_LOG_POLY_SIZES=2/full: full per-virtual-poly table
        let poly_size_log_level: u8 = std::env::var("JOLT_RECURSION_LOG_POLY_SIZES")
            .ok()
            .and_then(|v| {
                let v = v.trim().to_lowercase();
                match v.as_str() {
                    "" | "0" | "false" | "off" => Some(0),
                    "1" | "true" | "on" => Some(1),
                    "2" | "full" => Some(2),
                    _ => v.parse::<u8>().ok(),
                }
            })
            .unwrap_or(0);

        let builder = ConstraintSystemJaggedBuilder::from_constraints(&self.constraints);
        let (bijection, mapping) = builder.build();

        // Pre-allocate for exact dense size
        let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&bijection);
        let mut dense_evals = Vec::with_capacity(dense_size);

        if poly_size_log_level >= 1 {
            let num_polynomials = bijection.num_polynomials();
            let padded_size = dense_size.next_power_of_two();
            let row_size = 1usize << self.matrix.num_constraint_vars;

            // Summaries over the jagged packing.
            let mut count_by_num_vars: std::collections::BTreeMap<usize, usize> =
                std::collections::BTreeMap::new();
            // Use string keys so we don't require `PolyType: Ord`.
            let mut count_by_poly_type: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            let mut count_by_constraint_type: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();

            for poly_idx in 0..num_polynomials {
                let (constraint_idx, poly_type) = mapping.decode(poly_idx);
                let num_vars = <VarCountJaggedBijection as JaggedTransform<Fq>>::poly_num_vars(
                    &bijection, poly_idx,
                );

                *count_by_num_vars.entry(num_vars).or_insert(0) += 1;
                *count_by_poly_type
                    .entry(format!("{poly_type:?}"))
                    .or_insert(0) += 1;

                let cty = self
                    .constraints
                    .get(constraint_idx)
                    .map(|c| format!("{:?}", c.constraint_type))
                    .unwrap_or_else(|| "<unknown>".to_string());
                *count_by_constraint_type.entry(cty).or_insert(0) += 1;
            }

            tracing::info!(
                "[recursion::jagged] virtual-poly packing: num_constraints={}, num_polynomials={}, dense_native_size={}, dense_padded_size={}, dense_padding={}, row_size={}",
                self.constraints.len(),
                num_polynomials,
                dense_size,
                padded_size,
                padded_size - dense_size,
                row_size,
            );
            tracing::info!(
                "[recursion::jagged] virtual-poly packing: counts_by_num_vars={:?}",
                count_by_num_vars
            );
            tracing::info!(
                "[recursion::jagged] virtual-poly packing: counts_by_poly_type={:?}",
                count_by_poly_type
            );
            tracing::info!(
                "[recursion::jagged] virtual-poly packing: counts_by_constraint_type={:?}",
                count_by_constraint_type
            );

            if poly_size_log_level >= 2 {
                tracing::info!(
                    "[recursion::jagged] virtual-poly table: poly_idx, constraint_idx, constraint_type, poly_type, num_vars, native_size, dense_start, dense_end, matrix_row, matrix_row_offset"
                );

                for poly_idx in 0..num_polynomials {
                    let (constraint_idx, poly_type) = mapping.decode(poly_idx);
                    let num_vars = <VarCountJaggedBijection as JaggedTransform<Fq>>::poly_num_vars(
                        &bijection, poly_idx,
                    );

                    let dense_start = bijection.cumulative_size_before(poly_idx);
                    let dense_end = bijection.cumulative_size(poly_idx);
                    let native_size = dense_end - dense_start;

                    let matrix_row = self.matrix.row_index(poly_type, constraint_idx);
                    let matrix_row_offset = self.matrix.storage_offset(matrix_row);

                    let constraint_type = self
                        .constraints
                        .get(constraint_idx)
                        .map(|c| format!("{:?}", c.constraint_type))
                        .unwrap_or_else(|| "<unknown>".to_string());

                    tracing::info!(
                        "[recursion::jagged] poly_idx={} constraint_idx={} constraint_type={} poly_type={:?} num_vars={} native_size={} dense_range=[{}, {}) matrix_row={} matrix_row_offset={}",
                        poly_idx,
                        constraint_idx,
                        constraint_type,
                        poly_type,
                        num_vars,
                        native_size,
                        dense_start,
                        dense_end,
                        matrix_row,
                        matrix_row_offset,
                    );
                }
            }
        }

        // Extract only native (non-padded) evaluations
        for dense_idx in 0..dense_size {
            let poly_idx =
                <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&bijection, dense_idx);
            let eval_idx =
                <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&bijection, dense_idx);

            let (constraint_idx, poly_type) = mapping.decode(poly_idx);

            // Get the number of variables for this polynomial
            let _num_vars = <VarCountJaggedBijection as JaggedTransform<Fq>>::poly_num_vars(
                &bijection, poly_idx,
            );

            // Get the row in the matrix
            let matrix_row = self.matrix.row_index(poly_type, constraint_idx);
            let offset = self.matrix.storage_offset(matrix_row);

            // For 4-var polynomials with zero padding, values are stored directly
            // at the beginning of the padded array (no repetition).
            let sparse_idx = eval_idx; // Direct indexing for both 4-var and 8-var

            dense_evals.push(self.matrix.evaluations[offset + sparse_idx]);
        }

        // Pad to power of 2 for multilinear polynomial
        use ark_ff::Zero;
        let padded_size = dense_evals.len().next_power_of_two();
        dense_evals.resize(padded_size, Fq::zero());

        (
            crate::poly::dense_mlpoly::DensePolynomial::new(dense_evals),
            bijection,
            mapping,
        )
    }
}
