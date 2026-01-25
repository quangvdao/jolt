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
    ConstraintSystem, ConstraintType, MatrixConstraint, PolyType, PolyTypeSet,
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
        let constraint_types: Vec<ConstraintType> = constraints
            .iter()
            .map(|c| c.constraint_type.clone())
            .collect();
        let poly_type_set = PolyTypeSet::from_constraint_types(&constraint_types);
        Self {
            polynomials: poly_type_set.into_entries(),
        }
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
    /// Build the jagged layout (bijection + mapping) from the constraint list.
    ///
    /// This is the “layout” part of jagged compression: it does **not** read the sparse matrix.
    /// Call [`extract_dense_evals`] to actually extract dense evaluations from `self.matrix`.
    pub fn build_jagged_layout(&self) -> (VarCountJaggedBijection, ConstraintMapping) {
        let builder = ConstraintSystemJaggedBuilder::from_constraints(&self.constraints);
        builder.build()
    }

    /// Extract the dense evaluation vector from the sparse matrix using an already-built layout.
    ///
    /// Returns a vector padded to a power of two (ready for `DensePolynomial::new`).
    pub fn extract_dense_evals(
        &self,
        bijection: &VarCountJaggedBijection,
        mapping: &ConstraintMapping,
    ) -> Vec<Fq> {
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

        // Pre-allocate for exact dense size
        let dense_size = <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(bijection);
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
                    bijection, poly_idx,
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
                        bijection, poly_idx,
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
                <VarCountJaggedBijection as JaggedTransform<Fq>>::row(bijection, dense_idx);
            let eval_idx =
                <VarCountJaggedBijection as JaggedTransform<Fq>>::col(bijection, dense_idx);

            let (constraint_idx, poly_type) = mapping.decode(poly_idx);

            // Get the number of variables for this polynomial
            let _num_vars = <VarCountJaggedBijection as JaggedTransform<Fq>>::poly_num_vars(
                bijection, poly_idx,
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

        dense_evals
    }

    /// Build dense polynomial using generic jagged transform.
    ///
    /// This is a convenience wrapper that builds the layout and extracts dense evals in one call.
    /// Returns the dense polynomial, the bijection, and the mapping used to create it.
    pub fn build_dense_polynomial(
        &self,
    ) -> (
        crate::poly::dense_mlpoly::DensePolynomial<Fq>,
        VarCountJaggedBijection,
        ConstraintMapping,
    ) {
        let (bijection, mapping) = self.build_jagged_layout();
        let dense_evals = self.extract_dense_evals(&bijection, &mapping);
        (
            crate::poly::dense_mlpoly::DensePolynomial::new(dense_evals),
            bijection,
            mapping,
        )
    }
}
