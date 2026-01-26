//! Prefix packing for recursion PCS commitment.
//!
//! This module replaces the jagged packing path (Stage 4/5) by deterministically packing all
//! committed witness polynomials into a single dense multilinear polynomial using a
//! **prefix packing** layout.
//!
//! Key idea:
//! - Each committed witness polynomial `f_i` has a native size `2^{m_i}` (with `m_i ∈ {0,4,8,11}` today).
//! - We sort polynomials by decreasing `m_i` (power-of-two sizes) and pack their evaluation tables
//!   into one big table `F` of size `2^n`, aligned so that each `f_i` occupies a subcube defined by
//!   fixing the **high** `n - m_i` bits.
//! - This makes the indicator weights degenerate to simple equality on prefix bits, eliminating the
//!   jagged indicator branching program and the jagged transform sumchecks.
//!
//! Both prover and verifier can derive the exact same packing layout from public data
//! (`constraint_types`), and the leftover region (if any) is implicitly zero.

use crate::poly::dense_mlpoly::DensePolynomial;
use crate::zkvm::recursion::constraints::system::{ConstraintSystem, ConstraintType, PolyType};
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// Reverse the lowest `bits` bits of `x`.
#[inline]
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut y = 0usize;
    for _ in 0..bits {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}

/// One packed polynomial entry: identifies the source row and where it lives in the packed table.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrefixPackedEntry {
    /// Global constraint index (position in `constraint_types`)
    pub constraint_idx: usize,
    /// Which committed polynomial within that constraint (PolyType row)
    pub poly_type: PolyType,
    /// Native variable count `m` (native size = 2^m)
    pub num_vars: usize,
    /// Starting offset in the packed evaluation table (must be aligned to `2^num_vars`)
    pub offset: usize,
}

/// Deterministic prefix packing layout for all committed witness polynomials.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrefixPackingLayout {
    /// Number of variables of the packed dense polynomial.
    pub num_dense_vars: usize,
    /// Packed entries, in the exact order they are laid out in the packed table.
    pub entries: Vec<PrefixPackedEntry>,
}

impl PrefixPackingLayout {
    /// Build a canonical packing layout from the public constraint list.
    ///
    /// # Canonical Ordering Specification
    ///
    /// The packing layout is **deterministic** and **publicly derivable** from the constraint list.
    /// Both prover and verifier compute the identical layout without any additional communication.
    ///
    /// **Sort key** (lexicographic, applied to each committed polynomial):
    /// 1. **`num_vars` descending**: Larger polynomials (more variables) come first.
    /// 2. **`PolyType` ascending**: Within same size, order by `PolyType` discriminant.
    /// 3. **`constraint_idx` ascending**: Within same size and type, order by constraint index.
    ///
    /// This ordering ensures:
    /// - Power-of-two alignment is maintained (larger blocks first guarantees alignment).
    /// - The layout is stable across runs and implementations.
    /// - The ordering is independent of prover choices.
    ///
    /// **Stability Note**: This ordering is part of the proof format. Changing it would break
    /// proof compatibility. Any future modifications must be versioned.
    pub fn from_constraint_types(constraint_types: &[ConstraintType]) -> Self {
        // Collect all committed polynomial "rows" with their native var counts.
        let mut polys: Vec<(usize, PolyType, usize)> = Vec::new();
        for (constraint_idx, ct) in constraint_types.iter().enumerate() {
            for &(poly_type, num_vars) in ct.committed_poly_specs() {
                polys.push((constraint_idx, poly_type, num_vars));
            }
        }

        // Canonical ordering: decreasing size (num_vars), then PolyType-major, then constraint index.
        // IMPORTANT: This ordering is part of the proof format and must remain stable.
        polys.sort_by_key(|(constraint_idx, poly_type, num_vars)| {
            (
                std::cmp::Reverse(*num_vars),
                *poly_type as usize,
                *constraint_idx,
            )
        });

        // Assign aligned offsets by cumulative sum (alignment holds for power-of-two sizes).
        let mut entries: Vec<PrefixPackedEntry> = Vec::with_capacity(polys.len());
        let mut offset: usize = 0;
        for (constraint_idx, poly_type, num_vars) in polys {
            let native_size = 1usize << num_vars;
            debug_assert_eq!(
                offset % native_size,
                0,
                "prefix packing requires aligned offsets (offset={offset}, native_size={native_size})"
            );
            entries.push(PrefixPackedEntry {
                constraint_idx,
                poly_type,
                num_vars,
                offset,
            });
            offset += native_size;
        }

        let padded_size = std::cmp::max(1usize, offset).next_power_of_two();
        let num_dense_vars = padded_size.trailing_zeros() as usize;

        Self {
            num_dense_vars,
            entries,
        }
    }

    #[inline]
    pub fn packed_size(&self) -> usize {
        1usize << self.num_dense_vars
    }

    #[inline]
    pub fn codeword_len(&self, num_vars: usize) -> usize {
        debug_assert!(self.num_dense_vars >= num_vars);
        self.num_dense_vars - num_vars
    }

    #[inline]
    pub fn codeword_int(&self, entry: &PrefixPackedEntry) -> usize {
        entry.offset >> entry.num_vars
    }

    /// Compute the prefix-code equality weight `eq(codeword, r_prefix)` for this entry.
    ///
    /// - `r_full_lsb` is the packed polynomial evaluation point in **little-endian** variable order
    ///   (low-to-high), length = `num_dense_vars`.
    /// - The entry uses `num_vars` low variables for its native domain, so the prefix bits are the
    ///   remaining `num_dense_vars - num_vars` high variables.
    pub fn codeword_weight_lsb(&self, entry: &PrefixPackedEntry, r_full_lsb: &[Fq]) -> Fq {
        debug_assert_eq!(r_full_lsb.len(), self.num_dense_vars);
        let m = entry.num_vars;
        let l = self.codeword_len(m);
        let code = self.codeword_int(entry);
        let r_prefix = &r_full_lsb[m..];
        debug_assert_eq!(r_prefix.len(), l);

        // Compute Π_j (bit_j ? r_prefix[j] : (1 - r_prefix[j])).
        let mut acc = Fq::one();
        for j in 0..l {
            let bit = (code >> j) & 1;
            acc *= if bit == 1 {
                r_prefix[j]
            } else {
                Fq::one() - r_prefix[j]
            };
        }
        acc
    }
}

impl ConstraintSystem {
    /// Build the packed dense evaluation table under the given `layout`.
    ///
    /// NOTE: This currently extracts native prefixes out of the padded sparse matrix rows.
    /// This is correct under the current row encoding (native values stored first, zeros after),
    /// and will be further optimized to avoid materializing padded rows entirely.
    pub fn build_prefix_packed_evals(&self, layout: &PrefixPackingLayout) -> Vec<Fq> {
        let mut packed = vec![Fq::zero(); layout.packed_size()];
        if layout.entries.is_empty() {
            return packed;
        }

        // The underlying sparse matrix rows are stored with `matrix.num_constraint_vars` variables
        // (currently 11). We take only the first `2^num_vars` entries as the native table.
        for entry in &layout.entries {
            let native_size = 1usize << entry.num_vars;
            let dst = &mut packed[entry.offset..entry.offset + native_size];

            let row = self.matrix.row_index(entry.poly_type, entry.constraint_idx);
            let row_off = self.matrix.storage_offset(row);
            let src = &self.matrix.evaluations[row_off..row_off + native_size];

            // IMPORTANT: Sumcheck binds variables in `BindingOrder::LowToHigh` (LSB-first), and
            // recursion Stage 2 is **suffix-aligned** in the batched sumcheck. As a result, the
            // Stage-2 opening points for an m-var polynomial correspond to the *suffix* of the
            // common 11-var challenge vector.
            //
            // For the prefix-packing Stage 3 reduction, we reverse the Stage-2 `r_x` vector before
            // embedding it as the low bits of the packed opening point. To keep claim semantics
            // unchanged, we also reverse the variable order of every packed block by bit-reversing
            // its evaluation table.
            for t in 0..native_size {
                dst[t] = src[bit_reverse(t, entry.num_vars)];
            }
        }

        packed
    }

    /// Convenience wrapper: build layout + packed dense polynomial evals.
    pub fn build_prefix_packed_polynomial(&self) -> (DensePolynomial<Fq>, PrefixPackingLayout) {
        let constraint_types: Vec<ConstraintType> = self
            .constraints
            .iter()
            .map(|c| c.constraint_type.clone())
            .collect();
        let layout = PrefixPackingLayout::from_constraint_types(&constraint_types);
        let evals = self.build_prefix_packed_evals(&layout);
        (DensePolynomial::new(evals), layout)
    }
}

/// Compute the packed polynomial evaluation `F(r)` from a set of per-row (virtual) claims.
///
/// - `layout` determines the codeword for each polynomial.
/// - `r_full_lsb` is the packed opening point in little-endian order (low-to-high), length `n`.
/// - `get_claim(constraint_idx, poly_type)` must return the claimed evaluation of that
///   native polynomial at the corresponding low-bit prefix of `r_full_lsb`.
pub fn packed_eval_from_claims<FGet>(
    layout: &PrefixPackingLayout,
    r_full_lsb: &[Fq],
    mut get_claim: FGet,
) -> Fq
where
    FGet: FnMut(usize, PolyType) -> Fq,
{
    let mut acc = Fq::zero();
    for entry in &layout.entries {
        let claim = get_claim(entry.constraint_idx, entry.poly_type);
        let w = layout.codeword_weight_lsb(entry, r_full_lsb);
        acc += claim * w;
    }
    acc
}
