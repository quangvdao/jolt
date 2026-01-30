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

use super::constraints::system::{ConstraintType, PolyType};
use super::gt::indexing::{k_exp, k_mul};
use ark_bn254::Fq;
use ark_ff::{One, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// One packed polynomial entry: identifies the source row and where it lives in the packed table.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrefixPackedEntry {
    /// Global constraint index (position in `constraint_types`)
    pub constraint_idx: usize,
    /// Which committed polynomial within that constraint (PolyType row)
    pub poly_type: PolyType,
    /// If true, this entry is a GT row (not tied to a single `constraint_idx`).
    ///
    /// GT rows are packed as blocks over a GT-local `c` domain.
    /// Such entries are keyed only by `poly_type`; `constraint_idx` is ignored.
    pub is_gt: bool,
    /// If true, this entry is a G1-scalar-mul row (not tied to a single `constraint_idx`).
    ///
    /// G1 scalar-mul rows are packed as blocks over a
    /// family-local `c` domain. Such entries are keyed only by `poly_type`; `constraint_idx` is
    /// ignored.
    pub is_g1_scalar_mul: bool,
    /// If true, this entry is a G1-add row (not tied to a single `constraint_idx`).
    ///
    /// G1 add rows are packed as blocks over a family-local
    /// `c_add` domain. Such entries are keyed only by `poly_type`; `constraint_idx` is ignored.
    pub is_g1_add: bool,
    /// If true, this entry is a G2-scalar-mul row (not tied to a single `constraint_idx`).
    ///
    /// G2 scalar-mul rows are packed as blocks over a
    /// family-local `c` domain. Such entries are keyed only by `poly_type`; `constraint_idx` is
    /// ignored.
    pub is_g2_scalar_mul: bool,
    /// If true, this entry is a G2-add row (not tied to a single `constraint_idx`).
    ///
    /// G2 add rows are packed as blocks over a family-local
    /// `c_add` domain. Such entries are keyed only by `poly_type`; `constraint_idx` is ignored.
    pub is_g2_add: bool,
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
    /// Build the packing layout from the public constraint list.
    ///
    /// # Canonical Ordering Specification
    ///
    /// The packing layout is **deterministic** and **publicly derivable** from the constraint list.
    /// Both prover and verifier compute the identical layout without any additional communication.
    ///
    /// **Sort key** (lexicographic, applied to each committed polynomial):
    /// 1. **`num_vars` descending**: Larger polynomials (more variables) come first.
    /// 2. **`PolyType` ascending**: Within same size, order by `PolyType` discriminant.
    /// 3. **Family flags ascending**: Tie-breaker to keep ordering stable if `PolyType` ever
    ///    overlaps across families (today it does not).
    ///
    /// This ordering ensures:
    /// - Power-of-two alignment is maintained (larger blocks first guarantees alignment).
    /// - The layout is stable across runs and implementations.
    /// - The ordering is independent of prover choices.
    ///
    /// **Stability Note**: This ordering is part of the proof format. Changing it would break
    /// proof compatibility. Any future modifications must be versioned.
    pub fn from_constraint_types(constraint_types: &[ConstraintType]) -> Self {
        let num_gt = constraint_types
            .iter()
            .filter(|ct| matches!(ct, ConstraintType::GtExp | ConstraintType::GtMul))
            .count();
        let num_g1_smul = constraint_types
            .iter()
            .filter(|ct| matches!(ct, ConstraintType::G1ScalarMul { .. }))
            .count();
        let num_g1_add = constraint_types
            .iter()
            .filter(|ct| matches!(ct, ConstraintType::G1Add))
            .count();
        let num_g2_smul = constraint_types
            .iter()
            .filter(|ct| matches!(ct, ConstraintType::G2ScalarMul { .. }))
            .count();
        let num_g2_add = constraint_types
            .iter()
            .filter(|ct| matches!(ct, ConstraintType::G2Add))
            .count();

        // GT rows are committed at their family-local padded sizes.
        let num_vars_gt_exp = 11usize + k_exp(constraint_types);
        let num_vars_gt_mul = 4usize + k_mul(constraint_types);

        // Scalar-mul rows: native 8-var step traces plus a family-local padded `c` suffix.
        let num_vars_g1_smul = {
            let padded = num_g1_smul.max(1).next_power_of_two();
            8usize + (padded.trailing_zeros() as usize)
        };
        let num_vars_g2_smul = {
            let padded = num_g2_smul.max(1).next_power_of_two();
            8usize + (padded.trailing_zeros() as usize)
        };

        // Add rows: family-local padded `c_add` suffix only.
        let num_vars_g1_add = {
            let padded = num_g1_add.max(1).next_power_of_two();
            padded.trailing_zeros() as usize
        };
        let num_vars_g2_add = {
            let padded = num_g2_add.max(1).next_power_of_two();
            padded.trailing_zeros() as usize
        };

        // Collect committed polynomial rows. For family-packed entries, `constraint_idx` is ignored.
        type PolySpec = (usize, PolyType, bool, bool, bool, bool, bool, usize);
        let mut polys: Vec<PolySpec> = Vec::new();

        if num_gt > 0 {
            // IMPORTANT (no-padding GTMul): GTExp rows are 11-var (s,x) plus k vars, but GTMul rows
            // are natively 4-var (x) plus the same k vars.
            for poly_type in [PolyType::RhoPrev, PolyType::Quotient] {
                polys.push((
                    0usize,
                    poly_type,
                    true,
                    false,
                    false,
                    false,
                    false,
                    num_vars_gt_exp,
                ));
            }
            for poly_type in [
                PolyType::MulLhs,
                PolyType::MulRhs,
                PolyType::MulResult,
                PolyType::MulQuotient,
            ] {
                polys.push((
                    0usize,
                    poly_type,
                    true,
                    false,
                    false,
                    false,
                    false,
                    num_vars_gt_mul,
                ));
            }
        }

        if num_g1_smul > 0 {
            for poly_type in [
                PolyType::G1ScalarMulXA,
                PolyType::G1ScalarMulYA,
                PolyType::G1ScalarMulXT,
                PolyType::G1ScalarMulYT,
                PolyType::G1ScalarMulXANext,
                PolyType::G1ScalarMulYANext,
                PolyType::G1ScalarMulTIndicator,
                PolyType::G1ScalarMulAIndicator,
            ] {
                polys.push((
                    0usize,
                    poly_type,
                    false,
                    true,
                    false,
                    false,
                    false,
                    num_vars_g1_smul,
                ));
            }
        }

        if num_g1_add > 0 {
            for poly_type in [
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
            ] {
                polys.push((
                    0usize,
                    poly_type,
                    false,
                    false,
                    true,
                    false,
                    false,
                    num_vars_g1_add,
                ));
            }
        }

        if num_g2_smul > 0 {
            for poly_type in [
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
            ] {
                polys.push((
                    0usize,
                    poly_type,
                    false,
                    false,
                    false,
                    true,
                    false,
                    num_vars_g2_smul,
                ));
            }
        }

        if num_g2_add > 0 {
            for poly_type in [
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
            ] {
                polys.push((
                    0usize,
                    poly_type,
                    false,
                    false,
                    false,
                    false,
                    true,
                    num_vars_g2_add,
                ));
            }
        }

        // Canonical ordering: decreasing size (num_vars), then PolyType-major, then family flags,
        // then constraint index.
        polys.sort_by_key(
            |(
                constraint_idx,
                poly_type,
                is_gt,
                is_g1_scalar_mul,
                is_g1_add,
                is_g2_scalar_mul,
                is_g2_add,
                num_vars,
            )| {
                (
                    std::cmp::Reverse(*num_vars),
                    *poly_type as usize,
                    *is_gt as usize,
                    *is_g1_scalar_mul as usize,
                    *is_g1_add as usize,
                    *is_g2_scalar_mul as usize,
                    *is_g2_add as usize,
                    *constraint_idx,
                )
            },
        );

        // Assign aligned offsets by cumulative sum (alignment holds for power-of-two sizes).
        let mut entries: Vec<PrefixPackedEntry> = Vec::with_capacity(polys.len());
        let mut offset: usize = 0;
        for (
            constraint_idx,
            poly_type,
            is_gt,
            is_g1_scalar_mul,
            is_g1_add,
            is_g2_scalar_mul,
            is_g2_add,
            num_vars,
        ) in polys
        {
            let native_size = 1usize << num_vars;
            debug_assert_eq!(
                offset % native_size,
                0,
                "prefix packing requires aligned offsets (offset={offset}, native_size={native_size})"
            );
            entries.push(PrefixPackedEntry {
                constraint_idx,
                poly_type,
                is_gt,
                is_g1_scalar_mul,
                is_g1_add,
                is_g2_scalar_mul,
                is_g2_add,
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

    /// Length of the packed evaluation table that is actually populated by real data
    /// (i.e. before zero padding to `packed_size()`).
    ///
    /// For this prefix packing layout, populated entries are contiguous from 0..unpadded_len(),
    /// and the suffix [unpadded_len()..packed_size()) is implicitly zero.
    #[inline]
    pub fn unpadded_len(&self) -> usize {
        self.entries
            .last()
            .map(|e| e.offset + (1usize << e.num_vars))
            .unwrap_or(0)
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
    FGet: FnMut(&PrefixPackedEntry) -> Fq,
{
    let mut acc = Fq::zero();
    for entry in &layout.entries {
        let claim = get_claim(entry);
        let w = layout.codeword_weight_lsb(entry, r_full_lsb);
        acc += claim * w;
    }
    acc
}
