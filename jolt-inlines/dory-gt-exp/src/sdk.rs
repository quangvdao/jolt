//! Guest-side SDK for BN254 GT exponentiation inline (skeleton).
//!
//! ABI (proposed, matches existing Jolt core inlines):
//! - rs1 = exp_ptr  (4 x u64 limbs, little-endian)
//! - rs2 = base_ptr (48 x u64 limbs, little-endian, Fq12 tower flattening)
//! - rd  = out_ptr  (48 x u64 limbs, little-endian)
//!
//! Note: the tracer parses the R-format `rd` field as `rs3` (FormatInline), so
//! all three values are treated as input pointers and no architectural registers
//! may be modified by the inline.

use crate::{FR_LIMBS_U64, GT_LIMBS_U64};

/// Safe wrapper: computes `out = base^exp` (placeholder until inline is implemented).
///
/// This currently just calls the inline instruction on guest builds.
#[inline(always)]
pub fn bn254_gt_exp(base: [u64; GT_LIMBS_U64], exp: [u64; FR_LIMBS_U64]) -> [u64; GT_LIMBS_U64] {
    let mut out = [0u64; GT_LIMBS_U64];
    unsafe {
        bn254_gt_exp_inline(exp.as_ptr(), base.as_ptr(), out.as_mut_ptr());
    }
    out
}

/// Safe wrapper: computes `out = lhs * rhs`.
#[inline(always)]
pub fn bn254_gt_mul(lhs: [u64; GT_LIMBS_U64], rhs: [u64; GT_LIMBS_U64]) -> [u64; GT_LIMBS_U64] {
    let mut out = [0u64; GT_LIMBS_U64];
    bn254_gt_mul_into(&mut out, &lhs, &rhs);
    out
}

/// Writes `out = lhs * rhs` without moving/copying the input arrays.
#[inline(always)]
pub fn bn254_gt_mul_into(
    out: &mut [u64; GT_LIMBS_U64],
    lhs: &[u64; GT_LIMBS_U64],
    rhs: &[u64; GT_LIMBS_U64],
) {
    unsafe {
        bn254_gt_mul_inline(lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr());
    }
}

/// Safe wrapper: computes `out = x^2`.
#[inline(always)]
pub fn bn254_gt_sqr(x: [u64; GT_LIMBS_U64]) -> [u64; GT_LIMBS_U64] {
    let mut out = [0u64; GT_LIMBS_U64];
    bn254_gt_sqr_into(&mut out, &x);
    out
}

/// Writes `out = x^2` without moving/copying the input arrays.
#[inline(always)]
pub fn bn254_gt_sqr_into(out: &mut [u64; GT_LIMBS_U64], x: &[u64; GT_LIMBS_U64]) {
    unsafe {
        bn254_gt_sqr_inline(x.as_ptr(), out.as_mut_ptr());
    }
}

/// Safe wrapper: computes `out = x^{-1}` in BN254 GT.
///
/// This is implemented as **cyclotomic inverse / conjugation** (i.e., `c1 := -c1` in Fq12),
/// and is only correct when `x` is in the cyclotomic subgroup (which includes BN254 GT).
#[inline(always)]
pub fn bn254_gt_inv(x: [u64; GT_LIMBS_U64]) -> [u64; GT_LIMBS_U64] {
    let mut out = [0u64; GT_LIMBS_U64];
    bn254_gt_inv_into(&mut out, &x);
    out
}

/// Writes `out = x^{-1}` in BN254 GT without moving/copying the input array.
///
/// See `bn254_gt_inv()` for the cyclotomic-subgroup correctness contract.
#[inline(always)]
pub fn bn254_gt_inv_into(out: &mut [u64; GT_LIMBS_U64], x: &[u64; GT_LIMBS_U64]) {
    unsafe {
        bn254_gt_inv_inline(x.as_ptr(), out.as_mut_ptr());
    }
}

/// Low-level interface to the BN254 GT exponentiation inline instruction.
///
/// # Arguments
/// - `exp`  : pointer to 4 u64 words (32 bytes), little-endian limbs
/// - `base` : pointer to 48 u64 words (384 bytes), little-endian limbs (Fq12)
/// - `out`  : pointer to 48 u64 words (384 bytes) where result will be written
///
/// # Safety
/// - All pointers must be valid and 8-byte aligned.
/// - `exp` must be readable for 32 bytes.
/// - `base` must be readable for 384 bytes.
/// - `out` must be writable for 384 bytes.
/// - `out` may alias `base` (in-place update permitted) per the ABI spec, but the current
///   host-side inline expansion (sequence builder) **rejects** `out == base` because it uses
///   `out[..384]` as scratch. The safe wrapper `bn254_gt_exp()` always uses distinct buffers.
#[cfg(not(feature = "host"))]
#[inline(always)]
pub unsafe fn bn254_gt_exp_inline(exp: *const u64, base: *const u64, out: *mut u64) {
    use crate::{BN254_GT_EXP_FUNCT3, BN254_GT_EXP_FUNCT7, INLINE_OPCODE};
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const BN254_GT_EXP_FUNCT3,
        funct7 = const BN254_GT_EXP_FUNCT7,
        rd = in(reg) out,  // rd/rs3 - output address
        rs1 = in(reg) exp, // rs1 - exponent address
        rs2 = in(reg) base, // rs2 - base address
        options(nostack)
    );
}

/// Low-level interface to the BN254 GT multiplication inline instruction.
///
/// ABI:
/// - rs1 = lhs_ptr (48 x u64 limbs)
/// - rs2 = rhs_ptr (48 x u64 limbs)
/// - rd  = out_ptr (48 x u64 limbs)
///
/// # Safety
/// - All pointers must be valid and 8-byte aligned.
/// - `lhs` must be readable for 384 bytes.
/// - `rhs` must be readable for 384 bytes.
/// - `out` must be writable for 384 bytes.
/// - The current host-side inline expansion (sequence builder) **rejects** `out == rhs` because it
///   uses the `out[..384]` region as scratch while reading `rhs`.
#[cfg(not(feature = "host"))]
#[inline(always)]
pub unsafe fn bn254_gt_mul_inline(lhs: *const u64, rhs: *const u64, out: *mut u64) {
    use crate::{BN254_GT_MUL_FUNCT3, BN254_GT_MUL_FUNCT7, INLINE_OPCODE};
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const BN254_GT_MUL_FUNCT3,
        funct7 = const BN254_GT_MUL_FUNCT7,
        rd = in(reg) out, // rd/rs3 - output address
        rs1 = in(reg) lhs, // rs1 - lhs address
        rs2 = in(reg) rhs, // rs2 - rhs address
        options(nostack)
    );
}

/// Low-level interface to the BN254 GT squaring inline instruction.
///
/// ABI:
/// - rs1 = in_ptr  (48 x u64 limbs)
/// - rs2 = in_ptr  (duplicated; unused by the current builder)
/// - rd  = out_ptr (48 x u64 limbs)
///
/// # Safety
/// - All pointers must be valid and 8-byte aligned.
/// - `input` must be readable for 384 bytes.
/// - `out` must be writable for 384 bytes.
#[cfg(not(feature = "host"))]
#[inline(always)]
pub unsafe fn bn254_gt_sqr_inline(input: *const u64, out: *mut u64) {
    use crate::{BN254_GT_SQR_FUNCT3, BN254_GT_SQR_FUNCT7, INLINE_OPCODE};
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const BN254_GT_SQR_FUNCT3,
        funct7 = const BN254_GT_SQR_FUNCT7,
        rd = in(reg) out, // rd/rs3 - output address
        rs1 = in(reg) input, // rs1 - input address
        rs2 = in(reg) input, // rs2 - input address (duplicated)
        options(nostack)
    );
}

/// Low-level interface to the BN254 GT inversion inline instruction.
///
/// ABI:
/// - rs1 = in_ptr  (48 x u64 limbs)
/// - rs2 = in_ptr  (duplicated; unused by the current builder)
/// - rd  = out_ptr (48 x u64 limbs)
///
/// # Safety
/// - All pointers must be valid and 8-byte aligned.
/// - `input` must be readable for 384 bytes.
/// - `out` must be writable for 384 bytes.
///
/// # Correctness contract
/// This inline implements **cyclotomic inverse / conjugation** and is only correct for BN254 GT elements.
#[cfg(not(feature = "host"))]
#[inline(always)]
pub unsafe fn bn254_gt_inv_inline(input: *const u64, out: *mut u64) {
    use crate::{BN254_GT_INV_FUNCT3, BN254_GT_INV_FUNCT7, INLINE_OPCODE};
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const BN254_GT_INV_FUNCT3,
        funct7 = const BN254_GT_INV_FUNCT7,
        rd = in(reg) out, // rd/rs3 - output address
        rs1 = in(reg) input, // rs1 - input address
        rs2 = in(reg) input, // rs2 - input address (duplicated)
        options(nostack)
    );
}

/// Host build placeholder: until we implement a full host executor, fail loudly if called.
///
/// The inline is not expected to run on host; it expands via `sequence_builder` within the tracer.
///
/// # Safety
/// This function is `unsafe` to match the guest ABI, but on host builds it always panics and
/// performs no memory accesses.
#[cfg(feature = "host")]
#[inline(always)]
pub unsafe fn bn254_gt_exp_inline(_exp: *const u64, _base: *const u64, _out: *mut u64) {
    panic!("bn254_gt_exp_inline(host) is a placeholder; use tracer execution");
}

/// Host build placeholder (see `bn254_gt_exp_inline`).
///
/// # Safety
/// This function is `unsafe` to match the guest ABI, but on host builds it always panics and
/// performs no memory accesses.
#[cfg(feature = "host")]
#[inline(always)]
pub unsafe fn bn254_gt_mul_inline(_lhs: *const u64, _rhs: *const u64, _out: *mut u64) {
    panic!("bn254_gt_mul_inline(host) is a placeholder; use tracer execution");
}

/// Host build placeholder (see `bn254_gt_exp_inline`).
///
/// # Safety
/// This function is `unsafe` to match the guest ABI, but on host builds it always panics and
/// performs no memory accesses.
#[cfg(feature = "host")]
#[inline(always)]
pub unsafe fn bn254_gt_sqr_inline(_input: *const u64, _out: *mut u64) {
    panic!("bn254_gt_sqr_inline(host) is a placeholder; use tracer execution");
}

/// Host build placeholder (see `bn254_gt_exp_inline`).
///
/// # Safety
/// This function is `unsafe` to match the guest ABI, but on host builds it always panics and
/// performs no memory accesses.
#[cfg(feature = "host")]
#[inline(always)]
pub unsafe fn bn254_gt_inv_inline(_input: *const u64, _out: *mut u64) {
    panic!("bn254_gt_inv_inline(host) is a placeholder; use tracer execution");
}
