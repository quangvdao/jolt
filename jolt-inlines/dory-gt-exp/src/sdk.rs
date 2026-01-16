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

/// Host build placeholder: until we implement a full host executor, fail loudly if called.
///
/// The inline is not expected to run on host; it expands via `sequence_builder` within the tracer.
#[cfg(feature = "host")]
#[inline(always)]
pub unsafe fn bn254_gt_exp_inline(_exp: *const u64, _base: *const u64, _out: *mut u64) {
    panic!("bn254_gt_exp_inline(host) is a placeholder; use tracer execution");
}

