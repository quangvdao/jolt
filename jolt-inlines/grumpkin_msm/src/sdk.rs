//! Guest-side SDK wrappers for Grumpkin MSM + curve-op inlines.

/// Fixed MSM size used by Hyrax in recursion (for the Fibonacci proof observed in profiling).
pub const MSM_N: usize = 2048;

/// Low-level interface to the Grumpkin MSM(2048) inline instruction.
///
/// # ABI
/// - `bases`: pointer to an array of `MSM_N` affine points in guest memory
/// - `scalars`: pointer to an array of `MSM_N` scalars in guest memory
/// - `out`: pointer to an output `Projective` in guest memory (bytes are written verbatim)
///
/// The exact type/layout is currently assumed to match `ark_grumpkin::{Affine, Fr, Projective}`
/// for the trace-only host patcher.
///
/// # Safety
/// The pointers must be valid and properly aligned for 8-byte loads/stores.
#[cfg(not(feature = "host"))]
pub unsafe fn grumpkin_msm2048_inline(bases: *const u8, scalars: *const u8, out: *mut u8) {
    use crate::{GRUMPKIN_FUNCT7, GRUMPKIN_MSM2048_FUNCT3, INLINE_OPCODE};
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const GRUMPKIN_MSM2048_FUNCT3,
        funct7 = const GRUMPKIN_FUNCT7,
        rd = in(reg) out,
        rs1 = in(reg) bases,
        rs2 = in(reg) scalars,
        options(nostack)
    );
}

/// Host-side fallback (used for non-RISC-V unit tests / host builds).
///
/// This does *not* participate in tracing; tracing uses the registered inline sequence.
#[cfg(feature = "host")]
pub unsafe fn grumpkin_msm2048_inline(_bases: *const u8, _scalars: *const u8, _out: *mut u8) {}

/// Low-level interface to Grumpkin Jacobian point doubling.
///
/// # ABI
/// - `in_jac`: pointer to an input `types::JacobianPoint` in guest memory
/// - `out_jac`: pointer to an output `types::JacobianPoint` in guest memory
#[cfg(not(feature = "host"))]
pub unsafe fn grumpkin_double_jac_inline(in_jac: *const u8, out_jac: *mut u8) {
    use crate::{GRUMPKIN_DOUBLE_JAC_FUNCT3, GRUMPKIN_FUNCT7, INLINE_OPCODE};
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const GRUMPKIN_DOUBLE_JAC_FUNCT3,
        funct7 = const GRUMPKIN_FUNCT7,
        rd = in(reg) out_jac,
        rs1 = in(reg) in_jac,
        // rs2 unused for this op
        rs2 = in(reg) core::ptr::null::<u8>(),
        options(nostack)
    );
}

#[cfg(feature = "host")]
pub unsafe fn grumpkin_double_jac_inline(_in_jac: *const u8, _out_jac: *mut u8) {}

/// Low-level interface to Grumpkin mixed addition (Jacobian + affine).
///
/// # ABI
/// - `jac`: pointer to an input `types::JacobianPoint` in guest memory
/// - `aff`: pointer to an input `types::AffinePoint` in guest memory
/// - `out_jac`: pointer to an output `types::JacobianPoint` in guest memory
#[cfg(not(feature = "host"))]
pub unsafe fn grumpkin_add_mixed_inline(jac: *const u8, aff: *const u8, out_jac: *mut u8) {
    use crate::{GRUMPKIN_ADD_MIXED_FUNCT3, GRUMPKIN_FUNCT7, INLINE_OPCODE};
    core::arch::asm!(
        ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
        opcode = const INLINE_OPCODE,
        funct3 = const GRUMPKIN_ADD_MIXED_FUNCT3,
        funct7 = const GRUMPKIN_FUNCT7,
        rd = in(reg) out_jac,
        rs1 = in(reg) jac,
        rs2 = in(reg) aff,
        options(nostack)
    );
}

#[cfg(feature = "host")]
pub unsafe fn grumpkin_add_mixed_inline(_jac: *const u8, _aff: *const u8, _out_jac: *mut u8) {}

