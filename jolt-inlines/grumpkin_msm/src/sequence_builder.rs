//! Host-side sequence builder for the Grumpkin MSM(2048) inline.
//!
//! The sequence is intentionally tiny: it loads the output words via `VirtualAdvice`
//! (patched at trace-time) and stores them to the output pointer in `rs3`.

use core::mem::size_of;

use tracer::{
    instruction::{
        format::format_inline::FormatInline, sd::SD, virtual_advice::VirtualAdvice, Instruction,
    },
    utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterGuard},
};

fn advice_store_words(
    mut asm: InstrAssembler,
    operands: FormatInline,
    out_words: usize,
) -> Vec<Instruction> {
    // Single scratch inline-virtual register to stream advice->store.
    let tmp: VirtualRegisterGuard = asm.allocator.allocate_for_inline();

    for i in 0..out_words {
        asm.emit_j::<VirtualAdvice>(*tmp, 0);
        asm.emit_s::<SD>(operands.rs3, *tmp, (i * 8) as i64);
    }

    drop(tmp);
    asm.finalize_inline()
}

/// Builds the inline sequence for MSM(2048).
///
/// Semantics (trace-only):
/// - `rs1`: bases pointer
/// - `rs2`: scalars pointer
/// - `rs3`: output pointer (written verbatim as bytes of `ark_grumpkin::Projective`)
pub fn grumpkin_msm2048_inline_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    // We store the output as raw bytes of `ark_grumpkin::Projective` (assumed stable layout).
    let out_size = size_of::<ark_grumpkin::Projective>();
    assert_eq!(
        out_size % 8,
        0,
        "Expected Projective size to be u64-aligned, got {out_size}"
    );
    let out_words = out_size / 8;

    advice_store_words(asm, operands, out_words)
}

/// Builds the inline sequence for Jacobian point doubling.
///
/// Semantics (trace-only):
/// - `rs1`: input `JacobianPoint*`
/// - `rs2`: unused
/// - `rs3`: output `JacobianPoint*`
pub fn grumpkin_double_jac_inline_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let out_size = size_of::<crate::types::JacobianPoint>();
    assert_eq!(out_size % 8, 0, "JacobianPoint size must be 8-byte aligned");
    advice_store_words(asm, operands, out_size / 8)
}

/// Builds the inline sequence for mixed addition (Jacobian + affine).
///
/// Semantics (trace-only):
/// - `rs1`: input `JacobianPoint*`
/// - `rs2`: input `AffinePoint*`
/// - `rs3`: output `JacobianPoint*`
pub fn grumpkin_add_mixed_inline_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let out_size = size_of::<crate::types::JacobianPoint>();
    assert_eq!(out_size % 8, 0, "JacobianPoint size must be 8-byte aligned");
    advice_store_words(asm, operands, out_size / 8)
}

