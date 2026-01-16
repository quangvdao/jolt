//! Host-side implementation and registration (skeleton).

use crate::{
    BN254_GT_EXP_FUNCT3, BN254_GT_EXP_FUNCT7, BN254_GT_EXP_NAME, BN254_GT_INV_FUNCT3,
    BN254_GT_INV_FUNCT7, BN254_GT_INV_NAME, BN254_GT_MUL_FUNCT3, BN254_GT_MUL_FUNCT7,
    BN254_GT_MUL_NAME, BN254_GT_SQR_FUNCT3, BN254_GT_SQR_FUNCT7, BN254_GT_SQR_NAME, INLINE_OPCODE,
};
use tracer::register_inline;
use tracer::utils::inline_sequence_writer::{
    write_inline_trace, AppendMode, InlineDescriptor, SequenceInputs,
};

pub fn init_inlines() -> Result<(), String> {
    register_inline(
        INLINE_OPCODE,
        BN254_GT_EXP_FUNCT3,
        BN254_GT_EXP_FUNCT7,
        BN254_GT_EXP_NAME,
        std::boxed::Box::new(crate::sequence_builder::bn254_gt_exp_sequence_builder),
        None,
    )?;
    register_inline(
        INLINE_OPCODE,
        BN254_GT_MUL_FUNCT3,
        BN254_GT_MUL_FUNCT7,
        BN254_GT_MUL_NAME,
        std::boxed::Box::new(crate::sequence_builder::bn254_gt_mul_sequence_builder),
        None,
    )?;
    register_inline(
        INLINE_OPCODE,
        BN254_GT_SQR_FUNCT3,
        BN254_GT_SQR_FUNCT7,
        BN254_GT_SQR_NAME,
        std::boxed::Box::new(crate::sequence_builder::bn254_gt_sqr_sequence_builder),
        None,
    )?;
    register_inline(
        INLINE_OPCODE,
        BN254_GT_INV_FUNCT3,
        BN254_GT_INV_FUNCT7,
        BN254_GT_INV_NAME,
        std::boxed::Box::new(crate::sequence_builder::bn254_gt_inv_sequence_builder),
        None,
    )?;
    Ok(())
}

pub fn store_inlines() -> Result<(), String> {
    let inputs = SequenceInputs::default();

    // BN254_GT_EXP
    {
        let inline_info = InlineDescriptor::new(
            BN254_GT_EXP_NAME.to_string(),
            INLINE_OPCODE,
            BN254_GT_EXP_FUNCT3,
            BN254_GT_EXP_FUNCT7,
        );
        let instructions = crate::sequence_builder::bn254_gt_exp_sequence_builder(
            (&inputs).into(),
            (&inputs).into(),
        );
        write_inline_trace(
            "bn254_gt_exp_trace.joltinline",
            &inline_info,
            &inputs,
            &instructions,
            AppendMode::Overwrite,
        )
        .map_err(|e| e.to_string())?;
    }

    // BN254_GT_MUL
    {
        let inline_info = InlineDescriptor::new(
            BN254_GT_MUL_NAME.to_string(),
            INLINE_OPCODE,
            BN254_GT_MUL_FUNCT3,
            BN254_GT_MUL_FUNCT7,
        );
        let instructions = crate::sequence_builder::bn254_gt_mul_sequence_builder(
            (&inputs).into(),
            (&inputs).into(),
        );
        write_inline_trace(
            "bn254_gt_mul_trace.joltinline",
            &inline_info,
            &inputs,
            &instructions,
            AppendMode::Overwrite,
        )
        .map_err(|e| e.to_string())?;
    }

    // BN254_GT_SQR
    {
        let inline_info = InlineDescriptor::new(
            BN254_GT_SQR_NAME.to_string(),
            INLINE_OPCODE,
            BN254_GT_SQR_FUNCT3,
            BN254_GT_SQR_FUNCT7,
        );
        let instructions = crate::sequence_builder::bn254_gt_sqr_sequence_builder(
            (&inputs).into(),
            (&inputs).into(),
        );
        write_inline_trace(
            "bn254_gt_sqr_trace.joltinline",
            &inline_info,
            &inputs,
            &instructions,
            AppendMode::Overwrite,
        )
        .map_err(|e| e.to_string())?;
    }

    // BN254_GT_INV
    {
        let inline_info = InlineDescriptor::new(
            BN254_GT_INV_NAME.to_string(),
            INLINE_OPCODE,
            BN254_GT_INV_FUNCT3,
            BN254_GT_INV_FUNCT7,
        );
        let instructions = crate::sequence_builder::bn254_gt_inv_sequence_builder(
            (&inputs).into(),
            (&inputs).into(),
        );
        write_inline_trace(
            "bn254_gt_inv_trace.joltinline",
            &inline_info,
            &inputs,
            &instructions,
            AppendMode::Overwrite,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
#[ctor::ctor]
fn auto_register() {
    if let Err(e) = init_inlines() {
        tracing::error!("Failed to register BN254_GT_EXP inline: {e}");
    }

    if std::env::var("STORE_INLINE").unwrap_or_default() == "true" {
        if let Err(e) = store_inlines() {
            tracing::error!("Failed to store BN254_GT_EXP inline trace: {e}");
        }
    }
}
