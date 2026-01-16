//! Host-side implementation and registration (skeleton).

use crate::{BN254_GT_EXP_FUNCT3, BN254_GT_EXP_FUNCT7, BN254_GT_EXP_NAME, INLINE_OPCODE};
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
    Ok(())
}

pub fn store_inlines() -> Result<(), String> {
    let inline_info = InlineDescriptor::new(
        BN254_GT_EXP_NAME.to_string(),
        INLINE_OPCODE,
        BN254_GT_EXP_FUNCT3,
        BN254_GT_EXP_FUNCT7,
    );
    let inputs = SequenceInputs::default();
    let instructions = crate::sequence_builder::bn254_gt_exp_sequence_builder((&inputs).into(), (&inputs).into());
    write_inline_trace(
        "bn254_gt_exp_trace.joltinline",
        &inline_info,
        &inputs,
        &instructions,
        AppendMode::Overwrite,
    )
    .map_err(|e| e.to_string())?;
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

