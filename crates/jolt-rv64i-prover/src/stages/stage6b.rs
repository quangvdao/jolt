//! Local stage6b prover driver and reference-kernel registry.
use crate::plane::Rv64iPlane;
use crate::reference::bits_reduction::BitsReductionPrepare;
use crate::reference::bytecode::BytecodeReadCyclePrepare;
use crate::reference::ra_product::RamRaProductPrepare;
use jolt_field::JoltField;
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::stages::stage6b::{
    BitsReduction, BytecodeReadCycle, RamRaProduct, Stage6bChallenges, Stage6bInputClaims,
    Stage6bInputPoints, Stage6bOutputClaims, Stage6bOutputPoints,
    Stage6bSumchecks as VerifierStage6bSumchecks,
};
use std::ops::Deref;

pub struct Stage6bSumchecks<F: JoltField>(pub VerifierStage6bSumchecks<F>);
impl<F: JoltField> Deref for Stage6bSumchecks<F> {
    type Target = VerifierStage6bSumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[derive(KernelSlots)]
pub struct Stage6bKernels<F: JoltField> {
    pub bytecode_read_cycle: Box<dyn PrepareKernel<F, BytecodeReadCycle<F>, Rv64iPlane>>,
    pub ram_ra_product: Box<dyn PrepareKernel<F, RamRaProduct<F>, Rv64iPlane>>,
    pub bits_reduction: Box<dyn PrepareKernel<F, BitsReduction<F>, Rv64iPlane>>,
}
impl<F: JoltField> Default for Stage6bKernels<F> {
    fn default() -> Self {
        Self {
            bytecode_read_cycle: Box::<BytecodeReadCyclePrepare>::default(),
            ram_ra_product: Box::<RamRaProductPrepare>::default(),
            bits_reduction: Box::<BitsReductionPrepare>::default(),
        }
    }
}
jolt_rv64i_verifier::stage6b_sumchecks_members!(impl_stage_prover plane=Rv64iPlane, curate=|_batch,claims,_points| { Ok(claims.bits_reduction.columns.clone()) },);
