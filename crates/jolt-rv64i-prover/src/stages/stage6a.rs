//! Local stage6a prover driver and reference-kernel registry.
use crate::plane::Rv64iPlane;
use crate::reference::bytecode::BytecodeReadAddressPrepare;
use jolt_field::JoltField;
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::stages::stage6a::{
    BytecodeReadAddress, Stage6aChallenges, Stage6aInputClaims, Stage6aInputPoints,
    Stage6aOutputClaims, Stage6aOutputPoints, Stage6aSumchecks as VerifierStage6aSumchecks,
};
use std::ops::Deref;

pub struct Stage6aSumchecks<F: JoltField>(pub VerifierStage6aSumchecks<F>);
impl<F: JoltField> Deref for Stage6aSumchecks<F> {
    type Target = VerifierStage6aSumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[derive(KernelSlots)]
pub struct Stage6aKernels<F: JoltField> {
    pub bytecode_read_address: Box<dyn PrepareKernel<F, BytecodeReadAddress<F>, Rv64iPlane>>,
}
impl<F: JoltField> Default for Stage6aKernels<F> {
    fn default() -> Self {
        Self {
            bytecode_read_address: Box::<BytecodeReadAddressPrepare>::default(),
        }
    }
}
jolt_rv64i_verifier::stage6a_sumchecks_members!(impl_stage_prover plane=Rv64iPlane,);
