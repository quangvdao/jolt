//! Local prover driver and reference registry for batch 2.
use crate::plane::Rv64iPlane;
use crate::reference::spartan::SpartanInnerPrepare;
use jolt_field::{JoltField, F128};
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::stages::stage2::{
    SpartanInner, Stage2Challenges, Stage2InputClaims, Stage2InputPoints, Stage2OutputClaims,
    Stage2OutputPoints, Stage2Sumchecks as VerifierStage2Sumchecks,
};
use std::ops::Deref;

pub struct Stage2Sumchecks<F: JoltField>(pub VerifierStage2Sumchecks<F>);
impl<F: JoltField> Deref for Stage2Sumchecks<F> {
    type Target = VerifierStage2Sumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[derive(KernelSlots)]
pub struct Stage2Kernels<F: JoltField> {
    pub spartan_inner: Box<dyn PrepareKernel<F, SpartanInner<F>, Rv64iPlane>>,
}
impl Default for Stage2Kernels<F128> {
    fn default() -> Self {
        Self {
            spartan_inner: Box::<SpartanInnerPrepare>::default(),
        }
    }
}
jolt_rv64i_verifier::stage2_sumchecks_members!(impl_stage_prover plane=Rv64iPlane,);
