//! Local prover driver and reference registry for batch 1.
use crate::plane::Rv64iPlane;
use crate::reference::spartan::{SpartanOuterF128Prepare, SpartanOuterF2Prepare};
use jolt_field::{JoltField, F128};
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::stages::stage1::{
    SpartanOuterF128, SpartanOuterF2, Stage1Challenges, Stage1InputClaims, Stage1InputPoints,
    Stage1OutputClaims, Stage1OutputPoints, Stage1Sumchecks as VerifierStage1Sumchecks,
};
use std::ops::Deref;

pub struct Stage1Sumchecks<F: JoltField>(pub VerifierStage1Sumchecks<F>);
impl<F: JoltField> Deref for Stage1Sumchecks<F> {
    type Target = VerifierStage1Sumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[derive(KernelSlots)]
pub struct Stage1Kernels<F: JoltField> {
    pub spartan_outer_f2: Box<dyn PrepareKernel<F, SpartanOuterF2<F>, Rv64iPlane>>,
    pub spartan_outer_f128: Box<dyn PrepareKernel<F, SpartanOuterF128<F>, Rv64iPlane>>,
}
impl Default for Stage1Kernels<F128> {
    fn default() -> Self {
        Self {
            spartan_outer_f2: Box::<SpartanOuterF2Prepare>::default(),
            spartan_outer_f128: Box::<SpartanOuterF128Prepare>::default(),
        }
    }
}
jolt_rv64i_verifier::stage1_sumchecks_members!(impl_stage_prover plane=Rv64iPlane,);
