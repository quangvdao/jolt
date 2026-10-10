//! Prover driver for the five router cycle reductions.

use crate::plane::Rv64iPlane;
use crate::reference::routers::{
    RouterCycleBranchPrepare, RouterCycleComparePrepare, RouterCycleMemoryPrepare,
    RouterCycleShiftPrepare, RouterCycleVariantPrepare,
};
use jolt_field::{JoltField, F128};
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::stages::stage3b::{
    RouterCycleBranch, RouterCycleCompare, RouterCycleMemory, RouterCycleShift, RouterCycleVariant,
    Stage3bChallenges, Stage3bInputClaims, Stage3bInputPoints, Stage3bOutputClaims,
    Stage3bOutputPoints, Stage3bSumchecks as VerifierStage3bSumchecks,
};
use std::ops::Deref;

pub struct Stage3bSumchecks<F: JoltField>(pub VerifierStage3bSumchecks<F>);
impl<F: JoltField> Deref for Stage3bSumchecks<F> {
    type Target = VerifierStage3bSumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(KernelSlots)]
pub struct Stage3bKernels<F: JoltField> {
    pub variant: Box<dyn PrepareKernel<F, RouterCycleVariant<F>, Rv64iPlane>>,
    pub shift: Box<dyn PrepareKernel<F, RouterCycleShift<F>, Rv64iPlane>>,
    pub memory: Box<dyn PrepareKernel<F, RouterCycleMemory<F>, Rv64iPlane>>,
    pub compare: Box<dyn PrepareKernel<F, RouterCycleCompare<F>, Rv64iPlane>>,
    pub branch: Box<dyn PrepareKernel<F, RouterCycleBranch<F>, Rv64iPlane>>,
}
impl Default for Stage3bKernels<F128> {
    fn default() -> Self {
        Self {
            variant: Box::<RouterCycleVariantPrepare>::default(),
            shift: Box::<RouterCycleShiftPrepare>::default(),
            memory: Box::<RouterCycleMemoryPrepare>::default(),
            compare: Box::<RouterCycleComparePrepare>::default(),
            branch: Box::<RouterCycleBranchPrepare>::default(),
        }
    }
}

jolt_rv64i_verifier::stage3b_sumchecks_members!(impl_stage_prover plane=Rv64iPlane,);
