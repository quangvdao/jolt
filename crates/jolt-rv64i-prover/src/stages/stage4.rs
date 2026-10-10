//! Local prover driver and reference registry for batch 4.

use crate::plane::Rv64iPlane;
use crate::reference::read_checking::{
    RamOutputCheckPrepare, RamReadCheckingPrepare, RegistersReadCheckingPrepare,
};
use jolt_field::{JoltField, F128};
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::stages::stage4::{
    ram_output_check::RamOutputCheck,
    ram_read_checking::RamReadChecking,
    registers_read_checking::RegistersReadChecking,
    verify::{
        Stage4Challenges, Stage4InputClaims, Stage4InputPoints, Stage4OutputClaims,
        Stage4OutputPoints, Stage4Sumchecks as VerifierStage4Sumchecks,
    },
};
use std::ops::Deref;

pub struct Stage4Sumchecks<F: JoltField>(pub VerifierStage4Sumchecks<F>);
impl<F: JoltField> Deref for Stage4Sumchecks<F> {
    type Target = VerifierStage4Sumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[derive(KernelSlots)]
pub struct Stage4Kernels<F: JoltField> {
    pub registers_read_checking: Box<dyn PrepareKernel<F, RegistersReadChecking<F>, Rv64iPlane>>,
    pub ram_read_checking: Box<dyn PrepareKernel<F, RamReadChecking<F>, Rv64iPlane>>,
    pub ram_output_check: Box<dyn PrepareKernel<F, RamOutputCheck<F>, Rv64iPlane>>,
}
impl Default for Stage4Kernels<F128> {
    fn default() -> Self {
        Self {
            registers_read_checking: Box::<RegistersReadCheckingPrepare>::default(),
            ram_read_checking: Box::<RamReadCheckingPrepare>::default(),
            ram_output_check: Box::<RamOutputCheckPrepare>::default(),
        }
    }
}
jolt_rv64i_verifier::stage4_sumchecks_members!(impl_stage_prover plane = Rv64iPlane,);
