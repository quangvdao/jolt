//! The local stage-5 driver wraps the verifier-owned batch and member order.

use crate::plane::Rv64iPlane;
use crate::reference::val_evaluation::{RamValEvaluationPrepare, RegistersValEvaluationPrepare};
use jolt_field::{JoltField, F128};
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::stages::stage5::val_evaluation::{
    RamValEvaluation, RegistersValEvaluation,
};
use jolt_rv64i_verifier::stages::stage5::{
    Stage5Challenges, Stage5InputClaims, Stage5InputPoints, Stage5OutputClaims, Stage5OutputPoints,
    Stage5Sumchecks as VerifierStage5Sumchecks,
};
use std::ops::Deref;

pub struct Stage5Sumchecks<F: JoltField>(pub VerifierStage5Sumchecks<F>);
impl<F: JoltField> Deref for Stage5Sumchecks<F> {
    type Target = VerifierStage5Sumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[derive(KernelSlots)]
pub struct Stage5Kernels<F: JoltField> {
    pub registers_val_evaluation: Box<dyn PrepareKernel<F, RegistersValEvaluation<F>, Rv64iPlane>>,
    pub ram_val_evaluation: Box<dyn PrepareKernel<F, RamValEvaluation<F>, Rv64iPlane>>,
}
impl Default for Stage5Kernels<F128> {
    fn default() -> Self {
        Self {
            registers_val_evaluation: Box::<RegistersValEvaluationPrepare>::default(),
            ram_val_evaluation: Box::<RamValEvaluationPrepare>::default(),
        }
    }
}
jolt_rv64i_verifier::stage5_sumchecks_members!(impl_stage_prover plane=Rv64iPlane,);
