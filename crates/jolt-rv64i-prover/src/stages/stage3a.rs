//! Prover driver for the shared seventeen short router rounds.

use crate::plane::Rv64iPlane;
use crate::reference::routers::RouterShortPrepare;
use jolt_field::{JoltField, F128};
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::stages::stage3a::{
    RouterShort, Stage3aChallenges, Stage3aInputClaims, Stage3aInputPoints, Stage3aOutputClaims,
    Stage3aOutputPoints, Stage3aSumchecks as VerifierStage3aSumchecks,
};
use std::ops::Deref;

pub struct Stage3aSumchecks<F: JoltField>(pub VerifierStage3aSumchecks<F>);
impl<F: JoltField> Deref for Stage3aSumchecks<F> {
    type Target = VerifierStage3aSumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(KernelSlots)]
pub struct Stage3aKernels<F: JoltField> {
    pub router_short: Box<dyn PrepareKernel<F, RouterShort<F>, Rv64iPlane>>,
}
impl Default for Stage3aKernels<F128> {
    fn default() -> Self {
        Self {
            router_short: Box::<RouterShortPrepare>::default(),
        }
    }
}

jolt_rv64i_verifier::stage3a_sumchecks_members!(impl_stage_prover plane=Rv64iPlane,);
