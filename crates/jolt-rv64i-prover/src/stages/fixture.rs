//! Local driver for the one-member reduction fixture. This is not a protocol
//! batch, and its transcript is not the wire of batch 6b.
use crate::plane::Rv64iPlane;
use crate::reference::bits_reduction::BitsReductionPrepare;
use jolt_field::JoltField;
use jolt_kernels::{KernelSlots, PrepareKernel};
use jolt_prover::impl_stage_prover;
use jolt_rv64i_verifier::stages::fixture::{
    BitsReduction, ReductionOnlyChallenges, ReductionOnlyInputClaims, ReductionOnlyInputPoints,
    ReductionOnlyOutputClaims, ReductionOnlyOutputPoints,
    ReductionOnlySumchecks as VerifierReductionOnlySumchecks,
};
use std::ops::Deref;

pub struct ReductionOnlySumchecks<F: JoltField>(pub VerifierReductionOnlySumchecks<F>);
impl<F: JoltField> Deref for ReductionOnlySumchecks<F> {
    type Target = VerifierReductionOnlySumchecks<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[derive(KernelSlots)]
pub struct ReductionOnlyKernels<F: JoltField> {
    pub bits_reduction: Box<dyn PrepareKernel<F, BitsReduction<F>, Rv64iPlane>>,
}
impl<F: JoltField> Default for ReductionOnlyKernels<F> {
    fn default() -> Self {
        Self {
            bits_reduction: Box::<BitsReductionPrepare>::default(),
        }
    }
}
jolt_rv64i_verifier::reduction_only_sumchecks_members!(impl_stage_prover plane=Rv64iPlane,);
