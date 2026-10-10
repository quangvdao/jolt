//! Kernel registries for the eight ordered reduction batches.
use crate::stages::{
    stage1::Stage1Kernels, stage2::Stage2Kernels, stage3a::Stage3aKernels, stage3b::Stage3bKernels,
    stage4::Stage4Kernels, stage5::Stage5Kernels, stage6a::Stage6aKernels, stage6b::Stage6bKernels,
};
use jolt_field::{JoltField, F128};

/// One kernel registry per batch, with geometry and relations owned by the verifier batches.
pub struct Rv64iBackend<F: JoltField = F128> {
    pub stage1: Stage1Kernels<F>,
    pub stage2: Stage2Kernels<F>,
    pub stage3a: Stage3aKernels<F>,
    pub stage3b: Stage3bKernels<F>,
    pub stage4: Stage4Kernels<F>,
    pub stage5: Stage5Kernels<F>,
    pub stage6a: Stage6aKernels<F>,
    pub stage6b: Stage6bKernels<F>,
}
impl Rv64iBackend<F128> {
    /// Installs the dense reference kernels; packed witness storage remains shared by all batches.
    pub fn reference() -> Self {
        Self {
            stage1: Default::default(),
            stage2: Default::default(),
            stage3a: Default::default(),
            stage3b: Default::default(),
            stage4: Default::default(),
            stage5: Default::default(),
            stage6a: Default::default(),
            stage6b: Default::default(),
        }
    }
}
