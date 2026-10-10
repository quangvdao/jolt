//! Kernel registries for the eight ordered reduction batches.
use crate::stages::{
    stage1::Stage1Kernels, stage2::Stage2Kernels, stage3a::Stage3aKernels, stage3b::Stage3bKernels,
    stage4::Stage4Kernels, stage5::Stage5Kernels, stage6a::Stage6aKernels, stage6b::Stage6bKernels,
};
use jolt_field::F128;

/// One kernel registry per batch, with geometry and relations owned by the verifier batches.
pub struct Rv64iBackend {
    pub stage1: Stage1Kernels<F128>,
    pub stage2: Stage2Kernels<F128>,
    pub stage3a: Stage3aKernels<F128>,
    pub stage3b: Stage3bKernels<F128>,
    pub stage4: Stage4Kernels<F128>,
    pub stage5: Stage5Kernels<F128>,
    pub stage6a: Stage6aKernels<F128>,
    pub stage6b: Stage6bKernels<F128>,
}
impl Rv64iBackend {
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
