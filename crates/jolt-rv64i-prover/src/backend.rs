//! Kernel registries for the eight ordered reduction batches.
use crate::optimized::{
    outer::SpartanOuterF2Prepare,
    routers::{
        RouterCycleBranchPrepare, RouterCycleComparePrepare, RouterCycleMemoryPrepare,
        RouterCycleShiftPrepare, RouterCycleVariantPrepare, RouterShortPrepare,
    },
    tail::{BitsReductionPrepare, BytecodeReadCyclePrepare, RamRaProductPrepare},
};
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
    /// Replaces the outer, six router and three tail members; every other member stays reference.
    /// `mixed_registry_proofs_match_reference` pins its proof bytes to those of `reference()`.
    pub fn optimized() -> Self {
        let mut backend = Self::reference();
        backend.stage1.spartan_outer_f2 = Box::new(SpartanOuterF2Prepare);
        backend.stage3a.router_short = Box::new(RouterShortPrepare);
        backend.stage3b.variant = Box::new(RouterCycleVariantPrepare);
        backend.stage3b.shift = Box::new(RouterCycleShiftPrepare);
        backend.stage3b.memory = Box::new(RouterCycleMemoryPrepare);
        backend.stage3b.compare = Box::new(RouterCycleComparePrepare);
        backend.stage3b.branch = Box::new(RouterCycleBranchPrepare);
        backend.stage6b.bytecode_read_cycle = Box::new(BytecodeReadCyclePrepare);
        backend.stage6b.ram_ra_product = Box::new(RamRaProductPrepare);
        backend.stage6b.bits_reduction = Box::new(BitsReductionPrepare);
        backend
    }

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
