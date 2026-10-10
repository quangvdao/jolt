//! Batch 3b binds all five router cycle relations at one cycle point.

pub mod router_cycle;
pub mod verify;

use crate::points::PointsError;
use jolt_field::JoltField;
use jolt_rv64i_arith::Layout;
use jolt_verifier::stages::relations::SumcheckBatch;
pub use router_cycle::{
    RouterCycleBranch, RouterCycleBranchInputClaims, RouterCycleBranchOutputClaims,
    RouterCycleCompare, RouterCycleCompareInputClaims, RouterCycleCompareOutputClaims,
    RouterCycleMemory, RouterCycleMemoryInputClaims, RouterCycleMemoryOutputClaims,
    RouterCycleShift, RouterCycleShiftInputClaims, RouterCycleShiftOutputClaims,
    RouterCycleVariant, RouterCycleVariantInputClaims, RouterCycleVariantOutputClaims,
};

#[derive(SumcheckBatch)]
pub struct Stage3bSumchecks<F: JoltField> {
    pub variant: RouterCycleVariant<F>,
    pub shift: RouterCycleShift<F>,
    pub memory: RouterCycleMemory<F>,
    pub compare: RouterCycleCompare<F>,
    pub branch: RouterCycleBranch<F>,
}

impl<F: JoltField> Stage3bSumchecks<F> {
    pub fn new(layout: &Layout, r_1: Vec<F>, x: Vec<F>) -> Result<Self, PointsError> {
        Ok(Self {
            variant: RouterCycleVariant::new(layout, r_1.clone(), x.clone())?,
            shift: RouterCycleShift::new(layout, r_1.clone(), x.clone())?,
            memory: RouterCycleMemory::new(layout, r_1.clone(), x.clone())?,
            compare: RouterCycleCompare::new(layout, r_1.clone(), x.clone())?,
            branch: RouterCycleBranch::new(layout, r_1, x)?,
        })
    }
    pub fn input_points(&self) -> Result<Stage3bInputPoints<F>, PointsError> {
        Ok(Stage3bInputPoints {
            variant: self.variant.input_points()?,
            shift: self.shift.input_points()?,
            memory: self.memory.input_points()?,
            compare: self.compare.input_points()?,
            branch: self.branch.input_points()?,
        })
    }
}
