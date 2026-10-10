//! Batch 3b binds all five router cycle relations at one cycle point.

pub mod router_cycle;
pub mod verify;
pub use verify::Output;

use crate::points::PointsError;
use crate::statement::LOG_T_MAX;
use jolt_field::{JoltField, Zero, F128};
use jolt_rv64i_arith::Layout;
use jolt_verifier::stages::relations::SumcheckBatch;
pub use router_cycle::{
    RouterCycleBranch, RouterCycleBranchInputClaims, RouterCycleBranchOutputClaims,
    RouterCycleCompare, RouterCycleCompareInputClaims, RouterCycleCompareOutputClaims,
    RouterCycleMemory, RouterCycleMemoryInputClaims, RouterCycleMemoryOutputClaims,
    RouterCycleShift, RouterCycleShiftInputClaims, RouterCycleShiftOutputClaims,
    RouterCycleVariant, RouterCycleVariantInputClaims, RouterCycleVariantOutputClaims,
};

/// Generated member order and low-variable-first geometry of this batch.
#[derive(SumcheckBatch)]
pub struct Stage3bSumchecks<F: JoltField> {
    pub variant: RouterCycleVariant<F>,
    pub shift: RouterCycleShift<F>,
    pub memory: RouterCycleMemory<F>,
    pub compare: RouterCycleCompare<F>,
    pub branch: RouterCycleBranch<F>,
}

impl<F: JoltField> Stage3bSumchecks<F> {
    /// Checks the seventeen short slots `x` and the cycle point `r_1` from stage 1.
    /// Every member binds cycle variables low first; invalid dimensions return `PointsError`.
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

use crate::proof::batch_geometry;
stage3b_sumchecks_members!(batch_geometry);

impl Stage3bSumchecks<F128> {
    /// Constructs bounded zero short and cycle points solely to read the canonical router member geometry.
    /// Invalid point dimensions return `PointsError`.
    pub fn for_geometry(log_T: usize, layout: &Layout) -> Result<Self, PointsError> {
        if !(1..=usize::from(LOG_T_MAX)).contains(&log_T) {
            return Err(PointsError::Dimension {
                expected: usize::from(LOG_T_MAX),
                actual: log_T,
            });
        }
        use crate::claims::router_short::RouterShortSymbolic;
        use jolt_claims::SymbolicSumcheck as _;
        Self::new(
            layout,
            vec![F128::zero(); log_T],
            vec![F128::zero(); RouterShortSymbolic::new(()).rounds()],
        )
    }
}
