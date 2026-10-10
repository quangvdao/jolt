//! Batch 2 reduces the outer values over the witness columns.
pub mod spartan_inner;
pub mod verify;
pub use verify::Output;

use crate::points::PointsError;
use crate::public::matrices::RowMatrices;
use jolt_field::JoltField;
use jolt_verifier::stages::relations::SumcheckBatch;
pub use spartan_inner::SpartanInner;
use std::sync::Arc;

/// Generated member order and low-variable-first geometry of this batch.
#[derive(SumcheckBatch)]
pub struct Stage2Sumchecks<F: JoltField> {
    pub spartan_inner: SpartanInner<F>,
}
impl<F: JoltField> Stage2Sumchecks<F> {
    /// Checks the outer row points and their shared cycle point before reducing witness columns.
    /// All points use low-variable-first order; invalid dimensions return `PointsError`.
    pub fn new(
        matrices: Arc<RowMatrices>,
        rho_f2: Vec<F>,
        rho_f128: Vec<F>,
        r_1: Vec<F>,
    ) -> Result<Self, PointsError> {
        Ok(Self {
            spartan_inner: SpartanInner::new(matrices, rho_f2, rho_f128, r_1)?,
        })
    }
}

use crate::proof::unit_batch_geometry;
stage2_sumchecks_members!(unit_batch_geometry);
