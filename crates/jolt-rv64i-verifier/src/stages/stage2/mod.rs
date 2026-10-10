//! Batch 2 reduces the outer values over the witness columns.
pub mod spartan_inner;
pub mod verify;

use crate::points::PointsError;
use crate::public::matrices::RowMatrices;
use jolt_field::JoltField;
use jolt_verifier::stages::relations::SumcheckBatch;
pub use spartan_inner::SpartanInner;
use std::sync::Arc;

#[derive(SumcheckBatch)]
pub struct Stage2Sumchecks<F: JoltField> {
    pub spartan_inner: SpartanInner<F>,
}
impl<F: JoltField> Stage2Sumchecks<F> {
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
