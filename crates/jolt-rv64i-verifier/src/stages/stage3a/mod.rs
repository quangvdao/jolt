//! Batch 3a reduces the routed witness claim over 17 short slots.
pub mod router_short;
pub mod verify;
use crate::points::PointsError;
use crate::public::routes::RouteTensors;
use jolt_field::JoltField;
use jolt_verifier::stages::relations::SumcheckBatch;
pub use router_short::{RouterShort, RouterShortInputClaims, RouterShortOutputClaims};
use std::sync::Arc;
pub use verify::Output;
/// Generated member order and low-variable-first geometry of this batch.
#[derive(SumcheckBatch)]
pub struct Stage3aSumchecks<F: JoltField> {
    pub router_short: RouterShort<F>,
}
impl<F: JoltField> Stage3aSumchecks<F> {
    /// Checks the ten witness-column coordinates `w`; `r_1` is the verified stage-1 cycle suffix.
    /// The short slots bind low variable first, and an invalid column width returns `PointsError`.
    pub fn new(w: Vec<F>, r_1: Vec<F>, routes: Arc<RouteTensors>) -> Result<Self, PointsError> {
        Ok(Self {
            router_short: RouterShort::new(w, r_1, routes)?,
        })
    }
}
impl<F: JoltField> Stage3aSumchecks<F> {
    pub fn input_points(&self) -> Stage3aInputPoints<F> {
        Stage3aInputPoints {
            router_short: self.router_short.input_points(),
        }
    }
}

use crate::proof::batch_geometry;
stage3a_sumchecks_members!(batch_geometry);
