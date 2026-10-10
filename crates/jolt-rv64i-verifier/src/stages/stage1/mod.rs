//! Batch 1 aligns both row blocks on their shared cycle rounds.
pub mod spartan_outer;
pub mod verify;

use crate::points::PointsError;
use jolt_field::JoltField;
use jolt_verifier::stages::relations::SumcheckBatch;
pub use spartan_outer::{SpartanOuterF128, SpartanOuterF2};

#[derive(SumcheckBatch)]
pub struct Stage1Sumchecks<F: JoltField> {
    pub spartan_outer_f2: SpartanOuterF2<F>,
    pub spartan_outer_f128: SpartanOuterF128<F>,
}
impl<F: JoltField> Stage1Sumchecks<F> {
    pub fn new(
        log_T: usize,
        m_F: usize,
        tau_f2: Vec<F>,
        tau_f128: Vec<F>,
    ) -> Result<Self, PointsError> {
        Ok(Self {
            spartan_outer_f2: SpartanOuterF2::new(log_T, 8, tau_f2)?,
            spartan_outer_f128: SpartanOuterF128::new(log_T, m_F, tau_f128)?,
        })
    }
}
