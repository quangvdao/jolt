//! Batch 1 aligns both row blocks on their shared cycle rounds.
pub mod spartan_outer;
pub mod verify;
pub use verify::Output;

use crate::points::PointsError;
use crate::public::matrices::RowMatrices;
use crate::statement::LOG_T_MAX;
use jolt_field::{JoltField, Zero, F128};
use jolt_rv64i_arith::Layout;
use jolt_verifier::stages::relations::SumcheckBatch;
pub use spartan_outer::{SpartanOuterF128, SpartanOuterF2};

/// Generated member order and low-variable-first geometry of this batch.
#[derive(SumcheckBatch)]
pub struct Stage1Sumchecks<F: JoltField> {
    pub spartan_outer_f2: SpartanOuterF2<F>,
    pub spartan_outer_f128: SpartanOuterF128<F>,
}
impl<F: JoltField> Stage1Sumchecks<F> {
    /// Checks both row-first, cycle-last weights against the row widths and `log_T`.
    /// `tau_f2` and `tau_f128` are the two stage-1 draws; invalid dimensions return `PointsError`.
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

use crate::proof::batch_geometry;
stage1_sumchecks_members!(batch_geometry);

impl Stage1Sumchecks<F128> {
    /// Constructs bounded zero weights solely to read the canonical outer member geometry.
    /// Invalid trace or row dimensions return `PointsError`.
    pub fn for_geometry(log_T: usize, layout: &Layout) -> Result<Self, PointsError> {
        if !(1..=usize::from(LOG_T_MAX)).contains(&log_T) {
            return Err(PointsError::Dimension {
                expected: usize::from(LOG_T_MAX),
                actual: log_T,
            });
        }
        let m_F = RowMatrices::f128_row_variables_for(layout);
        Self::new(
            log_T,
            m_F,
            vec![F128::zero(); 8 + log_T],
            vec![F128::zero(); m_F + log_T],
        )
    }
}
