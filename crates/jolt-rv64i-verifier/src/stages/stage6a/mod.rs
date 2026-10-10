//! Address-phase batch of public bytecode read checking.
pub mod bytecode_read;
pub mod verify;
pub use verify::Output;

use crate::points::PointsError;
use crate::statement::LOG_T_MAX;
use jolt_field::JoltField;
use jolt_rv64i_arith::Layout;
use jolt_verifier::stages::relations::SumcheckBatch;

pub use bytecode_read::{
    BytecodeReadAddress, BytecodeReadAddressChallenges, BytecodeReadAddressInputClaims,
    BytecodeReadAddressOutputClaims, BytecodeReadPoints,
};

/// Generated address batch; its member binds bytecode address variables low first.
#[derive(SumcheckBatch)]
pub struct Stage6aSumchecks<F: JoltField> {
    pub bytecode_read_address: BytecodeReadAddress<F>,
}

impl<F: JoltField> Stage6aSumchecks<F> {
    /// Constructs dimension-only members for the generated schedule without statement or witness data.
    /// The layout bounds address dimensions; member constructors reject malformed short-point geometry.
    pub fn for_geometry(log_T: usize, layout: &Layout) -> Result<Self, PointsError> {
        if log_T > usize::from(LOG_T_MAX) {
            return Err(PointsError::Dimension {
                expected: usize::from(LOG_T_MAX),
                actual: log_T,
            });
        }
        let points = BytecodeReadPoints::new(
            &[F::zero(); 17],
            vec![F::zero(); 5],
            vec![F::zero(); log_T],
            vec![F::zero(); log_T],
            vec![F::zero(); log_T],
        )?;
        Ok(Self {
            bytecode_read_address: BytecodeReadAddress::new(layout.log_K_bytecode(), points, 0, 0)?,
        })
    }
}

use crate::proof::batch_geometry;
stage6a_sumchecks_members!(batch_geometry);
