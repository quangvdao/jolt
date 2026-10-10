//! Three-member terminal batch with column projections and a shared opening.
pub mod bits_reduction;
pub mod bytecode_read_cycle;
pub mod ram_ra_product;
pub mod verify;
pub use verify::Output;

use crate::points::PointsError;
use crate::statement::LOG_T_MAX;
pub use bits_reduction::{
    BitsReduction, BitsReductionChallenges, BitsReductionInputClaims, BitsReductionOutputClaims,
};
pub use bytecode_read_cycle::{
    BytecodeReadCycle, BytecodeReadCycleInputClaims, BytecodeReadCycleOutputClaims,
};
use jolt_field::JoltField;
use jolt_rv64i_arith::{Layout, BITS_COLUMNS};
use jolt_transcript::Transcript;
use jolt_verifier::stages::relations::{ConcreteSumcheck, SumcheckBatch};
use jolt_verifier::VerifierError;
pub use ram_ra_product::{
    RamRaProduct, RamRaProductChallenges, RamRaProductInputClaims, RamRaProductOutputClaims,
};

/// Generated terminal member order, with cycle variables bound low first in every member.
#[derive(SumcheckBatch)]
#[sumcheck_batch(no_draw_challenges, no_opening_values, no_output_shape)]
pub struct Stage6bSumchecks<F: JoltField> {
    pub bytecode_read_cycle: BytecodeReadCycle<F>,
    pub ram_ra_product: RamRaProduct<F>,
    pub bits_reduction: BitsReduction<F>,
}
impl<F: JoltField> Stage6bSumchecks<F> {
    /// Constructs dimension-only members for the generated schedule without a statement or witness.
    /// Concrete constructors validate all short, chunk and cycle dimensions from the layout.
    pub fn for_geometry(log_T: usize, layout: &Layout) -> Result<Self, PointsError> {
        if log_T > usize::from(LOG_T_MAX) {
            return Err(PointsError::Dimension {
                expected: usize::from(LOG_T_MAX),
                actual: log_T,
            });
        }
        Ok(Self {
            bytecode_read_cycle: BytecodeReadCycle::new(
                layout,
                [F::zero(); 5],
                vec![F::zero(); layout.log_K_bytecode()],
                vec![F::zero(); log_T],
                vec![F::zero(); log_T],
                vec![F::zero(); log_T],
            )?,
            ram_ra_product: RamRaProduct::new(
                layout,
                vec![F::zero(); layout.log_K_ram()],
                vec![F::zero(); log_T],
                vec![F::zero(); log_T],
            )?,
            bits_reduction: BitsReduction::new(
                layout,
                vec![F::zero(); log_T],
                vec![F::zero(); log_T],
                vec![F::zero(); log_T],
                vec![F::zero(); 10],
                vec![F::zero(); 17],
            )?,
        })
    }

    /// Draw the six committed-functional coefficients before the two RAM coefficients.
    pub fn draw_challenges<T: Transcript<Challenge = F>>(
        &self,
        transcript: &mut T,
    ) -> Result<Stage6bChallenges<F>, VerifierError> {
        let bits_reduction = self.bits_reduction.draw_challenges(transcript)?;
        let ram_ra_product = self.ram_ra_product.draw_challenges(transcript)?;
        let bytecode_read_cycle = self.bytecode_read_cycle.draw_challenges(transcript)?;
        Ok(Stage6bChallenges {
            bytecode_read_cycle,
            ram_ra_product,
            bits_reduction,
        })
    }
    /// Reconstruct both chunk vectors from exactly 256 committed column evaluations.
    pub fn expand(&self, columns: &[F]) -> Result<Stage6bOutputClaims<F>, VerifierError> {
        if columns.len() != BITS_COLUMNS {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage: "Stage6b".to_owned(),
                reason: format!(
                    "column vector has {} values, expected {BITS_COLUMNS}",
                    columns.len()
                ),
            });
        }
        Ok(Stage6bOutputClaims {
            bytecode_read_cycle: self
                .bytecode_read_cycle
                .project(columns)
                .map_err(BytecodeReadCycle::<F>::term_error)?,
            ram_ra_product: self
                .ram_ra_product
                .project(columns)
                .map_err(RamRaProduct::<F>::term_error)?,
            bits_reduction: BitsReductionOutputClaims {
                columns: columns.to_vec(),
            },
        })
    }
    pub fn opening_values(&self, claims: &Stage6bOutputClaims<F>) -> Vec<F> {
        claims.bits_reduction.columns.clone()
    }
    pub fn append_output_claims<T: Transcript<Challenge = F>>(
        &self,
        transcript: &mut T,
        claims: &Stage6bOutputClaims<F>,
    ) {
        for value in &claims.bits_reduction.columns {
            transcript.append_labeled(b"opening_claim", value);
        }
    }
}

use crate::proof::batch_geometry;
stage6b_sumchecks_members!(batch_geometry);
