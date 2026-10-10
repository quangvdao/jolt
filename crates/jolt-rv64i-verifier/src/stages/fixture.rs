//! One-member fixture that exercises the verifier/prover ownership split.
//! This is not a protocol batch, and its transcript is not the wire of batch 6b.

use jolt_field::JoltField;
use jolt_verifier::stages::relations::SumcheckBatch;

pub use crate::stages::stage6b::bits_reduction::{
    BitsReduction, BitsReductionChallenges, BitsReductionInputClaims, BitsReductionOutputClaims,
};

#[derive(SumcheckBatch)]
pub struct ReductionOnlySumchecks<F: JoltField> {
    pub bits_reduction: BitsReduction<F>,
}
