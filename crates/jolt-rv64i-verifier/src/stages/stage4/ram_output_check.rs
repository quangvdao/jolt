//! Final RAM agreement with the checked public I/O interval.
use super::verify::{evaluate_output, term_error};
use crate::claims::ram_output_check::RamOutputCheckSymbolic;
pub use crate::claims::ram_output_check::{
    RamOutputCheckChallenges, RamOutputCheckInputClaims, RamOutputCheckOutputClaims,
};
use crate::ids::{DerivedId, OutputCheckDerived};
use crate::points::{self, PointsError};
use crate::public::io::{io_mask, val_io, validate_io};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;
use jolt_program::preprocess::PublicIoMemory;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::VerifierError;
use std::sync::Arc;

/// Final-RAM agreement over low-variable-first addresses, at one fixed six-coordinate bit point.
/// `new` checks the point bounds and public I/O interval while sharing its words.
#[derive(Clone)]
pub struct RamOutputCheck<F: JoltField> {
    symbolic: RamOutputCheckSymbolic,
    tau: Vec<F>,
    r_bit: Vec<F>,
    io: Arc<PublicIoMemory>,
}
impl<F: JoltField> RamOutputCheck<F> {
    /// Establishes a six-coordinate bit point and at least five address coordinates.
    /// Rejects malformed dimensions or an I/O interval outside that address cube.
    pub fn new(
        tau: Vec<F>,
        r_bit: Vec<F>,
        io: impl Into<Arc<PublicIoMemory>>,
    ) -> Result<Self, PointsError> {
        let io = io.into();
        if r_bit.len() != 6 {
            return Err(PointsError::Dimension {
                expected: 6,
                actual: r_bit.len(),
            });
        }
        if tau.len() < 5 {
            return Err(PointsError::Dimension {
                expected: 5,
                actual: tau.len(),
            });
        }
        validate_io(&io, tau.len())?;
        Ok(Self {
            symbolic: RamOutputCheckSymbolic::new(tau.len()),
            tau,
            r_bit,
            io,
        })
    }
    /// Low-variable-first output-check address challenge.
    pub fn tau(&self) -> &[F] {
        &self.tau
    }
    /// Six bit coordinates, low variable first, shared by public and final words.
    pub fn r_bit(&self) -> &[F] {
        &self.r_bit
    }
    /// Public I/O words and mask validated by `new`, borrowed from the shared allocation.
    pub fn io(&self) -> &PublicIoMemory {
        &self.io
    }
}
impl<F: JoltField> ConcreteSumcheck<F> for RamOutputCheck<F> {
    type Symbolic = RamOutputCheckSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn instance_point_offset(&self, batch_num_vars: usize) -> Result<usize, VerifierError> {
        if batch_num_vars < self.rounds() {
            return Err(term_error(PointsError::Dimension {
                expected: self.rounds(),
                actual: batch_num_vars,
            }));
        }
        Ok(0)
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &RamOutputCheckInputClaims<Vec<F>>,
    ) -> Result<RamOutputCheckOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(term_error(PointsError::Dimension {
                expected: self.rounds(),
                actual: point.len(),
            }));
        }
        Ok(RamOutputCheckOutputClaims {
            ram_val_final: point.iter().chain(&self.r_bit).copied().collect(),
        })
    }
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "foreign derived ids fail closed as missing claims"
    )]
    fn derive_output_term(
        &self,
        id: &DerivedId,
        _inputs: &RamOutputCheckInputClaims<Vec<F>>,
        outputs: &RamOutputCheckOutputClaims<Vec<F>>,
        _challenges: &RamOutputCheckChallenges<F>,
    ) -> Result<F, VerifierError> {
        let address = outputs.ram_val_final.get(..self.rounds()).ok_or_else(|| {
            term_error(PointsError::Dimension {
                expected: self.rounds(),
                actual: outputs.ram_val_final.len(),
            })
        })?;
        match id {
            DerivedId::RamOutputCheck(OutputCheckDerived::EqTau) => {
                points::eq(&self.tau, address).map_err(term_error)
            }
            DerivedId::RamOutputCheck(OutputCheckDerived::IoMask) => {
                io_mask(&self.io, address).map_err(term_error)
            }
            DerivedId::RamOutputCheck(OutputCheckDerived::ValIo) => {
                val_io(&self.io, address, &self.r_bit).map_err(term_error)
            }
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        }
    }
    fn expected_output(
        &self,
        inputs: &RamOutputCheckInputClaims<Vec<F>>,
        values: &RamOutputCheckOutputClaims<F>,
        outputs: &RamOutputCheckOutputClaims<Vec<F>>,
        challenges: &RamOutputCheckChallenges<F>,
    ) -> Result<F, VerifierError> {
        evaluate_output(
            self.symbolic.output_expression(),
            values,
            challenges,
            |id| self.derive_output_term(id, inputs, outputs, challenges),
        )
    }
}
