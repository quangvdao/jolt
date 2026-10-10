//! Concrete registers read checking with address variables before cycles.
use super::verify::{check_read_points, evaluate_output, term_error};
use crate::claims::registers_read_checking::RegistersReadCheckingSymbolic;
pub use crate::claims::registers_read_checking::{
    RegistersReadCheckingChallenges, RegistersReadCheckingInputClaims,
    RegistersReadCheckingOutputClaims,
};
use crate::ids::{DerivedId, ReadCheckingDerived};
use crate::points::{self, PointsError};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::VerifierError;

/// Read checking over low-variable-first five register addresses followed by cycles, at the shared batch-3a bit point.
/// The read claims and their bit-then-cycle points must come from batch 3b.
#[derive(Clone)]
pub struct RegistersReadChecking<F: JoltField> {
    symbolic: RegistersReadCheckingSymbolic,
    r_bit: Vec<F>,
    r_3: Vec<F>,
}

impl<F: JoltField> RegistersReadChecking<F> {
    /// Establishes six bit coordinates and a nonempty cycle point bounded by `LOG_T_MAX`, returning `PointsError` on mismatch.
    /// The points must be those verified by batches 3a and 3b.
    pub fn new(r_bit: Vec<F>, r_3: Vec<F>) -> Result<Self, PointsError> {
        check_read_points(&r_bit, &r_3)?;
        Ok(Self {
            symbolic: RegistersReadCheckingSymbolic::new(r_3.len()),
            r_bit,
            r_3,
        })
    }
    /// The six low-variable-first bit coordinates verified by batch 3a.
    pub fn r_bit(&self) -> &[F] {
        &self.r_bit
    }
    /// The low-variable-first cycle coordinates verified by batch 3b.
    pub fn r_3(&self) -> &[F] {
        &self.r_3
    }
    /// Consumed read points in six-bit-then-cycle order, low variable first in each part.
    pub fn input_points(&self) -> RegistersReadCheckingInputClaims<Vec<F>> {
        let point: Vec<_> = self.r_bit.iter().chain(&self.r_3).copied().collect();
        RegistersReadCheckingInputClaims {
            rs1_value: point.clone(),
            rs2_value: point.clone(),
            rd_pre_value: point,
        }
    }
}
impl<F: JoltField> ConcreteSumcheck<F> for RegistersReadChecking<F> {
    type Symbolic = RegistersReadCheckingSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &RegistersReadCheckingInputClaims<Vec<F>>,
    ) -> Result<RegistersReadCheckingOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(term_error(PointsError::Dimension {
                expected: self.rounds(),
                actual: point.len(),
            }));
        }
        let address = point.get(..5).ok_or_else(|| {
            term_error(PointsError::Dimension {
                expected: 5,
                actual: point.len(),
            })
        })?;
        let cycle = point.get(5..).ok_or_else(|| {
            term_error(PointsError::Dimension {
                expected: 5,
                actual: point.len(),
            })
        })?;
        Ok(RegistersReadCheckingOutputClaims {
            rs1_ra: point.to_vec(),
            rs2_ra: point.to_vec(),
            rd_wa: point.to_vec(),
            registers_val: address
                .iter()
                .chain(&self.r_bit)
                .chain(cycle)
                .copied()
                .collect(),
        })
    }
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "foreign derived ids fail closed as missing claims"
    )]
    fn derive_output_term(
        &self,
        id: &DerivedId,
        _inputs: &RegistersReadCheckingInputClaims<Vec<F>>,
        outputs: &RegistersReadCheckingOutputClaims<Vec<F>>,
        _challenges: &RegistersReadCheckingChallenges<F>,
    ) -> Result<F, VerifierError> {
        match id {
            DerivedId::RegistersReadChecking(ReadCheckingDerived::EqCycle) => {
                let point = outputs.rs1_ra.get(5..).ok_or_else(|| {
                    term_error(PointsError::Dimension {
                        expected: self.rounds(),
                        actual: outputs.rs1_ra.len(),
                    })
                })?;
                points::eq(&self.r_3, point).map_err(term_error)
            }
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        }
    }
    fn expected_output(
        &self,
        inputs: &RegistersReadCheckingInputClaims<Vec<F>>,
        values: &RegistersReadCheckingOutputClaims<F>,
        outputs: &RegistersReadCheckingOutputClaims<Vec<F>>,
        challenges: &RegistersReadCheckingChallenges<F>,
    ) -> Result<F, VerifierError> {
        evaluate_output(
            self.symbolic.output_expression(),
            values,
            challenges,
            |id| self.derive_output_term(id, inputs, outputs, challenges),
        )
    }
}
