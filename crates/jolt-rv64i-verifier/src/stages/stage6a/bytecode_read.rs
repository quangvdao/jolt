//! Concrete address phase of public bytecode read checking.

use crate::proof::DimensionedRelation;
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;
use jolt_rv64i_arith::Layout;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::VerifierError;

use crate::claims::bytecode_read::BytecodeReadAddressSymbolic;
pub use crate::claims::bytecode_read::{
    BytecodeReadAddressChallenges, BytecodeReadAddressInputClaims, BytecodeReadAddressOutputClaims,
};
use crate::ids::{BytecodeAddressDerived, DerivedId};
use crate::points::{PointsError, WordLift};
pub use crate::public::bytecode::BytecodeReadPoints;
use crate::public::bytecode::BytecodeWeights;
use std::sync::Arc;

/// Address relation over a low-variable-first bytecode index, using earlier router and state points.
#[derive(Clone)]
pub struct BytecodeReadAddress<F: JoltField> {
    symbolic: BytecodeReadAddressSymbolic,
    points: BytecodeReadPoints<F>,
    lift: Arc<WordLift<F>>,
    entry_pc: F,
    final_pc: F,
}

impl<F: JoltField> BytecodeReadAddress<F> {
    /// Checks the bytecode exponent and the low-variable-first bit, kind, register and cycle points verified by batches 3a, 3b, 4 and 5.
    /// Entry and final PCs come from checked inputs; malformed point widths or an unsupported exponent return `PointsError`.
    pub fn new(
        log_K_bytecode: usize,
        points: BytecodeReadPoints<F>,
        entry_pc: u64,
        final_pc: u64,
    ) -> Result<Self, PointsError> {
        if !(1..=24).contains(&log_K_bytecode) {
            return Err(PointsError::Dimension {
                expected: 24,
                actual: log_K_bytecode,
            });
        }
        points.validate()?;
        let lift = Arc::new(WordLift::new(&points.r_bit)?);
        let entry_pc = lift.evaluate(entry_pc);
        let final_pc = points
            .r_3
            .iter()
            .fold(lift.evaluate(final_pc), |value, coordinate| {
                value * *coordinate
            });
        Ok(Self {
            symbolic: Self::symbolic_with(log_K_bytecode),
            points,
            lift,
            entry_pc,
            final_pc,
        })
    }

    /// Returns the earlier low-variable-first points used by public bytecode folds.
    pub fn public_points(&self) -> &BytecodeReadPoints<F> {
        &self.points
    }
    /// Prepares challenge-weighted public folds at this relation's checked low-variable-first points, sharing its bit-weight table.
    /// The coefficients must be this batch's draws; point or table allocation failures return `PointsError`.
    pub fn public_weights(
        &self,
        coefficients: &BytecodeReadAddressChallenges<F>,
    ) -> Result<BytecodeWeights<F>, PointsError> {
        BytecodeWeights::with_lift(&self.points, coefficients, Arc::clone(&self.lift))
    }

    /// Consumed word, selector and update points have low-variable-first fixed coordinates followed by their upstream cycle point.
    pub fn input_points(&self) -> BytecodeReadAddressInputClaims<Vec<F>> {
        self.points.input_points()
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for BytecodeReadAddress<F> {
    fn instance_point_offset(&self, batch_num_vars: usize) -> Result<usize, VerifierError> {
        Self::point_offset(self.rounds(), batch_num_vars)
    }

    type Symbolic = BytecodeReadAddressSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &BytecodeReadAddressInputClaims<Vec<F>>,
    ) -> Result<BytecodeReadAddressOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage: "BytecodeReadAddress".to_owned(),
                reason: PointsError::Dimension {
                    expected: self.rounds(),
                    actual: point.len(),
                }
                .to_string(),
            });
        }
        Ok(BytecodeReadAddressOutputClaims {
            address_claim: point.to_vec(),
        })
    }
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "foreign derived ids fail closed as missing claims"
    )]
    fn derive_input_term(
        &self,
        id: &DerivedId,
        _challenges: &BytecodeReadAddressChallenges<F>,
    ) -> Result<F, VerifierError> {
        match id {
            DerivedId::BytecodeReadAddress(BytecodeAddressDerived::EntryPc) => Ok(self.entry_pc),
            DerivedId::BytecodeReadAddress(BytecodeAddressDerived::FinalPc) => Ok(self.final_pc),
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        }
    }
    fn derive_output_term(
        &self,
        id: &DerivedId,
        _inputs: &BytecodeReadAddressInputClaims<Vec<F>>,
        _outputs: &BytecodeReadAddressOutputClaims<Vec<F>>,
        _challenges: &BytecodeReadAddressChallenges<F>,
    ) -> Result<F, VerifierError> {
        Err(VerifierError::MissingStageClaimDerived { id: (*id).into() })
    }
}

impl<F: JoltField> DimensionedRelation<F> for BytecodeReadAddress<F> {
    type Dimensions = usize;

    fn symbolic_with(dimensions: Self::Dimensions) -> Self::Symbolic {
        BytecodeReadAddressSymbolic::new(dimensions)
    }

    fn dimensions(_log_T: usize, layout: &Layout) -> Self::Dimensions {
        layout.log_K_bytecode()
    }
}
