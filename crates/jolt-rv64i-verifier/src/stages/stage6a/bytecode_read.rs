//! Concrete address phase of public bytecode read checking.

use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::VerifierError;

use crate::claims::bytecode_read::BytecodeReadAddressSymbolic;
pub use crate::claims::bytecode_read::{
    BytecodeReadAddressChallenges, BytecodeReadAddressInputClaims, BytecodeReadAddressOutputClaims,
};
use crate::ids::{BytecodeAddressDerived, DerivedId};
use crate::points::{self, PointsError};
pub use crate::public::bytecode::BytecodeReadPoints;

#[derive(Clone)]
pub struct BytecodeReadAddress<F: JoltField> {
    symbolic: BytecodeReadAddressSymbolic,
    points: BytecodeReadPoints<F>,
    entry_pc: F,
    final_pc: F,
}

impl<F: JoltField> BytecodeReadAddress<F> {
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
        let entry_pc = points::lift(entry_pc, &points.r_bit)?;
        let final_pc =
            points.r_3.iter().copied().product::<F>() * points::lift(final_pc, &points.r_bit)?;
        Ok(Self {
            symbolic: BytecodeReadAddressSymbolic::new(log_K_bytecode),
            points,
            entry_pc,
            final_pc,
        })
    }

    pub fn public_points(&self) -> &BytecodeReadPoints<F> {
        &self.points
    }
    pub fn input_points(&self) -> BytecodeReadAddressInputClaims<Vec<F>> {
        self.points.input_points()
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for BytecodeReadAddress<F> {
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
