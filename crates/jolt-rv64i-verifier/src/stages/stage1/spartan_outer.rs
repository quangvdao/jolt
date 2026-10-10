//! Concrete row-block zero-checks with a shared equality-weight implementation.

pub use crate::claims::spartan_outer::{
    SpartanOuterF128OutputClaims, SpartanOuterF2OutputClaims, SpartanOuterInputClaims,
};
use crate::claims::spartan_outer::{SpartanOuterF128Symbolic, SpartanOuterF2Symbolic};
use crate::ids::{DerivedId, OuterDerived, RowBlock};
use crate::points::{eq, PointsError};
use crate::statement::LOG_T_MAX;
use jolt_claims::{NoChallenges, OutputClaims, SumcheckChallenges, SymbolicSumcheck};
use jolt_field::JoltField;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::VerifierError;

#[derive(Clone)]
struct OuterInstance<F> {
    row_variables: usize,
    tau: Vec<F>,
}
impl<F: JoltField> OuterInstance<F> {
    fn new(log_T: usize, row_variables: usize, tau: Vec<F>) -> Result<Self, PointsError> {
        if !(1..=usize::from(LOG_T_MAX)).contains(&log_T) {
            return Err(PointsError::Dimension {
                expected: usize::from(LOG_T_MAX),
                actual: log_T,
            });
        }
        if row_variables > 8 {
            return Err(PointsError::Dimension {
                expected: 8,
                actual: row_variables,
            });
        }
        let expected = row_variables + log_T;
        if tau.len() != expected {
            return Err(PointsError::Dimension {
                expected,
                actual: tau.len(),
            });
        }
        Ok(Self { row_variables, tau })
    }
    fn term_error(error: PointsError) -> VerifierError {
        VerifierError::StageClaimSumcheckFailed {
            stage: "SpartanOuter".to_owned(),
            reason: error.to_string(),
        }
    }
}
macro_rules! outer {
    ($name:ident, $symbolic:ident, $outputs:ident, $block:ident) => {
        #[derive(Clone)]
        pub struct $name<F: JoltField> {
            symbolic: $symbolic,
            instance: OuterInstance<F>,
        }
        impl<F: JoltField> $name<F> {
            pub fn new(
                log_T: usize,
                row_variables: usize,
                tau: Vec<F>,
            ) -> Result<Self, PointsError> {
                let instance = OuterInstance::new(log_T, row_variables, tau)?;
                if RowBlock::$block == RowBlock::F2 && row_variables != 8 {
                    return Err(PointsError::Dimension {
                        expected: 8,
                        actual: row_variables,
                    });
                }
                Ok(Self {
                    symbolic: $symbolic::new(instance.tau.len()),
                    instance,
                })
            }
            pub fn tau(&self) -> &[F] {
                &self.instance.tau
            }
            pub fn row_variables(&self) -> usize {
                self.instance.row_variables
            }
            pub fn block(&self) -> RowBlock {
                RowBlock::$block
            }
        }
        impl<F: JoltField> ConcreteSumcheck<F> for $name<F> {
            type Symbolic = $symbolic;
            fn symbolic(&self) -> &Self::Symbolic {
                &self.symbolic
            }
            fn derive_opening_points(
                &self,
                point: &[F],
                _inputs: &SpartanOuterInputClaims<Vec<F>>,
            ) -> Result<$outputs<Vec<F>>, VerifierError> {
                if point.len() != self.rounds() {
                    return Err(OuterInstance::<F>::term_error(PointsError::Dimension {
                        expected: self.rounds(),
                        actual: point.len(),
                    }));
                }
                Ok($outputs {
                    az: point.to_vec(),
                    bz: point.to_vec(),
                    cz: point.to_vec(),
                })
            }
            fn expected_output(
                &self,
                input_points: &SpartanOuterInputClaims<Vec<F>>,
                output_values: &$outputs<F>,
                output_points: &$outputs<Vec<F>>,
                challenges: &NoChallenges<F>,
            ) -> Result<F, VerifierError> {
                let id = DerivedId::SpartanOuter(RowBlock::$block, OuterDerived::EqTau);
                let weight =
                    self.derive_output_term(&id, input_points, output_points, challenges)?;
                self.symbolic().output_expression::<F>().try_evaluate(
                    |id| {
                        output_values
                            .resolve_output(id)
                            .ok_or(VerifierError::MissingOpeningClaim { id: (*id).into() })
                    },
                    |id| {
                        challenges
                            .resolve_challenge(id)
                            .ok_or(VerifierError::MissingStageClaimChallenge { id: (*id).into() })
                    },
                    |requested| {
                        if *requested == id {
                            Ok(weight)
                        } else {
                            Err(VerifierError::MissingStageClaimDerived {
                                id: (*requested).into(),
                            })
                        }
                    },
                )
            }
            fn derive_output_term(
                &self,
                id: &DerivedId,
                _inputs: &SpartanOuterInputClaims<Vec<F>>,
                outputs: &$outputs<Vec<F>>,
                _challenges: &NoChallenges<F>,
            ) -> Result<F, VerifierError> {
                match id {
                    DerivedId::SpartanOuter(RowBlock::$block, OuterDerived::EqTau) => {
                        eq(self.tau(), &outputs.az).map_err(OuterInstance::<F>::term_error)
                    }
                    _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
                }
            }
        }
    };
}
outer!(
    SpartanOuterF2,
    SpartanOuterF2Symbolic,
    SpartanOuterF2OutputClaims,
    F2
);
outer!(
    SpartanOuterF128,
    SpartanOuterF128Symbolic,
    SpartanOuterF128OutputClaims,
    F128
);
