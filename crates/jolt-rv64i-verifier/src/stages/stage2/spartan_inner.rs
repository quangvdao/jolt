//! Concrete column reduction at the outer row and cycle points.

use crate::claims::spartan_inner::SpartanInnerSymbolic;
pub use crate::claims::spartan_inner::{
    SpartanInnerChallenges, SpartanInnerInputClaims, SpartanInnerOutputClaims,
};
use crate::ids::{DerivedId, InnerDerived};
use crate::points::PointsError;
use crate::proof::DimensionedRelation;
use crate::public::matrices::RowMatrices;
use crate::statement::LOG_T_MAX;
use jolt_claims::{OutputClaims, SumcheckChallenges, SymbolicSumcheck};
use jolt_field::JoltField;
use jolt_rv64i_arith::Layout;
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::VerifierError;
use std::sync::Arc;

/// Reduction over ten low-variable-first witness-column coordinates, fixed at the verified batch-1 row and cycle points.
/// The matrices must be those of the checked layout; `new` validates the supplied point widths.
#[derive(Clone)]
pub struct SpartanInner<F: JoltField> {
    symbolic: SpartanInnerSymbolic,
    matrices: Arc<RowMatrices>,
    rho_f2: Vec<F>,
    rho_f128: Vec<F>,
    r_1: Vec<F>,
}
struct InnerTerms<F> {
    matrix_weight: F,
    public_columns: F,
}

impl<F: Copy> InnerTerms<F> {
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "foreign derived ids fail closed as missing claims"
    )]
    fn resolve(&self, id: &DerivedId) -> Result<F, VerifierError> {
        match id {
            DerivedId::SpartanInner(InnerDerived::MatrixWeight) => Ok(self.matrix_weight),
            DerivedId::SpartanInner(InnerDerived::PublicColumns) => Ok(self.public_columns),
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        }
    }
}

impl<F: JoltField> SpartanInner<F> {
    /// Establishes eight binary row coordinates, the matrices' field-row width, and a cycle width in `1..=LOG_T_MAX`.
    /// The row and cycle points must come from batch 1; inconsistent dimensions return `PointsError`.
    pub fn new(
        matrices: Arc<RowMatrices>,
        rho_f2: Vec<F>,
        rho_f128: Vec<F>,
        r_1: Vec<F>,
    ) -> Result<Self, PointsError> {
        for (point, expected) in [(&rho_f2, 8), (&rho_f128, matrices.f128_row_variables())] {
            if point.len() != expected {
                return Err(PointsError::Dimension {
                    expected,
                    actual: point.len(),
                });
            }
        }
        if !(1..=usize::from(LOG_T_MAX)).contains(&r_1.len()) {
            return Err(PointsError::Dimension {
                expected: usize::from(LOG_T_MAX),
                actual: r_1.len(),
            });
        }
        Ok(Self {
            symbolic: SpartanInnerSymbolic,
            matrices,
            rho_f2,
            rho_f128,
            r_1,
        })
    }
    /// The verified batch-1 cycle point, with low variables first.
    pub fn r_1(&self) -> &[F] {
        &self.r_1
    }
    /// The folded public matrix evaluation at ten low-variable-first column coordinates.
    /// Rejects a column point of the wrong width; coefficients come from this batch's challenge draw.
    pub fn matrix_weight(
        &self,
        w: &[F],
        challenges: &SpartanInnerChallenges<F>,
    ) -> Result<F, PointsError> {
        Ok(self.terms_at(w, challenges)?.matrix_weight)
    }

    fn terms_at(
        &self,
        w: &[F],
        challenges: &SpartanInnerChallenges<F>,
    ) -> Result<InnerTerms<F>, PointsError> {
        let evaluations = self.matrices.evaluate(&self.rho_f2, &self.rho_f128, w)?;
        let [f2, f128] = evaluations.blocks;
        let matrix_weight = f2
            .into_iter()
            .chain(f128)
            .zip([
                challenges.az_f2,
                challenges.bz_f2,
                challenges.cz_f2,
                challenges.az_f128,
                challenges.bz_f128,
                challenges.cz_f128,
            ])
            .map(|(value, coefficient)| value * coefficient)
            .sum();
        Ok(InnerTerms {
            matrix_weight,
            public_columns: evaluations.public_columns,
        })
    }

    fn output_terms(
        &self,
        outputs: &SpartanInnerOutputClaims<Vec<F>>,
        challenges: &SpartanInnerChallenges<F>,
    ) -> Result<InnerTerms<F>, VerifierError> {
        let full = &outputs.witness_routed;
        if full.len() != self.rounds() + self.r_1.len() {
            return Err(Self::term_error(PointsError::Dimension {
                expected: self.rounds() + self.r_1.len(),
                actual: full.len(),
            }));
        }
        let w = full.get(..self.rounds()).ok_or_else(|| {
            Self::term_error(PointsError::Dimension {
                expected: self.rounds(),
                actual: full.len(),
            })
        })?;
        self.terms_at(w, challenges).map_err(Self::term_error)
    }

    fn term_error(error: PointsError) -> VerifierError {
        VerifierError::StageClaimSumcheckFailed {
            stage: "SpartanInner".to_owned(),
            reason: error.to_string(),
        }
    }
}
impl<F: JoltField> ConcreteSumcheck<F> for SpartanInner<F> {
    fn instance_point_offset(&self, batch_num_vars: usize) -> Result<usize, VerifierError> {
        Self::point_offset(self.rounds(), batch_num_vars)
    }

    type Symbolic = SpartanInnerSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &SpartanInnerInputClaims<Vec<F>>,
    ) -> Result<SpartanInnerOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(Self::term_error(PointsError::Dimension {
                expected: self.rounds(),
                actual: point.len(),
            }));
        }
        let full = point.iter().chain(&self.r_1).copied().collect::<Vec<_>>();
        Ok(SpartanInnerOutputClaims {
            witness_routed: full.clone(),
            direct_columns: full,
        })
    }
    fn expected_output(
        &self,
        _input_points: &SpartanInnerInputClaims<Vec<F>>,
        output_values: &SpartanInnerOutputClaims<F>,
        output_points: &SpartanInnerOutputClaims<Vec<F>>,
        challenges: &SpartanInnerChallenges<F>,
    ) -> Result<F, VerifierError> {
        let terms = self.output_terms(output_points, challenges)?;
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
            |id| terms.resolve(id),
        )
    }
    fn derive_output_term(
        &self,
        id: &DerivedId,
        _inputs: &SpartanInnerInputClaims<Vec<F>>,
        outputs: &SpartanInnerOutputClaims<Vec<F>>,
        challenges: &SpartanInnerChallenges<F>,
    ) -> Result<F, VerifierError> {
        self.output_terms(outputs, challenges)?.resolve(id)
    }
}

impl<F: JoltField> DimensionedRelation<F> for SpartanInner<F> {
    fn symbolic_for(_log_T: usize, _layout: &Layout) -> Self::Symbolic {
        SpartanInnerSymbolic::new(())
    }
}
