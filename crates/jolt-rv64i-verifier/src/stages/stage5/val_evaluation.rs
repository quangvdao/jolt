//! The register and RAM evaluations share their store and increment cells.

pub use crate::claims::val_evaluation::{
    RamValEvaluationChallenges, RamValEvaluationInputClaims, RamValEvaluationOutputClaims,
    RegistersValEvaluationInputClaims, RegistersValEvaluationOutputClaims,
};
use crate::claims::val_evaluation::{RamValEvaluationSymbolic, RegistersValEvaluationSymbolic};
use crate::ids::{
    CommittedPolynomial, DerivedId, OpeningId, RelationId, ValEvaluationDerived, VirtualPolynomial,
};
use crate::points::{self, PointsError};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::JoltField;
use jolt_verifier::{stages::relations::ConcreteSumcheck, VerifierError};

fn term_error(error: PointsError) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: "Stage5".to_owned(),
        reason: error.to_string(),
    }
}

#[derive(Clone)]
pub struct RegistersValEvaluation<F: JoltField> {
    symbolic: RegistersValEvaluationSymbolic,
    a_reg: Vec<F>,
    r_bit: Vec<F>,
    r_4: Vec<F>,
}
impl<F: JoltField> RegistersValEvaluation<F> {
    pub fn new(a_reg: Vec<F>, r_bit: Vec<F>, r_4: Vec<F>) -> Result<Self, PointsError> {
        for (point, expected) in [(&a_reg, 5), (&r_bit, 6)] {
            if point.len() != expected {
                return Err(PointsError::Dimension {
                    expected,
                    actual: point.len(),
                });
            }
        }
        Ok(Self {
            symbolic: RegistersValEvaluationSymbolic::new(r_4.len()),
            a_reg,
            r_bit,
            r_4,
        })
    }
    pub fn a_reg(&self) -> &[F] {
        &self.a_reg
    }
    pub fn r_bit(&self) -> &[F] {
        &self.r_bit
    }
    pub fn r_4(&self) -> &[F] {
        &self.r_4
    }
    pub fn input_points(&self) -> RegistersValEvaluationInputClaims<Vec<F>> {
        RegistersValEvaluationInputClaims {
            registers_val: self
                .a_reg
                .iter()
                .chain(&self.r_bit)
                .chain(&self.r_4)
                .copied()
                .collect(),
        }
    }
}
impl<F: JoltField> ConcreteSumcheck<F> for RegistersValEvaluation<F> {
    type Symbolic = RegistersValEvaluationSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &RegistersValEvaluationInputClaims<Vec<F>>,
    ) -> Result<RegistersValEvaluationOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(term_error(PointsError::Dimension {
                expected: self.rounds(),
                actual: point.len(),
            }));
        }
        Ok(RegistersValEvaluationOutputClaims {
            rd_wa: self.a_reg.iter().chain(point).copied().collect(),
            store: point.to_vec(),
            inc: self.r_bit.iter().chain(point).copied().collect(),
        })
    }
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "foreign derived ids fail closed as missing claims"
    )]
    fn derive_output_term(
        &self,
        id: &DerivedId,
        _inputs: &RegistersValEvaluationInputClaims<Vec<F>>,
        outputs: &RegistersValEvaluationOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        match id {
            DerivedId::RegistersValEvaluation(ValEvaluationDerived::Lt) => {
                points::lt(&outputs.store, &self.r_4).map_err(term_error)
            }
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        }
    }
}

#[derive(Clone)]
pub struct RamValEvaluation<F: JoltField> {
    symbolic: RamValEvaluationSymbolic,
    a_ram: Vec<F>,
    r_bit: Vec<F>,
    r_4: Vec<F>,
    init_eval: F,
}
impl<F: JoltField> RamValEvaluation<F> {
    pub fn new(
        a_ram: Vec<F>,
        r_bit: Vec<F>,
        r_4: Vec<F>,
        init_eval: F,
    ) -> Result<Self, PointsError> {
        if a_ram.len() < 5 {
            return Err(PointsError::Dimension {
                expected: 5,
                actual: a_ram.len(),
            });
        }
        if r_bit.len() != 6 {
            return Err(PointsError::Dimension {
                expected: 6,
                actual: r_bit.len(),
            });
        }
        Ok(Self {
            symbolic: RamValEvaluationSymbolic::new(r_4.len()),
            a_ram,
            r_bit,
            r_4,
            init_eval,
        })
    }
    pub fn a_ram(&self) -> &[F] {
        &self.a_ram
    }
    pub fn r_bit(&self) -> &[F] {
        &self.r_bit
    }
    pub fn r_4(&self) -> &[F] {
        &self.r_4
    }
    pub fn input_points(&self) -> RamValEvaluationInputClaims<Vec<F>> {
        let final_point: Vec<_> = self.a_ram.iter().chain(&self.r_bit).copied().collect();
        RamValEvaluationInputClaims {
            ram_val: final_point.iter().chain(&self.r_4).copied().collect(),
            ram_val_final: final_point,
        }
    }
}
impl<F: JoltField> ConcreteSumcheck<F> for RamValEvaluation<F> {
    type Symbolic = RamValEvaluationSymbolic;
    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }
    fn derive_opening_points(
        &self,
        point: &[F],
        _inputs: &RamValEvaluationInputClaims<Vec<F>>,
    ) -> Result<RamValEvaluationOutputClaims<Vec<F>>, VerifierError> {
        if point.len() != self.rounds() {
            return Err(term_error(PointsError::Dimension {
                expected: self.rounds(),
                actual: point.len(),
            }));
        }
        Ok(RamValEvaluationOutputClaims {
            ram_ra: self.a_ram.iter().chain(point).copied().collect(),
            store: point.to_vec(),
            inc: self.r_bit.iter().chain(point).copied().collect(),
        })
    }
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "foreign derived ids fail closed as missing claims"
    )]
    fn derive_input_term(
        &self,
        id: &DerivedId,
        _challenges: &RamValEvaluationChallenges<F>,
    ) -> Result<F, VerifierError> {
        match id {
            DerivedId::RamValEvaluation(ValEvaluationDerived::InitEval) => Ok(self.init_eval),
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        }
    }
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "foreign derived ids fail closed as missing claims"
    )]
    fn derive_output_term(
        &self,
        id: &DerivedId,
        _inputs: &RamValEvaluationInputClaims<Vec<F>>,
        outputs: &RamValEvaluationOutputClaims<Vec<F>>,
        _challenges: &RamValEvaluationChallenges<F>,
    ) -> Result<F, VerifierError> {
        match id {
            DerivedId::RamValEvaluation(ValEvaluationDerived::Lt) => {
                points::lt(&outputs.store, &self.r_4).map_err(term_error)
            }
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        }
    }
    fn aliased_output_openings() -> Vec<(OpeningId, OpeningId)> {
        vec![
            (
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::Store,
                    RelationId::RamValEvaluation,
                ),
                OpeningId::virtual_polynomial(
                    VirtualPolynomial::Store,
                    RelationId::RegistersValEvaluation,
                ),
            ),
            (
                OpeningId::committed(CommittedPolynomial::Inc, RelationId::RamValEvaluation),
                OpeningId::committed(CommittedPolynomial::Inc, RelationId::RegistersValEvaluation),
            ),
        ]
    }
}
