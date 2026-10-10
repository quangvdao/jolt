//! Value evaluations reduce replayed register and RAM state to cycle updates.

use crate::ids::{
    ChallengeId, CommittedPolynomial, DerivedId, FamilyExpr, OpeningId, RamValChallenge,
    RelationId, ValEvaluationDerived, VirtualPolynomial,
};
use jolt_claims::{
    challenge, derived, opening, InputClaims, NoChallenges, OutputClaims, SumcheckChallenges,
    SymbolicSumcheck,
};
use jolt_field::Ring;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct RegistersValEvaluationInputClaims<C> {
    #[opening(RegistersVal, from = RegistersReadChecking)]
    pub registers_val: C,
}
#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(RegistersValEvaluation)]
pub struct RegistersValEvaluationOutputClaims<C> {
    #[opening(RdWa)]
    pub rd_wa: C,
    #[opening(Store)]
    pub store: C,
    #[opening(committed = Inc)]
    pub inc: C,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct RamValEvaluationInputClaims<C> {
    #[opening(RamVal, from = RamReadChecking)]
    pub ram_val: C,
    #[opening(RamValFinal, from = RamOutputCheck)]
    pub ram_val_final: C,
}
#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(RamValEvaluation)]
pub struct RamValEvaluationOutputClaims<C> {
    #[opening(RamRa)]
    pub ram_ra: C,
    #[opening(Store)]
    pub store: C,
    #[opening(committed = Inc)]
    pub inc: C,
}
#[derive(Clone, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
pub struct RamValEvaluationChallenges<F> {
    #[challenge(RamValChallenge::Val)]
    pub val: F,
    #[challenge(RamValChallenge::Final)]
    pub final_value: F,
}

/// The cycle summand is `lt(j,r_4) RdWa(a_reg,j) (1 + Store[j]) Inc(r_bit,j)`.
#[derive(Clone)]
pub struct RegistersValEvaluationSymbolic {
    rounds: usize,
}
impl SymbolicSumcheck for RegistersValEvaluationSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = usize;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = RegistersValEvaluationInputClaims<C>;
    type Outputs<C> = RegistersValEvaluationOutputClaims<C>;
    fn new(rounds: usize) -> Self {
        Self { rounds }
    }
    fn id() -> RelationId {
        RelationId::RegistersValEvaluation
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn degree(&self) -> usize {
        4
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::RegistersVal,
            RelationId::RegistersReadChecking,
        ))
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        let lt = derived(DerivedId::RegistersValEvaluation(ValEvaluationDerived::Lt));
        let rd = opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::RdWa,
            Self::id(),
        ));
        let store = opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::Store,
            Self::id(),
        ));
        let inc = opening(OpeningId::committed(CommittedPolynomial::Inc, Self::id()));
        lt.clone() * rd.clone() * inc.clone() + lt * rd * store * inc
    }
}
/// The cycle summand is `(Val lt(j,r_4) + Final) RamRa(a_ram,j) Store[j] Inc(r_bit,j)`.
#[derive(Clone)]
pub struct RamValEvaluationSymbolic {
    rounds: usize,
}
impl SymbolicSumcheck for RamValEvaluationSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = usize;
    type Challenges<F> = RamValEvaluationChallenges<F>;
    type Inputs<C> = RamValEvaluationInputClaims<C>;
    type Outputs<C> = RamValEvaluationOutputClaims<C>;
    fn new(rounds: usize) -> Self {
        Self { rounds }
    }
    fn id() -> RelationId {
        RelationId::RamValEvaluation
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn degree(&self) -> usize {
        4
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        let val = challenge(ChallengeId::RamValEvaluation(RamValChallenge::Val));
        let final_value = challenge(ChallengeId::RamValEvaluation(RamValChallenge::Final));
        let init = derived(DerivedId::RamValEvaluation(ValEvaluationDerived::InitEval));
        val.clone()
            * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::RamVal,
                RelationId::RamReadChecking,
            ))
            + val * init.clone()
            + final_value.clone()
                * opening(OpeningId::virtual_polynomial(
                    VirtualPolynomial::RamValFinal,
                    RelationId::RamOutputCheck,
                ))
            + final_value * init
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        let val = challenge(ChallengeId::RamValEvaluation(RamValChallenge::Val));
        let final_value = challenge(ChallengeId::RamValEvaluation(RamValChallenge::Final));
        let lt = derived(DerivedId::RamValEvaluation(ValEvaluationDerived::Lt));
        let ra = opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::RamRa,
            Self::id(),
        ));
        let store = opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::Store,
            Self::id(),
        ));
        let inc = opening(OpeningId::committed(CommittedPolynomial::Inc, Self::id()));
        val * lt * ra.clone() * store.clone() * inc.clone() + final_value * ra * store * inc
    }
}
