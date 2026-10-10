//! Symbolic registers read checking over the binary field.

use crate::ids::{
    ChallengeId, DerivedId, FamilyExpr, OpeningId, ReadCheckingDerived, RegistersReadChallenge,
    RelationId, VirtualPolynomial,
};
use jolt_claims::{
    challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck,
};
use jolt_field::Ring;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct RegistersReadCheckingInputClaims<C> {
    #[opening(Rs1Value, from = RouterCycleVariant)]
    pub rs1_value: C,
    #[opening(Rs2Value, from = RouterCycleVariant)]
    pub rs2_value: C,
    #[opening(RdPreValue, from = RouterCycleVariant)]
    pub rd_pre_value: C,
}

#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(RegistersReadChecking)]
pub struct RegistersReadCheckingOutputClaims<C> {
    #[opening(Rs1Ra)]
    pub rs1_ra: C,
    #[opening(Rs2Ra)]
    pub rs2_ra: C,
    #[opening(RdWa)]
    pub rd_wa: C,
    #[opening(RegistersVal)]
    pub registers_val: C,
}

#[derive(Clone, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
pub struct RegistersReadCheckingChallenges<F> {
    #[challenge(RegistersReadChallenge::Rs1)]
    pub rs1: F,
    #[challenge(RegistersReadChallenge::Rs2)]
    pub rs2: F,
    #[challenge(RegistersReadChallenge::Rd)]
    pub rd: F,
}

#[derive(Clone)]
pub struct RegistersReadCheckingSymbolic {
    rounds: usize,
}

impl SymbolicSumcheck for RegistersReadCheckingSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = usize;
    type Challenges<F> = RegistersReadCheckingChallenges<F>;
    type Inputs<C> = RegistersReadCheckingInputClaims<C>;
    type Outputs<C> = RegistersReadCheckingOutputClaims<C>;
    fn new(t: usize) -> Self {
        Self { rounds: 5 + t }
    }
    fn id() -> RelationId {
        RelationId::RegistersReadChecking
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn degree(&self) -> usize {
        3
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        challenge(ChallengeId::RegistersReadChecking(
            RegistersReadChallenge::Rs1,
        )) * opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::Rs1Value,
            RelationId::RouterCycleVariant,
        )) + challenge(ChallengeId::RegistersReadChecking(
            RegistersReadChallenge::Rs2,
        )) * opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::Rs2Value,
            RelationId::RouterCycleVariant,
        )) + challenge(ChallengeId::RegistersReadChecking(
            RegistersReadChallenge::Rd,
        )) * opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::RdPreValue,
            RelationId::RouterCycleVariant,
        ))
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        derived(DerivedId::RegistersReadChecking(
            ReadCheckingDerived::EqCycle,
        )) * (challenge(ChallengeId::RegistersReadChecking(
            RegistersReadChallenge::Rs1,
        )) * opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::Rs1Ra,
            RelationId::RegistersReadChecking,
        )) + challenge(ChallengeId::RegistersReadChecking(
            RegistersReadChallenge::Rs2,
        )) * opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::Rs2Ra,
            RelationId::RegistersReadChecking,
        )) + challenge(ChallengeId::RegistersReadChecking(
            RegistersReadChallenge::Rd,
        )) * opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::RdWa,
            RelationId::RegistersReadChecking,
        ))) * opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::RegistersVal,
            RelationId::RegistersReadChecking,
        ))
    }
}
