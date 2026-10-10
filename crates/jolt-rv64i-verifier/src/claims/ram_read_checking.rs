//! Symbolic ram read checking over the binary field.

use crate::ids::{
    ChallengeId, DerivedId, FamilyExpr, OpeningId, ReadCheckingDerived, RelationId,
    VirtualPolynomial,
};
use jolt_claims::{derived, opening, InputClaims, NoChallenges, OutputClaims, SymbolicSumcheck};
use jolt_field::Ring;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct RamReadCheckingInputClaims<C> {
    #[opening(RamReadValue, from = RouterCycleMemory)]
    pub ram_read_value: C,
}

#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(RamReadChecking)]
pub struct RamReadCheckingOutputClaims<C> {
    #[opening(RamRa)]
    pub ram_ra: C,
    #[opening(RamVal)]
    pub ram_val: C,
}

pub type RamReadCheckingChallenges<F> = NoChallenges<F>;

#[derive(Clone)]
pub struct RamReadCheckingSymbolic {
    rounds: usize,
}

impl SymbolicSumcheck for RamReadCheckingSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = (usize, usize);
    type Challenges<F> = RamReadCheckingChallenges<F>;
    type Inputs<C> = RamReadCheckingInputClaims<C>;
    type Outputs<C> = RamReadCheckingOutputClaims<C>;
    fn new((a, t): Self::Shape) -> Self {
        Self { rounds: a + t }
    }
    fn id() -> RelationId {
        RelationId::RamReadChecking
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn degree(&self) -> usize {
        3
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::RamReadValue,
            RelationId::RouterCycleMemory,
        ))
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        derived(DerivedId::RamReadChecking(ReadCheckingDerived::EqCycle))
            * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::RamRa,
                RelationId::RamReadChecking,
            ))
            * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::RamVal,
                RelationId::RamReadChecking,
            ))
    }
}
