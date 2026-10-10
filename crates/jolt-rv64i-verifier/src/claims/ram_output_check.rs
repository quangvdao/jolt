//! Symbolic ram output check over the binary field.

use crate::ids::{
    ChallengeId, DerivedId, FamilyExpr, OpeningId, OutputCheckDerived, RelationId,
    VirtualPolynomial,
};
use jolt_claims::{
    derived, opening, Expr, InputClaims, NoChallenges, OutputClaims, SymbolicSumcheck,
};
use jolt_field::{JoltField, Ring};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// No consumed cells: the output-check input claim is zero.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RamOutputCheckInputClaims<C> {
    cell: PhantomData<C>,
}

impl<F: JoltField> InputClaims<F, OpeningId> for RamOutputCheckInputClaims<F> {
    fn canonical_order(&self) -> Vec<OpeningId> {
        Vec::new()
    }
    fn resolve_input(&self, _id: &OpeningId) -> Option<F> {
        None
    }
}

#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(RamOutputCheck)]
pub struct RamOutputCheckOutputClaims<C> {
    #[opening(RamValFinal)]
    pub ram_val_final: C,
}

pub type RamOutputCheckChallenges<F> = NoChallenges<F>;

#[derive(Clone)]
pub struct RamOutputCheckSymbolic {
    rounds: usize,
}

impl SymbolicSumcheck for RamOutputCheckSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = usize;
    type Challenges<F> = RamOutputCheckChallenges<F>;
    type Inputs<C> = RamOutputCheckInputClaims<C>;
    type Outputs<C> = RamOutputCheckOutputClaims<C>;
    fn new(a: usize) -> Self {
        Self { rounds: a }
    }
    fn id() -> RelationId {
        RelationId::RamOutputCheck
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn degree(&self) -> usize {
        3
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        Expr::zero()
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        derived(DerivedId::RamOutputCheck(OutputCheckDerived::EqTau))
            * derived(DerivedId::RamOutputCheck(OutputCheckDerived::IoMask))
            * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::RamValFinal,
                RelationId::RamOutputCheck,
            ))
            + derived(DerivedId::RamOutputCheck(OutputCheckDerived::EqTau))
                * derived(DerivedId::RamOutputCheck(OutputCheckDerived::IoMask))
                * derived(DerivedId::RamOutputCheck(OutputCheckDerived::ValIo))
    }
}
