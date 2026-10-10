//! Symbolic column reduction of the six row-block evaluations.

use crate::ids::{
    ChallengeId, CommittedPolynomial, DerivedId, FamilyExpr, InnerChallenge, InnerDerived,
    OpeningId, RelationId, VirtualPolynomial,
};
use jolt_claims::{
    challenge, derived, opening, Expr, InputClaims, OutputClaims, SumcheckChallenges,
    SymbolicSumcheck,
};
use jolt_field::Ring;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct SpartanInnerInputClaims<C> {
    #[opening(Az, from = SpartanOuterF2)]
    pub az_f2: C,
    #[opening(Bz, from = SpartanOuterF2)]
    pub bz_f2: C,
    #[opening(Cz, from = SpartanOuterF2)]
    pub cz_f2: C,
    #[opening(Az, from = SpartanOuterF128)]
    pub az_f128: C,
    #[opening(Bz, from = SpartanOuterF128)]
    pub bz_f128: C,
    #[opening(Cz, from = SpartanOuterF128)]
    pub cz_f128: C,
}
#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(SpartanInner)]
pub struct SpartanInnerOutputClaims<C> {
    #[opening(WitnessRouted)]
    pub witness_routed: C,
    #[opening(committed = DirectColumns)]
    pub direct_columns: C,
}
#[derive(Clone, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
pub struct SpartanInnerChallenges<F> {
    #[challenge(InnerChallenge::AzF2)]
    pub az_f2: F,
    #[challenge(InnerChallenge::BzF2)]
    pub bz_f2: F,
    #[challenge(InnerChallenge::CzF2)]
    pub cz_f2: F,
    #[challenge(InnerChallenge::AzF128)]
    pub az_f128: F,
    #[challenge(InnerChallenge::BzF128)]
    pub bz_f128: F,
    #[challenge(InnerChallenge::CzF128)]
    pub cz_f128: F,
}
/// The summand is `M[col] Z'(col,r_1)`, with the public, routed and direct
/// parts of `Z'` occupying the column sets of protocol §8.3.
#[derive(Clone)]
pub struct SpartanInnerSymbolic;
impl SymbolicSumcheck for SpartanInnerSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = ();
    type Challenges<F> = SpartanInnerChallenges<F>;
    type Inputs<C> = SpartanInnerInputClaims<C>;
    type Outputs<C> = SpartanInnerOutputClaims<C>;
    fn new((): ()) -> Self {
        Self
    }
    fn id() -> RelationId {
        RelationId::SpartanInner
    }
    fn rounds(&self) -> usize {
        10
    }
    fn degree(&self) -> usize {
        2
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        [
            (
                InnerChallenge::AzF2,
                VirtualPolynomial::Az,
                RelationId::SpartanOuterF2,
            ),
            (
                InnerChallenge::BzF2,
                VirtualPolynomial::Bz,
                RelationId::SpartanOuterF2,
            ),
            (
                InnerChallenge::CzF2,
                VirtualPolynomial::Cz,
                RelationId::SpartanOuterF2,
            ),
            (
                InnerChallenge::AzF128,
                VirtualPolynomial::Az,
                RelationId::SpartanOuterF128,
            ),
            (
                InnerChallenge::BzF128,
                VirtualPolynomial::Bz,
                RelationId::SpartanOuterF128,
            ),
            (
                InnerChallenge::CzF128,
                VirtualPolynomial::Cz,
                RelationId::SpartanOuterF128,
            ),
        ]
        .into_iter()
        .map(|(c, p, r)| {
            challenge(ChallengeId::SpartanInner(c)) * opening(OpeningId::virtual_polynomial(p, r))
        })
        .fold(Expr::zero(), |sum, term| sum + term)
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        let weight = derived(DerivedId::SpartanInner(InnerDerived::MatrixWeight));
        weight.clone()
            * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::WitnessRouted,
                RelationId::SpartanInner,
            ))
            + weight.clone()
                * opening(OpeningId::committed(
                    CommittedPolynomial::DirectColumns,
                    RelationId::SpartanInner,
                ))
            + weight * derived(DerivedId::SpartanInner(InnerDerived::PublicColumns))
    }
}
