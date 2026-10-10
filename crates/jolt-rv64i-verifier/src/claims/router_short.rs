//! Symbolic reduction of routed witness columns to five router folds.
use crate::ids::{
    ChallengeId, DerivedId, FamilyExpr, OpeningId, RelationId, Router, RouterShortDerived,
    VirtualPolynomial,
};
use crate::public::routes::ROUTERS;
use jolt_claims::{
    derived, opening, Expr, InputClaims, NoChallenges, OutputClaims, SymbolicSumcheck,
};
use jolt_field::Ring;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct RouterShortInputClaims<C> {
    #[opening(WitnessRouted, from = SpartanInner)]
    pub witness_routed: C,
}
#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(RouterShort)]
pub struct RouterShortOutputClaims<C> {
    #[opening(RouterFold(Router::Variant))]
    pub variant: C,
    #[opening(RouterFold(Router::Shift))]
    pub shift: C,
    #[opening(RouterFold(Router::Memory))]
    pub memory: C,
    #[opening(RouterFold(Router::Compare))]
    pub compare: C,
    #[opening(RouterFold(Router::Branch))]
    pub branch: C,
}
/// The 17-slot summand is the sum of five idle-weighted tensor contractions.
#[derive(Clone)]
pub struct RouterShortSymbolic;
impl SymbolicSumcheck for RouterShortSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = ();
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = RouterShortInputClaims<C>;
    type Outputs<C> = RouterShortOutputClaims<C>;
    fn new((): ()) -> Self {
        Self
    }
    fn id() -> RelationId {
        RelationId::RouterShort
    }
    fn rounds(&self) -> usize {
        17
    }
    fn degree(&self) -> usize {
        2
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::WitnessRouted,
            RelationId::SpartanInner,
        ))
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        ROUTERS
            .into_iter()
            .map(|r| {
                derived(DerivedId::RouterShort(RouterShortDerived::RouteWeight(r)))
                    * opening(OpeningId::virtual_polynomial(
                        VirtualPolynomial::RouterFold(r),
                        RelationId::RouterShort,
                    ))
            })
            .fold(Expr::zero(), |sum, term| sum + term)
    }
}
