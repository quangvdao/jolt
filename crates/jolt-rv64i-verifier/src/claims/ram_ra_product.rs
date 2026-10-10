//! Symbolic reduction of two RAM selectors to their committed digit chunks.
use crate::ids::{
    ChallengeId, DerivedId, FamilyExpr, OpeningId, RamRaChallenge, RamRaProductDerived, RelationId,
    VirtualPolynomial,
};
use jolt_claims::{
    challenge, derived, opening, Expr, InputClaims, OutputClaims, SumcheckChallenges,
    SymbolicSumcheck,
};
use jolt_field::Ring;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct RamRaProductInputClaims<C> {
    #[opening(RamRa, from = RamReadChecking)]
    pub ram_ra_read: C,
    #[opening(RamRa, from = RamValEvaluation)]
    pub ram_ra_val: C,
}
#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(RamRaProduct)]
pub struct RamRaProductOutputClaims<C> {
    #[opening(RamRaChunk)]
    pub chunks: Vec<C>,
}
#[derive(Clone, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
pub struct RamRaProductChallenges<F> {
    #[challenge(RamRaChallenge::Read)]
    pub read: F,
    #[challenge(RamRaChallenge::Val)]
    pub val: F,
}
/// The summand is `(c.Read eq(r_4,j) + c.Val eq(r_5,j)) Π_c RamRa_c(a_c,j)`.
#[derive(Clone)]
pub struct RamRaProductSymbolic {
    rounds: usize,
    chunks: usize,
}
impl SymbolicSumcheck for RamRaProductSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = (usize, usize);
    type Challenges<F> = RamRaProductChallenges<F>;
    type Inputs<C> = RamRaProductInputClaims<C>;
    type Outputs<C> = RamRaProductOutputClaims<C>;
    fn new((rounds, chunks): Self::Shape) -> Self {
        Self { rounds, chunks }
    }
    fn id() -> RelationId {
        RelationId::RamRaProduct
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn degree(&self) -> usize {
        self.chunks + 1
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        challenge(ChallengeId::RamRaProduct(RamRaChallenge::Read))
            * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::RamRa,
                RelationId::RamReadChecking,
            ))
            + challenge(ChallengeId::RamRaProduct(RamRaChallenge::Val))
                * opening(OpeningId::virtual_polynomial(
                    VirtualPolynomial::RamRa,
                    RelationId::RamValEvaluation,
                ))
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        let product = (0..self.chunks)
            .map(|c| {
                opening(OpeningId::virtual_polynomial(
                    VirtualPolynomial::RamRaChunk(c),
                    RelationId::RamRaProduct,
                ))
            })
            .fold(Expr::one(), |p, v| p * v);
        challenge(ChallengeId::RamRaProduct(RamRaChallenge::Read))
            * derived(DerivedId::RamRaProduct(RamRaProductDerived::EqRead))
            * product.clone()
            + challenge(ChallengeId::RamRaProduct(RamRaChallenge::Val))
                * derived(DerivedId::RamRaProduct(RamRaProductDerived::EqVal))
                * product
    }
}
