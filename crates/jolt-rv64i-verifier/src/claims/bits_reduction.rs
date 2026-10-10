//! Symbolic reduction of six committed functionals to the 256 cycle-column values.

use jolt_claims::{
    challenge, derived, opening, Expr, InputClaims, OutputClaims, SumcheckChallenges,
    SymbolicSumcheck,
};
use jolt_field::Ring;
use jolt_rv64i_arith::BITS_COLUMNS;
use serde::{Deserialize, Serialize};

use crate::ids::{
    BitsReductionChallenge, BitsReductionDerived, ChallengeId, CommittedPolynomial, DerivedId,
    FamilyExpr, OpeningId, RelationId,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct BitsReductionInputClaims<C> {
    #[opening(committed = DirectColumns, from = SpartanInner)]
    pub direct_columns: C,
    #[opening(committed = VariantBits, from = RouterCycleVariant)]
    pub variant_bits: C,
    #[opening(committed = PosRa0, from = RouterCycleShift)]
    pub pos_ra_0: C,
    #[opening(committed = PosRa1, from = RouterCycleShift)]
    pub pos_ra_1: C,
    #[opening(committed = ShouldBranch, from = RouterCycleBranch)]
    pub should_branch: C,
    #[opening(committed = Inc, from = RegistersValEvaluation)]
    pub inc: C,
}

#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(BitsReduction)]
pub struct BitsReductionOutputClaims<C> {
    #[opening(committed = Column)]
    pub columns: Vec<C>,
}

#[derive(Clone, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
pub struct BitsReductionChallenges<F> {
    #[challenge(BitsReductionChallenge::DirectColumns)]
    pub direct_columns: F,
    #[challenge(BitsReductionChallenge::VariantBits)]
    pub variant_bits: F,
    #[challenge(BitsReductionChallenge::PosRa0)]
    pub pos_ra_0: F,
    #[challenge(BitsReductionChallenge::PosRa1)]
    pub pos_ra_1: F,
    #[challenge(BitsReductionChallenge::ShouldBranch)]
    pub should_branch: F,
    #[challenge(BitsReductionChallenge::Inc)]
    pub inc: F,
}

/// The summand is `Σ_y (Σ_u c_u l_u(y) eq(t_u,j)) Bits[y,j]`.
#[derive(Clone)]
pub struct BitsReductionSymbolic {
    rounds: usize,
}

impl SymbolicSumcheck for BitsReductionSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = usize;
    type Challenges<F> = BitsReductionChallenges<F>;
    type Inputs<C> = BitsReductionInputClaims<C>;
    type Outputs<C> = BitsReductionOutputClaims<C>;

    fn new(rounds: usize) -> Self {
        Self { rounds }
    }
    fn id() -> RelationId {
        RelationId::BitsReduction
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn degree(&self) -> usize {
        2
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        let cells = [
            (
                BitsReductionChallenge::DirectColumns,
                CommittedPolynomial::DirectColumns,
                RelationId::SpartanInner,
            ),
            (
                BitsReductionChallenge::VariantBits,
                CommittedPolynomial::VariantBits,
                RelationId::RouterCycleVariant,
            ),
            (
                BitsReductionChallenge::PosRa0,
                CommittedPolynomial::PosRa0,
                RelationId::RouterCycleShift,
            ),
            (
                BitsReductionChallenge::PosRa1,
                CommittedPolynomial::PosRa1,
                RelationId::RouterCycleShift,
            ),
            (
                BitsReductionChallenge::ShouldBranch,
                CommittedPolynomial::ShouldBranch,
                RelationId::RouterCycleBranch,
            ),
            (
                BitsReductionChallenge::Inc,
                CommittedPolynomial::Inc,
                RelationId::RegistersValEvaluation,
            ),
        ];
        let linear: FamilyExpr<F> = cells
            .into_iter()
            .map(|(coefficient, polynomial, source)| {
                challenge(ChallengeId::BitsReduction(coefficient))
                    * opening(OpeningId::committed(polynomial, source))
            })
            .fold(Expr::zero(), |sum, term| sum + term);
        linear
            + challenge(ChallengeId::BitsReduction(BitsReductionChallenge::PosRa0))
                * derived(DerivedId::BitsReduction(BitsReductionDerived::PosZero(0)))
            + challenge(ChallengeId::BitsReduction(BitsReductionChallenge::PosRa1))
                * derived(DerivedId::BitsReduction(BitsReductionDerived::PosZero(1)))
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        (0..BITS_COLUMNS)
            .map(|column| {
                derived(DerivedId::BitsReduction(
                    BitsReductionDerived::ColumnWeight(column),
                )) * opening(OpeningId::committed(
                    CommittedPolynomial::Column(column),
                    RelationId::BitsReduction,
                ))
            })
            .fold(Expr::zero(), |sum, term| sum + term)
    }
}
