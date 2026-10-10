//! Symbolic zero-checks for the binary and extension-field row blocks.

use crate::ids::{
    ChallengeId, DerivedId, FamilyExpr, OpeningId, OuterDerived, RelationId, RowBlock,
    VirtualPolynomial,
};
use jolt_claims::{
    derived, opening, Expr, InputClaims, NoChallenges, OutputClaims, SymbolicSumcheck,
};
use jolt_field::{JoltField, Ring};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Empty consumed cells for the outer zero-checks. The claim derive requires
/// an opening field, so this marker implements the empty resolver directly.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpartanOuterInputClaims<C>(PhantomData<C>);
impl<F: JoltField> InputClaims<F, OpeningId> for SpartanOuterInputClaims<F> {
    fn canonical_order(&self) -> Vec<OpeningId> {
        Vec::new()
    }
    fn resolve_input(&self, _id: &OpeningId) -> Option<F> {
        None
    }
}

macro_rules! outer {
    ($symbolic:ident, $outputs:ident, $relation:ident, $block:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
        #[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
        #[protocol(ids = crate::ids)]
        #[relation($relation)]
        pub struct $outputs<C> {
            #[opening(Az)]
            pub az: C,
            #[opening(Bz)]
            pub bz: C,
            #[opening(Cz)]
            pub cz: C,
        }

        /// The summand is `eq(tau,(i,j)) (Az[i,j] Bz[i,j] + Cz[i,j])`.
        #[derive(Clone)]
        pub struct $symbolic {
            rounds: usize,
        }
        impl SymbolicSumcheck for $symbolic {
            type RelationId = RelationId;
            type OpeningId = OpeningId;
            type DerivedId = DerivedId;
            type ChallengeId = ChallengeId;
            type Shape = usize;
            type Challenges<F> = NoChallenges<F>;
            type Inputs<C> = SpartanOuterInputClaims<C>;
            type Outputs<C> = $outputs<C>;
            fn new(rounds: usize) -> Self {
                Self { rounds }
            }
            fn id() -> RelationId {
                RelationId::$relation
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
                let weight = derived(DerivedId::SpartanOuter(
                    RowBlock::$block,
                    OuterDerived::EqTau,
                ));
                weight.clone()
                    * opening(OpeningId::virtual_polynomial(
                        VirtualPolynomial::Az,
                        RelationId::$relation,
                    ))
                    * opening(OpeningId::virtual_polynomial(
                        VirtualPolynomial::Bz,
                        RelationId::$relation,
                    ))
                    + weight
                        * opening(OpeningId::virtual_polynomial(
                            VirtualPolynomial::Cz,
                            RelationId::$relation,
                        ))
            }
        }
    };
}
outer!(
    SpartanOuterF2Symbolic,
    SpartanOuterF2OutputClaims,
    SpartanOuterF2,
    F2
);
outer!(
    SpartanOuterF128Symbolic,
    SpartanOuterF128OutputClaims,
    SpartanOuterF128,
    F128
);
