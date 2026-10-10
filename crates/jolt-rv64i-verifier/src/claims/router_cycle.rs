//! Symbolic cycle reductions of the five binary-field RV64I routers.

use jolt_claims::{
    derived, opening, Expr, InputClaims, NoChallenges, OutputClaims, SymbolicSumcheck,
};
use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::ids::{
    ChallengeId, CommittedPolynomial, DerivedId, FamilyExpr, OpeningId, RelationId, Router,
    RouterCycleDerived, VirtualPolynomial,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct RouterCycleVariantInputClaims<C> {
    #[opening(RouterFold(Router::Variant), from = RouterShort)]
    pub fold: C,
}

#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(RouterCycleVariant)]
pub struct RouterCycleVariantOutputClaims<C> {
    #[opening(Rs1Value)]
    pub rs1_value: C,
    #[opening(Rs2Value)]
    pub rs2_value: C,
    #[opening(RdPreValue)]
    pub rd_pre_value: C,
    #[opening(Imm)]
    pub imm: C,
    #[opening(FallThroughPC)]
    pub fall_through_pc: C,
    #[opening(PCPlusImm)]
    pub pc_plus_imm: C,
    #[opening(PC)]
    pub pc: C,
    #[opening(NextPC)]
    pub next_pc: C,
    #[opening(committed = VariantBits)]
    pub variant_bits: C,
    #[opening(Variant)]
    pub variant: C,
}

#[derive(Clone)]
pub struct RouterCycleVariantSymbolic {
    rounds: usize,
}

impl SymbolicSumcheck for RouterCycleVariantSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = usize;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = RouterCycleVariantInputClaims<C>;
    type Outputs<C> = RouterCycleVariantOutputClaims<C>;

    fn new(rounds: usize) -> Self {
        Self { rounds }
    }
    fn id() -> RelationId {
        RelationId::RouterCycleVariant
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn degree(&self) -> usize {
        3
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::RouterFold(Router::Variant),
            RelationId::RouterShort,
        ))
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        let eq_cycle = derived(DerivedId::RouterCycle(
            Router::Variant,
            RouterCycleDerived::EqCycle,
        ));
        let words = [
            VirtualPolynomial::Rs1Value,
            VirtualPolynomial::Rs2Value,
            VirtualPolynomial::RdPreValue,
            VirtualPolynomial::Imm,
            VirtualPolynomial::FallThroughPC,
            VirtualPolynomial::PCPlusImm,
            VirtualPolynomial::PC,
            VirtualPolynomial::NextPC,
        ];
        let source = words
            .into_iter()
            .enumerate()
            .map(|(n, polynomial)| {
                derived(DerivedId::RouterCycle(
                    Router::Variant,
                    RouterCycleDerived::WordSlot(n),
                )) * opening(OpeningId::virtual_polynomial(
                    polynomial,
                    RelationId::RouterCycleVariant,
                ))
            })
            .fold(Expr::zero(), |sum, term| sum + term);
        eq_cycle
            * (source
                + opening(OpeningId::committed(
                    CommittedPolynomial::VariantBits,
                    RelationId::RouterCycleVariant,
                ))
                + derived(DerivedId::RouterCycle(
                    Router::Variant,
                    RouterCycleDerived::OneSlot,
                )))
            * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::Variant,
                RelationId::RouterCycleVariant,
            ))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct RouterCycleShiftInputClaims<C> {
    #[opening(RouterFold(Router::Shift), from = RouterShort)]
    pub fold: C,
}

#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(RouterCycleShift)]
pub struct RouterCycleShiftOutputClaims<C> {
    #[opening(Rs1Value)]
    pub rs1_value: C,
    #[opening(ShiftKind)]
    pub shift_kind: C,
    #[opening(committed = PosRa0)]
    pub pos_ra_0: C,
    #[opening(committed = PosRa1)]
    pub pos_ra_1: C,
}

#[derive(Clone)]
pub struct RouterCycleShiftSymbolic {
    rounds: usize,
}

impl SymbolicSumcheck for RouterCycleShiftSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = usize;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = RouterCycleShiftInputClaims<C>;
    type Outputs<C> = RouterCycleShiftOutputClaims<C>;

    fn new(rounds: usize) -> Self {
        Self { rounds }
    }
    fn id() -> RelationId {
        RelationId::RouterCycleShift
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn degree(&self) -> usize {
        5
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::RouterFold(Router::Shift),
            RelationId::RouterShort,
        ))
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        let eq_cycle = derived(DerivedId::RouterCycle(
            Router::Shift,
            RouterCycleDerived::EqCycle,
        ));
        eq_cycle
            * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::Rs1Value,
                RelationId::RouterCycleShift,
            ))
            * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::ShiftKind,
                RelationId::RouterCycleShift,
            ))
            * opening(OpeningId::committed(
                CommittedPolynomial::PosRa0,
                RelationId::RouterCycleShift,
            ))
            * opening(OpeningId::committed(
                CommittedPolynomial::PosRa1,
                RelationId::RouterCycleShift,
            ))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct RouterCycleMemoryInputClaims<C> {
    #[opening(RouterFold(Router::Memory), from = RouterShort)]
    pub fold: C,
}

#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(RouterCycleMemory)]
pub struct RouterCycleMemoryOutputClaims<C> {
    #[opening(RamReadValue)]
    pub ram_read_value: C,
    #[opening(Rs2Value)]
    pub rs2_value: C,
    #[opening(AccessKind)]
    pub access_kind: C,
    #[opening(committed = PosRa0)]
    pub pos_ra_0: C,
}

#[derive(Clone)]
pub struct RouterCycleMemorySymbolic {
    rounds: usize,
}

impl SymbolicSumcheck for RouterCycleMemorySymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = usize;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = RouterCycleMemoryInputClaims<C>;
    type Outputs<C> = RouterCycleMemoryOutputClaims<C>;

    fn new(rounds: usize) -> Self {
        Self { rounds }
    }
    fn id() -> RelationId {
        RelationId::RouterCycleMemory
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn degree(&self) -> usize {
        4
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::RouterFold(Router::Memory),
            RelationId::RouterShort,
        ))
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        let eq_cycle = derived(DerivedId::RouterCycle(
            Router::Memory,
            RouterCycleDerived::EqCycle,
        ));
        eq_cycle
            * (derived(DerivedId::RouterCycle(
                Router::Memory,
                RouterCycleDerived::WordSlot(0),
            )) * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::RamReadValue,
                RelationId::RouterCycleMemory,
            )) + derived(DerivedId::RouterCycle(
                Router::Memory,
                RouterCycleDerived::WordSlot(1),
            )) * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::Rs2Value,
                RelationId::RouterCycleMemory,
            )))
            * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::AccessKind,
                RelationId::RouterCycleMemory,
            ))
            * opening(OpeningId::committed(
                CommittedPolynomial::PosRa0,
                RelationId::RouterCycleMemory,
            ))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct RouterCycleCompareInputClaims<C> {
    #[opening(RouterFold(Router::Compare), from = RouterShort)]
    pub fold: C,
}

#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(RouterCycleCompare)]
pub struct RouterCycleCompareOutputClaims<C> {
    #[opening(Rs1Value)]
    pub rs1_value: C,
    #[opening(Rs2Value)]
    pub rs2_value: C,
    #[opening(Imm)]
    pub imm: C,
    #[opening(KeyKind)]
    pub key_kind: C,
    #[opening(committed = PosRa0)]
    pub pos_ra_0: C,
    #[opening(committed = PosRa1)]
    pub pos_ra_1: C,
}

#[derive(Clone)]
pub struct RouterCycleCompareSymbolic {
    rounds: usize,
}

impl SymbolicSumcheck for RouterCycleCompareSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = usize;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = RouterCycleCompareInputClaims<C>;
    type Outputs<C> = RouterCycleCompareOutputClaims<C>;

    fn new(rounds: usize) -> Self {
        Self { rounds }
    }
    fn id() -> RelationId {
        RelationId::RouterCycleCompare
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn degree(&self) -> usize {
        5
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::RouterFold(Router::Compare),
            RelationId::RouterShort,
        ))
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        let eq_cycle = derived(DerivedId::RouterCycle(
            Router::Compare,
            RouterCycleDerived::EqCycle,
        ));
        eq_cycle
            * (derived(DerivedId::RouterCycle(
                Router::Compare,
                RouterCycleDerived::WordSlot(0),
            )) * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::Rs1Value,
                RelationId::RouterCycleCompare,
            )) + derived(DerivedId::RouterCycle(
                Router::Compare,
                RouterCycleDerived::WordSlot(1),
            )) * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::Rs2Value,
                RelationId::RouterCycleCompare,
            )) + derived(DerivedId::RouterCycle(
                Router::Compare,
                RouterCycleDerived::WordSlot(2),
            )) * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::Imm,
                RelationId::RouterCycleCompare,
            )) + derived(DerivedId::RouterCycle(
                Router::Compare,
                RouterCycleDerived::OneSlot,
            )))
            * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::KeyKind,
                RelationId::RouterCycleCompare,
            ))
            * opening(OpeningId::committed(
                CommittedPolynomial::PosRa0,
                RelationId::RouterCycleCompare,
            ))
            * opening(OpeningId::committed(
                CommittedPolynomial::PosRa1,
                RelationId::RouterCycleCompare,
            ))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct RouterCycleBranchInputClaims<C> {
    #[opening(RouterFold(Router::Branch), from = RouterShort)]
    pub fold: C,
}

#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(RouterCycleBranch)]
pub struct RouterCycleBranchOutputClaims<C> {
    #[opening(FallThroughPC)]
    pub fall_through_pc: C,
    #[opening(PCPlusImm)]
    pub pc_plus_imm: C,
    #[opening(Branch)]
    pub branch: C,
    #[opening(committed = ShouldBranch)]
    pub should_branch: C,
}

#[derive(Clone)]
pub struct RouterCycleBranchSymbolic {
    rounds: usize,
}

impl SymbolicSumcheck for RouterCycleBranchSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = usize;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = RouterCycleBranchInputClaims<C>;
    type Outputs<C> = RouterCycleBranchOutputClaims<C>;

    fn new(rounds: usize) -> Self {
        Self { rounds }
    }
    fn id() -> RelationId {
        RelationId::RouterCycleBranch
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn degree(&self) -> usize {
        4
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::RouterFold(Router::Branch),
            RelationId::RouterShort,
        ))
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        let eq_cycle = derived(DerivedId::RouterCycle(
            Router::Branch,
            RouterCycleDerived::EqCycle,
        ));
        eq_cycle
            * (derived(DerivedId::RouterCycle(
                Router::Branch,
                RouterCycleDerived::WordSlot(0),
            )) * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::FallThroughPC,
                RelationId::RouterCycleBranch,
            )) + derived(DerivedId::RouterCycle(
                Router::Branch,
                RouterCycleDerived::WordSlot(1),
            )) * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::PCPlusImm,
                RelationId::RouterCycleBranch,
            )))
            * opening(OpeningId::virtual_polynomial(
                VirtualPolynomial::Branch,
                RelationId::RouterCycleBranch,
            ))
            * opening(OpeningId::committed(
                CommittedPolynomial::ShouldBranch,
                RelationId::RouterCycleBranch,
            ))
    }
}
