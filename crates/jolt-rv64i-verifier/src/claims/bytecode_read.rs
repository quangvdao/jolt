//! Symbolic address and cycle phases of public bytecode read checking.

use crate::ids::{
    BytecodeAddressDerived, BytecodeChallenge, BytecodeCycleDerived, ChallengeId, CycleWeight,
    DerivedId, FamilyExpr, OpeningId, RelationId, VirtualPolynomial,
};
use jolt_claims::{
    challenge, derived, opening, Expr, InputClaims, NoChallenges, OutputClaims, SumcheckChallenges,
    SymbolicSumcheck,
};
use jolt_field::Ring;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct BytecodeReadAddressInputClaims<C> {
    #[opening(Imm, from = RouterCycleVariant)]
    pub imm: C,
    #[opening(FallThroughPC, from = RouterCycleVariant)]
    pub fall_through_pc: C,
    #[opening(PCPlusImm, from = RouterCycleVariant)]
    pub pc_plus_imm: C,
    #[opening(PC, from = RouterCycleVariant)]
    pub pc: C,
    #[opening(NextPC, from = RouterCycleVariant)]
    pub next_pc: C,
    #[opening(Variant, from = RouterCycleVariant)]
    pub variant: C,
    #[opening(ShiftKind, from = RouterCycleShift)]
    pub shift_kind: C,
    #[opening(AccessKind, from = RouterCycleMemory)]
    pub access_kind: C,
    #[opening(KeyKind, from = RouterCycleCompare)]
    pub key_kind: C,
    #[opening(Branch, from = RouterCycleBranch)]
    pub branch: C,
    #[opening(Rs1Ra, from = RegistersReadChecking)]
    pub rs1_ra: C,
    #[opening(Rs2Ra, from = RegistersReadChecking)]
    pub rs2_ra: C,
    #[opening(RdWa, from = RegistersReadChecking)]
    pub rd_wa_read: C,
    #[opening(RdWa, from = RegistersValEvaluation)]
    pub rd_wa_write: C,
    #[opening(Store, from = RegistersValEvaluation)]
    pub store: C,
}
#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(BytecodeReadAddress)]
pub struct BytecodeReadAddressOutputClaims<C> {
    #[opening(BytecodeAddressClaim)]
    pub address_claim: C,
}
#[derive(Clone, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
pub struct BytecodeReadAddressChallenges<F> {
    #[challenge(BytecodeChallenge::Imm)]
    pub imm: F,
    #[challenge(BytecodeChallenge::FallThroughPC)]
    pub fall_through_pc: F,
    #[challenge(BytecodeChallenge::PCPlusImm)]
    pub pc_plus_imm: F,
    #[challenge(BytecodeChallenge::PC)]
    pub pc: F,
    #[challenge(BytecodeChallenge::Variant)]
    pub variant: F,
    #[challenge(BytecodeChallenge::ShiftKind)]
    pub shift_kind: F,
    #[challenge(BytecodeChallenge::AccessKind)]
    pub access_kind: F,
    #[challenge(BytecodeChallenge::KeyKind)]
    pub key_kind: F,
    #[challenge(BytecodeChallenge::Branch)]
    pub branch: F,
    #[challenge(BytecodeChallenge::Rs1Ra)]
    pub rs1_ra: F,
    #[challenge(BytecodeChallenge::Rs2Ra)]
    pub rs2_ra: F,
    #[challenge(BytecodeChallenge::RdWaRead)]
    pub rd_wa_read: F,
    #[challenge(BytecodeChallenge::RdWaWrite)]
    pub rd_wa_write: F,
    #[challenge(BytecodeChallenge::Store)]
    pub store: F,
    #[challenge(BytecodeChallenge::Entry)]
    pub entry: F,
    #[challenge(BytecodeChallenge::Next)]
    pub next: F,
}
/// The address summand is the sum of five products `H_t[k] R_t[k]`.
#[derive(Clone)]
pub struct BytecodeReadAddressSymbolic {
    rounds: usize,
}
impl SymbolicSumcheck for BytecodeReadAddressSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = usize;
    type Challenges<F> = BytecodeReadAddressChallenges<F>;
    type Inputs<C> = BytecodeReadAddressInputClaims<C>;
    type Outputs<C> = BytecodeReadAddressOutputClaims<C>;
    fn new(rounds: usize) -> Self {
        Self { rounds }
    }
    fn id() -> RelationId {
        RelationId::BytecodeReadAddress
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn degree(&self) -> usize {
        2
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        let terms = [
            (
                BytecodeChallenge::Imm,
                VirtualPolynomial::Imm,
                RelationId::RouterCycleVariant,
            ),
            (
                BytecodeChallenge::FallThroughPC,
                VirtualPolynomial::FallThroughPC,
                RelationId::RouterCycleVariant,
            ),
            (
                BytecodeChallenge::PCPlusImm,
                VirtualPolynomial::PCPlusImm,
                RelationId::RouterCycleVariant,
            ),
            (
                BytecodeChallenge::PC,
                VirtualPolynomial::PC,
                RelationId::RouterCycleVariant,
            ),
            (
                BytecodeChallenge::Variant,
                VirtualPolynomial::Variant,
                RelationId::RouterCycleVariant,
            ),
            (
                BytecodeChallenge::ShiftKind,
                VirtualPolynomial::ShiftKind,
                RelationId::RouterCycleShift,
            ),
            (
                BytecodeChallenge::AccessKind,
                VirtualPolynomial::AccessKind,
                RelationId::RouterCycleMemory,
            ),
            (
                BytecodeChallenge::KeyKind,
                VirtualPolynomial::KeyKind,
                RelationId::RouterCycleCompare,
            ),
            (
                BytecodeChallenge::Branch,
                VirtualPolynomial::Branch,
                RelationId::RouterCycleBranch,
            ),
            (
                BytecodeChallenge::Rs1Ra,
                VirtualPolynomial::Rs1Ra,
                RelationId::RegistersReadChecking,
            ),
            (
                BytecodeChallenge::Rs2Ra,
                VirtualPolynomial::Rs2Ra,
                RelationId::RegistersReadChecking,
            ),
            (
                BytecodeChallenge::RdWaRead,
                VirtualPolynomial::RdWa,
                RelationId::RegistersReadChecking,
            ),
            (
                BytecodeChallenge::RdWaWrite,
                VirtualPolynomial::RdWa,
                RelationId::RegistersValEvaluation,
            ),
            (
                BytecodeChallenge::Store,
                VirtualPolynomial::Store,
                RelationId::RegistersValEvaluation,
            ),
            (
                BytecodeChallenge::Next,
                VirtualPolynomial::NextPC,
                RelationId::RouterCycleVariant,
            ),
        ];
        terms
            .into_iter()
            .map(|(coefficient, polynomial, source)| {
                challenge(ChallengeId::BytecodeReadAddress(coefficient))
                    * opening(OpeningId::virtual_polynomial(polynomial, source))
            })
            .fold(Expr::zero(), |sum, term| sum + term)
            + challenge(ChallengeId::BytecodeReadAddress(BytecodeChallenge::Entry))
                * derived(DerivedId::BytecodeReadAddress(
                    BytecodeAddressDerived::EntryPc,
                ))
            + challenge(ChallengeId::BytecodeReadAddress(BytecodeChallenge::Next))
                * derived(DerivedId::BytecodeReadAddress(
                    BytecodeAddressDerived::FinalPc,
                ))
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::BytecodeAddressClaim,
            RelationId::BytecodeReadAddress,
        ))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(ids = crate::ids)]
pub struct BytecodeReadCycleInputClaims<C> {
    #[opening(BytecodeAddressClaim, from = BytecodeReadAddress)]
    pub address_claim: C,
}
#[derive(Clone, Debug, PartialEq, Eq, OutputClaims, Serialize, Deserialize)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
#[relation(BytecodeReadCycle)]
pub struct BytecodeReadCycleOutputClaims<C> {
    #[opening(BytecodeRaChunk)]
    pub chunks: Vec<C>,
}
pub type BytecodeReadCycleChallenges<F> = NoChallenges<F>;
/// The cycle summand is a weighted cycle selector times the address-chunk product.
#[derive(Clone)]
pub struct BytecodeReadCycleSymbolic {
    rounds: usize,
    chunks: usize,
}
impl SymbolicSumcheck for BytecodeReadCycleSymbolic {
    type RelationId = RelationId;
    type OpeningId = OpeningId;
    type DerivedId = DerivedId;
    type ChallengeId = ChallengeId;
    type Shape = (usize, usize);
    type Challenges<F> = BytecodeReadCycleChallenges<F>;
    type Inputs<C> = BytecodeReadCycleInputClaims<C>;
    type Outputs<C> = BytecodeReadCycleOutputClaims<C>;
    fn new((rounds, chunks): Self::Shape) -> Self {
        Self { rounds, chunks }
    }
    fn id() -> RelationId {
        RelationId::BytecodeReadCycle
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn degree(&self) -> usize {
        self.chunks + 1
    }
    fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
        opening(OpeningId::virtual_polynomial(
            VirtualPolynomial::BytecodeAddressClaim,
            RelationId::BytecodeReadAddress,
        ))
    }
    fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
        let product = (0..self.chunks)
            .map(|chunk| {
                opening(OpeningId::virtual_polynomial(
                    VirtualPolynomial::BytecodeRaChunk(chunk),
                    RelationId::BytecodeReadCycle,
                ))
            })
            .fold(Expr::one(), |product, chunk| product * chunk);
        [
            CycleWeight::Router,
            CycleWeight::Read,
            CycleWeight::Val,
            CycleWeight::Entry,
            CycleWeight::Next,
        ]
        .into_iter()
        .map(|weight| {
            derived(DerivedId::BytecodeReadCycle(
                BytecodeCycleDerived::BytecodeFold(weight),
            )) * derived(DerivedId::BytecodeReadCycle(BytecodeCycleDerived::Weight(
                weight,
            ))) * product.clone()
        })
        .fold(Expr::zero(), |sum, term| sum + term)
    }
}
