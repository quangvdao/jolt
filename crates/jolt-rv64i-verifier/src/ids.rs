//! RV64I binary identifiers encode an opening as `relation | kind << 8 |
//! tag << 16 | payload << 24`, a derived term as `relation | tag << 8 |
//! payload << 16`, and a challenge as `relation | tag << 8`.
//! Opening bits 9–15 and all unused payload bits are reserved. Decoding checks
//! the syntactic relation, tag and payload domains, rejects other families,
//! and preserves rejected composites unchanged. Oversized payloads saturate.

use jolt_claims::protocols::composed::{ComposedOpeningId, ExternalId};
use jolt_claims::Expr;
use jolt_verifier::stages::ids::{VerifierChallengeId, VerifierDerivedId};
use serde::{Deserialize, Serialize};

pub const FAMILY: &str = "rv64i-binary";
pub type FamilyExpr<F> = Expr<F, OpeningId, DerivedId, ChallengeId>;

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Router {
    Variant,
    Shift,
    Memory,
    Compare,
    Branch,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RowBlock {
    F2,
    F128,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CycleWeight {
    Router,
    Read,
    Val,
    Entry,
    Next,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RelationId {
    SpartanOuterF2,
    SpartanOuterF128,
    SpartanInner,
    RouterShort,
    RouterCycleVariant,
    RouterCycleShift,
    RouterCycleMemory,
    RouterCycleCompare,
    RouterCycleBranch,
    RegistersReadChecking,
    RamReadChecking,
    RamOutputCheck,
    RegistersValEvaluation,
    RamValEvaluation,
    BytecodeReadAddress,
    BytecodeReadCycle,
    RamRaProduct,
    BitsReduction,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CommittedPolynomial {
    DirectColumns,
    VariantBits,
    PosRa0,
    PosRa1,
    ShouldBranch,
    Inc,
    Column(usize),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VirtualPolynomial {
    Az,
    Bz,
    Cz,
    WitnessRouted,
    RouterFold(Router),
    Rs1Value,
    Rs2Value,
    RdPreValue,
    RamReadValue,
    NextPC,
    PC,
    Imm,
    FallThroughPC,
    PCPlusImm,
    Variant,
    ShiftKind,
    AccessKind,
    KeyKind,
    Branch,
    Store,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    RegistersVal,
    RamVal,
    RamValFinal,
    RamRa,
    BytecodeAddressClaim,
    BytecodeRaChunk(usize),
    RamRaChunk(usize),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PolynomialId {
    Committed(CommittedPolynomial),
    Virtual(VirtualPolynomial),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DerivedId {
    SpartanOuter(RowBlock, OuterDerived),
    SpartanInner(InnerDerived),
    RouterShort(RouterShortDerived),
    RouterCycle(Router, RouterCycleDerived),
    RegistersReadChecking(ReadCheckingDerived),
    RamReadChecking(ReadCheckingDerived),
    RamOutputCheck(OutputCheckDerived),
    RegistersValEvaluation(ValEvaluationDerived),
    RamValEvaluation(ValEvaluationDerived),
    BytecodeReadAddress(BytecodeAddressDerived),
    BytecodeReadCycle(BytecodeCycleDerived),
    RamRaProduct(RamRaProductDerived),
    BitsReduction(BitsReductionDerived),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ChallengeId {
    SpartanInner(InnerChallenge),
    RegistersReadChecking(RegistersReadChallenge),
    RamValEvaluation(RamValChallenge),
    BytecodeReadAddress(BytecodeChallenge),
    RamRaProduct(RamRaChallenge),
    BitsReduction(BitsReductionChallenge),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OuterDerived {
    EqTau,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InnerDerived {
    MatrixWeight,
    PublicColumns,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RouterShortDerived {
    RouteWeight(Router),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RouterCycleDerived {
    EqCycle,
    WordSlot(usize),
    OneSlot,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ReadCheckingDerived {
    EqCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OutputCheckDerived {
    EqTau,
    IoMask,
    ValIo,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ValEvaluationDerived {
    Lt,
    InitEval,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BytecodeAddressDerived {
    EntryPc,
    FinalPc,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BytecodeCycleDerived {
    BytecodeFold(CycleWeight),
    Weight(CycleWeight),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamRaProductDerived {
    EqRead,
    EqVal,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BitsReductionDerived {
    PosZero(usize),
    ColumnWeight(usize),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InnerChallenge {
    AzF2,
    BzF2,
    CzF2,
    AzF128,
    BzF128,
    CzF128,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegistersReadChallenge {
    Rs1,
    Rs2,
    Rd,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamValChallenge {
    Val,
    Final,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BytecodeChallenge {
    Imm,
    FallThroughPC,
    PCPlusImm,
    PC,
    Variant,
    ShiftKind,
    AccessKind,
    KeyKind,
    Branch,
    Rs1Ra,
    Rs2Ra,
    RdWaRead,
    RdWaWrite,
    Store,
    Entry,
    Next,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RamRaChallenge {
    Read,
    Val,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BitsReductionChallenge {
    DirectColumns,
    VariantBits,
    PosRa0,
    PosRa1,
    ShouldBranch,
    Inc,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OpeningId {
    pub polynomial: PolynomialId,
    pub relation: RelationId,
}
impl OpeningId {
    pub fn committed(polynomial: CommittedPolynomial, relation: RelationId) -> Self {
        Self {
            polynomial: PolynomialId::Committed(polynomial),
            relation,
        }
    }
    pub fn virtual_polynomial(polynomial: VirtualPolynomial, relation: RelationId) -> Self {
        Self {
            polynomial: PolynomialId::Virtual(polynomial),
            relation,
        }
    }
}

impl Router {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::Variant => (0, 0),
            Self::Shift => (1, 0),
            Self::Memory => (2, 0),
            Self::Compare => (3, 0),
            Self::Branch => (4, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::Variant),
            (1, 0) => Some(Self::Shift),
            (2, 0) => Some(Self::Memory),
            (3, 0) => Some(Self::Compare),
            (4, 0) => Some(Self::Branch),
            _ => None,
        }
    }
}

impl RowBlock {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::F2 => (0, 0),
            Self::F128 => (1, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::F2),
            (1, 0) => Some(Self::F128),
            _ => None,
        }
    }
}

impl CycleWeight {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::Router => (0, 0),
            Self::Read => (1, 0),
            Self::Val => (2, 0),
            Self::Entry => (3, 0),
            Self::Next => (4, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::Router),
            (1, 0) => Some(Self::Read),
            (2, 0) => Some(Self::Val),
            (3, 0) => Some(Self::Entry),
            (4, 0) => Some(Self::Next),
            _ => None,
        }
    }
}

impl RelationId {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::SpartanOuterF2 => (0, 0),
            Self::SpartanOuterF128 => (1, 0),
            Self::SpartanInner => (2, 0),
            Self::RouterShort => (3, 0),
            Self::RouterCycleVariant => (4, 0),
            Self::RouterCycleShift => (5, 0),
            Self::RouterCycleMemory => (6, 0),
            Self::RouterCycleCompare => (7, 0),
            Self::RouterCycleBranch => (8, 0),
            Self::RegistersReadChecking => (9, 0),
            Self::RamReadChecking => (10, 0),
            Self::RamOutputCheck => (11, 0),
            Self::RegistersValEvaluation => (12, 0),
            Self::RamValEvaluation => (13, 0),
            Self::BytecodeReadAddress => (14, 0),
            Self::BytecodeReadCycle => (15, 0),
            Self::RamRaProduct => (16, 0),
            Self::BitsReduction => (17, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::SpartanOuterF2),
            (1, 0) => Some(Self::SpartanOuterF128),
            (2, 0) => Some(Self::SpartanInner),
            (3, 0) => Some(Self::RouterShort),
            (4, 0) => Some(Self::RouterCycleVariant),
            (5, 0) => Some(Self::RouterCycleShift),
            (6, 0) => Some(Self::RouterCycleMemory),
            (7, 0) => Some(Self::RouterCycleCompare),
            (8, 0) => Some(Self::RouterCycleBranch),
            (9, 0) => Some(Self::RegistersReadChecking),
            (10, 0) => Some(Self::RamReadChecking),
            (11, 0) => Some(Self::RamOutputCheck),
            (12, 0) => Some(Self::RegistersValEvaluation),
            (13, 0) => Some(Self::RamValEvaluation),
            (14, 0) => Some(Self::BytecodeReadAddress),
            (15, 0) => Some(Self::BytecodeReadCycle),
            (16, 0) => Some(Self::RamRaProduct),
            (17, 0) => Some(Self::BitsReduction),
            _ => None,
        }
    }
}

impl CommittedPolynomial {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::DirectColumns => (0, 0),
            Self::VariantBits => (1, 0),
            Self::PosRa0 => (2, 0),
            Self::PosRa1 => (3, 0),
            Self::ShouldBranch => (4, 0),
            Self::Inc => (5, 0),
            Self::Column(payload) => (6, payload as u64),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::DirectColumns),
            (1, 0) => Some(Self::VariantBits),
            (2, 0) => Some(Self::PosRa0),
            (3, 0) => Some(Self::PosRa1),
            (4, 0) => Some(Self::ShouldBranch),
            (5, 0) => Some(Self::Inc),
            (6, payload) if payload < 256 => Some(Self::Column(usize::try_from(payload).ok()?)),
            _ => None,
        }
    }
}

impl VirtualPolynomial {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::Az => (0, 0),
            Self::Bz => (1, 0),
            Self::Cz => (2, 0),
            Self::WitnessRouted => (3, 0),
            Self::RouterFold(payload) => (4, payload.parts().0),
            Self::Rs1Value => (5, 0),
            Self::Rs2Value => (6, 0),
            Self::RdPreValue => (7, 0),
            Self::RamReadValue => (8, 0),
            Self::NextPC => (9, 0),
            Self::PC => (10, 0),
            Self::Imm => (11, 0),
            Self::FallThroughPC => (12, 0),
            Self::PCPlusImm => (13, 0),
            Self::Variant => (14, 0),
            Self::ShiftKind => (15, 0),
            Self::AccessKind => (16, 0),
            Self::KeyKind => (17, 0),
            Self::Branch => (18, 0),
            Self::Store => (19, 0),
            Self::Rs1Ra => (20, 0),
            Self::Rs2Ra => (21, 0),
            Self::RdWa => (22, 0),
            Self::RegistersVal => (23, 0),
            Self::RamVal => (24, 0),
            Self::RamValFinal => (25, 0),
            Self::RamRa => (26, 0),
            Self::BytecodeAddressClaim => (27, 0),
            Self::BytecodeRaChunk(payload) => (28, payload as u64),
            Self::RamRaChunk(payload) => (29, payload as u64),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::Az),
            (1, 0) => Some(Self::Bz),
            (2, 0) => Some(Self::Cz),
            (3, 0) => Some(Self::WitnessRouted),
            (4, payload) if payload < 5 => Some(Self::RouterFold(Router::decode(payload, 0)?)),
            (5, 0) => Some(Self::Rs1Value),
            (6, 0) => Some(Self::Rs2Value),
            (7, 0) => Some(Self::RdPreValue),
            (8, 0) => Some(Self::RamReadValue),
            (9, 0) => Some(Self::NextPC),
            (10, 0) => Some(Self::PC),
            (11, 0) => Some(Self::Imm),
            (12, 0) => Some(Self::FallThroughPC),
            (13, 0) => Some(Self::PCPlusImm),
            (14, 0) => Some(Self::Variant),
            (15, 0) => Some(Self::ShiftKind),
            (16, 0) => Some(Self::AccessKind),
            (17, 0) => Some(Self::KeyKind),
            (18, 0) => Some(Self::Branch),
            (19, 0) => Some(Self::Store),
            (20, 0) => Some(Self::Rs1Ra),
            (21, 0) => Some(Self::Rs2Ra),
            (22, 0) => Some(Self::RdWa),
            (23, 0) => Some(Self::RegistersVal),
            (24, 0) => Some(Self::RamVal),
            (25, 0) => Some(Self::RamValFinal),
            (26, 0) => Some(Self::RamRa),
            (27, 0) => Some(Self::BytecodeAddressClaim),
            (28, payload) if payload < 6 => {
                Some(Self::BytecodeRaChunk(usize::try_from(payload).ok()?))
            }
            (29, payload) if payload < 16 => Some(Self::RamRaChunk(usize::try_from(payload).ok()?)),
            _ => None,
        }
    }
}

impl OuterDerived {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::EqTau => (0, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::EqTau),
            _ => None,
        }
    }
}

impl InnerDerived {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::MatrixWeight => (0, 0),
            Self::PublicColumns => (1, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::MatrixWeight),
            (1, 0) => Some(Self::PublicColumns),
            _ => None,
        }
    }
}

impl RouterShortDerived {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::RouteWeight(payload) => (0, payload.parts().0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, payload) if payload < 5 => Some(Self::RouteWeight(Router::decode(payload, 0)?)),
            _ => None,
        }
    }
}

impl RouterCycleDerived {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::EqCycle => (0, 0),
            Self::WordSlot(payload) => (1, payload as u64),
            Self::OneSlot => (2, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::EqCycle),
            (1, payload) if payload < 8 => Some(Self::WordSlot(usize::try_from(payload).ok()?)),
            (2, 0) => Some(Self::OneSlot),
            _ => None,
        }
    }
}

impl ReadCheckingDerived {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::EqCycle => (0, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::EqCycle),
            _ => None,
        }
    }
}

impl OutputCheckDerived {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::EqTau => (0, 0),
            Self::IoMask => (1, 0),
            Self::ValIo => (2, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::EqTau),
            (1, 0) => Some(Self::IoMask),
            (2, 0) => Some(Self::ValIo),
            _ => None,
        }
    }
}

impl ValEvaluationDerived {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::Lt => (0, 0),
            Self::InitEval => (1, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::Lt),
            (1, 0) => Some(Self::InitEval),
            _ => None,
        }
    }
}

impl BytecodeAddressDerived {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::EntryPc => (0, 0),
            Self::FinalPc => (1, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::EntryPc),
            (1, 0) => Some(Self::FinalPc),
            _ => None,
        }
    }
}

impl BytecodeCycleDerived {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::BytecodeFold(payload) => (0, payload.parts().0),
            Self::Weight(payload) => (1, payload.parts().0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, payload) if payload < 5 => {
                Some(Self::BytecodeFold(CycleWeight::decode(payload, 0)?))
            }
            (1, payload) if payload < 5 => Some(Self::Weight(CycleWeight::decode(payload, 0)?)),
            _ => None,
        }
    }
}

impl RamRaProductDerived {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::EqRead => (0, 0),
            Self::EqVal => (1, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::EqRead),
            (1, 0) => Some(Self::EqVal),
            _ => None,
        }
    }
}

impl BitsReductionDerived {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::PosZero(payload) => (0, payload as u64),
            Self::ColumnWeight(payload) => (1, payload as u64),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, payload) if payload < 2 => Some(Self::PosZero(usize::try_from(payload).ok()?)),
            (1, payload) if payload < 256 => {
                Some(Self::ColumnWeight(usize::try_from(payload).ok()?))
            }
            _ => None,
        }
    }
}

impl InnerChallenge {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::AzF2 => (0, 0),
            Self::BzF2 => (1, 0),
            Self::CzF2 => (2, 0),
            Self::AzF128 => (3, 0),
            Self::BzF128 => (4, 0),
            Self::CzF128 => (5, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::AzF2),
            (1, 0) => Some(Self::BzF2),
            (2, 0) => Some(Self::CzF2),
            (3, 0) => Some(Self::AzF128),
            (4, 0) => Some(Self::BzF128),
            (5, 0) => Some(Self::CzF128),
            _ => None,
        }
    }
}

impl RegistersReadChallenge {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::Rs1 => (0, 0),
            Self::Rs2 => (1, 0),
            Self::Rd => (2, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::Rs1),
            (1, 0) => Some(Self::Rs2),
            (2, 0) => Some(Self::Rd),
            _ => None,
        }
    }
}

impl RamValChallenge {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::Val => (0, 0),
            Self::Final => (1, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::Val),
            (1, 0) => Some(Self::Final),
            _ => None,
        }
    }
}

impl BytecodeChallenge {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::Imm => (0, 0),
            Self::FallThroughPC => (1, 0),
            Self::PCPlusImm => (2, 0),
            Self::PC => (3, 0),
            Self::Variant => (4, 0),
            Self::ShiftKind => (5, 0),
            Self::AccessKind => (6, 0),
            Self::KeyKind => (7, 0),
            Self::Branch => (8, 0),
            Self::Rs1Ra => (9, 0),
            Self::Rs2Ra => (10, 0),
            Self::RdWaRead => (11, 0),
            Self::RdWaWrite => (12, 0),
            Self::Store => (13, 0),
            Self::Entry => (14, 0),
            Self::Next => (15, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::Imm),
            (1, 0) => Some(Self::FallThroughPC),
            (2, 0) => Some(Self::PCPlusImm),
            (3, 0) => Some(Self::PC),
            (4, 0) => Some(Self::Variant),
            (5, 0) => Some(Self::ShiftKind),
            (6, 0) => Some(Self::AccessKind),
            (7, 0) => Some(Self::KeyKind),
            (8, 0) => Some(Self::Branch),
            (9, 0) => Some(Self::Rs1Ra),
            (10, 0) => Some(Self::Rs2Ra),
            (11, 0) => Some(Self::RdWaRead),
            (12, 0) => Some(Self::RdWaWrite),
            (13, 0) => Some(Self::Store),
            (14, 0) => Some(Self::Entry),
            (15, 0) => Some(Self::Next),
            _ => None,
        }
    }
}

impl RamRaChallenge {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::Read => (0, 0),
            Self::Val => (1, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::Read),
            (1, 0) => Some(Self::Val),
            _ => None,
        }
    }
}

impl BitsReductionChallenge {
    fn parts(self) -> (u64, u64) {
        match self {
            Self::DirectColumns => (0, 0),
            Self::VariantBits => (1, 0),
            Self::PosRa0 => (2, 0),
            Self::PosRa1 => (3, 0),
            Self::ShouldBranch => (4, 0),
            Self::Inc => (5, 0),
        }
    }
    fn decode(tag: u64, payload: u64) -> Option<Self> {
        match (tag, payload) {
            (0, 0) => Some(Self::DirectColumns),
            (1, 0) => Some(Self::VariantBits),
            (2, 0) => Some(Self::PosRa0),
            (3, 0) => Some(Self::PosRa1),
            (4, 0) => Some(Self::ShouldBranch),
            (5, 0) => Some(Self::Inc),
            _ => None,
        }
    }
}

impl From<OpeningId> for ComposedOpeningId {
    fn from(id: OpeningId) -> Self {
        let (kind, tag, payload) = match id.polynomial {
            PolynomialId::Virtual(p) => {
                let (tag, payload) = p.parts();
                (0, tag, payload)
            }
            PolynomialId::Committed(p) => {
                let (tag, payload) = p.parts();
                (1, tag, payload)
            }
        };
        Self::External(ExternalId {
            family: FAMILY,
            index: id.relation.parts().0
                | kind << 8
                | tag << 16
                | payload.min((1_u64 << 40) - 1) << 24,
        })
    }
}
impl TryFrom<ComposedOpeningId> for OpeningId {
    type Error = ComposedOpeningId;
    fn try_from(id: ComposedOpeningId) -> Result<Self, Self::Error> {
        let index = match id {
            ComposedOpeningId::External(ExternalId {
                family: FAMILY,
                index,
            }) => index,
            ComposedOpeningId::External(_)
            | ComposedOpeningId::Jolt(_)
            | ComposedOpeningId::FieldInline(_) => return Err(id),
        };
        let relation = RelationId::decode(index & 255, 0).ok_or(id)?;
        let tag = (index >> 16) & 255;
        let payload = index >> 24;
        let polynomial = match (index >> 8) & 255 {
            0 => PolynomialId::Virtual(VirtualPolynomial::decode(tag, payload).ok_or(id)?),
            1 => PolynomialId::Committed(CommittedPolynomial::decode(tag, payload).ok_or(id)?),
            _ => return Err(id),
        };
        let decoded = Self {
            polynomial,
            relation,
        };
        if ComposedOpeningId::from(decoded) == id {
            Ok(decoded)
        } else {
            Err(id)
        }
    }
}
impl From<DerivedId> for VerifierDerivedId {
    fn from(id: DerivedId) -> Self {
        let (relation, tag, payload) = match id {
            DerivedId::SpartanOuter(block, term) => {
                let (tag, payload) = term.parts();
                (block.parts().0, tag, payload)
            }
            DerivedId::SpartanInner(term) => {
                let (tag, payload) = term.parts();
                (2, tag, payload)
            }
            DerivedId::RouterShort(term) => {
                let (tag, payload) = term.parts();
                (3, tag, payload)
            }
            DerivedId::RouterCycle(router, term) => {
                let (tag, payload) = term.parts();
                (4 + router.parts().0, tag, payload)
            }
            DerivedId::RegistersReadChecking(term) => {
                let (tag, payload) = term.parts();
                (9, tag, payload)
            }
            DerivedId::RamReadChecking(term) => {
                let (tag, payload) = term.parts();
                (10, tag, payload)
            }
            DerivedId::RamOutputCheck(term) => {
                let (tag, payload) = term.parts();
                (11, tag, payload)
            }
            DerivedId::RegistersValEvaluation(term) => {
                let (tag, payload) = term.parts();
                (12, tag, payload)
            }
            DerivedId::RamValEvaluation(term) => {
                let (tag, payload) = term.parts();
                (13, tag, payload)
            }
            DerivedId::BytecodeReadAddress(term) => {
                let (tag, payload) = term.parts();
                (14, tag, payload)
            }
            DerivedId::BytecodeReadCycle(term) => {
                let (tag, payload) = term.parts();
                (15, tag, payload)
            }
            DerivedId::RamRaProduct(term) => {
                let (tag, payload) = term.parts();
                (16, tag, payload)
            }
            DerivedId::BitsReduction(term) => {
                let (tag, payload) = term.parts();
                (17, tag, payload)
            }
        };
        Self::External(ExternalId {
            family: FAMILY,
            index: relation | tag << 8 | payload.min((1_u64 << 48) - 1) << 16,
        })
    }
}
impl TryFrom<VerifierDerivedId> for DerivedId {
    type Error = VerifierDerivedId;
    fn try_from(id: VerifierDerivedId) -> Result<Self, Self::Error> {
        let index = match id {
            VerifierDerivedId::External(ExternalId {
                family: FAMILY,
                index,
            }) => index,
            VerifierDerivedId::External(_)
            | VerifierDerivedId::Jolt(_)
            | VerifierDerivedId::FieldInline(_) => return Err(id),
        };
        let tag = (index >> 8) & 255;
        let payload = index >> 16;
        let decoded = match index & 255 {
            0 => Self::SpartanOuter(
                RowBlock::decode(0, 0).ok_or(id)?,
                OuterDerived::decode(tag, payload).ok_or(id)?,
            ),
            1 => Self::SpartanOuter(
                RowBlock::decode(1, 0).ok_or(id)?,
                OuterDerived::decode(tag, payload).ok_or(id)?,
            ),
            2 => Self::SpartanInner(InnerDerived::decode(tag, payload).ok_or(id)?),
            3 => Self::RouterShort(RouterShortDerived::decode(tag, payload).ok_or(id)?),
            4 => Self::RouterCycle(
                Router::decode(0, 0).ok_or(id)?,
                RouterCycleDerived::decode(tag, payload).ok_or(id)?,
            ),
            5 => Self::RouterCycle(
                Router::decode(1, 0).ok_or(id)?,
                RouterCycleDerived::decode(tag, payload).ok_or(id)?,
            ),
            6 => Self::RouterCycle(
                Router::decode(2, 0).ok_or(id)?,
                RouterCycleDerived::decode(tag, payload).ok_or(id)?,
            ),
            7 => Self::RouterCycle(
                Router::decode(3, 0).ok_or(id)?,
                RouterCycleDerived::decode(tag, payload).ok_or(id)?,
            ),
            8 => Self::RouterCycle(
                Router::decode(4, 0).ok_or(id)?,
                RouterCycleDerived::decode(tag, payload).ok_or(id)?,
            ),
            9 => Self::RegistersReadChecking(ReadCheckingDerived::decode(tag, payload).ok_or(id)?),
            10 => Self::RamReadChecking(ReadCheckingDerived::decode(tag, payload).ok_or(id)?),
            11 => Self::RamOutputCheck(OutputCheckDerived::decode(tag, payload).ok_or(id)?),
            12 => {
                Self::RegistersValEvaluation(ValEvaluationDerived::decode(tag, payload).ok_or(id)?)
            }
            13 => Self::RamValEvaluation(ValEvaluationDerived::decode(tag, payload).ok_or(id)?),
            14 => {
                Self::BytecodeReadAddress(BytecodeAddressDerived::decode(tag, payload).ok_or(id)?)
            }
            15 => Self::BytecodeReadCycle(BytecodeCycleDerived::decode(tag, payload).ok_or(id)?),
            16 => Self::RamRaProduct(RamRaProductDerived::decode(tag, payload).ok_or(id)?),
            17 => Self::BitsReduction(BitsReductionDerived::decode(tag, payload).ok_or(id)?),
            _ => return Err(id),
        };
        if VerifierDerivedId::from(decoded) == id {
            Ok(decoded)
        } else {
            Err(id)
        }
    }
}
impl From<InnerChallenge> for ChallengeId {
    fn from(id: InnerChallenge) -> Self {
        Self::SpartanInner(id)
    }
}
impl From<RegistersReadChallenge> for ChallengeId {
    fn from(id: RegistersReadChallenge) -> Self {
        Self::RegistersReadChecking(id)
    }
}
impl From<RamValChallenge> for ChallengeId {
    fn from(id: RamValChallenge) -> Self {
        Self::RamValEvaluation(id)
    }
}
impl From<BytecodeChallenge> for ChallengeId {
    fn from(id: BytecodeChallenge) -> Self {
        Self::BytecodeReadAddress(id)
    }
}
impl From<RamRaChallenge> for ChallengeId {
    fn from(id: RamRaChallenge) -> Self {
        Self::RamRaProduct(id)
    }
}
impl From<BitsReductionChallenge> for ChallengeId {
    fn from(id: BitsReductionChallenge) -> Self {
        Self::BitsReduction(id)
    }
}
impl From<ChallengeId> for VerifierChallengeId {
    fn from(id: ChallengeId) -> Self {
        let (relation, tag) = match id {
            ChallengeId::SpartanInner(term) => (2, term.parts().0),
            ChallengeId::RegistersReadChecking(term) => (9, term.parts().0),
            ChallengeId::RamValEvaluation(term) => (13, term.parts().0),
            ChallengeId::BytecodeReadAddress(term) => (14, term.parts().0),
            ChallengeId::RamRaProduct(term) => (16, term.parts().0),
            ChallengeId::BitsReduction(term) => (17, term.parts().0),
        };
        Self::External(ExternalId {
            family: FAMILY,
            index: relation | tag << 8,
        })
    }
}
impl TryFrom<VerifierChallengeId> for ChallengeId {
    type Error = VerifierChallengeId;
    fn try_from(id: VerifierChallengeId) -> Result<Self, Self::Error> {
        let index = match id {
            VerifierChallengeId::External(ExternalId {
                family: FAMILY,
                index,
            }) => index,
            VerifierChallengeId::External(_)
            | VerifierChallengeId::Jolt(_)
            | VerifierChallengeId::FieldInline(_) => return Err(id),
        };
        let tag = (index >> 8) & 255;
        let payload = index >> 16;
        let decoded = match index & 255 {
            2 => Self::SpartanInner(InnerChallenge::decode(tag, payload).ok_or(id)?),
            9 => {
                Self::RegistersReadChecking(RegistersReadChallenge::decode(tag, payload).ok_or(id)?)
            }
            13 => Self::RamValEvaluation(RamValChallenge::decode(tag, payload).ok_or(id)?),
            14 => Self::BytecodeReadAddress(BytecodeChallenge::decode(tag, payload).ok_or(id)?),
            16 => Self::RamRaProduct(RamRaChallenge::decode(tag, payload).ok_or(id)?),
            17 => Self::BitsReduction(BitsReductionChallenge::decode(tag, payload).ok_or(id)?),
            _ => return Err(id),
        };
        if VerifierChallengeId::from(decoded) == id {
            Ok(decoded)
        } else {
            Err(id)
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "identifier compatibility tests fail by assertions"
)]
mod tests {
    use super::*;
    use jolt_claims::protocols::jolt::{
        JoltChallengeId, JoltDerivedId, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
        RamReadWriteChallenge, RamReadWritePublic,
    };
    use jolt_rv64i_arith::Layout;
    use std::collections::BTreeSet;

    #[test]
    fn identifiers_round_trip_without_collisions_at_all_layouts() {
        let relations = [
            RelationId::SpartanOuterF2,
            RelationId::SpartanOuterF128,
            RelationId::SpartanInner,
            RelationId::RouterShort,
            RelationId::RouterCycleVariant,
            RelationId::RouterCycleShift,
            RelationId::RouterCycleMemory,
            RelationId::RouterCycleCompare,
            RelationId::RouterCycleBranch,
            RelationId::RegistersReadChecking,
            RelationId::RamReadChecking,
            RelationId::RamOutputCheck,
            RelationId::RegistersValEvaluation,
            RelationId::RamValEvaluation,
            RelationId::BytecodeReadAddress,
            RelationId::BytecodeReadCycle,
            RelationId::RamRaProduct,
            RelationId::BitsReduction,
        ];
        let routers = [
            Router::Variant,
            Router::Shift,
            Router::Memory,
            Router::Compare,
            Router::Branch,
        ];
        let cycle_weights = [
            CycleWeight::Router,
            CycleWeight::Read,
            CycleWeight::Val,
            CycleWeight::Entry,
            CycleWeight::Next,
        ];
        let mut virtuals = vec![
            VirtualPolynomial::Az,
            VirtualPolynomial::Bz,
            VirtualPolynomial::Cz,
            VirtualPolynomial::WitnessRouted,
            VirtualPolynomial::Rs1Value,
            VirtualPolynomial::Rs2Value,
            VirtualPolynomial::RdPreValue,
            VirtualPolynomial::RamReadValue,
            VirtualPolynomial::NextPC,
            VirtualPolynomial::PC,
            VirtualPolynomial::Imm,
            VirtualPolynomial::FallThroughPC,
            VirtualPolynomial::PCPlusImm,
            VirtualPolynomial::Variant,
            VirtualPolynomial::ShiftKind,
            VirtualPolynomial::AccessKind,
            VirtualPolynomial::KeyKind,
            VirtualPolynomial::Branch,
            VirtualPolynomial::Store,
            VirtualPolynomial::Rs1Ra,
            VirtualPolynomial::Rs2Ra,
            VirtualPolynomial::RdWa,
            VirtualPolynomial::RegistersVal,
            VirtualPolynomial::RamVal,
            VirtualPolynomial::RamValFinal,
            VirtualPolynomial::RamRa,
            VirtualPolynomial::BytecodeAddressClaim,
        ];
        virtuals.extend(routers.into_iter().map(VirtualPolynomial::RouterFold));
        virtuals.extend((0..6).map(VirtualPolynomial::BytecodeRaChunk));
        virtuals.extend((0..16).map(VirtualPolynomial::RamRaChunk));
        let mut committed = vec![
            CommittedPolynomial::DirectColumns,
            CommittedPolynomial::VariantBits,
            CommittedPolynomial::PosRa0,
            CommittedPolynomial::PosRa1,
            CommittedPolynomial::ShouldBranch,
            CommittedPolynomial::Inc,
        ];
        committed.extend((0..256).map(CommittedPolynomial::Column));
        let mut derived = vec![
            DerivedId::SpartanInner(InnerDerived::MatrixWeight),
            DerivedId::SpartanInner(InnerDerived::PublicColumns),
            DerivedId::RegistersReadChecking(ReadCheckingDerived::EqCycle),
            DerivedId::RamReadChecking(ReadCheckingDerived::EqCycle),
            DerivedId::RamOutputCheck(OutputCheckDerived::EqTau),
            DerivedId::RamOutputCheck(OutputCheckDerived::IoMask),
            DerivedId::RamOutputCheck(OutputCheckDerived::ValIo),
            DerivedId::RegistersValEvaluation(ValEvaluationDerived::Lt),
            DerivedId::RegistersValEvaluation(ValEvaluationDerived::InitEval),
            DerivedId::RamValEvaluation(ValEvaluationDerived::Lt),
            DerivedId::RamValEvaluation(ValEvaluationDerived::InitEval),
            DerivedId::BytecodeReadAddress(BytecodeAddressDerived::EntryPc),
            DerivedId::BytecodeReadAddress(BytecodeAddressDerived::FinalPc),
            DerivedId::RamRaProduct(RamRaProductDerived::EqRead),
            DerivedId::RamRaProduct(RamRaProductDerived::EqVal),
            DerivedId::SpartanOuter(RowBlock::F2, OuterDerived::EqTau),
            DerivedId::SpartanOuter(RowBlock::F128, OuterDerived::EqTau),
        ];
        for router in routers {
            derived.push(DerivedId::RouterShort(RouterShortDerived::RouteWeight(
                router,
            )));
            for term in [RouterCycleDerived::EqCycle, RouterCycleDerived::OneSlot]
                .into_iter()
                .chain((0..8).map(RouterCycleDerived::WordSlot))
            {
                derived.push(DerivedId::RouterCycle(router, term));
            }
        }
        for weight in cycle_weights {
            derived.push(DerivedId::BytecodeReadCycle(
                BytecodeCycleDerived::BytecodeFold(weight),
            ));
            derived.push(DerivedId::BytecodeReadCycle(BytecodeCycleDerived::Weight(
                weight,
            )));
        }
        derived.extend((0..2).map(|i| DerivedId::BitsReduction(BitsReductionDerived::PosZero(i))));
        derived.extend(
            (0..256).map(|i| DerivedId::BitsReduction(BitsReductionDerived::ColumnWeight(i))),
        );
        let challenges = [
            ChallengeId::SpartanInner(InnerChallenge::AzF2),
            ChallengeId::SpartanInner(InnerChallenge::BzF2),
            ChallengeId::SpartanInner(InnerChallenge::CzF2),
            ChallengeId::SpartanInner(InnerChallenge::AzF128),
            ChallengeId::SpartanInner(InnerChallenge::BzF128),
            ChallengeId::SpartanInner(InnerChallenge::CzF128),
            ChallengeId::RegistersReadChecking(RegistersReadChallenge::Rs1),
            ChallengeId::RegistersReadChecking(RegistersReadChallenge::Rs2),
            ChallengeId::RegistersReadChecking(RegistersReadChallenge::Rd),
            ChallengeId::RamValEvaluation(RamValChallenge::Val),
            ChallengeId::RamValEvaluation(RamValChallenge::Final),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::Imm),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::FallThroughPC),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::PCPlusImm),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::PC),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::Variant),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::ShiftKind),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::AccessKind),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::KeyKind),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::Branch),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::Rs1Ra),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::Rs2Ra),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::RdWaRead),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::RdWaWrite),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::Store),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::Entry),
            ChallengeId::BytecodeReadAddress(BytecodeChallenge::Next),
            ChallengeId::RamRaProduct(RamRaChallenge::Read),
            ChallengeId::RamRaProduct(RamRaChallenge::Val),
            ChallengeId::BitsReduction(BitsReductionChallenge::DirectColumns),
            ChallengeId::BitsReduction(BitsReductionChallenge::VariantBits),
            ChallengeId::BitsReduction(BitsReductionChallenge::PosRa0),
            ChallengeId::BitsReduction(BitsReductionChallenge::PosRa1),
            ChallengeId::BitsReduction(BitsReductionChallenge::ShouldBranch),
            ChallengeId::BitsReduction(BitsReductionChallenge::Inc),
        ];
        for (b, a) in [(4, 5), (8, 10), (20, 20), (20, 27)] {
            let layout = Layout::new(b, a, 0).unwrap();
            assert!(layout.used_columns() <= 256);
            let mut indices = BTreeSet::new();
            for relation in relations {
                for id in virtuals
                    .iter()
                    .copied()
                    .map(|p| OpeningId::virtual_polynomial(p, relation))
                    .chain(
                        committed
                            .iter()
                            .copied()
                            .map(|p| OpeningId::committed(p, relation)),
                    )
                {
                    let composite = ComposedOpeningId::from(id);
                    assert_eq!(OpeningId::try_from(composite), Ok(id));
                    assert!(indices.insert(composite));
                }
            }
            let mut indices = BTreeSet::new();
            for id in &derived {
                let composite = VerifierDerivedId::from(*id);
                assert_eq!(DerivedId::try_from(composite), Ok(*id));
                assert!(indices.insert(composite));
            }
            let mut indices = BTreeSet::new();
            for id in challenges {
                let composite = VerifierChallengeId::from(id);
                assert_eq!(ChallengeId::try_from(composite), Ok(id));
                assert!(indices.insert(composite));
            }
        }
        let inc = ComposedOpeningId::from(OpeningId::committed(
            CommittedPolynomial::Inc,
            RelationId::RegistersValEvaluation,
        ));
        assert_eq!(
            inc,
            ComposedOpeningId::External(ExternalId {
                family: FAMILY,
                index: 0x5_010c
            })
        );
        let column = ComposedOpeningId::from(OpeningId::committed(
            CommittedPolynomial::Column(229),
            RelationId::BitsReduction,
        ));
        assert_eq!(
            column,
            ComposedOpeningId::External(ExternalId {
                family: FAMILY,
                index: 0xe506_0111
            })
        );
    }

    #[test]
    fn identifier_decoders_enforce_every_syntactic_domain() {
        for relation in 0..=18_u64 {
            for kind in 0..=2_u64 {
                for tag in 0..=30_u64 {
                    let limit = match (kind, tag) {
                        (0, 4) => 5,
                        (0, 28) => 6,
                        (0, 29) => 16,
                        (0, 0..=29) | (1, 0..=5) => 1,
                        (1, 6) => 256,
                        _ => 0,
                    };
                    for payload in 0..=limit {
                        let composite = ComposedOpeningId::External(ExternalId {
                            family: FAMILY,
                            index: relation | kind << 8 | tag << 16 | payload << 24,
                        });
                        if relation < 18 && payload < limit {
                            let id = OpeningId::try_from(composite).unwrap();
                            assert_eq!(ComposedOpeningId::from(id), composite);
                        } else {
                            assert_eq!(OpeningId::try_from(composite), Err(composite));
                        }
                    }
                }
            }
        }
        for relation in 0..=18_u64 {
            for tag in 0..=3_u64 {
                let limit = match (relation, tag) {
                    (0..=1 | 9..=10, 0) => 1,
                    (2 | 12..=14 | 16, 0..=1) => 1,
                    (3, 0) => 5,
                    (4..=8, 1) => 8,
                    (4..=8, 0 | 2) | (11, 0..=2) => 1,
                    (15, 0..=1) => 5,
                    (17, 0) => 2,
                    (17, 1) => 256,
                    _ => 0,
                };
                for payload in 0..=limit {
                    let composite = VerifierDerivedId::External(ExternalId {
                        family: FAMILY,
                        index: relation | tag << 8 | payload << 16,
                    });
                    if payload < limit {
                        let id = DerivedId::try_from(composite).unwrap();
                        assert_eq!(VerifierDerivedId::from(id), composite);
                    } else {
                        assert_eq!(DerivedId::try_from(composite), Err(composite));
                    }
                }
            }
        }
        for relation in 0..=18_u64 {
            for tag in 0..=16_u64 {
                for payload in 0..=1_u64 {
                    let limit = match relation {
                        2 | 17 => 6,
                        9 => 3,
                        13 | 16 => 2,
                        14 => 16,
                        _ => 0,
                    };
                    let composite = VerifierChallengeId::External(ExternalId {
                        family: FAMILY,
                        index: relation | tag << 8 | payload << 16,
                    });
                    if tag < limit && payload == 0 {
                        let id = ChallengeId::try_from(composite).unwrap();
                        assert_eq!(VerifierChallengeId::from(id), composite);
                    } else {
                        assert_eq!(ChallengeId::try_from(composite), Err(composite));
                    }
                }
            }
        }
    }

    #[test]
    fn identifier_decoders_preserve_foreign_and_reserved_composites() {
        let foreign = ExternalId {
            family: "another-binary-family",
            index: 0,
        };
        let opening = ComposedOpeningId::External(foreign);
        assert_eq!(OpeningId::try_from(opening), Err(opening));
        let derived = VerifierDerivedId::External(foreign);
        assert_eq!(DerivedId::try_from(derived), Err(derived));
        let challenge = VerifierChallengeId::External(foreign);
        assert_eq!(ChallengeId::try_from(challenge), Err(challenge));
        let opening = ComposedOpeningId::Jolt(JoltOpeningId::virtual_polynomial(
            JoltVirtualPolynomial::RamRa,
            JoltRelationId::RamReadWriteChecking,
        ));
        assert_eq!(OpeningId::try_from(opening), Err(opening));
        let derived =
            VerifierDerivedId::Jolt(JoltDerivedId::RamReadWrite(RamReadWritePublic::EqCycle));
        assert_eq!(DerivedId::try_from(derived), Err(derived));
        let challenge =
            VerifierChallengeId::Jolt(JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma));
        assert_eq!(ChallengeId::try_from(challenge), Err(challenge));
        for bit in 9..16 {
            let opening = ComposedOpeningId::External(ExternalId {
                family: FAMILY,
                index: 1 << bit,
            });
            assert_eq!(OpeningId::try_from(opening), Err(opening));
        }
        for bit in 16..64 {
            let challenge = VerifierChallengeId::External(ExternalId {
                family: FAMILY,
                index: 0x11 | 1 << bit,
            });
            assert_eq!(ChallengeId::try_from(challenge), Err(challenge));
        }
        let huge = ComposedOpeningId::from(OpeningId::committed(
            CommittedPolynomial::Column(usize::MAX),
            RelationId::BitsReduction,
        ));
        assert_eq!(OpeningId::try_from(huge), Err(huge));
        let huge = VerifierDerivedId::from(DerivedId::BitsReduction(
            BitsReductionDerived::ColumnWeight(usize::MAX),
        ));
        assert_eq!(DerivedId::try_from(huge), Err(huge));
    }
}
