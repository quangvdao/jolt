//! Fixed bytecode selector indices and kinds read from the decode table.

use crate::decode::{Line, Rails, Source, DECODE_TABLE, NOOP_LINE};
use jolt_riscv::SourceInstructionKind as Kind;

/// Bit index in the 64-bit Variant column; indices 58..64 are reserved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
#[expect(
    non_camel_case_types,
    reason = "selector names follow instruction mnemonics"
)]
pub enum Variant {
    ADD = 0,
    ADDI = 1,
    SUB = 2,
    ADDW = 3,
    ADDIW = 4,
    SUBW = 5,
    AND = 6,
    ANDI = 7,
    OR = 8,
    ORI = 9,
    XOR = 10,
    XORI = 11,
    SLT = 12,
    SLTI = 13,
    SLTU = 14,
    SLTIU = 15,
    SLL = 16,
    SRL = 17,
    SRA = 18,
    SLLI = 19,
    SRLI = 20,
    SRAI = 21,
    SLLW = 22,
    SRLW = 23,
    SRAW = 24,
    SLLIW = 25,
    SRLIW = 26,
    SRAIW = 27,
    LB = 28,
    LH = 29,
    LW = 30,
    LD = 31,
    LBU = 32,
    LHU = 33,
    LWU = 34,
    SB = 35,
    SH = 36,
    SW = 37,
    SD = 38,
    BEQ = 39,
    BNE = 40,
    BLT = 41,
    BGE = 42,
    BLTU = 43,
    BGEU = 44,
    JAL = 45,
    JALR = 46,
    LUI = 47,
    AUIPC = 48,
    NOOP = 49,
    ECALL = 50,
    EBREAK = 51,
    JAL_X0 = 52,
    JALR_X0 = 53,
    LOAD1_X0 = 54,
    LOAD2_X0 = 55,
    LOAD4_X0 = 56,
    LOAD8_X0 = 57,
}

impl Variant {
    /// Number of occupied Variant selector bits.
    pub const COUNT: usize = 58;
    /// Variants in their wire order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::ADD,
        Self::ADDI,
        Self::SUB,
        Self::ADDW,
        Self::ADDIW,
        Self::SUBW,
        Self::AND,
        Self::ANDI,
        Self::OR,
        Self::ORI,
        Self::XOR,
        Self::XORI,
        Self::SLT,
        Self::SLTI,
        Self::SLTU,
        Self::SLTIU,
        Self::SLL,
        Self::SRL,
        Self::SRA,
        Self::SLLI,
        Self::SRLI,
        Self::SRAI,
        Self::SLLW,
        Self::SRLW,
        Self::SRAW,
        Self::SLLIW,
        Self::SRLIW,
        Self::SRAIW,
        Self::LB,
        Self::LH,
        Self::LW,
        Self::LD,
        Self::LBU,
        Self::LHU,
        Self::LWU,
        Self::SB,
        Self::SH,
        Self::SW,
        Self::SD,
        Self::BEQ,
        Self::BNE,
        Self::BLT,
        Self::BGE,
        Self::BLTU,
        Self::BGEU,
        Self::JAL,
        Self::JALR,
        Self::LUI,
        Self::AUIPC,
        Self::NOOP,
        Self::ECALL,
        Self::EBREAK,
        Self::JAL_X0,
        Self::JALR_X0,
        Self::LOAD1_X0,
        Self::LOAD2_X0,
        Self::LOAD4_X0,
        Self::LOAD8_X0,
    ];
    /// Position of this variant in the one-hot bytecode column.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// RV64I identity, with writes to x0 mapped to their no-write variant.
    /// Unsupported source kinds return None.
    pub fn from_source(kind: Kind, rd_is_x0: bool) -> Option<Self> {
        let variant = SOURCE_VARIANTS
            .iter()
            .find_map(|(source, variant)| (*source == kind).then_some(*variant))?;
        if !rd_is_x0 {
            return Some(variant);
        }
        Some(match variant {
            Self::JAL => Self::JAL_X0,
            Self::JALR => Self::JALR_X0,
            Self::LB | Self::LBU => Self::LOAD1_X0,
            Self::LH | Self::LHU => Self::LOAD2_X0,
            Self::LW | Self::LWU => Self::LOAD4_X0,
            Self::LD => Self::LOAD8_X0,
            Self::ADD
            | Self::ADDI
            | Self::SUB
            | Self::ADDW
            | Self::ADDIW
            | Self::SUBW
            | Self::AND
            | Self::ANDI
            | Self::OR
            | Self::ORI
            | Self::XOR
            | Self::XORI
            | Self::SLT
            | Self::SLTI
            | Self::SLTU
            | Self::SLTIU
            | Self::SLL
            | Self::SRL
            | Self::SRA
            | Self::SLLI
            | Self::SRLI
            | Self::SRAI
            | Self::SLLW
            | Self::SRLW
            | Self::SRAW
            | Self::SLLIW
            | Self::SRLIW
            | Self::SRAIW
            | Self::LUI
            | Self::AUIPC
            | Self::NOOP => Self::NOOP,
            Self::SB
            | Self::SH
            | Self::SW
            | Self::SD
            | Self::BEQ
            | Self::BNE
            | Self::BLT
            | Self::BGE
            | Self::BLTU
            | Self::BGEU
            | Self::ECALL
            | Self::EBREAK
            | Self::JAL_X0
            | Self::JALR_X0
            | Self::LOAD1_X0
            | Self::LOAD2_X0
            | Self::LOAD4_X0
            | Self::LOAD8_X0 => variant,
        })
    }
    /// The static decode-table row at this variant's index.
    #[inline]
    pub fn line(self) -> &'static Line {
        DECODE_TABLE.get(self.index()).unwrap_or(&NOOP_LINE)
    }
    /// Shift selector and its register or immediate amount source.
    #[inline]
    pub fn shift(self) -> Option<Shift> {
        self.line().shift
    }
    /// Access width and kind; loads to x0 carry width with no output kind.
    #[inline]
    pub fn access(self) -> Option<Access> {
        self.line().access
    }
    /// Comparison-key selector, present exactly on compare rails.
    #[inline]
    pub fn key_kind(self) -> Option<KeyKind> {
        match self.line().rails {
            Rails::Compare { keys, .. } => Some(keys),
            Rails::None | Rails::Adder { .. } | Rails::And { .. } => None,
        }
    }
    /// Branch predicate selector.
    #[inline]
    pub fn branch(self) -> Option<BranchCondition> {
        self.line().branch
    }
    /// Store = the decode row names a store AccessKind.
    #[inline]
    pub fn is_store(self) -> bool {
        self.access()
            .and_then(|access| access.kind)
            .is_some_and(AccessKind::is_store)
    }
}

const SOURCE_VARIANTS: [(Kind, Variant); 52] = [
    (Kind::ADD, Variant::ADD),
    (Kind::ADDI, Variant::ADDI),
    (Kind::SUB, Variant::SUB),
    (Kind::ADDW, Variant::ADDW),
    (Kind::ADDIW, Variant::ADDIW),
    (Kind::SUBW, Variant::SUBW),
    (Kind::AND, Variant::AND),
    (Kind::ANDI, Variant::ANDI),
    (Kind::OR, Variant::OR),
    (Kind::ORI, Variant::ORI),
    (Kind::XOR, Variant::XOR),
    (Kind::XORI, Variant::XORI),
    (Kind::SLT, Variant::SLT),
    (Kind::SLTI, Variant::SLTI),
    (Kind::SLTU, Variant::SLTU),
    (Kind::SLTIU, Variant::SLTIU),
    (Kind::SLL, Variant::SLL),
    (Kind::SRL, Variant::SRL),
    (Kind::SRA, Variant::SRA),
    (Kind::SLLI, Variant::SLLI),
    (Kind::SRLI, Variant::SRLI),
    (Kind::SRAI, Variant::SRAI),
    (Kind::SLLW, Variant::SLLW),
    (Kind::SRLW, Variant::SRLW),
    (Kind::SRAW, Variant::SRAW),
    (Kind::SLLIW, Variant::SLLIW),
    (Kind::SRLIW, Variant::SRLIW),
    (Kind::SRAIW, Variant::SRAIW),
    (Kind::LB, Variant::LB),
    (Kind::LH, Variant::LH),
    (Kind::LW, Variant::LW),
    (Kind::LD, Variant::LD),
    (Kind::LBU, Variant::LBU),
    (Kind::LHU, Variant::LHU),
    (Kind::LWU, Variant::LWU),
    (Kind::SB, Variant::SB),
    (Kind::SH, Variant::SH),
    (Kind::SW, Variant::SW),
    (Kind::SD, Variant::SD),
    (Kind::BEQ, Variant::BEQ),
    (Kind::BNE, Variant::BNE),
    (Kind::BLT, Variant::BLT),
    (Kind::BGE, Variant::BGE),
    (Kind::BLTU, Variant::BLTU),
    (Kind::BGEU, Variant::BGEU),
    (Kind::JAL, Variant::JAL),
    (Kind::JALR, Variant::JALR),
    (Kind::LUI, Variant::LUI),
    (Kind::AUIPC, Variant::AUIPC),
    (Kind::FENCE, Variant::NOOP),
    (Kind::ECALL, Variant::ECALL),
    (Kind::EBREAK, Variant::EBREAK),
];

/// Shift selector index, independent of register/immediate encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ShiftKind {
    SLL,
    SRL,
    SRA,
    SLLW,
    SRLW,
    SRAW,
}
impl ShiftKind {
    /// All selectors in wire order.
    pub const ALL: [Self; 6] = [
        Self::SLL,
        Self::SRL,
        Self::SRA,
        Self::SLLW,
        Self::SRLW,
        Self::SRAW,
    ];
    /// Position in its selector column.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Load/store selector index; widths are in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AccessKind {
    LB,
    LH,
    LW,
    LD,
    LBU,
    LHU,
    LWU,
    SB,
    SH,
    SW,
    SD,
}
impl AccessKind {
    /// All selectors in wire order.
    pub const ALL: [Self; 11] = [
        Self::LB,
        Self::LH,
        Self::LW,
        Self::LD,
        Self::LBU,
        Self::LHU,
        Self::LWU,
        Self::SB,
        Self::SH,
        Self::SW,
        Self::SD,
    ];
    /// Position in its selector column.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Comparison-key selector index, in declaration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum KeyKind {
    SignedRs2,
    SignedImm,
    UnsignedRs2,
    UnsignedImm,
    Equality,
}
impl KeyKind {
    /// All selectors in wire order.
    pub const ALL: [Self; 5] = [
        Self::SignedRs2,
        Self::SignedImm,
        Self::UnsignedRs2,
        Self::UnsignedImm,
        Self::Equality,
    ];
    /// Position in its selector column.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

impl ShiftKind {
    /// Whether the control rail restricts the shift amount to five bits.
    #[inline]
    pub const fn is_word(self) -> bool {
        matches!(self, Self::SLLW | Self::SRLW | Self::SRAW)
    }
}
impl AccessKind {
    /// The natural access width w in bytes.
    #[inline]
    pub const fn width(self) -> u8 {
        match self {
            Self::LB | Self::LBU | Self::SB => 1,
            Self::LH | Self::LHU | Self::SH => 2,
            Self::LW | Self::LWU | Self::SW => 4,
            Self::LD | Self::SD => 8,
        }
    }
    /// Store selector, used to split Inc between registers and RAM.
    #[inline]
    pub const fn is_store(self) -> bool {
        matches!(self, Self::SB | Self::SH | Self::SW | Self::SD)
    }
}
/// Branch predicate: Equal/NotEqual use KeysDiffer, Less/NotLess use the key ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BranchCondition {
    Equal,
    NotEqual,
    Less,
    NotLess,
}
/// Shift selector and amount source (Rs2Value or Imm).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shift {
    pub kind: ShiftKind,
    pub amount: Source,
}
/// Access width w; kind is absent for LOADw_X0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    pub width: u8,
    pub kind: Option<AccessKind>,
}
