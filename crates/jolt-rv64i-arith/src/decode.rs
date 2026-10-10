//! Decode rows as XOR-linear maps on words. A term's wires define its bit map.

use crate::bytecode::BytecodeRow;
use crate::layout::{bit, BitsRow, Layout};
use crate::variant::{Access, AccessKind, BranchCondition, KeyKind, Shift, ShiftKind, Variant};
use crate::words::BaseWords;
use thiserror::Error as ThisError;

/// Named input word; the final four sources hold a bit at position zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Source {
    Rs1Value,
    Rs2Value,
    RdWriteValue,
    Imm,
    FallThroughPC,
    PCPlusImm,
    PC,
    NextPC,
    Inc,
    RamReadValue,
    RamAddress,
    Pos,
    KeysDiffer,
    ShouldBranch,
    JalrLowBit,
    One,
}
impl Source {
    /// Source words in their declaration order.
    pub const ALL: [Self; 16] = [
        Self::Rs1Value,
        Self::Rs2Value,
        Self::RdWriteValue,
        Self::Imm,
        Self::FallThroughPC,
        Self::PCPlusImm,
        Self::PC,
        Self::NextPC,
        Self::Inc,
        Self::RamReadValue,
        Self::RamAddress,
        Self::Pos,
        Self::KeysDiffer,
        Self::ShouldBranch,
        Self::JalrLowBit,
        Self::One,
    ];
}
/// Stack values of the sixteen named source words.
#[derive(Clone, Copy, Debug)]
pub struct Sources([u64; 16]);
impl Sources {
    /// RamAddress = (RamIndex << 3) | (Pos & 7); flag sources are the committed bits.
    #[inline]
    pub fn new(layout: &Layout, row: &BytecodeRow, base: &BaseWords, bits: &BitsRow) -> Self {
        let pos = u64::from(layout.pos(bits));
        Self([
            base.rs1_value,
            base.rs2_value,
            base.rd_write_value,
            row.imm,
            row.fall_through_pc,
            row.pc_plus_imm,
            row.pc,
            base.next_pc,
            layout.inc(bits),
            base.ram_read_value,
            (layout.ram_index(bits) << 3) | (pos & 7),
            pos,
            u64::from(bit(bits, layout.keys_differ())),
            u64::from(bit(bits, layout.should_branch())),
            u64::from(bit(bits, layout.jalr_low_bit())),
            1,
        ])
    }
    /// The named source's integer bit pattern.
    #[inline]
    pub fn get(&self, source: Source) -> u64 {
        self.0.get(source as usize).copied().unwrap_or(0)
    }
}
/// Invalid copy/fill descriptor; both input and output positions must fit a word.
#[derive(Clone, Debug, PartialEq, Eq, ThisError)]
pub enum TermError {
    #[error("term length {len} is outside 1..=64")]
    LengthOutOfRange { len: u8 },
    #[error("term source range from {from} with length {len} and fill {fill} leaves the word")]
    SourceRangeOutOfRange { from: u8, len: u8, fill: bool },
    #[error("term output range to {to} with length {len} leaves the word")]
    OutputRangeOutOfRange { to: u8, len: u8 },
}
/// XOR source[from+i] into output[to+i], or source[from] into each output bit for fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Term {
    source: Source,
    from: u8,
    to: u8,
    len: u8,
    fill: bool,
}
impl Term {
    /// Checks 1 <= len <= 64, to+len <= 64, and from+len <= 64 for copies.
    /// A fill reads only from, which must be below 64.
    pub const fn new(
        source: Source,
        from: u8,
        to: u8,
        len: u8,
        fill: bool,
    ) -> Result<Self, TermError> {
        if len == 0 || len > 64 {
            return Err(TermError::LengthOutOfRange { len });
        }
        if from >= 64 || (!fill && from as u16 + len as u16 > 64) {
            return Err(TermError::SourceRangeOutOfRange { from, len, fill });
        }
        if to as u16 + len as u16 > 64 {
            return Err(TermError::OutputRangeOutOfRange { to, len });
        }
        Ok(Self::constant(source, from, to, len, fill))
    }
    const fn constant(source: Source, from: u8, to: u8, len: u8, fill: bool) -> Self {
        Self {
            source,
            from,
            to,
            len,
            fill,
        }
    }
    /// Named source word.
    #[inline]
    pub const fn source(self) -> Source {
        self.source
    }
    /// First source bit, or the repeated bit for a fill.
    #[inline]
    pub const fn from(self) -> u8 {
        self.from
    }
    /// First output bit.
    #[inline]
    pub const fn to(self) -> u8 {
        self.to
    }
    /// Number of output bits.
    #[inline]
    pub const fn length(self) -> u8 {
        self.len
    }
    /// Whether one source bit is repeated.
    #[inline]
    pub const fn fill(self) -> bool {
        self.fill
    }
    /// The word map ((source >> from) & mask(len)) << to; fills repeat source[from].
    #[inline]
    pub fn eval(self, sources: &Sources) -> u64 {
        let value = sources.get(self.source) >> self.from;
        let mask = u64::MAX >> (64 - self.len);
        if self.fill {
            if value & 1 == 0 {
                0
            } else {
                mask << self.to
            }
        } else {
            (value & mask) << self.to
        }
    }
    /// Lists (output bit, source bit) pairs defining this term.
    pub fn wires(self) -> impl Iterator<Item = Wire> {
        (0..self.len).map(move |i| Wire {
            out: self.to + i,
            source: self.source,
            bit: self.from + if self.fill { 0 } else { i },
        })
    }
}
/// One contribution output[out] += source[bit].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wire {
    pub out: u8,
    pub source: Source,
    pub bit: u8,
}
/// Static XOR sum of copy/fill terms.
pub type Form = &'static [Term];
/// Adder (`left ⊞ right = sum` modulo 2^64), AND (`left & right = out`),
/// or comparison rails whose ordered keys select `less_than`.
#[derive(Clone, Copy, Debug)]
pub enum Rails {
    None,
    Adder { left: Form, right: Form, sum: Form },
    And { left: Form, right: Form, out: Form },
    Compare { keys: KeyKind, less_than: Form },
}
/// Decode-table row. The three expected/control forms are XOR sums;
/// control places equality-branch, alignment, and shift residuals in bits 1..=10.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub rails: Rails,
    pub rd_expected: Form,
    pub next_pc_expected: Form,
    pub control: Form,
    pub shift: Option<Shift>,
    pub access: Option<Access>,
    pub branch: Option<BranchCondition>,
}
/// At most two terms, stored inline, for a positional shift or memory output.
#[derive(Clone, Copy, Debug)]
pub struct ShortForm {
    terms: [Term; 2],
    len: u8,
}
impl ShortForm {
    const EMPTY: Self = Self {
        terms: [Term::constant(Source::One, 0, 0, 1, false); 2],
        len: 0,
    };
    fn one(term: Term) -> Self {
        Self {
            terms: [term, term],
            len: 1,
        }
    }
    fn two(a: Term, b: Term) -> Self {
        Self {
            terms: [a, b],
            len: 2,
        }
    }
    /// The active XOR terms; padding is never exposed.
    #[inline]
    pub fn terms(&self) -> &[Term] {
        self.terms.get(..usize::from(self.len)).unwrap_or(&[])
    }
}
/// Invalid positional selector for a short form.
#[derive(Clone, Debug, PartialEq, Eq, ThisError)]
pub enum FormError {
    #[error("shift position {pos} is outside 0..64")]
    ShiftPositionOutOfRange { pos: u8 },
    #[error("access byte offset {offset} is outside 0..8")]
    AccessOffsetOutOfRange { offset: u8 },
}
/// ShiftOutput over Rs1Value at Pos=p. W outputs sign-extend bit 31;
/// at p>=32 SLLW/SRLW are zero and SRAW repeats bit 31.
#[inline]
pub fn shift_form(kind: ShiftKind, pos: u8) -> Result<ShortForm, FormError> {
    if pos >= 64 {
        return Err(FormError::ShiftPositionOutOfRange { pos });
    }
    let copy = |from, to, len| Term::constant(Source::Rs1Value, from, to, len, false);
    let fill = |from, to, len| Term::constant(Source::Rs1Value, from, to, len, true);
    Ok(match kind {
        ShiftKind::SLL => ShortForm::one(copy(0, pos, 64 - pos)),
        ShiftKind::SRL => ShortForm::one(copy(pos, 0, 64 - pos)),
        ShiftKind::SRA => {
            let first = copy(pos, 0, 64 - pos);
            if pos == 0 {
                ShortForm::one(first)
            } else {
                ShortForm::two(first, fill(63, 64 - pos, pos))
            }
        }
        ShiftKind::SLLW => {
            if pos >= 32 {
                ShortForm::EMPTY
            } else {
                ShortForm::two(copy(0, pos, 32 - pos), fill(31 - pos, 32, 32))
            }
        }
        ShiftKind::SRLW => {
            if pos >= 32 {
                ShortForm::EMPTY
            } else if pos == 0 {
                ShortForm::two(copy(0, 0, 32), fill(31, 32, 32))
            } else {
                ShortForm::one(copy(pos, 0, 32 - pos))
            }
        }
        ShiftKind::SRAW => {
            if pos >= 32 {
                ShortForm::one(fill(31, 0, 64))
            } else {
                ShortForm::two(copy(pos, 0, 32 - pos), fill(31, 32 - pos, 32 + pos))
            }
        }
    })
}
/// LoadOutput copies min(8w,64-8a) bits from byte offset a. Signed subword
/// loads repeat their sign bit when the full subword lies inside the word; stores give zero.
#[inline]
pub fn load_form(kind: AccessKind, offset: u8) -> Result<ShortForm, FormError> {
    if offset >= 8 {
        return Err(FormError::AccessOffsetOutOfRange { offset });
    }
    if kind.is_store() {
        return Ok(ShortForm::EMPTY);
    }
    let from = offset * 8;
    let width = kind.width() * 8;
    let len = width.min(64 - from);
    let copy = Term::constant(Source::RamReadValue, from, 0, len, false);
    Ok(
        if matches!(kind, AccessKind::LB | AccessKind::LH | AccessKind::LW) && from + width <= 64 {
            ShortForm::two(
                copy,
                Term::constant(
                    Source::RamReadValue,
                    from + width - 1,
                    width,
                    64 - width,
                    true,
                ),
            )
        } else {
            ShortForm::one(copy)
        },
    )
}
/// StoreInc = the old RAM subword XOR Rs2Value's low subword at byte offset a;
/// bytes beyond bit 63 are absent. Loads give zero.
#[inline]
pub fn store_form(kind: AccessKind, offset: u8) -> Result<ShortForm, FormError> {
    if offset >= 8 {
        return Err(FormError::AccessOffsetOutOfRange { offset });
    }
    if !kind.is_store() {
        return Ok(ShortForm::EMPTY);
    }
    let to = offset * 8;
    let len = (kind.width() * 8).min(64 - to);
    Ok(ShortForm::two(
        Term::constant(Source::RamReadValue, to, to, len, false),
        Term::constant(Source::Rs2Value, 0, to, len, false),
    ))
}
/// XOR sum of term maps, computed without allocation.
#[inline]
pub fn eval(form: &[Term], sources: &Sources) -> u64 {
    form.iter()
        .fold(0, |value, term| value ^ term.eval(sources))
}

const RS1: Form = &[Term::constant(Source::Rs1Value, 0, 0, 64, false)];
const RS2: Form = &[Term::constant(Source::Rs2Value, 0, 0, 64, false)];
const IMM: Form = &[Term::constant(Source::Imm, 0, 0, 64, false)];
const RD: Form = &[Term::constant(Source::RdWriteValue, 0, 0, 64, false)];
const FT: Form = &[Term::constant(Source::FallThroughPC, 0, 0, 64, false)];
const PC_IMM: Form = &[Term::constant(Source::PCPlusImm, 0, 0, 64, false)];
const PC: Form = &[Term::constant(Source::PC, 0, 0, 64, false)];
const NEXT_PC: Form = &[Term::constant(Source::NextPC, 0, 0, 64, false)];
const RAM_ADDRESS: Form = &[Term::constant(Source::RamAddress, 0, 0, 64, false)];
const RS1_W: Form = &[Term::constant(Source::Rs1Value, 0, 32, 32, false)];
const RS2_W: Form = &[Term::constant(Source::Rs2Value, 0, 32, 32, false)];
const IMM_W: Form = &[Term::constant(Source::Imm, 0, 32, 32, false)];
const RD_W: Form = &[Term::constant(Source::RdWriteValue, 0, 32, 32, false)];
const RD_SEXT: Form = &[
    Term::constant(Source::RdWriteValue, 0, 0, 32, false),
    Term::constant(Source::RdWriteValue, 31, 32, 32, true),
];
const RD_BIT: Form = &[Term::constant(Source::RdWriteValue, 0, 0, 1, false)];
const SIGNED_RS1: Form = &[
    Term::constant(Source::Rs1Value, 0, 0, 64, false),
    Term::constant(Source::One, 0, 63, 1, false),
];
const SIGNED_RS2: Form = &[
    Term::constant(Source::Rs2Value, 0, 0, 64, false),
    Term::constant(Source::One, 0, 63, 1, false),
];
const SIGNED_IMM: Form = &[
    Term::constant(Source::Imm, 0, 0, 64, false),
    Term::constant(Source::One, 0, 63, 1, false),
];
const EQUALITY: Form = &[
    Term::constant(Source::Rs1Value, 0, 0, 64, false),
    Term::constant(Source::Rs2Value, 0, 0, 64, false),
];
const JALR_SUM: Form = &[
    Term::constant(Source::NextPC, 0, 0, 64, false),
    Term::constant(Source::JalrLowBit, 0, 0, 1, false),
];
/// Taken-branch correction = FallThroughPC XOR PCPlusImm.
pub const BRANCH_FORM: Form = &[
    Term::constant(Source::FallThroughPC, 0, 0, 64, false),
    Term::constant(Source::PCPlusImm, 0, 0, 64, false),
];
impl KeyKind {
    /// Signed keys flip bit 63; unsigned keys copy operands; Equality uses
    /// (Rs1Value XOR Rs2Value, 0).
    #[inline]
    pub const fn keys(self) -> (Form, Form) {
        match self {
            Self::SignedRs2 => (SIGNED_RS1, SIGNED_RS2),
            Self::SignedImm => (SIGNED_RS1, SIGNED_IMM),
            Self::UnsignedRs2 => (RS1, RS2),
            Self::UnsignedImm => (RS1, IMM),
            Self::Equality => (EQUALITY, &[]),
        }
    }
}

pub(crate) const NOOP_LINE: Line = Line {
    rails: Rails::None,
    rd_expected: &[],
    next_pc_expected: FT,
    control: &[],
    shift: None,
    access: None,
    branch: None,
};

pub(crate) static DECODE_TABLE: [Line; Variant::COUNT] = [
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: RS2,
            sum: RD,
        },
        rd_expected: RD,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RD,
        },
        rd_expected: RD,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS2,
            right: RD,
            sum: RS1,
        },
        rd_expected: RD,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1_W,
            right: RS2_W,
            sum: RD_W,
        },
        rd_expected: RD_SEXT,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1_W,
            right: IMM_W,
            sum: RD_W,
        },
        rd_expected: RD_SEXT,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS2_W,
            right: RD_W,
            sum: RS1_W,
        },
        rd_expected: RD_SEXT,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::And {
            left: RS1,
            right: RS2,
            out: RD,
        },
        rd_expected: RD,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::And {
            left: RS1,
            right: IMM,
            out: RD,
        },
        rd_expected: RD,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::And {
            left: RS1,
            right: RS2,
            out: &[
                Term::constant(Source::RdWriteValue, 0, 0, 64, false),
                Term::constant(Source::Rs1Value, 0, 0, 64, false),
                Term::constant(Source::Rs2Value, 0, 0, 64, false),
            ],
        },
        rd_expected: RD,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::And {
            left: RS1,
            right: IMM,
            out: &[
                Term::constant(Source::RdWriteValue, 0, 0, 64, false),
                Term::constant(Source::Rs1Value, 0, 0, 64, false),
                Term::constant(Source::Imm, 0, 0, 64, false),
            ],
        },
        rd_expected: RD,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[
            Term::constant(Source::Rs1Value, 0, 0, 64, false),
            Term::constant(Source::Rs2Value, 0, 0, 64, false),
        ],
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[
            Term::constant(Source::Rs1Value, 0, 0, 64, false),
            Term::constant(Source::Imm, 0, 0, 64, false),
        ],
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::Compare {
            keys: KeyKind::SignedRs2,
            less_than: RD_BIT,
        },
        rd_expected: RD_BIT,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::Compare {
            keys: KeyKind::SignedImm,
            less_than: RD_BIT,
        },
        rd_expected: RD_BIT,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::Compare {
            keys: KeyKind::UnsignedRs2,
            less_than: RD_BIT,
        },
        rd_expected: RD_BIT,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::Compare {
            keys: KeyKind::UnsignedImm,
            less_than: RD_BIT,
        },
        rd_expected: RD_BIT,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[
            Term::constant(Source::Pos, 0, 5, 6, false),
            Term::constant(Source::Rs2Value, 0, 5, 6, false),
        ],
        shift: Some(Shift {
            kind: ShiftKind::SLL,
            amount: Source::Rs2Value,
        }),
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[
            Term::constant(Source::Pos, 0, 5, 6, false),
            Term::constant(Source::Rs2Value, 0, 5, 6, false),
        ],
        shift: Some(Shift {
            kind: ShiftKind::SRL,
            amount: Source::Rs2Value,
        }),
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[
            Term::constant(Source::Pos, 0, 5, 6, false),
            Term::constant(Source::Rs2Value, 0, 5, 6, false),
        ],
        shift: Some(Shift {
            kind: ShiftKind::SRA,
            amount: Source::Rs2Value,
        }),
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[
            Term::constant(Source::Pos, 0, 5, 6, false),
            Term::constant(Source::Imm, 0, 5, 6, false),
        ],
        shift: Some(Shift {
            kind: ShiftKind::SLL,
            amount: Source::Imm,
        }),
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[
            Term::constant(Source::Pos, 0, 5, 6, false),
            Term::constant(Source::Imm, 0, 5, 6, false),
        ],
        shift: Some(Shift {
            kind: ShiftKind::SRL,
            amount: Source::Imm,
        }),
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[
            Term::constant(Source::Pos, 0, 5, 6, false),
            Term::constant(Source::Imm, 0, 5, 6, false),
        ],
        shift: Some(Shift {
            kind: ShiftKind::SRA,
            amount: Source::Imm,
        }),
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[
            Term::constant(Source::Pos, 0, 5, 6, false),
            Term::constant(Source::Rs2Value, 0, 5, 5, false),
        ],
        shift: Some(Shift {
            kind: ShiftKind::SLLW,
            amount: Source::Rs2Value,
        }),
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[
            Term::constant(Source::Pos, 0, 5, 6, false),
            Term::constant(Source::Rs2Value, 0, 5, 5, false),
        ],
        shift: Some(Shift {
            kind: ShiftKind::SRLW,
            amount: Source::Rs2Value,
        }),
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[
            Term::constant(Source::Pos, 0, 5, 6, false),
            Term::constant(Source::Rs2Value, 0, 5, 5, false),
        ],
        shift: Some(Shift {
            kind: ShiftKind::SRAW,
            amount: Source::Rs2Value,
        }),
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[
            Term::constant(Source::Pos, 0, 5, 6, false),
            Term::constant(Source::Imm, 0, 5, 5, false),
        ],
        shift: Some(Shift {
            kind: ShiftKind::SLLW,
            amount: Source::Imm,
        }),
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[
            Term::constant(Source::Pos, 0, 5, 6, false),
            Term::constant(Source::Imm, 0, 5, 5, false),
        ],
        shift: Some(Shift {
            kind: ShiftKind::SRLW,
            amount: Source::Imm,
        }),
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[
            Term::constant(Source::Pos, 0, 5, 6, false),
            Term::constant(Source::Imm, 0, 5, 5, false),
        ],
        shift: Some(Shift {
            kind: ShiftKind::SRAW,
            amount: Source::Imm,
        }),
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: Some(Access {
            width: AccessKind::LB.width(),
            kind: Some(AccessKind::LB),
        }),
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[Term::constant(Source::Pos, 0, 2, 1, false)],
        shift: None,
        access: Some(Access {
            width: AccessKind::LH.width(),
            kind: Some(AccessKind::LH),
        }),
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[Term::constant(Source::Pos, 0, 2, 2, false)],
        shift: None,
        access: Some(Access {
            width: AccessKind::LW.width(),
            kind: Some(AccessKind::LW),
        }),
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[Term::constant(Source::Pos, 0, 2, 3, false)],
        shift: None,
        access: Some(Access {
            width: AccessKind::LD.width(),
            kind: Some(AccessKind::LD),
        }),
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: Some(Access {
            width: AccessKind::LBU.width(),
            kind: Some(AccessKind::LBU),
        }),
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[Term::constant(Source::Pos, 0, 2, 1, false)],
        shift: None,
        access: Some(Access {
            width: AccessKind::LHU.width(),
            kind: Some(AccessKind::LHU),
        }),
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[Term::constant(Source::Pos, 0, 2, 2, false)],
        shift: None,
        access: Some(Access {
            width: AccessKind::LWU.width(),
            kind: Some(AccessKind::LWU),
        }),
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: Some(Access {
            width: AccessKind::SB.width(),
            kind: Some(AccessKind::SB),
        }),
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[Term::constant(Source::Pos, 0, 2, 1, false)],
        shift: None,
        access: Some(Access {
            width: AccessKind::SH.width(),
            kind: Some(AccessKind::SH),
        }),
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[Term::constant(Source::Pos, 0, 2, 2, false)],
        shift: None,
        access: Some(Access {
            width: AccessKind::SW.width(),
            kind: Some(AccessKind::SW),
        }),
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[Term::constant(Source::Pos, 0, 2, 3, false)],
        shift: None,
        access: Some(Access {
            width: AccessKind::SD.width(),
            kind: Some(AccessKind::SD),
        }),
        branch: None,
    },
    Line {
        rails: Rails::Compare {
            keys: KeyKind::Equality,
            less_than: &[],
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[
            Term::constant(Source::ShouldBranch, 0, 1, 1, false),
            Term::constant(Source::One, 0, 1, 1, false),
            Term::constant(Source::KeysDiffer, 0, 1, 1, false),
        ],
        shift: None,
        access: None,
        branch: Some(BranchCondition::Equal),
    },
    Line {
        rails: Rails::Compare {
            keys: KeyKind::Equality,
            less_than: &[],
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[
            Term::constant(Source::ShouldBranch, 0, 1, 1, false),
            Term::constant(Source::KeysDiffer, 0, 1, 1, false),
        ],
        shift: None,
        access: None,
        branch: Some(BranchCondition::NotEqual),
    },
    Line {
        rails: Rails::Compare {
            keys: KeyKind::SignedRs2,
            less_than: &[Term::constant(Source::ShouldBranch, 0, 0, 1, false)],
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: Some(BranchCondition::Less),
    },
    Line {
        rails: Rails::Compare {
            keys: KeyKind::SignedRs2,
            less_than: &[
                Term::constant(Source::ShouldBranch, 0, 0, 1, false),
                Term::constant(Source::One, 0, 0, 1, false),
            ],
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: Some(BranchCondition::NotLess),
    },
    Line {
        rails: Rails::Compare {
            keys: KeyKind::UnsignedRs2,
            less_than: &[Term::constant(Source::ShouldBranch, 0, 0, 1, false)],
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: Some(BranchCondition::Less),
    },
    Line {
        rails: Rails::Compare {
            keys: KeyKind::UnsignedRs2,
            less_than: &[
                Term::constant(Source::ShouldBranch, 0, 0, 1, false),
                Term::constant(Source::One, 0, 0, 1, false),
            ],
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: Some(BranchCondition::NotLess),
    },
    Line {
        rails: Rails::None,
        rd_expected: FT,
        next_pc_expected: PC_IMM,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: JALR_SUM,
        },
        rd_expected: FT,
        next_pc_expected: NEXT_PC,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: IMM,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: PC_IMM,
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    NOOP_LINE,
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: PC,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: PC,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::None,
        rd_expected: &[],
        next_pc_expected: PC_IMM,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: JALR_SUM,
        },
        rd_expected: &[],
        next_pc_expected: NEXT_PC,
        control: &[],
        shift: None,
        access: None,
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[],
        shift: None,
        access: Some(Access {
            width: 1,
            kind: None,
        }),
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[Term::constant(Source::Pos, 0, 2, 1, false)],
        shift: None,
        access: Some(Access {
            width: 2,
            kind: None,
        }),
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[Term::constant(Source::Pos, 0, 2, 2, false)],
        shift: None,
        access: Some(Access {
            width: 4,
            kind: None,
        }),
        branch: None,
    },
    Line {
        rails: Rails::Adder {
            left: RS1,
            right: IMM,
            sum: RAM_ADDRESS,
        },
        rd_expected: &[],
        next_pc_expected: FT,
        control: &[Term::constant(Source::Pos, 0, 2, 3, false)],
        shift: None,
        access: Some(Access {
            width: 8,
            kind: None,
        }),
        branch: None,
    },
];

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests assert checked descriptor and selector contracts"
)]
mod tests {
    use super::*;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn assert_term(term: Term, rng: &mut ChaCha8Rng) {
        assert_eq!(
            Term::new(
                term.source(),
                term.from(),
                term.to(),
                term.length(),
                term.fill()
            ),
            Ok(term)
        );
        for _ in 0..32 {
            let mut values = [0; 16];
            for value in &mut values {
                *value = rng.next_u64();
            }
            let sources = Sources(values);
            let mut expected = 0;
            for wire in term.wires() {
                expected ^= ((sources.get(wire.source) >> wire.bit) & 1) << wire.out;
            }
            assert_eq!(term.eval(&sources), expected);
        }
    }

    #[test]
    fn every_table_term_matches_its_wire_definition_and_checked_domain() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x00de_c0de);
        for variant in Variant::ALL {
            let line = variant.line();
            let forms: &[Form] = match &line.rails {
                Rails::None => &[],
                Rails::Adder { left, right, sum } => &[*left, *right, *sum],
                Rails::And { left, right, out } => &[*left, *right, *out],
                Rails::Compare { less_than, .. } => &[*less_than],
            };
            for form in
                forms
                    .iter()
                    .copied()
                    .chain([line.rd_expected, line.next_pc_expected, line.control])
            {
                for term in form {
                    assert_term(*term, &mut rng);
                }
            }
        }
        for kind in KeyKind::ALL {
            let (left, right) = kind.keys();
            for term in left.iter().chain(right) {
                assert_term(*term, &mut rng);
            }
        }
        for term in BRANCH_FORM {
            assert_term(*term, &mut rng);
        }
        for kind in ShiftKind::ALL {
            for pos in 0..64 {
                for term in shift_form(kind, pos).unwrap().terms() {
                    assert_term(*term, &mut rng);
                }
            }
        }
        for kind in AccessKind::ALL {
            for offset in 0..8 {
                for short in [
                    load_form(kind, offset).unwrap(),
                    store_form(kind, offset).unwrap(),
                ] {
                    for term in short.terms() {
                        assert_term(*term, &mut rng);
                    }
                }
            }
        }
    }

    #[test]
    fn descriptor_errors_and_positional_domains() {
        assert_eq!(
            Term::new(Source::One, 0, 0, 0, false),
            Err(TermError::LengthOutOfRange { len: 0 })
        );
        assert_eq!(
            Term::new(Source::One, 0, 0, 65, false),
            Err(TermError::LengthOutOfRange { len: 65 })
        );
        assert_eq!(
            Term::new(Source::One, 64, 0, 1, true),
            Err(TermError::SourceRangeOutOfRange {
                from: 64,
                len: 1,
                fill: true
            })
        );
        assert_eq!(
            Term::new(Source::One, 63, 0, 2, false),
            Err(TermError::SourceRangeOutOfRange {
                from: 63,
                len: 2,
                fill: false
            })
        );
        assert_eq!(
            Term::new(Source::One, 0, 63, 2, false),
            Err(TermError::OutputRangeOutOfRange { to: 63, len: 2 })
        );
        assert!(Term::new(Source::One, 63, 0, 64, true).is_ok());
        assert!(Term::new(Source::One, 0, 0, 64, false).is_ok());
        for pos in 64..=u8::MAX {
            for kind in ShiftKind::ALL {
                assert!(
                    matches!(shift_form(kind, pos), Err(FormError::ShiftPositionOutOfRange { pos: found }) if found == pos)
                );
            }
        }
        for offset in 8..=u8::MAX {
            for kind in AccessKind::ALL {
                assert!(
                    matches!(load_form(kind, offset), Err(FormError::AccessOffsetOutOfRange { offset: found }) if found == offset)
                );
                assert!(
                    matches!(store_form(kind, offset), Err(FormError::AccessOffsetOutOfRange { offset: found }) if found == offset)
                );
            }
        }
    }

    #[test]
    fn decode_rail_inventory_and_fixed_kind_indices() {
        let mut counts = [0; 4];
        for variant in Variant::ALL {
            match variant.line().rails {
                Rails::Adder { .. } => counts[0] += 1,
                Rails::And { .. } => counts[1] += 1,
                Rails::Compare { .. } => counts[2] += 1,
                Rails::None => counts[3] += 1,
            }
        }
        assert_eq!(counts, [23, 4, 10, 21]);
        assert!(ShiftKind::ALL.into_iter().map(ShiftKind::index).eq(0..6));
        assert!(AccessKind::ALL.into_iter().map(AccessKind::index).eq(0..11));
        assert!(KeyKind::ALL.into_iter().map(KeyKind::index).eq(0..5));
        assert_eq!(Variant::ADD.index(), 0);
        assert_eq!(Variant::SLL.index(), 16);
        assert_eq!(Variant::LB.index(), 28);
        assert_eq!(Variant::JAL.index(), 45);
        assert_eq!(Variant::LOAD8_X0.index(), 57);
    }

    #[test]
    fn positional_outputs_pin_sign_extension_and_word_boundaries() {
        let mut sources = Sources([0; 16]);
        if let Some(value) = sources.0.get_mut(Source::Rs1Value as usize) {
            *value = 0x8000_0001;
        }
        assert_eq!(
            eval(shift_form(ShiftKind::SLLW, 0).unwrap().terms(), &sources),
            0xffff_ffff_8000_0001
        );
        assert_eq!(
            eval(shift_form(ShiftKind::SRLW, 1).unwrap().terms(), &sources),
            0x4000_0000
        );
        assert_eq!(
            eval(shift_form(ShiftKind::SRAW, 1).unwrap().terms(), &sources),
            0xffff_ffff_c000_0000
        );
        assert_eq!(
            eval(shift_form(ShiftKind::SRAW, 32).unwrap().terms(), &sources),
            u64::MAX
        );
        if let Some(value) = sources.0.get_mut(Source::RamReadValue as usize) {
            *value = 0x80fe_3412_7856_ffff;
        }
        assert_eq!(
            eval(load_form(AccessKind::LB, 6).unwrap().terms(), &sources),
            0xffff_ffff_ffff_fffe
        );
        assert_eq!(
            eval(load_form(AccessKind::LH, 7).unwrap().terms(), &sources),
            0x80
        );
        if let Some(value) = sources.0.get_mut(Source::Rs2Value as usize) {
            *value = 0xab;
        }
        assert_eq!(
            eval(store_form(AccessKind::SB, 6).unwrap().terms(), &sources),
            0x0055_0000_0000_0000
        );
    }
}
