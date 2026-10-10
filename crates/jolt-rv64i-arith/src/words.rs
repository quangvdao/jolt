//! Word evaluation and the 1,024-column witness of one cycle.

use crate::bytecode::BytecodeRow;
use crate::cycle::CycleFacts;
use crate::decode::{eval, load_form, shift_form, store_form, Rails, Source, Sources, BRANCH_FORM};
use crate::layout::{BitsRow, Layout};

/// Values supplied by the surrounding register, RAM and successor arguments.
///
/// These arguments must bind the fetched row to `BytecodeRa`, bind `NextPC` to
/// a valid bytecode address, and establish
/// `RdWriteValue = old_rd XOR ((1 XOR Store) AND Inc)` and
/// `ram_post = RamReadValue XOR (Store AND Inc)`. The local rows do not establish
/// these identities or perform the lookups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BaseWords {
    pub rs1_value: u64,
    pub rs2_value: u64,
    pub rd_write_value: u64,
    pub ram_read_value: u64,
    pub next_pc: u64,
}

impl BaseWords {
    /// Takes the read values, destination post-value and successor from facts.
    /// `ram_pre_value` is immaterial to local rows without an access kind.
    #[inline]
    pub fn from_facts(facts: &CycleFacts) -> Self {
        Self {
            rs1_value: facts.rs1_value,
            rs2_value: facts.rs2_value,
            rd_write_value: facts.rd_post_value,
            ram_read_value: facts.ram_pre_value,
            next_pc: facts.next_pc,
        }
    }
}

/// Decode-form words and their residuals, all represented as bit vectors.
/// Addition of words in these definitions means XOR; a rail product means AND.
/// The six rail words evaluate the adder/AND forms of the selected line;
/// `LeftKey`, `RightKey` evaluate its key-kind forms and `LessThan` is bit 0
/// of its comparison output form. Absent rails evaluate to zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Words {
    pub add_left: u64,
    pub add_right: u64,
    pub add_sum: u64,
    pub and_left: u64,
    pub and_right: u64,
    pub and_out: u64,
    pub left_key: u64,
    pub right_key: u64,
    pub less_than: u64,
    pub carry: u64,
    pub carry_left: u64,
    pub carry_right: u64,
    pub carry_step: u64,
    pub key_diff: u64,
    pub key_diff_above: u64,
    pub left_key_bit: u64,
    pub right_key_bit: u64,
    pub shift_output: u64,
    pub load_output: u64,
    pub store_inc: u64,
    pub branch_output: u64,
    pub rd_expected: u64,
    pub next_pc_expected: u64,
    pub rd_residual: u64,
    pub ram_residual: u64,
    pub next_pc_residual: u64,
    pub control_residual: u64,
}

impl Words {
    /// Evaluates the selected decode line. For invalid rows only
    /// `ControlResidual = 1` is nonzero.
    ///
    /// `Carry = AddLeft XOR AddRight XOR AddSum`,
    /// `CarryLeft = AddRight XOR AddSum`, `CarryRight = AddLeft XOR AddSum`,
    /// and `CarryStep = Carry XOR (Carry >> 1)` below bit 63, with
    /// `CarryStep[63] = Carry[0]`.
    /// With `f0 = full(PosRa_0)` and `f1 = full(PosRa_1)`, position p is hot
    /// when `f0[p % 8] AND f1[p / 8] = 1`; byte offset a is hot when `f0[a] = 1`.
    /// Selectors sum every hot pair, even on non-one-hot inputs.
    /// `KeyDiff = LeftKey XOR RightKey`; on comparison lines, `KeyDiffAbove`
    /// sums `KeyDiff` with bits `0..=p` cleared over hot p, and `LeftKeyBit`
    /// and `RightKeyBit` sum the respective keys' bit p. These are zero off
    /// comparisons. `ShiftOutput`, `LoadOutput`, `StoreInc` sum their positional
    /// forms, and `BranchOutput = ShouldBranch AND (FallThroughPC XOR PCPlusImm)`
    /// on branches. `RdExpected = line.rd_expected XOR ShiftOutput XOR LoadOutput`
    /// and `NextPCExpected = line.next_pc_expected XOR BranchOutput`.
    /// The residuals are `RdWriteValue XOR RdExpected`,
    /// `NextPC XOR NextPCExpected`, and `(Store AND Inc) XOR StoreInc`.
    /// On valid rows `ControlResidual = line.control`, in bits 1..=10.
    #[inline]
    pub fn compute(layout: &Layout, row: &BytecodeRow, base: &BaseWords, bits: &BitsRow) -> Self {
        let Some(variant) = row.variant else {
            return Self {
                control_residual: 1,
                ..Self::default()
            };
        };
        let line = variant.line();
        let src = Sources::new(layout, row, base, bits);
        let mut words = Self::default();
        match &line.rails {
            Rails::None => {}
            Rails::Adder { left, right, sum } => {
                words.add_left = eval(left, &src);
                words.add_right = eval(right, &src);
                words.add_sum = eval(sum, &src);
            }
            Rails::And { left, right, out } => {
                words.and_left = eval(left, &src);
                words.and_right = eval(right, &src);
                words.and_out = eval(out, &src);
            }
            Rails::Compare { keys, less_than } => {
                let (left, right) = keys.keys();
                words.left_key = eval(left, &src);
                words.right_key = eval(right, &src);
                words.less_than = eval(less_than, &src) & 1;
            }
        }
        words.carry = words.add_left ^ words.add_right ^ words.add_sum;
        words.carry_left = words.add_right ^ words.add_sum;
        words.carry_right = words.add_left ^ words.add_sum;
        words.carry_step =
            ((words.carry ^ (words.carry >> 1)) & (u64::MAX >> 1)) | ((words.carry & 1) << 63);
        words.key_diff = words.left_key ^ words.right_key;
        let [low, high] = layout.pos_ra();
        let f0 = low.full(bits);
        let f1 = high.full(bits);
        if line.shift.is_some() || variant.key_kind().is_some() {
            let mut hi = f1;
            while hi != 0 {
                let h = hi.trailing_zeros();
                hi &= hi - 1;
                let mut lo = f0;
                while lo != 0 {
                    let p = (h * 8 + lo.trailing_zeros()) as u8;
                    lo &= lo - 1;
                    if variant.key_kind().is_some() {
                        words.key_diff_above ^=
                            words.key_diff & u64::MAX.checked_shl(u32::from(p) + 1).unwrap_or(0);
                        words.left_key_bit ^= (words.left_key >> p) & 1;
                        words.right_key_bit ^= (words.right_key >> p) & 1;
                    }
                    if let Some(shift) = line.shift {
                        if let Ok(form) = shift_form(shift.kind, p) {
                            words.shift_output ^= eval(form.terms(), &src);
                        }
                    }
                }
            }
        }
        if let Some(access) = line.access {
            if let Some(kind) = access.kind {
                let mut lo = f0;
                while lo != 0 {
                    let a = lo.trailing_zeros() as u8;
                    lo &= lo - 1;
                    if kind.is_store() {
                        if let Ok(form) = store_form(kind, a) {
                            words.store_inc ^= eval(form.terms(), &src);
                        }
                    } else if let Ok(form) = load_form(kind, a) {
                        words.load_output ^= eval(form.terms(), &src);
                    }
                }
            }
        }
        if line.branch.is_some() && src.get(Source::ShouldBranch) != 0 {
            words.branch_output = eval(BRANCH_FORM, &src);
        }
        words.rd_expected = eval(line.rd_expected, &src) ^ words.shift_output ^ words.load_output;
        words.next_pc_expected = eval(line.next_pc_expected, &src) ^ words.branch_output;
        words.rd_residual = base.rd_write_value ^ words.rd_expected;
        words.next_pc_residual = base.next_pc ^ words.next_pc_expected;
        words.ram_residual = if variant.is_store() {
            layout.inc(bits)
        } else {
            0
        } ^ words.store_inc;
        words.control_residual = eval(line.control, &src);
        words
    }
}

/// Number of bit columns in one R1CS witness.
pub const WITNESS_COLUMNS: usize = 1024;

/// The 64-column aligned blocks of the witness, in wire order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Lane {
    Small,
    CarryLeft,
    CarryRight,
    CarryStep,
    AndLeft,
    AndRight,
    AndOut,
    KeyDiff,
    KeyDiffAbove,
    RdResidual,
    RamResidual,
    NextPCResidual,
    Bits0,
    Bits1,
    Bits2,
    Bits3,
}

/// Non-block witness positions; committed column `c` is at `BITS + c`.
pub mod column {
    pub const ONE: usize = 0;
    pub const LEFT_KEY_BIT: usize = 1;
    pub const RIGHT_KEY_BIT: usize = 2;
    pub const LESS_THAN: usize = 3;
    pub const CONTROL_RESIDUAL: usize = 16;
    pub const BITS: usize = 768;
}

/// Column `c` is bit `c % 64` of word `c / 64`; the last four words are `Bits`.
/// Canonical assembly sets `ONE`, clears unused small columns, and preserves
/// all committed bits including spare columns.
///
/// | Columns | Content |
/// |---|---|
/// | 0; 1, 2, 3 | ONE; LeftKeyBit, RightKeyBit, LessThan |
/// | 16..27 | ControlResidual[0..11] |
/// | 64..256 | CarryLeft, CarryRight, CarryStep, 64 bits each |
/// | 256..448 | AndLeft, AndRight, AndOut, 64 bits each |
/// | 448..576 | KeyDiff, KeyDiffAbove, 64 bits each |
/// | 576..768 | RdResidual, RamResidual, NextPCResidual, 64 bits each |
/// | 768..1024 | Bits0, Bits1, Bits2, Bits3, unchanged |
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WitnessRow(pub [u64; 16]);

impl WitnessRow {
    /// Evaluates the decode words and assembles their canonical witness.
    #[inline]
    pub fn compute(layout: &Layout, row: &BytecodeRow, base: &BaseWords, bits: &BitsRow) -> Self {
        Self::assemble(&Words::compute(layout, row, base, bits), bits)
    }

    /// Embeds `ONE`, the three key bits, `ControlResidual[0..11]` at column 16,
    /// eleven aligned virtual words and the four committed words.
    #[inline]
    pub fn assemble(words: &Words, bits: &BitsRow) -> Self {
        let [b0, b1, b2, b3] = *bits;
        Self([
            1 | ((words.left_key_bit & 1) << 1)
                | ((words.right_key_bit & 1) << 2)
                | ((words.less_than & 1) << 3)
                | ((words.control_residual & 0x7ff) << 16),
            words.carry_left,
            words.carry_right,
            words.carry_step,
            words.and_left,
            words.and_right,
            words.and_out,
            words.key_diff,
            words.key_diff_above,
            words.rd_residual,
            words.ram_residual,
            words.next_pc_residual,
            b0,
            b1,
            b2,
            b3,
        ])
    }

    /// Reads all 64 bits of an aligned block.
    #[inline]
    pub fn lane(&self, lane: Lane) -> u64 {
        self.0.get(lane as usize).copied().unwrap_or(0)
    }

    /// Reads column `c`, or `None` when `c >= 1024`.
    #[inline]
    pub fn bit(&self, column: usize) -> Option<bool> {
        self.0
            .get(column / 64)
            .map(|word| (word >> (column % 64)) & 1 != 0)
    }
}
