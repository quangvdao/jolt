//! Public bytecode rows contain a validity bit, a 64-bit variant selector,
//! four 64-bit words, and three 32-bit register selectors: 417 bits.

#![expect(
    non_snake_case,
    reason = "layout sizing names follow protocol notation"
)]

use std::collections::HashMap;

use jolt_riscv::{SourceExtension, SourceInstruction, SourceInstructionKind as Kind};
use thiserror::Error;

use crate::{Layout, Variant};

/// Sum of the nine bytecode column widths.
pub const BYTECODE_ROW_BITS: usize = 417;
/// The nine columns used by bytecode read checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BytecodeColumn {
    Valid,
    Variant,
    PC,
    Imm,
    FallThroughPC,
    PCPlusImm,
    Rs1Ra,
    Rs2Ra,
    RdWa,
}
impl BytecodeColumn {
    /// Columns in wire order.
    pub const ALL: [Self; 9] = [
        Self::Valid,
        Self::Variant,
        Self::PC,
        Self::Imm,
        Self::FallThroughPC,
        Self::PCPlusImm,
        Self::Rs1Ra,
        Self::Rs2Ra,
        Self::RdWa,
    ];
    /// Width: validity 1, variant and words 64, register selectors 32.
    pub const fn width(self) -> usize {
        match self {
            Self::Valid => 1,
            Self::Variant | Self::PC | Self::Imm | Self::FallThroughPC | Self::PCPlusImm => 64,
            Self::Rs1Ra | Self::Rs2Ra | Self::RdWa => 32,
        }
    }
}

/// A malformed decoded RV64I source row.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BytecodeRowError {
    #[error("register {register} is outside x0..x31")]
    RegisterOutOfRange { register: u8 },
    #[error("{kind:?} is missing required operand {operand}")]
    MissingOperand { kind: Kind, operand: &'static str },
    #[error("{kind:?} immediate {immediate} is not representable in its instruction encoding")]
    ImmediateOutOfRange { kind: Kind, immediate: i128 },
}

/// Decoded row; `None` has an all-zero wire row, irrespective of other fields.
/// `Imm` is the decoded immediate, with `LowestAddress` subtracted for accesses;
/// `FallThroughPC = PC + 4` and `PCPlusImm = PC + Imm` modulo `2^64`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct BytecodeRow {
    pub variant: Option<Variant>,
    pub pc: u64,
    pub imm: u64,
    pub fall_through_pc: u64,
    pub pc_plus_imm: u64,
    pub rs1: u8,
    pub rs2: u8,
    pub rd: u8,
}
#[derive(Clone, Copy)]
enum Shape {
    Register,
    Immediate,
    Shift,
    MemoryLoad,
    MemoryStore,
    Branch,
    Jump,
    Upper,
    System,
}
impl BytecodeRow {
    /// Reads normalized operands without decoding an instruction word. Required
    /// operands and immediate encoding ranges are checked. Unsupported kinds
    /// and compressed rows become invalid. The source is a decoded public program.
    pub fn from_source(
        instruction: &SourceInstruction,
        lowest_address: u64,
    ) -> Result<Self, BytecodeRowError> {
        let kind = instruction.kind();
        let source = instruction.row();
        if source.is_compressed || kind.source_extension() != Some(SourceExtension::Rv64I) {
            return Ok(Self::default());
        }
        let operands = source.operands;
        for register in [operands.rs1, operands.rs2, operands.rd]
            .into_iter()
            .flatten()
        {
            if register >= 32 {
                return Err(BytecodeRowError::RegisterOutOfRange { register });
            }
        }
        let shape = if matches!(
            kind,
            Kind::ADD
                | Kind::SUB
                | Kind::AND
                | Kind::OR
                | Kind::XOR
                | Kind::SLT
                | Kind::SLTU
                | Kind::SLL
                | Kind::SRL
                | Kind::SRA
                | Kind::ADDW
                | Kind::SUBW
                | Kind::SLLW
                | Kind::SRLW
                | Kind::SRAW
        ) {
            Shape::Register
        } else if matches!(
            kind,
            Kind::SLLI | Kind::SRLI | Kind::SRAI | Kind::SLLIW | Kind::SRLIW | Kind::SRAIW
        ) {
            Shape::Shift
        } else if matches!(
            kind,
            Kind::LB | Kind::LH | Kind::LW | Kind::LD | Kind::LBU | Kind::LHU | Kind::LWU
        ) {
            Shape::MemoryLoad
        } else if matches!(kind, Kind::SB | Kind::SH | Kind::SW | Kind::SD) {
            Shape::MemoryStore
        } else if matches!(
            kind,
            Kind::BEQ | Kind::BNE | Kind::BLT | Kind::BGE | Kind::BLTU | Kind::BGEU
        ) {
            Shape::Branch
        } else if kind == Kind::JAL {
            Shape::Jump
        } else if matches!(kind, Kind::LUI | Kind::AUIPC) {
            Shape::Upper
        } else if matches!(kind, Kind::FENCE | Kind::ECALL | Kind::EBREAK) {
            Shape::System
        } else {
            Shape::Immediate
        };
        let required = |value: Option<u8>, operand| {
            value.ok_or(BytecodeRowError::MissingOperand { kind, operand })
        };
        let (rs1, rs2, rd) = match shape {
            Shape::Register => (
                required(operands.rs1, "rs1")?,
                required(operands.rs2, "rs2")?,
                required(operands.rd, "rd")?,
            ),
            Shape::Immediate | Shape::Shift | Shape::MemoryLoad => (
                required(operands.rs1, "rs1")?,
                0,
                required(operands.rd, "rd")?,
            ),
            Shape::MemoryStore | Shape::Branch => (
                required(operands.rs1, "rs1")?,
                required(operands.rs2, "rs2")?,
                0,
            ),
            Shape::Jump | Shape::Upper => (0, 0, required(operands.rd, "rd")?),
            Shape::System => (0, 0, 0),
        };
        let immediate = operands.imm;
        let imm64 = immediate as u64;
        let signed = imm64 as i64;
        let pattern_representable =
            immediate >= i128::from(i64::MIN) && immediate <= i128::from(u64::MAX);
        let fits = pattern_representable
            && match shape {
                Shape::Register => immediate == 0,
                Shape::Immediate | Shape::MemoryLoad | Shape::MemoryStore => {
                    (-2048..=2047).contains(&signed)
                }
                Shape::Branch => (-4096..=4094).contains(&signed) && signed % 2 == 0,
                Shape::Jump => (-1_048_576..=1_048_574).contains(&signed) && signed % 2 == 0,
                Shape::Upper => i32::try_from(signed).is_ok() && imm64.trailing_zeros() >= 12,
                Shape::Shift => {
                    let mask = if matches!(kind, Kind::SLLIW | Kind::SRLIW | Kind::SRAIW) {
                        31
                    } else {
                        63
                    };
                    let tag = if matches!(kind, Kind::SRAI | Kind::SRAIW) {
                        0x400
                    } else {
                        0
                    };
                    imm64 & !mask == tag
                }
                Shape::System => immediate == i128::from(kind == Kind::EBREAK),
            };
        if !fits {
            return Err(BytecodeRowError::ImmediateOutOfRange { kind, immediate });
        }
        let imm = match shape {
            Shape::Register | Shape::System => 0,
            Shape::MemoryLoad | Shape::MemoryStore => imm64.wrapping_sub(lowest_address),
            Shape::Shift => {
                imm64
                    & if matches!(kind, Kind::SLLIW | Kind::SRLIW | Kind::SRAIW) {
                        31
                    } else {
                        63
                    }
            }
            Shape::Immediate | Shape::Branch | Shape::Jump | Shape::Upper => imm64,
        };
        let pc = source.address as u64;
        Ok(Self {
            variant: Variant::from_source(kind, rd == 0),
            pc,
            imm,
            fall_through_pc: pc.wrapping_add(4),
            pc_plus_imm: pc.wrapping_add(imm),
            rs1,
            rs2,
            rd,
        })
    }
    /// Column bit pattern, using one-hot variant and register selectors. Every
    /// column is zero for an invalid row. Malformed register fields give zero.
    #[inline]
    pub fn column(&self, column: BytecodeColumn) -> u64 {
        let Some(variant) = self.variant else {
            return 0;
        };
        let selector = |register: u8| if register < 32 { 1_u64 << register } else { 0 };
        match column {
            BytecodeColumn::Valid => 1,
            BytecodeColumn::Variant => 1_u64 << variant as u8,
            BytecodeColumn::PC => self.pc,
            BytecodeColumn::Imm => self.imm,
            BytecodeColumn::FallThroughPC => self.fall_through_pc,
            BytecodeColumn::PCPlusImm => self.pc_plus_imm,
            BytecodeColumn::Rs1Ra => selector(self.rs1),
            BytecodeColumn::Rs2Ra => selector(self.rs2),
            BytecodeColumn::RdWa => selector(self.rd),
        }
    }
}

/// A bytecode table or final-PC statement error.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BytecodeError {
    #[error("instruction list has {instructions} rows, exceeding bytecode capacity {rows}")]
    InstructionListTooLong { instructions: usize, rows: usize },
    #[error("bytecode exponent {log_K} cannot be represented by this platform")]
    TableSizeOutOfRange { log_K: usize },
    #[error("allocation of bytecode table with {rows} rows failed")]
    TableAllocationFailed { rows: usize },
    #[error("bytecode row {index}: {source}")]
    Row {
        index: usize,
        source: BytecodeRowError,
    },
    #[error("bytecode row {index} PC {pc} is not a multiple of four")]
    PcNotMultipleOfFour { index: usize, pc: u64 },
    #[error("bytecode rows {first} and {second} repeat PC {pc}")]
    DuplicatePc {
        first: usize,
        second: usize,
        pc: u64,
    },
    #[error("final PC {final_pc} is not in valid bytecode")]
    FinalPcNotInBytecode { final_pc: u64 },
}
/// Public table padded with invalid rows to exactly `2^log_K_bytecode` rows.
#[derive(Debug, Clone)]
pub struct Bytecode {
    rows: Vec<BytecodeRow>,
    log_K: usize,
    lowest_address: u64,
    pc_indices: HashMap<u64, usize>,
}
impl Bytecode {
    /// Preprocesses the decoded public program into the supplied layout. The
    /// input list is the decoded public program; this function validates source
    /// shape, immediate ranges, valid-PC alignment and uniqueness, and capacity.
    pub fn preprocess(
        instructions: &[SourceInstruction],
        layout: &Layout,
    ) -> Result<Self, BytecodeError> {
        let log_K = layout.log_K_bytecode();
        let capacity = 1_usize
            .checked_shl(log_K as u32)
            .ok_or(BytecodeError::TableSizeOutOfRange { log_K })?;
        if instructions.len() > capacity {
            return Err(BytecodeError::InstructionListTooLong {
                instructions: instructions.len(),
                rows: capacity,
            });
        }
        if capacity > isize::MAX as usize / std::mem::size_of::<BytecodeRow>() {
            return Err(BytecodeError::TableSizeOutOfRange { log_K });
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(capacity)
            .map_err(|_| BytecodeError::TableAllocationFailed { rows: capacity })?;
        let mut pc_indices = HashMap::with_capacity(instructions.len());
        for (index, instruction) in instructions.iter().enumerate() {
            let row = BytecodeRow::from_source(instruction, layout.lowest_address())
                .map_err(|source| BytecodeError::Row { index, source })?;
            if row.variant.is_some() {
                if row.pc % 4 != 0 {
                    return Err(BytecodeError::PcNotMultipleOfFour { index, pc: row.pc });
                }
                if let Some(first) = pc_indices.insert(row.pc, index) {
                    return Err(BytecodeError::DuplicatePc {
                        first,
                        second: index,
                        pc: row.pc,
                    });
                }
            }
            rows.push(row);
        }
        rows.resize(capacity, BytecodeRow::default());
        Ok(Self {
            rows,
            log_K,
            lowest_address: layout.lowest_address(),
            pc_indices,
        })
    }
    /// Exactly `2^log_K` rows, including invalid padding.
    #[inline]
    pub fn rows(&self) -> &[BytecodeRow] {
        &self.rows
    }
    /// Exponent from the supplied layout.
    #[inline]
    pub fn log_K(&self) -> usize {
        self.log_K
    }
    /// RAM base used to normalize memory immediates.
    #[inline]
    pub fn lowest_address(&self) -> u64 {
        self.lowest_address
    }
    /// Index of a valid row with this PC; invalid and padding rows are excluded.
    #[inline]
    pub fn index_of_pc(&self, pc: u64) -> Option<usize> {
        self.pc_indices.get(&pc).copied()
    }
    /// Checks the public `FinalPC` names a valid row.
    #[inline]
    pub fn final_pc_index(&self, final_pc: u64) -> Result<usize, BytecodeError> {
        self.index_of_pc(final_pc)
            .ok_or(BytecodeError::FinalPcNotInBytecode { final_pc })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests require literal decoded fixtures to succeed"
)]
mod tests {
    use jolt_program::image::decode::decode_instruction;
    use jolt_riscv::{NormalizedOperands, SourceInstructionRow, RV64IMAC_JOLT};

    use super::*;

    fn decode(word: u32, pc: u64) -> SourceInstruction {
        decode_instruction(word, pc, false, RV64IMAC_JOLT).unwrap()
    }
    #[test]
    fn literal_decoded_rows_and_column_widths() {
        assert_eq!(
            BytecodeColumn::ALL
                .into_iter()
                .map(BytecodeColumn::width)
                .sum::<usize>(),
            417
        );
        assert_eq!(BYTECODE_ROW_BITS, 417);
        let fixtures = [
            (0xfff2_8193, Variant::ADDI, 5, 0, 3, u64::MAX, 0xff),
            (0x4072_d193, Variant::SRAI, 5, 0, 3, 7, 0x107),
            (0x4072_d19b, Variant::SRAIW, 5, 0, 3, 7, 0x107),
            (
                0x8000_01b7,
                Variant::LUI,
                0,
                0,
                3,
                0xffff_ffff_8000_0000,
                0xffff_ffff_8000_0100,
            ),
            (
                0x8000_0197,
                Variant::AUIPC,
                0,
                0,
                3,
                0xffff_ffff_8000_0000,
                0xffff_ffff_8000_0100,
            ),
            (
                0xff82_8183,
                Variant::LB,
                5,
                0,
                3,
                0xffff_ffff_ffff_eff8,
                0xffff_ffff_ffff_f0f8,
            ),
            (
                0xfe62_bc23,
                Variant::SD,
                5,
                6,
                0,
                0xffff_ffff_ffff_eff8,
                0xffff_ffff_ffff_f0f8,
            ),
            (
                0xfe62_8ce3,
                Variant::BEQ,
                5,
                6,
                0,
                0xffff_ffff_ffff_fff8,
                0xf8,
            ),
            (
                0xff9f_f1ef,
                Variant::JAL,
                0,
                0,
                3,
                0xffff_ffff_ffff_fff8,
                0xf8,
            ),
            (
                0xff9f_f06f,
                Variant::JAL_X0,
                0,
                0,
                0,
                0xffff_ffff_ffff_fff8,
                0xf8,
            ),
            (0x0072_8067, Variant::JALR_X0, 5, 0, 0, 7, 0x107),
            (
                0xff82_8003,
                Variant::LOAD1_X0,
                5,
                0,
                0,
                0xffff_ffff_ffff_eff8,
                0xffff_ffff_ffff_f0f8,
            ),
            (
                0xff82_9003,
                Variant::LOAD2_X0,
                5,
                0,
                0,
                0xffff_ffff_ffff_eff8,
                0xffff_ffff_ffff_f0f8,
            ),
            (
                0xff82_a003,
                Variant::LOAD4_X0,
                5,
                0,
                0,
                0xffff_ffff_ffff_eff8,
                0xffff_ffff_ffff_f0f8,
            ),
            (
                0xff82_b003,
                Variant::LOAD8_X0,
                5,
                0,
                0,
                0xffff_ffff_ffff_eff8,
                0xffff_ffff_ffff_f0f8,
            ),
            (
                0xff82_c003,
                Variant::LOAD1_X0,
                5,
                0,
                0,
                0xffff_ffff_ffff_eff8,
                0xffff_ffff_ffff_f0f8,
            ),
            (
                0xff82_d003,
                Variant::LOAD2_X0,
                5,
                0,
                0,
                0xffff_ffff_ffff_eff8,
                0xffff_ffff_ffff_f0f8,
            ),
            (
                0xff82_e003,
                Variant::LOAD4_X0,
                5,
                0,
                0,
                0xffff_ffff_ffff_eff8,
                0xffff_ffff_ffff_f0f8,
            ),
            (0x0000_000f, Variant::NOOP, 0, 0, 0, 0, 0x100),
            (0x0000_0073, Variant::ECALL, 0, 0, 0, 0, 0x100),
            (0x0010_0073, Variant::EBREAK, 0, 0, 0, 0, 0x100),
        ];
        for (word, variant, rs1, rs2, rd, imm, pc_plus_imm) in fixtures {
            assert_eq!(
                BytecodeRow::from_source(&decode(word, 0x100), 0x1000).unwrap(),
                BytecodeRow {
                    variant: Some(variant),
                    pc: 0x100,
                    imm,
                    fall_through_pc: 0x104,
                    pc_plus_imm,
                    rs1,
                    rs2,
                    rd
                }
            );
        }
        let row = BytecodeRow::from_source(&decode(0x0062_81b3, 0x100), 0).unwrap();
        assert_eq!(
            BytecodeColumn::ALL.map(|column| row.column(column)),
            [1, 1, 0x100, 0, 0x104, 0x100, 32, 64, 8]
        );
    }
    #[test]
    fn every_alu_x0_keeps_source_selectors() {
        let register_words = [
            0x0062_8033,
            0x4062_8033,
            0x0062_f033,
            0x0062_e033,
            0x0062_c033,
            0x0062_a033,
            0x0062_b033,
            0x0062_9033,
            0x0062_d033,
            0x4062_d033,
            0x0062_803b,
            0x4062_803b,
            0x0062_903b,
            0x0062_d03b,
            0x4062_d03b,
        ];
        for word in register_words {
            assert_eq!(
                BytecodeRow::from_source(&decode(word, 0x100), 0).unwrap(),
                BytecodeRow {
                    variant: Some(Variant::NOOP),
                    pc: 0x100,
                    imm: 0,
                    fall_through_pc: 0x104,
                    pc_plus_imm: 0x100,
                    rs1: 5,
                    rs2: 6,
                    rd: 0
                }
            );
        }
        let immediate_words = [
            0x0072_8013,
            0x0072_f013,
            0x0072_e013,
            0x0072_c013,
            0x0072_a013,
            0x0072_b013,
            0x0072_801b,
            0x0072_9013,
            0x0072_d013,
            0x4072_d013,
            0x0072_901b,
            0x0072_d01b,
            0x4072_d01b,
        ];
        for word in immediate_words {
            assert_eq!(
                BytecodeRow::from_source(&decode(word, 0x100), 0).unwrap(),
                BytecodeRow {
                    variant: Some(Variant::NOOP),
                    pc: 0x100,
                    imm: 7,
                    fall_through_pc: 0x104,
                    pc_plus_imm: 0x107,
                    rs1: 5,
                    rs2: 0,
                    rd: 0
                }
            );
        }
        for word in [0x0000_7037, 0x0000_7017] {
            assert_eq!(
                BytecodeRow::from_source(&decode(word, 0x100), 0).unwrap(),
                BytecodeRow {
                    variant: Some(Variant::NOOP),
                    pc: 0x100,
                    imm: 0x7000,
                    fall_through_pc: 0x104,
                    pc_plus_imm: 0x7100,
                    rs1: 0,
                    rs2: 0,
                    rd: 0
                }
            );
        }
    }
    #[test]
    fn invalid_wire_row_is_all_zero() {
        let row = BytecodeRow {
            variant: None,
            pc: u64::MAX,
            imm: 1,
            fall_through_pc: 2,
            pc_plus_imm: 3,
            rs1: 5,
            rs2: 6,
            rd: 7,
        };
        assert_eq!(
            BytecodeColumn::ALL.map(|column| row.column(column)),
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            BytecodeRow::from_source(&decode(0x0262_81b3, 0x100), 0).unwrap(),
            BytecodeRow::default()
        );
        let compressed = decode_instruction(0x0062_81b3, 0x100, true, RV64IMAC_JOLT).unwrap();
        assert_eq!(
            BytecodeRow::from_source(&compressed, 0).unwrap(),
            BytecodeRow::default()
        );
    }
    #[test]
    fn preprocessing_capacity_padding_and_public_pc_checks() {
        let layout = Layout::new(3, 1, 0).unwrap();
        let instructions = [
            decode(0x0000_0013, 0x100),
            decode(0x0000_0013, 0x104),
            decode(0x0262_81b3, 0x108),
            decode(0x0000_0013, 0x10c),
            decode(0x0000_0013, 0x110),
        ];
        let code = Bytecode::preprocess(&instructions, &layout).unwrap();
        assert_eq!(code.rows().len(), 8);
        assert_eq!(code.log_K(), 3);
        assert_eq!(code.lowest_address(), 0);
        for (pc, index) in [(0x100, 0), (0x104, 1), (0x10c, 3), (0x110, 4)] {
            assert_eq!(code.index_of_pc(pc), Some(index));
            assert_eq!(code.final_pc_index(pc), Ok(index));
        }
        assert_eq!(code.index_of_pc(0x108), None);
        assert_eq!(code.index_of_pc(0), None);
        assert_eq!(
            code.rows().iter().skip(5).copied().collect::<Vec<_>>(),
            [BytecodeRow::default(); 3]
        );
        assert_eq!(
            code.final_pc_index(0x108),
            Err(BytecodeError::FinalPcNotInBytecode { final_pc: 0x108 })
        );
        assert_eq!(
            Bytecode::preprocess(&[decode(0x0000_0013, 6)], &layout).unwrap_err(),
            BytecodeError::PcNotMultipleOfFour { index: 0, pc: 6 }
        );
        assert_eq!(
            Bytecode::preprocess(&[decode(0x0000_0013, 4), decode(0x0000_0013, 4)], &layout)
                .unwrap_err(),
            BytecodeError::DuplicatePc {
                first: 0,
                second: 1,
                pc: 4
            }
        );
        let small = Layout::new(2, 1, 0).unwrap();
        assert_eq!(
            Bytecode::preprocess(&instructions, &small).unwrap_err(),
            BytecodeError::InstructionListTooLong {
                instructions: 5,
                rows: 4
            }
        );
        assert_eq!(
            Bytecode::preprocess(&[], &small).unwrap().rows(),
            [BytecodeRow::default(); 4]
        );
    }
    #[test]
    fn malformed_source_shape_and_immediate_domains() {
        let row = |kind, operands| {
            SourceInstruction::new(
                kind,
                SourceInstructionRow {
                    address: 4,
                    operands,
                    inline: None,
                    is_compressed: false,
                },
            )
        };
        let base = NormalizedOperands {
            rs1: Some(1),
            rs2: Some(2),
            rd: Some(3),
            imm: 0,
        };
        for (operands, operand) in [
            (NormalizedOperands { rs1: None, ..base }, "rs1"),
            (NormalizedOperands { rs2: None, ..base }, "rs2"),
            (NormalizedOperands { rd: None, ..base }, "rd"),
        ] {
            assert_eq!(
                BytecodeRow::from_source(&row(Kind::ADD, operands), 0),
                Err(BytecodeRowError::MissingOperand {
                    kind: Kind::ADD,
                    operand
                })
            );
        }
        let bad = row(
            Kind::ADD,
            NormalizedOperands {
                rd: Some(32),
                ..base
            },
        );
        let error = BytecodeRowError::RegisterOutOfRange { register: 32 };
        assert_eq!(BytecodeRow::from_source(&bad, 0), Err(error.clone()));
        assert_eq!(
            Bytecode::preprocess(&[bad], &Layout::new(1, 1, 0).unwrap()).unwrap_err(),
            BytecodeError::Row {
                index: 0,
                source: error
            }
        );
        for (kind, immediate) in [
            (Kind::ADD, 1),
            (Kind::ADDI, 2048),
            (Kind::LB, -2049),
            (Kind::SB, 2048),
            (Kind::BEQ, 3),
            (Kind::BEQ, 4096),
            (Kind::JAL, 1),
            (Kind::JAL, 1_048_576),
            (Kind::LUI, 1),
            (Kind::LUI, 1_i128 << 32),
            (Kind::SLLI, 64),
            (Kind::SRLI, 0x400),
            (Kind::SRAI, 0x440),
            (Kind::SRAIW, 0x420),
            (Kind::ADDI, 1_i128 << 64),
        ] {
            assert_eq!(
                BytecodeRow::from_source(
                    &row(
                        kind,
                        NormalizedOperands {
                            imm: immediate,
                            ..base
                        }
                    ),
                    0
                ),
                Err(BytecodeRowError::ImmediateOutOfRange { kind, immediate })
            );
        }
        for immediate in [-1, i128::from(u64::MAX)] {
            assert_eq!(
                BytecodeRow::from_source(
                    &row(
                        Kind::ADDI,
                        NormalizedOperands {
                            imm: immediate,
                            ..base
                        }
                    ),
                    0
                )
                .unwrap()
                .imm,
                u64::MAX
            );
        }
    }
    #[test]
    fn every_immediate_encoding_rejects_both_just_past_bounds() {
        for (kind, bounds) in [
            (Kind::ADDI, [-2049, 2048]),
            (Kind::LB, [-2049, 2048]),
            (Kind::SB, [-2049, 2048]),
            (Kind::BEQ, [-4098, 4096]),
            (Kind::JAL, [-1_048_578, 1_048_576]),
            (Kind::LUI, [-2_147_487_744, 2_147_483_648]),
            (Kind::SLLI, [-1, 64]),
            (Kind::SLLIW, [-1, 32]),
            (Kind::SRAI, [0x3ff, 0x440]),
            (Kind::SRAIW, [0x3ff, 0x420]),
        ] {
            for immediate in bounds {
                let source = SourceInstruction::new(
                    kind,
                    SourceInstructionRow {
                        address: 0,
                        operands: NormalizedOperands {
                            rs1: Some(1),
                            rs2: Some(2),
                            rd: Some(3),
                            imm: immediate,
                        },
                        inline: None,
                        is_compressed: false,
                    },
                );
                assert_eq!(
                    BytecodeRow::from_source(&source, 0),
                    Err(BytecodeRowError::ImmediateOutOfRange { kind, immediate })
                );
            }
        }
    }
}
