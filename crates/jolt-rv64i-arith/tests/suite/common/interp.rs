//! RV64I semantics from the unprivileged ISA, with separate instruction and data memory.

use std::collections::HashMap;
use std::fmt::{Display, Formatter};

/// A failed setup or execution; execution errors leave the failing step unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidRam { lowest_address: u64, log_k_ram: u8 },
    UnalignedProgram { address: u64 },
    DuplicateInstruction { address: u64 },
    InvalidRegister { register: u8 },
    InvalidRamIndex { index: u64 },
    MissingInstruction { pc: u64 },
    InvalidInstruction { pc: u64, word: u32 },
    InvalidTarget { pc: u64, target: u64 },
    OutsideRam { address: u64, width: u8 },
    UnalignedAccess { address: u64, width: u8 },
    RamCapacity { index: u64 },
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RV64I oracle: {self:?}")
    }
}

impl std::error::Error for Error {}

/// One naturally aligned data access; word values include every preserved byte.
/// The index is relative to the RAM base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    pub is_store: bool,
    pub address: u64,
    pub width: u8,
    pub word_index: u64,
    pub word_before: u64,
    pub word_after: u64,
}

/// Source and destination values are pre-state values, except `rd_post_value`.
/// Absent operands and writes to x0 have zero values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    pub pc: u64,
    pub word: u32,
    pub rs1_value: u64,
    pub rs2_value: u64,
    pub rd_pre_value: u64,
    pub rd_post_value: u64,
    pub access: Option<Access>,
    pub next_pc: u64,
}

#[derive(Clone, Copy)]
struct MemoryAccess {
    record: Access,
    loaded: u64,
}

#[derive(Clone, Copy)]
enum Op {
    Lui,
    Auipc,
    Jal,
    Jalr,
    Branch(u8),
    Load {
        width: u8,
        signed: bool,
    },
    Store(u8),
    Arithmetic {
        funct3: u8,
        alternate: bool,
        immediate: bool,
        word: bool,
    },
    Fence,
    System,
}

struct Instruction {
    op: Op,
    rd: Option<usize>,
    rs1: Option<usize>,
    rs2: Option<usize>,
    immediate: u64,
}

fn sign_extend(value: u64, bits: u32) -> u64 {
    ((value << (64 - bits)) as i64 >> (64 - bits)) as u64
}

fn decode(word: u32) -> Option<Instruction> {
    let opcode = word & 0x7f;
    let funct3 = ((word >> 12) & 7) as u8;
    let funct7 = word >> 25;
    let rd = Some(((word >> 7) & 31) as usize);
    let rs1 = Some(((word >> 15) & 31) as usize);
    let rs2 = Some(((word >> 20) & 31) as usize);
    let i_imm = sign_extend(u64::from(word >> 20), 12);
    let (op, rd, rs1, rs2, immediate) = match opcode {
        0x37 | 0x17 => (
            if opcode == 0x37 { Op::Lui } else { Op::Auipc },
            rd,
            None,
            None,
            sign_extend(u64::from(word & 0xffff_f000), 32),
        ),
        0x6f => {
            let imm = ((word >> 31) << 20)
                | (((word >> 12) & 0xff) << 12)
                | (((word >> 20) & 1) << 11)
                | (((word >> 21) & 0x3ff) << 1);
            (Op::Jal, rd, None, None, sign_extend(u64::from(imm), 21))
        }
        0x67 if funct3 == 0 => (Op::Jalr, rd, rs1, None, i_imm),
        0x63 if matches!(funct3, 0 | 1 | 4 | 5 | 6 | 7) => {
            let imm = ((word >> 31) << 12)
                | (((word >> 7) & 1) << 11)
                | (((word >> 25) & 0x3f) << 5)
                | (((word >> 8) & 15) << 1);
            (
                Op::Branch(funct3),
                None,
                rs1,
                rs2,
                sign_extend(u64::from(imm), 13),
            )
        }
        0x03 if funct3 <= 6 => {
            let width = 1 << (funct3 & 3);
            (
                Op::Load {
                    width,
                    signed: funct3 < 3,
                },
                rd,
                rs1,
                None,
                i_imm,
            )
        }
        0x23 if funct3 <= 3 => {
            let imm = ((word >> 25) << 5) | ((word >> 7) & 31);
            (
                Op::Store(1 << funct3),
                None,
                rs1,
                rs2,
                sign_extend(u64::from(imm), 12),
            )
        }
        0x13 | 0x1b | 0x33 | 0x3b => {
            let immediate = opcode & 0x20 == 0;
            let word_op = opcode & 8 != 0;
            let alternate = word & 0x4000_0000 != 0;
            let valid = if immediate {
                match funct3 {
                    0 => true,
                    1 => {
                        if word_op {
                            funct7 == 0
                        } else {
                            word >> 26 == 0
                        }
                    }
                    5 => {
                        if word_op {
                            matches!(funct7, 0 | 0x20)
                        } else {
                            matches!(word >> 26, 0 | 0x10)
                        }
                    }
                    _ => !word_op,
                }
            } else {
                (!word_op || matches!(funct3, 0 | 1 | 5))
                    && (funct7 == 0 || (funct7 == 0x20 && matches!(funct3, 0 | 5)))
            };
            if !valid {
                return None;
            }
            (
                Op::Arithmetic {
                    funct3,
                    alternate,
                    immediate,
                    word: word_op,
                },
                rd,
                rs1,
                if immediate { None } else { rs2 },
                i_imm,
            )
        }
        // The base ISA ignores FENCE's reserved fm, rs1 and rd fields.
        0x0f if funct3 == 0 => (Op::Fence, None, None, None, 0),
        0x73 if matches!(word, 0x0000_0073 | 0x0010_0073) => (Op::System, None, None, None, 0),
        _ => return None,
    };
    Some(Instruction {
        op,
        rd,
        rs1,
        rs2,
        immediate,
    })
}

/// Sparse RAM, a separate program map, and 32 registers with immutable x0.
/// Setup allocates; `step` uses only existing capacity and returns a plain record.
#[derive(Debug, PartialEq, Eq)]
pub struct Machine {
    pc: u64,
    x: [u64; 32],
    program: HashMap<u64, u32>,
    lowest_address: u64,
    ram_bytes: u128,
    ram: HashMap<u64, u64>,
}

impl Machine {
    /// Program addresses must be distinct multiples of four. RAM starts at a
    /// multiple of eight, does not wrap, and may end at 2^64.
    /// `ram_capacity` reserves at least that many distinct
    /// stored words independently of RAM's addressable size. Exhaustion returns
    /// `RamCapacity` without executing the store; budget writes during setup.
    pub fn new(
        program: &[(u64, u32)],
        pc: u64,
        lowest_address: u64,
        log_k_ram: u8,
        ram_capacity: usize,
    ) -> Result<Self, Error> {
        if !lowest_address.is_multiple_of(8) || log_k_ram > 61 {
            return Err(Error::InvalidRam {
                lowest_address,
                log_k_ram,
            });
        }
        let ram_bytes = 8_u128 << log_k_ram;
        if u128::from(lowest_address) + ram_bytes > 1_u128 << 64 {
            return Err(Error::InvalidRam {
                lowest_address,
                log_k_ram,
            });
        }
        let mut instructions = HashMap::with_capacity(program.len());
        for &(address, word) in program {
            if address % 4 != 0 {
                return Err(Error::UnalignedProgram { address });
            }
            if instructions.insert(address, word).is_some() {
                return Err(Error::DuplicateInstruction { address });
            }
        }
        Ok(Self {
            pc,
            x: [0; 32],
            program: instructions,
            lowest_address,
            ram_bytes,
            ram: HashMap::with_capacity(ram_capacity),
        })
    }

    /// The complete register file; x0 is always zero.
    pub fn registers(&self) -> &[u64; 32] {
        &self.x
    }

    /// The current instruction address, which need not have a stored instruction.
    pub fn pc(&self) -> u64 {
        self.pc
    }

    /// Select an entry address without fetching or changing other state.
    pub fn set_pc(&mut self, pc: u64) {
        self.pc = pc;
    }

    /// Initialize a register; writing x0 is ignored, indices above 31 are errors.
    pub fn set_register(&mut self, register: u8, value: u64) -> Result<(), Error> {
        let Some(slot) = self.x.get_mut(usize::from(register)) else {
            return Err(Error::InvalidRegister { register });
        };
        if register != 0 {
            *slot = value;
        }
        Ok(())
    }

    /// Unset in-range words read as zero; out-of-range indices are errors.
    pub fn ram_word(&self, index: u64) -> Result<u64, Error> {
        if u128::from(index) >= self.ram_bytes / 8 {
            return Err(Error::InvalidRamIndex { index });
        }
        Ok(self.ram.get(&index).copied().unwrap_or(0))
    }

    /// Initialize an in-range RAM word. Setup may grow the sparse storage.
    pub fn set_ram_word(&mut self, index: u64, value: u64) -> Result<(), Error> {
        let _ = self.ram_word(index)?;
        let _ = self.ram.insert(index, value);
        Ok(())
    }

    fn instruction(&self, pc: u64) -> Result<(u32, Instruction), Error> {
        let word = self
            .program
            .get(&pc)
            .copied()
            .ok_or(Error::MissingInstruction { pc })?;
        let instruction = decode(word).ok_or(Error::InvalidInstruction { pc, word })?;
        Ok((word, instruction))
    }

    fn access(&self, address: u64, width: u8, store: Option<u64>) -> Result<MemoryAccess, Error> {
        let offset = address.wrapping_sub(self.lowest_address);
        if u128::from(offset) + u128::from(width) > self.ram_bytes {
            return Err(Error::OutsideRam { address, width });
        }
        if !address.is_multiple_of(u64::from(width)) {
            return Err(Error::UnalignedAccess { address, width });
        }
        let word_index = offset / 8;
        let word_before = self.ram_word(word_index)?;
        let shift = (offset % 8) * 8;
        let mask = u64::MAX >> (64 - u32::from(width) * 8);
        let after = store.map_or(word_before, |value| {
            (word_before & !(mask << shift)) | ((value & mask) << shift)
        });
        if store.is_some()
            && !self.ram.contains_key(&word_index)
            && self.ram.len() == self.ram.capacity()
        {
            return Err(Error::RamCapacity { index: word_index });
        }
        Ok(MemoryAccess {
            record: Access {
                is_store: store.is_some(),
                address,
                width,
                word_index,
                word_before,
                word_after: after,
            },
            loaded: (word_before >> shift) & mask,
        })
    }

    /// Execute one atomic step. Jumps and taken branches validate the destination
    /// before any write; fall-through is validated only by the next step or `run`.
    pub fn step(&mut self) -> Result<Record, Error> {
        let (word, instruction) = self.instruction(self.pc)?;
        let a = instruction.rs1.map_or(0, |register| self.x[register]);
        let b = instruction.rs2.map_or(0, |register| self.x[register]);
        let imm = instruction.immediate;
        let mut next_pc = self.pc.wrapping_add(4);
        let mut access = None;
        let mut jump = false;
        let value = match instruction.op {
            Op::Lui => imm,
            Op::Auipc => self.pc.wrapping_add(imm),
            Op::Jal | Op::Jalr => {
                next_pc = if matches!(instruction.op, Op::Jal) {
                    self.pc.wrapping_add(imm)
                } else {
                    a.wrapping_add(imm) & !1
                };
                jump = true;
                self.pc.wrapping_add(4)
            }
            Op::Branch(condition) => {
                let taken = match condition {
                    0 => a == b,
                    1 => a != b,
                    4 => (a as i64) < (b as i64),
                    5 => (a as i64) >= (b as i64),
                    6 => a < b,
                    _ => a >= b,
                };
                if taken {
                    next_pc = self.pc.wrapping_add(imm);
                    jump = true;
                }
                0
            }
            Op::Load { width, signed } => {
                let memory = self.access(a.wrapping_add(imm), width, None)?;
                let loaded = memory.loaded;
                access = Some(memory);
                if signed {
                    sign_extend(loaded, u32::from(width) * 8)
                } else {
                    loaded
                }
            }
            Op::Store(width) => {
                access = Some(self.access(a.wrapping_add(imm), width, Some(b))?);
                0
            }
            Op::Arithmetic {
                funct3,
                alternate,
                immediate,
                word,
            } => {
                let rhs = if immediate { imm } else { b };
                let shift = (rhs & if word { 31 } else { 63 }) as u32;
                let result = match funct3 {
                    0 => {
                        if alternate && !immediate {
                            a.wrapping_sub(rhs)
                        } else {
                            a.wrapping_add(rhs)
                        }
                    }
                    1 => a << shift,
                    2 => u64::from((a as i64) < (rhs as i64)),
                    3 => u64::from(a < rhs),
                    4 => a ^ rhs,
                    5 => {
                        if alternate {
                            if word {
                                ((a as i32) >> shift) as u64
                            } else {
                                ((a as i64) >> shift) as u64
                            }
                        } else if word {
                            u64::from((a as u32) >> shift)
                        } else {
                            a >> shift
                        }
                    }
                    6 => a | rhs,
                    _ => a & rhs,
                };
                if word {
                    sign_extend(result & 0xffff_ffff, 32)
                } else {
                    result
                }
            }
            Op::Fence => 0,
            Op::System => {
                next_pc = self.pc;
                0
            }
        };
        if jump && self.instruction(next_pc).is_err() {
            return Err(Error::InvalidTarget {
                pc: self.pc,
                target: next_pc,
            });
        }
        let rd = instruction.rd.filter(|&register| register != 0);
        let rd_pre_value = rd.map_or(0, |register| self.x[register]);
        let record = Record {
            pc: self.pc,
            word,
            rs1_value: a,
            rs2_value: b,
            rd_pre_value,
            rd_post_value: rd.map_or(0, |_| value),
            access: access.map(|memory| memory.record),
            next_pc,
        };
        if let Some(memory) = access.filter(|memory| memory.record.is_store) {
            let _ = self
                .ram
                .insert(memory.record.word_index, memory.record.word_after);
        }
        if let Some(register) = rd {
            self.x[register] = value;
        }
        self.pc = next_pc;
        Ok(record)
    }

    /// Execute exactly `cycles` successful steps and require a valid final PC.
    /// On failure, earlier successful steps remain; the failing step is atomic.
    pub fn run(&mut self, cycles: usize) -> Result<(), Error> {
        for _ in 0..cycles {
            let _ = self.step()?;
        }
        let _ = self.instruction(self.pc)?;
        Ok(())
    }
}
