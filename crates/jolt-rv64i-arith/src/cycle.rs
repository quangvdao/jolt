//! Stateless generation of a cycle's four committed words from trace facts.

use thiserror::Error as ThisError;

use crate::bytecode::Bytecode;
use crate::decode::{eval, Sources};
use crate::layout::{set_bit, BitsRow, Layout, LayoutError};
use crate::variant::{BranchCondition, Variant};
use crate::words::BaseWords;

/// Facts for one fetched instruction. Register reads correspond to its source
/// selectors; destination values are zero when `rd = x0`. A data access's RAM
/// index is `(address - LowestAddress) / 8`, and words are little endian.
/// `next_pc` is the next cycle's PC, or `FinalPC` on the last cycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CycleFacts {
    pub bytecode_index: u32,
    pub rs1_value: u64,
    pub rs2_value: u64,
    pub rd_pre_value: u64,
    pub rd_post_value: u64,
    pub ram_word_index: u64,
    pub ram_pre_value: u64,
    pub ram_post_value: u64,
    pub next_pc: u64,
}

#[cfg(test)]
#[path = "cycle_tests.rs"]
mod tests;

/// A fact or layout that cannot produce the canonical committed row.
#[derive(Clone, Debug, PartialEq, Eq, ThisError)]
pub enum WitnessError {
    #[error("bytecode and committed layout have different sizes or lowest addresses")]
    LayoutMismatch,
    #[error("cycle facts length {facts} differs from output rows length {rows}")]
    LengthMismatch { facts: usize, rows: usize },
    #[error("bytecode index {bytecode_index} is outside the bytecode table")]
    BytecodeIndexOutOfRange { bytecode_index: u32 },
    #[error("bytecode index {bytecode_index} selects an invalid row")]
    InvalidRow { bytecode_index: u32 },
    #[error("relative RAM address {relative_address} is outside RAM")]
    AddressOutsideRam { relative_address: u64 },
    #[error("relative RAM address {relative_address} is not aligned to width {width}")]
    UnalignedAccess { relative_address: u64, width: u8 },
    #[error("RAM word index {found} differs from expected index {expected}")]
    RamWordIndexMismatch { expected: u64, found: u64 },
    #[error("committed layout write failed: {source}")]
    Layout { source: LayoutError },
}

/// The first failing cycle, relative to the slice passed to `fill`.
#[derive(Clone, Debug, PartialEq, Eq, ThisError)]
#[error("cycle {cycle}: {error}")]
pub struct CycleError {
    pub cycle: usize,
    pub error: WitnessError,
}

/// Borrowed setup for generating rows without retaining per-cycle state.
pub struct BitsBuilder<'a> {
    layout: &'a Layout,
    bytecode: &'a Bytecode,
}

impl<'a> BitsBuilder<'a> {
    /// Requires the bytecode and layout to agree on `log_K_bytecode` and
    /// `LowestAddress`.
    pub fn new(layout: &'a Layout, bytecode: &'a Bytecode) -> Result<Self, WitnessError> {
        if layout.log_K_bytecode() != bytecode.log_K()
            || layout.lowest_address() != bytecode.lowest_address()
        {
            return Err(WitnessError::LayoutMismatch);
        }
        Ok(Self { layout, bytecode })
    }

    /// Computes `Inc`, chunk indicators, `KeysDiffer`, `ShouldBranch` and
    /// `JalrLowBit`; all other fields start at zero. The scalar call on a stack
    /// `CycleFacts` is the path a trace adapter fuses with its existing pass.
    ///
    /// Every cycle reads `bytecode_index`. Non-stores read `rd_pre_value` and
    /// `rd_post_value` for `Inc`. Accesses read `rs1_value` and `ram_word_index`
    /// for the address; stores additionally read `ram_pre_value` and
    /// `ram_post_value` for `Inc`. RAM fields are ignored without an access,
    /// and a load ignores `ram_post_value`. Register shifts read `rs2_value`;
    /// comparisons read `rs1_value` and, for register keys, `rs2_value`.
    /// JALR reads `rs1_value`. `next_pc` is not read here.
    ///
    /// This does not execute the instruction or validate destination and RAM
    /// post-values. Those values feed the local rows through `BaseWords`,
    /// subject to its surrounding-protocol obligations.
    #[inline]
    pub fn bits_row(&self, facts: &CycleFacts) -> Result<BitsRow, WitnessError> {
        let row = self
            .bytecode
            .rows()
            .get(facts.bytecode_index as usize)
            .ok_or(WitnessError::BytecodeIndexOutOfRange {
                bytecode_index: facts.bytecode_index,
            })?;
        let variant = row.variant.ok_or(WitnessError::InvalidRow {
            bytecode_index: facts.bytecode_index,
        })?;
        let mut bits = [0; 4];
        self.layout
            .write_bytecode_index(&mut bits, u64::from(facts.bytecode_index))
            .map_err(|source| WitnessError::Layout { source })?;
        let mut pos = 0;
        if let Some(access) = variant.access() {
            let relative_address = facts.rs1_value.wrapping_add(row.imm);
            if u128::from(relative_address) >= (8u128 << self.layout.log_K_ram()) {
                return Err(WitnessError::AddressOutsideRam { relative_address });
            }
            if !relative_address.is_multiple_of(u64::from(access.width)) {
                return Err(WitnessError::UnalignedAccess {
                    relative_address,
                    width: access.width,
                });
            }
            let expected = relative_address >> 3;
            if expected != facts.ram_word_index {
                return Err(WitnessError::RamWordIndexMismatch {
                    expected,
                    found: facts.ram_word_index,
                });
            }
            self.layout
                .write_ram_index(&mut bits, expected)
                .map_err(|source| WitnessError::Layout { source })?;
            pos = (relative_address & 7) as u8;
        }
        if let Some(inc) = bits.first_mut() {
            *inc = if variant.is_store() {
                facts.ram_pre_value ^ facts.ram_post_value
            } else {
                facts.rd_pre_value ^ facts.rd_post_value
            };
        }
        let shift = variant.shift();
        let keys = variant.key_kind();
        if shift.is_some() || keys.is_some() {
            let base = BaseWords {
                rs1_value: facts.rs1_value,
                rs2_value: facts.rs2_value,
                ..BaseWords::default()
            };
            let src = Sources::new(self.layout, row, &base, &bits);
            if let Some(shift) = shift {
                let mask = if shift.kind.is_word() { 31 } else { 63 };
                pos = (src.get(shift.amount) & mask) as u8;
            }
            if let Some(keys) = keys {
                let (left_form, right_form) = keys.keys();
                let left = eval(left_form, &src);
                let right = eval(right_form, &src);
                let diff = left ^ right;
                let differs = diff != 0;
                if differs {
                    pos = diff.ilog2() as u8;
                }
                set_bit(&mut bits, self.layout.keys_differ(), differs);
                let less = differs && (right >> pos) & 1 != 0;
                if let Some(branch) = variant.branch() {
                    let taken = match branch {
                        BranchCondition::Equal => !differs,
                        BranchCondition::NotEqual => differs,
                        BranchCondition::Less => less,
                        BranchCondition::NotLess => !less,
                    };
                    set_bit(&mut bits, self.layout.should_branch(), taken);
                }
            }
        }
        if matches!(variant, Variant::JALR | Variant::JALR_X0) {
            set_bit(
                &mut bits,
                self.layout.jalr_low_bit(),
                facts.rs1_value.wrapping_add(row.imm) & 1 != 0,
            );
        }
        self.layout
            .write_pos(&mut bits, pos)
            .map_err(|source| WitnessError::Layout { source })?;
        Ok(bits)
    }

    /// Applies the scalar generator in order with no state between cycles.
    /// A length mismatch fails at the shorter length before writing anything;
    /// otherwise rows before the first error are written. Independently split
    /// slices produce the same rows as a single call.
    #[inline]
    pub fn fill(&self, facts: &[CycleFacts], out: &mut [BitsRow]) -> Result<(), CycleError> {
        if facts.len() != out.len() {
            return Err(CycleError {
                cycle: facts.len().min(out.len()),
                error: WitnessError::LengthMismatch {
                    facts: facts.len(),
                    rows: out.len(),
                },
            });
        }
        for (cycle, (fact, target)) in facts.iter().zip(out).enumerate() {
            *target = self
                .bits_row(fact)
                .map_err(|error| CycleError { cycle, error })?;
        }
        Ok(())
    }
}
