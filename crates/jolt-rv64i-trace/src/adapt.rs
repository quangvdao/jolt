use common::{constants::RAM_START_ADDRESS, jolt_device::MemoryLayout};
use jolt_program::execution::{RamAccess, SourceTraceRow};
use jolt_rv64i_arith::{
    BaseWords, BitsBuilder, Bytecode, CycleFacts, Layout, RowSystem, WitnessRow,
};
use rayon::iter::{
    plumbing::{Consumer, Folder, Reducer, UnindexedConsumer},
    ParallelExtend, ParallelIterator,
};

use crate::{AdapterError, StallCause};

const CHUNK_ROWS: usize = 1 << 16;

/// Power-of-two cycle facts and the candidate RAM exponent. Statement and RAM
/// bounds remain the consumer's responsibility.
#[derive(Debug)]
pub struct Execution {
    pub facts: Vec<CycleFacts>,
    pub log_K_ram: u8,
}

impl Execution {
    pub fn log_T(&self) -> u8 {
        self.facts.len().ilog2() as u8
    }

    pub fn final_pc(&self) -> u64 {
        self.facts.last().map_or(0, |fact| fact.next_pc)
    }
}

/// Converts each row with one indexed bytecode load and no replay. The first
/// cycle must match the entry; indices must select valid rows with the same PC.
/// Access words must be at or above RAM and eight-byte aligned. Each chunk of
/// 2^16 rows returns the maximum accessed index and earliest error, with checks
/// ordered as entry, identity, RAM base and alignment within each cycle.
///
/// The candidate RAM exponent covers I/O, image and accesses, with minimum 5.
/// Non-power-of-two traces require a final PC stall. Its fact is updated for the
/// state left by that cycle and checked once by the row system before padding.
/// Empty traces fail before conversion; stall, layout and fixed-point checks
/// follow all per-cycle checks. The facts vector is reserved at its final size
/// and each fact is written once, including padding.
pub fn adapt(
    bytecode: &Bytecode,
    image: &[(u64, u64)],
    memory_layout: &MemoryLayout,
    entry_pc: u64,
    rows: &[SourceTraceRow],
) -> Result<Execution, AdapterError> {
    let last = rows.last().ok_or(AdapterError::EmptyTrace)?;
    let count = rows
        .len()
        .max(2)
        .checked_next_power_of_two()
        .ok_or(AdapterError::TraceTooLarge { rows: rows.len() })?;
    let mut facts = Vec::new();
    facts
        .try_reserve_exact(count)
        .map_err(|_| AdapterError::Allocation { facts: count })?;
    let mut reduction = Reduction::default();
    let pass = Pass { bytecode, entry_pc };
    facts.par_extend(FactPass {
        pass: &pass,
        rows,
        reduction: &mut reduction,
    });
    if let Some((_, error)) = reduction.error {
        return Err(error);
    }
    let padded = rows.len() != count;
    let cycle = rows.len() - 1;
    if padded && last.next_pc() != last.pc() {
        return Err(AdapterError::NoStall {
            cycle,
            pc: last.pc(),
        });
    }
    let mask_end = memory_layout.remapped_word_address(RAM_START_ADDRESS)?;
    let image_end = image
        .last()
        .map_or(0_u128, |(index, _)| u128::from(*index) + 1);
    let words = u128::from(mask_end)
        .max(image_end)
        .max(u128::from(reduction.max_index) + 1)
        .max(32);
    let log_K_ram = (128 - (words - 1).leading_zeros()) as u8;
    let layout = Layout::new(
        bytecode.log_K(),
        usize::from(log_K_ram),
        bytecode.lowest_address(),
    )?;
    if padded {
        let mut stall = facts[cycle];
        let row = &bytecode.rows()[stall.bytecode_index as usize];
        stall.rd_pre_value = stall.rd_post_value;
        if row.rs1 == row.rd && row.rd != 0 {
            stall.rs1_value = stall.rd_post_value;
        }
        let checked = (|| {
            let bits = BitsBuilder::new(&layout, bytecode)?.bits_row(&stall)?;
            let witness = WitnessRow::compute(&layout, row, &BaseWords::from_facts(&stall), &bits);
            RowSystem::new(&layout).check(&witness)?;
            Ok::<_, StallCause>(())
        })();
        checked.map_err(|cause| AdapterError::StallNotFixedPoint {
            cycle,
            pc: last.pc(),
            cause,
        })?;
        facts.resize(count, stall);
    }
    Ok(Execution { facts, log_K_ram })
}

#[derive(Default)]
struct Reduction {
    max_index: u64,
    error: Option<(usize, AdapterError)>,
}

impl Reduction {
    fn combine(self, other: Self) -> Self {
        let error = match (self.error, other.error) {
            (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
            (left, right) => left.or(right),
        };
        Self {
            max_index: self.max_index.max(other.max_index),
            error,
        }
    }
}

struct Pass<'a> {
    bytecode: &'a Bytecode,
    entry_pc: u64,
}

impl Pass<'_> {
    #[inline]
    fn fact(&self, cycle: usize, row: &SourceTraceRow) -> Result<CycleFacts, AdapterError> {
        let pc = row.pc();
        if cycle == 0 && pc != self.entry_pc {
            return Err(AdapterError::EntryPc {
                cycle,
                pc,
                entry_pc: self.entry_pc,
            });
        }
        let index = row.instruction_index();
        if !self
            .bytecode
            .rows()
            .get(index as usize)
            .is_some_and(|instruction| instruction.variant.is_some() && instruction.pc == pc)
        {
            return Err(AdapterError::InstructionIndex { cycle, pc, index });
        }
        let ram_word_index = if matches!(row.ram_access(), RamAccess::NoOp) {
            0
        } else {
            let address = row.ram_address();
            let lowest = self.bytecode.lowest_address();
            if address < lowest {
                return Err(AdapterError::AddressBelowRam { cycle, address });
            }
            if !address.is_multiple_of(8) {
                return Err(AdapterError::UnalignedWordAddress { cycle, address });
            }
            (address - lowest) / 8
        };
        Ok(CycleFacts {
            bytecode_index: index,
            rs1_value: row.rs1_value(),
            rs2_value: row.rs2_value(),
            rd_pre_value: row.rd_pre_value(),
            rd_post_value: row.rd_post_value(),
            ram_word_index,
            ram_pre_value: row.ram_pre_value(),
            ram_post_value: row.ram_post_value(),
            next_pc: row.next_pc(),
        })
    }

    fn drive<C: Consumer<CycleFacts>>(
        &self,
        rows: &[SourceTraceRow],
        start: usize,
        consumer: C,
    ) -> (C::Result, Reduction) {
        let chunks = rows.len().div_ceil(CHUNK_ROWS);
        if chunks > 1 {
            let middle = (chunks / 2) * CHUNK_ROWS;
            let (left, right) = rows.split_at(middle);
            let (left_consumer, right_consumer, reducer) = consumer.split_at(middle);
            let ((left_result, left_reduction), (right_result, right_reduction)) = rayon::join(
                || self.drive(left, start, left_consumer),
                || self.drive(right, start + middle, right_consumer),
            );
            return (
                reducer.reduce(left_result, right_result),
                left_reduction.combine(right_reduction),
            );
        }
        let mut folder = consumer.into_folder();
        let mut reduction = Reduction::default();
        for (offset, row) in rows.iter().enumerate() {
            let cycle = start + offset;
            let fact = match self.fact(cycle, row) {
                Ok(fact) => {
                    reduction.max_index = reduction.max_index.max(fact.ram_word_index);
                    fact
                }
                Err(error) => {
                    if reduction.error.is_none() {
                        reduction.error = Some((cycle, error));
                    }
                    CycleFacts::default()
                }
            };
            folder = folder.consume(fact);
        }
        (folder.complete(), reduction)
    }
}

// Rayon uses opt_len to write directly into the reserved vector. drive splits
// its consumer at the same exact chunk boundaries as the source slice, so the
// collector receives every element once, in order, without intermediate buffers.
struct FactPass<'a, 'b> {
    pass: &'a Pass<'a>,
    rows: &'a [SourceTraceRow],
    reduction: &'b mut Reduction,
}

impl ParallelIterator for FactPass<'_, '_> {
    type Item = CycleFacts;

    fn drive_unindexed<C: UnindexedConsumer<Self::Item>>(self, consumer: C) -> C::Result {
        let (result, reduction) = self.pass.drive(self.rows, 0, consumer);
        *self.reduction = reduction;
        result
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.rows.len())
    }
}
