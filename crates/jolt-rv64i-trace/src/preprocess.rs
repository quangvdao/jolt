use std::collections::BTreeMap;

use common::{
    constants::RAM_START_ADDRESS,
    jolt_device::{MemoryConfig, MemoryLayout},
};
use jolt_program::image::{decode_elf_with_mode, DecodeMode};
use jolt_riscv::{SourceInstruction, RV64I};
use jolt_rv64i_arith::{Bytecode, Layout};

use crate::AdapterError;

/// Decoded bytecode and nonzero image words, with the ELF entry and a memory
/// configuration whose program size covers the loaded image exactly.
#[derive(Debug)]
pub struct Program {
    pub bytecode: Bytecode,
    pub image: Vec<(u64, u64)>,
    pub entry_pc: u64,
    pub memory_config: MemoryConfig,
}

/// Decodes under RV64I in the requested mode and checks that every decoded
/// instruction has a valid bytecode row. Image bytes are folded in input order,
/// later bytes replacing earlier ones; nonzero little-endian words are sorted
/// by their index relative to the canonical memory layout's lowest address.
pub fn preprocess(
    elf: &[u8],
    mut memory_config: MemoryConfig,
    decode: DecodeMode,
) -> Result<Program, AdapterError> {
    let decoded = decode_elf_with_mode(elf, RV64I, decode)?;
    memory_config.program_size = Some(decoded.program_end - RAM_START_ADDRESS);
    let memory = MemoryLayout::try_new(&memory_config)?;
    let lowest = memory.get_lowest_address();
    let b = decoded
        .instructions
        .len()
        .max(2)
        .next_power_of_two()
        .ilog2() as usize;
    let layout = Layout::new(b, 5, lowest)?;
    let bytecode = Bytecode::preprocess(&decoded.instructions, &layout)?;
    Program::valid_rows(&bytecode, &decoded.instructions)?;
    let image = Program::image_words(&decoded.memory_init, lowest);
    Ok(Program {
        bytecode,
        image,
        entry_pc: decoded.entry_address,
        memory_config,
    })
}

impl Program {
    fn valid_rows(
        bytecode: &Bytecode,
        instructions: &[SourceInstruction],
    ) -> Result<(), AdapterError> {
        for (index, (row, instruction)) in bytecode.rows().iter().zip(instructions).enumerate() {
            if row.variant.is_none() {
                return Err(AdapterError::InvalidBytecodeRow {
                    index,
                    pc: instruction.row().address as u64,
                });
            }
        }
        Ok(())
    }
}

impl Program {
    fn image_words(bytes: &[(u64, u8)], lowest: u64) -> Vec<(u64, u64)> {
        let mut words = BTreeMap::new();
        for &(address, byte) in bytes {
            let relative = address - lowest;
            let word = words.entry(relative / 8).or_insert(0_u64);
            let shift = (relative % 8) * 8;
            *word = (*word & !(0xff << shift)) | (u64::from(byte) << shift);
        }
        words.into_iter().filter(|(_, word)| *word != 0).collect()
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "literal format fixtures must succeed")]
mod tests {
    use super::*;
    use jolt_program::image::decode::decode_instruction;
    use jolt_riscv::RV64IMAC_JOLT;

    #[test]
    fn image_fold_replaces_bytes_omits_zero_words_and_crosses_boundaries() {
        assert_eq!(
            Program::image_words(&[(8, 7), (8, 0), (17, 9), (7, 0xaa), (8, 0xbb), (17, 0)], 0),
            vec![(0, 0xaa00_0000_0000_0000), (1, 0xbb)]
        );
    }

    #[test]
    fn unsupported_decoded_kind_is_rejected_by_validity_stage() {
        let instruction =
            decode_instruction(0x0220_81b3, RAM_START_ADDRESS, false, RV64IMAC_JOLT).unwrap();
        let bytecode =
            Bytecode::preprocess(&[instruction], &Layout::new(1, 5, 0).unwrap()).unwrap();
        assert!(matches!(
            Program::valid_rows(&bytecode, &[instruction]),
            Err(AdapterError::InvalidBytecodeRow {
                index: 0,
                pc: RAM_START_ADDRESS
            })
        ));
    }
}
