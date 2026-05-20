use std::io::{Read, Write};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_riscv::{JoltInstructionRow, RV64IMAC_JOLT};
use tracer::instruction::Cycle;

use crate::zkvm::bytecode::{BytecodePreprocessing, PreprocessingError};
use crate::zkvm::ram::RAMPreprocessing;

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct FullProgramPreprocessing {
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
}

impl Default for FullProgramPreprocessing {
    fn default() -> Self {
        Self {
            bytecode: BytecodePreprocessing::default(),
            ram: RAMPreprocessing {
                min_bytecode_address: 0,
                bytecode_words: Vec::new(),
            },
        }
    }
}

impl FullProgramPreprocessing {
    #[tracing::instrument(skip_all, name = "FullProgramPreprocessing::preprocess")]
    pub fn preprocess(
        instructions: Vec<JoltInstructionRow>,
        memory_init: Vec<(u64, u8)>,
        entry_address: u64,
    ) -> Result<Self, PreprocessingError> {
        Ok(Self {
            bytecode: BytecodePreprocessing::preprocess(
                instructions,
                entry_address,
                RV64IMAC_JOLT,
            )?,
            ram: RAMPreprocessing::preprocess(memory_init),
        })
    }

    #[inline(always)]
    pub fn get_pc(&self, cycle: &Cycle) -> usize {
        crate::zkvm::bytecode::get_pc_for_cycle(&self.bytecode, cycle)
    }

    #[inline(always)]
    pub fn entry_bytecode_index(&self) -> usize {
        crate::zkvm::bytecode::entry_bytecode_index(&self.bytecode)
    }
}

#[derive(Debug, Clone)]
pub enum ProgramPreprocessing {
    Full(FullProgramPreprocessing),
}

impl CanonicalSerialize for ProgramPreprocessing {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::Full(full) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                full.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            Self::Full(full) => full.serialized_size(compress),
        }
    }
}

impl Valid for ProgramPreprocessing {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::Full(full) => full.check(),
        }
    }
}

impl CanonicalDeserialize for ProgramPreprocessing {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match tag {
            0 => Ok(Self::Full(FullProgramPreprocessing::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl Default for ProgramPreprocessing {
    fn default() -> Self {
        Self::Full(FullProgramPreprocessing::default())
    }
}

impl ProgramPreprocessing {
    #[tracing::instrument(skip_all, name = "ProgramPreprocessing::preprocess")]
    pub fn preprocess(
        instructions: Vec<JoltInstructionRow>,
        memory_init: Vec<(u64, u8)>,
        entry_address: u64,
    ) -> Result<Self, PreprocessingError> {
        Ok(Self::Full(FullProgramPreprocessing::preprocess(
            instructions,
            memory_init,
            entry_address,
        )?))
    }

    pub fn as_full(&self) -> &FullProgramPreprocessing {
        match self {
            Self::Full(full) => full,
        }
    }

    pub fn bytecode(&self) -> &BytecodePreprocessing {
        &self.as_full().bytecode
    }

    pub fn ram(&self) -> &RAMPreprocessing {
        &self.as_full().ram
    }
}
