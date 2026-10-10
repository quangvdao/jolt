//! Checks an RV64I ELF's replayed cycle rows and final HTIF word.
#![forbid(unsafe_code)]

use std::{
    error::Error,
    io::{Error as IoError, ErrorKind},
    path::PathBuf,
    process::ExitCode,
    sync::Arc,
};

use common::jolt_device::MemoryConfig;
use jolt_program::image::DecodeMode;
use jolt_rv64i_arith::{RowSystem, WitnessRow};
use jolt_rv64i_prover::{commitment::transparent::TransparentBits, plane::Rv64iWitness};
use jolt_rv64i_trace::{adapt, preprocess, trace};
use jolt_rv64i_verifier::{
    preprocessing::VerifierPreprocessing,
    statement::{CheckedInputs, Statement},
};
use object::{File as ObjectFile, Object, ObjectSymbol};

struct Options {
    elf: PathBuf,
    decode: DecodeMode,
}

impl Options {
    fn parse() -> Result<Self, IoError> {
        let mut decode = DecodeMode::DataHoles;
        let mut elf = None;
        for argument in std::env::args_os().skip(1) {
            if argument == "--strict" {
                decode = DecodeMode::Strict;
            } else if argument.to_string_lossy().starts_with('-') || elf.is_some() {
                return Err(IoError::new(
                    ErrorKind::InvalidInput,
                    "usage: rv64i-arch-check [--strict] ELF",
                ));
            } else {
                elf = Some(PathBuf::from(argument));
            }
        }
        Ok(Self {
            elf: elf.ok_or_else(|| {
                IoError::new(
                    ErrorKind::InvalidInput,
                    "usage: rv64i-arch-check [--strict] ELF",
                )
            })?,
            decode,
        })
    }

    #[expect(clippy::print_stderr, reason = "the CLI reports the failing HTIF word")]
    fn run(self) -> Result<bool, Box<dyn Error>> {
        let elf = std::fs::read(&self.elf)?;
        let object = ObjectFile::parse(elf.as_slice())?;
        let tohost = object
            .symbols()
            .find(|symbol| symbol.name().is_ok_and(|name| name == "tohost"))
            .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "ELF has no tohost symbol"))?
            .address();
        let program = preprocess(
            &elf,
            MemoryConfig {
                max_input_size: 0,
                max_output_size: 0,
                max_trusted_advice_size: 0,
                max_untrusted_advice_size: 0,
                stack_size: 4096,
                heap_size: 1 << 20,
                program_size: None,
            },
            self.decode,
        )?;
        let output = trace(&elf, &[], &program.memory_config, self.decode)?;
        let preprocessing: VerifierPreprocessing<TransparentBits> =
            VerifierPreprocessing::new(program.bytecode, program.image, ())?;
        let execution = adapt(
            preprocessing.bytecode(),
            preprocessing.image(),
            &output.device.memory_layout,
            program.entry_pc,
            output.trace.rows(),
        )?;
        let statement = Statement {
            log_T: execution.log_T(),
            entry_pc: program.entry_pc,
            device: output.device,
        };
        let checked = CheckedInputs::of_statement(
            &preprocessing,
            &statement,
            execution.log_K_ram,
            execution.final_pc(),
        )?;
        let witness = Rv64iWitness::from_facts(
            checked.layout().clone(),
            Arc::clone(preprocessing.shared_bytecode()),
            &execution.facts,
            checked.initial_ram().to_vec(),
        )?;
        drop(execution);
        let system = RowSystem::new(&witness.layout);
        for (bits, words) in witness.bits.iter().zip(witness.words.iter()) {
            let index = witness.layout.bytecode_index(bits) as usize;
            let row = witness.bytecode.rows().get(index).ok_or_else(|| {
                IoError::new(ErrorKind::InvalidData, "replayed bytecode index is absent")
            })?;
            let variant = row.variant.ok_or_else(|| {
                IoError::new(ErrorKind::InvalidData, "replayed bytecode row is invalid")
            })?;
            let base = words.base_words(variant.is_store(), witness.layout.inc(bits));
            system.check(&WitnessRow::compute(&witness.layout, row, &base, bits))?;
        }
        let index = statement
            .device
            .memory_layout
            .remapped_word_address(tohost)?;
        if tohost % 8 != 0 {
            return Err(IoError::new(ErrorKind::InvalidData, "tohost is not word aligned").into());
        }
        let value = witness
            .final_ram
            .get(usize::try_from(index)?)
            .ok_or_else(|| {
                IoError::new(ErrorKind::InvalidData, "tohost is outside the replayed RAM")
            })?;
        if *value != 1 {
            eprintln!("HTIF failure: tohost at {tohost:#x} contains {value:#x}");
        }
        Ok(*value == 1)
    }
}

#[expect(clippy::print_stderr, reason = "the CLI reports errors before exiting")]
fn main() -> ExitCode {
    match Options::parse()
        .map_err(|error| Box::new(error) as Box<dyn Error>)
        .and_then(Options::run)
    {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("RV64I check failed: {error}");
            ExitCode::from(2)
        }
    }
}
