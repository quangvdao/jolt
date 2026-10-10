#![cfg(feature = "emulator")]
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "fixture failures fail the enclosing test"
)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use common::{constants::RAM_START_ADDRESS, jolt_device::MemoryConfig};
use jolt_program::image::DecodeMode;
use jolt_rv64i_arith::{RowSystem, WitnessRow};
use jolt_rv64i_prover::{commitment::transparent::TransparentBits, plane::Rv64iWitness};
use jolt_rv64i_trace::{adapt, preprocess, trace};
use jolt_rv64i_verifier::{
    preprocessing::VerifierPreprocessing,
    statement::{CheckedInputs, Statement},
};

#[test]
#[ignore = "Rust 1.95 recipe emits compressed code at 0x80000018 in input-loop despite -m,-a,-c; riscv-none-elf-gcc is absent. Rerun: CARGO_TARGET_DIR=/Users/quangdao/Documents/SNARKs/jolt-wt/trace-target cargo nextest run -p jolt-rv64i-trace --features emulator --run-ignored only --cargo-quiet -E 'binary(guests)'"]
fn bare_rust_guests_pass_strict_decode_rows_and_outputs() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let target =
        PathBuf::from(std::env::var_os("CARGO_TARGET_DIR").unwrap()).join("rv64i-fixtures");
    let build = Command::new(env!("CARGO"))
        .arg("build")
        .arg("--manifest-path")
        .arg(fixtures.join("Cargo.toml"))
        .args([
            "--release",
            "--bins",
            "--target",
            "riscv64imac-unknown-none-elf",
        ])
        .env("CARGO_TARGET_DIR", &target)
        .env("RUSTFLAGS", "-C target-feature=-m,-a,-c")
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "bare guest build failed:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
    for (name, input, expected) in [
        ("input-loop", 7_u64, 21_u64),
        ("byte-checksum", 0, 2016),
        ("recursion", 6, 22),
    ] {
        let elf = std::fs::read(
            target
                .join("riscv64imac-unknown-none-elf/release")
                .join(name),
        )
        .unwrap();
        let program = preprocess(
            &elf,
            MemoryConfig {
                max_input_size: 8,
                max_output_size: 8,
                max_trusted_advice_size: 0,
                max_untrusted_advice_size: 0,
                stack_size: 4096,
                heap_size: 4096,
                program_size: None,
            },
            DecodeMode::Strict,
        )
        .unwrap_or_else(|error| panic!("{name}: strict decode: {error}"));
        let output = trace(
            &elf,
            &input.to_le_bytes(),
            &program.memory_config,
            DecodeMode::Strict,
        )
        .unwrap_or_else(|error| panic!("{name}: trace: {error}"));
        assert!(!output.device.panic, "{name}");
        assert_eq!(output.device.outputs, expected.to_le_bytes(), "{name}");
        let preprocessing =
            VerifierPreprocessing::<TransparentBits>::new(program.bytecode, program.image, ())
                .unwrap();
        let execution = adapt(
            preprocessing.bytecode(),
            preprocessing.image(),
            &output.device.memory_layout,
            program.entry_pc,
            output.trace.rows(),
        )
        .unwrap();
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
        )
        .unwrap();
        let witness = Rv64iWitness::from_facts(
            checked.layout().clone(),
            Arc::clone(preprocessing.shared_bytecode()),
            &execution.facts,
            checked.initial_ram().to_vec(),
        )
        .unwrap();
        let system = RowSystem::new(&witness.layout);
        for (cycle, bits) in witness.bits.iter().enumerate() {
            let row = &witness.bytecode.rows()[witness.layout.bytecode_index(bits) as usize];
            let base = witness.words[cycle]
                .base_words(row.variant.unwrap().is_store(), witness.layout.inc(bits));
            system
                .check(&WitnessRow::compute(&witness.layout, row, &base, bits))
                .unwrap_or_else(|error| panic!("{name}: row {cycle}: {error}"));
        }
        witness.check_outputs(&checked).unwrap();
        let final_bytes: BTreeMap<_, _> = output
            .final_memory
            .unwrap()
            .bytes
            .into_iter()
            .map(|(offset, byte)| (RAM_START_ADDRESS + offset, byte))
            .collect();
        for (index, &word) in witness
            .final_ram
            .iter()
            .enumerate()
            .skip(checked.io().io_mask_end as usize)
        {
            let address = witness.layout.lowest_address() + 8 * index as u64;
            let expected = u64::from_le_bytes(std::array::from_fn(|offset| {
                final_bytes
                    .get(&(address + offset as u64))
                    .copied()
                    .unwrap_or(0)
            }));
            assert_eq!(word, expected, "{name}: final RAM word {index}");
        }
        for (source, fact) in output.trace.rows().iter().zip(&execution.facts) {
            assert_eq!(
                fact.bytecode_index as usize,
                preprocessing.bytecode().index_of_pc(source.pc()).unwrap(),
                "{name}"
            );
        }
    }
}
