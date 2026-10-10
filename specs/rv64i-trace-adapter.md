# Spec: RV64I Trace Adapter

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-10                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

An experiment in RV64I hash-based Jolt proves plain RV64I from one committed table of 256 bits per cycle. `specs/rv64i-binary-arithmetisation.md` defines the `CycleFacts` that fill that table, `specs/rv64i-binary-protocol.md` the statement, the preprocessing and the witness, and `specs/architectural-trace.md` the trace of an execution, one `SourceTraceRow` per executed instruction. This spec adds `jolt-rv64i-trace`, the adapter between them: from an RV64I ELF to its bytecode and program image, and from the rows of one execution to `2^t` cycle facts with the dimensions a statement needs. A trace row and a cycle fact agree almost field for field. The spec fixes the seven places where they do not, each as a rule with one owner, and makes the adapter the arithmetisation's end-to-end test on real programs, with no sum-check involved.

## Intent

### Goal

```rust
// crates/jolt-rv64i-trace. `DecodeMode` is the enum of jolt-program that specs/architectural-trace.md specifies.
pub struct Program {
    pub bytecode: Bytecode,            // row i is instruction i of the decoded list
    pub image: Vec<(u64, u64)>,        // word index, nonzero value; indices strictly increasing
    pub entry_pc: u64,                 // Rv64ProgramImage::entry_address
    pub memory_config: MemoryConfig,   // the caller's, with program_size = Some(program_end - RAM_START_ADDRESS)
}
pub fn preprocess(elf: &[u8], memory_config: MemoryConfig, decode: DecodeMode) -> Result<Program, AdapterError>;

pub struct Execution {
    pub facts: Vec<CycleFacts>,        // 2^log_T of them
    pub log_K_ram: u8,                 // the candidate of rule 2
}
impl Execution {
    pub fn log_T(&self) -> u8;         // from facts.len()
    pub fn final_pc(&self) -> u64;     // next_pc of the last fact
}
pub fn adapt(
    bytecode: &Bytecode, image: &[(u64, u64)], memory_layout: &MemoryLayout, entry_pc: u64,
    rows: &[SourceTraceRow],
) -> Result<Execution, AdapterError>;

#[cfg(feature = "emulator")]
pub fn trace(elf: &[u8], inputs: &[u8], memory_config: &MemoryConfig, decode: DecodeMode)
    -> Result<TraceOutput<OwnedTrace<SourceTraceRow>>, AdapterError>;
```

The library names no type of the verifier or prover crate, and `adapt` takes no commitment scheme. A host composes them in this order, stated here and nowhere else:

1. `preprocess` decodes the ELF under the `RV64I` profile, takes `LowestAddress` from `MemoryLayout::try_new` of the configuration and `b` as the least exponent, at least 1, with `2^b` rows for the instruction list, and builds the image and the bytecode (rules 1, 2, 5), the latter by `Bytecode::preprocess` under `Layout::new(b, 5, LowestAddress)`, of which only `b` and `LowestAddress` are read. The host builds `VerifierPreprocessing::new(program.bytecode, program.image, setup)`.
2. `trace` runs the ELF under `program.memory_config` with `SourceTracerBackend`.
3. `adapt(preprocessing.bytecode(), preprocessing.image(), &device.memory_layout, program.entry_pc, rows)` makes one pass over the rows (rules 1 to 4 and 7).
4. The host builds `Statement { log_T: execution.log_T(), entry_pc, device }` and calls `CheckedInputs::of_statement(&preprocessing, &statement, execution.log_K_ram, execution.final_pc())`.
5. `Rv64iWitness::from_facts(checked.layout().clone(), preprocessing.shared_bytecode().clone(), &execution.facts, checked.initial_ram().to_vec())` fills the rows and replays. It borrows the facts; the host drops them when it returns.
6. `witness.check_outputs(&checked)`, then `prove`.

The ACT4 checking binary stops after step 5. The adapter fills no row, replays nothing, assembles no initial RAM, builds no statement and judges no output.

### Invariants

Each rule names its owner. B1 to B11 are the invariants of `specs/architectural-trace.md`; §7, §8.1, §8.12 and §12 are sections of `specs/rv64i-binary-protocol.md`.

1. **Instruction identity.** `SourceTraceRow::instruction_index` indexes the instruction list the backend decoded (B1), and `Bytecode::preprocess` puts instruction `i` in row `i`. `preprocess` and the backend decode the same ELF in the same `DecodeMode`, so the map is the identity. It is not `(pc − base)/4`: the decoder omits zero halfwords and, in the data-hole mode, inadmissible slots. The fact's `bytecode_index` is by definition `Bytecode::index_of_pc(row.pc())`. `adapt` computes it without the map: it accepts `i = instruction_index()` when `bytecode.rows()[i]` exists, is valid (its `variant` is `Some`) and has `pc == row.pc()`. Valid PCs are unique in a `Bytecode` (`BytecodeError::DuplicatePc`), so an accepted `i` is the map's answer, for one indexed load in place of a `HashMap<u64, usize>` lookup. Anything else is `InstructionIndex { cycle, pc, index }`, with no fallback to the map: a miss means the rows and the bytecode come from different lists. *Owner:* `adapt`, per cycle; `BitsBuilder::bits_row` repeats the range and validity half.

2. **RAM.** A row with an access records `ram_address = ea & !7` and the whole little-endian doubleword there before and after the instruction, for every width (B3), so the adapter keeps no RAM of its own. Both values are copied on every row; without an access `SourceTraceRow::new` has stored zeros and the index is 0. On an access the index is `(ram_address − LowestAddress)/8`, with `LowestAddress = bytecode.lowest_address()`, after two checks: `ram_address < LowestAddress` is `AddressBelowRam { cycle, address }` and `ram_address % 8 != 0` is `UnalignedWordAddress { cycle, address }`. Alignment of a sub-word access is the backend's (`SourceTraceError::MisalignedAccess`) and again `bits_row`'s, which recomputes the address from `rs1_value` and the row's immediate (`AddressOutsideRam`, `UnalignedAccess`, `RamWordIndexMismatch`, in that order). `log_K_ram` is a candidate, not an acceptance: the least `a ≥ 5` with `2^a` at least the I/O mask end `memory_layout.remapped_word_address(RAM_START_ADDRESS)`, one past the last image index and one past the largest index accessed. Its owners decide whether it is admissible, and their checks are not restated: `Layout::new` (the range of `a`, no wrapping of `LowestAddress + 8·2^a`, the 256 columns shared with `b`) and `CheckedInputs::of_statement` (§7 checks 1, 3, 5 and 6, among them `compute_max_ram_k`). The read word of a cycle without an access is RAM word 0, which the replay owns (§8.1); the fact holds zero there and `from_facts` does not compare it. The initial RAM is `CheckedInputs::initial_ram()`, to which the adapter contributes the image: `preprocess` folds `Rv64ProgramImage::memory_init`, the data the decoder returns for each section at or above `RAM_START_ADDRESS` (text included; nothing for a section without file data), by byte address with a later entry replacing an earlier one, into little-endian words at `(address − LowestAddress)/8`, drops zero words and sorts by index. *Owner:* `preprocess` for the image, `adapt` for indices and the candidate, `Layout::new` and `of_statement` for admissibility.

3. **Registers.** The four register words are copied without a branch. For an operand the instruction does not name, the row holds 0 (B2) and `BytecodeRow::from_source` selects register 0. The replay reads 0 there as well: register 0 receives only the increments of cycles whose row has destination `x0` or none, and those facts have `rd_pre_value = rd_post_value = 0`. Both sides of the comparison of `from_facts` (§12) are therefore zero for an operand the row does not have. §12 skips such an operand, except on a `NOOP` row, where it compares all three; the adapter's facts pass either way. "Named" has two definitions, the decoder's operand sets (`jolt-program/src/image/decode.rs`) and the bytecode row's shapes. They coincide on the 50 traced kinds, `FENCE` naming none in both; `ECALL` and `EBREAK` differ (the decoder names `x0` twice, the row nothing) and are never traced. A load to `x0` still records its access, which the `LOADn_X0` variants need. *Owner:* no type pins the agreement of the two decoders; the operand-presence test does, and `from_facts` reports a disagreement on a traced cycle.

4. **Control.** `rows[0].pc()` must equal `entry_pc`, the ELF entry that the host puts in the statement (`EntryPc`). `next_pc` is copied. The backend ends the trace with the first row whose `next_pc` equals its `pc` and pads nothing (B11). `log_T` is the least `t ≥ 1` with `2^t ≥ rows.len()`; an empty trace is `EmptyTrace`, and the upper bound on `t` is `of_statement`'s. A trace of exactly `2^t` rows is converted as it is, with no condition on its last row: `final_pc` is that row's successor, and its validity is §7 check 7. Otherwise the last row must have `next_pc == pc` (`NoStall`), and the rows up to `2^t` are *stall cycles* (§8.1): the last instruction executed again in the state it left. The stall fact is the last fact with `rd_pre_value` set to its `rd_post_value`, and with `rs1_value` set to that value too when `rs1 = rd ≠ x0`. `adapt` builds `Layout::new(b, log_K_ram, LowestAddress)`, returning its `LayoutError`, and checks the stall fact once with `BitsBuilder::bits_row`, `WitnessRow::compute` on `BaseWords::from_facts` and `RowSystem::check` before it replicates it; a failure is `StallNotFixedPoint { cycle, pc, cause }`, and no facts are returned. On rows of the backend the outcome is known in advance: a `JAL` to itself, a taken branch with offset 0 and a `JALR` to itself with `rd = x0` or `rd ≠ rs1` are fixed points, and a `JALR` to itself with `rd = rs1 ≠ x0` is one exactly when `((pc + 4) + imm) & !1 == pc`, that is for the immediates −4 and −3; with immediate 0 its second execution leaves. The check decides the fixed point only; the read of RAM word 0 on a stall cycle is the replay's. *Owner:* `adapt`; the definition is §8.1's.

5. **Instruction set.** The `RV64I` profile admits the 52 kinds of `SourceExtension::Rv64I`. Under `DecodeMode::Strict`, the default, `preprocess` returns the decoder's rejection of any other word as `AdapterError::Program` (`IllegalSourceInstruction`, `IllegalCompressedInstruction` or `MalformedImage`; D1 of `specs/architectural-trace.md` says which word gets which). In the data-hole mode an inadmissible slot is absent from the list and its rejection moves to the fetch (`SourceTraceError::PcOutsideProgram`). `preprocess` then checks that each of the first `instructions.len()` bytecode rows is valid (`InvalidBytecodeRow { index, pc }`). `BytecodeRow::from_source` returns the invalid row for an unsupported instruction; behind the present decoder the check never fires, and it is kept as the adapter's own statement that every listed instruction has a row. `ECALL` and `EBREAK` have valid rows, and the backend rejects them when reached (`SourceTraceError::UnsupportedInstruction { pc, kind }`). The source trace has no virtual sequences. *Owner:* the decoder at preprocessing, the backend at the cycle, rule 1 as the backstop.

6. **I/O and termination.** The adapter handles no input, output or advice. The backend reads inputs from the device, where `CheckedInputs::initial_ram()` places the same bytes through `PublicInitialRam::inputs_only`; outputs are ordinary stores; the panic and termination words are write-once, with pre-value 0 and the stored doubleword as post-value (B9). All become ordinary facts, and `from_facts` compares their `ram_pre_value` with its replay from the canonical initial RAM, so a disagreement between the backend's I/O memory and the statement's shows at the first access. The device records less than the trace: only `panic`, set by a byte store at `memory_layout.panic` whatever the value, from which `PublicIoMemory` states the panic word as `panic as u64` and the termination word as 1 exactly when `panic` is false. Agreement of the statement with the final RAM is thus a property of the run, not of the types. `Rv64iWitness::check_outputs` (§12) reports it to the host from the final RAM its constructor retains; the binding check is the protocol's (§8.12). *Owner:* the prover crate; the adapter keeps no summary of stores.

7. **Cost.** One pass over `rows` in parallel chunks of `2^16`. A fact is a function of its own row and one bytecode row, and is written once into a vector allocated at `2^t`. A chunk returns two reductions of constant size: the largest word index accessed, and the smallest failing cycle with its error. Within a cycle the checks run in the order `EntryPc`, `InstructionIndex`, `AddressBelowRam`, `UnalignedWordAddress`, and the error returned is that of the smallest absolute cycle, whatever the scheduling. `EmptyTrace` precedes the pass; `NoStall`, the `LayoutError` and `StallNotFixedPoint` follow it, in that order. *Owner:* `adapt`.

### Non-Goals

Filling rows, replaying words, checking outputs and proving: `from_facts`, `check_outputs` and `prove` own them. Building or checking a statement: the host and `CheckedInputs::of_statement`. Checking the trace's semantics: PC continuity and the ISA are invariants of the backend (B1, B5), and the rows judge them. Advice, M, compressed code and inlines. The speed of tracing. Returning every guest fault as an error: `trace` returns what the backend returns, and a memory fault on which the emulator asserts aborts the process.

## Evaluation

### Acceptance Criteria

In every positive criterion, "passes the rows" means: steps 1 to 5 of Goal succeed and, for each cycle `j`, `RowSystem::check` accepts `WitnessRow::compute` of the fetched bytecode row, `CycleWords::base_words` and `bits[j]`.

- [ ] **Hand-assembled programs.** Four programs assembled in the test and wrapped by `test_elf::build_elf64` pass the rows: an ALU loop with every register alias (`rs1 = rs2`, `rd = rs1`, `rd = rs2`, all three equal), at `2^t − 1`, `2^t` and `2^t + 1` executed rows for one `t`; loads and stores of every width and offset, with a load to `x0`; call frames; and one that reads its input, writes outputs, stores 1 to the termination word and stalls on `jal x0, 0`. The last runs with no advice capacity and a nonzero first input doubleword, so that RAM word 0 is nonzero on every cycle without an access, stall cycles included, where the fact holds zero; `check_outputs` passes on it. `of_statement` admits each candidate `log_K_ram`. The witness's `final_ram` equals `TraceOutput::final_memory` at and above the mask end.
- [ ] **Stall forms.** At a length that needs padding, programs that end in `jal x1, 0`, in a taken `beq x0, x0, 0`, in `jalr x1, 0(x2)` and `jalr x0, 0(x1)` to their own PC, and in `jalr x1, -4(x1)` and `jalr x1, -3(x1)` to their own PC pass the rows. `jalr x1, 0(x1)` to its own PC returns `StallNotFixedPoint`; at exactly `2^t` rows the same program passes the rows with `final_pc` its PC.
- [ ] **Bare `no_std` ELFs.** Three Rust fixtures (`#![no_std]`, `#![no_main]`, a `_start` in assembly that sets the stack pointer, a panic handler that stores to the panic word and jumps to itself, their own linker script, no SDK runtime): an input-driven loop, a byte-wise checksum over a buffer with sub-word loads and stores, a recursion. Each writes its result to the output region, so that its memory operations survive optimisation, stores 1 to the termination word and jumps to itself. The complete linked ELF of each passes `preprocess` under `DecodeMode::Strict`, the gate that nothing outside RV64I was linked, then the rows and `check_outputs`.
- [ ] **ACT4**, conditional on the decode mode of Owed Changes. Every ELF that `run.sh --match 'I-*' --expect <n>` selects from the suite generated at the pinned commit `a7c99303516f4e668f7488f172043392e23b9dfd` of `third-party/riscv-arch-test`, decoded in the data-hole mode, passes the rows and has `final_ram` equal to 1 at the word of the ELF symbol `tohost`. `tests/arch-tests/smoke/fail.S`, built with `-march=rv64i` and decoded strictly, reaches its HTIF store and makes the checking binary exit with its HTIF-failure status.
- [ ] **Identity.** On every cycle of every trace above, `bytecode_index` equals `Bytecode::index_of_pc(pc)`. One program jumps over a zero word and executes past it, and one, in the data-hole mode, over a `MUL` word; in both the index differs from `(pc − base)/4`.
- [ ] **Operand presence.** For one decoded instruction of each of the 50 traced kinds, with every register field nonzero, the bytecode row's `rs1`, `rs2`, `rd` equal the decoder's operand where it names one and are 0 where it does not, and the row's variant has an access exactly for the eleven load and store kinds. On the traces above, `registers()` and `ram_access()` of every row agree with the fetched bytecode row in the same sense.
- [ ] **Image and exponent.** A literal `memory_init` with two entries for one byte address, a later zero byte that makes a word zero, and a run of bytes across a word boundary folds to the words written in the test. On hand-built rows, a largest accessed index of `2^a − 1` gives the candidate `a` and `2^a` gives `a + 1`; an index of `2^47` gives 48, which `Layout::new` rejects for every `b` (`BitsRowOverflow`); an access above `heap_end` converts, and `of_statement` rejects the candidate (`RamTooLarge`).
- [ ] **Determinism.** On a trace of more than `2^17` rows, `adapt` in `rayon` pools of 1 and of 8 threads returns byte-identical facts and the same candidate, with the largest access placed in each of two chunks in turn. With a failing row of a different kind planted in each of two chunks, both pools return the error of the smaller cycle.
- [ ] **Rejections.** Each on a trace of the programs above with one change, asserting the named error and the stage that returns it:
  1. An `instruction_index` replaced by the index of another valid row, of a padding row, and by `rows().len()`: `InstructionIndex`. A bytecode built from the list with one instruction removed: `InstructionIndex` at the first fetched row whose index shifted.
  2. A load's `ram_address` lowered below `LowestAddress`: `AddressBelowRam`. Raised by 4: `UnalignedWordAddress`. Raised by 8, on a load whose word and the next are zero in the initial RAM and never stored: `adapt` succeeds and `from_facts` returns `RamWordIndexMismatch` for that cycle. One image byte changed in a word whose first access is a load: `from_facts` returns its pre-state mismatch (§12) at that cycle, field `ram_pre_value`.
  3. A taken `BEQ` with nonzero operands, rebuilt through `SourceTraceRow::new` with `rs2` absent: `adapt` succeeds and `from_facts` returns the mismatch at that cycle, field `rs2_value`.
  4. An entry PC that is not `rows[0].pc()`: `EntryPc`. No rows: `EmptyTrace`. A trace cut to `2^t − 1` rows before its stall: `NoStall`; cut to `2^t` rows: accepted, with `final_pc` the successor of its last row.
  5. Strict mode, from `preprocess`: a text word that is `MUL`, `LR.D` or `CSRRW` (`IllegalSourceInstruction`), a word of an unassigned opcode (`MalformedImage`), a nonzero compressed halfword (`IllegalCompressedInstruction`), an instruction word placed after a single zero halfword (`MalformedImage`). From `trace`: a reached `ECALL` and a reached `EBREAK` (`UnsupportedInstruction`); unreached, both are accepted. Data-hole mode: the `MUL` word unreached is accepted, and reached it is `PcOutsideProgram` from `trace`. A hand-built list with a kind outside RV64I, passed to the validity check: `InvalidBytecodeRow`.

The output cases (a changed output byte, a missing or wrong termination store, a panic store) belong to `check_outputs` and are tested in `jolt-rv64i-prover`. This crate tests only that the fourth program passes it.

### Testing Strategy

Ground truth is independent of the adapter: the emulator's final memory, `PublicIoMemory`, the HTIF word of an ACT4 test, and `RowSystem::check` on a witness whose words `from_facts` replayed. There is no ZK mode, and no existing test changes. The checks are five separate commands:

```text
cargo nextest run -p jolt-rv64i-trace --features emulator --cargo-quiet
make arch-tests-rv64i          # generation, build of rv64i-arch-check, run.sh --match 'I-*' --expect <n>
make arch-tests-rv64i-smoke    # fail.S at -march=rv64i through rv64i-arch-check
make arch-tests-64imac         # the existing gate, unchanged
cargo bench -p jolt-eval --bench rv64i_trace_adapt
```

The nextest run needs the crate's `emulator` feature for `trace` and `tracer`'s `test-utils` for `test_elf::build_elf64`, which the crate enables on its dev-dependency as `jolt-eval` does on its dependency. The three guest fixtures are one package under `tests/fixtures/`, built once, by the one test function that checks all three, into a target directory of its own, so that no two builds compete.

*ACT4, what exists.* The submodule `third-party/riscv-arch-test`, pinned and not checked out in this worktree; the model directory `tests/arch-tests/jolt/`, whose configuration `jolt-rv64imac` declares I, M, A and C, whose `link.ld` collects `.text.init` and `.text.rvtest` and reserves 4 KiB of stack past the last section, and whose `rvmodel_macros.h` ends a test with a store of the HTIF word to `tohost` and `j 1b`; `tests/arch-tests/run.sh`, which runs every ELF of the work directory through `jolt-emu` and reads exit status 0 as a pass; `skip.txt`, with base names of the form `Zicsr-csrrc-00`. Nothing in it calls `decode_elf` or `SourceTracerBackend`.

*ACT4, what is added.* Generation and the model directory are reused unchanged. `run.sh` gains `--match <glob>` on the base name and `--expect <n>`, and fails when no ELF matches or when the number that match is not `n`; `n` is fixed in the Makefile beside the submodule commit at the first generation. The checking binary `rv64i-arch-check` exits 0 when its argument meets the ACT4 criterion, with one status for an HTIF word other than 1 and another for every other failure; the smoke target asserts the first, so that a decode failure does not pass for it. Its `MemoryConfig` takes `program_size` from `preprocess` and a stack that covers the 4 KiB of the linker script. The workflow's triggers gain the paths of this crate, of `jolt-rv64i-arith`, of the two protocol crates and of `jolt-program`.

*ACT4, what is not shown.* No file of the repository shows how many I tests the pinned suite generates, that their ELFs, built under a configuration that declares M, A and C, execute only RV64I encodings, that their text sections hold data, or that their accesses clear the stack canary of the device layout under the backend. The first generation decides each, and each failure is visible: a wrong count, `PcOutsideProgram`, a strict-decode error, an abort.

### Performance

The timed region is `adapt` alone, on rows traced beforehand: the allocation and first touch of the facts, the pass with its two reductions, the stall check and the padding. Decoding, `preprocess`, emulation, `of_statement` and `from_facts` are outside it. Executed and padded counts are reported separately, because padding nearly doubles the named workloads: the three programs of `source_trace_gen` execute `2^22 + 4` (`alu`), `2^22 + 6` (`memory`) and 4,325,382 (`call_frame`) rows, and each pads to `2^23` facts, 576 MiB. The pass reads 80 bytes per executed row and writes 72 per padded cycle, 112 bytes per padded cycle on the first two. The target is at most 8 ns per padded cycle on one thread, scaling with threads up to memory bandwidth. It is a target, not a measurement: it presumes 14 GB/s through pages touched for the first time, and the first measurement replaces it. The rows and the facts are both live during `adapt`; the facts, `72·2^t` bytes, are the only allocation of trace size the adapter makes, and their lifetime is the caller's (step 5).

New `jolt-eval` objective `rv64i_trace_adapt`: for each of the three programs, the median time of `adapt` per padded cycle and per executed row, on one thread and on the default pool, with the peak allocation. No existing objective moves.

## Design

### Architecture

```text
jolt-program        decode, SourceTraceRow
jolt-rv64i-arith    Bytecode, CycleFacts, Layout, BitsBuilder, RowSystem
        ▲
jolt-rv64i-trace    preprocess, adapt
        ▲  feature `emulator`:    tracer, `trace`
        ▲  feature `arch-check`:  jolt-rv64i-verifier, jolt-rv64i-prover, the checking binary
host                Statement, CheckedInputs, from_facts, check_outputs, prove
```

The adapter is a crate of its own. Its library needs `jolt-rv64i-arith`, `jolt-program`, `common`, `rayon` and `thiserror`, and `tracer` under `emulator`; neither protocol crate depends on it. Three contracts decide the placement: the adapter stops at the inputs of `from_facts`; ELF, device and backend types stay out of `jolt-rv64i-arith`, which carries the verifier-closure lints and has `jolt-program` as a dev-dependency only; and the emulator stays out of the prover's dependency closure and cannot enter the verifier's. The checking binary and the tests do run steps 4 and 5, so the feature `arch-check` (the binary's `required-features`) and the dev-dependencies add the two protocol crates, the prover's with `test-utils`: `CheckedInputs` is typed by a commitment scheme through `VerifierPreprocessing<S>`, and they name the test scheme `TransparentBits`, whose setup is `()`, for that type alone.

Files: `src/{lib, preprocess, adapt, error}.rs`, `src/bin/rv64i-arch-check.rs`, `tests/{programs, guests, rejections}.rs`, `tests/fixtures/`, `jolt-eval/benches/rv64i_trace_adapt.rs`; edits to `Makefile`, `tests/arch-tests/run.sh` and `.github/workflows/arch-tests.yml`.

### Alternatives Considered

1. **A module of `jolt-rv64i-prover` or of `jolt-rv64i-arith`.** Rejected by the three contracts above.
2. **`adapt` over `&VerifierPreprocessing<S>`, calling `of_statement` itself.** Rejected: conversion needs no commitment scheme, and the result would hold a layout beside its own exponent and a copy of the initial RAM. The host calls the one checker once.
3. **An adapter that replays RAM and builds `Rv64iWitness` directly.** Rejected: the replay has one owner, `from_facts`, and a second would owe §8.1 itself.
4. **Output consistency in the adapter, from the last store to each I/O word.** Rejected. The supplied post-values are not the committed increments: an `sd` whose RAM pair is changed from `(0, 1)` to `(1, 0)` keeps its increment and reverses the overlay. The reduction is also not of constant size. The final RAM has one producer, the witness's replay.
5. **A second entry point writing `BitsRow`s from trace rows.** Deferred until a measurement asks for it. It saves the facts vector: `72·T` bytes of peak and `144·T` bytes of traffic, the write and the read by the fill; 576 MiB and 1,152 MiB at `T = 2^23`. It still writes `32·T` bytes of rows, and the replay keeps its `40·T` bytes of `CycleWords`. The geometry of a row depends on `log_K_ram`, which is a result of the pass, so the fusion needs a bound from the caller or a sizing prepass over the addresses; `compute_max_ram_k` as the bound enlarges every RAM-side table and can exceed the 256 columns. It also gives up the comparison of facts with the replay.
6. **Padding with a no-op row appended to the bytecode.** Rejected: it changes the program and its digest, and the last executed cycle cannot reach it, since its own row fixes its successor.

## Documentation

No book change: the experiment has no user-facing surface. Rustdoc on `adapt` states rules 1 to 4 and 7, on `preprocess` rule 5 and the image fold. `tests/arch-tests/README.md` gains the two targets.

## Execution

Two PRs. The first adds `DecodeMode` to `jolt-program` and threads it through `tracer` (Owed Changes), with strict behaviour unchanged. The second adds this crate: `preprocess` with rule 5 and the image; the per-cycle function with rules 1 to 3; padding; the hand-assembled programs and the rejections; the fixtures; the checking binary, the runner flags and the Makefile targets.

The fixtures are built with `cargo build --release --target riscv64imac-unknown-none-elf` and `-C target-feature=-m,-a,-c,-zmmul,-zca,-zaamo,-zalrsc`, a target that `rust-toolchain.toml` installs. The flags govern the fixtures' own code; `core` arrives compiled with M and C, so the source must pull nothing from it, and the strict gate of the criterion tests that. If the gate cannot be met, the pinned toolchain offers no second Rust recipe (a `core` rebuilt for RV64I needs `rust-src` and the unstable `build-std`), and the fallback is the same three programs in C, compiled at `-march=rv64i -nostdlib` by the `riscv-none-elf-gcc` of `scripts/bootstrap` and run from a make target instead of nextest.

## Owed Changes

What the adapter relies on outside its own crate. Only the decode mode is still to be written.

**Protocol.** Four contracts, each stated in `specs/rv64i-binary-protocol.md` and implemented in the two protocol crates. The acceptance above depends on all four, and a change to one is a change to this spec.

1. *Stall cycle* (§8.1, "A trace shorter than `2^t`"). The repetition of rule 4, with `final_pc` the stall PC; a trace of exactly `2^t` cycles needs no fixed point.
2. *Final RAM and outputs* (§12). `Rv64iWitness` keeps the RAM of its constructor's replay as `final_ram`, a dense vector of `2^a` words that `RamValFinal` reads, and `check_outputs` compares it on the I/O mask with `PublicIoMemory`.
3. *Facts against replay* (§12). Inside its replay and before the update of each cycle, `from_facts` compares `rs1_value`, `rs2_value` and `rd_pre_value` with the replayed registers, for the operands the fetched row has (all three on a `NOOP` row), and `ram_pre_value` with the replayed word on a cycle whose variant has an access. The first difference is `FactMismatch` with the cycle, the field and both values. There is no second replay anywhere.
4. *One bytecode table* (§7). `VerifierPreprocessing` holds an `Arc<Bytecode>`, and `shared_bytecode()` gives the witness the same allocation.

**Arithmetisation crate.** Rustdoc only, on `CycleFacts` and on `BitsBuilder::bits_row`: on a cycle without an access the three RAM fields are zero by convention, `bits_row` ignores them, and they are not the cycle's `RamReadValue`. Nothing else is asked of the crate: `Bytecode::preprocess` keeps its `&Layout`, rule 5 keeps its own validity check, and no replay is exported.

**Decoder and backend** (`jolt-program`, `tracer`).

1. *Decode mode; the ACT4 criterion depends on it.* `DecodeMode`, `decode_elf_with_mode` and `SourceTracerBackend::with_decode_mode` are defined in `specs/architectural-trace.md` (Goal, and invariants D1 to D6): what `Strict` does with each class of word, the slot rule of `DataHoles`, and the one mode shared by the backend and by whoever resolves its rows. `preprocess` and `trace` here pass the caller's one value to `decode_elf_with_mode` and to the backend, so rule 1 holds in either mode; a hole is an absent PC, not an invalid row, and `Bytecode::index_of_pc` has no entry for it.

*Later, not required.* An address in `ProgramError::IllegalSourceInstruction`. Typed `SourceTraceError`s for the memory faults on which `Mmu::assert_effective_address` panics today. Emulator memory sizing as an explicit choice of the caller, where an ELF with a `tohost` symbol now switches `Emulator::setup_program` to its test sizing.

## References

- `specs/rv64i-binary-arithmetisation.md`; `crates/jolt-rv64i-arith/src/{bytecode, cycle, layout, words}.rs`: `Bytecode`, `CycleFacts`, `BitsBuilder`, `Layout::new`, the five obligations on `BaseWords`.
- `specs/rv64i-binary-protocol.md` §7, §8.1, §8.12, §12: statement checks, the canonical initial RAM, the witness contract and the stall cycle, the output check, `Rv64iWitness`.
- `specs/architectural-trace.md`; `tracer/src/source_trace.rs`; `crates/jolt-program/src/execution/trace/source_row.rs`; `crates/jolt-program/src/image/{elf, decode}.rs`; `crates/jolt-riscv/src/profile.rs`.
- `common/src/jolt_device.rs`: `MemoryConfig`, `MemoryLayout`, `JoltDevice`; `PublicIoMemory` and `PublicInitialRam` of `jolt-program`.
- `specs/act4-tests.md`; `Makefile`; `tests/arch-tests/`; `.github/workflows/arch-tests.yml`.
- `jolt-eval/src/objective/performance/source_trace_gen.rs`.
