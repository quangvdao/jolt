# Spec: RV64I Arithmetisation over the Binary Field

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-10                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

This spec adds `jolt-rv64i-arith`, the constraint system of an experiment in RV64I hash-based Jolt. The experiment proves RV64I on the restricted machine defined under Goal, over the binary field `F128`, with one cycle per executed instruction and 256 committed bits per cycle (the table `Bits`). Its witness generator, Spartan relations and routers must read the same columns, the same decode table and the same rows. The crate holds those definitions as data, a generator of the committed bits and a per-cycle checker, tested against an independent interpreter before any prover exists.

## Intent

### Goal

Define the committed layout, the bytecode row, the decode table, the per-cycle R1CS witness and the constraint rows in one crate with no sum-check and no commitment.

The machine is one hart executing RV64I as the unprivileged manual defines it, except as follows. The program is the public bytecode, fetched by PC. RAM is flat: `K_ram` little-endian doublewords at `[LowestAddress, LowestAddress + 8·K_ram)`, with no devices. There are no traps: `ECALL` and `EBREAK` are self-loops that change no state, `FENCE` does nothing, and a cycle has no witness if its data access is misaligned or outside RAM, or if its next PC, the last cycle's included, is not the PC of a valid bytecode row.

The public surface, abridged to what other crates call:

```rust
// layout: the only owner of column positions
pub type BitsRow = [u64; 4];                               // column c is bit c % 64 of word c / 64
pub const fn chunk_indicators(index_bits: usize) -> usize; // n(a) = 15 * (a / 4) + 2^(a % 4) - 1
pub struct Chunk { /* private */ }                         // new(), start(), bits(), stored(), full(), digit_bit(), write_digit()
impl Layout {
    pub fn new(log_K_bytecode: usize, log_K_ram: usize, lowest_address: u64) -> Result<Self, LayoutError>;
    pub fn bytecode_ra(&self) -> &[Chunk];  pub fn ram_ra(&self) -> &[Chunk];  pub fn pos_ra(&self) -> [Chunk; 2];
    // and keys_differ(), should_branch(), jalr_low_bit(); write_bytecode_index(), write_ram_index(), write_pos()
}
// variant, bytecode
#[repr(u8)] pub enum Variant { ADD = 0, /* 58 variants, fixed indices */ LOAD8_X0 = 57 }
impl Variant { pub fn line(self) -> &'static Line; }
pub struct BytecodeRow { pub variant: Option<Variant>, pub pc: u64, pub imm: u64, pub fall_through_pc: u64,
                         pub pc_plus_imm: u64, pub rs1: u8, pub rs2: u8, pub rd: u8 }
impl Bytecode {
    pub fn preprocess(instructions: &[jolt_riscv::SourceInstruction], layout: &Layout) -> Result<Self, BytecodeError>;
    pub fn rows(&self) -> &[BytecodeRow];                  // 2^log_K_bytecode rows
    pub fn final_pc_index(&self, final_pc: u64) -> Result<usize, BytecodeError>;
}
// decode: the table as data
pub struct Term { /* private */ }                          // new(), source(), from(), to(), length(), fill()
impl Term { pub fn eval(self, sources: &Sources) -> u64;  pub fn wires(self) -> impl Iterator<Item = Wire>; }
pub type Form = &'static [Term];
pub struct Line { pub rails: Rails, pub rd_expected: Form, pub next_pc_expected: Form, pub control: Form,
                  pub shift: Option<Shift>, pub access: Option<Access>, pub branch: Option<BranchCondition> }
// words: one cycle, on the stack
pub struct BaseWords { pub rs1_value: u64, pub rs2_value: u64, pub rd_write_value: u64,
                       pub ram_read_value: u64, pub next_pc: u64 }
pub struct WitnessRow(pub [u64; 16]);                      // 1,024 columns; column 0 is ONE; words 12..16 are the BitsRow
impl WitnessRow { pub fn compute(layout: &Layout, row: &BytecodeRow, base: &BaseWords, bits: &BitsRow) -> Self; }
// cycle: committed bits from plain facts
pub struct CycleFacts { pub bytecode_index: u32, pub rs1_value: u64, pub rs2_value: u64,
                        pub rd_pre_value: u64, pub rd_post_value: u64, pub ram_word_index: u64,
                        pub ram_pre_value: u64, pub ram_post_value: u64, pub next_pc: u64 }
impl<'a> BitsBuilder<'a> {
    pub fn new(layout: &'a Layout, bytecode: &'a Bytecode) -> Result<Self, WitnessError>;
    pub fn bits_row(&self, facts: &CycleFacts) -> Result<BitsRow, WitnessError>;
    pub fn fill(&self, facts: &[CycleFacts], out: &mut [BitsRow]) -> Result<(), CycleError>;
}
// rows: as data
pub struct LaneRows { pub a: Lane, pub b: Lane, pub c: Lane, pub ab_mask: u64, pub group: RowGroup }
impl LaneRows { pub fn values(&self, z: &WitnessRow) -> [u64; 3]; }   // Az, Bz, Cz at 64 bit positions
impl PackedRow { pub fn values(&self, z: &WitnessRow) -> [F128; 3]; }
impl RowSystem {
    pub fn new(layout: &Layout) -> Self;
    pub fn lane_rows(&self) -> &[LaneRows; 2];  pub fn packed_rows(&self) -> &[PackedRow];
    pub fn failing_rows(&self, z: &WitnessRow) -> RowSet;
    pub fn to_matrices(&self) -> jolt_r1cs::ConstraintMatrices<F128>;
}
```

### Invariants

1. **One source per fact.** `layout` owns the column ranges and `chunk_indicators`, `decode` the table, `rows` the row list; preprocessing, `BitsBuilder`, `WitnessRow`, the checker and `to_matrices` derive from them, and the tests take positions from them. `Store`, `Branch` and the kinds are read from `Variant::line`.
2. **Sizes.** A bytecode row has 417 bits. `Bits` uses `81 + n(log_K_bytecode) + n(log_K_ram)` of 256 columns. A cycle has `136 + ⌈log_K_bytecode/4⌉ + ⌈log_K_ram/4⌉ + 2` rows, rows 0–129 over `F_2`, and 1,024 witness columns.
3. **Completeness.** For every cycle the machine executes, `bits_row` succeeds and no row fails.
4. **Soundness, in two contracts.** *Local:* `failing_rows` tests the listed rows on the `WitnessRow` that `WitnessRow::compute` builds from a bytecode row, a `BaseWords` and a `BitsRow`. No row ties that bytecode row to `BytecodeRa`, `NextPC` to the bytecode, or `Inc` off stores to anything, and on `JALR` both `(NextPC, JalrLowBit)` with `NextPC XOR JalrLowBit = Rs1Value ⊞ Imm` pass. *Execution:* the surrounding protocol owes five obligations, all against the one machine state that precedes the cycle: authenticated bytecode selection (the row is the one `BytecodeRa` indexes); authenticated register reads (`Rs1Value` and `Rs2Value` are the contents of the registers that row names as `rs1` and `rs2`, with `x0` reading zero); valid-successor membership (`NextPC` is the next cycle's PC, or a `FinalPC` that passes `final_pc_index`); the register write identity `RdWriteValue = old_rd XOR ((1 XOR Store) AND Inc)`, with `old_rd` the content of the row's `rd`; and the RAM update identity (the word at the committed index goes from `RamReadValue`, its content, to `RamReadValue XOR (Store AND Inc)`). Under them a cycle that fails no row is the machine's transition, and the free committed fields are exactly: `Pos` where no shift, access or comparison of unequal keys reads it; its high chunk on an access; the RAM index off accesses; `ShouldBranch` off branches; `JalrLowBit` off JALR; the spare columns.
5. **Statement checks.** `Layout::new` rejects an exponent outside `1..=24` (bytecode) or `1..=61` (RAM), a `LowestAddress` that is not a multiple of 8, a RAM range past `2^64` and sizes needing more than 256 columns. `preprocess` takes the decoded public program, which is its precondition; it pads the list to `2^log_K_bytecode` with invalid rows, which are zero in all nine columns, so the bound of 24 on the exponent is the crate's limit on what preprocessing allocates (`2^24` rows, whatever the length of the list), and both of its collections are reserved fallibly; it rejects a longer list, a valid row whose PC repeats or is not a multiple of 4, a missing required operand (`x0` is never substituted), a register above 31 and an immediate outside the instruction's encoding. `final_pc_index` rejects a `FinalPC` that is not a valid row's PC; the rows constrain the value of the last cycle's successor, not its membership in the bytecode.
6. **Representation.** `Bits` is a `&[BitsRow]`, words are `u64`, and `WitnessRow` is computed on demand; no type holds a field element per committed bit. `F128` appears only in `PackedRow::values`, the checker's products and `to_matrices`, with constants from `F128::from_raw`, never `from_u64`.
7. **Two evaluators.** `LaneRows::values` and `PackedRow::values` equal `Az`, `Bz`, `Cz` of `to_matrices()` on every `WitnessRow`: a structured form reads its constant from the witness column `ONE`, as the matrices do. `WitnessRow::compute` always sets `ONE`; no row tests it, and the relation that consumes the matrices binds it as public. `Term::eval` equals the word assembled from `Term::wires`.
8. **Total functions.** `Chunk`, `Term` and `PackedTerm` have private fields, read accessors and constructors that return a typed error outside the type's domain; the crate's own tables are built on a private `const` path. Functions of a digit, index, position or byte offset return `Result`, and `write_digit` and `Layout::write_*` clear the chunk before setting one indicator. Over these enforced domains no public function panics; errors are `thiserror` types that name the offending value. The crate has `#![forbid(unsafe_code)]` and the lints of `specs/verifier-closure-lints.md`.
9. **Boundary.** The dependencies are `jolt-riscv`, `jolt-field` with feature `binary` only, `jolt-r1cs` and `thiserror`. The crate decodes no instruction word: `preprocess` reads the kind and operands of `jolt_riscv::SourceInstruction`.
10. **Facts.** `bits_row` reads `bytecode_index`; `rs1_value` and `rs2_value` where an address, a shift amount, a key or a `JALR` target uses them; `rd_pre_value` and `rd_post_value` off stores; `ram_word_index` on an access; `ram_pre_value` and `ram_post_value` on a store. `BaseWords::from_facts` reads `rs1_value`, `rs2_value`, `rd_post_value`, `ram_pre_value` and `next_pc`. Nothing reads a load's `ram_post_value`, and no row reads the RAM fields of a cycle without an access. Neither function executes the instruction or validates a field it does not read.

No `jolt-eval` invariant changes: nothing here is reachable from the existing prover.

### Non-Goals

Router tensors; any sum-check; any commitment; the arguments that discharge the obligations of invariant 4; the adapter from the tracer to `CycleFacts` (its contract is under Architecture); the M extension (variant indices 58–63 are left for it); traps and devices; private input; a word length other than 64; a `Bits` row wider than 256 columns.

## Evaluation

### Acceptance Criteria

"Accepted" means `failing_rows` is empty. "The oracle" and "the replay" are those of the Testing Strategy.

- [ ] **Sizes.** `chunk_indicators` is 0, 1, 3, 7, 15, 75, 82, 90, 97, 105 at 0, 1, 2, 3, 4, 20, 23, 24, 27, 28. With `log_K_bytecode = 20`, the used columns are 231, 238, 246, 253 at `log_K_ram` 20, 23, 24, 27, and 28 is `BitsRowOverflow`. The bytecode column widths sum to 417. There are 148, 149, 150 rows at `(20, 20)`, `(20, 23)`, `(21, 21)`. Each `LayoutError` is returned on an input that violates only its check; an exponent of 0 is `LogSizeOutOfRange`.
- [ ] **Preprocessing.** Rows from hand-encoded words, decoded by `jolt_program::image::decode::decode_instruction`, equal literal rows; the cases include negative immediates, `srai`, memory variants at a non-zero `LowestAddress` and every variant with destination `x0`. A kind outside RV64I and a compressed row give the invalid row. Five instructions give 8 rows at `log_K_bytecode = 3` and `InstructionListTooLong` at 2. Each other `BytecodeError` is returned, except `TableSizeOutOfRange`, which needs a target whose `usize` cannot hold the table, and `TableAllocationFailed`, which needs an allocator that refuses it.
- [ ] **Honest cycles.** Each of the 52 instructions has an oracle case against a hand-written literal. At least 100,000 oracle cycles over six layouts, with all 58 variants and both zero and non-zero destinations, are accepted.
- [ ] **One cycle.** For each of the 43 variants without a data access, a family of boundary operands, all 512 assignments of `(Pos, KeysDiffer, ShouldBranch, JalrLowBit)` and a candidate set of `(RdWriteValue, NextPC)` with `NextPC` among the PCs of valid rows: the accepted outcomes are the oracle's, and none when it reports no witness.
- [ ] **Memory.** For the 59 legal pairs of a memory variant and a byte offset, the honest cycle is accepted. Every single-bit change of a committed bit of `Inc`, the low chunk of `Pos` or the RAM index is rejected once the base words are reconstructed by the identities of invariant 4 (a change of `Inc` moves `RdWriteValue` off stores and the RAM post-word on stores); every single-bit change of the supplied `RdWriteValue`, the committed row fixed, is rejected by the rows alone. For `log_K_ram` from 1 to 5, every address within 24 bytes of RAM and all `K_ram · 64` assignments of the RAM index and `Pos`: the accepted set is the oracle's, and `bits_row` returns a typed error exactly when it is empty. Through a private constructor that skips the range check, `ld x0, 8(x0)` is accepted at `LowestAddress = 2^64 − 2^22`, a layout that `Layout::new` rejects.
- [ ] **Whole trace and free fields.** For five fixtures of at most 64 cycles each that together use all 58 variants, the table from `fill` replays to the oracle's registers, RAM and PCs. Each single-bit change of each of the 256 columns of each cycle is rejected by the replay or leaves that state unchanged, the second exactly for the free fields of invariant 4.
- [ ] **Successor outside the bytecode.** A jump out of the bytecode at an inner cycle is rejected by the replay for every next row. At the last cycle, by jump or by fall-through into a padding row, the rows accept and `final_pc_index` rejects.
- [ ] **One-hot rows.** For 15, 7, 3 and 1 indicators and every bit pattern, the chunk's row holds exactly when `Chunk::full` has weight one.
- [ ] **Adder and comparison.** The adder rows hold exactly when `S = A + B`, over all `2^24` byte triples in bits 56–63 and all `A, B < 2^8`, `S < 2^9` in the low bits. Over all `2^16` byte pairs at either end of the word and all `(Pos, KeysDiffer)`, the accepted results of `SLT`, `SLTU`, `BEQ` and `BNE` are those of integer comparison.
- [ ] **Decode table.** For each pair `A/B`, some honest cycle of `A` is rejected under a bytecode row carrying `B`: `SUB/ADD`, `ADD/SUB`, `ADDW/ADD`, `ADDI/ADD`, `AND/OR`, `OR/XOR`, `SLT/SLTU`, `SLTU/SLT`, `SLTI/SLTIU`, `SRA/SRL`, `SRL/SRA`, `SLL/SRL`, `SRAW/SRLW`, `SLLW/SLL`, `SRAI/SRAIW`, `LB/LBU`, `LH/LHU`, `LW/LWU`, `LWU/LW`, `LD/LW`, `SB/SH`, `SD/SW`, `BEQ/BNE`, `BLT/BGE`, `BLT/BLTU`, `BGEU/BGE`, `JAL/JAL_X0`, `JALR/JALR_X0`, `LUI/AUIPC`, `ECALL/NOOP`, `NOOP/ECALL`, `LOAD1_X0/LOAD2_X0`, `LOAD8_X0/LD`. For each of the eleven `RowGroup`s, a cycle that is wrong for the machine fails rows of that group only.
- [ ] **Two evaluators.** Invariant 7 holds on 10,000 seeded random `WitnessRow`s at two layouts, against a dense vector of `F128` built in the test, and on every term of the table.
- [ ] **No panic, no allocation.** Random `CycleFacts`, `BitsRow`s, `WitnessRow`s, source operands, sizes, digits, indices and positions pass through every public function without a panic. Under a counting allocator, `fill`, `WitnessRow::compute` and both `values` allocate nothing over 4,096 cycles.
- [ ] **Gates.** `fill` meets the gate of Performance. `cargo clippy -p jolt-rv64i-arith --all-targets -- -D warnings`, `cargo nextest run -p jolt-rv64i-arith --cargo-quiet` and `cargo fmt --check` pass.
- [ ] **Constant column.** `WitnessRow::compute` sets `ONE` on every input, and Two evaluators also holds on a random `WitnessRow` with `ONE` cleared at each layout.
- [ ] **Descriptor and operand domains.** `Chunk::new`, `Term::new` and `PackedTerm::new` succeed at each bound, fail just past it, and reproduce every entry of the crate's tables. `shift_form` rejects position 64, `load_form` and `store_form` offset 8, and `write_digit` and `Layout::write_*` an out-of-range digit or index, with the row unchanged. `BytecodeRow::from_source` returns `MissingOperand` for each absent register of an `ADD`, `RegisterOutOfRange` at 32, and `ImmediateOutOfRange` just past the bounds of each encoding format.
- [ ] **Invalid row.** With `variant = None` and every other field non-zero, `BytecodeRow::column` over `BytecodeColumn::ALL` is the literal `[0; 9]`.
- [ ] **No-access cycle.** At a non-zero `LowestAddress`, a stack `CycleFacts` without an access and with zero RAM fields gives a `bits_row` that is accepted, and the same row for arbitrary RAM fields; a load gives the same row for every `ram_post_value`.
- [ ] **JALR low bit.** For `jalr x0, 1(x1)` at PC 0 with `x1 = 0`, the rows accept both `(NextPC, JalrLowBit) = (0, 1)` and `(1, 0)`; the replay accepts the first and rejects the second, whose successor fails `final_pc_index`.
- [ ] **Binary-only build.** `cargo tree -p jolt-rv64i-arith -e normal,dev -f '{p} [{f}]'` shows `jolt-field` with the feature `binary` alone, and Two evaluators runs in that build.

The crate at `82d8ec6c4` has the seven modules, their unit tests and `tests/alloc.rs`. The PR is ready when these also hold:

- [ ] **Oracle.** `tests/suite/common/interp.rs` and `asm.rs`, with their known-answer tests in `tests/suite/oracle.rs`, are in the crate as at `8512e96b0`, import no workspace crate, and pass.
- [ ] **Harness.** `tests/suite/common/harness.rs` decodes the oracle's words with `decode_instruction`, preprocesses them and maps each oracle record to one `CycleFacts`; `replay.rs` is the replay. Every criterion above that names the oracle or the replay runs through them.
- [ ] **Mutation fixtures.** The five fixtures of Whole trace and free fields are literals in the test source (words, entry PC, initial registers, layout, cycle count), apart from the generators of Honest cycles.
- [ ] **JALR pair.** One test asserts both halves of JALR low bit.
- [ ] **Benchmark.** `benches/witness.rs` builds the corpus of Performance, asserts both checksums before timing and prints the eight figures.

### Testing Strategy

Ground truth for the machine is a test-only interpreter, `tests/suite/common/interp.rs`, with an encoder, `asm.rs`, both written from the RISC-V unprivileged ISA manual and the machine paragraph of Goal. They take `u32` words and `u64` addresses and import no workspace crate, so they share no code with the decoder, the preprocessing or `BitsBuilder`. The interpreter reports "no witness" where the machine has none. A test-only replay, `replay.rs`, stands in for the four obligations of invariant 4: from a table of `BitsRow`s and the bytecode it fetches the row at each committed index, recomputes the base words by the two identities on its own registers and RAM, and takes each successor from the next fetched row or from a `FinalPC` checked by `final_pc_index`.

A mutation changes either a committed bit, after which the replay reconstructs the base words, or a supplied base word with the committed row fixed; each test says which. Whole-trace mutation replays a trace once per column and cycle, so its fixtures are short and apart from the honest corpus; the exhaustive adder and comparison tests evaluate only the row groups they target. No time limit is promised for a test binary before it is measured under the repository's nextest profile. Tests follow `.claude/skills/test-policy/SKILL.md`. No existing test changes; the `host` and `zk` modes do not apply.

### Performance

| Routine | Tier | Requirement |
|---|---|---|
| `BitsBuilder::bits_row`, `fill` | production | `fill`: at most 50 ns per cycle on one core at `2^20` cycles; no heap allocation; no state between cycles |
| `WitnessRow::compute`, `LaneRows::values`, `PackedRow::values` | production | no heap allocation; word operations only; reported, not gated |
| `failing_rows`, `to_matrices`, `Term::wires`, all of `tests/` | reference | none; never called from a hot loop |

The 128 bitwise rows are evaluated for one cycle at all 64 bit positions at once, as `(Az & Bz) ^ Cz` on `u64`. The packed rows use shifts and XORs on `u128`; a multiplication in `F128` occurs only where the checker forms a product. A word selected by `Pos` is the sum over all hot values, as a router computes it, so a row whose `Pos` chunks are not one-hot costs more than an honest one; the figures are for honest rows. `fill` is chunk-parallel: on the two halves of a split of `facts` and `out` it writes what one call writes, and the caller owns the thread pool. `Bits` takes 32 bytes per cycle; a `WitnessRow` (128 bytes) lives on the stack.

`benches/witness.rs` measures a frozen corpus: a loop of 32 instructions at PCs `0x8000_0000` to `0x8000_007C`, with the words

```text
00d09293 0050c0b3 0070d293 0050c0b3 01109293 0050c0b3 7f80f313 00230333
00033383 00138433 00833023 00432483 4014853b 00a32023 00334583 00b303a3
00231603 0083b6b3 00a4a733 00e68263 00164263 0013f263 00b097b3 40c7d833
0050d89b 0107e933 011979b3 fff98a1b abcdeab7 00000b17 008b0be7 f85ff06f
```

An iteration is 12 additions, logic and upper-immediate instructions, 6 shifts, 2 comparisons, 4 loads, 3 stores, 3 branches of offset 4, a `jalr` and a `jal`, on a path that does not depend on the data. The layout is `(20, 20)` with `LowestAddress = 2^31 − 2^16`, where `preprocess` pads the loop to `2^20` rows. The oracle runs it from the seed `x1 = 0x9E37_79B9_7F4A_7C15`, `x2 = LowestAddress` and zero elsewhere. Before timing, the bench asserts the checksum of `Bits`, the 64-bit FNV-1a fold with each `u64` of the table, in order, in place of a byte: `0xFC7F_3B40_737C_010A` at `2^20` cycles and `0x1B85_E435_A7E6_ACAB` at `2^22`.

The build is the release profile with `-C target-cpu=native`. Setup (decoding, preprocessing, the oracle's run, allocation) is outside the timed region, and a figure is the median of three timed passes after one warm-up pass, in nanoseconds per cycle. The `compute` figure includes the lookup of the bytecode row and `BaseWords::from_facts`, which every caller pays with it; the figure for the row values excludes the computation of the witness and includes the fold of the values into the bench's checksum. At both cycle counts the bench reports `fill` on one performance core, `fill` over `rayon` chunks of `2^14` cycles on 12 threads, `WitnessRow::compute`, and the row values. Only the first at `2^20` is gated, on an Apple M4 Max; the others go in the PR description, and none bounds proving time. No `jolt-eval` objective moves.

## Design

### Architecture

Seven modules, by ownership: `layout`, `variant`, `decode` (sources, terms, the table), `bytecode` (rows, preprocessing, the `FinalPC` lookup), `words` (`BaseWords`, `WitnessRow` and its block layout), `cycle` (facts to bits) and `rows`. They are not layered: `Variant::line` returns a `decode::Line`, `Sources::new` takes a `words::BaseWords`, and `BaseWords::from_facts` a `cycle::CycleFacts`. Construction order is `Layout`, `Bytecode`, then `BitsBuilder` and `RowSystem`. `Words::compute` and `F2Words::compute` share the decode-rail dispatch, carry formulas and key-bit step over `Sources::from_parts`, while `RowSystem` reads the compact evaluator’s result through its existing row definitions without constructing a `WitnessRow` per cycle: `f2_lanes` gives the two word triples, and `f2_tail` derives, once, the table of sixteen bytes that gives the six bits of rows 128 and 129 from `LeftKeyBit`, `RightKeyBit`, `LessThan` and `KeysDiffer`, after checking that the forms of those rows read no other column than these and `ONE` and carry no coefficient other than 0 or 1 (`TailError` otherwise). `Words::compute` sums the key bits over every hot position of the position chunks and `F2Words::compute` selects the one position `Layout::pos` decodes; the two agree on a row whose position chunks store at most one indicator each, and `Rv64iWitness::replay` of the prover crate, which rejects any other row, is where that is enforced, for a witness as its constructors return it. The compact evaluator checks nothing itself: a witness whose public `bits` or `decoded` were changed after construction is outside the statement. The two lane families are stated once, over the six lanes the compact evaluator carries, and the families over a `WitnessRow` are derived from that statement. `SourceParts` is the one reading of a committed row’s source inputs: `from_bits` with the layout’s linear digit semantics, which `Sources::new` uses, and `from_checked_bits`, which rejects a chunk with two stored indicators through `Chunk::checked_digit` and returns the bytecode index with the parts. The sparse form of the rows reuses `jolt_r1cs::ConstraintMatrices`, whose `weighted_columns` and `linear_form_bilinear_eval` evaluate the multilinear extension of a row combination at a point.

The trace adapter, outside this crate, copies the index in the list given to `preprocess` (unsupported entries included) and the register values. Without an access it gives zero RAM facts. On an access it takes `checked_sub(LowestAddress)` of the enclosing doubleword's address, requires a multiple of 8, shifts right by three, and returns a typed error for a word outside the window. The recommended adapter is fused: one `CycleFacts` on the stack per cycle, passed to `bits_row` in the adapter's own pass; `fill` over a slice is for tests and chunked callers.

### Alternatives Considered

- **Decode table as code, one branch per variant.** A router needs each bit of a word as a sum over pairs of a source bit and a selector value. With `Term`s these pairs are an enumeration of the table; with code they would be transcribed a second time.
- **A word-length parameter, for exhaustive tests at small widths.** It would thread through every definition and need an oracle for a machine no manual defines. The adder and comparison tests are exhaustive on bytes at both ends of the 64-bit word.
- **Recorded fixtures as the oracle.** They cannot answer enumeration and mutation tests, whose inputs are chosen at run time.

## Documentation

No change to the Jolt book. The rustdoc of `layout`, `decode` and `rows` states the layout formulas, the table and the rows; that of `BaseWords`, `RowSystem` and `bits_row`, the obligations of invariant 4 and the fields read.

## Execution

Build in module order, each module with its literal tests, then the oracle, the harness and replay, the trace tests and the benchmark.

## References

- `specs/binary-field.md` (`F128`); `specs/verifier-closure-lints.md` (the lint set).
- `crates/jolt-riscv` (`SourceInstruction`); `crates/jolt-r1cs` (`ConstraintMatrices`).
- The RISC-V Instruction Set Manual, Volume I: Unprivileged Architecture.
