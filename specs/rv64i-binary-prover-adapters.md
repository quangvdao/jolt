# Spec: Adapters for the Packed-Bit Kernels of RV64I over the Binary Field

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-10                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The experiment in RV64I hash-based Jolt over binary fields has a protocol that `prove` and `verify` run end to end on reference kernels (`specs/rv64i-binary-protocol.md`), and a crate of cores that are held to a number of nanoseconds per cycle (`specs/rv64i-binary-prover-kernels.md`) and that `prove` cannot reach. This spec is the layer between them: the per-cycle tables of the witness that the cores read, a `PrepareKernel` for each of ten members that wraps a core, a constructor of the registry, and the two tests of the layer, which are equality of the proof bytes with the reference registry under mixed slots and a bench at `2^20` and `2^22` cycles. The layer adds no protocol and no arithmetic. What it can get wrong is an order, an index, a second walk of the trace or a copy of a table, and each of those has one owner here.

## Intent

### Goal

Put the cores of `jolt-rv64i-kernels` behind the stage registries of `jolt-rv64i-prover` for `SpartanOuterF2`, `RouterShort`, the five `RouterCycle` members, `BytecodeReadCycle`, `RamRaProduct` and `BitsReduction` with the vector `C`, so that `prove` with any subset of those ten slots optimised emits the bytes the reference registry emits, within the cores' thresholds plus a stated allowance.

**Notation.** The points `r_1`, `w`, `x`, `r_bit`, `r_3`, `r_4`, `r_5`, `a_bc`, `a_ram`, `r_6` and the symbols `t`, `T`, `b`, `a`, `d_b`, `d_a`, `n(·)` are those of `specs/rv64i-binary-protocol.md` §2, §3 and §5; `g` is the first committed column of the RAM chunks, `layout.ram_ra()[0].start()`. Cores, passes, shapes, `Fold_ρ`, `Source_ρ`, `Sel_f` and the restrictions `x|_ρ`, `x|src` are those of the Goal of `specs/rv64i-binary-prover-kernels.md`, "the kernels spec" below. Both specs list a point low variable first, so a point crosses this layer unchanged. A *slot* is a field of a registry (`Stage3bKernels::shift`); the variables of the short sum-check are *short slots* and the entries of a bank *word slots*.

**Adapters.** Each row is one `PrepareKernel<F128, Member, Rv64iPlane>` whose `prepare` returns a `'static` `SumcheckKernel`.

| Member (slot) | Core and passes | Shared state |
|---|---|---|
| `SpartanOuterF2` (`stage1.spartan_outer_f2`) | the lanes pass; `OuterF2Core` | none |
| `RouterShort` (`stage3a.router_short`) | `fold_pass`; `RouterShortCore` | validated source, scatter plan |
| `RouterCycle{Variant, Shift, Memory, Compare, Branch}` (`stage3b.*`) | `source_lift`; one `RoutersCycleCore` for the five; `claims_pass` | validated source, scatter plan, the cycle group |
| `BytecodeReadCycle` (`stage6b.bytecode_read_cycle`) | `combined_weight`; `ChunkProductCore` | validated source |
| `RamRaProduct` (`stage6b.ram_ra_product`) | `combined_weight`; `ChunkProductCore` | validated source |
| `BitsReduction` (`stage6b.bits_reduction`) | `g_pass_digits`; `ReductionCore`; `column_pass`, whose result is `C` | validated source |

**Surface.**

| Item | Where | Contract |
|---|---|---|
| `DecodedCycle`, `DigitFields`; `Rv64iWitness::{decoded, variant_cycles, row}` | `plane.rs` | Design, "The decoded row", "Lanes" |
| `WitnessColumns`, `WitnessSource`, `SharedSource` | `optimized/source.rs` | "Columns and words", "Source and shared state" |
| `WitnessLanes`, `SpartanOuterF2Prepare` | `optimized/outer.rs` | "Lanes", "Outer" |
| `router_shapes`, `RouterShortPrepare`, `RouterCycle{Variant, Shift, Memory, Compare, Branch}Prepare` | `optimized/routers.rs` | "Routers" |
| `BytecodeReadCyclePrepare`, `RamRaProductPrepare`, `BitsReductionPrepare` | `optimized/tail.rs` | "Tail and the vector `C`" |
| `Rv64iBackend::optimized` | `backend.rs` | "Registry" |
| `BankWord`, `Factor`, `Bank`, `bank` | `jolt-rv64i-verifier`, `public/routes.rs` | "Routers" |

Every optimised `…Prepare` is a unit struct with `Default`, like its namesake in `reference/`.

### Invariants

1. **No protocol change.** No message, round count, member, transcript event or byte of the wire changes. `prove_batch`, `SequentialRounds` and the generated stage drivers are used as they are. `PrepareKernel`, `SumcheckKernel`, `ProofSession`, `KernelSlots` and the plane parameter are used as they are: this spec asks for no change to a shared crate.
2. **Mixed registries.** Every assignment of the ten slots to the two tiers proves, and proves the same bytes. No optimised `prepare` assumes that another slot is optimised: what two adapters share is built by whichever needs it first and found by the other (Design, "Source and shared state").
3. **One owner per conversion.** The packing of the decoded row is `DigitFields`. The numbering of digit columns, trace words and bytecode words, and their correspondence with the committed columns, is `WitnessColumns`. The bank and the selector factors of each router are `routes::bank`. Point order has nothing to own: no adapter reverses or permutes a point. An adapter passes a point whole, or slices it by `routes::{source_slots, selector_slots}`, by `RouterShape::word_slots` or by the chunk points of a relation, and the only equality helper it calls is `round::eq` of the kernels crate.
4. **No relation expression.** An adapter evaluates no input or output expression; the stage does, through `Expr::try_evaluate`. An adapter writes a formula of the protocol in three places, and each is pinned where it is used: the weight of a chunk product as a list of terms, and the slot weights of `RouterCycleVariant`, both compared with the relation's derived terms at the final point by `validate_derived_tables`; and the three leg claims of the reduction, whose sum the kernel compares in round 0 with the claim the engine hands it.
5. **One pass, one owner.** This layer creates two tables of `T` elements, the decoded rows and the lanes. Each is written by one pass and has the owner and the drop point of the Memory table. The decoded rows are written by the replay the witness already makes; no adapter walks the committed rows to decode them.
6. **No copy of a table.** An adapter moves a table into a core by value or shares it by `Arc`. An adapter that copies a table of `T` or `bytecode_rows()` elements which the witness, a pass or a core holds is a defect. The converse is required of the cores: a core reads the tables of its source in place (Design, "Chunk digits in place").
7. **Boundary.** The adapters live in `jolt-rv64i-prover/src/optimized/`. `jolt-rv64i-kernels` gains no dependency on a relation type or on the arithmetisation. `prepare` clones the `Arc`s of the witness tables a core reads later, and nothing borrowed from the witness outlives it.
8. **Total functions.** `prepare` returns `KernelError::InvalidGeometry` with the display of the core's or the source's error; the round and output methods return the errors of their traits. Nothing panics on a malformed witness or geometry.
9. **Honest witness.** `OuterF2Core` requires `C = A & B` on every lane word and tail bit and does not check it. On a witness that violates a row, the optimised registry returns an error from batch 1 and no proof: the driver's comparison of `expected_final_claim` with the final claim fails, where the reference kernel fails a round check. The mechanism is that comparison, in `impl_stage_prover!`; no adapter adds a check.
10. **Determinism.** The bytes do not depend on the thread count, nor on which optimised member of a batch is prepared, driven or extracted first.
11. **Characteristic 2.** No adapter halves, scales by an integer or samples at an integer; coefficients are challenges and claims, combined with `+` and `·`.

No `jolt-eval` invariant changes.

### Non-Goals

- Cores or adapters for `SpartanOuterF128`, `SpartanInner`, batches 4 and 5 and `BytecodeReadAddress`. Their eight slots keep reference kernels. A second spec covers them: its adapters are `PrepareKernel`s that `Rv64iBackend::optimized` assigns to those fields, with session state types of their own.
- The internals of a core. Design, "What this spec needs from its neighbours" lists what the adapters require of the kernels crate.
- Layouts with `d_a > 7`, that is `a > 28`. `ChunkProductCore` takes at most seven columns; `RamRaProductPrepare::prepare` returns `InvalidGeometry` naming the count, before it builds a table, and such a layout uses the reference slot.
- The time of `prove`. With eight slots on reference kernels it measures the reference tier.

## Evaluation

### Acceptance Criteria

The five programs are `PROGRAMS` of `tests/support`. "Seeded" points have pairwise distinct coordinates.

- [ ] **Decoded rows.** For the five programs at `t = 6` through `from_facts`, for `from_bits` on their committed rows, and for `Rv64iWitness::synthetic` at `(b, a) = (4, 5)` and `(10, 14)`: every field of every `DecodedCycle`, read through `DigitFields`, equals `Layout::{bytecode_index, ram_index, pos, inc}` and the three flag columns of the committed row and the variant index of the fetched row; `variant_cycles` is the number of cycles of each variant; `from_bits` and `from_facts` give equal tables. `DigitFields::new` at `(3, 46)` puts the variant in bits 58–63. (`tests/optimized_sources.rs`)
- [ ] **Source against the committed table.** On the same witnesses, for every cycle `j` and column `y < used_columns()`: `Bits[y, j]` is bit `y` of trace word `Inc` for `y < 64`, and otherwise is `digit(c, j) == Some(v)` for `(c, v) = WitnessColumns::committed(y)`. The four kind digits equal `Variant::{shift, access, key_kind, branch}` of the fetched row, the trace and bytecode words equal the fields of `CycleWords` and `BytecodeRow`, and `ValidatedTrace::new` accepts the source. `g_pass_digits` under `WitnessColumns::column_map()`, with a seeded weight on every used column, equals `Σ_y L[y]·Bits[y, j]` summed over the committed rows. (`tests/optimized_sources.rs`)
- [ ] **Lanes.** On the five programs, `WitnessLanes` equals `LaneRows::values` of both lane families and the first two `PackedRow::values` of `Rv64iWitness::row(j)` for every cycle, and `OuterF2Core::check_rows` accepts it. (`tests/optimized_outer.rs`)
- [ ] **Shapes.** At `(4, 5)`, `(10, 14)`, the reference layout and `(3, 46)`: `router_shapes` returns five shapes that `RouterShape::new` accepts, whose bit, word and selector slots are `source_slots` and `selector_slots` of the router and whose `route()` is `RouteTensors::entries` of the router. `RouteTensors::new`, reading `bank`, gives the tensors whose digests `tests/routers.rs` pins. (`tests/optimized_routers.rs`)
- [ ] **Lockstep.** For each of the ten adapters, on each of the five programs at `t = 6`, with the `ProverInputs` its stage function builds on the outputs of the reference stages before it: the reference kernel and the optimised kernel, prepared in a session each, pass `run_lockstep_checked` under seeded challenges. That is equal coefficients in every round, equal typed outputs in canonical order, derived tables validated, and the relation's output expression on those values equal to the final claim. (`tests/optimized_{outer, routers, tail}.rs`)
- [ ] **Batch outputs.** On each of the five programs at `t = 6`: under each of the 32 subsets of the slots of batch 3b, after a batch 3a proved by the reference kernel and after one proved by the optimised kernel in the same `ProofSession`, `stage3b::prove` returns the `BatchProof` and the `Output` it returns under the reference kernels, every typed value with its ten alias cells and every point; under each of the 8 subsets of batch 6b, in an empty session and in one where an optimised batch 3a has left the validated source, `stage6b::prove` returns the same proof, output and `BitsColumns`. (`tests/optimized_{routers, tail}.rs`)
- [ ] **Mixed registries.** For the program of the "Proof vector" criterion of `specs/rv64i-binary-protocol.md`, the counting loop at `t = 6` with `TransparentBits`: `prove` returns a proof whose `to_bytes` is that of `Rv64iBackend::reference()`, the encoding whose digest `tests/wire.rs` fixes, and `prove_with_transcript` records the same events, under `Rv64iBackend::optimized()`; under each of the ten slots optimised alone; under the five slots of batch 3b; under the six router slots; and under the three slots of batch 6b. (`tests/optimized_registry.rs`)
- [ ] **End-to-end corpus.** For each of the five programs at `t ∈ {6, 8, 10}`, `prove` under `optimized()` returns the bytes it returns under `reference()` and `verify` accepts. The three tests of `tests/e2e.rs` that expect `prove` to fail, in batches 4, 4 and 6a, fail in the same batch with the same error under `optimized()`. (`tests/optimized_registry.rs`)
- [ ] **A witness whose rows do not hold.** On a counting-loop witness changed so that `OuterF2Core::check_rows` names a cycle, `prove` with the optimised outer slot returns an error from batch 1 and no proof. (`tests/optimized_outer.rs`)
- [ ] **Rejected inputs.** A `RouterCycle` member prepared twice in one session returns `InvalidGeometry`. A witness whose `decoded` has another length than `bits` is rejected by `prove` with its dimension error. The bound `d_a ≤ 7` has no test: its least input is a witness of `2^29` RAM words. (`tests/optimized_registry.rs`)
- [ ] **Bench.** `benches/adapters.rs` reports the three pipelines of Performance at `log_t` 20 and 22 on 1 and 12 threads, with the lanes pass apart.
- [ ] **Performance.** A result above its threshold at `log_t = 22` fails the PR that adds or changes the adapter.
- [ ] **Trace-sized allocations.** The bench lists every allocation of `T` bytes or more made during a pipeline, with its size. The PR that adds an adapter sets the list against the Memory table below and that of the kernels spec; an allocation in neither fails the PR.
- [ ] **Gates.** `cargo clippy --all-targets -- -D warnings` and `cargo nextest run --cargo-quiet` pass for `jolt-rv64i-prover --features test-utils` and `jolt-rv64i-verifier`; `cargo fmt --check` passes.

### Testing Strategy

The ground truths are the reference tier, the frozen digest of `tests/wire.rs`, the committed rows and the arithmetisation: `Layout`, `Variant`, `RowSystem`. No test has an oracle of its own for a relation. The lockstep and batch tests compare a kernel with the reference kernel of its relation, which is the ground truth `.claude/skills/test-policy/SKILL.md` names for an optimised kernel; none compares two optimised configurations. `run_lockstep_checked` drives one member, so equality of an alias with its source in another member is the "Batch outputs" criterion and not "Lockstep".

"Any subset of the ten slots" is 1,024 registries, and the criteria reach them in two steps. The bytes of a batch are a function of its inputs, of its kernels and of what the session holds. The session holds the validated source and the scatter plan, which are functions of the witness whoever builds them, and the cycle group, which does not outlive batch 3b. "Batch outputs" therefore exhausts the subsets of each batch that has more than one of the ten slots, with the session in both states; "Lockstep" is the same statement for batches 1 and 3a, which have one; and by induction over the batches every registry proves the bytes of the reference registry. "Mixed registries" then checks that conclusion on whole proofs for 14 registries and not for 1,024. Seeds are fixed. Every existing test of `jolt-rv64i-prover`, `jolt-rv64i-verifier` and `jolt-rv64i-kernels` passes unchanged; `tests/routers.rs` passes with `RouteTensors::new` reading `bank`. The `host` and `zk` modes do not apply.

### Performance

The bench is `benches/adapters.rs` of `jolt-rv64i-prover` (`harness = false`, feature `test-utils`), on the machine, flags, sizes and thread counts of the kernels spec's Performance section. Ids are `adapters/<pipeline>/<log_t>/<threads>`. Under the counting allocator of the kernels crate's benches, included by path as `tests/support` includes the arithmetisation's helpers, it reports wall time per cycle, the peak live bytes above the witness and the trace-sized allocations.

**What it drives.** Three pipelines, each through the `prepare` of its adapters and from an empty `ProofSession`:

- `outer`: `SpartanOuterF2Prepare::prepare`; the rounds of the one kernel as `run_core` of the kernels crate drives a core, `prove_round` under challenges from a fixed seed and then `finish_rounds`; `output_claims`; `park_residue`. It runs outside a batch, because the other member of batch 1 has no core.
- `routers`: the generated driver of batch 3a and then that of batch 3b (`StageProver::prove`, as `stage3a::prove` and `stage3b::prove` call it) in one session, batch 3b taking its claims and `x` from the output of 3a.
- `tail`: the generated driver of batch 6b, which ends with `C`.

The inputs are built before any timing: seeded points and challenges, and each claim by one summation over the witness. The input claim of `RouterShort` comes from a `fold_pass` of the setup and the route tensors, the chunk-product claims from the decoded rows, the six claims of the reduction from `column_pass` at `r_1`, `r_3`, `r_5` and `BitsReduction::weights()`. A wrong setup claim cannot pass for a result in `routers` or `tail`, since the driver compares `expected_final_claim` with the final claim of `prove_batch`. `outer` has no setup claim: its member has no input and its sum is zero.

The alternative, the stage functions with the uncovered slots stubbed, does not exist at this size. A stage function takes the outputs of the batches before it, and the driver checks the final claim of every member, so a stub has to be an honest prover of its relation: at `2^20` cycles that is the core the second spec has not built. A stub exempt from the check would run a batch with other members, degrees and claims than the one being priced. Batches 3a, 3b and 6b have no uncovered member, so their own drivers run; batch 1 has one, so the outer kernel runs alone.

The members of batch 3b share one pass per round and those of batch 6b one source and one `column_pass`, so a member alone has no time of its own. The pipeline is gated, as the kernels spec gates `routers` and `tail`.

**Witness.** One executed witness per size: a seeded straight-line program of `2^20` bytecode rows with the cycle mix of the kernels spec's `all_rows` profile and a last jump to its first row, run by the interpreter of `tests/support` for `2^log_t` cycles over `2^20` RAM words, through `Rv64iWitness::from_facts`. It is executed because the outer adapter needs every row to hold, which `Rv64iWitness::synthetic` does not give. It visits every row once at `log_t = 20` and four times at 22, except the rows a taken branch skips, which is the `ρ` of the kernels spec. The run is cut at `2^log_t` cycles and reaches no exit: the bench proves batches and no statement, and `from_facts` takes the final program counter from the last cycle.

**Thresholds.** `θ(id)` is the threshold of that row of the Requirements table of the kernels spec, which owns the number. Single thread, nanoseconds per cycle:

| Bench | Threshold | With the thresholds in force: `22`, `20` |
|---|---|---:|
| `adapters/outer` | `⌈1.05·θ(outer_f2/local)⌉ + λ`, `λ = 40` | 307, 307 |
| `adapters/routers` | `⌈1.05·θ(routers/all_rows)⌉` | 314, 395 |
| `adapters/tail` | `⌈1.05·θ(tail/local)⌉` | 202, 202 |

The last column evaluates the formula and defines nothing. On 12 threads at `log_t = 22` the requirement is wall time per cycle, the threshold divided by 9.6, as in the kernels spec. A threshold moves when its `θ` moves there, and in one other way: `λ`, below.

The 5% is the allowance of an adapter. It is a budget and not a model, for three things. The first is the work per proof that no core's model has: the shapes from the `N_R` route entries, the route weights at `x` evaluated once by the kernel's check and once by the driver at two multiplications per entry, the weight vectors of the reduction, the 256 column weights, and `Expr::try_evaluate`. The route weights are the counted part, `4 M` per entry and 0.55 ms at the `M` of point 1; with the shapes, which are not counted, 1 ms is an estimate, and it is 0.25 ns per cycle at `2^22` and 1 ns at `2^20`. The second is the source: a digit is a shift and a mask of one loaded word, and for the four kind columns a read of a 64-entry table, where the bench of a core reads a synthetic trace; nothing has measured the difference. The third is `output_claims`, which adds no pass to those the cores' models price. Each pipeline starts from an empty session, so `routers` and `tail` each pay the scan of `ValidatedTrace::new` that a proof pays once.

`λ` is the lanes pass, one `WitnessRow::compute` per cycle. Nothing has measured that function. `λ = 40` is a ceiling charged to the part of the prover's budget per cycle that the kernels spec does not allocate to its three pipelines; the PR that adds the bench reports the pass alone on a quiet machine and replaces 40 with 1.25 times the measurement if that is lower. A pass above 40 fails that PR.

**Memory.** The tables this layer creates or holds:

| Object | Size | Owner | Created | Dropped |
|---|---|---|---|---|
| Decoded rows | 16 bytes per cycle | `Rv64iWitness`, shared by `Arc` | the replay of the witness constructor | with the last `Arc`, the witness's or a kernel's |
| Lanes and tails | 48 and 1 bytes per cycle | `WitnessLanes`, in an `Arc` held by `OuterF2Core` | the lanes pass, in `prepare` of batch 1 | with the kernel, at `park_residue` of batch 1 |
| Validated source | four tables of 64 entries, the widths, three `Arc`s | `SharedSource` | first optimised `prepare` of batch 3a, 3b or 6b | with the session |
| Scatter plan | the kernels spec's row: 4 bytes per cycle | `SharedSource` | first optimised `prepare` of batch 3a or 3b | first optimised `park_residue` of batch 3b; else with the session |
| Word lifts | the kernels spec's row: 96 bytes per cycle | the cycle group | `source_lift`, first optimised `prepare` of batch 3b | the return of `claims_pass` |
| Recorded challenges | `t` elements per kernel | the kernel | the rounds | with the kernel |

Every other table of `T` or `bytecode_rows()` elements of a pipeline is created by a core or a pass, is in the Memory table of the kernels spec, and reaches its core by value. The lanes stand beside the 97 bytes per cycle of `OuterF2Core` for the whole of batch 1, because the core keeps its source until it is dropped. No `jolt-eval` objective moves.

## Design

### Architecture

#### The decoded row

```rust
#[repr(C)]
pub struct DecodedCycle { pub inc: u64, pub digits: u64 }   // 16 bytes
```

`digits` packs, from bit 0: the bytecode index (`b` bits), the RAM index (`a` bits), `Pos` (6 bits, the low digit first), `KeysDiffer`, `ShouldBranch` and `JalrLowBit` (one bit each) and the index of the fetched row's variant (6 bits, `Variant::COUNT = 58`). They fit: `81 + n(b) + n(a) ≤ 256`, which `Layout::new` checks, bounds `b + a` by 49, reached at `(3, 46)`, and `49 + 15 = 64`. `DigitFields::new(&Layout)` owns the offsets, checks the sum and gives a field `(shift, bits)` for each chunk of each index, each `Pos` digit, each flag and the variant. A chunk's offset inside its index is the sum of the widths of the chunks below it, which is how `Layout::bytecode_index` reads them. The replay packs through `DigitFields` and the source unpacks through it.

The row is a function of the committed row and of the fetched bytecode row: the indices and `Pos` are `Layout::{bytecode_index, ram_index, pos}`, the flags are the three columns, `inc` is `Layout::inc`. Every field is a value. A chunk digit of zero is the digit 0, which the committed row stores as no indicator, so no digit of a chunk or of `Pos` is absent.

`Rv64iWitness::replay` writes the table. `from_bits`, `from_facts` and `synthetic` all pass through it, and it is the one walk of the cycles a witness constructor makes. Per cycle it already holds the bytecode index, the fetched row and its variant, the RAM index, the committed row and `inc`, and it already rejects a chunk with two indicators (`MultipleIndicators`) and a row without a variant (`InvalidBytecode`). It gains `Layout::pos`, three bit reads, one pack and one increment: `variant_cycles[v]` counts the cycles of variant index `v`. The witness gains two fields, `decoded: Arc<[DecodedCycle]>`, allocated once at `T` and filled in place like `words`, and `variant_cycles: [u64; 64]`. `prove_inner` checks the length of `decoded` with those of `bits` and `words`.

`inc` repeats the first word of the committed row. It is in the decoded row so that a pass over words and digits reads 56 bytes per cycle from two streams, `words` and `decoded`, and not one word of each 32-byte row from a third.

#### Columns and words

`WitnessColumns::new(&Layout)` owns this table. With `d = d_b + d_a`:

| Digit column | Index | Bits | `by_row` | Digit at cycle `j` | Committed columns |
|---|---|---|---|---|---|
| bytecode chunk `c` | `c` | `bytecode_ra()[c].bits()` | no | its field; always `Some` | `Indicators` at `bytecode_ra()[c].start()` |
| RAM chunk `c` | `d_b + c` | `ram_ra()[c].bits()` | no | its field; always `Some` | `Indicators` at `ram_ra()[c].start()` |
| `Pos` digit `i < 2` | `d + i` | 3 | no | its field; always `Some` | `Indicators` at `pos_ra()[i].start()` |
| `Variant` | `d + 2` | 6 | yes | its field; always `Some`. `row_digit` is the index of `BytecodeRow::variant` | none |
| `ShiftKind`, `AccessKind`, `KeyKind` | `d + 3`, `d + 4`, `d + 5` | 3, 4, 3 | no | a table of 64 entries read at the variant field: the kind's index, or `None` | none |
| `Branch` | `d + 6` | 0 | no | the same kind of table: `Some(0)` for a conditional branch | none |
| `KeysDiffer`, `ShouldBranch`, `JalrLowBit` | `d + 7`, `d + 8`, `d + 9` | 0 | no | `Some(0)` when the flag bit is set | one `Flags` range at `keys_differ()` |

The four tables come from `Variant::{shift, access, key_kind, branch}`, with the kind indices that `RouteTensors::new` writes into a selector. Only `Variant` is `by_row`: it is the one factor of the one shape whose `Bytecode` slots `fold_pass` buckets per row, and every other `by_row` column would add a row cache and a comparison per cycle to the scan of `ValidatedTrace::new` for nothing. `Store` has no reader and no column.

| Word | Index | Read from |
|---|---|---|
| trace: `Rs1Value`, `Rs2Value`, `RdPreValue`, `RamReadValue`, `NextPC` | 0–4 | `words[j]` |
| trace: `Inc` | 5 | `decoded[j].inc` |
| bytecode: `Imm`, `FallThroughPC`, `PCPlusImm`, `PC` | 0–3 | `bytecode.rows()[k]` |

Four readers use the table and nothing else numbers a column or a word: `WitnessSource`; `router_shapes`, through `WitnessColumns::{word, factor}`, which map a `BankWord` to a `WordSlot` and a `Factor` to a digit column; the field lists of the two chunk products; and `WitnessColumns::column_map()`, the `ColumnMap` list of `g_pass_digits`: `Word { start: 0, trace_word: 5 }` and the `Indicators` and `Flags` ranges of the last column. `WitnessColumns::committed(y)` is the same correspondence read from the committed side, the pair (digit column, value) whose indicator column `y` is; a flag has the value 0.

#### Source and shared state

`WitnessSource` implements `CycleSource` over `Arc` clones of `words`, `decoded` and `bytecode`, a `DigitFields`, a `WitnessColumns` and the four kind tables. `cycles()` is `decoded.len()` and `bytecode_rows()` is `bytecode.rows().len()`. It holds nothing of size `T` of its own.

`SharedSource` is the value that `ProofSession::state_or_insert_with` keeps for what kernels of different batches share. It is inserted empty and filled by its own methods, because `ValidatedTrace::new` and `ScatterPlan::new` return a `Result` and the initialiser of the session cannot. The cycle group is a second state type, in `routers.rs`.

| Item | Built by | Used by | Released | When the other tier holds the slot |
|---|---|---|---|---|
| `Arc<ValidatedTrace<WitnessSource>>` | the first optimised `prepare` among batches 3a, 3b, 6b: `ValidatedTrace::new`, the one scan of the digits | every later optimised `prepare` of those batches | with the session | each adapter asks and builds on a miss; a reference kernel never asks |
| `Arc<ScatterPlan<WitnessSource>>` | the first optimised `prepare` among batches 3a, 3b | `fold_pass` in 3a, `claims_pass` in 3b | by the first optimised `park_residue` of 3b | 3a reference: 3b builds it. All of 3b reference: it stays to the end of `prove` |
| the cycle group | the first optimised `prepare` of batch 3b | the other optimised `prepare`s of 3b, each taking its member | by the first optimised `park_residue` of 3b | members of reference slots are never driven and drop with the group |

Nothing is handed over by `park` and `take`. A hand-over needs a producer and a taker that are both present, and in a mixed registry either may be a reference kernel; a state that is built on a miss needs neither. Three tables do not cross a batch at all. The fold tables are moved into `RouterShortCore::new` inside the `prepare` of batch 3a, and `ra_fold` is dropped there. The lifts are built from `x`, which exists only in batch 3b. The scatter that `claims_pass` returns has no taker in this spec and is dropped; the spec that covers `BytecodeReadAddress` adds its `park` with its `take`.

Session state and kernels are `MaybeAllocative`. Under the feature `allocative` the state types and the kernels implement `Allocative` with the fields of kernels-crate types skipped, so a heap snapshot attributes none of a core's tables.

#### Lanes

`Rv64iWitness::row(cycle) -> Result<WitnessRow, Rv64iProverError>` is the private `row` of `reference/spartan.rs` moved to the witness: the composition of the fetched row, the base words and the committed row into `WitnessRow::compute` has one owner for both tiers. `WitnessLanes::new(&Rv64iWitness)` builds `rows = RowSystem::new(&witness.layout)` once and makes one parallel pass over the cycles. Per cycle it takes `z = witness.row(j)`, the two triples `rows.lane_rows()[i].values(&z)`, and the tail byte from the first two of `rows.packed_rows()`, rows 128 and 129, whose values are 0 or 1 on every witness. It implements `LaneSource`.

No pass that exists can emit the lanes. They are functions of the derived words of `Words::compute`, which nothing on the proving path evaluates before batch 1, and of `NextPC`, which the replay writes one cycle late. The lanes pass is the first evaluation of the row words on the optimised path.

#### Outer

`prepare` builds the lanes and `OuterF2Core::new(Arc::new(lanes), relation.tau(), OuterF2Options::default())`. `tau()` is the 8 row coordinates and then the `t` cycle coordinates, the order of `τ`. `output_claims` returns `az`, `bz`, `cz` from `final_values()`. The adapter forms no derived term: `EqTau` is inside the core, from `tau` as passed.

#### Routers

The bank and the factors of a router are literals inside the private `Builder` of `routes.rs` today. They become data, and `Builder` and `router_shapes` both read it:

```rust
pub enum BankWord { Rs1Value, Rs2Value, RdPreValue, RamReadValue, NextPC, Inc, Imm, FallThroughPC, PCPlusImm, PC, One }
pub enum Factor { Variant, Pos(u8), ShiftKind, AccessKind, KeyKind, Branch, ShouldBranch }
pub struct Bank {
    pub words: &'static [BankWord],       // word slot n holds words[n]; One is the constant at bit 0
    pub committed: Option<Range<usize>>,  // Variant: columns g..used_columns(), y at source index 64·words.len() + y − g, then the constant
    pub factors: &'static [Factor],       // the digits a selector value spells, low factor first
}
pub fn bank(router: Router, layout: &Layout) -> Bank;
```

The tensors do not change, and `tests/routers.rs` pins them by digest.

`router_shapes(&WitnessColumns, &Layout, Option<&RouteTensors>)` returns the five `RouterShape`s in the order of `ROUTERS`. Word slot `n` is `WitnessColumns::word(words[n])`: a `Trace` or `Bytecode` slot, or `Bits([One])`. The committed range follows the words as `Bits` slots of 64 entries, entry `y − g` being `Indicator { column, value }` of `WitnessColumns::committed(y)`, and `One` after the last; `Zero` slots pad the bank to a power of two. The bit and word variables take `source_slots(router)`, the first six and the rest; the factors take `selector_slots(router)` in order, each as many as its column has bits. `S = 17`, `log_outputs = 10`, and `route` is `(column, source, selector)` of `RouteTensors::entries(router)`, or empty without tensors.

**`RouterShort`.** `prepare` takes the source and the plan from `SharedSource`, makes the shapes with `relation.routes()`, and runs `fold_pass(trace, shapes, relation.r_1(), plan, layout, &[])`. The `FoldLayout` is the default of the kernels spec: byte buckets for the eight most frequent selector values of the `Variant` shape, the lower value first among equal counts, and none for the other shapes. A selector value of that shape is a variant index, so the counts are `variant_cycles` and `selector_counts` does not walk the trace. `ra_fold` is dropped and the fold tables are moved into `RouterShortCore::new(shapes, relation.w(), folds)`. `output_claims` maps `Fold_ρ(x|_ρ)` to `variant`, `shift`, `memory`, `compare`, `branch`. `validate_derived_tables` compares the core's `Idle_ρ(x)·W_ρ(x|_ρ)` with `relation.derive_output_term` for `RouteWeight(ρ)` and returns `DerivedTableDrift` on a difference.

**The five `RouterCycle` members.** The first optimised `prepare` of batch 3b builds the cycle group: the shapes without tensors, `source_lift(trace, shapes, relation.x())`, `RoutersCycleCore::new(trace, shapes, relation.r_1(), relation.x(), tables)`, its five `members()`, the lifts and the plan. Every optimised `prepare` takes the member of its router and returns a kernel that holds the member and an `Arc` of the group. A relation whose `r_1()` or `x()` is not the group's, or a member taken twice, is `InvalidGeometry`. A kernel forwards `prove_round` and `finish_rounds` to its member and records the challenges, which are `r_3`. The first `output_claims` among the optimised members runs `claims_pass(trace, lifts, words, plan, r_3)` for trace words 0–4 and the four bytecode words, keeps the values in the group and drops the lifts and the group's `Arc` of the plan.

| Member | From `claims_pass`, at `(r_bit, r_3)` | From its member, `Sel_f(r_3)` |
|---|---|---|
| `Variant` | `rs1_value`, `rs2_value`, `rd_pre_value`, `imm`, `fall_through_pc`, `pc_plus_imm`, `pc`, `next_pc` | `variant` |
| `Shift` | `rs1_value` | `pos_ra_0`, `pos_ra_1`, `shift_kind` |
| `Memory` | `ram_read_value`, `rs2_value` | `pos_ra_0`, `access_kind` |
| `Compare` | `rs1_value`, `rs2_value`, `imm` | `pos_ra_0`, `pos_ra_1`, `key_kind` |
| `Branch` | `fall_through_pc`, `pc_plus_imm` | `branch`, `should_branch` |

An alias cell and its source are one value of the group, or the value of one shared factor column, so they are equal; against a reference kernel they are equal because both are the extension of one table at one point.

`variant_bits` is the one output that is neither. With `ω_n = eq(x|word, n)` over the word slots of the `Variant` shape and `c_V = eq(x|src, s)` for the source index `s` of its `One` entry,

`variant_bits = Source_V(x|src, r_3) + Σ_{n<8} ω_n·word_n + c_V`,

where `word_n` is the value of word slot `n` in the first row of the table. `prepare` computes the `ω_n` and `c_V` with `eq_table` from the shape, and `validate_derived_tables` of the `Variant` kernel compares them with `derive_output_term` for `WordSlot(n)` and `OneSlot`. The other four kernels hold no derived term.

#### Tail and the vector `C`

**Chunk products.** `BytecodeReadCycle` takes the `d_b` bytecode chunks with the points `relation.chunks()[c].1` and, for `[h_router, h_read, h_val, h_entry, h_next] = relation.folds()`, the weight `Dense(combined_weight(t, terms))` with the terms `Eq(h_router, r_3)`, `Eq(h_read, r_4)`, `Eq(h_val, r_5)`, `Eq(h_entry, 0^t)`, `Next(h_next, r_3)`. `RamRaProduct` takes the `d_a` RAM chunks with its `chunks()` and the terms `Eq(c.read, r_4)`, `Eq(c.val, r_5)` for its challenges `c`. `output_claims` returns `chunks[c] = Ra_c(r_6)`. `validate_derived_tables` compares `W(r_6)` of `final_values()` with the same combination of the relation's derived terms: `Weight(·)` under `folds()`, and `EqRead`, `EqVal` under the challenges. Both weights are `Dense`, the default of the kernels spec.

**`BitsReduction`.** With `c` the challenges and `relation.weights()` the six sparse supports in the order direct, variant, `Pos` low, `Pos` high, `ShouldBranch`, `Inc`, each expanded to 256 entries:

`L_1 = c.direct_columns·weights[0]`, `L_3 = c.variant_bits·weights[1] + c.pos_ra_0·weights[2] + c.pos_ra_1·weights[3] + c.should_branch·weights[4]`, `L_5 = c.inc·weights[5]`.

`g_pass_digits(trace, columns.column_map(), [L_1, L_3, L_5])` gives three tables, moved into `ReductionCore::new` with three legs of coefficient 1, at `r_1`, `r_3` and `r_5`. With `[z_0, z_1] = relation.pos_zero()` their claims are

`c.direct_columns·direct_columns`, `c.variant_bits·variant_bits + c.pos_ra_0·(pos_ra_0 + z_0) + c.pos_ra_1·(pos_ra_1 + z_1) + c.should_branch·should_branch`, `c.inc·inc`.

They regroup the member's input expression by cycle point. In round 0 the kernel checks that they sum to the claim it is handed and returns `SumcheckError::RoundCheckFailed` otherwise.

**The vector `C`.** `C` is the typed output `columns` of `BitsReduction`. The `curate` of stage 6b moves `claims.bits_reduction.columns` to the wire and `stage6b::prove` returns it as `BitsColumns`. It needs no slot of its own. The kernel keeps a clone of the `Arc` of `bits` and records its challenges, and `output_claims` returns `column_pass(&bits, r_6)`. The `final_values()` of the core are not outputs.

#### Chunk digits in place

`ChunkProductCore::new` copies the digits of its columns into one byte per column and cycle before its first round. In batch 6b that is two more walks of digits that `ValidatedTrace::new` and `g_pass_digits` have read, ten bytes written per cycle for the two members, and 40 MiB at `2^22` cycles. The decoded row is already the compact and validated form those bytes are: a chunk digit is a field of `digits`, present on every cycle and below `2^bits` by the construction of the row. The chunk cores read the decoded rows in place and the copy goes. The kernels crate gains

```rust
pub trait DigitWords: Send + Sync + 'static {
    fn cycles(&self) -> usize;
    fn digit_word(&self, cycle: usize) -> u64;
}
pub struct DigitField { pub shift: u8, pub bits: u8 }

impl ChunkProductCore {
    pub fn from_fields<W: DigitWords>(
        words: Arc<W>,
        fields: Vec<DigitField>,
        points: Vec<Vec<F128>>,
        weight: ChunkWeight,
    ) -> Result<Self, ChunkProductError>;
}
```

where digit `c` of cycle `j` is the `bits_c` bits of `digit_word(j)` from bit `shift_c`. The constructor checks one to seven fields, `bits ≤ 8`, `shift + bits ≤ 64`, and the points and the weight as `new` does. It reads no digit. `WitnessSource` implements `DigitWords`, and the two adapters pass the fields of their chunks from `DigitFields`. A lazy round reads one word per cycle at a stride of 16 bytes where it read `d` adjacent bytes; nothing has measured that read against the other.

A missing digit stays a deterministic error where one can occur. `new` over `DigitColumns` keeps `MissingDigit { column, cycle }`. A field cannot be absent, so `from_fields` has no such case to detect: the absence is excluded by the type and not by a scan. This is a change request to the kernels spec, to its Goal, its invariant 3, the "Chunk columns" row of its Memory table and the gather line of its tail model; its owner decides the internals. Until it lands, the adapters call `new(DigitColumns::from_validated(trace, columns), …)`, which proves the same bytes with the copy.

#### Registry

`Rv64iBackend::optimized()` is `reference()` with the ten slots of the adapter table replaced. There is no constructor per stage: every field of `Rv64iBackend` and of the `Stage*Kernels` is public, and a mixed registry is one of the two constructors with fields reassigned, which is how the tests build theirs.

#### What this spec needs from its neighbours

- **Kernels: `ChunkProductCore::from_fields`**, above. Without it nothing fails and invariant 6 holds on the adapter side only.
- **Kernels: a `RoutersCycleCore` driven on a subset of its members.** For each member that is driven, the messages and final values are those it emits when all five are driven, whichever subset is driven and whichever handle is first. Mixed registries need it, and so does `run_lockstep_checked`, which drives one member. Without it the five adapters are correct only when all five slots are optimised.
- **Kernels: named results.** The surface table of the kernels spec does not name the accessor of the short core's two values per shape, of a member's `Source_ρ(x|src, r')` and `Sel_f(r')` in the order of its factors, the type of the lifts between `source_lift` and `claims_pass`, or the order of the values `claims_pass` returns. The adapters are written against those four.
- **Kernels: two costs.** `ValidatedTrace::new` scans the digits on one thread, inside the timed `prepare` of `adapters/routers` and `adapters/tail`; the 12-thread requirement needs that scan parallel. `OuterF2Core` keeps its source through the cycle rounds; released when the core materialises its tables, the lanes would drop there.
- **Kernels: the default `FoldLayout`.** The number of byte-bucketed selector values, eight, is a default of the kernels spec that `RouterShortPrepare` restates. A constructor of the default layout from the selector counts of each shape would own it.
- **Protocol spec.** The listing of §12 gains `decoded` and `variant_cycles`.

### Alternatives Considered

- **The stage functions with stubbed slots, for the bench.** Performance, "What it drives".
- **The digit copy stays in the chunk cores.** It keeps the core independent of the source's layout at ten bytes written per cycle, 40 MiB at `2^22` and a second owner of digits the decoded row already holds valid. The independence is kept by the trait `DigitWords`, which names no layout.
- **The decoded rows built by the adapters, in a parallel pass at first use.** It leaves the witness as it is and is a second walk of the committed rows, where the replay holds every field of the row as it goes.
- **The lanes emitted by the replay.** The replay is sequential and does not evaluate `Words::compute`. It would carry the most expensive computation per cycle of the witness in its one loop, patch each row one cycle late for `NextPC`, and keep 49 bytes per cycle alive from the witness to the end of the proof for a table that batch 1 alone reads.
- **A slot for `C`.** `C` is a typed output of a member that has a slot. A second slot would be a second producer of one value and a registry field that is not a sum-check member.
- **`park` and `take` between batches 3a and 3b.** They fail in a mixed registry whichever side is the reference kernel, and the tables the kernels spec names for them do not cross the batch.
- **One `prepare` for the five cycle members.** The registry is a struct of one `PrepareKernel` per member, and a mixed registry replaces one field. A group in the session keeps that and builds the shared core once.
- **The bank restated in `router_shapes`.** A wrong bank is caught by "Lockstep", and it would be a second statement, in production code, of what `routes.rs` fixes.
- **`by_row` for the four kind columns.** They are functions of the row. No pass of the five shapes uses the declaration for them, and the scan pays for it.

## Documentation

No change to the Jolt book: nothing here is reachable from the SDK. Rustdoc on `DecodedCycle` and `DigitFields` states the packing and its bound; on `WitnessColumns`, that it is the one numbering; on each `…Prepare`, its core, what it shares and what it pins in `validate_derived_tables`; on `SpartanOuterF2Prepare`, invariant 9 in those terms: required of the witness, not checked, detected by the driver's comparison of the final claim.

## Execution

| Item | Files | Criteria | Starts after |
|---|---|---|---|
| **1. Sources** | prover `Cargo.toml` (the dependencies on `jolt-rv64i-kernels` and `rayon`; every `[[test]]` and `[[bench]]` of this spec, with one-line files so that the manifest resolves); `src/plane.rs`, `src/prover.rs`, `src/reference/spartan.rs`, `src/optimized/{mod, source}.rs`; `tests/optimized_sources.rs` | Decoded rows; Source against the committed table | nothing |
| **2. Outer** | `src/optimized/outer.rs`; `tests/optimized_outer.rs` | Lanes; Lockstep, one adapter; A witness whose rows do not hold | the first commit of 1 |
| **3. Routers** | verifier `src/public/routes.rs`; `src/optimized/routers.rs`; `tests/optimized_routers.rs` | Shapes; Lockstep, six adapters; Batch outputs, 3b | the first commit of 1; the router cores of the kernels crate |
| **4. Tail** | `src/optimized/tail.rs`; `tests/optimized_tail.rs` | Lockstep, three adapters; Batch outputs, 6b | the first commit of 1 |
| **5. Registry and bench** | `src/backend.rs`; `tests/optimized_registry.rs`; `benches/adapters.rs`, `benches/support/` | Mixed registries; End-to-end corpus; Rejected inputs; Bench; Performance; Trace-sized allocations; Gates | the first commit of 1 for the bench; 2, 3 and 4 for the tests |

The first commit of item 1 is `plane.rs` and `source.rs` complete with `Cargo.toml` and the one-line files; its test follows. Items 2, 3 and 4 then share no file. Item 5 writes the generated program, the bench runner and the registry test over slot masks against `reference()` while they are open, and switches slots as they land. Item 3 is the only one that edits the verifier crate, and the only one whose cores are not yet in the kernels crate: `router/{short, lift, cycle, claims}.rs` are one line each on this branch.

## References

- `specs/rv64i-binary-protocol.md`: §3 points, §5 batches and names, §8.4–8.9 routers, §8.17 and §10 the reduction and `C`, §12 the witness, the registries and the reference tier, the "Proof vector" criterion.
- `specs/rv64i-binary-prover-kernels.md`: Goal and its surface table, invariants 3 and 6, Performance, Design "Fit with `jolt-kernels` and the protocol".
- `specs/rv64i-binary-arithmetisation.md`: `Layout`, `RowSystem`, `WitnessRow`.
- `specs/binary-protocol-family-seams.md`, `specs/binary-kernel-primitives.md`: the plane parameter of `PrepareKernel`; `run_lockstep_checked`.
- `crates/jolt-kernels/src/{kernel.rs, backend.rs, optimized/parity.rs}`; `crates/jolt-prover/src/driver.rs`, the order of `prepare`, rounds, `validate_derived_tables`, `output_claims`, `park_residue` and the final-claim comparison.
- `crates/jolt-rv64i-prover/src/{plane.rs, backend.rs, prover.rs, stages/, reference/}`; `crates/jolt-rv64i-verifier/src/{public/routes.rs, stages/}`; `crates/jolt-rv64i-kernels/src/{source.rs, router/, chunk_product.rs, reduction/, column_pass.rs, outer_f2/}`.
- `.claude/skills/test-policy/SKILL.md`.
