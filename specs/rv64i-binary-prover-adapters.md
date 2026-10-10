# Spec: Adapters for the Packed-Bit Kernels of RV64I over the Binary Field

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-10                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The experiment in RV64I hash-based Jolt over binary fields has a protocol that `prove` and `verify` run end to end on reference kernels (`specs/rv64i-binary-protocol.md`), and a crate of cores that are held to a number of nanoseconds per cycle (`specs/rv64i-binary-prover-kernels.md`) and that `prove` cannot reach. This spec is the layer between them: the per-cycle tables of the witness that the cores read, one checked preparation of the source that the routers and the tail share, a `PrepareKernel` for each of ten members that wraps a core, a constructor of the registry, and the two tests of the layer, which are equality of the proof bytes with the reference registry under mixed slots and a bench at `2^20` and `2^22` cycles. The layer adds no protocol and no arithmetic. What it can get wrong is an order, an index, a second walk of the trace or a copy of a table, and each of those has one owner here.

## Intent

### Goal

Put the cores of `jolt-rv64i-kernels` behind the stage registries of `jolt-rv64i-prover` for `SpartanOuterF2`, `RouterShort`, the five `RouterCycle` members, `BytecodeReadCycle`, `RamRaProduct` and `BitsReduction` with the vector `C`, so that `prove` with any subset of those ten slots optimised emits the bytes the reference registry emits, at the cores' thresholds plus an itemised cost of the layer.

**Notation.** The points `r_1`, `w`, `x`, `r_bit`, `r_3`, `r_4`, `r_5`, `a_bc`, `a_ram`, `r_6` and the symbols `t`, `T`, `b`, `a`, `d_b`, `d_a`, `n(·)` are those of `specs/rv64i-binary-protocol.md` §2, §3 and §5; `g` is the first committed column of the RAM chunks, `layout.ram_ra()[0].start()`; `K` is the number of bytecode rows and `κ = K/T`. Cores, passes, shapes, `Fold_ρ`, `Source_ρ`, `Sel_f` and the restrictions `x|_ρ`, `x|src` are those of the Goal of `specs/rv64i-binary-prover-kernels.md`, "the kernels spec" below, and the unit costs `M`, `L`, `Bk`, `sct`, `w` are those of its Performance section at point 1. Both specs list a point low variable first, so a point crosses this layer unchanged. A *slot* is a field of a registry (`Stage3bKernels::shift`); the variables of the short sum-check are *short slots* and the entries of a bank *word slots*.

**Adapters.** Each row is one `PrepareKernel<F128, Member, Rv64iPlane>` whose `prepare` returns a `'static` `SumcheckKernel`.

| Member (slot) | Core and passes | Shared state |
|---|---|---|
| `SpartanOuterF2` (`stage1.spartan_outer_f2`) | the lanes pass; `OuterF2Core` | none |
| `RouterShort` (`stage3a.router_short`) | `fold_pass`; `RouterShortCore` | prepared source, scatter plan |
| `RouterCycle{Variant, Shift, Memory, Compare, Branch}` (`stage3b.*`) | `source_lift`; one `RoutersCycleCore` for the five; `claims_pass` | prepared source, selector bytes, scatter plan, the cycle group |
| `BytecodeReadCycle` (`stage6b.bytecode_read_cycle`) | `combined_weight`; `ChunkProductCore` | prepared source, its chunk bytes |
| `RamRaProduct` (`stage6b.ram_ra_product`) | `combined_weight`; `ChunkProductCore` | prepared source, its chunk bytes |
| `BitsReduction` (`stage6b.bits_reduction`) | `g_pass_digits`; `ReductionCore`; `column_pass`, whose result is `C` | prepared source |

**Surface.**

| Item | Where | Contract |
|---|---|---|
| `DecodedCycle`, `DigitFields`; `Rv64iWitness::{decoded, variant_cycles, cycles}`; `WitnessCycles`, `CycleParts` | `plane.rs` | Design, "The decoded row", "Lanes" |
| `WitnessColumns`, `WitnessSource`, `SharedSource` | `optimized/source.rs` | "Columns and words", "Source and shared state" |
| `WitnessLanes`, `SpartanOuterF2Prepare` | `optimized/outer.rs` | "Lanes", "Outer" |
| `router_shapes`, `RouterShortPrepare`, `RouterCycle{Variant, Shift, Memory, Compare, Branch}Prepare` | `optimized/routers.rs` | "Routers" |
| `BytecodeReadCyclePrepare`, `RamRaProductPrepare`, `BitsReductionPrepare` | `optimized/tail.rs` | "Tail and the vector `C`" |
| `Rv64iBackend::optimized` | `backend.rs` | "Registry" |
| `BankWord`, `Factor`, `Bank`, `bank` | `jolt-rv64i-verifier`, `public/routes.rs` | "Routers" |

Every optimised `…Prepare` is a unit struct with `Default`, like its namesake in `reference/`.

### Invariants

1. **No protocol change.** No message, round count, member, transcript event or byte of the wire changes. `prove_batch`, `SequentialRounds`, the generated stage drivers, `PrepareKernel`, `SumcheckKernel`, `ProofSession`, `KernelSlots` and the plane parameter are used as they are. What this spec asks of `jolt-rv64i-kernels` is the list at the head of Execution, and of the arithmetisation the evaluator of Design, "Lanes".
2. **Mixed registries.** Every assignment of the ten slots to the two tiers proves, and proves the same bytes. No optimised `prepare` assumes that another slot is optimised: what two adapters share is built by whichever needs it first and found by the other (Design, "Source and shared state").
3. **One owner per conversion.** The packing of the decoded row is `DigitFields`. The numbering of digit columns, trace words and bytecode words, and their correspondence with the committed columns, is `WitnessColumns`. The bank and the selector factors of each router are `routes::bank`. The opening point of a typed output is the relation's, from `derive_opening_points`. An adapter passes `tau`, `r_1`, `x` and the chunk points whole to a core or a pass; it reverses, permutes and slices none, and the only equality helper it calls is `round::eq::eq_table` of the kernels crate. The order of a fold table is the core's, increasing occupied short slots, and is not the order of the typed fold point, source slots and then selector slots; the two differ for `Memory` and `Compare`. No adapter converts between them, because a fold crosses this layer as a table moved into `RouterShortCore` and comes back as the value `Fold_ρ(x|_ρ)`.
4. **No relation expression.** An adapter evaluates no input or output expression; the stage does, through `Expr::try_evaluate`. An adapter writes a formula of the protocol in three places, and this is what pins each. `validate_derived_tables` of `RouterShort` compares five scalars, the core's `Idle_ρ(x)·W_ρ(x|_ρ)`, with the relation's `RouteWeight(ρ)`; that of `RouterCycleVariant` compares the eight coefficients `WordSlot(n)` and `OneSlot`; that of a chunk product compares one scalar, the weight at `r_6`, with the same combination of the relation's derived terms. The three leg claims of the reduction are pinned in round 0 by their sum alone. A wrong split of that sum between the legs, or a wrong weight vector, yields messages of another polynomial; it surfaces at the driver's comparison of `expected_final_claim` with the final claim, with the soundness error of the sum-check, and not at the point of the mistake. Each scalar comparison is at one point, and detects a wrong table with the error of an evaluation at a random point.
5. **One pass, one owner.** This layer creates two tables of `T` elements, the decoded rows and the lanes, and makes one walk of the source's digits, the preparation. The decoded rows are written by the replay the witness already makes; no adapter walks the committed rows to decode them. The lanes are written by one parallel pass. The preparation validates every digit and writes, in the same walk, the bytes that later passes would otherwise gather.
6. **Trace-sized copies.** An adapter moves a table into a core by value or shares it by `Arc`. An adapter that copies a table of `T` or `K` elements which the witness, a pass or a core holds is a defect, with three exceptions, each a deliberate compact form with one owner at a time and a stated release:

   | Copy | Size, at `2^22` | Owner | Released | The count that justifies it |
   |---|---|---|---|---|
   | Selector bytes: the eight distinct factor columns of the routers | `8T`, 32 MiB | `SharedSource`, then `RoutersCycleCore` | the fourth bind of batch 3b | the cycle rounds read 8 adjacent bytes per cycle and pass, where the source gives them a 16-byte row and four table lookups |
   | Chunk bytes: the `d_b + d_a` chunk digits | `10T`, 40 MiB at the reference layout | `SharedSource`, then the two `ChunkProductCore`s | the fourth bind of batch 6b | a core's lazy family reads column by column in five passes: 25 bytes read and 5 written per cycle and core, against 80 read from the 16-byte rows |
   | `Inc` in the decoded row | `8T`, 32 MiB | the witness | with the witness | a pass over words and digits reads two streams, 56 bytes per cycle, and not one word of each 32-byte committed row from a third |

   A group of bytes that no core takes, which happens only when the slot of its taker holds a reference kernel, is dropped with the session.
7. **Boundary.** The adapters live in `jolt-rv64i-prover/src/optimized/`. `jolt-rv64i-kernels` gains no dependency on a relation type or on the arithmetisation. `prepare` clones the `Arc`s of the witness tables a core reads later, and nothing borrowed from the witness outlives it.
8. **Total functions.** `prepare` returns `KernelError::InvalidGeometry` with the display of the core's or the source's error; the round and output methods return the errors of their traits. Nothing panics on a malformed witness or geometry.
9. **Honest witness.** `OuterF2Core` requires `C = A & B` on every lane word and tail bit and does not check it. The precondition of the optimised outer slot is therefore a witness whose rows hold. On a witness that violates a row no step rejects deterministically: the core's messages are those of another polynomial, and the violation is detected algebraically, at the driver's comparison of `expected_final_claim` with the final claim in `impl_stage_prover!`, with the soundness error of the outer sum-check. The detection is over the choice of `tau` and of the round challenges and is not certain: a `tau` with a Boolean coordinate, for one, gives the rows on the other side of it weight zero. The reference kernel detects the same violation at a round check, with the same kind of error. No adapter adds a check, and `OuterF2Core::check_rows` is a diagnostic that production does not call.
10. **Determinism.** The bytes do not depend on the thread count, nor on which optimised member of a batch is prepared, driven or extracted first. The error the preparation returns does not depend on the thread count either.
11. **Characteristic 2.** No adapter halves, scales by an integer or samples at an integer; coefficients are challenges and claims, combined with `+` and `·`.

No `jolt-eval` invariant changes.

### Non-Goals

- Cores or adapters for `SpartanOuterF128`, `SpartanInner`, batches 4 and 5 and `BytecodeReadAddress`. Their eight slots keep reference kernels. A second spec covers them: its adapters are `PrepareKernel`s that `Rv64iBackend::optimized` assigns to those fields, with session state types of their own. The pass of `SpartanOuterF128` over the rows is part of that spec; "Lanes" states what it can call here.
- The internals of a core.
- Layouts with more than seven chunks of one index, `b > 28` or `a > 28`. `ChunkProductCore` takes at most seven columns. The adapter of that member returns `InvalidGeometry` naming the count before it builds a table, the preparation is asked for no bytes of that index, and such a layout uses the reference slot.
- The time of `prove`. With eight slots on reference kernels it measures the reference tier.

## Evaluation

### Acceptance Criteria

The five programs are `PROGRAMS` of `tests/support`. The *separating fixture* is one executed program of `tests/support`, outside `PROGRAMS`, at `t = 6`, whose instructions use pairwise different registers with pairwise different values and which executes a shift, a load, a store, a comparison, a taken and an untaken branch and a `JALR`. "Seeded" points have pairwise distinct coordinates.

- [ ] **Decoded rows.** For the five programs at `t = 6` through `from_facts`, for `from_bits` on their committed rows, and for `Rv64iWitness::synthetic` at `(b, a) = (4, 5)` and `(10, 14)`: every field of every `DecodedCycle`, read through `DigitFields`, equals `Layout::{bytecode_index, ram_index, pos, inc}` and the three flag columns of the committed row and the variant index of the fetched row; `variant_cycles` is the number of cycles of each variant; `from_bits` and `from_facts` give equal tables. `DigitFields::new` at `(3, 46)` puts the variant in bits 58–63. (`tests/optimized_sources.rs`)
- [ ] **Source against the committed table.** On the same witnesses, for every cycle `j` and column `y < used_columns()`: `Bits[y, j]` is bit `y` of trace word `Inc` for `y < 64`, and otherwise is `digit(c, j) == Some(v)` for `(c, v) = WitnessColumns::committed(y)`. The four kind digits equal `Variant::shift`, `Variant::access().and_then(|a| a.kind)`, `Variant::key_kind` and the presence of `Variant::branch` on the fetched row, and the trace and bytecode words equal the fields of `CycleWords` and `BytecodeRow`. The preparation accepts the source, and every byte of the three groups it returns is the digit of its column and cycle, plus one in the selector group and zero there for an absent digit. `g_pass_digits` under `WitnessColumns::column_map()`, with a seeded weight on every used column, equals `Σ_y L[y]·Bits[y, j]` summed over the committed rows. (`tests/optimized_sources.rs`)
- [ ] **Lanes.** On the five programs and the separating fixture, `WitnessLanes` equals `LaneRows::values` of both lane families and the first two `PackedRow::values` of `WitnessCycles::row(j)` for every cycle, and `OuterF2Core::check_rows` accepts it. The fixture's `JALR` cycle is a cycle on which the adder lane depends on `NextPC`. (`tests/optimized_outer.rs`)
- [ ] **Shapes.** At `(4, 5)`, `(10, 14)`, the reference layout and `(3, 46)`: `router_shapes` returns five shapes that `RouterShape::new` accepts, whose bit, word and selector slots are `source_slots` and `selector_slots` of the router and whose `route()` is `RouteTensors::entries` of the router. The `Bits` slots of the `Variant` shape are 9 and 10 at the reference layout and 9 to 11 at `(3, 46)`. `RouteTensors::new`, reading `bank`, gives the tensors whose digests `tests/routers.rs` pins. On the five programs, `selector_counts` of the `Variant` shape equals `variant_cycles`. (`tests/optimized_routers.rs`)
- [ ] **Batch outputs.** On the separating fixture, one proof by the reference registry is run batch by batch and the inputs, the `BatchProof` and the `Output` of every batch are kept; each reference kernel runs once. The test first asserts that the fixture separates: among the kept values, the five folds of batch 3a are pairwise different, the 18 unique values of batch 3b are pairwise different, the `d_b + d_a` chunk values of batch 6b are pairwise different, and the six input claims of `BitsReduction` are pairwise different and nonzero. Then, against the kept proof and output of the batch, and of `BitsColumns` for 6b: batch 1 with the outer slot optimised; batch 3a with `RouterShort` optimised; batch 3b under each of its 31 non-empty slot masks, in an empty session and in a session in which the optimised batch 3a has just run; batch 6b under each of its 7 non-empty masks, in an empty session and in one whose `SharedSource` already holds the prepared source. That is 78 runs under masks. A failure names the first round whose message differs, or the first output. (`tests/optimized_{outer, routers, tail}.rs`)
- [ ] **Mixed registries.** For the program of the "Proof vector" criterion of `specs/rv64i-binary-protocol.md`, the counting loop at `t = 6` with `TransparentBits`: `prove` returns a proof whose `to_bytes` is that of `Rv64iBackend::reference()`, the encoding whose digest `tests/wire.rs` fixes, under `Rv64iBackend::optimized()`; under each of the ten slots optimised alone; under the five slots of batch 3b; under the six router slots; under the three slots of batch 6b; and under `RouterShort` with the three slots of batch 6b, which leaves the selector bytes and the plan in the session to its end. (`tests/optimized_registry.rs`)
- [ ] **End-to-end corpus.** For each of the five programs at `t ∈ {6, 8, 10}`, `prove` under `optimized()` returns the bytes it returns under `reference()`. The three tests of `tests/e2e.rs` that expect `prove` to fail, in batches 4, 4 and 6a, fail in the same batch with the same error under `optimized()`. (`tests/optimized_registry.rs`)
- [ ] **A witness whose rows do not hold.** On a counting-loop witness changed so that `OuterF2Core::check_rows` names a cycle, `prove` with the optimised outer slot returns `VerifierError::StageClaimSumcheckFailed` for batch 1 and no proof. The transcript of the test is fixed, so its outcome is; invariant 9 states what the outcome rests on. (`tests/optimized_outer.rs`)
- [ ] **Rejected inputs.** Each returns `InvalidGeometry`: a `RouterCycle` member prepared twice in one session; a chunk-product member prepared twice in one session, whose bytes are gone; a `RouterCycle` member prepared in a session whose prepared source was asked for no selector bytes. A witness whose `decoded` has another length than `bits` is rejected by `prove` with its dimension error. The bound of seven chunks has no test: its least input is a witness of `2^29` RAM words or bytecode rows. (`tests/optimized_registry.rs`)
- [ ] **Bench.** `benches/adapters.rs` reports the six ids of Performance at `log_t` 20 and 22 on 1 and 12 threads, each with the dynamic mix of its witness.
- [ ] **Performance.** A result above its threshold at `log_t = 22` fails the PR that adds or changes the adapter. A threshold moves by the rule of Performance and not to meet a result.
- [ ] **Trace-sized allocations.** The recorder of the bench lists every allocation of `T` bytes or more made during a pipeline, with its size and phase. The PR that adds an adapter sets the list against the Memory tables below and that of the kernels spec; an allocation in none of them fails the PR, and so does a recorder that overflowed.
- [ ] **Gates.** `cargo clippy --all-targets -- -D warnings` and `cargo nextest run --cargo-quiet` pass for `jolt-rv64i-prover --features test-utils`, `jolt-rv64i-verifier`, `jolt-rv64i-kernels` and `jolt-rv64i-arith`; `cargo fmt --check` passes.

### Testing Strategy

The ground truths are the reference tier, the frozen digest of `tests/wire.rs`, the committed rows and the arithmetisation: `Layout`, `Variant`, `RowSystem`. No test has an oracle of its own for a relation, and none compares two optimised configurations. The repository's test rule (`CLAUDE.md`, "Tests") lists "`jolt-kernels`' reference tier" among the independent ground truths of a permanent test. `.claude/skills/test-policy/SKILL.md` does not name that tier. It says that "Protocol parity and golden fixtures can be (d)", its class "Protocol / wire / format compatibility: verifier fixtures, frozen wire or transcript digests, spec vectors, golden files", and it lists for deletion a test "whose oracle is a second implementation of the same rule". The tests here are parity of the protocol's bytes with the reference tier, kept under the first rule and class (d), and the reference bytes they compare with are the ones whose digest `tests/wire.rs` freezes.

Equality of bytes detects a wrong order only where the two values differ. An exchange of `pos_ra_0` with `pos_ra_1`, both of three variables and both present on every cycle, leaves the bytes as they are on a witness whose two `Pos` digits are equal on every cycle, as they are when every position is zero; `Rs1Value` and `Rs2Value` agree on every cycle of a program whose instructions name one register. The separating fixture exists for that, and "Batch outputs" asserts that it separates before it compares anything.

"Any subset of the ten slots" is 1,024 registries, and the criteria reach them in two steps. The bytes of a batch are a function of its inputs, of its kernels and of what the session holds. At the entry of batch 3b the session is empty or holds the prepared source with its bytes and the plan; at the entry of batch 6b it is empty or holds the prepared source and the chunk bytes, beside state the tail does not ask for: the plan, an untaken group of selector bytes. The cycle group does not outlive batch 3b. "Batch outputs" therefore exhausts the masks of each batch that has more than one of the ten slots, with the session in both states, and runs the one slot of batches 1 and 3a; by induction over the batches every registry proves the bytes of the reference registry. A mask varies which kernels share state and not the data, so the masks run on one witness, 78 runs where each of five programs with its own baselines would make 400. The data vary in "End-to-end corpus", fifteen whole proofs, and in "Source against the committed table". "Mixed registries" checks the induction on whole proofs for 15 registries, among them the one that leaves state in the session which the tail must ignore.

No criterion drives a kernel alone against its reference kernel round by round. Each assertion such a run makes is made by the driver in a batch run, which validates the derived tables and compares the relation's output expression with the final claim, or follows from equality of the batch proof, which the test reports by round. No criterion asserts that `verify` accepts a proof whose bytes equal those of a proof `tests/e2e.rs` verifies, and none compares transcript events: no adapter sees a transcript. Seeds are fixed. Every existing test of `jolt-rv64i-prover`, `jolt-rv64i-verifier`, `jolt-rv64i-kernels` and `jolt-rv64i-arith` passes unchanged; `tests/routers.rs` passes with `RouteTensors::new` reading `bank`. The `host` and `zk` modes do not apply.

### Performance

The bench is `benches/adapters.rs` of `jolt-rv64i-prover` (`harness = false`, feature `test-utils`), on the machine, flags, sizes and thread counts of the kernels spec's Performance section. Ids are `adapters/<name>/<log_t>/<threads>`. It reports wall time per cycle by phase (`prepare`, `rounds`, `finish`, `extract`, `park`), the peak live bytes above the witness, the decoded rows apart, and the trace-sized allocations.

**What it drives.**

- `source`: the preparation, as a router adapter asks for it, in an empty `ProofSession`.
- `lanes`: `WitnessLanes::new`.
- `outer`: `SpartanOuterF2Prepare::prepare`, which contains the lanes pass, and then every call the driver makes on a member, in its order: the rounds under challenges from a fixed seed with the round check `prove_batch` makes, `finish_rounds`, `validate_derived_tables`, `output_claims`, `park_residue`, and the comparison of the relation's output expression on the outputs with the final claim. It runs outside a batch, because the other member of batch 1 has no core.
- `routers`: the generated driver of batch 3a and then that of batch 3b (`StageProver::prove`, as `stage3a::prove` and `stage3b::prove` call it) in one session, batch 3b taking its claims and `x` from the output of 3a. The session is *warm*: it holds the prepared source, made before the timer by the call the adapters make. The scatter plan is built inside, as the kernels spec assigns it to the construction of its `routers` pipeline.
- `tail`: the generated driver of batch 6b, which ends with `C`, in a warm session.
- `session`: from an empty session, the drivers of batches 3a, 3b and 6b in that order. It is the one run in which the chunk bytes live across batches.

The inputs are built before any timing, by direct summation over the witness and never by a reference stage or a reference kernel: at `2^22` cycles three dense tables of 256 rows are 48 GiB. The input claim of `RouterShort` comes from one `fold_pass` of the setup and the route tensors, the two chunk-product claims from two sums over the decoded rows, the six claims of the reduction from three calls of `column_pass`, at `r_1`, `r_3` and `r_5`, and `BitsReduction::weights()`. A wrong setup claim cannot pass for a result in `routers` or `tail`, since the driver compares `expected_final_claim` with the final claim of `prove_batch`. `outer` has no setup claim: its member has no input and its sum is zero.

Batches 3a, 3b and 6b have no uncovered member, so their own drivers run. Batch 1 has one, and a stub in its place would have to be an honest prover of its relation, which at this size is the core the second spec has not built; so the outer kernel runs alone. The members of batch 3b share one pass per round and those of batch 6b one source and one `column_pass`, so a member alone has no time of its own, and the pipeline is gated, as the kernels spec gates `routers` and `tail`.

**Witness.** One executed witness per size. The program is seeded and straight-line: `2^20` bytecode rows, the first an `AUIPC` into a register that no other row writes, the last a `JALR` through that register to the second row, and between them the instruction mix of the kernels spec's `all_rows` profile. A `JAL` cannot close the loop: it reaches `2^20` bytes either way and the rows span `2^22`. The bench's own setup runs it on the interpreter of `tests/support` for `2^log_t` cycles over `2^20` RAM words and builds the witness with `Rv64iWitness::from_facts`; it does not use `program_fixture`, which requires termination. The run is cut at `2^log_t` cycles and reaches no exit: the bench proves batches and no statement, and `from_facts` takes the final program counter from the last cycle. It is executed because the outer adapter needs every row to hold, which `Rv64iWitness::synthetic` does not give. Taken branches skip rows, so the executed mix is not the static one, and the bench prints with every figure the dynamic mix of the witness it was measured on: the cycles with a shift kind, a RAM access, a key kind, a taken branch, a `JALR`, a register write, and the number of visited rows. At `2^22` the principal arrays of `from_facts` are 640 MiB: the facts at 72 bytes per cycle (288 MiB), the committed rows at 32 (128), the words at 40 (160) and the decoded rows at 16 (64), beside about 40 MiB of bytecode rows and 8 MiB of final RAM. The facts and the interpreter are dropped before any measurement.

**Thresholds.** `θ(id)` is the threshold of that row of the Requirements table of the kernels spec, which owns the number and its rule. Each core keeps its threshold; the layer adds three items, each from a count. Single thread, nanoseconds per cycle:

| Bench | Threshold | With the thresholds in force: `22`, `20` |
|---|---|---:|
| `adapters/source` | the preparation, `σ` | 8, 8 |
| `adapters/lanes` | the lanes pass, `λ` | 40, 40 |
| `adapters/outer` | `θ(outer_f2/local) + λ` | 294, 294 |
| `adapters/routers` | `θ(routers/all_rows) + ζ`, `ζ` the setup of the routers | 299, 377 |
| `adapters/tail` | `θ(tail/local)` | 192, 192 |
| `adapters/session` | `σ` and the two rows above | 499, 577 |

The last column evaluates the formulas and defines nothing. On 12 threads at `log_t = 22` the requirement is wall time per cycle, the threshold divided by 9.6, as in the kernels spec; `ζ` is serial and is not divided.

- `σ`. At the reference layout, per cycle the preparation extracts 17 fields of the decoded row at `2 w` each, reads 4 kind tables at `L + 2 w`, makes 20 range checks at `w` and the one comparison with the row cache at `L + w`, and stores 18 bytes at `w`: `81 w + 5 L`, 6.05 ns. Per bytecode row it reads and caches one digit, `κ·(L + 2 w)`. The model is 6.2 ns at `2^22` and 6.6 at `2^20`. A pipeline pays it *cold*, when it is the first of its session to ask for the source, and not *warm*, when it finds the source in the session; a proof pays it once.
- `λ`. The unit is one evaluation of the row words, priced at the single-thread figure the line `witness compute` of `crates/jolt-rv64i-arith/benches/witness.rs` records for `WitnessRow::compute`, 32 ns. The evaluation of "Lanes" performs a subset of the operations of `compute`, so the price is an upper bound; the unit table does not price the mispredicted branch of the dispatch on a decode line, and no smaller model is derived from the count.
- `ζ`. The route weights at `x` are evaluated once by the kernel's check and once by the driver, two multiplications per entry each: `4 M·N_R`, 0.55 ms per proof, 0.13 ns per cycle at `2^22` and 0.53 at `2^20`. The construction of the shapes from the `N_R` entries is not counted; with it the setup is estimated at 1 ms. The setups of the outer and of the tail (256 column weights, three weight vectors, `Expr::try_evaluate`) are under a microsecond and are not itemised.

An item's threshold is its model at the unit costs of the kernels spec, times 1.25, to the nearest nanosecond. It moves when a unit it is built from is replaced by a measurement on a quiet machine, a row of the unit table or the `witness compute` line, and in no other way; it does not move to meet a result. The digit reads the cores' passes make through `WitnessSource` are charged to the cores' thresholds: at `2 w` a field and `L + 2 w` a kind they are 4.9 ns per cycle in `fold_pass` and `source_lift` and 1.5 in `g_pass_digits`.

One cost of the layer is outside the bench. The replay of the witness constructor writes the decoded row from values it already holds: 17 word operations per cycle when the rows are built from facts, 22 to 25 when they are read, and a 16-byte store. The budget below keeps 2 ns for it. It is counted and not gated, because the constructor is serial as a whole and this spec does not change that.

Against the budget of the kernels spec, 2,100 ns of single-core work per cycle and 918 ms of wall time at `2^22` cycles on 12 threads: the thresholds of the three cores are 745 ns, and this layer adds the thresholds of `σ` and `λ`, 8 and 40, and the counts of `ζ` and of the decoded rows, 0.13 and 2, for 795 ns, 37.9% of the budget. On 12 threads that is 346.5 ms for the cores, the preparation and the lanes at 9.6 effective cores, 0.6 ms for the serial setup and 8.4 ms for the serial write of the decoded rows: 355 ms of 918. What is left, 1,305 ns per cycle and 563 ms, is for the eight members that have no core, the rest of witness generation, the commitment and the opening proof, and this spec does not divide it.

**Memory.** The tables this layer creates or holds:

| Object | Size | Owner | Created | Dropped |
|---|---|---|---|---|
| Decoded rows | 16 bytes per cycle | `Rv64iWitness`, shared by `Arc` | the replay of the witness constructor | with the last `Arc`, the witness's or a kernel's |
| Lanes and tails | 48 and 1 bytes per cycle | `WitnessLanes`, in an `Arc` that `OuterF2Core` alone holds | the lanes pass, in `prepare` of batch 1 | with the core's reference: at `park_residue` of batch 1, or at the round of kernels change (c) |
| Prepared source | four tables of 64 entries, the widths, three `Arc`s; during the preparation, a row cache of 8 bytes per bytecode row | `SharedSource` | first optimised `prepare` of batch 3a, 3b or 6b | with the session; the row cache at the end of the preparation |
| Selector bytes, chunk bytes | invariant 6 | invariant 6 | the preparation | invariant 6 |
| Scatter plan | the kernels spec's row: 4 bytes per cycle | `SharedSource`, and the cycle group by `Arc` | first optimised `prepare` of batch 3a or 3b | the group's reference at the return of `claims_pass`, the session's at the first optimised `park_residue` of 3b; else with the session |
| Word lifts | the kernels spec's row: 96 bytes per cycle | the cycle group | `source_lift`, first optimised `prepare` of batch 3b | the return of `claims_pass` |
| Recorded challenges | `t` elements per kernel | the kernel | the rounds | with the kernel |

Every other table of `T` or `K` elements of a pipeline is created by a core or a pass, is in the Memory table of the kernels spec, and reaches its core by value. Summed with those, by count of capacities and not by measurement, the bytes a pipeline holds beyond the committed rows and the words (72 bytes per cycle, 288 MiB at `2^22`), at `2^22` cycles and `K = 2^20`:

| Pipeline, instant | Bytes per cycle | MiB | The three largest allocations |
|---|---:|---:|---|
| Outer, when the core has built its tables | 97 + 49 + 16 = 162 | 648 | the core's tables and tail, 388; the lanes, 196; the decoded rows, 64 |
| Routers, at the return of `source_lift` | 208.5 + 8 + 10 = 226.5 | 906 | the word lifts, 384; the five source tables, 320; the decoded rows, 64 |
| Routers, at the fourth selector bind | 252.5 + 10 = 262.5 | 1,050 | the five source tables with their second buffers, 480; the word lifts, 384; the decoded rows, 64, tied with the selector bytes and the dense columns that replace them |
| Tail | 135 + 16 = 151 | 604 | the three tables of the reduction with their buffers, 288; the two dense weights with theirs, 192; the decoded rows, 64 |

The 10 bytes per cycle on the routers' rows are the chunk bytes, written by the preparation in batch 3a and read in batch 6b; the 8 on the first are the selector bytes, which exist from the preparation and not from the construction of the cycle core. The tail's figure is an envelope: the chunk bytes and the dense columns that replace them can overlap while the old buffers are dropped in the background, up to 40 MiB more. `fold_pass` has scratch of its own, about 11.5 MiB per worker, that is released before the cycle rounds. The recorder is what checks these rows. No `jolt-eval` objective moves.

**The recorder.** The counting allocator of the kernels crate's benches keeps totals and cannot list an allocation. `benches/support/` of this crate carries a recorder inside the global allocator of the bench: a fixed array of 4,096 entries of a size and a phase, one for every allocation or reallocation of `T` bytes or more, and a count of the entries that did not fit, which fails the run when it is not zero. It allocates nothing. The phase is an atomic the runner sets. It sees the Rust allocator only: the purge of `crates/jolt-kernels/src/mem.rs`, which the lazy family of a chunk core calls at `log_t ≥ 22`, obtains its blocks from the C allocator on macOS and is not recorded. The command is `cargo bench -p jolt-rv64i-prover --features test-utils --bench adapters -- inventory`, documented at the head of the bench; it prints the inventory of each pipeline beside the rows of the Memory tables it matches.

## Design

### Architecture

#### The decoded row

```rust
#[repr(C)]
pub struct DecodedCycle { pub inc: u64, pub digits: u64 }   // 16 bytes
```

`digits` packs, from bit 0: the bytecode index (`b` bits), the RAM index (`a` bits), `Pos` (6 bits, the low digit first), `KeysDiffer`, `ShouldBranch` and `JalrLowBit` (one bit each) and the index of the fetched row's variant (6 bits, `Variant::COUNT = 58`). They fit: `81 + n(b) + n(a) ≤ 256`, which `Layout::new` checks, bounds `b + a` by 49, reached at `(3, 46)`, and `49 + 15 = 64`. `DigitFields::new(&Layout)` owns the offsets and is total: the sum is bounded by `Layout::new`, and the constructor restates the bound in a `debug_assert` that names it. It gives a field `(shift, bits)` for each chunk of each index, each `Pos` digit, the whole of `Pos`, each flag and the variant. A chunk's offset inside its index is the sum of the widths of the chunks below it, which is how `Layout::bytecode_index` reads them. The replay packs through `DigitFields` and the source unpacks through it.

The row is a function of the committed row and of the fetched bytecode row: the indices and `Pos` are `Layout::{bytecode_index, ram_index, pos}`, the flags are the three columns, `inc` is `Layout::inc`. Every field is a value. A chunk digit of zero is the digit 0, which the committed row stores as no indicator, so no digit of a chunk or of `Pos` is absent.

`Rv64iWitness::replay` writes the table. `from_bits`, `from_facts` and `synthetic` all pass through it, and it is the one walk of the cycles a witness constructor makes. Per cycle it already holds the bytecode index, the fetched row and its variant, the RAM index, the committed row and `inc`, and it already rejects a row without a variant (`InvalidBytecode`) and, when it reads committed rows, a chunk of an index or of `Pos` with two indicators (`MultipleIndicators`). What it packs is the cycle's `SourceParts` (`inc`, the RAM index, `Pos` and the three flags), and no digit is decoded for the packing alone: when the rows are built from facts, `BitsBuilder::bits_row_with_parts` returns the parts with the row it writes, and when they are read, `SourceParts::from_checked_bits` returns them with the bytecode index from the indicator check, through `Chunk::checked_digit`. The replay gains one pack and one increment: `variant_cycles[v]` counts the cycles of variant index `v`. The witness gains two fields, `decoded: Arc<[DecodedCycle]>`, allocated once at `T` and filled in place like `words`, and `variant_cycles: [u64; 64]`. `prove_inner` checks the length of `decoded` with those of `bits` and `words`.

#### Columns and words

`WitnessColumns::new(&Layout)` owns this table. With `d = d_b + d_a`:

| Digit column | Index | Bits | `by_row` | Digit at cycle `j` | Committed columns |
|---|---|---|---|---|---|
| bytecode chunk `c` | `c` | `bytecode_ra()[c].bits()` | no | its field; always `Some` | `Indicators` at `bytecode_ra()[c].start()` |
| RAM chunk `c` | `d_b + c` | `ram_ra()[c].bits()` | no | its field; always `Some` | `Indicators` at `ram_ra()[c].start()` |
| `Pos` digit `i < 2` | `d + i` | 3 | no | its field; always `Some` | `Indicators` at `pos_ra()[i].start()` |
| `Variant` | `d + 2` | 6 | yes | its field; always `Some`. `row_digit` is the index of `BytecodeRow::variant` | none |
| `ShiftKind`, `AccessKind`, `KeyKind` | `d + 3`, `d + 4`, `d + 5` | 3, 4, 3 | no | a table of 64 entries read at the variant field: the index of `Variant::shift`, of `Variant::access().and_then(\|a\| a.kind)`, of `Variant::key_kind`, or `None` | none |
| `Branch` | `d + 6` | 0 | no | a presence column, read from the same kind of table: `Some(0)` when `Variant::branch` is `Some`, a conditional branch, and `None` otherwise | none |
| `KeysDiffer`, `ShouldBranch`, `JalrLowBit` | `d + 7`, `d + 8`, `d + 9` | 0 | no | `Some(0)` when the flag bit is set | one `Flags` range at `keys_differ()` |

The kind indices are those `RouteTensors::new` writes into a selector. An access without a kind has no `AccessKind` digit. Only `Variant` is `by_row`: it is the one factor of the one shape whose `Bytecode` slots `fold_pass` buckets per row, which it does only when every factor of the shape is `by_row`, and every other `by_row` column would add a row cache and a comparison per cycle to the preparation for nothing. `Store` has no reader and no column.

| Word | Index | Read from |
|---|---|---|
| trace: `Rs1Value`, `Rs2Value`, `RdPreValue`, `RamReadValue`, `NextPC` | 0–4 | `words[j]` |
| trace: `Inc` | 5 | `decoded[j].inc` |
| bytecode: `Imm`, `FallThroughPC`, `PCPlusImm`, `PC` | 0–3 | `bytecode.rows()[k]` |

`WitnessColumns` gives each column and word of these tables by a named accessor, and nothing else numbers one. It has four readers: `WitnessSource`; `router_shapes`, whose one `match` sends a `BankWord` to a `WordSlot` and a `Factor` to a digit column; the lists of chunk columns the preparation is asked for; and `WitnessColumns::column_map()`, the `ColumnMap` list of `g_pass_digits`: `Word { start: 0, trace_word: 5 }` and the `Indicators` and `Flags` ranges of the last column. `WitnessColumns::committed(y)` is the same correspondence read from the committed side, the pair (digit column, value) whose indicator column `y` is; a flag has the value 0.

#### Source and shared state

`WitnessSource` implements `CycleSource` over `Arc` clones of `words`, `decoded` and `bytecode`, a `DigitFields`, a `WitnessColumns` and the four kind tables. `cycles()` is `decoded.len()` and `bytecode_rows()` is `bytecode.rows().len()`. It holds nothing of size `T` of its own.

**The preparation.** The first optimised `prepare` among batches 3a, 3b and 6b makes the one checked walk of the source, kernels change (a): the validation of `ValidatedTrace::new`, parallel over the chunks of `CycleChunks::new(t, 0)`, which in the same walk writes the *groups* it is asked for. A group is a list of digit columns and an owned buffer of one byte per column and cycle, the bytes of a cycle adjacent. A *present* group holds the digit, and a cycle on which one of its columns has no digit is an error of the preparation; an *optional* group holds the digit plus one, and zero for no digit. Three groups are asked for: the `d_b` bytecode chunks and the `d_a` RAM chunks, present, each only when it has at most seven columns; and, when the caller is an adapter of batch 3a or 3b, the eight distinct factor columns of the five shapes, optional, in the order in which `RoutersCycleCore` holds them, which the kernels crate lists for a set of shapes. A tail adapter that is the first to ask requests the two chunk groups and no selectors: batches 3a and 3b are behind it.

The walk reads each decoded row once and makes the reads the validation makes today; the groups add 18 byte stores per cycle at the reference layout, in `σ`. They replace three gathers, by the two chunk constructors and by the cycle core, each of which reads the 16-byte row again: 34 bytes of traffic per cycle, 16 read and 18 written, against 82, and by count 10 field reads for the chunks and 3 fields, 4 kinds and a flag for the selectors, 3.4 ns per cycle, that are not made twice. Nothing is added to what a pass reads, so the routers get their selector bytes at the cost of 8 stores per cycle. `ScatterPlan::new` keeps its two reads of the bytecode index per cycle: it is parallel already, `2 Bk + 6 w` by count, and the kernels spec assigns it to the construction of the routers.

The chunk bytes cannot come from `g_pass_digits`, which reads the same digits in batch 6b: the generated driver prepares the two chunk members before the reduction, whose weights that pass needs.

`SharedSource` is the value that `ProofSession::state_or_insert_with` keeps for what kernels of different batches share. It is inserted empty and filled by its own methods, because the preparation and `ScatterPlan::new` return a `Result` and the initialiser of the session cannot. The cycle group is a second state type, in `routers.rs`.

| Item | Built by | Used by | Released | When the other tier holds the slot |
|---|---|---|---|---|
| the prepared source, an `Arc` | the preparation | every later optimised `prepare` of batches 3a, 3b, 6b | with the session | each adapter asks and builds on a miss; a reference kernel never asks |
| the selector bytes | the preparation, for a router adapter | taken by value by the first optimised `prepare` of batch 3b, for `RoutersCycleCore` | by the core, at its fourth bind | all of 3b reference: they stay to the end of `prove` |
| the bytes of each chunk group | the preparation | taken by value by the optimised `prepare` of that chunk product | by its core, at its fourth bind | a reference chunk product: its group stays to the end of `prove` |
| `Arc<ScatterPlan<WitnessSource>>` | the first optimised `prepare` among batches 3a, 3b | `fold_pass` in 3a, `claims_pass` in 3b | "Routers", the third event | 3a reference: 3b builds it. All of 3b reference: it stays to the end of `prove` |
| the cycle group | the first optimised `prepare` of batch 3b | the other optimised `prepare`s of 3b, each taking its member | "Routers", the third event | members of reference slots are never driven and drop with the group |

A group is taken out of `SharedSource` by the adapter that moves it into a core; an adapter that finds its group gone, or never written, returns `InvalidGeometry`. This is not a hand-over by `ProofSession::park` and `take`, which needs a producer and a taker that are both present: in a mixed registry either may be a reference kernel, and a state that is built on a miss needs neither. Three tables do not cross a batch at all. The fold tables are moved into `RouterShortCore::new` inside the `prepare` of batch 3a, and `ra_fold` is dropped there. The lifts are built from `x`, which exists only in batch 3b. The scatter that `claims_pass` returns has no taker in this spec and is dropped (Execution, the optional item).

Session state and kernels are `MaybeAllocative`. Under the feature `allocative` the state types and the kernels implement `Allocative` with the fields of kernels-crate types skipped, so a heap snapshot attributes none of a core's tables.

#### Lanes

`Rv64iWitness::cycles()` returns `WitnessCycles`, a borrowed view of the witness with its `DigitFields` built once for a walk; no function takes a `DigitFields` beside a witness. `WitnessCycles::parts(cycle)` returns what a cycle decodes to without its committed row, as `CycleParts`: the fetched row, its variant, `CycleWords::base_words` and the `SourceParts` unpacked from `decoded[cycle]`, or a typed error for an absent cycle or an invalid fetch. `WitnessCycles::row(cycle) -> Result<WitnessRow, Rv64iProverError>` is `parts` and then `WitnessRow::compute` on the committed row. It is the private `row` of `reference/spartan.rs` moved to the witness: the composition has one owner, and it takes the bytecode index and `inc` from `decoded[cycle]` and does not decode them from the committed row again. The reference tier calls `row`, and the pass of `SpartanOuterF128` will; the lanes pass calls `parts`.

The lanes read less than that row: rows 0 to 129 read the six words `CarryLeft`, `CarryRight`, `CarryStep`, `AndLeft`, `AndRight`, `AndOut`, the bits `LeftKeyBit`, `RightKeyBit`, `LessThan` and the committed bit `KeysDiffer`. The arithmetisation gains one evaluator of those nine, and this is its contract; its owner decides the form.

1. `Words::compute` takes those words from it. The dispatch on the rails of the decode line, the carry formulas and the key-bit step each have one body, and no rail is written twice: the rails stay data of `decode.rs`.
2. It reads the sixteen source words built from parts, the fetched row, the base words, `inc`, the RAM index, `Pos` and the three flags, by a constructor of `Sources` that `Sources::new` also calls once it has decoded those parts from the committed row. `RamAddress` keeps one definition.
3. `RowSystem` gives the values of rows 0 to 129 on its result, the two word triples and `(A, B, C)` of rows 128 and 129, from the row definitions it holds. The adapter writes no row; it packs the six bits into the tail byte of `LaneSource`.
4. It evaluates no `KeyDiffAbove`, no positional form of a shift, a load or a store, no `BranchOutput`, no expected word and no residual, and builds no `WitnessRow`.

It takes `NextPC` among the base words. The sum of the adder rail on the two `JALR` lines is the form `JALR_SUM` of `decode.rs`, `NextPC + JalrLowBit`, so the three carry words depend on `NextPC` on those cycles; what the lanes do not read is the residual of `NextPC`.

*The count.* Per cycle `Words::compute` evaluates six forms, the three rails, `rd_expected`, `next_pc_expected` and `control`, and one more per hot position on a shift line, per hot offset on an access line and on a taken branch: 6.46 on the mix of the bench. The evaluator evaluates the three rails. It also leaves out the decoding of the RAM index and of `Pos` from the committed row, `d_a + 2` chunk reads, and the assembly of sixteen words. The pass reads 56 bytes per cycle of `words` and `decoded` and the fetched bytecode row, not the committed row, and writes 49. Its threshold is `λ` of Performance, which is the price of the full `compute` and holds whichever evaluation the pass calls.

The other course is to call `WitnessCycles::row` in the pass and take the lanes from the full row, at the same `λ`. It is taken only if the factoring slows `compute` or needs a rail twice. On reading `words.rs` and `decode.rs` neither holds: `compute` performs the operations it performs today, in the same order, the first of them inside the evaluator. That it is not slower is an inference from that structure and not a measurement. The PR that makes the factoring reports the `witness compute` line before and after; if it moves beyond the spread of the bench, the factoring is dropped and the pass calls `row`.

`WitnessLanes::new(&Rv64iWitness)` builds `RowSystem::new(&witness.layout)` and the view `witness.cycles()` once and makes one parallel pass over the chunks of `CycleChunks::new(t, 0)`, in `prepare` of batch 1. Both buffers are allocated zeroed, so that the pass is the first to write them and no thread touches them before it. Per cycle it calls `WitnessLanes::cycle(&view, &rows, j)`, which returns the two triples and the tail byte of cycle `j` from `parts(j)`, and stores them. It implements `LaneSource`. `cycle` is the function a second pass over the rows can call. If the pass of `SpartanOuterF128`, which needs the full row, is fused with this one, the fused pass evaluates `WitnessCycles::row` once per cycle and takes the lanes from it with `LaneRows::values`: the lanes move into that pass, `WitnessLanes::new` goes, and the evaluator keeps `Words::compute` as its one caller.

#### Outer

`prepare` builds the lanes and `OuterF2Core::new(Arc::new(lanes), relation.tau(), OuterF2Options::default())`, and keeps no reference to the lanes. `tau()` is the 8 row coordinates and then the `t` cycle coordinates, the order of `τ`. `output_claims` returns `az`, `bz`, `cz` from `final_values()`. The adapter forms no derived term: `EqTau` is inside the core, from `tau` as passed.

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

`router_shapes(&WitnessColumns, &Layout, Option<&RouteTensors>)` returns the five `RouterShape`s in the order of `ROUTERS`, each from one `RouterShapeRequest`. Word slot `n` is the `Trace` or `Bytecode` slot of `words[n]`, or `Bits([One])`. The committed range follows the words as `Bits` slots of 64 entries, entry `y − g` being `Indicator { column, value }` of `WitnessColumns::committed(y)`, and `One` after the last; `Zero` slots pad the bank to a power of two. For the `Variant` shape, whose nine words fill slots 0 to 8, the `Bits` slots are therefore 9 to `8 + ⌈(used_columns() − g + 1)/64⌉`: at the reference layout `g = 139` and 231 columns are used, 93 entries in slots 9 and 10, and slots 11 to 15 are `Zero`; at `(3, 46)`, `g = 71` and 256 are used, 186 entries in slots 9 to 11, and slots 12 to 15 are `Zero`. The bit and word variables take `source_slots(router)`, the first six and the rest; the factors take `selector_slots(router)` in order, each as many as its column has bits. `S = 17`, `log_outputs = 10`, and `route` is `(column, source, selector)` of `RouteTensors::entries(router)`, or empty without tensors.

**`RouterShort`.** `prepare` takes the prepared source and the plan from `SharedSource`, makes the shapes with `relation.routes()`, and runs `fold_pass(trace, shapes, relation.r_1(), plan, layout, &[])`. `layout` is `FoldLayout::new(trace, shapes, values)` with the default values of the kernels crate: its function that picks the selector values to bucket by byte from the counts of a shape and a limit, called with its constant for the default limit, on the `Variant` shape, and no values for the other four. A selector value of that shape is a variant index, so the counts are `variant_cycles` and `selector_counts` does not walk the trace. `ra_fold` is dropped and the fold tables are moved into `RouterShortCore::new(shapes, relation.w(), folds)`. `output_claims` maps the first value of each pair of `final_values()`, `Fold_ρ(x|_ρ)`, to `variant`, `shift`, `memory`, `compare`, `branch`, and `validate_derived_tables` compares the second as invariant 4 states, returning `DerivedTableDrift` on a difference.

**The five `RouterCycle` members.** The first optimised `prepare` of batch 3b builds the cycle group: the shapes without tensors, `source_lift(trace, shapes, relation.x())`, the cycle core from the source tables of its output and the selector bytes taken from `SharedSource`, by the consuming constructor of kernels change (b), its five `members()`, the lifts and the plan. Every optimised `prepare` takes the member of its router and returns a kernel that holds the member and an `Arc` of the group. A relation whose `r_1()` or `x()` is not the group's, or a member taken twice, is `InvalidGeometry`. A kernel forwards `prove_round` and `finish_rounds` to its member and records the challenges, which are `r_3`. A core driven on a subset of its members gives, for each member that is driven, the messages and final values it gives when all five are driven; mixed registries rest on that property of the kernels crate.

Three things end at three events, in the order the driver reaches them.

1. The first `finish_rounds` a member receives drops, inside the core, the five source tables with their second buffers and the partial sums. The later calls bind nothing.
2. The first `output_claims` among the optimised members runs `claims_pass(trace, lifts, words, plan, r_3)` for trace words 0–4, keeps the values in the group, and at the return drops the lifts and the group's `Arc` of the plan.
3. The first `park_residue` among the optimised members empties the session's cycle-group state and calls `SharedSource::release_plan`, which drops the session's `Arc` of the plan: the session's references are removed by those two calls and not by a drop. The plan's memory goes there. The group's goes with the last kernel that holds it, and `park_residue` then drops the kernel as its default does, on a background thread.

Outputs are mapped by name, through `Factor` and `BankWord`, and never by position. `final_values()` of a member gives its source value and its factors' values in factor order, which for `Shift`, `Memory` and `Compare` is the `Pos` digits and then the kind; the typed outputs put the kind first. `claims_pass` returns the trace words in the order asked and the bytecode words in source order. The typed outputs, in their declared order:

| Member | Typed outputs | Words, from `claims_pass` at `(r_bit, r_3)` | Factors, from the member, `Sel_f(r_3)` |
|---|---|---|---|
| `Variant` | `rs1_value`, `rs2_value`, `rd_pre_value`, `imm`, `fall_through_pc`, `pc_plus_imm`, `pc`, `next_pc`, `variant_bits`, `variant` | the first eight | `variant` |
| `Shift` | `rs1_value`, `shift_kind`, `pos_ra_0`, `pos_ra_1` | `rs1_value` | the other three |
| `Memory` | `ram_read_value`, `rs2_value`, `access_kind`, `pos_ra_0` | the first two | the other two |
| `Compare` | `rs1_value`, `rs2_value`, `imm`, `key_kind`, `pos_ra_0`, `pos_ra_1` | the first three | the other three |
| `Branch` | `fall_through_pc`, `pc_plus_imm`, `branch`, `should_branch` | the first two | the other two |

An alias cell and its source are one value of the group, or the value of one shared factor column, so they are equal; against a reference kernel they are equal because both are the extension of one table at one point.

`variant_bits` is the one output that is neither a word nor a factor. With `ω_n = eq(x|word, n)` over the word slots of the `Variant` shape and `c_V = eq(x|src, s)` for the source index `s` of its `One` entry,

`variant_bits = Source_V(x|src, r_3) + Σ_{n<8} ω_n·word_n + c_V`,

where `word_n` is the claim of word slot `n`: the eight named word claims of the `Variant` row above, which are in slot order. `prepare` computes the `ω_n` and `c_V` with `eq_table` from the shape, and `validate_derived_tables` of the `Variant` kernel compares them as invariant 4 states. The other four kernels hold no derived term.

#### Tail and the vector `C`

**Chunk products.** `BytecodeReadCycle` takes the group of the `d_b` bytecode chunks from `SharedSource` with the points `relation.chunks()[c].1` and, for `[h_router, h_read, h_val, h_entry, h_next] = relation.folds()`, the weight `Dense(combined_weight(t, terms))` with the terms `Eq(h_router, r_3)`, `Eq(h_read, r_4)`, `Eq(h_val, r_5)`, `Eq(h_entry, 0^t)`, `Next(h_next, r_3)`. `RamRaProduct` takes the group of the `d_a` RAM chunks with its `chunks()` and the terms `Eq(c.read, r_4)`, `Eq(c.val, r_5)` for its challenges `c`. Each moves its group into the consuming constructor of `ChunkProductCore`, kernels change (b), which reads no digit and copies no byte. `output_claims` returns `chunks[c] = Ra_c(r_6)`, the second part of `final_values()`, in column order, which is the order of the group and of `chunks()`. `validate_derived_tables` compares `W(r_6)` with the same combination of the relation's derived terms: `Weight(·)` under `folds()`, and `EqRead`, `EqVal` under the challenges. Both weights are `Dense`, the default of the kernels spec.

**`BitsReduction`.** With `c` the challenges and `relation.weights()` the six sparse supports in the order direct, variant, `Pos` low, `Pos` high, `ShouldBranch`, `Inc`, each expanded to 256 entries:

`L_1 = c.direct_columns·weights[0]`, `L_3 = c.variant_bits·weights[1] + c.pos_ra_0·weights[2] + c.pos_ra_1·weights[3] + c.should_branch·weights[4]`, `L_5 = c.inc·weights[5]`.

`g_pass_digits(trace, columns.column_map(), [L_1, L_3, L_5])` gives three tables, moved into `ReductionCore::new` with three legs of coefficient 1, at `r_1`, `r_3` and `r_5`. With `[z_0, z_1] = relation.pos_zero()` their claims are

`c.direct_columns·direct_columns`, `c.variant_bits·variant_bits + c.pos_ra_0·(pos_ra_0 + z_0) + c.pos_ra_1·(pos_ra_1 + z_1) + c.should_branch·should_branch`, `c.inc·inc`.

They regroup the member's input expression by cycle point. In round 0 the kernel checks that they sum to the claim it is handed and returns `SumcheckError::RoundCheckFailed` otherwise; that check sees the sum and nothing of the split (invariant 4).

**The vector `C`.** `C` is the typed output `columns` of `BitsReduction`. The `curate` of stage 6b moves `claims.bits_reduction.columns` to the wire and `stage6b::prove` returns it as `BitsColumns`. It needs no slot of its own. The kernel keeps a clone of the `Arc` of `bits` and records its challenges, and `output_claims` returns `column_pass(&bits, r_6)`. The `final_values()` of the core are not outputs.

#### Registry

`Rv64iBackend::optimized()` is `reference()` with the ten slots of the adapter table replaced. There is no constructor per stage: every field of `Rv64iBackend` and of the `Stage*Kernels` is public, and a mixed registry is one of the two constructors with fields reassigned, which is how the tests build theirs.

### Alternatives Considered

- **The stage functions with stubbed slots, for the bench.** Performance, "What it drives".
- **The chunk cores read the decoded rows in place.** It removes the 40 MiB of chunk bytes. The lazy family reads column by column, so a 16-byte row gives no single load per cycle: five passes read 80 bytes per cycle and core from the rows against 25 from the bytes, and 30 with the bytes' one write.
- **The chunk bytes gathered in batch 6b, by a pass of the tail.** No bytes are held from batch 3a to batch 6b, which is 10 bytes per cycle off the largest row of the Memory table, for a second walk of ten digits per cycle, 1 ns by count, over rows the preparation has read. The preparation's walk is chosen; the 40 MiB are its price.
- **The cycle core gathers its own selector bytes.** It keeps the preparation ignorant of the routers, for a second read of the decoded row and of the four kind tables per cycle, 2.4 ns by count. It is what the adapters do if the consuming constructor of the cycle core does not land: the selector group is then not asked for, and nothing else changes.
- **The decoded rows built by the adapters, in a parallel pass at first use.** It leaves the witness as it is and is a second walk of the committed rows, where the replay holds every field of the row as it goes.
- **The lanes emitted by the replay.** The replay is serial, and a serial nanosecond per cycle is 4.2 ms of wall time at `2^22`. It would carry the most expensive computation per cycle of the witness in its one loop, patch each row one cycle late for `NextPC`, and keep 49 bytes per cycle alive from the witness to the end of the proof for a table that batch 1 alone reads.
- **A slot for `C`.** `C` is a typed output of a member that has a slot. A second slot would be a second producer of one value and a registry field that is not a sum-check member.
- **`park` and `take` between batches 3a and 3b.** They fail in a mixed registry whichever side is the reference kernel, and the tables the kernels spec names for them do not cross the batch.
- **One `prepare` for the five cycle members.** The registry is a struct of one `PrepareKernel` per member, and a mixed registry replaces one field. A group in the session keeps that and builds the shared core once.
- **The bank restated in `router_shapes`.** A wrong bank is caught by "Batch outputs", and it would be a second statement, in production code, of what `routes.rs` fixes.
- **`by_row` for the four kind columns.** They are functions of the row. No pass of the five shapes uses the declaration for them, and the preparation pays for it.
- **A threshold as a percentage over a core's.** A percentage of 299 ns is a budget no count stands behind, and it hides which pass spends it. The three items are counted and benched apart.

## Documentation

No change to the Jolt book: nothing here is reachable from the SDK. Rustdoc on `DecodedCycle` and `DigitFields` states the packing and its bound; on `WitnessColumns`, that it is the one numbering; on `SharedSource`, the groups, who takes each and when an untaken one is dropped; on each `…Prepare`, its core, what it shares and what it pins in `validate_derived_tables`. The sentence of invariant 9 goes, in the same terms, into the rustdoc of `SpartanOuterF2Prepare` and into the module documentation of `reference/spartan.rs`: the rows are required of the witness and not checked; a violated row is detected algebraically, with the soundness error of the sum-check, at the driver's comparison of the final claim on the optimised tier and at a round check on the reference tier; neither tier rejects it deterministically.

## Execution

**Changes to `jolt-rv64i-kernels`, in the order they land, before the items below start.**

- **(a) The preparation.** The checks of `ValidatedTrace::new`, in parallel: the digits of the `by_row` columns over the bytecode rows, then the cycles over the chunks of `CycleChunks::new(t, 0)`. The error is the one the serial scan returns: that of the first offending row if a row offends, otherwise of the first offending cycle, and within a row or a cycle that of the first check in the order of the scan. It does not depend on the thread count. The same walk writes the groups it is asked for ("Source and shared state"), a present group of columns of at most 8 bits and an optional group of columns of at most 7; a missing digit of a present group is an error with its column and cycle, ordered with the others. A group's type has no other constructor, so a holder of one holds validated digits.
- **(b) The consuming constructors.** `ChunkProductCore::new` takes a present group by value, with the points and the weight. It checks one to seven columns (`Columns`), one point per column (`PointCount`), each as long as its column is wide (`PointLength`), and the weight against the cycles of the group (`WeightLength`, `TermPoint`), and returns those `ChunkProductError`s; the width bound of 8 is the preparation's, and the group's type carries it. The core owns the buffer from then on, reads no digit before its first round, copies nothing, and releases the buffer at the fourth bind; it has no `MissingDigit` case, which the group's type excludes. `RoutersCycleCore::new` is the same over an optional group: it checks that the group's columns are its distinct factor columns in its order, their widths and the cycles, with a typed `RouterError`, and `RoutersCycleCore::columns` lists those columns for a set of shapes. Neither core has a constructor over a borrowed source.
- **(c) `OuterF2Core` releases its source** at the round after which it reads only tables of its own, having built there what the cycle rounds read. The lanes, 49 bytes per cycle and 196 MiB at `2^22`, then leave the live set for the remaining rounds. This is not a reduction of the peak: the lanes and the core's 97 bytes per cycle coexist while those tables are built.

*Optional, and not recommended now: keeping the scatter of `r_3`.* `claims_pass` already returns it, so the kernels crate has nothing to change; keeping it is a `park` by the cycle adapters for batch 6a. It saves `1 M + 1 sct`, 3.2 ns per cycle, on one of the five histograms of that batch, and costs 16 bytes per bytecode row, 16 MiB at `K = 2^20`, held through batches 4 and 5, with a session entry whose only taker is a kernel no spec has yet. The adapter of `BytecodeReadAddress` adds the `park` with its `take`.

Three things the adapters use are named by the kernels crate and referred to here by role: the function of `FoldLayout` that returns the default selector values from the counts of a shape and a limit, with its constant for the default limit; the typed `RouterError` of each rejected input, which `prepare` displays; and `SourceLiftOutput`, of which the adapters read the source tables and the lifts.

| Item | Files it owns | Criteria | Starts after |
|---|---|---|---|
| **1. Sources** | prover `Cargo.toml` (the dependencies on `jolt-rv64i-kernels` and `rayon`; every `[[test]]` and `[[bench]]` of this spec, with one-line files so that the manifest resolves); `src/plane.rs`, `src/prover.rs`, `src/reference/spartan.rs`; `src/optimized/mod.rs`, every declaration of the module; `src/optimized/source.rs`; `tests/optimized_sources.rs`; arithmetisation `src/{cycle, layout}.rs` and `SourceParts` in `src/decode.rs` (the parts a row is built from and the checked chunk reader), after item 2; `tests/support/mod.rs` (the separating fixture; the reference proof kept batch by batch); `specs/rv64i-binary-protocol.md` §12 (`decoded`, `variant_cycles`) | Decoded rows; Source against the committed table | (a) |
| **2. Outer** | `src/optimized/outer.rs`; `tests/optimized_outer.rs`; arithmetisation `src/{words, decode, rows}.rs` and the sentence of `specs/rv64i-binary-arithmetisation.md` for the evaluator | Lanes; Batch outputs, batch 1; A witness whose rows do not hold | the first commit of 1 |
| **3. Routers** | verifier `src/public/routes.rs`; `src/optimized/routers.rs`; `tests/optimized_routers.rs` | Shapes; Batch outputs, 3a and 3b | the first commit of 1; (b) |
| **4. Tail** | `src/optimized/tail.rs`; `tests/optimized_tail.rs` | Batch outputs, 6b | the first commit of 1; (b) |
| **5. Registry and bench** | `src/backend.rs`; `tests/optimized_registry.rs`; `benches/adapters.rs`, `benches/support/` | Mixed registries; End-to-end corpus; Rejected inputs; Bench; Performance; Trace-sized allocations; Gates | the first commit of 1 for the bench; 2, 3 and 4 for the tests |

The first commit of item 1 is `plane.rs`, `source.rs`, `optimized/mod.rs` and `tests/support/mod.rs` complete, with `Cargo.toml` and the one-line files; its test follows. No two open items then share a file: items 2 and then 1 are the only ones in the arithmetisation, in that order, item 3 the only one in the verifier crate, and no item but 1 edits `optimized/mod.rs` or `tests/support/mod.rs`. Item 5 writes the generated program, the bench runner, the recorder and the registry test over slot masks against `reference()` while the others are open, and switches slots as they land.

## References

- `specs/rv64i-binary-protocol.md`: §3 points, §5 batches and names, §8.4–8.9 routers, §8.17 and §10 the reduction and `C`, §12 the witness, the registries and the reference tier, the "Proof vector" criterion.
- `specs/rv64i-binary-prover-kernels.md`: Goal and its surface table, invariants 3 and 6, Performance (unit costs, Requirements, the budget, Memory), Design "Routers" and "Fit with `jolt-kernels` and the protocol".
- `specs/rv64i-binary-arithmetisation.md`: `Layout`, `RowSystem`, `WitnessRow`.
- `specs/binary-protocol-family-seams.md`, `specs/binary-kernel-primitives.md`: the plane parameter of `PrepareKernel`.
- `crates/jolt-kernels/src/{kernel.rs, backend.rs, mem.rs}`; `crates/jolt-prover/src/driver.rs`, the order of `prepare`, rounds, `validate_derived_tables`, `output_claims`, `park_residue` and the final-claim comparison.
- `crates/jolt-rv64i-arith/src/{words.rs, decode.rs, rows.rs, variant.rs}`, `crates/jolt-rv64i-arith/benches/witness.rs`.
- `crates/jolt-rv64i-prover/src/{plane.rs, backend.rs, prover.rs, stages/, reference/}`, `crates/jolt-rv64i-prover/tests/{support/, e2e.rs, routers.rs, wire.rs}`; `crates/jolt-rv64i-verifier/src/{public/routes.rs, claims/router_cycle.rs, stages/}`; `crates/jolt-rv64i-kernels/src/{source.rs, par.rs, router/, packed/scatter.rs, chunk_product.rs, reduction/, column_pass.rs, outer_f2/}`, `crates/jolt-rv64i-kernels/benches/support/`.
- `CLAUDE.md`, "Tests"; `.claude/skills/test-policy/SKILL.md`.
