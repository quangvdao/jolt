# Spec: Memory, Packed-Row and Pair-Sum Prover Kernels for RV64I over the Binary Field

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-10                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

An experiment in RV64I hash-based Jolt over binary fields has, with `specs/rv64i-binary-prover-kernels.md` (below, the kernels spec), performance-tier cores for the bitwise outer sum-check, the routers, the chunk products and the reduction of committed claims. Eight members of `specs/rv64i-binary-protocol.md` still have only a dense reference kernel: `SpartanOuterF128`, `SpartanInner`, the three members of batch 4, the two of batch 5 and `BytecodeReadAddress`. The reference tables of batch 4 have `2^(a + log_t)` entries, so no proof exists beyond about `2^10` cycles, whatever the other cores do. This spec adds the cores of those eight to `jolt-rv64i-kernels`, as seven modules on the packed machinery, the source traits, the runner, the oracle and the unit table of the kernels spec, which it cites and does not restate. Per cycle the memory cores read the bytes of three groups that the one checked walk of the source writes, and one lifted word; no core of this spec reads a digit from the source a second time, no table has `2^a·T` entries, and the work proportional to the size of the RAM is one lift and one accumulation per RAM word in each of two folds. No shared crate changes. The read-write, value-evaluation and output kernels of `jolt-kernels` sample at integers and carry increments as integer differences; what serves as it is, is `SplitLt`, `LazyFoldedRa` and the split equality.

## Intent

### Goal

Build the cores of `SpartanOuterF128`, `SpartanInner`, `RegistersReadChecking`, `RamReadChecking`, `RamOutputCheck`, the two value evaluations and `BytecodeReadAddress`, with the views of the memory trace they share, each emitting exactly the round polynomials of its definition within a stated number of nanoseconds per cycle and with no table of `2^a·T` entries.

**Notation.** That of the kernels spec: `T = 2^log_t` cycles indexed by `j`, points low variable first, `eq(w, x) = Π_i (1 + w_i + x_i)`, every core binds low to high. In addition: `a` is the number of RAM address bits and `K = 2^a` the number of RAM words; `x = F128::from_raw(2)`; `r_bit` is a point of 6 coordinates and `lift(w) = Σ_{i<64} w[i]·eq(r_bit, i)`, a `WordLift` over the weights `eq_table(r_bit, None)`. Per cycle the memory trace gives a RAM index `addr_j < K`, a store bit `store_j` and three registers `rs1_j`, `rs2_j`, `rd_j` below 32; `U[j] = lift(Inc[j])` is the lift of the word `Inc`. With these,

`RegVal(k, j) = Σ_{j' < j, rd_j' = k, store_j' = 0} U[j']` and `RamVal(k, j) = lift(initial[k]) + Σ_{j' < j, addr_j' = k, store_j' = 1} U[j']`,

which are `RegistersVal(k, r_bit, j)` and `RamVal(k, r_bit, j)` of `specs/rv64i-binary-protocol.md` §3 under the witness contract of its §8.1: in characteristic 2, `(1 + Store)·Inc` is `Inc` on a cycle that is not a store and zero on a store. `ρ` is the number of visited bytecode rows per cycle, as in the kernels spec, and `κ = K/T` the number of RAM words per cycle. A core's challenge point is written `(z, r')`: `z` for its row or address rounds and `r'` for its cycle rounds.

**Memory views.** The cores read the witness through the two source traits of the kernels spec. No trait is added and `CycleSource` gains no method: every fact these cores need per cycle is a digit column or a trace word, and a second trait would be a second way to state a column. The digits reach the cores as byte groups of `ValidatedTrace::prepare`, the one checked walk of the source: a *present* group holds one byte per column and cycle for columns of at most 8 bits whose digit is always `Some`, an *optional* group the digit plus one, or zero, for columns of at most 7 bits. A core takes a group by value and reads no digit from the source. Module `memory` holds the views.

`MemoryTrace::new(ram, registers, store)` takes three groups of one prepared trace: `ram`, a present group of the digit columns of the RAM index, low chunk first, of `a` bits in all with `1 ≤ a ≤ 32`; `registers`, a present group of three columns of 5 bits, `rs1`, `rs2`, `rd` in that order; `store`, an optional group of one column of zero bits, set where its byte is non-zero. It makes no pass over the cycles: it checks the counts and the widths of the columns and that the groups have one number of cycles, and holds them, `d_a + 4` bytes per cycle for `d_a` RAM chunks. That a RAM digit or a register is present on every cycle is the walk's check, which returns `SourceError::MissingDigit`: a cycle without an access carries index 0 and an absent operand register 0, as the witness contract has them. `MemoryTrace` is not generic over the source, so no memory core is. `into_ram_chunks(self)` returns the RAM group, which the chunk product of `RamRaProduct` takes after batch 5, so that its bytes are written once.

- `inc_lift(trace, word, lift)` returns `U`, one element per cycle, for the trace word `word`.
- `fold_words(words, lift, point)` returns, for `2^m` words and a point of `n ≤ m` coordinates, the `2^(m−n)` elements `out[c] = Σ_{low < 2^n} eq(point, low)·lift(words[c·2^n + low])`.
- `address_column(memory, point)` returns `out[j] = eq(point, addr_j)` for a point of `a` coordinates.
- `row_weights(plan, weight)` returns, over the bytecode rows of a `ScatterPlan`, `R[k] = Σ_{j: bytecode_index(j) = k} E(j)` with `E(j) = eq(point, j)` for `RowWeight::Eq(point)`, and with `E(j) = eq(point, j − 1)` for `j ≥ 1`, `E(0) = 0`, for `RowWeight::Next(point)`, the convention of `ChunkWeightTerm::Next`.

**Packed rows.** A `PackedBlock { row_variables, words, flag, powers }` describes a block of `2^μ` rows with values in `F128`, `μ = row_variables` equal to 4 or 5: `words`, two trace words of the source; `flag`, an optional group of one column of zero bits, with `f_j = 1` where its byte is non-zero; and `powers`, present groups whose columns, taken in order, are `P` digit columns of at most 4 bits, with `6 + P ≤ 2^μ`. The groups are those of `ValidatedTrace::prepare` on the same source and are taken by value. With `pack(w) = F128::from_raw(w as u128)`, the rows of cycle `j` are

| Row | `Az` | `Bz` | `Cz` |
|---|---|---|---|
| 0 | `f_j` | `pack(word_0(j))` | 0 |
| 1 | `1 + f_j` | `pack(word_1(j))` | 0 |
| 2 to 5 | 0 | 1 | 0 |
| `6 + p`, for `p < P` and `d` the digit of power column `p` at `j` | `x^d` | `x^(2d)` | `x^(3d)` |
| the rest | 0 | 0 | 0 |

A power of `x` below `x^46` is `F128::from_raw(1 << e)`. For `τ` of `μ + log_t` coordinates, row variables first, `OuterF128Core` proves

`Σ_{i < 2^μ, j < T} eq(τ, (i, j))·(Az[i, j]·Bz[i, j] + Cz[i, j])`

in `μ + log_t` rounds of degree 3 and returns `Az`, `Bz`, `Cz` at `(z, r')`; `into_groups` returns the groups of the block, which are the chunk groups later batches take. It serves `SpartanOuterF128`, whose block has the two key gates in rows 0 and 1, four residual rows whose `Az` is zero on an honest witness, and one power row per chunk (invariant 1).

**Pair sums.** `PairSumCore::new(pairs)` takes 1 to 8 pairs `(H_t, R_t)` of tables of one length `2^n`, `n ≥ 1`, proves `Σ_t Σ_k H_t[k]·R_t[k]` in `n` rounds of degree 2 and returns every table at the challenge point. It serves `SpartanInner`, with three pairs over the 1,024 witness columns, and `BytecodeReadAddress`, with four pairs over the bytecode rows. For the first, `routed_columns(shapes, folds)` returns `out[o] = Σ_ρ Σ_{(o, s, h) ∈ route_ρ} Fold_ρ[s, h]` over the output domain of the shapes, from the tables `fold_pass` returns.

**Register reads.** For coefficients `c = (c_0, c_1, c_2)` and `Sel[k, j] = c_0·[rs1_j = k] + c_1·[rs2_j = k] + c_2·[rd_j = k]`, `RegistersCore` proves

`Σ_{k < 32, j < T} eq(r_cycle, j)·Sel[k, j]·RegVal(k, j)`

in `5 + log_t` rounds, the five address variables first, with messages of degree 2 in the address rounds and 3 in the cycle rounds, and returns `Sel` and `RegVal` at `(z, r')`. `register_claims(trace, columns, z, row_weights)` returns `Σ_k R[k]·eq(z, reg(k))` for `reg` each of three `by_row` columns of at most 5 bits, read by bytecode row with `row_digit`; a row whose digit is `None` contributes nothing. With the columns of `rs1`, `rs2`, `rd` and `R = row_weights(plan, Eq(r'))` these are the three selector columns at `(z, r')`: a row without a digit is then a row no cycle visits, by the walk's check on the register group, and its `R[k]` is zero.

**RAM reads.** `RamCore` proves

`Σ_{k < K, j < T} eq(r_cycle, j)·[addr_j = k]·RamVal(k, j)`

in `a + log_t` rounds, address variables first, with the same degrees, for `1 ≤ a ≤ 32`, and returns `Ra(r') = Σ_j eq(r', j)·eq(z, addr_j)` and `RamVal(z, r')`. Every cycle is a read of its index, a cycle without an access of word 0. `into_address_column` returns the table `eq(z, addr_j)` over the cycles, unbound. `RamOptions { phase_bits, block_entries }`, by default 10 and 4,096, with `phase_bits` in `1..=12` and `block_entries ≥ 1`, changes no message.

**RAM output.** For final words `final[k]`, a range `mask` of words and the words `io` of the public memory on that range, `RamOutputCore` proves

`Σ_{k < K} eq(τ, k)·[k ∈ mask]·lift(final[k] ^ io[k])`

in `a` rounds of degree 3, for `1 ≤ a ≤ 32`, and returns `Σ_k eq(z, k)·lift(final[k])`, the extension of the whole final RAM. An empty mask is accepted: the messages are zero and the value is the same fold.

**Value evaluation.** With `lt(j, r)` the extension of `[j < index]` at `r`, `Rd[j] = eq(a_reg, rd_j)`, `S[j] = store_j` and a table `Ra` over the cycles, `ValEvaluationCore` proves, as two members of one batch,

`Σ_j lt(j, r_cycle)·Rd[j]·(1 + S[j])·U[j]` and `Σ_j (c_val·lt(j, r_cycle) + c_final)·Ra[j]·S[j]·U[j]`

in `log_t` rounds of degree 4 and returns `Rd`, `S`, `U` and `Ra` at `r'`.

**Public surface.**

| Item | Where | Contract |
|---|---|---|
| `MemoryTrace`, `inc_lift`, `fold_words`, `address_column`, `RowWeight`, `row_weights`, `MemoryError` | `memory` | Section "Memory views". `MemoryTrace::new(ram: PresentGroup, registers: PresentGroup, store: OptionalGroup)`; `cycles()`, `address_bits()`, `into_ram_chunks(self) -> PresentGroup`; the reads of a cycle's index, registers and store bit are crate-visible |
| `split_lt` | `round::eq`, crate-visible | `split_lt(point)` is the `SplitLt<F128>` of `jolt-kernels` for `lt(·, point)`, whose round `i` consumes coordinate `i`; `RoundError::EmptyPoint` for no coordinate |
| `PackedBlock`, `OuterF128Core`, `OuterF128Error` | `outer_f128` | `PackedBlock { row_variables: usize, words: [usize; 2], flag: OptionalGroup, powers: Vec<PresentGroup> }`; `OuterF128Core::new(trace, block, τ)`; `ProveRounds<F128>`; `final_values() -> [F128; 3]`; `into_groups(self) -> (OptionalGroup, Vec<PresentGroup>)` |
| `PairSumCore`, `PairSumError` | `pair_sum` | `PairSumCore::new(pairs: Vec<(Vec<F128>, Vec<F128>)>)`; `ProveRounds<F128>`; `final_values() -> &[(F128, F128)]`, in the order of the pairs |
| `routed_columns` | `router` (file `router/columns.rs`) | `routed_columns(shapes, folds) -> Result<Vec<F128>, RouterError>`, of `2^log_outputs` elements; it is the one reader of `route` on the prover's side of the fold, which `fold_pass` does not read |
| `RegistersCore`, `register_claims`, `RegistersError` | `registers` | `RegistersCore::new(memory, inc, r_cycle, coefficients)`, with `memory: Arc<MemoryTrace>` and `inc: Arc<Vec<F128>>`; `ProveRounds<F128>`; `final_values() -> [F128; 2]`; `register_claims(trace, columns: [usize; 3], z, row_weights) -> Result<[F128; 3], RegistersError>` |
| `RamOptions`, `RamCore`, `RamError` | `ram` | `RamCore::new(memory, inc, initial, lift, r_cycle, options)`, with `initial: Arc<Vec<u64>>` of `K` words and `lift: Arc<WordLift>`; `ProveRounds<F128>`; `final_values() -> [F128; 2]`; `into_address_column(self) -> Vec<F128>` |
| `RamOutputCore`, `RamOutputError` | `ram_output` | `RamOutputCore::new(final_words, mask, io, lift, τ)`, with `final_words: Arc<Vec<u64>>`, `mask: Range<usize>` and `io: &[u64]` of `mask.len()` words; `ProveRounds<F128>`; `final_value() -> F128` |
| `ValEvaluationCore`, `ValEvaluationMember`, `ValEvaluationError` | `val_evaluation` | `ValEvaluationCore::new(memory, inc, ram_ra, a_reg, r_cycle, [c_val, c_final])`, with `ram_ra: Vec<F128>`; `members() -> [ValEvaluationMember; 2]`, registers first, each a `ProveRounds<F128>`; `final_values() -> [F128; 4]` |
| `SyntheticTrace::{memory_columns, packed_columns}`, `SyntheticMemory` | `synth`, feature `test-utils` | Performance, "Synthetic memory" |

Every constructor, view and pass checks the lengths it is given (points against `log_t`, `a`, 5 and `μ + log_t`; tables and groups against `T`, `K` and `bytecode_rows()`; `io` against the mask) and returns its module's error. A group carries its columns and widths, and a constructor checks those; that a group was prepared from the trace it is used with is the caller's, as invariant 1 states. A core with cycle rounds on a trace of one cycle returns `RoundError::EmptyPoint` wrapped in its own error, as the cores of the kernels spec do.

### Invariants

Invariants 1 to 12 of the kernels spec hold for every item of this one. These are added or extended:

1. **Honest inputs (extends 6).** Every core of this spec is definitional for the tables and groups it is given, a violated key gate included. Two preconditions are not checked. The first is the caller's: the groups, the lifted word and the address table a core is given belong to one trace. A group records its columns, widths and cycle count and not the source it was prepared from, so a core given the groups of another trace proves the sum of that trace. The second is the adapter's: rows 2 to 5 of the protocol's block carry the packed residual forms in `Az`, and `PackedBlock` has zero there. They agree when the four forms are zero on every cycle, which an honest witness satisfies; on a witness that violates one the values of `OuterF128Core` belong to another table and the proof fails at the final check of batch 2.
2. **Nothing of `2^a·T` entries.** Storage and time are linear in `T`, in `K` and in `bytecode_rows()`. The work per RAM word is one `fold_words` of the initial words inside `RamCore` and one of the final words inside `RamOutputCore`.
3. **Representation (extends 3).** Beyond the tables that invariant names, a table of `F128` exists here as: `U`, one element per cycle and proof, the same lift that `source_lift` computes for the word `Inc`; the two cycle tables of `RegistersCore` and of `RamCore` from their first cycle round; one element per entry of `RamCore`, with at most one entry per cycle; the tables of `OuterF128Core` from its last row round; the halves of `U` and `Ra` in `ValEvaluationCore`; the pairs of a `PairSumCore`; the outputs of the views. No core copies a digit: the groups of the walk are the one copy, `d_a + 4` bytes per cycle for the memory trace and `P + 1` for the packed block.
4. **Tables in cache (extends 7).** The exceptions here are: the tables indexed by bytecode row or by RAM word; the checkpoints of `RamCore`, one arena read once per block and round; and, in `OuterF128Core` on a block of more than 16 power columns, the histograms of its first pass (257 elements each: 56 KiB at 12 columns, 80 at 16, 128.5 at 24) and the pair tables of its streamed rounds (12 KiB per pair of columns: 72 KiB at 12, 96 at 16, 144 at 24). Tables that one loop iteration writes at an equal value are placed by `BucketPlacement` of the kernels spec, never a multiple of 4,096 bytes apart: the histograms of `OuterF128Core` are byte positions of `BucketPlacement::Byte`, and the three histograms of `register_claims` are adjacent in one array, 512 bytes apart. No other pass of this spec writes two tables in one iteration.
5. **One variable order (extends 12).** `round::eq` gains `split_lt`. No other module reverses a point or constructs a `SplitLt`.
6. **Options change no message.** The messages and final values of `RamCore` are those of its definition for every `RamOptions`, every thread count and every partition into blocks.
7. **No shared-crate change (extends 9), no new dependency (extends 10).** The crate uses `SplitLt`, `LazyFoldedRa` and `ChunkIndexSource` of `jolt-kernels` and the split equality of `jolt-poly` as they are.
8. **One walk of the digits.** No constructor, view, pass or round of this spec calls `digit` on the source. What is read from the source per cycle is a trace word: `Inc` once, by `inc_lift`, and the two words of a `PackedBlock` in the first pass of `OuterF128Core` and in each of its row rounds from round 2 on. `register_claims` reads `row_digit`, once per bytecode row.

No `jolt-eval` invariant changes: nothing here is reachable from the existing prover.

### Non-Goals

Implementations of `SumcheckKernel` and `PrepareKernel` for the eight relations and their acceptance against the reference kernels, which a separate spec owns; the tables an adapter builds from public data (`H_t` of `BytecodeReadAddress`, the matrix weights and public columns of `SpartanInner`, the placement of the histograms of `fold_pass` in `direct`, the dense initial RAM, `InitEval`); a RAM of more than `2^32` words; any change to the protocol, to `prove_batch` or to a shared crate; witness generation; any commitment.

## Evaluation

### Acceptance Criteria

"By summation" and "rejected" are as in the kernels spec: computed from the definitions in Goal with `oracle` on dense tables built in the test, and refused by the prover's round check or by the verifier of `jolt-sumcheck`. A core takes its value at 1 from the claim. Seeded points have pairwise distinct coordinates.

- [ ] **Memory views.** For `log_t` from 3 to 8, RAM chunks of 1 to 8 bits with `a` from 1 to 12 and seeded traces, on groups of `ValidatedTrace::prepare`: the index, the registers and the store bit that `MemoryTrace` gives at every cycle equal the digits of the source, and `into_ram_chunks` returns the bytes it was given; `inc_lift`, `fold_words` for every `n` from 0 to `m`, `address_column` and `row_weights` in both variants equal their definitions by summation; `Σ_k R[k]` is 1 for `Eq` and `1 + eq(point, T − 1)` for `Next`.
- [ ] **Packed rows.** For `μ` of 4 and 5, `log_t` from 1 to 8, `P` from 0 to `2^μ − 6`, odd counts included, power columns of 1 to 4 bits, and traces with satisfied and with violated key gates: every round's coefficients equal the round polynomial of the dense table of `2^μ·T` rows by summation, `final_values` equal `Az`, `Bz`, `Cz` at the challenge point, and the one-member proof verifies from the claim by summation. The power columns come as one group and as two. The same with every coordinate of `τ` in `{0, 1}`. A proof with one coefficient changed is rejected. `into_groups` returns the bytes the block was given.
- [ ] **Packed rows, smallest case.** `μ = 4`, two cycles, no power column, `τ = 0`; the flag set at cycle 0, `word_0(0) = 2`, `word_1(0) = 0`. The claim is `F128::from_raw(2)` and the first message is `x·(1 + X)^3`, with the coefficients `[2, 2, 2, 2]` as raw values. A core that exchanged the two words would send `[0, 2, 0, 2]`.
- [ ] **Pair sums.** For 1 to 8 pairs of 2 to `2^10` entries: coefficients and final values by summation, the proof verifies, a changed coefficient is rejected. For the one pair `H = [1, 2]`, `R = [3, 1]` in raw values the claim is 1 and the message is `[3, 7, 6]`.
- [ ] **Routed columns.** For the five shapes of the kernels spec's criteria with seeded `route` sets and the tables of `fold_pass`: `routed_columns` equals `Σ_j eq(r_cycle, j)·Σ_ρ Σ_{(o, s, h) ∈ route_ρ} Source_ρ[s, j]·Select_ρ[h, j]` by summation over the cycles; with every `route` empty it is zero.
- [ ] **Register reads.** For `log_t` from 1 to 8 on traces in which every register is read through each operand and written, register 0 included, with stores: every round's coefficients and the final values equal their definitions by summation on the dense table of `32·T` entries; `register_claims` equals the three selector columns at `(z, r')` by summation, and `c_0·rs1 + c_1·rs2 + c_2·rd` of its values equals the core's `Sel`. Smallest case: two cycles, `r_cycle = (1)`, `c = (1, 0, 0)`, cycle 0 writes register 1 with `U[0] = 2` and is not a store, cycle 1 has `rs1 = 1`. The claim is 2 and the first message is `[0, 0, 2]`.
- [ ] **RAM reads.** For `a` from 1 to 10, `log_t` from 1 to 8, `phase_bits` from 1 to 4 and `block_entries` from 1 to 8, on traces with a load of word 0, a store to word 0, cycles without an access before and after it, two stores to one word, loads of a word never stored, and a non-zero initial RAM: every round's coefficients and the final values equal their definitions by summation on the dense table of `K·T` entries, for every setting of the options; `into_address_column` equals `eq(z, addr_j)`. Smallest case: `a = 1`, two cycles, zero initial RAM, cycle 0 stores to word 1 with `U[0] = 2`, cycle 1 loads word 1. With `r_cycle = (1)` the claim is 2 and the first message is `[0, 0, 2]`; with `r_cycle = (0)` the claim and the message are zero.
- [ ] **RAM output.** For `a` from 1 to 10 and masks at every alignment, the empty mask and the whole RAM included, with final words that agree and that disagree with `io` on the mask: coefficients by summation, the final value equals the extension of all `K` final words, and a disagreement gives a non-zero claim. Smallest case: `K = 2`, mask `0..1`, `io = [final[0]]`, `final[1] = 1`, `r_bit = 0` so that `lift(1) = 1`, `τ = (0)`. The claim is 0 and the message is `X·(1 + X)^2`, `[0, 1, 0, 1]`. A core that read no word outside the mask would send zero.
- [ ] **Value evaluation.** For `log_t` from 1 to 8, traces with no store, with stores and with only stores, and a seeded `Ra`: both members, proved as one batch through `prove_batch`, emit coefficients equal to their round polynomials by summation, under `SequentialRounds` and under a scheduler written in the test that visits the members in reverse order; the four final values by summation. Smallest case: two cycles, `r_cycle = (1)`, `rd = 3` and no store on both, `U = [2, 0]`, `a_reg = (1, 1, 0, 0, 0)`. The registers claim is 2 and its message is `x·(1 + X)^2`, `[2, 0, 2, 0, 0]`; the RAM member's message is zero.
- [ ] **Memory end to end.** At `log_t = 8` on `SyntheticTrace` with two RAM chunks (`a = 8`) and its `SyntheticMemory`: batch 4 as three members through `prove_batch` with the windows of `specs/rv64i-binary-protocol.md` §5; `row_weights` and `register_claims` at its point; batch 5 with the table of `into_address_column`; a `PairSumCore` of four pairs with `R` tables from `row_weights` and seeded `H` tables. Then: the address point of `RegistersCore` is the last five coordinates of that of `RamCore`; the registers member of batch 5 has the input claim `RegVal(z, r')` of batch 4; the RAM member with `(c_val, c_final) = (1, 0)` has the claim `RamVal(z, r') + Init` and with `(0, 1)` the claim `Final + Init`, where `Final` is the value of `RamOutputCore` and `Init = fold_words(initial, lift, z)[0]`; every final value equals its definition by summation.
- [ ] **Determinism.** At `log_t = 13` with `a = 6`, under a pool of 1 thread and under a pool of 12: the coefficients of `OuterF128Core`, `RegistersCore`, `RamCore` with `block_entries = 512`, `RamOutputCore` and both members of `ValEvaluationCore` equal their round polynomials by summation, and the views their definitions.
- [ ] **Allocation and scratch.** Under the counting allocator of the kernels crate's allocation fixture. On a one-thread pool, at `log_t = 8` and at `log_t = 14`: a core allocates at most `16·rounds + 64` times between the return of `new` and the return of `finish_rounds`, and a view, `register_claims` or `routed_columns` at most 16 times. On a pool of 12 threads, built once and alive across the measurements, between `log_t = 13` and `log_t = 19`: the allocation count of a core grows by at most `16·6`, for its six more rounds, and that of a view or pass by nothing, beyond `RAYON_WORKER_ALLOWANCE` (2 allocations and 112 bytes per worker thread) and, for `OuterF128Core::new`, one scratch array per worker. A pass over the cycles goes from 2 chunks to 128 between those sizes and `RamCore` with `block_entries = 512` to 64 times as many blocks, so one allocation per chunk or per block exceeds what the allowance admits on 12 workers. At the return of every view, pass and `finish_rounds` the live bytes are those at its start plus its outputs, within the same allowance.
- [ ] **Rejected inputs.** Each input below returns the variant named, with fields that name the offending value, and none panics. A variant is tested once, through the constructor, view or pass that owns the check.

  | Error | Variant: input |
  |---|---|
  | `MemoryError` | `AddressBits`: a RAM group of no column, and one of 33 bits in all. `RegisterColumns`: a register group of two columns. `RegisterWidth`: `rs1` on a column of 4 bits. `StoreColumns`: a store group of two columns. `StoreWidth`: a store column of one bit. `Cycles`: a register group prepared from a trace of half the cycles. `Word`: `inc_lift` on word `trace_words()`. `Words`: `fold_words` on 12 words, and on 4 words with a point of 3 coordinates. `PointLength`: `address_column` with `a + 1` coordinates, `row_weights` with `log_t − 1` |
  | `OuterF128Error` | `RowVariables`: 3 and 6. `Rows`: 11 power columns with `μ = 4`. `Word`: an index equal to `trace_words()`. `FlagColumns`: a flag group of two columns. `FlagWidth`: a flag column of one bit. `PowerWidth`: a power column of 5 bits. `Cycles`: a power group prepared from a trace of half the cycles. `PointLength`: `τ` of `μ + log_t − 1` coordinates |
  | `PairSumError` | `Pairs`: none, and nine. `Length`: tables of 12 entries, tables of one entry, and an `R` of half the length of its `H` |
  | `RouterError`, from `routed_columns` | `TableLength`: four tables for five shapes; a `Fold_ρ` of half its length; a shape whose output domain differs from the first's |
  | `RegistersError` | `IncLength`: `T/2` elements. `PointLength`: `r_cycle` of `log_t + 1` coordinates, and `z` of 4 in `register_claims`. From `register_claims`: `Column`: a column equal to `digit_columns()`. `PerCycle`: a column not `by_row`. `Width`: a column of 6 bits. `RowWeights`: a table of half the bytecode rows |
  | `RamError` | `PhaseBits`: 0 and 13. `BlockEntries`: 0. `InitialLength`: `K/2` words. `IncLength`: `T/2` elements. `PointLength`: `r_cycle` of `log_t − 1` coordinates |
  | `RamOutputError` | `Words`: 12 words, and one word. `Mask`: an end above the word count, and a start above the end. `IoLength`: one word short. `PointLength`: `τ` of `a − 1` coordinates |
  | `ValEvaluationError` | `IncLength`, `RaLength`: `T/2` elements. `PointLength`: `a_reg` of 6 coordinates, and `r_cycle` of `log_t − 1` |

  A core builds the tables of its `LazyFoldedRa` and the point of its `SplitLt` from lengths it has checked; their errors convert into the core's and have no case here. A column out of range, a column too wide for its group and a digit absent from a present group are errors of `ValidatedTrace::prepare`, tested with it.

- [ ] **Probe.** `memory_probe` reports the two forms of the batch-5 message (Performance, "Probe") before `val_evaluation.rs` is written, and the PR of that file states the form it builds and the figures that decided it.
- [ ] **Performance.** Every benchmark of the Performance section is reported through the runner, at both sizes and both thread counts, against its row. A result above its threshold at `log_t = 22` fails the PR that adds or changes the kernel.
- [ ] **Gates.** Those of the kernels spec: `cargo clippy --all-targets -- -D warnings` and `cargo nextest run --cargo-quiet` pass for `jolt-rv64i-kernels`, `jolt-field --features binary`, `jolt-poly` and `jolt-kernels`; `cargo fmt --check` passes.

### Testing Strategy

The ground truth is `oracle`, on dense tables the test builds from the definitions of Goal: `32·T` and `K·T` entries for the read checks, `2^μ·T` rows for the packed block. It shares no view, checkpoint, entry or round assembly with the cores. Each test asserts an algebraic identity against it, a rejection or a typed error, clauses (c) and (e) of `.claude/skills/test-policy/SKILL.md`; none compares one setting of `RamOptions`, one thread count or one form of a message with another, and every setting is tested against the same summation. The smallest cases are literals computed by hand from the definitions; each names the wrong core it tells apart. The tests of the kernels spec and of the shared crates pass unchanged. The `host` and `zk` modes do not apply.

Rustdoc on the public items states the contracts of the surface table, and in particular what is not checked: that `inc` is the lift of the word the protocol means and `lift` the lift it was made with; that `initial`, `final_words` and `io` are the RAM of the statement; that `ram_ra` is the address column of the same trace; and, on `PackedBlock`, invariant 1. Comments in the round loops are limited to what `.claude/skills/comment-policy/SKILL.md` keeps.

### Performance

The machine, the sizes, the runner, the id form `<bench>/<profile>/<log_t>/<threads>`, the profiles, the unit table with its two price points and the rule that derives a threshold from a model are those of the kernels spec, Performance. This section states operation counts in its symbols and applies its rule; it adds no unit and replaces no row.

**Synthetic memory.** `SyntheticTrace` gains columns and words derived from what it has, so that no existing column, word or benchmark changes: three `by_row` register columns of 5 bits, from bits 0–4, 5–9 and 10–14 of bytecode word 0; five RAM chunk columns that equal the existing RAM chunks on a cycle whose row has an access kind and are `Some(0)` elsewhere, the witness contract's index 0; and two trace words that equal trace words 0 and 1 where the `KeysDiffer` flag is clear, respectively set, and are zero elsewhere, so that both key gates hold. `memory_columns()` names those RAM chunks, the register columns and the existing store flag, and `packed_columns()` the `KeysDiffer` flag, the two words and the twelve chunk columns, for a block of `μ = 5`. A benchmark's fixture asks `ValidatedTrace::prepare` for the groups in one request, outside the timed phases, as it makes the validated trace and the plans. `SyntheticMemory::new(trace, seed)` gives seeded initial words, the final words (the initial words with the increments of the stores applied), a mask of the first `2^12` words and the final words on it as `io`. On `local` and `all_rows` a cycle is a load with probability 0.25 and a store with 0.10; `RamCore` then has `f = 0.45` entries per cycle, of which `s = 0.10` are stores (Design, "RAM reads").

`outer_f128`, `ram`, `ram_output` and `val_evaluation` run on `local`; `registers`, `bytecode_address` and `memory` on `all_rows`, because they scatter by bytecode row. A benchmark covers the views that feed its core and nothing else pays for: `registers` covers `inc_lift`, the core, one `row_weights` and `register_claims`; `ram` covers `inc_lift` and the core at `a = 20`; `ram_output` covers the core on `2^20` words; `val_evaluation` covers `inc_lift`, `address_column` and both members through `prove_batch`; `bytecode_address` covers two `row_weights`, one of each variant, and a `PairSumCore` of four pairs; `inner/routes` covers `routed_columns` over five shapes with the 75,371 route triples of the reference layout and a `PairSumCore` of three pairs of 1,024 entries; `memory` covers the sequence of the end-to-end criterion at `a = 20`, each view once.

**Requirements.** Counts are per cycle, with the pairs of all cycle rounds summed.

| Benchmark | Operations per cycle |
|---|---|
| `outer_f128/local` | `17 M + 20 A + 4 R + 57 L + 15 Bk + 60 w`, and `3·10^4 M` once |
| `registers/all_rows` | `10 M + 40 A + 8 R + 28 L + 6 Bk + 1 sct + 24 w + ρ·3 Bk`, and `31 M` per chunk of 4,096 cycles |
| `ram/local` | `17.325 M + 20 A + 10.9 L + 2.2 Bk + 0.9 sct + 0.675 mrg + 24.1 w + κ·(8 L + 1 A)` |
| `val_evaluation/local` | `14.125 M + 11 A + 20 L + 8 X + 10 w` |
| `bytecode_address/all_rows` | `2 M + 2 sct + ρ·(8 M + 8 A)` |
| `memory/all_rows` | `42.45 M + 71 A + 8 R + 40.9 L + 8.2 Bk + 3.9 sct + 0.675 mrg + 8 X + 48.1 w + ρ·(8 M + 8 A + 3 Bk) + κ·(16 L + 2 A)`, and the terms paid once of `registers` and `ram_output` |

| Benchmark | Model at point 1: `22`, `20` | Threshold: `22`, `20` | Model at point 2: `22`, `20` | Threshold at point 2: `22`, `20` |
|---|---:|---:|---:|---:|
| `outer_f128/local` | 90.7 | 113 | 64.9 | 81 |
| `registers/all_rows` | 85.8, 87.1 | 107, 109 | 56.5, 57.8 | 71, 72 |
| `ram/local` | 63.1, 66.4 | 79, 83 | 38.9, 41.8 | 49, 52 |
| `val_evaluation/local` | 48.0 | 60 | 30.5 | 38 |
| `bytecode_address/all_rows` | 12.3, 29.9 | 15, 37 | 7.8, 17.4 | 10, 22 |
| `memory/all_rows` | 200.8, 226.3 | 251, 283 | 126.1, 142.9 | 158, 179 |

`memory` is the sum of `registers`, `ram`, `ram_output`, `val_evaluation` and `bytecode_address` less what they repeat: `inc_lift` twice and `address_column` once, `18 L + 1 M + 10 w` in all. The thresholds of point 1 are in force, under the kernels spec's rule and its rule for a replaced unit. On 12 threads at `log_t = 22` the requirement is the single-thread threshold divided by 9.6: 12 ns of wall time per cycle for `outer_f128` and 26 for `memory`.

The bytes of the groups are not in these counts. The walk writes them, and it is held to its own requirement where the source is specified; by count the groups of this spec add to it, at the reference layout, five byte stores per cycle, three comparisons of a register with its row and two flag reads, about 2 ns per cycle. The chunk bytes are the ones the chunk products already ask for.

Two cores have a model that the rule's rounding to a nanosecond per cycle cannot hold, and carry the same factor of 1.25 in their own unit:

| Benchmark id | Operations | Model, point 1 / point 2 | Requirement |
|---|---|---:|---|
| `ram_output/local` | `8 L + 1 A` per RAM word, and `16 L + 5 M + 2 A` per word of the mask once | 4.37 / 3.95 ns per RAM word | at most 5.5 ns per RAM word (4.9 at point 2): 1.4 ns per cycle at `log_t = 22` and 5.5 at `log_t = 20` |
| `inner/routes` | `75,371 Bk + 3,072·(2 M + 2 A)` per proof | 63.2 / 55.1 μs | at most 79 μs per proof (69 at point 2) |

`memory_machinery` reports three views on their own, through `run_machinery`: `fold_words` (`8 L + 1 A`, 4.3 ns per word), `address_column` (`1 M + 2 L + 10 w` at five chunks and `a ≤ 22`, 3.1 ns per cycle) and `row_weights` (`1 M + 1 sct`, 3.2). They have no requirement of their own: each is a term of a model above, and its units are the ones `machinery/lift` and `machinery/scatter` of the kernels spec already hold to a requirement.

**By phase**, at `log_t = 22`, with the values at `log_t = 20` in brackets where they differ:

| Phase of `outer_f128`, at 12 power columns and `μ = 5` | Operations | Point 1 | Point 2 |
|---|---|---:|---:|
| First pass: weights, gate sums, 15 histogram updates, keys formed from the group bytes | `1 M + 2 A + 15 Bk + 30 w` | 14.5 | 12.8 |
| Row rounds 0 and 1, from the statistics | `3·10^4 M` once | 0.0 | 0.0 |
| Row round 2, streamed | `2 M + 8 A + 2 R + 19 L + 10 w` | 22.0 | 15.9 |
| Row round 3, streamed | `2 M + 6 A + 2 R + 19 L + 10 w` | 19.8 | 14.5 |
| Row round 4, with materialisation and bind | `7 M + 2 A + 19 L + 10 w` | 23.1 | 15.8 |
| Cycle rounds | `5 M + 2 A` | 11.4 | 5.9 |
| **Total** | | **90.7** | **64.9** |

| Phase of `registers` | Operations | Point 1 | Point 2 |
|---|---|---:|---:|
| `inc_lift` | `8 L` | 3.2 | 3.2 |
| Checkpoints | `1 Bk`, and `31 M` per chunk | 0.6 | 0.6 |
| Address round 0 | `6 A + 1 Bk + 4 w` | 7.4 | 5.0 |
| Address rounds 1 to 4 | `1 M + 8 A + 2 R + 4 L + 1 Bk + 4 w` each | 57.7 | 37.2 |
| Cycle tables | `1 M + 4 L + 4 w` | 3.6 | 2.7 |
| Cycle rounds | `4 M + 2 A` | 9.5 | 5.0 |
| `row_weights`, `register_claims` | `1 M + 1 sct + ρ·3 Bk` | 3.7 [5.0] | 2.8 [4.1] |
| **Total** | | **85.8 [87.1]** | **56.5 [57.8]** |

| Phase of `ram`, at `a = 20` in five chunks, `f = 0.45`, `s = 0.10` | Operations | Point 1 | Point 2 |
|---|---|---:|---:|
| `inc_lift` | `8 L` | 3.2 | 3.2 |
| Entries and their weights, with the index read from its five bytes | `1 M + f·sct + 12.1 w` | 3.1 | 2.1 |
| 20 address rounds | `f·(1 M + 2 A) + s·(1 M + 1 Bk)` each | 41.1 | 23.7 |
| Checkpoints of two phases | `0.5·f M + 2·f L + 2·s Bk + 1.5·f mrg` | 1.1 | 0.9 |
| Regroup | `f·sct` | 0.6 | 0.6 |
| Cycle tables | `(1 + s) M + 2 L + 12 w` | 3.4 | 2.4 |
| Cycle rounds | `4 M + 2 A` | 9.5 | 5.0 |
| Fold of the initial words | `κ·(8 L + 1 A)` | 1.1 [4.3] | 1.0 [3.9] |
| **Total** | | **63.1 [66.4]** | **38.9 [41.8]** |

| Phase of `val_evaluation` | Operations | Point 1 | Point 2 |
|---|---|---:|---:|
| `inc_lift`; `address_column` | `8 L`; `1 M + 2 L + 10 w` | 6.3 | 5.4 |
| `lt` pairs | `2 M` | 3.7 | 1.8 |
| Three quadratics and their values at two nodes | `9 M + 8 X` | 18.1 | 9.7 |
| Products of the two members | `11 A` | 12.1 | 7.7 |
| Binds: two dense tables, two lazy columns | `2.125 M + 10 L` | 7.9 | 5.9 |
| **Total** | | **48.0** | **30.5** |

`bytecode_address` is two `row_weights`, 6.5 ns (4.6 at point 2), and four pairs at `ρ·(8 M + 8 A)`, 5.9 [23.4] ns (3.2 [12.8]). The worst case of `ram` is a trace in which every cycle is an access to a non-zero index and half are stores: `37 M + 42 A + 12 L + 11 Bk + 2 sct + 1.5 mrg + 24.5 w + κ·(8 L + 1 A)`, 131 ns per cycle at point 1 and 80 at point 2; a trace that touches every group of `2^phase_bits` words adds at most `κ·(8 L + 1 M + 2 mrg)`. The model of `ram` moves with the trace and its threshold is that of the `local` profile.

Between the two points only M, A and R move. The estimated units are 34.8 ns of `outer_f128` and 33.0 of `memory`: 23% of the two models at point 1 and 35% at point 2.

**Sensitivity to the estimates.** The first cores of the kernels spec, timed on a loaded machine, bounded the units from above at M 1.59, A 0.35, L 0.64, Bk 0.84, `sct` 3.1, `mrg` 1.1, X 1.2 and w 0.04. Those figures replace no row; they say which margins rest on an estimate. Three price sets follow: (i) Bk and `sct` at those figures and the rest at point 1; (ii) every unit at those figures, with R = M − A; (iii) every unit at the larger of its row and its figure.

| Benchmark | Threshold | (i) | (ii) | (iii) |
|---|---:|---:|---:|---:|
| `outer_f128/local` | 113 | 94.3 | 90.5 | 110.2 |
| `registers/all_rows` | 107 | 89.1 | 67.5 | 100.1 |
| `ram/local` | 79 | 65.2 | 49.2 | 68.8 |
| `val_evaluation/local` | 60 | 48.0 | 49.1 | 60.8 |
| `bytecode_address/all_rows` | 15 | 15.7 | 13.3 | 15.7 |
| `memory/all_rows` | 251 | 209.6 | 167.0 | 233.2 |
| `ram_output/local`, per RAM word | 5.5 | 4.4 | 5.5 | 6.3 |
| `inner/routes`, μs | 79 | 81.3 | 75.2 | 81.3 |

The two rows that carry the budget, `outer_f128` and `memory`, and the two largest cores inside `memory` keep a margin under all three. Four small requirements do not, and each is named for what it rests on. `bytecode_address` at `log_t = 22` is two scatters and little else. `inner/routes` is one bucket update per route triple. `ram_output` is one `WordLift` per RAM word, at 1.25 times a price that `machinery/lift` holds to 1.5 times: a lift at its own requirement fails this one (6.0 against 5.5). `val_evaluation` in the node form pays `8 X`. None of the four is moved to fit: a result above one of them is settled by the rule for a replaced unit, which recomputes every threshold from the measured price, and for `val_evaluation` by the probe.

**Probe.** `memory_probe` runs through `run_probe`, before `val_evaluation.rs` is written, the two forms of the batch-5 message on `2^20` pairs of the `local` profile, on 1 and 12 threads. The node form evaluates the two members at 0, at two nodes and at infinity: `11 A + 8 X` per pair. The coefficient form computes the coefficients of the same products and skips the one of `X^2`, which the claim gives: `16 A`. The difference, coefficient form less node form, is `5 A − 8 X`: 3.9 ns at point 1, 1.9 at point 2, and −7.9 at the loaded figures, where X was six times its row. The node form is built unless the coefficient form measures faster by more than the spread of repeated runs. The model above prices the node form and the threshold applies to the form that ships, as for the layouts of `fold_pass` in the kernels spec; the coefficient form is 51.9 at point 1 and 56.7 under price set (iii).

**Options.** `ram` is reported at `phase_bits` of 8, 10 and 12 and at the default `block_entries`; the default changes when another setting is faster by more than the spread of repeated runs. No other core has an option.

**What the models leave out.** Everything the kernels spec's models leave out: memory traffic of passes over tables of `T` elements, tables outside L1, short `fmadd` chains, allocation and first touch, parallel efficiency. In addition: `ValidatedTrace::prepare` with the groups it writes, `ScatterPlan::new` and the input claims, which a benchmark's fixture makes outside the timed phases, the claims by one sequential replay of the trace; the lookup tables, assigned to the construction phase; the dependence of a read on a write to the same cell one or two cycles earlier in the scans of `RegistersCore` and `RamCore`, a store followed by a load of one address, which the scans do not avoid and a chain of about 1.5 ns per write bounds; the cache behaviour of `RamCore` on a trace whose blocks leave the 16 KiB of one group's state, which the `local` profile does not exercise and the worst-case figure above does not price; and what the adapter builds, by count `ρ·10 M` for the `H` tables of batch 6a (4.6 ns per cycle at `log_t = 22`, 18.3 at 20) and `κ` word operations for the dense initial RAM. The thresholds are what a PR is held to. The models are not a claim that a core meets them.

**Budget.** The thresholds at `log_t = 22` are 113 ns for `outer_f128` and 251 for `memory`: 364 ns per cycle for the eight members (396 at `log_t = 20`), on models of 291.5 (317.0). `inner/routes` adds 0.02. The budget of the kernels spec is 2,100 ns of single-core work per cycle and 918 ms of wall time at `2^22` cycles on 12 threads. Its three pipelines hold 745 ns, and with the source walk, the lanes and the setup of `specs/rv64i-binary-prover-adapters.md` 795 ns are spoken for, which leaves 1,305 ns and 563 ms for these eight members, the rest of witness generation, the commitment and the opening. The eight cores take 364 ns of the 1,305, and at 9.6 effective cores 159 ms of the 563: 941 ns and 404 ms remain for witness generation, the commitment, the opening and the adapters' own passes over public tables. The sum fits. The largest single terms are `outer_f128` and the five address rounds of `registers`, 65 ns of that core's model. About 2 ns per cycle move from these cores into the walk with the groups, and are the walk's. As in the kernels spec this is an allocation of the budget and not a measurement of the prover.

**Memory.** Every table of `T`, `K` or `bytecode_rows()` elements has one owner:

| Object | Size | Owner | Created | Dropped |
|---|---|---|---|---|
| `MemoryTrace`: the RAM, register and store groups | `d_a + 4` bytes per cycle, 9 at the reference layout | the walk's caller, then `MemoryTrace` by value, shared by `Arc` | `ValidatedTrace::prepare`; the RAM group comes from `OuterF128Core::into_groups` when batch 1 held it | each of `RegistersCore` and `RamCore` drops its handle at its first cycle round and `ValEvaluationCore` at its fourth bind; the caller then takes the RAM group with `into_ram_chunks` and the rest is freed |
| `U` | 16 bytes per cycle | caller, shared by `Arc` | `inc_lift` | each of `RegistersCore` and `RamCore` drops its handle at its first cycle round; `ValEvaluationCore` at its first bind |
| Register checkpoints | 512 bytes per chunk of cycles, halved at each address round | `RegistersCore` | construction | first cycle round |
| `Sel`, `RegVal` over the cycles, with second buffers | 48 bytes per cycle | `RegistersCore` | first cycle round | `finish_rounds` |
| Entries; their increments | 24 bytes per entry and 16 per store, 12.4 bytes per cycle on `local`; twice during a regroup | `RamCore` | construction | first cycle round |
| RAM checkpoints | 16 KiB per block at `phase_bits = 10`: at most 16 bytes per RAM word and 16 KiB per `block_entries` entries | `RamCore` | each phase | the next phase |
| Initial words bound through a phase | 16 bytes per `2^phase_bits` RAM words | `RamCore` | each phase after the first | the next phase |
| `eq(z, addr_j)`, unbound; its halves | 16 and 12 bytes per cycle | `RamCore`, then the caller through `into_address_column`, then `ValEvaluationCore` | first cycle round | the halves at `finish_rounds`; the table at the first bind of batch 5 |
| `RamVal` over the cycles, with second buffer | 24 bytes per cycle | `RamCore` | first cycle round | `finish_rounds` |
| Initial and final RAM words | 8 bytes per RAM word each | caller, shared by `Arc` | by the adapter | after batch 4 |
| Window tables of the output check | 48 bytes per word of the mask, and two words | `RamOutputCore` | construction | `finish_rounds` |
| The groups of a `PackedBlock` | `P + 1` bytes per cycle, 11 at the reference layout | the walk's caller, then `OuterF128Core` by value | `ValidatedTrace::prepare` | returned by `into_groups` after `finish_rounds` |
| The two words of a `PackedBlock` | 16 bytes per cycle, in the source | the caller's source | by the adapter's pass over the rows | when the core drops its handle of the trace, at its last row round |
| `Az`, `Bz`, `Cz` of the two last cells | 96 bytes per cycle; from the first cycle round the tables of the second cell are the second buffers | `OuterF128Core` | last row round | `finish_rounds` |
| Halves of `U` and of `Ra` | 12 bytes per cycle each | `ValEvaluationCore` | first bind | `finish_rounds` |
| Lazy columns `Rd`, `S` | the bytes of the memory trace through three binds; 1.5 bytes per cycle and column from the fourth | `ValEvaluationCore` | fourth bind | with the core |
| Pairs, with second buffers | 48 bytes per entry and pair: 192 bytes per bytecode row for the four of batch 6a | `PairSumCore`, from the caller | by the caller | `finish_rounds` |
| Output of `row_weights`; its scatter buffer | 16 bytes per bytecode row; 16 bytes per cycle | caller; `ScatterPlan::scatter` | each call | when the pair that holds it is dropped; at the return |

The smaller state: the tables of one `split_eq` per core with an equality factor and of the one `SplitLt` (about `3·2^(log_t/2)` elements each); the 32 register cells and the `2^phase_bits` cells of a block's state, on the stack of a chunk; the histograms of `OuterF128Core::new`, from a `ScratchPool`; `WordLift` (32 KiB), shared. Bytes per cycle of the dominant capacities, the sources excluded: 107 for `OuterF128Core` at its last row round, its six tables and the groups, next to the 16 of the two words; 125 for batch 4 at its first cycle round, next to 16 bytes per RAM word of initial and final words; 57 for batch 5 at its first bind. They are estimates of capacities and not peaks under the counting allocator. No `jolt-eval` objective moves.

## Design

### Architecture

The round messages, the lifts, the equality helpers, the chunks of `CycleChunks`, the scratch pool and the scatter are those of the kernels spec, Design. What follows is what each core adds.

#### Memory views

`MemoryTrace` holds the three groups as the walk wrote them and adds no table. A read of a cycle's index shifts and ORs the `d_a` bytes of its RAM digits, two word operations per chunk; `RamCore` makes it once per cycle when it builds its entries and `address_column` once. The registers and the store bit are single bytes, three adjacent and one in a stream of its own, which the scans of `RegistersCore` read in cycle order. `fold_words` forms `eq(point, low)` as a product of two half tables: the inner sum is an `fmadd` chain against the low half, reduced once per block and multiplied by the high half. `address_column` splits the address into pieces of at most 11 bits, with one table per piece: two pieces and one multiplication for `a ≤ 22`. `row_weights` passes the weight `E(j)`, a product of two half-table entries, to `ScatterPlan::scatter`; for `Next` the index is `j − 1` and cycle 0 has weight zero.

#### Packed rows

*First pass.* With `e_j = eq(τ_cycle, j)` formed by one multiplication, the pass accumulates the four gate sums `G[w][f] = Σ_{j: f_j = f} e_j·pack(word_w(j))`, the flag sum `Φ = Σ_j e_j·f_j`, one histogram of `e_j` per pair `π` of power columns `(2π, 2π + 1)`, keyed by the byte `d + 16·d'` of their digits, and, for every two pairs that share a block of four rows (pairs `π` and `π + 1` with `π` odd), four histograms keyed by one digit of each. The digits are the bytes of the power groups, and the key of a pair is formed from two of them at each read; no key is stored. At 12 power columns that is 6 pair histograms, 8 cross histograms and the flag sum, 15 bucket updates per cycle in 56 KiB of scratch per worker. A cycle writes all 14 histograms, and writes several at an equal key whenever digits repeat, as the zero digits of a small index do. The histograms are therefore byte positions of `BucketPlacement::Byte`: histogram `n` starts at `position_offset(n)`, 257 elements after the last, and the scratch array of a worker has `position_offset(N)` elements for `N` histograms. Two entries then lie a multiple of 4,096 bytes apart exactly when the difference of their histogram numbers and that of their keys sum to a multiple of 256, which at an equal key no two of fewer than 256 histograms do. Unequal keys can still coincide, as the kernels spec says of its own tables; the placement removes the case that repeats on every cycle.

*Row rounds 0 and 1* come from those statistics. The round polynomial is `l(X)·q(X)` with `l` the linear factor of `eq(τ, ·)`, and `q` is given by `q(0)` and its leading coefficient; `q(1)` follows from the claim by the Gruen helpers. In round 0, with rows paired as `(2y, 2y + 1)`: the gate pair has `q(0) = G[0][1]` and the leading coefficient `G[0][0] + G[0][1] + G[1][0] + G[1][1]`; the two pairs of unit rows contribute nothing; pair `π` of power rows has `q(0) = 0` and the leading coefficient `Σ_κ Hist_π[κ]·(x^d + x^d')·(x^(2d) + x^(2d'))`. In round 1, after `r_0`, a folded power pair is `A_π = (1 + r_0)·x^d + r_0·x^d'`, and `B_π`, `C_π` likewise, as functions of the key. The block of the gate pair and a unit pair has `q(0) = Σ_f (f + r_0)·((1 + r_0)·G[0][f] + r_0·G[1][f])` and the leading coefficient `q(0) + Φ + r_0`; the block of a unit pair and power pair 0 has `q(0) = 0` and the leading coefficient `Σ_κ Hist_0[κ]·A_0·(1 + B_0)`; a block of two power pairs has `q(0) = Σ_κ Hist_π[κ]·(A_π·B_π + C_π)` and the leading coefficient `Σ Hist_π·A_π·B_π + Σ Hist_π'·A_π'·B_π'` plus the cross term `Σ_j e_j·(A_π·B_π' + A_π'·B_π)`, which is a sum of four terms each a function of one digit of each pair, read from the four cross histograms. Each pair or block is weighted by the equality factor of its remaining row variables. A third round from statistics would need histograms over four pairs.

*Streamed rounds.* From round 2 to round `μ − 2` the pass reads the bytes of the power groups, forms the keys and reads, per matrix and pair of columns, one table of 256 elements that holds the pair's two rows folded at the challenges so far. One lookup per pair and matrix (18 at 12 columns) and one by the flag give `Az` and `Cz` of every cell; the gate's `Bz` is two multiplications of a weight by a packed word, read from the source. The products of a cycle are accumulated unreduced, reduced once for `q(0)` and once for the leading coefficient, and multiplied into the block sums against the low half of the split equality.

*Last row round and cycle rounds.* Round `μ − 1` has two cells. The pass materialises `Az`, `Bz`, `Cz` of both, six tables of `T` elements, forms the message from them and drops its handle of the trace; the bind writes into the tables of the first cell. The groups stay with the core, unread, until `into_groups`. The cycle rounds are those of `OuterF2Core`: `5 M + 2 A` per pair.

#### Pair sums and routed columns

`PairSumCore` computes, per pair of adjacent entries and pair of tables, `H_0·R_0` and `(H_0 + H_1)·(R_0 + R_1)` by `fmadd` into two accumulators, takes the value at 1 from the claim, and binds between two buffers. `routed_columns` adds one entry of `Fold_ρ` into `out[o]` per route triple, through `RouterShape::fold_index`, the crate-visible function that gives the index of `(s, h)` in a shape's table; the index is not written twice.

For `SpartanInner` the pairs are `(M, routed)`, `(M, direct)` and `(M, public)` over the 1,024 columns, `M` being the matrix weight. `routed` is `routed_columns` of the tables of one `fold_pass` at `r_1`, which therefore runs before batch 2 and serves batch 3a as well; `direct` is filled from the histograms that pass returns for the chunk columns and the `KeysDiffer` flag. For `BytecodeReadAddress` the pairs are `(H_Router, R_Router)`, `(H_Read, R_Read)`, `(H_Val, R_Val)` and `(Pc, c.Entry·R_Entry + c.Next·R_Next)`, the last because `H_Entry` and `H_Next` are multiples of the one table `lift(row.pc)`. `R_Router` is the `row_weights` that `claims_pass` returns at `r_3`; `R_Read` and `R_Val` are `row_weights` at `r_4` and `r_5`; `R_Next` is `row_weights(Next(r_3))`; `R_Entry` is 1 at `bytecode_index(0)`. The member's one value is the sum of the products of the four pairs of final values.

#### Register reads

*Address rounds.* Round `i` binds bit `i` of the register index. Let `V_i[m, j]` be `RegVal` with its low `i` bits bound at the challenges so far, a table of `32 >> i` cells per cycle, and `p = c_x·eq(r_{<i}, low_i(x))` the weight of an operand `x` of cycle `j`, with `m = x >> i`. The message has degree 2:

`s(0) = Σ_j e_j Σ_{x: m even} p·V_i[m, j]`, and the leading coefficient is `claim + Σ_j e_j Σ_x p·V_i[m ^ 1, j]`,

with `s(1)` from the claim. The core never holds `V_i` over the cycles. It holds the cells at the start of every chunk of cycles, built at construction by one pass (`delta[rd_j] ^= U[j]` on a cycle that is not a store, then a running XOR over the chunks) and bound at each challenge, 31 multiplications per chunk in all. A round is one scan: each chunk starts from its checkpoint, and per cycle reads the cell and its sibling for the three operands, then applies the write `V[rd_j >> i] ^= eq(r_{<i}, low_i(rd_j))·U[j]`. In round 0 the weights are the three coefficients, applied per block, so a cycle costs six `fmadd` against the low half of the split equality. In rounds 1 to 4 the weights come from three tables of `2^i` entries; the six products of a cycle are accumulated unreduced, reduced twice and multiplied into the block sums.

*Cycle rounds.* After round 4 the checkpoints are one element per chunk, the value of `RegVal(z, ·)` at the chunk's start. One pass fills `Sel[j] = Σ_x c_x·eq(z, x_j)`, three lookups, and `RegVal(z, j)`, a running XOR of `eq(z, rd_j)·U[j]` over the cycles that are not stores. The `log_t` rounds are `Σ_j eq(r_cycle, j)·Sel[j]·RegVal(z, j)` in the form `l·q`.

`register_claims` adds `R[k]` into one of 32 cells per operand and bytecode row and then takes three inner products with `eq_table(z, None)`. The three histograms are one array of 96 elements, so that two of them written at one register lie 512 or 1,024 bytes apart. The core does not return the three columns separately because it binds their combination; `R = row_weights(plan, Eq(r'))` is needed by batch 6a in any case.

#### RAM reads

*Entries.* An entry is a weight `A`, an address and a time; a store has an increment `D` beside it. There is one entry per store, with `A = e_j` and `D = U[j]`; one per other cycle with a non-zero index; and one per run of cycles that are not stores and have index 0 between two consecutive stores, with `A` the sum of their `e_j`, which is exact because no cell changes between two stores. On `local` that is 0.10, 0.25 and about 0.10 entries per cycle.

*Phases.* The address bits are bound in phases of `ℓ = phase_bits`. In phase `p` an entry belongs to the group `address >> ℓ·(p + 1)` and its cell is the next `ℓ` bits below; entries are ordered by group, then time, and a group is cut into blocks of at most `block_entries` entries. A block has a checkpoint, the `2^ℓ` cells of its group at the time of its first entry: the initial words of the group, lifted in phase 0 and taken from `fold_words(initial, lift, r_{<ℓ·p})` afterwards, plus the increments of the group's earlier stores. The checkpoints of a phase are one arena, bound at each challenge. At a phase change the entries are regrouped by one partitioned move. The last phase has one group, in time order.

*A round* is one scan per block, with a state `M` copied from the block's checkpoint. For an entry with cell `c`: `A·M[c]` goes to `s(0)` when `c` is even and `A·M[c ^ 1]` to the leading coefficient, which is `claim` plus that sum, as for the registers; a store then does `M[c] ^= D`. The bind of the previous challenge is fused into the scan, one multiplication of `A` by `eq(r, bit)` per entry and one of `D` per store. Blocks are independent, so the scan is parallel over blocks and merges by XOR.

*Cycle rounds.* After the last address round, `address_column(memory, z)` gives `Ra[j]`, and `RamVal(z, j)` is the running XOR of `Ra[j]·U[j]` over the stores, started from the bound initial words. `Ra` stays unbound for `into_address_column`: its first bind writes a buffer of `T/2`.

#### RAM output

The round polynomial is `l(X)·q(X)` with `q` the extension of `[k ∈ mask]·F[k]`, `F[k] = lift(final[k] ^ io[k])`. The core holds `Mask`, `F` and `Io` on a window, the cells of the current round that meet the mask. A pair with one cell in the window and one outside contributes through the outside cell's `F`, so before each round the window is aligned to pairs by adding at most one cell on each side. An added cell of round `i` lies wholly outside the mask: its `Mask` and `Io` are zero and its `F` is `fold_words` of its `2^i` final words at the challenges so far. The added cells are disjoint dyadic ranges that tile the complement of the mask, so every word outside the mask is lifted and folded exactly once over the `a` rounds, and after the last round the window is the one cell `F(z) = Final(z) + Io(z)`; the value returned is `F(z) + Io(z)`. An empty mask starts from one cell with `Mask = 0`.

#### Value evaluation

The factors are `lt` (the `SplitLt` of `split_lt(r_cycle)`, two multiplications per pair), `Rd` and `S` (one `LazyFoldedRa` over a private two-column `ChunkIndexSource` on the memory trace: column 0 is `rd_j` with the table `eq_table(a_reg, None)` and the bound 32; column 1 is `Some(0)` on a store, with the table `[1]` and the bound 1), `U` (shared and read in round 0; the first bind writes the core's own `T/2`) and `Ra` (owned; dropped at the first bind). Per pair the core forms three quadratics, `Q_1 = lt·Rd`, `Q_2 = lt·Ra`, `Q_3 = S·U`, nine multiplications, and uses `(1 + S)·U = U + Q_3`. The registers member is `Q_1·(U + Q_3)`; the RAM member is `c_val·Q_2·Q_3 + c_final·Ra·Q_3`, the two coefficients applied after the sum over the pairs. In the node form each product is accumulated at 0, at the nodes `F128::from_raw(2)` and `F128::from_raw(3)` and, for the two of degree 4, at infinity; `coefficients_from_nodes` recovers the message at degree 4 with the value at 1 from each member's claim. The two members share one core behind a lock, the first one called in a round binds and makes the pass, as `RoutersCycleCore` does, and need no scheduler other than `SequentialRounds`.

#### Fit with `jolt-kernels` and the protocol

For each member, whether the kernel of `crates/jolt-kernels/src/optimized/` for the same shape serves: (a) as it is over `F128`, (b) after a generalisation of the shared crate, or (c) not.

| Member | Decision | Why |
|---|---|---|
| `SpartanOuterF128` | (c) | `spartan_outer.rs` evaluates rows as integers and extends them over an integer domain, as the kernels spec records; here the rows are 6 to 30 packed rows per cycle and five row variables |
| `SpartanInner` | (c) | No shared kernel has this shape; the sum is three dense pairs of 1,024 entries |
| `RegistersReadChecking`, `RamReadChecking` | (c) | `read_write.rs`, `rw_matrix.rs` and `registers_read_write/` prove a read and a write relation batched by a challenge, on sparse matrices whose entries carry values as `u64` and increments as `i128` turned into field elements by `from_u64` and `from_i128`, and they halve (`F::from_u64(2)`). Here an increment is an XOR of lifted words, the relation is one read sum, and values are packed words until lifted. The structure of `registers_read_write/address_first.rs` is what `RegistersCore` keeps: checkpoints per chunk of cycles, one scan per address round, a hand-off to cycle tables |
| `RamOutputCheck` | (c) | `ram_output_check.rs` holds the final RAM as field elements built by `from_u64` on the whole domain; here the words stay packed and are lifted once, inside the fold |
| `RegistersValEvaluation`, `RamValEvaluation` | (c), with (a) for `SplitLt` and `LazyFoldedRa` | `registers_val_evaluation.rs` and `ram_val_check.rs` prove products of three factors sampled at 0, 2 and 3, with increments from `from_i128`; here the products have four factors, the store bit among them, and 2 is zero |
| `BytecodeReadAddress` | (c) | `bytecode_read_raf.rs` proves a read sum with an identity term over integer table values; here the sum is four dense pairs |

Used as they are, (a): `SplitLt`, through `split_lt`; `LazyFoldedRa` and `ChunkIndexSource`; `GruenSplitEqPolynomial` with the Gruen helpers and `EqPolynomial::evals`, through `round::eq`. No (b) is requested. A generalisation that made the shared read-write kernels serve would need an increment type with its own group law and lift, a set of nodes, and the relation as a parameter, in types that are private to that crate; that is a rewrite of those kernels and not a change motivated by generality alone.

An adapter, in `jolt-rv64i-prover/src/optimized/`, supplies the source, the groups and the public tables.

- Its `CycleSource` exposes `rs1`, `rs2` and `rd` as `by_row` columns of 5 bits that are `Some` on every cycle (register 0 for an absent operand); a store column of zero bits, `Some(0)` on a store; RAM chunk digits that are `Some` on every cycle (index 0 without an access); the `KeysDiffer` flag; and the words `KeyDiffAbove` and `KeyDiff` as two trace words. `OuterF128Core` reads each of the two words up to four times per cycle, in its first pass and in the row rounds from round 2 on, so a source that evaluates the row on each read does not meet the model: the adapter's one pass over the rows stores them, 16 bytes per cycle.
- It makes one request to `ValidatedTrace::prepare`, before batch 1: the chunk groups, present; the register group, present; the store group and the `KeysDiffer` group, optional, of one column each; and whatever the cores of the kernels spec ask for. The chunk groups pass by value from `OuterF128Core::into_groups` after batch 1 to `MemoryTrace::new` (the RAM group) before batch 4 and, through `into_ram_chunks`, to the chunk products of batch 6b. A layout whose RAM chunks the chunk product does not take as a group still has the RAM group, for the memory trace.
- It supplies the initial RAM as `K` dense words, the final RAM, the public memory on the mask, and the tables it builds from public data (Non-Goals).

It receives final values and maps them to the typed outputs of `specs/rv64i-binary-protocol.md` §8:

| Member | From | Values |
|---|---|---|
| `SpartanOuterF128` | `OuterF128Core::final_values` | `az`, `bz`, `cz` at `ρ_F ++ r_1` |
| `SpartanInner` | the second values of pairs 0 and 1 | `witness_routed`, `direct_columns` |
| `RegistersReadChecking` | `register_claims`; `RegistersCore::final_values()[1]` | `rs1_ra`, `rs2_ra`, `rd_wa`; `registers_val` |
| `RamReadChecking` | `RamCore::final_values` | `ram_ra`, `ram_val` |
| `RamOutputCheck` | `RamOutputCore::final_value` | `ram_val_final` |
| `RegistersValEvaluation`, `RamValEvaluation` | `ValEvaluationCore::final_values` | `rd_wa`, `store`, `inc`; `ram_ra`, with `store` and `inc` as aliases |
| `BytecodeReadAddress` | the sum of the products of the four pairs of final values | `address_claim` |

Parked in `ProofSession` between stages: the groups of the block from batch 1 to their next takers; the `MemoryTrace` and `U` from batch 4 to batch 5; the table of `into_address_column` from batch 4 to batch 5; `R_Read` from batch 4 to batch 6a; the `row_weights` of `claims_pass` from batch 3b to batch 6a; the outputs of `fold_pass` from before batch 2 to batch 3a. One decision is left to the adapter: `source_lift` computes the lift of `Inc` at the same `r_bit` in batch 3b and drops it at the return of `claims_pass`. An adapter that keeps that table passes it as `inc` and saves `inc_lift`, 3.2 ns per cycle; the thresholds here price `inc_lift`, and a core takes the table from either. The acceptance of an adapter is that of the kernels spec, Design, "Fit with `jolt-kernels` and the protocol".

### Alternatives Considered

Counts are per cycle at `log_t = 22`; a saving is given at price point 1 and at price point 2.

- **A `MemorySource` trait.** Its methods would return, per cycle, what five digit columns and a flag already return, and an adapter would implement the same fact twice.
- **An 8-byte record per cycle, built by a pass of this crate.** The index as a `u32`, three registers and the store bit: one read per cycle in place of `d_a + 4` bytes. It is a second gather of digits the walk has validated, `4 L + 12 w` by count, 2.2 ns per cycle, and a second copy of the RAM chunks, against 20 word operations per cycle to read the index from its bytes twice.
- **Registers read through the bytecode index.** A table of three registers per bytecode row, 2 MiB at `2^20` rows, read at every cycle of six scans, against three sequential bytes. The register group costs the walk three comparisons with the row per cycle.
- **Registers through the entries of `RamCore`.** Three entries per cycle and a write on three cycles of four: `4 M + 6 A + 1 Bk` per address round against `1 M + 8 A + 2 R + 4 L + 1 Bk`, 14.5 ns against 14.4, and 84 bytes per cycle of entries against none. With 32 cells the weights are recomputed from a table of `2^i` entries; with `2^20` they are stored.
- **RAM in the scan form of the registers.** The state of a chunk would be `K` cells. Grouping by address is what bounds the state at `2^phase_bits` cells, and it needs the entries.
- **RAM with weights bucketed per cell.** Between two stores to a pair of cells the reads add their weights into a bucket, and a store multiplies the bucket out: `f Bk + s·(3 A + 1 Bk)` and a read-out of `0.375·f A` per round, against `2·f A + s Bk`. By count it saves 4.1 ns at point 1 and 0.6 at point 2 and loses 4.5 at the loaded figures. Its margin is a bucket update at its priced value; not built.
- **One phase of `a` bits.** A checkpoint of `K` cells per block, 16 MiB at `a = 20`.
- **Dense tables of `K` elements in the output check.** `κ·(24 L + 5 M + 2 A)` and 48 bytes per RAM word, against `κ·(8 L + 1 A)` and 48 bytes per word of the mask.
- **Packed rows as dense tables from round 0, or from round 2.** 96 elements per cycle, 1.5 KiB; or 24 elements per cycle, 384 bytes, and three more multiplications per cell and round.
- **A third row round from statistics.** A block of eight rows has four pairs of power columns, and its cross terms need histograms over two digits of each pair.
- **The gate products kept after round 0.** `−4 M`, 7.3 / 3.6 ns, for 16 bytes per cycle from round 2 to the last row round. Not taken while the core is under its threshold by count.
- **Five pairs in batch 6a.** `+ρ·(2 M + 2 A)`: 1.5 / 0.8 ns at `log_t = 22` and 5.9 / 3.2 at 20, and 48 MiB at `2^20` rows.
- **`Ra` recomputed for batch 5.** `+1 M + 2 L`, 2.6 / 1.7 ns, for 16 bytes per cycle less between the two batches. Taken if memory becomes the constraint.
- **The reference kernel for `SpartanInner`.** Its sum-check over 1,024 entries would do; its tables are built from one dense witness row per cycle, which is what does not scale. `PairSumCore` exists for batch 6a.
- **One split equality for the two read checks.** Both use `eq(r_3, ·)`; sharing saves 96 KiB and couples two members that are otherwise independent.

## Documentation

No change to the Jolt book. The rustdoc of `memory`, `outer_f128`, `pair_sum`, `registers`, `ram`, `ram_output` and `val_evaluation` states the definitions of Goal, the contracts of the surface table and what is not checked.

## Execution

Nine items, each one PR in `jolt-rv64i-kernels`. None changes a shared crate. Item 1 is the only one that edits files an item of the kernels spec owns, by one line or one function each; after it the items own disjoint files.

1. The bench declarations in the crate's `Cargo.toml`; the module lines in `lib.rs` and `router/mod.rs`; `memory.rs`; `split_lt` in `round/eq.rs`; the derived columns, words, `memory_columns`, `packed_columns` and `SyntheticMemory` in `synth.rs`; `tests/memory_views.rs`; `benches/memory_machinery.rs`; and stubs of every module, test and benchmark below. After `ValidatedTrace::prepare` and `BucketPlacement` of the kernels crate, which this spec uses and does not add. Accepted on "Memory views", the `MemoryError` row of "Rejected inputs" and the allocation criterion for the views.
2. `benches/memory_probe.rs`; after 1. Its report goes into the PR description.
3. `outer_f128.rs`, `benches/outer_f128.rs`, `tests/outer_f128.rs`; after 1.
4. `pair_sum.rs`, `router/columns.rs`, `benches/pair_sum.rs` (the ids `bytecode_address` and `inner`), `tests/pair_sum.rs`; after 1, and `routed_columns` after the item of the kernels spec that adds `RouterShortCore`.
5. `registers.rs`, `benches/registers.rs`, `tests/registers.rs`; after 1.
6. `ram.rs`, `benches/ram.rs`, `tests/ram.rs`; after 1.
7. `ram_output.rs`, `benches/ram_output.rs`, `tests/ram_output.rs`; after 1.
8. `val_evaluation.rs`, `benches/val_evaluation.rs`, `tests/val_evaluation.rs`; after 1 and 2.
9. `tests/memory.rs`, `benches/memory.rs`; after 4 to 8. Accepted on "Memory end to end" and the `memory` threshold.

Items 2 to 7 start together once item 1 has merged, and item 8 follows the probe. Items 5 and 6, the two cores without a counterpart in the kernels spec, are the ones whose models are least supported: each PR description reports the phases of its benchmark against the phase table above, at both sizes and both thread counts.

## References

- `specs/rv64i-binary-prover-kernels.md`: the machinery, the source traits, `ValidatedTrace::prepare` and its groups, `BucketPlacement`, the runner, the oracle, the unit table, the rule for thresholds and the invariants this spec builds on.
- `specs/rv64i-binary-prover-adapters.md`: the source of the experiment's witness, the one checked walk, and the share of the budget that precedes this spec's.
- `specs/rv64i-binary-protocol.md`: the tables and points (§3), the batches and windows (§5), the witness contract and the eight relations (§8.1–8.3, §8.10–8.14), the witness (§12).
- `specs/binary-kernel-primitives.md` (`LazyFoldedRa::try_new`, `ChunkIndexSource`, the Gruen helpers); `specs/binary-sumcheck.md` (member windows, the degree bound of a member).
- `crates/jolt-rv64i-arith/src/rows.rs` (the rows with values in `F128`).
- `crates/jolt-kernels/src/optimized/support.rs` (`SplitLt`), `crates/jolt-kernels/src/optimized/lazy_ra.rs`, and the kernels the reuse table names: `read_write.rs`, `rw_matrix.rs`, `registers_read_write/address_first.rs`, `registers_val_evaluation.rs`, `ram_val_check.rs`, `ram_output_check.rs`, `bytecode_read_raf.rs`, `spartan_outer.rs`.
- `crates/jolt-rv64i-prover/src/reference/{spartan, read_checking, val_evaluation, bytecode, views}.rs`: the dense kernels of the eight members.
