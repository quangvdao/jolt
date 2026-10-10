# Spec: Packed-Bit Prover Kernels for RV64I over the Binary Field

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-10                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

An experiment in RV64I hash-based Jolt over binary fields proves each cycle from 256 committed bits and a per-cycle R1CS witness of 1,024 bits, with every sum-check over `F128` (`specs/rv64i-binary-protocol.md`). Its reference tier holds one dense table of field elements per leaf of a relation, 16 bytes for each of those bits, and multiplies in the field where a word operation suffices; it is a test oracle, usable to about `2^10` cycles. This spec adds `jolt-rv64i-kernels` with three things. The first is the machinery every production kernel of the experiment shares for tables of bits: lifts through lookup tables, bucket accumulation in scratch that is bounded per worker, scatters partitioned by destination, round messages built in coefficient form, and the rule for when bits become field elements. The second is a benchmark harness: seeded synthetic profiles, one runner that every kernel benchmark goes through, and a probe that measures the unit costs the performance model assumes. The third is the cores of the members that take the most sum-check time by operation count, the Spartan outer sum-check over the rows with values in `F_2` and the routers, and the cores of the tail of the protocol: the two chunk products and the reduction of the committed claims of batch 6b, with the pass that builds the reduction's tables and the pass that produces the 256 column values. Each core stands alone, is tested against the definition of its sum, and is benchmarked through the runner at `2^20` and `2^22` cycles before any relation type or stage driver exists.

## Intent

### Goal

Build the packed-bit machinery, the benchmark harness and the cores of the bitwise outer sum-check, the routers, the chunk products and the reduction of committed claims with its two passes over the committed table, each emitting exactly the round polynomials of its definition within a stated number of nanoseconds per cycle.

**Notation.** `T = 2^log_t` cycles, indexed by `j`. A point lists its low variable first and every kernel binds low to high (`BindingOrder::LowToHigh`). `eq(w, x) = Π_i (1 + w_i + x_i)`. A sum-check is defined by a summand `g` built from multilinear extensions of tables over a Boolean cube; its round-`i` polynomial is `s_i(t) = Σ_x g(r_1, …, r_{i−1}, t, x)`. The committed table is a shared slice of rows `[u64; 4]`, one per cycle (`Rv64iWitness::bits`, an `Arc<[BitsRow]>`, in `specs/rv64i-binary-protocol.md` §12); `Bits[y, j]` is bit `y mod 64` of word `y / 64` of row `j`, for `y < 256`. A pass borrows the slice for the length of the call.

**Sources.** A `LaneSource` gives, per cycle, the words `A`, `B`, `C` of two lane groups (`lanes(j)`, indexed `[group][A|B|C]`) and a tail byte (`tail(j)`: bits 0–1 are `A` of rows 128 and 129, bits 2–3 `B`, bits 4–5 `C`). A `CycleSource` gives, per cycle, `trace_words()` words (`trace_word(i, j)`), a bytecode row index (`bytecode_index(j) < bytecode_rows()`) and, per digit column `c`, a digit `digit(c, j)` that is `None` or below `2^bits(c)`; per bytecode row it gives `bytecode_words()` words (`bytecode_word(i, k)`). A column declared `by_row` satisfies `digit(c, j) = row_digit(c, bytecode_index(j))`. `DigitColumns` pairs a shared `CycleSource` with a list of its columns and implements `ChunkIndexSource` of `jolt-kernels`, reporting `2^bits(c)` through `index_bound`.

Both traits require `Send + Sync + 'static`, and a core holds its source in an `Arc`: a core is `'static`, shares the source with the other cores of a proof, and copies nothing from it except into the tables invariant 3 names. An adapter implements the traits over the tables of the witness: the five base words of a cycle from `Rv64iWitness::words`, and a 16-byte row per cycle, decoded once from `Bits`, that holds the word `Inc` and every digit. The benchmarks implement them over a synthetic trace.

**Bitwise outer sum-check.** The row index is `y = p + 64·g` with position `p < 64` and group `g < 4`. `A[y, j]` is bit `p` of the group's `A` word for `g < 2`, the tail bit for `g = 2` and `p < 2`, and zero elsewhere; `B` and `C` likewise. For a point `τ` of `8 + log_t` coordinates `OuterF2Core` proves

`Σ_{y < 256, j < T} eq(τ, (y, j))·(A[y, j]·B[y, j] + C[y, j]) = 0`

in `8 + log_t` rounds of degree 3 and returns `A`, `B`, `C` at the challenge point. It is the core of the member `SpartanOuterF2`: the lanes are `LaneRows::values` of `specs/rv64i-binary-arithmetisation.md` (rows 0–63 the adder lane, 64–127 the AND lane) and the tail is rows 128 and 129.

**Routers.** A router has a bank of source bits, a selector that is a product of one-hot factors, and a public tensor; several routers share a space of `S` *slots*, the variables of one short sum-check. `specs/rv64i-binary-protocol.md` §8.4 defines the five routers of the experiment and their slots. A `RouterShape` carries the same data to a core:

- a bank: a power-of-two list of word slots, each `Trace(i)`, `Bytecode(i)`, `Bits(entries)` or `Zero`. A `Bits` slot has at most 64 entries, each `Indicator { column, value }` (the bit `digit(column, j) = Some(value)`), `DigitBit { column, bit }` (bit `bit` of the digit, zero on `None`), `One` or `Zero`. The source index of bit `i` of word slot `n` is `s = i + 64·n`;
- one to three digit columns as selector factors. A selector value `h` spells the factors' digits in mixed radix, low factor first;
- a slot map, which gives every variable of the shape a distinct slot below `S`. The six *bit* variables are slots 0–5, bit `i` at slot `i`. The *word* variables index the word slot. The *selector* variables are the index bits of each factor. The slots a shape does not use are its *idle* slots;
- an output size `2^log_outputs` and the set `route` of triples `(o, s, h)` on which the tensor `Route` is 1.

For `u ∈ {0,1}^S`, `u|_ρ` is the restriction of `u` to the slots of shape `ρ`, split into its source part `u|src` (the bit and word slots) and its selector part `u|sel`. A shape's tables are indexed by its variables in increasing slot order, the order in which the short sum-check binds them. With that:

- `Source_ρ[s, j]` is bit `i` of word slot `n` at cycle `j`; `Select_ρ[h, j]` is 1 when every factor's digit is `Some` and they spell `h`.
- `Fold_ρ[s, h] = Σ_j eq(r_cycle, j)·Source_ρ[s, j]·Select_ρ[h, j]`, for every source index and every selector value; `W_ρ[s, h] = Σ_o eq(w, o)·Route_ρ[o, s, h]`; and `Idle_ρ(u) = Π_k (1 + u_k)` over the idle slots `k` of `ρ`.
- The **short core** is one sum-check for all shapes. It proves `Σ_{u ∈ {0,1}^S} Σ_ρ Idle_ρ(u)·W_ρ[u|_ρ]·Fold_ρ[u|_ρ]` in `S` rounds of degree 2, slot 0 first, and returns, at its challenge point `x`, each `Fold_ρ(x|_ρ)` and each `Idle_ρ(x)·W_ρ(x|_ρ)`. `r_bit = x[0..6)` is common to all shapes.
- The **cycle core** proves, for each shape, `Σ_j eq(r_cycle, j)·Source_ρ(x|src, j)·Π_f Sel_f(j) = Fold_ρ(x|_ρ)`, where `Sel_f(j) = eq(x|_f, digit(c_f, j))`, or zero on `None`, and `x|_f` is `x` on the factor's slots. Each shape is one batch member of `log_t` rounds and degree `2 + F` for `F` factors; the member returns `Source_ρ(x|src, r')` and each `Sel_f(r')` at the batch's challenge point `r'`.

The extension `Fold_ρ(x|_ρ)` depends on every entry of the table, so `Fold_ρ` is built in full. The support of `Route` enters in one place, the table `W_ρ`.

The experiment has `S = 17` and five shapes, which serve the member `RouterShort` and the five `RouterCycle` members of `specs/rv64i-binary-protocol.md` §8.4–8.9. That section fixes their banks and tensors, and its slot table is the definition; the table below restates it. In the `Variant` shape, word slots 0–8 are the nine words of §8.4 (`Trace` for `Rs1Value`, `Rs2Value`, `RdPreValue`, `NextPC` and `Inc`; `Bytecode` for `Imm`, `FallThroughPC`, `PCPlusImm` and `PC`), slots 9 and 10 are `Bits` slots that hold the indicator and flag entries from source index 576 and then `One`, and slots 11–15 are `Zero`. In `Compare`, word slot 3 is a `Bits` slot whose entry 0 is `One`. The five values `Fold_ρ(x|_ρ)` are the values sent between the two batches.

| Shape | Word slots | Selector factors (index bits) | Word | Selector | Idle | Fold table |
|---|---:|---|---|---|---|---:|
| `Variant` | 16 | 1 (6) | 6–9 | 11–16 | 10 | `2^16` |
| `Shift` | 1 | 3 (3, 3, 3) | | 6–8, 9–11, 12–14 | 15–16 | `2^15` |
| `Memory` | 2 | 2 (3, 4) | 12 | 6–8, 13–16 | 9–11 | `2^14` |
| `Compare` | 4 | 3 (3, 3, 3) | 12–13 | 6–8, 9–11, 14–16 | | `2^17` |
| `Branch` | 2 | 2 (0, 0) | 12 | | 6–11, 13–16 | `2^7` |

The bit variables are slots 0–5 in every shape. The five tables have 245,888 entries.

**Chunk products.** For `d` digit columns (`1 ≤ d ≤ 7`) whose digits are all `Some`, a point `a_c` of `bits(c)` coordinates per column and a weight `W` over the cycles, `ChunkProductCore` proves

`Σ_j W[j]·Π_{c<d} eq(a_c, digit(c, j))`

in `log_t` rounds of degree `d + 1` and returns `W(r')` and each `Ra_c(r') = Σ_j eq(r', j)·eq(a_c, digit(c, j))`. The weight is a `ChunkWeight`. `Dense(table)` is a table of `T` elements; `combined_weight(log_t, terms)` builds `W[j] = Σ_i κ_i·E_i(j)` from terms `Eq { coefficient, point }`, with `E(j) = eq(point, j)`, and `Next { coefficient, point }`, with `E(j) = eq(point, j − 1)` for `j ≥ 1` and `E(0) = 0`. `EqTerms(terms)` is a list of `(coefficient, point, claim)` that stands for `Σ_i κ_i·eq(point_i, j)` without a table, where `claim_i` is the sum under the term's own weight. The core serves `BytecodeReadCycle` (five terms, one of them `Next`) and `RamRaProduct` (two terms).

**Reduction of committed claims.** For weight vectors `L_i ∈ F128^256`, a table builder returns the tables `G_i[j] = Σ_y L_i[y]·Bits[y, j]`: `g_pass_digits(source, map, weights)` from a `CycleSource`, the default, and `g_pass_bytes(rows, weights)` from the rows. A *leg* is a table, a cycle point `t`, a coefficient `κ` and a claim. `ReductionCore` proves `Σ_legs κ·Σ_j eq(t, j)·G[j]` in `log_t` rounds of degree 2 and returns each table at `r'`. `column_pass(rows, r)` returns `C[y] = Σ_j eq(r, j)·Bits[y, j]` for the 256 columns, so that `G_i(r') = Σ_y L_i[y]·C[y]`. They serve `BitsReduction`, whose claims at three cycle points have weights that are non-zero on 21, 20 and 8 byte positions of a row, and the vector `C` of batch 6b.

**Public surface.**

| Item | Where | Contract |
|---|---|---|
| `F128::mul_x` | `jolt-field`, `binary/f128.rs` | `a.mul_x() == a * F128::from_raw(2)`; a shift and a conditional XOR |
| `gruen_recover_q_one`, `gruen_mul_linear` and the methods `GruenSplitEqPolynomial::{recover_q_one, round_poly_from_q_coeffs}` | `jolt-poly`, `split_eq.rs` | Given `(l(0), l(1))`, `q(0)` and the claim, return `q(1)`, calling a supplied closure when `l(1)` has no inverse and returning the actual sum as the error when it disagrees with the claim; return the coefficients of `l·q` from those of `q`. Any characteristic |
| `ChunkIndexSource` with `num_polys`, `cycles`, `index` and the provided `index_bound` | `jolt-kernels`, `optimized/lazy_ra.rs`; the module becomes `pub` | `index_bound(i)` is `None` by default; `Some(b)` states that every digit of column `i` is below `b` |
| `LazyFoldedRa` with `try_new`, `num_polys`, `value`, `lo_hi`, `lo_hi_all`, `final_values`, `bind`; `LazyRaError` | same file | `try_new(tables, source)` returns an error when the number of tables differs from `source.num_polys()`, when a table's length or the cycle count is not a power of two, or when a digit is not below its table's length; it reads `index_bound` and scans the source for a column that gives none. `new` stays `pub(crate)`. The payloads of the two variants move into types with private fields |
| `LaneSource`, `CycleSource`, `DigitColumns` | `source` | Section "Sources" |
| `WordLift`, `NibbleLift` | `packed::lift` | `lift(w) = Σ_i w[i]·ω_i` for the weights given at construction |
| `NibbleBuckets`, `ByteBuckets`, `DigitHistogram` | `packed::buckets` | `xor(position, value, e)`, `merge`, then per-bit or per-digit sums of `e` |
| `ScratchPool` | `packed::pool` | `ScratchPool::new(len)`; `take()` lends an array of `len` elements, zeroed when first created; `merge()` returns the XOR of the arrays. At most `rayon::current_num_threads()` arrays exist |
| `ScatterPlan` | `packed::scatter` | `ScatterPlan::new(source)` counts the cycles per chunk of cycles and per range of bytecode rows; `scatter(weight)` returns the table `out[bytecode_index(j)] ^= weight(j)` of `bytecode_rows()` elements |
| `moebius`, `gather` | `packed::bits` | XOR Möbius transform of a word along its low position bits; compaction of a word with one significant bit per `2^m` positions |
| `coefficients_from_nodes`, `quadratic`, `eval_at_node` | `round` | Section "Round messages" |
| `OuterF2Options`, `OuterF2Core` | `outer_f2` | `OuterF2Options { monomial_rounds, nibble_round_2, folded_group_weights }`, by default 3, `false`, `false`; `OuterF2Core::new(source, τ, options)`; `ProveRounds<F128>`; `final_values() -> [F128; 3]`; `check_rows(source)` |
| `RouterShape`, `selector_counts`, `FoldLayout`, `fold_pass`, `RouterShortCore`, `source_lift`, `RoutersCycleCore`, `RouterCycleMember`, `claims_pass` | `router` | Section "Routers"; `RouterShape::new` validates the bank, the factors, the slot map and `route` |
| `combined_weight`, `ChunkWeight`, `ChunkProductCore` | `chunk_product` | `ChunkProductCore::new(columns, points, weight)`; `ProveRounds<F128>`; `final_values()` |
| `g_pass_digits`, `g_pass_bytes`, `ColumnMap`, `ReductionCore` | `reduction` | `ReductionCore::new(tables, legs)`; `ProveRounds<F128>`; `final_values()`; `check_claims()` |
| `column_pass` | `column_pass` | `column_pass(rows, r) -> [F128; 256]` |
| `SynthProfile`, `SyntheticTrace` | `synth`, feature `test-utils` | Seeded trace implementing `LaneSource` and `CycleSource`, with `rows()` |
| `round_polynomial`, `mle_at` | `oracle`, feature `test-utils` | The definitions above by direct summation |

Every constructor and pass of the table checks the lengths it is given (points against `log_t` and `bits(c)`, tables against `T`, weight vectors against 256, the number of points against the number of columns) and returns its module's error. A digit at or above `2^bits(c)` makes the pass that reads it return an error naming the column and the cycle.

### Invariants

1. **Definitional messages.** Every message of a core is the round polynomial of its definition, and every final value is the multilinear extension of its table at the challenge point. A table that a definition names is built on its whole domain: `fold_pass` computes `Fold_ρ[s, h]` for every `s` and `h` and does not read `route`.
2. **Characteristic 2.** No kernel samples a round polynomial at an integer, halves, scales by a power of two, or calls `from_u64`, `from_i128` or an integer-scalar `fmadd_*`. Constants are `F128::from_raw` values; multiplication by `F128::from_raw(2)` is `mul_x`.
3. **Representation.** Inputs are shared packed words; no kernel stores a field element per bit. A table of `F128` exists only where a variable ranging over bits has been bound or summed out: the outer core holds 6 elements and one byte per cycle from round 7; `source_lift` holds one element per trace word and cycle until `claims_pass`, and one per shape and cycle for `Source_ρ(x|src, ·)`; a selector or chunk column becomes dense at `T/16` after its fourth bind; a chunk product holds its weight when the weight is `Dense`; the reduction holds one element per table and cycle.
4. **Allocation and scratch.** No heap allocation per cycle, per pair or per chunk. Tables are allocated at construction or at one of the phase changes named in the Design and bound between two alternating buffers. Bucket arrays and histograms come from a `ScratchPool`, so a pass holds at most one layout per worker thread, whatever its number of chunks; a scatter holds one buffer of `T` pairs and no table per worker. Every table of `T` or `bytecode_rows()` elements has one owner and the drop point the Performance section lists.
5. **Determinism.** Messages and final values do not depend on the thread count, on the chunk length or on the order in which a batch visits its members. Chunks are unions of whole split-equality blocks; partial results combine by XOR (`F128Accumulator::merge`).
6. **Honest inputs.** Three cores take a precondition they do not check. `OuterF2Core` requires `C = A & B` on every lane word and on the tail bits. `ReductionCore` requires each leg's claim to equal `Σ_j eq(t, j)·G[j]`. `ChunkProductCore` with `EqTerms` requires each term's claim to equal the sum under that term's weight. On an input that violates one, the messages belong to another polynomial and the verifier rejects at its final check. `check_rows` reports the first violating cycle and `check_claims` the first violating leg; they are diagnostics a caller may run, and no acceptance criterion rests on them.
7. **Tables in cache.** The lookup tables one pass reads per thread total at most 96 KiB, except: the buckets of `fold_pass` (10.4 MiB per worker in the default layout); the tables indexed by bytecode row; the byte buckets of `column_pass` (128 KiB per worker); the window tables of `g_pass_bytes`, 4 KiB per byte position with a non-zero weight; and the window tables of `OuterF2Core` under `folded_group_weights` (112 KiB).
8. **Total functions.** Constructors and passes validate lengths, slots, indices and digit ranges and return a `thiserror` enum, one per module, naming the offending length, index, slot, column, cycle or digit; nothing panics on malformed input. The crate has `#![forbid(unsafe_code)]`.
9. **Upstream changes preserve behaviour.** The additions to `jolt-field` and `jolt-poly` change no existing function body or signature. In `jolt-kernels` the change to `optimized/lazy_ra.rs` is visibility, the provided method `index_bound`, the constructor `try_new` with `LazyRaError`, and the move of the enum's payloads into two types with private fields; no existing method returns a different value.
10. **Boundary.** The crate depends on `jolt-field` (feature `binary`), `jolt-poly`, `jolt-sumcheck`, `jolt-kernels`, `rayon` and `thiserror`, and on `rand_chacha` under `test-utils`. It depends on no relation type and on no arithmetisation crate; a row of the committed table is `[u64; 4]`.
11. **No protocol change.** No core, option or variant of this spec changes a message, a round count, a member or the transcript of `specs/rv64i-binary-protocol.md`.

No `jolt-eval` invariant changes: nothing here is reachable from the existing prover.

### Non-Goals

Implementations of `SumcheckKernel` and `PrepareKernel` for the relations of `specs/rv64i-binary-protocol.md`, and their acceptance against the reference kernels of that spec, which the Design states; the decode of the committed table into digits; the routing tensors and the public addends of a router's outputs; the cores of `SpartanOuterF128`, `SpartanInner`, the three members of batch 4, the two of batch 5 and `BytecodeReadAddress`, which follow on the same machinery and runner; a univariate-skip first round; the three prover-only variants of the routers listed under Alternatives; any change to `prove_batch`; witness generation; any commitment.

## Evaluation

### Acceptance Criteria

"By summation" means computed from the definitions in Goal with `oracle`, on tables built in the test. "Rejected" means rejected by the verifier of `jolt-sumcheck`, at whichever of its checks fails. A core takes its value at 1 from the claim, so a proof made for a wrong claim passes every round check and is rejected at the final one.

- [ ] **Field and split equality.** `mul_x` equals multiplication by `F128::from_raw(2)` on the 128 basis elements and 10,000 seeded values. For `Fr` and for `F128`, a seeded linear `l` and `q` of degree 1 to 5: `round_poly_from_q_coeffs` equals the product of the two `UnivariatePoly`s, and `recover_q_one` returns `q(1)`, including an equality coordinate of 0 (the closure is called; a claim off by one returns the actual sum), of 1, and a zero scalar. The existing tests of `split_eq.rs` pass unchanged.
- [ ] **Lazy columns.** `LazyFoldedRa::try_new` returns each of its errors on an input built for it: one table too few, a table of 12 entries, a source of 12 cycles, and a digit equal to its table's length, the last once through `index_bound` and once through the scan. The existing tests of `jolt-kernels` pass unchanged.
- [ ] **Lifts, buckets, nodes.** `WordLift` and `NibbleLift` equal `Σ_i w[i]·ω_i` on the unit words, `u64::MAX` and 10,000 seeded words. `NibbleBuckets`, `ByteBuckets` and `DigitHistogram` equal the per-bit and per-digit sums of a seeded sequence, and two merged halves equal the whole. For degrees 2 to 8, `coefficients_from_nodes` recovers a seeded polynomial from its leading coefficient and its values at 0, 1 and the nodes.
- [ ] **Scratch and scatter.** A pass of 64 chunks through a `ScratchPool` on a pool of 12 threads creates at most 12 arrays, and `merge` equals the XOR of the same updates applied by one thread. `ScatterPlan::scatter` equals `out[bytecode_index(j)] ^= weight(j)` applied in cycle order: with 16 visited rows, with every row visited, with one bytecode row and with `2^20`.
- [ ] **Outer messages.** For `log_t` from 3 to 8, seeded lanes and tails with satisfied rows, a seeded `τ` and seeded challenges, `monomial_rounds` in 3, 4, 5, 6 and both settings of each table option of `OuterF2Options`: every round's coefficients equal the round polynomial by summation, `final_values` equal `A`, `B`, `C` at the challenge point by summation, and the one-member batch proof verifies.
- [ ] **Outer, degenerate points.** The same with every coordinate of `τ` drawn from `{0, 1}`: all zero, all one, and alternating.
- [ ] **Outer rejection.** With one bit of `C` changed, in a lane word and in the tail: the proof is rejected at the check of `eq(τ, r)·(A(r)·B(r) + C(r))` against the final claim, and `check_rows` names the cycle. A proof with one coefficient changed is rejected, as is a proof made for an input claim of 1.
- [ ] **Complete fold.** For `log_t` from 3 to 8 and five shapes with the word slots, factors and slot maps of the table in Goal, mixing `Trace`, `Bytecode`, `Bits` and `Zero` slots, `Indicator`, `DigitBit` and `One` entries, `by_row` and per-cycle columns and `None` digits, on traces in which every bit of every word is set at some cycle under every selector value that occurs: every entry `Fold_ρ[s, h]` in the shape's slot order, `ra_fold` and the histograms equal their definitions by summation. This holds with every `route` set empty and with seeded `route` sets, for `r_cycle` seeded and in `{0, 1}^log_t`, and for a `FoldLayout` with no selector value by byte, with some, and with all.
- [ ] **Complete fold, smallest case.** `S = 6`; one shape with one `Trace` word slot, one factor of zero index bits that is present on every cycle, one output and `route = {(0, 0, 0)}`; eight cycles, the word of cycle 0 equal to 2 and the others seeded; `r_cycle = (0, 0, 0)`. Then `W = [1, 0, 0, …]`, `Fold = [0, 1, 0, …]`, the input claim is 0, and the first message of the short core has the coefficients `[0, 1, 1]`: it is `X + X^2`, which is `F128::from_raw(6)` at `F128::from_raw(2)`. A fold that kept only the entries `route` reads would give the zero polynomial.
- [ ] **Short core.** For the five shapes and seeded `route` sets: every round's coefficients equal the round polynomial of the 17-variable sum by summation; the final values equal `Fold_ρ(x|_ρ)` and `Idle_ρ(x)·W_ρ(x|_ρ)` by summation; the one-member proof verifies from the input claim `Σ_ρ Σ_{s,h} W_ρ[s, h]·Fold_ρ[s, h]` by summation. `RouterShape::new` rejects, with an error naming the slot, a slot map that repeats a slot, leaves a bit variable off slots 0–5 or names a slot at or above `S`; and, with an error naming the entry, a `Bits` slot of more than 64 entries, a bank whose length is not a power of two, and a `route` triple outside the output size, the bank or the selector range.
- [ ] **Cycle core.** For those shapes: `source_lift` equals `Source_ρ(x|src, ·)` by summation; the five members, proved as one batch through `prove_batch`, each emit coefficients equal to their round polynomial by summation in every round, under `SequentialRounds` and under a scheduler written in the test that visits the members in reverse order; each member's input claim is its short-core value `Fold_ρ(x|_ρ)`; the final values and every output of `claims_pass` equal their definitions by summation; the batch proof verifies. A proof with one coefficient changed is rejected.
- [ ] **Chunk products.** For `log_t` from 3 to 8, `d` from 1 to 7 with chunks of 4 bits and a top chunk of 1 to 4 bits, and weights of one, two and five terms, one of them `Next` and one with a point in `{0, 1}^log_t`: `combined_weight` equals its definition by summation; every round's coefficients equal the round polynomial by summation; the final values equal `W(r')` and `Ra_c(r')` by summation; the one-member proof verifies and a proof with one coefficient changed is rejected. The same holds for weights of one and two `Eq` terms given as `EqTerms`; with one term's claim changed, the proof is rejected.
- [ ] **Table passes.** For seeded rows: `g_pass_bytes` equals `Σ_y L[y]·Bits[y, j]` for weights that are non-zero on all 256 columns and for weights supported on columns 64–228, on 0–63 with 139–230, and on 0–63. On rows whose indicator ranges are one-hot or zero and a `ColumnMap` of those ranges, `g_pass_digits` equals the same sums by summation; a weight that is non-zero on a column the map does not cover is rejected with an error naming the column. `column_pass` equals `Σ_j eq(r, j)·Bits[y, j]` for all 256 columns, for seeded `r` and for `r` in `{0, 1}^log_t`.
- [ ] **Reduction.** With three legs on three tables, and with four legs on three tables of which two legs share a table: every round's coefficients equal the round polynomial by summation and the final values equal the tables at `r'` by summation. With one leg's claim changed, the proof is rejected; `check_claims` names the leg.
- [ ] **Tail end to end.** At `log_t = 8` on `SyntheticTrace`: `g_pass_digits`, two `combined_weight` builds, a batch of two `ChunkProductCore`s (five chunks each, weights of five and two terms) and one `ReductionCore` (three legs) through `prove_batch`, then `column_pass(rows, r')`. With `C` its result: each chunk value equals `eq(a_c, 0) + Σ_{k ≥ 1} (eq(a_c, k) + eq(a_c, 0))·C[start_c + k − 1]`; each `G_i(r')` equals `Σ_y L_i[y]·C[y]`; the batch's final claim equals the combination of the two weights at `r'` by summation, those chunk values and those sums under the batching coefficients; and for a seeded `ρ` of 8 coordinates, `Σ_y eq(ρ, y)·C[y]` equals the extension of `Bits` at `(ρ, r')` by summation.
- [ ] **Determinism.** At `log_t = 12`, the outer core, the router pipeline and the tail produce coefficients equal to the round polynomials by summation under a pool of 1 thread and under a pool of 12, and under two chunk lengths.
- [ ] **Allocation and scratch.** Under a counting allocator and a one-thread pool, at `log_t = 8` and at `log_t = 14`: a core allocates at most `16·rounds + 64` times between the return of `new` and the return of `finish_rounds`, and each pass allocates at most 256 times. Under a pool of 12 threads at `log_t = 14`: the peak live bytes during `fold_pass` and during `column_pass` are at most the pass's outputs, 12 bucket layouts and one scatter buffer, and at the return of every pass the live bytes are those at its start plus its outputs.
- [ ] **Runner.** `run_core` on the dense-product example prints nanoseconds per cycle for construction, rounds, finish and extraction, and the peak and final live bytes, at `log_t` 20 and 22 on 1 and 12 threads.
- [ ] **Performance.** Every benchmark of the Performance section is reported through the runner, at both sizes and both thread counts, against its row. A result above its threshold at `log_t = 22` fails the PR that adds or changes the kernel.
- [ ] **Gates.** `cargo clippy --all-targets -- -D warnings` and `cargo nextest run --cargo-quiet` pass for `jolt-rv64i-kernels`, `jolt-field --features binary`, `jolt-poly` and `jolt-kernels`; `cargo fmt --check` passes.

### Testing Strategy

The ground truth is `oracle`: it expands each leaf to a dense vector, binds by `f(r) = f_0 + r·(f_0 + f_1)`, evaluates the summand at `d + 1` distinct field elements and interpolates with `interpolate_nodes_to_coeffs` (`jolt-poly/src/lagrange.rs`). It shares no table, lift or round assembly with the cores. Verification in the tests goes through `prove_batch` and the verifier of `jolt-sumcheck` over `BooleanHypercube`.

Each test asserts an algebraic identity against that ground truth or a rejection by the verifier, clauses (c) and (e) of `.claude/skills/test-policy/SKILL.md`; none compares one configuration of a core with another, and every option and layout is tested against the same summation. A rejection test asserts the verifier's result; it does not assert the round at which the proof fails. Seeds are fixed. The existing tests of `jolt-field`, `jolt-poly`, `jolt-sumcheck` and `jolt-kernels` pass unchanged. The `host` and `zk` modes do not apply.

Rustdoc on the public items states the contracts of the surface table and, on `OuterF2Core::new`, `ReductionCore::new` and `ChunkWeight::EqTerms`, invariant 6 in those terms: required of the caller, not checked, detected by the verifier. Comments in the round loops are limited to what `.claude/skills/comment-policy/SKILL.md` keeps: the identity a formula implements where the code cannot show it, and the coupling of a variable order to the definitions in this spec.

### Performance

Benchmarks run on an Apple M4 Max with `-C target-cpu=native`, on one thread and on a `rayon` pool of 12 threads, at `log_t = 20` and `log_t = 22`, with `2^20` bytecode rows and `2^20` RAM words at both lengths. Every benchmark goes through one runner, `run_core` in `benches/support/`. It takes a constructor and a profile and times four phases: construction with the passes that feed it, the rounds under challenges from a fixed seed, `finish_rounds`, and the extraction of the final values with the passes that follow the rounds. Under a counting allocator it reports the peak and the final live bytes. Ids are `<bench>/<profile>/<log_t>/<threads>`.

`SyntheticTrace` is a function of a `SynthProfile` and a seed (`ChaCha20Rng`). Its rows carry the chunk indicators at stated column ranges, its digits agree with them, and its lanes satisfy `C = A & B`.

| Profile | Cycles | Visited bytecode rows |
|---|---|---|
| `local` | 25% loads, 10% stores, 5% shifts, 8% comparisons, 12% conditional branches of which half are taken, the rest ALU; three quarters of the cycles write a register; RAM accesses in a working set of `2^12` words, 90% of them in a window of 64 words | `2^16` of `2^20` |
| `all_rows` | as `local` | all `2^20` |
| `uniform_digits` | no trace; every digit of every column uniform and independent | none |

On `local` and `all_rows` a cycle has a shift kind with probability 0.05, a RAM access with 0.35, a key kind with 0.20 and is a taken branch with 0.06. `fold_pass` and `routers` run on `all_rows`; `outer_f2`, `reduction`, `column_pass` and `tail` on `local`; `chunk_product` on `uniform_digits`. `routers` covers `fold_pass`, the short core, `source_lift`, the five cycle members and `claims_pass`. `chunk_product` covers `combined_weight` with five terms and one core of five chunks. `reduction` covers `g_pass_digits` on the three supports of the acceptance criteria and a core of three legs. `tail` covers the sequence of the end-to-end criterion.

**Unit costs.** Operation counts are written in these symbols, in nanoseconds:

| Symbol | Operation | Point 1 | Point 2 | Source |
|---|---|---:|---:|---|
| M | `F128` multiplication | 1.83 | 0.9 | `cargo bench -p jolt-field --features binary --bench binary_kernels` |
| A | `F128Accumulator::fmadd`, in a chain of 1,024 terms | 1.10 | 0.7 | the same bench |
| R | `F128Accumulator::reduce` | 0.7 | 0.2 | M − A |
| L | lookup of a 16-byte entry and its XOR, table set in L1 | 0.4 | 0.4 | estimate |
| Bk | bucket load-XOR-store | 0.6 | 0.6 | estimate |
| `sct` | one cycle of a partitioned scatter | 1.4 | 1.4 | estimate |
| `mrg` | one element of a zero-fill, a merge or a bucket read-out | 0.3 | 0.3 | estimate |
| X | `mul_x` | 0.2 | 0.2 | estimate |
| w | one 64-bit word operation | 0.05 | 0.05 | estimate |

Point 1 is M and A measured on a quiet machine with the scalar field code. Point 2 is M and A with the field code that keeps products, reductions and accumulator state in vector registers, taken on a loaded machine; it is provisional until it is re-measured on a quiet one, and no figure derived from it is a measurement. The estimates are replaced by what `probe` and `machinery` report.

**Requirements.** `ρ` is the number of visited bytecode rows per cycle: 1/4 at `log_t = 22` and 1 at `log_t = 20` on `all_rows`. Counts are per cycle, with the pairs of all cycle rounds summed.

| Benchmark | Operations per cycle |
|---|---|
| `fold_pass/all_rows` | `1 M + 111.5 Bk + 1 sct + ρ·(64 Bk + 5 mrg)`, and `2.2·10^6 mrg` once |
| `routers/all_rows` | `38.5 M + 33 A + 4 R + 99 L + 111.5 Bk + 18 X + 2 sct + ρ·(64 Bk + 5 mrg + 32 L + 4 A + 13 ns)`, and `2.2·10^6 mrg + 245,888·(3 M + 2 A + 2.3 ns)` once |
| `outer_f2/local` | `15 M + 51 A + 12 R + 253 L + 1 Bk + 410 w` |
| `chunk_product/uniform_digits` | `16.3 M + 10 A + 1 R + 25 L + 18 X` |
| `reduction/local` | `3 M + 3 A + 37 L + 0.5 ns` |
| `column_pass/local` | `1 M + 32 Bk` |
| `tail/local` | `36.6 M + 21 A + 2 R + 87 L + 32 Bk + 36 X + 0.5 ns` |

| Benchmark | Model at point 1: `22`, `20` | Threshold: `22`, `20` | Model at point 2: `22`, `20` | Threshold at point 2: `22`, `20` |
|---|---:|---:|---:|---:|
| `fold_pass/all_rows` | 80.3, 110.7 | 100, 138 | 79.3, 109.7 | 99, 137 |
| `routers/all_rows` | 240.7, 295.5 | 301, 369 | 189.1, 242.1 | 236, 303 |
| `outer_f2/local` | 214.3 | 268 | 173.9 | 217 |
| `chunk_product/uniform_digits` | 55.1 | 69 | 35.5 | 44 |
| `reduction/local` | 24.1 | 30 | 20.1 | 25 |
| `column_pass/local` | 21.0 | 26 | 20.1 | 25 |
| `tail/local` | 153.2 | 191 | 109.7 | 137 |

A threshold is single-thread nanoseconds per cycle, the model times 1.25 to the nearest nanosecond. The thresholds of point 1 are in force. The PR that records a quiet measurement of M and A recomputes both the models and the thresholds from the operations column at the measured prices; the last column is what they become if point 2 holds. On 12 threads at `log_t = 22` the requirement is wall time per cycle, the single-thread threshold divided by 9.6: 31 ns for `routers`, 28 for `outer_f2` and 20 for `tail`.

| Benchmark id | Requirement |
|---|---|
| `machinery/lift` | at most 4.8 ns per `u64` lifted with one `WordLift` |
| `machinery/bucket` | at most 0.9 ns per `NibbleBuckets::xor` into the 10.4 MiB layout of `fold_pass`, with the selector values of `all_rows` |
| `machinery/scatter` | at most 2.1 ns per cycle for one scatter into `2^20` rows, all visited |
| `machinery/merge` | at most 0.45 ns per element for the zero-fill and the tree merge of the 10.4 MiB layout of `fold_pass` |
| `machinery/mul_x` | at most 0.3 ns |

`probe` has no requirement. It runs before the packed machinery exists and reports, on streams of `2^22` rows of the `local` and `all_rows` profiles and on 1 and 12 threads: L for table sets of 5, 32, 64, 69, 96 and 196 KiB under the access patterns of the outer core, `source_lift` and the two table builders; Bk for the 128 KiB buckets of `column_pass` and for the three layouts of `fold_pass`, with the share of cycles on the byte-bucketed selector values varied; a scatter into `2^20` rows with `2^16` and with `2^20` of them visited, as a direct scatter, with one table per worker and partitioned; `fmadd` chains of 1, 2, 4, 8 and 20 terms with their reduction; and the zero-fill and tree merge of a 10 MiB array. Its report replaces the estimates of the unit table and fixes the defaults named below.

**By phase**, at `log_t = 22`, with the values at `log_t = 20` in brackets where they differ:

| Outer phase | Operations | Point 1 | Point 2 |
|---|---|---:|---:|
| Round 1 | `8 L + 1 A + 40 w` | 6.3 | 5.9 |
| Round 2 | `16 L + 2 A + 130 w` | 15.1 | 14.3 |
| Round 3 | `56 L + 2 A + 240 w` | 36.6 | 35.8 |
| Rounds 4, 5, 6 | `40 L + 4 R` each, and `20 A`, `12 A`, `8 A` | 40.8, 32.0, 27.6 | 30.8, 25.2, 22.4 |
| Tail histogram, materialisation | `1 Bk + 48 L` | 19.8 | 19.8 |
| Rounds 7 and 8 | `5 L + 4 A + 10 M` | 24.7 | 13.8 |
| Cycle rounds | `2 A + 5 M` | 11.4 | 5.9 |
| **Total** | | **214.3** | **173.9** |

| Router phase | Operations | Point 1 | Point 2 |
|---|---|---:|---:|
| `fold_pass` | `1 M + 111.5 Bk + 1 sct + ρ·(64 Bk + 5 mrg)`; the 111.5 are 88 for the 16-slot shape (80 nibbles of five trace words, 7 digit columns, 1 flag column) and `16·0.05 + 32·0.35 + 48·0.20 + 32·0.06` for the other four | 80.3 [110.7] | 79.3 [109.7] |
| Short rounds | 245,888 table entries, `3 M + 2 A + 2.3 ns` each | 0.6 [2.3] | 0.4 [1.5] |
| `source_lift` | `59 L + 12 A + 4 R`, and 13 ns per visited row | 42.9 [52.6] | 36.1 [45.8] |
| Cycle rounds, products | `36 M + 16 A + 18 X` | 87.1 | 47.2 |
| Cycle rounds, 8 selector columns | `40 L + 0.5 M` | 16.9 | 16.5 |
| `claims_pass` | `1 M + 5 A + 1 sct + ρ·(32 L + 4 A)` | 13.0 [25.9] | 9.7 [21.4] |
| **Total** | | **240.7** [295.5] | **189.1** [242.1] |

| Tail phase | Operations | Point 1 | Point 2 |
|---|---|---:|---:|
| `combined_weight`, five terms and two terms | `4 A + 1 R` and `2 A + 1 R` | 5.1, 2.9 | 3.0, 1.6 |
| Chunk product, per member: column gathers | `25 L` | 10.0 | 10.0 |
| Chunk product, per member: rounds | `16.3 M + 6 A + 18 X` | 40.0 | 22.5 |
| `g_pass_digits` | `37 L`; 16 bytes read and 48 written, 0.5 ns | 15.3 | 15.3 |
| Reduction rounds, three legs | `3 A + 3 M` | 8.8 | 4.8 |
| `column_pass` | `1 M + 32 Bk` | 21.0 | 20.1 |
| **Total** | | **153.2** | **109.7** |

Between the two points only M, A and R move. The lookups, bucket XORs, scatters, shifts and word operations are 129.6 ns of `routers`, 122.3 of `outer_f2` and 61.7 of `tail`: 52% of the three models at point 1 and 66% at point 2, and all of it is estimated. Inside `routers` the largest phase is the cycle rounds at point 1 (104.0 against 80.3 for `fold_pass`) and `fold_pass` at point 2 (79.3 against 63.7).

**Options and variants.** Each is a setting of the kernel named, proves the same sum and is reported by that kernel's benchmark next to the default. The saving is against the default, at `log_t = 22`; a negative saving is a cost.

| Setting | Operations against the default | Saving, point 1 / point 2 | Memory |
|---|---|---:|---|
| `fold_pass`: byte buckets for the 8 most frequent selector values of the 16-slot shape | `−40·f Bk`, with `f` the share of cycles on those values; 28 at `f = 0.7` | 16.8 / 16.8 | 11.5 MiB of buckets per worker against 10.4 |
| `fold_pass`: byte buckets for all 64 | `−40 Bk`, and `−32 Bk` per visited row | 28.8 / 28.8 | 19.1 MiB per worker, and 8 MiB against 1 for the pass over the rows |
| `outer_f2`: nibble tables in round 2 | `+16 L` | −6.4 / −6.4 | 8 KiB of tables against 64 |
| `outer_f2`: group weights folded into the window tables of rounds 4 to 6 | `−6 R − 6 A` | 10.8 / 5.4 | 112 KiB of tables against 64 |
| `chunk_product`: `EqTerms` for a weight of two terms | `+2 A − 1 R − 6 X` | −0.3 / 0.0 | 24 bytes per cycle less |
| `reduction`: `g_pass_bytes` | `+12 L`, and 32 bytes read per cycle against 16 | −5.3 / −5.3 | 196 KiB of tables, of which 74 to 92 are read, against 69 |
| `reduction_shared`: four legs, the two legs on columns 0–63 sharing one table that is the `WordLift` of that word | `−16 L + 2 A` | 4.2 / 5.0 | 24 bytes per cycle more, held from `source_lift` |

The defaults are the layout with 8 byte-bucketed selector values, byte tables in round 2, unfolded group weights, `Dense` weights and `g_pass_digits`. `fold_pass` and `routers` run in all three layouts. Their models price the layout with no byte buckets, whose unit cost is least in doubt, and their thresholds apply to the layout that ships as the default. A default changes when `probe` or the kernel's own benchmark shows another setting faster by more than the spread of repeated runs; a table builder that does not win is deleted before the adapter of `BitsReduction` is written.

**What the models leave out.** A model prices every lookup as a hit in L1 and every stream as free, and it is conditional on the estimates of the unit table. It does not price: the memory traffic of passes over tables of `T` elements; bucket and table sets outside L1, where one Bk stands for arrays from 8 KiB to 19 MiB; the zero-fill, merge and read-out of the fold layout on 12 workers, which by count is 1.2 ns per cycle at `log_t = 22` and 4.9 at `log_t = 20` against 0.16 and 0.6 on one; `fmadd` chains shorter than the 1,024 terms A was measured on, which is every chain of rounds 4 to 6 of the outer core, of `source_lift` and of `combined_weight`; allocation and first touch; and parallel efficiency, of which no kernel has a measurement. The thresholds are what a PR is held to. The models are not a claim that a kernel meets them.

The prover of the experiment has 2,100 ns of single-core work per cycle from witness generation to the opening proof, and a wall-time target of 918 ms at `2^22` cycles on 12 threads, which is that budget at 9.6 effective cores. The thresholds of the three pipelines of this spec sum to 760 ns: 301 for the routers, 268 for the outer core and 191 for the tail.

**Memory.** Every routine is linear in `T` plus terms linear in `bytecode_rows()` and in the sizes of the fold tables. Each table of `T` or `bytecode_rows()` elements, and each scratch array, has one owner:

| Object | Size | Owner | Created | Dropped |
|---|---|---|---|---|
| `A`, `B`, `C` of lane groups 0 and 1 at the position point; tail bytes | 96 and 1 bytes per cycle | `OuterF2Core` | after position round 6 | `finish_rounds`; from the first cycle round the tables of group 1 are the second buffers |
| Bucket layout of `fold_pass` | 10.4 MiB per worker, 11.5 and 19.1 in the other layouts; 1 MiB for the pass over the rows | `ScratchPool` | start of `fold_pass` | its return |
| Scatter buffer | 20 bytes per cycle | `ScatterPlan::scatter` | each scatter | its return |
| `ra_fold`; the scatter of `claims_pass` | 16 bytes per bytecode row each | caller | `fold_pass`; `claims_pass` | `ra_fold` after `fold_pass` returns; the other by the bytecode address phase, which consumes it |
| `Fold_ρ`; `W_ρ` | 245,888 elements each | caller, then `RouterShortCore` | `fold_pass`; `RouterShortCore::new` | `finish_rounds` of the short core |
| Row tables of `source_lift` | 16 bytes per bytecode row for each shape with a `Bytecode` slot | `source_lift` | its start | its return |
| Word lifts | 16 bytes per cycle and trace word, 96 for six | caller | `source_lift` | after `claims_pass` |
| Source tables with second buffers | 24 bytes per cycle and shape, 120 for five | `RoutersCycleCore` | `source_lift` | the last member's `finish_rounds` |
| Selector columns | digits in the source through three binds; 1.5 bytes per cycle and column from the fourth, 12 for eight | `RoutersCycleCore` | fourth bind | with the core |
| Weight with second buffer | 24 bytes per cycle for `Dense`, none for `EqTerms` | `ChunkProductCore` | `combined_weight` | `finish_rounds` |
| Chunk columns | 1.5 bytes per cycle and column from the fourth bind, 7.5 for five | `ChunkProductCore` | fourth bind | with the core |
| `G` tables with second buffers | 24 bytes per cycle and table, 72 for three | `ReductionCore` | the table builder | `finish_rounds` |
| Buckets of `column_pass` | 128 KiB per worker | `ScratchPool` | start of `column_pass` | its return |

Peak bytes per cycle owned by cores and passes, the sources excluded: 97 for the outer core; 228 for the routers in the cycle rounds, and 216 with 48 MiB of row tables at the end of `source_lift`; 135 for the tail. At `log_t = 22` these are 388, 912 and 540 MiB. On 12 workers `fold_pass` adds 125 MiB of buckets and 80 MiB of scatter buffer for the length of the pass. No `jolt-eval` objective moves.

## Design

### Architecture

**Round messages.** A core returns `UnivariatePoly::new(coefficients)`; `prove_batch` and `BooleanHypercube::round_sum_coefficients` work on coefficients, and the proof omits the linear one (`specs/binary-sumcheck.md`). Internally a polynomial of degree `d` is known by its value at 0, its leading coefficient (the value "at infinity"), its value at 1 from the running claim, and its values at the `d − 2` nodes `F128::from_raw(2)`, …, `F128::from_raw(d − 1)`, for `d` up to 8. `coefficients_from_nodes` subtracts the leading term and interpolates with a matrix fixed per degree. A linear factor `u + t·v` at node 2 is `u + v.mul_x()`, at 3 is `(u + v) + v.mul_x()`, and so on; no node costs a field multiplication. Products of linear factors are formed in pairs: `quadratic` returns the three coefficients of a product of two by Karatsuba (3 multiplications), quadratics are evaluated at the nodes by `mul_x`, and the pointwise products follow.

Under an equality weight the round polynomial is `l(t)·q(t)` with `l(t) = σ·(1 + w + t)`, where `w` is the weight's coordinate for the round and `σ` the product of the bound coordinates' factors. A kernel accumulates `q(0)` and the higher values of `q`, obtains `q(1) = (claim + l(0)·q(0))/l(1)` from `recover_q_one`, interpolates `q` and multiplies by `l` with `round_poly_from_q_coeffs`. When `w = 0` the closure evaluates `q(1)` by the same loop on the other half of each pair. `gruen_poly_deg_3` and `gruen_poly_from_evals` evaluate `l` at 2 and 3 as integers and stay as they are for prime fields.

**Lifts and buckets.** A sum of bits against weights is a linear map of a word, so `WordLift` is eight tables of 256 entries (32 KiB) and a lift is eight lookups and seven XORs; `NibbleLift` uses 16-entry tables for words compacted by `gather`. Scalars that multiply a lift (a challenge monomial, a row weight) are folded into its tables. In the other direction, the sums `Σ_j e_j·bit_i(j)` for many bits `i` of a word are collected by `bucket[position][value] ^= e_j`, one load-XOR-store per nibble or byte per cycle, and read out at the end as XORs of the buckets whose index has bit `i`. `DigitHistogram` does the same for a digit column with one XOR per cycle. Neither direction multiplies in the field per bit.

**Equality over cycles.** Passes and the outer row rounds use two half tables over the cycle variables from `jolt_poly::EqPolynomial`: `e_j = E_hi[j_hi]·E_lo[j_lo]`. A sum of per-cycle values `v_j` is `inner.fmadd(E_lo[j_lo], v_j)` in an `F128Accumulator`, reduced and multiplied by `E_hi[j_hi]` once per block; a bucket pass forms `e_j` with one multiplication. Cycle rounds use `GruenSplitEqPolynomial` (`current_linear_evals`, `e_in_current`, `e_out_current`, `bind`) unchanged. Parallel loops are `par_chunks` over whole `E_hi` blocks, with a chunk length that depends on `log_t` and the round only.

**Scratch and scatters.** Accumulators live on the stack of a chunk. Bucket arrays and histograms come from a `ScratchPool`: a chunk takes an array when it starts and returns it when it ends, an array is zeroed when it is created and keeps accumulating across the chunks that use it, and at most `rayon::current_num_threads()` arrays exist. After the pass the arrays are merged pairwise in a tree, in parallel over disjoint ranges. Peak scratch is therefore the number of workers times the layout, whatever the number of chunks, and a layout of `n` elements costs `(2W − 1)·n` element operations for zero-fill and merge on `W` workers.

A scatter by bytecode row uses no table per worker, which would be `bytecode_rows()` elements each. `ScatterPlan::new(source)` counts, once, the cycles per chunk of cycles and per range of rows, for 256 ranges. A scatter writes `(row, weight)` pairs into one buffer of `T` pairs, held as an array of rows and an array of weights (20 bytes per cycle), at offsets taken from the counts; each range of rows is then applied by one worker into its own slice of the output. There is no merge, a range's slice of the output stays in cache while it is applied, and the plan serves every scatter over the same source.

**When bits become field elements.** A variable that ranges over the bits of a word (a lane position, a bit of a source word, a column of the committed table) is bound or summed out before any cycle variable, so that the first dense table holds one element per cycle and word. Until then the kernel reads words. A digit column stays a column of digits through three cycle binds (`LazyFoldedRa`) and is dense from the fourth.

#### Bitwise outer sum-check

Write `r_1, …, r_6` for the position challenges, `ρ_k(v)` for the equality weight of the position variables after the round variable at window pair `v`, and `ω_g` for the weight of group `g`. After `k` position rounds a *window* is `2^k` adjacent positions and `A_k(u) = Σ_i eq((r_1..r_k), i)·a[2^k·u + i]`; round `k + 1` needs, per pair of windows `(2v, 2v + 1)`,

`q(0) ∋ ρ_k(v)·(A_k(2v)·B_k(2v) + C_k(2v))` and `q(∞) ∋ ρ_k(v)·ΔA_k(v)·ΔB_k(v)`, with `ΔA_k(v) = A_k(2v) + A_k(2v + 1)`.

*Rounds 1 to `monomial_rounds` (default 3): products of bits as AND.* Let `μ(a)` be `moebius` of the lane word along its low `k + 1` position bits. For `S ⊆ {1..k}`, read as a number below `2^k`, bit `2^{k+1}·v + S` of `μ(a)` is the coefficient of `r^S` in `A_k(2v)`, and bit `2^{k+1}·v + S + 2^k` its coefficient in `ΔA_k(v)`. For an exponent vector `e ∈ {0,1,2}^k`, let `X_e` be the XOR of `μ(a)[S + 2^k] & μ(b)[S' + 2^k]` over the pairs of subsets whose indicator vectors sum to `e`, and `Y_e` the same without the `2^k`, each a word with one significant bit per window pair. Then `ΔA_k·ΔB_k = Σ_e r^e·X_e` (`3^k` words, `4^k` ANDs), and, because `A_k·B_k + C_k` has degree at most 2 in each `r_m` and vanishes on `{0,1}^k` when the rows are satisfied, `A_k(2v)·B_k(2v) + C_k(2v) = Σ_e (r^e + r^{red(e)})·Y_e`, where `red` replaces each 2 by 1 and only the `3^k − 2^k` vectors containing a 2 contribute (`4^k − 3^k` ANDs). The `C` lane is not read. In round 1 this gives `q(0) = q(1) = 0` and one word per group. Each word is compacted by `gather` and lifted by byte in rounds 1 and 2 and by nibble in round 3 (`OuterF2Options::nibble_round_2` moves round 2 to nibbles), with tables holding the word's scalar (`r^e`, or `r^e + r^{red(e)}`), `ρ_k(v)` and `ω_g`; the two per-cycle values go to `q(0)` and `q(∞)` with one `fmadd` each.

*Rounds up to 6: window lifts.* For `k ≥ 3` a window is `2^{k−3}` bytes, so every window value of a word is the XOR of byte lookups, eight per word whatever `k` is. The tables for `A` carry `ρ_k(v)` of the byte's window pair (8 tables), those for `C` likewise on the even windows (4 tables), those for `B` are unweighted (`2^{k−3}` tables): at most 64 KiB. Per window pair, `acc0.fmadd(A(2v), B(2v))`, `acc0.add(C(2v))` and `accinf.fmadd(ΔA, ΔB)`; per cycle and group the two accumulators are reduced and multiplied by `E_lo[j_lo]`; `ω_g` multiplies each group's total once per round. With `OuterF2Options::folded_group_weights` the `A` and `C` tables carry `ω_g` as well, at 112 KiB, and one accumulator pair serves the cycle.

*The switch.* Per group and round the monomial form lifts `2·3^k − 2^k` words, and the window form makes 20 byte lookups (40 nibble lookups below `k = 3`) and `2^{6−k}` multiply-accumulates. At price point 1 and `k = 2` (round 3) the first costs 37 ns per cycle and the second 74; at `k = 3` (round 4) they cost 76 and 41, and the order is the same at price point 2. `OuterF2Options::monomial_rounds` is in `3..=6` so that both forms are tested at rounds 4, 5 and 6.

*The tail.* Rows 128 and 129 are positions 0 and 1 of group 2. The pass of round 1 builds `H[i] = Σ_j e_j·[tail(j) = i]` for the 64 tail values with one XOR per cycle into a per-block histogram. The tail's terms of rounds 1 to 7 are sums over those 64 values of `H[i]` times the summand evaluated on the six bits of `i`.

*Rounds 7 and 8.* After round 6 the core materialises `A`, `B`, `C` of groups 0 and 1 at `(r_1..r_6)` with one `WordLift` (48 lookups, 96 bytes per cycle) and keeps the tail byte; the tail's values are lookups in three 64-entry tables. Round 7 pairs group 0 with group 1 on the tables and group 2 with the empty group 3 on `H`. Round 8 is per cycle: its low half is the bound pair of groups 0 and 1 and its high half is `(1 + r_7)` times the tail lookup. The binds at `r_7` and `r_8` are fused into the next round's pass and written in place.

*Cycle rounds.* Three tables of `T` elements under `GruenSplitEqPolynomial::new_with_scaling(τ_cycle, LowToHigh, Some(σ))`, `σ` being the product over the eight row rounds. Per pair: `q(0) = A_0·B_0 + C_0`, `q(∞) = ΔA·ΔB`, each multiplied into the split-equality accumulators, and three binds into the second buffers, which are the freed tables of group 1.

#### Routers

*What the slots share.* Bit variable `i` is slot `i` in every shape, so one point `r_bit` serves every source word: `source_lift` builds one `WordLift` and lifts each trace word once for all shapes, and every word value that `claims_pass` returns is at `(r_bit, r')`. Two shapes that give the same digit column the same slots have the same factor table `Sel_f`: the cycle core holds it once. In the experiment the low digit factor at slots 6–8 belongs to three shapes and the high digit factor at slots 9–11 to two, so the five members bind 8 selector columns for their 11 factors, and the two shapes with both digit factors form the product of the two columns once per pair. The slots do not reduce the bucket work of `fold_pass`: a shape's buckets are keyed by its own selector value, so a word in two banks is bucketed once per shape.

*`fold_pass(source, shapes, r_cycle, plan, layout, histogram_columns)`* makes one parallel pass with `e_j = eq(r_cycle, j)` and builds every entry of every `Fold_ρ`. For each shape and each cycle whose factor digits are all `Some`, spelling `h`, it XORs `e_j` under the key `h` into one bucket per nibble of every `Trace` slot of the shape, 16 per word, and into one bucket per digit column that a `Bits` entry of the shape names. The same pass scatters `ra_fold[bytecode_index(j)] ^= e_j` through `plan`. A `Bytecode` slot of a shape whose factors are all `by_row` is not bucketed per cycle: a pass over the rows with a non-zero entry of `ra_fold` XORs that entry into the nibble buckets of the row's words under the row's `h`. A `Bytecode` slot of any other shape is bucketed per cycle like a `Trace` slot. A `One` entry under `h` is the sum of `e_j` over the cycles that spell `h`, which is the total of the 16 buckets of any one nibble position under `h`. A requested histogram is a marginal where one exists: over `h`, of the digit buckets of a shape whose selector is present on every cycle; over the visited rows, of `ra_fold`, for a `by_row` column; otherwise it costs one more XOR per cycle. The read-out forms the entry of a source bit as the XOR of the 8 buckets of its nibble whose index has the bit. The only entries that cost nothing are those no cycle reaches: `Zero` slots and entries, and selector values no cycle spells.

`layout` is a `FoldLayout`: per shape, the selector values whose word buckets are by byte, 256 per byte position and 8 XORs per word. `selector_counts(source, shape)` returns the number of cycles per selector value, from which a caller picks the most frequent. Bucket arrays come from a `ScratchPool`. The outputs are the dense `Fold_ρ` of each shape, indexed in the shape's slot order, `ra_fold` and the histograms. For the five shapes of the experiment the pass costs, per cycle, one multiplication, one scatter and `88 + 16·p_shift + 32·p_access + 48·p_key + 32·p_taken` bucket XORs, with `p` the share of cycles that have a shift kind, a RAM access, a key kind, or are a taken branch; per visited row it costs 64 bucket XORs for the four `Bytecode` words of the 16-slot shape.

*`RouterShortCore::new(shapes, w, folds)`* scatters `eq(w, o)` over each `route` set into a dense `W_ρ` and runs one sum-check of `S` rounds. Each shape keeps its pair of tables, its current sum `S_ρ` and a scalar `λ_ρ`, initially 1. In round `k` a shape that has slot `k` contributes `λ_ρ` times the degree-2 round polynomial of `Σ W_ρ·Fold_ρ` in its next variable, and binds both tables at the challenge; a shape that does not have slot `k` contributes `λ_ρ·(1 + X)·S_ρ` and multiplies `λ_ρ` by `1 + x_k`. The member's message is the sum of the contributions. After the last round `Fold_ρ(x|_ρ)` is the shape's one remaining `Fold` entry and `λ_ρ` times its remaining `W` entry is `Idle_ρ(x)·W_ρ(x|_ρ)`.

*`source_lift(source, shapes, x)`* builds `Source_ρ(x|src, ·)` for every shape in one pass. Each distinct trace word is lifted once at `r_bit` with one `WordLift`, and the lifts are kept, one element per trace word and cycle. A shape's value is the combination of its slots' lifts under the equality weights of its word variables, by `fmadd` and one reduction, plus one lookup per digit column named by a `Bits` slot, in a table of that column's entries under their weights, plus one read of a per-row table that holds the same combination of the shape's `Bytecode` slots. The per-row tables are built first, in parallel over the bytecode rows, and dropped when the pass returns.

*`RoutersCycleCore::new(source, shapes, r_cycle, x, source_tables)`* owns the tables of `source_lift`, a second buffer of `T/2` for each, and one `LazyFoldedRa` over the distinct factor columns with tables `eq(x|_f, ·)`. `members()` returns one `RouterCycleMember` per shape; each implements `ProveRounds<F128>` and holds the core behind a shared lock that is taken once per member and round. The first member the engine calls in a round binds the previous challenge and makes the one pass of that round, accumulating the values of `q` for every shape; each member then recovers `q(1)` from its own claim and returns its own message. With one factor `q` has degree 2 and is known from 0 and infinity (2 multiplications and 2 `fmadd` per pair); with two factors it needs node 2 as well (6 and 3); with three, the source and the last factor form one quadratic, the first two factors another, and the nodes are 0, infinity, 2 and 3 (10 and 4; where two shapes have the same first two factors, the 3 multiplications of that quadratic are spent once for both). The bind of a source table is fused into the next round's pass. A pair in which some factor has no digit on either side during the lazy rounds contributes zero and is skipped.

*`claims_pass(source, lifts, words, plan, r')`* runs once after the cycle rounds. With `e'_j = eq(r', j)` it returns each listed trace word at `(r_bit, r')` (one `fmadd` per word and cycle on the kept lifts), the scatter of `e'_j` by bytecode row, through `plan`, and each bytecode word at `(r_bit, r')` as a sum over the rows with a non-zero entry of that scatter. With the members' selector values and source values these are the values a stage needs at `r'`, and no bound table per word is kept through the cycle rounds.

#### Chunk products

`ChunkProductCore::new(columns, points, weight)` holds a `Dense` weight with a second buffer of `T/2` and a `LazyFoldedRa` over the `d` columns with tables `eq(a_c, ·)`. While a column has fewer than four binds its value on a block of `2^k` cycles is the XOR of `2^k` lookups in tables pre-scaled by the equality weights of the bound challenges, so rounds 1 to 4 cost one lookup per column and cycle and no multiplication; the fourth bind materialises the column at `T/16`.

The summand has `d + 1` linear factors per pair: the weight and the `d` columns. They are paired as `(W·Ra_0), (Ra_1·Ra_2), …`, each pair one `quadratic`, with a last factor left linear when `d` is even. The quadratics are evaluated at the nodes by `mul_x` and multiplied pointwise, the last multiplication at each node fused into the accumulator. There is no linear factor to pull out, so the message is assembled by `coefficients_from_nodes` from the values at 0, infinity and the `d − 1` nodes, with the value at 1 from the claim. For five columns that is 9 multiplications for three quadratics, 12 at the six points, and one for the bind of the weight.

`combined_weight` forms each `Eq` term as a product of two half tables, with the term's coefficient folded into the high one, and accumulates the terms of a cycle unreduced with one reduction per cycle. A `Next` term reads the same half tables at the halves of `j − 1`. A term whose point lies in `{0, 1}^log_t` adds its coefficient to one entry.

With `ChunkWeight::EqTerms` the core holds no weight table. The product of the `d` columns is formed once per pair, at 0, infinity and `d − 2` nodes, and each term weights those values with its own split-equality tables and recovers its own `q(1)` from its own claim: for five columns and `m` terms, `16 + 5m` multiplications and accumulations per pair against `22.3 + m` with a dense weight, and 24 bytes per cycle less. A `Next` term is not a product over the bits of `j` and has no such form, so a weight with one is `Dense`.

#### Reduction of committed claims

*`g_pass_digits(source, map, weights)`*, the default, builds the tables from a `CycleSource`. A `ColumnMap` lists, for ranges of the 256 columns, what holds them: `Word { start, trace_word }` for 64 columns that are the bits of a trace word, and `Indicators { start, column }` for the `2^bits − 1` columns `start + k − 1` that are the bits `digit(column, j) = Some(k)`, `k ≥ 1`. A word costs eight byte lookups in tables `Λ_p[v] = Σ_{i<8} v[i]·L[8p + i]` of 256 entries, and an indicator range one lookup by digit. One parallel pass writes all tables. For the three weight vectors of the experiment that is 37 lookups per cycle in 69 KiB of tables.

*`g_pass_bytes(rows, weights)`* builds the same tables from the rows: for each weight vector and each byte position `p < 32` on which it is not zero, a table `Λ_p` of 256 entries, and `G[j]` is the XOR of `Λ_p[byte_p(row_j)]` over those positions. Where two weight vectors are non-zero on one byte position their entries are interleaved, so that one index serves both. For the experiment that is 49 lookups per cycle in 196 KiB of tables, of which the bytes that occur in rows with one-hot ranges read 74 to 92 KiB.

*`ReductionCore::new(tables, legs)`.* A leg with point `t` has the round polynomial `σ·(1 + t_i + X)·(A + X·B)`, where `B = Σ eq_rest·(G_lo + G_hi)` over the pairs and `A + t_i·B` is the leg's claim divided by `σ`. The core keeps that quotient per leg: it starts as the leg's claim and becomes `A + r·B` at each challenge `r`, so no round divides. Per pair a leg costs one `fmadd` for `B`, and each table one multiplication for its bind; legs on one table share the bind. The member's message is the sum of the legs' polynomials under their coefficients.

*`column_pass(rows, r)`* forms `e_j = eq(r, j)` with one multiplication per cycle and XORs it into one `ByteBuckets` entry per byte of the row, 32 per cycle; the arrays of a `ScratchPool` are merged by XOR, and the 256 column sums are read out from the 8,192 buckets.

#### Fit with `jolt-kernels` and the protocol

A core implements `ProveRounds<F128>`: `prove_round(bind, round, previous_claim)` binds the previous challenge and accumulates the round in one pass, and `finish_rounds` applies the last bind. A core is one `BatchMember` and sees member-local rounds; zero extension and idle rounds are the engine's. `RouterShortCore` is the one member of batch 3a; the five `RouterCycleMember`s are the members of batch 3b and need no scheduler other than `SequentialRounds`.

An implementation of `SumcheckKernel` for a relation of `specs/rv64i-binary-protocol.md` wraps a core and maps its final values to `output_claims`: `Fold_ρ(x|_ρ)` to the five values of `RouterShort`; the members' source and selector values and the outputs of `claims_pass` to the values of batch 3b; the chunk values to `chunks` of `BytecodeReadCycle` and `RamRaProduct`; the result of `column_pass` to `columns` of `BitsReduction`. These adapters live in `jolt-rv64i-prover/src/optimized/`. `prepare` receives the witness, clones the `Arc`s of the tables a core reads later, and returns the `'static` kernel. Tables that one stage builds for a later one (the outputs of `fold_pass`, the kept lifts, the scatter plan) are parked in `ProofSession` (`park`, `take`).

An adapter is accepted, in its own PR, when under the transcript of the reference kernel of its relation: every round's coefficients are equal; every typed output value and every alias the member declares is equal; `validate_derived_tables` passes and each derived term the kernel holds equals the verifier's formula at the final point; and the member's output expression on those values equals its final claim. `run_lockstep` (`crates/jolt-kernels/src/optimized/parity.rs`) compares coefficients and calls `finish_rounds`; it does not call `output_claims`, so it covers the first of the four. The adapters wait for the witness-plane parameter of `PrepareKernel` and for a reference prover that samples at distinct nodes of `F128` (`specs/binary-protocol-family-seams.md`).

### Alternatives Considered

Counts are per cycle at `log_t = 22`; a saving is given at price point 1 and at price point 2.

- **A fold restricted to the entries `Route` reads.** Fewer bucket XORs, and wrong. The short sum-check hands `Fold_ρ(x|_ρ)` to the cycle member, and that extension reads every entry. With one source variable, `W = [1, 0]` and `Fold = [a, b]`, the round polynomial is `(1 + X)·((1 + X)·a + X·b)`; with `b` dropped it is `(1 + X)^2·a`. Both sum to `a` over `{0, 1}` and they differ by `X·(1 + X)·b`, so the member that evaluates the complete `Source·Select` cannot meet the hand-off.
- **Reuse `optimized/spartan_outer.rs`.** It evaluates rows as integers (`i64`, `i128`), extends them over the centered integer domain by finite differences and accumulates field-by-integer products. None of the three exists in characteristic 2, and here the rows are sum-check variables with bit values. `GruenSplitEqPolynomial` is what carries over.
- **Univariate skip over `F8Domain` for the lane position.** One message of degree 189 needs, per lane group and cycle, the three lanes extended to 63 points of `F8`, 63 products and 63 accumulations against a 128-bit weight: about 156 ns at price point 1 counted the same way, against 158 for rounds 1 to 6, with 187 more field elements in the proof and a different round count.
- **Dense tables from round 1.** 768 elements per cycle, 12 KiB.
- **`accumulate_product_grid`.** It extends factors to the integers `1, …, n − 1` by finite differences; those points are 0 and 1 in characteristic 2.
- **Byte buckets for the four small shapes in the fold pass.** `8·p_shift + 16·p_access + 24·p_key + 16·p_taken` XORs less, 7 ns on the `local` mix, at 72 MiB of buckets per worker for the `Compare` shape alone.
- **One scatter table per worker.** `bytecode_rows()` elements per worker: 192 MiB on 12 workers at `2^20` rows, and every page a visited row touches becomes resident. It is measured by `probe` and not built.
- **Atomic XOR for the scatter.** Correct, since each half of an element is XORed independently, but every worker writes the rows of a hot loop.
- **A bound table per base word through the cycle rounds.** 20 more multiplications per cycle than the kept lifts and `claims_pass`. The opposite trade, lifting five of the six words again in `claims_pass`, costs 40 lookups (16 ns) and frees 80 bytes per cycle through the cycle rounds; it is not taken while memory is not the constraint.
- **One short core per shape, each a batch member.** A batch member is active on one contiguous window of rounds. A shape that skips a slot between two of its own (slot 10, slots 9–11, slots 6–11) is active on two windows.
- **A `RoundScheduler` that owns the cycle pass.** `prove_batch` accepts one, and it removes the lock. The members would then produce no message under `SequentialRounds`, which is what a stage driver passes.
- **Dense chunk columns from round 1.** Five multiplications and 80 bytes per cycle for five columns, in place of 25 lookups.
- **`G` tables of half length.** Round 1 of the reduction needs only the lifts of the XOR of adjacent rows, so the first stored tables can have `T/2` entries, at a second pass over the rows after the first challenge: 36 bytes per cycle in place of 72 for three tables with their buffers. Not taken while memory is not the constraint.
- **A table builder fused with `column_pass`.** `column_pass` needs the challenge point of the rounds that the builder precedes.

Three uses of the structure of the routers are candidates for after the cores above are measured. Each is prover-only and none is built here.

- **The fold of the 16-slot shape keyed by wiring pattern.** That shape's tensor is a sum of wires, each copying a source word to an output word under one of 12 bit patterns. Byte buckets keyed by (output word, pattern), filled with all 64 bits of each wired word, give the shape's six bit rounds on at most 144 pairs of 64-entry tables; the table over (word slot, selector value) that the remaining rounds need comes from one `fmadd` of `e_j` with each trace word's lift at `r_bit`. Per cycle: `8 Bk` per wire of the cycle's selector value and `8 Bk + 1 M + 5 A`, against `88 Bk`. With 4.6 wires per cycle, an estimate, the saving is 22 to 25 ns at point 1 and 25 to 28 at point 2, in at most 4.5 MiB of buckets. It gives the short core a second representation of one shape and a pass over the cycles in its sixth round.
- **Skipping the pairs of a cycle round on which a shape's selector is zero.** A pair of round `i` covers `2^i` cycles and contributes nothing when none of them has the shape's kind digit. On the `local` mix with independent cycles the product work that disappears is 77% for `Shift`, 26% for `Memory`, 45% for `Compare` and 74% for `Branch`: up to 37 ns at point 1 and 20 at point 2, and dependent on the trace. For the skip to cost less than it saves, the core keeps one activity bit per cycle and shape, folds the bitmap by OR at each bind and walks its set bits; it does not test selector values per pair. Four bitmaps of `T/8` bytes.
- **One evaluation for the five cycle members.** The members share `eq(r_cycle, ·)` and the batch sends the weighted sum of their polynomials. With each member's batching coefficient folded into its kind selector, the weighted summand nests as `Src_V·Var + P_0·(P_1·(Src_S·SK + Src_C·KK) + Src_M·AK) + Src_B·Br·SB` and costs `26 M + 4 A` per pair against `31 M + 16 A`: 22.4 ns less at point 1 and 12.9 at point 2. It yields one polynomial for the group and none per member, so it needs `prove_batch` to accept a weighted polynomial for a declared set of members and to keep one running claim for the set; `MemberRound` carries no weight today. The same nest as one relation of degree 5 in place of the five members costs the same and changes the transcript; it is not proposed.

## Documentation

No change to the Jolt book. The rustdoc of `source`, `packed`, `round`, `outer_f2`, `router`, `chunk_product`, `reduction` and `column_pass` states the definitions of Goal and the contracts of the surface table. The module documentation of `benches/support/` gives the command that runs a benchmark through the runner and the meaning of its four phases.

## Execution

Fourteen PRs with disjoint files. The first creates the crate with every module and benchmark target as a stub, so a later PR fills files and adds none to a shared list. No PR implements a fold restricted to the support of `Route`.

1. `jolt-rv64i-kernels`: `Cargo.toml`, `lib.rs`, `source.rs`, `synth.rs`, `oracle.rs`, `par.rs`, `benches/support/` (the runner, the counting allocator, the dense-product example core `Σ_j A[j]·B[j]`), `benches/example.rs`, and stubs of every other module and of `benches/{probe, machinery, fold, outer_f2, routers, chunk_product, reduction, column_pass, tail}.rs`. Accepted when each profile is a function of its seed, its lanes satisfy `C = A & B`, its digits agree with the indicator columns of its rows, `oracle` reproduces the messages of `prove_batch` on a dense product, and the runner criterion holds.
2. `jolt-field`: `mul_x` and its row in `benches/binary_kernels.rs`.
3. `jolt-poly`: the two functions and two methods of `split_eq.rs`.
4. `jolt-kernels`: `pub mod lazy_ra`, the visibility of its two items and their methods, `index_bound`, `try_new` and `LazyRaError`.
5. `benches/probe.rs`; after 1. Its report goes into the PR description and into the unit table of this spec.
6. `round/{nodes, product}.rs`; after 1, 2 and 3.
7. `packed/{lift, buckets, bits, pool, scatter}.rs` and `benches/machinery.rs`; after 1, with the report of 5.
8. `router/{shape, fold}.rs` and `benches/fold.rs`; after 7. Accepted on the two complete-fold criteria and the `fold_pass` threshold, with the three layouts and their scratch reported.
9. `chunk_product.rs` and its bench; after 4 and 6.
10. `reduction.rs` and its bench; after 6 and 7.
11. `column_pass.rs` and its bench; after 7.
12. `tests/tail.rs` and `benches/tail.rs`; after 9, 10 and 11.
13. `outer_f2.rs` and its bench; after 6 and 7.
14. `router/{short, lift, cycle, claims}.rs` and `benches/routers.rs`; after 4, 6 and 8.

PRs 1 to 4 start at once and 5 follows 1. PRs 6 and 7 run in parallel. PRs 8 to 11 run in parallel and 12 closes them, so the complete fold is shown correct and the tail is measured end to end before the two largest cores are written. PRs 13 and 14 run in parallel. None waits for a relation definition: the benchmarks and the tests drive the cores through `SyntheticTrace` and through shapes written in the tests. From PR 8 on, each PR description reports the four phases of its benchmarks at both sizes and both thread counts against its row of the Performance section.

## References

- `specs/rv64i-binary-protocol.md`: the batches, members, points and relations these cores serve (§5, §8), the slot table of the routers (§8.4), `C` (§10) and the witness (§12).
- `specs/rv64i-binary-arithmetisation.md` (`BitsRow`, `LaneRows`, `Term::wires`).
- `specs/binary-field.md`, `specs/binary-accumulators.md` (`F128`, `F128Accumulator`); `specs/binary-sumcheck.md` (wire form, `BooleanHypercube`, member windows); `specs/binary-uniskip-domain.md` (`F8Domain`); `specs/binary-protocol-family-seams.md` (`PrepareKernel` over a witness plane, the reference prover's nodes).
- `crates/jolt-poly/src/split_eq.rs`, `crates/jolt-poly/src/lagrange.rs`, `crates/jolt-sumcheck/src/prover.rs` (`ProveRounds`, `prove_batch`, `SequentialRounds`, `MemberRound`), `crates/jolt-sumcheck/src/batch.rs` (`BatchMember`), `crates/jolt-kernels/src/optimized/lazy_ra.rs`, `crates/jolt-kernels/src/optimized/parity.rs`.
