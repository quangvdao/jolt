# Spec: The Commitment Scheme for the Bit Table of RV64I over the Binary Field

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-10                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The front end of the binary-field RV64I experiment (`specs/rv64i-binary-protocol.md`) commits to one table of bits, `Bits`, with 256 columns and `T = 2^t` rows, and at the end asks for one multilinear evaluation of it at a point with coordinates in `F128`. The only other implementation of that contract is `TransparentBits`, a test stand-in that sends the table. This spec fixes the production scheme behind the same two traits, `BitsCommitmentScheme` and `BitsCommitmentProver`, with no change to either.

The scheme is hash-based. The opening field is `F192`, the cubic extension `F64[y]/(y^3 + y + 1)` of `F64`. The bits are packed 128 to a symbol: two consecutive 64-bit words of a row are the two coefficients of an element of the subspace `V = F64 + y·F64` of `F192`. The `2^(t+1)` symbols are encoded with an interleaved Reed-Solomon code on an additive domain of `F64`, which acts on `V` coefficient by coefficient, so that the level-0 codeword consists of words of `F64` and a symbol is 16 bytes. The codeword is committed with a Merkle tree, and the commitment carries one out-of-domain evaluation of every lane of the code, so that after the commit phase at most one table is selected and not a list of tables. The opening is a WHIR recursion over `F192`, analysed in the list-decoding regime up to the Johnson bound. The evaluation claim lives in `F128`, which is not a subfield of `F192`; a tensor reduction (ring switching) of shape 128 by 128 turns the 128 partial evaluations that the front end already publishes into one inner-product claim over `F192`. No element of `F128` is ever multiplied inside `F192`, so the missing embedding costs nothing.

At the reference size `t = 22` the table has `2^30` bits and `2^23` symbols. The parameter set is rate 1/2, fold schedule 5, 4, 4, 4, 4, query counts 259, 65, 37, 26, 20 and no grinding. Every error term of the scheme is at most `2^-128`; the weakest is the query term of the first level at `2^-128.003`, followed by its proximity-gap term at `2^-128.27`. The opening proof is 325,532 bytes in expectation and 336,792 at most, and the commitment is 800 bytes (counted). The public leanVM implementation of the same recursion, at the same level-0 oracle and for a claim in `F192`, measures 311 ms to commit and 730 ms to open on one thread and 67.8 ms and 142.6 ms on twelve, on a loaded host. The model of this spec for the scheme with the `F128` bridge and the commit-time sample is 157.2 ns per cycle on one thread (estimated), which gives single-thread thresholds of 94 ns for commit and 103 ns for open, together 197 ns per cycle and 9.4% of the prover's 2,100.

The packing of 128 bits into `V` is taken on operation counts against a packing of 64 bits into `F64` and against an opening field of `2^256` elements that contains `F128` (§7). It is confirmed or reversed by the first measurement of the bridge phase, and §7 states the result that reverses it.

## Intent

### Goal

Fix, completely enough to implement and to audit, the commitment scheme of the bit table: the committed object and the claim, every message of the commit phase and of the opening, every challenge and the bytes it is drawn from, the wire format, the parameter set and its soundness ledger, the prover's operation counts and memory, and the items of work.

Notation used throughout. `K = F64`, `H = F128` and `E = F192` are the types of `jolt_field::binary`. `E` is `K[y]/(y^3 + y + 1)` and `F192` stores its three coefficients in `K` in ascending degree, so `K ⊂ E` is the first coefficient, `mul_base` multiplies each coefficient by an element of `K`, and the canonical 24 bytes are the three coefficients in order; `H` is unrelated to `E`. `V = K + y·K ⊂ E` is the set of elements whose third coefficient is zero. It is a subspace over `K` of dimension 2, closed under addition and under multiplication by `K`, and not closed under multiplication. `pack(a, b) = F192::from_base_fn` of `(F64::from_raw(a), F64::from_raw(b), 0)` is the element `a + y·b` of `V` for two words `a`, `b`. `t = log_T`, and `μ = t + 1` is the number of variables of the packed table. Points are low-variable-first and sumchecks bind the lowest variable first, as in the front end. `eq_F(r, z)` is the equality polynomial over the field `F`, with bit `l` of the index `z` paired with `r[l]`. `β_i = x^i`, `i < 64`, is the basis of `K` over `F_2` in which `F64::from_raw` reads a word, and `ν_b = β_(b mod 64)·y^⌊b/64⌋`, `b < 128`, is the basis of `V` over `F_2`. `bit_h(e)` is bit `h` of the raw representation of an element of `H`; for `v = pack(a, b)`, `bit_b(v)` is bit `b` of `a` for `b < 64` and bit `b − 64` of `b` after it, the coordinate of `v` on `ν_b`. "Level" means one committed oracle of the recursion; level 0 is the commitment itself.

### Invariants

1. **The committed object.** The commitment binds one function `p : {0,1}^μ → V`. For an honest prover `p[g + 2·j] = pack(row_j[2g], row_j[2g + 1])` for `g < 2`, `j < T`, where `row_j : [u64; 4]` is row `j` of the table, so that `bit_b(p[g + 2·j]) = Bits[b + 128·g, j]` for `b < 128`. Index and bit order are those of §3 of the protocol spec and are frozen here.
2. **Unique binding at commit time.** After `verify_commit` returns, and except with the probability of the ledger rows "commit sample" and "Merkle", there is at most one table of bits for which any later opening can be accepted. The mechanism is the out-of-domain evaluation carried in the commitment and checked in the opening (§2, §4 of Architecture).
3. **The claim proved.** `verify_opening` accepts only if, for the bound table, `Σ_{z} eq_H(r, z)·bit_b(p[z]) = s_b` for all `b < 128`, where `r = (rho[7], r_6)` and `s_b = (1 + rho[7])·C[b] + rho[7]·C[b + 128]`, except with the probability of the ledger. This implies `Bits~(rho, r_6) = BitsOpening::value()`, which is the contract, and is strictly stronger. `rho[0..7)` is not used by the scheme.
4. **Fields.** Code symbols of level 0 are in `V`, and each is stored, hashed and sent as its two coefficients in `K`. Every challenge of the scheme, every later symbol and every sumcheck message is in `E`. No value of `H` is converted to `E` other than through the map `Φ_α` of §3, which reads its bits.
5. **Challenges.** Every challenge in `E` is 24 bytes from `squeeze_bytes`, that is two consecutive 16-byte draws of which the first 24 bytes are kept, read by `F192::from_bytes_le_checked`. It is uniform in `E`. No challenge of the scheme is sampled from a subset of `E`, and none is derived from another challenge except the powers of a batching challenge.
6. **Transcript.** Every message of the scheme is absorbed before the challenge that depends on it is drawn, in the order of §6. The scheme uses six labels, none of which is one of the nine of the front end.
7. **Canonical wire.** `BitsWire::read` accepts exactly the strings that `write` produces for the geometry. Lengths are checked against bounds computed from the geometry before any allocation. A commitment or an opening has one encoding.
8. **Parameters are constants.** The level schedule and the query counts are functions of `t` alone, stored as a table of integers (§5). No floating-point computation runs in the verifier or in the prover.
9. **The prover copies no table.** `commit` reads the rows through the `Arc` it is given and writes the level-0 codeword, which is a new buffer of twice the table's size. It makes no packed copy of the table. `open` reads the rows through the same `Arc`.
10. **Verifier safety.** The verifier's code is in a crate with `#![forbid(unsafe_code)]`, does not panic on any input, and allocates in proportion to validated lengths only.

### Non-Goals

- Zero knowledge. The scheme hides nothing.
- A change to the front end, to its proof, or to the two traits. The scheme is one more implementation of them.
- A general polynomial commitment scheme for `jolt-openings`. The generalisation of those traits is a separate later item (Execution, item 9).
- Recursion-friendliness of the verifier, batching of several tables or several points, and streaming commitment.
- A proof of the proximity-gap theorem or of the round-by-round theorem. The spec states which published results it rests on and with what constants.

## Evaluation

### Acceptance Criteria

Section numbers refer to Architecture.

**Parameters.**

- [ ] For every `1 ≤ t ≤ 32` the schedule function returns the `k_i`, `c_i`, `d_i`, `R` and `res` of §5; the rows for `t = 1`, `6`, `10` and `22` are literals in the test: `(0; 2; 3)`, `(5; 2; 3)`, `(5, 4; 6, 2; 7, 6)` and the table of §5.
- [ ] The exact-arithmetic test of item 1 confirms every query count and asserts every ledger row at most `2^-128` for every `t`.

**Field.**

- [ ] `F192Accumulator::fmadd_base(a, b)` followed by `reduce` equals `a.mul_base(b)` summed, on operands supported on single coefficients for every pair of positions, and on 1,024 random pairs.
- [ ] For `e ∈ E` and `v = pack(a, b)`, `fmadd_base(e, a)` followed by `fmadd_base(y·e, b)` and `reduce` equals the product `e·v` in `E`, on 1,024 random pairs; `y·(c_0, c_1, c_2) = (c_2, c_0 + c_2, c_1)` on the coefficients.

**Code.**

- [ ] The encoder by definition gives, for `c = 1`, `d = 2` and `f = (f_0, f_1)`: `f_0`, `f_0 + f_1`, `f_0 + β_1·f_1`, `f_0 + (β_1 + 1)·f_1` at positions 0 to 3. For `c = 2`, `d = 3`, where `s_1(β_1) = x^2 + x` and `Ŵ_1` takes the values 0, 0, 1, 1, `x^2 + x` at positions 0 to 4: position 2 gives `f_0 + x·f_1 + f_2 + x·f_3` and position 4 gives `f_0 + x^2·f_1 + (x^2 + x)·f_2 + (x^4 + x^3)·f_3`. These are written in the test as raw words: no product in them reaches degree 64.
- [ ] The transform equals the encoder by definition for every `1 ≤ c < d ≤ 8`, for 1, 2, 16 and 64 lanes, over `F64` and over `F192`, and the level-0 transform on 2, 32 and 64 words per position equals the encoder by definition applied to the lanes of `V` that pairs of words form; the transposed transform satisfies `⟨Enc(f), g⟩ = ⟨f, Enc^T(g)⟩` on random `f`, `g`.
- [ ] `W~_x(q)` equals `Σ_w eq_E(q, w)·X_w(x)` computed from the definition, for every `x` at `d ≤ 6`.

**Merkle.**

- [ ] The root of a tree of 4 leaves equals `H(H(H(l_0) ‖ H(l_1)) ‖ H(H(l_2) ‖ H(l_3)))` computed with `blake2::Blake2s256` in the test; the batched tree builder equals the tree built by that definition for every depth up to 10 and leaf sizes 16, 64, 384 and 512 bytes.
- [ ] For every nonempty subset of positions at depth 4, the multiproof written verifies, its digest count equals the count that the verifier derives, and it is rejected with any digest altered, with one digest removed, and with one digest appended.

**Bridge.**

- [ ] For a table with one nonzero symbol `p[z_0] = ν_b`, for `b = 3` and `b = 67`: with `r = 0` and `z_0 = 0`, `τ = ν_b`; with `r = (x, 0, …)` and `z_0 = 1`, `τ = ν_b·α`. Both are literals for a fixed `α`.
- [ ] For one variable and `r = (x)`: `w~((q_0)) = 1 + q_0 + α`, checked for a fixed `q_0` and `α`.
- [ ] For `μ ≤ 10`, random `p`, `r`, `α`: `s_b` computed bit by bit from the table and `eq_table`, then `τ = Σ_z p[z]·Φ_α(eq_H(r, z))`; the recurrence equals `Σ_z eq_E(q, z)·Φ_α(eq_H(r, z))` computed directly; the prover's stored vectors `W0`, `D` of §9 equal `Φ_α(eq_H(r, 2k))` and `Φ_α(eq_H(r, 2k)) + Φ_α(eq_H(r, 2k + 1))` computed bit by bit.
- [ ] With one `s_b` changed and everything else fixed, the bridge identity holds for at most 127 values of `α`; the test checks rejection for a fixed `α` whose acceptance it has excluded by computing the discrepancy polynomial, and does not assert rejection for every `α`.

**Wire.**

- [ ] `read(write(x)) = x` for commitments and openings of every `t ≤ 14` used in the tests; every proper prefix and every extension by one byte of an accepted string is rejected; `n_i = 0`, `n_i > Q_i` and `g_i > n_i·d_i` are rejected before allocation, shown by a test that feeds lengths of `2^32 − 1`.
- [ ] `read` and the two verifier functions return without panicking on 10,000 random strings and on every single-byte mutation of an accepted proof at `t = 6`.

**Transcript.**

- [ ] A recording transcript sees exactly the calls of §6, in order, for `t = 6` and `t = 11`; the six labels are disjoint from the nine of the front end; prover and verifier states are equal after `commit` and after `open`.
- [ ] For a fixed transcript state, the positions of a level with `d = 7`, `Q = 5` equal a literal computed in the test from the squeezed bytes by the rule of §6.

**Completeness.**

- [ ] For `t ∈ {1, 2, 5, 6, 9, 10, 11, 14}` and random tables, with `C` and the points produced as the front end produces them, `verify_commit` and `verify_opening` accept the output of `commit` and `open`, and `BitsOpening::value()` equals the evaluation of the table computed bit by bit.
- [ ] The existing contract tests of the two traits pass with `WhirBits` in place of `TransparentBits`.

**Rejections.** Each of the following makes `verify_opening` or `read` return an error, at `t = 6` and `t = 11`:

- [ ] one byte changed in each of: `root_0`, an element of `y_0`, each round coefficient, each later root, each `y_i`, each element of `f_R`, a leaf, a sibling digest, `n_i`, `g_i`;
- [ ] an opening produced for a table that differs from the committed one in one bit;
- [ ] `C` changed in two entries so that `value()` is unchanged and some `s_b` is not (this is the stronger claim of invariant 3);
- [ ] `C` changed in one entry; `rho[7]` or one coordinate of `r_6` changed; the geometry changed;
- [ ] an opening verified against a transcript that differs before the commit phase.

**Determinism.**

- [ ] The commitment and the opening bytes are identical on 1 and on 12 threads and across runs, and for the fixed table and transcript of the test at `t = 6` their Blake2b-256 digests equal literals fixed in item 7.

**Front end.**

- [ ] `prove` and `verify` of the experiment accept the counting loop at `t = 6` with `WhirBits`, and the tamper tests of the front end that target the commitment and the opening reject.

**Hygiene.**

- [ ] `jolt-rv64i-verifier` keeps `#![forbid(unsafe_code)]`; `jolt-rv64i-pcs` has `unsafe` only under `src/arch/`; `cargo clippy` with `-D warnings` passes for the features of the experiment; each file derived from the port carries its source's header.

### Testing Strategy

Ground truth is independent of the code under test. The encoder is tested against the definition of the code, evaluated with field operations and one inversion per basis polynomial; the tree against the `blake2` crate applied by the definition; the bridge against bits read from the table; the claim against a bit-by-bit evaluation. The optimised transform and the batched hash are tested against those definitions and not against each other. The smallest cases are literals written out in this spec. The ported implementation is not used as an oracle: its lane order, wire and transcript differ, and it lives outside the repository. Tests run with `cargo nextest`, in the crates of the items that own them. The exact-arithmetic test of the parameters is the only place where the ledger is computed in the repository.

Soundness is not tested by any of this. The rejection tests show that single faults are caught, and the ledger of §8 is an argument on paper with the open points listed at the end.

### Performance

Benchmarks follow the kernels spec: an Apple M4 Max, `-C target-cpu=native`, one thread and a `rayon` pool of 12, `log_t = 20` and `22`, through the runner of `crates/jolt-rv64i-prover/benches/support/`. The benchmark is `bits_whir` with the cases `commit` and `open` and the phases of the table below; the table is random bits from a seeded generator, since no kernel of the scheme depends on the data.

Every figure below is marked *counted* (derived from the algorithm), *estimated* (a count times a unit price that has not been measured for this code) or *measured*.

**Measured: the existing implementation at this geometry.** A separate harness ran the public leanVM implementation unchanged at `2^24` symbols of `F64`, rate 1/2, initial fold 6, later folds 4, no grinding, query counts 260, 65, 37, 26, 20, BLAKE2s-256, on the benchmark machine while it was loaded with other builds, five runs per configuration. Medians, in milliseconds:

| Phase | 12 threads | 1 thread |
|---|---:|---:|
| Packed copy of the rows | 9.75 | 12.07 |
| Level-0 encode (allocation, transposition, transform) | 37.71 | 161.58 |
| Level-0 tree | 19.86 | 136.82 |
| Ring switch, weights, bit slices, first round message | 81.65 | 447.19 |
| Rounds, folds, final message | 28.23 | 122.78 |
| Induced weights | 3.51 | 6.98 |
| Later encodes | 11.74 | 65.37 |
| Later trees | 14.79 | 88.57 |
| Samples of later levels | 0.85 | 1.49 |
| Queries and proof assembly | 0.76 | 0.78 |
| **Commit** | 67.77 | 311.27 |
| **Open** | 142.61 | 729.88 |
| Verify | 1.79 | 1.78 |
| Proof bytes (its own serialisation, pruned paths) | 332,112 | 332,112 |
| Peak resident memory, rows and packed copy included | 1,325 MiB | 1,319 MiB |

What these figures do not measure: the bridge of §3, since the harness opens a claim whose point is in `F192` through the source's own 64 by 192 ring switch and computes the 64 slice values that this scheme reads from `C`; the commit sample, since the source commits a root alone; the lane order and wire of this spec; and an idle host. The twelve threads were eight performance and four efficiency cores. They are evidence for the ported phases and an upper reference for the rest.

**Unit prices.** In nanoseconds, added to the table of the kernels spec.

| Symbol | Operation | Point 1 | Point 2 | Source |
|---|---|---:|---:|---|
| `c` | one carry-less multiplication of 64-bit words with its share of the reduction | 0.305 | 0.15 | estimated: M/6 of the kernels spec's unit table, to be measured by item 2 |
| `Lw` | lookup of a 24-byte entry and its XOR, tables of 96 KiB | 0.6 | 0.6 | estimated: 1.5 L, to be measured by item 5 |
| `nb0` | one level-0 butterfly, with allocation and copy | 0.535 | 0.535 | measured, loaded host: 161.58 ms over 301,989,888 |
| `nb1` | one later-level butterfly, with allocation and domain set-up | 1.50 | 1.50 | measured, loaded host: 65.37 ms over 43,515,904 |
| `hb0`, `hb1` | one BLAKE2s compression in a tree of 512-byte, 384-byte leaves | 29.0, 25.7 | the same | measured, loaded host: 136.82 ms over 4,718,591; 88.57 ms over 3,440,636 |

**Model, at `t = 22`, one thread.**

| Phase | Operations (counted) | Point 1, ms | Status |
|---|---|---:|---|
| Level-0 encode | 301,989,888 `nb0` | 161.6 | measured unit; includes a transposition that §2 removes |
| Level-0 tree | 4,718,591 `hb0` | 136.8 | measured unit |
| Commit sample | `(3·2^24 + 12·2^18) c` = 53,477,376 `c` | 16.3 | estimated |
| **Commit** | | **314.7** | |
| Bridge weight and round 1 | 327,155,712 `c` + 268,435,456 `Lw` | 99.8 + 161.1 = 260.8 | estimated |
| Rounds 2 to 24 | 301,989,852 `c` | 92.1 | estimated |
| Induced weights | 7,405,568 butterflies | 7.0 | measured phase |
| Later encodes | 43,515,904 `nb1` | 65.4 | measured unit |
| Later trees | 3,440,636 `hb1` | 88.6 | measured unit |
| Samples of later levels | | 1.5 | measured phase |
| Queries and assembly | 408 positions | 0.8 | measured phase |
| **Open** | | **516.1** | |

Per cycle that is 75.03 ns for commit and 123.06 ns for open, 198.09 ns together. At point 2, where only `c` changes, 73.06 and 99.81. The two estimated phases of the opening, 352.9 ms, stand where the source's measured ring switch and rounds take `447.2 + 122.8 = 570.0` ms; the difference is the slices that this scheme does not compute and a first round that is modelled and not built, and it is the least certain part of the model.

**Thresholds.** A threshold is single-thread nanoseconds per cycle at `log_t = 22`, the unrounded model times 1.25 to the nearest nanosecond, as in the kernels spec.

| Benchmark | Model, point 1 | Threshold | Model, point 2 | Threshold at point 2 |
|---|---:|---:|---:|---:|
| `bits_whir/commit` | 75.03 | 94 | 73.06 | 91 |
| `bits_whir/open` | 123.06 | 154 | 99.81 | 125 |

The thresholds of point 1 are in force. They move only when a row of the unit table is replaced by a measurement of the operation that the row names, on a quiet host and on this scheme's code, and then every model and threshold is recomputed in the same PR; no threshold moves to meet a result. The measured rows above are provisional in that sense: they come from a loaded host and from the source. At `log_t = 20` the benchmark reports and has no threshold: the schedule has four levels and a different final message, and no measurement exists at that size.

On twelve threads at `log_t = 22` the house rule divides the threshold by 9.6: 9.8 ns per cycle for commit and 16.0 for open, 41 ms and 67 ms. The measurement does not support that factor for this workload. The source reaches 16.2 and 34.0 ns per cycle, 67.8 ms and 142.6 ms, which is a speed-up of 4.6 and 5.1 over its single thread and not 9.6. The data cannot separate the load on the host from the memory traffic of the transform and the hashing and from the four efficiency cores. The requirement stays as the rule gives it, it is recorded as at risk, and item 8 measures the scaling on a quiet host before anyone relies on it (Open, 6).

**Budget.** The prover's budget is 2,100 ns per cycle on one core, 918 ms at `2^22` cycles on twelve threads at the factor 9.6. The scheme takes `94 + 154 = 248` ns of it, 11.8%; with the 745 ns of the kernels' thresholds, 993 ns are allotted and 1,107 remain. In wall time on twelve threads that is 108 ms if the factor of 9.6 holds and `248·2^22/5.02 = 207` ms at the scaling measured for the source. The earlier proposal of 400 ms, 180 for commit and 220 for open, is not adopted: the existing implementation measures 210 ms for both on a loaded host, and the model with the bridge and the commit sample is below the source on one thread. The budget is the thresholds above and nothing looser.

**Memory.** Counted, §9: 864 MiB owned at the peak, 992 MiB with the rows, against 1,325 MiB measured for the source with its packed copy. The benchmark reports the peak of owned allocations through the recorder of the bench support; the requirement is that it does not exceed the counted 864 MiB by more than 5% at `log_t = 22`.

**Verifier.** Counted at `t = 22`: at most `Σ Q_i·(leaf compressions + d_i) = 10,423` BLAKE2s compressions, 3,072 products in `E` for the recurrence, and 408 lane combinations and weight evaluations. The source verifies in 1.78 ms (measured). The benchmark reports the verifier's time; it has no threshold.

**Proof size.** Counted, §6: 326,252 bytes expected, 430,024 at most, for the opening, and 1,568 for the commitment. Item 8 reports the distribution over 1,000 transcripts.

## Design

### Architecture

#### 1. The committed object and the claim

The table has `256·T` bits. Row `j` is a `BitsRow = [u64; 4]`, and the bit of column `col` of row `j`, at flat index `col + 256·j`, is bit `col mod 64` of word `⌊col/64⌋`. The packed table takes two consecutive words to one symbol: `p[z] = pack(row_j[2g], row_j[2g + 1])` at `z = g + 2·j`, `g < 2`. It has `2^μ` symbols, `μ = t + 1`. The 16 canonical bytes of a symbol are its two coefficients in order, each little-endian, so the symbols in index order are the row buffer read as `u64` words in memory order, and the honest prover never builds the packed table. The seven low column variables are inside a symbol: variables 0 to 5 are the bit within a word and variable 6 selects the coefficient. The high column variable and the `t` cycle variables are the variables of `p`, in that order.

The front end calls the opening with the column point `rho ∈ H^8`, the cycle point `r_6 ∈ H^t` and the 256 column values `C`, after it has absorbed `C` and drawn `rho` (§11 of the protocol spec). The scheme derives

```text
r    = (rho[7], r_6[0], …, r_6[t−1])                             ∈ H^μ
s_b  = (1 + rho[7])·C[b] + rho[7]·C[b + 128]                      b < 128, in H
```

and proves `s_b = Σ_z eq_H(r, z)·bit_b(p[z])` for every `b`. For any `C`, `value() = Σ_{b<128} eq_H(rho[0..7), b)·s_b`, so the 128 equalities give `value() = Bits~(rho, r_6)`. The scheme does not need the bound `8/2^128` of the front end for `rho`: it proves the 128 partial evaluations and not one random combination of them. If some `C[b + 128·g]` differs from the column evaluation of the bound table, `s_b` differs from the partial evaluation unless `rho[7]` is the root of a nonzero polynomial of degree 1, which has probability at most `1/2^128` (derived; `C` is absorbed before `rho` is drawn).

#### 2. The code and the commit phase

**Domain and basis.** The evaluation domain of dimension `d` is `S_d = { F64::from_raw(x) : x < 2^d }`, the span of `β_0, …, β_{d−1}`. Its subspace polynomials are `s_0(X) = X` and `s_{l+1}(X) = s_l(X)·(s_l(X) + s_l(β_l))`; `s_l` vanishes on `S_l` and `s_l(β_l) ≠ 0` because `β_l ∉ S_l`. With `Ŵ_l(X) = s_l(X)/s_l(β_l)`, the basis polynomial of index `w < 2^c` is `X_w(X) = Π_{l<c} Ŵ_l(X)^{w_l}`, of degree `w`. The code of dimension `2^c` on `S_d` is

```text
Enc_{c,d}(f)[x] = Σ_{w<2^c} f[w] · X_w(F64::from_raw(x)),        x < 2^d,
```

for `f` with values in `K`, in `V` or in `E`. It is the Reed-Solomon code of the polynomials of degree below `2^c` on `S_d`, of rate `2^(c−d)`. Every `X_w(x)` is in `K`, so the encoder acts on each coefficient in `K` separately: it maps `V`-valued messages to `V`-valued codewords, and `Enc(pack(a, b)) = pack(Enc(a), Enc(b))`. For a fixed position `x` the map `f ↦ Enc(f)[x]` is the inner product with the vector `W_x[w] = X_w(x)`, whose multilinear extension is a product of `c` factors:

```text
W~_x(q) = Π_{l<c} (1 + q_l + q_l·Ŵ_l(F64::from_raw(x))).
```

**Levels.** Level `i` commits a function `f_i` of `m_i = k_i + c_i` variables: the low `k_i` variables index `2^(k_i)` lanes and the high `c_i` index the message of a lane. Its oracle is the matrix

```text
O_i[x][u] = Enc_{c_i, d_i}(f_i(u, ·))[x],        f_i(u, ·)[w] = f_i[u + 2^(k_i)·w],
```

with `2^(d_i)` positions `x` and `2^(k_i)` lanes `u`. Leaf `x` of its Merkle tree is the concatenation of `O_i[x][u]` for ascending `u`, each symbol in its canonical little-endian bytes: 16 bytes at level 0, where `f_0 = p` and a symbol of `V` is its two coefficients with the coefficient of 1 first, and 24 bytes at every later level. A string of 16 bytes is the encoding of exactly one element of `V`, so a level-0 leaf cannot carry a symbol outside `V`. A leaf digest is BLAKE2s-256 of the leaf bytes, a node is BLAKE2s-256 of its two children's 64 bytes, and the tree has depth `d_i`. The lane index is the low part of the symbol index. At level 0 the message matrix `p[u + 2^(k_0)·w]`, read position by position, is therefore the row buffer itself: position `w` is words `2^(k_0 + 1)·w` to `2^(k_0 + 1)·(w + 1) − 1` of the buffer (16 rows at `k_0 = 5`), lane `u` is words `2u` and `2u + 1` of them, and the encoder reads the buffer without a transposition. The level-0 oracle, as bytes, is the encoding of `2^(k_0 + 1)` lanes of words of `K`, and the transform of level 0 is a transform over `K`.

**Commit phase.** These are the messages of `commit` and `verify_commit`, step 6 of the preamble.

```text
1  P → V   root_0, the root of the tree of O_0                              absorbed
2  V       z_0 ∈ E^(c_0)                                                    c_0 challenges
3  P → V   y_0[u] = Σ_w eq_E(z_0, w)·p[u + 2^(k_0)·w]  ∈ E,  u < 2^(k_0)    absorbed
```

The commitment is `root_0` and the `2^(k_0)` values `y_0`: `32 + 24·32 = 800` bytes at `t = 22`. The sample is one evaluation of every lane's message at a common point, and not one evaluation of `p` at a point of `E^μ`, for a reason of cost: a claim about `p` would join the sumcheck at level 0 with a weight of `2^μ` elements of `E`, while the lane values combine, after the lane variables are folded, into one claim about `f_1`, whose weight has `2^(c_0)` elements (§4, §9). `verify_commit` checks nothing: it absorbs, draws, and keeps `(t, root_0, z_0, y_0)` as its state. What the sample buys is invariant 2. The word in the leaves is within the decoding radius of at most `L_0` codewords of the interleaved code (§8). Two different members differ in some lane, and two different lane messages agree at a uniform point of `E^(c_0)`, drawn after the root, with probability at most `c_0/|E|`, so after step 3 at most one member of the list is consistent with `(z_0, y_0)`, except with probability `C(L_0, 2)·c_0/|E|`. A commitment that is a root alone binds the list, and every error term of the front end would then be paid once per member of the list; at the reference size that is a loss of `log2 249 = 7.96` bits on terms that have no margin (§8).

#### 3. The bridge

The claims `s_b` are in `H`, the symbols in `V`, and the recursion runs in `E ⊃ V`. The bridge replaces the 128 claims by one inner-product claim over `E`, using only that the `ν_b` are independent over `F_2` and that elements of `H` are vectors of 128 bits.

*Transposition.* Put `t_h = Σ_{b<128} ν_b·bit_h(s_b) ∈ V` for `h < 128`: the 128 by 128 matrix of the bits of the `s_b`, read by columns. In words, `t_h = pack(lo, hi)` where bit `b` of `lo` is `bit_h(s_b)` and bit `b` of `hi` is `bit_h(s_(64 + b))`. If the `s_b` are the partial evaluations of `p`, then, because `bit_h` is `F_2`-linear and `bit_b(p[z]) ∈ F_2`,

```text
t_h = Σ_b ν_b · bit_h( Σ_z eq_H(r, z)·bit_b(p[z]) ) = Σ_z bit_h(eq_H(r, z)) · p[z].
```

*Batching.* The verifier draws `α ∈ E` and both sides are combined with the powers of `α`. Let `Φ_α : H → E` be the `F_2`-linear map `Φ_α(e) = Σ_{h<128} bit_h(e)·α^h`. Then

```text
τ = Σ_{h<128} t_h·α^h,        w[z] = Φ_α(eq_H(r, z)),        τ = Σ_z p[z]·w[z].        (bridge)
```

The products `t_h·α^h` and `p[z]·w[z]` are products in `E` of an element of `V` by an element of `E`. The verifier computes `τ` from `C`, `rho[7]` and `α`: 128 products in `H` for the `s_b`, and Horner's rule on `t_127, …, t_0`, 127 products in `E`. The prover sends nothing for the bridge. The identity `(bridge)` is the first claim of level 0.

*Soundness, in outline.* Fix the function `p` selected by the commitment. It is `V`-valued (Lemma 1 of §8). For claimed values `s_b` let `δ_h = t_h + Σ_z bit_h(eq_H(r, z))·p[z] ∈ V`. The two sides of `(bridge)` differ by `Σ_h δ_h·α^h`, a polynomial of degree at most 127 in `α` with coefficients in `E`, and `p`, `C`, `rho`, `r_6` are all fixed before `α` is drawn. If some `δ_h ≠ 0` the identity holds with probability at most `127/|E|`. If every `δ_h = 0`, the coordinate of `δ_h` on `ν_b` gives `bit_h(s_b) = Σ_z bit_h(eq_H(r, z))·bit_b(p[z])` for all `b` and `h`, which is `s_b = Σ_z eq_H(r, z)·bit_b(p[z])` for all `b`. The argument uses that the `ν_b` are independent over `F_2`. It uses no relation between `H` and `E`, and it does not need `V` to be closed under multiplication.

*The weight's extension.* The final check of the recursion needs `w~(q) = Σ_z eq_E(q, z)·w[z]` at a point `q ∈ E^μ`. Work in the ring `E ⊗_{F_2} H`, whose elements are written `Σ_h G[h] ⊗ x^h` with `G ∈ E^128`. The ring is not a field and `Φ_α` does not preserve products; the product is taken in the ring first and the `E`-linear map `G ↦ Σ_h G[h]·α^h` is applied after. Since `(1+q_l)⊗(1+r_l) + q_l⊗r_l = (1+q_l)⊗1 + 1⊗r_l`,

```text
Σ_z eq_E(q, z) ⊗ eq_H(r, z) = Π_{l<μ} ( (1 + q_l)⊗1 + 1⊗r_l ),
```

and `w~(q)` is the image of this product under that map. The verifier evaluates it by a recurrence on `G`, starting from `G = (1, 0, …, 0)`:

```text
for l in 0..μ:    G ← (1 + q_l)·G + M_{r_l}·G,        (M_r·G)[h] = Σ_{h'} bit_h(r·x^(h')) · G[h'];
w~(q) = Σ_h G[h]·α^h.
```

`M_r` is the 128 by 128 matrix over `F_2` of multiplication by `r` in `H`; its column `h'` is `r·x^(h')`, obtained from the previous column by `mul_x`. One step is 128 multiplications in `E` and at most `128·128` additions in `E`, half of that for a uniform `r`. At `μ = 23` the recurrence is 2,944 multiplications and at most 376,832 additions, and the final combination 127 multiplications by Horner's rule (counted).

The prover evaluates `Φ_α` with 16 tables of 256 entries of `E`, one per byte of the argument: `16·256·24 = 98,304` bytes, built from the powers of `α` with 16·255 additions each of one entry to another.

#### 4. The opening

Let `R` be the number of levels, `o_i = k_0 + … + k_{i−1}` the number of variables bound before level `i`, and `res = c_{R−1}` the number of variables of the final message; `m_i = μ − o_i` and `m_{i+1} = c_i`. The opening is one sumcheck of `μ` rounds for a claim `σ = Σ_z f(z)·ω(z)` whose weight `ω` grows by new terms at the start of each level. The round polynomial has degree 2.

```text
0   V       s_b, t_h from C and rho[7]; α ∈ E; τ                    1 challenge
for i = 0 .. R−1:
a   V       λ_i ∈ E, for i ≥ 1                                      1 challenge
            σ ← σ + Σ_{j≥1} λ_i^j·v_j over the new claims (v_j, ω_j) of level i, in order
            (at i = 0 there is no λ_0 and no new claim:  σ ← τ)
b   for each of k_i rounds:
    P → V   u_0, u_2 ∈ E                                            absorbed
    V       a ∈ E;  σ ← u_0 + (σ + u_2)·a + u_2·a^2                 1 challenge
            (the k_i challenges are a^(i) ∈ E^(k_i))
c   if i < R−1:
    P → V   root_{i+1}, the root of the tree of O_{i+1}             absorbed
    V       z_{i+1} ∈ E^(c_i)                                       c_i challenges
    P → V   y_{i+1} = f~_{i+1}(z_{i+1})                             absorbed
    else:
    P → V   f_R, 2^res elements of E                                absorbed
d   V       Q_i positions x of level i                              4·Q_i bytes
    P → V   the leaves O_i[x][·] at the distinct positions, and a Merkle multiproof
e   V       checks the multiproof against root_i;
            c_x = Σ_u eq_E(a^(i), u)·O_i[x][u] for each distinct x;
            if i < R−1: the new claims of level i+1 are (y_{i+1}, eq_E(z_{i+1}, ·)),
                        then (c_x, W_x) for the distinct x in ascending order,
                        then, for i = 0, the commit sample (Σ_u eq_E(a^(0), u)·y_0[u], eq_E(z_0, ·));
            else:       checks c_x = Σ_w f_R[w]·W_x[w] for each distinct x,
                        and, for i = 0, checks f~_R(z_0) = Σ_u eq_E(a^(0), u)·y_0[u]
closing:
    for each of res rounds:  P → V  u_0, u_2;  V  a ∈ E;  σ as in b
final:
    V       checks σ = f~_R(q[μ−res..μ)) · Ω(q),   q ∈ E^μ the μ round challenges in order
```

`f_{i+1}[w] = Σ_u eq_E(a^(i), u)·f_i[u + 2^(k_i)·w]` is the fold of `f_i` over its lane variables, an honest `f_{i+1}` has `c_i` variables, and the encoding is linear, so `c_x = Enc(f_{i+1})[x] = Σ_w f_{i+1}[w]·W_x[w]`: a query of level `i` is an inner-product claim on the next message, and it joins the sumcheck with the others. In the round message `u_0` and `u_2` are the constant and the quadratic coefficient; the linear one is `σ + u_2`, because the two evaluations at 0 and 1 sum to `σ`. A position is drawn with replacement; repeated positions are opened once and give one claim. When `Q_i` of §5 is "all", no position is drawn and every position is opened, in ascending order.

The total weight at the end is

```text
Ω(q) = w~(q)
     + Σ_{i=1}^{R−1} ( λ_i·eq_E(z_i, q[o_i..μ)) + Σ_{j=2}^{n_{i−1}+1} λ_i^j·W~_{x_j}(q[o_i..μ)) )
     + λ_1^(n_0 + 2)·eq_E(z_0, q[k_0..μ)),                          the last term only if R ≥ 2,
```

where `x_2 < x_3 < …` are the `n_{i−1}` distinct positions queried at level `i−1` and `W~_x` is taken with `c = c_{i−1}` on the domain of dimension `d_{i−1}`. The commit sample is a claim about `f_1`: the lane values `y_0[u]` are evaluations of the lanes of `f_0` at `z_0`, and `f_1` is their combination with `eq_E(a^(0), ·)`, so `f~_1(z_0) = Σ_u eq_E(a^(0), u)·y_0[u]`. It is the last claim of level 1, or, when level 0 is the only level, an equation on the final message. A weight that enters at level `i` is a function of the variables still free at that level, so it is evaluated at the suffix of `q` and carries no factor for the earlier coordinates. The verifier's work is the recurrence of §3, one equality polynomial per level, `c_{i−1}` multiplications of an element of `E` by an element of `K` per query claim after the `Ŵ_l(x)` have been computed in `K`, the lane combinations `c_x`, and the hashes of §9.

The opened leaves are not absorbed. They are fixed by a root that is absorbed before the positions are drawn, up to a collision of the hash, which the ledger prices separately.

#### 5. Parameters

The schedule is a function of `μ = t + 1`, for every `t` that the front end admits, `1 ≤ t ≤ 32`:

```text
k_0 = min(5, μ − 2),   c_0 = μ − k_0,   d_0 = c_0 + 1;
while c_i ≥ 6:   k_{i+1} = 4,   c_{i+1} = c_i − 4,   d_{i+1} = d_i − 1;
R = number of levels,   res = c_{R−1} ∈ {2, 3, 4, 5}.
```

Level 0 has rate 1/2 and each later level has a rate 8 times lower, since the message shrinks by 16 and the domain by 2. At `t = 1`, `k_0 = 0`: level 0 has one lane and no fold round, and its only message after the commitment is the final one. At `t = 22`: `μ = 23`, `R = 5`, `res = 2`.

| Level | `k_i` | `c_i` | `d_i` | Rate | Symbol | Leaf bytes | `Q_i` |
|---|---:|---:|---:|---|---|---:|---:|
| 0 | 5 | 18 | 19 | 1/2 | `V` | 512 | 259 |
| 1 | 4 | 14 | 18 | 1/16 | `E` | 384 | 65 |
| 2 | 4 | 10 | 17 | 1/128 | `E` | 384 | 37 |
| 3 | 4 | 6 | 16 | 1/1,024 | `E` | 384 | 26 |
| 4 | 4 | 2 | 15 | 1/8,192 | `E` | 384 | 20 |

There is no grinding at any level. The query counts are constants of the verifier, one row per `t`:

| `t` | `Q_0, Q_1, …` | | `t` | `Q_0, Q_1, …` |
|---:|---|---|---:|---|
| 1 to 9 | all | | 21 | 259, 65, 37, 26 |
| 10 | all, 59 | | 22 | 259, 65, 37, 26, 20 |
| 11 | 254, 62 | | 23 | 260, 65, 37, 26, 20 |
| 12 | 256, 63 | | 24 | 260, 65, 37, 26, 20 |
| 13 | 257, 64 | | 25 | 261, 65, 37, 26, 20 |
| 14 | 257, 64, 35 | | 26 | 262, 65, 37, 26, 20, 16 |
| 15 | 257, 64, 36 | | 27 | 262, 65, 37, 26, 20, 17 |
| 16 | 258, 65, 37 | | 28 | 263, 65, 37, 26, 20, 17 |
| 17 | 258, 65, 37 | | 29 | 264, 65, 37, 26, 20, 17 |
| 18 | 258, 65, 37, 25 | | 30 | 266, 65, 37, 26, 21, 17, 14 |
| 19 | 258, 65, 37, 26 | | 31 | 267, 65, 38, 26, 21, 17, 14 |
| 20 | 259, 65, 37, 26 | | 32 | 268, 66, 38, 27, 21, 17, 15 |

The table is derived by the rule of §8: for each level in order, with `η = √ϱ/m` and `m` running over the integers from 3 up to the last one whose fold term is at most `2^-128`, among the `m` that keep the sample and batching terms of the level at most `2^-128`, the smallest `m` that attains the fewest queries `Q = ⌈128 / log2(1/(√ϱ + η))⌉`, and "all" where that count reaches the number of positions. The batching term of a level uses the query count already chosen for the level before it. The derivation ran outside the repository in decimal arithmetic at 80 digits (computed); item 1 of Execution re-derives every entry with exact rational bounds and freezes the integers. The query condition needs no logarithm: with `η = √ϱ/m` it is `ϱ^Q·((m + 1)/m)^(2Q) ≤ 2^-256`, an inequality of integers after cross-multiplication. The verifier contains the integers and not the rule (invariant 8).

#### 6. Transcript, challenges and wire format

**Absorption.** In the notation of §13 of the protocol spec, `L(l)` is a label and `B(s)` one `append_bytes` call. The scheme's six labels are `whir_commit`, `whir_ood`, `whir_open`, `whir_round`, `whir_root` and `whir_final`.

```text
commit    L("whir_commit") B(root_0);   then z_0;   L("whir_ood") B(y_0)
open      L("whir_open");               then α
level i   λ_i
          per round:  L("whir_round") B(u_0 ‖ u_2);   then a
          i < R−1:    L("whir_root") B(root_{i+1});   then z_{i+1};   L("whir_ood") B(y_{i+1})
          i = R−1:    L("whir_final") B(f_R[0] ‖ … ‖ f_R[2^res − 1])
          then the positions of level i
closing   per round:  L("whir_round") B(u_0 ‖ u_2);   then a
```

An element of `E` is absorbed as its 24 canonical bytes and a root as its 32 bytes. Nothing of the opening request is absorbed again: the front end has absorbed `C` and has drawn `rho` and `r_6` from the same transcript, so its state binds them when `whir_open` is absorbed. `y_0` is absorbed as the concatenation of its `2^(k_0)` elements in lane order, in one call. The scheme absorbs nothing after the last closing round.

**Challenges.** An element of `E` is one call of `squeeze_bytes` for 24 bytes: two draws of 16 bytes, their little-endian encodings concatenated, the first 24 bytes kept and read as three little-endian `u64` coefficients. Vectors are drawn coordinate by coordinate in index order. The positions of level `i` are one call of `squeeze_bytes` for `4·Q_i` bytes; position `j` is the little-endian `u32` at bytes `4j..4j+4`, reduced to its low `d_i` bits. Since `2^(d_i)` divides `2^32` the positions are uniform and independent. `d_i ≤ 29` for every admitted `t`.

**Wire.** The commitment is `root_0 ‖ y_0[0] ‖ … ‖ y_0[2^(k_0) − 1]`, `32 + 24·2^(k_0)` bytes. The opening proof is the concatenation, in protocol order, of:

```text
for each level i:
    k_i rounds, each u_0 ‖ u_2                                    48 bytes per round
    i < R−1:  root_{i+1} ‖ y_{i+1}                                56 bytes
    i = R−1:  f_R                                                 24·2^res bytes
    n_i   as u32 little-endian, the number of distinct positions
    n_i leaves, in ascending position order                       2^(k_0)·16 at level 0, 2^(k_i)·24 after
    g_i   as u32 little-endian, the number of sibling digests
    g_i digests                                                   32 bytes each
res closing rounds, each u_0 ‖ u_2                                48 bytes per round
```

The multiproof is the standard one. With the distinct positions as the known nodes of the leaf layer, each layer is processed from the leaves up and, within a layer, from left to right: a known node whose sibling is not known takes the next digest of the list as that sibling. The digests are therefore in the order in which the verifier consumes them, and `g_i` is a function of the positions. `read` checks `1 ≤ n_i ≤ min(Q_i, 2^(d_i))`, `g_i ≤ n_i·d_i` and that the length of the string is exactly the sum of the parts, before it allocates; `verify_opening` checks that `n_i` and `g_i` are the counts that the drawn positions determine and rejects otherwise. No position, no claim value `c_x`, no bridge value and no nonce is on the wire.

At `t = 22` the fixed part is `24·48 + 4·56 + 4·24 + 5·8 = 1,512` bytes. The query part is at most `Σ Q_i·(leaf + 32·d_i) = 291,200 + 62,400 + 34,336 + 23,296 + 17,280 = 428,512` bytes, when no two paths share a node, and its expectation over the positions is 324,740 bytes (counted: a node at height `h` of a tree of depth `d` is on a queried path with probability `π_h = 1 − (1 − 2^(h−d))^Q`, and the expected number of digests is `Σ_{h<d} 2^(d−h)·π_h·(1 − π_h)`, which gives 2,623, 721, 403, 271 and 196 digests for the five levels). The opening proof is thus 326,252 bytes in expectation and at most 430,024, and the whole proof of the experiment is the front end's 15,024 bytes, its 26-byte envelope, the 1,568-byte commitment and the opening.

#### 7. The opening field: two designs

`H = F_2[x]/(x^128 + x^7 + x^2 + x + 1)` and `E = K[y]/(y^3 + y + 1)` have degrees 128 and 192 over `F_2`, and 128 does not divide 192, so `H` is not a subfield of `E`. The claim is in `H`. Two designs answer this.

**Design (a): `E = F192` and the 64 by 128 bridge.** This is the design of §1 to §6. Symbols are in `K`, 64 bits each, `2^24` of them at `t = 22`. Arithmetic in `E` exists in `jolt_field::binary` with its accumulator. Counted in carry-less multiplications of 64-bit words on aarch64, from the field code: a product in `K` is 3 (one product, a reduction of two); an element of `E` times an element of `K` is 9 (three products, three reductions), or 3 into an accumulator; a product in `E` is 12 (six products, three reductions), or 6 into an accumulator; a product in `H` is 6.

**Design (b): an opening field that contains `H`.** The smallest field that contains `H` and has at least `2^192` elements is the quadratic extension `E' = H[v]/(v^2 + v + x^121)`, with `2^256` elements. The polynomial is irreducible because the absolute trace of `x^121` in `H` is 1, checked by direct computation of `Σ_{k<128} (x^121)^(2^k)` outside the repository; it is the lowest power of `x` with trace 1. `E'` does not exist in the repository. A product in `E'` is three products in `H`, 18 carry-less multiplications, or 12 into an accumulator with a reduction of 4; an element of `E'` times an element of `H` is 12, or 8 into an accumulator; an element is 32 bytes. For symbols and challenges to multiply cheaply the code alphabet must be a subfield of `E'` in its representation, and the repository's `K` is not: `H` has a subfield of `2^64` elements, but it is not the set of words below `2^64` of `F128`, so a symbol of `K` would enter `E'` through a 64 by 64 matrix over `F_2`, or the code would run over that subfield with the arithmetic of `H` at twice the cost of `K` per butterfly. Design (b) therefore packs 128 bits to a symbol of `H`: `2^23` symbols at `t = 22`, level 0 with 32 lanes (`k_0 = 5`) and the same `c_0 = 18`, a domain inside `H`, and the same later schedule.

Design (b) does not remove the bridge. The bridge exists because bits are packed into symbols, not because of the missing embedding: the claims about a packed table are claims about its 128 bit-slices, and reducing them to one claim on symbols needs the bits of `eq_H(r, z)` in either design. In (b) it has shape 128 by 128, the weight is `Φ'_α(eq_H(r', z))` with `α ∈ E'` and `r' = (rho[7], r_6)`, and the verifier's recurrence is the same with `V ∈ E'^128`. What containment adds is a second form of the verifier's evaluation through the Frobenius automorphism of `E'` over `F_2`, which this spec does not use.

Counts at `t = 22`, for the prover algorithm of §9 in both designs:

| Quantity | (a) `K`, `E = F192` | (b) `H`, `E' = F256` |
|---|---:|---:|
| Symbols at level 0, leaf bytes | `2^24`, 512 | `2^23`, 512 |
| Level-0 butterflies × multiplications each | 301,989,888 × 3 | 150,994,944 × 6 |
| Later-level butterflies × multiplications each | 43,515,904 × 9 | 43,515,904 × 12 |
| Commit sample, multiplications | `3·2^24` = 50,331,648 | `8·2^23` = 67,108,864 |
| Sumcheck round 1, multiplications | `39·2^23` = 327,155,712 | `56·2^22` = 234,881,024 |
| Sumcheck later rounds, multiplications | `36·(2^23 − 1)` = 301,989,852 | `56·(2^22 − 1)` = 234,880,968 |
| Induced weights, 7,405,568 butterflies | × 9 = 66,650,112 | × 12 = 88,866,816 |
| **Carry-less multiplications, total** | **2,043,740,124** | **2,053,898,184** |
| Lookups of `Φ` (entry bytes) | 268,435,456 (24) | 134,217,728 (32) |
| BLAKE2s compressions, all trees | 8,159,227 | 9,142,267 |
| Later-level leaf bytes; query bytes at most | 384; 428,512 | 512; 447,456 |
| Weight vector; folded message after round 1 | 384 MiB; 192 MiB | 256 MiB; 128 MiB |
| Later codewords | 180 MiB | 240 MiB |
| Verifier bridge recurrence | 3,072 products in `E` | 2,944 products in `E'` |
| Proximity-gap term, weakest level | `2^-129.33` | `2^-193.33` |
| Query counts | 260, 65, 37, 26, 20 | the same to within 4 at level 0 |
| Field code that exists; measured implementation | all; yes, for `K` and `E` | none of `E'`; none |

The rows follow from §9: level 0 encodes `2^(k_0)` lanes with `c_0·2^(d_0 − 1)` butterflies each, so 64 lanes in `K` and 32 lanes in `H` cost the same 905,969,664 multiplications; later levels have the same butterfly counts and pay the wider field; the sumcheck of (b) runs over half as many symbols at a wider field. The level-0 tree is identical; the later trees of (b) hash 512-byte leaves, 8 compressions instead of 6 for each of 491,520 leaves. Query counts are set by the rates: with the slack going to zero the first level needs `128/log2(√2) = 256` queries, so a larger field buys at most 4.

Soundness. In (a) every algebraic term is below `2^-128` (§8). In (b) each gains 64 bits, and none of them is the weakest term in (a): the query terms are, and they do not depend on the field. The front end's terms are in `H` in both designs.

**Recommendation, for the owner to confirm: design (a).** The deciding count is the total of carry-less multiplications, 2,043,740,124 against 2,053,898,184, a difference of 0.5%: an opening field that contains `F128` does not make the prover cheaper in field arithmetic. What it does buy is 134,217,728 fewer table lookups and 192 MiB less at the peak, against 983,040 more hash compressions, 18,944 more proof bytes, a field type with its kernels and accumulator that does not exist, and a recursion in which no kernel of the measured implementation applies. No ledger term that is binding improves. The recommendation reverses if the lookups of `Φ` are measured to dominate the opening after the bridge is implemented and a kernel for `E'` is available at the product counts above.

#### 8. Security ledger

**Statement.** The scheme is analysed as an interactive protocol with round-by-round knowledge soundness: after each verifier challenge, the probability that a state with no valid witness becomes one with a valid witness is at most the error of that round. The target is an error of at most `2^-128` for every round of the scheme. This is a statement about the maximum over rounds and not about their sum. After the Fiat-Shamir transformation in the random-oracle model, a prover that makes `q` queries to the transcript's hash succeeds with probability at most `q` times the largest round error, plus the Merkle term. The ledger is a list of bounds under stated assumptions; nothing in it is a proof of security of the implementation.

**Assumptions.**

1. The proximity-gap theorem for Reed-Solomon codes up to the Johnson bound, in its list form for mutual correlated agreement (Ben-Sasson, Carmon, Haböck, Kopparty and Saraf, 2025, Theorem 4.6), with the constant below. It is proven and uses no conjecture on list sizes beyond the Johnson bound. It is applied to the code over `E` on the domain `S_d ⊂ K`; the theorem places no condition on the domain.
2. The round-by-round analysis of the WHIR recursion with interleaved codes, with the error terms listed below. It is taken from the analysis that accompanies the public implementation named in §10; the modifications of this spec (the commit sample, the bridge, the direct check of the last level) are argued here.
3. BLAKE2s-256 and the transcript's hash are random oracles.

**Quantities, per level.** `n = 2^d`, `ϱ = (2^c − 1)/2^d` (the rate parameter of the theorem for dimension `2^c`), slack `η`, radius `γ = 1 − √ϱ − η`, list bound `L = 1/(2·η·√ϱ)` (Johnson; the same for the interleaved code, since interleaving keeps the distance), `m = max(⌈√ϱ/η⌉, 3)`, and

```text
a = ( 2·(m + 1/2)^5 + 3·(m + 1/2)·γ·ϱ ) / (3·ϱ^(3/2)) · n + (m + 1/2)/√ϱ,        ε = a/|E|.
```

**Terms.** `J_i` is the number of claims batched at level `i ≥ 1`, the residual included: `J_1 = n_0 + 3` and `J_i = n_{i−1} + 2` after it, with `n_{i−1} ≤ Q_{i−1}`. Level 0 has one claim and no batching.

| Round | Error | What fails otherwise |
|---|---|---|
| Commit sample `z_0` | `C(L_0, 2)·c_0/\|E\|` | two members of the level-0 list agree at `z_0` in every lane |
| Lane combination of the commit sample | `L_0·k_0/\|E\|` | a list member that disagrees with `y_0` in some lane agrees after the combination with `eq_E(a^(0), ·)` |
| Bridge `α` | `127/\|E\|` | §3 |
| Batching `λ_i` | `(J_i − 1)·L_i/\|E\|` | a false claim cancels in the combination, for some list member |
| Fold round `j ≤ k_i` of level `i` | `2·L_i/\|E\| + 2^(k_i − j)·ε_i` | a sumcheck round of degree 2, or correlated agreement of the partial fold |
| Sample `z_{i+1}` | `C(L_{i+1}, 2)·c_i/\|E\|` | as the commit sample, for level `i+1` |
| Positions of level `i` | `(1 − γ_i)^(Q_i)` | every queried column agrees with a word that is `γ_i`-far |
| Closing round | `2/\|E\|` | a sumcheck round of degree 2 |
| Merkle | `q^2/2^257` for `q` hash queries | a collision of BLAKE2s-256 |

At the reference size, with `|E| = 2^192`, as `−log2` of the error (computed, in double precision):

| Level | `ϱ` | `m` | `η` | `γ` | `L` | `Q` | Positions | Fold, `j = 1` | Sample | Batching | `2L/\|E\|` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `(2^18−1)/2^19` | 187 | 0.003781 | 0.28911 | 187.0 | 260 | 128.00 | 129.33 | 173.74 | none | 183.45 |
| 1 | `(2^14−1)/2^18` | 47 | 0.005319 | 0.74469 | 376.0 | 65 | 128.03 | 137.74 | 171.72 | 175.41 | 182.45 |
| 2 | `(2^10−1)/2^17` | 35 | 0.002524 | 0.90913 | 2,242.2 | 37 | 128.02 | 136.33 | 166.93 | 174.82 | 179.87 |
| 3 | `(2^6−1)/2^16` | 16 | 0.001938 | 0.96706 | 8,322.0 | 26 | 128.02 | 138.33 | 163.63 | 173.73 | 177.98 |
| 4 | `(2^2−1)/2^15` | 5 | 0.001914 | 0.98852 | 27,306.7 | 20 | 128.89 | 142.17 | 160.94 | 172.51 | 176.26 |

The bridge is at 185.01 bits, the lane combination of the commit sample at 181.87 and a closing round at 191. The "Fold" column is the correlated-agreement part at its worst round, `2^(k_i − 1)·ε_i`; the sample of row `i` is the one that binds level `i` (the commit sample for row 0). The weakest term is the positions of level 0, `(1 − 0.28911)^260 = 2^-128.0006`, and the weakest algebraic term is the fold of level 0 at `2^-129.33`. Each `η` is the largest slack of the form `√ϱ/m` for which the query count is minimal while the fold, sample and batching terms of the level stay below `2^-128`. Every term of the scheme meets the target with no grinding.

**Why the list is over `E` and the table is still a table of bits.** The fold challenges are in `E`, so the relevant code is the one over `E`, and the word in the leaves of level 0, which is `K`-valued because a leaf symbol is 8 bytes, can be close to codewords whose messages are not `K`-valued. The commit sample selects at most one message `f` with values in `E`. The table that the scheme binds is the table of bits of the component of `f` in `K`, which is a function of `f` and so fixed at commit time, and §3 shows that an accepted opening proves the claims about that table.

**The direct check of the last level.** The claims of the last level's queries are checked against the final message, which is sent in the clear, and not batched; this removes the batching round that they would need and adds no term.

**What the field of the front end bounds.** Every challenge of the front end is in `H`, and each of its rounds has an error of the form `δ/2^128` with `δ ≥ 1`: the bound `8/2^128` for `rho` is `2^-125`, and a sumcheck round of degree `δ` is `δ/2^128`. A larger opening field does not strengthen any of them. The commit sample keeps them from being multiplied by the list size, and that is all the scheme does for them. An experiment that wants 128 bits from the front end needs challenges from a field larger than `H` in the front end, which is outside this spec.

**Grinding.** There is none. Grinding `g` bits before the positions of a level are drawn does not change that round's error; it multiplies the work of each attempt at that round by `2^g`. Counted as error it would let the query counts drop to 225, 56, 32, 23, 17 at `g = 17` with the position terms at `2^-111.0`, saving 57,760 of the 428,512 bytes at most (41,784 of 332,112 bytes measured on the existing implementation), for an expected `5·2^17` hash evaluations of grinding. It buys nothing for the fold, sample, batching, bridge or front-end terms. It is not adopted, so that the target is met as an error bound.

**What remains below the target.** Not the scheme's rounds. Below it are: the front end's terms, as above; the Merkle term, which is a statement about work and reaches `2^-128` only for `q ≤ 2^64.5` hash queries, a collision costing about `2^128` evaluations; and the multiplication by `q` of the Fiat-Shamir transformation, under which a round error of `2^-128` is security against `2^128/q`-fold advantage and not an error of `2^-128` after `q` attempts.

#### 9. The prover

Counts are in carry-less multiplications of 64-bit words on aarch64, read from `jolt_field::binary`: a product in `K` is 3 (one product, a reduction of two); an element of `E` times an element of `K` is 9 (three products, three reductions), or 3 into an accumulator; a product in `E` is 12 (six products, three reductions), or 6 into an accumulator; a product in `H` is 6 (four products, a reduction of two). An element `e` of `E` times `pack(a, b)` is `e·a + (y·e)·b`, two products of `E` by `K` and a multiplication by `y` that is a permutation and one addition of coefficients: 6 into an accumulator and 12 reduced. Every count below is counted from these.

**`commit`.** (1) Allocate the level-0 codeword, `2^(d_0)` positions of `2^(k_0)` symbols, `2^24` symbols, `2^25` words and 256 MiB at `t = 22`. For each position `w < 2^(c_0)` copy the `2^(k_0 + 1)` words of the row buffer into position `w` of each of the two cosets of the message (the rate is 1/2), and run the additive transform position-major, each butterfly acting on a whole row of words with one twiddle in `K`: `2^(k_0 + 1)·c_0·2^(d_0−1) = 64·18·2^18 = 301,989,888` butterflies, each one multiplication in `K` and two additions. (2) Hash `2^(d_0)` leaves and the tree: `2^19` leaves of 512 bytes are 8 compressions each and the `2^19 − 1` nodes one each, 4,718,591 compressions over 301,989,824 bytes. (3) Absorb the root, draw `z_0`, build the table `eq_E(z_0, ·)` of `2^(c_0)` elements of `E` (6 MiB, `2^18` products in `E`, `12·2^18 = 3,145,728` multiplications), and compute the `2^(k_0)` values `y_0` in one pass over the rows with one accumulator per word of a position: position `w` adds `eq_E(z_0, w)` times word `e` to accumulator `e`, one product of an element of `E` by an element of `K` into an accumulator per word, `3·2^24 = 50,331,648` multiplications, and `y_0[u]` is accumulator `2u` reduced plus `y` times accumulator `2u + 1` reduced.

**State between `commit` and `open`.** `ProverState` holds the `Arc` of the rows, the level-0 codeword, the level-0 tree, `z_0`, `y_0` and `t`. At `t = 22`: `2^25·8 = 268,435,456` bytes of codeword, `(2^20 − 1)·32 = 33,554,400` bytes of tree and `18·24 + 32·24 + 8 = 1,208` bytes of scalars, 301,991,064 bytes, plus the shared reference to the 128 MiB of rows, which the scheme does not own. `open` consumes the state.

**`open`, round 1.** Round 1 pairs the symbols `2k` and `2k + 1`, which are 32 adjacent bytes of the row buffer, for `k < 2^(μ−1)`. Let `e'[k] = eq_H(r[1..μ), k)`, the product of two entries of split tables of `2^11` and `2^(μ−12)` elements of `H`: one product in `H` per pair. The two `eq_H` values of the pair are `r[0]·e'[k]` at `2k + 1` and `e'[k] + r[0]·e'[k]` at `2k`. The map `e ↦ Φ_α(r[0]·e)` is `F_2`-linear in `e` like `Φ_α`, so both are read from one set of 16 tables of 256 entries, one table per byte of `e'[k]`, whose entry for a byte value is the pair of the two maps on that byte: `16·256·48 = 196,608` bytes. With `D[k] = Φ_α(e'[k])` and `W1 = Φ_α(r[0]·e'[k])` the sums of the 16 entries, the prover stores

```text
D[k] = w[2k] + w[2k+1],        W0[k] = w[2k] = D[k] + W1,
```

two vectors of `2^(μ−1)` elements of `E`, and no product by `r[0]` is computed per pair. The first round message is `u_0 = Σ_k p[2k]·W0[k]` and `u_2 = Σ_k (p[2k] + p[2k+1])·D[k]`, two products of an element of `V` by an element of `E` into accumulators per pair. After the challenge `a`, the message folds to `p[2k] + a·(p[2k] + p[2k+1])`, one reduced product of `E` by `V`, written to a new vector, and the weight to `W0[k] + a·D[k]`, one product in `E`, in place in `W0`. Per pair: `6 + 6 + 6 + 12 + 12 = 42` multiplications and 16 table entries of 48 bytes, priced as 32 lookups of 24 bytes, over `2^22` pairs: 176,160,768 multiplications and 134,217,728 lookups. The tables are set up once per opening from the 128 powers of `α` (127 products in `E`), the 128 values `r[0]·x^h` (127 calls of `mul_x`), their images under `Φ_α` (at most `128·128` additions in `E`) and `2·16·255` additions of one entry to another: under 25,000 operations in `E`, charged as zero time against `2^22` pairs. Level 0 has no other weight: the commit sample enters at level 1, where its equality table has `2^18` elements.

**`open`, later rounds.** Round `j ≥ 2` has `2^(μ−j)` pairs of elements of `E`: two products into accumulators for the message (12) and two products for the folds (24), 36 per pair and `36·(2^22 − 1) = 150,994,908` in all. The rounds pair adjacent elements from the first fold on, since the lanes are the low variables; the port takes the source's adjacent-pair rounds and not its dispatch on high-variable lanes or its rotation of the final point. At the start of level `i ≥ 1` the weights of the new claims are added: the equality table of the level's sample (and of the commit sample at level 1), and for the queries the vector `Σ_j λ_i^j·W_{x_j}`, which is the transpose of the encoder applied to the sparse vector with `λ_i^j` at position `x_j`. It is computed with one transposed transform on the domain of level `i−1`, `c_{i−1}·2^(d_{i−1}−1)` butterflies of `E` by `K`: 4,718,592, 1,835,008, 655,360 and 196,608 for the four later levels, 7,405,568 in all. That is the count of a dense transform and an upper bound: the input has at most `Q_{i−1}` nonzero entries, and the port keeps the source's choice, by query support, between the dense pass and a pass over the windows that contain a queried position. The equality tables are `2^18` products in `E` each for `z_1` and `z_0` and `2^14`, `2^10`, `2^6` for the later samples, 541,760 products and 6,501,120 multiplications, with `λ_i` folded into the first entry; merging a new weight into `ω` is additions only.

**`open`, later commitments.** After the `k_i` rounds of level `i < R−1` the folded message is `f_{i+1}`, already in memory. It is encoded with 16 lanes on the domain of dimension `d_{i+1}`: `16·c_{i+1}·2^(d_{i+1}−1)` butterflies of `E` by `K`, 29,360,128, 10,485,760, 3,145,728 and 524,288, in all 43,515,904. Its tree has `2^(d_{i+1})` leaves of 384 bytes, 6 compressions each: 3,440,636 compressions for the four later trees. The sample `y_{i+1}` is an inner product of `2^(c_i)` elements of `E` with the equality table of `z_{i+1}`, products into an accumulator: `6·(2^18 + 2^14 + 2^10 + 2^6) = 1,677,696` multiplications for the four samples. The subspace-polynomial constants and the twiddles of a domain depend on `d` alone; they are computed once per level and geometry and their cost is inside the measured unit `nb1`.

**Memory at `t = 22`.**

| Buffer | Bytes | Lifetime |
|---|---:|---|
| Rows (shared, not owned) | 128 MiB | caller |
| Level-0 codeword | 256 MiB | `commit` to the queries of level 0 |
| Level-0 tree | 32 MiB | the same |
| Weight `w`, `2^24` elements of `E`, folded in place | 384 MiB | `open`, round 1; 192 MiB after it, halving each round |
| Folded message, `2^23` elements of `E` | 192 MiB | from round 1, halving each round |
| Later codewords | 96, 48, 24, 12 MiB | each from its commitment to its queries |
| Later trees | 16, 8, 4, 2 MiB | the same |
| Tables of `Φ_α`, split equality tables | under 1 MiB | `open` |

The peak is in round 1 of `open`: `256 + 32 + 384 + 192 = 864` MiB owned by the scheme, 992 MiB with the rows (counted). By the time the level-1 codeword is allocated the weight and the message hold `2^18` elements each, 12 MiB together.

#### 10. Reuse

**Ported.** The recursion, its level schedule, the per-level parameter search, the additive transform over `F64` and over `F192` with twiddles in `F64`, the batched BLAKE2s leaf and node hashing, the lane-fold and round kernels, and the induction of query weights by the transposed transform exist in the public leanVM repository, in its `crates/pcs` (`whir.rs`, `whir_config.rs`, `whir_ntt_ext.rs`, `whir_induce.rs`, `ntt.rs`, `merkle.rs`). That repository is under the MIT licence, and its files in `crates/pcs` carry per-file credit and copyright lines with the identifier `Apache-2.0 OR MIT`. A port keeps the header of each source file verbatim at the top of every file derived from it, states in that file that it is modified, and adds the MIT licence text of the source with its copyright line to the repository's third-party notices. Jolt is itself under MIT and Apache-2.0, so no licence conflict arises. The port is a port of algorithms onto `jolt_field::binary`, not a dependency: the source has its own field types, transcript, allocator and thread pool.

**Not ported, and why.** The source commits a root alone and calls its commitment list binding; the commit sample is new. Its ring switch is 64 by 192, for a claim in `F192`, with a batching map built from six challenges and the Frobenius automorphism; the claim here is in `F128`, so the bridge of §3 and its prover kernel are new. The source's lanes are the high variables of the message and its level-0 leaf image is lane-descending; here the lanes are the low variables, which removes the packed copy and the transposition at level 0 and keeps the front end's variable order. The source's wire format and transcript are its own. The source's parameter calculator runs in floating point at configuration time; here its output is a frozen table.

**In the repository already.** `F64`, `F128` and `F192` with carry-less kernels and accumulators, `F192: ExtField<F64>` with `mul_base`, `F128::mul_x`, the Blake2b transcript with `squeeze_bytes`, the `blake2` crate (which provides BLAKE2s-256 for the verifier), `rayon`, and the two traits with their contract tests.

**New.** An accumulator method for an element of `F192` times an element of `F64` (three carry-less multiplications, no reduction); the additive transform; the Merkle tree and multiproof; the bridge on both sides; the protocol, its wire format and its parameter table.

**Where the code lives.** The verifier's half is a module `whir` of `jolt-rv64i-verifier`, in safe code, hashing with the `blake2` crate. The prover's kernels (transform, tree, bridge weights, rounds, induced weights) are a new crate, `crates/jolt-rv64i-pcs`, because `jolt-rv64i-prover` and `jolt-rv64i-kernels` forbid `unsafe` and a batched BLAKE2s needs vector intrinsics; the new crate denies `unsafe` outside `src/arch/`, where each use carries a `SAFETY:` comment naming the `cfg` that guarantees the instruction, as `jolt-field` does. `jolt-rv64i-prover` implements `BitsCommitmentProver` in `src/commitment/whir.rs` by calling it.

### Alternatives Considered

**An opening field that contains `F128`.** §7. Rejected on a total of 2,053,898,184 carry-less multiplications against 2,043,740,124, with a field and kernels that do not exist. Reopened by a measurement that the lookups of `Φ` dominate the opening.

**A commitment that is a root alone.** This is what the existing implementation does. It saves the commit sample: 53,477,376 carry-less multiplications, an estimated 16 ms on one thread, and 1,536 bytes. It leaves the commitment binding a list of up to `L_0 = 187` tables, and every round of the front end, whose errors are of the form `δ/2^128`, is then paid once per list member: 7.55 bits lost on each, with no way to recover them in the scheme. Rejected. It would be reopened only by a front end whose challenge field has that margin.

**One sample of the whole table instead of one per lane.** A single value `p~(z_0)` at `z_0 ∈ E^μ` makes the commitment 56 bytes instead of 1,568. Its weight has `2^24` entries at level 0: merged into the bridge weight it costs `12·2^24 = 201,326,592` carry-less multiplications, and carried as a separate term through the six lane rounds about `6·2^23 + 12·(2^23 − 2^18) = 147,849,216`. The per-lane sample costs `12·2^18` in the opening. Rejected on that count; 1,512 bytes are 0.46% of the proof.

**Grinding.** §8. At 17 bits: 57,760 bytes fewer at most, the position terms at `2^-111` as errors. Rejected because the target is an error bound. Reopened if the owner accepts a work factor in place of an error for those rounds.

**Rate 1/4 at level 0.** Halves the queries of level 0 (130 for 260) and doubles the level-0 codeword and tree. On the existing implementation, measured in one sweep on the loaded host at twelve threads: 219,192 bytes against 332,112, and 442.5 ms against 300.0 ms for commit and open together, with 1,611 MiB against 1,326 MiB of resident memory at the peak. Rejected while the prover's time is the binding budget. Reopened if proof size becomes the objective.

**An initial fold of 4 or 8 instead of 6.** Measured in the same sweep: 385.7 ms and 309,056 bytes at 4, 349.1 ms and 676,912 bytes at 8, against 300.0 ms and 332,112 bytes at 6. Rejected on both columns at 8 and on time at 4.

**Ligerito.** The recursion of this spec is of the same family: interleaved codes, partial sumchecks, one new commitment per level. The alternative is its analysis within the unique-decoding radius, with no list and so no commit sample, no samples at later levels and no proximity-gap term. Its position term is `(1 − (1 − ϱ)/2)^Q`: 309 queries at rate 1/2 for 260 and 141 at rate 1/16 for 65, which is `49·1,120 + 76·960 = 127,840` more bytes at most on the first two levels alone. Rejected on proof size. Reopened if the proximity-gap constant of assumption 1 is found not to apply as transcribed, since it is the one assumption that the unique-decoding analysis does not need.

**BaseFold-style folding of the codeword.** Fold the level-0 codeword itself, with one tree per fold of the same schedule and every oracle at rate 1/2. The prover saves every later encoding (43,515,904 butterflies, 65.4 ms measured on one thread) and most of the later hashing. Every query then opens a path in every oracle at the level-0 query count: `260·(1,120 + 864 + 736 + 608 + 480) = 990,080` bytes at most against 428,512, since the oracles have `2^19`, `2^15`, `2^11`, `2^7` and `2^3` leaves. Rejected on proof size, a factor of 2.31. Reopened if the proof is not transmitted or its size does not matter and the 154 ms of later encodings and trees do.

**A wider field for the code.** Symbols in `F128`, 128 bits each, halve the number of symbols and of lookups of `Φ` and leave the level-0 transform at the same 905,969,664 carry-less multiplications (§7). It is only useful with an opening field above `F128`, which is design (b). Symbols in `F192` with 64 bits each, to avoid mixed arithmetic, would triple the level-0 leaf bytes and the 4,194,304 leaf compressions of the first tree. Rejected.

**A weight that is never materialised.** Build `w` twice from the split tables, once for the first round message and once for the fold, and keep no vector of `2^24` elements of `E`: the peak falls by 192 MiB to 672 MiB, and the first round costs a second pass of `12·2^23` carry-less multiplications and 268,435,456 lookups, an estimated 192 ms on one thread at the unit prices of Performance. Rejected on time. Reopened if the prover's memory budget is set below the peak of §9.

## Documentation

The module documentation of `whir` in `jolt-rv64i-verifier` states the protocol of §2 to §6 in the order of the code and links this spec. The contract comment of `commitment.rs` is unchanged. The sentence of `BitsCommitmentProver` that neither phase copies a trace-sized buffer gains the precision of invariant 9: the codeword is a new buffer, the rows are shared. `specs/rv64i-binary-protocol.md` gains, in its performance section, the proof size with this scheme in place of the size "plus the scheme's bytes". No book page changes.

## Execution

Items 1 to 5 have no dependency on one another and touch disjoint files, apart from one `mod` line each in a module root; they are done in parallel. Item 6 needs the verifier halves of 1, 3, 4 and 5; item 7 needs 6 and the prover halves; item 8 needs 7. Item 9 is separate and later. No item leaves a function that is declared and not implemented: each item's files compile and are tested by its own criteria, and no item adds a public function whose body waits for a later item.

1. **Parameters.** `crates/jolt-rv64i-verifier/src/whir/params.rs`: the schedule of §5 as a function of `t`, the table of query counts as constants for `1 ≤ t ≤ 32`. A test re-derives every entry with exact rational arithmetic on certified bounds of the square roots and logarithms (an entry stands if the count is provably sufficient and the count minus one provably insufficient or the choice of `m` provably worse), and asserts every row of the ledger of §8 at most `2^-128` for every `t`. If an entry of §5 is not confirmed, the table in the code and in this spec is corrected in the same PR. Criteria: "Parameters".
2. **Field.** `crates/jolt-field/src/binary/accumulator.rs`: `F192Accumulator::fmadd_base(F192, F64)`. `crates/jolt-field/benches/binary_kernels.rs`: groups that time one carry-less product chain in `F64`, `F192` times `F64` reduced and into the accumulator, `F192` times `F192` reduced and into the accumulator. Criteria: "Field". Measurement obligation: the unit rows `c` and the products of §7, on a quiet host.
3. **Code and transform.** `crates/jolt-rv64i-verifier/src/whir/code.rs`: `Ŵ_l(x)`, `W~_x(q)`, and the encoder by its definition for tests. `crates/jolt-rv64i-pcs/src/ntt.rs`: the position-major transform of level 0 over `F64`, the transform over `F192` with twiddles in `F64`, and its transpose; ported, with the source's headers. Criteria: "Code". Measurement obligation: `nb0` and `nb1`.
4. **Merkle.** `crates/jolt-rv64i-verifier/src/whir/merkle.rs`: leaf and node hashing with the `blake2` crate, the multiproof verifier. `crates/jolt-rv64i-pcs/src/{merkle.rs, arch/}`: the tree builder with batched hashing, and the multiproof writer; ported. Criteria: "Merkle". Measurement obligation: `hb`, and whether a safe scalar hash reaches it, which decides whether `arch/` is needed.
5. **Bridge.** `crates/jolt-rv64i-verifier/src/whir/bridge.rs`: `s_i`, `t_b`, `τ`, the recurrence. `crates/jolt-rv64i-pcs/src/bridge.rs`: the tables of `Φ_α`, the weight and the fused first round. New. Needs item 2 for its price, not for its correctness. Criteria: "Bridge". Measurement obligation: `Lw`, and the first-round phase.
6. **Verifier.** `crates/jolt-rv64i-verifier/src/whir/{mod.rs, wire.rs, verify.rs}`: the type `WhirBits`, its `BitsWire` implementations, `verify_commit`, `verify_opening`. Criteria: "Wire", "Transcript", and the rejections that need no prover (malformed bytes).
7. **Prover.** `crates/jolt-rv64i-pcs/src/{commit.rs, open.rs, rounds.rs, induce.rs}` and `crates/jolt-rv64i-prover/src/commitment/whir.rs`. Criteria: "Completeness", "Rejections", "Determinism", "Front end".
8. **Benchmarks.** `crates/jolt-rv64i-prover/benches/bits_whir.rs` through the runner of `benches/support/`, with the phases of Performance as named phases. Measurement obligations: every row of the phase table on a quiet host at one and twelve threads, the peak of owned memory, the proof size distribution over 1,000 transcripts, and the replacement of the unit rows under the thresholds rule.
9. **Later, separate: the shared opening traits.** `crates/jolt-openings/src/schemes.rs` has one associated `Field` per scheme, sources that are multilinear polynomials over that field, `commit` and `prove` without a transcript between them, and batching that assumes an additive homomorphism. The generalisation gives the scheme trait a typed packed source, a claim field distinct from the opening field, a commit phase and an opening that both take the transcript, and no homomorphism bound on the base trait, with homomorphic batching as an extension trait. It is one self-contained change with `WhirBits` as its first caller, made after item 7, so that the shape of the trait is taken from a working scheme. Nothing in items 1 to 8 depends on it.

Risks carried by the items. The port changes the lane order (item 3 and item 7); the source's round kernels assume lanes in the high variables, and whether they carry over or are rewritten is decided in item 7. The first-round kernel of item 5 is new and is the largest estimated phase. The table of item 1 comes from a double-precision search.

## References

- `specs/rv64i-binary-protocol.md`: §3 (order of variables), §6 and §13 (transcript and bytes), §11 (the commitment contract).
- `specs/rv64i-binary-prover-kernels.md`: the unit table, the thresholds rule and the benchmark runner.
- `specs/binary-field.md`, `specs/binary-accumulators.md`: the fields and their accumulators.
- `crates/jolt-rv64i-verifier/src/commitment.rs`, `crates/jolt-rv64i-prover/src/commitment/mod.rs`: the two traits.
- G. Arnon, A. Chiesa, G. Fenzi, E. Yogev, "WHIR: Reed-Solomon Proximity Testing with Super-Fast Verification".
- B. Diamond, J. Posen, "Polylogarithmic Proofs for Multilinears over Binary Towers" (FRI-Binius; ring switching).
- H. Zeilberger, B. Chen, B. Fisch, "BaseFold: Efficient Field-Agnostic Polynomial Commitment Schemes from Foldable Codes".
- A. Novakovic, G. Angeris, "Ligerito: A Small and Concretely Fast Polynomial Commitment Scheme".
- E. Ben-Sasson, D. Carmon, U. Haböck, S. Kopparty, S. Saraf, "On Proximity Gaps for Reed-Solomon Codes" (2025), Theorem 4.6.
- The public leanVM repository, `crates/pcs` and the commitment-scheme annex of its documentation: the implementation that is ported and the round-by-round analysis of assumption 2.

## Open

1. **The proximity-gap constant.** The formula for `a` in §8 is taken from the analysis that accompanies the ported implementation and has not been checked against the statement of the theorem in the paper. The fold terms of the ledger, and through the search every `η` and query count, depend on it.
2. **The round-by-round theorem with the modifications.** The per-lane commit sample, its lane-combination term and the direct checks of the last level are argued in this spec and are not covered by the source's theorem. A written proof is owed before the ledger is cited outside this experiment.
3. **The table of §5.** Computed in double precision outside the repository. The entry for `t = 22` agrees with the query counts measured on the existing implementation; the others are unconfirmed until item 1.
4. **The hash.** BLAKE2s-256 is kept from the source. Whether BLAKE3 or SHA-256 with hardware support is cheaper at 512-byte and 384-byte leaves on the benchmark host is not measured; a change is a change of the wire format.
5. **Unit prices.** `c` and `Lw` are estimates. The measured rows were taken on a loaded host, on the source and not on this scheme. No phase of the bridge has been measured.
6. **Scaling to twelve threads.** The house factor of 9.6 is not supported by the one measurement available, which gives 4.6 for commit and 5.1 for open on a loaded host with eight performance and four efficiency cores. Whether the cause is the load, memory bandwidth in the transform and the hashing, or the core mix is not known.
7. **The lane order in the port.** Whether the source's round and fold kernels can be used with lanes in the low variables has not been established by reading them through.
8. **The trace of `x^121`.** Computed once by a script outside the repository. It matters only if design (b) is taken up.
9. **Small tables.** For `t ≤ 10` every position of level 0 is opened. The proof is then larger than the table for the smallest `t`, which is accepted for sizes used only in tests; whether the front end ever runs a production instance below `t = 11` is the owner's to say.
