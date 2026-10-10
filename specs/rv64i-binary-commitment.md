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

At the reference size `t = 22` the table has `2^30` bits and `2^23` symbols. The parameter set is rate 1/2, fold schedule 5, 4, 4, 4, 4, query counts 259, 65, 37, 26, 20 and no grinding. Under the proximity-gap theorem for Reed-Solomon codes up to the Johnson bound, in the form transcribed in the analysis that accompanies the public leanVM implementation, and under the invariant of §8, every algebraic and query transition of the scheme has error at most `2^-128` at every admitted `t`; the weakest is the query term of the first level at `2^-128.003`, followed by its proximity-gap term at `2^-128.27`. This is conditional round-by-round soundness of the scheme's own transitions. The hypotheses and constants of the theorem in its primary source, knowledge extraction and the composed protocol are obligations of the Open list; the Merkle and Fiat-Shamir terms depend on query budgets; and the front end over `F128` holds the composed protocol to 125.415 bits per round at the reference layout (§8). The opening proof is 325,532 bytes in expectation and 336,792 at most, and the commitment is 800 bytes (counted). The public leanVM implementation of the same recursion, at the same level-0 oracle and for a claim in `F192`, measures 311 ms to commit and 730 ms to open on one thread and 67.8 ms and 142.6 ms on twelve, on a loaded host. The model of this spec for the scheme with the `F128` bridge and the commit-time sample is 157.2 ns per cycle on one thread (estimated), which gives single-thread thresholds of 94 ns for commit and 103 ns for open, together 197 ns per cycle and 9.4% of the prover's 2,100.

The packing of 128 bits into `V` is taken on operation counts against a packing of 64 bits into `F64` and against an opening field of `2^256` elements that contains `F128` (§7). It is confirmed or reversed by the first measurement of the bridge phase, and §7 states the result that reverses it.

## Intent

### Goal

Fix, completely enough to implement and to audit, the commitment scheme of the bit table: the committed object and the claim, every message of the commit phase and of the opening, every challenge and the bytes it is drawn from, the wire format, the parameter set and its soundness ledger, the prover's operation counts and memory, and the items of work.

Notation used throughout. `K = F64`, `H = F128` and `E = F192` are the types of `jolt_field::binary`. `E` is `K[y]/(y^3 + y + 1)` and `F192` stores its three coefficients in `K` in ascending degree, so `K ⊂ E` is the first coefficient, `mul_base` multiplies each coefficient by an element of `K`, and the canonical 24 bytes are the three coefficients in order; `H` is unrelated to `E`. `V = K + y·K ⊂ E` is the set of elements whose third coefficient is zero. It is a subspace over `K` of dimension 2, closed under addition and under multiplication by `K`, and not closed under multiplication. `pack(a, b) = F192::from_base_fn` of `(F64::from_raw(a), F64::from_raw(b), 0)` is the element `a + y·b` of `V` for two words `a`, `b`. `t = log_T`, and `μ = t + 1` is the number of variables of the packed table. Points are low-variable-first and sumchecks bind the lowest variable first, as in the front end. `eq_F(r, z)` is the equality polynomial over the field `F`, with bit `l` of the index `z` paired with `r[l]`. `β_i = x^i`, `i < 64`, is the basis of `K` over `F_2` in which `F64::from_raw` reads a word, and `ν_b = β_(b mod 64)·y^⌊b/64⌋`, `b < 128`, is the basis of `V` over `F_2`. `bit_h(e)` is bit `h` of the raw representation of an element of `H`; for `v = pack(a, b)`, `bit_b(v)` is bit `b` of `a` for `b < 64` and bit `b − 64` of `b` after it, the coordinate of `v` on `ν_b`. "Level" means one committed oracle of the recursion; level 0 is the commitment itself.

### Invariants

1. **The committed object.** The commitment binds one function `p : {0,1}^μ → V`. For an honest prover `p[g + 2·j] = pack(row_j[2g], row_j[2g + 1])` for `g < 2`, `j < T`, where `row_j : [u64; 4]` is row `j` of the table, so that `bit_b(p[g + 2·j]) = Bits[b + 128·g, j]` for `b < 128`. Index and bit order are those of §3 of the protocol spec and are frozen here.
2. **Selection at commit time.** After `verify_commit` returns, and except with the probability of the ledger row "commit sample" and of a collision of the hash, the commitment selects at most one member of the Johnson list of the level-0 oracle, and possibly none. Every member of that list is `V`-valued (Lemma 1 of §8), so a selected member is one table of bits. An opening is a proof about that member and has its own errors in the ledger of §8: the selection does not make later errors disappear, and a commitment that selects no member is rejected by the opening except with those errors. The mechanism is the out-of-domain evaluation carried in the commitment and checked in the opening (§2, §4 of Architecture).
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
- [ ] The exact-arithmetic test of the parameters item confirms every query count and certifies every algebraic and query row of the ledger (commit sample, bridge, batching, fold, sample, positions, closing) at most `2^-128` for every `t`. It reports, and does not compare with `2^-128`, the collision bound `q_h·(q_h − 1)/2^257` and the compilation bound of §8, which depend on query budgets.

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

- [ ] `read(write(x)) = x` for commitments and openings of every `t ≤ 14` used in the tests; every proper prefix and every extension by one byte of an accepted string is rejected; `n_i = 0`, `n_i > Q_i` and `g_i > n_i·d_i`, and at a level that opens every position `n_i ≠ 2^(d_i)` and `g_i ≠ 0`, are rejected before allocation, shown by a test that feeds lengths of `2^32 − 1`.
- [ ] `read` and the two verifier functions return without panicking on 10,000 random strings and on every single-byte mutation of an accepted proof at `t = 6`.

**Transcript.**

- [ ] A recording transcript sees exactly the calls of §6, in order, for `t = 6` (one level, every position opened: no `λ` and no position bytes), `t = 10` (every position at level 0, drawn positions at level 1) and `t = 11`; the six labels are disjoint from the nine of the front end; the recorded calls and the final transcript states of prover and verifier are equal after `commit` and after `open`. The retained states of the two sides are different types and are not compared.
- [ ] For a fixed transcript state, the positions of a level with `d = 7`, `Q = 5` equal a literal computed in the test from the squeezed bytes by the rule of §6.

**Completeness.**

- [ ] For `t ∈ {1, 2, 5, 6, 9, 10, 11, 14}` and random tables, with `C` and the points produced as the front end produces them, `verify_commit` and `verify_opening` accept the output of `commit` and `open`, and `BitsOpening::value()` equals the evaluation of the table computed bit by bit.
- [ ] The contract tests of the two traits are generic over the scheme and pass for `WhirBits` and for `TransparentBits`: the lifecycle of commit, verify, open and verify from identically initialised transcripts, and `value()` against the evaluation of the table computed bit by bit. They are extracted from the tests of `TransparentBits`, whose checks of its own opening type, its error type and its limit on `log_T` stay with it as tests of that implementation.
- [ ] `verify_commit` returns a typed error for `t = 0`, `t = 33` and a commitment with a wrong number of lane values; `verify_opening` returns a typed error for a request whose geometry differs from the retained one and for column points, cycle points and column vectors of wrong length, without panicking.

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

**Measured: the source at this level-0 oracle.** A separate harness ran the public leanVM implementation at `2^24` words of `F64`, rate 1/2, 64 words to a leaf (its initial fold of 6), later folds of 4, no grinding, query counts 260, 65, 37, 26, 20, BLAKE2s-256, on the benchmark machine while it was loaded with other builds, five runs per configuration, each in a fresh process. The harness runs the source's protocol and kernel operations through a copy of its commitment crate that is instrumented with timers, and it builds against its own dependency lock, resolved offline, which is not the lock of the source's workspace. Medians, in milliseconds:

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
| Commit and open, paired | 207.91 | 1,042.89 |
| Verify | 1.79 | 1.78 |
| Proof bytes (its own serialisation, pruned paths) | 332,112 | 332,112 |
| Peak resident memory, rows and packed copy included | 1,325 MiB | 1,319 MiB |

What these figures do not measure: the bridge of §3, since the harness opens a claim whose point is in `F192` through the source's own 64 by 192 ring switch and computes the 64 slice values, where this scheme reads 128 from `C`; the packing of 128 bits and its products; the commit sample, since the source commits a root alone; the lane order and wire of this spec; and an idle host. The twelve threads were eight performance and four efficiency cores. Medians of phases do not sum to medians of totals: the paired median of commit and open on twelve threads is 207.91 ms, and the two separate medians sum to 210.38. The figures are a loaded-host reference for the source. They are not upper bounds on the port and not measurements of this scheme's bridge.

**Unit prices.** In nanoseconds, added to the table of the kernels spec.

| Symbol | Operation | Point 1 | Point 2 | Source |
|---|---|---:|---:|---|
| `c` | one carry-less multiplication of 64-bit words with its share of the reduction | 0.305 | 0.15 | estimated: M/6 of the kernels spec's unit table, to be measured by item 3 |
| `Lw` | lookup of one element of a 48-byte entry and its XOR, tables of 192 KiB | 0.6 | 0.6 | estimated: 1.5 L, taken from tables of 96 KiB; to be measured by item 6 on the tables of §9 |
| `nb0` | one level-0 butterfly, with allocation and copy | 0.535 | 0.535 | measured on the source, loaded host: 161.58 ms over 301,989,888 |
| `nb1` | one later-level butterfly, with allocation and domain set-up | 1.50 | 1.50 | measured on the source, loaded host: 65.37 ms over 43,515,904 |
| `hb0`, `hb1` | one BLAKE2s compression in a tree of 512-byte, 384-byte leaves | 29.0, 25.7 | the same | measured on the source, loaded host: 136.82 ms over 4,718,591; 88.57 ms over 3,440,636 |

The measured rows are prices of the source's kernels and not of the port's. The source documents its later butterfly as three carry-less multiplications; `mul_base` of `jolt_field` with its reductions is nine, which is 2.75 ns at point 1 and 1.35 at point 2 against the 1.50 measured, and a level-0 butterfly counted as `3c` is 0.92 and 0.45 against the 0.535 measured. The model takes the measured rows, and item 4 replaces them with the port's.

**Model, at `t = 22`, one thread.**

| Phase | Operations (counted) | Point 1, ms | Status |
|---|---|---:|---|
| Level-0 encode | 301,989,888 `nb0` | 161.6 | source unit; includes a transposition that §2 removes |
| Level-0 tree | 4,718,591 `hb0` | 136.8 | source unit |
| Commit sample | `(3·2^24 + 12·2^18) c` = 53,477,376 `c` | 16.3 | estimated |
| **Commit** | | **314.7** | |
| Tables, weights and round 1 | 176,160,768 `c` + 134,217,728 `Lw` | 53.7 + 80.5 = 134.3 | estimated |
| Rounds 2 to 23 | 150,994,908 `c` | 46.1 | estimated |
| Induced weights | at most 7,405,568 butterflies | 7.0 | source phase |
| Equality tables and later samples | `6,501,120 + 1,677,696` = 8,178,816 `c` | 2.5 | estimated |
| Later encodes | 43,515,904 `nb1` | 65.4 | source unit |
| Later trees | 3,440,636 `hb1` | 88.6 | source unit |
| Queries and assembly | 407 positions | 0.8 | source phase |
| **Open** | | **344.5** | |

Per cycle that is 75.03 ns for commit and 82.14 ns for open, 157.17 ns together (estimated). At point 2, where only `c` changes, 73.06 and 69.75. The set-up of the tables of `Φ_α` is under 25,000 operations and is charged as zero (§9); the release points of §9 add no copy. The estimated rows of the opening are 182.8 ms of the 344.5, 53%, and none of them has a measured counterpart: the source's ring switch and rounds take `447.2 + 122.8 = 570.0` ms for a reduction of shape 64 by 192 over `2^24` symbols with the slices computed. This is the least certain part of the model. Without the composed lookup round 1 would cost `48·2^22` multiplications and the opening 83.97 ns per cycle; what the composed lookup has to show in measurement is that tables of 192 KiB with 48-byte entries are read at the price of `Lw`.

**Thresholds.** A threshold is single-thread nanoseconds per cycle at `log_t = 22`, the unrounded model times 1.25 to the nearest nanosecond, as in the kernels spec.

| Benchmark | Model, point 1 | Threshold | Model, point 2 | Threshold at point 2 |
|---|---:|---:|---:|---:|
| `bits_whir/commit` | 75.03 | 94 | 73.06 | 91 |
| `bits_whir/open` | 82.14 | 103 | 69.75 | 87 |

The thresholds of point 1 are in force and are provisional: they are forecasts from a model, not ceilings that a run has shown. They move only when a row of the unit table is replaced by a measurement of the operation that the row names, on a quiet host and on this scheme's code, and then every model and threshold is recomputed in the same PR; no threshold moves to meet a result. Every source row above is to be replaced in that way. At `log_t = 20` the benchmark reports and has no threshold: the schedule has four levels and a different final message, and no measurement exists at that size.

**Twelve threads.** The kernels spec divides a single-thread threshold by 9.6 for twelve threads. This spec does not rely on that factor: acceptance on twelve threads is wall time, measured directly. The one measurement available, of the source, gives 67.8 ms and 142.6 ms, a speed-up of 4.59 for commit and 5.12 for open over its single thread, on a loaded host with eight performance and four efficiency cores. The data cannot separate the load from the memory traffic of the transform and the hashing and from the core mix, and the ratio of the source says nothing about the scaling of this scheme's bridge. The provisional requirement at `log_t = 22` on twelve threads is the single-thread threshold in wall time divided by the ratio measured for the source: `94·2^22 ns / 4.593 = 86` ms for commit and `103·2^22 ns / 5.118 = 84` ms for open (estimated). The benchmark item measures both on a quiet host, and the requirement is then replaced under the rule for unit rows. At 9.6 the figures would be 41 ms and 45 ms; that is an aspiration.

**Budget.** The prover's budget is 2,100 ns per cycle on one core. The scheme takes `94 + 103 = 197` ns of it, 9.4%; with the 745 ns of the kernels' thresholds, 942 ns are allotted and 1,158 remain. The prover's wall-time target of 918 ms at `2^22` cycles on twelve threads is that budget at 9.6 effective cores. The provisional twelve-thread requirement of this scheme, 170 ms, is 18.5% of it where the single-thread share is 9.4%, so meeting the thresholds above does not certify the wall-time target, which is measured on the whole prover. For reference, the source measures 207.9 ms for commit and open together on twelve threads on the loaded host.

**Memory.** Counted, §9: 576 MiB owned at the peak, 704 MiB with the rows, under the allocation lifetimes that §9 specifies; the source measures 1,325 MiB of resident memory with its packed copy. The benchmark reports the peak of owned allocations by capacity, not by length, and the shared rows separately, through the recorder of the bench support. The requirement is that the owned peak does not exceed the counted 576 MiB by more than 5% at `log_t = 22`, which an implementation that only truncates its vectors does not meet (688 MiB, §9).

**Verifier.** Counted at `t = 22`: at most `Σ Q_i·(leaf compressions + d_i) = 10,423` BLAKE2s compressions, 3,072 products in `E` for the recurrence, and 408 lane combinations and weight evaluations. The source verifies in 1.78 ms (measured). The benchmark reports the verifier's time; it has no threshold.

**Proof size.** §6: 325,531.93 bytes expected (computed) and 336,792 at most (counted) for the opening, and 800 for the commitment. The benchmark item reports the distribution over 1,000 transcripts, and the acceptance is that no opening exceeds 336,792 bytes and that the mean is within 1% of the expectation.

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

The commitment is `root_0` and the `2^(k_0)` values `y_0`: `32 + 24·32 = 800` bytes at `t = 22`. The sample is one evaluation of every lane's message at a common point, and not one evaluation of `p` at a point of `E^μ`, for a reason of cost: a claim about `p` would join the sumcheck at level 0 with a weight of `2^μ` elements of `E`, while the lane values combine, after the lane variables are folded, into one claim about `f_1`, whose weight has `2^(c_0)` elements (§4, §9). `verify_commit` validates the geometry, `1 ≤ t ≤ 32`, and the shape of the commitment, `2^(k_0)` values `y_0` for that `t`, and returns a typed error otherwise; then it absorbs, draws, and keeps `(t, root_0, z_0, y_0)` as its state. It makes no algebraic check. `verify_opening` first checks that the geometry of the request is the one retained and that the request has 8 column coordinates, `t` cycle coordinates and 256 column values, and returns a typed error before it indexes any of them. These checks are made on the typed values, since a caller of the two functions does not pass through `read`; offsets and products of lengths use checked arithmetic, and an allocation that fails is a typed error. What the sample buys is invariant 2. The word in the leaves is within the decoding radius of at most `L_0` codewords of the interleaved code (§8), where distance counts the positions at which any lane differs. For each pair of members choose one lane in which their messages differ: the two lane messages agree at a uniform point of `E^(c_0)`, drawn after the root, with probability at most `c_0/|E|`, and agreement in every lane implies agreement in that one. So after step 3 at most one member of the list is consistent with `(z_0, y_0)`, and possibly none, except with probability `C(L_0, 2)·c_0/|E|`. The bound has no factor for the number of lanes, and one point shared by all lanes loses nothing. The point is uniform over all of `E^(c_0)`: a point that falls in `K` or on the Boolean cube is used as drawn. A commitment that is a root alone binds the list, and every error term of the front end would then be paid once per member of the list; at the reference size that is a loss of `log2 249 = 7.96` bits on terms that have no margin (§8).

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

The prover evaluates `Φ_α`, and with it the composed map `e ↦ Φ_α(r[0]·e)`, from 16 tables of 256 entries, one table per byte of the argument (§9).

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
level i   i ≥ 1:      λ_i                                     (level 0 draws no λ)
          per round:  L("whir_round") B(u_0 ‖ u_2);   then a
          i < R−1:    L("whir_root") B(root_{i+1});   then z_{i+1};   L("whir_ood") B(y_{i+1})
          i = R−1:    L("whir_final") B(f_R[0] ‖ … ‖ f_R[2^res − 1])
          Q_i a number:  the positions of level i             (a level with Q_i "all" draws nothing)
closing   per round:  L("whir_round") B(u_0 ‖ u_2);   then a
```

An element of `E` is absorbed as its 24 canonical bytes and a root as its 32 bytes. Nothing of the opening request is absorbed again: the front end has absorbed `C` and has drawn `rho` and `r_6` from the same transcript, so its state binds them when `whir_open` is absorbed. `y_0` is absorbed as the concatenation of its `2^(k_0)` elements in lane order, in one call. The scheme absorbs nothing after the last closing round.

**Challenges.** An element of `E` is one call of `squeeze_bytes` for 24 bytes: two draws of 16 bytes, their little-endian encodings concatenated, the first 24 bytes kept and read as three little-endian `u64` coefficients. Vectors are drawn coordinate by coordinate in index order. The positions of level `i` are one call of `squeeze_bytes` for `4·Q_i` bytes; position `j` is the little-endian `u32` at bytes `4j..4j+4`, reduced to its low `d_i` bits. Since `2^(d_i)` divides `2^32` the positions are uniform and independent. `d_i ≤ 29` for every admitted `t`. A level whose `Q_i` is "all" makes no call: its positions are `0, …, 2^(d_i) − 1`.

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

The multiproof is the standard one. With the distinct positions as the known nodes of the leaf layer, each layer is processed from the leaves up and, within a layer, from left to right: a known node whose sibling is not known takes the next digest of the list as that sibling. The digests are therefore in the order in which the verifier consumes them, and `g_i` is a function of the positions. `read` works in two passes. The first walks the string without allocating: at each level it reads `n_i` and `g_i` at offsets computed with checked arithmetic, checks `1 ≤ n_i ≤ min(Q_i, 2^(d_i))` and `g_i ≤ n_i·d_i` for a level with a numeric `Q_i`, and `n_i = 2^(d_i)` and `g_i = 0` for a level whose `Q_i` is "all", and at the end checks that the length of the string is exactly the sum of the parts. The second pass constructs the proof. A string that `read` accepts is canonical in form; whether it is accepted as a proof is decided by `verify_opening`, which checks that `n_i` and `g_i` are the counts that the drawn positions determine and rejects otherwise. No position, no claim value `c_x`, no bridge value and no nonce is on the wire.

At `t = 22` the fixed part is `23·48 + 4·56 + 4·24 + 5·8 = 1,464` bytes (counted). The query part depends on the positions. A node that spans a fraction `π = 2^(h−d)` of the leaves, at height `h` of a tree of depth `d`, is sent as a sibling digest when no position falls under it and one falls under its sibling, which for `Q` draws has probability `(1 − π)^Q − (1 − 2π)^Q`; the two events are not independent and are not multiplied. So

```text
E[g] = Σ_{h<d} 2^(d−h)·( (1 − 2^(h−d))^Q − (1 − 2^(h+1−d))^Q ),        E[n] = 2^d·(1 − (1 − 2^(−d))^Q),
```

which gives 258.936, 64.992, 36.995, 25.995 and 19.994 distinct leaves and 2,615.04, 721.44, 404.06, 271.39 and 196.51 digests for the five levels, 324,067.93 bytes of leaves and digests, and an opening of 325,531.93 bytes in expectation (computed). For `q` distinct leaves the multiproof has at most `q·(d − ⌈log2 q⌉) + 2^⌈log2 q⌉ − q` digests, attained by balanced positions: 2,843, 778, 434, 292 and 212 at the query counts, so the opening is at most 336,792 bytes: 1,464 fixed, `259·512 + 148·384 = 189,440` of leaves and `4,559·32 = 145,888` of digests (counted). The bound `Σ Q_i·(leaf + 32·d_i) + 1,464 = 428,856` bytes, which lets every path be disjoint, is valid and is not attained. The whole proof of the experiment is the front end's 15,024 bytes, its 26-byte envelope, the 800-byte commitment and the opening, 341,381.93 bytes in expectation.

#### 7. The packing and the opening field

`H = F_2[x]/(x^128 + x^7 + x^2 + x + 1)` and `E = K[y]/(y^3 + y + 1)` have degrees 128 and 192 over `F_2`, and 128 does not divide 192, so `H` is not a subfield of `E`. The claim is in `H`. The bridge of §3 exists because bits are packed into symbols and not because of the missing embedding. Extraction of a coordinate is linear over `F_2` and not over `H`, so `bit_b(Σ_z ω_z·p[z])` is not `Σ_z ω_z·bit_b(p[z])` in any field, and a design whose opening field contains `H` keeps a tensor reduction of the same shape.

A code alphabet does not have to be a field. It has to be closed under addition and under multiplication by the twiddles of the code, which are in `K`, and it needs a basis over `F_2` to carry bits. `V` has both. No step of the scheme multiplies two symbols of level 0: the sumcheck multiplies a symbol by a weight in `E`, and the first fold multiplies it by a challenge in `E`, after which every value is in `E`. The representation of `F192` as three coefficients in `K` is what makes this free: a symbol of `V` is two words, the transform of level 0 is the transform over `K` on twice as many lanes, the leaves are the 512 bytes of 64 words, and Lemma 1 of §8 shows that every candidate of level 0 is `V`-valued.

Counts at `t = 22`, for the prover algorithm of §9 in three designs: this spec; 64 bits to a symbol of `K` with the 64 by 128 bridge, `μ = 24` and `k_0 = 6`; and 128 bits to a symbol of `H` with an opening field `E'` of `2^256` elements that contains `H` (Alternatives Considered gives its product counts). All three use the composed lookup of §9.

| Quantity (counted) | `V ⊂ F192`, 128 bits | `K ⊂ F192`, 64 bits | `H ⊂ F256`, 128 bits |
|---|---:|---:|---:|
| Symbols at level 0; leaf bytes | `2^23`; 512 | `2^24`; 512 | `2^23`; 512 |
| Level-0 encoding | 905,969,664 | 905,969,664 | 905,969,664 |
| Later encodings, 43,515,904 butterflies | × 9 = 391,643,136 | × 9 = 391,643,136 | × 12 = 522,190,848 |
| Commit sample, inner products | 50,331,648 | 50,331,648 | 67,108,864 |
| Sumcheck round 1 | `42·2^22` = 176,160,768 | `33·2^23` = 276,824,064 | `52·2^22` = 218,103,808 |
| Sumcheck later rounds | `36·(2^22 − 1)` = 150,994,908 | `36·(2^23 − 1)` = 301,989,852 | `60·(2^22 − 1)` = 251,658,180 |
| Induced weights, 7,405,568 butterflies | × 9 = 66,650,112 | × 9 = 66,650,112 | × 12 = 88,866,816 |
| **Subtotal of multiplications** | **1,741,750,236** | **1,993,408,476** | **2,053,898,180** |
| The same without the composed lookup | 1,766,916,060 | 2,043,740,124 | 2,079,064,004 |
| Lookups of `Φ`, entries of one element | 134,217,728 | 268,435,456 | 134,217,728 |
| BLAKE2s compressions, all trees | 8,159,227 | 8,159,227 | 9,142,267 |
| Vectors of round 1 (weights; folded message) | 192 MiB; 96 MiB | 384 MiB; 192 MiB | 256 MiB; 128 MiB |
| Commitment bytes | 800 | 1,568 | 1,056 |
| Verifier recurrence, products | 2,944 in `E` | 3,072 in `E` | 2,944 in `E'` |
| Weakest algebraic term | `2^-128.27` | `2^-129.33` | `2^-192.27` at the slack of this spec |
| Field code that exists | all but one accumulator method | the same | none of `E'` |

The rows follow from §9. Level 0 is the same transform over `K` on 64 words per position in the first two designs, and 32 lanes of `H` at 6 multiplications per butterfly in the third. The 64-bit packing has twice the pairs in every round of the sumcheck, at `6 + 3 + 3 + 9 + 12 = 33` multiplications per pair of round 1. The sumcheck of the third design runs over the same number of symbols as this spec in a wider field. The subtotal is not a total: it leaves out the equality tables and the later samples (8,178,816 multiplications in this spec, §9), the set-up of the tables, the reductions of per-thread accumulators and the challenges. Multiplication counts are not times either: they omit moves, additions and memory traffic, and the lookups are priced separately in Performance.

**The choice, and what reverses it.** The design of this spec is the first column. Against the 64-bit packing it has 251,658,240 fewer multiplications, half the lookups, 288 MiB less in round 1 and a commitment of 800 bytes for 1,568, with the same level-0 oracle, the same leaf bytes, the same later levels and the same field code. Against the opening field of `2^256` elements it has 312,147,944 fewer multiplications, 983,040 fewer hash compressions and a field that exists. The choice is taken on these counts and is to be confirmed by the first measurement of the bridge phase (Execution, item 6): the tables, the two weight vectors and all rounds of the sumcheck, on one thread at `t = 22`. The model of Performance prices that phase at 180.3 ms for this spec and at 337.6 ms for the 64-bit packing (estimated, at the provisional unit prices: `327,155,676 c + 134,217,728 Lw` against `578,813,916 c + 268,435,456 Lw`). The choice is reversed if the measured time of that phase is above the count of the 64-bit packing repriced at the unit prices that items 3 and 6 measure: the 64-bit packing is then implemented behind the same traits and measured, and the faster of the two is kept. The parameters of §5 and the wire of §6 are those of the design that is kept.

#### 8. Security ledger

**Statement.** The scheme is analysed as an interactive protocol, round by round. A state function marks every partial transcript as doomed or not; the state is doomed at the start when the claim is false, and for each verifier challenge the probability that a doomed state becomes one that is not doomed is the error of that transition. The claim of this spec is conditional: under assumptions 1 and 2 and the invariant of Lemma 2, every algebraic and query transition of the scheme's own protocol has error at most `2^-128`, for every admitted `t`, with no grinding. That is a bound on the maximum over transitions. It is not a bound on the failure of a whole interactive execution, which the sum of the transitions bounds, and it is not a bound after the Fiat-Shamir transformation that is independent of the number of queries. The relation is existential: an accepted opening implies that the commitment selects a table with the claimed partial evaluations. No extractor is given and knowledge soundness is not claimed (Open, 2). The Merkle term and the compilation term are stated separately below, each with its query budget. The ledger is a list of bounds under stated assumptions; nothing in it is a proof of security of the implementation.

**Assumptions.**

1. The proximity-gap theorem for Reed-Solomon codes up to the Johnson bound, in its list form for mutual correlated agreement (Ben-Sasson, Carmon, Haböck, Kopparty and Saraf, 2025, Theorem 4.6), with the constant `a` below. The constant is taken from the commitment-scheme annex of the public leanVM documentation and from its parameter calculator, which agree with each other. It has not been compared with the statement in the paper: the version of the paper, whether the constant is the one for the list form (the calculator warns of a factor of two between the list and the non-list parameter), and whether the hypotheses admit the code over `E` on a domain `S_d ⊂ K` that is a subspace over `F_2`, are open (Open, 1). Every numerical row below is conditional on it.
2. The round-by-round analysis of the WHIR recursion with interleaved codes in the same annex, which proves round-by-round soundness of an existential opening relation and not knowledge soundness. The modifications of this spec (the commit sample and its passage through the lane folds, the bridge, the direct check of the last level) are argued in Lemmas 1 and 2 and in the paragraphs after them.
3. BLAKE2s-256 and the transcript's hash are random oracles. This is a model, and nothing is proven of either function.

**Quantities, per level.** `n = 2^d`, `ϱ = (2^c − 1)/2^d` (the rate parameter of the theorem for dimension `2^c`), slack `η`, radius `γ = 1 − √ϱ − η`, list bound `L = 1/(2·η·√ϱ)` (Johnson; the same for the interleaved code, since interleaving keeps the distance), `m = max(⌈√ϱ/η⌉, 3)`, and

```text
a = ( 2·(m + 1/2)^5 + 3·(m + 1/2)·γ·ϱ ) / (3·ϱ^(3/2)) · n + (m + 1/2)/√ϱ,        ε = a/|E|.
```

**Terms.** `J_i` is the number of claims batched at level `i ≥ 1`, the residual included: `J_1 = n_0 + 3` and `J_i = n_{i−1} + 2` after it, with `n_{i−1} ≤ Q_{i−1}`. Level 0 has one claim and no batching.

| Transition | Error | What fails otherwise |
|---|---|---|
| Commit sample `z_0` | `C(L_0, 2)·c_0/\|E\|` | two members of the level-0 list agree at `z_0` in every lane |
| Bridge `α` | `127/\|E\|` | §3, for the one selected member |
| Batching `λ_i` | `(J_i − 1)·L_i/\|E\|` | a false claim cancels in the combination, for some list member |
| Fold round `j ≤ k_i` of level `i` | `2·L_i/\|E\| + 2^(k_i − j)·ε_i` | for some list member, a sumcheck round of degree 2 or, at level 0, the vanishing of its residual sample discrepancy (Lemma 2); or correlated agreement of the partial fold |
| Sample `z_{i+1}` | `C(L_{i+1}, 2)·c_i/\|E\|` | as the commit sample, for level `i+1` |
| Positions of level `i` | `(1 − γ_i)^(Q_i)`, and 0 where every position is opened | every queried column agrees with a word that is `γ_i`-far |
| Closing round | `2/\|E\|` | a sumcheck round of degree 2 |

Two further terms are not transitions with an error independent of the adversary's work, and they are kept apart from the table. A collision of BLAKE2s-256 as an ideal hash of 256 bits has probability at most `q_h·(q_h − 1)/2^257` for `q_h` queries to it; every row above is conditional on there being none. The Fiat-Shamir transformation, under whichever compilation theorem is taken for the transcript (Open, 3), multiplies the largest transition error by the number `q_t` of queries to the transcript's hash. `q_h` and `q_t` are different budgets.

At the reference size, with `|E| = 2^192`, as `−log2` of the error (computed, in decimal arithmetic at 80 digits):

| Level | `ϱ` | `m` | `η` | `γ` | `L` | `Q` | Positions | Fold, `j = 1` | Sample | Batching | `2L/\|E\|` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `(2^18−1)/2^19` | 249 | 0.002840 | 0.29005 | 249.0 | 259 | 128.003 | 128.270 | 172.92 | none | 183.04 |
| 1 | `(2^14−1)/2^18` | 47 | 0.005319 | 0.74469 | 376.0 | 65 | 128.029 | 137.74 | 171.72 | 175.42 | 182.45 |
| 2 | `(2^10−1)/2^17` | 35 | 0.002524 | 0.90913 | 2,242.2 | 37 | 128.022 | 136.33 | 166.93 | 174.82 | 179.87 |
| 3 | `(2^6−1)/2^16` | 16 | 0.001938 | 0.96706 | 8,322.0 | 26 | 128.021 | 138.33 | 163.63 | 173.73 | 177.98 |
| 4 | `(2^2−1)/2^15` | 5 | 0.001914 | 0.98852 | 27,306.7 | 20 | 128.890 | 142.17 | 160.94 | 172.51 | 176.26 |

The bridge is at 185.011 bits and a closing round at 191. The "Fold" column is the whole fold row at its worst round, `2·L_i/|E| + 2^(k_i − 1)·ε_i`, which the correlated-agreement part dominates; the sample of row `i` is the one that binds level `i` (the commit sample for row 0). The weakest term is the positions of level 0, `(√ϱ_0 + η_0)^259 = 2^-128.003083`, and the weakest algebraic term is the fold of level 0 at `2^-128.270474`. Each `m` is the smallest for which the query count is minimal while the fold, sample and batching terms of the level stay at most `2^-128`. Over the admitted sizes the minimum of the algebraic and query rows is `2^-128.000021`, at `t = 11`; for `t ≤ 9` every position of the single level is opened and the minimum is above 171.9 bits. Every algebraic and query transition meets the conditional target with no grinding; the hash and compilation terms depend on query budgets and are not in this table.

**Lemma 1 (the candidates of level 0 are `V`-valued).** Let `D = 2^(c_0)` and `n = 2D`, and let `g` be a codeword of the interleaved code over `E` on `S_(d_0)` that agrees with the level-0 oracle on more than a fraction `√ϱ_0` of the positions. Then the message of every lane of `g` is `V`-valued. *Proof in outline.* The agreement set has more than `√((D − 1)·n)` positions, which is at least `D − 1`, so it has at least `D`. Fix a lane and write its polynomial as `g_0 + y·g_1 + y^2·g_2` with each `g_k` in `K[X]` of degree below `D`, which is possible because `1, y, y^2` is a basis of `E` over `K`. The domain lies in `K`, so the value at a position `x` is `g_0(x) + y·g_1(x) + y^2·g_2(x)` with each `g_k(x)` in `K`. The symbol of the oracle at a position of agreement is in `V`, so `g_2` vanishes at `D` distinct points and is zero. The change from monomials to the basis `X_w` has coefficients in `K`, so the message is `V`-valued. Agreement of a position of the interleaved word is agreement in every lane, so the argument applies lane by lane. ∎ The lemma holds for every admitted `t`, since `c_0 ≥ 2`. The fold challenges are in `E` and the relevant code is the one over `E`, but its list at level 0 consists of tables of bits; the commit sample selects at most one of them, and §3 proves the claims about that one. Messages of later levels are folds with challenges in `E` and are `E`-valued.

**Lemma 2 (the commit sample through the lane folds).** Condition on no collision of the hash and let `Λ` be the list of level 0, of at most `L_0` members. For `f ∈ Λ` let `δ_f[u] = f~(u, ·)(z_0) + y_0[u]`, `u < 2^(k_0)`, be its discrepancy with the commit sample, and after `j` rounds of level 0 with challenges `a_1, …, a_j` let `δ_f^(j)` be `δ_f` folded over its `j` low variables with `eq_E((a_1, …, a_j), ·)`, a vector over the `2^(k_0 − j)` remaining lanes. Call the state after round `j` doomed when every candidate close to the partially folded oracle either has `δ^(j) ≠ 0` or violates the running claim `σ = Σ f^(j)·ω^(j)`. Then round `j + 1` leaves the doomed states with probability at most `2·L_0/|E| + 2^(k_0 − j − 1)·ε_0`, which is the fold row. *Proof in outline.* Before the challenge `a`, put each candidate in one of two classes. A candidate with `δ^(j) ≠ 0` has the next vector `δ^(j+1)[u] = δ^(j)[2u] + a·(δ^(j)[2u] + δ^(j)[2u + 1])`, affine in `a` and not identically zero, so it vanishes for at most one `a`; this event is counted whatever the candidate's sumcheck claim is. A candidate with `δ^(j) = 0` has a false running claim, so the round polynomial that was sent and the candidate's own are different polynomials of degree 2 and agree at no more than two values of `a`. Each candidate is in one class, so the union over the list costs at most `2·L_0/|E|`, and not that plus a term for the sample. The remaining event is that the oracle folded with `a` is close to a word that is not the fold of a member of the list, which is mutual correlated agreement, `2^(k_0 − j − 1)·ε_0` under assumption 1. ∎ After round `k_0` the residual vector has one entry, `f~_1(z_0) + Σ_u eq_E(a^(0), u)·y_0[u]`. It is the commit-sample claim that §4 adds to the claims of level 1, or checks on the final message when `R = 1`, and from there the source's argument at a level interface carries "violates the residual claim or the sample claim" as it carries its pool of claims, at the batching error with `J_1 = n_0 + 3`. The quantity `L_0·k_0/|E|`, `2^-181.72` at the reference size, bounds the event that some member with a wrong sample survives all `k_0` lane challenges. It is a diagnostic over the whole vector `a^(0)`, whose coordinates are drawn in `k_0` separate rounds with prover messages between them, and it is not a transition and not a row of the ledger.

**The bridge and the list.** The bridge error is `127/|E|` with no factor for the list: `α` is drawn after the commit sample has selected at most one member, and members that violate the sample are carried by Lemma 2.

**The direct check of the last level.** The claims of the last level's queries are checked against the final message, which is sent in the clear, and not batched; this removes the batching round that they would need and adds no term.

**The composition ledger.** Every challenge of the front end is in `H`, so the composed protocol has no claim of 128 bits whatever this scheme does. The bounds of the front end's own checks, from the protocol spec (derived):

| Front-end check | Error | `−log2` |
|---|---|---:|
| Column reduction through `value()` alone, the contract | `8/2^128` | 125 |
| Column reduction as this scheme proves it, for a fixed wrong `C` (§1) | `1/2^128` | 128 |
| A sumcheck round of degree 6, the highest of the reference layout | `6/2^128` | 125.415 |
| A sumcheck round of degree 17, the highest admitted | `17/2^128` | 123.913 |
| The `8 + t = 30` coordinates of `tau` of `SpartanOuterF2` at `t = 22`, as one transition | `30/2^128` | 123.093 |
| The `a = 61` coordinates of `tau` of the output check, at the largest admitted `a`, as one transition | `61/2^128` | 122.069 |
| The 185 rounds of the reference layout, union bound over their degrees alone | `640/2^128` | 118.678 |

The per-round ceiling of the composed protocol is 125.415 bits at the reference layout and 123.913 bits at the largest admitted degree, unless the front end grinds or changes its field, which is outside this spec. These are ceilings read from single round bounds and not a theorem about the composed protocol. With a vector of challenges counted as one transition the two vector rows are lower; counted one coordinate at a time they need an invariant that the protocol spec does not state; and the failure of a whole interactive execution is bounded by a sum, of which the last row is one part. This scheme improves one row, the column reduction, and the commit sample keeps every row from being multiplied by the list size. It does not improve the sumchecks, and a larger opening field would not either.

**Grinding.** There is none. Grinding `g` bits before the positions of a level are drawn does not change that round's error; it multiplies the work of each attempt at that round by `2^g`. Counted as error it would let the query counts drop to 225, 56, 32, 23, 17 at `g = 17` with the position terms at `2^-111.0`, saving 56,640 of the 427,392 bytes of the disjoint-path bound (counted; 41,784 of 332,112 bytes measured on the source), for an expected `5·2^17` hash evaluations of grinding. It buys nothing for the fold, sample, batching, bridge or front-end terms. It is not adopted, so that the target is met as an error bound.

**What "128 bits" does not cover.** The algebraic and query transitions of the scheme meet the target, conditionally. Three things do not. The front end's rows, as above. The hash: the collision bound is at most `2^-128` only for `q_h ≤ 2^64.5` queries, a collision with constant probability costs on the order of `2^128` evaluations, and both figures are for an ideal hash against a classical adversary. The compilation: a transition error of `2^-128` is an advantage of at most `q_t·2^-128` after `q_t` queries to the transcript's hash and not an error of `2^-128`.

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

Every buffer is an allocation of exactly the stated length, and the lifetimes are part of the spec: truncating a vector does not return its capacity, so a buffer is released only by dropping its allocation.

| Buffer (counted) | Capacity | Allocated | Released |
|---|---:|---|---|
| Rows (shared, not owned) | 128 MiB | caller | caller |
| Level-0 codeword; level-0 tree | 256 MiB; 32 MiB | `commit` | after the multiproof of level 0 is written |
| `W0` and `D`, `2^22` elements of `E` each | 96 MiB each | round 1, first pass | `D` after the fold of round 1; `W0` at the fold of round `k_0` |
| Folded message, `2^22` elements of `E` | 96 MiB | fold of round 1 | at the fold of round `k_0` |
| Weight and message of level `i ≥ 1`, `2^(c_{i−1})` elements each | 6 MiB each at level 1 | fold of the last round of level `i − 1` | at the fold of the last round of level `i` |
| Later codewords; later trees | 96, 48, 24, 12 MiB; 16, 8, 4, 2 MiB | commitment of the level | after the multiproof of the level is written |
| Equality tables and induced weights of a level; transposed-transform scratch | at most 30 MiB at level 1 | start of the level | once merged into `ω` |
| Tables of `Φ_α`, split equality tables, `eq_E(z_0, ·)` in `commit` | 192 KiB, 64 KiB, 6 MiB | where used | end of `open`, of round 1, of `commit` |

Three rules make these lifetimes hold without a copy. The weight of round 1 is two vectors and not one interleaved vector, so that the fold `W0[k] ← W0[k] + a·D[k]` leaves `W0` at its exact length and `D` is dropped whole. Folds are in place within a level, at constant capacity. The fold of the last round of a level `i < R − 1` writes its two outputs, `2^(c_i)` elements each, to new allocations and drops its inputs before the codeword of level `i + 1` is allocated. No buffer is moved to reclaim capacity: the output of that fold is written once in either case, and what is charged is the first touch of the new allocations, 12 MiB at level 0, inside the measured units.

The owned total over time at `t = 22`, in MiB: 288 after `commit`; 480 in the first pass of round 1; 576 during its fold; 480 from the end of round 1 to round 4; 492 during the fold of round 5 and 300 after it; 412 once level 1 is committed, with at most 30 of scratch at the start of level 1 after the 288 of level 0 are released; below 200 from then on. The peak is `256 + 32 + 96 + 96 + 96 = 576` MiB owned, in the fold of round 1, and 704 MiB with the rows (counted). An implementation that folds in place and only truncates holds 288 MiB of capacity in the weight and the message when level 1 is committed and reaches `288 + 288 + 112 = 688` MiB there; the rules above are what excludes it.

#### 10. Reuse

**Ported.** The recursion, its level schedule, the per-level parameter search, the additive transform over `F64` and over `F192` with twiddles in `F64`, the batched BLAKE2s leaf and node hashing, the lane-fold and round kernels, and the induction of query weights by the transposed transform exist in the public leanVM repository, in its `crates/pcs` (`whir.rs`, `whir_config.rs`, `whir_ntt_ext.rs`, `whir_induce.rs`, `ntt.rs`, `merkle.rs`). That repository is under the MIT licence, and its files in `crates/pcs` carry per-file credit and copyright lines with the identifier `Apache-2.0 OR MIT`. A port keeps the header of each source file verbatim at the top of every file derived from it, states in that file that it is modified, and adds the MIT licence text of the source with its copyright line to the repository's third-party notices. Jolt is itself under MIT and Apache-2.0, so no licence conflict arises. The port is a port of algorithms onto `jolt_field::binary`, not a dependency: the source has its own field types, transcript, allocator and thread pool.

**Not ported, and why.** The source commits a root alone and calls its commitment list binding; the commit sample is new. Its ring switch is 64 by 192, for a claim in `F192`, with a batching map built from six challenges and the Frobenius automorphism; the claim here is in `F128`, so the bridge of §3 and its prover kernel are new. The source's lanes are the high variables of the message and its level-0 leaf image is lane-descending; here the lanes are the low variables, which removes the packed copy and the transposition at level 0 and keeps the front end's variable order. The source's wire format and transcript are its own. The source's parameter calculator runs in floating point at configuration time; here its output is a frozen table.

**In the repository already.** `F64`, `F128` and `F192` with carry-less kernels and accumulators, `F192: ExtField<F64>` with `mul_base`, `F128::mul_x`, the Blake2b transcript with `squeeze_bytes`, the `blake2` crate (which provides BLAKE2s-256 for the verifier), `rayon`, and the two traits with their contract tests.

**New.** An accumulator method for an element of `F192` times an element of `F64` (three carry-less multiplications, no reduction); the additive transform; the Merkle tree and multiproof; the bridge on both sides; the protocol, its wire format and its parameter table.

**Where the code lives.** The verifier's half is a module `whir` of `jolt-rv64i-verifier`, in safe code, hashing with the `blake2` crate. The prover's kernels (transform, tree, bridge weights, rounds, induced weights) are a new crate, `crates/jolt-rv64i-pcs`, because `jolt-rv64i-prover` and `jolt-rv64i-kernels` forbid `unsafe` and a batched BLAKE2s needs vector intrinsics; the new crate denies `unsafe` outside `src/arch/`, where each use carries a `SAFETY:` comment naming the `cfg` that guarantees the instruction, as `jolt-field` does. `jolt-rv64i-prover` implements `BitsCommitmentProver` in `src/commitment/whir.rs` by calling it.

### Alternatives Considered

**64 bits to a symbol of `K`.** The packed table is the row buffer read word by word, `μ = t + 2`, `k_0 = 6`, the bridge has shape 64 by 128 with `t_h ∈ K`, and the level-0 oracle is byte for byte the one of this spec. The query counts at `t = 22` are 260, 65, 37, 26, 20 and the commitment 1,568 bytes. Its subtotal is 1,993,408,476 multiplications against 1,741,750,236, with 268,435,456 lookups against 134,217,728 and 576 MiB of round-1 vectors against 288 (counted, §7). Not adopted on those counts. Reopened by the measurement of the bridge phase stated in §7.

**An opening field of `2^256` elements that contains `F128`.** The smallest field that contains `H` and has at least `2^192` elements is `E' = H[v]/(v^2 + v + x^121)`. The polynomial is irreducible because the absolute trace of `x^121` in `H` is 1; among the monomials `x^i`, `i < 128`, exactly `x^121` and `x^127` have trace 1 (computed, by polynomial arithmetic modulo `x^128 + x^7 + x^2 + x + 1` outside the repository). `E'` does not exist in the repository. Its product has the coefficients `a_0·b_0 + x^121·a_1·b_1` and `a_0·b_1 + a_1·b_0 + a_1·b_1`: three products in `H` and the linear map of multiplication by `x^121`, which is not free and needs a kernel of its own. At 18 multiplications for a product, 12 into an accumulator, and 12 and 8 for an element of `E'` times an element of `H`, round 1 costs `12 + 16 + 12 + 18 = 58` per pair, or 52 with the composed lookup, and a later round `24 + 36 = 60`. The subtotal is 2,079,064,004 without the composed lookup, 1.73% above the 2,043,740,124 of the 64-bit packing on the same basis, before the constant map and the reduction of the extension are priced; figures of 56 per pair assume a product of 16, which needs a kernel that is specified and measured. Containment does not decide the packing. An embedding of `K` into the raw representation of `H` is a linear map over `F_2` with 128 output coordinates and 64 input coordinates; a representation of `H` over its subfield of `2^64` elements, with the cost of converting it to the raw representation, would have to be stated separately, and the design counted in §7 packs 128 bits to a symbol of `H` instead. It keeps a bridge of shape 128 by 128 (§7). It gains 64 bits on every algebraic term and none on the query terms, which are the weakest, and none on the front end's. Not adopted: 312,147,944 more multiplications than this spec, 983,040 more hash compressions (later leaves of 512 bytes, 8 compressions for 6 on 491,520 leaves) and no field code. Reopened by either of two results: the check of the theorem in its primary source (Open, 1) gives a constant that the search cannot absorb over `2^192` elements; or the front end moves its challenges to a field that contains `H` and a kernel for `E'`, constant map included, is measured at 16 multiplications or fewer per product.

**A commitment that is a root alone.** This is what the existing implementation does. It saves the commit sample: 53,477,376 carry-less multiplications, an estimated 16 ms on one thread, and 1,536 bytes. It leaves the commitment binding a list of up to `L_0 = 187` tables, and every round of the front end, whose errors are of the form `δ/2^128`, is then paid once per list member: 7.55 bits lost on each, with no way to recover them in the scheme. Rejected. It would be reopened only by a front end whose challenge field has that margin.

**One sample of the whole table instead of one per lane.** A single value `p~(z_0)` at `z_0 ∈ E^μ` makes the commitment 56 bytes instead of 1,568. Its weight has `2^24` entries at level 0: merged into the bridge weight it costs `12·2^24 = 201,326,592` carry-less multiplications, and carried as a separate term through the six lane rounds about `6·2^23 + 12·(2^23 − 2^18) = 147,849,216`. The per-lane sample costs `12·2^18` in the opening. Rejected on that count; 1,512 bytes are 0.46% of the proof.

**Grinding.** §8. At 17 bits: 57,760 bytes fewer at most, the position terms at `2^-111` as errors. Rejected because the target is an error bound. Reopened if the owner accepts a work factor in place of an error for those rounds.

**Rate 1/4 at level 0.** Halves the queries of level 0 (130 for 260) and doubles the level-0 codeword and tree. On the existing implementation, measured in one sweep on the loaded host at twelve threads: 219,192 bytes against 332,112, and 442.5 ms against 300.0 ms for commit and open together, with 1,611 MiB against 1,326 MiB of resident memory at the peak. Rejected while the prover's time is the binding budget. Reopened if proof size becomes the objective.

**An initial fold of 4 or 8 instead of 6.** Measured in the same sweep: 385.7 ms and 309,056 bytes at 4, 349.1 ms and 676,912 bytes at 8, against 300.0 ms and 332,112 bytes at 6. Rejected on both columns at 8 and on time at 4.

**Ligerito.** The recursion of this spec is of the same family: interleaved codes, partial sumchecks, one new commitment per level. The alternative is its analysis within the unique-decoding radius, with no list and so no commit sample, no samples at later levels and no proximity-gap term. Its position term is `(1 − (1 − ϱ)/2)^Q`: 309 queries at rate 1/2 for 260 and 141 at rate 1/16 for 65, which is `49·1,120 + 76·960 = 127,840` more bytes at most on the first two levels alone. Rejected on proof size. Reopened if the proximity-gap constant of assumption 1 is found not to apply as transcribed, since it is the one assumption that the unique-decoding analysis does not need.

**BaseFold-style folding of the codeword.** Fold the level-0 codeword itself, with one tree per fold of the same schedule and every oracle at rate 1/2. The prover saves every later encoding (43,515,904 butterflies, 65.4 ms measured on one thread) and most of the later hashing. Every query then opens a path in every oracle at the level-0 query count: `260·(1,120 + 864 + 736 + 608 + 480) = 990,080` bytes at most against 428,512, since the oracles have `2^19`, `2^15`, `2^11`, `2^7` and `2^3` leaves. Rejected on proof size, a factor of 2.31. Reopened if the proof is not transmitted or its size does not matter and the 154 ms of later encodings and trees do.

**Other alphabets.** Symbols that are full elements of `F192` carrying 64 or 128 bits each, to avoid mixed arithmetic, would hash 24 bytes per symbol where this spec hashes 16, and would multiply the 4,194,304 leaf compressions of the first tree by 3 or by 1.5. Symbols in `H` with `E = H` would make every algebraic term a term over `2^128` elements, where the fold term of level 0 has no room. Rejected.

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

1. **The proximity-gap theorem in its primary source.** The formula for `a` in §8 agrees between the annex and the calculator of the ported implementation and has not been checked against the paper: its version, whether the constant is that of the list form, and its hypotheses on the field and the domain. The fold terms of the ledger, and through the search every `η` and query count, depend on it. At the reference size the fold term of level 0 has a margin of 0.27 bits, because the search spends the bit gained by the initial fold of 5 on one query of level 0; a constant larger by a factor of two is not absorbed at `Q_0 = 259`, and at `Q_0 = 260` with `m = 187` the same term is `2^-130.33` (computed).
2. **Knowledge extraction and the written proofs.** The scheme claims round-by-round soundness of an existential relation. An extractor, and full proofs of Lemma 2 and of the level interface with the commit-sample claim in the pool, are owed before the ledger is cited outside this experiment. Lemmas 1 and 2 are given in outline. An exhaustive instance of Lemma 1, a subfield of 4 elements in a field of 16 with dimension 2 on a domain of 4 points, finds no candidate outside the subfield among the 1,072 pairs of a word and a polynomial that agree on two points or more (computed); that corroborates the lemma and does not prove it.
3. **The composed protocol.** No theorem is stated for the front end composed with this scheme. The convention for a vector of challenges (one transition per vector or one per coordinate), the sum over rounds, and the compilation theorem for the Fiat-Shamir transformation with its transcript model are not fixed; the ceiling of §8 is read from individual round bounds.
3. **The table of §5.** Computed in double precision outside the repository. The entry for `t = 22` agrees with the query counts measured on the existing implementation; the others are unconfirmed until item 1.
4. **The hash.** BLAKE2s-256 is kept from the source. Whether BLAKE3 or SHA-256 with hardware support is cheaper at 512-byte and 384-byte leaves on the benchmark host is not measured; a change is a change of the wire format.
5. **Unit prices.** `c` and `Lw` are estimates. The measured rows were taken on a loaded host, on the source and not on this scheme. No phase of the bridge has been measured.
6. **Scaling to twelve threads.** The house factor of 9.6 is not supported by the one measurement available, which gives 4.6 for commit and 5.1 for open on a loaded host with eight performance and four efficiency cores. Whether the cause is the load, memory bandwidth in the transform and the hashing, or the core mix is not known.
7. **The lane order in the port.** Whether the source's round and fold kernels can be used with lanes in the low variables has not been established by reading them through.
8. **The trace of `x^121`.** Computed once by a script outside the repository. It matters only if design (b) is taken up.
9. **Small tables.** For `t ≤ 10` every position of level 0 is opened. The proof is then larger than the table for the smallest `t`, which is accepted for sizes used only in tests; whether the front end ever runs a production instance below `t = 11` is the owner's to say.
