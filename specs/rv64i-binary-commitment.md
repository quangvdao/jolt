# Spec: The Commitment Scheme for the Bit Table of RV64I over the Binary Field

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-10                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

The front end of the binary-field RV64I experiment (`specs/rv64i-binary-protocol.md`) commits to one table of bits, `Bits`, with 256 columns and `T = 2^t` rows, and at the end asks for one multilinear evaluation of it at a point with coordinates in `F128`. The only other implementation of that contract is `TransparentBits`, a test stand-in that sends the table. This spec fixes the production scheme behind the same two traits, `BitsCommitmentScheme` and `BitsCommitmentProver`, with no change to either.

The scheme is hash-based. The opening field is `F192`, the cubic extension `F64[y]/(y^3 + y + 1)` of `F64`. The bits are packed 128 to a symbol: two consecutive 64-bit words of a row are the two coefficients of an element of the subspace `V = F64 + y·F64` of `F192`. The `2^(t+1)` symbols are encoded with an interleaved Reed-Solomon code on an additive domain of `F64`, which acts on `V` coefficient by coefficient, so that the level-0 codeword consists of words of `F64` and a symbol is 16 bytes. The codeword is committed with a Merkle tree, and the commitment carries one out-of-domain evaluation of every lane of the code: except with the error of that sample and of a collision of the hash, the commitment selects at most one table near the committed word, possibly none, and not a list of tables. The opening is a WHIR recursion over `F192`, analysed in the list-decoding regime up to the Johnson bound. The evaluation claim lives in `F128`, which is not a subfield of `F192`; a tensor reduction (ring switching) of shape 128 by 128 turns the 128 partial evaluations that the front end already publishes into one inner-product claim over `F192`. The reduction reads the bits of elements of `F128` and never multiplies one inside `F192`, so the scheme requires no embedding of `F128` into `F192`; the arithmetic, the lookups and the memory of the reduction are counted in §7 and §9.

At the reference size `t = 22` the table has `2^30` bits and `2^23` symbols. The parameter set is rate 1/2, fold schedule 5, 4, 4, 4, 4, query counts 260, 65, 37, 26, 20 and no grinding. Under the proximity-gap theorem for Reed-Solomon codes up to the Johnson bound, in the form transcribed in the analysis that accompanies the public leanVM implementation, and under the invariant of §8, every algebraic transition of the scheme has error at most `2^-130` and every query transition at most `2^-128`, at every admitted `t`; the weakest is the query term of the first level at `2^-128.0001`, and the weakest algebraic term at the reference size is the proximity-gap term of that level at `2^-130.33`. This is conditional round-by-round soundness of the scheme's own interactive transitions. The hypotheses and constants of the theorem in its primary source, knowledge extraction, the composed protocol and the compilation theorem for the Fiat-Shamir transformation are obligations of the Open list, and no bound is claimed for the compiled protocol; the Merkle term is stated for a declared budget of `2^64` hash queries; and the bounds of the front end over `F128` supply a guarantee of 125.415 bits per round at the reference layout (§8). The opening proof is 326,320.45 bytes in expectation and 337,592 at most, and the commitment is 800 bytes (counted). The public leanVM implementation of the same recursion, at the same level-0 oracle and for a claim in `F192`, measures 311 ms to commit and 730 ms to open on one thread and 67.8 ms and 142.6 ms on twelve, on a loaded host. The model of this spec for the scheme with the `F128` bridge and the commit-time sample is 155.59 ns per cycle on one thread (estimated), which gives single-thread thresholds of 93 ns for commit and 101 ns for open, together 194 ns per cycle and 9.2% of the prover's 2,100.

The packing of 128 bits into `V` is taken on operation counts against a packing of 64 bits into `F64` and against an opening field of `2^256` elements that contains `F128` (§7). The choice is a forecast from counts and no measurement supports it yet: the first measurement of the bridge phase decides whether it stands, and §7 states the result that reverses it.

## Intent

### Goal

Fix, completely enough to implement and to audit, the commitment scheme of the bit table: the committed object and the claim, every message of the commit phase and of the opening, every challenge and the bytes it is drawn from, the wire format, the parameter set and its soundness ledger, the prover's operation counts and memory, and the items of work.

Notation used throughout. `K = F64`, `H = F128` and `E = F192` are the types of `jolt_field::binary`. `E` is `K[y]/(y^3 + y + 1)` and `F192` stores its three coefficients in `K` in ascending degree, so `K ⊂ E` is the first coefficient, `mul_base` multiplies each coefficient by an element of `K`, and the canonical 24 bytes are the three coefficients in order; `H` is unrelated to `E`. `V = K + y·K ⊂ E` is the set of elements whose third coefficient is zero. It is a subspace over `K` of dimension 2, closed under addition and under multiplication by `K`, and not closed under multiplication. `pack(a_0, a_1) = F192::from_base_fn` of `(F64::from_raw(a_0), F64::from_raw(a_1), 0)` is the element `a_0 + y·a_1` of `V` for two words `a_0`, `a_1`. `t = log_T`, and `μ = t + 1` is the number of variables of the packed table. Points are low-variable-first and sumchecks bind the lowest variable first, as in the front end. `eq_F(r, z)` is the equality polynomial over the field `F`, with bit `l` of the index `z` paired with `r[l]`. `β_i = x^i`, `i < 64`, is the basis of `K` over `F_2` in which `F64::from_raw` reads a word, and `ν_b = β_(b mod 64)·y^⌊b/64⌋`, `b < 128`, is the basis of `V` over `F_2`. `bit_h(e)` is bit `h` of the raw representation of an element of `H`; for `v = pack(a_0, a_1)`, `bit_b(v)` is bit `b` of `a_0` when `b < 64` and bit `b − 64` of `a_1` otherwise, the coordinate of `v` on `ν_b`. Word order is the order of the row buffer, and wherever a word becomes bytes the encoding is little-endian, on every host. "Level" means one committed oracle of the recursion; level 0 is the commitment itself.

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
- A general polynomial commitment scheme for `jolt-openings`. The generalisation of those traits is a separate later item (Execution, item 10).
- Recursion-friendliness of the verifier, batching of several tables or several points, and streaming commitment.
- A proof of the proximity-gap theorem or of the round-by-round theorem. The spec states which published results it rests on and with what constants.

## Evaluation

### Acceptance Criteria

Section numbers refer to Architecture.

**Parameters.**

- [ ] For every `1 ≤ t ≤ 32` the schedule function returns the `k_i`, `c_i`, `d_i`, `R` and `res` of §5 and the query counts of its table, with "all" as a variant of a typed value and not as a number; the rows for `t = 1`, `6`, `10`, `22`, `31` and `32` are literals in the test, as `(k; c; d)`: `(0; 2; 3)`, `(5; 2; 3)`, `(5, 4; 6, 2; 7, 6)`, the level table of §5, `(5, 4, 4, 4, 4, 4, 3; 27, 23, 19, 15, 11, 7, 4; 28, 27, 26, 25, 24, 23, 22)` and `(5, 4, 4, 4, 4, 4, 3; 28, 24, 20, 16, 12, 8, 5; 29, 28, 27, 26, 25, 24, 23)`.
- [ ] The parameters test runs the rule of §5 in exact rational arithmetic for every `t` and every level, and its output equals the frozen `Q_i` and the `m_i` of the table of §5. For every entry it checks directly: `m_i` is admissible; `Q_i` satisfies the position inequality at `m_i` and `Q_i − 1` does not; no admissible `m` has a smaller effective count, which is checked at the largest admissible `m`; no smaller `m` attains the count; and `m + 1` is not admissible where `m` is the largest. It asserts every sample, batching, existing fold, bridge and closing row at most `2^-130` and every position row at most `2^-128`, and that `t = 1` has no fold row. It reports, and does not compare with a target, the collision bound of §8 at the declared budget.

**Field.**

- [ ] `F192Accumulator::fmadd_base(a, b)` followed by `reduce` equals `a.mul_base(b)` summed, on operands supported on single coefficients for every pair of positions, and on 1,024 random pairs.
- [ ] The E-by-V kernel of §9, as the reduced product `F192::mul_base_pair(e, [b_0, b_1])` and as `F192Accumulator::fmadd_base_pair(e, [b_0, b_1])` followed by `reduce`, equals the product in `F192` of `e` with the element whose coefficients are `(b_0, b_1, 0)`: on operands supported on single coefficients, for each of the three positions of `e` against each of the two of `v`, and on 1,024 random pairs.
- [ ] The formula of the kernel is checked exhaustively in a toy field that shares no code with `jolt_field`: with `K' = F_2[x]/(x^4 + x + 1)` and `E' = K'[y]/(y^3 + y + 1)`, both implemented in the test by shift-and-XOR polynomial arithmetic, the five-product formula equals schoolbook multiplication in `E'` reduced by `y^3 = y + 1` and `y^4 = y^2 + y`, on all `4,096·256 = 1,048,576` pairs of an element of `E'` and an element of `K' + y·K'`.
- [ ] `F192::mul_y` gives `y·(c_0, c_1, c_2) = (c_2, c_0 + c_2, c_1)` on the coefficients, and `fmadd_base(e, b_0)` followed by `fmadd_base(e.mul_y(), b_1)` and `reduce` equals `mul_base_pair(e, [b_0, b_1])`, on 1,024 random inputs; this composition of six products is the comparison construction of the measurement of item 3.

**Code.**

- [ ] The encoder by definition gives, for `c = 1`, `d = 2` and `f = (f_0, f_1)`: `f_0`, `f_0 + f_1`, `f_0 + β_1·f_1`, `f_0 + (β_1 + 1)·f_1` at positions 0 to 3. For `c = 2`, `d = 3`, where `s_1(β_1) = x^2 + x` and `Ŵ_1` takes the values 0, 0, 1, 1, `x^2 + x` at positions 0 to 4: position 2 gives `f_0 + x·f_1 + f_2 + x·f_3` and position 4 gives `f_0 + x^2·f_1 + (x^2 + x)·f_2 + (x^4 + x^3)·f_3`. These are written in the test as raw words: no product in them reaches degree 64.
- [ ] The transform equals the encoder by definition for every `1 ≤ c < d ≤ 8`, for 1, 2, 8, 16 and 64 lanes, over `F64` and over `F192`, and the level-0 transform on 2, 32 and 64 words per position equals the encoder by definition applied to the lanes of `V` that pairs of words form; the transposed transform satisfies `⟨Enc(f), g⟩ = ⟨f, Enc^T(g)⟩` on random `f`, `g`.
- [ ] `W~_x(q)` equals `Σ_w eq_E(q, w)·X_w(x)` computed from the definition, for every `x` at `d ≤ 6`.

**Merkle.**

- [ ] The root of a tree of 4 leaves equals `H(H(H(l_0) ‖ H(l_1)) ‖ H(H(l_2) ‖ H(l_3)))` computed with `blake2::Blake2s256` in the test; the batched tree builder equals the tree built by that definition for every depth up to 10 and leaf sizes 16, 64, 192, 384 and 512 bytes.
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
- [ ] The last fold of 3, which the schedule reaches only at `t = 31` and `32`, is exercised at a small size: the commit and opening functions of both sides take the schedule as a value, and for the schedule `(5, 3; 7, 4; 8, 7)` at `t = 11` with every position opened at both levels, built by a constructor that exists only under the feature `test-utils` of the verifier crate, `verify_opening` accepts the output of `open`, the last level has 8 lanes and leaves of 192 bytes, and the single-byte mutations of the first rejection criterion are rejected.
- [ ] The contract tests of the two traits are generic over the scheme and pass for `WhirBits` and for `TransparentBits`: the lifecycle of commit, verify, open and verify from identically initialised transcripts, and `value()` against the evaluation of the table computed bit by bit. They are extracted from the tests of `TransparentBits`, whose checks of its own opening type, its error type and its limit on `log_T` stay with it as tests of that implementation.
- [ ] Each variant of `WhirError` of §6 is returned by a test that asserts the variant and its fields, without panicking: `UnsupportedGeometry` from `verify_commit` and from `commit` for `t = 0` and `t = 33`; `Shape` with part `LaneValues` for a commitment with a wrong number of lane values, with parts `ColumnPoint`, `CyclePoint` and `Columns` for a request of wrong lengths, with parts `Levels`, `Rounds`, `FinalValues` and `Leaves` for a directly constructed proof that `read` would not produce, and with part `Rows` from `commit` for a buffer that is not `2^t` rows; `GeometryMismatch` for a request whose geometry differs from the retained one; `CountMismatch` with parts `Leaves` and `Digests` for a proof with one leaf or one digest removed or appended; `MerkleAuthentication`, `FinalCodeMismatch`, `CommitSampleMismatch` (at `t = 6`) and `ClosingIdentityMismatch` for the single-byte changes of "Rejections" that reach them, each named in the test. `LengthOverflow` and `Allocation` are returned by the size helper for a product that exceeds `usize` and for a reservation of `isize::MAX` bytes.

**Rejections.** Each of the following makes `verify_opening` or `read` return an error, at `t = 6` and `t = 11`:

- [ ] one byte changed in each of: `root_0`, an element of `y_0`, each round coefficient, each later root, each `y_i`, each element of `f_R`, a leaf, a sibling digest, `n_i`, `g_i`;
- [ ] an opening produced for a table that differs from the committed one in one bit;
- [ ] `C` changed in two entries so that `value()` is unchanged and some `s_b` is not (this is the stronger claim of invariant 3);
- [ ] `C` changed in one entry; `rho[7]` or one coordinate of `r_6` changed; the geometry changed;
- [ ] an opening verified against a transcript that differs before the commit phase.

**Determinism.**

- [ ] The commitment and the opening bytes are identical on 1 and on 12 threads and across runs, and for the fixed table and transcript of the test at `t = 6` their Blake2b-256 digests equal literals fixed in item 8.

**Front end.**

- [ ] `prove` and `verify` of the experiment accept the counting loop at `t = 6` with `WhirBits`, and the tamper tests of the front end that target the commitment and the opening reject.

**Hygiene.**

- [ ] `jolt-rv64i-verifier` keeps `#![forbid(unsafe_code)]`; `jolt-rv64i-pcs` denies `unsafe` at its root and allows it only under `src/arch/`, which exists only behind the feature `arch` and only with the measurement that justifies it; `cargo clippy` with `-D warnings` passes for the features of the experiment; each file derived from the port carries the header lines its source has, the source path and revision, a statement that it is modified and the path of the notices file; `crates/jolt-rv64i-pcs/THIRD_PARTY_NOTICES.md` carries the manifest with its revision, the MIT licence text and the copyright and credit lines of §10.

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

The measured rows are prices of the source's kernels and not of the port's. The arithmetic of a butterfly is the same in both. At the recorded revision the source's later butterfly scales each of the three coefficients of an element with one carry-less product and reduces it with two more, nine per butterfly with the reductions, which is the count of `mul_base` of `jolt_field`; the figure of three in a comment of the source counts the products before reduction. No arithmetic factor separates the source's later encode from the port's. Nine products at `c` are 2.75 ns at point 1 and 1.35 at point 2 against the 1.50 measured, and a level-0 butterfly at `3c` is 0.92 and 0.45 against the 0.535 measured, so `c` at point 1 overstates a product inside the source's transform. The transfer of the measured rows to the port is not verified, since layout, scheduling and host load differ. The model takes the measured rows, and item 4 replaces them with the port's.

**Model, at `t = 22`, one thread.** The model is computed from one record by one rule. The record is the unit table above, whose decimals are taken as exact, and the two phases that the model takes whole from the source, at their measured medians of 6.98 ms and 0.78 ms. Every other phase is a count times a unit price. Phases, sums and per-cycle figures are carried unrounded, and rounding is applied once, to a figure that is displayed or to a threshold, never to a value that enters a later computation. The milliseconds of the table are displays to two decimals.

| Phase | Operations (counted) | Point 1, ms | Status |
|---|---|---:|---|
| Level-0 encode | 301,989,888 `nb0` | 161.56 | source unit; includes a transposition that §2 removes |
| Level-0 tree | 4,718,591 `hb0` | 136.84 | source unit |
| Commit sample | `(5·2^23 + 12·2^18) c` = 45,088,768 `c` | 13.75 | estimated |
| **Commit** | | **312.16** | |
| Tables, weights and round 1 | 163,577,856 `c` + 134,217,728 `Lw` | 49.89 + 80.53 = 130.42 | estimated |
| Rounds 2 to 23 | 150,994,908 `c` | 46.05 | estimated |
| Induced weights | at most 7,405,568 butterflies | 6.98 | source phase |
| Equality tables and later samples | `6,501,120 + 1,677,696` = 8,178,816 `c` | 2.49 | estimated |
| Later encodes | 43,515,904 `nb1` | 65.27 | source unit |
| Later trees | 3,440,636 `hb1` | 88.42 | source unit |
| Queries and assembly | 408 positions | 0.78 | source phase |
| **Open** | | **340.43** | |

Per cycle, dividing the unrounded sums by `2^22`, that is 74.423743 ns for commit and 81.164377 ns for open, 155.588120 ns together (estimated). At point 2, where only `c` changes, 72.757493 and 69.237131. The set-up of the tables of `Φ_α` is under 25,000 operations and is charged as zero in the model; the benchmark times it inside the phase (§9), and the release points of §9 add no copy. The estimated rows of the opening are 178.97 ms of the 340.43, 53%, and none of them has a measured counterpart: the source's ring switch and rounds take `447.2 + 122.8 = 570.0` ms for a reduction of shape 64 by 192 over `2^24` symbols with the slices computed. This is the least certain part of the model. Without the composed lookup round 1 would cost `45·2^22` multiplications and the opening 82.99 ns per cycle; what the composed lookup has to show in measurement is that tables of 192 KiB with 48-byte entries are read at the price of `Lw`. With the six-product composition in place of the E-by-V kernel the model is 75.03 and 82.08 ns; what the kernel has to show is that its XORs do not cost the product it saves.

**Thresholds.** A threshold is single-thread nanoseconds per cycle at `log_t = 22`: the unrounded model times 1.25, rounded to the nearest nanosecond, as in the kernels spec. For commit `74.423743·1.25 = 93.03`, which gives 93, and for open `81.164377·1.25 = 101.46`, which gives 101. At point 2 the products are 90.95 and 86.55.

| Benchmark | Model, point 1 | Threshold | Model, point 2 | Threshold at point 2 |
|---|---:|---:|---:|---:|
| `bits_whir/commit` | 74.42 | 93 | 72.76 | 91 |
| `bits_whir/open` | 81.16 | 101 | 69.24 | 87 |

The open threshold is 0.04 below the point at which it would round to 102, and it does not depend on the precision of the two source phases: at their displays to one decimal, 7.0 and 0.8 ms, the product is 101.47. The thresholds of point 1 are in force and are provisional: they are forecasts from a model, and no run has shown that they can be met. They move only when a row of the unit table is replaced by a measurement of the operation that the row names, on a quiet host and on this scheme's code, and then every model and threshold is recomputed in the same change; no threshold moves to meet a result. Every source row above is to be replaced in that way. At `log_t = 20` the benchmark reports and has no threshold: the schedule has four levels and a different final message, and no measurement exists at that size.

**Twelve threads.** The kernels spec divides a single-thread threshold by 9.6 for twelve threads. This spec does not rely on that factor: acceptance on twelve threads is wall time, measured directly. The one measurement available, of the source, gives 67.8 ms and 142.6 ms, a speed-up of 4.59 for commit and 5.12 for open over its single thread, on a loaded host with eight performance and four efficiency cores. The data cannot separate the load from the memory traffic of the transform and the hashing and from the core mix, and the ratio of the source says nothing about the scaling of this scheme's bridge. The provisional requirement at `log_t = 22` on twelve threads is the single-thread threshold in wall time divided by the ratio measured for the source: `93·2^22 ns / 4.593 = 85` ms for commit and `101·2^22 ns / 5.118 = 83` ms for open (estimated; the ratios are `311.27/67.77` and `729.88/142.61`, and the quotients are 84.93 and 82.77). The benchmark item measures both on a quiet host, and the requirement is then replaced under the rule for unit rows. At 9.6 the figures would be 41 ms and 44 ms; that is an aspiration.

**Budget.** The prover's budget is 2,100 ns per cycle on one core. The scheme takes `93 + 101 = 194` ns of it, 9.2%; with the 745 ns of the kernels' thresholds, 939 ns are allotted and 1,161 remain. The prover's wall-time target of 918 ms at `2^22` cycles on twelve threads is that budget at 9.6 effective cores. The provisional twelve-thread requirement of this scheme, 168 ms, is 18.3% of it where the single-thread share is 9.2%, so meeting the thresholds above does not certify the wall-time target, which is measured on the whole prover. For reference, the source measures 207.9 ms for commit and open together on twelve threads on the loaded host.

**Memory.** Counted, §9: the dominant buffers total 576 MiB at the peak, in the fold of round 1, and 704 MiB with the shared rows, under the allocation lifetimes that §9 specifies. These are dominant-buffer subtotals and not the complete peak: the tables of `Φ_α` and the split equality tables add 256 KiB until the end of the first pass of round 1, and the scalars, the accumulators of the workers, the domain constants and the proof under construction add further capacity (§9 lists them). The source measures 1,325 MiB of resident memory with its packed copy. The benchmark reports the full peak of requested bytes and the shared rows separately, and the requirement is on that measured inventory: the peak, auxiliaries included, does not exceed the subtotal of 576 MiB by more than 5%, that is 604.8 MiB, at `log_t = 22`. An implementation that only truncates its vectors does not meet it (688 MiB, §9).

The recorder is the counting allocator of the bench support. It counts the sizes of the layouts requested from the allocator, which are capacities, and neither the lengths of vectors nor resident memory. One allocation interval starts after the rows, the set-up and the warmed thread pool exist and before `commit`; it spans `commit`, the retained state and `open`; and it ends with the opening proof still alive. The figure reported is the peak of live requested bytes in the interval minus the live bytes at its start, and the 128 MiB of the shared rows are reported separately and once. Parallel work is joined before a phase snapshot, and the live and the peak values are recorded at each release point of §9. A measurement of `open` alone starts with the retained state already resident and adds its capacity, 288 MiB at `log_t = 22`, explicitly; the combined interval does not add it a second time. The counters do not observe the temporary overlap inside the system allocator's `realloc`, the allocator's overhead or resident memory, which is one reason why §9 forbids reclaiming capacity by reallocation.

**Verifier.** Counted at `t = 22`, with at most 408 distinct positions over the five levels. The bridge is three separate pieces of work: the recurrence for `G`, `128·23 = 2,944` products in `E`; its final combination, 127 products in `E` by Horner's rule; and the computation of `τ` from the request, which is 128 products in `H` for the `s_b`, a transposition of bits that gives the `t_h` with no product, and 127 products in `E` by Horner's rule. Together that is 3,198 products in `E` and 128 in `H`. The lane combinations are `260·32 + 148·16 = 10,688` products: 8,320 of an element of `E` with a symbol of `V` at level 0 and 2,368 in `E` after it. A query weight `W~_x(q)` is a product of `c` factors: forming the factors is `c` products of an element of `E` by an element of `K`, and multiplying them is `c − 1` products in `E`. Over levels 0 to 3 that is `Σ Q_i·c_i = 6,116` products of `E` by `K` and `Σ Q_i·(c_i − 1) = 5,728` products in `E`. The 20 positions of the last level are checked on the final message directly, 80 products of `E` by `K`. Not counted here: the powers of `λ` and the products by them, the `Ŵ_l(x)` in `K`, the equality polynomials of the samples, the evaluation of the final message, the lane combination of the commit sample and the updates of `σ`. Hashing: a tree with `n` distinct opened leaves and `g` sibling digests costs `n·(b + 1) + g − 1` BLAKE2s compressions, where a leaf is `b = 8` blocks at level 0 and 6 after it and each of the `n + g − 1` internal nodes is one. That is 7,587.33 compressions in expectation and 7,939 at most (computed from the counts of §6). The bound `Σ Q_i·(b_i + d_i) = 260·27 + 65·24 + 37·23 + 26·22 + 20·21 = 10,423`, which lets every path be disjoint, is valid and is not attained. The source verifies in 1.78 ms (measured), which is not a measurement of this verifier. The benchmark reports the verifier's time; it has no threshold.

**Proof size.** §6: 326,320.45 bytes expected (computed) and 337,592 at most (counted) for the opening, and 800 for the commitment. The benchmark item reports the distribution over 1,000 transcripts, and the acceptance is that no opening exceeds 337,592 bytes and that the mean is within 1% of the expectation.

## Design

### Architecture

#### 1. The committed object and the claim

The table has `256·T` bits. Row `j` is a `BitsRow = [u64; 4]`, and the bit of column `col` of row `j`, at flat index `col + 256·j`, is bit `col mod 64` of word `⌊col/64⌋`. The packed table takes two consecutive words to one symbol: `p[z] = pack(row_j[2g], row_j[2g + 1])` at `z = g + 2·j`, `g < 2`. It has `2^μ` symbols, `μ = t + 1`. The 16 canonical bytes of a symbol are its two coefficients in order, each little-endian, so the symbols in index order are the words of the row buffer in word order: word `4j + 2g + h` of the buffer, `h < 2`, is coefficient `h` of `p[g + 2·j]`, and bit `i` of that word is flat bit `i + 64·h + 128·(g + 2·j)` of the table. This is an order of words and not of bytes. Where a word becomes bytes, in a leaf or on the wire, it is written little-endian by an explicit conversion on every host, and never by reinterpreting memory. The honest prover never builds the packed table. The seven low column variables are inside a symbol: variables 0 to 5 are the bit within a word and variable 6 selects the coefficient. The high column variable and the `t` cycle variables are the variables of `p`, in that order.

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

for `f` with values in `K`, in `V` or in `E`. It is the Reed-Solomon code of the polynomials of degree below `2^c` on `S_d`, of rate `2^(c−d)`. Every `X_w(x)` is in `K`, so the encoder acts on each coefficient in `K` separately: it maps `V`-valued messages to `V`-valued codewords, and `Enc(pack(a_0, a_1)) = pack(Enc(a_0), Enc(a_1))` for two messages of words. For a fixed position `x` the map `f ↦ Enc(f)[x]` is the inner product with the vector `W_x[w] = X_w(x)`, whose multilinear extension is a product of `c` factors:

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

The commitment is `root_0` and the `2^(k_0)` values `y_0`: `32 + 24·32 = 800` bytes at `t = 22`. The sample is one evaluation of every lane's message at a common point, and not one evaluation of `p` at a point of `E^μ`, for a reason of cost: a claim about `p` would join the sumcheck at level 0 with a weight of `2^μ` elements of `E`, while the lane values combine, after the lane variables are folded, into one claim about `f_1`, whose weight has `2^(c_0)` elements (§4, §9). `verify_commit` validates the geometry, `1 ≤ t ≤ 32`, and the shape of the commitment, `2^(k_0)` values `y_0` for that `t`, and returns `UnsupportedGeometry` or `Shape` with part `LaneValues` otherwise; then it absorbs, draws, and keeps `(t, root_0, z_0, y_0)` as its state. It makes no algebraic check. `verify_opening` first checks that the geometry of the request is the one retained (`GeometryMismatch`) and that the request has 8 column coordinates, `t` cycle coordinates and 256 column values (`Shape`), before it indexes any of them. The error type, the order of the checks and the rule for allocations are in §6. What the sample buys is invariant 2. The word in the leaves is within the decoding radius of at most `L_0` codewords of the interleaved code (§8), where distance counts the positions at which any lane differs. For each pair of members choose one lane in which their messages differ: the two lane messages agree at a uniform point of `E^(c_0)`, drawn after the root, with probability at most `c_0/|E|`, and agreement in every lane implies agreement in that one. So after step 3 at most one member of the list is consistent with `(z_0, y_0)`, and possibly none, except with probability `C(L_0, 2)·c_0/|E|`. The bound has no factor for the number of lanes, and one point shared by all lanes loses nothing. The point is uniform over all of `E^(c_0)`: a point that falls in `K` or on the Boolean cube is used as drawn. A commitment that is a root alone binds the list, and every error term of the front end would then be paid once per member of the list; at the reference size, where `L_0 = 187.0`, that is a loss of `log2 187 = 7.54` bits on terms that have no margin (§8).

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

The products `t_h·α^h` and `p[z]·w[z]` are products in `E` of an element of `V` by an element of `E`. The verifier computes `τ` from `C`, `rho[7]` and `α`: 128 products in `H` for the `s_b`, a transposition of bits that gives the `t_h` with no product, and Horner's rule on `t_127, …, t_0`, 127 products in `E`. The prover sends nothing for the bridge. The identity `(bridge)` is the first claim of level 0.

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

where `x_2 < x_3 < …` are the `n_{i−1}` distinct positions queried at level `i−1` and `W~_x` is taken with `c = c_{i−1}` on the domain of dimension `d_{i−1}`. The commit sample is a claim about `f_1`: the lane values `y_0[u]` are evaluations of the lanes of `f_0` at `z_0`, and `f_1` is their combination with `eq_E(a^(0), ·)`, so `f~_1(z_0) = Σ_u eq_E(a^(0), u)·y_0[u]`. It is the last claim of level 1, or, when level 0 is the only level, an equation on the final message. A weight that enters at level `i` is a function of the variables still free at that level, so it is evaluated at the suffix of `q` and carries no factor for the earlier coordinates. The verifier's work is the recurrence of §3, one equality polynomial per level, the query weights, the lane combinations `c_x`, and the hashes of the multiproofs. A query weight `W~_x` at a point is a product of `c_{i−1}` factors: once the `Ŵ_l(x)` have been computed in `K`, forming the factors is `c_{i−1}` products of an element of `E` by an element of `K`, and multiplying them is `c_{i−1} − 1` products in `E` (counted in Performance).

The opened leaves are not absorbed. They are fixed by a root that is absorbed before the positions are drawn, up to a collision of the hash, which the ledger prices separately.

#### 5. Parameters

The schedule is a function of `μ = t + 1`, for every `t` that the front end admits, `1 ≤ t ≤ 32`:

```text
k_0 = min(5, μ − 2),   c_0 = μ − k_0,   d_0 = c_0 + 1;
while c_i ≥ 6:   k_{i+1} = 3 if t ≥ 31 and c_i ≤ 8, else 4;
                 c_{i+1} = c_i − k_{i+1},   d_{i+1} = d_i − 1;
R = number of levels,   res = c_{R−1} ∈ {2, 3, 4, 5}.
```

Level 0 has rate 1/2 and each later level has a rate 8 times lower, since the message shrinks by 16 and the domain by 2. The exception is the last level at `t = 31` and `t = 32`, where the fold is 3: the levels are `c = 27, 23, 19, 15, 11, 7, 4` and `c = 28, 24, 20, 16, 12, 8, 5`, the last level has 8 lanes, leaves of 192 bytes and a rate 4 times lower than the level before it, and `d` is what a fold of 4 would give. With a fold of 4 there the fold row of the last level is at `2^-129.76` and `2^-128.91` at the smallest slack `m = 3`, so no parameter meets the reserve of the rule below (computed). At `t = 1`, `k_0 = 0`: level 0 has one lane, no fold round and no fold row in the ledger; it draws no `λ` and no position, its only message after the commitment is the final one, and the `μ = 2` rounds of the opening are closing rounds on the bridge claim. At `t = 22`: `μ = 23`, `R = 5`, `res = 2`.

| Level | `k_i` | `c_i` | `d_i` | Rate | Symbol | Leaf bytes | `Q_i` |
|---|---:|---:|---:|---|---|---:|---:|
| 0 | 5 | 18 | 19 | 1/2 | `V` | 512 | 260 |
| 1 | 4 | 14 | 18 | 1/16 | `E` | 384 | 65 |
| 2 | 4 | 10 | 17 | 1/128 | `E` | 384 | 37 |
| 3 | 4 | 6 | 16 | 1/1,024 | `E` | 384 | 26 |
| 4 | 4 | 2 | 15 | 1/8,192 | `E` | 384 | 20 |

There is no grinding at any level. The query counts are constants of the verifier, one row per `t`. The table also gives the folds `k_i`, the slack parameter `m_i` of each level, and the smallest algebraic row and the smallest position row of the ledger of §8 as `−log2` of the error, truncated to three decimals; "none" marks a size at which every position is opened and the position error is zero.

| `t` | `k_0, k_1, …` | `Q_0, Q_1, …` | `m_0, m_1, …` | Algebraic | Positions |
|---:|---|---|---|---:|---:|
| 1 | 0 | all | 3 | 185.011 | none |
| 2 | 1 | all | 3 | 178.423 | none |
| 3 | 2 | all | 3 | 177.423 | none |
| 4 | 3 | all | 3 | 176.423 | none |
| 5 | 4 | all | 3 | 175.423 | none |
| 6 | 5 | all | 3 | 174.423 | none |
| 7 | 5 | all | 3 | 173.758 | none |
| 8 | 5 | all | 3 | 172.907 | none |
| 9 | 5 | all | 3 | 171.978 | none |
| 10 | 5, 4 | all, 59 | 3, 38 | 150.628 | 128.032 |
| 11 | 5, 4 | 254, 62 | 838, 45 | 130.509 | 128.000 |
| 12 | 5, 4 | 256, 63 | 511, 97 | 133.083 | 128.000 |
| 13 | 5, 4 | 257, 64 | 430, 63 | 133.331 | 128.001 |
| 14 | 5, 4, 4 | 257, 64, 35 | 544, 127, 29 | 130.638 | 128.000 |
| 15 | 5, 4, 4 | 258, 64, 36 | 341, 255, 35 | 132.582 | 128.000 |
| 16 | 5, 4, 4 | 258, 65, 37 | 356, 43, 17 | 131.695 | 128.001 |
| 17 | 5, 4, 4 | 258, 65, 37 | 364, 45, 23 | 130.535 | 128.001 |
| 18 | 5, 4, 4, 4 | 259, 65, 37, 25 | 247, 46, 28, 16 | 132.328 | 128.001 |
| 19 | 5, 4, 4, 4 | 259, 65, 37, 26 | 248, 46, 31, 8 | 131.299 | 128.002 |
| 20 | 5, 4, 4, 4 | 259, 65, 37, 26 | 249, 47, 33, 12 | 130.270 | 128.005 |
| 21 | 5, 4, 4, 4 | 260, 65, 37, 26 | 187, 47, 34, 14 | 131.331 | 128.000 |
| 22 | 5, 4, 4, 4, 4 | 260, 65, 37, 26, 20 | 187, 47, 35, 16, 5 | 130.331 | 128.000 |
| 23 | 5, 4, 4, 4, 4 | 261, 65, 37, 26, 20 | 151, 47, 35, 17, 7 | 130.869 | 128.003 |
| 24 | 5, 4, 4, 4, 4 | 262, 65, 37, 26, 20 | 126, 47, 35, 18, 10 | 131.169 | 128.002 |
| 25 | 5, 4, 4, 4, 4 | 262, 65, 37, 26, 20 | 126, 47, 36, 18, 12 | 130.169 | 128.008 |
| 26 | 5, 4, 4, 4, 4, 4 | 263, 65, 37, 26, 20, 16 | 108, 47, 36, 19, 13, 7 | 130.277 | 128.002 |
| 27 | 5, 4, 4, 4, 4, 4 | 264, 65, 37, 26, 20, 17 | 95, 47, 36, 19, 14, 3 | 130.197 | 128.011 |
| 28 | 5, 4, 4, 4, 4, 4 | 266, 65, 37, 26, 21, 17 | 77, 47, 36, 19, 4, 3 | 130.135 | 128.025 |
| 29 | 5, 4, 4, 4, 4, 4 | 267, 65, 38, 26, 21, 17 | 70, 47, 11, 19, 4, 3 | 130.157 | 128.025 |
| 30 | 5, 4, 4, 4, 4, 4, 4 | 268, 66, 38, 27, 21, 17, 14 | 64, 24, 11, 6, 4, 3, 3 | 130.028 | 128.005 |
| 31 | 5, 4, 4, 4, 4, 4, 3 | 271, 66, 38, 27, 21, 17, 15 | 52, 24, 11, 6, 4, 3, 3 | 130.513 | 128.052 |
| 32 | 5, 4, 4, 4, 4, 4, 3 | 273, 66, 38, 27, 21, 17, 15 | 46, 24, 11, 6, 4, 3, 3 | 130.389 | 128.029 |

**The rule.** The table is the output of the following rule, which is the normative derivation. Every decision in it is an inequality of rational numbers, and no logarithm, square root or floating-point value enters a decision; the two decimal columns are displays (computed at 90 digits from the exact values). The targets are: every sample, batching, fold, bridge and closing row of the ledger of §8 at most `2^-130`, which is a reserve of two bits on the algebraic rows, and every position row at most `2^-128`.

Levels are taken in order. For level `i`, with `ϱ = (2^(c_i) − 1)/2^(d_i)` and `n = 2^(d_i)`, and for an integer `m ≥ 3`, put `η = √ϱ/m`, `h = m + 1/2`, `L = m/(2ϱ)` and

```text
U = n·(2·h^5/(3ϱ) + h) + h,        B = n·h·(m + 1)/m,        so that   a = U/√ϱ − B
```

for the constant `a` of §8. The integer `m` is admissible when three conditions hold. (1) If `k_i ≥ 1`, the fold row at its worst round `j = 1`, `(2·L + 2^(k_i − 1)·a)/|E| ≤ 2^-130`: with `R = 2^62 − 2·L + 2^(k_i − 1)·B` this is `R ≥ 0` and `(2^(k_i − 1)·U)^2 ≤ ϱ·R^2`. Later rounds halve the part with `a`, so `j = 1` decides every round, and a level with `k_i = 0` has no fold row and no condition (1). (2) The sample that binds the level: `L·(L − 1)·c'/2 ≤ 2^62`, with `c' = c_0` at level 0 and `c' = c_{i−1}` after it. (3) For `i ≥ 1`, batching: `(J_i − 1)·L ≤ 2^62`, with `J_i` of §8 taken at `n_{i−1} = P_{i−1}`, the effective position count already chosen for the level before. All three left sides increase with `m`, so the admissible values are an interval that starts at 3, and a schedule is valid only if 3 is admissible at every level.

For an admissible `m` the query count `Q(m)` is the least `Q` with `(ϱ·((m + 1)/m)^2)^Q ≤ 2^-256`, which is the position row `(√ϱ + η)^Q ≤ 2^-128` squared, and its effective count is `P(m) = min(Q(m), n)`. The rule takes the least effective count over the admissible `m` and, among the `m` that attain it, the smallest. When that count is `n` the entry is "all": every position is opened, no position is drawn, the position error is zero and `m = 3`. The bridge `127/|E|` and a closing round `2/|E|` do not depend on `m` and are below `2^-130`. The rule ran outside the repository (computed); the parameters test of Execution, item 2, runs it again in the repository and compares it with the frozen integers. The verifier contains the integers and not the rule (invariant 8).

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

An element of `E` is absorbed as its 24 canonical bytes and a root as its 32 bytes. Nothing of the opening request is absorbed again: the front end has absorbed `C` and has drawn `rho` and `r_6` from the same transcript, so its state binds them when `whir_open` is absorbed. The same holds at commit for the geometry, which the scheme does not absorb. This is a precondition on every caller of the four functions and not only on the front end of the experiment: before `commit` and `verify_commit` the transcript binds the geometry, and before `open` and `verify_opening` it binds `C`, `rho` and `r_6`. A caller that supplies them without having absorbed them, or having derived them from another transcript, is outside the soundness statement of §8. `y_0` is absorbed as the concatenation of its `2^(k_0)` elements in lane order, in one call. The scheme absorbs nothing after the last closing round.

**Challenges.** An element of `E` is one call of `squeeze_bytes` for 24 bytes: two draws of 16 bytes, their little-endian encodings concatenated, the first 24 bytes kept and read as three little-endian `u64` coefficients. The last 8 bytes of the second draw are discarded and are not carried into the next element. Vectors are drawn coordinate by coordinate in index order, one call per coordinate. The boundaries of the calls are part of the protocol: a point of `c` coordinates is `c` calls and `2c` draws, and one call for `24·c` bytes, which would make `⌈3c/2⌉` draws, is a different protocol. The positions of level `i` are one call of `squeeze_bytes` for `4·Q_i` bytes; position `j` is the little-endian `u32` at bytes `4j..4j+4`, reduced to its low `d_i` bits. Since `2^(d_i)` divides `2^32` the positions are uniform and independent. `d_i ≤ 29` for every admitted `t`. A level whose `Q_i` is "all" makes no call: its positions are `0, …, 2^(d_i) − 1`.

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

**Errors.** The scheme has one error type, `WhirError`, in `whir/error.rs` of the verifier crate. It is `BitsCommitmentScheme::Error` for `WhirBits`, so `commit`, `open`, `verify_commit` and `verify_opening` all return it, and it implements `Error + Send + Sync + 'static`. Its variants and the one check that produces each are:

| Variant | Produced when |
|---|---|
| `UnsupportedGeometry { log_T: usize }` | `log_T` is outside `1..=32`, in any of the four functions and in the schedule constructor |
| `GeometryMismatch { expected: BitsGeometry, actual: BitsGeometry }` | the geometry of an opening request differs from the one retained at commit |
| `Shape { part: WhirPart, expected: usize, actual: usize }` | a typed value has a length that the geometry does not give it |
| `CountMismatch { level: usize, part: WhirPart, expected: usize, actual: usize }` | `n_i` (part `Leaves`) or `g_i` (part `Digests`) of level `i` is not the count that the drawn positions determine |
| `LengthOverflow { part: WhirPart }` | a checked sum or product of lengths or offsets overflows `usize` |
| `Allocation { part: WhirPart }` | a fallible reservation fails |
| `MerkleAuthentication { level: usize }` | the root recomputed from the leaves and digests of level `i` is not `root_i` |
| `FinalCodeMismatch { position: usize }` | at the last level, `c_x ≠ Σ_w f_R[w]·W_x[w]` at position `x` |
| `CommitSampleMismatch` | with one level, `f~_R(z_0) ≠ Σ_u eq_E(a^(0), u)·y_0[u]` |
| `ClosingIdentityMismatch` | `σ ≠ f~_R(q[μ−res..μ))·Ω(q)` at the end |

`WhirPart` is an enum without data that names what was measured: `Rows` (the row buffer given to `commit`, `2^t` rows), `ColumnPoint` (8), `CyclePoint` (`t`), `Columns` (256), `LaneValues` (`y_0`, `2^(k_0)`), `Levels` (`R`), `Rounds` (the round messages of a level, `k_i`, and the closing rounds, `res`), `FinalValues` (`f_R`, `2^res`), `Leaves` (the number of leaves of a level and the length of each, `2^(k_i)` symbols) and `Digests`. With several mismatches the first in the order of the wire is reported. The checks run in this order: geometry; the shapes of the request and of the typed commitment and proof, every one before any element is indexed; then, level by level in the order of §4, the counts once the positions of the level are known, the multiproof, and the terminal equations. The verifier functions validate the typed proof in full and do not assume that it came from `read`, since a caller may construct one directly.

Every buffer that the scheme's code allocates, on either side, has a size computed by checked arithmetic and is reserved by `try_reserve_exact`, a failure of which is `Allocation`; `Vec::with_capacity`, `vec![…; n]` and `collect` into a vector of unvalidated length do not satisfy this. `BitsWire::read` returns `None` wherever it would produce one of these errors, because its signature has no error slot, and `write` cannot fail. No variant carries a string.

At `t = 22` the fixed part is `23·48 + 4·56 + 4·24 + 5·8 = 1,464` bytes (counted). The query part depends on the positions. A node that spans a fraction `π = 2^(h−d)` of the leaves, at height `h` of a tree of depth `d`, is sent as a sibling digest when no position falls under it and one falls under its sibling, which for `Q` draws has probability `(1 − π)^Q − (1 − 2π)^Q`; the two events are not independent and are not multiplied. So

```text
E[g] = Σ_{h<d} 2^(d−h)·( (1 − 2^(h−d))^Q − (1 − 2^(h+1−d))^Q ),        E[n] = 2^d·(1 − (1 − 2^(−d))^Q),
```

which gives 259.936, 64.992, 36.995, 25.995 and 19.994 distinct leaves and 2,623.69, 721.44, 404.06, 271.39 and 196.51 digests for the five levels, 324,856.45 bytes of leaves and digests, and an opening of 326,320.45 bytes in expectation (computed). For `q` distinct leaves the multiproof has at most `q·(d − ⌈log2 q⌉) + 2^⌈log2 q⌉ − q` digests, attained by balanced positions: 2,852, 778, 434, 292 and 212 at the query counts, so the opening is at most 337,592 bytes: 1,464 fixed, `260·512 + 148·384 = 189,952` of leaves and `4,568·32 = 146,176` of digests (counted). The bound `Σ Q_i·(leaf + 32·d_i) + 1,464 = 429,976` bytes, which lets every path be disjoint, is valid and is not attained. The whole proof of the experiment is the front end's 15,024 bytes, its 26-byte envelope, the 800-byte commitment and the opening, 342,170.45 bytes in expectation.

#### 7. The packing and the opening field

`H = F_2[x]/(x^128 + x^7 + x^2 + x + 1)` and `E = K[y]/(y^3 + y + 1)` have degrees 128 and 192 over `F_2`, and 128 does not divide 192, so `H` is not a subfield of `E`. The claim is in `H`. The bridge of §3 exists because bits are packed into symbols and not because of the missing embedding. Extraction of a coordinate is linear over `F_2` and not over `H`, so `bit_b(Σ_z ω_z·p[z])` is not `Σ_z ω_z·bit_b(p[z])` in any field, and a design whose opening field contains `H` keeps a tensor reduction of the same shape.

A code alphabet does not have to be a field. It has to be closed under addition and under multiplication by the twiddles of the code, which are in `K`, and it needs a basis over `F_2` to carry bits. `V` has both. No step of the scheme multiplies two symbols of level 0: the sumcheck multiplies a symbol by a weight in `E`, and the first fold multiplies it by a challenge in `E`, after which every value is in `E`. The representation of `F192` as three coefficients in `K` is what makes this free: a symbol of `V` is two words, the transform of level 0 is the transform over `K` on twice as many lanes, the leaves are the 512 bytes of 64 words, and Lemma 1 of §8 shows that every candidate of level 0 is `V`-valued.

Counts at `t = 22`, for the prover algorithm of §9 in three designs: this spec; 64 bits to a symbol of `K` with the 64 by 128 bridge, `μ = 24` and `k_0 = 6`; and 128 bits to a symbol of `H` with an opening field `E'` of `2^256` elements that contains `H` (Alternatives Considered gives its product counts). All three use the composed lookup of §9.

| Quantity (counted) | `V ⊂ F192`, 128 bits | `K ⊂ F192`, 64 bits | `H ⊂ F256`, 128 bits |
|---|---:|---:|---:|
| Symbols at level 0; leaf bytes | `2^23`; 512 | `2^24`; 512 | `2^23`; 512 |
| Level-0 encoding | 905,969,664 | 905,969,664 | 905,969,664 |
| Later encodings, 43,515,904 butterflies | × 9 = 391,643,136 | × 9 = 391,643,136 | × 12 = 522,190,848 |
| Commit sample, inner products | `5·2^23` = 41,943,040 | `3·2^24` = 50,331,648 | `8·2^23` = 67,108,864 |
| Sumcheck round 1 | `39·2^22` = 163,577,856 | `33·2^23` = 276,824,064 | `52·2^22` = 218,103,808 |
| Sumcheck later rounds | `36·(2^22 − 1)` = 150,994,908 | `36·(2^23 − 1)` = 301,989,852 | `60·(2^22 − 1)` = 251,658,180 |
| Induced weights, 7,405,568 butterflies | × 9 = 66,650,112 | × 9 = 66,650,112 | × 12 = 88,866,816 |
| **Subtotal of multiplications** | **1,720,778,716** | **1,993,408,476** | **2,053,898,180** |
| The same without the composed lookup | 1,745,944,540 | 2,043,740,124 | 2,079,064,004 |
| Lookups of `Φ`, entries of one element | 134,217,728 | 268,435,456 | 134,217,728 |
| BLAKE2s compressions, all trees | 8,159,227 | 8,159,227 | 9,142,267 |
| Vectors of round 1 (weights; folded message) | 192 MiB; 96 MiB | 384 MiB; 192 MiB | 256 MiB; 128 MiB |
| Commitment bytes | 800 | 1,568 | 1,056 |
| Verifier recurrence, products | 2,944 in `E` | 3,072 in `E` | 2,944 in `E'` |
| Query counts under the rule of §5 | 260, 65, 37, 26, 20 | 261, 65, 37, 26, 20 | 260, 65, 37, 26, 20 at the slack of this spec |
| Weakest algebraic term | `2^-130.33` | `2^-130.86` | `2^-194.33` at the slack of this spec |
| Field code that exists | all but the methods of item 3 | all but `fmadd_base` | none of `E'` |

The rows follow from §9. Level 0 is the same transform over `K` on 64 words per position in the first two designs, and 32 lanes of `H` at 6 multiplications per butterfly in the third. Round 1 of this spec is `6 + 5 + 5 + 11 + 12 = 39` multiplications per pair with the E-by-V kernel of §9, and its commit sample 5 per symbol. The 64-bit packing has twice the pairs in every round of the sumcheck, at `6 + 3 + 3 + 9 + 12 = 33` multiplications per pair of round 1, and twice the symbols at 3 each in the commit sample. The sumcheck of the third design runs over the same number of symbols as this spec in a wider field. The subtotal is not a total: it leaves out the equality tables and the later samples (8,178,816 multiplications in this spec, §9), the set-up of the tables, the reductions of per-thread accumulators and the challenges. Multiplication counts are not times either: they omit moves, additions and memory traffic, and the lookups are priced separately in Performance.

**The choice, and what reverses it.** The design of this spec is the first column. Against the 64-bit packing it has 272,629,760 fewer multiplications, half the lookups, 288 MiB less in round 1, one query fewer at level 0 and a commitment of 800 bytes for 1,568, with the same level-0 oracle, the same leaf bytes and the same later levels. Against the opening field of `2^256` elements it has 333,119,464 fewer multiplications, 983,040 fewer hash compressions and a field that exists. The choice is taken on these counts and is to be confirmed by the first measurement of the bridge phase: the tables, the two weight vectors and all rounds of the sumcheck, on one thread at `t = 22`. That phase exists once the prover of item 8 of Execution does, and item 9 measures it; item 6 measures its first-round part alone, 130.42 ms in the model, which is an early indication and not the decision. The model of Performance prices that phase at 176.48 ms for this spec and at 337.60 ms for the 64-bit packing (estimated, at the provisional unit prices: `314,572,764 c + 134,217,728 Lw` against `578,813,916 c + 268,435,456 Lw`). The choice is reversed if the measured time of that phase is above the count of the 64-bit packing repriced at the unit prices that items 3 and 6 measure: the 64-bit packing is then implemented behind the same traits and measured, and the faster of the two is kept. The parameters of §5 and the wire of §6 are those of the design that is kept.

#### 8. Security ledger

**Statement.** The scheme is analysed as an interactive protocol, round by round. A state function marks every partial transcript as doomed or not; the state is doomed at the start when the claim is false, and for each verifier challenge the probability that a doomed state becomes one that is not doomed is the error of that transition. The claim of this spec is conditional: under assumptions 1 and 2 and the invariant of Lemma 2, every algebraic transition of the scheme's own interactive protocol has error at most `2^-130` and every query transition at most `2^-128`, for every admitted `t`, with no grinding. That is a bound on the maximum over transitions. It is not a bound on the failure of a whole interactive execution, which the sum of the transitions bounds, and it is not a bound on the protocol compiled by the Fiat-Shamir transformation, for which this spec claims none. The relation is existential: an accepted opening implies that the commitment selects a table with the claimed partial evaluations. No extractor is given and knowledge soundness is not claimed (Open, 2). The Merkle term is stated separately below, at a declared query budget, and the compilation is an open obligation (Open, 3). The ledger is a list of bounds under stated assumptions; nothing in it is a proof of security of the implementation.

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
| Fold round `j ≤ k_i` of level `i`, which exists only when `k_i ≥ 1` | `2·L_i/\|E\| + 2^(k_i − j)·ε_i` | for some member of the current list of the partially folded oracle, a sumcheck round of degree 2 or, at level 0, the vanishing of its sample discrepancy (Lemma 2); or a member of the next list that is not the fold of a member of the current one |
| Sample `z_{i+1}` | `C(L_{i+1}, 2)·c_i/\|E\|` | as the commit sample, for level `i+1` |
| Positions of level `i` | `(1 − γ_i)^(Q_i)`, and 0 where every position is opened | every queried column agrees with a word that is `γ_i`-far |
| Closing round | `2/\|E\|` | a sumcheck round of degree 2 |

Two further matters are not transitions of the interactive protocol, and they are kept apart from the table.

*The hash.* A collision of BLAKE2s-256 as an ideal hash of 256 bits has probability at most `q_h·(q_h − 1)/2^257` for `q_h` distinct queries to it, and every row above is conditional on there being none. The spec fixes a declared regime for this term: `q_h ≤ 2^64` distinct queries to the ideal BLAKE2s-256 in total, those of the adversary and those of the honest parties together, for which the collision probability is below `2^-129`, since `2^64·(2^64 − 1)/2^257 < 2^-129` (derived). The regime is a declared work budget of the analysis. It is not a cap that the implementation enforces, and the number of hashes that the honest parties compute says nothing about the adversary's.

*The compilation.* The implementation is non-interactive, and its challenges come from the transcript's hash. No soundness bound is claimed for the compiled protocol until a compilation theorem, the model of the transcript's hash and the convention for counting its queries are selected (Open, 3). The interactive ledger alone does not establish a bound of the form `q_t` times the largest transition error, where `q_t` is the number of queries to the transcript's hash. `q_t` is a budget of its own, distinct from `q_h`, and this spec gives it no value.

At the reference size, with `|E| = 2^192`, as `−log2` of the error (computed at 90 digits from the exact values; the decisions are those of the rule of §5):

| Level | `ϱ` | `m` | `η` | `γ` | `L` | `Q` | Positions | Fold, `j = 1` | Sample | Batching | `2L/\|E\|` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `(2^18−1)/2^19` | 187 | 0.003781 | 0.28911 | 187.0 | 260 | 128.0001 | 130.331 | 173.74 | none | 183.45 |
| 1 | `(2^14−1)/2^18` | 47 | 0.005319 | 0.74469 | 376.0 | 65 | 128.028 | 137.73 | 171.72 | 175.41 | 182.44 |
| 2 | `(2^10−1)/2^17` | 35 | 0.002524 | 0.90913 | 2,242.2 | 37 | 128.022 | 136.33 | 166.93 | 174.82 | 179.86 |
| 3 | `(2^6−1)/2^16` | 16 | 0.001938 | 0.96706 | 8,322.0 | 26 | 128.021 | 138.32 | 163.63 | 173.72 | 177.97 |
| 4 | `(2^2−1)/2^15` | 5 | 0.001914 | 0.98852 | 27,306.7 | 20 | 128.889 | 142.16 | 160.94 | 172.50 | 176.26 |

The security columns are truncated and not rounded. The bridge is at 185.011 bits and a closing round at 191. The "Fold" column is the whole fold row at its worst round, `2·L_i/|E| + 2^(k_i − 1)·ε_i`, which the correlated-agreement part dominates; round `j` of a level is `j − 1` bits above it up to the constant `2·L_i/|E|`, so the five rounds of level 0 are at 130.331, 131.331, 132.331, 133.331 and 134.331 bits. The sample of row `i` is the one that binds level `i` (the commit sample for row 0). The weakest term is the positions of level 0, `(√ϱ_0 + η_0)^260 = 2^-128.000173`, and the weakest algebraic term is the fold of level 0 at `2^-130.331220`. With one query fewer at level 0 the slack that meets the position target is `m_0 = 249`, whose fold row is `2^-128.270474`, inside the target of 128 bits and outside the reserve; the reserve costs that one query, 800 bytes at most and 788.52 in expectation (§6). Over the admitted sizes the smallest position row is `2^-128.000020`, at `t = 11`, and the smallest algebraic row is `2^-130.028826`, at `t = 30`; for `t ≤ 9` every position of the single level is opened and the smallest row is above 171.9 bits. Every algebraic transition meets `2^-130` and every query transition `2^-128`, conditionally and with no grinding. The reserve of two bits absorbs a constant `a` that is larger by a factor of up to four; it does not establish assumption 1, whose check against the primary source stays open (Open, 1). The hash term depends on a query budget and is not in this table.

**Lemma 1 (the candidates of level 0 are `V`-valued).** Let `D = 2^(c_0)` and `n = 2D`, and let `g` be a codeword of the interleaved code over `E` on `S_(d_0)` that agrees with the level-0 oracle on more than a fraction `√ϱ_0` of the positions. Then the message of every lane of `g` is `V`-valued. *Proof in outline.* The agreement set has more than `√((D − 1)·n)` positions, which is at least `D − 1`, so it has at least `D`. Fix a lane and write its polynomial as `g_0 + y·g_1 + y^2·g_2` with each `g_k` in `K[X]` of degree below `D`, which is possible because `1, y, y^2` is a basis of `E` over `K`. The domain lies in `K`, so the value at a position `x` is `g_0(x) + y·g_1(x) + y^2·g_2(x)` with each `g_k(x)` in `K`. The symbol of the oracle at a position of agreement is in `V`, so `g_2` vanishes at `D` distinct points and is zero. The change from monomials to the basis `X_w` has coefficients in `K`, so the message is `V`-valued. Agreement of a position of the interleaved word is agreement in every lane, so the argument applies lane by lane. ∎ The lemma holds for every admitted `t`, since `c_0 ≥ 2`. The fold challenges are in `E` and the relevant code is the one over `E`, but its list at level 0 consists of tables of bits; the commit sample selects at most one of them, and §3 proves the claims about that one. Messages of later levels are folds with challenges in `E` and are `E`-valued.

**Lemma 2 (the commit sample through the lane folds).** Condition on no collision of the hash. For `0 ≤ j ≤ k_0` and challenges `a_1, …, a_j` let `O^(j)` be the level-0 oracle with its lanes folded over their `j` low variables with `eq_E((a_1, …, a_j), ·)`, a word of `2^(k_0 − j)` lanes; let `Γ_j` be the current list, the codewords of the interleaved code with `2^(k_0 − j)` lanes within the radius `γ_0` of `O^(j)`, of at most `L_0` members; and let `y_0^(j)` be the public vector `y_0` folded in the same way. For every `g ∈ Γ_j` its discrepancy with the commit sample is

```text
δ_j(g)[u] = g~(u, ·)(z_0) + y_0^(j)[u],        u < 2^(k_0 − j),
```

where `g~(u, ·)` is the multilinear extension of the message of lane `u` of `g`. The discrepancy is defined on every member of the current list, whether or not that member is the fold of a member of an earlier list, so the state function is defined on every partial transcript, including one that follows an earlier bad event. The state after round `j` is doomed when every `g ∈ Γ_j` either has `δ_j(g) ≠ 0` or violates the running claim `σ = Σ_z g(z)·ω^(j)(z)`; at `j = 0` this is the state after the bridge, in which the member that the sample selects, if there is one, violates the claim `σ = τ` and every other member has a nonzero discrepancy. Then round `j + 1` leaves the doomed states with probability at most `2·L_0/|E| + 2^(k_0 − j − 1)·ε_0`, which is the fold row. *Proof in outline.* Before the challenge `a`, put each `g ∈ Γ_j` in one of two classes. A member with `δ_j(g) ≠ 0`: its fold `g_a` with `a` has `δ_{j+1}(g_a)[u] = δ_j(g)[2u] + a·(δ_j(g)[2u] + δ_j(g)[2u + 1])`, because the lane messages of `g` and the vector `y_0^(j)` fold with the same `a`; some coordinate is affine in `a` and not identically zero, so the vector vanishes for at most one `a`. This event is counted whatever the member's sumcheck claim is. A member with `δ_j(g) = 0` violates the running claim, so the round polynomial that was sent and the member's own are different polynomials of degree 2 and agree at no more than two values of `a`. Each member is in exactly one class, so the union over `Γ_j` costs at most `2·L_0/|E|`, and not that plus a term for the sample. The remaining event is that `Γ_{j+1}` has a member that is not the fold of a member of `Γ_j`, which is mutual correlated agreement, `2^(k_0 − j − 1)·ε_0` under assumption 1. Outside these events every member of `Γ_{j+1}` is the fold of a member of `Γ_j` and has a nonzero discrepancy or violates the claim, so the state stays doomed. ∎ At `k_0 = 0` (`t = 1`) level 0 has no round, the lemma has no instance and the ledger has no fold row. After round `k_0` the residual vector has one entry, `f~_1(z_0) + Σ_u eq_E(a^(0), u)·y_0[u]`. It is the commit-sample claim that §4 adds to the claims of level 1, or checks on the final message when `R = 1`, and from there the source's argument at a level interface carries "violates the residual claim or the sample claim" as it carries its pool of claims, at the batching error with `J_1 = n_0 + 3`. The quantity `L_0·k_0/|E|`, `2^-182.13` at the reference size, bounds the event that some member with a wrong sample survives all `k_0` lane challenges. It is a diagnostic over the whole vector `a^(0)`, whose coordinates are drawn in `k_0` separate rounds with prover messages between them, and it is not a transition and not a row of the ledger.

**The bridge and the list.** The bridge error is `127/|E|` with no factor for the list: `α` is drawn after the commit sample has selected at most one member, and members that violate the sample are carried by Lemma 2.

**The direct check of the last level.** The claims of the last level's queries are checked against the final message, which is sent in the clear, and not batched; this removes the batching round that they would need and adds no term.

**The composition ledger.** Every challenge of the front end is in `H`, so the bounds of its own checks supply no guarantee of 128 bits for the composed protocol, whatever this scheme does. Those bounds, from the protocol spec (derived):

| Front-end check | Error | `−log2` |
|---|---|---:|
| Column reduction through `value()` alone, the contract | `8/2^128` | 125 |
| Column reduction as this scheme proves it, for a fixed wrong `C` (§1) | `1/2^128` | 128 |
| A sumcheck round of degree 6, the highest of the reference layout | `6/2^128` | 125.415 |
| A sumcheck round of degree 17, the highest admitted | `17/2^128` | 123.913 |
| The `8 + t = 30` coordinates of `tau` of `SpartanOuterF2` at `t = 22`, as one transition | `30/2^128` | 123.093 |
| The `a = 61` coordinates of `tau` of the output check, at the largest admitted `a`, as one transition | `61/2^128` | 122.069 |
| The 185 rounds of the reference layout, union bound over their degrees alone | `640/2^128` | 118.678 |

The guarantee supplied by these bounds is 125.415 bits per transition for a round of degree 6, the highest of the reference layout, and 123.913 bits for a round of degree 17, the highest admitted. They are upper bounds on error: they limit what this analysis guarantees, they do not show that an attack attains them, and a stronger guarantee needs further analysis or a front end that grinds or changes its field, which is outside this spec. They are read from single round bounds and are not a theorem about the composed protocol. With a vector of challenges counted as one transition the two vector rows are lower; counted one coordinate at a time they need an invariant that the protocol spec does not state; and the failure of a whole interactive execution is bounded by a sum, of which the last row is one part. This scheme improves one row, the column reduction, and the commit sample keeps every row from being multiplied by the list size. It does not improve the sumchecks, and a larger opening field would not either.

**Grinding.** There is none. Grinding `g` bits before the positions of a level are drawn does not change that round's error; it multiplies the work of each attempt at that round by `2^g`. Counted as error it would let the query counts drop to 226, 56, 32, 23, 17 at `g = 17` under the rule of §5 with the position target at `2^-111`, saving 56,640 of the 428,512 bytes of leaves and digests in the disjoint-path bound (counted; 41,784 of 332,112 bytes measured on the source), for an expected `5·2^17` hash evaluations of grinding. It buys nothing for the fold, sample, batching, bridge or front-end terms. It is not adopted, so that the target is met as an error bound.

**What "128 bits" does not cover.** The algebraic and query transitions of the scheme meet the target, conditionally. Three things do not. The front end's rows, as above. The hash: the collision bound is below `2^-129` in the declared regime `q_h ≤ 2^64` and is above `2^-128` beyond `q_h = 2^64.5`, a collision with constant probability costs on the order of `2^128` evaluations, and these figures are for an ideal hash against a classical adversary. The compilation: the rows are transitions of the interactive protocol, and no figure is stated for the compiled one.

#### 9. The prover

Counts are in carry-less multiplications of 64-bit words on aarch64, read from `jolt_field::binary`: a product in `K` is 3 (one product, a reduction of two); an element of `E` times an element of `K` is 9 (three products, three reductions), or 3 into an accumulator; a product in `E` is 12 (six products, three reductions), or 6 into an accumulator; a product in `H` is 6 (four products, a reduction of two). Every count below is counted from these and from the kernel that follows.

**The E-by-V kernel.** The product of `e = (a_0, a_1, a_2)` in `E` with `v = pack(b_0, b_1)` in `V` is a named operation of the field item (Execution, item 3), with five carry-less products of words:

```text
d_0  = a_0·b_0;        d_1  = a_1·b_1;
c_01 = (a_0 + a_1)·(b_0 + b_1) + d_0 + d_1;
c_02 = a_2·b_0;        c_12 = a_2·b_1;
e·v  = ( d_0 + c_12,   c_01 + c_12,   d_1 + c_02 ).
```

The five products are unreduced 128-bit products, every sum is an XOR, and the three results are the unreduced coefficients of `1`, `y` and `y^2`. The formula is the expansion `a_0·b_0 + (a_0·b_1 + a_1·b_0)·y + (a_1·b_1 + a_2·b_0)·y^2 + a_2·b_1·y^3` with `y^3 = y + 1` and one Karatsuba step for the coefficient of `y`; it is the six-product kernel of `F192` at a second operand whose third coefficient is zero, and it needs no field representation other than the one of `jolt_field`. `F192Accumulator` holds exactly these three unreduced coefficients, so the kernel is 5 into an accumulator, `fmadd_base_pair`, and 11 reduced, `mul_base_pair`, with one reduction of two products per coefficient. The same product composed as `e·b_0 + (y·e)·b_1` from two products of `E` by `K` and a multiplication by `y`, which is a permutation and one addition of coefficients, is 6 into an accumulator and 12 reduced. The model takes the five-product kernel. Its exhaustive check on a toy field is an acceptance criterion ("Field"), and item 3 measures it against the composition before the unit prices are frozen, since it trades one product for additional XORs and one more live register.

**`commit`.** (1) Allocate the level-0 codeword, `2^(d_0)` positions of `2^(k_0)` symbols, `2^24` symbols, `2^25` words and 256 MiB at `t = 22`. For each position `w < 2^(c_0)` copy the `2^(k_0 + 1)` words of the row buffer into position `w` of each of the two cosets of the message (the rate is 1/2), and run the additive transform position-major, each butterfly acting on a whole row of words with one twiddle in `K`: `2^(k_0 + 1)·c_0·2^(d_0−1) = 64·18·2^18 = 301,989,888` butterflies, each one multiplication in `K` and two additions. (2) Hash `2^(d_0)` leaves and the tree: `2^19` leaves of 512 bytes are 8 compressions each and the `2^19 − 1` nodes one each, 4,718,591 compressions over 301,989,824 bytes. (3) Absorb the root, draw `z_0`, build the table `eq_E(z_0, ·)` of `2^(c_0)` elements of `E` (6 MiB, `2^18` products in `E`, `12·2^18 = 3,145,728` multiplications; this and every other count for an equality table in this spec is an allowance of one product per entry, where the doubling construction with one XOR for the complement uses `N − 1` for `N` entries), and compute the `2^(k_0)` values `y_0` in one pass over the rows with one accumulator per lane: position `w` adds `eq_E(z_0, w)·p[u + 2^(k_0)·w]` to accumulator `u` with the E-by-V kernel, 5 multiplications per symbol and `5·2^23 = 41,943,040` in all, and `y_0[u]` is accumulator `u` reduced. The sum `a_0 + a_1` of the kernel is formed once per position, since the element of `E` is the same for the `2^(k_0)` lanes of a position. Each worker owns `2^(k_0)` accumulators of 48 bytes, merged at the end.

**State between `commit` and `open`.** `ProverState` holds the `Arc` of the rows, the level-0 codeword, the level-0 tree, `z_0`, `y_0` and `t`. At `t = 22`: `2^25·8 = 268,435,456` bytes of codeword, `(2^20 − 1)·32 = 33,554,400` bytes of tree and `18·24 + 32·24 + 8 = 1,208` bytes of scalars, 301,991,064 bytes, plus the shared reference to the 128 MiB of rows, which the scheme does not own. `open` consumes the state.

**`open`, round 1.** Round 1 pairs the symbols `2k` and `2k + 1`, which are 32 adjacent bytes of the row buffer, for `k < 2^(μ−1)`. This kernel runs when `k_0 ≥ 1`; the case `k_0 = 0` is below. Let `e'[k] = eq_H(r[1..μ), k)`. It is the product of two entries of split tables: with `b = ⌊(μ − 1)/2⌋` and `a = μ − 1 − b`, the tables are `Lo[i] = eq_H(r[1..1+b), i)` for `i < 2^b` and `Hi[j] = eq_H(r[1+b..μ), j)` for `j < 2^a`, each in low-variable-first index order, and `e'[k] = Lo[k mod 2^b]·Hi[⌊k/2^b⌋]`, one product in `H` per pair. Both exponents are nonnegative at every admitted `μ`, and a table over no variable is the single entry 1. At `μ = 23` both tables have `2^11` elements, 64 KiB together. The two `eq_H` values of the pair are `r[0]·e'[k]` at `2k + 1` and `e'[k] + r[0]·e'[k]` at `2k`. The map `e ↦ Φ_α(r[0]·e)` is `F_2`-linear in `e` like `Φ_α`, so both are read from one set of 16 tables of 256 entries, one table per byte of `e'[k]`, whose entry for a byte value is the pair of the two maps on that byte: `16·256·48 = 196,608` bytes. With `D[k] = Φ_α(e'[k])` and `W1 = Φ_α(r[0]·e'[k])` the sums of the 16 entries, the prover stores

```text
D[k] = w[2k] + w[2k+1],        W0[k] = w[2k] = D[k] + W1,
```

two vectors of `2^(μ−1)` elements of `E`, and no product by `r[0]` is computed per pair. The first round message is `u_0 = Σ_k p[2k]·W0[k]` and `u_2 = Σ_k (p[2k] + p[2k+1])·D[k]`, two products of an element of `V` by an element of `E` into accumulators per pair, each the E-by-V kernel. After the challenge `a`, the message folds to `p[2k] + a·(p[2k] + p[2k+1])`, one reduced product of `E` by `V`, written to a new vector, and the weight to `W0[k] + a·D[k]`, one product in `E`, in place in `W0`. Per pair: `6 + 5 + 5 + 11 + 12 = 39` multiplications (the product in `H`, the two accumulated kernels, the reduced kernel and the product in `E`) and 16 table entries of 48 bytes, priced as 32 lookups of 24 bytes, over `2^22` pairs: 163,577,856 multiplications and 134,217,728 lookups. The tables are set up once per opening from the 128 powers of `α` (127 products in `E`), the 128 values `r[0]·x^h` (127 calls of `mul_x`), their images under `Φ_α` (at most `128·128` additions in `E`) and `2·16·255` additions of one entry to another: under 25,000 operations in `E`, charged as zero in the model against `2^22` pairs and timed by the benchmark inside the phase. The tables of `Φ_α` and the split tables are dropped at the end of the first pass, when `W0` and `D` are complete and before the fold of round 1 allocates its output. Level 0 has no other weight: the commit sample enters at level 1, where its equality table has `2^18` elements.

**`open` at `k_0 = 0`.** At `t = 1` level 0 has one lane and no lane round, so the round-1 kernel does not run and `W0`, `D`, the split tables and the tables of `Φ_α` are not allocated. The prover computes the dense bridge weight `w[z] = Φ_α(eq_H(r, z))` for the four `z < 2^μ` from the definition, the bits of `eq_H(r, z)` against the powers of `α`. It sends the final message `f_R = p`, four elements of `E` whose third coefficients are zero, opens all eight positions of level 0 with no sibling digest, and runs the `μ = 2` closing rounds as rounds on adjacent pairs of elements of `E`, with message `p`, weight `w` and starting claim `σ = τ`. The verifier checks the eight leaves against `root_0`, `c_x = O_0[x][0] = Σ_w f_R[w]·W_x[w]` for every `x`, the commit sample `f~_R(z_0) = y_0[0]`, and at the end `σ = f~_R(q)·w~(q)`. No `λ` and no position is drawn.

**`open`, later rounds.** Round `j ≥ 2` has `2^(μ−j)` pairs of elements of `E`: two products into accumulators for the message (12) and two products for the folds (24), 36 per pair and `36·(2^22 − 1) = 150,994,908` in all. The rounds pair adjacent elements from the first fold on, since the lanes are the low variables; the port takes the source's adjacent-pair rounds and not its dispatch on high-variable lanes or its rotation of the final point. At the start of level `i ≥ 1` the weights of the new claims are added: the equality table of the level's sample (and of the commit sample at level 1), and for the queries the vector `Σ_j λ_i^j·W_{x_j}`, which is the transpose of the encoder applied to the sparse vector with `λ_i^j` at position `x_j`. It is computed with one transposed transform on the domain of level `i−1`, `c_{i−1}·2^(d_{i−1}−1)` butterflies of `E` by `K`: 4,718,592, 1,835,008, 655,360 and 196,608 for the four later levels, 7,405,568 in all. That is the count of a dense transform and an upper bound, since the input has at most `Q_{i−1}` nonzero entries.

*The transform and its transpose, by index.* Write `c`, `d` for the level whose code is used. A codeword buffer has `2^d` positions, and position `2^c·j + w`, for `j < 2^(d−c)` and `w < 2^c`, belongs to block `j`. The encoder copies the message into every block, `buf[2^c·j + w] = f[w]`, and then runs the layers `l = c − 1` down to `0`. Layer `l` works on every aligned run of `2^(l+1)` positions; with `x_0` the first position of the run and `t = Ŵ_l(F64::from_raw(x_0))`, it maps each pair `top = buf[x]`, `bot = buf[x + 2^l]`, for `x_0 ≤ x < x_0 + 2^l`, to `top ← top + t·bot` and then `bot ← bot + top`. With lanes, an entry of the buffer is the row of `2^k` symbols of a position and the butterfly acts on the whole row. The transpose takes a buffer `g` of `2^d` elements of `E`, runs the layers `l = 0` up to `c − 1` with the transposed butterfly `s = top + bot`, `top ← s`, `bot ← t·s + bot` on the same pairs with the same `t`, and then sums the blocks: `out[w] = Σ_j g[2^c·j + w]`, for `w < 2^c`. This is `Enc^T`, including the transpose of the copy into blocks, and it satisfies `⟨Enc(f), g⟩ = ⟨f, Enc^T(g)⟩`.

*Scratch.* The induced weight of level `i` uses one buffer of `2^(d_{i−1})` elements of `E`, zeroed, into which `λ_i^j` is written at index `x_j` for the distinct positions in ascending order; the transposed layers run in place in that buffer; and the block sums are written to one output buffer of `2^(c_{i−1})` elements, which is then added to `ω`. A run of a layer that contains no queried position is zero and is skipped, which changes the runs that execute and not the buffer: workers own disjoint runs of the one buffer, and no second buffer of the size of the domain, no buffer per window and no copy of the domain or of the output per worker is allocated. At the interface of levels 0 and 1 at `t = 22` the scratch is the domain buffer of 12 MiB, the coefficient output of 6 MiB and the two equality tables of `z_1` and `z_0` of 6 MiB each, 30 MiB (counted). The port takes the source's transposed butterfly and its skipping of empty runs, and not its allocation of a buffer per window or its per-query expansion with one accumulator vector per worker.

The equality tables are `2^18` products in `E` each for `z_1` and `z_0` and `2^14`, `2^10`, `2^6` for the later samples, 541,760 products and 6,501,120 multiplications, with `λ_i` folded into the first entry; merging a new weight into `ω` is additions only.

**`open`, later commitments.** After the `k_i` rounds of level `i < R−1` the folded message is `f_{i+1}`, already in memory. It is encoded with `2^(k_{i+1})` lanes, 16 at `t = 22`, on the domain of dimension `d_{i+1}`: `16·c_{i+1}·2^(d_{i+1}−1)` butterflies of `E` by `K`, 29,360,128, 10,485,760, 3,145,728 and 524,288, in all 43,515,904. Its tree has `2^(d_{i+1})` leaves of 384 bytes, 6 compressions each: 3,440,636 compressions for the four later trees. The sample `y_{i+1}` is an inner product of `2^(c_i)` elements of `E` with the equality table of `z_{i+1}`, products into an accumulator: `6·(2^18 + 2^14 + 2^10 + 2^6) = 1,677,696` multiplications for the four samples. **Domain constants and twiddles.** `Ŵ_l` is linear over `F_2`, vanishes at `β_0, …, β_{l−1}`, is 1 at `β_l`, and does not depend on the dimension of the domain. The constants of the code are therefore one triangular table, `Ŵ_l(β_i)` for `l < c_0` and `l < i < d_0`, at most 171 elements of `K` at `t = 22`, and it serves every level because `S_(d_i) ⊂ S_(d_0)` on the same basis. A twiddle `Ŵ_l(F64::from_raw(x_0))` of a run, and a value `Ŵ_l(F64::from_raw(x))` at a queried position, is the XOR of the entries selected by the bits of the argument at and above `l`, the entry at `l` being 1; no table with one twiddle per run is stored. The table is built by the recurrence of §2 for `s_l(β_i)` and one inversion per `l`, under 600 products in `K`. Its lifecycle is the call: it is a value built at the start of `commit`, of `open` and of `verify_opening` and dropped at their end. It is not kept in `ProverState`, there is no global and no lazily initialised cache, and its construction is inside the timed phases. The twiddles of the runs that a worker is processing are values on that worker's stack.

**Memory at `t = 22`.**

The figures of this part are dominant-buffer subtotals: they count the buffers of the first table, which decide the peak, and they are not a complete inventory of the allocations. The second table lists the auxiliaries, which the subtotals leave out and which the benchmark's measured peak includes. Every buffer is an allocation of exactly the stated length, obtained with a fallible reservation of a size computed with checked arithmetic, and the lifetimes are part of the spec: truncating a vector does not return its capacity, so a buffer is released only by dropping its allocation, and no buffer is grown or shrunk by reallocation.

| Dominant buffer (counted) | Capacity | Allocated | Released |
|---|---:|---|---|
| Rows (shared, not owned) | 128 MiB | caller | caller |
| Level-0 codeword; level-0 tree | 256 MiB; 32 MiB | `commit` | after the multiproof of level 0 is written |
| `W0` and `D`, `2^22` elements of `E` each | 96 MiB each | round 1, first pass | `D` after the fold of round 1; `W0` at the fold of round `k_0` |
| Folded message, `2^22` elements of `E` | 96 MiB | fold of round 1 | at the fold of round `k_0` |
| Weight and message of level `i ≥ 1`, `2^(c_{i−1})` elements each | 6 MiB each at level 1 | fold of the last round of level `i − 1` | at the fold of the last round of level `i` |
| Later codewords; later trees | 96, 48, 24, 12 MiB; 16, 8, 4, 2 MiB | commitment of the level | after the multiproof of the level is written |
| Interface scratch of level `i ≥ 1`: the equality table of `z_i`, at level 1 that of `z_0`, the domain buffer and the coefficient output of the transposed transform | `6 + 6 + 12 + 6 = 30` MiB at level 1 | the table of `z_i` with the sample `y_i`; the rest at the start of the level | once merged into `ω` |

| Auxiliary (counted) | Capacity at `t = 22` | Allocated | Released |
|---|---:|---|---|
| Tables of `Φ_α` | 192 KiB | start of `open` | end of the first pass of round 1 |
| Split equality tables `Lo`, `Hi` | 64 KiB | start of `open` | end of the first pass of round 1 |
| `eq_E(z_0, ·)` in `commit` | 6 MiB | step 3 of `commit` | end of `commit` |
| Accumulators of a pass | 48 bytes each: 32 per worker in the commit sample, 2 per worker in a round | with the pass | with the pass |
| Domain constants | under 2 KiB | start of the call | end of the call |
| Scalars: `z_i`, `y_i`, the challenges, the point `r`, the `s_b`, the powers of `α` and of `λ_i`, the final message | under 64 KiB | where used | end of the call |
| The proof under construction, with the opened leaves and digests it retains | at most 337,592 bytes | `open` | returned to the caller |

In `commit` the dominant buffers are the codeword and the tree, 288 MiB, and the equality table of `z_0` brings the total to 294 MiB before it is dropped. At the peak of `open`, in the fold of round 1, the tables of `Φ_α` and the split tables are already released, and what the auxiliaries add to the subtotal is under 512 KiB, most of it the proof if its capacity is reserved at the start; during the first pass of round 1, at 480 MiB, they add 256 KiB more. The allocator's own overhead and the stacks and queues of the thread pool are outside both tables and outside the benchmark's counters.

Three rules make these lifetimes hold without a copy. The weight of round 1 is two vectors and not one interleaved vector, so that the fold `W0[k] ← W0[k] + a·D[k]` leaves `W0` at its exact length and `D` is dropped whole. Folds are in place within a level, at constant capacity. The fold of the last round of a level `i < R − 1` writes its two outputs, `2^(c_i)` elements each, to new allocations and drops its inputs before the codeword of level `i + 1` is allocated. No buffer is moved to reclaim capacity: the output of that fold is written once in either case, and what is charged is the first touch of the new allocations, 12 MiB at level 0, inside the measured units.

The subtotal of the dominant buffers over time at `t = 22`, in MiB: 288 after `commit`; 480 in the first pass of round 1; 576 during its fold; 480 from the end of round 1 to round 4; 492 during the fold of round 5 and 300 after it; 412 once level 1 is committed, and 418 with the equality table of its sample; 124 once the multiproof of level 0 is written and the 288 of level 0 are released, and at most 154 with the scratch at the start of level 1; below 200 from then on. The peak of the subtotal is `256 + 32 + 96 + 96 + 96 = 576` MiB, in the fold of round 1, and 704 MiB with the rows (counted). The benchmark reports the full peak of requested bytes, auxiliaries included, and the rows separately (Performance). An implementation that folds in place and only truncates holds 288 MiB of capacity in the weight and the message when level 1 is committed and reaches `288 + 288 + 112 = 688` MiB there; the rules above are what excludes it.

#### 10. Reuse

**Ported.** The recursion, its level schedule, the per-level parameter search, the additive transform over `F64` and over `F192` with twiddles in `F64`, the Merkle tree and its multiproof, the later round kernels, and the induction of query weights by the transposed transform exist in the public leanVM repository. The port is taken from one recorded revision, `48a904208d682848dac0e18ef8b01ebfc40df9ad`, and from this manifest of files (read at that revision): in `crates/pcs/src`, `whir.rs`, `whir_config.rs`, `whir_ntt_ext.rs`, `whir_induce.rs`, `ntt.rs`, `ntt/additive_ntt_f64.rs` and `merkle.rs`; and, because `merkle.rs` delegates to them, `crates/fiat_shamir/src/merkle.rs` for the digest encoding and the multiproof, and `crates/primitives/src/hash.rs` for the batched hash, the latter only if item 5 of Execution adds an architectural kernel. A file enters the port only through this manifest, which the notices file reproduces with the revision.

The headers of those files are not uniform, and the port preserves what is there and invents nothing. Four of them (`whir.rs`, `whir_config.rs`, `whir_ntt_ext.rs`, `whir_induce.rs`) carry credit lines, copyright lines and the identifier `Apache-2.0 OR MIT`; `ntt.rs` and the two `merkle.rs` carry one credit line and no identifier; `ntt/additive_ntt_f64.rs` and `hash.rs` begin with documentation and carry none. Every file derived from a source file keeps that file's header lines verbatim where it has any, including the holders and upstream projects that they name, then names the source path and the revision, states that the file is modified, and points to the notices file by its path from the repository root. The notices file is `crates/jolt-rv64i-pcs/THIRD_PARTY_NOTICES.md`, created by item 1 of Execution. It carries the manifest above with the revision, the full text of the MIT licence, which is the option taken where a file offers a choice, the copyright notice of the source's root licence file, "Copyright (c) 2026 leanEthereum", and every copyright and credit line that appears in a header of the manifest, whether or not the holder it names appears elsewhere in this spec. The files of the verifier's `whir` module that are derived from the manifest point to the same file. Jolt is itself under MIT and Apache-2.0. This is a statement of what the port carries and not a legal opinion. The port is a port of algorithms onto `jolt_field::binary`, not a dependency: the source's field types, transcript, allocator, raw-pointer buffer handling and thread pool are not ported, and no algorithm brings them with it.

**Not ported, and why.** The source commits a root alone and calls its commitment list binding; the commit sample is new. Its ring switch is 64 by 192, for a claim in `F192`, with a batching map built from six challenges and the Frobenius automorphism; the claim here is in `F128` and a symbol carries 128 bits in `V`, so the 128 by 128 bridge of §3 and its prover kernel are new. The source's lanes are the high variables of the message and its level-0 leaf image is lane-descending; here the lanes are the low variables, which removes the packed copy and the transposition at level 0, keeps the front end's variable order, and lets every round fold adjacent pairs, so the source's dispatch on high lanes and its final rotation of the point have no counterpart. The source's wire format and transcript are its own. The source's parameter calculator runs in floating point at configuration time; here its output is a frozen table.

**In the repository already.** `F64`, `F128` and `F192` with carry-less kernels and accumulators, `F192: ExtField<F64>` with `mul_base`, `F128::mul_x`, the Blake2b transcript with `squeeze_bytes`, the `blake2` crate (which provides BLAKE2s-256 for the verifier), `rayon`, and the two traits with the tests of the transparent scheme, from which the shared contract tests are extracted.

**New.** The E-by-V kernel of §9 in `F192` (five carry-less multiplications into an accumulator, eleven reduced), an accumulator method for an element of `F192` times an element of `F64` (three, no reduction) and multiplication by `y`; the commit sample; the bridge on both sides; the wire format, the transcript schedule and the frozen parameter table.

**Where the code lives.** The verifier's half is a module `whir` of `jolt-rv64i-verifier`, in safe code, hashing with the `blake2` crate; that crate keeps its prohibition of `unsafe`. The prover's algorithms (transform, tree, bridge weights, rounds, induced weights) are a new crate, `crates/jolt-rv64i-pcs`, because they are substantive, specific to this scheme, and shared by no other prover kernel. It depends on the verifier crate and takes from its module `whir` everything that both sides compute (the geometry, the domain constants, the two hashes, the draws, the error type; Execution lists the items). It starts in safe portable code, hashing included, and denies `unsafe` at its root. An architectural kernel is added only when a measurement of the safe code misses the threshold of its phase (Execution, item 5), and then under `src/arch/` behind the feature `arch`, where each use carries a `SAFETY:` comment naming the `cfg` that guarantees the instruction, as `jolt-field` does; the carry-less products stay in `jolt-field`. `jolt-rv64i-prover` implements `BitsCommitmentProver` in `src/commitment/whir.rs` by calling the new crate.

### Alternatives Considered

**64 bits to a symbol of `K`.** The packed table is the row buffer read word by word, `μ = t + 2`, `k_0 = 6`, the bridge has shape 64 by 128 with `t_h ∈ K`, and the level-0 oracle is byte for byte the one of this spec. Its first fold is 6, so the fold row of level 0 carries one more factor of two, and the rule of §5 gives it `m_0 = 151` and the query counts 261, 65, 37, 26, 20 at `t = 22` (computed); the commitment is 1,568 bytes. Its subtotal is 1,993,408,476 multiplications against 1,720,778,716, with 268,435,456 lookups against 134,217,728 and 576 MiB of round-1 vectors against 288 (counted, §7). Not adopted on those counts. Reopened by the measurement of the bridge phase stated in §7.

**An opening field of `2^256` elements that contains `F128`.** The smallest field that contains `H` and has at least `2^192` elements is `E' = H[v]/(v^2 + v + x^121)`. The polynomial is irreducible because the absolute trace of `x^121` in `H` is 1; among the monomials `x^i`, `i < 128`, exactly `x^121` and `x^127` have trace 1 (computed, by polynomial arithmetic modulo `x^128 + x^7 + x^2 + x + 1` outside the repository). `E'` does not exist in the repository. Its product has the coefficients `a_0·b_0 + x^121·a_1·b_1` and `a_0·b_1 + a_1·b_0 + a_1·b_1`: three products in `H` and the linear map of multiplication by `x^121`, which is not free and needs a kernel of its own. At 18 multiplications for a product, 12 into an accumulator, and 12 and 8 for an element of `E'` times an element of `H`, round 1 costs `12 + 16 + 12 + 18 = 58` per pair, or 52 with the composed lookup, and a later round `24 + 36 = 60`. The subtotal is 2,079,064,004 without the composed lookup, 1.73% above the 2,043,740,124 of the 64-bit packing on the same basis, before the constant map and the reduction of the extension are priced; figures of 56 per pair assume a product of 16, which needs a kernel that is specified and measured. Containment does not decide the packing. An embedding of `K` into the raw representation of `H` is a linear map over `F_2` with 128 output coordinates and 64 input coordinates; a representation of `H` over its subfield of `2^64` elements, with the cost of converting it to the raw representation, would have to be stated separately, and the design counted in §7 packs 128 bits to a symbol of `H` instead. It keeps a bridge of shape 128 by 128 (§7). It gains 64 bits on every algebraic term and none on the query terms, which are the weakest, and none on the front end's. Not adopted: 333,119,464 more multiplications than this spec, 983,040 more hash compressions (later leaves of 512 bytes, 8 compressions for 6 on 491,520 leaves) and no field code. Reopened by either of two results: the check of the theorem in its primary source (Open, 1) gives a constant that the search cannot absorb over `2^192` elements; or the front end moves its challenges to a field that contains `H` and a kernel for `E'`, constant map included, is measured at 16 multiplications or fewer per product.

**A commitment that is a root alone.** This is what the source does. It saves the commit sample: 45,088,768 carry-less multiplications, an estimated 13.75 ms on one thread, and 768 bytes. It leaves the commitment binding a list of up to `L_0 = 187` tables, and the bound of every round of the front end, whose errors are of the form `δ/2^128`, is then paid once per list member: 7.54 bits lost on each, with no way to recover them in the scheme. Rejected. It would be reopened only by a front end whose challenge field has that margin.

**One sample of the whole table instead of one per lane.** A single value `p~(z_0)` at `z_0 ∈ E^μ` makes the commitment 56 bytes instead of 800. Its weight has `2^23` entries at level 0: merged into the bridge weight it costs `12·2^23 = 100,663,296` carry-less multiplications, and carried as a separate term through the five lane rounds about `6·2^22 + 12·(2^22 − 2^18) = 72,351,744` (counted). The per-lane sample costs `12·2^18 = 3,145,728` in the opening. Rejected on that count; 744 bytes are 0.23% of the expected opening.

**Grinding.** §8. At 17 bits: 56,640 bytes fewer at most, the position terms at `2^-111` as errors. Rejected because the target is an error bound. Reopened if the owner accepts a work factor in place of an error for those rounds.

**Rate 1/4 at level 0.** Halves the queries of level 0 and doubles the level-0 codeword and tree. On the source, which has 260 queries at level 0 and 130 at that rate, measured in one sweep on the loaded host at twelve threads: 219,192 bytes against 332,112, and 442.5 ms against 300.0 ms for commit and open together, with 1,611 MiB against 1,326 MiB of resident memory at the peak. Rejected while the prover's time is the binding budget. Reopened if proof size becomes the objective.

**A level-0 leaf of 16 or 256 words instead of 64.** That is `k_0` of 3 or 7 here, and an initial fold of 4 or 8 in the source, whose symbols are single words. Measured on the source in the same sweep: 385.7 ms and 309,056 bytes at 4, 349.1 ms and 676,912 bytes at 8, against 300.0 ms and 332,112 bytes at 6. Rejected on both columns at 8 and on time at 4; the measurement is of the source and is repeated on this scheme only if the bridge phase is measured above its model.

**Ligerito.** The recursion of this spec is of the same family: interleaved codes, partial sumchecks, one new commitment per level. The alternative is its analysis within the unique-decoding radius, with no list and so no commit sample, no samples at later levels and no proximity-gap term. Its position term is `(1 − (1 − ϱ)/2)^Q`: 309 queries at rate 1/2 for 260 and 141 at rate 1/16 for 65, which is `49·1,120 + 76·960 = 127,840` more bytes at most on the first two levels alone. Rejected on proof size. Reopened if the proximity-gap constant of assumption 1 is found not to apply as transcribed, since it is the one assumption that the unique-decoding analysis does not need.

**BaseFold-style folding of the codeword.** Fold the level-0 codeword itself, with one tree per fold of the same schedule and every oracle at rate 1/2. The prover saves every later encoding (43,515,904 butterflies, 65.4 ms measured on the source on one thread) and most of the later hashing. Every query then opens a path in every oracle at the level-0 query count: `260·(1,120 + 864 + 736 + 608 + 480) = 990,080` bytes at most against 428,512 on the same independent-path bound, since the oracles have `2^19`, `2^15`, `2^11`, `2^7` and `2^3` leaves. Rejected on proof size, a factor of 2.31. Reopened if the proof is not transmitted or its size does not matter and the 154 ms of later encodings and trees do.

**Other alphabets.** Symbols that are full elements of `F192` carrying 64 or 128 bits each, to avoid mixed arithmetic, would hash 24 bytes per symbol where this spec hashes 16, and would multiply the 4,194,304 leaf compressions of the first tree by 3 or by 1.5. Symbols in `H` with `E = H` would make every algebraic term a term over `2^128` elements, where the fold term of level 0 has no room. Rejected.

**A weight that is never materialised.** Build the weight twice from the split tables, once for the first round message and once for the fold, and keep neither `W0` nor `D`: the fold of round 1 then owns `288 + 96 + 96 = 480` MiB, 96 below the peak of §9, and the first round costs a second pass of `6·2^22` carry-less multiplications and 134,217,728 lookups, an estimated 88 ms on one thread at the unit prices of Performance. Rejected on time. Reopened if the prover's memory budget is set below the peak of §9.

## Documentation

The module documentation of `whir` in `jolt-rv64i-verifier` states the protocol of §2 to §6 in the order of the code and links this spec. The contract comment of `commitment.rs` is unchanged. The sentence of `BitsCommitmentProver` that neither phase copies a trace-sized buffer gains the precision of invariant 9: the codeword is a new buffer, the rows are shared. `specs/rv64i-binary-protocol.md` gains, in its performance section, the proof size with this scheme in place of the size "plus the scheme's bytes". No book page changes.

## Execution

One integrator owns every manifest, every crate root and every module root (`Cargo.toml` at the workspace and in the crates, `crates/jolt-rv64i-pcs/src/lib.rs`, `crates/jolt-rv64i-verifier/src/lib.rs`, `whir/mod.rs`, `crates/jolt-rv64i-prover/src/commitment/mod.rs`), and with them every `mod` line, every `pub` on one, every re-export, every feature and every `[[bench]]` entry. Item 1 is the integrator's and comes first; nothing in the verifier's module `whir` or in the new crate compiles before it, and item 3, which is in `jolt-field`, does not wait for it. Workers of items 2 to 6 edit only the implementation files named in their item, which are disjoint, so those five items run in parallel, with one ordering among them: the fused first-round kernel of item 6 calls the accumulator methods of item 3 and waits for it, while the verifier half of item 6 and its reference first round do not. Item 7 needs items 1 to 6, item 8 needs item 7, item 9 needs item 8, and item 10 is separate and later. The integrator merges the items and runs the Cargo commands one at a time, never two concurrently.

A helper that another item or the other crate calls is public, and it lands complete, with its tests, in the item that owns its file; the integrator makes its module `pub` in the same merge. The dependency runs one way: `jolt-rv64i-pcs` depends on `jolt-rv64i-verifier` and takes the geometry, the domain constants, the leaf and node hashes, the challenge draws and the error type from the module `whir`, so that no formula of §2 to §6 is written twice. No item adds a public function whose body waits for a later item, and an empty module file is scaffolding and declares nothing.

| Item | Public from `jolt_rv64i_verifier::whir` | First consumer outside its file |
|---|---|---|
| 1 | `error::{WhirError, WhirPart}`; `error::checked_product(part, &[usize]) -> Result<usize, WhirError>`; `error::try_vec::<T>(part, len) -> Result<Vec<T>, WhirError>`, an empty vector with reserved capacity `len` | `params` (item 2) |
| 2 | `params::{Schedule, Level, Queries}`: `Schedule::new(BitsGeometry) -> Result<Schedule, WhirError>`; `mu()`, `res()`, `levels() -> &[Level]`, `offset(i)`; `Level { k, c, d, queries, leaf_bytes }` with `lanes()`; `Queries::{All, Count(u32)}`; under `test-utils`, `Schedule::from_levels` | items 7 and 8; items 4 to 6 take `c`, `d`, `μ` and a leaf length as integers and do not depend on it |
| 4 | `code::DomainTable`: `new(c_0, d_0) -> Result<DomainTable, WhirError>`, `w_hat(l, x: u32) -> F64`, `query_weight(c, x, q: &[F192]) -> F192`; under `test-utils`, `code::encode_by_definition` | `src/ntt.rs` of the new crate, in the same item |
| 5 | `merkle::{Digest, hash_leaf, hash_node, verify_multiproof}` | `src/merkle.rs` of the new crate, in the same item |
| 6 | `bridge::{slice_claims, tau, BridgeWeight}`: `s_b` and `t_h` from the request, `τ`, and the recurrence for `G` as a value that absorbs one `q_l` at a time and ends in `w~(q)` | the bridge tests of the new crate, in the same item; then item 7 |
| 7 | `challenge::{draw_element, draw_point, draw_positions}` and the six label constants; `wire::{WhirCommitment, WhirOpeningProof}` with public fields; `WhirBits`, `WhirVerifierState`; under `test-utils`, the entry points that take a `Schedule` | `src/commit.rs` and `src/open.rs` (item 8) |

The signatures above fix names and the direction of data. Argument types that the table leaves open (slices against arrays, `u32` against `usize` for a position) are the owner's to choose and are fixed by the item that lands them.

1. **Bootstrap.** `crates/jolt-rv64i-pcs/Cargo.toml` with its entry in the workspace members and the workspace dependencies, its lint table and its features (none by default; `arch` reserved and empty until item 5 decides); `crates/jolt-rv64i-pcs/src/lib.rs` with the crate attributes of "Hygiene" and private `mod` lines for `ntt`, `merkle`, `bridge`, `commit`, `open`, `rounds`, `induce`, each an empty file; the line `pub mod whir;` in the verifier's root and `whir/mod.rs` with `pub mod error;` and private `mod` lines for `params`, `code`, `merkle`, `bridge`, `challenge`, `wire`, `verify`, each an empty file; the dependencies of the new crate on `jolt-field`, `jolt-rv64i-verifier`, `jolt-transcript`, `blake2` and `rayon`, its development dependency on the verifier with `test-utils`, and the dependency of `jolt-rv64i-prover` on the new crate; `whir/error.rs` complete, with `WhirError`, `WhirPart`, their `Display` and `Error` implementations and the two size helpers of the table; and `crates/jolt-rv64i-pcs/THIRD_PARTY_NOTICES.md` with the content of §10. Criteria: the workspace builds, "Hygiene" holds, and the two criteria on `LengthOverflow` and `Allocation` pass. Depends on nothing.
2. **Parameters.** `whir/params.rs` of the verifier: the schedule of §5 as a function of `t`, the table of query counts as constants for `1 ≤ t ≤ 32`. The public surface is the geometry that every later item reads: a schedule value built from `t` that fails with `UnsupportedGeometry` outside `1 ≤ t ≤ 32`, its levels with `k_i`, `c_i`, `d_i`, the lane count, the leaf length in bytes and the query count as a typed value with the variants "all" and a number, and `R`, `res`, `μ` and the offsets `o_i`. The test runs the rule of §5 in exact rational arithmetic, with the admissibility conditions as written there, and compares its output with the frozen integers; the rule lives in the test and the verifier holds the integers. If an entry of §5 is not confirmed, the table in the code and in this spec is corrected in the same change. `Schedule::from_levels`, under `test-utils`, builds a schedule from explicit `(k, c, d)` and query counts and checks only that the levels chain (`c_i = c_{i−1} − k_i`, `k_0 + c_0 = μ`); it claims no soundness and exists for the completeness criterion on a last fold of 3. Criteria: "Parameters". Depends on item 1; parallel with 3, 4, 5 and 6.
3. **Field.** `crates/jolt-field/src/binary/`: the E-by-V kernel of §9 as `F192::mul_base_pair(self, [F64; 2]) -> F192` (eleven products) and `F192Accumulator::fmadd_base_pair(&mut self, F192, [F64; 2])` (five), `F192Accumulator::fmadd_base(&mut self, F192, F64)` (three) and `F192::mul_y(self) -> F192`. The names say what the operands are in `jolt_field`, which has no type for `V`; the argument `[b_0, b_1]` is the element `b_0 + y·b_1`. The portable and the aarch64 kernels both implement the five-product formula as written, and neither relies on the compiler to remove products from the six-product kernel. Callers: the kernel is called by the commit sample, by round 1 and by the verifier's lane combination at level 0; `fmadd_base` by the direct check of the last level; `mul_y` only by the comparison construction of the benchmark. `crates/jolt-field/benches/binary_kernels.rs`: groups that time one carry-less product chain in `F64`, each of those products, `F192` times `F192` reduced and into the accumulator, and the E-by-V kernel against the composition `e·b_0 + (y·e)·b_1`, each reduced and into the accumulator. Criteria: "Field". Measurement obligation: the unit row `c`, the products of §7 and the comparison of the kernel with the composition, on a quiet host; if the composition is faster on the benchmark host, the counts of §7 and §9, the model and the thresholds are recomputed with it in the same change, under the thresholds rule. Depends on no other item, and item 1 does not touch `jolt-field`; parallel with 1, 2, 4 and 5 and with the unfused part of 6.
4. **Code and transform.** `whir/code.rs` of the verifier: the table of domain constants of §9, `Ŵ_l(x)`, `W~_x(q)`, and the encoder by its definition, the last under `test-utils`. `src/ntt.rs` of the new crate and `benches/ntt.rs`: the position-major transform of level 0 over `F64` on the two coordinates of a symbol, the transform over `F192` with twiddles in `F64`, and its transpose with the block sums, with the index layout of §9 for its input and output; ported under the rule of §10. Criteria: "Code". Measurement obligation: `nb0` and `nb1`, each timed with the allocation of its output and the construction of the constants inside the measurement. Depends on item 1; parallel with 2, 3, 5 and 6.
5. **Merkle.** `whir/merkle.rs` of the verifier: the digest type, leaf and node hashing with the `blake2` crate, the multiproof verifier with its traversal of §6 and its check of `n_i` and `g_i`. `src/merkle.rs` of the new crate and `benches/merkle.rs`: the tree builder and the multiproof writer, in safe code, calling the verifier's two hash functions, parallel over leaves and nodes; ported under the rule of §10. Criteria: "Merkle". Measurement obligation: `hb0` and `hb1` for the safe hash, on one thread at the tree sizes of `t = 22`. The threshold of a tree phase is 1.25 times its row of the model, 171.05 ms for the level-0 tree and 110.53 ms for the four later trees (estimated, from 136.839 ms and 88.424 ms). These two figures decide one thing, whether a batched kernel is written: it is added under `src/arch/` behind the feature `arch` only if the safe hash misses one of them, and is then measured against the same figure. They are not acceptance thresholds, which exist only for `commit` and `open` whole. Depends on item 1; parallel with 2, 3, 4 and 6.
6. **Bridge.** `whir/bridge.rs` of the verifier: `s_b`, `t_h`, `τ`, the recurrence for `G`. `src/bridge.rs` of the new crate and `benches/bridge.rs`: the split tables, the tables of `Φ_α` and their composition with the first coordinate, the vectors `W0` and `D`, a reference first round that builds the weight entry by entry from the definition, and the fused first round on the accumulator methods of item 3. New. Criteria: "Bridge", with the fused round tested against the reference. Measurement obligation: `Lw`, the table set-up and the first round alone (tables, `W0`, `D`, `u_0`, `u_2`) at `μ = 23` on one thread. The whole bridge phase of §7 includes the later rounds, which item 8 owns, and is measured by item 9. Depends on item 1, and on item 3 for the fused round; parallel with 2, 4 and 5.
7. **Verifier.** `whir/{challenge.rs, wire.rs, verify.rs}`: the label constants and the draws of §6, the wire types and their `BitsWire` implementations with the two-pass reader, the type `WhirBits` with `verify_commit` and `verify_opening`, the shape checks of §6 on typed values, and the entry points that take a `Schedule`. Criteria: "Wire", on typed values of the right shape filled with random elements, which need no valid proof; the literal positions of "Transcript"; and the criteria on `WhirError` that need no prover (geometry, shapes, and the counts of a level that opens every position). Depends on items 1 to 6.
8. **Prover.** `src/{commit.rs, open.rs, rounds.rs, induce.rs}` of the new crate and `crates/jolt-rv64i-prover/src/commitment/whir.rs`, which implements `BitsCommitmentProver` for `WhirBits` with `ProverState` holding the shared rows, the level-0 codeword, the tree, `z_0` and `y_0`; the recipe of §9 for `k_0 = 0`; the release points of §9; the shared contract tests extracted from the tests of the transparent scheme. Criteria: "Completeness", "Rejections", "Determinism", "Front end", the rest of "Transcript" and of the criteria on `WhirError`. The source implementation is not an oracle for any of them. Depends on item 7.
9. **Benchmarks.** `crates/jolt-rv64i-prover/benches/bits_whir.rs` through the runner of `benches/support/`, with the phases of Performance as named phases and the allocation interval of Performance. Measurement obligations: every row of the phase table on a quiet host at one and twelve threads, with all set-up inside the timed phases; the bridge phase of §7 and the decision that it carries; the peak of requested bytes and the values at the release points of §9, against the requirement of 604.8 MiB; the proof size distribution over 1,000 transcripts; and the replacement of the unit rows under the thresholds rule. Depends on item 8.
10. **Later, separate: the shared opening traits.** The scheme trait of `crates/jolt-openings/src/schemes.rs` has one field that serves as the field of the source, of the claim and of the transcript's challenges, and a `commit` that takes no transcript. That `commit` returns an `OpeningHint`, which `open` and the batch prover consume, so retained state exists; what the trait lacks is a commit phase of several messages coupled to the transcript. Its `open`, `verify` and batch methods take a transcript, and additive homomorphism is a separate trait that the base trait does not require. A generalisation adds a typed packed source, a claim field distinct from the opening field and a commit lifecycle coupled to the transcript; it keeps the existing contract of the hint and the separation of the homomorphic trait, and it lands in one change together with `WhirBits` as a working caller, after item 8. Nothing in items 1 to 9 depends on it, and no wrapper or state container is added for it before a caller needs one.

Risks carried by the items. The round kernels of the port run on adjacent pairs from the first fold, so the source's dispatch on high lanes and its final rotation of the point are removed and not adapted (items 4 and 8); the encoder and bit-evaluation definitions of item 4 are the oracle for that change. The first-round kernel of item 6 is new, is the largest estimated phase, and is 130.42 of the 176.48 ms of the phase that decides §7. The table of §5 is certified in the repository only by the test of item 2, and its two-bit reserve stands in for a check of the theorem that has not been made (Open, 1).

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

1. **The proximity-gap theorem in its primary source.** The formula for `a` in §8 agrees between the annex and the calculator of the ported implementation and has not been checked against the paper: its version, whether the constant is that of the list form, and its hypotheses on the field and the domain. The fold rows of the ledger, and through the rule of §5 every `m` and every query count, depend on it. The rule reserves two bits on every algebraic row, so a constant `a` larger by a factor of up to four leaves every fold row at most `2^-128` with the table unchanged. The reserve is a margin on a number: it does not establish that the theorem applies, and this check is a condition of any security claim made outside the experiment.
2. **Knowledge extraction and the written proofs.** The scheme claims round-by-round soundness of an existential relation. An extractor, and full proofs of Lemma 2 and of the level interface with the commit-sample claim in the pool, are owed before the ledger is cited outside this experiment. Lemmas 1 and 2 are given in outline. An exhaustive instance of Lemma 1, a subfield of 4 elements in a field of 16 with dimension 2 on a domain of 4 points, finds no candidate outside the subfield among the 1,072 pairs of a word and a polynomial that agree on two points or more (computed); that corroborates the lemma and does not prove it.
3. **The composed protocol and the compilation.** No theorem is stated for the front end composed with this scheme, and none for the compiled protocol. Owed for the first: the convention for a vector of challenges (one transition per vector or one per coordinate, the second of which needs an invariant that the protocol spec does not state) and the sum over rounds; the guarantee of §8 for the front end is read from individual round bounds. Owed for the second: a compilation theorem for the Fiat-Shamir transformation of a round-by-round sound protocol that applies to this transcript, with the model of the transcript's hash, the convention for counting its queries `q_t`, and the treatment of challenges in `E` that are assembled by `squeeze_bytes` from draws in `H`. §8 states no compiled bound until that theorem is selected.
4. **The table of §5.** It is the output of the rule of §5, run outside the repository in exact rational arithmetic. With the target `2^-128` on every row and the geometry of the source, an initial fold of 6 over `2^24` words, the same rule gives the query counts that the source runs with at `t = 22`, 260, 65, 37, 26, 20 (computed). No entry is certified in the repository until the parameters test of item 2 of Execution.
5. **The hash.** BLAKE2s-256 is kept from the source. Its collision term is stated at the declared budget `q_h ≤ 2^64` of §8, which is an assumption about work and not a property of the code. Whether BLAKE3 or SHA-256 with hardware support is cheaper at 512-byte and 384-byte leaves on the benchmark host is not measured; a change is a change of the wire format.
6. **Unit prices and the bridge.** `c` and `Lw` are estimates, and the measured rows were taken on a loaded host, on the source and not on this scheme. The butterflies of the source and of the port have the same arithmetic, nine carry-less products with the reductions at the later levels, so the open point for `nb0` and `nb1` is the transfer of a measurement to different code and a quiet host, not a count. Whether the five-product E-by-V kernel is faster than the six-product composition is not measured. No phase of the bridge has been measured, nor the setup and cache behaviour of the composed tables; the choice of §7 stands on counts until item 6 of Execution measures the first round and item 9 the whole phase.
7. **Scaling to twelve threads.** The one measurement available gives 4.6 for commit and 5.1 for open, on the source, on a loaded host with eight performance and four efficiency cores. Whether the cause is the load, memory bandwidth in the transform and the hashing, or the core mix is not known, and the twelve-thread requirement of Performance is provisional until item 9.
8. **Small tables.** For `t ≤ 10` every position of level 0 is opened. The proof is then larger than the table for the smallest `t`, which is accepted for sizes used only in tests; whether the front end ever runs a production instance below `t = 11` is the owner's to say.
