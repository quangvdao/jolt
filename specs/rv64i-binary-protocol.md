# Spec: RV64I Protocol over the Binary Field

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-10                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

An experiment in RV64I hash-based Jolt over binary fields proves plain RV64I over `F128` with one committed table, `Bits`, of 256 bits per cycle. `specs/rv64i-binary-arithmetisation.md` fixes the columns, the decode table and the constraint rows. This spec fixes everything between that crate and the commitment scheme at the level of the wire: the identifiers, the points, eighteen sum-check relations in eight batches (stages 1, 2, 3, 4, 5, 6a, 6b), the transcript order and its bytes, the proof and its bytes, the statement checks, and the interface a commitment scheme implements. Every claim on `Bits` ends at one cycle point `r_6`; the prover then sends the 256 column values `C[y] = Bits(y, r_6)`, the verifier draws `rho`, and the commitment scheme opens `Bits` once, at `(rho, r_6)`. The spec adds two crates, `jolt-rv64i-verifier` and `jolt-rv64i-prover`, written in the existing layers (symbolic relation, concrete relation, kernel, stage) on the seams of `specs/binary-protocol-family-seams.md`. The prover half of this spec is the reference tier only.

## Intent

### Goal

A verifier that accepts exactly the executions of the machine of `specs/rv64i-binary-arithmetisation.md` against a public statement, and a reference prover for it, such that every value on the wire, its position in the transcript and the equation that consumes it are stated here once.

The entry points:

```rust
// jolt-rv64i-verifier
pub fn verify<S: BitsCommitmentScheme>(
    preprocessing: &VerifierPreprocessing<S>,
    statement: &Statement,
    proof: &Rv64iProof<S>,
) -> Result<(), Rv64iVerifierError>;

// jolt-rv64i-prover
pub fn prove<S: BitsCommitmentProver>(
    preprocessing: &ProverPreprocessing<S>,
    statement: &Statement,
    witness: &Rv64iWitness,
    backend: &Rv64iBackend<F128>,
) -> Result<Rv64iProof<S>, Rv64iProverError>;
```

`BitsCommitmentScheme` and `BitsCommitmentProver` are the family's own traits (§11).

### Invariants

1. **One definition per fact.** The schedule (Design §5), each relation's expressions (§8) and the claim flow (the `from =` attributes of §8) are stated once in this document and once in code: the schedule is the member list of the `#[derive(SumcheckBatch)]` structs, an expression is the body of one `SymbolicSumcheck` method, and the claim flow is the `from =` attributes of the `InputClaims` structs. Round counts, value counts, the proof size, the table of §9 and the test literals are computed from those.
2. **Claims.** Every wire value before `C` is an input cell of exactly one later relation, the relation that reduces it. It may also be the source of alias cells of its own batch, which are the same value read by the terminal checks of other members. The 256 values of `C` have no later relation: the terminal check of batch 6b reads them through its three members, and the opening binds them. No relation consumes a claim of its own batch or of a later one. A cell that is not on the wire is either an alias of a wire cell of the same batch or a projection of `C` (§7), and nothing else.
3. **One opening.** The commitment scheme is asked for exactly one evaluation of `Bits`, at `(rho, r_6)`. No stage follows the last cycle sum-check.
4. **Transcript.** Every challenge is drawn after the absorption of every prover value it has to bind. The front end absorbs and draws in the order of §6, with the bytes of §13, and nothing else; the commitment scheme absorbs and draws only inside its two calls (§11). `C` is absorbed whole before `rho` is drawn.
5. **Characteristic 2.** Every coefficient of every expression is 1. No expression contains an integer constant above 1, a power-of-two scale, a halving or `Expr * i128`. Field values other than 0 and 1 enter an expression only as `Derived` or `Challenge` leaves. Subtraction is addition, and this document writes `+`.
6. **Batching coefficients.** Every random coefficient that combines claims is an independent transcript draw; none is a power of another. One combination has fixed coefficients: the five routers are summed with coefficient 1 inside `RouterShort`, because the claim that member proves is that sum (§8.4).
7. **Verifier half.** `jolt-rv64i-verifier` contains no prover code and does not depend on `jolt-kernels`, `jolt-prover` or `jolt-rv64i-prover`. It carries `#![forbid(unsafe_code)]` and the lints of `specs/verifier-closure-lints.md`. No verifier path panics, indexes unchecked, exponentiates an unchecked dimension or allocates in proportion to an unvalidated length of the proof.
8. **Public data.** The verifier computes every `Derived` term from the checked statement, the preprocessing, the two header values of the proof that the preamble absorbs (`log_K_ram`, which fixes the layout, and `final_pc`), the challenges and the points. No `Derived` term reads a round message, a wire value or an opening proof.
9. **Identifiers.** The conversions of §4 into `ExternalId` are injective on every identifier the family constructs, and their inverses reject every other family and every index outside the decode domain of §13.
10. **Bits.** The commitment scheme binds a table of bits, and that is a guarantee of the scheme's verifier (§11): no relation of this family proves that a committed value is 0 or 1. The protocol never treats `Bits` as a table of field elements, and no production prover type holds a field element per committed bit.
11. **Reference tier.** Every kernel of this spec is a dense test oracle, labelled as such in its module documentation, with no performance requirement.
12. **Canonical bytes.** Every transcript input of the front end and every byte of a proof is a function of the statement, the preprocessing and the proof's values, as §13 states it. A proof has one accepted byte string: the decoder rejects any other encoding of the same values and any trailing byte.

No `jolt-eval` invariant changes: nothing here is reachable from the existing prover.

### Non-Goals

Optimised kernels and their cost model (`specs/rv64i-binary-prover-kernels.md`). A production commitment scheme: §11 fixes the interface that one implements, and the only scheme of this spec is the stand-in of the tests. A reusable form of that interface in `jolt-openings`: no crate outside the two new ones changes. Zero knowledge and BlindFold. A univariate skip in stage 1: its rounds are Boolean, and a skip would change the wire. The M extension, a committed bytecode and a committed initial RAM. Private input: both advice vectors of the statement are empty, and the advice regions of the memory layout are ordinary RAM that starts at zero. The adapter from a tracer to `CycleFacts`. Lean generation for this family.

## Evaluation

### Acceptance Criteria

Sizes are written `t = log T`, `b = log K_bytecode`, `a = log K_ram`.

- [ ] **Identifiers.** For every `OpeningId`, `DerivedId` and `ChallengeId` that any relation constructs at layouts `(b, a)` in `{(4, 5), (8, 10), (20, 20), (20, 27)}`: `TryFrom(From(id)) == Ok(id)`; the composite indices are pairwise distinct within a kind; a composite with another `family`, a `Jolt` composite, an index with a reserved bit set, an unknown relation, an unknown tag and, for every variant that carries a payload, the first payload outside its decode domain of §13 each return `Err` with the composite unchanged. The indices of `(Inc, RegistersValEvaluation)` and `(Column(229), BitsReduction)` equal the literals of §4 written in the test.
- [ ] **Points.** `eq`, `lt` and `next` of §3 equal direct sums over the Boolean cube for 1 to 6 variables at seeded random `F128` points, and equal the indicator of `x = y`, `x < y` and `y = x + 1` on all Boolean pairs for 1 to 4 variables. `next(x, y)` summed over Boolean `y` is `1 + eq(x, 1…1)`, and `next(x, 0) = 0`. `chunk` of §3 equals the multilinear extension of the full one-hot digit table for chunks of 1 to 4 bits.
- [ ] **Claim flow.** A test walks the eight batches in order with the opening ids of `canonical_order()` of every `InputClaims` and `OutputClaims` struct at two layouts: each consumed id was produced by an earlier batch; each wire id is consumed exactly once or is a `Column`; each aliased id has a source that is a wire id of the same batch; each projected id is a chunk cell of batch 6b; the table the test prints equals §9.
- [ ] **Schedule.** For each batch at `(t, b, a) = (22, 20, 20)` and `(22, 20, 23)`, the generated `max_num_vars`, `max_degree`, member offsets and number of wire values equal the literals of §5. The round elements number 640 and 671, the wire values 299 at both. A transcript that records its squeezes sees, between the value of batch 6a and the input claims of batch 6b, the six scalars of `BitsReduction` and then the two of `RamRaProduct`.
- [ ] **Placement.** For batch 1 and batch 4 at `t = 3`, `a = 6`, with dense random members of the stated degrees, the batch's final claim equals `Σ_i c_i · λ_i · e_i` with `λ_i` the product of `1 + z_j` over the rounds outside the member's window of §5, each `e_i` computed by `Polynomial::evaluate` on the member's own tables at its window of the challenges.
- [ ] **Expression against definition.** For each relation, at seeded random challenges and for a seeded honest witness at `t = 5`: the input claim computed by `input_claim` equals the direct sum over the member's cube of its summand, where the summand is evaluated from `Rv64iWitness` and the closed forms of §3 by code in the test that does not call `output_expression`; and `expected_output` at the member's point equals that summand's extension evaluated by `Polynomial::evaluate` on tables built in the test.
- [ ] **Routers.** For 10,000 seeded cycles at two layouts with all 58 variants, `Σ_r Σ_{s,h} Route_r[c, s, h] · Source_r[s] · Select_r[h]` equals column `c` of `WitnessRow::compute` for every virtual column `c`, with 1 added at column 16. The source expansion of `RdWriteValue` agrees with `Words` on every variant. The digest of the generated tensors and their number of nonzero entries equal constants in the test. The set of `Bits` columns with a nonzero entry in `RowSystem::to_matrices()` is `[64, Layout::keys_differ()]` at four layouts.
- [ ] **Terminal check.** For a seeded bit table at `t = 5` and seeded points, the three member outputs of batch 6b computed from `C` by the formula of §10 equal the three summands' extensions at `r_6` computed from the table directly, and `Σ_y eq(rho, y) · C[y]` equals `Polynomial::evaluate` of the table at `(rho, r_6)`.
- [ ] **End to end.** Five programs are written with the encoder of the arithmetisation's tests and executed by its interpreter: a counting loop; a byte copy through every load and store width; calls and returns through JAL and JALR; a shift-and-XOR generator over the six shift kinds; a ladder over every branch and every set-less-than variant. Each ends with a store of 1 to the termination word and an instruction that jumps to itself, which repeats to `2^t` cycles: `jal x0, 0` in the first three, `ECALL` in the fourth and `EBREAK` in the fifth, which the machine defines as self-loops and which no other program executes. For each program, at `t` in `{6, 8, 10}` and the smallest `a` and `b` it admits, `prove` then `verify` succeeds with `TransparentBits`, and the prover's and verifier's transcript states agree after each batch. Together the five traces execute all 58 variants.
- [ ] **Tamper, per field.** For one program at `t = 8`, changing each of the following alone makes `verify` return an error: `log_K_ram`; `final_pc` to another valid PC, to an invalid PC and to a PC that is not a multiple of 4; the commitment; one coefficient of one round of each batch; the number of rounds of each batch; the opening proof; each statement field (entry PC, each of the 20 fields of the memory layout, an input byte, an output byte, the panic flag, `log_T`); the bytecode digest.
- [ ] **Tamper, per claim.** For the same proof, adding 1 to each of the 299 wire values alone makes `verify` return an error. A `C` of 255 or 257 values, a `rounds` field that is not a compressed clear proof, and a round message with no stored coefficient, with more than the batch's degree, or with two or more of which the last is zero, are rejected by `CheckedInputs::new` before any sum-check runs.
- [ ] **Order of `C` and `rho`.** A prover that draws `rho` before absorbing `C`, and then sends a `C` that differs from the honest one in two columns that no member of batch 6b weights, by differences that cancel at its own `rho`, passes the opening at that `rho`; the test asserts the two zero weights and the two nonzero differences. `verify`, which absorbs the `C` it received and then draws `rho`, rejects that proof, and the opening value at the verifier's `rho` differs from the claimed one.
- [ ] **Statement.** `CheckedInputs::new` rejects each of the following with its own error variant, before any table whose size the statement or the proof chooses is built. Dimensions: `t = 0`; `t` above `LOG_T_MAX`; `a` below 5; `LowestAddress + 8·K_ram` above `2^64`; `K_ram` above `compute_max_ram_k` of the memory layout. Memory layout, three instances that `MemoryLayout::try_new` does not reproduce: the layout with an empty I/O mask (with `R = RAM_START_ADDRESS`: both advice starts and `panic` at `R − 16`, `termination = output_start = R − 8`, `input_start = R`, empty inputs and outputs, `panic = false`, the one-instruction program `jal x0, 0` at `R` with image word `(2, 0x6f)`, `a = 5`, and a trace that never stores); a layout whose output region overlaps its input region; a layout with a region start that is not a multiple of 8. Segments: inputs longer than `max_input_size`; outputs longer than `max_output_size`; a nonempty advice vector. Program: a bytecode whose `LowestAddress` differs from the memory layout's; an I/O range that ends above `K_ram`; an image word at or above `K_ram`; an image word inside the I/O range; a `final_pc` that is not the PC of a valid row. With an honest trace otherwise, three inconsistent statements are rejected on both sides: a wrong output byte, a trace that never stores 1 to the termination word with `panic = false`, and an entry PC that is not the PC of cycle 0. `prove` returns the round-check error of the batch that owns the equation (batch 4 for the first two, batch 6a for the third), since an honest kernel cannot produce a round whose sum is the false input claim; `verify` rejects the proof of the true statement presented with the altered one.
- [ ] **Preprocessing.** `VerifierPreprocessing::new` rejects an image with a repeated word index, with indices out of increasing order and with a zero value. The digest of the one-instruction program above equals a literal in the test.
- [ ] **Preamble vector.** For the statement and preprocessing of the counting loop at `t = 6`, written out in the test, a transcript that records the argument of every `append_bytes` call sees 36 calls for the front-end preamble; their concatenation equals a hexadecimal literal in the test, and `Transcript::state()` after the commit phase of `TransparentBits` equals a 32-byte literal. PR 0 fixes both literals; they change only together with the label `jolt-rv64i-binary-v0`.
- [ ] **Proof vector.** For the same program and `TransparentBits`, the digest (the hash of §13) of `Rv64iProof::to_bytes` equals a literal that PR 6 fixes; `from_bytes(to_bytes(p))` equals `p`; every strict prefix, the encoding followed by one byte, a version byte other than 0 and a length prefix larger than the remaining input are rejected, the last without allocating that length.
- [ ] **Commitment lifecycle.** A test-local scheme that wraps `TransparentBits` proves and verifies the counting loop. Its commit phase absorbs the digest, draws 24 bytes by `squeeze_bytes` and absorbs a second message bound to them; its opening derives the 64 partial values `s_i` of §11 from `C` and checks each against the table. A verifier that runs that commit phase after the first `tau` draw rejects.
- [ ] **Stand-in scheme.** `TransparentBits::verify_opening` rejects a table that differs from the committed one in one bit and a `C` with one changed value; `BitsWire::read` of its opening proof rejects every length other than `32·2^t`; `log_T` above 20 is a typed error on both sides before any allocation. Its module documentation states how it meets the contract of §11, that it is not succinct and that it is not for use outside tests.
- [ ] **Ownership split.** The skeleton compiles with a batch declared in `jolt-rv64i-verifier` and `impl_stage_prover!` expanded in `jolt-rv64i-prover` on the local wrapper of §12.
- [ ] **Gates.** `cargo fmt --check` passes. `cargo clippy -p jolt-rv64i-verifier -p jolt-rv64i-prover --all-targets -- -D warnings` passes with default features and again with `--features jolt-rv64i-prover/test-utils`, which also enables `test-utils` of the verifier crate. `cargo nextest run -p jolt-rv64i-verifier -p jolt-rv64i-prover --features jolt-rv64i-prover/test-utils --cargo-quiet` passes; this is the command that builds the fixture batch, `TransparentBits` and the synthetic witness and runs every test that uses them. Each integration test of the prover crate that needs them declares `required-features = ["test-utils"]` in `Cargo.toml`, so the default-feature commands build without them and no test is skipped silently under the feature. `test-utils` is not a default feature of either crate. `cargo tree -p jolt-rv64i-verifier -e normal` lists none of `jolt-kernels`, `jolt-prover`, `jolt-rv64i-prover`.

### Testing Strategy

Ground truth is of five kinds, none of which is the code under test: direct sums over a Boolean cube written in the test; `Polynomial::evaluate` on tables built in the test; `WitnessRow::compute` and the interpreter and replay of the arithmetisation's test corpus for machine semantics; literals of this document for counts and identifiers; frozen byte vectors for the preamble and for one proof. The tests include the arithmetisation's helpers through `#[path]` from `crates/jolt-rv64i-arith/tests/suite/common/` (`asm.rs` the encoder, `interp.rs` the interpreter, `harness.rs` the conversion of a record to `CycleFacts`, `replay.rs` the replay of the five obligations), so machine semantics has one test oracle in the workspace. Each batch has a batch-local test that proves and verifies that batch alone from honest input claims computed from the witness; these tests are what lets the stage PRs of Execution proceed in parallel. Tests follow `.claude/skills/test-policy/SKILL.md`: the keep clauses that apply are public contract (end to end, commitment lifecycle), algebraic property (points, expressions, routers, terminal check), wire compatibility (identifiers, schedule literals, tensor digest, the two byte vectors) and soundness (tamper, statement, order of `C` and `rho`). No existing test changes. The `host` and `zk` modes do not apply.

### Performance

**Proof size.** With `d_b = ⌈b/4⌉`, `d_a = ⌈a/4⌉` and `D = max(d_b, d_a)`, the proof holds

```text
round elements    78 + (16 + D)·t + 3a + 2b
hand-off values   43
C                 256
```

elements of `F128`, by summing §5. At `(t, b, a) = (22, 20, 20)` that is 640 + 43 + 256 = 939 elements, 15,024 bytes. The envelope of §13 adds 26 bytes (the version, `log_K_ram`, `final_pc` and two length prefixes) and the bytes of the scheme's commitment and opening proof. At `a = 23` the round elements are 671. The value counts do not depend on the layout, because no chunk value is sent.

**Verifier.** The figures are counts of operations in the source as this document specifies it, at the reference sizes. None is a timing and none was obtained by running code. `M` is a multiplication of two elements of `F128` and `A` an addition. `lift` is one evaluation `Σ_{i<64} w[i]·eq(p, i)` of a `u64` against a 64-entry table: additions of table entries and no multiplication.

*Rounds and transcript.* A round message with `ℓ` stored coefficients costs `2ℓ − 1` `M` and `2ℓ + 1` `A` to check and to evaluate at the challenge (`crates/jolt-poly/src/compressed_univariate.rs`), one labelled absorb of `41 + 25ℓ` framed bytes and one squeeze. The table takes every message at the degree of its batch, which is the maximum (§13).

| Batch | Rounds × degree | Round `M` | Round `A` | Squeezes | Labelled absorbs |
|---|---:|---:|---:|---:|---:|
| 1 | 30 × 3 | 150 | 210 | 89 | 38 |
| 2 | 10 × 2 | 30 | 50 | 17 | 13 |
| 3a | 17 × 2 | 51 | 85 | 18 | 23 |
| 3b | 22 × 5 | 198 | 242 | 27 | 45 |
| 4 | 42 × 3 | 210 | 294 | 68 | 52 |
| 5 | 22 × 4 | 154 | 198 | 26 | 28 |
| 6a | 20 × 2 | 60 | 100 | 37 | 22 |
| 6b | 22 × 6 | 242 | 286 | 41 | 281 |
| all | 640 elements | 1,095 | 1,465 | 323 | 502 |

For any layout the round multiplications number `121 + (27 + 2D)·t + 5a + 3b` and the round additions `191 + (37 + 2D)·t + 7a + 5b`; at `a = 23` they are 1,154 and 1,530. The 323 squeezes are the 85 coordinates of the four vector draws, 35 scalar challenges, 18 batching coefficients and 185 round challenges, 16 bytes each. The 502 labelled absorbs are 18 input claims, 185 round messages and 299 values; a labelled element is 66 framed bytes, and the 502 carry 44,507 bytes at full degree. Squeezes and absorbs are events of the transcript interface, not invocations of the hash: the number of compressions follows from the byte counts and the sponge's rate. Before the first batch the preamble makes 36 `append_bytes` calls under six labels, `1,412` framed bytes plus the input and output bytes, followed by the commit phase of the scheme.

*Other work per batch.* The first column is the derived terms. The second is the two expressions of each member evaluated as §8 writes them, one `M` per product sign of the expanded form. The third is the batch fold: three `M` per member (the input claim and the expected output by the batching coefficient, the output by its idle factor) and one per idle round.

| Batch | Derived terms | Expressions | Fold |
|---|---|---:|---:|
| 1 | two equality weights: `8 + m_F + 2t`, 57 | 6 | 9 |
| 2 | equality tables of `2^8`, `2^{m_F}` and `2^10` entries; at most two `M` per nonzero of the row matrices. Their number `N_M` is a constant of the layout, 1,255 at the reference layout; `tests/spartan.rs` pins it as a literal | 9 | 3 |
| 3a | equality tables over the witness columns and over each router's source and selector domains, under `2^12` entries in all; two `M` per nonzero of the five route tensors. Their number `N_R` is a constant of the layout, 75,371 at the reference layout and 69,191 at `(b, a) = (4, 5)`; `tests/routers.rs` pins both, each with a digest of the tensors. 16 for the five idle factors | 5 | 3 |
| 3b | `5t` for the cycle weight of five members; under 100 for the slot weights | 67 | 15 |
| 4 | `2t + 3a`; a split equality table over the address, `2^{⌈a/2⌉+1}` entries; one `lift` and two `M` per public I/O word | 18 | 46 |
| 5 | `6t` for the two `lt` values; the same split table; one `lift` and two `M` per word of the initial RAM | 16 | 6 |
| 6a | `t` for `FinalPc`; 342 for the equality tables over the variant, the three kinds and the register index with their coefficients folded in, 118 for the five tables and 224 for the weighting; after the rounds, a split equality table over the bytecode index, `2^{⌈b/2⌉+1}` entries, and one pass over the valid bytecode rows: per row four `lift`s, lookups in the kind and register tables, and 9 `M`; two `M` after the pass for the entry and next coefficients | 17 | 3 |
| 6b | `6t` for six equality weights; 1,221 for `next(r_3, r_6)` (§3); 29 per chunk of four bits, 14 for its equality table and 15 for its indicators, and fewer for a shorter top chunk, for `d_b + d_a` chunks; `3t + 6` per column weight as §8.17 writes it, about 18,000 for the 256 columns, or `3t + 6·256` when the three equality values are computed once; `2^8 − 2 + 256` for the value at `rho` | 308 | 9 |

The expressions total 446 `M` and the folds 94.

The expression column counts the multiplications of the expression itself. The shared evaluator of `jolt-claims` does more today: `Expr::try_evaluate` starts each monomial at its coefficient and multiplies by every factor, so a coefficient of one costs one `M` per monomial, and the stage verifier rebuilds each expression to validate its shape. Measured on the merged stages the column reads 16, 27, 21, 225, 46, 41, 52 and 872 in place of 6, 9, 5, 67, 18, 16, 17 and 308. The difference is a property of the shared evaluator and not of this protocol; removing it is a change to `jolt-claims` that every protocol family receives, and the figures of this table are the target once it lands.

*Fixed set-up.* An equality table of `2^n` entries costs `2^n − 2` `M`, one per new entry. A `lift` reads one 64-entry table of bit weights per point, 62 `M`, built once and shared by every word lifted at that point (`points::WordLift`); a pass over public memory or over the bytecode builds it before its loop and multiplies nothing per `lift`. A split address table of `2^{⌈n/2⌉+1}` entries is two tables of `2^{⌈n/2⌉}` and `2^{⌊n/2⌋}` entries, built once per point and held for the pass that reads it. The tensor digest of `tests/routers.rs` is Blake2b-256 over, for each router in the order of `ROUTERS`: its identifier as one byte, its number of entries as a little-endian `u64`, and each entry's `column`, `source` and `selector` as little-endian `u64` in the order `RouteTensors::entries` returns them. At `(20, 20)` it is `62852e847e52bc33c5ee8b4da307019c131cdd2556834b2a27dd0e41cd905216` and at `(4, 5)` `f610b56418b8b90322fab152369b84229cfb28e0e9dace9f0c515e7b7f10b9ed`.

*Dependence on the trace length.* The front end reads no table of length `T`. Its work is a function of `t`, `a`, `b` and of public sizes: the number of public I/O words, of words of the initial RAM and of valid bytecode rows, and the constants `N_M` and `N_R` of the layout. With those fixed it grows as `t²` through `next` and as `t` elsewhere. Three terms are linear or square-root in a public domain and not logarithmic: the two split address tables, and the passes over the public memory and the bytecode. `a` is the prover's choice; check 3 of §7 bounds it by the memory layout of the statement, which bounds the address table. `CheckedInputs::new` rejects an unsupported dimension before any exponentiation or allocation. The cost of the commitment scheme is its own. `TransparentBits` verifies in time linear in `T`, and no statement of succinctness covers it.

**Prover.** This spec ships the reference tier, which is exempt from any per-cycle budget and usable up to about `t = 10`, `a = 10`, `b = 10`: its largest tables have `2^{a+t}` entries (batch 4), `2^17` entries (batch 3a) and `2^{t+9}` entries (the 512 leaf tables of the reduction in batch 6b). At those sizes an end-to-end run is expected to take seconds on one core; this is an estimate from table sizes and has not been measured. The types that a production path uses are fixed here so that optimised kernels do not need a second interface: `Bits` is `Arc<[BitsRow]>`, four `u64` per cycle; the five base words are `Arc<[CycleWords]>`, five `u64` per cycle; `BitsCommitmentProver::commit` receives the rows; a `Polynomial<F128>` of length proportional to `T` appears only in `reference`.

## Design

### Architecture

#### 1. Crates and modules

```text
jolt-rv64i-arith      (specs/rv64i-binary-arithmetisation.md)
        ▲
jolt-rv64i-verifier   ids, points, claims (symbolic), stages (concrete), public data, commitment interface, proof, verify
        ▲
jolt-rv64i-prover     witness plane, registry, reference kernels, commitment prover, prove
```

`jolt-rv64i-verifier` depends on `jolt-rv64i-arith`, `jolt-claims`, `jolt-verifier` (for `ConcreteSumcheck`, `#[derive(SumcheckBatch)]`, `VerifierError`), `jolt-sumcheck`, `jolt-transcript`, `jolt-field` with feature `binary`, `jolt-poly`, `jolt-crypto` (for `NoCommitment`), `jolt-r1cs`, `jolt-program` and `common` (the memory layout, `PublicIoMemory`, `PublicInitialRam`, `compute_max_ram_k`), `blake2`, `serde`, `thiserror`, `tracing`. It does not name `jolt-openings`. Its features are `allocative`, which the batch derive's output names, and `test-utils`, which compiles the fixture batch of the skeleton (Execution).

`jolt-rv64i-prover` adds `jolt-kernels` and `jolt-prover` (for `PrepareKernel`, `NaiveSumcheckProver`, `#[derive(KernelSlots)]`, `impl_stage_prover!`). Its features are `allocative`, `parallel` (forwarded to `jolt-kernels` and `jolt-sumcheck`) and `test-utils`, which compiles `commitment::transparent` and `Rv64iWitness::synthetic` and enables `test-utils` of the verifier crate.

Symbolic and concrete relations share one crate because the symbolic layer of this family has no second consumer. They stay in separate modules, and `claims` imports no transcript type.

```text
jolt-rv64i-verifier/src
  ids.rs                 §4
  points.rs              §3: eq, lt, next, lift, chunk, to_high_to_low
  claims/                claim structs and SymbolicSumcheck, one file per relation template
    spartan_outer.rs  spartan_inner.rs  router_short.rs  router_cycle.rs
    registers_read_checking.rs  ram_read_checking.rs  ram_output_check.rs
    val_evaluation.rs  bytecode_read.rs  ram_ra_product.rs  bits_reduction.rs
  public/                verifier-computed data
    matrices.rs  routes.rs  bytecode.rs  io.rs  ram_init.rs
  stages/
    stage1/ stage2/ stage3a/ stage3b/ stage4/ stage5/ stage6a/ stage6b/
                         each: mod.rs (batch struct), <relation>.rs (ConcreteSumcheck), verify.rs (wiring)
  commitment.rs          §11: BitsGeometry, BitsOpening, BitsWire, BitsCommitmentScheme, squeeze_bytes
  statement.rs           Statement, CheckedInputs (§7)
  preprocessing.rs       VerifierPreprocessing and its digest (§13)
  proof.rs               the wire structs and the envelope (§7, §13)
  transcript.rs          the preamble (§6, §13)
  verifier.rs  error.rs
jolt-rv64i-prover/src
  plane.rs               Rv64iWitness, CycleWords, Rv64iPlane
  commitment/            mod.rs (BitsCommitmentProver), transparent.rs (TransparentBits, feature test-utils)
  reference/             views.rs, and one file per group of relations:
    spartan.rs  routers.rs  read_checking.rs  val_evaluation.rs
    bytecode.rs  ra_product.rs  bits_reduction.rs
  optimized/             the PrepareKernel adapters of the kernels of specs/rv64i-binary-prover-kernels.md;
                         declared by the skeleton, filled by that spec's PRs
  stages/                one file per batch: the local wrapper of the batch, its kernel registry
                         and its impl_stage_prover! invocation (§12)
  backend.rs             Rv64iBackend, the eight registries
  prover.rs  error.rs
```

Relations and batches are generic in exactly one parameter `F: JoltField`, which the stage macros require, and are instantiated at `F128` only. Public data enters a relation through its constructor as values of `F`. `Rv64iVerifierError` wraps `VerifierError` and adds the family's variants for each statement check of §7, the proof decoder, the commit phase and the opening; the last two hold the scheme's error boxed.

#### 2. Sizes and notation

`T = 2^t` cycles, `K_b = 2^b` bytecode rows, `K = 2^a` RAM words. A chunk of an index has 4 bits, the top chunk `bits mod 4` when that is nonzero; `d_b` and `d_a` count the chunks, and `n(·)` is `chunk_indicators`. The rows of `RowSystem` form two blocks. The block over `F_2` is rows 0–129, padded with zero rows to `2^8`. The block over `F128` is rows 130 onward in the order of `RowSystem::to_matrices()`, renumbered from 0: there are `8 + d_b + d_a` of them, padded to `2^{m_F}` with `m_F = ⌈log2(8 + d_b + d_a)⌉`; `m_F` is 5 at the reference layout. The witness has 1,024 columns per cycle with the constant at column 0, so its column index has 10 variables.

The admitted dimensions are `1 ≤ t ≤ LOG_T_MAX` with `LOG_T_MAX = 32`, a constant of `statement.rs`; `1 ≤ b ≤ 24` and `a ≤ 61`, which `Layout::new` enforces together with the 256-column limit; and `a ≥ 5` (§7). Hence `d_b ≤ 6` and `d_a ≤ 16`, the degree of `BytecodeReadCycle` is at most 7 and that of `RamRaProduct` at most 17, and the block over `F128` has between 11 and 30 rows, so `m_F` is 4 or 5.

The reference sizes are `(t, b, a) = (22, 20, 20)`.

#### 3. Points

One rule holds for every table, point and sum-check of the family.

- **Index order.** A multilinear table is indexed by an integer whose bit `i` is variable `i`. A point is a `Vec<F>` whose coordinate `i` belongs to variable `i`. `p ++ q` is concatenation, low variables first.
- **Binding order.** Every sum-check binds `BindingOrder::LowToHigh`: round `u` of a member binds its variable `u`. A member's sum-check point is therefore its window of the batch challenges, in order, with no reversal.
- **Conversion.** `jolt_poly::Polynomial::evaluate` and `jolt_poly::EqPlusOnePolynomial` take the most significant variable first. `points::to_high_to_low` reverses a point; it is called inside `points::next` and in the reference tier, and nowhere else. The commitment scheme receives its points in the order of this section (§11).

Tables and their indices:

| Table | Index | Point |
|---|---|---|
| `Bits` | `y + 256·j` | `column (8) ++ cycle (t)` |
| a column of `Bits` | `j` | `cycle` |
| `Inc` and every 64-bit word | `i + 64·j` | `bit (6) ++ cycle` |
| a bytecode function of width `2^m` | `e + 2^m·j` | `inner (m) ++ cycle`; `m` is 6 for `PC`, `Imm`, `FallThroughPC`, `PCPlusImm`, `Variant`; 5 for `Rs1Ra`, `Rs2Ra`, `RdWa`; 4 for `AccessKind`; 3 for `ShiftKind`, `KeyKind`; 0 for `Branch`, `Store` |
| the witness | `c + 1024·j`, `c = i + 64·lane` | `column (10) ++ cycle` |
| outer tables | `row + 2^m·j` | `row (m) ++ cycle` |
| `RegistersVal` | `k + 32·i + 2048·j` | `address (5) ++ bit ++ cycle` |
| `RamVal` | `k + K·i + 64K·j` | `address (a) ++ bit ++ cycle` |
| `RamValFinal`, `RamValInit`, `ValIo` | `k + K·i` | `address ++ bit` |
| `RamRa`, `BytecodeRa`, the register selectors | `k + K·j` | `address ++ cycle` |
| chunk `c` of a selector, all digit values | `e + 2^{bits_c}·j` | `digit (bits_c) ++ cycle` |
| a router's short table | the 17 slots of §8.4 | `x` |

An address point `q` splits into digit points `q_c = q[4c .. 4c + bits_c)`, least significant chunk first, as `Layout` orders the chunks. A kind index is the discriminant of `ShiftKind`, `AccessKind` or `KeyKind` in `jolt-rv64i-arith`; a variant index is `Variant::index()`.

**Closed forms**, for points `x, y ∈ F^n`, a `u64` word `w` and a `Chunk` `ch` with `n_ch = ch.indicators()` stored indicators from column `ch.start()`:

```text
eq(x, y)      = ∏_i (1 + x_i + y_i)
lt(x, y)      = Σ_i (1 + x_i)·y_i·∏_{k>i} (1 + x_k + y_k)                          extension of [x < y]
next(x, y)    = Σ_i (∏_{k<i} x_k·(1 + y_k))·(1 + x_i)·y_i·∏_{k>i} (1 + x_k + y_k)    extension of [y = x + 1]
lift(w, p)    = Σ_{i<64} w[i]·eq(p, i)                                             p ∈ F^6
chunk(ch, p; V) = eq(p, 0) + Σ_{k=1}^{n_ch} (eq(p, k) + eq(p, 0))·V[ch.start() + k − 1]   p ∈ F^{ch.bits()}
```

`eq(x, k)` with an integer `k` is `eq` against the bits of `k`. `next` has no wrap: it is zero when `x` is the last index, and `next(x, 0) = 0` for every `x`. `chunk` is the extension in the digit of the full one-hot table of a chunk whose stored indicators have the values `V`; it is affine in `V` because the indicator of digit 0 is the complement `1 + Σ_k V[·]`. The zero-extension factor of an idle variable is `1 + X`.

`points::next(x, y)` has one owner, `jolt_poly::EqPlusOnePolynomial::new(to_high_to_low(x)).evaluate(&to_high_to_low(y))`, the evaluation that `crates/jolt-verifier/src/stages/stage3/spartan_shift.rs` exposes as the derived terms `SpartanShiftPublic::EqPlusOneOuter` and `EqPlusOneProduct`. It uses the field's one, sum, difference and product and no other constant, so it is correct in characteristic 2, where its per-variable equality factor is `1 + x + y`. It asserts that the two lengths agree; `points::next` checks them first and returns a typed error. Its cost is quadratic in the number of variables: term `k` of its sum builds two products afresh, `3n − k` multiplications, so one call is `(5n² + n)/2` multiplications, 1,221 at `n = 22` and 2,265 at `n = 30`. The verifier calls it once per proof, and that cost is accepted; this spec asks for no change to `jolt-poly`. The other forms cost at most `3n` multiplications with running products.

**Named points.** `z_n` is the challenge vector of batch `n`.

| Name | Definition | Length |
|---|---|---|
| `ρ_2`, `ρ_F` | `z_1[0..8)`, `z_1[8 − m_F .. 8)` | 8, `m_F` |
| `r_1` | `z_1[8 .. 8+t)` | `t` |
| `w` | `z_2`, the witness-column point | 10 |
| `x` | `z_3a`, the short point | 17 |
| `r_bit`, `p_0`, `p_1` | `x[0..6)`, `x[6..9)`, `x[9..12)`: the bit point of every word claim and the two digit points of `Pos` | 6, 3, 3 |
| `q_V`, `q_S`, `q_M`, `q_C` | `x[11..17)`, `x[12..15)`, `x[13..17)`, `x[14..17)`: the variant point and the three kind points | 6, 3, 4, 3 |
| `r_3` | `z_3b` | `t` |
| `a_ram`, `a_reg` | `z_4[0..a)`, `z_4[a − 5 .. a)` | `a`, 5 |
| `r_4` | `z_4[a .. a+t)` | `t` |
| `tau` | the address point drawn by the RAM output check | `a` |
| `r_5` | `z_5` | `t` |
| `a_bc` | `z_6a` | `b` |
| `r_6` | `z_6b` | `t` |
| `rho` | the column point drawn after `C` | 8 |

#### 4. Identifiers

`jolt_rv64i_verifier::ids` is the module that `#[protocol(ids = crate::ids)]` names. Every enum derives `Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize`.

```rust
pub const FAMILY: &str = "rv64i-binary";

pub enum Router { Variant, Shift, Memory, Compare, Branch }             // 0..=4
pub enum RowBlock { F2, F128 }                                           // 0, 1
pub enum CycleWeight { Router, Read, Val, Entry, Next }                  // eq(r_3,·) eq(r_4,·) eq(r_5,·) eq(0,·) next(r_3,·)

pub enum RelationId {                                                    // index
    SpartanOuterF2, SpartanOuterF128, SpartanInner, RouterShort,         // 0, 1, 2, 3
    RouterCycleVariant, RouterCycleShift, RouterCycleMemory,             // 4, 5, 6
    RouterCycleCompare, RouterCycleBranch,                               // 7, 8
    RegistersReadChecking, RamReadChecking, RamOutputCheck,              // 9, 10, 11
    RegistersValEvaluation, RamValEvaluation,                            // 12, 13
    BytecodeReadAddress, BytecodeReadCycle, RamRaProduct, BitsReduction, // 14, 15, 16, 17
}

pub enum CommittedPolynomial {                                           // tag; every variant is a functional of Bits
    DirectColumns,                                                       // 0   §8.3
    VariantBits,                                                         // 1   §8.5
    PosRa0, PosRa1,                                                      // 2, 3   chunk(pos_ra[d], digit; column values)
    ShouldBranch,                                                        // 4   column should_branch()
    Inc,                                                                 // 5   columns 0..64 as a word
    Column(usize),                                                       // 6   column y, y < 256
}

pub enum VirtualPolynomial {                                             // tag
    Az, Bz, Cz,                                                          // 0, 1, 2
    WitnessRouted,                                                       // 3
    RouterFold(Router),                                                  // 4
    Rs1Value, Rs2Value, RdPreValue, RamReadValue, NextPC,                // 5..=9    base words
    PC, Imm, FallThroughPC, PCPlusImm,                                   // 10..=13  bytecode words
    Variant, ShiftKind, AccessKind, KeyKind, Branch, Store,              // 14..=19  bytecode selectors
    Rs1Ra, Rs2Ra, RdWa,                                                  // 20, 21, 22
    RegistersVal, RamVal, RamValFinal,                                   // 23, 24, 25
    RamRa,                                                               // 26
    BytecodeAddressClaim,                                                // 27
    BytecodeRaChunk(usize), RamRaChunk(usize),                           // 28, 29
}

pub enum PolynomialId { Committed(CommittedPolynomial), Virtual(VirtualPolynomial) }
pub struct OpeningId { pub polynomial: PolynomialId, pub relation: RelationId }
impl OpeningId {
    pub fn committed(p: CommittedPolynomial, r: RelationId) -> Self;
    pub fn virtual_polynomial(p: VirtualPolynomial, r: RelationId) -> Self;
}
```

An opening id names one claim cell: the polynomial and the relation that produces the claim. The same polynomial produced by two relations is two ids (`RdWa` of `RegistersReadChecking` and of `RegistersValEvaluation`; `RamRa` of `RamReadChecking` and of `RamValEvaluation`), so no payload is needed to tell points apart. No id depends on the layout: a `usize` payload is a column or a chunk index.

`DerivedId` and `ChallengeId` have one variant per relation template, wrapping that template's sub-enum. `ChallengeId` has `From<S>` for each of its six sub-enums `S`, which `#[derive(SumcheckChallenges)]` calls; each is wrapped by exactly one variant. `DerivedId` has no `From`: `ReadCheckingDerived` and `ValEvaluationDerived` are each wrapped by two variants, and `OuterDerived` and `RouterCycleDerived` need a block or a router beside them. A relation builds its `DerivedId` values by naming the variant.

```rust
pub enum DerivedId {
    SpartanOuter(RowBlock, OuterDerived),                 // EqTau
    SpartanInner(InnerDerived),                           // MatrixWeight, PublicColumns
    RouterShort(RouterShortDerived),                      // RouteWeight(Router)
    RouterCycle(Router, RouterCycleDerived),              // EqCycle, WordSlot(usize), OneSlot
    RegistersReadChecking(ReadCheckingDerived),           // EqCycle
    RamReadChecking(ReadCheckingDerived),                 // EqCycle
    RamOutputCheck(OutputCheckDerived),                   // EqTau, IoMask, ValIo
    RegistersValEvaluation(ValEvaluationDerived),         // Lt
    RamValEvaluation(ValEvaluationDerived),               // Lt, InitEval
    BytecodeReadAddress(BytecodeAddressDerived),          // EntryPc, FinalPc
    BytecodeReadCycle(BytecodeCycleDerived),              // BytecodeFold(CycleWeight), Weight(CycleWeight)
    RamRaProduct(RamRaProductDerived),                    // EqRead, EqVal
    BitsReduction(BitsReductionDerived),                  // PosZero(usize), ColumnWeight(usize)
}

pub enum ChallengeId {
    SpartanInner(InnerChallenge),                         // AzF2, BzF2, CzF2, AzF128, BzF128, CzF128
    RegistersReadChecking(RegistersReadChallenge),        // Rs1, Rs2, Rd
    RamValEvaluation(RamValChallenge),                    // Val, Final
    BytecodeReadAddress(BytecodeChallenge),               // sixteen, §8.14
    RamRaProduct(RamRaChallenge),                         // Read, Val
    BitsReduction(BitsReductionChallenge),                // DirectColumns, VariantBits, PosRa0, PosRa1, ShouldBranch, Inc
}
```

Sub-enum variants are tagged in the order written, from 0. Every challenge with an id is one scalar, as `#[derive(SumcheckChallenges)]` requires; there are 35. The four vector challenges have no id, because no expression names them: `tau` of each outer relation and of the output check are stage-level draws held by the concrete relation, and `rho` is drawn by the wiring of batch 6b (§10).

**Encoding.** All three kinds map into `ExternalId { family: FAMILY, index }`:

```text
opening    index = relation | kind << 8 | tag << 16 | payload << 24     kind: 0 virtual, 1 committed
derived    index = relation | tag << 8 | payload << 16
challenge  index = relation | tag << 8
```

`relation` is the index above, below `2^8`; for `SpartanOuter` and `RouterCycle` it is the index of the instance. `tag` is below `2^8`. The payload of a unit variant is 0; of `Router` and `CycleWeight` their discriminant; of a `usize` variant the index. Bits 9–15 of an opening index are zero. The two literals of the acceptance criteria: `(Inc, RegistersValEvaluation)` is `12 | 1 << 8 | 5 << 16 = 0x5_010C`, and `(Column(229), BitsReduction)` is `17 | 1 << 8 | 6 << 16 | 229 << 24 = 0xE506_0111`.

`From<OpeningId> for ComposedOpeningId` and the two analogues are total. `TryFrom` returns `Ok` only if the family is `FAMILY`, the relation, the tag and the payload lie in the decode domain of §13, and re-encoding the decoded id gives the input; otherwise it returns the composite unchanged in `Err`. A payload has 40 bits in an opening index and 48 in a derived index; `From` maps a larger payload to the all-ones payload, which is outside every decode domain. Every payload the family constructs is inside its domain, so the conversions are injective on the identifiers in use, and a saturated id resolves to no claim.

#### 5. Batches and placement

A stage is one generated batch, except stages 3 and 6, which are two each. Members are listed in declaration order. A window `[lo, hi)` is the member's rounds within the batch. "Values" counts wire values.

| Batch | Struct | Member | Rounds | Window | Degree | Values |
|---|---|---|---:|---|---:|---:|
| 1 | `Stage1Sumchecks` | `SpartanOuterF2` | `8 + t` | full | 3 | 3 |
| | | `SpartanOuterF128` | `m_F + t` | `[8 − m_F, 8 + t)` | 3 | 3 |
| 2 | `Stage2Sumchecks` | `SpartanInner` | 10 | full | 2 | 2 |
| 3a | `Stage3aSumchecks` | `RouterShort` | 17 | full | 2 | 5 |
| 3b | `Stage3bSumchecks` | `RouterCycleVariant` | `t` | full | 3 | 10 |
| | | `RouterCycleShift` | `t` | full | 5 | 3 |
| | | `RouterCycleMemory` | `t` | full | 4 | 2 |
| | | `RouterCycleCompare` | `t` | full | 5 | 1 |
| | | `RouterCycleBranch` | `t` | full | 4 | 2 |
| 4 | `Stage4Sumchecks` | `RegistersReadChecking` | `5 + t` | `[a − 5, a + t)` | 3 | 4 |
| | | `RamReadChecking` | `a + t` | full | 3 | 2 |
| | | `RamOutputCheck` | `a` | `[0, a)` | 3 | 1 |
| 5 | `Stage5Sumchecks` | `RegistersValEvaluation` | `t` | full | 4 | 3 |
| | | `RamValEvaluation` | `t` | full | 4 | 1 |
| 6a | `Stage6aSumchecks` | `BytecodeReadAddress` | `b` | full | 2 | 1 |
| 6b | `Stage6bSumchecks` | `BytecodeReadCycle` | `t` | full | `d_b + 1` | 0 |
| | | `RamRaProduct` | `t` | full | `d_a + 1` | 0 |
| | | `BitsReduction` | `t` | full | 2 | 256 |

A batch has the rounds of its longest member and the degree of its highest. In the proof's bytes every round message is `degree` field elements, the linear coefficient omitted; in memory and in the transcript it is the shortest form of §13, which has fewer when the round polynomial has lower degree. Stage 3 is two batches because a batch has one degree: the 17 short rounds have degree 2 and the cycle rounds degree 5. Stage 6 is two batches for the same reason and because its two phases run over different variables. At the reference sizes the batches have 30, 10, 17, 22, 42, 22, 20 and 22 rounds of degree 3, 2, 2, 5, 3, 4, 2 and 6.

**Placement under zero extension.** A batch member is `BatchMember { input_claim, coefficient, rounds, offset }` of `crates/jolt-sumcheck/src/batch.rs`: it is active in the rounds `[offset, offset + rounds)` and idle in the others, where its summand carries the factor `1 + X`. The offset comes from `ConcreteSumcheck::instance_point_offset`, and the batch's final claim is `Σ_i c_i · λ_i · e_i` with `λ_i` the product of `1 + z_j` over the member's idle rounds. That is one contiguous window per member, and it expresses three of the four idle patterns:

| Idle rounds | Expressed by | Members |
|---|---|---|
| at the head (active on a tail window) | the default offset, `batch rounds − rounds` | `SpartanOuterF128`, `RegistersReadChecking` |
| at the tail (active on a head window) | `instance_point_offset` returning 0 | `RamOutputCheck` |
| on both sides of one window | `instance_point_offset` returning the offset; exercised by the `offset = 1` test of `specs/binary-sumcheck.md` | none |
| between a member's active rounds | not expressible | none, by the two choices below |

A member that is active on a set of rounds with a gap cannot be declared. The schedule would call for one in two places, and both are arranged so that no gap occurs.

- **Short rounds of stage 3.** The five routers use different subsets of the 17 slots (§8.4): the Variant router is idle in slot 10, the Memory router in slots 9–11, the Branch router in slots 6–11 and 13–16. They are therefore one member, `RouterShort`, whose summand is the sum of the five routers' summands, each multiplied by its own idle factor inside its public weight. The member occupies all 17 rounds, and its `λ` is 1. This is also what the claim requires: a residual word is the sum of the outputs of several routers, and only the sum is claimed.
- **Address rounds of stage 4.** The registers member has five address variables and the batch has `a` address rounds before its `t` cycle rounds. The registers' address rounds are the last five address rounds, `[a − 5, a)`, so its window `[a − 5, a + t)` is contiguous and is the default tail window, and `a_reg = a_ram[a − 5 .. a)`. Any other choice of five address rounds would leave a gap before the cycle rounds. The statement check `a ≥ 5` makes `RamReadChecking` the longest member.

The smallest change that would admit a gap is to replace `rounds` and `offset` of `BatchMember` by a list of windows, to have `ConcreteSumcheck::instance_point` return the concatenation of the member's windows of the batch point, and to take `λ_i` over the complement; `BatchMember::is_active` already decides activity per round. This spec does not need it and does not ask for it.

Three alignments carry meaning. In batch 1 the rows over `F128` occupy the last `m_F` row rounds, so both members bind their cycle variables in rounds `[8, 8 + t)` and share `r_1`. In batch 4 the output check runs on the address rounds of `RamReadChecking`, so both end at `a_ram`; §8.13 depends on this. In batch 6b all three members occupy all `t` rounds, so its final claim has no idle factor.

#### 6. Transcript

The transcript is `Blake2bTranscript<F128>`; a challenge is a 16-byte squeeze read as an element of `F128`.

This section fixes the order; §13 fixes the bytes of each step.

**Preamble**, in this order, before any challenge of the front end:

1. `Transcript::new(b"jolt-rv64i-binary-v0")`.
2. `params`: `t`, `b`, `a`, `LowestAddress`.
3. `statement`: the entry PC, the memory layout, the public input bytes, the public output bytes, the panic flag.
4. `program`: the 32-byte digest of the preprocessing.
5. `final_pc`.
6. The commit phase of the commitment scheme (§11), on the same transcript. The scheme absorbs its commitment and may exchange further messages and draw challenges of its own. The front end draws nothing before this phase returns.

**A batch**, as generated by `#[derive(SumcheckBatch)]` and called from the stage's `verify`:

1. Stage-level draws, each one `challenge_vector`: before batch 1, `tau` of `SpartanOuterF2` (`8 + t` scalars), then `tau` of `SpartanOuterF128` (`m_F + t`); before batch 4, `tau` of the output check (`a`). No other batch has one.
2. `draw_challenges`: each member in declaration order draws the fields of its `Challenges` struct in declaration order, one `challenge_scalar` each. Batch 6b is the exception. Its members are declared in the order `BytecodeReadCycle`, `RamRaProduct`, `BitsReduction`, and its scalars are drawn in the order: the six of `BitsReduction`, then the two of `RamRaProduct`. `Stage6bSumchecks` therefore carries `#[sumcheck_batch(no_draw_challenges)]`, and `stages/stage6b/mod.rs` holds the `draw_challenges` that both sides call and that assembles the batch's challenge aggregate.
3. `verify_clear` computes each member's input claim, absorbs them in member order under `b"sumcheck_claim"`, and draws one batching coefficient per member, in member order.
4. Per round: the round message under `b"sumcheck_poly"`, then the round challenge.
5. The verifier derives the opening points, evaluates `Σ_i c_i · λ_i · expected_output_i` and compares it with the final claim.
6. `append_output_claims`: every wire value under `b"opening_claim"`, members in declaration order, fields in declaration order, aliased cells skipped. Batch 6b absorbs `C[0], …, C[255]` in this step and nothing else.

The batches run in the order 1, 2, 3a, 3b, 4, 5, 6a, 6b. **After batch 6b** the verifier draws `rho` as `challenge_vector(8)`, absorbs nothing more, and calls the opening of the commitment scheme on the same transcript (§11).

The order gives each challenge the absorptions it needs. Every challenge follows the whole preamble, the commit phase included. The six fold coefficients of `SpartanInner` follow the six outer values. `tau` of the output check and the three read coefficients of `RegistersReadChecking` follow the 18 values of batch 3b and precede every round of batch 4; the batching coefficients of batch 4 follow its three input claims. The two coefficients of `RamValEvaluation` follow the seven values of batch 4 and precede the batching coefficients of batch 5. The sixteen coefficients of `BytecodeReadAddress` follow every bytecode claim value: the last of them is absorbed after batch 5. The six coefficients of `BitsReduction` and the two of `RamRaProduct` follow the rounds and the value of batch 6a, and the three batching coefficients of batch 6b follow them and the three input claims. `rho` follows `C`.

**Stage wiring.** Each `stages/<batch>/verify.rs` has one function, in the shape of `crates/jolt-verifier/src/stages/stage5/verify.rs`:

```rust
pub fn verify<T: Transcript<Challenge = F128>>(
    checked: &CheckedInputs<'_>,     // statement.rs: the statement, the preprocessing, Layout, PublicIoMemory, the initial RAM, t, b, a
    proof: &BatchProof<V>,           // the batch's field of Rv64iProof
    transcript: &mut T,
    /* one reference per earlier Output that it reads */
) -> Result<Output, Rv64iVerifierError>;
```

`CheckedInputs` exists only as the result of the statement checks of §7. `Output` holds the batch point, the batch's output cells with their points, and the stage's vector challenge when it has one; nothing else passes between stages. The function constructs the concrete members from `checked` and the earlier outputs, calls `expand` (§7), runs steps 1–6 above and returns. `verifier::verify` runs the statement checks, the preamble with the commit phase and the eight functions in order, and is their only caller; the function of batch 6b also receives the scheme's verifier state and ends with the opening call. The table lists what each function computes outside the generated flow; each entry is defined in the section it names.

| Batch | Reads the outputs of | Before the rounds | After the terminal check |
|---|---|---|---|
| 1 | | the two `tau` draws | |
| 2 | 1 | the matrices of the layout (`public/matrices.rs`, §8.3) | |
| 3a | 2 | the route tensors of the layout (`public/routes.rs`, §8.4) | |
| 3b | 1, 3a | | |
| 4 | 3a, 3b | the `tau` draw | |
| 5 | 3a, 4 | `InitEval` (`public/ram_init.rs`, §8.13) | |
| 6a | 3a, 3b, 4, 5 | `EntryPc`, `FinalPc` (§8.14) | `h_t` for the five weights (`public/bytecode.rs`, §8.14), returned in `Output` |
| 6b | all seven | the hand-written `draw_challenges`; `PosZero(d)` and the six `l_u` (§8.17); steps 1–2 of §10 | steps 4–6 of §10 |

On the prover side the same order is the sequence of `impl_stage_prover!` drivers of §12, each followed by the move of its absorbed values into the wire struct.

#### 7. Statement, preprocessing, proof

```rust
pub struct Statement {
    pub log_T: u8,
    pub entry_pc: u64,
    pub device: common::jolt_device::JoltDevice,   // memory layout, inputs, outputs, panic; both advice vectors empty
}

pub struct VerifierPreprocessing<S: BitsCommitmentScheme> {   // private fields, read through accessors
    bytecode: Arc<jolt_rv64i_arith::Bytecode>,      // 2^b rows, LowestAddress; shared with the witness
    image: Vec<(u64, u64)>,                         // the program image: word index, nonzero value; indices strictly increasing
    digest: [u8; 32],                               // §13
    scheme: S::VerifierSetup,
}
impl<S: BitsCommitmentScheme> VerifierPreprocessing<S> {
    pub fn new(bytecode: impl Into<Arc<Bytecode>>, image: Vec<(u64, u64)>, scheme: S::VerifierSetup)
        -> Result<Self, PreprocessingError>;
    pub fn bytecode(&self) -> &Bytecode;
    pub fn shared_bytecode(&self) -> &Arc<Bytecode>;      // the handle a host passes to `Rv64iWitness`
}

pub struct Rv64iProof<S: BitsCommitmentScheme> {
    pub log_K_ram: u8,
    pub final_pc: u64,
    pub bits_commitment: S::Commitment,                   // every prover message of the commit phase
    pub stage1: BatchProof<OuterValues>,                  // 6 values
    pub stage2: BatchProof<InnerValues>,                  // 2
    pub stage3a: BatchProof<RouterFoldValues>,            // 5
    pub stage3b: BatchProof<RouterCycleValues>,           // 18
    pub stage4: BatchProof<ReadCheckingValues>,           // 7
    pub stage5: BatchProof<ValEvaluationValues>,          // 4
    pub stage6a: BatchProof<BytecodeAddressValue>,        // 1
    pub stage6b: BatchProof<BitsColumns>,                 // 256
    pub opening: S::OpeningProof,
}
impl<S: BitsCommitmentScheme> Rv64iProof<S> {
    pub fn to_bytes(&self) -> Vec<u8>;                    // §13
    pub fn from_bytes(bytes: &[u8], log_T: u8, log_K_bytecode: u8) -> Result<Self, ProofDecodeError>;
}
pub struct BatchProof<V> { pub rounds: SumcheckProof<F128, NoCommitment>, pub values: V }

pub struct OuterValues {
    pub az_f2: F128, pub bz_f2: F128, pub cz_f2: F128,
    pub az_f128: F128, pub bz_f128: F128, pub cz_f128: F128,
}
pub struct InnerValues { pub witness_routed: F128, pub direct_columns: F128 }
pub struct RouterFoldValues {
    pub variant: F128, pub shift: F128, pub memory: F128, pub compare: F128, pub branch: F128,
}
pub struct ReadCheckingValues {
    pub rs1_ra: F128, pub rs2_ra: F128, pub rd_wa: F128, pub registers_val: F128,
    pub ram_ra: F128, pub ram_val: F128,
    pub ram_val_final: F128,
}
pub struct BytecodeAddressValue { pub address_claim: F128 }
pub struct RouterCycleValues {
    pub rs1_value: F128, pub rs2_value: F128, pub rd_pre_value: F128, pub imm: F128,
    pub fall_through_pc: F128, pub pc_plus_imm: F128, pub pc: F128, pub next_pc: F128,
    pub variant_bits: F128, pub variant: F128,
    pub shift_kind: F128, pub pos_ra_0: F128, pub pos_ra_1: F128,
    pub ram_read_value: F128, pub access_kind: F128,
    pub key_kind: F128,
    pub branch: F128, pub should_branch: F128,
}
pub struct ValEvaluationValues { pub rd_wa: F128, pub store: F128, pub inc: F128, pub ram_ra: F128 }
pub struct BitsColumns(pub Vec<F128>);                   // C, exactly BITS_COLUMNS values
```

`rounds` is `SumcheckProof::Clear(ClearProof::Compressed(_))` with every round message in the canonical form of §13; statement check 8 rejects anything else before the engine reads it. `NoCommitment` of `jolt-crypto` fills the commitment parameter that the clear path never reads. The wire structs do not derive `Serialize`: the bytes of a proof are those of `to_bytes`.

**Wire values and cells.** The proof carries the wire values of a batch and nothing else, in a struct of `proof.rs` whose fields are in the order of §6 step 6. `expand`, in the stage's `verify.rs`, builds the generated output aggregate of the batch from that struct before `verify_clear` runs. For batches 1, 2, 3a, 4 and 6a every output cell is a wire value and `expand` moves fields. For batches 3b and 5 it also copies each aliased cell from its source (§8.5–§8.9, §8.13); the relations declare the pairs in `aliased_output_openings`, so the generated absorb skips the copies and the generated `validate_aliases` re-checks them. For batch 6b, `values` is `C`, and `expand` builds the aggregate by projection: the `Column` cells of `BitsReduction` are `C`, and each chunk cell of the two products is `chunk` of §3 applied to `C` (§10). `Stage6bSumchecks` carries `#[sumcheck_batch(no_opening_values, no_output_shape, no_draw_challenges)]`: the stage absorbs `C` itself, checks `C.len() == BITS_COLUMNS` itself and draws its scalars in the order of §6. The proof format therefore has one owner, `proof.rs`, and does not change when a relation's claim struct gains or loses an alias.

`VerifierPreprocessing::new` rejects an image whose word indices are not strictly increasing or that holds a zero value, and computes `digest` (§13) over the bytecode rows, `LowestAddress` and the image. The bytecode is held behind an `Arc` so that a host builds the table once and the witness of §12 holds the same allocation; the digest is over its contents and does not depend on who else holds it. `K_ram` is the prover's choice per proof, so `log_K_ram` is a field of the proof and is absorbed in step 2 of the preamble.

**Statement checks**, run by `CheckedInputs::new(preprocessing, statement, proof)` in `statement.rs`, which `verifier::verify` calls before the preamble. They run in this order and each has its own error variant. No check before check 5 allocates in proportion to a size that the statement or the proof chooses; the only allocation before it is the two chunk lists of `Layout::new`, of at most 6 and 16 entries, made after that function has bounded `b` and `a`:

1. **Dimensions.** `1 ≤ t ≤ LOG_T_MAX`, `a ≥ 5`, and `Layout::new(b, a, LowestAddress)` succeeds with `LowestAddress = Bytecode::lowest_address()`: `1 ≤ b ≤ 24`, `a ≤ 61`, `LowestAddress` is a multiple of 8, `LowestAddress + 8·K_ram ≤ 2^64` without wrapping, and the row fits 256 columns.
2. **Memory layout.** Both advice vectors of the device are empty. The memory layout `m` of the device is canonical: `MemoryLayout::try_new` of the `MemoryConfig` with the fields `max_input_size`, `max_trusted_advice_size`, `max_untrusted_advice_size`, `max_output_size`, `stack_size`, `heap_size` of `m` and `program_size: Some(m.program_size)` succeeds and returns a layout equal to `m` in all 20 fields. `try_new` is the one owner of the layout's invariants, and equality transfers them to `m`: every capacity and boundary of the I/O range is a multiple of 8; its regions follow one another without gap or overlap in the order advice, input, output, panic word, termination word, from a lowest address above zero to `io_end`, which is at or below `RAM_START_ADDRESS`; each region has its stated capacity. Any canonical layout is admitted, advice capacities included.
3. **Base and bound.** `Bytecode::lowest_address()` equals `m.get_lowest_address()`, and `K_ram ≤ compute_max_ram_k(m)`, compared as exponents. The word index `(A − LowestAddress)/8` of the arithmetisation is then `MemoryLayout::remapped_word_address(A)` for every address, by the definition of that function in `common/src/jolt_device.rs`, and the prover's `a` is bounded by the statement.
4. **Segment lengths.** `inputs.len() ≤ m.max_input_size` and `outputs.len() ≤ m.max_output_size`. With check 2 each segment then lies inside its own region.
5. **I/O words.** `io = PublicIoMemory::new(&statement.device)` and `PublicInitialRam::inputs_only(&statement.device)` succeed, and `io.io_mask_end ≤ K_ram`. By check 2 the mask `[io.io_mask_start, io.io_mask_end)` runs from the first input word to the word of `RAM_START_ADDRESS`, is not empty, and contains the output region, the panic word and the termination word.
6. **Image.** Every word index of `image` is at or above `io.io_mask_end` and below `K_ram`.
7. **Final PC.** `Bytecode::final_pc_index(final_pc)` succeeds: `final_pc` is the PC of a valid row.
8. **Proof shape.** Every `rounds` field is a compressed clear proof with the batch's round count; every round message is canonical (§13): at least one stored coefficient, at most the batch's degree, and a nonzero last one when there are two or more; `C` has 256 values. The other value counts are fixed by the types.

`K_ram` is otherwise the prover's choice. A `K_ram` smaller than the memory the program touches has no witness, because the RAM index of a cycle is `d_a` committed digits.

Checks 1–7 read the proof only through `log_K_ram` and `final_pc`. They are the body of `CheckedInputs::of_statement(preprocessing, statement, log_K_ram, final_pc)`, which the prover calls with the values of its witness before it commits; `new` is that function followed by check 8. The checks have this one owner on both sides.

The initial RAM of the protocol is the image together with the input words of `PublicInitialRam::inputs_only`, and nothing else: the output region, the panic word and the termination word start at zero. It has one representation, the *canonical initial RAM*: a vector of pairs (word index, value) with strictly increasing indices, every index below `K_ram` and every value nonzero. `CheckedInputs::initial_ram()` is its one owner. It lists the nonzero input words in index order and then the image; the input words lie inside the I/O mask and the image at or above its end (check 6), so the concatenation is increasing, and a zero input word is omitted. Two initial memories are equal exactly when their canonical vectors are equal. `io.segments`, which also holds the outputs, the panic word and the termination word, is read by the output check alone (§8.12).

Three bindings are equations of the protocol, not checks of the statement. **Entry:** the public value `lift(entry_pc, r_bit)` is the claim `PC(r_bit, 0)`, one of the claims of bytecode read checking (§8.14). **Final PC:** `final_pc` supplies the last-cycle term of the next-cycle claim of bytecode read checking (§8.14). **Outputs and termination:** the RAM output check binds the final contents of the I/O word range to `PublicIoMemory`, whose termination word is 1 exactly when `panic` is false (§8.12, §8.13).

#### 8. Relations

##### 8.1 How a relation is stated

Each relation is a symbolic type in `claims/` (its `RelationId`, its claim structs and its two expressions) and a concrete type in `stages/` (its instance data, points and derived terms). The entries below give, in a fixed order: the instance, which is what the constructor holds; rounds and degree; the summand, which is the definition the tests use as ground truth; the input cells with the relation each comes `from`; the challenges; the output cells with their points; the two expressions; the derived terms with their cost in field multiplications.

In an expression, a name in `snake_case` is `opening(id)` of the cell of that name, `D.Name` is `Derived(..)` of the relation's sub-enum, and `c.Name` is `Challenge(..)`. A product over a sum stands for its expansion. Every coefficient is 1.

A cell's point is a concatenation of the member's sum-check point and of earlier points held by the instance; `derive_opening_points` returns exactly the points listed. A derived term of an output expression is computed in `derive_output_term` from the instance, the points and the challenges. A derived term of an input expression is computed by the stage wiring before the batch and held by the instance, because `derive_input_term` receives no points; these are marked "instance".

A relation with no input cell has `input_expression = 0`. A relation with one input cell and no other term has `input_expression` equal to that cell.

Three kinds of cell occur. A **virtual** cell is an evaluation of a table defined from `Bits`, the bytecode and the initial RAM: a base word (`Rs1Value`, `Rs2Value`, `RdPreValue`, `RamReadValue`, `NextPC`, defined below), a bytecode function of the fetched row (`PC`, `Imm`, `FallThroughPC`, `PCPlusImm`, the selectors), a state table (`RegistersVal`, `RamVal`, `RamValFinal`), an address indicator (`RamRa`, and the chunk tables), or a sum the protocol introduces. A **committed** cell is an evaluation of a fixed linear functional of the columns of `Bits`, plus a public constant for `PosRa0` and `PosRa1`. An **alias** is a cell equal by definition to a wire cell of another member of the same batch, at the same point.

A word `X` as a table has entry `X[i, j]`, bit `i` at cycle `j`, and `X(p, q) = Σ_j eq(q, j)·lift(X[j], p)`.

**The witness contract.** Every table of the protocol is a function of four things: the rows of `Bits`, the bytecode, the initial RAM (§7) and `final_pc`. Write `row_j` for the bytecode row at `Layout::bytecode_index` of cycle `j`, `Store[j]` for `Variant::is_store` of its variant and `Inc[j]` for `Layout::inc`. The state before cycle 0 is 32 registers equal to zero and the initial RAM. The step of cycle `j` is

```text
registers[row_j.rd]              += (1 + Store[j]) · Inc[j]
ram[Layout::ram_index(Bits[j])]  += Store[j] · Inc[j]
```

as words, `+` being XOR. `RegistersVal[·, ·, j]` and `RamVal[·, ·, j]` are the state before cycle `j`, and `RamValFinal` is the RAM after the last cycle. The five base words of cycle `j` are

```text
Rs1Value[j]     = registers[row_j.rs1]                 before cycle j
Rs2Value[j]     = registers[row_j.rs2]                 before cycle j
RdPreValue[j]   = registers[row_j.rd]                  before cycle j
RamReadValue[j] = ram[Layout::ram_index(Bits[j])]      before cycle j
NextPC[j]       = row_{j+1}.pc for j < T − 1,  final_pc for j = T − 1
```

Three points of this definition are conventions that a witness has to follow, because the read-checking members of batch 4 are not gated by the kind of the cycle.

- **A cycle without an access.** `BitsBuilder::bits_row` writes the RAM index only for a variant with `access()`, so on every other cycle the stored indicators are zero and `Layout::ram_index` is 0. `RamReadValue[j]` is then the current content of RAM word 0, the word at `LowestAddress`. That word is real memory: with no advice capacity it is the first input word. It is not `CycleFacts::ram_pre_value`, which an adapter leaves at zero without an access and which `BaseWords::from_facts` copies; the rows of the arithmetisation do not read `RamReadValue` on such a cycle, and `RamReadChecking` (§8.11) does.
- **An operand that the row does not have.** `BytecodeRow::column` gives the selector of register 0 for an absent `rs1`, `rs2` or `rd`. The base word is then the content of register 0.
- **Register 0.** The step above has no exception for register 0. In an execution it stays zero: a row whose destination is `x0` or absent has a variant without a register write (`Variant::from_source`), the rows force `RdWriteValue = 0` for such a variant, and `RdWriteValue = RdPreValue + (1 + Store)·Inc` (§8.4) then forces a zero increment as long as register 0 holds zero.

**A trace shorter than `2^t`.** The protocol has no padding cycle of its own: every one of the `2^t` rows is a cycle of the step above, fetched from a valid bytecode row. A host that holds fewer executed cycles extends them with repetitions of a *stall cycle*, a fetch that is its own successor and that, executed in the state it left, changes no register and no RAM word; its increment is zero, its `NextPC` is its own PC, and `final_pc` is that PC. Whether the last executed cycle admits such a repetition is a property of the program (an unconditional jump to itself does; a jump through a register that the jump itself overwrites need not), and deciding it is the host's obligation, not a check of the verifier. A trace of exactly `2^t` executed cycles needs no stall cycle: `final_pc` is then the successor of its last cycle, and check 7 of §7 requires it to be the PC of a valid row. Termination is not read from the shape of the trace in either case. It is the output check of §8.12, which binds the termination word.

A table that follows this definition makes every member whose input claim is a wire value a true sum, whether or not `Bits` is an execution. On an execution, `base_words` of `crates/jolt-rv64i-arith/tests/suite/common/replay.rs` makes the same four state reads from its own replayed state, and is the test oracle of the definition.

##### 8.2 `SpartanOuterF2`, `SpartanOuterF128`

One implementation over the row block `m`; two symbolic types, because a relation's id is a property of its type. `μ` is 8 for `F2` and `m_F` for `F128`.

- **Instance.** The block, `t`, and `tau ∈ F^{μ+t}`.
- **Rounds** `μ + t`, **degree** 3.
- **Summand** over row `i < 2^μ` and cycle `j`: `eq(tau, (i, j))·(Az[i, j]·Bz[i, j] + Cz[i, j])`, where `Az[i, j] = Σ_col A_m[i, col]·Z[col, j]`, `Z[·, j]` is the witness row of cycle `j` (`WitnessRow::compute`), and `A_m`, `B_m`, `C_m` are the block's rows of `RowSystem::to_matrices()`.
- **Inputs.** None. **Challenges.** None.
- **Outputs**, all at the member's point `ρ_m ++ r_1`: `az: Az`, `bz: Bz`, `cz: Cz`.
- **Output expression.** `D.EqTau·az·bz + D.EqTau·cz`.
- **Derived.** `EqTau = eq(tau, ρ_m ++ r_1)`: `μ + t`.

##### 8.3 `SpartanInner`

Let `Virtual = {1, 2, 3} ∪ [16, 27) ∪ [64, 768)` be the 718 witness columns that the routers produce, and `Direct` the set of `Bits` columns `c` whose witness column `768 + c` has a nonzero entry in a matrix. `Direct` is the interval `[64, Layout::keys_differ()]`: the bytecode, RAM and `Pos` indicators and `KeysDiffer`, 165 columns at the reference layout. The inner relation is stated for `Z'`, the vector per cycle with the constant 1 at column 0, the routed values at `Virtual`, the committed bits at `768 + Direct`, and zero elsewhere. An honest witness row agrees with `Z'` on every column that has a nonzero matrix entry, so `Z'` satisfies the same inner identity as `Z`.

- **Instance.** `ρ_2`, `ρ_F`, `r_1`, the matrices.
- **Rounds** 10, **degree** 2.
- **Summand** over column `col < 1024`: `M[col]·Z'(col, r_1)`, with `M[col] = Σ_m Σ_{X ∈ {A, B, C}} c.X_m · Σ_i eq(ρ_m, i)·X_m[i, col]`.
- **Inputs.** `az_f2, bz_f2, cz_f2` from `SpartanOuterF2`; `az_f128, bz_f128, cz_f128` from `SpartanOuterF128`.
- **Challenges.** `AzF2, BzF2, CzF2, AzF128, BzF128, CzF128`.
- **Input expression.** `c.AzF2·az_f2 + c.BzF2·bz_f2 + c.CzF2·cz_f2 + c.AzF128·az_f128 + c.BzF128·bz_f128 + c.CzF128·cz_f128`.
- **Outputs**, both at `w ++ r_1`:
  - `witness_routed: WitnessRouted`, with `WitnessRouted(w, r_1) = Σ_{c ∈ Virtual} eq(w, c)·(Z(c, r_1) + [c = 16])`;
  - `direct_columns: committed DirectColumns`, with `DirectColumns(w, r_1) = Σ_{c ∈ Direct} eq(w, 768 + c)·Bits(c, r_1)`.
- **Output expression.** `D.MatrixWeight·witness_routed + D.MatrixWeight·direct_columns + D.MatrixWeight·D.PublicColumns`.
- **Derived.** `MatrixWeight = Σ_col eq(w, col)·M[col]`: three equality tables (`2^8`, `2^{m_F}`, `2^10` entries) and at most two multiplications per nonzero entry. `PublicColumns = eq(w, 0) + eq(w, 16)`: the constant column, and the 1 that column 16 carries on every cycle beyond its routed part (`Z[16, j] = 1 + Valid[j]`).

The whole virtual part of the witness is one claim, because the terminal check of this relation uses it only under the weights `eq(w, ·)`, and the routers prove exactly that combination.

##### 8.4 The routers and `RouterShort`

A router `r` has a bank of source bits `Source_r[s, j]`, a selector `Select_r[h, j]` that is a product of one-hot factors, and a public tensor `Route_r[c, s, h] ∈ {0, 1}`. Together the five satisfy, for every cycle `j` and every `c ∈ Virtual`,

```text
Z[c, j] + [c = 16] = Σ_r Σ_{s, h} Route_r[c, s, h] · Source_r[s, j] · Select_r[h, j]
```

A source index is `s = i + 64·n` for bit `i` of word slot `n`. The short sum-check runs over 17 slot variables shared by the five routers:

| Router | Word slots of the bank | Selector factors | Bit | Word | Selector | Idle |
|---|---|---|---|---|---|---|
| `Variant` | 0–7: `Rs1Value, Rs2Value, RdPreValue, Imm, FallThroughPC, PCPlusImm, PC, NextPC`; 8: `Inc`; from source index 576, one bit per `Bits` column from `ram_ra()[0].start()` to `used_columns() − 1`; then the constant 1 | `Variant` (6 bits) | 0–5 | 6–9 | 11–16 | 10 |
| `Shift` | 0: `Rs1Value` | low `Pos` digit (3), high `Pos` digit (3), `ShiftKind` (3) | 0–5 | | 6–8, 9–11, 12–14 | 15–16 |
| `Memory` | 0: `RamReadValue`; 1: `Rs2Value` | low `Pos` digit (3), `AccessKind` (4) | 0–5 | 12 | 6–8, 13–16 | 9–11 |
| `Compare` | 0–2: `Rs1Value, Rs2Value, Imm`; source index 192: the constant 1 | low `Pos` digit, high `Pos` digit, `KeyKind` (3) | 0–5 | 12–13 | 6–8, 9–11, 14–16 | |
| `Branch` | 0: `FallThroughPC`; 1: `PCPlusImm` | the product `Branch·ShouldBranch`, no index | 0–5 | 12 | | 6–11, 13–16 |

The bit of a word is in slots 0–5 for every router, so every word claim of the cycle phase is at the one bit point `r_bit`; the two `Pos` digits are in slots 6–8 and 9–11 for every router that uses them, so the digit claims are at `p_0` and `p_1`. The `Variant` bank has `577 + n(a) + 17` entries, 669 at the reference layout, inside its `2^10`.

The `Variant` router produces the three carry words, the three AND rails, `KeyDiff`, `LessThan`, the eleven control bits, and the parts of the three residual words that are linear in the bank; `Shift` adds the shift output to the `rd` residual; `Memory` adds the load output to the `rd` residual and the store increment to the RAM residual; `Compare` produces `KeyDiffAbove` and the two key bits; `Branch` adds the taken-branch term to the next-PC residual. `public/routes.rs` generates the tensors from `jolt-rv64i-arith`: `Term::wires()` of every form of `Variant::line()`, of `shift_form`, `load_form`, `store_form`, `KeyKind::keys()` and `BRANCH_FORM`. Three sources are expanded on the way. `Source::RdWriteValue` under variant `v` becomes `RdPreValue`, plus `Inc` when `v` is not a store, since `RdWriteValue = RdPreValue + (1 + Store)·Inc`; entries that coincide cancel. `Source::RamAddress` and `Source::Pos` become sums of indicator bits through `Chunk::digit_bit_mask`. The expansion of `RdWriteValue` is one function of the variant in `jolt-rv64i-arith`, added with the routers.

Write `u|_r` for the restriction of `u ∈ {0,1}^17` to the active slots of router `r`, split into its source part and its selector part, and

```text
Fold_r[u|_r] = Σ_j eq(r_1, j) · Source_r[u|src, j] · Select_r[u|sel, j]
W_r[u|_r]    = Σ_{c ∈ Virtual} eq(w, c) · Route_r[c, u|src, u|sel]
Idle_r(u)    = ∏_{k idle for r} (1 + u_k)
```

- **Instance.** `w` and the tensors.
- **Rounds** 17, **degree** 2.
- **Summand** over `u ∈ {0,1}^17`: `Σ_r Idle_r(u)·W_r[u|_r]·Fold_r[u|_r]`. The idle factor belongs to the public weight and occurs once per router.
- **Inputs.** `witness_routed` from `SpartanInner`. **Challenges.** None.
- **Outputs.** `variant, shift, memory, compare, branch`, with ids `RouterFold(r)`, each at `x|_r`.
- **Output expression.** `Σ_r D.RouteWeight(r)·fold_r`.
- **Derived.** `RouteWeight(r) = Idle_r(x)·W_r(x|_r)`, the extension of `W_r` in the slot variables: equality tables over the 1,024 columns and over each router's source and selector domains, then two multiplications per nonzero entry of `Route_r`.

##### 8.5 `RouterCycleVariant`

The five cycle relations share this shape. The instance holds `r_1` and `x`. There are `t` rounds. The summand over cycle `j` is `eq(r_1, j)·Source_r(x|src, j)·Select_r(x|sel, j)`, with both factors extended in the slot variables. The one input cell is `fold: RouterFold(r)` from `RouterShort`. There are no challenges. `D.EqCycle = eq(r_1, r_3)` costs `t`. A word cell is at `r_bit ++ r_3`.

- **Degree** 3.
- **Outputs.**
  - `rs1_value, rs2_value, rd_pre_value, imm, fall_through_pc, pc_plus_imm, pc, next_pc`: the virtual words of slots 0–7, each at `r_bit ++ r_3`;
  - `variant_bits: committed VariantBits` at `x[0..10) ++ r_3`, with `g = ram_ra()[0].start()` and

    ```text
    VariantBits(x[0..10), r_3) = eq(x[6..10), 8) · Σ_{i<64} eq(r_bit, i) · Bits(i, r_3)
                               + Σ_{c=g}^{used_columns()−1} eq(x[0..10), 576 + c − g) · Bits(c, r_3)
    ```

  - `variant: Variant` at `q_V ++ r_3`, with `Variant(q, j) = eq(q, v)` when cycle `j` fetches a valid row of variant index `v`, and 0 on an invalid row.
- **Output expression.** `D.EqCycle·(Σ_{n<8} D.WordSlot(n)·word_n + variant_bits + D.OneSlot)·variant`, with `word_n` the eight word cells in slot order.
- **Derived.** `WordSlot(n) = eq(x[6..10), n)`. `OneSlot = eq(x[0..10), 576 + used_columns() − g)`. Under 60 in all.

##### 8.6 `RouterCycleShift`

- **Degree** 5.
- **Outputs.**
  - `rs1_value`: alias of `rs1_value` of `RouterCycleVariant`;
  - `shift_kind: ShiftKind` at `q_S ++ r_3`, with `ShiftKind(q, j) = eq(q, e)` when the variant of cycle `j` has `shift()` of kind index `e`, and 0 otherwise;
  - `pos_ra_0: committed PosRa0` at `p_0 ++ r_3` and `pos_ra_1: committed PosRa1` at `p_1 ++ r_3`, with `PosRa_d(p, r_3) = Σ_j eq(r_3, j)·chunk(pos_ra()[d], p; Bits(·, j)) = chunk(pos_ra()[d], p; Bits(·, r_3))`.
- **Output expression.** `D.EqCycle·rs1_value·shift_kind·pos_ra_0·pos_ra_1`.

##### 8.7 `RouterCycleMemory`

- **Degree** 4.
- **Outputs.**
  - `ram_read_value: RamReadValue` at `r_bit ++ r_3`;
  - `rs2_value`: alias of `rs2_value` of `RouterCycleVariant`;
  - `access_kind: AccessKind` at `q_M ++ r_3`, with `AccessKind(q, j) = eq(q, e)` when the variant of cycle `j` has `access()` with a kind of index `e`, and 0 otherwise;
  - `pos_ra_0`: alias of `pos_ra_0` of `RouterCycleShift`.
- **Output expression.** `D.EqCycle·(D.WordSlot(0)·ram_read_value + D.WordSlot(1)·rs2_value)·access_kind·pos_ra_0`.
- **Derived.** `WordSlot(0) = 1 + x[12]`, `WordSlot(1) = x[12]`.

##### 8.8 `RouterCycleCompare`

- **Degree** 5.
- **Outputs.**
  - `rs1_value, rs2_value, imm`: aliases of the cells of those names of `RouterCycleVariant`;
  - `key_kind: KeyKind` at `q_C ++ r_3`, with `KeyKind(q, j) = eq(q, e)` when the variant of cycle `j` has `key_kind()` of index `e`, and 0 otherwise;
  - `pos_ra_0, pos_ra_1`: aliases of the cells of those names of `RouterCycleShift`.
- **Output expression.** `D.EqCycle·(D.WordSlot(0)·rs1_value + D.WordSlot(1)·rs2_value + D.WordSlot(2)·imm + D.OneSlot)·key_kind·pos_ra_0·pos_ra_1`.
- **Derived.** `WordSlot(n) = eq(x[12..14), n)`. `OneSlot = eq(r_bit, 0)·eq(x[12..14), 3)`.

##### 8.9 `RouterCycleBranch`

- **Degree** 4.
- **Outputs.**
  - `fall_through_pc, pc_plus_imm`: aliases of the cells of those names of `RouterCycleVariant`;
  - `branch: Branch` at `r_3`, the indicator that the variant of the cycle has `branch()`;
  - `should_branch: committed ShouldBranch` at `r_3`, the value `Bits(should_branch(), r_3)`.
- **Output expression.** `D.EqCycle·(D.WordSlot(0)·fall_through_pc + D.WordSlot(1)·pc_plus_imm)·branch·should_branch`.
- **Derived.** `WordSlot(0) = 1 + x[12]`, `WordSlot(1) = x[12]`.

The ten alias cells of batch 3b are those named in §8.6–§8.9. Each relation returns its pairs from `aliased_output_openings`, the alias first and its source second.

##### 8.10 `RegistersReadChecking`

`RegistersVal[k, i, j]` is bit `i` of register `k` before cycle `j`. `Rs1Ra[k, j]`, `Rs2Ra[k, j]` and `RdWa[k, j]` are bit `k` of the columns of those names of the row that cycle `j` fetches (`BytecodeRow::column`).

- **Instance.** `r_bit`, `r_3`.
- **Rounds** `5 + t`, **degree** 3. The member's point is `a_reg ++ r_4`.
- **Summand** over register `k < 32` and cycle `j`: `eq(r_3, j)·(c.Rs1·Rs1Ra[k, j] + c.Rs2·Rs2Ra[k, j] + c.Rd·RdWa[k, j])·RegistersVal(k, r_bit, j)`.
- **Inputs.** `rs1_value, rs2_value, rd_pre_value` from `RouterCycleVariant`.
- **Challenges.** `Rs1, Rs2, Rd`.
- **Input expression.** `c.Rs1·rs1_value + c.Rs2·rs2_value + c.Rd·rd_pre_value`.
- **Outputs.** `rs1_ra: Rs1Ra`, `rs2_ra: Rs2Ra`, `rd_wa: RdWa` at `a_reg ++ r_4`; `registers_val: RegistersVal` at `a_reg ++ r_bit ++ r_4`.
- **Output expression.** `D.EqCycle·(c.Rs1·rs1_ra + c.Rs2·rs2_ra + c.Rd·rd_wa)·registers_val`.
- **Derived.** `EqCycle = eq(r_3, r_4)`: `t`.

The third read is what defines `RdPreValue`; the value written to `rd` is `RdPreValue + (1 + Store)·Inc` by the expansion of §8.4, and §8.13 applies the same increment to the register table.

##### 8.11 `RamReadChecking`

`RamVal[k, i, j]` is bit `i` of RAM word `k` before cycle `j`. `RamRa[k, j]` is 1 when `k` is the RAM index of cycle `j` (`Layout::ram_index`), on every cycle: a cycle without an access has index 0, and the relation is not gated, so it holds for `RamReadValue` as §8.1 defines it and for no other table.

- **Instance.** `r_bit`, `r_3`.
- **Rounds** `a + t`, **degree** 3. The member's point is `a_ram ++ r_4`.
- **Summand** over word `k < K` and cycle `j`: `eq(r_3, j)·RamRa[k, j]·RamVal(k, r_bit, j)`.
- **Inputs.** `ram_read_value` from `RouterCycleMemory`. **Challenges.** None.
- **Outputs.** `ram_ra: RamRa` at `a_ram ++ r_4`; `ram_val: RamVal` at `a_ram ++ r_bit ++ r_4`.
- **Output expression.** `D.EqCycle·ram_ra·ram_val`.
- **Derived.** `EqCycle = eq(r_3, r_4)`: `t`.

##### 8.12 `RamOutputCheck`

`RamValFinal[k, i]` is bit `i` of RAM word `k` after the last cycle. With `io` the `PublicIoMemory` of statement check 5: `IoMask[k] = 1` for `io.io_mask_start ≤ k < io.io_mask_end`, and `ValIo[k]` is the word that `io.segments` places at index `k` (the inputs, the outputs, the panic word, and the termination word 1 when `panic` is false), zero elsewhere.

- **Instance.** `tau ∈ F^a`, `r_bit`, `io`.
- **Rounds** `a`, **degree** 3, `instance_point_offset = 0`. The member's point is `a_ram`.
- **Summand** over word `k < K`: `eq(tau, k)·IoMask[k]·(RamValFinal(k, r_bit) + ValIo(k, r_bit))`. It sums to 0.
- **Inputs.** None. **Challenges.** None.
- **Outputs.** `ram_val_final: RamValFinal` at `a_ram ++ r_bit`.
- **Output expression.** `D.EqTau·D.IoMask·ram_val_final + D.EqTau·D.IoMask·D.ValIo`.
- **Derived.** `EqTau = eq(tau, a_ram)`: `a`. `IoMask = below(a_ram, io_mask_end) + below(a_ram, io_mask_start)`, with `below(p, n)` the extension of `[k < n]`, equal to `lt(p, n)` for `n < K` and to 1 for `n = K`: `2a`. `ValIo = Σ_k eq(a_ram, k)·lift(ValIo[k], r_bit)` over the words of `io.segments`: one `lift` and two multiplications per word with a split equality table.

##### 8.13 `RegistersValEvaluation`, `RamValEvaluation`

`Store[j]` is 1 when the variant of cycle `j` is a store (`Variant::is_store`). `Inc(p, q)` is the word of `Bits` columns 0–63. Registers start at zero. `RamValInit[k]` is the initial RAM of §7: the word of `image` at index `k`, or the input word that `PublicInitialRam::inputs_only` places at `k`, and zero elsewhere. It never reads `io.segments`, whose output, panic and termination words are final contents.

`RegistersValEvaluation`:

- **Instance.** `a_reg`, `r_bit`, `r_4`.
- **Rounds** `t`, **degree** 4.
- **Summand** over cycle `j`: `lt(j, r_4)·RdWa(a_reg, j)·(1 + Store[j])·Inc(r_bit, j)`.
- **Inputs.** `registers_val` from `RegistersReadChecking`. **Challenges.** None.
- **Outputs.** `rd_wa: RdWa` at `a_reg ++ r_5`; `store: Store` at `r_5`; `inc: committed Inc` at `r_bit ++ r_5`.
- **Output expression.** `D.Lt·rd_wa·inc + D.Lt·rd_wa·store·inc`.
- **Derived.** `Lt = lt(r_5, r_4)`: `3t`.

`RamValEvaluation` proves two identities with one sum, which is possible because `ram_val` and `ram_val_final` are at the same address point `a_ram` and the same bit point `r_bit`:

```text
RamVal(a_ram, r_bit, r_4)   + RamValInit(a_ram, r_bit) = Σ_j lt(j, r_4) · RamRa(a_ram, j) · Store[j] · Inc(r_bit, j)
RamValFinal(a_ram, r_bit)   + RamValInit(a_ram, r_bit) = Σ_j              RamRa(a_ram, j) · Store[j] · Inc(r_bit, j)
```

- **Instance.** `a_ram`, `r_bit`, `r_4`, and `InitEval`.
- **Rounds** `t`, **degree** 4.
- **Summand** over cycle `j`: `(c.Val·lt(j, r_4) + c.Final)·RamRa(a_ram, j)·Store[j]·Inc(r_bit, j)`.
- **Inputs.** `ram_val` from `RamReadChecking`; `ram_val_final` from `RamOutputCheck`.
- **Challenges.** `Val, Final`.
- **Input expression.** `c.Val·ram_val + c.Val·D.InitEval + c.Final·ram_val_final + c.Final·D.InitEval`.
- **Outputs.** `ram_ra: RamRa` at `a_ram ++ r_5`; `store`, `inc`: aliases of the cells of those names of `RegistersValEvaluation`.
- **Output expression.** `c.Val·D.Lt·ram_ra·store·inc + c.Final·ram_ra·store·inc`.
- **Derived.** `Lt = lt(r_5, r_4)`. `InitEval = RamValInit(a_ram, r_bit) = Σ_k eq(a_ram, k)·lift(RamValInit[k], r_bit)` over the nonzero words (instance): one `lift` and two multiplications per word with a split equality table.

Together with §8.12 the second identity binds the final contents of the I/O range, which the output check equates with the statement, to the stores of the trace.

##### 8.14 `BytecodeReadAddress`

Bytecode read checking proves every claim on a function of the fetched row. A claim `u` is a function `f_u` of the bytecode row, a cycle weight, and a value. `BytecodeRa[k, j]` is 1 when cycle `j` fetches row `k` (`Layout::bytecode_index`). The five cycle weights, in the order of `CycleWeight`:

```text
E_Router(j) = eq(r_3, j)    E_Read(j) = eq(r_4, j)    E_Val(j) = eq(r_5, j)    E_Entry(j) = eq(0, j)    E_Next(j) = next(r_3, j)
```

The sixteen claims, in the order of `BytecodeChallenge`. `row` is row `k`; every `f_u` is 0 on an invalid row.

| `u` | `f_u(k)` | Weight | Value |
|---|---|---|---|
| `Imm`, `FallThroughPC`, `PCPlusImm`, `PC` | `lift` of that word of `row` at `r_bit` | `Router` | `imm`, `fall_through_pc`, `pc_plus_imm`, `pc` of `RouterCycleVariant` |
| `Variant` | `eq(q_V, row.variant.index())` | `Router` | `variant` of `RouterCycleVariant` |
| `ShiftKind` | `eq(q_S, e)` for a shift of kind index `e`, else 0 | `Router` | `shift_kind` of `RouterCycleShift` |
| `AccessKind` | `eq(q_M, e)` for an access with a kind of index `e`, else 0 | `Router` | `access_kind` of `RouterCycleMemory` |
| `KeyKind` | `eq(q_C, e)` for a key kind of index `e`, else 0 | `Router` | `key_kind` of `RouterCycleCompare` |
| `Branch` | 1 when the variant has `branch()`, else 0 | `Router` | `branch` of `RouterCycleBranch` |
| `Rs1Ra`, `Rs2Ra`, `RdWaRead` | `Σ_{e<32} eq(a_reg, e)·bit e` of the column `Rs1Ra`, `Rs2Ra`, `RdWa` of `row` | `Read` | `rs1_ra`, `rs2_ra`, `rd_wa` of `RegistersReadChecking` |
| `RdWaWrite` | as `RdWaRead` | `Val` | `rd_wa` of `RegistersValEvaluation` |
| `Store` | 1 when the variant is a store, else 0 | `Val` | `store` of `RegistersValEvaluation` |
| `Entry` | `lift(row.pc, r_bit)` | `Entry` | the public value `lift(entry_pc, r_bit)` |
| `Next` | `lift(row.pc, r_bit)` | `Next` | `next_pc` of `RouterCycleVariant`, plus the public value `(∏_i r_3[i])·lift(final_pc, r_bit)` |

The `Next` claim is the next-cycle identity. `NextPC[j]` is the PC of cycle `j + 1` for `j < T − 1` and is `final_pc` at `j = T − 1`. The right side `Σ_{j'} next(r_3, j')·PC[j']` equals `Σ_{j < T − 1} eq(r_3, j)·PC[j + 1]`: the last source cycle `j = T − 1` contributes nothing, because `next` has no wrap, while the weight of the last target cycle, `next(r_3, T − 1) = eq(r_3, T − 2)`, is in general not zero. The public term `∏_i r_3[i]·lift(final_pc, r_bit) = eq(r_3, T − 1)·lift(NextPC[T − 1], r_bit)` removes that source cycle from the left side. The `Entry` claim fixes the PC of cycle 0. Validity of the fetched row needs no claim: the rows force column 16, `1 + Valid`, to zero, and `Valid` is the sum of the variant selectors that the `Variant` claim binds.

With `H_t[k] = Σ_{u of weight t} c.u·f_u(k)` and `R_t[k] = Σ_j E_t(j)·BytecodeRa[k, j]`:

- **Instance.** `r_bit`, the kind points, `a_reg`, `r_3`, `r_4`, `r_5`, the bytecode, and the two public values.
- **Rounds** `b`, **degree** 2.
- **Summand** over row `k < K_b`: `Σ_t H_t[k]·R_t[k]`, five terms.
- **Inputs.** `imm, fall_through_pc, pc_plus_imm, pc, next_pc, variant` from `RouterCycleVariant`; `shift_kind` from `RouterCycleShift`; `access_kind` from `RouterCycleMemory`; `key_kind` from `RouterCycleCompare`; `branch` from `RouterCycleBranch`; `rs1_ra, rs2_ra, rd_wa_read` from `RegistersReadChecking`; `rd_wa_write, store` from `RegistersValEvaluation`. Fifteen cells.
- **Challenges.** The sixteen of the table.
- **Input expression.** `c.Imm·imm + c.FallThroughPC·fall_through_pc + c.PCPlusImm·pc_plus_imm + c.PC·pc + c.Variant·variant + c.ShiftKind·shift_kind + c.AccessKind·access_kind + c.KeyKind·key_kind + c.Branch·branch + c.Rs1Ra·rs1_ra + c.Rs2Ra·rs2_ra + c.RdWaRead·rd_wa_read + c.RdWaWrite·rd_wa_write + c.Store·store + c.Entry·D.EntryPc + c.Next·next_pc + c.Next·D.FinalPc`.
- **Outputs.** `address_claim: BytecodeAddressClaim` at `a_bc`: the value `Σ_t H_t(a_bc)·R_t(a_bc)`.
- **Output expression.** `address_claim`.
- **Derived.** `EntryPc = lift(entry_pc, r_bit)` and `FinalPc = (∏_i r_3[i])·lift(final_pc, r_bit)` (instance): `t`.

The output is one value because the five `R_t(a_bc)` are not claims the verifier can use separately: the cycle phase proves their combination. This is the arrangement of the two phases of bytecode read checking in `crates/jolt-verifier/src/stages/stage6a` and `stage6b`.

After the batch, `public/bytecode.rs` computes `h_t = H_t(a_bc) = Σ_k eq(a_bc, k)·H_t[k]` for the five weights in one pass over the valid rows, with a split equality table over `a_bc`, tables of `c.u·eq(q, e)` for the kinds and registers, and one `lift` table for `r_bit`: at most 10 multiplications per row. The five values depend on the bytecode, the challenges and earlier points only.

##### 8.15 `BytecodeReadCycle`

- **Instance.** `h = [h_Router, h_Read, h_Val, h_Entry, h_Next]`, `a_bc`, `r_3`, `r_4`, `r_5`, the layout.
- **Rounds** `t`, **degree** `d_b + 1`.
- **Summand** over cycle `j`: `(Σ_t h_t·E_t(j))·∏_{c<d_b} BytecodeRa_c(a_bc_c, j)`, where `BytecodeRa_c(p, j) = chunk(bytecode_ra()[c], p; Bits(·, j))`.
- **Inputs.** `address_claim` from `BytecodeReadAddress`. **Challenges.** None.
- **Outputs.** `chunks: Vec<_>` with ids `BytecodeRaChunk(c)`, `c < d_b`, at `a_bc_c ++ r_6`. They are projections of `C` (§10) and are not on the wire.
- **Output expression.** `Σ_t D.BytecodeFold(t)·D.Weight(t)·∏_c chunks[c]`, five terms.
- **Derived.** `BytecodeFold(t) = h_t` (instance). `Weight(t) = E_t(r_6)`: `eq(r_3, r_6)`, `eq(r_4, r_6)`, `eq(r_5, r_6)`, `eq(0, r_6) = ∏_i (1 + r_6[i])` and `next(r_3, r_6)`, the last through `points::next` of §3; `4t` and one call.

##### 8.16 `RamRaProduct`

- **Instance.** `a_ram`, `r_4`, `r_5`, the layout.
- **Rounds** `t`, **degree** `d_a + 1`.
- **Summand** over cycle `j`: `(c.Read·eq(r_4, j) + c.Val·eq(r_5, j))·∏_{c<d_a} RamRa_c(a_ram_c, j)`, where `RamRa_c(p, j) = chunk(ram_ra()[c], p; Bits(·, j))`.
- **Inputs.** `ram_ra_read`: the cell `ram_ra` of `RamReadChecking`; `ram_ra_val`: the cell `ram_ra` of `RamValEvaluation`.
- **Challenges.** `Read, Val`.
- **Input expression.** `c.Read·ram_ra_read + c.Val·ram_ra_val`.
- **Outputs.** `chunks: Vec<_>` with ids `RamRaChunk(c)`, `c < d_a`, at `a_ram_c ++ r_6`: projections of `C`, not on the wire.
- **Output expression.** `c.Read·D.EqRead·∏_c chunks[c] + c.Val·D.EqVal·∏_c chunks[c]`.
- **Derived.** `EqRead = eq(r_4, r_6)`, `EqVal = eq(r_5, r_6)`: `2t`.

##### 8.17 `BitsReduction`

The six committed claims made before batch 6b, in the order of `BitsReductionChallenge`. Each has the form `v_u = k_u + Σ_y l_u(y)·Bits(y, t_u)` with a public constant `k_u`, a public weight vector `l_u` over the 256 columns and a cycle point `t_u`.

| `u` | Input cell, from | `t_u` | `k_u` | `l_u(y)`, zero where not stated |
|---|---|---|---|---|
| `DirectColumns` | `direct_columns`, `SpartanInner` | `r_1` | 0 | `eq(w, 768 + y)` for `64 ≤ y ≤ keys_differ()` |
| `VariantBits` | `variant_bits`, `RouterCycleVariant` | `r_3` | 0 | `eq(x[6..10), 8)·eq(r_bit, y)` for `y < 64`; `eq(x[0..10), 576 + y − g)` for `g ≤ y < used_columns()`, `g = ram_ra()[0].start()` |
| `PosRa0` | `pos_ra_0`, `RouterCycleShift` | `r_3` | `eq(p_0, 0)` | `eq(p_0, k) + eq(p_0, 0)` at `y = pos_ra()[0].start() + k − 1`, `1 ≤ k ≤ 7` |
| `PosRa1` | `pos_ra_1`, `RouterCycleShift` | `r_3` | `eq(p_1, 0)` | `eq(p_1, k) + eq(p_1, 0)` at `y = pos_ra()[1].start() + k − 1`, `1 ≤ k ≤ 7` |
| `ShouldBranch` | `should_branch`, `RouterCycleBranch` | `r_3` | 0 | 1 at `y = should_branch()` |
| `Inc` | `inc`, `RegistersValEvaluation` | `r_5` | 0 | `eq(r_bit, y)` for `y < 64` |

Every coefficient of every `l_u` is a product of equality values at points fixed before batch 6b. The supports are columns 64–228 for `DirectColumns`, 0–63 and 139–230 for `VariantBits`, 214–220 and 221–227 for the two `PosRa`, 229 for `ShouldBranch` and 0–63 for `Inc`, at the reference layout. The spare columns have weight zero in every claim and are bound by the commitment alone.

- **Instance.** `r_1`, `r_3`, `r_5`, the six vectors `l_u` in sparse form, and `eq(p_0, 0)`, `eq(p_1, 0)`.
- **Rounds** `t`, **degree** 2.
- **Summand** over cycle `j`: `Σ_y (Σ_u c.u·l_u(y)·eq(t_u, j))·Bits[y, j]`.
- **Inputs.** The six cells of the table.
- **Challenges.** `DirectColumns, VariantBits, PosRa0, PosRa1, ShouldBranch, Inc`.
- **Input expression.** `c.DirectColumns·direct_columns + c.VariantBits·variant_bits + c.PosRa0·pos_ra_0 + c.PosRa0·D.PosZero(0) + c.PosRa1·pos_ra_1 + c.PosRa1·D.PosZero(1) + c.ShouldBranch·should_branch + c.Inc·inc`. Adding `k_u` removes the constant of the two affine claims, so the sum is linear in `Bits`.
- **Outputs.** `columns: Vec<_>` with ids committed `Column(y)`, `y < 256`, each at `r_6`: the vector `C`.
- **Output expression.** `Σ_{y<256} D.ColumnWeight(y)·columns[y]`.
- **Derived.** `PosZero(d) = eq(p_d, 0)` (instance). `ColumnWeight(y) = eq(r_1, r_6)·L_{r_1}(y) + eq(r_3, r_6)·L_{r_3}(y) + eq(r_5, r_6)·L_{r_5}(y)`, with

  ```text
  L_{r_1}(y) = c.DirectColumns · l_DirectColumns(y)
  L_{r_3}(y) = c.VariantBits · l_VariantBits(y) + c.PosRa0 · l_PosRa0(y) + c.PosRa1 · l_PosRa1(y) + c.ShouldBranch · l_ShouldBranch(y)
  L_{r_5}(y) = c.Inc · l_Inc(y)
  ```

  The vectors `l_u` are built once in the constructor, under 1,000 multiplications. Each `ColumnWeight(y)` then costs `3t + 6`.

#### 9. Claim flow

The table inverts the `from` entries of §8: it lists every wire value by the batch that sends it, in the order of the wire, with the one relation that consumes it. The claim-flow test regenerates it from the claim structs.

| Batch | Wire values | Point | Consumed by |
|---|---|---|---|
| 1 | `az, bz, cz` of `SpartanOuterF2` | `ρ_2 ++ r_1` | `SpartanInner` |
| | `az, bz, cz` of `SpartanOuterF128` | `ρ_F ++ r_1` | `SpartanInner` |
| 2 | `witness_routed` | `w ++ r_1` | `RouterShort` |
| | `direct_columns` | `w ++ r_1` | `BitsReduction` |
| 3a | `variant, shift, memory, compare, branch` | `x` restricted to the router | the `RouterCycle` relation of that router |
| 3b | `rs1_value, rs2_value, rd_pre_value` | `r_bit ++ r_3` | `RegistersReadChecking` |
| | `imm, fall_through_pc, pc_plus_imm, pc, next_pc` | `r_bit ++ r_3` | `BytecodeReadAddress` |
| | `variant_bits` | `x[0..10) ++ r_3` | `BitsReduction` |
| | `variant` | `q_V ++ r_3` | `BytecodeReadAddress` |
| | `shift_kind` | `q_S ++ r_3` | `BytecodeReadAddress` |
| | `pos_ra_0`, `pos_ra_1` | `p_0 ++ r_3`, `p_1 ++ r_3` | `BitsReduction` |
| | `ram_read_value` | `r_bit ++ r_3` | `RamReadChecking` |
| | `access_kind`, `key_kind` | `q_M ++ r_3`, `q_C ++ r_3` | `BytecodeReadAddress` |
| | `branch` | `r_3` | `BytecodeReadAddress` |
| | `should_branch` | `r_3` | `BitsReduction` |
| 4 | `rs1_ra, rs2_ra, rd_wa` | `a_reg ++ r_4` | `BytecodeReadAddress` |
| | `registers_val` | `a_reg ++ r_bit ++ r_4` | `RegistersValEvaluation` |
| | `ram_ra` | `a_ram ++ r_4` | `RamRaProduct` |
| | `ram_val` | `a_ram ++ r_bit ++ r_4` | `RamValEvaluation` |
| | `ram_val_final` | `a_ram ++ r_bit` | `RamValEvaluation` |
| 5 | `rd_wa` | `a_reg ++ r_5` | `BytecodeReadAddress` |
| | `store` | `r_5` | `BytecodeReadAddress` |
| | `inc` | `r_bit ++ r_5` | `BitsReduction` |
| | `ram_ra` | `a_ram ++ r_5` | `RamRaProduct` |
| 6a | `address_claim` | `a_bc` | `BytecodeReadCycle` |
| 6b | `C[0], …, C[255]` | `r_6` | the opening at `(rho, r_6)` |

There are 43 values before `C`. By consumer: `SpartanInner` 6, `RouterShort` 1, the five `RouterCycle` relations 5, `RegistersReadChecking` 3, `RamReadChecking` 1, `RegistersValEvaluation` 1, `RamValEvaluation` 2, `BytecodeReadAddress` 15, `BytecodeReadCycle` 1, `RamRaProduct` 2, `BitsReduction` 6. The six consumed by `BitsReduction` are the committed claims; every other value is virtual and is consumed by the relation that proves it.

**The execution contract.** `specs/rv64i-binary-arithmetisation.md` leaves five obligations to the surrounding protocol, all against the one machine state that precedes the cycle. Each is discharged by members of the schedule; the state they share is the pair of tables `RegistersVal`, `RamVal` of §8.1.

| Obligation | Discharged by |
|---|---|
| Bytecode selection: every row value the constraints read is that of the row at the committed bytecode index | `BytecodeReadAddress` and `BytecodeReadCycle`. The fourteen claims of §8.14 other than `Entry` and `Next` are every function of the fetched row that another relation reads; batch 6b reduces them to the `d_b` committed chunks, which it reads from `C`. That the fetched row is valid is forced by the constraint on column 16 (§8.14) |
| Register reads: `Rs1Value` and `Rs2Value` are the contents, before the cycle, of the registers the row names, `x0` reading zero | The `Rs1` and `Rs2` terms of `RegistersReadChecking`, whose selector claims `rs1_ra`, `rs2_ra` are among the fourteen; `RegistersValEvaluation`, which makes `RegistersVal` the sum of the writes of strictly earlier cycles from a zero state. Register 0 holds zero by §8.1 |
| Successor: `NextPC` is the PC of the next cycle, or a `final_pc` that `Bytecode::final_pc_index` accepts | The `Next` claim of §8.14 and statement check 7 |
| Register write: `RdWriteValue = old_rd + (1 + Store)·Inc` | The `Rd` term of `RegistersReadChecking`, which makes `RdPreValue` the content of the row's `rd`; the expansion of `RdWriteValue` in the routers (§8.4); `RegistersValEvaluation`, which applies the same increment at `RdWa`; the claims `rd_wa` of both relations and `store`, among the fourteen |
| RAM update: the word at the committed index goes from its content `RamReadValue` to `RamReadValue + Store·Inc` | `RamReadChecking`, which makes `RamReadValue` that content on every cycle; the first identity of `RamValEvaluation`, which applies `Store·Inc` at `RamRa`; `RamRaProduct`, which reduces both `RamRa` claims to the `d_a` committed chunks |

Three bindings go beyond the five and tie the execution to the statement: the `Entry` claim, the output check with the second identity of `RamValEvaluation`, and the initial RAM inside `InitEval`.

#### 10. `C`, the terminal check and the opening

`stages/stage6b/verify.rs` runs these steps, in this order.

1. **Shape.** Reject unless `C` has `BITS_COLUMNS` values.
2. **Projection.** With `chunk` of §3, compute

   ```text
   B_c = chunk(layout.bytecode_ra()[c], a_bc_c;  C)     for c < d_b
   A_c = chunk(layout.ram_ra()[c],      a_ram_c; C)     for c < d_a
   ```

   and build the output aggregate of the batch: `chunks` of `BytecodeReadCycle` is `B`, `chunks` of `RamRaProduct` is `A`, `columns` of `BitsReduction` is `C`. A chunk of `bits_c` bits costs `2^{bits_c} + n_c` multiplications.
3. **Rounds and terminal check.** The generated `verify_clear` verifies the `t` rounds from the input claim `β_1·z_bc + β_2·I_ram + β_3·I_lin`, where `z_bc` is `address_claim` and `I_ram`, `I_lin` are the input expressions of §8.16 and §8.17, and then compares the reduced claim with `Σ_i β_i·expected_output_i`. No member has an idle round. Written out, with the weights of §8.15–§8.17:

   ```text
   W_bc  = h_Router·eq(r_3, r_6) + h_Read·eq(r_4, r_6) + h_Val·eq(r_5, r_6) + h_Entry·eq(0, r_6) + h_Next·next(r_3, r_6)
   W_ram = c.Read·eq(r_4, r_6) + c.Val·eq(r_5, r_6)
   G     = Σ_{y<256} ColumnWeight(y)·C[y]

   q_end = β_1·W_bc·∏_{c<d_b} B_c + β_2·W_ram·∏_{c<d_a} A_c + β_3·G
   ```

   The verifier rejects unless the reduced claim equals `q_end`.
4. **Absorb.** `C[0], …, C[255]` in index order, each under `b"opening_claim"`.
5. **Column point.** `rho = challenge_vector(8)`. The front end absorbs nothing after this draw.
6. **Opening.** `S::verify_opening(setup, state, &BitsOpening { geometry, column_point: &rho, cycle_point: &r_6, columns: &C }, &proof.opening, transcript)`, with `state` the value the commit phase returned (§11).

The value the scheme is asked to open is `v = Σ_{y<256} eq(rho, y)·C[y]`, computed by `BitsOpening::value` and nowhere else. The front end does not absorb `v`: it is a function of `C`, which is absorbed, and of `rho`. Step 4 precedes step 5 on both sides. If some `C[y]` differs from `Bits(y, r_6)`, the difference between `v` and `Bits(rho, r_6)` is a nonzero multilinear polynomial in `rho`, and it vanishes with probability at most `8/2^128`. The chunk values of step 2 are the extensions of the one-hot chunk tables at `r_6` because `chunk` is affine in the column values and `Σ_j eq(r_6, j) = 1`.

#### 11. Commitment seam

The family owns the interface of its commitment scheme. The committed object is a table of `256·2^t` bits, bit `y` of row `j` at index `y + 256·j` (§3), and the interface is two calls on each side, both on the protocol's transcript. The verifier half is `jolt-rv64i-verifier/src/commitment.rs`:

```rust
pub struct BitsGeometry { pub log_T: usize }             // BITS_COLUMNS columns, 2^log_T rows

pub struct BitsOpening<'a> {
    pub geometry: BitsGeometry,
    pub column_point: &'a [F128],                        // rho: 8 coordinates, low variable first
    pub cycle_point: &'a [F128],                         // r_6: log_T coordinates, low variable first
    pub columns: &'a [F128],                             // C: BITS_COLUMNS values, already absorbed
}
impl BitsOpening<'_> {
    pub fn value(&self) -> F128;                         // Σ_y eq(rho, y)·C[y]
}

pub trait BitsWire: Sized {
    fn write(&self, out: &mut Vec<u8>);
    fn read(bytes: &[u8], geometry: BitsGeometry) -> Option<Self>;   // the whole slice, or None
}

pub trait BitsCommitmentScheme {
    type VerifierSetup;
    type Commitment: BitsWire;                           // every prover message of the commit phase
    type VerifierState;
    type OpeningProof: BitsWire;
    type Error: std::error::Error + Send + Sync + 'static;

    fn verify_commit<T: Transcript<Challenge = F128>>(
        setup: &Self::VerifierSetup,
        geometry: BitsGeometry,
        commitment: &Self::Commitment,
        transcript: &mut T,
    ) -> Result<Self::VerifierState, Self::Error>;

    fn verify_opening<T: Transcript<Challenge = F128>>(
        setup: &Self::VerifierSetup,
        state: Self::VerifierState,
        opening: &BitsOpening<'_>,
        proof: &Self::OpeningProof,
        transcript: &mut T,
    ) -> Result<(), Self::Error>;
}

pub fn squeeze_bytes<T: Transcript<Challenge = F128>>(transcript: &mut T, out: &mut [u8]);
```

The prover half is `jolt-rv64i-prover/src/commitment/mod.rs`:

```rust
pub trait BitsCommitmentProver: BitsCommitmentScheme {
    type ProverSetup;
    type ProverState;

    fn commit<T: Transcript<Challenge = F128>>(
        setup: &Self::ProverSetup,
        geometry: BitsGeometry,
        bits: &Arc<[BitsRow]>,
        transcript: &mut T,
    ) -> Result<(Self::Commitment, Self::ProverState), Self::Error>;

    fn open<T: Transcript<Challenge = F128>>(
        setup: &Self::ProverSetup,
        state: Self::ProverState,
        opening: &BitsOpening<'_>,
        transcript: &mut T,
    ) -> Result<Self::OpeningProof, Self::Error>;
}
```

The contract has four parts.

1. **Commit phase.** `commit` and `verify_commit` are step 6 of the preamble, its last. Inside them the scheme absorbs its commitment and may exchange any further messages and draw any challenges; every prover message of the phase is part of `Commitment`. No challenge of the front end is drawn before the phase returns, so every one of them depends on the whole phase, and every challenge of the phase depends on the statement, the program digest and `final_pc`. The two retained states pass to the opening call and to nothing else. `commit` receives the rows by `Arc`, so a retained state shares them and copies nothing.
2. **Opening.** When the opening is called the front end has absorbed all of `C` and drawn `rho`, and it absorbs nothing afterwards. The call receives the point `(rho, r_6)` and the whole of `C` with the geometry, not one value. A scheme that needs partial evaluations of the table derives them from `C`: the 64 values `s_i = Σ_{h<4} eq(rho[6..8), h)·C[i + 64h]`, `i < 64`, are the evaluations at `r_6` of the table folded over its two high column variables. Values derived in this way, a change to another field, the absorption of the point, of `value()` and of derived values, and the scheme's challenges all belong to the scheme. None of them appears in a relation, in a stage or in the proof outside `S::Commitment` and `S::OpeningProof`. The scheme absorbs under labels of its own, distinct from the nine of the front end (§13).
3. **A table of bits.** If `verify_commit` returns a state, then except with the soundness error the scheme states, one table `B ∈ {0, 1}^{256·2^t}` exists such that `verify_opening` on that state accepts only when the multilinear extension of `B`, in the index order of §3, takes the value `opening.value()` at `(rho, r_6)`. The scheme authenticates a table of bits, not a multilinear polynomial with coefficients in `F128`. This is a property of the scheme's verifier: the type of the argument of `commit` restricts an honest prover only and does not provide it. Each scheme documents how it meets this part.
4. **Challenges in a larger field.** `Transcript` returns 16-byte challenges and has no squeeze of raw bytes. A scheme that needs `n` bytes of challenge, for an element of a field with an `n`-byte encoding or for anything else, calls `squeeze_bytes`, the one owner of this rule: it makes `⌈n/16⌉` consecutive `challenge()` draws, concatenates their `to_bytes_le` encodings in the order drawn and keeps the first `n` bytes. The scheme reads the bytes with its own field's decoder. An element of a 24-byte field is two draws with the last 8 bytes dropped.

**`TransparentBits`** (`jolt-rv64i-prover::commitment::transparent`, feature `test-utils`) is the stand-in the tests use, and implements both traits. Its setups are `()`. Its commit phase absorbs a 32-byte digest of the table (§13) under `b"bits_commitment"` and does nothing else; `Commitment` is that digest and both states hold it. Its opening proof is the table, as rows of four `u64`. `verify_opening` recomputes the digest and compares it, evaluates `Σ_{y, j} eq(rho, y)·eq(r_6, j)·B[y, j]` and compares it with `opening.value()`; it absorbs nothing and draws nothing. It meets part 3 by representation: a proof is packed words, so every table it can present is a table of bits, and collision resistance of the digest makes the table unique. It is not succinct: the proof has `32·T` bytes and verification is linear in `T`. It admits `log_T ≤ 20` and returns a typed error above that. Its module documentation says that it exists so that the protocol can be proved and verified end to end before a production scheme is attached, and that it is not for use outside tests.

#### 12. Witness, stage files and the reference tier

`Rv64iWitness` is the witness plane's data, and `Rv64iPlane` selects `&Rv64iWitness` as `WitnessPlane::Ref`:

```rust
pub struct Rv64iWitness {
    pub layout: Layout,
    pub bytecode: Arc<Bytecode>,
    pub bits: Arc<[BitsRow]>,          // T rows
    pub words: Arc<[CycleWords]>,      // T entries: the five base words of §8.1
    pub initial_ram: Vec<(u64, u64)>,  // the canonical initial RAM of §7
    pub final_ram: Vec<u64>,           // 2^a words: the RAM after the last cycle
    pub final_pc: u64,
}

pub struct CycleWords {
    pub rs1_value: u64,
    pub rs2_value: u64,
    pub rd_pre_value: u64,
    pub ram_read_value: u64,
    pub next_pc: u64,
}
```

The rows and the words are shared slices. `BitsCommitmentProver::commit` receives the `Arc` of the rows, and a kernel that needs the rows after `prepare` has returned, where it is handed no witness, keeps a clone of the same `Arc`; neither copies 32 bytes per cycle. A constructor allocates each shared slice once at its final length and fills it in place: converting a completed `Vec` into an `Arc<[_]>` copies every element into a second allocation, and no constructor does that with a buffer of the size of the trace.

Two constructors establish the contract of §8.1. Both return a typed error on rows they cannot read (a number of rows that is not a power of two, a bytecode index that selects no valid row, a chunk with more than one stored indicator) and on an `initial_ram` that is not canonical in the sense of §7 (indices not strictly increasing, an index at or above `K_ram` of the layout, a zero value); neither normalises its argument. A caller that goes on to `prove` obtains the vector from `CheckedInputs::initial_ram()`, which needs only the statement, the preprocessing, `log_K_ram` and `final_pc`; a batch-local test may write it out:

- `Rv64iWitness::from_bits(layout, bytecode, bits, initial_ram, final_pc)` replays the step of §8.1 over the rows once, in cycle order, and fills `words`. The replayed RAM is a dense vector of `2^a` words, allocated fallibly, and the witness keeps it as `final_ram`: the final state has this one producer, and `RamValFinal` and `check_outputs` read it without a second replay.
- `Rv64iWitness::from_facts(layout, bytecode, facts, initial_ram)` fills the rows with `BitsBuilder::fill`, takes `final_pc` from `next_pc` of the last fact and then does the same replay. It copies no other value word from the facts: `CycleFacts::ram_pre_value` on a cycle without an access is never read as `RamReadValue`. During the replay, before the step of each cycle, it compares the facts with the replayed state: `rs1_value`, `rs2_value` and `rd_pre_value` for each register operand that the fetched row has, and `ram_pre_value` on a cycle whose variant has an access. The first difference is a typed error naming the cycle, the field and the two values. The comparison is what makes a set of facts that passes this constructor one consistent run of the step from the initial state; it costs at most four word comparisons per cycle inside the pass that already reads both sides. It is skipped for an operand the row does not have and for the RAM word of a cycle without an access, where the facts hold zero and the replay holds register 0 and RAM word 0 (§8.1). A `NOOP` row, which is what padding and a normalised no-op decode to, is compared in all three operands: its selectors are 0, the replay holds register 0, and a fact other than zero is an error. An adapter therefore writes zero for every operand a cycle does not have.

`Rv64iWitness::check_outputs(&self, checked: &CheckedInputs<'_>)` compares `final_ram` on the I/O mask of §7 with the words of `PublicIoMemory` and returns a typed error naming the first word that differs, with both values. A host calls it between the constructor and `prove` to learn that a run's outputs, panic word or termination word disagree with its statement. `prove` does not call it: the equation that binds the outputs is the output check of §8.12, and a test of that equation needs a proof over a statement that disagrees with the run.

The fields are public. An adapter that derives the words in the pass in which it derives the rows builds the struct directly, and then owes the contract of §8.1 itself. `CycleWords::base_words(store, inc)` returns the `BaseWords` of the arithmetisation for `WitnessRow::compute`, with `rd_write_value = rd_pre_value + (1 + Store)·Inc` as §8.4 states it; it is the one place that forms `RdWriteValue`.

`ProverPreprocessing<S: BitsCommitmentProver>` holds a `VerifierPreprocessing<S>` and the scheme's `S::ProverSetup`. `prove` calls `CheckedInputs::of_statement` (§7) with `log_K_ram` and `final_pc` of the witness, checks that the witness has `2^t` rows, the layout of the checked inputs and an `initial_ram` equal as a vector to `CheckedInputs::initial_ram()`, runs the preamble of §6 with `S::commit`, runs the eight stage drivers in order, draws `rho` and calls `S::open`. It does not check the contract of §8.1. On a witness that violates it, some member's input claim is not the sum of its summand, and the result is an error of the sum-check engine or a proof that the verifier rejects. `Rv64iProverError` wraps `ProverError<F128>` and `Rv64iVerifierError`, and adds the errors of the two constructors and the scheme's error, boxed.

`Rv64iWitness::synthetic(seed, layout, bytecode, memory_layout, initial_ram, t)`, under `test-utils`, serves the batch-local tests. It draws a seeded bytecode index among the valid rows for each cycle, writes well-formed chunk indicators and seeded values in the other fields, takes the PC of a seeded valid row as `final_pc`, and builds the witness through `from_bits`. Its tables therefore follow §8.1, and it is linked to the bytecode: every `NextPC[j]` is the PC of the row that cycle `j + 1` fetches, `PC[0]` is the PC of the row that cycle 0 fetches, which the test uses as the entry PC, and the last successor is `final_pc`. Its stores are of two kinds. An ordinary store addresses a word outside the I/O mask of §7 or inside the output region. One designated store, the last store of the trace, is exempt from that restriction: it addresses the termination word, and its increment is chosen so that the word holds 1 afterwards; no later cycle stores. The constructor places the designated store itself, on a cycle that fetches a store row, whatever the seeded choice of rows would have been, so the trace has it for every seed; the bytecode a test supplies has a store row, and the constructor returns a typed error when it has none. A statement whose outputs are read from the replayed final RAM then passes the output check. It is not an execution: the rows of the arithmetisation do not hold on it.

`reference/views.rs` is the dense accessor: functions of `&Rv64iWitness` that return `Polynomial<F128>` tables in the index order of §3. PR 0 writes the whole file, and no later PR edits it. It holds the views with more than one consuming batch:

- a column of `Bits`, `Inc` at a bit point, and a chunk at a digit point;
- a base word at a bit point, from `words`, and a bytecode word (`PC`, `Imm`, `FallThroughPC`, `PCPlusImm`) of the fetched row at a bit point;
- the selectors `Variant`, `ShiftKind`, `AccessKind`, `KeyKind` at a kind point, and `Branch` and `Store`;
- a register selector as a table over `(k, j)` and at an address point;
- the state tables `RegistersVal` and `RamVal` at a bit point and `RamRa`, from the replay of §8.1, and `RamValFinal` at a bit point, from `final_ram`.

A view with one consuming batch lives in that batch's reference file and belongs to its PR: a column of the witness, from `WitnessRow::compute`, in `reference/spartan.rs` (PR 1); `VariantBits` and the routers' sources and selectors over their slots in `reference/routers.rs` (PR 2); the bytecode tables `H_t` and `R_t` in `reference/bytecode.rs` (PR 5).

Every relation except `BytecodeReadAddress` is served by `NaiveSumcheckProver` with `BindingOrder::LowToHigh`: its `PrepareKernel` impl builds one table per leaf of the output expression over the member's cube. The table of an opening leaf is the cell's polynomial with its other coordinates fixed at the instance's points; the table of an alias is the table of its source; the table of a derived leaf is the function whose extension at the member's point is the closed form of §8.

| Relation | Cube | Opening tables | Derived tables |
|---|---|---|---|
| `SpartanOuter` | `(i, j)` | `Az[i, j]`, `Bz[i, j]`, `Cz[i, j]` | `eq(tau, (i, j))` |
| `SpartanInner` | `col` | `Z(col, r_1) + [col = 16]` on `Virtual`; `Bits(col − 768, r_1)` on `768 + Direct`; each zero elsewhere | `M[col]`; `[col ∈ {0, 16}]` |
| `RouterShort` | `u` | `Fold_r[u\|_r]`, constant in the idle slots | `Idle_r(u)·W_r[u\|_r]` |
| `RouterCycle` | `j` | each word at `r_bit`; `VariantBits(x[0..10), j)`; each selector at its kind point; `PosRa_d(p_d, j)`; `Bits[should_branch(), j]` | `eq(r_1, j)`; constants for `WordSlot`, `OneSlot` |
| `RegistersReadChecking` | `(k, j)` | `Rs1Ra[k, j]`, `Rs2Ra[k, j]`, `RdWa[k, j]`, `RegistersVal(k, r_bit, j)` | `eq(r_3, j)` |
| `RamReadChecking` | `(k, j)` | `RamRa[k, j]`, `RamVal(k, r_bit, j)` | `eq(r_3, j)` |
| `RamOutputCheck` | `k` | `RamValFinal(k, r_bit)` | `eq(tau, k)`; `IoMask[k]`; `ValIo(k, r_bit)` |
| `RegistersValEvaluation` | `j` | `RdWa(a_reg, j)`, `Store[j]`, `Inc(r_bit, j)` | `lt(j, r_4)` |
| `RamValEvaluation` | `j` | `RamRa(a_ram, j)`, `Store[j]`, `Inc(r_bit, j)` | `lt(j, r_4)` |
| `BytecodeReadCycle` | `j` | `BytecodeRa_c(a_bc_c, j)` | the constant `h_t`; `E_t(j)`, with `E_Next` from `EqPlusOnePolynomial::evals` |
| `RamRaProduct` | `j` | `RamRa_c(a_ram_c, j)` | `eq(r_4, j)`; `eq(r_5, j)` |
| `BitsReduction` | `j` | `Bits[y, j]` for 256 columns | `Σ_u c.u·l_u(y)·eq(t_u, j)` for 256 columns |

`BytecodeReadAddress` has a hand-written dense kernel, `reference/bytecode.rs`, because its output expression is one cell and does not contain its summand: it holds the five pairs of tables `H_t[k]`, `R_t[k]` over the bytecode index, sends the degree-2 round polynomial of `Σ_t H_t·R_t`, and returns the reduced claim as `address_claim`. `crates/jolt-kernels/src/reference/bytecode_read_raf.rs` is the same arrangement.

**Stage files and the ownership split.** `impl_stage_prover!` emits `impl StageAggregates<F> for $batch<F>` and `impl StageProver<F, B> for $batch<F>`. Both traits belong to `jolt-prover` and the batches to `jolt-rv64i-verifier`, so an expansion in `jolt-rv64i-prover` on the batch itself would implement a foreign trait for a foreign type. The member-list callback emits the batch by its bare name, which resolves where the callback is invoked, so each stage file of the prover declares a local type of that name around the verifier's batch:

```rust
// jolt-rv64i-prover/src/stages/stage4.rs
use jolt_rv64i_verifier::stages::stage4::{ /* the relations and the generated aggregates, by bare name */ };

pub struct Stage4Sumchecks<F: JoltField>(pub jolt_rv64i_verifier::stages::stage4::Stage4Sumchecks<F>);

impl<F: JoltField> Deref for Stage4Sumchecks<F> {
    type Target = jolt_rv64i_verifier::stages::stage4::Stage4Sumchecks<F>;
    fn deref(&self) -> &Self::Target { &self.0 }
}

jolt_rv64i_verifier::stage4_sumchecks_members!(impl_stage_prover plane = Rv64iPlane,);
```

The expansion reaches the batch only through `self` and through `batch.<member>`, and both pass through `Deref` to the generated methods and the member fields of the verifier's type. The batches and their member fields are therefore `pub`, which makes the generated aggregates `pub` as well: the derive gives them the visibility of the batch and public fields. The member list, the relations and the construction of the concrete members keep their one owner in the verifier crate; the prover constructs a batch by calling the verifier's constructor and wrapping the result. The skeleton is the compile-time test of this arrangement (Execution). If the wrapper fails for a reason found in implementation, for instance an expansion that names an associated item as `$batch::<F>::…`, which `Deref` does not reach, the implementer reports it and does not copy a member list into the prover crate. This spec asks for no change to the macro.

Each stage file also holds the batch's registry, a struct with `#[derive(KernelSlots)]` and one `Box<dyn PrepareKernel<F, R<F>, Rv64iPlane>>` per member. `Rv64iBackend` is the struct of the eight registries, and `Rv64iBackend::reference()` fills every slot from `reference/`. `optimized/` is the module for the `PrepareKernel` adapters of the kernels of `specs/rv64i-binary-prover-kernels.md`; the skeleton declares it empty, and the PRs of that spec fill it and add the constructor that selects its kernels. `StageAggregates::curate_opening_values` returns the absorbed values of a stage: the default, the canonical order with aliases skipped, is the wire order for seven batches, and the stage of batch 6b passes a `curate` closure that returns the 256 column values. Each stage then moves that vector into its wire struct of §7.

The largest tables are `2^{a+t}` entries for batch 4, `2^17` for the ten tables of batch 3a and `2^t` for the 512 tables of `BitsReduction`. Tests therefore use `t ≤ 10` and a memory layout whose I/O range and image fit `a ≤ 10`.

#### 13. Byte-level encoding

This section fixes every byte that the front end absorbs and every byte of a proof. It adds no value and no order: §6 fixes the order and §7 the values.

**Primitive operations.** Each of the five is exactly one call of `Transcript::append_bytes`. `SpongeTranscript::append_bytes` (`crates/jolt-transcript/src/legacy.rs`) absorbs the byte `0x9B`, the length of the body as 8 bytes little-endian, and the body; "framed bytes" counts all three.

| Notation | Call | Body |
|---|---|---|
| `L(l)` | `append(&Label(l))` | 32 bytes: `l`, then zeros |
| `LC(l, n)` | `append(&LabelWithCount(l, n))` | 32 bytes: `l` and zeros to 24 bytes, then `n` as 8 bytes big-endian |
| `W(v)` | `append(&U64Word(v))` | 32 bytes: 24 zeros, then `v` as 8 bytes big-endian |
| `B(s)` | `append_bytes(s)` | the bytes `s`; one call also when `s` is empty |
| `E(x)` | `append(&x)`, `x: F128` | 16 bytes: `to_bytes_le` reversed |

A challenge is one squeeze of 16 bytes read by `F128::from_bytes_le_reduced`, for `challenge` and for `challenge_scalar` alike, so `to_bytes_le` of a challenge is the squeezed bytes.

**Preamble.** Steps 1–5 of §6 as calls:

```text
1  Transcript::new(b"jolt-rv64i-binary-v0")
2  L("params")     W(t)  W(b)  W(a)  W(LowestAddress)
3  L("statement")  W(entry_pc)
                   W(f) for the 20 fields f of MemoryLayout in declaration order:
                     program_size,
                     max_trusted_advice_size, trusted_advice_start, trusted_advice_end,
                     max_untrusted_advice_size, untrusted_advice_start, untrusted_advice_end,
                     max_input_size, max_output_size,
                     input_start, input_end, output_start, output_end,
                     stack_size, stack_end, heap_size, heap_end,
                     panic, termination, io_end
                   LC("inputs", inputs.len())    B(inputs)
                   LC("outputs", outputs.len())  B(outputs)
                   W(panic as u64)
4  L("program")    B(digest)
5  L("final_pc")   W(final_pc)
```

These are 36 calls under six labels, 5 + 27 + 2 + 2, and `1,412 + inputs.len() + outputs.len()` framed bytes. A length-prefixed field is a count in the label word followed by one `B` call. The two advice vectors are empty by statement check 2 and are not absorbed. Step 6, the commit phase, is the scheme's own (§11).

**A batch.** Steps 3, 4 and 6 of §6 as calls:

```text
input claims   L("sumcheck_claim") E(claim_i) for each member i in order; then one challenge per member
a round        LC("sumcheck_poly", ℓ)  E(c_0) E(c_2) … E(c_ℓ)                then one challenge
wire values    L("opening_claim") E(value) for each wire value in the order of §6 step 6
```

The front end uses nine labels: the six of the preamble and these three. A scheme uses none of them (§11).

**Round messages.** A round polynomial `c_0 + c_1·X + … + c_d·X^d` is carried without its linear coefficient, as the stored coefficients `c_0, c_2, …, c_d`. Its *canonical form* has `ℓ` stored coefficients with `1 ≤ ℓ ≤ degree`, `degree` being that of the batch (§5), and with `c_ℓ` nonzero when `ℓ ≥ 2`. It is the form that `prove_batch` of `jolt-sumcheck` produces, which removes trailing zero coefficients down to a polynomial of degree 1 before it absorbs (`trim_round_polynomial`), and it is the form of a round in memory and in the transcript, where `ℓ` is absorbed in the label word. The engine's verifier bounds `ℓ` above by the degree and below by 1 and accepts trailing zeros, so statement check 8 enforces the canonical form before the engine sees a message. In the proof's bytes a round has fixed width: its canonical coefficients followed by zeros to `degree` elements. The decoder removes trailing zeros down to one coefficient. The two maps are inverse to each other, so a polynomial has one form in the transcript and one in the bytes.

An honest message can be shorter than the degree of its batch. The canonical form applies whenever the leading stored coefficients of a round polynomial vanish, for whatever reason: the engine trims by the coefficients it computed and distinguishes no cause, and neither the encoder nor the verifier enumerates cases. Two common sufficient conditions show that short messages occur on ordinary traces, and they are not a classification. In batch 4 the output check has degree at most 2 in an address round whose variable the I/O mask does not depend on, and the two read-checking members have degree at most 2 in every address round: when `2^v` divides `io_mask_start` and `io_mask_end`, the first `v` rounds of the batch have `ℓ ≤ 2`. With no advice capacity `io_mask_start` is 0 and `io_mask_end` is a power of two, so this is the common case. In batch 3b the `Shift` and `Compare` members are zero on a trace in which no cycle shifts and none has a key kind, and every round then has `ℓ ≤ 4`. Cancellation of a leading coefficient after batching, tables that are constant in the bound variable and a batching coefficient equal to zero shorten a message in the same way. No test asserts a particular `ℓ` for an honest round.

**The preprocessing digest** is `Blake2b<U32>` of the `blake2` crate, with no key, salt or personalisation, of the bytes

```text
b                       8 bytes little-endian
LowestAddress           8 bytes little-endian
2^b rows, in index order, 53 bytes each, from BytecodeRow::column in the order of BytecodeColumn::ALL:
    Valid                                         1 byte
    Variant, PC, Imm, FallThroughPC, PCPlusImm    8 bytes little-endian each
    Rs1Ra, Rs2Ra, RdWa                            4 bytes little-endian each
image.len()             8 bytes little-endian
image entries           16 bytes each: the word index, then the value, 8 bytes little-endian each
```

An invalid row is 53 zero bytes. The nine columns are everything the protocol reads of a row. The image has one representation, because `VerifierPreprocessing::new` admits only strictly increasing word indices and nonzero values (§7).

**The stand-in scheme.** The digest of `TransparentBits` is the same hash of `log_T` as 8 bytes little-endian followed by the rows in cycle order, each as its four `u64` little-endian, word 0 first. Its commit phase is `L("bits_commitment") B(digest)`. `BitsWire` of its commitment is the 32 bytes of the digest, and `BitsWire` of its opening proof is the `32·2^t` bytes of the rows in the same order.

**The proof envelope.** `Rv64iProof::to_bytes` writes

```text
0x00                          the version
log_K_ram                     1 byte
final_pc                      8 bytes little-endian
commitment length             8 bytes little-endian, then the bytes of BitsWire::write
for each batch k in the order 1, 2, 3a, 3b, 4, 5, 6a, 6b:
    rounds_k · degree_k elements, round by round in fixed width
    the wire values of the batch, in the order of §6 step 6
opening length                8 bytes little-endian, then the bytes of BitsWire::write
```

An element is the 16 bytes of `to_bytes_le` and is read by `from_bytes_le_checked`. The envelope adds 26 bytes to the elements and the scheme's two byte strings.

`Rv64iProof::from_bytes(bytes, log_T, log_K_bytecode)` rejects a version other than 0. It then checks `1 ≤ log_T ≤ LOG_T_MAX`, `1 ≤ log_K_bytecode ≤ 24` and `5 ≤ log_K_ram ≤ 61` before it computes a round count, so `rounds_k` and `degree_k` of §5 are small numbers and every element section has a known length. It rejects a length prefix larger than the remaining input before it allocates, passes each scheme slice whole to `BitsWire::read`, and rejects any byte left after the opening proof. `BitsWire::read` accepts exactly the strings that `write` produces for the given geometry. A proof therefore has one accepted byte string, and the decoder allocates at most in proportion to its input.

**Decode domains.** `TryFrom` of §4 accepts an index only inside these ranges; every other bit of the index is zero. The domain is syntactic and does not depend on the layout. An id inside it that names no cell, term or challenge of its relation resolves to nothing, which the consumer reports as a missing claim.

| Kind | Relation | Tag | Payload |
|---|---|---|---|
| opening, virtual | below 18 | at most 29 | `RouterFold`: below 5; `BytecodeRaChunk`: below 6; `RamRaChunk`: below 16; otherwise 0 |
| opening, committed | below 18 | at most 6 | `Column`: below 256; otherwise 0 |
| derived | 0, 1 | 0 (`EqTau`) | 0 |
| | 2 | below 2 | 0 |
| | 3 | 0 (`RouteWeight`) | below 5 |
| | 4–8 | below 3 | `WordSlot`: below 8; otherwise 0 |
| | 9, 10 | 0 (`EqCycle`) | 0 |
| | 11 | below 3 | 0 |
| | 12, 13 | below 2 | 0 |
| | 14 | below 2 | 0 |
| | 15 | below 2 | below 5 |
| | 16 | below 2 | 0 |
| | 17 | below 2 | `PosZero`: below 2; `ColumnWeight`: below 256 |
| challenge | 2; 9; 13; 14; 16; 17 | below 6; 3; 2; 16; 2; 6 | none |

### Alternatives Considered

1. **One claim per virtual witness block after `SpartanInner`** (fifteen values in place of `witness_routed`). Rejected: the inner terminal check and the short router sum-check both use the blocks only under the weights `eq(w, ·)`, which the verifier knows, so the fifteen values would be recombined at once under those weights and would prove nothing beyond their combination.
2. **No fold values: one sum-check through the short rounds and the cycle rounds.** The short rounds have degree 2 and the cycle rounds degree 5, a batch has one degree, and the generated flow compares a reduced claim with an expression in wire values. Continuing one sum-check across the two phases needs a driver written by hand. A middle course, one joint value and one merged cycle member of degree 5, saves four elements and the ten aliases; rejected because each router then loses its own claim, which is what its kernel is tested against and what an optimised kernel recovers its round polynomial from.
3. **Five values `R_t(a_bc)` after batch 6a in place of `address_claim`.** The reference kernel would then be the naive one. Rejected: four more elements on the wire for a 30-line kernel.
4. **A window list on `BatchMember`.** §5 shows that the schedule needs none.
5. **Two sum-checks for the two row blocks of stage 1,** 57 rounds and 171 elements; one batch has 30 rounds and 90 elements, with one more batching coefficient. **One relation over all rows** has the same 90 elements and three fewer values; rejected because the rows with values in `F_2` have a kernel of their own and their `Az`, `Bz`, `Cz` tables are tables of bits.
6. **Vector challenges as fields of a `Challenges` struct.** `#[derive(SumcheckChallenges)]` accepts scalar fields only. The points `tau` are stage-level draws; the sixteen bytecode coefficients are sixteen named fields.
7. **The generated output aggregate as the wire format.** An alias is a present cell of the generated struct, so the proof would carry twelve values that the verifier then checks for equality with others, and batch 6b would carry the chunk values next to `C`. The wire structs of §7 carry each value once.
8. **A reduction sum-check over the column variables in place of `C`.** The two product members need the chunk values one by one, so the columns of the chunks would be sent in any case; sending all 256 lets one random point `rho` bind every column, with no further round.
9. **A univariate skip in stage 1.** A non-goal: it changes the first-round message and the domain of the row variables.
10. **`jolt_openings::CommitmentScheme` as the interface of the bit table.** Its `commit` receives no transcript, its `verify` receives one point and one value, and its committed object is a polynomial over the field. A scheme that exchanges messages at commitment time, that needs the 256 column values and not only their combination at `rho`, or whose guarantee is a table of bits has no place to state any of the three. §11 is the family's own interface, and no crate outside the two new ones changes.
11. **`TransparentBits` split between the two crates.** Its verifier half could live in `jolt-rv64i-verifier`. Rejected: it is a test stand-in, and one module behind one feature keeps it out of the verifier crate altogether.
12. **A serde format for the proof's bytes.** `F128`, `CompressedPoly` and `SumcheckProof` implement `Serialize`, and the workspace has `bincode` and `postcard`. Rejected: the bytes would carry a length prefix per round and enum tags whose encoding is a configuration of the library, a round message would have one byte form per length, and a decoder would allocate from those prefixes before any dimension is checked. The envelope of §13 has fixed-width rounds and two length prefixes.
13. **Exactly `degree` coefficients per round in the transcript.** Rejected: `prove_batch` of `jolt-sumcheck` trims trailing zeros before it absorbs, and honest rounds of lower degree occur (§13), so padding in the transcript would need a change to the engine. The canonical form gives the same determinacy, one accepted message per polynomial with its length absorbed, and the padding is confined to the proof's bytes.
14. **A fixture batch declared in the test of the skeleton.** A batch that is local to the test crate needs no wrapper and would not exercise the ownership split of §12. The fixture batch is therefore declared in the verifier crate behind `test-utils` and expanded in the prover crate (Execution).

## Documentation

Crate-level rustdoc for both crates states what the family is (an experiment in RV64I hash-based Jolt over binary fields), the layer each module belongs to and the conventions of §3. `ids.rs` documents the encoding of §4 and the decode domains; `points.rs` the closed forms; each relation's symbolic type its summand; `statement.rs` the eight checks and what each excludes; `transcript.rs` the grammar of the preamble; `proof.rs` the envelope and the canonical form of a round; `commitment.rs` the four parts of the contract of §11; `plane.rs` the witness contract of §8.1 and which constructor establishes it; `commitment/transparent.rs` how it meets the contract and its limits; `reference/mod.rs` that the tier is a test oracle; the fixture batch that it is not a batch of the protocol. No book page: the family is not reachable from the SDK.

## Execution

One PR builds the skeleton; five follow in parallel on disjoint files; one integrates. The series starts from a tree that holds `crates/jolt-rv64i-arith` with the helpers of its `tests/suite/common/`, and the seams of `specs/binary-protocol-family-seams.md`.

The skeleton declares every module of §1 in its `mod.rs` files and creates each module's file with its documentation line, so that a later PR fills files and adds none to a shared list. It contains the whole of `ids.rs`, `points.rs`, `proof.rs`, `transcript.rs`, `statement.rs`, `preprocessing.rs`, `commitment.rs`, both `error.rs`, `plane.rs` and `reference/views.rs`, which the others import and do not edit.

The skeleton is also the compile-time test of the ownership split of §12, and for that it needs one batch. Under `test-utils` the verifier crate declares the fixture batch `ReductionOnlySumchecks` in `stages/fixture.rs`: one member, `BitsReduction`, with the generated `draw_challenges`. The prover crate's `stages/fixture.rs` holds its local wrapper, its registry and its `impl_stage_prover!` invocation. The fixture is not batch 6b and its transcript is not the wire of batch 6b: that batch has three members, three batching coefficients and the draw order of §6. PR 5 writes `stages/stage6b/{mod, verify}.rs` and the prover's `stages/stage6b.rs`, and deletes the two fixture files, their two `mod` lines, the fixture's test and its `[[test]]` entry; the tests of that file that do not use the fixture batch (the stand-in scheme, the two witness constructors) move to `tests/bytecode.rs` or stay under a name of their own. Two more edits touch shared files of the skeleton. Each of PRs 1–5 adds the `[[test]]` entry of its test file, with `required-features = ["test-utils"]`, to the prover crate's `Cargo.toml`; the entries are independent and merge as a union. PR 6 removes the table of round counts and degrees that the skeleton's `proof.rs` holds as literals and has the envelope read the generated schedule of the eight batches, which is then their one owner (invariant 1).

| PR | Files | Acceptance | Smallest end-to-end test |
|---|---|---|---|
| **0. Skeleton** | both `Cargo.toml` and `lib.rs`, every `mod.rs`; verifier `ids.rs`, `points.rs`, `proof.rs`, `transcript.rs`, `statement.rs`, `preprocessing.rs`, `commitment.rs`, `error.rs`, `claims/bits_reduction.rs`, `stages/stage6b/bits_reduction.rs`, `stages/fixture.rs`; prover `plane.rs`, `error.rs`, `commitment/{mod, transparent}.rs`, `optimized/mod.rs`, `reference/{views, bits_reduction}.rs`, `stages/fixture.rs`, `tests/support/mod.rs` (the `#[path]` includes of the arithmetisation's helpers, the counting loop, and the conversion of a run to a statement, a preprocessing and a witness), `tests/{bits_reduction, wire}.rs` | Identifiers; Points; Preprocessing; Preamble vector; the rejections of Statement by `CheckedInputs::new`; the decoder rejections of Proof vector, on a proof of seeded values; Stand-in scheme; Ownership split; Terminal check for `G` and for the value at `rho` | A seeded table of bits at `t = 5` with well-formed chunks and six claims computed from the definitions of §8.17 at seeded points. After the commit phase of `TransparentBits`, the fixture batch is proved through its local wrapper, `C` is absorbed, `rho` is drawn and the scheme opens. The verifier accepts, and rejects one changed `C[y]`, one changed claim, and a prover that draws `rho` before absorbing `C`. The test's documentation says that this is not the wire of batch 6b |
| **1. Stages 1, 2** | `claims/{spartan_outer, spartan_inner}.rs`, `public/matrices.rs`, `stages/{stage1, stage2}/`; prover `reference/spartan.rs`, `stages/{stage1, stage2}.rs`, `tests/spartan.rs` | Placement for batch 1; Expression against definition for the three relations; the `Direct` interval of Routers; the constant `N_M` of Performance equals a literal in the test | The counting loop at `t = 6` through `tests/support`: batches 1 and 2 are proved and verified in sequence, and each of the eight values is rejected when changed |
| **2. Routers** | `jolt-rv64i-arith/src/decode.rs` (the expansion of `RdWriteValue`); `public/routes.rs`, `claims/{router_short, router_cycle}.rs`, `stages/{stage3a, stage3b}/`; prover `reference/routers.rs`, `stages/{stage3a, stage3b}.rs`, `tests/routers.rs` | Routers; Expression against definition for the six relations; the constant `N_R` of Performance equals a literal in the test | A synthetic witness at `t = 5`: `witness_routed` is computed by direct summation, batches 3a and 3b are proved and verified, and the 18 values equal `Polynomial::evaluate` of their tables |
| **3. Stage 4** | `claims/{registers_read_checking, ram_read_checking, ram_output_check}.rs`, `public/io.rs`, `stages/stage4/`; prover `reference/read_checking.rs`, `stages/stage4.rs`, `tests/read_checking.rs` | Placement for batch 4; Expression against definition for the three relations | A synthetic witness at `t = 4`, `a = 6`, with the statement's outputs read from the replayed final RAM: batch 4 is proved and verified, `a_reg` equals `a_ram[1..6)`, a changed output byte is rejected, and a witness whose `RamReadValue` is zero on the cycles without an access, over an initial RAM with a nonzero word 0, yields no accepted proof of the batch (the prover returns an error or the verifier rejects) |
| **4. Stage 5** | `claims/val_evaluation.rs`, `public/ram_init.rs`, `stages/stage5/`; prover `reference/val_evaluation.rs`, `stages/stage5.rs`, `tests/val_evaluation.rs` | Expression against definition for the two relations | A synthetic witness at `t = 5`: the three input values are evaluated from the replayed state tables at seeded points, batch 5 is proved and verified, and a changed word of the initial RAM is rejected |
| **5. Stage 6a and batch 6b** | `public/bytecode.rs`, `claims/{bytecode_read, ram_ra_product}.rs`, `stages/stage6a/`, `stages/stage6b/{mod, verify, bytecode_read_cycle, ram_ra_product}.rs`; prover `reference/{bytecode, ra_product}.rs`, `stages/{stage6a, stage6b}.rs`, `tests/bytecode.rs`; deletes the fixture | Expression against definition for the three relations; Terminal check in full; the draw order of batch 6b in Schedule | The synthetic witness of §12 at `t = 5`, `b = 6`, with the statement's entry PC the PC of the row of cycle 0: the fifteen input values and the six committed claims are computed from definitions, batches 6a and 6b are proved with the opening, the verifier accepts, and it rejects a changed `final_pc`, a changed entry PC and a changed `C[y]` inside a bytecode chunk |
| **6. Integration** (after 1–5) | `verifier.rs`, the batch geometry of `proof.rs`; prover `backend.rs`, `prover.rs`, `tests/{e2e, tamper, claim_flow, schedule, lifecycle}.rs` | every remaining criterion, among them Order of `C` and `rho` on the full path and the literal of Proof vector | The counting loop at `t = 6` through `prove` and `verify` |

A batch-local test proves its batch on one transcript and verifies it on a twin, with the input values computed in the test. The synthetic witness of §12 is enough for every member whose input claim is a wire value, because such a member is a sum-check of its summand for any tables that follow §8.1. Three members need more, and each need is met as follows.

- The outer relations are zero-checks and need satisfied rows, hence the counting loop in PR 1.
- The output check needs a statement that agrees with the final RAM. The synthetic witness keeps its ordinary stores out of the input words, the panic word and the termination word, sets the termination word by its one designated store (§12), and PR 3 reads the outputs from the replay.
- The input claim of `BytecodeReadAddress` holds two public terms, the entry PC and `final_pc`, and its `Next` claim relates the PCs of consecutive cycles. It is a true sum only on a witness whose `NextPC` is the PC of the row the next cycle fetches, whose cycle 0 is at the statement's entry PC and whose last successor is the proof's `final_pc`. A sequence of independently seeded PCs and successors does not have these properties. The synthetic witness has them because `from_bits` derives `NextPC` from the fetched rows; a run of the interpreter has them as well.

## References

- `specs/rv64i-binary-arithmetisation.md`: the columns of `Bits`, the witness, the rows, the decode table and the bytecode.
- `specs/binary-protocol-family-seams.md`: `ExternalId`, `WitnessPlane`, the exported stage macro, `#[protocol(ids = ..)]`, the reference kernel in characteristic 2.
- `specs/binary-sumcheck.md`: zero extension, member windows and the output scale.
- `specs/binary-field.md`: `F128`.
- `specs/sumcheck-batch-derive.md`, `specs/prover-stage-drivers.md`: the batch derive and the stage drivers.
- `specs/verifier-closure-lints.md`: the lints of the verifier crate.
- `specs/rv64i-binary-prover-kernels.md`: the packed-bit kernels that replace the reference tier.
- `crates/jolt-verifier/src/stages/relations.rs`: `ConcreteSumcheck`.
- `crates/jolt-sumcheck/src/batch.rs`: `BatchMember`.
- `crates/jolt-claims-derive/src/lib.rs`, `crates/jolt-verifier-derive/src/lib.rs`: the claim and batch derives.
- `crates/jolt-poly/src/eq_plus_one.rs` and `crates/jolt-verifier/src/stages/stage3/spartan_shift.rs`: the next-cycle evaluation and its use as a derived term.
- `crates/jolt-kernels/src/reference/{naive.rs, bytecode_read_raf.rs}`: the reference kernels.
- `crates/jolt-prover/src/driver.rs`: `impl_stage_prover!`, `StageProver`.
- `crates/jolt-transcript/src/legacy.rs`: `Transcript`, `SpongeTranscript::append_bytes`, `Label`, `LabelWithCount`, `U64Word`.
- `crates/jolt-sumcheck/src/{prover.rs, verifier.rs}`: `prove_batch` with `trim_round_polynomial`, and `verify_compressed`.
- `crates/jolt-poly/src/compressed_univariate.rs`: `CompressedPoly`.
- `common/src/jolt_device.rs`: `JoltDevice`, `MemoryLayout::try_new`, `get_lowest_address`, `remapped_word_address`.
- `crates/jolt-program/src/preprocess/{public_io.rs, ram.rs}`: `PublicIoMemory::new`, `PublicInitialRam::inputs_only`, `compute_max_ram_k`.
- `crates/jolt-rv64i-arith/tests/suite/common/`: the encoder, the interpreter, the conversion to `CycleFacts` and the replay that the tests include.


