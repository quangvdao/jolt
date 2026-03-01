# Hachi PCS Integration into Jolt

Tracking document for replacing Dory (pairing-based) with Hachi (lattice-based) PCS in Jolt.

Branch: `lz/integrate-hachi`

---

## Architecture

### Mega-polynomial approach

All main witness polynomials are committed as a single Hachi commitment using `log₂(N)` selector variables. Given N polynomials P_0, ..., P_{N-1}, the mega-polynomial is:

```
P*(x, y) = Σ_i eq(y, i) · P_i(x)    where y ∈ {0,1}^{log₂ N}
```

**Batch opening** reduces to a single Hachi opening: after Stages 1–7 produce claims v_i = P_i(r), the verifier samples ρ ∈ F^{log₂ N} and the combined claim is `P*(r, ρ) = Σ_i eq(ρ, i) · v_i`. One Hachi proof suffices.

This eliminates the need for homomorphic batching (`combine_commitments` / `combine_hints`), which is impossible for Ajtai commitments due to nonlinear gadget decomposition G^{-1}.

### Increment → one-hot conversion

RamInc and RdInc (currently dense i128 polynomials) are converted to one-hot format via offset encoding: `inc + 2^64` maps the signed range to unsigned, then decomposed into `d_inc = ⌈65/8⌉ = 9` chunks of 8-bit one-hot polynomials per increment type (18 new committed RA polynomials total).

The IncClaimReduction sumcheck is replaced by a fused RAF-style evaluation sumcheck:
- **Stage 6 (cycle phase):** produces pushforward polynomials G_inc_l(k) for each chunk
- **Stage 7 (address phase):** evaluates `Σ_k G_inc_l(k) · unmap_inc_l(k)` fused with HammingWeight

This eliminates the `lagrange_factor` workaround in Stage 8.

### Advice handling (deferred)

Advice polynomials (TrustedAdvice, UntrustedAdvice) remain on separate Dory commitments for the first pass. They have a different lifecycle (committed outside streaming pipeline) and can be opened via a separate proof. Converting advice to one-hot and joining the mega-commitment is a future workstream.

---

## Status

### Phase 0 — Trait refactoring (DONE)

- [x] `CommitmentScheme` trait: `&self` instance methods, `Config` associated type, `from_proof()`, `config()`, `Default` supertrait
- [x] All implementations updated: Dory, Mock, HyperKZG
- [x] All call sites updated: prover, verifier, witness, SDK macros, benches
- [x] `DoryBatchedProof` wraps `ArkDoryProof` + `DoryLayout`
- [x] `balanced_sigma_nu` extracted as standalone function
- [x] Partial DoryGlobals removal from `commit`, `prove`, `process_chunk`, `aggregate_chunks`
- [x] Merged into `lz/integrate-hachi`, clippy clean

### Phase 1 — Prerequisites

- [ ] Abstract layout out of `OneHotPolynomial`
  - Replace `DoryGlobals::get_num_columns()`, `get_layout()`, `get_T()` with explicit parameters
  - `commit_rows`, `vector_matrix_product`, `num_rows` need layout parameter
  - ~15 DoryGlobals call sites in `one_hot_polynomial.rs`
- [ ] Abstract layout out of `RLCPolynomial`
  - ~11 DoryGlobals call sites
  - Hachi path won't use RLCPolynomial (no homomorphic RLC), but need clean separation
- [ ] Add `streaming_layout()` to `StreamingCommitmentScheme` trait
  - Returns chunk size, alignment, num_chunks for a given polynomial length
  - Replaces `DoryGlobals::get_num_columns()` in `prover.rs:604`
- [ ] Move `dory_layout` from `JoltProof` into `PCS::Config`
  - Already `type Config = DoryLayout` for Dory
  - Makes proof serialization PCS-agnostic
- [ ] Remove DoryGlobals from prover.rs streaming loop
  - `DoryGlobals::initialize_context` → PCS config
  - `DoryLayout::AddressMajor` branching → PCS-level decision

### Phase 2 — Hachi streaming commitment

- [ ] Implement `StreamingCommitmentScheme` for `HachiCommitmentScheme`
  - `ChunkState`: `(partial_u, s, t_hat, ring_coeffs)` — partial outer Ajtai contribution + hint material
  - `process_chunk`: field → ring packing + inner Ajtai per block
  - `process_chunk_onehot`: sparse ring construction + `inner_ajtai_onehot`
  - `aggregate_chunks`: sum partial_u vectors + assemble `HachiCommitmentHint`
- [ ] Small-scalar path for `process_chunk` (generic over SmallScalar or upcast)
- [ ] Implement `JoltToHachiTranscript` adapter (mirror `JoltToDoryTranscript`)
- [ ] Create `jolt-core/src/poly/commitment/hachi/` module
  - `mod.rs`, `commitment_scheme.rs` implementing Jolt's `CommitmentScheme` trait
  - Delegates to `hachi-pcs` crate

### Phase 3 — Increment → one-hot

- [ ] Design `UnmapIncPolynomial` family (scaled `IdentityPolynomial` with offset)
- [ ] Modify witness generation to produce one-hot Inc polynomials
  - `RamInc` → 9 `RamIncRa(d)` one-hot polynomials
  - `RdInc` → 9 `RdIncRa(d)` one-hot polynomials
  - Offset encoding: `inc + 2^64` → unsigned → 8-bit chunks
- [ ] Modify Stage 6 `IncClaimReduction` to produce pushforward G_inc_l(k)
- [ ] Extend Stage 7 `HammingWeight` fusion to include Inc RA polynomials (+18 G polys)
- [ ] Remove `lagrange_factor` workaround in Stage 8 (`prover.rs:1392-1395`, `verifier.rs:582-585`)

### Phase 4 — Batch opening redesign

- [ ] Design mega-polynomial coefficient layout
  - Group by size class (dense T, one-hot K·T)
  - Zero-pad shorter polynomials
  - Determine selector variable ordering
- [ ] Implement `HachiCommitmentScheme::batch_prove` / `batch_verify`
  - Selector sumcheck (~6 rounds)
  - Phase 0 inner evaluation (α = log₂ D rounds)
  - Standard Hachi opening
- [ ] Adapt `ProverOpeningAccumulator` / `VerifierOpeningAccumulator`
  - Selector challenge sampling
  - Combined claim computation: `Σ eq(ρ, i) · v_i`
- [ ] Remove Dory-specific batch infrastructure from Hachi path
  - `BatchPolynomialSource`, `StreamingBatchSource` (keep for Dory)
  - `RLCPolynomial` streaming RLC (keep for Dory)

### Phase 5 — End-to-end + cleanup

- [ ] Wire up `JoltProof` generic over PCS
- [ ] Run `muldiv` e2e test with Hachi PCS
- [ ] Advice → one-hot conversion (deferred from Phase 3)
- [ ] Full DoryGlobals removal (~79 remaining call sites across 8 files)

---

## Key design parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Ring degree D | 1024 | α = log₂ D = 10 inner evaluation rounds |
| Field | Fp128 (q = 2^128 - 275) | Solinas prime, `JoltFp128` wrapper exists |
| k_chunk (one-hot) | 256 | 4 cycles per ring element (4 × 256 = 1024 = D) |
| d_inc (increment chunks) | 9 | ⌈65/8⌉ for 65-bit signed range |
| Selector variables | ~7 | log₂(~82 polynomials) |
| Extra Stage 8 rounds | ~17 | 7 selector + 10 inner evaluation |

## Key files

| File | Role |
|------|------|
| `jolt-core/src/poly/commitment/commitment_scheme.rs` | `CommitmentScheme` + `StreamingCommitmentScheme` traits |
| `jolt-core/src/poly/commitment/dory/commitment_scheme.rs` | Dory impl (reference for Hachi impl) |
| `jolt-core/src/poly/one_hot_polynomial.rs` | OneHotPolynomial (needs layout abstraction) |
| `jolt-core/src/poly/opening_proof.rs` | Accumulator, batch source, lagrange factors |
| `jolt-core/src/zkvm/prover.rs` | Streaming commit orchestration, Stage 8 |
| `jolt-core/src/zkvm/verifier.rs` | Stage 8 batch verify |
| `jolt-core/src/zkvm/witness.rs` | CommittedPolynomial enum, streaming witness gen |
| `jolt-core/src/zkvm/claim_reductions/increments.rs` | IncClaimReduction (to be replaced) |
| `jolt-core/src/zkvm/claim_reductions/advice.rs` | Advice claim reduction (deferred) |
| `jolt-core/src/zkvm/ram/raf_evaluation.rs` | RAF evaluation sumcheck (template for fused Inc) |
| `jolt-core/src/field/fp128.rs` | JoltFp128 wrapper over hachi's Prime128M8M4M1M0 |
| `../hachi/src/protocol/commitment_scheme.rs` | Hachi's HachiCommitmentScheme |
| `../hachi/src/protocol/commitment/commit.rs` | Hachi commit_coeffs, commit_onehot |
| `../hachi/docs/HACHI_FOR_JOLT.md` | Integration design doc |

## Non-goals (explicit)

- Full DoryGlobals removal: deferred, not blocking Hachi integration
- Hachi recursion: out of scope for initial integration
- DA layer / Data Proof migration: separate workstream (see `../hachi/docs/DATA_PROOF_HACHI_MIGRATION.md`)
- Backward compatibility shims: full cutover, no dual-PCS runtime
