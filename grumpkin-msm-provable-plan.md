## Provable Grumpkin MSM speedup plan (Hyrax recursion)

### Background + current state (where we are now)

- **Hyrax verification is dominated by two Grumpkin MSMs** inside `HyraxOpeningProof::verify`. The two callsites are in `jolt-core/src/poly/commitment/hyrax.rs:L440-L497` (MSM rows / MSM product).
- We added a **trace-only MSM “precompile”** under `jolt-inlines/grumpkin_msm` that works by **patching `VirtualAdvice` payloads** during tracing:
  - Warning in crate docs: `jolt-inlines/grumpkin_msm/src/lib.rs:L3-L8`.
  - Patch hook computes MSM and injects output words: `jolt-inlines/grumpkin_msm/src/host.rs:L28-L93`.
  - Tracer supports patch hooks: `tracer/src/instruction/inline.rs:L25-L34`, applied at `:L198-L237`.

**Important constraint:** Inline instruction *expansions* are executed as a straight-line list during tracing:

```text
tracer/src/instruction/inline.rs:L235-L237
for instr in inline_sequence.drain(..) { instr.trace(cpu, ...); }
```

This means **branch/jump control-flow inside an inline sequence does not work** (PC changes won’t affect which instruction runs next), so a “single `.insn` = full MSM” is only realistic if the MSM can be expressed as a fixed straight-line, branchless computation (not practical for Pippenger).

Therefore, the **provable path** should be:

- Use `.insn` only for **fixed, branchless primitives** (field mul/reduce, curve add/double, etc.).
- Implement the **MSM algorithm in normal guest Rust** (loops + branches), calling those primitives. This is exactly how SHA-256 gets speedups: high-level streaming API in Rust (`jolt-inlines/sha2/src/sdk.rs:L60-L146`) + one-block compression as `.insn` (`jolt-inlines/sha2/src/sdk.rs:L329-L341`).

### Goal

Replace the current Hyrax guest MSM (slow Rust + arkworks arithmetic) with a provably-correct MSM that:

- Is **fully constrained** by existing instruction constraints (no oracle / no `VirtualAdvice` data dependency).
- Reduces cycles materially in `hyrax_verify_msm_product` and `hyrax_verify_msm_rows`.
- Starts with the known hot case: **Grumpkin MSM with n = 2048** (from profiling).

### Non-goals (for the first iteration)

- Variable-size MSM ABI.
- Endomorphisms / GLV.
- Multi-threading / Rayon.
- Proving-time custom constraints for MSM as a single “native gadget”.

### Architecture overview

We will evolve `jolt-inlines/grumpkin_msm` into **two layers**:

1. **Low-level inlines (provable)**: fixed straight-line `.insn` expansions
   - Base-field arithmetic modulo \(p\): add/sub/mul/sqr/reduce
   - Group operations: Jacobian double, mixed add (Jacobian + affine)
2. **High-level MSM (provable)**: normal Rust code (loops, branching) that uses those inlines
   - Pippenger / bucket method for n=2048, \(w=8\) (matching current Hyrax heuristic)

This mirrors the SHA2 pattern: `.insn` only for fixed blocks, loops in Rust.

### Phase 0 — Make the “trace-only MSM precompile” clearly non-default

**Why:** The current `VirtualAdvice`-patched MSM is *not sound* for proving because there’s no constraint tying the patched advice to `(bases, scalars)`.

Deliverables:

- Split feature flags:
  - `grumpkin-msm-trace-oracle`: enables the current patch-hook MSM (`jolt-inlines/grumpkin_msm/src/host.rs:L28-L93`).
  - `grumpkin-msm-provable`: enables the provable primitives + provable MSM algorithm.
- Ensure Hyrax only uses the trace-oracle path in **trace-only builds** (never in proof generation).

### Phase 1 — Pick explicit, stable memory layouts

**Key decision:** avoid relying on Rust struct layout of `ark_grumpkin::*` in guest memory.

Define `#[repr(C)]` types in `jolt-inlines/grumpkin_msm`:

- **Field element** (base field): `FqLimbs([u64; 4])`
- **Scalar**: `FrLimbs([u64; 4])` (only needed for bit extraction in Rust)
- **Affine point**: `Affine { x: FqLimbs, y: FqLimbs, infinity: u64 }`
- **Jacobian point**: `Jac { x: FqLimbs, y: FqLimbs, z: FqLimbs }`

Conversion points:

- In Hyrax, inputs currently arrive as `&[G::Affine]` and `&[G::ScalarField]` (see `jolt-core/src/poly/commitment/hyrax.rs:L440-L497`). For the provable MSM, add a Grumpkin-specialized path that:
  - Converts bases and scalars once into limb arrays.
  - Runs MSM in limb form.
  - Converts result back into `ark_grumpkin::Projective` once.

Completion criteria:

- A round-trip unit test: `Affine -> limbs -> Affine` matches, and `Fr -> limbs -> Fr` matches (host tests).

### Phase 2 — Implement base-field arithmetic as inlines (provable)

We need fast \( \mathbb{F}_q \) ops for curve arithmetic:

- `fq_add(out, a, b)`
- `fq_sub(out, a, b)`
- `fq_mul(out, a, b)`
- `fq_sqr(out, a)`

Implementation approach:

- Use a 4x64-limb representation in **Montgomery form**.
- Implement `fq_mul` as:
  - 256x256 → 512 multiply (schoolbook with `MUL` + `MULHU`, similar spirit to `jolt-inlines/bigint/src/multiplication/sequence_builder.rs:L89-L146`)
  - Montgomery reduction (FIAT-crypto style, fixed 4 limbs, branchless)
- `fq_add/sub`: limb-wise add/sub + conditional correction by modulus, implemented branchlessly via carry/borrow masks.

Where this lives:

- New module tree under `jolt-inlines/grumpkin_msm/src/fq/` (guest SDK + host sequence builders).
- Each op is a separate `.insn` (funct3 disambiguates op; funct7 reserved for “grumpkin field ops”).

Testing:

- Add a host-only harness like other inlines (pattern: `jolt-inlines/blake2/src/sequence_builder.rs` tests; and `jolt-inlines/bigint/src/multiplication/test_utils.rs`).
- Property tests: random inputs compare to `ark_grumpkin::Fq` arithmetic.

Completion criteria:

- `fq_mul` and `fq_add/sub` pass 1k random tests and are faster (cycle count) than baseline Rust ops when traced.

### Phase 3 — Implement curve operations as inlines (provable)

Implement two critical group ops in Jacobian coords:

- `grumpkin_double_jac(out, in)`
- `grumpkin_add_mixed(out, jac, affine)`

Why mixed-add: Pippenger buckets hold Jacobian points; bases are affine.

Implementation notes:

- Use complete-ish formulas for short Weierstrass (avoid inversions).
- Handle infinity:
  - Jacobian infinity can be represented as `Z=0`.
  - Mixed-add needs to treat `affine.infinity != 0` as no-op.

Testing:

- Compare to arkworks for random points/scalars:
  - `double(P)` and `P + Q` vs arkworks.
  - Edge cases: infinity, adding inverse, etc.

Completion criteria:

- `add_mixed` and `double` pass randomized tests, including edge cases.

### Phase 4 — Implement provable MSM in normal guest Rust (uses the inlines)

At this stage we stop trying to make MSM itself a single `.insn`.

We implement:

- `grumpkin_msm_2048(bases: &[Affine], scalars: &[FrLimbs]) -> Jac`

Algorithm:

- Windowed Pippenger, \(w = 8\), `num_windows = ceil(254 / 8) = 32` (matching the existing heuristic in Hyrax around `jolt-core/src/poly/commitment/hyrax.rs:L309-L326`).
- For each window:
  - Clear `buckets[1..255]` to infinity (Jacobian `Z=0`).
  - For each i in 0..2048:
    - Extract window bits from scalar limbs (normal Rust bit ops).
    - If bucket != 0: `bucket[b] = bucket[b] + bases[i]` using `grumpkin_add_mixed` inline.
  - Compute window sum via running sum from high bucket to low:
    - `running += buckets[j]` (Jac+Jac add; if needed, implement `add_jac` inline too)
    - `window_acc += running`
  - Shift global result by `w` doublings and add window result.

Memory:

- Use a single `buckets: [Jac; 256]` scratch buffer reused per window to keep memory bounded.

Integration:

- In `HyraxOpeningProof::verify`, replace (feature-gated) `try_grumpkin_msm2048(...)` call sites (`jolt-core/src/poly/commitment/hyrax.rs:L406-L489`) with calls to this new provable MSM function under `feature="grumpkin-msm-provable"`.

Completion criteria:

- Recursion trace still verifies correctly (dot product + commitment equality) with the new MSM outputs.
- Cycles in `hyrax_verify_msm_product`/`hyrax_verify_msm_rows` drop materially.

### Phase 5 — Remove or quarantine the oracle MSM path for proofs

Once the provable path lands:

- Keep the trace-oracle `.insn` MSM only under an explicit profiling feature.
- Ensure proof-generation builds do not enable it (e.g., CI guard or feature checks).

Completion criteria:

- Proof generation passes with provable MSM feature enabled.
- No build profile accidentally includes the oracle path.

### Phase 6 — Optimize + specialize (after correctness)

High leverage optimizations:

- **Use mixed-add everywhere** (avoid affine->jac conversions).
- **Exploit sparsity** (identity row commitments) to skip work early (already observed in profiling).
- **Tune window size** for n=2048 on Jolt (might differ from conventional CPU-optimal).
- **Batch normalization** only where needed (avoid `normalize_batch` hot paths if possible).

### Risks / open questions

- **Instruction budget**: field mul/reduction sequences will be long; ensure virtual register allocator capacity is sufficient (see allocator behavior in `tracer/src/utils/virtual_registers.rs:L52-L73`).
- **Correctness pitfalls**: Montgomery constants, carry handling, infinity handling.
- **Generic `G: CurveGroup`**: Hyrax is generic; the optimized path must be safely gated to Grumpkin only (like the current gating in `hyrax.rs:L422-L427`).

### Suggested milestone order (practical)

1. Phase 0 feature split (make oracle path safe-by-default)
2. Phase 1 explicit limb layouts + conversions
3. Phase 2 field add/sub/mul/sqr inlines + tests
4. Phase 3 point double + mixed-add inlines + tests
5. Phase 4 MSM-in-Rust using inlines + Hyrax integration
6. Phase 6 optimizations

