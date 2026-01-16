### BN254 `GT_EXP` inline — security-first design notes (single-exp, no ABI changes)

This document summarizes constraints and proposes a **correctness-first** plan for implementing `BN254_GT_EXP` as a Jolt inline, with careful attention to:
- **Soundness** under untrusted prover control,
- **Register pressure** in the inline expansion model,
- **Where “advice” can (and cannot) help**, given Jolt’s current inline and virtual-instruction machinery.

This is intentionally scoped to **single exponentiation only** (no multi-exp) and assumes **no ABI tweaks** beyond what is already specified in `BN254_GT_INLINES.md`.

---

### Non-negotiables / scope

- **No ABI changes**: `rs1 = exp_ptr`, `rs2 = base_ptr`, `rs3 = out_ptr` as already specified in `BN254_GT_INLINES.md` (see “Inline contract (BN254_GT_EXP)”).
- **Single exponentiation only** (no multi-exp inline).
- **Functional semantics must match arkworks** `Field::pow` output (value-level semantics):
  - exponent bits are interpreted MSB→LSB (BE bit-iterator), but *value output* is the only observable requirement.
  - reference: `ark_ff::Field::pow` (quoted in `BN254_GT_INLINES.md`), and “Exact exponentiation algorithm (`pow`)” section.

---

### Current status (as of 2026-01-16)

- **Implemented a correctness-first `BN254_GT_EXP`** in `jolt-inlines/dory-gt-exp/src/sequence_builder.rs`:
  - `bn254_gt_exp_sequence_builder`: `jolt-inlines/dory-gt-exp/src/sequence_builder.rs` ~`1112:1242`
  - **Algorithm**: fixed 256-bit MSB→LSB schedule with **branchless conditional multiply** using a delta update:
    \[
    acc \leftarrow acc^2;\quad
    \Delta \leftarrow acc\cdot base - acc;\quad
    acc \leftarrow acc + bit \cdot \Delta
    \]
  - **Register plan**: `acc` = 48 regs, plus `FqScratch` (14 regs), `Fq2WorkTight` (24 regs), and 1 `exp_limb` reg.
- **Aliasing**: the inline currently **asserts** `base_ptr != out_ptr` (in-place update is not supported yet).
  - See `bn254_gt_exp_sequence_builder`: `jolt-inlines/dory-gt-exp/src/sequence_builder.rs` ~`1159:1165`.
- **Differential tests** exist for:
  - exp==2 vs `arkworks Fq12::square()`
  - random exp vs `arkworks Fq12::pow()`
  - See `jolt-inlines/dory-gt-exp/src/sequence_builder.rs` ~`1802:1862`.
- **Integrated into the recursion guest verifier (measurement path)**:
  - GT exponentiation in Dory `combine_commitments` calls the inline on guest (`cfg(not(feature="host"))`).
  - See `jolt-core/src/poly/commitment/dory/commitment_scheme.rs` ~`32:113` and ~`347:382`.
- **Measured performance (recursion trace, fibonacci embed)**:
  - `dory_gt_exp_ops`: **1,401,740,725 virtual cycles** for 41 exps ⇒ **~34.2M virtual cycles / exp**
  - Baseline prior to this work (from `BN254_GT_INLINES.md`): ~9.97M virtual cycles / exp, so this version is **slower**.

### Hard constraints imposed by Jolt’s current inline machinery

#### 1) Inline expansions are straight-line execution (no “real” control flow)

The tracer expands an inline into a `Vec<Instruction>` and then **traces every instruction in order**. If the inline has an advice function, `VirtualAdvice` placeholders get filled, but the instruction list length and execution order are fixed.

- Reference: `tracer/src/instruction/inline.rs` L198–L245 (sequential trace of the inline instruction sequence, plus advice filling).

Implication:
- Any “skip work if exponent bit is 0” strategy **cannot rely on branching** inside the inline expansion. You can still compute a bit and *select outputs*, but you cannot skip the internal instruction sequence.

#### 2) Inline virtual register budget is tight

Jolt has:
- 32 architectural RISC-V regs,
- 96 virtual regs total,
- 7 virtual regs reserved for virtual instructions.

So the inline has at most **96 − 7 = 89** usable virtual registers.

- Reference: `common/src/constants.rs` L1–L5.
- Allocation behavior: `tracer/src/utils/virtual_registers.rs` (inline regs come from the non-reserved region).

Implication:
- One `Fq12` element is **48×u64 limbs** (your ABI), so you *can* hold **one** GT element in regs (48 regs) with headroom for temporaries.
- You **cannot** hold **two** GT elements in regs simultaneously (96 regs > 89). Any algorithm that assumes “keep `acc` and `base` fully in registers” is impossible.

#### 3) Every allocated inline virtual register has a real-cost cleanup

All inline-allocated virtual registers are zeroed at `finalize_inline()` time:

- Reference: `tracer/src/utils/inline_helpers.rs` L142–L150.

Implication:
- “Just allocate more regs” has a direct, linear cost in additional instructions at the end of the inline. Keep the register footprint bounded and predictable.

#### 4) Advice exists and is plumbed, but must be made sound by checks

Advice values come from:
- either generic advice instructions used by normal ISA (e.g. `DIV`/`REM`), or
- inline-specific advice functions that fill `VirtualAdvice` instructions.

References:
- `tracer/src/instruction/virtual_advice.rs` (what `VirtualAdvice` does),
- `tracer/src/instruction/inline.rs` L203–L237 (how per-inline advice is applied).

The canonical soundness pattern in-tree is: “advice gives a candidate; guest code checks; on failure spoil proof” (see secp256k1 division).

- Reference pattern: `jolt-inlines/secp256k1/src/sdk.rs` L279–L325 (advice inverse + multiply check + explicit spoil on edge case).

Implication:
- Advice is only useful if we can enforce *cheap enough* verification constraints.
- For GT exponentiation, verification is nontrivial; we must be explicit about what we can check cheaply vs what would effectively re-compute.

#### 5) Inline sequence length must fit `u16` virtual-sequence indexing (hard limit today)

Jolt uses `virtual_sequence_remaining: Option<u16>` to distinguish multiple virtual instructions that share the same base instruction address. The inline assembler back-fills this as:

- `virtual_sequence_remaining = (len - i - 1) as u16`
- Reference: `tracer/src/utils/inline_helpers.rs` ~`127:140`

If an inline sequence length exceeds 65535, this value **wraps**, and bytecode preprocessing can detect non-decreasing inline-sequence indices and panic:

- Reference: `jolt-core/src/zkvm/bytecode/mod.rs` ~`70:88`

**Consequence:** a monolithic “full exponentiation” inline that emits hundreds of thousands/millions of virtual instructions cannot currently be used with the standard proving pipeline. For measurement we worked around this by avoiding guest preprocessing in *trace-only* mode, but this is not a long-term solution for proving.

---

### What makes `GT_EXP` hard (and what we can still do)

#### A) We can’t make exponent-dependent control flow “cheap”

Arkworks’ `pow` does “square always, multiply sometimes”. With no control-flow skipping, a naive bit-by-bit implementation tends to devolve into “square always, multiply always, then select”, which roughly doubles the number of GT multiplies.

This may still be a net win **if** our GT mul/sqr inner implementation is sufficiently fast compared to the current arkworks-on-zkVM baseline.

#### B) The “division trick” does not directly translate at the GT level

For division, advice saves you from implementing division. Verification is cheap because it reduces to multiplication + remainder checks.

For `GT_EXP`, the analogous “advice provides intermediate states” approach only helps if:
- we have a **very cheap** way to check `Fq12` multiplication/squaring, or
- we introduce a new, dedicated **virtual instruction / lookup** that can enforce the multiplication relation more directly (this would be a core change, not just an inline crate change).

So, for the inline-crate-only path, the primary lever is still: **implement faster `Fq12` mul/sqr in the zkVM**, and then build exp out of that.

---

### Recommended plan (incremental, correctness-first)

#### Step 0: Decide which correctness envelope we are proving

We have two plausible envelopes:

- **Envelope 0 (max safety, max generality)**: implement general `Fq12` multiplication/squaring and exponentiation. Works for any `Fq12` input.
- **Envelope 1 (GT-specific, potentially faster)**: assume inputs are in the BN254 GT subgroup (pairing target group), and use cyclotomic-optimized formulas (faster squaring, etc.). This requires an **explicit membership check** (or proven upstream invariant) before using subgroup-only formulas.

Given “security critical”, start with **Envelope 0**, and only later consider Envelope 1 with a clear subgroup-membership argument + test coverage.

#### Step 1: Build a *verified* “square” (and “square-then-mul”) step at the **Fq12 layer**

Goal: the smallest unit we can test thoroughly and compose.

We want a step gadget that transforms an accumulator:

- **Square step**: `acc_next = acc^2`
- **Square-add (square-then-mul) step**: `acc_next = acc^2 * base`

Then exponentiation is composition of these steps.

Why start here:
- Squaring happens at every bit; getting a correct, register-feasible squaring pipeline is foundational.
- It forces us to confront representation/layout issues immediately (Montgomery form, tower flatten order, limb ordering), but on a smaller surface area than full exp.

---

### Sketch: first composable building block (“verified square” / “verified square-add”)

This section sketches what the first building block should look like **inside the inline sequence builder**, keeping the register cap in mind and not relying on control-flow.

#### Inputs / memory layout (already specified)

From `BN254_GT_INLINES.md`:
- `exp_ptr`: 4×u64 limbs (LS limb first) for exponent `e`.
- `base_ptr`: 48×u64 limbs encoding `Fq12` in canonical tower order, Montgomery form.
- `out_ptr`: 48×u64 limbs where the accumulator/result lives.

We will treat `out_ptr` as the accumulator storage throughout exp.

#### IMPORTANT practical constraint: in-place `out_ptr == base_ptr`

The existing ABI allows `out_ptr == base_ptr`. For exponentiation, *supporting this efficiently is extremely difficult* because the algorithm needs to reference `base` repeatedly while mutating the accumulator.

Given “no ABI tweak”, the practical options are:
- **Option A (recommended initially)**: treat `out_ptr == base_ptr` as **unsupported/UB** for the inline, and ensure all call sites use distinct buffers (the safe wrapper in `jolt-inlines/dory-gt-exp/src/sdk.rs` already does this by construction).
- **Option B (later)**: add a slow-path (outside the inline) that copies `base` to a temporary buffer before invoking the inline.

This is an engineering constraint, not a cryptographic one, but it must be stated up-front to avoid accidental misuse.

#### “Verified square” step (compute-and-store, no advice)

At the lowest-risk starting point, “verified” means: we **execute** the squaring deterministically in the inline expansion, and the zkVM proof enforces every instruction.

**Step signature (conceptual):**
- Input: `acc` at `out_ptr`
- Output: overwrite `out_ptr` with `acc^2`

**Register-pressure plan:**
- Load the full accumulator `acc` (48 limbs) into **48 virtual registers**.
  - This is feasible under the 89-reg cap, leaving ~41 regs for temporaries.
- Compute `acc^2` using a reference-correct `Fq12` squaring formula (no subgroup-only shortcuts yet).
- Store the 48 output limbs back to `out_ptr`.
- Keep temporary reg usage bounded and release guards promptly to minimize finalize-zero cost.

**Correctness-first formula choice:**
- Prefer using the standard quadratic-extension squaring identity at the `Fq12 = Fq6[w]/(w^2 - v)` level:
  - Let `x = (c0, c1)` with `c0,c1 ∈ Fq6`.
  - Compute:
    - `t0 = c0^2`
    - `t1 = c1^2`
    - `c0' = t0 + v * t1`
    - `c1' = (c0 + c1)^2 - t0 - t1`
  - Then `x^2 = (c0', c1')`.

This is correct for all `Fq12` elements and uses only:
- `Fq6` add/sub,
- `Fq6` squaring,
- and multiplication by the fixed nonresidue `v` (a structured op).

Later we can swap `Fq6` square with a faster dedicated routine, and/or move to cyclotomic formulas once subgroup membership is established.

#### “Verified square-add” step (square then mul-by-base)

Once “square” is correct, extend to:

- `acc_sq = acc^2`
- `acc_next = acc_sq * base`

**Register-pressure plan:**
- Keep `acc_sq` in registers (48 regs).
- Stream `base` from memory in smaller chunks rather than loading all 48 limbs into regs at once (since `acc_sq + base` would exceed the reg cap).
  - In practice this means designing the `Fq12` multiplication routine to load the required coefficients/limbs of `base` on demand.

This is the first point where the “can’t hold two Fq12s” constraint becomes operationally important.

#### How this composes into exponentiation (fixed schedule)

Given we cannot rely on control flow inside the inline expansion, a safe starting point is:
- Implement a **fixed 256-iteration square-and-multiply schedule** (value-correct for all `e`, including `e=0`), even though arkworks skips leading zeros.

Value correctness note:
- Doing extra leading squarings starting from `acc = 1` does not change the final value; it only changes cost.

Cost note:
- This schedule normally wants conditional multiplies; with straight-line inlines, we may temporarily accept “always-multiply-then-select” *or* accept a constant-time multiply-by-identity path only if we can make that path cheap enough (future optimization).

**Implemented variant (current):** we avoid per-limb selects by using the delta trick:
\[
acc \leftarrow acc^2;\quad
\Delta \leftarrow acc\cdot base - acc;\quad
acc \leftarrow acc + bit \cdot \Delta
\]
This keeps control-flow free but still performs an `Fq12` multiplication every bit (so it’s expensive).

---

### Test plan for Step 1 (must-have before optimization)

Before attempting a full exp:
- Build a host-side differential test harness similar to existing inline tests.
  - Example reference: `jolt-inlines/secp256k1/src/tests.rs` uses `InlineTestHarness`.

For `Fq12` squaring:
- Generate random `Fq12` elements in arkworks (host).
- Serialize to the ABI limb layout (Montgomery form, tower flatten order).
- Run the inline sequence under the tracer harness.
- Compare limb output to arkworks `x.square()` (or `x * x`).

Edge cases:
- `x = 0`, `x = 1`, `x = -1`, sparse limbs, and a few hand-picked values.

Only after the squaring step is correct:
- add multiplication-by-base tests,
- then build up to exponentiation.

**Status:** done in `jolt-inlines/dory-gt-exp/src/sequence_builder.rs` (tests around ~`1802:1862`).

---

### Where advice may become useful later (without changing ABI)

Once we have deterministic `Fq12` mul/sqr:
- Advice may help with **micro-optimizations** (e.g., providing precomputed constants or decompositions) but it will not eliminate the need for the core field arithmetic unless we also add new *virtual instructions* that can enforce large-field relations more directly.

If we decide to pursue an advice-heavy “verify instead of execute” design for exp, we should do it with a very explicit accounting of:
- advice size (witness bloat),
- instruction count saved (trace length reduced),
- and soundness constraints (no “unchecked secp256k1-style” shortcuts for GT exp).

