# Miller loop recursion: front-alignment + packing refactor plan

## Goal

Make **pairing recursion (MultiMillerLoop)** integrate cleanly with the existing recursion SNARK:

- Recursion proves the **Miller loop(s)** and the internal GT multiplication chain, and the outside verifier performs **only final exponentiation**.
- The recursion proof must remain compatible with **single-point Stage‑3 prefix packing** (one packed PCS opening), i.e. we must not introduce “open the same witness at a second point” as a requirement.
- We want to support **front-aligned** (prefix-active) instances in Stage‑2 batching in a principled way, without per-family hacks.

This document is a design + migration plan. It explains why naive changes break, what invariants we must preserve, and how to roll out the refactor safely.

---

## Background: current recursion structure (relevant pieces)

### Stage‑2 batching is “front-loaded” (suffix-aligned instances)

`BatchedSumcheck` implements a **front-loaded** batching scheme: if an instance has fewer rounds, it is active only in the **last** `num_rounds()` global rounds (dummy rounds occur first).

- **Code**: `jolt-core/src/subprotocols/sumcheck.rs` (`BatchedSumcheck`)
- **Instance control**: `SumcheckInstance{Prover,Verifier}::round_offset(max_num_rounds)`

### Stage‑3 prefix packing assumes suffix-aligned Stage‑2 openings

Stage‑3 derives the “native” low bits of the packed opening point by taking the **suffix** of `r_stage2` and reversing it so those suffix-openings become prefixes in the packed point.

- **Code**: `jolt-core/src/zkvm/recursion/prover.rs` (`prove_stage3_prefix_packing`)
- **Key invariant**: for any native polynomial with `m` vars, its Stage‑2 opening point is the **suffix** length `m` of the global Stage‑2 point (modulo family-specific normalization such as dropping dummy c-bits).

This is why Stage‑3 can reduce many openings to **one** packed PCS opening.

---

## The subtlety: GTMul vs GTExp (why suffix alignment exists)

### What we want

In GT:

- GTExp operates on the packed “x” domain of size \(2^{11}\) (step+elem) plus a c-index suffix.
- GTMul operates on only the **elem/u** domain (size \(2^4\)) plus a c-index suffix.

We do **not** want to pay for GTMul on \(2^{11}\) (no “pad GTMul by 7 step bits”).

### How it works today

Today’s suffix-alignment makes GTMul “small” without padding:

- If Stage‑2 has `max_num_rounds = 11 + k_gt`, then GTMul has `num_rounds = 4 + k_gt`.
- Under suffix alignment, GTMul is active in the **last** `4 + k_gt` rounds.
- Those “u” rounds land on the **last 4 x-bits** of the 11-bit x-domain, so GTMul can be interpreted as a subcube restriction (u-only) of the packed x-domain.

This is embedded in:

- **GTExp Stage‑2 openings**: `jolt-core/src/zkvm/recursion/gt/stage2_openings.rs` (“Store in [x11 low bits, c_gt high bits] …”).
- **GTMul opening normalization**: `jolt-core/src/zkvm/recursion/gt/multiplication.rs` (`normalize_opening_point`) opens at `(u, c_tail)`, explicitly dropping dummy c bits.

### Why naive “global front alignment” breaks GTMul

If we simply make everything prefix-active (offset 0), then GTMul’s 4 u-bits become the **first** 4 rounds of the shared x-window.
Under the current x-bit meaning/order, that forces GTMul to “see” step bits or otherwise changes semantics, unless we pad or redesign x-bit order.

**Takeaway**: any plan to support global prefix-active families must either:

- preserve the cheap GTMul subcube semantics (by choosing an x-bit order where the first 4 rounds correspond exactly to the elem/u bits), or
- accept padding GTMul to 11 (unacceptable), or
- introduce a second opening point (bad for Stage‑3).

---

## The observed failure when adding MultiMillerLoop (MML)

### What MultiMillerLoop looks like

MultiMillerLoop traces are naturally an **11-var** packed domain:

- 7 step bits (128 steps)
- 4 elem bits (16 Fq12 basis evaluations)

Per pairing instance we commit 26 packed 11-var witness polynomials (F, FNext, T coords, inverses, l_val, …).

### Why it failed in Stage‑2 initially

GT wiring consumes producer values (GTExp rho/base, GTMul outputs, etc.) from the Stage‑2 opening accumulator and combines them using its own evaluation point logic.

When we introduced MML, wiring consumed `SumcheckId::MultiMillerLoop` claims, but:

- MML (11 rounds) was suffix-aligned and therefore opened at the **suffix-11** slice of the global Stage‑2 point.
- GT wiring logic assumed “x = first 11 rounds” (prefix-11), so it combined values evaluated at inconsistent points.

This manifested as Stage‑2 sumcheck verification failures in the GT wiring instance.

### Why the “quick fix” (front-align MML) broke Stage‑3 / PCS

We prototyped making MML front-aligned (offset=0) and compensated for dummy-after rounds by scaling its round messages.
This made Stage‑2 pass, but Stage‑3 failed (PCS opening mismatch), because Stage‑3 still assumes that native openings come from the **suffix** of `r_stage2`.

**Takeaway**: with single-point packing, either:

- Stage‑2 point conventions remain global (all families agree), or
- packing must become multi-point (not desired).

---

## NEW (found during implementation): prefix packing cannot mix `x11`-native and `u+c`-native families if we also include `x11`-only MML

When we add MultiMillerLoop as **11-var `x11`** committed traces (step+elem) and also keep GTMul/GTExp-base as **12-var `u4 + c_gt`** polynomials (no step replication),
we introduce an additional structural incompatibility with **single-point prefix packing**:

- Prefix packing assumes: an entry with `m` vars is evaluated at the **first `m` low variables** of the packed opening point \(r_\text{full}\).
- If we include any **11-var** entries (MML), then the first 11 low variables must correspond to the semantics of `x11` (step+elem).
- But GTMul and GTExp-base “small” polynomials are natively over **`u4 + c_gt`** (12 vars), i.e. their evaluation point is **not** `x11` plus one more bit.

This means that, under strict prefix packing, we cannot simultaneously have:

- an 11-var family over `x11 = (step7, elem4)` **and**
- a 12-var family over `(u4, c_gt)` (no step)

without doing one of the following:

- **Pad/replicate the small families across step bits** (turn `u4 + c` into `x11 + c` by making them constant in step), which is a large footprint increase, or
- **Split the commitment** into multiple packed dense polynomials (e.g. one for `x11(+c)` and one for `u+c`) and open both, or
- **Change the packing scheme** away from strict prefix packing (e.g. jagged/multi-point style), or
- **Change MML’s native variables** so it is not an `x11`-only 11-var family (e.g. move the step dimension into a family-local `c`-suffix / family-packing scheme).

This explains why simply “fix Stage‑3 to use prefix instead of suffix” is not sufficient on its own: the layout must also become compatible with the coexistence of 11-var and 12-var native families.

---

## Design constraints for the refactor

We need a design that satisfies:

1. **Single-point packing**: all Stage‑2 openings used in Stage‑3 packing must correspond to a single canonical mapping from `r_stage2` → native points.
2. **No GTMul padding**: GTMul remains \(2^{4+k}\), not \(2^{11+k}\).
3. **Supports prefix-active instances**: we can set `round_offset=0` (or other non-suffix offsets) without breaking batched sumcheck algebra.
4. **Migration feasibility**: the rollout should have intermediate compile-pass steps and a clear “flip” moment where conventions change.

---

## Proposed prong‑2 refactor

This is a two-part change (plus a migration plan).

### Part A: Make `BatchedSumcheck` sound for arbitrary `round_offset`

Today `BatchedSumcheck` assumes dummy rounds come **before** the active window.
If an instance has dummy rounds **after** its active window (e.g. offset=0), the batched protocol is still correct, but only if we apply an additional scaling:

- Let `dummy_after = max_num_rounds - (offset + num_rounds)`.
- During active rounds, treat the instance’s internal `previous_claim` as `previous_claim / 2^{dummy_after}` and scale its univariate message by `2^{dummy_after}`.
- Dummy-after rounds then divide by 2 each time, cancelling the scaling and recovering the correct final output claim.

**Implementation plan**:

- Extend `BatchedSumcheck::{prove,verify}` to compute `dummy_after` per instance and apply this transformation consistently.
- Add unit tests that:
  - run a toy instance with `offset=0` inside a longer batch and verify proof correctness, and
  - ensure existing suffix-aligned instances are unchanged.

This part is “infrastructure only”: it should be safe to land without changing any instance offsets (keep behavior unchanged initially).

### Part B: Redefine the 11‑var x ordering so GTMul can be prefix-aligned

To allow global prefix-active instances without padding GTMul, we need the first 4 x-rounds (under `LowToHigh` binding) to correspond to the **elem/u** bits.

Concretely, change the packed x11 indexing convention from:

- **old**: `index = elem * 128 + step` (elem in high bits, step in low bits)
- **new**: `index = step * 16 + elem` (step in high bits, elem in low bits)

Effect:

- the low 4 bits are elem/u → GTMul’s 4 u-rounds can be prefix-active without touching step bits,
- the remaining 7 bits are step,
- GTExp/MML remain 11-var.

This is the major semantic change and must be applied consistently across:

- GTExp packed witnesses / stage2 openings / wiring helpers that construct padded x11 tables,
- MML packed witness generation (and any helpers that expand 4-var or 7-var MLEs into 11-var replicated tables),
- any other code that assumes the old `x*128+s` layout.

### Part C: Update Stage‑3 prefix packing to the new global convention

Once we change Stage‑2 point semantics (front-alignment and new x-bit meaning), Stage‑3 must stop assuming the native point comes from the **suffix** of `r_stage2`.

We will define a new canonical mapping:

- `r_native` is the **prefix** of `r_stage2` of length `max_native_vars` (or a well-defined transformation if we keep additional family suffixes).
- The packed opening point is formed from `r_native` (in the agreed endianness) + fresh `r_pack` challenges.

This change is global: it affects all families included in packing.

---

## Implementation checklist (file targets)

This is the concrete “what files change” checklist for the prong‑2 refactor.

### A. BatchedSumcheck: support dummy-after scaling for arbitrary offsets

- **Implement**: per-instance `dummy_after = max_num_rounds - (offset + num_rounds)` scaling in both prover and verifier paths.
  - **File**: `jolt-core/src/subprotocols/sumcheck.rs`
  - **Hot spots**:
    - where `individual_claims` are initialized (`mul_pow_2(max_num_rounds - num_rounds)`)
    - the `active` branch that calls `compute_message(...)`
    - the `active` branch that calls `ingest_challenge(...)`
    - the `cache_openings` slice `&r_sumcheck[offset..offset + num_rounds]` (should remain instance-local)

- **Remove/disable any per-instance “dummy-after scaling wrappers” after this lands**.
  - **Why**: once `BatchedSumcheck` supports arbitrary offsets, wrappers that manually scale active-round univariates will double-scale and break correctness.
  - **Concrete example (currently present)**: `FrontAlignedMultiMillerLoopProver` in
    `jolt-core/src/zkvm/recursion/pairing/multi_miller_loop.rs` (as of 2026‑02‑03, around `FrontAlignedMultiMillerLoopProver::compute_message`, ~L1018–L1073).
    - It forces `round_offset() = 0` and scales the univariate by `2^{dummy_after}` during active rounds.
    - This must be removed or made a no-op when Part A is implemented in `BatchedSumcheck`.

- **Add tests**:
  - a toy `SumcheckInstance{Prover,Verifier}` with controllable `round_offset` that is provably correct under both suffix and prefix active windows.
  - **File**: `jolt-core/src/subprotocols/sumcheck.rs` (test module) or `jolt-core/src/subprotocols/sumcheck_*` test area.

### B. x11 layout flip (old `elem*128 + step` → new `step*16 + elem`)

This is the “semantic flip” that changes which x-bits are “u/elem” vs “step” under `BindingOrder::LowToHigh`.

#### GT family

- **GTExp Stage‑1 padding helpers** (replication / packed witness construction)
  - **File**: `jolt-core/src/zkvm/recursion/prover.rs`
  - **Look for**: helper comments like `index = x * 128 + s` and update to `index = s * 16 + x`.

- **GTExp packed witness layout (hot path; avoid runtime reordering)**
  - **File**: `jolt-core/src/zkvm/recursion/gt/types.rs` (`GtExpWitness::from_steps`)
  - **Must-fix concrete sites** (as of 2026‑02‑03):
    - doc comment says `index = x * 128 + s` and describes a “Phase 1 step / Phase 2 elem” bind order (around ~L116–L120)
    - multiple pack loops write `*_packed[x * step_size + s] = ...` (e.g. around ~L165+)
  - **Performance note**: after flipping to `idx = s * 16 + x`, build the packed arrays directly in that order (step-major) rather than generating in old order then permuting.

- **GTExp Stage‑2 claim-reduction / cached openings**
  - **File**: `jolt-core/src/zkvm/recursion/gt/stage2_openings.rs`
  - **Look for**: “Store in [x11 low bits, c_gt high bits] …” and any logic that assumes a particular x11 packing order.

- **GTExp base openings + base-power consistency**
  - **Files**:
    - `jolt-core/src/zkvm/recursion/gt/stage2_base_openings.rs`
    - `jolt-core/src/zkvm/recursion/gt/base_power.rs`

- **GT wiring backend**
  - **File**: `jolt-core/src/zkvm/recursion/gt/wiring.rs`
  - **Look for**:
    - `pad_4var_to_11var_replicated` and `pad_7var_to_11var_replicated` (these bake in x11 indexing)
    - any explicit `STEP_VARS/ELEM_VARS` slicing assumptions
    - any internal comments referring to `x * 128 + s`

- **GT shift check / GT helper types**
  - **Files**:
    - `jolt-core/src/zkvm/recursion/gt/shift.rs`
    - `jolt-core/src/zkvm/recursion/gt/types.rs`
    - `jolt-core/src/zkvm/recursion/gt/exponentiation.rs` (module-level docs currently state `idx = x * 128 + s`)

#### MultiMillerLoop (MML)

MML is also a packed 11-var trace, so its internal packed arrays must match the global x11 order.

- **Witness generator (source of 2048-long packed rows)**
  - **File**: `jolt-core/src/poly/commitment/dory/witness/multi_miller_loop.rs`
  - **Must-fix concrete sites**:
    - Module doc comment currently states `idx = x * 128 + s` (as of 2026‑02‑03, ~L7–L8).
    - `pack_step_and_elem(...)` writes `packed[x * 128 + s]` (as of 2026‑02‑03, ~L36–L46).
    - `pack_step_only(...)` writes `packed[x * 128 + s]` (as of 2026‑02‑03, ~L49–L58).
  - These must flip to `idx = s * 16 + x` in the same atomic change as GT wiring and any GTExp producers/consumers.

- **Recursion MML sumcheck + public/shared MLEs**
  - **File**: `jolt-core/src/zkvm/recursion/pairing/multi_miller_loop.rs`
  - **Look for**:
    - helpers that “expand 4-var to 11-var” and “step replicated across elem” (or vice versa)
    - any MLE evaluation helpers that assume a particular bit order
  - **Must-fix concrete site**:
    - constraint evaluation loops currently compute `idx = x * step_size + s` (as of 2026‑02‑03, around ~L1239), and then read many packed witness/shared arrays at `[...] [idx]`.
    - this must flip to `idx = s * elem_size + x` (and preferably loop `s` outer, `x` inner for cache locality).

- **MML shift-check gadget**
  - **File**: `jolt-core/src/zkvm/recursion/pairing/shift.rs`
  - **Look for**: `expand_step_7_to_11` / `expand_elem_4_to_11` and any assumptions about which bits are low/high.
  - **Must-fix concrete sites** (as of 2026‑02‑03):
    - `expand_step_7_to_11`: uses `base = x * STEP_SIZE` and `evals_11[base..base+STEP_SIZE]` copies (around ~L57–L65)
    - `expand_elem_4_to_11`: uses `base = x * STEP_SIZE` and writes `evals_11[base + s]` (around ~L67–L77)
  - These are “layout definers” for the shift gadget and must match the global x11 convention.

#### GTMul (keep it 4-var)

GTMul must remain 4-var + c, but under the new x11 order those 4 vars must align with the u/elem bits in the shared x-window.

- **File**: `jolt-core/src/zkvm/recursion/gt/multiplication.rs`
  - Ensure `normalize_opening_point` remains correct and corresponds to the new x11 convention.
  - Update tests such as `normalize_opening_point_drops_dummy_c_bits` if they assume old bit positioning.

### C. Stage‑3 prefix packing mapping (global point convention)

- **Prover**: change how `r_native` is derived from `r_stage2` (currently suffix+reverse).
  - **File**: `jolt-core/src/zkvm/recursion/prover.rs` (`prove_stage3_prefix_packing`)

- **Verifier**: mirror the same mapping.
  - **File**: `jolt-core/src/zkvm/recursion/verifier.rs` (`verify_stage3_prefix_packing`)

Note: `PrefixPackingLayout` and `packed_eval_from_claims` in `jolt-core/src/zkvm/recursion/prefix_packing.rs` should not need semantic changes; the critical coupling is the Stage‑2→`r_full_lsb` mapping.

### D. Final cleanup / invariants

- Remove temporary debug hooks once alignment is validated (e.g., “print opening points vs wiring assumed points”).
- Add an explicit “x11 layout” invariant section to `jolt-core/src/zkvm/recursion/spec.md` once the flip lands, to prevent regressions.
- Update any existing recursion documentation that hardcodes “Phase 1 binds step vars” for x11 (this changes under `idx = step * 16 + elem` with `BindingOrder::LowToHigh`).
  - **Concrete sites to update**: `jolt-core/src/zkvm/recursion/spec.md` (multiple “index = x * 128 + s” references) and `jolt-core/src/zkvm/recursion/gt/types.rs` docs.

### E. Performance / implementation notes (after x11 flip)

This refactor is not just “semantic”; it can be implemented efficiently if we build arrays directly in the new order.

- **Avoid runtime permutes**: do not generate packed tables in old `x * 128 + s` order and then transpose; write directly into `packed[s * 16 + x]`.
  - **Targets**: `jolt-core/src/zkvm/recursion/gt/types.rs` (`GtExpWitness::from_steps`), `jolt-core/src/poly/commitment/dory/witness/multi_miller_loop.rs` packers, `jolt-core/src/zkvm/recursion/prover.rs` Stage‑1 padding helper.
- **Prefer step-major contiguous writes** under `idx = s * 16 + x`:
  - For “step row of 16 evals”: use `packed[s*16..s*16+16].copy_from_slice(row)` (one contiguous copy).
  - For “step-constant replicated across x”: use `packed[s*16..s*16+16].fill(v)`.
- **Padding helpers become cheaper**: “replicate 4-var across step” becomes “repeat a 16-vector for each step” (contiguous copies) instead of strided writes.
  - **Target**: `jolt-core/src/zkvm/recursion/gt/wiring.rs` padding helpers.

#### Efficient prover strategy after variable reordering: “spliced dummy rounds” (avoid materializing larger domains)

Even with a global `(elem/u, step, c)` Stage‑2 point, some producers/consumers are fundamentally **lower-arity**:

- GTMul witness rows are naturally `(u4, c_mul)` (size \(2^{4+k}\)), and we do not want to pad them to \(2^{11+k}\).
- Some cache-only openings are also naturally u-only + c-tail.

The key efficiency trick is to make such instances **participate** in the global Stage‑2 challenge stream **without expanding their tables**:

- Treat the “missing variables” (e.g. the 7 step bits) as **dummy rounds** for that instance:
  - `compute_message`: emit a constant univariate with \(H(0)=H(1)=\text{prev}/2\).
  - `ingest_challenge`: do nothing (skip binding) on those rounds.
- When caching openings, **splice** the inner instance-local challenge vector out of the global Stage‑2 challenges:
  - take `u = r_stage2[..4]`
  - take `c = r_stage2[11..11+k_common]` (and then take the family tail bits as usual)

This preserves single-point packing semantics (all Stage‑2 instances still refer to the same *global* `r_stage2`), while keeping small-arity instances small in memory and compute.

**Concrete implementation (current HEAD, 2026‑02‑03)**:

- `jolt-core/src/zkvm/recursion/gt/multiplication.rs`
  - `SplicedGtMulProver`, `SplicedGtMulVerifier`: embed GTMul’s `(u4, c_common)` sumcheck into an `(x11, c_common)` Stage‑2 point by treating step rounds as dummy and splicing challenges for `cache_openings`.
- `jolt-core/src/zkvm/recursion/gt/stage2_base_openings.rs`
  - `GtExpBaseStage2Openings{Prover,Verifier}` were updated to have `num_rounds = 11 + k_common` and to bind only the u-bits + c-tail (skipping step rounds).

This pattern generalizes: if we eventually move to **global prefix-active Stage‑2 instances**, then “splicing” becomes the way to keep *any* smaller-variable family (c-only, u-only, etc.) compatible with a shared global point without materializing huge replicated tables at runtime.

### F. Cross-group (G1/G2/GT) interactions: what changes vs what stays the same

- **Wiring already spans groups today**, but is implemented as *separate* Stage‑2 wiring instances per group.
  - **Where Stage‑2 wiring is appended**: `jolt-core/src/zkvm/recursion/prover.rs` “Wiring/boundary constraints appended LAST in Stage 2” (as of 2026‑02‑03, around ~L1356–L1405).
  - **Plan structure**: `jolt-core/src/zkvm/recursion/wiring_plan.rs` defines `WiringPlan { gt, g1, g2 }` edge lists.
- **The x11 layout flip is GT/MML-internal** (packed 2048-entry tables). It does not inherently change G1/G2 scalar-mul/add constraints, nor the fact that wiring constraints only compare “ports” at a shared Stage‑2 point.
- **What *does* matter globally**: Stage‑2 point convention and Stage‑3 packing mapping are global. If we change prefix/suffix conventions, all groups (GT/G1/G2) that participate in Stage‑3 packing must follow the same Stage‑2→Stage‑3 mapping.

#### What the cross-group interactions actually are (and how they worked before)

There are two distinct kinds of “cross-group” interactions in this codebase:

1. **AST-driven wiring (Stage‑2 wiring instances)**: explicit “copy constraints” between producer/consumer ports.
   - They are **group-local at the sumcheck level** (one wiring sumcheck each for GT, G1, G2), but the *plan derivation* can reference boundary objects shared across groups.
   - The main join point is GT wiring, because it wires:
     - GTExp outputs into GTMul inputs,
     - the GTMul chain into `PairingBoundary::{rhs,miller_rhs}` boundary constants,
     - and optional boundary inputs like `joint_commitment` / GT-valued `AstOp::Input`s.
   - **Code anchors**:
     - plan derivation: `jolt-core/src/zkvm/recursion/wiring_plan.rs` (`GtProducer::{GtExpRho,GtMulResult,MultiMillerLoopOut,...}`, `GtConsumer::{GtMulLhs,GtMulRhs,PairingBoundaryMillerRhs,...}`)
     - backend: `jolt-core/src/zkvm/recursion/gt/wiring.rs`

2. **Pairing recursion itself (MML as a bridge)**: the new semantic bridge between G1/G2 and GT.
   - `ConstraintType::MultiMillerLoop` consumes **G1/G2 points** as public constants (the traced pairing pairs) and produces a GT-valued “miller output” polynomial `f(s,u)` (11-var).
   - That output is then wired into the GTMul chain as `GtProducer::MultiMillerLoopOut { instance }`.
   - Before MML landed, there was **no** internal “G1/G2 → GT” operation proved by recursion; the external pairing check compared GT values directly (and was heavier). With MML, recursion proves the Miller loop and the outside verifier does only final exponentiation.

**Why this matters for the refactor**:

- Stage‑2 wiring compares values evaluated at a shared Stage‑2 point; if variable order / offset conventions change, then every wiring backend that consumes openings must agree on:
  - how to partition the global challenges into `(elem/u, step, c)`,
  - how each family drops dummy c bits / normalizes to its committed row arity,
  - and which producers are step-dependent vs step-independent.

## Migration plan (safe rollout)

### Phase 0: land the plan doc + invariants

- This file.
- Add an “x11 layout” section to `spec.md` if desired.

### Phase 1: infrastructure-only (no behavior change)

- Implement Part A (dummy-after support) in `BatchedSumcheck`.
- Keep all instances using the current suffix offsets.
- Add tests for the new batching behavior using a toy instance.
- After Part A lands, remove any per-instance “front-alignment + scaling wrappers” that were introduced to compensate for missing `dummy_after` support (see `FrontAlignedMultiMillerLoopProver` note above).

### Phase 2: introduce new x11 layout helpers (behind no semantic flip yet)

- Add helper functions to map between old/new x11 index order (or rewrite generators to directly produce the new order).
- Ensure we can compile both old/new producers during migration (temporary duplication is OK).

### Phase 3: atomic semantic flip (must be “big bang”)

In one PR/commit set:

- Switch all GT and MML producers/consumers to the new x11 layout.
- Switch any round offsets that are intended to be prefix-active (if we decide to do that globally).
- Update Stage‑3 prefix packing mapping accordingly.
- Update any verifier logic that reconstructs points/claims.

### Phase 4: cleanups

- Delete compatibility helpers and debug-only glue that is no longer needed.
- Ensure no “second point openings” were introduced.

---

## Testing / validation checklist

1. **Unit tests**:
   - BatchedSumcheck: offset=0 toy instance verifies.
   - GTMul normalize_opening_point remains correct under new x11 ordering.
   - MML constraint + shift gadget round-trip tests still pass.
   - x11 layout sanity:
     - `gt/wiring.rs` padding helpers (`pad_4var_to_11var_replicated`, `pad_7var_to_11var_replicated`) match the global x11 convention.
     - `witness/multi_miller_loop.rs` packers use the same convention as GT wiring and GTExp/MML consumers.

2. **Integration**:
   - `cargo run --release -p recursion -- generate --example fibonacci --recursion` passes end-to-end (Stage‑2 + Stage‑3 + PCS).
   - Cycle tracking: measure `jolt_external_pairing_check` cost before/after.

3. **Debug hooks** (temporary):
   - Keep (and then remove) logs that print consumed opening points vs wiring assumed points, to confirm alignment.

---

## Debugging playbook (fast iteration; avoid full `verify`)

The standalone `verify` subcommand can be expensive at large scales. For fast signal while iterating on
Stage‑2/Stage‑3 alignment, prefer a tiny-scale `generate --recursion` run, which traces + proves and
then immediately attempts recursion verification.

- **Fast smoke test (tiny scale)**:

```bash
RUST_LOG=info cargo run --release -p recursion -- generate \
  --example fibonacci \
  --workdir output/smoke_scale10 \
  --committed \
  --layout address-major \
  --scale 10 \
  --recursion
```

- **Localize Stage‑2 failures to one instance**:
  - `BatchedSumcheck` has a narrow-scoped override to isolate recursion Stage‑2 instances (only when
    `instances ∈ {14,16}` and `max_num_rounds == 19`).
  - Set `JOLT_DEBUG_RECURSION_STAGE2_SINGLE_INDEX=<i>` to zero out all batching coeffs except one.

```bash
JOLT_DEBUG_RECURSION_STAGE2_SINGLE_INDEX=0 RUST_LOG=info cargo run --release -p recursion -- generate \
  --example fibonacci --workdir output/smoke_idx0 --committed --layout address-major --scale 10 --recursion
```

- **Dump per-instance contributions when Stage‑2 fails**:
  - Set `JOLT_DEBUG_SUMCHECK_VERBOSE=1` to print each instance’s `expected_output_claim` and its batching
    coefficient when the batched output claim mismatches.

```bash
JOLT_DEBUG_SUMCHECK_VERBOSE=1 RUST_LOG=error cargo run --release -p recursion -- generate \
  --example fibonacci --workdir output/smoke_verbose --committed --layout address-major --scale 10 --recursion
```

---

## Current implementation status (in-progress notes)

These are “live” notes from the ongoing prong‑2 implementation, to record what has been tried and what
it revealed.

### Status snapshot (2026‑02‑03)

- **Tiny-scale recursion smoke test currently fails** (fast): recursion verification reaches Stage‑2 and fails with
  `"Stage 2 failed: Sumcheck verification failed"`.
  - Using `JOLT_DEBUG_SUMCHECK_VERBOSE=1` shows the mismatch is dominated by the **Stage‑2 wiring instances**
    (`verify_recursion_stage2_{gt,g1,g2}_wiring_total`) when running the full batch.
  - When isolating a “non-wiring” instance with `JOLT_DEBUG_RECURSION_STAGE2_SINGLE_INDEX=0`, Stage‑2 and Stage‑3 can
    succeed, and the failure moves later to **PCS opening** (`"PCS opening failed: Proof verification failed"`).

### Key lesson: you cannot “force offset = 0 for everything” without also reordering other families

While it is tempting to set `round_offset() = 0` for all Stage‑2 instances to make Stage‑3 prefix packing simpler,
this breaks a critical invariant:

- **Stage‑2 wiring instances compare “ports” produced by other Stage‑2 sumchecks**, and those ports must be opened at
  the **same effective Stage‑2 point** (same coordinates for the shared variables they reference).

In today’s recursion Stage‑2, some instances are intentionally **c-only** (e.g. add constraints), while others are
**(step,c)** (scalar-mul / wiring) or **(x11,c)** (GT/MML). If we make everything prefix-active without also changing
variable order, then:

- a c-only instance will accidentally interpret “step” challenges as its “c” challenges, and
- wiring constraints will compare values evaluated at **different points**, making Stage‑2 verification fail even at
  tiny scale.

**Implication for the migration plan**:

- Any “global prefix-active” convention must come with a corresponding **within-family variable reordering** so that
  smaller projections (e.g. c-only) can still be prefix projections of the larger instances they interact with.
- The atomic “big bang” for prong‑2 is therefore larger than “flip GT x11 + adjust Stage‑3 mapping”: it must also
  account for how G1/G2 (step,c) and c-only add instances embed into the common Stage‑2 point.

### Status snapshot (2026‑02‑03, later)

- Implemented the “spliced dummy rounds” embedding for GTMul and GTExp base openings (see above), to ensure u-only producers can coexist with an `(x11,c)`-shaped Stage‑2 challenge stream without padding tables.
- **Result**: isolating the GT wiring instance still fails Stage‑2 verification.
  - Command used:

```bash
JOLT_DEBUG_RECURSION_STAGE2_SINGLE_INDEX=13 RUST_LOG=info cargo run --release -p recursion -- generate \
  --example fibonacci --workdir output/smoke_idx13_splice_fix --committed --layout address-major --scale 10 --recursion
```

  - Outcome: recursion verification reaches Stage‑2 and fails with `"Stage 2 failed: Sumcheck verification failed"`.
  - With `JOLT_DEBUG_SUMCHECK_VERBOSE=1`, the mismatch report shows `inst[13] label=verify_recursion_stage2_gt_wiring_total coeff=1` and all other instances have `coeff=0` (confirming GT wiring is the only contributor under isolation).

**Next investigation steps** (high signal):

- Run the same isolated wiring instance with `JOLT_DEBUG_SUMCHECK_VERBOSE=1` to capture the batched-claim mismatch and confirm the failing instance label is `verify_recursion_stage2_gt_wiring_total`.
- Enable `JOLT_DEBUG_GT_WIRING_POINTS=1` *with* `RUST_LOG=info` so the verifier prints the opening-point lengths it consumes for:
  - `SumcheckId::{GtExpClaimReduction,GtMul,GtExpBaseClaimReduction,MultiMillerLoop}`
  - and compare those lengths/partitions to the `gt/wiring.rs` assumption `(elem, step, c)`.

## Risks / gotchas

- **Proof compatibility**: changing x11 layout and/or stage‑3 mapping changes proof semantics; do not expect old proofs to verify.
- **No explicit recursion-proof version marker today**: `RecursionProof` / `RecursionArtifact` serialization does not self-identify which x11/layout + Stage‑3 mapping convention was used (as of 2026‑02‑03). Consider adding a `proof_version` / `x11_layout_version` field so verifiers can reject incompatible proofs early with a clear error.
- **Mixed conventions are not packable**: partial migration (only GT or only MML) will almost certainly break Stage‑3 unless we temporarily disable packing or introduce multi-point openings (avoid).
- **Endianness pitfalls**: Stage‑3 uses lsb/bit order carefully; document and test any changes to reversal/ordering.

