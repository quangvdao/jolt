# Committing GTExp bases (`base/base2/base3`) on the **x4** domain
#
# Goal
Reduce **guest verifier** cycle count by removing expensive per-GTExp-instance work from
`GtExpVerifier::expected_output_claim` while keeping the recursion protocol **fully constrained**
and **sound**.
#
# Context / Current behavior (updated)
We now commit GTExp base powers on the native **x4** element domain and consume those openings in
Stage 1, so the verifier no longer recomputes `base^2/base^3` inside
`GtExpVerifier::expected_output_claim`.
#
# The only remaining “per GTExp instance” public-input folding in Stage 1 is for **digits** (scalar
# bits), which is expected/acceptable: the verifier must evaluate at least one public-input MLE.
#
# The key remaining requirements are:
# - `Base2/Base3` are not “free” committed rows: they must be proven consistent with `Base` via
#   constraints (not verifier recomputation), and
# - `Base` itself is bound to the verifier-derived Dory AST (without requiring per-op base hints).
#
# Motivation
For large traces we see Stage 1 dominated by `expected_output_claim` work. If Hyrax/PCS is already
paying a padded cost (e.g. effective `2^21`) and the dense matrix has headroom, it can be attractive
to **move some verifier-time computation into committed witness polynomials**.
#
# Design requirements (per user)
1. `base`, `base2`, `base3` must all be **committed** and **proven consistent**.
2. These must live over a **smaller element domain**: **x4** (the 4 element/tower variables),
   **not** the packed x11 (step+element) domain.
3. Scalars/digits remain exposed as public inputs for now (we keep that linear folding if needed).
4. Result must be **sound** (no “free variables” / unconstrained witness degrees of freedom).
5. **No backward compatibility required**: we may change proof format and verifier input structure.

# Status (progress so far)
As of this WIP implementation, we have:

- **Committed rows added (x4 domain)**:
  - Added `PolyType::{GtExpBase,GtExpBase2,GtExpBase3}` and treated them as GT-family packed rows
    with arity \(4 + k_{exp}\) (native `u` vars plus the GTExp family-local `c` suffix).
  - Included these rows in prefix packing and dense witness emission.
  - Added two additional committed rows (also x4 domain) for base-power quotient checks:
    - `PolyType::GtExpBaseSquareQuotient` (`Q2`)
    - `PolyType::GtExpBaseCubeQuotient` (`Q3`)

- **Virtual polynomial ID space extended (protocol surface)**:
  - Appended `GtExpTerm::{Base,Base2,Base3}` (do **not** reorder existing terms).
  - Appended `GtExpTerm::{BaseSquareQuotient,BaseCubeQuotient}` for `Q2/Q3`.
  - Added `SumcheckId::GtExpBaseClaimReduction` to namespace the new openings.

- **Stage 2 base claim reduction implemented**:
  - Stage-2 cache-only instance `gt/stage2_base_openings.rs` caches
    `GtExpTerm::{Base,Base2,Base3,BaseSquareQuotient,BaseCubeQuotient}` at the Stage-2 point
    \((u, c_{exp\_tail})\).
  - This instance is now **purely cache-only** (`expected_output_claim = 0`).
  - Binding of the committed `Base` row is enforced by **Stage-2 GT wiring** (the wiring verifier
    consumes the committed `GtExpTerm::Base` claim for `GtConsumer::GtExpBase { .. }` edges).
  - `GtExpPublicInputs` are now **bits-only**; base values only appear as boundary data for true AST
    inputs (see `RecursionVerifierInput.gt_exp_base_inputs: Vec<Option<Fq12>>`).
  - New Stage-2 sumcheck instance `gt/base_power.rs`:
    - enforces correctness of `Base2/Base3` against `Base` using the pointwise quotient relations
      (see below), batched by a transcript scalar \(\beta\).

- **Stage 3 prefix packing consumes the new rows**:
  - Prover/verifier Stage-3 mapping was updated so
    `PolyType::{GtExpBase,GtExpBase2,GtExpBase3,GtExpBaseSquareQuotient,GtExpBaseCubeQuotient}`
    are consumed from `SumcheckId::GtExpBaseClaimReduction`.

- **Stage 1 consumption (perf-critical) implemented**:
  - Stage 1 caches openings for `B/B2/B3` at the Stage-1 point (drop `s`, normalize `c` tail) via
    `gt/stage1_base_openings.rs`, and `GtExpVerifier::expected_output_claim` consumes those openings.
#
# Terminology / domains
Let:
 - `s ∈ {0,1}^7` be the step bits (base-4 digit index)
 - `u ∈ {0,1}^4` be the tower/element bits (16-point encoding of GT elements)
 - `c_gt ∈ {0,1}^{k_gt}` be the GT-local constraint index bits in Stage 1
 - `c_exp ∈ {0,1}^{k_exp}` be the GTExp-family index tail (with split-k dummy replication)
#
# The **current Stage 1** GTExp sumcheck lives over `(s,u,c_gt)` (11 + k_gt rounds).
#
# Proposed redesign overview
We introduce three additional *committed* GTExp-family polynomials, each defined over `(u, c_exp)`
with **only 4 tower variables** (plus the family-local `c` index variables), i.e. constant in `s`:
 - `B(c,u)`   : base
 - `B2(c,u)`  : base²
 - `B3(c,u)`  : base³
#
# These are committed inside the same recursion dense matrix commitment (Hyrax), and opened via the
# opening accumulator like other virtual polynomials.
#
# Then:
 - Stage 1 GTExp constraint can use `B/B2/B3` openings (so the verifier does not have to
   recompute base²/base³ inside `GtExpVerifier::expected_output_claim`).
 - Stage 2 binds the committed `Base` row to the verifier-derived Dory AST via GT wiring, ensuring it
   is not a “free witness” polynomial (and eliminating the need for per-op base hints).
#
# This eliminates the expensive per-instance base-power evaluation inside Stage 1 while keeping the
# statement unchanged: the bases are still the actual Dory AST bases.
#
# Key soundness principle
Committing `B/B2/B3` only helps if they are **fully constrained**:
1. **Binding constraint**: `B` must be bound to the verifier-derived GTExp base for each instance
   (AST), without introducing a per-instance verifier bottleneck.
2. **Algebraic consistency**: `B2` and `B3` must represent the true square/cube of `B` in the tower
   representation (not coordinate-wise squaring in Fq).
#
# If either is missing, the protocol becomes unsound (“there exists some B/B2/B3 making constraints
hold”), which changes what is proven.
#
# Why x4 is possible (and what “careful” means)
`B/B2/B3` are independent of `s`. In the x11 encoding, they are implemented by replication across
the step bits, but that is unnecessary for commitments:
 - The exponentiation constraint only ever needs `B/B2/B3` evaluated at the sampled `u` point.
 - Treating them as x4 polynomials reduces witness size and avoids 7 redundant variables.
#
# The “careful” part is **opening-point normalization**:
 - Stage 1 challenges include `(s,u,c_gt)`.
 - `B/B2/B3` should be opened at `(u, c_{exp\_tail})` only: drop the 7 `s` coordinates and drop the
   dummy low bits of `c_gt` (split-\(k\) normalization).
 - This must match the polynomial’s declared variable order/endianness in the matrix layout.
#
# Proposed concrete constraints
## 1) Stage 1 packed GTExp constraint (unchanged form, new inputs)
The packed constraint uses:
 - witness openings: `rho`, `rho_next`, `quotient` (already committed)
 - digit bits: still derived from `scalar_bits` (public input) at `r_s`
 - base powers: **open** `B/B2/B3` at `(r_u, r_c)` and compute:
   `base_power = w0 + w1*B + w2*B2 + w3*B3` where weights `w0..w3` come from digit MLEs
#
# The only change is: `B/B2/B3` come from accumulator openings instead of verifier recomputation.
#
# ## 2) Base-power correctness (updated approach: bind via wiring, prove the rest)
We lock down the approach to:
1. **Bind `Base` via Stage-2 GT wiring** (AST-derived copy-constraints), so the verifier does **no**
   per-GTExp base evaluation in Stage 1 or Stage 2.
2. Prove `Base2/Base3` correctness **algebraically** via sumcheck, without verifier recomputation.

### 2.1 Bind `Base` via wiring (Stage 2)
For each GTExp instance `i`, the GT wiring backend (`gt/wiring.rs`) already derives an edge that
binds the GTExp base port to the AST producer of that base (see `wiring_plan.rs`, “Base/point binding
edges”):
- If the base is an AST output (e.g. from `GTMul` or another `GTExp`), wiring binds it to that
  proven output.
- If the base is an AST input, wiring binds it to the corresponding boundary GT input value derived
  from the base proof / setup (no non-input hints required).

Concretely, in this refactor `GtExpBaseClaimReduction` (`gt/stage2_base_openings.rs`) becomes purely
cache-only, and `WiringGt*` consumes the committed `GtExpTerm::Base` opening for
`GtConsumer::GtExpBase { instance }` instead of treating GTExp base as a verifier-side constant.

### 2.2 Prove `Base2/Base3` consistency via quotient relations (Stage 2)
We introduce two additional committed x4 polynomials (`Q2/Q3`) and prove the pointwise identities
over the 4-var GT element domain `u` (with the same public `g(u)` as GTMul):
- `B(u)^2 - B2(u) = Q2(u) · g(u)`
- `B2(u)·B(u) - B3(u) = Q3(u) · g(u)`

These are enforced by the dedicated Stage-2 sumcheck instance `GtExpBasePow` (`gt/base_power.rs`),
batched by a transcript scalar \(\beta\):
\[
  (B·B - B2 - Q2·g) + \beta·(B2·B - B3 - Q3·g) = 0.
\]

Rationale:
- The verifier does **not** compute \(\widehat{B^2}, \widehat{B^3}\) from public inputs at all.
- The additional cost is “cheap”: a handful of `Fq` multiplies/adds at the final point (plus the new
  sumcheck instance), rather than expensive tower-basis multiplications per GTExp instance.
#
# ## 3) Binding bases to the verifier-derived AST (wiring)
Even if `B/B2/B3` are internally consistent, we still need `B` to be the *correct* base values.
This must be tied to the verifier’s deterministic AST reconstruction.
#
## Updated plan (no-backcompat): eliminate `NonInputBaseHints` for GTExp bases
We will remove the remaining dependency on `NonInputBaseHints.gt_exp_base_hints` by moving *all* GTExp
base binding into the GT wiring constraints:

- **Stage 1**: consumes committed `Base/Base2/Base3` openings (already done).
- **Stage 2**:
  - `GtExpBaseStage2Openings*` becomes *purely cache-only* (no `B-\hat B` public-input fold).
  - `WiringGt*` consumes the committed `GtExpTerm::Base` opening (stacked) for
    `GtConsumer::GtExpBase { instance }` and checks copy-constraints against the AST producer for that
    base.

This yields the desired “val-init binding” effect for **non-input** bases without the verifier ever
materializing those bases as `Fq12` values. True AST-input GT values remain as boundary data derived
from the base proof / setup (unavoidable).

### Practical consequence
After this change:
- `GtExpPublicInputs` only needs `scalar_bits` (digits); `base` is no longer a per-op public input.
- `NonInputBaseHints` is no longer needed for GTExp bases.
#
# Implementation checklist (updated)
- [x] **A) Extend recursion polynomial ID space**
  - appended `GtExpTerm::{Base,Base2,Base3,BaseSquareQuotient,BaseCubeQuotient}`
- [x] **B) Extend `PolyType` layout** with x4 GTExp base rows and quotient rows
  - `PolyType::{GtExpBase,GtExpBase2,GtExpBase3,GtExpBaseSquareQuotient,GtExpBaseCubeQuotient}`
- [x] **C) Emit x4 rows** in `witness_generation` without step replication
- [x] **D) Stage 2 claim reduction (cache-only rows available at Stage-2 point)**
  - `GtExpBaseClaimReduction` caches openings for `Base/Base2/Base3/Q2/Q3`
- [x] **E) Stage 2 base-power correctness**
  - `GtExpBasePow` enforces `Base2/Base3` consistency via committed `Q2/Q3`
- [x] **F) Stage 3 prefix packing consumption** (layout + Stage-3 mapping updated)
- [x] **G) Stage 1 consumption (perf-critical)**:
  - cache openings for `B/B2/B3` at the **Stage-1** point (drop `s`, normalize `c` tail), and
  - update `GtExpVerifier::expected_output_claim` to use those openings (instead of per-instance folding)

- [x] **H) Move base binding into GT wiring (no-backcompat)**:
  - `WiringGt*` consumes the committed `GtExpTerm::Base` claim for `GtConsumer::GtExpBase { .. }`.
  - The verifier no longer does per-instance `Fq12` base evaluation for non-input bases; only true
    AST inputs are materialized via `gt_exp_base_inputs: Vec<Option<Fq12>>`.

- [x] **I) Remove Stage-2 `B-\hat B` public-input fold**:
  - `GtExpBaseStage2OpeningsVerifier::expected_output_claim` is `0` (cache-only).

- [x] **J) Eliminate `NonInputBaseHints` for GTExp bases**:
  - `NonInputBaseHints` no longer carries GTExp base hints; plan derivation uses wiring + boundary
    bases for true AST inputs instead.

# Soundness argument (proof sketch, updated)
We assume:
- Hyrax/PCS binding: the opening proof soundly binds all opening claims to a single committed dense
  polynomial `CommittedPolynomial::DoryDenseMatrix` (`poly/opening_proof.rs`).
- Sumcheck soundness for each instance (standard Schwartz–Zippel over `Fq`).

Then:
1. From PCS binding, the claimed evaluations of `B/B2/B3` are evaluations of *fixed committed*
   polynomials.
2. From GT wiring constraints, the committed `Base` row is bound to the verifier-derived AST producer
   for each GTExp base (and ultimately to AST inputs from the base proof / setup), except with
   negligible soundness error.
3. From `GtExpBasePow`, `Base2/Base3` are algebraically constrained to be the correct square/cube of
   `Base` (modulo `g`), without verifier recomputation.
4. Once Stage 1 consumes the committed `B/B2/B3` openings, the packed GTExp constraint is checked
   using those bound base powers, so the statement proven is unchanged.

# Remaining linear cost
With only bases committed and digits still public, the verifier remains linear in `#GTExp` due to
digit folding (scalar bits). This is expected and acceptable per user requirements.

# Open questions / risks
1. **Wiring-plan edge coverage for GT**:
   - Fixed: `derive_wiring_plan` now errors if it cannot produce edges for
     - GTExp base binding (`GtConsumer::GtExpBase`),
     - GTMul operands (`GtConsumer::{GtMulLhs,GtMulRhs}`), or
     - pairing RHS (`GtConsumer::PairingBoundaryRhs`).
2. **Direct GT inputs feeding GTMul ports**:
   - Current behavior is intentionally strict: a GTMul operand must have a producer, and an AST
     `Input` only resolves if it is mapped to some `GtProducer::GtExpBase { instance }`.
   - If a real Dory AST ever uses a GT `Input` *only* as a GTMul operand (never as any GTExp base),
     the wiring-plan will now fail fast. To support that case, introduce an explicit “GT input”
     producer/consumer port (and corresponding boundary value list) rather than relying on
     `gt_exp_base_by_value`.
3. **Endianness and variable order** for x4 openings:
   - Ensure opening points for base rows match the encoding of the 16-value GT representation.
4. **Split-k dummy replication**:
   - Base rows use family-local indexing (`k_exp`), while Stage 1/2 use the GT-local suffix (`k_gt`).

# Next steps
- Re-run cycle markers / benchmarking to quantify verifier savings.
- (Optional) Remove the remaining “hint-like” combine-leaf base inputs:
  - Today, combine-leaf GTExp bases are provided as explicit `Fq12` values in
    `gt_exp_base_inputs.push(Some(commitment.0))`.
  - In principle these could be removed too (treat combine leaves like other bound witness bases),
    but it’s a minor follow-up and not required for correctness.
- Extend the same “bind bases via wiring + committed rows” pattern to G1/G2 scalar-mul base points to
  remove the remaining `NonInputBaseHints` usage.

