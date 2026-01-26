## Fused Wiring Backend (Design Notes)

This note explains what the “value source seam” in `gt/wiring.rs` means, and sketches how a **fused** wiring backend could work if we stop caching **per-instance** port openings in Stage 2.

### Context: what wiring currently assumes

Today, the Stage-2 wiring sumchecks (`gt/wiring.rs`, `g1/wiring.rs`, `g2/wiring.rs`) verify copy/boundary constraints by:

- deriving an explicit **edge list** (`WiringPlan`) from the AST (`wiring_plan.rs`)
- sampling per-edge randomizers \(\{\lambda_e\}\)
- evaluating, at the wiring sumcheck’s final point, a randomized linear combination of edge equalities:
  \[
  \sum_{e} \lambda_e \cdot (\text{src}_e - \text{dst}_e) \stackrel{?}{=} 0.
  \]

Crucially, the verifier currently obtains `src_e`/`dst_e` by **reading per-instance openings** from the `VerifierOpeningAccumulator`:

- GT: `VirtualPolynomial::gt_exp_rho(i)` under `SumcheckId::GtExpClaimReduction`, `VirtualPolynomial::gt_mul_{lhs,rhs,result}(j)` under `SumcheckId::GtMul`.
- G1/G2: scalar-mul output endpoint ports under `SumcheckId::{G1ScalarMul,G2ScalarMul}` and add ports under `SumcheckId::{G1Add,G2Add}`.

That is exactly the coupling that breaks if we “fuse all sumchecks of a given type” and stop emitting openings with an `instance` index.

### What the seam means (the refactor you asked about)

In `jolt-core/src/zkvm/recursion/gt/wiring.rs`, I introduced a small internal helper (`LegacyGtWiringValueSource`) that encapsulates:

- how the verifier computes the step-selector factor `Eq(s, s_out)` for a GTExp producer
- how it fetches each producer/consumer value (or boundary constant) at the wiring point

So the wiring polynomial structure (the sumcheck relation) stays the same, but the *mechanism* for producing the `(src, dst, eq_s)` triple is centralized.

This is intentionally a “swap point”: a fused backend would replace `LegacyGtWiringValueSource` with something that does **not** read `VirtualPolynomial::* (instance)` openings.

### Goal for a fused wiring backend

Replace “per-edge lookups of per-instance openings” with a small number of openings that scale like:

- **O(#port poly-types)** (and maybe a few aux openings),
- not O(#edges) and not O(#instances).

The wiring verifier may still do O(#edges) scalar work (loops) — that’s already true — but it should not require O(#edges) accumulator openings.

### Current direction (decisions for the next implementation pass)

These decisions are intended to make the next implementation bite-sized and consistent with existing “fused” precedent in this repo.

- **Scope**: implement **GT-only fused wiring** first (do not attempt G1/G2 fused wiring in the same pass).
- **Style**: make it “like `G1AddFused`”:
  - cache **one opening per fused object** (not per instance),
  - include an explicit \(c\)-index (global `constraint_idx`) in the sumcheck variables,
  - use sparse `Eq(r_c, idx)` weighting for family selection / endpoint selection.
  Reference prototype: `jolt-core/src/zkvm/recursion/g1/fused_addition.rs` (see `FusedG1AddParams::num_rounds` and the
  `Eq(r_c, idx)` computation in `FusedG1AddVerifier`).
- **Where fused openings “live”**: keep them under the **existing family `SumcheckId`s** (no new `SumcheckId` needed).
- **GT global constants without an instance** (`JointCommitment`, `PairingBoundaryRhs`): treat them as anchored to the producer,
  i.e. set `c_dst := c_src` for these endpoints (so the fused \(\sum_c\) check does not pick up an extra scaling factor).

### The core obstacle

Edges connect **different instance indices**:

- an edge refers to a specific source operation instance and a specific destination operation instance
- i.e., values live at indices like `gt_mul_lhs(dst_instance)` and `gt_exp_rho(src_instance)`.

If we only have fused openings like “one scalar per fused polynomial” (e.g. \(P_t(r_c,r_x)\)), we cannot directly “index out” an individual instance’s value.

So a fused wiring backend must *change what is opened* (or change the wiring sumcheck statement) so that the verifier can check all edge equalities without per-instance opens.

### Design direction: fuse wiring **over the instance index** \(c\)

The fused-virtual-polynomial plan introduces an explicit MLE index variable \(c\) over `num_constraints_padded = 2^k`.

#### Revision: \(c\) is the **global constraint index**, not a family-local `instance`
`WiringPlan` edges store *family-local* instance indices (e.g. “the 3rd `GTMul` op” or “the 5th `G1ScalarMul` op”).
In the fused framework, \(c\) ranges over the **global** `constraint_idx` domain used by the recursion matrix row blocks:
`c ∈ [0, num_constraints_padded)`.

So any fused wiring backend needs deterministic maps:

- `gt_mul_instance -> constraint_idx`
- `gt_exp_instance -> constraint_idx`
- `g1_scalar_mul_instance -> constraint_idx`
- `g2_scalar_mul_instance -> constraint_idx`
- `g1_add_instance -> constraint_idx`
- `g2_add_instance -> constraint_idx`

derived by scanning `RecursionVerifierInput.constraint_types` in order.

Reference points in current code:

- `RecursionVerifierInput.constraint_types` is the global ordering (`jolt-core/src/zkvm/recursion/verifier.rs` L93-L115).
- Legacy wiring verifiers already reconstruct instance-ordered base lists by scanning `constraint_types`:
  - `WiringG1Verifier::new` (`jolt-core/src/zkvm/recursion/g1/wiring.rs` L354-L365)
  - `WiringG2Verifier::new` (`jolt-core/src/zkvm/recursion/g2/wiring.rs` L445-L458)

The key wiring idea is to rewrite the randomized edge sum
\[
\sum_{e} \lambda_e (\text{src}_e - \text{dst}_e)
\]
as a **sum over constraint indices** \(c\), where each \(c\) accumulates the contributions of all edges incident to that index:
\[
\sum_{c \in \{0,1\}^k}
\Big(
  \sum_{\text{ports }p} W_p(c)\cdot V_p(c)
\Big),
\]
where:

- \(V_p(c)\) is the (opened) value of port-type \(p\) for constraint index \(c\) at the wiring selector point.
- \(W_p(c)\) is a weight that equals a signed sum of \(\lambda_e\)’s for edges that reference that port at index \(c\).

Then we can check this via a single sumcheck over \(c\) (or via evaluation at a random point \(r_c\) in verifier time), using multilinear extensions of \(W_p\) and \(V_p\).

#### What are “port types”?

For GT wiring, the port types are essentially:

- **producers**: `GtExpOut` (rho at the output step), `GtMulOut` (mul result)
- **consumers**: `GtMulLhs`, `GtMulRhs`
- **boundary**: `JointCommitment`, `PairingBoundaryRhs`, and (depending on policy) `GtExpBase`

For G1/G2 wiring, port types are:

- scalar-mul output endpoint (x,y,ind) / (x0,x1,y0,y1,ind)
- add ports (P/Q/R)
- scalar-mul base (boundary)
- pairing boundary p1/p2/p3 (boundary)

### Selector points: how \(V_p(c)\) is defined

Wiring never needs the full polynomial tables; it only needs fixed “selector-point” evaluations:

- **GT** uses an element-selector \(\tau \in \mathbb{F}^4\) and a step-selector:
  - for GTMul ports: \(u=\tau\)
  - for GTExp output: \(u=\tau\) and \(s = s_{\text{out}}(c)\) (depends on packed step count)
- **G1/G2** uses the fixed last step \(s_{\text{last}}=255\) (8 bits) with no extra element variables.

So each \(V_p\) is a function **only of \(c\)** (after applying the selector), even if the underlying witness is higher-variate.

### How \(V_p(c)\) relates to fused row polynomials \(P_t(c,x)\)

Under the fused-row view (porting plan), for each `PolyType` \(t\) we have a fused polynomial:
\[
P_t(c, x) := \text{(the row-block value for poly type }t\text{ at constraint index }c\text{ and within-row point }x).
\]

Then:

- GTMul `lhs/rhs/result` are \(P_{\text{MulLhs}}(c, u)\), etc. (replicated / independent of step vars).
- Packed GTExp `rho` (actually `RhoPrev`) is \(P_{\text{RhoPrev}}(c, (s,u))\) (11-var).

The selector-point value needed for wiring is:

- \(V_{\text{GtMulLhs}}(c) = P_{\text{MulLhs}}(c, u=\tau)\)
- \(V_{\text{GtExpOut}}(c) = P_{\text{RhoPrev}}(c, s=s_{\text{out}}(c), u=\tau)\)

Similarly for G1/G2 scalar-mul outputs: the endpoint is a fixed step index (255), so it is just a selector into the 8-var trace.

### How to represent the weights \(W_p(c)\)

Given a fixed wiring plan and sampled \(\{\lambda_e\}\), define for each port type \(p\) an array `W_p[c]`:

- for each edge `e = (src -> dst)`:
  - add \(+\lambda_e\) into `W_{src_port}[src_c]`
  - add \(-\lambda_e\) into `W_{dst_port}[dst_c]`

This is a pure function of `(wiring plan, transcript challenges)`. Both prover and verifier can compute it.

We then treat `W_p` as an MLE over \(k\) bits (domain size `num_constraints_padded`).

### Candidate fused wiring check (GT example)

Pick the same transcript-sampled \(\tau\) as today.

Define:

- \(V_{\rho}(c) := P_{\text{RhoPrev}}(c, s=s_{\text{out}}(c), u=\tau)\)
- \(V_{\text{lhs}}(c) := P_{\text{MulLhs}}(c, u=\tau)\)
- \(V_{\text{rhs}}(c) := P_{\text{MulRhs}}(c, u=\tau)\)
- \(V_{\text{res}}(c) := P_{\text{MulResult}}(c, u=\tau)\)

and weights \(W_{\rho}, W_{\text{lhs}}, W_{\text{rhs}}, W_{\text{res}}\).

Then the edge check reduces to:
\[
\sum_{c} \Big(
  W_{\rho}(c) \cdot V_{\rho}(c)
  + W_{\text{res}}(c) \cdot V_{\text{res}}(c)
  + W_{\text{lhs}}(c) \cdot V_{\text{lhs}}(c)
  + W_{\text{rhs}}(c) \cdot V_{\text{rhs}}(c)
\Big) \stackrel{?}{=} 0
\]
(with the sign convention embedded in the \(W_p\)’s).

This can be proven by a sumcheck over \(c\) (k rounds) with degree 2 (products of multilinears), and it requires openings only for:

- the few \(V_p\) polynomials at the final point \(r_c\), and
- (depending on how we implement it) possibly the \(W_p\) polynomials at \(r_c\).

### Two implementation strategies

#### Strategy A: treat \(W_p\) as verifier-computed (no extra openings)

Verifier loop:

- compute \(W_p(r_c)\) by evaluating the MLE of the weights at the random point \(r_c\)
  - this is \(O(2^k)\) if done naively, or \(O(\#\{c : W_p[c]\neq 0\})\) if done as a sparse sum of `Eq(r_c, c)` over touched indices
- multiply by the opened \(V_p(r_c)\) values and sum.

Pros:
- no extra openings beyond the port \(V_p\)’s.

Cons:
- verifier cost may be high inside the RISC-V guest if `num_constraints_padded` is large.

#### Strategy B: open \(W_p(r_c)\) as aux scalars (and verify them cheaply)

Have the wiring prover append **aux virtual openings** for \(W_p\) at \(r_c\) (one scalar each), and the wiring verifier:

- recomputes \(W_p(r_c)\) from the edge list using a sparse MLE evaluation
- checks it matches the opened aux scalar (so the prover can’t lie)
- uses that scalar in the expected-output-claim computation

This keeps the verifier’s “main” polynomial evaluation cheap and keeps the number of extra claims O(#port-types), not O(#edges).

#### Strategy C (recommended): skip explicit \(W_p\) MLEs; compute the final scalar by scanning edges
You can avoid materializing `W_p[c]` entirely and instead compute the verifier’s expected output claim directly as a sparse
sum of `Eq(r_c, c_endpoint)` terms:

\[
\delta(r_c) = \sum_{e=(src\to dst)} \lambda_e \cdot \big(
  Eq(r_c, c_{src})\cdot V_{src}(r_c) \;-\; Eq(r_c, c_{dst})\cdot V_{dst}(r_c)
\big)
\]

where \(c_{src}\) / \(c_{dst}\) are the **global** `constraint_idx` values for the edge endpoints, and \(V_{\*}(r_c)\) are the
opened fused port values (plus any selector-kernel factors, e.g. GT’s `Eq(u,τ)` / step selection).

This matches the existing verifier structure (edge loop) but swaps “per-instance opening lookup” for “fused opening + `Eq(r_c, c)` weight”:

- GT edge loop today: `jolt-core/src/zkvm/recursion/gt/wiring.rs` L591-L600
- G1 edge loop today: `jolt-core/src/zkvm/recursion/g1/wiring.rs` L486-L493
- G2 edge loop today: `jolt-core/src/zkvm/recursion/g2/wiring.rs` L617-L627

### Handling GTExp output-step selection \(s_{\text{out}}(c)\)

In the current (non-fused) GT wiring prover, step selection is handled by a per-exp `eq_s_by_exp` polynomial; in a fused wiring scheme it becomes a function of `c`.

Two options:

- **Option 1 (public function)**: treat `s_out(c)` as public from `gt_exp_public_inputs[c]` (digits length), and in the prover’s polynomial table construction, bake the step selector into the construction of \(V_{\rho}(c)\).
- **Option 2 (aux selector polynomial)**: create a public/aux MLE `EqOutStep(c,s)` over `(c,s)` that equals 1 when `s = s_out(c)` and 0 otherwise, and use it to select the correct step inside the sumcheck.

Option 1 is likely simpler if we already build \(V_{\rho}\) as a “collapsed” c-only table.

### Extension to G1/G2 wiring

G1/G2 wiring is simpler because:

- the endpoint step is a fixed constant (`255`), so no c-dependent step selector is required.
- ports are “coordinate-batched” with \(\mu\) (already done today).

The fused approach is the same pattern:

- define c-only port values \(V_p(c)\) for each port-type \(p\)
- build weights \(W_p(c)\) from the edge list + \(\lambda_e\)
- prove a single fused identity over \(c\)

### What this would change in code (high level)

1. Introduce a trait (or enum) for wiring value sources, e.g.:

```rust
trait GtWiringValueSource {
    fn src_contrib(&self, edge: &GtWiringEdge) -> Fq; // or (value, weight)
    fn dst_contrib(&self, edge: &GtWiringEdge) -> Fq;
}
```

2. Add a new backend implementation that:

- does **not** call `acc.get_virtual_polynomial_opening(VirtualPolynomial::*(instance), ...)`
- instead uses openings of **fused port polynomials** (and possibly aux \(W_p\)’s)

3. Change the wiring sumcheck instance itself from “over (s,u)” / “over s” to “over c” (or “over (c,selector-vars)” if we keep selector-vars explicit).

#### Revision: variable/challenge ordering should preserve Stage-2 “\(r_x\) is a suffix”
In recursion Stage 2, sumcheck instances are **suffix-aligned** in the batched protocol: shorter instances get a suffix of the
max-length challenge vector. The verifier therefore interprets `r_x` as the **suffix** of the Stage-2 challenge vector
(`jolt-core/src/zkvm/recursion/verifier.rs` L181-L210).

To keep legacy 11-var \(x\)-based instances compatible while adding extra index variables (like \(c\)), a fused wiring instance
should treat \(x\) as the **last** 11 rounds and prepend the extra rounds for \(c\) (i.e., bind \(c\) first, then \(x\)).

#### Prover-side efficiency note: “bind \(c\) first” is not inherently slower
With the fused-row physical layout `P_t(c,x)` using variable order **`[x vars (low bits), c vars (high bits)]`**
(see the fused porting plan), binding \(c\) first corresponds to binding the **most significant** variables.
In the current polynomial engine this is explicitly supported:

- `MultilinearPolynomial::sumcheck_evals_array` has both `HighToLow` and `LowToHigh` paths
  (`jolt-core/src/poly/multilinear_polynomial.rs` L372-L408).
- For dense polynomials, `DensePolynomial::bind_parallel` dispatches on `BindingOrder`
  (`jolt-core/src/poly/dense_mlpoly.rs` L83-L92).
  - `HighToLow` (top-bit binding) combines the **left and right halves** via `split_at_mut`, which is a linear scan over two
    contiguous ranges (good locality), and is in-place in the default optimized path
    (`bound_poly_var_top_zero_optimized`, `jolt-core/src/poly/dense_mlpoly.rs` L130-L145).
  - `LowToHigh` (bottom-bit binding) combines contiguous pairs via `par_chunks_exact(2)` (also good locality), but in the
    default optimized path it allocates a fresh buffer each round
    (`bound_poly_var_bot_01_optimized`, `jolt-core/src/poly/dense_mlpoly.rs` L221-L240).

Intuition: binding \(c\) first quickly collapses the \(2^k\) rows (“constraints”) into a single random linear combination of rows,
leaving a plain 11-var \(x\)-table; the last 11 rounds are then exactly the familiar “11-var” tail.

### Open questions / things to decide

- **Where do fused port polynomials live?**
  - legacy matrix-backed layout (as in the porting plan), or
  - streaming layout (construct c-indexed tables from `ConstraintSystem` stores).
- **Verifier cost target**:
  - is O(2^k) acceptable in the recursion guest, or do we need aux openings for \(W_p(r_c)\)?
- **Do we fuse “wiring” now or keep it separate?**
  - wiring is already “one sumcheck instance per type”; fusing means changing its *variables* to include \(c\), not just batching edges.

### Practical next step (recommended)

Prototype a **GT-only fused wiring** check in a standalone module that:

- builds the local-instance → global-`constraint_idx` maps from `constraint_types`
- reads fused port values from `P_t(r_c,r_x)` openings (plus any aux selectors)
- computes the expected output claim via Strategy C’s edge scan (no explicit `W_*[c]` tables)
- runs a small sumcheck whose variable order is compatible with Stage-2 suffix alignment (extra vars first, then the 11 \(x\) vars)

This gives a clear measurement of:

- how many openings are saved,
- prover cost to build the fused c-tables, and
- verifier cost (guest cycles) to evaluate weights and check the final claim.

### Implementation checklist (handoff-ready)

This is what still needs to be pinned down (or implemented) before another agent can start coding confidently.

#### 0) Dependencies / “what exists” contract
- **Fused row openings must exist and be addressable in the accumulator**:
  - Decide the `VirtualPolynomial` IDs for `P_t(c,x)` openings (e.g. a new variant like the porting plan suggests),
    and which `SumcheckId` they are keyed under (`jolt-core/src/poly/opening_proof.rs` `SumcheckId`, L162-L217).
  - Decide the canonical ordering for caching these openings (consensus-critical).

**In-repo precedent (already implemented):**

- `RecursionPoly` already has a fused variant (`RecursionPoly::G1AddFused { term }`) and `VirtualPolynomial::g1_add_fused(term)`
  (`jolt-core/src/zkvm/witness.rs` L760-L804 and L1126-L1129).
- There is an (as-of-now unused) fused sumcheck implementation that caches “one opening per term, across all constraints”:
  - `FusedG1AddParams::num_rounds = k + 11` (`jolt-core/src/zkvm/recursion/g1/fused_addition.rs` L110-L113)
  - `FusedG1AddProver::cache_openings` appends `VirtualPolynomial::g1_add_fused(term)` under `SumcheckId::G1Add`
    (`.../g1/fused_addition.rs` L273-L291)
  - `FusedG1AddVerifier` shows how to evaluate an indicator \(I_{\text{family}}(r_c)\) as a sparse sum
    \(\sum_{idx} Eq(r_c, idx)\) using `index_to_binary` + `EqPolynomial::mle`
    (`.../g1/fused_addition.rs` L349-L362; helper `index_to_binary` is in
    `jolt-core/src/zkvm/recursion/constraints/system.rs` L14-L23; `EqPolynomial::mle` is in
    `jolt-core/src/poly/eq_poly.rs` L18-L34).

**Proposed resolution (generalize this pattern):**

- Add a generic fused ID for “polytype row fused across constraints”, e.g.
  `RecursionPoly::FusedPolyType { poly_type: PolyType }` and a ctor
  `VirtualPolynomial::recursion_fused_poly_type(poly_type)`.
- Key these fused polytype openings under the *same* `SumcheckId` that semantically “owns” them (mirroring `G1AddFused`):
  - `SumcheckId::GtMul` for `{MulLhs,MulRhs,MulResult,MulQuotient}`
  - `SumcheckId::GtExpClaimReduction` for `{RhoPrev,Quotient}` (if Stage-1b remains the producer of the stage-2-point rho/quotient)
  - `SumcheckId::G1ScalarMul`, `SumcheckId::G2ScalarMul`, `SumcheckId::G1Add`, `SumcheckId::G2Add`, ...
  This avoids needing a new `SumcheckId` and keeps “where to fetch” local to each family.
- Canonical cache order: iterate `PolyType` in ascending discriminant order (`poly_type as usize`).

#### 1) Keep wiring appended last in Stage 2
Legacy wiring relies on earlier Stage-2 instances to have already cached the port openings it reads, so it is appended **last**:

- Prover: `RecursionProver::prove_stage2` pushes wiring provers last (`jolt-core/src/zkvm/recursion/prover.rs` L1162-L1197).
- Verifier: `RecursionVerifier::verify_stage2` pushes wiring verifiers last (`jolt-core/src/zkvm/recursion/verifier.rs` L531-L542).

The fused wiring backend should keep the same “appended last” placement so its required fused openings are available.

#### 2) Challenge layout (Stage-2 suffix alignment)
Batched sumcheck uses `round_offset = max_num_rounds - num_rounds` so shorter instances are **suffix-aligned**
(`jolt-core/src/subprotocols/sumcheck.rs` L80-L93, L160-L166).

The recursion verifier explicitly parses Stage-2 challenges as:

- `r_x` = **suffix** of length `num_constraint_vars`
- optional `r_c` = the `k` challenges immediately before `r_x`

(`jolt-core/src/zkvm/recursion/verifier.rs` L181-L210).

Any fused wiring instance that introduces \(c\) must therefore treat the 11 “\(x\)” variables as the **last** 11 rounds and the
\(c\)-variables as earlier rounds.

**Important consequence (resolved):** if multiple Stage-2 instances use \(c\), they should all have the **same** `num_rounds = k + 11`.
Otherwise, suffix alignment will drop some of the \(c\) challenges for shorter instances.

This is exactly how the existing fused G1Add prototype is structured (`FusedG1AddParams::num_rounds = k + 11`,
`jolt-core/src/zkvm/recursion/g1/fused_addition.rs` L110-L113).

#### 3) Instance→constraint index maps (required for all wiring groups)
Implement deterministic maps from family-local instance indices in `WiringPlan` to global `constraint_idx`:

- Build by scanning `RecursionVerifierInput.constraint_types` (global order) (`jolt-core/src/zkvm/recursion/verifier.rs` L93-L115).
- Use existing patterns in:
  - `WiringG1Verifier::new` (`jolt-core/src/zkvm/recursion/g1/wiring.rs` L354-L365)
  - `WiringG2Verifier::new` (`jolt-core/src/zkvm/recursion/g2/wiring.rs` L445-L458)

#### 4) Port-type table + selector kernels (must be explicit)
For each wiring group (GT/G1/G2), write down a table:

- **port type** (e.g. `GtMulLhs`, `G1ScalarMulOut.x`, `G2AddInP.ind`, …)
- **source of truth**:
  - fused `PolyType` row (`jolt-core/src/zkvm/recursion/constraints/system.rs` has committed poly specs; e.g.
    `G1_ADD_SPECS` at L430-L443, `G2_ADD_SPECS` at L469-L491),
  - or public/boundary constant (and whether it is per-constraint or global).
- **selector kernel factor** \(S_p(x)\) needed so comparisons happen on the same subcube/point:
  - GT uses `Eq(u,τ)` and a step selector (see the seam in `jolt-core/src/zkvm/recursion/gt/wiring.rs` L458-L599).
  - Scalar-mul traces are “8-var in an 11-var ambient domain” with a pad selector (see
    `G1ScalarMulPublicInputs::evaluate_bit_mle`, `jolt-core/src/zkvm/recursion/g1/scalar_multiplication.rs` L74-L94).

**Resolved (GT wiring: concrete table + how the fused expected-claim is computed):**

GT wiring currently computes, at the Stage-2 wiring point \(r=(r_{\text{step}}, r_{\text{elem}})\),

\[
\sum_{e} \lambda_e \cdot Eq(u=\tau, r_{\text{elem}})\cdot Eq(s=s_{\text{src}}, r_{\text{step}})\cdot (\text{src}_e(r) - \text{dst}_e(r))
\]

(`jolt-core/src/zkvm/recursion/gt/wiring.rs` `expected_output_claim`, L570-L601).

In the fused backend we extend this to \(r=(r_c, r_{\text{step}}, r_{\text{elem}})\) by multiplying each endpoint by an
endpoint selector `Eq(r_c, c_endpoint)`:

\[
\delta(r) = \sum_{e=(src\to dst)} \lambda_e \cdot Eq(u=\tau, r_{\text{elem}})\cdot Eq(s=s_{\text{src}}, r_{\text{step}})
\cdot \Big( Eq(r_c, c_{src})\cdot \text{SrcPoly}(r) \;-\; Eq(r_c, c_{dst})\cdot \text{DstPolyOrConst}(r) \Big)
\]

where:

- \(c_{src}\) is the global `constraint_idx` of the source producer instance.
- \(c_{dst}\) is:
  - the global `constraint_idx` of the destination instance (for `GtMulLhs/Rhs` and `GtExpBase`), or
  - **\(c_{dst} := c_{src}\)** for global constants without an instance (`JointCommitment`, `PairingBoundaryRhs`),
    so the edge is anchored to the producing constraint.
- `Eq(r_c, c)` is computed as `EqPolynomial::mle(&r_c, &index_to_binary(c, k))`
  (`index_to_binary`: `jolt-core/src/zkvm/recursion/constraints/system.rs` L14-L23; `EqPolynomial::mle`:
  `jolt-core/src/poly/eq_poly.rs` L18-L34).
- `Eq(s=s_src, r_step)` uses the *existing* per-edge logic in the GT seam
  (`LegacyGtWiringValueSource::eq_s_for_src`, `jolt-core/src/zkvm/recursion/gt/wiring.rs` L460-L487),
  so we do **not** need a new `(c,s)` selector polynomial for `s_out(c)` in the verifier.

Port/constant mapping for GT wiring:

- **src = `GtExpRho { instance }`**
  - **c index**: `c_src = gt_exp_constraint_idx[instance]` (from `constraint_types`)
  - **opened value**: fused `PolyType::RhoPrev` at the Stage-2 point (same point wiring uses)
  - **step selector**: `Eq(s, s_out(instance))` (existing logic; `gt_exp_out_step[instance]` is derived from
    `RecursionVerifierInput.gt_exp_public_inputs`, `jolt-core/src/zkvm/recursion/gt/wiring.rs` L533-L542)
- **src = `GtMulResult { instance }`**
  - **c index**: `c_src = gt_mul_constraint_idx[instance]`
  - **opened value**: fused `PolyType::MulResult` at the Stage-2 point
  - **step selector**: `Eq(s, 0)` (existing logic in the seam, `gt/wiring.rs` L479-L486)
- **dst = `GtMulLhs { instance }` / `GtMulRhs { instance }`**
  - **c index**: `c_dst = gt_mul_constraint_idx[instance]`
  - **opened value**: fused `PolyType::{MulLhs,MulRhs}` at the Stage-2 point
  - **selector**: shares the edge’s `Eq(u,τ)*Eq(s,...)` weight (wiring weights depend on `edge.src` today)
- **dst = `GtExpBase { instance }`**
  - **c index**: `c_dst = gt_exp_constraint_idx[instance]`
  - **value**: public base `gt_exp_public_inputs[instance].base` packed-evaluated at `r_elem`
    (`eval_fq12_packed_at`, `jolt-core/src/zkvm/recursion/gt/wiring.rs` L94-L97 and `dst_at_r`, L518-L521)
- **dst = `JointCommitment`**
  - **c index**: `c_dst := c_src` (anchor to producer)
  - **value**: `RecursionVerifierInput.joint_commitment` packed-evaluated at `r_elem`
    (`dst_at_r`, `jolt-core/src/zkvm/recursion/gt/wiring.rs` L519-L520)
- **dst = `PairingBoundaryRhs`**
  - **c index**: `c_dst := c_src` (anchor to producer)
  - **value**: `RecursionVerifierInput.pairing_boundary.rhs` packed-evaluated at `r_elem`
    (`dst_at_r`, `jolt-core/src/zkvm/recursion/gt/wiring.rs` L520-L521)

#### 5) Decide how to handle c-dependent boundary constants
Some wiring endpoints are not in the committed rows but depend on `constraint_idx`:

- GTExp base (`GtConsumer::GtExpBase { instance }`) is per GTExp constraint.
- G1/G2 scalar-mul base points are per scalar-mul constraint.

You must decide whether these are:

- evaluated as a **public MLE over \(c\)** in the verifier (sparse `Eq(r_c,c)` sum), or
- introduced as **aux opened polynomials/scalars** (more claims, lower verifier cost), or
- absorbed directly into the edge-scan contribution logic.

**Resolved recommendation:** absorb boundary constants directly in the edge loop (no aux openings).

Rationale:

- Wiring already loops over edges in the verifier (`gt/wiring.rs` L591-L600; `g1/wiring.rs` L486-L493; `g2/wiring.rs` L617-L627).
- For per-instance constants (`GtExpBase`, scalar-mul bases), the edge already carries the local `instance`, so the verifier can:
  - map it to `constraint_idx` (for the `Eq(r_c, c_dst)` factor), and
  - evaluate the constant directly from public inputs (no accumulator opening required).
- For global constants (`JointCommitment`, `PairingBoundaryRhs`), anchoring to `c_src` avoids any `2^k` amplification in the sum over `c`.

Concretely, what “\(2^k\) amplification” meant:

- In the fused world, each equality we want to enforce must be “located” at a specific constraint index \(c_0\).
  The standard way to do that is to multiply by a Lagrange factor \(Eq(c, c_0)\).
- A boundary equality like “producer output equals `joint_commitment`” is only meant to apply at the producer’s constraint.
  So we encode it as:

  \[
  Eq(c, c_{\text{src}})\cdot(\text{producer}(c,x) - \text{joint\_commitment}(x))
  \]

- If instead you treated `joint_commitment` as “living at every \(c\)” (i.e., omitted \(Eq(c, c_0)\) / didn’t anchor it),
  you would be changing the statement you prove: the constant term would contribute across the whole \(2^k\)-sized \(c\)-domain,
  which is not equivalent to the legacy per-edge constraint.
  Setting `c_dst := c_src` for the constant endpoints is exactly the “anchor” that keeps semantics aligned with the legacy wiring check.

#### 6) GT “value source seam” needs a fused backend (and G1/G2 likely need the same seam)
GT already has a seam struct (`LegacyGtWiringValueSource`) around which a fused backend can be implemented
(`jolt-core/src/zkvm/recursion/gt/wiring.rs` L443-L601).

G1/G2 wiring verifiers still fetch per-instance openings directly in `eval_value_at_r`
(`jolt-core/src/zkvm/recursion/g1/wiring.rs` L377-L462, `jolt-core/src/zkvm/recursion/g2/wiring.rs` L473-L593).

For a clean handoff implementation, it helps to factor analogous “value source” seams in G1/G2 as well.

**Resolved (GT implementation sketch, using the existing seam):**

- Extend `WiringGtVerifier::num_rounds()` from 11 to `k + 11` (where `k = log2(num_constraints_padded)`),
  and parse `sumcheck_challenges` as:
  - `r_c = &sumcheck_challenges[..k]`
  - `r_step = &sumcheck_challenges[k..k + STEP_VARS]` (7)
  - `r_elem = &sumcheck_challenges[k + STEP_VARS..]` (4)
- Add a second value-source implementation (alongside `LegacyGtWiringValueSource`) that:
  - computes `Eq(r_c, c_idx)` using `index_to_binary` + `EqPolynomial::mle` (see the fused G1Add verifier precedent),
  - fetches **fused** opened claims for the relevant `PolyType`s (e.g. `RhoPrev`, `Mul{Lhs,Rhs,Result}`),
  - handles boundary constants directly (Section “Resolved recommendation” above),
  - and returns per-edge contributions in the form:
    - `src_term = Eq(r_c, c_src) * src_opened_at_r`
    - `dst_term = Eq(r_c, c_dst) * dst_opened_at_r_or_const`
- Modify `expected_output_claim` to compute:
  - `eq_u = eq_lsb_mle(tau, r_elem_chal)` (same as today, `gt/wiring.rs` L580-L581)
  - `eq_s = eq_s_for_src(edge.src)` (same as today, `gt/wiring.rs` L460-L487)
  - `delta += λ_e * (eq_u * eq_s) * (src_term - dst_term)`

This preserves the polynomial structure and degree bound (per-variable degree stays 2), but swaps “how values are sourced”
behind the seam, which is exactly what the refactor was intended for (`gt/wiring.rs` L443-L447).

#### 7) Tests needed for a safe first implementation
- **Mapping correctness**: unit test that the instance→constraint index maps match the constraint list counts.
- **Equivalence sanity** (small circuits): for small `k`, compare fused wiring expected-claim computation against the legacy edge
loop after constructing compatible inputs (edge list, lambdas, selectors).
- Keep existing recursion integration tests as regression (wiring already has explicit tests).

