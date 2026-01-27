## Goal (what must be implemented, and why it’s soundness-critical)

We need to implement the **missing Stage-2 wiring/boundary constraints** (“copy constraints”) so the recursion proof enforces that all proven ops form **one coherent DAG**, not a multiset of unrelated valid ops.

> **2026-01 update (important)**: The first end-to-end GT fused path caused a large witness blowup by
> replicating GTMul 4-var tables into the 11-var packed GTExp domain. This plan is updated to avoid
> that entirely by making `c_gt` a **suffix** in Stage-2 challenge order and keeping GTMul fused
> polynomials on a **(u,c_gt)** domain (4 + k rounds). See
> `jolt-core/src/zkvm/recursion/FUSED_WIRING_BACKEND.md` for the detailed fused-backend notes.

- **Insertion points (Stage 2 TODOs)**:
  - Prover: `jolt-core/src/zkvm/recursion/prover.rs` (`RecursionProver::prove_stage2`, ~L1267)

```rust
// TODO: Add wiring/boundary constraints sumcheck (AST-driven).
```

  - Verifier: `jolt-core/src/zkvm/recursion/verifier.rs` (`RecursionVerifier::verify_stage2`, ~L515)

```rust
// TODO: wiring/boundary constraints.
```

---

## Core design decisions (locked in)

### AST source = Option A
Verifier derives wiring plan **deterministically from the AST reconstructed from public verification inputs**, not from prover metadata:
- **Current code reality**: the recursion verifier already reconstructs the Stage 8 batching state *and* a symbolic AST during Stage 8 recursion verification (`jolt-core/src/zkvm/verifier.rs::verify_stage8_with_recursion`, ~L1208+). Wiring should consume this AST rather than introducing a new reconstruction pathway.

```rust
// 1) Reconstruct the Stage 8 batching state + build the symbolic AST ...
let dory_snap = self.build_dory_verify_snapshot()?;
let (..., ast) = {
    let _span = tracing::info_span!("stage8_reconstruct_ast_and_batching").entered();
    // ...
};
```

**Plumbing note**: today `RecursionVerifierInput` does not carry an AST, and Stage 2 does not implement wiring yet. Implementing wiring will require passing (or deterministically re-deriving) the AST at the point where Stage-2 verifiers are constructed.

### Scope = both AST edges and combine_commitments edges
Wiring must include:
- **All Dory AST dataflow edges** (typed GT/G1/G2 values).
- **All deterministic “combine_commitments offload” edges** (GTExp outputs feeding a deterministic balanced fold of GTMul ops).
  - The balanced fold shape is formalized as `CombineDag` (`jolt-core/src/zkvm/recursion/combine_dag.rs`).
  - In the current streaming constraint planner, combine constraints are appended (exp first, then muls in layer order) in `witness_generation.rs`:

```rust
// Append GT exp constraints for combine terms.
for exp_wit in &cw.exp_witnesses { ... }

// Append GT mul constraints for combine reduction tree.
for layer in &cw.mul_layers {
    for mul_wit in layer { ... }
}
```

**Gotcha (current code)**: GT mul combine witnesses are skipped if their quotient is empty (`gt_mul_rows_from_op_witness`), so any verifier-side derivation must match this inclusion rule exactly.

### Infinity indicator must be wired (G1/G2)
Scalar-mul traces already have an “A is infinity” indicator in the constraint system (`a_indicator`), and wiring must include it. You do **not** want to wire only `(x,y)`.

- **G1 scalar mul**: the planned/native store includes `a_indicator` (see `jolt-core/src/zkvm/recursion/witness_generation.rs`, ~L222–L254), and Stage 2 constraints include it as a committed poly (`PolyType::G1ScalarMulAIndicator`).

```rust
let rows = G1ScalarMulNative {
    // ...
    a_indicator: witness.a_is_infinity_mles[0].clone(),
};
```

---

## Why wiring must be **last** inside Stage 2 (within-instance ordering gotcha)

Wiring’s verifier must read virtual openings that are cached by earlier Stage-2 instances. This is ordering-sensitive because `BatchedSumcheck::verify` calls `cache_openings()` **before** `expected_output_claim()` per instance, in instance order:

```rust
sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
let claim = sumcheck.expected_output_claim(opening_accumulator, r_slice);
```

(See `jolt-core/src/subprotocols/sumcheck.rs`, ~L225–L230.)

So wiring must be appended **after** the families that populate the accumulator (GtExpClaimReduction, GtMul, G1/G2 scalar mul, G1/G2 add, …). Wiring’s own `cache_openings` must be **a no-op** (no duplicates, no new sumcheck-id keys).

---

## The formal sumcheck relation (the part that must be exactly right)

### High-level statement
For each type \(t \in \{\mathrm{GT},\mathrm{G1},\mathrm{G2}\}\), we build a canonical edge list \(E_t\) and prove a single batched identity:
\[
\sum_{x\in\{0,1\}^{n_t}} \mathrm{Eq}(a_t, x)\cdot \Big(\sum_{e\in E_t}\lambda_{t,e}\cdot \Delta_{t,e}(x)\Big)=0
\]
where:
- \(n_{\mathrm{GT}}=11\), \(n_{\mathrm{G1}}=8\), \(n_{\mathrm{G2}}=8\)
- \(a_t\) is a fixed “selector point” (details below) sampled deterministically from transcript + constants
- \(\lambda_{t,e}\) are per-edge Fiat–Shamir challenges in canonical edge order
- \(\Delta_{t,e}(x)\) is the **coordinate-batched port difference polynomial** for edge \(e\)

By the MLE identity,
\[
\sum_{x} \mathrm{Eq}(a,x)\cdot F(x) \;=\; F(a),
\]
this enforces:
\[
\sum_{e\in E_t}\lambda_{t,e}\cdot \Delta_{t,e}(a_t)\;=\;0
\]
and with \(\lambda_{t,e}\) random and independent, implies every \(\Delta_{t,e}(a_t)=0\) except negligible probability.

### Variable layouts and selector points \(a_t\)

#### GT (11 vars = 7 step + 4 element)
Packed GTExp layout is explicitly: `index = x * 128 + s` (step vars are **low bits**, element vars are **high bits**).

```rust
/// Data layout: index = x * 128 + s (s in low 7 bits, x in high 4 bits)
/// This allows LowToHigh binding to give us:
/// - Phase 1 (rounds 0-6): bind step variables s
/// - Phase 2 (rounds 7-10): bind element variables x
```

(See `jolt-core/src/zkvm/recursion/gt/exponentiation.rs`, ~L165–L170.)

Define:
- \(s_{\text{last}} = (1,1,1,1,1,1,1)\in\{0,1\}^7\) (step 127; this is the “endpoint” slice)
- sample \(\tau \in \mathbb{F}^4\) from transcript (4 challenges)
- set \(a_{\mathrm{GT}} = (s_{\text{last}} \,\|\, \tau)\in\mathbb{F}^{11}\) in **round order** (low-to-high vars: first 7 are step vars, last 4 are element vars)

#### GT fused indexing (adds `c_gt` as a suffix; removes GTMul padding)

In fully fused GT mode, we also introduce a GT-local constraint index `c_gt` (k bits):

- `num_gt_constraints = #(GtExp) + #(GtMul)` in global order
- `num_gt_constraints_padded = next_pow2(max(1, num_gt_constraints))`
- `k_gt = log2(num_gt_constraints_padded)`

**Stage-2 batched-sumcheck challenge order (round order, `BindingOrder::LowToHigh`) is:**

- `r_step` (7 rounds) — first
- `r_u` (4 rounds) — next
- `r_c_gt` (k rounds) — **suffix**, last

This is the key redesign: any GT instance that needs only `(u,c_gt)` is suffix-aligned without ambiguity.

#### G1 / G2 (8 vars = scalar-mul step index)
Define:
- \(s_{\text{last}} = (1,\dots,1)\in\{0,1\}^8\) (step 255)
- set \(a_{\mathrm{G1}}=a_{\mathrm{G2}}=s_{\text{last}}\)

No extra “element vars” exist; you’re wiring endpoints of step-indexed traces and also constant add ports.

### Port differences \(\Delta_{t,e}\)

#### GT edges
Each GT edge is between:
- GTExp “output port” = \(\rho(s,u)\) evaluated at \(s=s_{\text{last}}\) (endpoint slice), or
- GTMul “output port” = `result(u)` (4-var)
to a GTMul input port `lhs(u)` or `rhs(u)`.

Define for each edge \(e\):
\[
\Delta_{\mathrm{GT},e}(s,u)= \mathrm{Out}_e(s,u)-\mathrm{In}_e(u)
\]
where:
- if source is GTExp instance \(i\): \(\mathrm{Out}_e(s,u)=\rho_i(s,u)\) (11-var MLE)
- if source is GTMul instance \(j\): \(\mathrm{Out}_e(s,u)=\mathrm{res}_j(u)\) (independent of \(s\))
- destination is always a GTMul port polynomial: \(\mathrm{In}_e(u)\in\{\mathrm{lhs}_k(u),\mathrm{rhs}_k(u)\}\)

Then the enforced equality is at the selector point:
\[
\Delta_{\mathrm{GT},e}(a_{\mathrm{GT}})=
\begin{cases}
\rho_i(s_{\text{last}},\tau) - \mathrm{lhs}_k(\tau) & \text{(GTExp→GTMul lhs)}\\
\mathrm{res}_j(\tau) - \mathrm{rhs}_k(\tau) & \text{(GTMul→GTMul rhs)}\\
\text{etc.}
\end{cases}
\]

This is exactly the arity-mismatch resolution: GTMul ports are 4-var; GTExp rho is 11-var; the Eq-kernel with fixed \(s_{\text{last}}\) and random \(\tau\) makes it a single scalar equality per edge.

**Design decision (updated):** even in fully fused GT mode, GTMul ports remain 4-var at `r_u`; we do **not**
replicate them over step bits to manufacture an 11-var table.

#### G1 edges (coordinate batching with μ, includes indicator)
Each G1 element is represented by \((x,y,\mathrm{ind})\).
Sample \(\mu_{\mathrm{G1}}\in\mathbb{F}\).

For edge \(e\):
\[
\Delta_{\mathrm{G1},e}(s)=
(x^{out}_e(s)-x^{in}_e)\;+\;\mu_{\mathrm{G1}}(y^{out}_e(s)-y^{in}_e)\;+\;\mu_{\mathrm{G1}}^2(\mathrm{ind}^{out}_e(s)-\mathrm{ind}^{in}_e)
\]
- For scalar-mul outputs, `out` is the step-indexed trace column at that step.
- For add ports, `in/out` are 0-var constants (replicated conceptually).

The enforced equality is \(\Delta_{\mathrm{G1},e}(s_{\text{last}})=0\).

#### G2 edges (5 coords, includes indicator)
Each G2 element is \((x_{c0},x_{c1},y_{c0},y_{c1},\mathrm{ind})\).
Sample \(\mu_{\mathrm{G2}}\in\mathbb{F}\).

\[
\Delta_{\mathrm{G2},e}(s)=\sum_{j=0}^{4}\mu_{\mathrm{G2}}^j\cdot(\text{coord}^{out}_{e,j}(s)-\text{coord}^{in}_{e,j})
\]
Again enforce at \(s_{\text{last}}\).

---

## Implementation blueprint (what to code, where, and how)

### 1) New module: `jolt-core/src/zkvm/recursion/wiring.rs`
Add:
- **Edge types**
  - `enum WireType { GT, G1, G2 }`
  - `struct WireEdge { wire_type, src_value: ValueId, dst_op: OpId, dst_slot: usize, ... }`
  - also store enough info to map to concrete ports: `src_op_type`, `dst_op_type`, and the `ValueType`

- **Port mapping helpers**
  - Map (op_type, slot) → destination port kind:
    - `GTMul`: slot0=lhs, slot1=rhs
    - `G1Add`: slot0=P, slot1=Q
    - `G2Add`: slot0=P, slot1=Q
  - Map op_type → source output port kind:
    - `GTExp`: output is `rho` endpoint
    - `GTMul`: output is `result`
    - `G1ScalarMul`: output is `(x_a_next,y_a_next,a_indicator)` endpoint
    - `G2ScalarMul`: output is `(x_a_next_c0,c1,y_a_next_c0,c1,a_indicator)` endpoint
    - `G1Add`: output is `(x_r,y_r,ind_r)`
    - `G2Add`: output is `(x_r_c0,c1,y_r_c0,c1,ind_r)`

- **Instance-index maps**
  - Build deterministic `HashMap<OpId, usize>` for each relevant op family by sorting OpIds (canonical).
  - Same for GTExp and GTMul.
  - **Important**: this must match how constraints were added (witness lists are sorted by `op_id` before adding; e.g. G1Add/G2Add in `recursion/prover.rs` L560–577).

### 2) Edge derivation

#### 2a) Dory AST edges (public / verifier-derivable)
Given `AstGraph<BN254>`:
- For each node `dst` with `dst_op_id` and `dst_op`:
  - enumerate its input `ValueId`s in slot order (slot index is the “port index”)
  - for each input value `v`:
    - if producer of `v` is `AstOp::Input { .. }`: boundary input → **skip** (not a copy constraint)
    - else: create a `WireEdge` from `v` to `(dst_op_id, slot)`
- Partition into GT/G1/G2 edges using `ValueType` from the AST.

Canonical ordering (must match prover/verifier):
- Sort by `(dst_op_id, dst_slot)` (and optionally break ties by `src_value_id`).

#### 2b) combine_commitments edges (no hints; pure public derivation)
These edges are not in the Dory AST. Derive them from **counts** and the deterministic balanced fold wiring.

How to derive counts deterministically:
- From reconstructed AST:
  - `base_gt_exp = #AstOp::GTExp`
  - `base_gt_mul = #AstOp::GTMul`
- From recursion public inputs / constraint metadata:
  - `total_gt_exp = input.gt_exp_public_inputs.len()` (already used by recursion verifier)
  - `total_gt_mul = count of ConstraintType::GtMul in input.constraint_types`
- Then:
  - `combine_gt_exp = total_gt_exp - base_gt_exp`
  - `combine_gt_mul = total_gt_mul - base_gt_mul`
- Indices:
  - combine GTExp instance indices are `[base_gt_exp .. total_gt_exp)` in exactly the order they were appended (matches `add_combine_witness` exp loop).
  - combine GTMul instance indices are `[base_gt_mul .. total_gt_mul)` in level-major/left-to-right order (matches `add_combine_witness` mul loop).

Now generate edges:
- Level 0 “nodes” are the combine GTExp outputs in index order.
- For each fold level:
  - pair adjacent nodes left-to-right; create a GTMul op per pair (consume 1 GTMul instance index)
    - wire `child_left → mul.lhs`
    - wire `child_right → mul.rhs`
    - mul output becomes next-level node
  - if odd tail, carry forward node unchanged
- Continue until one node remains.

Canonical ordering:
- Emit edges in `(level, mul_index_within_level, inputslot)` order.

### 3) Wiring sumcheck instances (custom, not `ConstraintListProver/Verifier`)

You cannot use `ConstraintListVerifier` directly because wiring needs to pull openings from **multiple SumcheckIds** (GtMul vs GtExpClaimReduction vs G1Add…) and because the “instances” aren’t a fixed public constraint family.

Instead implement three custom instances implementing:
- `SumcheckInstanceProver<Fq, T>`
- `SumcheckInstanceVerifier<Fq, T>`

Model them after `GtExpClaimReduction{Prover,Verifier}` (it’s already a custom sumcheck and is the closest template).

#### 3a) Parameters
- **GT wiring**
  - `num_rounds = 11`
  - `degree = 2` (product of two multilinears: Eq(a,x) * Δ(x))
- **G1 wiring**
  - `num_rounds = 8`
  - `degree = 2`
- **G2 wiring**
  - `num_rounds = 8`
  - `degree = 2`

#### 3b) Transcript sampling (must be identical on prover and verifier)
At construction time (when building the Stage-2 instance vectors):
- sample \(a_t\):
  - GT: `a = [1;7] ++ [tau0..tau3]` where `tau_i` are transcript challenges
  - G1/G2: `a = [1;8]`
- sample μ for G1 and μ for G2
- sample λ-vector of length `|E_t|` (use transcript.challenge_vector)

This is order-sensitive: **create provers/verifiers in the same order** so transcript state matches.

#### 3c) Prover `compute_message` (how to actually compute the round polynomial)
Maintain:
- `eq_poly = MultilinearPolynomial::from(EqPolynomial::evals(&a_t))` over `n_t` vars (size 2^{n_t})
- a representation of the **edge-batched difference** Δ(x), computed from witness tables:
  - For GT:
    - ρ polynomials: from `ConstraintSystem.gt_exp_witnesses[i].rho_packed` (size 2048) → `MultilinearPolynomial`
    - GTMul polynomials: from `extract_gt_mul_constraints` (size 16) used **as 4-var** objects.
      In fused GT mode these are instead provided as fused `(u,c_gt)` rows (see “GT fused indexing” above), so no replication to 11 vars is needed.
  - For G1/G2:
    - scalar mul polynomials from `extract_g1_scalar_mul_constraints` / `extract_g2_scalar_mul_constraints` (each is size 256 for 8 vars)
    - add constants from `extract_g1_add_constraints` / `extract_g2_add_constraints` (size 1)

Then per half-index `i` (same pattern as claim reduction):
- `eq_evals = eq_poly.sumcheck_evals_array::<2>(i, LowToHigh)`
- compute `delta_evals[t]` for `t∈{0,2}` by summing all edges’ contributions at that `(i,t)`:
  - for each edge `e`, pull the relevant polynomials’ `sumcheck_evals_array::<2>(i, LowToHigh)` and compute `Δ_evals[t]`
  - multiply by `λ_e` and accumulate
- accumulate `term_evals[t] += eq_evals[t] * delta_evals[t]`
- reduce across threads
- return `UniPoly::from_evals_and_hint(previous_claim, &term_evals)`

#### 3d) Prover `ingest_challenge`
Bind:
- `eq_poly.bind_parallel(r_j, LowToHigh)`
- all mutable `MultilinearPolynomial` objects used to produce Δ (ρ polys, replicated GTMul polys, scalar mul polys, etc.) **in the same variable order** (LowToHigh)

#### 3e) Prover `cache_openings`
**NO-OP**. Do not append any new openings; wiring consumes already-cached claims.

### 4) Verifier-side `expected_output_claim` (must use the existing accumulator)

Verifier gets `sumcheck_challenges` = `r_slice` in **round order** (LowToHigh). It must compute:
\[
\mathrm{Eq}(a_t, r)\cdot \Big(\sum_{e\in E_t}\lambda_{t,e}\cdot \Delta_{t,e}(r)\Big)
\]
where:
- `Eq(a_t, r)` must be computed with the same reverse convention used by `ConstraintListVerifier` (reverse only for the Eq factor; keep round-order for everything else):

```rust
let eval_point_round: Vec<F> = sumcheck_challenges.iter().map(|c| (*c).into()).collect();
let eval_point_for_eq: Vec<F> = eval_point_round.iter().rev().copied().collect();
let eq_eval = EqPolynomial::mle(&eq_point_f, &eval_point_for_eq);
```

(See `jolt-core/src/zkvm/recursion/constraints/sumcheck.rs`, ~L475–L486.)

- For each edge, fetch the required claims from `VerifierOpeningAccumulator` using the **correct producing SumcheckId**:
  - **GTExp rho**: comes from **GtExpClaimReduction**, not from `GtExp` (claim reduction caches `VirtualPolynomial::gt_exp_rho(w)` under its own sumcheck id):

```rust
append_virtual_claims(
    accumulator,
    transcript,
    self.params.sumcheck_id, // SumcheckId::GtExpClaimReduction
    &opening_point,
    &claims,
);
```

(See `jolt-core/src/zkvm/recursion/gt/claim_reduction.rs`, ~L219–L225.)

  - **GTMul ports**: `SumcheckId::GtMul`
  - **G1 scalar mul ports**: `SumcheckId::G1ScalarMul`
  - **G2 scalar mul ports**: `SumcheckId::G2ScalarMul`
  - **G1 add ports**: `SumcheckId::G1Add` (constants)
  - **G2 add ports**: `SumcheckId::G2Add` (constants)

- Correct slicing of `r`:
  - In fused GT mode with `c_gt` suffix:
    - `r_c_gt` = suffix of length `k_gt`
    - `r_u` = the 4 challenges immediately before `r_c_gt`
    - `r_step` = the 7 challenges immediately before `r_u`
  - In legacy (non-fused) GT mode: `r_step = r[0..7]`, `r_u = r[7..11]`
  - GTMul openings are at `r_u` (4 vars), not at full 11 vars.
  - For any u-only port, interpret its value at full `r` as the claim at `r_u`.

Verifier `cache_openings` for wiring:
- **NO-OP** (critical, no duplicates).

### 5) Integrate into Stage 2 (both prover and verifier)

- Build wiring provers/verifiers **after all existing Stage-2 instances** and push them at the end of the vectors.
- Skip creating the instance for a type if `E_t.is_empty()` (but do so deterministically on both sides based on AST + combine counts).

---

## Concrete “port → opening id” mapping table (no ambiguity)

### GT
- **GTExp output port** for instance `i`:
  - polynomial: `VirtualPolynomial::gt_exp_rho(i)`
  - sumcheck id: `SumcheckId::GtExpClaimReduction`
  - evaluate at:
    - in verifier `Δ(r)`: use claim at full `r` (11 vars)
    - in enforced point `a_GT`: achieved via Eq-kernel in the sum, not by direct opening
- **GTMul ports** for instance `j`:
  - lhs: `VirtualPolynomial::gt_mul_lhs(j)` @ `SumcheckId::GtMul`
  - rhs: `VirtualPolynomial::gt_mul_rhs(j)` @ `SumcheckId::GtMul`
  - result: `VirtualPolynomial::gt_mul_result(j)` @ `SumcheckId::GtMul`
  - claims are at 4-var `r_u` (suffix of the 11-var `r`)

### G1 scalar mul output endpoint
For scalar mul instance `i`:
- `VirtualPolynomial::g1_scalar_mul_xa_next(i)` @ `SumcheckId::G1ScalarMul`
- `VirtualPolynomial::g1_scalar_mul_ya_next(i)` @ `SumcheckId::G1ScalarMul`
- `VirtualPolynomial::g1_scalar_mul_a_indicator(i)` @ `SumcheckId::G1ScalarMul`

### G1 add ports (constants)
For add instance `j`:
- input P (slot0): `g1_add_xp`, `g1_add_yp`, `g1_add_p_indicator`
- input Q (slot1): `g1_add_xq`, `g1_add_yq`, `g1_add_q_indicator`
- output R: `g1_add_xr`, `g1_add_yr`, `g1_add_r_indicator`
(all under `SumcheckId::G1Add`)

### G2 (analogous)
- scalar mul output: `g2_scalar_mul_xa_next_c0/c1`, `...ya_next_c0/c1`, `...a_indicator` under `SumcheckId::G2ScalarMul`
- add ports: `g2_add_*` under `SumcheckId::G2Add`

---

## Verifier infra: where to obtain AST (Option A) in recursion verification mode

In the recursion verification path (`zkvm/verifier.rs::verify_stage8_with_recursion`), after doing Stage-8 transcript replay to recover the Dory opening context, call `reconstruct_ast(...)` using:
- the Stage-8 Dory proof,
- the Dory verifier setup,
- the replay transcript (must match the prover),
- the Stage-8 opening point and evaluation (already computed in Stage 8),
- the commitment,
- `payload.stage9_pcs_hint` as the hints.

Then pass that `AstGraph` down so that the prover and verifier can derive the exact same wiring plan. (The prover-side storage already exists.)

- verifier has the reconstructed AST available when building wiring verifiers.

(You can keep using `payload.stage10_recursion_metadata` for non-wiring recursion inputs in the short term; the only strict requirement here is “AST comes from reconstruct_ast”, not from metadata.)

---

## Gotchas / footguns (must not get wrong)

- **No GTExp “base” wiring**: treat GTExp bases as boundary `AstOp::Input` and skip. If an AST edge ever indicates GTExp base is produced by a non-input op, fail fast (this is the policy in the migration plan).
- **Do not append openings in wiring**:
  - `cache_openings` must be a no-op on both prover and verifier.
  - Otherwise you’ll create duplicate `(VirtualPolynomial, SumcheckId)` keys or mutate transcript in a new way.
- **Use the correct SumcheckId for GTExp rho**:
  - Wiring must read `gt_exp_rho(i)` under `SumcheckId::GtExpClaimReduction`, not `SumcheckId::GtExp`.
- **Edge ordering must be identical on both sides**:
  - Derive edges only from AST + deterministic combine procedure.
  - Sort canonically.
  - Sample `λ_e` in that order.
- **Instance index mapping must be identical**:
  - It should be based on `OpId` sorting (AST is expected to carry OpIds).
  - Combine instance indices are derived from counts and the fact combine constraints are appended after base constraints (as per `add_combine_witness`).
- **Round order / slicing**:
  - Stage-2 sumchecks are suffix-aligned (documented in `RecursionProver::prove_sumchecks`):

```rust
// NOTE: Recursion constraint sumchecks are **suffix-aligned** in the batched sumcheck
// (`round_offset = max_num_rounds - num_rounds`), so shorter points are suffixes of longer
// ones in the batched challenge order.
```

(See `jolt-core/src/zkvm/recursion/prover.rs`, ~L819–L826.)

  - GT wiring uses the step+elem rounds (11) and, in fused GT mode, additionally the `c_gt` suffix (k).
  - GTMul uses the suffix `(u,c_gt)` in fused mode, and the suffix 4 in legacy mode.
- **Eq evaluation reversal**:
  - Follow the existing convention: reverse only for computing Eq-mle at the final point (like `ConstraintListVerifier`).
- **Degree bound**:
  - Wiring polynomial is `Eq(a,x) * (Σ λ·Δ(x))` → degree ≤ 2. Match `GtExpClaimReduction` style (`degree() == 2`).
- **Performance** (optional but practical):
  - Don’t naively allocate one “replicated 11-var GTMul polynomial” per edge; reuse per-instance polynomials and reference them from edges.
  - Likewise for add constants: avoid building 2^8 tables per constant if you can compute `[c,c]` directly.

---

## Minimal integration checklist (so the next AI can just execute)

- **Add `wiring.rs`** with:
  - `derive_ast_edges(ast) -> {gt_edges,g1_edges,g2_edges}`
  - `derive_combine_edges(ast_counts, total_counts) -> gt_edges_extra`
  - `build_instance_maps(ast, recursion_input) -> maps`
  - `WiringGtProver/WiringGtVerifier`, `WiringG1Prover/...`, `WiringG2Prover/...`

- **Prover**:
  - In `RecursionProver::prove_stage2` build the wiring provers at the end using `self.ast.as_ref().unwrap()` and `self.constraint_system`.
  - Append them to `provers`.

- **Verifier**:
  - Ensure recursion verification reconstructs AST (Option A) and passes it down.
  - In `RecursionVerifier::verify_stage2_constraints` build wiring verifiers at the end using reconstructed `AstGraph` and recursion input counts, then append to `verifiers`.

With the above, there should be no remaining design ambiguity: edge derivation is deterministic, the sumcheck polynomial is formally specified, and all “where do claims come from / which SumcheckId / which slice of r” details are pinned down.