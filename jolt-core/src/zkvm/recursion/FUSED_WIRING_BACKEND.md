## Fused Wiring Backend (Design Notes)

This note explains what the “value source seam” in `gt/wiring.rs` means, and sketches how a **fused** wiring backend could work if we stop caching **per-instance** port openings in Stage 2.

---

## 2026-01 update: avoid GTMul padding by making `c_gt` a suffix

We hit a major practical issue with the first “end-to-end GT fused” implementation:

- We fused GT over a GT-local constraint index `c_gt` (padded to a power of two, `k_gt` bits).
- We also **embedded GTMul’s native 4-var element domain into the packed GTExp 11-var domain** by replicating over the 7 step bits.

That replication is **unnecessary for soundness** (wiring only needs a selector-point evaluation on the 4 element variables), and it causes a large witness/polynomial blowup:

- GTMul natural table size is \(2^4 = 16\).
- Replicating into 11 vars makes it \(2^{11} = 2048\) (a \(128\times\) blowup) *per instance*.
- When additionally fused over `c_gt`, the packed witness contains full rows of size \(2^{11+k_{gt}}\) even for GTMul ports.

### Proposed fix (this document)

We keep **GT-local fusion** (so `k_gt = log2(next_pow2(#(GtExp)+#(GtMul)))`), but we change the **Stage-2 variable order** so that `c_gt` is a **suffix** of the batched Stage-2 challenge vector.

That allows:

- GTExp-fused instances to use \((s,u,c)\) with \(7+4+k_{gt}\) rounds.
- GTMul-fused instances to use only \((u,c)\) with \(4+k_{gt}\) rounds, as a **suffix** of the GTExp-fused challenge vector (no padding to 11 vars).
- (Optionally) a wiring check over only `c_gt` (k rounds) can also be suffix-aligned.

This change removes the 4→11 replication entirely and reduces the packed witness size drastically while keeping Stage-2 batching and Stage-3 prefix packing coherent.

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

The fused-virtual-polynomial plan introduces an explicit MLE index variable \(c\) over a padded domain \(2^k\).

**In GT fused mode we use a GT-local domain**:

- \(c \equiv c_{gt}\) ranges over only `{GtExp,GtMul}` constraints, in global order.
- `num_gt_constraints_padded = next_pow2(num_gt_constraints)` and `k_gt = log2(num_gt_constraints_padded)`.

This is implemented by `gt/indexing.rs` (`num_gt_constraints_padded`, `k_gt`) and avoids paying `k = log2(total_constraints)` when we only need to wire GT edges.

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

- **No GTMul padding (new)**:
  - GTMul `lhs/rhs/result` are **4-var** functions \(P_{\text{MulLhs}}(c, u)\), etc. with \(u\in\{0,1\}^4\).
  - Packed GTExp `rho` (actually `RhoPrev`) remains an **11-var** function \(P_{\text{RhoPrev}}(c, (s,u))\) with \(s\in\{0,1\}^7\).

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

---

## Further optimization: **separate index domains** (`k_exp` and `k_mul`) with one Stage-2 suffix

The GT-local “union domain” `c_gt` (ranging over `{GtExp,GtMul}` together) is easy to reason about, but it can cost ~2× in
commitment size when `next_pow2(n_exp + n_mul) = 2 * next_pow2(max(n_exp, n_mul))`.

We can eliminate that factor by introducing two **family-local** domains:

- `c_exp ∈ {0,1}^{k_exp}`, where `k_exp = log2(next_pow2(num_gt_exp))`
- `c_mul ∈ {0,1}^{k_mul}`, where `k_mul = log2(next_pow2(num_gt_mul))`

and only ever fusing exp rows over `c_exp` and mul rows over `c_mul`.

### Constraint: Stage-2 batched sumcheck has only one challenge suffix

Stage 2 produces one max-length challenge vector (suffix-aligned across instances), so we cannot have two disjoint “index suffix blocks”.
Instead, we define a single suffix length:

\[
k := \max(k_{\mathrm{exp}}, k_{\mathrm{mul}}),
\]

and interpret `r_c ∈ F^k` as the Stage-2 index suffix for **both** families, but with each family reading only the tail bits it needs:

- `r_{c,exp}` = the last `k_exp` challenges of `r_c`
- `r_{c,mul}` = the last `k_mul` challenges of `r_c`

This preserves suffix-alignment compatibility while allowing the *committed fused rows* to use the smaller `k_exp`/`k_mul`.

### Option B (formal statement): one fused GT wiring identity, two k’s

We want to enforce the randomized edge sum (at the GT selector point \(a_{GT}=(s_{\mathrm{src}},\tau)\)):

\[
0 \stackrel{!}{=} \sum_{e\in E_{GT}} \lambda_e \cdot (\mathrm{Src}_e(a_{GT})-\mathrm{Dst}_e(a_{GT})).
\]

Define the fused port polynomials we will open at the wiring point \(r=(r_{\mathrm{step}}, r_u, r_c)\):

- Exp producer port (rho):
  \[
  \rho^{\mathrm{exp}}(c_{\mathrm{exp}}, s, u),\quad c_{\mathrm{exp}}\in\{0,1\}^{k_{\mathrm{exp}}},\ s\in\{0,1\}^7,\ u\in\{0,1\}^4.
  \]
- Mul ports (no 4→11 padding):
  \[
  L^{\mathrm{mul}}(c_{\mathrm{mul}},u),\ R^{\mathrm{mul}}(c_{\mathrm{mul}},u),\ Out^{\mathrm{mul}}(c_{\mathrm{mul}},u),
  \quad c_{\mathrm{mul}}\in\{0,1\}^{k_{\mathrm{mul}}},\ u\in\{0,1\}^4.
  \]

Let `Eq_k(·,·)` be the usual multilinear equality kernel on a \(k\)-bit Boolean hypercube, and similarly `Eq_{k_exp}`, `Eq_{k_mul}`.
Let \(Eq_u(u,\tau)\) be the GT element-selector kernel (as in the current GT wiring).

Because the wiring sumcheck (for simplicity) will run over a single \(k\)-bit suffix \(c\in\{0,1\}^k\), but the exp/mul rows only
depend on \(k_{\mathrm{exp}}\) / \(k_{\mathrm{mul}}\) tail bits, we include normalization factors:

\[
\beta_{\mathrm{exp}} := 2^{-(k-k_{\mathrm{exp}})},\qquad
\beta_{\mathrm{mul}} := 2^{-(k-k_{\mathrm{mul}})}.
\]

These exactly cancel the dummy-variable replication factor from summing over the extra \(k-k_{\*}\) bits.

Now define the wiring polynomial (conceptually, degree ≤ 2 in each variable) over \((s,u,c)\in\{0,1\}^{7}\times\{0,1\}^{4}\times\{0,1\}^{k}\):

\[
\boxed{
F(s,u,c) :=
Eq_u(u,\tau)\cdot\Big(
\beta_{\mathrm{exp}}\cdot \sum_{e:\;\mathrm{src}(e)=\mathrm{GtExpRho}(i)}
\lambda_e \cdot Eq_{k_{\mathrm{exp}}}(c_{\mathrm{tail}}, c_{\mathrm{exp}}(i))\cdot Eq_7(s,s_{\mathrm{out}}(i))\cdot \rho^{\mathrm{exp}}(c_{\mathrm{exp}}(i),s,u)
\;+\;
\beta_{\mathrm{mul}}\cdot \sum_{e}
\lambda_e \cdot \Phi_e(s,u,c)
\Big)
}
\]

where \(\Phi_e\) is the appropriate mul-side term for edge \(e\), using `Eq_{k_mul}(c_tail, c_mul(j))` and the correct port:

- if edge consumes `GtMulLhs(j)`: contribute \(-Eq_{k_{\mathrm{mul}}}(c_{\mathrm{tail}},c_{\mathrm{mul}}(j))\cdot Eq_s(s;\mathrm{src}(e))\cdot L^{\mathrm{mul}}(c_{\mathrm{mul}}(j),u)\)
- if edge consumes `GtMulRhs(j)`: same with \(R^{\mathrm{mul}}\)
- if edge produces `GtMulOut(j)`: contribute \(+Eq_{k_{\mathrm{mul}}}(c_{\mathrm{tail}},c_{\mathrm{mul}}(j))\cdot Eq_s(s;\mathrm{src}(e))\cdot Out^{\mathrm{mul}}(c_{\mathrm{mul}}(j),u)\)

and \(Eq_s(s;\mathrm{src}(e))\) is exactly the existing step-selector convention used today in GT wiring:

- `GtExpRho(i)`: \(Eq_7(s, s_{\mathrm{out}}(i))\)
- `GtMulOut(j)`: \(Eq_7(s, 0)\)

Finally, the sumcheck statement is:

\[
\boxed{
0 \stackrel{!}{=} \sum_{s,u,c} F(s,u,c).
}
\]

### How to implement Option B **without adding any Stage-3 prefix packing overhead**

“Overhead in prefix packing” mainly means “new committed packed rows / new Stage-3 entries / new auxiliary openings”.
We avoid all of that:

- **No new packed rows**: Option B uses only the already-needed fused port rows:
  - exp: `RhoPrev` (and `Quotient` for exp correctness, but wiring needs only `RhoPrev`)
  - mul: `Mul{Lhs,Rhs,Result}` (mul correctness already needs these)
- **No aux `W_p` polynomials**: we use Strategy C: the verifier (and prover) compute the edge weights as a sparse sum of `Eq(r_c_tail, idx)`
  factors inside the edge scan, rather than materializing and committing `W_p(c)` tables.
- **No extra opening points**: all needed fused port openings are at the Stage-2 wiring point \(r\) (or at the Stage-2 point slices that
  correspond to their arity; e.g. mul ports only need `(u,c_mul)` and exp rho needs `(s,u,c_exp)`).

As a result, Stage 3 continues to do exactly what it already does:
reduce whatever Stage-2 virtual openings exist to one packed evaluation claim. We’re only changing *how wiring consumes* those openings,
not creating new ones.

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

To avoid padding GTMul up to 11 vars, we want **different fused GT instances to share a common suffix**.

**Proposed Stage-2 GT variable order (round order, `BindingOrder::LowToHigh`):**

- `r_step` (7 rounds) — **low bits**, first
- `r_u` (4 rounds) — next
- `r_c_gt` (k rounds) — **suffix**, last

So the max-round GTExp-fused instances use \(7+4+k\) rounds over `(s,u,c)` and any smaller GT instance that needs only `(u,c)` can be suffix-aligned without ambiguity.

Concrete consequences:

- **Fused GTExp / fused GT shift / fused GTExp Stage-2 openings**: `num_rounds = 11 + k_gt` over `(s,u,c_gt)`.
- **Fused GTMul**: `num_rounds = 4 + k_gt` over `(u,c_gt)` (no 4→11 replication).
- **(Optional) fused GT wiring backend over `c_gt` only**: `num_rounds = k_gt` (uses suffix `r_c_gt` directly).

#### Prover-side note: physical table layout vs Stage-2 variable order
With the **`c_gt`-as-suffix** proposal, the Stage-2 round order for fused GT is `(s,u,c_gt)` (low→high).

For fused-row materialization, it is still generally advantageous to store dense tables with `c_gt` as the **high bits**
(i.e. the fastest-moving index is within-row `x`, and `c` selects a row). This matches how we conceptually build
`P_t(c, ·)` as “row \(c\) of port type \(t\)”.

The important invariant is not “bind c first” vs “bind c last”, but simply:
- all fused GT instances must agree on the same variable order, and
- the batched Stage-2 suffix alignment must expose the same `r_c_gt` suffix to every GT-fused instance that needs it.

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
- reads fused port values from:
  - `RhoPrev(c_gt, s,u)` at the Stage-2 point for GTExp producers, and
  - `Mul{Lhs,Rhs,Result}(c_gt,u)` at the Stage-2 point for GTMul ports
- computes the expected output claim via Strategy C’s edge scan (no explicit `W_*[c]` tables)
- runs a small sumcheck whose challenge layout is compatible with Stage-2 suffix alignment (with `c_gt` as a suffix)

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

For the **GT-fused no-padding** redesign, we standardize a GT-only suffix layout:

- Stage-2 max GT round order is `(r_step (7), r_u (4), r_c_gt (k_gt suffix))`.
- Define `r_c_gt` as the **suffix** of length `k_gt`.
- Define `r_u` as the 4 challenges immediately before `r_c_gt`.
- Define `r_step` as the 7 challenges immediately before `r_u`.

Then:

- any GT-fused instance that needs `(u,c_gt)` uses `num_rounds = 4 + k_gt` and is suffix-aligned to `(r_u, r_c_gt)`.
- any GT-fused instance that needs `(s,u,c_gt)` uses `num_rounds = 11 + k_gt` and sees the full GT suffix.

This is what allows GTMul to remain 4-var (no replication to 11 vars) while still sharing the same `r_c_gt` with GTExp-fused checks.

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

- Extend `WiringGtVerifier::num_rounds()` from 11 to `k_gt + 11`, and parse `sumcheck_challenges` in the **(s,u,c_gt)** order:
  - `r_step = &sumcheck_challenges[..STEP_VARS]` (7)
  - `r_elem = &sumcheck_challenges[STEP_VARS..STEP_VARS + ELEM_VARS]` (4)
  - `r_c_gt = &sumcheck_challenges[STEP_VARS + ELEM_VARS..]` (k_gt suffix)
- Add a second value-source implementation (alongside `LegacyGtWiringValueSource`) that:
  - computes `Eq(r_c_gt, c_idx)` using `index_to_binary` + `EqPolynomial::mle`,
  - fetches **fused** opened claims for the relevant `PolyType`s (e.g. `RhoPrev`, `Mul{Lhs,Rhs,Result}`),
  - handles boundary constants directly (Section “Resolved recommendation” above),
  - and returns per-edge contributions in the form:
    - `src_term = Eq(r_c_gt, c_src) * src_opened_at_r`
    - `dst_term = Eq(r_c_gt, c_dst) * dst_opened_at_r_or_const`
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

---

## Implementation checklist (concrete code touchpoints for the no-padding redesign)

This section is the “do these edits in this order” checklist to implement:

- `c_gt` is a **suffix** in Stage-2 GT-fused challenge order, and
- GTMul stays **4-var** (no 4→11 replication).

### 1) Prefix packing layout: split fused GT row arities

- **File**: `jolt-core/src/zkvm/recursion/prefix_packing.rs`
- **Change**: in `PrefixPackingLayout::from_constraint_types_gt_fused`
  - keep `RhoPrev` / `Quotient` fused rows at `num_vars = 11 + k_gt`
  - change `Mul{Lhs,Rhs,Result,Quotient}` fused rows to `num_vars = 4 + k_gt`

### 2) Dense witness emission: stop GTMul 4→11 replication

- **File**: `jolt-core/src/zkvm/recursion/witness_generation.rs` (`emit_dense` → `fill_entry` for `entry.is_gt_fused`)
- **Change**:
  - build GTExp fused sources over `(s,u,c_gt)` (size \(2^{11+k}\)) as needed
  - build GTMul fused sources over `(u,c_gt)` (size \(2^{4+k}\)) and fill only those fused entries
  - remove the “replicate 4-var over 7 step bits” loops for GTMul fused packing

### 3) Fused GTMul sumcheck: move to `(u,c_gt)` with `4 + k_gt` rounds

- **File**: `jolt-core/src/zkvm/recursion/gt/fused_multiplication.rs`
- **Change**:
  - `num_constraint_vars = 4` (element vars only)
  - `num_rounds = 4 + k_gt`
  - remove padding of `g(u)` to 11 vars and remove any per-x replication logic
  - materialize fused GTMul term tables in `(u,c_gt)` order consistent with Stage-2 binding order

### 4) GT-fused “point-capture” instance: cache openings at the new points

- **File**: `jolt-core/src/zkvm/recursion/gt/fused_stage2_openings.rs`
- **Change**:
  - ensure it binds/evaluates fused GTExp polynomials at the Stage-2 point whose round order is `(s,u,c_gt)`
  - ensure it does **not** expect `c_gt` as a prefix segment in `sumcheck_challenges`

### 5) GT-fused shift: update variable slicing to `(s,u,c_gt)` (with `c_gt` suffix)

- **File**: `jolt-core/src/zkvm/recursion/gt/fused_shift.rs`
- **Change**:
  - parse `sumcheck_challenges` into `(r_step, r_u, r_c_gt)` in that order
  - update any fused-rho table layout assumptions so the step bits are still the “low bits” for the shift relation

### 6) GT wiring (and binding check): update challenge parsing and eliminate GTMul padding

- **Files**:
  - `jolt-core/src/zkvm/recursion/gt/wiring.rs`
  - `jolt-core/src/zkvm/recursion/gt/wiring_binding.rs`
- **Change**:
  - if wiring is extended to include `c_gt`, parse as `(r_step, r_u, r_c_gt)`
  - for u-only GTMul ports, evaluate at `r_u` and avoid allocating 11-var replicated tables
  - any `Eq(r_c, ·)` factors must use the **`r_c_gt` suffix**, not a prefix slice

### 7) Stage-3 prefix packing reduction: ensure r-suffix conventions still line up

- **Files**:
  - `jolt-core/src/zkvm/recursion/prover.rs` (`prove_stage3_prefix_packing`)
  - `jolt-core/src/zkvm/recursion/verifier.rs` (`verify_stage3_prefix_packing`)
- **Change**:
  - verify that the “Stage‑2 suffix → reverse → packed low variables” mapping still matches the new GT fused table layouts
  - ensure each packed entry of arity `m` consumes the correct suffix slice of the Stage‑2 challenge vector (via the existing `max_native_vars` / suffix logic)

### 8) Sanity check: dense witness size drops below the Hyrax setup cap

- **Expected outcome** (e.g. `fibonacci`):
  - `dense_num_vars` should drop (in our failing run it was 23 due to 6×\(2^{20}\) GT-fused rows).
  - Hyrax setup mismatch (`r_size=4096` vs `generators.len=2048`) should disappear once the packed size no longer crosses the `2^23` threshold.

