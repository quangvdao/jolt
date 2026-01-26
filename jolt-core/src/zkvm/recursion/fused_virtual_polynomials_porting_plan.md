## Fused Virtual Polynomials Across Instances — Comprehensive Porting Plan

Last updated: 2026-01-24

This document is a **protocol + implementation plan** to port the current recursion SNARK to a
design where we **fuse virtual polynomials across instances** (across constraint indices) so that
Stage 2 and Stage 3 can operate with **O(#poly-types)** virtual claims rather than
**O(#constraints × #poly-types)**.

The headline goal is: **Stage 2 emits (and Stage 3 consumes) one scalar claim per fused polynomial**
(e.g., per `PolyType`, optionally plus a small fixed set of “aux” fused polynomials), while still
maintaining soundness for *nonlinear* constraints.

> Note on references: Line numbers below refer to the repository state as of this doc’s creation.
> They will drift as we implement changes.

---

## 1) Current baseline (what we’re changing)

### 1.1 Matrix layout and Stage 3 today

The recursion constraint system is represented as a big multilinear matrix `M(s, x)` where:

- `x`: “constraint variables” (currently 11 vars, row width \(2^{11}\))
- `s`: selects a row (currently `num_s_vars = log2(num_rows)`).

Rows are arranged **by `PolyType` first, then by constraint index**:

```341:345:/Users/quang.dao/Documents/SNARKs/jolt-recursion/jolt-core/src/zkvm/recursion/constraints_sys.rs
pub fn row_index(&self, poly_type: PolyType, constraint_idx: usize) -> usize {
    (poly_type as usize) * self.num_constraints_padded + constraint_idx
}
```

Stage 3 (“virtualization”) currently:

- Extracts **one scalar per (constraint, poly-type)** from the opening accumulator
- Checks that `M(r_s, r_x)` equals the multilinear extension in the `s` dimension:
  \[
    M(r_s, r_x) \stackrel{?}{=} \sum_{row} Eq(r_s, row)\cdot v_{row}
  \]

This is the place where the **O(#constraints × #poly-types)** claim vector is consumed:

```301:314:/Users/quang.dao/Documents/SNARKs/jolt-recursion/jolt-core/src/zkvm/recursion/stage3/virtualization.rs
for constraint_idx in 0..self.params.num_constraints {
    for poly_idx in 0..NUM_POLY_TYPES {
        let claim_idx = virtual_claim_index(constraint_idx, poly_idx);
        let s_idx =
            matrix_s_index(poly_idx, constraint_idx, self.params.num_constraints_padded);
        result += eq_evals[s_idx] * self.virtual_claims[claim_idx];
    }
}
```

### 1.2 Stage 2 today: why it emits per-instance claims

The generic Stage-2 constraint sumcheck helper (`constraint_list_sumcheck.rs`) is explicitly
instance-major in its `cache_openings`:

```354:379:/Users/quang.dao/Documents/SNARKs/jolt-recursion/jolt-core/src/subprotocols/constraint_list_sumcheck.rs
// Order is instance-major, then per-instance opening order.
for i in 0..self.spec.num_instances() {
    let global_idx = self.constraint_indices[i];
    for spec in opening_specs {
        let claim = polys_by_kind[spec.kind][i].get_bound_coeff(0);
        let poly_id = self.spec.build_virtual_poly(spec.term_index, global_idx);
        accumulator.append_virtual(transcript, poly_id, sumcheck_id, opening_point.clone(), claim);
    }
}
```

And the verifier correspondingly fetches *every instance*’s opened claims to evaluate a generally
nonlinear constraint at the final point:

```478:499:/Users/quang.dao/Documents/SNARKs/jolt-recursion/jolt-core/src/subprotocols/constraint_list_sumcheck.rs
for i in 0..self.spec.num_instances() {
    let global_idx = self.constraint_indices[i];
    let mut claims = Vec::with_capacity(opening_specs.len());
    for spec in opening_specs {
        let poly_id = self.spec.build_virtual_poly(spec.term_index, global_idx);
        let (_, claim) = accumulator.get_virtual_polynomial_opening(poly_id, sumcheck_id);
        claims.push(claim);
    }
    let constraint_value = self.spec.eval_constraint_at_point(i, &claims, &shared_scalars, &eval_point, self.term_batch_coeff);
    total += batch_power * constraint_value;
}
```

This is the Stage-2 “blocker”: **because constraints are nonlinear, the verifier can’t in general
reconstruct \(\sum_i \gamma^i C_i(r)\) from only a few aggregated openings** without changing the
polynomial structure being proved.

---

## 2) Target design (what we want)

### 2.1 High-level goals

- **Stage 2**:
  - Do not append per-instance/per-row virtual claims.
  - Instead, define **fused polynomials** that include the instance index as additional MLE variables.
  - Run sumchecks against these fused polynomials and cache **one opening per fused polynomial**.

- **Stage 3**:
  - Do not consume a length `num_constraints * PolyType::NUM_TYPES` claim vector.
  - Instead, consume **one scalar per fused `PolyType`** (or fewer if we exploit sparsity).

- **Protocol-wide**:
  - Preserve deterministic transcript ordering (consensus-critical).
  - Keep endianness conventions consistent with `OpeningPoint<BIG_ENDIAN, _>` used throughout
    `opening_proof.rs`.

### 2.2 Key mathematical trick: factor the row index

Because:

- `num_constraints_padded` is a power of 2
- `PolyType::NUM_TYPES` is 64 (power of 2)
- `row = poly_type * num_constraints_padded + constraint_idx`

We can write:

- Let \(N = \text{num_constraints_padded} = 2^k\)
- Let \(T = \text{PolyType::NUM_TYPES} = 64 = 2^\ell\) with \(\ell = 6\)

Then:
\[
  row = (poly\_type \ll k)\;|\;constraint\_idx
\]
and the row-selector variable vector \(s \in \mathbb{F}^{k+\ell}\) can be decomposed as:

- \(r_s = (r_c, r_t)\)
  - `r_c`: the first \(k\) challenges correspond to `constraint_idx` bits (LSB-first)
  - `r_t`: the next \(\ell\) challenges correspond to `poly_type` bits

This factorization is what allows Stage 3 to reduce the “sum over all rows” into a “sum over
poly-types of fused-in-constraint-index values.”

---

## 3) Proposed final protocol (end-to-end)

This section defines the “final form” we are porting toward.

### 3.1 Invariants we rely on (and should assert)

- **Row index factorization**: `row = poly_type * num_constraints_padded + constraint_idx`
  (`constraints_sys.rs` `row_index`, see above).
- **Power-of-two row count**: because `PolyType::NUM_TYPES` and `num_constraints_padded` are powers
  of two, `num_rows_unpadded = PolyType::NUM_TYPES * num_constraints_padded` is also a power of two,
  so the matrix is not “extra padded” in the row dimension:

```1305:1311:/Users/quang.dao/Documents/SNARKs/jolt-recursion/jolt-core/src/zkvm/recursion/constraints_sys.rs
let num_rows_unpadded = PolyType::NUM_TYPES * num_constraints_padded;
let num_s_vars = (num_rows_unpadded as f64).log2().ceil() as usize;
let num_rows = 1 << num_s_vars;
```

- **Zero-row convention**: For each global constraint index, the builder pushes a real row only for
  the relevant `PolyType`s and pushes all-zero rows for all other `PolyType`s:

```408:420:/Users/quang.dao/Documents/SNARKs/jolt-recursion/jolt-core/src/zkvm/recursion/constraints_sys.rs
fn push_zero_rows_except(&mut self, row_size: usize, except: &[PolyType]) {
    let zero_row = vec![Fq::zero(); row_size];
    for poly_type in PolyType::all() {
        if except.contains(&poly_type) {
            continue;
        }
        self.rows_by_type[poly_type as usize].push(zero_row.clone());
    }
}
```

This convention is what makes “type gating” optional for some families (but still recommended for
clarity and safety).

### 3.2 Fused “row-block” polynomials \(P_t(c,x)\)

For each `PolyType` value \(t\), define a fused polynomial:

\[
P_t(c, x) := M((c,t), x)
\]

Interpretation:

- Fix `t`: pick a poly-type row block
- Vary `c`: pick the constraint index within that block
- Vary `x`: evaluate within-row (11-variable) coordinates

**Important implementation choice**:

- Represent each `P_t` with variable order **`[x vars (11), c vars (k)]`**, low-to-high.
  This matches the natural “row-major” flattening of the matrix blocks and preserves current
  `BindingOrder::LowToHigh` conventions.

### 3.3 Stage 2: fused constraint-check sumcheck (single opening point)

Stage 2 becomes (conceptually) a single sumcheck over variables \((x,c)\) of total length \(11 + k\)
that proves a unified constraint polynomial \(C(c,x)\) is zero (as a low-degree polynomial).

#### 3.3.1 Constraint polynomial \(C(c,x)\)

We define:

\[
C(c,x) = \sum_{\text{family}} I_{\text{family}}(c)\cdot C_{\text{family}}(c,x)
\]

- \(I_{\text{family}}(c)\) is a type-indicator MLE over `c` that is 1 on constraints of that family
  and 0 otherwise. (We can treat these as public; see Section 6.4.)
- Each \(C_{\text{family}}(c,x)\) is written in terms of the relevant `PolyType` row polynomials
  \(P_t(c,x)\) and any family-specific public inputs.

Example: GT-mul family uses `MulLhs`, `MulRhs`, `MulResult`, `MulQuotient` and `g(x)`.

#### 3.3.2 Sumcheck statement

Use the same “eq-weighted” sumcheck pattern used throughout the repo:

\[
0 = \sum_{(c,x)\in\{0,1\}^{k+11}} Eq(z^\*, (c,x))\cdot C(c,x)
\]

where:

- \(z^\*\in\mathbb{F}^{k+11}\) is sampled from the transcript (like `eq_point` today).

The sumcheck yields a random evaluation point \(r=(r_x,r_c)\) (the sumcheck challenges), and the
verifier’s final check is equivalent to checking a low-degree identity at that random point,
computed from the opened fused polynomials.

#### 3.3.3 What Stage 2 caches (the “few claims” output)

Stage 2 caches:

- For each `PolyType` \(t\) needed downstream:
  - the single scalar opening claim \(p_t := P_t(r_c,r_x)\)

**This is the “one claim per fused poly” contract.**

Stage 2 may also cache a small constant number of auxiliary fused polynomials if we decide to
represent some public inputs as “opened aux polynomials” (e.g., base-point coordinates). This is
still O(1), not O(#constraints).

### 3.4 Stage 3: fused virtualization (direct evaluation)

Stage 3 changes from consuming a length `num_constraints * PolyType::NUM_TYPES` vector to consuming
the 64 fused values \(\{p_t\}\).

#### 3.4.1 Challenges

- Stage 2 already produced:
  - `r_x` (11 challenges)
  - `r_c` (k challenges)
- Stage 3 samples `r_t` (6 challenges) and sets:
  - `r_s = (r_c, r_t)` in low-to-high order (constraint bits first, then poly-type bits)

#### 3.4.2 Prover computation (unchanged core)

Prover evaluates `M(r_s, r_x)` by binding `x` then evaluating on `s`, as today.

#### 3.4.3 Verifier computation (new)

Let \(Eq_t := EqPolynomial::evals(r_t)\), an array of length 64.

Compute:

\[
M_{\text{expected}} = \sum_{t=0}^{63} Eq_t[t]\cdot p_t
\]

and check it equals the prover’s `M(r_s, r_x)`.

This replaces the current double loop over `(constraint_idx, poly_idx)` in Stage 3
(`stage3/virtualization.rs`).

### 3.5 Stages 4 and 5 (jagged) and PCS opening

Stages 4/5 operate only on:

- the single scalar `M(r_s,r_x)` (as `VirtualPolynomial::DorySparseConstraintMatrix`)
- the jagged transform proof state
- the final PCS opening of `CommittedPolynomial::DoryDenseMatrix`

So Stage 4/5 should remain structurally unchanged; only the way Stage 3 produces/justifies
`M(r_s,r_x)` changes.

---

## 4) Concrete first target: GT-mul family (fully specified)

This section gives a concrete “first implementable family” under the fused framework.

### 4.1 PolyTypes involved

From `constraints_sys.rs`:

```124:129:/Users/quang.dao/Documents/SNARKs/jolt-recursion/jolt-core/src/zkvm/recursion/constraints_sys.rs
MulLhs = 2,
MulRhs = 3,
MulResult = 4,
MulQuotient = 5,
```

So the only required fused row polynomials are:

- `P_MulLhs(c,x)`
- `P_MulRhs(c,x)`
- `P_MulResult(c,x)`
- `P_MulQuotient(c,x)`

### 4.2 Constraint

Reuse the existing GT-mul constraint expression:

```174:180:/Users/quang.dao/Documents/SNARKs/jolt-recursion/jolt-core/src/zkvm/recursion/stage2/gt_mul.rs
lhs * rhs - result - quotient * g
```

Define:

\[
C_{\text{gtmul}}(c,x) :=
P_{\text{MulLhs}}(c,x)\cdot P_{\text{MulRhs}}(c,x)
- P_{\text{MulResult}}(c,x)
- P_{\text{MulQuotient}}(c,x)\cdot g(x)
\]

Optionally include `I_gtmul(c)` as a factor (recommended once we unify families).

### 4.3 Stage-2 openings needed

At \(r=(r_c,r_x)\), Stage 2 needs only:

- `p_MulLhs = P_MulLhs(r_c,r_x)`
- `p_MulRhs = P_MulRhs(r_c,r_x)`
- `p_MulResult = P_MulResult(r_c,r_x)`
- `p_MulQuotient = P_MulQuotient(r_c,r_x)`

That’s 4 scalar claims total for GT-mul correctness (plus whatever other families require).

---

## 5) How we build the fused polynomials efficiently (avoid 64× copying)

### 5.1 Observed physical layout of the matrix

The matrix is stored row-major, and rows are grouped by `PolyType`:

```307:339:/Users/quang.dao/Documents/SNARKs/jolt-recursion/jolt-core/src/zkvm/recursion/constraints_sys.rs
/// Giant multilinear matrix M(s, x) that stores all Dory polynomials in a single structure.
///
/// Layout: M(s, x) where s is the row index and x are the constraint variables
/// Physical layout: rows are organized as [all base] [all rho_prev] [all rho_curr] [all quotient]
/// Row index = poly_type * num_constraints_padded + constraint_index
pub struct DoryMultilinearMatrix {
    pub num_s_vars: usize,
    pub num_constraint_vars: usize,
    pub num_constraints_padded: usize,
    pub evaluations: Vec<Fq>,
}
```

And the builder copies rows in `PolyType::all()` order, then pads within each block:

```1339:1350:/Users/quang.dao/Documents/SNARKs/jolt-recursion/jolt-core/src/zkvm/recursion/constraints_sys.rs
for poly_type in PolyType::all() {
    let rows = &self.rows_by_type[poly_type as usize];
    // Copy actual rows
    for row in rows {
        std::ptr::copy_nonoverlapping(row.as_ptr(), eval_ptr.add(offset), row_size);
        offset += row_size;
    }
    // Skip zero padding (already initialized to zero)
    offset += (num_constraints_padded - rows.len()) * row_size;
}
```

### 5.2 Slice-to-`P_t(c,x)` mapping

Let:

- `row_size = 1 << num_constraint_vars` (currently 2048)
- `block_size = num_constraints_padded * row_size`
- `t = poly_type as usize`

Then the evaluation table for `P_t(c,x)` is exactly:

- `matrix.evaluations[t*block_size .. (t+1)*block_size]`

Within that block:

- The low `num_constraint_vars` bits index `x`
- The high `k` bits index `c`

So `P_t` has variable order `[x (low), c (high)]` as desired.

### 5.3 Practical representation choices

We have three viable implementation options:

- **Option A (quick prototype)**: copy each block into a `DensePolynomial` and wrap it as
  `MultilinearPolynomial::LargeScalars`. Easy, but duplicates memory.
- **Option B (recommended)**: introduce a “view”/slice-backed multilinear polynomial variant that
  references `Arc<[Fq]>` plus `(offset, len)` without copying.
- **Option C**: avoid constructing `P_t` as a polynomial object at all; implement the fused Stage-2
  prover `compute_message` by indexing into the matrix evaluation table directly.

For a first working port, do Option A, then upgrade to Option B for performance.

---

## 6) Public inputs and type gating (the part that makes nonlinear families safe)

Some families (notably scalar multiplication) use per-instance public parameters that are not stored
in the matrix rows (e.g., base points), so we must ensure constraints evaluate to 0 on constraint
indices that are not in that family.

### 6.1 Scalar-mul “bit” polynomials already exist as `PolyType`s

The matrix already includes `G1ScalarMulBit` and `G2ScalarMulBit`:

```130:157:/Users/quang.dao/Documents/SNARKs/jolt-recursion/jolt-core/src/zkvm/recursion/constraints_sys.rs
G1ScalarMulBit = 14,
// ...
G2ScalarMulBit = 29,
```

So in the fused design we should **prefer using these as witness polynomials** instead of
recomputing bit MLEs from scalars inside the verifier loop (which would otherwise be expensive).

Reference: current scalar-mul verifier computes the bit MLE from the scalar:

```60:72:/Users/quang.dao/Documents/SNARKs/jolt-recursion/jolt-core/src/zkvm/recursion/stage2/g1_scalar_mul.rs
pub fn evaluate_bit_mle<F: JoltField>(&self, eval_point: &[F]) -> F {
    let pad_factor = EqPolynomial::<F>::zero_selector(&eval_point[..3]);
    let eq_step = EqPolynomial::<F>::evals(&eval_point[3..]);
    // ...
    pad_factor * acc
}
```

In the fused plan, we can instead rely on the matrix row values for `PolyType::G1ScalarMulBit`
/ `PolyType::G2ScalarMulBit` and open them once per fused poly type.

### 6.2 Base points and other per-instance parameters

Base points (`x_p, y_p`) are stored as Rust-side vectors today and accessed by instance index:

```339:343:/Users/quang.dao/Documents/SNARKs/jolt-recursion/jolt-core/src/zkvm/recursion/stage2/g1_scalar_mul.rs
let (x_p, y_p) = self.base_points[instance];
```

In the fused design we cannot index by boolean `instance`; we need a polynomially defined value at
non-boolean `r_c`.

We have two options:

- **Option 1 (public MLE over c)**:
  - Build an evaluation table `Bx[c] = x_p(c)` of length `num_constraints_padded`
  - Evaluate its MLE at `r_c` during verification in \(O(num_constraints_padded)\)
  - Same for `By`
- **Option 2 (aux fused polynomials opened once)**:
  - Define aux fused polynomials `Bx(c,x)` and `By(c,x)` that are constant in `x`
  - Have the prover append their openings at `(r_c,r_x)` (2 scalar claims)
  - Verifier uses these scalar openings directly

Option 2 keeps verifier time small and is simplest initially. It costs only O(1) extra claims.

### 6.3 Type indicator polynomials

For safety and clarity (and required for families with external public inputs), define indicator
polynomials:

- `I_gtmul(c)`, `I_g1_scalar_mul(c)`, `I_g2_scalar_mul(c)`, `I_g1_add(c)`, `I_g2_add(c)`,
  `I_packed_gt_exp(c)`, ...

Same choice as base points:

- compute as public MLE over c, or
- open as aux scalars at `r_c`

### 6.4 Recommended initial policy

To get a working system fast:

- Treat `PolyType` rows (64 of them) as opened fused polynomials (one scalar each).
- Treat base points as opened aux scalars (Option 2).
- Treat type indicators as opened aux scalars (Option 2) *or* compute them publicly if counts are small.

Then optimize in later passes.

---

## 7) What changes in code (high-level diffs)

### 7.1 Add fused IDs

- Add a new `VirtualPolynomial` representation for “fused poly-type opening claim”:
  - e.g. `VirtualPolynomial::Recursion(RecursionPoly::FusedRow { poly_type })`
  - Provide constructors like:
    - `VirtualPolynomial::recursion_fused_row(PolyType::MulLhs)` etc.

### 7.2 Stage 2: new fused sumcheck module

- New prover/verifier module (suggested):
  - `jolt-core/src/zkvm/recursion/stage2/fused_stage2.rs`
- Responsibilities:
  - Build accessors for each `P_t(c,x)` (matrix slice view)
  - Define and evaluate `C(c,x)` per family with gating
  - Implement `SumcheckInstanceProver` / `SumcheckInstanceVerifier`
  - In `cache_openings`, append fused openings:
    - for each `PolyType::all()` (and any aux) append `P_t(r_c,r_x)`

### 7.3 Stage 3: consume fused openings

- Update `stage3/virtualization.rs`:
  - remove `extract_virtual_claims_from_accumulator`
  - add `get_fused_polytype_claims_from_accumulator`
  - compute expected `M(r_s,r_x)` as `Σ_t Eq(r_t,t) * p_t`

### 7.4 Recursion orchestrators

- `recursion_prover.rs` / `recursion_verifier.rs`:
  - Stage 2 now returns `(r_x, r_c)` (or the full point `r`)
  - Stage 3 uses `r_c` from Stage 2 and samples `r_t` itself

---

## 8) Test plan

### 8.1 New unit tests

- Row factorization:
  - `row_index(t,c) == (t << k) | c`
  - bit decomposition sanity (LSB-first).

- Stage 3 fused identity:
  - Verify: `M((r_c,r_t),r_x) == Σ_t Eq(r_t,t)*P_t(r_c,r_x)`

### 8.2 Regression tests

- Run existing recursion E2E tests after each phase.

---

## 9) Phased porting milestones (recommended execution order)

1. **Introduce fused `VirtualPolynomial` IDs + ordering**
2. **Rewrite Stage 3 to use fused per-PolyType claims**
3. **Add a temporary “fusion step” that computes fused claims from the matrix**
4. **Replace Stage 2 constraint sumchecks with the fused Stage 2**
5. **Fold shift checks (optional)**
6. **Tackle packed GT exp last (either keep as Stage 1 initially, or fully unify)**

---

## 10) “Definition of done”

We consider the port complete when:

- Stage 2 and Stage 3 no longer append/consume per-(constraint, poly-type) claims.
- Proof size contribution from row claims is **O(PolyType::NUM_TYPES)** (plus O(1) aux).
- `cargo test -p jolt-core test_recursion_snark_e2e_with_dory -- --nocapture` passes.


